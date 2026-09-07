// Copyright (c) 2026, WSO2 LLC. (http://www.wso2.com).
//
// WSO2 LLC. licenses this file to you under the Apache License,
// Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

import ballerina/ai;
import ballerina/http;
import ballerina/log;
import ballerinax/aws;

# The maximum number of `_id`s looked up in a single `terms` query, per OpenSearch's cap on the
# number of terms in a `terms` clause.
const int DOC_ID_LOOKUP_CHUNK_SIZE = 65536;

# How far the hybrid fusion weights may sum from 1.0 and still be accepted. Binary floating point
# does not sum 0.7 and 0.3 to exactly 1.0, and rejecting that split would be absurd.
const float WEIGHT_SUM_TOLERANCE = 1.0e-6;

# The `SearchMode` a caller who names none gets: dense search over a 1536-dimensional vector, the
# width of the most common general-purpose embedding models.
#
# Named rather than written inline as `init`'s default, because a bare record literal is ambiguous
# against a union whose variants share field names — `queryMode` discriminates the type but does
# not, on its own, resolve the literal.
final DenseSearch & readonly DEFAULT_SEARCH_MODE = {queryMode: ai:DENSE, indexConfig: {dimension: 1536}};

# An `ai:VectorStore` backed by Amazon OpenSearch — either a managed OpenSearch Service domain, or
# an OpenSearch Serverless vector search collection (Classic or NextGen).
#
# There is no `ballerinax/opensearch` connector to wrap, so this class talks the OpenSearch REST
# API directly over `ballerina/http`, signing every request with AWS Signature Version 4 via
# `ballerinax/aws.auth` (or, on a managed domain with fine-grained access control, HTTP basic
# auth against the internal user database).
#
# # Deployment-specific behavior
# `add` is an upsert on `MANAGED_DOMAIN` and `SERVERLESS_NEXTGEN` (both support a caller-supplied
# `_id`), but is **append-only** on `SERVERLESS_CLASSIC` — re-adding the same logical id creates a
# duplicate document there, since Serverless Classic vector collections reject a custom `_id` on
# write. `delete` is a single `_bulk` call on the first two, and a search-then-bulk-delete on
# `SERVERLESS_CLASSIC`; on all types, deleting a non-existent id is treated as success (this
# module ignores `not_found`, unlike `ai:InMemoryVectorStore`, which errors). None of this varies
# by `SearchMode` — `delete` never touches a vector field.
#
# Write visibility is immediate on a managed domain when `ManagedDomainDeployment.refreshOnWrite`
# is set, otherwise governed by the index's normal refresh interval; on Serverless it is a fixed,
# unforceable ~10 seconds (NextGen) or ~60 seconds (Classic) — poll rather than assume immediacy.
#
# # Construction is not query readiness on Serverless Classic
# When `init` creates the index on `SERVERLESS_CLASSIC`, a successful construction does not mean
# the index can be searched yet: `query` can fail with `index_not_found_exception` for roughly ten
# seconds afterwards, while `_mapping` and `_settings` already report the index as present. It is
# a propagation delay that clears on its own with no write and no intervention, and `add` is
# unaffected — a caller that writes before it queries never sees it. Code that queries straight
# after constructing a Classic store should retry that error for a few seconds rather than treat
# it as fatal. This is not waited out inside `init`, which would otherwise pay the delay on every
# Classic construction; see `ensureIndex`.
#
# # Search modes
# The `SearchMode` chosen at construction decides the index mapping, which `ai:Embedding` member
# `add` and `query` accept, and how a query is scored. The match is strict in both directions: a
# store built for one mode and handed another's embedding fails rather than indexing a field no
# query reads.
#
# `DENSE` maps a `knn_vector` field and queries it with a `knn` clause. `SPARSE` maps a
# `rank_features` field and queries it with `neural_sparse` carrying precomputed `query_tokens`.
# `HYBRID` maps both and issues them as one `hybrid` query, with the two scores normalized and
# combined server-side.
#
# `SPARSE` and `HYBRID` need the `neural-search` plugin, which ships in the standard OpenSearch
# distribution and is present on AWS managed domains from 2.9; raw `query_tokens` additionally
# need OpenSearch 2.14 or later, and the `hybrid` query needs 2.11. AWS documents `neural_sparse`
# and `hybrid` for Serverless without distinguishing collection generation, so both are permitted
# on every `Deployment` here — but neither has been verified against a real Serverless collection,
# and a Classic vector search collection may yet reject a `rank_features` mapping.
#
# # Similarity score range
# What `VectorMatch.similarityScore` means depends on the mode, and the ranges do not line up:
#
# Under `DENSE`, it is the raw OpenSearch `_score`, whose range is space-dependent — `l2` gives
# `(0, 1]`, `innerproduct` is piecewise, and `cosinesimil` gives `[0, 1]` rather than the
# `[-1, 1]` a caller may expect from "cosine similarity". `DenseSearch.normalizeCosineScore`
# (default `true`, and honored only under `COSINE`) converts it back to `[-1, 1]` via
# `cos = 2 * score - 1`.
#
# Under `SPARSE`, it is the dot product of the query weights against the stored ones. It is
# **unbounded above** and is not a similarity in `[0, 1]` at all — a threshold tuned against a
# dense store means nothing here. The cosine transform is never applied, and cannot be: it is not
# a field `SparseSearch` declares.
#
# Under `HYBRID`, it is the fused score, which `min_max` + `arithmetic_mean` puts in `(0.0, 1.0]`.
# It has already been normalized by the pipeline, so no further transform applies.
#
# In every mode, a query with no embedding (a metadata-only filter, or neither embedding nor
# filters) gets OpenSearch's constant score, which is not a similarity, so `similarityScore` is
# always `0.0` for those matches.
#
# # Returned embeddings
# `VectorMatch.embedding` is whatever storage holds, read straight back out of `_source`, in the
# member this store's mode uses — an `ai:Vector`, an `ai:SparseVector`, or an `ai:HybridVector`.
# It is not guaranteed to be what was passed to `add`, in any mode:
#
# A stored `rank_features` weight keeps roughly nine significant bits, so a round-tripped sparse
# weight carries about 0.4% relative error. A sparse vector also comes back sorted by index
# ascending, whatever order it was written in. On `SERVERLESS_NEXTGEN` under `COSINE`, storage
# holds a unit-normalized copy of the dense vector, so its original magnitude is unrecoverable —
# AWS discards it at write time. A `HYBRID` round trip is therefore inexact on both halves at
# once. Never compare a returned embedding to a locally-held one with exact float equality.
#
# # Note
# The `close` method releases the underlying AWS credential provider's background refresh
# threads and STS/SSO HTTP connections. It is not part of the `ai:VectorStore` contract (which
# has no lifecycle hook), so nothing calls it automatically — call it explicitly when the store
# is no longer needed, or simply let it live for the application's lifetime.
public isolated class VectorStore {
    *ai:VectorStore;

    private final OpenSearchTransport transport;
    private final string indexName;
    private final Deployment & readonly deployment;
    private final SearchMode & readonly searchMode;
    private final Configuration & readonly config;

    # Initializes the AWS OpenSearch vector store.
    #
    # + serviceUrl - The OpenSearch endpoint URL, e.g.
    # `https://my-domain.us-east-1.es.amazonaws.com` for a managed domain,
    # `https://<collection-id>.<region>.aoss.amazonaws.com` for a Serverless Classic
    # per-collection endpoint, or `https://<collection-id>.aoss.<region>.on.aws` /
    # `https://<account-id>.aoss.<region>.on.aws` for a Serverless NextGen endpoint
    # + region - The AWS region the endpoint is in
    # + indexName - The OpenSearch index this store reads and writes
    # + deploymentConfig - The target deployment and its flavour-specific settings — credentials,
    # which are required rather than defaulted, and whichever of
    # `engine`/`refreshOnWrite`/`collectionName`/`collectionId`/`compressionLevel`/`vectorMode`
    # that flavour actually honors. Settings shared by all three live on `storeConfig` instead
    # + searchMode - The kind of search this store performs — `DenseSearch`, `SparseSearch` or
    # `HybridSearch` — together with the index shape, field names and fusion settings that only
    # that kind honors
    # + storeConfig - Behavioral configuration honored under every search mode and on every
    # deployment type
    # + httpConfig - Underlying HTTP client configuration. `httpVersion` and
    # `http1Settings.chunking` are pinned by this module to `HTTP_1_1` and `CHUNKING_NEVER`
    # respectively and cannot be overridden: Ballerina's HTTP/2 outbound path corrupts a signed
    # JSON body, and a chunked body carries no `Content-Length` for SigV4 to be verified against.
    # Every other field is passed through unchanged
    # + return - An `ai:Error` if construction, validation, or (when
    # `Configuration.createIndexIfNotExists` is `true`) index creation fails; otherwise `()`
    public isolated function init(
            @display {label: "Service URL"} string serviceUrl,
            @display {label: "Region"} aws:Region|string region,
            @display {label: "Index Name"} string indexName,
            @display {label: "Deployment Configuration"} Deployment deploymentConfig,
            @display {label: "Search Mode"} SearchMode searchMode = DEFAULT_SEARCH_MODE,
            @display {label: "Store Configuration"} Configuration storeConfig = {},
            @display {label: "HTTP Configuration"} http:ClientConfiguration httpConfig = {})
            returns ai:Error? {
        check validateConfiguration(serviceUrl, deploymentConfig, searchMode, storeConfig);

        OpenSearchTransport transport = check new (serviceUrl, region, deploymentConfig,
            storeConfig.retryConfig, httpConfig
        );
        ai:Error? ensureResult = ensureIndex(transport, indexName, searchMode, storeConfig, deploymentConfig);
        if ensureResult is ai:Error {
            // Best-effort cleanup; the closing outcome is intentionally not surfaced so it
            // cannot mask the more relevant `ensureResult` failure below.
            ai:Error? closeResult = transport.close();
            if closeResult is ai:Error {
                log:printWarn("Failed to close the AWS credential provider after a failed index-ensure step",
                        'error = closeResult);
            }
            return ensureResult;
        }

        self.transport = transport;
        self.indexName = indexName;
        self.deployment = deploymentConfig.cloneReadOnly();
        self.searchMode = searchMode.cloneReadOnly();
        self.config = storeConfig.cloneReadOnly();
    }

    # Adds vector entries to the store. Entries without an `id` receive a generated UUID.
    # Chunked into `_bulk` requests of at most `Configuration.maxBulkSize` entries.
    #
    # + entries - The vector entries to add. Every embedding must be the `ai:Embedding` member
    # this store's `SearchMode` requires — an `ai:Vector` under `DENSE`, an `ai:SparseVector`
    # under `SPARSE`, an `ai:HybridVector` under `HYBRID` — or the call fails naming the entry.
    # A sparse weight outside Lucene's accepted range fails the same way, before anything is sent
    # + return - An `ai:Error` naming the failing entries if any `_bulk` item fails (OpenSearch
    # returns HTTP 200 even on partial failure — this is always checked explicitly), otherwise `()`
    public isolated function add(ai:VectorEntry[] entries) returns ai:Error? {
        if entries.length() == 0 {
            return;
        }
        PreparedEntry[] prepared = check prepareEntries(entries, self.searchMode);

        foreach PreparedEntry[] batch in chunkPreparedEntries(prepared, self.config.maxBulkSize) {
            byte[] body = check buildAddBulkBody(batch, self.indexName, self.deployment, self.searchMode,
                    self.config);
            BulkResponse response = check self.transport.bulk(body, isIndexingBatch = true);
            // Attributed by position against this batch's ids, so a failure names the caller's
            // entry even on `SERVERLESS_CLASSIC`, where the response `_id` is server-generated.
            string[] submittedIds = from PreparedEntry entry in batch
                select entry.id;
            BulkFailure[] failures = extractIndexFailures(response, submittedIds);
            if failures.length() > 0 {
                return error(string `Failed to index ${failures.length()} of ${batch.length()} ` +
                        string `entries: ${summarizeBulkFailures(failures)}`);
            }
        }
    }

    # Searches the store for vectors similar to a query embedding, filtered by metadata, or both.
    # All four combinations of `(embedding, filters)` are supported; when neither is given, every
    # entry is returned (subject to `topK`) with `similarityScore: 0.0`.
    #
    # + query - The vector store query. Any `topK < 1` means "return all", capped at
    # `Configuration.maxResultWindow`; an explicit `topK` above that ceiling is rejected. The
    # embedding, when given, must be the member this store's `SearchMode` requires, matching
    # `add`; under `SPARSE`/`HYBRID` it must also carry no more than `maxQueryTokens` terms, and
    # every sparse weight must be in `(0, 64]`
    # + return - The matching entries, or an `ai:Error` on failure
    public isolated function query(ai:VectorStoreQuery query) returns ai:VectorMatch[]|ai:Error {
        json body = check buildSearchBody(query, self.searchMode, self.config);
        SearchResponse response = check self.transport.search(self.indexName, body,
                searchPipelineName(self.searchMode));
        // buildSearchBody already rejected an embedding that does not match this store's
        // `SearchMode`, so any embedding still present here produced a scoring clause.
        boolean scoreIsMeaningful = query.embedding !is ();
        ai:VectorMatch[] matches = [];
        foreach SearchHit hit in response.hits.hits {
            matches.push(check hitToVectorMatch(hit, self.searchMode, self.config, scoreIsMeaningful));
        }
        return matches;
    }

    # Deletes vector entries by their logical id. Deleting an id that does not exist is treated as
    # success, matching the Pinecone/pgvector/weaviate precedent (`ai:InMemoryVectorStore` errors
    # on a missing id; this module does not).
    #
    # On `SERVERLESS_CLASSIC`, this is a two-phase operation — a `terms` search on
    # `Configuration.idFieldName` to discover the internal `_id`s, then a bulk delete of those —
    # because a custom `_id` cannot be written there and `_delete_by_query` is not on the AOSS
    # whitelist. A document added less than the collection's refresh interval ago will not be
    # found by the search phase and will survive the delete; this is unavoidable and not a bug.
    #
    # + ids - The logical id, or ids, to delete
    # + return - An `ai:Error` naming the failing ids if any (non-`not_found`) `_bulk` item fails,
    # otherwise `()`
    public isolated function delete(string|string[] ids) returns ai:Error? {
        string[] idList;
        if ids is string {
            idList = [ids];
        } else {
            idList = ids;
        }
        if idList.length() == 0 {
            return;
        }
        if self.deployment is ServerlessClassicDeployment {
            return self.deleteOnServerlessClassic(idList);
        }
        byte[] body = buildDeleteBulkBody(self.indexName, idList);
        BulkResponse response = check self.transport.bulk(body);
        BulkFailure[] failures = extractDeleteFailures(response, idList);
        if failures.length() > 0 {
            return error(string `Failed to delete ${failures.length()} of ${idList.length()} ` +
                    string `entries: ${summarizeBulkFailures(failures)}`);
        }
    }

    # Releases the underlying AWS credential provider's background refresh threads and STS/SSO
    # HTTP connections. Not part of the `ai:VectorStore` contract; call explicitly when this store
    # is no longer needed, or let it live for the application's lifetime.
    #
    # + return - An `ai:Error` if releasing the resources fails, otherwise `()`
    public isolated function close() returns ai:Error? {
        return self.transport.close();
    }

    # Implements the search-then-bulk-delete path required on `SERVERLESS_CLASSIC`.
    #
    # Because `add` is append-only there, a heavily re-added logical id can accumulate more
    # duplicate documents than a single lookup's `size` (`Configuration.maxResultWindow`) can see.
    # Rather than silently deleting only the first page and leaving the rest behind, a lookup that
    # comes back exactly full is treated as "possibly incomplete" and fails loudly.
    #
    # + idList - The logical ids to delete
    # + return - An `ai:Error` naming the failing internal ids if any (non-`not_found`) `_bulk`
    # item fails, or if a lookup could not confirm it saw every duplicate, otherwise `()`
    private isolated function deleteOnServerlessClassic(string[] idList) returns ai:Error? {
        string[] internalIds = [];
        // Grown in lockstep with `internalIds`: the bulk delete below is submitted against
        // internal `_id`s, and this is what lets a per-item failure be reported against the
        // logical id the caller actually asked to delete.
        string[] logicalIds = [];
        foreach string[] chunk in chunkIds(idList, DOC_ID_LOOKUP_CHUNK_SIZE) {
            json searchBody = buildDocIdLookupBody(self.config.idFieldName, chunk, self.config.maxResultWindow);
            SearchResponse searchResponse = check self.transport.search(self.indexName, searchBody);
            int hitCount = searchResponse.hits.hits.length();
            if hitCount >= self.config.maxResultWindow {
                return error(string `A lookup for ${chunk.length()} id(s) to delete on SERVERLESS_CLASSIC ` +
                        string `returned ${hitCount} documents, meeting the configured 'maxResultWindow' ` +
                        string `(${self.config.maxResultWindow}) exactly. Because 'add' is append-only on ` +
                        "Serverless Classic, there may be more duplicate documents than this lookup could " +
                        "see, which would be left undeleted. Raise 'Configuration.maxResultWindow' and retry");
            }
            foreach SearchHit hit in searchResponse.hits.hits {
                internalIds.push(hit._id);
                map<json>? hitSource = hit?._source;
                json logicalId = hitSource is map<json> ? hitSource[self.config.idFieldName] : ();
                // A document written outside this module may carry no logical id at all; the
                // internal `_id` is then the only handle there is.
                logicalIds.push(logicalId is string ? logicalId : hit._id);
            }
        }
        if internalIds.length() == 0 {
            return;
        }
        byte[] body = buildDeleteBulkBody(self.indexName, internalIds);
        BulkResponse response = check self.transport.bulk(body);
        BulkFailure[] failures = extractDeleteFailures(response, logicalIds);
        if failures.length() > 0 {
            return error(string `Failed to delete ${failures.length()} of ${internalIds.length()} ` +
                    string `entries: ${summarizeBulkFailures(failures)}`);
        }
    }
}

# Runs the fail-fast construction validations before any network I/O is attempted.
#
# Most of what this function once checked is now unrepresentable rather than rejected: `BasicAuth`
# on a Serverless collection, a non-Faiss engine on Serverless Classic, `refreshOnWrite` off a
# managed domain, and quantization off NextGen are all type errors since those fields moved onto
# the `Deployment` variant that honors them. What is left is the range and format checking that no
# type can express.
#
# + serviceUrl - The service URL, checked for a parseable host
# + deployment - The target deployment
# + searchMode - The requested search mode
# + config - The vector store configuration
# + return - An `ai:Error` naming the first failing rule, otherwise `()`
isolated function validateConfiguration(string serviceUrl, Deployment deployment,
        SearchMode searchMode, Configuration config) returns ai:Error? {
    if searchMode is SparseSearch && searchMode.maxQueryTokens < 1 {
        return error(string `'SparseSearch.maxQueryTokens' must be a positive integer, got: ` +
                string `${searchMode.maxQueryTokens}`);
    }
    if searchMode is HybridSearch {
        if searchMode.maxQueryTokens < 1 {
            return error(string `'HybridSearch.maxQueryTokens' must be a positive integer, got: ` +
                    string `${searchMode.maxQueryTokens}`);
        }
        check validateHybridFusion(searchMode.fusion);
    }
    IndexConfig? indexConfig = indexConfigOf(searchMode);
    if indexConfig is IndexConfig && indexConfig.dimension < 1 {
        return error(string `'IndexConfig.dimension' must be a positive integer, got: ` +
                string `${indexConfig.dimension}`);
    }
    check validateFieldNames(searchMode, config);
    string _ = check extractHost(serviceUrl);
    if config.maxBulkSize < 1 {
        return error(string `'Configuration.maxBulkSize' must be a positive integer, got: ${config.maxBulkSize}`);
    }
    if config.maxResultWindow < 1 {
        return error(string `'Configuration.maxResultWindow' must be a positive integer, got: ` +
                string `${config.maxResultWindow}`);
    }
    if deployment is ServerlessNextGenDeployment {
        // Both default to `()`, so "set" is distinguishable from "unset" -- which makes this worth
        // rejecting rather than ignoring. They shape the `knn_vector` field, and a caller who set
        // them for an index that has no such field has misunderstood something. Staying silent
        // would repeat the very failure mode `IndexConfig` already warns about.
        if searchMode is SparseSearch &&
                (deployment.compressionLevel is CompressionLevel || deployment.vectorMode is VectorMode) {
            return error("'ServerlessNextGenDeployment.compressionLevel' and '.vectorMode' shape the " +
                    "'knn_vector' field, which a 'SPARSE' index does not have; leave both unset, or " +
                    "use 'HYBRID' if the index should also hold dense vectors");
        }
        if deployment?.collectionName is string && deployment?.collectionId is string {
            return error("'ServerlessNextGenDeployment.collectionName' and '.collectionId' are " +
                    "alternatives; set at most one. Sending both collection headers leaves it to " +
                    "the AOSS proxy to decide which one identifies the target collection");
        }
        check validateQuantization(deployment);
    }
}

# Rejects a document schema whose field names collide.
#
# Every name here addresses a distinct field in the same document, so a duplicate silently
# produces a broken index rather than an error: `vectorFieldName: "content"` would map the chunk
# text as a `knn_vector` and then overwrite it with the vector on every write, leaving `query`
# returning empty content. `HYBRID` makes this materially easier to hit, since it names two vector
# fields that must also differ from each other.
#
# The chunk-type field is fixed rather than configurable, so it is included as a literal.
#
# + searchMode - The search mode, supplying whichever vector field names it declares
# + config - The vector store configuration
# + return - An `ai:Error` naming the colliding field, otherwise `()`
isolated function validateFieldNames(SearchMode searchMode, Configuration config) returns ai:Error? {
    string[] names = [config.contentFieldName, config.idFieldName, CHUNK_TYPE_FIELD];
    names.push(...vectorFieldNamesOf(searchMode));
    // A blank `metadataFieldName` is the documented way to ask for a flat schema, where metadata
    // has no field of its own to collide with. `buildEntrySource` guards individual metadata keys
    // against the reserved names separately.
    if config.metadataFieldName != "" {
        names.push(config.metadataFieldName);
    }

    string[] seen = [];
    foreach string name in names {
        if name.trim() == "" {
            return error("Document field names must not be blank; only " +
                    "'Configuration.metadataFieldName' may be \"\", which selects a flat schema");
        }
        if seen.indexOf(name) !is () {
            return error(string `The document field name '${name}' is used more than once. The ` +
                    "vector, content, id, chunk-type and metadata fields must all be distinct, or " +
                    "one silently overwrites another in every stored document");
        }
        seen.push(name);
    }
}

# Validates the fusion settings a `HYBRID` store will send, against the rules the
# `normalization-processor` enforces server-side.
#
# Checked at construction rather than left to the server because a rejected weight list fails the
# whole search, and the pipeline is sent inline on every query — so the same mistake would surface
# on every call rather than once.
#
# + fusion - The fusion configuration to check
# + return - An `ai:Error` naming the failing rule, otherwise `()`
isolated function validateHybridFusion(HybridFusion fusion) returns ai:Error? {
    if fusion is NamedSearchPipeline {
        if fusion.name.trim() == "" {
            return error("'NamedSearchPipeline.name' must not be blank; a blank " +
                    "'?search_pipeline=' is rejected by OpenSearch. Set a 'HybridSearchConfig' " +
                    "instead to have this module send the pipeline inline");
        }
        return;
    }
    float denseWeight = fusion.denseWeight;
    float sparseWeight = fusion.sparseWeight;
    if denseWeight < 0.0 || denseWeight > 1.0 || sparseWeight < 0.0 || sparseWeight > 1.0 {
        return error(string `'HybridSearchConfig.denseWeight' and '.sparseWeight' must each be ` +
                string `between 0.0 and 1.0, got: ${denseWeight} and ${sparseWeight}`);
    }
    // A tolerance rather than an equality test: a caller writing 0.7 and 0.3 has expressed a
    // valid split, and binary floating point does not sum those to exactly 1.0.
    float sum = denseWeight + sparseWeight;
    if sum - 1.0 > WEIGHT_SUM_TOLERANCE || 1.0 - sum > WEIGHT_SUM_TOLERANCE {
        return error(string `'HybridSearchConfig.denseWeight' and '.sparseWeight' must sum to 1.0, ` +
                string `got: ${denseWeight} + ${sparseWeight} = ${sum}. The ` +
                "'normalization-processor' rejects a weight list that does not sum to 1.0");
    }
}

# Validates the two quantization knobs against each other. Both are `SERVERLESS_NEXTGEN`-only by
# construction — no other `Deployment` variant declares them — so all that is left to check here is
# the one combination NextGen itself rejects.
#
# + deployment - The NextGen deployment to check
# + return - An `ai:Error` naming the failing rule, otherwise `()`
isolated function validateQuantization(ServerlessNextGenDeployment deployment) returns ai:Error? {
    if deployment.vectorMode == ON_DISK && deployment.compressionLevel == COMPRESSION_1X {
        return error("'ServerlessNextGenDeployment.compressionLevel' cannot be 'COMPRESSION_1X' when " +
                "'vectorMode' is 'ON_DISK'; the server rejects that mapping with " +
                "'Cannot specify \"x1\" compression level when using \"on_disk\" mode'. Use " +
                "'IN_MEMORY' to store vectors uncompressed");
    }
}
