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
# module ignores `not_found`, unlike `ai:InMemoryVectorStore`, which errors).
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
# # Similarity score range
# `VectorMatch.similarityScore` passes the raw OpenSearch `_score` through by default. Its range
# is space-dependent: `l2` gives `(0, 1]`, `innerproduct` is piecewise, and `cosinesimil` gives
# `[0, 1]` rather than the `[-1, 1]` a caller may expect from "cosine similarity". Set
# `Configuration.normalizeCosineScore = true` (valid only with `SimilarityMetric.COSINE`) to
# convert it back to `[-1, 1]` (`cos = 2 * score - 1`). When a query has no embedding (a
# metadata-only filter, or neither embedding nor filters), OpenSearch's constant score is not a
# similarity at all, so `similarityScore` is always `0.0` for those matches regardless of this
# setting.
#
# # Returned embeddings
# `VectorMatch.embedding` is whatever storage holds, read straight back out of `_source`. It is
# not guaranteed to be the vector that was passed to `add`. On `SERVERLESS_NEXTGEN` under
# `COSINE`, storage holds a unit-normalized copy, so a round-tripped vector comes back with
# magnitude 1 and the original magnitude is unrecoverable — AWS discards it at write time, and
# nothing client-side can restore it. Do not treat a returned embedding as a byte-exact echo of
# the input, and do not compare one to a locally-held vector with exact float equality.
#
# # Scope
# This release supports dense vectors only; `add`/`query` return an `ai:Error` for a
# `SparseVector`/`HybridVector` embedding.
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
    # + deployment - The target deployment and its flavour-specific settings — credentials, which
    # are required rather than defaulted, and whichever of
    # `engine`/`refreshOnWrite`/`compressionLevel`/`vectorMode` that flavour actually honors.
    # Settings shared by all three live on `config` instead
    # + config - Index-shape and behavioral configuration honored on every deployment type
    # + queryMode - Reserved for future sparse/hybrid support; only `ai:DENSE` is accepted in this
    # release
    # + httpConfig - Underlying HTTP client configuration. `httpVersion` and
    # `http1Settings.chunking` are pinned by this module to `HTTP_1_1` and `CHUNKING_NEVER`
    # respectively and cannot be overridden: Ballerina's HTTP/2 outbound path corrupts a signed
    # JSON body, and a chunked body carries no `Content-Length` for SigV4 to be verified against.
    # Every other field is passed through unchanged
    # + return - An `ai:Error` if construction, validation, or (when
    # `IndexConfig.createIndexIfNotExists` is `true`) index creation fails; otherwise `()`
    public isolated function init(
            @display {label: "Service URL"} string serviceUrl,
            @display {label: "Region"} aws:Region|string region,
            @display {label: "Index Name"} string indexName,
            @display {label: "Deployment"} Deployment deployment,
            @display {label: "Configuration"} Configuration config = {indexConfig: {dimension: 1536}},
            @display {label: "Query Mode"} ai:VectorStoreQueryMode queryMode = ai:DENSE,
            @display {label: "HTTP Configuration"} http:ClientConfiguration httpConfig = {})
            returns ai:Error? {
        check validateConfiguration(serviceUrl, deployment, queryMode, config);

        OpenSearchTransport transport = check new (serviceUrl, region, deployment,
            config.additionalHeaders, config.retryConfig, httpConfig
        );
        ai:Error? ensureResult = ensureIndex(transport, indexName, config, deployment);
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
        self.deployment = deployment.cloneReadOnly();
        self.config = config.cloneReadOnly();
    }

    # Adds vector entries to the store. Entries without an `id` receive a generated UUID.
    # Chunked into `_bulk` requests of at most `Configuration.maxBulkSize` entries.
    #
    # + entries - The vector entries to add. A `SparseVector`/`HybridVector` embedding on any
    # entry fails the call — this release supports dense vectors only
    # + return - An `ai:Error` naming the failing entries if any `_bulk` item fails (OpenSearch
    # returns HTTP 200 even on partial failure — this is always checked explicitly), otherwise `()`
    public isolated function add(ai:VectorEntry[] entries) returns ai:Error? {
        if entries.length() == 0 {
            return;
        }
        PreparedEntry[] prepared = check prepareEntries(entries, self.config.indexConfig.similarityMetric);

        foreach PreparedEntry[] batch in chunkPreparedEntries(prepared, self.config.maxBulkSize) {
            byte[] body = check buildAddBulkBody(batch, self.indexName, self.deployment, self.config);
            BulkResponse response = check self.transport.bulk(body);
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
    # `Configuration.maxResultWindow`; an explicit `topK` above that ceiling is rejected. A
    # `SparseVector`/`HybridVector` embedding fails the call — this release supports dense vectors
    # only
    # + return - The matching entries, or an `ai:Error` on failure
    public isolated function query(ai:VectorStoreQuery query) returns ai:VectorMatch[]|ai:Error {
        json body = check buildSearchBody(query, self.config);
        SearchResponse response = check self.transport.search(self.indexName, body);
        // buildSearchBody already rejected a sparse/hybrid embedding, so by this point
        // query.embedding is either absent or an ai:Vector.
        boolean scoreIsMeaningful = query.embedding is ai:Vector;
        ai:VectorMatch[] matches = [];
        foreach SearchHit hit in response.hits.hits {
            matches.push(check hitToVectorMatch(hit, self.config, scoreIsMeaningful));
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
# + queryMode - The requested query mode
# + config - The vector store configuration
# + return - An `ai:Error` naming the first failing rule, otherwise `()`
isolated function validateConfiguration(string serviceUrl, Deployment deployment,
        ai:VectorStoreQueryMode queryMode, Configuration config) returns ai:Error? {
    if queryMode != ai:DENSE {
        return error(string `This module supports 'ai:DENSE' query mode only; ` +
                string `got '${queryMode}'`);
    }
    if config.indexConfig.dimension < 1 {
        return error(string `'IndexConfig.dimension' must be a positive integer, got: ` +
                string `${config.indexConfig.dimension}`);
    }
    string _ = check extractHost(serviceUrl);
    if config.maxBulkSize < 1 {
        return error(string `'Configuration.maxBulkSize' must be a positive integer, got: ${config.maxBulkSize}`);
    }
    if config.maxResultWindow < 1 {
        return error(string `'Configuration.maxResultWindow' must be a positive integer, got: ` +
                string `${config.maxResultWindow}`);
    }
    if deployment is ServerlessNextGenDeployment {
        check validateQuantization(deployment);
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
