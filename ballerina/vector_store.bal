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

// OpenSearch "Terms query" docs: `index.max_terms_count` defaults to 65,536.
const int DOC_ID_LOOKUP_CHUNK_SIZE = 65536;

// Binary floating point does not sum 0.7 and 0.3 to exactly 1.0.
const float WEIGHT_SUM_TOLERANCE = 1.0e-6;

// Named rather than inline in `init`, since a bare record literal is ambiguous against the union.
# The default search mode: dense search over 1536-dimensional vectors.
final DenseSearch & readonly DEFAULT_SEARCH_MODE = {queryMode: ai:DENSE, indexConfig: {dimension: 1536}};

// Talks to the OpenSearch REST API directly, since there is no `ballerinax/opensearch` connector.
// Per-deployment behavior, score ranges and hazards are documented in the README.
# An `ai:VectorStore` backed by an Amazon OpenSearch Service domain or an OpenSearch Serverless
# collection.
public isolated class VectorStore {
    *ai:VectorStore;

    private final OpenSearchTransport transport;
    private final string indexName;
    private final Deployment & readonly deployment;
    private final SearchMode & readonly searchMode;
    private final Configuration & readonly config;

    # Initializes the store and, by default, creates and checks the index.
    #
    # + serviceUrl - The endpoint, e.g. `https://my-domain.us-east-1.es.amazonaws.com`
    # + region - The AWS region of the endpoint
    # + indexName - The index this store reads and writes
    # + deploymentConfig - The deployment type, its credentials and its deployment-specific settings
    # + searchMode - Dense, sparse or hybrid search, with the index shape and fields it uses
    # + storeConfig - Settings shared by every search mode and deployment
    # + httpConfig - HTTP client settings. `httpVersion` and `http1Settings.chunking` are always set
    # to `HTTP_1_1` and `CHUNKING_NEVER`, which SigV4 signing needs
    # + return - An `ai:Error` if validation, index creation or verification fails
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
        ai:Error? preparation = prepareIndex(transport, indexName, searchMode, storeConfig, deploymentConfig);
        if preparation is ai:Error {
            // Best effort: a close failure must not mask the preparation error.
            ai:Error? closeResult = transport.close();
            if closeResult is ai:Error {
                log:printWarn("Failed to close the AWS credential provider after a failed index-preparation step",
                        'error = closeResult);
            }
            return preparation;
        }

        self.transport = transport;
        self.indexName = indexName;
        self.deployment = deploymentConfig.cloneReadOnly();
        self.searchMode = searchMode.cloneReadOnly();
        self.config = storeConfig.cloneReadOnly();
    }

    # Adds entries in `_bulk` batches of at most `maxBulkSize`. Entries without an `id` get a UUID.
    #
    # + entries - Entries whose embeddings match this store's search mode (`ai:Vector` for `DENSE`,
    # `ai:SparseVector` for `SPARSE`, `ai:HybridVector` for `HYBRID`)
    # + return - An `ai:Error` naming the failing entries, if any
    public isolated function add(ai:VectorEntry[] entries) returns ai:Error? {
        if entries.length() == 0 {
            return;
        }
        PreparedEntry[] prepared = check prepareEntries(entries, self.searchMode);

        foreach PreparedEntry[] batch in chunkPreparedEntries(prepared, self.config.maxBulkSize) {
            byte[] body = check buildAddBulkBody(batch, self.indexName, self.deployment, self.searchMode,
                    self.config);
            BulkResponse response = check self.transport.bulk(body, isIndexingBatch = true);
            // Matched by position, because on `SERVERLESS_CLASSIC` the response `_id` is
            // server-generated.
            string[] submittedIds = from PreparedEntry entry in batch
                select entry.id;
            BulkFailure[] failures = extractIndexFailures(response, submittedIds);
            if failures.length() > 0 {
                return error(string `Failed to index ${failures.length()} of ${batch.length()} ` +
                        string `entries: ${summarizeBulkFailures(failures)}`);
            }
        }
    }

    # Searches by embedding, by metadata filter, or both. Without an embedding every match scores
    # 0.0.
    #
    # + query - The query. Any `topK < 1` returns all entries, up to `maxResultWindow`
    # + return - The matches, or an `ai:Error`
    public isolated function query(ai:VectorStoreQuery query) returns ai:VectorMatch[]|ai:Error {
        json body = check buildSearchBody(query, self.searchMode, self.config);
        // `buildSearchBody` rejects a mismatched embedding, so one present here is being scored.
        boolean scoreIsMeaningful = query.embedding !is ();
        SearchResponse response = check self.transport.search(self.indexName, body,
                searchPipelineName(self.searchMode, scoreIsMeaningful));
        ai:VectorMatch[] matches = [];
        foreach SearchHit hit in response.hits.hits {
            matches.push(check hitToVectorMatch(hit, self.searchMode, self.config, scoreIsMeaningful));
        }
        return matches;
    }

    // A missing id is not an error, as in the Pinecone, pgvector and Weaviate stores (unlike
    // `ai:InMemoryVectorStore`).
    # Deletes entries by logical id. Ids that do not exist are ignored.
    #
    # + ids - The id or ids to delete
    # + return - An `ai:Error` naming the ids that failed, if any
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

    # Releases the AWS credential provider's refresh threads and connections. Not called
    # automatically, since `ai:VectorStore` has no lifecycle hook.
    #
    # + return - An `ai:Error` if releasing them fails
    public isolated function close() returns ai:Error? {
        return self.transport.close();
    }

    // `_delete_by_query` is not available on AOSS, so internal `_id`s are looked up first.
    // Entries added within the last refresh interval are not found yet and survive.
    # Deletes on Serverless Classic: looks up the internal `_id`s, then bulk-deletes them.
    #
    # + idList - The logical ids to delete
    # + return - An `ai:Error` naming the ids that failed, or if a lookup may have missed some
    private isolated function deleteOnServerlessClassic(string[] idList) returns ai:Error? {
        string[] internalIds = [];
        // Parallel to `internalIds`, so a failure is reported against the caller's id.
        string[] logicalIds = [];
        foreach string[] chunk in chunkIds(idList, DOC_ID_LOOKUP_CHUNK_SIZE) {
            json searchBody = buildDocIdLookupBody(self.config.idFieldName, chunk, self.config.maxResultWindow);
            SearchResponse searchResponse = check self.transport.search(self.indexName, searchBody);
            int hitCount = searchResponse.hits.hits.length();
            // `add` is append-only here, so a full page may have missed duplicates.
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
                // A document written elsewhere may have no logical id.
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

// The version is checked before the index is created, so a cluster too old for the search mode
// fails as such instead of leaving an unusable index. A just-created mapping needs no check.
# Runs the construction steps that talk to the cluster.
#
# + transport - The transport to issue requests through
# + indexName - The index this store is pointed at
# + searchMode - The store's search mode
# + config - The vector store configuration
# + deployment - The target deployment
# + return - An `ai:Error` if the cluster or index cannot serve this store, otherwise `()`
isolated function prepareIndex(OpenSearchTransport transport, string indexName, SearchMode searchMode,
        Configuration config, Deployment deployment) returns ai:Error? {
    if config.verifyOnInit {
        check verifyClusterVersion(transport, searchMode, deployment);
    }
    boolean created = check ensureIndex(transport, indexName, searchMode, config, deployment);
    if config.verifyOnInit && !created {
        check verifyIndexMapping(transport, indexName, searchMode);
    }
}

# Runs the construction checks that need no network I/O.
#
# + serviceUrl - The service URL, checked for a parseable host
# + deployment - The target deployment
# + searchMode - The requested search mode
# + config - The vector store configuration
# + return - An `ai:Error` naming the first failing rule, otherwise `()`
isolated function validateConfiguration(string serviceUrl, Deployment deployment,
        SearchMode searchMode, Configuration config) returns ai:Error? {
    if searchMode is SparseSearch {
        if searchMode.maxQueryTokens < 1 {
            return error(string `'SparseSearch.maxQueryTokens' must be a positive integer, got: ` +
                    string `${searchMode.maxQueryTokens}`);
        }
        check validateTwoPhaseAcceleration(searchMode.twoPhaseAcceleration);
    }
    if searchMode is HybridSearch {
        if searchMode.maxQueryTokens < 1 {
            return error(string `'HybridSearch.maxQueryTokens' must be a positive integer, got: ` +
                    string `${searchMode.maxQueryTokens}`);
        }
        check validateHybridFusion(searchMode.fusion);
        check validateEfSearch(searchMode.efSearch, "HybridSearch");
    }
    if searchMode is DenseSearch {
        check validateEfSearch(searchMode.efSearch, "DenseSearch");
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
        // Quantization shapes a `knn_vector` field, which a SPARSE index does not have, so setting
        // it is a mistake worth reporting rather than ignoring.
        if searchMode is SparseSearch && deployment.compressionLevel is CompressionLevel {
            return error("'ServerlessNextGenDeployment.compressionLevel' shapes the 'knn_vector' " +
                    "field, which a 'SPARSE' index does not have; leave it unset, or use 'HYBRID' " +
                    "if the index should also hold dense vectors");
        }
        if deployment?.collectionName is string && deployment?.collectionId is string {
            return error("'ServerlessNextGenDeployment.collectionName' and '.collectionId' are " +
                    "alternatives; set at most one. Sending both collection headers leaves it to " +
                    "the AOSS proxy to decide which one identifies the target collection");
        }
    }
}

// A collision breaks the index silently: `vectorFieldName: "content"` would overwrite the chunk
// text with the vector on every write.
# Rejects document field names that are blank or used twice.
#
# + searchMode - The search mode, supplying whichever vector field names it declares
# + config - The vector store configuration
# + return - An `ai:Error` naming the colliding field, otherwise `()`
isolated function validateFieldNames(SearchMode searchMode, Configuration config) returns ai:Error? {
    string[] names = [config.contentFieldName, config.idFieldName, CHUNK_TYPE_FIELD];
    names.push(...vectorFieldNamesOf(searchMode));
    // `""` selects a flat schema with no metadata field; `buildEntrySource` checks keys instead.
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

// Checked up front because the pipeline goes with every query, so a bad value fails every call.
# Checks hybrid fusion settings against what the `normalization-processor` accepts.
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
    if fusion is RrfFusion {
        return;
    }
    float denseWeight = fusion.denseWeight;
    float sparseWeight = fusion.sparseWeight;
    if denseWeight < 0.0 || denseWeight > 1.0 || sparseWeight < 0.0 || sparseWeight > 1.0 {
        return error(string `'HybridSearchConfig.denseWeight' and '.sparseWeight' must each be ` +
                string `between 0.0 and 1.0, got: ${denseWeight} and ${sparseWeight}`);
    }
    float sum = denseWeight + sparseWeight;
    if sum - 1.0 > WEIGHT_SUM_TOLERANCE || 1.0 - sum > WEIGHT_SUM_TOLERANCE {
        return error(string `'HybridSearchConfig.denseWeight' and '.sparseWeight' must sum to 1.0, ` +
                string `got: ${denseWeight} + ${sparseWeight} = ${sum}. The ` +
                "'normalization-processor' rejects a weight list that does not sum to 1.0");
    }
}

# Rejects a non-positive `efSearch`. Unset is always valid.
#
# + efSearch - The configured value, or `()`
# + owner - The `SearchMode` variant the field is on, for the error message
# + return - An `ai:Error` if the value is not positive, otherwise `()`
isolated function validateEfSearch(int? efSearch, string owner) returns ai:Error? {
    if efSearch is int && efSearch < 1 {
        return error(string `'${owner}.efSearch' must be a positive integer, got: ${efSearch}. ` +
                "Leave it unset to let the index's own 'index.knn.algo_param.ef_search' apply");
    }
}

# Rejects a two-phase `pruneRatio` outside `[0, 1]`.
#
# + acceleration - The configured acceleration, or `()` when the processor is not used
# + return - An `ai:Error` naming the failing rule, otherwise `()`
isolated function validateTwoPhaseAcceleration(TwoPhaseAcceleration? acceleration) returns ai:Error? {
    if acceleration is () {
        return;
    }
    float pruneRatio = acceleration.pruneRatio;
    if pruneRatio < 0.0 || pruneRatio > 1.0 {
        return error(string `'TwoPhaseAcceleration.pruneRatio' must be between 0.0 and 1.0, got: ` +
                string `${pruneRatio}. It is a fraction of the query's largest token weight, and ` +
                "is what separates the tokens that select candidates from the ones that rescore them");
    }
}
