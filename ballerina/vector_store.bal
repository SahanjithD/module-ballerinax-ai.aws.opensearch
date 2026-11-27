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
import ballerina/lang.runtime;
import ballerina/log;
import ballerinax/aws;

// OpenSearch "Terms query" docs: `index.max_terms_count` defaults to 65,536.
const int DOC_ID_LOOKUP_CHUNK_SIZE = 65536;

// Binary floating point does not sum 0.7 and 0.3 to exactly 1.0.
const float WEIGHT_SUM_TOLERANCE = 1.0e-6;

// OpenSearch's documented `knn_vector` maximum: "[1, 16,000]" (OpenSearch docs, k-NN vector field).
const int MAX_DIMENSION = 16000;

// Named rather than inline in `init`, since a bare record literal is ambiguous against the union.
# The default search mode: dense search over 1536-dimensional vectors.
final DenseSearch & readonly DEFAULT_SEARCH_MODE = {queryMode: ai:DENSE, indexConfig: {dimension: 1536}};

// Talks to the OpenSearch REST API directly, since there is no `ballerinax/opensearch` connector.
// `ManagedDomainVectorStore` and `ServerlessVectorStore` are thin public fronts over this class;
// per-deployment behavior, score ranges and hazards are documented in the README.
# The store implementation shared by both public stores.
isolated class VectorStoreCore {
    private final OpenSearchTransport transport;
    private final string indexName;
    private final Deployment & readonly deployment;
    private final SearchMode & readonly searchMode;
    private final StoreConfiguration & readonly config;

    # Initializes the store and, by default, creates and checks the index.
    #
    # + serviceUrl - The endpoint
    # + region - The AWS region of the endpoint
    # + indexName - The index this store reads and writes
    # + auth - SigV4 credentials, or `BasicAuth` on a managed domain
    # + deploymentConfig - The target deployment and its deployment-specific settings
    # + searchMode - Dense, sparse or hybrid search, with the index shape and fields it uses
    # + storeConfig - Settings shared by every search mode and deployment
    # + httpConfig - HTTP client settings
    # + createIndexIfNotExists - Whether `init` creates the index when it does not exist
    # + return - An `ai:Error` if validation, index creation or verification fails
    isolated function init(string serviceUrl, aws:Region|string region, string indexName,
            ManagedDomainAuthConfig auth, Deployment deploymentConfig, SearchMode searchMode = DEFAULT_SEARCH_MODE,
            StoreConfiguration storeConfig = {}, http:ClientConfiguration httpConfig = {},
            boolean createIndexIfNotExists = true)
            returns ai:Error? {
        check validateConfiguration(serviceUrl, deploymentConfig, searchMode, storeConfig);
        check validateIndexName(indexName);

        OpenSearchTransport transport = check new (serviceUrl, region, auth, deploymentConfig,
            storeConfig.retryConfig, httpConfig
        );
        ai:Error? preparation = prepareIndex(transport, indexName, searchMode, storeConfig, deploymentConfig,
                createIndexIfNotExists);
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
    isolated function add(ai:VectorEntry[] entries) returns ai:Error? {
        if entries.length() == 0 {
            return;
        }
        PreparedEntry[] prepared = check prepareEntries(entries, self.searchMode);

        foreach PreparedEntry[] batch in chunkPreparedEntries(prepared, self.config.maxBulkSize) {
            check self.indexBatch(batch);
        }
    }

    // A `_bulk` that returns 200 can still reject items with 429 when the cluster is overloaded.
    // Those were not applied, so they are re-sent on their own with `retryConfig`'s backoff, on
    // Serverless Classic too, but only when every failure is a 429.
    # Indexes one batch, re-sending the items the cluster rejected as overloaded.
    #
    # + batch - The entries of one `_bulk` request
    # + return - An `ai:Error` naming the entries that could not be indexed
    private isolated function indexBatch(PreparedEntry[] batch) returns ai:Error? {
        RetryConfig retryConfig = self.config.retryConfig;
        PreparedEntry[] pending = batch;
        // `maxDelay` is the longest delay between attempts, the first included.
        decimal delay = retryConfig.initialDelay < retryConfig.maxDelay ? retryConfig.initialDelay
            : retryConfig.maxDelay;
        int attempt = 0;
        while true {
            byte[] body = check buildAddBulkBody(pending, self.indexName, self.deployment, self.searchMode,
                    self.config);
            BulkResponse response = check self.transport.bulk(body, isIndexingBatch = true);
            if response.items.length() != pending.length() {
                return error(string `The '_bulk' response carried ${response.items.length()} results for ` +
                        string `${pending.length()} entries`);
            }
            // Matched by position, because on Serverless Classic the response `_id` is
            // server-generated.
            string[] submittedIds = from PreparedEntry entry in pending
                select entry.id;
            BulkFailure[] failures = extractIndexFailures(response, submittedIds);
            if failures.length() == 0 {
                return;
            }
            int[] rejected = extractRejectedPositions(response);
            if attempt >= retryConfig.maxRetries || rejected.length() != failures.length() {
                return error(string `Failed to index ${failures.length()} of ${batch.length()} ` +
                        string `entries: ${summarizeBulkFailures(failures)}`);
            }
            pending = from int position in rejected
                select pending[position];
            runtime:sleep(delay);
            decimal nextDelay = delay * retryConfig.backoffFactor;
            delay = nextDelay < retryConfig.maxDelay ? nextDelay : retryConfig.maxDelay;
            attempt += 1;
        }
    }

    # Searches by embedding, by metadata filter, or both. Without an embedding every match scores
    # 0.0.
    #
    # + query - The query. `topK < 1` returns all entries up to `maxResultWindow` (at most the
    # nearest 10,000 for a dense query); a HYBRID query with an embedding needs `topK >= 1`
    # + return - The matches, or an `ai:Error`
    isolated function query(ai:VectorStoreQuery query) returns ai:VectorMatch[]|ai:Error {
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
    isolated function delete(string|string[] ids) returns ai:Error? {
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
        return self.deleteInBatches(idList, idList);
    }

    # Deletes documents by `_id` in `_bulk` batches of at most `maxBulkSize`.
    #
    # + documentIds - The `_id`s to delete
    # + logicalIds - The caller's id for each, parallel to `documentIds`, to name failures by
    # + return - An `ai:Error` naming the ids that failed, if any
    private isolated function deleteInBatches(string[] documentIds, string[] logicalIds) returns ai:Error? {
        BulkFailure[] failures = [];
        int offset = 0;
        while offset < documentIds.length() {
            int end = int:min(offset + self.config.maxBulkSize, documentIds.length());
            byte[] body = buildDeleteBulkBody(self.indexName, documentIds.slice(offset, end));
            BulkResponse response = check self.transport.bulk(body);
            failures.push(...extractDeleteFailures(response, logicalIds.slice(offset, end)));
            offset = end;
        }
        if failures.length() > 0 {
            return error(string `Failed to delete ${failures.length()} of ${documentIds.length()} ` +
                    string `entries: ${summarizeBulkFailures(failures)}`);
        }
    }

    // `add` is append-only on Classic, so one id may match several documents, and a lookup that
    // fills the window may have missed some. It is split in half and retried until it fits.
    # Finds the internal `_id`s of the documents carrying the given logical ids.
    #
    # + ids - The logical ids to look up
    # + internalIds - Receives the internal `_id` of each match
    # + logicalIds - Receives each match's logical id, parallel to `internalIds`
    # + return - An `ai:Error` if a single id matches a full window of documents
    private isolated function lookUpInternalIds(string[] ids, string[] internalIds, string[] logicalIds)
            returns ai:Error? {
        json searchBody = buildDocIdLookupBody(self.config.idFieldName, ids, self.config.maxResultWindow);
        SearchResponse searchResponse = check self.transport.search(self.indexName, searchBody);
        // A timed-out search or a failed shard still answers 200, with only some of the hits.
        // Absent fields count as complete, since not every endpoint may send them.
        int failedShards = searchResponse?._shards?.failed ?: 0;
        if searchResponse?.timed_out == true || failedShards > 0 {
            return error(string `A lookup of ${ids.length()} id(s) to delete on Serverless Classic ` +
                    string `returned partial results (timed out, or ${failedShards} failed shard(s)), ` +
                    "so some copies could be left undeleted. Nothing was deleted; retry");
        }
        int hitCount = searchResponse.hits.hits.length();
        if hitCount >= self.config.maxResultWindow {
            if ids.length() > 1 {
                int half = ids.length() / 2;
                check self.lookUpInternalIds(ids.slice(0, half), internalIds, logicalIds);
                return self.lookUpInternalIds(ids.slice(half), internalIds, logicalIds);
            }
            return error(string `The id '${ids[0]}' matches at least ${hitCount} documents on Serverless ` +
                    string `Classic, the configured 'maxResultWindow', so some copies could be left ` +
                    "undeleted. Raise 'StoreConfiguration.maxResultWindow' and retry");
        }
        foreach SearchHit hit in searchResponse.hits.hits {
            internalIds.push(hit._id);
            map<json>? hitSource = hit?._source;
            json logicalId = hitSource is map<json> ? hitSource[self.config.idFieldName] : ();
            // A document written elsewhere may have no logical id.
            logicalIds.push(logicalId is string ? logicalId : hit._id);
        }
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
        // A chunk of more ids than the window could never fit, so none is larger.
        // Half the window, so a chunk whose ids each match one document leaves room for copies
        // instead of filling the window and being split.
        int chunkSize = int:min(DOC_ID_LOOKUP_CHUNK_SIZE, int:max(1, self.config.maxResultWindow / 2));
        foreach string[] chunk in chunkIds(idList, chunkSize) {
            check self.lookUpInternalIds(chunk, internalIds, logicalIds);
        }
        return self.deleteInBatches(internalIds, logicalIds);
    }
}

// The version is checked before the index is created, so a cluster too old for the search mode
// fails as such instead of leaving an unusable index. The mapping's own floors apply only when
// this call creates the index, since an existing one may use another engine than configured.
// A just-created mapping needs no check.
# Runs the construction steps that talk to the cluster.
#
# + transport - The transport to issue requests through
# + indexName - The index this store is pointed at
# + searchMode - The store's search mode
# + config - The vector store configuration
# + deployment - The target deployment
# + createIndexIfNotExists - Whether a missing index is created rather than reported
# + return - An `ai:Error` if the cluster or index cannot serve this store, otherwise `()`
isolated function prepareIndex(OpenSearchTransport transport, string indexName, SearchMode searchMode,
        StoreConfiguration config, Deployment deployment, boolean createIndexIfNotExists) returns ai:Error? {
    boolean missing = false;
    if createIndexIfNotExists {
        missing = !check transport.indexExists(indexName);
    }
    check verifyClusterVersion(transport, searchMode, deployment, missing);
    boolean created = missing && check createMissingIndex(transport, indexName, searchMode, config, deployment);
    if !created {
        check verifyIndexMapping(transport, indexName, searchMode);
    }
}

// A negative delay sleeps not at all, and a factor below 1 shrinks the delay, so either retries an
// overloaded cluster with no backoff.
# Checks that retries back off.
#
# + retryConfig - `StoreConfiguration.retryConfig`
# + return - An `ai:Error` naming the first invalid setting, otherwise `()`
isolated function validateRetryConfig(RetryConfig retryConfig) returns ai:Error? {
    if retryConfig.maxRetries < 0 {
        return error(string `'RetryConfig.maxRetries' must not be negative, got: ${retryConfig.maxRetries}`);
    }
    if retryConfig.initialDelay < 0d || retryConfig.maxDelay < 0d {
        return error(string `'RetryConfig.initialDelay' and '.maxDelay' must not be negative, got: ` +
                string `${retryConfig.initialDelay} and ${retryConfig.maxDelay}`);
    }
    if retryConfig.backoffFactor < 1d {
        return error(string `'RetryConfig.backoffFactor' must be at least 1, got: ${retryConfig.backoffFactor}`);
    }
}

// The rules OpenSearch applies to both index and alias names (`MetadataCreateIndexService.
// validateIndexOrAliasName`, `Strings.INVALID_FILENAME_CHARS`, OpenSearch 2.19.1). Lowercase is
// required of index names only, so it is not checked: an alias may hold uppercase letters. A blank
// name reads every index (`HEAD /` answers 200, `GET //_mapping` returns them all), and a pattern
// or list cannot be written to or created, and does not check indices it matches later.
# Checks that the index name is a single index or alias that OpenSearch would accept.
#
# + indexName - The configured index or alias
# + return - An `ai:Error` naming the problem, otherwise `()`
isolated function validateIndexName(string indexName) returns ai:Error? {
    if indexName == "" {
        return error("'indexName' must not be empty");
    }
    if indexName == "." || indexName == ".." {
        return error(string `'indexName' must not be '${indexName}'`);
    }
    if indexName.startsWith("_") || indexName.startsWith("-") || indexName.startsWith("+") {
        return error(string `'indexName' must not start with '_', '-' or '+', got: '${indexName}'`);
    }
    foreach string:Char c in indexName {
        if "\\/*?\"<>|,#: ".includes(c) {
            return error(string `'indexName' must not contain '${c}', which OpenSearch does not allow ` +
                    string `in an index or alias name, got: '${indexName}'`);
        }
    }
    if indexName.toBytes().length() > 255 {
        return error(string `'indexName' must be at most 255 bytes, got ${indexName.toBytes().length()}`);
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
        SearchMode searchMode, StoreConfiguration config) returns ai:Error? {
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
    if indexConfig is IndexConfig && (indexConfig.dimension < 1 || indexConfig.dimension > MAX_DIMENSION) {
        return error(string `'IndexConfig.dimension' must be between 1 and ${MAX_DIMENSION}, got: ` +
                string `${indexConfig.dimension}`);
    }
    check validateFieldNames(searchMode, config);
    string _ = check extractHost(serviceUrl);
    if config.maxBulkSize < 1 {
        return error(string `'StoreConfiguration.maxBulkSize' must be a positive integer, got: ${config.maxBulkSize}`);
    }
    if config.maxResultWindow < 1 {
        return error(string `'StoreConfiguration.maxResultWindow' must be a positive integer, got: ` +
                string `${config.maxResultWindow}`);
    }
    check validateRetryConfig(config.retryConfig);
    // A Classic delete looks each id up and treats a full window as "there may be more copies", so
    // a window of 1 could never delete an id that exists.
    if deployment is ServerlessClassicDeployment && config.maxResultWindow < 2 {
        return error(string `'StoreConfiguration.maxResultWindow' must be at least 2 on Serverless ` +
                string `Classic, where 'delete' looks ids up first, got: ${config.maxResultWindow}`);
    }
    if deployment is ServerlessNextGenDeployment {
        // Quantization shapes a `knn_vector` field, which a SPARSE index does not have, so setting
        // it is a mistake worth reporting rather than ignoring.
        if searchMode is SparseSearch && deployment.compressionLevel is CompressionLevel {
            return error("'ServerlessNextGenDeployment.compressionLevel' shapes the 'knn_vector' " +
                    "field, which a 'SPARSE' index does not have; leave it unset, or use 'HYBRID' " +
                    "if the index should also hold dense vectors");
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
isolated function validateFieldNames(SearchMode searchMode, StoreConfiguration config) returns ai:Error? {
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
                    "'StoreConfiguration.metadataFieldName' may be \"\", which selects a flat schema");
        }
        if seen.indexOf(name) !is () {
            return error(string `The document field name '${name}' is used more than once. The ` +
                    "vector, content, id, chunk-type and metadata fields must all be distinct, or " +
                    "one silently overwrites another in every stored document");
        }
        seen.push(name);
    }
    // OpenSearch maps a dotted name as nested objects, so `verifyMappedField` would not find the
    // field the next time `init` checks an index this module created.
    foreach string name in vectorFieldNamesOf(searchMode) {
        if name.includes(".") {
            return error(string `The vector field name '${name}' must not contain '.', which ` +
                    "OpenSearch reads as a path into nested objects");
        }
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
    float? denseWeightValue = fusion.denseWeight;
    float? sparseWeightValue = fusion.sparseWeight;
    if denseWeightValue is () && sparseWeightValue is () {
        return;
    }
    if denseWeightValue is () || sparseWeightValue is () {
        return error("'HybridSearchConfig.denseWeight' and '.sparseWeight' must be set together, " +
                "or both left unset to weigh the sub-queries equally");
    }
    float denseWeight = denseWeightValue;
    float sparseWeight = sparseWeightValue;
    // Written so `NaN`, for which every ordered comparison is false, fails too.
    if !(0.0 <= denseWeight && denseWeight <= 1.0) || !(0.0 <= sparseWeight && sparseWeight <= 1.0) {
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
    float? pruneRatio = acceleration.pruneRatio;
    // neural-search `PruneUtils.isValidPruneRatio` (2.19.1.0) accepts `[0, 1)` for this prune
    // type, and rejects 1.0. Written so `NaN` fails too.
    if pruneRatio is float && !(0.0 <= pruneRatio && pruneRatio < 1.0) {
        return error(string `'TwoPhaseAcceleration.pruneRatio' must be at least 0.0 and below 1.0, got: ` +
                string `${pruneRatio}. It is a fraction of the query's largest token weight, and ` +
                "is what separates the tokens that select candidates from the ones that rescore them");
    }
}
