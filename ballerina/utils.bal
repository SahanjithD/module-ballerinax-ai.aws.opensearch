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
import ballerina/time;
import ballerina/uuid;

// Request and response shaping. Pure, so it is unit-testable without a cluster.

# Field holding `ai:Chunk.'type`, so a chunk's type survives a round trip.
const string CHUNK_TYPE_FIELD = "chunk_type";

# The mode's `IndexConfig`, or `()` under `SPARSE`, which has no dense vector.
#
# + searchMode - The kind of search
# + return - The `knn_vector` shape, or `()` if this mode has no dense vector
isolated function indexConfigOf(SearchMode searchMode) returns IndexConfig? {
    if searchMode is DenseSearch {
        return searchMode.indexConfig;
    }
    if searchMode is HybridSearch {
        return searchMode.indexConfig;
    }
    return ();
}

# The document field holding the dense vector, or `()` under `SPARSE`.
#
# + searchMode - The kind of search
# + return - The `knn_vector` field name, or `()` if this mode stores no dense vector
isolated function denseFieldNameOf(SearchMode searchMode) returns string? {
    if searchMode is DenseSearch {
        return searchMode.vectorFieldName;
    }
    if searchMode is HybridSearch {
        return searchMode.vectorFieldName;
    }
    return ();
}

# The document field holding the sparse vector, or `()` under `DENSE`.
#
# + searchMode - The kind of search
# + return - The `rank_features` field name, or `()` if this mode stores no sparse vector
isolated function sparseFieldNameOf(SearchMode searchMode) returns string? {
    if searchMode is SparseSearch {
        return searchMode.sparseVectorFieldName;
    }
    if searchMode is HybridSearch {
        return searchMode.sparseVectorFieldName;
    }
    return ();
}

# Every field the mode stores a vector in: one under `DENSE` or `SPARSE`, two under `HYBRID`.
#
# + searchMode - The kind of search
# + return - The vector field names, in dense-then-sparse order
isolated function vectorFieldNamesOf(SearchMode searchMode) returns string[] {
    string[] names = [];
    string? denseName = denseFieldNameOf(searchMode);
    if denseName is string {
        names.push(denseName);
    }
    string? sparseName = sparseFieldNameOf(searchMode);
    if sparseName is string {
        names.push(sparseName);
    }
    return names;
}

// Built only by `prepareEntries`: `DENSE` sets `embedding`, `SPARSE` sets `rankFeatures`, and
// `HYBRID` sets both.
# An entry checked against the store's mode, with its id resolved and sparse half converted.
type PreparedEntry record {|
    # The resolved logical id.
    string id;
    # The dense embedding, or `()` under `SPARSE`.
    ai:Vector? embedding = ();
    # The sparse embedding as `{"<index>": <weight>}`, or `()` under `DENSE`.
    map<float>? rankFeatures = ();
    # The source chunk.
    ai:Chunk chunk;
|};

# Names the `ai:Embedding` member a value is, for an error message.
#
# + embedding - The embedding to describe
# + return - `dense`, `sparse`, or `hybrid`
isolated function describeEmbeddingKind(ai:Embedding embedding) returns string {
    if embedding is ai:Vector {
        return "dense";
    }
    return embedding is ai:SparseVector ? "sparse" : "hybrid";
}

# The error for an embedding that does not match the store's search mode, in `add` or `query`.
#
# + subject - What to name, e.g. `Entry 'abc'` or `The query`
# + embedding - The offending embedding
# + searchMode - The store's search mode
# + return - The error
isolated function embeddingMismatchError(string subject, ai:Embedding embedding, SearchMode searchMode)
        returns ai:Error {
    string required = searchMode is DenseSearch ? "'ai:Vector'"
        : searchMode is SparseSearch ? "'ai:SparseVector'" : "'ai:HybridVector'";
    return error(string `${subject} has a ${describeEmbeddingKind(embedding)} embedding, but this ` +
            string `store was constructed with '${searchMode.queryMode}' query mode, which requires ` +
            string `${required}`);
}

# Checks entries against the store's search mode and resolves their ids.
#
# + entries - The caller-supplied entries
# + searchMode - The store's search mode, which decides the required `ai:Embedding` member
# + return - The prepared entries, or an `ai:Error` naming the offending entry
isolated function prepareEntries(ai:VectorEntry[] entries, SearchMode searchMode) returns PreparedEntry[]|ai:Error {
    IndexConfig? indexConfig = indexConfigOf(searchMode);
    ai:SimilarityMetric metric = indexConfig is IndexConfig ? indexConfig.similarityMetric : ai:COSINE;
    PreparedEntry[] prepared = [];
    foreach ai:VectorEntry entry in entries {
        ai:Embedding embedding = entry.embedding;
        string id = entry.id ?: uuid:createRandomUuid();
        string subject = string `Entry '${id}'`;
        if searchMode is SparseSearch {
            if embedding !is ai:SparseVector {
                return embeddingMismatchError(subject, embedding, searchMode);
            }
            // No ceiling at index time; the `(0, 64]` bound applies to queries only.
            prepared.push({
                id,
                rankFeatures: check toSparseTokenMap(embedding, subject, float:Infinity),
                chunk: entry.chunk
            });
            continue;
        }
        if searchMode is HybridSearch {
            if embedding !is ai:HybridVector {
                return embeddingMismatchError(subject, embedding, searchMode);
            }
            check validateEmbedding(embedding.dense, metric, id);
            prepared.push({
                id,
                embedding: embedding.dense,
                rankFeatures: check toSparseTokenMap(embedding.sparse, subject, float:Infinity),
                chunk: entry.chunk
            });
            continue;
        }
        if embedding !is ai:Vector {
            return embeddingMismatchError(subject, embedding, searchMode);
        }
        check validateEmbedding(embedding, metric, id);
        prepared.push({id, embedding, chunk: entry.chunk});
    }
    return prepared;
}

# Splits prepared entries into groups of at most `chunkSize`.
#
# + entries - The prepared entries
# + chunkSize - The maximum number of entries per group; must be positive
# + return - The entries split into groups, preserving order. Empty `entries` produce no groups
isolated function chunkPreparedEntries(PreparedEntry[] entries, int chunkSize) returns PreparedEntry[][] {
    PreparedEntry[][] batches = [];
    int offset = 0;
    while offset < entries.length() {
        int end = int:min(offset + chunkSize, entries.length());
        batches.push(entries.slice(offset, end));
        offset = end;
    }
    return batches;
}

# Splits ids into groups of at most `chunkSize`.
#
# + ids - The ids to split
# + chunkSize - The maximum number of ids per group; must be positive
# + return - The ids split into groups, preserving order. Empty `ids` produce no groups
isolated function chunkIds(string[] ids, int chunkSize) returns string[][] {
    string[][] batches = [];
    int offset = 0;
    while offset < ids.length() {
        int end = int:min(offset + chunkSize, ids.length());
        batches.push(ids.slice(offset, end));
        offset = end;
    }
    return batches;
}

# Rejects an all-zero vector under cosine, which would otherwise fail the whole `_bulk` request.
#
# + embedding - The dense vector to check
# + metric - The index's configured similarity metric
# + id - The entry id, used to name the offending entry in the error message
# + return - An `ai:Error` if `embedding` is all zeros under `COSINE`, otherwise `()`
isolated function validateEmbedding(ai:Vector embedding, ai:SimilarityMetric metric, string id) returns ai:Error? {
    if metric != ai:COSINE || embedding.length() == 0 {
        return;
    }
    foreach float value in embedding {
        if value != 0.0 {
            return;
        }
    }
    return error(string `Entry '${id}' has a zero vector, which is invalid under cosine similarity ` +
            "(the magnitude is zero)");
}

// Under a flat schema a metadata key such as `content` would overwrite a reserved field, so it is
// rejected.
# Builds the stored document for an entry, with metadata nested or, under a flat schema, spread.
#
# + entry - The prepared entry
# + searchMode - The store's search mode, which decides which vector field(s) are written
# + config - The vector store configuration
# + return - The document, or an `ai:Error` if a flat-schema metadata key is a reserved name
isolated function buildEntrySource(PreparedEntry entry, SearchMode searchMode, Configuration config)
        returns map<json>|ai:Error {
    anydata content = entry.chunk.content;
    map<json> sourceDoc = {
        [config.contentFieldName]: content is string ? content : content.toString(),
        [config.idFieldName]: entry.id,
        [CHUNK_TYPE_FIELD]: entry.chunk.'type
    };
    // Written first, so the flat-schema collision check below covers the vector fields too.
    string? denseFieldName = denseFieldNameOf(searchMode);
    ai:Vector? embedding = entry.embedding;
    if denseFieldName is string && embedding is ai:Vector {
        sourceDoc[denseFieldName] = embedding.cloneReadOnly();
    }
    string? sparseFieldName = sparseFieldNameOf(searchMode);
    map<float>? rankFeatures = entry.rankFeatures;
    if sparseFieldName is string && rankFeatures is map<float> {
        sourceDoc[sparseFieldName] = rankFeatures.cloneReadOnly();
    }
    map<json> metadata = transformMetadata(entry.chunk?.metadata);
    if config.metadataFieldName == "" {
        foreach [string, json] [metaKey, metaValue] in metadata.entries() {
            if sourceDoc.hasKey(metaKey) {
                return error(string `Entry '${entry.id}' has a metadata key '${metaKey}' that collides ` +
                        "with a reserved document field under a flat schema " +
                        "('Configuration.metadataFieldName' is \"\"); rename the metadata key, or nest " +
                        "metadata by setting a non-empty 'metadataFieldName'");
            }
            sourceDoc[metaKey] = metaValue;
        }
    } else {
        sourceDoc[config.metadataFieldName] = metadata;
    }
    return sourceDoc;
}

// Serverless Classic rejects a custom `_id`, so the action line omits it there. Elsewhere `_id`
// makes the write an upsert.
# Builds the NDJSON body of a `_bulk` indexing request.
#
# + entries - The prepared entries to index, already sized to fit one `_bulk` request
# + indexName - The target index
# + deployment - The target deployment, which gates whether `_id` is written
# + searchMode - The store's search mode, passed through to the document builder
# + config - The vector store configuration
# + return - The bytes to sign and send (empty for no entries), or an `ai:Error` from
# `buildEntrySource`
isolated function buildAddBulkBody(PreparedEntry[] entries, string indexName, Deployment deployment,
        SearchMode searchMode, Configuration config) returns byte[]|ai:Error {
    if entries.length() == 0 {
        return [];
    }
    string[] lines = [];
    foreach PreparedEntry entry in entries {
        json action = deployment is ServerlessClassicDeployment
            ? {"index": {"_index": indexName}}
            : {"index": {"_index": indexName, "_id": entry.id}};
        lines.push(action.toJsonString());
        lines.push((check buildEntrySource(entry, searchMode, config)).toJsonString());
    }
    return (string:'join("\n", ...lines) + "\n").toBytes();
}

# Builds the NDJSON body of a `_bulk` delete for the given `_id`s.
#
# + indexName - The target index
# + ids - The `_id`s to delete
# + return - The bytes to sign and send, empty for no ids
isolated function buildDeleteBulkBody(string indexName, string[] ids) returns byte[] {
    if ids.length() == 0 {
        return [];
    }
    string[] lines = [];
    foreach string id in ids {
        json action = {"delete": {"_index": indexName, "_id": id}};
        lines.push(action.toJsonString());
    }
    return (string:'join("\n", ...lines) + "\n").toBytes();
}

# Builds the search that finds the internal `_id`s for logical ids, for a Serverless Classic delete.
#
# + idFieldName - The document field carrying the logical id (`Configuration.idFieldName`)
# + ids - The logical ids to look up. Must be chunked by the caller to at most 65,536 terms
# + maxResultWindow - The `size`; one logical id may match several documents on Classic
# + return - The search request body
isolated function buildDocIdLookupBody(string idFieldName, string[] ids, int maxResultWindow) returns json {
    return {
        "size": maxResultWindow,
        // Returns the logical id too, so a failed delete can name the id the caller knows.
        "_source": [idFieldName],
        "track_total_hits": false,
        "query": {
            "terms": {
                [idFieldName]: ids
            }
        }
    };
}

# Builds the `_search` body for a query with any combination of embedding and filters.
#
# + query - The vector store query
# + searchMode - The store's search mode, which decides the scoring clause
# + config - The vector store configuration
# + return - The search request body, or an `ai:Error` if `topK` or the filters are invalid
isolated function buildSearchBody(ai:VectorStoreQuery query, SearchMode searchMode, Configuration config)
        returns json|ai:Error {
    int topK = query.topK;
    if topK > config.maxResultWindow {
        return error(string `'topK' (${topK}) exceeds the configured 'maxResultWindow' ` +
                string `(${config.maxResultWindow})`);
    }
    int size = topK < 1 ? config.maxResultWindow : topK;

    ai:Embedding? embedding = query.embedding;
    ai:MetadataFilters? filters = query.filters;

    json? filterClause = ();
    if filters is ai:MetadataFilters {
        filterClause = check convertFilters(filters, config.metadataFieldName);
    }

    json sourceDirective = config.includeEmbeddingsInResults
        ? true
        : {"excludes": vectorFieldNamesOf(searchMode)};

    json queryClause;
    // Only set when there is an embedding for the pipeline to act on, matching
    // `searchPipelineName`.
    json? inlinePipeline = ();
    if embedding is () {
        queryClause = filterClause is () ? {"match_all": {}} : {"bool": {"filter": [filterClause]}};
    } else if searchMode is SparseSearch {
        if embedding !is ai:SparseVector {
            return embeddingMismatchError("The query", embedding, searchMode);
        }
        queryClause = check buildSparseQueryClause(embedding, searchMode.sparseVectorFieldName,
                searchMode.maxQueryTokens, filterClause);
        TwoPhaseAcceleration? acceleration = searchMode.twoPhaseAcceleration;
        if acceleration is TwoPhaseAcceleration {
            inlinePipeline = buildInlineTwoPhasePipeline(acceleration);
        }
    } else if searchMode is HybridSearch {
        if embedding !is ai:HybridVector {
            return embeddingMismatchError("The query", embedding, searchMode);
        }
        // A `NamedSearchPipeline` goes as a query parameter instead; OpenSearch rejects both.
        HybridFusion fusion = searchMode.fusion;
        if fusion is InlineFusion {
            inlinePipeline = buildInlineFusionPipeline(fusion);
        }
        queryClause = check buildHybridQueryClause(embedding, searchMode, size, filterClause);
    } else {
        if embedding !is ai:Vector {
            return embeddingMismatchError("The query", embedding, searchMode);
        }
        queryClause = buildDenseQueryClause(embedding, searchMode.vectorFieldName, size, filterClause,
                searchMode.efSearch);
    }

    map<json> body = {
        "size": size,
        "_source": sourceDirective,
        "track_total_hits": false,
        "query": queryClause
    };
    if inlinePipeline !is () {
        body["search_pipeline"] = inlinePipeline;
    }
    return body;
}

// `hybrid` must be the top-level query, so this is never wrapped. Filters are repeated in each
// sub-query because a top-level `hybrid.filter` is OpenSearch 3.0+.
# Builds the `hybrid` clause from a dense and a sparse sub-query.
#
# + embedding - The query hybrid vector
# + searchMode - The hybrid search mode, supplying both field names and the token ceiling
# + size - The `k` to request from the dense sub-query
# + filterClause - The translated metadata filter, or `()`
# + return - The `hybrid` clause, or an `ai:Error` if the sparse half is invalid
isolated function buildHybridQueryClause(ai:HybridVector embedding, HybridSearch searchMode, int size,
        json? filterClause) returns json|ai:Error {
    // Dense first: `buildInlineFusionPipeline` applies the weights by position.
    json[] subQueries = [
        buildDenseQueryClause(embedding.dense, searchMode.vectorFieldName, size, filterClause,
                searchMode.efSearch),
        check buildSparseQueryClause(embedding.sparse, searchMode.sparseVectorFieldName,
                searchMode.maxQueryTokens, filterClause)
    ];
    return {"hybrid": {"queries": subQueries}};
}

// Without a fusion processor, a `hybrid` query returns OpenSearch's negative sub-query delimiter
// scores as results; `checkFusedScore` catches that.
# Builds the inline `search_pipeline` that fuses the sub-query scores.
#
# + fusion - The fusion settings, score-based or rank-based
# + return - The inline pipeline object
isolated function buildInlineFusionPipeline(InlineFusion fusion) returns json {
    if fusion is RrfFusion {
        // The technique alone; see `RrfFusion` for why nothing else is sent.
        return {
            "phase_results_processors": [
                {"score-ranker-processor": {"combination": {"technique": fusion.technique}}}
            ]
        };
    }
    // In the sub-query order of `buildHybridQueryClause`.
    float[] weights = [fusion.denseWeight, fusion.sparseWeight];
    return {
        "phase_results_processors": [
            {
                "normalization-processor": {
                    "normalization": {"technique": fusion.normalization},
                    "combination": {
                        "technique": fusion.combination,
                        "parameters": {"weights": weights}
                    }
                }
            }
        ]
    };
}

# Builds the inline `search_pipeline` for two-phase `neural_sparse` acceleration.
#
# + acceleration - The two-phase settings
# + return - The inline pipeline object
isolated function buildInlineTwoPhasePipeline(TwoPhaseAcceleration acceleration) returns json => {
    "request_processors": [
        {
            "neural_sparse_two_phase_processor": {
                "enabled": true,
                // Singular, as OpenSearch names it; an unknown key fails the search. The other
                // parameters are left to the server's defaults.
                "two_phase_parameter": {"prune_ratio": acceleration.pruneRatio}
            }
        }
    ]
};

// Never both inline and named: OpenSearch rejects a request carrying both.
# The pipeline name to send as `?search_pipeline=`, or `()` if fusion is inline or there is no
# embedding to fuse.
#
# + searchMode - The store's search mode
# + hasEmbedding - Whether the query carried an embedding, and so produced a `hybrid` clause
# + return - The pipeline name, or `()`
isolated function searchPipelineName(SearchMode searchMode, boolean hasEmbedding) returns string? {
    if searchMode !is HybridSearch || !hasEmbedding {
        return ();
    }
    HybridFusion fusion = searchMode.fusion;
    return fusion is NamedSearchPipeline ? fusion.name : ();
}

// Filters go inside `knn` rather than in `post_filter`, which can return fewer than `k` results.
// `method_parameters` is OpenSearch 2.16+, so it is sent only when `efSearch` is set.
# Builds the `knn` clause for a dense query.
#
# + embedding - The query vector
# + vectorFieldName - The `knn_vector` field to search
# + size - The `k` to request
# + filterClause - The translated metadata filter, or `()`
# + efSearch - The search-time HNSW candidate count, or `()` to leave the index setting in force
# + return - The `knn` clause
isolated function buildDenseQueryClause(ai:Vector embedding, string vectorFieldName, int size,
        json? filterClause, int? efSearch = ()) returns json {
    map<json> knnBody = {"vector": embedding.cloneReadOnly(), "k": size};
    if filterClause !is () {
        knnBody["filter"] = filterClause;
    }
    if efSearch is int {
        knnBody["method_parameters"] = {"ef_search": efSearch};
    }
    return {"knn": {[vectorFieldName]: knnBody}};
}

// `neural_sparse` over `rank_features` has no `filter` parameter, so the filter is a `bool`
// sibling, which does not affect the score.
# Builds the `neural_sparse` clause, wrapped in a `bool` when there is a filter.
#
# + embedding - The query sparse vector
# + sparseVectorFieldName - The `rank_features` field to search
# + maxQueryTokens - The client-side ceiling on the number of non-zero terms
# + filterClause - The translated metadata filter, or `()`
# + return - The query clause, or an `ai:Error` if the vector is invalid or too wide
isolated function buildSparseQueryClause(ai:SparseVector embedding, string sparseVectorFieldName,
        int maxQueryTokens, json? filterClause) returns json|ai:Error {
    int tokenCount = embedding.indices.length();
    // Checked here: past `max_clause_count` the server fails with a shard exception naming neither.
    if tokenCount > maxQueryTokens {
        return error(string `The query sparse vector has ${tokenCount} terms, which exceeds the ` +
                string `configured 'maxQueryTokens' (${maxQueryTokens}). A 'neural_sparse' query ` +
                "becomes one Lucene clause per term and is bounded by the cluster's " +
                "'indices.query.bool.max_clause_count'; prune the query vector to its highest-weighted " +
                "terms, or raise both that cluster setting and 'maxQueryTokens'");
    }
    map<float> queryTokens = check toSparseTokenMap(embedding, "The query", MAX_QUERY_TOKEN_WEIGHT);
    json neuralSparse = {"neural_sparse": {[sparseVectorFieldName]: {"query_tokens": queryTokens}}};
    if filterClause is () {
        return neuralSparse;
    }
    return {"bool": {"must": [neuralSparse], "filter": [filterClause]}};
}

# Translates metadata filters into an OpenSearch query clause. Empty groups add no clause.
#
# + node - A filter leaf or a nested filter group
# + metadataFieldName - `Configuration.metadataFieldName`; `""` addresses bare `<key>` paths
# + return - The translated clause, `()` if `node` (or every branch under it) is empty, or an
# `ai:Error` if a leaf is malformed (e.g. `IN`/`NOT_IN` with a non-array value)
isolated function convertFilters(ai:MetadataFilters|ai:MetadataFilter node, string metadataFieldName)
        returns json?|ai:Error {
    if node is ai:MetadataFilter {
        json value = node.value;
        if (node.key == "createdAt" || node.key == "modifiedAt") && value is time:Utc {
            value = time:utcToString(value);
        }
        return buildFilterClause(fieldPath(metadataFieldName, node.key), node.operator, value);
    }

    json[] clauses = [];
    foreach ai:MetadataFilters|ai:MetadataFilter child in node.filters {
        json? clause = check convertFilters(child, metadataFieldName);
        if clause !is () {
            clauses.push(clause);
        }
    }
    if clauses.length() == 0 {
        return ();
    }
    if node.condition == ai:OR {
        return {"bool": {"should": clauses, "minimum_should_match": 1}};
    }
    return {"bool": {"filter": clauses}};
}

# Translates a single `ai:MetadataFilter` leaf into its OpenSearch clause.
#
# + path - The full field path, e.g. `metadata.author`
# + operator - The comparison operator
# + value - The comparison value, already coerced for `time:Utc` fields
# + return - The translated clause, or an `ai:Error` for `IN`/`NOT_IN` with a non-array value
isolated function buildFilterClause(string path, ai:MetadataFilterOperator operator, json value)
        returns json|ai:Error {
    match operator {
        ai:EQUAL => {
            return {"term": {[path]: value}};
        }
        ai:NOT_EQUAL => {
            return {"bool": {"must_not": [{"term": {[path]: value}}]}};
        }
        ai:GREATER_THAN => {
            return {"range": {[path]: {"gt": value}}};
        }
        ai:GREATER_THAN_OR_EQUAL => {
            return {"range": {[path]: {"gte": value}}};
        }
        ai:LESS_THAN => {
            return {"range": {[path]: {"lt": value}}};
        }
        ai:LESS_THAN_OR_EQUAL => {
            return {"range": {[path]: {"lte": value}}};
        }
        ai:IN => {
            if value !is json[] {
                return error(string `'IN' filter on '${path}' requires an array value, got: ${value.toJsonString()}`);
            }
            return {"terms": {[path]: value}};
        }
        ai:NOT_IN => {
            if value !is json[] {
                return error(
                        string `'NOT_IN' filter on '${path}' requires an array value, got: ${value.toJsonString()}`);
            }
            return {"bool": {"must_not": [{"terms": {[path]: value}}]}};
        }
    }
    return error(string `Unsupported metadata filter operator: ${operator}`);
}

# The field path for a metadata key: bare under a flat schema, otherwise nested.
#
# + metadataFieldName - `Configuration.metadataFieldName`
# + key - The metadata key
# + return - `key` unchanged if `metadataFieldName` is `""`, else `<metadataFieldName>.<key>`
isolated function fieldPath(string metadataFieldName, string key) returns string {
    if metadataFieldName == "" {
        return key;
    }
    return string `${metadataFieldName}.${key}`;
}

# Converts `ai:Metadata` to its stored form, writing `createdAt`/`modifiedAt` as RFC 3339 strings.
#
# + metadata - The chunk metadata, or `()`
# + return - The wire-form metadata map, `{}` if `metadata` is `()`
isolated function transformMetadata(ai:Metadata? metadata) returns map<json> {
    if metadata is () {
        return {};
    }
    map<json> properties = {};
    foreach string key in metadata.keys() {
        anydata value = metadata.get(key);
        if value is time:Utc && (key == "createdAt" || key == "modifiedAt") {
            properties[key] = time:utcToString(value);
        } else if value is json {
            properties[key] = value;
        }
    }
    return properties;
}

// Other fractional numbers come back as `decimal`, as Ballerina parses non-integral JSON numbers
// that way. Coercing to `float` would lose precision for callers who passed a `decimal`.
# Converts stored metadata back to `ai:Metadata`, restoring `createdAt`, `modifiedAt`, `fileSize`.
#
# + metadata - The stored metadata map, or `()`
# + return - The metadata, `()` if `metadata` is `()`, or an `ai:Error` if a known field cannot be
# converted
isolated function createAiMetadata(map<json>? metadata) returns ai:Metadata?|ai:Error {
    if metadata is () {
        return ();
    }
    do {
        ai:Metadata result = {};
        foreach [string, json] [key, value] in metadata.entries() {
            if (key == "createdAt" || key == "modifiedAt") && value is string {
                result[key] = check time:utcFromString(value);
            } else if key == "fileSize" {
                result[key] = check value.cloneWithType(decimal);
            } else {
                result[key] = value;
            }
        }
        return result;
    } on fail error e {
        return error(string `Failed to convert stored metadata to 'ai:Metadata': ${e.message()}`, e);
    }
}

// OpenSearch scores `cosinesimil` as `(1 + cos) / 2`; this inverts it.
# Converts a cosine `_score` to a cosine similarity in `[-1, 1]` when asked to.
#
# + score - The raw `_score` returned by OpenSearch
# + metric - The index's configured similarity metric
# + normalize - `DenseSearch.normalizeCosineScore`
# + return - The cosine under `COSINE` with `normalize` set, otherwise `score` unchanged
isolated function normalizeScore(float score, ai:SimilarityMetric metric, boolean normalize) returns float {
    if !normalize || metric != ai:COSINE {
        return score;
    }
    return 2.0 * score - 1.0;
}

// Without a fusion processor, OpenSearch's negative sub-query delimiter scores come back as
// results. Every supported fusion technique gives non-negative scores.
# Rejects a negative `hybrid` score, which means no fusion processor ran.
#
# + score - The `_score` OpenSearch returned for a hit
# + fusion - Where the store's fusion configuration comes from, which decides what to blame
# + id - The entry id, for naming the offending hit
# + return - An `ai:Error` if the score is a leaked sentinel, otherwise `()`
isolated function checkFusedScore(float score, HybridFusion fusion, string id) returns ai:Error? {
    if score >= 0.0 {
        return;
    }
    if fusion is NamedSearchPipeline {
        return error(string `The hybrid search pipeline '${fusion.name}' returned an unfused ` +
                string `score (${score}) for entry '${id}'. A 'hybrid' query whose pipeline ` +
                "carries no 'normalization-processor' or 'score-ranker-processor' does not fail — " +
                "OpenSearch's negative sub-query delimiter scores survive into the results instead. " +
                string `Check that '${fusion.name}' exists on the cluster and declares one of those ` +
                "processors under 'phase_results_processors', or set a 'HybridSearchConfig' to have " +
                "this module send the pipeline inline");
    }
    return error(string `The hybrid query returned an unfused score (${score}) for entry ` +
            string `'${id}', meaning the inline fusion processor did not run. This module sends ` +
            "one with every hybrid query, so the cluster rejected or ignored it: confirm the " +
            "cluster runs OpenSearch 2.11 or later with the 'neural-search' plugin, and that it " +
            "honors an inline 'search_pipeline' in the search body");
}

# Converts a search hit to an `ai:VectorMatch`, restoring the stored chunk type.
#
# + hit - The search hit
# + searchMode - The store's search mode, which decides the shape of the returned embedding
# + config - The vector store configuration
# + scoreIsMeaningful - `false` for queries with no embedding, whose constant score is reported
# as 0.0
# + return - The match, or an `ai:Error` if stored data is malformed or a hybrid score is unfused
isolated function hitToVectorMatch(SearchHit hit, SearchMode searchMode, Configuration config,
        boolean scoreIsMeaningful) returns ai:VectorMatch|ai:Error {
    map<json> src = hit?._source ?: {};

    json idValue = src[config.idFieldName];
    string id = idValue is string ? idValue : hit._id;

    ai:Embedding embedding = check readStoredEmbedding(src, searchMode, id);

    string content = "";
    json contentValue = src[config.contentFieldName];
    if contentValue is string {
        content = contentValue;
    }

    string chunkType = "text-chunk";
    json chunkTypeValue = src[CHUNK_TYPE_FIELD];
    if chunkTypeValue is string {
        chunkType = chunkTypeValue;
    }

    map<json>? rawMetadata = extractStoredMetadata(src, searchMode, config);
    ai:Metadata? metadata = rawMetadata is () ? () : check createAiMetadata(rawMetadata);

    ai:Chunk chunk;
    if chunkType == "text-chunk" {
        chunk = <ai:TextChunk>{content, metadata};
    } else {
        chunk = {'type: chunkType, content, metadata};
    }

    // Only dense scores are converted; the transform would corrupt sparse and fused scores.
    float similarityScore = 0.0;
    if scoreIsMeaningful {
        if searchMode is HybridSearch {
            check checkFusedScore(hit._score, searchMode.fusion, id);
        }
        similarityScore = searchMode is DenseSearch
            ? normalizeScore(hit._score, searchMode.indexConfig.similarityMetric, searchMode.normalizeCosineScore)
            : hit._score;
    }
    return {
        id,
        embedding,
        chunk,
        similarityScore
    };
}

# Reads the stored embedding in the mode's shape. An absent or excluded field reads as empty.
#
# + src - The stored `_source` document
# + searchMode - The store's search mode
# + id - The entry id, used to name the offending entry in an error message
# + return - The embedding, or an `ai:Error` if what is stored cannot be parsed
isolated function readStoredEmbedding(map<json> src, SearchMode searchMode, string id)
        returns ai:Embedding|ai:Error {
    if searchMode is SparseSearch {
        return readStoredSparse(src, searchMode.sparseVectorFieldName, id);
    }
    if searchMode is HybridSearch {
        return {
            dense: check readStoredDense(src, searchMode.vectorFieldName, id),
            sparse: check readStoredSparse(src, searchMode.sparseVectorFieldName, id)
        };
    }
    return readStoredDense(src, searchMode.vectorFieldName, id);
}

# Reads the stored dense vector, or `[]` when the field is absent or excluded.
#
# + src - The stored `_source` document
# + vectorFieldName - The `knn_vector` field
# + id - The entry id, used to name the offending entry in an error message
# + return - The vector, or an `ai:Error` if what is stored is not an array of numbers
isolated function readStoredDense(map<json> src, string vectorFieldName, string id)
        returns ai:Vector|ai:Error {
    json vectorValue = src[vectorFieldName];
    if vectorValue !is json[] {
        return [];
    }
    ai:Vector|error converted = vectorValue.cloneWithType();
    if converted is error {
        return error(string `Failed to parse the stored embedding for entry '${id}'`, converted);
    }
    return converted;
}

# Reads the stored sparse vector, or an empty one when the field is absent or excluded.
#
# + src - The stored `_source` document
# + sparseVectorFieldName - The `rank_features` field
# + id - The entry id, used to name the offending entry in an error message
# + return - The sparse vector, or an `ai:Error` if a stored token is not an integer index
isolated function readStoredSparse(map<json> src, string sparseVectorFieldName, string id)
        returns ai:SparseVector|ai:Error {
    json sparseValue = src[sparseVectorFieldName];
    if sparseValue !is map<json> {
        return {indices: [], values: []};
    }
    return sparseFromRankFeatures(sparseValue, id);
}

# Extracts metadata from a stored document. Under a flat schema it is every non-reserved field.
#
# + src - The stored `_source` document
# + searchMode - The store's search mode, which decides which vector field names are reserved
# + config - The vector store configuration
# + return - The metadata map, or `()` if there is none
isolated function extractStoredMetadata(map<json> src, SearchMode searchMode, Configuration config)
        returns map<json>? {
    if config.metadataFieldName == "" {
        string[] vectorFieldNames = vectorFieldNamesOf(searchMode);
        map<json> flat = {};
        foreach [string, json] [key, value] in src.entries() {
            if vectorFieldNames.indexOf(key) is () && key != config.contentFieldName &&
                    key != config.idFieldName && key != CHUNK_TYPE_FIELD {
                flat[key] = value;
            }
        }
        return flat.length() == 0 ? () : flat;
    }
    json metadataValue = src[config.metadataFieldName];
    return metadataValue is map<json> ? metadataValue : ();
}

// Error responses use `root_cause`; `_bulk` item errors use only `caused_by`.
# The nested causes of an error, `root_cause[0]` before `caused_by`.
#
# + detail - The error object to inspect
# + return - The nested causes, `root_cause[0]` before `caused_by`; empty if it wraps nothing
isolated function causeCandidates(ErrorDetail detail) returns ErrorDetail[] {
    ErrorDetail[] candidates = [];
    ErrorDetail[]? rootCause = detail?.root_cause;
    if rootCause is ErrorDetail[] && rootCause.length() > 0 {
        candidates.push(rootCause[0]);
    }
    ErrorDetail? cause = detail?.caused_by;
    if cause is ErrorDetail {
        candidates.push(cause);
    }
    return candidates;
}

// A wrapper's reason is often generic ("all shards failed") with the explanation nested. Errors
// that wrap nothing repeat their reason in `root_cause[0]`, so an identical one is not appended.
# The error's reason, followed by its most specific nested reason when that adds something.
#
# + detail - The error object to describe
# + return - The description, or `""` if the error has neither a reason nor a type
isolated function describeErrorDetail(ErrorDetail detail) returns string {
    string outerReason = detail.reason ?: (detail.'type ?: "");
    ErrorDetail[] candidates = causeCandidates(detail);
    foreach ErrorDetail candidate in candidates {
        ErrorDetail deepest = deepestCause(candidate);
        string reason = deepest.reason ?: "";
        if reason.length() == 0 || reason == outerReason {
            continue;
        }
        string causeType = deepest.'type ?: "";
        return causeType.length() > 0
            ? string `${outerReason}: [${causeType}] ${reason}`
            : string `${outerReason}: ${reason}`;
    }
    return outerReason;
}

# Follows the `caused_by` chain to its end.
#
# + detail - The error object to walk from
# + return - The deepest cause, or `detail` itself when it wraps nothing
isolated function deepestCause(ErrorDetail detail) returns ErrorDetail {
    ErrorDetail current = detail;
    ErrorDetail? next = current?.caused_by;
    while next is ErrorDetail {
        current = next;
        next = current?.caused_by;
    }
    return current;
}

# The nested cause's type when it differs from the wrapper's, e.g. under
# `search_phase_execution_exception`.
#
# + detail - The error object
# + return - The underlying cause's type when it differs from the wrapper's own, otherwise `()`
isolated function specificErrorType(ErrorDetail detail) returns string? {
    string outerType = detail.'type ?: "";
    ErrorDetail[] candidates = causeCandidates(detail);
    foreach ErrorDetail candidate in candidates {
        string causeType = deepestCause(candidate).'type ?: "";
        if causeType.length() > 0 && causeType != outerType {
            return causeType;
        }
    }
    return ();
}

# The caller's id at a failed item's position, falling back to the response `_id`.
#
# + result - The failing item's result
# + submittedIds - The submitted ids, in submission order
# + position - The item's index within `BulkResponse.items`
# + return - The submitted id at `position`, falling back to the response `_id`
isolated function bulkFailureId(BulkItemResult result, string[] submittedIds, int position) returns string {
    if position < submittedIds.length() {
        return submittedIds[position];
    }
    return result?._id ?: "<unknown>";
}

# A failed `_bulk` item's reason, or `unknown error`.
#
# + err - The failing item's error object
# + return - The described reason, or `unknown error`
isolated function bulkFailureReason(ErrorDetail err) returns string {
    string reason = describeErrorDetail(err);
    return reason.length() > 0 ? reason : "unknown error";
}

// Matched by position: on Serverless Classic the response `_id` is server-generated and means
// nothing to the caller.
# The failed items of a `_bulk` indexing response, which is HTTP 200 even when items fail.
#
# + response - The parsed `_bulk` response
# + submittedIds - The logical ids of the submitted entries, in submission order
# + return - The failing ids and their reasons; empty if every item succeeded
isolated function extractIndexFailures(BulkResponse response, string[] submittedIds) returns BulkFailure[] {
    if !response.errors {
        return [];
    }
    BulkFailure[] failures = [];
    foreach int position in 0 ..< response.items.length() {
        foreach [string, BulkItemResult] [_, result] in response.items[position].entries() {
            ErrorDetail? err = result?.'error;
            if err is ErrorDetail {
                failures.push({
                    id: bulkFailureId(result, submittedIds, position),
                    reason: bulkFailureReason(err)
                });
            }
        }
    }
    return failures;
}

// Matched by position, since the Classic delete path submits internal `_id`s.
# The failed items of a `_bulk` delete response, ignoring `not_found`.
#
# + response - The parsed `_bulk` response
# + submittedIds - The logical ids behind the submitted delete actions, in submission order
# + return - The failing ids and their reasons; `not_found` items are not included
isolated function extractDeleteFailures(BulkResponse response, string[] submittedIds) returns BulkFailure[] {
    BulkFailure[] failures = [];
    foreach int position in 0 ..< response.items.length() {
        foreach [string, BulkItemResult] [_, result] in response.items[position].entries() {
            ErrorDetail? err = result?.'error;
            if err is ErrorDetail {
                failures.push({
                    id: bulkFailureId(result, submittedIds, position),
                    reason: bulkFailureReason(err)
                });
                continue;
            }
            int status = result?.status ?: 200;
            string? outcome = result?.result;
            if status == 404 || outcome == "not_found" {
                continue;
            }
            if status >= 300 {
                failures.push({
                    id: bulkFailureId(result, submittedIds, position),
                    reason: string `HTTP ${status}`
                });
            }
        }
    }
    return failures;
}

# A single failing item from a `_bulk` response.
type BulkFailure record {|
    # The caller's id for the failed entry.
    string id;
    # The reason, including the nested cause. See `describeErrorDetail`.
    string reason;
|};

# Renders bulk failures as one message, naming at most `limit` of them.
#
# + failures - The failures to render
# + limit - The maximum number of failures to name individually
# + return - A human-readable summary
isolated function summarizeBulkFailures(BulkFailure[] failures, int 'limit = 10) returns string {
    int shown = int:min(failures.length(), 'limit);
    string[] parts = [];
    foreach int i in 0 ..< shown {
        BulkFailure f = failures[i];
        parts.push(string `'${f.id}': ${f.reason}`);
    }
    string suffix = failures.length() > shown ? string ` (and ${failures.length() - shown} more)` : "";
    return string:'join("; ", ...parts) + suffix;
}
