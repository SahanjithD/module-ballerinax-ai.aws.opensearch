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

// Every function in this file is pure (no network I/O), so the whole request/response shaping
// logic is unit-testable without a live OpenSearch endpoint.

# The document field carrying `ai:Chunk.'type`, so a stored entry's chunk type round-trips
# through a query instead of every match being reconstructed as a hardcoded `ai:TextChunk`.
const string CHUNK_TYPE_FIELD = "chunk_type";

# A prepared vector entry ready to be serialized into a `_bulk` action/source pair: the embedding
# is confirmed dense and non-zero, and the id is resolved (generated if the caller omitted one).
type PreparedEntry record {|
    # The resolved logical id.
    string id;
    # The confirmed-dense embedding.
    ai:Vector embedding;
    # The source chunk.
    ai:Chunk chunk;
|};

# Validates and normalizes a batch of vector entries prior to indexing.
#
# + entries - The caller-supplied entries
# + metric - The index's configured similarity metric, used for the zero-vector guard
# + return - The prepared entries, or an `ai:Error` naming the offending entry
isolated function prepareEntries(ai:VectorEntry[] entries, ai:SimilarityMetric metric) returns PreparedEntry[]|ai:Error {
    PreparedEntry[] prepared = [];
    foreach ai:VectorEntry entry in entries {
        ai:Embedding embedding = entry.embedding;
        if embedding !is ai:Vector {
            return error(
                    "OpenSearch vector store currently supports dense vectors only; got a sparse/hybrid embedding");
        }
        string id = entry.id ?: uuid:createRandomUuid();
        check validateEmbedding(embedding, metric, id);
        prepared.push({id, embedding, chunk: entry.chunk});
    }
    return prepared;
}

# Splits a prepared-entry batch into groups of at most `chunkSize`, for chunking the `add`
# operation into `_bulk` requests of at most `Configuration.maxBulkSize` entries.
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

# Splits an id list into groups of at most `chunkSize`, for chunking a `SERVERLESS_CLASSIC`
# delete's `terms` lookup below OpenSearch's 65,536-term cap.
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

# Rejects a zero-magnitude vector under cosine similarity before it reaches the server, where it
# would otherwise fail the entire `_bulk` request with an opaque exception.
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

# Builds the `_source` document for a single prepared entry, per the schema in the module design:
# the vector, the content, the logical id (`doc_id` by default), the chunk type, and metadata
# either nested under `metadataFieldName` or spread as top-level fields when it is `""`.
#
# Under a flat schema (`metadataFieldName == ""`), a metadata key that collides with one of the
# fixed schema field names would otherwise silently overwrite that field (e.g. a metadata key
# literally named `"content"` clobbering the chunk content) — `ai:Metadata` is an open record, so
# a caller can legally supply any key. This is rejected with an `ai:Error` naming the offending
# entry and key instead.
#
# + entry - The prepared entry
# + config - The vector store configuration
# + return - The document to index, or an `ai:Error` if a flat-schema metadata key collides with
# a reserved field name
isolated function buildEntrySource(PreparedEntry entry, Configuration config) returns map<json>|ai:Error {
    anydata content = entry.chunk.content;
    map<json> sourceDoc = {
        [config.vectorFieldName]: entry.embedding.cloneReadOnly(),
        [config.contentFieldName]: content is string ? content : content.toString(),
        [config.idFieldName]: entry.id,
        [CHUNK_TYPE_FIELD]: entry.chunk.'type
    };
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

# Builds the NDJSON body of a `POST /_bulk` indexing request for a batch of prepared entries.
# Serverless Classic collections reject a caller-supplied `_id` on write, so the action
# line omits `_id` there; every other deployment type writes `_id` for a free upsert.
#
# + entries - The prepared entries to index, already sized to fit one `_bulk` request
# + indexName - The target index
# + deploymentType - The deployment flavour, which gates whether `_id` is written
# + config - The vector store configuration
# + return - The exact bytes to sign and send. Empty entries produce an empty byte array. An
# `ai:Error` if a flat-schema metadata key collides with a reserved field name (see
# `buildEntrySource`)
isolated function buildAddBulkBody(PreparedEntry[] entries, string indexName, DeploymentType deploymentType,
        Configuration config) returns byte[]|ai:Error {
    if entries.length() == 0 {
        return [];
    }
    string[] lines = [];
    foreach PreparedEntry entry in entries {
        json action = deploymentType == SERVERLESS_CLASSIC
            ? {"index": {"_index": indexName}}
            : {"index": {"_index": indexName, "_id": entry.id}};
        lines.push(action.toJsonString());
        lines.push((check buildEntrySource(entry, config)).toJsonString());
    }
    return (string:'join("\n", ...lines) + "\n").toBytes();
}

# Builds the NDJSON body of a `POST /_bulk` delete request for a set of `_id`s. Used directly on
# `MANAGED_DOMAIN` / `SERVERLESS_NEXTGEN` (where the logical id is the `_id`), and as the second
# phase of a `SERVERLESS_CLASSIC` delete, against the internal `_id`s discovered by a prior search.
#
# + indexName - The target index
# + ids - The `_id`s to delete
# + return - The exact bytes to sign and send. Empty `ids` produce an empty byte array
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

# Builds a `POST /<index>/_search` body that looks up the internal `_id`s of documents whose
# `idFieldName` matches one of `ids`. This is phase one of a `SERVERLESS_CLASSIC` delete, since a
# custom `_id` cannot be written there and `_delete_by_query` is not on the AOSS whitelist.
#
# + idFieldName - The document field carrying the logical id (`Configuration.idFieldName`)
# + ids - The logical ids to look up. Must be chunked by the caller to at most 65,536 terms
# + maxResultWindow - The `size` to request, sized generously since one logical id may map to
# several documents on an append-only store
# + return - The search request body
isolated function buildDocIdLookupBody(string idFieldName, string[] ids, int maxResultWindow) returns json {
    return {
        "size": maxResultWindow,
        "_source": false,
        "track_total_hits": false,
        "query": {
            "terms": {
                [idFieldName]: ids
            }
        }
    };
}

# Builds a `POST /<index>/_search` body for an `ai:VectorStoreQuery`, covering all four
# combinations of `(embedding, filters)`. Pre-filters inside the `knn` clause rather than using
# `post_filter`, since `post_filter` can silently return fewer than `k` results.
#
# + query - The vector store query
# + config - The vector store configuration
# + return - The search request body, or an `ai:Error` if `topK` or the filters are invalid
isolated function buildSearchBody(ai:VectorStoreQuery query, Configuration config) returns json|ai:Error {
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
        : {"excludes": [config.vectorFieldName]};

    json queryClause;
    if embedding is () {
        queryClause = filterClause is () ? {"match_all": {}} : {"bool": {"filter": [filterClause]}};
    } else {
        if embedding !is ai:Vector {
            return error(
                    "OpenSearch vector store currently supports dense vectors only; got a sparse/hybrid embedding");
        }
        map<json> knnBody = {"vector": embedding.cloneReadOnly(), "k": size};
        if filterClause !is () {
            knnBody["filter"] = filterClause;
        }
        queryClause = {"knn": {[config.vectorFieldName]: knnBody}};
    }

    return {
        "size": size,
        "_source": sourceDirective,
        "track_total_hits": false,
        "query": queryClause
    };
}

# Recursively translates `ai:MetadataFilters`/`ai:MetadataFilter` into an OpenSearch `bool` query
# clause. Uses `bool.filter` for `AND` (unscored, cacheable) and `bool.should` +
# `minimum_should_match: 1` for `OR`. An empty filter list at any level of nesting contributes no
# clause at all, rather than an empty `{"bool":{"filter":[]}}`.
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

# Resolves the field path for a metadata key, honoring `Configuration.metadataFieldName == ""` as
# a request for bare `<key>` paths (for a pre-existing index with a flat schema).
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

# Converts `ai:Metadata` to its wire form. Only `createdAt`/`modifiedAt` receive special
# treatment (`time:Utc` → RFC 3339 string, matching the index mapping's `date` field); every other
# value passes through unchanged. Mirrors `ai.pinecone`'s key-allowlist approach rather than
# `ai.weaviate`'s blind `time:utcFromString` attempt on every string field.
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

# Converts a stored metadata map back to `ai:Metadata`. The inverse of `transformMetadata`:
# `createdAt`/`modifiedAt` strings are parsed back to `time:Utc`, `fileSize` is converted to
# `decimal` (JSON only round-trips `int`/`float`), and every other value passes through.
#
# + metadata - The stored metadata map, or `()`
# + return - The reconstructed `ai:Metadata`, `()` if `metadata` is `()`, or an `ai:Error` if a
# `createdAt`/`modifiedAt`/`fileSize` value cannot be converted
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

# Converts a `_score` back to a `[-1, 1]` cosine similarity when requested. OpenSearch's
# `cosinesimil` space scores as `(2 - d) / 2` where `d = 1 - cos`, i.e. `score = (1 + cos) / 2` —
# inverted here as `cos = 2 * score - 1`. This is distinct from the `l1`/`l2`/`linf` score formula
# `1 / (1 + d)`; do not conflate the two.
#
# + score - The raw `_score` returned by OpenSearch
# + metric - The index's configured similarity metric
# + normalize - `Configuration.normalizeCosineScore`
# + return - The normalized cosine similarity when `metric` is `COSINE` and `normalize` is `true`,
# otherwise `score` unchanged
isolated function normalizeScore(float score, ai:SimilarityMetric metric, boolean normalize) returns float {
    if !normalize || metric != ai:COSINE {
        return score;
    }
    return 2.0 * score - 1.0;
}

# Converts a single OpenSearch search hit to an `ai:VectorMatch`. The stored `chunk_type` is read
# back so the returned chunk round-trips as the same `ai:Chunk` shape it was written as: a
# `"text-chunk"` value reconstructs as `ai:TextChunk` (so `cloneWithType(ai:TextChunk)` on the
# result is not a lie), anything else reconstructs as a plain `ai:Chunk` carrying that `'type`.
#
# + hit - The search hit
# + config - The vector store configuration
# + scoreIsMeaningful - Whether `hit._score` reflects an actual similarity computation. `false`
# for the two query shapes with no embedding (`match_all` / a bare `bool` filter), where
# OpenSearch returns a constant score that is not a similarity — the result is `0.0` regardless
# of `hit._score`, matching `ai:InMemoryVectorStore` and this module's own documented behavior
# + return - The converted match, or an `ai:Error` if the stored embedding or metadata is malformed
isolated function hitToVectorMatch(SearchHit hit, Configuration config, boolean scoreIsMeaningful)
        returns ai:VectorMatch|ai:Error {
    map<json> src = hit?._source ?: {};

    json idValue = src[config.idFieldName];
    string id = idValue is string ? idValue : hit._id;

    ai:Vector embedding = [];
    json vectorValue = src[config.vectorFieldName];
    if vectorValue is json[] {
        ai:Vector|error converted = vectorValue.cloneWithType();
        if converted is error {
            return error(string `Failed to parse the stored embedding for entry '${id}'`, converted);
        }
        embedding = converted;
    }

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

    map<json>? rawMetadata = extractStoredMetadata(src, config);
    ai:Metadata? metadata = rawMetadata is () ? () : check createAiMetadata(rawMetadata);

    ai:Chunk chunk;
    if chunkType == "text-chunk" {
        chunk = <ai:TextChunk>{content, metadata};
    } else {
        chunk = {'type: chunkType, content, metadata};
    }

    float similarityScore = scoreIsMeaningful
        ? normalizeScore(hit._score, config.indexConfig.similarityMetric, config.normalizeCosineScore)
        : 0.0;
    return {
        id,
        embedding,
        chunk,
        similarityScore
    };
}

# Extracts the metadata portion of a stored `_source` document, honoring
# `Configuration.metadataFieldName == ""` by collecting every field that is not one of the fixed
# schema fields (vector, content, id, chunk type).
#
# + src - The stored `_source` document
# + config - The vector store configuration
# + return - The metadata map, or `()` if there is none
isolated function extractStoredMetadata(map<json> src, Configuration config) returns map<json>? {
    if config.metadataFieldName == "" {
        map<json> flat = {};
        foreach [string, json] [key, value] in src.entries() {
            if key != config.vectorFieldName && key != config.contentFieldName &&
                    key != config.idFieldName && key != CHUNK_TYPE_FIELD {
                flat[key] = value;
            }
        }
        return flat.length() == 0 ? () : flat;
    }
    json metadataValue = src[config.metadataFieldName];
    return metadataValue is map<json> ? metadataValue : ();
}

# Extracts the per-item failures from a `_bulk` indexing response. OpenSearch returns HTTP
# 200 even when some items failed, so `errors: true` must always be checked explicitly.
#
# + response - The parsed `_bulk` response
# + return - The failing ids and their reasons; empty if every item succeeded
isolated function extractIndexFailures(BulkResponse response) returns BulkFailure[] {
    if !response.errors {
        return [];
    }
    BulkFailure[] failures = [];
    foreach map<BulkItemResult> item in response.items {
        foreach [string, BulkItemResult] [_, result] in item.entries() {
            ErrorDetail? err = result?.'error;
            if err is ErrorDetail {
                failures.push({id: result?._id ?: "<unknown>", reason: err.reason ?: (err.'type ?: "unknown error")});
            }
        }
    }
    return failures;
}

# Extracts the per-item failures from a `_bulk` delete response, ignoring `not_found` items — a
# distributed delete of an id that no longer exists (or never did) is not treated as a caller
# error, matching the Pinecone/pgvector/weaviate precedent rather than `InMemoryVectorStore`'s.
#
# + response - The parsed `_bulk` response
# + return - The failing ids and their reasons; `not_found` items are not included
isolated function extractDeleteFailures(BulkResponse response) returns BulkFailure[] {
    BulkFailure[] failures = [];
    foreach map<BulkItemResult> item in response.items {
        foreach [string, BulkItemResult] [_, result] in item.entries() {
            ErrorDetail? err = result?.'error;
            if err is ErrorDetail {
                failures.push({id: result?._id ?: "<unknown>", reason: err.reason ?: (err.'type ?: "unknown error")});
                continue;
            }
            int status = result?.status ?: 200;
            string? outcome = result?.result;
            if status == 404 || outcome == "not_found" {
                continue;
            }
            if status >= 300 {
                failures.push({id: result?._id ?: "<unknown>", reason: string `HTTP ${status}`});
            }
        }
    }
    return failures;
}

# A single failing item from a `_bulk` response.
type BulkFailure record {|
    # The failing document's `_id`.
    string id;
    # The failure reason, from OpenSearch's `error.reason` (or `error.type` if `reason` is absent).
    string reason;
|};

# Renders a list of bulk failures into a single error message, capped at `limit` named entries.
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
