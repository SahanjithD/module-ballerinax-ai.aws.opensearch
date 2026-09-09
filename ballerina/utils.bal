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

# The `IndexConfig` this `SearchMode` declares, or `()` under `SPARSE`, which maps no
# `knn_vector` field and so has no dimension, space type or HNSW tuning to describe.
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

# Every document field this `SearchMode` stores a vector in — one name under `DENSE`/`SPARSE`, two
# under `HYBRID`. Used to build the `_source` exclusion list when
# `Configuration.includeEmbeddingsInResults` is `false`, and to keep vector fields out of the
# metadata collected under a flat schema.
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

# A prepared vector entry ready to be serialized into a `_bulk` action/source pair: the embedding
# is confirmed to match the store's `SearchMode`, the sparse half is already converted to its
# `rank_features` wire form, and the id is resolved (generated if the caller omitted one).
#
# Which of the two vector fields is populated is decided by the mode, and the invariant is
# established at the single construction site in `prepareEntries`: `DENSE` sets `embedding` alone,
# `SPARSE` sets `rankFeatures` alone, and `HYBRID` sets both.
type PreparedEntry record {|
    # The resolved logical id.
    string id;
    # The confirmed-dense embedding. `()` under `SPARSE`, which stores no dense vector.
    ai:Vector? embedding = ();
    # The sparse embedding in its `rank_features` document form, `{"<index>": <weight>}`. `()`
    # under `DENSE`, which stores no sparse vector.
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

# Builds the error raised when an entry's embedding is not the `ai:Embedding` member this store's
# `SearchMode` requires.
#
# The check is deliberately symmetric between `add` and `query`: a store configured for one mode
# and handed another mode's embedding is a configuration mistake either way, and letting a sparse
# embedding through to a dense index would simply index a field that no query ever reads.
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

# Validates and normalizes a batch of vector entries prior to indexing.
#
# + entries - The caller-supplied entries
# + searchMode - The store's search mode, which decides which `ai:Embedding` member is required
# and supplies the similarity metric the zero-vector guard needs
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
            // `float:Infinity` as the ceiling: the `(0, 64]` bound is a query-time rule only, and
            // a stored `rank_features` weight has no documented maximum.
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
# + searchMode - The store's search mode, which decides which vector field(s) are written
# + config - The vector store configuration
# + return - The document to index, or an `ai:Error` if a flat-schema metadata key collides with
# a reserved field name
isolated function buildEntrySource(PreparedEntry entry, SearchMode searchMode, Configuration config)
        returns map<json>|ai:Error {
    anydata content = entry.chunk.content;
    map<json> sourceDoc = {
        [config.contentFieldName]: content is string ? content : content.toString(),
        [config.idFieldName]: entry.id,
        [CHUNK_TYPE_FIELD]: entry.chunk.'type
    };
    // Written before the flat-schema metadata loop below, so the reserved-name collision guard
    // there covers the vector fields too.
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

# Builds the NDJSON body of a `POST /_bulk` indexing request for a batch of prepared entries.
# Serverless Classic collections reject a caller-supplied `_id` on write, so the action
# line omits `_id` there; every other deployment type writes `_id` for a free upsert.
#
# + entries - The prepared entries to index, already sized to fit one `_bulk` request
# + indexName - The target index
# + deployment - The target deployment, which gates whether `_id` is written
# + searchMode - The store's search mode, passed through to the document builder
# + config - The vector store configuration
# + return - The exact bytes to sign and send. Empty entries produce an empty byte array. An
# `ai:Error` if a flat-schema metadata key collides with a reserved field name (see
# `buildEntrySource`)
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
        // `idFieldName` rather than `false`: the delete that follows submits internal `_id`s, so
        // without the logical id travelling back alongside each hit, a per-item delete failure
        // could only be reported against an internal id the caller has never seen. One keyword
        // field per hit is a negligible cost next to the `_id`s already being returned.
        "_source": [idFieldName],
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
    // Set only when the query actually produced a clause the pipeline has something to do to. A
    // query with no embedding scores nothing, so there is nothing to fuse under `HYBRID` and no
    // `neural_sparse` clause to accelerate under `SPARSE`; `searchPipelineName` keeps the named
    // variant in step with that same rule.
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
        // Only the inline variants contribute a body-level `search_pipeline`; a
        // `NamedSearchPipeline` travels as a query parameter instead, and sending both is an
        // error OpenSearch raises outright.
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

# Builds the `hybrid` clause: a `knn` sub-query and a `neural_sparse` sub-query, scored separately
# and then normalized and combined by the fusion pipeline.
#
# `hybrid` must be the top-level query. OpenSearch rejects it inside `bool`, `function_score`,
# `constant_score`, `script_score` or `boosting`, so this clause is returned for `query` directly
# and is never wrapped.
#
# # Why the filter is duplicated
# A top-level `hybrid.filter` is OpenSearch 3.0+. AWS still offers 2.x domains and the container
# suite runs 2.19.1, so the filter goes into each sub-query instead — documented as equivalent,
# and working on every version that has the `hybrid` query at all. The two shapes differ: `knn`
# takes a `filter` inside the vector object, while `neural_sparse` needs the `bool` sibling.
#
# + embedding - The query hybrid vector
# + searchMode - The hybrid search mode, supplying both field names and the token ceiling
# + size - The `k` to request from the dense sub-query
# + filterClause - The translated metadata filter, or `()`
# + return - The `hybrid` clause, or an `ai:Error` if the sparse half is invalid
isolated function buildHybridQueryClause(ai:HybridVector embedding, HybridSearch searchMode, int size,
        json? filterClause) returns json|ai:Error {
    // Order is load-bearing: `buildInlineFusionPipeline` emits its weights positionally against
    // this array, dense first and sparse second.
    json[] subQueries = [
        buildDenseQueryClause(embedding.dense, searchMode.vectorFieldName, size, filterClause,
                searchMode.efSearch),
        check buildSparseQueryClause(embedding.sparse, searchMode.sparseVectorFieldName,
                searchMode.maxQueryTokens, filterClause)
    ];
    return {"hybrid": {"queries": subQueries}};
}

# Builds the inline `search_pipeline` object that normalizes and combines the two sub-query scores.
#
# Sent with every hybrid query rather than provisioned once as a named pipeline. Fusion weights are
# query-time semantics, and binding them to durable cluster state would make them silently stale
# the moment a caller changed one — the trap `IndexConfig` already documents for index-creation
# settings. It also keeps this module's IAM surface at index scope, since an AOSS search pipeline
# is a collection-scoped resource, and avoids AWS's documented up-to-15-second Serverless
# propagation delay during which a freshly created pipeline is not yet resolvable.
#
# A `hybrid` query without this does not fail cleanly: OpenSearch delimits each sub-query's results
# with sentinel scores of about `-9.5e9` and `-4.4e9`, and it is the fusion processor that strips
# them. Without one they reach the caller as results, which `checkFusedScore` is the last line of
# defence against.
#
# Both variants are `phase_results_processors` entries and differ only in which processor they name
# and what it is given. `RrfFusion` is sent as the bare technique, because neither knob the
# `score-ranker-processor` documents has a wire shape that behaves the same across the versions its
# 2.19 floor admits — see "Why this variant has no settings" on `RrfFusion`.
#
# + fusion - The fusion settings, score-based or rank-based
# + return - The inline pipeline object
isolated function buildInlineFusionPipeline(InlineFusion fusion) returns json {
    if fusion is RrfFusion {
        // Nothing beyond the technique. `rank_constant` and `weights` both have wire shapes that
        // are honored on one side of neural-search 3.1 and ignored or rejected on the other; see
        // "Why this variant has no settings" on `RrfFusion`. What is left is portable everywhere
        // the processor exists.
        return {
            "phase_results_processors": [
                {"score-ranker-processor": {"combination": {"technique": fusion.technique}}}
            ]
        };
    }
    // Positional against the sub-query array built in `buildHybridQueryClause`.
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

# Builds the inline `search_pipeline` object that accelerates a `neural_sparse` query by splitting
# its query tokens into a high-weight filtering pass and a low-weight rescoring pass.
#
# A `request_processors` entry rather than a `phase_results_processors` one: it rewrites the query
# before it runs, where the fusion processors rearrange results after. That is also why the two
# never need to be combined here — the processor does not descend into a `hybrid` query, so it is
# reachable from `SPARSE` alone.
#
# + acceleration - The two-phase settings
# + return - The inline pipeline object
isolated function buildInlineTwoPhasePipeline(TwoPhaseAcceleration acceleration) returns json => {
    "request_processors": [
        {
            "neural_sparse_two_phase_processor": {
                "enabled": true,
                // Singular, and deliberately so -- OpenSearch names this field
                // `two_phase_parameter`, not `two_phase_parameters`, and an unrecognised key here
                // fails the whole search rather than being ignored.
                // `expansion_rate` and `max_window_size` are deliberately absent rather than
                // sent at their defaults: omitting them lets the server own those values, so a
                // future OpenSearch that tunes them differently is not overridden by this module
                // restating today's numbers. See `TwoPhaseAcceleration`.
                "two_phase_parameter": {"prune_ratio": acceleration.pruneRatio}
            }
        }
    ]
};

# The named search pipeline to send as `?search_pipeline=`, or `()` when this store sends its
# fusion configuration inline in the request body instead.
#
# Never both: OpenSearch rejects a request carrying an inline pipeline object and a named pipeline
# parameter with "Both named and inline search pipeline were specified". The `HybridFusion` union
# is what makes sending both unrepresentable rather than merely unlikely.
#
# A query carrying no embedding produces no `hybrid` clause — it scores nothing, so there is
# nothing to normalize and no pipeline to run. The inline variant already omits its pipeline in
# that case; `hasEmbedding` is what keeps the named variant from disagreeing with it, rather than
# attaching a fusion pipeline to a `match_all`.
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

# Builds the `knn` clause for a dense query, pre-filtering inside the clause rather than with
# `post_filter`, which can silently return fewer than `k` results.
#
# `k` is the requested `size`, which makes `topK` double as the retrieval depth. That is the right
# answer under `DENSE`, where the `k` nearest are exactly the results; under `HYBRID` it is a
# ceiling on what fusion has to work with, and is documented as such on `HybridSearch`.
#
# `method_parameters` is emitted only when `efSearch` is set, so an unset dial leaves the request
# byte-identical to what this module sent before the knob existed — which also keeps it off
# clusters older than the 2.16 that first accepted the object.
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

# Builds the `neural_sparse` clause for a sparse query, wrapped in a `bool` when there is a filter.
#
# The filter is a **sibling** of the `neural_sparse` clause rather than a parameter of it: a
# `neural_sparse` query over a `rank_features` field has no `filter` parameter at all (only the
# `sparse_vector` ANN field type, which is OpenSearch 3.3+, accepts one under `method_parameters`).
# `bool.filter` clauses do not contribute to `_score`, so the sparse dot product remains the sole
# score.
#
# The token-count ceiling is checked here rather than left to the server. A `neural_sparse` query
# compiles to one Lucene clause per term, so a query vector with more terms than the cluster's
# `indices.query.bool.max_clause_count` fails as a `too_many_clauses` shard exception that names
# neither the limit that was hit nor the query that hit it.
#
# + embedding - The query sparse vector
# + sparseVectorFieldName - The `rank_features` field to search
# + maxQueryTokens - The client-side ceiling on the number of non-zero terms
# + filterClause - The translated metadata filter, or `()`
# + return - The query clause, or an `ai:Error` if the vector is invalid or too wide
isolated function buildSparseQueryClause(ai:SparseVector embedding, string sparseVectorFieldName,
        int maxQueryTokens, json? filterClause) returns json|ai:Error {
    int tokenCount = embedding.indices.length();
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
# A custom metadata key holding a fractional number does not round-trip its Ballerina type: it is
# written as a `float` and read back as a `decimal`, because JSON has a single number type and
# Ballerina's parser maps every non-integral value to `decimal`. The value is preserved exactly;
# only the basic type differs, so `readBack["rating"] == 4.25` is `false` after a round trip.
# This is deliberate, and matches `ai.pinecone`'s identical conversion. Coercing back to `float`
# would silently lose precision for a caller who passed a `decimal` on purpose, which is the worse
# failure — `fileSize` is restored explicitly only because `ai:Metadata` declares it `decimal`.
#
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

# Rejects a `hybrid` `_score` that no fusion processor ever touched.
#
# OpenSearch delimits each sub-query's results inside a `hybrid` query with large negative sentinel
# scores (around `-9.5e9` and `-4.4e9`), and it is the fusion phase-results processor that strips
# them and replaces every score with a normalized one. A `hybrid` query that runs without such a
# processor does **not** fail: the sentinels survive the fetch phase and reach the caller as
# `_score` values, so a corrupted ranking is returned as though it were a real one. The container
# suite characterises exactly this — see `testContainerHybridWithoutPipelineLeaksSentinelScores`.
#
# Every fusion technique this module can ask for produces a non-negative score: `min_max` and `l2`
# both emit `[0.0, 1.0]` with an exact `0.0` replaced by `0.001`, and RRF sums reciprocals of
# ranks. A negative score under `HYBRID` therefore means no processor ran, and is never a value a
# caller should be handed.
#
# The inline variant cannot reach this — this module always sends a processor with the query — but
# it is checked there too, since "the processor did not run" is a server-side outcome rather than
# a request-shape one, and the check costs nothing.
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

# Converts a single OpenSearch search hit to an `ai:VectorMatch`. The stored `chunk_type` is read
# back so the returned chunk round-trips as the same `ai:Chunk` shape it was written as: a
# `"text-chunk"` value reconstructs as `ai:TextChunk` (so `cloneWithType(ai:TextChunk)` on the
# result is not a lie), anything else reconstructs as a plain `ai:Chunk` carrying that `'type`.
#
# + hit - The search hit
# + searchMode - The store's search mode, which decides the shape of the returned embedding
# + config - The vector store configuration
# + scoreIsMeaningful - Whether `hit._score` reflects an actual similarity computation. `false`
# for the two query shapes with no embedding (`match_all` / a bare `bool` filter), where
# OpenSearch returns a constant score that is not a similarity — the result is `0.0` regardless
# of `hit._score`, matching `ai:InMemoryVectorStore` and this module's own documented behavior
# + return - The converted match, or an `ai:Error` if the stored embedding or metadata is malformed,
# or if a `HYBRID` score shows that no fusion processor ran — see `checkFusedScore`
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

    // `normalizeScore` is reached only through `DenseSearch`, which is the only mode that declares
    // a `similarityMetric` and a `normalizeCosineScore` at all. A `SPARSE` score is an unbounded
    // dot product with no cosine in it, and a `HYBRID` score has already been normalized to
    // `(0.0, 1.0]` by the fusion pipeline; applying `2 * score - 1` to either would corrupt it.
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

# Reads whichever `ai:Embedding` member this store's `SearchMode` stores back out of a `_source`
# document.
#
# An absent or excluded field yields that member's empty value rather than an error, since
# `Configuration.includeEmbeddingsInResults = false` legitimately omits it and
# `ai:VectorMatch.embedding` is not optional.
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

# Extracts the metadata portion of a stored `_source` document, honoring
# `Configuration.metadataFieldName == ""` by collecting every field that is not one of the fixed
# schema fields (vector, content, id, chunk type).
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

# Gathers the places an `ErrorDetail` can nest its underlying cause, most-authoritative first.
#
# The two are not alternatives so much as different dialects: an error *response* wraps its causes
# in `root_cause`, while a `_bulk` item error has no `root_cause` and uses `caused_by` alone. Both
# are collected so callers can walk them in one pass without caring which shape they were handed.
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

# Recovers the most specific explanation an `ErrorDetail` carries. The reason at its top level is
# routinely a wrapper's rather than the failure's, so reporting that alone tells a caller nothing
# about what they did wrong.
#
# Shaped against OpenSearch 2.19.1 response bodies. A shard-level search failure — what a
# wrong-dimension query vector produces — reports `all shards failed` at the top and puts the
# explanation in `root_cause[0].reason`
# (`failed to create query: Query vector has invalid dimension: 4. Dimension should be: 8`). A
# failed `_bulk` item carries no `root_cause` at all: it reports
# `failed to parse field [embedding] of type [knn_vector] ... Preview of field's value: 'null'`
# at the top and puts the explanation in `caused_by.reason`
# (`Vector dimension mismatch. Expected: 8, Given: 2`). Both nestings are therefore searched.
#
# Errors that wrap nothing — `index_not_found_exception`, `parsing_exception` — repeat their own
# reason verbatim in `root_cause[0]`, so a nested reason is appended only when it differs from the
# one already being reported, and the common case stays a single unduplicated sentence.
#
# + detail - The error object to describe
# + return - The top-level reason, followed by the nested cause when that adds something; `""` if
# the error object carries neither a reason nor a type
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

# Follows an `ErrorDetail`'s `caused_by` chain to its end. Terminates because the chain is decoded
# from a JSON response body, which is a finite tree.
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

# Resolves the error classification worth branching on. An error object's own `type` names the
# wrapper when there is one — a failed search reports `search_phase_execution_exception`, which
# says only that the query phase failed and nothing about why.
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

# Resolves which id to report for a failing `_bulk` item, preferring the caller's own id at that
# position over the `_id` the server reports back.
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

# Renders a failing `_bulk` item's reason, falling back when the error object says nothing.
#
# + err - The failing item's error object
# + return - The described reason, or `unknown error`
isolated function bulkFailureReason(ErrorDetail err) returns string {
    string reason = describeErrorDetail(err);
    return reason.length() > 0 ? reason : "unknown error";
}

# Extracts the per-item failures from a `_bulk` indexing response. OpenSearch returns HTTP
# 200 even when some items failed, so `errors: true` must always be checked explicitly.
#
# # Failures are attributed by position, not by `_id`
# `_bulk` returns its items in submission order, so the item at position `i` is the entry at
# `submittedIds[i]`. That correspondence is the only reliable way back to the caller's entry: the
# `_id` in the response is the caller's id only where a custom `_id` can be written at all. On
# `SERVERLESS_CLASSIC` the action line deliberately carries no `_id` (see `buildAddBulkBody`) and
# the server invents one, so reporting it would name the caller's failing entries as strings like
# `1%3A0%3AjNUSV6ABrlsmLW-dso53` — leaving someone who submitted 500 entries and got 3 failures
# with no way to tell which 3 were theirs.
#
# The response `_id` is used only for a position beyond the end of `submittedIds`, which would
# mean the server returned more items than were submitted and the positional correspondence is
# already broken.
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

# Extracts the per-item failures from a `_bulk` delete response, ignoring `not_found` items — a
# distributed delete of an id that no longer exists (or never did) is not treated as a caller
# error, matching the Pinecone/pgvector/weaviate precedent rather than `InMemoryVectorStore`'s.
#
# Failures are attributed by position for the same reason as in `extractIndexFailures`, and it
# matters here even on deployments that write a custom `_id`: the `SERVERLESS_CLASSIC` delete path
# submits the *internal* `_id`s a lookup discovered, so `submittedIds` is what carries the
# caller's logical ids back into the message.
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
    # The failing entry's logical id as the caller supplied it, recovered from the item's position
    # in the response rather than from the `_id` the server reported.
    string id;
    # The failure reason: OpenSearch's `error.reason`, extended with the nested cause that
    # actually explains it. See `describeErrorDetail`.
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
