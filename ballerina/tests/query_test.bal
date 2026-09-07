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
import ballerina/test;

isolated function queryMode() returns DenseSearch => {queryMode: ai:DENSE, indexConfig: {dimension: 3}};

// `normalizeCosineScore` defaults to `true`, so a test that wants OpenSearch's raw `_score` has to
// ask for it. See `testHitToVectorMatchNormalizesCosineByDefault` for why that is the default.
isolated function rawScoreMode() returns DenseSearch =>
    {queryMode: ai:DENSE, indexConfig: {dimension: 3}, normalizeCosineScore: false};

isolated function asJsonMap(json value) returns map<json>|error => value.ensureType();

// --- the four (embedding, filters) combinations ------------------------------------------------

@test:Config
isolated function testQueryNeitherEmbeddingNorFilters() returns error? {
    ai:VectorStoreQuery query = {};
    json body = check buildSearchBody(query, queryMode(), {});
    map<json> bodyMap = check asJsonMap(body);
    map<json> queryClause = check asJsonMap(bodyMap["query"]);
    test:assertTrue(queryClause.hasKey("match_all"));
    test:assertEquals(bodyMap["size"], 10);
}

@test:Config
isolated function testQueryFiltersOnlyNoEmbedding() returns error? {
    ai:VectorStoreQuery query = {
        filters: {filters: [{key: "language", operator: ai:EQUAL, value: "en"}]}
    };
    json body = check buildSearchBody(query, queryMode(), {});
    map<json> bodyMap = check asJsonMap(body);
    map<json> queryClause = check asJsonMap(bodyMap["query"]);
    test:assertTrue(queryClause.hasKey("bool"), "expected a bool.filter query when only filters are given");
    test:assertFalse(queryClause.hasKey("knn"));
}

@test:Config
isolated function testQueryEmbeddingOnlyNoFilters() returns error? {
    ai:VectorStoreQuery query = {embedding: [0.1, 0.2, 0.3]};
    json body = check buildSearchBody(query, queryMode(), {});
    map<json> bodyMap = check asJsonMap(body);
    map<json> queryClause = check asJsonMap(bodyMap["query"]);
    map<json> knn = check asJsonMap(queryClause["knn"]);
    map<json> knnField = check asJsonMap(knn["embedding"]);
    test:assertEquals(knnField["vector"], <json[]>[0.1, 0.2, 0.3]);
    test:assertEquals(knnField["k"], 10);
    test:assertFalse(knnField.hasKey("filter"), "no pre-filter should be present when filters are absent");
}

@test:Config
isolated function testQueryEmbeddingAndFiltersPreFiltersInsideKnn() returns error? {
    ai:VectorStoreQuery query = {
        embedding: [0.1, 0.2, 0.3],
        filters: {filters: [{key: "language", operator: ai:EQUAL, value: "en"}]}
    };
    json body = check buildSearchBody(query, queryMode(), {});
    map<json> bodyMap = check asJsonMap(body);
    map<json> queryClause = check asJsonMap(bodyMap["query"]);
    map<json> knn = check asJsonMap(queryClause["knn"]);
    map<json> knnField = check asJsonMap(knn["embedding"]);
    test:assertTrue(knnField.hasKey("filter"), "the filter must be pre-filtered inside the knn clause, not post_filter");
    test:assertFalse(bodyMap.hasKey("post_filter"), "post_filter must never be used");
}

@test:Config
isolated function testQueryWithEmptyFiltersBehavesAsNoFilters() returns error? {
    // An ai:MetadataFilters with an empty list translates to no clause; the query must fall back
    // to match_all rather than emitting an empty {"bool":{"filter":[]}}.
    ai:VectorStoreQuery query = {filters: {filters: []}};
    json body = check buildSearchBody(query, queryMode(), {});
    map<json> bodyMap = check asJsonMap(body);
    map<json> queryClause = check asJsonMap(bodyMap["query"]);
    test:assertTrue(queryClause.hasKey("match_all"));
}

// --- topK handling -------------------------------------------------------------------------------

@test:Config
isolated function testTopKMinusOneMeansAll() returns error? {
    ai:VectorStoreQuery query = {topK: -1};
    json body = check buildSearchBody(query, queryMode(), {});
    map<json> bodyMap = check asJsonMap(body);
    test:assertEquals(bodyMap["size"], 10000);
}

@test:Config
isolated function testTopKZeroMeansAll() returns error? {
    // Matches ai:InMemoryVectorStore: any topK < 1 means "return all", not just -1.
    ai:VectorStoreQuery query = {topK: 0};
    json body = check buildSearchBody(query, queryMode(), {});
    map<json> bodyMap = check asJsonMap(body);
    test:assertEquals(bodyMap["size"], 10000);
}

@test:Config
isolated function testTopKNegativeBelowMinusOneMeansAll() returns error? {
    ai:VectorStoreQuery query = {topK: -5};
    json body = check buildSearchBody(query, queryMode(), {});
    map<json> bodyMap = check asJsonMap(body);
    test:assertEquals(bodyMap["size"], 10000);
}

@test:Config
isolated function testTopKPositiveIsHonored() returns error? {
    ai:VectorStoreQuery query = {topK: 5};
    json body = check buildSearchBody(query, queryMode(), {});
    map<json> bodyMap = check asJsonMap(body);
    test:assertEquals(bodyMap["size"], 5);
}

@test:Config
isolated function testTopKAboveMaxResultWindowIsError() {
    ai:VectorStoreQuery query = {topK: 20000};
    json|ai:Error result = buildSearchBody(query, queryMode(), {});
    test:assertTrue(result is ai:Error, "expected an ai:Error when topK exceeds maxResultWindow");
}

@test:Config
isolated function testCustomMaxResultWindowRaisesTheCeiling() returns error? {
    DenseSearch configMode = {queryMode: ai:DENSE, indexConfig: {dimension: 3}};
    Configuration config = {maxResultWindow: 50000};
    ai:VectorStoreQuery query = {topK: 20000};
    json body = check buildSearchBody(query, configMode, config);
    map<json> bodyMap = check asJsonMap(body);
    test:assertEquals(bodyMap["size"], 20000);
}

// --- _source directive -----------------------------------------------------------------------

@test:Config
isolated function testSourceIncludesEmbeddingsByDefault() returns error? {
    json body = check buildSearchBody({}, queryMode(), {});
    map<json> bodyMap = check asJsonMap(body);
    test:assertEquals(bodyMap["_source"], true);
}

@test:Config
isolated function testSourceExcludesEmbeddingsWhenConfigured() returns error? {
    DenseSearch configMode = {queryMode: ai:DENSE, indexConfig: {dimension: 3}};
    Configuration config = {includeEmbeddingsInResults: false};
    json body = check buildSearchBody({}, configMode, config);
    map<json> bodyMap = check asJsonMap(body);
    map<json> sourceDirective = check asJsonMap(bodyMap["_source"]);
    test:assertEquals(sourceDirective["excludes"], <json[]>["embedding"]);
}

@test:Config
isolated function testTrackTotalHitsAlwaysFalse() returns error? {
    json body = check buildSearchBody({}, queryMode(), {});
    map<json> bodyMap = check asJsonMap(body);
    test:assertEquals(bodyMap["track_total_hits"], false);
}

@test:Config
isolated function testSparseEmbeddingQueryIsError() {
    ai:VectorStoreQuery query = {embedding: {indices: [0], values: [0.5]}};
    json|ai:Error result = buildSearchBody(query, queryMode(), {});
    test:assertTrue(result is ai:Error, "expected an ai:Error for a sparse embedding in a query");
}

// --- score normalization (C1) -------------------------------------------------------------------

@test:Config
isolated function testCosineScoreNormalizedWhenEnabled() {
    // score = (1 + cos) / 2  =>  cos = 2 * score - 1
    test:assertEquals(normalizeScore(1.0, ai:COSINE, true), 1.0);
    test:assertEquals(normalizeScore(0.5, ai:COSINE, true), 0.0);
    float oneThird = 1.0 / 3.0;
    test:assertEquals(normalizeScore(oneThird, ai:COSINE, true), 2.0 * oneThird - 1.0);
}

@test:Config
isolated function testCosineScoreUnchangedWhenDisabled() {
    test:assertEquals(normalizeScore(0.5, ai:COSINE, false), 0.5);
}

@test:Config
isolated function testNormalizationIgnoredForNonCosineMetrics() {
    test:assertEquals(normalizeScore(0.5, ai:EUCLIDEAN, true), 0.5);
    test:assertEquals(normalizeScore(0.5, ai:DOT_PRODUCT, true), 0.5);
}

// --- hitToVectorMatch -----------------------------------------------------------------------

@test:Config
isolated function testHitToVectorMatchNestedMetadata() returns error? {
    SearchHit hit = {
        _id: "internal-1",
        _score: 0.75,
        _source: {
            "embedding": [0.1, 0.2, 0.3],
            "content": "hello",
            "doc_id": "logical-1",
            "chunk_type": "text-chunk",
            "metadata": {"language": "en"}
        }
    };
    ai:VectorMatch 'match = check hitToVectorMatch(hit, queryMode(), {}, true);
    test:assertEquals('match.id, "logical-1");
    test:assertEquals('match.embedding, <ai:Vector>[0.1, 0.2, 0.3]);
    test:assertEquals('match.chunk.content, "hello");
    test:assertEquals('match.chunk.metadata, {"language": "en"});
    // 0.75 normalized: `2 * 0.75 - 1`.
    test:assertEquals('match.similarityScore, 0.5);
}

@test:Config
isolated function testHitToVectorMatchFallsBackToInternalId() returns error? {
    // When doc_id is absent from _source (e.g. an index the module did not create), fall back to
    // the internal _id rather than failing.
    SearchHit hit = {_id: "internal-only", _score: 1.0, _source: {"content": "x"}};
    ai:VectorMatch 'match = check hitToVectorMatch(hit, queryMode(), {}, true);
    test:assertEquals('match.id, "internal-only");
}

@test:Config
isolated function testHitToVectorMatchMissingEmbeddingIsEmptyVector() returns error? {
    SearchHit hit = {_id: "1", _score: 1.0, _source: {"content": "x"}};
    ai:VectorMatch 'match = check hitToVectorMatch(hit, queryMode(), {}, true);
    test:assertEquals('match.embedding, <ai:Vector>[]);
}

@test:Config
isolated function testHitToVectorMatchFlatMetadata() returns error? {
    DenseSearch configMode = {queryMode: ai:DENSE, indexConfig: {dimension: 3}};
    Configuration config = {metadataFieldName: ""};
    SearchHit hit = {
        _id: "1",
        _score: 1.0,
        _source: {
            "embedding": [0.1],
            "content": "x",
            "doc_id": "1",
            "chunk_type": "text-chunk",
            "language": "en"
        }
    };
    ai:VectorMatch 'match = check hitToVectorMatch(hit, configMode, config, true);
    test:assertEquals('match.chunk.metadata, {"language": "en"});
}

@test:Config
isolated function testHitToVectorMatchScoreZeroedWhenNotMeaningful() returns error? {
    // OpenSearch's constant score for a match_all/bool-only query (typically 1.0) is not a
    // similarity and must not be surfaced, regardless of hit._score.
    SearchHit hit = {_id: "1", _score: 1.0, _source: {"content": "x"}};
    ai:VectorMatch 'match = check hitToVectorMatch(hit, queryMode(), {}, false);
    test:assertEquals('match.similarityScore, 0.0);
}

@test:Config
isolated function testHitToVectorMatchScorePassedThroughWhenMeaningful() returns error? {
    SearchHit hit = {_id: "1", _score: 0.42, _source: {"content": "x"}};
    ai:VectorMatch 'match = check hitToVectorMatch(hit, rawScoreMode(), {}, true);
    test:assertEquals('match.similarityScore, 0.42);
}

// The default exists so `similarityScore` means the same thing here as in
// `ai:InMemoryVectorStore`, which returns a true cosine in [-1, 1]. OpenSearch's `cosinesimil`
// `_score` is [0, 1], so without this a threshold tuned against any other `ai:VectorStore`
// implementation would silently mean something else against this one.
@test:Config
isolated function testHitToVectorMatchNormalizesCosineByDefault() returns error? {
    SearchHit hit = {_id: "1", _score: 0.42, _source: {"content": "x"}};
    ai:VectorMatch 'match = check hitToVectorMatch(hit, queryMode(), {}, true);
    test:assertEquals('match.similarityScore, <float>(2 * 0.42 - 1));
}

@test:Config
isolated function testHitToVectorMatchRoundTripsTextChunkType() returns error? {
    SearchHit hit = {
        _id: "1",
        _score: 1.0,
        _source: {"content": "x", "chunk_type": "text-chunk"}
    };
    ai:VectorMatch 'match = check hitToVectorMatch(hit, queryMode(), {}, true);
    test:assertEquals('match.chunk.'type, "text-chunk");
    test:assertTrue('match.chunk is ai:TextChunk, "a stored 'text-chunk' should reconstruct as ai:TextChunk");
}

@test:Config
isolated function testHitToVectorMatchRoundTripsCustomChunkType() returns error? {
    SearchHit hit = {
        _id: "1",
        _score: 1.0,
        _source: {"content": "x", "chunk_type": "custom-chunk"}
    };
    ai:VectorMatch 'match = check hitToVectorMatch(hit, queryMode(), {}, true);
    test:assertEquals('match.chunk.'type, "custom-chunk");
    test:assertFalse('match.chunk is ai:TextChunk,
            "a non-'text-chunk' stored type must not be silently reconstructed as ai:TextChunk");
}

@test:Config
isolated function testHitToVectorMatchDefaultsToTextChunkWhenTypeAbsent() returns error? {
    // A pre-existing index the module did not write to may have no chunk_type field at all.
    SearchHit hit = {_id: "1", _score: 1.0, _source: {"content": "x"}};
    ai:VectorMatch 'match = check hitToVectorMatch(hit, queryMode(), {}, true);
    test:assertEquals('match.chunk.'type, "text-chunk");
}

// --- sparse query bodies ----------------------------------------------------------------------

@test:Config
isolated function testSparseQueryUsesNeuralSparseWithQueryTokens() returns error? {
    ai:VectorStoreQuery query = {embedding: {indices: [1055, 2048], values: [5.5, 1.25]}};
    json body = check buildSearchBody(query, sparseMode(), {});
    map<json> bodyMap = check asJsonMap(body);
    map<json> queryClause = check asJsonMap(bodyMap["query"]);
    map<json> neuralSparse = check asJsonMap(queryClause["neural_sparse"]);
    map<json> sparseField = check asJsonMap(neuralSparse["sparse_embedding"]);
    test:assertEquals(sparseField["query_tokens"], <json>{"1055": 5.5, "2048": 1.25});
    test:assertFalse(queryClause.hasKey("knn"), "a SPARSE query must carry no knn clause");
}

// `neural_sparse` over a rank_features field has no `filter` parameter of its own, so the filter
// has to be a sibling inside a `bool`. A bool.filter clause does not contribute to `_score`, so
// the sparse dot product stays the sole score.
@test:Config
isolated function testSparseQueryPutsFilterAsBoolSibling() returns error? {
    ai:VectorStoreQuery query = {
        embedding: {indices: [1055], values: [5.5]},
        filters: {filters: [{key: "language", operator: ai:EQUAL, value: "en"}]}
    };
    json body = check buildSearchBody(query, sparseMode(), {});
    map<json> bodyMap = check asJsonMap(body);
    map<json> boolClause = check asJsonMap((check asJsonMap(bodyMap["query"]))["bool"]);
    json[] must = check boolClause["must"].ensureType();
    json[] filter = check boolClause["filter"].ensureType();
    test:assertEquals(must.length(), 1);
    test:assertEquals(filter.length(), 1);
    map<json> neuralSparse = check asJsonMap((check asJsonMap(must[0]))["neural_sparse"]);
    map<json> sparseField = check asJsonMap(neuralSparse["sparse_embedding"]);
    test:assertFalse(sparseField.hasKey("filter"),
            "neural_sparse over rank_features takes no filter parameter; it must be a bool sibling");
}

@test:Config
isolated function testSparseQueryHonoursCustomFieldName() returns error? {
    ai:VectorStoreQuery query = {embedding: {indices: [1], values: [0.5]}};
    json body = check buildSearchBody(query, sparseMode("tokens"), {});
    map<json> neuralSparse = check asJsonMap((check asJsonMap((check asJsonMap(body))["query"]))["neural_sparse"]);
    test:assertTrue(neuralSparse.hasKey("tokens"));
}

@test:Config
isolated function testSparseQueryExcludesSparseFieldFromSource() returns error? {
    json body = check buildSearchBody({}, sparseMode(), {includeEmbeddingsInResults: false});
    map<json> sourceDirective = check asJsonMap((check asJsonMap(body))["_source"]);
    test:assertEquals(sourceDirective["excludes"], <json[]>["sparse_embedding"]);
}

// A query weight outside Lucene's (0, 64] fails the search with a shard exception naming nothing
// useful, so it is caught here instead.
@test:Config
isolated function testSparseQueryRejectsWeightAboveLuceneCeiling() {
    ai:VectorStoreQuery query = {embedding: {indices: [1], values: [64.1]}};
    json|ai:Error result = buildSearchBody(query, sparseMode(), {});
    if result !is ai:Error {
        test:assertFail("a query weight above 64 must be rejected before the request is sent");
    }
    test:assertTrue(result.message().includes("(0, 64]"));
}

// One Lucene clause per term, bounded by the cluster's indices.query.bool.max_clause_count.
// Exceeding it is a `too_many_clauses` shard exception that names neither the limit nor the query.
@test:Config
isolated function testSparseQueryRejectsTooManyTokens() {
    int[] indices = [];
    float[] values = [];
    foreach int index in 0 ..< 5 {
        indices.push(index);
        values.push(1.0);
    }
    ai:VectorStoreQuery query = {embedding: {indices, values}};
    json|ai:Error result = buildSearchBody(query, sparseMode("sparse_embedding", 4), {});
    if result !is ai:Error {
        test:assertFail("a query wider than 'maxQueryTokens' must be rejected before the wire");
    }
    test:assertTrue(result.message().includes("max_clause_count"),
            string `the message should explain the ceiling, got: ${result.message()}`);
}

@test:Config
isolated function testSparseQueryWithoutEmbeddingIsMatchAll() returns error? {
    json body = check buildSearchBody({}, sparseMode(), {});
    map<json> queryClause = check asJsonMap((check asJsonMap(body))["query"]);
    test:assertTrue(queryClause.hasKey("match_all"));
}

// --- mode/embedding strictness on the query path ----------------------------------------------

@test:Config
isolated function testSparseModeRejectsDenseQueryEmbedding() {
    json|ai:Error result = buildSearchBody({embedding: [0.1, 0.2, 0.3]}, sparseMode(), {});
    if result !is ai:Error {
        test:assertFail("a dense query embedding must be rejected by a SPARSE store");
    }
    test:assertTrue(result.message().includes("ai:SparseVector"));
}

@test:Config
isolated function testDenseModeRejectsSparseQueryEmbedding() {
    json|ai:Error result = buildSearchBody({embedding: {indices: [1], values: [0.5]}}, queryMode(), {});
    if result !is ai:Error {
        test:assertFail("a sparse query embedding must be rejected by a DENSE store");
    }
    test:assertTrue(result.message().includes("ai:Vector"));
}

@test:Config
isolated function testDenseModeRejectsHybridQueryEmbedding() {
    ai:VectorStoreQuery query = {embedding: {dense: [0.1], sparse: {indices: [1], values: [0.5]}}};
    json|ai:Error result = buildSearchBody(query, queryMode(), {});
    test:assertTrue(result is ai:Error, "a hybrid query embedding must be rejected by a DENSE store");
}

// --- reading a sparse hit back ----------------------------------------------------------------

@test:Config
isolated function testHitToVectorMatchSparseReturnsSparseVector() returns error? {
    SearchHit hit = {
        _id: "1",
        _score: 14.75,
        _source: {"sparse_embedding": {"9": 0.1, "2": 0.2}, "content": "hello", "doc_id": "1"}
    };
    ai:VectorMatch hitMatch = check hitToVectorMatch(hit, sparseMode(), {}, true);
    ai:Embedding embedding = hitMatch.embedding;
    if embedding !is ai:SparseVector {
        test:assertFail("a SPARSE store must return an ai:SparseVector");
    }
    test:assertEquals(embedding.indices, [2, 9]);
}

// A sparse score is an unbounded dot product. The cosine transform lives on DenseSearch and is
// unreachable here, so the raw score has to come through untouched.
@test:Config
isolated function testSparseScoreIsNotCosineNormalized() returns error? {
    SearchHit hit = {_id: "1", _score: 14.75, _source: {"doc_id": "1"}};
    ai:VectorMatch hitMatch = check hitToVectorMatch(hit, sparseMode(), {}, true);
    test:assertEquals(hitMatch.similarityScore, 14.75);
}

@test:Config
isolated function testSparseHitWithNoStoredVectorReturnsEmptySparseVector() returns error? {
    SearchHit hit = {_id: "1", _score: 1.0, _source: {"content": "hello", "doc_id": "1"}};
    ai:VectorMatch hitMatch = check hitToVectorMatch(hit, sparseMode(), {}, true);
    ai:Embedding embedding = hitMatch.embedding;
    if embedding !is ai:SparseVector {
        test:assertFail("a SPARSE store must return an ai:SparseVector even when the field is excluded");
    }
    test:assertEquals(embedding.indices, []);
}

// --- hybrid query bodies ----------------------------------------------------------------------

isolated function hybridQuery() returns ai:VectorStoreQuery =>
    {embedding: {dense: [0.1, 0.2, 0.3], sparse: {indices: [1055], values: [5.5]}}};

isolated function hybridClauseOf(json body) returns map<json>|error {
    map<json> bodyMap = check asJsonMap(body);
    map<json> queryClause = check asJsonMap(bodyMap["query"]);
    return asJsonMap(queryClause["hybrid"]);
}

// `hybrid` must be the top-level query: OpenSearch rejects it inside bool, function_score,
// constant_score, script_score or boosting.
@test:Config
isolated function testHybridQueryIsTopLevelWithTwoSubQueries() returns error? {
    json body = check buildSearchBody(hybridQuery(), hybridMode(), {});
    map<json> bodyMap = check asJsonMap(body);
    map<json> queryClause = check asJsonMap(bodyMap["query"]);
    test:assertTrue(queryClause.hasKey("hybrid"), "the hybrid clause must sit directly under 'query'");
    test:assertFalse(queryClause.hasKey("bool"), "a hybrid query must never be wrapped in a bool");

    map<json> hybrid = check asJsonMap(queryClause["hybrid"]);
    json[] subQueries = check hybrid["queries"].ensureType();
    test:assertEquals(subQueries.length(), 2);
    test:assertTrue((check asJsonMap(subQueries[0])).hasKey("knn"), "the dense sub-query comes first");
    map<json> sparseSub = check asJsonMap(subQueries[1]);
    test:assertTrue(sparseSub.hasKey("neural_sparse"), "the sparse sub-query comes second");
    test:assertFalse(hybrid.hasKey("boost"), "the hybrid query does not accept a boost");
}

@test:Config
isolated function testHybridQueryEmitsInlineNormalizationPipeline() returns error? {
    json body = check buildSearchBody(hybridQuery(), hybridMode(), {});
    map<json> bodyMap = check asJsonMap(body);
    map<json> pipeline = check asJsonMap(bodyMap["search_pipeline"]);
    json[] processors = check pipeline["phase_results_processors"].ensureType();
    map<json> normalization = check asJsonMap((check asJsonMap(processors[0]))["normalization-processor"]);
    test:assertEquals((check asJsonMap(normalization["normalization"]))["technique"], "min_max");
    map<json> combination = check asJsonMap(normalization["combination"]);
    test:assertEquals(combination["technique"], "arithmetic_mean");
    test:assertEquals((check asJsonMap(combination["parameters"]))["weights"], <json[]>[0.5, 0.5]);
}

// The weights array is positional against the sub-query array. An asymmetric split is what makes
// a swapped pair detectable at all -- with the 0.5/0.5 default it would pass either way.
@test:Config
isolated function testHybridWeightsAreOrderedDenseThenSparse() returns error? {
    HybridSearch mode = hybridMode({normalization: L2, combination: GEOMETRIC_MEAN,
            denseWeight: 0.3, sparseWeight: 0.7});
    json body = check buildSearchBody(hybridQuery(), mode, {});
    map<json> pipeline = check asJsonMap((check asJsonMap(body))["search_pipeline"]);
    json[] processors = check pipeline["phase_results_processors"].ensureType();
    map<json> normalization = check asJsonMap((check asJsonMap(processors[0]))["normalization-processor"]);
    test:assertEquals((check asJsonMap(normalization["normalization"]))["technique"], "l2");
    map<json> combination = check asJsonMap(normalization["combination"]);
    test:assertEquals(combination["technique"], "geometric_mean");
    test:assertEquals((check asJsonMap(combination["parameters"]))["weights"], <json[]>[0.3, 0.7],
            "weights are positional: dense first, matching the sub-query order");
}

// A top-level hybrid.filter is OpenSearch 3.0+. AWS still offers 2.x domains, so the filter is
// duplicated into each sub-query instead -- documented as equivalent, and the two shapes differ.
@test:Config
isolated function testHybridFilterIsDuplicatedIntoBothSubQueries() returns error? {
    ai:VectorStoreQuery query = {
        embedding: {dense: [0.1, 0.2, 0.3], sparse: {indices: [1055], values: [5.5]}},
        filters: {filters: [{key: "language", operator: ai:EQUAL, value: "en"}]}
    };
    json body = check buildSearchBody(query, hybridMode(), {});
    map<json> hybrid = check hybridClauseOf(body);
    test:assertFalse(hybrid.hasKey("filter"),
            "a top-level hybrid.filter is 3.0+ only and must not be emitted");

    json[] subQueries = check hybrid["queries"].ensureType();
    map<json> knnField = check asJsonMap((check asJsonMap((check asJsonMap(subQueries[0]))["knn"]))["embedding"]);
    test:assertTrue(knnField.hasKey("filter"), "the knn sub-query takes its filter inside the vector object");
    map<json> sparseBool = check asJsonMap((check asJsonMap(subQueries[1]))["bool"]);
    json[] sparseFilter = check sparseBool["filter"].ensureType();
    test:assertEquals(sparseFilter.length(), 1, "the sparse sub-query needs a bool sibling filter");
}

// A NamedSearchPipeline travels as ?search_pipeline=, never as a body object. OpenSearch rejects a
// request carrying both with "Both named and inline search pipeline were specified".
@test:Config
isolated function testHybridNamedPipelineEmitsNoInlineObject() returns error? {
    HybridSearch mode = hybridMode({name: "my-pipeline"});
    json body = check buildSearchBody(hybridQuery(), mode, {});
    map<json> bodyMap = check asJsonMap(body);
    test:assertFalse(bodyMap.hasKey("search_pipeline"),
            "a named pipeline must not also be sent inline in the body");
    test:assertEquals(searchPipelineName(mode), "my-pipeline");
}

@test:Config
isolated function testInlineFusionSendsNoPipelineQueryParameter() {
    test:assertTrue(searchPipelineName(hybridMode()) is (),
            "inline fusion must not also set the ?search_pipeline= parameter");
    test:assertTrue(searchPipelineName(denseMode()) is ());
    test:assertTrue(searchPipelineName(sparseMode()) is ());
}

// A query with no embedding scores nothing, so there is nothing to normalize. Emitting a pipeline
// would ask OpenSearch to fuse a constant score with itself.
@test:Config
isolated function testHybridQueryWithoutEmbeddingEmitsNoPipeline() returns error? {
    json body = check buildSearchBody({}, hybridMode(), {});
    map<json> bodyMap = check asJsonMap(body);
    test:assertFalse(bodyMap.hasKey("search_pipeline"));
    map<json> queryClause = check asJsonMap(bodyMap["query"]);
    test:assertTrue(queryClause.hasKey("match_all"));
}

@test:Config
isolated function testHybridExcludesBothVectorFieldsFromSource() returns error? {
    json body = check buildSearchBody({}, hybridMode(), {includeEmbeddingsInResults: false});
    map<json> sourceDirective = check asJsonMap((check asJsonMap(body))["_source"]);
    test:assertEquals(sourceDirective["excludes"], <json[]>["embedding", "sparse_embedding"]);
}

@test:Config
isolated function testHybridModeRejectsDenseOnlyQueryEmbedding() {
    json|ai:Error result = buildSearchBody({embedding: [0.1, 0.2, 0.3]}, hybridMode(), {});
    if result !is ai:Error {
        test:assertFail("a dense-only query embedding must be rejected by a HYBRID store");
    }
    test:assertTrue(result.message().includes("ai:HybridVector"));
}

@test:Config
isolated function testHybridModeRejectsSparseOnlyQueryEmbedding() {
    json|ai:Error result = buildSearchBody({embedding: {indices: [1], values: [0.5]}}, hybridMode(), {});
    test:assertTrue(result is ai:Error, "a sparse-only query embedding must be rejected by a HYBRID store");
}

@test:Config
isolated function testHitToVectorMatchHybridReturnsBothHalves() returns error? {
    SearchHit hit = {
        _id: "1",
        _score: 0.5005,
        _source: {"embedding": [0.1, 0.2], "sparse_embedding": {"7": 0.5}, "doc_id": "1"}
    };
    ai:VectorMatch hitMatch = check hitToVectorMatch(hit, hybridMode(), {}, true);
    ai:Embedding embedding = hitMatch.embedding;
    if embedding !is ai:HybridVector {
        test:assertFail("a HYBRID store must return an ai:HybridVector");
    }
    test:assertEquals(embedding.dense, <ai:Vector>[0.1, 0.2]);
    test:assertEquals(embedding.sparse.indices, [7]);
    // Already normalized to (0.0, 1.0] by the fusion pipeline; the cosine transform is unreachable
    // here because HybridSearch declares no normalizeCosineScore.
    test:assertEquals(hitMatch.similarityScore, 0.5005);
}
