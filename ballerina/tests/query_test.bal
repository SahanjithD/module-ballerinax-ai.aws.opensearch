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

isolated function queryConfig() returns Configuration => {indexConfig: {dimension: 3}};

// `normalizeCosineScore` defaults to `true`, so a test that wants OpenSearch's raw `_score` has to
// ask for it. See `testHitToVectorMatchNormalizesCosineByDefault` for why that is the default.
isolated function rawScoreConfig() returns Configuration =>
    {indexConfig: {dimension: 3}, normalizeCosineScore: false};

isolated function asJsonMap(json value) returns map<json>|error => value.ensureType();

// --- the four (embedding, filters) combinations ------------------------------------------------

@test:Config
isolated function testQueryNeitherEmbeddingNorFilters() returns error? {
    ai:VectorStoreQuery query = {};
    json body = check buildSearchBody(query, queryConfig());
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
    json body = check buildSearchBody(query, queryConfig());
    map<json> bodyMap = check asJsonMap(body);
    map<json> queryClause = check asJsonMap(bodyMap["query"]);
    test:assertTrue(queryClause.hasKey("bool"), "expected a bool.filter query when only filters are given");
    test:assertFalse(queryClause.hasKey("knn"));
}

@test:Config
isolated function testQueryEmbeddingOnlyNoFilters() returns error? {
    ai:VectorStoreQuery query = {embedding: [0.1, 0.2, 0.3]};
    json body = check buildSearchBody(query, queryConfig());
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
    json body = check buildSearchBody(query, queryConfig());
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
    json body = check buildSearchBody(query, queryConfig());
    map<json> bodyMap = check asJsonMap(body);
    map<json> queryClause = check asJsonMap(bodyMap["query"]);
    test:assertTrue(queryClause.hasKey("match_all"));
}

// --- topK handling -------------------------------------------------------------------------------

@test:Config
isolated function testTopKMinusOneMeansAll() returns error? {
    ai:VectorStoreQuery query = {topK: -1};
    json body = check buildSearchBody(query, queryConfig());
    map<json> bodyMap = check asJsonMap(body);
    test:assertEquals(bodyMap["size"], 10000);
}

@test:Config
isolated function testTopKZeroMeansAll() returns error? {
    // Matches ai:InMemoryVectorStore: any topK < 1 means "return all", not just -1.
    ai:VectorStoreQuery query = {topK: 0};
    json body = check buildSearchBody(query, queryConfig());
    map<json> bodyMap = check asJsonMap(body);
    test:assertEquals(bodyMap["size"], 10000);
}

@test:Config
isolated function testTopKNegativeBelowMinusOneMeansAll() returns error? {
    ai:VectorStoreQuery query = {topK: -5};
    json body = check buildSearchBody(query, queryConfig());
    map<json> bodyMap = check asJsonMap(body);
    test:assertEquals(bodyMap["size"], 10000);
}

@test:Config
isolated function testTopKPositiveIsHonored() returns error? {
    ai:VectorStoreQuery query = {topK: 5};
    json body = check buildSearchBody(query, queryConfig());
    map<json> bodyMap = check asJsonMap(body);
    test:assertEquals(bodyMap["size"], 5);
}

@test:Config
isolated function testTopKAboveMaxResultWindowIsError() {
    ai:VectorStoreQuery query = {topK: 20000};
    json|ai:Error result = buildSearchBody(query, queryConfig());
    test:assertTrue(result is ai:Error, "expected an ai:Error when topK exceeds maxResultWindow");
}

@test:Config
isolated function testCustomMaxResultWindowRaisesTheCeiling() returns error? {
    Configuration config = {indexConfig: {dimension: 3}, maxResultWindow: 50000};
    ai:VectorStoreQuery query = {topK: 20000};
    json body = check buildSearchBody(query, config);
    map<json> bodyMap = check asJsonMap(body);
    test:assertEquals(bodyMap["size"], 20000);
}

// --- _source directive -----------------------------------------------------------------------

@test:Config
isolated function testSourceIncludesEmbeddingsByDefault() returns error? {
    json body = check buildSearchBody({}, queryConfig());
    map<json> bodyMap = check asJsonMap(body);
    test:assertEquals(bodyMap["_source"], true);
}

@test:Config
isolated function testSourceExcludesEmbeddingsWhenConfigured() returns error? {
    Configuration config = {indexConfig: {dimension: 3}, includeEmbeddingsInResults: false};
    json body = check buildSearchBody({}, config);
    map<json> bodyMap = check asJsonMap(body);
    map<json> sourceDirective = check asJsonMap(bodyMap["_source"]);
    test:assertEquals(sourceDirective["excludes"], <json[]>["embedding"]);
}

@test:Config
isolated function testTrackTotalHitsAlwaysFalse() returns error? {
    json body = check buildSearchBody({}, queryConfig());
    map<json> bodyMap = check asJsonMap(body);
    test:assertEquals(bodyMap["track_total_hits"], false);
}

@test:Config
isolated function testSparseEmbeddingQueryIsError() {
    ai:VectorStoreQuery query = {embedding: {indices: [0], values: [0.5]}};
    json|ai:Error result = buildSearchBody(query, queryConfig());
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
    ai:VectorMatch 'match = check hitToVectorMatch(hit, queryConfig(), true);
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
    ai:VectorMatch 'match = check hitToVectorMatch(hit, queryConfig(), true);
    test:assertEquals('match.id, "internal-only");
}

@test:Config
isolated function testHitToVectorMatchMissingEmbeddingIsEmptyVector() returns error? {
    SearchHit hit = {_id: "1", _score: 1.0, _source: {"content": "x"}};
    ai:VectorMatch 'match = check hitToVectorMatch(hit, queryConfig(), true);
    test:assertEquals('match.embedding, <ai:Vector>[]);
}

@test:Config
isolated function testHitToVectorMatchFlatMetadata() returns error? {
    Configuration config = {indexConfig: {dimension: 3}, metadataFieldName: ""};
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
    ai:VectorMatch 'match = check hitToVectorMatch(hit, config, true);
    test:assertEquals('match.chunk.metadata, {"language": "en"});
}

@test:Config
isolated function testHitToVectorMatchScoreZeroedWhenNotMeaningful() returns error? {
    // OpenSearch's constant score for a match_all/bool-only query (typically 1.0) is not a
    // similarity and must not be surfaced, regardless of hit._score.
    SearchHit hit = {_id: "1", _score: 1.0, _source: {"content": "x"}};
    ai:VectorMatch 'match = check hitToVectorMatch(hit, queryConfig(), false);
    test:assertEquals('match.similarityScore, 0.0);
}

@test:Config
isolated function testHitToVectorMatchScorePassedThroughWhenMeaningful() returns error? {
    SearchHit hit = {_id: "1", _score: 0.42, _source: {"content": "x"}};
    ai:VectorMatch 'match = check hitToVectorMatch(hit, rawScoreConfig(), true);
    test:assertEquals('match.similarityScore, 0.42);
}

// The default exists so `similarityScore` means the same thing here as in
// `ai:InMemoryVectorStore`, which returns a true cosine in [-1, 1]. OpenSearch's `cosinesimil`
// `_score` is [0, 1], so without this a threshold tuned against any other `ai:VectorStore`
// implementation would silently mean something else against this one.
@test:Config
isolated function testHitToVectorMatchNormalizesCosineByDefault() returns error? {
    SearchHit hit = {_id: "1", _score: 0.42, _source: {"content": "x"}};
    ai:VectorMatch 'match = check hitToVectorMatch(hit, queryConfig(), true);
    test:assertEquals('match.similarityScore, <float>(2 * 0.42 - 1));
}

@test:Config
isolated function testHitToVectorMatchRoundTripsTextChunkType() returns error? {
    SearchHit hit = {
        _id: "1",
        _score: 1.0,
        _source: {"content": "x", "chunk_type": "text-chunk"}
    };
    ai:VectorMatch 'match = check hitToVectorMatch(hit, queryConfig(), true);
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
    ai:VectorMatch 'match = check hitToVectorMatch(hit, queryConfig(), true);
    test:assertEquals('match.chunk.'type, "custom-chunk");
    test:assertFalse('match.chunk is ai:TextChunk,
            "a non-'text-chunk' stored type must not be silently reconstructed as ai:TextChunk");
}

@test:Config
isolated function testHitToVectorMatchDefaultsToTextChunkWhenTypeAbsent() returns error? {
    // A pre-existing index the module did not write to may have no chunk_type field at all.
    SearchHit hit = {_id: "1", _score: 1.0, _source: {"content": "x"}};
    ai:VectorMatch 'match = check hitToVectorMatch(hit, queryConfig(), true);
    test:assertEquals('match.chunk.'type, "text-chunk");
}
