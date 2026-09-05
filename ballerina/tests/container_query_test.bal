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

// `query` against a real k-NN index. `query_test.bal` asserts the search body that
// `buildSearchBody` emits; these assert what OpenSearch does with it -- result ordering, the
// `size`/`k` ceiling, the `_source` exclusion, and the score arithmetic, none of which can be
// checked by inspecting a request body.

import ballerina/ai;
import ballerina/test;

# Seeds four entries at decreasing cosine similarity to `queryVector()`.
#
# + store - The store to seed
# + return - An error if seeding fails
isolated function seedProximityEntries(VectorStore store) returns error? {
    check store.add([
        entry("nearest", vec(1.0), "nearest"),
        entry("near", vec(0.9, 0.1), "near"),
        entry("far", vec(0.5, 0.8), "far"),
        entry("orthogonal", vec(0.0, 1.0), "orthogonal")
    ]);
}

@test:Config {groups: ["docker"]}
isolated function testContainerQueryOrdersByProximity() returns error? {
    string indexName = containerIndexName("q-order");
    VectorStore store = check newContainerStore(indexName);
    check seedProximityEntries(store);

    ai:VectorMatch[] matches = check store.query({embedding: queryVector(), topK: 4});
    test:assertEquals(matches.length(), 4);
    test:assertEquals(matches[0].id, "nearest", "results should come back most-similar first");
    test:assertEquals(matches[3].id, "orthogonal", "the least similar entry should come back last");
    check store.close();
}

@test:Config {groups: ["docker"]}
isolated function testContainerQueryTopKLimitsResults() returns error? {
    string indexName = containerIndexName("q-topk");
    VectorStore store = check newContainerStore(indexName);
    check seedProximityEntries(store);

    ai:VectorMatch[] matches = check store.query({embedding: queryVector(), topK: 2});
    test:assertEquals(matches.length(), 2, "'topK' should reach the server as both 'k' and 'size'");
    assertIdsEqual(matches, ["nearest", "near"], "the two nearest entries should be returned");
    check store.close();
}

@test:Config {groups: ["docker"]}
isolated function testContainerQueryNegativeTopKReturnsEverything() returns error? {
    string indexName = containerIndexName("q-all");
    Configuration config = {
        indexConfig: {dimension: CONTAINER_DIMENSION},
        maxResultWindow: 100
    };
    VectorStore store = check newContainerStore(indexName, config);
    check seedProximityEntries(store);

    ai:VectorMatch[] matches = check store.query({embedding: queryVector(), topK: -1});
    test:assertEquals(matches.length(), 4, "a negative 'topK' means 'return everything'");
    check store.close();
}

@test:Config {groups: ["docker"]}
isolated function testContainerQueryMaxResultWindowCapsReturnAll() returns error? {
    string indexName = containerIndexName("q-window");
    Configuration config = {
        indexConfig: {dimension: CONTAINER_DIMENSION},
        maxResultWindow: 2
    };
    VectorStore store = check newContainerStore(indexName, config);
    check seedProximityEntries(store);

    ai:VectorMatch[] matches = check store.query({embedding: queryVector(), topK: -1});
    test:assertEquals(matches.length(), 2,
            "'return everything' should be capped at 'maxResultWindow', not the index size");
    check store.close();
}

@test:Config {groups: ["docker"]}
isolated function testContainerQueryExcludesEmbeddingsWhenConfigured() returns error? {
    string indexName = containerIndexName("q-noembed");
    Configuration config = {
        indexConfig: {dimension: CONTAINER_DIMENSION},
        includeEmbeddingsInResults: false
    };
    VectorStore store = check newContainerStore(indexName, config);
    check store.add([entry("e", queryVector(), "content survives")]);

    ai:VectorMatch[] matches = check store.query({embedding: queryVector(), topK: 1});
    test:assertEquals(matches.length(), 1);
    test:assertEquals(matches[0].embedding, <ai:Vector>[],
            "the '_source' exclusion should leave the returned embedding empty");
    test:assertEquals(matches[0].chunk.content, "content survives",
            "excluding the vector must not affect the rest of the document");
    check store.close();
}

@test:Config {groups: ["docker"]}
isolated function testContainerQueryIncludesEmbeddingsByDefault() returns error? {
    string indexName = containerIndexName("q-embed");
    VectorStore store = check newContainerStore(indexName);
    check store.add([entry("e", vec(0.6, 0.8), "with vector")]);

    ai:VectorMatch[] matches = check store.query({embedding: queryVector(), topK: 1});
    test:assertEquals(matches[0].embedding, vec(0.6, 0.8));
    check store.close();
}

@test:Config {groups: ["docker"]}
isolated function testContainerQueryRawCosineScore() returns error? {
    string indexName = containerIndexName("q-rawscore");
    Configuration config = {
        indexConfig: {dimension: CONTAINER_DIMENSION},
        normalizeCosineScore: false
    };
    VectorStore store = check newContainerStore(indexName, config);
    check store.add([entry("same", vec(1.0)), entry("orthogonal", vec(0.0, 1.0))]);

    ai:VectorMatch[] matches = check store.query({embedding: queryVector(), topK: 2});
    // OpenSearch's `cosinesimil` space scores as `(1 + cos) / 2`, so an identical vector is 1.0
    // and an orthogonal one is 0.5 -- not the [-1, 1] a caller may expect from "cosine similarity".
    test:assertEquals(matches[0].id, "same");
    assertScoreCloseTo(matches[0].similarityScore, 1.0, "an identical vector should score 1.0 raw");
    assertScoreCloseTo(matches[1].similarityScore, 0.5, "an orthogonal vector should score 0.5 raw");
    check store.close();
}

@test:Config {groups: ["docker"]}
isolated function testContainerQueryNormalizedCosineScore() returns error? {
    string indexName = containerIndexName("q-normscore");
    // Normalization is the default; named explicitly here so the test reads against its opposite.
    Configuration config = {
        indexConfig: {dimension: CONTAINER_DIMENSION},
        normalizeCosineScore: true
    };
    VectorStore store = check newContainerStore(indexName, config);
    check store.add([entry("same", vec(1.0)), entry("orthogonal", vec(0.0, 1.0))]);

    ai:VectorMatch[] matches = check store.query({embedding: queryVector(), topK: 2});
    // `cos = 2 * score - 1` maps those back to 1.0 and 0.0.
    assertScoreCloseTo(matches[0].similarityScore, 1.0, "an identical vector normalizes to 1.0");
    assertScoreCloseTo(matches[1].similarityScore, 0.0, "an orthogonal vector normalizes to 0.0");
    check store.close();
}

@test:Config {groups: ["docker"]}
isolated function testContainerQueryWithoutEmbeddingHasZeroScore() returns error? {
    string indexName = containerIndexName("q-noscore");
    VectorStore store = check newContainerStore(indexName);
    check store.add([entry("a", queryVector(), "doc", {"language": "en"})]);

    // With no embedding, OpenSearch returns a constant score that is not a similarity at all, so
    // the module reports 0.0 rather than passing a meaningless number through.
    ai:VectorMatch[] filtered = check store.query({
        filters: {filters: [{key: "language", operator: ai:EQUAL, value: "en"}]},
        topK: 10
    });
    test:assertEquals(filtered.length(), 1);
    test:assertEquals(filtered[0].similarityScore, 0.0, "a filters-only match has no meaningful score");

    ai:VectorMatch[] all = check store.query({topK: 10});
    test:assertEquals(all.length(), 1);
    test:assertEquals(all[0].similarityScore, 0.0, "a 'match_all' match has no meaningful score");
    check store.close();
}

@test:Config {groups: ["docker"]}
isolated function testContainerQueryNeitherEmbeddingNorFiltersReturnsAll() returns error? {
    string indexName = containerIndexName("q-matchall");
    VectorStore store = check newContainerStore(indexName);
    check seedProximityEntries(store);

    ai:VectorMatch[] matches = check store.query({topK: 10});
    assertIdsEqual(matches, ["nearest", "near", "far", "orthogonal"],
            "with neither embedding nor filters, every entry should be returned");
    check store.close();
}

@test:Config {groups: ["docker"]}
isolated function testContainerQueryOnEmptyIndexReturnsNoMatches() returns error? {
    string indexName = containerIndexName("q-empty");
    VectorStore store = check newContainerStore(indexName);

    ai:VectorMatch[] byVector = check store.query({embedding: queryVector(), topK: 10});
    test:assertEquals(byVector.length(), 0);
    ai:VectorMatch[] all = check store.query({topK: 10});
    test:assertEquals(all.length(), 0);
    check store.close();
}

@test:Config {groups: ["docker"]}
isolated function testContainerQueryRejectsTopKAboveMaxResultWindow() returns error? {
    string indexName = containerIndexName("q-overwindow");
    Configuration config = {
        indexConfig: {dimension: CONTAINER_DIMENSION},
        maxResultWindow: 5
    };
    VectorStore store = check newContainerStore(indexName, config);

    ai:VectorMatch[]|ai:Error result = store.query({embedding: queryVector(), topK: 6});
    test:assertTrue(result is ai:Error, "'topK' above 'maxResultWindow' should be rejected client-side");
    check store.close();
}

# Asserts a similarity score matches an expected value, allowing for float arithmetic on the
# server and the wire.
#
# + actual - The score returned
# + expected - The score expected
# + message - The assertion message
isolated function assertScoreCloseTo(float actual, float expected, string message) {
    float delta = actual - expected;
    float magnitude = delta < 0.0 ? -delta : delta;
    test:assertTrue(magnitude < 0.001,
            string `${message} (expected ~${expected}, got ${actual})`);
}
