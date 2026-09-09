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

// Integration coverage for SPARSE against the throwaway OpenSearch container.
//
// A unit test can assert that a `rank_features` mapping is emitted; only a real server can prove
// that OpenSearch accepts an index with no `settings` block at all, and that a `neural_sparse`
// query against it works with no ML model, no ingest pipeline and no `index.knn` setting.

import ballerina/ai;
import ballerina/test;

# The sparse search mode the container suite uses.
#
# + return - The search mode
isolated function containerSparseMode() returns SparseSearch => {queryMode: ai:SPARSE};

# Constructs a SPARSE store against the plain-HTTP container.
#
# + indexName - The target index
# + return - The store, or an `ai:Error` if construction fails
isolated function newSparseContainerStore(string indexName) returns VectorStore|ai:Error =>
    newContainerStore(indexName, containerSparseMode());

# Builds a sparse vector entry.
#
# + id - The logical entry id
# + indices - The feature indices
# + values - The feature weights
# + language - The `language` metadata value the filter tests key off
# + return - The entry
isolated function sparseVectorEntry(string id, int[] indices, float[] values, string language = "en")
        returns ai:VectorEntry =>
    {
        id,
        embedding: {indices, values},
        chunk: <ai:TextChunk>{content: string `content for ${id}`, metadata: {"language": language}}
    };

@test:Config {groups: ["docker"]}
isolated function testContainerSparseIndexIsCreatedWithoutSettings() returns error? {
    string indexName = containerIndexName("sparse-init");
    VectorStore store = check newSparseContainerStore(indexName);

    map<json> mappings = check indexMapping(indexName);
    map<json> properties = check mapField(mappings, "properties");
    map<json> sparseField = check mapField(properties, "sparse_embedding");
    test:assertEquals(sparseField["type"], "rank_features");

    // The mapping carries no `settings` block, so OpenSearch must have defaulted `index.knn` to
    // absent rather than rejecting the create.
    map<json> settings = check indexSettings(indexName);
    test:assertFalse(settings.hasKey("knn"),
            string `a SPARSE index must not request the k-NN codec, got: ${settings.toJsonString()}`);
    check store.close();
}

// A zero weight is rejected by Lucene at index time. Asserting the document count stayed at zero
// proves the guard fired client-side rather than the whole _bulk failing at the server.
@test:Config {groups: ["docker"]}
isolated function testContainerSparseZeroWeightIsRejectedBeforeTheWire() returns error? {
    string indexName = containerIndexName("sparse-zero");
    VectorStore store = check newSparseContainerStore(indexName);

    ai:Error? result = store.add([sparseVectorEntry("zero", [7], [0.0])]);
    test:assertTrue(result is ai:Error, "a zero sparse weight must be rejected");
    test:assertEquals(check documentCount(indexName), 0, "nothing should have reached the server");
    check store.close();
}
// The gating test for the whole sparse design: it proves the `neural-search` plugin is present in
// the container image and that raw `query_tokens` scoring works with no model deployed anywhere.
// The expected order is the plain dot product of the query weights against the stored ones.
@test:Config {groups: ["docker"]}
isolated function testContainerSparseQueryRanksByDotProduct() returns error? {
    string indexName = containerIndexName("sparse-rank");
    VectorStore store = check newSparseContainerStore(indexName);
    check store.add([
        // against query {1055: 2.0, 2048: 3.0}: 2*5.5 + 3*1.25 = 14.75
        sparseVectorEntry("alpha", [1055, 2048], [5.5, 1.25]),
        // 3*4.0 = 12.0
        sparseVectorEntry("gamma", [2048], [4.0]),
        // 2*1.0 = 2.0
        sparseVectorEntry("beta", [1055, 3000], [1.0, 9.0], "fr")
    ]);

    ai:VectorMatch[] matches = check store.query({
        embedding: {indices: [1055, 2048], values: [2.0, 3.0]},
        topK: 10
    });
    test:assertEquals(matches.length(), 3);
    test:assertEquals(matches[0].id, "alpha");
    test:assertEquals(matches[1].id, "gamma");
    test:assertEquals(matches[2].id, "beta");
    check store.close();
}

// A sparse score is an unbounded dot product, not a similarity in [0, 1]. This pins that the
// cosine transform is never applied to it -- 14.75 could not survive `2 * score - 1`.
@test:Config {groups: ["docker"]}
isolated function testContainerSparseScoreIsUnboundedAndUntransformed() returns error? {
    string indexName = containerIndexName("sparse-score");
    VectorStore store = check newSparseContainerStore(indexName);
    check store.add([sparseVectorEntry("alpha", [1055, 2048], [5.5, 1.25])]);

    ai:VectorMatch[] matches = check store.query({
        embedding: {indices: [1055, 2048], values: [2.0, 3.0]},
        topK: 1
    });
    test:assertTrue(matches[0].similarityScore > 1.0,
            string `a sparse dot product is unbounded above 1.0, got: ${matches[0].similarityScore}`);
    assertScoreCloseTo(matches[0].similarityScore, 14.75, "the score should be the raw dot product");
    check store.close();
}

// `neural_sparse` over a rank_features field takes no `filter` parameter of its own, so the filter
// travels as a sibling inside a `bool`. This proves that shape actually filters.
@test:Config {groups: ["docker"]}
isolated function testContainerSparseFilterAppliesAsBoolSibling() returns error? {
    string indexName = containerIndexName("sparse-filter");
    VectorStore store = check newSparseContainerStore(indexName);
    check store.add([
        sparseVectorEntry("english", [1055], [5.0], "en"),
        sparseVectorEntry("french", [1055], [9.0], "fr")
    ]);

    ai:VectorMatch[] matches = check store.query({
        embedding: {indices: [1055], values: [2.0]},
        filters: {filters: [{key: "language", operator: ai:EQUAL, value: "en"}]},
        topK: 10
    });
    test:assertEquals(matches.length(), 1, "the French entry should have been filtered out");
    test:assertEquals(matches[0].id, "english");
    check store.close();
}


// `rank_features` keeps roughly nine significant bits, so a round-tripped weight carries about
// 0.4% relative error. Asserted as a tolerance so nobody later writes an exact-equality check
// here and watches it flake.
@test:Config {groups: ["docker"]}
isolated function testContainerSparseRoundTripIsLossy() returns error? {
    string indexName = containerIndexName("sparse-roundtrip");
    VectorStore store = check newSparseContainerStore(indexName);
    check store.add([sparseVectorEntry("alpha", [7, 9], [0.3, 2.5])]);

    ai:VectorMatch[] matches = check store.query({embedding: {indices: [7], values: [1.0]}, topK: 1});
    ai:Embedding embedding = matches[0].embedding;
    if embedding !is ai:SparseVector {
        test:assertFail("a SPARSE store must return an ai:SparseVector");
    }
    test:assertEquals(embedding.indices, [7, 9], "indices must come back sorted ascending");
    foreach int position in 0 ..< 2 {
        float expected = [0.3, 2.5][position];
        float actual = embedding.values[position];
        float drift = (actual - expected) / expected;
        test:assertTrue(drift < 0.01 && drift > -0.01,
                string `a stored weight should be within 1% of the original, got ${actual} for ${expected}`);
    }
    check store.close();
}

// --- two-phase acceleration ------------------------------------------------------------------------

// The query the two-phase tests issue. The weights are chosen so the split actually happens: with
// the default `pruneRatio` of 0.4 the threshold is `0.4 * 10.0 = 4.0`, so token 1055 (weight 10.0)
// scores in the first phase and token 2001 (weight 1.0) is left to the rescoring pass. A query
// whose tokens all clear the threshold would exercise the pipeline without exercising the split.
isolated function twoPhaseContainerQuery() returns ai:VectorStoreQuery =>
    {embedding: {indices: [1055, 2001], values: [10.0, 1.0]}, topK: 10};

// The two entries the two-phase tests seed. `high-then-low` wins on the first-phase token and
// `low-then-high` wins on the rescoring token, so a rescoring pass that never ran, or that dropped
// its contribution, changes the scores.
isolated function twoPhaseSeedEntries() returns ai:VectorEntry[] => [
    sparseVectorEntry("high-then-low", [1055, 2001], [9.0, 1.0]),
    sparseVectorEntry("low-then-high", [1055, 2001], [1.0, 9.0])
];

// The gating test for `TwoPhaseAcceleration`: it proves the container accepts an inline
// `request_processors` pipeline carrying `neural_sparse_two_phase_processor` (2.15+), and that the
// accelerated result is the same as the unaccelerated one. A rejected pipeline fails the search
// outright rather than degrading.
//
// Comparing against the same query without the processor is what makes this meaningful. The
// acceleration is an optimisation, so on a corpus this small -- well inside `expansionRate * size`
// -- it must reproduce the exact scores rather than merely something plausible. Asserting the
// scores rather than only the order is what would catch a rescoring pass that silently dropped its
// tokens' contribution: the ranking here survives that, but the scores do not.
@test:Config {groups: ["docker"]}
isolated function testContainerTwoPhaseAccelerationMatchesTheUnacceleratedResult() returns error? {
    string indexName = containerIndexName("sparse-twophase");
    VectorStore seed = check newSparseContainerStore(indexName);
    check seed.add(twoPhaseSeedEntries());
    ai:VectorMatch[] baseline = check seed.query(twoPhaseContainerQuery());
    check seed.close();

    SparseSearch accelerated = {queryMode: ai:SPARSE, twoPhaseAcceleration: {}};
    VectorStore store = check newContainerStore(indexName, accelerated, {createIndexIfNotExists: false});
    ai:VectorMatch[] matches = check store.query(twoPhaseContainerQuery());
    check store.close();

    test:assertEquals(matches.length(), baseline.length(),
            "the two-phase pass must return the same matches as the single-phase one");
    foreach int i in 0 ..< baseline.length() {
        test:assertEquals(matches[i].id, baseline[i].id,
                string `the acceleration must not reorder results, differing at position ${i}`);
        test:assertEquals(matches[i].similarityScore, baseline[i].similarityScore,
                string `entry '${baseline[i].id.toString()}' scored ${matches[i].similarityScore} under the ` +
                string `two-phase pass and ${baseline[i].similarityScore} without it; the low-weight ` +
                "tokens' contribution must survive the rescoring pass");
    }
}

// The processor rewrites the query, so it has to survive the `bool` wrapper a filtered sparse
// query is nested in -- the only other clause shape this mode produces.
@test:Config {groups: ["docker"]}
isolated function testContainerTwoPhaseAccelerationWorksWithAFilter() returns error? {
    string indexName = containerIndexName("sparse-twophase-filter");
    VectorStore seed = check newSparseContainerStore(indexName);
    check seed.add([
        sparseVectorEntry("english", [1055], [9.0], "en"),
        sparseVectorEntry("french", [1055], [9.5], "fr")
    ]);
    check seed.close();

    SparseSearch accelerated = {queryMode: ai:SPARSE, twoPhaseAcceleration: {}};
    VectorStore store = check newContainerStore(indexName, accelerated, {createIndexIfNotExists: false});
    ai:VectorMatch[] matches = check store.query({
        embedding: {indices: [1055], values: [2.0]},
        filters: {filters: [{key: "language", value: "en"}]},
        topK: 10
    });
    check store.close();

    test:assertEquals(matches.length(), 1, "the filter must still apply under the two-phase rewrite");
    test:assertEquals(matches[0].id, "english");
}
