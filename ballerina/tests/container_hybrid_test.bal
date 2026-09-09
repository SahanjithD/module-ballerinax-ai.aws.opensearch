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

// Integration coverage for HYBRID against the throwaway OpenSearch container.
//
// The load-bearing test here is `testContainerHybridInlinePipelineIsAcceptedAndScoresAreSane`.
// An inline `search_pipeline` carrying `phase_results_processors` is confirmed by the OpenSearch
// source but has no official documentation example, and the whole zero-provisioning design rests
// on it. Its negative control sits immediately below: without the processor, OpenSearch leaks
// sentinel scores into the results rather than failing, so a test that only asserted "no error"
// would pass against completely broken output.
//
// Every test seeds at least two documents. With a single hit, min_max normalization has
// `max == min` and the resulting score is an implementation detail not worth asserting.

import ballerina/ai;
import ballerina/http;
import ballerina/test;

# The sentinel score OpenSearch uses to delimit hybrid sub-query results, which the
# `normalization-processor` is responsible for stripping. Anything at this magnitude in a result
# means no processor ran.
const float HYBRID_SENTINEL_CEILING = -1.0e9;

# A hybrid search mode over `CONTAINER_DIMENSION`.
#
# + fusion - The fusion configuration, defaulting to inline min_max/arithmetic_mean
# + return - The search mode
isolated function containerHybridMode(HybridFusion fusion = {}) returns HybridSearch => {
    queryMode: ai:HYBRID,
    indexConfig: {dimension: CONTAINER_DIMENSION},
    fusion
};

# Builds a hybrid vector entry.
#
# + id - The logical entry id
# + dense - The dense half
# + indices - The sparse feature indices
# + values - The sparse feature weights
# + language - The `language` metadata value the filter test keys off
# + return - The entry
isolated function hybridVectorEntry(string id, ai:Vector dense, int[] indices, float[] values,
        string language = "en") returns ai:VectorEntry =>
    {
        id,
        embedding: {dense, sparse: {indices, values}},
        chunk: <ai:TextChunk>{content: string `content for ${id}`, metadata: {"language": language}}
    };

# The two entries most tests here seed: one that wins on the dense side, one on the sparse side.
#
# + return - The entries
isolated function hybridSeedEntries() returns ai:VectorEntry[] => [
    // Identical to the query vector, but a weak sparse match.
    hybridVectorEntry("dense-favoured", vec(1.0), [1055], [1.0]),
    // Orthogonal to the query vector, but a strong sparse match.
    hybridVectorEntry("sparse-favoured", vec(0.0, 1.0), [1055], [9.0], "fr")
];

# The hybrid query the tests here issue.
#
# + return - The query
isolated function hybridContainerQuery() returns ai:VectorStoreQuery => {
    embedding: {dense: queryVector(), sparse: {indices: [1055], values: [2.0]}},
    topK: 10
};

// The critical test. It proves two things at once: that OpenSearch 2.19.1 accepts an inline
// `search_pipeline` object carrying `phase_results_processors` on a plain `_search` request, and
// that the processor actually ran -- a score in (0.0, 1.0] is only possible if the sentinels were
// stripped and min_max normalization was applied.
@test:Config {groups: ["docker"]}
isolated function testContainerHybridInlinePipelineIsAcceptedAndScoresAreSane() returns error? {
    string indexName = containerIndexName("hybrid-inline");
    VectorStore store = check newContainerStore(indexName, containerHybridMode());
    check store.add(hybridSeedEntries());

    ai:VectorMatch[] matches = check store.query(hybridContainerQuery());
    test:assertEquals(matches.length(), 2);
    foreach ai:VectorMatch hybridMatch in matches {
        test:assertTrue(hybridMatch.similarityScore > 0.0,
                string `a fused score must be positive; a sentinel leaking through means no ` +
                string `normalization processor ran, got: ${hybridMatch.similarityScore}`);
        test:assertTrue(hybridMatch.similarityScore <= 1.0,
                string `min_max + arithmetic_mean yields (0.0, 1.0], got: ${hybridMatch.similarityScore}`);
    }
    check store.close();
}

// The negative control, without which the test above is vacuous. The same hybrid query sent with
// the `search_pipeline` key stripped must produce the sentinel scores, confirming that the
// positive test's (0.0, 1.0] range is evidence the processor ran rather than a coincidence.
@test:Config {groups: ["docker"]}
isolated function testContainerHybridWithoutPipelineLeaksSentinelScores() returns error? {
    string indexName = containerIndexName("hybrid-nopipe");
    VectorStore store = check newContainerStore(indexName, containerHybridMode());
    check store.add(hybridSeedEntries());
    check store.close();

    // Built by this module, then stripped -- so the only difference from the passing case is the
    // absence of the processor.
    map<json> body = check buildSearchBody(hybridContainerQuery(), containerHybridMode(), {}).ensureType();
    test:assertTrue(body.hasKey("search_pipeline"), "the module must normally send an inline pipeline");
    _ = body.remove("search_pipeline");

    http:Client cl = check rawClient();
    json response = check cl->post(string `/${indexName}/_search`, body);
    map<json> hits = check mapField(check response.ensureType(), "hits");
    json[] hitList = check hits["hits"].ensureType();

    boolean sawSentinel = false;
    foreach json hit in hitList {
        map<json> hitMap = check hit.ensureType();
        float score = check hitMap["_score"].ensureType(float);
        if score < HYBRID_SENTINEL_CEILING {
            sawSentinel = true;
        }
    }
    test:assertTrue(sawSentinel,
            string `a hybrid query with no normalization processor should leak sentinel scores; if ` +
            string `this ever stops holding, the (0, 1] assertion in the inline-pipeline test has ` +
            string `stopped being evidence that the processor ran. Got: ${hitList.toJsonString()}`);
}

// Without this, a silently ignored `combination.parameters` block would pass every other test
// here: the scores would still be in range and still be ordered somehow.
@test:Config {groups: ["docker"]}
isolated function testContainerHybridWeightsShiftRanking() returns error? {
    string indexName = containerIndexName("hybrid-weights");
    VectorStore store = check newContainerStore(indexName, containerHybridMode());
    check store.add(hybridSeedEntries());
    check store.close();

    VectorStore denseHeavy = check newContainerStore(indexName,
            containerHybridMode({denseWeight: 0.9, sparseWeight: 0.1}), {createIndexIfNotExists: false});
    ai:VectorMatch[] denseFirst = check denseHeavy.query(hybridContainerQuery());
    check denseHeavy.close();

    VectorStore sparseHeavy = check newContainerStore(indexName,
            containerHybridMode({denseWeight: 0.1, sparseWeight: 0.9}), {createIndexIfNotExists: false});
    ai:VectorMatch[] sparseFirst = check sparseHeavy.query(hybridContainerQuery());
    check sparseHeavy.close();

    test:assertEquals(denseFirst[0].id, "dense-favoured",
            "weighting the dense sub-query at 0.9 should surface the vector-identical entry");
    test:assertEquals(sparseFirst[0].id, "sparse-favoured",
            "weighting the sparse sub-query at 0.9 should surface the strong token match");
}

// Proves per-sub-query filter duplication is a working substitute for the top-level
// `hybrid.filter`, which is OpenSearch 3.0+ and unavailable on the 2.x this container runs.
@test:Config {groups: ["docker"]}
isolated function testContainerHybridFilterAppliesToBothSubQueries() returns error? {
    string indexName = containerIndexName("hybrid-filter");
    VectorStore store = check newContainerStore(indexName, containerHybridMode());
    // "sparse-favoured" is tagged `fr` and would otherwise win the sparse sub-query outright.
    check store.add(hybridSeedEntries());

    ai:VectorMatch[] matches = check store.query({
        embedding: {dense: queryVector(), sparse: {indices: [1055], values: [2.0]}},
        filters: {filters: [{key: "language", operator: ai:EQUAL, value: "en"}]},
        topK: 10
    });
    test:assertEquals(matches.length(), 1,
            string `the French entry must be filtered out of both sub-queries, got: ${matches.length()}`);
    test:assertEquals(matches[0].id, "dense-favoured");
    check store.close();
}

// Proves the ?search_pipeline= escape hatch works end to end, including the transport change.
@test:Config {groups: ["docker"]}
isolated function testContainerHybridNamedPipelineViaQueryParameter() returns error? {
    string indexName = containerIndexName("hybrid-named");
    string pipelineName = string `${indexName}-norm`;
    http:Client cl = check rawClient();
    json _ = check cl->put(string `/_search/pipeline/${pipelineName}`, {
        "phase_results_processors": [
            {
                "normalization-processor": {
                    "normalization": {"technique": "min_max"},
                    "combination": {"technique": "arithmetic_mean"}
                }
            }
        ]
    });

    VectorStore store = check newContainerStore(indexName, containerHybridMode({name: pipelineName}));
    check store.add(hybridSeedEntries());
    ai:VectorMatch[] matches = check store.query(hybridContainerQuery());
    test:assertEquals(matches.length(), 2);
    foreach ai:VectorMatch hybridMatch in matches {
        test:assertTrue(hybridMatch.similarityScore > 0.0 && hybridMatch.similarityScore <= 1.0,
                string `a named pipeline should normalize just as the inline one does, got: ` +
                string `${hybridMatch.similarityScore}`);
    }
    check store.close();
    json _ = check cl->delete(string `/_search/pipeline/${pipelineName}`);
}

@test:Config {groups: ["docker"]}
isolated function testContainerHybridWithoutEmbeddingReturnsEveryEntry() returns error? {
    string indexName = containerIndexName("hybrid-noembed");
    VectorStore store = check newContainerStore(indexName, containerHybridMode());
    check store.add(hybridSeedEntries());

    ai:VectorMatch[] matches = check store.query({
        filters: {filters: [{key: "language", operator: ai:EQUAL, value: "en"}]},
        topK: 10
    });
    test:assertEquals(matches.length(), 1);
    test:assertEquals(matches[0].similarityScore, 0.0,
            "a query with no embedding scores nothing, so no pipeline is sent and the score is 0.0");
    check store.close();
}

@test:Config {groups: ["docker"]}
isolated function testContainerHybridRoundTripsBothHalves() returns error? {
    string indexName = containerIndexName("hybrid-roundtrip");
    VectorStore store = check newContainerStore(indexName, containerHybridMode());
    check store.add(hybridSeedEntries());

    ai:VectorMatch[] matches = check store.query(hybridContainerQuery());
    ai:Embedding embedding = matches[0].embedding;
    if embedding !is ai:HybridVector {
        test:assertFail("a HYBRID store must return an ai:HybridVector");
    }
    test:assertEquals(embedding.dense.length(), CONTAINER_DIMENSION);
    test:assertEquals(embedding.sparse.indices, [1055]);
    check store.close();
}

// --- reciprocal rank fusion ----------------------------------------------------------------------

// The RRF counterpart to `testContainerHybridInlinePipelineIsAcceptedAndScoresAreSane`, and it
// proves the same two things for a different processor: that OpenSearch 2.19.1 accepts an inline
// `score-ranker-processor` and that it actually ran. The score range is the evidence -- RRF sums
// `1 / (rankConstant + rank)` across two sub-queries, so with the default constant of 60 no score
// can exceed `2 / 61`, and any sentinel leaking through would be hugely negative.
@test:Config {groups: ["docker"]}
isolated function testContainerRrfFusionIsAcceptedAndScoresAreSane() returns error? {
    string indexName = containerIndexName("hybrid-rrf");
    VectorStore store = check newContainerStore(indexName, containerHybridMode({technique: RRF}));
    check store.add(hybridSeedEntries());

    ai:VectorMatch[] matches = check store.query(hybridContainerQuery());
    test:assertEquals(matches.length(), 2);
    foreach ai:VectorMatch rrfMatch in matches {
        test:assertTrue(rrfMatch.similarityScore > 0.0,
                string `an RRF score is a sum of positive reciprocals; a sentinel means no ` +
                string `processor ran, got: ${rrfMatch.similarityScore}`);
        test:assertTrue(rrfMatch.similarityScore <= 2.0 / 61.0,
                string `two sub-queries at rank 1 cap the RRF score at 2/(60+1), got: ` +
                string `${rrfMatch.similarityScore}`);
    }
    check store.close();
}

// The counterpart to the weights test above, and the reason `RrfFusion` carries no settings. The
// processor must be sent as the technique alone: neural-search reads `rank_constant` from
// `combination.parameters` through 3.0 and from `combination` itself from 3.1, where the old
// location became a hard error. Naming it in either place breaks on one side of that line.
@test:Config {groups: ["docker"]}
isolated function testContainerRrfSendsTechniqueAlone() returns error? {
    json body = check buildSearchBody(hybridContainerQuery(), containerHybridMode({technique: RRF}), {});
    map<json> bodyMap = check body.ensureType();
    string pipeline = bodyMap["search_pipeline"].toJsonString();
    test:assertFalse(pipeline.includes("rank_constant"),
            string `a rank_constant has no portable wire shape and must not be sent, got: ${pipeline}`);
    test:assertFalse(pipeline.includes("parameters"),
            string `a 'parameters' map is rejected outright by neural-search 3.1+, got: ${pipeline}`);
    test:assertTrue(pipeline.includes("rrf"));
}

// --- ef_search on the wire -------------------------------------------------------------------------

// `method_parameters` is 2.16+, and a rejected one fails the whole search. This proves the
// container accepts the object this module builds rather than merely that the JSON looks right.
@test:Config {groups: ["docker"]}
isolated function testContainerEfSearchIsAcceptedByTheDenseSubQuery() returns error? {
    string indexName = containerIndexName("hybrid-efsearch");
    HybridSearch mode = {
        queryMode: ai:HYBRID,
        indexConfig: {dimension: CONTAINER_DIMENSION},
        efSearch: 512
    };
    VectorStore store = check newContainerStore(indexName, mode);
    check store.add(hybridSeedEntries());

    ai:VectorMatch[] matches = check store.query(hybridContainerQuery());
    test:assertEquals(matches.length(), 2, "raising ef_search must not change which documents match");
    check store.close();
}
