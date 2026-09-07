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

// Unit tests for the `ai:SparseVector` <-> `rank_features` conversion. Every bound asserted here
// is Lucene's; the point of checking them client-side is that a violation names the offending
// term instead of failing a whole `_bulk` request or arriving as an opaque shard exception.

import ballerina/ai;
import ballerina/test;

// The bound that applies at index time: Lucene imposes no ceiling on a stored weight.
final float NO_MAX_WEIGHT = float:Infinity;

isolated function assertErrorMentions(ai:Error err, string fragment) {
    test:assertTrue(err.message().includes(fragment),
            string `the message should mention '${fragment}', got: ${err.message()}`);
}

// --- toSparseTokenMap: the happy path ---------------------------------------------------------

@test:Config
isolated function testSparseTokenMapStringifiesIndices() returns error? {
    map<float> tokens = check toSparseTokenMap({indices: [1055, 2048], values: [5.5, 1.25]},
            "Entry 'a'", NO_MAX_WEIGHT);
    test:assertEquals(tokens, {"1055": 5.5, "2048": 1.25});
}

@test:Config
isolated function testSparseTokenMapEmptyVectorProducesEmptyMap() returns error? {
    map<float> tokens = check toSparseTokenMap({indices: [], values: []}, "Entry 'a'", NO_MAX_WEIGHT);
    test:assertEquals(tokens.length(), 0);
}

// --- toSparseTokenMap: the rejections ---------------------------------------------------------

@test:Config
isolated function testSparseTokenMapRejectsLengthMismatch() {
    map<float>|ai:Error result = toSparseTokenMap({indices: [1, 2, 3], values: [0.1, 0.2]},
            "Entry 'a'", NO_MAX_WEIGHT);
    if result !is ai:Error {
        test:assertFail("an indices/values length mismatch must be rejected");
    }
    assertErrorMentions(result, "3 vs 2");
}

@test:Config
isolated function testSparseTokenMapRejectsNegativeIndex() {
    // A negative index stringifies to "-1", a legal feature name, so it would otherwise be stored
    // rather than rejected -- and would never match a query built from a non-negative index.
    map<float>|ai:Error result = toSparseTokenMap({indices: [-1], values: [0.5]},
            "Entry 'a'", NO_MAX_WEIGHT);
    if result !is ai:Error {
        test:assertFail("a negative sparse index must be rejected");
    }
    assertErrorMentions(result, "non-negative");
}

@test:Config
isolated function testSparseTokenMapRejectsZeroWeight() {
    map<float>|ai:Error result = toSparseTokenMap({indices: [7], values: [0.0]},
            "Entry 'a'", NO_MAX_WEIGHT);
    if result !is ai:Error {
        test:assertFail("a zero weight is rejected by Lucene at index time and must be caught here");
    }
    assertErrorMentions(result, "Drop the term");
}

@test:Config
isolated function testSparseTokenMapRejectsNegativeWeight() {
    map<float>|ai:Error result = toSparseTokenMap({indices: [7], values: [-0.5]},
            "Entry 'a'", NO_MAX_WEIGHT);
    test:assertTrue(result is ai:Error, "a negative weight must be rejected");
}

@test:Config
isolated function testSparseTokenMapRejectsSubnormalWeight() {
    // Below Float.MIN_NORMAL but above zero: Lucene rejects this too, and it could not survive
    // `rank_features`' nine-significant-bit storage in any case.
    map<float>|ai:Error result = toSparseTokenMap({indices: [7], values: [1.0e-45]},
            "Entry 'a'", NO_MAX_WEIGHT);
    test:assertTrue(result is ai:Error, "a subnormal weight must be rejected");
}

@test:Config
isolated function testSparseTokenMapRejectsNaNWeight() {
    // Every ordering comparison against NaN is false, so the finiteness check has to come before
    // the range checks or this would slip through to the server.
    map<float>|ai:Error result = toSparseTokenMap({indices: [7], values: [float:NaN]},
            "Entry 'a'", NO_MAX_WEIGHT);
    if result !is ai:Error {
        test:assertFail("a NaN weight must be rejected");
    }
    assertErrorMentions(result, "non-finite");
}

@test:Config
isolated function testSparseTokenMapRejectsInfiniteWeight() {
    map<float>|ai:Error result = toSparseTokenMap({indices: [7], values: [float:Infinity]},
            "Entry 'a'", NO_MAX_WEIGHT);
    if result !is ai:Error {
        test:assertFail("an infinite weight must be rejected");
    }
    assertErrorMentions(result, "non-finite");
}

@test:Config
isolated function testSparseTokenMapRejectsDuplicateIndex() {
    map<float>|ai:Error result = toSparseTokenMap({indices: [7, 7], values: [0.5, 0.6]},
            "Entry 'a'", NO_MAX_WEIGHT);
    if result !is ai:Error {
        test:assertFail("a repeated sparse index must be rejected");
    }
    assertErrorMentions(result, "repeats sparse index 7");
}

// --- toSparseTokenMap: the query-time weight ceiling ------------------------------------------

@test:Config
isolated function testSparseTokenMapAcceptsExactlyMinRankFeatureValue() returns error? {
    map<float> tokens = check toSparseTokenMap({indices: [7], values: [MIN_RANK_FEATURE_VALUE]},
            "Entry 'a'", NO_MAX_WEIGHT);
    test:assertEquals(tokens["7"], MIN_RANK_FEATURE_VALUE);
}

@test:Config
isolated function testSparseTokenMapAcceptsExactlyMaxQueryTokenWeight() returns error? {
    map<float> tokens = check toSparseTokenMap({indices: [7], values: [MAX_QUERY_TOKEN_WEIGHT]},
            "The query", MAX_QUERY_TOKEN_WEIGHT);
    test:assertEquals(tokens["7"], 64.0);
}

@test:Config
isolated function testSparseTokenMapRejectsWeightAboveQueryCeiling() {
    map<float>|ai:Error result = toSparseTokenMap({indices: [7], values: [64.1]},
            "The query", MAX_QUERY_TOKEN_WEIGHT);
    if result !is ai:Error {
        test:assertFail("a query weight above 64 is rejected by Lucene and must be caught here");
    }
    assertErrorMentions(result, "(0, 64]");
}

@test:Config
isolated function testSparseTokenMapAllowsLargeWeightAtIndexTime() returns error? {
    // The (0, 64] ceiling is a query-time bound only; a stored weight has none.
    map<float> tokens = check toSparseTokenMap({indices: [7], values: [1000.0]},
            "Entry 'a'", NO_MAX_WEIGHT);
    test:assertEquals(tokens["7"], 1000.0);
}

// --- sparseFromRankFeatures ------------------------------------------------------------------

@test:Config
isolated function testSparseFromRankFeaturesSortsByNumericIndex() returns error? {
    // Pins both that an order is imposed at all -- map key order is not guaranteed -- and that it
    // is numeric rather than lexicographic, where "10" would sort before "2".
    ai:SparseVector sparse = check sparseFromRankFeatures({"9": 0.1, "2": 0.2, "10": 0.3}, "a");
    test:assertEquals(sparse.indices, [2, 9, 10]);
    test:assertEquals(sparse.values, [0.2, 0.1, 0.3]);
}

@test:Config
isolated function testSparseFromRankFeaturesEmptyMap() returns error? {
    ai:SparseVector sparse = check sparseFromRankFeatures({}, "a");
    test:assertEquals(sparse.indices, []);
    test:assertEquals(sparse.values, []);
}

@test:Config
isolated function testSparseFromRankFeaturesAcceptsIntegerWeight() returns error? {
    // A whole-numbered weight can come back from OpenSearch as a JSON integer.
    ai:SparseVector sparse = check sparseFromRankFeatures({"3": 5}, "a");
    test:assertEquals(sparse.values, [5.0]);
}

@test:Config
isolated function testSparseFromRankFeaturesRejectsTextToken() {
    // A real sparse encoder can emit text tokens. They cannot round-trip through an `int[]`, so
    // they are reported rather than silently dropped.
    ai:SparseVector|ai:Error result = sparseFromRankFeatures({"##ing": 0.4}, "a");
    if result !is ai:Error {
        test:assertFail("a non-integer token cannot be represented and must be reported");
    }
    assertErrorMentions(result, "'##ing' is not an integer index");
}

@test:Config
isolated function testSparseFromRankFeaturesRejectsNonNumericWeight() {
    ai:SparseVector|ai:Error result = sparseFromRankFeatures({"3": "high"}, "a");
    test:assertTrue(result is ai:Error, "a non-numeric stored weight must be reported");
}

@test:Config
isolated function testSparseTokenMapRoundTripsThroughRankFeatures() returns error? {
    ai:SparseVector original = {indices: [2, 9, 10], values: [0.2, 0.1, 0.3]};
    map<float> tokens = check toSparseTokenMap(original, "Entry 'a'", NO_MAX_WEIGHT);
    ai:SparseVector restored = check sparseFromRankFeatures(tokens, "a");
    test:assertEquals(restored, original, "an index-ordered sparse vector should survive a round trip");
}
