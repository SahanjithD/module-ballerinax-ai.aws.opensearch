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

// Unit tests for the construction-time verification in `preflight.bal`. The version parsing, the
// floor comparison, the requirement set a `SearchMode` implies, and the mapping comparison are all
// pure, so the only part that needs a cluster is the two reads that feed them.

import ballerina/ai;
import ballerina/test;

// --- version parsing ---------------------------------------------------------------------------

@test:Config
isolated function testParseVersionReadsThreeComponents() {
    test:assertEquals(parseVersion("2.19.1"), [2, 19, 1]);
}

@test:Config
isolated function testParseVersionReadsTwoComponents() {
    test:assertEquals(parseVersion("2.11"), [2, 11]);
}

// A build reporting a pre-release suffix must parse to the release it precedes rather than failing
// and downgrading the whole check to a warning.
@test:Config
isolated function testParseVersionStopsAtPreReleaseSuffix() {
    test:assertEquals(parseVersion("3.0.0-SNAPSHOT"), [3, 0, 0]);
    test:assertEquals(parseVersion("2.11.0-rc1"), [2, 11, 0]);
}

@test:Config
isolated function testParseVersionRejectsNonNumeric() {
    test:assertTrue(parseVersion("opensearch") is ());
    test:assertTrue(parseVersion("") is ());
}

// --- floor comparison --------------------------------------------------------------------------

@test:Config
isolated function testIsAtLeastComparesMinorWithinSameMajor() {
    test:assertTrue(isAtLeast([2, 19, 1], [2, 11]));
    test:assertTrue(isAtLeast([2, 11, 0], [2, 11]));
    test:assertFalse(isAtLeast([2, 9, 0], [2, 11]));
}

// The comparison must not be lexicographic on the string: 2.9 is older than 2.11, which is exactly
// the case a naive check gets backwards.
@test:Config
isolated function testIsAtLeastDoesNotCompareVersionsAsStrings() {
    test:assertFalse(isAtLeast([2, 9], [2, 11]),
            "2.9 predates 2.11; comparing the components as text would reverse this");
}

@test:Config
isolated function testIsAtLeastComparesMajorFirst() {
    test:assertTrue(isAtLeast([3, 0, 0], [2, 19]));
    test:assertFalse(isAtLeast([1, 3], [2, 11]));
}

// A version shorter than the floor pads with zeros rather than being rejected.
@test:Config
isolated function testIsAtLeastPadsShortVersion() {
    test:assertTrue(isAtLeast([3], [2, 11]));
    test:assertFalse(isAtLeast([2], [2, 11]));
}

// --- the floors a search mode implies ------------------------------------------------------------

@test:Config
isolated function testDenseModeImpliesNoVersionFloor() {
    test:assertEquals(versionRequirements(denseMode()).length(), 0,
            "DENSE needs no plugin and no compound query, so nothing gates it");
}

@test:Config
isolated function testDenseWithEfSearchImpliesMethodParametersFloor() {
    VersionRequirement[] requirements = versionRequirements(
            {queryMode: ai:DENSE, indexConfig: {dimension: 3}, efSearch: 512});
    test:assertEquals(requirements.length(), 1);
    test:assertEquals(requirements[0].floor, [2, 16]);
}

@test:Config
isolated function testSparseModeImpliesRawQueryTokensFloor() {
    VersionRequirement[] requirements = versionRequirements(sparseMode());
    test:assertEquals(requirements.length(), 1);
    test:assertEquals(requirements[0].floor, [2, 14]);
}

@test:Config
isolated function testTwoPhaseAccelerationRaisesTheSparseFloor() {
    SparseSearch mode = {queryMode: ai:SPARSE, twoPhaseAcceleration: {}};
    VersionRequirement[] requirements = versionRequirements(mode);
    test:assertEquals(requirements[0].floor, [2, 15],
            "the processor's 2.15 floor outranks raw query_tokens' 2.14 and must be reported first");
}

// The ordering is the whole point of sorting: a caller told to upgrade to 2.11 for the `hybrid`
// query would upgrade and then immediately hit the 2.19 floor that RRF carries.
@test:Config
isolated function testHybridReportsTheHighestFloorFirst() {
    HybridSearch mode = hybridMode({technique: RRF});
    VersionRequirement[] requirements = versionRequirements(mode);
    test:assertEquals(requirements[0].floor, [2, 19],
            "RRF's floor outranks both the hybrid query's 2.11 and raw query_tokens' 2.14");
    test:assertTrue(requirements[0].feature.includes("score-ranker-processor"));
}

@test:Config
isolated function testHybridWithInlineNormalizationFloorsAtRawQueryTokens() {
    VersionRequirement[] requirements = versionRequirements(hybridMode());
    test:assertEquals(requirements[0].floor, [2, 14]);
    // The `hybrid` query's own 2.11 floor is still carried, just outranked.
    test:assertEquals(requirements[requirements.length() - 1].floor, [2, 11]);
}

// --- mapping verification ------------------------------------------------------------------------

isolated function denseMapping(int dimension = 3) returns map<json> => {
    "embedding": {"type": "knn_vector", "dimension": dimension},
    "content": {"type": "text"}
};

@test:Config
isolated function testMappingVerificationAcceptsAMatchingDenseField() {
    ai:Error? result = verifyMappedField(denseMapping(), "idx", "embedding", "knn_vector", 3);
    test:assertTrue(result is ());
}

@test:Config
isolated function testMappingVerificationRejectsAMissingField() {
    ai:Error? result = verifyMappedField(denseMapping(), "idx", "renamed", "knn_vector", 3);
    if result !is ai:Error {
        test:assertFail("a vector field the index does not declare must be reported");
    }
    test:assertTrue(result.message().includes("'renamed'"),
            string `the error should name the missing field, got: ${result.message()}`);
}

// The `SPARSE`-store-aimed-at-a-dense-index case. Writes would succeed forever and every query
// would fail with a bare 400.
@test:Config
isolated function testMappingVerificationRejectsAWrongFieldType() {
    map<json> properties = {"sparse_embedding": {"type": "knn_vector", "dimension": 3}};
    ai:Error? result = verifyMappedField(properties, "idx", "sparse_embedding", "rank_features", ());
    if result !is ai:Error {
        test:assertFail("a field mapped as the wrong type must be reported");
    }
    test:assertTrue(result.message().includes("rank_features") && result.message().includes("knn_vector"),
            string `the error should name both types, got: ${result.message()}`);
}

// The auto-created-index case: `action.auto_create_index` maps a vector as a plain float array,
// which comes back as a `float` type rather than `knn_vector`.
@test:Config
isolated function testMappingVerificationRejectsAnAutoCreatedFloatField() {
    map<json> properties = {"embedding": {"type": "float"}};
    ai:Error? result = verifyMappedField(properties, "idx", "embedding", "knn_vector", 3);
    if result !is ai:Error {
        test:assertFail("a vector auto-mapped as a plain float array must be reported");
    }
    test:assertTrue(result.message().includes("auto_create_index"),
            string `the error should point at the likely cause, got: ${result.message()}`);
}

@test:Config
isolated function testMappingVerificationRejectsADriftedDimension() {
    ai:Error? result = verifyMappedField(denseMapping(1536), "idx", "embedding", "knn_vector", 768);
    if result !is ai:Error {
        test:assertFail("a dimension that disagrees with the index must be reported");
    }
    test:assertTrue(result.message().includes("1536") && result.message().includes("768"),
            string `the error should name both dimensions, got: ${result.message()}`);
}

// `rank_features` carries no dimension, so there is nothing to compare and its absence is not a
// disagreement.
@test:Config
isolated function testMappingVerificationIgnoresDimensionForSparseFields() {
    map<json> properties = {"sparse_embedding": {"type": "rank_features"}};
    ai:Error? result = verifyMappedField(properties, "idx", "sparse_embedding", "rank_features", ());
    test:assertTrue(result is ());
}

// Only the vector fields are checked. Pointing this module at a pre-existing index whose content,
// id and metadata fields look nothing like the ones it would have written is supported.
@test:Config
isolated function testMappingVerificationIgnoresNonVectorFields() {
    map<json> properties = {"embedding": {"type": "knn_vector", "dimension": 3}};
    ai:Error? result = verifyMappedField(properties, "idx", "embedding", "knn_vector", 3);
    test:assertTrue(result is (), "an index with no content or metadata mapping is still usable");
}
