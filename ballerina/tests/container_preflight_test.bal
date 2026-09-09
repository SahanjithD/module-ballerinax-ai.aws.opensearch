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

// The construction-time verification, exercised against a real cluster.
//
// The unit tests in `preflight_test.bal` cover the comparison logic against hand-written mappings.
// What only a live cluster can prove is that `GET /` and `GET /<index>/_mapping` return what this
// module expects to read out of them -- and, in the mapping's case, that a mapping OpenSearch
// itself stored still satisfies the check that a mapping this module wrote produced. A verification
// that false-positives on the module's own index would be worse than no verification.

import ballerina/ai;
import ballerina/test;

// The load-bearing case: the version this module reads out of `GET /` has to be the container's
// actual version, or every floor comparison is meaningless.
@test:Config {groups: ["docker"]}
isolated function testContainerClusterVersionIsReadable() returns error? {
    OpenSearchTransport transport = check new (containerUrl, CONTAINER_REGION,
            {deploymentType: MANAGED_DOMAIN, auth: containerAuth}, {}, {});
    string? reported = check transport.clusterVersion();
    if reported !is string {
        test:assertFail("'GET /' must report a version on a managed domain");
    }
    int[]? parsed = parseVersion(reported);
    if parsed !is int[] {
        test:assertFail(string `the reported version must parse, got: ${reported}`);
    }
    test:assertTrue(isAtLeast(parsed, [2, 11]),
            string `the container image is 2.19.1; if this fails the version read is wrong, got: ${reported}`);
    check transport.close();
}

// Verification must be silent on an index this module created, in every mode. This is the test
// that would catch a check strict enough to reject the module's own output -- OpenSearch stores a
// mapping back with fields the create request never sent, so the two are not identical documents.
@test:Config {groups: ["docker"]}
isolated function testContainerVerificationAcceptsTheModulesOwnIndex() returns error? {
    string denseIndex = containerIndexName("verify-dense");
    VectorStore dense = check newContainerStore(denseIndex, containerDenseMode());
    check dense.close();
    // A second store over the same index takes the `!created` branch, so verification actually runs.
    VectorStore denseAgain = check newContainerStore(denseIndex, containerDenseMode(),
            {createIndexIfNotExists: false});
    check denseAgain.close();

    string sparseIndex = containerIndexName("verify-sparse");
    VectorStore sparse = check newSparseContainerStore(sparseIndex);
    check sparse.close();
    VectorStore sparseAgain = check newContainerStore(sparseIndex, containerSparseMode(),
            {createIndexIfNotExists: false});
    check sparseAgain.close();

    string hybridIndex = containerIndexName("verify-hybrid");
    VectorStore hybrid = check newContainerStore(hybridIndex, containerHybridMode());
    check hybrid.close();
    VectorStore hybridAgain = check newContainerStore(hybridIndex, containerHybridMode(),
            {createIndexIfNotExists: false});
    check hybridAgain.close();
}

// The `createIndexIfNotExists: false` hazard, caught at construction instead of becoming an index
// that accepts every write and fails every query.
@test:Config {groups: ["docker"]}
isolated function testContainerVerificationRejectsAMissingIndex() {
    string indexName = containerIndexName("verify-missing");
    VectorStore|ai:Error result = newContainerStore(indexName, containerDenseMode(),
            {createIndexIfNotExists: false});
    if result !is ai:Error {
        test:assertFail("a store pointed at an index that does not exist must not construct");
    }
    test:assertTrue(result.message().includes("auto_create_index"),
            string `the error should explain why writing anyway would not fail, got: ${result.message()}`);
}

// The mode-mismatch case: a SPARSE store aimed at an index built for DENSE. Without verification
// this writes successfully forever and fails every query with a bare 400.
@test:Config {groups: ["docker"]}
isolated function testContainerVerificationRejectsAModeMismatch() returns error? {
    string indexName = containerIndexName("verify-mismatch");
    VectorStore dense = check newContainerStore(indexName, containerDenseMode());
    check dense.close();

    VectorStore|ai:Error result = newContainerStore(indexName, containerSparseMode(),
            {createIndexIfNotExists: false});
    if result !is ai:Error {
        test:assertFail("a SPARSE store aimed at a dense index must not construct");
    }
    test:assertTrue(result.message().includes("sparse_embedding"),
            string `the error should name the field the index does not declare, got: ${result.message()}`);
}

// The drifted-dimension case: the embedding model changed under a store that was not rebuilt.
@test:Config {groups: ["docker"]}
isolated function testContainerVerificationRejectsADriftedDimension() returns error? {
    string indexName = containerIndexName("verify-dimension");
    VectorStore store = check newContainerStore(indexName, containerDenseMode());
    check store.close();

    DenseSearch widened = {queryMode: ai:DENSE, indexConfig: {dimension: CONTAINER_DIMENSION + 1}};
    VectorStore|ai:Error result = newContainerStore(indexName, widened, {createIndexIfNotExists: false});
    if result !is ai:Error {
        test:assertFail("a dimension that disagrees with the index must not construct");
    }
    test:assertTrue(result.message().includes(CONTAINER_DIMENSION.toString()),
            string `the error should name the index's own dimension, got: ${result.message()}`);
}

// A renamed `vectorFieldName` reads and writes a field that is not there. Writes succeed, and
// every query returns nothing rather than failing.
@test:Config {groups: ["docker"]}
isolated function testContainerVerificationRejectsARenamedVectorField() returns error? {
    string indexName = containerIndexName("verify-renamed");
    VectorStore store = check newContainerStore(indexName, containerDenseMode());
    check store.close();

    DenseSearch renamed = {
        queryMode: ai:DENSE,
        indexConfig: {dimension: CONTAINER_DIMENSION},
        vectorFieldName: "vector"
    };
    VectorStore|ai:Error result = newContainerStore(indexName, renamed, {createIndexIfNotExists: false});
    if result !is ai:Error {
        test:assertFail("a vector field name the index does not declare must not construct");
    }
    test:assertTrue(result.message().includes("'vector'"),
            string `the error should name the missing field, got: ${result.message()}`);
}

// Opting out has to actually opt out, or the setting is decoration.
@test:Config {groups: ["docker"]}
isolated function testContainerVerificationCanBeSwitchedOff() returns error? {
    string indexName = containerIndexName("verify-off");
    VectorStore store = check newContainerStore(indexName, containerSparseMode(),
            {createIndexIfNotExists: false, verifyOnInit: false});
    check store.close();
}
