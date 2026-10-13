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

// Construction-time verification against a real cluster. It must accept the module's own index,
// whose stored mapping has fields the create request never sent.

import ballerina/ai;
import ballerina/test;

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

@test:Config {groups: ["docker"]}
isolated function testContainerVerificationAcceptsTheModulesOwnIndex() returns error? {
    string denseIndex = containerIndexName("verify-dense");
    VectorStore dense = check newContainerStore(denseIndex, containerDenseMode());
    check dense.close();
    // A second store over the same index skips creation, so verification runs.
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

@test:Config {groups: ["docker"]}
isolated function testContainerVerificationCanBeSwitchedOff() returns error? {
    string indexName = containerIndexName("verify-off");
    VectorStore store = check newContainerStore(indexName, containerSparseMode(),
            {createIndexIfNotExists: false, verifyOnInit: false});
    check store.close();
}
