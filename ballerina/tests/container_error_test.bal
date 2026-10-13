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

// `mapErrorResponse` and `executeWithRetry` against real error responses, whose shape a stub would
// likely get wrong.

import ballerina/ai;
import ballerina/test;

// Verification is off too, since it would refuse to construct the store over a missing index.
# A store over an index that does not exist and will not be created.
#
# + indexName - The index that will not exist
# + return - The store, or an `ai:Error`
isolated function storeOnMissingIndex(string indexName) returns VectorStore|ai:Error =>
    newContainerStore(indexName, containerDenseMode(),
            {createIndexIfNotExists: false, verifyOnInit: false});

@test:Config {groups: ["docker"]}
isolated function testContainerQueryOnMissingIndexMapsTo404() returns error? {
    string indexName = containerIndexName("err-404");
    VectorStore store = check storeOnMissingIndex(indexName);

    ai:VectorMatch[]|ai:Error result = store.query({embedding: queryVector(), topK: 1});
    if result !is ai:Error {
        test:assertFail("querying a missing index should fail");
    }
    test:assertEquals(result.detail()["status"], 404, "the HTTP status should be carried on the error");
    test:assertEquals(result.detail()["openSearchErrorType"], "index_not_found_exception",
            "OpenSearch's own error classification should be carried through");
    test:assertTrue(result.message().includes("createIndexIfNotExists"),
            string `the 404 hint should point at the relevant setting, got: ${result.message()}`);
    check store.close();
}

@test:Config {groups: ["docker"]}
isolated function testContainerDeleteOnMissingIndexIsReported() returns error? {
    string indexName = containerIndexName("err-del404");
    VectorStore store = check storeOnMissingIndex(indexName);

    ai:Error? result = store.delete("anything");
    if result !is ai:Error {
        test:assertFail("deleting from a missing index should fail");
    }
    // Both a missing index and a missing document are per-item 404s; only the index one carries
    // an `error` object and counts as a failure.
    test:assertTrue(result.message().includes("no such index"),
            string `the server's reason should be preserved, got: ${result.message()}`);
    check store.close();
}

@test:Config {groups: ["docker"]}
isolated function testContainerAddOnMissingIndexAutoCreatesAnUnusableIndex() returns error? {
    string indexName = containerIndexName("err-autocreate");
    VectorStore store = check storeOnMissingIndex(indexName);

    // `action.auto_create_index` creates an index from the document's shape: no `index.knn` and
    // a plain `float` vector, so writes succeed and every k-NN query fails.
    check store.add([entry("orphan", queryVector())]);
    test:assertTrue(check rawIndexExists(indexName), "the server should have auto-created the index");

    map<json> settings = check indexSettings(indexName);
    test:assertFalse(settings.hasKey("knn"), "an auto-created index has no 'index.knn' setting");

    map<json> mappings = check indexMapping(indexName);
    map<json> properties = check mapField(mappings, "properties");
    map<json> vectorField = check mapField(properties, "embedding");
    test:assertEquals(vectorField["type"], "float",
            "an auto-created index maps the vector as a plain float array, not a 'knn_vector'");

    ai:VectorMatch[]|ai:Error queryResult = store.query({embedding: queryVector(), topK: 1});
    if queryResult !is ai:Error {
        test:assertFail("a k-NN query against an auto-created index should fail");
    }
    test:assertEquals(queryResult.detail()["status"], 400);
    check store.close();
}

@test:Config {groups: ["docker"]}
isolated function testContainerDimensionMismatchIsReported() returns error? {
    string indexName = containerIndexName("err-dim");
    VectorStore store = check newContainerStore(indexName);

    ai:Error? result = store.add([
        {id: "short", embedding: [0.1, 0.2], chunk: <ai:TextChunk>{content: "two dimensions"}}
    ]);
    if result !is ai:Error {
        test:assertFail("a dimension mismatch should fail the 'add'");
    }
    test:assertTrue(result.message().includes("short"),
            string `the error should name the failing entry, got: ${result.message()}`);
    test:assertTrue(result.message().includes("dimension") || result.message().includes("vector"),
            string `the server's reason should be preserved, got: ${result.message()}`);
    // The useful numbers are only in the item's nested `caused_by`.
    test:assertTrue(result.message().includes("Expected: 4") && result.message().includes("Given: 2"),
            string `the nested cause carries the only actionable detail, got: ${result.message()}`);
    check store.close();
}

// A shard-level failure says only `all shards failed`; the explanation is in `root_cause[0]`.
@test:Config {groups: ["docker"]}
isolated function testContainerQueryDimensionMismatchSurfacesRootCause() returns error? {
    string indexName = containerIndexName("err-qdim");
    VectorStore store = check newContainerStore(indexName);
    check store.add([entry("only", queryVector())]);

    ai:Vector wrongDimension = [0.1, 0.2];
    ai:VectorMatch[]|ai:Error result = store.query({embedding: wrongDimension, topK: 1});
    if result !is ai:Error {
        test:assertFail("a query vector of the wrong dimension should fail");
    }
    test:assertEquals(result.detail()["status"], 400);
    test:assertTrue(result.message().includes("all shards failed"),
            string `the server's own top-level reason should still be reported, got: ${result.message()}`);
    test:assertTrue(result.message().includes("Dimension should be: 4"),
            string `the nested root cause must reach the caller, got: ${result.message()}`);
    // The wrapper's type is kept, with the specific one beside it.
    test:assertEquals(result.detail()["openSearchErrorType"], "search_phase_execution_exception");
    test:assertEquals(result.detail()["openSearchRootCauseType"], "query_shard_exception");
    check store.close();
}

// A 404's `root_cause` repeats the outer reason, which must not be appended twice.
@test:Config {groups: ["docker"]}
isolated function testContainerNonWrapperErrorIsNotDuplicated() returns error? {
    string indexName = containerIndexName("err-nodup");
    VectorStore store = check storeOnMissingIndex(indexName);

    ai:VectorMatch[]|ai:Error result = store.query({embedding: queryVector(), topK: 1});
    if result !is ai:Error {
        test:assertFail("querying a missing index should fail");
    }
    string message = result.message();
    int firstOccurrence = <int>message.indexOf("no such index");
    test:assertEquals(message.lastIndexOf("no such index"), firstOccurrence,
            string `a self-referential root cause should not be appended to itself, got: ${message}`);
    test:assertEquals(result.detail()["openSearchRootCauseType"], (),
            "there is no nested classification to carry on a non-wrapper error");
    check store.close();
}

// The failing entry is named by the caller's id, and successful entries are left out.
@test:Config {groups: ["docker"]}
isolated function testContainerBulkFailureNamesTheCallersEntry() returns error? {
    string indexName = containerIndexName("err-attrib");
    VectorStore store = check newContainerStore(indexName);

    ai:Error? result = store.add([
        entry("good-0", queryVector()),
        entry("bad-1", [0.1, 0.2]),
        entry("good-2", queryVector())
    ]);
    if result !is ai:Error {
        test:assertFail("one bad entry should fail the 'add'");
    }
    string message = result.message();
    test:assertTrue(message.includes("Failed to index 1 of 3 entries"),
            string `only the one bad entry should be counted, got: ${message}`);
    test:assertTrue(message.includes("'bad-1'"),
            string `the failing entry should be named by the caller's own id, got: ${message}`);
    test:assertFalse(message.includes("good-0") || message.includes("good-2"),
            string `entries that succeeded should not appear, got: ${message}`);
    check store.close();
}

@test:Config {groups: ["docker"]}
isolated function testContainerCreateIndexOnBadMappingIsReported() returns error? {
    string indexName = containerIndexName("err-mapping");
    // The module sets no dimension ceiling, so the server's rejection must fail construction.
    DenseSearch configMode = {queryMode: ai:DENSE, indexConfig: {dimension: 99999}};
    VectorStore|ai:Error result = newContainerStore(indexName, configMode);

    if result !is ai:Error {
        check result.close();
        test:assertFail("an out-of-range dimension should fail index creation");
    }
    test:assertEquals(result.detail()["status"], 400);
    test:assertFalse(check rawIndexExists(indexName), "no index should have been left behind");
}

@test:Config {groups: ["docker"]}
isolated function testContainerUnreachableEndpointFailsAfterRetries() returns error? {
    // Nothing listens on 9299, and with no retries there is no backoff to wait out.
    DenseSearch configMode = {queryMode: ai:DENSE, indexConfig: {dimension: CONTAINER_DIMENSION}};
    Configuration config = {retryConfig: {maxRetries: 0}};
    VectorStore|ai:Error result = new ("http://localhost:9299", CONTAINER_REGION, "unreachable",
        containerDeployment(), configMode, config
    );

    if result !is ai:Error {
        check result.close();
        test:assertFail("construction against an unreachable endpoint should fail");
    }
    test:assertTrue(result.message().includes("attempt"),
            string `the transport error should report the attempt count, got: ${result.message()}`);
}
