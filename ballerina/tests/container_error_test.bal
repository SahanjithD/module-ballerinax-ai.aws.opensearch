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

// `mapErrorResponse` and `executeWithRetry` against responses a real cluster produces. The error
// bodies OpenSearch returns are the one part of the protocol that cannot be usefully guessed at:
// the `error.type`/`error.reason` shape, and the fact that some failures never carry a JSON body
// at all, are exactly what a hand-written stub would get wrong.

import ballerina/ai;
import ballerina/test;

# A store pointed at an index that does not exist, with index creation switched off so `init`
# performs no I/O and every subsequent call hits a missing index.
#
# + indexName - The index that will not exist
# + return - The store, or an `ai:Error`
isolated function storeOnMissingIndex(string indexName) returns VectorStore|ai:Error =>
    newContainerStore(indexName, {
        indexConfig: {dimension: CONTAINER_DIMENSION, createIndexIfNotExists: false},
        refreshOnWrite: true
    });

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
    // A missing *index* and a missing *document* both come back as a per-item 404 at HTTP 200, and
    // `extractDeleteFailures` has to tell them apart: only the missing index carries an `error`
    // object, and only it is a real failure. `testContainerDeleteMissingIdIsNotAnError` pins the
    // other half of that distinction. Note this is a bulk-item failure, not a mapped HTTP
    // response, so it carries no `status` detail.
    test:assertTrue(result.message().includes("no such index"),
            string `the server's reason should be preserved, got: ${result.message()}`);
    check store.close();
}

@test:Config {groups: ["docker"]}
isolated function testContainerAddOnMissingIndexAutoCreatesAnUnusableIndex() returns error? {
    string indexName = containerIndexName("err-autocreate");
    VectorStore store = check storeOnMissingIndex(indexName);

    // OpenSearch ships with `action.auto_create_index: true`, so a `_bulk` write against an index
    // that does not exist does not fail -- it creates one from the document's inferred shape. The
    // result is an index with no `index.knn` setting and `embedding` mapped as a plain `float`
    // array, which accepts writes happily and then rejects every k-NN query against it.
    //
    // This is the failure mode of setting `createIndexIfNotExists: false` against an index that
    // was never provisioned out of band, and it is silent at the point of the mistake. Pinned here
    // so the behaviour is at least documented and any future guard against it is a deliberate
    // change rather than an accidental one.
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
    // The server's own explanation has to survive into the message, not be replaced by a generic one.
    test:assertTrue(result.message().includes("dimension") || result.message().includes("vector"),
            string `the server's reason should be preserved, got: ${result.message()}`);
    check store.close();
}

@test:Config {groups: ["docker"]}
isolated function testContainerCreateIndexOnBadMappingIsReported() returns error? {
    string indexName = containerIndexName("err-mapping");
    // `dimension` is passed to the server untouched -- the module deliberately enforces no ceiling,
    // because AWS's own documentation disagrees about what it is. This confirms the server's
    // rejection surfaces as a construction failure rather than a silent half-built store.
    Configuration config = {indexConfig: {dimension: 99999}, refreshOnWrite: true};
    VectorStore|ai:Error result = newContainerStore(indexName, config);

    if result !is ai:Error {
        check result.close();
        test:assertFail("an out-of-range dimension should fail index creation");
    }
    test:assertEquals(result.detail()["status"], 400);
    test:assertFalse(check rawIndexExists(indexName), "no index should have been left behind");
}

@test:Config {groups: ["docker"]}
isolated function testContainerUnreachableEndpointFailsAfterRetries() returns error? {
    // Nothing is listening on 9299. With retries switched off this is a single attempt, so the
    // test asserts `executeWithRetry`'s give-up path without waiting out any backoff.
    Configuration config = {
        indexConfig: {dimension: CONTAINER_DIMENSION},
        retryConfig: {maxRetries: 0}
    };
    VectorStore|ai:Error result = new ("http://localhost:9299", CONTAINER_REGION, "unreachable",
        MANAGED_DOMAIN, containerAuth, config
    );

    if result !is ai:Error {
        check result.close();
        test:assertFail("construction against an unreachable endpoint should fail");
    }
    test:assertTrue(result.message().includes("attempt"),
            string `the transport error should report the attempt count, got: ${result.message()}`);
}
