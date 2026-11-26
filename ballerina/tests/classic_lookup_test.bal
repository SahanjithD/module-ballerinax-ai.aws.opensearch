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

// A stand-in for the lookup a Serverless Classic delete makes first. The index name picks how
// complete the `_search` answer is.

import ballerina/ai;
import ballerina/http;
import ballerina/test;

# The port the lookup stand-in listens on.
const int LOOKUP_PORT = 18304;

isolated int lookupBulkCalls = 0;

listener http:Listener lookupListener = new (LOOKUP_PORT);

service / on lookupListener {
    isolated resource function get [string index]/_mapping() returns json => {
        [index]: {"mappings": {"properties": {"embedding": {"type": "knn_vector", "dimension": CONTAINER_DIMENSION}}}}
    };

    isolated resource function post [string index]/_search() returns http:Ok {
        json[] hits = [{"_id": "internal-1", "_source": {"doc_id": "a"}}];
        map<json> body = {"hits": {"hits": hits}};
        if index == "timed-out" {
            body["timed_out"] = true;
        } else if index == "failed-shard" {
            body["_shards"] = {"total": 2, "successful": 1, "failed": 1};
        } else {
            body["timed_out"] = false;
            body["_shards"] = {"total": 2, "successful": 2, "failed": 0};
        }
        return {body};
    }

    isolated resource function post _bulk() returns http:Ok {
        lock {
            lookupBulkCalls += 1;
        }
        return {body: {"errors": false, "items": [{"delete": {"_id": "internal-1", "status": 200}}]}};
    }
}

isolated function takeLookupBulkCalls() returns int {
    lock {
        int calls = lookupBulkCalls;
        lookupBulkCalls = 0;
        return calls;
    }
}

isolated function classicStoreOn(string indexName) returns VectorStoreCore|ai:Error =>
    new (string `http://localhost:${LOOKUP_PORT}`, CONTAINER_REGION, indexName, containerAuth, classicDeployment(),
    {queryMode: ai:DENSE, indexConfig: {dimension: CONTAINER_DIMENSION}}, createIndexIfNotExists = false
);

@test:Config
isolated function testClassicDeleteRejectsAPartialLookup() returns error? {
    foreach string indexName in ["timed-out", "failed-shard"] {
        _ = takeLookupBulkCalls();
        VectorStoreCore store = check classicStoreOn(indexName);
        ai:Error? result = store.delete("a");
        if result !is ai:Error {
            test:assertFail(string `a '${indexName}' lookup may have missed copies, so the delete must fail`);
        }
        test:assertTrue(result.message().includes("partial"), result.message());
        test:assertEquals(takeLookupBulkCalls(), 0, "nothing should be deleted after a partial lookup");
    }
}

@test:Config {dependsOn: [testClassicDeleteRejectsAPartialLookup]}
isolated function testClassicDeleteAcceptsACompleteLookup() returns error? {
    _ = takeLookupBulkCalls();
    VectorStoreCore store = check classicStoreOn("complete");
    check store.delete("a");
    test:assertEquals(takeLookupBulkCalls(), 1);
}

// The container tests show a split delete still deletes everything; this one counts the requests.
@test:Config {dependsOn: [testClassicDeleteAcceptsACompleteLookup]}
isolated function testDeleteSendsOneBulkPerMaxBulkSizeIds() returns error? {
    _ = takeLookupBulkCalls();
    VectorStoreCore store = check new (string `http://localhost:${LOOKUP_PORT}`, CONTAINER_REGION, "complete",
        containerAuth, containerDeployment(), {queryMode: ai:DENSE, indexConfig: {dimension: CONTAINER_DIMENSION}},
        {maxBulkSize: 2}, createIndexIfNotExists = false
    );
    check store.delete(["a", "b", "c", "d", "e"]);
    test:assertEquals(takeLookupBulkCalls(), 3, "five ids at two per request is three requests");
}
