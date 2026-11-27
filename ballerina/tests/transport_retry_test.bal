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

// A local listener that fails the first `_bulk` with a 503, so the retry's signature can be
// compared against the first attempt's.

import ballerina/ai;
import ballerina/http;
import ballerina/test;

# The port the retrying stand-in listens on.
const int RETRY_PORT = 18300;

# The signing headers of one attempt, as the stand-in received them.
type RetryCapture record {|
    # The `x-amz-date` header, or `()` when absent.
    string? amzDate;
    # The `authorization` header, or `()` when absent.
    string? authorization;
    # The number of body bytes received.
    int bodyLength;
|};

isolated RetryCapture[] retryCaptures = [];

isolated function recordRetryCapture(RetryCapture capture) returns int {
    lock {
        retryCaptures.push(capture.clone());
        return retryCaptures.length();
    }
}

isolated function takeRetryCaptures() returns RetryCapture[] {
    lock {
        RetryCapture[] captures = retryCaptures;
        retryCaptures = [];
        return captures.clone();
    }
}

listener http:Listener retryListener = new (RETRY_PORT);

service / on retryListener {
    // Answers the construction-time `_mapping` read, so only `_bulk` attempts are recorded.
    isolated resource function get [string index]/_mapping() returns json => {
        [index]: {"mappings": {"properties": {"embedding": {"type": "knn_vector", "dimension": CONTAINER_DIMENSION}}}}
    };

    isolated resource function post [string... path](http:Request request)
            returns http:Ok|http:ServiceUnavailable {
        byte[]|http:ClientError body = request.getBinaryPayload();
        int attempt = recordRetryCapture({
            amzDate: optionalHeader(request, "x-amz-date"),
            authorization: optionalHeader(request, "authorization"),
            bodyLength: body is byte[] ? body.length() : -1
        });
        if attempt == 1 {
            return <http:ServiceUnavailable>{};
        }
        http:Ok ok = {body: {"errors": false, "items": [{"index": {"_id": "retry-1", "status": 201}}]}};
        return ok;
    }
}

@test:Config
isolated function testRetryIsSignedAfresh() returns error? {
    // `x-amz-date` has one-second resolution, so the backoff must exceed a second to tell the
    // attempts apart.
    VectorStoreCore store = check new (string `http://localhost:${RETRY_PORT}`, CONTAINER_REGION, "retry-index",
        containerAuth, containerDeployment(), {queryMode: ai:DENSE, indexConfig: {dimension: CONTAINER_DIMENSION}},
        {retryConfig: {maxRetries: 1, initialDelay: 1.1}},
        createIndexIfNotExists = false
    );
    check store.add([entry("retry-1", vec(1.0))]);

    RetryCapture[] captures = takeRetryCaptures();
    test:assertEquals(captures.length(), 2, "the 503 should be retried exactly once");
    RetryCapture first = captures[0];
    RetryCapture second = captures[1];
    test:assertTrue(first.amzDate is string && second.amzDate is string, "both attempts must be signed");
    test:assertNotEquals(second.amzDate, first.amzDate, "a retry must carry a fresh 'x-amz-date'");
    test:assertNotEquals(second.authorization, first.authorization,
            "a retry must carry a signature computed for its own 'x-amz-date'");
    test:assertEquals(second.bodyLength, first.bodyLength, "a retry must resend the same body");
}

// Only the connector retries: a second, HTTP-client retry would resend a stale signature.
@test:Config {dependsOn: [testRetryIsSignedAfresh]}
isolated function testHttpClientRetryIsCleared() returns error? {
    _ = takeRetryCaptures();
    VectorStoreCore store = check new (string `http://localhost:${RETRY_PORT}`, CONTAINER_REGION, "retry-index",
        containerAuth, containerDeployment(), {queryMode: ai:DENSE, indexConfig: {dimension: CONTAINER_DIMENSION}},
        {retryConfig: {maxRetries: 0}},
        {retryConfig: {count: 3, interval: 0.1, statusCodes: [503]}},
        createIndexIfNotExists = false
    );
    ai:Error? result = store.add([entry("retry-2", vec(1.0))]);

    test:assertTrue(result is ai:Error, "the 503 should not be retried with 'maxRetries' at 0");
    test:assertEquals(takeRetryCaptures().length(), 1, "'httpConfig.retryConfig' must not add attempts");
}

// A stand-in whose first `_bulk` reports the second item as rejected with 429; later ones succeed.

# The port the item-rejecting stand-in listens on.
const int REJECT_PORT = 18301;

isolated string[] rejectBodies = [];

isolated function recordRejectBody(string body) returns int {
    lock {
        rejectBodies.push(body);
        return rejectBodies.length();
    }
}

isolated function takeRejectBodies() returns string[] {
    lock {
        string[] bodies = rejectBodies;
        rejectBodies = [];
        return bodies.clone();
    }
}

listener http:Listener rejectListener = new (REJECT_PORT);

service / on rejectListener {
    isolated resource function get [string index]/_mapping() returns json => {
        [index]: {"mappings": {"properties": {"embedding": {"type": "knn_vector", "dimension": CONTAINER_DIMENSION}}}}
    };

    isolated resource function post [string... path](http:Request request) returns http:Ok {
        string|http:ClientError body = request.getTextPayload();
        int call = recordRejectBody(body is string ? body : "");
        if call == 1 {
            return {
                body: {
                    "errors": true,
                    "items": [
                        {"index": {"_id": "rej-1", "status": 201, "result": "created"}},
                        {
                            "index": {
                                "_id": "rej-2",
                                "status": 429,
                                "error": {"type": "es_rejected_execution_exception", "reason": "queue full"}
                            }
                        }
                    ]
                }
            };
        }
        return {body: {"errors": false, "items": [{"index": {"_id": "rej-2", "status": 201, "result": "created"}}]}};
    }
}

isolated function rejectingStore(Deployment deployment) returns VectorStoreCore|ai:Error =>
    new (string `http://localhost:${REJECT_PORT}`, CONTAINER_REGION, "reject-index", containerAuth, deployment,
    {queryMode: ai:DENSE, indexConfig: {dimension: CONTAINER_DIMENSION}},
    {retryConfig: {maxRetries: 2, initialDelay: 0.01}},
    createIndexIfNotExists = false
);

@test:Config
isolated function testItemsRejectedWith429AreResentAlone() returns error? {
    _ = takeRejectBodies();
    VectorStoreCore store = check rejectingStore(containerDeployment());
    check store.add([entry("rej-1", vec(1.0)), entry("rej-2", vec(0.0, 1.0))]);

    string[] bodies = takeRejectBodies();
    test:assertEquals(bodies.length(), 2, "the rejected item should be re-sent once");
    test:assertTrue(bodies[1].includes("rej-2"), "the re-send should carry the rejected item");
    test:assertFalse(bodies[1].includes("rej-1"), "the re-send must not repeat the item that was applied");
}

// A 429'd item was not applied, so re-sending it alone cannot duplicate it, even without a custom `_id`.
@test:Config {dependsOn: [testItemsRejectedWith429AreResentAlone]}
isolated function testItemsRejectedWith429AreResentAloneOnClassic() returns error? {
    _ = takeRejectBodies();
    VectorStoreCore store = check rejectingStore(classicDeployment());
    check store.add([entry("rej-1", vec(1.0)), entry("rej-2", vec(0.0, 1.0))]);

    string[] bodies = takeRejectBodies();
    test:assertEquals(bodies.length(), 2, "the rejected item should be re-sent once");
    test:assertTrue(bodies[1].includes("rej-2"), "the re-send should carry the rejected item");
    test:assertFalse(bodies[1].includes("rej-1"), "the re-send must not repeat the item that was applied");
}

// The retry stand-in fails the first `_bulk` with a 503, whose outcome is unknown.
@test:Config {dependsOn: [testHttpClientRetryIsCleared]}
isolated function testClassicIndexingIsNotRetriedOnA503() returns error? {
    _ = takeRetryCaptures();
    VectorStoreCore store = check new (string `http://localhost:${RETRY_PORT}`, CONTAINER_REGION, "retry-index",
        containerAuth, classicDeployment(), {queryMode: ai:DENSE, indexConfig: {dimension: CONTAINER_DIMENSION}},
        {retryConfig: {maxRetries: 2, initialDelay: 0.01}},
        createIndexIfNotExists = false
    );
    ai:Error? result = store.add([entry("c-503", vec(1.0))]);

    test:assertTrue(result is ai:Error, "a 503 on Classic may have applied part of the batch, so it must not be replayed");
    test:assertEquals(takeRetryCaptures().length(), 1);
}

// A stand-in that throttles the first `_bulk` with an HTTP 429.

# The port the throttling stand-in listens on.
const int THROTTLE_PORT = 18305;

isolated int throttledBulkCalls = 0;

listener http:Listener throttleListener = new (THROTTLE_PORT);

service / on throttleListener {
    isolated resource function get [string index]/_mapping() returns json => {
        [index]: {"mappings": {"properties": {"embedding": {"type": "knn_vector", "dimension": CONTAINER_DIMENSION}}}}
    };

    isolated resource function post [string... path]() returns http:Ok|http:TooManyRequests {
        int call;
        lock {
            throttledBulkCalls += 1;
            call = throttledBulkCalls;
        }
        if call == 1 {
            return <http:TooManyRequests>{};
        }
        http:Ok ok = {body: {"errors": false, "items": [{"index": {"_id": "server-generated", "status": 201}}]}};
        return ok;
    }
}

// A whole-request 429 can follow partial execution when shard indexing pressure is enforced, so on
// Classic it is not replayed either.
@test:Config
isolated function testClassicIndexingIsNotRetriedOnA429() returns error? {
    lock {
        throttledBulkCalls = 0;
    }
    VectorStoreCore store = check new (string `http://localhost:${THROTTLE_PORT}`, CONTAINER_REGION, "throttle-index",
        containerAuth, classicDeployment(), {queryMode: ai:DENSE, indexConfig: {dimension: CONTAINER_DIMENSION}},
        {retryConfig: {maxRetries: 2, initialDelay: 0.01}},
        createIndexIfNotExists = false
    );
    ai:Error? result = store.add([entry("c-429", vec(1.0))]);
    test:assertTrue(result is ai:Error, "the 429 should be reported, not replayed");
    lock {
        test:assertEquals(throttledBulkCalls, 1);
    }
}

// Deletes are idempotent, so a Classic delete `_bulk` is still retried; only indexing is not.
@test:Config {dependsOn: [testClassicIndexingIsNotRetriedOnA503]}
isolated function testClassicDeleteBulkIsStillRetried() returns error? {
    _ = takeRetryCaptures();
    OpenSearchTransport transport = check new (string `http://localhost:${RETRY_PORT}`, CONTAINER_REGION,
        containerAuth, classicDeployment(), {maxRetries: 1, initialDelay: 0.01}, {});
    _ = check transport.bulk(buildDeleteBulkBody("retry-index", ["internal-1"]));
    test:assertEquals(takeRetryCaptures().length(), 2, "the 503 should be retried once");
}
