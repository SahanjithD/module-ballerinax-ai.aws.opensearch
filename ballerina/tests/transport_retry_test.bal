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
        http:Ok ok = {body: {"errors": false, "items": []}};
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
