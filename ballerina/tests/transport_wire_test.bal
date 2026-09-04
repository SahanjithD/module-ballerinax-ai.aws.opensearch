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

// Guards how `OpenSearchTransport` frames a request on the wire, which neither the pure unit
// tests nor the container suite can see: both are blind to the HTTP version and to whether the
// body was chunked. `OpenSearchTransport.init` pins `HTTP_1_1` and `CHUNKING_NEVER` over the
// caller's `httpConfig` because Ballerina's defaults (`HTTP_2_0`, `CHUNKING_AUTO`) each break a
// SigV4-signed request in a way that surfaces as a misleading 403 -- see the comment there. This
// suite catches a regression that would otherwise only appear against real AWS.
//
// A local `http:Listener` stands in for OpenSearch: it records how the request arrived and
// answers with an empty, successful `_bulk` response.

import ballerina/ai;
import ballerina/http;
import ballerina/test;

# The port the recording stand-in listens on.
const int WIRE_PORT = 18299;

# Large enough that a chunking-enabled client would chunk the `_bulk` body (~100 KB of JSON).
const int WIRE_DIMENSION = 4096;

# How a single request arrived at the stand-in.
type WireCapture record {|
    # The HTTP version the request was sent over.
    string httpVersion;
    # The `content-length` header, or `()` when absent.
    string? contentLength;
    # The `transfer-encoding` header, or `()` when absent.
    string? transferEncoding;
    # The number of body bytes actually received.
    int bodyLength;
|};

isolated WireCapture? lastWireCapture = ();

isolated function recordWireCapture(WireCapture capture) {
    lock {
        lastWireCapture = capture.clone();
    }
}

isolated function takeWireCapture() returns WireCapture? {
    lock {
        WireCapture? capture = lastWireCapture;
        lastWireCapture = ();
        return capture.clone();
    }
}

isolated function optionalHeader(http:Request request, string name) returns string? {
    string|http:HeaderNotFoundError value = request.getHeader(name);
    return value is string ? value : ();
}

listener http:Listener wireListener = new (WIRE_PORT);

service / on wireListener {
    isolated resource function post [string... path](http:Request request) returns http:Ok {
        byte[]|http:ClientError body = request.getBinaryPayload();
        recordWireCapture({
                              httpVersion: request.httpVersion,
                              contentLength: optionalHeader(request, "content-length"),
                              transferEncoding: optionalHeader(request, "transfer-encoding"),
                              bodyLength: body is byte[] ? body.length() : -1
                          });
        // `http:Ok` rather than a bare `json`, which Ballerina would answer with a 201 that
        // `OpenSearchTransport.bulk` treats as a failure.
        return {body: {"errors": false, "items": []}};
    }
}

isolated function wireStore() returns VectorStore|ai:Error =>
    new (string `http://localhost:${WIRE_PORT}`, CONTAINER_REGION, "wire-index", containerDeployment(),
    {indexConfig: {dimension: WIRE_DIMENSION, createIndexIfNotExists: false}}
);

isolated function wireEmbedding() returns ai:Vector {
    ai:Vector embedding = [];
    foreach int i in 0 ..< WIRE_DIMENSION {
        embedding.push(<float>(i % 13) + 0.5);
    }
    return embedding;
}

@test:Config
isolated function testLargeBodyIsSentAsUnchunkedHttp11() returns error? {
    VectorStore store = check wireStore();
    check store.add([{id: "wire-1", embedding: wireEmbedding(), chunk: <ai:TextChunk>{content: "framing"}}]);

    WireCapture capture = check takeWireCapture().ensureType();
    test:assertEquals(capture.httpVersion, "1.1",
            "the transport must pin HTTP/1.1: Ballerina's HTTP/2 outbound path alters a signed " +
            "JSON body between signing and the wire");
    test:assertEquals(capture.transferEncoding, (),
            "a chunked body has no 'Content-Length' for SigV4 to be verified against");
    test:assertEquals(capture.contentLength, capture.bodyLength.toString(),
            "the signed body must be framed by an accurate 'Content-Length'");
    test:assertTrue(capture.bodyLength > 16384,
            string `the body must be large enough to trigger chunking under CHUNKING_AUTO, ` +
            string `got ${capture.bodyLength} bytes`);
}

@test:Config
isolated function testCallerSuppliedHttpConfigCannotUnpinTheFraming() returns error? {
    VectorStore store = check new (string `http://localhost:${WIRE_PORT}`, CONTAINER_REGION, "wire-index",
        containerDeployment(), {indexConfig: {dimension: WIRE_DIMENSION, createIndexIfNotExists: false}},
        ai:DENSE, {httpVersion: http:HTTP_2_0, http1Settings: {chunking: http:CHUNKING_ALWAYS}}
    );
    check store.add([{id: "wire-2", embedding: wireEmbedding(), chunk: <ai:TextChunk>{content: "framing"}}]);

    WireCapture capture = check takeWireCapture().ensureType();
    test:assertEquals(capture.httpVersion, "1.1", "an explicit HTTP_2_0 must still be overridden");
    test:assertEquals(capture.transferEncoding, (), "an explicit CHUNKING_ALWAYS must still be overridden");
    test:assertEquals(capture.contentLength, capture.bodyLength.toString());
}
