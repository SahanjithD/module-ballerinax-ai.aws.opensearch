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

import ballerina/ai;
import ballerina/test;
import ballerina/time;

// The fixed [int, decimal] tuple <-> RFC 3339 string pair used throughout these tests, copied
// from ai.pinecone's test convention: deterministic, no clock dependency, and exercises 6-digit
// microsecond precision on both `time:utcToString` and `time:utcFromString`.
const string FIXED_TIMESTAMP_STRING = "2025-09-25T09:36:47.798845Z";

isolated function fixedTimestamp() returns time:Utc => [1758793007, 0.798845];

@test:Config
isolated function testTransformMetadataNil() {
    map<json> result = transformMetadata(());
    test:assertEquals(result, {});
}

@test:Config
isolated function testCreateAiMetadataNil() returns error? {
    ai:Metadata? result = check createAiMetadata(());
    test:assertEquals(result, ());
}

@test:Config
isolated function testTransformMetadataCreatedAt() {
    ai:Metadata metadata = {createdAt: fixedTimestamp()};
    map<json> result = transformMetadata(metadata);
    test:assertEquals(result, {"createdAt": FIXED_TIMESTAMP_STRING});
}

@test:Config
isolated function testCreateAiMetadataCreatedAt() returns error? {
    map<json> stored = {"createdAt": FIXED_TIMESTAMP_STRING};
    ai:Metadata? result = check createAiMetadata(stored);
    test:assertEquals(result, {createdAt: fixedTimestamp()});
}

@test:Config
isolated function testTransformMetadataModifiedAt() {
    ai:Metadata metadata = {modifiedAt: fixedTimestamp()};
    map<json> result = transformMetadata(metadata);
    test:assertEquals(result, {"modifiedAt": FIXED_TIMESTAMP_STRING});
}

@test:Config
isolated function testCreateAiMetadataModifiedAt() returns error? {
    map<json> stored = {"modifiedAt": FIXED_TIMESTAMP_STRING};
    ai:Metadata? result = check createAiMetadata(stored);
    test:assertEquals(result, {modifiedAt: fixedTimestamp()});
}

@test:Config
isolated function testFileSizeIntToDecimal() returns error? {
    // A document read back from OpenSearch decodes a whole-number JSON field as `int`, never
    // `decimal` — createAiMetadata must convert it explicitly for the `fileSize` key.
    map<json> stored = {"fileSize": 100};
    ai:Metadata? result = check createAiMetadata(stored);
    ai:Metadata expected = {fileSize: 100d};
    test:assertEquals(result, expected);
    test:assertTrue(result is ai:Metadata && result?.fileSize is decimal, "fileSize must convert to decimal");
}

@test:Config
isolated function testTransformMetadataFileSizePassesThroughAsJson() {
    ai:Metadata metadata = {fileSize: 100d};
    map<json> result = transformMetadata(metadata);
    test:assertEquals(result, {"fileSize": 100d});
}

@test:Config
isolated function testArbitraryStringFieldPassesThroughUnchanged() returns error? {
    ai:Metadata metadata = {"language": "en"};
    map<json> written = transformMetadata(metadata);
    test:assertEquals(written, {"language": "en"});

    ai:Metadata? readBack = check createAiMetadata(written);
    test:assertEquals(readBack, {"language": "en"});
}

@test:Config
isolated function testFileNameAndMimeTypePassThrough() returns error? {
    ai:Metadata metadata = {fileName: "report.pdf", mimeType: "application/pdf"};
    map<json> written = transformMetadata(metadata);
    test:assertEquals(written, {"fileName": "report.pdf", "mimeType": "application/pdf"});

    ai:Metadata? readBack = check createAiMetadata(written);
    test:assertEquals(readBack, {fileName: "report.pdf", mimeType: "application/pdf"});
}

@test:Config
isolated function testCreateAiMetadataInvalidTimestampIsError() {
    map<json> stored = {"createdAt": "not-a-timestamp"};
    ai:Metadata?|ai:Error result = createAiMetadata(stored);
    test:assertTrue(result is ai:Error, "expected an ai:Error for an unparsable 'createdAt' value");
}

@test:Config
isolated function testFullRoundTrip() returns error? {
    ai:Metadata original = {
        createdAt: fixedTimestamp(),
        modifiedAt: fixedTimestamp(),
        fileName: "notes.txt",
        mimeType: "text/plain",
        fileSize: 2048d,
        "language": "en"
    };
    map<json> written = transformMetadata(original);
    ai:Metadata? readBack = check createAiMetadata(written);
    test:assertEquals(readBack, original);
}
