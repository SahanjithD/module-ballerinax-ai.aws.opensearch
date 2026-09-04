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

isolated function textEntry(string? id, ai:Vector embedding, string content = "hello world") returns ai:VectorEntry =>
    {id, embedding, chunk: {'type: "text-chunk", content}};

// --- prepareEntries ------------------------------------------------------------------------

@test:Config
isolated function testPrepareEntriesGeneratesIdWhenAbsent() returns error? {
    PreparedEntry[] prepared = check prepareEntries([textEntry((), [0.1, 0.2])], ai:COSINE);
    test:assertEquals(prepared.length(), 1);
    test:assertTrue(prepared[0].id.length() > 0, "a UUID should have been generated");
}

@test:Config
isolated function testPrepareEntriesKeepsCallerSuppliedId() returns error? {
    PreparedEntry[] prepared = check prepareEntries([textEntry("my-id", [0.1, 0.2])], ai:COSINE);
    test:assertEquals(prepared[0].id, "my-id");
}

@test:Config
isolated function testPrepareEntriesRejectsSparseEmbedding() {
    ai:VectorEntry entry = {id: "1", embedding: {indices: [0, 1], values: [0.1, 0.2]}, chunk: {'type: "text-chunk", content: "x"}};
    PreparedEntry[]|ai:Error result = prepareEntries([entry], ai:COSINE);
    test:assertTrue(result is ai:Error, "expected an ai:Error for a sparse embedding");
}

@test:Config
isolated function testPrepareEntriesRejectsHybridEmbedding() {
    ai:VectorEntry entry = {
        id: "1",
        embedding: {dense: [0.1, 0.2], sparse: {indices: [0], values: [0.5]}},
        chunk: {'type: "text-chunk", content: "x"}
    };
    PreparedEntry[]|ai:Error result = prepareEntries([entry], ai:COSINE);
    test:assertTrue(result is ai:Error, "expected an ai:Error for a hybrid embedding");
}

@test:Config
isolated function testPrepareEntriesRejectsZeroVectorUnderCosine() {
    PreparedEntry[]|ai:Error result = prepareEntries([textEntry("z", [0.0, 0.0, 0.0])], ai:COSINE);
    test:assertTrue(result is ai:Error, "expected an ai:Error for a zero vector under COSINE");
}

@test:Config
isolated function testPrepareEntriesAllowsZeroVectorUnderEuclidean() returns error? {
    PreparedEntry[] prepared = check prepareEntries([textEntry("z", [0.0, 0.0, 0.0])], ai:EUCLIDEAN);
    test:assertEquals(prepared.length(), 1);
}

// --- buildEntrySource ------------------------------------------------------------------------

@test:Config
isolated function testBuildEntrySourceNestedMetadata() returns error? {
    Configuration config = {indexConfig: {dimension: 2}};
    PreparedEntry entry = {
        id: "1",
        embedding: [0.1, 0.2],
        chunk: {'type: "text-chunk", content: "hello", metadata: {"language": "en"}}
    };
    map<json> src = check buildEntrySource(entry, config);
    test:assertEquals(src["embedding"], <json[]>[0.1, 0.2]);
    test:assertEquals(src["content"], "hello");
    test:assertEquals(src["doc_id"], "1");
    test:assertEquals(src["chunk_type"], "text-chunk");
    test:assertEquals(src["metadata"], {"language": "en"});
}

@test:Config
isolated function testBuildEntrySourceFlatMetadata() returns error? {
    Configuration config = {indexConfig: {dimension: 2}, metadataFieldName: ""};
    PreparedEntry entry = {
        id: "1",
        embedding: [0.1, 0.2],
        chunk: {'type: "text-chunk", content: "hello", metadata: {"language": "en"}}
    };
    map<json> src = check buildEntrySource(entry, config);
    test:assertEquals(src["language"], "en");
    test:assertFalse(src.hasKey("metadata"), "flat metadata must not be nested under a 'metadata' field");
}

@test:Config
isolated function testBuildEntrySourceFlatMetadataCollisionIsError() {
    Configuration config = {indexConfig: {dimension: 2}, metadataFieldName: ""};
    PreparedEntry entry = {
        id: "1",
        embedding: [0.1, 0.2],
        chunk: {'type: "text-chunk", content: "hello", metadata: {"content": "clobbered!"}}
    };
    map<json>|ai:Error result = buildEntrySource(entry, config);
    test:assertTrue(result is ai:Error,
            "a flat-schema metadata key colliding with a reserved field name must be rejected");
}

// --- buildAddBulkBody ------------------------------------------------------------------------

@test:Config
isolated function testAddBulkBodyEmptyEntriesProducesEmptyBytes() returns error? {
    byte[] body = check buildAddBulkBody([], "my-index", managedDeployment(),
            {indexConfig: {dimension: 2}});
    test:assertEquals(body.length(), 0);
}

@test:Config
isolated function testAddBulkBodyManagedDomainWritesId() returns error? {
    Configuration config = {indexConfig: {dimension: 2}};
    PreparedEntry entry = {id: "abc", embedding: [0.1, 0.2], chunk: {'type: "text-chunk", content: "hi"}};
    byte[] body = check buildAddBulkBody([entry], "my-index", managedDeployment(), config);
    string ndjson = check string:fromBytes(body);
    test:assertTrue(ndjson.endsWith("\n"), "the NDJSON body must end with a trailing newline");
    string[] lines = re `\n`.split(ndjson);
    json actionLine = check lines[0].fromJsonString();
    test:assertEquals(actionLine, {"index": {"_index": "my-index", "_id": "abc"}});
}

@test:Config
isolated function testAddBulkBodyServerlessClassicOmitsId() returns error? {
    Configuration config = {indexConfig: {dimension: 2}};
    PreparedEntry entry = {id: "abc", embedding: [0.1, 0.2], chunk: {'type: "text-chunk", content: "hi"}};
    byte[] body = check buildAddBulkBody([entry], "my-index", classicDeployment(), config);
    string ndjson = check string:fromBytes(body);
    json actionLine = check re `\n`.split(ndjson)[0].fromJsonString();
    test:assertEquals(actionLine, {"index": {"_index": "my-index"}});
}

@test:Config
isolated function testAddBulkBodyServerlessNextGenWritesId() returns error? {
    Configuration config = {indexConfig: {dimension: 2}};
    PreparedEntry entry = {id: "abc", embedding: [0.1, 0.2], chunk: {'type: "text-chunk", content: "hi"}};
    byte[] body = check buildAddBulkBody([entry], "my-index", nextGenDeployment(), config);
    string ndjson = check string:fromBytes(body);
    json actionLine = check re `\n`.split(ndjson)[0].fromJsonString();
    test:assertEquals(actionLine, {"index": {"_index": "my-index", "_id": "abc"}});
}

@test:Config
isolated function testAddBulkBodyHasTwoLinesPerEntry() returns error? {
    Configuration config = {indexConfig: {dimension: 2}};
    PreparedEntry[] entries = [
        {id: "1", embedding: [0.1, 0.2], chunk: {'type: "text-chunk", content: "a"}},
        {id: "2", embedding: [0.3, 0.4], chunk: {'type: "text-chunk", content: "b"}}
    ];
    byte[] body = check buildAddBulkBody(entries, "my-index", managedDeployment(), config);
    string ndjson = check string:fromBytes(body);
    // trailing newline means the final split element is empty; drop it before counting
    string[] lines = re `\n`.split(ndjson);
    test:assertEquals(lines.length(), 5, "2 entries * 2 lines + a trailing empty element");
    test:assertEquals(lines[4], "");
}

// --- buildDeleteBulkBody ---------------------------------------------------------------------

@test:Config
isolated function testDeleteBulkBodyEmptyIdsProducesEmptyBytes() {
    byte[] body = buildDeleteBulkBody("my-index", []);
    test:assertEquals(body.length(), 0);
}

@test:Config
isolated function testDeleteBulkBodyShapeAndTrailingNewline() returns error? {
    byte[] body = buildDeleteBulkBody("my-index", ["1", "2"]);
    string ndjson = check string:fromBytes(body);
    test:assertTrue(ndjson.endsWith("\n"));
    string[] lines = re `\n`.split(ndjson);
    test:assertEquals(lines.length(), 3);
    json first = check lines[0].fromJsonString();
    test:assertEquals(first, {"delete": {"_index": "my-index", "_id": "1"}});
    json second = check lines[1].fromJsonString();
    test:assertEquals(second, {"delete": {"_index": "my-index", "_id": "2"}});
}

// --- buildDocIdLookupBody ---------------------------------------------------------------------

@test:Config
isolated function testDocIdLookupBodyShape() {
    json body = buildDocIdLookupBody("doc_id", ["a", "b"], 10000);
    // `_source` fetches the logical id field rather than being switched off: the bulk delete this
    // feeds submits internal `_id`s, and the logical id has to travel back alongside each hit for
    // a per-item delete failure to be reportable against an id the caller recognises.
    json expected = {
        "size": 10000,
        "_source": ["doc_id"],
        "track_total_hits": false,
        "query": {"terms": {"doc_id": ["a", "b"]}}
    };
    test:assertEquals(body, expected);
}

@test:Config
isolated function testDocIdLookupBodyFetchesConfiguredIdField() {
    json body = buildDocIdLookupBody("external_ref", ["a"], 500);
    map<json> asMap = <map<json>>body;
    test:assertEquals(asMap["_source"], ["external_ref"],
            "the lookup should fetch whatever field 'Configuration.idFieldName' names");
}

// --- chunking helpers -------------------------------------------------------------------------

@test:Config
isolated function testChunkPreparedEntriesBoundaries() {
    PreparedEntry[] entries = from int i in 0 ..< 5
        select {id: i.toString(), embedding: [0.1], chunk: {'type: "text-chunk", content: "x"}};

    PreparedEntry[][] batches = chunkPreparedEntries(entries, 2);
    test:assertEquals(batches.length(), 3);
    test:assertEquals(batches[0].length(), 2);
    test:assertEquals(batches[1].length(), 2);
    test:assertEquals(batches[2].length(), 1);
}

@test:Config
isolated function testChunkPreparedEntriesExactMultiple() {
    PreparedEntry[] entries = from int i in 0 ..< 4
        select {id: i.toString(), embedding: [0.1], chunk: {'type: "text-chunk", content: "x"}};
    PreparedEntry[][] batches = chunkPreparedEntries(entries, 2);
    test:assertEquals(batches.length(), 2);
}

@test:Config
isolated function testChunkPreparedEntriesEmpty() {
    PreparedEntry[][] batches = chunkPreparedEntries([], 500);
    test:assertEquals(batches.length(), 0);
}

@test:Config
isolated function testChunkPreparedEntriesSizeLargerThanInput() {
    PreparedEntry[] entries = from int i in 0 ..< 3
        select {id: i.toString(), embedding: [0.1], chunk: {'type: "text-chunk", content: "x"}};
    PreparedEntry[][] batches = chunkPreparedEntries(entries, 500);
    test:assertEquals(batches.length(), 1);
    test:assertEquals(batches[0].length(), 3);
}

@test:Config
isolated function testChunkIdsBoundaries() {
    string[] ids = from int i in 0 ..< 5
        select i.toString();
    string[][] batches = chunkIds(ids, 2);
    test:assertEquals(batches.length(), 3);
    test:assertEquals(batches[2], ["4"]);
}

// --- bulk response parsing ---------------------------------------------------------------------

@test:Config
isolated function testExtractIndexFailuresNoneWhenErrorsFalse() {
    BulkResponse response = {
        errors: false,
        items: [
            {"index": {"_id": "1", status: 400, 'error: {'type: "x", reason: "should be ignored"}}}
        ]
    };
    BulkFailure[] failures = extractIndexFailures(response, ["a"]);
    test:assertEquals(failures.length(), 0);
}

@test:Config
isolated function testExtractIndexFailuresMixedSuccessAndFailure() {
    BulkResponse response = {
        errors: true,
        items: [
            {"index": {"_id": "1", status: 201, result: "created"}},
            {"index": {"_id": "2", status: 400, 'error: {'type: "mapper_parsing_exception", reason: "bad doc"}}}
        ]
    };
    BulkFailure[] failures = extractIndexFailures(response, ["first", "second"]);
    test:assertEquals(failures.length(), 1);
    test:assertEquals(failures[0].id, "second");
    test:assertEquals(failures[0].reason, "bad doc");
}

// The `SERVERLESS_CLASSIC` shape: the action line carried no `_id`, so every `_id` coming back is
// one the server invented and none of them is anything the caller can act on. Only the item's
// position ties a failure back to the entry that caused it.
@test:Config
isolated function testExtractIndexFailuresAttributesByPositionNotServerId() {
    BulkResponse response = {
        errors: true,
        items: [
            {"index": {"_id": "1%3A0%3AjNUSV6ABrlsmLW-dso51", status: 201, result: "created"}},
            {
                "index": {
                    "_id": "1%3A0%3AjNUSV6ABrlsmLW-dso53",
                    status: 400,
                    'error: {'type: "mapper_parsing_exception", reason: "failed to parse field"}
                }
            },
            {"index": {"_id": "1%3A0%3AjNUSV6ABrlsmLW-dso57", status: 201, result: "created"}}
        ]
    };
    BulkFailure[] failures = extractIndexFailures(response, ["good-0", "bad-1", "good-2"]);
    test:assertEquals(failures.length(), 1);
    test:assertEquals(failures[0].id, "bad-1",
            "the caller's own id, not the server-generated '_id', should name the failing entry");
}

// A response carrying more items than were submitted means the positional correspondence this
// relies on has already broken; the server's `_id` is all that is left.
@test:Config
isolated function testExtractIndexFailuresFallsBackToServerIdBeyondSubmitted() {
    BulkResponse response = {
        errors: true,
        items: [
            {"index": {"_id": "known", status: 400, 'error: {reason: "first"}}},
            {"index": {"_id": "unexpected", status: 400, 'error: {reason: "second"}}}
        ]
    };
    BulkFailure[] failures = extractIndexFailures(response, ["only-one"]);
    test:assertEquals(failures.length(), 2);
    test:assertEquals(failures[0].id, "only-one");
    test:assertEquals(failures[1].id, "unexpected");
}

@test:Config
isolated function testExtractDeleteFailuresIgnoresNotFound() {
    BulkResponse response = {
        errors: false,
        items: [
            {"delete": {"_id": "1", status: 200, result: "deleted"}},
            {"delete": {"_id": "2", status: 404, result: "not_found"}}
        ]
    };
    BulkFailure[] failures = extractDeleteFailures(response, ["a", "b"]);
    test:assertEquals(failures.length(), 0);
}

@test:Config
isolated function testExtractDeleteFailuresReportsRealErrors() {
    BulkResponse response = {
        errors: true,
        items: [
            {"delete": {"_id": "1", status: 500, 'error: {'type: "internal", reason: "boom"}}}
        ]
    };
    BulkFailure[] failures = extractDeleteFailures(response, ["logical-1"]);
    test:assertEquals(failures.length(), 1);
    test:assertEquals(failures[0].id, "logical-1");
    test:assertEquals(failures[0].reason, "boom");
}

// The `SERVERLESS_CLASSIC` delete submits internal `_id`s discovered by a lookup, so without
// positional attribution a failure would be reported against an id the caller never supplied.
@test:Config
isolated function testExtractDeleteFailuresReportsLogicalIdsNotInternalIds() {
    BulkResponse response = {
        errors: true,
        items: [
            {"delete": {"_id": "jNUSV6ABrlsmLW-dso53", status: 200, result: "deleted"}},
            {"delete": {"_id": "kNUSV6ABrlsmLW-dso61", status: 409, 'error: {reason: "version conflict"}}}
        ]
    };
    BulkFailure[] failures = extractDeleteFailures(response, ["doc-a", "doc-b"]);
    test:assertEquals(failures.length(), 1);
    test:assertEquals(failures[0].id, "doc-b");
}

// A non-error item that still failed carries no `error` object, only a status; it must be
// attributed the same way.
@test:Config
isolated function testExtractDeleteFailuresStatusOnlyUsesSubmittedId() {
    BulkResponse response = {
        errors: true,
        items: [
            {"delete": {"_id": "internal-1", status: 503}}
        ]
    };
    BulkFailure[] failures = extractDeleteFailures(response, ["doc-a"]);
    test:assertEquals(failures.length(), 1);
    test:assertEquals(failures[0].id, "doc-a");
    test:assertEquals(failures[0].reason, "HTTP 503");
}

@test:Config
isolated function testSummarizeBulkFailuresWithinLimit() {
    BulkFailure[] failures = [{id: "1", reason: "bad"}, {id: "2", reason: "worse"}];
    string summary = summarizeBulkFailures(failures, 10);
    test:assertEquals(summary, "'1': bad; '2': worse");
}

@test:Config
isolated function testSummarizeBulkFailuresRespectsLimit() {
    BulkFailure[] failures = [{id: "1", reason: "a"}, {id: "2", reason: "b"}, {id: "3", reason: "c"}];
    string summary = summarizeBulkFailures(failures, 2);
    test:assertEquals(summary, "'1': a; '2': b (and 1 more)");
}
