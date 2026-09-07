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
    PreparedEntry[] prepared = check prepareEntries([textEntry((), [0.1, 0.2])], denseMode(3, ai:COSINE));
    test:assertEquals(prepared.length(), 1);
    test:assertTrue(prepared[0].id.length() > 0, "a UUID should have been generated");
}

@test:Config
isolated function testPrepareEntriesKeepsCallerSuppliedId() returns error? {
    PreparedEntry[] prepared = check prepareEntries([textEntry("my-id", [0.1, 0.2])], denseMode(3, ai:COSINE));
    test:assertEquals(prepared[0].id, "my-id");
}

@test:Config
isolated function testPrepareEntriesRejectsSparseEmbedding() {
    ai:VectorEntry entry = {id: "1", embedding: {indices: [0, 1], values: [0.1, 0.2]}, chunk: {'type: "text-chunk", content: "x"}};
    PreparedEntry[]|ai:Error result = prepareEntries([entry], denseMode(3, ai:COSINE));
    test:assertTrue(result is ai:Error, "expected an ai:Error for a sparse embedding");
}

@test:Config
isolated function testPrepareEntriesRejectsHybridEmbedding() {
    ai:VectorEntry entry = {
        id: "1",
        embedding: {dense: [0.1, 0.2], sparse: {indices: [0], values: [0.5]}},
        chunk: {'type: "text-chunk", content: "x"}
    };
    PreparedEntry[]|ai:Error result = prepareEntries([entry], denseMode(3, ai:COSINE));
    test:assertTrue(result is ai:Error, "expected an ai:Error for a hybrid embedding");
}

@test:Config
isolated function testPrepareEntriesRejectsZeroVectorUnderCosine() {
    PreparedEntry[]|ai:Error result = prepareEntries([textEntry("z", [0.0, 0.0, 0.0])], denseMode(3, ai:COSINE));
    test:assertTrue(result is ai:Error, "expected an ai:Error for a zero vector under COSINE");
}

@test:Config
isolated function testPrepareEntriesAllowsZeroVectorUnderEuclidean() returns error? {
    PreparedEntry[] prepared = check prepareEntries([textEntry("z", [0.0, 0.0, 0.0])], denseMode(3, ai:EUCLIDEAN));
    test:assertEquals(prepared.length(), 1);
}

// --- buildEntrySource ------------------------------------------------------------------------

@test:Config
isolated function testBuildEntrySourceNestedMetadata() returns error? {
    DenseSearch configMode = {queryMode: ai:DENSE, indexConfig: {dimension: 2}};
    PreparedEntry entry = {
        id: "1",
        embedding: [0.1, 0.2],
        chunk: {'type: "text-chunk", content: "hello", metadata: {"language": "en"}}
    };
    map<json> src = check buildEntrySource(entry, configMode, {});
    test:assertEquals(src["embedding"], <json[]>[0.1, 0.2]);
    test:assertEquals(src["content"], "hello");
    test:assertEquals(src["doc_id"], "1");
    test:assertEquals(src["chunk_type"], "text-chunk");
    test:assertEquals(src["metadata"], {"language": "en"});
}

@test:Config
isolated function testBuildEntrySourceFlatMetadata() returns error? {
    DenseSearch configMode = {queryMode: ai:DENSE, indexConfig: {dimension: 2}};
    Configuration config = {metadataFieldName: ""};
    PreparedEntry entry = {
        id: "1",
        embedding: [0.1, 0.2],
        chunk: {'type: "text-chunk", content: "hello", metadata: {"language": "en"}}
    };
    map<json> src = check buildEntrySource(entry, configMode, config);
    test:assertEquals(src["language"], "en");
    test:assertFalse(src.hasKey("metadata"), "flat metadata must not be nested under a 'metadata' field");
}

@test:Config
isolated function testBuildEntrySourceFlatMetadataCollisionIsError() {
    DenseSearch configMode = {queryMode: ai:DENSE, indexConfig: {dimension: 2}};
    Configuration config = {metadataFieldName: ""};
    PreparedEntry entry = {
        id: "1",
        embedding: [0.1, 0.2],
        chunk: {'type: "text-chunk", content: "hello", metadata: {"content": "clobbered!"}}
    };
    map<json>|ai:Error result = buildEntrySource(entry, configMode, config);
    test:assertTrue(result is ai:Error,
            "a flat-schema metadata key colliding with a reserved field name must be rejected");
}

// --- buildAddBulkBody ------------------------------------------------------------------------

@test:Config
isolated function testAddBulkBodyEmptyEntriesProducesEmptyBytes() returns error? {
    byte[] body = check buildAddBulkBody([], "my-index", managedDeployment(), denseMode(2), {});
    test:assertEquals(body.length(), 0);
}

@test:Config
isolated function testAddBulkBodyManagedDomainWritesId() returns error? {
    DenseSearch configMode = {queryMode: ai:DENSE, indexConfig: {dimension: 2}};
    PreparedEntry entry = {id: "abc", embedding: [0.1, 0.2], chunk: {'type: "text-chunk", content: "hi"}};
    byte[] body = check buildAddBulkBody([entry], "my-index", managedDeployment(), configMode, {});
    string ndjson = check string:fromBytes(body);
    test:assertTrue(ndjson.endsWith("\n"), "the NDJSON body must end with a trailing newline");
    string[] lines = re `\n`.split(ndjson);
    json actionLine = check lines[0].fromJsonString();
    test:assertEquals(actionLine, {"index": {"_index": "my-index", "_id": "abc"}});
}

@test:Config
isolated function testAddBulkBodyServerlessClassicOmitsId() returns error? {
    DenseSearch configMode = {queryMode: ai:DENSE, indexConfig: {dimension: 2}};
    PreparedEntry entry = {id: "abc", embedding: [0.1, 0.2], chunk: {'type: "text-chunk", content: "hi"}};
    byte[] body = check buildAddBulkBody([entry], "my-index", classicDeployment(), configMode, {});
    string ndjson = check string:fromBytes(body);
    json actionLine = check re `\n`.split(ndjson)[0].fromJsonString();
    test:assertEquals(actionLine, {"index": {"_index": "my-index"}});
}

@test:Config
isolated function testAddBulkBodyServerlessNextGenWritesId() returns error? {
    DenseSearch configMode = {queryMode: ai:DENSE, indexConfig: {dimension: 2}};
    PreparedEntry entry = {id: "abc", embedding: [0.1, 0.2], chunk: {'type: "text-chunk", content: "hi"}};
    byte[] body = check buildAddBulkBody([entry], "my-index", nextGenDeployment(), configMode, {});
    string ndjson = check string:fromBytes(body);
    json actionLine = check re `\n`.split(ndjson)[0].fromJsonString();
    test:assertEquals(actionLine, {"index": {"_index": "my-index", "_id": "abc"}});
}

@test:Config
isolated function testAddBulkBodyHasTwoLinesPerEntry() returns error? {
    DenseSearch configMode = {queryMode: ai:DENSE, indexConfig: {dimension: 2}};
    PreparedEntry[] entries = [
        {id: "1", embedding: [0.1, 0.2], chunk: {'type: "text-chunk", content: "a"}},
        {id: "2", embedding: [0.3, 0.4], chunk: {'type: "text-chunk", content: "b"}}
    ];
    byte[] body = check buildAddBulkBody(entries, "my-index", managedDeployment(), configMode, {});
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

// --- mode/embedding strictness ----------------------------------------------------------------

// The check is symmetric in `add` and `query`: a store configured for one mode and handed
// another's embedding is a configuration mistake either way. Letting a sparse embedding through
// to a dense index would index a field no query ever reads.
isolated function sparseEntry(string id, int[] indices, float[] values) returns ai:VectorEntry =>
    {id, embedding: {indices, values}, chunk: {'type: "text-chunk", content: "hello"}};

isolated function hybridEntry(string id, ai:Vector dense, int[] indices, float[] values) returns ai:VectorEntry =>
    {id, embedding: {dense, sparse: {indices, values}}, chunk: {'type: "text-chunk", content: "hello"}};

@test:Config
isolated function testPrepareEntriesSparseModeAcceptsSparseEmbedding() returns error? {
    PreparedEntry[] prepared = check prepareEntries([sparseEntry("s", [1055, 2048], [5.5, 1.25])], sparseMode());
    test:assertEquals(prepared[0].rankFeatures, {"1055": 5.5, "2048": 1.25});
    test:assertTrue(prepared[0].embedding is (), "a SPARSE entry carries no dense vector");
}

@test:Config
isolated function testPrepareEntriesHybridModeSplitsBothHalves() returns error? {
    PreparedEntry[] prepared = check prepareEntries([hybridEntry("h", [0.1, 0.2], [7], [0.5])], hybridMode());
    test:assertEquals(prepared[0].embedding, <ai:Vector>[0.1, 0.2]);
    test:assertEquals(prepared[0].rankFeatures, {"7": 0.5});
}

@test:Config
isolated function testPrepareEntriesSparseModeRejectsDenseEmbedding() {
    PreparedEntry[]|ai:Error result = prepareEntries([textEntry("d", [0.1, 0.2])], sparseMode());
    if result !is ai:Error {
        test:assertFail("a dense embedding must be rejected by a SPARSE store");
    }
    test:assertTrue(result.message().includes("ai:SparseVector"),
            string `the message should name the required type, got: ${result.message()}`);
}

@test:Config
isolated function testPrepareEntriesSparseModeRejectsHybridEmbedding() {
    PreparedEntry[]|ai:Error result = prepareEntries([hybridEntry("h", [0.1], [7], [0.5])], sparseMode());
    test:assertTrue(result is ai:Error, "a hybrid embedding must be rejected by a SPARSE store");
}

@test:Config
isolated function testPrepareEntriesDenseModeRejectsSparseEmbedding() {
    PreparedEntry[]|ai:Error result = prepareEntries([sparseEntry("s", [1], [0.5])], denseMode());
    if result !is ai:Error {
        test:assertFail("a sparse embedding must be rejected by a DENSE store");
    }
    test:assertTrue(result.message().includes("ai:Vector"),
            string `the message should name the required type, got: ${result.message()}`);
}

@test:Config
isolated function testPrepareEntriesHybridModeRejectsDenseEmbedding() {
    PreparedEntry[]|ai:Error result = prepareEntries([textEntry("d", [0.1, 0.2])], hybridMode());
    if result !is ai:Error {
        test:assertFail("a dense-only embedding must be rejected by a HYBRID store");
    }
    test:assertTrue(result.message().includes("ai:HybridVector"),
            string `the message should name the required type, got: ${result.message()}`);
}

@test:Config
isolated function testPrepareEntriesHybridModeRejectsSparseEmbedding() {
    PreparedEntry[]|ai:Error result = prepareEntries([sparseEntry("s", [1], [0.5])], hybridMode());
    test:assertTrue(result is ai:Error, "a sparse-only embedding must be rejected by a HYBRID store");
}

// The Lucene bounds are enforced by `toSparseTokenMap`; this pins that `prepareEntries` actually
// routes through it, so a bad weight fails naming the entry rather than failing a whole _bulk.
@test:Config
isolated function testPrepareEntriesSparseModeRejectsZeroWeightNamingTheEntry() {
    PreparedEntry[]|ai:Error result = prepareEntries([sparseEntry("bad-one", [7], [0.0])], sparseMode());
    if result !is ai:Error {
        test:assertFail("a zero sparse weight must be rejected before the wire");
    }
    test:assertTrue(result.message().includes("bad-one"),
            string `the message should name the offending entry, got: ${result.message()}`);
}

// --- sparse and hybrid documents --------------------------------------------------------------

@test:Config
isolated function testBuildEntrySourceSparseWritesOnlyTheSparseField() returns error? {
    PreparedEntry entry = {id: "1", rankFeatures: {"7": 0.5}, chunk: {'type: "text-chunk", content: "hello"}};
    map<json> src = check buildEntrySource(entry, sparseMode(), {});
    test:assertEquals(src["sparse_embedding"], <json>{"7": 0.5});
    test:assertFalse(src.hasKey("embedding"), "a SPARSE document carries no dense vector field");
    test:assertEquals(src["content"], "hello");
    test:assertEquals(src["doc_id"], "1");
}

@test:Config
isolated function testBuildEntrySourceHybridWritesBothFields() returns error? {
    PreparedEntry entry = {
        id: "1",
        embedding: [0.1, 0.2],
        rankFeatures: {"7": 0.5},
        chunk: {'type: "text-chunk", content: "hello"}
    };
    map<json> src = check buildEntrySource(entry, hybridMode(), {});
    test:assertEquals(src["embedding"], <json[]>[0.1, 0.2]);
    test:assertEquals(src["sparse_embedding"], <json>{"7": 0.5});
}

// The vector fields are written before the flat-schema metadata loop, so the existing
// reserved-name collision guard covers the sparse field too, with no second check needed.
@test:Config
isolated function testBuildEntrySourceFlatSchemaRejectsSparseFieldCollision() {
    PreparedEntry entry = {
        id: "1",
        rankFeatures: {"7": 0.5},
        chunk: {'type: "text-chunk", content: "hello", metadata: {"sparse_embedding": "collides"}}
    };
    map<json>|ai:Error result = buildEntrySource(entry, sparseMode(), {metadataFieldName: ""});
    test:assertTrue(result is ai:Error,
            "a flat-schema metadata key colliding with the sparse vector field must be rejected");
}

@test:Config
isolated function testExtractStoredMetadataExcludesSparseFieldUnderFlatSchema() {
    map<json> src = {"sparse_embedding": {"7": 0.5}, "content": "hello", "doc_id": "1", "language": "en"};
    map<json>? metadata = extractStoredMetadata(src, sparseMode(), {metadataFieldName: ""});
    if metadata is () {
        test:assertFail("the non-reserved field should have been collected as metadata");
    }
    test:assertFalse(metadata.hasKey("sparse_embedding"),
            "the sparse vector field must not be returned as metadata");
    test:assertEquals(metadata["language"], "en");
}
