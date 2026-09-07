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

// `add`'s `_bulk` chunking and per-item failure handling. `bulk_test.bal` covers the pure
// splitting and summarising functions; what needs a server is the fact that OpenSearch answers a
// partially-failed `_bulk` with HTTP 200, so a failure is only visible if `errors: true` is read
// out of the body -- the single easiest thing to get wrong in a bulk client, and invisible to any
// test that does not talk to a real cluster.

import ballerina/ai;
import ballerina/test;

const int BULK_ENTRY_COUNT = 250;
const int BULK_CHUNK_SIZE = 100;

@test:Config {groups: ["docker"]}
isolated function testContainerBulkAddChunksLargeBatch() returns error? {
    string indexName = containerIndexName("bulk-large");
    DenseSearch configMode = {queryMode: ai:DENSE, indexConfig: {dimension: CONTAINER_DIMENSION}};
    Configuration config = {maxBulkSize: BULK_CHUNK_SIZE};
    VectorStore store = check newContainerStore(indexName, configMode, config);

    ai:VectorEntry[] entries = [];
    foreach int i in 0 ..< BULK_ENTRY_COUNT {
        entries.push(entry(string `bulk-${i}`, vec(1.0, <float>i / 1000.0), string `entry ${i}`));
    }
    check store.add(entries);

    // 250 entries at 100 per request is three `_bulk` calls; every entry has to survive all three.
    test:assertEquals(check documentCount(indexName), BULK_ENTRY_COUNT,
            "every entry across every chunk should have been indexed");
    check store.close();
}

@test:Config {groups: ["docker"]}
isolated function testContainerBulkAddStopsAtTheFailingChunk() returns error? {
    string indexName = containerIndexName("bulk-stop");
    DenseSearch configMode = {queryMode: ai:DENSE, indexConfig: {dimension: CONTAINER_DIMENSION}};
    Configuration config = {maxBulkSize: 2};
    VectorStore store = check newContainerStore(indexName, configMode, config);

    // Batches at `maxBulkSize: 2` are [ok-1, bad], [ok-3, ok-4]. `add` fails on the first batch,
    // so the second is never sent -- which is what distinguishes real chunking from one large
    // request, where `_bulk`'s per-item semantics would have indexed ok-3 and ok-4 as well.
    ai:Error? result = store.add([
        entry("ok-1", queryVector()),
        {id: "bad", embedding: [0.1, 0.2, 0.3], chunk: <ai:TextChunk>{content: "wrong dimension"}},
        entry("ok-3", queryVector()),
        entry("ok-4", queryVector())
    ]);

    test:assertTrue(result is ai:Error, "a rejected entry should fail the 'add'");
    if result is ai:Error {
        test:assertTrue(result.message().includes("bad"),
                string `the error should name the failing entry, got: ${result.message()}`);
    }
    test:assertEquals(check documentCount(indexName), 1,
            "only the first chunk's valid entry should have been indexed");
    check store.close();
}

@test:Config {groups: ["docker"]}
isolated function testContainerBulkPartialFailureIsDetectedAtHttp200() returns error? {
    string indexName = containerIndexName("bulk-partial");
    VectorStore store = check newContainerStore(indexName);

    // OpenSearch returns HTTP 200 for this request; only `errors: true` in the body reveals that
    // one item was rejected.
    ai:Error? result = store.add([
        entry("good", queryVector()),
        {id: "oversized", embedding: [0.1, 0.2, 0.3, 0.4, 0.5], chunk: <ai:TextChunk>{content: "5 dims"}}
    ]);

    test:assertTrue(result is ai:Error, "a per-item failure at HTTP 200 must not be reported as success");
    if result is ai:Error {
        test:assertTrue(result.message().includes("oversized"),
                string `the error should name the failing entry, got: ${result.message()}`);
    }
    test:assertEquals(check documentCount(indexName), 1, "the valid entry in the batch is still indexed");
    check store.close();
}

@test:Config {groups: ["docker"]}
isolated function testContainerBulkDeleteChunkIsAtomicPerItem() returns error? {
    string indexName = containerIndexName("bulk-del");
    VectorStore store = check newContainerStore(indexName);

    ai:VectorEntry[] entries = [];
    string[] ids = [];
    foreach int i in 0 ..< 20 {
        string id = string `del-${i}`;
        ids.push(id);
        entries.push(entry(id, vec(1.0, <float>i / 100.0)));
    }
    check store.add(entries);
    test:assertEquals(check documentCount(indexName), 20);

    // A single `_bulk` carrying both real and imaginary ids: the `not_found` items must be
    // dropped by `extractDeleteFailures` while the real ones still take effect.
    check store.delete([...ids, "ghost-1", "ghost-2"]);
    test:assertEquals(check documentCount(indexName), 0);
    check store.close();
}
