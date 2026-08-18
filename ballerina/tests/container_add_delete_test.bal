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

// `add`, `query`, and `delete` end to end. Everything here runs with `refreshOnWrite`, so a write
// is visible to the next query with no polling -- the determinism `live_test.bal` gives up because
// Serverless will not honour `refresh=wait_for`.

import ballerina/ai;
import ballerina/test;

@test:Config {groups: ["docker"]}
isolated function testContainerAddQueryRoundTrip() returns error? {
    string indexName = containerIndexName("rt");
    VectorStore store = check newContainerStore(indexName);

    check store.add([entry("rt-1", queryVector(), "round trip content")]);
    ai:VectorMatch[] matches = check store.query({embedding: queryVector(), topK: 10});

    test:assertEquals(matches.length(), 1);
    ai:VectorMatch first = matches[0];
    test:assertEquals(first.id, "rt-1");
    test:assertEquals(first.chunk.content, "round trip content");
    test:assertEquals(first.embedding, queryVector(), "the stored embedding should come back intact");
    check store.close();
}

@test:Config {groups: ["docker"]}
isolated function testContainerAddGeneratesIdWhenAbsent() returns error? {
    string indexName = containerIndexName("genid");
    VectorStore store = check newContainerStore(indexName);

    check store.add([entry((), queryVector(), "no id supplied")]);
    ai:VectorMatch[] matches = check store.query({embedding: queryVector(), topK: 10});

    test:assertEquals(matches.length(), 1);
    // The generated UUID must have reached the server as both `_id` and `doc_id`, otherwise the
    // id read back here would be empty or a server-assigned opaque value.
    test:assertTrue((matches[0].id ?: "").length() > 0, "a generated id should have been stored and returned");
    test:assertEquals(matches[0].chunk.content, "no id supplied");
    check store.close();
}

@test:Config {groups: ["docker"]}
isolated function testContainerAddSameIdTwiceUpserts() returns error? {
    string indexName = containerIndexName("upsert");
    VectorStore store = check newContainerStore(indexName);

    check store.add([entry("dup", queryVector(), "v1")]);
    check store.add([entry("dup", queryVector(), "v2")]);

    test:assertEquals(check documentCount(indexName), 1,
            "re-adding the same id on a managed domain must upsert, not duplicate");
    ai:VectorMatch[] matches = check store.query({embedding: queryVector(), topK: 10});
    test:assertEquals(matches.length(), 1);
    test:assertEquals(matches[0].chunk.content, "v2", "the second write should have replaced the first");
    check store.close();
}

@test:Config {groups: ["docker"]}
isolated function testContainerAddPreservesChunkType() returns error? {
    string indexName = containerIndexName("chunktype");
    VectorStore store = check newContainerStore(indexName);

    check store.add([
        {id: "text", embedding: queryVector(), chunk: <ai:TextChunk>{content: "a text chunk"}},
        {id: "custom", embedding: vec(0.0, 1.0), chunk: {'type: "markdown-chunk", content: "# heading"}}
    ]);

    ai:VectorMatch[] matches = check store.query({topK: 10});
    test:assertEquals(matches.length(), 2);
    foreach ai:VectorMatch hit in matches {
        if hit.id == "text" {
            test:assertTrue(hit.chunk is ai:TextChunk,
                    "a 'text-chunk' should reconstruct as an 'ai:TextChunk'");
        } else {
            test:assertEquals(hit.chunk.'type, "markdown-chunk",
                    "a non-text chunk type should round-trip unchanged");
        }
    }
    check store.close();
}

@test:Config {groups: ["docker"]}
isolated function testContainerDeleteSingleId() returns error? {
    string indexName = containerIndexName("del-one");
    VectorStore store = check newContainerStore(indexName);

    check store.add([entry("keep", queryVector()), entry("drop", vec(0.0, 1.0))]);
    test:assertEquals(check documentCount(indexName), 2);

    check store.delete("drop");
    ai:VectorMatch[] matches = check store.query({topK: 10});
    assertIdsEqual(matches, ["keep"], "only the deleted entry should be gone");
    check store.close();
}

@test:Config {groups: ["docker"]}
isolated function testContainerDeleteMultipleIds() returns error? {
    string indexName = containerIndexName("del-many");
    VectorStore store = check newContainerStore(indexName);

    check store.add([
        entry("a", queryVector()),
        entry("b", vec(0.0, 1.0)),
        entry("c", vec(0.0, 0.0, 1.0)),
        entry("d", vec(0.0, 0.0, 0.0, 1.0))
    ]);

    check store.delete(["b", "d"]);
    ai:VectorMatch[] matches = check store.query({topK: 10});
    assertIdsEqual(matches, ["a", "c"], "exactly the requested ids should have been deleted");
    check store.close();
}

@test:Config {groups: ["docker"]}
isolated function testContainerDeleteMissingIdIsNotAnError() returns error? {
    string indexName = containerIndexName("del-missing");
    VectorStore store = check newContainerStore(indexName);
    check store.add([entry("present", queryVector())]);

    // OpenSearch returns a per-item `not_found` here, at HTTP 200. `extractDeleteFailures` has to
    // recognise and drop it; if it did not, this would surface as a spurious failure.
    check store.delete("never-existed");
    check store.delete(["present", "also-never-existed"]);

    test:assertEquals(check documentCount(indexName), 0);
    check store.close();
}

@test:Config {groups: ["docker"]}
isolated function testContainerEmptyAddAndDeleteAreNoOps() returns error? {
    string indexName = containerIndexName("empty");
    VectorStore store = check newContainerStore(indexName);

    check store.add([]);
    check store.delete([]);
    test:assertEquals(check documentCount(indexName), 0);

    check store.add([entry("x", queryVector())]);
    check store.add([]);
    test:assertEquals(check documentCount(indexName), 1, "an empty 'add' must not disturb existing data");
    check store.close();
}

@test:Config {groups: ["docker"]}
isolated function testContainerAddRejectsZeroVectorUnderCosine() returns error? {
    string indexName = containerIndexName("zerovec");
    VectorStore store = check newContainerStore(indexName);

    // The client-side guard must fire before anything is sent, leaving the index untouched --
    // reaching the server would fail the whole `_bulk` request with an opaque exception.
    ai:Error? result = store.add([entry("zero", vec(0.0, 0.0, 0.0, 0.0))]);
    test:assertTrue(result is ai:Error, "a zero vector under COSINE should be rejected");
    test:assertEquals(check documentCount(indexName), 0, "nothing should have been written");
    check store.close();
}
