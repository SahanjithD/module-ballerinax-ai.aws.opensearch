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

// `add`, `query` and `delete` end to end, with `refreshOnWrite` so writes are visible at once.

import ballerina/ai;
import ballerina/http;
import ballerina/test;

@test:Config {groups: ["docker"]}
isolated function testContainerAddQueryRoundTrip() returns error? {
    string indexName = containerIndexName("rt");
    VectorStoreCore store = check newContainerStore(indexName, containerDenseMode(), {includeEmbeddingsInResults: true});

    check store.add([entry("rt-1", queryVector(), "round trip content")]);
    ai:VectorMatch[] matches = check store.query({embedding: queryVector(), topK: 10});

    test:assertEquals(matches.length(), 1);
    ai:VectorMatch first = matches[0];
    test:assertEquals(first.id, "rt-1");
    test:assertEquals(first.chunk.content, "round trip content");
    test:assertEquals(first.embedding, queryVector(), "the stored embedding should come back intact");
}

@test:Config {groups: ["docker"]}
isolated function testContainerAddGeneratesIdWhenAbsent() returns error? {
    string indexName = containerIndexName("genid");
    VectorStoreCore store = check newContainerStore(indexName);

    check store.add([entry((), queryVector(), "no id supplied")]);
    ai:VectorMatch[] matches = check store.query({embedding: queryVector(), topK: 10});

    test:assertEquals(matches.length(), 1);
    test:assertTrue((matches[0].id ?: "").length() > 0, "a generated id should have been stored and returned");
    test:assertEquals(matches[0].chunk.content, "no id supplied");
}

@test:Config {groups: ["docker"]}
isolated function testContainerAddSameIdTwiceUpserts() returns error? {
    string indexName = containerIndexName("upsert");
    VectorStoreCore store = check newContainerStore(indexName);

    check store.add([entry("dup", queryVector(), "v1")]);
    check store.add([entry("dup", queryVector(), "v2")]);

    test:assertEquals(check documentCount(indexName), 1,
            "re-adding the same id on a managed domain must upsert, not duplicate");
    ai:VectorMatch[] matches = check store.query({embedding: queryVector(), topK: 10});
    test:assertEquals(matches.length(), 1);
    test:assertEquals(matches[0].chunk.content, "v2", "the second write should have replaced the first");
}

@test:Config {groups: ["docker"]}
isolated function testContainerAddPreservesChunkType() returns error? {
    string indexName = containerIndexName("chunktype");
    VectorStoreCore store = check newContainerStore(indexName);

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
}

@test:Config {groups: ["docker"]}
isolated function testContainerDeleteSingleId() returns error? {
    string indexName = containerIndexName("del-one");
    VectorStoreCore store = check newContainerStore(indexName);

    check store.add([entry("keep", queryVector()), entry("drop", vec(0.0, 1.0))]);
    test:assertEquals(check documentCount(indexName), 2);

    check store.delete("drop");
    ai:VectorMatch[] matches = check store.query({topK: 10});
    assertIdsEqual(matches, ["keep"], "only the deleted entry should be gone");
}

@test:Config {groups: ["docker"]}
isolated function testContainerDeleteMultipleIds() returns error? {
    string indexName = containerIndexName("del-many");
    VectorStoreCore store = check newContainerStore(indexName);

    check store.add([
        entry("a", queryVector()),
        entry("b", vec(0.0, 1.0)),
        entry("c", vec(0.0, 0.0, 1.0)),
        entry("d", vec(0.0, 0.0, 0.0, 1.0))
    ]);

    check store.delete(["b", "d"]);
    ai:VectorMatch[] matches = check store.query({topK: 10});
    assertIdsEqual(matches, ["a", "c"], "exactly the requested ids should have been deleted");
}

@test:Config {groups: ["docker"]}
isolated function testContainerDeleteMissingIdIsNotAnError() returns error? {
    string indexName = containerIndexName("del-missing");
    VectorStoreCore store = check newContainerStore(indexName);
    check store.add([entry("present", queryVector())]);

    // A per-item `not_found` is not a failure.
    check store.delete("never-existed");
    check store.delete(["present", "also-never-existed"]);

    test:assertEquals(check documentCount(indexName), 0);
}

@test:Config {groups: ["docker"]}
isolated function testContainerEmptyAddAndDeleteAreNoOps() returns error? {
    string indexName = containerIndexName("empty");
    VectorStoreCore store = check newContainerStore(indexName);

    check store.add([]);
    check store.delete([]);
    test:assertEquals(check documentCount(indexName), 0);

    check store.add([entry("x", queryVector())]);
    check store.add([]);
    test:assertEquals(check documentCount(indexName), 1, "an empty 'add' must not disturb existing data");
}

@test:Config {groups: ["docker"]}
isolated function testContainerAddRejectsZeroVectorUnderCosine() returns error? {
    string indexName = containerIndexName("zerovec");
    VectorStoreCore store = check newContainerStore(indexName);

    // Rejected before sending, so the index is untouched.
    ai:Error? result = store.add([entry("zero", vec(0.0, 0.0, 0.0, 0.0))]);
    test:assertTrue(result is ai:Error, "a zero vector under COSINE should be rejected");
    test:assertEquals(check documentCount(indexName), 0, "nothing should have been written");
}

# Makes everything written so far searchable; Serverless Classic has no `refreshOnWrite`.
#
# + indexName - The index to refresh
# + return - An error if the refresh fails
isolated function rawRefresh(string indexName) returns error? {
    http:Client cl = check rawClient();
    http:Response _ = check cl->post(string `/${indexName}/_refresh`, ());
}

# A Classic store against the container, with a tiny result window so lookups fill it.
#
# + indexName - The target index
# + return - The store, or an `ai:Error`
isolated function classicStoreWithSmallWindow(string indexName) returns VectorStoreCore|ai:Error =>
    new (containerUrl, CONTAINER_REGION, indexName, containerAuth, classicDeployment(), containerDenseMode(),
    {maxResultWindow: 4}
);

@test:Config {groups: ["docker"]}
isolated function testContainerClassicDeletesMoreIdsThanTheWindow() returns error? {
    string indexName = containerIndexName("classic-many");
    VectorStoreCore store = check classicStoreWithSmallWindow(indexName);
    string[] ids = from int i in 0 ..< 10
        select string `c-${i}`;
    check store.add(from string id in ids
        select entry(id, vec(1.0)));
    check rawRefresh(indexName);

    check store.delete(ids);
    check rawRefresh(indexName);
    test:assertEquals(check documentCount(indexName), 0, "every id should be deleted despite a window of 4");
}

@test:Config {groups: ["docker"]}
isolated function testContainerClassicRejectsAnIdWithAFullWindowOfCopies() returns error? {
    string indexName = containerIndexName("classic-copies");
    VectorStoreCore store = check classicStoreWithSmallWindow(indexName);
    // Append-only on Classic: the same id four times is four documents.
    foreach int _ in 0 ..< 4 {
        check store.add([entry("dup", vec(1.0))]);
    }
    check rawRefresh(indexName);

    ai:Error? result = store.delete("dup");
    if result !is ai:Error {
        test:assertFail("an id with a full window of copies may have more, so the delete must fail");
    }
    test:assertTrue(result.message().includes("'dup'"), result.message());
}

@test:Config {groups: ["docker"]}
isolated function testContainerDeleteIsSplitByMaxBulkSize() returns error? {
    string indexName = containerIndexName("delete-batches");
    VectorStoreCore store = check newContainerStore(indexName, containerDenseMode(), {maxBulkSize: 2});
    string[] ids = from int i in 0 ..< 5
        select string `b-${i}`;
    check store.add(from string id in ids
        select entry(id, vec(1.0)));
    test:assertEquals(check documentCount(indexName), 5);

    check store.delete(ids);
    test:assertEquals(check documentCount(indexName), 0, "every batch of the delete should have run");
}

@test:Config {groups: ["docker"]}
isolated function testContainerClassicDeleteIsSplitByMaxBulkSize() returns error? {
    string indexName = containerIndexName("classic-batches");
    VectorStoreCore store = check new (containerUrl, CONTAINER_REGION, indexName, containerAuth, classicDeployment(),
        containerDenseMode(), {maxBulkSize: 2}
    );
    string[] ids = from int i in 0 ..< 5
        select string `cb-${i}`;
    check store.add(from string id in ids
        select entry(id, vec(1.0)));
    check rawRefresh(indexName);

    check store.delete(ids);
    check rawRefresh(indexName);
    test:assertEquals(check documentCount(indexName), 0, "every batch of the delete should have run");
}
