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

// Integration coverage for SPARSE against the throwaway OpenSearch container.
//
// A unit test can assert that a `rank_features` mapping is emitted; only a real server can prove
// that OpenSearch accepts an index with no `settings` block at all, and that a `neural_sparse`
// query against it works with no ML model, no ingest pipeline and no `index.knn` setting.

import ballerina/ai;
import ballerina/test;

# The sparse search mode the container suite uses.
#
# + return - The search mode
isolated function containerSparseMode() returns SparseSearch => {queryMode: ai:SPARSE};

# Constructs a SPARSE store against the plain-HTTP container.
#
# + indexName - The target index
# + return - The store, or an `ai:Error` if construction fails
isolated function newSparseContainerStore(string indexName) returns VectorStore|ai:Error =>
    newContainerStore(indexName, containerSparseMode());

# Builds a sparse vector entry.
#
# + id - The logical entry id
# + indices - The feature indices
# + values - The feature weights
# + language - The `language` metadata value the filter tests key off
# + return - The entry
isolated function sparseVectorEntry(string id, int[] indices, float[] values, string language = "en")
        returns ai:VectorEntry =>
    {
        id,
        embedding: {indices, values},
        chunk: <ai:TextChunk>{content: string `content for ${id}`, metadata: {"language": language}}
    };

@test:Config {groups: ["docker"]}
isolated function testContainerSparseIndexIsCreatedWithoutSettings() returns error? {
    string indexName = containerIndexName("sparse-init");
    VectorStore store = check newSparseContainerStore(indexName);

    map<json> mappings = check indexMapping(indexName);
    map<json> properties = check mapField(mappings, "properties");
    map<json> sparseField = check mapField(properties, "sparse_embedding");
    test:assertEquals(sparseField["type"], "rank_features");

    // The mapping carries no `settings` block, so OpenSearch must have defaulted `index.knn` to
    // absent rather than rejecting the create.
    map<json> settings = check indexSettings(indexName);
    test:assertFalse(settings.hasKey("knn"),
            string `a SPARSE index must not request the k-NN codec, got: ${settings.toJsonString()}`);
    check store.close();
}

// A zero weight is rejected by Lucene at index time. Asserting the document count stayed at zero
// proves the guard fired client-side rather than the whole _bulk failing at the server.
@test:Config {groups: ["docker"]}
isolated function testContainerSparseZeroWeightIsRejectedBeforeTheWire() returns error? {
    string indexName = containerIndexName("sparse-zero");
    VectorStore store = check newSparseContainerStore(indexName);

    ai:Error? result = store.add([sparseVectorEntry("zero", [7], [0.0])]);
    test:assertTrue(result is ai:Error, "a zero sparse weight must be rejected");
    test:assertEquals(check documentCount(indexName), 0, "nothing should have reached the server");
    check store.close();
}
