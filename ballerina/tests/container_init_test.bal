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

// `ensureIndex` and `buildIndexMapping` against a real cluster. `mapping_test.bal` already asserts
// the *shape* of the JSON `buildIndexMapping` produces; what it cannot show is whether OpenSearch
// accepts that JSON and materialises the field types the rest of the module depends on. A mapping
// that OpenSearch silently reinterprets -- a `text` field where a `keyword` was intended -- passes
// every unit test and then returns zero hits in production.

import ballerina/ai;
import ballerina/test;

@test:Config {groups: ["docker"]}
isolated function testContainerInitCreatesIndex() returns error? {
    string indexName = containerIndexName("init-create");
    test:assertFalse(check rawIndexExists(indexName), "the index must not exist before 'init'");

    VectorStore store = check newContainerStore(indexName);
    test:assertTrue(check rawIndexExists(indexName), "'init' should have created the index");
    check store.close();
}

@test:Config {groups: ["docker"]}
isolated function testContainerInitOnExistingIndexIsANoOp() returns error? {
    string indexName = containerIndexName("init-existing");
    VectorStore first = check newContainerStore(indexName);
    check first.add([entry("seed", queryVector())]);

    // A second store on the same index must take the `exists` short-circuit in `ensureIndex`
    // rather than re-issuing `PUT /<index>`, which would fail with 400.
    VectorStore second = check newContainerStore(indexName);
    ai:VectorMatch[] matches = check second.query({embedding: queryVector(), topK: 10});
    test:assertEquals(matches.length(), 1, "the second store should see the first store's data");

    check first.close();
    check second.close();
}

@test:Config {groups: ["docker"]}
isolated function testContainerCreateIndexIfNotExistsFalseSkipsCreation() returns error? {
    string indexName = containerIndexName("init-nocreate");
    Configuration config = {
        indexConfig: {dimension: CONTAINER_DIMENSION, createIndexIfNotExists: false}
    };
    VectorStore store = check newContainerStore(indexName, config);
    test:assertFalse(check rawIndexExists(indexName),
            "'createIndexIfNotExists: false' must perform no network I/O and create nothing");
    check store.close();
}

@test:Config {groups: ["docker"]}
isolated function testContainerMappingEnablesKnn() returns error? {
    string indexName = containerIndexName("init-knn");
    VectorStore store = check newContainerStore(indexName);

    map<json> settings = check indexSettings(indexName);
    // OpenSearch echoes index settings back as strings, hence the `toString()` rather than a
    // comparison against the boolean `true` that was sent.
    test:assertEquals(settings["knn"].toString(), "true",
            "'index.knn' must be enabled, or a 'knn' query is rejected outright");
    check store.close();
}

@test:Config {groups: ["docker"]}
isolated function testContainerMappingVectorFieldShape() returns error? {
    string indexName = containerIndexName("init-vecfield");
    Configuration config = {
        indexConfig: {dimension: CONTAINER_DIMENSION, efConstruction: 200, m: 24}
    };
    VectorStore store = check newContainerStore(indexName, config, containerDeployment(FAISS));

    map<json> mappings = check indexMapping(indexName);
    map<json> properties = check mapField(mappings, "properties");
    map<json> vectorField = check mapField(properties, "embedding");
    test:assertEquals(vectorField["type"], "knn_vector");
    test:assertEquals(vectorField["dimension"], CONTAINER_DIMENSION);

    map<json> method = check mapField(vectorField, "method");
    test:assertEquals(method["name"], "hnsw");
    test:assertEquals(method["engine"], "faiss");

    map<json> parameters = check mapField(method, "parameters");
    test:assertEquals(parameters["ef_construction"], 200);
    test:assertEquals(parameters["m"], 24);
    check store.close();
}

@test:Config {groups: ["docker"], dataProvider: spaceTypeDataProvider}
isolated function testContainerMappingSpaceType(ai:SimilarityMetric metric, string expectedSpaceType)
        returns error? {
    string indexName = containerIndexName("init-space");
    Configuration config = {
        indexConfig: {dimension: CONTAINER_DIMENSION, similarityMetric: metric}
    };
    VectorStore store = check newContainerStore(indexName, config);

    map<json> mappings = check indexMapping(indexName);
    map<json> properties = check mapField(mappings, "properties");
    map<json> vectorField = check mapField(properties, "embedding");
    map<json> method = check mapField(vectorField, "method");
    test:assertEquals(method["space_type"], expectedSpaceType,
            string `'${metric}' should map to the '${expectedSpaceType}' space`);
    check store.close();
}

isolated function spaceTypeDataProvider() returns [ai:SimilarityMetric, string][] => [
    [ai:COSINE, "cosinesimil"],
    [ai:EUCLIDEAN, "l2"],
    [ai:DOT_PRODUCT, "innerproduct"]
];

@test:Config {groups: ["docker"]}
isolated function testContainerLuceneEngineIsAccepted() returns error? {
    string indexName = containerIndexName("init-lucene");
    Configuration config = {indexConfig: {dimension: CONTAINER_DIMENSION}};
    VectorStore store = check newContainerStore(indexName, config, containerDeployment(LUCENE));

    map<json> mappings = check indexMapping(indexName);
    map<json> properties = check mapField(mappings, "properties");
    map<json> vectorField = check mapField(properties, "embedding");
    map<json> method = check mapField(vectorField, "method");
    test:assertEquals(method["engine"], "lucene");

    // Prove the index is usable and not merely creatable.
    check store.add([entry("a", queryVector())]);
    ai:VectorMatch[] matches = check store.query({embedding: queryVector(), topK: 1});
    test:assertEquals(matches.length(), 1);
    check store.close();
}

@test:Config {groups: ["docker"]}
isolated function testContainerMappingHonoursCustomFieldNames() returns error? {
    string indexName = containerIndexName("init-fields");
    Configuration config = {
        indexConfig: {dimension: CONTAINER_DIMENSION},
        vectorFieldName: "vec",
        contentFieldName: "body",
        idFieldName: "entry_id",
        metadataFieldName: "props"
    };
    VectorStore store = check newContainerStore(indexName, config);

    map<json> mappings = check indexMapping(indexName);
    map<json> properties = check mapField(mappings, "properties");
    map<json> vectorField = check mapField(properties, "vec");
    test:assertEquals(vectorField["type"], "knn_vector");
    map<json> contentField = check mapField(properties, "body");
    test:assertEquals(contentField["type"], "text");
    map<json> idField = check mapField(properties, "entry_id");
    test:assertEquals(idField["type"], "keyword");
    test:assertTrue(properties.hasKey("props"), "the metadata object should use the configured name");
    test:assertFalse(properties.hasKey("embedding"), "the default field names should not also be mapped");

    // The renamed fields have to survive a write/read round trip, not just appear in the mapping.
    check store.add([entry("custom", queryVector(), "renamed fields")]);
    ai:VectorMatch[] matches = check store.query({embedding: queryVector(), topK: 1});
    test:assertEquals(matches.length(), 1);
    test:assertEquals(matches[0].id, "custom");
    test:assertEquals((<ai:TextChunk>matches[0].chunk).content, "renamed fields");
    check store.close();
}

@test:Config {groups: ["docker"]}
isolated function testContainerStringMetadataIsMappedAsKeyword() returns error? {
    string indexName = containerIndexName("init-dyntpl");
    VectorStore store = check newContainerStore(indexName);
    check store.add([entry("k", queryVector(), "doc", {"language": "en"})]);

    map<json> mappings = check indexMapping(indexName);
    map<json> properties = check mapField(mappings, "properties");
    map<json> metadata = check mapField(properties, "metadata");
    map<json> metadataProperties = check mapField(metadata, "properties");
    map<json> language = check mapField(metadataProperties, "language");
    // Without the `dynamic_templates` block, OpenSearch would map this as analyzed `text` and
    // every `term` filter on it would silently match nothing.
    test:assertEquals(language["type"], "keyword",
            "the dynamic template must map string metadata to 'keyword', not 'text'");
    check store.close();
}

@test:Config {groups: ["docker"]}
isolated function testContainerConcurrentInitOnSameIndexSucceeds() returns error? {
    string indexName = containerIndexName("init-race");
    // Whichever caller loses the race sees `resource_already_exists_exception` from `PUT /<index>`;
    // `isAlreadyExistsError` is what stops that from surfacing as a construction failure. There is
    // no way to provoke this deterministically, so this asserts the invariant instead: concurrent
    // construction on one index name never fails, however the interleaving lands.
    future<ai:Error?> first = start initAndClose(indexName);
    future<ai:Error?> second = start initAndClose(indexName);
    future<ai:Error?> third = start initAndClose(indexName);

    ai:Error? firstResult = check wait first;
    ai:Error? secondResult = check wait second;
    ai:Error? thirdResult = check wait third;

    test:assertTrue(firstResult is (), "concurrent 'init' should not fail: " + errorText(firstResult));
    test:assertTrue(secondResult is (), "concurrent 'init' should not fail: " + errorText(secondResult));
    test:assertTrue(thirdResult is (), "concurrent 'init' should not fail: " + errorText(thirdResult));
    test:assertTrue(check rawIndexExists(indexName), "the index should exist once the race resolves");
}

isolated function initAndClose(string indexName) returns ai:Error? {
    VectorStore store = check newContainerStore(indexName);
    return store.close();
}

isolated function errorText(ai:Error? err) returns string => err is ai:Error ? err.message() : "";
