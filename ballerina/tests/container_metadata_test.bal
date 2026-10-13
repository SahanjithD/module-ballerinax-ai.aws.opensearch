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

// Metadata written and read back through OpenSearch.

import ballerina/ai;
import ballerina/test;
import ballerina/time;

@test:Config {groups: ["docker"]}
isolated function testContainerMetadataKnownFieldsRoundTrip() returns error? {
    string indexName = containerIndexName("md-known");
    VectorStore store = check newContainerStore(indexName);

    time:Utc created = check time:utcFromString("2023-11-14T22:13:20Z");
    time:Utc modified = check time:utcFromString("2024-02-29T08:00:00Z");
    ai:Metadata metadata = {
        fileName: "notes.txt",
        mimeType: "text/plain",
        fileSize: 2048.0d,
        createdAt: created,
        modifiedAt: modified
    };
    check store.add([entry("md", queryVector(), "doc", metadata)]);

    ai:VectorMatch[] matches = check store.query({embedding: queryVector(), topK: 1});
    ai:Metadata? readBack = matches[0].chunk?.metadata;
    if readBack is () {
        test:assertFail("metadata should have been returned");
    }
    test:assertEquals(readBack.fileName, "notes.txt");
    test:assertEquals(readBack.mimeType, "text/plain");
    // `fileSize` is `decimal`, which JSON cannot carry, so it is converted back explicitly.
    test:assertEquals(readBack.fileSize, 2048.0d);
    test:assertEquals(readBack.createdAt, created);
    test:assertEquals(readBack.modifiedAt, modified);
    check store.close();
}

@test:Config {groups: ["docker"]}
isolated function testContainerMetadataCustomKeysRoundTrip() returns error? {
    string indexName = containerIndexName("md-custom");
    VectorStore store = check newContainerStore(indexName);

    check store.add([
        entry("md", queryVector(), "doc", {
                                              "language": "en",
                                              "year": 2024,
                                              "published": true,
                                              "attrs": {"nested": "value", "depth": 2}
                                          })
    ]);

    ai:VectorMatch[] matches = check store.query({embedding: queryVector(), topK: 1});
    ai:Metadata? readBack = matches[0].chunk?.metadata;
    if readBack is () {
        test:assertFail("metadata should have been returned");
    }
    test:assertEquals(readBack["language"], "en", "a string should pass through unchanged");
    test:assertEquals(readBack["year"], 2024, "an integer should pass through unchanged");
    test:assertEquals(readBack["published"], true, "a boolean should pass through unchanged");
    test:assertEquals(readBack["attrs"], <json>{"nested": "value", "depth": 2},
            "a nested object should pass through unchanged");
    check store.close();
}

@test:Config {groups: ["docker"]}
isolated function testContainerMetadataFractionalNumberIsReadBackAsDecimal() returns error? {
    string indexName = containerIndexName("md-frac");
    VectorStore store = check newContainerStore(indexName);
    check store.add([entry("md", queryVector(), "doc", {"rating": 4.25})]);

    ai:VectorMatch[] matches = check store.query({embedding: queryVector(), topK: 1});
    ai:Metadata? readBack = matches[0].chunk?.metadata;
    if readBack is () {
        test:assertFail("metadata should have been returned");
    }
    // Ballerina parses a non-integral JSON number as `decimal`; the value survives, the type does
    // not.
    test:assertEquals(readBack["rating"], 4.25d, "a fractional value is read back as 'decimal'");
    check store.close();
}

@test:Config {groups: ["docker"]}
isolated function testContainerEntryWithoutMetadataRoundTrips() returns error? {
    string indexName = containerIndexName("md-none");
    VectorStore store = check newContainerStore(indexName);
    check store.add([entry("plain", queryVector(), "no metadata")]);

    ai:VectorMatch[] matches = check store.query({embedding: queryVector(), topK: 1});
    ai:Metadata? readBack = matches[0].chunk?.metadata;
    // No metadata is stored as an empty object, which reads back as `{}` rather than `()`.
    test:assertEquals(readBack, <ai:Metadata>{}, "absent metadata should read back empty, not null");
    test:assertEquals(matches[0].chunk.content, "no metadata");
    check store.close();
}

@test:Config {groups: ["docker"]}
isolated function testContainerMetadataUnderFlatSchemaRoundTrips() returns error? {
    string indexName = containerIndexName("md-flat");
    DenseSearch configMode = {queryMode: ai:DENSE, indexConfig: {dimension: CONTAINER_DIMENSION}};
    Configuration config = {metadataFieldName: ""};
    VectorStore store = check newContainerStore(indexName, configMode, config);
    check store.add([entry("flat", queryVector(), "flat doc", {"language": "en", "year": 2024})]);

    ai:VectorMatch[] matches = check store.query({embedding: queryVector(), topK: 1});
    ai:Metadata? readBack = matches[0].chunk?.metadata;
    if readBack is () {
        test:assertFail("metadata should have been returned");
    }
    // Under a flat schema, metadata is every top-level field that is not a fixed schema field.
    test:assertEquals(readBack["language"], "en");
    test:assertEquals(readBack["year"], 2024);
    test:assertFalse(readBack.hasKey("content"), "the fixed schema fields must not leak into metadata");
    test:assertFalse(readBack.hasKey("embedding"), "the fixed schema fields must not leak into metadata");
    test:assertFalse(readBack.hasKey("doc_id"), "the fixed schema fields must not leak into metadata");
    test:assertFalse(readBack.hasKey("chunk_type"), "the fixed schema fields must not leak into metadata");
    check store.close();
}

@test:Config {groups: ["docker"]}
isolated function testContainerFlatSchemaKeyCollisionIsRejectedBeforeWriting() returns error? {
    string indexName = containerIndexName("md-collide");
    DenseSearch configMode = {queryMode: ai:DENSE, indexConfig: {dimension: CONTAINER_DIMENSION}};
    Configuration config = {metadataFieldName: ""};
    VectorStore store = check newContainerStore(indexName, configMode, config);

    // A `content` metadata key would overwrite the chunk content under a flat schema.
    ai:Error? result = store.add([entry("collide", queryVector(), "real content", {"content": "hijacked"})]);
    test:assertTrue(result is ai:Error, "a colliding metadata key should be rejected");
    test:assertEquals(check documentCount(indexName), 0, "nothing should have been written");
    check store.close();
}

@test:Config {groups: ["docker"]}
isolated function testContainerMetadataSurvivesUpsert() returns error? {
    string indexName = containerIndexName("md-upsert");
    VectorStore store = check newContainerStore(indexName);

    check store.add([entry("m", queryVector(), "v1", {"language": "en", "year": 2020})]);
    check store.add([entry("m", queryVector(), "v2", {"language": "fr"})]);

    ai:VectorMatch[] matches = check store.query({embedding: queryVector(), topK: 10});
    test:assertEquals(matches.length(), 1);
    ai:Metadata? readBack = matches[0].chunk?.metadata;
    if readBack is () {
        test:assertFail("metadata should have been returned");
    }
    // An upsert replaces the whole document rather than merging.
    test:assertEquals(readBack["language"], "fr");
    test:assertFalse(readBack.hasKey("year"), "an upsert should replace metadata, not merge it");
    check store.close();
}
