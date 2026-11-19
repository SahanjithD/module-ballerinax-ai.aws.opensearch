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

// The two public stores are thin fronts over `VectorStoreCore`, which the other suites test. These
// cover what the fronts add: how their parameters reach the core.

import ballerina/ai;
import ballerina/test;

@test:Config
isolated function testSharedStoreConfigurationCopiesEverySharedField() {
    ManagedDomainStoreConfiguration managed = {
        contentFieldName: "text",
        idFieldName: "entry_id",
        metadataFieldName: "",
        includeEmbeddingsInResults: true,
        maxBulkSize: 7,
        maxResultWindow: 42,
        createIndexIfNotExists: false,
        retryConfig: {maxRetries: 1, initialDelay: 0.1, maxDelay: 0.2, backoffFactor: 3.0},
        engine: LUCENE,
        refreshOnWrite: true
    };
    StoreConfiguration expected = {
        contentFieldName: "text",
        idFieldName: "entry_id",
        metadataFieldName: "",
        includeEmbeddingsInResults: true,
        maxBulkSize: 7,
        maxResultWindow: 42,
        createIndexIfNotExists: false,
        retryConfig: {maxRetries: 1, initialDelay: 0.1, maxDelay: 0.2, backoffFactor: 3.0}
    };
    test:assertEquals(sharedStoreConfiguration(managed), expected);
}

// Validation runs before any request, so an unreachable URL is fine.
@test:Config
isolated function testServerlessStoreRejectsCompressionWithSparse() {
    ServerlessVectorStore|ai:Error result = new (UNREACHABLE_URL, "us-east-1", "idx",
        {generation: NEXTGEN, auth: TEST_CREDENTIALS, compressionLevel: COMPRESSION_1X}, {queryMode: ai:SPARSE}
    );
    if result !is ai:Error {
        test:assertFail("'compressionLevel' with a SPARSE store should fail construction");
    }
    test:assertTrue(result.message().includes("compressionLevel"), result.message());
}

@test:Config {groups: ["docker"]}
isolated function testContainerManagedDomainStoreRoundTrip() returns error? {
    string indexName = containerIndexName("public-managed");
    ManagedDomainVectorStore store = check new (containerUrl, CONTAINER_REGION, indexName, containerAuth,
        containerDenseMode(), {refreshOnWrite: true, includeEmbeddingsInResults: true}
    );
    check store.add([entry("p-1", queryVector(), "through the public store")]);

    // Visible straight away only because `refreshOnWrite` reached the transport.
    ai:VectorMatch[] matches = check store.query({embedding: queryVector(), topK: 1});
    test:assertEquals(matches.length(), 1);
    test:assertEquals(matches[0].id, "p-1");
    test:assertEquals(matches[0].embedding, queryVector(), "'includeEmbeddingsInResults' should reach the query");

    check store.delete("p-1");
    test:assertEquals(check documentCount(indexName), 0);
}

@test:Config {groups: ["docker"]}
isolated function testContainerManagedDomainStorePassesTheEngine() returns error? {
    string indexName = containerIndexName("public-engine");
    _ = check new ManagedDomainVectorStore(containerUrl, CONTAINER_REGION, indexName, containerAuth,
        containerDenseMode(), {engine: LUCENE}
    );
    map<json> mapping = check indexMapping(indexName);
    map<json> properties = check mapField(mapping, "properties");
    map<json> embedding = check mapField(properties, "embedding");
    map<json> method = check mapField(embedding, "method");
    test:assertEquals(method["engine"], "lucene");
}
