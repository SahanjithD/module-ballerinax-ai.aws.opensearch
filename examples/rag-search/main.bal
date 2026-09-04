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
import ballerina/io;
import ballerinax/ai.aws.opensearch;
import ballerinax/aws.auth;

configurable string serviceUrl = ?;
configurable string region = "us-east-1";
configurable string indexName = "rag-search-example";
// The managed-domain variant of `opensearch:Deployment`. `auth` has no default on any variant, so
// opting into the ambient AWS credential chain is stated rather than assumed.
configurable opensearch:ManagedDomainDeployment deployment = {
    deploymentType: opensearch:MANAGED_DOMAIN,
    auth: auth:DEFAULT_CREDENTIALS
};

type Article record {
    string title;
    string language;
    float[] embedding;
};

// Toy embeddings standing in for a real embedding model's output — small enough to read at a
// glance, and close enough to each other that the query embedding below is nearest to "article-1".
public function main() returns error? {
    opensearch:VectorStore vectorStore = check new (
        serviceUrl,
        region,
        indexName,
        deployment,
        {indexConfig: {dimension: 4}}
    );

    Article[] articles = [
        {title: "Getting started with Ballerina", language: "en", embedding: [0.90, 0.10, 0.05, 0.02]},
        {title: "Deploying to Kubernetes", language: "en", embedding: [0.20, 0.85, 0.10, 0.05]},
        {title: "Ballerina pour débutants", language: "fr", embedding: [0.88, 0.12, 0.06, 0.03]}
    ];

    ai:Error? addResult = vectorStore.add(from Article article in articles
        select {
            embedding: article.embedding,
            chunk: {
                'type: "text-chunk",
                content: article.title,
                metadata: {"language": article.language}
            }
        }
    );
    if addResult is ai:Error {
        io:println("Error occurred while adding entries to the vector store: ", addResult);
        return;
    }

    // A query embedding close to the "Getting started with Ballerina" article, restricted to
    // English content — so the French article with a similar embedding is excluded.
    ai:Vector queryEmbedding = [0.91, 0.09, 0.05, 0.03];
    ai:VectorMatch[] results = check vectorStore.query({
        embedding: queryEmbedding,
        topK: 5,
        filters: {filters: [{'key: "language", operator: ai:EQUAL, value: "en"}]}
    });

    io:println("Query results:");
    foreach ai:VectorMatch 'match in results {
        io:println(string `  - ${'match.chunk.content.toString()} (score: ${'match.similarityScore})`);
    }

    check vectorStore.close();
}
