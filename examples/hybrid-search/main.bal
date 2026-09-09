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
configurable string indexName = "hybrid-search-example";
configurable opensearch:ManagedDomainDeployment deployment = {
    deploymentType: opensearch:MANAGED_DOMAIN,
    auth: auth:DEFAULT_CREDENTIALS
};

type Article record {
    string title;
    string language;
    // The dense half, from an embedding model.
    float[] dense;
    // The sparse half, from a learned sparse encoder such as SPLADE: term ids and their weights.
    // Sent to OpenSearch as they are -- no model is deployed server-side.
    int[] termIds;
    float[] termWeights;
};

// Toy vectors standing in for real model output, small enough to read at a glance. "Deploying to
// Kubernetes" is the closest dense match to the query below, while "Ballerina on Kubernetes" is
// the strongest term match -- so the two sub-queries disagree, which is the point of fusing them.
public function main() returns error? {
    opensearch:VectorStore vectorStore = check new (
        serviceUrl,
        region,
        indexName,
        deployment,
        {
            queryMode: ai:HYBRID,
            indexConfig: {dimension: 4},
            // Leaning on the sparse side. The weights must sum to 1.0; the pipeline that applies
            // them is sent inline with every query, so nothing is provisioned on the cluster and
            // retuning these takes effect immediately.
            //
            // Normalizing two differently-scaled scores onto a shared range is one way to fuse
            // them. The other is to ignore the scores and combine the two rankings instead, which
            // needs no comparable scales at all: `fusion: {technique: opensearch:RRF}` on
            // OpenSearch 2.19+. It takes no weights, so this example uses the weighted form.
            fusion: {denseWeight: 0.4, sparseWeight: 0.6}
        }
    );

    Article[] articles = [
        {
            title: "Getting started with Ballerina",
            language: "en",
            dense: [0.90, 0.10, 0.05, 0.02],
            termIds: [101, 205],
            termWeights: [2.1, 0.4]
        },
        {
            title: "Deploying to Kubernetes",
            language: "en",
            dense: [0.20, 0.85, 0.10, 0.05],
            termIds: [310],
            termWeights: [1.2]
        },
        {
            title: "Ballerina on Kubernetes",
            language: "en",
            dense: [0.35, 0.60, 0.10, 0.05],
            termIds: [101, 310],
            termWeights: [3.4, 2.8]
        },
        {
            title: "Ballerina pour debutants",
            language: "fr",
            dense: [0.88, 0.12, 0.06, 0.03],
            termIds: [101],
            termWeights: [2.9]
        }
    ];

    ai:VectorEntry[] entries = from Article article in articles
        select {
            id: article.title,
            // A HYBRID store requires both halves on every entry. An entry missing one would score
            // zero on that sub-query and be penalised by the combination rather than ignored.
            embedding: {
                dense: article.dense,
                sparse: {indices: article.termIds, values: article.termWeights}
            },
            chunk: <ai:TextChunk>{
                content: article.title,
                metadata: {"language": article.language}
            }
        };
    check vectorStore.add(entries);
    io:println(string `Indexed ${entries.length()} articles.`);

    // Dense-nearest to "Deploying to Kubernetes"; term-nearest to "Ballerina on Kubernetes".
    ai:VectorMatch[] matches = check vectorStore.query({
        embedding: {
            dense: [0.25, 0.80, 0.10, 0.05],
            sparse: {indices: [101, 310], values: [2.0, 2.0]}
        },
        // Applied to both sub-queries, which is what keeps the French article out of either.
        filters: {filters: [{key: "language", operator: ai:EQUAL, value: "en"}]},
        topK: 3
    });

    // Scores are the fused, normalized result in (0.0, 1.0] -- not a cosine, and not the
    // unbounded dot product a SPARSE store would return.
    io:println("\nTop matches:");
    foreach ai:VectorMatch item in matches {
        io:println(string `  ${item.similarityScore}  ${item.chunk.content.toString()}`);
    }

    check vectorStore.close();
}
