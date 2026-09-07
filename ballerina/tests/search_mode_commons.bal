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

// `SearchMode` builders for the offline unit tests, mirroring `deployment_commons.bal`.
//
// `queryMode` is a required field on every variant -- deliberately, so a record literal resolves
// to exactly one member of the union -- but restating it at every call site says nothing a
// reader does not already know from the builder's name. These supply it.

import ballerina/ai;

# A dense search mode over a small vector, sized for readable request bodies.
#
# + dimension - The vector dimension
# + similarityMetric - The similarity metric
# + return - The search mode
isolated function denseMode(int dimension = 3, ai:SimilarityMetric similarityMetric = ai:COSINE)
        returns DenseSearch =>
    {queryMode: ai:DENSE, indexConfig: {dimension, similarityMetric}};

# A sparse search mode.
#
# + sparseVectorFieldName - The `rank_features` field name
# + maxQueryTokens - The client-side ceiling on query token count
# + return - The search mode
isolated function sparseMode(string sparseVectorFieldName = "sparse_embedding", int maxQueryTokens = 1024)
        returns SparseSearch =>
    {queryMode: ai:SPARSE, sparseVectorFieldName, maxQueryTokens};

# A hybrid search mode.
#
# + fusion - How the dense and sparse scores are fused; inline with the defaults unless overridden
# + dimension - The dense vector dimension
# + return - The search mode
isolated function hybridMode(HybridFusion fusion = {}, int dimension = 3) returns HybridSearch =>
    {queryMode: ai:HYBRID, indexConfig: {dimension}, fusion};
