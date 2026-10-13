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

// Every deployment gets an explicit `method` block: relying on the default engine breaks k-NN
// pre-filtering on Serverless Classic.
# Builds the `knn_vector` field definition.
#
# + indexConfig - The `knn_vector` shape
# + deployment - The deployment, which gates `engine` and the quantization parameter
# + return - The field definition
isolated function buildDenseVectorField(IndexConfig indexConfig, Deployment deployment) returns json {
    string spaceType = toSpaceType(indexConfig.similarityMetric);

    map<json> vectorField = {
        "type": "knn_vector",
        "dimension": indexConfig.dimension
    };
    map<json> method = {"name": "hnsw"};
    if deployment is ServerlessNextGenDeployment {
        // NextGen rejects `engine` inside `method` and fills in Faiss itself.
        vectorField["space_type"] = spaceType;
        CompressionLevel? compressionLevel = deployment.compressionLevel;
        if compressionLevel is CompressionLevel {
            vectorField["compression_level"] = compressionLevel;
        }
    } else {
        method["engine"] = deployment is ManagedDomainDeployment ? deployment.engine : FAISS;
        method["space_type"] = spaceType;
    }
    method["parameters"] = {
        "ef_construction": indexConfig.efConstruction,
        "m": indexConfig.m
    };
    vectorField["method"] = method;
    return vectorField;
}

// The dynamic template maps string metadata to `keyword`; as analyzed `text`, `term` filters would
// match nothing. `index.knn` turns on the k-NN codec, so it is set only with a `knn_vector` field.
# Builds the `PUT /<index>` body: the mode's vector fields plus the content, id and metadata fields.
#
# + searchMode - The kind of search, which decides which vector field(s) the mapping declares
# + config - The vector store configuration
# + deployment - The deployment, which gates `engine` and the quantization parameter
# + return - The index-creation request body
isolated function buildIndexMapping(SearchMode searchMode, Configuration config, Deployment deployment)
        returns json {
    map<json> properties = {
        [config.contentFieldName]: {"type": "text"},
        [config.idFieldName]: {"type": "keyword"},
        "chunk_type": {"type": "keyword"}
    };

    IndexConfig? indexConfig = indexConfigOf(searchMode);
    string? denseFieldName = denseFieldNameOf(searchMode);
    if indexConfig is IndexConfig && denseFieldName is string {
        properties[denseFieldName] = buildDenseVectorField(indexConfig, deployment);
    }
    string? sparseFieldName = sparseFieldNameOf(searchMode);
    if sparseFieldName is string {
        properties[sparseFieldName] = {"type": "rank_features"};
    }

    if config.metadataFieldName != "" {
        properties[config.metadataFieldName] = {
            "type": "object",
            "properties": {
                "createdAt": {"type": "date"},
                "modifiedAt": {"type": "date"},
                "fileName": {"type": "keyword"},
                "mimeType": {"type": "keyword"},
                "fileSize": {"type": "double"}
            }
        };
    }

    string metadataPathMatch = config.metadataFieldName == "" ? "*" : string `${config.metadataFieldName}.*`;

    map<json> mapping = {
        "mappings": {
            "dynamic_templates": [
                {
                    "metadataStringsAsKeyword": {
                        "path_match": metadataPathMatch,
                        "match_mapping_type": "string",
                        "mapping": {"type": "keyword", "ignore_above": 8191}
                    }
                }
            ],
            "properties": properties
        }
    };
    if denseFieldName is string {
        mapping["settings"] = {"index": {"knn": true}};
    }
    return mapping;
}

# Maps `ai:SimilarityMetric` to the OpenSearch `space_type`.
#
# + metric - The similarity metric
# + return - `cosinesimil`, `l2`, or `innerproduct`
isolated function toSpaceType(ai:SimilarityMetric metric) returns string {
    match metric {
        ai:EUCLIDEAN => {
            return "l2";
        }
        ai:DOT_PRODUCT => {
            return "innerproduct";
        }
        _ => {
            return "cosinesimil";
        }
    }
}

// On Serverless Classic a new index can briefly fail searches with `index_not_found_exception`.
// That is not waited out here, so callers that write before they query do not pay for it.
# Creates the index when it is missing and `createIndexIfNotExists` allows it.
#
# + transport - The transport to issue `HEAD`/`PUT` requests through
# + indexName - The index to ensure
# + searchMode - The kind of search, passed through to the mapping builder
# + config - The vector store configuration
# + deployment - The deployment, passed through to the mapping builder
# + return - `true` if this call created the index, `false` if it existed or creation is off, or an
# `ai:Error`
isolated function ensureIndex(OpenSearchTransport transport, string indexName, SearchMode searchMode,
        Configuration config, Deployment deployment) returns boolean|ai:Error {
    if !config.createIndexIfNotExists {
        return false;
    }
    boolean exists = check transport.indexExists(indexName);
    if exists {
        return false;
    }
    json mapping = buildIndexMapping(searchMode, config, deployment);
    ai:Error? result = transport.createIndex(indexName, mapping);
    if result is ai:Error {
        if !isAlreadyExistsError(result) {
            return result;
        }
        // Lost a creation race. Reported as existing, so `verifyIndexMapping` checks the winner's
        // mapping.
        return false;
    }
    return true;
}

# Whether a `createIndex` error is OpenSearch's `resource_already_exists_exception`.
#
# + err - The error to inspect
# + return - `true` if the index already existed
isolated function isAlreadyExistsError(ai:Error err) returns boolean {
    var errorType = err.detail()["openSearchErrorType"];
    return errorType is string && errorType == "resource_already_exists_exception";
}
