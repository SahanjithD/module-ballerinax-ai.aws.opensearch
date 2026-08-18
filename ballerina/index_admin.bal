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

# Builds the `PUT /<index>` request body: a `knn_vector` field sized and spaced per `IndexConfig`,
# the fixed schema fields (content/id/chunk type), and a `dynamic_templates` block mapping
# metadata strings to `keyword` — without it, `term` filters on string metadata silently return
# zero hits, because the default dynamic mapping would map them as analyzed `text`.
#
# `SERVERLESS_NEXTGEN` omits the `method` block entirely (it auto-configures its own engine and
# rejects one being specified) and puts `space_type` at the field's top level instead; every other
# deployment type emits an explicit `method` block, since relying on the default engine is what
# silently breaks k-NN pre-filtering on Serverless Classic.
#
# + config - The vector store configuration
# + deploymentType - The deployment flavour, which gates the `method` block
# + return - The index-creation request body
isolated function buildIndexMapping(Configuration config, DeploymentType deploymentType) returns json {
    string spaceType = toSpaceType(config.indexConfig.similarityMetric);

    map<json> vectorField = {
        "type": "knn_vector",
        "dimension": config.indexConfig.dimension
    };
    if deploymentType == SERVERLESS_NEXTGEN {
        vectorField["space_type"] = spaceType;
    } else {
        vectorField["method"] = {
            "name": "hnsw",
            "engine": config.indexConfig.engine,
            "space_type": spaceType,
            "parameters": {
                "ef_construction": config.indexConfig.efConstruction,
                "m": config.indexConfig.m
            }
        };
    }

    map<json> properties = {
        [config.vectorFieldName]: vectorField,
        [config.contentFieldName]: {"type": "text"},
        [config.idFieldName]: {"type": "keyword"},
        "chunk_type": {"type": "keyword"}
    };
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

    return {
        "settings": {
            "index": {"knn": true}
        },
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

# Ensures the target index exists, creating it if `IndexConfig.createIndexIfNotExists` allows it.
# When it is `false`, this function performs no network I/O at all — this is what lets `init` run
# fully offline for least-privilege deployments and for testing.
#
# A `resource_already_exists_exception` on creation is treated as success, covering two instances
# racing to create the same index.
#
# + transport - The transport to issue `HEAD`/`PUT` requests through
# + indexName - The index to ensure
# + config - The vector store configuration
# + deploymentType - The deployment flavour, passed through to the mapping builder
# + return - An `ai:Error` on failure, otherwise `()`
isolated function ensureIndex(OpenSearchTransport transport, string indexName, Configuration config,
        DeploymentType deploymentType) returns ai:Error? {
    if !config.indexConfig.createIndexIfNotExists {
        return;
    }
    boolean exists = check transport.indexExists(indexName);
    if exists {
        return;
    }
    json mapping = buildIndexMapping(config, deploymentType);
    ai:Error? result = transport.createIndex(indexName, mapping);
    if result is ai:Error && !isAlreadyExistsError(result) {
        return result;
    }
}

# Checks whether an `ai:Error` returned by `createIndex` wraps OpenSearch's
# `resource_already_exists_exception`.
#
# + err - The error to inspect
# + return - `true` if the error's `openSearchErrorType` detail is `resource_already_exists_exception`
isolated function isAlreadyExistsError(ai:Error err) returns boolean {
    var errorType = err.detail()["openSearchErrorType"];
    return errorType is string && errorType == "resource_already_exists_exception";
}
