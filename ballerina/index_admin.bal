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

# Builds the `knn_vector` field definition, sized and spaced per `IndexConfig`.
#
# Every deployment type gets an explicit `method` block carrying the HNSW `parameters`, since
# relying on the default engine is what silently breaks k-NN pre-filtering on Serverless Classic.
# What varies is only what may accompany the block.
#
# `MANAGED_DOMAIN` and `SERVERLESS_CLASSIC` carry `engine` and `space_type` inside the block —
# Classic hard-coded to Faiss, the only engine it supports. `SERVERLESS_NEXTGEN` gets neither:
# `engine` inside the block fails there with a flat 400 reading "Field parameter 'engine' is not
# supported", raised by the AOSS proxy rather than by OpenSearch, so it carries no error type to
# map. Its `space_type` goes at the field's top level instead, and it is the only deployment that
# may also carry `compression_level` there.
#
# A block of `{"name": "hnsw", "parameters": {...}}` without `engine` was verified accepted on
# NextGen with its parameters stored intact; the stored mapping comes back with `engine: faiss`
# supplied by NextGen itself, alongside the `compression_level` it was sent.
#
# # Quantization on NextGen
# A NextGen field created without one comes back mapped as `on_disk`/`32x`, so every vector is
# quantized unless something says otherwise, with nothing in the request or the response to
# announce it. `ServerlessNextGenDeployment.compressionLevel` is emitted here so that choice can be
# made explicit. It is unset by default, which reproduces the server's behavior exactly.
#
# The sibling `mode` parameter is not emitted, on any deployment. The AOSS proxy rejects every
# value of it outright — `Field parameter 'mode' is not supported`, for both `in_memory` and
# `on_disk` — so a field carrying it could not be created at all, and NextGen is the only flavour
# whose record ever declared it.
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

# Builds the `PUT /<index>` request body: whichever vector field(s) the `SearchMode` calls for, the
# fixed schema fields (content/id/chunk type), and a `dynamic_templates` block mapping metadata
# strings to `keyword` — without it, `term` filters on string metadata silently return zero hits,
# because the default dynamic mapping would map them as analyzed `text`.
#
# # What each mode gets
# `DenseSearch` maps a `knn_vector` field; `SparseSearch` maps a `rank_features` field;
# `HybridSearch` maps both, because a `hybrid` query scores every document on both sub-queries.
#
# `index.knn` is set only when a `knn_vector` field is actually present. It is not an inert flag:
# it switches the index onto the k-NN codec and wires up per-shard native-memory circuit-breaker
# accounting. Setting it on a `SPARSE` index would buy that for a field that does not exist, and
# would misdescribe the index to anyone reading `_settings`. Verified against OpenSearch 2.19.1
# that a `rank_features`-only index is created and queried successfully with no `settings` block
# at all.
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

# Ensures the target index exists, creating it if `Configuration.createIndexIfNotExists` allows it.
# When it is `false`, this function performs no network I/O at all — this is what lets `init` run
# fully offline for least-privilege deployments and for testing.
#
# Note that "no network I/O" also means no existence check, and a missing index is not
# self-announcing: OpenSearch's `action.auto_create_index` default silently creates one from the
# first document's inferred shape, without `index.knn` and with the vector mapped as a plain
# `float` array. See the hazard note on `Configuration.createIndexIfNotExists`.
#
# A `resource_already_exists_exception` on creation is treated as success, covering two instances
# racing to create the same index.
#
# # Creation is not searchability on Serverless Classic
# Returning successfully means the index exists, not that it can be searched yet. On
# `SERVERLESS_CLASSIC` a newly created index answers `GET _mapping` and `GET _settings`
# immediately while `_count`/`_search` against it still fail with `index_not_found_exception`,
# for roughly ten seconds. This is a propagation delay rather than an unassigned shard: the index
# becomes searchable on its own, with no write and no intervention. Writes are unaffected and
# succeed immediately, so a caller that adds before it queries never observes it.
#
# The delay is deliberately not waited out here. Blocking construction until the index answers a
# search would add up to those ten seconds to every `SERVERLESS_CLASSIC` construction, including
# for the callers who write first and would never have hit it. A caller that must query
# immediately after constructing the store should instead retry an `index_not_found_exception`
# for a few seconds before treating it as fatal. `SERVERLESS_NEXTGEN` is searchable at once and
# does not need this.
#
# + transport - The transport to issue `HEAD`/`PUT` requests through
# + indexName - The index to ensure
# + searchMode - The kind of search, passed through to the mapping builder
# + config - The vector store configuration
# + deployment - The deployment, passed through to the mapping builder
# + return - `true` if this call created the index, `false` if it already existed or creation was
# switched off, or an `ai:Error` on failure. The distinction is what lets `init` skip
# `verifyIndexMapping` against a mapping this module just wrote, which is correct by construction
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
        // Another instance won the race and created it. Its mapping was built from its own
        // configuration, which is not necessarily this store's -- so this is reported as "already
        // existed", leaving `verifyIndexMapping` to confirm the winner's mapping suits this store.
        return false;
    }
    return true;
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
