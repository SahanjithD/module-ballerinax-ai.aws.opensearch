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
import ballerina/log;

// Construction-time checks for `Configuration.verifyOnInit`. A check that disagrees fails `init`;
// one that cannot run (e.g. no permission) only warns, since refusing a working store is worse.

# A feature the store's configuration uses, with the first OpenSearch version that supports it.
type VersionRequirement record {|
    # The version floor, as `[major, minor]`.
    int[] floor;
    # The feature, as the caller would recognise it.
    string feature;
    # How the caller can avoid needing it.
    string remedy;
|};

// Highest first, so the error names the floor furthest out of reach rather than one that would
// only reveal the next after an upgrade.
# The OpenSearch version floors a search mode implies, highest first.
#
# + searchMode - The store's search mode
# + return - The requirements, highest floor first
isolated function versionRequirements(SearchMode searchMode) returns VersionRequirement[] {
    VersionRequirement[] requirements = [];
    if searchMode is SparseSearch {
        requirements.push({
            floor: [2, 14],
            feature: "a 'neural_sparse' query carrying raw 'query_tokens'",
            remedy: "use 'DENSE', which needs no plugin and no version floor"
        });
        if searchMode.twoPhaseAcceleration is TwoPhaseAcceleration {
            requirements.push({
                floor: [2, 15],
                feature: "the 'neural_sparse_two_phase_processor'",
                remedy: "leave 'SparseSearch.twoPhaseAcceleration' unset"
            });
        }
    }
    if searchMode is HybridSearch {
        requirements.push({
            floor: [2, 14],
            feature: "a 'neural_sparse' sub-query carrying raw 'query_tokens'",
            remedy: "use 'DENSE', which needs no plugin and no version floor"
        });
        if searchMode.fusion is RrfFusion {
            requirements.push({
                floor: [2, 19],
                feature: "the 'score-ranker-processor' that performs reciprocal rank fusion",
                remedy: "set a 'HybridSearchConfig' instead, whose 'normalization-processor' is 2.10+"
            });
        }
        if searchMode.efSearch is int {
            requirements.push({
                floor: [2, 16],
                feature: "'method_parameters' in a 'knn' query",
                remedy: "leave 'HybridSearch.efSearch' unset"
            });
        }
    }
    if searchMode is DenseSearch && searchMode.efSearch is int {
        requirements.push({
            floor: [2, 16],
            feature: "'method_parameters' in a 'knn' query",
            remedy: "leave 'DenseSearch.efSearch' unset"
        });
    }
    if searchMode is HybridSearch {
        requirements.push({
            floor: [2, 11],
            feature: "the 'hybrid' query",
            remedy: "use 'DENSE' or 'SPARSE', which need no compound query"
        });
    }
    return requirements.sort("descending", isolated function(VersionRequirement r) returns int =>
            r.floor[0] * 1000 + r.floor[1]);
}

// Serverless rejects `GET /`, so the check is skipped there rather than warning every time.
# Checks the cluster's OpenSearch version against the floors the search mode implies.
#
# + transport - The transport to read `GET /` through
# + searchMode - The store's search mode, supplying the floors
# + deployment - The deployment, which decides whether the version is readable at all
# + return - An `ai:Error` if the cluster is definitively too old, otherwise `()`
isolated function verifyClusterVersion(OpenSearchTransport transport, SearchMode searchMode,
        Deployment deployment) returns ai:Error? {
    VersionRequirement[] requirements = versionRequirements(searchMode);
    if requirements.length() == 0 || deployment !is ManagedDomainDeployment {
        return;
    }
    string?|ai:Error reported = transport.clusterVersion();
    if reported is ai:Error {
        log:printWarn("Could not read the OpenSearch version from 'GET /', so this store's search " +
                "mode was not checked against the cluster's capabilities. Grant the calling " +
                "principal read access to '/', or set 'Configuration.verifyOnInit' to false to " +
                "stop attempting this", 'error = reported);
        return;
    }
    if reported is () {
        return;
    }
    int[]? version = parseVersion(reported);
    if version is () {
        log:printWarn(string `Could not parse the OpenSearch version '${reported}' reported by ` +
                "'GET /', so this store's search mode was not checked against the cluster's " +
                "capabilities");
        return;
    }
    foreach VersionRequirement requirement in requirements {
        if isAtLeast(version, requirement.floor) {
            continue;
        }
        return error(string `This cluster reports OpenSearch ${reported}, which does not support ` +
                string `${requirement.feature}. That needs ` +
                string `${requirement.floor[0]}.${requirement.floor[1]} or later. Upgrade the ` +
                string `domain, ${requirement.remedy}, or set 'Configuration.verifyOnInit' to ` +
                "false to send the query anyway and let the server reject it");
    }
}

// Only the vector fields are checked; an index created elsewhere may differ in the rest.
# Checks that the index declares this store's vector fields with the expected type and dimension.
#
# + transport - The transport to read `GET /<index>/_mapping` through
# + indexName - The index to check
# + searchMode - The store's search mode, supplying the expected field names and shapes
# + return - An `ai:Error` if the index is missing or its vector fields disagree, otherwise `()`
isolated function verifyIndexMapping(OpenSearchTransport transport, string indexName,
        SearchMode searchMode) returns ai:Error? {
    map<json>?|ai:Error properties = transport.indexMappingProperties(indexName);
    if properties is ai:Error {
        if properties.detail()["status"] == 404 {
            return error(string `The index '${indexName}' does not exist, and ` +
                    "'Configuration.createIndexIfNotExists' is false so this module did not " +
                    "create it. Writing to it anyway would not fail: OpenSearch's " +
                    "'action.auto_create_index' default would build an index from the first " +
                    "document's inferred shape, with no 'index.knn' setting and the vector mapped " +
                    "as a plain float array, after which every query fails. Create the index out " +
                    "of band with a compatible mapping, or set 'createIndexIfNotExists' to true",
                    properties);
        }
        log:printWarn(string `Could not read the mapping of index '${indexName}', so this store's ` +
                "search mode was not checked against it. Grant the calling principal " +
                "'indices:admin/mappings/get' on the index, or set 'Configuration.verifyOnInit' " +
                "to false to stop attempting this", 'error = properties);
        return;
    }
    if properties is () {
        return error(string `The index '${indexName}' declares no field mappings at all, so it ` +
                "cannot serve this store. An index in this state is usually one that " +
                "'action.auto_create_index' created from a write to a missing index. Delete it " +
                "and let this module create it, or map it out of band to match this store's " +
                "'SearchMode'");
    }

    IndexConfig? indexConfig = indexConfigOf(searchMode);
    string? denseFieldName = denseFieldNameOf(searchMode);
    if indexConfig is IndexConfig && denseFieldName is string {
        check verifyMappedField(properties, indexName, denseFieldName, "knn_vector",
                indexConfig.dimension);
    }
    string? sparseFieldName = sparseFieldNameOf(searchMode);
    if sparseFieldName is string {
        check verifyMappedField(properties, indexName, sparseFieldName, "rank_features", ());
    }
}

# Checks one vector field's presence, type and, for a `knn_vector`, dimension.
#
# + properties - The `properties` object of the index's mapping
# + indexName - The index, for the error message
# + fieldName - The field this store expects
# + expectedType - `knn_vector` or `rank_features`
# + expectedDimension - The configured dimension, or `()` for a `rank_features` field
# + return - An `ai:Error` naming the specific disagreement, otherwise `()`
isolated function verifyMappedField(map<json> properties, string indexName, string fieldName,
        string expectedType, int? expectedDimension) returns ai:Error? {
    json declared = properties[fieldName];
    if declared !is map<json> {
        return error(string `The index '${indexName}' declares no field '${fieldName}', which ` +
                string `this store's search mode reads and writes as its '${expectedType}' field. ` +
                "Either the store's field name does not match the index, or the index was built " +
                "for a different search mode. Point the store at the right index, correct the " +
                "field name, or reindex into an index this module creates");
    }
    json declaredType = declared["type"];
    if declaredType !is string || declaredType != expectedType {
        return error(string `The index '${indexName}' maps field '${fieldName}' as ` +
                string `'${declaredType is string ? declaredType : declaredType.toJsonString()}', ` +
                string `but this store's search mode requires '${expectedType}'. A mapping cannot ` +
                "be changed in place; reindex into an index with the right shape. If the index " +
                "maps the vector as a plain float array, it was almost certainly created by " +
                "'action.auto_create_index' from a write to a missing index rather than by this " +
                "module");
    }
    if expectedDimension is () {
        return;
    }
    json declaredDimension = declared["dimension"];
    if declaredDimension is int && declaredDimension != expectedDimension {
        return error(string `The index '${indexName}' maps field '${fieldName}' with dimension ` +
                string `${declaredDimension}, but this store is configured for ` +
                string `${expectedDimension}. 'IndexConfig.dimension' is honored at index creation ` +
                "only, so this store would keep writing vectors the index rejects — or, if the " +
                "embeddings themselves still match the index, would misreport its own shape. Set " +
                string `'IndexConfig.dimension' to ${declaredDimension}, or reindex into an index ` +
                string `built for ${expectedDimension}`);
    }
}

# Parses the leading numbers of a version, so `3.0.0-SNAPSHOT` reads as `[3, 0, 0]`.
#
# + version - The version string, e.g. `2.19.1`
# + return - The components, e.g. `[2, 19, 1]`, or `()` if none could be read
isolated function parseVersion(string version) returns int[]? {
    int[] components = [];
    string current = "";
    foreach string:Char c in version {
        if c >= "0" && c <= "9" {
            current += c;
            continue;
        }
        if c != "." || current == "" {
            break;
        }
        int|error component = int:fromString(current);
        if component is error {
            return ();
        }
        components.push(component);
        current = "";
    }
    if current != "" {
        int|error component = int:fromString(current);
        if component is error {
            return ();
        }
        components.push(component);
    }
    return components.length() == 0 ? () : components;
}

# Whether a version meets a floor. Missing components count as zero.
#
# + version - The parsed cluster version
# + floor - The required version, as `[major, minor]`
# + return - `true` if `version` is greater than or equal to `floor`
isolated function isAtLeast(int[] version, int[] floor) returns boolean {
    foreach int i in 0 ..< floor.length() {
        int actual = i < version.length() ? version[i] : 0;
        if actual != floor[i] {
            return actual > floor[i];
        }
    }
    return true;
}
