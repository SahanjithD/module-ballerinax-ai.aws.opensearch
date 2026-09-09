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

// The construction-time checks that connect a store's configuration to the cluster and index it is
// actually pointed at, governed by `Configuration.verifyOnInit`.
//
// Everything else in this module validates a request before it is sent or a response after it
// arrives. These two functions are the only ones that ask the server what it *is*, and they exist
// because the alternative is silence: a `SPARSE` store aimed at a dense index writes successfully
// forever and fails every query with a bare 400, and a cluster three minor versions too old to
// serve a `hybrid` query says so only as a parse failure buried in a 400 body.
//
// Both follow the same failure policy, and the distinction is the point: a check that *runs* and
// disagrees is fatal, because the store is provably misconfigured. A check that cannot run — a
// least-privilege principal without `cluster:monitor/main` or `indices:admin/mappings/get` — is a
// warning, because refusing to construct a store that would have worked is the worse outcome.

# A capability this store's configuration will exercise, paired with the first OpenSearch version
# that serves it and the setting that asked for it.
type VersionRequirement record {|
    # The version floor, as `[major, minor]`.
    int[] floor;
    # What needs it, named as the caller would recognise it.
    string feature;
    # What the caller can change to drop the requirement.
    string remedy;
|};

# The version floors this store's `SearchMode` implies.
#
# Ordered highest floor first, so the error a caller sees names the requirement that is furthest
# out of reach rather than whichever happened to be checked first — upgrading to satisfy a lower
# floor would only surface the next one.
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
    // Listed last and sorted to the front only if nothing outranks it: the `hybrid` query is the
    // lowest floor `HYBRID` carries, so naming it while a 2.19 processor is also configured would
    // send a caller to an upgrade that still does not work.
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

# Checks the cluster's OpenSearch version against every floor this store's configuration implies.
#
# Only meaningful on a managed domain. Both Serverless generations reject `GET /` outright — the
# AOSS proxy does not expose the root endpoint — so there is nothing to read and nothing to
# conclude, and the check is skipped rather than attempted and warned about on every construction.
# That absence is itself informative: AWS documents which of its Serverless regions serve neural
# search at all, and a region that does not will fail the search rather than the version check.
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

# Checks that the target index actually declares the vector field(s) this store reads and writes.
#
# This is the check with no substitute anywhere else in the module. Every other setting that can be
# wrong against an existing index is wrong loudly — a bad filter path returns no hits, a bad
# credential returns 403 — but a vector field that is absent, is mapped as something else, or holds
# a different number of dimensions produces writes that succeed and queries that fail with a bare
# `400`, and nothing in either says which end is wrong.
#
# Only the vector fields are checked. The content, id and metadata fields are deliberately left
# alone: pointing this module at a pre-existing index with a different document shape is a
# supported use (`Configuration.metadataFieldName = ""` exists for exactly that), and a mapping
# this module did not write is entitled to look different everywhere the vectors are not.
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

# Checks one vector field against the index's mapping: that it is declared, that its `type` is the
# one this store's mode requires, and — for a `knn_vector` — that its `dimension` matches.
#
# The three failures are reported separately because they have three different causes. A missing
# field usually means a renamed `vectorFieldName` or an index built for a different mode; a wrong
# type means the store's `SearchMode` and the index disagree about what kind of search this is; a
# wrong dimension means the embedding model changed under a store that was not rebuilt.
#
# + properties - The `properties` object of the index's mapping
# + indexName - The index, for the error message
# + fieldName - The field this store expects
# + expectedType - `knn_vector` or `rank_features`
# + expectedDimension - The configured dimension for a `knn_vector`, or `()` for a
# `rank_features` field, which has none
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

# Parses an OpenSearch version string into its numeric components.
#
# Tolerant by design: it reads the leading dot-separated integers and stops at anything else, so a
# distribution suffix (`3.0.0-SNAPSHOT`, `2.11.0-rc1`) parses to the release it precedes rather
# than failing. A string with no leading digit at all yields `()`, and the caller warns rather than
# guessing.
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

# Whether a parsed version meets a floor, comparing only as many components as the floor names.
#
# A version shorter than the floor is padded with zeros rather than rejected: a cluster reporting
# `3` satisfies a `2.11` floor, since a missing minor component means zero and 3.0 is past 2.11.
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
