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

// Fixtures for the `docker` group, which runs against the containers in
// `tests/resources/compose.yaml`. `./gradlew test` starts them; to run the group directly:
//
//   docker compose -f tests/resources/compose.yaml up -d
//   bal test --groups docker
//
// Skip it with `bal test --disable-groups docker`. Only `MANAGED_DOMAIN` can run in a container;
// Serverless behavior is covered by `live_test.bal`.

import ballerina/ai;
import ballerina/http;
import ballerina/test;
import ballerina/uuid;
import ballerinax/aws.auth;

# The plain-HTTP container, with the security plugin disabled.
configurable string containerUrl = "http://localhost:9200";

# The HTTPS container, with the security plugin and its internal user database enabled.
configurable string secureContainerUrl = "https://localhost:9201";

# The internal-user-database username on the secure container.
configurable string secureContainerUsername = "admin";

# The internal-user-database password on the secure container, matching
# `OPENSEARCH_INITIAL_ADMIN_PASSWORD` in `compose.yaml`.
configurable string secureContainerPassword = "Ballerina@OpenSearch1";

# Small enough for readable request bodies, large enough for orthogonal test vectors.
const int CONTAINER_DIMENSION = 4;

# Region for the SigV4 scope. The container ignores the signature, but signing must still succeed.
const string CONTAINER_REGION = "us-east-1";

# The prefix every index created by this suite carries, so `@test:AfterSuite` can recognise them.
const string CONTAINER_INDEX_PREFIX = "it-";

// `DEFAULT_CREDENTIALS` would walk the AWS provider chain and fail on a CI runner.
# Dummy credentials, so every request takes the SigV4 path. The container ignores them.
final auth:StaticAuthConfig & readonly containerAuth = {
    accessKeyId: "AKIAIOSFODNN7EXAMPLE",
    secretAccessKey: "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
};

# Every index this suite has created, drained by `@test:AfterSuite` in
# `container_cleanup_test.bal`.
isolated string[] createdIndexNames = [];

# Reserves a unique index name for one test and registers it for teardown.
#
# + prefix - A short, test-specific tag, to make a leaked index traceable to its test
# + return - The reserved index name
isolated function containerIndexName(string prefix) returns string {
    string name = string `${CONTAINER_INDEX_PREFIX}${prefix}-${uuid:createRandomUuid()}`;
    lock {
        createdIndexNames.push(name);
    }
    return name;
}

# Returns a snapshot of the index names reserved so far.
#
# + return - The reserved index names
isolated function trackedIndexNames() returns string[] {
    lock {
        return createdIndexNames.clone();
    }
}

# The default store configuration.
#
# + return - The default container test configuration
isolated function containerConfig() returns Configuration => {};

# The dense search mode most container tests use: an index sized to `CONTAINER_DIMENSION`.
#
# + similarityMetric - The similarity metric
# + return - The default container test search mode
isolated function containerDenseMode(ai:SimilarityMetric similarityMetric = ai:COSINE) returns DenseSearch => {
    queryMode: ai:DENSE,
    indexConfig: {dimension: CONTAINER_DIMENSION, similarityMetric}
};

# A managed domain with `refreshOnWrite`, so a write is visible to the next query without polling.
#
# + engine - The ANN engine to configure, which only a managed domain lets a caller choose
# + return - The default container test deployment
isolated function containerDeployment(Engine engine = FAISS) returns ManagedDomainDeployment => {
    deploymentType: MANAGED_DOMAIN,
    auth: containerAuth,
    engine,
    refreshOnWrite: true
};

# Constructs a store against the plain-HTTP container.
#
# + indexName - The target index
# + searchMode - The search mode, defaulting to dense over `CONTAINER_DIMENSION`
# + config - The store configuration
# + deployment - The deployment, defaulting to a managed domain with `refreshOnWrite` set
# + return - The store, or an `ai:Error` if construction fails
isolated function newContainerStore(string indexName, SearchMode searchMode = {
            queryMode: ai:DENSE,
            indexConfig: {dimension: CONTAINER_DIMENSION}
        }, Configuration config = {}, ManagedDomainDeployment deployment = {
            deploymentType: MANAGED_DOMAIN,
            auth: containerAuth,
            refreshOnWrite: true
        }) returns VectorStore|ai:Error =>
    new (containerUrl, CONTAINER_REGION, indexName, deployment, searchMode, config);

# Builds a dense test vector of `CONTAINER_DIMENSION` components.
#
# + a - The first component
# + b - The second component
# + c - The third component
# + d - The fourth component
# + return - The vector
isolated function vec(float a, float b = 0.0, float c = 0.0, float d = 0.0) returns ai:Vector => [a, b, c, d];

# The reference query vector: `vec(1.0)` scores 1.0 against it under cosine, `vec(0.0, 1.0)` 0.0.
#
# + return - The reference vector
isolated function queryVector() returns ai:Vector => vec(1.0);

# Builds a text entry.
#
# + id - The logical id, or `()` to let the module generate one
# + embedding - The dense embedding
# + content - The chunk content
# + metadata - The chunk metadata, if any
# + return - The entry
isolated function entry(string? id, ai:Vector embedding, string content = "hello",
        ai:Metadata? metadata = ()) returns ai:VectorEntry =>
    {id, embedding, chunk: <ai:TextChunk>{content, metadata}};

// --- raw cluster access, bypassing the module so an assertion cannot share its bugs ------------

# An HTTP client straight to a container.
#
# + url - The container URL
# + return - The client, or an error if it cannot be created
isolated function rawClient(string url = containerUrl) returns http:Client|error {
    if url == secureContainerUrl {
        return new (url,
            secureSocket = {enable: false},
            auth = {username: secureContainerUsername, password: secureContainerPassword}
        );
    }
    return new (url, secureSocket = {enable: false});
}

# Reads back the `mappings` object of an index.
#
# + indexName - The index to inspect
# + return - The `mappings` object, or an error
isolated function indexMapping(string indexName) returns map<json>|error {
    http:Client cl = check rawClient();
    json payload = check cl->get(string `/${indexName}/_mapping`);
    map<json> byIndex = check payload.ensureType();
    map<json> forIndex = check mapField(byIndex, indexName);
    return mapField(forIndex, "mappings");
}

# Reads back the `settings.index` object of an index.
#
# + indexName - The index to inspect
# + return - The `settings.index` object, or an error
isolated function indexSettings(string indexName) returns map<json>|error {
    http:Client cl = check rawClient();
    json payload = check cl->get(string `/${indexName}/_settings`);
    map<json> byIndex = check payload.ensureType();
    map<json> forIndex = check mapField(byIndex, indexName);
    map<json> settings = check mapField(forIndex, "settings");
    return mapField(settings, "index");
}

# Counts the documents in an index, independent of any query this module builds.
#
# + indexName - The index to count
# + return - The document count, or an error
isolated function documentCount(string indexName) returns int|error {
    http:Client cl = check rawClient();
    json payload = check cl->get(string `/${indexName}/_count`);
    map<json> body = check payload.ensureType();
    return body["count"].ensureType();
}

# Checks whether an index exists, bypassing `OpenSearchTransport.indexExists`.
#
# + indexName - The index to check
# + return - `true` if it exists, or an error
isolated function rawIndexExists(string indexName) returns boolean|error {
    http:Client cl = check rawClient();
    http:Response resp = check cl->head(string `/${indexName}`);
    return resp.statusCode == 200;
}

# Deletes an index on the plain-HTTP container, tolerating one that is already gone.
#
# + indexName - The index to delete
# + return - An error only if the request itself fails
isolated function rawDeleteIndex(string indexName) returns error? => rawDeleteIndexOn(containerUrl, indexName);

# Deletes an index on a given container, tolerating one that is already gone.
#
# + url - The container URL
# + indexName - The index to delete
# + return - An error only if the request itself fails
isolated function rawDeleteIndexOn(string url, string indexName) returns error? {
    http:Client cl = check rawClient(url);
    http:Response resp = check cl->delete(string `/${indexName}`);
    // A 404 means the test never created it; anything else would leak indices silently.
    if resp.statusCode != 200 && resp.statusCode != 404 {
        return error(string `Deleting '${indexName}' returned HTTP ${resp.statusCode}`);
    }
}

# Navigates one level into a JSON object.
#
# + parent - The enclosing object
# + key - The key to descend into
# + return - The nested object, or an error if the key is absent or not an object
isolated function mapField(map<json> parent, string key) returns map<json>|error => parent[key].ensureType();

# Asserts that exactly the expected ids came back, in any order.
#
# + matches - The query results
# + expectedIds - The ids that should be present
# + message - The assertion message
isolated function assertIdsEqual(ai:VectorMatch[] matches, string[] expectedIds, string message) {
    string[] actual = from ai:VectorMatch m in matches
        let string id = m.id ?: ""
        order by id
        select id;
    string[] expected = from string id in expectedIds
        order by id
        select id;
    test:assertEquals(actual, expected, message);
}
