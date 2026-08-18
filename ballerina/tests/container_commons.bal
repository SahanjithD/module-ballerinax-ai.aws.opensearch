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

// Shared fixtures for the `docker` test group -- the integration suite that runs against the two
// throwaway OpenSearch containers in `tests/resources/compose.yaml`.
//
// These cover the layer the unit tests structurally cannot reach: `OpenSearchTransport`'s signing,
// sending, response parsing and error mapping, `ensureIndex`'s create-or-skip logic, and whether
// the request bodies the pure functions build are actually *accepted* by an OpenSearch server. A
// unit test can assert that `convertFilters` emits a `term` clause; only a real server can prove
// that clause matches the documents it is supposed to match.
//
// `./gradlew test` starts and stops the containers automatically. To run them directly:
//
//   docker compose -f tests/resources/compose.yaml up -d
//   bal test --groups docker
//
// To skip them when Docker is unavailable:
//
//   bal test --disable-groups docker
//
// Deployment type is always `MANAGED_DOMAIN` here: a container is not AWS, so the append-only
// `add` and search-then-bulk-delete semantics of `SERVERLESS_CLASSIC` cannot be reproduced
// locally. Those stay in `live_test.bal`.

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

# Small enough to keep request bodies readable in a failure message, large enough for the
# orthogonal-vs-identical vectors the scoring tests rely on.
const int CONTAINER_DIMENSION = 4;

# The region name folded into the SigV4 credential scope. The container never verifies the
# signature, but signing still has to succeed for the request to be sent at all.
const string CONTAINER_REGION = "us-east-1";

# The prefix every index created by this suite carries, so `@test:AfterSuite` can recognise them.
const string CONTAINER_INDEX_PREFIX = "it-";

# Dummy static credentials. The point is not that they are valid -- the container ignores the
# `Authorization` header entirely -- but that `sendSigned` takes its SigV4 path and produces a
# well-formed signature for every request the suite makes. `DEFAULT_CREDENTIALS` would instead
# walk the AWS provider chain and fail outright on a CI runner with no AWS configuration.
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

# The configuration most container tests use: an index sized to `CONTAINER_DIMENSION`, with
# `refreshOnWrite` set so a write is visible to the very next query. That flag is what makes this
# suite deterministic where `live_test.bal` has to poll -- a managed domain (and a container)
# honors `refresh=wait_for`, while Serverless does not.
#
# + return - The default container test configuration
isolated function containerConfig() returns Configuration => {
    indexConfig: {dimension: CONTAINER_DIMENSION},
    refreshOnWrite: true
};

# Constructs a store against the plain-HTTP container.
#
# + indexName - The target index
# + config - The store configuration
# + return - The store, or an `ai:Error` if construction fails
isolated function newContainerStore(string indexName, Configuration config = {
            indexConfig: {dimension: CONTAINER_DIMENSION},
            refreshOnWrite: true
        }) returns VectorStore|ai:Error =>
    new (containerUrl, CONTAINER_REGION, indexName, MANAGED_DOMAIN, containerAuth, config);

# Builds a dense test vector of `CONTAINER_DIMENSION` components.
#
# + a - The first component
# + b - The second component
# + c - The third component
# + d - The fourth component
# + return - The vector
isolated function vec(float a, float b = 0.0, float c = 0.0, float d = 0.0) returns ai:Vector => [a, b, c, d];

# The reference query vector. `vec(1.0)` scores 1.0 against it under cosine and `vec(0.0, 1.0)`
# scores 0.0, which is what the scoring assertions key off.
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

// --- raw cluster access, for asserting server-side state the module itself never reads ---------

# An HTTP client pointed straight at a container, bypassing this module entirely. Used to inspect
# what the server actually stored -- a mapping the module only ever writes, a document count the
# module has no API for -- so an assertion cannot be satisfied by the same bug it is meant to catch.
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
    // 404 just means the test never created it. Anything else -- most plausibly a 401 from the
    // secure container -- would otherwise leak indices silently, one per run.
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
