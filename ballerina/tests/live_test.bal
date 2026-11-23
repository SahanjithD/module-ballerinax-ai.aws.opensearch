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

// Live tests against real AWS resources, disabled unless `isLiveServer` is set. Set it and the
// service URL for each deployment you have (in Config.toml or with `-C`), then run:
//
//   bal test --groups live
//
// or `./gradlew clean test -Pgroups=live`. A test whose URL is unset returns without checking
// anything. Serverless is polled, since its refresh interval cannot be forced. Every index a test
// creates is deleted after the suite, which on Serverless needs `aoss:DeleteIndex` as well.

import ballerina/ai;
import ballerina/http;
import ballerina/lang.runtime;
import ballerina/log;
import ballerina/test;
import ballerina/uuid;
import ballerinax/aws.auth;

configurable boolean isLiveServer = false;

configurable string managedDomainUrl = "";
configurable string managedDomainRegion = "us-east-1";
// Optional: a fine-grained access control internal user, for the basic auth test.
configurable string managedDomainUsername = "";
configurable string managedDomainPassword = "";

configurable string serverlessClassicUrl = "";
configurable string serverlessClassicRegion = "us-east-1";

configurable string serverlessNextGenUrl = "";
configurable string serverlessNextGenRegion = "us-east-1";
// Optional: a per-account NextGen endpoint and the collection to target through it.
configurable string serverlessNextGenAccountUrl = "";
configurable string serverlessNextGenCollectionName = "";

const int LIVE_TEST_DIMENSION = 8;
const decimal POLL_INTERVAL_SECONDS = 3;
// Classic documents its refresh as ~60 s and NextGen as ~10 s; these leave headroom over both.
const decimal CLASSIC_VISIBILITY_TIMEOUT = 90;
const decimal NEXTGEN_VISIBILITY_TIMEOUT = 30;

# Where a live index lives, so the suite can delete it afterwards.
type LiveTarget record {|
    # The service URL.
    string url;
    # The AWS region.
    string region;
    # The SigV4 signing name, `es` or `aoss`.
    string signingName;
    # Extra headers to sign, e.g. the NextGen collection header.
    map<string> headers = {};
|};

# An index a live test created.
type LiveIndex record {|
    # Where it lives.
    LiveTarget target;
    # Its name.
    string indexName;
|};

isolated LiveIndex[] liveIndexes = [];

isolated function managedTarget() returns LiveTarget =>
    {url: managedDomainUrl, region: managedDomainRegion, signingName: SIGNING_NAME_MANAGED};

isolated function classicTarget() returns LiveTarget =>
    {url: serverlessClassicUrl, region: serverlessClassicRegion, signingName: SIGNING_NAME_SERVERLESS};

isolated function nextGenTarget() returns LiveTarget =>
    {url: serverlessNextGenUrl, region: serverlessNextGenRegion, signingName: SIGNING_NAME_SERVERLESS};

isolated function nextGenAccountTarget() returns LiveTarget => {
    url: serverlessNextGenAccountUrl,
    region: serverlessNextGenRegion,
    signingName: SIGNING_NAME_SERVERLESS,
    headers: {"x-amz-aoss-collection-name": serverlessNextGenCollectionName}
};

# Reserves a unique index name and registers it for deletion after the suite.
#
# + target - Where the index will be created
# + prefix - A short, test-specific tag
# + return - The index name
isolated function liveIndexName(LiveTarget target, string prefix) returns string {
    string indexName = string `${prefix}-${uuid:createRandomUuid()}`;
    lock {
        liveIndexes.push({target: target.clone(), indexName});
    }
    return indexName;
}

isolated function liveTestVector() returns ai:Vector => [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

isolated function liveEntry(string id, string content = "hello", ai:Metadata metadata = {}) returns ai:VectorEntry =>
    {id, embedding: liveTestVector(), chunk: <ai:TextChunk>{content, metadata}};

isolated function liveDenseMode() returns DenseSearch =>
    {queryMode: ai:DENSE, indexConfig: {dimension: LIVE_TEST_DIMENSION}};

# Sends a SigV4-signed request with the default credential chain, bypassing the store.
#
# + target - Where to send it
# + method - The HTTP method
# + path - The request path
# + return - The response, or an error
isolated function liveSignedRequest(LiveTarget target, string method, string path) returns http:Response|error {
    auth:CredentialProvider provider = check new (auth:DEFAULT_CREDENTIALS);
    auth:Credentials credentials = check provider.getCredentials();
    map<string> signed = check auth:getSignedHeaders({
                                                         method,
                                                         host: check extractHost(target.url),
                                                         path,
                                                         queryParams: {},
                                                         headers: target.headers,
                                                         payload: []
                                                     }, credentials, target.region, target.signingName);
    check provider.close();
    http:Client httpClient = check new (target.url, {httpVersion: http:HTTP_1_1, http1Settings: {chunking: http:CHUNKING_NEVER}});
    http:Request request = new;
    foreach [string, string] [name, value] in signed.entries() {
        request.setHeader(name, value);
    }
    return httpClient->execute(method, path, request);
}

# Reads an index's mapped `properties`, bypassing the store.
#
# + target - Where the index lives
# + indexName - The index
# + return - The `properties` object, or an error
isolated function liveMappingProperties(LiveTarget target, string indexName) returns map<json>|error {
    http:Response response = check liveSignedRequest(target, "GET", string `/${indexName}/_mapping`);
    json payload = check response.getJsonPayload();
    map<json> byIndex = check payload.ensureType();
    foreach json entry in byIndex {
        return check entry.mappings.properties.ensureType();
    }
    return error(string `no mapping returned for '${indexName}'`);
}

// Failures are only logged, so a teardown error cannot mask a test result.
# Deletes every index the live tests created.
@test:AfterSuite {alwaysRun: true}
function cleanUpLiveIndexes() {
    LiveIndex[] indexes;
    lock {
        indexes = liveIndexes.clone();
    }
    foreach LiveIndex index in indexes {
        http:Response|error response = liveSignedRequest(index.target, "DELETE", "/" + index.indexName);
        if response is error || (response.statusCode != 200 && response.statusCode != 404) {
            log:printWarn("Failed to delete a live test index", indexName = index.indexName,
                    status = response is http:Response ? response.statusCode : -1);
        }
    }
}

# Polls `query` until a match with the given id appears, or `timeoutSeconds` elapses.
#
# + store - The vector store to poll
# + id - The logical id to wait for
# + timeoutSeconds - The maximum time to wait
# + return - `true` if found within the timeout
isolated function pollUntilVisible(ai:VectorStore store, string id, decimal timeoutSeconds) returns boolean|ai:Error {
    decimal waited = 0;
    while waited < timeoutSeconds {
        if check countMatches(store, id) > 0 {
            return true;
        }
        runtime:sleep(POLL_INTERVAL_SECONDS);
        waited += POLL_INTERVAL_SECONDS;
    }
    return false;
}

# Polls `query` until no match with the given id remains, or `timeoutSeconds` elapses.
#
# + store - The vector store to poll
# + id - The logical id to wait to disappear
# + timeoutSeconds - The maximum time to wait
# + return - `true` if the id is gone within the timeout
isolated function pollUntilGone(ai:VectorStore store, string id, decimal timeoutSeconds) returns boolean|ai:Error {
    decimal waited = 0;
    while waited < timeoutSeconds {
        if check countMatches(store, id) == 0 {
            return true;
        }
        runtime:sleep(POLL_INTERVAL_SECONDS);
        waited += POLL_INTERVAL_SECONDS;
    }
    return false;
}

# Counts the matches carrying the given id among the top 10.
#
# + store - The vector store to query
# + id - The logical id to count
# + return - The count
isolated function countMatches(ai:VectorStore store, string id) returns int|ai:Error {
    ai:VectorMatch[] matches = check store.query({embedding: liveTestVector(), topK: 10});
    int occurrences = 0;
    foreach ai:VectorMatch m in matches {
        if m.id == id {
            occurrences += 1;
        }
    }
    return occurrences;
}

// --- managed domain ----------------------------------------------------------------------------

isolated function newManagedStore(string indexName, SearchMode searchMode = liveDenseMode(),
        ManagedDomainStoreConfiguration storeConfig = {refreshOnWrite: true}, boolean createIndexIfNotExists = true)
        returns ManagedDomainVectorStore|ai:Error =>
    new (managedDomainUrl, managedDomainRegion, indexName, auth:DEFAULT_CREDENTIALS, createIndexIfNotExists,
        searchMode, storeConfig);

@test:Config {groups: ["live"], enable: isLiveServer}
isolated function testManagedDomainAddQueryDeleteRoundTrip() returns error? {
    if managedDomainUrl == "" {
        return;
    }
    ManagedDomainVectorStore store = check newManagedStore(liveIndexName(managedTarget(), "md-roundtrip"));

    string id = uuid:createRandomUuid();
    check store.add([liveEntry(id)]);

    ai:VectorMatch[] matches = check store.query({embedding: liveTestVector(), topK: 1});
    test:assertEquals(matches.length(), 1);
    test:assertEquals(matches[0].id, id);

    check store.delete(id);
    test:assertTrue(check pollUntilGone(store, id, 15), "the entry should be gone straight after delete");
}

@test:Config {groups: ["live"], enable: isLiveServer}
isolated function testManagedDomainSameIdTwiceUpserts() returns error? {
    if managedDomainUrl == "" {
        return;
    }
    ManagedDomainVectorStore store = check newManagedStore(liveIndexName(managedTarget(), "md-upsert"));

    string id = uuid:createRandomUuid();
    check store.add([liveEntry(id, "v1")]);
    check store.add([liveEntry(id, "v2")]);

    test:assertEquals(check countMatches(store, id), 1, "re-adding the same id must upsert, not duplicate");
}

@test:Config {groups: ["live"], enable: isLiveServer}
isolated function testManagedDomainFilteredQuery() returns error? {
    if managedDomainUrl == "" {
        return;
    }
    ManagedDomainVectorStore store = check newManagedStore(liveIndexName(managedTarget(), "md-filter"));

    string matchingId = uuid:createRandomUuid();
    string otherId = uuid:createRandomUuid();
    check store.add([
        liveEntry(matchingId, "en doc", {"language": "en"}),
        liveEntry(otherId, "fr doc", {"language": "fr"})
    ]);

    ai:VectorMatch[] matches = check store.query({
        embedding: liveTestVector(),
        topK: 10,
        filters: {filters: [{key: "language", operator: ai:EQUAL, value: "en"}]}
    });
    test:assertEquals(matches.length(), 1);
    test:assertEquals(matches[0].id, matchingId);
}

@test:Config {groups: ["live"], enable: isLiveServer}
isolated function testManagedDomainResultsCarryVectorsOnlyWhenAsked() returns error? {
    if managedDomainUrl == "" {
        return;
    }
    string indexName = liveIndexName(managedTarget(), "md-embeddings");
    ManagedDomainVectorStore defaults = check newManagedStore(indexName);
    check defaults.add([liveEntry("e-1")]);

    ai:VectorMatch[] withoutVectors = check defaults.query({embedding: liveTestVector(), topK: 1});
    test:assertEquals(withoutVectors[0].embedding, <ai:Vector>[], "vectors should be left out by default");

    ManagedDomainVectorStore asking = check newManagedStore(indexName, storeConfig = {includeEmbeddingsInResults: true});
    ai:VectorMatch[] withVectors = check asking.query({embedding: liveTestVector(), topK: 1});
    test:assertEquals(withVectors[0].embedding, liveTestVector());
}

@test:Config {groups: ["live"], enable: isLiveServer}
isolated function testManagedDomainMissingIndexFailsInitWhenNotCreating() returns error? {
    if managedDomainUrl == "" {
        return;
    }
    string indexName = liveIndexName(managedTarget(), "md-missing");
    ManagedDomainVectorStore|ai:Error result = newManagedStore(indexName, createIndexIfNotExists = false);
    if result !is ai:Error {
        test:assertFail("a missing index that may not be created should fail construction");
    }
    test:assertTrue(result.message().includes("createIndexIfNotExists"), result.message());

    http:Response head = check liveSignedRequest(managedTarget(), "HEAD", "/" + indexName);
    test:assertEquals(head.statusCode, 404, "construction must not have created the index");
}

@test:Config {groups: ["live"], enable: isLiveServer}
isolated function testManagedDomainDimensionMismatchFailsInit() returns error? {
    if managedDomainUrl == "" {
        return;
    }
    string indexName = liveIndexName(managedTarget(), "md-dimension");
    _ = check newManagedStore(indexName);

    ManagedDomainVectorStore|ai:Error result = newManagedStore(indexName,
            {queryMode: ai:DENSE, indexConfig: {dimension: LIVE_TEST_DIMENSION * 2}});
    if result !is ai:Error {
        test:assertFail("a store whose dimension differs from the existing index should fail construction");
    }
    test:assertTrue(result.message().includes("dimension"), result.message());
}

@test:Config {groups: ["live"], enable: isLiveServer}
isolated function testManagedDomainCreatesTheConfiguredEngine() returns error? {
    if managedDomainUrl == "" {
        return;
    }
    string indexName = liveIndexName(managedTarget(), "md-lucene");
    _ = check newManagedStore(indexName, storeConfig = {engine: LUCENE});

    map<json> properties = check liveMappingProperties(managedTarget(), indexName);
    test:assertEquals(check properties.embedding.method.engine, "lucene");
}

@test:Config {groups: ["live"], enable: isLiveServer}
isolated function testManagedDomainSparseRoundTrip() returns error? {
    if managedDomainUrl == "" {
        return;
    }
    ManagedDomainVectorStore store = check newManagedStore(liveIndexName(managedTarget(), "md-sparse"),
            {queryMode: ai:SPARSE});
    check store.add([
        {id: "sparse-hit", embedding: {indices: [7, 9], values: [0.3, 2.5]}, chunk: <ai:TextChunk>{content: "hit"}},
        {id: "sparse-miss", embedding: {indices: [11], values: [1.0]}, chunk: <ai:TextChunk>{content: "miss"}}
    ]);

    ai:VectorMatch[] matches = check store.query({embedding: {indices: [9], values: [1.0]}, topK: 10});
    test:assertEquals(matches.length(), 1, "only the document sharing a token should match");
    test:assertEquals(matches[0].id, "sparse-hit");
}

@test:Config {groups: ["live"], enable: isLiveServer}
isolated function testManagedDomainHybridRoundTrip() returns error? {
    if managedDomainUrl == "" {
        return;
    }
    ManagedDomainVectorStore store = check newManagedStore(liveIndexName(managedTarget(), "md-hybrid"),
            {queryMode: ai:HYBRID, indexConfig: {dimension: LIVE_TEST_DIMENSION}});
    check store.add([
        {
            id: "hybrid-1",
            embedding: {dense: liveTestVector(), sparse: {indices: [42], values: [1.5]}},
            chunk: <ai:TextChunk>{content: "hybrid"}
        }
    ]);

    ai:VectorMatch[] matches = check store.query({
        embedding: {dense: liveTestVector(), sparse: {indices: [42], values: [1.0]}},
        topK: 5
    });
    test:assertEquals(matches.length(), 1);
    test:assertEquals(matches[0].id, "hybrid-1");
    test:assertTrue(matches[0].similarityScore > 0.0 && matches[0].similarityScore <= 1.0,
            string `a fused score should be in (0.0, 1.0], got ${matches[0].similarityScore}`);
}

@test:Config {groups: ["live"], enable: isLiveServer}
isolated function testManagedDomainBasicAuthRoundTrip() returns error? {
    if managedDomainUrl == "" || managedDomainUsername == "" {
        return;
    }
    ManagedDomainVectorStore store = check new (managedDomainUrl, managedDomainRegion,
        liveIndexName(managedTarget(), "md-basic"), {username: managedDomainUsername, password: managedDomainPassword},
        searchMode = liveDenseMode(), storeConfig = {refreshOnWrite: true}
    );
    check store.add([liveEntry("basic-1")]);
    test:assertEquals(check countMatches(store, "basic-1"), 1);
}

// --- Serverless Classic ------------------------------------------------------------------------

@test:Config {groups: ["live"], enable: isLiveServer}
isolated function testServerlessClassicSameIdTwiceIsAppendOnlyAndDeleteRemovesBoth() returns error? {
    if serverlessClassicUrl == "" {
        return;
    }
    ServerlessVectorStore store = check new (serverlessClassicUrl, serverlessClassicRegion,
        liveIndexName(classicTarget(), "sc-append"), {generation: CLASSIC, auth: auth:DEFAULT_CREDENTIALS},
        searchMode = liveDenseMode()
    );

    string id = uuid:createRandomUuid();
    check store.add([liveEntry(id, "v1")]);
    test:assertTrue(check pollUntilVisible(store, id, CLASSIC_VISIBILITY_TIMEOUT),
            "the first write should become visible within the Classic refresh window");

    check store.add([liveEntry(id, "v2")]);
    runtime:sleep(65); // let the second write cross the refresh boundary too
    test:assertEquals(check countMatches(store, id), 2,
            "re-adding the same id on Serverless Classic is append-only and must produce two hits");

    // Delete looks the internal `_id`s up first, so it must remove both copies.
    check store.delete(id);
    test:assertTrue(check pollUntilGone(store, id, CLASSIC_VISIBILITY_TIMEOUT),
            "delete should remove every copy of the id");
}

// --- Serverless NextGen ------------------------------------------------------------------------

isolated function newNextGenStore(string indexName, SearchMode searchMode = liveDenseMode(),
        CompressionLevel? compressionLevel = ()) returns ServerlessVectorStore|ai:Error =>
    new (serverlessNextGenUrl, serverlessNextGenRegion, indexName,
        {generation: NEXTGEN, auth: auth:DEFAULT_CREDENTIALS, compressionLevel}, searchMode = searchMode
    );

@test:Config {groups: ["live"], enable: isLiveServer}
isolated function testServerlessNextGenSameIdTwiceUpserts() returns error? {
    if serverlessNextGenUrl == "" {
        return;
    }
    ServerlessVectorStore store = check newNextGenStore(liveIndexName(nextGenTarget(), "ng-upsert"));

    string id = uuid:createRandomUuid();
    check store.add([liveEntry(id, "v1")]);
    test:assertTrue(check pollUntilVisible(store, id, NEXTGEN_VISIBILITY_TIMEOUT),
            "the first write should become visible within the NextGen refresh window");

    check store.add([liveEntry(id, "v2")]);
    runtime:sleep(15); // let the second write cross the refresh boundary too
    test:assertEquals(check countMatches(store, id), 1, "re-adding the same id on NextGen must upsert, not duplicate");

    check store.delete(id);
    test:assertTrue(check pollUntilGone(store, id, NEXTGEN_VISIBILITY_TIMEOUT));
}

@test:Config {groups: ["live"], enable: isLiveServer}
isolated function testServerlessNextGenUncompressedVectors() returns error? {
    if serverlessNextGenUrl == "" {
        return;
    }
    string indexName = liveIndexName(nextGenTarget(), "ng-1x");
    ServerlessVectorStore store = check newNextGenStore(indexName, compressionLevel = COMPRESSION_1X);

    map<json> properties = check liveMappingProperties(nextGenTarget(), indexName);
    test:assertEquals(check properties.embedding.compression_level, "1x",
            "the stored mapping should carry the requested compression level, not NextGen's 32x default");

    check store.add([liveEntry("ng-1x-1")]);
    test:assertTrue(check pollUntilVisible(store, "ng-1x-1", NEXTGEN_VISIBILITY_TIMEOUT));
}

@test:Config {groups: ["live"], enable: isLiveServer}
isolated function testServerlessNextGenSparseRoundTrip() returns error? {
    if serverlessNextGenUrl == "" {
        return;
    }
    ServerlessVectorStore store = check newNextGenStore(liveIndexName(nextGenTarget(), "ng-sparse"),
            {queryMode: ai:SPARSE});
    check store.add([
        {id: "sparse-hit", embedding: {indices: [7, 9], values: [0.3, 2.5]}, chunk: <ai:TextChunk>{content: "hit"}}
    ]);

    decimal waited = 0;
    ai:VectorMatch[] matches = [];
    while waited < NEXTGEN_VISIBILITY_TIMEOUT && matches.length() == 0 {
        matches = check store.query({embedding: {indices: [9], values: [1.0]}, topK: 10});
        if matches.length() == 0 {
            runtime:sleep(POLL_INTERVAL_SECONDS);
            waited += POLL_INTERVAL_SECONDS;
        }
    }
    test:assertEquals(matches.length(), 1, "a sparse write should become searchable on NextGen");
    test:assertEquals(matches[0].id, "sparse-hit");
}

@test:Config {groups: ["live"], enable: isLiveServer}
isolated function testServerlessNextGenPerAccountEndpoint() returns error? {
    if serverlessNextGenAccountUrl == "" || serverlessNextGenCollectionName == "" {
        return;
    }
    ServerlessVectorStore store = check new (serverlessNextGenAccountUrl, serverlessNextGenRegion,
        liveIndexName(nextGenAccountTarget(), "ng-account"),
        {generation: NEXTGEN, auth: auth:DEFAULT_CREDENTIALS, collection: {name: serverlessNextGenCollectionName}},
        searchMode = liveDenseMode()
    );

    check store.add([liveEntry("account-1")]);
    test:assertTrue(check pollUntilVisible(store, "account-1", NEXTGEN_VISIBILITY_TIMEOUT),
            "the collection header should route reads and writes to the named collection");
}
