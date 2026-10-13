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
// or `./gradlew clean test -Pgroups=live`. Serverless is polled, since its refresh interval cannot
// be forced.

import ballerina/ai;
import ballerina/lang.runtime;
import ballerina/test;
import ballerina/uuid;
import ballerinax/aws.auth;

configurable boolean isLiveServer = false;

configurable string managedDomainUrl = "";
configurable string managedDomainRegion = "us-east-1";

configurable string serverlessClassicUrl = "";
configurable string serverlessClassicRegion = "us-east-1";

configurable string serverlessNextGenUrl = "";
configurable string serverlessNextGenRegion = "us-east-1";

const int LIVE_TEST_DIMENSION = 8;
const decimal POLL_INTERVAL_SECONDS = 3;

isolated function liveTestVector() returns ai:Vector => [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

isolated function liveIndexName(string prefix) returns string => string `${prefix}-${uuid:createRandomUuid()}`;

# Polls the `query` operation until a match with the given id appears, or `timeoutSeconds` elapses.
#
# + store - The vector store to poll
# + id - The logical id to wait for
# + timeoutSeconds - The maximum time to wait
# + return - `true` if found within the timeout
isolated function pollUntilVisible(VectorStore store, string id, decimal timeoutSeconds) returns boolean|ai:Error {
    decimal waited = 0;
    while waited < timeoutSeconds {
        ai:VectorMatch[] matches = check store.query({embedding: liveTestVector(), topK: 10});
        foreach ai:VectorMatch m in matches {
            if m.id == id {
                return true;
            }
        }
        runtime:sleep(POLL_INTERVAL_SECONDS);
        waited += POLL_INTERVAL_SECONDS;
    }
    return false;
}

# Polls the `query` operation until a match with the given id disappears, or `timeoutSeconds` elapses.
#
# + store - The vector store to poll
# + id - The logical id to wait to disappear
# + timeoutSeconds - The maximum time to wait
# + return - `true` if the id is gone within the timeout
isolated function pollUntilGone(VectorStore store, string id, decimal timeoutSeconds) returns boolean|ai:Error {
    decimal waited = 0;
    while waited < timeoutSeconds {
        ai:VectorMatch[] matches = check store.query({embedding: liveTestVector(), topK: 10});
        boolean stillPresent = false;
        foreach ai:VectorMatch m in matches {
            if m.id == id {
                stillPresent = true;
            }
        }
        if !stillPresent {
            return true;
        }
        runtime:sleep(POLL_INTERVAL_SECONDS);
        waited += POLL_INTERVAL_SECONDS;
    }
    return false;
}

@test:Config {groups: ["live"], enable: isLiveServer}
isolated function testManagedDomainAddQueryDeleteRoundTrip() returns error? {
    if managedDomainUrl == "" {
        return;
    }
    string indexName = liveIndexName("md-roundtrip");
    VectorStore store = check new (managedDomainUrl, managedDomainRegion, indexName,
        {deploymentType: MANAGED_DOMAIN, auth: auth:DEFAULT_CREDENTIALS, refreshOnWrite: true},
        {queryMode: ai:DENSE, indexConfig: {dimension: LIVE_TEST_DIMENSION}}
    );

    string id = uuid:createRandomUuid();
    check store.add([{id, embedding: liveTestVector(), chunk: {'type: "text-chunk", content: "hello"}}]);

    ai:VectorMatch[] matches = check store.query({embedding: liveTestVector(), topK: 1});
    test:assertEquals(matches.length(), 1);
    test:assertEquals(matches[0].id, id);

    check store.delete(id);
    boolean gone = check pollUntilGone(store, id, 15);
    test:assertTrue(gone, "the entry should be gone immediately after delete on a managed domain");

    check store.close();
}

@test:Config {groups: ["live"], enable: isLiveServer}
isolated function testManagedDomainSameIdTwiceUpserts() returns error? {
    if managedDomainUrl == "" {
        return;
    }
    string indexName = liveIndexName("md-upsert");
    VectorStore store = check new (managedDomainUrl, managedDomainRegion, indexName,
        {deploymentType: MANAGED_DOMAIN, auth: auth:DEFAULT_CREDENTIALS, refreshOnWrite: true},
        {queryMode: ai:DENSE, indexConfig: {dimension: LIVE_TEST_DIMENSION}}
    );

    string id = uuid:createRandomUuid();
    check store.add([{id, embedding: liveTestVector(), chunk: {'type: "text-chunk", content: "v1"}}]);
    check store.add([{id, embedding: liveTestVector(), chunk: {'type: "text-chunk", content: "v2"}}]);

    ai:VectorMatch[] matches = check store.query({embedding: liveTestVector(), topK: 10});
    int occurrences = 0;
    foreach ai:VectorMatch m in matches {
        if m.id == id {
            occurrences += 1;
        }
    }
    test:assertEquals(occurrences, 1, "re-adding the same id on a managed domain must upsert, not duplicate");

    check store.delete(id);
    check store.close();
}

@test:Config {groups: ["live"], enable: isLiveServer}
isolated function testManagedDomainFilteredQuery() returns error? {
    if managedDomainUrl == "" {
        return;
    }
    string indexName = liveIndexName("md-filter");
    VectorStore store = check new (managedDomainUrl, managedDomainRegion, indexName,
        {deploymentType: MANAGED_DOMAIN, auth: auth:DEFAULT_CREDENTIALS, refreshOnWrite: true},
        {queryMode: ai:DENSE, indexConfig: {dimension: LIVE_TEST_DIMENSION}}
    );

    string matchingId = uuid:createRandomUuid();
    string otherId = uuid:createRandomUuid();
    check store.add([
        {
            id: matchingId,
            embedding: liveTestVector(),
            chunk: {'type: "text-chunk", content: "en doc", metadata: {"language": "en"}}
        },
        {
            id: otherId,
            embedding: liveTestVector(),
            chunk: {'type: "text-chunk", content: "fr doc", metadata: {"language": "fr"}}
        }
    ]);

    ai:VectorMatch[] matches = check store.query({
        embedding: liveTestVector(),
        topK: 10,
        filters: {filters: [{key: "language", operator: ai:EQUAL, value: "en"}]}
    });
    test:assertEquals(matches.length(), 1);
    test:assertEquals(matches[0].id, matchingId);

    check store.delete([matchingId, otherId]);
    check store.close();
}

@test:Config {groups: ["live"], enable: isLiveServer}
isolated function testServerlessClassicSameIdTwiceIsAppendOnly() returns error? {
    if serverlessClassicUrl == "" {
        return;
    }
    string indexName = liveIndexName("sc-append");
    VectorStore store = check new (serverlessClassicUrl, serverlessClassicRegion, indexName,
        {deploymentType: SERVERLESS_CLASSIC, auth: auth:DEFAULT_CREDENTIALS}, {queryMode: ai:DENSE, indexConfig: {dimension: LIVE_TEST_DIMENSION}}
    );

    string id = uuid:createRandomUuid();
    check store.add([{id, embedding: liveTestVector(), chunk: {'type: "text-chunk", content: "v1"}}]);
    boolean visible = check pollUntilVisible(store, id, 90);
    test:assertTrue(visible, "the first write should become visible within the ~60s Classic refresh window");

    check store.add([{id, embedding: liveTestVector(), chunk: {'type: "text-chunk", content: "v2"}}]);
    runtime:sleep(65); // let the second write cross the refresh boundary too

    ai:VectorMatch[] matches = check store.query({embedding: liveTestVector(), topK: 10});
    int occurrences = 0;
    foreach ai:VectorMatch m in matches {
        if m.id == id {
            occurrences += 1;
        }
    }
    test:assertEquals(occurrences, 2,
            "re-adding the same id on Serverless Classic is append-only and must produce two hits");

    check store.delete(id);
    check store.close();
}

@test:Config {groups: ["live"], enable: isLiveServer}
isolated function testServerlessNextGenSameIdTwiceUpserts() returns error? {
    if serverlessNextGenUrl == "" {
        return;
    }
    string indexName = liveIndexName("ng-upsert");
    VectorStore store = check new (serverlessNextGenUrl, serverlessNextGenRegion, indexName,
        {deploymentType: SERVERLESS_NEXTGEN, auth: auth:DEFAULT_CREDENTIALS}, {queryMode: ai:DENSE, indexConfig: {dimension: LIVE_TEST_DIMENSION}}
    );

    string id = uuid:createRandomUuid();
    check store.add([{id, embedding: liveTestVector(), chunk: {'type: "text-chunk", content: "v1"}}]);
    boolean firstVisible = check pollUntilVisible(store, id, 30);
    test:assertTrue(firstVisible, "the first write should become visible within the ~10s NextGen refresh window");

    check store.add([{id, embedding: liveTestVector(), chunk: {'type: "text-chunk", content: "v2"}}]);
    runtime:sleep(15); // let the second write cross the refresh boundary too

    ai:VectorMatch[] matches = check store.query({embedding: liveTestVector(), topK: 10});
    int occurrences = 0;
    foreach ai:VectorMatch m in matches {
        if m.id == id {
            occurrences += 1;
        }
    }
    test:assertEquals(occurrences, 1, "re-adding the same id on Serverless NextGen must upsert, not duplicate");

    check store.delete(id);
    check store.close();
}
