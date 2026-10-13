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

// `BasicAuth` over TLS against the container running the security plugin: the only path in
// `sendSigned` that skips SigV4. `secureSocket: {enable: false}` only accepts the demo certificate.

import ballerina/ai;
import ballerina/http;
import ballerina/test;

# Skips verification of the container's self-signed demo certificate.
#
# + return - The HTTP client configuration used against the TLS container
isolated function secureContainerHttpConfig() returns http:ClientConfiguration => {
    secureSocket: {enable: false}
};

# Constructs a store against the TLS container using basic authentication.
#
# + indexName - The target index
# + password - The internal-user-database password to present
# + return - The store, or an `ai:Error`
isolated function newSecureContainerStore(string indexName, string password = secureContainerPassword)
        returns VectorStore|ai:Error {
    BasicAuth auth = {username: secureContainerUsername, password};
    return new (secureContainerUrl, CONTAINER_REGION, indexName,
        {deploymentType: MANAGED_DOMAIN, auth, refreshOnWrite: true},
        containerDenseMode(), containerConfig(), secureContainerHttpConfig()
    );
}

@test:Config {groups: ["docker"]}
isolated function testSecureContainerBasicAuthRoundTrip() returns error? {
    string indexName = containerIndexName("tls-rt");
    VectorStore store = check newSecureContainerStore(indexName);

    check store.add([entry("secure-1", queryVector(), "over TLS", {"language": "en"})]);
    ai:VectorMatch[] matches = check store.query({embedding: queryVector(), topK: 10});
    test:assertEquals(matches.length(), 1);
    test:assertEquals(matches[0].id, "secure-1");
    test:assertEquals(matches[0].chunk.content, "over TLS");

    ai:VectorMatch[] filtered = check store.query({
        filters: {filters: [{key: "language", operator: ai:EQUAL, value: "en"}]},
        topK: 10
    });
    test:assertEquals(filtered.length(), 1, "filtering should work identically under basic auth");

    check store.delete("secure-1");
    ai:VectorMatch[] afterDelete = check store.query({topK: 10});
    test:assertEquals(afterDelete.length(), 0);
    check store.close();
}

@test:Config {groups: ["docker"]}
isolated function testSecureContainerRejectsWrongPassword() returns error? {
    string indexName = containerIndexName("tls-badpw");
    VectorStore|ai:Error result = newSecureContainerStore(indexName, "definitely-not-the-password");

    if result !is ai:Error {
        check result.close();
        test:assertFail("a wrong password should fail construction");
    }
    test:assertEquals(result.detail()["status"], 401,
            string `expected a 401, got: ${result.message()}`);
    // The hint covers both managed and Serverless, since the module cannot tell which applies.
    test:assertTrue(result.message().includes("FGAC") || result.message().includes("IAM"),
            string `the auth hint should be attached, got: ${result.message()}`);
}

@test:Config {groups: ["docker"]}
isolated function testSecureContainerRejectsSigV4Credentials() returns error? {
    string indexName = containerIndexName("tls-sigv4");
    // SigV4 credentials against an FGAC-only domain should fail as a clean 401.
    VectorStore|ai:Error result = new (secureContainerUrl, CONTAINER_REGION, indexName,
        containerDeployment(), containerDenseMode(), containerConfig(), secureContainerHttpConfig()
    );

    if result !is ai:Error {
        check result.close();
        test:assertFail("SigV4 credentials should not authenticate against the internal user database");
    }
    test:assertEquals(result.detail()["status"], 401,
            string `expected a 401, got: ${result.message()}`);
}
