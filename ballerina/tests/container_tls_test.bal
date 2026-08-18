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

// The `BasicAuth` branch of the transport, over TLS, against the second container -- the one
// running the security plugin and its internal user database. This is the fine-grained access
// control setup a managed domain uses, and it is the only path in `sendSigned` that does not go
// through SigV4. Nothing else in the suite reaches it.
//
// `secureSocket: {enable: false}` skips verification of the container's self-signed demo
// certificate. That is a property of the fixture, not a recommendation: the point being tested is
// that `httpConfig` is passed through to the underlying client at all.

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
    return new (secureContainerUrl, CONTAINER_REGION, indexName, MANAGED_DOMAIN, auth,
        containerConfig(), ai:DENSE, secureContainerHttpConfig()
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
    // The 401/403 hint names both the managed-domain and the Serverless diagnosis, since the
    // module cannot tell which one the caller is looking at.
    test:assertTrue(result.message().includes("FGAC") || result.message().includes("IAM"),
            string `the auth hint should be attached, got: ${result.message()}`);
}

@test:Config {groups: ["docker"]}
isolated function testSecureContainerRejectsSigV4Credentials() returns error? {
    string indexName = containerIndexName("tls-sigv4");
    // The security plugin has no idea what a SigV4 `Authorization` header is. This is the same
    // failure a caller gets by pointing SigV4 credentials at an FGAC domain, so it is worth
    // confirming it arrives as a clean 401 rather than an unparsed transport error.
    VectorStore|ai:Error result = new (secureContainerUrl, CONTAINER_REGION, indexName, MANAGED_DOMAIN,
        containerAuth, containerConfig(), ai:DENSE, secureContainerHttpConfig()
    );

    if result !is ai:Error {
        check result.close();
        test:assertFail("SigV4 credentials should not authenticate against the internal user database");
    }
    test:assertEquals(result.detail()["status"], 401,
            string `expected a 401, got: ${result.message()}`);
}
