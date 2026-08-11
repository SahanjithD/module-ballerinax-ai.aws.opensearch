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
import ballerina/test;
import ballerinax/aws.auth;

const string VALID_URL = "https://my-domain.us-east-1.es.amazonaws.com";

isolated function validConfig() returns Configuration => {indexConfig: {dimension: 1536}};

// --- rule 1: BasicAuth is MANAGED_DOMAIN only ---------------------------------------------------

@test:Config
isolated function testBasicAuthRejectedOnServerlessClassic() {
    BasicAuth basicAuth = {username: "u", password: "p"};
    ai:Error? result = validateConfiguration(VALID_URL, SERVERLESS_CLASSIC, basicAuth, ai:DENSE, validConfig());
    test:assertTrue(result is ai:Error);
}

@test:Config
isolated function testBasicAuthRejectedOnServerlessNextGen() {
    BasicAuth basicAuth = {username: "u", password: "p"};
    ai:Error? result = validateConfiguration(VALID_URL, SERVERLESS_NEXTGEN, basicAuth, ai:DENSE, validConfig());
    test:assertTrue(result is ai:Error);
}

@test:Config
isolated function testBasicAuthAllowedOnManagedDomain() {
    BasicAuth basicAuth = {username: "u", password: "p"};
    ai:Error? result = validateConfiguration(VALID_URL, MANAGED_DOMAIN, basicAuth, ai:DENSE, validConfig());
    test:assertTrue(result is ());
}

// --- rule 2: engine must be FAISS on SERVERLESS_CLASSIC -----------------------------------------

@test:Config
isolated function testNonFaissEngineRejectedOnServerlessClassic() {
    Configuration config = {indexConfig: {dimension: 8, engine: NMSLIB}};
    ai:Error? result = validateConfiguration(VALID_URL, SERVERLESS_CLASSIC, auth:DEFAULT_CREDENTIALS, ai:DENSE, config);
    test:assertTrue(result is ai:Error);
}

@test:Config
isolated function testFaissEngineAllowedOnServerlessClassic() {
    Configuration config = {indexConfig: {dimension: 8, engine: FAISS}};
    ai:Error? result = validateConfiguration(VALID_URL, SERVERLESS_CLASSIC, auth:DEFAULT_CREDENTIALS, ai:DENSE, config);
    test:assertTrue(result is ());
}

@test:Config
isolated function testNonFaissEngineAllowedOnManagedDomain() {
    Configuration config = {indexConfig: {dimension: 8, engine: LUCENE}};
    ai:Error? result = validateConfiguration(VALID_URL, MANAGED_DOMAIN, auth:DEFAULT_CREDENTIALS, ai:DENSE, config);
    test:assertTrue(result is ());
}

// --- rule 3: only DENSE query mode is supported --------------------------------------------------

@test:Config
isolated function testSparseQueryModeRejected() {
    ai:Error? result = validateConfiguration(VALID_URL, MANAGED_DOMAIN, auth:DEFAULT_CREDENTIALS, ai:SPARSE, validConfig());
    test:assertTrue(result is ai:Error);
}

@test:Config
isolated function testHybridQueryModeRejected() {
    ai:Error? result = validateConfiguration(VALID_URL, MANAGED_DOMAIN, auth:DEFAULT_CREDENTIALS, ai:HYBRID, validConfig());
    test:assertTrue(result is ai:Error);
}

// --- rule 4: dimension must be positive ------------------------------------------------------

@test:Config
isolated function testZeroDimensionRejected() {
    Configuration config = {indexConfig: {dimension: 0}};
    ai:Error? result = validateConfiguration(VALID_URL, MANAGED_DOMAIN, auth:DEFAULT_CREDENTIALS, ai:DENSE, config);
    test:assertTrue(result is ai:Error);
}

@test:Config
isolated function testNegativeDimensionRejected() {
    Configuration config = {indexConfig: {dimension: -1}};
    ai:Error? result = validateConfiguration(VALID_URL, MANAGED_DOMAIN, auth:DEFAULT_CREDENTIALS, ai:DENSE, config);
    test:assertTrue(result is ai:Error);
}

// --- rule 5: serviceUrl must parse to a host -----------------------------------------------------

@test:Config
isolated function testServiceUrlWithoutSchemeRejected() {
    ai:Error? result = validateConfiguration("my-domain.us-east-1.es.amazonaws.com", MANAGED_DOMAIN,
            auth:DEFAULT_CREDENTIALS, ai:DENSE, validConfig());
    test:assertTrue(result is ai:Error);
}

@test:Config
isolated function testExtractHostStripsScheme() returns error? {
    test:assertEquals(check extractHost("https://my-domain.us-east-1.es.amazonaws.com"),
            "my-domain.us-east-1.es.amazonaws.com");
    test:assertEquals(check extractHost("http://localhost:9200"), "localhost:9200");
}

@test:Config
isolated function testExtractHostStripsTrailingPath() returns error? {
    test:assertEquals(check extractHost("https://my-domain.us-east-1.es.amazonaws.com/some/path"),
            "my-domain.us-east-1.es.amazonaws.com");
}

@test:Config
isolated function testExtractHostRejectsMissingHost() {
    string|ai:Error result = extractHost("https://");
    test:assertTrue(result is ai:Error);
}

// --- SigV4 query-string encoding -------------------------------------------------------------

@test:Config
isolated function testPercentEncodeLeavesUnreservedCharactersAlone() returns error? {
    test:assertEquals(check percentEncode("wait_for"), "wait_for");
    test:assertEquals(check percentEncode("abcXYZ012-._~"), "abcXYZ012-._~");
}

@test:Config
isolated function testPercentEncodeEncodesReservedCharacters() returns error? {
    // A space must become %20, matching RFC 3986/SigV4 canonicalization — NOT '+', which is what
    // ballerina/url:encode's application/x-www-form-urlencoded behavior would produce.
    test:assertEquals(check percentEncode("a b"), "a%20b");
    test:assertEquals(check percentEncode("a=b"), "a%3Db");
    test:assertEquals(check percentEncode("a&b"), "a%26b");
    test:assertEquals(check percentEncode("a/b"), "a%2Fb");
}

@test:Config
isolated function testPercentEncodeIsByteExactForMultiByteCharacters() returns error? {
    // "é" is U+00E9, UTF-8 bytes 0xC3 0xA9.
    test:assertEquals(check percentEncode("é"), "%C3%A9");
}

@test:Config
isolated function testBuildQueryStringSingleParam() returns error? {
    test:assertEquals(check buildQueryString({"refresh": "wait_for"}), "refresh=wait_for");
}

// --- rule 6: refreshOnWrite is MANAGED_DOMAIN only -----------------------------------------------

@test:Config
isolated function testRefreshOnWriteRejectedOnServerlessClassic() {
    Configuration config = {indexConfig: {dimension: 8}, refreshOnWrite: true};
    ai:Error? result = validateConfiguration(VALID_URL, SERVERLESS_CLASSIC, auth:DEFAULT_CREDENTIALS, ai:DENSE, config);
    test:assertTrue(result is ai:Error);
}

@test:Config
isolated function testRefreshOnWriteRejectedOnServerlessNextGen() {
    Configuration config = {indexConfig: {dimension: 8}, refreshOnWrite: true};
    ai:Error? result = validateConfiguration(VALID_URL, SERVERLESS_NEXTGEN, auth:DEFAULT_CREDENTIALS, ai:DENSE, config);
    test:assertTrue(result is ai:Error);
}

@test:Config
isolated function testRefreshOnWriteAllowedOnManagedDomain() {
    Configuration config = {indexConfig: {dimension: 8}, refreshOnWrite: true};
    ai:Error? result = validateConfiguration(VALID_URL, MANAGED_DOMAIN, auth:DEFAULT_CREDENTIALS, ai:DENSE, config);
    test:assertTrue(result is ());
}

// --- rule 7: maxBulkSize / maxResultWindow must be positive ---------------------------------------

@test:Config
isolated function testZeroMaxBulkSizeRejected() {
    Configuration config = {indexConfig: {dimension: 8}, maxBulkSize: 0};
    ai:Error? result = validateConfiguration(VALID_URL, MANAGED_DOMAIN, auth:DEFAULT_CREDENTIALS, ai:DENSE, config);
    test:assertTrue(result is ai:Error);
}

@test:Config
isolated function testZeroMaxResultWindowRejected() {
    Configuration config = {indexConfig: {dimension: 8}, maxResultWindow: 0};
    ai:Error? result = validateConfiguration(VALID_URL, MANAGED_DOMAIN, auth:DEFAULT_CREDENTIALS, ai:DENSE, config);
    test:assertTrue(result is ai:Error);
}

// --- a valid configuration passes all seven rules --------------------------------------------

@test:Config
isolated function testFullyValidConfigurationPasses() {
    ai:Error? result = validateConfiguration(VALID_URL, MANAGED_DOMAIN, auth:DEFAULT_CREDENTIALS, ai:DENSE, validConfig());
    test:assertTrue(result is ());
}

// --- fully offline construction ----------------------------------------------------------------

@test:Config
isolated function testOfflineConstructionWithCreateIndexDisabled() returns error? {
    // With createIndexIfNotExists = false, init() must perform no network I/O at all, so
    // construction succeeds even with a service URL that resolves to nothing.
    VectorStore store = check new (
        VALID_URL,
        "us-east-1",
        "test-index",
        MANAGED_DOMAIN,
        {accessKeyId: "AKIAFAKEFAKEFAKEFAKE", secretAccessKey: "fake-secret"},
        {indexConfig: {dimension: 8, createIndexIfNotExists: false}}
    );
    ai:Error? closeResult = store.close();
    test:assertTrue(closeResult is (), "closing a fully offline store should not fail");
}
