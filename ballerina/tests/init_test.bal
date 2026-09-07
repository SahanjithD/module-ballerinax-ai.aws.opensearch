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

const string VALID_URL = "https://my-domain.us-east-1.es.amazonaws.com";

isolated function validMode() returns DenseSearch => {queryMode: ai:DENSE, indexConfig: {dimension: 1536}};

// The rules this file once exercised at runtime -- `BasicAuth` outside a managed domain, a
// non-Faiss engine on Serverless Classic, `refreshOnWrite` off a managed domain, and quantization
// off NextGen -- are no longer reachable: each of those fields now lives on the one `Deployment`
// variant that honors it, so the wrong combination does not compile. There is no way to write a
// Ballerina test that asserts a compile error, and a runtime assertion would have to construct the
// very value the type system forbids, so those cases are covered by the types themselves and are
// deliberately absent here. What remains is the range and format checking no type can express.

// --- accepted deployments --------------------------------------------------------------------

@test:Config
isolated function testManagedDomainWithBasicAuthPasses() {
    BasicAuth basicAuth = {username: "u", password: "p"};
    ManagedDomainDeployment deployment = {deploymentType: MANAGED_DOMAIN, auth: basicAuth};
    ai:Error? result = validateConfiguration(VALID_URL, deployment, validMode(), {});
    test:assertTrue(result is ());
}

@test:Config
isolated function testManagedDomainWithNonFaissEnginePasses() {
    ai:Error? result = validateConfiguration(VALID_URL, managedDeployment(LUCENE), validMode(), {});
    test:assertTrue(result is ());
}

@test:Config
isolated function testManagedDomainWithRefreshOnWritePasses() {
    ManagedDomainDeployment deployment = managedDeployment(refreshOnWrite = true);
    ai:Error? result = validateConfiguration(VALID_URL, deployment, validMode(), {});
    test:assertTrue(result is ());
}

@test:Config
isolated function testEveryDeploymentTypePassesWithDefaults() {
    Deployment[] deployments = [
        managedDeployment(),
        classicDeployment(),
        nextGenDeployment()
    ];
    foreach Deployment deployment in deployments {
        ai:Error? result = validateConfiguration(VALID_URL, deployment, validMode(), {});
        test:assertTrue(result is (),
                string `a default ${deployment.deploymentType} deployment should validate`);
    }
}

// A record literal carrying only the discriminator has to resolve to exactly one member of the
// union -- which is why `deploymentType` is a required field on all three variants rather than a
// defaulted one. Were it defaulted everywhere, `{}` would be ambiguous and would not compile.
@test:Config
isolated function testDiscriminatorSelectsTheVariant() {
    Deployment managed = managedDeployment();
    Deployment classic = classicDeployment();
    Deployment nextGen = nextGenDeployment();
    test:assertTrue(managed is ManagedDomainDeployment);
    test:assertTrue(classic is ServerlessClassicDeployment);
    test:assertTrue(nextGen is ServerlessNextGenDeployment);
}

// --- rule 1: the supported query modes ---------------------------------------------------------

// The assertion here is that this compiles at all. `SparseSearch` declares no `IndexConfig`, so a
// sparse store cannot be made to name a vector dimension, a similarity metric or an HNSW
// parameter -- none of which a `rank_features` index has any use for.
@test:Config
isolated function testSparseModeNeedsNoDimension() {
    ai:Error? result = validateConfiguration(VALID_URL, managedDeployment(), {queryMode: ai:SPARSE}, {});
    test:assertTrue(result is (), "a SPARSE store should construct without an IndexConfig");
}

@test:Config
isolated function testHybridModeConstructsWithDefaultFusion() {
    ai:Error? result = validateConfiguration(VALID_URL, managedDeployment(),
            {queryMode: ai:HYBRID, indexConfig: {dimension: 8}}, {});
    test:assertTrue(result is (), "a HYBRID store should construct with the default inline fusion");
}

// --- rule 1b: hybrid fusion settings -----------------------------------------------------------

// The `normalization-processor` rejects a weight list that does not sum to 1.0. The pipeline is
// sent inline on every query, so an unchecked mistake would fail every call rather than one.
@test:Config
isolated function testHybridWeightsMustSumToOne() {
    ai:Error? result = validateConfiguration(VALID_URL, managedDeployment(),
            {queryMode: ai:HYBRID, indexConfig: {dimension: 8}, fusion: {denseWeight: 0.7, sparseWeight: 0.5}}, {});
    if result !is ai:Error {
        test:assertFail("fusion weights that do not sum to 1.0 must be rejected");
    }
    test:assertTrue(result.message().includes("sum to 1.0"));
}

// Binary floating point does not sum 0.7 and 0.3 to exactly 1.0; rejecting that split would be
// absurd, so the check carries a tolerance.
@test:Config
isolated function testHybridWeightsAcceptAnUnevenButValidSplit() {
    ai:Error? result = validateConfiguration(VALID_URL, managedDeployment(),
            {queryMode: ai:HYBRID, indexConfig: {dimension: 8}, fusion: {denseWeight: 0.7, sparseWeight: 0.3}}, {});
    test:assertTrue(result is (), string `0.7/0.3 is a valid split, got: ${(result is ai:Error).toString()}`);
}

@test:Config
isolated function testHybridWeightsMustBeInRange() {
    ai:Error? result = validateConfiguration(VALID_URL, managedDeployment(),
            {queryMode: ai:HYBRID, indexConfig: {dimension: 8}, fusion: {denseWeight: 1.5, sparseWeight: -0.5}}, {});
    if result !is ai:Error {
        test:assertFail("a weight outside [0.0, 1.0] must be rejected");
    }
    test:assertTrue(result.message().includes("between 0.0 and 1.0"));
}

@test:Config
isolated function testHybridNamedPipelineMustNotBeBlank() {
    ai:Error? result = validateConfiguration(VALID_URL, managedDeployment(),
            {queryMode: ai:HYBRID, indexConfig: {dimension: 8}, fusion: {name: "  "}}, {});
    if result !is ai:Error {
        test:assertFail("a blank pipeline name must be rejected");
    }
    test:assertTrue(result.message().includes("HybridSearchConfig"),
            string `the message should name the alternative, got: ${result.message()}`);
}

// A named pipeline defines the normalization technique, combination technique and weights itself,
// so naming one and setting the inline knobs are alternatives rather than layers. The union is
// what makes the combination unrepresentable instead of a runtime rule.
@test:Config
isolated function testHybridNamedPipelineConstructs() {
    ai:Error? result = validateConfiguration(VALID_URL, managedDeployment(),
            {queryMode: ai:HYBRID, indexConfig: {dimension: 8}, fusion: {name: "my-pipeline"}}, {});
    test:assertTrue(result is ());
}

@test:Config
isolated function testZeroMaxQueryTokensRejected() {
    ai:Error? result = validateConfiguration(VALID_URL, managedDeployment(),
            {queryMode: ai:SPARSE, maxQueryTokens: 0}, {});
    if result !is ai:Error {
        test:assertFail("'maxQueryTokens' must be positive");
    }
    test:assertTrue(result.message().includes("maxQueryTokens"));
}

// Both quantization knobs default to `()`, so "explicitly set" is distinguishable from "unset".
// They shape a `knn_vector` field that a SPARSE index does not have, and staying silent would
// repeat the failure mode `IndexConfig` already warns about at length.
@test:Config
isolated function testSparseModeRejectsNextGenQuantization() {
    ai:Error? result = validateConfiguration(VALID_URL, nextGenDeployment(COMPRESSION_8X),
            {queryMode: ai:SPARSE}, {});
    if result !is ai:Error {
        test:assertFail("quantization has no meaning on an index with no knn_vector field");
    }
    test:assertTrue(result.message().includes("HYBRID"),
            string `the message should name the way out, got: ${result.message()}`);
}

@test:Config
isolated function testSparseModeAllowsNextGenWithoutQuantization() {
    ai:Error? result = validateConfiguration(VALID_URL, nextGenDeployment(), {queryMode: ai:SPARSE}, {});
    test:assertTrue(result is (), "NextGen without quantization is a legitimate SPARSE target");
}

// --- rule 2: dimension must be positive --------------------------------------------------------

@test:Config
isolated function testZeroDimensionRejected() {
    DenseSearch configMode = {queryMode: ai:DENSE, indexConfig: {dimension: 0}};
    ai:Error? result = validateConfiguration(VALID_URL, managedDeployment(), configMode, {});
    test:assertTrue(result is ai:Error);
}

@test:Config
isolated function testNegativeDimensionRejected() {
    DenseSearch configMode = {queryMode: ai:DENSE, indexConfig: {dimension: -1}};
    ai:Error? result = validateConfiguration(VALID_URL, managedDeployment(), configMode, {});
    test:assertTrue(result is ai:Error);
}

// --- rule 3: serviceUrl must parse to a host ---------------------------------------------------

@test:Config
isolated function testServiceUrlWithoutSchemeRejected() {
    ai:Error? result = validateConfiguration("my-domain.us-east-1.es.amazonaws.com",
            managedDeployment(), validMode(), {});
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

// --- rule 4: maxBulkSize / maxResultWindow must be positive ------------------------------------

@test:Config
isolated function testZeroMaxBulkSizeRejected() {
    DenseSearch configMode = {queryMode: ai:DENSE, indexConfig: {dimension: 8}};
    Configuration config = {maxBulkSize: 0};
    ai:Error? result = validateConfiguration(VALID_URL, managedDeployment(), configMode, config);
    test:assertTrue(result is ai:Error);
}

@test:Config
isolated function testZeroMaxResultWindowRejected() {
    DenseSearch configMode = {queryMode: ai:DENSE, indexConfig: {dimension: 8}};
    Configuration config = {maxResultWindow: 0};
    ai:Error? result = validateConfiguration(VALID_URL, managedDeployment(), configMode, config);
    test:assertTrue(result is ai:Error);
}

// --- a valid configuration passes every rule ---------------------------------------------------

@test:Config
isolated function testFullyValidConfigurationPasses() {
    ai:Error? result = validateConfiguration(VALID_URL, managedDeployment(), validMode(), {});
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
        managedDeployment(),
        {queryMode: ai:DENSE, indexConfig: {dimension: 8}},
        {createIndexIfNotExists: false}
    );
    ai:Error? closeResult = store.close();
    test:assertTrue(closeResult is (), "closing a fully offline store should not fail");
}

// --- rule 5: the two quantization knobs are mutually constrained -------------------------------

@test:Config
isolated function testQuantizationAllowedOnServerlessNextGen() {
    ai:Error? result = validateConfiguration(VALID_URL, nextGenDeployment(COMPRESSION_1X, IN_MEMORY),
            validMode(), {});
    test:assertTrue(result is (), "in_memory/1x is the documented way to opt out of quantization");
}

// Mirrors the server's own validation, which fails index creation with
// `Cannot specify "x1" compression level when using "on_disk" mode`. Caught at construction so it
// surfaces as a configuration error rather than a mapper_parsing_exception from a PUT.
@test:Config
isolated function testOnDiskWithNoCompressionRejected() {
    ai:Error? result = validateConfiguration(VALID_URL, nextGenDeployment(COMPRESSION_1X, ON_DISK),
            validMode(), {});
    if result !is ai:Error {
        test:assertFail("'ON_DISK' with 'COMPRESSION_1X' is rejected by the server and should be caught here");
    }
    test:assertTrue(result.message().includes("IN_MEMORY"),
            string `the message should name the way out, got: ${result.message()}`);
}

@test:Config
isolated function testUnsetQuantizationIsAllowedOnNextGen() {
    ai:Error? result = validateConfiguration(VALID_URL, nextGenDeployment(), validMode(), {});
    test:assertTrue(result is (), "leaving both unset reproduces the server's own default");
}

@test:Config
isolated function testCompressionLevelWithoutVectorModeIsAllowed() {
    ai:Error? result = validateConfiguration(VALID_URL, nextGenDeployment(COMPRESSION_1X), validMode(), {});
    test:assertTrue(result is (),
            "'COMPRESSION_1X' is only rejected alongside 'ON_DISK', which is not set here");
}

// --- rule 6: the two NextGen collection headers are alternatives -------------------------------

@test:Config
isolated function testBothCollectionIdentifiersRejected() {
    ServerlessNextGenDeployment deployment = {
        deploymentType: SERVERLESS_NEXTGEN,
        auth: TEST_CREDENTIALS,
        collectionName: "vectors",
        collectionId: "abc123"
    };
    ai:Error? result = validateConfiguration(VALID_URL, deployment, validMode(), {});
    if result !is ai:Error {
        test:assertFail("naming a collection twice should be rejected, not silently resolved");
    }
    test:assertTrue(result.message().includes("alternatives"),
            string `the message should say they are alternatives, got: ${result.message()}`);
}

@test:Config
isolated function testEitherCollectionIdentifierAloneIsAccepted() {
    ServerlessNextGenDeployment byName = {
        deploymentType: SERVERLESS_NEXTGEN,
        auth: TEST_CREDENTIALS,
        collectionName: "vectors"
    };
    ServerlessNextGenDeployment byId = {
        deploymentType: SERVERLESS_NEXTGEN,
        auth: TEST_CREDENTIALS,
        collectionId: "abc123"
    };
    test:assertTrue(validateConfiguration(VALID_URL, byName, validMode(), {}) is ());
    test:assertTrue(validateConfiguration(VALID_URL, byId, validMode(), {}) is ());
}

// --- the signed collection headers -------------------------------------------------------------

@test:Config
isolated function testCollectionHeadersOnlyOnNextGen() {
    test:assertEquals(buildCollectionHeaders(managedDeployment()), {},
                                                                   "a managed domain identifies its target by hostname");
    test:assertEquals(buildCollectionHeaders(classicDeployment()), {},
                                                                   "a Classic endpoint is per-collection, so it needs no header");
    test:assertEquals(buildCollectionHeaders(nextGenDeployment()), {},
                                                                   "an unset collection means a per-collection NextGen endpoint; no header to send");
}

@test:Config
isolated function testCollectionNameAndIdMapToTheirHeaders() {
    ServerlessNextGenDeployment byName = {
        deploymentType: SERVERLESS_NEXTGEN,
        auth: TEST_CREDENTIALS,
        collectionName: "vectors"
    };
    ServerlessNextGenDeployment byId = {
        deploymentType: SERVERLESS_NEXTGEN,
        auth: TEST_CREDENTIALS,
        collectionId: "abc123"
    };
    test:assertEquals(buildCollectionHeaders(byName), {"x-amz-aoss-collection-name": "vectors"});
    test:assertEquals(buildCollectionHeaders(byId), {"x-amz-aoss-collection-id": "abc123"});
}
