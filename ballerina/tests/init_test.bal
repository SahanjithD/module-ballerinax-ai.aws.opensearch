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

// Loopback discard port: refuses at once rather than hanging, and stays off the network.
# A well-formed URL that nothing answers on.
const string UNREACHABLE_URL = "http://127.0.0.1:9";

isolated function validMode() returns DenseSearch => {queryMode: ai:DENSE, indexConfig: {dimension: 1536}};

// Wrong deployment/setting combinations do not compile, so only range and format checks are here.

// --- accepted deployments --------------------------------------------------------------------

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
                string `a default ${deploymentLabel(deployment)} deployment should validate`);
    }
}

// `generation` is required on both Serverless variants so a literal resolves to exactly one.
@test:Config
isolated function testGenerationSelectsTheServerlessVariant() {
    ServerlessDeployment classic = {generation: CLASSIC};
    ServerlessDeployment nextGen = {generation: NEXTGEN};
    test:assertTrue(classic is ServerlessClassicDeployment);
    test:assertTrue(nextGen is ServerlessNextGenDeployment);
}

// --- rule 1: the supported query modes ---------------------------------------------------------

// `SparseSearch` has no `IndexConfig`; a `rank_features` index has no dimension or HNSW settings.
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

// The `normalization-processor` rejects weights that do not sum to 1.0.
@test:Config
isolated function testHybridWeightsMustSumToOne() {
    ai:Error? result = validateConfiguration(VALID_URL, managedDeployment(),
            {queryMode: ai:HYBRID, indexConfig: {dimension: 8}, fusion: {denseWeight: 0.7, sparseWeight: 0.5}}, {});
    if result !is ai:Error {
        test:assertFail("fusion weights that do not sum to 1.0 must be rejected");
    }
    test:assertTrue(result.message().includes("sum to 1.0"));
}

// 0.7 + 0.3 is not exactly 1.0 in binary floating point, so the check needs a tolerance.
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
isolated function testHybridWeightsMustBeSetTogether() {
    ai:Error? result = validateConfiguration(VALID_URL, managedDeployment(),
            {queryMode: ai:HYBRID, indexConfig: {dimension: 8}, fusion: {denseWeight: 0.7}}, {});
    if result !is ai:Error {
        test:assertFail("one weight without the other must be rejected");
    }
    test:assertTrue(result.message().includes("set together"));
}

@test:Config
isolated function testHybridWeightsMayBothBeUnset() {
    ai:Error? result = validateConfiguration(VALID_URL, managedDeployment(),
            {queryMode: ai:HYBRID, indexConfig: {dimension: 8}, fusion: {normalization: L2}}, {});
    test:assertTrue(result is (), "unset weights leave equal weighting to OpenSearch");
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

// Quantization shapes a `knn_vector` field, which a SPARSE index does not have.
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

// --- rule 2: dimension must be 1 to 16,000 --------------------------------------------------------

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

@test:Config
isolated function testDimensionAboveMaximumRejected() {
    DenseSearch configMode = {queryMode: ai:DENSE, indexConfig: {dimension: 16001}};
    ai:Error? result = validateConfiguration(VALID_URL, managedDeployment(), configMode, {});
    test:assertTrue(result is ai:Error);
}

@test:Config
isolated function testDimensionAtMaximumAccepted() {
    DenseSearch configMode = {queryMode: ai:DENSE, indexConfig: {dimension: 16000}};
    ai:Error? result = validateConfiguration(VALID_URL, managedDeployment(), configMode, {});
    test:assertTrue(result is ());
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
    StoreConfiguration config = {maxBulkSize: 0};
    ai:Error? result = validateConfiguration(VALID_URL, managedDeployment(), configMode, config);
    test:assertTrue(result is ai:Error);
}

@test:Config
isolated function testZeroMaxResultWindowRejected() {
    DenseSearch configMode = {queryMode: ai:DENSE, indexConfig: {dimension: 8}};
    StoreConfiguration config = {maxResultWindow: 0};
    ai:Error? result = validateConfiguration(VALID_URL, managedDeployment(), configMode, config);
    test:assertTrue(result is ai:Error);
}

// --- a valid configuration passes every rule ---------------------------------------------------

@test:Config
isolated function testFullyValidConfigurationPasses() {
    ai:Error? result = validateConfiguration(VALID_URL, managedDeployment(), validMode(), {});
    test:assertTrue(result is ());
}

// --- construction-time checks ------------------------------------------------------------------

// Verification that cannot run (here, an unreachable endpoint) must warn, not fail construction.
@test:Config
isolated function testVerificationDoesNotFailConstructionWhenItCannotRun() returns error? {
    _ = check new VectorStoreCore(
        UNREACHABLE_URL,
        "us-east-1",
        "test-index",
        TEST_CREDENTIALS,
        managedDeployment(),
        {queryMode: ai:DENSE, indexConfig: {dimension: 8}},
        // No retries, to skip the backoff.
        {retryConfig: {maxRetries: 0}},
        createIndexIfNotExists = false
    );
}

// --- rule 5: every quantization ratio is one the deployment can store --------------------------

@test:Config
isolated function testEveryCompressionLevelConstructsOnNextGen() {
    CompressionLevel[] levels =
        [COMPRESSION_1X, COMPRESSION_2X, COMPRESSION_8X, COMPRESSION_16X, COMPRESSION_32X];
    foreach CompressionLevel level in levels {
        ai:Error? result = validateConfiguration(VALID_URL, nextGenDeployment(level), validMode(), {});
        test:assertTrue(result is (),
                string `'${level}' is offered by the enum and so must construct without error`);
    }
}

@test:Config
isolated function testUnsetQuantizationIsAllowedOnNextGen() {
    ai:Error? result = validateConfiguration(VALID_URL, nextGenDeployment(), validMode(), {});
    test:assertTrue(result is (), "leaving it unset reproduces the server's own default");
}

// --- rule 6: the NextGen collection is named by name or by id ----------------------------------

@test:Config
isolated function testEitherCollectionIdentifierIsAccepted() {
    ServerlessNextGenDeployment byName = {
        generation: NEXTGEN,
        collection: {name: "vectors"}
    };
    ServerlessNextGenDeployment byId = {
        generation: NEXTGEN,
        collection: {id: "abc123"}
    };
    test:assertTrue(validateConfiguration(VALID_URL, byName, validMode(), {}) is ());
    test:assertTrue(validateConfiguration(VALID_URL, byId, validMode(), {}) is ());
}

// Both records are closed, so a value naming the collection both ways is neither.
@test:Config
isolated function testCollectionCannotBeNamedBothWays() {
    map<string> both = {name: "vectors", id: "abc123"};
    test:assertFalse(both is CollectionName|CollectionId);
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
        generation: NEXTGEN,
        collection: {name: "vectors"}
    };
    ServerlessNextGenDeployment byId = {
        generation: NEXTGEN,
        collection: {id: "abc123"}
    };
    test:assertEquals(buildCollectionHeaders(byName), {"x-amz-aoss-collection-name": "vectors"});
    test:assertEquals(buildCollectionHeaders(byId), {"x-amz-aoss-collection-id": "abc123"});
}

// --- rule 6: document field names must be distinct ---------------------------------------------

// A duplicate name breaks the index silently: here the vector would overwrite the chunk text.
@test:Config
isolated function testVectorFieldNameCollidingWithContentRejected() {
    ai:Error? result = validateConfiguration(VALID_URL, managedDeployment(),
            {queryMode: ai:DENSE, indexConfig: {dimension: 8}, vectorFieldName: "content"}, {});
    if result !is ai:Error {
        test:assertFail("a vector field named 'content' silently clobbers the chunk text");
    }
    test:assertTrue(result.message().includes("more than once"));
}

// HYBRID's two vector fields must also differ from each other.
@test:Config
isolated function testHybridVectorFieldNamesMustDifferFromEachOther() {
    ai:Error? result = validateConfiguration(VALID_URL, managedDeployment(), {
        queryMode: ai:HYBRID,
        indexConfig: {dimension: 8},
        vectorFieldName: "vec",
        sparseVectorFieldName: "vec"
    }, {});
    if result !is ai:Error {
        test:assertFail("the dense and sparse vector fields must be distinct");
    }
    test:assertTrue(result.message().includes("'vec'"));
}

@test:Config
isolated function testSparseFieldNameCollidingWithIdRejected() {
    ai:Error? result = validateConfiguration(VALID_URL, managedDeployment(),
            {queryMode: ai:SPARSE, sparseVectorFieldName: "doc_id"}, {});
    test:assertTrue(result is ai:Error, "the sparse field must not collide with the id field");
}

@test:Config
isolated function testFieldNameCollidingWithChunkTypeRejected() {
    ai:Error? result = validateConfiguration(VALID_URL, managedDeployment(),
            {queryMode: ai:SPARSE, sparseVectorFieldName: "chunk_type"}, {});
    test:assertTrue(result is ai:Error, "the fixed chunk-type field is reserved too");
}

@test:Config
isolated function testMetadataFieldNameCollidingWithContentRejected() {
    ai:Error? result = validateConfiguration(VALID_URL, managedDeployment(), validMode(),
            {metadataFieldName: "content"});
    test:assertTrue(result is ai:Error);
}

// A blank `metadataFieldName` selects a flat schema, which has no metadata field to collide.
@test:Config
isolated function testBlankMetadataFieldNameIsStillAllowed() {
    ai:Error? result = validateConfiguration(VALID_URL, managedDeployment(), validMode(),
            {metadataFieldName: ""});
    test:assertTrue(result is (), "a flat schema is a supported configuration");
}

@test:Config
isolated function testBlankVectorFieldNameRejected() {
    ai:Error? result = validateConfiguration(VALID_URL, managedDeployment(),
            {queryMode: ai:DENSE, indexConfig: {dimension: 8}, vectorFieldName: "  "}, {});
    test:assertTrue(result is ai:Error, "only metadataFieldName may be blank");
}

@test:Config
isolated function testDistinctFieldNamesPass() {
    ai:Error? result = validateConfiguration(VALID_URL, managedDeployment(), {
        queryMode: ai:HYBRID,
        indexConfig: {dimension: 8},
        vectorFieldName: "vec",
        sparseVectorFieldName: "tokens"
    }, {contentFieldName: "body", idFieldName: "entry_id", metadataFieldName: "props"});
    test:assertTrue(result is ());
}

// Validation runs before any request, so neither URL is contacted.
@test:Config
isolated function testBasicAuthNeedsHttps() {
    BasicAuth basicAuth = {username: "u", password: "p"};
    ManagedDomainVectorStore|ai:Error result = new ("http://my-domain.us-east-1.es.amazonaws.com", "us-east-1",
        "idx", basicAuth, false
    );
    if result !is ai:Error {
        test:assertFail("'BasicAuth' over 'http://' should fail construction");
    }
    test:assertTrue(result.message().includes("https://"), result.message());
}

@test:Config
isolated function testSigV4IsAllowedOverHttp() {
    OpenSearchTransport|ai:Error transport = new (UNREACHABLE_URL, "us-east-1", TEST_CREDENTIALS, managedDeployment(),
        {}, {}
    );
    test:assertTrue(transport is OpenSearchTransport, "SigV4 never sends the secret key, so 'http://' stays allowed");
}

// OpenSearch maps a dotted name as nested objects, which the next `init` could not find.
@test:Config
isolated function testDottedVectorFieldNamesAreRejected() {
    DenseSearch dense = {queryMode: ai:DENSE, indexConfig: {dimension: 8}, vectorFieldName: "vectors.dense"};
    ai:Error? denseResult = validateConfiguration(VALID_URL, managedDeployment(), dense, {});
    if denseResult !is ai:Error {
        test:assertFail("a dotted 'vectorFieldName' should be rejected");
    }
    test:assertTrue(denseResult.message().includes("vectors.dense"), denseResult.message());

    SparseSearch sparse = {queryMode: ai:SPARSE, sparseVectorFieldName: "vectors.sparse"};
    test:assertTrue(validateConfiguration(VALID_URL, managedDeployment(), sparse, {}) is ai:Error,
            "a dotted 'sparseVectorFieldName' should be rejected");
}

// Every ordered comparison with `NaN` is false, so it must fail the checks rather than slip past.
@test:Config
isolated function testNaNHybridWeightIsRejected() {
    HybridSearch mode = {
        queryMode: ai:HYBRID,
        indexConfig: {dimension: 8},
        fusion: {denseWeight: float:NaN, sparseWeight: 0.3}
    };
    test:assertTrue(validateConfiguration(VALID_URL, managedDeployment(), mode, {}) is ai:Error);
}

@test:Config
isolated function testPruneRatioMustBeBelowOne() {
    foreach float pruneRatio in [float:NaN, 1.0, -0.1] {
        SparseSearch mode = {queryMode: ai:SPARSE, twoPhaseAcceleration: {pruneRatio}};
        test:assertTrue(validateConfiguration(VALID_URL, managedDeployment(), mode, {}) is ai:Error,
                string `pruneRatio ${pruneRatio} should be rejected`);
    }
    foreach float pruneRatio in [0.0, 0.4, 0.99] {
        SparseSearch mode = {queryMode: ai:SPARSE, twoPhaseAcceleration: {pruneRatio}};
        test:assertTrue(validateConfiguration(VALID_URL, managedDeployment(), mode, {}) is (),
                string `pruneRatio ${pruneRatio} should be accepted`);
    }
}

// A domain in Elasticsearch compatibility mode reports 7.10.2 whatever it really runs.
@test:Config
isolated function testCompatibilityModeVersionIsNotTrusted() {
    test:assertEquals(openSearchVersion("7.10.2"), ());
    test:assertEquals(openSearchVersion("2.19.1"), [2, 19, 1]);
    test:assertEquals(openSearchVersion("1.3.0"), [1, 3, 0]);
    test:assertEquals(openSearchVersion("3.0.0"), [3, 0, 0]);
}
