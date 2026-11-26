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

// Stand-ins that report a cluster version, for the NMSLIB cosine check. `GET /` is the only
// request the check makes, and only for an NMSLIB field.

import ballerina/ai;
import ballerina/http;
import ballerina/test;

# The port of the stand-in that reports a pre-2.19 cluster.
const int OLD_CLUSTER_PORT = 18302;
# The port of the stand-in that reports a 2.19 cluster.
const int CURRENT_CLUSTER_PORT = 18303;

listener http:Listener oldClusterListener = new (OLD_CLUSTER_PORT);
listener http:Listener currentClusterListener = new (CURRENT_CLUSTER_PORT);

service / on oldClusterListener {
    isolated resource function get .() returns json => {"version": {"number": "2.18.0"}};
}

service / on currentClusterListener {
    isolated resource function get .() returns json => {"version": {"number": "2.19.1"}};
}

isolated function transportTo(int port) returns OpenSearchTransport|ai:Error =>
    new (string `http://localhost:${port}`, "us-east-1", TEST_CREDENTIALS, managedDeployment(), {}, {});

isolated function knnField(string engine, string spaceType) returns map<json> => {
    "type": "knn_vector",
    "dimension": 4,
    "method": {"name": "hnsw", "engine": engine, "space_type": spaceType}
};

@test:Config
isolated function testSpaceTypeMustMatchTheMetric() returns error? {
    // A mismatch is decided from the mapping alone, so the transport is never used.
    OpenSearchTransport transport = check new (UNREACHABLE_URL, "us-east-1", TEST_CREDENTIALS, managedDeployment(),
        {}, {});
    ai:Error? result = verifyVectorSpace(transport, knnField("faiss", "l2"), "idx", "embedding", ai:COSINE, true);
    if result !is ai:Error {
        test:assertFail("an 'l2' index must not be read as cosine");
    }
    test:assertTrue(result.message().includes("'l2'") && result.message().includes("cosinesimil"),
            result.message());
}

@test:Config
isolated function testTopLevelSpaceTypeIsChecked() returns error? {
    OpenSearchTransport transport = check new (UNREACHABLE_URL, "us-east-1", TEST_CREDENTIALS, managedDeployment(),
        {}, {});
    map<json> declared = {"type": "knn_vector", "dimension": 4, "space_type": "innerproduct"};
    test:assertTrue(verifyVectorSpace(transport, declared, "idx", "embedding", ai:COSINE, true) is ai:Error);
}

@test:Config
isolated function testMatchingSpaceTypePasses() returns error? {
    OpenSearchTransport transport = check new (UNREACHABLE_URL, "us-east-1", TEST_CREDENTIALS, managedDeployment(),
        {}, {});
    test:assertEquals(verifyVectorSpace(transport, knnField("faiss", "cosinesimil"), "idx", "embedding", ai:COSINE,
            true), ());
    test:assertEquals(verifyVectorSpace(transport, knnField("lucene", "l2"), "idx", "embedding", ai:EUCLIDEAN,
            true), ());
}

@test:Config
isolated function testNmslibCosineBefore219CannotBeNormalized() returns error? {
    OpenSearchTransport transport = check transportTo(OLD_CLUSTER_PORT);
    ai:Error? result = verifyVectorSpace(transport, knnField("nmslib", "cosinesimil"), "idx", "embedding",
            ai:COSINE, true);
    if result !is ai:Error {
        test:assertFail("NMSLIB cosine on 2.18 scores 1 / (2 - cos), which 'normalizeCosineScore' cannot invert");
    }
    test:assertTrue(result.message().includes("normalizeCosineScore"), result.message());
}

@test:Config
isolated function testNmslibCosineBefore219PassesWithoutNormalization() returns error? {
    OpenSearchTransport transport = check transportTo(OLD_CLUSTER_PORT);
    test:assertEquals(verifyVectorSpace(transport, knnField("nmslib", "cosinesimil"), "idx", "embedding",
            ai:COSINE, false), (), "raw scores are returned as is, so nothing needs inverting");
}

@test:Config
isolated function testNmslibCosineFrom219CanBeNormalized() returns error? {
    OpenSearchTransport transport = check transportTo(CURRENT_CLUSTER_PORT);
    test:assertEquals(verifyVectorSpace(transport, knnField("nmslib", "cosinesimil"), "idx", "embedding",
            ai:COSINE, true), (), "from 2.19 NMSLIB scores cosine as (1 + cos) / 2 like every engine");
}

// Without a `method` the space is set elsewhere; it is warned about, not guessed.
@test:Config
isolated function testAbsentSpaceTypeIsNotGuessed() returns error? {
    OpenSearchTransport transport = check new (UNREACHABLE_URL, "us-east-1", TEST_CREDENTIALS, managedDeployment(),
        {}, {});
    map<json> declared = {"type": "knn_vector", "dimension": 4};
    test:assertEquals(verifyVectorSpace(transport, declared, "idx", "embedding", ai:COSINE, true), (),
            "an index-level or model space may well be cosine, so this must not fail");
}
