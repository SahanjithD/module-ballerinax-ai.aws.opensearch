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

// `Deployment` builders for the offline unit tests. Credentials are a required field on every
// variant -- deliberately, so a store can never quietly fall back to the ambient AWS credential
// chain -- but most of these tests exercise mapping and bulk-body construction, where the
// credentials never leave the record. These builders supply a fixed fake so that requirement does
// not clutter every call site.
//
// `container_commons.bal` has its own builders for the Docker suite, which needs credentials the
// container actually accepts.

import ballerinax/aws.auth;

# Static credentials for tests that never issue a signed request.
final readonly & auth:StaticAuthConfig TEST_CREDENTIALS = {
    accessKeyId: "AKIAFAKEFAKEFAKEFAKE",
    secretAccessKey: "fake-secret"
};

# A managed-domain deployment with test credentials.
#
# + engine - The ANN engine
# + refreshOnWrite - Whether `add` should request `?refresh=wait_for`
# + return - The deployment
isolated function managedDeployment(Engine engine = FAISS, boolean refreshOnWrite = false)
        returns ManagedDomainDeployment =>
    {deploymentType: MANAGED_DOMAIN, auth: TEST_CREDENTIALS, engine, refreshOnWrite};

# A Serverless Classic deployment with test credentials.
#
# + return - The deployment
isolated function classicDeployment() returns ServerlessClassicDeployment =>
    {deploymentType: SERVERLESS_CLASSIC, auth: TEST_CREDENTIALS};

# A Serverless NextGen deployment with test credentials.
#
# + compressionLevel - The vector quantization ratio, unset by default
# + return - The deployment
isolated function nextGenDeployment(CompressionLevel? compressionLevel = ())
        returns ServerlessNextGenDeployment =>
    {deploymentType: SERVERLESS_NEXTGEN, auth: TEST_CREDENTIALS, compressionLevel};

# All three deployment types with their defaults, for tests that assert behavior shared by every
# flavour.
#
# + return - One deployment of each variant
isolated function allDeployments() returns Deployment[] =>
    [managedDeployment(), classicDeployment(), nextGenDeployment()];
