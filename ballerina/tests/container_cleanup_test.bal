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

import ballerina/log;
import ballerina/test;

// Failures are only logged, so a teardown error cannot mask a test result.
# Drops every index the `docker` group created, for local runs that keep the containers up.
@test:AfterSuite {alwaysRun: true}
function cleanUpContainerIndexes() {
    foreach string indexName in trackedIndexNames() {
        error? result = rawDeleteIndex(indexName);
        if result is error {
            log:printWarn("Failed to delete a test index", indexName = indexName, 'error = result);
        }
        error? secureResult = rawDeleteIndexOn(secureContainerUrl, indexName);
        if secureResult is error {
            log:printWarn("Failed to delete a test index on the secure container",
                    indexName = indexName, 'error = secureResult);
        }
    }
}
