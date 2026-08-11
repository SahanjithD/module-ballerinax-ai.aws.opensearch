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
import ballerina/time;

@test:Config
isolated function testEqualFilter() returns error? {
    ai:MetadataFilters filters = {filters: [{key: "author", operator: ai:EQUAL, value: "Jane Doe"}]};
    json? result = check convertFilters(filters, "metadata");
    json expected = {"bool": {"filter": [{"term": {"metadata.author": "Jane Doe"}}]}};
    test:assertEquals(result, expected);
}

@test:Config
isolated function testNotEqualFilter() returns error? {
    ai:MetadataFilters filters = {filters: [{key: "author", operator: ai:NOT_EQUAL, value: "Jane Doe"}]};
    json? result = check convertFilters(filters, "metadata");
    json expected = {
        "bool": {
            "filter": [
                {"bool": {"must_not": [{"term": {"metadata.author": "Jane Doe"}}]}}
            ]
        }
    };
    test:assertEquals(result, expected);
}

@test:Config
isolated function testRangeOperators() returns error? {
    ai:MetadataFilterOperator[] operators = [
        ai:GREATER_THAN,
        ai:GREATER_THAN_OR_EQUAL,
        ai:LESS_THAN,
        ai:LESS_THAN_OR_EQUAL
    ];
    string[] expectedKeys = ["gt", "gte", "lt", "lte"];
    foreach int i in 0 ..< operators.length() {
        ai:MetadataFilters filters = {filters: [{key: "fileSize", operator: operators[i], value: 100}]};
        json? result = check convertFilters(filters, "metadata");
        json expected = {"bool": {"filter": [{"range": {"metadata.fileSize": {[expectedKeys[i]]: 100}}}]}};
        test:assertEquals(result, expected, string `operator ${operators[i]} produced an unexpected clause`);
    }
}

@test:Config
isolated function testInFilter() returns error? {
    ai:MetadataFilters filters = {
        filters: [{key: "language", operator: ai:IN, value: ["en", "fr"]}]
    };
    json? result = check convertFilters(filters, "metadata");
    json expected = {"bool": {"filter": [{"terms": {"metadata.language": ["en", "fr"]}}]}};
    test:assertEquals(result, expected);
}

@test:Config
isolated function testNotInFilter() returns error? {
    ai:MetadataFilters filters = {
        filters: [{key: "language", operator: ai:NOT_IN, value: ["en", "fr"]}]
    };
    json? result = check convertFilters(filters, "metadata");
    json expected = {
        "bool": {
            "filter": [
                {"bool": {"must_not": [{"terms": {"metadata.language": ["en", "fr"]}}]}}
            ]
        }
    };
    test:assertEquals(result, expected);
}

@test:Config
isolated function testInFilterWithNonArrayValueIsError() {
    ai:MetadataFilters filters = {filters: [{key: "language", operator: ai:IN, value: "en"}]};
    json?|ai:Error result = convertFilters(filters, "metadata");
    test:assertTrue(result is ai:Error, "expected an ai:Error for a non-array IN value");
}

@test:Config
isolated function testNotInFilterWithNonArrayValueIsError() {
    ai:MetadataFilters filters = {filters: [{key: "language", operator: ai:NOT_IN, value: "en"}]};
    json?|ai:Error result = convertFilters(filters, "metadata");
    test:assertTrue(result is ai:Error, "expected an ai:Error for a non-array NOT_IN value");
}

@test:Config
isolated function testNestedAndCondition() returns error? {
    ai:MetadataFilters filters = {
        condition: ai:AND,
        filters: [
            {key: "author", operator: ai:EQUAL, value: "Jane Doe"},
            {key: "language", operator: ai:EQUAL, value: "en"}
        ]
    };
    json? result = check convertFilters(filters, "metadata");
    json expected = {
        "bool": {
            "filter": [
                {"term": {"metadata.author": "Jane Doe"}},
                {"term": {"metadata.language": "en"}}
            ]
        }
    };
    test:assertEquals(result, expected);
}

@test:Config
isolated function testNestedOrCondition() returns error? {
    ai:MetadataFilters filters = {
        condition: ai:OR,
        filters: [
            {key: "author", operator: ai:EQUAL, value: "Jane Doe"},
            {key: "author", operator: ai:EQUAL, value: "John Smith"}
        ]
    };
    json? result = check convertFilters(filters, "metadata");
    json expected = {
        "bool": {
            "should": [
                {"term": {"metadata.author": "Jane Doe"}},
                {"term": {"metadata.author": "John Smith"}}
            ],
            "minimum_should_match": 1
        }
    };
    test:assertEquals(result, expected);
}

@test:Config
isolated function testDeeplyNestedFilters() returns error? {
    ai:MetadataFilters filters = {
        condition: ai:AND,
        filters: [
            {key: "language", operator: ai:EQUAL, value: "en"},
            {
                condition: ai:OR,
                filters: [
                    {key: "author", operator: ai:EQUAL, value: "Jane Doe"},
                    {key: "author", operator: ai:EQUAL, value: "John Smith"}
                ]
            }
        ]
    };
    json? result = check convertFilters(filters, "metadata");
    json expected = {
        "bool": {
            "filter": [
                {"term": {"metadata.language": "en"}},
                {
                    "bool": {
                        "should": [
                            {"term": {"metadata.author": "Jane Doe"}},
                            {"term": {"metadata.author": "John Smith"}}
                        ],
                        "minimum_should_match": 1
                    }
                }
            ]
        }
    };
    test:assertEquals(result, expected);
}

@test:Config
isolated function testEmptyFilterListReturnsNil() returns error? {
    ai:MetadataFilters filters = {filters: []};
    json? result = check convertFilters(filters, "metadata");
    test:assertEquals(result, ());
}

@test:Config
isolated function testEmptyNestedGroupIsDropped() returns error? {
    // A nested group that itself resolves to no clauses must not leave an empty {"bool":{}} behind.
    ai:MetadataFilters filters = {
        filters: [
            {key: "language", operator: ai:EQUAL, value: "en"},
            {filters: []}
        ]
    };
    json? result = check convertFilters(filters, "metadata");
    json expected = {"bool": {"filter": [{"term": {"metadata.language": "en"}}]}};
    test:assertEquals(result, expected);
}

@test:Config
isolated function testCreatedAtTimeUtcCoercion() returns error? {
    time:Utc createdAt = [1758793007, 0.798845];
    ai:MetadataFilters filters = {filters: [{key: "createdAt", operator: ai:GREATER_THAN, value: createdAt}]};
    json? result = check convertFilters(filters, "metadata");
    json expected = {
        "bool": {
            "filter": [
                {"range": {"metadata.createdAt": {"gt": "2025-09-25T09:36:47.798845Z"}}}
            ]
        }
    };
    test:assertEquals(result, expected);
}

@test:Config
isolated function testModifiedAtTimeUtcCoercion() returns error? {
    time:Utc modifiedAt = [1758793007, 0.798845];
    ai:MetadataFilters filters = {filters: [{key: "modifiedAt", operator: ai:EQUAL, value: modifiedAt}]};
    json? result = check convertFilters(filters, "metadata");
    json expected = {"bool": {"filter": [{"term": {"metadata.modifiedAt": "2025-09-25T09:36:47.798845Z"}}]}};
    test:assertEquals(result, expected);
}

@test:Config
isolated function testFlatMetadataFieldNameProducesBareKeyPaths() returns error? {
    ai:MetadataFilters filters = {filters: [{key: "author", operator: ai:EQUAL, value: "Jane Doe"}]};
    json? result = check convertFilters(filters, "");
    json expected = {"bool": {"filter": [{"term": {"author": "Jane Doe"}}]}};
    test:assertEquals(result, expected);
}

@test:Config
isolated function testFieldPathHelper() {
    test:assertEquals(fieldPath("metadata", "author"), "metadata.author");
    test:assertEquals(fieldPath("", "author"), "author");
}
