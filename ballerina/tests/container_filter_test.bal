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

// `convertFilters` against the real query DSL. This is where a container earns the most: a unit
// test can only confirm that a `term` clause was emitted, and a `term` clause on a field the
// dynamic template mapped as analyzed `text` is emitted perfectly and matches nothing. Every
// operator here is asserted by the documents it actually selects.

import ballerina/ai;
import ballerina/test;

# Seeds the shared filter dataset. Every filter test creates its own index and seeds it, so the
# tests stay order-independent and can run in any subset.
#
# + store - The store to seed
# + return - An error if seeding fails
isolated function seedFilterEntries(VectorStore store) returns error? {
    check store.add([
        entry("en-2020", vec(1.0), "english 2020",
                {"language": "en", "year": 2020, "rating": 4.5, "published": true, "tags": ["a", "b"]}),
        entry("en-2024", vec(0.9, 0.1), "english 2024",
                {"language": "en", "year": 2024, "rating": 3.0, "published": false, "tags": ["b", "c"]}),
        entry("fr-2022", vec(0.0, 1.0), "french 2022",
                {"language": "fr", "year": 2022, "rating": 5.0, "published": true, "tags": ["a"]}),
        entry("de-2024", vec(0.0, 0.0, 1.0), "german 2024",
                {"language": "de", "year": 2024, "rating": 2.0, "published": false, "tags": ["c"]})
    ]);
}

# Seeds a fresh index and runs a filters-only query against it.
#
# + prefix - The index-name tag for this test
# + filters - The filters to apply
# + return - The matching entries, or an error
isolated function queryWithFilters(string prefix, ai:MetadataFilters filters) returns ai:VectorMatch[]|error {
    string indexName = containerIndexName(prefix);
    VectorStore store = check newContainerStore(indexName);
    check seedFilterEntries(store);
    ai:VectorMatch[] matches = check store.query({filters, topK: 10});
    check store.close();
    return matches;
}

@test:Config {groups: ["docker"]}
isolated function testContainerFilterEqualOnString() returns error? {
    ai:VectorMatch[] matches = check queryWithFilters("f-eq",
            {filters: [{key: "language", operator: ai:EQUAL, value: "en"}]});
    // A `term` clause only matches here because the dynamic template mapped `metadata.language`
    // as `keyword`; under the default `text` mapping this would silently return nothing.
    assertIdsEqual(matches, ["en-2020", "en-2024"], "EQUAL should select both English entries");
}

@test:Config {groups: ["docker"]}
isolated function testContainerFilterNotEqualOnString() returns error? {
    ai:VectorMatch[] matches = check queryWithFilters("f-neq",
            {filters: [{key: "language", operator: ai:NOT_EQUAL, value: "en"}]});
    assertIdsEqual(matches, ["fr-2022", "de-2024"], "NOT_EQUAL should exclude both English entries");
}

@test:Config {groups: ["docker"]}
isolated function testContainerFilterGreaterThan() returns error? {
    ai:VectorMatch[] matches = check queryWithFilters("f-gt",
            {filters: [{key: "year", operator: ai:GREATER_THAN, value: 2022}]});
    assertIdsEqual(matches, ["en-2024", "de-2024"], "GREATER_THAN should be exclusive of the bound");
}

@test:Config {groups: ["docker"]}
isolated function testContainerFilterGreaterThanOrEqual() returns error? {
    ai:VectorMatch[] matches = check queryWithFilters("f-gte",
            {filters: [{key: "year", operator: ai:GREATER_THAN_OR_EQUAL, value: 2022}]});
    assertIdsEqual(matches, ["en-2024", "fr-2022", "de-2024"],
            "GREATER_THAN_OR_EQUAL should include the bound");
}

@test:Config {groups: ["docker"]}
isolated function testContainerFilterLessThan() returns error? {
    ai:VectorMatch[] matches = check queryWithFilters("f-lt",
            {filters: [{key: "year", operator: ai:LESS_THAN, value: 2022}]});
    assertIdsEqual(matches, ["en-2020"], "LESS_THAN should be exclusive of the bound");
}

@test:Config {groups: ["docker"]}
isolated function testContainerFilterLessThanOrEqual() returns error? {
    ai:VectorMatch[] matches = check queryWithFilters("f-lte",
            {filters: [{key: "year", operator: ai:LESS_THAN_OR_EQUAL, value: 2022}]});
    assertIdsEqual(matches, ["en-2020", "fr-2022"], "LESS_THAN_OR_EQUAL should include the bound");
}

@test:Config {groups: ["docker"]}
isolated function testContainerFilterIn() returns error? {
    ai:VectorMatch[] matches = check queryWithFilters("f-in",
            {filters: [{key: "language", operator: ai:IN, value: ["en", "de"]}]});
    assertIdsEqual(matches, ["en-2020", "en-2024", "de-2024"], "IN should select every listed value");
}

@test:Config {groups: ["docker"]}
isolated function testContainerFilterNotIn() returns error? {
    ai:VectorMatch[] matches = check queryWithFilters("f-nin",
            {filters: [{key: "language", operator: ai:NOT_IN, value: ["en", "de"]}]});
    assertIdsEqual(matches, ["fr-2022"], "NOT_IN should exclude every listed value");
}

@test:Config {groups: ["docker"]}
isolated function testContainerFilterOnFloatValue() returns error? {
    ai:VectorMatch[] matches = check queryWithFilters("f-float",
            {filters: [{key: "rating", operator: ai:GREATER_THAN_OR_EQUAL, value: 4.0}]});
    assertIdsEqual(matches, ["en-2020", "fr-2022"], "a float range bound should compare numerically");
}

@test:Config {groups: ["docker"]}
isolated function testContainerFilterOnBooleanValue() returns error? {
    ai:VectorMatch[] matches = check queryWithFilters("f-bool",
            {filters: [{key: "published", operator: ai:EQUAL, value: true}]});
    assertIdsEqual(matches, ["en-2020", "fr-2022"], "a boolean term should match the boolean mapping");
}

@test:Config {groups: ["docker"]}
isolated function testContainerFilterOnArrayValuedMetadata() returns error? {
    ai:VectorMatch[] matches = check queryWithFilters("f-array",
            {filters: [{key: "tags", operator: ai:EQUAL, value: "a"}]});
    // A `term` against a multi-valued keyword field matches if any element matches.
    assertIdsEqual(matches, ["en-2020", "fr-2022"], "a term should match any element of an array field");
}

@test:Config {groups: ["docker"]}
isolated function testContainerFilterAndCombinesConditions() returns error? {
    ai:VectorMatch[] matches = check queryWithFilters("f-and", {
                                                                   condition: ai:AND,
                                                                   filters: [
                                                                       {key: "language", operator: ai:EQUAL, value: "en"},
                                                                       {key: "year", operator: ai:EQUAL, value: 2024}
                                                                   ]
                                                               });
    assertIdsEqual(matches, ["en-2024"], "AND should require every clause to match");
}

@test:Config {groups: ["docker"]}
isolated function testContainerFilterOrCombinesConditions() returns error? {
    ai:VectorMatch[] matches = check queryWithFilters("f-or", {
                                                                  condition: ai:OR,
                                                                  filters: [
                                                                      {key: "language", operator: ai:EQUAL, value: "fr"},
                                                                      {key: "year", operator: ai:EQUAL, value: 2020}
                                                                  ]
                                                              });
    // `bool.should` without `minimum_should_match: 1` would match everything; this is what proves
    // that clause is present and doing its job.
    assertIdsEqual(matches, ["fr-2022", "en-2020"], "OR should require at least one clause to match");
}

@test:Config {groups: ["docker"]}
isolated function testContainerFilterNestedGroups() returns error? {
    ai:VectorMatch[] matches = check queryWithFilters("f-nested", {
                                                                      condition: ai:AND,
                                                                      filters: [
                                                                          {key: "year", operator: ai:EQUAL, value: 2024},
                                                                          {
                                                                              condition: ai:OR,
                                                                              filters: [
                                                                                  {key: "language", operator: ai:EQUAL, value: "de"},
                                                                                  {key: "rating", operator: ai:GREATER_THAN, value: 4.0}
                                                                              ]
                                                                          }
                                                                      ]
                                                                  });
    assertIdsEqual(matches, ["de-2024"], "a nested OR inside an AND should be evaluated as written");
}

@test:Config {groups: ["docker"]}
isolated function testContainerEmptyFilterListMatchesEverything() returns error? {
    ai:VectorMatch[] matches = check queryWithFilters("f-emptylist", {filters: []});
    // An empty group contributes no clause at all, rather than an unsatisfiable `bool.filter: []`.
    assertIdsEqual(matches, ["en-2020", "en-2024", "fr-2022", "de-2024"],
            "an empty filter list should not restrict the result set");
}

@test:Config {groups: ["docker"]}
isolated function testContainerFilterPreFiltersKnnQuery() returns error? {
    string indexName = containerIndexName("f-knn");
    VectorStore store = check newContainerStore(indexName);
    check seedFilterEntries(store);

    // `fr-2022` is the *least* similar entry to `queryVector()`, so a `topK: 1` k-NN search that
    // returns it can only have filtered before ranking. A `post_filter` would have ranked first,
    // taken `en-2020`, then filtered it away and returned nothing.
    ai:VectorMatch[] matches = check store.query({
        embedding: queryVector(),
        topK: 1,
        filters: {filters: [{key: "language", operator: ai:EQUAL, value: "fr"}]}
    });
    test:assertEquals(matches.length(), 1, "pre-filtering should still yield a full 'k' results");
    test:assertEquals(matches[0].id, "fr-2022");
    check store.close();
}

@test:Config {groups: ["docker"]}
isolated function testContainerFilterUnderFlatSchema() returns error? {
    string indexName = containerIndexName("f-flat");
    Configuration config = {
        indexConfig: {dimension: CONTAINER_DIMENSION},
        metadataFieldName: ""
    };
    VectorStore store = check newContainerStore(indexName, config);
    check store.add([
        entry("flat-en", vec(1.0), "english", {"language": "en"}),
        entry("flat-fr", vec(0.0, 1.0), "french", {"language": "fr"})
    ]);

    // With `metadataFieldName: ""` the filter path is a bare `language`, not `metadata.language`,
    // and metadata is spread across the top level of the document rather than nested.
    ai:VectorMatch[] matches = check store.query({
        filters: {filters: [{key: "language", operator: ai:EQUAL, value: "en"}]},
        topK: 10
    });
    assertIdsEqual(matches, ["flat-en"], "a flat schema should filter on bare field paths");
    test:assertEquals(matches[0].chunk.metadata, <ai:Metadata>{"language": "en"},
            "flat metadata should be reassembled on read");
    check store.close();
}

@test:Config {groups: ["docker"]}
isolated function testContainerFilterOnUnknownKeyMatchesNothing() returns error? {
    ai:VectorMatch[] matches = check queryWithFilters("f-unknown",
            {filters: [{key: "nosuchfield", operator: ai:EQUAL, value: "x"}]});
    test:assertEquals(matches.length(), 0,
            "filtering on a field no document has should return nothing, not error");
}
