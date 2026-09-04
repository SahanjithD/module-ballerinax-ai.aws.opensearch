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

// `describeErrorDetail` and `specificErrorType` against error bodies captured verbatim from
// OpenSearch 2.19.1 -- the same engine version a managed domain runs.
//
// Every body below was produced by issuing the named request against a real server and pasting
// back what came out, because the nesting is the whole point: where OpenSearch puts the sentence
// that actually explains a failure varies by failure, and a hand-invented body would simply
// encode the guess being tested. Two shapes matter and differ, and both are pinned here:
// an error *response* nests its explanation in `root_cause[0]`, while a `_bulk` item error has no
// `root_cause` at all and nests it in `caused_by`.
//
// These also cover the decode itself. `ErrorDetail` is recursive through `caused_by`, so these
// assert that `cloneWithType` reconstructs the nesting from real JSON rather than flattening it.

import ballerina/test;

# Decodes a captured response body into the `error` object this module reads.
#
# + body - The verbatim response body
# + return - The decoded error detail
isolated function errorDetailOf(json body) returns ErrorDetail|error {
    OpenSearchErrorResponse parsed = check body.cloneWithType(OpenSearchErrorResponse);
    ErrorDetail? detail = parsed?.'error;
    if detail is () {
        return error("the captured body carries no 'error' object");
    }
    return detail;
}

// `POST /<index>/_search` with `{"knn": {"embedding": {"vector": [4 floats], "k": 3}}}` against a
// field of dimension 8. This is the case that reads as `all shards failed` and nothing else: the
// only statement of what the caller did wrong is two levels down.
final readonly & json DIMENSION_MISMATCH_SEARCH_BODY = {
    "error": {
        "root_cause": [
            {
                "type": "query_shard_exception",
                "reason": "failed to create query: Query vector has invalid dimension: 4. Dimension should be: 8",
                "index": "probe-dim",
                "index_uuid": "XOxx0pZMTsuN1vJ6coq-XQ"
            }
        ],
        "type": "search_phase_execution_exception",
        "reason": "all shards failed",
        "phase": "query",
        "grouped": true,
        "failed_shards": [
            {
                "shard": 0,
                "index": "probe-dim",
                "node": "e0UxRAhaSqWCnSo3JQL4sg",
                "reason": {
                    "type": "query_shard_exception",
                    "reason": "failed to create query: Query vector has invalid dimension: 4. Dimension should be: 8",
                    "index": "probe-dim",
                    "index_uuid": "XOxx0pZMTsuN1vJ6coq-XQ",
                    "caused_by": {
                        "type": "illegal_argument_exception",
                        "reason": "Query vector has invalid dimension: 4. Dimension should be: 8"
                    }
                }
            }
        ]
    },
    "status": 400
};

@test:Config
isolated function testDescribeErrorDetailRecoversShardFailureReason() returns error? {
    ErrorDetail detail = check errorDetailOf(DIMENSION_MISMATCH_SEARCH_BODY);
    string described = describeErrorDetail(detail);

    test:assertTrue(described.startsWith("all shards failed"),
            string `the server's own top-level reason should still lead, got: ${described}`);
    test:assertTrue(described.includes("Dimension should be: 8"),
            string `the nested explanation is the only actionable part and must survive, got: ${described}`);
    test:assertTrue(described.includes("query_shard_exception"),
            string `the nested classification should be named, got: ${described}`);
}

@test:Config
isolated function testSpecificErrorTypeUnwrapsSearchPhaseException() returns error? {
    ErrorDetail detail = check errorDetailOf(DIMENSION_MISMATCH_SEARCH_BODY);
    // The wrapper's own type names the phase that failed, which is not something a caller can
    // branch on; `openSearchErrorType` keeps reporting it, and this is the useful one alongside.
    test:assertEquals(detail.'type, "search_phase_execution_exception");
    test:assertEquals(specificErrorType(detail), "query_shard_exception");
}

// `GET /no-such-index/_search`. A non-wrapper error repeats itself in `root_cause[0]`, so there is
// nothing to append and the message must not say the same thing twice.
@test:Config
isolated function testDescribeErrorDetailDoesNotDuplicateSelfReferentialRootCause() returns error? {
    ErrorDetail detail = check errorDetailOf({
                                                 "error": {
                                                     "root_cause": [
                                                         {
                                                             "type": "index_not_found_exception",
                                                             "reason": "no such index [no-such-index]",
                                                             "index": "no-such-index",
                                                             "resource.id": "no-such-index",
                                                             "resource.type": "index_or_alias",
                                                             "index_uuid": "_na_"
                                                         }
                                                     ],
                                                     "type": "index_not_found_exception",
                                                     "reason": "no such index [no-such-index]",
                                                     "index": "no-such-index",
                                                     "resource.id": "no-such-index",
                                                     "resource.type": "index_or_alias",
                                                     "index_uuid": "_na_"
                                                 },
                                                 "status": 404
                                             });

    test:assertEquals(describeErrorDetail(detail), "no such index [no-such-index]");
    test:assertEquals(specificErrorType(detail), (),
            "a self-referential root cause adds no classification worth carrying");
}

// A failed `_bulk` item from `POST /_bulk` indexing a 2-dimensional vector into a dimension-8
// field. Note the absence of `root_cause`: the outer reason is a parse-failure summary whose
// preview of the offending value is the useless string `null`, and `caused_by` holds the numbers.
@test:Config
isolated function testDescribeErrorDetailRecoversBulkItemCause() returns error? {
    BulkResponse response = check {
        "errors": true,
        "items": [
            {
                "index": {
                    "_index": "probe-dim",
                    "_id": "m-SlV6ABeXhjp-wm-InJ",
                    "status": 400,
                    "error": {
                        "type": "mapper_parsing_exception",
                        "reason": "failed to parse field [embedding] of type [knn_vector] in document " +
                            "with id 'm-SlV6ABeXhjp-wm-InJ'. Preview of field's value: 'null'",
                        "caused_by": {
                            "type": "illegal_argument_exception",
                            "reason": "Vector dimension mismatch. Expected: 8, Given: 2"
                        }
                    }
                }
            }
        ]
    }.cloneWithType(BulkResponse);

    BulkFailure[] failures = extractIndexFailures(response, ["bad-1"]);
    test:assertEquals(failures.length(), 1);
    test:assertEquals(failures[0].id, "bad-1");
    test:assertTrue(failures[0].reason.includes("Expected: 8, Given: 2"),
            string `the item's 'caused_by' carries the only usable explanation, got: ${failures[0].reason}`);
}

// `caused_by` chains nest more than one level; the deepest link is the specific one.
@test:Config
isolated function testDescribeErrorDetailWalksToTheDeepestCause() {
    ErrorDetail detail = {
        'type: "wrapper",
        reason: "outermost",
        caused_by: {
            'type: "middle",
            reason: "intermediate",
            caused_by: {'type: "innermost", reason: "the actual problem"}
        }
    };
    test:assertEquals(describeErrorDetail(detail), "outermost: [innermost] the actual problem");
    test:assertEquals(specificErrorType(detail), "innermost");
}

@test:Config
isolated function testDescribeErrorDetailFallsBackToTypeAndEmpty() {
    test:assertEquals(describeErrorDetail({'type: "some_exception"}), "some_exception",
            "an error object with no reason should fall back to its type");
    test:assertEquals(describeErrorDetail({}), "",
            "an empty error object describes nothing, leaving the fallback to the caller");
}
