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

// Conversion between `ai:SparseVector` and the `{"<token>": <weight>}` map that OpenSearch uses
// for both a `rank_features` document value and a `neural_sparse` `query_tokens` object. Pure, so
// every constraint below is unit-testable without a server.
//
// The constraints are Lucene's, not this module's. They are enforced here so a violation names
// the offending entry rather than failing an entire `_bulk` request, or arriving at query time as
// an opaque shard exception with no indication of which term caused it.

# The smallest weight Lucene's `FeatureField` will index — Java's `Float.MIN_NORMAL`.
#
# A `rank_features` value below this is not stored as zero; it throws
# `featureValue must be a positive normal float`. Since `0.0` and every negative value are below
# it, this one bound rejects all three cases. A term whose weight is zero carries no information
# anyway and belongs out of the sparse vector entirely.
const float MIN_RANK_FEATURE_VALUE = 1.17549435e-38;

# The largest weight `FeatureField.newLinearQuery` accepts, which is Lucene's `Long.SIZE`.
#
# Applies to `neural_sparse` `query_tokens` only — there is no equivalent ceiling at index time.
# `NeuralSparseQueryBuilder` passes the values straight through with no filtering or clamping, so
# a weight above this fails the query outright.
const float MAX_QUERY_TOKEN_WEIGHT = 64.0;

# One term of a sparse vector, used to sort a stored token map back into index order.
type SparseToken record {|
    # The feature index, parsed from the stored token name.
    int index;
    # The feature weight.
    float weight;
|};

# Converts an `ai:SparseVector` to the `{"<token>": <weight>}` map OpenSearch takes, validating
# every constraint Lucene enforces on the way.
#
# Weights are rejected rather than clamped. Clamping an out-of-range weight would silently rewrite
# the relevance ranking the caller's encoder produced, making results irreproducible and the cause
# invisible; this matches `validateEmbedding`, which rejects a zero dense vector rather than
# substituting an epsilon.
#
# + sparse - The sparse vector to convert
# + subject - What to name in an error message, e.g. `Entry 'abc'` or `The query`
# + maxWeight - The inclusive upper bound on a weight: `float:Inf` at index time, where Lucene
# imposes none, and `MAX_QUERY_TOKEN_WEIGHT` at query time, where it does
# + return - The token map, or an `ai:Error` naming the first violation and the index it is at
isolated function toSparseTokenMap(ai:SparseVector sparse, string subject, float maxWeight)
        returns map<float>|ai:Error {
    int[] indices = sparse.indices;
    float[] values = sparse.values;
    if indices.length() != values.length() {
        return error(string `${subject} has a sparse vector whose 'indices' and 'values' differ in ` +
                string `length (${indices.length()} vs ${values.length()}); every index must have ` +
                "exactly one weight");
    }

    map<float> tokens = {};
    foreach int position in 0 ..< indices.length() {
        int index = indices[position];
        float weight = values[position];
        // A negative index stringifies to a perfectly legal feature name, so without this it
        // would be stored rather than rejected, and would never match a query built from a
        // non-negative index.
        if index < 0 {
            return error(string `${subject} has a negative sparse index (${index}); ` +
                    "'ai:SparseVector.indices' must be non-negative");
        }
        // Checked before any comparison below: every ordering comparison against NaN is false, so
        // a NaN weight would slip past the range checks and reach the server.
        if !weight.isFinite() {
            return error(string `${subject} has a non-finite sparse weight (${weight}) at index ` +
                    string `${index}; a 'rank_features' weight must be a finite number`);
        }
        if weight < MIN_RANK_FEATURE_VALUE {
            return error(string `${subject} has a sparse weight of ${weight} at index ${index}; a ` +
                    string `'rank_features' weight must be at least ${MIN_RANK_FEATURE_VALUE} ` +
                    "(Lucene's 'Float.MIN_NORMAL'), so zero and negative weights cannot be indexed. " +
                    "Drop the term from the sparse vector rather than sending a zero weight");
        }
        if weight > maxWeight {
            return error(string `${subject} has a sparse weight of ${weight} at index ${index}, ` +
                    string `above the maximum of ${maxWeight}. A 'neural_sparse' 'query_tokens' ` +
                    "weight must be in (0, 64]; rescale the query vector before passing it");
        }
        string token = index.toString();
        if tokens.hasKey(token) {
            return error(string `${subject} repeats sparse index ${index}; a 'rank_features' field ` +
                    "rejects two weights for the same feature");
        }
        tokens[token] = weight;
    }
    return tokens;
}

# Reconstructs an `ai:SparseVector` from a stored `rank_features` token map.
#
# The result is sorted by integer index ascending. Map key order is not guaranteed — `_source`
# echoes the document as it was written, which may be a document this module did not write — so
# some deterministic order has to be imposed, and ascending index is the one canonical choice. The
# sort is numeric rather than lexicographic, so `"10"` follows `"2"` rather than preceding it.
#
# A stored weight is not a byte-exact echo of what was written: `rank_features` keeps roughly nine
# significant bits, giving about 0.4% relative error. Never compare a round-tripped weight to a
# locally-held one with exact float equality.
#
# A token that is not an integer is an error rather than a skipped term. `ai:SparseVector.indices`
# is `int[]`, so a text-token sparse field — which a real encoder may well produce, and which this
# module never writes — cannot be represented at all; dropping those terms silently would make the
# returned vector a quiet lie about what is stored. This mirrors how a malformed stored *dense*
# embedding is reported rather than swallowed.
#
# + features - The stored token map
# + id - The entry id, used to name the offending entry in an error message
# + return - The reconstructed sparse vector, or an `ai:Error` if a token is not an integer
isolated function sparseFromRankFeatures(map<json> features, string id) returns ai:SparseVector|ai:Error {
    SparseToken[] tokens = [];
    foreach [string, json] [token, rawWeight] in features.entries() {
        int|error index = int:fromString(token);
        if index is error {
            return error(string `Failed to parse the stored sparse embedding for entry '${id}': the ` +
                    string `token '${token}' is not an integer index. 'ai:SparseVector.indices' is ` +
                    "'int[]', so a text-token sparse field written outside this module cannot be " +
                    "represented; set 'Configuration.includeEmbeddingsInResults' to false to skip " +
                    "reading it");
        }
        // A whole-numbered weight can come back as a JSON integer rather than a float, so all
        // three numeric types are accepted here.
        float weight;
        if rawWeight is float {
            weight = rawWeight;
        } else if rawWeight is int {
            weight = <float>rawWeight;
        } else if rawWeight is decimal {
            weight = <float>rawWeight;
        } else {
            return error(string `Failed to parse the stored sparse embedding for entry '${id}': the ` +
                    string `token '${token}' has a non-numeric weight ` +
                    string `('${rawWeight.toJsonString()}')`);
        }
        tokens.push({index, weight});
    }

    int[] indices = [];
    float[] values = [];
    foreach SparseToken token in from SparseToken candidate in tokens
            order by candidate.index ascending
            select candidate {
        indices.push(token.index);
        values.push(token.weight);
    }
    return {indices, values};
}
