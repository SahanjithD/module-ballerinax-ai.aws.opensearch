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

// Conversion between `ai:SparseVector` and OpenSearch's `{"<token>": <weight>}` map. The bounds are
// Lucene's, checked here so a violation names the entry instead of failing the whole request.

// Lucene `FeatureField` rejects anything below `Float.MIN_NORMAL`, including zero and negatives.
const float MIN_RANK_FEATURE_VALUE = 1.17549435e-38;

// Lucene `FeatureField.newLinearQuery` caps a query weight at 64; stored weights have no ceiling.
const float MAX_QUERY_TOKEN_WEIGHT = 64.0;

# One term of a sparse vector, used to sort a stored token map back into index order.
type SparseToken record {|
    # The feature index, parsed from the stored token name.
    int index;
    # The feature weight.
    float weight;
|};

// Out-of-range weights are rejected rather than clamped, which would silently change the ranking.
# Converts an `ai:SparseVector` to OpenSearch's token map, checking Lucene's constraints.
#
# + sparse - The sparse vector to convert
# + subject - What to name in an error message, e.g. `Entry 'abc'` or `The query`
# + maxWeight - Inclusive weight ceiling: `float:Infinity` when indexing, `MAX_QUERY_TOKEN_WEIGHT`
# for a query
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
        // A negative index is a valid feature name but would never match a query.
        if index < 0 {
            return error(string `${subject} has a negative sparse index (${index}); ` +
                    "'ai:SparseVector.indices' must be non-negative");
        }
        // First, since every comparison with NaN is false and it would pass the range checks.
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

// A non-integer token is an error rather than skipped: `indices` is `int[]`, and dropping terms
// would misreport what is stored.
# Rebuilds an `ai:SparseVector` from a stored token map, sorted by index.
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
        // A whole-numbered weight can come back as a JSON integer.
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
