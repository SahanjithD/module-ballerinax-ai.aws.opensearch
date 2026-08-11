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
import ballerinax/aws.auth;

# The deployment flavour of the target OpenSearch cluster. Determines the SigV4 signing name,
# whether a custom document `_id` can be written, how `delete` is implemented, and whether the
# index mapping needs an explicit `method` block.
#
# OpenSearch Serverless now ships two generations that behave differently on the data plane, and
# there is no documented way to detect which one a collection is from the data plane alone — the
# generation must be declared by the operator.
public enum DeploymentType {
    # An OpenSearch Service (managed) domain. SigV4 signing name `es`, or HTTP basic auth when
    # fine-grained access control with the internal user database is enabled.
    MANAGED_DOMAIN,
    # An OpenSearch Serverless "Classic" vector search collection. SigV4 signing name `aoss`.
    # Custom document `_id` is rejected on write, so `add` is append-only and `delete` requires a
    # search-then-bulk-delete round trip. Refresh interval is a fixed 60 seconds.
    SERVERLESS_CLASSIC,
    # An OpenSearch Serverless "NextGen" vector search collection. SigV4 signing name `aoss`.
    # Custom document `_id` is supported, so `add` is an upsert and `delete` is a single bulk
    # call, matching managed-domain semantics. Vectors are excluded from `_source` by default,
    # which this module always overrides explicitly. Refresh interval is a fixed 10 seconds.
    SERVERLESS_NEXTGEN
}

# The approximate nearest-neighbor engine backing the `knn_vector` field on a managed domain or a
# Serverless Classic collection. Ignored on `SERVERLESS_NEXTGEN`, which auto-configures its engine
# and rejects an explicit `method` block in the index mapping.
public enum Engine {
    # Facebook AI Similarity Search. The only engine Serverless Classic supports, and the only one
    # under which efficient k-NN pre-filtering works.
    FAISS = "faiss",
    # Apache Lucene. Supports efficient k-NN pre-filtering; not available on Serverless.
    LUCENE = "lucene",
    # Non-Metric Space Library. Historical default on managed domains; does not support efficient
    # k-NN pre-filtering and is rejected on Serverless.
    NMSLIB = "nmslib"
}

# HTTP basic authentication against a fine-grained-access-control internal user database.
# Valid only on `MANAGED_DOMAIN` — Serverless has no basic-auth path and `init` rejects this
# combination. A domain access policy containing IAM principals requires SigV4-signed requests,
# so a request cannot carry both a username/password and IAM credentials on the same domain.
public type BasicAuth record {|
    # The internal-user-database username.
    string username;
    # The internal-user-database password.
    string password;
|};

# The authentication mechanism used to sign or authorize requests to the OpenSearch endpoint:
# an AWS credential source (SigV4), or HTTP basic auth (`MANAGED_DOMAIN` only).
public type Auth auth:AuthConfig|BasicAuth;

# Configuration for the `knn_vector` field and the index-creation mapping this module generates.
public type IndexConfig record {|
    # The dimensionality of the dense vectors stored in this index. No client-side ceiling is
    # enforced — AWS's own documentation disagrees on the maximum (10,000 on some pages, 16,000 on
    # others), so the server is left to reject an out-of-range value with its own message.
    int dimension;
    # The vector similarity metric, mapped to the OpenSearch `space_type`: `COSINE` →
    # `cosinesimil`, `EUCLIDEAN` → `l2`, `DOT_PRODUCT` → `innerproduct`.
    ai:SimilarityMetric similarityMetric = ai:COSINE;
    # The ANN engine. Ignored on `SERVERLESS_NEXTGEN`. Forced to `FAISS` on `SERVERLESS_CLASSIC`;
    # any other value is rejected at construction.
    Engine engine = FAISS;
    # The HNSW `ef_construction` parameter (build-time accuracy/speed trade-off).
    int efConstruction = 128;
    # The HNSW `m` parameter (max bi-directional links per node).
    int m = 16;
    # Whether `init` should create the index if it does not already exist. When `false`, `init`
    # performs no network I/O at all — useful for least-privilege deployments where the calling
    # principal has no `CreateIndex` permission, and for fully offline construction in tests.
    boolean createIndexIfNotExists = true;
|};

# Exponential-backoff retry configuration applied to retryable HTTP responses
# (`429`, `408`, `500`, `502`, `503`, `504`).
public type RetryConfig record {|
    # The maximum number of retry attempts after the initial request.
    int maxRetries = 3;
    # The delay, in seconds, before the first retry.
    decimal initialDelay = 0.5;
    # The maximum delay, in seconds, between retries.
    decimal maxDelay = 8;
    # The multiplier applied to the delay after each retry.
    decimal backoffFactor = 2;
|};

# Configuration for the AWS OpenSearch vector store.
public type Configuration record {|
    # Shape of the `knn_vector` field and the index-creation mapping.
    IndexConfig indexConfig;
    # The document field that stores the dense vector.
    string vectorFieldName = "embedding";
    # The document field that stores the chunk content.
    string contentFieldName = "content";
    # The document field that carries the logical entry id. Duplicates `_id` on deployments where
    # a custom `_id` can be written, and is the only handle for identity where it cannot
    # (`SERVERLESS_CLASSIC`).
    string idFieldName = "doc_id";
    # The document field metadata is nested under. Metadata filters are translated to
    # `<metadataFieldName>.<key>` paths. Set to `""` to address bare `<key>` paths instead, for
    # pointing this module at a pre-existing index with a flat schema.
    string metadataFieldName = "metadata";
    # Whether `query` results include the stored vector. When `false`, the vector field is
    # excluded from `_source` and `VectorMatch.embedding` is returned as `[]` — a meaningful
    # bandwidth saving at high dimensions and large `topK`.
    boolean includeEmbeddingsInResults = true;
    # Whether to convert a `cosinesimil` `_score` back to a `[-1, 1]` cosine similarity
    # (`cos = 2 * score - 1`) before returning it as `VectorMatch.similarityScore`. Applies only
    # when `indexConfig.similarityMetric` is `COSINE`; ignored otherwise.
    boolean normalizeCosineScore = false;
    # Whether `add` requests `?refresh=wait_for` so newly-added entries are immediately visible to
    # `query`. `MANAGED_DOMAIN` only — both Serverless generations have a fixed, unforceable
    # refresh interval and `init` rejects this combination for them.
    boolean refreshOnWrite = false;
    # The maximum number of entries per `_bulk` request issued by `add`. Large `add` calls are
    # chunked into requests of at most this size, to stay under the endpoint's HTTP payload cap.
    int maxBulkSize = 500;
    # The `size` sent when `VectorStoreQuery.topK` requests "all entries" (any `topK < 1`), and the
    # upper bound enforced on an explicit `topK`. OpenSearch's own `k`/`size` ceiling is 10,000;
    # raise this only if the index's `index.max_result_window` setting has also been raised.
    int maxResultWindow = 10000;
    # Extra headers to include, and SigV4-sign, on every request. Needed for AOSS's per-account
    # NextGen endpoint, which identifies the target collection via a signed
    # `x-amz-aoss-collection-name` or `x-amz-aoss-collection-id` header rather than the hostname.
    map<string> additionalHeaders = {};
    # Retry behavior for transient failures.
    RetryConfig retryConfig = {};
|};

// --- Wire types (module-private) -------------------------------------------------------------
// Deliberately open records: OpenSearch response bodies carry many fields this module never
// reads (`took`, `timed_out`, `_shards`, `_version`, ...), and open records let those pass
// through `cloneWithType` without requiring every field to be declared here.

# The `error` object embedded in an OpenSearch error response, or in a failed `_bulk` item.
type ErrorDetail record {
    # OpenSearch's error classification, e.g. `resource_already_exists_exception`.
    string 'type?;
    # A human-readable explanation of the error.
    string reason?;
};

# The per-action result nested one level inside a `_bulk` response item, e.g. the value of the
# `"index"` or `"delete"` key in `{"index": {...}}` / `{"delete": {...}}`.
type BulkItemResult record {
    # The index the action targeted.
    string _index?;
    # The document `_id` the action targeted.
    string _id?;
    # The per-item HTTP-equivalent status code.
    int status?;
    # The outcome, e.g. `created`, `deleted`, `not_found`.
    string result?;
    # Present when this item failed.
    ErrorDetail 'error?;
};

# The response body of a `POST /_bulk` request. Each element of `items` is a single-key map whose
# key names the action (`index`, `create`, `update`, or `delete`); the key itself is not modelled
# because this module only ever issues `index` and `delete` actions and reads the result the same
# way regardless of which key is present.
type BulkResponse record {
    # `true` if any item in `items` failed. HTTP 200 is returned either way, so this must always
    # be checked explicitly (§5.3).
    boolean errors;
    # The per-item results, in request order.
    map<BulkItemResult>[] items = [];
};

# A single hit within a `_search` response.
type SearchHit record {
    # The document's internal `_id`.
    string _id;
    # The relevance/similarity score.
    float _score = 0.0;
    # The stored document, subject to the request's `_source` directive.
    map<json> _source?;
};

# The `hits` object of a `_search` response.
type SearchHits record {
    # The matching documents.
    SearchHit[] hits = [];
};

# The response body of a `POST /_search` request.
type SearchResponse record {
    # The search results.
    SearchHits hits;
};

# The response body of a non-2xx OpenSearch response.
type OpenSearchErrorResponse record {
    # The error detail, when present.
    ErrorDetail 'error?;
    # The status, mirroring the HTTP status code.
    int status?;
};
