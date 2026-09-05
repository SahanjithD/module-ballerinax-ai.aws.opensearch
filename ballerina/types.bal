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
# whether a custom document `_id` can be written, how `delete` is implemented, and how the
# `knn_vector` field's `method` block is shaped.
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
    #
    # The custom `_id` support was verified through the path this module actually uses — the
    # `_bulk` action line — where re-adding an id replaces the document rather than duplicating
    # it. Parts of the AWS documentation state that vector search collections reject a custom
    # document id; that is true of `SERVERLESS_CLASSIC` but not of NextGen.
    #
    # The only `knn_vector` mapping parameter NextGen rejects is `engine` inside the `method`
    # block (and, per AWS, `mode`): a block carrying it fails with a flat
    # `400 Field parameter 'engine' is not supported`, raised by the AOSS proxy in front of
    # OpenSearch, which is why it carries no OpenSearch error type to map. A block of
    # `{"name": "hnsw", "parameters": {...}}` without `engine` is accepted and its parameters are
    # stored intact — NextGen fills in `engine: faiss` itself in the stored mapping. HNSW tuning
    # therefore works here exactly as it does elsewhere; see `buildIndexMapping`.
    SERVERLESS_NEXTGEN
}

# The approximate nearest-neighbor engine backing the `knn_vector` field. Selectable only on
# `ManagedDomainDeployment`: `SERVERLESS_CLASSIC` supports Faiss alone, and `SERVERLESS_NEXTGEN`
# rejects the `engine` parameter inside the index mapping's `method` block outright, filling in
# `faiss` itself. Neither variant exposes the field, so a wrong value is not expressible rather
# than rejected at construction.
# NMSLIB is deliberately absent: it was deprecated in OpenSearch 2.16 and removed in 3.0, where
# creating a new NMSLIB index is blocked outright. Offering it would hand callers a value that
# degrades filtering on 2.x and fails index creation on 3.x.
public enum Engine {
    # Facebook AI Similarity Search. The only engine Serverless Classic supports, and the engine
    # the quantization parameters require.
    FAISS = "faiss",
    # Apache Lucene. Supports efficient k-NN pre-filtering; not available on Serverless.
    LUCENE = "lucene"
}

# The vector quantization ratio applied to the `knn_vector` field, trading recall for memory.
# `SERVERLESS_NEXTGEN` only — see `ServerlessNextGenDeployment.compressionLevel`.
#
# That restriction is this module's own scoping decision, not a server constraint. An earlier note
# here claimed OpenSearch accepts `compression_level` beside a `method` block while silently
# emptying the block's `parameters`; that was tested against OpenSearch 2.19.1 and is false. The
# combination is accepted with `ef_construction`/`m` stored intact, and an incompatible pairing is
# rejected loudly instead (`"faiss" does not support "4x" compression`). Offering these on a
# managed domain would therefore be safe, and is simply not in scope for this release.
public enum CompressionLevel {
    # No compression; vectors are stored at full precision.
    COMPRESSION_1X = "1x",
    COMPRESSION_2X = "2x",
    COMPRESSION_4X = "4x",
    COMPRESSION_8X = "8x",
    COMPRESSION_16X = "16x",
    # The ratio `SERVERLESS_NEXTGEN` applies when nothing is configured.
    COMPRESSION_32X = "32x"
}

# Where the `knn_vector` field's index is held. `SERVERLESS_NEXTGEN` only — see
# `ServerlessNextGenDeployment.vectorMode`.
public enum VectorMode {
    # Full-precision vectors held in memory. Defaults to `1x` compression.
    IN_MEMORY = "in_memory",
    # Vectors held on disk with a quantized in-memory copy. Defaults to `32x` compression, and
    # rejects `COMPRESSION_1X` outright.
    ON_DISK = "on_disk"
}

# HTTP basic authentication against a fine-grained-access-control internal user database.
# Reachable only through `ManagedDomainDeployment.auth` — Serverless has no basic-auth path, and
# neither Serverless variant admits this type. A domain access policy containing IAM principals
# requires SigV4-signed requests, so a request cannot carry both a username/password and IAM
# credentials on the same domain.
public type BasicAuth record {|
    # The internal-user-database username.
    string username;
    # The internal-user-database password.
    string password;
|};

# The authentication mechanism used to sign or authorize requests to the OpenSearch endpoint:
# an AWS credential source (SigV4), or HTTP basic auth (managed domains only).
public type Auth auth:AuthConfig|BasicAuth;

# The deployment flavour of the target cluster, together with the settings that only that flavour
# honors. Selecting a variant is the first choice a caller makes, and it is what makes a
# deployment-specific setting unreachable everywhere else: `refreshOnWrite` cannot be named on a
# Serverless collection, `compressionLevel` cannot be named on a managed domain, and `BasicAuth`
# cannot be handed to either Serverless generation. Those were once `init`-time validation rules;
# they are now type errors.
#
# Settings honored identically everywhere — dimension, similarity metric, HNSW tuning, field
# names, bulk sizing, retries — live on `Configuration`/`IndexConfig` instead, so a caller
# switching deployments carries them across untouched.
public type Deployment ManagedDomainDeployment|ServerlessClassicDeployment|ServerlessNextGenDeployment;

# Targets an OpenSearch Service (managed) domain. The only deployment that accepts a non-Faiss
# engine, `BasicAuth`, or a forced refresh on write.
public type ManagedDomainDeployment record {|
    # Discriminates this variant. Required, so a record literal resolves to exactly one member of
    # `Deployment`.
    MANAGED_DOMAIN deploymentType;
    # SigV4 credentials, or `BasicAuth` against a fine-grained-access-control internal user
    # database. A domain does one or the other, never both.
    #
    # Required rather than defaulted: a store that silently falls back to the ambient AWS
    # credential chain hides a misconfiguration until it surfaces as a 403 from the wrong
    # principal. Pass `auth:DEFAULT_CREDENTIALS` to opt into that chain explicitly.
    Auth auth;
    # The ANN engine backing the `knn_vector` field.
    Engine engine = FAISS;
    # Whether `add` requests `?refresh=wait_for` so newly-added entries are immediately visible to
    # `query`. Only forceable here: both Serverless generations have a fixed refresh interval
    # (60 seconds on Classic, 10 on NextGen) that no request parameter overrides.
    boolean refreshOnWrite = false;
|};

# Targets an OpenSearch Serverless "Classic" vector search collection.
#
# Nothing is configurable beyond credentials, and that is the honest shape: Classic's differences
# from a managed domain are behavioral rather than tunable. It supports Faiss alone, so `engine`
# would have exactly one legal value; it rejects a custom document `_id`, which makes `add`
# append-only and `delete` a search-then-bulk-delete round trip; and its refresh interval is fixed.
public type ServerlessClassicDeployment record {|
    # Discriminates this variant.
    SERVERLESS_CLASSIC deploymentType;
    # SigV4 credentials. Serverless has no basic-auth path, so `BasicAuth` is not admitted here.
    # Required rather than defaulted, for the same reason as `ManagedDomainDeployment.auth`.
    auth:AuthConfig auth;
|};

# Targets an OpenSearch Serverless "NextGen" vector search collection. The only deployment that
# exposes vector quantization, which it applies by default whether or not a caller asks.
public type ServerlessNextGenDeployment record {|
    # Discriminates this variant.
    SERVERLESS_NEXTGEN deploymentType;
    # SigV4 credentials. Serverless has no basic-auth path, so `BasicAuth` is not admitted here.
    # Required rather than defaulted, for the same reason as `ManagedDomainDeployment.auth`.
    auth:AuthConfig auth;
    # The target collection's name, sent as a signed `x-amz-aoss-collection-name` header.
    #
    # The NextGen endpoint is per-account rather than per-collection, so the hostname alone does
    # not say which collection a request is for — one of this or `collectionId` identifies it.
    # Neither is required here, since a per-collection endpoint URL needs no such header.
    string collectionName?;
    # The target collection's id, sent as a signed `x-amz-aoss-collection-id` header. An
    # alternative to `collectionName`; setting both is rejected at construction.
    string collectionId?;
    # The vector quantization ratio, emitted as the `knn_vector` field's `compression_level`.
    #
    # Leaving this unset preserves the server's default, which is not lossless: NextGen provisions
    # every vector field as `on_disk`/`32x` when nothing is configured, so vectors are quantized by
    # default and the stored mapping is the only place that says so. Set `COMPRESSION_1X` (together
    # with `vectorMode = IN_MEMORY`, since `on_disk` rejects `1x`) to opt out of quantization.
    #
    # Verified accepted by OpenSearch 2.19.1 itself, and accepted by the AOSS NextGen proxy
    # alongside a `method` block carrying HNSW `parameters` — both survive into the stored mapping.
    CompressionLevel? compressionLevel = ();
    # Where the vector index is held, emitted as the `knn_vector` field's `mode`. Carries the same
    # caveats as `compressionLevel`.
    #
    # `ON_DISK` with `COMPRESSION_1X` is rejected at construction, matching the server's own
    # validation (`Cannot specify "x1" compression level when using "on_disk" mode`).
    VectorMode? vectorMode = ();
|};

# Configuration for the `knn_vector` field and the index-creation mapping this module generates.
#
# # These apply at index creation only
# Every field here except `createIndexIfNotExists` is sent in exactly one request — the
# `PUT /<index>` that `init` issues when the index does not yet exist. Once the index exists,
# `init` sees it and returns without building a mapping at all, so **editing any of these has no
# effect on an existing index, and produces no error**. The same is true of
# `ManagedDomainDeployment.engine` and of `ServerlessNextGenDeployment.compressionLevel`/
# `vectorMode`.
#
# Nothing in a later request restates them: a `_bulk` document carries a vector, and a `knn` query
# carries a vector and `k`. Neither mentions `space_type`, `engine`, or `compression_level`, so the
# server has nothing to disagree with and reports nothing. A changed `similarityMetric` keeps
# returning scores computed under the original metric; a `compressionLevel` set to opt out of
# `SERVERLESS_NEXTGEN`'s default quantization leaves the vectors quantized.
#
# `dimension` is not an exception. It is never checked against the embeddings this module sends —
# a mismatched `dimension` beside unchanged embeddings simply works, and stays wrong silently. An
# error appears only when the *embeddings themselves* change length, because OpenSearch then
# rejects the vector against the stored mapping, at the first `add` rather than at `init`.
#
# To actually change the shape of an existing index, reindex into a new one. This module does not
# update mappings.
public type IndexConfig record {|
    # The dimensionality of the dense vectors stored in this index. No client-side ceiling is
    # enforced — AWS's own documentation disagrees on the maximum (10,000 on some pages, 16,000 on
    # others), so the server is left to reject an out-of-range value with its own message.
    int dimension;
    # The vector similarity metric, mapped to the OpenSearch `space_type`: `COSINE` →
    # `cosinesimil`, `EUCLIDEAN` → `l2`, `DOT_PRODUCT` → `innerproduct`.
    ai:SimilarityMetric similarityMetric = ai:COSINE;
    # The HNSW `ef_construction` parameter (build-time accuracy/speed trade-off). Honored on all
    # three deployment types: the `method` block carrying it is accepted everywhere, including on
    # `SERVERLESS_NEXTGEN` once `engine` is left out of it.
    int efConstruction = 128;
    # The HNSW `m` parameter (max bi-directional links per node). Honored on all three deployment
    # types, for the same reason as `efConstruction`.
    int m = 16;
    # Whether `init` should create the index if it does not already exist. When `false`, `init`
    # performs no network I/O at all — useful for least-privilege deployments where the calling
    # principal has no `CreateIndex` permission, and for fully offline construction in tests.
    #
    # # Hazard when `false`
    # `init` does not verify that the index exists, and a missing index does not make `add` fail:
    # OpenSearch ships with `action.auto_create_index: true`, so the first `_bulk` write creates
    # an index from the document's inferred shape instead. That index has no `index.knn` setting
    # and maps the vector as a plain `float` array rather than a `knn_vector`, so writes keep
    # succeeding while every `query` fails with a `400`. Recovering means deleting the index and
    # reindexing from source.
    #
    # Only set this to `false` against an index you know was provisioned out of band with a
    # compatible mapping.
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
    # bandwidth saving at high dimensions and large `topK`. When `true`, what comes back is the
    # vector as storage holds it, which is not necessarily the one that was written — see
    # "Returned embeddings" on `VectorStore`.
    boolean includeEmbeddingsInResults = true;
    # Whether to convert a `cosinesimil` `_score` back to a `[-1, 1]` cosine similarity
    # (`cos = 2 * score - 1`) before returning it as `VectorMatch.similarityScore`. Applies only
    # when `indexConfig.similarityMetric` is `COSINE`; ignored otherwise.
    #
    # Defaults to `true` so `VectorMatch.similarityScore` means the same thing here as it does in
    # `ai:InMemoryVectorStore`, which returns a true cosine in `[-1, 1]`. OpenSearch's raw
    # `cosinesimil` `_score` is `[0, 1]`, so leaving this off would make a threshold tuned against
    # any other `ai:VectorStore` implementation silently mean something else against this one.
    # Set `false` to get OpenSearch's `_score` through untouched.
    boolean normalizeCosineScore = true;
    # The maximum number of entries per `_bulk` request issued by `add`. Large `add` calls are
    # chunked into requests of at most this size, to stay under the endpoint's HTTP payload cap.
    int maxBulkSize = 500;
    # The `size` sent when `VectorStoreQuery.topK` requests "all entries" (any `topK < 1`), and the
    # upper bound enforced on an explicit `topK`. OpenSearch's own `k`/`size` ceiling is 10,000;
    # raise this only if the index's `index.max_result_window` setting has also been raised.
    int maxResultWindow = 10000;
    # Retry behavior for transient failures.
    RetryConfig retryConfig = {};
|};

// --- Wire types (module-private) -------------------------------------------------------------
// Deliberately open records: OpenSearch response bodies carry many fields this module never
// reads (`took`, `timed_out`, `_shards`, `_version`, ...), and open records let those pass
// through `cloneWithType` without requiring every field to be declared here.

# The `error` object embedded in an OpenSearch error response, or in a failed `_bulk` item.
#
# `reason` is frequently the least useful field on it. A shard-level search failure puts
# `all shards failed` there and the actual explanation in `root_cause[0]`; a failed `_bulk` item
# puts `failed to parse field [embedding] ... Preview of field's value: 'null'` there and the
# actual explanation in `caused_by`. Both nestings are modelled so `describeErrorDetail` can
# recover the specific reason instead of reporting the wrapper's.
type ErrorDetail record {
    # OpenSearch's error classification, e.g. `resource_already_exists_exception`. On a wrapper
    # error such as `search_phase_execution_exception` this is the wrapper's own type, not the
    # underlying cause's.
    string 'type?;
    # A human-readable explanation of the error.
    string reason?;
    # The underlying cause, when this error wraps one. Nests arbitrarily deep; carried by `_bulk`
    # item errors and by some error responses.
    ErrorDetail caused_by?;
    # The originating error(s) behind a wrapper error, carried by error responses. A `_bulk` item
    # error has no `root_cause` and uses `caused_by` alone.
    ErrorDetail[] root_cause?;
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
    # be checked explicitly.
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
