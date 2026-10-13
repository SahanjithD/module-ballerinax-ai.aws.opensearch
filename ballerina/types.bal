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

// The two Serverless generations behave differently on the data plane and cannot be told apart
// from it, so the caller declares which one they have.
# The kind of OpenSearch deployment, which decides request signing, write semantics and the
# `knn_vector` mapping.
public enum DeploymentType {
    # An OpenSearch Service domain, signed as `es` or using fine-grained access control basic auth.
    MANAGED_DOMAIN,
    # A Serverless Classic collection. No custom `_id`, so `add` appends and `delete` searches
    # before deleting.
    SERVERLESS_CLASSIC,
    // Some AWS pages say vector collections reject a custom `_id`; that holds for Classic only.
    # A Serverless NextGen collection. Custom `_id` is supported, so `add` upserts.
    SERVERLESS_NEXTGEN
}

// NMSLIB is left out: deprecated in OpenSearch 2.16 and removed in 3.0.
# The approximate nearest-neighbor engine. Only a managed domain can choose; Serverless uses Faiss.
public enum Engine {
    # Facebook AI Similarity Search. The engine Serverless Classic and quantization require.
    FAISS = "faiss",
    # Apache Lucene. Supports efficient k-NN pre-filtering; not available on Serverless.
    LUCENE = "lucene"
}

// No `4x`: NextGen rejects it ("Unknown value [4x] for field [compression_level]") and Faiss does
// not support it either, so it would fail everywhere this enum is reachable.
# How much `knn_vector` storage is quantized, trading recall for memory. Serverless NextGen only.
public enum CompressionLevel {
    # No compression; vectors are stored at full precision.
    COMPRESSION_1X = "1x",
    COMPRESSION_2X = "2x",
    COMPRESSION_8X = "8x",
    COMPRESSION_16X = "16x",
    # NextGen's default when nothing is set.
    COMPRESSION_32X = "32x"
}

# HTTP basic auth against a fine-grained access control internal user database. Managed domains
# only.
public type BasicAuth record {|
    # The internal user's name.
    string username;
    # The internal user's password.
    string password;
|};

# SigV4 credentials, or HTTP basic auth on a managed domain.
public type Auth auth:AuthConfig|BasicAuth;

// Each setting lives on the one variant that honors it, so a wrong combination (e.g. `BasicAuth`
// on Serverless) is a type error rather than a runtime check.
# The target deployment, with the settings only that deployment honors.
public type Deployment ManagedDomainDeployment|ServerlessClassicDeployment|ServerlessNextGenDeployment;

# An OpenSearch Service (managed) domain.
public type ManagedDomainDeployment record {|
    # Selects this variant.
    MANAGED_DOMAIN deploymentType;
    // Required rather than defaulted: silently falling back to the ambient credential chain hides a
    // misconfiguration until it shows up as a 403 for the wrong principal.
    # SigV4 credentials (`auth:DEFAULT_CREDENTIALS` for the default chain), or `BasicAuth`.
    Auth auth;
    # The engine backing the `knn_vector` field.
    Engine engine = FAISS;
    # Whether `add` waits for a refresh so new entries are queryable immediately.
    boolean refreshOnWrite = false;
|};

// Nothing to tune beyond credentials: Classic supports Faiss only, rejects custom `_id`s and has a
// fixed refresh interval.
# An OpenSearch Serverless Classic vector search collection.
public type ServerlessClassicDeployment record {|
    # Selects this variant.
    SERVERLESS_CLASSIC deploymentType;
    # SigV4 credentials. Serverless has no basic auth.
    auth:AuthConfig auth;
|};

# An OpenSearch Serverless NextGen vector search collection.
public type ServerlessNextGenDeployment record {|
    # Selects this variant.
    SERVERLESS_NEXTGEN deploymentType;
    # SigV4 credentials. Serverless has no basic auth.
    auth:AuthConfig auth;
    // A per-account NextGen endpoint does not say which collection a request is for, so it needs
    // one of these two; a per-collection endpoint needs neither.
    # Target collection name, sent as the signed `x-amz-aoss-collection-name` header.
    string collectionName?;
    # Target collection id, sent as `x-amz-aoss-collection-id`. Set this or `collectionName`.
    string collectionId?;
    // `mode` is not offered: the AOSS proxy rejects every value of it.
    # Quantization ratio. Unset keeps NextGen's default of `32x`, which is lossy.
    CompressionLevel? compressionLevel = ();
|};

// Applied only by the `PUT /<index>` that `init` sends when the index is missing. Changing these
// later has no effect on an existing index and raises no error; reindex to change them.
# Shape of the `knn_vector` field, applied when the index is created.
public type IndexConfig record {|
    // No client-side ceiling: AWS documents both 10,000 and 16,000, so the server decides.
    # Number of dimensions in each dense vector.
    int dimension;
    # Similarity metric, mapped to the `cosinesimil`, `l2` or `innerproduct` space type.
    ai:SimilarityMetric similarityMetric = ai:COSINE;
    # HNSW `ef_construction`. Higher builds a more accurate graph, more slowly.
    int efConstruction = 128;
    # HNSW `m`, the maximum number of graph links per node.
    int m = 16;
|};

// The processor splits a raw `query_tokens` map directly, so this works without a `model_id`.
// It does not descend into a `hybrid` query, which is why `HybridSearch` does not offer it.
# Two-phase `neural_sparse` acceleration (OpenSearch 2.15+): high-weight tokens select candidates
# and low-weight tokens rescore only those.
public type TwoPhaseAcceleration record {|
    # Fraction of the largest query-token weight that separates high from low tokens, in `[0, 1]`.
    float pruneRatio = 0.4;
|};

// No `z_score`: it is OpenSearch 3.0+, and AWS documents only `min_max` and `l2` for Serverless.
# How each sub-query's scores are normalized before they are combined. `HYBRID` only.
public enum NormalizationTechnique {
    // An exact 0.0 is replaced with 0.001, since OpenSearch reads a 0.0 score as `match_none`.
    # Rescales each sub-query against its own minimum and maximum, giving scores in `(0.0, 1.0]`.
    MIN_MAX = "min_max",
    # Divides by the L2 norm of the sub-query's scores, preserving their ratios.
    L2 = "l2"
}

# How the normalized dense and sparse scores are combined. `HYBRID` only.
public enum CombinationTechnique {
    # Weighted arithmetic mean. A document matching only one sub-query is penalised.
    ARITHMETIC_MEAN = "arithmetic_mean",
    GEOMETRIC_MEAN = "geometric_mean",
    HARMONIC_MEAN = "harmonic_mean"
}

// Sent inline with every query rather than as a named pipeline: the weights cannot go stale in
// cluster state, IAM stays at index scope, and there is no Serverless pipeline propagation delay.
# Score-based fusion for `HYBRID`, sent as an inline `normalization-processor`.
public type HybridSearchConfig record {|
    # How each sub-query's scores are rescaled.
    NormalizationTechnique normalization = MIN_MAX;
    # How the rescaled scores are combined.
    CombinationTechnique combination = ARITHMETIC_MEAN;
    # Weight of the dense sub-query, in `[0, 1]`. Must sum to 1.0 with `sparseWeight`.
    float denseWeight = 0.5;
    # Weight of the sparse sub-query. See `denseWeight`.
    float sparseWeight = 0.5;
|};

# Selects `RrfFusion`; also the `combination.technique` value the `score-ranker-processor` takes.
public const RRF = "rrf";

// No settings: neural-search 3.1 moved `rank_constant` and made the old location an error, and
// `weights` is ignored before 3.1, so neither can be sent portably.
# Reciprocal rank fusion (OpenSearch 2.19+). Fuses by rank rather than score, so scores are sums
# of reciprocal ranks (about `(0, 0.033]`), not similarities.
public type RrfFusion record {|
    // Required, so a bare `{}` still resolves to `HybridSearchConfig`.
    # Selects this variant.
    RRF technique;
|};

// For principals that cannot create pipelines, or to share one tuned pipeline. This module neither
// creates nor checks it.
# Fusion by a search pipeline already on the cluster, sent as `?search_pipeline=<name>`.
public type NamedSearchPipeline record {|
    # The name the pipeline is registered under.
    string name;
|};

// Lets the request builder tell fusion sent in the body apart from a named pipeline.
type InlineFusion HybridSearchConfig|RrfFusion;

# Where `HYBRID` fusion comes from: inline (`HybridSearchConfig`, `RrfFusion`) or a named pipeline.
public type HybridFusion HybridSearchConfig|RrfFusion|NamedSearchPipeline;

// Each setting lives on the one variant that honors it, as with `Deployment`. Field names and the
// index shape apply at index creation only.
# The kind of search, chosen at construction, with the settings only that kind honors.
public type SearchMode DenseSearch|SparseSearch|HybridSearch;

# Dense vector search over a `knn_vector` field.
public type DenseSearch record {|
    # Selects this variant.
    ai:DENSE queryMode;
    # Shape of the `knn_vector` field.
    IndexConfig indexConfig;
    # Field holding the dense vector.
    string vectorFieldName = "embedding";
    // On by default so a score means the same here as in `ai:InMemoryVectorStore`.
    # Under `COSINE`, converts OpenSearch's `[0, 1]` score to a cosine in `[-1, 1]`.
    boolean normalizeCosineScore = true;
    // Unset omits `method_parameters`, so the index's own `ef_search` (default 100) applies.
    # HNSW `ef_search`, sent with every query. Higher improves recall but adds latency.
    # OpenSearch 2.16+.
    int? efSearch = ();
|};

// The weights come from the caller's encoder; no ML model, ingest pipeline or `index.knn` is used.
# Sparse vector search over a `rank_features` field, using `neural_sparse` with precomputed tokens.
# Needs the neural-search plugin and OpenSearch 2.14+.
public type SparseSearch record {|
    # Selects this variant.
    ai:SPARSE queryMode;
    # Field holding the sparse vector.
    string sparseVectorFieldName = "sparse_embedding";
    // One Lucene clause per term, bounded by `indices.query.bool.max_clause_count` (default 1024).
    # Most non-zero terms a query vector may carry. Raise only with the cluster's clause limit.
    int maxQueryTokens = 1024;
    # Two-phase acceleration (OpenSearch 2.15+). Unset sends no pipeline.
    TwoPhaseAcceleration? twoPhaseAcceleration = ();
|};

// Filters are copied into each sub-query because a top-level `hybrid.filter` is OpenSearch 3.0+.
# Hybrid search: a `knn` and a `neural_sparse` sub-query fused server-side (OpenSearch 2.11+).
# `add` requires an `ai:HybridVector`, so both fields are always populated.
public type HybridSearch record {|
    # Selects this variant.
    ai:HYBRID queryMode;
    # Shape of the `knn_vector` field.
    IndexConfig indexConfig;
    # Field holding the dense vector.
    string vectorFieldName = "embedding";
    # Field holding the sparse vector.
    string sparseVectorFieldName = "sparse_embedding";
    # Most non-zero terms a query sparse vector may carry. See `SparseSearch.maxQueryTokens`.
    int maxQueryTokens = 1024;
    # HNSW `ef_search` for the dense sub-query. See `DenseSearch.efSearch`.
    int? efSearch = ();
    # How the sub-query scores are fused: inline, or by a named pipeline.
    HybridFusion fusion = {};
|};

# Exponential backoff for retryable responses (`429`, `408`, `500`, `502`, `503`, `504`).
public type RetryConfig record {|
    # Retries after the initial request.
    int maxRetries = 3;
    # Delay before the first retry, in seconds.
    decimal initialDelay = 0.5;
    # Longest delay between retries, in seconds.
    decimal maxDelay = 8;
    # Multiplier applied to the delay after each retry.
    decimal backoffFactor = 2;
|};

# Settings honored the same way under every search mode and deployment.
public type Configuration record {|
    # Field holding the chunk content.
    string contentFieldName = "content";
    // Duplicates `_id` where a custom `_id` can be written; on Serverless Classic it is the only
    // handle on an entry's identity.
    # Field holding the entry's logical id.
    string idFieldName = "doc_id";
    # Field metadata is nested under. `""` stores metadata at the top level of the document.
    string metadataFieldName = "metadata";
    # Whether `query` results carry the stored vectors. `false` saves bandwidth.
    boolean includeEmbeddingsInResults = true;
    // Keeps each request under the endpoint's HTTP payload limit.
    # Most entries per `_bulk` request; larger `add` calls are split.
    int maxBulkSize = 500;
    // 10,000 is OpenSearch's default `index.max_result_window`.
    # The `size` used when `topK < 1`, and the cap on an explicit `topK`.
    int maxResultWindow = 10000;
    // With this off, a missing index is not caught on write: `action.auto_create_index` builds one
    // without the vector mapping. `verifyOnInit` catches that at construction.
    # Whether `init` creates the index when it does not exist.
    boolean createIndexIfNotExists = true;
    // A check that runs and disagrees fails `init`. One that cannot run (e.g. no permission) only
    // logs a warning, since refusing a store that would have worked is the worse outcome.
    # Whether `init` checks the cluster version and the index's vector fields against this store.
    boolean verifyOnInit = true;
    # Retry behavior for transient failures.
    RetryConfig retryConfig = {};
|};

// Wire types. Open records, so response fields this module never reads pass through
// `cloneWithType`.

// A wrapper's own `reason` is often unhelpful ("all shards failed"); the specific one is in
// `root_cause[0]` on error responses and in `caused_by` on `_bulk` items.
# The `error` object of an OpenSearch error response or a failed `_bulk` item.
type ErrorDetail record {
    # OpenSearch's error type, e.g. `resource_already_exists_exception`.
    string 'type?;
    # Human-readable explanation.
    string reason?;
    # The wrapped cause, which may nest further.
    ErrorDetail caused_by?;
    # The originating errors behind a wrapper error. Absent on `_bulk` item errors.
    ErrorDetail[] root_cause?;
};

# The result of one `_bulk` action, e.g. the value under `"index"` in `{"index": {...}}`.
type BulkItemResult record {
    # Index the action targeted.
    string _index?;
    # Document `_id` the action targeted.
    string _id?;
    # Per-item HTTP-equivalent status.
    int status?;
    # Outcome, e.g. `created`, `deleted`, `not_found`.
    string result?;
    # Present when this item failed.
    ErrorDetail 'error?;
};

# The `POST /_bulk` response. Each item is keyed by its action, which is read the same either way.
type BulkResponse record {
    # Whether any item failed. The HTTP status is 200 either way.
    boolean errors;
    # Per-item results, in request order.
    map<BulkItemResult>[] items = [];
};

# One hit in a `_search` response.
type SearchHit record {
    # Internal document id.
    string _id;
    # Relevance score.
    float _score = 0.0;
    # Stored document, subject to the request's `_source` directive.
    map<json> _source?;
};

# The `hits` object of a `_search` response.
type SearchHits record {
    # Matching documents.
    SearchHit[] hits = [];
};

# The `POST /_search` response.
type SearchResponse record {
    # Search results.
    SearchHits hits;
};

# The body of a non-2xx OpenSearch response.
type OpenSearchErrorResponse record {
    # Error detail, when present.
    ErrorDetail 'error?;
    # Mirrors the HTTP status.
    int status?;
};
