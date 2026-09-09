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
# That restriction is this module's own scoping decision, not a server constraint: OpenSearch 2.19.1
# accepts `compression_level` beside a `method` block with `ef_construction`/`m` stored intact.
# Offering it on a managed domain would be safe, and is simply not in scope for this release.
#
# # Why there is no `4x`
# OpenSearch's own ratios include `4x`, and this enum deliberately omits it, because no deployment
# this module targets can store it. AOSS NextGen — the one flavour that may carry this field at
# all — names its accepted set in its own rejection, which reads
# `Unknown value [4x] for field [compression_level]` and lists `[, 1x, 2x, 8x, 16x, 32x]`.
# And `4x` is a Lucene-only ratio, which OpenSearch reaches only when it picks the engine itself;
# this module always names an engine explicitly in the `method` block, and Faiss rejects the
# pairing with `"faiss" does not support "4x" compression`. A member that fails everywhere is
# worse than an absent one, so it is absent.
public enum CompressionLevel {
    # No compression; vectors are stored at full precision.
    COMPRESSION_1X = "1x",
    COMPRESSION_2X = "2x",
    COMPRESSION_8X = "8x",
    COMPRESSION_16X = "16x",
    # The ratio `SERVERLESS_NEXTGEN` applies when nothing is configured.
    COMPRESSION_32X = "32x"
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
    # default and the stored mapping is the only place that says so. Set `COMPRESSION_1X` to opt out
    # of quantization; NextGen moves the field off `on_disk` itself to honor it, since a `1x`
    # on-disk field is a combination OpenSearch rejects.
    #
    # The companion `mode` parameter is deliberately not surfaced. It is a real `knn_vector`
    # parameter that a managed 2.19 domain accepts and stores, but the AOSS NextGen proxy — the
    # only place this record reaches — implements no value for it, answering with a flat 400
    # reading `Field parameter 'mode' is not supported`, for both `in_memory` and `on_disk`.
    #
    # Verified accepted by OpenSearch 2.19.1 itself, and accepted by the AOSS NextGen proxy
    # alongside a `method` block carrying HNSW `parameters` — both survive into the stored mapping.
    CompressionLevel? compressionLevel = ();
|};

# Configuration for the `knn_vector` field and the index-creation mapping this module generates.
#
# # These apply at index creation only
# Every field here is sent in exactly one request — the
# `PUT /<index>` that `init` issues when the index does not yet exist. Once the index exists,
# `init` sees it and returns without building a mapping at all, so **editing any of these has no
# effect on an existing index, and produces no error**. The same is true of
# `ManagedDomainDeployment.engine` and of `ServerlessNextGenDeployment.compressionLevel`.
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
|};

# Acceleration for a `neural_sparse` query via the `neural_sparse_two_phase_processor`, sent as an
# inline `search_pipeline` carrying a `request_processors` entry.
#
# Without it, a `neural_sparse` query scores every matching document against every one of its query
# tokens. The processor splits the query's tokens by weight: the high-weight tokens run first and
# select a candidate set, and the low-weight tokens then rescore only those candidates. The
# ranking is approximate but close, and the saving grows with the width of the query vector.
#
# This module always sends **raw `query_tokens`** rather than a `model_id`, and the processor
# handles that case specifically: `NeuralSparseQueryBuilder.prepareTwoPhaseQuery` splits a supplied
# token map directly instead of deferring the split to model inference, so the acceleration is
# real here and not just on the model-backed path.
#
# # Requirements and hazards
# OpenSearch 2.15 or later. On an older cluster the inline pipeline is rejected and the search
# fails, which is why this is unset by default rather than always sent.
#
# The processor rewrites `neural_sparse` clauses that sit at the top level or inside a `bool` — the
# only two shapes this module's `SPARSE` mode produces. It does **not** descend into a `hybrid`
# query, which is why this is reachable from `SparseSearch` alone and not from `HybridSearch`,
# where it would be accepted and then quietly do nothing.
#
# # Why only one setting
# The processor also takes `expansion_rate` (how many documents the rescoring pass considers, as a
# multiple of the query size) and `max_window_size` (the ceiling beyond which it stops applying).
# Neither is exposed: they cannot be set meaningfully without knowing the algorithm's internals,
# and OpenSearch's own defaults — `5.0` and `10000` — are what this module's benchmarking used.
# They are simply left out of the request, so the server supplies them.
public type TwoPhaseAcceleration record {|
    # The fraction of the highest query-token weight that separates the high-weight tokens from
    # the low-weight ones. Must be in `[0.0, 1.0]`, checked at construction.
    #
    # The threshold is `pruneRatio * (the largest weight in the query vector)`: tokens above it run
    # in the first phase, the rest in the rescore. A larger value prunes more aggressively — fewer
    # tokens select the candidates, so the search is faster and the ranking is a coarser
    # approximation.
    float pruneRatio = 0.4;
|};

# The score-normalization technique the `normalization-processor` applies to each sub-query's
# scores before they are combined. `HYBRID` only.
#
# `z_score` is deliberately absent. It is OpenSearch 3.0+, and AWS documents only `min_max` and
# `l2` for Serverless, so offering it would hand callers a value that fails outright on every 2.x
# domain and is unattested on Serverless.
public enum NormalizationTechnique {
    # Rescales each sub-query's scores to `[0, 1]` against that sub-query's own minimum and
    # maximum. The highest-scoring document in a sub-query always lands on exactly `1.0`, and an
    # exact `0.0` is replaced with `0.001` (a literal `0.0` means `match_none` to OpenSearch), so
    # a combined score is `(0.0, 1.0]` rather than `[0, 1]`.
    #
    # Because only an exact `0.0` is substituted, a sub-query whose raw scores are wildly spread —
    # which an unbounded sparse dot product easily is — can have its two lowest documents reorder.
    MIN_MAX = "min_max",
    # Divides each score by the L2 norm of that sub-query's score vector, which preserves score
    # ratios where `MIN_MAX` does not. Named for the normalization, and unrelated to the `l2`
    # *space type* that `IndexConfig.similarityMetric` selects.
    L2 = "l2"
}

# How the normalized dense and sparse scores are folded into a single `_score`. `HYBRID` only.
public enum CombinationTechnique {
    # The weighted arithmetic mean. A document matching only one sub-query still counts in the
    # denominator, so it is penalised relative to one matching both.
    ARITHMETIC_MEAN = "arithmetic_mean",
    GEOMETRIC_MEAN = "geometric_mean",
    HARMONIC_MEAN = "harmonic_mean"
}

# The dense/sparse score fusion `HYBRID` asks OpenSearch to perform, sent as an inline
# `search_pipeline` object in the search body.
#
# Nothing is provisioned server-side. A `hybrid` query needs a normalization pipeline to produce
# usable scores, but a *named* one would make these knobs durable cluster state this module owns
# the lifecycle of — created once, then silently stale the moment a caller changed a weight, which
# is exactly the trap `IndexConfig` already documents for index-creation settings. Fusion weights
# are query-time semantics, so they travel with the query. Sending them inline also keeps this
# module's IAM surface at index scope (an AOSS search pipeline is a *collection*-scoped resource,
# needing `aoss:CreateCollectionItems`) and avoids AWS's documented up-to-15-second Serverless
# pipeline propagation delay, during which a freshly created pipeline is not yet resolvable.
#
# Verified against OpenSearch 2.19.1: an inline `search_pipeline` carrying `phase_results_processors`
# is accepted on `POST /<index>/_search` and the processor runs.
public type HybridSearchConfig record {|
    # How each sub-query's scores are rescaled before combination.
    NormalizationTechnique normalization = MIN_MAX;
    # How the two normalized scores are folded into one.
    CombinationTechnique combination = ARITHMETIC_MEAN;
    # The weight given the dense (`knn`) sub-query. Must be in `[0.0, 1.0]`, and must sum with
    # `sparseWeight` to `1.0` — the `normalization-processor` rejects any other weight list. Both
    # rules are checked at construction.
    float denseWeight = 0.5;
    # The weight given the sparse (`neural_sparse`) sub-query. See `denseWeight`.
    float sparseWeight = 0.5;
|};

# Discriminates `RrfFusion` against the other `HybridFusion` variants, and is the only value
# OpenSearch's `score-ranker-processor` accepts for `combination.technique`.
public const RRF = "rrf";

# Reciprocal rank fusion: combines the sub-queries by the *rank* each document reached in each of
# them rather than by its score, sent as an inline `score-ranker-processor`.
#
# Each document scores `sum over sub-queries of 1 / (60 + rank)`, so the sub-queries' raw scores
# never need to be on comparable scales — only their orderings matter. That is the case this
# module's `HYBRID` mode actually presents: a bounded `cosinesimil` score in `[0, 1]` fused against
# an unbounded sparse dot product. `HybridSearchConfig` has to rescale those onto a shared scale
# first, and how well that works depends on each sub-query's score distribution; RRF sidesteps the
# question. OpenSearch's own guidance is to prefer rank-based combination when sub-query scores are
# distributed differently.
#
# The cost is that all score magnitude is discarded: a document that beat the runner-up by a wide
# margin contributes exactly what a document that barely beat it contributes.
#
# # Requirements and hazards
# The `score-ranker-processor` is OpenSearch 2.19 or later — a higher floor than the `hybrid` query
# itself (2.11) and than `HybridSearchConfig`'s `normalization-processor` (2.10). On an older
# cluster the search fails outright rather than degrading.
#
# `VectorMatch.similarityScore` means something different here. An RRF score is a sum of
# reciprocal ranks, not a normalized similarity: it lands in roughly `(0, 0.033]` for two
# sub-queries, it depends on how many documents were retrieved, and it is comparable only against
# other scores from the same query. A threshold tuned against `HybridSearchConfig`'s `(0.0, 1.0]`
# is meaningless against it.
#
# # Why this variant has no settings
# `score-ranker-processor` documents two knobs, and neither can be sent portably across the
# versions this floor admits:
#
# `rank_constant` (the constant added to each rank before its reciprocal is taken, fixed at
# OpenSearch's default of `60` here) moved between releases. neural-search 2.19 and 3.0 read it
# from `combination.parameters.rank_constant`; 3.1 moved it to `combination.rank_constant` and made
# the old location a *hard error*, because the combination technique now validates its parameter
# map and rejects anything but `weights`. Sending either shape therefore breaks on the other side
# of 3.1, and sending both breaks on 3.1+. Only the shape that names no constant at all works
# everywhere.
#
# `combination.parameters.weights` is accepted by every version and honored by none before 3.1 —
# through 2.19 and 3.0 the RRF technique reads it and discards it
# (`RRFScoreCombinationTechnique`: "Not currently using weights for RRF").
#
# Both would therefore be knobs that silently do nothing, or that fail, depending on a version this
# module does not otherwise care about. Use `HybridSearchConfig` when the sub-queries must be
# weighted unequally; it takes weights everywhere and honors them everywhere.
public type RrfFusion record {|
    # Discriminates this variant. Required, so a record literal resolves to exactly one member of
    # `HybridFusion` — and so a bare `{}` keeps resolving to `HybridSearchConfig`, as it did
    # before this variant existed.
    RRF technique;
|};

# Defers fusion to a search pipeline already provisioned on the cluster, sent as
# `?search_pipeline=<name>` instead of an inline object.
#
# This module provisions nothing and validates nothing about the named pipeline: the pipeline
# itself defines the normalization technique, the combination technique and the weights. That is
# why naming one and setting `HybridSearchConfig`'s knobs is not expressible — they are
# alternatives, not layers. Sending both forms on one request is rejected by OpenSearch outright
# ("Both named and inline search pipeline were specified"), and the union is what makes that
# unreachable rather than merely unlikely.
#
# Use this for a least-privilege deployment whose calling principal cannot create pipelines, or to
# share one tuned pipeline across several stores.
public type NamedSearchPipeline record {|
    # The pipeline name, as registered with `PUT /_search/pipeline/<name>`.
    string name;
|};

# The two fusion configurations this module sends inline in the search body: score-based
# combination via the `normalization-processor`, or rank-based combination via the
# `score-ranker-processor`. Both are `phase_results_processors` entries and are mutually
# exclusive — a `hybrid` query is fused once.
#
# Module-private, and deliberately: it exists so the request builder can ask "does this fusion
# travel in the body or as a query parameter?" in one place. A caller never names it — they pick a
# concrete variant of `HybridFusion` — so making it public would add a type to the API surface that
# says nothing a reader of `HybridFusion` does not already know.
type InlineFusion HybridSearchConfig|RrfFusion;

# Where a `HYBRID` store's fusion configuration comes from: inline on every request — normalized
# and combined (`HybridSearchConfig`) or rank-fused (`RrfFusion`) — or a pipeline already on the
# cluster.
public type HybridFusion HybridSearchConfig|RrfFusion|NamedSearchPipeline;

# The kind of search this store performs, together with the settings only that kind honors.
# Chosen once at construction — the `ai:VectorStore` contract carries no mode, so `add` and
# `query` cannot be told which one to use per call.
#
# What varies is more than the query clause: the index-creation mapping differs (a `knn_vector`
# field, a `rank_features` field, or both), and `add` accepts a different `ai:Embedding` member in
# each. Selecting a variant is what makes a setting unreachable everywhere else — `IndexConfig`
# and `normalizeCosineScore` cannot be named on a `SparseSearch`, which has no dense vector and no
# cosine to recover; `sparseVectorFieldName` cannot be named on a `DenseSearch`; and the fusion
# knobs exist only where fusion happens. These were never `init`-time validation rules; they are
# type errors from the start, for the same reason the `Deployment` variants are.
#
# Like `IndexConfig`, the field names and index shape here are honored at index creation only.
# Changing `vectorFieldName` or `sparseVectorFieldName` against an index that already exists does
# not re-map anything — it simply reads and writes fields that are not there.
public type SearchMode DenseSearch|SparseSearch|HybridSearch;

# Dense vector search over a `knn_vector` field, using a `knn` query clause. The original and
# default mode, and the only one that works on every OpenSearch version this module targets
# without a plugin requirement.
public type DenseSearch record {|
    # Discriminates this variant. Required, so a record literal resolves to exactly one member of
    # `SearchMode`.
    ai:DENSE queryMode;
    # Shape of the `knn_vector` field and the index-creation mapping.
    IndexConfig indexConfig;
    # The document field that stores the dense vector.
    string vectorFieldName = "embedding";
    # Whether to convert a `cosinesimil` `_score` back to a `[-1, 1]` cosine similarity
    # (`cos = 2 * score - 1`) before returning it as `VectorMatch.similarityScore`. Applies only
    # when `indexConfig.similarityMetric` is `COSINE`; ignored otherwise.
    #
    # Defaults to `true` so `VectorMatch.similarityScore` means the same thing here as it does in
    # `ai:InMemoryVectorStore`, which returns a true cosine in `[-1, 1]`. OpenSearch's raw
    # `cosinesimil` `_score` is `[0, 1]`, so leaving this off would make a threshold tuned against
    # any other `ai:VectorStore` implementation silently mean something else against this one.
    # Set `false` to get OpenSearch's `_score` through untouched.
    #
    # Reachable here only. A `SPARSE` score is an unbounded dot product with no cosine in it, and
    # a `HYBRID` score has already been normalized to `(0.0, 1.0]` by the fusion pipeline — in
    # both cases the transform would corrupt the score rather than correct it, so the field does
    # not exist there rather than being quietly ignored.
    boolean normalizeCosineScore = true;
    # The HNSW `ef_search` parameter: how many candidate vectors the graph traversal examines
    # before returning the top `k`. The search-time recall/latency dial, and the counterpart to
    # `IndexConfig.efConstruction`/`.m`, which are fixed when the index is created.
    #
    # Unlike those, this is sent with **every query**, as `method_parameters.ef_search` inside the
    # `knn` clause, so changing it takes effect immediately on an existing index. Raising it
    # improves recall at the cost of latency. Must be positive, checked at construction.
    #
    # Leaving it unset (the default) omits `method_parameters` entirely and lets the index's own
    # `index.knn.algo_param.ef_search` apply, whose default is `100`. AWS's own vector-store CDK
    # construct provisions collections with `512`, so that is a reasonable starting point for a
    # recall-sensitive store; measure rather than assume.
    #
    # Needs OpenSearch 2.16 or later, which is when a `knn` query first accepted
    # `method_parameters`; on an older cluster the clause is rejected. Under `LUCENE`, OpenSearch
    # passes the larger of `k` and `ef_search` to the engine rather than honoring `ef_search`
    # directly; under `FAISS` it overrides the index setting outright.
    int? efSearch = ();
|};

# Sparse vector search over a `rank_features` field, using a `neural_sparse` query clause carrying
# precomputed `query_tokens`.
#
# An `ai:SparseVector`'s `indices` become the feature names and its `values` the feature weights,
# so a stored document holds `{"<index>": <weight>}`. No ML model, no ingest pipeline and no
# `index.knn` setting are involved: the weights are computed by the caller's own encoder and sent
# as they are.
#
# # Requirements and hazards
# The `neural_sparse` query clause comes from the `neural-search` plugin, which is bundled in the
# standard OpenSearch distribution and present on AWS managed domains from 2.9. Raw `query_tokens`
# specifically need OpenSearch 2.14 or later; on an older cluster the clause parses but the field
# is rejected.
#
# `rank_features` weights are stored with roughly nine significant bits, so a round-tripped weight
# carries about 0.4% relative error and must never be compared to a locally-held one with exact
# float equality. The field also supports no `exists` query, no aggregation and no sorting; this
# module performs none of those against it.
#
# On OpenSearch Serverless, neural search — which the `neural_sparse` query depends on — is
# available in a subset of AWS regions rather than everywhere Serverless is. In an unsupported
# region the collection is created normally and only the search fails, with nothing in the error
# naming the region as the cause. Check the "Supported AWS Regions" list on AWS's *Configure
# neural search and hybrid search on OpenSearch Serverless* page before choosing `SPARSE` for a
# Serverless deployment.
public type SparseSearch record {|
    # Discriminates this variant.
    ai:SPARSE queryMode;
    # The document field that stores the sparse vector, mapped as `rank_features`.
    string sparseVectorFieldName = "sparse_embedding";
    # The maximum number of non-zero terms a *query* sparse vector may carry, checked before the
    # request is sent.
    #
    # A `neural_sparse` query compiles to one Lucene clause per term and is bounded by the
    # cluster's `indices.query.bool.max_clause_count`, whose default is 1024; exceeding it fails
    # as a shard exception well after the request has left. Raise this only if that cluster
    # setting has also been raised.
    int maxQueryTokens = 1024;
    # Two-phase acceleration for the `neural_sparse` query, sent as an inline `search_pipeline`.
    # Unset by default, which sends no pipeline and scores every matching document against every
    # query token — the behavior on any cluster older than 2.15. See `TwoPhaseAcceleration`.
    TwoPhaseAcceleration? twoPhaseAcceleration = ();
|};

# Hybrid search: a `knn` clause and a `neural_sparse` clause issued as one `hybrid` query, with
# their scores normalized and combined server-side.
#
# The index carries both a `knn_vector` and a `rank_features` field, and `add` requires an
# `ai:HybridVector` so both are always populated — a document missing one half would score zero on
# that sub-query and be penalised by the combination rather than ignored by it.
#
# # Requirements and hazards
# Needs the `neural-search` plugin (see `SparseSearch`) and OpenSearch 2.11 or later for the
# `hybrid` query itself. A `hybrid` query without a normalization pipeline does not fail cleanly:
# it leaks large negative sentinel scores into the results, so this module always arranges one.
#
# Metadata filters are duplicated into each sub-query rather than sent as a single top-level
# `hybrid.filter`, which is OpenSearch 3.0+ only. The two forms are documented as equivalent, and
# duplicating works on every version that has the `hybrid` query at all.
#
# On OpenSearch Serverless, neural search — which both the `neural_sparse` sub-query and the fusion
# processors depend on — is available in a subset of AWS regions rather than everywhere Serverless
# is. In an unsupported region the collection is created normally and only the search fails, with
# nothing in the error naming the region as the cause. Check the "Supported AWS Regions" list on
# AWS's *Configure neural search and hybrid search on OpenSearch Serverless* page before choosing
# `HYBRID` for a Serverless deployment.
#
# # Retrieval depth
# `VectorStoreQuery.topK` doubles as the per-sub-query retrieval depth: it becomes both the search
# `size` and the `k` of the dense sub-query, so `topK: 10` gives fusion exactly ten dense
# candidates to work with. Hybrid relevance normally improves when each sub-query retrieves deeper
# than the final result count, since a document ranked eleventh by one sub-query and first by the
# other cannot be promoted if the first never returned it.
#
# There is no separate depth control here. OpenSearch's own lever is the `hybrid` query's
# `pagination_depth`, which is 2.19 or later and would silently do nothing on the 2.11 floor this
# mode otherwise supports. Until that is exposed, request a larger `topK` than you intend to use
# and truncate the results yourself — the retrieved depth is the only thing that changes.
public type HybridSearch record {|
    # Discriminates this variant.
    ai:HYBRID queryMode;
    # Shape of the `knn_vector` field and the index-creation mapping.
    IndexConfig indexConfig;
    # The document field that stores the dense vector.
    string vectorFieldName = "embedding";
    # The document field that stores the sparse vector, mapped as `rank_features`.
    string sparseVectorFieldName = "sparse_embedding";
    # The maximum number of non-zero terms a query sparse vector may carry. See
    # `SparseSearch.maxQueryTokens`.
    int maxQueryTokens = 1024;
    # The search-time HNSW recall/latency dial for the dense sub-query, sent with every query and
    # unset by default. See `DenseSearch.efSearch` for what it does and what it requires.
    #
    # It matters more here than under `DENSE`. Fusion can only rank what each sub-query retrieved,
    # so a dense sub-query that missed a relevant document cannot be rescued by the sparse one
    # agreeing about it — see the note on retrieval depth above.
    int? efSearch = ();
    # How the dense and sparse sub-query scores are fused: inline on every request — normalized
    # and combined, or rank-fused — or by a pipeline already provisioned on the cluster.
    HybridFusion fusion = {};
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

# Behavioral configuration honored identically under every `SearchMode` and on every deployment
# type: the non-vector field names, bulk sizing, the result-window ceiling, and retries.
#
# Anything whose meaning depends on the kind of search being performed lives on `SearchMode`
# instead — the `knn_vector` shape and the cosine-score transform on `DenseSearch`/`HybridSearch`,
# the `rank_features` field name on `SparseSearch`/`HybridSearch`, the fusion knobs on
# `HybridSearch` alone. That split is what keeps a setting from being reachable where it would do
# nothing.
public type Configuration record {|
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
    # Whether `query` results include the stored vector(s). When `false`, every vector field this
    # store's `SearchMode` declares is excluded from `_source` and `VectorMatch.embedding` comes
    # back empty — a meaningful bandwidth saving at high dimensions and large `topK`, and a larger
    # one under `SPARSE`/`HYBRID`, where a token map can be big. When `true`, what comes back is
    # the vector as storage holds it, which is not necessarily what was written — see
    # "Returned embeddings" on `VectorStore`.
    boolean includeEmbeddingsInResults = true;
    # The maximum number of entries per `_bulk` request issued by `add`. Large `add` calls are
    # chunked into requests of at most this size, to stay under the endpoint's HTTP payload cap.
    int maxBulkSize = 500;
    # The `size` sent when `VectorStoreQuery.topK` requests "all entries" (any `topK < 1`), and the
    # upper bound enforced on an explicit `topK`. OpenSearch's own `k`/`size` ceiling is 10,000;
    # raise this only if the index's `index.max_result_window` setting has also been raised.
    int maxResultWindow = 10000;
    # Whether `init` should create the index if it does not already exist. When `false`, `init`
    # creates nothing — useful for least-privilege deployments where the calling principal has no
    # `CreateIndex` permission.
    #
    # Lives here rather than on a `SearchMode` variant because it governs what `init` *does*, not
    # what the index looks like, and every mode needs the same opt-out.
    #
    # # Hazard when `false`
    # A missing index does not make `add` fail: OpenSearch enables `action.auto_create_index` by
    # default, so the first `_bulk` write creates an index from the document's inferred shape
    # instead. That index carries none of the mapping this module would have built — under
    # `DENSE`/`HYBRID` it has no `index.knn` setting and maps the vector as a plain `float` array
    # rather than a `knn_vector`; under `SPARSE`/`HYBRID` it maps the token map as a nested object
    # rather than `rank_features`. Writes keep succeeding while every `query` fails with a `400`.
    # Recovering means deleting the index and reindexing from source.
    #
    # `verifyOnInit` is what catches this, and it is on by default — so the hazard is reached only
    # when both are switched off. Only set both to `false` against an index you know was
    # provisioned out of band with a compatible mapping.
    boolean createIndexIfNotExists = true;
    # Whether `init` checks that the cluster and the target index can actually serve this store
    # before any `add` or `query` is issued. Two reads, both at construction only:
    #
    # 1. `GET /` for the cluster's OpenSearch version, compared against the floor this store's
    # configuration needs — 2.11 for the `hybrid` query, 2.14 for raw `query_tokens`, 2.15 for
    # `SparseSearch.twoPhaseAcceleration`, 2.16 for `efSearch`, 2.19 for `RrfFusion`. Skipped on
    # both Serverless generations, which do not answer `GET /` at all.
    # 2. `GET /<index>/_mapping` for the vector field(s) this store's `SearchMode` declares,
    # checking that each exists, is the expected type (`knn_vector` / `rank_features`), and — for
    # a `knn_vector` — carries the `dimension` this store was configured with. Skipped when `init`
    # just created the index, which is correct by construction.
    #
    # This is the only thing that connects a store's configuration to the index it is pointed at.
    # Without it a `SPARSE` store aimed at a dense index, a `vectorFieldName` renamed against an
    # existing index, or a `dimension` that drifted are all silent: writes succeed and queries
    # either fail with a bare `400` or return nothing, with no indication of which end is wrong.
    # It is also what turns the `createIndexIfNotExists: false` hazard above from a corrupted index
    # discovered later into an error at construction.
    #
    # # What a failure means
    # A verification that *runs* and disagrees is fatal — `init` returns an `ai:Error` naming the
    # specific mismatch. A verification that cannot run is not: if the calling principal is not
    # permitted to read `/` or `/<index>/_mapping`, the failure is logged as a warning and
    # construction continues, since refusing to build a store that would have worked is the worse
    # outcome. A missing index is the one exception, and is reported as an error.
    #
    # # Cost
    # Up to two extra round trips per construction, and nothing per operation afterwards. Set
    # `false` to skip them — for a least-privilege principal that would only produce warnings, or
    # together with `createIndexIfNotExists: false` for construction that performs no network I/O
    # at all.
    boolean verifyOnInit = true;
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
