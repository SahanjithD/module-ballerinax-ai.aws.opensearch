## Overview

[Amazon OpenSearch Service](https://aws.amazon.com/opensearch-service/) is a managed search and
analytics engine that also supports k-NN (approximate nearest-neighbor) vector search. This
package provides an `ai:VectorStore` implementation backed by OpenSearch, so it can be used
anywhere the `ballerina/ai` RAG APIs expect a vector store — including a managed OpenSearch
Service domain and both generations of OpenSearch Serverless (Classic and NextGen).

There is no `ballerinax/opensearch` connector this package wraps: it talks the OpenSearch REST
API directly over `ballerina/http`, signing every request with AWS Signature Version 4 via
[`ballerinax/aws.auth`](https://central.ballerina.io/ballerinax/aws.auth) (or, on a managed
domain with fine-grained access control enabled, HTTP basic authentication against the internal
user database).

### Key features

- Managed OpenSearch Service domains and both OpenSearch Serverless generations (Classic and
  NextGen), each with correct SigV4 signing, index-mapping, and delete semantics
- Automatic index creation with a `knn_vector` mapping, opt-out-able for least-privilege
  deployments
- Full metadata filtering (`==`, `!=`, `>`, `>=`, `<`, `<=`, `in`, `nin`, nested `AND`/`OR`)
  translated to native OpenSearch `bool` queries, pre-filtered inside the `knn` clause
- `time:Utc` metadata round-trips through RFC 3339 strings; `fileSize` round-trips through
  `decimal`
- Exponential-backoff retry on transient failures, and explicit parsing of `_bulk`'s per-item
  partial-failure responses (OpenSearch returns HTTP 200 even when individual items fail)
- GraalVM-compatible

### Scope

This release supports **dense vectors only**. `add`/`query` return an `ai:Error` for a
`SparseVector`/`HybridVector` embedding. Sparse/hybrid support is left as a documented future
extension point (see `changelog.md`) built on OpenSearch's `neural_sparse`/`hybrid` query
features and a search pipeline, once those are needed.

## Prerequisites

### 1. Create an OpenSearch deployment

Choose one of:

- **Managed domain** — an [OpenSearch Service domain](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/createupdatedomains.html),
  endpoint shaped like `https://<domain>.<region>.es.amazonaws.com`.
- **Serverless Classic** — an [OpenSearch Serverless vector search collection](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/serverless-vector-search.html#serverless-vector-classic),
  endpoint shaped like `https://<collection-id>.<region>.aoss.amazonaws.com`.
- **Serverless NextGen** — the current default generation for new Serverless vector collections,
  endpoint shaped like `https://<collection-id>.aoss.<region>.on.aws` (per-collection) or
  `https://<account-id>.aoss.<region>.on.aws` (per-account, see
  [Collection endpoints](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/serverless-collection-endpoints.html)).

The two Serverless generations behave differently at the data-plane level (custom document `_id`
support, refresh interval, and whether the search response includes the vector by default), and
AWS does not expose a way to detect which generation a collection is from the data plane — you
must tell this module which one you have via `DeploymentType`.

### 2. Obtain credentials

For a managed domain or Serverless (SigV4), the module authenticates via any AWS credential
source supported by `ballerinax/aws.auth`: static keys, a named profile, an assumed role, web
identity federation, IAM Identity Center (SSO), or `DEFAULT_CREDENTIALS` (the standard AWS
credential provider chain — environment, shared config, EC2/ECS instance role, and so on).
`DEFAULT_CREDENTIALS` is the recommended choice in production.

For a managed domain with fine-grained access control and the internal user database enabled,
you may instead use `BasicAuth` (username/password). Serverless has no basic-auth path — a
domain access policy that accepts IAM principals requires SigV4-signed requests, so you cannot
mix a username/password with IAM credentials on the same request
([AWS: fine-grained access control](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/fgac.html)).

### 3. Grant the required IAM permissions

**Managed domain** — a domain access policy granting `es:ESHttp*` on the domain ARN:

```json
{
  "Effect": "Allow",
  "Principal": {"AWS": "arn:aws:iam::<account-id>:root"},
  "Action": "es:ESHttp*",
  "Resource": "arn:aws:es:<region>:<account-id>:domain/<domain>/*"
}
```

**Serverless** — a [data access policy](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/serverless-data-access.html)
granting at least:

```json
[
  {
    "Rules": [
      {
        "ResourceType": "index",
        "Resource": ["index/<collection>/<index>"],
        "Permission": ["aoss:CreateIndex", "aoss:DescribeIndex", "aoss:WriteDocument", "aoss:ReadDocument"]
      }
    ],
    "Principal": ["arn:aws:iam::<account-id>:role/<your-role>"]
  }
]
```

Drop `aoss:CreateIndex` if you set `IndexConfig.createIndexIfNotExists = false` and manage the
index yourself.

## Quickstart

### Step 1: Import the module

```ballerina
import ballerinax/ai.aws.opensearch;
```

### Step 2: Initialize the vector store

```ballerina
import ballerina/ai;
import ballerinax/ai.aws.opensearch;
import ballerinax/aws.auth;

opensearch:VectorStore vectorStore = check new (
    serviceUrl = "https://my-domain.us-east-1.es.amazonaws.com",
    region = "us-east-1",
    indexName = "my-knowledge-base",
    deploymentConfig = {
        deploymentType: opensearch:MANAGED_DOMAIN,
        auth: auth:DEFAULT_CREDENTIALS
    },
    storeConfig = {indexConfig: {dimension: 1536}}
);
```

`deploymentConfig` is required, and selects one of three record types. Each exposes only the settings its
flavour actually honors — so a setting that does not apply cannot be named at all:

```ballerina
// Managed domain: the only flavour with a selectable engine, basic auth, or a forced refresh.
deploymentConfig = {
    deploymentType: opensearch:MANAGED_DOMAIN,
    auth: {username: "...", password: "..."},
    engine: opensearch:LUCENE,
    refreshOnWrite: true
}

// Serverless Classic: credentials only. Faiss is the sole engine it supports, and its refresh
// interval is fixed, so neither is a choice.
deploymentConfig = {
    deploymentType: opensearch:SERVERLESS_CLASSIC,
    auth: auth:DEFAULT_CREDENTIALS
}

// Serverless NextGen: the only flavour exposing vector quantization.
deploymentConfig = {
    deploymentType: opensearch:SERVERLESS_NEXTGEN,
    auth: auth:DEFAULT_CREDENTIALS,
    collectionName: "my-collection",   // or collectionId; needed on a per-account endpoint
    vectorMode: opensearch:IN_MEMORY,
    compressionLevel: opensearch:COMPRESSION_1X
}
```

`auth` is required on every variant. It is deliberately not defaulted to
`auth:DEFAULT_CREDENTIALS`: a store that silently picks up the ambient AWS credential chain hides a
misconfiguration until it surfaces as a 403 from a principal nobody intended to use. Pass
`auth:DEFAULT_CREDENTIALS` to opt into that chain explicitly.

Settings honored identically everywhere — `dimension`, `similarityMetric`, `efConstruction`, `m`,
field names, bulk sizing, retries — live on `storeConfig` instead, so switching deployments carries
them across untouched.

By default (`IndexConfig.createIndexIfNotExists = true`), `init` creates the index with the
correct `knn_vector` mapping if it does not already exist.

### Step 3: Add, query, and delete vector entries

```ballerina
check vectorStore.add([
    {
        id: "doc-1",
        embedding: [0.12, 0.98, ...],
        chunk: {'type: "text-chunk", content: "Ballerina is a cloud-native programming language.",
                metadata: {"language": "en"}}
    }
]);

ai:VectorMatch[] results = check vectorStore.query({
    embedding: [0.11, 0.97, ...],
    topK: 5,
    filters: {filters: [{key: "language", operator: ai:EQUAL, value: "en"}]}
});

check vectorStore.delete("doc-1");
```

`vectorStore` satisfies `ai:VectorStore`, so it plugs directly into `ai:KnowledgeBase` and any
other `ballerina/ai` API that accepts a vector store.

## Configuration

### Deployment types and their behavioral differences

| | `MANAGED_DOMAIN` | `SERVERLESS_CLASSIC` | `SERVERLESS_NEXTGEN` |
|---|---|---|---|
| SigV4 signing name | `es` | `aoss` | `aoss` |
| `BasicAuth` allowed | yes | no | no |
| `add` semantics | upsert | **append-only** | upsert |
| `delete` | single `_bulk` call | search-then-bulk-delete | single `_bulk` call |
| Write visibility | immediate with `refreshOnWrite`, else normal refresh | ~60 s, unforceable | ~10 s, unforceable |
| Index mapping | `method` block with `engine` configurable | `method` block, forced `faiss` | `method` block without `engine`; `space_type` at the field's top level |
| HNSW tuning (`efConstruction`/`m`) | yes | yes | yes |
| Vector quantization | no | no | yes |

On `SERVERLESS_CLASSIC`, a custom document `_id` cannot be written, so re-`add`ing the same
logical id creates a duplicate document rather than upserting, and `delete` must first search for
the internal `_id`s carrying that logical id (via the `doc_id` field this module always writes)
before it can delete them. A document added less than the collection's refresh interval ago will
not be found by that search and will survive the delete — this is a real limitation of the
platform, not a bug in this module.

> [!NOTE]
> **On `SERVERLESS_CLASSIC`, a successful `init` does not mean the index can be queried yet.**
> When `init` creates the index, `_mapping` and `_settings` report it as present immediately while
> `_search`/`_count` against it still fail with `index_not_found_exception` for roughly ten
> seconds. It is a propagation delay, not a missing shard: the index becomes searchable on its own
> with no write and no intervention. **Writes are unaffected** and succeed immediately, so code
> that `add`s before it queries never sees this. `init` deliberately does not wait the delay out,
> which would add up to ten seconds to every Classic construction including the callers who write
> first; if you must query straight after constructing the store, retry an
> `index_not_found_exception` for a few seconds before treating it as fatal.
> `SERVERLESS_NEXTGEN` is searchable at once.

### Authentication

Credentials are part of the `deploymentConfig` argument, since which mechanisms are available depends on
the flavour.

```ballerina
// Static keys
deploymentConfig = {..., auth: {accessKeyId: "...", secretAccessKey: "..."}}

// The standard AWS credential chain (recommended)
deploymentConfig = {..., auth: auth:DEFAULT_CREDENTIALS}

// HTTP basic auth — requires FGAC + an internal user database, and is accepted only by
// `ManagedDomainDeployment`. Neither Serverless variant's `auth` field admits this type.
deploymentConfig = {deploymentType: opensearch:MANAGED_DOMAIN, auth: {username: "...", password: "..."}}
```

### Index configuration (`IndexConfig`)

| Field | Default | Notes |
|---|---|---|
| `dimension` | *(required)* | No client-side ceiling — AWS's own docs disagree on the maximum (10,000 vs. 16,000); the server enforces its own limit |
| `similarityMetric` | `ai:COSINE` | Maps to OpenSearch `space_type`: `COSINE`→`cosinesimil`, `EUCLIDEAN`→`l2`, `DOT_PRODUCT`→`innerproduct` |
| `efConstruction` | `128` | HNSW build-time accuracy/speed trade-off. Honored on all three deployment types |
| `m` | `16` | HNSW max bi-directional links per node. Honored on all three deployment types |
| `createIndexIfNotExists` | `true` | When `false`, `init` performs **no network I/O at all** — including no existence check. See the warning below before setting it |

`engine`, `compressionLevel`, and `vectorMode` are not here: they live on the `Deployment` variant
that honors them (`ManagedDomainDeployment.engine`, and `compressionLevel`/`vectorMode` on
`ServerlessNextGenDeployment`), so naming one against a deployment that ignores it does not
compile.


> [!WARNING]
> **`createIndexIfNotExists = false` against an index that does not exist corrupts silently.**
> `init` does no existence check, and a missing index does not make `add` fail — OpenSearch ships
> with `action.auto_create_index: true`, so the first `_bulk` write creates an index from the
> document's inferred shape. That index has no `index.knn` setting and maps the vector as a plain
> `float` array instead of a `knn_vector`. Writes keep succeeding; every `query` then fails with a
> `400`. Recovery means deleting the index and reindexing from source. Only set this to `false`
> against an index you know was provisioned out of band with a compatible mapping.

> [!NOTE]
> **Vectors are quantized by default on `SERVERLESS_NEXTGEN`.** NextGen auto-configures the
> `knn_vector` field, and what it configures is `on_disk` storage with `32x` compression — the
> same index created on a managed domain stores full-precision vectors. Nothing in the request or
> the response announces this; the stored mapping is the only place it shows up. To store vectors
> uncompressed, set both `vectorMode: IN_MEMORY` and `compressionLevel: COMPRESSION_1X`
> (`ON_DISK` rejects `1x`, and is caught at construction). Leaving both unset preserves the
> platform default rather than substituting one of this module's own.
>
> Both fields are declared on `ServerlessNextGenDeployment` alone, so they are simply not
> expressible against a managed domain or a Classic collection.

> [!WARNING]
> **Index-shape settings apply only when the index is created.** `dimension`, `similarityMetric`,
> `efConstruction`, `m`, `engine`, `compressionLevel` and `vectorMode` are sent in exactly one
> request — the `PUT /<index>` that `init` issues when the index does not yet exist. After that,
> `init` sees the index and returns without building a mapping, so **editing any of them has no
> effect and produces no error.**
>
> Nothing in a later request restates them: a `_bulk` document carries a vector, and a `knn` query
> carries a vector and `k`. Neither mentions `space_type`, `engine` or `compression_level`, so the
> server has nothing to disagree with. A changed `similarityMetric` keeps returning scores computed
> under the original metric; a `compressionLevel` set to opt out of NextGen's default quantization
> leaves the vectors quantized.
>
> `dimension` is not an exception — it is never checked against the embeddings this module sends. A
> mismatched `dimension` beside unchanged embeddings simply works and stays wrong silently. An error
> appears only when the *embeddings themselves* change length, at the first `add` rather than at
> `init`.
>
> To change the shape of an existing index, reindex into a new one. This module does not update
> mappings.

### Other configuration (`Configuration`)

| Field | Default | Notes |
|---|---|---|
| `vectorFieldName` | `"embedding"` | |
| `contentFieldName` | `"content"` | |
| `idFieldName` | `"doc_id"` | The field carrying the logical entry id; always written, and the only identity handle on `SERVERLESS_CLASSIC` |
| `metadataFieldName` | `"metadata"` | Set to `""` to address bare `<key>` paths — for pointing this module at a pre-existing index with a flat schema. A metadata key that would then collide with a reserved field name (`embedding`, `content`, `doc_id`, `chunk_type`) is rejected with an `ai:Error` rather than silently overwriting that field |
| `includeEmbeddingsInResults` | `true` | Set `false` to exclude the vector from `_source` — a meaningful bandwidth saving at high dimensions and large `topK` |
| `normalizeCosineScore` | `true` | Converts a `cosinesimil` `_score` back to `[-1, 1]` (`cos = 2 * score - 1`), so scores mean the same thing as in `ai:InMemoryVectorStore`. Set `false` for OpenSearch's raw `_score`; see below |
| `maxBulkSize` | `500` | Entries per `_bulk` request; `add` chunks larger batches automatically |
| `maxResultWindow` | `10000` | The `size` used when `topK < 1` ("return all"), and the ceiling enforced on an explicit `topK` |
| `retryConfig` | exponential backoff, 3 retries | Applied to `429`/`408`/`5xx` responses |

### Similarity score range

Under `ai:COSINE`, `VectorMatch.similarityScore` is a true cosine in `[-1, 1]` — the same range
`ai:InMemoryVectorStore` returns, so a threshold tuned against one store means the same thing
against the other. That conversion (`cos = 2 * score - 1`) is what `normalizeCosineScore = true`
does, and it is the default for exactly that reason.

Set `normalizeCosineScore = false` to get OpenSearch's raw `_score` instead. Its range is
space-dependent: `cosinesimil` gives `[0, 1]`, `l2` gives `(0, 1]`, and `innerproduct` is
piecewise. The setting applies only under `ai:COSINE`; the other metrics always pass through.

When a query has no embedding (a metadata-only filter, or neither embedding nor filters),
OpenSearch's constant score is not a similarity at all, so `similarityScore` is always `0.0` for
those matches regardless of this setting.

## Limitations

- **Dense vectors only** in this release — see Scope above.
- **`topK` above `maxResultWindow`** is rejected; raise `maxResultWindow` (and the index's own
  `index.max_result_window` setting) if you need more than 10,000 results.
- **A pre-existing index** this module did not create must map `metadata.*` (or, under a flat
  schema, every metadata field) as `keyword`, not the default analyzed `text` — otherwise `==`
  and `in` filters on string metadata will silently return zero hits.
- **`SERVERLESS_CLASSIC` delete** can fail loudly (rather than silently under-deleting) if a
  single logical id has accumulated more duplicate documents than `maxResultWindow` can see in
  one lookup — raise `maxResultWindow` and retry.
- **A `SERVERLESS_CLASSIC` index is not searchable for ~10 s after `init` creates it**, though it
  is writable immediately — see the note under "Deployment types" above.
- **Fractional metadata values change Ballerina type on a round trip.** A custom metadata key
  written as a `float` (`{"rating": 4.25}`) is read back as a `decimal`, because JSON has one
  number type and Ballerina's parser maps every non-integral value to `decimal`. The value is
  exact; only the basic type differs, so `readBack["rating"] == 4.25` is `false`. `fileSize` is
  the one numeric field restored to its declared type, because `ai:Metadata` declares it
  `decimal`. This matches `ai.pinecone`'s behaviour, so metadata semantics stay identical when
  swapping vector stores behind the `ai:VectorStore` interface.

## Examples

The `ai.aws.opensearch` package provides a practical example illustrating real-world use.

1. [RAG search](https://github.com/ballerina-platform/module-ballerinax-ai.aws.opensearch/tree/main/examples/rag-search) —
   indexes a handful of text chunks and answers a similarity query with a metadata filter.
