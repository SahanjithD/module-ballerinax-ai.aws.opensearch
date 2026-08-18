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
    deploymentType = opensearch:MANAGED_DOMAIN,
    auth = auth:DEFAULT_CREDENTIALS,
    config = {indexConfig: {dimension: 1536}}
);
```

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
| Index mapping | explicit `method` block, `engine` configurable | explicit `method` block, forced `faiss` | no `method` block; auto-configured |

On `SERVERLESS_CLASSIC`, a custom document `_id` cannot be written, so re-`add`ing the same
logical id creates a duplicate document rather than upserting, and `delete` must first search for
the internal `_id`s carrying that logical id (via the `doc_id` field this module always writes)
before it can delete them. A document added less than the collection's refresh interval ago will
not be found by that search and will survive the delete — this is a real limitation of the
platform, not a bug in this module.

### Authentication

```ballerina
// Static keys
config = {..., auth: {accessKeyId: "...", secretAccessKey: "..."}}

// The standard AWS credential chain (recommended)
config = {..., auth: auth:DEFAULT_CREDENTIALS}

// HTTP basic auth (MANAGED_DOMAIN only, requires FGAC + internal user database)
config = {..., auth: {username: "...", password: "..."}}
```

### Index configuration (`IndexConfig`)

| Field | Default | Notes |
|---|---|---|
| `dimension` | *(required)* | No client-side ceiling — AWS's own docs disagree on the maximum (10,000 vs. 16,000); the server enforces its own limit |
| `similarityMetric` | `ai:COSINE` | Maps to OpenSearch `space_type`: `COSINE`→`cosinesimil`, `EUCLIDEAN`→`l2`, `DOT_PRODUCT`→`innerproduct` |
| `engine` | `FAISS` | Forced to `FAISS` on `SERVERLESS_CLASSIC`; ignored on `SERVERLESS_NEXTGEN` |
| `efConstruction` | `128` | HNSW build-time accuracy/speed trade-off |
| `m` | `16` | HNSW max bi-directional links per node |
| `createIndexIfNotExists` | `true` | When `false`, `init` performs **no network I/O at all** — including no existence check. See the warning below before setting it |


> [!WARNING]
> **`createIndexIfNotExists = false` against an index that does not exist corrupts silently.**
> `init` does no existence check, and a missing index does not make `add` fail — OpenSearch ships
> with `action.auto_create_index: true`, so the first `_bulk` write creates an index from the
> document's inferred shape. That index has no `index.knn` setting and maps the vector as a plain
> `float` array instead of a `knn_vector`. Writes keep succeeding; every `query` then fails with a
> `400`. Recovery means deleting the index and reindexing from source. Only set this to `false`
> against an index you know was provisioned out of band with a compatible mapping.

### Other configuration (`Configuration`)

| Field | Default | Notes |
|---|---|---|
| `vectorFieldName` | `"embedding"` | |
| `contentFieldName` | `"content"` | |
| `idFieldName` | `"doc_id"` | The field carrying the logical entry id; always written, and the only identity handle on `SERVERLESS_CLASSIC` |
| `metadataFieldName` | `"metadata"` | Set to `""` to address bare `<key>` paths — for pointing this module at a pre-existing index with a flat schema. A metadata key that would then collide with a reserved field name (`embedding`, `content`, `doc_id`, `chunk_type`) is rejected with an `ai:Error` rather than silently overwriting that field |
| `includeEmbeddingsInResults` | `true` | Set `false` to exclude the vector from `_source` — a meaningful bandwidth saving at high dimensions and large `topK` |
| `normalizeCosineScore` | `false` | Converts a `cosinesimil` `_score` back to `[-1, 1]` (`cos = 2 * score - 1`); see below |
| `refreshOnWrite` | `false` | `?refresh=wait_for` on `add`. `MANAGED_DOMAIN` only |
| `maxBulkSize` | `500` | Entries per `_bulk` request; `add` chunks larger batches automatically |
| `maxResultWindow` | `10000` | The `size` used when `topK < 1` ("return all"), and the ceiling enforced on an explicit `topK` |
| `additionalHeaders` | `{}` | Extra headers, **SigV4-signed** — needed for a Serverless NextGen per-account endpoint's `x-amz-aoss-collection-name`/`x-amz-aoss-collection-id` header |
| `retryConfig` | exponential backoff, 3 retries | Applied to `429`/`408`/`5xx` responses |

### Similarity score range

`VectorMatch.similarityScore` passes OpenSearch's raw `_score` through by default, and its range
is space-dependent: `l2` gives `(0, 1]`, `innerproduct` is piecewise, and `cosinesimil` gives
`[0, 1]` rather than the `[-1, 1]` a caller may expect from "cosine similarity". Set
`normalizeCosineScore = true` (with `similarityMetric: ai:COSINE`) to convert it back to
`[-1, 1]`. When a query has no embedding (a metadata-only filter, or neither embedding nor
filters), OpenSearch's constant score is not a similarity at all, so `similarityScore` is always
`0.0` for those matches.

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
