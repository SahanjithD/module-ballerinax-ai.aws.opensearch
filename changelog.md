# Change Log

This file documents all significant changes made to the Ballerina `ai.aws.opensearch` package across releases.

## [Unreleased]

### Added
- `ServerlessNextGenDeployment.collectionName` and `.collectionId`, which become the signed
  `x-amz-aoss-collection-name`/`x-amz-aoss-collection-id` headers a per-account NextGen endpoint
  needs to know which collection a request is for. They replace the free-form
  `Configuration.additionalHeaders` map, which offered the same capability untyped and on every
  deployment type, including the two where no such header means anything. Setting both is rejected
  at construction.
- The AWS request id is now carried on every mapped error as a `requestId` detail field, read from
  `x-amzn-RequestId` (AOSS) or `x-amz-request-id` (managed domains). It appears nowhere in the
  response body, so a failure that has to be escalated to AWS support was previously
  unattributable after the fact. This module read no response headers at all before.
- Initial implementation of `VectorStore` integration with Amazon OpenSearch, covering
  managed domains and both OpenSearch Serverless generations (Classic and NextGen).
- SigV4 request signing via `ballerinax/aws.auth`, plus HTTP basic auth for managed domains
  with fine-grained access control enabled.
- Automatic index creation with a `knn_vector` mapping, opt-out-able via
  `IndexConfig.createIndexIfNotExists`.
- The underlying HTTP client is pinned to HTTP/1.1 with chunking disabled, regardless of the
  caller-supplied `httpConfig`. Ballerina's HTTP/2 outbound path alters an `application/json`
  body between SigV4 signing and the wire, which `SERVERLESS_NEXTGEN` rejects with
  `403 Request Content Checksum Verification Failed`; a chunked body carries no `Content-Length`
  for the signature to be verified against. Whether the same alteration happens on Classic and
  managed domains and simply goes unverified there was not established — pinning HTTP/1.1
  everywhere sidesteps the question.
- `Deployment`, a closed union of `ManagedDomainDeployment`, `ServerlessClassicDeployment`, and
  `ServerlessNextGenDeployment`, replacing `init`'s separate `deploymentType` and `auth`
  parameters. Each variant declares only the settings its flavour honors, so combinations that
  were previously rejected at construction — `BasicAuth` on a Serverless collection, a non-Faiss
  `engine` on Classic, `refreshOnWrite` off a managed domain, quantization off NextGen — are now
  compile errors rather than runtime ones. This also gives form-driven tooling a variant to select
  instead of a flat parameter list where most fields do not apply. `IndexConfig.engine` moved to
  `ManagedDomainDeployment.engine`; `IndexConfig.compressionLevel`/`vectorMode` moved to
  `ServerlessNextGenDeployment`; `Configuration.refreshOnWrite` moved to
  `ManagedDomainDeployment`. `init`'s `deployment` parameter is required, as is `auth` on every
  variant — neither defaults, so a store cannot be constructed that quietly resolves credentials
  from the ambient AWS chain when the caller named none. Pass `auth:DEFAULT_CREDENTIALS` to opt
  into that chain explicitly.
- `ServerlessNextGenDeployment.compressionLevel` and `.vectorMode`, which surface the `knn_vector`
  field's `compression_level` and `mode`. A NextGen collection provisions every vector field as
  `on_disk`/`32x` when nothing is configured, so vectors are quantized by default with no
  diagnostic; `IN_MEMORY` with `COMPRESSION_1X` opts out. Both are unset by default, which leaves
  the server's behavior exactly as it was.
- Dense-vector search only in this release; `SPARSE`/`HYBRID` embeddings return a clear
  `ai:Error`. Planned for a future release: `rank_features` + `neural_sparse` for `SPARSE`,
  and a `hybrid` query with a normalization search pipeline for `HYBRID`, both behind an
  opt-in `Configuration.searchPipeline` field.

### Changed
- `init`'s `config` parameter is renamed `storeConfig` (display label "Store Configuration"), so it
  reads as a pair with `deploymentConfig` rather than as a second unqualified "Configuration".
- `Configuration.normalizeCosineScore` now defaults to `true`. `ai:InMemoryVectorStore` returns a
  true cosine in `[-1, 1]`; OpenSearch's raw `cosinesimil` `_score` is `[0, 1]`. With the old
  default, a similarity threshold tuned against any other `ai:VectorStore` implementation silently
  meant something different against this one. Set `false` for the raw `_score`.
- `Engine.NMSLIB` is removed. NMSLIB was deprecated in OpenSearch 2.16 and removed in 3.0, where
  creating a new NMSLIB index is blocked outright; it also never supported efficient k-NN
  pre-filtering. `Engine` is now `FAISS`/`LUCENE`, and is selectable only on a managed domain.
- `Configuration.additionalHeaders` is removed in favour of the two typed NextGen fields above.

### Fixed
- A `_bulk` batch that indexes documents is no longer retried after a `5xx` or a connection failure
  on `SERVERLESS_CLASSIC`. That deployment rejects a custom document `_id`, so the action line
  carries none, and a retry after the server had already applied part of the batch appended those
  documents a second time — silently, since `add` then reported success. Everywhere else the
  action line names an `_id`, which makes the retry an idempotent upsert, so retries are unchanged
  there. Delete batches are retried on every deployment type: a repeated delete is idempotent and
  `not_found` is already ignored.
- A `Retry-After` header on a `429`/`503` is now honoured instead of being ignored in favour of the
  computed backoff delay, clamped to `RetryConfig.maxDelay` so an implausible value cannot park the
  calling thread. Only the delay-seconds form is read; the RFC 9110 HTTP-date form is not sent by
  OpenSearch or the AOSS proxy, and misreading one as a duration would be worse than falling back
  to the curve.
- HNSW tuning (`IndexConfig.efConstruction`/`m`) now reaches `SERVERLESS_NEXTGEN`. The module
  previously sent NextGen no `method` block at all, because the block it built always carried
  `engine`, which the AOSS proxy rejects with a flat
  `400 Field parameter 'engine' is not supported`. `engine` turns out to be the only part it
  rejects: a block of `{"name": "hnsw", "parameters": {...}}` is accepted with the parameters
  stored intact, and NextGen fills in `engine: faiss` itself in the stored mapping, alongside any
  `mode`/`compression_level` sent at the field's top level. The block is now built without
  `engine` there and emitted on every deployment type, so `efConstruction`/`m` are honored on all
  three rather than being silently dropped on one. The mapping sent to `MANAGED_DOMAIN` and
  `SERVERLESS_CLASSIC` is unchanged.
- `_bulk` item failures are now attributed to the caller's entry by the item's position in the
  response rather than by the `_id` the server reports. `_bulk` returns items in submission order,
  while the response `_id` is the caller's id only where a custom `_id` can be written: on
  `SERVERLESS_CLASSIC` the action line carries no `_id` and the server generates one, so `add`
  previously reported failures against identifiers like `1%3A0%3AjNUSV6ABrlsmLW-dso53` and left a
  caller who submitted 500 entries unable to tell which of theirs had failed. The
  `SERVERLESS_CLASSIC` delete path had the same defect against the internal `_id`s its lookup
  discovers, and now carries the logical id through the lookup to report that instead.
- Error messages now include the nested cause that explains the failure instead of stopping at the
  wrapper's reason. A query vector of the wrong dimension surfaced as
  `OpenSearch request failed with status 400: all shards failed`, with the sentence naming the
  expected dimension discarded; it now reads
  `... all shards failed: [query_shard_exception] failed to create query: Query vector has invalid dimension: 4. Dimension should be: 8`.
  A failed `_bulk` item nests its cause differently — under `caused_by`, with no `root_cause` — and
  is unwrapped the same way. The nested classification is also carried as an
  `openSearchRootCauseType` error detail; `openSearchErrorType` still reports the response's own
  `error.type`.

### Documentation
- `IndexConfig` and the README now state that index-shape settings apply only at index creation.
  `dimension`, `similarityMetric`, `efConstruction`, `m`, `engine`, `compressionLevel` and
  `vectorMode` are sent in the single `PUT /<index>` that `init` issues when the index does not
  exist; afterwards `init` returns early and editing them has no effect and raises no error,
  because no later request restates them. `dimension` is not an exception — it is never compared
  against the embeddings this module sends, so a mismatched value beside unchanged embeddings stays
  wrong silently, and an error appears only when the embeddings themselves change length.
- The note claiming OpenSearch accepts `compression_level` beside a `method` block while silently
  emptying the block's `parameters` is withdrawn: tested against OpenSearch 2.19.1, the combination
  is accepted with `ef_construction`/`m` stored intact, and an incompatible pairing is rejected
  loudly instead (`"faiss" does not support "4x" compression`). Keeping the quantization fields
  `SERVERLESS_NEXTGEN`-only is therefore this module's scoping decision, not a server constraint.
- Documented that on `SERVERLESS_CLASSIC`, a successful construction does not mean the index can
  be queried yet: a newly created index answers `_mapping` and `_settings` immediately while
  `_search`/`_count` still fail with `index_not_found_exception` for roughly ten seconds. It is a
  propagation delay that clears on its own with no write, and writes are unaffected. `init`
  deliberately does not wait it out, which would add that delay to every Classic construction
  including the callers who write first; code that queries immediately after construction should
  retry that error briefly instead.
