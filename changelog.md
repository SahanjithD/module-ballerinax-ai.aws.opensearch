# Change Log

This file documents all significant changes made to the Ballerina `ai.aws.opensearch` package across releases.

## [Unreleased]

### Added
- Initial implementation of `VectorStore` integration with Amazon OpenSearch, covering
  managed domains and both OpenSearch Serverless generations (Classic and NextGen).
- SigV4 request signing via `ballerinax/aws.auth`, plus HTTP basic auth for managed domains
  with fine-grained access control enabled.
- Automatic index creation with a `knn_vector` mapping, opt-out-able via
  `IndexConfig.createIndexIfNotExists`.
- Dense-vector search only in this release; `SPARSE`/`HYBRID` embeddings return a clear
  `ai:Error`. Planned for a future release: `rank_features` + `neural_sparse` for `SPARSE`,
  and a `hybrid` query with a normalization search pipeline for `HYBRID`, both behind an
  opt-in `Configuration.searchPipeline` field.
