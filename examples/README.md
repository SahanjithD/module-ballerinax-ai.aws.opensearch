## Examples

The Ballerina AWS OpenSearch vector store module provides practical examples illustrating usage.
Explore these [examples](https://github.com/ballerina-platform/module-ballerinax-ai.aws.opensearch/tree/main/examples).

1. [RAG search](https://github.com/ballerina-platform/module-ballerinax-ai.aws.opensearch/tree/main/examples/rag-search)
   This example shows how to use the AWS OpenSearch vector store APIs to index a handful of text
   chunks and run a similarity query combined with a metadata filter.

2. [Hybrid search](https://github.com/ballerina-platform/module-ballerinax-ai.aws.opensearch/tree/main/examples/hybrid-search)
   This example shows `HYBRID` mode, where each entry carries both a dense embedding and a sparse
   term-weight vector and OpenSearch fuses the two scores. It also shows how the fusion weights
   change which article wins.

## Prerequisites

1. Follow the [module README](../ballerina/README.md#prerequisites) to create an OpenSearch
   managed domain and obtain AWS credentials. Both examples use `ManagedDomainVectorStore`.

2. For the example, create a `Config.toml` file with your service URL and, if they differ from the
   defaults, your region and index name:

   ```toml
   serviceUrl = "<Your OpenSearch endpoint URL>"
   region = "<Your AWS region>"
   indexName = "<Your index name>"
   ```

   The examples pass `auth:DEFAULT_CREDENTIALS` in `main.bal`, which uses the standard AWS
   credential chain (environment variables, shared config/credentials file, EC2/ECS instance role,
   and so on), so no credentials go in `Config.toml`.
