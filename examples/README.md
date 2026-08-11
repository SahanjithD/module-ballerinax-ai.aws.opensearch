## Examples

The Ballerina AWS OpenSearch vector store module provides a practical example illustrating usage.
Explore this [example](https://github.com/ballerina-platform/module-ballerinax-ai.aws.opensearch/tree/main/examples).

1. [RAG search](https://github.com/ballerina-platform/module-ballerinax-ai.aws.opensearch/tree/main/examples/rag-search)
   This example shows how to use the AWS OpenSearch vector store APIs to index a handful of text
   chunks and run a similarity query combined with a metadata filter.

## Prerequisites

1. Follow the [module README](../ballerina/README.md#prerequisites) to create an OpenSearch
   deployment (managed domain or Serverless collection) and obtain AWS credentials.

2. For the example, create a `Config.toml` file with your service URL and, if it differs from the
   defaults, your region, index name, and deployment type:

   ```toml
   serviceUrl = "<Your OpenSearch endpoint URL>"
   region = "<Your AWS region>"
   indexName = "<Your index name>"
   deploymentType = "MANAGED_DOMAIN"
   ```

   AWS credentials are resolved via the standard credential chain (environment variables, shared
   config/credentials file, EC2/ECS instance role, and so on) — no credentials need to go in
   `Config.toml`.
