# RAG search

This example indexes a handful of short articles into an AWS OpenSearch vector store, each with a
toy embedding and a `language` metadata field, then runs a similarity query restricted to English
content via a metadata filter.

## Prerequisites

Follow the [module README](../../ballerina/README.md#prerequisites) to create an OpenSearch
deployment and obtain AWS credentials, then create a `Config.toml` in this directory:

```toml
serviceUrl = "https://my-domain.us-east-1.es.amazonaws.com"
region = "us-east-1"
indexName = "rag-search-example"

[deployment]
deploymentType = "MANAGED_DOMAIN"
```

## Run

```bash
bal run
```

Expected output (embeddings and scores will vary slightly, but the French article is excluded by
the `language == "en"` filter regardless of how close its embedding is to the query):

```
Query results:
  - Getting started with Ballerina (score: 0.9...)
  - Deploying to Kubernetes (score: 0.7...)
```
