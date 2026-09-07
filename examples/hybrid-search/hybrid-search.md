# Hybrid search

This example indexes a handful of short articles into an AWS OpenSearch vector store using
`HYBRID` search: each article carries both a dense embedding and a sparse term-weight vector, and
a query scores against both, with OpenSearch normalizing and combining the two scores.

The sparse half stands in for a learned sparse encoder such as SPLADE. Its output is sent to
OpenSearch as-is — term ids become `rank_features` token names and their weights become the
feature values — so **no model is deployed on the cluster** and no ingest pipeline is involved.

The toy vectors are chosen so the two sub-queries disagree: "Deploying to Kubernetes" is the
closest dense match to the query, while "Ballerina on Kubernetes" is the strongest term match.
Which one wins is decided by `fusion`, which is what makes the fusion weights worth seeing.

## Prerequisites

`HYBRID` needs the `neural-search` plugin for its `neural_sparse` sub-query. It ships in the
standard OpenSearch distribution and is present on AWS managed domains from **2.9**; the `hybrid`
query itself needs **2.11+**, and raw sparse `query_tokens` need **2.14+**.

Follow the [module README](../../ballerina/README.md#prerequisites) to create an OpenSearch
deployment and obtain AWS credentials, then create a `Config.toml` in this directory:

```toml
serviceUrl = "https://my-domain.us-east-1.es.amazonaws.com"
region = "us-east-1"
indexName = "hybrid-search-example"

[deployment]
deploymentType = "MANAGED_DOMAIN"
```

## Run the example

```bash
bal run
```

Expected output. Scores are the fused, normalized result in `(0.0, 1.0]` — not a cosine, and not
the unbounded dot product a `SPARSE` store returns. The French article is excluded by the
`language == "en"` filter, which is applied to *both* sub-queries:

```
Indexed 4 articles.

Top matches:
  1.0  Ballerina on Kubernetes
  0.4...  Deploying to Kubernetes
  0.2...  Getting started with Ballerina
```

At `denseWeight: 0.4, sparseWeight: 0.6` the term match wins. Flip the weights to
`denseWeight: 0.8, sparseWeight: 0.2` and "Deploying to Kubernetes" takes the top spot instead —
the pipeline carrying those weights is sent inline with every query, so the change takes effect
immediately with nothing to reprovision on the cluster.
