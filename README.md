# Ballerina AWS OpenSearch Vector Store

[![Build](https://github.com/ballerina-platform/module-ballerinax-ai.aws.opensearch/actions/workflows/ci.yml/badge.svg)](https://github.com/ballerina-platform/module-ballerinax-ai.aws.opensearch/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/ballerina-platform/module-ballerinax-ai.aws.opensearch/branch/main/graph/badge.svg)](https://codecov.io/gh/ballerina-platform/module-ballerinax-ai.aws.opensearch)
[![GitHub Last Commit](https://img.shields.io/github/last-commit/ballerina-platform/module-ballerinax-ai.aws.opensearch.svg)](https://github.com/ballerina-platform/module-ballerinax-ai.aws.opensearch/commits/main)

An [`ai:VectorStore`](https://central.ballerina.io/ballerina/ai/latest) implementation for
[Amazon OpenSearch Service](https://aws.amazon.com/opensearch-service/), covering managed
domains and both OpenSearch Serverless generations (Classic and NextGen). See the
[module README](ballerina/README.md) for usage.

## Build options

Execute the commands below to build from the source.

1. To build the package:

   ```bash
   ./gradlew clean build
   ```

2. To run the tests:

   ```bash
   ./gradlew clean test
   ```

3. To run a group of tests (e.g. the live tests, which need real AWS resources):

   ```bash
   ./gradlew clean test -Pgroups=live
   ```

4. To build without the tests:

   ```bash
   ./gradlew clean build -x test
   ```

5. To debug the package with a remote debugger:

   ```bash
   ./gradlew clean build -Pdebug=<port>
   ```

6. To debug with the Ballerina language:

   ```bash
   ./gradlew clean build -PbalJavaDebug=<port>
   ```

7. Publish the generated artifacts to the local Ballerina central repository:

   ```bash
   ./gradlew clean build -PpublishToLocalCentral=true
   ```

8. Publish the generated artifacts to the Ballerina central repository:

   ```bash
   ./gradlew clean build -PpublishToCentral=true
   ```

## Running live tests

The unit test suite needs no AWS access and runs in CI on every PR. A separate `live` test
group exercises this module against real AWS resources (a managed domain and/or Serverless
collections) and is disabled by default. To run it, provide a `Config.toml` under
`ballerina/tests/` (or pass `-C` flags) setting `isLiveServer = true` and the relevant
`managedDomainUrl` / `serverlessClassicUrl` / `serverlessNextGenUrl` values, then:

```bash
./gradlew clean test -Pgroups=live
```

Each deployment type's tests are independently gated on its own URL being non-empty, so you can
exercise just the one you have access to.

## How you can contribute

As an open-source project, Ballerina welcomes contributions from the community. For more
information, see the [contribution guidelines](https://github.com/ballerina-platform/ballerina-lang/blob/master/CONTRIBUTING.md).

## Code of conduct

All contributors are encouraged to read the [Ballerina Code of Conduct](https://ballerina.io/code-of-conduct).

## Useful links

- Discuss code changes of the Ballerina project via [ballerina-dev@googlegroups.com](mailto:ballerina-dev@googlegroups.com).
- Chat live with us via our [Discord server](https://discord.gg/ballerinalang).
- Post technical questions on Stack Overflow with the [#ballerina](https://stackoverflow.com/questions/tagged/ballerina) tag.
