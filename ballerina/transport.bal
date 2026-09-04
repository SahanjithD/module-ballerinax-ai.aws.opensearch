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
import ballerina/http;
import ballerina/lang.runtime;
import ballerinax/aws;
import ballerinax/aws.auth;

# The SigV4 signing name for an OpenSearch Service (managed) domain.
const string SIGNING_NAME_MANAGED = "es";

# The SigV4 signing name for OpenSearch Serverless (both generations).
const string SIGNING_NAME_SERVERLESS = "aoss";

# Owns the HTTP client, request signing, retry, and error mapping for talking to the OpenSearch
# REST API directly — there is no `ballerinax/opensearch` connector to wrap. Every public
# operation signs the exact bytes it sends and never re-serializes between signing and sending,
# since a payload-hash mismatch fails as an unexplained 403. For the same reason the HTTP client
# is pinned to HTTP/1.1 with chunking disabled — see the comment in `init`.
isolated class OpenSearchTransport {
    private final http:Client httpClient;
    private final string host;
    private final string signingName;
    private final aws:Region|string region;
    private final auth:CredentialProvider? credProvider;
    private final string? basicAuthHeader;
    private final map<string> & readonly additionalHeaders;
    private final RetryConfig & readonly retryConfig;
    private final boolean refreshOnWrite;

    isolated function init(string serviceUrl, aws:Region|string region, Deployment deployment,
            map<string> additionalHeaders, RetryConfig retryConfig, http:ClientConfiguration httpConfig)
            returns ai:Error? {
        self.host = check extractHost(serviceUrl);
        self.signingName = deployment is ManagedDomainDeployment ? SIGNING_NAME_MANAGED : SIGNING_NAME_SERVERLESS;
        self.region = region;
        self.additionalHeaders = additionalHeaders.cloneReadOnly();
        self.retryConfig = retryConfig.cloneReadOnly();
        // Only a managed domain can force a refresh, and only that variant carries the field —
        // the Serverless generations have a fixed, unforceable refresh interval.
        self.refreshOnWrite = deployment is ManagedDomainDeployment && deployment.refreshOnWrite;

        Auth authConfig = deployment.auth;
        if authConfig is BasicAuth {
            string credentials = string `${authConfig.username}:${authConfig.password}`;
            self.basicAuthHeader = string `Basic ${credentials.toBytes().toBase64()}`;
            self.credProvider = ();
        } else {
            auth:CredentialProvider|auth:CredentialResolutionError provider = new (authConfig);
            if provider is error {
                return error("Failed to initialize the AWS credential provider", provider);
            }
            self.credProvider = provider;
            self.basicAuthHeader = ();
        }

        // HTTP/1.1 and an unchunked body are correctness requirements here, not preferences, so
        // both are pinned over whatever the caller passed.
        //
        // Ballerina's HTTP/2 outbound path alters an `application/json` body between signing and
        // the wire. On `SERVERLESS_NEXTGEN`, which verifies the SigV4 payload checksum, every
        // `_search` fails with `403 Request Content Checksum Verification Failed` while the
        // byte-identical request over HTTP/1.1 succeeds; `POST /_bulk` (`application/x-ndjson`)
        // is unaffected on the same connection. What was measured on Classic and managed domains
        // is only that HTTP/2 works there — whether the body is left intact, or is altered the
        // same way and accepted because those endpoints do not verify the checksum, was not
        // tested. Pinning HTTP/1.1 everywhere makes the question moot rather than answering it.
        //
        // Chunked transfer encoding breaks independently: SigV4 signs the whole body and the
        // signature is matched against a `Content-Length`-framed request, which a chunked body
        // does not provide. This bites any hand-signed AWS call whose payload crosses the
        // chunking threshold — for this module, any `_bulk` batch of realistic dimensionality.
        // Ballerina's defaults (`HTTP_2_0`, `CHUNKING_AUTO`) are wrong on both counts, and the
        // failure they produce is a 403 whose text points at IAM.
        // (Copy-then-assign, not a spread with overriding fields: Ballerina rejects a record
        // constructor whose explicit key duplicates one supplied by the spread.)
        http:ClientHttp1Settings http1Settings = {...httpConfig.http1Settings};
        http1Settings.chunking = http:CHUNKING_NEVER;
        http:ClientConfiguration effectiveConfig = {...httpConfig};
        effectiveConfig.httpVersion = http:HTTP_1_1;
        effectiveConfig.http1Settings = http1Settings;

        http:Client|http:ClientError httpClient = new (serviceUrl, effectiveConfig);
        if httpClient is error {
            return error("Failed to initialize the OpenSearch HTTP client", httpClient);
        }
        self.httpClient = httpClient;
    }

    # Checks whether an index already exists via `HEAD /<index>`.
    #
    # + indexName - The index to check
    # + return - `true`/`false`, or an `ai:Error` on an unexpected response
    isolated function indexExists(string indexName) returns boolean|ai:Error {
        http:Response resp = check self.sendSigned("HEAD", "/" + indexName, {}, {}, []);
        if resp.statusCode == 200 {
            return true;
        }
        if resp.statusCode == 404 {
            return false;
        }
        return mapErrorResponse(resp);
    }

    # Creates an index with the given mapping via `PUT /<index>`.
    #
    # + indexName - The index to create
    # + mapping - The index-creation body (settings + mappings)
    # + return - An `ai:Error` on failure, otherwise `()`
    isolated function createIndex(string indexName, json mapping) returns ai:Error? {
        byte[] payload = mapping.toJsonString().toBytes();
        http:Response resp = check self.sendSigned("PUT", "/" + indexName, {},
                {"content-type": "application/json"}, payload);
        if resp.statusCode >= 200 && resp.statusCode < 300 {
            return;
        }
        return mapErrorResponse(resp);
    }

    # Issues a `POST /_bulk` request. Requests `?refresh=wait_for` when configured, and only on a
    # managed domain — both Serverless generations have a fixed, unforceable refresh interval.
    #
    # + body - The exact NDJSON bytes to sign and send; an empty body is a no-op
    # + return - The parsed bulk response (which may itself carry per-item failures — see
    # `extractIndexFailures`/`extractDeleteFailures`), or an `ai:Error` on a transport/HTTP failure
    isolated function bulk(byte[] body) returns BulkResponse|ai:Error {
        if body.length() == 0 {
            return {errors: false, items: []};
        }
        map<string> queryParams = {};
        if self.refreshOnWrite {
            queryParams["refresh"] = "wait_for";
        }
        http:Response resp = check self.sendSigned("POST", "/_bulk", queryParams,
                {"content-type": "application/x-ndjson"}, body);
        if resp.statusCode != 200 {
            return mapErrorResponse(resp);
        }
        json|error payload = resp.getJsonPayload();
        if payload is error {
            return error("Failed to parse the '_bulk' response body", payload);
        }
        BulkResponse|error parsed = payload.cloneWithType(BulkResponse);
        if parsed is error {
            return error("Failed to parse the '_bulk' response body", parsed);
        }
        return parsed;
    }

    # Issues a `POST /<index>/_search` request.
    #
    # + indexName - The index to search
    # + body - The search request body
    # + return - The parsed search response, or an `ai:Error` on failure
    isolated function search(string indexName, json body) returns SearchResponse|ai:Error {
        byte[] payload = body.toJsonString().toBytes();
        http:Response resp = check self.sendSigned("POST", "/" + indexName + "/_search", {},
                {"content-type": "application/json"}, payload);
        if resp.statusCode != 200 {
            return mapErrorResponse(resp);
        }
        json|error respPayload = resp.getJsonPayload();
        if respPayload is error {
            return error("Failed to parse the '_search' response body", respPayload);
        }
        SearchResponse|error parsed = respPayload.cloneWithType(SearchResponse);
        if parsed is error {
            return error("Failed to parse the '_search' response body", parsed);
        }
        return parsed;
    }

    # Releases the AWS credential provider's background refresh threads and STS/SSO HTTP
    # connections. A no-op when the store was constructed with `BasicAuth`.
    #
    # + return - An `ai:Error` if releasing the resources fails, otherwise `()`
    isolated function close() returns ai:Error? {
        auth:CredentialProvider? provider = self.credProvider;
        if provider is () {
            return;
        }
        auth:Error? result = provider.close();
        if result is error {
            return error("Failed to close the AWS credential provider", result);
        }
    }

    # Signs and sends a single request, retrying on transport failures and retryable HTTP status
    # codes. The NDJSON/JSON body is built once by the caller and its exact bytes are passed to
    # both the signer and the request — never re-serialized in between.
    #
    # + method - The HTTP method, upper case
    # + path - The request path, unencoded, starting with `/`
    # + queryParams - Query parameters, unencoded
    # + extraHeaders - Extra headers to sign and send, e.g. `content-type`
    # + payload - The exact request body bytes
    # + return - The HTTP response (any status code), or an `ai:Error` if signing or the transport
    # itself fails after exhausting retries
    private isolated function sendSigned(string method, string path, map<string> queryParams,
            map<string> extraHeaders, byte[] payload) returns http:Response|ai:Error {
        map<string> headersToSign = {};
        foreach [string, string] [name, value] in self.additionalHeaders.entries() {
            headersToSign[name] = value;
        }
        foreach [string, string] [name, value] in extraHeaders.entries() {
            headersToSign[name] = value;
        }

        map<string> signedHeaders;
        string? basicAuthHeader = self.basicAuthHeader;
        if basicAuthHeader is string {
            signedHeaders = headersToSign.clone();
            signedHeaders["authorization"] = basicAuthHeader;
        } else {
            auth:CredentialProvider? provider = self.credProvider;
            if provider is () {
                return error("Internal error: no AWS credential provider is configured");
            }
            auth:Credentials|auth:CredentialResolutionError creds = provider.getCredentials();
            if creds is error {
                return error("Failed to resolve AWS credentials", creds);
            }
            map<string>|auth:SigningError signed = auth:getSignedHeaders({
                                                                             method,
                                                                             host: self.host,
                                                                             path,
                                                                             queryParams,
                                                                             headers: headersToSign,
                                                                             payload
                                                                         }, creds, self.region, self.signingName);
            if signed is error {
                return error("Failed to sign the OpenSearch request with AWS Signature Version 4", signed);
            }
            signedHeaders = signed;
        }

        http:Request request = new;
        foreach [string, string] [name, value] in signedHeaders.entries() {
            request.setHeader(name, value);
        }
        if payload.length() > 0 {
            request.setBinaryPayload(payload);
        }

        string fullPath = queryParams.length() == 0 ? path : path + "?" + check buildQueryString(queryParams);
        return self.executeWithRetry(method, fullPath, request);
    }

    private isolated function executeWithRetry(string method, string path, http:Request request)
            returns http:Response|ai:Error {
        int attempt = 0;
        decimal delay = self.retryConfig.initialDelay;
        while true {
            http:Response|http:ClientError result = self.httpClient->execute(method, path, request);
            boolean shouldRetry = attempt < self.retryConfig.maxRetries &&
                    (result is http:ClientError || isRetryableStatus(result.statusCode));
            if !shouldRetry {
                if result is http:ClientError {
                    return error(string `Request to OpenSearch failed after ${attempt + 1} attempt(s)`, result);
                }
                return result;
            }
            attempt += 1;
            runtime:sleep(delay);
            decimal nextDelay = delay * self.retryConfig.backoffFactor;
            delay = nextDelay < self.retryConfig.maxDelay ? nextDelay : self.retryConfig.maxDelay;
        }
    }
}

# Determines whether an HTTP status code warrants a retry with exponential backoff.
#
# + status - The HTTP status code
# + return - `true` for `429`, `408`, `500`, `502`, `503`, `504`
isolated function isRetryableStatus(int status) returns boolean {
    return status == 429 || status == 408 || status == 500 || status == 502 || status == 503 || status == 504;
}

# Extracts the host from a service URL, for use as the SigV4 `host` header value.
#
# + serviceUrl - The full service URL, e.g. `https://my-domain.us-east-1.es.amazonaws.com`
# + return - The host, e.g. `my-domain.us-east-1.es.amazonaws.com`, or an `ai:Error` if
# `serviceUrl` has no recognizable scheme or host
isolated function extractHost(string serviceUrl) returns string|ai:Error {
    string withoutScheme;
    if serviceUrl.startsWith("https://") {
        withoutScheme = serviceUrl.substring(8);
    } else if serviceUrl.startsWith("http://") {
        withoutScheme = serviceUrl.substring(7);
    } else {
        return error(string `'serviceUrl' must start with 'https://' or 'http://', got: ${serviceUrl}`);
    }
    int? slashIndex = withoutScheme.indexOf("/");
    string host = slashIndex is int ? withoutScheme.substring(0, slashIndex) : withoutScheme;
    if host.trim().length() == 0 {
        return error(string `'serviceUrl' does not contain a host: ${serviceUrl}`);
    }
    return host;
}

# Percent-encodes and joins query parameters into a query string.
#
# + queryParams - The unencoded query parameters
# + return - The encoded query string, without a leading `?`, or an `ai:Error` if a name or value
# contains an invalid Unicode code point
isolated function buildQueryString(map<string> queryParams) returns string|ai:Error {
    string[] parts = [];
    foreach [string, string] [name, value] in queryParams.entries() {
        parts.push(string `${check percentEncode(name)}=${check percentEncode(value)}`);
    }
    return string:'join("&", ...parts);
}

# Percent-encodes a string per RFC 3986, the encoding AWS Signature Version 4 canonicalization
# requires for query parameters — leaving only unreserved characters (`A-Z a-z 0-9 - . _ ~`)
# unescaped. Deliberately not `ballerina/url:encode`, whose `application/x-www-form-urlencoded`
# behavior differs (e.g. a space becomes `+`, not `%20`): the query string sent over the wire must
# byte-for-byte match what the signer canonicalized, or AWS rejects the request with an opaque
# 403. There is only ever one query parameter
# in this module today (`refresh=wait_for`, which needs no encoding either way), but a correct
# encoder here removes the landmine for `Configuration.additionalHeaders`-style extensions.
#
# + value - The raw (unencoded) value
# + return - The percent-encoded value, or an `ai:Error` if `value` contains an invalid code point
isolated function percentEncode(string value) returns string|ai:Error {
    string result = "";
    foreach int codepoint in value.toCodePointInts() {
        string|error ch = string:fromCodePointInt(codepoint);
        if ch is error {
            return error(string `Invalid Unicode code point in query parameter: ${codepoint}`, ch);
        }
        if isUnreservedCodepoint(codepoint) {
            result += ch;
        } else {
            foreach byte b in ch.toBytes() {
                string hex = (<int>b).toHexString().toUpperAscii();
                result += string `%${hex.length() == 1 ? "0" + hex : hex}`;
            }
        }
    }
    return result;
}

# Checks whether a Unicode code point is one of RFC 3986's unreserved characters.
#
# + codepoint - The code point to check
# + return - `true` for `A-Z`, `a-z`, `0-9`, `-`, `.`, `_`, `~`
isolated function isUnreservedCodepoint(int codepoint) returns boolean {
    return (codepoint >= 65 && codepoint <= 90) || (codepoint >= 97 && codepoint <= 122) ||
            (codepoint >= 48 && codepoint <= 57) ||
            codepoint == 45 || codepoint == 46 || codepoint == 95 || codepoint == 126;
}

# Maps a non-2xx OpenSearch HTTP response to an `ai:Error`, extracting `error.type`/`error.reason`
# from the response body when present (AOSS edge 403s are sometimes not JSON), and attaching an
# actionable hint per status code.
#
# The reason is taken from `describeErrorDetail` rather than from `error.reason` directly. On a
# shard-level failure `error.reason` is the wrapper's — a query vector of the wrong dimension
# surfaces as `all shards failed`, and the sentence naming the expected dimension is nested a
# level down. Reporting only the wrapper leaves the caller with a message that says something went
# wrong and nothing about what.
#
# `openSearchErrorType` stays the response's own `error.type`, so an existing check against it
# keeps working; `openSearchRootCauseType` carries the nested classification, and is present only
# when the response actually wraps a different one.
#
# + response - The non-2xx HTTP response
# + return - The mapped error, carrying `status` and (when available) `openSearchErrorType` and
# `openSearchRootCauseType` as detail fields
isolated function mapErrorResponse(http:Response response) returns ai:Error {
    int status = response.statusCode;
    string reason = "";
    string? errorType = ();
    string? rootCauseType = ();
    json|error body = response.getJsonPayload();
    if body is json {
        OpenSearchErrorResponse|error parsed = body.cloneWithType(OpenSearchErrorResponse);
        if parsed is OpenSearchErrorResponse {
            ErrorDetail? detail = parsed?.'error;
            if detail is ErrorDetail {
                errorType = detail.'type;
                rootCauseType = specificErrorType(detail);
                reason = describeErrorDetail(detail);
            }
        }
    }
    string message = reason.length() > 0
        ? string `OpenSearch request failed with status ${status}: ${reason}`
        : string `OpenSearch request failed with status ${status}`;
    message += hintForStatus(status);

    if errorType is string {
        if rootCauseType is string {
            return error ai:Error(message, status = status, openSearchErrorType = errorType,
                    openSearchRootCauseType = rootCauseType);
        }
        return error ai:Error(message, status = status, openSearchErrorType = errorType);
    }
    return error ai:Error(message, status = status);
}

# Builds the actionable, status-specific hint appended to a mapped error message.
#
# + status - The HTTP status code
# + return - The hint text (including leading space), or `""` for a status with no specific hint
isolated function hintForStatus(int status) returns string {
    match status {
        401|403 => {
            return " (on a managed domain, check IAM 'es:ESHttp*' permissions and FGAC role " +
                    "mapping; on Serverless, check the data access policy — separate from the " +
                    "IAM policy — and that the signing name is 'aoss')";
        }
        404 => {
            return " (the index may not exist; see 'IndexConfig.createIndexIfNotExists')";
        }
        413 => {
            return " (the request payload was too large; see 'Configuration.maxBulkSize')";
        }
        _ => {
            return "";
        }
    }
}
