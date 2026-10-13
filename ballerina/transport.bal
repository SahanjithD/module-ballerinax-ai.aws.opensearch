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

// AWS docs: requests to an OpenSearch Service domain are signed with the service name `es`.
const string SIGNING_NAME_MANAGED = "es";

// AWS docs: OpenSearch Serverless requests are signed with the service name `aoss`.
const string SIGNING_NAME_SERVERLESS = "aoss";

// Signs the exact bytes it sends and never re-serializes them, since a payload-hash mismatch fails
// as an unexplained 403.
# HTTP client, SigV4 signing, retries and error mapping for the OpenSearch REST API.
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
    private final boolean retryWrites;

    isolated function init(string serviceUrl, aws:Region|string region, Deployment deployment,
            RetryConfig retryConfig, http:ClientConfiguration httpConfig) returns ai:Error? {
        self.host = check extractHost(serviceUrl);
        self.signingName = deployment is ManagedDomainDeployment ? SIGNING_NAME_MANAGED : SIGNING_NAME_SERVERLESS;
        self.region = region;
        self.additionalHeaders = buildCollectionHeaders(deployment).cloneReadOnly();
        self.retryConfig = retryConfig.cloneReadOnly();
        // Serverless has a fixed refresh interval, so only a managed domain carries this.
        self.refreshOnWrite = deployment is ManagedDomainDeployment && deployment.refreshOnWrite;
        // Without a custom `_id` (Classic), retrying a partly applied batch appends duplicates.
        // Elsewhere the `_id` makes a retry an idempotent upsert.
        self.retryWrites = deployment !is ServerlessClassicDeployment;

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

        // Both Ballerina defaults break SigV4 with a misleading 403: HTTP/2 can alter a signed JSON
        // body, and a chunked body has no `Content-Length` to check the signature against.
        // Copied then assigned, since a spread cannot be combined with a key it also supplies.
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

    # Reads the cluster's OpenSearch version from `GET /`. Not available on Serverless.
    #
    # + return - The version, e.g. `2.19.1`, `()` if the body carries none, or an `ai:Error` if the
    # request failed
    isolated function clusterVersion() returns string?|ai:Error {
        http:Response resp = check self.sendSigned("GET", "/", {}, {}, []);
        if resp.statusCode != 200 {
            return mapErrorResponse(resp);
        }
        json|error payload = resp.getJsonPayload();
        if payload is error {
            return ();
        }
        json|error number = payload.version.number;
        return number is string ? number : ();
    }

    // The response is keyed by the concrete index name, which differs from the requested one for
    // an alias, so the entry is taken by position.
    # Reads the index's mapped `properties` from `GET /<index>/_mapping`.
    #
    # + indexName - The index or alias to read
    # + return - The `properties` object, `()` if the index declares none, or an `ai:Error`
    # (a `404` if the index does not exist)
    isolated function indexMappingProperties(string indexName) returns map<json>?|ai:Error {
        http:Response resp = check self.sendSigned("GET", "/" + indexName + "/_mapping", {}, {}, []);
        if resp.statusCode != 200 {
            return mapErrorResponse(resp);
        }
        json|error payload = resp.getJsonPayload();
        if payload !is map<json> {
            return error("Failed to parse the '_mapping' response body",
                    payload is error ? payload : error("the body was not a JSON object"));
        }
        foreach json indexEntry in payload {
            json|error properties = indexEntry.mappings.properties;
            if properties is map<json> {
                return properties;
            }
        }
        return ();
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

    # Sends a `POST /_bulk`, adding `?refresh=wait_for` when `refreshOnWrite` is set.
    #
    # + body - The NDJSON bytes to sign and send; an empty body sends nothing
    # + isIndexingBatch - Whether the batch indexes rather than deletes. Indexing batches are not
    # retried where a custom `_id` cannot be written, since a replay would duplicate documents
    # + return - The parsed response, which may carry per-item failures, or an `ai:Error`
    isolated function bulk(byte[] body, boolean isIndexingBatch = false) returns BulkResponse|ai:Error {
        if body.length() == 0 {
            return {errors: false, items: []};
        }
        map<string> queryParams = {};
        if self.refreshOnWrite {
            queryParams["refresh"] = "wait_for";
        }
        boolean retryable = self.retryWrites || !isIndexingBatch;
        http:Response resp = check self.sendSigned("POST", "/_bulk", queryParams,
                {"content-type": "application/x-ndjson"}, body, retryable);
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
    # + searchPipeline - A named pipeline, sent as `?search_pipeline=`. Never set alongside an
    # inline pipeline in `body`
    # + return - The parsed search response, or an `ai:Error` on failure
    isolated function search(string indexName, json body, string? searchPipeline = ())
            returns SearchResponse|ai:Error {
        byte[] payload = body.toJsonString().toBytes();
        map<string> queryParams = {};
        if searchPipeline is string {
            queryParams["search_pipeline"] = searchPipeline;
        }
        http:Response resp = check self.sendSigned("POST", "/" + indexName + "/_search", queryParams,
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

    # Releases the AWS credential provider's threads and connections. A no-op under `BasicAuth`.
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

    # Signs and sends a request, retrying transport failures and retryable statuses.
    #
    # + method - The HTTP method, upper case
    # + path - The request path, unencoded, starting with `/`
    # + queryParams - Query parameters, unencoded
    # + extraHeaders - Extra headers to sign and send, e.g. `content-type`
    # + payload - The request body, signed and sent as is
    # + retryable - Whether the request may be re-sent; `false` where a replay could duplicate data
    # + return - The response (any status), or an `ai:Error` if signing or every attempt failed
    private isolated function sendSigned(string method, string path, map<string> queryParams,
            map<string> extraHeaders, byte[] payload, boolean retryable = true)
            returns http:Response|ai:Error {
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
        return self.executeWithRetry(method, fullPath, request, retryable);
    }

    private isolated function executeWithRetry(string method, string path, http:Request request,
            boolean retryable) returns http:Response|ai:Error {
        int attempt = 0;
        decimal delay = self.retryConfig.initialDelay;
        while true {
            http:Response|http:ClientError result = self.httpClient->execute(method, path, request);
            boolean shouldRetry = retryable && attempt < self.retryConfig.maxRetries &&
                    (result is http:ClientError || isRetryableStatus(result.statusCode));
            if !shouldRetry {
                if result is http:ClientError {
                    return error(string `Request to OpenSearch failed after ${attempt + 1} attempt(s)`, result);
                }
                return result;
            }
            attempt += 1;
            // Prefer the server's `Retry-After`, capped at `maxDelay` so a bad value cannot park
            // the thread.
            decimal sleepFor = delay;
            if result is http:Response {
                decimal? retryAfter = retryAfterSeconds(result);
                if retryAfter is decimal {
                    sleepFor = retryAfter < self.retryConfig.maxDelay ? retryAfter : self.retryConfig.maxDelay;
                }
            }
            runtime:sleep(sleepFor);
            decimal nextDelay = delay * self.retryConfig.backoffFactor;
            delay = nextDelay < self.retryConfig.maxDelay ? nextDelay : self.retryConfig.maxDelay;
        }
    }
}

# The signed headers that name the target collection on a per-account NextGen endpoint.
#
# + deployment - The target deployment
# + return - The headers to sign and send on every request, empty when none apply
isolated function buildCollectionHeaders(Deployment deployment) returns map<string> {
    if deployment !is ServerlessNextGenDeployment {
        return {};
    }
    string? collectionName = deployment?.collectionName;
    if collectionName is string {
        return {"x-amz-aoss-collection-name": collectionName};
    }
    string? collectionId = deployment?.collectionId;
    if collectionId is string {
        return {"x-amz-aoss-collection-id": collectionId};
    }
    return {};
}

// Only the seconds form is read: neither OpenSearch nor AOSS sends an HTTP-date, and misreading one
// would sleep far too long.
# Reads `Retry-After` as a number of seconds.
#
# + response - The HTTP response to read
# + return - The delay in seconds, or `()` when the header is absent or not a usable duration
isolated function retryAfterSeconds(http:Response response) returns decimal? {
    string|error header = response.getHeader("retry-after");
    if header !is string {
        return ();
    }
    int|error seconds = int:fromString(header.trim());
    if seconds !is int || seconds < 0 {
        return ();
    }
    return <decimal>seconds;
}

// AOSS sends `x-amzn-RequestId`, managed domains `x-amz-request-id`. Header lookup ignores case.
# The AWS request id from a response, which AWS support asks for.
#
# + response - The HTTP response to read
# + return - The request id, or `()` when the response carries none
isolated function extractRequestId(http:Response response) returns string? {
    foreach string name in ["x-amzn-requestid", "x-amz-request-id", "x-amzn-request-id"] {
        string|error value = response.getHeader(name);
        if value is string && value.trim().length() > 0 {
            return value;
        }
    }
    return ();
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

// Not `url:encode`, which writes a space as `+`: the sent query must match what was signed byte
// for byte, or AWS answers with an opaque 403.
# Percent-encodes a string per RFC 3986, as SigV4 query canonicalization requires.
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

// `openSearchErrorType` is the response's own type and `openSearchRootCauseType` the nested one
// when it differs. `requestId` is what AWS support needs and appears nowhere in the body.
# Maps a non-2xx response to an `ai:Error` with OpenSearch's reason and a hint for the status.
#
# + response - The non-2xx HTTP response
# + return - The mapped error, carrying `status`, `requestId`, `openSearchErrorType` and
# `openSearchRootCauseType` as detail fields, each nil when the response did not supply it
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

    // Absent values are attached as nil, which reads the same as a missing key.
    return error ai:Error(message, status = status, requestId = extractRequestId(response),
            openSearchErrorType = errorType, openSearchRootCauseType = rootCauseType);
}

# The hint appended to an error message for a given status.
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
            return " (the index may not exist; see 'Configuration.createIndexIfNotExists')";
        }
        413 => {
            return " (the request payload was too large; see 'Configuration.maxBulkSize')";
        }
        _ => {
            return "";
        }
    }
}
