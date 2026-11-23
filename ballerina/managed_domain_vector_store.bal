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
import ballerinax/aws;

# Vector store for a managed Amazon OpenSearch Service domain.
@display {label: "OpenSearch Managed Vector Store"}
public isolated class ManagedDomainVectorStore {
    *ai:VectorStore;

    private final VectorStoreCore core;

    // `auth` is required rather than defaulted: silently falling back to the ambient credential
    // chain hides a misconfiguration until it shows up as a 403 for the wrong principal.
    // `createIndexIfNotExists` is an `init` parameter rather than a store setting so that creating
    // a resource on the domain is a visible choice; with it on, a mistyped index name silently
    // creates an empty index.
    # Initializes the store and, by default, creates and checks the index.
    #
    # + serviceUrl - The domain endpoint, e.g. `https://search-my-domain-abc123.us-east-1.es.amazonaws.com`
    # + region - The AWS region of the domain
    # + indexName - The index this store reads and writes
    # + auth - AWS credentials or a fine-grained access control user (`BasicAuth`);
    # `auth:DEFAULT_CREDENTIALS` uses the default chain
    # + createIndexIfNotExists - Creates the index if it doesn't exist. Turn off to fail on a missing
    # or mistyped index name instead
    # + searchMode - Dense, sparse or hybrid search and the index shape it uses. Defaults to dense
    # search over 1536-dimensional vectors
    # + storeConfig - Shared store settings plus the engine and refresh behavior
    # + httpConfig - HTTP client settings. `httpVersion` and `http1Settings.chunking` are always set
    # to `HTTP_1_1` and `CHUNKING_NEVER`, which SigV4 signing needs
    # + return - An `ai:Error` if validation, index creation or verification fails
    public isolated function init(
            @display {label: "Service URL"} string serviceUrl,
            @display {label: "Region"} aws:Region|string region,
            @display {label: "Index Name"} string indexName,
            @display {label: "Authentication"} ManagedDomainAuthConfig auth,
            @display {label: "Create Index If Missing"} boolean createIndexIfNotExists = true,
            @display {label: "Search Mode"} SearchMode searchMode = DEFAULT_SEARCH_MODE,
            @display {label: "Store Configuration"} ManagedDomainStoreConfiguration storeConfig = {},
            @display {label: "HTTP Configuration"} http:ClientConfiguration httpConfig = {})
            returns ai:Error? {
        ManagedDomainDeployment deployment = {
            auth,
            engine: storeConfig.engine,
            refreshOnWrite: storeConfig.refreshOnWrite
        };
        self.core = check new (serviceUrl, region, indexName, deployment, searchMode,
            sharedStoreConfiguration(storeConfig), httpConfig, createIndexIfNotExists
        );
    }

    # Adds entries in `_bulk` batches of at most `maxBulkSize`. An entry with an existing `id` is
    # overwritten; entries without an `id` get a UUID.
    #
    # + entries - Entries whose embeddings match this store's search mode
    # + return - An `ai:Error` naming the failing entries, if any
    public isolated function add(ai:VectorEntry[] entries) returns ai:Error? => self.core.add(entries);

    # Searches by embedding, by metadata filter, or both.
    #
    # + query - The query. Any `topK < 1` returns all entries, up to `maxResultWindow`
    # + return - The matches, or an `ai:Error`
    public isolated function query(ai:VectorStoreQuery query) returns ai:VectorMatch[]|ai:Error =>
        self.core.query(query);

    # Deletes entries by id. Ids that do not exist are ignored.
    #
    # + ids - The id or ids to delete
    # + return - An `ai:Error` naming the ids that failed, if any
    public isolated function delete(string|string[] ids) returns ai:Error? => self.core.delete(ids);
}

// Spelled out field by field: a closed record with extra fields does not convert to
// `StoreConfiguration` on its own.
# The shared part of a managed domain store configuration.
#
# + config - The managed domain store configuration
# + return - The same settings without the managed-only fields
isolated function sharedStoreConfiguration(ManagedDomainStoreConfiguration config) returns StoreConfiguration => {
    contentFieldName: config.contentFieldName,
    idFieldName: config.idFieldName,
    metadataFieldName: config.metadataFieldName,
    includeEmbeddingsInResults: config.includeEmbeddingsInResults,
    maxBulkSize: config.maxBulkSize,
    maxResultWindow: config.maxResultWindow,
    retryConfig: config.retryConfig
};
