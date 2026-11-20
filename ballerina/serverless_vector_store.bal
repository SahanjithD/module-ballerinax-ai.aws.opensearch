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

# Vector store for an Amazon OpenSearch Serverless vector search collection.
@display {label: "OpenSearch Serverless Vector Store"}
public isolated class ServerlessVectorStore {
    *ai:VectorStore;

    private final VectorStoreCore core;

    # Initializes the store and, by default, creates and checks the index.
    #
    # + serviceUrl - The collection endpoint: `https://abc123.us-east-1.aoss.amazonaws.com` (Classic) or
    # `https://abc123.aoss.us-east-1.on.aws` (NextGen)
    # + region - The AWS region of the collection
    # + indexName - The index this store reads and writes
    # + deployment - The collection's generation, its credentials and its generation-specific
    # settings
    # + searchMode - Dense, sparse or hybrid search and the index shape it uses. Defaults to dense
    # search over 1536-dimensional vectors
    # + storeConfig - Settings shared by every search mode
    # + httpConfig - HTTP client settings. `httpVersion` and `http1Settings.chunking` are always set
    # to `HTTP_1_1` and `CHUNKING_NEVER`, which SigV4 signing needs
    # + return - An `ai:Error` if validation, index creation or verification fails
    public isolated function init(
            @display {label: "Service URL"} string serviceUrl,
            @display {label: "Region"} aws:Region|string region,
            @display {label: "Index Name"} string indexName,
            @display {label: "Collection Configuration"} ServerlessDeployment deployment,
            @display {label: "Search Mode"} SearchMode searchMode = DEFAULT_SEARCH_MODE,
            @display {label: "Store Configuration"} StoreConfiguration storeConfig = {},
            @display {label: "HTTP Configuration"} http:ClientConfiguration httpConfig = {})
            returns ai:Error? {
        self.core = check new (serviceUrl, region, indexName, deployment, searchMode, storeConfig,
            httpConfig
        );
    }

    // Classic has no custom `_id`, so `add` cannot overwrite there.
    # Adds entries in `_bulk` batches of at most `maxBulkSize`. On NextGen an entry with an existing
    # `id` is overwritten; on Classic it is added again as a duplicate.
    #
    # + entries - Entries whose embeddings match this store's search mode
    # + return - An `ai:Error` naming the failing entries, if any
    public isolated function add(ai:VectorEntry[] entries) returns ai:Error? => self.core.add(entries);

    # Searches by embedding, by metadata filter, or both. New entries become visible after the
    # collection's refresh interval.
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
