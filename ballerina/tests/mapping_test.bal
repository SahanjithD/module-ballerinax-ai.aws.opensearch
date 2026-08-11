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
import ballerina/test;

isolated function baseConfig(ai:SimilarityMetric metric = ai:COSINE) returns Configuration => {
    indexConfig: {dimension: 1536, similarityMetric: metric}
};

isolated function asMap(json value) returns map<json>|error => value.ensureType();

@test:Config
isolated function testSpaceTypeMapping() {
    test:assertEquals(toSpaceType(ai:COSINE), "cosinesimil");
    test:assertEquals(toSpaceType(ai:EUCLIDEAN), "l2");
    test:assertEquals(toSpaceType(ai:DOT_PRODUCT), "innerproduct");
}

@test:Config
isolated function testMappingHasKnnSettingEnabled() returns error? {
    map<json> mapping = check asMap(buildIndexMapping(baseConfig(), MANAGED_DOMAIN));
    map<json> settings = check asMap(mapping["settings"]);
    map<json> index = check asMap(settings["index"]);
    test:assertEquals(index["knn"], true);
}

isolated function vectorFieldOf(json mapping) returns map<json>|error {
    map<json> root = check asMap(mapping);
    map<json> mappings = check asMap(root["mappings"]);
    map<json> properties = check asMap(mappings["properties"]);
    return asMap(properties["embedding"]);
}

@test:Config
isolated function testManagedDomainEmitsMethodBlock() returns error? {
    json mapping = buildIndexMapping(baseConfig(), MANAGED_DOMAIN);
    map<json> vectorField = check vectorFieldOf(mapping);
    test:assertEquals(vectorField["type"], "knn_vector");
    test:assertEquals(vectorField["dimension"], 1536);
    test:assertFalse(vectorField.hasKey("space_type"),
            "space_type must not appear at the top level when a method block is present");
    map<json> method = check asMap(vectorField["method"]);
    test:assertEquals(method["name"], "hnsw");
    test:assertEquals(method["engine"], "faiss");
    test:assertEquals(method["space_type"], "cosinesimil");
}

@test:Config
isolated function testServerlessClassicEmitsMethodBlock() returns error? {
    json mapping = buildIndexMapping(baseConfig(), SERVERLESS_CLASSIC);
    map<json> vectorField = check vectorFieldOf(mapping);
    map<json> method = check asMap(vectorField["method"]);
    test:assertEquals(method["engine"], "faiss");
    test:assertEquals(method["name"], "hnsw");
}

@test:Config
isolated function testServerlessNextGenOmitsMethodBlock() returns error? {
    json mapping = buildIndexMapping(baseConfig(), SERVERLESS_NEXTGEN);
    map<json> vectorField = check vectorFieldOf(mapping);
    test:assertEquals(vectorField["type"], "knn_vector");
    test:assertEquals(vectorField["dimension"], 1536);
    test:assertEquals(vectorField["space_type"], "cosinesimil");
    test:assertFalse(vectorField.hasKey("method"), "NextGen must not receive an explicit 'method' block");
}

@test:Config
isolated function testSpaceTypePropagatesPerDeploymentType() returns error? {
    DeploymentType[] deploymentTypes = [MANAGED_DOMAIN, SERVERLESS_CLASSIC, SERVERLESS_NEXTGEN];
    foreach DeploymentType deploymentType in deploymentTypes {
        map<json> vectorField = check vectorFieldOf(buildIndexMapping(baseConfig(ai:EUCLIDEAN), deploymentType));
        json spaceType;
        if deploymentType == SERVERLESS_NEXTGEN {
            spaceType = vectorField["space_type"];
        } else {
            map<json> method = check asMap(vectorField["method"]);
            spaceType = method["space_type"];
        }
        test:assertEquals(spaceType, "l2", string `unexpected space_type for ${deploymentType}`);
    }
}

@test:Config
isolated function testDynamicTemplatesAlwaysPresentForNestedMetadata() returns error? {
    map<json> mapping = check asMap(buildIndexMapping(baseConfig(), MANAGED_DOMAIN));
    map<json> mappings = check asMap(mapping["mappings"]);
    json[] templates = check mappings["dynamic_templates"].ensureType();
    test:assertEquals(templates.length(), 1);
    map<json> template = check asMap(templates[0]);
    map<json> rule = check asMap(template["metadataStringsAsKeyword"]);
    test:assertEquals(rule["path_match"], "metadata.*");
    test:assertEquals(rule["match_mapping_type"], "string");
    map<json> ruleMapping = check asMap(rule["mapping"]);
    test:assertEquals(ruleMapping["type"], "keyword");
}

@test:Config
isolated function testDynamicTemplatesPathMatchForFlatMetadata() returns error? {
    Configuration config = {indexConfig: {dimension: 8}, metadataFieldName: ""};
    map<json> mapping = check asMap(buildIndexMapping(config, MANAGED_DOMAIN));
    map<json> mappings = check asMap(mapping["mappings"]);
    json[] templates = check mappings["dynamic_templates"].ensureType();
    map<json> template = check asMap(templates[0]);
    map<json> rule = check asMap(template["metadataStringsAsKeyword"]);
    test:assertEquals(rule["path_match"], "*");
}

@test:Config
isolated function testFlatMetadataOmitsMetadataObjectMapping() returns error? {
    Configuration config = {indexConfig: {dimension: 8}, metadataFieldName: ""};
    map<json> mapping = check asMap(buildIndexMapping(config, MANAGED_DOMAIN));
    map<json> mappings = check asMap(mapping["mappings"]);
    map<json> properties = check asMap(mappings["properties"]);
    test:assertFalse(properties.hasKey("metadata"),
            "a flat-metadata mapping must not declare a 'metadata' object field");
}

@test:Config
isolated function testFixedSchemaFieldsAlwaysPresent() returns error? {
    map<json> mapping = check asMap(buildIndexMapping(baseConfig(), MANAGED_DOMAIN));
    map<json> mappings = check asMap(mapping["mappings"]);
    map<json> properties = check asMap(mappings["properties"]);

    map<json> content = check asMap(properties["content"]);
    test:assertEquals(content["type"], "text");

    map<json> docId = check asMap(properties["doc_id"]);
    test:assertEquals(docId["type"], "keyword");

    map<json> chunkType = check asMap(properties["chunk_type"]);
    test:assertEquals(chunkType["type"], "keyword");

    map<json> metadata = check asMap(properties["metadata"]);
    test:assertEquals(metadata["type"], "object");
}

@test:Config
isolated function testConfigurableFieldNamesHonored() returns error? {
    Configuration config = {
        indexConfig: {dimension: 8},
        vectorFieldName: "vec",
        contentFieldName: "text",
        idFieldName: "myId"
    };
    map<json> mapping = check asMap(buildIndexMapping(config, MANAGED_DOMAIN));
    map<json> mappings = check asMap(mapping["mappings"]);
    map<json> properties = check asMap(mappings["properties"]);
    test:assertTrue(properties.hasKey("vec"));
    test:assertTrue(properties.hasKey("text"));
    test:assertTrue(properties.hasKey("myId"));
    test:assertFalse(properties.hasKey("embedding"));
}

@test:Config
isolated function testHnswParametersHonored() returns error? {
    Configuration config = {indexConfig: {dimension: 8, efConstruction: 256, m: 32}};
    map<json> vectorField = check vectorFieldOf(buildIndexMapping(config, MANAGED_DOMAIN));
    map<json> method = check asMap(vectorField["method"]);
    map<json> parameters = check asMap(method["parameters"]);
    test:assertEquals(parameters["ef_construction"], 256);
    test:assertEquals(parameters["m"], 32);
}
