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

isolated function baseMode(ai:SimilarityMetric metric = ai:COSINE) returns DenseSearch => {
    queryMode: ai:DENSE,
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
    map<json> mapping = check asMap(buildIndexMapping(baseMode(), {}, managedDeployment()));
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

// --- the `method` block ------------------------------------------------------------------------

@test:Config
isolated function testManagedDomainEmitsMethodBlock() returns error? {
    json mapping = buildIndexMapping(baseMode(), {}, managedDeployment());
    map<json> vectorField = check vectorFieldOf(mapping);
    test:assertEquals(vectorField["type"], "knn_vector");
    test:assertEquals(vectorField["dimension"], 1536);
    test:assertFalse(vectorField.hasKey("space_type"),
            "space_type must not appear at the top level when the method block carries it");
    map<json> method = check asMap(vectorField["method"]);
    test:assertEquals(method["name"], "hnsw");
    test:assertEquals(method["engine"], "faiss");
    test:assertEquals(method["space_type"], "cosinesimil");
}

@test:Config
isolated function testManagedDomainEngineIsSelectable() returns error? {
    map<json> vectorField = check vectorFieldOf(
            buildIndexMapping(baseMode(), {}, managedDeployment(LUCENE)));
    map<json> method = check asMap(vectorField["method"]);
    test:assertEquals(method["engine"], "lucene");
}

// Classic supports Faiss alone, so the variant exposes no `engine` field and the mapping hard-codes
// it. Relying on the server's default engine instead is what silently breaks k-NN pre-filtering.
@test:Config
isolated function testServerlessClassicEmitsMethodBlockPinnedToFaiss() returns error? {
    json mapping = buildIndexMapping(baseMode(), {}, classicDeployment());
    map<json> vectorField = check vectorFieldOf(mapping);
    map<json> method = check asMap(vectorField["method"]);
    test:assertEquals(method["engine"], "faiss");
    test:assertEquals(method["name"], "hnsw");
}

// NextGen gets a `method` block too, carrying the HNSW `parameters` but no `engine`. `engine`
// inside the block is the one thing its proxy rejects; a block without it is accepted and the
// parameters are stored intact, with NextGen supplying `engine: faiss` itself in the stored
// mapping. Sending no block at all -- which this module used to do -- cost HNSW tuning here for no
// reason AWS imposes.
@test:Config
isolated function testServerlessNextGenEmitsMethodBlockWithoutEngine() returns error? {
    json mapping = buildIndexMapping(baseMode(), {}, nextGenDeployment());
    map<json> vectorField = check vectorFieldOf(mapping);
    test:assertEquals(vectorField["type"], "knn_vector");
    test:assertEquals(vectorField["dimension"], 1536);
    test:assertEquals(vectorField["space_type"], "cosinesimil");

    map<json> method = check asMap(vectorField["method"]);
    test:assertEquals(method["name"], "hnsw");
    test:assertFalse(method.hasKey("engine"),
            "'engine' inside the block is what NextGen's proxy rejects with a flat 400");
    test:assertFalse(method.hasKey("space_type"),
            "NextGen carries 'space_type' at the field's top level, not inside the block");
}

@test:Config
isolated function testHnswParametersHonoredOnEveryDeploymentType() returns error? {
    DenseSearch configMode = {queryMode: ai:DENSE, indexConfig: {dimension: 8, efConstruction: 256, m: 32}};
    foreach Deployment deployment in allDeployments() {
        map<json> vectorField = check vectorFieldOf(buildIndexMapping(configMode, {}, deployment));
        map<json> method = check asMap(vectorField["method"]);
        map<json> parameters = check asMap(method["parameters"]);
        test:assertEquals(parameters["ef_construction"], 256,
                string `'ef_construction' should reach ${deployment.deploymentType}`);
        test:assertEquals(parameters["m"], 32,
                string `'m' should reach ${deployment.deploymentType}`);
    }
}

@test:Config
isolated function testSpaceTypePropagatesPerDeploymentType() returns error? {
    foreach Deployment deployment in allDeployments() {
        map<json> vectorField = check vectorFieldOf(buildIndexMapping(baseMode(ai:EUCLIDEAN), {}, deployment));
        json spaceType;
        if deployment is ServerlessNextGenDeployment {
            spaceType = vectorField["space_type"];
        } else {
            map<json> method = check asMap(vectorField["method"]);
            spaceType = method["space_type"];
        }
        test:assertEquals(spaceType, "l2",
                string `unexpected space_type for ${deployment.deploymentType}`);
    }
}

// --- quantization (NextGen only, by construction) ----------------------------------------------

// The default has to stay byte-identical to what was sent before these fields existed: a NextGen
// collection provisions `on_disk`/`32x` on its own, and this module deciding to send something
// different by default would silently change the storage of every existing index it recreates.
@test:Config
isolated function testQuantizationOmittedWhenUnset() returns error? {
    map<json> vectorField = check vectorFieldOf(
            buildIndexMapping(baseMode(), {}, nextGenDeployment()));
    test:assertFalse(vectorField.hasKey("compression_level"),
            "an unset 'compressionLevel' must leave the server's own default in place");
    test:assertFalse(vectorField.hasKey("mode"),
            "an unset 'vectorMode' must leave the server's own default in place");
}

// `compression_level`/`mode` sit at the field's top level and coexist with the `method` block --
// verified on a live NextGen collection, which stored the HNSW parameters and both quantization
// parameters together.
@test:Config
isolated function testQuantizationEmittedAtFieldTopLevelBesideTheMethodBlock() returns error? {
    map<json> vectorField = check vectorFieldOf(
            buildIndexMapping(baseMode(), {}, nextGenDeployment(COMPRESSION_1X, IN_MEMORY)));
    test:assertEquals(vectorField["compression_level"], "1x");
    test:assertEquals(vectorField["mode"], "in_memory");

    map<json> method = check asMap(vectorField["method"]);
    map<json> parameters = check asMap(method["parameters"]);
    test:assertEquals(parameters["m"], 16,
            "the HNSW parameters must survive alongside the quantization parameters");
}

@test:Config
isolated function testCompressionLevelAloneIsEmitted() returns error? {
    map<json> vectorField = check vectorFieldOf(
            buildIndexMapping(baseMode(), {}, nextGenDeployment(COMPRESSION_4X)));
    test:assertEquals(vectorField["compression_level"], "4x");
    test:assertFalse(vectorField.hasKey("mode"),
            "'compression_level' is accepted on its own; 'mode' should not be invented alongside it");
}

// `compressionLevel`/`vectorMode` are declared only on `ServerlessNextGenDeployment`, so a managed
// or Classic mapping cannot carry them however the caller is configured. This pins that the mapping
// builder invents neither.
@test:Config
isolated function testQuantizationNeverAppearsOffNextGen() returns error? {
    Deployment[] methodBlockDeployments = [
        managedDeployment(),
        classicDeployment()
    ];
    foreach Deployment deployment in methodBlockDeployments {
        map<json> vectorField = check vectorFieldOf(buildIndexMapping(baseMode(), {}, deployment));
        test:assertFalse(vectorField.hasKey("compression_level"),
                string `${deployment.deploymentType} must not carry 'compression_level'`);
        test:assertFalse(vectorField.hasKey("mode"),
                string `${deployment.deploymentType} must not carry 'mode'`);
    }
}

// --- the rest of the mapping -------------------------------------------------------------------

@test:Config
isolated function testDynamicTemplatesAlwaysPresentForNestedMetadata() returns error? {
    map<json> mapping = check asMap(buildIndexMapping(baseMode(), {}, managedDeployment()));
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
    DenseSearch configMode = {queryMode: ai:DENSE, indexConfig: {dimension: 8}};
    Configuration config = {metadataFieldName: ""};
    map<json> mapping = check asMap(buildIndexMapping(configMode, config, managedDeployment()));
    map<json> mappings = check asMap(mapping["mappings"]);
    json[] templates = check mappings["dynamic_templates"].ensureType();
    map<json> template = check asMap(templates[0]);
    map<json> rule = check asMap(template["metadataStringsAsKeyword"]);
    test:assertEquals(rule["path_match"], "*");
}

@test:Config
isolated function testFlatMetadataOmitsMetadataObjectMapping() returns error? {
    DenseSearch configMode = {queryMode: ai:DENSE, indexConfig: {dimension: 8}};
    Configuration config = {metadataFieldName: ""};
    map<json> mapping = check asMap(buildIndexMapping(configMode, config, managedDeployment()));
    map<json> mappings = check asMap(mapping["mappings"]);
    map<json> properties = check asMap(mappings["properties"]);
    test:assertFalse(properties.hasKey("metadata"),
            "a flat-metadata mapping must not declare a 'metadata' object field");
}

@test:Config
isolated function testFixedSchemaFieldsAlwaysPresent() returns error? {
    map<json> mapping = check asMap(buildIndexMapping(baseMode(), {}, managedDeployment()));
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
    DenseSearch configMode = {queryMode: ai:DENSE, indexConfig: {dimension: 8}, vectorFieldName: "vec"};
    Configuration config = {contentFieldName: "text", idFieldName: "myId"};
    map<json> mapping = check asMap(buildIndexMapping(configMode, config, managedDeployment()));
    map<json> mappings = check asMap(mapping["mappings"]);
    map<json> properties = check asMap(mappings["properties"]);
    test:assertTrue(properties.hasKey("vec"));
    test:assertTrue(properties.hasKey("text"));
    test:assertTrue(properties.hasKey("myId"));
    test:assertFalse(properties.hasKey("embedding"));
}
