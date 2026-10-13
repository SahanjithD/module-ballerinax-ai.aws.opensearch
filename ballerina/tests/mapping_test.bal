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

// Classic supports Faiss only; leaving the engine to the server breaks k-NN pre-filtering.
@test:Config
isolated function testServerlessClassicEmitsMethodBlockPinnedToFaiss() returns error? {
    json mapping = buildIndexMapping(baseMode(), {}, classicDeployment());
    map<json> vectorField = check vectorFieldOf(mapping);
    map<json> method = check asMap(vectorField["method"]);
    test:assertEquals(method["engine"], "faiss");
    test:assertEquals(method["name"], "hnsw");
}

// NextGen gets a `method` block with the HNSW parameters but no `engine`, which its proxy rejects.
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

// Unset must leave NextGen's own default in place.
@test:Config
isolated function testQuantizationOmittedWhenUnset() returns error? {
    map<json> vectorField = check vectorFieldOf(
            buildIndexMapping(baseMode(), {}, nextGenDeployment()));
    test:assertFalse(vectorField.hasKey("compression_level"),
            "an unset 'compressionLevel' must leave the server's own default in place");
}

// `compression_level` sits at the field's top level, beside the `method` block.
@test:Config
isolated function testQuantizationEmittedAtFieldTopLevelBesideTheMethodBlock() returns error? {
    map<json> vectorField = check vectorFieldOf(
            buildIndexMapping(baseMode(), {}, nextGenDeployment(COMPRESSION_1X)));
    test:assertEquals(vectorField["compression_level"], "1x");

    map<json> method = check asMap(vectorField["method"]);
    map<json> parameters = check asMap(method["parameters"]);
    test:assertEquals(parameters["m"], 16,
            "the HNSW parameters must survive alongside the quantization parameter");
}

// Pins the enum to the ratios NextGen accepts: 1x, 2x, 8x, 16x and 32x.
@test:Config
isolated function testEveryCompressionLevelIsOneNextGenAccepts() returns error? {
    string[] acceptedByNextGen = ["1x", "2x", "8x", "16x", "32x"];
    CompressionLevel[] levels =
        [COMPRESSION_1X, COMPRESSION_2X, COMPRESSION_8X, COMPRESSION_16X, COMPRESSION_32X];
    foreach CompressionLevel level in levels {
        map<json> vectorField = check vectorFieldOf(
                buildIndexMapping(baseMode(), {}, nextGenDeployment(level)));
        test:assertEquals(vectorField["compression_level"], level);
        test:assertTrue(acceptedByNextGen.indexOf(level) is int,
                string `'${level}' is not in the set the AOSS proxy names as acceptable`);
    }
    test:assertEquals(levels.length(), acceptedByNextGen.length(),
            "'CompressionLevel' must offer exactly the ratios NextGen stores, no more and no fewer");
}

// The AOSS proxy rejects every value of `mode`, so no mapping may carry it.
@test:Config
isolated function testVectorModeIsNeverEmitted() returns error? {
    Deployment[] deployments = [nextGenDeployment(), nextGenDeployment(COMPRESSION_1X),
        nextGenDeployment(COMPRESSION_32X), managedDeployment(), classicDeployment()];
    foreach Deployment deployment in deployments {
        map<json> vectorField = check vectorFieldOf(buildIndexMapping(baseMode(), {}, deployment));
        test:assertFalse(vectorField.hasKey("mode"),
                string `${deployment.deploymentType} must not carry 'mode'; the AOSS proxy rejects ` +
                "every value of it and no other deployment ever declared the field");
    }
}

// Only NextGen declares `compressionLevel`; the builder must not add it elsewhere.
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

// --- sparse and hybrid mappings ---------------------------------------------------------------

@test:Config
isolated function testSparseMappingUsesRankFeatures() returns error? {
    map<json> mapping = check asMap(buildIndexMapping(sparseMode(), {}, managedDeployment()));
    map<json> mappings = check asMap(mapping["mappings"]);
    map<json> properties = check asMap(mappings["properties"]);
    map<json> sparseField = check asMap(properties["sparse_embedding"]);
    test:assertEquals(sparseField["type"], "rank_features");
    test:assertFalse(properties.hasKey("embedding"),
            "a SPARSE index stores no dense vector, so it must map no knn_vector field");
}

// `index.knn` turns on the k-NN codec, which a SPARSE index has no use for.
@test:Config
isolated function testSparseMappingOmitsSettingsEntirely() returns error? {
    map<json> mapping = check asMap(buildIndexMapping(sparseMode(), {}, managedDeployment()));
    test:assertFalse(mapping.hasKey("settings"),
            "a SPARSE index has no knn_vector field and must not request the k-NN codec");
}

// No `knn_vector` settings belong in a SPARSE mapping on any deployment.
@test:Config
isolated function testSparseMappingLeaksNoDenseVectorParameters() returns error? {
    foreach Deployment deployment in allDeployments() {
        json mapping = buildIndexMapping(sparseMode(), {}, deployment);
        string serialized = mapping.toJsonString();
        foreach string leaked in ["method", "engine", "space_type", "compression_level", "hnsw", "dimension"] {
            test:assertFalse(serialized.includes(leaked),
                    string `a SPARSE mapping must not carry '${leaked}', got: ${serialized}`);
        }
    }
}

@test:Config
isolated function testSparseMappingHonoursCustomFieldName() returns error? {
    map<json> mapping = check asMap(buildIndexMapping(sparseMode("tokens"), {}, managedDeployment()));
    map<json> mappings = check asMap(mapping["mappings"]);
    map<json> properties = check asMap(mappings["properties"]);
    test:assertTrue(properties.hasKey("tokens"));
    test:assertFalse(properties.hasKey("sparse_embedding"));
}

@test:Config
isolated function testHybridMappingCarriesBothVectorFields() returns error? {
    map<json> mapping = check asMap(buildIndexMapping(hybridMode(), {}, managedDeployment()));
    map<json> mappings = check asMap(mapping["mappings"]);
    map<json> properties = check asMap(mappings["properties"]);
    map<json> denseField = check asMap(properties["embedding"]);
    map<json> sparseField = check asMap(properties["sparse_embedding"]);
    test:assertEquals(denseField["type"], "knn_vector");
    test:assertEquals(sparseField["type"], "rank_features");
    map<json> settings = check asMap(mapping["settings"]);
    map<json> index = check asMap(settings["index"]);
    test:assertEquals(index["knn"], true, "a HYBRID index does hold a knn_vector and needs the codec");
}
