import XCTest
import ManifoldMLX
@_spi(Testing) import ManifoldMLX
import ManifoldInference

/// Unit tests for ``MLXModelProbe/produceManifest(at:detectedThinkingMarkers:supportsVision:)``.
///
/// These tests don't load real MLX weights — they construct a temporary
/// directory with a synthetic `config.json` and assert the probe extracts the
/// right context window. The Metal-bound path (real `loadContainer`) lives in
/// `ManifoldMLXIntegrationTests` and is hardware-gated.
final class MLXManifestProbeTests: XCTestCase {

    // MARK: - Helpers

    private func makeTempDir() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlx-manifest-probe-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        addTeardownBlock {
            try? FileManager.default.removeItem(at: dir)
        }
        return dir
    }

    private func writeConfig(_ json: [String: Any], in dir: URL) throws {
        let data = try JSONSerialization.data(withJSONObject: json, options: [])
        try data.write(to: dir.appendingPathComponent("config.json"))
    }

    // MARK: - Context window extraction

    func test_extractsTopLevelMaxPositionEmbeddings() throws {
        let dir = try makeTempDir()
        try writeConfig([
            "model_type": "qwen3",
            "max_position_embeddings": 32_768,
        ], in: dir)

        let manifest = MLXModelProbe.produceManifest(
            at: dir,
            detectedThinkingMarkers: nil,
            supportsVision: false
        )

        XCTAssertEqual(manifest.contextWindow, 32_768)
        XCTAssertEqual(manifest.modelIdentifier, dir.lastPathComponent)
        XCTAssertEqual(manifest.producerKind, .local)
        XCTAssertFalse(manifest.supportsThinking,
                       "supportsThinking must be false when no markers are detected")
    }

    func test_extractsNestedTextConfigMaxPositionEmbeddings() throws {
        let dir = try makeTempDir()
        try writeConfig([
            "model_type": "gemma4",
            "text_config": [
                "max_position_embeddings": 131_072,
            ] as [String: Any],
        ], in: dir)

        let manifest = MLXModelProbe.produceManifest(
            at: dir,
            detectedThinkingMarkers: nil,
            supportsVision: false
        )
        XCTAssertEqual(manifest.contextWindow, 131_072,
                       "text_config.max_position_embeddings must be preferred over the top-level value")
    }

    func test_textConfigWinsOverTopLevel_whenBothPresent() throws {
        let dir = try makeTempDir()
        try writeConfig([
            "model_type": "vlm-test",
            "max_position_embeddings": 4096,
            "text_config": [
                "max_position_embeddings": 200_000,
            ] as [String: Any],
        ], in: dir)

        let manifest = MLXModelProbe.produceManifest(
            at: dir,
            detectedThinkingMarkers: nil,
            supportsVision: true
        )
        XCTAssertEqual(manifest.contextWindow, 200_000,
                       "text_config wins so VLM/MoE configs reflect the text-encoder window, not the (smaller) image-encoder window")
    }

    func test_fallsBackToModelMaxLength_whenMaxPositionAbsent() throws {
        let dir = try makeTempDir()
        try writeConfig([
            "model_type": "legacy-mlx",
            "model_max_length": 16_384,
        ], in: dir)

        let manifest = MLXModelProbe.produceManifest(
            at: dir,
            detectedThinkingMarkers: nil,
            supportsVision: false
        )
        XCTAssertEqual(manifest.contextWindow, 16_384)
    }

    func test_returnsUnknownDefaults_whenConfigJsonMissing() throws {
        let dir = try makeTempDir()
        // No config.json written.

        let manifest = MLXModelProbe.produceManifest(
            at: dir,
            detectedThinkingMarkers: nil,
            supportsVision: false
        )
        XCTAssertEqual(manifest.contextWindow, 8192,
                       "Missing config.json must fall back to the conservative 8k default")
        XCTAssertFalse(manifest.supportsThinking)
        XCTAssertFalse(manifest.supportsTools,
                       "ModelManifest.unknown reports no tool support")
    }

    func test_fallsBackTo8k_whenNoContextHintFound() throws {
        let dir = try makeTempDir()
        try writeConfig([
            "model_type": "minimal",
        ], in: dir)

        let manifest = MLXModelProbe.produceManifest(
            at: dir,
            detectedThinkingMarkers: nil,
            supportsVision: false
        )
        XCTAssertEqual(manifest.contextWindow, 8192,
                       "Configs without max_position_embeddings / model_max_length must fall back to 8k")
    }

    // MARK: - positiveInt coercion of max_position_embeddings

    func test_maxPositionEmbeddings_asJSONString_parses() throws {
        let dir = try makeTempDir()
        try writeConfig([
            "model_type": "qwen3",
            "max_position_embeddings": "32768",
        ], in: dir)

        let manifest = MLXModelProbe.produceManifest(
            at: dir, detectedThinkingMarkers: nil, supportsVision: false
        )
        XCTAssertEqual(manifest.contextWindow, 32_768,
                       "a string-encoded max_position_embeddings must coerce to Int")
    }

    func test_maxPositionEmbeddings_asDouble_parses() throws {
        let dir = try makeTempDir()
        // A fractional literal forces JSONSerialization to read it back as Double.
        try writeConfig([
            "model_type": "qwen3",
            "max_position_embeddings": 32_768.0,
        ], in: dir)

        let manifest = MLXModelProbe.produceManifest(
            at: dir, detectedThinkingMarkers: nil, supportsVision: false
        )
        XCTAssertEqual(manifest.contextWindow, 32_768,
                       "a floating-point max_position_embeddings must coerce to Int")
    }

    func test_maxPositionEmbeddings_asInt64_parses() throws {
        let dir = try makeTempDir()
        try writeConfig([
            "model_type": "qwen3",
            "max_position_embeddings": Int64(32_768),
        ], in: dir)

        let manifest = MLXModelProbe.produceManifest(
            at: dir, detectedThinkingMarkers: nil, supportsVision: false
        )
        XCTAssertEqual(manifest.contextWindow, 32_768,
                       "an Int64 max_position_embeddings must coerce to Int")
    }

    func test_maxPositionEmbeddings_zero_fallsBackToDefault() throws {
        let dir = try makeTempDir()
        try writeConfig([
            "model_type": "qwen3",
            "max_position_embeddings": 0,
        ], in: dir)

        let manifest = MLXModelProbe.produceManifest(
            at: dir, detectedThinkingMarkers: nil, supportsVision: false
        )
        XCTAssertEqual(manifest.contextWindow, 8192,
                       "zero is rejected by positiveInt and must fall back to the 8k default")
    }

    func test_maxPositionEmbeddings_negative_fallsBackToDefault() throws {
        let dir = try makeTempDir()
        try writeConfig([
            "model_type": "qwen3",
            "max_position_embeddings": -4096,
        ], in: dir)

        let manifest = MLXModelProbe.produceManifest(
            at: dir, detectedThinkingMarkers: nil, supportsVision: false
        )
        XCTAssertEqual(manifest.contextWindow, 8192,
                       "a negative value is rejected by positiveInt and must fall back to the 8k default")
    }

    // MARK: - Thinking marker plumbing

    func test_carriesDetectedThinkingMarkers() throws {
        let dir = try makeTempDir()
        try writeConfig([
            "model_type": "qwen3",
            "max_position_embeddings": 32_768,
        ], in: dir)

        let manifest = MLXModelProbe.produceManifest(
            at: dir,
            detectedThinkingMarkers: .qwen3,
            supportsVision: false
        )
        XCTAssertEqual(manifest.thinkingMarkers, .qwen3)
        XCTAssertTrue(manifest.supportsThinking,
                      "supportsThinking is implied by non-nil thinkingMarkers")
    }

    // MARK: - isUnsupportedGemma4 (upstream #282/#802 crash guard)

    /// Every Gemma 4 model is refused — both the dense/multimodal e4b path
    /// (#282) and the MoE path (#802) crash, regardless of factory routing.
    func test_isUnsupportedGemma4_trueForGemma4() {
        XCTAssertTrue(MLXModelProbe.isUnsupportedGemma4(modelType: "gemma4"))
    }

    /// Neighbouring Gemma generations and other architectures are unaffected,
    /// as is a missing model_type.
    func test_isUnsupportedGemma4_falseForOtherArchitectures() {
        XCTAssertFalse(MLXModelProbe.isUnsupportedGemma4(modelType: "gemma3"))
        XCTAssertFalse(MLXModelProbe.isUnsupportedGemma4(modelType: "gemma3n"))
        XCTAssertFalse(MLXModelProbe.isUnsupportedGemma4(modelType: "llama"))
        XCTAssertFalse(MLXModelProbe.isUnsupportedGemma4(modelType: nil))
    }

    // MARK: - isUnsupportedQwen35 (upstream #157 gated-DeltaNet crash guard)

    /// Both the dense and MoE Qwen 3.5 model types crash in the linear-attention
    /// path and are refused.
    func test_isUnsupportedQwen35_trueForQwen35Variants() {
        XCTAssertTrue(MLXModelProbe.isUnsupportedQwen35(modelType: "qwen3_5"))
        XCTAssertTrue(MLXModelProbe.isUnsupportedQwen35(modelType: "qwen3_5_moe"))
    }

    /// Earlier Qwen generations (which tick fine) and other architectures are
    /// unaffected.
    func test_isUnsupportedQwen35_falseForOtherArchitectures() {
        XCTAssertFalse(MLXModelProbe.isUnsupportedQwen35(modelType: "qwen2"))
        XCTAssertFalse(MLXModelProbe.isUnsupportedQwen35(modelType: "qwen3"))
        XCTAssertFalse(MLXModelProbe.isUnsupportedQwen35(modelType: "qwen3_moe"))
        XCTAssertFalse(MLXModelProbe.isUnsupportedQwen35(modelType: nil))
    }
}
