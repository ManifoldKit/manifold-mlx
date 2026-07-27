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
        XCTAssertNil(manifest.contextWindow,
                     "Missing config.json must leave contextWindow nil — ModelManifest.unknown no longer fabricates 8192 (core PR #2404)")
        XCTAssertFalse(manifest.supportsThinking)
        XCTAssertFalse(manifest.supportsTools,
                       "ModelManifest.unknown reports no tool support")
    }

    func test_leavesContextWindowNil_whenNoContextHintFound() throws {
        let dir = try makeTempDir()
        try writeConfig([
            "model_type": "minimal",
        ], in: dir)

        let manifest = MLXModelProbe.produceManifest(
            at: dir,
            detectedThinkingMarkers: nil,
            supportsVision: false
        )
        XCTAssertNil(manifest.contextWindow,
                     "Configs without max_position_embeddings / model_max_length must leave contextWindow nil rather than guessing 8k")
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

    func test_maxPositionEmbeddings_zero_leavesContextWindowNil() throws {
        let dir = try makeTempDir()
        try writeConfig([
            "model_type": "qwen3",
            "max_position_embeddings": 0,
        ], in: dir)

        let manifest = MLXModelProbe.produceManifest(
            at: dir, detectedThinkingMarkers: nil, supportsVision: false
        )
        XCTAssertNil(manifest.contextWindow,
                     "zero is rejected by positiveInt and must leave contextWindow nil, not guess 8k")
    }

    func test_maxPositionEmbeddings_negative_leavesContextWindowNil() throws {
        let dir = try makeTempDir()
        try writeConfig([
            "model_type": "qwen3",
            "max_position_embeddings": -4096,
        ], in: dir)

        let manifest = MLXModelProbe.produceManifest(
            at: dir, detectedThinkingMarkers: nil, supportsVision: false
        )
        XCTAssertNil(manifest.contextWindow,
                     "a negative value is rejected by positiveInt and must leave contextWindow nil, not guess 8k")
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

    // MARK: - #2348 review round 2, item 2: closes the `_injectManifest`-only test gap
    //
    // Every executed test for the #2348 context-ceiling warning
    // (`MLXBackendGenerationTests.swift`) goes through
    // `MLXBackend._injectManifest(_:)`, a test-only seam that bypasses
    // `loadModel(from:plan:)` entirely. That leaves the claim "the primary
    // warning is live in production" resting on a static trace through
    // `loadModel` → `produceManifest` → `withStateLock { _manifest = ... }`,
    // not on anything executed — the exact gap MK-4 closed on the
    // ManifoldKit side (`ModelStorageServiceTests.swift`'s
    // `test_discoverModels_mlxDirectory_populatesDetectedContextLengthFromConfig`).
    // `loadModel` itself needs real Apple-Silicon hardware and can't be
    // exercised here (`MLXBackendTests.swift`'s own header comment: hardware-
    // gated load→unload cycles are covered by `ManifoldE2ETests`, not local
    // unit tests) — but the piece that actually determines the warning's
    // threshold — `produceManifest` reading a real on-disk `config.json` —
    // needs neither GPU nor weights. This test is the explicit bridge: it
    // names the connection to the #2348 warning feature so a future reader
    // doesn't have to rediscover it, and pins the value (131072) at the
    // realistic scale `MLXBackendGenerationTests`'s own fixtures use.

    /// A value that would drive a real #2348 primary-warning threshold, read
    /// from disk with no injection anywhere in the call chain — this is what
    /// `MLXBackend.loadModel` assigns to `_manifest`, which
    /// `reportContextCheck`'s primary warning compares against. Since core
    /// PR #2404, a non-`nil` `contextWindow` on its own is what tells the
    /// primary warning the number was measured, not guessed — there is no
    /// separate detection side-channel to assert on any more.
    func test_contextWindow_fromRealConfigOnDisk_matchesWhatWouldDriveThePrimaryWarningThreshold() throws {
        let dir = try makeTempDir()
        try writeConfig([
            "model_type": "llama",
            "max_position_embeddings": 131_072,
        ], in: dir)

        let manifest = MLXModelProbe.produceManifest(
            at: dir,
            detectedThinkingMarkers: nil,
            supportsVision: false
        )

        XCTAssertEqual(
            manifest.contextWindow, 131_072,
            "produceManifest must read the real trained context from config.json on disk"
        )

        // Sabotage check: replacing `extractContextWindow(from: json)` with
        // `nil` (simulating a broken key-reader) turns this test red —
        // verified manually, reverted after confirming.
    }
}
