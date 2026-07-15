import XCTest
import ManifoldRuntime
import ManifoldPersistenceSwiftData
import ManifoldInference
import ManifoldBackendTestKit
import ManifoldTestSupport
import ManifoldMLX
@_spi(Testing) import ManifoldMLX

/// Tests for MLXBackend state, capabilities, and lifecycle.
///
/// Hardware-gated tests (those that require a real model load) are skipped in CI.
/// Full load→unload cycles are covered by `ManifoldE2ETests` on Apple Silicon.
final class MLXBackendTests: XCTestCase {

    // MARK: - State on Init (no hardware gate)

    func test_init_isNotLoaded() {
        let b = MLXBackend()
        XCTAssertFalse(b.isModelLoaded)
    }

    func test_init_isNotGenerating() {
        let b = MLXBackend()
        XCTAssertFalse(b.isGenerating)
    }

    // MARK: - Capabilities (no hardware gate)

    func test_capabilities_doesNotRequirePromptTemplate() {
        XCTAssertFalse(MLXBackend().capabilities.requiresPromptTemplate)
    }

    func test_capabilities_supportsSystemPrompt() {
        XCTAssertTrue(MLXBackend().capabilities.supportsSystemPrompt)
    }

    func test_capabilities_supportsTemperature() {
        XCTAssertTrue(MLXBackend().capabilities.supportedParameters.contains(.temperature))
    }

    func test_capabilities_supportsAdditivePenaltyKnobs() {
        let caps = MLXBackend().capabilities
        XCTAssertTrue(caps.supportedParameters.contains(.topK))
        XCTAssertTrue(caps.supportedParameters.contains(.minP))
        XCTAssertTrue(caps.supportedParameters.contains(.repetitionPenalty))
        XCTAssertTrue(caps.supportedParameters.contains(.presencePenalty))
        XCTAssertTrue(caps.supportedParameters.contains(.frequencyPenalty))
    }

    // MARK: - Load Options Plumbing (no hardware gate)

    func test_loadOptions_defaultUsesBackendTunedDefaults() {
        let opts = MLXBackend().loadOptionsForTesting
        XCTAssertEqual(opts.kvCacheQuantization, .q8)
        // Intentionally NOT asserting `flashAttention` against
        // `BackendLoadOptions.platformDefaultFlashAttention` here — that is the
        // very constant `BackendLoadOptions.default` uses to initialise the field,
        // so the comparison is tautological and would pass for any value. The
        // round-trip / override behaviour of flashAttention is covered by
        // `test_setLoadOptions_persistsForNextLoad`.
        XCTAssertNil(opts.prefillBatchSize)
    }

    func test_setLoadOptions_persistsForNextLoad() {
        let backend = MLXBackend()
        backend.setLoadOptions(BackendLoadOptions(
            kvCacheQuantization: .q4,
            flashAttention: true,
            prefillBatchSize: 2048
        ))
        let opts = backend.loadOptionsForTesting
        XCTAssertEqual(opts.kvCacheQuantization, .q4)
        XCTAssertTrue(opts.flashAttention,
                      "flashAttention must round-trip through state even though MLX silently ignores it on the generate path")
        XCTAssertEqual(opts.prefillBatchSize, 2048)
    }

    func test_capabilities_contextSize() {
        XCTAssertEqual(MLXBackend().capabilities.maxContextTokens, 8192)
    }

    func test_capabilities_supportsKVCachePersistence_defaultsOnDisableToOptOut() {
        XCTAssertTrue(MLXBackend().capabilities.supportsKVCachePersistence)
        XCTAssertFalse(MLXBackend(enableKVCacheReuse: false).capabilities.supportsKVCachePersistence)
    }

    func test_capabilities_supportsVision_falseBeforeLoad() {
        XCTAssertFalse(MLXBackend().capabilities.supportsVision)
    }

    // MARK: - Lifecycle (no hardware gate)

    func test_generate_beforeLoad_throws() {
        XCTAssertThrowsError(
            try MLXBackend().generate(prompt: "hi", systemPrompt: nil, config: GenerationConfig(), hints: GenerationRuntimeHints())
        )
    }

    func test_generate_withSupportedGrammar_isAcceptedNotRejected() {
        // Option B (#96): MLX now executes grammar-constrained sampling, so a
        // *supported* GBNF grammar must NOT throw `unsupportedGrammar`. With no
        // model loaded the grammar parses cleanly and the call then fails the
        // no-model check — proving the grammar was accepted, not rejected.
        XCTAssertTrue(MLXBackend().capabilities.supportsGrammarConstrainedSampling)

        var config = GenerationConfig()
        config.grammar = "root ::= \"yes\" | \"no\""
        XCTAssertThrowsError(
            try MLXBackend().generate(prompt: "hi", systemPrompt: nil, config: config, hints: GenerationRuntimeHints())
        ) { error in
            if case InferenceError.unsupportedGrammar = error {
                return XCTFail("A supported grammar must not throw unsupportedGrammar, got \(error)")
            }
            guard case InferenceError.inferenceFailure = error else {
                return XCTFail("Expected inferenceFailure (No model loaded), got \(error)")
            }
        }
    }

    func test_generate_withUnsupportedGrammar_throwsUnsupportedGrammar() {
        // A grammar the executor cannot compile (here: no `root` rule) MUST throw
        // `unsupportedGrammar` before any model work — never silently drop the
        // constraint (issue #96 decision: throw, don't degrade).
        var config = GenerationConfig()
        config.grammar = "notroot ::= \"x\""
        XCTAssertThrowsError(
            try MLXBackend().generate(prompt: "hi", systemPrompt: nil, config: config, hints: GenerationRuntimeHints())
        ) { error in
            guard case InferenceError.unsupportedGrammar(let reason) = error else {
                return XCTFail("Expected unsupportedGrammar, got \(error)")
            }
            XCTAssertFalse(reason.isEmpty)
        }
    }

    func test_unloadModel_beforeLoad_isNoOp_andLeavesBackendUnloaded() {
        let backend = MLXBackend()
        backend.unloadModel()
        XCTAssertFalse(backend.isModelLoaded,
            "unloadModel before any load must leave the backend in its unloaded zero state")
        XCTAssertFalse(backend.isGenerating,
            "unloadModel before any load must not put the backend into a generating state")
    }

    func test_stopGeneration_beforeLoad_isNoOp_andLeavesBackendIdle() {
        let backend = MLXBackend()
        backend.stopGeneration()
        XCTAssertFalse(backend.isGenerating,
            "stopGeneration before any generation must leave isGenerating false")
        XCTAssertFalse(backend.isModelLoaded,
            "stopGeneration must not flip the model-loaded state")
    }

    func test_unloadModel_afterMockInjection_doesNotCrash() {
        let backend = MLXBackend(enableKVCacheReuse: true)
        backend._inject(MockMLXModelContainer())
        backend.unloadModel()
        XCTAssertFalse(backend.isModelLoaded)
    }

    // MARK: - Hardware-gated

    func test_loadModel_invalidDirectory_throws() async throws {
        try XCTSkipUnless(HardwareRequirements.isAppleSilicon, "MLX requires Apple Silicon")
        let b = MLXBackend()
        let badURL = URL(fileURLWithPath: "/nonexistent-mlx-model-\(UUID().uuidString)")
        do {
            try await b.loadModel(from: badURL, plan: .testStub(effectiveContextSize: 512))
            XCTFail("Should throw for invalid model directory")
        } catch {
            // The directory has no config.json, so `validateArchitecture` is a
            // no-op and the failure surfaces from the mlx-swift-lm factory load,
            // which `loadModel` wraps in `InferenceError.modelLoadFailed`.
            guard case InferenceError.modelLoadFailed = error else {
                return XCTFail("Expected InferenceError.modelLoadFailed for a nonexistent model directory, got \(error)")
            }
            XCTAssertFalse(b.isModelLoaded,
                "A failed load must leave the backend unloaded")
        }
    }

    func test_unloadModel_afterLoad_clearsState() async throws {
        try XCTSkipUnless(HardwareRequirements.isAppleSilicon, "MLX requires Apple Silicon")
        // Full load→unload→verify is covered in ManifoldE2ETests on Apple Silicon.
        throw XCTSkip("Full load→unload cycle covered in ManifoldE2ETests on Apple Silicon")
    }

    // MARK: - Architecture preflight (no hardware gate)

    /// Writes a throwaway `config.json` into a temp directory so we can exercise
    /// `MLXModelProbe.validateArchitecture` without invoking the real MLX load path
    /// (which would trip the metallib guard in `swift test`).
    private func writeTempConfig(_ json: [String: Any]) throws -> URL {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("mlx-arch-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let data = try JSONSerialization.data(withJSONObject: json)
        try data.write(to: dir.appendingPathComponent("config.json"))
        addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
        return dir
    }

    func test_validateArchitecture_acceptsQwen3() throws {
        let url = try writeTempConfig(["model_type": "qwen3"])
        XCTAssertNoThrow(try MLXModelProbe.validateArchitecture(at: url))
    }

    func test_validateArchitecture_acceptsGemma4() throws {
        // mlx-community/gemma-4-* ships `model_type: "gemma4"` (added to
        // mlx-swift-lm's LLMTypeRegistry in 3.31.3). Sabotage check: removing
        // "gemma4" from `supportedLMArchitectures` makes this throw.
        let url = try writeTempConfig(["model_type": "gemma4"])
        XCTAssertNoThrow(try MLXModelProbe.validateArchitecture(at: url))
    }

    func test_validateArchitecture_acceptsQwen25VL() throws {
        let url = try writeTempConfig([
            "model_type": "qwen2_5_vl",
            "vision_config": ["hidden_size": 1]
        ])
        XCTAssertNoThrow(try MLXModelProbe.validateArchitecture(at: url))
    }

    func test_validateArchitecture_acceptsLlamaViaArchitectures() throws {
        // HF repos that omit `model_type` but ship `architectures: ["LlamaForCausalLM"]`
        // must still pass — snake_case prefix match keeps older snapshots working.
        let url = try writeTempConfig(["architectures": ["LlamaForCausalLM"]])
        XCTAssertNoThrow(try MLXModelProbe.validateArchitecture(at: url))
    }

    func test_validateArchitecture_rejectsVisionEncoder() throws {
        let url = try writeTempConfig(["model_type": "clip"])
        XCTAssertThrowsError(try MLXModelProbe.validateArchitecture(at: url)) { error in
            guard case InferenceError.unsupportedModelArchitecture(let arch) = error else {
                return XCTFail("Expected .unsupportedModelArchitecture, got \(error)")
            }
            XCTAssertEqual(arch, "clip")
        }
        // Sabotage confirmation: adding "clip" to `supportedLMArchitectures`
        // makes this assertion fail (no throw) — verified locally before commit.
    }

    func test_validateArchitecture_rejectsEmbeddings() throws {
        let url = try writeTempConfig(["model_type": "bert"])
        XCTAssertThrowsError(try MLXModelProbe.validateArchitecture(at: url)) { error in
            guard case InferenceError.unsupportedModelArchitecture = error else {
                return XCTFail("Expected .unsupportedModelArchitecture, got \(error)")
            }
        }
    }

    func test_validateArchitecture_rejectsVisionViaArchitectures() throws {
        // `model_type` missing, `architectures` says CLIPModel — must still be refused.
        let url = try writeTempConfig(["architectures": ["CLIPModel"]])
        XCTAssertThrowsError(try MLXModelProbe.validateArchitecture(at: url)) { error in
            guard case InferenceError.unsupportedModelArchitecture = error else {
                return XCTFail("Expected .unsupportedModelArchitecture, got \(error)")
            }
        }
    }

    func test_validateArchitecture_missingConfigIsNoOp() throws {
        // A directory with no config.json must not throw — the subsequent MLX load
        // path will surface the real "missing config" diagnostic instead.
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("mlx-arch-empty-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
        XCTAssertNoThrow(try MLXModelProbe.validateArchitecture(at: dir))
    }

    // MARK: - VLM Factory Routing (issue #752, no hardware gate)

    func test_requiresVLMFactory_whenTextConfigEnablesMoE_returnsTrue() throws {
        // mlx-community/gemma-4-26b-a4b-it-4bit ships
        //   { "text_config": { "enable_moe_block": true, … } }
        // and only loads correctly via VLMModelFactory in mlx-swift-lm 3.31.3.
        // Sabotage check: flipping `requiresVLMFactory` to always return false
        // makes this assertion fail (verified locally before commit).
        let url = try writeTempConfig([
            "text_config": ["enable_moe_block": true]
        ])
        XCTAssertTrue(MLXModelProbe.requiresVLMFactory(at: url))
    }

    func test_requiresVLMFactory_whenVisionConfigPresent_returnsTrue() throws {
        let url = try writeTempConfig([
            "model_type": "qwen2_5_vl",
            "vision_config": ["hidden_size": 1]
        ])
        XCTAssertTrue(MLXModelProbe.requiresVLMFactory(at: url))
    }

    func test_requiresVLMFactory_whenTextConfigOmitsMoE_returnsFalse() throws {
        // Dense Gemma 4 variants (e.g. gemma-4-e4b-it-4bit) ship
        //   { "text_config": { "enable_moe_block": false, … } }
        // and must stay on LLMModelFactory to avoid loading vision-tower
        // weights into resident memory.
        let url = try writeTempConfig([
            "text_config": ["enable_moe_block": false]
        ])
        XCTAssertFalse(MLXModelProbe.requiresVLMFactory(at: url))
    }

    func test_requiresVLMFactory_whenConfigHasNoTextConfig_returnsFalse() throws {
        // Regular LLMs (qwen3, llama, mistral, …) have a flat config.json with
        // no `text_config` block. They route through LLMModelFactory.
        let url = try writeTempConfig(["model_type": "qwen3"])
        XCTAssertFalse(MLXModelProbe.requiresVLMFactory(at: url))
    }

    func test_requiresVLMFactory_whenConfigMissing_returnsFalse() throws {
        // Conservative fallback matches `validateArchitecture` — let the MLX
        // load path produce the real diagnostic for missing config.json.
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("mlx-vlm-empty-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
        XCTAssertFalse(MLXModelProbe.requiresVLMFactory(at: dir))
    }

    func test_requiresVLMFactory_qwen2VLTopLevelVisionConfig_returnsTrue() throws {
        // #22: mlx-community/Qwen2-VL-2B-Instruct-4bit ships a flat config.json
        // whose `vision_config` sits at the top level (alongside the image/video
        // token-id fields), NOT nested under `text_config`. The probe must route
        // this stock VLM checkpoint through the VLM factory. Sabotage check:
        // dropping the top-level `vision_config` branch from `requiresVLMFactory`
        // makes this fail (verified locally before commit).
        let url = try writeTempConfig([
            "model_type": "qwen2_vl",
            "architectures": ["Qwen2VLForConditionalGeneration"],
            "image_token_id": 151_655,
            "video_token_id": 151_656,
            "vision_start_token_id": 151_652,
            "vision_end_token_id": 151_653,
            "vision_token_id": 151_654,
            "vision_config": [
                "depth": 32,
                "hidden_size": 1280,
                "in_chans": 3,
            ],
        ])
        XCTAssertTrue(MLXModelProbe.requiresVLMFactory(at: url))
    }

    func test_requiresVLMFactory_gemma3nWithVisionAndAudioConfig_returnsFalse() throws {
        // #56: the full mlx-community/gemma-3n-E4B-it-4bit checkpoint ships
        //   { "model_type": "gemma3n", "vision_config": {…}, "audio_config": {…},
        //     "text_config": { "model_type": "gemma3n_text", … } }
        // but mlx-swift-lm only implements the gemma3n *text* decoder, registered
        // on the LLM factory (`LLMTypeRegistry`). The VLM factory has no `gemma3n`
        // creator, so routing here would throw
        // `ModelFactoryError.unsupportedModelType("gemma3n")` at load. The probe
        // must keep gemma3n on the LLM factory despite the vision/audio signals.
        // Sabotage check: removing the `llmFactoryOnlyMultimodalArchitectures`
        // guard makes this fail (the top-level `vision_config` branch fires).
        let url = try writeTempConfig([
            "model_type": "gemma3n",
            "architectures": ["Gemma3nForConditionalGeneration"],
            "vision_config": ["hidden_size": 2048],
            "audio_config": ["hidden_size": 1536],
            "text_config": ["model_type": "gemma3n_text"],
        ])
        XCTAssertFalse(MLXModelProbe.requiresVLMFactory(at: url))
    }

    func test_requiresVLMFactory_qwen2VLModelTypeWithoutVisionConfig_returnsTrue() throws {
        // A lossy conversion may strip the `vision_config` block while leaving the
        // `_vl` model_type intact. The architecture-name fallback must still route
        // such a checkpoint to the VLM factory.
        let url = try writeTempConfig(["model_type": "qwen2_5_vl"])
        XCTAssertTrue(MLXModelProbe.requiresVLMFactory(at: url))
    }

    func test_requiresVLMFactory_nullVisionConfig_returnsFalse() throws {
        // An explicit `"vision_config": null` decodes to NSNull and must NOT be
        // treated as vision support — only a real object counts.
        let url = try writeTempConfig([
            "model_type": "qwen3",
            "vision_config": NSNull(),
        ])
        XCTAssertFalse(MLXModelProbe.requiresVLMFactory(at: url))
    }

    func test_requiresVLMFactory_modelTypeEndingInBareVL_returnsFalse() throws {
        // The architecture-name fallback matches the `_vl` suffix (the mlx-swift-lm
        // registry convention), NOT a bare `vl` substring/suffix — otherwise an
        // unrelated text model_type that merely ends in the two letters "vl" would
        // be misrouted to the VLM factory. Sabotage check: widening the match to
        // `hasSuffix("vl")` makes this fail.
        let url = try writeTempConfig(["model_type": "someothervl"])
        XCTAssertFalse(MLXModelProbe.requiresVLMFactory(at: url))
    }
}

// MARK: - Backend Contract

extension MLXBackendTests {
    func test_contract_allInvariants() {
        BackendContractChecks.assertAllInvariants { MLXBackend() }
    }
}
