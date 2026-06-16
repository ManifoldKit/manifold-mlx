import XCTest
import ManifoldInference
import ManifoldTestSupport
import ManifoldMLX
@_spi(Testing) import ManifoldMLX

/// Real-inference smoke test for a Gemma 4 MoE model
/// (`mlx-community/gemma-4-26b-a4b-it-4bit` or any local variant).
/// Validates the VLM-factory routing added in PR #769 (closes #752): the
/// model has `text_config.enable_moe_block: true`, so
/// `MLXModelProbe.requiresVLMFactory` should send it to
/// `VLMModelFactory.shared.loadContainer` rather than the LLM factory's
/// no-MoE `Gemma4Model`.
///
/// Skipped automatically when no Gemma 4 MoE weights are discoverable.
/// Set `MLX_TEST_MODEL=gemma-4` or `MANIFOLD_DISCOVER_LOCAL_MODELS=1` to opt in.
@MainActor
final class Gemma4MoESmokeTests: XCTestCase {

    private var backend: MLXBackend!
    private var modelURL: URL!

    override func setUp() async throws {
        try await super.setUp()

        // Skip BEFORE any model discovery/load. The on-disk Gemma4 MoE weights
        // trigger an uncatchable upstream mlx-swift-lm C++ broadcast crash (#802),
        // and loading the model before skipping hangs/kills the integration lane
        // when the weights are present (#26). The skip must precede every load.
        throw XCTSkip("Skipped pending upstream mlx-swift-lm fix for Gemma4 MoE broadcast crash (issue #802)")

        try XCTSkipUnless(HardwareRequirements.isAppleSilicon, "Requires Apple Silicon")
        try XCTSkipUnless(HardwareRequirements.hasMetalDevice, "Requires Metal GPU")

        guard let candidate = HardwareRequirements.findMLXModelDirectory(nameContains: "gemma-4"),
              MLXModelProbe.requiresVLMFactory(at: candidate) else {
            throw XCTSkip("No Gemma 4 MoE model found — set MLX_TEST_MODEL=gemma-4 or MANIFOLD_DISCOVER_LOCAL_MODELS=1")
        }
        modelURL = candidate

        backend = MLXBackend()
        try await backend.loadModel(from: modelURL, plan: .testStub(effectiveContextSize: 2048))
        XCTAssertTrue(backend.isModelLoaded, "Backend should report model loaded")
    }

    override func tearDown() async throws {
        backend?.unloadModel()
        backend = nil
        modelURL = nil
        try await super.tearDown()
    }

    func test_loadAndGenerate_producesNonEmptyResponse() async throws {
        // Defense-in-depth: setUp() already skips before any model load (#26/#802),
        // so this body never executes while the upstream MoE crash persists.
        throw XCTSkip("Skipped pending upstream mlx-swift-lm fix for Gemma4 MoE broadcast crash (issue #802)")

        let config = GenerationConfig(temperature: 0.3, maxOutputTokens: 32)
        let stream = try backend.generate(
            prompt: "Reply with exactly one word.",
            systemPrompt: nil,
            config: config
        )
        let response = try await collectTokens(stream)

        XCTAssertFalse(response.isEmpty, "MoE Gemma 4 should produce a response")
    }
}
