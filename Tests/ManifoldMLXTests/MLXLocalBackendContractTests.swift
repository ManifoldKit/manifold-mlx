import XCTest
import MLXLMCommon
import ManifoldInference
import ManifoldTestSupport
import ManifoldBackendTestKit
import ManifoldMLX
@_spi(Testing) import ManifoldMLX

/// MLX participant for the local-backend contract suite.
///
/// Moves to manifold-mlx with the backend (#1749). Scenario implementations
/// live in ``ManifoldBackendTestKit/LocalBackendContractRunner``.
///
/// The `makeBackend` factory creates an `MLXBackend` in its initial
/// unconfigured state — no real model is loaded. This is intentional:
/// the `isGenerating == false on init` invariant and the `capabilities`
/// snapshot checks do not require a model. Scenarios that call `generate()`
/// are gated behind `RUN_SLOW_TESTS=1` so they only run in the nightly tier
/// where a real model is present.
///
/// `MLXBackend.capabilities` before any `loadModel` call falls back to
/// the 8192-token conservative context default, which is reflected in the
/// participant's `capabilities` snapshot below.
final class MLXLocalBackendContractTests: XCTestCase {

    private static let participant = LocalBackendContractParticipant(
        label: "mlx.backend",
        fixtureDirectory: "mlx",
        capabilities: BackendCapabilities(
            supportedParameters: [
                .temperature, .topP, .topK, .repeatPenalty,
                .minP, .repetitionPenalty, .presencePenalty, .frequencyPenalty,
            ],
            maxContextTokens: 8192,
            requiresPromptTemplate: false,
            supportsSystemPrompt: true,
            supportsToolCalling: true,
            supportsStructuredOutput: false,
            supportsNativeJSONMode: false,
            cancellationStyle: .cooperative,
            supportsTokenCounting: true,
            memoryStrategy: .resident,
            maxOutputTokens: 4096,
            supportsStreaming: true,
            isRemote: false,
            supportsKVCachePersistence: false,
            supportsThinking: true,
            supportsVision: false,
            sharesMLXProcessResources: true
        ),
        requiresSlowTests: true,
        makeBackend: {
            // No model loaded — factory returns the backend in its zero state.
            // Generation scenarios gate themselves behind RUN_SLOW_TESTS=1 and
            // Metal availability via the runner's hardware gate.
            MLXBackend()
        }
    )

    func test_generate_simplePrompt_emitsTokensInOrder() async throws {
        try await LocalBackendContractRunner.assertSimplePromptEmitsTokensInOrder(
            participant: Self.participant,
            fixturesRoot: LocalBackendContractRunner.locateFixturesRoot()
        )
    }

    func test_generate_stopsGenerating_afterStreamEnd() async throws {
        try await LocalBackendContractRunner.assertStopsGeneratingAfterStreamEnd(
            participant: Self.participant
        )
    }

    func test_capabilityGate_disclaimedRequirementThrows() async {
        await LocalBackendContractRunner.assertCapabilityGateDisclaimedRequirementThrows(
            participant: Self.participant
        )
    }

    // MARK: - Fast-lane companions (no weights / Metal)

    // The two RUN_SLOW_TESTS-gated scenarios above
    // (`test_generate_simplePrompt_emitsTokensInOrder`,
    // `test_generate_stopsGenerating_afterStreamEnd`) only run in the nightly
    // real-model tier, so on every normal CI run they silently skip and assert
    // nothing. They are intentionally kept (issue #28 tracks the real-model lane
    // decision). The two tests below give the SAME contract a fast lane that
    // always runs: they inject a scripted `MockMLXModelContainer` through the
    // `@_spi(Testing) _inject(...)` seam on `MLXBackend` (mirroring
    // `MLXBackendGenerationTests`), so the token-ordering and
    // isGenerating-clears guarantees are exercised in CI without weights or
    // Metal. `MockMLXModelContainer`'s `MLXModelContainerProtocol` conformance
    // is provided in `MLXBackendGenerationTests.swift` in this same test target.

    /// Fast-lane companion to `test_generate_simplePrompt_emitsTokensInOrder`:
    /// the injected mock yields a scripted token sequence and the backend must
    /// surface those `.token` events in order.
    func test_generate_simplePrompt_emitsTokensInOrder_injectedMock() async throws {
        let mock = MockMLXModelContainer()
        mock.tokensToYield = ["The", " quick", " brown", " fox"]

        let backend = MLXBackend()
        backend._inject(mock)

        let stream = try backend.generate(
            prompt: "hi",
            systemPrompt: nil,
            config: GenerationConfig()
        )

        var tokens: [String] = []
        for try await event in stream.events {
            if case .token(let text) = event { tokens.append(text) }
        }

        XCTAssertEqual(tokens, ["The", " quick", " brown", " fox"],
            "Backend must emit the injected tokens in order")
    }

    /// Fast-lane companion to `test_generate_stopsGenerating_afterStreamEnd`:
    /// once a fully-drained stream ends naturally, `isGenerating` must return
    /// to false.
    func test_generate_stopsGenerating_afterStreamEnd_injectedMock() async throws {
        let mock = MockMLXModelContainer()
        mock.tokensToYield = ["a", "b", "c"]

        let backend = MLXBackend()
        backend._inject(mock)

        XCTAssertFalse(backend.isGenerating,
            "Backend must not report generating before generate() is called")

        let stream = try backend.generate(
            prompt: "hi",
            systemPrompt: nil,
            config: GenerationConfig()
        )

        // Drain the full stream so it terminates naturally.
        for try await _ in stream.events {}

        // The terminal state is applied asynchronously on the @MainActor task;
        // poll with a tight deadline, mirroring MLXBackendGenerationTests.
        let cleared = expectation(description: "isGenerating clears after stream end")
        Task {
            let deadline = ContinuousClock.now + .seconds(2)
            while backend.isGenerating, ContinuousClock.now < deadline {
                await Task.yield()
            }
            cleared.fulfill()
        }
        await fulfillment(of: [cleared], timeout: 3)

        XCTAssertFalse(backend.isGenerating,
            "isGenerating must return to false after the stream drains to completion")
    }
}

/// Real-driver coverage + adapter-shape checks for the MLX generation driver.
/// Split out of the shared `LocalBackendRealDriverCoverageTest` /
/// `LocalInferenceAdapterSmokeTests` so the file can move to manifold-mlx.
final class MLXLocalDriverCoverageTests: XCTestCase {

    @MainActor
    func test_mlxDriverHasRealPathForEveryClaim() throws {
        let driver = MLXGenerationDriver()
        try LocalDriverCoverageChecks.assertCoverage(
            adapter: driver,
            sourceFileSuffix: "Sources/ManifoldMLX/MLX/MLXGenerationDriver.swift"
        )
    }

    @MainActor
    func test_mlxGenerationDriverConformsToProtocol() {
        let driver = MLXGenerationDriver()
        LocalDriverCoverageChecks.assertAdapterShape(driver, expectedName: "mlx.generation")
        XCTAssertTrue(
            driver.declaredCapabilities.sharesMLXProcessResources,
            "MLX driver must advertise shared process-global resources"
        )
    }
}
