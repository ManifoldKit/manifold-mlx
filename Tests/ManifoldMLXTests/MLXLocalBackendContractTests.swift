import XCTest
import Foundation
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
/// unconfigured state by default — the `isGenerating == false on init`
/// invariant and the `capabilities` snapshot checks do not require a model,
/// and `test_capabilityGate_disclaimedRequirementThrows` runs against the
/// zero state on every PR. For the two generation scenarios below
/// (`test_generate_simplePrompt_emitsTokensInOrder`,
/// `test_generate_stopsGenerating_afterStreamEnd`), a real model MUST be
/// loaded before `generate()` is called — `MLXBackend.generate()` throws
/// `.inferenceFailure("No model loaded")` on the zero state, so those two
/// scenarios previously failed deterministically whenever `RUN_SLOW_TESTS=1`
/// was set without a model discoverable on disk (the nightly `slow-tests.yml`
/// lane's actual failure mode — its runner never has a real MLX snapshot).
/// `skipUnlessRealModelAvailable()` now discovers a model via
/// `HardwareRequirements.findMLXModelDirectory()` (honouring `MLX_TEST_MODEL`
/// / `MANIFOLD_DISCOVER_LOCAL_MODELS=1`, same convention as
/// `ManifoldMLXIntegrationTests`) and skips cleanly when none is found,
/// stashing the discovered URL for `makeBackend` to load.
///
/// `MLXBackend.capabilities` before any `loadModel` call falls back to
/// the 8192-token conservative context default, which is reflected in the
/// participant's `capabilities` snapshot below.
final class MLXLocalBackendContractTests: XCTestCase {

    /// Set by `skipUnlessRealModelAvailable()` before a gated generation
    /// scenario runs; consumed by `makeBackend` to load a real model instead
    /// of returning the zero state. `nil` for every fast-lane scenario
    /// (capabilities/init checks, the injected-mock companions), which never
    /// call the skip helper and so never touch real hardware.
    ///
    /// Lock-guarded rather than a bare `nonisolated(unsafe) static var`: this
    /// is written from the test method (before the runner call) and read from
    /// `makeBackend`'s `@Sendable` closure, which XCTest may invoke off the
    /// test method's thread — an unlocked seam here would be exactly the
    /// footgun `UnlockedNonisolatedUnsafeTestSeamAuditTest` (core repo) exists
    /// to catch.
    private static let discoveredModelURLLock = NSLock()
    private static nonisolated(unsafe) var _discoveredModelURLStorage: URL?
    private static var discoveredModelURL: URL? {
        get {
            discoveredModelURLLock.lock()
            defer { discoveredModelURLLock.unlock() }
            return _discoveredModelURLStorage
        }
        set {
            discoveredModelURLLock.lock()
            defer { discoveredModelURLLock.unlock() }
            _discoveredModelURLStorage = newValue
        }
    }

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
            // Zero-state (no model loaded) → `_dialect == .unknown`, so tool
            // calling is correctly reported `false`: the live capability is now
            // conditional on a recognised tool dialect (Phase 0 / #2005). A
            // loaded Qwen/Llama/Mistral model flips this to `true`.
            supportsToolCalling: false,
            supportsStructuredOutput: false,
            supportsNativeJSONMode: false,
            cancellationStyle: .cooperative,
            supportsTokenCounting: true,
            memoryStrategy: .resident,
            maxOutputTokens: 4096,
            supportsStreaming: true,
            isRemote: false,
            supportsKVCachePersistence: true,
            supportsThinking: true,
            supportsVision: false,
            sharesMLXProcessResources: true
        ),
        requiresSlowTests: true,
        makeBackend: {
            let backend = MLXBackend()
            // Only the two RUN_SLOW_TESTS-gated generation scenarios set
            // `discoveredModelURL` (via `skipUnlessRealModelAvailable()`
            // beforehand); every other scenario using this participant
            // (capabilities snapshot, capability-gate throw, the fast-lane
            // injected-mock companions) gets the zero state, matching the
            // existing doc comment above.
            if let modelURL = discoveredModelURL {
                do {
                    try await backend.loadModel(from: modelURL, plan: .testStub(effectiveContextSize: 2048))
                } catch {
                    // Surface the failure loudly via the returned backend
                    // staying unloaded — the scenario's own `generate()` call
                    // will then throw "No model loaded" and the test fails
                    // with a clear signal, rather than this factory silently
                    // swallowing a real load failure (e.g. a corrupt pinned
                    // model on the runner).
                    Log.inference.error("MLXLocalBackendContractTests: failed to load discovered model at \(modelURL.path, privacy: .public): \(error, privacy: .public)")
                }
            }
            return backend
        }
    )

    /// Clear the model-URL stash after every test so it only influences the
    /// gated generation scenario that set it (via
    /// `skipUnlessRealModelAvailable()`). Without this, on a real-model host
    /// the stash written by an earlier gated test would leak into every later
    /// `makeBackend()` in the same process — and
    /// `test_capabilityGate_disclaimedRequirementThrows` depends on the
    /// zero-state backend (its passing today under alphabetical ordering is
    /// luck, not design).
    override func tearDown() async throws {
        Self.discoveredModelURL = nil
        try await super.tearDown()
    }

    /// Skips (via `XCTSkip`) unless a real MLX model is discoverable on this
    /// host, and stashes the discovered URL for `makeBackend` to load.
    ///
    /// The runner's own `skipIfHardwareGated` only checks `RUN_SLOW_TESTS=1`
    /// and the simulator — it has no way to know whether a *model* is
    /// present, because that's a per-backend concern the shared runner
    /// deliberately doesn't own. Calling this first, before invoking the
    /// runner, closes that gap for the MLX participant specifically: no
    /// model discoverable → clean skip instead of a deterministic
    /// "No model loaded" failure.
    private static func skipUnlessRealModelAvailable() throws {
        try XCTSkipIf(
            ProcessInfo.processInfo.environment["RUN_SLOW_TESTS"] != "1",
            "[mlx.backend] contract scenarios require RUN_SLOW_TESTS=1"
        )
        try XCTSkipUnless(
            HardwareRequirements.isAppleSilicon && HardwareRequirements.hasMetalDevice,
            "[mlx.backend] requires Apple Silicon + a real Metal device"
        )
        guard let modelURL = HardwareRequirements.findMLXModelDirectory() else {
            throw XCTSkip(
                "[mlx.backend] no MLX model discoverable — set MLX_TEST_MODEL=<name-or-path> " +
                "or MANIFOLD_DISCOVER_LOCAL_MODELS=1 with a model under ~/Documents/Models"
            )
        }
        discoveredModelURL = modelURL
    }

    func test_generate_simplePrompt_emitsTokensInOrder() async throws {
        try Self.skipUnlessRealModelAvailable()
        try await LocalBackendContractRunner.assertSimplePromptEmitsTokensInOrder(
            participant: Self.participant,
            fixturesRoot: LocalBackendContractRunner.locateFixturesRoot()
        )
    }

    func test_generate_stopsGenerating_afterStreamEnd() async throws {
        try Self.skipUnlessRealModelAvailable()
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
