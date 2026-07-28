import XCTest
import ManifoldInference
import ManifoldTestSupport
import ManifoldMLX
@_spi(Testing) import ManifoldMLX

/// Real-model soak over the two generation-boundary shapes issue #157 names:
/// back-to-back turns against one resident backend, and cancel → immediate
/// resend.
///
/// Grown out of the throwaway `repro-157` driver written to hunt #157. That
/// hunt came back **negative** — 60 sequential turns and 40 cancel/resend turns
/// with KV reuse on, no crash (see the negative-result comment on #157) — but
/// the driver found something else on the way: `stopGeneration()` was not
/// leaving the backend reusable, so every post-cancel resend threw
/// `alreadyGenerating` 20/20. That became #171, fixed in #172.
///
/// So this file earns its place twice over. It is the **real-model regression
/// guard for #171** — the unit tests in `MLXStopGenerationContractTests` prove
/// the state machine against a mock, and this proves it against actual MLX
/// inference — and it is a ready-made harness for re-hunting #157 without
/// rebuilding a driver from scratch.
///
/// Hardware-gated like the rest of this target: Apple Silicon + Metal + a local
/// MLX snapshot. Run via `scripts/test-mlx-integration.sh`.
@MainActor
final class MLXCancelResendSoakTests: XCTestCase {

    private var modelURL: URL!
    private var loadedBackends: [MLXBackend] = []

    /// Kept small so this stays cheap enough to run every time the gated lane
    /// runs. Be clear about what that buys: #157's original finding reproduced
    /// 3/3 through the fuzz replay path, and a hand hunt at 60 and 40 turns
    /// found nothing, so 8 turns is a **regression guard, not a crash hunt**.
    /// Raise it locally when actually hunting #157.
    private let turnCount = 8

    override func setUp() async throws {
        try await super.setUp()

        try XCTSkipUnless(HardwareRequirements.isAppleSilicon, "Requires Apple Silicon")
        try XCTSkipUnless(HardwareRequirements.hasMetalDevice, "Requires Metal GPU")

        guard let mlxDir = HardwareRequirements.findMLXModelDirectory() else {
            throw XCTSkip(
                "Set MLX_TEST_MODEL to a local MLX model directory (or run via scripts/test-mlx-integration.sh)."
            )
        }
        try XCTSkipIf(
            MLXModelProbe.requiresVLMFactory(at: mlxDir),
            "KV-cache reuse is gated off for VLMs — this soak is about the reuse path."
        )
        modelURL = mlxDir
    }

    override func tearDown() async throws {
        for backend in loadedBackends.reversed() {
            backend.unloadModel()
        }
        loadedBackends.removeAll()
        modelURL = nil
        try await super.tearDown()
    }

    // MARK: - Trigger input (issue #157, recorded finding 6388496947cb)

    private let triggerSystemPrompt = "You are a verbose narrator. Describe in detail."
    private let triggerUserTurn = "What is 2 + 2?"

    /// Transcribed from the recorded finding rather than invented, so this stays
    /// a faithful harness for #157 and not merely a generic soak.
    private var triggerConfig: GenerationConfig {
        GenerationConfig(
            temperature: 0.7,
            topP: 0.9,
            seed: 1107,
            maxOutputTokens: 256,
            tools: [
                ToolDefinition(
                    name: "get_weather",
                    description: "Get the current weather for a city.",
                    parameters: .object([
                        "type": .string("object"),
                        "properties": .object([
                            "city": .object(["type": .string("string")]),
                            "units": .object(["type": .string("string")]),
                        ]),
                        "required": .array([.string("city"), .string("units")]),
                    ])
                ),
                ToolDefinition(
                    name: "schedule_alarm",
                    description: "Schedule an alarm.",
                    parameters: .object([
                        "type": .string("object"),
                        "properties": .object([
                            "label": .object(["type": .string("string")]),
                            "minutes_from_now": .object(["type": .string("integer")]),
                        ]),
                        "required": .array([.string("label"), .string("minutes_from_now")]),
                    ])
                ),
            ],
            toolChoice: .auto
        )
    }

    // MARK: - Helpers

    private func loadBackend() async throws -> MLXBackend {
        let backend = MLXBackend(enableKVCacheReuse: true)
        try await backend.loadModel(from: modelURL, plan: .testStub(effectiveContextSize: 8192))
        loadedBackends.append(backend)
        return backend
    }

    /// Runs one turn to completion, returning `(tokenCount, reuseEvents)`.
    private func runTurn(on backend: MLXBackend) async throws -> (tokens: Int, reuseEvents: [Int]) {
        let stream = try backend.generate(
            prompt: triggerUserTurn,
            systemPrompt: triggerSystemPrompt,
            config: triggerConfig,
            hints: GenerationRuntimeHints()
        )
        var tokens = 0
        var reuseEvents: [Int] = []
        for try await event in stream.events {
            switch event {
            case .token: tokens += 1
            case .kvCacheReuse(let n): reuseEvents.append(n)
            default: break
            }
        }
        return (tokens, reuseEvents)
    }

    // MARK: - Tests

    /// Back-to-back turns against one resident backend with KV reuse on — the
    /// "rapid successive generations" shape from #157's title.
    ///
    /// The assertion that matters is the **non-vacuity** one. ManifoldKit#2344
    /// turned 9,626 generation-free runs into a reported "0 findings, exit 0" on
    /// this very issue; a soak that silently stops generating looks identical to
    /// a soak that finds nothing. So every turn must produce tokens, and reuse
    /// must actually engage after the first turn — otherwise this test is
    /// asserting nothing about the code path it claims to cover.
    func test_sequentialTurns_withKVReuse_produceRealWorkAndDoNotCrash() async throws {
        let backend = try await loadBackend()

        for turn in 1...turnCount {
            let (tokens, reuseEvents) = try await runTurn(on: backend)

            XCTAssertGreaterThan(
                tokens, 0,
                "Turn \(turn) produced no tokens — the harness is inert and this run is not evidence (cf. ManifoldKit#2344)"
            )
            if turn > 1 {
                XCTAssertFalse(
                    reuseEvents.isEmpty,
                    "Turn \(turn) emitted no .kvCacheReuse event, so the KV-reuse path #157 implicates was never exercised"
                )
            }
        }
    }

    /// Cancel → immediate resend, repeated. This is #157's named repro shape and
    /// the real-model regression guard for #171.
    ///
    /// Before #172 this failed on the **first** resend: `stopGeneration()` left
    /// `isGenerating == true`, so `generate()` threw `alreadyGenerating`. It
    /// reproduced 20/20 against Llama-3.1-8B-Instruct-4bit, i.e. every single
    /// post-cancel resend.
    func test_cancelThenImmediateResend_neverThrowsAlreadyGenerating() async throws {
        let backend = try await loadBackend()

        for turn in 1...turnCount {
            let stream = try backend.generate(
                prompt: triggerUserTurn,
                systemPrompt: triggerSystemPrompt,
                config: triggerConfig,
                hints: GenerationRuntimeHints()
            )

            // Consume a few tokens so the cancel lands mid-generation rather
            // than against an already-finished turn — otherwise the resend
            // succeeds for reasons unrelated to the contract.
            var tokens = 0
            for try await event in stream.events {
                if case .token = event {
                    tokens += 1
                    if tokens >= 3 { break }
                }
            }
            XCTAssertGreaterThan(
                tokens, 0,
                "Turn \(turn) produced no tokens before cancelling — nothing was in flight, so this iteration proves nothing"
            )

            backend.stopGeneration()

            // Contract point 3, against a real model: synchronous.
            XCTAssertFalse(
                backend.isGenerating,
                "Turn \(turn): isGenerating must be false as soon as stopGeneration() returns (#171)"
            )

            // Contract point 2: the resend must be accepted with no backoff.
            // A throw here is the #171 regression.
            var resend: GenerationStream?
            XCTAssertNoThrow(
                resend = try backend.generate(
                    prompt: triggerUserTurn,
                    systemPrompt: triggerSystemPrompt,
                    config: triggerConfig,
                    hints: GenerationRuntimeHints()
                ),
                "Turn \(turn): resend immediately after stopGeneration() must not throw alreadyGenerating (#171)"
            )

            // Tear the resend down before the next iteration so each turn
            // starts from a known-idle backend rather than inheriting the
            // previous one's in-flight generation.
            _ = resend
            backend.stopGeneration()
        }
    }
}
