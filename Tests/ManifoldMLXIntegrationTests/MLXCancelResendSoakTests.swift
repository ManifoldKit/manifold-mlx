import ManifoldInference
import ManifoldMLX
@_spi(Testing) import ManifoldMLX
import ManifoldTestSupport
import XCTest

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
/// rebuilding a driver from scratch, covering both the reuse-on and reuse-off
/// configurations the issue implicates — its SIGSEGV was seen under both
/// settings (3/3 on, 1/3 off) and its SIGABRT only with the flag off.
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

  /// Matches the `--cancel-after 5` the driver used when #171 reproduced
  /// 20/20, so the cancel lands at the same point in the stream.
  private let cancelAfterTokens = 5

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
      // `unloadAndWait()`, not fire-and-forget `unloadModel()` (review
      // finding 1). This is the only file in the target that reaches
      // teardown with a *deliberately cancelled* generation still
      // unwinding on the GPU, and `unloadModel()` spawns an un-awaited
      // arbiter release that can end in `Memory.clearCache()`. The
      // arbiter's cleanup chaining is per-instance, so nothing otherwise
      // orders one test's release against the next test's claim.
      //
      // Scope, stated precisely (review finding D): this awaits the
      // *arbiter cleanup task*, not the generation task, so it closes the
      // cross-instance ordering hole and nothing more. It does **not**
      // restore the 2s settle the driver this file replaces used at the
      // same point ("tearing down mid-flight produces its own
      // driver-artefact crash and confounds the reproduction we are
      // actually hunting"). Ordering a cancelled generation's unwinding
      // against `Memory.clearCache()` would need a settle-sleep or a full
      // drain, costing wall clock or #157 fidelity; the residual risk
      // here is identical to every sibling in this target.
      await backend.unloadAndWait()
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

  private func loadBackend(enableReuse: Bool = true) async throws -> MLXBackend {
    let backend = MLXBackend(enableKVCacheReuse: enableReuse)
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

  /// The same sequential soak with KV reuse **off**.
  ///
  /// Review finding 2: #157 reports two crashes, and the second — the
  /// `addCompletedHandler`-after-commit SIGABRT — was seen **2/3 on the
  /// `--kv-reuse off` runs**, on ordinary generation/teardown rather than the
  /// cancel path. A harness that only ever ran reuse-on would have been
  /// blind to the configuration in which that crash was most often observed,
  /// while describing itself as ready for re-hunting #157.
  ///
  /// Asserts the mirror image of the reuse-on test: real work every turn, and
  /// no `.kvCacheReuse` events at all. To be clear about what is new here:
  /// the opt-out gate itself is already guarded by
  /// `MLXKVReuseIntegrationTests.test_kvReuseDisabledExplicitly` over two
  /// turns, so that assertion is a stronger instance of an existing check,
  /// not a new one. The novel contribution is the reuse-off **crash
  /// surface** — eight consecutive full-prefill turns against one resident
  /// backend, which is the configuration the SIGABRT was observed in.
  func test_sequentialTurns_withReuseDisabled_produceRealWorkAndDoNotCrash() async throws {
    let backend = try await loadBackend(enableReuse: false)

    for turn in 1...turnCount {
      let (tokens, reuseEvents) = try await runTurn(on: backend)

      XCTAssertGreaterThan(
        tokens, 0,
        "Turn \(turn) produced no tokens — the harness is inert and this run is not evidence (cf. ManifoldKit#2344)"
      )
      XCTAssertTrue(
        reuseEvents.isEmpty,
        "Turn \(turn) emitted .kvCacheReuse with reuse disabled — the opt-out gate is not holding"
      )
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
          if tokens >= cancelAfterTokens { break }
        }
      }
      // Not `> 0` (review finding A): the inner loop exits either by
      // hitting `cancelAfterTokens` or by the stream finishing on its
      // own. If a turn ran out early, `stopGeneration()` lands on an
      // already-finished generation and the resend succeeds for reasons
      // unrelated to the contract — exactly the degenerate case this
      // precondition exists to exclude. Equality is what distinguishes
      // "broke out mid-flight" from "ran out". Safe for the models this
      // targets (#157's runs recorded 21–87 tokens/turn), and a red here
      // means the discovered model is too terse for this harness rather
      // than that the contract regressed.
      XCTAssertEqual(
        tokens, cancelAfterTokens,
        "Turn \(turn) ended after \(tokens) tokens instead of being cancelled at \(cancelAfterTokens) — nothing was in flight, so this iteration proves nothing"
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

      // Prove the resend is *usable*, not merely admitted (review
      // finding 3). Without this, contract point 2 is only asserted as
      // "generate() did not throw" — a backend that admitted the call and
      // then produced nothing would pass. It also restores the replay leg
      // of #157's named "cancel -> resend -> replay" shape, which the
      // driver had and the first version of this test dropped.
      var resendTokens = 0
      if let resend {
        for try await event in resend.events {
          if case .token = event {
            resendTokens += 1
            break
          }
        }
      }
      XCTAssertGreaterThan(
        resendTokens, 0,
        "Turn \(turn): the post-cancel resend was admitted but generated nothing — accepted is not the same as usable (#171)"
      )

      // Tear the resend down before the next iteration so each turn
      // starts from a known-idle backend rather than inheriting the
      // previous one's in-flight generation.
      backend.stopGeneration()
    }
  }
}
