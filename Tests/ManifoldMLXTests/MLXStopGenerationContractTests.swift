import XCTest
import ManifoldContract
import ManifoldInference
import ManifoldMLX
@_spi(Testing) import ManifoldMLX

/// Issue #171 — `stopGeneration()` must satisfy core's `InferenceBackend`
/// contract: after it returns, `isGenerating` is `false` (point 3) and a fresh
/// `generate()` succeeds without a reload or reset (point 2).
///
/// Both tests hold a generation genuinely **in flight** via
/// `MLXBackend._yieldHookForTesting`. That matters: a mock generation that has
/// already finished has `_isGenerating == false` for reasons unrelated to
/// cancellation, and every assertion here would pass vacuously against the
/// unfixed code.
/// Drains a stream to completion, ignoring events and cancellation errors — we
/// only need to know when the underlying generation task has unwound. Free
/// function rather than a method so the consuming `Task` does not capture the
/// non-`Sendable` `XCTestCase`.
private func drain(_ stream: GenerationStream) async {
    do {
        for try await _ in stream {}
    } catch {
        // A cancelled stream terminating with an error is expected here.
    }
}

final class MLXStopGenerationContractTests: XCTestCase {

    /// Blocks the generation task inside the driver's per-chunk yield, and lets
    /// the test observe that it actually got there — so the sequencing is
    /// deterministic rather than sleep-based.
    private actor Gate {
        private var releaseWaiters: [CheckedContinuation<Void, Never>] = []
        private var arrivalWaiters: [CheckedContinuation<Void, Never>] = []
        private var hasArrived = false
        private var isOpen = false

        /// Called from the generation task.
        func wait() async {
            hasArrived = true
            arrivalWaiters.forEach { $0.resume() }
            arrivalWaiters.removeAll()
            if isOpen { return }
            await withCheckedContinuation { releaseWaiters.append($0) }
        }

        /// Called from the test: returns once a generation has reached the gate.
        func waitUntilArrived() async {
            if hasArrived { return }
            await withCheckedContinuation { arrivalWaiters.append($0) }
        }

        func open() {
            isOpen = true
            releaseWaiters.forEach { $0.resume() }
            releaseWaiters.removeAll()
        }
    }

    private func stallingConfig() -> GenerationConfig {
        var config = GenerationConfig()
        // Hook fires every `yieldEveryNTokens` chunks; 1 makes arrival prompt.
        config.yieldEveryNTokens = 1
        return config
    }

    private func makeMock() -> MockMLXModelContainer {
        let mock = MockMLXModelContainer()
        mock.tokensToYield = Array(repeating: "tok", count: 64)
        return mock
    }

    override func tearDown() {
        MLXBackend._yieldHookForTesting = nil
        super.tearDown()
    }

    /// Contract points 2 and 3: cancel mid-stream, then resend immediately.
    ///
    /// Against the unfixed backend this fails at the first assertion —
    /// `_isGenerating` was cleared only by the cancelled task's own teardown,
    /// which has not run yet when `stopGeneration()` returns.
    func test_stopGeneration_midStream_clearsIsGeneratingAndAllowsImmediateResend() async throws {
        let gate = Gate()
        MLXBackend._yieldHookForTesting = { await gate.wait() }

        let backend = MLXBackend()
        backend._inject(makeMock())

        let firstStream = try backend.generate(
            prompt: "first",
            systemPrompt: nil,
            config: stallingConfig(),
            hints: GenerationRuntimeHints()
        )
        let firstConsumer = Task { await drain(firstStream) }

        await gate.waitUntilArrived()
        XCTAssertTrue(backend.isGenerating,
                      "Precondition: the first generation must be genuinely in flight, or this test is vacuous")

        backend.stopGeneration()

        XCTAssertFalse(backend.isGenerating,
                       "Contract point 3: isGenerating must be false synchronously after stopGeneration() returns")
        XCTAssertNoThrow(
            try backend.generate(
                prompt: "second",
                systemPrompt: nil,
                config: stallingConfig(),
                hints: GenerationRuntimeHints()
            ),
            "Contract point 2: the backend must accept a new generate() immediately after stopGeneration(), with no backoff"
        )

        await gate.open()
        _ = await firstConsumer.value
    }

    /// The trap the naive fix walks into: clearing `_isGenerating` inside
    /// `stopGeneration()` without a generation identity means the cancelled
    /// task's teardown — which runs later, from a task nobody owns any more —
    /// clears the *successor's* flag, leaving a live generation reporting
    /// `isGenerating == false` and admitting a third concurrent `generate()`.
    ///
    /// Sequencing is fully deterministic: the superseded generation is released
    /// and awaited to completion before the assertion, while the successor is
    /// still parked on its own gate.
    func test_supersededGenerationTeardown_doesNotClearSuccessorIsGenerating() async throws {
        let firstGate = Gate()
        let secondGate = Gate()

        MLXBackend._yieldHookForTesting = { await firstGate.wait() }
        let backend = MLXBackend()
        backend._inject(makeMock())

        let firstStream = try backend.generate(
            prompt: "first",
            systemPrompt: nil,
            config: stallingConfig(),
            hints: GenerationRuntimeHints()
        )
        let firstConsumer = Task { await drain(firstStream) }
        await firstGate.waitUntilArrived()

        backend.stopGeneration()

        // The successor parks on its own gate, so it stays in flight for the
        // whole window in which the superseded task tears down.
        MLXBackend._yieldHookForTesting = { await secondGate.wait() }
        let secondStream = try backend.generate(
            prompt: "second",
            systemPrompt: nil,
            config: stallingConfig(),
            hints: GenerationRuntimeHints()
        )
        let secondConsumer = Task { await drain(secondStream) }
        await secondGate.waitUntilArrived()
        XCTAssertTrue(backend.isGenerating, "Precondition: the successor must be in flight")

        // Release the superseded generation and wait for it to fully unwind —
        // this is the moment its teardown would clobber the successor's flag.
        await firstGate.open()
        _ = await firstConsumer.value

        XCTAssertTrue(
            backend.isGenerating,
            "A superseded generation's teardown must not clear the successor's isGenerating flag"
        )

        await secondGate.open()
        _ = await secondConsumer.value
    }
}
