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
@discardableResult
private func drain(_ stream: GenerationStream) async -> Int {
    var tokenCount = 0
    do {
        for try await event in stream {
            if case .token = event { tokenCount += 1 }
        }
    } catch {
        // A cancelled stream terminating with an error is expected here.
    }
    return tokenCount
}

/// Fails the test instead of hanging it. `waitUntilArrived()` and awaiting a
/// consumer task are both unbounded, and XCTest applies no default timeout to
/// an `async` test method — so if the driver ever stops reaching the yield hook
/// (a throw before the first chunk, or a change in yield cadence) these would
/// park forever. On the merge gate that surfaces as a hung CI job killed at the
/// workflow timeout with no useful output, which is strictly worse than a red
/// assertion. Bounded waits turn it back into a diagnosable failure.
private func withTestTimeout(
    _ seconds: Double = 10,
    _ description: String,
    file: StaticString = #filePath,
    line: UInt = #line,
    _ body: @escaping @Sendable () async -> Void
) async {
    let completed = await withTaskGroup(of: Bool.self) { group in
        group.addTask { await body(); return true }
        group.addTask {
            try? await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
            return false
        }
        let first = await group.next() ?? false
        group.cancelAll()
        return first
    }
    if !completed {
        XCTFail("Timed out after \(seconds)s waiting for: \(description)", file: file, line: line)
    }
}

/// Value-returning variant of `withTestTimeout`, for awaits whose result the
/// test needs. Returns `nil` and fails the test on timeout.
private func withTestTimeoutValue<T: Sendable>(
    _ seconds: Double = 10,
    _ description: String,
    file: StaticString = #filePath,
    line: UInt = #line,
    _ body: @escaping @Sendable () async -> T
) async -> T? {
    let result = await withTaskGroup(of: T?.self) { group in
        group.addTask { await body() }
        group.addTask {
            try? await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
            return nil
        }
        let first = await group.next() ?? nil
        group.cancelAll()
        return first
    }
    if result == nil {
        XCTFail("Timed out after \(seconds)s waiting for: \(description)", file: file, line: line)
    }
    return result
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

    /// One-shot latch so a hook that may fire more than once acts only the
    /// first time. `NSLock`-guarded rather than a bare `Bool` because the hook
    /// is `@Sendable`.
    private final class OneShot: @unchecked Sendable {
        private let lock = NSLock()
        private var used = false
        func claim() -> Bool {
            lock.lock(); defer { lock.unlock() }
            if used { return false }
            used = true
            return true
        }
    }

    /// Enough chunks that a *completed* generation is unmistakably distinct
    /// from a cancelled one.
    private let mockTokenCount = 64

    private func stallingConfig() -> GenerationConfig {
        var config = GenerationConfig()
        // Hook fires every `yieldEveryNTokens` chunks; 1 makes arrival prompt.
        config.yieldEveryNTokens = 1
        return config
    }

    private func makeMock() -> MockMLXModelContainer {
        let mock = MockMLXModelContainer()
        mock.tokensToYield = Array(repeating: "tok", count: mockTokenCount)
        return mock
    }

    override func tearDown() {
        MLXBackend._yieldHookForTesting = nil
        MLXBackend._preTaskInstallHookForTesting = nil
        super.tearDown()
    }

    /// Review finding F1: `stopGeneration()` landing between `generate()`'s
    /// critical section and the install of `_generationTask`.
    ///
    /// The stop finds no task to cancel, but still bumps the epoch and clears
    /// `_isGenerating`. If the launching call then installed unconditionally,
    /// it would leave an **uncancelled** task that no later `stopGeneration()`
    /// can reach, while `isGenerating` reads `false` — so the next `generate()`
    /// is admitted and two generations run concurrently on one
    /// `ModelContainer`.
    ///
    /// The window predates #171 but was benign then: the missed cancel left
    /// `_isGenerating == true`, so a resend bounced with `alreadyGenerating`.
    /// Clearing the flag synchronously is what makes it real concurrency, so
    /// this test guards the fix that made the clearing safe — not the clearing.
    func test_stopGenerationDuringLaunchWindow_cancelsTheOrphanedGeneration() async throws {
        let gate = Gate()
        MLXBackend._yieldHookForTesting = { await gate.wait() }

        let backend = MLXBackend()
        backend._inject(makeMock())

        // Fire exactly once, in the launch window, from inside generate().
        let stopped = OneShot()
        MLXBackend._preTaskInstallHookForTesting = { [weak backend] in
            guard stopped.claim() else { return }
            backend?.stopGeneration()
        }

        let stream = try backend.generate(
            prompt: "raced",
            systemPrompt: nil,
            config: stallingConfig(),
            hints: GenerationRuntimeHints()
        )
        let consumer = Task { await drain(stream) }

        XCTAssertFalse(
            backend.isGenerating,
            "The stop landed after the epoch was claimed, so the backend must report idle"
        )

        // THE assertion. Everything else here (isGenerating false, the backend
        // still usable, the consumer completing) holds whether or not the
        // install honours the epoch — a first draft of this test asserted only
        // those and passed against the unfixed code. What actually differs is
        // whether the orphaned generation was *cancelled*: guarded, it is
        // cancelled at install and yields almost nothing; unguarded, it is
        // installed uncancelled and runs the mock to completion.
        await gate.open()
        let producedTokens = await withTestTimeoutValue(10, "orphaned generation to unwind") {
            await consumer.value
        } ?? mockTokenCount

        XCTAssertLessThan(
            producedTokens, mockTokenCount,
            """
            A generation orphaned by a stop in the launch window must be cancelled \
            at install, not left running unreachable. Producing all \(mockTokenCount) \
            mock tokens means it ran to completion with no owner — and no later \
            stopGeneration() could ever have reached it.
            """
        )

        XCTAssertFalse(
            backend.isGenerating,
            "A cancelled-at-install generation must not leave the backend marked as generating"
        )
        XCTAssertNoThrow(
            try backend.generate(
                prompt: "after",
                systemPrompt: nil,
                config: stallingConfig(),
                hints: GenerationRuntimeHints()
            ),
            "The backend must remain usable after a stop that raced the launch window"
        )
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

        await withTestTimeout(10, "first generation to reach the yield hook") {
            await gate.waitUntilArrived()
        }
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

        // Load-bearing: the resend above is parked on this same gate, so
        // opening it is what lets that task unwind before the test returns.
        // Do not "tidy" it away.
        await gate.open()
        await withTestTimeout(10, "cancelled generation to unwind") {
            _ = await firstConsumer.value
        }
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
        await withTestTimeout(10, "first generation to reach its gate") {
            await firstGate.waitUntilArrived()
        }

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
        await withTestTimeout(10, "successor generation to reach its gate") {
            await secondGate.waitUntilArrived()
        }
        XCTAssertTrue(backend.isGenerating, "Precondition: the successor must be in flight")

        // Release the superseded generation and wait for it to fully unwind —
        // this is the moment its teardown would clobber the successor's flag.
        await firstGate.open()
        await withTestTimeout(10, "superseded generation to fully unwind") {
            _ = await firstConsumer.value
        }

        XCTAssertTrue(
            backend.isGenerating,
            "A superseded generation's teardown must not clear the successor's isGenerating flag"
        )

        await secondGate.open()
        await withTestTimeout(10, "successor generation to unwind") {
            _ = await secondConsumer.value
        }
    }
}
