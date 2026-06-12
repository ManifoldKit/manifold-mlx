import XCTest
import ManifoldInference
import ManifoldTestSupport
@_spi(Testing) import ManifoldMLX

/// Re-homed from core's cross-family `KVCacheReuseRaceRegressionTests`
/// (Tests/ManifoldBackendsTests, retired in core PR C2 when the families
/// split out — ManifoldKit#1749). This is the MLX half: the #1382
/// stale-snapshot race guard and the byte-exact prefix-trim guarantee,
/// driven through `MockMLXModelContainer` so they run without Metal.
/// The Llama half (real-GGUF warm-vs-cold determinism) lives in
/// manifold-llama.
///
/// ## Correctness model (why a non-byte-exact reuse is unrepresentable)
///
/// A turn may restore a cached KV prefix only for the leading run of tokens
/// that are a **byte-exact** match against the cached token sequence: MLX
/// restores the cached prompt cache, then clamps the resume offset to the
/// shared prefix length; generation can only ever resume from a position
/// that was decoded identically on the prior turn. Because the reuse length
/// is *derived* from the common-prefix scan rather than assumed from a
/// length or a hash, over-reuse is not a guarded path — it is
/// unrepresentable.
final class MLXKVCacheReuseRaceRegressionTests: XCTestCase {

    // MARK: - Helpers

    private func drainEvents(_ stream: GenerationStream) async throws -> [GenerationEvent] {
        var events: [GenerationEvent] = []
        for try await event in stream.events {
            events.append(event)
        }
        return events
    }

    private func reuseCounts(in events: [GenerationEvent]) -> [Int] {
        events.compactMap { event in
            if case .kvCacheReuse(let count) = event { return count }
            return nil
        }
    }

    // MARK: - A. #1382 stale-snapshot race (runs without Metal via mock)

    /// The #1382 defect: `MLXBackend.generate()` synchronously captured
    /// `_promptCacheState.snapshot` at call entry — *before* the prior turn's
    /// asynchronous snapshot-capture task had written it. The second turn
    /// therefore read a nil/stale snapshot and emitted `.kvCacheReuse(0)` (a
    /// cache miss) even though the prompt prefix matched exactly.
    ///
    /// The fix replaced the eager capture with a `currentSnapshot` closure the
    /// driver invokes *after* awaiting `pendingSnapshotTask`. This test pins that
    /// behaviour: when turn 2 starts while turn 1's snapshot task is still
    /// in-flight (the exact race window), reuse must still fire for the full
    /// shared prefix.
    ///
    /// Sabotage: reading the eagerly-captured snapshot instead of the closure
    /// (the pre-#1382 code) makes `reuseCounts` `[0]` and trips both assertions.

    func test_mlx_secondTurnReusesPrefixWhileSnapshotTaskStillPending() async throws {
        let mock = MockMLXModelContainer()
        mock.tokensToYield = ["ok"]
        mock.preparedTokenBatches = [
            [101, 102, 103, 104],
            [101, 102, 103, 104, 105],
        ]

        let backend = MLXBackend(enableKVCacheReuse: true)
        backend._inject(mock)

        _ = try await drainEvents(try backend.generate(
            prompt: "turn-1",
            systemPrompt: nil,
            config: GenerationConfig()
        ))

        // After turn 1's stream finishes, the snapshot lineage exists (the
        // capture task is scheduled before the stream's continuation finishes).
        // This is the #1382 race window: turn 2 begins while the task may not
        // yet have written `snapshot`. We do NOT wait for it to settle.
        XCTAssertTrue(
            backend._hasPromptCacheSnapshotForTesting(),
            "Turn 1 must schedule a snapshot lineage — without it the race window doesn't exist and the test is vacuous"
        )

        let secondEvents = try await drainEvents(try backend.generate(
            prompt: "turn-2",
            systemPrompt: nil,
            config: GenerationConfig()
        ))

        XCTAssertEqual(
            reuseCounts(in: secondEvents), [4],
            "Turn 2 must reuse the full 4-token shared prefix even when started during the snapshot task's in-flight window (#1382)"
        )
        XCTAssertEqual(
            mock.lastInitialCacheOffsets, [4],
            "Generation must resume from the restored 4-token prefix, not a cold cache"
        )
    }

    // MARK: - B. Byte-exact prefix trim on divergence

    /// The structural guarantee against the #1382 hazard: reuse is derived from
    /// the byte-exact common prefix, so a divergent prompt can never reuse past
    /// the first differing token. Turn 1 = `[201,202,203,204]`,
    /// turn 2 = `[201,202,999,205]` share exactly two leading tokens, so reuse
    /// must be exactly 2 — never 3 or 4.
    ///
    /// Sabotage: changing turn 2's batch to `[201,999,...]` drops the expected
    /// reuse to 1; keeping `[201,202,203,...]` raises it to 3. Either makes the
    /// `[2]` assertions fail, proving the trim tracks the true common prefix.

    func test_mlx_divergentPromptReusesOnlyByteExactCommonPrefix() async throws {
        let mock = MockMLXModelContainer()
        mock.tokensToYield = ["ok"]
        mock.preparedTokenBatches = [
            [201, 202, 203, 204],
            [201, 202, 999, 205],
        ]

        let backend = MLXBackend(enableKVCacheReuse: true)
        backend._inject(mock)

        _ = try await drainEvents(try backend.generate(
            prompt: "turn-1",
            systemPrompt: nil,
            config: GenerationConfig()
        ))

        let secondEvents = try await drainEvents(try backend.generate(
            prompt: "turn-2",
            systemPrompt: nil,
            config: GenerationConfig()
        ))

        XCTAssertEqual(
            reuseCounts(in: secondEvents), [2],
            "Only the 2-token byte-exact common prefix may be reused after divergence at token index 2"
        )
        XCTAssertEqual(
            mock.lastInitialCacheOffsets, [2],
            "Restored cache must be trimmed to the byte-exact common prefix before generation resumes"
        )
    }
}
