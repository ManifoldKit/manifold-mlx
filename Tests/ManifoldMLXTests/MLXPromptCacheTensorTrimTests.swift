import XCTest
import MLXLMCommon
@_spi(Testing) import ManifoldMLX

/// Headless coverage for `restorePromptCache`'s **control flow** — the decline
/// branches that return *before* any tensor work, so they run in the plain
/// `swift test` lane (no Metal / metallib). These exercise the authoritative
/// restore guards (zero-reuse, layer-count mismatch, per-layer type mismatch)
/// that the offset-only `MockMLXModelContainer` never reaches.
///
/// Real tensor `copy()` / state-slicing / `trim(excess)` *materialisation* — and
/// the trim-returns-wrong-amount guard, which must take the tensor branch — live
/// in `MLXPromptCacheTensorTrimIntegrationTests` (Metal-gated): constructing or
/// operating on a real `MLXArray` schedules GPU work that aborts the process
/// when the metallib is absent (the same constraint that kept these paths in the
/// Xcode-only suite — #27). All snapshots here use empty (`[]`) layer state so
/// no `MLXArray` is ever built.
@MainActor
final class MLXPromptCacheTrimDeclineTests: XCTestCase {

    private typealias Coordinator = MLXPromptCacheCoordinator

    /// Snapshot whose single layer carries NO tensor state — restore's decline
    /// guards we test here all fire before the state/trim branch, so no real
    /// `MLXArray` is needed (and none is built).
    private func emptyStateSnapshot(typeName: String, offset: Int, promptTokens: [Int]) -> Coordinator.Snapshot {
        Coordinator.Snapshot(
            promptTokens: promptTokens,
            layers: [
                Coordinator.CachedLayerState(
                    cacheTypeName: typeName,
                    offset: offset,
                    state: [],
                    metaState: [""]
                )
            ]
        )
    }

    /// Reuse of zero tokens is never valid — restore must decline immediately,
    /// before inspecting any layer.
    func test_restorePromptCache_zeroReuse_declines() {
        let snap = emptyStateSnapshot(typeName: "KVCacheSimple", offset: 3, promptTokens: [1, 2, 3])
        let live = MLXPromptCache([KVCacheSimple()])

        let reason = Coordinator.restorePromptCache(
            snap,
            into: live,
            reusedPromptTokenCount: 0
        )

        XCTAssertEqual(reason, .noCommonPrefix)
    }

    /// When the snapshot/live layer counts disagree the restore must decline
    /// with the layer-count-mismatch reason before touching any layer.
    func test_restorePromptCache_layerCountMismatch_declines() {
        let snap = emptyStateSnapshot(typeName: "KVCacheSimple", offset: 3, promptTokens: [1, 2, 3])
        let live = MLXPromptCache([KVCacheSimple(), KVCacheSimple()])

        let reason = Coordinator.restorePromptCache(
            snap,
            into: live,
            reusedPromptTokenCount: 2
        )

        XCTAssertEqual(reason, .layerCountMismatch(snapshot: 1, live: 2))
    }

    /// A live layer whose runtime type differs from the snapshot's recorded type
    /// must decline with a type-mismatch reason — the guard fires before the
    /// state-restore branch, so empty state suffices.
    func test_restorePromptCache_typeMismatch_declines() {
        let snap = emptyStateSnapshot(typeName: "RotatingKVCache", offset: 4, promptTokens: [1, 2, 3, 4])
        let live = MLXPromptCache([KVCacheSimple()]) // live type != recorded type

        let reason = Coordinator.restorePromptCache(
            snap,
            into: live,
            reusedPromptTokenCount: 3
        )

        XCTAssertEqual(
            reason,
            .layerTypeMismatch(layerIndex: 0, expected: "RotatingKVCache", found: "KVCacheSimple")
        )
    }

    /// Empty-state restore into a fresh `KVCacheSimple` sets the offset directly
    /// (the no-tensor continuation path, lines ~526–534) and trims the surplus —
    /// `KVCacheSimple.trim` is pure integer arithmetic on a nil-tensor cache, so
    /// this reuse-success path runs headless without any `eval`.
    func test_restorePromptCache_emptyState_setsOffsetAndTrimsToReuseLength() {
        let snap = emptyStateSnapshot(typeName: "KVCacheSimple", offset: 6, promptTokens: Array(20 ..< 26))
        let live = MLXPromptCache([KVCacheSimple()])

        let reason = Coordinator.restorePromptCache(
            snap,
            into: live,
            reusedPromptTokenCount: 4
        )

        XCTAssertEqual(reason, .reused(promptTokensReused: 4))
        XCTAssertEqual(live.value[0].offset, 4,
            "Empty-state restore must set offset to the snapshot length then trim to the reuse count")
    }

    /// Restore into a non-KVCacheSimple live cache when the snapshot carries
    /// empty state must decline: the empty-state branch can only set `offset`
    /// directly on a `KVCacheSimple` (the only KVCache subtype that exposes a
    /// mutable `offset`). Any other type must be treated as not reusable.
    func test_restorePromptCache_emptyState_nonSimpleLive_declines() {
        // Snapshot records a RotatingKVCache layer with empty state.
        let snap = Coordinator.Snapshot(
            promptTokens: Array(1 ..< 5),
            layers: [
                Coordinator.CachedLayerState(
                    cacheTypeName: "RotatingKVCache",
                    offset: 4,
                    state: [],
                    metaState: [""]
                )
            ]
        )
        let live = MLXPromptCache([RotatingKVCache(maxSize: 16)])

        let reason = Coordinator.restorePromptCache(
            snap,
            into: live,
            reusedPromptTokenCount: 3
        )

        XCTAssertEqual(
            reason,
            .layerNotReusable(layerIndex: 0, cacheTypeName: "RotatingKVCache"),
            "An empty-state layer in a non-KVCacheSimple cache cannot have its offset set directly"
        )
    }
}

// MARK: - Capture control-flow

/// Headless coverage for `captureSnapshot`'s **control-flow branches** that fire
/// before any tensor work — so they run in the plain `swift test` lane (no
/// Metal). All branches here return `nil` or produce empty-state layers that
/// require only integer arithmetic on `KVCacheSimple.offset` / `.trim`.
///
/// The tensor copy/eval/trim materialisation path lives in
/// `MLXPromptCacheTensorTrimIntegrationTests` (Metal-gated). (#27)
@MainActor
final class MLXPromptCacheCaptureTests: XCTestCase {

    private typealias Coordinator = MLXPromptCacheCoordinator

    // MARK: - Early-exit guards

    /// An empty prompt token array must immediately return nil with `.emptyPrompt`
    /// — there is no valid prompt-prefix state to snapshot.
    func test_captureSnapshot_emptyPrompt_returnsNilAndEmptyPromptReason() {
        let cache = MLXPromptCache([KVCacheSimple()])

        let (snapshot, reason) = Coordinator.captureSnapshot(from: cache, promptTokens: [])

        XCTAssertNil(snapshot)
        XCTAssertEqual(reason, .emptyPrompt)
    }

    /// A cache with no layers must return nil with `.noCaches` — there is nothing
    /// to snapshot even if the prompt is non-empty.
    func test_captureSnapshot_noCaches_returnsNilAndNoCachesReason() {
        let emptyCache = MLXPromptCache([])

        let (snapshot, reason) = Coordinator.captureSnapshot(from: emptyCache, promptTokens: [1, 2, 3])

        XCTAssertNil(snapshot)
        XCTAssertEqual(reason, .noCaches)
    }

    /// When a layer's offset is below the prompt token count the capture must
    /// abort with `.layerOffsetBelowPrompt` — the prompt-prefix KV state for that
    /// layer was never fully computed (e.g. the generation was cancelled early).
    func test_captureSnapshot_layerOffsetBelowPrompt_returnsNilWithLayerIndex() {
        let layer = KVCacheSimple()
        layer.offset = 1  // below promptTokens.count = 3
        let cache = MLXPromptCache([layer])

        let (snapshot, reason) = Coordinator.captureSnapshot(from: cache, promptTokens: [10, 20, 30])

        XCTAssertNil(snapshot)
        XCTAssertEqual(reason, .layerOffsetBelowPrompt(layerIndex: 0))
    }

    // MARK: - Empty-state capture path

    /// When a `KVCacheSimple` holds no tensor state (`state.isEmpty == true`) the
    /// capture path skips `copy()`/`eval`/`trim` and records the layer with
    /// `offset = promptTokens.count` directly. This is the offset-only code path
    /// that `MockMLXModelContainer` exercises in the integration-facing backend
    /// tests; this unit test pins it at the coordinator level so the logic is
    /// verified without a model container in the loop.
    ///
    /// `KVCacheSimple.offset` is a plain stored `var`; setting it and calling
    /// `trim` are pure integer operations — no `MLXArray` is allocated, so this
    /// runs headless.
    func test_captureSnapshot_emptyStateLayer_capturesWithPromptLength() {
        let layer = KVCacheSimple()
        // Simulate post-generation offset: prompt (3) + generated tail (2).
        layer.offset = 5
        let cache = MLXPromptCache([layer])
        let promptTokens = [1, 2, 3]

        let (snapshot, reason) = Coordinator.captureSnapshot(from: cache, promptTokens: promptTokens)

        XCTAssertEqual(reason, .captured(layers: 1))
        let captured = try! XCTUnwrap(snapshot)
        XCTAssertEqual(captured.layers.count, 1)
        XCTAssertEqual(captured.layers[0].cacheTypeName, "KVCacheSimple")
        XCTAssertEqual(
            captured.layers[0].offset, promptTokens.count,
            "Empty-state capture must record the prompt length, not the post-generation offset"
        )
        XCTAssertTrue(
            captured.layers[0].state.isEmpty,
            "Empty-state capture must not materialise any MLXArray tensors"
        )
    }

    /// Multi-layer snapshot where the first layer is empty-state and the second
    /// would require Metal to copy — aborts on the second layer with
    /// `.layerOffsetBelowPrompt` when that layer's offset was never advanced past
    /// the prompt boundary.
    func test_captureSnapshot_multiLayer_firstEmptySecondBelowPrompt_abortsWithReason() {
        let first = KVCacheSimple()
        first.offset = 3  // matches promptTokens.count — will be captured
        let second = KVCacheSimple()
        second.offset = 2  // below promptTokens.count = 3 — must abort
        let cache = MLXPromptCache([first, second])
        let promptTokens = [1, 2, 3]

        let (snapshot, reason) = Coordinator.captureSnapshot(from: cache, promptTokens: promptTokens)

        XCTAssertNil(snapshot, "Any per-layer failure must abort the entire capture")
        XCTAssertEqual(reason, .layerOffsetBelowPrompt(layerIndex: 1))
    }
}
