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
}
