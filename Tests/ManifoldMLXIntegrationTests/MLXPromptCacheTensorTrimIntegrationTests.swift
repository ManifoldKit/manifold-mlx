import XCTest
@preconcurrency import MLX
import MLXLMCommon
import ManifoldTestSupport
@_spi(Testing) import ManifoldMLX

/// Real-tensor coverage for the prompt-cache `copy()` / state-slicing /
/// `trim(excess)` materialisation paths (#27): the layer the offset-only
/// `MockMLXModelContainer` in the unit suite deliberately stubs out, and which
/// the original audit flagged as exercised only by the hardware suite.
///
/// These drive `captureSnapshot` and `restorePromptCache` against real
/// `KVCacheSimple` objects holding small KV tensors and call MLX `eval`, so they
/// require a Metal GPU + the compiled metallib (present in the Xcode/xcodebuild
/// test bundle, absent under plain `swift test` — where a real `eval` aborts the
/// process). Hardware-gated like the rest of this suite; the control-flow
/// decline branches that need no `eval` live in the headless unit suite
/// (`MLXPromptCacheTrimDeclineTests`).
///
/// No model load is required — only the MLX tensor runtime — but we still gate
/// on `hasMetalDevice` so the suite skips cleanly off-hardware.
@MainActor
final class MLXPromptCacheTensorTrimIntegrationTests: XCTestCase {

    private typealias Coordinator = MLXPromptCacheCoordinator

    override func setUp() async throws {
        try await super.setUp()
        try XCTSkipUnless(HardwareRequirements.isAppleSilicon, "Requires Apple Silicon")
        try XCTSkipUnless(HardwareRequirements.hasMetalDevice, "Requires Metal GPU")
        // `hasMetalDevice` only proves a Metal *device* exists — the compiled
        // metallib (MLX GPU kernels) ships only in the Xcode/xcodebuild test
        // bundle, NOT under plain `swift test`, where a real `eval` aborts the
        // process with "Failed to load the default metallib". The rest of this
        // suite couples Metal use with a discoverable model; we mirror that gate
        // (a model load initialises the runtime + the bundle carries the
        // metallib) so this file runs only under `scripts/test-mlx-integration.sh`
        // / Xcode and skips cleanly in the CI `swift test` lane.
        try XCTSkipIf(
            HardwareRequirements.findMLXModelDirectory() == nil,
            "Set MLX_TEST_MODEL / run via scripts/test-mlx-integration.sh so the metallib is present."
        )
    }

    /// Builds a `KVCacheSimple` holding `tokenCount` tokens of real (tiny) KV
    /// tensor state by driving its `update(keys:values:)` path — exactly how a
    /// model populates a cache during prefill. Shape `[1, 2, tokenCount, 4]`.
    private func makeSimpleCache(tokenCount: Int) -> KVCacheSimple {
        let cache = KVCacheSimple()
        let shape = [1, 2, tokenCount, 4]
        let keys = MLXArray(0 ..< Int32(shape.reduce(1, *)))
            .reshaped(shape)
            .asType(.float32)
        let values = keys + Float(100)
        _ = cache.update(keys: keys, values: values)
        eval(cache.state)
        return cache
    }

    // MARK: - captureSnapshot: real copy() + state slice + trim(excess)

    /// When the live cache holds a generated tail beyond the prompt, capture must
    /// `copy()`, `trim(excess)` back to the prompt length, and the resulting
    /// snapshot's tensor state must reflect the trimmed offset — without mutating
    /// the live cache (capture operates on a copy).
    func test_captureSnapshot_trimsGeneratedTailToPromptLength() {
        let promptTokens = [10, 11, 12, 13] // 4 prompt tokens
        // Cache holds 4 prompt + 3 generated = 7 tokens of real tensor state.
        let cache = MLXPromptCache([makeSimpleCache(tokenCount: 7)])

        let (snapshot, reason) = Coordinator.captureSnapshot(
            from: cache,
            promptTokens: promptTokens
        )

        XCTAssertEqual(reason.description, "captured(1 layers)")
        let snap = try! XCTUnwrap(snapshot)
        XCTAssertEqual(snap.layers.count, 1)
        let layer = snap.layers[0]
        XCTAssertEqual(layer.offset, 4,
            "trim(excess) must reduce the copied cache to exactly the prompt length")
        XCTAssertEqual(layer.cacheTypeName, "KVCacheSimple")
        XCTAssertEqual(layer.state.count, 2, "KVCacheSimple state is [keys, values]")
        eval(layer.state)
        XCTAssertEqual(layer.state[0].dim(2), 4,
            "Sliced key tensor must have seqLen == prompt length after trim")
        XCTAssertEqual(layer.state[1].dim(2), 4,
            "Sliced value tensor must have seqLen == prompt length after trim")

        XCTAssertEqual(cache.value[0].offset, 7,
            "capture must trim a copy, never mutate the live cache")
    }

    /// A cache whose offset is below the prompt length cannot have its prompt
    /// prefix recovered — capture must decline with the diagnostic reason.
    func test_captureSnapshot_offsetBelowPrompt_declines() {
        let cache = MLXPromptCache([makeSimpleCache(tokenCount: 2)])
        let (snapshot, reason) = Coordinator.captureSnapshot(
            from: cache,
            promptTokens: [1, 2, 3, 4] // 4 > cached 2
        )
        XCTAssertNil(snapshot)
        XCTAssertEqual(reason.description, "layer 0 offset is below the prompt length")
    }

    // MARK: - restorePromptCache: post-restore offset + trim arithmetic

    private func makeSnapshot(promptTokens: [Int], cacheTokenCount: Int) -> Coordinator.Snapshot {
        let cache = MLXPromptCache([makeSimpleCache(tokenCount: cacheTokenCount)])
        let (snapshot, reason) = Coordinator.captureSnapshot(from: cache, promptTokens: promptTokens)
        XCTAssertTrue(reason.description.hasPrefix("captured"),
            "Snapshot fixture must capture cleanly, got: \(reason.description)")
        return snapshot!
    }

    /// Restoring a snapshot into a fresh live cache must install the tensor state,
    /// then trim the surplus so the post-restore offset equals
    /// `reusedPromptTokenCount`.
    func test_restorePromptCache_offsetEqualsReusedCount() {
        let snapshot = makeSnapshot(promptTokens: Array(20 ..< 26), cacheTokenCount: 6)
        let live = MLXPromptCache([KVCacheSimple()])

        let reason = Coordinator.restorePromptCache(
            snapshot,
            into: live,
            reusedPromptTokenCount: 4
        )

        XCTAssertEqual(reason, .reused(promptTokensReused: 4))
        XCTAssertEqual(live.value[0].offset, 4,
            "Post-restore offset must equal reusedPromptTokenCount after trim(excess)")
    }

    /// Verbatim continuation: restoring at exactly the cached length needs no trim
    /// and the offset must land on the full prompt length.
    func test_restorePromptCache_verbatimContinuation_noTrim() {
        let snapshot = makeSnapshot(promptTokens: Array(30 ..< 35), cacheTokenCount: 5)
        let live = MLXPromptCache([KVCacheSimple()])

        let reason = Coordinator.restorePromptCache(
            snapshot,
            into: live,
            reusedPromptTokenCount: 5
        )

        XCTAssertEqual(reason, .reused(promptTokensReused: 5))
        XCTAssertEqual(live.value[0].offset, 5)
    }

    /// A trimmable cache whose `trim` reports the *wrong* amount (an MLX-library
    /// trim-contract regression) must be caught by restore's authoritative
    /// post-restore guard (`target.trim(excess) == excess`) and reported as
    /// not-reusable, naming the offending layer — never silently accepted as a
    /// partial restore. Requires the non-empty tensor-state branch (real
    /// `MLXArray` state), hence the Metal gate.
    func test_restorePromptCache_trimReturnsWrongAmount_declinesNotReusable() {
        let typeName = String(describing: LyingTrimKVCache.self)
        let shape = [1, 2, 6, 4]
        let keys = MLXArray(0 ..< Int32(shape.reduce(1, *)))
            .reshaped(shape)
            .asType(.float32)
        let state = [keys, keys + Float(100)]
        eval(state)
        let snapshot = Coordinator.Snapshot(
            promptTokens: Array(40 ..< 46),
            layers: [
                Coordinator.CachedLayerState(
                    cacheTypeName: typeName,
                    offset: 6,
                    state: state,
                    metaState: [""]
                )
            ]
        )
        let live = MLXPromptCache([LyingTrimKVCache()])

        // Reuse 4 of 6 ⇒ excess 2, but the lying cache trims only 1.
        let reason = Coordinator.restorePromptCache(
            snapshot,
            into: live,
            reusedPromptTokenCount: 4
        )

        XCTAssertEqual(reason, .layerNotReusable(layerIndex: 0, cacheTypeName: typeName),
            "A trim that removes the wrong amount must be rejected as not-reusable")
    }
}

// MARK: - Fakes

/// A trimmable `KVCache` double whose `trim(_:)` deliberately removes ONE FEWER
/// token than requested — modeling an MLX-library regression where the trim
/// contract is violated. Proves `restorePromptCache`'s authoritative post-restore
/// guard (`target.trim(excess) == excess`) rejects it rather than silently
/// accepting a partial restore.
private final class LyingTrimKVCache: KVCache {
    private var keys: MLXArray?
    private var values: MLXArray?
    var offset: Int = 0
    var maxSize: Int? { nil }

    init() {}

    func innerState() -> [MLXArray] { [keys, values].compactMap { $0 } }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        (keys, values)
    }

    var state: [MLXArray] {
        get {
            guard let keys, let values else { return [] }
            return [keys, values]
        }
        set {
            guard newValue.count == 2 else {
                self.keys = nil
                self.values = nil
                self.offset = 0
                return
            }
            self.keys = newValue[0]
            self.values = newValue[1]
            self.offset = newValue[0].dim(2)
        }
    }

    var metaState: [String] {
        get { [""] }
        set { _ = newValue }
    }

    var isTrimmable: Bool { true }

    /// Removes one fewer than asked — the contract violation under test.
    @discardableResult
    func trim(_ n: Int) -> Int {
        let lie = max(n - 1, 0)
        offset -= lie
        return lie
    }

    func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        .none
    }

    func copy() -> any KVCache {
        let new = LyingTrimKVCache()
        new.state = state.map { $0[.ellipsis] }
        return new
    }
}
