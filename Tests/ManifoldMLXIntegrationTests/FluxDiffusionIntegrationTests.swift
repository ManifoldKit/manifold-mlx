
import XCTest
import ManifoldMLX
import ManifoldInference
import ManifoldTestSupport
import FluxSwift
import MLX
import MLXNN
#if canImport(Darwin)
import Darwin
#endif

/// Current resident memory of this process, in bytes, via
/// `mach_task_basic_info.resident_size` (the same Mach API a prior worker used
/// for the memory-guard, see issue #39). Returns `nil` off Darwin.
private func residentMemoryBytes() -> UInt64? {
    #if canImport(Darwin)
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(
        MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
        }
    }
    return kr == KERN_SUCCESS ? info.resident_size : nil
    #else
    return nil
    #endif
}

/// Metal-bound integration tests for ``FluxDiffusionBackend``.
///
/// These exercise the part of the backend the unit suite cannot: loading a real
/// FLUX.1 Schnell snapshot and running the denoise loop, which allocates and
/// evaluates `MLXArray` values on a Metal device. They auto-skip unless:
/// - running on Apple Silicon with a Metal GPU, and
/// - `MANIFOLD_FLUX_MODEL` points at a directory containing a FLUX model
///   (either flux.swift's quantized `metadata.json` layout, a diffusers FP16
///   layout, or a COMPLETE pre-quantized 4-bit diffusers bundle).
///
/// ## Running against a 4-bit bundle on constrained hardware (issue #39)
///
/// fp16 FLUX.1-schnell is ~33.7 GB resident, so it cannot load on a 24 GB
/// machine. A complete pre-quantized 4-bit bundle (~6–7 GB resident) can. Point
/// `MANIFOLD_FLUX_MODEL` at such a bundle (see `scripts/assemble-flux-4bit-bundle.sh`
/// for how to assemble one) and `test_loadModel_4bitBundle_takesPreQuantizedBranch`
/// asserts the loader actually took the pre-quantized branch
/// (`loadedQuantizedWeights == true`) rather than the fp16-then-quantize path.
/// That test auto-skips for the fp16 / metadata.json layouts.
///
/// **Xcode-only** — the MLX metallib is compiled by Xcode, not `swift build`.
/// Run via `scripts/test-mlx-integration.sh`.
@MainActor
final class FluxDiffusionIntegrationTests: XCTestCase {

    private func requireFluxModelURL() throws -> URL {
        try XCTSkipUnless(HardwareRequirements.isAppleSilicon, "Requires Apple Silicon")
        try XCTSkipUnless(HardwareRequirements.hasMetalDevice, "Requires Metal GPU")

        guard let raw = ProcessInfo.processInfo.environment["MANIFOLD_FLUX_MODEL"],
              !raw.isEmpty else {
            throw XCTSkip("Set MANIFOLD_FLUX_MODEL to a local FLUX.1 model directory to run.")
        }
        let url = URL(fileURLWithPath: raw, isDirectory: true)
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir), isDir.boolValue else {
            throw XCTSkip("MANIFOLD_FLUX_MODEL did not resolve to a directory: \(raw)")
        }
        return url
    }

    /// GAP 2 (issue #39): an mflux `Embedding` carrying a matching `.scales`
    /// tensor must be converted to a `QuantizedEmbedding` by the same gating
    /// predicate `applyPreQuantization` uses, while an fp16 embedding (no
    /// `.scales`) stays a plain `Embedding` (backward compatible). This is
    /// Metal-bound (`quantize`/`MLXArray`) so it lives in the integration suite
    /// but needs no model snapshot.
    func test_quantizedEmbedding_branch_convertsOnlyWhenScalesPresent() throws {
        // Per the integration-target contract (Package.swift): every test here
        // must XCTSkip under plain `swift test` unless run through the Xcode
        // harness (scripts/test-mlx-integration.sh), which injects this marker
        // AND the compiled metallib. Hardware checks alone are insufficient: the
        // CI macOS runner reports a Metal device but has no metallib under
        // `swift test`, so the `quantize`/`MLXArray` calls below would abort the
        // process. Unlike the other tests this one needs no model, so it can't
        // lean on the MANIFOLD_FLUX_MODEL guard.
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["MANIFOLD_DISCOVER_LOCAL_MODELS"] == "1",
            "Metal-bound; run via scripts/test-mlx-integration.sh")
        try XCTSkipUnless(HardwareRequirements.isAppleSilicon, "Requires Apple Silicon")
        try XCTSkipUnless(HardwareRequirements.hasMetalDevice, "Requires Metal GPU")

        final class EmbHolder: Module {
            // `@ModuleInfo` is required so `quantize`'s `update(modules:)` can
            // swap the leaf for a `QuantizedEmbedding`.
            @ModuleInfo var emb: Embedding
            override init() {
                self._emb.wrappedValue = Embedding(embeddingCount: 128, dimensions: 64)
                super.init()
            }
        }

        func runGatedQuantize(_ holder: EmbHolder, weights: [String: MLXArray]) {
            quantize(model: holder, filter: { path, m in
                guard m is Linear || m is Embedding,
                      weights["\(path).scales"] != nil else { return nil }
                return (groupSize: 64, bits: 4)
            })
        }

        // fp16 (no .scales): stays a plain Embedding.
        let fp16Holder = EmbHolder()
        runGatedQuantize(fp16Holder, weights: [:])
        XCTAssertFalse(
            fp16Holder.emb is QuantizedEmbedding,
            "No .scales tensor ⇒ embedding must stay fp16 (backward compatible).")

        // Quantized (.scales present): converted to QuantizedEmbedding.
        let qHolder = EmbHolder()
        runGatedQuantize(qHolder, weights: ["emb.scales": MLXArray([Float(1.0)])])
        XCTAssertTrue(
            qHolder.emb is QuantizedEmbedding,
            "An Embedding with a matching .scales tensor must become a QuantizedEmbedding.")
    }

    func test_loadModel_realSnapshot_setsIsLoaded() async throws {
        let url = try requireFluxModelURL()
        let backend = FluxDiffusionBackend()
        try await backend.loadModel(from: url)
        XCTAssertTrue(backend.isLoaded, "A successful loadModel must flip isLoaded")
        backend.unloadModel()
        XCTAssertFalse(backend.isLoaded)
    }

    /// Issue #39: when `MANIFOLD_FLUX_MODEL` points at a COMPLETE pre-quantized
    /// 4-bit diffusers bundle, the diffusers branch must detect the on-disk
    /// quantization and SKIP the in-memory `quantize(...)` pass — proven by
    /// `loadedQuantizedWeights == true`. Auto-skips for the fp16 layout (where
    /// the flag is false by design) and for the flux.swift `metadata.json`
    /// single-bundle layout (which manages quantization internally).
    func test_loadModel_4bitBundle_takesPreQuantizedBranch() async throws {
        let url = try requireFluxModelURL()

        // The metadata.json single-bundle path is a different loader; this test
        // only covers the diffusers multi-folder pre-quantized path.
        let hasMetadataJson = FileManager.default.fileExists(
            atPath: url.appending(component: "metadata.json").path)
        try XCTSkipIf(
            hasMetadataJson,
            "metadata.json single-bundle layout — covered by FLUX.loadQuantized, not the pre-quantized diffusers branch.")

        // Only assert the pre-quantized branch for an actual 4-bit bundle; the
        // diffusers fp16 layout legitimately loads with loadedQuantizedWeights
        // == false, so skip rather than fail there.
        let layout = FluxBundleLayout.validate(url)
        try XCTSkipUnless(
            layout == .complete,
            "MANIFOLD_FLUX_MODEL is not a complete diffusers bundle: \(layout)")
        let isPreQuantized = FluxModelCore.quantizationConfig(
            in: url.appending(path: "transformer")) != nil
        try XCTSkipUnless(
            isPreQuantized,
            "MANIFOLD_FLUX_MODEL is an fp16 bundle — point it at a 4-bit bundle to exercise the pre-quantized branch.")

        let backend = FluxDiffusionBackend()
        try await backend.loadModel(from: url)
        defer { backend.unloadModel() }

        XCTAssertTrue(backend.isLoaded)
        XCTAssertTrue(
            backend.loadedQuantizedWeights,
            "A complete pre-quantized 4-bit bundle must take the pre-quantized branch and skip the in-memory quantize pass.")
    }

    /// Issue #39 peak-memory regression guard. Loading a COMPLETE pre-quantized
    /// 4-bit bundle must stay far below the fp16 ~33.7 GB resident: the validated
    /// mflux 4-bit load peaked ~17 GB, so a 20 GB ceiling catches a regression
    /// that silently reintroduces an fp16-then-quantize round-trip (which would
    /// blow past 33 GB and OOM a 24 GB machine) without flaking on the real load.
    ///
    /// CI gating (this exact bug bit PR #66): like its sibling Metal tests this
    /// must XCTSkip under plain `swift test`. `requireFluxModelURL()` already
    /// skips when MANIFOLD_FLUX_MODEL is unset (the case on CI), but we ALSO
    /// require the integration-target marker MANIFOLD_DISCOVER_LOCAL_MODELS=1
    /// that scripts/test-mlx-integration.sh injects — the same belt-and-braces
    /// guard `test_quantizedEmbedding_branch_convertsOnlyWhenScalesPresent` uses
    /// so no `quantize`/`MLXArray` call runs without the Xcode-compiled metallib.
    func test_loadModel_4bitBundle_peakMemoryUnderCeiling() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["MANIFOLD_DISCOVER_LOCAL_MODELS"] == "1",
            "Metal-bound; run via scripts/test-mlx-integration.sh")
        let url = try requireFluxModelURL()

        // Only meaningful for a complete pre-quantized 4-bit bundle. fp16 and the
        // metadata.json single-bundle layouts have different peaks, so skip them
        // rather than fail.
        let hasMetadataJson = FileManager.default.fileExists(
            atPath: url.appending(component: "metadata.json").path)
        try XCTSkipIf(hasMetadataJson, "metadata.json single-bundle layout — not the pre-quantized diffusers branch.")
        try XCTSkipUnless(
            FluxBundleLayout.validate(url) == .complete,
            "MANIFOLD_FLUX_MODEL is not a complete diffusers bundle.")
        try XCTSkipUnless(
            FluxModelCore.quantizationConfig(in: url.appending(path: "transformer")) != nil,
            "MANIFOLD_FLUX_MODEL is an fp16 bundle — point it at a 4-bit bundle.")

        // 20 GB ceiling: comfortably below fp16 ~33.7 GB, above the ~17 GB the
        // validated mflux 4-bit load peaked at.
        let ceilingBytes: UInt64 = 20 * 1024 * 1024 * 1024

        let baseline = residentMemoryBytes()
        let backend = FluxDiffusionBackend()
        try await backend.loadModel(from: url)
        defer { backend.unloadModel() }
        XCTAssertTrue(backend.isLoaded)
        XCTAssertTrue(
            backend.loadedQuantizedWeights,
            "A 4-bit bundle must take the pre-quantized branch; an fp16-then-quantize round-trip would blow the memory ceiling.")

        let peak = try XCTUnwrap(residentMemoryBytes(), "Resident memory unavailable")
        let peakGB = Double(peak) / 1_073_741_824.0
        if let baseline {
            let deltaGB = Double(Int64(peak) - Int64(baseline)) / 1_073_741_824.0
            print("[mem-guard] baseline \(Double(baseline) / 1_073_741_824.0) GB -> peak \(peakGB) GB (delta \(deltaGB) GB)")
        }
        XCTAssertLessThan(
            peak, ceilingBytes,
            "Loading a 4-bit bundle peaked at \(peakGB) GB resident, over the 20 GB ceiling — likely a regression to the fp16-then-quantize path (issue #39).")
    }

    func test_generate_realSnapshot_writesPNG() async throws {
        let url = try requireFluxModelURL()
        let outDir = FileManager.default.temporaryDirectory
            .appending(component: "FluxIT-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: outDir) }

        let backend = FluxDiffusionBackend()
        try await backend.loadModel(from: url)
        defer { backend.unloadModel() }

        var config = ImageGenerationConfig(steps: 2, width: 512, height: 512)
        config.outputDirectory = outDir

        var sawProgress = false
        var producedURL: URL?
        let stream = try backend.generate(prompt: "a red apple on a table", config: config)
        for try await event in stream {
            switch event {
            case .progress:
                sawProgress = true
            case .completed(let imageURL):
                producedURL = imageURL
            // TODO: assert on intermediate preview frames once the backend emits
            // ImageGenerationEvent.preview (VAE-decode preview emission is deferred).
            case .preview:
                break
            // ImageGenerationEvent is a non-frozen core enum; @unknown default
            // keeps this compiling across ManifoldKit pin bumps that add cases
            // (same break-class that .promptRendered caused for GenerationEvent).
            @unknown default:
                break
            }
        }

        XCTAssertTrue(sawProgress, "Expected at least one progress tick")
        let finalURL = try XCTUnwrap(producedURL, "Expected a completed image URL")
        XCTAssertTrue(FileManager.default.fileExists(atPath: finalURL.path),
                      "The completed URL must point at a file on disk")
        XCTAssertEqual(finalURL.pathExtension, "png")
    }
}
