
import XCTest
import ManifoldMLX
import ManifoldInference
import ManifoldTestSupport
import FluxSwift

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
/// `MANIFOLD_FLUX_MODEL` at such a bundle (see `Scripts/assemble-flux-4bit-bundle.sh`
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
