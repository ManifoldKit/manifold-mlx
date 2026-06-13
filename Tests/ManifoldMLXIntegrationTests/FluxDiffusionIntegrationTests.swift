
import XCTest
import ManifoldMLX
import ManifoldInference
import ManifoldTestSupport

/// Metal-bound integration tests for ``FluxDiffusionBackend``.
///
/// These exercise the part of the backend the unit suite cannot: loading a real
/// FLUX.1 Schnell snapshot and running the denoise loop, which allocates and
/// evaluates `MLXArray` values on a Metal device. They auto-skip unless:
/// - running on Apple Silicon with a Metal GPU, and
/// - `MANIFOLD_FLUX_MODEL` points at a directory containing a FLUX model
///   (either flux.swift's quantized `metadata.json` layout or a diffusers
///   FP16 layout).
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
            }
        }

        XCTAssertTrue(sawProgress, "Expected at least one progress tick")
        let finalURL = try XCTUnwrap(producedURL, "Expected a completed image URL")
        XCTAssertTrue(FileManager.default.fileExists(atPath: finalURL.path),
                      "The completed URL must point at a file on disk")
        XCTAssertEqual(finalURL.pathExtension, "png")
    }
}
