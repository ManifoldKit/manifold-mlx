
import XCTest
import ManifoldMLX
@_spi(Testing) import ManifoldMLX
import ManifoldInference
import ManifoldTestSupport
import StableDiffusion
import ImageIO
import CoreGraphics
import UniformTypeIdentifiers

/// Metal-bound integration test for ``MLXDiffusionBackend`` against a real
/// SDXL-Turbo diffusers snapshot.
///
/// Phase D (2026-07-22 companion breakage hunt) found that unlike
/// ``FluxDiffusionBackend``, `MLXDiffusionBackend` has ZERO real-weight
/// coverage anywhere in this repo — `MLXDiffusionBackendTests.swift` is
/// unit-only against a fake `DiffusionGenerator`, and no CLI driver exists
/// either. This is the first real invocation of the SDXL path with actual
/// weights.
///
/// Auto-skips unless:
/// - running on Apple Silicon with a Metal GPU, and
/// - `MANIFOLD_SD_MODEL` points at a directory containing a diffusers-layout
///   Stable Diffusion snapshot (`unet/`, `vae/`, `text_encoder/`, `scheduler/`,
///   and — for SDXL — `text_encoder_2/`). A local SDXL-Turbo checkout
///   (`stabilityai/sdxl-turbo`) is the intended target: `detectPreset` picks
///   the SDXL-Turbo preset (cfgWeight 0, steps 2) whenever `text_encoder_2/`
///   is present.
///
/// **Xcode-only** — the MLX metallib is compiled by Xcode, not `swift build`.
/// Run via `scripts/test-mlx-integration.sh`.
@MainActor
final class SDXLDiffusionIntegrationTests: XCTestCase {

    private func requireSDModelURL() throws -> URL {
        try XCTSkipUnless(HardwareRequirements.isAppleSilicon, "Requires Apple Silicon")
        try XCTSkipUnless(HardwareRequirements.hasMetalDevice, "Requires Metal GPU")

        guard let raw = ProcessInfo.processInfo.environment["MANIFOLD_SD_MODEL"],
              !raw.isEmpty else {
            throw XCTSkip("Set MANIFOLD_SD_MODEL to a local diffusers-layout SD/SDXL model directory to run.")
        }
        let url = URL(fileURLWithPath: raw, isDirectory: true)
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir), isDir.boolValue else {
            throw XCTSkip("MANIFOLD_SD_MODEL did not resolve to a directory: \(raw)")
        }
        return url
    }

    /// Confirms `detectPreset` reads a real SDXL-Turbo diffusers checkout the
    /// way Phase D's recon predicted: presence of `text_encoder_2/` selects
    /// the SDXL-Turbo preset over SD 2.1 Base. This needs no Metal work, just
    /// the directory-sniffing logic, so it does not require the Xcode-compiled
    /// metallib marker the Metal-bound tests below require.
    func test_detectPreset_realSDXLTurboSnapshot_resolvesSDXLTurbo() throws {
        let url = try requireSDModelURL()
        let preset = try MLXDiffusionBackend.detectPreset(at: url)
        XCTAssertEqual(
            preset.id, StableDiffusionConfiguration.presetSDXLTurbo.id,
            "MANIFOLD_SD_MODEL has a text_encoder_2/ subdirectory, so detectPreset must resolve SDXL-Turbo, not SD 2.1 Base."
        )
    }

    func test_loadModel_realSnapshot_setsIsLoaded() async throws {
        let url = try requireSDModelURL()
        let backend = MLXDiffusionBackend()
        try await backend.loadModel(from: url)
        XCTAssertTrue(backend.isLoaded, "A successful loadModel must flip isLoaded")
        backend.unloadModel()
        XCTAssertFalse(backend.isLoaded)
    }

    /// The real end-to-end check: load the snapshot, generate one image at
    /// the SDXL-Turbo preset's own defaults (cfgWeight 0, steps 2 —
    /// `StableDiffusionConfiguration.presetSDXLTurbo.defaultParameters`), and
    /// assert the PNG that comes back is a real photograph-shaped image, not
    /// the classic silent-diffusion-failure degenerate output (a uniform
    /// black or single-color frame). `guidanceScale` and `steps` are left at
    /// their `ImageGenerationConfig` init defaults deliberately — `nil`
    /// guidanceScale lets the backend apply the turbo preset's own cfgWeight
    /// 0 rather than a full-SD 7.5 that this distilled model was never tuned
    /// for; `steps: 2` matches the preset. Width/height default to 1024,
    /// SDXL's native resolution.
    func test_generate_realSDXLTurboSnapshot_writesNonDegeneratePNG() async throws {
        let url = try requireSDModelURL()
        let outDir = FileManager.default.temporaryDirectory
            .appending(component: "SDXLIT-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: outDir) }

        let backend = MLXDiffusionBackend()
        try await backend.loadModel(from: url)
        defer { backend.unloadModel() }

        var config = ImageGenerationConfig(steps: 2)
        config.outputDirectory = outDir

        var sawProgress = false
        var producedURL: URL?
        let stream = try backend.generate(prompt: "a red apple on a wooden table, photograph", config: config)
        for try await event in stream {
            switch event {
            case .progress:
                sawProgress = true
            case .completed(let imageURL):
                producedURL = imageURL
            case .preview:
                break
            // ImageGenerationEvent is a non-frozen core enum; @unknown default
            // keeps this compiling across ManifoldKit pin bumps that add cases.
            @unknown default:
                break
            }
        }

        XCTAssertTrue(sawProgress, "Expected at least one progress tick")
        let finalURL = try XCTUnwrap(producedURL, "Expected a completed image URL")
        XCTAssertTrue(FileManager.default.fileExists(atPath: finalURL.path),
                      "The completed URL must point at a file on disk")
        XCTAssertEqual(finalURL.pathExtension, "png")

        let attrs = try FileManager.default.attributesOfItem(atPath: finalURL.path)
        let byteSize = (attrs[.size] as? Int) ?? 0
        // An SDXL-resolution (1024x1024) PNG of real photographic content
        // compresses to well over 50 KB; a uniform/degenerate frame (solid
        // black, solid gray) compresses to a few KB because PNG's filter +
        // deflate collapse a constant image almost entirely.
        XCTAssertGreaterThan(
            byteSize, 50_000,
            "Output PNG is only \(byteSize) bytes — too small for real 1024x1024 photographic "
            + "content, suggests a degenerate (near-uniform) image."
        )

        let stats = try Self.pixelStatistics(of: finalURL)
        XCTAssertGreaterThan(
            stats.luminanceStdDev, 5.0,
            "Sampled pixel luminance has almost no variance (\(stats.luminanceStdDev)) — "
            + "this is the classic silent-diffusion-failure shape (a uniform black or flat-color frame), "
            + "not real generated content."
        )
        XCTAssertGreaterThan(
            stats.meanLuminance, 2.0,
            "Sampled mean luminance (\(stats.meanLuminance)) is near-zero — image is effectively all black."
        )
    }

    // MARK: - Pixel sampling

    private struct PixelStatistics {
        let meanLuminance: Double
        let luminanceStdDev: Double
    }

    /// Decodes the PNG at `url` and computes mean + standard deviation of
    /// per-pixel luminance across a downsampled grid. A uniform/black image
    /// (the classic silent diffusion failure — VAE decode returning all
    /// zeros, or a denoise loop that never actually stepped) reads as a
    /// near-zero standard deviation; real photographic content does not.
    private static func pixelStatistics(of url: URL) throws -> PixelStatistics {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw XCTSkip("Could not decode PNG at \(url.path) for pixel inspection.")
        }

        // Downsample into a small fixed grid via CGContext — cheap and avoids
        // hand-parsing the PNG's native bit depth/color space.
        let side = 32
        var buffer = [UInt8](repeating: 0, count: side * side * 4)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: &buffer,
            width: side, height: side,
            bitsPerComponent: 8, bytesPerRow: side * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw XCTSkip("Could not create a CGContext to sample \(url.path).")
        }
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: side, height: side))

        var luminances: [Double] = []
        luminances.reserveCapacity(side * side)
        for pixel in 0..<(side * side) {
            let offset = pixel * 4
            let r = Double(buffer[offset])
            let g = Double(buffer[offset + 1])
            let b = Double(buffer[offset + 2])
            luminances.append(0.299 * r + 0.587 * g + 0.114 * b)
        }
        let mean = luminances.reduce(0, +) / Double(luminances.count)
        let variance = luminances.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Double(luminances.count)
        return PixelStatistics(meanLuminance: mean, luminanceStdDev: variance.squareRoot())
    }
}
