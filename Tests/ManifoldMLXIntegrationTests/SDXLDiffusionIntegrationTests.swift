
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
/// - `MANIFOLD_SD_MODEL` points at a directory containing an SDXL-Turbo
///   diffusers-layout snapshot (`unet/`, `vae/`, `text_encoder/`,
///   `text_encoder_2/`, `scheduler/`) — `text_encoder_2/` is what makes it
///   SDXL rather than SD 2.1 Base. A local SDXL-Turbo checkout
///   (`stabilityai/sdxl-turbo`) is the intended target: `detectPreset` picks
///   the SDXL-Turbo preset (cfgWeight 0, steps 2) whenever `text_encoder_2/`
///   is present. These tests hard-assert the SDXL-Turbo preset's own values
///   (steps 2, 1024x1024) — they are NOT a generic SD-family harness; an
///   SD 2.1 Base snapshot (no `text_encoder_2/`, native 768x768) would fail
///   the step-count and image-size assumptions below.
///
/// **Xcode-only** — the MLX metallib is compiled by Xcode, not `swift build`.
/// Run via `scripts/test-mlx-integration.sh`.
///
/// **First run is slow — this is expected, not a hang.** The first real
/// invocation against a fresh `.build` directory pays a one-time Metal shader
/// JIT compile at SDXL's 1024x1024 resolution (a wall of
/// `[Metal Compiler Warning]` lines in the log is normal, not an error) —
/// verified 2026-07-22: ~4 minutes end to end on an Apple Silicon M-series
/// machine. Subsequent runs reuse the compiled shader cache and the actual
/// `generate` test drops to ~9 seconds. Do not kill a run that looks quiet
/// for a few minutes on its first invocation.
///
/// **Generated images are copied to a durable path for human inspection**
/// (`$TMPDIR/manifold-mlx-phase-d/sdxl-turbo-<uuid>.png`, printed to the test
/// log) before the per-test `outDir` is deleted — deliberately unconditional,
/// not gated behind an env var, because the automated byte-size + pixel-
/// variance checks below can pass on a coherent-but-wrong image (right
/// statistics, garbage content) that only a human glance catches.
@MainActor
final class SDXLDiffusionIntegrationTests: XCTestCase {

    override func setUp() {
        super.setUp()
        // XCTest's own per-test execution-time-allowance defaults to 600s.
        // Nothing enables that mechanism today (the script passes neither
        // `-test-timeouts-enabled` nor
        // `-default-test-execution-time-allowance`), so it's inert now, but
        // `test_generate_realSDXLTurboSnapshot_writesNonDegeneratePNG`'s own
        // 900s deadline exceeds it — if that mechanism is ever switched on,
        // XCTest's own 600s would fire first and hard-kill the runner
        // instead of yielding that test's own clean SDXLGenerateTimeoutError.
        // Set here (not mid-test) so the allowance is armed before whatever
        // internal timer XCTest starts it against, whichever that turns out
        // to be once the mechanism is actually enabled.
        self.executionTimeAllowance = 1200
    }

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

    /// Per the integration-target contract (Package.swift) and the identical
    /// guard `FluxDiffusionIntegrationTests.swift:88-90` applies: hardware
    /// checks alone are insufficient for a test that actually drives MLX/GPU
    /// work. The CI macOS runner reports a Metal device but has no compiled
    /// metallib under plain `swift test` — running the denoise loop there
    /// hard-aborts the whole test binary with "Failed to load the default
    /// metallib" instead of cleanly skipping. Only the Xcode harness
    /// (`scripts/test-mlx-integration.sh`) sets this marker, so call this from
    /// every test that loads/generates, but NOT from
    /// `test_detectPreset_realSDXLTurboSnapshot_resolvesSDXLTurbo`, which is
    /// pure directory-sniffing and needs no metallib.
    private func requireMetalBoundTestMarker() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["MANIFOLD_DISCOVER_LOCAL_MODELS"] == "1",
            "Metal-bound; run via scripts/test-mlx-integration.sh")
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
        try requireMetalBoundTestMarker()
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
        try requireMetalBoundTestMarker()
        let url = try requireSDModelURL()
        let outDir = FileManager.default.temporaryDirectory
            .appending(component: "SDXLIT-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: outDir) }

        let backend = MLXDiffusionBackend()
        try await backend.loadModel(from: url)
        defer { backend.unloadModel() }

        var config = ImageGenerationConfig(steps: 2)
        config.outputDirectory = outDir

        // Deadline-raced collection — same pattern as
        // `MLXVLMRealWeightVisionTests.runTurn` (VLM stream hangs are a named
        // recurring hazard, issue #26). SDXL pays a documented ~4min shader
        // JIT compile on first run plus a 10.27GB fp32 unet; a bare
        // `for try await` here would hang the whole
        // `ManifoldMLXIntegrationTests` target on a stalled denoise loop with
        // no way to fail the run. The timeout is generous (600s+) to comfortably
        // cover that first-run JIT.
        struct SDXLGenerateTimeoutError: Error, CustomStringConvertible {
            let seconds: Double
            var description: String { "SDXL generate stream timed out after \(seconds)s — possible denoise-loop hang" }
        }
        let timeoutSeconds: Double = 900
        // Create the stream on the main actor (it captures `backend`/`config`,
        // both main-actor-isolated) and hand only the already-created,
        // `Sendable` stream into the detached collection task below.
        let stream = try backend.generate(prompt: "a red apple on a wooden table, photograph", config: config)
        let events: [ImageGenerationEvent] = try await withThrowingTaskGroup(of: [ImageGenerationEvent].self) { group in
            group.addTask {
                var collected: [ImageGenerationEvent] = []
                for try await event in stream {
                    collected.append(event)
                }
                return collected
            }
            group.addTask {
                try await Task.sleep(nanoseconds: UInt64(timeoutSeconds * 1_000_000_000))
                throw SDXLGenerateTimeoutError(seconds: timeoutSeconds)
            }
            defer {
                group.cancelAll()
                // This defer runs on EVERY exit of the group's closure, not
                // only the timeout branch — including the normal success
                // path, right after `group.next()` returns below.
                // `group.cancelAll()` only cancels the collect Task above;
                // the detached denoise loop inside `backend.generate()` is a
                // separately-owned Task.detached that checks
                // `_stopRequested` per iteration (MLXDiffusionBackend.swift),
                // not `Task.isCancelled`, and `onTermination(.cancelled)` on
                // the underlying AsyncThrowingStream is documented
                // unreliable (MLXDiffusionBackend.swift:230-234) — so on a
                // real timeout, without this explicit call, the denoise loop
                // could keep running (and keep `backend`'s `_generator`
                // referenced) even after this test function returns and its
                // `unloadModel()` defer clears `_generator` out from under
                // it. `stopGeneration()` sets `_stopRequested` directly,
                // which the loop's own per-iteration check honors.
                //
                // On the success path this call is a verified no-op: by the
                // time `group.next()` returns the winning collect Task's
                // result, `_isGenerating` is already `false` (set before
                // `continuation.finish()`), nothing downstream reads
                // `_stopRequested` after that point, and both
                // `unloadModel()` and the next `generate()` call reset it
                // anyway. It runs unconditionally here (rather than only on
                // the timeout branch) because there's no way to distinguish
                // the two branches from inside this shared `defer` — that's
                // fine precisely because it's a no-op on the branch where it
                // doesn't matter.
                backend.stopGeneration()
            }
            return try await group.next() ?? []
        }

        var sawProgress = false
        var lastStep = 0
        var lastTotal = 0
        var producedURL: URL?
        for event in events {
            switch event {
            case .progress(let step, let total):
                sawProgress = true
                lastStep = step
                lastTotal = total
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
        // Turbo models run 1-4 denoising steps (the preset's own default is 2);
        // many more would mean the config path is mis-wired and this run's
        // output would not mean what the test thinks. Always logged, pass or
        // fail, per the same "make the evidence visible" reasoning as the VLM
        // test's reply logging.
        print("[SDXLDiffusionIntegrationTests] progress: step=\(lastStep) total=\(lastTotal)")

        XCTAssertTrue(sawProgress, "Expected at least one progress tick")
        // `makeParams` passes `config.steps` straight through, so these are
        // well-defined, not a loose sanity check: a mismatch means the config
        // path is mis-wired and this run's output doesn't mean what the test
        // thinks it does.
        XCTAssertEqual(lastStep, 2, "Expected the final progress step to match config.steps (2)")
        XCTAssertEqual(lastTotal, 2, "Expected the reported total steps to match config.steps (2)")
        let finalURL = try XCTUnwrap(producedURL, "Expected a completed image URL")
        XCTAssertTrue(FileManager.default.fileExists(atPath: finalURL.path),
                      "The completed URL must point at a file on disk")
        XCTAssertEqual(finalURL.pathExtension, "png")

        // Copy the produced image to a stable, non-cleaned-up location before
        // this test's `defer` deletes `outDir` — a human needs to actually look
        // at it (an automated variance check can pass on a coherent-but-wrong
        // image; a glance catches what the assertion can't).
        let debugDir = FileManager.default.temporaryDirectory.appending(component: "manifold-mlx-phase-d")
        try? FileManager.default.createDirectory(at: debugDir, withIntermediateDirectories: true)
        let debugCopyURL = debugDir.appending(component: "sdxl-turbo-\(UUID().uuidString).png")
        do {
            try FileManager.default.copyItem(at: finalURL, to: debugCopyURL)
            print("[SDXLDiffusionIntegrationTests] saved a durable copy for human inspection: \(debugCopyURL.path)")
        } catch {
            print("[SDXLDiffusionIntegrationTests] FAILED to save a durable copy to \(debugCopyURL.path): \(error)")
        }

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

        let stats = try XCTUnwrap(
            Self.pixelStatistics(of: finalURL),
            "The backend just wrote \(finalURL.path) as a PNG a moment ago; if it cannot be decoded "
            + "for pixel inspection that is a defect (a truncated/malformed PNG), not a reason to skip "
            + "the degeneracy check this test exists to run."
        )
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
    ///
    /// Returns `nil` (never `XCTSkip`s) when the PNG can't be decoded — a
    /// truncated/malformed PNG the backend just wrote a moment ago is a
    /// defect, not a reason to mark the whole test skipped-green with zero
    /// degeneracy coverage. The call site `XCTUnwrap`s this.
    private static func pixelStatistics(of url: URL) -> PixelStatistics? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            return nil
        }

        // Downsample into a small fixed grid via CGContext — cheap and avoids
        // hand-parsing the PNG's native bit depth/color space.
        let side = 32
        var buffer = [UInt8](repeating: 0, count: side * side * 4)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        // `buffer`'s address is only stable for the duration of this closure —
        // `CGContext(data:)` does NOT copy or retain the array's storage, it
        // writes through the raw pointer handed to it. Creating the context
        // from `&buffer` directly and using it (and reading `buffer`) after
        // the call returns is escaping-pointer UB. `withUnsafeMutableBytes`
        // keeps context creation, the draw, and the read loop all within the
        // pointer's valid lifetime.
        return buffer.withUnsafeMutableBytes { raw -> PixelStatistics? in
            guard let context = CGContext(
                data: raw.baseAddress,
                width: side, height: side,
                bitsPerComponent: 8, bytesPerRow: side * 4,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            ) else {
                return nil
            }
            context.draw(cgImage, in: CGRect(x: 0, y: 0, width: side, height: side))

            var luminances: [Double] = []
            luminances.reserveCapacity(side * side)
            for pixel in 0..<(side * side) {
                let offset = pixel * 4
                let r = Double(raw[offset])
                let g = Double(raw[offset + 1])
                let b = Double(raw[offset + 2])
                luminances.append(0.299 * r + 0.587 * g + 0.114 * b)
            }
            let mean = luminances.reduce(0, +) / Double(luminances.count)
            let variance = luminances.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Double(luminances.count)
            return PixelStatistics(meanLuminance: mean, luminanceStdDev: variance.squareRoot())
        }
    }
}
