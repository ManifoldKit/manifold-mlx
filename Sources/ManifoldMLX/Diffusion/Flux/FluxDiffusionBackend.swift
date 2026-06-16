
import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers
import ManifoldInference
import Hub
import MLX
import FluxSwift

// MARK: - Errors

public enum FluxDiffusionError: Error, LocalizedError {
    case notLoaded
    case noLatentsProduced
    case pngEncodingFailed(URL)

    public var errorDescription: String? {
        switch self {
        case .notLoaded:
            return "No FLUX model loaded. Call loadModel(from:) first."
        case .noLatentsProduced:
            return "The denoising loop produced no latents."
        case .pngEncodingFailed(let url):
            return "Failed to write PNG to \(url.path)."
        }
    }
}

// MARK: - FluxDiffusionBackend

/// ``ImageGenerationBackend`` that drives FLUX.1 Schnell via `mzbac/flux.swift`.
///
/// ## Loading
///
/// `loadModel(from:)` tries two paths in order:
/// 1. `FLUX.loadQuantized(from:)` — for flux.swift's own 4-bit quantization format
///    (directory contains `metadata.json` written by `FLUX.saveQuantizedWeights`).
/// 2. `Flux1Schnell(hub:modelDirectory:) + loadWeights(from:)` — for standard FP16
///    safetensors weights following the diffusers directory layout.
///
/// ## Concurrency
///
/// Mirrors `MLXDiffusionBackend`'s NSLock + `@unchecked Sendable` pattern.
/// The denoising loop is a long-running synchronous sequence; wrapping it in a
/// detached `Task` keeps the main actor unblocked while the loop runs. All
/// mutable state is protected by `lock`.
public final class FluxDiffusionBackend: ImageGenerationBackend, @unchecked Sendable {

    private let lock = NSLock()
    private var _generator: (any DiffusionGenerator)?
    private var _isGenerating = false
    private var _stopRequested = false

    public init() {}

    /// Test-only seam: construct a backend with a pre-installed
    /// ``DiffusionGenerator`` (typically a fake), bypassing `loadModel(from:)`
    /// and the Metal-bound weight load. Production code never calls this.
    @_spi(Testing)
    public init(generator: any DiffusionGenerator) {
        self._generator = generator
    }

    @discardableResult
    private func withLock<T>(_ body: () throws -> T) rethrows -> T {
        lock.lock(); defer { lock.unlock() }
        return try body()
    }

    // MARK: - ImageGenerationBackend

    public var isLoaded: Bool {
        withLock { _generator != nil }
    }

    public var isGenerating: Bool {
        withLock { _isGenerating }
    }

    public func loadModel(from url: URL) async throws {
        // Try flux.swift's own quantized format first (requires metadata.json).
        let textToImage: any TextToImageGenerator
        let quantizedMetadata = url.appending(component: "metadata.json")
        if FileManager.default.fileExists(atPath: quantizedMetadata.path) {
            let flux = try await FLUX.loadQuantized(
                from: url.path,
                modelType: "schnell",
                hub: HubApi(useOfflineMode: true)
            )
            guard let ttig = flux as? any TextToImageGenerator else {
                throw FluxDiffusionError.notLoaded
            }
            textToImage = ttig
        } else {
            // Standard FP16 / BF16 diffusers layout.
            let hub = HubApi(useOfflineMode: true)
            let model = try Flux1Schnell(hub: hub, modelDirectory: url)
            try model.loadWeights(from: url, dtype: .float16)
            textToImage = model
        }

        withLock {
            _generator = RealFluxGenerator(generator: textToImage)
            _stopRequested = false
        }
    }

    public func generate(
        prompt: String,
        config: ImageGenerationConfig
    ) throws -> AsyncThrowingStream<ImageGenerationEvent, Error> {
        let hasModel = withLock { _generator != nil }
        guard hasModel else { throw FluxDiffusionError.notLoaded }
        withLock { _isGenerating = true; _stopRequested = false }

        return AsyncThrowingStream { [self] continuation in
            let task = Task.detached(priority: .userInitiated) { [self] in
                defer { self.withLock { self._isGenerating = false } }
                do {
                    guard let generator = self.withLock({ self._generator }) else {
                        throw FluxDiffusionError.notLoaded
                    }

                    let run = generator.makeRun(prompt: prompt, config: config)
                    let totalSteps = run.totalSteps

                    var step = 0
                    var producedAny = false
                    while true {
                        try Task.checkCancellation()
                        if self.withLock({ self._stopRequested }) { throw CancellationError() }

                        guard try run.step() else { break }
                        step += 1
                        producedAny = true
                        continuation.yield(.progress(step: step, total: totalSteps))
                    }

                    guard producedAny else {
                        throw FluxDiffusionError.noLatentsProduced
                    }

                    let url = try run.finishImage(to: config.outputDirectory)
                    continuation.yield(.completed(url))
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            continuation.onTermination = { @Sendable [self] _ in
                task.cancel()
                self.stopGeneration()
            }
        }
    }

    public func stopGeneration() {
        withLock { _stopRequested = true }
    }

    public func unloadModel() {
        let wasLoaded = withLock {
            let had = _generator != nil
            _generator = nil
            _stopRequested = false
            return had
        }
        if wasLoaded {
            MLX.Memory.cacheLimit = 0
        }
    }

    // MARK: - Private helpers

    /// Unpacks FLUX packed latents from [1, h×w, 64] to [1, H/8, W/8, 16].
    ///
    /// The denoising transformer operates on 2×2 patch-packed latents. The VAE
    /// decoder expects spatial [H/8, W/8, 16] latents, so unpacking is required
    /// before decode. This is the inverse of the pack operation in flux.swift's
    /// `ImageToImageGenerator` extension.
    static func unpackLatents(_ latents: MLXArray, height: Int, width: Int) -> MLXArray {
        let h = height / 16
        let w = width / 16
        // [1, h*w, 64] → [1, h, w, 16, 2, 2]
        let reshaped = latents.reshaped(1, h, w, 16, 2, 2)
        // Inverse of transposed(0,1,3,5,2,4) is transposed(0,1,4,2,5,3)
        let transposed = reshaped.transposed(0, 1, 4, 2, 5, 3)
        // [1, h, 2, w, 2, 16] → [1, H/8, W/8, 16]
        return transposed.reshaped(1, h * 2, w * 2, 16)
    }

    /// Converts a decoded MLXArray [1, H, W, 3] float32 in [0, 1] to a PNG on disk.
    ///
    /// Follows the same CGContext pattern as the vendored `StableDiffusion/Image.swift`
    /// (RGBA, `noneSkipLast`, `byteOrder32Big`) so both backends write identical-format PNGs.
    static func savePNG(_ decoded: MLXArray, to directory: URL?) throws -> URL {
        let dir = directory ?? FileManager.default.temporaryDirectory
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let dest = dir.appending(component: "\(UUID().uuidString).png")

        // [1, H, W, 3] → [H, W, 3] uint8, padded to RGBA (CGContext requires 4 bytes/pixel).
        let rgb = clip(decoded.squeezed() * 255, min: MLXArray(Int32(0)), max: MLXArray(Int32(255)))
            .asType(DType.uint8)
        let (H, W, _) = rgb.shape3
        let alpha = full([H, W, 1], values: UInt8(255), type: UInt8.self)
        let rgba = concatenated([rgb, alpha], axis: -1)    // [H, W, 4]
        eval(rgba)

        // Get bytes via asArray for a guaranteed contiguous copy.
        var bytes = rgba.asArray(UInt8.self)
        let C = 4

        return try bytes.withUnsafeMutableBytes { ptr in
            let cs = CGColorSpace(name: CGColorSpace.sRGB)!
            guard let ctx = CGContext(
                data: ptr.baseAddress, width: W, height: H,
                bitsPerComponent: 8, bytesPerRow: W * C, space: cs,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
                    | CGBitmapInfo.byteOrder32Big.rawValue
            ), let cgImage = ctx.makeImage() else {
                throw FluxDiffusionError.pngEncodingFailed(dest)
            }
            guard let dst = CGImageDestinationCreateWithURL(
                dest as CFURL, UTType.png.identifier as CFString, 1, nil
            ) else {
                throw FluxDiffusionError.pngEncodingFailed(dest)
            }
            CGImageDestinationAddImage(dst, cgImage, nil)
            guard CGImageDestinationFinalize(dst) else {
                throw FluxDiffusionError.pngEncodingFailed(dest)
            }
            return dest
        }
    }
}

// MARK: - Real generator seam implementation

/// Production ``DiffusionGenerator`` wrapping a vendored FluxSwift
/// ``TextToImageGenerator``. Owns all `MLXArray` handling — the
/// `FluxDiffusionBackend.generate(...)` loop never touches MLX directly.
private struct RealFluxGenerator: DiffusionGenerator {
    let generator: any TextToImageGenerator

    func makeRun(prompt: String, config: ImageGenerationConfig) -> any DiffusionRun {
        var params = EvaluateParameters(
            width: config.width,
            height: config.height,
            numInferenceSteps: config.steps,
            seed: config.seed,
            prompt: prompt
        )
        if let guidance = config.guidanceScale {
            params.guidance = Float(guidance)
        }
        return RealFluxRun(
            generator: generator,
            denoiser: generator.generateLatents(parameters: params),
            totalSteps: params.numInferenceSteps,
            height: config.height,
            width: config.width
        )
    }
}

private final class RealFluxRun: DiffusionRun {
    let generator: any TextToImageGenerator
    var denoiser: DenoiseIterator
    let totalSteps: Int
    let height: Int
    let width: Int
    private var lastLatent: MLXArray?

    init(
        generator: any TextToImageGenerator,
        denoiser: DenoiseIterator,
        totalSteps: Int,
        height: Int,
        width: Int
    ) {
        self.generator = generator
        self.denoiser = denoiser
        self.totalSteps = totalSteps
        self.height = height
        self.width = width
    }

    func step() throws -> Bool {
        guard let xt = denoiser.next() else { return false }
        eval(xt)
        lastLatent = xt
        return true
    }

    func finishImage(to outputDirectory: URL?) throws -> URL {
        guard let finalLatent = lastLatent else {
            throw FluxDiffusionError.noLatentsProduced
        }
        let unpacked = FluxDiffusionBackend.unpackLatents(finalLatent, height: height, width: width)
        let decoded = generator.decode(xt: unpacked)
        return try FluxDiffusionBackend.savePNG(decoded, to: outputDirectory)
    }
}

// MARK: - Registration

public extension ImageGenerationService {
    /// Registers the FLUX.1 Schnell backend for `.fluxSchnell` format.
    ///
    /// Call once at app startup before loading any image model. The factory
    /// creates a bare `FluxDiffusionBackend` instance; `ImageGenerationService`
    /// calls `loadModel(from:)` on it immediately after construction.
    @MainActor
    func registerFluxDiffusionBackend() {
        registerBackendFactory(for: .fluxSchnell) { _ in
            FluxDiffusionBackend()
        }
    }
}

