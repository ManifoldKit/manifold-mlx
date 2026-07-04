
import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers
import ManifoldInference
import Hub
import MLX
import StableDiffusion

// MARK: - Errors

public enum MLXDiffusionError: Error, LocalizedError {
    case unsupportedModelLayout(URL)
    case insufficientMemory([ImageModelLoadPlan.Reason])
    case notLoaded
    case noLatentsProduced
    case pngEncodingFailed(URL)

    public var errorDescription: String? {
        switch self {
        case .unsupportedModelLayout(let url):
            return "Unrecognised diffusion layout at '\(url.lastPathComponent)'. Expected SD 2.1 Base or SDXL Turbo weights."
        case .insufficientMemory(let reasons):
            return "Insufficient memory: \(reasons.map { "\($0)" }.joined(separator: ", "))"
        case .notLoaded:
            return "No model loaded. Call loadModel(from:) first."
        case .noLatentsProduced:
            return "The denoising loop produced no latents."
        case .pngEncodingFailed(let url):
            return "Failed to write PNG to \(url.path)."
        }
    }
}

// MARK: - MLXDiffusionBackend

/// An ``ImageGenerationBackend`` that drives on-device diffusion inference
/// via the `mlx-swift-examples` StableDiffusion library.
///
/// ## Supported layouts (auto-detected from directory contents)
///
/// - `stabilityai/stable-diffusion-2-1-base` — 512×512, single text encoder
/// - `stabilityai/sdxl-turbo` — 1024×1024, dual text encoder
///
/// Other layouts throw ``MLXDiffusionError/unsupportedModelLayout``.
///
/// ## Directory convention
///
/// `loadModel(from:)` loads weights directly from the directory it is given —
/// the `directoryURL` carried by `ImageModelInfo`. That directory holds the
/// diffusers submodules (`unet/`, `vae/`, `scheduler/`, …) at its top level and
/// may be either a flat install (`.../<org>__<name>/`) or a Hub leaf
/// (`.../models/<org>/<name>/`). Files are resolved relative to that directory
/// via `StableDiffusionConfiguration.resolvingFiles(in:)`, so no particular Hub
/// `downloadBase` shape — and no bridging symlink — is required.
///
/// ## Concurrency
///
/// Mirrors ``LlamaBackend``'s NSLock + `@unchecked Sendable` pattern. The
/// denoising loop is a long-running synchronous call (~6–10 s on Apple Silicon);
/// using an `Actor` would hold isolation across that span and block
/// `stopGeneration()` / `unloadModel()` from the main actor.
public final class MLXDiffusionBackend: ImageGenerationBackend, @unchecked Sendable {

    private let lock = NSLock()
    private var _generator: (any DiffusionGenerator)?
    private var _preset: StableDiffusionConfiguration?
    private var _isGenerating = false
    private var _stopRequested = false

    public init() {}

    /// Test-only seam: construct a backend with a pre-installed
    /// ``DiffusionGenerator`` (typically a fake), bypassing `loadModel(from:)`
    /// and the Metal-bound weight load. Production code never calls this — the
    /// public `init()` leaves the backend unloaded and `loadModel(from:)`
    /// installs the real generator.
    @_spi(Testing)
    public init(generator: any DiffusionGenerator, preset: StableDiffusionConfiguration = .presetStableDiffusion21Base) {
        self._generator = generator
        self._preset = preset
    }

    // MARK: - Sync lock helper
    //
    // NSLock.lock() / .unlock() are marked @available(*, noasync) in Swift 6.
    // Wrapping them in a synchronous method avoids the diagnostic at call
    // sites in async functions — the restriction only fires at direct call
    // sites, not transitively through a sync helper.
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
        // Stage the bundled `mlx.metallib` next to the running binary before any
        // MLX GPU work — see `MLXMetallibStaging` and `MLXBackend.loadModel`. A
        // diffusion-only consumer never loads a text model, so without this the
        // first generate aborts with "Failed to load the default metallib" under a
        // plain `swift build` / `swift run` (issue #82). No-op under Xcode builds.
        MLXMetallibStaging.ensureStaged()

        let preset = try Self.detectPreset(at: url)
        let inputs = try Self.loadPlanInputs(at: url, preset: preset)
        let plan = ImageModelLoadPlan.compute(inputs: inputs)
        if case .deny = plan.verdict {
            throw MLXDiffusionError.insufficientMemory(plan.reasons)
        }

        // `url` is the actual model directory handed to us by the storage
        // layer — `ImageGenerationService` passes `ImageModelInfo.directoryURL`.
        // That directory can be either a Hub leaf (`.../models/<org>/<name>`)
        // or a flat install (`.../<org>__<name>/`); both carry the diffusers
        // submodules (`unet/`, `vae/`, `scheduler/`, …) directly inside it.
        //
        // Resolve weights relative to `url` itself rather than reconstructing a
        // Hub `downloadBase` by walking three components up and assuming a
        // `models/<org>/<name>` shape — that assumption breaks for flat installs
        // (walking up lands at `~/Library/Application Support`, where Hub then
        // looks for a non-existent `models/<org>/<name>` tree). An offline
        // `HubApi` is still supplied so nothing attempts a network fetch; its
        // `localRepoLocation` is overridden by `resolvingFiles(in:)`.
        let hub = HubApi(useOfflineMode: true)
        guard let textToImage = try preset.resolvingFiles(in: url).textToImageGenerator(
            hub: hub, configuration: .init(quantize: true)
        ) else {
            throw MLXDiffusionError.unsupportedModelLayout(url)
        }

        withLock {
            _generator = RealStableDiffusionGenerator(generator: textToImage, preset: preset)
            _preset = preset
            _stopRequested = false
        }
    }

    public func generate(
        prompt: String,
        config: ImageGenerationConfig
    ) throws -> AsyncThrowingStream<ImageGenerationEvent, Error> {
        // Capture generator + preset under lock before creating the stream.
        // Only `self` is captured into the @Sendable Task closure (the class
        // is @unchecked Sendable); generator/preset are re-read inside the task.
        let hasModel = withLock { _generator != nil && _preset != nil }
        guard hasModel else { throw MLXDiffusionError.notLoaded }
        withLock { _isGenerating = true; _stopRequested = false }

        return AsyncThrowingStream { [self] continuation in
            // Strong capture: generation owns a logical unit of work; weak
            // capture would silently drop events on dealloc.
            let task = Task.detached(priority: .userInitiated) { [self] in
                // Backstop only: `_isGenerating` is cleared explicitly right
                // before every `continuation.finish()` call below so a
                // consumer observing stream completion never races a
                // `isGenerating` read against this defer still being pending
                // (#132 — `test_stopGeneration_midStream_finishesEarly_andClearsIsGenerating`
                // flaked ~1-in-6 because the defer ran strictly after
                // `continuation.finish()` made completion visible to the
                // consumer).
                defer { self.withLock { self._isGenerating = false } }
                do {
                    // Re-read under lock — unloadModel() could have run, but
                    // _isGenerating=true prevents concurrent unload on the
                    // same thread. Guard anyway for safety.
                    guard let generator = self.withLock({ self._generator }) else {
                        throw MLXDiffusionError.notLoaded
                    }
                    let run = generator.makeRun(prompt: prompt, config: config)
                    let totalSteps = run.totalSteps

                    var step = 0
                    var producedAny = false
                    while true {
                        try Task.checkCancellation()
                        let stopped = self.withLock { self._stopRequested }
                        if stopped { throw CancellationError() }

                        guard try run.step() else { break }
                        step += 1
                        producedAny = true
                        continuation.yield(.progress(step: step, total: totalSteps))

                        // Opt-in live preview: only when previewStride is set.
                        // nil stride takes none of this path — no decode, no
                        // emit — preserving the byte-for-byte no-preview
                        // behaviour. Each emit costs one extra VAE decode (see
                        // RealStableDiffusionRun.previewImage()).
                        if DiffusionPreviewThrottle.shouldEmit(step: step, total: totalSteps, stride: config.previewStride) {
                            let data = try run.previewImage()
                            continuation.yield(.preview(step: step, total: totalSteps, image: data))
                        }
                    }

                    guard producedAny else {
                        self.withLock { self._isGenerating = false }
                        continuation.finish(throwing: MLXDiffusionError.noLatentsProduced)
                        return
                    }

                    let imageURL = try run.finishImage(to: config.outputDirectory)
                    continuation.yield(.completed(imageURL))
                    self.withLock { self._isGenerating = false }
                    continuation.finish()
                } catch is CancellationError {
                    self.withLock { self._isGenerating = false }
                    continuation.finish()
                } catch {
                    self.withLock { self._isGenerating = false }
                    continuation.finish(throwing: error)
                }
            }

            // Both cancellation paths:
            // 1. task.cancel() — cooperative; checked via Task.checkCancellation().
            // 2. stopGeneration() via _stopRequested — checked per-iteration.
            //
            // Note: onTermination(.cancelled) from a downstream consumer
            // cancellation is unreliable in Swift's AsyncThrowingStream (stdlib
            // limitation — see PR 2 review note). External stopGeneration() is
            // the load-bearing path for runtime-driven cancellation.
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
            _preset = nil
            _stopRequested = false
            return had
        }
        // Only touch the GPU cache if Metal was actually used — calling
        // MLX.GPU.set(cacheLimit:) before any MLX work triggers Metal
        // initialisation and crashes in test environments.
        if wasLoaded {
            MLX.Memory.cacheLimit = 0
        }
    }

    // MARK: - Private helpers

    @_spi(Testing) public static func detectPreset(at url: URL) throws -> StableDiffusionConfiguration {
        let fm = FileManager.default
        // SDXL has a second text encoder; SD 2.1 does not.
        if fm.fileExists(atPath: url.appending(component: "text_encoder_2").path) {
            return .presetSDXLTurbo
        }
        guard fm.fileExists(atPath: url.appending(component: "unet").path) else {
            throw MLXDiffusionError.unsupportedModelLayout(url)
        }
        return .presetStableDiffusion21Base
    }

    @_spi(Testing) public static func loadPlanInputs(
        at url: URL,
        preset: StableDiffusionConfiguration
    ) throws -> ImageModelLoadPlan.Inputs {
        func bytes(at relativePath: String) -> Int64 {
            let path = url.appending(component: relativePath).path
            return (try? FileManager.default.attributesOfItem(atPath: path))?[.size] as? Int64 ?? 0
        }
        let unet   = bytes(at: "unet/diffusion_pytorch_model.safetensors")
        let vae    = bytes(at: "vae/diffusion_pytorch_model.safetensors")
        let te1    = bytes(at: "text_encoder/model.safetensors")
        let te2    = bytes(at: "text_encoder_2/model.safetensors")
        // Activation working set: latent area × float16 bytes × safety factor.
        let isXL   = preset.id.contains("sdxl")
        let latent: Int64 = isXL ? 128 * 128 : 64 * 64
        let activation = latent * 2 * 8  // float16, ×8 safety
        let available  = max(0, Int64(ProcessInfo.processInfo.physicalMemory) - 1_073_741_824)
        return ImageModelLoadPlan.Inputs(
            unetWeightBytes:        unet,
            vaeWeightBytes:         vae,
            textEncoderWeightBytes: te1 + te2,
            activationMemoryBytes:  activation,
            availableMemoryBytes:   available,
            targetWidth:  isXL ? 1024 : 512,
            targetHeight: isXL ? 1024 : 512
        )
    }

    @_spi(Testing) public static func makeParams(
        prompt: String,
        config: ImageGenerationConfig,
        preset: StableDiffusionConfiguration
    ) -> EvaluateParameters {
        let defaults = preset.defaultParameters()
        return EvaluateParameters(
            cfgWeight: config.guidanceScale ?? defaults.cfgWeight,
            steps: config.steps,
            latentSize: [config.height / 8, config.width / 8],
            seed: config.seed ?? defaults.seed,
            prompt: prompt
        )
    }

    static func savePNG(_ image: Image, to directory: URL?) throws -> URL {
        let dir = directory ?? FileManager.default.temporaryDirectory
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let dest = dir.appending(component: "\(UUID().uuidString).png")
        try image.save(url: dest)
        return dest
    }

    /// Encodes an ``Image`` to PNG bytes **in memory** (no disk write) — the
    /// representation carried by ``ImageGenerationEvent/preview(step:total:image:)``.
    static func encodePNGData(_ image: Image) throws -> Data {
        let cgImage = image.asCGImage()
        let mutable = NSMutableData()
        guard let dst = CGImageDestinationCreateWithData(
            mutable as CFMutableData, UTType.png.identifier as CFString, 1, nil
        ) else {
            throw MLXDiffusionError.pngEncodingFailed(URL(fileURLWithPath: "(memory)"))
        }
        CGImageDestinationAddImage(dst, cgImage, nil)
        guard CGImageDestinationFinalize(dst) else {
            throw MLXDiffusionError.pngEncodingFailed(URL(fileURLWithPath: "(memory)"))
        }
        return mutable as Data
    }
}

// MARK: - Real generator seam implementation

/// Production ``DiffusionGenerator`` wrapping a vendored StableDiffusion
/// ``TextToImageGenerator``. Owns all `MLXArray` handling — the
/// `MLXDiffusionBackend.generate(...)` loop never touches MLX directly.
// `@unchecked Sendable`: the vendored StableDiffusion `TextToImageGenerator` is
// not `Sendable`, but this wrapper is only ever read inside the backend's
// serialised generate task (the same constraint that lets `MLXDiffusionBackend`
// itself be `@unchecked Sendable` and capture the generator into its detached
// task). Mirrors `LlamaBackend`'s pattern.
private struct RealStableDiffusionGenerator: DiffusionGenerator, @unchecked Sendable {
    let generator: any TextToImageGenerator
    let preset: StableDiffusionConfiguration

    func makeRun(prompt: String, config: ImageGenerationConfig) -> any DiffusionRun {
        let params = MLXDiffusionBackend.makeParams(prompt: prompt, config: config, preset: preset)
        return RealStableDiffusionRun(generator: generator, iterator: generator.generateLatents(parameters: params))
    }
}

private final class RealStableDiffusionRun: DiffusionRun {
    let generator: any TextToImageGenerator
    var iterator: DenoiseIterator
    private var lastLatent: MLXArray?

    let totalSteps: Int

    init(generator: any TextToImageGenerator, iterator: DenoiseIterator) {
        self.generator = generator
        self.iterator = iterator
        self.totalSteps = iterator.underestimatedCount
    }

    func step() throws -> Bool {
        guard let latent = iterator.next() else { return false }
        // Force evaluation here to bound peak memory: without this, MLX
        // accumulates the full N-step compute graph before running anything.
        MLX.eval(latent)
        lastLatent = latent
        return true
    }

    func previewImage() throws -> Data {
        guard let latent = lastLatent else {
            throw MLXDiffusionError.notLoaded
        }
        // Extra VAE decode of the *current* intermediate latent — pure GPU cost
        // on top of the denoise the loop is already doing (see the protocol's
        // `previewImage()` doc). No GPU-cache flush here: unlike the terminal
        // decode, a preview tick is interleaved with ongoing denoising, so we
        // must not zero the activation cache the next `step()` will reuse.
        let decode = generator.detachedDecoder()
        let decoded = decode(latent)
        let frame = decoded.ndim == 4 ? decoded[0] : decoded
        return try MLXDiffusionBackend.encodePNGData(Image((frame * 255).asType(.uint8)))
    }

    func finishImage(to outputDirectory: URL?) throws -> URL {
        guard let finalLatent = lastLatent else {
            throw MLXDiffusionError.notLoaded
        }
        // Get a VAE-only decoder closure, then flush the GPU activation cache
        // built up during denoising before the decode step starts. This
        // prevents UNet activations and VAE activations from competing for GPU
        // memory simultaneously.
        let decode = generator.detachedDecoder()
        MLX.Memory.cacheLimit = 0

        let decoded = decode(finalLatent)
        // decode() returns float32 in [0, 1]; Image expects uint8 [0, 255].
        let frame = decoded.ndim == 4 ? decoded[0] : decoded
        return try MLXDiffusionBackend.savePNG(Image((frame * 255).asType(.uint8)), to: outputDirectory)
    }
}

// MARK: - Factory registration

public extension ImageGenerationService {
    /// Registers the built-in MLX diffusion backend for `.mlxDiffusion` format.
    ///
    /// Hosts that want a custom factory skip this and register their own via
    /// ``ImageGenerationService/registerBackendFactory(for:factory:)``.
    @MainActor
    func registerMLXDiffusionBackend() {
        registerBackendFactory(for: .mlxDiffusion) { info in
            let backend = MLXDiffusionBackend()
            try await backend.loadModel(from: info.directoryURL)
            return backend
        }
    }
}

