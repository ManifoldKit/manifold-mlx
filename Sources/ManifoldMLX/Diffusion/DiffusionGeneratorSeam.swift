import Foundation
import ManifoldInference

// MARK: - Diffusion generator seam

/// Injectable seam over the denoise/decode/encode pipeline that
/// `MLXDiffusionBackend` and `FluxDiffusionBackend` drive inside their
/// `generate(...)` stream.
///
/// ## Why this exists
///
/// The vendored `TextToImageGenerator` protocols (in `StableDiffusion` /
/// `FluxSwift`) are *not* an injectable seam for the generate loop: their
/// `generateLatents(...)` return a **concrete** `DenoiseIterator` struct whose
/// `next()` runs real Metal compute, and the loop also calls `MLX.eval`,
/// `decode(...)`, and `Image(...)` directly. None of that can run under a plain
/// `swift test` runner — touching the MLX runtime without a loaded model aborts
/// the process with "Failed to load default metallib" (see CLAUDE.md hardware
/// constraints + `MLXResourceArbiterTests`).
///
/// So this seam abstracts the loop at a higher level: a ``DiffusionGenerator``
/// vends a ``DiffusionRun`` that owns *all* `MLXArray` handling internally —
/// per-step evaluation, final-latent decode, and PNG save. The real
/// implementations wrap the vendored generator; a test fake yields a fixed
/// number of canned steps and writes a stub PNG, performing **zero** MLX work.
///
/// Production behaviour is identical when not injected: `loadModel(from:)`
/// installs the real ``DiffusionGenerator`` exactly as before.
@_spi(Testing)
public protocol DiffusionRun: AnyObject {
    /// Total denoise steps this run will produce (the `total` reported in
    /// ``ImageGenerationEvent/progress(step:total:)``).
    var totalSteps: Int { get }

    /// Advance one denoise step, evaluating the produced latent to bound peak
    /// GPU memory. Returns `false` once the iterator is exhausted.
    func step() throws -> Bool

    /// Decode the most-recently-produced latent and write the final image to
    /// disk under `outputDirectory` (or a temporary location when `nil`),
    /// returning the file URL. Throws if the run produced no latents.
    func finishImage(to outputDirectory: URL?) throws -> URL
}

/// Factory for a single ``DiffusionRun``. One per loaded model; `makeRun` is
/// called once per `generate(...)` invocation.
@_spi(Testing)
public protocol DiffusionGenerator: Sendable {
    func makeRun(prompt: String, config: ImageGenerationConfig) -> any DiffusionRun
}
