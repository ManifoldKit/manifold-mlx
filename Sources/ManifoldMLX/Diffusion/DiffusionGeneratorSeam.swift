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

    /// Decode the most-recently-produced *intermediate* latent and return the
    /// encoded image bytes in memory — the same representation carried by
    /// ``ImageGenerationEvent/preview(step:total:image:)``. **No disk write.**
    ///
    /// The generate loop calls this only when emitting a throttled preview tick
    /// (i.e. when `ImageGenerationConfig.previewStride` is non-nil and the
    /// current step lands on the stride). It is never called on the `nil`-stride
    /// path, so the no-preview behaviour stays byte-for-byte identical.
    ///
    /// ## GPU cost
    ///
    /// Each call runs a **full extra VAE decode** of the current latent (and, on
    /// FLUX, an unpack) on top of the denoise compute the loop is already doing.
    /// That is the dominant per-tick cost — roughly one decode's worth of GPU
    /// work and a transient activation allocation per emitted preview. Throttle
    /// (large `previewStride`) accordingly on memory-constrained devices; the
    /// `nil`-stride default pays none of it.
    func previewImage() throws -> Data
}

/// Factory for a single ``DiffusionRun``. One per loaded model; `makeRun` is
/// called once per `generate(...)` invocation.
@_spi(Testing)
public protocol DiffusionGenerator: Sendable {
    func makeRun(prompt: String, config: ImageGenerationConfig) -> any DiffusionRun
}

// MARK: - Preview throttle

/// Decides which denoise steps emit a live ``ImageGenerationEvent/preview``.
///
/// Both backends share this so short Turbo/Schnell (1–4 step) runs and long
/// (20–50 step) runs throttle identically. The rule is deliberately simple and
/// caller-driven: emit on every `stride`-th step (1-based). The terminal step is
/// never previewed — the final image arrives as ``ImageGenerationEvent/completed``,
/// so a preview that coincides with the last step would be a redundant extra VAE
/// decode of the same frame the loop is about to decode anyway.
///
/// `stride == nil` (the default) emits nothing — the no-preview path. A
/// non-positive stride is treated as disabled rather than trapping (recoverable
/// input from a host-supplied config).
@_spi(Testing)
public enum DiffusionPreviewThrottle {
    public static func shouldEmit(step: Int, total: Int, stride: Int?) -> Bool {
        guard let stride, stride > 0 else { return false }
        guard step < total else { return false }   // final frame = .completed
        return step % stride == 0
    }
}
