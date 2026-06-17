# Changelog

## [0.2.4](https://github.com/roryford/manifold-mlx/compare/v0.2.3...v0.2.4) (2026-06-17)


### Features

* **mlx:** emit live .preview denoising events from diffusion backends ([#8](https://github.com/roryford/manifold-mlx/issues/8)) ([#37](https://github.com/roryford/manifold-mlx/issues/37)) ([5e62b91](https://github.com/roryford/manifold-mlx/commit/5e62b91f34219b46cb0f7e2dc7b347d22485783c))
* **mlx:** injectable TextToImageGenerator seam for diffusion generate/stop coverage ([#29](https://github.com/roryford/manifold-mlx/issues/29)) ([#34](https://github.com/roryford/manifold-mlx/issues/34)) ([7616090](https://github.com/roryford/manifold-mlx/commit/76160902e93f5d5694f003829e07c2a8bd4fb1bc))


### Bug Fixes

* bump ManifoldKit pin to v0.53.0 ([#38](https://github.com/roryford/manifold-mlx/issues/38)) ([62fb12c](https://github.com/roryford/manifold-mlx/commit/62fb12cf0aecc4ec3c5cb268f5178079084bcca4))
* **mlx:** detect top-level vision_config in requiresVLMFactory ([#22](https://github.com/roryford/manifold-mlx/issues/22)) ([#24](https://github.com/roryford/manifold-mlx/issues/24)) ([9c4865f](https://github.com/roryford/manifold-mlx/commit/9c4865f70cfcfd46bfb24e427c2c0ea2bea16127))
* **mlx:** skip Gemma4-MoE smoke test before model load to avoid integration hang ([#26](https://github.com/roryford/manifold-mlx/issues/26)) ([#35](https://github.com/roryford/manifold-mlx/issues/35)) ([e989772](https://github.com/roryford/manifold-mlx/commit/e989772c0b3229aed350f64c31ae3c1d9e10ffda))

## [0.2.3](https://github.com/roryford/manifold-mlx/compare/v0.2.2...v0.2.3) (2026-06-15)

### Highlights

**Tracks ManifoldKit 0.52** ([#20](https://github.com/roryford/manifold-mlx/issues/20)) — the core pin moves to `.upToNextMinor(from: "0.52.0")`, building against the 0.52 release (opt-in rendered-prompt observability via `GenerationConfig.captureRenderedPrompt`, batteries-included context-compression policies, idle model auto-unload, and headless model selection).

**Parity test future-proofed against core enum growth** ([#18](https://github.com/roryford/manifold-mlx/issues/18)) — 0.52 adds the non-frozen `GenerationEvent.promptRendered` case, which broke the event-order parity test's exhaustive switch. The switch now carries an `@unknown default`, so it compiles across pin bumps and survives all future `GenerationEvent` additions instead of failing to build on each new case.

## [0.2.2](https://github.com/roryford/manifold-mlx/compare/v0.2.1...v0.2.2) (2026-06-14)

### Highlights

**MLX rejects unsupported grammars instead of dropping them silently** ([#13](https://github.com/roryford/manifold-mlx/issues/13)) — `MLXBackend` has no grammar-constrained sampling path, but it previously ignored a non-nil `GenerationConfig.grammar` and returned unconstrained free text with no error. It now throws `InferenceError.unsupportedGrammar`, matching the `InferenceBackend` contract that the cloud backends already enforce — so a caller asking for grammar-constrained output on MLX gets a clear failure instead of silent free-form text ([#14](https://github.com/roryford/manifold-mlx/issues/14)).

**Tracks ManifoldKit 0.51** ([#16](https://github.com/roryford/manifold-mlx/issues/16)) — the core pin moves to `.upToNextMinor(from: "0.51.0")`, building against the 0.51 release (grammar-constrained tool calling, model-capability flags, and the pre-1.0 Contract wire-type freeze). The new additive `GenerationEvent.generationCompleted` case is handled. Bump and rebuild — no other source changes required.

## [0.2.1](https://github.com/roryford/manifold-mlx/compare/v0.2.0...v0.2.1) (2026-06-13)

### Highlights

**Upgrade from 0.2.0 to pick up the ManifoldKit 0.50 core.** `v0.2.0` was pinned to ManifoldKit 0.49.0; this is the recommended successor for anyone on `from: "0.2.0"`. The core pin moves to `.upToNextMinor(from: "0.50.0")` ([#9](https://github.com/roryford/manifold-mlx/issues/9)), building against ManifoldKit 0.50 — which adds an additive `ImageGenerationEvent.preview` case for live denoising previews; this release handles the new case (the actual MLX preview-frame emit is a planned follow-up). No source changes are required — bump and rebuild.

**Versioning reconciled** ([#10](https://github.com/roryford/manifold-mlx/issues/10)) — a manual `v0.2.0` tag had been pushed out-of-band, so release-please (which versions from `.release-please-manifest.json`, not git tags) cut a `v0.1.1` *below* it, leaving the published high tag pointing at older code. The manifest is reset to `0.2.0` so the version line resumes correctly at `0.2.1` and can never regress below `v0.2.0` again.

## [0.1.1](https://github.com/roryford/manifold-mlx/compare/v0.1.0...v0.1.1) (2026-06-13)

### Highlights

**Tracks ManifoldKit 0.50** ([#9](https://github.com/roryford/manifold-mlx/issues/9)) — the core pin moves to `.upToNextMinor(from: "0.50.0")`, building against the 0.50 release. ManifoldKit 0.50 adds an additive `ImageGenerationEvent.preview` case for live denoising previews; this release handles the new case (the actual MLX preview-frame emit is a planned follow-up).

**Test & docs hardening** — added unit coverage for `FluxDiffusionBackend` and `TransformersTokenizerLoader` (previously untested), corrected stale `MLX`-trait claims in the DocC landing page, and made the Gemma4 MoE smoke tests use standard model discovery + a VLM-factory skip guard instead of a hardcoded model path.

## 0.1.0 (2026-06-12)

The MLX inference and diffusion backends now ship as their own package. They were split out of ManifoldKit core in the v0.48 packaging release ([ManifoldKit#1749](https://github.com/roryford/ManifoldKit/issues/1749)) so that `swift build` of core never drags mlx-swift — the heavy backends are one `.package` line and one registrar call away.

### Highlights

**`ManifoldMLX` is now a companion package** ([#2](https://github.com/roryford/manifold-mlx/issues/2)) — Apple-Silicon-native text generation (mlx-swift-lm families incl. MoE Gemma 4 via MLXVLM routing), prompt/KV cache coordination, the resource arbiter, capability probing, and FLUX.1 / Stable Diffusion image generation move out of core and plug back in through a single `MLXBackends` registrar. The vendored `FluxSwift` and `StableDiffusion` targets ship alongside it. Module names are restored to their canonical form ahead of this first tag — no temporary `Kit`-suffixed targets.

```swift
// Package.swift
.package(url: "https://github.com/roryford/ManifoldKit", from: "0.48.0"),
.package(url: "https://github.com/roryford/manifold-mlx", from: "0.1.0"),

// App entry point
import ManifoldKit
import ManifoldMLX

let kit = try await ManifoldKit.quickStart(backends: [MLXBackends.self])
```

**Pinned to ManifoldKit 0.48.x** — this release tracks core via `.upToNextMinor(from: "0.48.0")` and builds against the post-split core, where the backend seam and registrar surface are frozen and verified by the out-of-package split proof.
