# Changelog

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
