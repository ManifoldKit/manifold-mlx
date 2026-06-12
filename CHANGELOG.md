# Changelog

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
