# Changelog

## [0.2.11](https://github.com/roryford/manifold-mlx/compare/v0.2.10...v0.2.11) (2026-06-23)


### Features

* **mlx:** add .mistral tool dialect ([TOOL_CALLS] format) ([#95](https://github.com/roryford/manifold-mlx/issues/95)) ([47015ab](https://github.com/roryford/manifold-mlx/commit/47015abdd8fbf4e7c0c3c8d53097c7aa3c4a2005))
* **mlx:** surface tool-call dialect on BackendCapabilities ([#91](https://github.com/roryford/manifold-mlx/issues/91)) ([7688668](https://github.com/roryford/manifold-mlx/commit/7688668e1d1458a60982eeac6680a7ed91f912ec))


### Bug Fixes

* **mlx:** drop empty-args tool calls that look like parse failures ([#93](https://github.com/roryford/manifold-mlx/issues/93)) ([d6d1a76](https://github.com/roryford/manifold-mlx/commit/d6d1a76df8d4c0578dd567f246d59c84484ee478))
* **mlx:** strengthen llama tool-use steering to close list_dir dispatch gap ([#94](https://github.com/roryford/manifold-mlx/issues/94)) ([1a694df](https://github.com/roryford/manifold-mlx/commit/1a694df7e825f4f72619164ab8f6e1b2e0ca4d32))

## [0.2.10](https://github.com/roryford/manifold-mlx/compare/v0.2.9...v0.2.10) (2026-06-22)

### Highlights

**Tracks ManifoldKit 0.60** ([#90](https://github.com/roryford/manifold-mlx/issues/90), [#84](https://github.com/roryford/manifold-mlx/issues/84)) — the core pin moves to `.upToNextMinor(from: "0.60.0")`, jumping past 0.59 to build against the 0.60 release: the measured tool-call conformance spine (a `ToolCallConformance` cache port, tool-call *dialect* on `BackendCapabilities`, transcript attribution + scorer, public JSON-Schema → GBNF surface) and the Mistral system-prompt-fold renderer fix. No source changes required — bump and rebuild.

**MLX tool-call dispatch reliability** ([#71](https://github.com/roryford/manifold-mlx/issues/71), [#80](https://github.com/roryford/manifold-mlx/issues/80), [#88](https://github.com/roryford/manifold-mlx/issues/88), [#79](https://github.com/roryford/manifold-mlx/issues/79)) — a prefer-tools preamble steers Llama models to actually dispatch tool calls; mlx-swift-lm's native tool-call events are now forwarded so inline (Llama-style) calls dispatch instead of being dropped; and the streaming phase is set correctly when a tool call arrives only via the normalizer tail.

**Graceful handling of unsupported model shapes** ([#89](https://github.com/roryford/manifold-mlx/issues/89), [#83](https://github.com/roryford/manifold-mlx/issues/83)) — the `manifold-tools-mlx` text harness now detects vision-language model dirs (by their `preprocessor_config.json` marker) and errors with a clear message instead of a SIGSEGV mid-sweep; plus a Mistral system-prompt rendering fix and Gemma 4 / Qwen 3.5 load guards.

**Other fixes** — throw `noLatentsProduced` on empty diffusion output, matching `FluxDiffusionBackend` ([#78](https://github.com/roryford/manifold-mlx/issues/78)); robust unknown-scenario exit code + sweep-script hygiene ([#77](https://github.com/roryford/manifold-mlx/issues/77)).

## [0.2.9](https://github.com/roryford/manifold-mlx/compare/v0.2.8...v0.2.9) (2026-06-21)


### Features

* **tools:** advertise N decoy tools to measure tool-selection accuracy ([#73](https://github.com/roryford/manifold-mlx/issues/73)) ([868f6b7](https://github.com/roryford/manifold-mlx/commit/868f6b719632f188bc65b45266ed7f31d470bad7))
* **tools:** score decoy tool-selection with 0.58 classification metrics ([#76](https://github.com/roryford/manifold-mlx/issues/76)) ([cbb4bcf](https://github.com/roryford/manifold-mlx/commit/cbb4bcf754f8ee0986139c41e2e8f412baa48045))


### Bug Fixes

* bump ManifoldKit pin to v0.58.0 ([#75](https://github.com/roryford/manifold-mlx/issues/75)) ([f81145c](https://github.com/roryford/manifold-mlx/commit/f81145c5244645434fb8b1c9d59b2bc0970cfcf4))

## [0.2.8](https://github.com/roryford/manifold-mlx/compare/v0.2.7...v0.2.8) (2026-06-21)


### Features

* add manifold-tools-mlx CLI for running tool-calling scenarios against real MLX models ([#54](https://github.com/roryford/manifold-mlx/issues/54)) ([10dfc0e](https://github.com/roryford/manifold-mlx/commit/10dfc0e2631ac468e38e034bbf62ae67778a3954))
* **flux:** load pre-quantized 4-bit FLUX weights ([#63](https://github.com/roryford/manifold-mlx/issues/63)) ([a76b481](https://github.com/roryford/manifold-mlx/commit/a76b4816bb1e65a5878cc28761501cce7c97d26d))
* **flux:** load real mflux MLX-4bit FLUX.1-schnell bundles ([#39](https://github.com/roryford/manifold-mlx/issues/39)) ([#66](https://github.com/roryford/manifold-mlx/issues/66)) ([c96b5e5](https://github.com/roryford/manifold-mlx/commit/c96b5e5b506e2da644386a1e73db7bbe2bc7522d))
* **flux:** self-quantize fp16-&gt;4-bit + peak-memory regression guard ([#68](https://github.com/roryford/manifold-mlx/issues/68)) ([2694c15](https://github.com/roryford/manifold-mlx/commit/2694c153409c4b5e2ac3c6050c684722c57950ff)), closes [#39](https://github.com/roryford/manifold-mlx/issues/39)
* **flux:** wire 4-bit bundle integration test + document complete layout ([#64](https://github.com/roryford/manifold-mlx/issues/64)) ([ed7c99e](https://github.com/roryford/manifold-mlx/commit/ed7c99e98589872ec2909252a519c310309d7230))


### Bug Fixes

* **mlx:** keep system message first in chat template assembly ([#61](https://github.com/roryford/manifold-mlx/issues/61)) ([fe91cc8](https://github.com/roryford/manifold-mlx/commit/fe91cc88e0b3763f92bc5b259f93acbfedd08a87))
* **mlx:** recognise Llama tool-call dialect so llama-3.2 dispatches tools ([#65](https://github.com/roryford/manifold-mlx/issues/65)) ([8150bec](https://github.com/roryford/manifold-mlx/commit/8150bec9ec9d67da95dc263515c7d8b480d113cf))
* **mlx:** recover Llama python_tag tool channel dropped by MLX detokenizer ([#69](https://github.com/roryford/manifold-mlx/issues/69)) ([8804949](https://github.com/roryford/manifold-mlx/commit/88049495a5b4a5447e3eaa64dc7be53cff404aad))
* **mlx:** route multimodal gemma3n checkpoints to the LLM factory ([#62](https://github.com/roryford/manifold-mlx/issues/62)) ([874d20d](https://github.com/roryford/manifold-mlx/commit/874d20dd03031e4876f2543a1cbbba93dd28138f))
* register only each scenario's requiredTools in manifold-tools-mlx (was advertising all 6, overloading small models) ([#60](https://github.com/roryford/manifold-mlx/issues/60)) ([e5fb594](https://github.com/roryford/manifold-mlx/commit/e5fb59442ae7af9faad3373e58670a0a1cee58d2))
* **release:** resync release-please manifest to 0.2.7 ([#72](https://github.com/roryford/manifold-mlx/issues/72)) ([a0a7cc5](https://github.com/roryford/manifold-mlx/commit/a0a7cc5930272dfa11736e8563b547d5dfccdc88))

## [0.2.6](https://github.com/roryford/manifold-mlx/compare/v0.2.5...v0.2.6) (2026-06-20)


### Features

* **mlx:** emit .usage(TokenUsage) at end-of-turn ([#44](https://github.com/roryford/manifold-mlx/issues/44)) ([#49](https://github.com/roryford/manifold-mlx/issues/49)) ([6b45259](https://github.com/roryford/manifold-mlx/commit/6b4525942120d0475202e2b7b3815dc62b0e03ce))


### Bug Fixes

* bump ManifoldKit pin to v0.55.0 ([#45](https://github.com/roryford/manifold-mlx/issues/45)) ([86ace57](https://github.com/roryford/manifold-mlx/commit/86ace574aafb8cd44188f2243f03d1b258d93d4a))
* **ci+tests:** wire slow-tests lane, golden fixture, and integration hang guards ([#50](https://github.com/roryford/manifold-mlx/issues/50)) ([0eb267a](https://github.com/roryford/manifold-mlx/commit/0eb267aba855e71bc15c1030eedf221e4c9036b3))

## [0.2.5](https://github.com/roryford/manifold-mlx/compare/v0.2.4...v0.2.5) (2026-06-18)

### Highlights

**Tracks ManifoldKit 0.54** ([#41](https://github.com/roryford/manifold-mlx/issues/41)) — the core pin moves to `.upToNextMinor(from: "0.54.0")`, building against the 0.54 release: real GGUF Jinja chat-template rendering, a server-side HTTP/SSE transport for the MCP host, and continued pre-1.0 Contract API hardening (backend-neutral `InferenceError.idleTimeout`, the `streamsToolCallArgumentDeltas` capability-alias deprecation, and documented `EmbeddingBackend` guarantees). No source changes required — bump and rebuild.

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
