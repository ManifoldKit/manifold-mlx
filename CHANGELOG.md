# Changelog

## [0.3.4](https://github.com/ManifoldKit/manifold-mlx/compare/v0.3.3...v0.3.4) (2026-07-17)


### Features

* **mlx:** enable prompt KV-cache reuse by default ([#153](https://github.com/ManifoldKit/manifold-mlx/issues/153)) ([91a3412](https://github.com/ManifoldKit/manifold-mlx/commit/91a3412617a306bb930475f1b7dfd6b9a4175b3d))


### Bug Fixes

* bump ManifoldKit pin to v0.72.0 ([#156](https://github.com/ManifoldKit/manifold-mlx/issues/156)) ([ec596b9](https://github.com/ManifoldKit/manifold-mlx/commit/ec596b91baf9f5af92243733862e5149ec256354))

## [0.3.3](https://github.com/ManifoldKit/manifold-mlx/compare/v0.3.2...v0.3.3) (2026-07-13)


### Highlights

#### ManifoldKit 0.71.0

Re-pins the core dependency to [ManifoldKit 0.71.0](https://github.com/ManifoldKit/ManifoldKit/releases/tag/v0.71.0) — the Phase A API-surface tightening (29 core internals demoted to `package`) and the experimental-tier declaration. No MLX-side source changes were needed; the full build+test gate passed against the new core unchanged. See [#151](https://github.com/ManifoldKit/manifold-mlx/issues/151).

## [0.3.2](https://github.com/ManifoldKit/manifold-mlx/compare/v0.3.1...v0.3.2) (2026-07-11)


### Bug Fixes

* bump ManifoldKit pin to v0.70.0 ([#147](https://github.com/ManifoldKit/manifold-mlx/issues/147)) ([dd4031d](https://github.com/ManifoldKit/manifold-mlx/commit/dd4031d3410a4a1e6f09b10c6b7e481739a0549a))

## [0.3.1](https://github.com/ManifoldKit/manifold-mlx/compare/v0.3.0...v0.3.1) (2026-07-11)


### Bug Fixes

* bump ManifoldKit pin to v0.68.0 ([#142](https://github.com/ManifoldKit/manifold-mlx/issues/142)) ([5602c7c](https://github.com/ManifoldKit/manifold-mlx/commit/5602c7ce05785259f01c03008818b016fa92b074))
* bump ManifoldKit pin to v0.69.0 (wave-2 ClaimRegistry adapt) ([#144](https://github.com/ManifoldKit/manifold-mlx/issues/144)) ([2196529](https://github.com/ManifoldKit/manifold-mlx/commit/219652903b6404531fac4cb89566cfc937426334))

## [0.3.0](https://github.com/ManifoldKit/manifold-mlx/compare/v0.2.19...v0.3.0) (2026-07-09)


### ⚠ BREAKING CHANGES

* `MLXBackend.generate` and `MLXGenerationDriver.generate` require a `hints: GenerationRuntimeHints` parameter. This PR references ManifoldKit's unreleased main branch (core #2152) — it will not build against the released 0.66.0 pin and is staged to merge combined with the core-bump pin bump once the core release ships.
* MLXBackend.generate(prompt:systemPrompt:config:) now requires a hints: GenerationRuntimeHints parameter to match ManifoldKit 0.67.0's InferenceBackend protocol.

### Features

* adopt GenerationRuntimeHints (config→hints split, ManifoldKit [#2152](https://github.com/ManifoldKit/manifold-mlx/issues/2152)) ([#139](https://github.com/ManifoldKit/manifold-mlx/issues/139)) ([50b6fc3](https://github.com/ManifoldKit/manifold-mlx/commit/50b6fc3cca878a1c7219d0af83caed74afa83b80))


### Bug Fixes

* adopt GenerationRuntimeHints for ManifoldKit 0.67.0 ([#140](https://github.com/ManifoldKit/manifold-mlx/issues/140)) ([bc0a92d](https://github.com/ManifoldKit/manifold-mlx/commit/bc0a92d3e4566491798f17bc9b20e9a49bcc7da1))

## [0.2.19](https://github.com/ManifoldKit/manifold-mlx/compare/v0.2.18...v0.2.19) (2026-07-07)


### Bug Fixes

* bump ManifoldKit pin to v0.66.0 ([#138](https://github.com/ManifoldKit/manifold-mlx/issues/138)) ([e92b283](https://github.com/ManifoldKit/manifold-mlx/commit/e92b283b0e944349b1da794e666834b4183e6bca))
* lift Qwen 3.5 generation-crash guard (mlx-swift-lm 3.31.4 fixes [#157](https://github.com/ManifoldKit/manifold-mlx/issues/157)) ([#135](https://github.com/ManifoldKit/manifold-mlx/issues/135)) ([5ad93e5](https://github.com/ManifoldKit/manifold-mlx/commit/5ad93e50f99e04f34b5a8443272df755c201e232))

## [0.2.18](https://github.com/ManifoldKit/manifold-mlx/compare/v0.2.17...v0.2.18) (2026-07-04)

### Highlights

**mlx-swift-lm bumped to 3.31.4** ([#134](https://github.com/ManifoldKit/manifold-mlx/issues/134)) — fixes the gated-DeltaNet broadcast crash that blocked Qwen3.5 from loading (ManifoldKit#2061, upstream ml-explore/mlx-swift-lm#157): `Qwen3.5-9B-4bit` now loads and generates, scoring 22/25 (88%) on the BFCL AST suite. Gemma 4 and Gemma 3n still fail to load, but the failure mode changed from a process-killing mid-generation crash to a clean, catchable missing-weight-key error — both remain open as weight-layout gaps. Co-bumps swift-syntax to 602.0.0 and AnyLanguageModel to 0.9.0.

### Bug Fixes

* **ci:** repair nightly slow lanes — model-loading contract participants, diffusion isGenerating race, de-schedule redundant lane ([#132](https://github.com/ManifoldKit/manifold-mlx/issues/132)) ([d0a59b1](https://github.com/ManifoldKit/manifold-mlx/commit/d0a59b12a89060946ea8b29382b21444af878fa7))

## [0.2.17](https://github.com/ManifoldKit/manifold-mlx/compare/v0.2.16...v0.2.17) (2026-07-03)

### Highlights

**Tracks ManifoldKit 0.65** ([#129](https://github.com/ManifoldKit/manifold-mlx/issues/129)) — the core pin moves to `.upToNextMinor(from: "0.65.0")`, the release that honors advertised structured-output and cache-usage capabilities on cloud backends and removes dead public surface flagged by the inert-code audit. Re-resolved, built, and tested green against the new core.

### Bug Fixes

* Bump ManifoldKit pin to v0.65.0 ([#129](https://github.com/ManifoldKit/manifold-mlx/issues/129))

## [0.2.16](https://github.com/ManifoldKit/manifold-mlx/compare/v0.2.15...v0.2.16) (2026-07-02)


### Features

* **tools:** add BFCL argument-level eval subcommand for MLXBackend ([#122](https://github.com/ManifoldKit/manifold-mlx/issues/122)) ([6c185b1](https://github.com/ManifoldKit/manifold-mlx/commit/6c185b13e2a3af297a4edb75bb3cff4639b0222d))


### Bug Fixes

* bump ManifoldKit pin to v0.64.0 ([#126](https://github.com/ManifoldKit/manifold-mlx/issues/126)) ([3d7b50d](https://github.com/ManifoldKit/manifold-mlx/commit/3d7b50d20f337d274e95945d6167ac15f6368cdd))
* stage mlx.metallib in diffusion backends for swift-run image gen ([#123](https://github.com/ManifoldKit/manifold-mlx/issues/123)) ([e9d4491](https://github.com/ManifoldKit/manifold-mlx/commit/e9d44910aa968e11e117246268940179603035fb))

## [0.2.15](https://github.com/ManifoldKit/manifold-mlx/compare/v0.2.14...v0.2.15) (2026-06-28)

### Highlights

**A full GBNF grammar engine for on-device MLX generation.** The MLX backend gains a complete GBNF engine — any-byte `.` matching, Unicode escapes, and precise diagnostics — bringing grammar-constrained decoding toward parity with the llama.cpp path ([#97](https://github.com/ManifoldKit/manifold-mlx/issues/97), [#118](https://github.com/ManifoldKit/manifold-mlx/issues/118)). It is paired with an on-device grammar mask and a prose-turn circuit breaker that keep constrained decoding fast and stop a model from running away on free-form turns ([#114](https://github.com/ManifoldKit/manifold-mlx/issues/114), [#117](https://github.com/ManifoldKit/manifold-mlx/issues/117)).

**Tracks ManifoldKit 0.63** ([#121](https://github.com/ManifoldKit/manifold-mlx/issues/121)) — the core pin moves to `.upToNextMinor(from: "0.63.0")`, the release that ships the on-device `Score`/`EvalScorer` eval surface, the `ManifoldTelemetryOTLP` OTLP/HTTP span exporter, and AGENTS.md ambient-instruction skills support. Re-resolved, built, and tested green against the new core.

### Features

* **mlx:** full GBNF engine — any-byte dot matching, Unicode escapes, and precise diagnostics ([#97](https://github.com/ManifoldKit/manifold-mlx/issues/97), [#118](https://github.com/ManifoldKit/manifold-mlx/issues/118))

### Bug Fixes

* **mlx:** terminate tool-calling turns by preserving tool-result history, so a tool call no longer leaves the turn hanging ([#115](https://github.com/ManifoldKit/manifold-mlx/issues/115))
* **deps:** bump ManifoldKit pin to v0.63.0 ([#121](https://github.com/ManifoldKit/manifold-mlx/issues/121))

### Performance Improvements

* **mlx:** on-device grammar mask + prose-turn circuit breaker for constrained decoding ([#114](https://github.com/ManifoldKit/manifold-mlx/issues/114), [#117](https://github.com/ManifoldKit/manifold-mlx/issues/117))

## [0.2.14](https://github.com/roryford/manifold-mlx/compare/v0.2.13...v0.2.14) (2026-06-27)

### Highlights

**Grammar-constrained Mistral tool calling — F3 closed** ([#109](https://github.com/roryford/manifold-mlx/issues/109)) — this release closes the gap flagged in the 0.2.13 correction note. The 0.2.13 normalizer repaired the unit-fixture drop pattern but not live mangling (unquoted keys, unbalanced braces), leaving tool-selection F1 at 0.0 across all reference scenarios. The fix is decode-time grammar masking: logits are now constrained to the `[TOOL_CALLS]` envelope grammar during generation, so structural tokens are guaranteed by the constraint rather than repaired after the fact. The normalizer remains in place for residual detokenizer artifacts, but the dominant failure mode — free-form generation diverging from the envelope — is eliminated. The grammar path is gated on the Mistral dialect; Llama's python-tag path is unchanged.

**Grammar-constrained sampling performance** ([#110](https://github.com/roryford/manifold-mlx/issues/110)) — the per-token O(vocab × accept) scan that made the GBNF path unusably slow (pinned CPU, GPU idle, minutes per turn at 32k–150k vocab) is replaced by three layered fixes. A byte trie over the vocabulary (`GBNFTokenTrie`) is built once at load; each step walks the grammar over the trie so shared token prefixes are tested once rather than once per token. A state→mask cache memoizes the allowed-token set by matcher state, skipping the trie walk entirely on repeated states (which recur constantly within a JSON string value). The matcher core switches to flattened integer stack positions (`GBNFFastMatcher` / `GBNFCompiled`), eliminating per-step allocations. A randomized fuzzer parity suite asserts byte-identical accept/reject semantics against the reference `GBNFMatcher`.

**Tracks ManifoldKit 0.62** ([#111](https://github.com/roryford/manifold-mlx/issues/111), [#113](https://github.com/roryford/manifold-mlx/issues/113)) — the core pin moves to `.upToNextMinor(from: "0.62.0")`. The `manifold-tools-mlx` CLI drops its nine vendored scenario JSON files and the `loadScenarios()` workaround now that MK 0.62 fixes `ScenarioLoader.loadBuiltIn()` to resolve via `Bundle.module`. `ConformanceScorer` and `MatrixRenderer` are wired in, replacing the hand-rolled confusion-counts summary — every run now writes a deterministic `MATRIX.md` alongside the JSONL, and a new `--emit-records` flag writes `ConformanceRecord[]` JSON for cross-leg collation with Ollama and cloud legs. An `MLXRenderConsistencyGateTests` suite folds the committed Qwen/Mistral/Hermes/Gemma template corpus over `RenderConsistencyChecker` and asserts no silent dialect drops at CI time, without requiring Apple Silicon or a live model.

## [0.2.13](https://github.com/roryford/manifold-mlx/compare/v0.2.12...v0.2.13) (2026-06-25)

### Highlights

**MLX-Mistral tool calling, end to end** ([#102](https://github.com/roryford/manifold-mlx/issues/102), [#104](https://github.com/roryford/manifold-mlx/issues/104)) — two changes close the Mistral tool-call path on MLX (umbrella #2005 F3). Structural tools are now threaded into the MLX chat-template render so MLX-Mistral-v0.3 actually emits `[TOOL_CALLS]` blocks (additive). But the MLX streaming detokenizer strips the JSON quote/space structural tokens, so the emitted call was malformed and extracted zero calls — `MLXMistralToolCallNormalizer` now sits before the output parser and repairs that exact drop pattern (re-quoting bare keys and values, mapping the call-name key to `"name"`, balancing dropped braces), mirroring the existing Llama Python-tag normalizer. It is conservative by design: valid JSON passes through byte-for-byte and genuine junk is left unrepaired so the parser fails cleanly rather than fabricating a call.

> **Correction (2026-06-27).** A post-release re-measure (ManifoldKit `manifold-tools` against `mlx-community/Mistral-7B-Instruct-v0.3-4bit`, d0) found this path is **not** functionally closed. MLX-Mistral-v0.3 *does* now emit `[TOOL_CALLS]` blocks (the structural-threading half landed), but the live output stays malformed past the normalizer (e.g. `[TOOL_CALLS][{function:now,arguments:{}]` — unquoted keys, unbalanced braces), so **zero calls extract and tool-selection F1 = 0.0** across all 9 reference scenarios. The conservative normalizer repairs the unit-fixture drop pattern but not the broader live mangling. The real fix is **decode-time grammar-constrained decoding** (force well-formed tool-call output during generation), tracked as the umbrella-#2005 / mlx#106 follow-up — `[TOOL_CALLS]` emission alone does not close F3.

**Tracks ManifoldKit 0.61** ([#105](https://github.com/roryford/manifold-mlx/issues/105)) — the core pin moves to `.upToNextMinor(from: "0.61.0")`, picking up the SwiftData-backed tool-call conformance cache (persisted `model × quant × backend` verdicts) and the core-side Mistral tool-result and Gemma close-delimiter fixes.

## [0.2.12](https://github.com/roryford/manifold-mlx/compare/v0.2.11...v0.2.12) (2026-06-23)


### Features

* **mlx:** grammar-constrained sampling via a GBNF executor ([#96](https://github.com/roryford/manifold-mlx/issues/96)) ([#98](https://github.com/roryford/manifold-mlx/issues/98)) ([44723a1](https://github.com/roryford/manifold-mlx/commit/44723a1c031dbdd4af86bcafc3e419c02f949b1d))

## [0.2.11](https://github.com/roryford/manifold-mlx/compare/v0.2.10...v0.2.11) (2026-06-23)

### Highlights

**Mistral tool-call dialect** ([#95](https://github.com/roryford/manifold-mlx/issues/95), [#91](https://github.com/roryford/manifold-mlx/issues/91)) — adds `.mistral` to `MLXToolDialect` with full `[TOOL_CALLS] [{…}]` parsing (single and parallel calls), tool-block injection, and history replay. Detection covers `mistral`, `ministral`, and `mixtral` `model_type` values. The tool-call dialect is also surfaced on `BackendCapabilities` so callers can inspect which wire format a loaded model uses.

**MLX tool-call reliability** ([#93](https://github.com/roryford/manifold-mlx/issues/93), [#94](https://github.com/roryford/manifold-mlx/issues/94)) — an empty-args circuit breaker drops calls where the model clearly intended non-empty arguments but parsing produced `{}`; dispatching those caused tool errors and model retry loops. Llama tool-use steering is strengthened from hedging language to a directive "MUST call it" instruction, closing the `list_dir` empty-turn gap on Llama-3.2-3B.


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
