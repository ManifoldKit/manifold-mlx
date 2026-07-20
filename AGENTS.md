# manifold-mlx — MLX inference & diffusion backend for ManifoldKit

MLX inference and diffusion backends for [ManifoldKit](https://github.com/ManifoldKit/ManifoldKit) — Rory's on-device AI SDK — split out of core in the v0.48 packaging release (ManifoldKit#1749) so a plain `swift build` of core never drags in mlx-swift. Ships the `ManifoldMLX` product: Apple-Silicon-native text generation via mlx-swift-lm (including MoE Gemma 4 routed through MLXVLM), prompt/KV cache coordination, a resource arbiter, capability probing, and FLUX.1 / Stable Diffusion image generation via the vendored `FluxSwift` and `StableDiffusion` targets. Consumed by ManifoldKit-based apps (fireside, idlewick, basechat) through the `MLXBackends` registrar. Its independent assurance harness is the separate `manifold-eval` repo — this repo's own tests are unit/contract-level, not the evals.

## Build & test

```sh
swift build   # first cold build resolves the mlx-swift / mlx-swift-lm dependency tree (minutes on a cold SwiftPM cache, ~100s once checkouts are cached); also compiles mlx.metallib via the MLXMetallibPlugin prebuild plugin if the Metal Toolchain component is installed
swift test    # NO --parallel — ManifoldBackendTestKit's BackendContractChecks claims registry is process-global; this is the full-suite merge gate
```

Verified 2026-07-06 in a fresh worktree on this machine (Xcode 26 / Swift 6.3.3, cached SwiftPM checkouts): `swift build` completed clean (`Build complete! (96.81s)`, exit 0). `swift test` was attempted locally and crashed mid-run with an MLX "Failed to load the default metallib" fatal error inside `ManifoldMLXIntegrationTests` (a GPU-device-init issue, despite a compiled `mlx.metallib` on disk) — most likely this dev session lacking real GPU/Metal device access rather than a repo defect, since `ci.yml`'s `swift test` has been green on every `main` push this week (`gh run list`, last 5 runs all `success`, ~4-8 min). Treat `swift test` as derived-from-CI/confirmed-green-on-main rather than independently re-verified end-to-end in this session.

Slow / gated lanes exist but are **not** part of the merge gate — do not run them expecting fast or reliable results:

```sh
swift test --filter ManifoldMLXTests             # nightly full re-run w/ RUN_SLOW_TESTS=1 (slow-tests.yml) — on hosted runners this is just a backstop re-run of the fast suite
swift test --filter MLXLocalBackendContractTests # real-model streaming contract; needs RUN_SLOW_TESTS=1 + MANIFOLD_DISCOVER_LOCAL_MODELS=1 + a local model (model-tests.yml, workflow_dispatch-only; always vacuous on hosted runners today)
scripts/test-mlx-integration.sh [<model-name>]   # real-model E2E via xcodebuild; discovers models under ~/Documents/Models; needs Apple Silicon + Metal + multi-GB snapshots; never run in CI
```

CI (`ci.yml`) runs on `macos-15` and force-selects the newest installed Xcode 26 toolchain before building/testing — the runner-image default (Xcode 16.4 / Swift 6.1) mis-resolves the dependency graph against core's trait-disabled `mlx-swift-lm` 2.x edge.

## Constraints & gotchas

- Command-line `swift build` produces `mlx.metallib` automatically only if the Metal Toolchain component is installed (`xcodebuild -downloadComponent MetalToolchain`). Without it the build still succeeds but the generate step aborts at GPU init with "Failed to load the default metallib" — backend registration, model discovery, and load-planning keep working regardless. A normal Xcode app-target build always has it; a bare `swift run`/`swift build` CLI may not.
- `swift test` must never run with `--parallel`: the contract-suite claims registry (`ManifoldBackendTestKit`'s `BackendContractChecks`) is process-global and races under explicit parallelism.
- Real-model tests (`Tests/ManifoldMLXIntegrationTests`, and the two `RUN_SLOW_TESTS`-gated streaming scenarios in `MLXLocalBackendContractTests`) need Apple Silicon + Metal + local model snapshots. No self-hosted Apple-Silicon runner is registered yet, so `model-tests.yml` and the `integration-tests` job in `ci.yml` cannot produce real coverage on hosted GitHub runners — a green run there does not mean the model path executed; check for the "non-vacuity guard" pass in the log.
- A plain `swift test` on a dev Mac without real GPU/Metal device access can crash (not skip) with an MLX "Failed to load the default metallib" fatal error inside `ManifoldMLXIntegrationTests` (e.g. `MLXGrammarSamplingE2ETests`), even when a compiled `mlx.metallib` is present on disk — observed once in a sandboxed session on this machine. `ci.yml`'s hosted `macos-15` runner has real GPU access and is consistently green, so treat a local crash as an environment check first, not a regression.
- MLX backend is `rendersFullPrompt` — core's `preferTools` steering never fires for it. Tool-calling steering has to happen MLX-side (system-prompt / grammar level), not by relying on the core seam.
- Grammar-constrained tool-call decoding (issue #96) wraps the bare envelope in `<tool_call>` and drops `kvBits`; it is O(vocab) per step (perf tracked in #97) — don't assume it's free.
- Dense Gemma 4 models crash at tick (`broadcast_shapes`) and are guarded at load + skipped in E2E (PR #83) — this is a permanent guard around a real MLX/mlx-swift-lm limitation, not a stale TODO; don't remove it expecting it now works without re-verifying upstream first.
- `Sources/FluxSwift` (from `mzbac/flux.swift`) and `Sources/StableDiffusion` (from `ml-explore/mlx-swift-examples`) are vendored MIT sources, not package dependencies — upstream `flux.swift` pins `swift-transformers` 0.1.x while ManifoldKit requires 1.2.x+. Keep the in-tree provenance headers; re-check that pin conflict before ever converting either to a `.package` dependency.
- Pre-`0.1.0` tag, targets/products keep temporary non-`Kit`-suffixed canonical names (`ManifoldMLX`, `FluxSwift`, `StableDiffusion`) reserved against SwiftPM's package-graph-wide target-name uniqueness constraint against core. See the `NOTE(C2)` block at the top of `Package.swift` before renaming anything here.
- `manifold-tools-mlx` no longer vendors any `ManifoldTools` scenarios/fixtures — they resolve from the published `ManifoldTools` resource bundle (core 0.62+ / #2042). Don't reintroduce a local fixture tree.
- This package pins ManifoldKit via `.upToNextMinor(from: "0.65.0")` (traits: `[]`). `core-bump.yml` auto-bumps that pin on every ManifoldKit release, gates on a full rebuild+retest, and admin-merges the resulting PR. A pure pin republish is committed as `deps:` (+ a `Release-As:` trailer forcing the patch), so it lands under the CHANGELOG's **Dependencies** heading rather than **Bug Fixes** — it isn't a bug fix and shouldn't be counted as one. The shared workflow falls back to `fix(deps):` when any `feat`/`fix`/breaking change is already queued since the last tag (letting release-please compute the version rather than forcing a patch that could under-version it). The `deps` section is made visible by the explicit `changelog-sections` in `release-please-config.json`; release-please's empty-config default would discard it. Convention owned centrally — see ManifoldKit `AGENTS.md` → "Companion pin-bump releases".
- On spike branches (e.g. `spike/bfcl-mlx-driver`) `Package.swift` may carry a local path pin to `../ManifoldKit` instead of a version pin — that's intentional while the branch depends on an unreleased core API, not drift to "fix" reflexively.
- No secrets are read by this package — there is nothing for a `.env.tpl` to cover here (estate's `env_tpl_if_secrets` requirement is N/A unless that changes).

## Conventions

- Estate-wide rules apply (worktrees, secrets via `op run --env-file .env.tpl`, conventional commits) — see `~/Repos/estate/estate.yaml` `conventions:`.
- Family-wide Swift conventions (concurrency, PR/draft-PR review loop, platform policy) live in ManifoldKit's own `AGENTS.md` — this file only covers what's specific to this companion repo.
