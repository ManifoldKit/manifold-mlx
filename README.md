# manifold-mlx

MLX inference and diffusion backends for [ManifoldKit](https://github.com/roryford/ManifoldKit) — the `ManifoldMLX` module plus its vendored `FluxSwift` and `StableDiffusion` targets, split out of the core package as part of the v0.48 packaging release (ManifoldKit#1749) so that `swift build` of core never drags mlx-swift, and heavy backends are one `.package` line away.

It provides Apple-Silicon-native text generation (mlx-swift-lm model families incl. MoE Gemma 4 via MLXVLM routing), prompt/KV cache coordination, a resource arbiter, capability probing, and FLUX.1 / Stable Diffusion image generation.

> **Temporary module names — pre-0.48 only.** Until ManifoldKit's C2 removal PR deletes the in-core `ManifoldMLX`/`FluxSwift`/`StableDiffusion` targets, SwiftPM's graph-wide target-name uniqueness forces this package to ship the modules as **`ManifoldMLX`**, `FluxSwift`, `StableDiffusion`. They are renamed to their real names in one commit before the first `0.1.0` tag (see the `NOTE(C2)` in `Package.swift`). If you are reading this after a 0.1.0 tag exists and still see `Kit` names, file an issue.

## Install

```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/roryford/ManifoldKit", branch: "main"),
    .package(url: "https://github.com/roryford/manifold-mlx", branch: "main"),
],
targets: [
    .target(name: "MyApp", dependencies: [
        .product(name: "ManifoldKit", package: "ManifoldKit"),
        .product(name: "ManifoldMLX", package: "manifold-mlx"),
    ]),
]
```

Register the backend via the `MLXBackends` registrar (the seam shipped in core's B2 work; the registrar moved here in core's C2 split):

```swift
import ManifoldKit
import ManifoldMLX

let kit = try await ManifoldKit.quickStart(backends: [MLXBackends.self])
```

> [!IMPORTANT]
> **MLX needs an Xcode `.app` build — it cannot generate from a plain `swift run` / bare SwiftPM executable.** mlx-swift compiles its Metal kernels into a `default.metallib` that only the Xcode / `xcodebuild` build path produces and bundles; a SwiftPM executable never builds it, so generation aborts at model load with `MLX error: Failed to load the default metallib` (`mlx-c/.../stream.cpp`). Everything up to that point — backend registration, model discovery/classification, and the load *plan* — works under `swift run`; only the generate step fails.
>
> A normal SwiftUI **app** target is an Xcode build, so it works out of the box. For a **headless / CLI** local backend use the GGUF/llama.cpp companion ([manifold-llama](https://github.com/roryford/manifold-llama)) instead, which runs fine from `swift run`, or build your CLI target with `xcodebuild`. See ManifoldKit's [`docs/QUICKSTART-CLI.md` §4](https://github.com/roryford/ManifoldKit/blob/main/docs/QUICKSTART-CLI.md) for the full constraint and recipe.

## Compatibility

| manifold-mlx | ManifoldKit |
|---|---|
| `main` | `main` (pre-0.48) |
| `0.1.0` (not yet tagged) | `0.48.x` (`.upToNextMinor` pin) |

Pre-tag, this package tracks core `main` by branch; the pin flips to `.upToNextMinor(from: "0.48.0")` at the 0.48 release train.

## Tests

```bash
swift build
swift test   # no --parallel: the contract-suite claims registry is process-global
```

The unit/contract suite is the MLX-family subset of core's `ManifoldBackendsTests` plus the shared `ManifoldBackendTestKit` checks; Metal-dependent tests skip themselves off Apple Silicon.

Real-model E2E tests live in `Tests/ManifoldMLXIntegrationTests` and are **not** run in CI (they need Metal and multi-GB local model snapshots). Run them locally:

```bash
scripts/test-mlx-integration.sh            # discovers models under ~/Documents/Models
scripts/test-mlx-integration.sh <name>     # prefer a model dir matching <name>
```

## Provenance & history

Imported as a fresh copy from `roryford/ManifoldKit` (see the `Imported-From:` trailer on the import commit). **History before 2026-06 lives in [ManifoldKit](https://github.com/roryford/ManifoldKit)** — `git log` there for the archaeology.

`Sources/FluxSwift` is vendored from [mzbac/flux.swift](https://github.com/mzbac/flux.swift) (MIT) and `Sources/StableDiffusion` from [ml-explore/mlx-swift-examples](https://github.com/ml-explore/mlx-swift-examples) (MIT, LICENSE kept in-tree) — vendored because upstream flux.swift pins swift-transformers 0.1.x while ManifoldKit requires 1.2.x.

## License

MIT — see [LICENSE](LICENSE). Vendored components keep their upstream MIT licenses and provenance headers.
