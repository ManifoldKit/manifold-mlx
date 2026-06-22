// swift-tools-version: 6.1
import PackageDescription

// NOTE(C2, resolved): the targets/products/modules carried temporary `Kit`
// suffixes while core still declared `ManifoldMLX` / `FluxSwift` /
// `StableDiffusion` / `ManifoldMLXIntegrationTests` targets (SwiftPM requires
// target names to be unique across the package graph). Core's C2 removal PR
// deletes those targets; this branch restores the canonical names and merges
// immediately after core's C2.
let package = Package(
    name: "manifold-mlx",
    platforms: [
        .iOS(.v18),
        .macOS(.v15),
    ],
    products: [
        .library(name: "ManifoldMLX", targets: ["ManifoldMLX"]),
        .executable(name: "manifold-tools-mlx", targets: ["manifold-tools-mlx"]),
    ],
    dependencies: [
        // Pulls ManifoldInference/Tools/Runtime/PersistenceSwiftData plus the
        // ManifoldTestSupport / ManifoldBackendTestKit products this package
        // consumes.
        // traits: [] builds core's products trait-less (the post-C2 world).
        .package(url: "https://github.com/roryford/ManifoldKit", .upToNextMinor(from: "0.59.0"), traits: []),
        // Pins copied from core's Package.swift.
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.31.3"),
        // 3.31.3 ships the decoupled MLXHuggingFace target and adds the
        // `gemma4` model_type to LLMTypeRegistry.
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", from: "3.31.3"),
        // Explicit dep required: mlx-swift-lm no longer pulls
        // swift-transformers transitively.
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.2.0"),
        // swift-log: pulled in by vendored FluxSwift source.
        .package(url: "https://github.com/apple/swift-log.git", from: "1.5.0"),
    ],
    targets: [
        // Vendored FluxSwift (mzbac/flux.swift, MIT — see Sources/FluxSwift
        // provenance headers). Vendored instead of a package dependency
        // because flux.swift pins swift-transformers 0.1.x; ManifoldKit
        // requires 1.2.x.
        .target(
            name: "FluxSwift",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "Logging", package: "swift-log"),
            ],
            path: "Sources/FluxSwift"
        ),
        // Vendored StableDiffusion (from mlx-swift-examples, MIT — LICENSE
        // kept in-tree), used by MLXDiffusionBackend.
        .target(
            name: "StableDiffusion",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "Hub", package: "swift-transformers"),
            ],
            path: "Sources/StableDiffusion",
            exclude: ["LICENSE"]
        ),
        // MLX inference backend, resource arbiter, capability probe,
        // MLX-specific tool dialect, and the FLUX / Stable Diffusion image
        // backends. Imported from roryford/ManifoldKit (see the Imported-From
        // commit trailer); the `#if MLX` / `#if HuggingFace` trait gates were
        // stripped at import — both are always-on here.
        .target(
            name: "ManifoldMLX",
            dependencies: [
                .product(name: "ManifoldInference", package: "ManifoldKit"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                // MLXVLM ships the MoE Gemma 4 decoder that LLMModelFactory's
                // Gemma4Text.swift lacks; MLXBackend routes
                // `text_config.enable_moe_block == true` models to
                // VLMModelFactory.shared.loadContainer. See ManifoldKit#752.
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                // No MLXHuggingFace product here: its macro plugin pulls
                // swift-syntax into every build. The one macro we used is
                // hand-expanded in TransformersTokenizerLoader.swift.
                .product(name: "Tokenizers", package: "swift-transformers"),
                // Hub is consumed directly by the FLUX diffusion backend for
                // repository snapshot downloads.
                .product(name: "Hub", package: "swift-transformers"),
                "StableDiffusion",
                "FluxSwift",
            ],
            path: "Sources/ManifoldMLX"
        ),
        .testTarget(
            name: "ManifoldMLXTests",
            dependencies: [
                "ManifoldMLX",
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "ManifoldInference", package: "ManifoldKit"),
                .product(name: "ManifoldRuntime", package: "ManifoldKit"),
                .product(name: "ManifoldPersistenceSwiftData", package: "ManifoldKit"),
                .product(name: "ManifoldTestSupport", package: "ManifoldKit"),
                .product(name: "ManifoldBackendTestKit", package: "ManifoldKit"),
            ]
        ),
        // Real-model E2E tests: require Apple Silicon + Metal + local MLX
        // model snapshots. Run via scripts/test-mlx-integration.sh (xcodebuild
        // with a patched .xctestrun) — every test XCTSkips under plain
        // `swift test` unless MANIFOLD_DISCOVER_LOCAL_MODELS=1 is injected.
        .testTarget(
            name: "ManifoldMLXIntegrationTests",
            dependencies: [
                "ManifoldMLX",
                "FluxSwift",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "ManifoldInference", package: "ManifoldKit"),
                .product(name: "ManifoldRuntime", package: "ManifoldKit"),
                .product(name: "ManifoldPersistenceSwiftData", package: "ManifoldKit"),
                .product(name: "ManifoldTestSupport", package: "ManifoldKit"),
            ],
            path: "Tests/ManifoldMLXIntegrationTests"
        ),
        // Tool-calling validation CLI: reuses ManifoldKit's published
        // ManifoldTools library (its bundled scenarios + reference toolset)
        // and drives them against a real MLX model. Needs Apple Silicon +
        // Metal + a local model dir to actually run; compiles everywhere.
        .executableTarget(
            name: "manifold-tools-mlx",
            dependencies: [
                .product(name: "ManifoldTools", package: "ManifoldKit"),
                .product(name: "ManifoldInference", package: "ManifoldKit"),
                "ManifoldMLX",
            ],
            path: "Sources/manifold-tools-mlx",
            exclude: ["README.md"],
            resources: [
                .copy("Fixtures/manifold-tools"),
                .copy("Scenarios/built-in"),
            ]
        ),
    ]
)
