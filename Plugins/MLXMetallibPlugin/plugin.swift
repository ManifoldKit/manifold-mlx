import Foundation
import PackagePlugin

/// SwiftPM **prebuild** plugin that compiles mlx-swift's generated Metal
/// kernels into a colocated `mlx.metallib` during a plain `swift build`.
///
/// Why this exists
/// ---------------
/// mlx-swift's `Device` constructor loads a precompiled metallib and aborts if
/// none is found (mlx `device.cpp::load_default_library`). That metallib is
/// normally produced by an Xcode / `xcodebuild` Metal-shader compile phase; a
/// command-line `swift build` never runs it, so the MLX inference path dies at
/// GPU init with "Failed to load the default metallib". See issue #82.
///
/// This plugin runs `scripts/build-mlx-metallib.sh` against the resolved
/// mlx-swift checkout's generated kernels (`Source/Cmlx/mlx-generated/metal`)
/// and emits `mlx.metallib` into the plugin output directory, which SwiftPM
/// then bundles as a resource of the `ManifoldMLX` target. `MLXMetallibStaging`
/// (in `Sources/ManifoldMLX`) copies it next to the running executable at
/// startup so mlx-swift's colocated lookup (`<binary dir>/mlx.metallib`) finds
/// it — the only one of mlx's five search locations a dependency package can
/// populate.
///
/// Compiling the resolved checkout (rather than vendoring a prebuilt binary)
/// keeps the metallib in lockstep with whatever mlx-swift version resolves, and
/// avoids committing a large, toolchain-coupled binary blob.
///
/// Graceful degradation: if the Metal toolchain is unavailable (no Metal
/// Toolchain component, non-macOS host), the script warns and produces no
/// output; the build still succeeds and behaves exactly as before this change
/// (compiles fine, only aborts at MLX GPU init).
@main
struct MLXMetallibPlugin: BuildToolPlugin {
    func createBuildCommands(context: PluginContext, target: Target) async throws -> [Command] {
        guard let mlxDir = mlxSwiftDirectory(in: context) else {
            Diagnostics.warning(
                "MLXMetallibPlugin: could not locate the mlx-swift dependency checkout; "
                + "skipping default.metallib build. MLX GPU init will fail at runtime "
                + "unless a metallib is produced another way (e.g. an Xcode build)."
            )
            return []
        }

        let generatedMetalDir = mlxDir.appending(path: "Source/Cmlx/mlx-generated/metal")
        let script = context.package.directoryURL.appending(path: "scripts/build-mlx-metallib.sh")
        let outputDir = context.pluginWorkDirectoryURL.appending(path: "Generated")
        // `context.tool(named:)` only resolves toolchain / in-package tools, not
        // PATH binaries, so reference the system shell by absolute path. (An
        // in-package executable target can't be used here either: prebuild
        // commands run before the package's own targets are built.)
        let bash = URL(filePath: "/bin/bash")

        return [
            .prebuildCommand(
                displayName: "Compiling mlx-swift Metal kernels → mlx.metallib",
                executable: bash,
                arguments: [
                    script.path(percentEncoded: false),
                    generatedMetalDir.path(percentEncoded: false),
                    outputDir.path(percentEncoded: false),
                ],
                outputFilesDirectory: outputDir
            )
        ]
    }

    /// Finds the resolved mlx-swift package's source directory among this
    /// package's dependencies.
    private func mlxSwiftDirectory(in context: PluginContext) -> URL? {
        for dependency in context.package.dependencies {
            let pkg = dependency.package
            if pkg.id.lowercased() == "mlx-swift"
                || pkg.displayName.lowercased() == "mlx-swift"
                || pkg.directoryURL.lastPathComponent.lowercased() == "mlx-swift" {
                return pkg.directoryURL
            }
        }
        return nil
    }
}
