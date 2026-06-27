import Foundation
import os
import ManifoldInference

/// Stages the bundled `mlx.metallib` next to the running binary so mlx-swift's
/// colocated metallib lookup can find it under a plain command-line `swift
/// build` (issue #82).
///
/// Background
/// ----------
/// mlx-swift's `Device` loads a precompiled metallib at first GPU use and
/// aborts with "Failed to load the default metallib" if none is found. Its
/// search order (mlx `device.cpp::load_default_library`) is:
///
///   1. `<binary dir>/mlx.metallib`              ← colocated (what we target)
///   2. `<binary dir>/Resources/mlx.metallib`
///   3. `mlx-swift_Cmlx.bundle/default.metallib` ← Xcode-only, hardcoded name
///   4. `<binary dir>/Resources/default.metallib`
///   5. a fixed fallback path
///
/// Under Xcode/`xcodebuild` the Metal-shader compile phase fills #3. Under a
/// command-line `swift build` nothing fills any of them. The only location a
/// *dependency* package can populate is #1, since #3 is hardcoded to mlx-swift's
/// own bundle name. The `MLXMetallibPlugin` prebuild plugin compiles the
/// resolved mlx-swift kernels into this package's resource bundle as
/// `mlx.metallib`; this shim copies that resource to `<binary dir>/mlx.metallib`
/// before MLX initialises the GPU device.
///
/// `<binary dir>` is mlx's `current_binary_dir()` — `dladdr`-resolved to the
/// directory of the image the mlx code is linked into. For the static SwiftPM
/// link that is the running executable / test binary, which lives in the same
/// directory as this package's resource bundle. We therefore stage to the
/// resource bundle's parent directory (and, defensively, to the main
/// executable's directory when different).
///
/// Best-effort and idempotent: missing resource, read-only destinations, or a
/// metallib already in place are all no-ops. If staging can't happen the build
/// behaves exactly as before — it compiles, and only the MLX generate path
/// aborts at GPU init.
public enum MLXMetallibStaging {

    private static let logger = Logger(
        subsystem: ManifoldConfiguration.shared.logSubsystem,
        category: "mlx-metallib"
    )

    /// Runs the staging copy at most once per process. Safe and cheap to call
    /// from any MLX entry point before the first GPU operation.
    public static func ensureStaged() {
        _ = stagedOnce
    }

    /// `let`-backed one-shot: the closure body runs exactly once, on first
    /// access, with Swift's lazy-static thread-safety.
    private static let stagedOnce: Void = {
        performStaging()
    }()

    private static func performStaging() {
        guard let source = Bundle.module.url(forResource: "mlx", withExtension: "metallib") else {
            // No bundled metallib — the prebuild plugin produced nothing (Metal
            // toolchain absent) or this is an Xcode build that supplies its own.
            logger.debug("No bundled mlx.metallib to stage; relying on the build's own metallib.")
            return
        }

        var destinations: [URL] = []
        // Primary: the resource bundle's parent == the binary dir for a static
        // SwiftPM link (executable/test binary and bundle are colocated).
        destinations.append(Bundle.module.bundleURL.deletingLastPathComponent())
        // Defensive: the main executable's directory, when it differs.
        if let exeDir = Bundle.main.executableURL?.resolvingSymlinksInPath().deletingLastPathComponent() {
            destinations.append(exeDir)
        }

        let fileManager = FileManager.default
        var seen = Set<String>()
        for directory in destinations {
            let key = directory.standardizedFileURL.path
            guard seen.insert(key).inserted else { continue }
            stage(source: source, intoDirectory: directory, using: fileManager)
        }
    }

    private static func stage(source: URL, intoDirectory directory: URL, using fileManager: FileManager) {
        let destination = directory.appendingPathComponent("mlx.metallib")

        // Skip if an identically sized metallib is already in place.
        if let existing = try? destination.resourceValues(forKeys: [.fileSizeKey]).fileSize,
           let incoming = try? source.resourceValues(forKeys: [.fileSizeKey]).fileSize,
           existing == incoming {
            logger.debug("mlx.metallib already staged at \(destination.path, privacy: .public)")
            return
        }

        do {
            if fileManager.fileExists(atPath: destination.path) {
                try fileManager.removeItem(at: destination)
            }
            try fileManager.copyItem(at: source, to: destination)
            logger.info("Staged mlx.metallib → \(destination.path, privacy: .public)")
        } catch {
            // Read-only or otherwise unwritable destination: harmless, another
            // candidate (or the build's own metallib) may still satisfy mlx.
            logger.debug(
                "Could not stage mlx.metallib to \(destination.path, privacy: .public): \(error.localizedDescription, privacy: .public)"
            )
        }
    }
}
