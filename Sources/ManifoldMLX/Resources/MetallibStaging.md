# MLX metallib staging marker

This file exists so SwiftPM always generates `Bundle.module` for the
`ManifoldMLX` target, even on builds where the `MLXMetallibPlugin` prebuild
plugin produces no `mlx.metallib` (e.g. when the Metal Toolchain component is
not installed, so the kernels can't be compiled).

When the plugin *does* compile the kernels, the resulting `mlx.metallib` is
bundled alongside this file and `MLXMetallibStaging` copies it next to the
running binary at first use, satisfying mlx-swift's colocated metallib lookup.
See `MLXMetallibStaging.swift` and `Plugins/MLXMetallibPlugin/plugin.swift`.
