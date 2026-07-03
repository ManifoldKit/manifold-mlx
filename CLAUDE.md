# manifold-mlx — Claude Code Instructions

MLX inference + diffusion/image/video backends for [ManifoldKit](https://github.com/ManifoldKit/ManifoldKit),
split out of core in the v0.48 packaging release (ManifoldKit#1749) so `swift build`
of core never drags in mlx-swift. Product/module: **`ManifoldMLX`**. Also vendors
two MIT-licensed targets, **`FluxSwift`** (from `mzbac/flux.swift`) and
**`StableDiffusion`** (from `ml-explore/mlx-swift-examples`) — vendored, not a
package dependency, because upstream `flux.swift` pins `swift-transformers` 0.1.x
while ManifoldKit requires 1.2.x+.

For family-wide conventions (Swift 6 concurrency gotchas, commit style, PR/draft-PR
review loop, platform policy) see ManifoldKit's own `CLAUDE.md` — this file only
covers what's specific to this companion repo.

## Testing

```bash
swift build
swift test   # no --parallel: BackendContractChecks' claims registry is process-global
```

Real-model E2E tests (`Tests/ManifoldMLXIntegrationTests`) need Apple Silicon +
Metal + local model snapshots and are **not run in CI**. Run locally via
`scripts/test-mlx-integration.sh` — it drives `xcodebuild build-for-testing` /
`test-without-building` with a patched `.xctestrun` because `xcodebuild` does not
propagate env vars to the spawned test runner otherwise, and the suite must run
serially (self-hosted-only lane in `ci.yml`, `workflow_dispatch` gated).

## Metal / metallib constraints

mlx-swift aborts at GPU init with "Failed to load the default metallib" unless a
compiled `mlx.metallib` is colocated with the binary. The `MLXMetallibPlugin`
SwiftPM prebuild plugin (see `scripts/build-mlx-metallib.sh`) compiles it
automatically during `swift build`, provided the Metal Toolchain component is
installed (`xcodebuild -downloadComponent MetalToolchain`); it degrades gracefully
(no metallib, but build still succeeds) when the toolchain is missing. No Metal in
the iOS Simulator — this only matters for macOS / device runs.

## Pin / release model

This repo pins ManifoldKit with `.upToNextMinor(from: "0.63.0")` in `Package.swift`
(traits: `[]`, the post-C2 trait-less world). `core-bump.yml` auto-bumps this pin
on every ManifoldKit release (via `repository_dispatch: core-release`), gates on a
full rebuild+retest against the new core, and admin-merges a `fix:` PR — which
trips this repo's own `release-please` for a patch release. Commits here follow
Conventional Commits; Release Please reads them for version bumps.

## No more vendored scenarios / fixtures

`manifold-tools-mlx` no longer vendors any ManifoldTools content. Scenarios
resolve via `ScenarioLoader.loadBuiltIn()` (`Bundle.module` on the published
`ManifoldTools` product, since MK 0.62 / #2042) and the `read_file`/`list_dir`
fixture tree resolves via `ScenarioCLIHarness.resolveFixturesRoot(_:)` →
`ToolFixtures.bundledRoot()` — both ship inside `ManifoldTools`'s own resource
bundle, so there is nothing left here to drift or re-sync. The former
hand-copied `Sources/manifold-tools-mlx/Fixtures/manifold-tools/` tree and
`scripts/check-vendored-sync.sh` drift-check (plus its `ci.yml`
`vendored-sync-check` job) were removed accordingly — see the cross-repo
simplification plan item D1.
