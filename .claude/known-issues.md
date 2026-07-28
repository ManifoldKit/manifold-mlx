# Known issues

Symptom → root cause → fix, for non-obvious build/CI/infra failures in this
repo. **Check here before diagnosing anything.** Append a line when you solve
something that cost real diagnosis time.

## CI / release

- **Release PR sits `BLOCKED` with "no checks reported", nothing failing** →
  `ci.yml` deliberately skips release-please PRs via `paths-ignore`
  (`CHANGELOG.md`, `.release-please-manifest.json`), but `test / build-and-test`
  is a required check, so it can never report → the repo's documented release
  flow is an **admin merge**. See the comment at the top of `ci.yml`. Note a
  manual `gh workflow run ci.yml --ref release-please--branches--main` does
  **not** satisfy the required context — verified 2026-07-28, the dispatched run
  went green and the PR stayed `BLOCKED`.

- **A PR's only failing check is `test / build-and-test` with "This PR is a
  draft, so the build/test gate did not run"** → not a real failure. It is a
  deliberate red sentinel that blocks accidental merges of unverified drafts.
  `gh pr ready` is the CI trigger; the check runs for real once out of draft.

- **ManifoldKit release PR shows no checks and its workflow runs sit in
  `action_required`** → release-please's token doesn't auto-approve workflow
  runs. Approve them:
  `gh api -X POST repos/ManifoldKit/ManifoldKit/actions/runs/<id>/approve`.

- **`core-bump.yml` fails after a ManifoldKit release** → usually the pin bump
  alone doesn't compile or test against the new core, i.e. this repo needs a
  real adaptation and the auto-bump can't land by itself. Read the failing test
  before assuming the workflow is broken (e.g. core v0.75.0's optional
  `contextWindow` needed manifold-mlx#167).

## Tests

- **`ManifoldMLXIntegrationTests` crashes with "Failed to load the default
  metallib" under `swift test`** → that target cannot run via SwiftPM; the
  metallib isn't staged where the xctest bundle looks. Use
  `scripts/test-mlx-integration.sh [<model>] [--only <Class>]`, which builds via
  xcodebuild and patches the `.xctestrun` with the discovery env vars.

- **An integration test "passes" instantly** → it almost certainly `XCTSkip`ped.
  This target skips silently without `MANIFOLD_DISCOVER_LOCAL_MODELS=1` /
  `MLX_TEST_MODEL`, and a green run does not mean the model path executed.
  Check for per-test `skipped` in the log, and prefer tests that assert their
  own non-vacuity (see `MLXCancelResendSoakTests`).

## Local environment

- **`git commit` fails with `error: 1Password: failed to fill whole buffer` /
  `fatal: failed to write commit object`** → the 1Password SSH agent has locked;
  commit signing can't reach the key. Unlock the 1Password app and retry. Do
  **not** work around it with `--no-gpg-sign`. Staged work is safe meanwhile.
