# MLX backend fixtures

MLX generates non-deterministic token streams because the Metal GPU scheduler
does not guarantee bit-for-bit reproducibility across runs. Fixtures for the
`LocalBackendContractTests` MLX participant are captured by hand on a specific
model against Apple Silicon hardware and committed to this directory.

## Capturing fixtures

There is no automated recording tool in this repo (core's
`scripts/record-fixture.sh` is a generic SSE/NDJSON redaction pipe with an
incompatible stdin interface — it does not drive a backend). To re-record by
hand: with `RUN_SLOW_TESTS=1` set and an MLX model discoverable
(`MLX_TEST_MODEL=<name-or-path>` or `MANIFOLD_DISCOVER_LOCAL_MODELS=1`) on
Apple Silicon, call
`generate(prompt: "Hello", systemPrompt: nil, config: GenerationConfig())`
through `MLXBackend` (mirror `MLXLocalBackendContractTests.makeBackend`'s
model-loading path), collect the emitted `GenerationEvent`s, and write each
one as a JSON object per line into `streaming/simple-prompt/expected.jsonl`.
Example output (model-dependent):

```jsonl
{"event":"token","text":"Hello"}
{"event":"token","text":"!"}
{"event":"token","text":" How"}
```

## Nightly tier

The MLX contract participant skips generation scenarios unless
`RUN_SLOW_TESTS=1` is set in the environment. Per-PR CI does not set this
variable, so these fixtures are only required in the nightly tier where a
real model and Apple Silicon hardware are present.
