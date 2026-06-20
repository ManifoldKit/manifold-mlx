# manifold-tools-mlx

A tool-calling validation CLI that runs ManifoldKit's bundled tool-calling
scenarios against a **real MLX model** (e.g. Gemma) on Apple Silicon.

It reuses ManifoldKit's published `ManifoldTools` library product — its bundled
scenarios (`ScenarioLoader.loadBuiltIn()`), reference toolset, `ScenarioRunner`,
and JSONL `TranscriptLogger`. This target only adds the MLX backend wiring and a
small argument parser; no ManifoldKit core changes are needed.

## Usage

```sh
swift run manifold-tools-mlx --model /path/to/mlx/gemma-2-2b-it --scenario all
```

List the available scenarios (no model required):

```sh
swift run manifold-tools-mlx --list
```

### Flags

| Flag | Description |
|------|-------------|
| `--model <path>` | **Required** (except `--list` / `--help`). Path to the MLX model directory (`config.json` + tokenizer + safetensors). |
| `--scenario <id\|all>` | Scenario id or `all` (default). |
| `--output <path.jsonl>` | Transcript JSONL destination. Default `tmp/manifold-tools-mlx/<iso>.jsonl`. |
| `--fixtures-root <dir>` | Root for the `read_file` / `list_dir` / `repo_search` tools. Defaults to the fixtures bundled with this CLI. |
| `--list` | Print available scenarios and exit. |
| `--help` | Show usage. |

### Exit codes

- `0` — all scenarios passed
- `1` — at least one scenario / assertion failed, or a runtime error
- `2` — bad arguments

## Requirements

MLX inference requires **Apple Silicon + Metal** and a real model directory on
disk. Compilation works anywhere, but actually running scenarios needs a model
snapshot and the Metal runtime (an Xcode-built `metallib`).
