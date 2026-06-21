# Llama 3.2 MLX tool-call fixtures (issue #59)

`calc-dispatch.live.jsonl` — a real JSONL transcript captured from
`manifold-tools-mlx --model …/Llama-3.2-3B-Instruct-4bit --scenario 02-calc`
on Apple Silicon **after** the issue-59 fix landed. It shows `llama-3.2-3b`
dispatching the `calc` tool (a `tool_call` event with
`{"a":7823,"b":41,"op":"*"}`), receiving the result `320743`, and quoting it in
the final answer.

Before the fix `MLXToolDialect.detect` mapped `model_type == "llama"` to
`.unknown`, so no tool grammar was injected and the model never produced a
parseable call — it narrated the call as prose (`calc(7823 * 41) = …`) or
invented its own `<calc>…</calc>` wrapper.

This is a documentation / provenance fixture (not wired into a test bundle); the
machine-checked regression coverage lives in
`Tests/ManifoldMLXTests/MLXLlamaToolDialectTests.swift`, which pins the exact
`<tool_call>{"name":…,"parameters":…}</tool_call>` body the live model emits.
