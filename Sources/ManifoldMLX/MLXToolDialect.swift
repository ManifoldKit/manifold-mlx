import Foundation
import ManifoldHardware

/// Identifies the tool-call dialect a locally loaded MLX model uses.
///
/// Different model families emit tool invocations in different text formats.
/// `MLXToolDialect` is detected at load time by reading `config.json` from
/// the model directory, so the generate path can apply the right injection
/// and parsing strategy without guessing per token.
///
/// Currently supported dialects:
/// - `.qwen25` — Qwen 2.5 / Qwen 3 format: tools injected as a
///   `<tools>…</tools>` JSON block appended to the system message; model
///   emits `<tool_call>{"name":"…","arguments":{…}}</tool_call>`.
/// - `.llama` — Llama 3.x format: tools injected as a JSON list with the
///   official "respond with a JSON object" instruction; model emits a
///   `<|python_tag|>{"name":…,"parameters":…}` block (or a bare
///   `{"name":…,"parameters":…}` JSON object).
/// - `.mistral` — Mistral / Mixtral format: tools injected as a JSON array
///   with the `[TOOL_CALLS]` wire format instruction; model emits
///   `[TOOL_CALLS] [{"name":…,"arguments":{…}}, …]` (parallel calls in one
///   JSON array, sentinel prefix, no closing delimiter — ends at EOS).
/// - `.unknown` — no recognised tool dialect; tool calling is a no-op.
public enum MLXToolDialect: Equatable, Sendable {
    /// Qwen 2.5 / Qwen 3 tool-call format.
    ///
    /// Tool definitions are serialised as a JSON array and wrapped in
    /// `<tools>…</tools>` tags appended to the system message (or injected
    /// as a synthetic system message when the caller did not supply one).
    /// The model responds with one or more `<tool_call>…</tool_call>` blocks,
    /// each containing a JSON object with `"name"` and `"arguments"` keys.
    case qwen25

    /// Llama 3.x tool-call format.
    ///
    /// Tool definitions are serialised as a JSON list and appended to the
    /// system message with Llama's "you have access to the following
    /// functions … respond with a JSON for a function call" instruction. The
    /// model emits the call either prefixed with the `<|python_tag|>` special
    /// token or as a bare top-level JSON object, with `"name"` and
    /// `"parameters"` keys (Llama uses `parameters`, not Qwen's `arguments`).
    /// Without this injection Llama-3.2 never produces a parseable tool call —
    /// it narrates the call as prose or invents its own `<tool>…</tool>`
    /// wrapper (issue #59).
    case llama

    /// Mistral / Mixtral tool-call format (issue #86).
    ///
    /// Tool definitions are serialised as a JSON array and appended to the
    /// system message instructing the model to emit the `[TOOL_CALLS]` sentinel
    /// prefix. The model responds with:
    ///   `[TOOL_CALLS] [{"name": "fn", "arguments": {…}}, …]`
    /// One or more calls appear in a single JSON array; parallel calls are
    /// supported natively. The sentinel has no closing delimiter — the block
    /// ends at end-of-generation (`closesAtEnd: true`). Both the `arguments`
    /// key and the tolerated `parameters` alias are accepted during parsing.
    /// Mistral's chat template rejects a standalone `system` role and enforces
    /// strict user/assistant alternation; the system message is folded into the
    /// first user turn by `MLXPromptCacheCoordinator` when the template raises.
    case mistral

    /// No recognised tool dialect — tool calling is disabled for this model.
    case unknown

    // MARK: - Core dialect mapping

    /// Maps this internal dialect to the corresponding ``ToolCallDialect`` from
    /// `ManifoldHardware` so ``MLXBackend`` can surface it on
    /// ``BackendCapabilities/toolDialect``.
    ///
    /// Llama uses `<tool_call>` as its *primary* textual delimiter (the one the
    /// model is steered onto via ``MLXChatMessageEncoder``), but falls back to
    /// `<|python_tag|>` / `<|eom_id|>` when the detokeniser surfaces those
    /// special tokens. We report the fallback delimiter pair here — the primary
    /// `<tool_call>` path is identical to Qwen's and is already handled by the
    /// `.qwen25` mapping — so callers can see that the Llama path is buried
    /// (no guaranteed opening delimiter on the native path).
    public var coreDialect: ToolCallDialect {
        switch self {
        case .qwen25:
            return ToolCallDialect(
                family: .qwen,
                openDelimiter: "<tool_call>",
                closeDelimiter: "</tool_call>",
                argEncoding: .json,
                extractability: .clean
            )
        case .llama:
            return ToolCallDialect(
                family: .llamaPythonTag,
                openDelimiter: "<|python_tag|>",
                closeDelimiter: "<|eom_id|>",
                argEncoding: .json,
                extractability: .buried
            )
        case .unknown:
            return ToolCallDialect(
                family: .unknown,
                extractability: .toolLess
            )
        }
    }

    // MARK: - Detection

    /// Reads `config.json` inside `url` and returns the best-matching dialect.
    ///
    /// Detection is best-effort: a missing or unreadable config, or one that
    /// does not declare a recognised `model_type`, maps to `.unknown`.
    ///
    /// - Parameter url: The model directory URL (same one passed to
    ///   `MLXBackend.loadModel(from:plan:)`).
    /// - Returns: `.qwen25` when `config.json` reports `model_type == "qwen2"`
    ///   or `"qwen3"`; `.llama` when it reports `"llama"` / `"mllama"`;
    ///   `.unknown` otherwise.
    public static func detect(at url: URL) -> MLXToolDialect {
        let configURL = url.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return .unknown
        }

        let modelType = (json["model_type"] as? String)?
            .lowercased()
            .trimmingCharacters(in: .whitespaces) ?? ""

        // qwen2 covers both Qwen 2.5 and earlier 2.x checkpoints;
        // qwen3 / qwen3_* are also compatible with the same prompt format.
        if modelType.hasPrefix("qwen2") || modelType.hasPrefix("qwen3") {
            return .qwen25
        }

        // Llama 3.x (`model_type == "llama"`) and the multimodal Llama 3.2
        // vision checkpoints (`"mllama"`) share the `<|python_tag|>` + JSON
        // tool-call format. Llama 2 predates tool calling but uses the same
        // `model_type`, so the prefix match is intentionally broad — a Llama 2
        // checkpoint simply won't emit calls, which is the same no-op outcome
        // as `.unknown`.
        if modelType.hasPrefix("llama") || modelType.hasPrefix("mllama") {
            return .llama
        }

        // Mistral and the Mixtral MoE family (`"mistral"`, `"ministral"`)
        // share the `[TOOL_CALLS]` array format. `mixtral` and `mistral-nemo`
        // also carry `"mistral"` in their `model_type` on most HuggingFace
        // checkpoints, so the prefix match covers the whole family.
        if modelType.hasPrefix("mistral") || modelType.hasPrefix("ministral") {
            return .mistral
        }

        return .unknown
    }
}
