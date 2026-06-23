import Foundation
import ManifoldInference
import MLXLMCommon

/// MLX tool-call dialect markers for ``ToolCallTransform``.
///
/// Replaces the former `MLXToolCallParser`: the scanning / holdback / chunk
/// safety now lives once in `ToolCallTransform`, and only the MLX-specific
/// delimiters and JSON body parsing stay here, injected as a `@Sendable`
/// closure.
///
/// Three model families are supported:
/// - **Qwen 2.5 / Qwen 3** wrap tool calls in `<tool_call>` / `</tool_call>`
///   with a `{"name":…,"arguments":{…}}` JSON body.
/// - **Llama 3.x** emit a `<|python_tag|>`-prefixed JSON object (closed by the
///   `<|eom_id|>` / `<|eot_id|>` special tokens) — or a bare top-level JSON
///   object — with a `{"name":…,"parameters":{…}}` body. Without dialect-aware
///   markers Llama calls were never parsed (issue #59).
/// - **Mistral / Mixtral** emit `[TOOL_CALLS] [{"name":…,"arguments":{…}},…]`
///   — a bare `[TOOL_CALLS] ` sentinel followed by a JSON array of calls with
///   no closing delimiter (the block ends at end-of-generation). Multiple calls
///   in one array map to multiple `.toolCall` events (issue #86).
// @_spi(Testing): published only for backend test targets (companion-package split, #1749).
@_spi(Testing) public enum MLXToolMarkers {

    // Qwen 2.5 / Qwen 3 delimiters.
    private static let qwenOpenTag  = "<tool_call>"
    private static let qwenCloseTag = "</tool_call>"

    // Llama 3.x native delimiters — kept as *fallback* markers. The MLX
    // streaming detokenizer usually drops `<|python_tag|>` / `<|eom_id|>` /
    // `<|eot_id|>` as special tokens, so the primary Llama path is the textual
    // `<tool_call>` wrapper we steer the model onto (see
    // `MLXChatMessageEncoder.llamaToolBlock`). When a build *does* surface these
    // tokens we still parse them: `<|python_tag|>` opens a call terminated by
    // either `<|eom_id|>` (documented tool terminator) or `<|eot_id|>`.
    private static let llamaOpenTag  = "<|python_tag|>"
    private static let llamaEomTag   = "<|eom_id|>"
    private static let llamaEotTag   = "<|eot_id|>"

    // Mistral `[TOOL_CALLS]` sentinel. The model emits this prefix followed by
    // a JSON array of `{name, arguments}` objects (no closing delimiter — the
    // block ends at end-of-generation). `closesAtEnd: true` tells the transform
    // to buffer to EOS and parse in `finalize()`.
    private static let mistralOpenTag = "[TOOL_CALLS] "

    /// The marker set MLX hands to a `ToolCallTransform` for `dialect`.
    ///
    /// `.qwen25` → the single `<tool_call>` dialect. `.llama` → the
    /// `<|python_tag|>` dialect (two markers, one per close token). `.mistral`
    /// → the `[TOOL_CALLS] ` EOS-keyed multi-call dialect (one sentinel, no
    /// closing delimiter, JSON array body). `.unknown` → an empty set (the
    /// driver never builds a tool stage for `.unknown`, but returning `[]` keeps
    /// this total and side-effect-free).
    public static func markers(dialect: MLXToolDialect) -> [ToolCallMarker] {
        switch dialect {
        case .qwen25:
            return [
                ToolCallMarker(open: qwenOpenTag, close: qwenCloseTag) { body in
                    parseToolCall(body)
                }
            ]
        case .llama:
            // Primary: the textual `<tool_call>…</tool_call>` delimiters we
            // instruct Llama to emit (see `MLXChatMessageEncoder.llamaToolBlock`)
            // — these survive detokenisation, unlike Llama's native `<|eom_id|>`
            // terminator. Fallback markers cover a model that still reaches for
            // the `<|python_tag|>` prefix: if it does emit a visible `<|eom_id|>`
            // / `<|eot_id|>` close we parse it, but the primary path is what the
            // overnight scenarios exercise. The body parser accepts both the
            // Llama `parameters` key and the Qwen `arguments` key.
            return [
                ToolCallMarker(open: qwenOpenTag, close: qwenCloseTag) { body in
                    parseToolCall(body)
                },
                ToolCallMarker(open: llamaOpenTag, close: llamaEomTag) { body in
                    parseToolCall(body)
                },
                ToolCallMarker(open: llamaOpenTag, close: llamaEotTag) { body in
                    parseToolCall(body)
                },
            ]
        case .mistral:
            // Mistral emits `[TOOL_CALLS] [{"name":…,"arguments":{…}},…]`.
            // There is no closing delimiter — the array body runs to
            // end-of-generation, so `closesAtEnd: true` tells the transform to
            // buffer everything and call `parseBodyMulti` in `finalize()`.
            // One JSON array may carry multiple calls; `parseBodyMulti` returns
            // all of them and the transform emits one `.toolCall` per element.
            return [
                ToolCallMarker(open: mistralOpenTag, closesAtEnd: true) { body in
                    parseMistralToolCalls(body)
                }
            ]
        case .unknown:
            return []
        }
    }

    /// Back-compat overload: the original single-dialect (Qwen) marker set.
    ///
    /// Retained so existing call sites and tests that predate the dialect
    /// parameter keep compiling. New code should pass an explicit `dialect:`.
    public static func markers() -> [ToolCallMarker] {
        markers(dialect: .qwen25)
    }

    /// Parses a Mistral `[TOOL_CALLS]` body — a JSON array of
    /// `{"name":…,"arguments":{…}}` objects — into zero or more `ToolCall`s.
    ///
    /// The body is everything after the `[TOOL_CALLS] ` sentinel (which the
    /// transform strips as the open tag), so it is a raw JSON array starting
    /// with `[`. Both the `arguments` key (canonical) and the `parameters`
    /// alias (tolerated) are accepted by delegating each element to
    /// `parseToolCall(_:)`. Elements that fail to parse are dropped silently —
    /// the overall call is still valid if at least one element succeeds.
    private static func parseMistralToolCalls(_ body: String) -> [ManifoldInference.ToolCall] {
        let trimmed = body.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty,
              let data = trimmed.data(using: .utf8),
              let array = (try? JSONSerialization.jsonObject(with: data)) as? [[String: Any]]
        else {
            return []
        }

        return array.compactMap { element -> ManifoldInference.ToolCall? in
            // Re-serialise each element and reuse the existing single-call
            // parser so the `arguments` / `parameters` normalisation lives once.
            guard let elementData = try? JSONSerialization.data(withJSONObject: element),
                  let elementStr = String(data: elementData, encoding: .utf8)
            else {
                return nil
            }
            return parseToolCall(elementStr)
        }
    }

    /// Attempts to parse a buffered tool-call body into a `ToolCall`.
    ///
    /// Maps an `MLXLMCommon.ToolCall` — produced when mlx-swift-lm's own
    /// `ToolCallProcessor` parses an *inline* tool call — into our ``ToolCall``.
    ///
    /// For inline tool-call formats (Llama 3's `<|python_tag|>`/JSON, and bare
    /// top-level JSON) mlx-swift-lm's loop handler intercepts and *swallows* the
    /// call text upstream and emits it as a structured `Generation.toolCall`
    /// rather than delivering it as a `.chunk`. The textual `<tool_call>` /
    /// `<|python_tag|>` markers above therefore never see those bytes, so
    /// ``MLXGenerationDriver`` forwards MLX's already-parsed call through this
    /// mapper instead (issue #59 follow-up: the call was being silently dropped).
    ///
    /// The argument dictionary is re-serialised to the JSON string that
    /// ``ToolCall/arguments`` carries, mirroring ``parseToolCall(_:)``. The `id`
    /// uses the same `mlx-<name>-<uuid8>` shape so downstream call-id handling is
    /// uniform across the textual and native paths.
    @_spi(Testing) public static func toolCall(fromNative native: MLXLMCommon.ToolCall) -> ManifoldInference.ToolCall {
        let argumentsString: String
        if let data = try? JSONEncoder().encode(native.function.arguments),
           let str = String(data: data, encoding: .utf8) {
            argumentsString = str
        } else {
            argumentsString = "{}"
        }
        let id = "mlx-\(native.function.name)-\(UUID().uuidString.prefix(8))"
        return ManifoldInference.ToolCall(id: id, toolName: native.function.name, arguments: argumentsString)
    }

    /// Expects `{"name":"…","arguments":{…}}` (Qwen) or `{"name":"…",
    /// "parameters":{…}}` (Llama). The argument object is re-serialised to a
    /// JSON string so `ToolCall.arguments` always carries a valid JSON string
    /// regardless of how the model formatted it. The `try? JSONSerialization`
    /// idiom here is audit-approved (decoding at a trust boundary).
    private static func parseToolCall(_ json: String) -> ManifoldInference.ToolCall? {
        var trimmed = json.trimmingCharacters(in: .whitespacesAndNewlines)
        // Llama sometimes leaves a trailing `<|eot_id|>`/`<|eom_id|>` in the
        // body when the close token that fired was the *other* one, and may
        // emit several space- or semicolon-separated calls. Parse only the
        // first balanced JSON object so a trailing tail never breaks decoding.
        trimmed = firstJSONObject(in: trimmed) ?? trimmed
        guard !trimmed.isEmpty,
              let data = trimmed.data(using: .utf8),
              let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
              let name = obj["name"] as? String, !name.isEmpty
        else {
            return nil
        }

        // The argument payload lives under `arguments` (Qwen) or `parameters`
        // (Llama). It may be a pre-parsed dictionary or (rarely) a JSON string.
        let rawArgs = obj["arguments"] ?? obj["parameters"]
        let argumentsString: String
        if let argsDict = rawArgs as? [String: Any] {
            if let serialised = try? JSONSerialization.data(withJSONObject: argsDict),
               let str = String(data: serialised, encoding: .utf8) {
                argumentsString = str
            } else {
                argumentsString = "{}"
            }
        } else if let argsString = rawArgs as? String {
            argumentsString = argsString
        } else {
            argumentsString = "{}"
        }

        let id = "mlx-\(name)-\(UUID().uuidString.prefix(8))"
        return ManifoldInference.ToolCall(id: id, toolName: name, arguments: argumentsString)
    }

    /// Returns the substring spanning the first balanced top-level `{…}` JSON
    /// object in `text`, or `nil` if none. Brace counting respects JSON string
    /// literals (and their escapes) so a `}` inside a quoted value doesn't
    /// close the object early. Used to peel one Llama call off a body that may
    /// carry trailing special-token text or a second concatenated call.
    private static func firstJSONObject(in text: String) -> String? {
        guard let start = text.firstIndex(of: "{") else { return nil }
        var depth = 0
        var inString = false
        var escaped = false
        var index = start
        while index < text.endIndex {
            let ch = text[index]
            if inString {
                if escaped {
                    escaped = false
                } else if ch == "\\" {
                    escaped = true
                } else if ch == "\"" {
                    inString = false
                }
            } else {
                switch ch {
                case "\"": inString = true
                case "{": depth += 1
                case "}":
                    depth -= 1
                    if depth == 0 {
                        return String(text[start...index])
                    }
                default: break
                }
            }
            index = text.index(after: index)
        }
        return nil
    }
}
