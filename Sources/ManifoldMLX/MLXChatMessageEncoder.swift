import CoreImage
import Foundation
import MLXLMCommon
import ManifoldInference

/// Encodes Manifold chat history into the message shapes accepted by MLX.
// @_spi(Testing): published only for backend test targets (companion-package split, #1749).
@_spi(Testing) public enum MLXChatMessageEncoder {
    /// Assembles the prepared chat-message inputs for the MLX container's two
    /// `prepare(...)` overloads. Returns both shapes because the caller picks
    /// `prepare(chat:)` for vision history (non-nil first element) and
    /// `prepare(messages:)` otherwise.
    public static func buildChatMessages(
        prompt: String,
        effectiveSystemPrompt: String?,
        conversationHistory: [(role: String, content: String)],
        toolAwareHistory: [ToolAwareHistoryEntry]?,
        structuredHistory: [StructuredMessage]?,
        dialect: MLXToolDialect
    ) throws -> (chatMessages: [Chat.Message]?, messages: [[String: String]]) {
        let chatMessages: [Chat.Message]? =
            if let structuredHistory, !structuredHistory.isEmpty {
                if let toolAwareHistory, !toolAwareHistory.isEmpty {
                    try toolAwareChatMessages(
                        structuredHistory: structuredHistory,
                        toolAwareHistory: toolAwareHistory,
                        systemPrompt: effectiveSystemPrompt,
                        dialect: dialect
                    )
                } else {
                    try plainChatMessages(
                        history: structuredHistory,
                        systemPrompt: effectiveSystemPrompt
                    )
                }
            } else {
                nil
            }

        var msgs: [[String: String]] = []
        if let sp = effectiveSystemPrompt, !sp.isEmpty {
            msgs.append(["role": "system", "content": sp])
        }
        if let toolHistory = toolAwareHistory, !toolHistory.isEmpty {
            for entry in toolHistory {
                msgs.append(encodeToolAwareEntryAsText(entry, dialect: dialect))
            }
        } else if !conversationHistory.isEmpty {
            for msg in conversationHistory {
                msgs.append(["role": msg.role, "content": msg.content])
            }
        } else {
            msgs.append(["role": "user", "content": prompt])
        }
        return (normalizeChatMessages(chatMessages), normalizeSystemMessages(msgs))
    }

    /// Guarantees the assembled `[[String: String]]` message array contains at
    /// most one `system` message and that it sits at index 0.
    ///
    /// Several model families (Qwen 3.x, Mistral v0.3, …) ship Jinja chat
    /// templates that hard-assert *"System message must be at the beginning"*
    /// and that no second system turn appears mid-conversation. Our message
    /// assembly can produce a leading `system` turn from
    /// `effectiveSystemPrompt` **and** a second `system` turn carried inside the
    /// conversation / tool-aware history (the orchestrator replays the full
    /// transcript, system turn included). That misplaced or duplicate system
    /// message crashes the template at render time, before any generation —
    /// see issue #57.
    ///
    /// This collapses every `system` turn into a single message at index 0,
    /// preserving relative order of the system fragments and dropping the now
    /// promoted entries from their original positions. Non-system turns keep
    /// their order. Qwen 2.5 (which tolerates the looser shape) is unaffected
    /// because a single leading system message is valid for it too.
    static func normalizeSystemMessages(_ messages: [[String: String]]) -> [[String: String]] {
        let systemFragments = messages
            .filter { $0["role"] == "system" }
            .compactMap { $0["content"] }
            .filter { !$0.isEmpty }
        let nonSystem = messages.filter { $0["role"] != "system" }

        guard !systemFragments.isEmpty else { return nonSystem }

        let merged = systemFragments.joined(separator: "\n\n")
        return [["role": "system", "content": merged]] + nonSystem
    }

    /// Folds a leading `system` message into the first `user` turn, for chat
    /// templates that reject a standalone `system` role.
    ///
    /// Mistral v0.3's Jinja template has no `system` branch and enforces strict
    /// user/assistant alternation, so a `system`-first array raises
    /// *"Conversation roles must alternate user/assistant/…"* at render time,
    /// before any generation. Rather than hardcode a model-type list, the
    /// caller (``MLXPromptCacheCoordinator``) attempts the real template first
    /// and only invokes this fallback when the render throws — so it adapts to
    /// any system-hostile template, not just Mistral's.
    ///
    /// Mistral v0.3 has no `<<SYS>>` markers (that's a Llama-2 idiom), so the
    /// system text is prepended to the first user message as a plain paragraph.
    /// When there is no user turn to fold into, the system content is re-tagged
    /// as a leading `user` turn so the instruction is still delivered rather
    /// than dropped. A no-op (returns the input unchanged) when there is no
    /// `system` message — the caller relies on this to detect "nothing to retry"
    /// and rethrow the original error.
    @_spi(Testing) public static func foldSystemIntoFirstUser(
        _ messages: [[String: String]]
    ) -> [[String: String]] {
        guard let systemIndex = messages.firstIndex(where: { $0["role"] == "system" }) else {
            return messages
        }
        let systemContent = messages[systemIndex]["content"] ?? ""
        var result = messages
        result.remove(at: systemIndex)

        guard let userIndex = result.firstIndex(where: { $0["role"] == "user" }) else {
            result.insert(["role": "user", "content": systemContent], at: 0)
            return result
        }
        let existing = result[userIndex]["content"] ?? ""
        let merged = systemContent.isEmpty ? existing : "\(systemContent)\n\n\(existing)"
        result[userIndex] = ["role": "user", "content": merged]
        return result
    }

    /// `Chat.Message` analogue of ``normalizeSystemMessages(_:)``: folds every
    /// `.system` turn into a single leading message so the vision / structured
    /// history path obeys the same "system message must be first" contract.
    /// Image attachments only ride on user/assistant/tool turns, so collapsing
    /// system turns never drops image inputs.
    static func normalizeChatMessages(_ messages: [Chat.Message]?) -> [Chat.Message]? {
        guard let messages else { return nil }
        let systemFragments = messages
            .filter { $0.role == .system }
            .map(\.content)
            .filter { !$0.isEmpty }
        let nonSystem = messages.filter { $0.role != .system }

        guard !systemFragments.isEmpty else { return nonSystem }

        let merged = systemFragments.joined(separator: "\n\n")
        return [.system(merged)] + nonSystem
    }

    /// Composes the full effective system prompt for an MLX generation turn:
    /// the `preferTools` imperative preamble (when tools are present) joined
    /// with the application system prompt and the dialect's wire-format tool
    /// block.
    ///
    /// This is the single-call replacement for the inline assembly that lived
    /// in `MLXGenerationDriver`. Pulling it here lets unit tests verify the
    /// preamble injection without spinning up a live model.
    ///
    /// - Parameters:
    ///   - systemPrompt: The app-level system prompt (may be nil or empty).
    ///   - config: The generation config whose `tools` array drives preamble and
    ///     tool-block injection.
    ///   - dialect: Active tool wire-format dialect.
    /// - Returns: The assembled system prompt, or `nil` when the result would be
    ///   empty (no system prompt, no tools, no preamble).
    @_spi(Testing) public static func effectiveSystemPrompt(
        systemPrompt: String?,
        config: GenerationConfig,
        dialect: MLXToolDialect
    ) -> String? {
        let preamble = ToolSystemPromptBuilder.preferTools(for: config.tools)
        let toolBlock = buildQwenToolBlock(config: config, dialect: dialect)

        var parts: [String] = []
        if !preamble.isEmpty { parts.append(preamble) }
        if let sp = systemPrompt, !sp.isEmpty { parts.append(sp) }
        if let tb = toolBlock { parts.append(tb) }

        guard !parts.isEmpty else { return nil }
        return parts.joined(separator: "\n\n")
    }

    /// Returns the tool-definition block to append to the system prompt for the
    /// active dialect, or `nil` when the dialect doesn't inject tools this way
    /// or the caller supplied no tools.
    ///
    /// - `.qwen25` emits a `<tools>…</tools>` JSON block with the
    ///   `<tool_call>`-XML response instruction.
    /// - `.llama` emits a Llama 3.x function block listing the tools as JSON and
    ///   instructing a `<tool_call>{"name":…,"parameters":…}</tool_call>`
    ///   response (issue #59 — without it Llama never produces a parseable
    ///   call). See ``llamaToolBlock(config:)`` for why the textual wrapper is
    ///   used instead of Llama's native `<|python_tag|>`.
    /// - `.mistral` emits a Mistral function block listing the tools as JSON
    ///   and instructing the `[TOOL_CALLS] [{"name":…,"arguments":{…}}]`
    ///   response format (issue #86).
    /// - `.unknown` returns `nil`.
    ///
    /// The name retains the historical `buildQwenToolBlock` spelling for source
    /// compatibility with existing call sites and tests; it now dispatches on
    /// `dialect`.
    public static func buildQwenToolBlock(
        config: GenerationConfig,
        dialect: MLXToolDialect
    ) -> String? {
        guard !config.tools.isEmpty else { return nil }
        switch dialect {
        case .qwen25:
            return qwenToolBlock(config: config)
        case .llama:
            return llamaToolBlock(config: config)
        case .mistral:
            return mistralToolBlock(config: config)
        case .unknown:
            return nil
        }
    }

    /// Serialises `tools` to a pretty-printed JSON array of `{name, description,
    /// parameters}` function descriptors. Shared by both dialect blocks.
    private static func toolFunctionsJSON(_ config: GenerationConfig, wrapInFunctionType: Bool) -> String {
        let toolObjects: [[String: Any]] = config.tools.map { tool -> [String: Any] in
            var function_: [String: Any] = [
                "name": tool.name,
                "description": tool.description,
            ]
            if let paramsData = try? JSONEncoder().encode(tool.parameters),
               let paramsObj = try? JSONSerialization.jsonObject(with: paramsData) {
                function_["parameters"] = paramsObj
            } else {
                function_["parameters"] = ["type": "object", "properties": [String: Any]()]
            }
            return wrapInFunctionType ? ["type": "function", "function": function_] : function_
        }
        if let data = try? JSONSerialization.data(withJSONObject: toolObjects, options: [.prettyPrinted]),
           let str = String(data: data, encoding: .utf8) {
            return str
        }
        return "[]"
    }

    private static func qwenToolBlock(config: GenerationConfig) -> String {
        let toolsJSON = toolFunctionsJSON(config, wrapInFunctionType: true)
        return "\n\n# Tools\n\nYou may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. Here are the available tools:\n\n<tools>\n\(toolsJSON)\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>"
    }

    /// Llama 3.x function-calling system block.
    ///
    /// Lists the available functions as JSON, then instructs the model to emit
    /// a single `<tool_call>{"name":…,"parameters":…}</tool_call>` block.
    ///
    /// Why the textual wrapper instead of Llama's native format: Llama's own
    /// tool path uses the `<|python_tag|>` special token followed by JSON,
    /// terminated by `<|eom_id|>` / `<|eot_id|>` — all special tokens the MLX
    /// streaming detokenizer drops, so the call body never reaches our
    /// `ToolCallTransform` as visible text and is silently lost (issue #59). The
    /// `<tool_call>…</tool_call>` delimiters DO survive detokenisation, and
    /// Llama-3.2 follows the instruction to use them on the arithmetic / time /
    /// multi-tool scenarios. The body keeps Llama's `parameters` key;
    /// `MLXToolMarkers` accepts both `parameters` and `arguments`.
    private static func llamaToolBlock(config: GenerationConfig) -> String {
        let toolsJSON = toolFunctionsJSON(config, wrapInFunctionType: false)
        return """


        You have access to the following functions. To call a function, emit \
        EXACTLY one block of the form:
        <tool_call>{"name": <function-name>, "parameters": <arguments-json-object>}</tool_call>
        Emit the literal `<tool_call>` and `</tool_call>` tags around a single \
        JSON object. Do not narrate the call, do not wrap it in code fences, and \
        do not invent your own tags. Use the exact function name. Only call a \
        function when one can answer the request; otherwise reply normally.

        Available functions:
        \(toolsJSON)
        """
    }

    /// Mistral tool-calling system block (issue #86).
    ///
    /// Lists the available functions as JSON and instructs the model to emit
    /// `[TOOL_CALLS] [{"name":…,"arguments":{…}}]`. One JSON array per
    /// generation turn; multiple parallel calls appear as sibling objects in
    /// the same array.
    ///
    /// Note on system-message placement: Mistral's Jinja chat template enforces
    /// strict user/assistant alternation and rejects a standalone `system` role.
    /// Rather than handling this here, `MLXPromptCacheCoordinator` attempts the
    /// real template first and falls back to `foldSystemIntoFirstUser` when it
    /// raises — so the mistral tool block rides into the first user turn as plain
    /// text and the system-hostility is handled transparently at the render layer.
    private static func mistralToolBlock(config: GenerationConfig) -> String {
        let toolsJSON = toolFunctionsJSON(config, wrapInFunctionType: false)
        return """


        You have access to the following functions. To call one or more \
        functions, emit EXACTLY one block of the form:
        [TOOL_CALLS] [{"name": <function-name>, "arguments": <args-json-object>}]
        You may include multiple calls in the JSON array for parallel execution. \
        Do not narrate the call, do not wrap it in code fences, and do not invent \
        your own format. Use the exact function name. Only call a function when \
        one can answer the request; otherwise reply normally.

        Available functions:
        \(toolsJSON)
        """
    }

    /// Encodes a ``ToolAwareHistoryEntry`` into a plain `[String: String]` message
    /// compatible with MLX chat-template preparation.
    ///
    /// For the Qwen 2.5 dialect:
    /// - Assistant entries with `toolCalls` have the calls serialised as
    ///   `<tool_call>{"name":…, "arguments":…}</tool_call>` appended to (or
    ///   replacing) the textual content.
    /// - Tool-role entries (carrying a ``ToolResult``) are represented as
    ///   `role: "tool"` with the result content. The MLX chat template for
    ///   Qwen maps the `tool` role to an `<tool_response>` block internally.
    ///
    /// For the `.unknown` dialect (and plain text turns) the entry collapses to
    /// a simple `{role, content}` pair.
    // `@_spi(Testing) public` so backend test targets can assert the
    // dialect-specific replay encoding directly (issue #59 Llama coverage).
    @_spi(Testing) public static func encodeToolAwareEntryAsText(
        _ entry: ToolAwareHistoryEntry,
        dialect: MLXToolDialect
    ) -> [String: String] {
        // For dialects without a recognised tool wire-format (`.unknown`) or
        // plain turns, fall back to the bare shape.
        guard dialect == .qwen25 || dialect == .llama || dialect == .mistral else {
            return ["role": entry.role, "content": entry.content]
        }

        if let calls = entry.toolCalls, !calls.isEmpty {
            // Assistant turn that triggered tool calls: encode calls as text in
            // the dialect the model itself emits, so a replayed transcript is
            // self-consistent with what generation produced.
            var parts: [String] = []
            if !entry.content.isEmpty {
                parts.append(entry.content)
            }

            if dialect == .mistral {
                // Mistral packs ALL calls in one JSON array prefixed by the
                // `[TOOL_CALLS]` sentinel: `[TOOL_CALLS] [{…},{…}]`.
                // Build the array from all calls in this entry at once.
                let callObjects: [[String: Any]] = calls.compactMap { call in
                    let argsValue: Any
                    if let data = call.arguments.data(using: .utf8),
                       let parsed = try? JSONSerialization.jsonObject(with: data) {
                        argsValue = parsed
                    } else {
                        argsValue = [String: Any]()
                    }
                    return ["name": call.toolName, "arguments": argsValue]
                }
                if let data = try? JSONSerialization.data(withJSONObject: callObjects),
                   let jsonStr = String(data: data, encoding: .utf8) {
                    parts.append("[TOOL_CALLS] \(jsonStr)")
                }
            } else {
                for call in calls {
                    let argsValue: Any
                    if let data = call.arguments.data(using: .utf8),
                       let parsed = try? JSONSerialization.jsonObject(with: data) {
                        argsValue = parsed
                    } else {
                        argsValue = [String: Any]()
                    }
                    switch dialect {
                    case .qwen25:
                        let callObj: [String: Any] = ["name": call.toolName, "arguments": argsValue]
                        if let data = try? JSONSerialization.data(withJSONObject: callObj),
                           let jsonStr = String(data: data, encoding: .utf8) {
                            parts.append("<tool_call>\n\(jsonStr)\n</tool_call>")
                        }
                    case .llama:
                        // Llama uses the `parameters` key; we replay the call in the
                        // same textual `<tool_call>` wrapper we instruct it to emit
                        // (Llama's native `<|python_tag|>`/`<|eom_id|>` terminator is
                        // a special token that doesn't round-trip through plain text).
                        let callObj: [String: Any] = ["name": call.toolName, "parameters": argsValue]
                        if let data = try? JSONSerialization.data(withJSONObject: callObj),
                           let jsonStr = String(data: data, encoding: .utf8) {
                            parts.append("<tool_call>\n\(jsonStr)\n</tool_call>")
                        }
                    case .mistral, .unknown:
                        break
                    }
                }
            }
            return ["role": "assistant", "content": parts.joined(separator: "\n")]
        }

        // Tool result turn: pass role and content as-is. Both the Qwen and
        // Llama tokenizer templates handle `role: "tool"` natively.
        return ["role": entry.role, "content": entry.content]
    }

    private static func chatRole(for role: String) -> Chat.Message.Role {
        switch role {
        case "assistant": .assistant
        case "system": .system
        case "tool": .tool
        default: .user
        }
    }

    private static func userInputImage(from data: Data, mimeType: String) throws -> UserInput.Image {
        guard let image = CIImage(data: data) else {
            throw InferenceError.inferenceFailure(
                "Unsupported image attachment format (\(mimeType))."
            )
        }
        return .ciImage(image)
    }

    private static func imageInputs(from parts: [MessagePart]) throws -> [UserInput.Image] {
        try parts.compactMap { part in
            guard case let .image(data, mimeType, _) = part else { return nil }
            return try userInputImage(from: data, mimeType: mimeType)
        }
    }

    private static func plainChatMessages(
        history: [StructuredMessage],
        systemPrompt: String?
    ) throws -> [Chat.Message]? {
        let containsImages = history.contains { message in
            message.parts.contains { part in
                if case .image = part { return true }
                return false
            }
        }
        guard containsImages else { return nil }

        var messages: [Chat.Message] = []
        if let systemPrompt, !systemPrompt.isEmpty {
            messages.append(.system(systemPrompt))
        }
        for message in history {
            messages.append(
                Chat.Message(
                    role: chatRole(for: message.role),
                    content: message.textContent,
                    images: try imageInputs(from: message.parts)
                )
            )
        }
        return messages
    }

    private static func toolAwareChatMessages(
        structuredHistory: [StructuredMessage],
        toolAwareHistory: [ToolAwareHistoryEntry],
        systemPrompt: String?,
        dialect: MLXToolDialect
    ) throws -> [Chat.Message]? {
        guard structuredHistory.contains(where: { message in
            message.parts.contains { part in
                if case .image = part { return true }
                return false
            }
        }) else {
            return nil
        }

        var messages: [Chat.Message] = []
        if let systemPrompt, !systemPrompt.isEmpty {
            messages.append(.system(systemPrompt))
        }

        let structuredImageParts = try structuredHistory.map { try imageInputs(from: $0.parts) }
        for (index, entry) in toolAwareHistory.enumerated() {
            let encoded = encodeToolAwareEntryAsText(entry, dialect: dialect)
            let role = encoded["role"] ?? entry.role
            let content = encoded["content"] ?? entry.content
            let images = index < structuredImageParts.count ? structuredImageParts[index] : []
            messages.append(
                Chat.Message(
                    role: chatRole(for: role),
                    content: content,
                    images: images
                )
            )
        }

        if toolAwareHistory.count < structuredHistory.count {
            for (offset, structuredMessage) in structuredHistory.dropFirst(toolAwareHistory.count).enumerated() {
                let images = structuredImageParts[toolAwareHistory.count + offset]
                messages.append(
                    Chat.Message(
                        role: chatRole(for: structuredMessage.role),
                        content: structuredMessage.textContent,
                        images: images
                    )
                )
            }
        }

        return messages
    }
}
