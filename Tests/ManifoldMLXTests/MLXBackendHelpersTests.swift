import XCTest
import ManifoldInference
import ManifoldMLX
@_spi(Testing) import ManifoldMLX

/// Unit tests for the chat-message encoding helpers extracted from `MLXBackend.generate` (#1113).
///
/// These run in CI without Apple Silicon — every helper is value-in / value-out
/// and never touches the MLX runtime or Metal stack.
final class MLXChatMessageEncoderTests: XCTestCase {

    // MARK: - buildQwenToolBlock

    func test_buildQwenToolBlock_returnsNilWhenNoTools() {
        let block = MLXChatMessageEncoder.buildQwenToolBlock(
            config: GenerationConfig(),
            dialect: .qwen25
        )
        XCTAssertNil(block)
    }

    func test_buildQwenToolBlock_returnsNilForUnknownDialect() {
        var config = GenerationConfig()
        config.tools = [
            ToolDefinition(name: "echo", description: "Echo input", parameters: .object([:]))
        ]
        let block = MLXChatMessageEncoder.buildQwenToolBlock(config: config, dialect: .unknown)
        XCTAssertNil(block)
    }

    func test_buildQwenToolBlock_serialisesToolsForQwenDialect() {
        var config = GenerationConfig()
        config.tools = [
            ToolDefinition(
                name: "get_weather",
                description: "Look up current weather",
                parameters: .object([
                    "type": .string("object"),
                    "properties": .object([
                        "city": .object(["type": .string("string")])
                    ]),
                    "required": .array([.string("city")])
                ])
            )
        ]
        let block = MLXChatMessageEncoder.buildQwenToolBlock(config: config, dialect: .qwen25)
        XCTAssertNotNil(block)
        XCTAssertTrue(block!.contains("<tools>"))
        XCTAssertTrue(block!.contains("</tools>"))
        XCTAssertTrue(block!.contains("get_weather"))
        XCTAssertTrue(block!.contains("Look up current weather"))
        // The Qwen contract expects the model to emit <tool_call>…</tool_call>.
        XCTAssertTrue(block!.contains("<tool_call>"))
        XCTAssertTrue(block!.contains("</tool_call>"))
    }

    // MARK: - Single tool-injection invariant (ManifoldKit 0.57 / #1985)

    /// Pins the no-double-injection contract after the MK 0.57 harness reshape
    /// (#1985 moved `ScenarioRunner` onto `InferenceService`).
    ///
    /// MK's `GenerationQueue.dispatchToBackend` only renders the chat template +
    /// injects tools through `PromptRenderer` / `ToolSystemPromptBuilder` when the
    /// backend reports `capabilities.requiresPromptTemplate == true`. `MLXBackend`
    /// reports `false` (and `rendersFullPrompt == true`), so the service takes the
    /// non-template branch: it forwards the *bare* last user message as `prompt`
    /// and the *raw* `systemPrompt` (no tool augmentation), then installs structured
    /// history. The MLX driver alone owns input-side templating + tool injection via
    /// `buildQwenToolBlock`. This test reproduces that exact branch's inputs and
    /// asserts the tool block is injected **exactly once** — guarding against a
    /// regression where MK starts double-injecting (or MLX starts honouring a
    /// pre-rendered tool prompt) and the model sees its tools twice.
    func test_singleToolInjection_serviceForwardsBarePromptAndRawSystem() throws {
        var config = GenerationConfig()
        config.tools = [
            ToolDefinition(
                name: "get_weather",
                description: "Look up current weather",
                parameters: .object(["type": .string("object")])
            )
        ]

        // Mirror MLXGenerationDriver.generate: effectiveSystemPrompt = raw system
        // prompt + the single tool block the driver appends for the dialect.
        let toolBlock = MLXChatMessageEncoder.buildQwenToolBlock(config: config, dialect: .qwen25)
        let rawSystemPrompt = "You are a helpful assistant."
        let effectiveSystemPrompt = (rawSystemPrompt) + (toolBlock ?? "")

        let (_, messages) = try MLXChatMessageEncoder.buildChatMessages(
            // The service passes the bare last user message here (non-template
            // branch), NOT a pre-rendered prompt carrying tools.
            prompt: "What's the weather in Paris?",
            effectiveSystemPrompt: effectiveSystemPrompt,
            conversationHistory: [],
            toolAwareHistory: nil,
            structuredHistory: nil,
            dialect: .qwen25
        )

        // Exactly one system message, sitting first.
        let systemMessages = messages.filter { $0["role"] == "system" }
        XCTAssertEqual(systemMessages.count, 1, "tools must not produce a second system turn")
        XCTAssertEqual(messages.first?["role"], "system")

        // The <tools> block — the tool-definition injection — appears exactly once
        // across the entire assembled message set. A double-injection regression
        // would surface here as 2.
        let assembled = messages.map { ($0["content"] ?? "") }.joined(separator: "\n")
        let toolsBlockCount = assembled.components(separatedBy: "<tools>").count - 1
        XCTAssertEqual(toolsBlockCount, 1, "tool definitions must be injected exactly once")
        XCTAssertEqual(
            assembled.components(separatedBy: "get_weather").count - 1, 1,
            "the tool name must not appear twice (no double-injection)"
        )

        // The bare user prompt is carried verbatim and is NOT itself tool-laden.
        let userMessage = messages.first { $0["role"] == "user" }
        XCTAssertEqual(userMessage?["content"], "What's the weather in Paris?")
        XCTAssertFalse(
            (userMessage?["content"] ?? "").contains("<tools>"),
            "the user turn must stay a bare prompt — tools live only in the system turn"
        )
    }

    // MARK: - buildChatMessages

    func test_buildChatMessages_fallsBackToBarePromptWhenNoHistory() throws {
        let (chatMessages, messages) = try MLXChatMessageEncoder.buildChatMessages(
            prompt: "hi there",
            effectiveSystemPrompt: nil,
            conversationHistory: [],
            toolAwareHistory: nil,
            structuredHistory: nil,
            dialect: .unknown
        )
        XCTAssertNil(chatMessages, "no images → no Chat.Message path")
        XCTAssertEqual(messages.count, 1)
        XCTAssertEqual(messages[0]["role"], "user")
        XCTAssertEqual(messages[0]["content"], "hi there")
    }

    func test_buildChatMessages_prependsSystemPromptWhenSet() throws {
        let (_, messages) = try MLXChatMessageEncoder.buildChatMessages(
            prompt: "hi",
            effectiveSystemPrompt: "You are helpful.",
            conversationHistory: [],
            toolAwareHistory: nil,
            structuredHistory: nil,
            dialect: .unknown
        )
        XCTAssertEqual(messages.count, 2)
        XCTAssertEqual(messages[0]["role"], "system")
        XCTAssertEqual(messages[0]["content"], "You are helpful.")
        XCTAssertEqual(messages[1]["role"], "user")
        XCTAssertEqual(messages[1]["content"], "hi")
    }

    func test_buildChatMessages_replaysConversationHistoryWhenSet() throws {
        let (_, messages) = try MLXChatMessageEncoder.buildChatMessages(
            prompt: "ignored when history present",
            effectiveSystemPrompt: nil,
            conversationHistory: [
                (role: "user", content: "first"),
                (role: "assistant", content: "ack"),
                (role: "user", content: "second")
            ],
            toolAwareHistory: nil,
            structuredHistory: nil,
            dialect: .unknown
        )
        XCTAssertEqual(messages.count, 3)
        XCTAssertEqual(messages.map { $0["content"] }, ["first", "ack", "second"])
        XCTAssertEqual(messages.map { $0["role"] }, ["user", "assistant", "user"])
    }

    func test_buildChatMessages_toolAwareHistorySupersedesPlain() throws {
        let toolEntry = ToolAwareHistoryEntry(
            role: "user",
            content: "what's the weather?"
        )
        let (_, messages) = try MLXChatMessageEncoder.buildChatMessages(
            prompt: "ignored",
            effectiveSystemPrompt: nil,
            conversationHistory: [(role: "user", content: "plain history that should NOT appear")],
            toolAwareHistory: [toolEntry],
            structuredHistory: nil,
            dialect: .qwen25
        )
        XCTAssertEqual(messages.count, 1)
        XCTAssertEqual(messages[0]["role"], "user")
        XCTAssertEqual(messages[0]["content"], "what's the weather?")
    }

    func test_buildChatMessages_toolAwareAssistantEntryEncodesToolCalls() throws {
        let toolEntry = ToolAwareHistoryEntry(
            role: "assistant",
            content: "",
            toolCalls: [
                ToolCall(id: "c1", toolName: "get_weather", arguments: "{\"city\":\"Paris\"}")
            ]
        )
        let (_, messages) = try MLXChatMessageEncoder.buildChatMessages(
            prompt: "ignored",
            effectiveSystemPrompt: nil,
            conversationHistory: [],
            toolAwareHistory: [toolEntry],
            structuredHistory: nil,
            dialect: .qwen25
        )
        XCTAssertEqual(messages.count, 1)
        XCTAssertEqual(messages[0]["role"], "assistant")
        let content = try XCTUnwrap(messages[0]["content"])
        XCTAssertTrue(content.contains("<tool_call>"), "Qwen tool calls render as <tool_call> blocks")
        XCTAssertTrue(content.contains("</tool_call>"))
        XCTAssertTrue(content.contains("get_weather"), "tool name must appear in the encoded call")
        // The arguments must round-trip into a parsed object, not the raw string.
        let open = try XCTUnwrap(content.range(of: "{"))
        let close = try XCTUnwrap(content.range(of: "}", options: .backwards))
        let jsonSlice = String(content[open.lowerBound...close.lowerBound])
        let obj = try XCTUnwrap(
            try JSONSerialization.jsonObject(
                with: XCTUnwrap(jsonSlice.data(using: .utf8))
            ) as? [String: Any]
        )
        XCTAssertEqual(obj["name"] as? String, "get_weather")
        let args = try XCTUnwrap(obj["arguments"] as? [String: Any])
        XCTAssertEqual(args["city"] as? String, "Paris")
    }

    func test_buildChatMessages_toolAwareMalformedArgumentsFallBackToEmptyObject() throws {
        let toolEntry = ToolAwareHistoryEntry(
            role: "assistant",
            content: "",
            toolCalls: [
                ToolCall(id: "c1", toolName: "broken", arguments: "{not valid json")
            ]
        )
        let (_, messages) = try MLXChatMessageEncoder.buildChatMessages(
            prompt: "ignored",
            effectiveSystemPrompt: nil,
            conversationHistory: [],
            toolAwareHistory: [toolEntry],
            structuredHistory: nil,
            dialect: .qwen25
        )
        XCTAssertEqual(messages.count, 1)
        let content = try XCTUnwrap(messages[0]["content"])
        // The call is preserved (not dropped) with an empty args object.
        XCTAssertTrue(content.contains("<tool_call>"))
        XCTAssertTrue(content.contains("broken"))
        let open = try XCTUnwrap(content.range(of: "{"))
        let close = try XCTUnwrap(content.range(of: "}", options: .backwards))
        let jsonSlice = String(content[open.lowerBound...close.lowerBound])
        let obj = try XCTUnwrap(
            try JSONSerialization.jsonObject(
                with: XCTUnwrap(jsonSlice.data(using: .utf8))
            ) as? [String: Any]
        )
        XCTAssertEqual(obj["name"] as? String, "broken")
        let args = try XCTUnwrap(obj["arguments"] as? [String: Any])
        XCTAssertTrue(args.isEmpty, "malformed argument JSON must collapse to an empty object, not drop the call")
    }

    func test_buildChatMessages_returnsNilChatMessagesWhenStructuredHistoryHasNoImages() throws {
        let history = [
            StructuredMessage(role: "user", content: "text only"),
            StructuredMessage(role: "assistant", content: "still text")
        ]
        let (chatMessages, _) = try MLXChatMessageEncoder.buildChatMessages(
            prompt: "ignored",
            effectiveSystemPrompt: nil,
            conversationHistory: [],
            toolAwareHistory: nil,
            structuredHistory: history,
            dialect: .unknown
        )
        XCTAssertNil(chatMessages, "vision path stays off when no .image parts present")
    }

    func test_buildChatMessages_returnsChatMessagesWhenStructuredHistoryHasImages() throws {
        // 1×1 PNG (smallest valid) so CIImage(data:) succeeds.
        let pngBytes: [UInt8] = [
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
            0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4,
            0x89, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x44, 0x41,
            0x54, 0x78, 0x9C, 0x63, 0x00, 0x01, 0x00, 0x00,
            0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00,
            0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE,
            0x42, 0x60, 0x82
        ]
        let pngData = Data(pngBytes)
        let history = [
            StructuredMessage(role: "user", parts: [
                .text("describe this"),
                .image(data: pngData, mimeType: "image/png")
            ])
        ]
        let (chatMessages, _) = try MLXChatMessageEncoder.buildChatMessages(
            prompt: "ignored",
            effectiveSystemPrompt: "system!",
            conversationHistory: [],
            toolAwareHistory: nil,
            structuredHistory: history,
            dialect: .unknown
        )
        let chat = try XCTUnwrap(chatMessages)
        XCTAssertEqual(chat.count, 2, "system + user")
        // Roles: system prompt first, then the structured user turn.
        XCTAssertEqual(chat.map(\.role), [.system, .user])
        // The system message carries the supplied prompt.
        XCTAssertEqual(chat[0].content, "system!")
        // The user message text is the .text part of the structured message...
        XCTAssertEqual(chat[1].content, "describe this")
        // ...and the decoded image lands in its images array.
        XCTAssertEqual(chat[1].images.count, 1, "the single .image part must produce one Chat image")
    }

    // MARK: - System-message normalization (issue #57)

    /// Regression for #57: with a system prompt set **and** a `system` turn
    /// replayed inside the tool-aware history, the assembled array must still
    /// expose exactly one system message at index 0. Qwen 3.x / Mistral v0.3
    /// chat templates hard-crash ("System message must be at the beginning.")
    /// otherwise — and a misplaced/duplicate system turn aborts before any
    /// generation runs.
    func test_buildChatMessages_systemMessageStaysFirstWithToolsAndHistory() throws {
        let history = [
            // The orchestrator replays the full transcript, system turn first.
            ToolAwareHistoryEntry(role: "system", content: "history system turn"),
            ToolAwareHistoryEntry(role: "user", content: "what's 2+2?"),
            ToolAwareHistoryEntry(
                role: "assistant",
                content: "",
                toolCalls: [ToolCall(id: "c1", toolName: "calc", arguments: "{\"expr\":\"2+2\"}")]
            ),
            ToolAwareHistoryEntry(role: "tool", content: "4", toolCallId: "c1")
        ]
        let (_, messages) = try MLXChatMessageEncoder.buildChatMessages(
            prompt: "ignored",
            effectiveSystemPrompt: "You are helpful.\n\n<tools>[…]</tools>",
            conversationHistory: [],
            toolAwareHistory: history,
            structuredHistory: nil,
            dialect: .qwen25
        )
        // Exactly one system message, and it is first.
        let systemIndices = messages.indices.filter { messages[$0]["role"] == "system" }
        XCTAssertEqual(systemIndices, [0], "there must be exactly one system message, at index 0")
        // Both system fragments are merged into the leading message.
        let systemContent = try XCTUnwrap(messages[0]["content"])
        XCTAssertTrue(systemContent.contains("You are helpful."))
        XCTAssertTrue(systemContent.contains("<tools>"))
        XCTAssertTrue(systemContent.contains("history system turn"))
        // Remaining turns keep their order and roles.
        XCTAssertEqual(
            messages.dropFirst().map { $0["role"] },
            ["user", "assistant", "tool"]
        )
    }

    /// Plain conversation history carrying a stray mid-array `system` turn is
    /// also folded up to index 0 (covers the non-tool path).
    func test_buildChatMessages_foldsMidArraySystemTurnFromPlainHistory() throws {
        let (_, messages) = try MLXChatMessageEncoder.buildChatMessages(
            prompt: "ignored",
            effectiveSystemPrompt: "primary system",
            conversationHistory: [
                (role: "user", content: "hi"),
                (role: "system", content: "stray system"),
                (role: "assistant", content: "hello")
            ],
            toolAwareHistory: nil,
            structuredHistory: nil,
            dialect: .unknown
        )
        XCTAssertEqual(messages.first?["role"], "system")
        XCTAssertEqual(
            messages.filter { $0["role"] == "system" }.count,
            1,
            "stray system turn must be merged, not left mid-array"
        )
        let systemContent = try XCTUnwrap(messages[0]["content"])
        XCTAssertTrue(systemContent.contains("primary system"))
        XCTAssertTrue(systemContent.contains("stray system"))
        XCTAssertEqual(messages.dropFirst().map { $0["role"] }, ["user", "assistant"])
    }
}

/// Unit tests for the small thinking-marker resolution helper retained on `MLXBackend`.
final class MLXBackendThinkingMarkerResolutionTests: XCTestCase {
    func test_resolveThinkingMarkers_returnsNilWhenMaxThinkingTokensIsZero() {
        var config = GenerationConfig()
        config.maxThinkingTokens = 0
        // Even with both override and auto-detected available, zero budget wins (#597).
        let resolved = MLXBackend.resolveThinkingMarkers(
            config: config,
            autoDetected: .qwen3
        )
        XCTAssertNil(resolved)
    }

    func test_resolveThinkingMarkers_prefersConfigOverrideWhenSet() {
        var config = GenerationConfig()
        let override = ThinkingMarkers(open: "<a>", close: "</a>")
        config.thinkingMarkers = override
        let resolved = MLXBackend.resolveThinkingMarkers(
            config: config,
            autoDetected: .qwen3
        )
        XCTAssertEqual(resolved, override)
    }

    func test_resolveThinkingMarkers_fallsBackToAutoDetectedWhenConfigUnset() {
        let resolved = MLXBackend.resolveThinkingMarkers(
            config: GenerationConfig(),
            autoDetected: .qwen3
        )
        XCTAssertEqual(resolved, .qwen3)
    }

    func test_resolveThinkingMarkers_returnsNilWhenNeitherSourceProvided() {
        let resolved = MLXBackend.resolveThinkingMarkers(
            config: GenerationConfig(),
            autoDetected: nil
        )
        XCTAssertNil(resolved)
    }

}
