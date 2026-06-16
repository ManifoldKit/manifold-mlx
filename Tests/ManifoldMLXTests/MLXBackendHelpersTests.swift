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
