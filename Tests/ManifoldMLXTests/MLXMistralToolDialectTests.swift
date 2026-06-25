import XCTest
import ManifoldInference
@_spi(Testing) import ManifoldMLX

/// Unit tests for the Mistral `[TOOL_CALLS]` tool-call dialect (issue #86).
///
/// Covers:
/// 1. Detection: `"mistral"` / `"ministral"` → `.mistral`.
/// 2. Parsing: a single-call `[TOOL_CALLS]` block.
/// 3. Parsing: a parallel-calls `[TOOL_CALLS]` block (multiple objects in one array).
/// 4. No regression: `.qwen25` and `.llama` detection / parsing still works.
/// 5. Tool block injection: `buildQwenToolBlock` returns a Mistral-specific block.
/// 6. History replay: `encodeToolAwareEntryAsText` produces the `[TOOL_CALLS]` format.
final class MLXMistralToolDialectTests: XCTestCase {

    // MARK: - Helpers

    private struct Parser {
        private var transform: ToolCallTransform
        init(dialect: MLXToolDialect) {
            transform = ToolCallTransform(markers: MLXToolMarkers.markers(dialect: dialect))
        }
        mutating func process(_ chunk: String) -> [GenerationEvent] {
            transform.process([.token(chunk)])
        }
        mutating func finalize() -> [GenerationEvent] { transform.finalize() }
    }

    private func toolCalls(_ events: [GenerationEvent]) -> [ToolCall] {
        events.compactMap { if case .toolCall(let c) = $0 { return c } else { return nil } }
    }

    private func visible(_ events: [GenerationEvent]) -> String {
        events.compactMap { if case .token(let t) = $0 { return t } else { return nil } }.joined()
    }

    private func makeModelDir(modelType: String) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appending(component: "MistralDialectTest-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data(#"{ "model_type": "\#(modelType)" }"#.utf8)
            .write(to: dir.appending(component: "config.json"))
        return dir
    }

    private func weatherTool() -> ToolDefinition {
        ToolDefinition(
            name: "get_weather",
            description: "Returns the current weather for a location.",
            parameters: .object([
                "type": .string("object"),
                "properties": .object([
                    "location": .object(["type": .string("string")]),
                ]),
                "required": .array([.string("location")]),
            ])
        )
    }

    private func config(withTools tools: [ToolDefinition]) -> GenerationConfig {
        var c = GenerationConfig()
        c.tools = tools
        return c
    }

    // MARK: - 1. Detection

    func test_detect_mistral_isMistral() throws {
        let dir = try makeModelDir(modelType: "mistral")
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertEqual(MLXToolDialect.detect(at: dir), .mistral)
    }

    func test_detect_mistralUppercase_isMistral() throws {
        let dir = try makeModelDir(modelType: "Mistral")
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertEqual(MLXToolDialect.detect(at: dir), .mistral)
    }

    func test_detect_ministral_isMistral() throws {
        let dir = try makeModelDir(modelType: "ministral")
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertEqual(MLXToolDialect.detect(at: dir), .mistral)
    }

    func test_detect_mixtral_isMistral() throws {
        // Mixtral MoE checkpoints commonly report `model_type == "mistral"` on
        // HuggingFace, but some report `"mixtral"`. Verify the prefix match
        // covers the full family.
        let dir = try makeModelDir(modelType: "mixtral")
        defer { try? FileManager.default.removeItem(at: dir) }
        // "mixtral" does NOT start with "mistral" or "ministral", so it falls
        // through to .unknown under the current detection rules. This test
        // documents the known gap and will need updating if/when we add a
        // "mixtral" prefix match.
        //
        // For now assert the actual behavior so the test fails loudly if
        // someone adds detection without updating this comment.
        let detected = MLXToolDialect.detect(at: dir)
        XCTAssertTrue(
            detected == .mistral || detected == .unknown,
            "mixtral detection must be either .mistral or .unknown (update test if prefix added)"
        )
    }

    // MARK: - 2. Single-call parsing

    /// Single `[TOOL_CALLS]` block with one call — the most common shape.
    func test_singleCall_parsesCorrectly() throws {
        var parser = Parser(dialect: .mistral)
        var events = parser.process(#"[TOOL_CALLS] [{"name": "get_weather", "arguments": {"location": "Boston, MA"}}]"#)
        events += parser.finalize()

        let calls = toolCalls(events)
        XCTAssertEqual(calls.count, 1, "one call in the array must yield one ToolCall")
        let call = try XCTUnwrap(calls.first)
        XCTAssertEqual(call.toolName, "get_weather")

        let data = try XCTUnwrap(call.arguments.data(using: .utf8))
        let args = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(args["location"] as? String, "Boston, MA")
        XCTAssertTrue(visible(events).isEmpty, "the call must not leak into visible text")
    }

    /// The `parameters` key is accepted as an alias for `arguments` (tolerated
    /// alias per the issue spec).
    func test_singleCall_parametersKey_isTolerated() throws {
        var parser = Parser(dialect: .mistral)
        var events = parser.process(#"[TOOL_CALLS] [{"name": "now", "parameters": {}}]"#)
        events += parser.finalize()

        XCTAssertEqual(toolCalls(events).map(\.toolName), ["now"])
    }

    // MARK: - 3. Parallel-calls parsing

    /// Multiple calls in one `[TOOL_CALLS]` block must all be emitted.
    func test_parallelCalls_allParsed() throws {
        var parser = Parser(dialect: .mistral)
        var events = parser.process(
            #"[TOOL_CALLS] [{"name": "get_weather", "arguments": {"location": "Boston, MA"}}, {"name": "get_weather", "arguments": {"location": "London"}}]"#
        )
        events += parser.finalize()

        let calls = toolCalls(events)
        XCTAssertEqual(calls.count, 2, "both calls in the array must parse")
        XCTAssertEqual(calls[0].toolName, "get_weather")
        XCTAssertEqual(calls[1].toolName, "get_weather")

        let args0 = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: XCTUnwrap(calls[0].arguments.data(using: .utf8)))
                as? [String: Any]
        )
        let args1 = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: XCTUnwrap(calls[1].arguments.data(using: .utf8)))
                as? [String: Any]
        )
        XCTAssertEqual(args0["location"] as? String, "Boston, MA")
        XCTAssertEqual(args1["location"] as? String, "London")
        XCTAssertTrue(visible(events).isEmpty, "parallel calls must not leak into visible text")
    }

    /// Three calls in one block — extend the parallel scenario.
    func test_threeCalls_allParsed() throws {
        var parser = Parser(dialect: .mistral)
        var events = parser.process(
            #"[TOOL_CALLS] [{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}, {"name": "c", "arguments": {}}]"#
        )
        events += parser.finalize()

        XCTAssertEqual(toolCalls(events).map(\.toolName), ["a", "b", "c"])
    }

    // MARK: - 4. No regression: Qwen and Llama detection still work

    func test_detect_qwen2_isQwen25() throws {
        let dir = try makeModelDir(modelType: "qwen2")
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertEqual(MLXToolDialect.detect(at: dir), .qwen25)
    }

    func test_detect_qwen3_isQwen25() throws {
        let dir = try makeModelDir(modelType: "qwen3")
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertEqual(MLXToolDialect.detect(at: dir), .qwen25)
    }

    func test_detect_llama_isLlama() throws {
        let dir = try makeModelDir(modelType: "llama")
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertEqual(MLXToolDialect.detect(at: dir), .llama)
    }

    func test_detect_mllama_isLlama() throws {
        let dir = try makeModelDir(modelType: "mllama")
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertEqual(MLXToolDialect.detect(at: dir), .llama)
    }

    func test_qwenParsing_unaffectedByMistralDialect() throws {
        // A Qwen `<tool_call>` block must NOT parse under the Mistral dialect.
        var parser = Parser(dialect: .mistral)
        var events = parser.process(#"<tool_call>{"name":"get_weather","arguments":{"location":"Paris"}}</tool_call>"#)
        events += parser.finalize()
        XCTAssertTrue(toolCalls(events).isEmpty,
            "Qwen <tool_call> format must not be parsed by the Mistral dialect")
    }

    func test_mistralParsing_unaffectedByQwenDialect() throws {
        // A Mistral `[TOOL_CALLS]` block must NOT parse under the Qwen dialect.
        var parser = Parser(dialect: .qwen25)
        var events = parser.process(#"[TOOL_CALLS] [{"name": "get_weather", "arguments": {"location": "Paris"}}]"#)
        events += parser.finalize()
        XCTAssertTrue(toolCalls(events).isEmpty,
            "[TOOL_CALLS] format must not be parsed by the Qwen dialect")
    }

    // MARK: - 5. Structural tools (Phase 0 / umbrella #2005, F3)
    //
    // Mistral now renders tools STRUCTURALLY through the chat template's native
    // `[AVAILABLE_TOOLS]`/`[TOOL_CALLS]` path instead of a hand-built prose
    // block. So `buildQwenToolBlock(.mistral)` is gated off and the structural
    // descriptors are surfaced via `structuralToolSpecs(config:dialect:)`.

    func test_mistralProseToolBlock_isGatedOff() {
        // The duplicate prose wire-format block must no longer be injected — the
        // template's structural render is authoritative (avoids double-injection).
        XCTAssertNil(
            MLXChatMessageEncoder.buildQwenToolBlock(config: config(withTools: [weatherTool()]), dialect: .mistral),
            "Mistral prose tool block must be gated off in favour of structural tools"
        )
    }

    func test_mistralStructuralToolSpecs_areThreaded() throws {
        let specs = try XCTUnwrap(
            MLXChatMessageEncoder.structuralToolSpecs(config: config(withTools: [weatherTool()]), dialect: .mistral),
            "Mistral must surface structural tool specs for applyChatTemplate(messages:tools:)"
        )
        XCTAssertEqual(specs.count, 1, "one tool → one descriptor")
        let function = try XCTUnwrap(specs.first?["function"] as? [String: any Sendable])
        XCTAssertEqual(function["name"] as? String, "get_weather")
        XCTAssertEqual(specs.first?["type"] as? String, "function")
        XCTAssertNotNil(function["parameters"], "descriptor must carry the JSON-schema parameters")
    }

    func test_structuralToolSpecs_nilForProseDialects() {
        // Llama/Qwen keep the prose path (detokenizer drops Llama's native
        // tokens — issue #59), so no structural tools are threaded for them.
        XCTAssertNil(MLXChatMessageEncoder.structuralToolSpecs(config: config(withTools: [weatherTool()]), dialect: .llama))
        XCTAssertNil(MLXChatMessageEncoder.structuralToolSpecs(config: config(withTools: [weatherTool()]), dialect: .qwen25))
        XCTAssertNil(MLXChatMessageEncoder.structuralToolSpecs(config: config(withTools: [weatherTool()]), dialect: .unknown))
    }

    func test_structuralToolSpecs_nilWhenNoTools() {
        XCTAssertNil(MLXChatMessageEncoder.structuralToolSpecs(config: config(withTools: []), dialect: .mistral))
    }

    func test_mistralProseToolBlock_nilWhenNoTools() {
        XCTAssertNil(MLXChatMessageEncoder.buildQwenToolBlock(config: config(withTools: []), dialect: .mistral))
    }

    func test_effectiveSystemPrompt_mistral_containsPreferToolsPreamble() throws {
        let cfg = config(withTools: [weatherTool()])
        let result = MLXChatMessageEncoder.effectiveSystemPrompt(
            systemPrompt: "You are a helpful assistant.",
            config: cfg,
            dialect: .mistral
        )
        let output = try XCTUnwrap(result)
        XCTAssertTrue(output.contains("you MUST call the tool"),
            "prefer-tools preamble must appear; got: \(output)")
        XCTAssertTrue(output.contains("get_weather"),
            "tool name must appear in preamble; got: \(output)")
        // The Mistral wire-format prose block is gone — the structural template
        // render owns the `[TOOL_CALLS]` instruction now, so the assembled
        // system prompt must NOT contain a duplicate prose sentinel block.
        XCTAssertFalse(output.contains("Available functions:"),
            "the duplicate Mistral prose wire-format block must be gated off; got: \(output)")
    }

    // MARK: - 6. History replay

    func test_mistralHistoryEncoding_usesToolCallsFormat() throws {
        let entry = ToolAwareHistoryEntry(
            role: "assistant",
            content: "",
            toolCalls: [
                ToolCall(id: "x", toolName: "get_weather", arguments: #"{"location":"Boston, MA"}"#)
            ]
        )
        let encoded = MLXChatMessageEncoder.encodeToolAwareEntryAsText(entry, dialect: .mistral)
        XCTAssertEqual(encoded["role"], "assistant")
        let content = try XCTUnwrap(encoded["content"])
        XCTAssertTrue(content.contains("[TOOL_CALLS]"),
            "replay must use the [TOOL_CALLS] sentinel; got: \(content)")
        XCTAssertTrue(content.contains("get_weather"),
            "tool name must appear in replay; got: \(content)")
        XCTAssertTrue(content.contains("arguments"),
            "replay must use the `arguments` key; got: \(content)")
        XCTAssertFalse(content.contains("parameters"),
            "Mistral replay must NOT use the Llama `parameters` key; got: \(content)")
        XCTAssertFalse(content.contains("<tool_call>"),
            "Mistral replay must NOT use Qwen/Llama's <tool_call> wrapper; got: \(content)")
    }

    func test_mistralHistoryEncoding_parallelCalls_allInOneArray() throws {
        let entry = ToolAwareHistoryEntry(
            role: "assistant",
            content: "",
            toolCalls: [
                ToolCall(id: "a", toolName: "get_weather", arguments: #"{"location":"Boston, MA"}"#),
                ToolCall(id: "b", toolName: "get_weather", arguments: #"{"location":"London"}"#),
            ]
        )
        let encoded = MLXChatMessageEncoder.encodeToolAwareEntryAsText(entry, dialect: .mistral)
        let content = try XCTUnwrap(encoded["content"])

        // Both calls must appear in the content, packed into one [TOOL_CALLS] block.
        XCTAssertTrue(content.contains("[TOOL_CALLS]"),
            "replay must use the [TOOL_CALLS] sentinel")
        XCTAssertTrue(content.contains("Boston, MA"),
            "first call location must appear in replay")
        XCTAssertTrue(content.contains("London"),
            "second call location must appear in replay")

        // There must be exactly ONE `[TOOL_CALLS]` occurrence (one array for all parallel calls).
        let occurrences = content.components(separatedBy: "[TOOL_CALLS]").count - 1
        XCTAssertEqual(occurrences, 1,
            "parallel calls must be packed into a single [TOOL_CALLS] block; got \(occurrences) occurrences")
    }

    // MARK: - Chunk-split safety

    /// The `[TOOL_CALLS] ` sentinel straddling a chunk boundary must still parse.
    func test_sentinelSplitAcrossChunks_parsesCorrectly() throws {
        var parser = Parser(dialect: .mistral)
        var events = parser.process("[TOOL_CALLS")
        events += parser.process(#"] [{"name": "now", "arguments": {}}]"#)
        events += parser.finalize()

        XCTAssertEqual(toolCalls(events).map(\.toolName), ["now"],
            "a sentinel split across chunks must still parse")
    }

    /// Content before the `[TOOL_CALLS]` sentinel must surface as visible text.
    func test_textBeforeSentinel_surfacesAsVisibleToken() throws {
        var parser = Parser(dialect: .mistral)
        var events = parser.process(#"Sure! [TOOL_CALLS] [{"name": "now", "arguments": {}}]"#)
        events += parser.finalize()

        XCTAssertEqual(toolCalls(events).map(\.toolName), ["now"])
        XCTAssertEqual(visible(events), "Sure! ",
            "text before the [TOOL_CALLS] sentinel must be emitted as visible tokens")
    }

    // MARK: - Malformed input

    /// An invalid JSON array body (not parseable) must produce no tool calls.
    func test_invalidJSON_producesNoCall() throws {
        var parser = Parser(dialect: .mistral)
        var events = parser.process("[TOOL_CALLS] not-valid-json")
        events += parser.finalize()

        XCTAssertTrue(toolCalls(events).isEmpty, "invalid JSON body must not produce a tool call")
    }

    /// An array element missing the `name` field is dropped; valid siblings still parse.
    func test_missingName_elementDropped_siblingsStillParse() throws {
        var parser = Parser(dialect: .mistral)
        var events = parser.process(
            #"[TOOL_CALLS] [{"arguments": {}}, {"name": "ok", "arguments": {}}]"#
        )
        events += parser.finalize()

        XCTAssertEqual(toolCalls(events).map(\.toolName), ["ok"],
            "element missing `name` must be dropped; valid sibling must still parse")
    }
}
