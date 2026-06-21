import XCTest
import ManifoldInference
@_spi(Testing) import ManifoldMLX

/// Regression tests for Llama 3.x tool calling (issue #59).
///
/// The overnight tool-calling campaign found `llama-3.2-3b` (MLX) NEVER
/// dispatched a tool: `MLXToolDialect.detect` mapped `model_type == "llama"`
/// to `.unknown`, so no tool grammar was injected and the model narrated calls
/// as prose (`calc(7823 * 41) = …`) or invented its own `<calc>…</calc>`
/// wrapper — neither parseable. These tests lock in:
///
/// 1. Detection: `"llama"` / `"mllama"` → `.llama`.
/// 2. Injection: a tool block is appended for `.llama` instructing the
///    `<tool_call>{"name":…,"parameters":…}</tool_call>` shape.
/// 3. Parsing: `MLXToolMarkers.markers(dialect: .llama)` parses the exact body
///    the real model emits (captured from a live re-run), including the Llama
///    `parameters` key, the `<|python_tag|>` prefix variant, and chunk
///    splitting — without regressing Qwen `<tool_call>` / `arguments` parsing.
final class MLXLlamaToolDialectTests: XCTestCase {

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

    /// The exact, verbatim tool-call body `llama-3.2-3b` emitted in the live
    /// re-run against the calc scenario (see PR #59 fixture). Note `parameters`
    /// (Llama), not `arguments` (Qwen), and the model's own key ordering.
    private static let realLlamaCalcCall =
        #"<tool_call>{"name": "calc", "parameters": {"op": "*", "b": 41, "a": 7823}}</tool_call>"#

    // MARK: - Detection

    private func makeModelDir(modelType: String) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appending(component: "LlamaDialectTest-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data(#"{ "model_type": "\#(modelType)" }"#.utf8)
            .write(to: dir.appending(component: "config.json"))
        return dir
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

    func test_detect_llamaUppercase_isLlama() throws {
        let dir = try makeModelDir(modelType: "LLaMA")
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertEqual(MLXToolDialect.detect(at: dir), .llama)
    }

    // MARK: - Parsing the real captured format

    func test_realLlamaCalcCall_dispatches() throws {
        var parser = Parser(dialect: .llama)
        var events = parser.process(Self.realLlamaCalcCall)
        events += parser.finalize()
        let calls = toolCalls(events)
        XCTAssertEqual(calls.count, 1, "the captured Llama calc call must parse to exactly one tool call")
        let call = try XCTUnwrap(calls.first)
        XCTAssertEqual(call.toolName, "calc")
        let args = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: XCTUnwrap(call.arguments.data(using: .utf8))) as? [String: Any]
        )
        XCTAssertEqual(args["op"] as? String, "*")
        XCTAssertEqual(args["a"] as? Int, 7823)
        XCTAssertEqual(args["b"] as? Int, 41)
        XCTAssertTrue(visible(events).isEmpty, "the call must not leak into visible text")
    }

    /// `parameters` (Llama key) is honoured exactly like Qwen's `arguments`.
    func test_parametersKey_isParsedAsArguments() throws {
        var parser = Parser(dialect: .llama)
        let events = parser.process(#"<tool_call>{"name":"read_file","parameters":{"path":"a.txt"}}</tool_call>"#)
        let call = try XCTUnwrap(toolCalls(events).first)
        XCTAssertEqual(call.toolName, "read_file")
        XCTAssertEqual(call.arguments, #"{"path":"a.txt"}"#)
    }

    /// The `<|python_tag|>` prefix variant, terminated by a visible `<|eom_id|>`
    /// close, also parses (fallback path).
    func test_pythonTagWithEomClose_parses() throws {
        var parser = Parser(dialect: .llama)
        let events = parser.process(#"<|python_tag|>{"name":"now","parameters":{}}<|eom_id|>"#)
        XCTAssertEqual(toolCalls(events).map(\.toolName), ["now"])
    }

    /// A trailing special-token tail after the JSON object is peeled off rather
    /// than breaking decoding.
    func test_trailingTailAfterJSON_isIgnored() throws {
        var parser = Parser(dialect: .llama)
        let events = parser.process(#"<tool_call>{"name":"calc","parameters":{"a":1,"op":"+","b":2}} <|eot_id|></tool_call>"#)
        XCTAssertEqual(toolCalls(events).map(\.toolName), ["calc"])
    }

    /// Chunk-split open tag still parses (streaming safety).
    func test_splitAcrossChunks_parses() {
        var parser = Parser(dialect: .llama)
        var events = parser.process(#"pre<tool"#)
        events += parser.process(#"_call>{"name":"split","parameters":{}}</tool_call>"#)
        events += parser.finalize()
        XCTAssertEqual(toolCalls(events).map(\.toolName), ["split"])
        XCTAssertEqual(visible(events), "pre")
    }

    // MARK: - No regression of Qwen parsing

    func test_qwenArgumentsKey_stillParsesUnderQwenDialect() throws {
        var parser = Parser(dialect: .qwen25)
        let events = parser.process(#"<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>"#)
        let call = try XCTUnwrap(toolCalls(events).first)
        XCTAssertEqual(call.toolName, "get_weather")
        XCTAssertEqual(call.arguments, #"{"city":"Paris"}"#)
    }

    func test_unknownDialect_hasNoMarkers() {
        XCTAssertTrue(MLXToolMarkers.markers(dialect: .unknown).isEmpty)
    }

    // MARK: - Injection

    private func calcTool() -> ToolDefinition {
        ToolDefinition(
            name: "calc",
            description: "Evaluates a single arithmetic expression.",
            parameters: .object([
                "type": .string("object"),
                "properties": .object([
                    "a": .object(["type": .string("number")]),
                    "op": .object(["type": .string("string")]),
                    "b": .object(["type": .string("number")]),
                ]),
                "required": .array([.string("a"), .string("op"), .string("b")]),
            ])
        )
    }

    private func config(withTools tools: [ToolDefinition]) -> GenerationConfig {
        var c = GenerationConfig()
        c.tools = tools
        return c
    }

    func test_llamaToolBlock_isInjectedWithToolGrammar() throws {
        let block = try XCTUnwrap(
            MLXChatMessageEncoder.buildQwenToolBlock(config: config(withTools: [calcTool()]), dialect: .llama),
            "a non-empty tool block must be injected for the Llama dialect (issue #59 — was nil, so no grammar reached the model)"
        )
        XCTAssertTrue(block.contains("<tool_call>"), "must instruct the textual <tool_call> wrapper")
        XCTAssertTrue(block.contains("parameters"), "must instruct Llama's `parameters` key")
        XCTAssertTrue(block.contains("\"calc\"") || block.contains("calc"), "must list the available function")
    }

    func test_noToolBlock_whenNoTools() {
        XCTAssertNil(MLXChatMessageEncoder.buildQwenToolBlock(config: config(withTools: []), dialect: .llama))
    }

    func test_unknownDialect_getsNoToolBlock() {
        XCTAssertNil(MLXChatMessageEncoder.buildQwenToolBlock(config: config(withTools: [calcTool()]), dialect: .unknown))
    }

    // MARK: - Tool-aware history replay

    func test_llamaHistoryEncoding_usesToolCallWrapperWithParameters() throws {
        let entry = ToolAwareHistoryEntry(
            role: "assistant",
            content: "",
            toolCalls: [ToolCall(id: "x", toolName: "calc", arguments: #"{"a":1,"op":"+","b":2}"#)]
        )
        let encoded = MLXChatMessageEncoder.encodeToolAwareEntryAsText(entry, dialect: .llama)
        XCTAssertEqual(encoded["role"], "assistant")
        let content = try XCTUnwrap(encoded["content"])
        XCTAssertTrue(content.contains("<tool_call>"))
        XCTAssertTrue(content.contains("\"parameters\""), "replayed Llama calls use the `parameters` key")
        XCTAssertFalse(content.contains("\"arguments\""))
    }

    // MARK: - python_tag recovery (issue #59 tail)
    //
    // The MLX streaming detokeniser drops Llama's `<|eom_id|>` / `<|eot_id|>`
    // close tokens (they are stop tokens, so the generate loop breaks before
    // detokenising them). `<|python_tag|>` survives, so a native tool call
    // arrives as an *unterminated* `<|python_tag|>{json…}` block and was
    // discarded. `MLXLlamaPythonTagNormalizer` injects a synthetic `<|eom_id|>`
    // close at stream end so the body parses. These tests run the normaliser in
    // front of `ToolCallTransform` exactly as the driver does.

    /// Drives `MLXLlamaPythonTagNormalizer` → `ToolCallTransform`, mirroring the
    /// MLX driver: each chunk is normalised, fed to the transform, then the
    /// normaliser's finalize tail is fed in before the transform finalizes.
    private struct NormalizingParser {
        private var normalizer: MLXLlamaPythonTagNormalizer
        private var transform: ToolCallTransform
        init(dialect: MLXToolDialect) {
            normalizer = MLXLlamaPythonTagNormalizer(dialect: dialect)
            transform = ToolCallTransform(markers: MLXToolMarkers.markers(dialect: dialect))
        }
        mutating func process(_ chunk: String) -> [GenerationEvent] {
            transform.process([.token(normalizer.process(chunk))])
        }
        mutating func finalize() -> [GenerationEvent] {
            var events = transform.process([.token(normalizer.finalize())])
            events += transform.finalize()
            return events
        }
    }

    /// The whole point of issue #59's tail: a `<|python_tag|>` body whose close
    /// token MLX dropped still dispatches once the normaliser injects the close.
    func test_pythonTag_withoutVisibleClose_dispatchesAfterNormalisation() throws {
        var parser = NormalizingParser(dialect: .llama)
        // No `<|eom_id|>` / `<|eot_id|>` — exactly what reaches us after MLX
        // breaks the loop on the (dropped) stop token.
        var events = parser.process(#"<|python_tag|>{"name":"read_file","parameters":{"path":"a.txt"}}"#)
        events += parser.finalize()
        let calls = toolCalls(events)
        XCTAssertEqual(calls.map(\.toolName), ["read_file"],
                       "an unterminated python_tag block must dispatch after the synthetic close is injected")
        XCTAssertEqual(calls.first?.arguments, #"{"path":"a.txt"}"#)
        XCTAssertTrue(visible(events).isEmpty, "the recovered call must not leak into visible text")
    }

    /// The list_dir scenario shape, streamed across several chunks (including a
    /// chunk boundary inside the `<|python_tag|>` open tag).
    func test_pythonTag_listDir_splitChunks_dispatches() throws {
        var parser = NormalizingParser(dialect: .llama)
        var events = parser.process("<|python")
        events += parser.process(#"_tag|>{"name":"list_dir","#)
        events += parser.process(#""parameters":{"path":"."}}"#)
        events += parser.finalize()
        XCTAssertEqual(toolCalls(events).map(\.toolName), ["list_dir"])
    }

    /// A python_tag block the model *did* close with a visible `<|eom_id|>` is
    /// parsed without a doubled close (the normaliser sees the close and does
    /// not inject another).
    func test_pythonTag_withVisibleEomClose_dispatchesOnce() throws {
        var parser = NormalizingParser(dialect: .llama)
        var events = parser.process(#"<|python_tag|>{"name":"now","parameters":{}}<|eom_id|>"#)
        events += parser.finalize()
        XCTAssertEqual(toolCalls(events).map(\.toolName), ["now"],
                       "a visibly-closed python_tag must dispatch exactly once")
    }

    /// An empty `<tool_call></tool_call>` (the failure symptom: special tokens
    /// stripped leaving an empty body) must NOT produce a spurious tool call.
    func test_emptyToolCall_producesNoSpuriousCall() throws {
        var parser = NormalizingParser(dialect: .llama)
        var events = parser.process("<tool_call></tool_call>")
        events += parser.finalize()
        XCTAssertTrue(toolCalls(events).isEmpty,
                      "an empty tool_call body must not be mistaken for a call")
    }

    /// Plain prose with no python tag is passed through byte-for-byte and yields
    /// no tool call and no injected close.
    func test_plainProse_isUnchanged_andNoCall() throws {
        var parser = NormalizingParser(dialect: .llama)
        let prose = "I cannot read files, but here is some Python: open('a.txt')."
        var events = parser.process(prose)
        events += parser.finalize()
        XCTAssertTrue(toolCalls(events).isEmpty)
        XCTAssertEqual(visible(events), prose, "prose must survive normalisation verbatim")
    }

    /// The textual `<tool_call>` wrapper (the steered Llama path that already
    /// worked) is unaffected: the normaliser never opens a python-tag block, so
    /// no synthetic close is injected and the call parses exactly once.
    func test_textualToolCallWrapper_stillDispatchesUnderNormaliser() throws {
        var parser = NormalizingParser(dialect: .llama)
        var events = parser.process(Self.realLlamaCalcCall)
        events += parser.finalize()
        XCTAssertEqual(toolCalls(events).map(\.toolName), ["calc"])
    }

    // MARK: - Streaming-phase regression: tool-call-only via normalizer tail

    /// Regression test for the bug where a Llama `<|python_tag|>` tool call with
    /// NO preceding visible text caused `generationStream.setPhase(.streaming)` to
    /// be skipped entirely. The main generation loop swallows all the content into
    /// the tool parser, leaving `isFirstToken == true`; then `llamaNormalizer.finalize()`
    /// fires the synthetic close that produces the `.toolCall` event — but the old
    /// code never called `setPhase(.streaming)` in that tail path.
    ///
    /// This test drives `MLXBackend` end-to-end via `MockMLXModelContainer` (no
    /// Metal/GPU required), injecting a single chunk that is entirely the open
    /// python_tag with no text before it. The normalizer tail emits the synthetic
    /// close and session produces the `.toolCall`. We assert:
    ///   (a) a `.toolCall` event is emitted naming "f"
    ///   (b) `stream.phase` reached `.streaming` (not stuck at `.preparing`)
    func test_pythonTag_toolCallOnly_setsStreamingPhase() async throws {
        let mock = MockMLXModelContainer()
        // Single chunk: entire output is a bare python_tag tool call body.
        // No text precedes it, so `isFirstToken` stays `true` through the main
        // loop — the `.toolCall` event can only come from the normalizer tail.
        mock.generationsToYield = [
            .chunk(#"<|python_tag|>{"name":"f","parameters":{}}"#),
        ]

        let backend = MLXBackend()
        // Inject with .llama dialect so `MLXLlamaPythonTagNormalizer` activates.
        backend._inject(mock, dialect: .llama)

        var config = GenerationConfig()
        config.tools = [
            ToolDefinition(name: "f", description: "no-op tool", parameters: .object([:]))
        ]

        let stream = try backend.generate(
            prompt: "call f",
            systemPrompt: nil,
            config: config
        )

        // Verify that setPhase(.streaming) is called in the normalizer-tail path.
        //
        // Architecture: MLXGenerationDriver is @MainActor. Both setPhase(.streaming)
        // (in the tail loop) and setPhase(.done) (in MLXBackend after driver returns)
        // execute in the same main-actor run before the consumer task ever resumes.
        // Observing the intermediate .streaming state from outside that turn via
        // @Observable phase polling is architecturally impossible — both mutations
        // happen before the consumer is ever resumed. Instead, we use the
        // @_spi(Testing) synchronous callback MLXGenerationDriver._streamingPhaseSetInTailHook,
        // which fires on the main actor immediately after setPhase(.streaming) in the
        // tail path — before any further mutations.
        //
        // With the fix:    hook fires → streamingWasSet = true
        // Without the fix: hook never fires → streamingWasSet = false
        nonisolated(unsafe) var streamingWasSet = false
        MLXGenerationDriver._streamingPhaseSetInTailHook = {
            streamingWasSet = true
        }
        defer { MLXGenerationDriver._streamingPhaseSetInTailHook = nil }

        var collectedEvents: [GenerationEvent] = []
        for try await event in stream.events {
            collectedEvents.append(event)
        }

        // (a) A .toolCall event naming "f" must be emitted.
        let toolCalls = collectedEvents.compactMap { ev -> ToolCall? in
            if case .toolCall(let c) = ev { return c } else { return nil }
        }
        XCTAssertEqual(toolCalls.map(\.toolName), ["f"],
            "A python_tag tool call with no preceding text must dispatch via the normalizer tail")

        // (b) The driver must have called setPhase(.streaming) in the tail path.
        // Without the fix, the tail loop has no phase-transition logic, so
        // _streamingPhaseSetInTailHook never fires and streamingWasSet stays false.
        XCTAssertTrue(streamingWasSet,
            "setPhase(.streaming) must be called in the normalizer-tail path when a tool call is the first (and only) event. Without the fix, the stream goes directly from .connecting to .done, breaking callers that gate on stream.phase == .streaming.")
    }

    // MARK: - Normaliser unit behaviour (no transform)

    func test_normaliser_qwenDialect_isIdentity() {
        var n = MLXLlamaPythonTagNormalizer(dialect: .qwen25)
        XCTAssertEqual(n.process(#"<|python_tag|>{"name":"x"}"#), #"<|python_tag|>{"name":"x"}"#)
        XCTAssertEqual(n.finalize(), "", "non-Llama dialects never inject a close")
    }

    func test_normaliser_injectsCloseOnlyWhenBlockOpen() {
        var open = MLXLlamaPythonTagNormalizer(dialect: .llama)
        _ = open.process(#"<|python_tag|>{"name":"x","parameters":{}}"#)
        XCTAssertEqual(open.finalize(), "<|eom_id|>", "an open python_tag gets a synthetic close")

        var closed = MLXLlamaPythonTagNormalizer(dialect: .llama)
        _ = closed.process(#"<|python_tag|>{"name":"x"}<|eom_id|>"#)
        XCTAssertEqual(closed.finalize(), "", "a visibly-closed python_tag gets no extra close")

        var none = MLXLlamaPythonTagNormalizer(dialect: .llama)
        _ = none.process("just some prose")
        XCTAssertEqual(none.finalize(), "", "no python_tag means no injected close")
    }

    // MARK: - prefer-tools preamble injection (issue #71)
    //
    // The MLX backend sets `rendersFullPrompt: true`, so core's
    // `GenerationQueue.toolAugmentedSystemPrompt` — which folds the
    // `ToolSystemPromptBuilder.preferTools` imperative steering — never fires
    // for MLX turns. Without it, Llama-3.2-3B opens `<tool_call>` and then emits
    // an empty body for under-described tools (e.g. list_dir), so the call is
    // never dispatched. The fix adds the preamble MLX-side via
    // `MLXChatMessageEncoder.effectiveSystemPrompt`.

    /// With a tool registered, the assembled system prompt must contain:
    ///   1. The imperative "you MUST call the tool" steering from the standard
    ///      `ToolSystemPromptBuilder.preferTools` preamble.
    ///   2. The tool name from the preamble's enumeration.
    ///   3. The original app system prompt (passed through unchanged).
    ///   4. The wire-format `<tool_call>` marker from the Llama tool block.
    func test_effectiveSystemPrompt_containsPreferToolsPreamble_forTools() throws {
        let cfg = config(withTools: [calcTool()])
        let appSystemPrompt = "You are a helpful assistant."
        let result = MLXChatMessageEncoder.effectiveSystemPrompt(
            systemPrompt: appSystemPrompt,
            config: cfg,
            dialect: .llama
        )
        let output = try XCTUnwrap(result, "effectiveSystemPrompt must not be nil when tools are present")
        XCTAssertTrue(
            output.contains("you MUST call the tool"),
            "standard prefer-tools preamble imperative must appear; got: \(output)"
        )
        XCTAssertTrue(
            output.contains("calc"),
            "tool name must be listed in the preamble; got: \(output)"
        )
        XCTAssertTrue(
            output.contains(appSystemPrompt),
            "app system prompt must be retained in the output; got: \(output)"
        )
        XCTAssertTrue(
            output.contains("<tool_call>"),
            "Llama wire-format tool block marker must still be present; got: \(output)"
        )
    }

    /// With no tools, `effectiveSystemPrompt` must return the bare app system
    /// prompt — no preamble injected, tool block omitted.
    func test_effectiveSystemPrompt_noPreamble_whenNoTools() {
        let cfg = config(withTools: [])
        let appSystemPrompt = "You are a helpful assistant."
        let result = MLXChatMessageEncoder.effectiveSystemPrompt(
            systemPrompt: appSystemPrompt,
            config: cfg,
            dialect: .llama
        )
        XCTAssertEqual(
            result,
            appSystemPrompt,
            "with no tools, the output must equal the bare app system prompt"
        )
    }
}
