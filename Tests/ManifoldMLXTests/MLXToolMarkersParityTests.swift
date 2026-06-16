import XCTest
import ManifoldInference
@_spi(Testing) import ManifoldMLX

/// Unit tests for the unified tool-call transform under MLX markers
/// (`MLXToolMarkers.markers()`).
///
/// The former `MLXToolCallParser` had **no** dedicated unit test — its behavior
/// was only covered indirectly through `MLXBackendThinkingTests` end-to-end.
/// These tests port the `<tool_call>` JSON parsing, multi-call reset,
/// prefix-preservation, and malformed-drop behaviors directly against
/// `ToolCallTransform` configured with `MLXToolMarkers.markers()`, so the MLX
/// dialect now has a fast first-class regression net after the #1593 unification.
final class MLXToolMarkersParityTests: XCTestCase {

    /// Shim presenting the old `MLXToolCallParser` API over the unified transform.
    private struct Parser {
        private var transform: ToolCallTransform
        init(surfaceTruncatedToolBody: Bool = false) {
            transform = ToolCallTransform(
                markers: MLXToolMarkers.markers(),
                surfaceTruncatedToolBody: surfaceTruncatedToolBody
            )
        }
        mutating func process(_ chunk: String) -> [GenerationEvent] {
            transform.process([.token(chunk)])
        }
        mutating func finalize() -> [GenerationEvent] {
            transform.finalize()
        }
    }

    private func toolCalls(_ events: [GenerationEvent]) -> [ToolCall] {
        events.compactMap { if case .toolCall(let c) = $0 { return c } else { return nil } }
    }

    private func visible(_ events: [GenerationEvent]) -> String {
        events.compactMap { if case .token(let t) = $0 { return t } else { return nil } }.joined()
    }

    private func parseFailures(_ events: [GenerationEvent]) -> [String] {
        events.compactMap { if case .toolCallParseFailed(let body) = $0 { return body } else { return nil } }
    }

    private func truncations(_ events: [GenerationEvent]) -> [String] {
        events.compactMap { if case .toolCallTruncated(let body) = $0 { return body } else { return nil } }
    }

    /// Extracts the exact `arguments` JSON string from the single emitted call.
    private func soleArguments(_ events: [GenerationEvent]) throws -> String {
        let calls = toolCalls(events)
        XCTAssertEqual(calls.count, 1)
        return try XCTUnwrap(calls.first).arguments
    }

    // MARK: - Single JSON tool call

    func test_singleJSONCall_emitsToolCall() throws {
        var parser = Parser()
        let events = parser.process("<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}</tool_call>")
        let calls = toolCalls(events)
        XCTAssertEqual(calls.count, 1)
        let call = try XCTUnwrap(calls.first)
        XCTAssertEqual(call.toolName, "get_weather")
        let data = try XCTUnwrap(call.arguments.data(using: .utf8))
        let args = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(args["city"] as? String, "Paris")
    }

    /// `arguments` omitted entirely → fall back to the empty-object string.
    func test_singleJSONCall_argumentsOmitted_fallsBackToEmptyObject() throws {
        var parser = Parser()
        let events = parser.process("<tool_call>{\"name\":\"noargs\"}</tool_call>")
        XCTAssertEqual(toolCalls(events).map(\.toolName), ["noargs"])
        XCTAssertEqual(try soleArguments(events), "{}")
    }

    /// `arguments` supplied as a JSON *string* is passed through verbatim.
    func test_singleJSONCall_argumentsAsJSONString_passedThrough() throws {
        var parser = Parser()
        let events = parser.process(
            "<tool_call>{\"name\":\"strargs\",\"arguments\":\"{\\\"k\\\":1}\"}</tool_call>"
        )
        XCTAssertEqual(toolCalls(events).map(\.toolName), ["strargs"])
        XCTAssertEqual(try soleArguments(events), "{\"k\":1}")
    }

    /// `arguments` as a non-object scalar (number) → empty-object fallback.
    func test_singleJSONCall_argumentsAsScalar_fallsBackToEmptyObject() throws {
        var parser = Parser()
        let events = parser.process("<tool_call>{\"name\":\"scalar\",\"arguments\":42}</tool_call>")
        XCTAssertEqual(toolCalls(events).map(\.toolName), ["scalar"])
        XCTAssertEqual(try soleArguments(events), "{}")
    }

    // MARK: - Prefix preservation

    func test_visibleTextBeforeOpenTag_emittedAsToken() throws {
        var parser = Parser()
        let events = parser.process("Here you go: <tool_call>{\"name\":\"f\"}</tool_call>")
        XCTAssertEqual(visible(events), "Here you go: ")
        let calls = toolCalls(events)
        XCTAssertEqual(calls.map(\.toolName), ["f"])
        // The call here omits `arguments`, exercising the empty-object fallback.
        XCTAssertEqual(calls.first?.arguments, "{}")
    }

    // MARK: - Multi-call reset

    func test_multipleCalls_resetBetweenAndEmitBoth() {
        var parser = Parser()
        let events = parser.process(
            "<tool_call>{\"name\":\"a\"}</tool_call>mid<tool_call>{\"name\":\"b\"}</tool_call>"
        )
        XCTAssertEqual(toolCalls(events).map(\.toolName), ["a", "b"])
        XCTAssertEqual(visible(events), "mid")
    }

    // MARK: - Malformed bodies dropped

    func test_invalidJSON_isDropped() {
        var parser = Parser()
        let events = parser.process("<tool_call>not json</tool_call>")
        XCTAssertTrue(toolCalls(events).isEmpty)
        // The raw body must not leak as visible text...
        XCTAssertTrue(visible(events).isEmpty, "rejected body must not surface as visible text")
        // ...it surfaces as a non-fatal diagnostic carrying the body instead (#1857).
        XCTAssertEqual(parseFailures(events), ["not json"])
    }

    func test_missingNameField_isDropped() {
        var parser = Parser()
        let events = parser.process("<tool_call>{\"arguments\":{}}</tool_call>")
        XCTAssertTrue(toolCalls(events).isEmpty)
        XCTAssertTrue(visible(events).isEmpty, "rejected body must not surface as visible text")
        XCTAssertEqual(parseFailures(events), ["{\"arguments\":{}}"])
    }

    // MARK: - Chunk safety

    func test_openTagSplitAcrossChunks_parsesCorrectly() {
        var parser = Parser()
        var events = parser.process("pre<tool")
        events += parser.process("_call>{\"name\":\"split\"}</tool_call>")
        events += parser.finalize()
        XCTAssertEqual(toolCalls(events).map(\.toolName), ["split"])
        XCTAssertEqual(visible(events), "pre")
    }

    func test_singleByteChunks_parseCorrectly() {
        var parser = Parser()
        let input = "x<tool_call>{\"name\":\"bytewise\"}</tool_call>y"
        var events: [GenerationEvent] = []
        for ch in input {
            events += parser.process(String(ch))
        }
        events += parser.finalize()
        XCTAssertEqual(toolCalls(events).map(\.toolName), ["bytewise"])
        XCTAssertEqual(visible(events), "xy")
    }

    // MARK: - finalize discards dangling block

    func test_finalize_discardsUnterminatedBlock() {
        var parser = Parser()
        var events = parser.process("ok<tool_call>{\"name\":\"f")
        events += parser.finalize()
        XCTAssertTrue(toolCalls(events).isEmpty)
        XCTAssertEqual(visible(events), "ok")
        // Default (flag off) discards silently — no truncation diagnostic.
        XCTAssertTrue(truncations(events).isEmpty)
    }

    /// With `surfaceTruncatedToolBody` enabled, finalizing an unterminated
    /// `<tool_call>` surfaces the partial body as a non-fatal
    /// `.toolCallTruncated(rawBody:)` diagnostic instead of dropping it (#1858).
    func test_finalize_surfacesTruncatedBlockWhenFlagEnabled() {
        var parser = Parser(surfaceTruncatedToolBody: true)
        var events = parser.process("ok<tool_call>{\"name\":\"f")
        events += parser.finalize()
        XCTAssertTrue(toolCalls(events).isEmpty)
        XCTAssertEqual(visible(events), "ok")
        XCTAssertEqual(truncations(events), ["{\"name\":\"f"])
    }
}
