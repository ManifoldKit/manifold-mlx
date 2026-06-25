import XCTest
import ManifoldInference
@_spi(Testing) import ManifoldMLX

/// Deterministic, model-free parse-back tests for the Mistral `[TOOL_CALLS]`
/// output normalizer (umbrella #2005, F3).
///
/// Each test feeds a VERBATIM malformed emission observed in the Phase-0 soak
/// (2026-06-25) — the shapes MLX's streaming detokenizer produces after dropping
/// the JSON quote/space structural tokens — through the normalizer and then the
/// real `MLXToolMarkers.mistral` parser, asserting the correct `ToolCall`.
///
/// The mechanism mirrors issue #59 (`MLXLlamaPythonTagNormalizer`): the model
/// emits the right tokens, but the detokenizer mangles them on the way out.
final class MLXMistralToolCallNormalizerTests: XCTestCase {

    // MARK: - Pipeline harness

    /// Mirrors the driver pipeline: raw stream → normalizer → `ToolCallTransform`
    /// (built from `MLXToolMarkers.mistral`) → events. The normalizer buffers the
    /// whole stream and repairs at `finalize()`, so the harness feeds the raw
    /// text as one chunk, takes the normalizer's finalize output, and runs that
    /// through the transform exactly as `MLXGenerationDriver` does.
    private func parseBack(_ raw: String) -> [ToolCall] {
        var normalizer = MLXMistralToolCallNormalizer(dialect: .mistral)
        // process() buffers and returns "" for Mistral; finalize() emits.
        _ = normalizer.process(raw)
        let repaired = normalizer.finalize()

        var transform = ToolCallTransform(markers: MLXToolMarkers.markers(dialect: .mistral))
        var events = transform.process([.token(repaired)])
        events += transform.finalize()
        return events.compactMap { if case .toolCall(let c) = $0 { return c } else { return nil } }
    }

    private func args(_ call: ToolCall) throws -> [String: Any] {
        let data = try XCTUnwrap(call.arguments.data(using: .utf8))
        return try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    // MARK: - Fixture 01: `now`, no args

    /// `[TOOL_CALLS][{function:now,arguments:{}]`
    /// Dropped: quotes, sentinel trailing space, the closing `}` of the object.
    func test_fixture01_now_parsesToNowCallWithEmptyArgs() throws {
        let calls = parseBack("[TOOL_CALLS][{function:now,arguments:{}]")
        XCTAssertEqual(calls.map(\.toolName), ["now"],
            "fixture 01 must repair to a single `now` call")
        try XCTAssertEqual(args(XCTUnwrap(calls.first)).count, 0,
            "`now` takes no arguments")
    }

    // MARK: - Fixture 02: `calc`, numeric + single-quoted string args

    /// `325087\n\n[{function:calc,arguments:{a:7823,b:41,op:'*'}}]`
    /// Dropped: quotes; the `[TOOL_CALLS]` sentinel itself is gone (only the bare
    /// array survives, with stray leading prose). String value is single-quoted.
    func test_fixture02_calc_parsesWithNumericAndStringArgs() throws {
        // Note: this fixture has NO `[TOOL_CALLS]` sentinel — only the bare array.
        // The normalizer keys off the sentinel, so to exercise repair we prepend
        // the sentinel the driver's marker requires; the verbatim payload after it
        // is the observed bare array. (See junk test for the no-sentinel case.)
        let calls = parseBack("[TOOL_CALLS]325087\n\n[{function:calc,arguments:{a:7823,b:41,op:'*'}}]")
        XCTAssertEqual(calls.map(\.toolName), ["calc"],
            "fixture 02 must repair to a single `calc` call")
        let a = try args(XCTUnwrap(calls.first))
        XCTAssertEqual(a["a"] as? Int, 7823)
        XCTAssertEqual(a["b"] as? Int, 41)
        XCTAssertEqual(a["op"] as? String, "*",
            "single-quoted string value must be re-quoted to a JSON string")
    }

    // MARK: - Fixture 03: `read_file`, function:→name: mapping + empty key

    /// `[TOOL_CALLS]_[{function:,name:read_file,arguments:{path:'example.txt'},type:function]_`
    /// Dropped: quotes; stray `_` artifacts; an empty `function:` key precedes the
    /// real `name:read_file`; trailing `]_` and a missing object `}`.
    func test_fixture03_readFile_mapsNameAndPreservesArgKeys() throws {
        let calls = parseBack("[TOOL_CALLS]_[{function:,name:read_file,arguments:{path:'example.txt'},type:function]_")
        XCTAssertEqual(calls.map(\.toolName), ["read_file"],
            "fixture 03 must repair to a single `read_file` call (name: wins over empty function:)")
        let a = try args(XCTUnwrap(calls.first))
        XCTAssertEqual(a["path"] as? String, "example.txt",
            "the real argument key `path` must be preserved (only the call-name key is remapped)")
    }

    // MARK: - Pass-through: valid JSON unchanged

    /// A canonical, already-valid `[TOOL_CALLS]` block must pass through the
    /// normalizer byte-for-byte and parse normally.
    func test_validJSON_passesThroughUnchanged() throws {
        let valid = #"[TOOL_CALLS] [{"name": "get_weather", "arguments": {"location": "Boston, MA"}}]"#

        var normalizer = MLXMistralToolCallNormalizer(dialect: .mistral)
        _ = normalizer.process(valid)
        let out = normalizer.finalize()
        XCTAssertEqual(out, valid,
            "valid JSON must pass through the normalizer byte-for-byte")

        let calls = parseBack(valid)
        XCTAssertEqual(calls.map(\.toolName), ["get_weather"])
        try XCTAssertEqual(args(XCTUnwrap(calls.first))["location"] as? String, "Boston, MA")
    }

    /// A valid parallel-call array also passes through untouched.
    func test_validParallelJSON_passesThroughUnchanged() throws {
        let valid = #"[TOOL_CALLS] [{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}]"#
        var normalizer = MLXMistralToolCallNormalizer(dialect: .mistral)
        _ = normalizer.process(valid)
        XCTAssertEqual(normalizer.finalize(), valid)
        XCTAssertEqual(parseBack(valid).map(\.toolName), ["a", "b"])
    }

    // MARK: - Junk stays unparsed (no fabricated calls)

    /// Genuinely junk text with the sentinel but no reconstructable call must
    /// yield ZERO tool calls — the normalizer must not fabricate one.
    func test_junkAfterSentinel_staysUnparsed() {
        XCTAssertTrue(parseBack("[TOOL_CALLS] the quick brown fox jumped").isEmpty,
            "non-JSON prose after the sentinel must not fabricate a call")
    }

    /// A `[TOOL_CALLS]` array of objects that carry no `name` must NOT parse.
    func test_namelessObjects_stayUnparsed() {
        XCTAssertTrue(parseBack("[TOOL_CALLS][{arguments:{x:1}},{foo:bar}]").isEmpty,
            "objects with no call-name key must not fabricate a call")
    }

    /// Plain assistant prose with no sentinel at all is left untouched and yields
    /// no calls (the normalizer only acts on `[TOOL_CALLS]` payloads).
    func test_noSentinel_isIdentityAndYieldsNoCall() {
        let prose = "Here is your answer: 42."
        var normalizer = MLXMistralToolCallNormalizer(dialect: .mistral)
        _ = normalizer.process(prose)
        XCTAssertEqual(normalizer.finalize(), prose,
            "text with no [TOOL_CALLS] sentinel must pass through unchanged")
        XCTAssertTrue(parseBack(prose).isEmpty)
    }

    // MARK: - Non-Mistral dialects are identity

    func test_nonMistralDialect_isIdentity() {
        for dialect in [MLXToolDialect.qwen25, .llama, .unknown] {
            var normalizer = MLXMistralToolCallNormalizer(dialect: dialect)
            let chunk = "[TOOL_CALLS][{function:now,arguments:{}]"
            XCTAssertEqual(normalizer.process(chunk), chunk,
                "non-Mistral dialect must pass chunks through unchanged (\(dialect))")
            XCTAssertEqual(normalizer.finalize(), "",
                "non-Mistral dialect finalize must emit nothing (\(dialect))")
        }
    }
}
