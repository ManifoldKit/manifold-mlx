import XCTest
import ManifoldInference
@_spi(Testing) import ManifoldMLX

/// Unit tests for the Mistral `[TOOL_CALLS]` envelope grammar (#106).
///
/// These run off-GPU: they only exercise the grammar builder + the byte-level
/// matcher, so the "accepts / rejects" assertions prove the constraint surface
/// without loading a model. The decode-time effect (forcing the model onto this
/// envelope) is validated by the live `tool-decoy-sweep d0` soak.
final class MLXMistralToolGrammarTests: XCTestCase {

    // MARK: - Fixtures

    private func tool(_ name: String) -> ToolDefinition {
        ToolDefinition(name: name, description: "test tool \(name)")
    }

    private let now = "now"
    private let calc = "calc"

    private func grammar(
        _ tools: [String],
        _ choice: ToolChoice,
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws -> GBNFGrammar {
        try XCTUnwrap(
            MLXMistralToolGrammar.build(tools: tools.map(tool), toolChoice: choice),
            "expected a grammar for \(tools) / \(choice)",
            file: file, line: line
        )
    }

    private func assertMatches(_ s: String, _ g: GBNFGrammar, _ msg: String = "") {
        XCTAssertTrue(GBNFMatcher.matches(s, grammar: g), "should accept \(s.debugDescription) \(msg)")
    }
    private func assertRejects(_ s: String, _ g: GBNFGrammar, _ msg: String = "") {
        XCTAssertFalse(GBNFMatcher.matches(s, grammar: g), "should reject \(s.debugDescription) \(msg)")
    }

    // MARK: - Well-formed envelopes (strict / .required)

    func test_strict_acceptsCanonicalSingleCall() throws {
        let g = try grammar([now, calc], .required)
        assertMatches(#"[TOOL_CALLS] [{"name":"now","arguments":{}}]"#, g)
    }

    func test_strict_acceptsWhitespaceVariant() throws {
        let g = try grammar([now, calc], .required)
        // `ws` (`[ \t\n]*`) admits the spaced canonical form Mistral documents.
        assertMatches(#"[TOOL_CALLS] [{"name": "now", "arguments": {}}]"#, g)
    }

    func test_strict_acceptsParallelCalls() throws {
        let g = try grammar([now, calc], .required)
        assertMatches(
            #"[TOOL_CALLS] [{"name":"now","arguments":{}}, {"name":"calc","arguments":{}}]"#,
            g
        )
    }

    func test_strict_acceptsNonEmptyArgumentsObject() throws {
        let g = try grammar([now, calc], .required)
        // `arguments` lowers to the generic JSON object (no parameter schema), so
        // any well-formed object is admitted.
        assertMatches(#"[TOOL_CALLS] [{"name":"calc","arguments":{"a":7823,"b":41}}]"#, g)
    }

    // MARK: - Rejects the #106 malformed emissions (strict)

    func test_strict_rejectsBareObjectWithoutEnvelope() throws {
        let g = try grammar([now, calc], .required)
        // #106: many samples emit bare JSON with no `[TOOL_CALLS]` envelope.
        assertRejects(#"{"name":"now","arguments":{}}"#, g)
    }

    func test_strict_rejectsMangledNoSpaceUnquotedKeys() throws {
        let g = try grammar([now, calc], .required)
        // #106: `[TOOL_CALLS][{function:now,arguments:{}]` — dropped space,
        // unquoted keys, wrong key name.
        assertRejects("[TOOL_CALLS][{function:now,arguments:{}]", g)
    }

    func test_strict_rejectsUnquotedKeysWithSentinelSpace() throws {
        let g = try grammar([now, calc], .required)
        assertRejects("[TOOL_CALLS] [{name: now}]", g)
    }

    func test_strict_rejectsMissingClosingBraces() throws {
        let g = try grammar([now, calc], .required)
        // #106: mismatched brackets / missing `}`.
        assertRejects(#"[TOOL_CALLS] [{"name":"now","arguments":{}"#, g)
    }

    func test_strict_rejectsInterleavedProse() throws {
        let g = try grammar([now, calc], .required)
        // #106: tool call interleaved with hallucinated prose.
        assertRejects(
            "[TOOL_CALLS] [{\"name\":\"now\",\"arguments\":{}}]\n\n\"2022-10-14T15:37:46Z\"",
            g
        )
    }

    func test_strict_rejectsUnadvertisedToolName() throws {
        let g = try grammar([now, calc], .required)
        // `"name"` is pinned to the enum of advertised tools — a decoy/hallucinated
        // name cannot be emitted.
        assertRejects(#"[TOOL_CALLS] [{"name":"banana","arguments":{}}]"#, g)
    }

    // MARK: - .tool(name:) narrows the union

    func test_toolChoiceNamed_acceptsOnlyThatTool() throws {
        let g = try grammar([now, calc], .tool(name: now))
        assertMatches(#"[TOOL_CALLS] [{"name":"now","arguments":{}}]"#, g)
        assertRejects(#"[TOOL_CALLS] [{"name":"calc","arguments":{}}]"#, g)
    }

    // MARK: - Permissive (.auto): prose OR a well-formed envelope

    func test_permissive_acceptsWellFormedEnvelope() throws {
        let g = try grammar([now, calc], .auto)
        assertMatches(#"[TOOL_CALLS] [{"name":"now","arguments":{}}]"#, g)
    }

    func test_permissive_acceptsPlainProse() throws {
        let g = try grammar([now, calc], .auto)
        // `.auto` must let the model decline to call a tool.
        assertMatches("I can't help with that.", g)
    }

    func test_permissive_rejectsMangledEnvelopeOnceCommitted() throws {
        let g = try grammar([now, calc], .auto)
        // A leading `[` commits to the envelope branch (prose-head excludes `[`),
        // so the dropped-space mangled form is still rejected under `.auto`.
        assertRejects("[TOOL_CALLS][{function:now,arguments:{}]", g)
    }

    // MARK: - Degenerate inputs

    func test_emptyTools_returnsNil() throws {
        XCTAssertNil(MLXMistralToolGrammar.build(tools: [], toolChoice: .required))
    }

    func test_toolChoiceNone_returnsNil() throws {
        XCTAssertNil(MLXMistralToolGrammar.build(tools: [tool(now)], toolChoice: .none))
    }

    // MARK: - Emitted source shape

    func test_buildGBNF_emitsSentinelAndRenamedRoot() throws {
        let src = try XCTUnwrap(
            MLXMistralToolGrammar.buildGBNF(tools: [tool(now)], toolChoice: .required),
            "expected GBNF source"
        )
        XCTAssertTrue(src.contains(#""[TOOL_CALLS] ""#), "envelope must emit the spaced sentinel")
        XCTAssertTrue(src.contains("\nroot ::= "), "must define a root rule")
        XCTAssertTrue(src.contains("call ::= "), "core's `root` union must be renamed to `call`")
        XCTAssertFalse(src.contains("\nroot ::= toolcall"), "core's bare-union root must be renamed away")
    }

    func test_buildGBNF_permissiveEmitsProseEscape() throws {
        let src = try XCTUnwrap(
            MLXMistralToolGrammar.buildGBNF(tools: [tool(now)], toolChoice: .auto),
            "expected GBNF source"
        )
        XCTAssertTrue(src.contains("prose"), "permissive mode must emit a prose escape")
        XCTAssertTrue(src.contains("envelope ::= "), "permissive root must reference an envelope rule")
    }
}
