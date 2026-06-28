import XCTest
@_spi(Testing) import ManifoldMLX

/// Off-GPU unit tests for the GBNF executor (#96, option B): the parser + the
/// byte-level matcher. These run in CI without Apple Silicon — they exercise no
/// MLX/Metal code, only the grammar engine.
final class GBNFGrammarTests: XCTestCase {

    private func grammar(_ text: String) throws -> GBNFGrammar {
        try GBNFGrammar(parsing: text)
    }

    private func assertMatches(_ text: String, _ g: GBNFGrammar, _ msg: String = "") {
        XCTAssertTrue(GBNFMatcher.matches(text, grammar: g), "should accept \(text.debugDescription) \(msg)")
    }
    private func assertRejects(_ text: String, _ g: GBNFGrammar, _ msg: String = "") {
        XCTAssertFalse(GBNFMatcher.matches(text, grammar: g), "should reject \(text.debugDescription) \(msg)")
    }

    // MARK: - Literals & sequence

    func test_literalSequence() throws {
        let g = try grammar(#"root ::= "ab" "c""#)
        assertMatches("abc", g)
        assertRejects("ab", g, "incomplete")
        assertRejects("abcd", g, "trailing")
        assertRejects("axc", g)
    }

    func test_quotedLiteral_withEscapes() throws {
        // root matches the 6-byte string: "north" including the quotes.
        let g = try grammar(#"root ::= "\"north\"""#)
        assertMatches("\"north\"", g)
        assertRejects("north", g)
    }

    // MARK: - Alternation, grouping, repetition

    func test_alternation() throws {
        let g = try grammar(#"root ::= "yes" | "no""#)
        assertMatches("yes", g)
        assertMatches("no", g)
        assertRejects("maybe", g)
    }

    func test_optional() throws {
        let g = try grammar(##"root ::= "a" "b"?"##)
        assertMatches("a", g)
        assertMatches("ab", g)
        assertRejects("abb", g)
    }

    func test_star_and_plus() throws {
        let star = try grammar(##"root ::= "a"*"##)
        assertMatches("", star)
        assertMatches("aaaa", star)
        assertRejects("aab", star)

        let plus = try grammar(##"root ::= "a"+"##)
        assertRejects("", plus, "+ requires at least one")
        assertMatches("a", plus)
        assertMatches("aaa", plus)
    }

    func test_grouping_with_repetition() throws {
        let g = try grammar(##"root ::= "x" ( "," "y" )*"##)
        assertMatches("x", g)
        assertMatches("x,y", g)
        assertMatches("x,y,y,y", g)
        assertRejects("x,", g)
        assertRejects("x,z", g)
    }

    // MARK: - Character classes

    func test_charClass_ranges() throws {
        let g = try grammar("root ::= [0-9] [a-fA-F]")
        assertMatches("0a", g)
        assertMatches("9F", g)
        assertRejects("0g", g)
        assertRejects("aa", g)
    }

    func test_charClass_negation_isByteLevel() throws {
        // [^{] should accept any byte except '{', including a multi-byte char.
        let g = try grammar("root ::= [^{]")
        assertMatches("a", g)
        assertRejects("{", g)
    }

    func test_hexEscape_inClass() throws {
        let g = try grammar(#"root ::= [^\x00]"#)
        assertMatches("Z", g)
    }

    // MARK: - Recursion (JSON value subset)

    func test_recursiveJSON_object() throws {
        // A trimmed version of the generic JSON rules the builder emits.
        let g = try grammar(#"""
        root ::= object
        object ::= "{" ws ( member ( ws "," ws member )* )? ws "}"
        member ::= string ws ":" ws value
        value ::= object | string | "true" | "false"
        string ::= "\"" char* "\""
        char ::= [^"\\]
        ws ::= [ \t\n]*
        """#)
        assertMatches(#"{}"#, g)
        assertMatches(#"{"a": "b"}"#, g)
        assertMatches(#"{"a": {"b": "c"}, "d": true}"#, g)
        assertRejects(#"{"a": }"#, g)
        assertRejects(#"{"a" "b"}"#, g, "missing colon")
    }

    // MARK: - Tool-call envelope (exact ToolGrammarBuilder shape)

    func test_toolCallEnvelope_constrainsNameAndArgs() throws {
        // Mirrors the golden output for a single tool with an enum arg.
        let g = try grammar(#"""
        root ::= toolcall-0
        args-0-0 ::= ("\"north\"" | "\"south\"")
        args-0 ::= "{" ws "\"direction\"" ws ":" ws args-0-0 ws "}"
        toolcall-0 ::= "{" ws "\"name\"" ws ":" ws "\"set_direction\"" ws "," ws "\"arguments\"" ws ":" ws args-0 ws "}"
        ws ::= [ \t\n]*
        """#)
        assertMatches(#"{"name": "set_direction", "arguments": {"direction": "north"}}"#, g)
        assertMatches(#"{"name":"set_direction","arguments":{"direction":"south"}}"#, g)
        // Wrong tool name is impossible under the grammar.
        assertRejects(#"{"name": "other", "arguments": {"direction": "north"}}"#, g)
        // Enum value outside the set is impossible.
        assertRejects(#"{"name": "set_direction", "arguments": {"direction": "east"}}"#, g)
    }

    // MARK: - Root wrapping (dialect <tool_call> delimiters)

    func test_wrappingRoot_requiresDelimiters() throws {
        let bare = try grammar(#"root ::= "{" ws "\"x\"" ws "}""#
            + "\nws ::= [ \\t\\n]*")
        let wrapped = bare.wrappingRoot(
            prefix: Array("<tool_call>\n".utf8),
            suffix: Array("\n</tool_call>".utf8)
        )
        XCTAssertTrue(
            GBNFMatcher.matches("<tool_call>\n{\"x\"}\n</tool_call>", grammar: wrapped),
            "wrapped grammar must accept the delimited envelope"
        )
        XCTAssertFalse(
            GBNFMatcher.matches("{\"x\"}", grammar: wrapped),
            "bare envelope without delimiters must be rejected once wrapped"
        )
    }

    // MARK: - Acceptable-byte pruning

    func test_acceptableFirstBytes_atStart() throws {
        let g = try grammar(#"root ::= "{" "a""#)
        let m = GBNFMatcher(grammar: g)
        XCTAssertEqual(m.acceptableFirstBytes(), Set([UInt8(ascii: "{")]))
        XCTAssertFalse(m.isComplete, "root is non-empty at start")
    }

    // MARK: - Unsupported / errors → throw (issue #96 decision)

    func test_missingRoot_throws() {
        XCTAssertThrowsError(try grammar(#"foo ::= "x""#)) { error in
            XCTAssertEqual(error as? GBNFError, .missingRoot)
        }
    }

    func test_undefinedRule_throws() {
        XCTAssertThrowsError(try grammar("root ::= missing")) { error in
            XCTAssertEqual(error as? GBNFError, .undefinedRule("missing"))
        }
    }

    // MARK: - Scanner edge cases (regression: out-of-bounds on truncated `::=`)

    func test_truncatedDefine_throwsNotCrash() {
        // A lone `:` whose `::=` runs off the end of input must throw a syntax
        // error, not index out of bounds. Previously the `i + 2 == n` branch let
        // `chars[i + 2]` read one past the end and trapped.
        XCTAssertThrowsError(try grammar("a::")) { error in
            XCTAssertNotNil(error as? GBNFError)
        }
        XCTAssertThrowsError(try grammar("a:")) { error in
            XCTAssertNotNil(error as? GBNFError)
        }
        // `::=` itself is fine when followed by a body.
        XCTAssertNoThrow(try grammar(#"root ::= "x""#))
    }

    func test_emptyAndCommentOnlyGrammars_throwMissingRoot() {
        // None of these should crash; all lack a `root` rule.
        XCTAssertThrowsError(try grammar("")) { XCTAssertEqual($0 as? GBNFError, .missingRoot) }
        XCTAssertThrowsError(try grammar("   \n\t ")) { XCTAssertEqual($0 as? GBNFError, .missingRoot) }
        XCTAssertThrowsError(try grammar("# just a comment\n")) { XCTAssertEqual($0 as? GBNFError, .missingRoot) }
    }

    func test_nullableRoot_acceptsEmpty() throws {
        // An empty rule body is epsilon — the root completes consuming nothing.
        let g = try grammar("root ::= ")
        assertMatches("", g)
        XCTAssertTrue(GBNFMatcher(grammar: g).isComplete)
    }

    func test_trailingDashInClass_isLiteral() throws {
        // `[a-]` is the set {'a', '-'}, not a malformed range.
        let g = try grammar("root ::= [a-]")
        assertMatches("a", g)
        assertMatches("-", g)
        assertRejects("b", g)
    }

    // MARK: - Any-byte dot (#97)

    func test_dot_matchesAnyByte() throws {
        let g = try grammar("root ::= .")
        assertMatches("a", g)
        assertMatches("Z", g)
        assertMatches("0", g)
        assertRejects("", g, "dot requires exactly one byte")
        assertRejects("ab", g, "dot is a single byte")
    }

    func test_dot_inSequence() throws {
        let g = try grammar(#"root ::= "a" . "b""#)
        assertMatches("axb", g)
        assertMatches("a b", g)
        assertRejects("ab", g, "middle byte required")
        assertRejects("axyz b", g, "only one middle byte")
    }

    func test_dot_star_matchesAnything() throws {
        let g = try grammar("root ::= .*")
        assertMatches("", g)
        assertMatches("hello world", g)
        assertMatches("{}[]\"", g)
    }

    // MARK: - Unicode escapes \u{NNNN} and \UHHHHHHHH (#97)

    func test_unicodeBraced_ascii_inString() throws {
        // \u{41} = 'A' (0x41)
        let g = try grammar(#"root ::= "\u{41}""#)
        assertMatches("A", g)
        assertRejects("B", g)
    }

    func test_unicodeBraced_multibyte_inString() throws {
        // \u{1F600} = 😀, encoded as 4 UTF-8 bytes
        let g = try grammar(#"root ::= "\u{1F600}""#)
        assertMatches("😀", g)
        assertRejects("A", g)
    }

    func test_unicodeLongForm_inString() throws {
        // \U00000041 = 'A'
        let g = try grammar(#"root ::= "\U00000041""#)
        assertMatches("A", g)
        assertRejects("a", g)
    }

    func test_unicodeLongForm_emoji_inString() throws {
        // \U0001F600 = 😀
        let g = try grammar(#"root ::= "\U0001F600""#)
        assertMatches("😀", g)
    }

    func test_unicodeBraced_inCharClass_ascii_succeeds() throws {
        // \u{61} = 'a' (single byte — OK in char class)
        let g = try grammar(#"root ::= [\u{61}-\u{7A}]"#)
        assertMatches("a", g)
        assertMatches("z", g)
        assertRejects("A", g)
    }

    func test_unicodeBraced_inCharClass_multibyte_throws() throws {
        // \u{1F600} is a 4-byte UTF-8 sequence — not allowed in a char class range.
        XCTAssertThrowsError(try grammar(#"root ::= [\u{1F600}]"#)) { error in
            guard case GBNFError.unsupported(let msg) = error else {
                return XCTFail("expected .unsupported, got \(error)")
            }
            XCTAssertTrue(msg.contains("multi-byte"), "message should mention multi-byte: \(msg)")
        }
    }

    func test_unicodeMalformed_missingBrace_throws() throws {
        XCTAssertThrowsError(try grammar(#"root ::= "\u41""#)) { error in
            XCTAssertNotNil(error as? GBNFError)
        }
    }

    func test_unicodeLongForm_wrongDigitCount_throws() throws {
        // \U with fewer than 8 hex digits should fail.
        XCTAssertThrowsError(try grammar(#"root ::= "\U0041""#)) { error in
            XCTAssertNotNil(error as? GBNFError)
        }
    }

    // MARK: - Precise unsupported diagnostics (#97)

    func test_multibyteCharInClass_preciseMessage() throws {
        // A literal multi-byte Unicode character directly in a char class.
        XCTAssertThrowsError(try grammar("root ::= [😀]")) { error in
            guard case GBNFError.unsupported(let msg) = error else {
                return XCTFail("expected .unsupported, got \(error)")
            }
            XCTAssertTrue(msg.contains("multi-byte"), "message should name the problem: \(msg)")
        }
    }

    func test_rightRecursiveNullableCycle_terminates() throws {
        // B is nullable; A references B then itself. Normalization must not loop.
        let g = try grammar(#"""
        root ::= a
        a ::= b a | "x"
        b ::= "" | "y"
        """#)
        assertMatches("x", g)
        assertMatches("yx", g)
        assertMatches("yyx", g)
        assertRejects("y", g, "must terminate in x")
    }
}
