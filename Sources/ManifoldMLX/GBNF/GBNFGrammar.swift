import Foundation

/// A parsed GBNF grammar in a byte-level normal form suitable for incremental
/// matching during sampling (#96, option B).
///
/// This executor targets the GBNF subset that `ToolGrammarBuilder` emits (and
/// the common hand-written shapes): rule defs `::=`, references + recursion,
/// concatenation, alternation `|`, grouping `()`, postfix repetition `* + ?`,
/// string literals (with escapes), and character classes `[...]` including
/// ranges, negation `[^...]`, and `\xHH` / `\n` / `\t` / `\r` escapes.
///
/// Anything outside that subset (`$ref`-style external refs, `.` any-char,
/// numeric repetition `{m,n}`, etc.) raises ``GBNFError`` at parse time — the
/// backend surfaces it as `InferenceError.unsupportedGrammar` rather than
/// silently mis-constraining output (issue #96 decision).
///
/// ## Normal form
///
/// Every rule is a list of *alternatives*; each alternative is a sequence of
/// ``Symbol``s (a `.charset` matching exactly one byte, or a `.rule` reference).
/// Grouping and repetition are desugared into synthetic rules so the matcher
/// only ever deals with charsets and rule references:
/// - `( A B )`  → fresh rule `A B`
/// - `X?`       → fresh rule with alts `[X]` and `[]` (epsilon)
/// - `X*`       → fresh rule `S` with alts `[X, S]` and `[]`
/// - `X+`       → `X` followed by `X*`
@_spi(Testing) public struct GBNFGrammar: Sendable {
    /// Inclusive byte ranges; matches a single byte. `negated` flips membership
    /// over the full `0...255` byte space (so `[^"]` matches every other byte,
    /// including UTF-8 continuation bytes).
    struct CharSet: Hashable, Sendable {
        var ranges: [ClosedRange<UInt8>]
        var negated: Bool

        func contains(_ byte: UInt8) -> Bool {
            let hit = ranges.contains { $0.contains(byte) }
            return negated ? !hit : hit
        }

        /// A set matching exactly one byte value.
        static func single(_ b: UInt8) -> CharSet { CharSet(ranges: [b...b], negated: false) }
    }

    enum Symbol: Hashable, Sendable {
        case charset(CharSet)
        case rule(Int)
    }

    /// `rules[id]` = the alternatives for rule `id`; each alternative is a
    /// sequence of symbols (index 0 is matched first). An empty alternative is
    /// epsilon (the rule may complete consuming nothing).
    private(set) var rules: [[[Symbol]]]
    let rootId: Int

    func alternatives(_ id: Int) -> [[Symbol]] { rules[id] }

    /// Returns a copy whose root is wrapped: `prefix <old-root> suffix`. Used to
    /// wrap a bare tool-call envelope grammar in a dialect's textual delimiters
    /// (e.g. `<tool_call>` … `</tool_call>`) so the model emits the wrapper the
    /// existing `ToolCallTransform` extracts, while the inner JSON stays
    /// schema-constrained (#96).
    @_spi(Testing) public func wrappingRoot(prefix: [UInt8], suffix: [UInt8]) -> GBNFGrammar {
        var newRules = rules
        let pre: [Symbol] = prefix.map { .charset(.single($0)) }
        let post: [Symbol] = suffix.map { .charset(.single($0)) }
        newRules.append([pre + [.rule(rootId)] + post])
        return GBNFGrammar(rules: newRules, rootId: newRules.count - 1)
    }
}

@_spi(Testing) public enum GBNFError: Error, CustomStringConvertible, Equatable {
    case unsupported(String)
    case syntax(String)
    case undefinedRule(String)
    case missingRoot

    public var description: String {
        switch self {
        case .unsupported(let m): return "unsupported GBNF construct: \(m)"
        case .syntax(let m): return "GBNF syntax error: \(m)"
        case .undefinedRule(let n): return "GBNF references undefined rule: \(n)"
        case .missingRoot: return "GBNF has no `root` rule"
        }
    }
}

// MARK: - Parser

@_spi(Testing) public extension GBNFGrammar {
    /// Parses a GBNF grammar string. The entry rule must be named `root`
    /// (llama.cpp convention; `ToolGrammarBuilder` always emits it).
    init(parsing text: String) throws {
        var parser = GBNFParser(text)
        self = try parser.parse()
    }
}

/// One-pass recursive-descent GBNF parser that lowers directly into the
/// byte-level normal form. Synthetic rules (grouping/repetition) are appended as
/// they are created, so the final `rules` array holds both named and generated
/// rules; `nameToId` only maps the named ones.
private struct GBNFParser {
    private let scanner: GBNFScanner
    private var rules: [[[GBNFGrammar.Symbol]]] = []
    private var nameToId: [String: Int] = [:]

    init(_ text: String) { scanner = GBNFScanner(text) }

    mutating func parse() throws -> GBNFGrammar {
        let tokens = try scanner.scan()
        // Split the flat token stream into rules at each `IDENT ::=` header.
        var i = 0
        var ruleBodies: [(name: String, body: [GBNFToken])] = []
        while i < tokens.count {
            guard case let .ident(name) = tokens[i] else {
                throw GBNFError.syntax("expected rule name, found \(tokens[i])")
            }
            guard i + 1 < tokens.count, tokens[i + 1] == .define else {
                throw GBNFError.syntax("expected `::=` after rule name \(name)")
            }
            var j = i + 2
            var body: [GBNFToken] = []
            while j < tokens.count {
                // A new rule begins at `IDENT ::=`.
                if case .ident = tokens[j], j + 1 < tokens.count, tokens[j + 1] == .define { break }
                body.append(tokens[j])
                j += 1
            }
            ruleBodies.append((name, body))
            i = j
        }

        // Reserve ids for all named rules first so forward references resolve.
        for (name, _) in ruleBodies where nameToId[name] == nil {
            nameToId[name] = reserve()
        }
        for (name, body) in ruleBodies {
            let id = nameToId[name]!
            var cursor = GBNFTokenCursor(body)
            let alts = try parseAlternatives(&cursor)
            guard cursor.isAtEnd else {
                throw GBNFError.syntax("unexpected trailing tokens in rule \(name)")
            }
            rules[id] = alts
        }

        guard let rootId = nameToId["root"] else { throw GBNFError.missingRoot }
        // Validate every rule reference resolves.
        for alts in rules {
            for seq in alts {
                for sym in seq {
                    if case let .rule(rid) = sym, rid >= rules.count {
                        throw GBNFError.syntax("internal: dangling rule id \(rid)")
                    }
                }
            }
        }
        return GBNFGrammar(rules: rules, rootId: rootId)
    }

    // MARK: rule-body grammar

    private mutating func reserve() -> Int {
        rules.append([])
        return rules.count - 1
    }

    /// `alternatives ::= sequence ( "|" sequence )*`
    private mutating func parseAlternatives(_ c: inout GBNFTokenCursor) throws -> [[GBNFGrammar.Symbol]] {
        var alts: [[GBNFGrammar.Symbol]] = []
        alts.append(try parseSequence(&c))
        while c.peek == .pipe {
            c.advance()
            alts.append(try parseSequence(&c))
        }
        return alts
    }

    /// `sequence ::= term*` — terms concatenate until `|`, `)`, or end.
    private mutating func parseSequence(_ c: inout GBNFTokenCursor) throws -> [GBNFGrammar.Symbol] {
        var seq: [GBNFGrammar.Symbol] = []
        while let t = c.peek, t != .pipe, t != .rparen {
            seq.append(contentsOf: try parseTerm(&c))
        }
        return seq
    }

    /// `term ::= atom postfix?` where postfix is `* + ?`.
    private mutating func parseTerm(_ c: inout GBNFTokenCursor) throws -> [GBNFGrammar.Symbol] {
        let atom = try parseAtom(&c)
        switch c.peek {
        case .star?:
            c.advance()
            return [.rule(makeStar(atom))]
        case .plus?:
            c.advance()
            // X+  ≡  X X*
            return atom + [.rule(makeStar(atom))]
        case .question?:
            c.advance()
            return [.rule(makeOptional(atom))]
        default:
            return atom
        }
    }

    /// `atom ::= literal | charclass | ruleref | "(" alternatives ")"`.
    /// Returns a symbol sequence (a literal expands to one charset per byte).
    private mutating func parseAtom(_ c: inout GBNFTokenCursor) throws -> [GBNFGrammar.Symbol] {
        guard let t = c.peek else { throw GBNFError.syntax("unexpected end of rule body") }
        switch t {
        case .ident(let name):
            c.advance()
            guard let id = nameToId[name] else { throw GBNFError.undefinedRule(name) }
            return [.rule(id)]
        case .string(let bytes):
            c.advance()
            return bytes.map { .charset(.single($0)) }
        case .charclass(let set):
            c.advance()
            return [.charset(set)]
        case .lparen:
            c.advance()
            let alts = try parseAlternatives(&c)
            guard c.peek == .rparen else { throw GBNFError.syntax("expected `)`") }
            c.advance()
            let id = reserve()
            rules[id] = alts
            return [.rule(id)]
        default:
            throw GBNFError.syntax("unexpected token \(t)")
        }
    }

    /// `X*` → fresh rule `S ::= <X> S | ε`.
    private mutating func makeStar(_ atom: [GBNFGrammar.Symbol]) -> Int {
        let id = reserve()
        rules[id] = [atom + [.rule(id)], []]
        return id
    }

    /// `X?` → fresh rule `S ::= <X> | ε`.
    private mutating func makeOptional(_ atom: [GBNFGrammar.Symbol]) -> Int {
        let id = reserve()
        rules[id] = [atom, []]
        return id
    }
}
