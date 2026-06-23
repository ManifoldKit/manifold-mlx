import Foundation

/// Lexical tokens for the supported GBNF subset (#96).
enum GBNFToken: Equatable {
    case ident(String)
    case define          // ::=
    case pipe            // |
    case lparen          // (
    case rparen          // )
    case star            // *
    case plus            // +
    case question        // ?
    case string([UInt8]) // "…" decoded to raw bytes
    case charclass(GBNFGrammar.CharSet)
}

/// Cursor over a flat token list used by the recursive-descent rule-body parser.
struct GBNFTokenCursor {
    private let tokens: [GBNFToken]
    private var index = 0
    init(_ tokens: [GBNFToken]) { self.tokens = tokens }
    var peek: GBNFToken? { index < tokens.count ? tokens[index] : nil }
    var isAtEnd: Bool { index >= tokens.count }
    mutating func advance() { index += 1 }
}

/// Hand-written GBNF lexer. Skips `#` line comments and whitespace; decodes
/// string literals and character classes (with `\xHH` / `\n` / `\t` / `\r` /
/// `\\` / `\"` / `\]` / `\-` escapes) into the byte-level forms the matcher
/// consumes.
struct GBNFScanner {
    private let chars: [Character]
    init(_ text: String) { chars = Array(text) }

    func scan() throws -> [GBNFToken] {
        var tokens: [GBNFToken] = []
        var i = 0
        let n = chars.count
        while i < n {
            let ch = chars[i]
            if ch == "#" {                       // comment to end of line
                while i < n, chars[i] != "\n" { i += 1 }
                continue
            }
            if ch.isWhitespace { i += 1; continue }

            switch ch {
            case ":":
                guard i + 2 < n, chars[i + 1] == ":", chars[i + 2] == "=" else {
                    throw GBNFError.syntax("expected `::=`")
                }
                tokens.append(.define); i += 3
            case "|": tokens.append(.pipe); i += 1
            case "(": tokens.append(.lparen); i += 1
            case ")": tokens.append(.rparen); i += 1
            case "*": tokens.append(.star); i += 1
            case "+": tokens.append(.plus); i += 1
            case "?": tokens.append(.question); i += 1
            case "\"":
                let (bytes, next) = try scanString(from: i)
                tokens.append(.string(bytes)); i = next
            case "[":
                let (set, next) = try scanCharClass(from: i)
                tokens.append(.charclass(set)); i = next
            default:
                if Self.isWordChar(ch) {
                    var j = i
                    while j < n, Self.isWordChar(chars[j]) { j += 1 }
                    tokens.append(.ident(String(chars[i..<j]))); i = j
                } else {
                    throw GBNFError.syntax("unexpected character '\(ch)'")
                }
            }
        }
        return tokens
    }

    /// llama.cpp `is_word_char`: `[a-zA-Z0-9-]`.
    private static func isWordChar(_ c: Character) -> Bool {
        c.isLetter || c.isNumber || c == "-"
    }

    // MARK: literals

    /// Scans `"…"` starting at the opening quote; returns the decoded UTF-8
    /// bytes and the index after the closing quote.
    private func scanString(from start: Int) throws -> ([UInt8], Int) {
        var i = start + 1
        let n = chars.count
        var bytes: [UInt8] = []
        while i < n {
            let ch = chars[i]
            if ch == "\"" { return (bytes, i + 1) }
            if ch == "\\" {
                let (b, next) = try scanEscape(from: i)
                bytes.append(contentsOf: b); i = next
            } else {
                bytes.append(contentsOf: Array(String(ch).utf8)); i += 1
            }
        }
        throw GBNFError.syntax("unterminated string literal")
    }

    /// Scans `[...]` starting at `[`; returns the charset and the index after
    /// the closing `]`.
    private func scanCharClass(from start: Int) throws -> (GBNFGrammar.CharSet, Int) {
        var i = start + 1
        let n = chars.count
        var negated = false
        if i < n, chars[i] == "^" { negated = true; i += 1 }

        var ranges: [ClosedRange<UInt8>] = []
        // Decodes one class member (a single byte), honouring escapes.
        func member(at idx: Int) throws -> (UInt8, Int) {
            if chars[idx] == "\\" {
                let (b, next) = try scanEscape(from: idx)
                guard b.count == 1 else {
                    throw GBNFError.unsupported("multi-byte escape in character class")
                }
                return (b[0], next)
            }
            let utf8 = Array(String(chars[idx]).utf8)
            guard utf8.count == 1 else {
                throw GBNFError.unsupported("multi-byte character in character class")
            }
            return (utf8[0], idx + 1)
        }

        while i < n {
            if chars[i] == "]" {
                return (GBNFGrammar.CharSet(ranges: ranges, negated: negated), i + 1)
            }
            let (lo, afterLo) = try member(at: i)
            // Range `a-b` when a `-` (not the closing `]`) follows.
            if afterLo < n, chars[afterLo] == "-", afterLo + 1 < n, chars[afterLo + 1] != "]" {
                let (hi, afterHi) = try member(at: afterLo + 1)
                guard lo <= hi else { throw GBNFError.syntax("inverted char range") }
                ranges.append(lo...hi); i = afterHi
            } else {
                ranges.append(lo...lo); i = afterLo
            }
        }
        throw GBNFError.syntax("unterminated character class")
    }

    /// Decodes a backslash escape starting at `\`. Returns the byte(s) and the
    /// index after the escape. `\xHH` yields one byte; the named escapes one;
    /// any other escaped char is taken literally (its UTF-8 bytes).
    private func scanEscape(from start: Int) throws -> ([UInt8], Int) {
        let n = chars.count
        guard start + 1 < n else { throw GBNFError.syntax("dangling escape") }
        let e = chars[start + 1]
        switch e {
        case "n": return ([0x0A], start + 2)
        case "r": return ([0x0D], start + 2)
        case "t": return ([0x09], start + 2)
        case "\\": return ([0x5C], start + 2)
        case "\"": return ([0x22], start + 2)
        case "]": return ([0x5D], start + 2)
        case "[": return ([0x5B], start + 2)
        case "-": return ([0x2D], start + 2)
        case "/": return ([0x2F], start + 2)
        case "x":
            guard start + 3 < n,
                  let hi = chars[start + 2].hexDigitValue,
                  let lo = chars[start + 3].hexDigitValue else {
                throw GBNFError.syntax("malformed \\xHH escape")
            }
            return ([UInt8(hi * 16 + lo)], start + 4)
        default:
            // Unknown escape → the escaped character literally.
            return (Array(String(e).utf8), start + 2)
        }
    }
}
