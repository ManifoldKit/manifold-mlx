import Foundation

/// Repairs Mistral `[TOOL_CALLS]` tool-call payloads that the MLX streaming
/// detokenizer mangles into unparseable JSON.
///
/// ## Why this exists (umbrella #2005, F3 — same mechanism as issue #59)
///
/// Threading structural tools into the chat-template render (PR #102) made
/// `mistral-v0.3-4bit` actually reach for its native `[TOOL_CALLS]` channel.
/// But MLX's streaming detokenizer treats the JSON structural tokens (the
/// `"` quote token, the inter-token space) as special and **drops them from
/// the visible text stream** — the same class of bug as the Llama
/// `<|python_tag|>` close-token drop (#59, see ``MLXLlamaPythonTagNormalizer``).
///
/// The model's *intended* canonical output is:
///
/// ```
/// [TOOL_CALLS] [{"name": "<tool>", "arguments": {<json>}}]
/// ```
///
/// After detokenizer mangling the soak observed emissions like:
///
/// ```
/// [TOOL_CALLS][{function:now,arguments:{}]
/// 325087\n\n[{function:calc,arguments:{a:7823,b:41,op:'*'}}]
/// [TOOL_CALLS]_[{function:,name:read_file,arguments:{path:'example.txt'},type:function]_
/// ```
///
/// The observed drop pattern:
/// - double-quotes around keys/strings are GONE (`function:` not `"name":`);
/// - the call-name key surfaces as bare `function:` and/or `name:`;
/// - string values use single quotes (`'*'`, `'example.txt'`);
/// - inter-token spaces are dropped (no space after the `[TOOL_CALLS]` sentinel);
/// - stray `_` appears where a space/token was elided;
/// - closing braces are sometimes missing.
///
/// `MLXToolMarkers.mistral` parses with `JSONSerialization`, which rejects all of
/// the above, so the call extracts to **zero** `ToolCall`s (F1 = 0).
///
/// ## What this does
///
/// This is a thin, conservative text rewriter that sits *before* the
/// `OutputParserSession` in the MLX driver (exactly where
/// ``MLXLlamaPythonTagNormalizer`` sits, but for the `.mistral` dialect). When
/// it sees a `[TOOL_CALLS]` payload that is *not* already valid JSON, it rebuilds
/// canonical JSON from the mangled bytes:
/// - re-quotes bare object keys (`function:` → `"function":`);
/// - re-quotes single-quoted and bare string values (`'*'` → `"*"`);
/// - maps the call-name key `function:`/`name:` → `"name":` (only at the call
///   level; real argument keys are preserved);
/// - strips the stray `_` tokens the detokenizer left behind;
/// - emits the `[TOOL_CALLS] ` sentinel WITH the trailing space the
///   `MLXToolMarkers.mistral` open marker requires.
///
/// It is deliberately conservative:
/// - **Valid JSON passes through byte-for-byte.** If the payload already parses
///   as a JSON array, the normalizer leaves the whole chunk unchanged.
/// - **Genuine junk stays unparsed.** If repair cannot reconstruct a JSON array
///   whose objects carry a non-empty `name`, the original chunk is returned
///   unchanged so the downstream parser fails cleanly — no fabricated calls.
/// - **Identity for every non-`.mistral` dialect.**
///
/// Like ``MLXLlamaPythonTagNormalizer`` this is a plain text rewriter, not a
/// `Stage` (the `OutputParserSession` `Stage` enum is a closed set in
/// `ManifoldKit` and cannot be extended from this package).
// @_spi(Testing): published only for backend test targets (companion-package
// split, #1749) so the repair has a first-class deterministic unit net.
@_spi(Testing) public struct MLXMistralToolCallNormalizer {

    /// Canonical sentinel the `MLXToolMarkers.mistral` open marker matches
    /// (note the trailing space — the marker is `"[TOOL_CALLS] "`).
    private static let canonicalSentinel = "[TOOL_CALLS] "

    /// Bare sentinel as the model emits it (the trailing space is among the
    /// dropped tokens, so the raw stream often shows no space).
    private static let bareSentinel = "[TOOL_CALLS]"

    /// Active only for the Mistral dialect; every other dialect is identity.
    private let enabled: Bool

    /// Whole-stream buffer. The `[TOOL_CALLS]` payload runs to end-of-generation
    /// (no closing delimiter — `MLXToolMarkers.mistral` is `closesAtEnd`), and the
    /// mangling spreads structure across the array, so we cannot repair a single
    /// chunk in isolation. We buffer the whole stream and repair once at
    /// ``finalize()``, emitting nothing until then. This mirrors the EOS-keyed
    /// shape of the dialect itself.
    private var buffer = ""

    public init(dialect: MLXToolDialect) {
        self.enabled = (dialect == .mistral)
    }

    /// Buffer the chunk (Mistral) or pass it through (every other dialect).
    ///
    /// For Mistral we emit nothing per-chunk: the payload is EOS-keyed and the
    /// mangled structure can straddle chunk boundaries, so repair must see the
    /// whole stream. The buffered text is flushed (repaired) by ``finalize()``.
    public mutating func process(_ chunk: String) -> String {
        guard enabled else { return chunk }
        buffer += chunk
        return ""
    }

    /// Flush the buffered stream, repairing a mangled `[TOOL_CALLS]` payload into
    /// canonical JSON when needed. Identity for non-Mistral dialects.
    public mutating func finalize() -> String {
        guard enabled else { return "" }
        let raw = buffer
        buffer = ""
        return Self.repair(raw)
    }

    // MARK: - Repair

    /// Repair `raw` if it carries a `[TOOL_CALLS]` payload that does not already
    /// parse. Returns the original `raw` unchanged when (a) there is no
    /// `[TOOL_CALLS]` payload, (b) the payload is already valid JSON, or (c)
    /// repair cannot reconstruct a usable call array (junk stays unparsed).
    static func repair(_ raw: String) -> String {
        // Locate the sentinel. Repair only touches the payload that follows it;
        // any prose before the sentinel is preserved verbatim (it surfaces as
        // visible text downstream, like the canonical path).
        guard let sentinelRange = raw.range(of: bareSentinel) else {
            return raw
        }
        let prefix = String(raw[raw.startIndex..<sentinelRange.lowerBound])
        var payload = String(raw[sentinelRange.upperBound...])

        // Strip the leading separators the detokenizer leaves between the
        // sentinel and the array: dropped-space artifacts (`_`) and whitespace.
        payload = Self.stripLeadingArtifacts(payload)

        // Trim trailing artifacts (`_`, whitespace) the detokenizer appended.
        payload = Self.stripTrailingArtifacts(payload)

        // (b) Already-valid JSON array → pass the whole chunk through unchanged.
        if Self.isValidCallArray(payload) {
            return raw
        }

        // (c) Attempt repair. On failure, return `raw` unchanged so the parser
        //     fails cleanly rather than us fabricating a call.
        guard let repaired = Self.rebuildCallArray(payload) else {
            return raw
        }
        return prefix + canonicalSentinel + repaired
    }

    /// `true` when `payload` parses as a JSON array of objects each carrying a
    /// non-empty `name` string — i.e. the canonical, already-valid shape.
    private static func isValidCallArray(_ payload: String) -> Bool {
        guard let data = payload.data(using: .utf8),
              let array = (try? JSONSerialization.jsonObject(with: data)) as? [[String: Any]],
              !array.isEmpty
        else {
            return false
        }
        // Every element must look like a call (have a usable name). A valid array
        // whose elements use the `name` key passes through untouched.
        return array.allSatisfy { ($0["name"] as? String).map { !$0.isEmpty } ?? false }
    }

    private static func stripLeadingArtifacts(_ s: String) -> String {
        var idx = s.startIndex
        while idx < s.endIndex, s[idx] == "_" || s[idx].isWhitespace {
            idx = s.index(after: idx)
        }
        return String(s[idx...])
    }

    private static func stripTrailingArtifacts(_ s: String) -> String {
        var end = s.endIndex
        while end > s.startIndex {
            let prev = s.index(before: end)
            if s[prev] == "_" || s[prev].isWhitespace {
                end = prev
            } else {
                break
            }
        }
        return String(s[s.startIndex..<end])
    }

    // MARK: - Tolerant rebuild

    /// Rebuild canonical JSON for the call array from the mangled `payload`.
    ///
    /// The payload should begin with `[` (the array open). We walk it with a
    /// hand tokenizer that understands the *mangled* grammar — bare identifiers,
    /// single-quoted strings, numbers, `{}`/`[]` nesting, stray `_` — and
    /// re-emit canonical JSON: keys and string values get double-quoted, the
    /// call-name key is normalised to `"name"`, and missing closing braces are
    /// balanced. Returns `nil` if the result does not parse into at least one
    /// call with a non-empty name (so junk stays unparsed).
    private static func rebuildCallArray(_ payload: String) -> String? {
        let chars = Array(payload)
        guard let firstBracket = chars.firstIndex(of: "[") else { return nil }

        var i = firstBracket
        var out = ""
        // Bracket stack tracks container kinds so we can auto-close anything the
        // detokenizer dropped at end-of-payload. `objectDepth` tells us when a
        // bare identifier is a KEY (immediately inside `{`, before its `:`).
        var stack: [Character] = []
        // For the current object level, true once we have emitted the key and the
        // following value is being read (so the next identifier is a value, not a
        // key). One flag per object-nesting level.
        var expectingValueStack: [Bool] = []

        func atKeyPosition() -> Bool {
            // We're at a key position when the innermost container is an object
            // and we are not currently expecting a value.
            guard stack.last == "{" else { return false }
            return expectingValueStack.last == false
        }

        // Mark the current object value-slot as filled, so the empty-value guard
        // at `,` / `}` does not wrongly inject a `null` after a real value.
        func valueEmitted() {
            if stack.last == "{", !expectingValueStack.isEmpty {
                expectingValueStack[expectingValueStack.count - 1] = false
            }
        }

        // An empty value slot (`key:` immediately followed by `,` or `}`, as in
        // fixture 03's `function:,`) — retract the dangling `"key":` we already
        // emitted, plus a leading separator if present, so no orphan key remains.
        // Returns true when a retraction happened (so the caller drops the
        // separator too). Retracting (rather than filling with `null`) avoids a
        // duplicate-`name` object whose first value would shadow the real call
        // name — `JSONSerialization` keeps the FIRST duplicate key.
        func retractEmptyKey() {
            guard out.hasSuffix(":") else { return }
            out.removeLast() // the colon
            // Remove the quoted key: a `"…"` token immediately before the colon.
            guard out.hasSuffix("\"") else { return }
            out.removeLast() // closing quote
            while let last = out.last, last != "\"" { out.removeLast() }
            if out.last == "\"" { out.removeLast() } // opening quote
            // Drop a separator that now dangles before the retracted key.
            if out.last == "," { out.removeLast() }
        }

        while i < chars.count {
            let c = chars[i]
            switch c {
            case "_":
                // Stray detokenizer artifact — drop it.
                i += 1
            case " ", "\t", "\n", "\r":
                i += 1
            case "[":
                stack.append("[")
                out += "["
                i += 1
            case "{":
                stack.append("{")
                expectingValueStack.append(false)
                out += "{"
                i += 1
            case "]":
                // The detokenizer often drops the object's closing `}` before the
                // array `]` (fixture 01: `…arguments:{}]`). Auto-close any objects
                // open above the nearest array so the array closes well-formed.
                while let top = stack.last, top == "{" {
                    if expectingValueStack.last == true { out += "null" }
                    out += "}"
                    stack.removeLast()
                    if !expectingValueStack.isEmpty { expectingValueStack.removeLast() }
                }
                if stack.last == "[" { stack.removeLast() }
                out += "]"
                i += 1
            case "}":
                // A `:` with no value before the `}` (an empty key) — retract the
                // dangling key so the object stays well-formed.
                if stack.last == "{", expectingValueStack.last == true {
                    retractEmptyKey()
                }
                if stack.last == "{" {
                    stack.removeLast()
                    if !expectingValueStack.isEmpty { expectingValueStack.removeLast() }
                }
                out += "}"
                // Closing an object completes the value slot of its parent object.
                valueEmitted()
                i += 1
            case ",":
                // Empty value slot (`key:,`, fixture 03's `function:,`) — retract
                // the dangling key. If that leaves `out` at an object/array open,
                // drop this comma too so we never emit a leading `,`.
                if stack.last == "{", expectingValueStack.last == true {
                    retractEmptyKey()
                    if out.last == "{" || out.last == "[" {
                        if stack.last == "{", !expectingValueStack.isEmpty {
                            expectingValueStack[expectingValueStack.count - 1] = false
                        }
                        i += 1
                        continue
                    }
                }
                out += ","
                // After a comma inside an object the next token is a key again.
                if stack.last == "{", !expectingValueStack.isEmpty {
                    expectingValueStack[expectingValueStack.count - 1] = false
                }
                i += 1
            case ":":
                out += ":"
                // After the colon inside an object we expect the value.
                if stack.last == "{", !expectingValueStack.isEmpty {
                    expectingValueStack[expectingValueStack.count - 1] = true
                }
                i += 1
            case "\"":
                // Already-quoted string. In a key position it is a key; in a value
                // position it is a value.
                let (token, next) = Self.readDoubleQuoted(chars, from: i)
                if atKeyPosition() {
                    out += token
                } else {
                    out += token
                    valueEmitted()
                }
                i = next
            case "'":
                // Single-quoted string → re-quote as a JSON double-quoted string.
                let (inner, next) = Self.readSingleQuoted(chars, from: i)
                if atKeyPosition() {
                    out += Self.jsonString(inner)
                } else {
                    out += Self.jsonString(inner)
                    valueEmitted()
                }
                i = next
            default:
                if c == "-" || c.isNumber {
                    // Bare number (or negative). Only treat as a number when in a
                    // VALUE position; a bare identifier in a key position is a key.
                    let (token, next) = Self.readBareToken(chars, from: i)
                    if atKeyPosition() {
                        out += Self.normalizedKey(token)
                    } else {
                        out += Self.numericOrString(token)
                        valueEmitted()
                    }
                    i = next
                } else if c.isLetter || c == "_" {
                    let (token, next) = Self.readBareToken(chars, from: i)
                    if atKeyPosition() {
                        out += Self.normalizedKey(token)
                    } else {
                        // Bare value identifier (true/false/null pass through;
                        // everything else becomes a quoted string).
                        out += Self.bareValue(token)
                        valueEmitted()
                    }
                    i = next
                } else {
                    // Unknown punctuation — drop it rather than corrupt the JSON.
                    i += 1
                }
            }
        }

        // Auto-close any containers the detokenizer dropped at end-of-payload.
        while let open = stack.popLast() {
            out += (open == "{") ? "}" : "]"
        }
        out = Self.stripTrailingArtifacts(out)

        // Parse the rebuilt JSON and re-serialise to a clean canonical array.
        // Re-serialising (rather than returning `out` verbatim) collapses any
        // duplicate `name` keys deterministically — fixture 03 produces an empty
        // `function:` collapsed to `"name":null` followed by the real
        // `name:read_file`; `JSONSerialization` keeps the last value, so the
        // canonical array carries `"name":"read_file"`. Returns `nil` when the
        // result is not at least one named call so genuine junk stays unparsed.
        guard let data = out.data(using: .utf8),
              let array = (try? JSONSerialization.jsonObject(with: data)) as? [[String: Any]],
              !array.isEmpty
        else {
            return nil
        }
        let calls = array.filter { ($0["name"] as? String).map { !$0.isEmpty } ?? false }
        guard !calls.isEmpty else { return nil }
        guard let canonical = try? JSONSerialization.data(withJSONObject: calls),
              let canonicalStr = String(data: canonical, encoding: .utf8)
        else {
            return nil
        }
        return canonicalStr
    }

    /// Map a bare key token to a canonical double-quoted JSON key. The call-name
    /// keys (`function`, `name`) collapse to `"name"`; every other key keeps its
    /// own identity (argument keys are preserved).
    private static func normalizedKey(_ token: String) -> String {
        let trimmed = token.trimmingCharacters(in: CharacterSet(charactersIn: "_"))
        if trimmed == "function" || trimmed == "name" {
            return "\"name\""
        }
        return Self.jsonString(trimmed)
    }

    /// A bare value identifier: JSON literals pass through; anything else is a
    /// quoted string (the model meant a string but the quotes were dropped).
    private static func bareValue(_ token: String) -> String {
        switch token {
        case "true", "false", "null":
            return token
        default:
            return Self.jsonString(token)
        }
    }

    /// A bare numeric token stays a number when it is a valid JSON number;
    /// otherwise it becomes a quoted string.
    private static func numericOrString(_ token: String) -> String {
        if Double(token) != nil {
            return token
        }
        return Self.jsonString(token)
    }

    /// Read a bare token (identifier / number) starting at `start`. Stops at any
    /// JSON structural char, quote, colon, comma, or whitespace.
    private static func readBareToken(_ chars: [Character], from start: Int) -> (String, Int) {
        var i = start
        var token = ""
        while i < chars.count {
            let c = chars[i]
            if c == ":" || c == "," || c == "{" || c == "}" || c == "[" || c == "]"
                || c == "\"" || c == "'" || c == " " || c == "\t" || c == "\n" || c == "\r" {
                break
            }
            token.append(c)
            i += 1
        }
        return (token, i)
    }

    /// Read a single-quoted string body (between `'`…`'`), returning the inner
    /// text and the index just past the closing quote.
    private static func readSingleQuoted(_ chars: [Character], from start: Int) -> (String, Int) {
        var i = start + 1
        var inner = ""
        while i < chars.count, chars[i] != "'" {
            inner.append(chars[i])
            i += 1
        }
        if i < chars.count { i += 1 } // consume closing quote
        return (inner, i)
    }

    /// Copy an already-double-quoted JSON string verbatim (honouring `\` escapes),
    /// returning the token (with quotes) and the index just past the closing quote.
    private static func readDoubleQuoted(_ chars: [Character], from start: Int) -> (String, Int) {
        var i = start + 1
        var token = "\""
        while i < chars.count {
            let c = chars[i]
            token.append(c)
            if c == "\\", i + 1 < chars.count {
                token.append(chars[i + 1])
                i += 2
                continue
            }
            if c == "\"" {
                i += 1
                return (token, i)
            }
            i += 1
        }
        return (token, i)
    }

    /// Encode `s` as a JSON double-quoted string (escaping `"` and `\`).
    private static func jsonString(_ s: String) -> String {
        var out = "\""
        for ch in s {
            switch ch {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            default: out.append(ch)
            }
        }
        out += "\""
        return out
    }
}
