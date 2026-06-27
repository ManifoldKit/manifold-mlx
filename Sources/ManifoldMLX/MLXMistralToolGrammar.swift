import Foundation
import ManifoldInference

/// Builds the GBNF grammar that constrains MLX-Mistral decoding to a well-formed
/// `[TOOL_CALLS]` envelope (#106).
///
/// ## Why this exists
///
/// Mistral's tool-call wire format is a sentinel-prefixed JSON *array* with no
/// closing delimiter:
///
/// ```
/// [TOOL_CALLS] [ {"name": "<tool>", "arguments": {…}} (, {…})* ]
/// ```
///
/// Core's `ToolGrammarBuilder` (#1859) emits only the *bare object union*
/// (`{"name": …, "arguments": …}`), which is the shape Qwen/Llama emit inside a
/// `<tool_call>` wrapper. For Mistral that bare union is the wrong envelope — and
/// under `toolChoice == .auto` (the queue's default) core emits a *permissive*
/// grammar whose prose escape (`[^{]`) lets Mistral free-run the `[TOOL_CALLS]`
/// channel completely unconstrained (it starts with `[`, not `{`). The
/// detokenizer then mangles the quotes/spaces and `MLXToolMarkers.mistral`
/// extracts zero calls — `tool-decoy-sweep d0` F1 = 0.000 (#106, #104).
///
/// This builder wraps core's inner object union in the `[TOOL_CALLS]` array
/// envelope so decoding can only follow a parseable path:
///
/// ```
/// root     ::= "[TOOL_CALLS] " "[" ws call ( ws "," ws call )* ws "]"   (strict)
/// root     ::= envelope | prose                                          (permissive)
/// call     ::= <core's toolcall-N union, renamed from `root`>
/// ```
///
/// The mandatory trailing space in `"[TOOL_CALLS] "` matches the
/// `MLXToolMarkers.mistral` open marker exactly, so the existing parser path
/// extracts the calls with no normalizer assistance.
@_spi(Testing) public enum MLXMistralToolGrammar {

    /// Builds the parsed `[TOOL_CALLS]` envelope grammar for `tools`, honouring
    /// `toolChoice`, or `nil` when no envelope can be constrained (`tools` empty,
    /// `toolChoice == .none`, or the inner union could not be built/parsed).
    ///
    /// - `.auto` → permissive: the model may emit prose *or* a `[TOOL_CALLS]`
    ///   envelope (the first byte disambiguates: `[` enters the envelope).
    /// - `.required` → strict union over every advertised tool.
    /// - `.tool(name:)` → strict, single-tool union.
    /// - `.none` → `nil` (no constraint — the caller must not force a call).
    @_spi(Testing) public static func build(
        tools: [ToolDefinition],
        toolChoice: ToolChoice
    ) -> GBNFGrammar? {
        guard let source = buildGBNF(tools: tools, toolChoice: toolChoice) else { return nil }
        return try? GBNFGrammar(parsing: source)
    }

    /// Builds the envelope grammar's GBNF *source string* (the parse step is
    /// split out so tests can inspect the emitted text). Returns `nil` under the
    /// same conditions as ``build(tools:toolChoice:)``.
    @_spi(Testing) public static func buildGBNF(
        tools: [ToolDefinition],
        toolChoice: ToolChoice
    ) -> String? {
        guard !tools.isEmpty else { return nil }

        // Map the request's toolChoice to the inner-union mode. `.none` means
        // "must not call a tool", so there is no envelope to constrain.
        let permissive: Bool
        let innerMode: ToolGrammarBuilder.Mode
        switch toolChoice {
        case .auto:
            permissive = true
            innerMode = .strict(only: nil)
        case .required:
            permissive = false
            innerMode = .strict(only: nil)
        case .tool(let name):
            permissive = false
            innerMode = .strict(only: name)
        case .none:
            return nil
        }

        // Core emits the bare object union with `root ::= toolcall-0 | …` as its
        // entry rule plus every helper/generic rule. Use `.strict` so the inner
        // union is a clean alternation of call objects — the prose escape (for
        // `.auto`) is added at the envelope level below, not inside `call`.
        guard let inner = ToolGrammarBuilder().buildGrammar(for: tools, mode: innerMode) else {
            return nil
        }

        // Rename core's entry rule `root` → `call`. The builder always emits
        // `root ::= …` as the first line and never references `root` from any
        // other rule, so renaming the definition is sufficient and safe.
        var lines = inner.components(separatedBy: "\n")
        let rootPrefix = "root ::= "
        guard let rootIdx = lines.firstIndex(where: { $0.hasPrefix(rootPrefix) }) else {
            return nil
        }
        lines[rootIdx] = "call ::= " + lines[rootIdx].dropFirst(rootPrefix.count)

        // The `[TOOL_CALLS] ` sentinel (mandatory trailing space, matching the
        // `MLXToolMarkers.mistral` open marker) followed by the JSON array of one
        // or more calls. `ws` is core's generic whitespace rule (already emitted
        // by the inner grammar), so optional inter-element whitespace is allowed.
        let envelopeRHS =
            #""[TOOL_CALLS] " "[" ws call ( ws "," ws call )* ws "]""#

        var out: [String] = []
        if permissive {
            // `.auto`: the first sampled token may be prose (never forced to a
            // call) or the envelope. Disambiguated at the first byte — the
            // envelope begins with `[`, so `prose-head` excludes `[` (\x5B).
            // `prose` is non-empty (a mandatory head byte) so the root never
            // matches empty and the decoder can't collapse onto EOS at token 0.
            out.append("root ::= envelope | prose")
            out.append("envelope ::= " + envelopeRHS)
            out.append("prose ::= prose-head prose-tail*")
            out.append(#"prose-head ::= [^\x5B]"#)
            out.append(#"prose-tail ::= [^\x00]"#)
        } else {
            out.append("root ::= " + envelopeRHS)
        }
        out.append(contentsOf: lines)
        return out.joined(separator: "\n")
    }
}
