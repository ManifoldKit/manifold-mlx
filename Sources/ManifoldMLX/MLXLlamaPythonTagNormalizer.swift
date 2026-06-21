import Foundation

/// Recovers Llama 3.x native `<|python_tag|>` tool calls that the MLX streaming
/// detokenizer would otherwise leave unterminated.
///
/// ## Why this exists (issue #59 tail)
///
/// When `llama-3.2-3b` reaches for its **native** tool-call format it emits
/// `<|python_tag|>{"name":…,"parameters":…}` and closes the call with the
/// `<|eom_id|>` (documented tool terminator) or `<|eot_id|>` special token. MLX
/// (mlx-swift-lm `generateLoopTask`) treats `<|eom_id|>` / `<|eot_id|>` as
/// stop tokens and **breaks the generation loop without detokenising them** —
/// the close token never reaches the visible text stream the
/// `MLXToolMarkers` / `ToolCallTransform` parser scans. The `<|python_tag|>`
/// *open* token survives detokenisation (it is not a stop token), so the parser
/// opens a `<|python_tag|>` block that is never closed and discards the buffered
/// JSON body at `finalize()`. Net effect: `read_file` / `list_dir` (where the
/// model fell back to the native format) never dispatched.
///
/// ## What this does
///
/// This normaliser sits *before* the `OutputParserSession` in the MLX driver and
/// rewrites only the raw text chunks for the `.llama` dialect. It tracks whether
/// the stream has an *open* `<|python_tag|>` that has not yet been closed by a
/// visible `<|eom_id|>` / `<|eot_id|>`. If the stream ends with such an open
/// block, `finalize()` injects a synthetic `<|eom_id|>` close so the downstream
/// `<|python_tag|>` → `<|eom_id|>` marker fires and the JSON body is parsed into
/// a `ToolCall`. The synthetic close is appended only at stream end and only
/// when a python-tag block is genuinely open, so a normal turn (no python tag,
/// or one already closed by a *visible* terminator) is passed through byte-for-byte.
///
/// It is deliberately a thin text rewriter (not a `Stage`): the
/// `OutputParserSession` `Stage` enum is a closed set defined in `ManifoldKit`
/// and cannot be extended from this package, and the fix must not regress the
/// Qwen / textual-`<tool_call>` paths — so for any non-`.llama` dialect, and for
/// the textual wrapper, this is an identity transform.
// @_spi(Testing): published only for backend test targets (companion-package
// split, #1749) so the python-tag recovery has a first-class unit net.
@_spi(Testing) public struct MLXLlamaPythonTagNormalizer {

    /// Llama native delimiters. Mirrors `MLXToolMarkers` so the synthetic close
    /// we inject matches a marker the transform actually scans for.
    private static let openTag = "<|python_tag|>"
    private static let eomTag  = "<|eom_id|>"
    private static let eotTag  = "<|eot_id|>"

    /// Active only for the Llama dialect; every other dialect is identity.
    private let enabled: Bool

    /// `true` once a `<|python_tag|>` open has been seen with no visible close
    /// (`<|eom_id|>` / `<|eot_id|>`) after it. Reset when a close is observed.
    private var pythonTagOpen = false

    /// Carries a partial open/close tag that straddled a chunk boundary so a tag
    /// split across two MLX chunks is still recognised. Bounded by the longest
    /// tag length, so this never grows.
    private var holdback = ""

    public init(dialect: MLXToolDialect) {
        self.enabled = (dialect == .llama)
    }

    /// The maximum number of trailing characters we must retain to detect a tag
    /// that is split across chunk boundaries (one less than the longest tag).
    private static let maxTagSuffix = max(openTag.count, max(eomTag.count, eotTag.count)) - 1

    /// Rewrite one raw text chunk. For non-Llama dialects this returns `chunk`
    /// unchanged. For Llama it tracks python-tag open/close state and passes the
    /// text through verbatim (the only mutation this normaliser makes is the
    /// synthetic close appended by ``finalize()``).
    public mutating func process(_ chunk: String) -> String {
        guard enabled else { return chunk }

        // Scan the (held-back tail + new chunk) for tag transitions, then emit
        // everything except a possible partial-tag suffix we hold for next time.
        let combined = holdback + chunk
        updateOpenState(scanning: combined)

        // Decide how much of the tail to hold back: the longest suffix of
        // `combined` that is a strict prefix of any tag could be the start of a
        // tag continued in the next chunk. We only need to retain text we have
        // not already emitted via a prior holdback, so emit `combined` minus the
        // retained suffix.
        let retain = Self.partialTagSuffixLength(of: combined)
        let emitCount = combined.count - retain
        let emit = String(combined.prefix(emitCount))
        holdback = String(combined.suffix(retain))
        return emit
    }

    /// Flush the held-back tail and, when a `<|python_tag|>` block is still open,
    /// inject a synthetic `<|eom_id|>` close so the downstream marker fires.
    public mutating func finalize() -> String {
        guard enabled else { return "" }
        // The retained tail may itself complete a tag; re-scan it so a python tag
        // that ended exactly at the stream boundary is accounted for.
        updateOpenState(scanning: holdback)
        var tail = holdback
        holdback = ""
        if pythonTagOpen {
            // Inject `<|eom_id|>` (not `<|eot_id|>`): `MLXToolMarkers.markers`
            // lists the `<|python_tag|>` → `<|eom_id|>` marker *before* the
            // `<|eot_id|>` variant, and `ToolCallTransform` binds an open
            // `<|python_tag|>` to the first matching marker — so the active block
            // is waiting on `<|eom_id|>`. `<|eom_id|>` is also Llama's documented
            // tool terminator, so this is the faithful close to synthesise.
            tail += Self.eomTag
            pythonTagOpen = false
        }
        return tail
    }

    /// Update `pythonTagOpen` by walking `text` left-to-right: an open tag sets
    /// it, a visible close tag clears it. Only the final state matters, so a
    /// block opened and closed within one chunk nets to closed.
    private mutating func updateOpenState(scanning text: String) {
        guard !text.isEmpty else { return }
        var index = text.startIndex
        while index < text.endIndex {
            if matches(Self.openTag, in: text, at: index) {
                pythonTagOpen = true
                index = text.index(index, offsetBy: Self.openTag.count)
            } else if pythonTagOpen,
                      matches(Self.eomTag, in: text, at: index) {
                pythonTagOpen = false
                index = text.index(index, offsetBy: Self.eomTag.count)
            } else if pythonTagOpen,
                      matches(Self.eotTag, in: text, at: index) {
                pythonTagOpen = false
                index = text.index(index, offsetBy: Self.eotTag.count)
            } else {
                index = text.index(after: index)
            }
        }
    }

    private func matches(_ tag: String, in text: String, at index: String.Index) -> Bool {
        guard let end = text.index(index, offsetBy: tag.count, limitedBy: text.endIndex) else {
            return false
        }
        return text[index..<end] == tag
    }

    /// Length of the longest suffix of `text` that is a strict prefix of any
    /// Llama tag — i.e. text that might be the start of a tag continued next
    /// chunk and so must be held back rather than emitted. Capped at
    /// ``maxTagSuffix`` so a long chunk costs only a bounded tail scan.
    private static func partialTagSuffixLength(of text: String) -> Int {
        let tags = [openTag, eomTag, eotTag]
        let upper = min(maxTagSuffix, text.count)
        var best = 0
        var length = 1
        while length <= upper {
            let suffix = text.suffix(length)
            if tags.contains(where: { $0.count > length && $0.hasPrefix(suffix) }) {
                best = length
            }
            length += 1
        }
        return best
    }
}
