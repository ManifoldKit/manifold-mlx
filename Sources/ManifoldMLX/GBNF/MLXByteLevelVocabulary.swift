import Foundation
import MLXLMCommon

/// Recovers the *raw bytes* each token id denotes, so the GBNF matcher — which
/// works on bytes — can test candidate tokens (#96, option B).
///
/// Two vocabulary alphabets are supported, auto-detected per model:
///
/// - **GPT-2 / Qwen / Llama byte-level BPE** (``Style/gpt2ByteLevel``): each
///   byte is mapped to a printable Unicode code point via the GPT-2
///   `bytes_to_unicode` table (so the vocab is valid UTF-8 text).
///   `convertIdToToken` returns a token in *that* alphabet; this type inverts
///   the mapping to get the original bytes.
///
/// - **SentencePiece / Metaspace BPE** (``Style/sentencePieceMetaspace``, e.g.
///   Mistral / Mixtral / Ministral, #106): tokens are literal UTF-8 substrings
///   with the space marker `▁` (U+2581) standing in for a leading space, plus
///   `<0xHH>` byte-fallback tokens for raw bytes. The GPT-2 table has *no* entry
///   for U+2581, so under the GPT-2 path **every** space-bearing token decodes to
///   `nil` and the grammar masks almost the entire vocabulary — collapsing
///   constrained Mistral decoding onto an empty/EOS output. This style maps `▁`
///   → `0x20`, `<0xHH>` → that byte, and every other scalar → its UTF-8 bytes.
///
/// Tokens that don't decode cleanly map to `nil` — they have no grammar byte
/// representation and are never allowed mid-grammar (EOS is handled separately
/// by the processor).
@_spi(Testing) public struct MLXByteLevelVocabulary {

    /// The vocabulary alphabet, auto-detected from the token strings.
    @_spi(Testing) public enum Style: Equatable, Sendable {
        /// GPT-2 `bytes_to_unicode` byte-level BPE (GPT-2 / Qwen / Llama-3).
        case gpt2ByteLevel
        /// SentencePiece / Metaspace BPE with the `▁` (U+2581) space marker
        /// (Mistral / Mixtral / Ministral).
        case sentencePieceMetaspace
    }

    /// `bytes[id]` = the raw bytes for token `id`, or `nil` if it has no
    /// byte-level representation (special/added token, or an undecodable token).
    let bytes: [[UInt8]?]

    /// The alphabet that was detected and used to build ``bytes``. Exposed for
    /// diagnostics/tests.
    let style: Style

    /// Builds the table for `vocabSize` ids using `tokenizer.convertIdToToken`.
    /// `vocabSize` is taken from the model's logits width at generation time. The
    /// alphabet is auto-detected from the token strings (see ``Style``).
    init(tokenizer: any MLXLMCommon.Tokenizer, vocabSize: Int) {
        var tokens: [String?] = Array(repeating: nil, count: vocabSize)
        for id in 0..<vocabSize {
            tokens[id] = tokenizer.convertIdToToken(id)
        }
        let style = Self.detectStyle(tokens)
        self.style = style
        self.bytes = tokens.map { token in
            token.flatMap { Self.recoverBytes($0, style: style) }
        }
    }

    /// Detects the vocabulary alphabet from a sample of token strings.
    ///
    /// Presence of the Metaspace marker `▁` (U+2581) — which GPT-2/Qwen/Llama
    /// vocabularies never use (they use `Ġ`, U+0120, for a leading space) — is a
    /// reliable discriminator for SentencePiece/Metaspace BPE.
    @_spi(Testing) public static func detectStyle(_ tokens: [String?]) -> Style {
        for case let token? in tokens {
            for scalar in token.unicodeScalars where scalar.value == 0x2581 {
                return .sentencePieceMetaspace
            }
        }
        return .gpt2ByteLevel
    }

    /// Recovers the raw bytes a single token string denotes under `style`, or
    /// `nil` when it has no clean byte-level representation.
    @_spi(Testing) public static func recoverBytes(_ token: String, style: Style) -> [UInt8]? {
        switch style {
        case .gpt2ByteLevel:
            return gpt2Bytes(token)
        case .sentencePieceMetaspace:
            return metaspaceBytes(token)
        }
    }

    // MARK: - GPT-2 byte-level

    private static func gpt2Bytes(_ token: String) -> [UInt8]? {
        let decoder = Self.unicodeToByte
        var out: [UInt8] = []
        for scalar in token.unicodeScalars {
            guard let b = decoder[scalar.value] else { return nil }
            out.append(b)
        }
        return out
    }

    // MARK: - SentencePiece / Metaspace

    /// Recovers bytes for a SentencePiece/Metaspace token:
    /// - a `<0xHH>` byte-fallback token → that single raw byte;
    /// - the metaspace marker `▁` (U+2581) → an ASCII space (`0x20`);
    /// - every other scalar → its UTF-8 bytes (tokens are literal text).
    static func metaspaceBytes(_ token: String) -> [UInt8]? {
        if let raw = byteFallback(token) { return [raw] }
        var out: [UInt8] = []
        for scalar in token.unicodeScalars {
            if scalar.value == 0x2581 {
                out.append(0x20)
            } else {
                out.append(contentsOf: Array(String(scalar).utf8))
            }
        }
        return out
    }

    /// Parses a `<0xHH>` byte-fallback token into its raw byte, or `nil` if the
    /// token is not in that exact form.
    private static func byteFallback(_ token: String) -> UInt8? {
        let scalars = Array(token.unicodeScalars)
        guard scalars.count == 6,
              scalars[0] == "<", scalars[1] == "0",
              (scalars[2] == "x" || scalars[2] == "X"),
              scalars[5] == ">",
              let hi = Character(scalars[3]).hexDigitValue,
              let lo = Character(scalars[4]).hexDigitValue
        else {
            return nil
        }
        return UInt8(hi * 16 + lo)
    }

    /// Inverse of GPT-2 `bytes_to_unicode`: Unicode scalar value → original byte.
    /// Built once; mirrors the canonical reference implementation.
    static let unicodeToByte: [UInt32: UInt8] = {
        // Printable ASCII/Latin-1 ranges map to themselves; the remaining bytes
        // are remapped to U+0100… in order.
        var byteToUnicode: [UInt8: UInt32] = [:]
        var printable: [UInt8] = []
        func appendRange(_ lo: UInt32, _ hi: UInt32) {
            for v in lo...hi { printable.append(UInt8(v)) }
        }
        appendRange(UInt32(UInt8(ascii: "!")), UInt32(UInt8(ascii: "~")))
        appendRange(0xA1, 0xAC)
        appendRange(0xAE, 0xFF)
        for b in printable { byteToUnicode[b] = UInt32(b) }

        var n: UInt32 = 0
        for b in 0...255 {
            let byte = UInt8(b)
            if byteToUnicode[byte] == nil {
                byteToUnicode[byte] = 256 + n
                n += 1
            }
        }
        var inverse: [UInt32: UInt8] = [:]
        for (byte, scalar) in byteToUnicode { inverse[scalar] = byte }
        return inverse
    }()
}
