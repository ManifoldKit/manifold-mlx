import Foundation
import MLXLMCommon

/// Recovers the *raw bytes* each token id denotes for a byte-level-BPE tokenizer
/// (GPT-2 / Qwen family), so the GBNF matcher — which works on bytes — can test
/// candidate tokens (#96, option B).
///
/// Byte-level BPE doesn't store raw bytes in the vocab: each byte is mapped to a
/// printable Unicode code point via the GPT-2 `bytes_to_unicode` table (so the
/// vocab is valid UTF-8 text). `convertIdToToken` returns a token in *that*
/// alphabet; this type inverts the mapping to get the original bytes.
///
/// Tokens that don't decode cleanly through the table (special tokens like
/// `<|im_end|>`, control tokens) map to `nil` — they have no grammar byte
/// representation and are never allowed mid-grammar (EOS is handled separately
/// by the processor).
struct MLXByteLevelVocabulary {
    /// `bytes[id]` = the raw bytes for token `id`, or `nil` if it has no
    /// byte-level representation (special/added token).
    let bytes: [[UInt8]?]

    /// Builds the table for `vocabSize` ids using `tokenizer.convertIdToToken`.
    /// `vocabSize` is taken from the model's logits width at generation time.
    init(tokenizer: any MLXLMCommon.Tokenizer, vocabSize: Int) {
        let decoder = Self.unicodeToByte
        var table: [[UInt8]?] = Array(repeating: nil, count: vocabSize)
        for id in 0..<vocabSize {
            guard let token = tokenizer.convertIdToToken(id) else { continue }
            var out: [UInt8] = []
            var ok = true
            for scalar in token.unicodeScalars {
                guard let b = decoder[scalar.value] else { ok = false; break }
                out.append(b)
            }
            table[id] = ok ? out : nil
        }
        bytes = table
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
