import XCTest
@_spi(Testing) import ManifoldMLX

/// Off-GPU unit tests for ``MLXByteLevelVocabulary`` byte recovery (#96, #106).
///
/// The decode-time table is what the GBNF matcher tests candidate tokens
/// against. Mistral/Mixtral ship a SentencePiece/Metaspace BPE vocab whose space
/// marker `▁` (U+2581) has no GPT-2 `bytes_to_unicode` entry — under the GPT-2
/// path every space-bearing token decodes to `nil` and the grammar masks almost
/// the whole vocab, collapsing constrained Mistral decoding onto EOS (#106).
/// These tests pin the per-style recovery and the auto-detection that routes
/// Mistral onto the Metaspace path while leaving GPT-2/Qwen/Llama unchanged.
final class MLXByteLevelVocabularyTests: XCTestCase {

    // MARK: - Style detection

    func test_detect_metaspace_whenMarkerPresent() {
        // A `▁`-prefixed token is the Metaspace signature.
        let tokens: [String?] = ["{", "\u{2581}the", "name"]
        XCTAssertEqual(MLXByteLevelVocabulary.detectStyle(tokens), .sentencePieceMetaspace)
    }

    func test_detect_gpt2_whenNoMetaspaceMarker() {
        // GPT-2/Qwen/Llama use `Ġ` (U+0120) for a leading space, never `▁`.
        let tokens: [String?] = ["{", "\u{0120}the", "name", nil]
        XCTAssertEqual(MLXByteLevelVocabulary.detectStyle(tokens), .gpt2ByteLevel)
    }

    // MARK: - Metaspace recovery

    func test_metaspace_marker_decodesToSpace() {
        XCTAssertEqual(
            MLXByteLevelVocabulary.recoverBytes("\u{2581}", style: .sentencePieceMetaspace),
            [0x20]
        )
    }

    func test_metaspace_spacePrefixedWord() {
        // `▁the` → " the"
        XCTAssertEqual(
            MLXByteLevelVocabulary.recoverBytes("\u{2581}the", style: .sentencePieceMetaspace),
            [0x20, 0x74, 0x68, 0x65]
        )
    }

    func test_metaspace_asciiPunctuationIsLiteral() {
        XCTAssertEqual(MLXByteLevelVocabulary.recoverBytes("{", style: .sentencePieceMetaspace), [0x7B])
        XCTAssertEqual(MLXByteLevelVocabulary.recoverBytes("\"", style: .sentencePieceMetaspace), [0x22])
    }

    func test_metaspace_toolCallsSentinelToken() {
        // The `[TOOL_CALLS]` special token must decode to its literal ASCII bytes
        // so the grammar's `"[TOOL_CALLS] "` literal can match it.
        let bytes = MLXByteLevelVocabulary.recoverBytes("[TOOL_CALLS]", style: .sentencePieceMetaspace)
        XCTAssertEqual(bytes.map { String(decoding: $0, as: UTF8.self) }, "[TOOL_CALLS]")
    }

    func test_metaspace_byteFallbackToken() {
        XCTAssertEqual(
            MLXByteLevelVocabulary.recoverBytes("<0x0A>", style: .sentencePieceMetaspace),
            [0x0A]
        )
        XCTAssertEqual(
            MLXByteLevelVocabulary.recoverBytes("<0xC3>", style: .sentencePieceMetaspace),
            [0xC3]
        )
    }

    func test_metaspace_nonAsciiIsUtf8() {
        // SentencePiece tokens are literal UTF-8 text outside the marker/fallback.
        XCTAssertEqual(
            MLXByteLevelVocabulary.recoverBytes("é", style: .sentencePieceMetaspace),
            Array("é".utf8)
        )
    }

    // MARK: - GPT-2 recovery (no regression)

    func test_gpt2_leadingSpaceMarkerDecodesToSpace() {
        // GPT-2 maps space (0x20) → U+0120 (Ġ); recovery inverts that.
        XCTAssertEqual(MLXByteLevelVocabulary.recoverBytes("\u{0120}", style: .gpt2ByteLevel), [0x20])
    }

    func test_gpt2_asciiIsLiteral() {
        XCTAssertEqual(MLXByteLevelVocabulary.recoverBytes("a", style: .gpt2ByteLevel), [0x61])
        XCTAssertEqual(MLXByteLevelVocabulary.recoverBytes("{", style: .gpt2ByteLevel), [0x7B])
    }

    func test_gpt2_metaspaceMarkerIsUndecodable() {
        // The Metaspace marker has no GPT-2 byte-unicode entry → nil (the #106
        // failure mode when the wrong style is applied to a Mistral vocab).
        XCTAssertNil(MLXByteLevelVocabulary.recoverBytes("\u{2581}", style: .gpt2ByteLevel))
    }
}
