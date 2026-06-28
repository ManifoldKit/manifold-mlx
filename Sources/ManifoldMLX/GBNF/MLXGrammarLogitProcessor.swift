import Foundation
import MLX
import MLXLMCommon

/// Grammar-constrained sampling for MLX (#96, option B): a `LogitProcessor` that
/// masks, at every step, every token whose bytes the ``GBNFMatcher`` would
/// reject — so generation can only follow paths the grammar permits.
///
/// Composes over the parameters' penalty processor (`base`): penalties are
/// applied first, then the grammar mask, so repetition/presence shaping still
/// acts among the grammar-legal tokens.
///
/// The token→bytes table (``MLXByteLevelVocabulary``) is built lazily on the
/// first `process(logits:)` call, when the logits width reveals the vocab size.
///
/// ## Performance
///
/// The additive grammar mask is a pure function of the matcher's current state
/// (its set of live stacks), the vocabulary, and the EOS ids — none of which
/// change mid-step. So the mask is **memoized by matcher state**: the expensive
/// O(vocab) acceptance scan runs once per *distinct* grammar state, and every
/// later decode step that revisits that state (every step inside a JSON string
/// charset shares one state, as do repeated structural contexts) reuses the
/// cached `MLXArray` with no Swift-side scan at all. On a cache miss the scan
/// itself is cheaper than a per-token `accepts` loop because
/// ``GBNFMatcher/allowedTokenIDs(byteTable:)`` hoists the leading-byte
/// transition (≤256 evaluations instead of ~vocab). The remaining trie/
/// prefix-pruning pass for the cache-miss scan is tracked in #97.
final class MLXGrammarLogitProcessor: LogitProcessor {
    private var matcher: GBNFMatcher
    private let tokenizer: any MLXLMCommon.Tokenizer
    private let eosIds: Set<Int>
    private var base: LogitProcessor?

    /// Lazily built once the vocab size is known.
    private var vocab: MLXByteLevelVocabulary?

    /// Memoized additive masks keyed by matcher state. Bounded so a long
    /// structured-output generation with many distinct states cannot grow the
    /// cache without limit; past the cap, masks are recomputed but not stored.
    private var maskCache: [Set<[GBNFGrammar.Symbol]>: MLXArray] = [:]
    private static let maskCacheCap = 512

    init(grammar: GBNFGrammar, tokenizer: any MLXLMCommon.Tokenizer, base: LogitProcessor?) {
        self.matcher = GBNFMatcher(grammar: grammar)
        self.tokenizer = tokenizer
        self.base = base
        var eos: Set<Int> = []
        if let id = tokenizer.eosTokenId { eos.insert(id) }
        self.eosIds = eos
    }

    // MARK: - LogitProcessor

    func prompt(_ prompt: MLXArray) {
        base?.prompt(prompt)
    }

    func process(logits: MLXArray) -> MLXArray {
        let penalised = base?.process(logits: logits) ?? logits
        let vocabSize = penalised.shape.last ?? penalised.size
        let table = vocabBytes(vocabSize: vocabSize)

        let mask = maskArray(vocabSize: vocabSize, table: table)
        return penalised + mask.reshaped(penalised.shape)
    }

    /// The additive grammar mask (`0` for legal tokens, `-1e9` otherwise) for the
    /// matcher's current state — served from cache when the state recurs.
    private func maskArray(vocabSize: Int, table: MLXByteLevelVocabulary) -> MLXArray {
        let key = matcher.stacks
        if let cached = maskCache[key] { return cached }

        let allowed = matcher.allowedTokenIDs(byteTable: table.bytes)
        // EOS is allowed exactly when the grammar can terminate here.
        let allowEOS = matcher.isComplete

        var mask = [Float](repeating: -1e9, count: vocabSize)
        var anyAllowed = false
        for id in allowed where !eosIds.contains(id) {
            // EOS ids are governed solely by `allowEOS` below (an EOS token has
            // no grammar-byte representation, so it normally never appears in
            // `allowed`; the guard keeps the invariant even if one does).
            mask[id] = 0
            anyAllowed = true
        }
        if allowEOS {
            for id in eosIds { mask[id] = 0; anyAllowed = true }
        }
        // Safety valve: if nothing is allowed (a grammar/vocab mismatch that
        // would otherwise deadlock the sampler), permit EOS so generation can
        // end cleanly rather than emitting an out-of-grammar token or hanging.
        if !anyAllowed {
            for id in eosIds { mask[id] = 0 }
        }

        let built = MLXArray(mask)
        if maskCache.count < Self.maskCacheCap {
            maskCache[key] = built
        }
        return built
    }

    func didSample(token: MLXArray) {
        base?.didSample(token: token)
        let id = token.item(Int.self)
        if eosIds.contains(id) { return }
        if let table = vocab, let tokenBytes = table.bytes[id] {
            matcher.advance(tokenBytes)
        }
    }

    // MARK: - Helpers

    private func vocabBytes(vocabSize: Int) -> MLXByteLevelVocabulary {
        if let vocab, vocab.bytes.count == vocabSize { return vocab }
        let built = MLXByteLevelVocabulary(tokenizer: tokenizer, vocabSize: vocabSize)
        vocab = built
        return built
    }
}
