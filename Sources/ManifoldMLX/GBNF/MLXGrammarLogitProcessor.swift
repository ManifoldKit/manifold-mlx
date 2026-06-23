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
/// Performance note: the per-step check is O(vocab × token-bytes) with a
/// first-byte prune. That is correct but not yet optimized for large vocabs —
/// the trie/prefix-pruning pass is tracked in #97.
final class MLXGrammarLogitProcessor: LogitProcessor {
    private var matcher: GBNFMatcher
    private let tokenizer: any MLXLMCommon.Tokenizer
    private let eosIds: Set<Int>
    private var base: LogitProcessor?

    /// Lazily built once the vocab size is known.
    private var vocab: MLXByteLevelVocabulary?

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

        // EOS is allowed exactly when the grammar can terminate here.
        let allowEOS = matcher.isComplete
        let acceptableFirst = matcher.acceptableFirstBytes()

        var mask = [Float](repeating: -1e9, count: vocabSize)
        var anyAllowed = false
        for id in 0..<vocabSize {
            if eosIds.contains(id) {
                if allowEOS { mask[id] = 0; anyAllowed = true }
                continue
            }
            guard let tokenBytes = table.bytes[id], let first = tokenBytes.first else { continue }
            // Cheap reject: a token whose first byte no live stack accepts cannot
            // be legal — skip the full (copying) acceptance check.
            guard acceptableFirst.contains(first) else { continue }
            if matcher.accepts(tokenBytes) {
                mask[id] = 0
                anyAllowed = true
            }
        }

        // Safety valve: if nothing is allowed (a grammar/vocab mismatch that
        // would otherwise deadlock the sampler), permit EOS so generation can
        // end cleanly rather than emitting an out-of-grammar token or hanging.
        if !anyAllowed {
            for id in eosIds { mask[id] = 0 }
        }

        return penalised + MLXArray(mask).reshaped(penalised.shape)
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
