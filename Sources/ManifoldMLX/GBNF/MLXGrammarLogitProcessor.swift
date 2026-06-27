import Foundation
import MLX
import MLXLMCommon

/// Grammar-constrained sampling for MLX (#96, option B): a `LogitProcessor` that
/// masks, at every step, every token whose bytes the grammar would reject — so
/// generation can only follow paths the grammar permits.
///
/// Composes over the parameters' penalty processor (`base`): penalties are
/// applied first, then the grammar mask, so repetition/presence shaping still
/// acts among the grammar-legal tokens.
///
/// ## Performance (#100)
///
/// The original implementation scanned the entire vocabulary every step and ran
/// an allocation-heavy acceptance check per token — ~2 min/turn on a 150k-vocab
/// model, and minutes even at 32k. Three changes remove the full-vocab scan:
///
/// 1. **Token trie + grammar walk** (``GBNFTokenTrie``): shared token prefixes
///    are tested once per step, turning O(vocab × accept) into ~O(reachable
///    nodes).
/// 2. **State→mask cache**: the allowed-token set depends only on the matcher's
///    stack state, and states recur across steps (e.g. every byte inside a JSON
///    string value sits in the same loop state), so the computed id list is
///    memoized by state.
/// 3. **Cheaper matcher core** (``GBNFFastMatcher``): integer stack positions
///    into a flattened grammar, so transitions no longer hash `CharSet`s.
///
/// Accept/reject semantics are byte-identical to the reference ``GBNFMatcher``
/// (`GBNFFastMatcherParityTests`).
final class MLXGrammarLogitProcessor: LogitProcessor {
    private var matcher: GBNFFastMatcher
    private let tokenizer: any MLXLMCommon.Tokenizer
    private let eosIds: Set<Int>
    private var base: LogitProcessor?

    /// Lazily built once the vocab size is known.
    private var vocab: MLXByteLevelVocabulary?
    private var trie: GBNFTokenTrie?

    /// `matcher state → (allowed token ids, EOS allowed)`. The masking result is
    /// a pure function of the matcher state, and states recur across steps, so
    /// the (expensive, first-time-only) trie walk is reused.
    private var maskCache: [Set<[Int]>: (ids: [Int], allowEOS: Bool)] = [:]

    init(grammar: GBNFGrammar, tokenizer: any MLXLMCommon.Tokenizer, base: LogitProcessor?) {
        self.matcher = GBNFFastMatcher(grammar: grammar)
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
        _ = vocabBytes(vocabSize: vocabSize)
        let trie = self.trie!

        let state = matcher.state
        let result: (ids: [Int], allowEOS: Bool)
        if let cached = maskCache[state] {
            result = cached
        } else {
            let ids = trie.allowedTokenIDs(from: matcher)
            result = (ids, matcher.isComplete)
            maskCache[state] = result
        }

        var mask = [Float](repeating: -1e9, count: vocabSize)
        var anyAllowed = false
        for id in result.ids where id < vocabSize {
            mask[id] = 0
            anyAllowed = true
        }
        // EOS is allowed exactly when the grammar can terminate here.
        if result.allowEOS {
            for id in eosIds where id < vocabSize { mask[id] = 0; anyAllowed = true }
        }

        // Safety valve: if nothing is allowed (a grammar/vocab mismatch that
        // would otherwise deadlock the sampler), permit EOS so generation can
        // end cleanly rather than emitting an out-of-grammar token or hanging.
        if !anyAllowed {
            for id in eosIds where id < vocabSize { mask[id] = 0 }
        }

        return penalised + MLXArray(mask).reshaped(penalised.shape)
    }

    func didSample(token: MLXArray) {
        base?.didSample(token: token)
        let id = token.item(Int.self)
        if eosIds.contains(id) { return }
        if let table = vocab, id < table.bytes.count, let tokenBytes = table.bytes[id] {
            matcher.advance(tokenBytes)
        }
    }

    // MARK: - Helpers

    private func vocabBytes(vocabSize: Int) -> MLXByteLevelVocabulary {
        if let vocab, vocab.bytes.count == vocabSize { return vocab }
        let built = MLXByteLevelVocabulary(tokenizer: tokenizer, vocabSize: vocabSize)
        vocab = built
        trie = GBNFTokenTrie(vocab: built)
        return built
    }
}
