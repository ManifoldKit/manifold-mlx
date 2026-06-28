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
/// ## Performance (#100, #114)
///
/// The original implementation scanned the entire vocabulary every step and ran
/// an allocation-heavy acceptance check per token — ~2 min/turn on a 150k-vocab
/// model, and minutes even at 32k. Four changes remove the full-vocab scan:
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
/// 4. **On-device mask** (#114): instead of building a full-vocab CPU float
///    array and uploading it each step, start with an on-device `full(-1e9)`
///    array and scatter-zero only the allowed positions via an integer-index
///    write. For 32k vocab this eliminates ~128 KB/step of CPU→GPU transfer.
///
/// Accept/reject semantics are byte-identical to the reference ``GBNFMatcher``
/// (`GBNFFastMatcherParityTests`).
@_spi(Testing) public final class MLXGrammarLogitProcessor: LogitProcessor {
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

    /// Circuit breaker: if the grammar is still open after this many sampled
    /// tokens, force EOS so a looping model can't generate to `maxTokens` under
    /// the constrained path (prose runaway, #114).
    private var breaker: GrammarStepBreaker

    @_spi(Testing) public init(
        grammar: GBNFGrammar,
        tokenizer: any MLXLMCommon.Tokenizer,
        base: LogitProcessor?,
        maxConstrainedSteps: Int = 512
    ) {
        self.matcher = GBNFFastMatcher(grammar: grammar)
        self.tokenizer = tokenizer
        self.base = base
        self.breaker = GrammarStepBreaker(maxSteps: maxConstrainedSteps)
        var eos: Set<Int> = []
        if let id = tokenizer.eosTokenId { eos.insert(id) }
        self.eosIds = eos
    }

    // MARK: - LogitProcessor

    public func prompt(_ prompt: MLXArray) {
        base?.prompt(prompt)
    }

    public func process(logits: MLXArray) -> MLXArray {
        let penalised = base?.process(logits: logits) ?? logits
        let vocabSize = penalised.shape.last ?? penalised.size
        _ = vocabBytes(vocabSize: vocabSize)

        breaker.advance()

        // Circuit breaker: grammar still open after the step cap — force EOS.
        if breaker.capExceeded && !matcher.isComplete {
            return eosOnlyMask(penalised: penalised, vocabSize: vocabSize)
        }

        let state = matcher.state
        let result: (ids: [Int], allowEOS: Bool)
        if let cached = maskCache[state] {
            result = cached
        } else {
            let ids = trie!.allowedTokenIDs(from: matcher)
            result = (ids, matcher.isComplete)
            maskCache[state] = result
        }

        // Collect all allowed positions (non-EOS tokens + EOS if grammar permits).
        var allowed = result.ids.filter { $0 < vocabSize }
        if result.allowEOS {
            allowed += eosIds.filter { $0 < vocabSize }
        }

        // Safety valve: grammar/vocab mismatch — nothing legal, permit EOS so
        // generation ends cleanly rather than deadlocking the sampler.
        if allowed.isEmpty {
            allowed = Array(eosIds.filter { $0 < vocabSize })
        }

        return penalised + buildMask(allowed: allowed, vocabSize: vocabSize, shape: penalised.shape)
    }

    public func didSample(token: MLXArray) {
        base?.didSample(token: token)
        let id = token.item(Int.self)
        if eosIds.contains(id) { return }
        if let table = vocab, id < table.bytes.count, let tokenBytes = table.bytes[id] {
            matcher.advance(tokenBytes)
        }
    }

    // MARK: - Helpers

    /// Build an on-device mask: full-vocab `-1e9`, with `allowed` positions zeroed.
    private func buildMask(allowed: [Int], vocabSize: Int, shape: [Int]) -> MLXArray {
        var mask = MLXArray.full([vocabSize], values: MLXArray(Float(-1e9)))
        if !allowed.isEmpty {
            let indices = MLXArray(allowed.map { Int32($0) })
            mask[indices] = MLXArray(Float(0))
        }
        return mask.reshaped(shape)
    }

    /// Mask allowing only EOS — used by the circuit breaker.
    private func eosOnlyMask(penalised: MLXArray, vocabSize: Int) -> MLXArray {
        let allowed = Array(eosIds.filter { $0 < vocabSize })
        return penalised + buildMask(allowed: allowed, vocabSize: vocabSize, shape: penalised.shape)
    }

    private func vocabBytes(vocabSize: Int) -> MLXByteLevelVocabulary {
        if let vocab, vocab.bytes.count == vocabSize { return vocab }
        let built = MLXByteLevelVocabulary(tokenizer: tokenizer, vocabSize: vocabSize)
        vocab = built
        trie = GBNFTokenTrie(vocab: built)
        return built
    }
}
