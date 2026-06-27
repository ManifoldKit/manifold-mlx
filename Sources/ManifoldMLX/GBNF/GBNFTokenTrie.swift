import Foundation

/// Byte trie over the model vocabulary, built once at load, so grammar masking
/// can walk *shared token prefixes a single time* per decode step instead of
/// re-deriving acceptance for every token independently (#100).
///
/// The reference processor tested all ~vocab tokens per step with an
/// O(token-bytes) acceptance check each. Inside a JSON string value almost every
/// byte is legal, so the cheap first-byte prune fails and tens of thousands of
/// full checks run per token — the ~2 min/turn pathology on large vocabs.
///
/// Walking the grammar over this trie turns that into ~O(reachable trie nodes):
/// the matcher transition for a shared prefix (e.g. the bytes common to many
/// tokens) is computed once and reused for every token beneath it. This is the
/// approach llama.cpp / XGrammar / Outlines take.
@_spi(Testing) public final class GBNFTokenTrie {
    /// `children[node][byte]` = child node index for that outgoing byte.
    private var children: [[UInt8: Int]] = [[:]]
    /// `terminals[node]` = token ids whose byte string ends exactly at `node`.
    private var terminals: [[Int]] = [[]]

    /// Builds the trie from a byte-level vocabulary. Tokens with no byte-level
    /// representation (special/added tokens, `nil` bytes) and empty tokens are
    /// excluded — EOS and friends are handled separately by the processor.
    convenience init(vocab: MLXByteLevelVocabulary) {
        self.init(tokenBytes: vocab.bytes)
    }

    /// Testing entry point: build directly from a token→bytes table.
    @_spi(Testing) public init(tokenBytes: [[UInt8]?]) {
        for id in tokenBytes.indices {
            guard let bytes = tokenBytes[id], !bytes.isEmpty else { continue }
            insert(bytes, id: id)
        }
    }

    private func insert(_ bytes: [UInt8], id: Int) {
        var node = 0
        for b in bytes {
            if let next = children[node][b] {
                node = next
            } else {
                let new = children.count
                children.append([:])
                terminals.append([])
                children[node][b] = new
                node = new
            }
        }
        terminals[node].append(id)
    }

    /// Cache key for a single byte transition out of a matcher state. Hashing a
    /// `Set<[Int]>` is cheap (integer payloads), so memoizing transitions across
    /// the many trie nodes that share a state — e.g. every node inside a JSON
    /// string value sits in the same `char*` loop state — collapses the walk.
    private struct TransitionKey: Hashable {
        let state: Set<[Int]>
        let byte: UInt8
    }

    /// All token ids whose bytes are a valid grammar continuation from the
    /// matcher's current state. Byte-identical to scanning the whole vocab and
    /// calling `matcher.accepts(tokenBytes)` on each (proven by the parity
    /// corpus), but computed in ~O(reachable nodes).
    @_spi(Testing) public func allowedTokenIDs(from matcher: GBNFFastMatcher) -> [Int] {
        var ids: [Int] = []
        var memo: [TransitionKey: Set<[Int]>] = [:]
        walk(node: 0, state: matcher.state, matcher: matcher, ids: &ids, memo: &memo)
        return ids
    }

    private func walk(
        node: Int,
        state: Set<[Int]>,
        matcher: GBNFFastMatcher,
        ids: inout [Int],
        memo: inout [TransitionKey: Set<[Int]>]
    ) {
        for (byte, child) in children[node] {
            let key = TransitionKey(state: state, byte: byte)
            let next: Set<[Int]>
            if let cached = memo[key] {
                next = cached
            } else {
                let computed = matcher.step(state, byte: byte)
                memo[key] = computed
                next = computed
            }
            // A dead transition means no token through this byte can be legal —
            // prune the whole subtree (mirrors `accepts` bailing on empty state).
            if next.isEmpty { continue }
            // Any token ending here consumed all its bytes with the state still
            // alive, i.e. it is an accepted prefix → allowed.
            if !terminals[child].isEmpty { ids.append(contentsOf: terminals[child]) }
            if !children[child].isEmpty {
                walk(node: child, state: next, matcher: matcher, ids: &ids, memo: &memo)
            }
        }
    }
}
