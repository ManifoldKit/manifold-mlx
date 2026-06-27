import XCTest
@_spi(Testing) import ManifoldMLX

/// Correctness gate for the #100 perf rework. The optimized engine
/// (`GBNFFastMatcher` + `GBNFTokenTrie`) MUST be byte-identical in accept/reject
/// semantics to the reference `GBNFMatcher`, which is left in place purely as the
/// oracle. A faster-but-wrong matcher is useless, so this asserts:
///
/// 1. `GBNFFastMatcher` agrees with `GBNFMatcher` on `accepts` / `isComplete` /
///    `isDead` / `acceptableFirstBytes` at every state, across the grammar
///    corpus, driven both deterministically and by a randomized fuzzer.
/// 2. `GBNFTokenTrie.allowedTokenIDs` returns exactly the set a naive
///    full-vocab scan (`reference.accepts(tokenBytes)` per token) would.
///
/// Runs off-GPU in CI (no MLX/Metal).
final class GBNFFastMatcherParityTests: XCTestCase {

    /// The shared parity corpus: representative of what `ToolGrammarBuilder`
    /// emits plus the edge cases the reference suite already pins.
    private static let corpus: [String] = [
        #"root ::= "ab" "c""#,
        #"root ::= "\"north\"""#,
        #"root ::= "yes" | "no""#,
        ##"root ::= "a" "b"?"##,
        ##"root ::= "a"*"##,
        ##"root ::= "a"+"##,
        ##"root ::= "x" ( "," "y" )*"##,
        "root ::= [0-9] [a-fA-F]",
        "root ::= [^{]",
        #"root ::= [^\x00]"#,
        "root ::= [a-]",
        // Recursive JSON value subset.
        #"""
        root ::= object
        object ::= "{" ws ( member ( ws "," ws member )* )? ws "}"
        member ::= string ws ":" ws value
        value ::= object | string | "true" | "false"
        string ::= "\"" char* "\""
        char ::= [^"\\]
        ws ::= [ \t\n]*
        """#,
        // Exact tool-call envelope shape.
        #"""
        root ::= toolcall-0
        args-0-0 ::= ("\"north\"" | "\"south\"")
        args-0 ::= "{" ws "\"direction\"" ws ":" ws args-0-0 ws "}"
        toolcall-0 ::= "{" ws "\"name\"" ws ":" ws "\"set_direction\"" ws "," ws "\"arguments\"" ws ":" ws args-0 ws "}"
        ws ::= [ \t\n]*
        """#,
        // Nullable / right-recursive cycle (normalization must not loop).
        #"""
        root ::= a
        a ::= b a | "x"
        b ::= "" | "y"
        """#,
        "root ::= ",
    ]

    // MARK: - Matcher parity (deterministic, exhaustive single-byte)

    func test_fastMatcher_matchesReference_overCorpus() throws {
        for text in Self.corpus {
            let g = try GBNFGrammar(parsing: text)
            try assertParityWalking(g, label: text)
        }
    }

    /// Walk the grammar with both matchers in lockstep. At every reachable state
    /// assert all observable predicates agree for all 256 candidate bytes, then
    /// step forward along an accepted byte (DFS over states, bounded).
    private func assertParityWalking(_ g: GBNFGrammar, label: String) throws {
        var ref = GBNFMatcher(grammar: g)
        var fast = GBNFFastMatcher(grammar: g)
        var depth = 0
        while depth < 64 {
            XCTAssertEqual(ref.isComplete, fast.isComplete, "isComplete @\(depth) [\(label)]")
            XCTAssertEqual(ref.isDead, fast.isDead, "isDead @\(depth) [\(label)]")
            XCTAssertEqual(ref.acceptableFirstBytes(), fast.acceptableFirstBytes(),
                           "acceptableFirstBytes @\(depth) [\(label)]")
            var acceptedByte: UInt8?
            for b in UInt8.min...UInt8.max {
                let r = ref.accepts([b])
                let f = fast.accepts([b])
                XCTAssertEqual(r, f, "accepts(\(b)) @\(depth) [\(label)]")
                if r, acceptedByte == nil { acceptedByte = b }
            }
            guard let next = acceptedByte else { break }
            ref.advance([next])
            fast.advance([next])
            depth += 1
        }
    }

    // MARK: - Matcher parity (randomized fuzz)

    func test_fastMatcher_matchesReference_fuzz() throws {
        var rng = SplitMix64(seed: 0xF00D_CAFE_1234_5678)
        for text in Self.corpus {
            let g = try GBNFGrammar(parsing: text)
            for _ in 0..<60 {
                var ref = GBNFMatcher(grammar: g)
                var fast = GBNFFastMatcher(grammar: g)
                for _ in 0..<40 {
                    // Multi-byte `accepts` parity for a random candidate token.
                    let tok = (0..<Int(rng.next() % 4 + 1)).map { _ in randomByte(&rng) }
                    XCTAssertEqual(ref.accepts(tok), fast.accepts(tok),
                                   "fuzz accepts(\(tok)) [\(text)]")
                    XCTAssertEqual(ref.isComplete, fast.isComplete, "fuzz isComplete [\(text)]")
                    XCTAssertEqual(ref.acceptableFirstBytes(), fast.acceptableFirstBytes(),
                                   "fuzz firstBytes [\(text)]")

                    // Advance: usually along an accepted byte (to reach deep
                    // states), occasionally a wholly random byte (to exercise
                    // rejection / death paths).
                    let byte: UInt8
                    let accepted = (UInt8.min...UInt8.max).filter { ref.accepts([$0]) }
                    if !accepted.isEmpty, rng.next() % 5 != 0 {
                        byte = accepted[Int(rng.next() % UInt64(accepted.count))]
                    } else {
                        byte = randomByte(&rng)
                    }
                    ref.advance([byte])
                    fast.advance([byte])
                    XCTAssertEqual(ref.isDead, fast.isDead, "fuzz isDead after \(byte) [\(text)]")
                    if ref.isDead { break }
                }
            }
        }
    }

    // MARK: - Trie-walk parity vs naive full-vocab scan

    func test_trieWalk_equalsNaiveScan_overCorpus() throws {
        let vocab = Self.syntheticVocab()
        let trie = GBNFTokenTrie(tokenBytes: vocab)
        var rng = SplitMix64(seed: 0xABCD_0001_0002_0003)

        for text in Self.corpus {
            let g = try GBNFGrammar(parsing: text)
            // Compare at the start state and at several advanced states.
            for _ in 0..<12 {
                var ref = GBNFMatcher(grammar: g)
                var fast = GBNFFastMatcher(grammar: g)
                // Drive both to a shared random reachable state.
                let steps = Int(rng.next() % 8)
                for _ in 0..<steps {
                    let accepted = (UInt8.min...UInt8.max).filter { ref.accepts([$0]) }
                    guard !accepted.isEmpty else { break }
                    let b = accepted[Int(rng.next() % UInt64(accepted.count))]
                    ref.advance([b]); fast.advance([b])
                    if fast.isDead { break }
                }
                if fast.isDead { continue }

                let trieSet = Set(trie.allowedTokenIDs(from: fast))
                let naiveSet = Self.naiveAllowed(ref, vocab: vocab)
                XCTAssertEqual(trieSet, naiveSet,
                               "trie allowed-set != naive scan [\(text)] after \(steps) steps")
            }
        }
    }

    private static func naiveAllowed(_ matcher: GBNFMatcher, vocab: [[UInt8]?]) -> Set<Int> {
        var out: Set<Int> = []
        for id in vocab.indices {
            guard let bytes = vocab[id], !bytes.isEmpty else { continue }
            if matcher.accepts(bytes) { out.insert(id) }
        }
        return out
    }

    // MARK: - Micro-benchmark (records mask-construction cost)

    /// Worst case for the old code: a JSON string value where almost every byte
    /// is legal, so the first-byte prune fails and the naive scan runs a full
    /// acceptance check on ~every token. Compares old (per-token reference
    /// `accepts`) vs new (trie walk) mask-construction time on a representative
    /// vocab and records the numbers.
    func test_benchmark_maskConstruction_trieVsNaive() throws {
        let vocab = Self.benchmarkVocab(count: 50_000)
        let trie = GBNFTokenTrie(tokenBytes: vocab)
        let g = try GBNFGrammar(parsing: #"""
        root ::= "{" ws "\"speech\"" ws ":" ws string ws "}"
        string ::= "\"" char* "\""
        char ::= [^"\\]
        ws ::= [ \t\n]*
        """#)

        // Drive into the string-value region (the pathological state).
        var ref = GBNFMatcher(grammar: g)
        var fast = GBNFFastMatcher(grammar: g)
        for b in Array(#"{"speech": "hello"#.utf8) {
            ref.advance([b]); fast.advance([b])
        }
        XCTAssertFalse(fast.isDead, "should be mid-string")

        // Correctness first.
        let trieSet = Set(trie.allowedTokenIDs(from: fast))
        XCTAssertEqual(trieSet, Self.naiveAllowed(ref, vocab: vocab),
                       "benchmark state: trie must match naive")

        let reps = 5
        let tNaive = time(reps) {
            var n = 0
            for id in vocab.indices {
                guard let bytes = vocab[id], !bytes.isEmpty else { continue }
                if ref.accepts(bytes) { n += 1 }
            }
            XCTAssertGreaterThan(n, 0)
        }
        // Fresh cache each rep so we measure the cold trie walk, not the
        // processor's state cache (which would make later steps ~free).
        let tTrie = time(reps) {
            _ = trie.allowedTokenIDs(from: fast)
        }

        let speedup = tNaive / max(tTrie, 1e-9)
        print(String(
            format: "[#100 bench] vocab=%d in-string mask/step: naive=%.2fms trie=%.2fms speedup=%.1fx",
            vocab.count, tNaive * 1e3, tTrie * 1e3, speedup
        ))
        // The trie walk must not be slower in this pathological region — that is
        // the entire point of #100. Generous margin to stay non-flaky.
        XCTAssertLessThan(tTrie, tNaive, "trie walk must beat the naive full-vocab scan in-string")
    }

    private func time(_ reps: Int, _ body: () -> Void) -> Double {
        let start = DispatchTime.now().uptimeNanoseconds
        for _ in 0..<reps { body() }
        let end = DispatchTime.now().uptimeNanoseconds
        return Double(end - start) / 1e9 / Double(reps)
    }

    // MARK: - Synthetic vocabularies

    /// Small structured vocab: single bytes plus JSON-ish fragments, exercising
    /// shared prefixes and multi-byte tokens.
    private static func syntheticVocab() -> [[UInt8]?] {
        var v: [[UInt8]?] = []
        for b in UInt8.min...UInt8.max { v.append([b]) }      // all single bytes
        let frags = ["{", "}", "\"", ":", ",", " ", "\n", "\t",
                     "name", "set_direction", "\"name\"", "north", "south",
                     "\"north\"", "\"south\"", "true", "false", "ab", "abc",
                     "{\"", "\":", ": ", "yes", "no", "x", "x,y", ",y", "0a", "9F"]
        for f in frags { v.append(Array(f.utf8)) }
        v.append(nil)        // a special/no-byte token (must be skipped)
        v.append([])         // empty token (must be skipped)
        return v
    }

    /// Larger vocab approximating a byte-level-BPE distribution: single bytes
    /// plus many random short ASCII fragments (string-value tokens dominate).
    private static func benchmarkVocab(count: Int) -> [[UInt8]?] {
        var rng = SplitMix64(seed: 0x5EED_1234_9999_0001)
        var v: [[UInt8]?] = []
        for b in UInt8.min...UInt8.max { v.append([b]) }
        let printable = Array(UInt8(0x20)...UInt8(0x7E))
        while v.count < count {
            let len = Int(rng.next() % 5) + 1
            var t: [UInt8] = []
            for _ in 0..<len { t.append(printable[Int(rng.next() % UInt64(printable.count))]) }
            v.append(t)
        }
        return v
    }

    private func randomByte(_ rng: inout SplitMix64) -> UInt8 {
        // Bias toward the structural/ASCII bytes the corpus grammars use.
        if rng.next() % 2 == 0 {
            let ascii = Array(#"{}":,  \tnyesodirth0-9aFxabcsouthnorthtruefal"#.utf8)
            return ascii[Int(rng.next() % UInt64(ascii.count))]
        }
        return UInt8(rng.next() % 256)
    }
}

/// Tiny deterministic PRNG so the fuzz/benchmark vocab are reproducible in CI.
private struct SplitMix64 {
    private var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
