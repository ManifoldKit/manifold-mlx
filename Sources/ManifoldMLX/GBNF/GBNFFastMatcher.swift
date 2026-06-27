import Foundation

/// Compiled, flattened form of a ``GBNFGrammar`` for the fast matcher (#100).
///
/// The reference ``GBNFMatcher`` represents matcher state as a `Set<[Symbol]>`
/// — sets of symbol arrays whose elements carry `CharSet` values. Hashing and
/// copying those arrays (especially the `CharSet` payloads) dominates the
/// per-token acceptance check, which runs O(vocab) times per decode step.
///
/// `GBNFCompiled` lays every alternative of every rule out contiguously into a
/// single `positions` array. A *position* is just an `Int` index into that
/// array; the symbol that follows in the same alternative is always at
/// `index + 1`, and an `.end` marker terminates each alternative. A matcher
/// "stack" is therefore an `[Int]` (cheap to hash/copy), and the live state is a
/// `Set<[Int]>` — semantically identical to the reference's `Set<[Symbol]>` but
/// with integer payloads, so `step`/`normalize` stop hashing `CharSet`s.
final class GBNFCompiled {
    enum Pos {
        case charset(GBNFGrammar.CharSet)
        case rule(Int)
        /// One-past-the-last symbol of an alternative (the alternative completes
        /// when this position is reached).
        case end
    }

    /// Flattened symbols for every alternative of every rule, each alternative
    /// followed by an `.end` marker.
    let positions: [Pos]
    /// `altStarts[ruleId]` = the start `positions` index of each alternative of
    /// rule `ruleId`.
    let altStarts: [[Int]]
    let rootId: Int

    init(_ grammar: GBNFGrammar) {
        var positions: [Pos] = []
        var altStarts: [[Int]] = Array(repeating: [], count: grammar.rules.count)
        for ruleId in grammar.rules.indices {
            var starts: [Int] = []
            for alt in grammar.rules[ruleId] {
                starts.append(positions.count)
                for sym in alt {
                    switch sym {
                    case .charset(let cs): positions.append(.charset(cs))
                    case .rule(let id): positions.append(.rule(id))
                    }
                }
                positions.append(.end)
            }
            altStarts[ruleId] = starts
        }
        self.positions = positions
        self.altStarts = altStarts
        self.rootId = grammar.rootId
    }
}

/// Allocation-light incremental byte matcher over a ``GBNFCompiled`` grammar
/// (#100). Byte-for-byte equivalent to ``GBNFMatcher`` (proven by the parity
/// corpus in `GBNFFastMatcherParityTests`), but with integer stack positions so
/// the masking hot path is no longer dominated by `CharSet` hashing.
///
/// A stack is an `[Int]` of grammar positions; the *top* (next symbol to match)
/// is the **last** element so it can be advanced/popped in O(1). Stacks are kept
/// normalized: every non-empty stack's top is a `.charset`. The empty stack
/// (`[]`) means the root derivation can complete here (EOS is allowed).
@_spi(Testing) public struct GBNFFastMatcher {
    let compiled: GBNFCompiled
    private var stacks: Set<[Int]>

    public init(grammar: GBNFGrammar) {
        let c = GBNFCompiled(grammar)
        self.compiled = c
        self.stacks = Self.normalize(c.altStarts[c.rootId].map { [$0] }, compiled: c)
    }

    /// The live state — exposed so the token trie can walk it (#100).
    @_spi(Testing) public var state: Set<[Int]> { stacks }

    /// True when no stack survives.
    public var isDead: Bool { stacks.isEmpty }

    /// True when the root derivation can terminate here (EOS permitted).
    public var isComplete: Bool { stacks.contains([]) }

    /// Whole-string acceptance — test/diagnostic convenience mirroring
    /// ``GBNFMatcher/matches(_:grammar:)``.
    public static func matches(_ text: String, grammar: GBNFGrammar) -> Bool {
        var m = GBNFFastMatcher(grammar: grammar)
        for b in Array(text.utf8) {
            guard m.accepts([b]) else { return false }
            m.advance([b])
        }
        return m.isComplete
    }

    /// The set of byte values at least one live stack accepts next. Off the hot
    /// path now (the trie prunes via `step`); kept for parity and the backend
    /// guard.
    public func acceptableFirstBytes() -> Set<UInt8> {
        var bytes: Set<UInt8> = []
        for s in stacks {
            guard let top = s.last, case let .charset(cs) = compiled.positions[top] else { continue }
            var b = 0
            while b <= 255 {
                if cs.contains(UInt8(b)) { bytes.insert(UInt8(b)) }
                b += 1
            }
        }
        return bytes
    }

    /// Whether `bytes` form a valid prefix of some grammar continuation from the
    /// current state (non-mutating). Equivalent to ``GBNFMatcher/accepts(_:)``.
    public func accepts(_ bytes: [UInt8]) -> Bool {
        var cur = stacks
        for b in bytes {
            cur = Self.step(cur, byte: b, compiled: compiled)
            if cur.isEmpty { return false }
        }
        return true
    }

    /// Commit the sampled token's bytes, moving the live state forward.
    public mutating func advance(_ bytes: [UInt8]) {
        for b in bytes { stacks = Self.step(stacks, byte: b, compiled: compiled) }
    }

    /// One byte transition from an arbitrary state — used by the trie walk so it
    /// can fan out over many candidate continuations without mutating `self`.
    @_spi(Testing) public func step(_ input: Set<[Int]>, byte: UInt8) -> Set<[Int]> {
        Self.step(input, byte: byte, compiled: compiled)
    }

    // MARK: - Core

    /// Drop trailing `.end` markers: an alternative that has matched all its
    /// symbols pops back to the caller frame below it.
    private static func canonical(_ s: [Int], _ c: GBNFCompiled) -> [Int] {
        var s = s
        while let top = s.last, case .end = c.positions[top] { s.removeLast() }
        return s
    }

    /// Consume one byte from a stack set: keep stacks whose top charset matches,
    /// advance past it, and re-normalize the tails.
    static func step(_ input: Set<[Int]>, byte: UInt8, compiled c: GBNFCompiled) -> Set<[Int]> {
        var tails: [[Int]] = []
        for s in input {
            guard let top = s.last, case let .charset(cs) = c.positions[top], cs.contains(byte) else { continue }
            var t = s
            t[t.count - 1] = top + 1
            tails.append(t)
        }
        return tails.isEmpty ? [] : normalize(tails, compiled: c)
    }

    /// Expand leading rule references into their alternatives until every
    /// non-empty stack's top is a charset. The `visited` set over *canonical*
    /// stacks makes cyclic/nullable expansions terminate.
    static func normalize(_ initial: [[Int]], compiled c: GBNFCompiled) -> Set<[Int]> {
        var result: Set<[Int]> = []
        var visited: Set<[Int]> = []
        var work = initial
        while let raw = work.popLast() {
            let s = canonical(raw, c)
            if !visited.insert(s).inserted { continue }
            guard let top = s.last else { result.insert([]); continue }
            switch c.positions[top] {
            case .charset:
                result.insert(s)
            case .rule(let id):
                // Suspend the current frame at the symbol after this reference,
                // then push each alternative of `id` on top.
                // Canonicalize base before appending so that nullable right-recursive
                // grammars (e.g. `a ::= b a | "x"; b ::= "" | "y"`) don't accumulate
                // unbounded chains of trailing .end return-address frames — collapsing
                // them makes semantically equivalent states hash-equal so `visited`
                // terminates the expansion.
                var base = s
                base[base.count - 1] = top + 1
                let canonBase = canonical(base, c)
                for st in c.altStarts[id] {
                    work.append(canonBase + [st])
                }
            case .end:
                // Unreachable: `canonical` already popped trailing ends.
                break
            }
        }
        return result
    }
}
