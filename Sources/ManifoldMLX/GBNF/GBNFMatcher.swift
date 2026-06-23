import Foundation

/// Incremental byte-level matcher for a parsed ``GBNFGrammar`` (#96, option B).
///
/// Mirrors llama.cpp's grammar algorithm: the live state is a *set of stacks*,
/// where a stack is the sequence of symbols still to be matched (index 0 next).
/// Stacks are kept normalized so every non-empty stack's head is a `.charset`;
/// rule references are expanded into the alternatives they stand for, branching
/// the set. An empty stack means the root derivation can complete here (EOS is
/// allowed).
///
/// From this state the matcher answers two questions per generation step:
/// - which next *bytes* are legal (`acceptableFirstBytes`), and
/// - does a given byte string keep at least one stack alive (`accepts(_:)`),
///
/// then `advance(_:)` commits the bytes of the sampled token.
@_spi(Testing) public struct GBNFMatcher {
    private let grammar: GBNFGrammar
    private var stacks: Set<[GBNFGrammar.Symbol]>

    public init(grammar: GBNFGrammar) {
        self.grammar = grammar
        self.stacks = []
        self.stacks = normalize(grammar.alternatives(grammar.rootId).map { $0 })
    }

    /// True when no stack survives — the output so far cannot extend to any
    /// grammar-valid string (should never happen mid-generation if every emitted
    /// token was grammar-checked first).
    public var isDead: Bool { stacks.isEmpty }

    /// True when the root derivation can terminate here, i.e. EOS is permitted.
    public var isComplete: Bool { stacks.contains([]) }

    /// Whole-string acceptance: feeds `text`'s UTF-8 bytes through a fresh
    /// matcher and reports whether they form a complete grammar-valid string.
    /// Test/diagnostic convenience.
    public static func matches(_ text: String, grammar: GBNFGrammar) -> Bool {
        var m = GBNFMatcher(grammar: grammar)
        for b in Array(text.utf8) {
            guard m.accepts([b]) else { return false }
            m.advance([b])
        }
        return m.isComplete
    }

    /// The set of byte values that at least one live stack would accept next.
    /// Used to prune the vocabulary before per-token checks.
    public func acceptableFirstBytes() -> Set<UInt8> {
        var bytes: Set<UInt8> = []
        for s in stacks {
            guard let head = s.first, case let .charset(cs) = head else { continue }
            for b in UInt8.min...UInt8.max where cs.contains(b) { bytes.insert(b) }
        }
        return bytes
    }

    /// Whether `bytes` can be consumed from the current state without killing
    /// every stack (a valid *prefix* of some grammar continuation). Non-mutating
    /// — used to test candidate tokens.
    public func accepts(_ bytes: [UInt8]) -> Bool {
        var current = stacks
        for b in bytes {
            current = step(current, byte: b)
            if current.isEmpty { return false }
        }
        return true
    }

    /// Commit the sampled token's bytes, moving the live state forward.
    public mutating func advance(_ bytes: [UInt8]) {
        for b in bytes { stacks = step(stacks, byte: b) }
    }

    // MARK: - Core

    /// Consume one byte from a stack set: keep stacks whose head charset matches,
    /// drop the head, and re-normalize the tails.
    private func step(_ input: Set<[GBNFGrammar.Symbol]>, byte: UInt8) -> Set<[GBNFGrammar.Symbol]> {
        var tails: [[GBNFGrammar.Symbol]] = []
        for s in input {
            guard let head = s.first, case let .charset(cs) = head, cs.contains(byte) else { continue }
            tails.append(Array(s.dropFirst()))
        }
        return tails.isEmpty ? [] : normalize(tails)
    }

    /// Normalize a set of stacks: expand leading rule references into their
    /// alternatives until every non-empty stack's head is a charset. The
    /// `visited` worklist guard makes cyclic/nullable expansions terminate.
    private func normalize(_ initial: [[GBNFGrammar.Symbol]]) -> Set<[GBNFGrammar.Symbol]> {
        var result: Set<[GBNFGrammar.Symbol]> = []
        var work = initial
        var visited: Set<[GBNFGrammar.Symbol]> = []
        while let s = work.popLast() {
            if !visited.insert(s).inserted { continue }
            guard let head = s.first else { result.insert([]); continue }
            switch head {
            case .charset:
                result.insert(s)
            case .rule(let id):
                let tail = Array(s.dropFirst())
                for alt in grammar.alternatives(id) {
                    work.append(alt + tail)
                }
            }
        }
        return result
    }
}
