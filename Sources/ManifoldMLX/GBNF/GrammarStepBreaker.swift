/// Counts sampled tokens during a grammar-constrained turn and signals when the
/// step cap is exceeded (#114). Extracted for testability — the MLX-dependent
/// `MLXGrammarLogitProcessor` uses this to force EOS on prose-runaway turns.
@_spi(Testing) public struct GrammarStepBreaker {
    public let maxSteps: Int
    @_spi(Testing) public private(set) var stepCount: Int = 0

    @_spi(Testing) public init(maxSteps: Int) {
        self.maxSteps = maxSteps
    }

    /// Increments the step counter. Call once per `process(logits:)` invocation.
    @_spi(Testing) public mutating func advance() {
        stepCount += 1
    }

    /// True when the step count exceeds `maxSteps`. The caller should check
    /// `GBNFFastMatcher.isComplete` before acting — the cap only fires to
    /// prevent prose runaway, not to interrupt a legitimately-terminating grammar.
    @_spi(Testing) public var capExceeded: Bool { stepCount > maxSteps }
}
