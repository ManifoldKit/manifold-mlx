import XCTest
@_spi(Testing) import ManifoldMLX

/// Tests for the grammar-constrained sampling circuit breaker and the
/// `GrammarStepBreaker` type introduced in #114.
///
/// `MLXGrammarLogitProcessor.process(logits:)` requires Metal and is tested in
/// the integration tier. This suite covers the pure-Swift circuit-breaker logic
/// that `process` delegates to.
final class MLXGrammarLogitProcessorTests: XCTestCase {

    // MARK: - GrammarStepBreaker

    func test_breaker_capNotExceeded_beforeStepLimit() {
        var b = GrammarStepBreaker(maxSteps: 3)
        b.advance(); b.advance(); b.advance()
        XCTAssertFalse(b.capExceeded, "cap must not fire at exactly maxSteps")
    }

    func test_breaker_capExceeded_afterStepLimit() {
        var b = GrammarStepBreaker(maxSteps: 3)
        b.advance(); b.advance(); b.advance(); b.advance()
        XCTAssertTrue(b.capExceeded, "cap must fire one step past maxSteps")
    }

    func test_breaker_stepCount_tracksAdvances() {
        var b = GrammarStepBreaker(maxSteps: 100)
        for _ in 0..<7 { b.advance() }
        XCTAssertEqual(b.stepCount, 7)
    }

    func test_breaker_maxSteps1_firesOnSecondStep() {
        var b = GrammarStepBreaker(maxSteps: 1)
        b.advance()
        XCTAssertFalse(b.capExceeded, "first step must not exceed cap of 1")
        b.advance()
        XCTAssertTrue(b.capExceeded, "second step must exceed cap of 1")
    }

    func test_breaker_zeroMaxSteps_firesImmediately() {
        var b = GrammarStepBreaker(maxSteps: 0)
        b.advance()
        XCTAssertTrue(b.capExceeded, "maxSteps=0 must fire on the very first step")
    }
}
