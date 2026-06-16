
import XCTest
@_spi(Testing) import ManifoldMLX

/// Unit tests for ``MLXPreparedInput/suffix(from:)`` on the token-id branch.
///
/// Constructed via `init(promptTokenIds:)` (the no-`LMInput` path), so these
/// tests exercise prefix-reuse trimming without any MLX runtime types.
final class MLXPreparedInputTests: XCTestCase {

    func test_suffix_fromZero_returnsUnchangedTokens() {
        let input = MLXPreparedInput(promptTokenIds: [10, 20, 30, 40])
        XCTAssertEqual(input.suffix(from: 0).promptTokenIds, [10, 20, 30, 40])
    }

    func test_suffix_dropsReusedPrefix() {
        let input = MLXPreparedInput(promptTokenIds: [10, 20, 30, 40])
        XCTAssertEqual(input.suffix(from: 2).promptTokenIds, [30, 40])
    }

    func test_suffix_fromCount_returnsEmpty() {
        let input = MLXPreparedInput(promptTokenIds: [10, 20, 30, 40])
        XCTAssertEqual(input.suffix(from: 4).promptTokenIds, [])
    }

    func test_suffix_largerThanCount_returnsEmptyNoCrash() {
        let input = MLXPreparedInput(promptTokenIds: [10, 20, 30, 40])
        XCTAssertEqual(input.suffix(from: 99).promptTokenIds, [])
    }

    func test_suffix_negative_returnsUnchanged() {
        // reusedPromptTokenCount <= 0 short-circuits to `self`.
        let input = MLXPreparedInput(promptTokenIds: [10, 20, 30, 40])
        XCTAssertEqual(input.suffix(from: -3).promptTokenIds, [10, 20, 30, 40])
    }
}
