import XCTest

/// Re-homed from core's cross-family `MLXMemoryPressureMissingTests`
/// (Tests/ManifoldBackendsTests, retired in core PR C2 — ManifoldKit#1749).
/// This is the negative MLX guard of the audit asymmetry; the positive
/// Llama guard (LlamaBackend registers a MemoryPressureHandler, #415)
/// lives in manifold-llama.
///
/// The test reads the source file as a plain string and asserts the
/// substring absence directly. Reflection won't work — the field would be
/// `private` and Swift's `Mirror` doesn't expose private storage across
/// module boundaries — and `Mirror`-based checks would also miss the case
/// where the handler is registered without a matching stored field.
final class MLXMemoryPressureMissingTests: XCTestCase {

    private func sourcePath(fileName: String) -> String {
        let testFileURL = URL(fileURLWithPath: #filePath)
        let packageRoot = testFileURL
            .deletingLastPathComponent()  // Tests/ManifoldMLXTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // package root
        return packageRoot
            .appendingPathComponent("Sources")
            .appendingPathComponent("ManifoldMLX")
            .appendingPathComponent(fileName)
            .path
    }

    // FIXME: When MLX gains a memory-pressure handler, flip this assertion.
    // Confirms audit asymmetry: LlamaBackend has it (#415), MLX does not.
    func test_mlxBackend_hasNoMemoryPressureHandlerYet() throws {
        let source = try String(
            contentsOfFile: sourcePath(fileName: "MLXBackend.swift"),
            encoding: .utf8
        )
        XCTAssertFalse(
            source.contains("MemoryPressureHandler"),
            "MLXBackend.swift unexpectedly contains 'MemoryPressureHandler' — if the handler was added, flip this assertion to XCTAssertTrue and remove the FIXME above."
        )
    }
}
