import XCTest
import Foundation

/// Source-contract audit: every MLX backend that drives the Metal GPU must stage
/// `mlx.metallib` next to the running binary before its first GPU op, or a
/// command-line `swift build` / `swift run` consumer aborts at GPU init with
/// "Failed to load the default metallib" (issue #82).
///
/// ## Why a source audit instead of a behavioural test
///
/// `MLXMetallibStaging.ensureStaged()` runs its copy exactly once per process,
/// and it stages to the *shared* binary directory. In manifold-mlx's own
/// `swift test` process a text `MLXBackend` load almost always runs first and
/// stages the metallib next to the shared test binary — which would satisfy a
/// diffusion generate that followed, masking a *missing* staging call in a
/// diffusion backend. A diffusion-only downstream consumer (the DX image-gen
/// app: no text model ever loaded) gets no such free staging and crashes. That
/// failure is invisible to any in-process behavioural assertion, so we lock the
/// invariant at the source level: each backend's `loadModel` must invoke
/// `MLXMetallibStaging.ensureStaged()` on a live (non-comment) line.
final class MetallibStagingContractTests: XCTestCase {

    /// Resolves a package-relative path from this test file's compile-time
    /// location: `<root>/Tests/ManifoldMLXTests/<thisFile>.swift` → `<root>`.
    private func packageSourceURL(_ relativePath: String) -> URL {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // ManifoldMLXTests
            .deletingLastPathComponent()   // Tests
            .deletingLastPathComponent()   // <package root>
        return root.appendingPathComponent(relativePath)
    }

    private func assertStagesMetallib(
        _ relativePath: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        let url = packageSourceURL(relativePath)
        let source = try String(contentsOf: url, encoding: .utf8)

        // A live call — not a mention in a comment. Consider only the code
        // portion of each line (everything before the first `//`), so a full-line
        // `//`/`///` comment OR a trailing comment mentioning the token can't
        // satisfy the contract.
        let callsEnsureStaged = source
            .split(separator: "\n", omittingEmptySubsequences: false)
            .contains { rawLine in
                let code = String(rawLine).components(separatedBy: "//").first ?? ""
                return code.contains("MLXMetallibStaging.ensureStaged()")
            }

        XCTAssertTrue(
            callsEnsureStaged,
            """
            \(relativePath) must call MLXMetallibStaging.ensureStaged() in loadModel (issue #82). \
            Without it a diffusion-only `swift run` consumer aborts at GPU init with \
            'Failed to load the default metallib' — see MetallibStagingContractTests for why a \
            behavioural test cannot catch this.
            """,
            file: file,
            line: line
        )
    }

    func test_fluxDiffusionBackend_stagesMetallib() throws {
        try assertStagesMetallib("Sources/ManifoldMLX/Diffusion/Flux/FluxDiffusionBackend.swift")
    }

    func test_mlxDiffusionBackend_stagesMetallib() throws {
        try assertStagesMetallib("Sources/ManifoldMLX/MLXDiffusionBackend.swift")
    }

    func test_mlxTextBackend_stagesMetallib() throws {
        // The text backend already stages; pin it so a future refactor can't
        // silently drop the call that the diffusion backends now depend on the
        // precedent of.
        try assertStagesMetallib("Sources/ManifoldMLX/MLXBackend.swift")
    }

    /// Guards the audit itself: a bogus path must fail loudly (file read throws),
    /// proving the assertion isn't vacuously passing on unreadable sources.
    func test_auditReadsRealSources() throws {
        let url = packageSourceURL("Sources/ManifoldMLX/MLXMetallibStaging.swift")
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: url.path),
            "Audit path resolution is broken: expected \(url.path) to exist. "
                + "Fix packageSourceURL before trusting the staging assertions."
        )
    }
}
