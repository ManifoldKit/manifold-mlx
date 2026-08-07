import XCTest
import Foundation
// Only for computing this test bundle's OWN expected decoy-pool values from
// the real ManifoldTools.DecoyTools at test-build time — see the decoy-names
// tests below. Never used to drive the CLI itself; that stays subprocess-only
// per this file's own header note (an executable target's CLI/​pure functions
// still cannot be imported directly).
import ManifoldTools

/// Tests for the manifold-tools-mlx CLI argument-parse contract.
///
/// `manifold-tools-mlx` is an **executable** target, so its `CLI` struct
/// cannot be imported directly by this test bundle.  Instead we drive the
/// built binary via `Process`, testing only pure arg-parse / scenario-match
/// behaviour that does NOT require a model or GPU.
///
/// The binary path is resolved from the SwiftPM build directory via
/// `swift build --show-bin-path`.  If the binary is absent the tests are
/// skipped (so CI that runs only `swift test` without a prior build will
/// report "skipped" rather than "failed").
///
/// **Exit-code contract under test**
///   exit 0 — clean success (or --list / --help)
///   exit 2 — bad argument (e.g. --scenario <unknown>, missing --model)
///   exit 1 — runtime failure (model load, fixture resolution, …)
///
/// FIX validated here: an unknown --scenario previously returned exit 1 (runtime
/// failure) and omitted the "manifold-tools-mlx: " stderr prefix.  After the
/// fix it calls `CLI.fail(…)` which exits 2 with the standard prefix.
final class CLIParseTests: XCTestCase {

    // MARK: - Helpers

    private struct RunResult {
        let exitCode: Int32
        let stdout: String
        let stderr: String
    }

    /// Resolve the path to the built manifold-tools-mlx binary, or nil if missing.
    private static let cachedBinaryPath: String? = {
        let showBin = Process()
        showBin.executableURL = URL(fileURLWithPath: "/usr/bin/swift")
        showBin.arguments = ["build", "--product", "manifold-tools-mlx", "--show-bin-path"]

        let pipe = Pipe()
        showBin.standardOutput = pipe
        showBin.standardError = Pipe() // suppress build output

        do {
            try showBin.run()
        } catch {
            return nil
        }
        showBin.waitUntilExit()

        let rawPath = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !rawPath.isEmpty else { return nil }

        let bin = (rawPath as NSString).appendingPathComponent("manifold-tools-mlx")
        return FileManager.default.fileExists(atPath: bin) ? bin : nil
    }()

    /// Run the manifold-tools-mlx binary with the given arguments, or skip the test if unavailable.
    private func runBinary(args: [String]) throws -> RunResult {
        guard let bin = CLIParseTests.cachedBinaryPath else {
            throw XCTSkip("manifold-tools-mlx binary not found — run `swift build` first")
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: bin)
        process.arguments = args

        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        try process.run()
        process.waitUntilExit()

        let stdout = String(data: stdoutPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        let stderr = String(data: stderrPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        return RunResult(exitCode: process.terminationStatus, stdout: stdout, stderr: stderr)
    }

    // MARK: - Tests

    /// An unknown --scenario value must exit 2 (bad argument) and emit the
    /// standard "manifold-tools-mlx: " prefix on stderr.
    ///
    /// This is the core regression guard for the fix that replaced the old
    /// `exit(1)` + bare stderr write with `CLI.fail(…)`.
    func test_unknownScenario_exits2_withStandardPrefix() throws {
        let result = try runBinary(args: ["--scenario", "no-such-scenario-xyz", "--model", "/dev/null"])
        // Exit code must be 2 (arg error), NOT 1 (runtime failure).
        XCTAssertEqual(result.exitCode, 2,
            "unknown --scenario must exit 2 (bad argument); got \(result.exitCode). stderr: \(result.stderr)")
        // Standard arg-error prefix must appear on stderr.
        XCTAssertTrue(result.stderr.contains("manifold-tools-mlx:"),
            "stderr must contain 'manifold-tools-mlx:' prefix; got: \(result.stderr)")
        // The unknown scenario id should appear in the message.
        XCTAssertTrue(result.stderr.contains("no-such-scenario-xyz"),
            "stderr should echo the bad scenario id; got: \(result.stderr)")
    }

    /// Missing --model must also exit 2 (regression guard: established contract).
    func test_missingModel_exits2() throws {
        let result = try runBinary(args: ["--scenario", "all"])
        XCTAssertEqual(result.exitCode, 2,
            "missing --model must exit 2; got \(result.exitCode). stderr: \(result.stderr)")
        XCTAssertTrue(result.stderr.contains("manifold-tools-mlx:"),
            "stderr must contain standard prefix; got: \(result.stderr)")
    }

    /// --list must exit 0 and print at least one scenario line (no model needed).
    func test_listFlag_exits0_andPrintsScenarios() throws {
        let result = try runBinary(args: ["--list"])
        XCTAssertEqual(result.exitCode, 0,
            "--list must exit 0; got \(result.exitCode). stderr: \(result.stderr)")
        XCTAssertTrue(result.stdout.contains("Available scenarios:"),
            "--list should print 'Available scenarios:'; got stdout: \(result.stdout)")
    }

    // MARK: - decoy-names (migration off manifold-mlx's stale local DecoyTools
    // copy onto core's published `ManifoldTools.DecoyTools`, tools/local matrix
    // night 2026-08-07). A count or set-membership assertion would pass on a
    // truncated OR reordered pool — those are the exact two failure modes that
    // let the stale copy drift undetected. These pin the exact ordered prefix.
    //
    // rev-183 HIGH finding: an earlier revision of these tests hardcoded the
    // expected names as string literals. `companion-core-bump.yml`'s gate is
    // `swift build && swift test` and only opens the pin-bump PR on success —
    // so the next core release that renames a pool entry (core#2413 already
    // did this: `get_weather`->`get_air_quality`, `search_web`->
    // `get_movie_showtimes`) would red this gate and abort the fan-out BEFORE
    // a PR exists to fix it (the bump PR is bot-authored, no human in the
    // loop — "update the literal in the same PR" cannot happen). So the
    // expected values below are computed from the real
    // `ManifoldTools.DecoyTools` at test-build time (this test bundle's own
    // `import ManifoldTools`, resolved through the SAME `Package.resolved` the
    // CLI binary under test is built against), not a snapshot. A regression
    // back to a stale LOCAL copy in the executable target still fails these
    // tests — the test bundle has no local shadow to fall into, so its
    // `DecoyTools` reference always resolves to the real library, and the
    // subprocess's stdout is compared against that live value.

    /// The pool's max count must match the real library's `DecoyTools.maxCount`
    /// (was a stale local copy of 24) — the first, cheapest signal a truncated
    /// pool is back, and now immune to the pool ever growing past 46.
    func test_decoyNames_maxCountMatchesRealPool() throws {
        let expectedMax = DecoyTools.maxCount
        let result = try runBinary(args: ["decoy-names", String(expectedMax + 1)])
        // Out-of-range must still name the CURRENT pool size in its error, not
        // a stale one.
        XCTAssertEqual(result.exitCode, 2)
        XCTAssertTrue(result.stderr.contains("(\(expectedMax))"),
            "expected pool size \(expectedMax) in the error; got: \(result.stderr)")
    }

    /// N=3 pinned to the exact ordered names — catches both truncation
    /// (wrong count) and reordering (right set, wrong sequence).
    func test_decoyNames_first3_matchesRealPoolOrderedPrefix() throws {
        let result = try runBinary(args: ["decoy-names", "3"])
        XCTAssertEqual(result.exitCode, 0, "stderr: \(result.stderr)")
        let expected = DecoyTools.names(3).joined(separator: "\n")
        XCTAssertEqual(result.stdout.trimmingCharacters(in: .whitespacesAndNewlines), expected)
    }

    /// N=20 (tonight's planned maximum decoy level) matched against the exact
    /// ordered list the real pool resolves to, so a matched-decoy-level
    /// cross-runtime comparison is verifiably advertising the same distractors
    /// this repo resolves today.
    func test_decoyNames_first20_matchesRealPoolOrderedPrefix() throws {
        let result = try runBinary(args: ["decoy-names", "20"])
        XCTAssertEqual(result.exitCode, 0, "stderr: \(result.stderr)")
        let expected = DecoyTools.names(20).joined(separator: "\n")
        XCTAssertEqual(result.stdout.trimmingCharacters(in: .whitespacesAndNewlines), expected)
    }

    /// N=0 must be empty, not a stray newline or an error — the boundary case.
    func test_decoyNames_zero_isEmpty() throws {
        let result = try runBinary(args: ["decoy-names", "0"])
        XCTAssertEqual(result.exitCode, 0, "stderr: \(result.stderr)")
        XCTAssertEqual(result.stdout, "")
    }

    /// `--extra-tools` must accept up to the real pool size — this failed at
    /// 24 before the migration, silently capping every decoy sweep below the
    /// ceiling the other two runtimes could reach. Computed against
    /// `DecoyTools.maxCount` so this doesn't need updating on a pin bump either.
    func test_extraToolsAcceptsRealMaxCount_rejectsOneOver() throws {
        let expectedMax = DecoyTools.maxCount
        let ok = try runBinary(args: ["--list", "--extra-tools", String(expectedMax)])
        XCTAssertEqual(ok.exitCode, 0, "stderr: \(ok.stderr)")

        let over = try runBinary(args: ["--list", "--extra-tools", String(expectedMax + 1)])
        XCTAssertEqual(over.exitCode, 2)
        XCTAssertTrue(over.stderr.contains("(\(expectedMax))"),
            "expected the current pool size \(expectedMax) in the rejection message; got: \(over.stderr)")
    }
}
