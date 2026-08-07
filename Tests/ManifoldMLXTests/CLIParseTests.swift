import XCTest
import Foundation

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

    // MARK: - record-identity (ManifoldKit#178: manifold-tools-mlx used to
    // hardcode coreCommit "unknown" and never stamped backend/model/quant on
    // the transcript logger at all, so every emitted ConformanceRecord read
    // backend=unknown model=unknown quant=unknown coreCommit=unknown — the
    // exact string a no-op fix would still produce. These assertions pin the
    // REAL derived values, not merely "non-empty" / "not nil", so a
    // regression back to the placeholder fails loudly.

    /// A 4bit MLX-community-style directory name must decompose into a real
    /// model id (quant suffix stripped) and the real quant label — never the
    /// "unknown" placeholder that used to collapse every MLX cell into one
    /// unidentifiable row for collate's comparability guard.
    func test_recordIdentity_4bitSnapshot_decomposesModelAndQuant() throws {
        let result = try runBinary(args: [
            "record-identity", "/models/Mistral-7B-Instruct-v0.3-4bit",
        ])
        XCTAssertEqual(result.exitCode, 0, "stderr: \(result.stderr)")
        XCTAssertEqual(
            result.stdout.trimmingCharacters(in: .whitespacesAndNewlines),
            "backend=mlx model=Mistral-7B-Instruct-v0.3 quant=4bit coreCommit=unknown"
        )
    }

    /// A directory name with no recognizable quant suffix keeps the full
    /// name as `model` (never silently dropped) and reports no quant —
    /// distinct from, and not confusable with, the "unknown" placeholder.
    func test_recordIdentity_noQuantSuffix_keepsFullNameAsModel() throws {
        let result = try runBinary(args: ["record-identity", "gemma-3-27b-it"])
        XCTAssertEqual(result.exitCode, 0, "stderr: \(result.stderr)")
        XCTAssertEqual(
            result.stdout.trimmingCharacters(in: .whitespacesAndNewlines),
            "backend=mlx model=gemma-3-27b-it quant=(none) coreCommit=unknown"
        )
    }

    /// `--core-commit` must be threaded through into the record's coreCommit
    /// field verbatim — this is the field the issue reports as permanently
    /// hardcoded to "unknown", breaking collate's cross-leg comparability
    /// guard (`ConformanceScorer.RecordContext`'s doc comment: "runs are
    /// only comparable across the same core binary").
    func test_recordIdentity_coreCommitFlag_isThreadedThrough() throws {
        let result = try runBinary(args: [
            "record-identity", "Mistral-7B-Instruct-v0.3-4bit",
            "--core-commit", "deadbeef1",
        ])
        XCTAssertEqual(result.exitCode, 0, "stderr: \(result.stderr)")
        XCTAssertTrue(
            result.stdout.contains("coreCommit=deadbeef1"),
            "expected coreCommit=deadbeef1 in stdout, got: \(result.stdout)"
        )
        XCTAssertFalse(
            result.stdout.contains("coreCommit=unknown"),
            "an explicit --core-commit must never be overridden by the 'unknown' placeholder; got: \(result.stdout)"
        )
    }

    /// `$MANIFOLD_CORE_COMMIT` must resolve when no `--core-commit` override
    /// is given — the mechanism `scripts/local-integration-sweep.sh`-style
    /// drivers use (matching core `manifold-tools score --core-commit`'s own
    /// `$MANIFOLD_CORE_COMMIT` fallback convention).
    func test_recordIdentity_coreCommitEnvVar_resolvesWithoutFlag() throws {
        guard let bin = CLIParseTests.cachedBinaryPath else {
            throw XCTSkip("manifold-tools-mlx binary not found — run `swift build` first")
        }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: bin)
        process.arguments = ["record-identity", "Mistral-7B-Instruct-v0.3-4bit"]
        process.environment = ["MANIFOLD_CORE_COMMIT": "envcommit9"]
        let stdoutPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = Pipe()
        try process.run()
        process.waitUntilExit()
        let stdout = String(data: stdoutPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        XCTAssertEqual(process.terminationStatus, 0)
        XCTAssertTrue(
            stdout.contains("coreCommit=envcommit9"),
            "expected $MANIFOLD_CORE_COMMIT to resolve into coreCommit; got: \(stdout)"
        )
    }
}
