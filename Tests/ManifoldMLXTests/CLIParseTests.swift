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

    /// Run the manifold-tools-mlx binary with the given arguments.
    /// Returns nil and skips the calling test if the binary is not found.
    private func runBinary(
        args: [String],
        file: StaticString = #file,
        line: UInt = #line
    ) -> RunResult? {
        guard let bin = binaryPath(file: file, line: line) else { return nil }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: bin)
        process.arguments = args

        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        do {
            try process.run()
            process.waitUntilExit()
        } catch {
            XCTFail("Failed to launch binary at \(bin): \(error)", file: file, line: line)
            return nil
        }

        let stdout = String(data: stdoutPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        let stderr = String(data: stderrPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        return RunResult(exitCode: process.terminationStatus, stdout: stdout, stderr: stderr)
    }

    /// Resolve the path to the built manifold-tools-mlx binary.
    /// Returns nil (and marks the test skipped) if it can't be found.
    private func binaryPath(file: StaticString, line: UInt) -> String? {
        // Ask SwiftPM where it puts binaries for the current configuration.
        let showBin = Process()
        showBin.executableURL = URL(fileURLWithPath: "/usr/bin/swift")
        showBin.arguments = ["build", "--product", "manifold-tools-mlx", "--show-bin-path"]

        let pipe = Pipe()
        showBin.standardOutput = pipe
        showBin.standardError = Pipe() // suppress build output

        do {
            try showBin.run()
            showBin.waitUntilExit()
        } catch {
            throw XCTSkip("Could not invoke `swift build --show-bin-path`: \(error)")
        }

        let rawPath = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !rawPath.isEmpty else {
            throw XCTSkip("swift build --show-bin-path returned empty output")
        }

        let bin = (rawPath as NSString).appendingPathComponent("manifold-tools-mlx")
        guard FileManager.default.fileExists(atPath: bin) else {
            throw XCTSkip("manifold-tools-mlx binary not found at \(bin) — run `swift build` first")
        }
        return bin
    }

    // MARK: - Tests

    /// An unknown --scenario value must exit 2 (bad argument) and emit the
    /// standard "manifold-tools-mlx: " prefix on stderr.
    ///
    /// This is the core regression guard for the fix that replaced the old
    /// `exit(1)` + bare stderr write with `CLI.fail(…)`.
    func test_unknownScenario_exits2_withStandardPrefix() throws {
        guard let result = runBinary(args: ["--scenario", "no-such-scenario-xyz", "--model", "/dev/null"]) else {
            throw XCTSkip("binary not available")
        }
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
        guard let result = runBinary(args: ["--scenario", "all"]) else {
            throw XCTSkip("binary not available")
        }
        XCTAssertEqual(result.exitCode, 2,
            "missing --model must exit 2; got \(result.exitCode). stderr: \(result.stderr)")
        XCTAssertTrue(result.stderr.contains("manifold-tools-mlx:"),
            "stderr must contain standard prefix; got: \(result.stderr)")
    }

    /// --list must exit 0 and print at least one scenario line (no model needed).
    func test_listFlag_exits0_andPrintsScenarios() throws {
        guard let result = runBinary(args: ["--list"]) else {
            throw XCTSkip("binary not available")
        }
        XCTAssertEqual(result.exitCode, 0,
            "--list must exit 0; got \(result.exitCode). stderr: \(result.stderr)")
        XCTAssertTrue(result.stdout.contains("Available scenarios:"),
            "--list should print 'Available scenarios:'; got stdout: \(result.stdout)")
    }
}
