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

    // MARK: - decoy-names (migration off manifold-mlx's stale local DecoyTools
    // copy onto core's published `ManifoldTools.DecoyTools`, tools/local matrix
    // night 2026-08-07). A count or set-membership assertion would pass on a
    // truncated OR reordered pool — those are the exact two failure modes that
    // let the stale copy drift undetected. These pin the exact ordered prefix.
    //
    // The expected values below are the pool as resolved through THIS repo's
    // pinned ManifoldKit dependency (Package.resolved, currently v0.75.0 /
    // 3b4d1d2a) — NOT core's live main, which has since renamed positions 1/3
    // (`get_weather`->`get_air_quality`, `search_web`->`get_movie_showtimes`,
    // ManifoldKit#2413). That rename lands here automatically the next time the
    // pin advances; when it does, this test's expected values must be updated
    // in the same PR that bumps the pin — a failure here after a pin bump is
    // the intended signal, not a false alarm.

    /// The pool must now be core's real 46 entries (was a stale local copy of
    /// 24) — `maxCount` is the first, cheapest signal a truncated pool is back.
    func test_decoyNames_maxCountIs46() throws {
        let result = try runBinary(args: ["decoy-names", "9999"])
        // Out-of-range must still name the CURRENT pool size in its error, not
        // a stale one.
        XCTAssertEqual(result.exitCode, 2)
        XCTAssertTrue(result.stderr.contains("(46)"),
            "expected pool size 46 in the error; got: \(result.stderr)")
    }

    /// N=3 pinned to the exact ordered names — catches both truncation
    /// (wrong count) and reordering (right set, wrong sequence).
    func test_decoyNames_first3_exactOrderedList() throws {
        let result = try runBinary(args: ["decoy-names", "3"])
        XCTAssertEqual(result.exitCode, 0, "stderr: \(result.stderr)")
        XCTAssertEqual(
            result.stdout.trimmingCharacters(in: .whitespacesAndNewlines),
            "get_weather\nsend_email\nsearch_web"
        )
    }

    /// N=20 (tonight's planned maximum decoy level) pinned to the exact
    /// ordered list, so a matched-decoy-level cross-runtime comparison is
    /// verifiably advertising the same distractors this repo resolves today.
    func test_decoyNames_first20_exactOrderedList() throws {
        let result = try runBinary(args: ["decoy-names", "20"])
        XCTAssertEqual(result.exitCode, 0, "stderr: \(result.stderr)")
        let expected = """
        get_weather
        send_email
        search_web
        translate_text
        set_timer
        convert_currency
        create_calendar_event
        get_stock_price
        roll_dice
        convert_units
        send_sms
        get_directions
        play_music
        set_reminder
        get_news_headlines
        book_flight
        get_definition
        create_note
        get_traffic
        shorten_url
        """
        XCTAssertEqual(result.stdout.trimmingCharacters(in: .whitespacesAndNewlines), expected)
    }

    /// N=0 must be empty, not a stray newline or an error — the boundary case.
    func test_decoyNames_zero_isEmpty() throws {
        let result = try runBinary(args: ["decoy-names", "0"])
        XCTAssertEqual(result.exitCode, 0, "stderr: \(result.stderr)")
        XCTAssertEqual(result.stdout, "")
    }

    /// `--extra-tools` must accept up to the real pool size (46) — this failed
    /// at 24 before the migration, silently capping every decoy sweep below
    /// the ceiling the other two runtimes could reach.
    func test_extraTools46_isAccepted_47IsRejected() throws {
        let ok = try runBinary(args: ["--list", "--extra-tools", "46"])
        XCTAssertEqual(ok.exitCode, 0, "stderr: \(ok.stderr)")

        let over = try runBinary(args: ["--list", "--extra-tools", "47"])
        XCTAssertEqual(over.exitCode, 2)
        XCTAssertTrue(over.stderr.contains("(46)"),
            "expected the current pool size 46 in the rejection message; got: \(over.stderr)")
    }
}
