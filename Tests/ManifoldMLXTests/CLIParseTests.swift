import Foundation
import XCTest

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
    showBin.standardError = Pipe()  // suppress build output

    do {
      try showBin.run()
    } catch {
      return nil
    }
    showBin.waitUntilExit()

    let rawPath =
      String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)?
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

    let stdout =
      String(data: stdoutPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
    let stderr =
      String(data: stderrPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
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
    XCTAssertEqual(
      result.exitCode, 2,
      "unknown --scenario must exit 2 (bad argument); got \(result.exitCode). stderr: \(result.stderr)"
    )
    // Standard arg-error prefix must appear on stderr.
    XCTAssertTrue(
      result.stderr.contains("manifold-tools-mlx:"),
      "stderr must contain 'manifold-tools-mlx:' prefix; got: \(result.stderr)")
    // The unknown scenario id should appear in the message.
    XCTAssertTrue(
      result.stderr.contains("no-such-scenario-xyz"),
      "stderr should echo the bad scenario id; got: \(result.stderr)")
  }

  /// Missing --model must also exit 2 (regression guard: established contract).
  func test_missingModel_exits2() throws {
    let result = try runBinary(args: ["--scenario", "all"])
    XCTAssertEqual(
      result.exitCode, 2,
      "missing --model must exit 2; got \(result.exitCode). stderr: \(result.stderr)")
    XCTAssertTrue(
      result.stderr.contains("manifold-tools-mlx:"),
      "stderr must contain standard prefix; got: \(result.stderr)")
  }

  /// An explicit `--core-commit` on the MAIN run path (`CLI.parse`, not the
  /// `record-identity` seam) must never trigger the degraded-fallback stderr
  /// note — rev-182 LOW 4: `coreCommit`'s resolution used to run as a stored
  /// property's default-value expression, evaluated at `CLI(common:)`
  /// construction time, BEFORE the parse loop had even looked at argv for
  /// `--core-commit`. So the "Package.resolved falls back to unknown" /
  /// "coreCommit resolved from ..." notes fired unconditionally on every
  /// parse, misleading the one channel an operator reads to confirm
  /// provenance is live, even when an explicit override was about to win.
  /// `--model` is deliberately omitted so this exercises `CLI.parse` alone
  /// (exit 2 for missing --model) without needing a real model load —
  /// `coreCommit` is fully resolved by the time `CLI.parse` returns, before
  /// the caller ever checks for a model path.
  func test_coreCommitFlag_neverTriggersPackageResolvedStderrNote() throws {
    let result = try runBinary(args: ["--scenario", "all", "--core-commit", "deadbeef2"])
    XCTAssertEqual(result.exitCode, 2, "stderr: \(result.stderr)")
    XCTAssertFalse(
      result.stderr.contains("Package.resolved"),
      "an explicit --core-commit must short-circuit before Package.resolved is ever consulted; got: \(result.stderr)"
    )
  }

  /// --list must exit 0 and print at least one scenario line (no model needed).
  func test_listFlag_exits0_andPrintsScenarios() throws {
    let result = try runBinary(args: ["--list"])
    XCTAssertEqual(
      result.exitCode, 0,
      "--list must exit 0; got \(result.exitCode). stderr: \(result.stderr)")
    XCTAssertTrue(
      result.stdout.contains("Available scenarios:"),
      "--list should print 'Available scenarios:'; got stdout: \(result.stdout)")
  }

  // MARK: - record-identity (ManifoldKit#178: manifold-tools-mlx used to
  // hardcode coreCommit "unknown" and never stamped backend/model/quant on
  // the transcript logger at all, so every emitted ConformanceRecord read
  // backend=unknown model=unknown quant=unknown coreCommit=unknown — the
  // exact string a no-op fix would still produce. These assertions pin the
  // REAL derived values, not merely "non-empty" / "not nil", so a
  // regression back to the placeholder fails loudly.

  /// Runs `record-identity` with a caller-chosen CWD and environment, so
  /// coreCommit-resolution tests can be hermetic: the default CWD inherited
  /// from the test process is the package root, which has a real
  /// `Package.resolved` — the two tests below that need "no
  /// `Package.resolved` present" or "no `$MANIFOLD_CORE_COMMIT` set"
  /// override one or both explicitly rather than assuming the ambient
  /// environment/CWD, per rev-182's MODERATE 3 finding: `runBinary`
  /// inherits the parent environment, so a hard-coded `coreCommit=unknown`
  /// expectation silently breaks the moment an operator (or tonight's
  /// sweep script) exports `$MANIFOLD_CORE_COMMIT` — exactly the operator
  /// this feature targets.
  private func runBinaryHermetic(
    args: [String],
    cwd: URL? = nil,
    environment: [String: String] = [:]
  ) throws -> RunResult {
    guard let bin = CLIParseTests.cachedBinaryPath else {
      throw XCTSkip("manifold-tools-mlx binary not found — run `swift build` first")
    }
    let process = Process()
    process.executableURL = URL(fileURLWithPath: bin)
    process.arguments = args
    process.environment = environment
    if let cwd { process.currentDirectoryURL = cwd }
    let stdoutPipe = Pipe()
    let stderrPipe = Pipe()
    process.standardOutput = stdoutPipe
    process.standardError = stderrPipe
    try process.run()
    process.waitUntilExit()
    let stdout =
      String(data: stdoutPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
    let stderr =
      String(data: stderrPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
    return RunResult(exitCode: process.terminationStatus, stdout: stdout, stderr: stderr)
  }

  /// A 4bit MLX-community-style directory name must decompose into a real
  /// model id (quant suffix stripped) and the real quant label — never the
  /// "unknown" placeholder that used to collapse every MLX cell into one
  /// unidentifiable row for collate's comparability guard. Scoped to
  /// backend/model/quant only — coreCommit resolution has its own tests
  /// below, since it now depends on CWD/environment and asserting it here
  /// too would make this test non-hermetic.
  func test_recordIdentity_4bitSnapshot_decomposesModelAndQuant() throws {
    let result = try runBinary(args: [
      "record-identity", "/models/Mistral-7B-Instruct-v0.3-4bit",
    ])
    XCTAssertEqual(result.exitCode, 0, "stderr: \(result.stderr)")
    XCTAssertTrue(
      result.stdout.hasPrefix("backend=mlx model=Mistral-7B-Instruct-v0.3 quant=4bit "),
      "got: \(result.stdout)"
    )
  }

  /// A directory name with no recognizable quant suffix keeps the full
  /// name as `model` (never silently dropped) and reports no quant.
  /// `quant=(none)` is this seam's own display for "no marker recognized"
  /// — but downstream in core, `ConformanceScorer+Records.swift` maps a
  /// nil quant to the same `"unknown"` string via `row.quant ?? "unknown"`,
  /// so the persisted record's `quant` field is NOT distinguishable from
  /// the placeholder once it reaches a real `ConformanceRecord` (rev-182's
  /// MODERATE 4 finding — verified by reading that mapping directly, not
  /// assumed). That is core's own documented contract for "genuinely
  /// unresolvable", not a defect this PR introduces or can fix locally.
  func test_recordIdentity_noQuantSuffix_keepsFullNameAsModel() throws {
    let result = try runBinary(args: ["record-identity", "gemma-3-27b-it"])
    XCTAssertEqual(result.exitCode, 0, "stderr: \(result.stderr)")
    XCTAssertTrue(
      result.stdout.hasPrefix("backend=mlx model=gemma-3-27b-it quant=(none) "),
      "got: \(result.stdout)"
    )
  }

  /// `--core-commit` must be threaded through into the record's coreCommit
  /// field verbatim, and must win even when a real `Package.resolved` is
  /// present (default CWD here IS the package root) — this is priority 1
  /// of `resolveCoreCommit`'s 4-source order.
  func test_recordIdentity_coreCommitFlag_winsOverPackageResolved() throws {
    let result = try runBinaryHermetic(
      args: ["record-identity", "Mistral-7B-Instruct-v0.3-4bit", "--core-commit", "deadbeef1"],
      environment: [:]
    )
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

  /// `$MANIFOLD_CORE_COMMIT` must win over the `Package.resolved` default
  /// when no `--core-commit` override is given (priority 2 of 4) — the
  /// mechanism `scripts/local-integration-sweep.sh`-style drivers use,
  /// matching core `manifold-tools score --core-commit`'s own
  /// `$MANIFOLD_CORE_COMMIT` fallback convention. Default CWD here IS the
  /// package root (a real `Package.resolved` is present, per the next
  /// test), so this specifically proves the env var outranks it.
  func test_recordIdentity_coreCommitEnvVar_winsOverPackageResolved() throws {
    let result = try runBinaryHermetic(
      args: ["record-identity", "Mistral-7B-Instruct-v0.3-4bit"],
      environment: ["MANIFOLD_CORE_COMMIT": "envcommit9"]
    )
    XCTAssertEqual(result.exitCode, 0, "stderr: \(result.stderr)")
    XCTAssertTrue(
      result.stdout.contains("coreCommit=envcommit9"),
      "expected $MANIFOLD_CORE_COMMIT to resolve into coreCommit; got: \(result.stdout)"
    )
  }

  /// Reads the `manifoldkit` pin's revision directly out of the repo root's
  /// `Package.resolved` — the same file (and same parsing shape) production
  /// reads via `coreCommitFromPackageResolved(cwd:)`. Computed at test-run
  /// time rather than hardcoded, so a pin bump never reds this test (rev-182
  /// HIGH 1: `companion-core-bump.yml`'s gate is `swift build && swift test`
  /// and only opens the automated bump PR on success, so a hardcoded SHA
  /// here would abort the fan-out on the very release that bumps it — before
  /// a PR exists to fix the literal in, since that PR is bot-authored with
  /// no human in the loop).
  private func expectedManifoldKitRevisionFromPackageResolved() throws -> String {
    // Repo root == this test bundle's CWD, matching the same assumption
    // `coreCommitFromPackageResolved` makes for the binary under test.
    let url = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
      .appendingPathComponent("Package.resolved")
    let data = try Data(contentsOf: url)
    let root = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
    let pins = try XCTUnwrap(root["pins"] as? [[String: Any]])
    let manifoldKitPin = try XCTUnwrap(pins.first { ($0["identity"] as? String) == "manifoldkit" })
    let state = try XCTUnwrap(manifoldKitPin["state"] as? [String: Any])
    return try XCTUnwrap(state["revision"] as? String)
  }

  /// With neither override set, `coreCommit` resolves from THIS repo's own
  /// `Package.resolved` (priority 3 of 4) — the default source, and the
  /// fix for rev-182's SEVERE 1 finding: a default run (no flag, no env)
  /// used to emit the literal string `"unknown"` in every record, which
  /// structurally could not satisfy ManifoldKit#178's acceptance criteria
  /// from inside this repo. Also asserts the resolved value is 40-hex (a
  /// real git SHA shape) and never the `"unknown"` placeholder — the two
  /// properties AC1 actually requires, independent of which SHA it is.
  func test_recordIdentity_defaultResolvesFromPackageResolved() throws {
    let expected = try expectedManifoldKitRevisionFromPackageResolved()
    let result = try runBinaryHermetic(
      args: ["record-identity", "Mistral-7B-Instruct-v0.3-4bit"],
      environment: [:]
    )
    XCTAssertEqual(result.exitCode, 0, "stderr: \(result.stderr)")
    XCTAssertTrue(
      result.stdout.contains("coreCommit=\(expected)"),
      "expected the pinned ManifoldKit revision (\(expected)) to resolve from Package.resolved by default; got: \(result.stdout)"
    )
    XCTAssertEqual(expected.count, 40, "a git SHA is 40 hex chars; got: \(expected)")
    XCTAssertTrue(
      expected.allSatisfy(\.isHexDigit),
      "expected an all-hex SHA; got: \(expected)"
    )
    XCTAssertNotEqual(expected, "unknown")
  }

  /// When no override is set AND there is no `Package.resolved` to read
  /// (CWD pointed at an empty directory), resolution falls all the way
  /// through to the documented `"unknown"` placeholder (priority 4 of 4) —
  /// proves the fallback chain terminates instead of crashing or fabricating
  /// a value, and that a real ordinary run (which always has a
  /// `Package.resolved` once built via SwiftPM) is the only path that
  /// reaches the earlier, real-value priorities.
  func test_recordIdentity_noPackageResolved_fallsBackToUnknown() throws {
    let emptyDir = FileManager.default.temporaryDirectory
      .appendingPathComponent("record-identity-no-resolved-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: emptyDir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: emptyDir) }

    let result = try runBinaryHermetic(
      args: ["record-identity", "Mistral-7B-Instruct-v0.3-4bit"],
      cwd: emptyDir,
      environment: [:]
    )
    XCTAssertEqual(result.exitCode, 0, "stderr: \(result.stderr)")
    XCTAssertTrue(
      result.stdout.contains("coreCommit=unknown"),
      "expected the 'unknown' placeholder with no Package.resolved reachable; got: \(result.stdout)"
    )
    XCTAssertTrue(
      result.stderr.contains("Package.resolved"),
      "a degraded fallback must explain itself on stderr, not fail silently; got: \(result.stderr)"
    )
  }
}
