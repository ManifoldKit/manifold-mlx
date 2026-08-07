import XCTest
import Foundation
import ManifoldTestSupport

/// Exercises the REAL decoy-padding production call sites — not the
/// `decoy-names` pure-function seam.
///
/// Reviewer finding on the decoy-pool migration PR (tools/local matrix night
/// 2026-08-07): `decoy-names` shares only `DecoyTools.names(_:)` with
/// production. Production separately calls `DecoyTools.executors(extraTools)`
/// in `makeRegistry` (pads the `ToolRegistry` actually advertised to the
/// model) and `DecoyTools.names(cli.extraTools)` in `runCLI` (the `decoyNames`
/// set used for wrong-tool-selection detection AND fed into every emitted
/// record's `decoyLevel`, via core's `ConformanceScorer+Records.swift`
/// pool-membership count). The reviewer sabotaged ONLY those two production
/// call sites (capped the count at 5) and found `CLIParseTests` stayed
/// entirely green — 8/8 — while a real run at `--extra-tools 10` advertised
/// only 5 names and the emitted record read `decoyLevel: 5`. That's the exact
/// silent miscount this migration exists to eliminate, invisible to a check
/// that only exercises `decoy-names`.
///
/// This test closes that gap: it runs the real binary against real weights
/// with `--extra-tools N` and reads the actual emitted `ConformanceRecord`,
/// asserting `decoyLevel == N`.
///
/// **N=12, not N=10** — a second reviewer finding. The stale local pool (pre-
/// migration) and core's live pool are IDENTICAL in the first 11 names; the
/// stale pool's only two names absent from core's pool sit at positions 12
/// (`create_reminder`) and 19 (`lookup_dictionary`). So `--extra-tools 10`
/// yields `decoyLevel: 10` on BOTH the fixed and the unfixed code — it
/// validates against a case that doesn't need the capability the migration
/// adds. `--extra-tools 12` is the smallest N where a correct migration and
/// the stale pool it replaced actually diverge (12 vs 11), so it's the
/// smallest N that can tell them apart.
final class DecoyPoolProductionIntegrationTests: XCTestCase {

    private struct EmittedRecord: Decodable {
        let decoyLevel: Int
    }

    private static let cachedBinaryPath: String? = {
        let showBin = Process()
        showBin.executableURL = URL(fileURLWithPath: "/usr/bin/swift")
        showBin.arguments = ["build", "--product", "manifold-tools-mlx", "--show-bin-path"]
        let pipe = Pipe()
        showBin.standardOutput = pipe
        showBin.standardError = Pipe()
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

    /// Runs the real binary against real weights at `--extra-tools 12` and
    /// asserts the emitted record's `decoyLevel` is exactly 12 — the smallest
    /// N at which a truncated/stale pool (25 names, capped effectively at 11
    /// shared with core) and the real 46-entry migrated pool disagree.
    func test_extraTools12_realModel_emitsDecoyLevel12_notTruncated() throws {
        guard let bin = Self.cachedBinaryPath else {
            throw XCTSkip("manifold-tools-mlx binary not found — run `swift build` first")
        }
        guard let modelDir = HardwareRequirements.findMLXModelDirectory(
            nameContains: "Mistral-7B-Instruct-v0.3"
        ) else {
            throw XCTSkip(
                "No Mistral-7B-Instruct-v0.3 MLX snapshot found on disk — set MLX_TEST_MODEL to one."
            )
        }

        let workDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("decoy-pool-production-integration-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: workDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: workDir) }
        let transcriptURL = workDir.appendingPathComponent("transcript.jsonl")
        let recordsURL = workDir.appendingPathComponent("records.json")

        let process = Process()
        process.executableURL = URL(fileURLWithPath: bin)
        process.arguments = [
            "--model", modelDir.path,
            "--scenario", "01-now",
            "--extra-tools", "12",
            "--output", transcriptURL.path,
            "--emit-records", recordsURL.path,
        ]
        let stderrPipe = Pipe()
        process.standardOutput = Pipe()
        process.standardError = stderrPipe
        try process.run()
        process.waitUntilExit()
        // Exit code not asserted — a scenario assertion FAIL is a valid
        // measured outcome; only an empty records file is a harness failure,
        // caught by the read below.
        let stderr = String(data: stderrPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""

        let data: Data
        do {
            data = try Data(contentsOf: recordsURL)
        } catch {
            XCTFail("--emit-records wrote no file at \(recordsURL.path); stderr: \(stderr)")
            return
        }
        let records = try JSONDecoder().decode([EmittedRecord].self, from: data)
        XCTAssertFalse(records.isEmpty, "expected at least one emitted record; stderr: \(stderr)")
        for record in records {
            XCTAssertEqual(
                record.decoyLevel, 12,
                "requested --extra-tools 12 but the emitted record's decoyLevel was \(record.decoyLevel) — "
                    + "the advertised pool was truncated or diverged from core's, a silent miscount"
            )
        }
    }
}
