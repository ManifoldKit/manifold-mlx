import XCTest
import Foundation
import ManifoldTestSupport

/// Exercises the REAL `manifold-tools-mlx --emit-records` emission path
/// against a real MLX model — not the `record-identity` pure-function seam.
///
/// rev-182's SEVERE 2 finding: the `record-identity` subcommand shares only
/// the pure helpers (`modelIdentity(from:)`, `quantLabel(from:)`,
/// `resolveCoreCommit`) with production. It does not construct a
/// `TranscriptLogger` at all, so reverting the PRODUCTION wiring — the
/// `TranscriptLogger(url:backend:model:quant:)` call in `runCLI()` back to
/// the bare `TranscriptLogger(url:)` that caused ManifoldKit#178 — left every
/// `CLIParseTests` case green, because none of them touch that code path.
/// `backend: "mlx"` appears as two independent literals in `main.swift` (the
/// production call site and, separately, inside `record-identity`'s own
/// output), so a wiring regression at one site is invisible to a check that
/// only exercises the other.
///
/// This test closes that gap by running the actual binary end-to-end and
/// reading the actual `ConformanceRecord` JSON it writes — the same
/// live-verification rev-182 itself performed manually, promoted to
/// something CI (or any operator with local weights) reruns automatically.
/// Skips gracefully when no local MLX weights are available, matching every
/// other real-weight test in this target.
final class RecordProvenanceIntegrationTests: XCTestCase {

    private struct EmittedRecord: Decodable {
        let backend: String
        let model: String
        let quant: String
        let coreCommit: String
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

    /// Runs the real binary against a real model, scoring the lightest
    /// scenario (`01-now`, single tool, no filesystem chain) to keep this
    /// fast, and asserts the ACTUAL emitted record — not a proxy for it —
    /// carries real backend/model/quant/coreCommit, never the "unknown"
    /// placeholder that was ManifoldKit#178's whole symptom.
    func test_emitRecords_realModel_writesRealProvenance_notUnknown() throws {
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
            .appendingPathComponent("record-provenance-integration-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: workDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: workDir) }
        let transcriptURL = workDir.appendingPathComponent("transcript.jsonl")
        let recordsURL = workDir.appendingPathComponent("records.json")

        let process = Process()
        process.executableURL = URL(fileURLWithPath: bin)
        process.arguments = [
            "--model", modelDir.path,
            "--scenario", "01-now",
            "--output", transcriptURL.path,
            "--emit-records", recordsURL.path,
            "--core-commit", "integrationtestsha",
        ]
        let stderrPipe = Pipe()
        process.standardOutput = Pipe()
        process.standardError = stderrPipe
        try process.run()
        process.waitUntilExit()
        // Exit code is NOT asserted here — a scenario assertion FAIL (the
        // model declining the task) is a valid measured outcome and still
        // writes real records; only a harness/infra failure would leave
        // `recordsURL` empty, which the read below catches.
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
            XCTAssertEqual(record.backend, "mlx", "backend must be the real value, never a placeholder")
            XCTAssertNotEqual(record.model, "unknown", "model must be derived from the loaded directory")
            XCTAssertTrue(
                record.model.contains("Mistral"),
                "expected the real model id to survive into the record; got: \(record.model)"
            )
            XCTAssertEqual(record.quant, "4bit", "quant must be the real decomposed label")
            XCTAssertEqual(record.coreCommit, "integrationtestsha", "coreCommit must thread through to the real record")
        }
    }
}
