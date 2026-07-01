// manifold-tools-mlx bfcl — run ManifoldKit's BFCL argument-level tool-call eval
// against a real MLX model.
//
// Reuses the shared, backend-agnostic `BFCLRunner` from the ManifoldTools library
// (scoring loop, output format, capture records) and the vendored BFCL fixtures;
// the only MLX-specific wiring is loading the model directory through `MLXBackend`
// and injecting it via the `InferenceService(backend:…)` seam (the model is loaded
// first, so the harness/offline init marks it ready immediately) — identical to
// the scenario harness in main.swift.
//
// NOTE: MLX tool-calling is known to be partly broken (the detokenizer drops
// tool-call JSON quotes; the structural-tool path is emit-but-unparseable), so a
// near-zero AST score here is expected and is itself useful backend signal.
import Foundation
import ManifoldInference
import ManifoldTools
import ManifoldMLX

@MainActor
enum BFCLMLXCLI {

    static func run(_ args: [String]) async -> Int32 {
        func err(_ s: String) {
            FileHandle.standardError.write(Data((s + "\n").utf8))
        }

        var modelPath: String?
        var category = "multiple"
        var dumpPath: String?
        var timeoutSeconds: Double = 120

        var i = 0
        while i < args.count {
            switch args[i] {
            case "--model":
                guard i + 1 < args.count else { err("--model requires a value"); return 2 }
                modelPath = args[i + 1]; i += 1
            case "--category":
                guard i + 1 < args.count else { err("--category requires a value"); return 2 }
                category = args[i + 1]; i += 1
            case "--dump":
                guard i + 1 < args.count else { err("--dump requires a value"); return 2 }
                dumpPath = args[i + 1]; i += 1
            case "--timeout":
                guard i + 1 < args.count else { err("--timeout requires a value"); return 2 }
                guard let t = Double(args[i + 1]), t > 0 else {
                    err("--timeout must be a positive number"); return 2
                }
                timeoutSeconds = t; i += 1
            case "-h", "--help":
                print("usage: manifold-tools-mlx bfcl --model <model-dir> [--category simple|multiple] [--dump PATH.jsonl] [--timeout SECONDS]")
                return 0
            default:
                err("unknown flag '\(args[i])'")
                return 2
            }
            i += 1
        }

        guard let modelPath else {
            err("manifold-tools-mlx bfcl: --model <model-dir> is required")
            return 2
        }
        let modelURL = URL(fileURLWithPath: modelPath, isDirectory: true)
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            err("model directory not found: \(modelURL.path)")
            return 1
        }

        // Pre-flight: refuse VL model directories before attempting to load them.
        // Loading a vision-language model via the text-only MLX backend path causes
        // a hard SIGSEGV (exit 139) with an empty log, making failures undebuggable.
        let vlMarkerFiles: [String] = [
            "preprocessor_config.json",
            "processor_config.json",
            "video_preprocessor_config.json",
        ]
        let fm = FileManager.default
        for marker in vlMarkerFiles {
            let markerPath = modelURL.appendingPathComponent(marker).path
            if fm.fileExists(atPath: markerPath) {
                err("'\(modelPath)' looks like a vision-language model (found \(marker)); this is a text-only tool harness and would crash on load — skipped")
                return 1
            }
        }

        let cases: [BFCLLoadedCase]
        do {
            cases = try BFCLCaseLoader.loadBundled(category: category)
        } catch {
            err("failed to load BFCL '\(category)' cases: \(error)")
            return 1
        }
        print("BFCL category: \(category)")

        // Load the MLX model once, then inject it via the InferenceService seam
        // (marks the model loaded immediately — the production load-from-ModelInfo
        // path is GGUF-specific). Empty registry: BFCLRunner captures the model's
        // first tool call and scores it; we never dispatch.
        guard !cases.isEmpty else {
            err("no BFCL cases found for category '\(category)' — check bundled fixtures")
            return 1
        }

        let backend = MLXBackend()
        defer { backend.unloadModel() }
        print("Loading MLX model from \(modelURL.path) …")
        do {
            try await backend.loadModel(from: modelURL, plan: .systemManaged(requestedContextSize: 4096))
        } catch {
            let detail = (error as? LocalizedError)?.errorDescription ?? "\(error)"
            err("LOAD FAILED: \(modelURL.lastPathComponent): \(detail)")
            return 3
        }

        let service = InferenceService(
            backend: backend,
            name: "mlx",
            modelName: modelURL.lastPathComponent,
            toolRegistry: ToolRegistry()
        )

        let outcome = await BFCLRunner().run(
            cases: cases,
            service: service,
            modelLabel: "mlx/\(modelURL.lastPathComponent)",
            perCaseTimeoutSeconds: timeoutSeconds
        )

        // Exit 1 if any cases errored at the backend (timeout, crash, etc.) —
        // a regression signal distinct from low-but-expected AST scores.
        var exitCode: Int32 = outcome.summary.errored > 0 ? 1 : 0

        if let dumpPath {
            do {
                var body = try outcome.records.map { try $0.jsonLine() }.joined(separator: "\n")
                body += "\n"
                try body.write(toFile: dumpPath, atomically: true, encoding: .utf8)
                print("\nWrote \(outcome.records.count) case record(s) → \(dumpPath)")
            } catch {
                err("failed to write dump to \(dumpPath): \(error)")
                exitCode = 1
            }
        }
        return exitCode
    }
}
