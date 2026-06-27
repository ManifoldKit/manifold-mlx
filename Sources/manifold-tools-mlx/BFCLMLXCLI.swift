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
        var modelPath: String?
        var category = "multiple"
        var dumpPath: String?

        var i = 0
        while i < args.count {
            switch args[i] {
            case "--model":
                if i + 1 < args.count { modelPath = args[i + 1]; i += 1 }
            case "--category":
                if i + 1 < args.count { category = args[i + 1]; i += 1 }
            case "--dump":
                if i + 1 < args.count { dumpPath = args[i + 1]; i += 1 }
            case "-h", "--help":
                print("usage: manifold-tools-mlx bfcl --model <model-dir> [--category simple|multiple] [--dump PATH.jsonl]")
                return 0
            default:
                FileHandle.standardError.write(Data("unknown flag '\(args[i])'\n".utf8))
                return 2
            }
            i += 1
        }

        guard let modelPath else {
            FileHandle.standardError.write(Data("manifold-tools-mlx bfcl: --model <model-dir> is required\n".utf8))
            return 2
        }
        let modelURL = URL(fileURLWithPath: modelPath, isDirectory: true)
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            FileHandle.standardError.write(Data("model directory not found: \(modelURL.path)\n".utf8))
            return 1
        }

        let cases: [BFCLLoadedCase]
        do {
            cases = try BFCLCaseLoader.loadBundled(category: category)
        } catch {
            FileHandle.standardError.write(Data("failed to load BFCL '\(category)' cases: \(error)\n".utf8))
            return 1
        }
        print("BFCL category: \(category)")

        // Load the MLX model once, then inject it via the InferenceService seam
        // (marks the model loaded immediately — the production load-from-ModelInfo
        // path is GGUF-specific). Empty registry: BFCLRunner captures the model's
        // first tool call and scores it; we never dispatch.
        let backend = MLXBackend()
        print("Loading MLX model from \(modelURL.path) …")
        do {
            try await backend.loadModel(from: modelURL, plan: .systemManaged(requestedContextSize: 4096))
        } catch {
            let detail = (error as? LocalizedError)?.errorDescription ?? "\(error)"
            FileHandle.standardError.write(Data("LOAD FAILED: \(modelURL.lastPathComponent): \(detail)\n".utf8))
            return 3
        }
        defer { backend.unloadModel() }

        let service = InferenceService(
            backend: backend,
            name: "mlx",
            modelName: modelURL.lastPathComponent,
            toolRegistry: ToolRegistry()
        )

        let outcome = await BFCLRunner().run(
            cases: cases,
            service: service,
            modelLabel: "mlx/\(modelURL.lastPathComponent)"
        )

        var exitCode: Int32 = 0
        if let dumpPath {
            do {
                let body = try outcome.records.map { try $0.jsonLine() }.joined(separator: "\n")
                try (body + "\n").write(toFile: dumpPath, atomically: true, encoding: .utf8)
                print("\nWrote \(outcome.records.count) case record(s) → \(dumpPath)")
            } catch {
                FileHandle.standardError.write(Data("failed to write dump to \(dumpPath): \(error)\n".utf8))
                exitCode = 1
            }
        }
        return exitCode
    }
}
