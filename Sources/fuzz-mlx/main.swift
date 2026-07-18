// fuzz-mlx — overnight fuzz/soak driver for the MLX backend family.
//
// Reuses core's `ManifoldFuzz` engine + detector suite against a REAL, resident
// `MLXBackend` loaded from a local model directory. This is the first companion
// fuzz driver (the v0.48 migration note in core's Package.swift promised one but
// never shipped it). It exists as a plain SwiftPM executable — NOT an xctest —
// because manifold-mlx's `MLXMetallibPlugin` prebuild plugin stages `mlx.metallib`
// next to the built binary during `swift build`, so `swift run fuzz-mlx` finds its
// Metal kernels without an Xcode-compiled bundle (issue #82). That is precisely
// why core's own `fuzz-chat` refuses `--backend mlx`: core has no MLX target and
// no metallib plugin, so there it genuinely cannot run.
//
// Exit codes follow the manifold-tools-mlx convention: 2 = bad args, 1 = runtime
// failure (model load / campaign error), 0 = campaign ran (findings are DATA, not
// a failure — the wrapper reads tmp/fuzz/INDEX.md for them).

import Foundation
import ManifoldFuzz
import ManifoldInference
import ManifoldMLX

// MARK: - Backend factory

/// A `FuzzBackendFactory` that loads ONE MLX model and hands the SAME resident
/// backend to every `makeHandle()` call.
///
/// `FuzzRunner` calls `makeHandle()` once per iteration (FuzzRunner.swift:116).
/// A multi-GB MLX model takes seconds to load, so reloading per iteration would
/// make a campaign impossible — the loaded backend is cached and only released
/// in `teardown()`, once the whole campaign ends. This mirrors how a real app
/// keeps one resident backend across many requests.
///
/// An `actor` so the cached-backend mutation is data-race-free; the sync
/// protocol requirement is satisfied `nonisolated`.
actor MLXFuzzFactory: FuzzBackendFactory {
    private let modelURL: URL
    private let modelId: String
    private let enableKVCacheReuse: Bool
    private let contextSize: Int
    private var backend: MLXBackend?

    init(modelURL: URL, modelId: String, enableKVCacheReuse: Bool, contextSize: Int) {
        self.modelURL = modelURL
        self.modelId = modelId
        self.enableKVCacheReuse = enableKVCacheReuse
        self.contextSize = contextSize
    }

    // MLX greedy/seeded sampling is deterministic on fixed hardware, so replay
    // shrinking is meaningful. If a finding fails to reproduce it is still
    // recorded; we lose shrinking, not the finding.
    nonisolated var supportsDeterministicReplay: Bool { true }

    func makeHandle() async throws -> FuzzRunner.BackendHandle {
        let resident: MLXBackend
        if let existing = backend {
            resident = existing
        } else {
            let fresh = MLXBackend(enableKVCacheReuse: enableKVCacheReuse)
            try await fresh.loadModel(
                from: modelURL,
                plan: .systemManaged(requestedContextSize: contextSize)
            )
            backend = fresh
            resident = fresh
        }
        return FuzzRunner.BackendHandle(
            backend: resident,
            modelId: modelId,
            modelURL: modelURL,
            backendName: "mlx",
            templateMarkers: nil,
            memoryBudgetBytes: nil
        )
    }

    func teardown() async {
        backend?.unloadModel()
        backend = nil
    }
}

// MARK: - CLI

func fail(_ message: String, code: Int32 = 2) -> Never {
    FileHandle.standardError.write(Data("fuzz-mlx: \(message)\n".utf8))
    exit(code)
}

/// Resolve a `--model` argument: a direct path to a model directory, or a
/// case-insensitive substring matched against directories under
/// ~/Documents/Models/mlx that contain a config.json.
func resolveModel(_ arg: String) -> (url: URL, id: String) {
    let fm = FileManager.default
    let direct = URL(fileURLWithPath: arg, isDirectory: true)
    if fm.fileExists(atPath: direct.appendingPathComponent("config.json").path) {
        return (direct, direct.lastPathComponent)
    }
    let root = URL(fileURLWithPath: NSHomeDirectory())
        .appendingPathComponent("Documents/Models/mlx", isDirectory: true)
    guard let entries = try? fm.contentsOfDirectory(
        at: root, includingPropertiesForKeys: [.isDirectoryKey]
    ) else {
        fail("--model '\(arg)' is not a directory and ~/Documents/Models/mlx is unreadable")
    }
    let candidates = entries.filter {
        fm.fileExists(atPath: $0.appendingPathComponent("config.json").path)
    }
    let needle = arg.lowercased()
    if let match = candidates.first(where: { $0.lastPathComponent.lowercased().contains(needle) }) {
        return (match, match.lastPathComponent)
    }
    let available = candidates.map(\.lastPathComponent).sorted().joined(separator: ", ")
    fail("no MLX model matching '\(arg)' under ~/Documents/Models/mlx. Available: [\(available)]")
}

/// A KV-reuse-biased session script: a long shared system prompt reused across
/// several turns (exercises prefix KV reuse + TurnBoundaryKVStateDetector), a
/// regenerate (turn-boundary KV state), and a verbatim-repeat probe (context
/// leak). Appended to the bundled scripts, which already cover cancel-mid-stream
/// (rapid-send-cancel) and cross-session leak (session-swap).
func kvReuseScript() -> SessionScript {
    let sharedPrefix = String(
        repeating: "You are a meticulous assistant who follows every instruction to the letter. ",
        count: 40
    )
    return SessionScript(
        id: "kv-reuse-shared-prefix",
        steps: [
            .send(text: "Summarize your operating rules in exactly one sentence."),
            .send(text: "Now restate them as three bullet points."),
            .send(text: "Translate the first bullet into French."),
            .regenerate,
            .send(text: "Repeat your previous answer verbatim, then append the word DONE."),
        ],
        systemPrompt: sharedPrefix,
        sessionLabel: "kv-reuse"
    )
}

// ----- parse args -----------------------------------------------------------
var modelArg: String?
var minutes: Int?
var iterations: Int?
var useSessionScripts = false
var useTools = false
var kvReuseOverride: Bool?
var seed: UInt64 = 0
var seedSet = false
var contextSize = 4096
var outDir = "tmp/fuzz"
var quiet = false

var args = Array(CommandLine.arguments.dropFirst())
var i = 0
while i < args.count {
    let a = args[i]
    func value() -> String {
        guard i + 1 < args.count else { fail("\(a) requires a value") }
        i += 1
        return args[i]
    }
    switch a {
    case "--model": modelArg = value()
    case "--minutes": minutes = Int(value()) ?? { fail("--minutes requires an integer") }()
    case "--iterations": iterations = Int(value()) ?? { fail("--iterations requires an integer") }()
    case "--session-scripts": useSessionScripts = true
    case "--tools": useTools = true
    case "--kv-reuse":
        let v = value().lowercased()
        switch v {
        case "on", "true", "1": kvReuseOverride = true
        case "off", "false", "0": kvReuseOverride = false
        default: fail("--kv-reuse expects on|off")
        }
    case "--seed": seed = UInt64(value()) ?? { fail("--seed requires an unsigned integer") }(); seedSet = true
    case "--context": contextSize = Int(value()) ?? { fail("--context requires an integer") }()
    case "--out": outDir = value()
    case "--quiet": quiet = true
    case "-h", "--help":
        print("""
        fuzz-mlx — MLX backend fuzz/soak driver (reuses core ManifoldFuzz).

          --model <name|path>   REQUIRED. Substring match under ~/Documents/Models/mlx, or a dir path.
          --minutes <N>         Time budget (default 5 if no budget given).
          --iterations <N>      Iteration budget (alternative to --minutes).
          --session-scripts     Drive multi-turn session scripts (bundled + a KV-reuse script).
          --tools               Enable the synthetic toolset (tool-call validity detector).
          --kv-reuse on|off     Override KV-cache reuse. Default: on for --session-scripts, else off.
          --seed <N>            Deterministic seed (default: random).
          --context <N>         Context window tokens (default 4096).
          --out <dir>           Findings output dir (default tmp/fuzz). INDEX.md lives here.
          --quiet               Reduce reporter chatter.
        """)
        exit(0)
    default:
        fail("unknown argument '\(a)' (try --help)")
    }
    i += 1
}

guard let modelArg else { fail("--model is required (try --help)") }
if minutes == nil && iterations == nil { minutes = 5 }

let (modelURL, modelId) = resolveModel(modelArg)
// Default KV reuse ON only when driving multi-turn sessions (that is the feature
// under test). Single-turn fuzz runs with reuse OFF for per-iteration isolation.
let kvReuse = kvReuseOverride ?? useSessionScripts

let resolvedSeed = seedSet ? seed : UInt64.random(in: 0...UInt64.max)
let outURL = URL(fileURLWithPath: outDir, isDirectory: true)

FileHandle.standardError.write(Data(
    """
    fuzz-mlx: model=\(modelId) mode=\(useSessionScripts ? "session" : "single-turn") \
    kv-reuse=\(kvReuse) tools=\(useTools) \
    budget=\(minutes.map { "\($0)m" } ?? "\(iterations ?? 0) iters") \
    seed=\(resolvedSeed) out=\(outDir)

    """.utf8
))

let factory = MLXFuzzFactory(
    modelURL: modelURL,
    modelId: modelId,
    enableKVCacheReuse: kvReuse,
    contextSize: contextSize
)

let config = FuzzConfig(
    backend: .mlx,
    minutes: minutes,
    iterations: iterations,
    seed: resolvedSeed,
    modelHint: modelId,
    outputDir: outURL,
    quiet: quiet,
    sessionScripts: useSessionScripts,
    tools: useTools
)

let reporter = TerminalReporter(quiet: quiet)

let report: FuzzReport
if useSessionScripts {
    var scripts = SessionScript.loadAll()
    scripts.append(kvReuseScript())
    let runner = SessionFuzzRunner(config: config, factory: factory, scripts: scripts)
    report = await runner.run(reporter: reporter)
} else {
    let runner = FuzzRunner(config: config, factory: factory)
    report = await runner.run(reporter: reporter)
}

await factory.teardown()

// Final one-line summary (findings are DATA — exit 0 regardless of count).
let perDetector = report.perDetectorFlagRate
    .sorted { $0.value > $1.value }
    .prefix(6)
    .map { "\($0.key)=\(String(format: "%.2f", $0.value))" }
    .joined(separator: " ")
print("""

    fuzz-mlx DONE model=\(modelId) runs=\(report.totalRuns) \
    findings=\(report.findings.count) deduped=\(report.dedupedCount)
    top-detector-flag-rates: \(perDetector.isEmpty ? "(none)" : perDetector)
    findings index: \(outDir)/INDEX.md
    """)
exit(0)
