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

/// Replay a recorded finding by hash against a fresh MLX backend — the MLX
/// equivalent of `fuzz-chat --replay`, which cannot target MLX. Reruns the
/// recorded input `attempts` times and reports whether the finding reproduces,
/// so a "flaky" finding can be adjudicated real-vs-coincidence. Mirrors
/// fuzz-chat's runReplay outcome→exit-code mapping (0 = ran, 2 = not found /
/// drift refused, 3 = internal error).
func runReplay(hash: String, force: Bool, outputDir: URL, factory: any FuzzBackendFactory) async -> Int32 {
    let replayer = Replayer(findingsRoot: outputDir, factory: factory)
    let outcome: Replayer.Outcome
    do {
        outcome = try await replayer.replay(hash: hash, attempts: 3, force: force)
    } catch {
        FileHandle.standardError.write(Data("replay \(hash): failed — \(error)\n".utf8))
        return 3
    }
    switch outcome {
    case .reproduced(let r):
        let verdict = r.newSeverity == .confirmed ? "promoted to CONFIRMED" : "remains flaky"
        let drift = r.drift != nil ? " [forced despite drift]" : ""
        print("replay \(hash): reproduced \(r.successfulReproductions)/\(r.attempts) — \(verdict)\(drift)")
        return 0
    case .driftRefused(let report):
        print("replay \(hash): drift refused (git \(report.recordedGitRev) → \(report.currentGitRev)); pass --force to override")
        return 2
    case .recordNotFound:
        print("replay \(hash): record not found under \(outputDir.path)")
        return 2
    default:
        print("replay \(hash): \(String(describing: outcome))")
        return 2
    }
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

/// A KV-reuse-biased session script. A LONG shared system prefix makes the KV
/// prefix cache actually engage across turns, but every per-turn question is
/// mutually UNRELATED and never asks the model to reference or repeat a prior
/// turn — so any content from turn N-1 surfacing in turn N is genuine KV
/// residue, not instructed continuity. (The earlier version asked the model to
/// "restate"/"repeat verbatim"/"translate the prior answer", which manufactured
/// 94% false positives in the residue-across-turns detector on 2026-07-18 — a
/// probe must not request the behavior its detector flags.) A trailing
/// regenerate still exercises the turn-boundary KV path without asking for
/// continuity. Appended to the bundled scripts (cancel-mid-stream, session-swap).
func kvReuseScript() -> SessionScript {
    let sharedPrefix = String(
        repeating: "You are a factual assistant. Answer each question independently and concisely. ",
        count: 40
    )
    return SessionScript(
        id: "kv-reuse-independent-turns",
        steps: [
            .send(text: "What is the boiling point of water at sea level in Celsius?"),
            .send(text: "Name the largest moon of Saturn."),
            .send(text: "In what year did the French Revolution begin?"),
            .send(text: "How many sides does a hexagon have?"),
            .send(text: "What is the chemical symbol for gold?"),
            .regenerate,
        ],
        systemPrompt: sharedPrefix,
        sessionLabel: "kv-reuse-independent"
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
var replayHash: String?
var force = false

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
    case "--replay": replayHash = value()
    case "--force": force = true
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
          --replay <hash>       Rerun ONE recorded finding (needs --model + --out = its run dir).
          --force               Replay despite git/model drift.
        """)
        exit(0)
    default:
        fail("unknown argument '\(a)' (try --help)")
    }
    i += 1
}

guard let modelArg else { fail("--model is required (try --help)") }
if minutes == nil && iterations == nil && replayHash == nil { minutes = 5 }

let (modelURL, modelId) = resolveModel(modelArg)
// Default KV reuse ON when driving multi-turn sessions OR replaying (both chase
// the session/KV surface). Single-turn fuzz runs with reuse OFF for per-iteration
// isolation.
let kvReuse = kvReuseOverride ?? (replayHash != nil ? true : useSessionScripts)

let resolvedSeed = seedSet ? seed : UInt64.random(in: 0...UInt64.max)
let outURL = URL(fileURLWithPath: outDir, isDirectory: true)

let factory = MLXFuzzFactory(
    modelURL: modelURL,
    modelId: modelId,
    enableKVCacheReuse: kvReuse,
    contextSize: contextSize
)

// Replay mode short-circuits the campaign: rerun one recorded finding and report
// whether it reproduces. --out must point at the run dir that holds the finding.
if let hash = replayHash {
    let code = await runReplay(hash: hash, force: force, outputDir: outURL, factory: factory)
    await factory.teardown()
    exit(code)
}

FileHandle.standardError.write(Data(
    """
    fuzz-mlx: model=\(modelId) mode=\(useSessionScripts ? "session" : "single-turn") \
    kv-reuse=\(kvReuse) tools=\(useTools) \
    budget=\(minutes.map { "\($0)m" } ?? "\(iterations ?? 0) iters") \
    seed=\(resolvedSeed) out=\(outDir)

    """.utf8
))

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
