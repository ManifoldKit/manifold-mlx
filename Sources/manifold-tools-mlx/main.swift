// manifold-tools-mlx — runs ManifoldKit's bundled tool-calling scenarios
// against a real MLX model (e.g. Gemma) on Apple Silicon.
//
// This reuses ManifoldKit's published `ManifoldTools` library product: the
// bundled scenarios (`ScenarioLoader.loadBuiltIn`), the reference toolset, the
// `ScenarioRunner`, and the JSONL `TranscriptLogger`. The only thing this
// target adds is the MLX backend wiring + a small arg parser. No ManifoldKit
// core changes are required.
//
// Structure mirrors ManifoldKit's `Sources/manifold-tools/main.swift`, trimmed:
// the `test-uplift` subcommand and the Ollama / mock backend factories are
// dropped — there is exactly one backend here, MLXBackend.
import Foundation
import ManifoldInference
import ManifoldTools
import ManifoldMLX

/// Hand-rolled argument parser — pulling in `swift-argument-parser` for a small
/// harness is not worth the Package.swift churn.
struct CLI {

    var scenarioFilter: String = "all"
    var modelPath: String?
    var output: URL = defaultOutputURL()
    var fixturesRoot: URL? = nil
    var emitRecords: URL? = nil
    var list: Bool = false
    /// Number of decoy tools to advertise alongside each scenario's required
    /// tools, to measure correct-tool selection under distractor pressure.
    var extraTools: Int = 0

    static func defaultOutputURL() -> URL {
        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        return cwd.appendingPathComponent("tmp/manifold-tools-mlx/\(TranscriptLogger.defaultFilename())")
    }

    /// Argument errors exit with status 2 via `exit(2)` + stderr rather than
    /// `precondition` / `fatalError` (those trap with SIGABRT in debug builds,
    /// producing a confusing stack trace instead of the documented exit code).
    static func fail(_ message: String) -> Never {
        FileHandle.standardError.write(Data("manifold-tools-mlx: \(message)\n".utf8))
        exit(2)
    }

    static func parse(_ argv: [String]) -> CLI {
        var cli = CLI()
        var i = 0
        while i < argv.count {
            let arg = argv[i]
            switch arg {
            case "--scenario":
                i += 1
                guard i < argv.count else { fail("--scenario requires a value") }
                cli.scenarioFilter = argv[i]
            case "--model":
                i += 1
                guard i < argv.count else { fail("--model requires a value") }
                cli.modelPath = argv[i]
            case "--output":
                i += 1
                guard i < argv.count else { fail("--output requires a value") }
                cli.output = URL(fileURLWithPath: argv[i])
            case "--fixtures-root":
                i += 1
                guard i < argv.count else { fail("--fixtures-root requires a value") }
                cli.fixturesRoot = URL(fileURLWithPath: argv[i], isDirectory: true)
            case "--emit-records":
                i += 1
                guard i < argv.count else { fail("--emit-records requires a value") }
                cli.emitRecords = URL(fileURLWithPath: argv[i])
            case "--extra-tools":
                i += 1
                guard i < argv.count else { fail("--extra-tools requires a value") }
                guard let n = Int(argv[i]), n >= 0 else { fail("--extra-tools requires a non-negative integer") }
                guard n <= DecoyTools.maxCount else {
                    fail("--extra-tools \(n) exceeds the decoy pool size (\(DecoyTools.maxCount))")
                }
                cli.extraTools = n
            case "--list":
                cli.list = true
            case "--help", "-h":
                printUsage()
                exit(0)
            default:
                fail("unknown argument: \(arg)")
            }
            i += 1
        }
        return cli
    }

    static func printUsage() {
        let text = """
        manifold-tools-mlx — tool-calling validation against a real MLX model

        USAGE
          manifold-tools-mlx --model <path> [--scenario <id|all>]
                    [--output path.jsonl] [--fixtures-root <dir>] [--list]

        FLAGS
          --model <path>        REQUIRED (except for --list/--help). Path to the
                                MLX model directory (config.json + tokenizer +
                                safetensors).
          --scenario <id>       Scenario id (matches JSON 'id') or 'all'. Default: all.
          --output <path>       Transcript JSONL destination.
                                Default: tmp/manifold-tools-mlx/<iso>.jsonl.
          --emit-records <path> Write ConformanceRecord[] JSON to this path after
                                the run. Enables cross-leg matrix collation.
          --fixtures-root <dir> Root for read_file / list_dir / repo_search tools.
                                Default: the fixtures bundled with this CLI.
          --extra-tools <N>     Advertise N decoy tools alongside each scenario's
                                required tools to probe correct-tool selection
                                under distractor pressure. Default: 0 (no decoys).
          --list                Print available scenarios and exit (no model needed).
          --help                Show this text.

        EXIT
          0 — all scenarios passed.
          1 — at least one scenario or assertion failed, or a runtime error.
          2 — bad arguments.

        MLX requires Apple Silicon + Metal and a real model directory on disk.
        The transcript is one JSONL line per event so runs can be diffed.
        """
        print(text)
    }
}

/// Resolves the fixtures root: explicit `--fixtures-root` override, else the
/// `manifold-tools` directory bundled as a `.copy` resource with this CLI.
func resolveFixturesRoot(_ override: URL?) -> URL? {
    if let override { return override }
    // `.copy("Fixtures/manifold-tools")` flattens to `<bundle>/manifold-tools`
    // (SwiftPM copies the leaf directory, dropping the `Fixtures/` prefix), so
    // the unqualified lookup is the real one. The `subdirectory:` form is a
    // defensive fallback in case SwiftPM's layout ever changes.
    return Bundle.module.url(forResource: "manifold-tools", withExtension: nil)
        ?? Bundle.module.url(forResource: "manifold-tools", withExtension: nil, subdirectory: "Fixtures")
}

/// Builds a `ToolRegistry` containing ONLY the tools named in
/// `scenario.requiredTools`.
///
/// Advertising all six reference tools to every scenario overloads small models
/// (the toolset itself warns that results degrade beyond ~5 tools) and makes
/// tool-dispatch outcomes uninterpretable — you can't tell whether a miss is the
/// model failing the task or drowning in irrelevant tool definitions. Scoping
/// the registry to each scenario's declared dependency removes that confound.
///
/// A scenario with an empty `requiredTools` (e.g. `structured-json-extraction`)
/// correctly yields a registry with zero tools — that's the intended no-tool
/// behavior. An unrecognized tool name is logged to stderr and skipped rather
/// than trapping, so a future scenario referencing a tool this CLI doesn't know
/// degrades gracefully instead of crashing the whole run.
@MainActor
func makeRegistry(for scenario: Scenario, fixturesRoot: URL, extraTools: Int = 0) -> ToolRegistry {
    let registry = ToolRegistry()
    for name in scenario.requiredTools {
        switch name {
        case "now":
            registry.register(NowTool.makeExecutor())
        case "calc":
            registry.register(CalcTool.makeExecutor())
        case "read_file":
            registry.register(ReadFileTool.makeExecutor(root: fixturesRoot))
        case "list_dir":
            registry.register(ListDirTool.makeExecutor(root: fixturesRoot))
        case "sample_repo_search":
            registry.register(SampleRepoSearchTool.makeExecutor(root: fixturesRoot))
        case "http_get_fixture":
            registry.register(HttpGetFixtureTool.makeExecutor())
        default:
            FileHandle.standardError.write(Data(
                "manifold-tools-mlx: scenario '\(scenario.id)' requires unknown tool '\(name)' — skipping\n".utf8))
        }
    }
    // Pad the registry with N decoys so the model has plausible-but-wrong tools
    // to (mis)select among. Advertising them additionally requires augmenting
    // the scenario's requiredTools — see `scenarioAdvertising(_:extraToolNames:)`.
    for decoy in DecoyTools.executors(count: extraTools) {
        registry.register(decoy)
    }
    return registry
}

/// Returns a copy of `scenario` whose `requiredTools` also lists `extraToolNames`.
///
/// `ScenarioRunner` filters the advertised tool definitions to
/// `requiredTools.contains($0.name)` (ManifoldKit, read-only), so decoys added
/// to the registry are invisible to the model unless their names appear in
/// `requiredTools` too. `Scenario`'s memberwise init is internal to ManifoldKit,
/// so we patch the one field through its `Codable` representation. Augmenting
/// `requiredTools` does not affect assertions — those key off `assertion.value`,
/// not this list. A no-op when there are no decoys.
func scenarioAdvertising(_ scenario: Scenario, extraToolNames: [String]) -> Scenario {
    guard !extraToolNames.isEmpty else { return scenario }
    do {
        let data = try JSONEncoder().encode(scenario)
        var object = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
        var required = (object["requiredTools"] as? [String]) ?? []
        required.append(contentsOf: extraToolNames)
        object["requiredTools"] = required
        let patched = try JSONSerialization.data(withJSONObject: object)
        return try JSONDecoder().decode(Scenario.self, from: patched)
    } catch {
        FileHandle.standardError.write(Data(
            "manifold-tools-mlx: could not augment requiredTools for '\(scenario.id)' (\(error)); decoys will be filtered out\n".utf8))
        return scenario
    }
}

/// Fixed 3-decimal formatting for the precision/recall/F1 values so the
/// `SUMMARY` line is stable and greppable (e.g. `f1=0.900`, never `f1=0.9`).
func fmt3(_ value: Double) -> String { String(format: "%.3f", value) }

@MainActor
func runCLI() async -> Int32 {
    let argv = Array(CommandLine.arguments.dropFirst())
    // BFCL argument-level eval subcommand — drives the shared ManifoldTools
    // BFCLRunner against this package's MLXBackend (mirrors core manifold-tools).
    if argv.first == "bfcl" {
        return await BFCLMLXCLI.run(Array(argv.dropFirst()))
    }
    let cli = CLI.parse(argv)

    let scenarios: [Scenario]
    do {
        // MK 0.62 (#2042) fixed `ScenarioLoader.loadBuiltIn()` to resolve via
        // `Bundle.module` rather than CWD, so we can call it directly. The
        // previously-vendored `Scenarios/built-in/` copy was removed at that point.
        scenarios = try ScenarioLoader.loadBuiltIn()
    } catch {
        FileHandle.standardError.write(Data("failed to load scenarios: \(error)\n".utf8))
        return 1
    }

    if cli.list {
        print("Available scenarios:")
        for s in scenarios {
            print("  \(s.id) — \(s.description)")
        }
        return 0
    }

    guard let modelPath = cli.modelPath else {
        FileHandle.standardError.write(Data("manifold-tools-mlx: --model <path> is required (see --help)\n".utf8))
        return 2
    }

    // Pre-flight: refuse VL model directories before attempting to load them.
    // Loading a vision-language model via the text-only MLX backend path causes
    // a hard SIGSEGV (exit 139) with an empty log, making failures undebuggable.
    // The reliable marker of a VL model dir is any of these config files —
    // extend the array to cover future VL architectures as needed.
    let vlMarkerFiles: [String] = [
        "preprocessor_config.json",
        "processor_config.json",
        "video_preprocessor_config.json",
    ]
    let fm = FileManager.default
    var isDirectory: ObjCBool = false
    if fm.fileExists(atPath: modelPath, isDirectory: &isDirectory), isDirectory.boolValue {
        for marker in vlMarkerFiles {
            let markerPath = (modelPath as NSString).appendingPathComponent(marker)
            if fm.fileExists(atPath: markerPath) {
                CLI.fail("'\(modelPath)' looks like a vision-language model (found \(marker)); "
                    + "this is a text-only tool harness and would crash on load — skipped")
            }
        }
    }

    let filtered: [Scenario]
    if cli.scenarioFilter == "all" {
        filtered = scenarios
    } else {
        filtered = scenarios.filter { $0.id == cli.scenarioFilter }
        if filtered.isEmpty {
            CLI.fail("no scenario matches id '\(cli.scenarioFilter)' — run --list for valid IDs")
        }
    }

    guard let fixturesRoot = resolveFixturesRoot(cli.fixturesRoot) else {
        FileHandle.standardError.write(Data("could not resolve fixtures root — pass --fixtures-root explicitly\n".utf8))
        return 1
    }

    let logger: TranscriptLogger
    do {
        logger = try TranscriptLogger(url: cli.output)
    } catch {
        FileHandle.standardError.write(Data("failed to open log: \(error)\n".utf8))
        return 1
    }
    print("Logging to \(logger.destination.path)")
    print("Fixtures root: \(fixturesRoot.path)")

    // Load the MLX model once and reuse it across every scenario.
    let backend = MLXBackend()
    let modelURL = URL(fileURLWithPath: modelPath, isDirectory: true)
    print("Loading MLX model from \(modelURL.path) …")
    do {
        try await backend.loadModel(from: modelURL, plan: .systemManaged(requestedContextSize: 4096))
    } catch {
        FileHandle.standardError.write(Data("failed to load MLX model: \(error)\n".utf8))
        return 1
    }
    defer { backend.unloadModel() }

    let decoyNames = Set(DecoyTools.names(count: cli.extraTools))
    if cli.extraTools > 0 {
        print("Advertising \(cli.extraTools) decoy tool(s) per scenario: \(decoyNames.sorted().joined(separator: ", "))")
    }

    var allPassed = true
    var total = 0
    var passedCount = 0
    var cleanCount = 0
    var decoyCallTotal = 0
    for scenario in filtered {
        print("\n── \(scenario.id) via mlx ──")
        total += 1
        // Build a fresh registry scoped to this scenario's required tools, padded
        // with N decoys, then advertise the decoys by augmenting requiredTools.
        let registry = makeRegistry(for: scenario, fixturesRoot: fixturesRoot, extraTools: cli.extraTools)
        let advertised = scenarioAdvertising(scenario, extraToolNames: DecoyTools.names(count: cli.extraTools))
        do {
            // MK 0.57 (#1985) reshaped the harness: `ScenarioRunner` now drives a
            // production `InferenceService` instead of dispatching tools itself.
            // We construct a fresh service per scenario, injecting the already-
            // loaded MLX backend directly via `InferenceService(backend:…)` (the
            // harness/offline init, which marks the model loaded immediately) and
            // attaching this scenario's scoped `ToolRegistry`. The runner reads
            // `service.toolRegistry` to derive the advertised tool definitions and
            // dispatches each emitted `.toolCall` through it — no per-scenario
            // re-load of the model (the backend instance is reused).
            let service = InferenceService(
                backend: backend,
                name: "mlx",
                modelName: modelURL.lastPathComponent,
                toolRegistry: registry
            )
            let runner = ScenarioRunner(service: service, logger: logger)
            // Run the decoy-augmented scenario so the runner advertises the
            // decoys; assertions are identical (only requiredTools changed).
            let outcome = try await runner.run(advertised)
            for assertion in outcome.assertions {
                let marker = assertion.passed ? "  PASS" : "  FAIL"
                print("\(marker) \(assertion.message)")
            }
            // Correct-tool selection: a clean dispatch passes every assertion AND
            // never invokes a decoy. Calling a decoy is always a wrong selection
            // (the pool is orthogonal to every scenario's real task).
            let invoked = Set(outcome.toolCallsExecuted)
            let decoysCalled = invoked.intersection(decoyNames)
            decoyCallTotal += decoysCalled.count
            if !decoysCalled.isEmpty {
                print("  WRONG-TOOL (decoy): \(decoysCalled.sorted().joined(separator: ", "))")
            }
            // Classification counts vs the ORIGINAL required tools (not the
            // decoy-augmented set) — decoys must count as false positives.
            let expected = Set(scenario.requiredTools)
            if !expected.isEmpty {
                let counts = ConfusionCounts.compute(actual: invoked, expected: expected)
                print("  tools: tp=\(counts.tp) fp=\(counts.fp) fn=\(counts.fn) (p=\(fmt3(counts.precision)) r=\(fmt3(counts.recall)) f1=\(fmt3(counts.f1)))")
            }
            if outcome.passed { passedCount += 1 }
            if outcome.passed && decoysCalled.isEmpty { cleanCount += 1 }
            if !outcome.passed {
                allPassed = false
                print("  final answer: \(outcome.finalAnswer.prefix(200))")
            }
        } catch {
            allPassed = false
            print("  ERROR \(error)")
        }
    }

    // Re-score from the written JSONL via ConformanceScorer (MK 0.62: fixes
    // #2043/#2049 precision/recall attribution for multi-call turns).
    // `renderer` is caller-declared — ManifoldMLX uses swift-transformers for
    // chat-template application; `coreCommit` is unknown at CLI runtime.
    let scoringCtx = ConformanceScorer.RecordContext(
        renderer: "swift-transformers",
        coreCommit: "unknown",
        transcriptRef: logger.destination.path
    )
    let conformanceRecords = ConformanceScorer.records(
        fileAt: logger.destination,
        context: scoringCtx
    )
    let toolBearingRecords = conformanceRecords.filter { $0.toolSelection != nil }
    let avgPrec: Double
    let avgRecall: Double
    let avgF1: Double
    if toolBearingRecords.isEmpty {
        (avgPrec, avgRecall, avgF1) = (0.0, 0.0, 0.0)
    } else {
        let n = Double(toolBearingRecords.count)
        avgPrec   = toolBearingRecords.map { $0.toolSelection!.precision }.reduce(0, +) / n
        avgRecall = toolBearingRecords.map { $0.toolSelection!.recall }.reduce(0, +) / n
        avgF1     = toolBearingRecords.map { $0.toolSelection!.f1 }.reduce(0, +) / n
    }

    // Machine-readable one-liner the sweep script greps. Headline metric is the
    // macro-averaged F1 of correct-tool selection across tool-bearing scenarios.
    print("SUMMARY extra_tools=\(cli.extraTools) passed=\(passedCount)/\(total) clean=\(cleanCount)/\(total) precision=\(fmt3(avgPrec)) recall=\(fmt3(avgRecall)) f1=\(fmt3(avgF1)) decoy_calls=\(decoyCallTotal) scored=\(toolBearingRecords.count)")

    // Write a deterministic MATRIX.md alongside the transcript.
    let matrixURL = logger.destination
        .deletingLastPathComponent()
        .appendingPathComponent("MATRIX.md")
    let matrixMarkdown = MatrixRenderer.render(conformanceRecords)
    if (try? matrixMarkdown.write(to: matrixURL, atomically: true, encoding: .utf8)) != nil {
        print("Matrix written to \(matrixURL.path)")
    }

    // Write ConformanceRecord[] JSON for cross-leg collation if requested.
    if let emitURL = cli.emitRecords,
       let data = try? ConformanceScorer.encodeJSON(conformanceRecords) {
        if (try? data.write(to: emitURL)) != nil {
            print("Records written to \(emitURL.path)")
        }
    }

    if allPassed {
        print("\nAll scenarios passed.")
        return 0
    } else {
        print("\nOne or more scenarios failed — see \(logger.destination.path)")
        return 1
    }
}

let exitCode = await runCLI()
exit(exitCode)
