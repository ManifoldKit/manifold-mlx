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
    private static func fail(_ message: String) -> Never {
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

/// Loads the bundled scenarios.
///
/// `ScenarioLoader.loadBuiltIn()` resolves the scenario directory relative to
/// the *current working directory* (`<cwd>/Sources/ManifoldTools/Scenarios/built-in`),
/// which only exists inside the ManifoldKit package checkout — not in this
/// companion repo. We vendor a copy of the same JSON scenarios as a `.copy`
/// bundle resource and load them via the public `ScenarioLoader.load(from:)`
/// so the CLI is self-contained regardless of CWD.
func loadScenarios() throws -> [Scenario] {
    // `.copy("Scenarios/built-in")` flattens to `<bundle>/built-in`, so the
    // unqualified lookup resolves; the `subdirectory:` form is a fallback.
    if let bundled = Bundle.module.url(forResource: "built-in", withExtension: nil)
        ?? Bundle.module.url(forResource: "built-in", withExtension: nil, subdirectory: "Scenarios") {
        return try ScenarioLoader.load(from: bundled)
    }
    // Fall back to the CWD-relative loader (works when run from a ManifoldKit checkout).
    return try ScenarioLoader.loadBuiltIn()
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
    let cli = CLI.parse(argv)

    let scenarios: [Scenario]
    do {
        scenarios = try loadScenarios()
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

    let filtered: [Scenario]
    if cli.scenarioFilter == "all" {
        filtered = scenarios
    } else {
        filtered = scenarios.filter { $0.id == cli.scenarioFilter }
        if filtered.isEmpty {
            FileHandle.standardError.write(Data("no scenario matches id '\(cli.scenarioFilter)'\n".utf8))
            return 1
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
    // Tool-selection scored as classification (ManifoldInference 0.58): one
    // `ConfusionCounts` per tool-bearing scenario — tp = required tool invoked,
    // fp = a non-required tool (a decoy, or any other) invoked, fn = required
    // tool missed. Macro-averaged across scenarios into precision/recall/F1.
    // No-tool scenarios (empty requiredTools) are excluded from the macro metric
    // because `ConfusionCounts`'s empty-set semantics score 0.0, which would
    // wrongly penalise correct abstention; their decoy mis-fires still land in
    // `decoyCallTotal`.
    var perScenarioCounts: [ConfusionCounts] = []
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
                perScenarioCounts.append(counts)
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

    // Machine-readable one-liner the sweep script greps. Headline metric is the
    // macro-averaged F1 of correct-tool selection across tool-bearing scenarios
    // (precision falls as the model grabs decoys; recall falls as it misses the
    // required tool). `clean` and `passed` are kept as coarser companions, and
    // `decoy_calls` is the raw count of decoy invocations across all scenarios.
    let macro = MacroAveragedMetrics(perClass: perScenarioCounts)
    print("SUMMARY extra_tools=\(cli.extraTools) passed=\(passedCount)/\(total) clean=\(cleanCount)/\(total) precision=\(fmt3(macro.precision)) recall=\(fmt3(macro.recall)) f1=\(fmt3(macro.f1)) decoy_calls=\(decoyCallTotal) scored=\(perScenarioCounts.count)")

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
