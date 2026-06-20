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
    return Bundle.module.url(forResource: "manifold-tools", withExtension: nil, subdirectory: "Fixtures")
        ?? Bundle.module.url(forResource: "manifold-tools", withExtension: nil)
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
    if let bundled = Bundle.module.url(forResource: "built-in", withExtension: nil, subdirectory: "Scenarios")
        ?? Bundle.module.url(forResource: "built-in", withExtension: nil) {
        return try ScenarioLoader.load(from: bundled)
    }
    // Fall back to the CWD-relative loader (works when run from a ManifoldKit checkout).
    return try ScenarioLoader.loadBuiltIn()
}

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

    let registry = ToolRegistry()
    registry.register(NowTool.makeExecutor())
    registry.register(CalcTool.makeExecutor())
    registry.register(ReadFileTool.makeExecutor(root: fixturesRoot))
    registry.register(ListDirTool.makeExecutor(root: fixturesRoot))
    registry.register(SampleRepoSearchTool.makeExecutor(root: fixturesRoot))
    registry.register(HttpGetFixtureTool.makeExecutor())

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

    var allPassed = true
    for scenario in filtered {
        print("\n── \(scenario.id) via mlx ──")
        do {
            let runner = ScenarioRunner(backend: backend, registry: registry, logger: logger)
            let outcome = try await runner.run(scenario)
            for assertion in outcome.assertions {
                let marker = assertion.passed ? "  PASS" : "  FAIL"
                print("\(marker) \(assertion.message)")
            }
            if !outcome.passed {
                allPassed = false
                print("  final answer: \(outcome.finalAnswer.prefix(200))")
            }
        } catch {
            allPassed = false
            print("  ERROR \(error)")
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
