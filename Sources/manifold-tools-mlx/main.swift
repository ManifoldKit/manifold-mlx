// manifold-tools-mlx — runs ManifoldKit's bundled tool-calling scenarios
// against a real MLX model (e.g. Gemma) on Apple Silicon.
//
// This reuses ManifoldKit's published `ManifoldTools` library product: the
// bundled scenarios (`ScenarioLoader.loadBuiltIn`), the bundled fixture tree
// (`ToolFixtures.bundledRoot()` via `ScenarioCLIHarness.resolveFixturesRoot`),
// the reference toolset, `ScenarioRunner`, `VLModelDetector`, the JSONL
// `TranscriptLogger`, and the shared `ScenarioCLIHarness` (common flag
// parsing, scenario filtering, the run loop, and exit-code policy). The only
// things this target adds are the MLX backend wiring, the `--extra-tools`
// decoy pool, the `bfcl` subcommand, and per-run conformance-record/matrix
// output. No ManifoldKit core changes are required.
//
// Structure mirrors ManifoldKit's `Sources/manifold-tools/main.swift`, trimmed:
// the `test-uplift` subcommand and the Ollama / mock backend factories are
// dropped — there is exactly one backend here, MLXBackend.
import Foundation
import ManifoldInference
import ManifoldTools
import ManifoldMLX

/// Hand-rolled argument parser for this CLI's own flags (`--model`,
/// `--emit-records`) — pulling in `swift-argument-parser` for a small harness
/// is not worth the Package.swift churn. Flags shared with the companion CLIs
/// (`--scenario`, `--output`, `--fixtures-root`, `--extra-tools`, `--list`,
/// `--help`) are parsed by `ScenarioCLIHarness.parseCommonFlags`.
struct CLI {

    var common: ScenarioCLIHarness.Options
    var modelPath: String?
    var emitRecords: URL? = nil

    var scenarioFilter: String { common.scenarioFilter }
    var output: URL { common.output }
    var fixturesRoot: URL? { common.fixturesRoot }
    var list: Bool { common.list }
    /// Number of decoy tools to advertise alongside each scenario's required
    /// tools, to measure correct-tool selection under distractor pressure.
    var extraTools: Int { common.extraTools }

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

    /// Parses the flags common to every scenario-CLI harness consumer via
    /// `ScenarioCLIHarness`, then walks the remainder for this CLI's own
    /// flags (`--model`, `--emit-records`).
    static func parse(_ argv: [String]) -> CLI {
        let commonOptions: ScenarioCLIHarness.Options
        let remainder: [String]
        switch ScenarioCLIHarness.parseCommonFlags(argv, defaultOutput: defaultOutputURL()) {
        case .options(let options, let rest):
            commonOptions = options
            remainder = rest
        case .helpRequested:
            printUsage()
            exit(0)
        case .failure(let message):
            fail(message)
        }
        // `ScenarioCLIHarness.parseCommonFlags` only enforces `--extra-tools`
        // is a non-negative integer — this CLI's decoy pool is finite, so
        // re-validate the shared-parsed value against its bound here.
        guard commonOptions.extraTools <= DecoyTools.maxCount else {
            fail("--extra-tools \(commonOptions.extraTools) exceeds the decoy pool size (\(DecoyTools.maxCount))")
        }

        var cli = CLI(common: commonOptions)
        var i = 0
        while i < remainder.count {
            let arg = remainder[i]
            switch arg {
            case "--model":
                i += 1
                guard i < remainder.count else { fail("--model requires a value") }
                cli.modelPath = remainder[i]
            case "--emit-records":
                i += 1
                guard i < remainder.count else { fail("--emit-records requires a value") }
                cli.emitRecords = URL(fileURLWithPath: remainder[i])
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

        SUBCOMMANDS
          bfcl      BFCL argument-level eval (run `bfcl --help` for flags)

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
                                Default: the fixtures bundled with ManifoldTools.
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

/// Builds a `ToolRegistry` containing ONLY the tools named in
/// `scenario.requiredTools`, padded with `extraTools` decoys.
///
/// Advertising all six reference tools to every scenario overloads small models
/// (the toolset itself warns that results degrade beyond ~5 tools) and makes
/// tool-dispatch outcomes uninterpretable — you can't tell whether a miss is the
/// model failing the task or drowning in irrelevant tool definitions. Scoping
/// the registry to each scenario's declared dependency removes that confound.
///
/// A scenario with an empty `requiredTools` (e.g. `structured-json-extraction`)
/// correctly yields a registry with zero required tools (still padded with
/// decoys when `extraTools > 0`) — that's the intended no-tool behavior. An
/// unrecognized tool name is logged to stderr and skipped rather than
/// trapping, so a future scenario referencing a tool this CLI doesn't know
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
    // Pad the registry with N decoys so the model has plausible-but-wrong
    // tools to (mis)select among. Advertising them is handled by
    // `ScenarioRunner(passAllRegisteredTools:)` — see the call site — rather
    // than by patching `scenario.requiredTools`.
    for decoy in DecoyTools.executors(count: extraTools) {
        registry.register(decoy)
    }
    return registry
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
        // ManifoldKit 0.62 (#2042) fixed `ScenarioLoader.loadBuiltIn()` to
        // resolve via `Bundle.module` rather than CWD, so it's callable
        // directly from this companion package with no vendored copy.
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
    // Loading a vision-language model via the text-only MLX backend path
    // causes a hard SIGSEGV (exit 139) with an empty log, making failures
    // undebuggable. `VLModelDetector` is ManifoldKit's single canonical
    // implementation of this check (both manifold-mlx and manifold-llama had
    // independently hand-rolled it).
    let modelDirectory = URL(fileURLWithPath: modelPath, isDirectory: true)
    if let marker = VLModelDetector.matchedMarkerFile(at: modelDirectory) {
        CLI.fail("'\(modelPath)' looks like a vision-language model (found \(marker)); "
            + "this is a text-only tool harness and would crash on load — skipped")
    }

    // NOTE: unlike core `manifold-tools` (which treats a scenario-id mismatch
    // as a runtime failure, exit 1), this CLI has always treated it as a bad
    // argument (exit 2, standard "manifold-tools-mlx: " prefix) — regression-
    // guarded by `CLIParseTests.test_unknownScenario_exits2_withStandardPrefix`.
    // `CLI.fail` (not the generic catch-and-return-1 core uses) preserves that
    // established, tested contract; flagged as an ambiguity in the PR
    // description against fully unifying onto the shared harness's exit code.
    let filtered: [Scenario]
    do {
        filtered = try ScenarioCLIHarness.filterScenarios(scenarios, matching: cli.scenarioFilter)
    } catch {
        CLI.fail("\(error)")
    }

    let fixturesRoot = ScenarioCLIHarness.resolveFixturesRoot(cli.fixturesRoot)

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
    let modelURL = modelDirectory
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

    // Counters captured by the `runOne` closure below — `ScenarioCLIHarness.runAll`
    // owns the (scenario × model) iteration and PASS/FAIL/error printing; this
    // CLI's decoy accounting and per-scenario tool-selection metrics still need
    // to happen per-run, so they're computed inside `runOne` against the
    // returned `Outcome` and folded into these mutable totals.
    var total = 0
    var passedCount = 0
    var cleanCount = 0
    var decoyCallTotal = 0

    let allPassed = await ScenarioCLIHarness.runAll(
        scenarios: filtered,
        displayName: "mlx",
        modelsFor: { _ in [modelURL.lastPathComponent] }
    ) { scenario, _ in
        total += 1
        // Build a fresh registry scoped to this scenario's required tools,
        // padded with N decoys.
        let registry = makeRegistry(for: scenario, fixturesRoot: fixturesRoot, extraTools: cli.extraTools)
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
        // `passAllRegisteredTools` advertises every registered tool (required
        // + decoys) to the model when decoys are present — the harness-native
        // replacement for the old requiredTools-patching hack.
        let runner = ScenarioRunner(
            service: service,
            logger: logger,
            passAllRegisteredTools: cli.extraTools > 0
        )
        let outcome = try await runner.run(scenario)

        // Correct-tool selection: a clean dispatch passes every assertion AND
        // never invokes a decoy. Calling a decoy is always a wrong selection
        // (the pool is orthogonal to every scenario's real task).
        let invoked = Set(outcome.toolCallsExecuted)
        let decoysCalled = invoked.intersection(decoyNames)
        decoyCallTotal += decoysCalled.count
        if !decoysCalled.isEmpty {
            print("  WRONG-TOOL (decoy): \(decoysCalled.sorted().joined(separator: ", "))")
        }
        // Classification counts vs the required tools — decoys must count as
        // false positives.
        let expected = Set(scenario.requiredTools)
        if !expected.isEmpty {
            let counts = ConfusionCounts.compute(actual: invoked, expected: expected)
            print("  tools: tp=\(counts.tp) fp=\(counts.fp) fn=\(counts.fn) (p=\(fmt3(counts.precision)) r=\(fmt3(counts.recall)) f1=\(fmt3(counts.f1)))")
        }
        if outcome.passed { passedCount += 1 }
        if outcome.passed && decoysCalled.isEmpty { cleanCount += 1 }
        return outcome
    }

    // Re-score from the written JSONL via ConformanceScorer (MK 0.62: fixes
    // #2043/#2049 precision/recall attribution for multi-call turns). This is
    // deliberately NOT `ScenarioCLIHarness.printToolSelectionSummary` — that
    // helper macro-averages live per-scenario `ConfusionCounts`, which is the
    // pre-#2043/#2049 computation this CLI moved away from because it
    // mis-attributed precision/recall on multi-call turns. Re-scoring from the
    // transcript is the more accurate source; see the PR description for this
    // as a flagged ambiguity against unifying onto the shared helper.
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

    return ScenarioCLIHarness.finish(allPassed: allPassed, transcriptPath: cli.output)
}

let exitCode = await runCLI()
exit(exitCode)
