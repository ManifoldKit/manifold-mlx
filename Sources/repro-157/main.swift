// Minimal non-replay reproduction for manifold-mlx#157.
//
// The issue's two Metal command-buffer/encoder lifecycle crashes (SIGSEGV in
// a KV-cache slice write, SIGABRT adding a completion handler after commit)
// were only ever reproduced through the `fuzz-mlx --replay` path. Issue #157's
// top acceptance criterion is a repro that does NOT need the fuzz harness:
// construct an `MLXBackend`, load the model, drive the recorded trigger input
// straight through `generate(...)`.
//
// This driver does exactly that and nothing else — no ManifoldFuzz, no
// recorded findings, no drift gate. Everything below is the trigger input as
// transcribed from the issue (recorded finding 6388496947cb).
//
// Usage:
//   swift build --product repro-157
//   .build/debug/repro-157 [--turns N] [--kv-reuse on|off] [--model <path>]
//
// Exit 0 means the loop completed without crashing (NOT a clean bill of
// health — see the note printed at the end). A crash is the reproduction.

import Foundation
import ManifoldContract
import ManifoldHardware
import ManifoldMLX

// MARK: - Trigger input (issue #157, recorded finding 6388496947cb)

let systemPrompt = "You are a verbose narrator. Describe in detail."
let userTurn = "What is 2 + 2?"

let tools: [ToolDefinition] = [
    ToolDefinition(
        name: "get_weather",
        description: "Get the current weather for a city.",
        parameters: .object([
            "type": .string("object"),
            "properties": .object([
                "city": .object(["type": .string("string")]),
                "units": .object(["type": .string("string")]),
            ]),
            "required": .array([.string("city"), .string("units")]),
        ])
    ),
    ToolDefinition(
        name: "schedule_alarm",
        description: "Schedule an alarm.",
        parameters: .object([
            "type": .string("object"),
            "properties": .object([
                "label": .object(["type": .string("string")]),
                "minutes_from_now": .object(["type": .string("integer")]),
            ]),
            "required": .array([.string("label"), .string("minutes_from_now")]),
        ])
    ),
]

func triggerConfig() -> GenerationConfig {
    var config = GenerationConfig()
    config.temperature = 0.7
    config.topP = 0.9
    config.seed = 1107
    config.maxOutputTokens = 256
    config.tools = tools
    config.toolChoice = .auto
    return config
}

// MARK: - CLI

var turns = 6
var kvReuse = true
/// Cancel each turn after this many tokens and immediately start the next one.
/// `0` disables cancellation (turns run to natural completion).
var cancelAfterTokens = 0
var modelPath = NSString(string: "~/Documents/Models/mlx/Llama-3.1-8B-Instruct-4bit")
    .expandingTildeInPath

var argIndex = 1
let argv = CommandLine.arguments
while argIndex < argv.count {
    switch argv[argIndex] {
    case "--turns":
        argIndex += 1
        turns = Int(argv[argIndex]) ?? turns
    case "--kv-reuse":
        argIndex += 1
        kvReuse = argv[argIndex] == "on"
    case "--cancel-after":
        argIndex += 1
        cancelAfterTokens = Int(argv[argIndex]) ?? cancelAfterTokens
    case "--model":
        argIndex += 1
        modelPath = NSString(string: argv[argIndex]).expandingTildeInPath
    default:
        FileHandle.standardError.write(Data("repro-157: unknown argument \(argv[argIndex])\n".utf8))
        exit(2)
    }
    argIndex += 1
}

let modelURL = URL(fileURLWithPath: modelPath)
guard FileManager.default.fileExists(atPath: modelURL.path) else {
    FileHandle.standardError.write(Data("repro-157: no model at \(modelURL.path)\n".utf8))
    exit(2)
}

// MARK: - Driver

/// Drives `turns` back-to-back generations against ONE resident backend — the
/// normal chat loop the issue describes (send, next turn, …). The KV cache is
/// populated by turn 1; every later turn shares its prefix, which is the state
/// the SIGSEGV stack (`SliceUpdate::eval_gpu` → KV-cache slice write) implicates.
@MainActor
func runRepro() async {
    print("repro-157: model=\(modelURL.lastPathComponent) turns=\(turns) kvReuse=\(kvReuse)")

        let backend = MLXBackend(enableKVCacheReuse: kvReuse)
        do {
            try await backend.loadModel(
                from: modelURL,
                plan: .systemManaged(requestedContextSize: 8192)
            )
        } catch {
            FileHandle.standardError.write(Data("repro-157: load failed — \(error)\n".utf8))
            exit(2)
        }
        print("repro-157: loaded. supportsKVCachePersistence=\(backend.capabilities.supportsKVCachePersistence)")

        /// Counts spurious `alreadyGenerating` throws on the cancel → resend
        /// path. Non-zero is itself a finding, independent of any crash.
        var alreadyGeneratingThrows = 0

        for turn in 1...turns {
            let started = Date()
            var tokenCount = 0
            do {
                let stream = try backend.generate(
                    prompt: userTurn,
                    systemPrompt: systemPrompt,
                    config: triggerConfig(),
                    hints: GenerationRuntimeHints()
                )
                for try await event in stream {
                    if case .token = event { tokenCount += 1 }
                    if case .kvCacheReuse(let n) = event {
                        print("repro-157: turn \(turn) reused \(n) cached prompt tokens")
                    }
                    // Cancel mid-stream, then fall straight into the next turn
                    // without letting this one drain. This is the "cancel →
                    // resend" shape issue #157 names, and it is the only shape
                    // that can leave a command buffer committed while a
                    // completion handler is still being attached (the SIGABRT
                    // variant). Sequential run-to-completion turns do NOT
                    // reproduce either crash: 60/60 clean with --kv-reuse on,
                    // every turn reusing 543 cached prompt tokens.
                    if cancelAfterTokens > 0, tokenCount >= cancelAfterTokens {
                        backend.stopGeneration()
                        break
                    }
                }
            } catch {
                // `stopGeneration()` is not synchronous: the backend clears
                // `_isGenerating` only when the cancelled driver task actually
                // unwinds, so an immediate resend loses the race and throws
                // `alreadyGenerating`. A real app's cancel → resend hits this.
                // Back off and retry rather than exiting — exiting mid-flight
                // tears the process down while MLX is still on the GPU, which
                // produces its own (driver-artefact) crash and confounds the
                // reproduction we are actually hunting.
                let description = String(describing: error)
                if description.contains("alreadyGenerating") {
                    alreadyGeneratingThrows += 1
                    try? await Task.sleep(nanoseconds: 200_000_000)
                    continue
                }
                FileHandle.standardError.write(Data("repro-157: turn \(turn) threw — \(error)\n".utf8))
                // Let the in-flight generation settle before unwinding.
                try? await Task.sleep(nanoseconds: 2_000_000_000)
                exit(3)
            }
            let elapsed = Date().timeIntervalSince(started)
            // Non-vacuity guard: this driver is worthless if it is not actually
            // generating. The inert-fuzz incident on this very issue (core
            // #2344) reported 9,626 "clean" runs in which no generation ever
            // ran. A turn that produces no tokens is a broken harness, not a
            // passing test — fail loudly rather than counting it as evidence.
            guard tokenCount > 0 else {
                let message = "repro-157: turn \(turn) produced 0 tokens in "
                    + "\(String(format: "%.2f", elapsed))s — harness is inert, "
                    + "result is INVALID (cf. ManifoldKit#2344)\n"
                FileHandle.standardError.write(Data(message.utf8))
                exit(4)
            }
            print("repro-157: turn \(turn) ok — \(tokenCount) tokens in \(String(format: "%.2f", elapsed))s")
        }

        backend.unloadModel()
        if alreadyGeneratingThrows > 0 {
            print("repro-157: \(alreadyGeneratingThrows) spurious `alreadyGenerating` "
                + "throws on the cancel → resend path (stopGeneration is not synchronous)")
        }
        print("""
        repro-157: completed \(turns) turns without crashing.
        NOTE: the issue reports 3/3 reproduction with --kv-reuse on, so a clean
        run here is evidence about THIS configuration only, not an all-clear.
        """)
}

await runRepro()
