import XCTest
import ManifoldInference
@_spi(Testing) import ManifoldMLX

/// Render-side golden corpus for the MLX tool-call prompt assembly.
///
/// Phase 0 / Layer-1a of the tool-calling architecture work (umbrella #2005).
///
/// ## What this guards
///
/// The MLX backend does NOT render structural tools through the tokenizer's
/// Jinja `tools=` parameter. Instead `MLXChatMessageEncoder` hand-builds a
/// per-dialect PROSE tool block (`buildQwenToolBlock` → `llamaToolBlock` /
/// `mistralToolBlock` / `qwenToolBlock`) and folds it into the system message,
/// then assembles the `[[String: String]]` message array that
/// `MLXPromptCacheCoordinator` feeds to `container.prepare(messages:)`. That
/// `(effectiveSystemPrompt, messages)` pair is the last *deterministic,
/// model-free* boundary before the real tokenizer/Metal render — everything
/// downstream needs on-disk model files and a GPU.
///
/// These tests capture that boundary byte-exact, per tool-bearing family, so an
/// upcoming ADDITIVE structural-tools change can prove it does not alter the
/// existing render for the currently-passing families (especially Llama-3.2,
/// issue #59).
///
/// ## Render path fidelity
///
/// `renderedAssembly(...)` mirrors `MLXGenerationDriver.generate(...)` lines
/// ~173–186 EXACTLY: it calls the same `MLXChatMessageEncoder.effectiveSystemPrompt`
/// then `buildChatMessages` with the same arguments the driver passes at
/// generation time. If that wiring changes, update this helper in lockstep.
///
/// ## Goldens are SELF-CAPTURED, not transformers-oracle'd
///
/// The transformers / `apply_chat_template` byte-match oracle flow (the pattern
/// used for the swift-jinja #1966 fix) lives in the ManifoldKit *core* repo
/// (`Tests/ManifoldInferenceTests/Fixtures/ChatTemplates/regenerate.py`), NOT
/// in this companion package — and the MLX prose tool blocks are a Manifold
/// invention (issues #59/#86), not a transformers chat-template feature, so
/// transformers has nothing to oracle them against anyway. The goldens here are
/// therefore SELF-CAPTURED snapshots of the current encoder output: they are a
/// **regression guard** (does the render change?), not a **correctness oracle**
/// (is the render right per the model card?). Regenerate intentionally via
/// `MANIFOLD_REGEN_RENDER_GOLDENS=1 swift test --filter MLXRenderGoldenTests`
/// after a deliberate render change, and review the diff by eye.
final class MLXRenderGoldenTests: XCTestCase {

    // MARK: - Canonical conversation (mirrors core's RenderConsistencyChecker probe)

    /// The single tool the canonical conversation exercises: weather-in-a-city.
    private func weatherTool() -> ToolDefinition {
        ToolDefinition(
            name: "get_weather",
            description: "Get the current weather for a location.",
            parameters: .object([
                "type": .string("object"),
                "properties": .object([
                    "location": .object([
                        "type": .string("string"),
                        "description": .string("The city, e.g. Paris"),
                    ]),
                ]),
                "required": .array([.string("location")]),
            ])
        )
    }

    /// Fixed canonical conversation: a user asks for the weather in Paris, the
    /// assistant emits a `get_weather({"location":"Paris"})` tool call, and a
    /// tool turn returns the result. This is the same probe shape core's
    /// `RenderConsistencyChecker` uses, so cross-backend goldens stay
    /// comparable.
    private func canonicalHistory() -> [ToolAwareHistoryEntry] {
        [
            ToolAwareHistoryEntry(
                role: "user",
                content: "What's the weather in Paris?"
            ),
            ToolAwareHistoryEntry(
                role: "assistant",
                content: "",
                toolCalls: [
                    ToolCall(
                        id: "call_1",
                        toolName: "get_weather",
                        arguments: #"{"location":"Paris"}"#
                    )
                ]
            ),
            ToolAwareHistoryEntry(
                role: "tool",
                content: #"{"temperature":"18C","condition":"cloudy"}"#,
                toolCallId: "call_1"
            ),
        ]
    }

    private func config(withTools tools: [ToolDefinition]) -> GenerationConfig {
        var c = GenerationConfig()
        c.tools = tools
        return c
    }

    // MARK: - Render boundary (mirrors MLXGenerationDriver lines ~173–186)

    /// Produces the deterministic, model-free `(effectiveSystemPrompt, messages)`
    /// the driver hands to `MLXPromptCacheCoordinator.prepareInputAndCache` at
    /// generation time, then serialises it to a stable, diff-friendly text form.
    ///
    /// For the text-only tool-call path (no image attachments) `chatMessages` is
    /// `nil` and the tokenizer consumes the `messages` array, so that array —
    /// plus the effective system prompt that fed it — is the full render input.
    private func renderedAssembly(
        dialect: MLXToolDialect,
        appSystemPrompt: String?
    ) throws -> String {
        let cfg = config(withTools: [weatherTool()])

        // EXACT mirror of MLXGenerationDriver.generate step 1.
        let effectiveSystemPrompt = MLXChatMessageEncoder.effectiveSystemPrompt(
            systemPrompt: appSystemPrompt,
            config: cfg,
            dialect: dialect
        )

        // EXACT mirror of MLXGenerationDriver.generate step 2.
        let (chatMessages, messages) = try MLXChatMessageEncoder.buildChatMessages(
            prompt: "What's the weather in Paris?",
            effectiveSystemPrompt: effectiveSystemPrompt,
            conversationHistory: [],
            toolAwareHistory: canonicalHistory(),
            structuredHistory: nil,
            dialect: dialect
        )

        // The text-only tool path never produces structured chat messages
        // (those carry image attachments only). Assert the premise so a future
        // change that routes tool turns through the vision path can't silently
        // bypass this golden.
        XCTAssertNil(
            chatMessages,
            "text-only tool render must not produce structured Chat.Message history"
        )

        // EXACT mirror of MLXGenerationDriver.generate step 3 (Phase 0 / #2005):
        // the structural `tools` array threaded into
        // `applyChatTemplate(messages:tools:)`. For tools-aware-template
        // dialects (Mistral) this is the native-tool-render input — non-nil here
        // means the model's own template (NOT a prose block) produces the
        // `[AVAILABLE_TOOLS]` / `[TOOL_CALLS]` shape downstream. `nil` for the
        // prose dialects (Llama/Qwen), so their golden is byte-unchanged.
        let toolSpecs = MLXChatMessageEncoder.structuralToolSpecs(
            config: cfg,
            dialect: dialect
        )

        return Self.serialize(messages: messages, toolSpecs: toolSpecs)
    }

    /// Serialises the `[[String: String]]` message array to a stable,
    /// human-diffable text block. Each turn is rendered as a `=== role ===`
    /// header followed by its content. A trailing newline keeps the golden file
    /// POSIX-clean.
    ///
    /// The content is JSON-canonicalised (see ``canonicalizeJSONSpans(in:)``)
    /// because the encoder serialises the tool descriptors and replayed calls
    /// from unordered `[String: Any]` dictionaries — raw `JSONSerialization`
    /// key order is hash-seeded and varies run-to-run, which would make a
    /// byte-exact golden flaky. Canonicalising to sorted keys captures the
    /// render's *content* deterministically without changing encoder behaviour.
    /// (The key-ordering instability is an encoder property this golden does
    /// NOT assert on; the upcoming structural-tools change should consider
    /// emitting sorted keys, but that is out of scope for this goldens-only PR.)
    private static func serialize(
        messages: [[String: String]],
        toolSpecs: [[String: any Sendable]]? = nil
    ) -> String {
        var out = ""
        for msg in messages {
            let role = msg["role"] ?? "<missing-role>"
            let content = canonicalizeJSONSpans(in: msg["content"] ?? "")
            out += "=== \(role) ===\n"
            out += content
            out += "\n"
        }
        // The structural-tools section captures the `tools=` array now threaded
        // into `applyChatTemplate(messages:tools:)` (Phase 0 / #2005). Its
        // presence is the deterministic, model-free proof that a tools-aware
        // template (Mistral) renders its native tool block — the actual
        // `[AVAILABLE_TOOLS]` Jinja expansion needs the on-disk template + Metal
        // and lives in the live soak. Serialised with sorted keys so the golden
        // is stable; `nil`/empty omits the section entirely (prose dialects).
        if let toolSpecs, !toolSpecs.isEmpty {
            out += "=== tools (structural, threaded into applyChatTemplate) ===\n"
            out += serializeToolSpecs(toolSpecs)
            out += "\n"
        }
        return out
    }

    /// Serialises the structural tool-spec array to stable, sorted-key JSON.
    private static func serializeToolSpecs(_ toolSpecs: [[String: any Sendable]]) -> String {
        guard
            let data = try? JSONSerialization.data(
                withJSONObject: toolSpecs,
                options: [.prettyPrinted, .sortedKeys]
            ),
            let str = String(data: data, encoding: .utf8)
        else {
            return "<unserializable tool specs>"
        }
        return str
    }

    /// Rewrites every balanced top-level JSON object/array span in `text` to its
    /// sorted-key canonical form, leaving the surrounding prose untouched.
    ///
    /// Scans for a `{` or `[` that begins a span parseable by
    /// `JSONSerialization`, re-emits it with `.sortedKeys` (and `.prettyPrinted`
    /// when the original span spanned multiple lines, to preserve the readable
    /// block shape), and copies all other characters verbatim. A `{`/`[` that
    /// does not begin valid JSON (e.g. the literal `{"name": <function-name>}`
    /// instruction templates, which contain `<…>` placeholders) is copied
    /// through unchanged.
    private static func canonicalizeJSONSpans(in text: String) -> String {
        let chars = Array(text)
        var out = ""
        var i = 0
        while i < chars.count {
            let c = chars[i]
            guard c == "{" || c == "[" else {
                out.append(c)
                i += 1
                continue
            }
            // Find the matching close by bracket-counting, respecting strings.
            if let end = matchingBracketEnd(chars, from: i),
               let canonical = canonicalJSON(String(chars[i...end])) {
                out += canonical
                i = end + 1
            } else {
                out.append(c)
                i += 1
            }
        }
        return out
    }

    /// Index of the bracket that closes the one at `start`, or nil if unbalanced.
    private static func matchingBracketEnd(_ chars: [Character], from start: Int) -> Int? {
        var depth = 0
        var inString = false
        var escaped = false
        var i = start
        while i < chars.count {
            let c = chars[i]
            if inString {
                if escaped { escaped = false }
                else if c == "\\" { escaped = true }
                else if c == "\"" { inString = false }
            } else {
                switch c {
                case "\"": inString = true
                case "{", "[": depth += 1
                case "}", "]":
                    depth -= 1
                    if depth == 0 { return i }
                default: break
                }
            }
            i += 1
        }
        return nil
    }

    /// Re-serialises a JSON string with sorted keys, or nil if it isn't valid
    /// JSON. Pretty-prints when the input was multi-line so block descriptors
    /// stay readable; compact otherwise so inline calls stay on one line.
    private static func canonicalJSON(_ span: String) -> String? {
        guard let data = span.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])
        else { return nil }
        var opts: JSONSerialization.WritingOptions = [.sortedKeys]
        if span.contains("\n") { opts.insert(.prettyPrinted) }
        guard let out = try? JSONSerialization.data(withJSONObject: obj, options: opts),
              let str = String(data: out, encoding: .utf8)
        else { return nil }
        return str
    }

    // MARK: - Golden comparison

    private func assertGolden(
        _ rendered: String,
        named goldenName: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        // Opt-in regeneration: writes the current render back to the SOURCE
        // fixtures dir (not the test bundle copy) so a `git diff` shows the
        // change for review. Never runs in CI.
        if ProcessInfo.processInfo.environment["MANIFOLD_REGEN_RENDER_GOLDENS"] == "1" {
            let sourceURL = Self.sourceFixtureURL(for: goldenName)
            try FileManager.default.createDirectory(
                at: sourceURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try rendered.write(to: sourceURL, atomically: true, encoding: .utf8)
            // Still assert against the freshly-written file so the run is green.
        }

        let expected = try Self.loadGolden(named: goldenName)
        XCTAssertEqual(
            rendered,
            expected,
            """
            Render-side golden mismatch for "\(goldenName)".

            The MLX encoder's deterministic render of the canonical tool-call \
            conversation changed. If this change is INTENTIONAL, regenerate the \
            goldens with:

              MANIFOLD_REGEN_RENDER_GOLDENS=1 swift test --filter MLXRenderGoldenTests

            and review the resulting `git diff` by eye. These are self-captured \
            regression guards, not a transformers correctness oracle.
            """,
            file: file,
            line: line
        )
    }

    /// Loads a checked-in golden from the test bundle (CI-safe; no filesystem
    /// assumptions beyond `Bundle.module`).
    private static func loadGolden(named name: String) throws -> String {
        // `.copy("Fixtures/RenderGoldens")` lands the directory at the bundle
        // root as `RenderGoldens/` (the `Fixtures/` prefix is stripped).
        guard let url = Bundle.module.url(
            forResource: name,
            withExtension: "txt",
            subdirectory: "RenderGoldens"
        ) else {
            throw XCTSkip(
                "Missing render golden \"\(name).txt\". Generate it with " +
                "MANIFOLD_REGEN_RENDER_GOLDENS=1 swift test --filter MLXRenderGoldenTests."
            )
        }
        return try String(contentsOf: url, encoding: .utf8)
    }

    /// Resolves the SOURCE fixture path (under `Tests/...`) for regeneration.
    /// `#filePath` points at this file in the source tree, so we walk up to the
    /// test-target root and into the fixtures dir.
    private static func sourceFixtureURL(for name: String) -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent() // .../ManifoldMLXTests
            .appendingPathComponent("Fixtures/RenderGoldens/\(name).txt")
    }

    // MARK: - Tests (one per tool-bearing dialect)

    /// Llama 3.x — the family the architecture change must NOT regress (#59).
    func test_render_llama_canonicalToolCall() throws {
        let rendered = try renderedAssembly(
            dialect: .llama,
            appSystemPrompt: "You are a helpful assistant."
        )
        try assertGolden(rendered, named: "llama-weather")
    }

    /// Qwen 2.5 / Qwen 3 — `<tools>…</tools>` + `<tool_call>` wire format.
    func test_render_qwen_canonicalToolCall() throws {
        let rendered = try renderedAssembly(
            dialect: .qwen25,
            appSystemPrompt: "You are a helpful assistant."
        )
        try assertGolden(rendered, named: "qwen-weather")
    }

    /// Mistral v0.3 — `[TOOL_CALLS]` array wire format. NOTE: this captures the
    /// assembly BEFORE `foldSystemIntoFirstUser`, which the coordinator applies
    /// only when the live template rejects a standalone `system` role. The fold
    /// is render-time recovery and needs the real template to trigger, so the
    /// deterministic golden is the pre-fold shape; a separate test pins the fold.
    func test_render_mistral_canonicalToolCall() throws {
        let rendered = try renderedAssembly(
            dialect: .mistral,
            appSystemPrompt: "You are a helpful assistant."
        )
        try assertGolden(rendered, named: "mistral-weather")
    }

    /// Mistral's system-hostile fallback: the `foldSystemIntoFirstUser` recovery
    /// the coordinator applies when the real template rejects `system`. Pinned
    /// here because it is the shape Mistral models actually see, and it is fully
    /// deterministic (no template needed to compute the fold itself).
    func test_render_mistral_systemFolded_canonicalToolCall() throws {
        let cfg = config(withTools: [weatherTool()])
        let effectiveSystemPrompt = MLXChatMessageEncoder.effectiveSystemPrompt(
            systemPrompt: "You are a helpful assistant.",
            config: cfg,
            dialect: .mistral
        )
        let (chatMessages, messages) = try MLXChatMessageEncoder.buildChatMessages(
            prompt: "What's the weather in Paris?",
            effectiveSystemPrompt: effectiveSystemPrompt,
            conversationHistory: [],
            toolAwareHistory: canonicalHistory(),
            structuredHistory: nil,
            dialect: .mistral
        )
        XCTAssertNil(chatMessages)
        let folded = MLXChatMessageEncoder.foldSystemIntoFirstUser(messages)
        try assertGolden(Self.serialize(messages: folded), named: "mistral-weather-system-folded")
    }

    // MARK: - Premise tripwires
    //
    // Cheap structural assertions that fail loudly if the encoder's tool-block
    // dispatch changes shape, independent of the byte-exact goldens. These keep
    // a stale golden from masking a wholesale dispatch break.

    func test_premise_proseDialects_injectAProseBlock() {
        // Llama and Qwen keep the hand-built prose tool block (Llama's native
        // tokens are dropped by the MLX detokenizer — issue #59). Their render
        // goldens depend on it.
        let cfg = config(withTools: [weatherTool()])
        for dialect in [MLXToolDialect.llama, .qwen25] {
            XCTAssertNotNil(
                MLXChatMessageEncoder.buildQwenToolBlock(config: cfg, dialect: dialect),
                "\(dialect) must inject a prose tool block; render goldens depend on it"
            )
            XCTAssertNil(
                MLXChatMessageEncoder.structuralToolSpecs(config: cfg, dialect: dialect),
                "\(dialect) must NOT thread structural tools (prose path only)"
            )
        }
    }

    func test_premise_mistral_usesStructuralToolsNotProse() {
        // Phase 0 / #2005, F3: Mistral renders tools structurally through the
        // chat template, so the prose block is gated off and the structural
        // tool specs are threaded into applyChatTemplate(messages:tools:).
        let cfg = config(withTools: [weatherTool()])
        XCTAssertNil(
            MLXChatMessageEncoder.buildQwenToolBlock(config: cfg, dialect: .mistral),
            "Mistral prose block must be gated off in favour of structural tools"
        )
        XCTAssertNotNil(
            MLXChatMessageEncoder.structuralToolSpecs(config: cfg, dialect: .mistral),
            "Mistral must thread structural tools into the chat template"
        )
    }

    func test_premise_unknownDialect_injectsNoBlock() {
        // Gemma and other families map to `.unknown` (no tool dialect path), so
        // there is neither a prose tool block nor structural tools to golden —
        // documented gap, not a bug.
        let cfg = config(withTools: [weatherTool()])
        XCTAssertNil(
            MLXChatMessageEncoder.buildQwenToolBlock(config: cfg, dialect: .unknown),
            ".unknown (e.g. Gemma) has no tool dialect, so no render golden applies"
        )
        XCTAssertNil(
            MLXChatMessageEncoder.structuralToolSpecs(config: cfg, dialect: .unknown),
            ".unknown (e.g. Gemma) threads no structural tools"
        )
    }
}
