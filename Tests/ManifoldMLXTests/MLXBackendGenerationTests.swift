import XCTest
import MLXLMCommon
import ManifoldRuntime
import ManifoldPersistenceSwiftData
import ManifoldInference
import ManifoldTestSupport
import ManifoldMLX
@_spi(Testing) import ManifoldMLX

// Conform MockMLXModelContainer to the internal protocol in this test target,
// where both the internal protocol and the public mock type are visible.
extension MockMLXModelContainer: MLXModelContainerProtocol {
    func prepare(messages: [[String : String]]) async throws -> MLXPreparedInput {
        let promptTokenIds = try await prepareForGeneration(messages: messages)
        return MLXPreparedInput(promptTokenIds: promptTokenIds)
    }

    func prepare(
        messages: [[String: String]],
        tools: [[String: any Sendable]]?
    ) async throws -> MLXPreparedInput {
        recordPrepareTools(tools)
        let promptTokenIds = try await prepareForGeneration(messages: messages)
        return MLXPreparedInput(promptTokenIds: promptTokenIds)
    }

    func prepare(chat: SendableChatMessages) async throws -> MLXPreparedInput {
        let promptTokenIds = try await prepareForGeneration(chat: chat.value)
        return MLXPreparedInput(promptTokenIds: promptTokenIds)
    }

    func makeCache(parameters: GenerateParameters) async throws -> MLXPromptCache {
        MLXPromptCache(makeCacheForGeneration(parameters: parameters))
    }

    func generate(
        input: MLXPreparedInput,
        cache: MLXPromptCache?,
        parameters: GenerateParameters
    ) async throws -> AsyncStream<Generation> {
        try await generatePreparedInput(
            promptTokenIds: input.promptTokenIds,
            cache: cache.map { SendableKVCacheList($0.value) },
            parameters: parameters
        )
    }
}

/// Unit tests for `MLXBackend.generate()` using `MockMLXModelContainer`.
///
/// These tests run in CI without Apple Silicon — the generation path is driven
/// entirely by the injected mock which never touches the Metal GPU stack.
/// `test_sendableLMInput_wrapsAndUnwraps` is limited to compile-time/type-level
/// wrapping checks and does not perform an MLX/Metal runtime round trip.
final class MLXBackendGenerationTests: XCTestCase {

    // MARK: - Helpers

    /// Drains all `.token` events from a `GenerationStream` into an ordered array.
    private func collectTokens(
        from stream: GenerationStream
    ) async throws -> [String] {
        var tokens: [String] = []
        for try await event in stream.events {
            if case .token(let text) = event {
                tokens.append(text)
            }
        }
        return tokens
    }

    private final class YieldCounter: @unchecked Sendable {
        private let lock = NSLock()
        private var _count = 0

        func increment() {
            lock.lock(); defer { lock.unlock() }
            _count += 1
        }

        var count: Int {
            lock.lock(); defer { lock.unlock() }
            return _count
        }
    }

    // MARK: - test_generate_yieldsInjectedTokens

    func test_generate_yieldsInjectedTokens() async throws {
        let mock = MockMLXModelContainer()
        mock.tokensToYield = ["Hello", " world"]

        let backend = MLXBackend()
        backend._inject(mock)

        let stream = try backend.generate(
            prompt: "hi",
            systemPrompt: nil,
            config: GenerationConfig()
        )

        let tokens = try await collectTokens(from: stream)

        XCTAssertEqual(tokens, ["Hello", " world"],
            "Stream must yield exactly the injected tokens in order")
        XCTAssertEqual(mock.generateCallCount, 1)

        // Verify the messages were assembled with user role.
        XCTAssertEqual(mock.lastMessages?.last?["role"], "user")
        XCTAssertEqual(mock.lastMessages?.last?["content"], "hi")

        // Sabotage check: tokensToYield = [] would produce an empty array,
        // failing the equality assertion.
    }

    // MARK: - test_generate_respectsMaxOutputTokens

    func test_generate_respectsMaxOutputTokens() async throws {
        let mock = MockMLXModelContainer()
        mock.tokensToYield = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

        let backend = MLXBackend()
        backend._inject(mock)

        var config = GenerationConfig()
        config.maxOutputTokens = 3

        let stream = try backend.generate(
            prompt: "hi",
            systemPrompt: nil,
            config: config
        )

        let tokens = try await collectTokens(from: stream)

        XCTAssertEqual(tokens.count, 3,
            "Backend must stop yielding after maxOutputTokens tokens")
        XCTAssertEqual(tokens, ["A", "B", "C"])

        // Sabotage check: setting maxOutputTokens = 100 yields all 10 tokens,
        // causing the count assertion to fail.
    }

    func test_generate_withVisionHistory_preparesStructuredChat() async throws {
        let mock = MockMLXModelContainer()
        mock.tokensToYield = ["seen"]

        let backend = MLXBackend()
        backend._inject(mock, supportsVision: true)
        backend.setStructuredHistory([
            StructuredMessage(role: "user", parts: [
                .text("Describe this image."),
                .image(data: ImageFixtures.oneByOnePNGData, mimeType: "image/png"),
            ])
        ])

        let stream = try backend.generate(
            prompt: "fallback",
            systemPrompt: nil,
            config: GenerationConfig()
        )

        let tokens = try await collectTokens(from: stream)

        XCTAssertEqual(tokens, ["seen"])
        XCTAssertEqual(mock.lastChatMessages?.count, 1)
        XCTAssertEqual(mock.lastChatMessages?.first?.content, "Describe this image.")
        XCTAssertEqual(mock.lastChatMessages?.first?.images.count, 1)
        XCTAssertNil(mock.lastMessages, "Vision turns should go through UserInput(chat:) instead of text-only message dictionaries")
    }

    func test_generate_toolAwareVisionHistory_preservesOriginalImages() async throws {
        let mock = MockMLXModelContainer()
        mock.tokensToYield = ["ok"]

        let backend = MLXBackend()
        backend._inject(mock, supportsVision: true, dialect: .qwen25)
        backend.setStructuredHistory([
            StructuredMessage(role: "user", parts: [
                .text("Please inspect this chart."),
                .image(data: ImageFixtures.oneByOnePNGData, mimeType: "image/png"),
            ])
        ])
        backend.setToolAwareHistory([
            ToolAwareHistoryEntry(role: "user", content: "Please inspect this chart."),
            ToolAwareHistoryEntry(
                role: "assistant",
                content: "",
                toolCalls: [
                    ToolCall(id: "call_1", toolName: "summarize_image", arguments: "{}")
                ]
            ),
        ])

        _ = try await collectTokens(from: try backend.generate(
            prompt: "fallback",
            systemPrompt: nil,
            config: GenerationConfig()
        ))

        XCTAssertEqual(mock.lastChatMessages?.count, 2)
        XCTAssertEqual(mock.lastChatMessages?.first?.images.count, 1)
        XCTAssertTrue(
            mock.lastChatMessages?[1].content.contains("<tool_call>") == true,
            "Tool-aware vision turns must keep the serialized tool call content while preserving the earlier image"
        )
    }

    func test_generate_withVisionHistory_andSystemPrompt_prependsSystemChatMessage() async throws {
        let mock = MockMLXModelContainer()
        mock.tokensToYield = ["seen"]

        let backend = MLXBackend()
        backend._inject(mock, supportsVision: true)
        backend.setStructuredHistory([
            StructuredMessage(role: "user", parts: [
                .text("Describe this image."),
                .image(data: ImageFixtures.oneByOnePNGData, mimeType: "image/png"),
            ])
        ])

        _ = try await collectTokens(from: try backend.generate(
            prompt: "fallback",
            systemPrompt: "You are a precise image analyst.",
            config: GenerationConfig()
        ))

        XCTAssertEqual(mock.lastChatMessages?.count, 2)
        XCTAssertEqual(mock.lastChatMessages?.first?.content, "You are a precise image analyst.")
        XCTAssertEqual(mock.lastChatMessages?.first?.images.count, 0)
        XCTAssertEqual(mock.lastChatMessages?[1].images.count, 1)
    }

    func test_generate_reusesPromptCachePrefixOnMatchingTurn() async throws {
        let mock = MockMLXModelContainer()
        mock.tokensToYield = ["ok"]
        mock.preparedTokenBatches = [
            [11, 12, 13, 14],
            [11, 12, 13, 14, 15],
        ]

        let backend = MLXBackend(enableKVCacheReuse: true)
        backend._inject(mock)

        _ = try await collectAllEvents(from: try backend.generate(
            prompt: "first",
            systemPrompt: nil,
            config: GenerationConfig()
        ))

        let secondStream = try backend.generate(
            prompt: "second",
            systemPrompt: nil,
            config: GenerationConfig()
        )
        let secondEvents = try await collectAllEvents(from: secondStream)
        let reuseCounts = secondEvents.compactMap { event -> Int? in
            if case .kvCacheReuse(let count) = event { return count }
            return nil
        }

        XCTAssertEqual(reuseCounts, [4],
            "Second turn should emit kvCacheReuse for the shared 4-token prompt prefix")
        XCTAssertEqual(mock.lastInitialCacheOffsets, [4],
            "Generation should resume from the restored prefix length, not from a cold cache")

        // Sabotage check: disabling enableKVCacheReuse or clearing _promptCacheSnapshot
        // before the second turn makes reuseCounts empty and the cache offset 0.
    }

    func test_generate_withReuseDisabled_doesNotPersistPromptSnapshot() async throws {
        let mock = MockMLXModelContainer()
        mock.tokensToYield = ["ok"]
        mock.preparedTokenBatches = [[16, 17, 18, 19]]

        let backend = MLXBackend(enableKVCacheReuse: false)
        backend._inject(mock)

        XCTAssertFalse(backend._hasPromptCacheSnapshotForTesting())
        _ = try await collectAllEvents(from: try backend.generate(
            prompt: "first",
            systemPrompt: nil,
            config: GenerationConfig()
        ))

        XCTAssertFalse(
            backend._hasPromptCacheSnapshotForTesting(),
            "Default-off reuse must not retain prompt-cache state between turns"
        )

        // Sabotage check: capturing snapshots unconditionally leaves a retained
        // prompt snapshot here and fails the final assertion.
    }

    func test_generate_reusesOnlySharedPrefixAfterPromptDivergence() async throws {
        let mock = MockMLXModelContainer()
        mock.tokensToYield = ["ok"]
        mock.preparedTokenBatches = [
            [21, 22, 23, 24],
            [21, 22, 99, 100],
        ]

        let backend = MLXBackend(enableKVCacheReuse: true)
        backend._inject(mock)

        _ = try await collectAllEvents(from: try backend.generate(
            prompt: "first",
            systemPrompt: nil,
            config: GenerationConfig()
        ))

        let secondEvents = try await collectAllEvents(from: try backend.generate(
            prompt: "second",
            systemPrompt: nil,
            config: GenerationConfig()
        ))
        let reuseCounts = secondEvents.compactMap { event -> Int? in
            if case .kvCacheReuse(let count) = event { return count }
            return nil
        }

        XCTAssertEqual(reuseCounts, [2],
            "Only the shared head of the prompt should be reused after divergence")
        XCTAssertEqual(mock.lastInitialCacheOffsets, [2],
            "Restored cache should be trimmed to the shared-prefix length before generation")

        // Sabotage check: changing the second prepared token batch to start with
        // [21, 99, ...] drops reuse to 1 and fails the assertions.
    }

    func test_generate_unsupportedCacheShapeBypassesReuseAndClearsSnapshot() async throws {
        let mock = MockMLXModelContainer()
        mock.tokensToYield = ["ok"]
        mock.preparedTokenBatches = [
            [31, 32, 33, 34],
            [31, 32, 33, 34, 35],
            [31, 32, 33, 34, 35, 36],
        ]

        let backend = MLXBackend(enableKVCacheReuse: true)
        backend._inject(mock)

        _ = try await collectAllEvents(from: try backend.generate(
            prompt: "first",
            systemPrompt: nil,
            config: GenerationConfig()
        ))

        mock.cacheFactory = { [RotatingKVCache(maxSize: 32)] }
        let secondEvents = try await collectAllEvents(from: try backend.generate(
            prompt: "second",
            systemPrompt: nil,
            config: GenerationConfig()
        ))
        let secondReuseCounts = secondEvents.compactMap { event -> Int? in
            if case .kvCacheReuse(let count) = event { return count }
            return nil
        }

        XCTAssertTrue(secondReuseCounts.isEmpty,
            "Unsupported cache families must bypass prompt-cache restore")
        XCTAssertEqual(mock.lastInitialCacheOffsets, [0],
            "Unsupported cache families should start from a cold cache")

        mock.cacheFactory = { [KVCacheSimple()] }
        let thirdEvents = try await collectAllEvents(from: try backend.generate(
            prompt: "third",
            systemPrompt: nil,
            config: GenerationConfig()
        ))
        let thirdReuseCounts = thirdEvents.compactMap { event -> Int? in
            if case .kvCacheReuse(let count) = event { return count }
            return nil
        }

        XCTAssertTrue(thirdReuseCounts.isEmpty,
            "A turn that bypasses reuse must clear the prior snapshot rather than keeping stale state")
        XCTAssertEqual(mock.lastInitialCacheOffsets, [0],
            "After an unsupported turn the next request should still begin cold")

        // Sabotage check: if unsupported caches leave the old snapshot intact, the
        // third turn reuses 4+ tokens and both assertions fail.
    }

    func test_resetConversation_invalidatesPromptCache() async throws {
        let mock = MockMLXModelContainer()
        mock.tokensToYield = ["ok"]
        mock.preparedTokenBatches = [
            [41, 42, 43, 44],
            [41, 42, 43, 44, 45],
        ]

        let backend = MLXBackend(enableKVCacheReuse: true)
        backend._inject(mock)

        _ = try await collectAllEvents(from: try backend.generate(
            prompt: "first",
            systemPrompt: nil,
            config: GenerationConfig()
        ))

        backend.resetConversation()

        let secondEvents = try await collectAllEvents(from: try backend.generate(
            prompt: "second",
            systemPrompt: nil,
            config: GenerationConfig()
        ))
        let reuseCounts = secondEvents.compactMap { event -> Int? in
            if case .kvCacheReuse(let count) = event { return count }
            return nil
        }

        XCTAssertTrue(reuseCounts.isEmpty,
            "resetConversation must invalidate any cached prompt prefix")
        XCTAssertEqual(mock.lastInitialCacheOffsets, [0],
            "After resetConversation the next turn should start from a cold cache")

        // Sabotage check: removing the resetConversation() call restores a 4-token hit.
    }

    func test_generate_offsetOnlyMockRestoresPromptLengthAfterPriorCompletionTail() async throws {
        let mock = MockMLXModelContainer()
        mock.tokensToYield = ["ok"]
        mock.simulatedCacheCompletionTokenCount = 3
        mock.preparedTokenBatches = [
            [51, 52, 53, 54],
            [51, 52, 53, 54, 55],
        ]

        let backend = MLXBackend(enableKVCacheReuse: true)
        backend._inject(mock)

        _ = try await collectAllEvents(from: try backend.generate(
            prompt: "first",
            systemPrompt: nil,
            config: GenerationConfig()
        ))

        let secondEvents = try await collectAllEvents(from: try backend.generate(
            prompt: "second",
            systemPrompt: nil,
            config: GenerationConfig()
        ))
        let reuseCounts = secondEvents.compactMap { event -> Int? in
            if case .kvCacheReuse(let count) = event { return count }
            return nil
        }

        XCTAssertEqual(reuseCounts, [4],
            "Only prompt tokens, not the simulated completion tail, should be restorable on the next turn")
        XCTAssertEqual(mock.lastInitialCacheOffsets, [4],
            "Even on the offset-only mock path, restore should clamp back to the prompt length before reuse")

        // Sabotage check: if the restore path keeps the simulated 3-token completion
        // tail, lastInitialCacheOffsets becomes 7 and fails this assertion. The
        // real tensor-state trim/copy path is covered by the Xcode-only MLX
        // integration suite, not this CI-safe mock.
    }

    // MARK: - test_generate_cancellation

    func test_generate_cancellation() async throws {
        let mock = MockMLXModelContainer()
        // Use many tokens so the stream is still live when we cancel.
        mock.tokensToYield = Array(repeating: "x", count: 50)

        let backend = MLXBackend()
        backend._inject(mock)

        let stream = try backend.generate(
            prompt: "hi",
            systemPrompt: nil,
            config: GenerationConfig()
        )

        // Consume one token, then break to cancel the stream.
        var receivedCount = 0
        for try await event in stream.events {
            if case .token = event {
                receivedCount += 1
                if receivedCount == 1 { break }
            }
        }

        // The stream's onTermination fires task.cancel() which sets isGenerating = false
        // asynchronously in the @MainActor task. Poll with a tight deadline.
        let expectation = expectation(description: "isGenerating clears after cancel")
        Task {
            let deadline = ContinuousClock.now + .seconds(2)
            while backend.isGenerating, ContinuousClock.now < deadline {
                await Task.yield()
            }
            expectation.fulfill()
        }
        await fulfillment(of: [expectation], timeout: 3)

        XCTAssertFalse(backend.isGenerating,
            "isGenerating must be false after the stream is cancelled")
        XCTAssertEqual(receivedCount, 1,
            "Should have consumed exactly one token before cancelling")

        // Sabotage check: removing the break would drain the full stream, making
        // isGenerating false for the wrong reason (completion, not cancellation).
    }

    // MARK: - test_generate_generateThrows_propagatesError

    func test_generate_generateThrows_propagatesError() async throws {
        struct GenerateFailure: Error {}

        let mock = MockMLXModelContainer()
        mock.generateError = GenerateFailure()

        let backend = MLXBackend()
        backend._inject(mock)

        let stream = try backend.generate(
            prompt: "hi",
            systemPrompt: nil,
            config: GenerationConfig()
        )

        // Drain the stream — the thrown error surfaces via the events stream.
        var didThrow = false
        do {
            for try await _ in stream.events {}
        } catch {
            didThrow = true
            XCTAssertTrue(error is GenerateFailure, "Expected GenerateFailure, got \(error)")
        }

        XCTAssertTrue(didThrow, "Stream must propagate the error thrown by generate")

        // Verify the GenerationStream reached the .failed phase.
        let phase = await MainActor.run { stream.phase }
        if case .failed = phase {
            // Expected.
        } else {
            XCTFail("Expected stream phase .failed, got \(phase)")
        }

        // Sabotage check: removing mock.generateError leaves generateError as nil,
        // so the stream succeeds and didThrow stays false, failing the assertion.
    }

    // MARK: - test_sendableLMInput_wrapsAndUnwraps

    // MARK: - Stop reason surfacing (#515)

    /// Helper: drains every event (token, thinking, usage, tool-call) into an ordered array.
    private func collectAllEvents(
        from stream: GenerationStream
    ) async throws -> [GenerationEvent] {
        var events: [GenerationEvent] = []
        for try await event in stream.events {
            events.append(event)
        }
        return events
    }

    /// Documents today's conflation: both natural end-of-stream AND `maxOutputTokens`
    /// cutoff collapse to `GenerationStream.Phase.done` with no dedicated
    /// "stop_reason" signal. Downstream UI that wants to render "response truncated"
    /// today has nothing to key off.
    ///
    /// The fixture drives both paths in a single test so the shared baseline assertion
    /// ("all terminations look identical today") is written once. When structured
    /// stop reasons land (GenerationEvent.stopReason(.endOfStream / .maxTokens)), flip
    /// the per-branch assertions to the target shape — see FIXMEs inline.
    func test_stopReason_currentlyCollapsesToDone_forBothNaturalAndMaxTokens() async throws {
        // MARK: Natural EOS — mock finishes its finite token list
        do {
            let mock = MockMLXModelContainer()
            mock.tokensToYield = ["Hi", "."]

            let backend = MLXBackend()
            backend._inject(mock)

            let stream = try backend.generate(
                prompt: "hi",
                systemPrompt: nil,
                config: GenerationConfig()
            )

            let events = try await collectAllEvents(from: stream)
            let tokens = events.compactMap { ev -> String? in
                if case .token(let t) = ev { return t } else { return nil }
            }
            XCTAssertEqual(tokens, ["Hi", "."],
                "Natural EOS must surface every injected token before termination")

            // GenerationEvent has no stopReason case — Claude/MLX stop reasons are not surfaced. Documents the current gap.
            let phase = await MainActor.run { stream.phase }
            if case .done = phase {
                // Expected today.
            } else {
                XCTFail("Expected .done phase on natural EOS, got \(phase)")
            }
        }

        // MARK: Hit maxOutputTokens — same .done phase, no structured distinction
        do {
            let mock = MockMLXModelContainer()
            mock.tokensToYield = Array(repeating: "x", count: 50)

            let backend = MLXBackend()
            backend._inject(mock)

            var config = GenerationConfig()
            config.maxOutputTokens = 3

            let stream = try backend.generate(
                prompt: "hi",
                systemPrompt: nil,
                config: config
            )

            let events = try await collectAllEvents(from: stream)
            let tokens = events.compactMap { ev -> String? in
                if case .token(let t) = ev { return t } else { return nil }
            }
            XCTAssertEqual(tokens.count, 3,
                "maxOutputTokens must cap visible tokens — this is the truncation the UI cares about")

            // GenerationEvent has no stopReason case — Claude/MLX stop reasons are not surfaced. Documents the current gap.
            let phase = await MainActor.run { stream.phase }
            if case .done = phase {
                // Expected today — the conflation this fixture exists to flag.
            } else {
                XCTFail("Expected .done phase on maxOutputTokens cutoff, got \(phase)")
            }
        }

        // Sabotage check: changing the first mock.tokensToYield to ["Hi"] makes the
        // natural-EOS branch's XCTAssertEqual(tokens, ["Hi", "."]) fail. Raising the
        // second branch's maxOutputTokens to 100 breaks the tokens.count == 3 check.
    }

    // MARK: - Tool call extraction (#517 — shipped)

    /// MLX now extracts tool calls from the stream (#517). Extraction is
    /// whole-call by design: `MLXBackend` wires `ToolCallTransform` with
    /// `MLXToolMarkers.markers()` into the driver pipeline and emits a single
    /// `.toolCall` event per `<tool_call>…</tool_call>` block — there is no
    /// per-token delta event. The transform stage only activates when the call
    /// passes tools AND the model speaks a known dialect, so this drives the
    /// full `generate` → `MLXGenerationDriver` path (not just the transform unit,
    /// which `MLXToolMarkersParityTests` covers separately).
    ///
    /// Runs on the injected mock container — no Metal, consistent with the other
    /// tests in this suite.
    func test_toolCall_extractedFromStream_wholeCall() async throws {
        let mock = MockMLXModelContainer()
        mock.tokensToYield = [
            "<tool_call>",
            "{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}",
            "</tool_call>",
        ]

        let backend = MLXBackend()
        // A known dialect is required for the tool stage to activate.
        backend._inject(mock, dialect: .qwen25)

        var config = GenerationConfig()
        config.tools = [
            ToolDefinition(name: "get_weather", description: "weather", parameters: .object([:]))
        ]

        let stream = try backend.generate(
            prompt: "weather?",
            systemPrompt: nil,
            config: config
        )

        let events = try await collectAllEvents(from: stream)

        let visibleTokens = events.compactMap { ev -> String? in
            if case .token(let t) = ev { return t } else { return nil }
        }
        XCTAssertFalse(visibleTokens.joined().contains("<tool_call>"),
            "tool_call tags must not leak into visible .token output")

        let extractedNames = events.compactMap { ev -> String? in
            if case .toolCall(let call) = ev { return call.toolName } else { return nil }
        }
        XCTAssertEqual(extractedNames, ["get_weather"],
            "A single whole-call .toolCall event naming get_weather must surface")

        // Sabotage check: removing the tool stage in MLXGenerationDriver makes the
        // raw `<tool_call>` bytes leak back into .token events, failing both asserts.
    }

    // MARK: - Structural tools threading (Phase 0 / umbrella #2005, F3)

    /// Mistral (a tools-aware-template dialect) must thread the structural
    /// `tools` array into `prepare(messages:tools:)` so the model's own chat
    /// template renders its native `[AVAILABLE_TOOLS]` block.
    func test_generate_mistral_threadsStructuralTools() async throws {
        let mock = MockMLXModelContainer()
        mock.tokensToYield = ["ok"]
        let backend = MLXBackend()
        backend._inject(mock, dialect: .mistral)

        var config = GenerationConfig()
        config.tools = [
            ToolDefinition(name: "get_weather", description: "weather", parameters: .object([:]))
        ]

        let stream = try backend.generate(prompt: "weather?", systemPrompt: nil, config: config)
        _ = try await collectAllEvents(from: stream)

        let threaded = try XCTUnwrap(mock.lastTools, "prepare(messages:tools:) must be the path taken")
        let specs = try XCTUnwrap(threaded, "Mistral must thread a non-nil structural tools array")
        XCTAssertEqual(specs.count, 1)
        let function = try XCTUnwrap(specs.first?["function"] as? [String: any Sendable])
        XCTAssertEqual(function["name"] as? String, "get_weather")

        // Sabotage check: dropping `toolSpecs` from the driver's
        // prepareInputAndCache call makes `lastTools` resolve to nil here.
    }

    /// Llama keeps the prose path (issue #59) — no structural tools threaded, so
    /// the existing ✅ render is unchanged at runtime.
    func test_generate_llama_threadsNoStructuralTools() async throws {
        let mock = MockMLXModelContainer()
        mock.tokensToYield = ["ok"]
        let backend = MLXBackend()
        backend._inject(mock, dialect: .llama)

        var config = GenerationConfig()
        config.tools = [
            ToolDefinition(name: "get_weather", description: "weather", parameters: .object([:]))
        ]

        let stream = try backend.generate(prompt: "weather?", systemPrompt: nil, config: config)
        _ = try await collectAllEvents(from: stream)

        // The tools-aware overload IS hit, but with a nil tools array (prose path).
        let threaded = try XCTUnwrap(mock.lastTools, "prepare(messages:tools:) is always the path taken")
        XCTAssertNil(threaded, "Llama must NOT thread structural tools (prose path preserved)")
    }

    // MARK: - Detokenization fragmentation + nil-chunk (#27)

    /// A `<tool_call>` marker that real BPE/SentencePiece detokenization can
    /// split across chunk boundaries (`<tool` + `_call>`) must still be
    /// reassembled by the driver's `ToolCallTransform` stage and surface as a
    /// single whole-call `.toolCall` event — none of the marker bytes may leak
    /// into visible `.token` output.
    ///
    /// This differs from `test_toolCall_extractedFromStream_wholeCall`, which
    /// feeds *pre-split, well-formed* `<tool_call>` strings: here the opening and
    /// closing markers are each fragmented mid-token, so the test exercises the
    /// transform's cross-chunk buffering rather than a clean tag boundary.
    func test_toolCall_markerFragmentedAcrossChunks_reassembled() async throws {
        let mock = MockMLXModelContainer()
        // Fragment BOTH markers across chunk boundaries, the way real
        // detokenization can. The payload is also split.
        mock.generationsToYield = [
            .chunk("<tool"),
            .chunk("_call>"),
            .chunk("{\"name\":\"get_weather\","),
            .chunk("\"arguments\":{\"city\":\"Paris\"}}"),
            .chunk("</tool"),
            .chunk("_call>"),
        ]

        let backend = MLXBackend()
        backend._inject(mock, dialect: .qwen25)

        var config = GenerationConfig()
        config.tools = [
            ToolDefinition(name: "get_weather", description: "weather", parameters: .object([:]))
        ]

        let stream = try backend.generate(
            prompt: "weather?",
            systemPrompt: nil,
            config: config
        )

        let events = try await collectAllEvents(from: stream)

        let visibleTokens = events.compactMap { ev -> String? in
            if case .token(let t) = ev { return t } else { return nil }
        }
        XCTAssertFalse(visibleTokens.joined().contains("<tool"),
            "Fragmented tool-call marker bytes must not leak into visible tokens")
        XCTAssertFalse(visibleTokens.joined().contains("_call>"),
            "The split marker tail must not leak into visible tokens either")

        let extractedNames = events.compactMap { ev -> String? in
            if case .toolCall(let call) = ev { return call.toolName } else { return nil }
        }
        let extractedArgs = events.compactMap { ev -> String? in
            if case .toolCall(let call) = ev { return call.arguments } else { return nil }
        }
        XCTAssertEqual(extractedNames, ["get_weather"],
            "A marker fragmented across chunks must still reassemble into one whole-call event")
        XCTAssertTrue(extractedArgs.first?.contains("Paris") == true,
            "Reassembled tool call must carry the full arguments payload")
    }

    /// Realistic streams interleave `.info(...)` completion metadata with text
    /// chunks; `Generation.info`'s `chunk` accessor is `nil`, so the driver's
    /// `guard let text = generation.chunk else { continue }` nil path (run loop
    /// ~line 342) must skip it without dropping or duplicating surrounding text.
    func test_run_nilChunkGeneration_isSkippedWithoutDisturbingText() async throws {
        let mock = MockMLXModelContainer()
        // `.info` has a nil `chunk` — it must be silently skipped, and the
        // text on either side must still surface intact and in order.
        let info = Generation.info(
            GenerateCompletionInfo(
                promptTokenCount: 4,
                generationTokenCount: 2,
                promptTime: 0.01,
                generationTime: 0.02
            )
        )
        mock.generationsToYield = [
            .chunk("Hello"),
            info,
            .chunk(" world"),
        ]

        let backend = MLXBackend()
        backend._inject(mock)

        let stream = try backend.generate(
            prompt: "hi",
            systemPrompt: nil,
            config: GenerationConfig()
        )

        let tokens = try await collectTokens(from: stream)

        XCTAssertEqual(tokens, ["Hello", " world"],
            "The nil-chunk .info generation must be skipped, leaving surrounding text intact and ordered")
    }

    // MARK: - Chat-template detection (#516)

    /// Covers the observable slice of MLX chat-template handling: the `messages`
    /// dictionary array the backend hands to the container, unchanged.
    ///
    /// The full detection matrix (missing `tokenizer_config.json`, template without an
    /// `<|assistant|>` marker) is now driven via `simulatedTokenizerApplyFailure`
    /// on the mock container — see the two sibling tests below.
    func test_chatTemplate_messagesPassThroughUnchanged() async throws {
        let mock = MockMLXModelContainer()
        mock.tokensToYield = ["ok"]

        let backend = MLXBackend()
        backend._inject(mock)

        let systemPrompt = "You are helpful."
        let userPrompt = "hello"

        let stream = try backend.generate(
            prompt: userPrompt,
            systemPrompt: systemPrompt,
            config: GenerationConfig()
        )

        // Drain so the generate task completes and `lastMessages` is populated.
        _ = try await collectAllEvents(from: stream)

        // The backend hands the MLX container the [role:system, role:user] pair so the
        // real tokenizer's chat template can apply. A template missing `<|assistant|>`
        // will fail at tokenize time, not here — that branch is covered by the sibling
        // skip test below plus #551.
        let sent = try XCTUnwrap(mock.lastMessages,
            "MLXBackend.generate must pass a messages array to the container")
        XCTAssertEqual(sent.count, 2,
            "Expected [system, user] when both prompts are provided")
        XCTAssertEqual(sent.first?["role"], "system")
        XCTAssertEqual(sent.first?["content"], systemPrompt)
        XCTAssertEqual(sent.last?["role"], "user")
        XCTAssertEqual(sent.last?["content"], userPrompt)

        // Sabotage check: reordering the msgs.append calls in MLXBackend.generate would
        // flip system and user positions, failing the first/last XCTAssertEquals.
    }

    // MARK: - test_chatTemplate_missingTemplate_propagatesApplyErrorUnchanged

    /// When the loaded tokenizer has no `chat_template` (e.g. `tokenizer_config.json`
    /// missing the field, or the file itself absent in the model snapshot), the
    /// MLX container raises an error from `apply_chat_template` *during generation*.
    ///
    /// `MLXBackend` does NOT wrap that mid-generation error in
    /// `InferenceError.modelLoadFailed` — `modelLoadFailed` is reserved for the
    /// `loadModel(...)` path. The original error surfaces UNCHANGED through the
    /// GenerationStream's `try await events`, and the stream phase becomes
    /// `.failed`. This test pins that pass-through contract (named for what it
    /// actually asserts, not for `modelLoadFailed`, which is the opposite of the
    /// real behaviour) so a future structured-error change has a concrete fixture.
    func test_chatTemplate_missingTemplate_propagatesApplyErrorUnchanged() async throws {
        struct MissingChatTemplateError: Error, Equatable {
            let detail = "tokenizer_config.json missing chat_template field"
        }

        let mock = MockMLXModelContainer()
        mock.simulatedTokenizerApplyFailure = MissingChatTemplateError()

        let backend = MLXBackend()
        backend._inject(mock)

        let stream = try backend.generate(
            prompt: "hi",
            systemPrompt: "You are helpful.",
            config: GenerationConfig()
        )

        var caught: Error?
        do {
            for try await _ in stream.events {}
        } catch {
            caught = error
        }

        let unwrapped = try XCTUnwrap(caught,
            "Stream must surface the tokenizer-apply error rather than completing silently")
        // Today MLXBackend does not wrap mid-generation errors in `modelLoadFailed`;
        // the underlying error propagates as-is. If a future change wraps these
        // errors structurally, flip this assertion to match.
        XCTAssertTrue(unwrapped is MissingChatTemplateError,
            "Expected MissingChatTemplateError to propagate unchanged, got \(type(of: unwrapped))")

        // Phase must be .failed so observers can react to the error.
        let phase = await MainActor.run { stream.phase }
        if case .failed = phase {
            // Expected.
        } else {
            XCTFail("Expected stream phase .failed, got \(phase)")
        }

        // Sabotage check: clearing simulatedTokenizerApplyFailure makes the stream
        // succeed and `caught` stays nil, failing the XCTUnwrap.
    }

    // MARK: - test_chatTemplate_noAssistantMarker_propagatesApplyErrorUnchanged

    /// When the chat template is present but malformed (e.g. it never emits an
    /// `<|assistant|>` marker so the tokenizer has nowhere to start the model's
    /// turn), `apply_chat_template` raises a different error class. Same surfacing
    /// contract as the missing-template case: the error propagates UNCHANGED
    /// through the GenerationStream (the backend does not re-wrap it into any
    /// structured InferenceError), and the container's `generate` is reached
    /// before the apply fails. Renamed from `..._throwsStructuredError` because
    /// no wrapping/structuring happens — the assertion is plain pass-through.
    func test_chatTemplate_noAssistantMarker_propagatesApplyErrorUnchanged() async throws {
        struct NoAssistantMarkerError: Error, Equatable {
            let detail = "chat template missing <|assistant|> marker"
        }

        let mock = MockMLXModelContainer()
        mock.simulatedTokenizerApplyFailure = NoAssistantMarkerError()

        let backend = MLXBackend()
        backend._inject(mock)

        let stream = try backend.generate(
            prompt: "hi",
            systemPrompt: nil,
            config: GenerationConfig()
        )

        var caught: Error?
        do {
            for try await _ in stream.events {}
        } catch {
            caught = error
        }

        let unwrapped = try XCTUnwrap(caught,
            "Stream must surface the malformed-template error")
        XCTAssertTrue(unwrapped is NoAssistantMarkerError,
            "Expected NoAssistantMarkerError to propagate unchanged, got \(type(of: unwrapped))")

        // The mock must have observed the call before throwing — confirms the
        // backend reached the container's generate path before the apply failed.
        XCTAssertEqual(mock.generateCallCount, 1)
        XCTAssertEqual(mock.lastMessages?.last?["role"], "user")

        // Sabotage check: setting mock.simulatedTokenizerApplyFailure = nil lets
        // the stream complete with the default tokens, leaving `caught` nil and
        // failing the XCTUnwrap.
    }

    // MARK: - test_generate_usesConversationHistory

    func test_generate_usesConversationHistory() async throws {
        let mock = MockMLXModelContainer()
        mock.tokensToYield = ["ok"]

        let backend = MLXBackend()
        backend._inject(mock)

        // Simulate two prior turns before the current user message.
        backend.setConversationHistory([
            ("user", "What is the capital of France?"),
            ("assistant", "Paris."),
            ("user", "And Germany?"),
        ])

        let stream = try backend.generate(
            prompt: "And Germany?",
            systemPrompt: nil,
            config: GenerationConfig()
        )
        _ = try await collectTokens(from: stream)

        let sent = try XCTUnwrap(mock.lastMessages)
        XCTAssertEqual(sent.count, 3,
            "All three history turns must be forwarded to the container")
        XCTAssertEqual(sent[0]["role"], "user")
        XCTAssertEqual(sent[0]["content"], "What is the capital of France?")
        XCTAssertEqual(sent[1]["role"], "assistant")
        XCTAssertEqual(sent[1]["content"], "Paris.")
        XCTAssertEqual(sent[2]["role"], "user")
        XCTAssertEqual(sent[2]["content"], "And Germany?")

        // Sabotage check: removing the setConversationHistory call causes the
        // backend to fall back to the bare prompt path, producing a 1-element
        // messages array and failing the count assertion.
    }

    // MARK: - WindowServer yield cadence (#747)

    /// Asserts the cooperative yield inserted to prevent WindowServer GPU-queue
    /// starvation fires every `yieldEveryNTokens` MLX-emitted chunks. We replace
    /// the production `Task.yield()` with a counting hook
    /// so the test is deterministic and free of timing assumptions. Setting
    /// `yieldEveryNTokens = 0` must disable the yield entirely.
    ///
    /// `#if MLX` plus the existing mock-container path keep this test running
    /// in CI without Metal — the production code path is identical, the test
    /// just substitutes the sleep with a counter.
    func test_yieldEveryNTokens_firesAtConfiguredCadence() async throws {
        let counter = YieldCounter()
        MLXBackend._yieldHookForTesting = { counter.increment() }
        defer { MLXBackend._yieldHookForTesting = nil }

        // Configured cadence: every 4 chunks. With 12 chunks emitted we expect
        // exactly 3 yields (at 4, 8, 12). maxOutputTokens is set high enough
        // that the limit doesn't truncate before the full chunk count.
        let mock = MockMLXModelContainer()
        mock.tokensToYield = Array(repeating: "x", count: 12)

        let backend = MLXBackend()
        backend._inject(mock)

        var config = GenerationConfig()
        config.yieldEveryNTokens = 4
        config.maxOutputTokens = 100

        let stream = try backend.generate(
            prompt: "hi",
            systemPrompt: nil,
            config: config
        )
        _ = try await collectTokens(from: stream)

        XCTAssertEqual(counter.count, 3,
            "Yield must fire exactly every yieldEveryNTokens chunks (12 / 4 = 3)")

        // Now verify yieldEveryNTokens = 0 disables the yield entirely.
        let counter2 = YieldCounter()
        MLXBackend._yieldHookForTesting = { counter2.increment() }

        let mock2 = MockMLXModelContainer()
        mock2.tokensToYield = Array(repeating: "x", count: 12)

        let backend2 = MLXBackend()
        backend2._inject(mock2)

        var disabledConfig = GenerationConfig()
        disabledConfig.yieldEveryNTokens = 0
        disabledConfig.maxOutputTokens = 100

        let stream2 = try backend2.generate(
            prompt: "hi",
            systemPrompt: nil,
            config: disabledConfig
        )
        _ = try await collectTokens(from: stream2)

        XCTAssertEqual(counter2.count, 0,
            "yieldEveryNTokens = 0 must skip the cooperative yield entirely")

        // Sabotage check: changing the modulo condition to `% (yieldEvery + 1)`
        // in MLXBackend would yield 2 times for 12 chunks at cadence 4 (at 5, 10),
        // failing the count == 3 assertion.
    }

    func test_yieldEveryNTokens_defaultCadenceIsEight() async throws {
        let counter = YieldCounter()
        MLXBackend._yieldHookForTesting = { counter.increment() }
        defer { MLXBackend._yieldHookForTesting = nil }

        let mock = MockMLXModelContainer()
        mock.tokensToYield = Array(repeating: "x", count: 16)

        let backend = MLXBackend()
        backend._inject(mock)

        var config = GenerationConfig()
        config.maxOutputTokens = 100

        let stream = try backend.generate(
            prompt: "hi",
            systemPrompt: nil,
            config: config
        )
        _ = try await collectTokens(from: stream)

        XCTAssertEqual(counter.count, 2,
            "Default MLX yield cadence should fire every 8 emitted chunks")

        // Sabotage check: changing GenerationConfig's default yieldEveryNTokens
        // away from 8 changes the number of hook invocations for 16 chunks.
    }

    /// Cancellation during the cooperative yield must not crash, must not leak
    /// `CancellationError` to the consumer (the production sleep is `try?`'d),
    /// and the next loop iteration's `Task.isCancelled` check must terminate
    /// the stream cleanly.
    ///
    /// The hook substitutes for `Task.yield()`, so to model "cancellation while
    /// yielding" we cancel the surrounding generation Task from inside the hook
    /// itself — this exercises the same control-flow shape as a real cancel
    /// arriving at a cooperative yield point.
    func test_yieldEveryNTokens_cancellationDuringYield_terminatesCleanly() async throws {
        let mock = MockMLXModelContainer()
        mock.tokensToYield = Array(repeating: "x", count: 32)

        let backend = MLXBackend()
        backend._inject(mock)

        // Cancel mid-generation from inside the yield hook. By the time the
        // first yield fires (at chunk 4) we cancel via stopGeneration(), which
        // cancels the underlying generation Task. The next iteration of the
        // mlxStream `for await` should observe `Task.isCancelled` and break.
        MLXBackend._yieldHookForTesting = { [weak backend] in
            backend?.stopGeneration()
        }
        defer { MLXBackend._yieldHookForTesting = nil }

        var config = GenerationConfig()
        config.yieldEveryNTokens = 4
        config.maxOutputTokens = 100

        let stream = try backend.generate(
            prompt: "hi",
            systemPrompt: nil,
            config: config
        )

        // Drain the stream — must complete without throwing, even though the
        // surrounding task was cancelled mid-yield. The yield hook does not throw,
        // matching production `Task.yield()`, so the for-await observes
        // cancellation at the top of the next iteration.
        let tokens = try await collectTokens(from: stream)

        // Expect at most ~yieldEvery tokens to have been emitted before the
        // cancel propagated. Strictly: no more than one full cadence past the
        // cancel point. The point of the assertion is simply that we exited
        // cleanly and didn't drain all 32 tokens.
        XCTAssertLessThan(tokens.count, 32,
            "Cancellation during yield must terminate the stream early")
    }

    func test_sendableLMInput_wrapsAndUnwraps() throws {
        // This test verifies `SendableLMInput` at the type level: the wrapper must
        // satisfy Swift's `Sendable` requirement so the compiler allows cross-actor
        // transfer. The MLX runtime (Metal) is NOT accessed here.
        //
        // Full round-trip testing (with a real LMInput) requires Metal and is
        // exercised in ManifoldE2ETests on Apple Silicon hardware.

        // Verify that SendableLMInput is Sendable at compile time.
        // If the @unchecked Sendable annotation were removed, the compiler would
        // reject the line below with a Sendable violation.
        func assertSendable<T: Sendable>(_: T.Type) {}
        assertSendable(SendableLMInput.self)

        // Verify the wrapper's public surface at compile time without constructing
        // an LMInput: both the initializer and `.value` must be accessible.
        let initializer = SendableLMInput.init
        let valueKeyPath = \SendableLMInput.value
        _ = (initializer, valueKeyPath)
    }

    // MARK: - test_generate_emitsUsageEvent_fromStreamInfo

    /// Verifies that a `.usage(TokenUsage)` event is emitted when the stream
    /// yields a `.info(GenerateCompletionInfo)` element (the real MLX path).
    func test_generate_emitsUsageEvent_fromStreamInfo() async throws {
        let mock = MockMLXModelContainer()
        let completionInfo = GenerateCompletionInfo(
            promptTokenCount: 7,
            generationTokenCount: 3,
            promptTime: 0.1,
            generationTime: 0.2
        )
        mock.generationsToYield = [
            .chunk("Hello"),
            .chunk(" world"),
            .info(completionInfo),
        ]

        let backend = MLXBackend()
        backend._inject(mock)

        let stream = try backend.generate(
            prompt: "hi",
            systemPrompt: nil,
            config: GenerationConfig()
        )

        var usageEvent: TokenUsage?
        for try await event in stream.events {
            if case .usage(let usage) = event {
                usageEvent = usage
            }
        }

        let usage = try XCTUnwrap(usageEvent, "Stream must emit a .usage event")
        XCTAssertEqual(usage.promptTokens, 7)
        XCTAssertEqual(usage.completionTokens, 3)
    }

    // MARK: - test_generate_emitsUsageEvent_fallback

    /// Verifies that a `.usage(TokenUsage)` event is still emitted when the
    /// stream does NOT yield a `.info` element (mock / legacy path).
    func test_generate_emitsUsageEvent_fallback() async throws {
        let mock = MockMLXModelContainer()
        mock.tokensToYield = ["A", "B", "C"]

        let backend = MLXBackend()
        backend._inject(mock)

        let stream = try backend.generate(
            prompt: "hi",
            systemPrompt: nil,
            config: GenerationConfig()
        )

        var usageEvent: TokenUsage?
        for try await event in stream.events {
            if case .usage(let usage) = event {
                usageEvent = usage
            }
        }

        let usage = try XCTUnwrap(usageEvent, "Stream must emit a .usage event even without .info")
        XCTAssertGreaterThan(usage.completionTokens, 0,
            "completionTokens must be positive when tokens were generated")
    }

    // MARK: - Native tool-call forwarding (#59 follow-up)

    /// `MLXToolMarkers.toolCall(fromNative:)` maps an `MLXLMCommon.ToolCall`
    /// (the structured call mlx-swift-lm emits for inline formats) into our
    /// `ToolCall`, re-serialising the `[String: JSONValue]` arguments to a JSON
    /// string.
    func test_toolCallFromNative_mapsNameAndArguments() throws {
        let native = MLXLMCommon.ToolCall(
            function: .init(name: "calc", arguments: ["op": "*", "a": 7823, "b": 41])
        )
        let mapped = MLXToolMarkers.toolCall(fromNative: native)

        XCTAssertEqual(mapped.toolName, "calc")
        XCTAssertTrue(mapped.id.hasPrefix("mlx-calc-"), "id should follow the mlx-<name>-<uuid> shape")
        let data = try XCTUnwrap(mapped.arguments.data(using: .utf8))
        let obj = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(obj["op"] as? String, "*")
        XCTAssertEqual(obj["a"] as? Int, 7823)
        XCTAssertEqual(obj["b"] as? Int, 41)
    }

    /// When mlx-swift-lm swallows an inline tool call and emits it as a
    /// structured `Generation.toolCall` (no `.chunk` text), the driver must
    /// forward it as a `GenerationEvent.toolCall`. Before the fix the driver
    /// only read `.chunk`/`.info`, so the parsed call was silently dropped —
    /// the cause of Llama-3.2's empty answers / 0 dispatched tools.
    func test_generate_forwardsNativeToolCallEvent() async throws {
        let mock = MockMLXModelContainer()
        mock.generationsToYield = [
            .toolCall(MLXLMCommon.ToolCall(
                function: .init(name: "calc", arguments: ["op": "*", "a": 7823, "b": 41])
            ))
        ]

        let backend = MLXBackend()
        backend._inject(mock, dialect: .llama)

        var config = GenerationConfig()
        config.tools = [ToolDefinition(name: "calc", description: "calculator", parameters: .object([:]))]

        let events = try await collectAllEvents(from: try backend.generate(
            prompt: "What is 7823 * 41?",
            systemPrompt: nil,
            config: config
        ))

        let toolCalls = events.compactMap { event -> ManifoldInference.ToolCall? in
            if case .toolCall(let call) = event { return call }
            return nil
        }
        XCTAssertEqual(toolCalls.count, 1,
            "a native MLX tool call must surface as exactly one .toolCall event")
        XCTAssertEqual(toolCalls.first?.toolName, "calc")

        // Sabotage check: removing the `generation.toolCall` forwarding in
        // MLXGenerationDriver leaves `toolCalls` empty (the previous behavior).
    }
}
