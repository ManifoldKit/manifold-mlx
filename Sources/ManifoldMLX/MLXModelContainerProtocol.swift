import MLXLMCommon

// @_spi(Testing): published only for backend test targets (companion-package
// split, #1749) — keeps the underlying MLX types off the supported public API.
@_spi(Testing) public struct MLXPreparedInput: @unchecked Sendable {
    private let value: LMInput?
    private let promptTokenIdsOverride: [Int]?

    public init(_ value: LMInput) {
        self.value = value
        self.promptTokenIdsOverride = nil
    }

    public init(promptTokenIds: [Int]) {
        self.value = nil
        self.promptTokenIdsOverride = promptTokenIds
    }

    public var lmInput: LMInput {
        guard let value else {
            preconditionFailure("Test-only MLXPreparedInput has no LMInput payload")
        }
        return value
    }

    public var promptTokenIds: [Int] {
        promptTokenIdsOverride ?? lmInput.text.tokens.asArray(Int.self)
    }

    public func suffix(from reusedPromptTokenCount: Int) -> MLXPreparedInput {
        guard reusedPromptTokenCount > 0 else { return self }
        if let value {
            let remainingText = value.text[text: reusedPromptTokenCount...]
            return MLXPreparedInput(
                LMInput(text: remainingText, image: value.image, video: value.video)
            )
        }
        return MLXPreparedInput(promptTokenIds: Array(promptTokenIds.dropFirst(reusedPromptTokenCount)))
    }
}

// @_spi(Testing): see MLXPreparedInput.
@_spi(Testing) public struct MLXPromptCache: @unchecked Sendable {
    public let value: [any KVCache]

    public init(_ value: [any KVCache]) {
        self.value = value
    }
}

// @_spi(Testing): see MLXPreparedInput.
@_spi(Testing) public struct SendableChatMessages: @unchecked Sendable {
    public let value: [Chat.Message]

    public init(_ value: [Chat.Message]) {
        self.value = value
    }
}

/// Abstraction over `ModelContainer` so `MLXBackend` can be tested without real hardware.
///
/// `LMInput` and `[KVCache]` are wrapped so `MLXBackend` can own prompt preparation,
/// cache creation, prefix reuse, and token streaming while the concrete
/// `ModelContainer` conformance keeps the underlying MLX types off the public API.
// @_spi(Testing): published so backend test targets can stub the container
// (MockMLXModelContainer) without @testable access — companion split, #1749.
@_spi(Testing) public protocol MLXModelContainerProtocol: Sendable {
    func prepare(messages: [[String: String]]) async throws -> MLXPreparedInput

    /// Prepares text messages, threading structural `tools` into the tokenizer's
    /// `applyChatTemplate(messages:tools:)` so a tools-aware chat template (e.g.
    /// Mistral) renders its native tool block (Phase 0 / umbrella #2005, F3).
    ///
    /// `tools` is the mlx-swift-lm `ToolSpec` shape (`[[String: any Sendable]]`).
    /// A `nil`/empty `tools` is identical to `prepare(messages:)`. The default
    /// implementation forwards to `prepare(messages:)` ignoring tools so mock
    /// containers conform for free.
    func prepare(
        messages: [[String: String]],
        tools: [[String: any Sendable]]?
    ) async throws -> MLXPreparedInput

    func prepare(chat: SendableChatMessages) async throws -> MLXPreparedInput
    func makeCache(parameters: GenerateParameters) async throws -> MLXPromptCache
    func generate(
        input: MLXPreparedInput,
        cache: MLXPromptCache?,
        parameters: GenerateParameters
    ) async throws -> AsyncStream<Generation>

    /// Grammar-constrained generation (#96, option B).
    ///
    /// When `grammar` is non-nil the concrete container builds an
    /// ``MLXGrammarLogitProcessor`` (it needs the tokenizer, only reachable
    /// inside `perform`), composes it over the parameters' penalty processor,
    /// and drives generation through the direct `TokenIterator` init so the
    /// grammar mask is applied at every step. The default implementation ignores
    /// the grammar and forwards to the plain `generate(input:cache:parameters:)`,
    /// so mock containers conform for free.
    func generate(
        input: MLXPreparedInput,
        cache: MLXPromptCache?,
        parameters: GenerateParameters,
        grammar: GBNFGrammar?
    ) async throws -> AsyncStream<Generation>
}

@_spi(Testing) public extension MLXModelContainerProtocol {
    func generate(
        input: MLXPreparedInput,
        cache: MLXPromptCache?,
        parameters: GenerateParameters,
        grammar: GBNFGrammar?
    ) async throws -> AsyncStream<Generation> {
        try await generate(input: input, cache: cache, parameters: parameters)
    }

    /// Default: ignore structural `tools` and forward to `prepare(messages:)`.
    /// Mock containers get the tools-aware overload for free; the real
    /// `ModelContainer` conformance overrides it to thread tools through
    /// `applyChatTemplate`.
    func prepare(
        messages: [[String: String]],
        tools: [[String: any Sendable]]?
    ) async throws -> MLXPreparedInput {
        try await prepare(messages: messages)
    }
}

@_spi(Testing) extension ModelContainer: MLXModelContainerProtocol {
    public func prepare(messages: [[String: String]]) async throws -> MLXPreparedInput {
        let input = try await prepare(input: .init(messages: messages))
        return MLXPreparedInput(input)
    }

    public func prepare(
        messages: [[String: String]],
        tools: [[String: any Sendable]]?
    ) async throws -> MLXPreparedInput {
        // No structural tools → identical to the plain path so the Llama/Qwen
        // prose render is byte-unchanged (Phase 0 / #2005).
        guard let tools, !tools.isEmpty else {
            return try await prepare(messages: messages)
        }
        // `UserInput.tools` flows into `applyChatTemplate(messages:tools:)` via
        // `LLMUserInputProcessor.prepare(input:)`, so a tools-aware template
        // renders its native tool block (Mistral's `[AVAILABLE_TOOLS]`).
        let input = try await prepare(input: .init(messages: messages, tools: tools))
        return MLXPreparedInput(input)
    }

    public func prepare(chat: SendableChatMessages) async throws -> MLXPreparedInput {
        try await perform(nonSendable: chat.value) { context, chat in
            let input = try await context.processor.prepare(input: .init(chat: chat))
            return MLXPreparedInput(input)
        }
    }

    public func makeCache(parameters: GenerateParameters) async throws -> MLXPromptCache {
        await perform { context in
            MLXPromptCache(context.model.newCache(parameters: parameters))
        }
    }

    public func generate(
        input: MLXPreparedInput,
        cache: MLXPromptCache?,
        parameters: GenerateParameters
    ) async throws -> AsyncStream<Generation> {
        try await perform(nonSendable: (input.lmInput, cache?.value)) { context, values in
            let (input, cache) = values
            return try MLXLMCommon.generate(
                input: input,
                cache: cache,
                parameters: parameters,
                context: context,
                wiredMemoryTicket: nil
            )
        }
    }

    public func generate(
        input: MLXPreparedInput,
        cache: MLXPromptCache?,
        parameters: GenerateParameters,
        grammar: GBNFGrammar?
    ) async throws -> AsyncStream<Generation> {
        // No grammar → identical to the plain path (keeps KV-cache
        // quantization, speculative paths, etc.).
        guard let grammar else {
            return try await generate(input: input, cache: cache, parameters: parameters)
        }
        // Grammar path (#96, option B): build the grammar processor inside
        // `perform` where the tokenizer is reachable, compose it over the
        // parameters' penalty processor, and drive the direct `TokenIterator`.
        //
        // NOTE: the direct `TokenIterator(input:model:cache:processor:sampler:…)`
        // init does NOT apply KV-cache quantization (`kvBits` is dropped). This is
        // the accepted KV-quant tradeoff (issue #96) — grammar-active turns run
        // with an unquantized KV cache.
        return try await perform(nonSendable: (input.lmInput, cache?.value)) { context, values in
            let (input, cache) = values
            let processor = MLXGrammarLogitProcessor(
                grammar: grammar,
                tokenizer: context.tokenizer,
                base: parameters.processor()
            )
            let iterator = try TokenIterator(
                input: input,
                model: context.model,
                cache: cache,
                processor: processor,
                sampler: parameters.sampler(),
                prefillStepSize: parameters.prefillStepSize,
                maxTokens: parameters.maxTokens
            )
            let (stream, _) = MLXLMCommon.generateTask(
                promptTokenCount: input.text.tokens.size,
                modelConfiguration: context.configuration,
                tokenizer: context.tokenizer,
                iterator: iterator
            )
            return stream
        }
    }
}
