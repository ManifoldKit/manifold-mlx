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

  var promptTokenCount: Int {
    promptTokenIdsOverride?.count ?? lmInput.text.tokens.size
  }

  func promptTokenSuffix(maxCount: Int) -> [Int] {
    guard maxCount > 0 else { return [] }
    if let promptTokenIdsOverride {
      return Array(promptTokenIdsOverride.suffix(maxCount))
    }
    let tokens = lmInput.text.tokens.flattened()
    guard tokens.size > 0 else { return [] }
    return tokens[max(0, tokens.size - maxCount)...].asArray(Int.self)
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
  /// Prepares text messages with the structural tool list and the request's
  /// optional thinking override in the tokenizer's template context.
  func prepare(
    messages: [[String: String]],
    tools: [[String: any Sendable]]?,
    thinkingEnabled: Bool?
  ) async throws -> MLXPreparedInput

  func prepare(chat: SendableChatMessages, thinkingEnabled: Bool?) async throws -> MLXPreparedInput
  func decodePromptTail(_ input: MLXPreparedInput, maxTokens: Int) async -> String
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

@_spi(Testing) extension MLXModelContainerProtocol {
  public func generate(
    input: MLXPreparedInput,
    cache: MLXPromptCache?,
    parameters: GenerateParameters,
    grammar: GBNFGrammar?
  ) async throws -> AsyncStream<Generation> {
    try await generate(input: input, cache: cache, parameters: parameters)
  }
}

@_spi(Testing) extension ModelContainer: MLXModelContainerProtocol {
  public func prepare(
    messages: [[String: String]],
    tools: [[String: any Sendable]]?,
    thinkingEnabled: Bool?
  ) async throws -> MLXPreparedInput {
    // Preserve the existing absent-tools context for the prose path.
    let input = try await prepare(
      input: .init(
        messages: messages,
        tools: tools?.isEmpty == false ? tools : nil,
        additionalContext: thinkingEnabled.map { ["enable_thinking": $0] }
      )
    )
    return MLXPreparedInput(input)
  }

  public func prepare(
    chat: SendableChatMessages,
    thinkingEnabled: Bool?
  ) async throws -> MLXPreparedInput {
    try await perform(nonSendable: chat.value) { context, chat in
      let input = try await context.processor.prepare(
        input: .init(chat: chat, additionalContext: thinkingEnabled.map { ["enable_thinking": $0] })
      )
      return MLXPreparedInput(input)
    }
  }

  public func decodePromptTail(_ input: MLXPreparedInput, maxTokens: Int) async -> String {
    await perform { context in
      context.tokenizer.decode(
        tokenIds: input.promptTokenSuffix(maxCount: maxTokens), skipSpecialTokens: false
      )
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
