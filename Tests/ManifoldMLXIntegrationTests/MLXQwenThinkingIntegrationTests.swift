import ManifoldInference
@_spi(Testing) import ManifoldMLX
import ManifoldTestSupport
import XCTest

/// Opt-in check of the native tokenizer and decoder using a local Qwen3.5-9B snapshot.
@MainActor
final class MLXQwenThinkingIntegrationTests: XCTestCase {
  private struct Reply {
    var text = ""
    var reasoning = ""
    var reasoningEvents = 0
    var reasoningCompletions = 0
  }

  private func collect(_ stream: GenerationStream) async throws -> Reply {
    var reply = Reply()
    for try await event in stream.events {
      switch event {
      case .token(let text): reply.text += text
      case .thinkingToken(let text):
        reply.reasoning += text
        reply.reasoningEvents += 1
      case .thinkingCompleted: reply.reasoningCompletions += 1
      default: break
      }
    }
    return reply
  }

  func test_qwen35NativeDecoderSeparatesPrefilledThinkingAndVisibleAnswer() async throws {
    guard let path = ProcessInfo.processInfo.environment["MANIFOLD_QWEN_THINKING_TEST_MODEL"] else {
      throw XCTSkip("Set MANIFOLD_QWEN_THINKING_TEST_MODEL to a local Qwen3.5-9B MLX snapshot.")
    }
    guard HardwareRequirements.isAppleSilicon, HardwareRequirements.hasMetalDevice else {
      XCTFail("The opted-in native Qwen smoke requires Apple Silicon and Metal.")
      return
    }
    let dir = URL(fileURLWithPath: path, isDirectory: true)
    let prompt = "What is 17 plus 26? Reply with only the number."
    let tokenizer = try await TransformersTokenizerLoader().load(from: dir)
    let messages = [["role": "user", "content": prompt]]

    for (flag, expectedSuffix) in [
      (nil as Bool?, "<think>\n"),
      (false, "<think>\n\n</think>\n\n"),
      (true, "<think>\n"),
    ] {
      let ids = try tokenizer.applyChatTemplate(
        messages: messages, tools: nil, additionalContext: flag.map { ["enable_thinking": $0] }
      )
      let tail = tokenizer.decode(tokenIds: Array(ids.suffix(128)), skipSpecialTokens: false)
      XCTAssertTrue(
        tail.hasSuffix(expectedSuffix),
        "Unexpected Qwen prompt tail for \(String(describing: flag))")
    }

    let service = InferenceService(backend: MockInferenceBackend())
    MLXBackends.register(with: service, thinkingDefault: .disabled)
    let model = ModelInfo(
      name: dir.lastPathComponent, fileName: dir.lastPathComponent, url: dir,
      fileSize: 0, modelType: .mlx
    )
    try await service.loadModel(from: model, plan: .testStub())
    defer { service.unloadModel() }

    let direct = try await collect(
      service.generate(
        messages: [("user", prompt)], temperature: 0, maxOutputTokens: 16
      ))
    XCTAssertEqual(direct.text.trimmingCharacters(in: .whitespacesAndNewlines), "43")
    XCTAssertEqual(direct.reasoningEvents, 0)
    XCTAssertEqual(direct.reasoningCompletions, 0)

    let thinking = try await collect(
      service.generate(
        messages: [("user", prompt)], temperature: 0, maxOutputTokens: 16, maxThinkingTokens: 1
      ))
    XCTAssertEqual(thinking.text.trimmingCharacters(in: .whitespacesAndNewlines), "43")
    XCTAssertFalse(thinking.reasoning.isEmpty)
    XCTAssertEqual(thinking.reasoningEvents, 1)
    XCTAssertEqual(thinking.reasoningCompletions, 1)

    let constrainedRequest = try service.enqueue(
      messages: [.user(prompt)],
      config: GenerationConfig(
        temperature: 0, maxOutputTokens: 32, maxThinkingTokens: 1,
        grammar: #"root ::= "{\"label\":\"<think>demo</think>\"}""#
      )
    )
    let constrained = try await collect(constrainedRequest.stream)
    XCTAssertEqual(constrained.text, #"{"label":"<think>demo</think>"}"#)
    XCTAssertEqual(constrained.reasoningEvents, 0)
    XCTAssertEqual(constrained.reasoningCompletions, 0)
  }
}
