import XCTest
import ManifoldInference
import ManifoldTestSupport
@_spi(Testing) import ManifoldMLX

/// End-to-end proof on real Apple-Silicon hardware that MLX now executes
/// grammar-constrained sampling (#96, option B): a GBNF grammar derived by
/// `ToolGrammarBuilder` (the exact grammar ManifoldKit's `GenerationQueue`
/// applies on the llama path) is enforced during MLX generation, so the model
/// emits a schema-valid tool call instead of unconstrained prose.
///
/// Requires a Qwen-2.5/3 MLX model; skipped otherwise. Run with:
/// ```
/// xcodebuild test -scheme manifold-mlx-Package -destination 'platform=macOS' \
///   -only-testing:ManifoldMLXIntegrationTests/MLXGrammarSamplingE2ETests
/// ```
@MainActor
final class MLXGrammarSamplingE2ETests: XCTestCase {

    private var backend: MLXBackend!
    private var modelURL: URL!

    override func setUp() async throws {
        try await super.setUp()
        try XCTSkipUnless(HardwareRequirements.isAppleSilicon, "Requires Apple Silicon")
        try XCTSkipUnless(HardwareRequirements.hasMetalDevice, "Requires Metal GPU")
        guard let dir = HardwareRequirements.findMLXModelDirectory(nameContains: "Qwen2.5") else {
            throw XCTSkip("No Qwen-2.5 MLX model found. Set MLX_TEST_MODEL to a Qwen-2.5 snapshot.")
        }
        modelURL = dir
        if let reason = MLXModelProbe.unsupportedGenerationReason(at: dir) {
            throw XCTSkip("\(reason) Model: \(dir.lastPathComponent)")
        }
        backend = MLXBackend()
        try await backend.loadModel(from: modelURL, plan: .testStub(effectiveContextSize: 2048))
    }

    override func tearDown() async throws {
        backend?.unloadModel()
        backend = nil
        modelURL = nil
        try await super.tearDown()
    }

    private func weatherTool() -> ToolDefinition {
        ToolDefinition(
            name: "get_current_weather",
            description: "Get the current weather for a city.",
            parameters: .object([
                "type": .string("object"),
                "properties": .object([
                    "location": .object(["type": .string("string")]),
                ]),
                "required": .array([.string("location")]),
            ])
        )
    }

    /// The capability MLX now advertises.
    func test_capabilities_supportsGrammarConstrainedSampling_isTrue() {
        XCTAssertTrue(backend.capabilities.supportsGrammarConstrainedSampling)
    }

    /// The core option-B claim: the `ToolGrammarBuilder` grammar is enforced, so
    /// even a conversational prompt yields a schema-valid tool call.
    func test_toolGrammar_forcesSchemaValidToolCall() async throws {
        let tool = weatherTool()
        let gbnf = try XCTUnwrap(
            ToolGrammarBuilder().buildGrammar(for: [tool]),
            "builder must emit a grammar for a non-empty tool list"
        )

        var config = GenerationConfig(temperature: 0.0, maxOutputTokens: 96)
        config.tools = [tool]
        config.grammar = gbnf

        let stream = try backend.generate(
            prompt: "Hi there! How are you today?",
            systemPrompt: "You are a helpful assistant with access to tools.",
            config: config
        )

        var toolCalls: [ToolCall] = []
        var text = ""
        for try await event in stream.events {
            switch event {
            case .toolCall(let c): toolCalls.append(c)
            case .token(let t): text += t
            default: break
            }
        }
        print("[#96 grammar] toolCalls=\(toolCalls.count) args=\(toolCalls.first?.arguments ?? "") text=\(text.prefix(120))")

        let call = try XCTUnwrap(toolCalls.first, "grammar must force a parseable tool call")
        XCTAssertEqual(call.toolName, "get_current_weather", "the grammar pins the only tool's name")

        // Arguments must be valid JSON with the schema's required `location` key.
        let data = try XCTUnwrap(call.arguments.data(using: .utf8))
        let obj = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: data) as? [String: Any],
            "arguments must be a JSON object"
        )
        XCTAssertNotNil(obj["location"], "schema requires `location`; grammar must enforce it")
        XCTAssertTrue(obj["location"] is String, "`location` must be a JSON string per schema")
    }
}
