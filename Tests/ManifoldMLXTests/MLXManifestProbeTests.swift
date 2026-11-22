import ManifoldInference
@_spi(Testing) @testable import ManifoldMLX
import XCTest

/// Unit tests for ``MLXModelProbe/produceManifest(at:detectedThinkingMarkers:supportsVision:)``.
///
/// These tests don't load real MLX weights — they construct a temporary
/// directory with a synthetic `config.json` and assert the probe extracts the
/// right context window. The Metal-bound path (real `loadContainer`) lives in
/// `ManifoldMLXIntegrationTests` and is hardware-gated.
final class MLXManifestProbeTests: XCTestCase {

  // MARK: - Helpers

  private func makeTempDir() throws -> URL {
    let dir = FileManager.default.temporaryDirectory
      .appendingPathComponent("mlx-manifest-probe-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    addTeardownBlock {
      try? FileManager.default.removeItem(at: dir)
    }
    return dir
  }

  private func writeConfig(_ json: [String: Any], in dir: URL) throws {
    let data = try JSONSerialization.data(withJSONObject: json, options: [])
    try data.write(to: dir.appendingPathComponent("config.json"))
  }

  // MARK: - Effective tokenizer chat template

  private func writeTokenizerConfig(_ template: Any? = nil, in dir: URL) throws {
    var config: [String: Any] = ["tokenizer_class": "PreTrainedTokenizerFast"]
    config["chat_template"] = template
    try JSONSerialization.data(withJSONObject: config)
      .write(to: dir.appendingPathComponent("tokenizer_config.json"))
    let data = #"""
      {
        "model": {"type": "BPE", "merges": [], "vocab": {"\n": 0, "x": 1}},
        "added_tokens": [
          {"id": 2, "content": "<think>", "special": true},
          {"id": 3, "content": "</think>", "special": true},
          {"id": 4, "content": "<reasoning>", "special": true},
          {"id": 5, "content": "</reasoning>", "special": true}
        ]
      }
      """#
    try Data(data.utf8).write(to: dir.appendingPathComponent("tokenizer.json"))
  }

  private func writeExternalJSONTemplate(_ template: String, in dir: URL) throws {
    try JSONSerialization.data(withJSONObject: ["chat_template": template])
      .write(to: dir.appendingPathComponent("chat_template.json"))
  }

  private func renderTemplate(
    in dir: URL, thinkingEnabled: Bool? = nil, withTools: Bool = false
  ) async throws -> String {
    let tokenizer = try await TransformersTokenizerLoader().load(from: dir)
    let ids = try tokenizer.applyChatTemplate(
      messages: [["role": "user", "content": "x"]],
      tools: withTools ? [["type": "function"]] : nil,
      additionalContext: thinkingEnabled.map { ["enable_thinking": $0] }
    )
    return tokenizer.decode(tokenIds: ids, skipSpecialTokens: false)
  }

  func test_detectsThinkingMarkersInExternalJinjaTemplate() async throws {
    let dir = try makeTempDir()
    try writeTokenizerConfig("{{ '<reasoning></reasoning>' }}", in: dir)
    try writeExternalJSONTemplate("{{ 'x' }}", in: dir)
    // Default-on branch used by Qwen3.5-9B.
    let template = #"""
      {%- if enable_thinking is defined and enable_thinking is false -%}
      {{- '<think>\n\n</think>\n\n' -}}
      {%- else -%}
      {{- '<think>\n' -}}
      {%- endif -%}
      """#
    try Data(template.utf8).write(to: dir.appendingPathComponent("chat_template.jinja"))

    let markers = MLXModelProbe.detectThinkingMarkers(at: dir)
    XCTAssertEqual(markers.standard, .qwen3)
    XCTAssertEqual(markers.withTools, .qwen3)
    let modelDefault = try await renderTemplate(in: dir)
    let enabled = try await renderTemplate(in: dir, thinkingEnabled: true)
    let disabled = try await renderTemplate(in: dir, thinkingEnabled: false)
    XCTAssertEqual(modelDefault, "<think>\n")
    XCTAssertEqual(enabled, "<think>\n")
    XCTAssertEqual(disabled, "<think>\n\n</think>\n\n")
  }

  func test_externalJSONTemplateOverridesEmbeddedAndCanDefaultToDisabled() async throws {
    let dir = try makeTempDir()
    try writeTokenizerConfig("{{ '<reasoning></reasoning>' }}", in: dir)
    // Default-off branch used by Qwen3.5-2B.
    try writeExternalJSONTemplate(#"""
      {%- if enable_thinking is defined and enable_thinking is true -%}
      {{- '<think>\n' -}}
      {%- else -%}
      {{- '<think>\n\n</think>\n\n' -}}
      {%- endif -%}
      """#, in: dir)

    XCTAssertEqual(MLXModelProbe.detectThinkingMarkers(at: dir).standard, .qwen3)
    let modelDefault = try await renderTemplate(in: dir)
    let enabled = try await renderTemplate(in: dir, thinkingEnabled: true)
    let disabled = try await renderTemplate(in: dir, thinkingEnabled: false)
    XCTAssertEqual(modelDefault, "<think>\n\n</think>\n\n")
    XCTAssertEqual(enabled, "<think>\n")
    XCTAssertEqual(disabled, modelDefault)
  }

  func test_existingUnreadableJinjaFallsBackToEmbeddedTemplate() async throws {
    let dir = try makeTempDir()
    try writeTokenizerConfig("{{ '<reasoning></reasoning>' }}", in: dir)
    try writeExternalJSONTemplate("{{ '<think></think>' }}", in: dir)
    try Data([0xFF]).write(to: dir.appendingPathComponent("chat_template.jinja"))

    XCTAssertEqual(MLXModelProbe.detectThinkingMarkers(at: dir).standard, .phi4)
    let rendered = try await renderTemplate(in: dir)
    XCTAssertEqual(rendered, "<reasoning></reasoning>")
  }

  func test_embeddedMultiTemplateUsesDefaultAndExactToolUseName() async throws {
    let dir = try makeTempDir()
    try writeTokenizerConfig([
      ["name": "Default", "template": "{{ 'x' }}"],
      ["name": "default", "template": "{{ '<think></think>' }}"],
      ["name": "tool_use", "template": "{{ '<reasoning></reasoning>' }}"],
    ], in: dir)

    let markers = MLXModelProbe.detectThinkingMarkers(at: dir)
    XCTAssertEqual(markers.standard, .qwen3)
    XCTAssertEqual(markers.withTools, .phi4)
    let plain = try await renderTemplate(in: dir)
    let withTools = try await renderTemplate(in: dir, withTools: true)
    XCTAssertEqual(plain, "<think></think>")
    XCTAssertEqual(withTools, "<reasoning></reasoning>")
  }

  func test_manifestDescribesOnlyReachableThinkingTemplates() throws {
    let dir = try makeTempDir()
    try writeConfig(["model_type": "fixture", "max_position_embeddings": 2048], in: dir)
    try writeTokenizerConfig([
      ["name": "default", "template": "{{ 'x' }}"],
      ["name": "tool_use", "template": "{{ '<thinking></thinking>' }}"],
    ], in: dir)

    let markers = MLXModelProbe.detectThinkingMarkers(at: dir)
    let mistral = MLXModelProbe.produceManifest(
      at: dir, detectedThinkingMarkers: markers.forManifest(dialect: .mistral), supportsVision: false
    )
    let qwen = MLXModelProbe.produceManifest(
      at: dir, detectedThinkingMarkers: markers.forManifest(dialect: .qwen25), supportsVision: false
    )

    XCTAssertTrue(mistral.supportsThinking)
    XCTAssertEqual(mistral.thinkingMarkers, .mistralReasoning)
    XCTAssertFalse(qwen.supportsThinking)
    XCTAssertNil(qwen.thinkingMarkers)
  }

  func test_doesNotInventThinkingMarkersForMissingOrUnselectedTemplate() throws {
    let dir = try makeTempDir()
    XCTAssertNil(MLXModelProbe.detectThinkingMarkers(at: dir).standard)
    try writeTokenizerConfig([
      ["name": "Default", "template": "{{ '<think></think>' }}"]
    ], in: dir)

    let markers = MLXModelProbe.detectThinkingMarkers(at: dir)
    XCTAssertNil(markers.standard)
    XCTAssertNil(markers.withTools)
  }

  // MARK: - Context window extraction

  func test_extractsTopLevelMaxPositionEmbeddings() throws {
    let dir = try makeTempDir()
    try writeConfig(
      [
        "model_type": "qwen3",
        "max_position_embeddings": 32_768,
      ], in: dir)

    let manifest = MLXModelProbe.produceManifest(
      at: dir,
      detectedThinkingMarkers: nil,
      supportsVision: false
    )

    XCTAssertEqual(manifest.contextWindow, 32_768)
    XCTAssertEqual(manifest.modelIdentifier, dir.lastPathComponent)
    XCTAssertEqual(manifest.producerKind, .local)
    XCTAssertFalse(
      manifest.supportsThinking,
      "supportsThinking must be false when no markers are detected")
  }

  func test_extractsNestedTextConfigMaxPositionEmbeddings() throws {
    let dir = try makeTempDir()
    try writeConfig(
      [
        "model_type": "gemma4",
        "text_config": [
          "max_position_embeddings": 131_072
        ] as [String: Any],
      ], in: dir)

    let manifest = MLXModelProbe.produceManifest(
      at: dir,
      detectedThinkingMarkers: nil,
      supportsVision: false
    )
    XCTAssertEqual(
      manifest.contextWindow, 131_072,
      "text_config.max_position_embeddings must be preferred over the top-level value")
  }

  func test_textConfigWinsOverTopLevel_whenBothPresent() throws {
    let dir = try makeTempDir()
    try writeConfig(
      [
        "model_type": "vlm-test",
        "max_position_embeddings": 4096,
        "text_config": [
          "max_position_embeddings": 200_000
        ] as [String: Any],
      ], in: dir)

    let manifest = MLXModelProbe.produceManifest(
      at: dir,
      detectedThinkingMarkers: nil,
      supportsVision: true
    )
    XCTAssertEqual(
      manifest.contextWindow, 200_000,
      "text_config wins so VLM/MoE configs reflect the text-encoder window, not the (smaller) image-encoder window"
    )
  }

  func test_fallsBackToModelMaxLength_whenMaxPositionAbsent() throws {
    let dir = try makeTempDir()
    try writeConfig(
      [
        "model_type": "legacy-mlx",
        "model_max_length": 16_384,
      ], in: dir)

    let manifest = MLXModelProbe.produceManifest(
      at: dir,
      detectedThinkingMarkers: nil,
      supportsVision: false
    )
    XCTAssertEqual(manifest.contextWindow, 16_384)
  }

  func test_returnsUnknownDefaults_whenConfigJsonMissing() throws {
    let dir = try makeTempDir()
    // No config.json written.

    let manifest = MLXModelProbe.produceManifest(
      at: dir,
      detectedThinkingMarkers: nil,
      supportsVision: false
    )
    XCTAssertNil(
      manifest.contextWindow,
      "Missing config.json must leave contextWindow nil — ModelManifest.unknown no longer fabricates 8192 (core PR #2404)"
    )
    XCTAssertFalse(manifest.supportsThinking)
    XCTAssertFalse(
      manifest.supportsTools,
      "ModelManifest.unknown reports no tool support")
  }

  func test_leavesContextWindowNil_whenNoContextHintFound() throws {
    let dir = try makeTempDir()
    try writeConfig(
      [
        "model_type": "minimal"
      ], in: dir)

    let manifest = MLXModelProbe.produceManifest(
      at: dir,
      detectedThinkingMarkers: nil,
      supportsVision: false
    )
    XCTAssertNil(
      manifest.contextWindow,
      "Configs without max_position_embeddings / model_max_length must leave contextWindow nil rather than guessing 8k"
    )
  }

  // MARK: - positiveInt coercion of max_position_embeddings

  func test_maxPositionEmbeddings_asJSONString_parses() throws {
    let dir = try makeTempDir()
    try writeConfig(
      [
        "model_type": "qwen3",
        "max_position_embeddings": "32768",
      ], in: dir)

    let manifest = MLXModelProbe.produceManifest(
      at: dir, detectedThinkingMarkers: nil, supportsVision: false
    )
    XCTAssertEqual(
      manifest.contextWindow, 32_768,
      "a string-encoded max_position_embeddings must coerce to Int")
  }

  func test_maxPositionEmbeddings_asDouble_parses() throws {
    let dir = try makeTempDir()
    // A fractional literal forces JSONSerialization to read it back as Double.
    try writeConfig(
      [
        "model_type": "qwen3",
        "max_position_embeddings": 32_768.0,
      ], in: dir)

    let manifest = MLXModelProbe.produceManifest(
      at: dir, detectedThinkingMarkers: nil, supportsVision: false
    )
    XCTAssertEqual(
      manifest.contextWindow, 32_768,
      "a floating-point max_position_embeddings must coerce to Int")
  }

  func test_maxPositionEmbeddings_asInt64_parses() throws {
    let dir = try makeTempDir()
    try writeConfig(
      [
        "model_type": "qwen3",
        "max_position_embeddings": Int64(32_768),
      ], in: dir)

    let manifest = MLXModelProbe.produceManifest(
      at: dir, detectedThinkingMarkers: nil, supportsVision: false
    )
    XCTAssertEqual(
      manifest.contextWindow, 32_768,
      "an Int64 max_position_embeddings must coerce to Int")
  }

  func test_maxPositionEmbeddings_zero_leavesContextWindowNil() throws {
    let dir = try makeTempDir()
    try writeConfig(
      [
        "model_type": "qwen3",
        "max_position_embeddings": 0,
      ], in: dir)

    let manifest = MLXModelProbe.produceManifest(
      at: dir, detectedThinkingMarkers: nil, supportsVision: false
    )
    XCTAssertNil(
      manifest.contextWindow,
      "zero is rejected by positiveInt and must leave contextWindow nil, not guess 8k")
  }

  func test_maxPositionEmbeddings_negative_leavesContextWindowNil() throws {
    let dir = try makeTempDir()
    try writeConfig(
      [
        "model_type": "qwen3",
        "max_position_embeddings": -4096,
      ], in: dir)

    let manifest = MLXModelProbe.produceManifest(
      at: dir, detectedThinkingMarkers: nil, supportsVision: false
    )
    XCTAssertNil(
      manifest.contextWindow,
      "a negative value is rejected by positiveInt and must leave contextWindow nil, not guess 8k")
  }

  // MARK: - Thinking marker plumbing

  func test_carriesDetectedThinkingMarkers() throws {
    let dir = try makeTempDir()
    try writeConfig(
      [
        "model_type": "qwen3",
        "max_position_embeddings": 32_768,
      ], in: dir)

    let manifest = MLXModelProbe.produceManifest(
      at: dir,
      detectedThinkingMarkers: .qwen3,
      supportsVision: false
    )
    XCTAssertEqual(manifest.thinkingMarkers, .qwen3)
    XCTAssertTrue(
      manifest.supportsThinking,
      "supportsThinking is implied by non-nil thinkingMarkers")
  }

  // MARK: - isUnsupportedGemma4 (upstream #282/#802 crash guard)

  /// Every Gemma 4 model is refused — both the dense/multimodal e4b path
  /// (#282) and the MoE path (#802) crash, regardless of factory routing.
  func test_isUnsupportedGemma4_trueForGemma4() {
    XCTAssertTrue(MLXModelProbe.isUnsupportedGemma4(modelType: "gemma4"))
  }

  /// Neighbouring Gemma generations and other architectures are unaffected,
  /// as is a missing model_type.
  func test_isUnsupportedGemma4_falseForOtherArchitectures() {
    XCTAssertFalse(MLXModelProbe.isUnsupportedGemma4(modelType: "gemma3"))
    XCTAssertFalse(MLXModelProbe.isUnsupportedGemma4(modelType: "gemma3n"))
    XCTAssertFalse(MLXModelProbe.isUnsupportedGemma4(modelType: "llama"))
    XCTAssertFalse(MLXModelProbe.isUnsupportedGemma4(modelType: nil))
  }

  // MARK: - #2348 review round 2, item 2: closes the `_injectManifest`-only test gap
  //
  // Every executed test for the #2348 context-ceiling warning
  // (`MLXBackendGenerationTests.swift`) goes through
  // `MLXBackend._injectManifest(_:)`, a test-only seam that bypasses
  // `loadModel(from:plan:)` entirely. That leaves the claim "the primary
  // warning is live in production" resting on a static trace through
  // `loadModel` → `produceManifest` → `withStateLock { _manifest = ... }`,
  // not on anything executed — the exact gap MK-4 closed on the
  // ManifoldKit side (`ModelStorageServiceTests.swift`'s
  // `test_discoverModels_mlxDirectory_populatesDetectedContextLengthFromConfig`).
  // `loadModel` itself needs real Apple-Silicon hardware and can't be
  // exercised here (`MLXBackendTests.swift`'s own header comment: hardware-
  // gated load→unload cycles are covered by `ManifoldE2ETests`, not local
  // unit tests) — but the piece that actually determines the warning's
  // threshold — `produceManifest` reading a real on-disk `config.json` —
  // needs neither GPU nor weights. This test is the explicit bridge: it
  // names the connection to the #2348 warning feature so a future reader
  // doesn't have to rediscover it, and pins the value (131072) at the
  // realistic scale `MLXBackendGenerationTests`'s own fixtures use.

  /// A value that would drive a real #2348 primary-warning threshold, read
  /// from disk with no injection anywhere in the call chain — this is what
  /// `MLXBackend.loadModel` assigns to `_manifest`, which
  /// `reportContextCheck`'s primary warning compares against. Since core
  /// PR #2404, a non-`nil` `contextWindow` on its own is what tells the
  /// primary warning the number was measured, not guessed — there is no
  /// separate detection side-channel to assert on any more.
  func test_contextWindow_fromRealConfigOnDisk_matchesWhatWouldDriveThePrimaryWarningThreshold()
    throws
  {
    let dir = try makeTempDir()
    try writeConfig(
      [
        "model_type": "llama",
        "max_position_embeddings": 131_072,
      ], in: dir)

    let manifest = MLXModelProbe.produceManifest(
      at: dir,
      detectedThinkingMarkers: nil,
      supportsVision: false
    )

    XCTAssertEqual(
      manifest.contextWindow, 131_072,
      "produceManifest must read the real trained context from config.json on disk"
    )

    // Sabotage check: replacing `extractContextWindow(from: json)` with
    // `nil` (simulating a broken key-reader) turns this test red —
    // verified manually, reverted after confirming.
  }
}
