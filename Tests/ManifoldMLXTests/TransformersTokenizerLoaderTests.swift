
import XCTest
import MLXLMCommon
import Tokenizers
@_spi(Testing) import ManifoldMLX

/// Unit tests for ``TransformersTokenizerLoader``.
///
/// The loader is a thin pass-through to swift-transformers'
/// `AutoTokenizer.from(modelFolder:)`. These tests cover its deterministic
/// error surface — a missing directory, a directory with no
/// `tokenizer_config.json`, and a malformed config file — none of which touch
/// Metal or require model weights. The happy path (a real tokenizer that
/// round-trips text → ids → text) needs a full on-disk tokenizer and lives in
/// the integration tier (``TransformersTokenizerLoaderIntegrationTests``).
final class TransformersTokenizerLoaderTests: XCTestCase {

    private func makeTempDir() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appending(component: "TokLoaderTest-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    // MARK: - Error paths

    func test_load_missingDirectory_throws() async {
        // A directory that was never created on disk.
        let dir = FileManager.default.temporaryDirectory
            .appending(component: "TokLoaderTest-missing-\(UUID().uuidString)")

        let loader = TransformersTokenizerLoader()
        do {
            _ = try await loader.load(from: dir)
            XCTFail("Expected load(from:) to throw for a non-existent directory")
        } catch {
            // expected — no tokenizer config can be read
            XCTAssertNotNil(error)
        }
    }

    func test_load_emptyDirectory_throws() async throws {
        // An existing directory with no tokenizer files at all. swift-transformers
        // cannot resolve a config or tokenizer.json, so load must throw rather
        // than returning a degenerate tokenizer.
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        let loader = TransformersTokenizerLoader()
        do {
            _ = try await loader.load(from: dir)
            XCTFail("Expected load(from:) to throw for a directory with no tokenizer files")
        } catch {
            // The concrete error type varies (missing tokenizer_config.json vs.
            // missing tokenizer.json depending on which read fails first); the
            // contract is only that an empty folder cannot produce a tokenizer.
            XCTAssertNotNil(error)
        }
    }

    func test_load_malformedConfigJSON_throws() async throws {
        // tokenizer_config.json present but not valid JSON. Loading must fail
        // rather than silently produce a degenerate tokenizer.
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        let configURL = dir.appending(component: "tokenizer_config.json")
        try Data("{ this is not valid json ".utf8).write(to: configURL)

        let loader = TransformersTokenizerLoader()
        do {
            _ = try await loader.load(from: dir)
            XCTFail("Expected load(from:) to throw for a malformed tokenizer_config.json")
        } catch {
            // Any thrown error is acceptable — the contract is that malformed
            // config does not yield a usable tokenizer.
            XCTAssertNotNil(error)
        }
    }

    func test_load_configWithoutTokenizerClass_throws() async throws {
        // Valid JSON, but no tokenizer_class entry and no tokenizer.json data.
        // swift-transformers cannot determine a tokenizer model, so load fails.
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        let configURL = dir.appending(component: "tokenizer_config.json")
        try Data(#"{ "bos_token": "<s>" }"#.utf8).write(to: configURL)

        let loader = TransformersTokenizerLoader()
        do {
            _ = try await loader.load(from: dir)
            XCTFail("Expected load(from:) to throw when no tokenizer model can be resolved")
        } catch {
            XCTAssertNotNil(error)
        }
    }
}

// MARK: - TokenizerBridge adapter

/// Records calls into a `Tokenizers.Tokenizer` so the private `TokenizerBridge`
/// (reached via the `@_spi(Testing)` `makeBridge(from:)` seam) can be tested
/// without a real on-disk tokenizer.
///
/// Only the methods the bridge actually forwards are wired to recorders; the
/// remaining protocol requirements are trap-stubbed because the bridge never
/// touches them.
private final class StubTokenizer: Tokenizers.Tokenizer, @unchecked Sendable {

    // Recorded inputs / configured outputs for the methods the bridge forwards.
    var lastEncodeText: String?
    var lastEncodeAddSpecial: Bool?
    var encodeResult: [Int] = []

    var lastDecodeTokens: [Int]?
    var lastDecodeSkipSpecial: Bool?
    var decodeResult: String = ""

    var lastConvertTokenToId: String?
    var convertTokenToIdResult: Int?

    var lastConvertIdToToken: Int?
    var convertIdToTokenResult: String?

    var bosToken: String?
    var eosToken: String?
    var unknownToken: String?

    /// When set, the chat-template overload the bridge calls throws this.
    var chatTemplateError: Error?
    /// Otherwise returns this token array.
    var chatTemplateResult: [Int] = []
    var lastChatTemplateMessages: [Tokenizers.Message]?
    var lastChatTemplateTools: [Tokenizers.ToolSpec]?

    // MARK: Forwarded surface

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        lastEncodeText = text
        lastEncodeAddSpecial = addSpecialTokens
        return encodeResult
    }

    func decode(tokens: [Int], skipSpecialTokens: Bool) -> String {
        lastDecodeTokens = tokens
        lastDecodeSkipSpecial = skipSpecialTokens
        return decodeResult
    }

    func convertTokenToId(_ token: String) -> Int? {
        lastConvertTokenToId = token
        return convertTokenToIdResult
    }

    func convertIdToToken(_ id: Int) -> String? {
        lastConvertIdToToken = id
        return convertIdToTokenResult
    }

    func applyChatTemplate(
        messages: [Tokenizers.Message],
        tools: [Tokenizers.ToolSpec]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        lastChatTemplateMessages = messages
        lastChatTemplateTools = tools
        if let chatTemplateError { throw chatTemplateError }
        return chatTemplateResult
    }

    // MARK: Unused requirements (bridge never calls these)

    func tokenize(text: String) -> [String] { fatalError("unused") }
    func encode(text: String) -> [Int] { fatalError("unused") }
    var bosTokenId: Int? { nil }
    var eosTokenId: Int? { nil }
    var unknownTokenId: Int? { nil }

    func applyChatTemplate(messages: [Tokenizers.Message]) throws -> [Int] { fatalError("unused") }
    func applyChatTemplate(messages: [Tokenizers.Message], tools: [Tokenizers.ToolSpec]?) throws -> [Int] {
        fatalError("unused")
    }
    func applyChatTemplate(messages: [Tokenizers.Message], chatTemplate: ChatTemplateArgument) throws -> [Int] {
        fatalError("unused")
    }
    func applyChatTemplate(messages: [Tokenizers.Message], chatTemplate: String) throws -> [Int] {
        fatalError("unused")
    }
    func applyChatTemplate(
        messages: [Tokenizers.Message],
        chatTemplate: ChatTemplateArgument?,
        addGenerationPrompt: Bool,
        truncation: Bool,
        maxLength: Int?,
        tools: [Tokenizers.ToolSpec]?
    ) throws -> [Int] {
        fatalError("unused")
    }
    func applyChatTemplate(
        messages: [Tokenizers.Message],
        chatTemplate: ChatTemplateArgument?,
        addGenerationPrompt: Bool,
        truncation: Bool,
        maxLength: Int?,
        tools: [Tokenizers.ToolSpec]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        fatalError("unused")
    }
}

/// Unit tests for the private `TokenizerBridge` adapter, driven through the
/// `@_spi(Testing)` `TransformersTokenizerLoader.makeBridge(from:)` seam.
///
/// The bridge maps `MLXLMCommon.Tokenizer` calls onto a `Tokenizers.Tokenizer`.
/// The load-bearing bits are the `decode(tokenIds:)` → `decode(tokens:)` label
/// remap and the `missingChatTemplate` error translation.
final class TokenizerBridgeTests: XCTestCase {

    private func makeBridge(_ stub: StubTokenizer) -> any MLXLMCommon.Tokenizer {
        TransformersTokenizerLoader.makeBridge(from: stub)
    }

    func test_encode_forwardsArgsAndResult() {
        let stub = StubTokenizer()
        stub.encodeResult = [1, 2, 3]
        let bridge = makeBridge(stub)

        let result = bridge.encode(text: "hello", addSpecialTokens: true)

        XCTAssertEqual(result, [1, 2, 3])
        XCTAssertEqual(stub.lastEncodeText, "hello")
        XCTAssertEqual(stub.lastEncodeAddSpecial, true)
    }

    func test_decode_remapsTokenIdsLabelToUpstreamTokens() {
        let stub = StubTokenizer()
        stub.decodeResult = "decoded"
        let bridge = makeBridge(stub)

        let result = bridge.decode(tokenIds: [7, 8, 9], skipSpecialTokens: true)

        XCTAssertEqual(result, "decoded")
        // The bridge calls upstream.decode(tokens:) — the label remap is the point.
        XCTAssertEqual(stub.lastDecodeTokens, [7, 8, 9])
        XCTAssertEqual(stub.lastDecodeSkipSpecial, true)
    }

    func test_convertTokenToId_passesThroughIncludingNil() {
        let stub = StubTokenizer()
        stub.convertTokenToIdResult = 42
        let bridge = makeBridge(stub)
        XCTAssertEqual(bridge.convertTokenToId("<eos>"), 42)
        XCTAssertEqual(stub.lastConvertTokenToId, "<eos>")

        stub.convertTokenToIdResult = nil
        XCTAssertNil(bridge.convertTokenToId("???"))
    }

    func test_convertIdToToken_passesThroughIncludingNil() {
        let stub = StubTokenizer()
        stub.convertIdToTokenResult = "<bos>"
        let bridge = makeBridge(stub)
        XCTAssertEqual(bridge.convertIdToToken(0), "<bos>")
        XCTAssertEqual(stub.lastConvertIdToToken, 0)

        stub.convertIdToTokenResult = nil
        XCTAssertNil(bridge.convertIdToToken(-1))
    }

    func test_specialTokens_forward() {
        let stub = StubTokenizer()
        stub.bosToken = "<s>"
        stub.eosToken = "</s>"
        stub.unknownToken = "<unk>"
        let bridge = makeBridge(stub)

        XCTAssertEqual(bridge.bosToken, "<s>")
        XCTAssertEqual(bridge.eosToken, "</s>")
        XCTAssertEqual(bridge.unknownToken, "<unk>")
    }

    func test_applyChatTemplate_success_forwardsTokens() throws {
        let stub = StubTokenizer()
        stub.chatTemplateResult = [10, 20, 30]
        let bridge = makeBridge(stub)

        let messages: [[String: any Sendable]] = [["role": "user", "content": "hi"]]
        let result = try bridge.applyChatTemplate(messages: messages, tools: nil, additionalContext: nil)

        XCTAssertEqual(result, [10, 20, 30])
        XCTAssertEqual(stub.lastChatTemplateMessages?.count, 1)
    }

    func test_applyChatTemplate_missingChatTemplate_isTranslated() {
        let stub = StubTokenizer()
        stub.chatTemplateError = Tokenizers.TokenizerError.missingChatTemplate
        let bridge = makeBridge(stub)

        do {
            _ = try bridge.applyChatTemplate(messages: [], tools: nil, additionalContext: nil)
            XCTFail("Expected MLXLMCommon.TokenizerError.missingChatTemplate")
        } catch MLXLMCommon.TokenizerError.missingChatTemplate {
            // expected — the upstream missingChatTemplate is re-thrown as the
            // MLXLMCommon equivalent.
        } catch {
            XCTFail("Expected MLXLMCommon.TokenizerError.missingChatTemplate, got \(error)")
        }
    }

    func test_applyChatTemplate_otherError_propagatesUnchanged() {
        let stub = StubTokenizer()
        // A different upstream error must NOT be translated.
        stub.chatTemplateError = Tokenizers.TokenizerError.chatTemplate("boom")
        let bridge = makeBridge(stub)

        do {
            _ = try bridge.applyChatTemplate(messages: [], tools: nil, additionalContext: nil)
            XCTFail("Expected the original Tokenizers.TokenizerError to propagate")
        } catch Tokenizers.TokenizerError.chatTemplate(let detail) {
            XCTAssertEqual(detail, "boom")
        } catch {
            XCTFail("Expected Tokenizers.TokenizerError.chatTemplate, got \(error)")
        }
    }
}
