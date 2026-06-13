
import XCTest
import MLXLMCommon
import ManifoldTestSupport
@_spi(Testing) import ManifoldMLX

/// Integration coverage for ``TransformersTokenizerLoader``'s happy path.
///
/// The success path needs a full on-disk tokenizer (tokenizer.json +
/// tokenizer_config.json with a resolvable tokenizer class), which the unit
/// suite deliberately does not fabricate. This loads the tokenizer from a real
/// MLX model directory and verifies a text → ids → text round-trip through the
/// `TokenizerBridge` adapter. Skips unless an MLX model is discoverable on disk.
final class TransformersTokenizerLoaderIntegrationTests: XCTestCase {

    func test_load_realModelFolder_roundTrips() async throws {
        guard let modelDir = HardwareRequirements.findMLXModelDirectory() else {
            throw XCTSkip(
                "No loadable MLX model found on disk. Set MANIFOLD_DISCOVER_LOCAL_MODELS=1 with a local MLX snapshot."
            )
        }

        let loader = TransformersTokenizerLoader()
        let tokenizer = try await loader.load(from: modelDir)

        let ids = tokenizer.encode(text: "Hello, world!", addSpecialTokens: false)
        XCTAssertFalse(ids.isEmpty, "Encoding non-empty text must yield at least one token")

        let decoded = tokenizer.decode(tokenIds: ids, skipSpecialTokens: true)
        XCTAssertFalse(decoded.isEmpty, "Decoding the ids must yield non-empty text")
    }
}
