
import XCTest
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
