
import XCTest
import ManifoldMLX

/// Unit tests for ``MLXToolDialect/detect(at:)``.
///
/// Detection reads `config.json` from a model directory and maps the
/// `model_type` field to a tool-call dialect. These tests drive every branch
/// with on-disk fixtures in a temp directory — no model weights, no Metal.
final class MLXToolDialectTests: XCTestCase {

    private func makeTempDir() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appending(component: "ToolDialectTest-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// Writes `config.json` containing `contents` into a fresh temp dir and
    /// returns the directory URL.
    private func makeModelDir(configContents: String) throws -> URL {
        let dir = try makeTempDir()
        try Data(configContents.utf8).write(to: dir.appending(component: "config.json"))
        return dir
    }

    // MARK: - Recognised model types → .qwen25

    func test_detect_qwen2_isQwen25() throws {
        let dir = try makeModelDir(configContents: #"{ "model_type": "qwen2" }"#)
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertEqual(MLXToolDialect.detect(at: dir), .qwen25)
    }

    func test_detect_qwen25_isQwen25() throws {
        let dir = try makeModelDir(configContents: #"{ "model_type": "qwen2.5" }"#)
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertEqual(MLXToolDialect.detect(at: dir), .qwen25)
    }

    func test_detect_qwen3_isQwen25() throws {
        let dir = try makeModelDir(configContents: #"{ "model_type": "qwen3" }"#)
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertEqual(MLXToolDialect.detect(at: dir), .qwen25)
    }

    func test_detect_qwen3Moe_isQwen25() throws {
        let dir = try makeModelDir(configContents: #"{ "model_type": "qwen3_moe" }"#)
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertEqual(MLXToolDialect.detect(at: dir), .qwen25)
    }

    func test_detect_uppercase_isCaseInsensitive() throws {
        let dir = try makeModelDir(configContents: #"{ "model_type": "QWEN2" }"#)
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertEqual(MLXToolDialect.detect(at: dir), .qwen25)
    }

    func test_detect_leadingTrailingWhitespace_isTrimmed() throws {
        let dir = try makeModelDir(configContents: #"{ "model_type": " qwen3 " }"#)
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertEqual(MLXToolDialect.detect(at: dir), .qwen25)
    }

    // MARK: - Unrecognised → .unknown

    func test_detect_notQwenPrefix_isUnknown() throws {
        // Detection is prefix-based, not contains-based: "notqwen2" must not match.
        let dir = try makeModelDir(configContents: #"{ "model_type": "notqwen2" }"#)
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertEqual(MLXToolDialect.detect(at: dir), .unknown)
    }

    func test_detect_llama_isUnknown() throws {
        let dir = try makeModelDir(configContents: #"{ "model_type": "llama" }"#)
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertEqual(MLXToolDialect.detect(at: dir), .unknown)
    }

    func test_detect_missingModelTypeKey_isUnknown() throws {
        let dir = try makeModelDir(configContents: "{}")
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertEqual(MLXToolDialect.detect(at: dir), .unknown)
    }

    func test_detect_missingFileOrDirectory_isUnknown() {
        // A directory that was never created — config.json cannot be read.
        let dir = FileManager.default.temporaryDirectory
            .appending(component: "ToolDialectTest-missing-\(UUID().uuidString)")
        XCTAssertEqual(MLXToolDialect.detect(at: dir), .unknown)
    }

    func test_detect_malformedJSON_isUnknown() throws {
        let dir = try makeModelDir(configContents: "not json")
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertEqual(MLXToolDialect.detect(at: dir), .unknown)
    }

    func test_detect_nonObjectJSON_isUnknown() throws {
        // Valid JSON but a top-level array, not an object → cast to
        // [String: Any] fails → .unknown.
        let dir = try makeModelDir(configContents: "[]")
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertEqual(MLXToolDialect.detect(at: dir), .unknown)
    }
}
