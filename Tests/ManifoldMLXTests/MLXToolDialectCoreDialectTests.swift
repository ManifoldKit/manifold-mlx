import XCTest
import ManifoldMLX
import ManifoldInference

/// Unit tests for ``MLXToolDialect/coreDialect``.
///
/// Each internal dialect case must map to the correct ``ToolCallDialect``
/// family, delimiters, and extractability so ``MLXBackend`` surfaces accurate
/// dialect metadata on ``BackendCapabilities/toolDialect``.
final class MLXToolDialectCoreDialectTests: XCTestCase {

    func test_qwen25_mapsToQwenFamily() {
        let dialect = MLXToolDialect.qwen25.coreDialect
        XCTAssertEqual(dialect.family, .qwen)
    }

    func test_qwen25_hasCorrectOpenDelimiter() {
        let dialect = MLXToolDialect.qwen25.coreDialect
        XCTAssertEqual(dialect.openDelimiter, "<tool_call>")
    }

    func test_qwen25_hasCorrectCloseDelimiter() {
        let dialect = MLXToolDialect.qwen25.coreDialect
        XCTAssertEqual(dialect.closeDelimiter, "</tool_call>")
    }

    func test_qwen25_hasJSONArgEncoding() {
        let dialect = MLXToolDialect.qwen25.coreDialect
        XCTAssertEqual(dialect.argEncoding, .json)
    }

    func test_qwen25_isCleanExtractability() {
        let dialect = MLXToolDialect.qwen25.coreDialect
        XCTAssertEqual(dialect.extractability, .clean)
    }

    func test_llama_mapsToLlamaPythonTagFamily() {
        let dialect = MLXToolDialect.llama.coreDialect
        XCTAssertEqual(dialect.family, .llamaPythonTag)
    }

    func test_llama_hasCorrectOpenDelimiter() {
        let dialect = MLXToolDialect.llama.coreDialect
        XCTAssertEqual(dialect.openDelimiter, "<|python_tag|>")
    }

    func test_llama_hasCorrectCloseDelimiter() {
        let dialect = MLXToolDialect.llama.coreDialect
        XCTAssertEqual(dialect.closeDelimiter, "<|eom_id|>")
    }

    func test_llama_hasJSONArgEncoding() {
        let dialect = MLXToolDialect.llama.coreDialect
        XCTAssertEqual(dialect.argEncoding, .json)
    }

    func test_llama_isBuriedExtractability() {
        let dialect = MLXToolDialect.llama.coreDialect
        XCTAssertEqual(dialect.extractability, .buried)
    }

    func test_unknown_mapsToUnknownFamily() {
        let dialect = MLXToolDialect.unknown.coreDialect
        XCTAssertEqual(dialect.family, .unknown)
    }

    func test_unknown_hasNilOpenDelimiter() {
        let dialect = MLXToolDialect.unknown.coreDialect
        XCTAssertNil(dialect.openDelimiter)
    }

    func test_unknown_hasNilCloseDelimiter() {
        let dialect = MLXToolDialect.unknown.coreDialect
        XCTAssertNil(dialect.closeDelimiter)
    }

    func test_unknown_isToolLessExtractability() {
        let dialect = MLXToolDialect.unknown.coreDialect
        XCTAssertEqual(dialect.extractability, .toolLess)
    }
}
