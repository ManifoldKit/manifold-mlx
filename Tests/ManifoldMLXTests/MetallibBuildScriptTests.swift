import Foundation
import XCTest

final class MetallibBuildScriptTests: XCTestCase {
  func testCompilerFailuresAndCacheInvalidation() throws {
    let root = URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    let log = FileManager.default.temporaryDirectory
      .appendingPathComponent("metallib-fixtures-\(UUID().uuidString).log")
    try Data().write(to: log)
    defer {
      do { try FileManager.default.removeItem(at: log) }
      catch { XCTFail("Could not remove fixture log: \(error)") }
    }
    let output = try FileHandle(forWritingTo: log)
    defer {
      do { try output.close() }
      catch { XCTFail("Could not close fixture log: \(error)") }
    }
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
    process.arguments = [
      root.appendingPathComponent("scripts/tests/test-build-mlx-metallib.py").path,
      root.appendingPathComponent("scripts/build-mlx-metallib.sh").path,
    ]
    // A file prevents pipe-buffer deadlock when a failed fixture prints diagnostics.
    process.standardOutput = output
    process.standardError = output
    try process.run()
    process.waitUntilExit()
    let diagnostics = try String(contentsOf: log, encoding: .utf8)
    XCTAssertEqual(process.terminationStatus, 0, diagnostics)
    XCTAssertTrue(diagnostics.contains("Ran 8 tests"), diagnostics)
  }
}
