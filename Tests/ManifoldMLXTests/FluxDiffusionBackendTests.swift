
import XCTest
import ManifoldMLX
import ManifoldInference

/// Unit tests for ``FluxDiffusionBackend``.
///
/// These cover the deterministic, non-Metal-bound surface only: initial state,
/// the not-loaded error path, idle state-machine safety, the missing-model
/// directory load failure, and ``FluxDiffusionError`` descriptions. The packed-
/// latent unpack/decode math and PNG encode run on `MLXArray` values that require
/// a Metal device and a real FLUX model — those are exercised in
/// ``FluxDiffusionIntegrationTests`` (Apple Silicon + Metal + a local FLUX
/// snapshot), never here, because the metallib only compiles under Xcode and a
/// real denoise loop would fatally crash a plain `swift test` runner.
final class FluxDiffusionBackendTests: XCTestCase {

    // MARK: - Initial state

    func test_isLoaded_initiallyFalse() {
        XCTAssertFalse(FluxDiffusionBackend().isLoaded)
    }

    func test_isGenerating_initiallyFalse() {
        XCTAssertFalse(FluxDiffusionBackend().isGenerating)
    }

    // MARK: - Error surfaces that don't require Metal

    func test_generate_notLoaded_throwsNotLoaded() throws {
        let backend = FluxDiffusionBackend()
        XCTAssertFalse(backend.isLoaded)
        do {
            _ = try backend.generate(prompt: "a cat", config: .init())
            XCTFail("Expected FluxDiffusionError.notLoaded")
        } catch FluxDiffusionError.notLoaded {
            // expected
        }
        // generate() must not flip isGenerating when it rejects an unloaded backend.
        XCTAssertFalse(backend.isGenerating)
    }

    func test_loadModel_missingDirectory_throws() async {
        // A directory that does not exist on disk. Neither the quantized path
        // (no metadata.json) nor the diffusers path can succeed, so loadModel
        // must throw rather than leaving the backend half-loaded.
        let dir = FileManager.default.temporaryDirectory
            .appending(component: "FluxTest-missing-\(UUID().uuidString)")

        let backend = FluxDiffusionBackend()
        do {
            try await backend.loadModel(from: dir)
            XCTFail("Expected loadModel(from:) to throw for a non-existent directory")
        } catch {
            // Any thrown error is acceptable here — the contract is that a bad
            // path does not silently produce a loaded backend.
        }
        XCTAssertFalse(backend.isLoaded,
                       "A failed loadModel must leave isLoaded == false")
    }

    func test_loadModel_emptyDirectory_throws() async throws {
        // An existing but empty directory: no metadata.json (skips quantized
        // path) and no diffusers weights (diffusers path fails).
        let dir = FileManager.default.temporaryDirectory
            .appending(component: "FluxTest-empty-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let backend = FluxDiffusionBackend()
        do {
            try await backend.loadModel(from: dir)
            XCTFail("Expected loadModel(from:) to throw for an empty directory")
        } catch {
            // expected — empty diffusers layout cannot load
        }
        XCTAssertFalse(backend.isLoaded)
    }

    // MARK: - State machine safety (no Metal required)

    func test_stopGeneration_whenIdle_doesNotCrash() {
        let backend = FluxDiffusionBackend()
        backend.stopGeneration()  // must not crash or deadlock
        XCTAssertFalse(backend.isGenerating)
    }

    func test_unloadModel_whenNotLoaded_doesNotCrash() {
        let backend = FluxDiffusionBackend()
        backend.unloadModel()  // must not crash; no cacheLimit reset path taken
        XCTAssertFalse(backend.isLoaded)
    }

    func test_repeatedUnload_isIdempotent() {
        let backend = FluxDiffusionBackend()
        backend.unloadModel()
        backend.unloadModel()
        XCTAssertFalse(backend.isLoaded)
    }

    // MARK: - Error descriptions

    func test_errorDescription_notLoaded_mentionsLoadModel() {
        let desc = FluxDiffusionError.notLoaded.errorDescription
        XCTAssertNotNil(desc)
        XCTAssertTrue(desc?.contains("loadModel") ?? false,
                      "notLoaded should point the caller at loadModel(from:)")
    }

    func test_errorDescription_noLatentsProduced_nonEmpty() {
        XCTAssertNotNil(FluxDiffusionError.noLatentsProduced.errorDescription)
        XCTAssertFalse(FluxDiffusionError.noLatentsProduced.errorDescription?.isEmpty ?? true)
    }

    func test_errorDescription_pngEncodingFailed_includesPath() {
        let url = URL(fileURLWithPath: "/tmp/flux-out/img.png")
        let desc = FluxDiffusionError.pngEncodingFailed(url).errorDescription
        XCTAssertNotNil(desc)
        XCTAssertTrue(desc?.contains(url.path) ?? false,
                      "pngEncodingFailed should surface the failing path")
    }

    // MARK: - Config handling that the backend reads (no Metal)

    func test_imageGenerationConfig_defaults_areForwardable() {
        // Sanity that the config fields the backend reads exist with the
        // expected defaults; the backend maps these straight onto FLUX's
        // EvaluateParameters (width/height/steps/seed/guidance).
        let config = ImageGenerationConfig()
        XCTAssertEqual(config.steps, 20)
        XCTAssertEqual(config.width, 1024)
        XCTAssertEqual(config.height, 1024)
        XCTAssertNil(config.seed)
        XCTAssertNil(config.guidanceScale)
    }
}
