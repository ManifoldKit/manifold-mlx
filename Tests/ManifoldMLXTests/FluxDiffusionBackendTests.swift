
import XCTest
import ManifoldMLX
@_spi(Testing) import ManifoldMLX
import ManifoldInference
import FluxSwift

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

    // MARK: - generate() pipeline via injected fake (no Metal)

    func test_generate_emitsProgressSequenceThenCompleted() async throws {
        let fake = FakeDiffusionGenerator(steps: 4)
        let backend = FluxDiffusionBackend(generator: fake)
        XCTAssertTrue(backend.isLoaded)

        let outDir = FileManager.default.temporaryDirectory
            .appending(component: "FluxDiffGen-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: outDir) }

        let stream = try backend.generate(
            prompt: "a fox", config: .init(steps: 4, outputDirectory: outDir)
        )
        let events = try await DiffusionTestHelpers.collect(stream)

        XCTAssertEqual(events.compactMap { $0.progressStep }, [1, 2, 3, 4])
        if case let .progress(_, total) = events.first {
            XCTAssertEqual(total, 4)
        } else {
            XCTFail("First event must be .progress")
        }
        XCTAssertTrue(events.last?.isCompleted ?? false)
        XCTAssertEqual(events.filter { $0.isCompleted }.count, 1)
        XCTAssertFalse(backend.isGenerating)
    }

    func test_stopGeneration_midStream_finishesEarly_andClearsIsGenerating() async throws {
        let holder = FluxBackendHolder()
        let fake = FakeDiffusionGenerator(steps: 10) { stepIndex in
            if stepIndex == 0 { holder.backend?.stopGeneration() }
        }
        let backend = FluxDiffusionBackend(generator: fake)
        holder.backend = backend

        let stream = try backend.generate(prompt: "a fox", config: .init(steps: 10))
        let events = try await DiffusionTestHelpers.collect(stream)

        XCTAssertEqual(events.compactMap { $0.progressStep }, [1])
        XCTAssertFalse(events.contains { $0.isCompleted })
        XCTAssertFalse(backend.isGenerating)
    }

    func test_generate_noLatents_finishesWithError() async {
        // Flux's contract throws noLatentsProduced when the loop yields nothing.
        let backend = FluxDiffusionBackend(generator: FakeDiffusionGenerator(steps: 0))
        do {
            let stream = try backend.generate(prompt: "x", config: .init(steps: 0))
            _ = try await DiffusionTestHelpers.collect(stream)
            XCTFail("Expected noLatentsProduced")
        } catch FluxDiffusionError.noLatentsProduced {
            // expected
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
        XCTAssertFalse(backend.isGenerating)
    }

    // MARK: - loadModel branch selection (past fileExists, no Metal)

    func test_loadModel_metadataJsonPresent_takesQuantizedBranch_andFailsClosed() async {
        // metadata.json present → quantized branch (FLUX.loadQuantized). With no
        // real quantized weights it must throw and leave isLoaded == false —
        // proving the branch is selected past the fileExists check.
        let dir = FileManager.default.temporaryDirectory
            .appending(component: "FluxQuant-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try? Data("{}".utf8).write(to: dir.appending(component: "metadata.json"))
        defer { try? FileManager.default.removeItem(at: dir) }

        let backend = FluxDiffusionBackend()
        do {
            try await backend.loadModel(from: dir)
            XCTFail("Quantized branch must fail without real weights")
        } catch {
            // expected
        }
        XCTAssertFalse(backend.isLoaded)
    }

    // MARK: - Pre-quantized weight detection (config.json quantization block)
    //
    // These exercise the MLX-LLM-style detection that lets FluxModelCore load
    // already-4-bit weights and SKIP the in-memory quantize(...) pass. The
    // detection reader is pure (filesystem + JSON), so it runs without Metal or
    // a real FLUX snapshot — the actual QuantizedLinear application requires a
    // full model load and is covered only by the gated integration test.

    private func makeComponentDir(configJSON: String?) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appending(component: "FluxQuantCfg-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        if let configJSON {
            try Data(configJSON.utf8).write(to: dir.appending(component: "config.json"))
        }
        return dir
    }

    func test_quantizationConfig_present_parsesBitsAndGroupSize() throws {
        let dir = try makeComponentDir(
            configJSON: #"{"quantization": {"group_size": 64, "bits": 4}}"#)
        defer { try? FileManager.default.removeItem(at: dir) }

        let cfg = FluxModelCore.quantizationConfig(in: dir)
        XCTAssertNotNil(cfg, "A config.json with a quantization block must be detected")
        XCTAssertEqual(cfg?.bits, 4)
        XCTAssertEqual(cfg?.groupSize, 64)
    }

    func test_quantizationConfig_8bitGroup128_parsed() throws {
        let dir = try makeComponentDir(
            configJSON: #"{"quantization": {"group_size": 128, "bits": 8}}"#)
        defer { try? FileManager.default.removeItem(at: dir) }

        let cfg = FluxModelCore.quantizationConfig(in: dir)
        XCTAssertEqual(cfg?.bits, 8)
        XCTAssertEqual(cfg?.groupSize, 128)
    }

    func test_quantizationConfig_noBlock_returnsNil_fp16Path() throws {
        // A config.json WITHOUT a quantization block must read as fp16 (nil) so
        // the loader keeps the backward-compatible fp16-then-quantize behaviour.
        let dir = try makeComponentDir(configJSON: #"{"_class_name": "FluxTransformer2DModel"}"#)
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertNil(FluxModelCore.quantizationConfig(in: dir))
    }

    func test_quantizationConfig_missingFile_returnsNil() throws {
        let dir = try makeComponentDir(configJSON: nil)
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertNil(FluxModelCore.quantizationConfig(in: dir),
                     "No config.json → fp16 path (nil)")
    }

    func test_quantizationConfig_defaultsWhenKeysOmitted() throws {
        // An empty quantization block still signals "quantized", defaulting to
        // the standard 4-bit / group-64 used elsewhere in the loader.
        let dir = try makeComponentDir(configJSON: #"{"quantization": {}}"#)
        defer { try? FileManager.default.removeItem(at: dir) }
        let cfg = FluxModelCore.quantizationConfig(in: dir)
        XCTAssertEqual(cfg?.bits, 4)
        XCTAssertEqual(cfg?.groupSize, 64)
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
