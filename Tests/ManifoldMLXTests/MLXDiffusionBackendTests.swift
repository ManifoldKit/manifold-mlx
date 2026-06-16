
import XCTest
import ManifoldMLX
@_spi(Testing) import ManifoldMLX
import ManifoldInference

/// Unit tests for ``MLXDiffusionBackend``.
///
/// These tests avoid real model weights — Metal shaders can't run in the
/// simulator and the XCTest runner can't access actual diffusion models. Tests
/// cover path-safe error surfaces and state-machine behaviour without touching
/// the MLX inference stack. Real E2E coverage lives in
/// ``MLXDiffusionIntegrationTests`` (Xcode-only, MANIFOLD_DISCOVER_LOCAL_MODELS).
final class MLXDiffusionBackendTests: XCTestCase {

    // MARK: - Initial state

    func test_isLoaded_initiallyFalse() {
        XCTAssertFalse(MLXDiffusionBackend().isLoaded)
    }

    func test_isGenerating_initiallyFalse() {
        XCTAssertFalse(MLXDiffusionBackend().isGenerating)
    }

    // MARK: - Error surfaces that don't require Metal

    func test_loadModel_emptyDirectory_throwsUnsupportedLayout() async throws {
        let dir = FileManager.default.temporaryDirectory
            .appending(component: "BCKTest-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let backend = MLXDiffusionBackend()
        do {
            try await backend.loadModel(from: dir)
            XCTFail("Expected MLXDiffusionError.unsupportedModelLayout")
        } catch MLXDiffusionError.unsupportedModelLayout {
            // expected
        }
    }

    func test_loadModel_onlyVaeNoUnet_throwsUnsupportedLayout() async throws {
        let dir = FileManager.default.temporaryDirectory
            .appending(component: "BCKTest-\(UUID().uuidString)")
        // Has vae/ but no unet/ and no text_encoder_2/ → should throw
        let vaeDir = dir.appending(component: "vae")
        try FileManager.default.createDirectory(at: vaeDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let backend = MLXDiffusionBackend()
        do {
            try await backend.loadModel(from: dir)
            XCTFail("Expected MLXDiffusionError.unsupportedModelLayout")
        } catch MLXDiffusionError.unsupportedModelLayout {
            // expected
        }
    }

    func test_generate_notLoaded_throwsNotLoaded() throws {
        let backend = MLXDiffusionBackend()
        XCTAssertFalse(backend.isLoaded)
        do {
            _ = try backend.generate(prompt: "a cat", config: .init())
            XCTFail("Expected MLXDiffusionError.notLoaded")
        } catch MLXDiffusionError.notLoaded {
            // expected
        }
    }

    // MARK: - State machine safety (no Metal required)

    func test_stopGeneration_whenIdle_doesNotCrash() {
        let backend = MLXDiffusionBackend()
        backend.stopGeneration()  // must not crash or deadlock
        XCTAssertFalse(backend.isGenerating)
    }

    func test_unloadModel_whenNotLoaded_doesNotCrash() {
        let backend = MLXDiffusionBackend()
        backend.unloadModel()  // must not crash
        XCTAssertFalse(backend.isLoaded)
    }

    // MARK: - Layout detection (SD 2.1 vs SDXL)

    func test_detectPreset_unetOnly_selectsSD21() throws {
        let dir = makeModelDir(withXLEncoder: false)
        defer { try? FileManager.default.removeItem(at: dir) }
        // Test the layout-detection logic directly without going through
        // loadModel → textToImageGenerator, which triggers Metal/MLX
        // initialisation and fatally crashes if the metallib isn't compiled.
        let preset = try MLXDiffusionBackend.detectPreset(at: dir)
        XCTAssertTrue(preset.id.contains("stable-diffusion-2-1"),
                      "unet/ without text_encoder_2/ must select the SD 2.1 preset")
    }

    func test_detectPreset_textEncoder2_selectsSDXL() throws {
        let dir = makeModelDir(withXLEncoder: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let preset = try MLXDiffusionBackend.detectPreset(at: dir)
        XCTAssertTrue(preset.id.contains("sdxl"),
                      "text_encoder_2/ present must select the SDXL preset")
    }

    // MARK: - Sabotage check (documented in CLAUDE.md test conventions)
    //
    // To verify test_loadModel_emptyDirectory_throwsUnsupportedLayout is
    // truly testing the right thing, temporarily comment out the throw in
    // detectPreset and confirm the test fails. Restored before commit.

    // MARK: - Helpers

    private func makeModelDir(withXLEncoder: Bool) -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appending(component: "BCKTest-\(UUID().uuidString)")
        let unetDir = dir.appending(component: "unet")
        try? FileManager.default.createDirectory(at: unetDir, withIntermediateDirectories: true)
        if withXLEncoder {
            let te2Dir = dir.appending(component: "text_encoder_2")
            try? FileManager.default.createDirectory(at: te2Dir, withIntermediateDirectories: true)
        }
        return dir
    }

    // MARK: - makeParams (pure parameter mapping, no Metal)

    func test_makeParams_latentSizeIsHeightFirstAndDividedBy8() {
        let config = ImageGenerationConfig(steps: 4, width: 768, height: 512)
        let params = MLXDiffusionBackend.makeParams(
            prompt: "p", config: config, preset: .presetStableDiffusion21Base
        )
        // latentSize == [height/8, width/8] — height first.
        XCTAssertEqual(params.latentSize, [512 / 8, 768 / 8])
    }

    func test_makeParams_stepsAndPromptForwarded() {
        let config = ImageGenerationConfig(steps: 7, width: 512, height: 512)
        let params = MLXDiffusionBackend.makeParams(
            prompt: "a fox", config: config, preset: .presetStableDiffusion21Base
        )
        XCTAssertEqual(params.steps, 7)
        XCTAssertEqual(params.prompt, "a fox")
    }

    func test_makeParams_guidanceScaleNil_fallsBackToPresetDefault() {
        let preset = try! MLXDiffusionBackend.detectPreset(at: makeModelDir(withXLEncoder: false))
        let config = ImageGenerationConfig(width: 512, height: 512, guidanceScale: nil)
        let params = MLXDiffusionBackend.makeParams(prompt: "p", config: config, preset: preset)
        XCTAssertEqual(params.cfgWeight, preset.defaultParameters().cfgWeight)
    }

    func test_makeParams_guidanceScaleSet_overridesDefault() {
        let config = ImageGenerationConfig(width: 512, height: 512, guidanceScale: 3.25)
        let params = MLXDiffusionBackend.makeParams(
            prompt: "p", config: config, preset: .presetStableDiffusion21Base
        )
        XCTAssertEqual(params.cfgWeight, 3.25)
    }

    func test_makeParams_seedSet_isForwarded() {
        let config = ImageGenerationConfig(width: 512, height: 512, seed: 4242)
        let params = MLXDiffusionBackend.makeParams(
            prompt: "p", config: config, preset: .presetStableDiffusion21Base
        )
        XCTAssertEqual(params.seed, 4242)
    }

    // MARK: - loadPlanInputs (filesystem byte accounting, no Metal)

    /// Writes a file of exactly `byteCount` zero bytes at `relativePath` under `dir`.
    private func writeFile(_ dir: URL, _ relativePath: String, bytes byteCount: Int) {
        let fileURL = dir.appending(path: relativePath)
        try? FileManager.default.createDirectory(
            at: fileURL.deletingLastPathComponent(), withIntermediateDirectories: true
        )
        try? Data(count: byteCount).write(to: fileURL)
    }

    func test_loadPlanInputs_sumsWeightBytesAndPresetDims_SD21() throws {
        let dir = makeModelDir(withXLEncoder: false)
        defer { try? FileManager.default.removeItem(at: dir) }
        writeFile(dir, "unet/diffusion_pytorch_model.safetensors", bytes: 1000)
        writeFile(dir, "vae/diffusion_pytorch_model.safetensors", bytes: 200)
        writeFile(dir, "text_encoder/model.safetensors", bytes: 300)
        // No text_encoder_2 for SD 2.1 → te2 contributes 0 (missing file → 0).

        let inputs = try MLXDiffusionBackend.loadPlanInputs(
            at: dir, preset: .presetStableDiffusion21Base
        )

        XCTAssertEqual(inputs.unetWeightBytes, 1000)
        XCTAssertEqual(inputs.vaeWeightBytes, 200)
        XCTAssertEqual(inputs.textEncoderWeightBytes, 300, "te1 only; missing te2 → 0")
        // SD 2.1: 512×512 target, activation = (64*64) * 2 * 8.
        XCTAssertEqual(inputs.targetWidth, 512)
        XCTAssertEqual(inputs.targetHeight, 512)
        XCTAssertEqual(inputs.activationMemoryBytes, Int64(64 * 64) * 2 * 8)
    }

    func test_loadPlanInputs_sumsTwoTextEncoders_andSDXLDims() throws {
        let dir = makeModelDir(withXLEncoder: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        writeFile(dir, "text_encoder/model.safetensors", bytes: 300)
        writeFile(dir, "text_encoder_2/model.safetensors", bytes: 400)

        let inputs = try MLXDiffusionBackend.loadPlanInputs(at: dir, preset: .presetSDXLTurbo)

        XCTAssertEqual(inputs.textEncoderWeightBytes, 700, "te1 + te2 summed")
        // SDXL: 1024×1024 target, activation = (128*128) * 2 * 8.
        XCTAssertEqual(inputs.targetWidth, 1024)
        XCTAssertEqual(inputs.targetHeight, 1024)
        XCTAssertEqual(inputs.activationMemoryBytes, Int64(128 * 128) * 2 * 8)
    }

    func test_loadPlanInputs_missingWeightFiles_areZero() throws {
        // unet/ dir exists (so detectPreset would pass) but no weight files written.
        let dir = makeModelDir(withXLEncoder: false)
        defer { try? FileManager.default.removeItem(at: dir) }

        let inputs = try MLXDiffusionBackend.loadPlanInputs(
            at: dir, preset: .presetStableDiffusion21Base
        )
        XCTAssertEqual(inputs.unetWeightBytes, 0)
        XCTAssertEqual(inputs.vaeWeightBytes, 0)
        XCTAssertEqual(inputs.textEncoderWeightBytes, 0)
    }

    func test_loadPlanInputs_availableMemoryIsPhysicalMinus1GiB() throws {
        let dir = makeModelDir(withXLEncoder: false)
        defer { try? FileManager.default.removeItem(at: dir) }

        let inputs = try MLXDiffusionBackend.loadPlanInputs(
            at: dir, preset: .presetStableDiffusion21Base
        )
        let expected = max(0, Int64(ProcessInfo.processInfo.physicalMemory) - 1_073_741_824)
        XCTAssertEqual(inputs.availableMemoryBytes, expected)
    }

    // MARK: - MLXDiffusionError.errorDescription

    func test_errorDescription_notLoaded_mentionsLoadModel() {
        XCTAssertTrue(
            MLXDiffusionError.notLoaded.errorDescription?.contains("loadModel") ?? false
        )
    }

    func test_errorDescription_pngEncodingFailed_containsPath() {
        let url = URL(fileURLWithPath: "/tmp/some/output.png")
        let desc = MLXDiffusionError.pngEncodingFailed(url).errorDescription ?? ""
        XCTAssertTrue(desc.contains(url.path), "PNG-failure message must contain the file path")
    }

    func test_errorDescription_unsupportedModelLayout_containsLastPathComponent() {
        let url = URL(fileURLWithPath: "/models/org__weirdmodel")
        let desc = MLXDiffusionError.unsupportedModelLayout(url).errorDescription ?? ""
        XCTAssertTrue(desc.contains("weirdmodel"))
    }

    func test_errorDescription_insufficientMemory_joinsReasons() {
        let reasons: [ImageModelLoadPlan.Reason] = [
            .unetTooLarge(required: 100, available: 50),
            .totalExceedsBudget(required: 200, available: 50),
        ]
        let desc = MLXDiffusionError.insufficientMemory(reasons).errorDescription ?? ""
        // Both reasons must appear in the joined message.
        XCTAssertTrue(desc.contains("\(reasons[0])"))
        XCTAssertTrue(desc.contains("\(reasons[1])"))
    }
}

