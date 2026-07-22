import XCTest
import ManifoldInference
import ManifoldTestSupport
import ManifoldMLX
@_spi(Testing) import ManifoldMLX
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

/// Real-weight, real-image vision test for MLX vision-language models.
///
/// Phase D (2026-07-22 companion breakage hunt) found that MLX's vision
/// wiring is real — `MLXChatMessageEncoder` threads `MessagePart.image` into
/// `UserInput.Image` and on into `VLMModelFactory.shared.loadContainer`'s
/// `container.generate()` — and is unit-proven against a mock
/// (`Tests/ManifoldMLXTests/MLXBackendGenerationTests.swift`), but the only
/// existing real-VLM integration test (`MLXVLMGateExperimentTests`) sends
/// text-only prompts: it tests KV-cache-reuse gating, not vision itself. So
/// "an image reaches a loaded VLM and produces a coherent, image-grounded
/// reply" has never actually run against real weights. This test closes that
/// gap.
///
/// The probe image is synthesised in-process (a solid-color square) rather
/// than shipped as a test asset, so this test has no fixture to go stale or
/// go missing.
///
/// Opt-in via `MLX_VLM_TEST_MODEL` (same selector convention as
/// `MLXVLMGateExperimentTests`: an absolute path, or a name substring under
/// `~/Documents/Models/`). Per the 2026-07-22 vision-capability sweep
/// (see project memory `mk-vision-capability-by-engine`), `Qwen3.5-2B-4bit`
/// is the cheapest confirmed vision-capable MLX checkpoint on disk — prefer
/// it for routine runs. `gemma-3-4b-it-4bit` is NOT usable here: its
/// `vision_config.skip_vision` is `true`, so it would silently no-op rather
/// than genuinely exercise vision.
@MainActor
final class MLXVLMRealWeightVisionTests: XCTestCase {

    private var modelURL: URL!
    private var loadedBackends: [MLXBackend] = []

    override func setUp() async throws {
        try await super.setUp()

        try XCTSkipUnless(HardwareRequirements.isAppleSilicon, "Requires Apple Silicon")
        try XCTSkipUnless(HardwareRequirements.hasMetalDevice, "Requires Metal GPU")

        let env = ProcessInfo.processInfo.environment
        guard let selector = env["MLX_VLM_TEST_MODEL"], !selector.isEmpty else {
            throw XCTSkip(
                "Set MLX_VLM_TEST_MODEL to a local MLX VLM directory (or a name substring under "
                + "~/Documents/Models/) to run the real-weight vision test. Qwen3.5-2B-4bit is the "
                + "cheapest confirmed vision-capable checkpoint on disk."
            )
        }

        let resolved = try resolveVLMModelURL(selector: selector)
        try XCTSkipUnless(
            MLXModelProbe.requiresVLMFactory(at: resolved),
            "MLX_VLM_TEST_MODEL=\(selector) does not resolve to a VLM (requiresVLMFactory returned false). "
            + "Pick a model with a vision_config in config.json."
        )
        modelURL = resolved
    }

    override func tearDown() async throws {
        for backend in loadedBackends.reversed() {
            backend.unloadModel()
        }
        loadedBackends.removeAll()
        modelURL = nil
        try await super.tearDown()
    }

    // MARK: - Selector (mirrors MLXVLMGateExperimentTests)

    private func resolveVLMModelURL(selector: String) throws -> URL {
        let expanded = (selector as NSString).expandingTildeInPath
        if expanded.contains("/") {
            let url = URL(fileURLWithPath: expanded)
            if FileManager.default.fileExists(atPath: url.path) {
                return url
            }
        }
        var searchEnv = ProcessInfo.processInfo.environment
        searchEnv["MANIFOLD_DISCOVER_LOCAL_MODELS"] = "1"
        if let url = HardwareRequirements.findMLXModelDirectory(
            nameContains: selector,
            environment: searchEnv
        ) {
            return url
        }
        throw XCTSkip("MLX_VLM_TEST_MODEL=\(selector) did not resolve to a loadable model directory.")
    }

    // MARK: - Synthetic probe image

    /// Draws a solid red square PNG in-memory — an unambiguous,
    /// asset-free probe: any working VLM should be able to name the
    /// dominant color when asked directly, without depending on
    /// photographic detail or an external fixture file surviving on disk.
    private static func redSquarePNGData(side: Int = 256) throws -> Data {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: nil,
            width: side, height: side,
            bitsPerComponent: 8, bytesPerRow: side * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw XCTSkip("Could not create a CGContext to synthesise the probe image.")
        }
        context.setFillColor(red: 0.9, green: 0.05, blue: 0.05, alpha: 1.0)
        context.fill(CGRect(x: 0, y: 0, width: side, height: side))
        guard let cgImage = context.makeImage() else {
            throw XCTSkip("Could not render the probe image to a CGImage.")
        }

        let mutable = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(
            mutable as CFMutableData, UTType.png.identifier as CFString, 1, nil
        ) else {
            throw XCTSkip("Could not create a PNG destination for the probe image.")
        }
        CGImageDestinationAddImage(dest, cgImage, nil)
        guard CGImageDestinationFinalize(dest) else {
            throw XCTSkip("Could not finalize the probe image PNG.")
        }
        return mutable as Data
    }

    // MARK: - Turn driver (timeout-race pattern copied from MLXVLMGateExperimentTests)

    private var deterministicConfig: GenerationConfig {
        GenerationConfig(
            temperature: 0.0,
            topP: 1.0,
            repeatPenalty: 1.0,
            seed: 749,
            maxOutputTokens: 48,
            maxThinkingTokens: 0
        )
    }

    private struct RunTurnTimeoutError: Error, CustomStringConvertible {
        let seconds: Double
        var description: String { "runTurn timed out after \(seconds)s — possible VLM stream hang (#26)" }
    }

    /// CRITICAL (per Phase D direction): VLM stream hangs are a named
    /// recurring hazard in this repo — issue #26, the Qwen2-VL two-turn hang.
    /// A bare `for try await` loop over the generation stream risks hanging
    /// an entire unattended overnight run. This races the collect loop
    /// against a ~90s deadline and cancels whichever branch loses, exactly as
    /// `MLXVLMGateExperimentTests.runTurn` does at :120-123.
    private func runTurn(
        on backend: MLXBackend,
        prompt: String,
        history: [StructuredMessage],
        timeoutSeconds: Double = 90
    ) async throws -> [GenerationEvent] {
        let config = deterministicConfig
        return try await withThrowingTaskGroup(of: [GenerationEvent].self) { group in
            group.addTask {
                let stream = try backend.generate(
                    prompt: prompt,
                    systemPrompt: nil,
                    config: config,
                    hints: GenerationRuntimeHints(history: history)
                )
                var events: [GenerationEvent] = []
                for try await event in stream.events {
                    events.append(event)
                }
                return events
            }
            group.addTask {
                try await Task.sleep(nanoseconds: UInt64(timeoutSeconds * 1_000_000_000))
                throw RunTurnTimeoutError(seconds: timeoutSeconds)
            }
            defer { group.cancelAll() }
            return try await group.next() ?? []
        }
    }

    private func collectAssistantText(from events: [GenerationEvent]) -> String {
        var text = ""
        for event in events {
            if case .token(let chunk) = event {
                text += chunk
            }
        }
        return text
    }

    // MARK: - The real vision test

    /// Loads a real vision-capable MLX model, sends it a synthesised solid
    /// red square with a direct question about its color, and asserts the
    /// reply is both non-empty and actually grounded in the image (mentions
    /// a color/shape word a model that never looked at the image would have
    /// no way to guess). The assertion is deliberately loose — several
    /// plausible tokens, not one exact string — so it is robust to phrasing
    /// differences across VLM architectures/quantizations rather than
    /// brittle against a specific model's wording.
    func test_realVLM_sentSyntheticImage_respondsWithImageGroundedContent() async throws {
        let backend = MLXBackend(enableKVCacheReuse: false)
        try await backend.loadModel(from: modelURL, plan: .testStub(effectiveContextSize: 2048))
        loadedBackends.append(backend)

        let imageData = try Self.redSquarePNGData()
        let question = "What color is the shape in this image? Answer in one word."

        // The current turn's image must ride in `history` (not just `prompt`)
        // — `MLXChatMessageEncoder`'s image threading only inspects the
        // structured history, so the turn carrying the image must appear
        // there, mirroring how `MLXVLMGateExperimentTests` always includes
        // the live turn as the final history entry.
        let history: [StructuredMessage] = [
            StructuredMessage(
                role: "user",
                parts: [
                    .image(data: imageData, mimeType: "image/png"),
                    .text(question),
                ]
            ),
        ]

        let events = try await runTurn(on: backend, prompt: question, history: history)
        let text = collectAssistantText(from: events)
        // Always logged (pass or fail) so a human can eyeball the actual
        // model output — an automated token-membership check can pass on a
        // technically-matching but nonsensical reply, and a failure needs
        // the verbatim text to judge whether the assertion or the model is
        // wrong (see this file's header doc).
        print("[MLXVLMRealWeightVisionTests] model=\(modelURL.lastPathComponent) reply=\"\(text)\"")

        XCTAssertFalse(text.isEmpty, "A real vision-capable model must produce a non-empty reply to an image question.")

        let lowered = text.lowercased()
        let plausibleTokens = ["red", "square", "crimson", "scarlet", "maroon", "pink"]
        XCTAssertTrue(
            plausibleTokens.contains(where: lowered.contains),
            """
            Reply did not mention any plausible color/shape token for a solid red square. \
            Got: "\(text)". This suggests the image never actually reached the model \
            (silently dropped/ignored) rather than a real vision failure — the wiring is \
            supposed to be real per MLXChatMessageEncoder, but this is the first real-weight \
            check of that claim.
            """
        )
    }
}
