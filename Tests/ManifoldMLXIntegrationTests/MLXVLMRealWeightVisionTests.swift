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
        // `requiresVLMFactory` only checks for a top-level `vision_config` key
        // — it does not read `vision_config.skip_vision`. Some checkpoints
        // (e.g. gemma-3-4b-it-4bit, per this file's header doc) ship
        // `vision_config` but set `skip_vision: true`, meaning the vision
        // tower is a documented no-op: the turn below would "pass" the
        // grounded-color assertion in name only, having never actually
        // exercised vision. Read config.json directly and skip those.
        try XCTSkipIf(
            Self.configDeclaresSkipVision(at: resolved),
            "MLX_VLM_TEST_MODEL=\(selector) has vision_config.skip_vision == true — its vision tower "
            + "is a documented no-op, so this test cannot genuinely exercise vision against it."
        )
        modelURL = resolved
    }

    /// Reads `config.json` at `url` and returns whether `vision_config.skip_vision == true`.
    /// `false` (never skip on this basis) for any read/parse failure — matches
    /// the conservative default `MLXModelProbe.requiresVLMFactory` itself uses
    /// when config.json is missing/unreadable.
    private static func configDeclaresSkipVision(at url: URL) -> Bool {
        let configURL = url.appending(component: "config.json")
        guard let data = try? Data(contentsOf: configURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let visionConfig = json["vision_config"] as? [String: Any] else {
            return false
        }
        return (visionConfig["skip_vision"] as? Bool) == true
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

    /// A candidate fill color for the probe square: a name to match in the
    /// model's reply plus the RGB it's rendered with. Deliberately excludes
    /// red — "What color is the shape?" asked of a text-only model that never
    /// saw an image plausibly guesses "red" (the modal color-word answer),
    /// which is exactly the null hypothesis this test needs to reject. Using
    /// green/purple/orange instead means a guess has to land on the specific
    /// randomly-chosen color to pass, not just the most common answer.
    private struct ProbeColor {
        let name: String
        let red: CGFloat
        let green: CGFloat
        let blue: CGFloat
    }

    private static let probeColorPalette: [ProbeColor] = [
        ProbeColor(name: "green", red: 0.05, green: 0.75, blue: 0.1),
        ProbeColor(name: "purple", red: 0.55, green: 0.05, blue: 0.75),
        ProbeColor(name: "orange", red: 0.95, green: 0.5, blue: 0.02),
    ]

    /// A CG/ImageIO step failed while synthesising the in-process probe
    /// image. This is thrown, never `XCTSkip`ped — `solidSquarePNGData` is
    /// called (see the call site below) AFTER `loadModel` has already run,
    /// so an `XCTSkip` here would report the whole test skipped-green having
    /// already paid the model load, with zero vision coverage. The harness
    /// failing to synthesise its own probe image is a defect in this test,
    /// not an environment gate to skip past.
    private struct ProbeImageSynthesisError: Error, CustomStringConvertible {
        let stage: String
        var description: String { "Could not synthesise the probe image: \(stage)" }
    }

    /// Draws a solid-color square PNG in-memory — an unambiguous, asset-free
    /// probe: any working VLM should be able to name the dominant color when
    /// asked directly, without depending on photographic detail or an
    /// external fixture file surviving on disk.
    private static func solidSquarePNGData(color: ProbeColor, side: Int = 256) throws -> Data {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: nil,
            width: side, height: side,
            bitsPerComponent: 8, bytesPerRow: side * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw ProbeImageSynthesisError(stage: "could not create a CGContext")
        }
        context.setFillColor(red: color.red, green: color.green, blue: color.blue, alpha: 1.0)
        context.fill(CGRect(x: 0, y: 0, width: side, height: side))
        guard let cgImage = context.makeImage() else {
            throw ProbeImageSynthesisError(stage: "could not render to a CGImage")
        }

        let mutable = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(
            mutable as CFMutableData, UTType.png.identifier as CFString, 1, nil
        ) else {
            throw ProbeImageSynthesisError(stage: "could not create a PNG destination")
        }
        CGImageDestinationAddImage(dest, cgImage, nil)
        guard CGImageDestinationFinalize(dest) else {
            throw ProbeImageSynthesisError(stage: "could not finalize the PNG")
        }
        return mutable as Data
    }

    /// Word-boundary tokenizer for matching color names in a model reply.
    /// `String.contains` alone false-positives: "red" matches inside
    /// "colored", "hundred", "answered". Splitting on non-alphanumeric
    /// characters into a token set makes membership exact.
    private static func wordTokens(of text: String) -> Set<String> {
        Set(text.lowercased().split(whereSeparator: { !$0.isLetter && !$0.isNumber }).map(String.init))
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

    private static let visionQuestion = "What color is the shape in this image? Answer in one word."

    // MARK: - The real vision test

    /// Loads a real vision-capable MLX model and runs two turns against it:
    ///
    /// 1. **Blind control turn** — the same question, no image anywhere in
    ///    history. Whatever color name(s) the model volunteers here are
    ///    exactly what it would say by guessing alone.
    /// 2. **Grounded turn** — the same question, this time with a probe
    ///    square whose color is chosen to be one the blind turn did NOT
    ///    name. Asserts the reply names that color and none of the
    ///    remaining palette colors.
    ///
    /// This derives the probe color from the model's own blind answer
    /// instead of rolling it independently at random. An earlier revision
    /// picked the blind-turn color and the grounded-turn color as two
    /// independent random draws from the same 3-color palette: since a
    /// deterministic (temperature 0, fixed seed) model gives the same blind
    /// reply on every run, a 1-in-3 random draw would coincidentally match
    /// that reply and the test would flake red exactly once every three
    /// runs — the worst possible signature for an unattended lane. Deriving
    /// the probe color from the actual blind reply instead of rolling
    /// independently of it makes the discrimination hold *by construction*:
    /// the grounded turn's color is provably not something this model would
    /// say without looking at the image, on this run, so a passing grounded
    /// turn is real evidence of vision — zero flake risk, and one model
    /// load instead of two.
    ///
    /// `resetConversation()` runs between the two turns so the blind turn's
    /// question/reply isn't sitting in the grounded turn's context — the
    /// two turns must be independent, not a two-turn conversation.
    func test_realVLM_sentSyntheticImage_respondsWithImageGroundedContent() async throws {
        let backend = MLXBackend(enableKVCacheReuse: false)
        try await backend.loadModel(from: modelURL, plan: .testStub(effectiveContextSize: 2048))
        loadedBackends.append(backend)

        // 1. Blind control turn — no image anywhere in history.
        let blindHistory: [StructuredMessage] = [
            StructuredMessage(role: "user", parts: [.text(Self.visionQuestion)]),
        ]
        let blindEvents = try await runTurn(on: backend, prompt: Self.visionQuestion, history: blindHistory)
        let blindText = collectAssistantText(from: blindEvents)
        print("[MLXVLMRealWeightVisionTests] (blind control, no image) model=\(modelURL.lastPathComponent) reply=\"\(blindText)\"")
        let blindTokens = Self.wordTokens(of: blindText)

        backend.resetConversation()

        // Candidates: palette colors the blind turn did NOT name. Any one of
        // these is provable-by-construction to not be a guess this model
        // would make on this run.
        let candidates = Self.probeColorPalette.filter { !blindTokens.contains($0.name) }
        let chosen = try XCTUnwrap(
            candidates.first,
            "Blind reply \"\(blindText)\" named every color in the probe palette "
            + "(\(Self.probeColorPalette.map(\.name))) — this palette can't discriminate vision from "
            + "a guess for this model. Widen probeColorPalette or pick a less color-chatty question."
        )
        let others = Self.probeColorPalette.filter { $0.name != chosen.name }

        // 2. Grounded turn — same question, this time with the probe image.
        let imageData = try Self.solidSquarePNGData(color: chosen)
        // The current turn's image must ride in `history` (not just `prompt`)
        // — `MLXChatMessageEncoder`'s image threading only inspects the
        // structured history, so the turn carrying the image must appear
        // there, mirroring how `MLXVLMGateExperimentTests` always includes
        // the live turn as the final history entry.
        let groundedHistory: [StructuredMessage] = [
            StructuredMessage(
                role: "user",
                parts: [
                    .image(data: imageData, mimeType: "image/png"),
                    .text(Self.visionQuestion),
                ]
            ),
        ]
        let events = try await runTurn(on: backend, prompt: Self.visionQuestion, history: groundedHistory)
        let text = collectAssistantText(from: events)
        // Always logged (pass or fail) so a human can eyeball the actual
        // model output — an automated token-membership check can pass on a
        // technically-matching but nonsensical reply, and a failure needs
        // the verbatim text to judge whether the assertion or the model is
        // wrong (see this file's header doc).
        print("[MLXVLMRealWeightVisionTests] (grounded) model=\(modelURL.lastPathComponent) chosenColor=\(chosen.name) reply=\"\(text)\"")

        XCTAssertFalse(text.isEmpty, "A real vision-capable model must produce a non-empty reply to an image question.")

        let tokens = Self.wordTokens(of: text)
        XCTAssertTrue(
            tokens.contains(chosen.name),
            """
            Reply did not mention "\(chosen.name)", the actual fill color of the probe square (chosen \
            specifically because the blind control turn, \"\(blindText)\", did not name it). \
            Got: "\(text)". This suggests the image never actually reached the model \
            (silently dropped/ignored) rather than a real vision failure — the wiring is \
            supposed to be real per MLXChatMessageEncoder, but this is the first real-weight \
            check of that claim.
            """
        )
        for other in others {
            XCTAssertFalse(
                tokens.contains(other.name),
                "Reply unexpectedly also named \"\(other.name)\", which was not the probe square's "
                + "color (\(chosen.name)). Got: \"\(text)\"."
            )
        }
    }
}
