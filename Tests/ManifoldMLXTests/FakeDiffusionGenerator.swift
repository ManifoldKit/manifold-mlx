import Foundation
@_spi(Testing) import ManifoldMLX
import ManifoldInference

/// Test fake for the ``DiffusionGenerator`` seam. Drives the diffusion
/// `generate(...)` loop without any MLX / Metal work: it produces a fixed
/// number of canned steps and writes a small stub PNG on `finishImage`.
///
/// This is the headless substitute that lets us assert the `.progress` /
/// `.completed` sequence and `stopGeneration()` cancellation contract that the
/// real (Metal-bound) pipeline can't exercise under `swift test`.
final class FakeDiffusionGenerator: DiffusionGenerator, @unchecked Sendable {
    let stepCount: Int
    /// Set to throw from a run's `step()` to simulate a mid-loop failure.
    let stepError: Error?
    /// Optional hook fired at the start of each `step()` — used to drive
    /// `stopGeneration()` from inside the loop in cancellation tests.
    let onStep: (@Sendable (Int) -> Void)?

    init(steps: Int, stepError: Error? = nil, onStep: (@Sendable (Int) -> Void)? = nil) {
        self.stepCount = steps
        self.stepError = stepError
        self.onStep = onStep
    }

    func makeRun(prompt: String, config: ImageGenerationConfig) -> any DiffusionRun {
        FakeDiffusionRun(totalSteps: stepCount, stepError: stepError, onStep: onStep)
    }
}

final class FakeDiffusionRun: DiffusionRun {
    let totalSteps: Int
    private let stepError: Error?
    private let onStep: (@Sendable (Int) -> Void)?
    private var produced = 0

    init(totalSteps: Int, stepError: Error?, onStep: (@Sendable (Int) -> Void)?) {
        self.totalSteps = totalSteps
        self.stepError = stepError
        self.onStep = onStep
    }

    func step() throws -> Bool {
        onStep?(produced)
        if let stepError { throw stepError }
        guard produced < totalSteps else { return false }
        produced += 1
        return true
    }

    func finishImage(to outputDirectory: URL?) throws -> URL {
        let dir = outputDirectory ?? FileManager.default.temporaryDirectory
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let dest = dir.appending(component: "fake-\(UUID().uuidString).png")
        // Minimal valid 1×1 PNG so callers see a real on-disk file.
        try Data(Self.onePixelPNG).write(to: dest)
        return dest
    }

    /// 1×1 transparent PNG.
    private static let onePixelPNG: [UInt8] = [
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
        0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
        0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4,
        0x89, 0x00, 0x00, 0x00, 0x0A, 0x49, 0x44, 0x41,
        0x54, 0x78, 0x9C, 0x63, 0x00, 0x01, 0x00, 0x00,
        0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00,
        0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE,
        0x42, 0x60, 0x82,
    ]
}

/// Helpers shared by both backend test suites.
enum DiffusionTestHelpers {
    /// Drains an image-generation stream into an ordered list of events.
    static func collect(
        _ stream: AsyncThrowingStream<ImageGenerationEvent, Error>
    ) async throws -> [ImageGenerationEvent] {
        var events: [ImageGenerationEvent] = []
        for try await event in stream { events.append(event) }
        return events
    }
}

/// Sendable box letting an `@Sendable` `onStep` hook reach the backend under
/// test (so the fake can call `stopGeneration()` from inside the denoise loop).
/// Set/read on the same actor before/after the stream drains.
final class BackendHolder: @unchecked Sendable {
    var backend: MLXDiffusionBackend?
}

final class FluxBackendHolder: @unchecked Sendable {
    var backend: FluxDiffusionBackend?
}

extension ImageGenerationEvent {
    var progressStep: Int? {
        if case let .progress(step, _) = self { return step }
        return nil
    }
    var isCompleted: Bool {
        if case .completed = self { return true }
        return false
    }
}
