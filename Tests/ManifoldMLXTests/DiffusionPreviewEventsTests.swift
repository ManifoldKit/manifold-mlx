import XCTest
@_spi(Testing) import ManifoldMLX
import ManifoldInference

/// Headless coverage for the live `.preview` denoising channel (#8).
///
/// Drives both diffusion backends through the injected ``FakeDiffusionGenerator``
/// so the throttle cadence and the `nil`-stride no-op are asserted with **zero**
/// MLX/Metal work. Bytes-decode-to-viewable-image at a real consumer and the
/// measured intermediate-decode GPU cost are deferred to the local sweep.
final class DiffusionPreviewEventsTests: XCTestCase {

    // MARK: - Throttle decision (pure)

    func test_throttle_nilStride_neverEmits() {
        for step in 1...50 {
            XCTAssertFalse(
                DiffusionPreviewThrottle.shouldEmit(step: step, total: 50, stride: nil)
            )
        }
    }

    func test_throttle_nonPositiveStride_neverEmits() {
        XCTAssertFalse(DiffusionPreviewThrottle.shouldEmit(step: 1, total: 4, stride: 0))
        XCTAssertFalse(DiffusionPreviewThrottle.shouldEmit(step: 2, total: 4, stride: -3))
    }

    func test_throttle_excludesFinalStep() {
        // The terminal frame arrives as .completed, never as a preview.
        XCTAssertFalse(DiffusionPreviewThrottle.shouldEmit(step: 4, total: 4, stride: 1))
        XCTAssertFalse(DiffusionPreviewThrottle.shouldEmit(step: 4, total: 4, stride: 2))
    }

    func test_throttle_strideCadence() {
        func emitted(total: Int, stride: Int) -> [Int] {
            (1...total).filter {
                DiffusionPreviewThrottle.shouldEmit(step: $0, total: total, stride: stride)
            }
        }
        XCTAssertEqual(emitted(total: 10, stride: 3), [3, 6, 9])
        XCTAssertEqual(emitted(total: 20, stride: 5), [5, 10, 15])   // 20 excluded (final)
        XCTAssertEqual(emitted(total: 4, stride: 1), [1, 2, 3])      // short Turbo run
        XCTAssertEqual(emitted(total: 4, stride: 2), [2])            // short, coarse stride
    }

    // MARK: - SD backend: cadence end-to-end

    func test_sd_previewEmittedOnThrottledSubset() async throws {
        let fake = FakeDiffusionGenerator(steps: 10)
        let backend = MLXDiffusionBackend(generator: fake)
        let stream = try backend.generate(
            prompt: "a cat", config: .init(steps: 10, previewStride: 3)
        )
        let events = try await DiffusionTestHelpers.collect(stream)

        XCTAssertEqual(events.compactMap { $0.previewStep }, [3, 6, 9])
        XCTAssertEqual(fake.previewDecodeCount.count, 3,
                       "one VAE decode per emitted preview")
        // Preview carries non-empty in-memory bytes (deferred: viewable decode).
        XCTAssertTrue(events.contains { ($0.previewImageData?.isEmpty == false) })
        XCTAssertTrue(events.last?.isCompleted ?? false)
    }

    func test_sd_shortTurboRun_stride1() async throws {
        let fake = FakeDiffusionGenerator(steps: 4)
        let backend = MLXDiffusionBackend(generator: fake)
        let stream = try backend.generate(
            prompt: "a cat", config: .init(steps: 4, previewStride: 1)
        )
        let events = try await DiffusionTestHelpers.collect(stream)
        XCTAssertEqual(events.compactMap { $0.previewStep }, [1, 2, 3])
        XCTAssertEqual(fake.previewDecodeCount.count, 3)
    }

    func test_sd_nilStride_zeroPreviewsZeroDecodes() async throws {
        let fake = FakeDiffusionGenerator(steps: 10)
        let backend = MLXDiffusionBackend(generator: fake)
        let stream = try backend.generate(
            prompt: "a cat", config: .init(steps: 10)   // previewStride defaults nil
        )
        let events = try await DiffusionTestHelpers.collect(stream)

        XCTAssertTrue(events.compactMap { $0.previewStep }.isEmpty,
                      "nil stride must emit zero previews")
        XCTAssertEqual(fake.previewDecodeCount.count, 0,
                       "nil stride must run zero intermediate decodes")
        // Behaviour otherwise unchanged: progress 1...10 then completed.
        XCTAssertEqual(events.compactMap { $0.progressStep }, Array(1...10))
        XCTAssertTrue(events.last?.isCompleted ?? false)
    }

    // MARK: - FLUX backend: cadence + nil no-op

    func test_flux_previewEmittedOnThrottledSubset() async throws {
        let fake = FakeDiffusionGenerator(steps: 8)
        let backend = FluxDiffusionBackend(generator: fake)
        let stream = try backend.generate(
            prompt: "a dog", config: .init(steps: 8, previewStride: 2)
        )
        let events = try await DiffusionTestHelpers.collect(stream)

        XCTAssertEqual(events.compactMap { $0.previewStep }, [2, 4, 6])   // 8 excluded
        XCTAssertEqual(fake.previewDecodeCount.count, 3)
        XCTAssertTrue(events.last?.isCompleted ?? false)
    }

    func test_flux_nilStride_zeroPreviewsZeroDecodes() async throws {
        let fake = FakeDiffusionGenerator(steps: 8)
        let backend = FluxDiffusionBackend(generator: fake)
        let stream = try backend.generate(
            prompt: "a dog", config: .init(steps: 8)
        )
        let events = try await DiffusionTestHelpers.collect(stream)

        XCTAssertTrue(events.compactMap { $0.previewStep }.isEmpty)
        XCTAssertEqual(fake.previewDecodeCount.count, 0)
        XCTAssertEqual(events.compactMap { $0.progressStep }, Array(1...8))
        XCTAssertTrue(events.last?.isCompleted ?? false)
    }
}
