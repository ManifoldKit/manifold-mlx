
import XCTest
import ManifoldInference
import ManifoldTestSupport
import ManifoldMLX

/// Unit tests for ``MLXBackends/register(with:)``.
///
/// The registrar wires an MLX backend factory into an `InferenceService` and
/// declares `.mlx` support. These tests verify the registration through the
/// service's public read APIs only — no model load, no Metal.
@MainActor
final class MLXBackendsRegistrarTests: XCTestCase {

    private func makeService() -> InferenceService {
        // DEBUG-only convenience init that pre-loads a (mock) backend. The mock
        // never runs here — we only exercise the backend registry, not generation.
        InferenceService(backend: MockInferenceBackend())
    }

    func test_register_declaresMLXSupport() {
        let service = makeService()
        MLXBackends.register(with: service)

        XCTAssertTrue(
            service.compatibility(for: .mlx).isSupported,
            "After registration, .mlx must be reported as supported"
        )
    }

    func test_register_snapshotContainsMLX() {
        let service = makeService()
        MLXBackends.register(with: service)

        let snapshot = service.registeredBackendSnapshot()
        XCTAssertTrue(
            snapshot.localModelTypes.contains(.mlx),
            "registeredBackendSnapshot().localModelTypes must contain .mlx"
        )
    }

    func test_register_doesNotDeclareOtherLocalTypes() {
        let service = makeService()
        MLXBackends.register(with: service)

        // The registrar's factory returns nil for any non-.mlx model type (the
        // `default` arm), so .gguf must NOT be reported supported.
        XCTAssertFalse(
            service.compatibility(for: .gguf).isSupported,
            ".gguf must remain unsupported — the MLX registrar only handles .mlx"
        )
        XCTAssertFalse(
            service.registeredBackendSnapshot().localModelTypes.contains(.gguf)
        )
    }

    func test_register_leavesCloudProvidersEmpty() {
        let service = makeService()
        MLXBackends.register(with: service)

        XCTAssertTrue(
            service.registeredBackendSnapshot().cloudProviders.isEmpty,
            "The MLX registrar declares no cloud providers"
        )
    }
}
