import ManifoldInference

/// The MLX family registrar.
///
/// Moved in from core's `ManifoldBackendsUmbrella` in the v0.48 companion
/// split (core PR C2, ManifoldKit#1749) and de-`#if`'d — this package always
/// compiles the backend, so registration is unconditional.
///
/// ```swift
/// import ManifoldKit
/// import ManifoldMLX
///
/// let kit = try await ManifoldKit.quickStart(backends: [MLXBackends.self])
/// ```
public enum MLXBackends: BackendRegistrar {
    @MainActor
    public static func register(with service: InferenceService) {
        service.registerBackendFactory { modelType in
            switch modelType {
            case .mlx: return MLXBackend()
            default:   return nil
            }
        }
        service.declareSupport(for: .mlx)
    }
}
