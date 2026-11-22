import ManifoldInference

/// The template's thinking behavior when the request leaves `maxThinkingTokens` unset.
public enum MLXThinkingDefault: Sendable {
  /// Leave the template's default behavior in effect.
  case modelDefault
  /// Pass `enable_thinking: false` to templates that accept that setting.
  case disabled
}

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
    register(with: service, thinkingDefault: .modelDefault)
  }

  /// Registers an MLX factory with a default for requests that omit a thinking budget.
  /// Zero asks the template to disable thinking. A positive budget asks it to enable
  /// thinking and limits the reasoning emitted by the backend.
  @MainActor
  public static func register(
    with service: InferenceService,
    thinkingDefault: MLXThinkingDefault
  ) {
    service.registerBackendFactory { modelType in
      switch modelType {
      case .mlx: return MLXBackend(thinkingDefault: thinkingDefault)
      default: return nil
      }
    }
    service.declareSupport(for: .mlx)
  }
}
