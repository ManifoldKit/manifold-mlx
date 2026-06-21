import Foundation

/// Pure, filesystem-only validation of a FLUX.1 on-disk bundle layout.
///
/// `FluxModelCore.loadWeights(from:)` consumes the diffusers multi-folder
/// layout (see ``FluxConfiguration/flux1Schnell``). For a COMPLETE bundle the
/// loader needs every component present — transformer, VAE, both text encoders
/// (CLIP + T5-XXL), and both tokenizers — regardless of whether the weights are
/// fp16 or pre-quantized 4-bit.
///
/// This type lets callers (and tests) check that a directory is a complete
/// bundle BEFORE handing it to the Metal-bound loader, and to distinguish the
/// two common shapes seen in the wild:
///
///  - A **complete diffusers bundle** (loadable here).
///  - The **argmax single-file 4-bit bundle**
///    (`argmaxinc/mlx-FLUX.1-schnell-4bit-quantized`), which ships ONLY a
///    quantized transformer + autoencoder with **no T5 text encoder and no
///    tokenizer** — it cannot satisfy `FluxModelCore.loadWeights`. Detecting it
///    up front lets a caller emit a clear "incomplete bundle" diagnostic
///    instead of a deep, opaque weight-load failure.
///
/// All checks are pure filesystem reads (no MLX, no Metal, no weight parsing),
/// so this is fully unit-testable against tiny synthetic directories with no
/// real weights.
public enum FluxBundleLayout {

    /// A single component folder + the relative glob/path the loader reads from
    /// it. `requiresSafetensors == true` means at least one `.safetensors` file
    /// must exist anywhere under the folder; otherwise a named file is required.
    public struct RequiredComponent: Sendable, Equatable {
        public let folder: String
        /// A specific filename that must exist directly under `folder`, or `nil`
        /// when any `*.safetensors` under `folder` satisfies the requirement.
        public let requiredFile: String?

        public init(folder: String, requiredFile: String?) {
            self.folder = folder
            self.requiredFile = requiredFile
        }
    }

    /// The components a COMPLETE diffusers bundle must provide for
    /// `FluxModelCore.loadWeights(from:)` to succeed. Mirrors the globs in
    /// ``FluxConfiguration/flux1Schnell``.
    public static let requiredComponents: [RequiredComponent] = [
        // Transformer: one or more sharded *.safetensors.
        RequiredComponent(folder: "transformer", requiredFile: nil),
        // VAE: a single named file (matches loadVAEWeights).
        RequiredComponent(folder: "vae", requiredFile: "diffusion_pytorch_model.safetensors"),
        // CLIP text encoder: a single named file (matches loadCLIPEncoderWeights).
        RequiredComponent(folder: "text_encoder", requiredFile: "model.safetensors"),
        // T5-XXL text encoder: one or more sharded *.safetensors.
        RequiredComponent(folder: "text_encoder_2", requiredFile: nil),
    ]

    /// Tokenizer folders a complete bundle must carry. The argmax single-file
    /// bundle omits these entirely, which is the cheapest way to spot it.
    public static let requiredTokenizerFolders: [String] = ["tokenizer", "tokenizer_2"]

    /// The outcome of validating a directory.
    public enum Validation: Equatable, Sendable {
        /// All required components and tokenizers are present.
        case complete
        /// A transformer + VAE exist but the T5 text encoder and/or tokenizers
        /// are missing — the shape of the argmax single-file 4-bit bundle. The
        /// associated value lists the missing top-level pieces.
        case incompleteArgmaxStyle(missing: [String])
        /// Not a recognizable FLUX bundle at all (e.g. empty directory).
        case notABundle(missing: [String])
    }

    /// Validates `directory` against the complete-bundle requirements.
    ///
    /// - Returns `.complete` when every required component, every tokenizer, and
    ///   `model_index.json` are present.
    /// - Returns `.incompleteArgmaxStyle` when a transformer (and usually a VAE)
    ///   is present but the T5 text encoder and/or tokenizers are missing — the
    ///   transformer-only argmax shape this loader cannot consume.
    /// - Returns `.notABundle` when even the transformer is missing.
    public static func validate(_ directory: URL) -> Validation {
        let fm = FileManager.default

        func folderHasSafetensors(_ folder: String) -> Bool {
            let dir = directory.appending(path: folder)
            guard let items = try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil) else {
                return false
            }
            return items.contains { $0.pathExtension == "safetensors" }
        }

        func fileExists(_ relative: String) -> Bool {
            fm.fileExists(atPath: directory.appending(path: relative).path)
        }

        func componentPresent(_ c: RequiredComponent) -> Bool {
            if let file = c.requiredFile {
                return fileExists("\(c.folder)/\(file)")
            }
            return folderHasSafetensors(c.folder)
        }

        var missing: [String] = []
        for component in requiredComponents where !componentPresent(component) {
            missing.append(component.folder)
        }
        for folder in requiredTokenizerFolders where !fileExists(folder) {
            missing.append(folder)
        }
        if !fileExists("model_index.json") {
            missing.append("model_index.json")
        }

        if missing.isEmpty {
            return .complete
        }

        // A transformer present but T5/tokenizers absent is the argmax single-
        // file shape. If even the transformer is gone, it isn't a FLUX bundle.
        let hasTransformer = componentPresent(requiredComponents[0])
        return hasTransformer
            ? .incompleteArgmaxStyle(missing: missing)
            : .notABundle(missing: missing)
    }

    /// `true` only when `directory` is a complete, loadable diffusers bundle.
    public static func isCompleteBundle(_ directory: URL) -> Bool {
        validate(directory) == .complete
    }
}
