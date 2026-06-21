
import Foundation
import Hub
import MLX
import MLXNN
import MLXRandom
import Tokenizers
import Logging

private let logger = Logger(label: "flux.swift.FluxModelCore")

public struct FluxModelConfiguration {
    public let transformerConfig: MultiModalDiffusionConfiguration
    public let t5Config: T5Configuration
    public let clipConfig: CLIPConfiguration
    public let vaeConfig: VAEConfiguration
    public let t5MaxSequenceLength: Int
    public let clipMaxSequenceLength: Int
    public let clipPaddingToken: Int32
    
    nonisolated(unsafe) public static let schnell = FluxModelConfiguration(
        transformerConfig: MultiModalDiffusionConfiguration(),
        t5Config: T5Configuration(),
        clipConfig: CLIPConfiguration(),
        vaeConfig: VAEConfiguration(),
        t5MaxSequenceLength: 256,
        clipMaxSequenceLength: 77,
        clipPaddingToken: 49407
    )
    
    nonisolated(unsafe) public static let dev = FluxModelConfiguration(
        transformerConfig: MultiModalDiffusionConfiguration(guidanceEmbeds: true),
        t5Config: T5Configuration(),
        clipConfig: CLIPConfiguration(),
        vaeConfig: VAEConfiguration(),
        t5MaxSequenceLength: 512,
        clipMaxSequenceLength: 77,
        clipPaddingToken: 49407
    )
    
    nonisolated(unsafe) public static let kontextDev = FluxModelConfiguration(
        transformerConfig: MultiModalDiffusionConfiguration(guidanceEmbeds: true),
        t5Config: T5Configuration(
            vocabSize: 32128,
            dModel: 4096,
            dKv: 64,
            dFf: 10240,
            numHeads: 64,
            numLayers: 24
        ),
        clipConfig: CLIPConfiguration(
            hiddenSize: 768,
            intermediateSize: 3072,
            headDimension: 64,
            batchSize: 1,
            numAttentionHeads: 12,
            positionEmbeddingsCount: 77,
            tokenEmbeddingsCount: 49408,
            numHiddenLayers: 11
        ),
        vaeConfig: VAEConfiguration(),
        t5MaxSequenceLength: 512,
        clipMaxSequenceLength: 77,
        clipPaddingToken: 49407
    )
}

public class FluxModelCore: @unchecked Sendable {
    public let transformer: MultiModalDiffusionTransformer
    public let vae: VAE
    public let t5Encoder: T5Encoder
    public let clipEncoder: CLIPEncoder
    
    var clipTokenizer: CLIPTokenizer
    var t5Tokenizer: any Tokenizer
    
    public let configuration: FluxModelConfiguration
    public var modelDirectory: URL?
    
    public init(hub: HubApi, fluxConfiguration: FluxConfiguration, modelConfiguration: FluxModelConfiguration) throws {
        self.configuration = modelConfiguration
        
        let repo = Hub.Repo(id: fluxConfiguration.id)
        let directory = hub.localRepoLocation(repo)
        
        (self.t5Tokenizer, self.clipTokenizer) = try FLUX.loadTokenizers(directory: directory, hub: hub)
        
        self.transformer = MultiModalDiffusionTransformer(modelConfiguration.transformerConfig)
        self.vae = VAE(modelConfiguration.vaeConfig)
        self.t5Encoder = T5Encoder(modelConfiguration.t5Config)
        self.clipEncoder = CLIPEncoder(modelConfiguration.clipConfig)
    }
    
    public init(hub: HubApi, modelDirectory: URL, modelConfiguration: FluxModelConfiguration) throws {
        self.configuration = modelConfiguration
        self.modelDirectory = modelDirectory
        
        logger.info("Initializing from quantized model directory: \(modelDirectory.path)")
        
        (self.t5Tokenizer, self.clipTokenizer) = try FLUX.loadTokenizers(directory: modelDirectory, hub: hub)
        
        self.transformer = MultiModalDiffusionTransformer(modelConfiguration.transformerConfig)
        self.vae = VAE(modelConfiguration.vaeConfig)
        self.t5Encoder = T5Encoder(modelConfiguration.t5Config)
        self.clipEncoder = CLIPEncoder(modelConfiguration.clipConfig)
    }
    
    /// Set to `true` after ``loadWeights(from:dtype:)`` when the on-disk weights
    /// were already quantized (each component's `config.json` carried a
    /// `quantization` block, or the safetensors shipped `.scales`/`.biases`
    /// tensors). When this is `true`, the caller MUST NOT run the in-memory
    /// `quantize(...)` pass in `FluxConfiguration` — the layers are already
    /// `QuantizedLinear` and re-quantizing would corrupt them.
    ///
    /// Mirrors the MLX-LLM convention where a checkpoint declares its
    /// quantization in `config.json` so the loader skips fp16-then-quantize.
    public private(set) var loadedQuantized = false

    public func loadWeights(from directory: URL, dtype: DType = .float16) throws {
        self.modelDirectory = directory
        logger.info("Loading weights from: \(directory.path)")
        logger.info("Using dtype: \(dtype)")

        var anyQuantized = false
        anyQuantized = try loadTransformerWeights(from: directory.appending(path: "transformer"), dtype: dtype) || anyQuantized
        anyQuantized = try loadVAEWeights(from: directory.appending(path: "vae"), dtype: dtype) || anyQuantized
        anyQuantized = try loadT5EncoderWeights(from: directory.appending(path: "text_encoder_2"), dtype: dtype) || anyQuantized
        anyQuantized = try loadCLIPEncoderWeights(from: directory.appending(path: "text_encoder"), dtype: dtype) || anyQuantized

        self.loadedQuantized = anyQuantized
        logger.info("All weights loaded successfully (quantized: \(anyQuantized))")
    }

    /// Reads an MLX-LLM-style `quantization` block from a component's
    /// `config.json`, e.g. `{"quantization": {"group_size": 64, "bits": 4}}`.
    /// Returns `nil` when no such block exists (fp16 checkpoint).
    private func quantizationConfig(in directory: URL) -> (groupSize: Int, bits: Int)? {
        Self.quantizationConfig(in: directory)
    }

    /// Static, testable form of the `config.json` quantization-block reader.
    /// Mirrors the MLX-LLM convention: a checkpoint declares its quantization in
    /// `config.json` so loaders can skip the fp16-then-quantize round-trip.
    public static func quantizationConfig(in directory: URL) -> (groupSize: Int, bits: Int)? {
        let configURL = directory.appending(path: "config.json")
        guard let data = try? Data(contentsOf: configURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let quant = json["quantization"] as? [String: Any] else {
            return nil
        }
        let groupSize = (quant["group_size"] as? Int) ?? QuantizationUtils.defaultGroupSize
        let bits = (quant["bits"] as? Int) ?? QuantizationUtils.defaultBits
        return (groupSize, bits)
    }

    /// Converts the matching `Linear` layers of `module` into `QuantizedLinear`
    /// so the pre-quantized `.weight`/`.scales`/`.biases` tensors land in the
    /// right slots when `update(parameters:)` runs. A layer is quantized only if
    /// the loaded weight dict carries a `<path>.scales` tensor for it — this
    /// matches `applySelectiveQuantization` used by the metadata.json path and
    /// keeps non-quantized layers (e.g. small embeddings) in fp16.
    ///
    /// mflux additionally 4-bit-quantizes the CLIP token/position embeddings and
    /// the T5 `shared` embedding (issue #39, GAP 2), so `Embedding` layers are
    /// converted to `QuantizedEmbedding` as well — but only when a matching
    /// `.scales` tensor is present, so fp16 bundles (and the diffusers-quantized
    /// bundles that keep embeddings in fp16) are unaffected. `quantizeSingle`
    /// (the default `apply`) already produces a `QuantizedEmbedding` for an
    /// `Embedding` because `Embedding` is `Quantizable`.
    private func applyPreQuantization(
        to module: Module, weights: [String: MLXArray], groupSize: Int, bits: Int
    ) {
        quantize(model: module, filter: { path, m in
            guard m is Linear || m is Embedding,
                  weights["\(path).scales"] != nil else { return nil }
            return (groupSize: groupSize, bits: bits)
        })
    }

    /// Returns `true` when the component was loaded as pre-quantized weights.
    @discardableResult
    private func loadTransformerWeights(from directory: URL, dtype: DType) throws -> Bool {
        var transformerWeights = [String: MLXArray]()

        guard let enumerator = FileManager.default.enumerator(
            at: directory, includingPropertiesForKeys: nil
        ) else {
            throw FluxError.weightsNotFound("Unable to enumerate transformer directory: \(directory)")
        }

        for case let url as URL in enumerator {
            if url.pathExtension == "safetensors" {
                let w = try loadArrays(url: url)
                for (key, value) in w {
                    let newKey = FLUX.remapWeightKey(key)
                    // .scales/.biases of quantized layers are uint32/half — never
                    // re-cast them through dtype; only fp32/fp16 fp weights.
                    if value.dtype != .bfloat16 && value.dtype != .uint32 {
                        transformerWeights[newKey] = value.asType(dtype)
                    } else {
                        transformerWeights[newKey] = value
                    }
                }
            }
        }

        let isQuantized = quantizationConfig(in: directory) != nil
            || transformerWeights.keys.contains { $0.hasSuffix(".scales") }
        if isQuantized {
            let (groupSize, bits) = quantizationConfig(in: directory)
                ?? (QuantizationUtils.defaultGroupSize, QuantizationUtils.defaultBits)
            applyPreQuantization(to: transformer, weights: transformerWeights, groupSize: groupSize, bits: bits)
        }
        transformer.update(parameters: ModuleParameters.unflattened(transformerWeights))
        return isQuantized
    }

    @discardableResult
    private func loadVAEWeights(from directory: URL, dtype: DType) throws -> Bool {
        let vaeURL = directory.appending(path: "diffusion_pytorch_model.safetensors")
        let rawVAEWeights = try loadArrays(url: vaeURL)

        // mflux wraps each VAE conv/group-norm in an extra `.conv2d.`/`.norm.`
        // segment (`decoder.conv_in.conv2d.weight`); strip it to match our
        // module tree. HF-diffusers VAE keys lack the segment, so this is a
        // no-op there (issue #39, GAP 3).
        let isMflux = FLUX.isMfluxVAE(rawVAEWeights.keys)
        var vaeWeights = [String: MLXArray]()
        for (key, value) in rawVAEWeights {
            vaeWeights[FLUX.remapVAEKey(key)] = value
        }

        for (key, value) in vaeWeights {
            if value.dtype != .bfloat16 && value.dtype != .uint32 {
                vaeWeights[key] = value.asType(dtype)
            }
            // HF-diffusers conv weights are `[out, in, kh, kw]` and need a
            // channels-last transpose; mflux already stores them channels-last
            // (`[out, kh, kw, in]`), so transposing would corrupt them.
            if value.ndim == 4 && !isMflux {
                vaeWeights[key] = value.transposed(0, 2, 3, 1)
            }
        }
        let isQuantized = quantizationConfig(in: directory) != nil
            || vaeWeights.keys.contains { $0.hasSuffix(".scales") }
        if isQuantized {
            let (groupSize, bits) = quantizationConfig(in: directory)
                ?? (QuantizationUtils.defaultGroupSize, QuantizationUtils.defaultBits)
            applyPreQuantization(to: vae, weights: vaeWeights, groupSize: groupSize, bits: bits)
        }
        vae.update(parameters: ModuleParameters.unflattened(vaeWeights))
        return isQuantized
    }

    @discardableResult
    private func loadT5EncoderWeights(from directory: URL, dtype: DType) throws -> Bool {
        var weights = [String: MLXArray]()

        guard let enumerator = FileManager.default.enumerator(
            at: directory, includingPropertiesForKeys: nil
        ) else {
            throw FluxError.weightsNotFound("Unable to enumerate T5 encoder directory: \(directory)")
        }

        for case let url as URL in enumerator {
            if url.pathExtension == "safetensors" {
                let w = try loadArrays(url: url)
                for (key, value) in w {
                    // Translate mflux-flattened T5 keys (`t5_blocks.N...`,
                    // `final_layer_norm.*`) into our nested HF-diffusers schema.
                    // HF-diffusers keys pass through unchanged (issue #39, GAP 1).
                    let newKey = FLUX.remapT5EncoderKey(key)
                    if value.dtype != .bfloat16 && value.dtype != .uint32 {
                        weights[newKey] = value.asType(dtype)
                    } else {
                        weights[newKey] = value
                    }
                }
            }
        }

        // Both HF-diffusers and mflux nest relative_attention_bias inside block
        // 0's SelfAttention; hoist it (and, for a quantized bundle, its
        // `.scales`/`.biases`) to the top level our `T5Encoder` module expects.
        // Hoisting the quantization tensors lets `applyPreQuantization` convert
        // the top-level `relative_attention_bias` Embedding to a
        // QuantizedEmbedding (issue #39, GAP 2); fp16 bundles only carry
        // `.weight`, so the `.scales`/`.biases` copies are simply absent.
        for suffix in ["weight", "scales", "biases"] {
            let src = "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.\(suffix)"
            if let tensor = weights[src] {
                weights["relative_attention_bias.\(suffix)"] = tensor
            }
        }
        // mflux ships a `relative_attention_bias` inside EVERY T5 block, but our
        // `MultiHeadAttention` module has no such submodule (only q/k/v/o) — the
        // bias is shared and lives once at the top level. Drop the per-block
        // copies so `update(parameters:)` doesn't try to bind keys with no
        // matching module. (HF-diffusers only ships block 0's, already hoisted.)
        for key in weights.keys
        where key.contains(".SelfAttention.relative_attention_bias.") {
            weights.removeValue(forKey: key)
        }

        let isQuantized = quantizationConfig(in: directory) != nil
            || weights.keys.contains { $0.hasSuffix(".scales") }
        if isQuantized {
            let (groupSize, bits) = quantizationConfig(in: directory)
                ?? (QuantizationUtils.defaultGroupSize, QuantizationUtils.defaultBits)
            applyPreQuantization(to: t5Encoder, weights: weights, groupSize: groupSize, bits: bits)
        }
        t5Encoder.update(parameters: ModuleParameters.unflattened(weights))
        return isQuantized
    }

    @discardableResult
    private func loadCLIPEncoderWeights(from directory: URL, dtype: DType) throws -> Bool {
        let weightsURL = directory.appending(path: "model.safetensors")
        var weights = try loadArrays(url: weightsURL)

        for (key, value) in weights {
            if value.dtype != .bfloat16 && value.dtype != .uint32 {
                weights[key] = value.asType(dtype)
            }
        }
        let isQuantized = quantizationConfig(in: directory) != nil
            || weights.keys.contains { $0.hasSuffix(".scales") }
        if isQuantized {
            let (groupSize, bits) = quantizationConfig(in: directory)
                ?? (QuantizationUtils.defaultGroupSize, QuantizationUtils.defaultBits)
            applyPreQuantization(to: clipEncoder, weights: weights, groupSize: groupSize, bits: bits)
        }
        clipEncoder.update(parameters: ModuleParameters.unflattened(weights))
        return isQuantized
    }
    
    
    public func conditionText(prompt: String) -> (MLXArray, MLXArray) {
        let t5Tokens = t5Tokenizer.encode(text: prompt, addSpecialTokens: true)
        let paddedT5Tokens = Array(t5Tokens.prefix(configuration.t5MaxSequenceLength))
            + Array(repeating: 0, count: max(0, configuration.t5MaxSequenceLength - min(t5Tokens.count, configuration.t5MaxSequenceLength)))
        
        let clipTokens = clipTokenizer.tokenize(text: prompt)
        let paddedClipTokens = Array(clipTokens.prefix(configuration.clipMaxSequenceLength))
            + Array(repeating: configuration.clipPaddingToken, count: max(0, configuration.clipMaxSequenceLength - min(clipTokens.count, configuration.clipMaxSequenceLength)))
        
        let promptEmbeddings = t5Encoder(MLXArray(paddedT5Tokens)[.newAxis])
        let pooledPromptEmbeddings = clipEncoder(MLXArray(paddedClipTokens)[.newAxis])
        
        return (promptEmbeddings, pooledPromptEmbeddings)
    }
    
    
    public func ensureLoaded() {
        eval(transformer, t5Encoder, clipEncoder)
    }
    
    public func decode(xt: MLXArray) -> MLXArray {
        var x = vae.decode(xt)
        x = clip(x / 2 + 0.5, min: 0, max: 1)
        return x
    }
    
    public func detachedDecoder() -> ImageDecoder {
        let autoencoder = self.vae
        func decode(xt: MLXArray) -> MLXArray {
            var x = autoencoder.decode(xt)
            x = clip(x / 2 + 0.5, min: 0, max: 1)
            return x
        }
        return decode(xt:)
    }
}


extension FluxModelCore: FLUXComponents {}
