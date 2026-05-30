//
//  Pipeline.swift
//  FluxKit
//
//  FluxPipeline: unified text-to-image generator.
//  TextToImageGenerator protocol and DenoiseIterator.
//

import Foundation
import MLX
import MLXRandom
import MLXNN
import Hub
import Tokenizers
import Logging

private let logger = Logger(label: "FluxKit.Pipeline")

// MARK: - Protocols

/// Image generator with decode capability.
public protocol ImageGenerator: Sendable {
    func decode(xt: MLXArray) -> MLXArray
}

/// Text-to-image generator returning a denoising iterator.
public protocol TextToImageGenerator: ImageGenerator {
    func generateLatents(parameters: EvaluateParameters) -> DenoiseIterator
    func conditionText(prompt: String) -> (MLXArray, MLXArray)
}

// MARK: - Denoise Iterator

/// Iterates through denoising steps, yielding latents at each step.
public struct DenoiseIterator: Sequence, IteratorProtocol {
    let transformer: MultiModalDiffusionTransformer
    let promptEmbeds: MLXArray
    let pooledPromptEmbeds: MLXArray
    let guidanceEmbeds: Bool
    let parameters: EvaluateParameters
    let ropeCos: MLXArray
    let ropeSin: MLXArray

    var latents: MLXArray
    public private(set) var i: Int = 0

    public mutating func next() -> MLXArray? {
        guard i < parameters.numInferenceSteps else { return nil }

        let t = parameters.sigmas[i]
        let tNext = parameters.sigmas[i + 1]

        // Guidance embedding (for Dev/Kontext models)
        let guidance: MLXArray? = guidanceEmbeds
            ? MLXArray(parameters.guidance * Float(parameters.numTrainSteps))
            : nil

        // Predict velocity via transformer
        let timestepValue = t * Float(parameters.numTrainSteps)
        let noise = transformer(
            hiddenStates: latents,
            promptEmbeds: promptEmbeds,
            pooledPromptEmbeds: pooledPromptEmbeds,
            timestep: timestepValue.expandedDimensions(axis: 0),
            ropeCos: ropeCos,
            ropeSin: ropeSin,
            guidance: guidance?.expandedDimensions(axis: 0)
        )

        let dt = tNext - t

        #if FLUXKIT_DEBUG
        // Per-step magnitude probe. Each .item() is a synchronous GPU read that
        // stalls the pipeline — keep gated so release builds never pay for it.
        let nMin = noise.min().item(Float.self)
        let nMax = noise.max().item(Float.self)
        let nMean = noise.mean().item(Float.self)
        let tVal = t.item(Float.self)
        let dtVal = dt.item(Float.self)
        logger.debug("step=\(i) t=\(tVal) dt=\(dtVal) noise_min=\(nMin) noise_max=\(nMax) noise_mean=\(nMean)")
        #endif

        // Euler step: latents += velocity * dt
        latents += noise * dt

        MLX.eval(latents)
        i += 1
        return latents
    }
}

// MARK: - Flux Pipeline

/// Unified Flux pipeline for text-to-image generation.
public class FluxPipeline: @unchecked Sendable {
    public let transformer: MultiModalDiffusionTransformer
    public let vae: VAE
    public let t5Encoder: T5Encoder
    public let clipEncoder: CLIPEncoder
    public let clipTokenizer: CLIPBPETokenizer
    public let t5Tokenizer: any Tokenizer
    public let guidanceEmbeds: Bool
    public let t5MaxSeqLen: Int

    init(
        transformer: MultiModalDiffusionTransformer, vae: VAE,
        t5Encoder: T5Encoder, clipEncoder: CLIPEncoder,
        clipTokenizer: CLIPBPETokenizer, t5Tokenizer: any Tokenizer,
        guidanceEmbeds: Bool, t5MaxSeqLen: Int = 256
    ) {
        self.transformer = transformer
        self.vae = vae
        self.t5Encoder = t5Encoder
        self.clipEncoder = clipEncoder
        self.clipTokenizer = clipTokenizer
        self.t5Tokenizer = t5Tokenizer
        self.guidanceEmbeds = guidanceEmbeds
        self.t5MaxSeqLen = t5MaxSeqLen
    }

    // MARK: - Factory

    /// Load a Flux pipeline from a FluxConfiguration.
    public static func load(
        config: FluxConfiguration, hub: HubApi = HubApi(),
        loadConfig: LoadConfiguration = LoadConfiguration()
    ) throws -> TextToImageGenerator {
        let dtype = loadConfig.dType
        let repo = Hub.Repo(id: config.id)
        let repoPath = hub.localRepoLocation(repo)
        guard FileManager.default.fileExists(atPath: repoPath.appendingPathComponent("transformer").path) else {
            throw FluxKitError.modelNotDownloaded(config.id)
        }

        logger.info("Loading FluxPipeline from \(repoPath.path)")
        logMLXMemory("Before pipeline load")

        // Create models and load weights one at a time.
        // Quantize each component immediately after loading to minimize peak memory.
        // Without this, ALL float16 weights (~22GB) would coexist before quantization.
        let transformer = MultiModalDiffusionTransformer(guidanceEmbeds: config.guidanceEmbeds)
        try loadTransformerWeights(transformer, from: repoPath.appendingPathComponent("transformer"), dtype: dtype)
        if loadConfig.quantize { quantize(model: transformer, groupSize: 64, bits: 8) }
        MLX.GPU.clearCache()
        logMLXMemory("Transformer loaded" + (loadConfig.quantize ? " + quantized" : ""))

        // VAE must stay float16 — 4-bit quantization destroys image decoding quality.
        // VAE is small (~300MB) so keeping it unquantized is fine.
        let vae = VAE()
        try loadVAEWeights(vae, from: repoPath.appendingPathComponent("vae/diffusion_pytorch_model.safetensors"), dtype: dtype)
        MLX.GPU.clearCache()
        logMLXMemory("VAE loaded (float16)")

        // T5 can be quantized — saves ~4GB with acceptable quality loss for text encoding.
        let t5 = T5Encoder()
        try loadT5EncoderWeights(t5, from: repoPath.appendingPathComponent("text_encoder_2"), dtype: dtype)
        if loadConfig.quantize { quantize(model: t5, groupSize: 64, bits: 4) }
        MLX.GPU.clearCache()
        logMLXMemory("T5 encoder loaded" + (loadConfig.quantize ? " + quantized" : ""))

        // CLIP stays float16 — small model (~500MB), quality-sensitive.
        let clip = CLIPEncoder()
        try loadCLIPEncoderWeights(clip, from: repoPath.appendingPathComponent("text_encoder/model.safetensors"), dtype: dtype)
        MLX.GPU.clearCache()
        logMLXMemory("All components loaded")

        // Load tokenizers
        let clipTokenizer = try CLIPBPETokenizer(
            vocabURL: repoPath.appendingPathComponent("tokenizer/vocab.json"),
            mergesURL: repoPath.appendingPathComponent("tokenizer/merges.txt")
        )
        let t5Tokenizer = try loadT5Tokenizer(from: repoPath.appendingPathComponent("tokenizer_2"))

        let t5MaxSeq = config.guidanceEmbeds ? 512 : 256

        let pipeline = FluxPipeline(
            transformer: transformer, vae: vae,
            t5Encoder: t5, clipEncoder: clip,
            clipTokenizer: clipTokenizer, t5Tokenizer: t5Tokenizer,
            guidanceEmbeds: config.guidanceEmbeds, t5MaxSeqLen: t5MaxSeq
        )

        logger.info("FluxPipeline loaded successfully (dtype=\(dtype), quantize=\(loadConfig.quantize))")
        return pipeline
    }
}

// MARK: - TextToImageGenerator Conformance

extension FluxPipeline: TextToImageGenerator {
    public func conditionText(prompt: String) -> (MLXArray, MLXArray) {
        // T5 encoding
        let t5Tokens = t5Tokenizer.encode(text: prompt)
        let t5Input = MLXArray(Array(t5Tokens.prefix(t5MaxSeqLen))).expandedDimensions(axis: 0)
        let promptEmbeds = t5Encoder(t5Input)

        // CLIP encoding
        let clipTokens = clipTokenizer.encode(prompt)
        let clipInput = MLXArray(clipTokens).expandedDimensions(axis: 0)
        let (_, pooledOutput) = clipEncoder(clipInput)

        return (promptEmbeds, pooledOutput)
    }

    public func generateLatents(parameters: EvaluateParameters) -> DenoiseIterator {
        let (promptEmbeds, pooledPromptEmbeds) = conditionText(prompt: parameters.prompt)

        // Create random latents
        let h16 = parameters.height / 16
        let w16 = parameters.width / 16
        let latentShape = [1, h16 * w16, 64]

        if let seed = parameters.seed {
            MLXRandom.seed(seed)
        }
        let latents = MLXRandom.normal(latentShape).asType(promptEmbeds.dtype)

        // Compute RoPE cos/sin from position IDs
        let ids = createPositionIDs(h16: h16, w16: w16, promptLen: promptEmbeds.dim(1))
        let posEmbed = EmbedND(axesDims: [16, 56, 56])
        let (ropeCos, ropeSin) = posEmbed(ids)

        return DenoiseIterator(
            transformer: transformer,
            promptEmbeds: promptEmbeds,
            pooledPromptEmbeds: pooledPromptEmbeds,
            guidanceEmbeds: guidanceEmbeds,
            parameters: parameters,
            ropeCos: ropeCos,
            ropeSin: ropeSin,
            latents: latents
        )
    }

    public func decode(xt: MLXArray) -> MLXArray {
        vae.decode(xt)
    }
}

// MARK: - Position IDs

/// Create 3D position IDs (time=0, height, width) for RoPE embedding.
func createPositionIDs(h16: Int, w16: Int, promptLen: Int) -> MLXArray {
    // Text positions: all zeros
    let textIds = MLX.zeros([promptLen, 3])

    // Image positions: (0, row, col)
    var imagePositions: [[Int32]] = []
    for row in 0..<h16 {
        for col in 0..<w16 {
            imagePositions.append([0, Int32(row), Int32(col)])
        }
    }
    let imageIds = MLXArray(imagePositions.flatMap { $0 }).reshaped(h16 * w16, 3)

    return MLX.concatenated([textIds, imageIds.asType(.float32)], axis: 0).expandedDimensions(axis: 0)
}

// MARK: - Memory Logging

/// Log MLX GPU memory stats (active + cache) for debugging memory pressure
private func logMLXMemory(_ label: String) {
    let snap = GPU.snapshot()
    let activeMB = snap.activeMemory / (1024 * 1024)
    let cacheMB = snap.cacheMemory / (1024 * 1024)
    let peakMB = snap.peakMemory / (1024 * 1024)
    logger.info("[FluxKit] \(label): active=\(activeMB)MB cache=\(cacheMB)MB peak=\(peakMB)MB")
}

// MARK: - Errors

public enum FluxKitError: LocalizedError {
    case modelNotDownloaded(String)
    case loadFailed(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotDownloaded(let id): return "Model '\(id)' not downloaded. Go to Settings → Image Models."
        case .loadFailed(let msg): return "Failed to load model: \(msg)"
        }
    }
}
