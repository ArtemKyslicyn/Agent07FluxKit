// swiftlint:disable file_length type_body_length function_body_length
//
//  Transformer.swift
//  FluxKit
//
//  Multi-Modal Diffusion Transformer (MMDiT) for Flux image generation.
//  19 joint transformer blocks + 38 single transformer blocks.
//

import Foundation
import Logging
import MLX
import MLXFast
import MLXNN

private let logger = Logger(label: "FluxKit.Transformer")

// MARK: - Conditioning Embedders

/// Projects pooled text embeddings to inner dimension.
class TextEmbedder: Module, @unchecked Sendable {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(inputDim: Int = 768, innerDim: Int = 3072) {
        _linear1 = ModuleInfo(wrappedValue: Linear(inputDim, innerDim), key: "linear_1")
        _linear2 = ModuleInfo(wrappedValue: Linear(innerDim, innerDim), key: "linear_2")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { linear2(MLXNN.silu(linear1(x))) }
}

/// Timestep embedding via sinusoidal projection + MLP.
class TimestepEmbedder: Module, @unchecked Sendable {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(innerDim: Int = 3072) {
        _linear1 = ModuleInfo(wrappedValue: Linear(256, innerDim), key: "linear_1")
        _linear2 = ModuleInfo(wrappedValue: Linear(innerDim, innerDim), key: "linear_2")
    }

    func callAsFunction(_ t: MLXArray) -> MLXArray {
        let projected = timestepProjection(t)
        return linear2(MLXNN.silu(linear1(projected)))
    }
}

/// Combines timestep + text pool + optional guidance embeddings.
class TimeTextEmbed: Module, @unchecked Sendable {
    @ModuleInfo(key: "timestep_embedder") var timestepEmbedder: TimestepEmbedder
    @ModuleInfo(key: "text_embedder") var textEmbedder: TextEmbedder
    @ModuleInfo(key: "guidance_embedder") var guidanceEmbedder: TimestepEmbedder?

    init(innerDim: Int = 3072, pooledDim: Int = 768, hasGuidance: Bool = false) {
        _timestepEmbedder = ModuleInfo(wrappedValue: TimestepEmbedder(innerDim: innerDim), key: "timestep_embedder")
        _textEmbedder = ModuleInfo(wrappedValue: TextEmbedder(inputDim: pooledDim, innerDim: innerDim), key: "text_embedder")
        _guidanceEmbedder = ModuleInfo(wrappedValue: hasGuidance ? TimestepEmbedder(innerDim: innerDim) : nil, key: "guidance_embedder")
    }

    func callAsFunction(timestep: MLXArray, pooledProjection: MLXArray, guidance: MLXArray?) -> MLXArray {
        var emb = timestepEmbedder(timestep) + textEmbedder(pooledProjection)
        if let ge = guidanceEmbedder, let g = guidance {
            emb += ge(g)
        }
        return emb
    }
}

// MARK: - Adaptive Layer Norms

/// AdaLayerNorm for joint blocks: produces 6 chunks (shift/scale/gate for attn + mlp).
class AdaLayerNormZero: Module, @unchecked Sendable {
    @ModuleInfo(key: "linear") var linear: Linear

    init(innerDim: Int = 3072) {
        _linear = ModuleInfo(wrappedValue: Linear(innerDim, innerDim * 6), key: "linear")
    }

    func callAsFunction(_ emb: MLXArray) -> (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) {
        let chunks = MLXNN.silu(emb).expandedDimensions(axis: 1)
        let projected = linear(chunks)
        let parts = projected.split(parts: 6, axis: -1)
        return (parts[0], parts[1], parts[2], parts[3], parts[4], parts[5])
    }
}

/// AdaLayerNorm for single blocks: produces 3 chunks.
class AdaLayerNormZeroSingle: Module, @unchecked Sendable {
    @ModuleInfo(key: "linear") var linear: Linear

    init(innerDim: Int = 3072) {
        _linear = ModuleInfo(wrappedValue: Linear(innerDim, innerDim * 3), key: "linear")
    }

    func callAsFunction(_ emb: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let chunks = MLXNN.silu(emb).expandedDimensions(axis: 1)
        let projected = linear(chunks)
        let parts = projected.split(parts: 3, axis: -1)
        return (parts[0], parts[1], parts[2])
    }
}

/// Continuous adaptive norm for output projection.
class AdaLayerNormContinuous: Module, @unchecked Sendable {
    @ModuleInfo(key: "linear") var linear: Linear
    @ModuleInfo(key: "norm") var norm: LayerNorm

    init(condDim: Int = 3072, embDim: Int = 3072) {
        _linear = ModuleInfo(wrappedValue: Linear(condDim, embDim * 2), key: "linear")
        _norm = ModuleInfo(wrappedValue: LayerNorm(dimensions: embDim, eps: 1e-6), key: "norm")
    }

    func callAsFunction(_ x: MLXArray, conditioning: MLXArray) -> MLXArray {
        let params = linear(MLXNN.silu(conditioning)).expandedDimensions(axis: 1)
        let parts = params.split(parts: 2, axis: -1)
        return norm(x) * (1 + parts[0]) + parts[1]
    }
}

// MARK: - Joint Attention (Dual-stream)

class JointAttention: Module, @unchecked Sendable {
    let numHeads: Int
    let headDim: Int

    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: [Linear]
    @ModuleInfo(key: "add_q_proj") var addQProj: Linear
    @ModuleInfo(key: "add_k_proj") var addKProj: Linear
    @ModuleInfo(key: "add_v_proj") var addVProj: Linear
    @ModuleInfo(key: "to_add_out") var toAddOut: Linear
    @ModuleInfo(key: "norm_q") var normQ: RMSNorm
    @ModuleInfo(key: "norm_k") var normK: RMSNorm
    @ModuleInfo(key: "norm_added_q") var normAddedQ: RMSNorm
    @ModuleInfo(key: "norm_added_k") var normAddedK: RMSNorm

    init(innerDim: Int = 3072, numHeads: Int = 24) {
        self.numHeads = numHeads
        self.headDim = innerDim / numHeads
        _toQ = ModuleInfo(wrappedValue: Linear(innerDim, innerDim), key: "to_q")
        _toK = ModuleInfo(wrappedValue: Linear(innerDim, innerDim), key: "to_k")
        _toV = ModuleInfo(wrappedValue: Linear(innerDim, innerDim), key: "to_v")
        _toOut = ModuleInfo(wrappedValue: [Linear(innerDim, innerDim)], key: "to_out")
        _addQProj = ModuleInfo(wrappedValue: Linear(innerDim, innerDim), key: "add_q_proj")
        _addKProj = ModuleInfo(wrappedValue: Linear(innerDim, innerDim), key: "add_k_proj")
        _addVProj = ModuleInfo(wrappedValue: Linear(innerDim, innerDim), key: "add_v_proj")
        _toAddOut = ModuleInfo(wrappedValue: Linear(innerDim, innerDim), key: "to_add_out")
        _normQ = ModuleInfo(wrappedValue: RMSNorm(dimensions: headDim), key: "norm_q")
        _normK = ModuleInfo(wrappedValue: RMSNorm(dimensions: headDim), key: "norm_k")
        _normAddedQ = ModuleInfo(wrappedValue: RMSNorm(dimensions: headDim), key: "norm_added_q")
        _normAddedK = ModuleInfo(wrappedValue: RMSNorm(dimensions: headDim), key: "norm_added_k")
    }

    func callAsFunction(hidden: MLXArray, context: MLXArray, ropeCos: MLXArray, ropeSin: MLXArray) -> (MLXArray, MLXArray) {
        let b = hidden.dim(0)
        let hiddenLen = hidden.dim(1)
        let contextLen = context.dim(1)

        // Image stream Q/K/V
        var q = toQ(hidden).reshaped(b, hiddenLen, numHeads, headDim)
        var k = toK(hidden).reshaped(b, hiddenLen, numHeads, headDim)
        let v = toV(hidden).reshaped(b, hiddenLen, numHeads, headDim)
        q = normQ(q)
        k = normK(k)

        // Context stream Q/K/V
        var addQ = addQProj(context).reshaped(b, contextLen, numHeads, headDim)
        var addK = addKProj(context).reshaped(b, contextLen, numHeads, headDim)
        let addV = addVProj(context).reshaped(b, contextLen, numHeads, headDim)
        addQ = normAddedQ(addQ)
        addK = normAddedK(addK)

        // Apply RoPE to concatenated Q/K (context + image positions)
        var fullQ = MLX.concatenated([addQ, q], axis: 1)  // [B, ctxLen+hidLen, heads, headDim]
        var fullK = MLX.concatenated([addK, k], axis: 1)
        fullQ = applyRope(fullQ, cos: ropeCos, sin: ropeSin)
        fullK = applyRope(fullK, cos: ropeCos, sin: ropeSin)
        fullQ = fullQ.transposed(0, 2, 1, 3)
        fullK = fullK.transposed(0, 2, 1, 3)
        let fullV = MLX.concatenated([addV, v], axis: 1).transposed(0, 2, 1, 3)

        // Scaled dot-product attention
        let scale = Float(1.0 / Foundation.sqrt(Float(headDim)))
        let attnOut = MLXFast.scaledDotProductAttention(queries: fullQ, keys: fullK, values: fullV, scale: scale, mask: nil)
        let merged = attnOut.transposed(0, 2, 1, 3).reshaped(b, contextLen + hiddenLen, numHeads * headDim)

        // Split back
        let contextOut = toAddOut(merged[0..., ..<contextLen, 0...])
        let hiddenOut = toOut[0](merged[0..., contextLen..., 0...])
        return (hiddenOut, contextOut)
    }
}

// MARK: - Feed-Forward

class FeedForward: Module, @unchecked Sendable {
    @ModuleInfo(key: "net") var net: [Module]

    init(innerDim: Int = 3072) {
        let hiddenDim = innerDim * 4  // GELU-gated: hidden = 4x
        let geluLinear = GeluLinear(inputDim: innerDim, outputDim: hiddenDim)
        let outLinear = OutLinear(inputDim: hiddenDim, outputDim: innerDim)
        _net = ModuleInfo(wrappedValue: [geluLinear, outLinear], key: "net")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard let gelu = net[0] as? GeluLinear, let out = net[1] as? OutLinear else { return x }
        return out(gelu(x))
    }
}

class GeluLinear: Module, @unchecked Sendable {
    @ModuleInfo(key: "proj") var proj: Linear
    init(inputDim: Int, outputDim: Int) {
        _proj = ModuleInfo(wrappedValue: Linear(inputDim, outputDim), key: "proj")
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { MLXNN.geluApproximate(proj(x)) }
}

/// Output linear in FeedForward. Wraps Linear with @ModuleInfo so that:
/// 1) Weights load correctly (path: ff.net.1.proj.weight)
/// 2) quantize() can replace Linear with QuantizedLinear via @ModuleInfo setter
class OutLinear: Module, @unchecked Sendable {
    @ModuleInfo(key: "proj") var proj: Linear
    init(inputDim: Int, outputDim: Int) {
        _proj = ModuleInfo(wrappedValue: Linear(inputDim, outputDim), key: "proj")
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { proj(x) }
}

// MARK: - Joint Transformer Block

class JointTransformerBlock: Module, @unchecked Sendable {
    @ModuleInfo(key: "norm1") var norm1: AdaLayerNormZero      // image
    @ModuleInfo(key: "norm1_context") var norm1Context: AdaLayerNormZero  // text
    @ModuleInfo(key: "attn") var attn: JointAttention
    @ModuleInfo(key: "ff") var ff: FeedForward
    @ModuleInfo(key: "ff_context") var ffContext: FeedForward

    let innerDim: Int

    init(innerDim: Int = 3072, numHeads: Int = 24) {
        self.innerDim = innerDim
        _norm1 = ModuleInfo(wrappedValue: AdaLayerNormZero(innerDim: innerDim), key: "norm1")
        _norm1Context = ModuleInfo(wrappedValue: AdaLayerNormZero(innerDim: innerDim), key: "norm1_context")
        _attn = ModuleInfo(wrappedValue: JointAttention(innerDim: innerDim, numHeads: numHeads), key: "attn")
        _ff = ModuleInfo(wrappedValue: FeedForward(innerDim: innerDim), key: "ff")
        _ffContext = ModuleInfo(wrappedValue: FeedForward(innerDim: innerDim), key: "ff_context")
    }

    func callAsFunction(hidden: MLXArray, context: MLXArray, emb: MLXArray, ropeCos: MLXArray, ropeSin: MLXArray) -> (MLXArray, MLXArray) {
        // AdaLN for image stream
        let (shiftMSA, scaleMSA, gateMSA, shiftMLP, scaleMLP, gateMLP) = norm1(emb)
        let normedHidden = LayerNorm(dimensions: innerDim, eps: 1e-6)(hidden) * (1 + scaleMSA) + shiftMSA

        // AdaLN for context stream
        let (cShiftMSA, cScaleMSA, cGateMSA, cShiftMLP, cScaleMLP, cGateMLP) = norm1Context(emb)
        let normedContext = LayerNorm(dimensions: innerDim, eps: 1e-6)(context) * (1 + cScaleMSA) + cShiftMSA

        // Joint attention
        let (attnHidden, attnContext) = attn(hidden: normedHidden, context: normedContext, ropeCos: ropeCos, ropeSin: ropeSin)
        var newHidden = hidden + gateMSA * attnHidden
        var newContext = context + cGateMSA * attnContext

        // Feed-forward
        let normedH2 = LayerNorm(dimensions: innerDim, eps: 1e-6)(newHidden) * (1 + scaleMLP) + shiftMLP
        newHidden += gateMLP * ff(normedH2)

        let normedC2 = LayerNorm(dimensions: innerDim, eps: 1e-6)(newContext) * (1 + cScaleMLP) + cShiftMLP
        newContext += cGateMLP * ffContext(normedC2)

        return (newHidden, newContext)
    }
}

// MARK: - Single Transformer Block

class SingleTransformerBlock: Module, @unchecked Sendable {
    @ModuleInfo(key: "norm") var norm: AdaLayerNormZeroSingle
    @ModuleInfo(key: "attn") var attn: SingleBlockAttention
    @ModuleInfo(key: "proj_mlp") var projMLP: Linear
    @ModuleInfo(key: "proj_out") var projOut: Linear

    let innerDim: Int

    init(innerDim: Int = 3072, numHeads: Int = 24) {
        self.innerDim = innerDim
        let mlpHiddenDim = innerDim * 4
        _norm = ModuleInfo(wrappedValue: AdaLayerNormZeroSingle(innerDim: innerDim), key: "norm")
        _attn = ModuleInfo(wrappedValue: SingleBlockAttention(innerDim: innerDim, numHeads: numHeads), key: "attn")
        _projMLP = ModuleInfo(wrappedValue: Linear(innerDim, mlpHiddenDim), key: "proj_mlp")
        _projOut = ModuleInfo(wrappedValue: Linear(innerDim + mlpHiddenDim, innerDim), key: "proj_out")
    }

    func callAsFunction(_ x: MLXArray, emb: MLXArray, ropeCos: MLXArray, ropeSin: MLXArray) -> MLXArray {
        let (shift, scale, gate) = norm(emb)
        let normed = LayerNorm(dimensions: innerDim, eps: 1e-6)(x) * (1 + scale) + shift

        // Parallel attention + MLP
        let attnOut = attn(normed, ropeCos: ropeCos, ropeSin: ropeSin)
        let mlpOut = MLXNN.geluApproximate(projMLP(normed))

        // Concatenate and project
        let combined = MLX.concatenated([attnOut, mlpOut], axis: -1)
        let output = gate * projOut(combined)

        #if FLUXKIT_DEBUG
        // Each .item() is a synchronous GPU read — gate keeps it out of release builds.
        struct SingleDebug { nonisolated(unsafe) static var count = 0 }
        if SingleDebug.count < 3 || SingleDebug.count == 37 {
            let gateAbs = MLX.abs(gate).mean().item(Float.self)
            let outAbs = MLX.abs(output).mean().item(Float.self)
            let xAbs = MLX.abs(x).mean().item(Float.self)
            logger.debug("single_block[\(SingleDebug.count)]: gate=\(gateAbs) output=\(outAbs) x_in=\(xAbs)")
        }
        SingleDebug.count += 1
        #endif

        return x + output
    }
}

/// Self-attention for single blocks.
class SingleBlockAttention: Module, @unchecked Sendable {
    let numHeads: Int
    let headDim: Int

    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "norm_q") var normQ: RMSNorm
    @ModuleInfo(key: "norm_k") var normK: RMSNorm

    init(innerDim: Int = 3072, numHeads: Int = 24) {
        self.numHeads = numHeads
        self.headDim = innerDim / numHeads
        _toQ = ModuleInfo(wrappedValue: Linear(innerDim, innerDim), key: "to_q")
        _toK = ModuleInfo(wrappedValue: Linear(innerDim, innerDim), key: "to_k")
        _toV = ModuleInfo(wrappedValue: Linear(innerDim, innerDim), key: "to_v")
        _normQ = ModuleInfo(wrappedValue: RMSNorm(dimensions: headDim), key: "norm_q")
        _normK = ModuleInfo(wrappedValue: RMSNorm(dimensions: headDim), key: "norm_k")
    }

    func callAsFunction(_ x: MLXArray, ropeCos: MLXArray, ropeSin: MLXArray) -> MLXArray {
        let b = x.dim(0), seqLen = x.dim(1)
        var q = toQ(x).reshaped(b, seqLen, numHeads, headDim)
        var k = toK(x).reshaped(b, seqLen, numHeads, headDim)
        let v = toV(x).reshaped(b, seqLen, numHeads, headDim)
        q = normQ(q)
        k = normK(k)

        // Apply RoPE
        q = applyRope(q, cos: ropeCos, sin: ropeSin)
        k = applyRope(k, cos: ropeCos, sin: ropeSin)

        let qT = q.transposed(0, 2, 1, 3)
        let kT = k.transposed(0, 2, 1, 3)
        let vT = v.transposed(0, 2, 1, 3)

        let scale = Float(1.0 / Foundation.sqrt(Float(headDim)))
        let attnOut = MLXFast.scaledDotProductAttention(queries: qT, keys: kT, values: vT, scale: scale, mask: nil)
        return attnOut.transposed(0, 2, 1, 3).reshaped(b, seqLen, numHeads * headDim)
    }
}

// MARK: - Multi-Modal Diffusion Transformer (Main Model)

public class MultiModalDiffusionTransformer: Module, @unchecked Sendable {
    let innerDim: Int
    let numHeads: Int
    let inChannels: Int

    @ModuleInfo(key: "x_embedder") var xEmbedder: Linear
    @ModuleInfo(key: "context_embedder") var contextEmbedder: Linear
    @ModuleInfo(key: "time_text_embed") var timeTextEmbed: TimeTextEmbed
    @ModuleInfo(key: "transformer_blocks") var jointBlocks: [JointTransformerBlock]
    @ModuleInfo(key: "single_transformer_blocks") var singleBlocks: [SingleTransformerBlock]
    @ModuleInfo(key: "norm_out") var normOut: AdaLayerNormContinuous
    @ModuleInfo(key: "proj_out") var projOut: Linear

    public let posEmbed: EmbedND

    public init(
        numHeads: Int = 24, headDim: Int = 128,
        numJointLayers: Int = 19, numSingleLayers: Int = 38,
        inChannels: Int = 64, contextDim: Int = 4096,
        pooledDim: Int = 768, guidanceEmbeds: Bool = false
    ) {
        let innerDim = numHeads * headDim
        self.innerDim = innerDim
        self.numHeads = numHeads
        self.inChannels = inChannels
        self.posEmbed = EmbedND(axesDims: [16, 56, 56])

        _xEmbedder = ModuleInfo(wrappedValue: Linear(inChannels, innerDim), key: "x_embedder")
        _contextEmbedder = ModuleInfo(wrappedValue: Linear(contextDim, innerDim), key: "context_embedder")
        _timeTextEmbed = ModuleInfo(wrappedValue: TimeTextEmbed(innerDim: innerDim, pooledDim: pooledDim, hasGuidance: guidanceEmbeds), key: "time_text_embed")
        _jointBlocks = ModuleInfo(wrappedValue:
            (0..<numJointLayers).map { _ in JointTransformerBlock(innerDim: innerDim, numHeads: numHeads) }, key: "transformer_blocks")
        _singleBlocks = ModuleInfo(wrappedValue:
            (0..<numSingleLayers).map { _ in SingleTransformerBlock(innerDim: innerDim, numHeads: numHeads) }, key: "single_transformer_blocks")
        _normOut = ModuleInfo(wrappedValue: AdaLayerNormContinuous(condDim: innerDim, embDim: innerDim), key: "norm_out")
        _projOut = ModuleInfo(wrappedValue: Linear(innerDim, inChannels), key: "proj_out")
    }

    /// Forward pass: predict noise from noisy latents + text conditioning.
    public func callAsFunction(
        hiddenStates: MLXArray,
        promptEmbeds: MLXArray,
        pooledPromptEmbeds: MLXArray,
        timestep: MLXArray,
        ropeCos: MLXArray, ropeSin: MLXArray,
        guidance: MLXArray? = nil
    ) -> MLXArray {
        var hidden = xEmbedder(hiddenStates)
        var context = contextEmbedder(promptEmbeds)

        let emb = timeTextEmbed(timestep: timestep, pooledProjection: pooledPromptEmbeds, guidance: guidance)

        #if FLUXKIT_DEBUG
        struct DebugOnce { nonisolated(unsafe) static var step = 0 }
        let isFirstStep = DebugOnce.step == 0
        DebugOnce.step += 1
        if isFirstStep {
            logger.debug("emb: abs_mean=\(MLX.abs(emb).mean().item(Float.self))")
            logger.debug("hidden_in: abs_mean=\(MLX.abs(hidden).mean().item(Float.self))")
            logger.debug("context_in: abs_mean=\(MLX.abs(context).mean().item(Float.self))")
            logger.debug("ropeCos: shape=\(ropeCos.shape) ropeSin: shape=\(ropeSin.shape)")
        }
        #endif

        // Joint blocks (dual-stream)
        for (i, block) in jointBlocks.enumerated() {
            let result = block(hidden: hidden, context: context, emb: emb, ropeCos: ropeCos, ropeSin: ropeSin)
            hidden = result.0
            context = result.1
            #if FLUXKIT_DEBUG
            if isFirstStep && (i == 0 || i == 1 || i == 9 || i == 18) {
                logger.debug("joint[\(i)]: hidden_abs=\(MLX.abs(hidden).mean().item(Float.self)) ctx_abs=\(MLX.abs(context).mean().item(Float.self))")
            }
            #endif
        }

        // Single blocks (merged stream)
        var merged = MLX.concatenated([context, hidden], axis: 1)
        for (i, block) in singleBlocks.enumerated() {
            merged = block(merged, emb: emb, ropeCos: ropeCos, ropeSin: ropeSin)
            #if FLUXKIT_DEBUG
            if isFirstStep && (i == 0 || i == 1 || i == 19 || i == 37) {
                logger.debug("single[\(i)]: merged_abs=\(MLX.abs(merged).mean().item(Float.self))")
            }
            #endif
        }

        // Extract hidden states (skip context prefix)
        hidden = merged[0..., context.dim(1)..., 0...]

        #if FLUXKIT_DEBUG
        if isFirstStep {
            logger.debug("pre_norm: hidden_abs=\(MLX.abs(hidden).mean().item(Float.self))")
        }
        #endif

        // Output projection
        let normed = normOut(hidden, conditioning: emb)

        #if FLUXKIT_DEBUG
        if isFirstStep {
            logger.debug("post_norm: normed_abs=\(MLX.abs(normed).mean().item(Float.self))")
        }
        #endif

        return projOut(normed)
    }
}
