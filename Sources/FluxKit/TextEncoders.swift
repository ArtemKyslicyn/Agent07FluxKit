// swiftlint:disable file_length type_body_length
//
//  TextEncoders.swift
//  FluxKit
//
//  T5 encoder (24 layers) and CLIP encoder (12 layers) for text conditioning.
//

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - T5 Encoder

/// T5 multi-head self-attention with relative position bias.
class T5MultiHeadAttention: Module, @unchecked Sendable {
    @ModuleInfo(key: "q") var queryProj: Linear
    @ModuleInfo(key: "k") var keyProj: Linear
    @ModuleInfo(key: "v") var valueProj: Linear
    @ModuleInfo(key: "o") var outProj: Linear

    let numHeads: Int
    let dKv: Int

    init(dModel: Int = 4096, numHeads: Int = 64, dKv: Int = 64) {
        self.numHeads = numHeads
        self.dKv = dKv
        _queryProj = ModuleInfo(wrappedValue: Linear(dModel, numHeads * dKv, bias: false), key: "q_proj")
        _keyProj = ModuleInfo(wrappedValue: Linear(dModel, numHeads * dKv, bias: false), key: "k_proj")
        _valueProj = ModuleInfo(wrappedValue: Linear(dModel, numHeads * dKv, bias: false), key: "v_proj")
        _outProj = ModuleInfo(wrappedValue: Linear(numHeads * dKv, dModel, bias: false), key: "out_proj")
    }

    func callAsFunction(_ x: MLXArray, positionBias: MLXArray) -> MLXArray {
        let b = x.dim(0)
        let seqLen = x.dim(1)

        let q = queryProj(x).reshaped(b, seqLen, numHeads, dKv).transposed(0, 2, 1, 3)
        let k = keyProj(x).reshaped(b, seqLen, numHeads, dKv).transposed(0, 2, 1, 3)
        let v = valueProj(x).reshaped(b, seqLen, numHeads, dKv).transposed(0, 2, 1, 3)

        // Attention with position bias
        let scale = Float(1.0 / Foundation.sqrt(Float(dKv)))
        var scores = MLX.matmul(q, k.transposed(0, 1, 3, 2)) * scale
        scores += positionBias
        let attnWeights = MLX.softmax(scores.asType(.float32), axis: -1).asType(scores.dtype)
        let attnOut = MLX.matmul(attnWeights, v)

        let merged = attnOut.transposed(0, 2, 1, 3).reshaped(b, seqLen, numHeads * dKv)
        return outProj(merged)
    }
}

/// T5 encoder layer with self-attention and feed-forward.
class T5EncoderLayer: Module, @unchecked Sendable {
    @ModuleInfo(key: "layer") var layers: [Module]

    init(dModel: Int = 4096, dFf: Int = 10240, numHeads: Int = 64, dKv: Int = 64) {
        let attention = T5AttentionBlock(dModel: dModel, numHeads: numHeads, dKv: dKv)
        let ff = T5FeedForward(dModel: dModel, dFf: dFf)
        _layers = ModuleInfo(wrappedValue: [attention, ff], key: "layers")
    }

    func callAsFunction(_ x: MLXArray, positionBias: MLXArray) -> MLXArray {
        guard let attn = layers[0] as? T5AttentionBlock,
              let ff = layers[1] as? T5FeedForward else { return x }
        let afterAttn = attn(x, positionBias: positionBias)
        return ff(afterAttn)
    }
}

/// T5 attention block with pre-norm and residual.
class T5AttentionBlock: Module, @unchecked Sendable {
    @ModuleInfo(key: "layer_norm") var norm: RMSNorm
    @ModuleInfo(key: "SelfAttention") var attention: T5MultiHeadAttention

    init(dModel: Int = 4096, numHeads: Int = 64, dKv: Int = 64) {
        _norm = ModuleInfo(wrappedValue: RMSNorm(dimensions: dModel, eps: 1e-6), key: "layer_norm")
        _attention = ModuleInfo(wrappedValue: T5MultiHeadAttention(dModel: dModel, numHeads: numHeads, dKv: dKv), key: "SelfAttention")
    }

    func callAsFunction(_ x: MLXArray, positionBias: MLXArray) -> MLXArray {
        x + attention(norm(x), positionBias: positionBias)
    }
}

/// T5 feed-forward with gated GELU activation.
class T5FeedForward: Module, @unchecked Sendable {
    @ModuleInfo(key: "layer_norm") var norm: RMSNorm
    @ModuleInfo(key: "DenseReluDense") var dense: T5DenseGatedGelu

    init(dModel: Int = 4096, dFf: Int = 10240) {
        _norm = ModuleInfo(wrappedValue: RMSNorm(dimensions: dModel, eps: 1e-6), key: "layer_norm")
        _dense = ModuleInfo(wrappedValue: T5DenseGatedGelu(dModel: dModel, dFf: dFf), key: "DenseReluDense")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x + dense(norm(x))
    }
}

/// T5 gated GELU dense layer.
class T5DenseGatedGelu: Module, @unchecked Sendable {
    @ModuleInfo(key: "wi_0") var gate: Linear
    @ModuleInfo(key: "wi_1") var proj: Linear
    @ModuleInfo(key: "wo") var output: Linear

    init(dModel: Int = 4096, dFf: Int = 10240) {
        _gate = ModuleInfo(wrappedValue: Linear(dModel, dFf, bias: false), key: "wi_0")
        _proj = ModuleInfo(wrappedValue: Linear(dModel, dFf, bias: false), key: "wi_1")
        _output = ModuleInfo(wrappedValue: Linear(dFf, dModel, bias: false), key: "wo")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gated = MLXNN.geluApproximate(gate(x)) * proj(x)
        return output(gated)
    }
}

/// Full T5 encoder with relative position bias.
public class T5Encoder: Module, @unchecked Sendable {
    @ModuleInfo(key: "shared") var embedding: Embedding
    @ModuleInfo(key: "encoder") var encoder: T5EncoderStack
    @ModuleInfo(key: "relative_attention_bias") var relativeAttentionBias: Embedding

    let numHeads: Int
    let numBuckets: Int
    let maxDistance: Int

    public init(
        vocabSize: Int = 32128, dModel: Int = 4096, numHeads: Int = 64,
        numLayers: Int = 24, dFf: Int = 10240, dKv: Int = 64
    ) {
        self.numHeads = numHeads
        self.numBuckets = 32
        self.maxDistance = 128
        _embedding = ModuleInfo(wrappedValue: Embedding(embeddingCount: vocabSize, dimensions: dModel), key: "shared")
        _encoder = ModuleInfo(wrappedValue: T5EncoderStack(
            numLayers: numLayers, dModel: dModel, dFf: dFf, numHeads: numHeads, dKv: dKv), key: "encoder")
        _relativeAttentionBias = ModuleInfo(wrappedValue: Embedding(embeddingCount: numBuckets, dimensions: numHeads), key: "relative_attention_bias")
    }

    public func callAsFunction(_ tokenIds: MLXArray) -> MLXArray {
        let embedded = embedding(tokenIds)
        let seqLen = tokenIds.dim(1)
        let bias = computePositionBias(seqLen: seqLen)
        return encoder(embedded, positionBias: bias)
    }

    /// Compute T5 relative position bias using bucketing.
    func computePositionBias(seqLen: Int) -> MLXArray {
        let positions = MLXArray(Array(0..<seqLen))
        let relPos = positions.expandedDimensions(axis: 0) - positions.expandedDimensions(axis: 1)

        // Bucket relative positions
        let bucketIndices = Self.relativeBuckets(relPos, numBuckets: numBuckets, maxDistance: maxDistance)
        let bias = relativeAttentionBias(bucketIndices)  // [seqLen, seqLen, numHeads]
        return bias.transposed(2, 0, 1).expandedDimensions(axis: 0)  // [1, numHeads, seqLen, seqLen]
    }

    /// T5-style relative position bucketing.
    static func relativeBuckets(_ relPos: MLXArray, numBuckets: Int = 32, maxDistance: Int = 128) -> MLXArray {
        var buckets = MLX.where(relPos .> 0, MLXArray(numBuckets / 2), MLXArray(0))
        let absPos = MLX.abs(relPos)

        let halfBuckets = numBuckets / 2
        let maxExact = halfBuckets / 2

        let isSmall = absPos .< maxExact
        let smallBuckets = absPos

        let logRatio = MLX.log(absPos.asType(.float32) / Float(maxExact))
        let logMax = Foundation.log(Float(maxDistance) / Float(maxExact))
        let largeBuckets = MLXArray(maxExact) + (logRatio / logMax * Float(halfBuckets - maxExact)).asType(.int32)
        let clampedLarge = MLX.minimum(largeBuckets, MLXArray(halfBuckets - 1))

        buckets += MLX.where(isSmall, smallBuckets, clampedLarge)
        return buckets
    }
}

/// T5 encoder stack (N layers + final norm).
class T5EncoderStack: Module, @unchecked Sendable {
    @ModuleInfo(key: "block") var blocks: [T5EncoderLayer]
    @ModuleInfo(key: "final_layer_norm") var finalNorm: RMSNorm

    init(numLayers: Int = 24, dModel: Int = 4096, dFf: Int = 10240, numHeads: Int = 64, dKv: Int = 64) {
        _blocks = ModuleInfo(wrappedValue: (0..<numLayers).map { _ in T5EncoderLayer(dModel: dModel, dFf: dFf, numHeads: numHeads, dKv: dKv) }, key: "block")
        _finalNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: dModel, eps: 1e-6), key: "final_layer_norm")
    }

    func callAsFunction(_ x: MLXArray, positionBias: MLXArray) -> MLXArray {
        var hidden = x
        for block in blocks { hidden = block(hidden, positionBias: positionBias) }
        return finalNorm(hidden)
    }
}

// MARK: - CLIP Encoder

/// CLIP self-attention using scaled dot-product attention.
class CLIPSdpaAttention: Module, @unchecked Sendable {
    @ModuleInfo(key: "q_proj") var queryProj: Linear
    @ModuleInfo(key: "k_proj") var keyProj: Linear
    @ModuleInfo(key: "v_proj") var valueProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    let numHeads: Int
    let headDim: Int

    init(hiddenSize: Int = 768, numHeads: Int = 12) {
        self.numHeads = numHeads
        self.headDim = hiddenSize / numHeads
        _queryProj = ModuleInfo(wrappedValue: Linear(hiddenSize, hiddenSize), key: "q_proj")
        _keyProj = ModuleInfo(wrappedValue: Linear(hiddenSize, hiddenSize), key: "k_proj")
        _valueProj = ModuleInfo(wrappedValue: Linear(hiddenSize, hiddenSize), key: "v_proj")
        _outProj = ModuleInfo(wrappedValue: Linear(hiddenSize, hiddenSize), key: "out_proj")
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let b = x.dim(0)
        let seqLen = x.dim(1)
        let scale = Float(1.0 / Foundation.sqrt(Float(headDim)))

        let q = queryProj(x).reshaped(b, seqLen, numHeads, headDim).transposed(0, 2, 1, 3)
        let k = keyProj(x).reshaped(b, seqLen, numHeads, headDim).transposed(0, 2, 1, 3)
        let v = valueProj(x).reshaped(b, seqLen, numHeads, headDim).transposed(0, 2, 1, 3)

        // Manual attention with additive causal mask (MLXFast doesn't support additive masks)
        var scores = MLX.matmul(q, k.transposed(0, 1, 3, 2)) * scale
        if let mask = mask {
            scores += mask
        }
        let attnWeights = MLX.softmax(scores.asType(.float32), axis: -1).asType(scores.dtype)
        let attnOut = MLX.matmul(attnWeights, v)

        let merged = attnOut.transposed(0, 2, 1, 3).reshaped(b, seqLen, numHeads * headDim)
        return outProj(merged)
    }
}

/// CLIP MLP: fc1 → GELU → fc2.
class CLIPMLP: Module, @unchecked Sendable {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(hiddenSize: Int = 768, intermediateSize: Int = 3072) {
        _fc1 = ModuleInfo(wrappedValue: Linear(hiddenSize, intermediateSize), key: "fc1")
        _fc2 = ModuleInfo(wrappedValue: Linear(intermediateSize, hiddenSize), key: "fc2")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(MLXNN.geluFastApproximate(fc1(x)))
    }
}

/// CLIP encoder layer: norm → attention → residual → norm → MLP → residual.
class CLIPEncoderLayer: Module, @unchecked Sendable {
    @ModuleInfo(key: "self_attn") var selfAttn: CLIPSdpaAttention
    @ModuleInfo(key: "layer_norm1") var norm1: LayerNorm
    @ModuleInfo(key: "layer_norm2") var norm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: CLIPMLP

    init(hiddenSize: Int = 768, intermediateSize: Int = 3072, numHeads: Int = 12) {
        _selfAttn = ModuleInfo(wrappedValue: CLIPSdpaAttention(hiddenSize: hiddenSize, numHeads: numHeads), key: "self_attn")
        _norm1 = ModuleInfo(wrappedValue: LayerNorm(dimensions: hiddenSize), key: "layer_norm1")
        _norm2 = ModuleInfo(wrappedValue: LayerNorm(dimensions: hiddenSize), key: "layer_norm2")
        _mlp = ModuleInfo(wrappedValue: CLIPMLP(hiddenSize: hiddenSize, intermediateSize: intermediateSize), key: "mlp")
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let attnOut = x + selfAttn(norm1(x), mask: mask)
        return attnOut + mlp(norm2(attnOut))
    }
}

/// Full CLIP text encoder.
public class CLIPEncoder: Module, @unchecked Sendable {
    @ModuleInfo(key: "text_model") var textModel: CLIPTextModel

    public init(hiddenSize: Int = 768, intermediateSize: Int = 3072,
                numHeads: Int = 12, numLayers: Int = 12,
                vocabSize: Int = 49408, maxPositionEmbeddings: Int = 77) {
        _textModel = ModuleInfo(wrappedValue: CLIPTextModel(
            hiddenSize: hiddenSize, intermediateSize: intermediateSize,
            numHeads: numHeads, numLayers: numLayers,
            vocabSize: vocabSize, maxPositionEmbeddings: maxPositionEmbeddings), key: "text_model")
    }

    /// Returns (lastHiddenState, pooledOutput).
    public func callAsFunction(_ tokenIds: MLXArray) -> (MLXArray, MLXArray) {
        textModel(tokenIds)
    }
}

/// CLIP text model: embeddings + encoder + final norm.
class CLIPTextModel: Module, @unchecked Sendable {
    @ModuleInfo(key: "embeddings") var embeddings: CLIPEmbeddings
    @ModuleInfo(key: "encoder") var encoder: CLIPEncoderStack
    @ModuleInfo(key: "final_layer_norm") var finalNorm: LayerNorm

    init(hiddenSize: Int, intermediateSize: Int, numHeads: Int,
         numLayers: Int, vocabSize: Int, maxPositionEmbeddings: Int) {
        _embeddings = ModuleInfo(wrappedValue: CLIPEmbeddings(
            hiddenSize: hiddenSize, vocabSize: vocabSize, maxLength: maxPositionEmbeddings), key: "embeddings")
        _encoder = ModuleInfo(wrappedValue: CLIPEncoderStack(
            hiddenSize: hiddenSize, intermediateSize: intermediateSize,
            numHeads: numHeads, numLayers: numLayers), key: "encoder")
        _finalNorm = ModuleInfo(wrappedValue: LayerNorm(dimensions: hiddenSize), key: "final_layer_norm")
    }

    func callAsFunction(_ tokenIds: MLXArray) -> (MLXArray, MLXArray) {
        let x = embeddings(tokenIds)
        let seqLen = tokenIds.dim(1)

        // Causal mask
        let mask = Self.causalMask(seqLen)
        let hidden = encoder(x, mask: mask)
        let normed = finalNorm(hidden)

        // Pool at EOS token position (last non-padding token)
        let eosPos = MLX.argMax(tokenIds, axis: -1)
        let pooled = normed[0..., eosPos, 0...]  // Simplified pooling
        return (normed, pooled.squeezed(axis: 1))
    }

    static func causalMask(_ size: Int) -> MLXArray {
        // Large negative value for masked positions
        // Shape [1, 1, size, size] for broadcasting with [batch, heads, seqLen, seqLen]
        let mask = MLX.full([size, size], values: Float(-1e9))
        let tril = MLX.tril(MLX.ones([size, size]))
        let flat = MLX.where(tril .> 0, MLXArray(Float(0)), mask)
        return flat.reshaped(1, 1, size, size)
    }
}

/// CLIP embeddings: token + position.
class CLIPEmbeddings: Module, @unchecked Sendable {
    @ModuleInfo(key: "token_embedding") var tokenEmbedding: Embedding
    @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

    init(hiddenSize: Int, vocabSize: Int, maxLength: Int) {
        _tokenEmbedding = ModuleInfo(wrappedValue: Embedding(embeddingCount: vocabSize, dimensions: hiddenSize), key: "token_embedding")
        _positionEmbedding = ModuleInfo(wrappedValue: Embedding(embeddingCount: maxLength, dimensions: hiddenSize), key: "position_embedding")
    }

    func callAsFunction(_ tokenIds: MLXArray) -> MLXArray {
        let seqLen = tokenIds.dim(1)
        let posIds = MLXArray(Array(0..<seqLen)).expandedDimensions(axis: 0)
        return tokenEmbedding(tokenIds) + positionEmbedding(posIds)
    }
}

/// CLIP encoder stack.
class CLIPEncoderStack: Module, @unchecked Sendable {
    @ModuleInfo(key: "layers") var layers: [CLIPEncoderLayer]

    init(hiddenSize: Int, intermediateSize: Int, numHeads: Int, numLayers: Int) {
        _layers = ModuleInfo(wrappedValue: (0..<numLayers).map { _ in CLIPEncoderLayer(
                hiddenSize: hiddenSize, intermediateSize: intermediateSize, numHeads: numHeads) }, key: "layers")
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        var hidden = x
        for layer in layers { hidden = layer(hidden, mask: mask) }
        return hidden
    }
}
