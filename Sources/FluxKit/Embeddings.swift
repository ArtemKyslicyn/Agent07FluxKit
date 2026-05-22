//
//  Embeddings.swift
//  FluxKit
//
//  Rotary positional embeddings (RoPE) and timestep sinusoidal projection.
//

import Foundation
import MLX
import MLXNN

// MARK: - RoPE Embedding (3-axis: time, height, width)

/// Computes cos/sin frequencies for rotary positional embedding on a single axis.
/// Returns (cos, sin) each of shape [seqLen, dim/2].
func ropeFreqs(pos: MLXArray, dim: Int, theta: Float = 10000.0) -> (MLXArray, MLXArray) {
    let halfDim = dim / 2
    let indices = MLXArray(Array(stride(from: Float(0), to: Float(halfDim), by: 1)))
    let omega = 1.0 / MLX.pow(MLXArray(theta), 2.0 * indices / Float(dim))

    // pos: [seqLen], omega: [halfDim]
    let angles = pos.expandedDimensions(axis: -1) * omega  // [seqLen, halfDim]
    return (MLX.cos(angles), MLX.sin(angles))
}

/// Multi-axis RoPE for 3D positions (time, height, width).
/// Returns (cos, sin) each of shape [1, seqLen, totalHalfDim].
public class EmbedND: Module, @unchecked Sendable {
    let axesDims: [Int]  // [16, 56, 56] for Flux → totalHalfDim = 8+28+28 = 64
    let theta: Float

    public init(axesDims: [Int], theta: Float = 10000.0) {
        self.axesDims = axesDims
        self.theta = theta
    }

    /// Returns (cos, sin) concatenated across all axes.
    public func callAsFunction(_ ids: MLXArray) -> (MLXArray, MLXArray) {
        // ids: [1, numPositions, 3]
        var allCos: [MLXArray] = []
        var allSin: [MLXArray] = []
        for (axis, dim) in axesDims.enumerated() {
            let posSlice = ids[0, 0..., axis]  // [numPositions]
            let (c, s) = ropeFreqs(pos: posSlice, dim: dim, theta: theta)
            allCos.append(c)  // [numPositions, dim/2]
            allSin.append(s)
        }
        // Concatenate along last axis: [seqLen, sum(dim/2)] = [seqLen, 64]
        let cosAll = MLX.concatenated(allCos, axis: -1).expandedDimensions(axis: 0)  // [1, seqLen, 64]
        let sinAll = MLX.concatenated(allSin, axis: -1).expandedDimensions(axis: 0)
        return (cosAll, sinAll)
    }
}

// MARK: - Apply RoPE

/// Apply rotary position embeddings to queries/keys.
/// x: [batch, seqLen, numHeads, headDim=128]
/// cos, sin: [1, seqLen, halfDim=64]
public func applyRope(_ x: MLXArray, cos ropeC: MLXArray, sin ropeS: MLXArray) -> MLXArray {
    let headDim = x.dim(-1)
    let halfDim = headDim / 2

    // Split head dim into pairs: [B, S, H, halfDim, 2]
    let xPairs = x.reshaped(x.dim(0), x.dim(1), x.dim(2), halfDim, 2)
    let x0 = xPairs[0..., 0..., 0..., 0..., 0]  // [B, S, H, halfDim]
    let x1 = xPairs[0..., 0..., 0..., 0..., 1]

    // cos/sin: [1, S, halfDim] → expand for heads: [1, S, 1, halfDim]
    let c = ropeC.expandedDimensions(axis: 2)  // [1, S, 1, halfDim]
    let s = ropeS.expandedDimensions(axis: 2)

    // Complex rotation: (x0 + i*x1) * (cos + i*sin)
    let rotX0 = c * x0 - s * x1
    let rotX1 = s * x0 + c * x1

    // Interleave back: [B, S, H, halfDim, 2] → [B, S, H, headDim]
    let result = MLX.stacked([rotX0, rotX1], axis: -1)
    return result.reshaped(x.dim(0), x.dim(1), x.dim(2), headDim)
}

// MARK: - Timestep Sinusoidal Projection

/// Sinusoidal timestep embedding (matching Flux paper, dim=256).
public func timestepProjection(_ timestep: MLXArray, dim: Int = 256, maxPeriod: Float = 10000.0) -> MLXArray {
    let halfDim = dim / 2
    let indices = MLXArray(Array(stride(from: Float(0), to: Float(halfDim), by: 1)))
    let freqs = MLX.exp(-Foundation.log(maxPeriod) * indices / Float(halfDim))
    let args = timestep.expandedDimensions(axis: -1) * freqs
    return MLX.concatenated([MLX.cos(args), MLX.sin(args)], axis: -1)
}
