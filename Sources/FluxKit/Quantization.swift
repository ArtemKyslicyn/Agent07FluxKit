//
//  Quantization.swift
//  FluxKit
//
//  Runtime quantization using MLXNN.quantize for reduced memory usage.
//

import Foundation
import Logging
import MLX
import MLXNN

private let logger = Logger(label: "FluxKit.Quantization")

/// Quantize a FluxPipeline's models for reduced memory usage.
public func quantizeFluxPipeline(
    transformer: MultiModalDiffusionTransformer,
    vae: VAE,
    t5Encoder: T5Encoder,
    clipEncoder: CLIPEncoder,
    groupSize: Int = 64,
    bits: Int = 4
) {
    // Quantize all models with default filter (all Linear layers)
    quantize(model: transformer, groupSize: groupSize, bits: bits)
    quantize(model: vae, groupSize: groupSize, bits: bits)
    quantize(model: t5Encoder, groupSize: groupSize, bits: bits)
    quantize(model: clipEncoder, groupSize: groupSize, bits: bits)
    logger.info("Quantized to \(bits)-bit (groupSize=\(groupSize))")
}
