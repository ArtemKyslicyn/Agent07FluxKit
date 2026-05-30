//
//  WeightLoading.swift
//  FluxKit
//
//  Load safetensors weights into model modules with key remapping.
//

import Foundation
import Logging
import MLX
import MLXNN

private let logger = Logger(label: "FluxKit.WeightLoading")

// MARK: - Weight Key Remapping

/// Remap PyTorch weight keys to FluxKit module paths.
/// Safetensors keys like `ff.net.2.weight` map to `ff.net.1.proj.weight`
/// because OutLinear wraps Linear with @ModuleInfo(key: "proj").
func remapWeightKey(_ key: String) -> String {
    var k = key
    // ff.net.0.proj → ff.net.0.proj (GELU linear — already correct)
    // ff.net.2.{weight,bias} → ff.net.1.proj.{weight,bias} (output linear via OutLinear.proj)
    k = k.replacingOccurrences(of: "ff.net.2.", with: "ff.net.1.proj.")
    // ff_context same pattern
    k = k.replacingOccurrences(of: "ff_context.net.2.", with: "ff_context.net.1.proj.")
    return k
}

// MARK: - Load Functions

/// Load transformer weights from a directory of safetensors files.
public func loadTransformerWeights(
    _ transformer: MultiModalDiffusionTransformer,
    from directory: URL,
    dtype: DType
) throws {
    let fm = FileManager.default
    let files = try fm.contentsOfDirectory(atPath: directory.path)
        .filter { $0.hasSuffix(".safetensors") }
        .sorted()

    let tensorCount: Int
    do {
        var allWeights: [String: MLXArray] = [:]
        for file in files {
            let url = directory.appendingPathComponent(file)
            let arrays = try MLX.loadArrays(url: url)
            for (key, value) in arrays {
                let remapped = remapWeightKey(key)
                allWeights[remapped] = value.asType(dtype)
            }
        }
        tensorCount = allWeights.count

        // Debug: compare weight keys vs model items paths
        let weightKeys = Set(allWeights.keys)
        let itemKeys = Set(transformer.filterMap(
            filter: Module.filterValidParameters, map: Module.mapParameters(map: { $0 })
        ).flattened().map { $0.0 })
        let matched = weightKeys.intersection(itemKeys)
        let unmatchedWeights = weightKeys.subtracting(itemKeys).sorted().prefix(10)
        let unmatchedItems = itemKeys.subtracting(weightKeys).sorted().prefix(10)
        logger.info("Keys: \(weightKeys.count) weights, \(itemKeys.count) item params, \(matched.count) matched")
        if !unmatchedWeights.isEmpty { logger.warning("Unmatched WEIGHT keys (not in model): \(Array(unmatchedWeights))") }
        if !unmatchedItems.isEmpty { logger.warning("Unmatched ITEM keys (no weight): \(Array(unmatchedItems))") }

        // Verify shapes match before loading
        let modelParams = transformer.filterMap(
            filter: Module.filterValidParameters, map: Module.mapParameters(map: { $0 })
        ).flattened()
        var shapeMismatches = 0
        for (key, modelArray) in modelParams {
            if let weightArray = allWeights[key] {
                if modelArray.shape != weightArray.shape {
                    logger.warning("SHAPE MISMATCH: \(key) model=\(modelArray.shape) weight=\(weightArray.shape)")
                    shapeMismatches += 1
                }
            }
        }
        if shapeMismatches > 0 {
            logger.warning("\(shapeMismatches) shape mismatches detected")
        } else {
            logger.debug("All matched weights have correct shapes")
        }

        let params = ModuleParameters.unflattened(allWeights)
        try transformer.update(parameters: params, verify: .none)
    }
    // allWeights dict fully released before returning
    MLX.GPU.clearCache()
    logger.info("Transformer loaded: \(tensorCount) tensors")
}

/// Load VAE weights. Transposes 4D Conv2d weights from NCHW to NHWC.
public func loadVAEWeights(_ vae: VAE, from file: URL, dtype: DType) throws {
    let tensorCount: Int
    do {
        var arrays = try MLX.loadArrays(url: file)

        // Transpose 4D weights: PyTorch NCHW → MLX NHWC
        for (key, value) in arrays {
            if value.ndim == 4 {
                arrays[key] = value.transposed(0, 2, 3, 1).asType(dtype)
            } else {
                arrays[key] = value.asType(dtype)
            }
        }

        tensorCount = arrays.count
        let params = ModuleParameters.unflattened(arrays)
        try vae.update(parameters: params, verify: .none)
    }
    MLX.GPU.clearCache()
    logger.info("VAE loaded: \(tensorCount) tensors")
}

/// Load T5 encoder weights.
public func loadT5EncoderWeights(_ encoder: T5Encoder, from directory: URL, dtype: DType) throws {
    let fm = FileManager.default
    let files = try fm.contentsOfDirectory(atPath: directory.path)
        .filter { $0.hasSuffix(".safetensors") }
        .sorted()

    let tensorCount: Int
    do {
        var allWeights: [String: MLXArray] = [:]
        for file in files {
            let url = directory.appendingPathComponent(file)
            let arrays = try MLX.loadArrays(url: url)
            for (key, value) in arrays {
                allWeights[key] = value.asType(dtype)
            }
        }

        // Copy relative attention bias from first block to top-level
        if let bias = allWeights["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"] {
            allWeights["relative_attention_bias.weight"] = bias
        }

        tensorCount = allWeights.count
        let params = ModuleParameters.unflattened(allWeights)
        try encoder.update(parameters: params, verify: .none)
    }
    MLX.GPU.clearCache()
    logger.info("T5 Encoder loaded: \(tensorCount) tensors")
}

/// Load CLIP encoder weights.
public func loadCLIPEncoderWeights(_ encoder: CLIPEncoder, from file: URL, dtype: DType) throws {
    let tensorCount: Int
    do {
        let arrays = try MLX.loadArrays(url: file)
        var casted: [String: MLXArray] = [:]
        for (key, value) in arrays {
            casted[key] = value.asType(dtype)
        }

        tensorCount = casted.count
        let params = ModuleParameters.unflattened(casted)
        try encoder.update(parameters: params, verify: .none)
    }
    MLX.GPU.clearCache()
    logger.info("CLIP Encoder loaded: \(tensorCount) tensors")
}
