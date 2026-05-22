// swiftlint:disable file_length type_body_length
//
//  VAE.swift
//  FluxKit
//
//  Variational Autoencoder: Encoder (image→latents) and Decoder (latents→image).
//

import Foundation
import MLX
import MLXNN

// MARK: - ResNet Block

class ResnetBlock2D: Module, @unchecked Sendable {
    @ModuleInfo(key: "norm1") var norm1: GroupNorm
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "norm2") var norm2: GroupNorm
    @ModuleInfo(key: "conv2") var conv2: Conv2d
    @ModuleInfo(key: "conv_shortcut") var convShortcut: Conv2d?

    init(inChannels: Int, outChannels: Int) {
        _norm1 = ModuleInfo(wrappedValue: GroupNorm(groupCount: 32, dimensions: inChannels, eps: 1e-6, pytorchCompatible: true), key: "norm1")
        _conv1 = ModuleInfo(wrappedValue: Conv2d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 3, padding: 1), key: "conv1")
        _norm2 = ModuleInfo(wrappedValue: GroupNorm(groupCount: 32, dimensions: outChannels, eps: 1e-6, pytorchCompatible: true), key: "norm2")
        _conv2 = ModuleInfo(wrappedValue: Conv2d(inputChannels: outChannels, outputChannels: outChannels, kernelSize: 3, padding: 1), key: "conv2")
        _convShortcut = ModuleInfo(wrappedValue: inChannels != outChannels ? Conv2d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 1) : nil, key: "conv_shortcut")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = MLXNN.silu(norm1(x))
        h = conv1(h)
        h = MLXNN.silu(norm2(h))
        h = conv2(h)
        return h + (convShortcut?(x) ?? x)
    }
}

// MARK: - VAE Attention

class VAEAttention: Module, @unchecked Sendable {
    @ModuleInfo(key: "group_norm") var groupNorm: GroupNorm
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: [Linear]
    let channels: Int

    init(channels: Int = 512) {
        self.channels = channels
        _groupNorm = ModuleInfo(wrappedValue: GroupNorm(groupCount: 32, dimensions: channels, eps: 1e-6, pytorchCompatible: true), key: "group_norm")
        _toQ = ModuleInfo(wrappedValue: Linear(channels, channels), key: "to_q")
        _toK = ModuleInfo(wrappedValue: Linear(channels, channels), key: "to_k")
        _toV = ModuleInfo(wrappedValue: Linear(channels, channels), key: "to_v")
        _toOut = ModuleInfo(wrappedValue: [Linear(channels, channels)], key: "to_out")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        let b = x.dim(0), h = x.dim(1), w = x.dim(2), c = x.dim(3)
        let normed = groupNorm(x).reshaped(b, h * w, c)
        let q = toQ(normed), k = toK(normed), v = toV(normed)
        let scale = Float(1.0 / Foundation.sqrt(Float(c)))
        var scores = MLX.matmul(q, k.transposed(0, 2, 1)) * scale
        scores = MLX.softmax(scores.asType(.float32), axis: -1).asType(scores.dtype)
        return toOut[0](MLX.matmul(scores, v)).reshaped(b, h, w, c) + residual
    }
}

// MARK: - Mid Block

class UnetMidBlock: Module, @unchecked Sendable {
    @ModuleInfo(key: "resnets") var resnets: [ResnetBlock2D]
    @ModuleInfo(key: "attentions") var attentions: [VAEAttention]

    init(channels: Int = 512) {
        _resnets = ModuleInfo(wrappedValue: [ResnetBlock2D(inChannels: channels, outChannels: channels), ResnetBlock2D(inChannels: channels, outChannels: channels)], key: "resnets")
        _attentions = ModuleInfo(wrappedValue: [VAEAttention(channels: channels)], key: "attentions")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = resnets[0](x); h = attentions[0](h); return resnets[1](h)
    }
}

// MARK: - Samplers

class DownSampler: Module, @unchecked Sendable {
    @ModuleInfo(key: "conv") var conv: Conv2d
    init(channels: Int) {
        _conv = ModuleInfo(wrappedValue: Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 3, stride: 2), key: "conv")
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Asymmetric pad: right+1, bottom+1 for stride-2 conv
        let padded = MLX.padded(x, widths: [IntOrPair((0, 0)), IntOrPair((0, 1)), IntOrPair((0, 1)), IntOrPair((0, 0))])
        return conv(padded)
    }
}

class UpSampler: Module, @unchecked Sendable {
    @ModuleInfo(key: "conv") var conv: Conv2d
    init(channels: Int) {
        _conv = ModuleInfo(wrappedValue: Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 3, padding: 1), key: "conv")
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0), h = x.dim(1), w = x.dim(2), c = x.dim(3)
        // Nearest-neighbor 2x via Upsample
        let upsample = Upsample(scaleFactor: 2.0, mode: .nearest)
        let up = upsample(x)
        return conv(up)
    }
}

// MARK: - Blocks

class DownBlock: Module, @unchecked Sendable {
    @ModuleInfo(key: "resnets") var resnets: [ResnetBlock2D]
    @ModuleInfo(key: "downsamplers") var downsamplers: [DownSampler]?

    init(inChannels: Int, outChannels: Int, numLayers: Int = 2, addDownsample: Bool = true) {
        var blocks = [ResnetBlock2D(inChannels: inChannels, outChannels: outChannels)]
        for _ in 1..<numLayers { blocks.append(ResnetBlock2D(inChannels: outChannels, outChannels: outChannels)) }
        _resnets = ModuleInfo(wrappedValue: blocks, key: "resnets")
        _downsamplers = ModuleInfo(wrappedValue: addDownsample ? [DownSampler(channels: outChannels)] : nil, key: "downsamplers")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x; for r in resnets { h = r(h) }; if let d = downsamplers { h = d[0](h) }; return h
    }
}

class UpBlock: Module, @unchecked Sendable {
    @ModuleInfo(key: "resnets") var resnets: [ResnetBlock2D]
    @ModuleInfo(key: "upsamplers") var upsamplers: [UpSampler]?

    init(inChannels: Int, outChannels: Int, numLayers: Int = 3, addUpsample: Bool = true) {
        var blocks = [ResnetBlock2D(inChannels: inChannels, outChannels: outChannels)]
        for _ in 1..<numLayers { blocks.append(ResnetBlock2D(inChannels: outChannels, outChannels: outChannels)) }
        _resnets = ModuleInfo(wrappedValue: blocks, key: "resnets")
        _upsamplers = ModuleInfo(wrappedValue: addUpsample ? [UpSampler(channels: outChannels)] : nil, key: "upsamplers")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x; for r in resnets { h = r(h) }; if let u = upsamplers { h = u[0](h) }; return h
    }
}

// MARK: - Encoder / Decoder

class VAEEncoder: Module, @unchecked Sendable {
    @ModuleInfo(key: "conv_in") var convIn: Conv2d
    @ModuleInfo(key: "down_blocks") var downBlocks: [DownBlock]
    @ModuleInfo(key: "mid_block") var midBlock: UnetMidBlock
    @ModuleInfo(key: "conv_norm_out") var normOut: GroupNorm
    @ModuleInfo(key: "conv_out") var convOut: Conv2d

    override init() {
        _convIn = ModuleInfo(wrappedValue: Conv2d(inputChannels: 3, outputChannels: 128, kernelSize: 3, padding: 1), key: "conv_in")
        _downBlocks = ModuleInfo(wrappedValue: [
            DownBlock(inChannels: 128, outChannels: 128), DownBlock(inChannels: 128, outChannels: 256),
            DownBlock(inChannels: 256, outChannels: 512), DownBlock(inChannels: 512, outChannels: 512, addDownsample: false)
        ], key: "down_blocks")
        _midBlock = ModuleInfo(wrappedValue: UnetMidBlock(channels: 512), key: "mid_block")
        _normOut = ModuleInfo(wrappedValue: GroupNorm(groupCount: 32, dimensions: 512, eps: 1e-6, pytorchCompatible: true), key: "conv_norm_out")
        _convOut = ModuleInfo(wrappedValue: Conv2d(inputChannels: 512, outputChannels: 32, kernelSize: 3, padding: 1), key: "conv_out")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = convIn(x); for b in downBlocks { h = b(h) }; h = midBlock(h); return convOut(MLXNN.silu(normOut(h)))
    }
}

class VAEDecoder: Module, @unchecked Sendable {
    @ModuleInfo(key: "conv_in") var convIn: Conv2d
    @ModuleInfo(key: "up_blocks") var upBlocks: [UpBlock]
    @ModuleInfo(key: "mid_block") var midBlock: UnetMidBlock
    @ModuleInfo(key: "conv_norm_out") var normOut: GroupNorm
    @ModuleInfo(key: "conv_out") var convOut: Conv2d

    override init() {
        _convIn = ModuleInfo(wrappedValue: Conv2d(inputChannels: 16, outputChannels: 512, kernelSize: 3, padding: 1), key: "conv_in")
        _upBlocks = ModuleInfo(wrappedValue: [
            UpBlock(inChannels: 512, outChannels: 512), UpBlock(inChannels: 512, outChannels: 512),
            UpBlock(inChannels: 512, outChannels: 256), UpBlock(inChannels: 256, outChannels: 128, addUpsample: false)
        ], key: "up_blocks")
        _midBlock = ModuleInfo(wrappedValue: UnetMidBlock(channels: 512), key: "mid_block")
        _normOut = ModuleInfo(wrappedValue: GroupNorm(groupCount: 32, dimensions: 128, eps: 1e-6, pytorchCompatible: true), key: "conv_norm_out")
        _convOut = ModuleInfo(wrappedValue: Conv2d(inputChannels: 128, outputChannels: 3, kernelSize: 3, padding: 1), key: "conv_out")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = convIn(x); h = midBlock(h); for b in upBlocks { h = b(h) }; return convOut(MLXNN.silu(normOut(h)))
    }
}

// MARK: - VAE (Public)

public class VAE: Module, @unchecked Sendable {
    @ModuleInfo(key: "encoder") var encoder: VAEEncoder
    @ModuleInfo(key: "decoder") var decoder: VAEDecoder

    public static let scalingFactor: Float = 0.3611
    public static let shiftFactor: Float = 0.1159

    public override init() {
        _encoder = ModuleInfo(wrappedValue: VAEEncoder(), key: "encoder")
        _decoder = ModuleInfo(wrappedValue: VAEDecoder(), key: "decoder")
    }

    public func decode(_ latents: MLXArray) -> MLXArray {
        decoder((latents / Self.scalingFactor) + Self.shiftFactor)
    }

    public func encode(_ image: MLXArray) -> MLXArray {
        let encoded = encoder(image)
        let parts = encoded.split(parts: 2, axis: -1)
        return (parts[0] - Self.shiftFactor) * Self.scalingFactor
    }
}
