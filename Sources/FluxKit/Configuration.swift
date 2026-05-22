//
//  Configuration.swift
//  FluxKit
//
//  Model configurations, load settings, evaluation parameters, and sigma schedules.
//

import Foundation
import MLX
import Hub  // Re-exported from swift-transformers

// MARK: - Load Configuration

public struct LoadConfiguration: Sendable {
    public let float16: Bool
    public let quantize: Bool
    public let loraPath: String?

    public init(float16: Bool = true, quantize: Bool = false, loraPath: String? = nil) {
        self.float16 = float16
        self.quantize = quantize
        self.loraPath = loraPath
    }

    public var dType: DType { float16 ? .float16 : .float32 }
}

// MARK: - Evaluate Parameters

public struct EvaluateParameters: @unchecked Sendable {
    public var width: Int
    public var height: Int
    public var numInferenceSteps: Int
    public var guidance: Float
    public var seed: UInt64?
    public var prompt: String
    public var numTrainSteps: Int
    public var shiftSigmas: Bool
    public var sigmas: MLXArray

    public init(
        width: Int = 512, height: Int = 512,
        numInferenceSteps: Int = 4, guidance: Float = 4.0,
        seed: UInt64? = nil, prompt: String = "",
        numTrainSteps: Int = 1000, shiftSigmas: Bool = false
    ) {
        // Round to multiples of 16
        self.width = max(256, (width / 16) * 16)
        self.height = max(256, (height / 16) * 16)
        self.numInferenceSteps = numInferenceSteps
        self.guidance = guidance
        self.seed = seed
        self.prompt = prompt
        self.numTrainSteps = numTrainSteps
        self.shiftSigmas = shiftSigmas
        self.sigmas = Self.computeSigmas(
            steps: numInferenceSteps, shift: shiftSigmas, width: width, height: height
        )
    }

    /// Compute noise schedule sigmas using linear spacing with optional time-shift.
    public static func computeSigmas(steps: Int, shift: Bool, width: Int, height: Int) -> MLXArray {
        let base = MLX.linspace(Float32(1.0), Float32(1.0) / Float32(steps), count: steps)

        let shifted: MLXArray
        if shift {
            // Adaptive shift based on resolution (from Flux paper)
            let pixels = Float(width / 16) * Float(height / 16)
            let m: Float = (1.5 - 0.5) / (4096.0 - 256.0)
            let b: Float = 0.5 - m * 256.0
            let mu = m * pixels + b
            // Apply sigmoid-like shift
            let expMu = MLXArray(Foundation.exp(mu))
            shifted = expMu / (expMu + (1.0 / base) - 1.0)
        } else {
            shifted = base
        }

        // Append zero at end
        return MLX.concatenated([shifted, MLXArray([Float32(0.0)])])
    }
}

// MARK: - File Keys

public enum FileKey: String, Sendable {
    case transformer = "transformer"
    case vae = "vae"
    case textEncoder = "text_encoder"
    case textEncoder2 = "text_encoder_2"
    case tokenizer = "tokenizer"
    case tokenizer2 = "tokenizer_2"
    case modelIndex = "model_index"
}

// MARK: - Flux Configuration

public struct FluxConfiguration: Sendable {
    public let id: String
    public let defaultParameters: @Sendable () -> EvaluateParameters
    public let guidanceEmbeds: Bool

    /// File matching patterns for HuggingFace download
    public let filePatterns: [String]

    public init(
        id: String,
        guidanceEmbeds: Bool = false,
        defaultParameters: @escaping @Sendable () -> EvaluateParameters,
        filePatterns: [String] = Self.defaultFilePatterns
    ) {
        self.id = id
        self.guidanceEmbeds = guidanceEmbeds
        self.defaultParameters = defaultParameters
        self.filePatterns = filePatterns
    }

    public static let defaultFilePatterns: [String] = [
        "transformer/*.safetensors",
        "text_encoder/model.safetensors",
        "text_encoder_2/*.safetensors",
        "tokenizer/*",
        "tokenizer_2/*",
        "vae/diffusion_pytorch_model.safetensors",
        "model_index.json"
    ]

    // MARK: - Presets

    public static let flux1Schnell = FluxConfiguration(
        id: "black-forest-labs/FLUX.1-schnell",
        guidanceEmbeds: false,
        defaultParameters: {
            EvaluateParameters(
                width: 512, height: 512,
                numInferenceSteps: 4, guidance: 4.0,
                shiftSigmas: false
            )
        }
    )

    public static let flux1Dev = FluxConfiguration(
        id: "black-forest-labs/FLUX.1-dev",
        guidanceEmbeds: true,
        defaultParameters: {
            EvaluateParameters(
                width: 512, height: 512,
                numInferenceSteps: 20, guidance: 4.0,
                shiftSigmas: true
            )
        }
    )

    public static let flux1KontextDev = FluxConfiguration(
        id: "black-forest-labs/FLUX.1-Kontext-dev",
        guidanceEmbeds: true,
        defaultParameters: {
            EvaluateParameters(
                width: 512, height: 512,
                numInferenceSteps: 30, guidance: 4.0,
                shiftSigmas: true
            )
        }
    )

    // MARK: - Download

    /// Download model files from HuggingFace Hub.
    public func download(
        hub: HubApi,
        progressHandler: (@Sendable (Progress) -> Void)? = nil
    ) async throws {
        let repo = Hub.Repo(id: id)
        _ = try await hub.snapshot(from: repo, matching: filePatterns, progressHandler: progressHandler ?? { _ in })
    }

    // MARK: - Factory

    /// Create a TextToImageGenerator for this configuration.
    public func textToImageGenerator(
        hub: HubApi = HubApi(),
        configuration: LoadConfiguration = LoadConfiguration()
    ) throws -> TextToImageGenerator {
        try FluxPipeline.load(config: self, hub: hub, loadConfig: configuration)
    }
}
