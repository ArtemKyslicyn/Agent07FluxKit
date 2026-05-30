//
//  FluxKitDemo.swift
//  FluxKitDemo
//
//  Command-line image generation demo for FluxKit.
//
//  Run: swift run flux-demo --prompt "a red apple on a white table"
//

import ArgumentParser
import CoreGraphics
import Foundation
import FluxKit
import Hub
import ImageIO
import UniformTypeIdentifiers

@main
struct FluxKitDemo: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "flux-demo",
        abstract: "Generate an image from a text prompt using FluxKit on Apple Silicon (MLX).",
        discussion: """
        First run will download model weights from HuggingFace (~24 GB for Schnell,
        ~50 GB for Dev). Pass --download to fetch before generation; subsequent runs
        reuse the cached weights.

        Examples:
          swift run flux-demo --download --prompt "a red apple on a white table"
          swift run flux-demo --model dev --steps 20 --guidance 4 --out out.png
          swift run flux-demo --width 768 --height 512 --seed 1234 --quantize
        """
    )

    @Option(name: .shortAndLong, help: "Text prompt describing the image.")
    var prompt: String = "a red apple on a white table, photorealistic"

    @Option(help: "Model preset: schnell | dev | kontext")
    var model: String = "schnell"

    @Option(name: .shortAndLong, help: "Override number of denoising steps.")
    var steps: Int?

    @Option(name: .shortAndLong, help: "Override classifier-free guidance scale.")
    var guidance: Float?

    @Option(help: "Image width in pixels (rounded down to multiple of 16).")
    var width: Int = 512

    @Option(help: "Image height in pixels (rounded down to multiple of 16).")
    var height: Int = 512

    @Option(help: "Random seed for reproducibility.")
    var seed: Int = 42

    @Flag(help: "Apply INT4 quantization to transformer + T5 (saves ~12 GB RAM).")
    var quantize: Bool = false

    @Flag(help: "Download model weights from HuggingFace before generating.")
    var download: Bool = false

    @Option(name: .shortAndLong, help: "Output PNG path.")
    var out: String = "flux-demo.png"

    mutating func run() async throws {
        let config = try resolveConfiguration(model)
        let hub = HubApi()

        if download {
            print("[flux-demo] downloading \(config.id) from HuggingFace…")
            let throttle = ProgressThrottle()
            try await config.download(hub: hub) { progress in
                let pct = Int(progress.fractionCompleted * 100)
                if throttle.shouldEmit(pct) {
                    print("[flux-demo]   download: \(pct)%")
                }
            }
            print("[flux-demo] download complete")
        }

        print("[flux-demo] loading pipeline (quantize=\(quantize))…")
        let loadConfig = LoadConfiguration(float16: true, quantize: quantize)
        let generator: TextToImageGenerator
        do {
            generator = try config.textToImageGenerator(hub: hub, configuration: loadConfig)
        } catch let error as FluxKitError {
            // The most common cause: model not downloaded yet. Print a hint
            // instead of dumping a stack trace.
            FileHandle.standardError.write(Data("[flux-demo] \(error.localizedDescription)\n".utf8))
            FileHandle.standardError.write(Data("[flux-demo] Rerun with --download to fetch weights.\n".utf8))
            throw ExitCode.failure
        }

        var params = config.defaultParameters()
        params.prompt = prompt
        params.width = width
        params.height = height
        if let steps { params.numInferenceSteps = steps }
        if let guidance { params.guidance = guidance }
        params.seed = UInt64(bitPattern: Int64(seed))
        // Recompute sigmas because any of (steps, width, height, shift) may have changed
        // after the default factory built them.
        params.sigmas = EvaluateParameters.computeSigmas(
            steps: params.numInferenceSteps,
            shift: params.shiftSigmas,
            width: params.width,
            height: params.height
        )

        print("[flux-demo] generating \(params.width)×\(params.height), \(params.numInferenceSteps) steps, seed=\(seed)")
        print("[flux-demo] prompt: \(prompt)")

        let started = Date()
        var iterator = generator.generateLatents(parameters: params)
        var lastLatent: MLXArray?
        while let latent = iterator.next() {
            let elapsed = Date().timeIntervalSince(started)
            print("[flux-demo]   step \(iterator.i)/\(params.numInferenceSteps) (\(String(format: "%.1fs", elapsed)))")
            lastLatent = latent
        }
        guard let latents = lastLatent else {
            throw RuntimeError.generationProducedNoLatents
        }

        let pixels = try decodeAndNormalize(latents: latents, generator: generator, params: params)
        let pixelHeight = pixels.dim(0)
        let pixelWidth = pixels.dim(1)
        let channels = pixels.dim(2)
        guard channels == 3 else {
            throw RuntimeError.unexpectedChannelCount(channels)
        }
        let bytes: [UInt8] = pixels.asArray(UInt8.self)
        let outputURL = URL(fileURLWithPath: out)
        try writePNG(rgb: bytes, width: pixelWidth, height: pixelHeight, to: outputURL)

        let totalElapsed = Date().timeIntervalSince(started)
        print("[flux-demo] saved \(outputURL.path) (\(pixelWidth)×\(pixelHeight), total \(String(format: "%.1fs", totalElapsed)))")
    }
}

// MARK: - Configuration

private func resolveConfiguration(_ name: String) throws -> FluxConfiguration {
    switch name.lowercased() {
    case "schnell", "flux1-schnell", "flux.1-schnell":
        return .flux1Schnell
    case "dev", "flux1-dev", "flux.1-dev":
        return .flux1Dev
    case "kontext", "kontext-dev", "flux1-kontext-dev", "flux.1-kontext-dev":
        return .flux1KontextDev
    default:
        throw ValidationError("Unknown model '\(name)'. Use one of: schnell, dev, kontext.")
    }
}

// MARK: - Latent decode

private func decodeAndNormalize(
    latents: MLXArray,
    generator: TextToImageGenerator,
    params: EvaluateParameters
) throws -> MLXArray {
    // Unpack [1, h16*w16, 64] → [1, h16*2, w16*2, 16] for VAE input
    let h16 = params.height / 16
    let w16 = params.width / 16
    let unpacked = latents
        .reshaped(1, h16, w16, 16, 2, 2)
        .transposed(0, 1, 4, 2, 5, 3)
        .reshaped(1, h16 * 2, w16 * 2, 16)
    let decoded = generator.decode(xt: unpacked)

    // Normalize from [-1, 1] → [0, 255] UInt8 → [H, W, 3]
    let normalized = MLX.clip((decoded.squeezed() + 1.0) / 2.0, min: 0, max: 1)
    return (normalized * 255).asType(.uint8)
}

// MARK: - PNG output

/// Write an interleaved 8-bit RGB buffer to a PNG file via ImageIO.
private func writePNG(rgb bytes: [UInt8], width: Int, height: Int, to url: URL) throws {
    guard bytes.count == width * height * 3 else {
        throw RuntimeError.byteCountMismatch(expected: width * height * 3, got: bytes.count)
    }

    // CGImage with 24-bit RGB requires the data block to live as long as the
    // CGDataProvider, so wrap a copy into Data and hand its CFData to the provider.
    let data = Data(bytes)
    guard let provider = CGDataProvider(data: data as CFData) else {
        throw RuntimeError.pngEncodingFailed("CGDataProvider init failed")
    }

    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)

    guard let image = CGImage(
        width: width,
        height: height,
        bitsPerComponent: 8,
        bitsPerPixel: 24,
        bytesPerRow: width * 3,
        space: colorSpace,
        bitmapInfo: bitmapInfo,
        provider: provider,
        decode: nil,
        shouldInterpolate: false,
        intent: .defaultIntent
    ) else {
        // Some CG configurations reject 24-bit RGB; pad to RGBA and retry.
        try writePNGViaRGBA(rgb: bytes, width: width, height: height, to: url)
        return
    }

    guard let dest = CGImageDestinationCreateWithURL(
        url as CFURL,
        UTType.png.identifier as CFString,
        1,
        nil
    ) else {
        throw RuntimeError.pngEncodingFailed("CGImageDestination init failed")
    }

    CGImageDestinationAddImage(dest, image, nil)
    guard CGImageDestinationFinalize(dest) else {
        throw RuntimeError.pngEncodingFailed("CGImageDestinationFinalize failed")
    }
}

/// Fallback that pads RGB → RGBA before encoding.
private func writePNGViaRGBA(rgb bytes: [UInt8], width: Int, height: Int, to url: URL) throws {
    var rgba = [UInt8]()
    rgba.reserveCapacity(width * height * 4)
    var i = 0
    while i < bytes.count {
        rgba.append(bytes[i])
        rgba.append(bytes[i + 1])
        rgba.append(bytes[i + 2])
        rgba.append(0xFF)
        i += 3
    }

    let data = Data(rgba)
    guard let provider = CGDataProvider(data: data as CFData) else {
        throw RuntimeError.pngEncodingFailed("CGDataProvider init failed (RGBA fallback)")
    }
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue)
    guard let image = CGImage(
        width: width,
        height: height,
        bitsPerComponent: 8,
        bitsPerPixel: 32,
        bytesPerRow: width * 4,
        space: colorSpace,
        bitmapInfo: bitmapInfo,
        provider: provider,
        decode: nil,
        shouldInterpolate: false,
        intent: .defaultIntent
    ) else {
        throw RuntimeError.pngEncodingFailed("CGImage init failed (RGBA fallback)")
    }
    guard let dest = CGImageDestinationCreateWithURL(
        url as CFURL,
        UTType.png.identifier as CFString,
        1,
        nil
    ) else {
        throw RuntimeError.pngEncodingFailed("CGImageDestination init failed (RGBA fallback)")
    }
    CGImageDestinationAddImage(dest, image, nil)
    guard CGImageDestinationFinalize(dest) else {
        throw RuntimeError.pngEncodingFailed("CGImageDestinationFinalize failed (RGBA fallback)")
    }
}

// MARK: - Progress throttle

/// Suppress duplicate progress lines from the HuggingFace download callback.
/// HF calls the closure many times per second; we want at most one print per
/// 5% increment. The callback is `@Sendable` and HF serializes calls, so an
/// `@unchecked Sendable` reference type with an `os_unfair_lock` is overkill —
/// a serial-access counter is enough.
private final class ProgressThrottle: @unchecked Sendable {
    private var lastEmitted: Int = -1

    func shouldEmit(_ pct: Int) -> Bool {
        guard pct != lastEmitted, pct % 5 == 0 else { return false }
        lastEmitted = pct
        return true
    }
}

// MARK: - Errors

enum RuntimeError: LocalizedError {
    case generationProducedNoLatents
    case unexpectedChannelCount(Int)
    case byteCountMismatch(expected: Int, got: Int)
    case pngEncodingFailed(String)

    var errorDescription: String? {
        switch self {
        case .generationProducedNoLatents:
            return "Denoise iterator yielded no latents — generation parameters likely invalid."
        case .unexpectedChannelCount(let c):
            return "Expected 3-channel image, decoded \(c) channels."
        case .byteCountMismatch(let exp, let got):
            return "Pixel buffer size mismatch: expected \(exp) bytes, got \(got)."
        case .pngEncodingFailed(let msg):
            return "PNG encoding failed: \(msg)"
        }
    }
}
