import Testing
import Foundation
@testable import FluxKit

/// Integration tests — require a downloaded FLUX.1-schnell model.
struct IntegrationTests {

    private func modelAvailable() -> Bool {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let paths = [
            home.appendingPathComponent("Documents/huggingface/models/black-forest-labs/FLUX.1-schnell/transformer"),
            home.appendingPathComponent(".cache/huggingface/hub/models--black-forest-labs--FLUX.1-schnell/snapshots"),
        ]
        return paths.contains { FileManager.default.fileExists(atPath: $0.path) }
    }

    @Test func testPipelineLoad() throws {
        guard modelAvailable() else { print("[Test] SKIP"); return }

        let config = FluxConfiguration.flux1Schnell
        let generator = try config.textToImageGenerator(configuration: LoadConfiguration(float16: true, quantize: false))
        #expect(generator is FluxPipeline)
        print("[Test] Pipeline loaded OK")
    }

    @Test func testGenerateAndSavePNG() throws {
        guard modelAvailable() else { print("[Test] SKIP"); return }

        let config = FluxConfiguration.flux1Schnell
        let generator = try config.textToImageGenerator(configuration: LoadConfiguration(float16: true, quantize: false))

        var params = config.defaultParameters()
        params.prompt = "a red apple on a white table, photorealistic"
        params.width = 512
        params.height = 512
        params.numInferenceSteps = 4
        params.seed = 42

        print("[Test] Generating \(params.width)x\(params.height), \(params.numInferenceSteps) steps...")
        var denoiser = generator.generateLatents(parameters: params)
        var lastXt: MLXArray?
        var step = 0
        while let xt = denoiser.next() {
            step += 1
            lastXt = xt
            print("[Test] Step \(step)/\(params.numInferenceSteps)")
        }

        guard let latents = lastXt else { Issue.record("No latents"); return }
        #expect(step == 4)

        // Unpack latents
        let h16 = params.height / 16
        let w16 = params.width / 16
        let unpacked = latents
            .reshaped(1, h16, w16, 16, 2, 2)
            .transposed(0, 1, 4, 2, 5, 3)
            .reshaped(1, h16 * 2, w16 * 2, 16)

        // Decode
        let decoded = generator.decode(xt: unpacked)
        let imageArray = decoded.squeezed() // [H, W, 3] in [-1, 1]
        print("[Test] Raw range: min=\(imageArray.min().item(Float.self)), max=\(imageArray.max().item(Float.self))")

        // Normalize [-1,1] → [0,1]
        let normalized = (imageArray + 1.0) / 2.0
        let clipped = MLX.clip(normalized, min: 0, max: 1)
        let uint8 = (clipped * 255).asType(.uint8)

        let imgH = uint8.dim(0)
        let imgW = uint8.dim(1)
        let bytes = uint8.asArray(UInt8.self)
        print("[Test] Image: \(imgW)x\(imgH), \(bytes.count) bytes")

        // Quality check: not uniform
        let sample = Array(bytes.prefix(3000))
        let unique = Set(sample).count
        print("[Test] Unique pixel values in first 3000 bytes: \(unique)")
        #expect(unique > 20, "Image is too uniform (\(unique) unique values) — likely broken")

        // Check mean pixel value (should be around 100-160 for a natural image, not 0 or 255)
        let sum = sample.reduce(0) { $0 + Int($1) }
        let mean = Double(sum) / Double(sample.count)
        print("[Test] Mean pixel value: \(String(format: "%.1f", mean)) (expected ~80-180 for natural image)")
        #expect(mean > 30 && mean < 240, "Mean pixel \(mean) suggests blank image")

        print("[Test] PASSED")
    }
}
