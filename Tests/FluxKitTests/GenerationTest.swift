import Testing
import Foundation
@testable import FluxKit

/// Standalone generation test that can be run via: swift test --filter GenerationTest
/// Requires: 1) downloaded FLUX.1-schnell model 2) Metal GPU available (Xcode, not CLI)
struct GenerationTest {

    @Test func testConfigSigmasCorrect() {
        // This test doesn't need GPU
        let params = EvaluateParameters(
            width: 256, height: 256,
            numInferenceSteps: 4, shiftSigmas: false
        )
        let sigmas = params.sigmas.asArray(Float.self)
        #expect(sigmas.count == 5, "4 steps + 1 zero = 5 sigmas")
        #expect(sigmas.last == 0.0, "Last sigma must be 0")
        #expect(sigmas[0] > 0.9, "First sigma ~1.0")
        print("[GenerationTest] Sigmas: \(sigmas)")
    }

    @Test func testModelPathResolution() {
        // Check that HubApi finds the model in Documents
        let home = FileManager.default.homeDirectoryForCurrentUser
        let modelPath = home.appendingPathComponent("Documents/huggingface/models/black-forest-labs/FLUX.1-schnell")
        let hasTransformer = FileManager.default.fileExists(atPath: modelPath.appendingPathComponent("transformer").path)
        let hasVAE = FileManager.default.fileExists(atPath: modelPath.appendingPathComponent("vae").path)
        let hasTokenizer = FileManager.default.fileExists(atPath: modelPath.appendingPathComponent("tokenizer").path)
        let hasT5 = FileManager.default.fileExists(atPath: modelPath.appendingPathComponent("text_encoder_2").path)

        print("[GenerationTest] Model at: \(modelPath.path)")
        print("[GenerationTest] transformer: \(hasTransformer), vae: \(hasVAE), tokenizer: \(hasTokenizer), t5: \(hasT5)")

        if hasTransformer {
            #expect(hasVAE, "VAE must exist if transformer exists")
            #expect(hasTokenizer, "Tokenizer must exist")
            #expect(hasT5, "T5 encoder must exist")
            print("[GenerationTest] All model components present")
        } else {
            print("[GenerationTest] Model not downloaded — skipping file checks")
        }
    }
}
