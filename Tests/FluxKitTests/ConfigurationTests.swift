import Testing
import Foundation
@testable import FluxKit

struct ConfigurationTests {

    @Test func testSchnellConfig() {
        let config = FluxConfiguration.flux1Schnell
        #expect(config.id == "black-forest-labs/FLUX.1-schnell")
        #expect(!config.guidanceEmbeds)
        let params = config.defaultParameters()
        #expect(params.numInferenceSteps == 4)
    }

    @Test func testDevConfig() {
        let config = FluxConfiguration.flux1Dev
        #expect(config.id == "black-forest-labs/FLUX.1-dev")
        #expect(config.guidanceEmbeds)
        let params = config.defaultParameters()
        #expect(params.numInferenceSteps == 20)
        #expect(params.shiftSigmas)
    }

    @Test func testKontextConfig() {
        let config = FluxConfiguration.flux1KontextDev
        #expect(config.guidanceEmbeds)
        let params = config.defaultParameters()
        #expect(params.numInferenceSteps == 30)
    }

    @Test func testDimensionRounding() {
        let params = EvaluateParameters(width: 769, height: 513)
        #expect(params.width == 768)
        #expect(params.height == 512)
    }

    @Test func testMinimumDimension() {
        let params = EvaluateParameters(width: 100, height: 50)
        #expect(params.width == 256)
        #expect(params.height == 256)
    }

    @Test func testSigmaCount() {
        let params = EvaluateParameters(numInferenceSteps: 4)
        // Should have steps+1 sigmas (last one is 0)
        #expect(params.sigmas.dim(0) == 5)
    }

    @Test func testSigmaRange() {
        let params = EvaluateParameters(numInferenceSteps: 4, shiftSigmas: false)
        let sigmas = params.sigmas.asArray(Float.self)
        #expect(sigmas.first ?? 0 > 0.9)  // First sigma close to 1.0
        #expect(sigmas.last == 0.0)        // Last sigma is 0
    }

    @Test func testLoadConfiguration() {
        let lc = LoadConfiguration(float16: true, quantize: false)
        #expect(lc.dType == .float16)

        let lc32 = LoadConfiguration(float16: false)
        #expect(lc32.dType == .float32)
    }
}
