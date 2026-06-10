# Agent07FluxKit

Swift package — clean-room implementation of Flux text-to-image generation on Apple Silicon, built on top of [MLX](https://github.com/ml-explore/mlx-swift).

> **Status: experimental / under active development.** The pipeline runs end-to-end and produces images, but output quality is still being tuned. See [Known issues](#known-issues--limitations) and [Development history](#development-history) before using in production. Logging is wired through `swift-log` and hot-path probes are gated behind `FLUXKIT_DEBUG` — see [Logging](#logging).

---

## About

FluxKit is the **public MLX FLUX engine** of the [Agent07](https://github.com/ArtemKyslicyn/Agent07) open-core ecosystem — a macOS app for visual DAG-based AI agent pipelines. The image-generation layer was extracted out of the app so it can be reused, audited, and improved independently of the closed product.

Inside Agent07, FluxKit is consumed by the **private `Agent07FluxServices` package**, which adapts this engine behind the app's internal `FluxServicing` protocol (actor isolation, `AsyncStream` progress, cancellation, and app wiring all live there — none of it leaks into this package). FluxKit itself stays dependency-light and app-agnostic: it knows about MLX, HuggingFace Hub, and `swift-log`, and nothing about Agent07.

This is a **clean-room** Swift/MLX reimplementation of the FLUX.1 inference architecture — not a port of the PyTorch reference. See [Development history](#development-history) for the path from "produces noise" to "produces images."

---

## Table of contents

- [About](#about)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Example CLI](#example-cli)
- [Supported models](#supported-models)
- [Architecture](#architecture)
- [Public API surface](#public-api-surface)
- [Logging](#logging)
- [Known issues & limitations](#known-issues--limitations)
- [Development history](#development-history)
- [Roadmap](#roadmap)
- [Testing](#testing)
- [Dependencies](#dependencies)
- [License](#license)

---

## Requirements

| | |
|---|---|
| Platform | macOS 14+ |
| Language | Swift 6.0 (strict concurrency on) |
| Hardware | Apple Silicon (M1/M2/M3/M4). MLX has no CUDA fallback. |
| Memory  | 16 GB RAM minimum, 32 GB recommended for FLUX.1-dev / Kontext |
| Disk | ~24 GB per model (FLUX.1-schnell), ~50 GB for FLUX.1-dev |

---

## Installation

```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/ArtemKyslicyn/Agent07FluxKit.git", from: "0.1.0")
],
targets: [
    .target(name: "YourTarget", dependencies: [
        .product(name: "FluxKit", package: "Agent07FluxKit")
    ])
]
```

---

## Quick start

```swift
import FluxKit
import Hub

// 1. Pick a model (Schnell = 4 steps, fastest)
let config = FluxConfiguration.flux1Schnell

// 2. Download weights from HuggingFace (first run only, ~24 GB)
let hub = HubApi()
try await config.download(hub: hub) { progress in
    print("Download: \(progress.fractionCompleted * 100)%")
}

// 3. Build the generator
let generator = try config.textToImageGenerator(
    hub: hub,
    configuration: LoadConfiguration(float16: true, quantize: false)
)

// 4. Configure generation
var params = config.defaultParameters()
params.prompt = "a red apple on a white table, photorealistic"
params.width = 512
params.height = 512
params.numInferenceSteps = 4   // Schnell needs only 4 steps
params.seed = 42

// 5. Run the denoise loop — yields a latent per step
var denoiser = generator.generateLatents(parameters: params)
var lastLatent: MLXArray?
while let latent = denoiser.next() {
    lastLatent = latent
}
guard let latents = lastLatent else { return }

// 6. Unpack + VAE decode → [H, W, 3] in [-1, 1]
let h16 = params.height / 16, w16 = params.width / 16
let unpacked = latents
    .reshaped(1, h16, w16, 16, 2, 2)
    .transposed(0, 1, 4, 2, 5, 3)
    .reshaped(1, h16 * 2, w16 * 2, 16)
let decoded = generator.decode(xt: unpacked)

// 7. Normalize to [0, 1] and convert to UInt8 for image encoding
let normalized = MLX.clip((decoded.squeezed() + 1.0) / 2.0, min: 0, max: 1)
let uint8 = (normalized * 255).asType(.uint8)
// → feed into NSBitmapImageRep / CGImage as you wish
```

The `Tests/FluxKitTests/IntegrationTests.swift` test covers the full pipeline with an actual model on disk; use it as a reference for the latent-unpack and decode math.

---

## Example CLI

A runnable CLI lives under [`Examples/FluxKitDemo`](Examples/FluxKitDemo) — a standalone SPM package that consumes FluxKit via a local path dependency and writes a PNG to disk.

```bash
cd Examples/FluxKitDemo

# First run — download Flux Schnell weights (~24 GB) and generate
swift run flux-demo \
    --download \
    --prompt "a red apple on a white table, photorealistic" \
    --out apple.png

# Subsequent runs reuse the cached weights
swift run flux-demo --prompt "a forest at dawn" --seed 7 --out forest.png

# Dev model, 20 steps, INT4 quantization
swift run flux-demo --model dev --steps 20 --quantize --out dev.png
```

`swift run flux-demo --help` for the full flag list. See [`Examples/README.md`](Examples/README.md) for the full set of demos and instructions for pointing them at a published FluxKit tag instead of the working copy.

---

## Supported models

| Preset | HuggingFace ID | Steps | CFG | Notes |
|---|---|---|---|---|
| `FluxConfiguration.flux1Schnell` | `black-forest-labs/FLUX.1-schnell` | 4 | 4.0 | Apache 2.0, fastest. **Most tested.** |
| `FluxConfiguration.flux1Dev` | `black-forest-labs/FLUX.1-dev` | 20 | 4.0 | Non-commercial license. `shiftSigmas=true`. |
| `FluxConfiguration.flux1KontextDev` | `black-forest-labs/FLUX.1-Kontext-dev` | 30 | 4.0 | In-context conditioning. **Not integration-tested.** |

All presets use the same `MultiModalDiffusionTransformer`, T5 + CLIP text encoders, and VAE. The differences are in step count, guidance handling (`guidanceEmbeds`), and the sigma schedule.

---

## Architecture

```
ImageGenerator (protocol) ◄── TextToImageGenerator (protocol, Pipeline.swift)
└── FluxPipeline (concrete)
    ├── T5Encoder           (TextEncoders.swift)  — prompt tokens → embeddings
    ├── CLIPEncoder         (TextEncoders.swift)  — pooled prompt vector
    ├── MultiModalDiffusionTransformer (Transformer.swift)
    │   ├── JointAttention blocks (19 layers)  — image + text co-attention with RoPE
    │   └── SingleBlockAttention   (38 layers)  — image-only with RoPE
    └── VAE                 (VAE.swift)         — decode latents → pixels

DenoiseIterator (struct) — Euler step loop, yields one latent per inference step
```

Source files (`Sources/FluxKit/`):

| File | Purpose |
|---|---|
| `FluxKit.swift` | Umbrella, re-exports MLX, public API summary |
| `Configuration.swift` | `FluxConfiguration`, `LoadConfiguration`, `EvaluateParameters`, sigma schedule, presets |
| `Pipeline.swift` | `TextToImageGenerator` protocol, `FluxPipeline`, `DenoiseIterator`, position-ID generation |
| `Transformer.swift` | `MultiModalDiffusionTransformer`, `JointAttention`, `SingleBlockAttention`, gate/modulation logic |
| `Embeddings.swift` | RoPE (`ropeFreqs`, `applyRope`), timestep sinusoidal projection |
| `TextEncoders.swift` | `T5Encoder`, `CLIPEncoder` with manual additive causal mask |
| `Tokenizer.swift` | T5 + CLIP tokenizer wrappers around `swift-transformers` |
| `VAE.swift` | Variational autoencoder for latent → image |
| `WeightLoading.swift` | safetensors → MLX weight loading with key remapping |
| `Quantization.swift` | Optional INT4 quantization of transformer + T5 (VAE/CLIP stay fp16) |

---

## Public API surface

The library exposes a small surface. Most callers only need:

- `FluxConfiguration` — model presets and factory entry point (`textToImageGenerator(hub:configuration:)`, `download(hub:progressHandler:)`, `defaultParameters()`)
- `LoadConfiguration` — `float16` / `quantize` / `loraPath` toggles
- `EvaluateParameters` — `width`, `height`, `numInferenceSteps`, `guidance`, `seed`, `prompt`, `sigmas`; plus the static `computeSigmas(steps:shift:width:height:)` helper
- `ImageGenerator` (protocol) — base capability: `decode(xt:) -> MLXArray`
- `TextToImageGenerator` (protocol, refines `ImageGenerator`) — adds `generateLatents(parameters:) -> DenoiseIterator` and `conditionText(prompt:) -> (MLXArray, MLXArray)`
- `FluxPipeline` (concrete, conforms to `TextToImageGenerator`) — the engine returned by the factory; also exposes `static load(...)` for callers that want to construct it directly
- `DenoiseIterator` — `Sequence & IteratorProtocol`, yields one `MLXArray` latent per inference step (`i` exposes the current step index)
- `FluxKitError` (enum, `LocalizedError`) — thrown for missing weights, unreadable configs, etc.; surface its `localizedDescription` to users
- `MultiModalDiffusionTransformer` — exposed for advanced users wanting to swap in custom blocks

Lower-level building blocks (`T5Encoder`, `CLIPEncoder`, `VAE`, `EmbedND`, the `applyRope` / `timestepProjection` functions, the `loadTransformerWeights` / `loadVAEWeights` / `loadT5EncoderWeights` / `loadCLIPEncoderWeights` loaders, `quantizeFluxPipeline`, and the tokenizer helpers) are also `public` for assembling a custom pipeline, but most callers should stay on `FluxConfiguration` + `TextToImageGenerator`.

Re-exports: `MLX` is `@_exported` from `FluxKit`, so callers don't need a separate `import MLX`.

---

## Logging

FluxKit emits its logs through [`swift-log`](https://github.com/apple/swift-log). Each source file uses a labelled `Logger`:

| Label | What it covers |
|---|---|
| `FluxKit.Pipeline` | Pipeline load milestones, MLX memory snapshots, per-step magnitude probes (debug only). |
| `FluxKit.Transformer` | Joint/single block input/output magnitudes (debug only). |
| `FluxKit.WeightLoading` | Tensor counts, unmatched keys, shape mismatches. |
| `FluxKit.Quantization` | Quantization summary. |

To capture or filter logs in your host app, bootstrap the logging system before calling FluxKit:

```swift
import Logging

LoggingSystem.bootstrap { label in
    var handler = StreamLogHandler.standardOutput(label: label)
    handler.logLevel = .info  // .debug for verbose, .warning to silence
    return handler
}
```

**Hot-path probes.** Per-step and per-block magnitude probes call `.item()` on MLX arrays — a synchronous GPU read that stalls the pipeline. They are gated behind the `FLUXKIT_DEBUG` compile flag and are entirely excluded from release builds. To re-enable them for debugging:

```bash
swift build -Xswiftc -DFLUXKIT_DEBUG
```

---

## Known issues & limitations

### Active, will affect users

1. **Output quality still being tuned.** The pipeline produces structured images, but fidelity vs. the reference PyTorch implementation has not been formally benchmarked. Magnitude growth across denoise steps is verified linear (see commit `f2f1e146`), but pixel-level correctness has not been bit-exact validated.
2. **`EvaluateParameters` is `@unchecked Sendable`** because `MLXArray` is not `Sendable`. Treat as a value type; do not mutate concurrently.
3. **No streaming back-pressure.** `DenoiseIterator.next()` blocks for one step's worth of compute (hundreds of ms to seconds depending on resolution/quantization). Wrap in `Task.detached` if you need async cancellation.

### Won't affect most users, but worth knowing

4. **`FluxConfiguration.flux1KontextDev` has no integration test.** Only `flux1Schnell` is covered end-to-end in `IntegrationTests.testGenerateAndSavePNG`.
5. **LoRA support is a stub.** `LoadConfiguration.loraPath` is accepted but not wired through weight loading yet.
6. **VAE and CLIP are never quantized** even when `LoadConfiguration.quantize=true`. Only the transformer and T5 encoder are quantized — by design, since VAE/CLIP quantization broke output in earlier experiments (`f370b319`).
7. **Sigma schedule** uses linear spacing with optional `shiftSigmas` from the Flux paper. No support yet for DPM-Solver, Karras, or custom schedulers.

---

## Development history

FluxKit is a **clean-room** Swift/MLX reimplementation, not a port. The path from "produces noise" to "produces images" went through several substantial fixes, each landed as a single commit on the parent Agent07 repo before extraction:

| Commit | Issue | Fix |
|---|---|---|
| `b4906f6a` | Initial rewrite | First working build; output was uniform noise. |
| `f370b319` | Weights silently dropped (966 of 1156), CLIP attention broken | `@ModuleInfo key:` added to every layer init; manual additive causal mask in CLIP (MLXFast doesn't support additive masks); VAE/CLIP excluded from quantization; `OutLinear` wrapper for FF output. |
| `10d2011a` | Garbage output even with correct weights | Fixed `AdaLayerNormContinuous` SiLU activation order; reverted latent unpack to correct `[C, Ph, Pw]` axis order. |
| `1a612490` | Position-invariant output — magnitudes ~500-1000× expected | **Root cause:** `JointAttention` and `SingleBlockAttention` accepted `ropeEmb` but never applied it to Q/K. Added explicit `applyRope` calls. |
| `f2f1e146` | Couldn't tell if fixes converged | Added per-step and per-block magnitude logging; verified velocity magnitude grows linearly, not exponentially. |
| `42224dc7` | RoPE rotation-matrix approach was slow + memory-heavy | Refactored RoPE to direct `cos`/`sin` (computed once via `ropeFreqs`, applied as `x * cos + rotate_half(x) * sin`). Drops 4D rotation-matrix tensor. |
| `04fa08ec` | Memory safety on first load; broken `patch-deps` build | Memory safety check using available RAM before loading; build fix. |
| `7285e872` | Actor isolation issues in consumer adapter | Hardened actor isolation, AsyncStream lifecycle, JSON-RPC framing (in `Agent07FluxServices`, not FluxKit core, but related work). |
| `2442d0c3` | Extraction to standalone repo | This repo. |
| `v0.2.0`   | Standalone-consumer ergonomics | `fluxLog` removed; all logging goes through `swift-log`. Per-step / per-block magnitude probes gated behind `#if FLUXKIT_DEBUG` so the hot path no longer does synchronous GPU reads in release builds. `Examples/FluxKitDemo` CLI added. |

For context, the project memory has a detailed write-up of the RoPE diagnosis at the time of the bug — see [`memory/flux_debug_status.md`](https://github.com/ArtemKyslicyn/Agent07) in the parent repo.

---

## Roadmap

Sorted roughly by priority. Contributions welcome.

- [x] **Replace `fluxLog` with `swift-log`** — landed in 0.2.0. All FluxKit modules now log through labelled `Logging.Logger` instances; the Agent07-specific file writer is gone.
- [x] **Strip per-step / per-block logging from default builds** — landed in 0.2.0. Magnitude probes are gated behind `#if FLUXKIT_DEBUG`; release builds drop them entirely.
- [x] **CI** — `.github/workflows/ci.yml` builds + runs unit tests on `macos-15` for every push and PR; `release.yml` cuts a Release on `v*` tags.
- [x] **Example CLI** — [`Examples/FluxKitDemo`](Examples/FluxKitDemo) ships a runnable `swift run flux-demo`.
- [ ] **Bit-exact validation** against PyTorch reference on a fixed seed + prompt. Without this, "the output looks plausible" is the best we can claim.
- [ ] **Integration test for Kontext-dev.**
- [ ] **LoRA loading** — wire `LoadConfiguration.loraPath` into `WeightLoading`.
- [ ] **Make `MLXArray`-carrying value types properly `Sendable`** (or document the threading contract more loudly).
- [ ] **Schedulers** beyond linear + sigma-shift (DPM-Solver, Karras).
- [ ] **Batch inference** — currently one image per pipeline call.

---

## Testing

```bash
swift test --package-path .
```

CI runs the suite through **`xcodebuild`** rather than `swift test` (see `.github/workflows/ci.yml`):

```bash
xcodebuild test \
    -scheme FluxKit \
    -destination 'platform=macOS,arch=arm64' \
    -configuration Debug \
    -parallel-testing-enabled NO \
    CODE_SIGNING_ALLOWED=NO
```

Two MLX-specific reasons for this:

- **`xcodebuild` bundles MLX's metallib correctly.** On hosted runners, `swift test` fails with `Failed to load the default metallib` because its resource-lookup path differs from Xcode's.
- **`-parallel-testing-enabled NO`** keeps Metal device init single-threaded — mlx-swift crashes when multiple test workers try to load the metallib at once.

CI runs build (debug) → unit tests (xcodebuild, serial) → build (release smoke check) on `macos-15` / Xcode 16.4 for every push to `main` and every PR (docs-only changes are path-ignored). `release.yml` cuts a GitHub Release on `v*` tags.

Tests fall into two tiers:

**Unit tests** — always run, no model required:
- `ConfigurationTests.swift` — preset sigmas, dimension rounding, sigma count/range, config load
- `GenerationTest.swift` — `testConfigSigmasCorrect`, `testModelPathResolution`

**Integration tests** — auto-skip unless a model is present on disk:
- `IntegrationTests.testPipelineLoad` — loads FLUX.1-schnell end-to-end
- `IntegrationTests.testGenerateAndSavePNG` — generates a 512×512, 4-step image and asserts uniqueness / mean-pixel sanity (not bit-exact)

Skip logic looks for the model under either:

- `~/Documents/huggingface/models/black-forest-labs/FLUX.1-schnell/transformer/`
- `~/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-schnell/snapshots/`

To run integration tests, download via `config.download(hub:progressHandler:)` once (~24 GB).

---

## Dependencies

| Package | Version | Why |
|---|---|---|
| [ml-explore/mlx-swift](https://github.com/ml-explore/mlx-swift) | `.upToNextMinor(from: "0.25.4")` | Tensor ops, NN modules, fast attention |
| [huggingface/swift-transformers](https://github.com/huggingface/swift-transformers) | `.upToNextMinor(from: "0.1.21")` | T5 + CLIP tokenizers, HuggingFace Hub downloads |
| [apple/swift-log](https://github.com/apple/swift-log) | `from: "1.5.3"` | Logging facade — every FluxKit module logs through a labelled `Logging.Logger` (see [Logging](#logging)) |

MLX is Apple Silicon only by design — there is no CUDA / Linux build target.

---

## License

MIT — see [LICENSE](LICENSE). The original PyTorch Flux models from Black Forest Labs carry their own licenses (Apache 2.0 for Schnell, non-commercial research for Dev and Kontext-dev). This package only implements the inference architecture; you are responsible for complying with the model weight licenses you choose to download.
