# FluxKit examples

Each subdirectory is a standalone SPM package that depends on FluxKit via a
local path. They are not built as part of the root `FluxKit` package — open or
build them from their own directory.

| Demo | What it does |
|---|---|
| [`FluxKitDemo/`](FluxKitDemo) | CLI: text prompt → PNG via FluxKit. Supports Schnell/Dev/Kontext, INT4 quantization, HuggingFace weight download. |

## Running the CLI demo

```bash
cd Examples/FluxKitDemo

# First run — download Flux Schnell weights (~24 GB) and generate
swift run flux-demo \
    --download \
    --prompt "a red apple on a white table, photorealistic" \
    --out apple.png

# Subsequent runs — weights are cached
swift run flux-demo --prompt "a forest at dawn" --seed 7 --out forest.png

# Dev model, 20 steps, custom guidance
swift run flux-demo --model dev --steps 20 --guidance 4 --out dev.png

# Quantize to save ~12 GB RAM (transformer + T5 → INT4)
swift run flux-demo --quantize --out quantized.png
```

Run `swift run flux-demo --help` for the full flag list.

## Switching the demo to a published FluxKit tag

By default the demo consumes the parent FluxKit working copy via a path
dependency. To pin against a published version instead, edit
`FluxKitDemo/Package.swift`:

```swift
// Replace:
.package(path: "../..")
// With:
.package(url: "https://github.com/ArtemKyslicyn/Agent07FluxKit.git", from: "0.2.0")
```
