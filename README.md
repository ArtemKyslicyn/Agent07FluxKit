# Agent07FluxKit

Swift package for Flux image generation on Apple Silicon (MLX).

Extracted from [Agent07](https://github.com/ArtemKyslicyn/Agent07) for reuse as a standalone library.

## Requirements

- macOS 14+
- Swift 6.0
- Apple Silicon (MLX)

## Installation

```swift
dependencies: [
    .package(url: "https://github.com/ArtemKyslicyn/Agent07FluxKit.git", from: "0.1.0")
]
```

Then add `FluxKit` to your target dependencies.

## Dependencies

- [mlx-swift](https://github.com/ml-explore/mlx-swift)
- [swift-transformers](https://github.com/huggingface/swift-transformers)
- [swift-log](https://github.com/apple/swift-log)

## License

MIT — see [LICENSE](LICENSE).
