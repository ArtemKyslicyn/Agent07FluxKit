// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "FluxKit",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "FluxKit", targets: ["FluxKit"])
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.25.4")),
        .package(url: "https://github.com/huggingface/swift-transformers", .upToNextMinor(from: "0.1.21")),
        .package(url: "https://github.com/apple/swift-log.git", from: "1.5.3")
    ],
    targets: [
        .target(
            name: "FluxKit",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Transformers", package: "swift-transformers"),
                .product(name: "Logging", package: "swift-log")
            ]
        ),
        .testTarget(name: "FluxKitTests", dependencies: ["FluxKit"])
    ],
    swiftLanguageModes: [.v6]
)
