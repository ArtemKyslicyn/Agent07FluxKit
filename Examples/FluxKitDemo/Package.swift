// swift-tools-version: 6.0
//
//  FluxKitDemo — command-line demo for FluxKit.
//
//  Standalone SPM package consuming FluxKit via a local path dependency.
//  `swift run flux-demo --prompt "…"` from this directory.
//

import PackageDescription

let package = Package(
    name: "FluxKitDemo",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "flux-demo", targets: ["FluxKitDemo"])
    ],
    dependencies: [
        // Consume the parent FluxKit package via path so the demo always
        // tracks the working copy, not a published tag. Switch to `url:` +
        // `from: "0.2.0"` to consume the released package instead.
        .package(path: "../.."),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.3.0")
    ],
    targets: [
        .executableTarget(
            name: "FluxKitDemo",
            dependencies: [
                // Path dependency: SPM derives the package name from the directory
                // ("Agent07FluxKit"), not the `name:` field in the parent Package.swift.
                .product(name: "FluxKit", package: "Agent07FluxKit"),
                .product(name: "ArgumentParser", package: "swift-argument-parser")
            ]
        )
    ],
    swiftLanguageModes: [.v6]
)
