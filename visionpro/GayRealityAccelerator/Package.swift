// swift-tools-version: 5.9
// Gay.jl Reality Accelerator for visionOS
// Tests Strong Parallelism Invariance through spatial Gedankenexperimente

import PackageDescription

let package = Package(
    name: "GayRealityAccelerator",
    platforms: [
        .visionOS(.v1),
        .iOS(.v17),
        .macOS(.v14)
    ],
    products: [
        .library(
            name: "GayColors",
            targets: ["GayColors"]),
        .library(
            name: "GedankenExperimente", 
            targets: ["GedankenExperimente"]),
    ],
    dependencies: [],
    targets: [
        .target(
            name: "GayColors",
            dependencies: [],
            path: "Sources/GayColors"),
        .target(
            name: "GedankenExperimente",
            dependencies: ["GayColors"],
            path: "Sources/GedankenExperimente"),
        .testTarget(
            name: "GayColorsTests",
            dependencies: ["GayColors"],
            path: "Tests/GayColorsTests"),
        .testTarget(
            name: "SPITests",
            dependencies: ["GayColors", "GedankenExperimente"],
            path: "Tests/SPITests"),
    ]
)
