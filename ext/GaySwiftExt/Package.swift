// swift-tools-version: 5.9
// GaySwiftExt: Swift Package for SPI-guaranteed color generation with R1 reasoning

import PackageDescription

let package = Package(
    name: "GaySwiftExt",
    platforms: [
        .iOS(.v15),
        .macOS(.v12),
        .tvOS(.v15),
        .watchOS(.v8),
        .visionOS(.v1)
    ],
    products: [
        .library(
            name: "GaySwiftExt",
            targets: ["GaySwiftExt"]
        ),
        .library(
            name: "GayMetal",
            targets: ["GayMetal"]
        ),
        .library(
            name: "GayR1Bridge",
            targets: ["GayR1Bridge"]
        )
    ],
    dependencies: [
        // AIProxy for secure R1 API access
        .package(url: "https://github.com/lzell/AIProxySwift.git", from: "1.0.0"),
    ],
    targets: [
        // Core SPI color generation
        .target(
            name: "GaySwiftExt",
            dependencies: [],
            path: "Sources/GaySwiftExt"
        ),
        
        // Metal GPU acceleration for Apple Silicon
        .target(
            name: "GayMetal",
            dependencies: ["GaySwiftExt"],
            path: "Sources/GayMetal",
            resources: [
                .process("Shaders")
            ]
        ),
        
        // DeepSeek R1 reasoning bridge
        .target(
            name: "GayR1Bridge",
            dependencies: [
                "GaySwiftExt",
                .product(name: "AIProxy", package: "AIProxySwift")
            ],
            path: "Sources/GayR1Bridge"
        ),
        
        // Tests
        .testTarget(
            name: "GaySwiftExtTests",
            dependencies: ["GaySwiftExt", "GayMetal"]
        )
    ]
)
