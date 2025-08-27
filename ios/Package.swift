// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "AINewsKit",
    platforms: [
        .iOS(.v17)
    ],
    products: [
        .library(name: "AINewsKit", targets: ["AINewsKit"])
    ],
    targets: [
        .target(
            name: "AINewsKit",
            path: "Sources"
        )
    ]
)

