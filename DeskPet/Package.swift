// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "RealPet",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "RealPet",
            path: ".",
            exclude: ["Package.swift"]
        )
    ]
)
