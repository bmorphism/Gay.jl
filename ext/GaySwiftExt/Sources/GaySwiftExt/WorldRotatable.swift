/// WorldRotatable: SO(3) Color Space Rotations with Celibacy Preservation
///
/// A WorldRotatable entity maintains its chromatic identity through rotations
/// while optionally allowing transformation based on celibacy factor.
///
/// Celibacy = 1.0: Color fully preserved (no mixing)
/// Celibacy = 0.0: Color fully transforms with rotation

import Foundation
import simd

// MARK: - WorldRotatable Protocol

/// Protocol for entities that can be rotated in color space
public protocol WorldRotatable: Sendable {
    /// Unique originary seed for SPI color generation
    var seed: UInt64 { get }
    
    /// RGB color tuple
    var rgb: SIMD3<Float> { get }
    
    /// Celibacy factor: how much color is preserved through rotation
    var celibacy: Float { get }
    
    /// Apply rotation and return new rotated entity
    func rotated(by rotation: simd_float3x3) -> Self
    
    /// Rotation axis contribution (x, y, or z)
    var rotationAxis: RotationAxis { get }
    
    /// Rotation angle based on celibacy
    var rotationAngle: Float { get }
}

public enum RotationAxis: Int, Sendable, CaseIterable {
    case x = 0
    case y = 1
    case z = 2
    
    public var vector: SIMD3<Float> {
        switch self {
        case .x: return SIMD3<Float>(1, 0, 0)
        case .y: return SIMD3<Float>(0, 1, 0)
        case .z: return SIMD3<Float>(0, 0, 1)
        }
    }
}

// MARK: - Rotation Matrix Construction

public struct RotationMatrix {
    /// Create rotation matrix around axis by angle
    public static func rotation(axis: RotationAxis, angle: Float) -> simd_float3x3 {
        let c = cos(angle)
        let s = sin(angle)
        
        switch axis {
        case .x:
            return simd_float3x3(rows: [
                SIMD3<Float>(1, 0, 0),
                SIMD3<Float>(0, c, -s),
                SIMD3<Float>(0, s, c)
            ])
        case .y:
            return simd_float3x3(rows: [
                SIMD3<Float>(c, 0, s),
                SIMD3<Float>(0, 1, 0),
                SIMD3<Float>(-s, 0, c)
            ])
        case .z:
            return simd_float3x3(rows: [
                SIMD3<Float>(c, -s, 0),
                SIMD3<Float>(s, c, 0),
                SIMD3<Float>(0, 0, 1)
            ])
        }
    }
    
    /// Compose multiple rotations (right to left)
    public static func compose(_ rotations: [simd_float3x3]) -> simd_float3x3 {
        rotations.reduce(matrix_identity_float3x3) { $1 * $0 }
    }
    
    /// Verify rotation matrix is in SO(3)
    public static func isValidRotation(_ m: simd_float3x3) -> Bool {
        let det = simd_determinant(m)
        return abs(det - 1.0) < 0.001
    }
}

// MARK: - OriginaryHue (WorldRotatable Implementation)

/// A unique chromatic identity that can be rotated in color space
public struct OriginaryHue: WorldRotatable, Hashable, Codable, Sendable {
    public let seed: UInt64
    public let name: String
    public let rgb: SIMD3<Float>
    public let celibacy: Float
    public let rotationAxis: RotationAxis
    
    public var rotationAngle: Float {
        celibacy * .pi * 2 / 3  // Max 120° for throuple symmetry
    }
    
    public init(name: String, axis: RotationAxis = .x) {
        self.name = name
        self.seed = Self.nameToSeed(name)
        self.rgb = Self.seedToRGB(self.seed)
        self.celibacy = Self.seedToCelibacy(self.seed)
        self.rotationAxis = axis
    }
    
    public init(seed: UInt64, name: String, rgb: SIMD3<Float>, celibacy: Float, axis: RotationAxis) {
        self.seed = seed
        self.name = name
        self.rgb = rgb
        self.celibacy = celibacy
        self.rotationAxis = axis
    }
    
    public func rotated(by rotation: simd_float3x3) -> OriginaryHue {
        let rotatedRGB = rotation * rgb
        // Apply celibacy: blend original and rotated
        let preserved = celibacy * rgb + (1 - celibacy) * rotatedRGB
        let clamped = simd_clamp(preserved, SIMD3<Float>(0, 0, 0), SIMD3<Float>(1, 1, 1))
        
        return OriginaryHue(
            seed: seed,
            name: name,
            rgb: clamped,
            celibacy: celibacy,
            axis: rotationAxis
        )
    }
    
    // MARK: - SPI Functions
    
    private static func nameToSeed(_ name: String) -> UInt64 {
        var h: UInt64 = 0xcbf29ce484222325
        for byte in name.utf8 {
            h ^= UInt64(byte)
            h &*= 0x100000001b3
        }
        return h
    }
    
    private static func splitmix64(_ seed: UInt64) -> UInt64 {
        var z = seed &+ 0x9e3779b97f4a7c15
        z = (z ^ (z >> 30)) &* 0xbf58476d1ce4e5b9
        z = (z ^ (z >> 27)) &* 0x94d049bb133111eb
        return z ^ (z >> 31)
    }
    
    private static func seedToRGB(_ seed: UInt64) -> SIMD3<Float> {
        let state = splitmix64(seed)
        let r = Float(state & 0xFFFF) / 65535.0
        let g = Float((state >> 16) & 0xFFFF) / 65535.0
        let b = Float((state >> 32) & 0xFFFF) / 65535.0
        return SIMD3<Float>(r, g, b)
    }
    
    private static func seedToCelibacy(_ seed: UInt64) -> Float {
        let phase = Float(seed & 0xFFFF) / 65535.0 * .pi * 2
        return 0.5 + 0.5 * cos(phase)
    }
}

// MARK: - Throuple (Three WorldRotatable Entities)

/// A throuple of three WorldRotatable entities with 2-cell validation
public struct Throuple<T: WorldRotatable>: Sendable {
    public let entities: (T, T, T)
    public let combinedRotation: simd_float3x3
    public let coherence: Float
    
    public init(_ a: T, _ b: T, _ c: T) {
        self.entities = (a, b, c)
        
        // Compose rotations from each entity
        let rotations = [
            RotationMatrix.rotation(axis: a.rotationAxis, angle: a.rotationAngle),
            RotationMatrix.rotation(axis: b.rotationAxis, angle: b.rotationAngle),
            RotationMatrix.rotation(axis: c.rotationAxis, angle: c.rotationAngle)
        ]
        self.combinedRotation = RotationMatrix.compose(rotations)
        
        // Calculate coherence via 2-cell validation
        self.coherence = Self.calculateCoherence(a, b, c)
    }
    
    /// Rotate all entities by the combined throuple rotation
    public func rotateAll() -> (T, T, T) {
        (
            entities.0.rotated(by: combinedRotation),
            entities.1.rotated(by: combinedRotation),
            entities.2.rotated(by: combinedRotation)
        )
    }
    
    /// 2-cell validation: each transformation witnessed by the third
    private static func calculateCoherence(_ a: T, _ b: T, _ c: T) -> Float {
        // Validation mass = 1 - distance from witness to midpoint of transformation
        func validationMass(source: SIMD3<Float>, target: SIMD3<Float>, witness: SIMD3<Float>) -> Float {
            let mid = (source + target) / 2
            let dist = simd_length(witness - mid)
            return max(0, 1 - dist)
        }
        
        let masses: [Float] = [
            validationMass(source: a.rgb, target: b.rgb, witness: c.rgb),
            validationMass(source: b.rgb, target: a.rgb, witness: c.rgb),
            validationMass(source: a.rgb, target: c.rgb, witness: b.rgb),
            validationMass(source: c.rgb, target: a.rgb, witness: b.rgb),
            validationMass(source: b.rgb, target: c.rgb, witness: a.rgb),
            validationMass(source: c.rgb, target: b.rgb, witness: a.rgb)
        ]
        
        return masses.reduce(0, +) / Float(masses.count)
    }
    
    /// Check if rotation matrix is valid SO(3)
    public var isValidRotation: Bool {
        RotationMatrix.isValidRotation(combinedRotation)
    }
}

// MARK: - CelibateWorldRotator

/// A world rotator that preserves color identity based on celibacy
public struct CelibateWorldRotator: Sendable {
    public let rotation: simd_float3x3
    public let determinant: Float
    
    public init(entities: [any WorldRotatable]) {
        let rotations = entities.map { entity in
            RotationMatrix.rotation(axis: entity.rotationAxis, angle: entity.rotationAngle)
        }
        self.rotation = RotationMatrix.compose(rotations)
        self.determinant = simd_determinant(rotation)
    }
    
    public init(rotation: simd_float3x3) {
        self.rotation = rotation
        self.determinant = simd_determinant(rotation)
    }
    
    /// Apply rotation to a color
    public func rotate(_ color: SIMD3<Float>, celibacy: Float) -> SIMD3<Float> {
        let rotated = rotation * color
        let preserved = celibacy * color + (1 - celibacy) * rotated
        return simd_clamp(preserved, SIMD3<Float>(0, 0, 0), SIMD3<Float>(1, 1, 1))
    }
    
    /// Compose with another rotator
    public func composed(with other: CelibateWorldRotator) -> CelibateWorldRotator {
        CelibateWorldRotator(rotation: other.rotation * rotation)
    }
    
    /// Inverse rotation
    public var inverse: CelibateWorldRotator {
        CelibateWorldRotator(rotation: rotation.transpose)
    }
}
