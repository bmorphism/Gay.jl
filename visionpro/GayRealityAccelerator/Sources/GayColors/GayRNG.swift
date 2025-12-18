/// Gay.jl SplitMix64 RNG ported to Swift for visionOS
/// Strong Parallelism Invariance: same seed = same colors, always
///
/// "BE CNOT BE; DO CNOT DO" - the RNG state is quantum-like:
/// splitting creates entangled streams that can be "undecided" by re-seeding

import Foundation
import simd

/// Default seed: "gay_colo" as bytes (matches Gay.jl GAY_SEED)
public let GAY_SEED: UInt64 = 0x6761795f636f6c6f

/// SplitMix64 PRNG with tracking for SPI verification
public class GayRNG {
    public private(set) var state: UInt64
    public let seed: UInt64
    public private(set) var invocation: UInt64
    
    public init(seed: UInt64 = GAY_SEED) {
        self.seed = seed
        self.state = seed
        self.invocation = 0
    }
    
    /// Split the RNG - creates independent stream (SPI)
    /// Each split is deterministic: same sequence of splits = same results
    @discardableResult
    public func split() -> UInt64 {
        invocation += 1
        var z = state &+ 0x9e3779b97f4a7c15
        state = z
        z = (z ^ (z >> 30)) &* 0xbf58476d1ce4e5b9
        z = (z ^ (z >> 27)) &* 0x94d049bb133111eb
        return z ^ (z >> 31)
    }
    
    /// Get float in [0, 1)
    public func nextFloat() -> Float {
        Float(split()) / Float(UInt64.max)
    }
    
    /// Get double in [0, 1)
    public func nextDouble() -> Double {
        Double(split()) / Double(UInt64.max)
    }
    
    /// CNOT CNOT = undecide: reset to seed state
    public func undecide() {
        state = seed
        invocation = 0
    }
    
    /// Fork: create independent RNG for parallel execution (SPI)
    public func fork() -> GayRNG {
        let forkSeed = split()
        return GayRNG(seed: forkSeed)
    }
}

// MARK: - LCH Color Space

/// LCH color (Lightness, Chroma, Hue) - perceptually uniform
public struct LCH: Equatable, Hashable {
    public var L: Float  // 0-100
    public var C: Float  // 0-100  
    public var H: Float  // 0-360
    
    public init(L: Float, C: Float, H: Float) {
        self.L = L
        self.C = C
        self.H = H
    }
    
    /// Convert to RGB via Lab → XYZ → sRGB
    public func toRGB() -> RGB {
        let hRad = H * .pi / 180
        let a = C * cos(hRad)
        let b = C * sin(hRad)
        
        let fy = (L + 16) / 116
        let fx = a / 500 + fy
        let fz = fy - b / 200
        
        let delta: Float = 6.0 / 29.0
        let xn: Float = 0.95047
        let yn: Float = 1.0
        let zn: Float = 1.08883
        
        func finv(_ t: Float, _ n: Float) -> Float {
            t > delta ? n * pow(t, 3) : (t - 16/116) * 3 * delta * delta * n
        }
        
        let x = finv(fx, xn)
        let y = finv(fy, yn)
        let z = finv(fz, zn)
        
        // XYZ to sRGB
        var r = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
        var g = -0.969266 * x + 1.8760108 * y + 0.041556 * z
        var bl = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z
        
        func gamma(_ v: Float) -> Float {
            v > 0.0031308 ? 1.055 * pow(v, 1/2.4) - 0.055 : 12.92 * v
        }
        
        r = max(0, min(1, gamma(r)))
        g = max(0, min(1, gamma(g)))
        bl = max(0, min(1, gamma(bl)))
        
        return RGB(r: r, g: g, b: bl)
    }
}

/// RGB color (0-1 per channel)
public struct RGB: Equatable, Hashable {
    public var r: Float
    public var g: Float
    public var b: Float
    
    public init(r: Float, g: Float, b: Float) {
        self.r = r
        self.g = g
        self.b = b
    }
    
    /// Convert to SIMD for Metal/RealityKit
    public var simd: SIMD3<Float> {
        SIMD3(r, g, b)
    }
    
    /// Convert to SIMD4 with alpha
    public func simd4(alpha: Float = 1.0) -> SIMD4<Float> {
        SIMD4(r, g, b, alpha)
    }
    
    public var hex: String {
        String(format: "#%02X%02X%02X",
               Int(r * 255), Int(g * 255), Int(b * 255))
    }
}

// MARK: - Thread Palette

/// 6-color palette from seed (deterministic)
public struct ThreadPalette {
    public let seed: UInt64
    public let colors: [LCH]
    
    public init(seed: UInt64) {
        self.seed = seed
        var rng = GayRNG(seed: seed)
        var colors: [LCH] = []
        for _ in 0..<6 {
            colors.append(LCH(
                L: rng.nextFloat() * 100,
                C: rng.nextFloat() * 100,
                H: rng.nextFloat() * 360
            ))
        }
        self.colors = colors
    }
    
    /// Seed from UUID string (FNV-1a hash)
    public static func seedFromUUID(_ uuid: String) -> UInt64 {
        var hash: UInt64 = 0xcbf29ce484222325
        for byte in uuid.utf8 {
            hash ^= UInt64(byte)
            hash = hash &* 0x100000001b3
        }
        return hash
    }
    
    public var rgbColors: [RGB] {
        colors.map { $0.toRGB() }
    }
}

// MARK: - Magnetized S-Expression (GaySexpr)

/// Spin for magnetized sexp nodes
public enum Spin: Int {
    case down = -1
    case up = 1
    
    public var symbol: String {
        self == .up ? "⁺" : "⁻"
    }
    
    /// CNOT: flip if control is .up
    public func cnot(control: Spin) -> Spin {
        control == .up ? (self == .up ? .down : .up) : self
    }
    
    /// CNOT CNOT = identity (undecide)
    public func undecide() -> Spin {
        self // Already self after CNOT CNOT
    }
}

/// A magnetized S-expression node for Vision Pro visualization
public class GaySexpr {
    public let content: String?
    public let depth: Int
    public let position: Int
    public var color: RGB
    public var spin: Spin
    public var children: [GaySexpr]
    
    /// Has this node been "measured" (rendered)?
    public var measured: Bool = false
    
    public init(content: String?, depth: Int, position: Int, color: RGB, spin: Spin, children: [GaySexpr] = []) {
        self.content = content
        self.depth = depth
        self.position = position
        self.color = color
        self.spin = spin
        self.children = children
    }
    
    /// Expected spin from depth/position (ground state)
    public static func expectedSpin(depth: Int, position: Int) -> Spin {
        ((depth ^ position) & 1 == 0) ? .up : .down
    }
    
    /// Is this node wrongly magnetized?
    public var isWronglyMagnetized: Bool {
        spin != GaySexpr.expectedSpin(depth: depth, position: position)
    }
    
    /// Chi-gong: correct magnetization if not yet measured
    public func chiGong() -> Bool {
        guard !measured else { return false }
        if isWronglyMagnetized {
            spin = GaySexpr.expectedSpin(depth: depth, position: position)
            return true
        }
        return false
    }
    
    /// Measure (collapse) - makes undecision impossible
    public func measure() {
        measured = true
        children.forEach { $0.measure() }
    }
    
    /// Total magnetization ⟨M⟩ = Σσ/N
    public func magnetization() -> Float {
        var totalSpin = 0
        var count = 0
        
        func walk(_ node: GaySexpr) {
            totalSpin += node.spin.rawValue
            count += 1
            node.children.forEach { walk($0) }
        }
        
        walk(self)
        return Float(totalSpin) / Float(count)
    }
}

/// Build magnetized sexp from expression
public func magnetizedSexpr(
    from expression: Any,
    seed: UInt64 = 0xDEADBEEF,
    depth: Int = 0,
    positionCounter: inout Int,
    nDepths: Int = 8
) -> GaySexpr {
    let pos = positionCounter
    positionCounter += 1
    
    // Get color from interleaved streams
    let streamIdx = (depth + 1) % nDepths
    let streamSeed = seed ^ UInt64(streamIdx * 0x9e3779b97f4a7c15)
    var rng = GayRNG(seed: streamSeed)
    for _ in 0..<pos { _ = rng.split() }
    
    let color = LCH(
        L: rng.nextFloat() * 100,
        C: rng.nextFloat() * 100,
        H: rng.nextFloat() * 360
    ).toRGB()
    
    let spin = GaySexpr.expectedSpin(depth: depth, position: pos)
    
    if let array = expression as? [Any] {
        let children = array.map { child -> GaySexpr in
            magnetizedSexpr(from: child, seed: seed, depth: depth + 1,
                           positionCounter: &positionCounter, nDepths: nDepths)
        }
        return GaySexpr(content: nil, depth: depth, position: pos,
                       color: color, spin: spin, children: children)
    } else {
        return GaySexpr(content: String(describing: expression),
                       depth: depth, position: pos,
                       color: color, spin: spin)
    }
}
