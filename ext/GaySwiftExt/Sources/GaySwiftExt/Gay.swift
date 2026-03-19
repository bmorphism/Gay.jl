/// GaySwiftExt: Strong Parallelism Invariance for Swift
///
/// SPI guarantees: Same seed → Same colors, regardless of:
/// - Thread count
/// - Execution order
/// - Device architecture (M1/M2/M3/M4)
///
/// Core insight: O(1) hash-based color generation via splitmix64

import Foundation

/// A Gay color with Okhsl representation and RGB conversion
public struct GayColor: Hashable, Codable, Sendable {
    public let seed: UInt64
    public let h: Float  // Hue: 0-360
    public let s: Float  // Saturation: 0-1
    public let l: Float  // Lightness: 0-1
    
    public var rgb: (r: Float, g: Float, b: Float) {
        okhslToRGB(h: h, s: s, l: l)
    }
    
    public init(seed: UInt64) {
        self.seed = seed
        let state = splitmix64(seed)
        
        // SPI: deterministic color from seed
        self.h = Float(state & 0xFFFF) / 65535.0 * 360.0
        self.s = 0.5 + Float((state >> 16) & 0xFFFF) / 65535.0 * 0.4  // 0.5-0.9
        self.l = 0.35 + Float((state >> 32) & 0xFFFF) / 65535.0 * 0.4  // 0.35-0.75
    }
}

/// Cryptochrome color: bandwidth determines blueness (pigeon magnetoreception)
public struct CryptochromeColor: Hashable, Codable, Sendable {
    public let bandwidth: Float  // 0-1, higher = more blue
    public let blueIntensity: Float
    public let magneticPhase: Float  // 0-2π
    public let quantumCoherence: Float  // 0-1
    
    public var rgb: (r: Float, g: Float, b: Float) {
        // High bandwidth → blue, low → red/orange
        let r = (1 - bandwidth) * 0.9 + bandwidth * 0.2
        let g = 0.3 + 0.4 * quantumCoherence
        let b = bandwidth * 0.95 + (1 - bandwidth) * 0.1
        return (r: r, g: g, b: b)
    }
    
    public init(bandwidth: Float, magneticPhase: Float = 0) {
        self.bandwidth = min(max(bandwidth, 0), 1)
        self.blueIntensity = 1 / (1 + exp(-10 * (bandwidth - 0.5)))
        self.magneticPhase = magneticPhase
        self.quantumCoherence = 0.5 + 0.5 * cos(magneticPhase)
    }
}

/// Balanced ternary trit: -1, 0, +1
public typealias BalancedTrit = Int8

public enum Trit {
    public static let neg: BalancedTrit = -1
    public static let zero: BalancedTrit = 0
    public static let pos: BalancedTrit = 1
}

/// TritWord: 12 balanced trits (4 per RGB channel)
public struct TritWord: Hashable, Codable, Sendable {
    public var trits: [BalancedTrit]
    
    public init(_ trits: [BalancedTrit] = Array(repeating: 0, count: 12)) {
        self.trits = trits
    }
    
    public init(from color: GayColor) {
        let rgb = color.rgb
        var trits: [BalancedTrit] = []
        
        for channel in [rgb.r, rgb.g, rgb.b] {
            var val = min(max(channel, 0), 1)
            for _ in 0..<4 {
                val *= 3
                if val < 1 {
                    trits.append(Trit.neg)
                } else if val < 2 {
                    trits.append(Trit.zero)
                    val -= 1
                } else {
                    trits.append(Trit.pos)
                    val -= 2
                }
            }
        }
        
        self.trits = trits
    }
    
    /// Tritwise XOR (balanced ternary addition mod 3)
    public func xor(with other: TritWord) -> TritWord {
        assert(trits.count == other.trits.count)
        var result: [BalancedTrit] = []
        for i in 0..<trits.count {
            var sum = Int(trits[i]) + Int(other.trits[i])
            if sum > 1 { sum -= 3 }
            else if sum < -1 { sum += 3 }
            result.append(BalancedTrit(sum))
        }
        return TritWord(result)
    }
    
    /// String representation: T=-, 0=0, 1=+
    public var string: String {
        String(trits.map { t in
            t == Trit.neg ? Character("T") : t == Trit.zero ? Character("0") : Character("1")
        })
    }
}

// MARK: - Core SPI Functions

/// splitmix64: The SPI core hash function
/// O(1) deterministic mapping from seed to pseudo-random bits
@inline(__always)
public func splitmix64(_ seed: UInt64) -> UInt64 {
    var z = seed &+ 0x9e3779b97f4a7c15
    z = (z ^ (z >> 30)) &* 0xbf58476d1ce4e5b9
    z = (z ^ (z >> 27)) &* 0x94d049bb133111eb
    return z ^ (z >> 31)
}

/// O(1) color generation at any position
public func colorAt(seed: UInt64, position: UInt64) -> GayColor {
    let combined = seed ^ (position &* 0x9e3779b97f4a7c15)
    return GayColor(seed: combined)
}

/// O(1) next color from current seed
public func nextColor(from seed: UInt64) -> (color: GayColor, nextSeed: UInt64) {
    let next = splitmix64(seed)
    return (GayColor(seed: next), next)
}

/// Name to seed conversion (FNV-1a hash)
public func nameToSeed(_ name: String) -> UInt64 {
    var h: UInt64 = 0xcbf29ce484222325
    for byte in name.utf8 {
        h ^= UInt64(byte)
        h &*= 0x100000001b3
    }
    return h
}

// MARK: - Color Conversion

/// Okhsl to RGB conversion (simplified)
private func okhslToRGB(h: Float, s: Float, l: Float) -> (r: Float, g: Float, b: Float) {
    // Simplified HSL to RGB (Okhsl approximation)
    let c = (1 - abs(2 * l - 1)) * s
    let x = c * (1 - abs(fmod(h / 60, 2) - 1))
    let m = l - c / 2
    
    var r: Float, g: Float, b: Float
    
    if h < 60 {
        (r, g, b) = (c, x, 0)
    } else if h < 120 {
        (r, g, b) = (x, c, 0)
    } else if h < 180 {
        (r, g, b) = (0, c, x)
    } else if h < 240 {
        (r, g, b) = (0, x, c)
    } else if h < 300 {
        (r, g, b) = (x, 0, c)
    } else {
        (r, g, b) = (c, 0, x)
    }
    
    return (r: r + m, g: g + m, b: b + m)
}

// MARK: - Bandwidth Measurement

/// Measure color bandwidth (diversity) of a seed
public struct SeedBandwidth {
    public let name: String
    public let seed: UInt64
    public let bandwidthScore: Float
    public let uniqueRatio: Float
    public let hueCoverage: Float
    public let entropy: Float
    
    public init(name: String, samples: Int = 1000) {
        self.name = name
        self.seed = nameToSeed(name)
        
        var state = seed
        var bins = Set<UInt16>()
        var hueBuckets = Array(repeating: 0, count: 12)
        
        for _ in 0..<samples {
            state = splitmix64(state)
            let color = GayColor(seed: state)
            
            // 16-bit color bin
            let r = UInt16(min(max(color.rgb.r, 0), 1) * 15) & 0xF
            let g = UInt16(min(max(color.rgb.g, 0), 1) * 15) & 0xF
            let b = UInt16(min(max(color.rgb.b, 0), 1) * 15) & 0xF
            let bin = (r << 8) | (g << 4) | b
            bins.insert(bin)
            
            // Hue bucket
            let bucket = min(11, Int(color.h / 30))
            hueBuckets[bucket] += 1
        }
        
        self.uniqueRatio = Float(bins.count) / Float(samples)
        self.hueCoverage = Float(hueBuckets.filter { $0 > 0 }.count) / 12.0
        
        // Shannon entropy (simplified)
        let maxEntropy = log2(Float(samples))
        self.entropy = maxEntropy * uniqueRatio
        
        self.bandwidthScore = 0.3 * uniqueRatio + 0.3 * hueCoverage + 0.4 * (entropy / maxEntropy)
    }
}

// MARK: - Canonical Seeds

public enum GaySeed {
    public static let alice = nameToSeed("Alice")
    public static let bob = nameToSeed("Bob")
    public static let causality = nameToSeed("Causality")
    public static let entropy = nameToSeed("Entropy")
    public static let gay = nameToSeed("Gay")
    public static let emma = nameToSeed("Emma")
    public static let cat69: UInt64 = (69 << 48) | (0xFF << 32) | (0x00 << 16) | 0x01
}
