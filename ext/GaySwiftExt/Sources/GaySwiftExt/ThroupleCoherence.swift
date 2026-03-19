/// ThroupleCoherence: 2-Cell Validation and Witness Mass Calculation
///
/// Implements the categorical structure where each transformation
/// in a throuple is witnessed by the third entity.

import Foundation
import simd

// MARK: - 2-Cell Structure

/// A 2-cell representing a witnessed transformation
public struct TwoCell: Sendable, Hashable {
    public let sourceIndex: Int
    public let targetIndex: Int
    public let witnessIndex: Int
    public let validationMass: Float
    
    public init(source: Int, target: Int, witness: Int, mass: Float) {
        self.sourceIndex = source
        self.targetIndex = target
        self.witnessIndex = witness
        self.validationMass = mass
    }
}

// MARK: - Throuple Coherence Calculator

/// Calculates coherence metrics for a throuple
public struct ThroupleCoherenceCalculator: Sendable {
    
    /// Calculate all 2-cells for a throuple
    public static func calculateTwoCells(
        colors: (SIMD3<Float>, SIMD3<Float>, SIMD3<Float>)
    ) -> [TwoCell] {
        let colorArray = [colors.0, colors.1, colors.2]
        var cells: [TwoCell] = []
        
        for witness in 0..<3 {
            for source in 0..<3 {
                for target in 0..<3 {
                    if witness != source && source != target && witness != target {
                        let mass = validationMass(
                            source: colorArray[source],
                            target: colorArray[target],
                            witness: colorArray[witness]
                        )
                        cells.append(TwoCell(source: source, target: target, witness: witness, mass: mass))
                    }
                }
            }
        }
        
        return cells
    }
    
    /// Calculate validation mass for a single 2-cell
    public static func validationMass(
        source: SIMD3<Float>,
        target: SIMD3<Float>,
        witness: SIMD3<Float>
    ) -> Float {
        let midpoint = (source + target) / 2
        let distance = simd_length(witness - midpoint)
        return max(0, 1 - distance)
    }
    
    /// Calculate overall coherence from 2-cells
    public static func coherence(from cells: [TwoCell]) -> Float {
        guard !cells.isEmpty else { return 0 }
        return cells.map(\.validationMass).reduce(0, +) / Float(cells.count)
    }
    
    /// Check if throuple has sufficient mass for validation
    public static func hasSufficientMass(_ cells: [TwoCell], threshold: Float = 0.2) -> Bool {
        coherence(from: cells) >= threshold
    }
}

// MARK: - Coherence Result

/// Result of throuple coherence analysis
public struct CoherenceResult: Sendable {
    public let twoCells: [TwoCell]
    public let coherence: Float
    public let hasSufficientMass: Bool
    public let strongestWitness: Int?
    public let weakestWitness: Int?
    
    public init(colors: (SIMD3<Float>, SIMD3<Float>, SIMD3<Float>), threshold: Float = 0.2) {
        self.twoCells = ThroupleCoherenceCalculator.calculateTwoCells(colors: colors)
        self.coherence = ThroupleCoherenceCalculator.coherence(from: twoCells)
        self.hasSufficientMass = coherence >= threshold
        
        // Find strongest witness
        let witnessMasses = Dictionary(grouping: twoCells, by: \.witnessIndex)
            .mapValues { cells in cells.map(\.validationMass).reduce(0, +) }
        
        self.strongestWitness = witnessMasses.max(by: { $0.value < $1.value })?.key
        self.weakestWitness = witnessMasses.min(by: { $0.value < $1.value })?.key
    }
}

// MARK: - Throuple Builder

/// Builder for creating validated throuples
public struct ThroupleBuilder {
    private var entities: [OriginaryHue] = []
    
    public init() {}
    
    /// Add an entity to the throuple
    public mutating func add(_ name: String, axis: RotationAxis) -> Self {
        guard entities.count < 3 else { return self }
        entities.append(OriginaryHue(name: name, axis: axis))
        return self
    }
    
    /// Build the throuple if we have exactly 3 entities
    public func build() -> Throuple<OriginaryHue>? {
        guard entities.count == 3 else { return nil }
        return Throuple(entities[0], entities[1], entities[2])
    }
    
    /// Build with coherence analysis
    public func buildWithAnalysis() -> (Throuple<OriginaryHue>, CoherenceResult)? {
        guard let throuple = build() else { return nil }
        let result = CoherenceResult(colors: (
            throuple.entities.0.rgb,
            throuple.entities.1.rgb,
            throuple.entities.2.rgb
        ))
        return (throuple, result)
    }
}

// MARK: - Canonical Throuples

/// Pre-defined throuples for common use cases
public enum CanonicalThrouples {
    /// The Æther-Möbius-Ouroboros throuple
    public static var philosophical: Throuple<OriginaryHue> {
        Throuple(
            OriginaryHue(name: "Æther", axis: .x),
            OriginaryHue(name: "Möbius", axis: .y),
            OriginaryHue(name: "Ouroboros", axis: .z)
        )
    }
    
    /// RGB primary colors
    public static var primary: Throuple<OriginaryHue> {
        Throuple(
            OriginaryHue(name: "Red", axis: .x),
            OriginaryHue(name: "Green", axis: .y),
            OriginaryHue(name: "Blue", axis: .z)
        )
    }
    
    /// CMY secondary colors
    public static var secondary: Throuple<OriginaryHue> {
        Throuple(
            OriginaryHue(name: "Cyan", axis: .x),
            OriginaryHue(name: "Magenta", axis: .y),
            OriginaryHue(name: "Yellow", axis: .z)
        )
    }
    
    /// Alice-Bob-Carol agents
    public static var agents: Throuple<OriginaryHue> {
        Throuple(
            OriginaryHue(name: "Alice", axis: .x),
            OriginaryHue(name: "Bob", axis: .y),
            OriginaryHue(name: "Carol", axis: .z)
        )
    }
    
    /// Gay seed canonical
    public static var gay: Throuple<OriginaryHue> {
        Throuple(
            OriginaryHue(name: "Gay", axis: .x),
            OriginaryHue(name: "Splittable", axis: .y),
            OriginaryHue(name: "Chromatic", axis: .z)
        )
    }
}

// MARK: - Throuple Rotation Sequence

/// A sequence of rotations applied to a throuple
public struct ThroupleRotationSequence: Sendable {
    public let throuple: Throuple<OriginaryHue>
    public let steps: Int
    
    public init(throuple: Throuple<OriginaryHue>, steps: Int) {
        self.throuple = throuple
        self.steps = steps
    }
    
    /// Generate all rotation states
    public func states() -> [(Throuple<OriginaryHue>, CoherenceResult)] {
        var results: [(Throuple<OriginaryHue>, CoherenceResult)] = []
        var current = throuple
        
        for _ in 0..<steps {
            let rotated = current.rotateAll()
            current = Throuple(rotated.0, rotated.1, rotated.2)
            
            let coherence = CoherenceResult(colors: (
                current.entities.0.rgb,
                current.entities.1.rgb,
                current.entities.2.rgb
            ))
            
            results.append((current, coherence))
        }
        
        return results
    }
    
    /// Find the most coherent state in the sequence
    public func mostCoherentState() -> (Throuple<OriginaryHue>, CoherenceResult)? {
        states().max(by: { $0.1.coherence < $1.1.coherence })
    }
}
