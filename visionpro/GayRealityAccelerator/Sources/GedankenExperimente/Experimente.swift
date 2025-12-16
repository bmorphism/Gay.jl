/// Gedankenexperimente for Vision Pro Reality Accelerator
/// Testing Strong Parallelism Invariance through spatial visualization
///
/// "Gedankudukkkkexperimente" - thought experiments with DuckDB-style
/// deterministic verification of parallel color streams

import Foundation
import GayColors

// MARK: - Experiment Protocol

public protocol Gedankenexperiment {
    var name: String { get }
    var description: String { get }
    
    /// Run the experiment, return SPI verification result
    func run() -> ExperimentResult
}

public struct ExperimentResult {
    public let passed: Bool
    public let message: String
    public let colors: [RGB]
    public let magnetization: Float?
    
    public init(passed: Bool, message: String, colors: [RGB] = [], magnetization: Float? = nil) {
        self.passed = passed
        self.message = message
        self.colors = colors
        self.magnetization = magnetization
    }
}

// MARK: - Experiment 1: BE CNOT BE

/// Visualize existence as quantum entanglement
/// |BE⟩ ⊗ |world⟩ → |BE⟩ ⊗ |world ⊕ BE⟩
public class BeCnotBeExperiment: Gedankenexperiment {
    public let name = "BE CNOT BE"
    public let description = "To BE is to entangle with the world"
    
    public init() {}
    
    public func run() -> ExperimentResult {
        // Initial state: |BE⟩|00⟩ + |NOT BE⟩|00⟩ (superposition)
        var worldState: [(be: Spin, world: Spin)] = [
            (.up, .down),    // BE, world initially down
            (.down, .down),  // NOT BE, world initially down
        ]
        
        // Apply CNOT (BE controls world)
        for i in 0..<worldState.count {
            worldState[i].world = worldState[i].world.cnot(control: worldState[i].be)
        }
        
        // After CNOT: |BE⟩|up⟩ (flipped), |NOT BE⟩|down⟩ (unchanged)
        let entangled = worldState[0].be == .up && worldState[0].world == .up
        let separable = worldState[1].be == .down && worldState[1].world == .down
        
        // Generate visualization colors
        let palette = ThreadPalette(seed: 0x424543_4E4F54) // "BE CNOT"
        
        return ExperimentResult(
            passed: entangled && separable,
            message: entangled && separable 
                ? "BE entangles with world; NOT BE leaves world unchanged"
                : "CNOT failed",
            colors: palette.rgbColors
        )
    }
}

// MARK: - Experiment 2: DO CNOT DO (Wu-wei)

/// Action vs non-action in quantum terms
public class DoCnotDoExperiment: Gedankenexperiment {
    public let name = "DO CNOT DO"
    public let description = "Wu-wei: the sage in superposition of action/non-action"
    
    public init() {}
    
    public func run() -> ExperimentResult {
        // The sage: superposition of DO and NOT DO
        // Represented by both states existing
        
        let doState: Spin = .up
        let notDoState: Spin = .down
        var outcome: Spin = .down
        
        // In DO branch, CNOT flips outcome
        let outcomeIfDo = outcome.cnot(control: doState)
        // In NOT DO branch, outcome unchanged
        let outcomeIfNotDo = outcome.cnot(control: notDoState)
        
        // Superposition: both outcomes exist until measurement
        let wuWei = outcomeIfDo != outcomeIfNotDo
        
        let palette = ThreadPalette(seed: 0x57555F574549) // "WU_WEI"
        
        return ExperimentResult(
            passed: wuWei,
            message: "Outcome uncertain until observed: DO→\(outcomeIfDo.symbol), NOT DO→\(outcomeIfNotDo.symbol)",
            colors: palette.rgbColors
        )
    }
}

// MARK: - Experiment 3: CNOT CNOT Undecision

/// CNOT · CNOT = I : the decision was never made
public class UndecisionExperiment: Gedankenexperiment {
    public let name = "CNOT CNOT Undecision"
    public let description = "Can we undecide? Only if we haven't measured"
    
    public init() {}
    
    public func run() -> ExperimentResult {
        // Create two entangled qubits
        var control: Spin = .up
        var target: Spin = .down
        let originalTarget = target
        
        // CNOT 1: decide (entangle)
        target = target.cnot(control: control)
        let afterDecide = target
        
        // CNOT 2: undecide (disentangle)
        target = target.cnot(control: control)
        let afterUndecide = target
        
        // Should return to original
        let undecided = afterUndecide == originalTarget
        
        // But what if we measured in between?
        var measuredTarget: Spin = .down
        measuredTarget = measuredTarget.cnot(control: control)
        let measured = measuredTarget // "observation" happens here
        // CNOT again
        measuredTarget = measuredTarget.cnot(control: control)
        
        // Classically it returns, but quantumly the measurement changed everything
        let palette = ThreadPalette(seed: 0x554E444543494445) // "UNDECIDE"
        
        return ExperimentResult(
            passed: undecided,
            message: """
                Original: \(originalTarget.symbol)
                After CNOT: \(afterDecide.symbol)  
                After CNOT CNOT: \(afterUndecide.symbol)
                Undecision \(undecided ? "succeeded" : "failed")
                """,
            colors: palette.rgbColors
        )
    }
}

// MARK: - Experiment 4: SPI Parallel Streams

/// Test Strong Parallelism Invariance: parallel execution = sequential execution
public class SPIExperiment: Gedankenexperiment {
    public let name = "Strong Parallelism Invariance"
    public let description = "Same seed, same colors, regardless of execution order"
    
    let seed: UInt64
    let nStreams: Int
    
    public init(seed: UInt64 = GAY_SEED, nStreams: Int = 4) {
        self.seed = seed
        self.nStreams = nStreams
    }
    
    public func run() -> ExperimentResult {
        // Sequential generation
        var sequentialColors: [[RGB]] = []
        for streamIdx in 0..<nStreams {
            let streamSeed = seed ^ UInt64(streamIdx &* 0x9e3779b97f4a7c15)
            let palette = ThreadPalette(seed: streamSeed)
            sequentialColors.append(palette.rgbColors)
        }
        
        // "Parallel" generation (reversed order to simulate different execution)
        var parallelColors: [[RGB]] = Array(repeating: [], count: nStreams)
        for streamIdx in (0..<nStreams).reversed() {
            let streamSeed = seed ^ UInt64(streamIdx &* 0x9e3779b97f4a7c15)
            let palette = ThreadPalette(seed: streamSeed)
            parallelColors[streamIdx] = palette.rgbColors
        }
        
        // Verify SPI: same colors regardless of order
        var allMatch = true
        for i in 0..<nStreams {
            for j in 0..<6 {
                if sequentialColors[i][j] != parallelColors[i][j] {
                    allMatch = false
                    break
                }
            }
        }
        
        return ExperimentResult(
            passed: allMatch,
            message: allMatch 
                ? "SPI verified: \(nStreams) streams × 6 colors match"
                : "SPI VIOLATION: execution order affected colors",
            colors: sequentialColors.flatMap { $0 }
        )
    }
}

// MARK: - Experiment 5: Chi-Gong Rebalancing

/// Correct wrongly magnetized sexp nodes
public class ChiGongExperiment: Gedankenexperiment {
    public let name = "Chi-Gong Rebalancing"
    public let description = "Cultivate qi to correct wrongly magnetized S-expressions"
    
    public init() {}
    
    public func run() -> ExperimentResult {
        // Create a sexp tree with some wrong spins
        var pos = 0
        let sexp = magnetizedSexpr(
            from: ["defn", "fib", ["n"], 
                   ["if", ["<", "n", 2], "n",
                    ["+", ["fib", ["-", "n", 1]], 
                          ["fib", ["-", "n", 2]]]]],
            seed: 0x4649424F4E41434349, // "FIBONACCI"
            depth: 0,
            positionCounter: &pos
        )
        
        // Intentionally flip some spins to create "wrong" magnetization
        func flipSome(_ node: GaySexpr, probability: Float) {
            if Float.random(in: 0...1) < probability {
                node.spin = node.spin == .up ? .down : .up
            }
            node.children.forEach { flipSome($0, probability: probability) }
        }
        flipSome(sexp, probability: 0.3)
        
        let beforeMag = sexp.magnetization()
        
        // Count wrong nodes before
        func countWrong(_ node: GaySexpr) -> Int {
            var count = node.isWronglyMagnetized ? 1 : 0
            node.children.forEach { count += countWrong($0) }
            return count
        }
        let wrongBefore = countWrong(sexp)
        
        // Apply chi-gong
        func applyChiGong(_ node: GaySexpr) -> Int {
            var corrected = node.chiGong() ? 1 : 0
            node.children.forEach { corrected += applyChiGong($0) }
            return corrected
        }
        let corrected = applyChiGong(sexp)
        
        let afterMag = sexp.magnetization()
        let wrongAfter = countWrong(sexp)
        
        let palette = ThreadPalette(seed: 0x434849474F4E47) // "CHIGONG"
        
        return ExperimentResult(
            passed: wrongAfter == 0,
            message: """
                Before: \(wrongBefore) wrong, ⟨M⟩=\(String(format: "%.2f", beforeMag))
                Chi-gong corrected: \(corrected) nodes
                After: \(wrongAfter) wrong, ⟨M⟩=\(String(format: "%.2f", afterMag))
                """,
            colors: palette.rgbColors,
            magnetization: afterMag
        )
    }
}

// MARK: - Experiment 6: Galois Connection

/// Test α(γ(c)) ⊆ c for color abstraction/concretization
public class GaloisExperiment: Gedankenexperiment {
    public let name = "Galois Connection"
    public let description = "α: Message → Color, γ: Color → Message"
    
    public init() {}
    
    public func run() -> ExperimentResult {
        // α (abstraction): map message to color class
        func alpha(_ message: String) -> Int {
            // Hash to color index 0-5
            var hash: UInt64 = 0
            for byte in message.utf8 {
                hash = hash &* 31 &+ UInt64(byte)
            }
            return Int(hash % 6)
        }
        
        // γ (concretization): get all messages with given color
        let messages = ["hello", "world", "foo", "bar", "baz", "qux"]
        func gamma(_ colorIdx: Int) -> [String] {
            messages.filter { alpha($0) == colorIdx }
        }
        
        // Verify closure: α(γ(c)) ⊆ c
        var closureHolds = true
        for c in 0..<6 {
            let concretized = gamma(c)
            for msg in concretized {
                if alpha(msg) != c {
                    closureHolds = false
                }
            }
        }
        
        let palette = ThreadPalette(seed: 0x47414C4F4953) // "GALOIS"
        
        return ExperimentResult(
            passed: closureHolds,
            message: closureHolds 
                ? "Galois closure verified: α(γ(c)) ⊆ c for all color classes"
                : "Galois connection broken!",
            colors: palette.rgbColors
        )
    }
}

// MARK: - Run All Experiments

public func runAllGedankenexperimente() -> [ExperimentResult] {
    let experiments: [Gedankenexperiment] = [
        BeCnotBeExperiment(),
        DoCnotDoExperiment(),
        UndecisionExperiment(),
        SPIExperiment(),
        ChiGongExperiment(),
        GaloisExperiment(),
    ]
    
    return experiments.map { exp in
        print("Running: \(exp.name)")
        let result = exp.run()
        print("  \(result.passed ? "✓" : "✗") \(result.message)")
        return result
    }
}
