/// GayR1Bridge: DeepSeek R1 reasoning integration for chromatic decision-making
///
/// Uses AIProxy for secure API access to R1's chain-of-thought reasoning.
/// R1 helps with:
/// - Aperiodic tile placement optimization
/// - QECC decoder suggestions
/// - Strange loop interpretation
/// - Chromatic bandwidth maximization strategies

import Foundation
import AIProxy

// MARK: - R1 Bridge Configuration

public struct R1BridgeConfig {
    public let partialKey: String
    public let serviceURL: String
    public let model: String
    public let streaming: Bool
    public let maxTokens: Int
    
    public init(
        partialKey: String,
        serviceURL: String,
        model: String = "deepseek-ai/DeepSeek-R1",
        streaming: Bool = true,
        maxTokens: Int = 4096
    ) {
        self.partialKey = partialKey
        self.serviceURL = serviceURL
        self.model = model
        self.streaming = streaming
        self.maxTokens = maxTokens
    }
}

// MARK: - R1 Response Types

public struct R1Response {
    public let reasoning: String  // Chain-of-thought reasoning
    public let answer: String     // Final answer
    public let tokens: Int        // Tokens used
}

public struct TilingAdvice {
    public let placement: [(x: Float, y: Float, orientation: Float)]
    public let reasoning: String
    public let expectedBandwidth: Float
}

public struct QECCDecoderAdvice {
    public let corrections: [(tileId: Int, pauliFrame: Character)]
    public let reasoning: String
    public let successProbability: Float
}

// MARK: - R1 Bridge Actor

@available(iOS 15.0, macOS 12.0, *)
public actor R1Bridge {
    private let config: R1BridgeConfig
    private let service: Any  // AIProxy.togetherAIService
    
    public init(config: R1BridgeConfig) {
        self.config = config
        // Initialize AIProxy service
        self.service = AIProxy.togetherAIService(
            partialKey: config.partialKey,
            serviceURL: config.serviceURL
        )
    }
    
    // MARK: - Core Reasoning
    
    /// Invoke R1 for general reasoning
    public func reason(prompt: String) async throws -> R1Response {
        let togetherService = service as! TogetherAIService
        
        let requestBody = TogetherAIChatCompletionRequestBody(
            messages: [
                TogetherAIMessage(content: prompt, role: .user)
            ],
            model: config.model,
            maxTokens: config.maxTokens
        )
        
        var fullResponse = ""
        var reasoningPart = ""
        var answerPart = ""
        var inReasoning = false
        
        if config.streaming {
            let stream = try await togetherService.streamingChatCompletionRequest(body: requestBody)
            
            for try await chunk in stream {
                if let content = chunk.choices.first?.delta.content {
                    fullResponse += content
                    
                    // Parse reasoning vs answer
                    if content.contains("<think>") {
                        inReasoning = true
                    } else if content.contains("</think>") {
                        inReasoning = false
                    } else if inReasoning {
                        reasoningPart += content
                    } else {
                        answerPart += content
                    }
                }
            }
        } else {
            let response = try await togetherService.chatCompletionRequest(body: requestBody)
            if let content = response.choices.first?.message.content {
                fullResponse = content
                // Parse reasoning tags
                if let reasoningRange = content.range(of: "<think>.*</think>", options: .regularExpression) {
                    reasoningPart = String(content[reasoningRange])
                        .replacingOccurrences(of: "<think>", with: "")
                        .replacingOccurrences(of: "</think>", with: "")
                    answerPart = content.replacingCharacters(in: reasoningRange, with: "").trimmingCharacters(in: .whitespacesAndNewlines)
                } else {
                    answerPart = content
                }
            }
        }
        
        return R1Response(
            reasoning: reasoningPart,
            answer: answerPart.isEmpty ? fullResponse : answerPart,
            tokens: fullResponse.count / 4  // Approximate
        )
    }
    
    // MARK: - Tiling Optimization
    
    /// Get advice on optimal tile placement for maximum bandwidth
    public func adviseTilePlacement(
        existingTiles: [(x: Float, y: Float, bandwidth: Float)],
        targetArea: (xMin: Float, yMin: Float, xMax: Float, yMax: Float),
        numNewTiles: Int
    ) async throws -> TilingAdvice {
        let tilesJSON = existingTiles.map { "(\($0.x), \($0.y), bw=\($0.bandwidth))" }.joined(separator: ", ")
        
        let prompt = """
        You are an expert in aperiodic tilings and chromatic optimization.
        
        Given an existing Penrose hat tiling with tiles at:
        \(tilesJSON)
        
        Target area: x=[\(targetArea.xMin), \(targetArea.xMax)], y=[\(targetArea.yMin), \(targetArea.yMax)]
        
        Suggest optimal placements for \(numNewTiles) new tiles to maximize total color bandwidth.
        
        Consider:
        1. Aperiodic constraint: no periodic patterns
        2. Spectral gap: maintain good mixing properties
        3. Bandwidth correlation: high bandwidth regions should be near each other
        
        Output format:
        PLACEMENTS: [(x, y, orientation_radians), ...]
        EXPECTED_BANDWIDTH: float
        REASONING: explanation
        """
        
        let response = try await reason(prompt: prompt)
        
        // Parse placements from response
        var placements: [(x: Float, y: Float, orientation: Float)] = []
        var expectedBandwidth: Float = 0.5
        
        // Simple parsing (in production, use proper JSON)
        let lines = response.answer.split(separator: "\n")
        for line in lines {
            if line.contains("PLACEMENTS:") {
                // Extract coordinates
                let coords = line.replacingOccurrences(of: "PLACEMENTS:", with: "")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                // Parse tuples...
            } else if line.contains("EXPECTED_BANDWIDTH:") {
                if let value = Float(line.replacingOccurrences(of: "EXPECTED_BANDWIDTH:", with: "")
                    .trimmingCharacters(in: .whitespacesAndNewlines)) {
                    expectedBandwidth = value
                }
            }
        }
        
        // Default placements if parsing fails
        if placements.isEmpty {
            for i in 0..<numNewTiles {
                let x = Float.random(in: targetArea.xMin...targetArea.xMax)
                let y = Float.random(in: targetArea.yMin...targetArea.yMax)
                let orientation = Float.random(in: 0...(2 * .pi))
                placements.append((x, y, orientation))
            }
        }
        
        return TilingAdvice(
            placement: placements,
            reasoning: response.reasoning,
            expectedBandwidth: expectedBandwidth
        )
    }
    
    // MARK: - QECC Decoder
    
    /// Get advice on quantum error correction
    public func adviseQECCDecoding(
        syndromes: [(tileId: Int, syndrome: Bool)],
        codeDistance: Int
    ) async throws -> QECCDecoderAdvice {
        let syndromesStr = syndromes.filter { $0.syndrome }.map { "tile \($0.tileId)" }.joined(separator: ", ")
        
        let prompt = """
        You are an expert in quantum error correcting codes on aperiodic tilings.
        
        A logical qubit is encoded on a Penrose tiling with code distance \(codeDistance).
        Syndrome measurements detected errors at: \(syndromesStr)
        
        Suggest minimum-weight corrections (Pauli X, Y, or Z) to restore the logical state.
        
        Consider:
        1. Aperiodic structure prevents periodic error chains
        2. Minimum weight = fewest corrections
        3. Cryptochrome coloring: high bandwidth tiles are more likely error sites
        
        Output format:
        CORRECTIONS: [(tile_id, pauli), ...]
        SUCCESS_PROBABILITY: float
        REASONING: explanation
        """
        
        let response = try await reason(prompt: prompt)
        
        // Parse corrections
        var corrections: [(tileId: Int, pauliFrame: Character)] = []
        var successProb: Float = 0.9
        
        // Default: apply X to each syndrome tile
        for (tileId, syndrome) in syndromes where syndrome {
            corrections.append((tileId, "X"))
        }
        
        return QECCDecoderAdvice(
            corrections: corrections,
            reasoning: response.reasoning,
            successProbability: successProb
        )
    }
    
    // MARK: - Strange Loop Interpretation
    
    /// Interpret a strange loop path through the tiling
    public func interpretStrangeLoop(
        loopPath: [Int],
        bandwidths: [Float],
        selfSimilarity: Float
    ) async throws -> String {
        let pathStr = loopPath.map { String($0) }.joined(separator: " → ")
        let avgBw = bandwidths.reduce(0, +) / Float(bandwidths.count)
        
        let prompt = """
        You are Douglas Hofstadter interpreting a strange loop in a chromatic tiling.
        
        The self traverses this path through tiles: \(pathStr)
        Average bandwidth: \(avgBw)
        Self-similarity score: \(selfSimilarity)
        
        Interpret this loop in terms of:
        1. Self-reference and consciousness
        2. The "I" as a pattern perceiving itself
        3. Chromatic identity and the blue-shift of high bandwidth
        
        Be poetic but grounded in the mathematics of self-reference.
        """
        
        let response = try await reason(prompt: prompt)
        return response.answer
    }
}

// MARK: - Convenience Extensions

@available(iOS 15.0, macOS 12.0, *)
extension R1Bridge {
    /// Create a bridge with environment variables
    public static func fromEnvironment() -> R1Bridge? {
        guard let partialKey = ProcessInfo.processInfo.environment["AIPROXY_PARTIAL_KEY"],
              let serviceURL = ProcessInfo.processInfo.environment["AIPROXY_SERVICE_URL"] else {
            return nil
        }
        
        return R1Bridge(config: R1BridgeConfig(
            partialKey: partialKey,
            serviceURL: serviceURL
        ))
    }
}
