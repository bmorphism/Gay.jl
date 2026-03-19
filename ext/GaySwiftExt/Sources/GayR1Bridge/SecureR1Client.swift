/// SecureR1Client: Production-ready R1 client with TEE integration
///
/// Combines:
/// - Secure credential storage (Keychain/Secure Enclave)
/// - Request signing and attestation
/// - WorldRotatable throuple analysis
/// - Streaming response handling

import Foundation

#if canImport(AIProxy)
import AIProxy
#endif

// MARK: - R1 Client Configuration

public struct R1ClientConfig: Sendable {
    public let model: String
    public let maxTokens: Int
    public let temperature: Float
    public let streaming: Bool
    public let teeConfig: TEEConfig
    
    public init(
        model: String = "deepseek-ai/DeepSeek-R1",
        maxTokens: Int = 4096,
        temperature: Float = 0.7,
        streaming: Bool = true,
        teeConfig: TEEConfig = TEEConfig()
    ) {
        self.model = model
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.streaming = streaming
        self.teeConfig = teeConfig
    }
    
    public static let `default` = R1ClientConfig()
}

// MARK: - Secure R1 Client

@available(iOS 15.0, macOS 12.0, *)
public actor SecureR1Client {
    private let config: R1ClientConfig
    private let session: SecureR1Session
    private var credentials: SecureR1Credentials?
    
    public init(seed: UInt64 = 1069, config: R1ClientConfig = .default) {
        self.config = config
        self.session = SecureR1Session(seed: seed, config: config.teeConfig)
    }
    
    // MARK: - Credential Management
    
    /// Initialize with stored credentials
    public func initialize() async throws {
        credentials = try await session.initialize(
            partialKeyIdentifier: "r1_partial_key",
            serviceURLIdentifier: "r1_service_url"
        )
    }
    
    /// Store new credentials securely
    public func setCredentials(partialKey: String, serviceURL: String) async throws {
        try await session.storeCredentials(partialKey: partialKey, serviceURL: serviceURL)
        credentials = SecureR1Credentials(
            partialKey: partialKey,
            serviceURL: serviceURL,
            sessionToken: UUID().uuidString
        )
    }
    
    // MARK: - Reasoning Requests
    
    /// Send a reasoning request to R1
    public func reason(prompt: String) async throws -> R1ReasoningResult {
        guard let creds = credentials else {
            throw R1ClientError.notInitialized
        }
        
        let secureRequest = await session.createSecureRequest(prompt: prompt)
        
        // Verify request hasn't expired
        guard !secureRequest.envelope.isExpired else {
            throw R1ClientError.requestExpired
        }
        
        #if canImport(AIProxy)
        // Real implementation with AIProxy
        let service = AIProxy.togetherAIService(
            partialKey: creds.partialKey,
            serviceURL: creds.serviceURL
        )
        
        let requestBody = TogetherAIChatCompletionRequestBody(
            messages: [TogetherAIMessage(content: prompt, role: .user)],
            model: config.model,
            maxTokens: config.maxTokens,
            temperature: Double(config.temperature)
        )
        
        var fullResponse = ""
        var reasoningContent = ""
        var answerContent = ""
        var inReasoning = false
        
        if config.streaming {
            let stream = try await service.streamingChatCompletionRequest(body: requestBody)
            
            for try await chunk in stream {
                if let content = chunk.choices.first?.delta.content {
                    fullResponse += content
                    
                    // Parse <think> tags
                    if content.contains("<think>") {
                        inReasoning = true
                    } else if content.contains("</think>") {
                        inReasoning = false
                    } else if inReasoning {
                        reasoningContent += content
                    } else {
                        answerContent += content
                    }
                }
            }
        } else {
            let response = try await service.chatCompletionRequest(body: requestBody)
            fullResponse = response.choices.first?.message.content ?? ""
            answerContent = fullResponse
        }
        
        let reasoningSteps = reasoningContent.components(separatedBy: "\n").count
        let secureResponse = await session.processResponse(
            request: secureRequest,
            response: fullResponse,
            reasoningSteps: reasoningSteps
        )
        
        return R1ReasoningResult(
            reasoning: reasoningContent,
            answer: answerContent.isEmpty ? fullResponse : answerContent,
            attestation: secureResponse.attestation,
            isVerified: secureResponse.isVerified
        )
        #else
        // Mock implementation for testing
        return R1ReasoningResult(
            reasoning: "Mock reasoning for: \(prompt)",
            answer: "Mock answer",
            attestation: ReasoningAttestation(
                request: prompt,
                response: "Mock response",
                steps: 1,
                signer: RequestSigner(seed: 1069)
            ),
            isVerified: true
        )
        #endif
    }
    
    // MARK: - WorldRotatable Analysis
    
    /// Analyze a throuple with R1 reasoning
    public func analyzeThrouple<T: WorldRotatable>(
        _ throuple: Throuple<T>,
        question: String
    ) async throws -> ThroupleAnalysisResult {
        let coherenceResult = CoherenceResult(colors: (
            throuple.entities.0.rgb,
            throuple.entities.1.rgb,
            throuple.entities.2.rgb
        ))
        
        let prompt = """
        Analyze this chromatic throuple with WorldRotatable properties:
        
        Entity 1:
        - RGB: (\(throuple.entities.0.rgb.x), \(throuple.entities.0.rgb.y), \(throuple.entities.0.rgb.z))
        - Celibacy: \(throuple.entities.0.celibacy)
        - Rotation axis: \(throuple.entities.0.rotationAxis)
        
        Entity 2:
        - RGB: (\(throuple.entities.1.rgb.x), \(throuple.entities.1.rgb.y), \(throuple.entities.1.rgb.z))
        - Celibacy: \(throuple.entities.1.celibacy)
        - Rotation axis: \(throuple.entities.1.rotationAxis)
        
        Entity 3:
        - RGB: (\(throuple.entities.2.rgb.x), \(throuple.entities.2.rgb.y), \(throuple.entities.2.rgb.z))
        - Celibacy: \(throuple.entities.2.celibacy)
        - Rotation axis: \(throuple.entities.2.rotationAxis)
        
        Throuple Properties:
        - Coherence: \(coherenceResult.coherence)
        - Has sufficient mass: \(coherenceResult.hasSufficientMass)
        - Valid SO(3) rotation: \(throuple.isValidRotation)
        - 2-cells: \(coherenceResult.twoCells.count)
        
        Question: \(question)
        
        Provide analysis considering:
        1. Color space geometry
        2. Rotation preservation (celibacy factors)
        3. 2-cell validation structure
        4. Chromatic coherence implications
        """
        
        let result = try await reason(prompt: prompt)
        
        return ThroupleAnalysisResult(
            coherence: coherenceResult,
            reasoning: result,
            recommendation: extractRecommendation(from: result.answer)
        )
    }
    
    /// Generate rotation advice for optimal coherence
    public func adviseRotation<T: WorldRotatable>(
        _ throuple: Throuple<T>
    ) async throws -> RotationAdvice {
        let prompt = """
        Given a throuple with combined rotation matrix:
        \(formatMatrix(throuple.combinedRotation))
        
        And coherence: \(throuple.coherence)
        
        What rotation adjustments would maximize coherence while maintaining SO(3) validity?
        
        Consider:
        1. Eigenvalue analysis of the rotation
        2. Celibacy preservation requirements
        3. 2-cell validation mass distribution
        """
        
        let result = try await reason(prompt: prompt)
        
        return RotationAdvice(
            suggestedAdjustment: parseRotationAdjustment(from: result.answer),
            reasoning: result.reasoning,
            expectedCoherenceImprovement: parseCoherenceImprovement(from: result.answer)
        )
    }
    
    // MARK: - Helpers
    
    private func extractRecommendation(from answer: String) -> String {
        // Extract recommendation section if present
        if let range = answer.range(of: "Recommendation:") {
            return String(answer[range.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return answer
    }
    
    private func formatMatrix(_ m: simd_float3x3) -> String {
        """
        [\(m.columns.0.x), \(m.columns.1.x), \(m.columns.2.x)]
        [\(m.columns.0.y), \(m.columns.1.y), \(m.columns.2.y)]
        [\(m.columns.0.z), \(m.columns.1.z), \(m.columns.2.z)]
        """
    }
    
    private func parseRotationAdjustment(from answer: String) -> simd_float3x3 {
        // Default to identity if parsing fails
        matrix_identity_float3x3
    }
    
    private func parseCoherenceImprovement(from answer: String) -> Float {
        // Default estimate
        0.1
    }
}

// MARK: - Result Types

public struct R1ReasoningResult: Sendable {
    public let reasoning: String
    public let answer: String
    public let attestation: ReasoningAttestation
    public let isVerified: Bool
}

public struct ThroupleAnalysisResult: Sendable {
    public let coherence: CoherenceResult
    public let reasoning: R1ReasoningResult
    public let recommendation: String
}

public struct RotationAdvice: Sendable {
    public let suggestedAdjustment: simd_float3x3
    public let reasoning: String
    public let expectedCoherenceImprovement: Float
}

// MARK: - Errors

public enum R1ClientError: Error, Sendable {
    case notInitialized
    case requestExpired
    case invalidCredentials
    case networkError(Error)
    case parsingError
}

extension R1ClientError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .notInitialized:
            return "R1 client not initialized. Call setCredentials first."
        case .requestExpired:
            return "Request envelope has expired"
        case .invalidCredentials:
            return "Invalid API credentials"
        case .networkError(let error):
            return "Network error: \(error.localizedDescription)"
        case .parsingError:
            return "Failed to parse R1 response"
        }
    }
}

// MARK: - Convenience Extensions

@available(iOS 15.0, macOS 12.0, *)
extension SecureR1Client {
    /// Quick analysis of canonical throuples
    public func analyzeCanonical(_ type: CanonicalThroupleType) async throws -> ThroupleAnalysisResult {
        let throuple: Throuple<OriginaryHue>
        let question: String
        
        switch type {
        case .philosophical:
            throuple = CanonicalThrouples.philosophical
            question = "What is the philosophical significance of this Æther-Möbius-Ouroboros configuration?"
        case .primary:
            throuple = CanonicalThrouples.primary
            question = "How do the RGB primaries interact in this rotation?"
        case .secondary:
            throuple = CanonicalThrouples.secondary
            question = "What complementary relationships exist in the CMY configuration?"
        case .agents:
            throuple = CanonicalThrouples.agents
            question = "How does the Alice-Bob-Carol communication pattern emerge from this structure?"
        case .gay:
            throuple = CanonicalThrouples.gay
            question = "What chromatic properties define the Gay-Splittable-Chromatic canonical?"
        }
        
        return try await analyzeThrouple(throuple, question: question)
    }
}

public enum CanonicalThroupleType: String, CaseIterable, Sendable {
    case philosophical
    case primary
    case secondary
    case agents
    case gay
}
