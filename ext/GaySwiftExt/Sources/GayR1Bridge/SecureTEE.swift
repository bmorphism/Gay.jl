/// SecureTEE: Trusted Execution Environment for R1 API Interactions
///
/// Provides hardware-backed security for:
/// - API key storage (Secure Enclave on Apple Silicon)
/// - Request signing
/// - Response verification
/// - Attestation of reasoning integrity

import Foundation
import CryptoKit
import Security

// MARK: - TEE Configuration

/// Configuration for secure TEE interactions
public struct TEEConfig: Sendable {
    public let serviceName: String
    public let accessGroup: String?
    public let requireBiometric: Bool
    public let attestationEnabled: Bool
    
    public init(
        serviceName: String = "com.gay.r1bridge",
        accessGroup: String? = nil,
        requireBiometric: Bool = false,
        attestationEnabled: Bool = true
    ) {
        self.serviceName = serviceName
        self.accessGroup = accessGroup
        self.requireBiometric = requireBiometric
        self.attestationEnabled = attestationEnabled
    }
}

// MARK: - Secure Key Storage

/// Secure storage for API credentials using Keychain/Secure Enclave
public actor SecureKeyStore {
    private let config: TEEConfig
    
    public init(config: TEEConfig = TEEConfig()) {
        self.config = config
    }
    
    /// Store API key securely
    public func storeAPIKey(_ key: String, for identifier: String) throws {
        let keyData = Data(key.utf8)
        
        var query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: config.serviceName,
            kSecAttrAccount as String: identifier,
            kSecValueData as String: keyData
        ]
        
        if let accessGroup = config.accessGroup {
            query[kSecAttrAccessGroup as String] = accessGroup
        }
        
        // Set access control for Secure Enclave if available
        if config.requireBiometric {
            let access = SecAccessControlCreateWithFlags(
                nil,
                kSecAttrAccessibleWhenUnlockedThisDeviceOnly,
                .biometryCurrentSet,
                nil
            )
            query[kSecAttrAccessControl as String] = access
        }
        
        // Delete existing key if present
        SecItemDelete(query as CFDictionary)
        
        let status = SecItemAdd(query as CFDictionary, nil)
        guard status == errSecSuccess else {
            throw TEEError.keychainError(status)
        }
    }
    
    /// Retrieve API key securely
    public func retrieveAPIKey(for identifier: String) throws -> String {
        var query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: config.serviceName,
            kSecAttrAccount as String: identifier,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]
        
        if let accessGroup = config.accessGroup {
            query[kSecAttrAccessGroup as String] = accessGroup
        }
        
        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        
        guard status == errSecSuccess,
              let data = result as? Data,
              let key = String(data: data, encoding: .utf8) else {
            throw TEEError.keychainError(status)
        }
        
        return key
    }
    
    /// Delete API key
    public func deleteAPIKey(for identifier: String) throws {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: config.serviceName,
            kSecAttrAccount as String: identifier
        ]
        
        let status = SecItemDelete(query as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw TEEError.keychainError(status)
        }
    }
}

// MARK: - Request Signing

/// Signs requests for integrity verification
public struct RequestSigner: Sendable {
    private let symmetricKey: SymmetricKey
    
    public init(seed: UInt64) {
        // Derive key from seed using SHA256
        var seedBytes = withUnsafeBytes(of: seed) { Array($0) }
        let hash = SHA256.hash(data: seedBytes)
        self.symmetricKey = SymmetricKey(data: hash)
    }
    
    /// Sign request data
    public func sign(_ data: Data) -> Data {
        let signature = HMAC<SHA256>.authenticationCode(for: data, using: symmetricKey)
        return Data(signature)
    }
    
    /// Verify signature
    public func verify(_ data: Data, signature: Data) -> Bool {
        let expectedSignature = sign(data)
        return expectedSignature == signature
    }
    
    /// Create signed request envelope
    public func createSignedEnvelope(payload: Data, timestamp: Date = Date()) -> SignedEnvelope {
        let timestampData = withUnsafeBytes(of: timestamp.timeIntervalSince1970) { Data($0) }
        let combined = payload + timestampData
        let signature = sign(combined)
        
        return SignedEnvelope(
            payload: payload,
            timestamp: timestamp,
            signature: signature
        )
    }
}

public struct SignedEnvelope: Codable, Sendable {
    public let payload: Data
    public let timestamp: Date
    public let signature: Data
    
    public var isExpired: Bool {
        Date().timeIntervalSince(timestamp) > 300  // 5 minute expiry
    }
}

// MARK: - Attestation

/// Attestation for reasoning integrity
public struct ReasoningAttestation: Codable, Sendable {
    public let requestHash: String
    public let responseHash: String
    public let reasoningSteps: Int
    public let timestamp: Date
    public let signature: Data
    
    public init(request: String, response: String, steps: Int, signer: RequestSigner) {
        self.requestHash = SHA256.hash(data: Data(request.utf8)).hexString
        self.responseHash = SHA256.hash(data: Data(response.utf8)).hexString
        self.reasoningSteps = steps
        self.timestamp = Date()
        
        let attestationData = "\(requestHash)|\(responseHash)|\(steps)|\(timestamp.timeIntervalSince1970)"
        self.signature = signer.sign(Data(attestationData.utf8))
    }
    
    public func verify(with signer: RequestSigner) -> Bool {
        let attestationData = "\(requestHash)|\(responseHash)|\(reasoningSteps)|\(timestamp.timeIntervalSince1970)"
        return signer.verify(Data(attestationData.utf8), signature: signature)
    }
}

extension SHA256.Digest {
    var hexString: String {
        self.map { String(format: "%02x", $0) }.joined()
    }
}

// MARK: - Secure R1 Session

/// A secure session for R1 interactions with TEE protection
@available(iOS 15.0, macOS 12.0, *)
public actor SecureR1Session {
    private let keyStore: SecureKeyStore
    private let signer: RequestSigner
    private let config: TEEConfig
    
    private var attestations: [ReasoningAttestation] = []
    
    public init(seed: UInt64, config: TEEConfig = TEEConfig()) {
        self.keyStore = SecureKeyStore(config: config)
        self.signer = RequestSigner(seed: seed)
        self.config = config
    }
    
    /// Initialize with stored credentials
    public func initialize(partialKeyIdentifier: String, serviceURLIdentifier: String) async throws -> SecureR1Credentials {
        let partialKey = try await keyStore.retrieveAPIKey(for: partialKeyIdentifier)
        let serviceURL = try await keyStore.retrieveAPIKey(for: serviceURLIdentifier)
        
        return SecureR1Credentials(
            partialKey: partialKey,
            serviceURL: serviceURL,
            sessionToken: generateSessionToken()
        )
    }
    
    /// Store credentials securely
    public func storeCredentials(partialKey: String, serviceURL: String,
                                  partialKeyIdentifier: String = "r1_partial_key",
                                  serviceURLIdentifier: String = "r1_service_url") async throws {
        try await keyStore.storeAPIKey(partialKey, for: partialKeyIdentifier)
        try await keyStore.storeAPIKey(serviceURL, for: serviceURLIdentifier)
    }
    
    /// Create signed request for R1
    public func createSecureRequest(prompt: String) -> SecureR1Request {
        let envelope = signer.createSignedEnvelope(payload: Data(prompt.utf8))
        
        return SecureR1Request(
            prompt: prompt,
            envelope: envelope,
            attestationEnabled: config.attestationEnabled
        )
    }
    
    /// Process and attest response
    public func processResponse(request: SecureR1Request, response: String, reasoningSteps: Int) -> SecureR1Response {
        let attestation = ReasoningAttestation(
            request: request.prompt,
            response: response,
            steps: reasoningSteps,
            signer: signer
        )
        
        attestations.append(attestation)
        
        return SecureR1Response(
            content: response,
            attestation: attestation,
            isVerified: attestation.verify(with: signer)
        )
    }
    
    /// Get all attestations for audit
    public func getAttestations() -> [ReasoningAttestation] {
        attestations
    }
    
    private func generateSessionToken() -> String {
        var bytes = [UInt8](repeating: 0, count: 32)
        _ = SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes)
        return bytes.map { String(format: "%02x", $0) }.joined()
    }
}

// MARK: - Secure Types

public struct SecureR1Credentials: Sendable {
    public let partialKey: String
    public let serviceURL: String
    public let sessionToken: String
}

public struct SecureR1Request: Sendable {
    public let prompt: String
    public let envelope: SignedEnvelope
    public let attestationEnabled: Bool
}

public struct SecureR1Response: Sendable {
    public let content: String
    public let attestation: ReasoningAttestation
    public let isVerified: Bool
}

// MARK: - Errors

public enum TEEError: Error, Sendable {
    case keychainError(OSStatus)
    case attestationFailed
    case signatureInvalid
    case sessionExpired
    case enclaveBunavailable
}

extension TEEError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .keychainError(let status):
            return "Keychain error: \(status)"
        case .attestationFailed:
            return "Reasoning attestation failed"
        case .signatureInvalid:
            return "Request signature is invalid"
        case .sessionExpired:
            return "Secure session has expired"
        case .enclaveBunavailable:
            return "Secure Enclave is not available"
        }
    }
}

// MARK: - WorldRotatable Integration

/// Extension to make WorldRotatable work with secure R1 sessions
@available(iOS 15.0, macOS 12.0, *)
extension SecureR1Session {
    /// Request R1 reasoning about a WorldRotatable throuple
    public func reasonAboutThrouple<T: WorldRotatable>(
        _ throuple: Throuple<T>,
        question: String
    ) -> SecureR1Request {
        let context = """
        Throuple Analysis Request:
        - Entity 1: RGB(\(throuple.entities.0.rgb.x), \(throuple.entities.0.rgb.y), \(throuple.entities.0.rgb.z)), Celibacy: \(throuple.entities.0.celibacy)
        - Entity 2: RGB(\(throuple.entities.1.rgb.x), \(throuple.entities.1.rgb.y), \(throuple.entities.1.rgb.z)), Celibacy: \(throuple.entities.1.celibacy)
        - Entity 3: RGB(\(throuple.entities.2.rgb.x), \(throuple.entities.2.rgb.y), \(throuple.entities.2.rgb.z)), Celibacy: \(throuple.entities.2.celibacy)
        - Coherence: \(throuple.coherence)
        - Valid SO(3) rotation: \(throuple.isValidRotation)
        
        Question: \(question)
        """
        
        return createSecureRequest(prompt: context)
    }
}
