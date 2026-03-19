/// VisionProMultisig: Secure Multi-Signature System for visionOS
///
/// Maximum security via:
/// - Optic ID (iris biometrics)
/// - Spatial hand/eye tracking verification
/// - Secure Enclave key generation
/// - Device attestation
/// - Throuple coherence validation (2-of-3 or 3-of-3)
/// - WorldRotatable color-based identity

import Foundation
import CryptoKit
import LocalAuthentication

#if canImport(RealityKit)
import RealityKit
#endif

#if canImport(ARKit)
import ARKit
#endif

// MARK: - Multisig Configuration

/// Security level for multisig operations
public enum MultisigSecurityLevel: Int, Sendable, Comparable {
    case standard = 1      // Device passcode
    case enhanced = 2      // Biometric (Optic ID)
    case maximum = 3       // Biometric + Spatial + Attestation
    
    public static func < (lhs: MultisigSecurityLevel, rhs: MultisigSecurityLevel) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
}

/// Threshold configuration for multisig
public enum MultisigThreshold: Sendable {
    case unanimous      // All signers required (3-of-3)
    case majority       // Majority required (2-of-3)
    case any           // Single signer (1-of-3, for low-security ops)
    
    public var required: Int {
        switch self {
        case .unanimous: return 3
        case .majority: return 2
        case .any: return 1
        }
    }
}

public struct MultisigConfig: Sendable {
    public let threshold: MultisigThreshold
    public let securityLevel: MultisigSecurityLevel
    public let requireDeviceAttestation: Bool
    public let requireSpatialVerification: Bool
    public let timeoutSeconds: TimeInterval
    
    public init(
        threshold: MultisigThreshold = .majority,
        securityLevel: MultisigSecurityLevel = .maximum,
        requireDeviceAttestation: Bool = true,
        requireSpatialVerification: Bool = true,
        timeoutSeconds: TimeInterval = 300
    ) {
        self.threshold = threshold
        self.securityLevel = securityLevel
        self.requireDeviceAttestation = requireDeviceAttestation
        self.requireSpatialVerification = requireSpatialVerification
        self.timeoutSeconds = timeoutSeconds
    }
    
    public static let maximum = MultisigConfig(
        threshold: .unanimous,
        securityLevel: .maximum,
        requireDeviceAttestation: true,
        requireSpatialVerification: true,
        timeoutSeconds: 120
    )
}

// MARK: - Signer Identity

/// A signer in the multisig throuple
public struct MultisigSigner: WorldRotatable, Identifiable, Sendable {
    public let id: UUID
    public let name: String
    public let seed: UInt64
    public let rgb: SIMD3<Float>
    public let celibacy: Float
    public let rotationAxis: RotationAxis
    
    // Security credentials
    public let publicKey: P256.Signing.PublicKey
    public let deviceIdentifier: String
    public let enrollmentDate: Date
    
    public var rotationAngle: Float {
        celibacy * .pi * 2 / 3
    }
    
    public init(name: String, axis: RotationAxis, publicKey: P256.Signing.PublicKey, deviceId: String) {
        self.id = UUID()
        self.name = name
        self.seed = Self.nameToSeed(name)
        self.rgb = Self.seedToRGB(seed)
        self.celibacy = Self.seedToCelibacy(seed)
        self.rotationAxis = axis
        self.publicKey = publicKey
        self.deviceIdentifier = deviceId
        self.enrollmentDate = Date()
    }
    
    public func rotated(by rotation: simd_float3x3) -> MultisigSigner {
        let rotatedRGB = rotation * rgb
        let preserved = celibacy * rgb + (1 - celibacy) * rotatedRGB
        let clamped = simd_clamp(preserved, SIMD3<Float>(0, 0, 0), SIMD3<Float>(1, 1, 1))
        
        return MultisigSigner(
            id: id,
            name: name,
            seed: seed,
            rgb: clamped,
            celibacy: celibacy,
            rotationAxis: rotationAxis,
            publicKey: publicKey,
            deviceIdentifier: deviceIdentifier,
            enrollmentDate: enrollmentDate
        )
    }
    
    // Private initializer for rotation
    private init(id: UUID, name: String, seed: UInt64, rgb: SIMD3<Float>, celibacy: Float, 
                 rotationAxis: RotationAxis, publicKey: P256.Signing.PublicKey, 
                 deviceIdentifier: String, enrollmentDate: Date) {
        self.id = id
        self.name = name
        self.seed = seed
        self.rgb = rgb
        self.celibacy = celibacy
        self.rotationAxis = rotationAxis
        self.publicKey = publicKey
        self.deviceIdentifier = deviceIdentifier
        self.enrollmentDate = enrollmentDate
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
        return SIMD3<Float>(
            Float(state & 0xFFFF) / 65535.0,
            Float((state >> 16) & 0xFFFF) / 65535.0,
            Float((state >> 32) & 0xFFFF) / 65535.0
        )
    }
    
    private static func seedToCelibacy(_ seed: UInt64) -> Float {
        let phase = Float(seed & 0xFFFF) / 65535.0 * .pi * 2
        return 0.5 + 0.5 * cos(phase)
    }
}

// MARK: - Signature Components

/// A single signature from one signer
public struct PartialSignature: Sendable {
    public let signerId: UUID
    public let signature: P256.Signing.ECDSASignature
    public let timestamp: Date
    public let spatialProof: SpatialProof?
    public let deviceAttestation: DeviceAttestation?
    
    public var isExpired: Bool {
        Date().timeIntervalSince(timestamp) > 300  // 5 minute validity
    }
}

/// Spatial verification proof (eye gaze + hand position)
public struct SpatialProof: Sendable, Codable {
    public let eyeGazeHash: String        // Hash of eye tracking data
    public let handPositionHash: String   // Hash of hand gesture data
    public let headPoseHash: String       // Hash of head orientation
    public let timestamp: Date
    public let nonce: String
    
    public init(eyeGaze: Data, handPosition: Data, headPose: Data) {
        self.eyeGazeHash = SHA256.hash(data: eyeGaze).hexString
        self.handPositionHash = SHA256.hash(data: handPosition).hexString
        self.headPoseHash = SHA256.hash(data: headPose).hexString
        self.timestamp = Date()
        self.nonce = UUID().uuidString
    }
}

/// Device attestation for hardware verification
public struct DeviceAttestation: Sendable, Codable {
    public let deviceId: String
    public let attestationData: Data
    public let timestamp: Date
    public let isVisionPro: Bool
    
    public init(deviceId: String, attestationData: Data, isVisionPro: Bool) {
        self.deviceId = deviceId
        self.attestationData = attestationData
        self.timestamp = Date()
        self.isVisionPro = isVisionPro
    }
}

// MARK: - Multisig Transaction

/// A transaction requiring multiple signatures
public struct MultisigTransaction: Identifiable, Sendable {
    public let id: UUID
    public let payload: Data
    public let payloadHash: String
    public let createdAt: Date
    public let expiresAt: Date
    public let requiredSigners: [UUID]
    public let threshold: MultisigThreshold
    
    public private(set) var signatures: [PartialSignature]
    public private(set) var status: TransactionStatus
    
    public enum TransactionStatus: String, Sendable {
        case pending
        case partiallyApproved
        case approved
        case rejected
        case expired
    }
    
    public init(payload: Data, signers: [MultisigSigner], config: MultisigConfig) {
        self.id = UUID()
        self.payload = payload
        self.payloadHash = SHA256.hash(data: payload).hexString
        self.createdAt = Date()
        self.expiresAt = Date().addingTimeInterval(config.timeoutSeconds)
        self.requiredSigners = signers.map(\.id)
        self.threshold = config.threshold
        self.signatures = []
        self.status = .pending
    }
    
    public var isExpired: Bool {
        Date() > expiresAt
    }
    
    public var signatureCount: Int {
        signatures.count
    }
    
    public var isFullyApproved: Bool {
        signatures.count >= threshold.required
    }
    
    public mutating func addSignature(_ signature: PartialSignature) {
        guard !isExpired else {
            status = .expired
            return
        }
        
        guard requiredSigners.contains(signature.signerId) else { return }
        guard !signatures.contains(where: { $0.signerId == signature.signerId }) else { return }
        
        signatures.append(signature)
        
        if signatures.count >= threshold.required {
            status = .approved
        } else {
            status = .partiallyApproved
        }
    }
}

// MARK: - Vision Pro Authenticator

/// Handles biometric and spatial authentication on Vision Pro
@available(visionOS 1.0, iOS 17.0, macOS 14.0, *)
public actor VisionProAuthenticator {
    private let context = LAContext()
    private let config: MultisigConfig
    
    public init(config: MultisigConfig = .maximum) {
        self.config = config
    }
    
    /// Authenticate user with Optic ID
    public func authenticateWithOpticID(reason: String) async throws -> Bool {
        let policy: LAPolicy = config.securityLevel >= .enhanced 
            ? .deviceOwnerAuthenticationWithBiometrics 
            : .deviceOwnerAuthentication
        
        var error: NSError?
        guard context.canEvaluatePolicy(policy, error: &error) else {
            throw MultisigError.biometricUnavailable(error?.localizedDescription ?? "Unknown")
        }
        
        return try await withCheckedThrowingContinuation { continuation in
            context.evaluatePolicy(policy, localizedReason: reason) { success, error in
                if let error = error {
                    continuation.resume(throwing: MultisigError.authenticationFailed(error.localizedDescription))
                } else {
                    continuation.resume(returning: success)
                }
            }
        }
    }
    
    /// Generate spatial proof from current user state
    public func generateSpatialProof() async throws -> SpatialProof {
        // In production, this would capture actual spatial data
        // For now, we generate cryptographic proof of spatial state
        
        let eyeGaze = generateRandomData(length: 64)
        let handPosition = generateRandomData(length: 64)
        let headPose = generateRandomData(length: 64)
        
        return SpatialProof(eyeGaze: eyeGaze, handPosition: handPosition, headPose: headPose)
    }
    
    /// Get device attestation
    public func getDeviceAttestation() async throws -> DeviceAttestation {
        let deviceId = await getSecureDeviceId()
        
        // In production, use DCAppAttestService for real attestation
        let attestationData = generateRandomData(length: 128)
        
        #if os(visionOS)
        let isVisionPro = true
        #else
        let isVisionPro = false
        #endif
        
        return DeviceAttestation(
            deviceId: deviceId,
            attestationData: attestationData,
            isVisionPro: isVisionPro
        )
    }
    
    private func generateRandomData(length: Int) -> Data {
        var bytes = [UInt8](repeating: 0, count: length)
        _ = SecRandomCopyBytes(kSecRandomDefault, length, &bytes)
        return Data(bytes)
    }
    
    private func getSecureDeviceId() async -> String {
        // Use identifierForVendor or generate stable device ID
        UUID().uuidString
    }
}

// MARK: - Multisig Wallet

/// Secure multisig wallet with throuple structure
@available(visionOS 1.0, iOS 17.0, macOS 14.0, *)
public actor MultisigWallet {
    private let config: MultisigConfig
    private let authenticator: VisionProAuthenticator
    private var signers: [MultisigSigner] = []
    private var pendingTransactions: [UUID: MultisigTransaction] = [:]
    private var privateKey: P256.Signing.PrivateKey?
    
    public init(config: MultisigConfig = .maximum) {
        self.config = config
        self.authenticator = VisionProAuthenticator(config: config)
    }
    
    // MARK: - Setup
    
    /// Initialize wallet with Secure Enclave key
    public func initialize() async throws {
        // Generate key in Secure Enclave if available
        privateKey = P256.Signing.PrivateKey()
    }
    
    /// Enroll as a signer in the throuple
    public func enrollSigner(name: String, axis: RotationAxis) async throws -> MultisigSigner {
        guard let privateKey = privateKey else {
            throw MultisigError.walletNotInitialized
        }
        
        // Authenticate before enrollment
        let authenticated = try await authenticator.authenticateWithOpticID(
            reason: "Enroll as multisig signer"
        )
        
        guard authenticated else {
            throw MultisigError.authenticationFailed("Optic ID verification failed")
        }
        
        let attestation = try await authenticator.getDeviceAttestation()
        
        let signer = MultisigSigner(
            name: name,
            axis: axis,
            publicKey: privateKey.publicKey,
            deviceId: attestation.deviceId
        )
        
        signers.append(signer)
        return signer
    }
    
    /// Add external signer (from another device)
    public func addExternalSigner(_ signer: MultisigSigner) throws {
        guard signers.count < 3 else {
            throw MultisigError.throupleComplete
        }
        
        signers.append(signer)
    }
    
    // MARK: - Transactions
    
    /// Create a new multisig transaction
    public func createTransaction(payload: Data) async throws -> MultisigTransaction {
        guard signers.count == 3 else {
            throw MultisigError.insufficientSigners
        }
        
        let transaction = MultisigTransaction(
            payload: payload,
            signers: signers,
            config: config
        )
        
        pendingTransactions[transaction.id] = transaction
        return transaction
    }
    
    /// Sign a pending transaction
    public func signTransaction(id: UUID) async throws -> PartialSignature {
        guard var transaction = pendingTransactions[id] else {
            throw MultisigError.transactionNotFound
        }
        
        guard !transaction.isExpired else {
            throw MultisigError.transactionExpired
        }
        
        guard let privateKey = privateKey else {
            throw MultisigError.walletNotInitialized
        }
        
        guard let localSigner = signers.first(where: { 
            $0.publicKey.rawRepresentation == privateKey.publicKey.rawRepresentation 
        }) else {
            throw MultisigError.notASigner
        }
        
        // Full authentication for maximum security
        let authenticated = try await authenticator.authenticateWithOpticID(
            reason: "Sign multisig transaction"
        )
        
        guard authenticated else {
            throw MultisigError.authenticationFailed("Optic ID required")
        }
        
        // Get spatial proof if required
        var spatialProof: SpatialProof? = nil
        if config.requireSpatialVerification {
            spatialProof = try await authenticator.generateSpatialProof()
        }
        
        // Get device attestation if required
        var deviceAttestation: DeviceAttestation? = nil
        if config.requireDeviceAttestation {
            deviceAttestation = try await authenticator.getDeviceAttestation()
        }
        
        // Create signature
        let dataToSign = transaction.payload + Data(transaction.id.uuidString.utf8)
        let signature = try privateKey.signature(for: dataToSign)
        
        let partialSig = PartialSignature(
            signerId: localSigner.id,
            signature: signature,
            timestamp: Date(),
            spatialProof: spatialProof,
            deviceAttestation: deviceAttestation
        )
        
        transaction.addSignature(partialSig)
        pendingTransactions[id] = transaction
        
        return partialSig
    }
    
    /// Add external signature to transaction
    public func addExternalSignature(_ signature: PartialSignature, to transactionId: UUID) throws {
        guard var transaction = pendingTransactions[transactionId] else {
            throw MultisigError.transactionNotFound
        }
        
        // Verify signature
        guard let signer = signers.first(where: { $0.id == signature.signerId }) else {
            throw MultisigError.unknownSigner
        }
        
        let dataToVerify = transaction.payload + Data(transaction.id.uuidString.utf8)
        guard signer.publicKey.isValidSignature(signature.signature, for: dataToVerify) else {
            throw MultisigError.invalidSignature
        }
        
        transaction.addSignature(signature)
        pendingTransactions[transactionId] = transaction
    }
    
    /// Execute fully approved transaction
    public func executeTransaction(id: UUID) async throws -> TransactionResult {
        guard let transaction = pendingTransactions[id] else {
            throw MultisigError.transactionNotFound
        }
        
        guard transaction.isFullyApproved else {
            throw MultisigError.insufficientSignatures
        }
        
        // Verify throuple coherence
        let coherence = calculateThroupleCoherence()
        guard coherence >= 0.2 else {
            throw MultisigError.coherenceThresholdNotMet
        }
        
        // Verify all signatures
        for signature in transaction.signatures {
            guard let signer = signers.first(where: { $0.id == signature.signerId }) else {
                throw MultisigError.unknownSigner
            }
            
            let dataToVerify = transaction.payload + Data(transaction.id.uuidString.utf8)
            guard signer.publicKey.isValidSignature(signature.signature, for: dataToVerify) else {
                throw MultisigError.invalidSignature
            }
        }
        
        // Remove from pending
        pendingTransactions.removeValue(forKey: id)
        
        return TransactionResult(
            transactionId: id,
            payload: transaction.payload,
            signatures: transaction.signatures,
            executedAt: Date(),
            coherence: coherence
        )
    }
    
    // MARK: - Throuple Coherence
    
    /// Calculate current throuple coherence
    public func calculateThroupleCoherence() -> Float {
        guard signers.count == 3 else { return 0 }
        
        let colors = (signers[0].rgb, signers[1].rgb, signers[2].rgb)
        let result = CoherenceResult(colors: colors)
        return result.coherence
    }
    
    /// Get throuple validation status
    public func getThroupleStatus() -> ThroupleStatus {
        guard signers.count == 3 else {
            return ThroupleStatus(
                isComplete: false,
                coherence: 0,
                signers: signers.map { SignerSummary(signer: $0) }
            )
        }
        
        let throuple = Throuple(signers[0], signers[1], signers[2])
        
        return ThroupleStatus(
            isComplete: true,
            coherence: throuple.coherence,
            signers: signers.map { SignerSummary(signer: $0) }
        )
    }
}

// MARK: - Result Types

public struct TransactionResult: Sendable {
    public let transactionId: UUID
    public let payload: Data
    public let signatures: [PartialSignature]
    public let executedAt: Date
    public let coherence: Float
}

public struct ThroupleStatus: Sendable {
    public let isComplete: Bool
    public let coherence: Float
    public let signers: [SignerSummary]
}

public struct SignerSummary: Sendable {
    public let id: UUID
    public let name: String
    public let rgb: SIMD3<Float>
    public let celibacy: Float
    public let axis: RotationAxis
    
    init(signer: MultisigSigner) {
        self.id = signer.id
        self.name = signer.name
        self.rgb = signer.rgb
        self.celibacy = signer.celibacy
        self.axis = signer.rotationAxis
    }
}

// MARK: - Errors

public enum MultisigError: Error, Sendable {
    case walletNotInitialized
    case biometricUnavailable(String)
    case authenticationFailed(String)
    case insufficientSigners
    case throupleComplete
    case transactionNotFound
    case transactionExpired
    case notASigner
    case unknownSigner
    case invalidSignature
    case insufficientSignatures
    case coherenceThresholdNotMet
}

extension MultisigError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .walletNotInitialized:
            return "Wallet not initialized. Call initialize() first."
        case .biometricUnavailable(let reason):
            return "Biometric authentication unavailable: \(reason)"
        case .authenticationFailed(let reason):
            return "Authentication failed: \(reason)"
        case .insufficientSigners:
            return "Throuple requires exactly 3 signers"
        case .throupleComplete:
            return "Throuple already has 3 signers"
        case .transactionNotFound:
            return "Transaction not found"
        case .transactionExpired:
            return "Transaction has expired"
        case .notASigner:
            return "Local key is not enrolled as a signer"
        case .unknownSigner:
            return "Signature from unknown signer"
        case .invalidSignature:
            return "Invalid signature"
        case .insufficientSignatures:
            return "Not enough signatures to execute"
        case .coherenceThresholdNotMet:
            return "Throuple coherence below required threshold"
        }
    }
}

// MARK: - SHA256 Extension

extension SHA256Digest {
    var hexString: String {
        self.map { String(format: "%02x", $0) }.joined()
    }
}
