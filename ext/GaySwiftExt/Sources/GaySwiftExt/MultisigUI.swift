/// MultisigUI: SwiftUI Views for Vision Pro Multisig Interactions
///
/// Immersive UI components for:
/// - Signer enrollment with spatial visualization
/// - Transaction approval with color coherence display
/// - Throuple status visualization

import SwiftUI

#if canImport(RealityKit)
import RealityKit
#endif

// MARK: - Color Extensions

extension SIMD3 where Scalar == Float {
    var color: Color {
        Color(red: Double(x), green: Double(y), blue: Double(z))
    }
}

// MARK: - Signer Card View

public struct SignerCardView: View {
    let signer: SignerSummary
    let isLocal: Bool
    
    public init(signer: SignerSummary, isLocal: Bool = false) {
        self.signer = signer
        self.isLocal = isLocal
    }
    
    public var body: some View {
        VStack(spacing: 12) {
            // Color orb
            Circle()
                .fill(signer.rgb.color)
                .frame(width: 60, height: 60)
                .shadow(color: signer.rgb.color.opacity(0.6), radius: 10)
                .overlay(
                    Circle()
                        .stroke(isLocal ? Color.white : Color.clear, lineWidth: 3)
                )
            
            // Name
            Text(signer.name)
                .font(.headline)
                .foregroundColor(.white)
            
            // Celibacy indicator
            HStack(spacing: 4) {
                Image(systemName: celibacyIcon)
                Text("\(Int(signer.celibacy * 100))%")
                    .font(.caption)
            }
            .foregroundColor(.secondary)
            
            // Rotation axis
            Text("\(signer.axis)".uppercased())
                .font(.caption2)
                .padding(.horizontal, 8)
                .padding(.vertical, 2)
                .background(axisColor.opacity(0.3))
                .cornerRadius(4)
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(.ultraThinMaterial)
        )
    }
    
    private var celibacyIcon: String {
        signer.celibacy > 0.7 ? "lock.fill" :
        signer.celibacy > 0.3 ? "lock.open.fill" : "sparkles"
    }
    
    private var axisColor: Color {
        switch signer.axis {
        case .x: return .red
        case .y: return .green
        case .z: return .blue
        }
    }
}

// MARK: - Throuple Visualization

public struct ThroupleVisualizationView: View {
    let status: ThroupleStatus
    
    public init(status: ThroupleStatus) {
        self.status = status
    }
    
    public var body: some View {
        VStack(spacing: 24) {
            // Coherence meter
            CoherenceMeterView(coherence: status.coherence)
            
            // Signers in triangle formation
            if status.signers.count >= 3 {
                ZStack {
                    // Connection lines
                    ThroupleConnectionsView(signers: status.signers)
                    
                    // Signer orbs
                    GeometryReader { geometry in
                        let center = CGPoint(x: geometry.size.width / 2, y: geometry.size.height / 2)
                        let radius = min(geometry.size.width, geometry.size.height) * 0.35
                        
                        ForEach(0..<3) { index in
                            let angle = Double(index) * 2 * .pi / 3 - .pi / 2
                            let x = center.x + radius * cos(angle)
                            let y = center.y + radius * sin(angle)
                            
                            SignerOrbView(signer: status.signers[index])
                                .position(x: x, y: y)
                        }
                    }
                }
                .frame(height: 300)
            } else {
                // Incomplete throuple
                VStack {
                    Image(systemName: "person.3.fill")
                        .font(.largeTitle)
                        .foregroundColor(.secondary)
                    Text("Throuple Incomplete")
                        .font(.headline)
                    Text("\(status.signers.count)/3 signers enrolled")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
        }
        .padding()
    }
}

struct SignerOrbView: View {
    let signer: SignerSummary
    
    var body: some View {
        VStack(spacing: 8) {
            ZStack {
                Circle()
                    .fill(signer.rgb.color)
                    .frame(width: 80, height: 80)
                
                Circle()
                    .stroke(Color.white.opacity(0.3), lineWidth: 2)
                    .frame(width: 80, height: 80)
                
                // Celibacy ring
                Circle()
                    .trim(from: 0, to: CGFloat(signer.celibacy))
                    .stroke(Color.white, lineWidth: 4)
                    .frame(width: 90, height: 90)
                    .rotationEffect(.degrees(-90))
            }
            
            Text(signer.name)
                .font(.caption)
                .foregroundColor(.white)
        }
    }
}

struct ThroupleConnectionsView: View {
    let signers: [SignerSummary]
    
    var body: some View {
        GeometryReader { geometry in
            let center = CGPoint(x: geometry.size.width / 2, y: geometry.size.height / 2)
            let radius = min(geometry.size.width, geometry.size.height) * 0.35
            
            Path { path in
                for i in 0..<3 {
                    let angle1 = Double(i) * 2 * .pi / 3 - .pi / 2
                    let angle2 = Double((i + 1) % 3) * 2 * .pi / 3 - .pi / 2
                    
                    let p1 = CGPoint(
                        x: center.x + radius * cos(angle1),
                        y: center.y + radius * sin(angle1)
                    )
                    let p2 = CGPoint(
                        x: center.x + radius * cos(angle2),
                        y: center.y + radius * sin(angle2)
                    )
                    
                    path.move(to: p1)
                    path.addLine(to: p2)
                }
            }
            .stroke(
                LinearGradient(
                    colors: signers.map { $0.rgb.color },
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                ),
                lineWidth: 2
            )
        }
    }
}

// MARK: - Coherence Meter

public struct CoherenceMeterView: View {
    let coherence: Float
    
    public init(coherence: Float) {
        self.coherence = coherence
    }
    
    public var body: some View {
        VStack(spacing: 8) {
            Text("Throuple Coherence")
                .font(.caption)
                .foregroundColor(.secondary)
            
            ZStack(alignment: .leading) {
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color.gray.opacity(0.3))
                    .frame(height: 24)
                
                RoundedRectangle(cornerRadius: 8)
                    .fill(coherenceColor)
                    .frame(width: CGFloat(coherence) * 200, height: 24)
            }
            .frame(width: 200)
            
            HStack {
                Text("\(Int(coherence * 100))%")
                    .font(.headline)
                    .foregroundColor(coherenceColor)
                
                Image(systemName: statusIcon)
                    .foregroundColor(coherenceColor)
            }
        }
    }
    
    private var coherenceColor: Color {
        if coherence >= 0.5 { return .green }
        if coherence >= 0.2 { return .yellow }
        return .red
    }
    
    private var statusIcon: String {
        if coherence >= 0.5 { return "checkmark.circle.fill" }
        if coherence >= 0.2 { return "exclamationmark.circle.fill" }
        return "xmark.circle.fill"
    }
}

// MARK: - Transaction Approval View

public struct TransactionApprovalView: View {
    let transaction: MultisigTransaction
    let signers: [SignerSummary]
    let onApprove: () async throws -> Void
    let onReject: () -> Void
    
    @State private var isApproving = false
    @State private var error: String?
    
    public init(
        transaction: MultisigTransaction,
        signers: [SignerSummary],
        onApprove: @escaping () async throws -> Void,
        onReject: @escaping () -> Void
    ) {
        self.transaction = transaction
        self.signers = signers
        self.onApprove = onApprove
        self.onReject = onReject
    }
    
    public var body: some View {
        VStack(spacing: 24) {
            // Header
            VStack(spacing: 8) {
                Image(systemName: "signature")
                    .font(.largeTitle)
                    .foregroundColor(.blue)
                
                Text("Signature Required")
                    .font(.title2)
                    .fontWeight(.semibold)
            }
            
            // Transaction details
            VStack(alignment: .leading, spacing: 12) {
                DetailRow(label: "Transaction ID", value: transaction.id.uuidString.prefix(8) + "...")
                DetailRow(label: "Payload Hash", value: transaction.payloadHash.prefix(16) + "...")
                DetailRow(label: "Threshold", value: thresholdText)
                DetailRow(label: "Signatures", value: "\(transaction.signatureCount)/\(transaction.threshold.required)")
                DetailRow(label: "Expires", value: expiryText)
            }
            .padding()
            .background(RoundedRectangle(cornerRadius: 12).fill(.ultraThinMaterial))
            
            // Signer status
            HStack(spacing: 16) {
                ForEach(signers, id: \.id) { signer in
                    SignerStatusView(
                        signer: signer,
                        hasSigned: transaction.signatures.contains { $0.signerId == signer.id }
                    )
                }
            }
            
            // Error message
            if let error = error {
                Text(error)
                    .font(.caption)
                    .foregroundColor(.red)
            }
            
            // Actions
            HStack(spacing: 16) {
                Button(action: onReject) {
                    Label("Reject", systemImage: "xmark")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .tint(.red)
                
                Button(action: approve) {
                    if isApproving {
                        ProgressView()
                            .frame(maxWidth: .infinity)
                    } else {
                        Label("Approve", systemImage: "checkmark")
                            .frame(maxWidth: .infinity)
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(isApproving)
            }
        }
        .padding()
    }
    
    private var thresholdText: String {
        switch transaction.threshold {
        case .unanimous: return "3 of 3 (Unanimous)"
        case .majority: return "2 of 3 (Majority)"
        case .any: return "1 of 3 (Any)"
        }
    }
    
    private var expiryText: String {
        let formatter = RelativeDateTimeFormatter()
        return formatter.localizedString(for: transaction.expiresAt, relativeTo: Date())
    }
    
    private func approve() {
        isApproving = true
        error = nil
        
        Task {
            do {
                try await onApprove()
            } catch {
                self.error = error.localizedDescription
            }
            isApproving = false
        }
    }
}

struct DetailRow: View {
    let label: String
    let value: any StringProtocol
    
    var body: some View {
        HStack {
            Text(label)
                .foregroundColor(.secondary)
            Spacer()
            Text(value)
                .fontWeight(.medium)
        }
        .font(.caption)
    }
}

struct SignerStatusView: View {
    let signer: SignerSummary
    let hasSigned: Bool
    
    var body: some View {
        VStack(spacing: 4) {
            ZStack {
                Circle()
                    .fill(signer.rgb.color)
                    .frame(width: 40, height: 40)
                
                if hasSigned {
                    Circle()
                        .fill(Color.green.opacity(0.8))
                        .frame(width: 40, height: 40)
                    
                    Image(systemName: "checkmark")
                        .foregroundColor(.white)
                        .fontWeight(.bold)
                }
            }
            
            Text(signer.name)
                .font(.caption2)
        }
    }
}

// MARK: - Enrollment View

public struct SignerEnrollmentView: View {
    @State private var name = ""
    @State private var selectedAxis: RotationAxis = .x
    @State private var isEnrolling = false
    @State private var previewColor: SIMD3<Float>?
    
    let onEnroll: (String, RotationAxis) async throws -> MultisigSigner
    let onComplete: (MultisigSigner) -> Void
    
    public init(
        onEnroll: @escaping (String, RotationAxis) async throws -> MultisigSigner,
        onComplete: @escaping (MultisigSigner) -> Void
    ) {
        self.onEnroll = onEnroll
        self.onComplete = onComplete
    }
    
    public var body: some View {
        VStack(spacing: 24) {
            Text("Enroll as Signer")
                .font(.title)
                .fontWeight(.bold)
            
            // Name input
            TextField("Your Name", text: $name)
                .textFieldStyle(.roundedBorder)
                .onChange(of: name) { _, newValue in
                    updatePreviewColor(name: newValue)
                }
            
            // Preview orb
            if let color = previewColor {
                Circle()
                    .fill(color.color)
                    .frame(width: 100, height: 100)
                    .shadow(color: color.color.opacity(0.6), radius: 15)
            }
            
            // Axis selection
            VStack(alignment: .leading) {
                Text("Rotation Axis")
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                Picker("Axis", selection: $selectedAxis) {
                    Text("X").tag(RotationAxis.x)
                    Text("Y").tag(RotationAxis.y)
                    Text("Z").tag(RotationAxis.z)
                }
                .pickerStyle(.segmented)
            }
            
            // Enroll button
            Button(action: enroll) {
                if isEnrolling {
                    ProgressView()
                        .frame(maxWidth: .infinity)
                } else {
                    Text("Enroll with Optic ID")
                        .frame(maxWidth: .infinity)
                }
            }
            .buttonStyle(.borderedProminent)
            .disabled(name.isEmpty || isEnrolling)
            
            Text("Enrollment requires Optic ID verification")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding()
    }
    
    private func updatePreviewColor(name: String) {
        guard !name.isEmpty else {
            previewColor = nil
            return
        }
        
        var h: UInt64 = 0xcbf29ce484222325
        for byte in name.utf8 {
            h ^= UInt64(byte)
            h &*= 0x100000001b3
        }
        
        var z = h &+ 0x9e3779b97f4a7c15
        z = (z ^ (z >> 30)) &* 0xbf58476d1ce4e5b9
        z = (z ^ (z >> 27)) &* 0x94d049bb133111eb
        z = z ^ (z >> 31)
        
        previewColor = SIMD3<Float>(
            Float(z & 0xFFFF) / 65535.0,
            Float((z >> 16) & 0xFFFF) / 65535.0,
            Float((z >> 32) & 0xFFFF) / 65535.0
        )
    }
    
    private func enroll() {
        isEnrolling = true
        
        Task {
            do {
                let signer = try await onEnroll(name, selectedAxis)
                onComplete(signer)
            } catch {
                // Handle error
            }
            isEnrolling = false
        }
    }
}

// MARK: - Previews

#if DEBUG
struct MultisigUI_Previews: PreviewProvider {
    static var previews: some View {
        VStack {
            SignerCardView(
                signer: SignerSummary(
                    id: UUID(),
                    name: "Alice",
                    rgb: SIMD3<Float>(0.8, 0.2, 0.5),
                    celibacy: 0.85,
                    axis: .x
                ),
                isLocal: true
            )
            
            CoherenceMeterView(coherence: 0.67)
        }
        .padding()
        .background(Color.black)
    }
}
#endif
