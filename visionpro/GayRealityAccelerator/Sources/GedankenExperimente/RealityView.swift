/// Vision Pro RealityKit visualization for Gedankenexperimente
/// Spatial rendering of quantum/parallel color experiments

#if os(visionOS)
import SwiftUI
import RealityKit
import GayColors

// MARK: - Experiment Visualization View

@available(visionOS 1.0, *)
public struct GedankenRealityView: View {
    @State private var experimentResults: [ExperimentResult] = []
    @State private var selectedExperiment: Int = 0
    
    public init() {}
    
    public var body: some View {
        HStack {
            // Experiment selector
            VStack(alignment: .leading) {
                Text("Gedankenexperimente")
                    .font(.title)
                
                ForEach(0..<6) { idx in
                    Button(experimentNames[idx]) {
                        selectedExperiment = idx
                        runExperiment(idx)
                    }
                    .buttonStyle(.bordered)
                    .tint(selectedExperiment == idx ? .blue : .gray)
                }
                
                Spacer()
                
                if !experimentResults.isEmpty {
                    Text(experimentResults[selectedExperiment].passed ? "✓ Passed" : "✗ Failed")
                        .font(.headline)
                        .foregroundColor(experimentResults[selectedExperiment].passed ? .green : .red)
                    
                    Text(experimentResults[selectedExperiment].message)
                        .font(.caption)
                        .frame(maxWidth: 200)
                }
            }
            .padding()
            .frame(width: 250)
            
            // 3D Reality view
            RealityView { content in
                // Add experiment visualization entities
                let anchor = AnchorEntity(world: .zero)
                content.add(anchor)
                
                // Initial spheres representing color palette
                addColorSpheres(to: anchor, colors: ThreadPalette(seed: GAY_SEED).rgbColors)
            } update: { content in
                // Update when experiment changes
                if let anchor = content.entities.first as? AnchorEntity,
                   selectedExperiment < experimentResults.count {
                    updateVisualization(anchor: anchor, result: experimentResults[selectedExperiment])
                }
            }
            .frame(minWidth: 400, minHeight: 400)
        }
        .onAppear {
            experimentResults = runAllGedankenexperimente()
        }
    }
    
    private let experimentNames = [
        "BE CNOT BE",
        "DO CNOT DO", 
        "CNOT CNOT Undecide",
        "SPI Parallel",
        "Chi-Gong",
        "Galois Connection"
    ]
    
    private func runExperiment(_ idx: Int) {
        let experiments: [Gedankenexperiment] = [
            BeCnotBeExperiment(),
            DoCnotDoExperiment(),
            UndecisionExperiment(),
            SPIExperiment(),
            ChiGongExperiment(),
            GaloisExperiment(),
        ]
        if idx < experiments.count {
            let result = experiments[idx].run()
            if idx < experimentResults.count {
                experimentResults[idx] = result
            }
        }
    }
    
    private func addColorSpheres(to anchor: AnchorEntity, colors: [RGB]) {
        for (i, color) in colors.enumerated() {
            let sphere = MeshResource.generateSphere(radius: 0.05)
            var material = SimpleMaterial()
            material.color = .init(tint: .init(
                red: CGFloat(color.r),
                green: CGFloat(color.g),
                blue: CGFloat(color.b),
                alpha: 1.0
            ))
            
            let entity = ModelEntity(mesh: sphere, materials: [material])
            
            // Arrange in hexagon
            let angle = Float(i) * .pi / 3
            let radius: Float = 0.15
            entity.position = [
                cos(angle) * radius,
                sin(angle) * radius,
                0
            ]
            
            anchor.addChild(entity)
        }
    }
    
    private func updateVisualization(anchor: AnchorEntity, result: ExperimentResult) {
        // Remove old children
        anchor.children.removeAll()
        
        // Add new color spheres based on result
        addColorSpheres(to: anchor, colors: result.colors.isEmpty 
            ? ThreadPalette(seed: GAY_SEED).rgbColors 
            : Array(result.colors.prefix(6)))
        
        // Add magnetization indicator if present
        if let mag = result.magnetization {
            let indicator = MeshResource.generateBox(size: [0.02, abs(mag) * 0.3, 0.02])
            var material = SimpleMaterial()
            material.color = .init(tint: mag > 0 ? .red : .blue)
            
            let entity = ModelEntity(mesh: indicator, materials: [material])
            entity.position = [0.3, mag * 0.15, 0]
            anchor.addChild(entity)
        }
    }
}

// MARK: - CNOT Gate Visualization

@available(visionOS 1.0, *)
public struct CNOTGateView: View {
    let controlState: Spin
    let targetState: Spin
    @State private var afterCNOT: Spin = .down
    
    public init(control: Spin, target: Spin) {
        self.controlState = control
        self.targetState = target
    }
    
    public var body: some View {
        RealityView { content in
            let anchor = AnchorEntity(world: .zero)
            
            // Control qubit (top)
            let controlSphere = MeshResource.generateSphere(radius: 0.04)
            var controlMat = SimpleMaterial()
            controlMat.color = .init(tint: controlState == .up ? .green : .gray)
            let controlEntity = ModelEntity(mesh: controlSphere, materials: [controlMat])
            controlEntity.position = [0, 0.1, 0]
            anchor.addChild(controlEntity)
            
            // Target qubit (bottom)  
            let targetSphere = MeshResource.generateSphere(radius: 0.04)
            var targetMat = SimpleMaterial()
            targetMat.color = .init(tint: targetState == .up ? .red : .blue)
            let targetEntity = ModelEntity(mesh: targetSphere, materials: [targetMat])
            targetEntity.position = [0, -0.1, 0]
            anchor.addChild(targetEntity)
            
            // Control line
            let line = MeshResource.generateBox(size: [0.005, 0.2, 0.005])
            var lineMat = SimpleMaterial()
            lineMat.color = .init(tint: .white)
            let lineEntity = ModelEntity(mesh: line, materials: [lineMat])
            lineEntity.position = [0, 0, 0]
            anchor.addChild(lineEntity)
            
            // XOR symbol at target
            let xorRing = MeshResource.generateBox(size: [0.06, 0.005, 0.005])
            let xorEntity = ModelEntity(mesh: xorRing, materials: [lineMat])
            xorEntity.position = [0, -0.1, 0]
            anchor.addChild(xorEntity)
            
            content.add(anchor)
        }
    }
}

// MARK: - Magnetized Sexp 3D Tree

@available(visionOS 1.0, *)
public struct MagnetizedSexpView: View {
    let sexp: GaySexpr
    
    public init(sexp: GaySexpr) {
        self.sexp = sexp
    }
    
    public var body: some View {
        RealityView { content in
            let anchor = AnchorEntity(world: .zero)
            addSexpNode(sexp, to: anchor, position: .zero, depth: 0)
            content.add(anchor)
        }
    }
    
    private func addSexpNode(_ node: GaySexpr, to parent: Entity, position: SIMD3<Float>, depth: Int) {
        // Node sphere with spin-dependent size
        let radius: Float = node.spin == .up ? 0.03 : 0.025
        let sphere = MeshResource.generateSphere(radius: radius)
        
        var material = SimpleMaterial()
        material.color = .init(tint: .init(
            red: CGFloat(node.color.r),
            green: CGFloat(node.color.g),
            blue: CGFloat(node.color.b),
            alpha: node.isWronglyMagnetized ? 0.5 : 1.0
        ))
        
        let entity = ModelEntity(mesh: sphere, materials: [material])
        entity.position = position
        parent.addChild(entity)
        
        // Add children in arc
        let childCount = node.children.count
        for (i, child) in node.children.enumerated() {
            let angle = Float(i) / Float(max(childCount - 1, 1)) * .pi - .pi / 2
            let childPos = position + SIMD3<Float>(
                cos(angle) * 0.1,
                -0.08,
                sin(angle) * 0.05
            )
            
            // Connection line
            let lineLength = simd_length(childPos - position)
            let line = MeshResource.generateBox(size: [0.003, lineLength, 0.003])
            var lineMat = SimpleMaterial()
            lineMat.color = .init(tint: .white.withAlphaComponent(0.3))
            let lineEntity = ModelEntity(mesh: line, materials: [lineMat])
            lineEntity.position = (position + childPos) / 2
            parent.addChild(lineEntity)
            
            addSexpNode(child, to: parent, position: childPos, depth: depth + 1)
        }
    }
}

#endif

// MARK: - Cross-platform test runner (macOS/iOS fallback)

#if !os(visionOS)
import Foundation

public func runGedankenexperimenteCLI() {
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  Gay.jl Reality Accelerator - Gedankenexperimente             ║")
    print("║  (CLI mode - run on visionOS for spatial visualization)       ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print()
    
    let results = runAllGedankenexperimente()
    
    print()
    print("Summary:")
    let passed = results.filter { $0.passed }.count
    print("  \(passed)/\(results.count) experiments passed")
    
    // Show color palettes
    print()
    print("Color palettes (hex):")
    for (i, result) in results.enumerated() {
        let hexes = result.colors.prefix(6).map { $0.hex }.joined(separator: " ")
        print("  Experiment \(i + 1): \(hexes)")
    }
}
#endif
