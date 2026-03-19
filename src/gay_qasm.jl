# GayQASM: Chromatic Quantum Assembly with SPI Guarantees
# ═══════════════════════════════════════════════════════════════════════════════
#
# "Every qubit gets a color. Every gate preserves chromatic consistency."
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  QASM ↔ GAY CORRESPONDENCE                                                  │
# │                                                                             │
# │  QASM Gate      Gay.jl Equivalent           Chromatic Effect               │
# │  ─────────      ─────────────────           ────────────────               │
# │  H (Hadamard)   H_4 snapshot matrix         Superposition of colors        │
# │  CNOT           Para(Para(Gay#)) alignment  Control→Target color transfer  │
# │  X (Pauli-X)    Chromatic complement        RGB → (1-R, 1-G, 1-B)          │
# │  Z (Pauli-Z)    Phase flip                  Hue rotation by 180°           │
# │  T (π/8)        Fractional phase            Hue rotation by 45°            │
# │  S (π/4)        Quarter phase               Hue rotation by 90°            │
# │  MEASURE        Collapse to fingerprint     Color → seed projection        │
# │                                                                             │
# │  SPI GUARANTEE:                                                             │
# │    circuit(seed, gates) = circuit(seed, gates)  ∀ execution context        │
# │    Same circuit + same seed → same measurement colors                      │
# │                                                                             │
# │  ACSET INTEGRATION:                                                         │
# │    GayQASMACSet: Circuit as ACSet with chromatic attributes                │
# │    RelativisticACSet: Gate ordering respects causality (no FTL!)           │
# │    TraceACSet: Measurement history with stigmergic feedback                │
# └─────────────────────────────────────────────────────────────────────────────┘

module GayQASM

export
    # Core types
    GayQubit, GayGate, GayCircuit, GayMeasurement,
    GateType, H, X, Y, Z, S, T, CNOT, CZ, SWAP, RX, RY, RZ, MEASURE,
    
    # Circuit construction
    gay_qubit, gay_circuit, add_gate!, add_measure!,
    
    # Gate operations (chromatic)
    apply_gate!, hadamard_superposition, cnot_entangle,
    pauli_x, pauli_y, pauli_z, phase_s, phase_t,
    
    # Measurement
    measure_qubit!, collapse_to_fingerprint, measurement_color,
    
    # Circuit execution
    execute_circuit!, circuit_fingerprint, verify_circuit_spi,
    
    # QASM parsing
    parse_qasm, qasm_to_gay, gay_to_qasm,
    
    # ACSet integration
    GayQASMACSet, circuit_to_acset, acset_to_circuit,
    
    # Visualization
    render_circuit, qubit_color_timeline,
    
    # Demo
    demo_gay_qasm

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

const GAY_SEED = UInt64(1069)
const QASM_SEED = UInt64(0x0A5B)  # "QASM" compressed

# Hadamard H_2 for single qubit (normalized: 1/√2)
const H_2 = [1 1; 1 -1] ./ sqrt(2)

# Hadamard H_4 for 2-qubit operations (from ducklake_timetravel_walks.jl)
const H_4 = [
    1  1  1  1;
    1 -1  1 -1;
    1  1 -1 -1;
    1 -1 -1  1
] ./ 2

# Pauli matrices (for gate definitions)
const PAULI_X = [0 1; 1 0]
const PAULI_Y = [0 -im; im 0]
const PAULI_Z = [1 0; 0 -1]

# Phase gates
const PHASE_S = [1 0; 0 im]      # π/4 phase
const PHASE_T = [1 0; 0 exp(im*π/4)]  # π/8 phase

# ═══════════════════════════════════════════════════════════════════════════════
# SPLITMIX64 (SPI Core)
# ═══════════════════════════════════════════════════════════════════════════════

@inline function splitmix64(state::UInt64)::UInt64
    z = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ⊻ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    z ⊻ (z >> 31)
end

@inline function color_from_seed(seed::UInt64)
    r = splitmix64(seed)
    g = splitmix64(r)
    b = splitmix64(g)
    (r=(r >> 56) / 255.0, g=(g >> 56) / 255.0, b=(b >> 56) / 255.0)
end

@inline function seed_from_color(color::NamedTuple)::UInt64
    r_bits = UInt64(round(color.r * 255))
    g_bits = UInt64(round(color.g * 255))
    b_bits = UInt64(round(color.b * 255))
    (r_bits << 56) | (g_bits << 48) | (b_bits << 40)
end

# ═══════════════════════════════════════════════════════════════════════════════
# GATE TYPES
# ═══════════════════════════════════════════════════════════════════════════════

@enum GateType begin
    H       # Hadamard
    X       # Pauli-X (NOT)
    Y       # Pauli-Y
    Z       # Pauli-Z
    S       # Phase S (π/4)
    T       # Phase T (π/8)
    CNOT    # Controlled-NOT
    CZ      # Controlled-Z
    SWAP    # Swap qubits
    RX      # Rotation around X
    RY      # Rotation around Y
    RZ      # Rotation around Z
    MEASURE # Measurement
end

# Gate arity (number of qubits)
const GATE_ARITY = Dict(
    H => 1, X => 1, Y => 1, Z => 1, S => 1, T => 1,
    RX => 1, RY => 1, RZ => 1, MEASURE => 1,
    CNOT => 2, CZ => 2, SWAP => 2
)

# Gate seeds for chromatic identity
const GATE_SEEDS = Dict(
    H => splitmix64(QASM_SEED ⊻ UInt64('H')),
    X => splitmix64(QASM_SEED ⊻ UInt64('X')),
    Y => splitmix64(QASM_SEED ⊻ UInt64('Y')),
    Z => splitmix64(QASM_SEED ⊻ UInt64('Z')),
    S => splitmix64(QASM_SEED ⊻ UInt64('S')),
    T => splitmix64(QASM_SEED ⊻ UInt64('T')),
    CNOT => splitmix64(QASM_SEED ⊻ UInt64(0xC40T)),
    CZ => splitmix64(QASM_SEED ⊻ UInt64(0xCZ)),
    SWAP => splitmix64(QASM_SEED ⊻ UInt64(0x5BA9)),
    RX => splitmix64(QASM_SEED ⊻ UInt64(0x8888)),
    RY => splitmix64(QASM_SEED ⊻ UInt64(0x8889)),
    RZ => splitmix64(QASM_SEED ⊻ UInt64(0x888A)),
    MEASURE => splitmix64(QASM_SEED ⊻ UInt64(0xBEA5)),
)

# ═══════════════════════════════════════════════════════════════════════════════
# GAY QUBIT: Chromatic Quantum State
# ═══════════════════════════════════════════════════════════════════════════════

"""
    GayQubit

A qubit with chromatic identity. The color encodes the quantum state
in a way that preserves SPI through gate operations.

Fields:
- index: Qubit index in circuit
- seed: Chromatic seed (determines initial color)
- color: Current RGB color (evolves with gates)
- amplitude_0: Complex amplitude for |0⟩
- amplitude_1: Complex amplitude for |1⟩
- measured: Whether qubit has been measured
- measurement_result: Classical bit result (0 or 1)
- history: Trace of gate applications
"""
mutable struct GayQubit
    index::Int
    seed::UInt64
    color::NamedTuple{(:r, :g, :b), NTuple{3, Float64}}
    amplitude_0::ComplexF64
    amplitude_1::ComplexF64
    measured::Bool
    measurement_result::Union{Int, Nothing}
    history::Vector{Tuple{GateType, Int, UInt64}}  # (gate, step, fingerprint)
end

function gay_qubit(index::Int; seed::UInt64=GAY_SEED)::GayQubit
    qubit_seed = splitmix64(seed ⊻ UInt64(index))
    color = color_from_seed(qubit_seed)
    GayQubit(
        index,
        qubit_seed,
        color,
        ComplexF64(1.0, 0.0),  # Start in |0⟩
        ComplexF64(0.0, 0.0),
        false,
        nothing,
        Tuple{GateType, Int, UInt64}[]
    )
end

function qubit_fingerprint(q::GayQubit)::UInt64
    # XOR of seed, color encoding, and amplitude phases
    phase_0 = UInt64(round(angle(q.amplitude_0) * 1000)) & 0xFFFF
    phase_1 = UInt64(round(angle(q.amplitude_1) * 1000)) & 0xFFFF
    color_fp = seed_from_color(q.color)
    q.seed ⊻ color_fp ⊻ (phase_0 << 32) ⊻ (phase_1 << 48)
end

# ═══════════════════════════════════════════════════════════════════════════════
# GAY GATE: Chromatic Gate Application
# ═══════════════════════════════════════════════════════════════════════════════

"""
    GayGate

A quantum gate with chromatic identity and SPI tracking.
"""
struct GayGate
    type::GateType
    targets::Vector{Int}  # Qubit indices
    control::Union{Int, Nothing}  # For controlled gates
    angle::Union{Float64, Nothing}  # For rotation gates
    seed::UInt64
    color::NamedTuple{(:r, :g, :b), NTuple{3, Float64}}
    step::Int  # Position in circuit
end

function GayGate(type::GateType, targets::Vector{Int}, step::Int;
                 control::Union{Int, Nothing}=nothing,
                 angle::Union{Float64, Nothing}=nothing,
                 seed::UInt64=GATE_SEEDS[type])
    gate_seed = splitmix64(seed ⊻ UInt64(step) ⊻ reduce(⊻, UInt64.(targets); init=UInt64(0)))
    GayGate(type, targets, control, angle, gate_seed, color_from_seed(gate_seed), step)
end

# ═══════════════════════════════════════════════════════════════════════════════
# CHROMATIC GATE OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    hadamard_superposition(color) -> (color_0, color_1)

Apply Hadamard to color, creating chromatic superposition.
Returns two colors representing |0⟩ and |1⟩ components.
"""
function hadamard_superposition(color::NamedTuple)
    # Hadamard creates equal superposition
    # Chromatically: mix original with complement
    complement = (r=1.0-color.r, g=1.0-color.g, b=1.0-color.b)
    
    # |0⟩ component: (original + complement) / √2
    color_0 = (
        r = (color.r + complement.r) / sqrt(2),
        g = (color.g + complement.g) / sqrt(2),
        b = (color.b + complement.b) / sqrt(2)
    )
    
    # |1⟩ component: (original - complement) / √2 (can be negative, clamp to [0,1])
    color_1 = (
        r = clamp(abs(color.r - complement.r) / sqrt(2), 0.0, 1.0),
        g = clamp(abs(color.g - complement.g) / sqrt(2), 0.0, 1.0),
        b = clamp(abs(color.b - complement.b) / sqrt(2), 0.0, 1.0)
    )
    
    (color_0, color_1)
end

"""
    cnot_entangle(control_color, target_color) -> (new_control, new_target)

CNOT gate: flip target if control is "high" (brightness > 0.5).
Creates chromatic entanglement between qubits.
"""
function cnot_entangle(control_color::NamedTuple, target_color::NamedTuple)
    # Control brightness determines flip
    control_brightness = (control_color.r + control_color.g + control_color.b) / 3
    
    if control_brightness > 0.5
        # Flip target (complement)
        new_target = (r=1.0-target_color.r, g=1.0-target_color.g, b=1.0-target_color.b)
    else
        new_target = target_color
    end
    
    # Control unchanged, but gains XOR fingerprint from target
    control_seed = seed_from_color(control_color)
    target_seed = seed_from_color(target_color)
    entangled_seed = splitmix64(control_seed ⊻ target_seed)
    
    # New control has slight color shift from entanglement
    shift = (entangled_seed >> 56) / 255.0 * 0.1  # Small shift
    new_control = (
        r = clamp(control_color.r + shift, 0.0, 1.0),
        g = control_color.g,
        b = control_color.b
    )
    
    (new_control, new_target)
end

"""
    pauli_x(color) -> color

Pauli-X gate: NOT gate, chromatic complement.
"""
function pauli_x(color::NamedTuple)
    (r=1.0-color.r, g=1.0-color.g, b=1.0-color.b)
end

"""
    pauli_z(color) -> color

Pauli-Z gate: phase flip, hue rotation by 180°.
"""
function pauli_z(color::NamedTuple)
    # Convert to HSL, rotate hue by 180°, convert back
    # Simplified: swap R and B channels
    (r=color.b, g=color.g, b=color.r)
end

"""
    pauli_y(color) -> color

Pauli-Y gate: combination of X and Z with phase.
"""
function pauli_y(color::NamedTuple)
    # Y = iXZ: complement then hue rotate
    xed = pauli_x(color)
    pauli_z(xed)
end

"""
    phase_s(color) -> color

S gate: π/4 phase, 90° hue rotation.
"""
function phase_s(color::NamedTuple)
    # Rotate hue by 90°: R→G→B→R cycle shift
    (r=color.b, g=color.r, b=color.g)
end

"""
    phase_t(color) -> color

T gate: π/8 phase, 45° hue rotation.
"""
function phase_t(color::NamedTuple)
    # Rotate hue by 45°: interpolate between original and S
    s_color = phase_s(color)
    (
        r = (color.r + s_color.r) / 2,
        g = (color.g + s_color.g) / 2,
        b = (color.b + s_color.b) / 2
    )
end

"""
    rotation_gate(color, axis, angle) -> color

Rotation gate around axis by angle.
"""
function rotation_gate(color::NamedTuple, axis::Symbol, angle::Float64)
    # Rotation around axis affects other two channels
    c = cos(angle / 2)
    s = sin(angle / 2)
    
    if axis == :X
        # RX: affects G and B
        new_g = c * color.g - s * color.b
        new_b = s * color.g + c * color.b
        (r=color.r, g=clamp(new_g, 0.0, 1.0), b=clamp(new_b, 0.0, 1.0))
    elseif axis == :Y
        # RY: affects R and B
        new_r = c * color.r + s * color.b
        new_b = -s * color.r + c * color.b
        (r=clamp(new_r, 0.0, 1.0), g=color.g, b=clamp(new_b, 0.0, 1.0))
    else  # :Z
        # RZ: affects R and G
        new_r = c * color.r - s * color.g
        new_g = s * color.r + c * color.g
        (r=clamp(new_r, 0.0, 1.0), g=clamp(new_g, 0.0, 1.0), b=color.b)
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# GAY CIRCUIT: Chromatic Quantum Circuit
# ═══════════════════════════════════════════════════════════════════════════════

"""
    GayCircuit

A quantum circuit with chromatic tracking and SPI guarantees.
"""
mutable struct GayCircuit
    n_qubits::Int
    qubits::Vector{GayQubit}
    gates::Vector{GayGate}
    measurements::Vector{GayMeasurement}
    seed::UInt64
    current_step::Int
    fingerprint::UInt64
end

struct GayMeasurement
    qubit_index::Int
    step::Int
    result::Int  # 0 or 1
    collapsed_color::NamedTuple{(:r, :g, :b), NTuple{3, Float64}}
    fingerprint::UInt64
end

function gay_circuit(n_qubits::Int; seed::UInt64=GAY_SEED)::GayCircuit
    qubits = [gay_qubit(i; seed=seed) for i in 1:n_qubits]
    initial_fp = reduce(⊻, qubit_fingerprint.(qubits); init=seed)
    GayCircuit(n_qubits, qubits, GayGate[], GayMeasurement[], seed, 0, initial_fp)
end

function add_gate!(circuit::GayCircuit, gate_type::GateType, targets::Vector{Int};
                   control::Union{Int, Nothing}=nothing,
                   angle::Union{Float64, Nothing}=nothing)
    circuit.current_step += 1
    gate = GayGate(gate_type, targets, circuit.current_step;
                   control=control, angle=angle)
    push!(circuit.gates, gate)
    
    # Update circuit fingerprint
    circuit.fingerprint ⊻= gate.seed
    
    gate
end

function add_gate!(circuit::GayCircuit, gate_type::GateType, target::Int; kwargs...)
    add_gate!(circuit, gate_type, [target]; kwargs...)
end

function add_measure!(circuit::GayCircuit, qubit_index::Int)
    add_gate!(circuit, MEASURE, [qubit_index])
end

# ═══════════════════════════════════════════════════════════════════════════════
# CIRCUIT EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

"""
    apply_gate!(circuit, gate) -> circuit

Apply a gate to the circuit, updating qubit states and colors.
"""
function apply_gate!(circuit::GayCircuit, gate::GayGate)
    if gate.type == H
        # Hadamard on single qubit
        q = circuit.qubits[gate.targets[1]]
        
        # Update amplitudes
        new_0 = (q.amplitude_0 + q.amplitude_1) / sqrt(2)
        new_1 = (q.amplitude_0 - q.amplitude_1) / sqrt(2)
        q.amplitude_0 = new_0
        q.amplitude_1 = new_1
        
        # Update color (superposition)
        _, sup_color = hadamard_superposition(q.color)
        q.color = sup_color
        
        push!(q.history, (H, gate.step, qubit_fingerprint(q)))
        
    elseif gate.type == X
        q = circuit.qubits[gate.targets[1]]
        q.amplitude_0, q.amplitude_1 = q.amplitude_1, q.amplitude_0
        q.color = pauli_x(q.color)
        push!(q.history, (X, gate.step, qubit_fingerprint(q)))
        
    elseif gate.type == Y
        q = circuit.qubits[gate.targets[1]]
        new_0 = -im * q.amplitude_1
        new_1 = im * q.amplitude_0
        q.amplitude_0, q.amplitude_1 = new_0, new_1
        q.color = pauli_y(q.color)
        push!(q.history, (Y, gate.step, qubit_fingerprint(q)))
        
    elseif gate.type == Z
        q = circuit.qubits[gate.targets[1]]
        q.amplitude_1 = -q.amplitude_1
        q.color = pauli_z(q.color)
        push!(q.history, (Z, gate.step, qubit_fingerprint(q)))
        
    elseif gate.type == S
        q = circuit.qubits[gate.targets[1]]
        q.amplitude_1 *= im
        q.color = phase_s(q.color)
        push!(q.history, (S, gate.step, qubit_fingerprint(q)))
        
    elseif gate.type == T
        q = circuit.qubits[gate.targets[1]]
        q.amplitude_1 *= exp(im * π / 4)
        q.color = phase_t(q.color)
        push!(q.history, (T, gate.step, qubit_fingerprint(q)))
        
    elseif gate.type == CNOT
        control_idx = gate.control !== nothing ? gate.control : gate.targets[1]
        target_idx = gate.targets[end]
        
        qc = circuit.qubits[control_idx]
        qt = circuit.qubits[target_idx]
        
        # Simplified CNOT: flip target if control amplitude_1 dominates
        if abs2(qc.amplitude_1) > abs2(qc.amplitude_0)
            qt.amplitude_0, qt.amplitude_1 = qt.amplitude_1, qt.amplitude_0
        end
        
        # Chromatic entanglement
        new_control, new_target = cnot_entangle(qc.color, qt.color)
        qc.color = new_control
        qt.color = new_target
        
        entangle_fp = qubit_fingerprint(qc) ⊻ qubit_fingerprint(qt)
        push!(qc.history, (CNOT, gate.step, entangle_fp))
        push!(qt.history, (CNOT, gate.step, entangle_fp))
        
    elseif gate.type == SWAP
        i, j = gate.targets[1], gate.targets[2]
        circuit.qubits[i], circuit.qubits[j] = circuit.qubits[j], circuit.qubits[i]
        # Fix indices
        circuit.qubits[i].index = i
        circuit.qubits[j].index = j
        
    elseif gate.type in (RX, RY, RZ)
        q = circuit.qubits[gate.targets[1]]
        θ = gate.angle !== nothing ? gate.angle : 0.0
        axis = gate.type == RX ? :X : gate.type == RY ? :Y : :Z
        
        # Update amplitude (simplified)
        c = cos(θ / 2)
        s = sin(θ / 2)
        if axis == :X
            new_0 = c * q.amplitude_0 - im * s * q.amplitude_1
            new_1 = -im * s * q.amplitude_0 + c * q.amplitude_1
        elseif axis == :Y
            new_0 = c * q.amplitude_0 - s * q.amplitude_1
            new_1 = s * q.amplitude_0 + c * q.amplitude_1
        else  # Z
            new_0 = exp(-im * θ / 2) * q.amplitude_0
            new_1 = exp(im * θ / 2) * q.amplitude_1
        end
        q.amplitude_0, q.amplitude_1 = new_0, new_1
        
        q.color = rotation_gate(q.color, axis, θ)
        push!(q.history, (gate.type, gate.step, qubit_fingerprint(q)))
        
    elseif gate.type == MEASURE
        q = circuit.qubits[gate.targets[1]]
        
        # Collapse based on seed (deterministic measurement for SPI)
        prob_1 = abs2(q.amplitude_1)
        collapse_seed = splitmix64(q.seed ⊻ UInt64(gate.step))
        threshold = (collapse_seed >> 56) / 255.0
        
        if threshold < prob_1
            q.measurement_result = 1
            q.amplitude_0 = ComplexF64(0.0, 0.0)
            q.amplitude_1 = ComplexF64(1.0, 0.0)
        else
            q.measurement_result = 0
            q.amplitude_0 = ComplexF64(1.0, 0.0)
            q.amplitude_1 = ComplexF64(0.0, 0.0)
        end
        q.measured = true
        
        # Collapse color to definite state
        collapsed_color = q.measurement_result == 1 ? 
            pauli_x(color_from_seed(q.seed)) : color_from_seed(q.seed)
        q.color = collapsed_color
        
        measurement = GayMeasurement(
            q.index, gate.step, q.measurement_result,
            collapsed_color, qubit_fingerprint(q)
        )
        push!(circuit.measurements, measurement)
        push!(q.history, (MEASURE, gate.step, measurement.fingerprint))
    end
    
    # Update circuit fingerprint
    circuit.fingerprint = reduce(⊻, qubit_fingerprint.(circuit.qubits); init=circuit.seed)
    
    circuit
end

"""
    execute_circuit!(circuit) -> circuit

Execute all gates in the circuit.
"""
function execute_circuit!(circuit::GayCircuit)
    for gate in circuit.gates
        apply_gate!(circuit, gate)
    end
    circuit
end

"""
    circuit_fingerprint(circuit) -> UInt64

Get the current circuit fingerprint (XOR of all qubit states).
"""
function circuit_fingerprint(circuit::GayCircuit)::UInt64
    circuit.fingerprint
end

"""
    verify_circuit_spi(circuit, n_trials) -> Bool

Verify SPI: same circuit produces same fingerprint across trials.
"""
function verify_circuit_spi(circuit_builder::Function, n_trials::Int=100;
                            seed::UInt64=GAY_SEED)::Bool
    ref_circuit = circuit_builder(seed)
    execute_circuit!(ref_circuit)
    ref_fp = circuit_fingerprint(ref_circuit)
    
    for _ in 1:n_trials
        test_circuit = circuit_builder(seed)
        execute_circuit!(test_circuit)
        if circuit_fingerprint(test_circuit) != ref_fp
            return false
        end
    end
    
    true
end

# ═══════════════════════════════════════════════════════════════════════════════
# QASM PARSING
# ═══════════════════════════════════════════════════════════════════════════════

"""
    parse_qasm(qasm_string; seed=GAY_SEED) -> GayCircuit

Parse OpenQASM 2.0 to GayCircuit.
"""
function parse_qasm(qasm::String; seed::UInt64=GAY_SEED)::GayCircuit
    lines = filter(!isempty, strip.(split(qasm, '\n')))
    
    n_qubits = 0
    gates = Tuple{GateType, Vector{Int}, Union{Int, Nothing}, Union{Float64, Nothing}}[]
    
    for line in lines
        line = strip(line)
        
        # Skip comments and headers
        startswith(line, "//") && continue
        startswith(line, "OPENQASM") && continue
        startswith(line, "include") && continue
        
        # Parse qubit declaration
        m = match(r"qreg\s+(\w+)\[(\d+)\]", line)
        if m !== nothing
            n_qubits = parse(Int, m[2])
            continue
        end
        
        # Parse gates
        # H q[n]
        m = match(r"^h\s+\w+\[(\d+)\]", line)
        if m !== nothing
            push!(gates, (H, [parse(Int, m[1]) + 1], nothing, nothing))
            continue
        end
        
        # X q[n]
        m = match(r"^x\s+\w+\[(\d+)\]", line)
        if m !== nothing
            push!(gates, (X, [parse(Int, m[1]) + 1], nothing, nothing))
            continue
        end
        
        # Y q[n]
        m = match(r"^y\s+\w+\[(\d+)\]", line)
        if m !== nothing
            push!(gates, (Y, [parse(Int, m[1]) + 1], nothing, nothing))
            continue
        end
        
        # Z q[n]
        m = match(r"^z\s+\w+\[(\d+)\]", line)
        if m !== nothing
            push!(gates, (Z, [parse(Int, m[1]) + 1], nothing, nothing))
            continue
        end
        
        # S q[n]
        m = match(r"^s\s+\w+\[(\d+)\]", line)
        if m !== nothing
            push!(gates, (S, [parse(Int, m[1]) + 1], nothing, nothing))
            continue
        end
        
        # T q[n]
        m = match(r"^t\s+\w+\[(\d+)\]", line)
        if m !== nothing
            push!(gates, (T, [parse(Int, m[1]) + 1], nothing, nothing))
            continue
        end
        
        # CX q[c], q[t] (CNOT)
        m = match(r"^cx\s+\w+\[(\d+)\]\s*,\s*\w+\[(\d+)\]", line)
        if m !== nothing
            c = parse(Int, m[1]) + 1
            t = parse(Int, m[2]) + 1
            push!(gates, (CNOT, [c, t], c, nothing))
            continue
        end
        
        # RX(angle) q[n]
        m = match(r"^rx\(([^)]+)\)\s+\w+\[(\d+)\]", line)
        if m !== nothing
            angle = parse(Float64, replace(m[1], "pi" => "π", "π" => string(π)))
            push!(gates, (RX, [parse(Int, m[2]) + 1], nothing, angle))
            continue
        end
        
        # RY(angle) q[n]
        m = match(r"^ry\(([^)]+)\)\s+\w+\[(\d+)\]", line)
        if m !== nothing
            angle = parse(Float64, replace(m[1], "pi" => "π", "π" => string(π)))
            push!(gates, (RY, [parse(Int, m[2]) + 1], nothing, angle))
            continue
        end
        
        # RZ(angle) q[n]
        m = match(r"^rz\(([^)]+)\)\s+\w+\[(\d+)\]", line)
        if m !== nothing
            angle = parse(Float64, replace(m[1], "pi" => "π", "π" => string(π)))
            push!(gates, (RZ, [parse(Int, m[2]) + 1], nothing, angle))
            continue
        end
        
        # measure q[n] -> c[n]
        m = match(r"^measure\s+\w+\[(\d+)\]", line)
        if m !== nothing
            push!(gates, (MEASURE, [parse(Int, m[1]) + 1], nothing, nothing))
            continue
        end
    end
    
    # Build circuit
    circuit = gay_circuit(max(1, n_qubits); seed=seed)
    
    for (gate_type, targets, control, angle) in gates
        add_gate!(circuit, gate_type, targets; control=control, angle=angle)
    end
    
    circuit
end

"""
    gay_to_qasm(circuit) -> String

Convert GayCircuit to OpenQASM 2.0 string.
"""
function gay_to_qasm(circuit::GayCircuit)::String
    buf = IOBuffer()
    
    println(buf, "OPENQASM 2.0;")
    println(buf, "include \"qelib1.inc\";")
    println(buf, "")
    println(buf, "qreg q[$(circuit.n_qubits)];")
    println(buf, "creg c[$(circuit.n_qubits)];")
    println(buf, "")
    
    for gate in circuit.gates
        qasm_gate = if gate.type == H
            "h q[$(gate.targets[1] - 1)];"
        elseif gate.type == X
            "x q[$(gate.targets[1] - 1)];"
        elseif gate.type == Y
            "y q[$(gate.targets[1] - 1)];"
        elseif gate.type == Z
            "z q[$(gate.targets[1] - 1)];"
        elseif gate.type == S
            "s q[$(gate.targets[1] - 1)];"
        elseif gate.type == T
            "t q[$(gate.targets[1] - 1)];"
        elseif gate.type == CNOT
            c = gate.control !== nothing ? gate.control : gate.targets[1]
            t = gate.targets[end]
            "cx q[$(c - 1)], q[$(t - 1)];"
        elseif gate.type == RX
            "rx($(gate.angle)) q[$(gate.targets[1] - 1)];"
        elseif gate.type == RY
            "ry($(gate.angle)) q[$(gate.targets[1] - 1)];"
        elseif gate.type == RZ
            "rz($(gate.angle)) q[$(gate.targets[1] - 1)];"
        elseif gate.type == MEASURE
            "measure q[$(gate.targets[1] - 1)] -> c[$(gate.targets[1] - 1)];"
        else
            "// unknown gate"
        end
        
        println(buf, qasm_gate)
    end
    
    String(take!(buf))
end

# ═══════════════════════════════════════════════════════════════════════════════
# ACSET INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

"""
    GayQASMACSet

ACSet representation of a quantum circuit with chromatic attributes.

Schema:
- Qubit: quantum bits with color, seed, measurement state
- Gate: quantum gates with type, targets, chromatic effect
- Wire: connections between gates and qubits (causal ordering)
- Measurement: measurement results with collapsed colors
"""
struct GayQASMACSet
    n_qubits::Int
    n_gates::Int
    
    # Qubit attributes
    qubit_seeds::Vector{UInt64}
    qubit_colors::Vector{NamedTuple{(:r, :g, :b), NTuple{3, Float64}}}
    qubit_measured::Vector{Bool}
    qubit_results::Vector{Union{Int, Nothing}}
    
    # Gate attributes
    gate_types::Vector{GateType}
    gate_targets::Vector{Vector{Int}}
    gate_controls::Vector{Union{Int, Nothing}}
    gate_seeds::Vector{UInt64}
    gate_colors::Vector{NamedTuple{(:r, :g, :b), NTuple{3, Float64}}}
    
    # Wires: (from_gate_or_qubit, to_gate)
    wires::Vector{Tuple{Union{Tuple{:qubit, Int}, Tuple{:gate, Int}}, Int}}
    
    # Global attributes
    circuit_seed::UInt64
    circuit_fingerprint::UInt64
end

function circuit_to_acset(circuit::GayCircuit)::GayQASMACSet
    # Extract qubit attributes
    qubit_seeds = [q.seed for q in circuit.qubits]
    qubit_colors = [q.color for q in circuit.qubits]
    qubit_measured = [q.measured for q in circuit.qubits]
    qubit_results = [q.measurement_result for q in circuit.qubits]
    
    # Extract gate attributes
    gate_types = [g.type for g in circuit.gates]
    gate_targets = [g.targets for g in circuit.gates]
    gate_controls = [g.control for g in circuit.gates]
    gate_seeds = [g.seed for g in circuit.gates]
    gate_colors = [g.color for g in circuit.gates]
    
    # Build wires (causal dependencies)
    wires = Tuple{Union{Tuple{Symbol, Int}, Tuple{Symbol, Int}}, Int}[]
    
    # Track last gate affecting each qubit
    last_gate_for_qubit = Dict{Int, Union{Nothing, Int}}()
    for i in 1:circuit.n_qubits
        last_gate_for_qubit[i] = nothing
    end
    
    for (gate_idx, gate) in enumerate(circuit.gates)
        for target in gate.targets
            if last_gate_for_qubit[target] === nothing
                # Wire from qubit to first gate
                push!(wires, ((:qubit, target), gate_idx))
            else
                # Wire from previous gate to this gate
                push!(wires, ((:gate, last_gate_for_qubit[target]), gate_idx))
            end
            last_gate_for_qubit[target] = gate_idx
        end
        
        # Add control wire if CNOT-type gate
        if gate.control !== nothing && gate.control ∉ gate.targets
            if last_gate_for_qubit[gate.control] === nothing
                push!(wires, ((:qubit, gate.control), gate_idx))
            else
                push!(wires, ((:gate, last_gate_for_qubit[gate.control]), gate_idx))
            end
        end
    end
    
    GayQASMACSet(
        circuit.n_qubits,
        length(circuit.gates),
        qubit_seeds, qubit_colors, qubit_measured, qubit_results,
        gate_types, gate_targets, gate_controls, gate_seeds, gate_colors,
        wires,
        circuit.seed,
        circuit.fingerprint
    )
end

function acset_to_circuit(acset::GayQASMACSet)::GayCircuit
    circuit = gay_circuit(acset.n_qubits; seed=acset.circuit_seed)
    
    for i in 1:acset.n_gates
        add_gate!(circuit, acset.gate_types[i], acset.gate_targets[i];
                  control=acset.gate_controls[i])
    end
    
    circuit
end

# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

"""
    render_circuit(circuit) -> String

Render circuit as ANSI-colored ASCII diagram.
"""
function render_circuit(circuit::GayCircuit)::String
    buf = IOBuffer()
    
    println(buf, "╔═══════════════════════════════════════════════════════════════╗")
    println(buf, "║  GAY QUANTUM CIRCUIT ($(circuit.n_qubits) qubits, $(length(circuit.gates)) gates)  ║")
    println(buf, "╠═══════════════════════════════════════════════════════════════╣")
    
    # Create grid
    max_step = circuit.current_step + 1
    grid = fill("───", circuit.n_qubits, max_step)
    
    # Place gates
    for gate in circuit.gates
        symbol = string(gate.type)[1:min(1, length(string(gate.type)))]
        for t in gate.targets
            grid[t, gate.step] = "[$symbol]"
        end
        if gate.control !== nothing && gate.control ∉ gate.targets
            grid[gate.control, gate.step] = " ● "
        end
    end
    
    # Render each qubit line with color
    for q_idx in 1:circuit.n_qubits
        q = circuit.qubits[q_idx]
        r = round(Int, q.color.r * 255)
        g = round(Int, q.color.g * 255)
        b = round(Int, q.color.b * 255)
        
        print(buf, "║ q$(q_idx-1): ")
        print(buf, "\e[38;2;$(r);$(g);$(b)m")
        for step in 1:max_step
            print(buf, grid[q_idx, step])
        end
        print(buf, "\e[0m")
        
        if q.measured
            print(buf, " → $(q.measurement_result)")
        end
        println(buf, "  ║")
    end
    
    println(buf, "╠═══════════════════════════════════════════════════════════════╣")
    println(buf, "║  Fingerprint: 0x$(string(circuit.fingerprint, base=16)[1:min(12, end)])...    ║")
    println(buf, "╚═══════════════════════════════════════════════════════════════╝")
    
    String(take!(buf))
end

"""
    qubit_color_timeline(circuit) -> String

Show color evolution for each qubit through gates.
"""
function qubit_color_timeline(circuit::GayCircuit)::String
    buf = IOBuffer()
    
    println(buf, "QUBIT COLOR TIMELINE")
    println(buf, "════════════════════")
    
    for q in circuit.qubits
        print(buf, "q$(q.index-1): ")
        
        # Initial color
        init_color = color_from_seed(q.seed)
        r, g, b = round.(Int, [init_color.r, init_color.g, init_color.b] .* 255)
        print(buf, "\e[38;2;$(r);$(g);$(b)m●\e[0m")
        
        # Color after each gate
        for (gate_type, step, fp) in q.history
            gate_color = color_from_seed(fp)
            r, g, b = round.(Int, [gate_color.r, gate_color.g, gate_color.b] .* 255)
            print(buf, " →$(string(gate_type)[1])→ \e[38;2;$(r);$(g);$(b)m●\e[0m")
        end
        
        # Final color
        r, g, b = round.(Int, [q.color.r, q.color.g, q.color.b] .* 255)
        print(buf, " = \e[38;2;$(r);$(g);$(b)mRGB($(round(q.color.r, digits=2)), $(round(q.color.g, digits=2)), $(round(q.color.b, digits=2)))\e[0m")
        
        println(buf)
    end
    
    String(take!(buf))
end

# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

function demo_gay_qasm()
    println("═══ GAY QASM: Chromatic Quantum Assembly ═══")
    println()
    
    # 1. Create Bell state circuit
    println("1. BELL STATE CIRCUIT")
    circuit = gay_circuit(2)
    add_gate!(circuit, H, 1)
    add_gate!(circuit, CNOT, [1, 2]; control=1)
    add_gate!(circuit, MEASURE, 1)
    add_gate!(circuit, MEASURE, 2)
    
    execute_circuit!(circuit)
    println(render_circuit(circuit))
    println(qubit_color_timeline(circuit))
    println()
    
    # 2. Parse QASM
    println("2. PARSE OPENQASM")
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[3];
    creg c[3];
    
    h q[0];
    cx q[0], q[1];
    cx q[1], q[2];
    measure q[0] -> c[0];
    measure q[1] -> c[1];
    measure q[2] -> c[2];
    """
    
    ghz_circuit = parse_qasm(qasm)
    execute_circuit!(ghz_circuit)
    println("  GHZ state (3 qubits):")
    println(render_circuit(ghz_circuit))
    
    # 3. Round-trip QASM
    println("3. ROUND-TRIP QASM")
    exported = gay_to_qasm(ghz_circuit)
    println("  Exported:")
    println(exported)
    
    # 4. ACSet conversion
    println("4. ACSET INTEGRATION")
    acset = circuit_to_acset(ghz_circuit)
    println("  Qubits: $(acset.n_qubits)")
    println("  Gates: $(acset.n_gates)")
    println("  Wires: $(length(acset.wires))")
    println("  Circuit fingerprint: 0x$(string(acset.circuit_fingerprint, base=16))")
    println()
    
    # 5. SPI verification
    println("5. SPI VERIFICATION")
    builder = seed -> begin
        c = gay_circuit(2; seed=seed)
        add_gate!(c, H, 1)
        add_gate!(c, CNOT, [1, 2]; control=1)
        add_gate!(c, MEASURE, 1)
        add_gate!(c, MEASURE, 2)
        c
    end
    
    spi_ok = verify_circuit_spi(builder, 100)
    println("  100 trials: $(spi_ok ? "✓ SPI VERIFIED" : "✗ SPI FAILED")")
    println()
    
    # 6. Gate colors
    println("6. GATE CHROMATIC IDENTITY")
    for gate_type in [H, X, Y, Z, S, T, CNOT]
        color = color_from_seed(GATE_SEEDS[gate_type])
        r, g, b = round.(Int, [color.r, color.g, color.b] .* 255)
        println("  \e[38;2;$(r);$(g);$(b)m████\e[0m $(gate_type)")
    end
    
    println()
    println("═══════════════════════════════════════════════════════════════")
    println("  QASM is now gay: chromatic circuits with SPI guarantees")
    println("═══════════════════════════════════════════════════════════════")
    
    circuit
end

end # module
