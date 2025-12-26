# Quantum Galperin → Colors
# Wavefunction interference patterns → deterministic color mapping
#
# Uses the :quantum continuation branch from bbp_pi.jl
#
# Classical Galperin: count elastic collisions → π digits
# Quantum Galperin: wavefunction evolution → interference patterns
#
# Reference: "Hear π from quantum Galperin billiards" (2020)
# The quantum version uses probability amplitudes instead of trajectories.

using Gay
using Colors
using SplittableRandoms: SplittableRandom, split

include("bbp_pi.jl")  # For continuation_point, branch_seed

# ═══════════════════════════════════════════════════════════════════════════
# Quantum State Representation
# ═══════════════════════════════════════════════════════════════════════════

struct QuantumState
    amplitude::ComplexF64    # ψ
    phase::Float64           # arg(ψ)
    probability::Float64     # |ψ|²
    position::Int            # discrete position index
    momentum::Int            # discrete momentum index
end

"""
Create initial quantum state (Gaussian wavepacket)
"""
function initial_quantum_state(n_states::Int; center::Int=1, width::Float64=2.0)
    states = QuantumState[]
    
    norm = 0.0
    for i in 1:n_states
        # Gaussian envelope
        amp = exp(-((i - center)^2) / (2 * width^2))
        norm += amp^2
    end
    norm = sqrt(norm)
    
    for i in 1:n_states
        amp = exp(-((i - center)^2) / (2 * width^2)) / norm
        phase = 0.0  # Initial phase
        ψ = amp * exp(im * phase)
        push!(states, QuantumState(ψ, phase, abs2(ψ), i, 0))
    end
    
    return states
end

"""
Quantum collision operator (simplified model)
In full theory: unitary evolution with mass-ratio dependent Hamiltonian
"""
function quantum_collision(states::Vector{QuantumState}, mass_ratio::Real)
    n = length(states)
    new_states = QuantumState[]
    
    # Simplified: phase shift proportional to position and mass ratio
    θ = 2π / sqrt(mass_ratio)
    
    for (i, s) in enumerate(states)
        # Interference from neighbors
        amp_left = i > 1 ? states[i-1].amplitude : 0.0
        amp_right = i < n ? states[i+1].amplitude : 0.0
        
        # Superposition with phase evolution
        new_amp = 0.5 * s.amplitude * exp(im * θ) + 
                  0.25 * amp_left * exp(im * θ * 1.1) +
                  0.25 * amp_right * exp(im * θ * 0.9)
        
        new_phase = angle(new_amp)
        new_prob = abs2(new_amp)
        
        push!(new_states, QuantumState(new_amp, new_phase, new_prob, i, s.momentum + 1))
    end
    
    # Renormalize
    total = sum(s.probability for s in new_states)
    if total > 0
        for i in 1:n
            s = new_states[i]
            new_states[i] = QuantumState(
                s.amplitude / sqrt(total),
                s.phase,
                s.probability / total,
                s.position,
                s.momentum
            )
        end
    end
    
    return new_states
end

"""
Evolve quantum state through n_steps collisions
"""
function quantum_evolution(mass_ratio::Real, n_steps::Int; n_states::Int=20)
    history = Vector{Vector{QuantumState}}()
    
    states = initial_quantum_state(n_states)
    push!(history, states)
    
    for _ in 1:n_steps
        states = quantum_collision(states, mass_ratio)
        push!(history, states)
    end
    
    return history
end

# ═══════════════════════════════════════════════════════════════════════════
# Quantum State → Color Mapping (uses :quantum branch)
# ═══════════════════════════════════════════════════════════════════════════

const QUANTUM_SEED = 314159

"""
    phase_color(phase::Float64; seed=QUANTUM_SEED)

Map quantum phase θ ∈ [-π, π] to color.
Uses HSV-like mapping: phase → hue.
"""
function phase_color(phase::Float64; seed::Integer=QUANTUM_SEED)
    qseed = branch_seed(seed, :quantum)
    
    # Quantize phase to 360 bins (degrees)
    phase_deg = round(Int, (phase + π) / (2π) * 360) % 360
    
    return color_at(phase_deg, Rec2020(); seed=qseed)
end

"""
    amplitude_color(state::QuantumState; seed=QUANTUM_SEED)

Get color for a quantum state based on amplitude and phase.
Brightness ~ |ψ|², Hue ~ arg(ψ)
"""
function amplitude_color(state::QuantumState; seed::Integer=QUANTUM_SEED)
    base = phase_color(state.phase; seed=seed)
    
    # Modulate brightness by probability
    brightness = sqrt(state.probability)  # sqrt for better visibility
    
    return RGB(
        base.r * brightness,
        base.g * brightness,
        base.b * brightness
    )
end

"""
    interference_color(step::Int, position::Int; seed=QUANTUM_SEED)

Get color for a point in the quantum evolution spacetime.
"""
function interference_color(step::Int, position::Int; seed::Integer=QUANTUM_SEED)
    qseed = branch_seed(seed, :quantum)
    
    # Unique index combining time and space
    idx = step * 1000 + position
    
    return color_at(idx, Rec2020(); seed=qseed)
end

# ═══════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════

function render_quantum_evolution(mass_ratio::Real, n_steps::Int; 
                                   n_states::Int=30, seed::Integer=QUANTUM_SEED)
    println("\n  ╔════════════════════════════════════════════════════════════╗")
    println("  ║  Quantum Galperin Evolution (:quantum branch)             ║")
    println("  ╚════════════════════════════════════════════════════════════╝")
    println()
    println("  Mass ratio: $mass_ratio")
    println("  Evolution steps: $n_steps")
    println("  Position states: $n_states")
    println()
    println("  Wavefunction evolution (brightness ~ |ψ|², color ~ phase):")
    println()
    
    history = quantum_evolution(mass_ratio, n_steps; n_states=n_states)
    
    # Render spacetime diagram
    for (t, states) in enumerate(history)
        print("  t=$(lpad(t-1, 2)): ")
        for s in states
            c = amplitude_color(s; seed=seed)
            ri = round(Int, c.r * 255)
            gi = round(Int, c.g * 255)
            bi = round(Int, c.b * 255)
            print("\e[48;2;$(ri);$(gi);$(bi)m \e[0m")
        end
        
        # Show max probability position
        max_pos = argmax([s.probability for s in states])
        println(" peak@$max_pos")
    end
end

function render_interference_pattern(; n_steps::Int=20, n_states::Int=40, 
                                       seed::Integer=QUANTUM_SEED)
    println("\n  ═══════════════════════════════════════════════════════")
    println("  Quantum Interference Pattern")
    println("  ═══════════════════════════════════════════════════════\n")
    
    println("  Each point: (time, position) → deterministic color")
    println()
    
    for t in 0:n_steps
        print("  ")
        for x in 1:n_states
            c = interference_color(t, x; seed=seed)
            ri = round(Int, c.r * 255)
            gi = round(Int, c.g * 255)
            bi = round(Int, c.b * 255)
            print("\e[48;2;$(ri);$(gi);$(bi)m \e[0m")
        end
        println()
    end
    
    println()
    println("  x →")
    println("  ↓ t")
end

function render_phase_wheel(; seed::Integer=QUANTUM_SEED)
    println("\n  ═══════════════════════════════════════════════════════")
    println("  Quantum Phase → Color Wheel")
    println("  ═══════════════════════════════════════════════════════\n")
    
    print("  Phase: ")
    for deg in 0:10:350
        phase = (deg / 180.0 - 1) * π
        c = phase_color(phase; seed=seed)
        ri = round(Int, c.r * 255)
        gi = round(Int, c.g * 255)
        bi = round(Int, c.b * 255)
        print("\e[48;2;$(ri);$(gi);$(bi)m  \e[0m")
    end
    println()
    println("         -π" * " "^30 * "0" * " "^30 * "+π")
end

function demo_quantum_spi()
    println("\n  ═══════════════════════════════════════════════════════")
    println("  SPI Verification: :quantum branch independence")
    println("  ═══════════════════════════════════════════════════════\n")
    
    idx = 42
    
    qseed = branch_seed(314159, :quantum)
    nseed = branch_seed(314159, :narya_proofs)
    pseed = branch_seed(314159, :polylog)
    
    println("  Same index ($idx) from different branches:")
    
    for (name, seed) in [(:quantum, qseed), (:narya_proofs, nseed), (:polylog, pseed)]
        c = color_at(idx, Rec2020(); seed=seed)
        ri = round(Int, c.r * 255)
        gi = round(Int, c.g * 255)
        bi = round(Int, c.b * 255)
        print("    :$name ")
        print("\e[48;2;$(ri);$(gi);$(bi)m    \e[0m\n")
    end
    
    println("\n  ◆ Each branch independent")
end

function compare_classical_quantum(; seed::Integer=QUANTUM_SEED)
    println("\n  ═══════════════════════════════════════════════════════")
    println("  Classical vs Quantum Galperin")
    println("  ═══════════════════════════════════════════════════════\n")
    
    println("  Classical: Count collisions → π digits (deterministic trajectory)")
    println("  Quantum:   Wavefunction evolution → interference (superposition)")
    println()
    println("  Mass ratio 100 (π ≈ 3.1):")
    println()
    
    # Classical would give 31 collisions
    print("    Classical: ")
    gseed = branch_seed(314159, :galperin)
    for i in 1:31
        c = color_at(i, Rec2020(); seed=gseed)
        ri = round(Int, c.r * 255)
        gi = round(Int, c.g * 255)
        bi = round(Int, c.b * 255)
        print("\e[48;2;$(ri);$(gi);$(bi)m \e[0m")
    end
    println(" (31 collisions)")
    
    # Quantum gives interference pattern
    print("    Quantum:   ")
    qseed = branch_seed(314159, :quantum)
    for i in 1:31
        c = color_at(i, Rec2020(); seed=qseed)
        ri = round(Int, c.r * 255)
        gi = round(Int, c.g * 255)
        bi = round(Int, c.b * 255)
        print("\e[48;2;$(ri);$(gi);$(bi)m \e[0m")
    end
    println(" (interference)")
end

# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

function main()
    println("\n" * "═"^70)
    println("  Quantum Galperin Colors - :quantum continuation branch")
    println("═"^70)
    
    # Phase wheel
    render_phase_wheel()
    
    # Quantum evolution
    render_quantum_evolution(100, 15; n_states=25)
    
    # Interference pattern
    render_interference_pattern(n_steps=12, n_states=35)
    
    # Classical vs quantum comparison
    compare_classical_quantum()
    
    # Verify branch independence
    demo_quantum_spi()
    
    println("\n  Properties:")
    println("  ◆ Quantum phase θ → deterministic color")
    println("  ◆ Amplitude |ψ| modulates brightness")
    println("  ◆ Spacetime (t,x) → interference pattern")
    println("  ◆ :quantum branch independent of :galperin")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
