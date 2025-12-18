module DescentTower

using Printf

export DescentLevel, DESCENT_TOWER, TOWER, depth_to_frequency, branches_to_chord, sonify_descent_level, world_descent_tower

struct DescentLevel
    depth::Int
    name::String
    branches::Int
    frequency::Float64
    chord::Vector{Float64}
end

# 3^depth branching factor, octave doubling C2→C9
const TOWER = [
    DescentLevel(0, "decide_sheaf_tree_shape", 1, 65.41, [1.0]),                          # C2 unison
    DescentLevel(1, "adhesion_filter", 3, 130.81, [1.0, 1.26, 1.5]),                      # C3 triad
    DescentLevel(2, "Set operations", 9, 261.63, [1.0, 1.26, 1.5, 1.78]),                 # C4 7th
    DescentLevel(3, "Julia runtime", 27, 523.25, [1.0, 1.26, 1.5, 1.78, 2.0]),            # C5 9th
    DescentLevel(4, "Atomic ops", 81, 1046.5, [1.0, 1.19, 1.33, 1.5, 1.68, 1.89]),        # C6 11th
    DescentLevel(5, "Bit-level", 243, 2093.0, [1.0, 1.12, 1.26, 1.41, 1.5, 1.68, 1.89]),  # C7 13th
    DescentLevel(6, "Transistor", 729, 4186.0, [1.0, 1.06, 1.12, 1.19, 1.26, 1.33, 1.41, 1.5]),  # C8 cluster
    DescentLevel(7, "Quantum/Decoherence", 2187, 8372.0, [1.0, 1.03, 1.06, 1.09, 1.12, 1.15, 1.19, 1.22, 1.26]),  # C9 microtonality
]

"""
    depth_to_frequency(d::Int) -> Float64

C2 (65.41 Hz) doubling per octave descent level.
"""
depth_to_frequency(d::Int) = 65.41 * 2^d

"""
    branches_to_chord(n::Int) -> Vector{Float64}

Map branch count to harmonic ratios. More branches = denser chord voicing.
"""
function branches_to_chord(n::Int)
    n == 1 && return [1.0]
    n <= 3 && return [1.0, 1.26, 1.5]              # major triad
    n <= 9 && return [1.0, 1.26, 1.5, 1.78]        # dominant 7th
    n <= 27 && return [1.0, 1.26, 1.5, 1.78, 2.0]  # 9th
    n <= 81 && return [1.0, 1.19, 1.33, 1.5, 1.68, 1.89]  # 11th
    n <= 243 && return [1.0, 1.12, 1.26, 1.41, 1.5, 1.68, 1.89]  # 13th
    return [1.0, 1.06, 1.12, 1.19, 1.26, 1.33, 1.41, 1.5]  # chromatic cluster
end

# Alias for export
const DESCENT_TOWER = TOWER

"""
    sonify_descent_level(level::DescentLevel) -> NamedTuple

Get audio parameters for a single descent level.
"""
function sonify_descent_level(level::DescentLevel)
    (
        depth = level.depth,
        name = level.name,
        frequency = level.frequency,
        chord_freqs = chord_frequencies(level),
        waveform = level.depth < 4 ? :sine : (level.depth < 6 ? :triangle : :noise),
        duration = 0.5 / (level.depth + 1)  # Faster at deeper levels
    )
end

"""
    chord_frequencies(level::DescentLevel) -> Vector{Float64}

Compute actual frequencies for a level's chord.
"""
chord_frequencies(level::DescentLevel) = level.frequency .* level.chord

"""
    world_descent_tower()

Visualize the 7-level sheaf decomposition tower.

Key physical correspondence at depth 7:
- Decoherence = Descent violation (gluing failure)
- Charge conservation = Sheaf condition (local→global coherence)
"""
function world_descent_tower()
    println("╔══════════════════════════════════════════════════════════════════════╗")
    println("║           WORLD DESCENT TOWER - Sheaf Decomposition                  ║")
    println("╠══════════════════════════════════════════════════════════════════════╣")
    
    for level in TOWER
        freqs = chord_frequencies(level)
        freq_str = join([@sprintf("%.1f", f) for f in freqs[1:min(3, end)]], ", ")
        if length(freqs) > 3
            freq_str *= "..."
        end
        
        bar = "█" ^ min(level.depth + 1, 8)
        
        @printf("║ %d │ %-24s │ 3^%d=%4d │ %7.1f Hz │ [%s]\n",
                level.depth, level.name, level.depth, level.branches, 
                level.frequency, freq_str)
    end
    
    println("╠══════════════════════════════════════════════════════════════════════╣")
    println("║  DEPTH 7 CORRESPONDENCE:                                             ║")
    println("║    Decoherence         ↔  Descent violation (gluing failure)         ║")
    println("║    Charge conservation ↔  Sheaf condition (local→global)             ║")
    println("╚══════════════════════════════════════════════════════════════════════╝")
    
    return TOWER
end

# Sonification helper: generate sample data for audio synthesis
function tower_to_samples(duration_per_level::Float64=0.5, sample_rate::Int=44100)
    samples = Float64[]
    t_total = 0.0
    
    for level in TOWER
        freqs = chord_frequencies(level)
        n_samples = round(Int, duration_per_level * sample_rate)
        
        for i in 1:n_samples
            t = i / sample_rate
            # Additive synthesis of chord
            sample = sum(sin(2π * f * t) / length(freqs) for f in freqs)
            # Envelope: fade in/out
            env = sin(π * i / n_samples)
            push!(samples, sample * env * 0.3)
        end
    end
    
    return samples
end

end # module
