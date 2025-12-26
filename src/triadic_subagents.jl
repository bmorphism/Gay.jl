# Triadic Subagents: Synthetic 3-Agent Parallelism via GF(3) Polarities
# ======================================================================
#
# Creates 3 synthetic subagents from a single seed using the trialectic:
#
#   MINUS (−)   : phase ≡ 0 (mod 3) - contraction, reduction, refactoring
#   ERGODIC (_) : phase ≡ 1 (mod 3) - observation, verification, invariants
#   PLUS (+)    : phase ≡ 2 (mod 3) - expansion, generation, exploration
#
# Key insight: SplitMix64's splittable nature allows us to derive 3 INDEPENDENT
# RNG streams from a single seed, each with its own polarity twist. These streams
# can run in parallel with zero coordination, and their XOR-combined fingerprints
# verify that all 3 contributed (schedule invariance).
#
# This maps to the 2-transducer structure from Loregian arXiv:2509.06769:
#   - Objects: color streams (one per agent)
#   - 1-cells: (Q, t) transducers between streams
#   - 2-cells: natural transformations (the trialectic cycle)
#
# The trialectic cycle forms a closed loop:
#   (−) ═α₋₍⟹ (_) ═α₍₊⟹ (+) ═α₊₋⟹ (−)
#
# Usage:
#   agents = Triad(seed)
#   
#   # Parallel sampling from all 3 agents
#   colors = parallel_sample!(agents, 100)
#   
#   # Verify schedule invariance
#   fp = combined_fingerprint(agents)

module TriadicSubagents

using Random

# Import from parent module
using ..Gay: GAY_SEED, splitmix64, GOLDEN, color_at, SRGB
using Colors: RGB

export Polarity, MINUS, ERGODIC, PLUS
export TriadicAgent, Triad
export sample_agent!, parallel_sample!, combined_fingerprint
export verify_triadic_spi, phase_to_polarity, polarity_twist
export world_triadic_subagents

# ═══════════════════════════════════════════════════════════════════════════
# GF(3) Polarities - The Trialectic
# ═══════════════════════════════════════════════════════════════════════════

"""
    Polarity

The three polarities of the trialectic, corresponding to GF(3) elements.
"""
@enum Polarity begin
    MINUS = 0    # Contraction: reduce, simplify, refactor
    ERGODIC = 1  # Afference: observe, verify, maintain invariants
    PLUS = 2     # Expansion: add, extend, explore
end

# Polarity twists (XOR constants for seed derivation)
const MINUS_TWIST   = 0x2d2d2d2d2d2d2d2d  # "-" repeated
const ERGODIC_TWIST = 0x5f5f5f5f5f5f5f5f  # "_" repeated  
const PLUS_TWIST    = 0x2b2b2b2b2b2b2b2b  # "+" repeated

"""
    polarity_twist(p::Polarity) -> UInt64

Get the XOR twist constant for a polarity.
"""
function polarity_twist(p::Polarity)
    p == MINUS && return MINUS_TWIST
    p == ERGODIC && return ERGODIC_TWIST
    return PLUS_TWIST
end

"""
    phase_to_polarity(phase::Integer) -> Polarity

Map phase number to polarity via GF(3): phase mod 3.
"""
function phase_to_polarity(phase::Integer)
    Polarity(mod(phase, 3))
end

"""
    next_polarity(p::Polarity) -> Polarity

Trialectic cycle: (−) → (_) → (+) → (−)
"""
function next_polarity(p::Polarity)
    Polarity(mod(Int(p) + 1, 3))
end

"""
    polarity_symbol(p::Polarity) -> String
"""
polarity_symbol(p::Polarity) = p == MINUS ? "−" : p == ERGODIC ? "_" : "+"

"""
    polarity_name(p::Polarity) -> String
"""
polarity_name(p::Polarity) = p == MINUS ? "MINUS" : p == ERGODIC ? "ERGODIC" : "PLUS"

# ═══════════════════════════════════════════════════════════════════════════
# Triadic Agent: Single Agent in the Triad
# ═══════════════════════════════════════════════════════════════════════════

"""
    TriadicAgent

A single synthetic agent with its own polarity and RNG stream.

Fields:
- `polarity`: MINUS, ERGODIC, or PLUS
- `seed`: Base seed XOR'd with polarity twist
- `state`: Current SplitMix64 state
- `phase`: Number of samples taken
- `fingerprint`: XOR of all generated color hashes
"""
mutable struct TriadicAgent
    polarity::Polarity
    seed::UInt64
    state::UInt64
    phase::Int
    fingerprint::UInt64
end

"""
    TriadicAgent(polarity, base_seed)

Create a triadic agent with independent RNG stream derived from base seed.
"""
function TriadicAgent(polarity::Polarity, base_seed::UInt64)
    # Derive agent seed: base XOR polarity twist
    agent_seed = base_seed ⊻ polarity_twist(polarity)
    # Initialize state via SplitMix64
    state = splitmix64(agent_seed)
    
    TriadicAgent(polarity, agent_seed, state, 0, agent_seed)
end

"""
    sample_agent!(agent::TriadicAgent) -> (RGB, phase)

Generate next color from agent's independent stream.
Updates state and fingerprint.
"""
function sample_agent!(agent::TriadicAgent)
    # Advance state
    agent.state = splitmix64(agent.state)
    
    # Generate RGB from state bits
    r = Float64((agent.state >> 16) & 0xFF) / 255.0
    g = Float64((agent.state >> 8) & 0xFF) / 255.0
    b = Float64(agent.state & 0xFF) / 255.0
    color = RGB(r, g, b)
    
    # Update fingerprint (XOR accumulation)
    agent.fingerprint = agent.fingerprint ⊻ agent.state
    
    # Advance phase
    agent.phase += 1
    
    return (color, agent.phase)
end

"""
    agent_fingerprint(agent::TriadicAgent) -> UInt64

Get current fingerprint for this agent.
"""
agent_fingerprint(agent::TriadicAgent) = agent.fingerprint

# ═══════════════════════════════════════════════════════════════════════════
# Triadic Subagents: The Full Triad
# ═══════════════════════════════════════════════════════════════════════════

"""
    Triad

Three synthetic subagents with independent, parallel color streams.

The key insight: SplitMix64's splittable structure means we can derive
3 completely independent RNG streams from a single seed. Each stream
has its own polarity twist, ensuring no correlation between agents.

XOR fingerprint combination is schedule-invariant:
  fp₁ ⊻ fp₂ ⊻ fp₃ is the same regardless of sampling order.

This enables massively parallel color generation where:
  - Each agent can run on a different thread/core
  - No synchronization needed during sampling
  - Final fingerprint verifies all agents contributed
"""
struct Triad
    seed::UInt64
    minus::TriadicAgent
    ergodic::TriadicAgent
    plus::TriadicAgent
end

"""
    Triad(seed=GAY_SEED)

Create 3 synthetic subagents from a single seed.
"""
function Triad(seed::UInt64=GAY_SEED)
    Triad(
        seed,
        TriadicAgent(MINUS, seed),
        TriadicAgent(ERGODIC, seed),
        TriadicAgent(PLUS, seed)
    )
end

Triad(seed::Integer) = Triad(UInt64(seed))

"""
    get_agent(agents::Triad, p::Polarity) -> TriadicAgent
"""
function get_agent(agents::Triad, p::Polarity)
    p == MINUS && return agents.minus
    p == ERGODIC && return agents.ergodic
    return agents.plus
end

"""
    all_agents(agents::Triad) -> Vector{TriadicAgent}
"""
all_agents(agents::Triad) = [agents.minus, agents.ergodic, agents.plus]

# ═══════════════════════════════════════════════════════════════════════════
# Parallel Sampling
# ═══════════════════════════════════════════════════════════════════════════

"""
    parallel_sample!(agents::Triad, n::Int) -> Dict{Polarity, Vector}

Sample n colors from each agent IN PARALLEL.
Returns dict mapping polarity to color sequence.

The agents are independent, so this can run on 3 threads with zero coordination.
"""
function parallel_sample!(agents::Triad, n::Int)
    result = Dict{Polarity, Vector{Tuple{RGB{Float64}, Int}}}()
    
    # These can run in parallel (Threads.@spawn) - no shared state
    Threads.@threads for p in instances(Polarity)
        agent = get_agent(agents, p)
        colors = Vector{Tuple{RGB{Float64}, Int}}(undef, n)
        for i in 1:n
            colors[i] = sample_agent!(agent)
        end
        result[p] = colors
    end
    
    return result
end

"""
    interleaved_sample!(agents::Triad, n::Int) -> Vector

Sample n colors round-robin across all 3 agents.
Returns sequence of (color, polarity, phase) tuples.
"""
function interleaved_sample!(agents::Triad, n::Int)
    results = Vector{Tuple{RGB{Float64}, Polarity, Int}}()
    sizehint!(results, n)
    
    for i in 1:n
        p = phase_to_polarity(i - 1)
        agent = get_agent(agents, p)
        color, phase = sample_agent!(agent)
        push!(results, (color, p, phase))
    end
    
    return results
end

# ═══════════════════════════════════════════════════════════════════════════
# Fingerprint Verification
# ═══════════════════════════════════════════════════════════════════════════

"""
    combined_fingerprint(agents::Triad) -> UInt64

XOR-combine fingerprints from all 3 agents.
This is SCHEDULE-INVARIANT: same colors in any order produce same fingerprint.
"""
function combined_fingerprint(agents::Triad)
    agents.seed ⊻ 
    agent_fingerprint(agents.minus) ⊻
    agent_fingerprint(agents.ergodic) ⊻
    agent_fingerprint(agents.plus)
end

"""
    per_agent_fingerprints(agents::Triad) -> Dict

Get fingerprint from each agent.
"""
function per_agent_fingerprints(agents::Triad)
    Dict(
        MINUS => agent_fingerprint(agents.minus),
        ERGODIC => agent_fingerprint(agents.ergodic),
        PLUS => agent_fingerprint(agents.plus)
    )
end

"""
    verify_triadic_spi(agents1::Triad, agents2::Triad) -> Bool

Verify two agent triads with same seed have same combined fingerprint,
regardless of the order colors were sampled.
"""
function verify_triadic_spi(agents1::Triad, agents2::Triad)
    combined_fingerprint(agents1) == combined_fingerprint(agents2)
end

# ═══════════════════════════════════════════════════════════════════════════
# Trialectic Transformations (2-cells)
# ═══════════════════════════════════════════════════════════════════════════

"""
    transform_minus_to_ergodic(color, seed) -> RGB

2-cell: MINUS ⟹ ERGODIC (contraction → observation)
"""
function transform_minus_to_ergodic(color::RGB, seed::UInt64)
    # Twist the color through the transition
    hash_val = splitmix64(seed ⊻ UInt64(round(color.r * 255)) ⊻ MINUS_TWIST ⊻ ERGODIC_TWIST)
    r = Float64((hash_val >> 16) & 0xFF) / 255.0
    g = Float64((hash_val >> 8) & 0xFF) / 255.0
    b = Float64(hash_val & 0xFF) / 255.0
    RGB(r, g, b)
end

"""
    transform_ergodic_to_plus(color, seed) -> RGB

2-cell: ERGODIC ⟹ PLUS (observation → expansion)
"""
function transform_ergodic_to_plus(color::RGB, seed::UInt64)
    hash_val = splitmix64(seed ⊻ UInt64(round(Int, color.g * 255) << 8) ⊻ ERGODIC_TWIST ⊻ PLUS_TWIST)
    r = Float64((hash_val >> 16) & 0xFF) / 255.0
    g = Float64((hash_val >> 8) & 0xFF) / 255.0
    b = Float64(hash_val & 0xFF) / 255.0
    RGB(r, g, b)
end

"""
    transform_plus_to_minus(color, seed) -> RGB

2-cell: PLUS ⟹ MINUS (expansion → contraction) - completes the cycle
"""
function transform_plus_to_minus(color::RGB, seed::UInt64)
    hash_val = splitmix64(seed ⊻ UInt64(round(Int, color.b * 255) << 16) ⊻ PLUS_TWIST ⊻ MINUS_TWIST)
    r = Float64((hash_val >> 16) & 0xFF) / 255.0
    g = Float64((hash_val >> 8) & 0xFF) / 255.0
    b = Float64(hash_val & 0xFF) / 255.0
    RGB(r, g, b)
end

"""
    trialectic_cycle(color, seed) -> RGB

Apply full trialectic cycle: (−) → (_) → (+) → (−)
"""
function trialectic_cycle(color::RGB, seed::UInt64)
    c1 = transform_minus_to_ergodic(color, seed)
    c2 = transform_ergodic_to_plus(c1, seed)
    transform_plus_to_minus(c2, seed)
end

# ═══════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════

function ansi_rgb(c::RGB)
    r = round(Int, clamp(c.r, 0, 1) * 255)
    g = round(Int, clamp(c.g, 0, 1) * 255)
    b = round(Int, clamp(c.b, 0, 1) * 255)
    "\e[48;2;$(r);$(g);$(b)m"
end
const RESET = "\e[0m"
const BOLD = "\e[1m"
const DIM = "\e[2m"

function polarity_color(p::Polarity)
    p == MINUS && return "\e[31m"    # Red
    p == ERGODIC && return "\e[33m"  # Yellow
    return "\e[32m"                  # Green
end

"""
    visualize_agents(agents::Triad)

Display the current state of all 3 agents.
"""
function visualize_agents(agents::Triad)
    println()
    println("╔══════════════════════════════════════════════════════════════════╗")
    println("║  $(BOLD)TRIADIC SUBAGENTS$(RESET) - Synthetic Parallel Color Streams          ║")
    println("╚══════════════════════════════════════════════════════════════════╝")
    println()
    println("  Seed: 0x$(string(agents.seed, base=16, pad=16))")
    println()
    
    for p in instances(Polarity)
        agent = get_agent(agents, p)
        c = polarity_color(p)
        fp_hex = string(agent.fingerprint, base=16, pad=16)[1:12]
        
        println("  $(c)$(polarity_symbol(p)) $(polarity_name(p))$(RESET)")
        println("    Seed:        0x$(string(agent.seed, base=16, pad=16))")
        println("    Phase:       $(agent.phase)")
        println("    Fingerprint: 0x$(fp_hex)...")
        println()
    end
    
    fp = combined_fingerprint(agents)
    println("  Combined XOR Fingerprint: 0x$(string(fp, base=16, pad=16))")
    println()
end

"""
    visualize_interleaved(colors::Vector)

Display interleaved colors with polarity markers.
"""
function visualize_interleaved(colors::Vector{Tuple{RGB{Float64}, Polarity, Int}}; width::Int=60)
    println()
    println("  Interleaved sequence ($(min(length(colors), width)) colors):")
    println()
    print("  ")
    for (i, (c, p, _)) in enumerate(colors[1:min(end, width)])
        print("$(ansi_rgb(c))$(polarity_symbol(p))$(RESET)")
    end
    println()
    println()
    
    # Legend
    for p in instances(Polarity)
        c = polarity_color(p)
        println("  $(c)$(polarity_symbol(p))$(RESET) = $(polarity_name(p)) (phase ≡ $(Int(p)) mod 3)")
    end
    println()
end

# ═══════════════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════════════

"""
    world_triadic_subagents(; seed=GAY_SEED, n_samples=30)

Build composable triadic subagents state with schedule invariance verification.
"""
function world_triadic_subagents(; seed::UInt64=GAY_SEED, n_samples::Int=30)
    # Create two identical agent triads
    agents1 = Triad(seed)
    agents2 = Triad(seed)

    # Agents1: sample in order (MINUS, ERGODIC, PLUS)
    for _ in 1:n_samples
        sample_agent!(agents1.minus)
        sample_agent!(agents1.ergodic)
        sample_agent!(agents1.plus)
    end

    # Agents2: sample in reverse order (PLUS, ERGODIC, MINUS)
    for _ in 1:n_samples
        sample_agent!(agents2.plus)
        sample_agent!(agents2.ergodic)
        sample_agent!(agents2.minus)
    end

    fp1 = combined_fingerprint(agents1)
    fp2 = combined_fingerprint(agents2)

    # Interleaved sampling
    agents3 = Triad(seed)
    interleaved = interleaved_sample!(agents3, 45)  # 15 per agent
    fp3 = combined_fingerprint(agents3)

    fps1 = per_agent_fingerprints(agents1)
    fps3 = per_agent_fingerprints(agents3)

    start_color = RGB(0.8, 0.2, 0.3)
    cycled = trialectic_cycle(start_color, seed)

    (
        agents = (
            sequential = agents1,
            reverse = agents2,
            interleaved = agents3,
        ),
        fingerprints = (
            sequential = fp1,
            reverse = fp2,
            interleaved = fp3,
            schedule_invariant = fp1 == fp2 && fp1 == fp3,
        ),
        per_agent = (
            sequential = fps1,
            interleaved = fps3,
        ),
        trialectic = (
            start = start_color,
            cycled = cycled,
        ),
        seed = seed,
        n_samples = n_samples,
    )
end

end # module
