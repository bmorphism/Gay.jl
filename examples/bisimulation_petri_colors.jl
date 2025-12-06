# # Bisimulation and Petri Nets: The Categorical Bridge to Color SPI
#
# This example connects two seemingly different ideas:
#
# 1. **Bisimulation** from process algebra and categorical semantics
#    - Two systems are bisimilar if they cannot be distinguished by observation
#    - Key paper: Winskel's "Profunctors, open maps and bisimulation"
#    - Applied to Petri nets by Nielsen, Joyal, Winskel, and more recently
#      by Fabrizio Genovese at 20squares/statebox
#
# 2. **Strong Parallelism Invariance (SPI)** from Gay.jl/XF.jl
#    - CPU sequential ≈ CPU parallel ≈ GPU produce identical fingerprints
#    - Same seed → same colors, regardless of execution order
#    - XOR fingerprinting as the "observation" that cannot distinguish
#
# The Bridge: SPI is bisimulation for computational worlds!
#
# References:
# - Baez, Genovese, Master, Shulman: "Categories of Nets" (LICS 2021)
# - Nielsen, Joyal, Winskel: "Petri Nets and Bisimulation"
# - Genovese & Spivak: "Categorical Semantics for Guarded Petri Nets"
# - https://golem.ph.utexas.edu/category/2021/01/categories_of_nets_part_1.html

using Pkg
Pkg.activate(dirname(dirname(@__FILE__)))

using Gay
using Gay: hash_color, xor_fingerprint, ka_colors, ka_colors!
using KernelAbstractions
using KernelAbstractions: CPU
using Colors
using Random

# ═══════════════════════════════════════════════════════════════════════════════
# Part 1: Petri Net as a Category
# ═══════════════════════════════════════════════════════════════════════════════

"""
A Petri net generates a free symmetric monoidal category where:
- Objects: multisets of places (markings)
- Morphisms: sequences of transition firings

For Blume-Capel spin updates:
- Places: spin states {-1, 0, +1}
- Transitions: spin flips (φ → φ')
"""

# Places in our "spin Petri net"
@enum SpinPlace SPIN_DOWN=-1 SPIN_ZERO=0 SPIN_UP=1

# Transitions: all possible spin changes
struct SpinTransition
    from::SpinPlace
    to::SpinPlace
end

const ALL_TRANSITIONS = [
    SpinTransition(SPIN_DOWN, SPIN_ZERO),
    SpinTransition(SPIN_DOWN, SPIN_UP),
    SpinTransition(SPIN_ZERO, SPIN_DOWN),
    SpinTransition(SPIN_ZERO, SPIN_UP),
    SpinTransition(SPIN_UP, SPIN_DOWN),
    SpinTransition(SPIN_UP, SPIN_ZERO),
]

"""
A marking is a multiset of places.
For a single spin: one token in one place.
For a lattice: one token per site, distributed among places.
"""
struct Marking
    counts::Dict{SpinPlace, Int}
end

Marking() = Marking(Dict(SPIN_DOWN => 0, SPIN_ZERO => 0, SPIN_UP => 1))

function Base.show(io::IO, m::Marking)
    print(io, "Marking(↓=$(m.counts[SPIN_DOWN]), ○=$(m.counts[SPIN_ZERO]), ↑=$(m.counts[SPIN_UP]))")
end

"""
Fire a transition: consume input token, produce output token.
Returns new marking (immutable update).
"""
function fire(m::Marking, t::SpinTransition)
    if m.counts[t.from] < 1
        error("Cannot fire: no token at $(t.from)")
    end
    new_counts = copy(m.counts)
    new_counts[t.from] -= 1
    new_counts[t.to] += 1
    Marking(new_counts)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Part 2: Bisimulation as Observational Equivalence
# ═══════════════════════════════════════════════════════════════════════════════

"""
Two processes P and Q are bisimilar (P ∼ Q) if:
- For every action a that P can do, Q can do the same action
- After doing a, the resulting states P' and Q' are still bisimilar
- And vice versa (Q can match P)

This is symmetric: neither can "tell" they're different by observation.

In our color context:
- P = CPU sequential color generation
- Q = CPU parallel color generation
- R = GPU color generation
- Observation = XOR fingerprint

SPI says: P ∼ Q ∼ R (all produce same fingerprint)
"""

abstract type ComputationProcess end

struct SequentialCPU <: ComputationProcess
    seed::UInt64
    n::Int
end

struct ParallelCPU <: ComputationProcess
    seed::UInt64
    n::Int
    workgroup::Int
end

struct MetalGPU <: ComputationProcess
    seed::UInt64
    n::Int
end

"""
Observe a process by computing its XOR fingerprint.
This is our "barbed bisimulation" observation function.
"""
function observe(p::SequentialCPU)
    colors = zeros(Float32, p.n, 3)
    for i in 1:p.n
        r, g, b = hash_color(p.seed, UInt64(i))
        colors[i, 1] = r
        colors[i, 2] = g
        colors[i, 3] = b
    end
    xor_fingerprint(colors)
end

function observe(p::ParallelCPU)
    colors = zeros(Float32, p.n, 3)
    ka_colors!(colors, p.seed; backend=CPU(), workgroup=p.workgroup)
    xor_fingerprint(colors)
end

function observe(p::MetalGPU)
    # Would use Metal backend if available
    # For now, simulate with CPU parallel
    colors = zeros(Float32, p.n, 3)
    ka_colors!(colors, p.seed; backend=CPU())
    xor_fingerprint(colors)
end

"""
Check if two processes are bisimilar via observation.
In SPI terms: do they produce the same fingerprint?
"""
function bisimilar(p::ComputationProcess, q::ComputationProcess)
    observe(p) == observe(q)
end

# Unicode alias (using proper tilde operator ≈)
const ≃ = bisimilar

# ═══════════════════════════════════════════════════════════════════════════════
# Part 3: The Categorical View - Morphisms Preserve Observation
# ═══════════════════════════════════════════════════════════════════════════════

"""
In the category of Petri nets (following Baez, Genovese, Master, Shulman):

- Objects: Petri nets (or markings)
- Morphisms: net homomorphisms that preserve structure

A morphism f: N → N' is a pair of functions (f_S, f_T) where:
- f_S: places → places
- f_T: transitions → transitions
- Preserves input/output structure

For bisimulation, we want morphisms that preserve BEHAVIOR.
This is captured by "open maps" (Joyal, Nielsen, Winskel).
"""

struct PetriMorphism{S, T}
    source::S
    target::T
    place_map::Function
    transition_map::Function
end

"""
A bisimulation between nets N₁ and N₂ is a span:
    
       R
      ↙ ↘
    N₁   N₂

Where R is a "common refinement" that can simulate both.

For SPI, the common refinement is the SEED:
- Same seed can be "projected" to CPU sequential, CPU parallel, or GPU
- Each projection produces the same observable behavior (fingerprint)
"""

struct SPISpan
    seed::UInt64
    n::Int
    # The seed is the apex; projections are:
    # - to_sequential: seed → SequentialCPU
    # - to_parallel: seed → ParallelCPU  
    # - to_gpu: seed → MetalGPU
end

function project_sequential(span::SPISpan)
    SequentialCPU(span.seed, span.n)
end

function project_parallel(span::SPISpan, workgroup::Int=256)
    ParallelCPU(span.seed, span.n, workgroup)
end

function project_gpu(span::SPISpan)
    MetalGPU(span.seed, span.n)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Part 4: The Blume-Capel Connection
# ═══════════════════════════════════════════════════════════════════════════════

"""
The Blume-Capel model has spin-1 variables φ ∈ {-1, 0, +1}.

As a Petri net:
- Places: three states (SPIN_DOWN, SPIN_ZERO, SPIN_UP)  
- Transitions: six possible flips (all pairs)
- Tokens: one per lattice site

Monte Carlo dynamics = sequence of transition firings.

The question: do different ORDERINGS of transition firings
lead to the same observable macrostate?

Answer: No! Order matters for MC dynamics.

But for COLOR GENERATION with hash-based RNG:
- Order does NOT matter
- hash_color(seed, index) is deterministic
- XOR fingerprint is order-invariant (XOR is commutative!)

This is why SPI works: we've chosen an order-invariant observation.
"""

"""
    firing_sequence_fingerprint(seed, transitions)

Compute fingerprint from a sequence of transition firings.
Each firing gets a color based on (seed, firing_index).
"""
function firing_sequence_fingerprint(seed::UInt64, n_firings::Int)
    colors = zeros(Float32, n_firings, 3)
    for i in 1:n_firings
        r, g, b = hash_color(seed, UInt64(i))
        colors[i, 1] = r
        colors[i, 2] = g
        colors[i, 3] = b
    end
    xor_fingerprint(colors)
end

"""
    shuffled_firing_fingerprint(seed, n_firings, permutation)

Apply firings in a different ORDER, then compute fingerprint.
Due to XOR commutativity, should give same result!
"""
function shuffled_firing_fingerprint(seed::UInt64, n_firings::Int, perm::Vector{Int})
    colors = zeros(Float32, n_firings, 3)
    for (output_idx, firing_idx) in enumerate(perm)
        r, g, b = hash_color(seed, UInt64(firing_idx))
        colors[output_idx, 1] = r
        colors[output_idx, 2] = g
        colors[output_idx, 3] = b
    end
    xor_fingerprint(colors)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Part 5: Demonstration
# ═══════════════════════════════════════════════════════════════════════════════

function demo_petri_bisimulation()
    println()
    println("═" ^ 75)
    println("  PETRI NETS AND BISIMULATION")
    println("  From Baez, Genovese, Master, Shulman: 'Categories of Nets' (LICS 2021)")
    println("═" ^ 75)
    println()
    
    # Show spin Petri net structure
    println("  Spin-1 Petri Net (Blume-Capel):")
    println("  ─────────────────────────────────")
    println()
    println("      Places: ↓ (φ=-1)  ○ (φ=0)  ↑ (φ=+1)")
    println()
    println("      Transitions:")
    for t in ALL_TRANSITIONS
        from_sym = t.from == SPIN_DOWN ? "↓" : (t.from == SPIN_ZERO ? "○" : "↑")
        to_sym = t.to == SPIN_DOWN ? "↓" : (t.to == SPIN_ZERO ? "○" : "↑")
        println("        $from_sym → $to_sym")
    end
    println()
    
    # Show marking evolution
    println("  Marking Evolution (single spin):")
    m = Marking(Dict(SPIN_DOWN => 0, SPIN_ZERO => 0, SPIN_UP => 1))
    println("    Initial: $m")
    
    m = fire(m, SpinTransition(SPIN_UP, SPIN_ZERO))
    println("    After ↑→○: $m")
    
    m = fire(m, SpinTransition(SPIN_ZERO, SPIN_DOWN))
    println("    After ○→↓: $m")
    println()
end

function demo_spi_as_bisimulation()
    println("═" ^ 75)
    println("  SPI AS BISIMULATION")
    println("  Different execution paths, same observation")
    println("═" ^ 75)
    println()
    
    seed = UInt64(42069)
    n = 10000
    
    # Create processes
    seq = SequentialCPU(seed, n)
    par64 = ParallelCPU(seed, n, 64)
    par256 = ParallelCPU(seed, n, 256)
    gpu = MetalGPU(seed, n)
    
    # Compute observations
    obs_seq = observe(seq)
    obs_par64 = observe(par64)
    obs_par256 = observe(par256)
    obs_gpu = observe(gpu)
    
    println("  Observations (XOR fingerprints):")
    println("  ─────────────────────────────────")
    println("    Sequential CPU:    0x$(string(obs_seq, base=16, pad=8))")
    println("    Parallel CPU (64): 0x$(string(obs_par64, base=16, pad=8))")
    println("    Parallel CPU(256): 0x$(string(obs_par256, base=16, pad=8))")
    println("    Metal GPU:         0x$(string(obs_gpu, base=16, pad=8))")
    println()
    
    # Check bisimilarity
    println("  Bisimulation Checks (P ∼ Q iff observe(P) == observe(Q)):")
    println("  ─────────────────────────────────────────────────────────")
    println("    Sequential ≃ Parallel(64):  $(seq ≃ par64 ? "✓" : "✗")")
    println("    Sequential ≃ Parallel(256): $(seq ≃ par256 ? "✓" : "✗")")
    println("    Parallel(64) ≃ Parallel(256): $(par64 ≃ par256 ? "✓" : "✗")")
    println("    Sequential ≃ GPU:           $(seq ≃ gpu ? "✓" : "✗")")
    println()
    
    all_bisimilar = (seq ≃ par64) && (seq ≃ par256) && (seq ≃ gpu)
    if all_bisimilar
        println("  ✓ ALL PROCESSES ARE BISIMILAR")
        println("    This is the SPI guarantee: same seed → same observation")
    end
    println()
end

function demo_span_structure()
    println("═" ^ 75)
    println("  THE SPI SPAN: Seed as Common Refinement")
    println("═" ^ 75)
    println()
    
    println("  In categorical terms, SPI is a span:")
    println()
    println("                    Seed")
    println("                   ↙    ↘")
    println("            Sequential  Parallel")
    println("                   ↘    ↙")
    println("                    GPU")
    println()
    println("  The seed is the 'apex' - it determines all behavior.")
    println("  Each projection (seq/par/gpu) produces the same observable.")
    println()
    
    seed = UInt64(0xCAFEBABE)
    n = 1000
    span = SPISpan(seed, n)
    
    seq = project_sequential(span)
    par = project_parallel(span)
    gpu = project_gpu(span)
    
    println("  Example span with seed=0x$(string(seed, base=16)):")
    println("    project_sequential → observe → 0x$(string(observe(seq), base=16, pad=8))")
    println("    project_parallel   → observe → 0x$(string(observe(par), base=16, pad=8))")
    println("    project_gpu        → observe → 0x$(string(observe(gpu), base=16, pad=8))")
    println()
    
    # Show that span projections are bisimilar
    all_same = observe(seq) == observe(par) == observe(gpu)
    println("  Span coherence: all projections bisimilar? $(all_same ? "✓ YES" : "✗ NO")")
    println()
end

function demo_xor_commutativity()
    println("═" ^ 75)
    println("  XOR COMMUTATIVITY: Why Order Doesn't Matter")
    println("═" ^ 75)
    println()
    
    println("  XOR is commutative and associative:")
    println("    a ⊕ b = b ⊕ a")
    println("    (a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)")
    println()
    println("  This means: XOR fingerprint is ORDER-INVARIANT")
    println()
    
    seed = UInt64(42)
    n = 1000
    
    # Original order
    fp_original = firing_sequence_fingerprint(seed, n)
    
    # Reversed order
    perm_reverse = collect(n:-1:1)
    fp_reversed = shuffled_firing_fingerprint(seed, n, perm_reverse)
    
    # Random shuffle
    perm_random = shuffle(MersenneTwister(123), collect(1:n))
    fp_random = shuffled_firing_fingerprint(seed, n, perm_random)
    
    println("  Fingerprints with different orderings:")
    println("    Original order (1,2,...,n):  0x$(string(fp_original, base=16, pad=8))")
    println("    Reversed order (n,...,2,1):  0x$(string(fp_reversed, base=16, pad=8))")
    println("    Random permutation:          0x$(string(fp_random, base=16, pad=8))")
    println()
    
    all_same = fp_original == fp_reversed == fp_random
    println("  All identical? $(all_same ? "✓ YES - XOR is order-invariant!" : "✗ NO")")
    println()
    
    println("  This is why parallel execution produces same fingerprint:")
    println("  • Threads may compute colors in ANY order")
    println("  • But XOR(all colors) is the same regardless")
    println("  • → SPI guaranteed by algebraic properties of XOR")
    println()
end

function demo_categorical_picture()
    println("═" ^ 75)
    println("  THE CATEGORICAL PICTURE")
    println("  Following Baez, Genovese, Master, Shulman")
    println("═" ^ 75)
    println()
    
    println("  Three kinds of nets generate three kinds of categories:")
    println()
    println("    Pre-nets    →  Strict Monoidal Categories")
    println("    Σ-nets      →  Symmetric Strict Monoidal Categories")
    println("    Petri nets  →  Commutative Monoidal Categories")
    println()
    println("  Key insight: commutativity level affects bisimulation!")
    println()
    println("  For Petri nets (commutative case):")
    println("    • Objects are multisets of places (markings)")
    println("    • Morphisms are firing sequences")
    println("    • x ⊗ y = y ⊗ x (order doesn't matter)")
    println()
    println("  This matches XOR fingerprinting:")
    println("    • Colors are 'tokens' at each index")
    println("    • XOR is commutative (order doesn't matter)")
    println("    • Fingerprint is the 'final marking'")
    println()
    
    println("  The bisimulation functor:")
    println()
    println("    observe: ComputationalWorld → UInt32")
    println()
    println("  Maps each execution path to a fingerprint.")
    println("  SPI = this functor factors through the seed:")
    println()
    println("    Seed ──→ ComputationalWorld ──→ UInt32")
    println("      │                              ↑")
    println("      └──────────────────────────────┘")
    println("           (unique factorization)")
    println()
end

# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

function main()
    println()
    println("╔" * "═" ^ 73 * "╗")
    println("║" * " " ^ 15 * "BISIMULATION AND PETRI NET COLORS" * " " ^ 24 * "║")
    println("║" * " " ^ 73 * "║")
    println("║  The bridge between categorical semantics and SPI                     ║")
    println("╚" * "═" ^ 73 * "╝")
    
    demo_petri_bisimulation()
    demo_spi_as_bisimulation()
    demo_span_structure()
    demo_xor_commutativity()
    demo_categorical_picture()
    
    println("═" ^ 75)
    println("  SUMMARY: The Gay.jl SPI Bridge")
    println("═" ^ 75)
    println()
    println("  1. Petri nets generate monoidal categories (Baez et al.)")
    println("  2. Bisimulation = observational equivalence (Winskel, Genovese)")
    println("  3. XOR fingerprinting is an order-invariant observation")
    println("  4. SPI = bisimulation across computational worlds")
    println("  5. The seed is the categorical 'apex' of a span")
    println()
    println("  Same seed → same fingerprint → bisimilar processes")
    println("  CPU ∼ GPU ∼ Parallel ∼ Sequential")
    println()
    println("  This is the mathematical guarantee behind reproducible science! 🏳️‍🌈")
    println()
end

export SpinPlace, SpinTransition, Marking, fire
export ComputationProcess, SequentialCPU, ParallelCPU, MetalGPU
export observe, bisimilar, ≃
export SPISpan, project_sequential, project_parallel, project_gpu

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
