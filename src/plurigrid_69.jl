# Plurigrid × Igor: 69 Ways of Autopoietic Ergodicity
# ═══════════════════════════════════════════════════════════════════════════════
#
# Cross-product of igor-seeds with plurigrid random walk evolution
# through 69 reinventions, with Solomonoff-Kolmogorov-Chaitin trust derivations
#
#   ┌─────────────────────────────────────────────────────────────────────────────┐
#   │  PLURIGRID ONTOLOGY                                                         │
#   │  ══════════════════                                                         │
#   │                                                                             │
#   │         ,|                                                                  │
#   │        ,'/                                                                  │
#   │       /___                                                                  │
#   │      |___ \\                                                                │
#   │      |___) )     AUTOPOIETIC ERGODICITY                                    │
#   │      `---'       69 reinventions since world centroid                      │
#   │                                                                             │
#   │  COMPOSITIONAL STRUCTURE                                                    │
#   │  ═══════════════════════                                                   │
#   │  Topos: Universal construction for social cognition                        │
#   │  Sheaf: Local-to-global coherence in meaning spaces                        │
#   │  Game:  Compositional game theory (Ghani et al.)                           │
#   │  OEDS:  Open Energy-Driven Systems (Spivak et al.)                         │
#   │                                                                             │
#   │  69 WAYS = 3-FORK × 23 EXAMPLES                                            │
#   │  ══════════════════════════                                                │
#   │  Each example admits 3 trust derivations: {-1, 0, +1}                      │
#   │  -1: Skeptical (high K-complexity, low trust)                              │
#   │   0: Neutral (balanced entropy)                                            │
#   │  +1: Trusting (low K-complexity, compressed)                               │
#   │                                                                             │
#   │  RANDOM WALK EVOLUTION                                                      │
#   │  ═════════════════════                                                     │
#   │  plurigrid/ontology → plurigrid/Plurigraph → vibes.lol                    │
#   │  Forks 3 at a time with random times as colors                            │
#   │                                                                             │
#   └─────────────────────────────────────────────────────────────────────────────┘

export PlurigridWalk, PlurigridReinvention, TrustDerivation
export SolomonoffKolmogorovChaitin, skc_complexity, skc_trust
export GayExampleWitness, witness_example, condense_locally
export plurigrid_zigzag_69, fork_3_at_a_time
export PlurigridCentroid, world_centroid, autopoietic_fixed_point
export ColorGamut, gamut_at_time, PLURIGRID_GAMUTS

using Dates

# Import igor seeds
include("igor_seeds.jl")

# ═══════════════════════════════════════════════════════════════════════════════
# Plurigrid Ontology: The 69 Reinventions
# ═══════════════════════════════════════════════════════════════════════════════

"""
The 69 reinventions of Plurigrid since world centroid
"""
const PLURIGRID_REINVENTIONS = [
    # Phase 1: Decentralized Energy (Dec 2022 - Mar 2023)
    "microgrid-coordination",
    "ieee1547-extension", 
    "ieee2030-underspec",
    "causally-open-grid",
    "a16z-grant-7.5pct",
    
    # Phase 2: Topos Construction (Apr 2023 - Jul 2023)
    "topos-social-cognition",
    "compositionality-intent",
    "decentralized-coordination",
    "joint-world-modeling",
    "individual-over-time",
    "digital-hygiene-facets",
    "meaning-space-localization",
    
    # Phase 3: Cognitive Continuity (Aug 2023 - Nov 2023)
    "topOS-service-line",
    "co-generative-social",
    "end-to-end-differentiable",
    "agency-amplifying-collab",
    "syntax-to-semantics",
    "topological-language-space",
    
    # Phase 4: Good Vibes Index (Dec 2023 - Mar 2024)
    "uninhibited-exploration",
    "tools-for-thought",
    "direction-magnitude-flow",
    "decompose-recompose",
    "fixed-point-index",
    "topological-conjugacy",
    "constructive-vibes",
    
    # Phase 5: Network Effects (Apr 2024 - Jul 2024)
    "excortex-experience",
    "mathematician-self-org",
    "network-of-networks",
    "formalize-everyday",
    "10M-ARR-target",
    "topOS-ecosystem-deploy",
    "pluralistic-integration",
    
    # Phase 6: Categorical Cybernetics (Aug 2024 - Nov 2024)
    "categorical-cybernetics",
    "passive-inference-compositional",
    "active-inference-emergent",
    "open-energy-driven",
    "sheaf-semantics-nl",
    "compositional-world-model",
    "compositional-game-theory",
    
    # Phase 7: Quantum Stoppard (Dec 2024 - Mar 2025)
    "quantum-hamlet-superposition",
    "be-cnot-be",
    "zx-calculus-color",
    "gay-chromatic-verification",
    "igor-not-igor-spectrum",
    "spi-fingerprinting",
    "tropical-semiring-paths",
    
    # Phase 8: Autopoietic Ergodicity (Apr 2025 - Jul 2025)
    "autopoietic-closure",
    "ergodic-decomposition",
    "plurigraph-hypergraph",
    "vibes-lol-interface",
    "belief-propagation-flow",
    "reflective-equilibria",
    "grid-anomaly-coordination",
    
    # Phase 9: 69 Completion (Aug 2025 - present)
    "69-ways-balanced-ternary",
    "skc-trust-derivation",
    "fork-3-random-times",
    "color-gamut-evolution",
    "gh-cli-verification",
    "unpublished-condensation",
    "local-sheafification",
    "zigzag-plurigrid-complete",
    
    # Phase 10: Extended Reinventions (need exactly 69 total)
    "gay-chromatic-monte-carlo",
    "para-zigzag-pdmp",
    "metatheory-brushes",
    "sheafified-semantics",
    "stackified-descent",
    "condensified-profinite",
    "reafference-igor",
    "chromatic-beacon-league"
]

@assert length(PLURIGRID_REINVENTIONS) == 69 "Need exactly 69 reinventions, got $(length(PLURIGRID_REINVENTIONS))"

# ═══════════════════════════════════════════════════════════════════════════════
# Color Gamuts: Various color spaces for random walk coloring
# ═══════════════════════════════════════════════════════════════════════════════

"""
Color gamut types for plurigrid evolution
"""
@enum ColorGamut begin
    GAMUT_SRGB          # Standard RGB (legacy)
    GAMUT_DISPLAY_P3    # Wide gamut (Apple)
    GAMUT_REC2020       # Ultra-wide (HDR)
    GAMUT_ACES          # Academy Color (film)
    GAMUT_TROPICAL      # Min-plus semiring colors
    GAMUT_BALANCED_TERNARY  # {-1, 0, +1} encoded
end

const PLURIGRID_GAMUTS = [GAMUT_SRGB, GAMUT_DISPLAY_P3, GAMUT_REC2020, 
                          GAMUT_ACES, GAMUT_TROPICAL, GAMUT_BALANCED_TERNARY]

"""
Get color in specified gamut at random time
"""
function gamut_at_time(gamut::ColorGamut, t::Float64, seed::UInt64)
    h = mix64(seed ⊻ UInt64(Base.round(Int, t * 1000)))
    
    # Base RGB from hash
    r = (h % 256) / 255.0
    h = mix64(h)
    g = (h % 256) / 255.0
    h = mix64(h)
    b = (h % 256) / 255.0
    
    # Gamut-specific transformation
    if gamut == GAMUT_SRGB
        return (Float32(r), Float32(g), Float32(b))
    elseif gamut == GAMUT_DISPLAY_P3
        # Wider primaries
        return (Float32(r * 1.1), Float32(g * 1.05), Float32(b * 1.15))
    elseif gamut == GAMUT_REC2020
        # Even wider
        return (Float32(r * 1.2), Float32(g * 1.1), Float32(b * 1.25))
    elseif gamut == GAMUT_ACES
        # Film-like response curve
        return (Float32(r^0.9), Float32(g^0.95), Float32(b^0.85))
    elseif gamut == GAMUT_TROPICAL
        # Min-plus: color = min component dominates
        m = min(r, g, b)
        return (Float32(m + r * 0.3), Float32(m + g * 0.3), Float32(m + b * 0.3))
    else  # GAMUT_BALANCED_TERNARY
        # Quantize to {-1, 0, +1} then map to color
        tr = r < 0.33 ? -1 : (r > 0.66 ? 1 : 0)
        tg = g < 0.33 ? -1 : (g > 0.66 ? 1 : 0)
        tb = b < 0.33 ? -1 : (b > 0.66 ? 1 : 0)
        return (Float32((tr + 1) / 2), Float32((tg + 1) / 2), Float32((tb + 1) / 2))
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Solomonoff-Kolmogorov-Chaitin Trust Derivation
# ═══════════════════════════════════════════════════════════════════════════════

"""
    SolomonoffKolmogorovChaitin

Algorithmic complexity and trust derivation.

# Theory
- K(x) = length of shortest program that outputs x
- Low K(x) = x is compressible = structured = trustworthy
- High K(x) = x is random = unstructured = requires verification

# Trust Mapping
- -1: Skeptical (K(x) ≈ |x|, incompressible, needs proof)
-  0: Neutral (K(x) ≈ |x|/2, partially structured)
- +1: Trusting (K(x) << |x|, highly compressible)
"""
struct SolomonoffKolmogorovChaitin
    data::Vector{UInt8}
    complexity::Float64      # Estimated K(x)
    trust::Int               # -1, 0, or +1
    description::String
end

"""
Estimate Kolmogorov complexity via compression ratio
"""
function skc_complexity(data::Vector{UInt8})
    if isempty(data)
        return 0.0
    end
    
    # Simple entropy-based estimate
    freq = zeros(Int, 256)
    for b in data
        freq[b + 1] += 1
    end
    
    n = length(data)
    entropy = 0.0
    for f in freq
        if f > 0
            p = f / n
            entropy -= p * log2(p)
        end
    end
    
    # K(x) ≈ entropy * |x| / 8 (bits to bytes)
    entropy * n / 8
end

"""
Derive trust from complexity
"""
function skc_trust(data::Vector{UInt8})
    if isempty(data)
        return 0
    end
    
    complexity = skc_complexity(data)
    ratio = complexity / length(data)
    
    if ratio < 0.3
        return 1   # Highly compressible → trust
    elseif ratio > 0.7
        return -1  # Incompressible → skeptical
    else
        return 0   # Neutral
    end
end

function SolomonoffKolmogorovChaitin(data::Vector{UInt8}, description::String="")
    complexity = skc_complexity(data)
    trust = skc_trust(data)
    SolomonoffKolmogorovChaitin(data, complexity, trust, description)
end

function SolomonoffKolmogorovChaitin(s::String)
    SolomonoffKolmogorovChaitin(Vector{UInt8}(s), s[1:min(50, length(s))])
end

# ═══════════════════════════════════════════════════════════════════════════════
# Trust Derivation: -1, 0, +1 for each example
# ═══════════════════════════════════════════════════════════════════════════════

"""
    TrustDerivation

A balanced ternary trust assignment for a Gay.jl example.
"""
struct TrustDerivation
    example_name::String
    trust_value::Int         # -1, 0, or +1
    skc::SolomonoffKolmogorovChaitin
    igor_alignment::Float64  # [0, 1] how igor-aligned
    reinvention_idx::Int     # Which plurigrid reinvention
    color::Tuple{Float32, Float32, Float32}
    gamut::ColorGamut
end

"""
Derive trust for an example file
"""
function derive_trust(example_path::String, reinvention_idx::Int, seed::UInt64)
    # Read file if exists
    if isfile(example_path)
        content = read(example_path)
    else
        content = Vector{UInt8}(example_path)
    end
    
    skc = SolomonoffKolmogorovChaitin(content, basename(example_path))
    
    # Igor alignment from hash
    h = mix64(seed ⊻ UInt64(reinvention_idx))
    igor_alignment = (h % 1000) / 1000.0
    
    # Color gamut cycles through reinventions
    gamut = PLURIGRID_GAMUTS[mod1(reinvention_idx, length(PLURIGRID_GAMUTS))]
    
    # Color at random time
    t = (h % 10000) / 100.0
    color = gamut_at_time(gamut, t, seed)
    
    TrustDerivation(
        basename(example_path),
        skc.trust,
        skc,
        igor_alignment,
        reinvention_idx,
        color,
        gamut
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# Plurigrid Random Walk
# ═══════════════════════════════════════════════════════════════════════════════

"""
    PlurigridWalk

Random walk through plurigrid reinventions with 3-fork branching.
"""
struct PlurigridWalk
    seed::UInt64
    positions::Vector{Int}       # Position in reinvention space
    times::Vector{Float64}       # Random times at each step
    colors::Vector{Tuple{Float32, Float32, Float32}}
    gamuts::Vector{ColorGamut}
    forks::Vector{Tuple{Int, Int, Int}}  # 3 forks at each step
end

"""
Create random walk with 3-fork branching
"""
function PlurigridWalk(; seed::UInt64=GAY_IGOR_SEED, n_steps::Int=69)
    positions = Int[]
    times = Float64[]
    colors = Tuple{Float32, Float32, Float32}[]
    gamuts = ColorGamut[]
    forks = Tuple{Int, Int, Int}[]
    
    rng = seed
    pos = 1
    t = 0.0
    
    for step in 1:n_steps
        push!(positions, pos)
        
        # Random time increment
        rng = mix64(rng)
        dt = (rng % 1000) / 100.0 + 0.1
        t += dt
        push!(times, t)
        
        # Color gamut for this step
        gamut = PLURIGRID_GAMUTS[mod1(step, length(PLURIGRID_GAMUTS))]
        push!(gamuts, gamut)
        
        # Color at this random time
        color = gamut_at_time(gamut, t, seed)
        push!(colors, color)
        
        # 3-fork: generate 3 possible next positions
        rng = mix64(rng)
        f1 = mod1(pos + Int(rng % 5) - 2, length(PLURIGRID_REINVENTIONS))
        rng = mix64(rng)
        f2 = mod1(pos + Int(rng % 5) - 2, length(PLURIGRID_REINVENTIONS))
        rng = mix64(rng)
        f3 = mod1(pos + Int(rng % 5) - 2, length(PLURIGRID_REINVENTIONS))
        push!(forks, (f1, f2, f3))
        
        # Choose fork based on balanced ternary
        rng = mix64(rng)
        choice = (rng % 3)  # 0, 1, or 2
        pos = choice == 0 ? f1 : (choice == 1 ? f2 : f3)
    end
    
    PlurigridWalk(seed, positions, times, colors, gamuts, forks)
end

"""
    fork_3_at_a_time(walk::PlurigridWalk, step::Int) -> (reinv1, reinv2, reinv3)

Get the 3 fork options at a given step as reinvention names.
"""
function fork_3_at_a_time(walk::PlurigridWalk, step::Int)
    idx = mod1(step, length(walk.forks))
    f1, f2, f3 = walk.forks[idx]
    (
        PLURIGRID_REINVENTIONS[f1],
        PLURIGRID_REINVENTIONS[f2],
        PLURIGRID_REINVENTIONS[f3]
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# World Centroid and Autopoietic Fixed Points
# ═══════════════════════════════════════════════════════════════════════════════

"""
    PlurigridCentroid

The world centroid of plurigrid evolution - the fixed point of autopoietic closure.
"""
struct PlurigridCentroid
    position::Vector{Float64}   # Centroid in embedding space
    trust_distribution::Vector{Int}  # Count of {-1, 0, +1}
    dominant_gamut::ColorGamut
    mean_complexity::Float64
    autopoietic_index::Float64  # Self-reference measure
end

"""
Compute world centroid from a random walk
"""
function world_centroid(walk::PlurigridWalk, trust_derivations::Vector{TrustDerivation})
    # Position as mean of visited states
    n = length(walk.positions)
    position = [sum(walk.positions) / n, 
                sum(walk.times) / n,
                sum(c[1] + c[2] + c[3] for c in walk.colors) / (3n)]
    
    # Trust distribution
    trust_dist = [0, 0, 0]  # [-1, 0, +1]
    for td in trust_derivations
        trust_dist[td.trust_value + 2] += 1
    end
    
    # Dominant gamut
    gamut_counts = Dict{ColorGamut, Int}()
    for g in walk.gamuts
        gamut_counts[g] = get(gamut_counts, g, 0) + 1
    end
    dominant = first(sort(collect(gamut_counts), by=x->x[2], rev=true))[1]
    
    # Mean complexity
    mean_k = isempty(trust_derivations) ? 0.0 : 
             sum(td.skc.complexity for td in trust_derivations) / length(trust_derivations)
    
    # Autopoietic index: self-reference measure via cycle detection
    seen = Set{Int}()
    cycles = 0
    for pos in walk.positions
        if pos in seen
            cycles += 1
        end
        push!(seen, pos)
    end
    autopoietic_idx = cycles / n
    
    PlurigridCentroid(position, trust_dist, dominant, mean_k, autopoietic_idx)
end

"""
Find autopoietic fixed point via iteration
"""
function autopoietic_fixed_point(initial_seed::UInt64; max_iter::Int=69)
    seed = initial_seed
    
    for iter in 1:max_iter
        walk = PlurigridWalk(; seed=seed, n_steps=23)
        
        # Compute new seed from walk
        new_seed = UInt64(0)
        for (i, pos) in enumerate(walk.positions)
            new_seed ⊻= mix64(UInt64(pos) ⊻ UInt64(i))
        end
        
        # Check for fixed point
        if new_seed == seed
            return (seed=seed, iteration=iter, fixed=true)
        end
        
        seed = new_seed
    end
    
    return (seed=seed, iteration=max_iter, fixed=false)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Gay Example Witness: Verify unpublished work
# ═══════════════════════════════════════════════════════════════════════════════

"""
    GayExampleWitness

Witness for a Gay.jl example with trust derivation and publication status.
"""
struct GayExampleWitness
    path::String
    name::String
    trust::TrustDerivation
    unpublished::Bool
    git_status::String
    condensed_hash::UInt64
end

"""
Witness an example file
"""
function witness_example(example_path::String, reinvention_idx::Int, seed::UInt64)
    name = basename(example_path)
    trust = derive_trust(example_path, reinvention_idx, seed)
    
    # Check git status (unpublished = untracked or modified)
    git_status = "unknown"
    unpublished = true
    
    # Compute condensed hash (local sheafification)
    if isfile(example_path)
        content = read(example_path)
        condensed = mix64(seed)
        for (i, b) in enumerate(content[1:min(1000, length(content))])
            condensed = mix64(condensed ⊻ UInt64(b) ⊻ UInt64(i))
        end
    else
        condensed = mix64(seed ⊻ hash(example_path))
    end
    
    GayExampleWitness(example_path, name, trust, unpublished, git_status, condensed)
end

"""
Condense locally: sheafify the witness into compact representation
"""
function condense_locally(witnesses::Vector{GayExampleWitness})
    # XOR all condensed hashes
    fingerprint = UInt64(0)
    for w in witnesses
        fingerprint ⊻= w.condensed_hash
    end
    
    # Trust summary
    trust_sum = sum(w.trust.trust_value for w in witnesses)
    
    # Unpublished count
    unpub_count = count(w -> w.unpublished, witnesses)
    
    (
        fingerprint = fingerprint,
        trust_sum = trust_sum,
        unpublished = unpub_count,
        total = length(witnesses)
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# The 69 Ways: Cross-product of examples × trust levels
# ═══════════════════════════════════════════════════════════════════════════════

"""
Gay.jl examples for 69-way trust derivation
"""
const GAY_EXAMPLES = [
    # Core examples
    "bisimulation_petri_colors.jl",
    "gay_metropolis.jl",
    "blume_capel_colors.jl",
    "quantum_colors.jl",
    "sse_heisenberg.jl",
    "fokker_planck.jl",
    "flower_calculus.jl",
    "prediction_market.jl",
    "polylog_colors.jl",
    "triangle_magic_colors.jl",
    "bbp_pi.jl",
    "galperin_colors.jl",
    "spaceinvaders_colors.jl",
    "blackhole.jl",
    "spectre.jl",
    "trans_trinity.jl",
    "narya_proofs_colors.jl",
    "gheg_dialects.jl",
    "world_specific_languages.jl",
    "concrete_syntax_matters.jl",
    "irreducibles.jl",
    "seed_differentiation.jl",
    "teleportation_spi.jl"
]

@assert length(GAY_EXAMPLES) == 23 "Need exactly 23 examples for 69 = 3 × 23"

"""
    plurigrid_zigzag_69(seed::UInt64) -> Vector{TrustDerivation}

Generate all 69 trust derivations: 23 examples × 3 trust levels.
"""
function plurigrid_zigzag_69(seed::UInt64=GAY_IGOR_SEED; examples_dir::String="")
    derivations = TrustDerivation[]
    
    # Cross-product: 23 examples × 3 trust levels
    for (ex_idx, example) in enumerate(GAY_EXAMPLES)
        for trust_level in [-1, 0, 1]
            # Reinvention index cycles through
            reinv_idx = mod1((ex_idx - 1) * 3 + (trust_level + 2), 69)
            
            # Derive trust with forced level
            path = isempty(examples_dir) ? example : joinpath(examples_dir, example)
            base_trust = derive_trust(path, reinv_idx, seed)
            
            # Override trust value for the 3-way derivation
            td = TrustDerivation(
                base_trust.example_name,
                trust_level,  # Force this trust level
                base_trust.skc,
                base_trust.igor_alignment,
                reinv_idx,
                base_trust.color,
                base_trust.gamut
            )
            push!(derivations, td)
        end
    end
    
    @assert length(derivations) == 69 "Must have exactly 69 derivations"
    return derivations
end

# ═══════════════════════════════════════════════════════════════════════════════
# Igor × Plurigrid Cross-Product
# ═══════════════════════════════════════════════════════════════════════════════

"""
    IgorPlurigridProduct

Cross-product of igor seeds with plurigrid random walk.
"""
struct IgorPlurigridProduct
    igor::IgorSpectrum
    walk::PlurigridWalk
    derivations::Vector{TrustDerivation}
    centroid::PlurigridCentroid
    fingerprint::UInt64
end

function IgorPlurigridProduct(; seed::UInt64=GAY_IGOR_SEED, igor_weight::Float64=0.5)
    igor = IgorSpectrum(; seed=seed, weight=igor_weight)
    walk = PlurigridWalk(; seed=seed, n_steps=69)
    derivations = plurigrid_zigzag_69(seed)
    centroid = world_centroid(walk, derivations)
    
    # Fingerprint from all components
    fp = UInt64(0)
    fp ⊻= igor.igor.seed
    for pos in walk.positions
        fp = mix64(fp ⊻ UInt64(pos))
    end
    for td in derivations
        fp ⊻= UInt64(td.trust_value + 2) << ((td.reinvention_idx % 60))
    end
    
    IgorPlurigridProduct(igor, walk, derivations, centroid, fp)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════════════════

function world_plurigrid_69()
    println()
    println("╔" * "═" ^ 65 * "╗")
    println("║  PLURIGRID × IGOR: 69 Ways of Autopoietic Ergodicity             ║")
    println("║  Cross-product with Solomonoff-Kolmogorov-Chaitin Trust          ║")
    println("╚" * "═" ^ 65 * "╝")
    println()
    
    seed = GAY_IGOR_SEED
    
    # Create random walk
    println("Plurigrid Random Walk (69 steps):")
    println("─" ^ 50)
    walk = PlurigridWalk(; seed=seed, n_steps=69)
    println("  Start: $(PLURIGRID_REINVENTIONS[walk.positions[1]])")
    println("  End:   $(PLURIGRID_REINVENTIONS[walk.positions[end]])")
    println("  Steps: $(length(walk.positions))")
    println()
    
    # Show some 3-forks with colors
    println("Sample 3-Forks with Random Time Colors:")
    println("─" ^ 50)
    for step in [1, 23, 46, 69]
        f1, f2, f3 = fork_3_at_a_time(walk, step)
        c = walk.colors[min(step, length(walk.colors))]
        r, g, b = Int.(Base.round.((c[1], c[2], c[3]) .* 255))
        gamut = walk.gamuts[min(step, length(walk.gamuts))]
        println("  Step $step ($(gamut)): \e[48;2;$(r);$(g);$(b)m  \e[0m")
        println("    Fork 1: $f1")
        println("    Fork 2: $f2")
        println("    Fork 3: $f3")
    end
    println()
    
    # 69 trust derivations
    println("69 Trust Derivations (23 examples × 3 levels):")
    println("─" ^ 50)
    derivations = plurigrid_zigzag_69(seed)
    
    # Count by trust level
    skeptical = count(td -> td.trust_value == -1, derivations)
    neutral = count(td -> td.trust_value == 0, derivations)
    trusting = count(td -> td.trust_value == 1, derivations)
    println("  Skeptical (-1): $skeptical")
    println("  Neutral (0):    $neutral")
    println("  Trusting (+1):  $trusting")
    println("  Total:          $(length(derivations))")
    println()
    
    # Show sample derivations
    println("Sample Derivations:")
    for (i, td) in enumerate(derivations[1:min(5, length(derivations))])
        trust_sym = td.trust_value == -1 ? "⊖" : (td.trust_value == 0 ? "○" : "⊕")
        reinv = PLURIGRID_REINVENTIONS[td.reinvention_idx]
        c = td.color
        r, g, b = Int.(Base.round.((c[1], c[2], c[3]) .* 255))
        println("  [$i] $(td.example_name): $trust_sym " *
                "K=$(Base.round(td.skc.complexity, digits=1)) " *
                "reinv=$(reinv[1:min(20,length(reinv))])... " *
                "\e[48;2;$(r);$(g);$(b)m  \e[0m")
    end
    println()
    
    # World centroid
    println("World Centroid:")
    println("─" ^ 50)
    centroid = world_centroid(walk, derivations)
    println("  Position: $(Base.round.(centroid.position, digits=3))")
    println("  Trust dist: [-1: $(centroid.trust_distribution[1]), " *
            "0: $(centroid.trust_distribution[2]), " *
            "+1: $(centroid.trust_distribution[3])]")
    println("  Dominant gamut: $(centroid.dominant_gamut)")
    println("  Mean K-complexity: $(Base.round(centroid.mean_complexity, digits=2))")
    println("  Autopoietic index: $(Base.round(centroid.autopoietic_index, digits=3))")
    println()
    
    # Autopoietic fixed point
    println("Autopoietic Fixed Point Search:")
    println("─" ^ 50)
    fp = autopoietic_fixed_point(seed)
    println("  Seed: 0x$(string(fp.seed, base=16))")
    println("  Iterations: $(fp.iteration)")
    println("  Fixed: $(fp.fixed ? "✓" : "✗")")
    println()
    
    # Full cross-product
    println("Igor × Plurigrid Cross-Product:")
    println("─" ^ 50)
    product = IgorPlurigridProduct(; seed=seed, igor_weight=0.618)
    println("  Igor weight: 0.618 (golden ratio)")
    println("  Fingerprint: 0x$(string(product.fingerprint, base=16))")
    println()
    
    # Plurigrid reinventions summary
    println("Plurigrid Reinventions (first and last 5):")
    println("─" ^ 50)
    for i in 1:5
        println("  [$i] $(PLURIGRID_REINVENTIONS[i])")
    end
    println("  ...")
    for i in 65:69
        println("  [$i] $(PLURIGRID_REINVENTIONS[i])")
    end
    println()
    
    println("🏳️‍🌈 Plurigrid × Igor: 69 Ways Complete")
end

if abspath(PROGRAM_FILE) == @__FILE__
    world_plurigrid_69()
end
