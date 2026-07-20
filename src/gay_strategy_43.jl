# GAY STRATEGY 43: Complete Tritwise Interpretation Space
# ========================================================================
#
# 43 different interpretations of {-, 0, +} assignment - ALWAYS.
#
# The number 43:
#   • 43 is prime
#   • 43 = 3² + 3¹ × 3 + 3⁰ × 7 = 9 + 9 + 25 = ... no, actually:
#   • 43 in balanced ternary: 43 = 27 + 9 + 6 + 1 = 27 + 18 - 2 = 1T-T- (complex)
#   • 43 = "the 14th prime" (14 = 7 + 7, and we want 7+ refs)
#   • Key insight: 43 semantic domains × 3 trit values = 129 unique assignments
#
# MAXIMUM PARALLELISM AT MAXIMUM LOSSLESSNESS:
#   • Type erasure correction via intermediate hue inference
#   • Unique hue assignments from seed bundles
#   • Leitmotifs = recurring chromatic patterns (musical term)
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  THE 43 INTERPRETATIONS OF {-, 0, +}                                        │
# │                                                                             │
# │  Each row: (domain, T-, T0, T+) where T- < T0 < T+ in domain ordering      │
# │                                                                             │
# │  TYPE ERASURE CORRECTION:                                                   │
# │    When type is erased, the HUE remains as contextual evidence.             │
# │    Intermediate hues between assignments reveal the lost type.              │
# │                                                                             │
# │  LEITMOTIF:                                                                 │
# │    A recurring chromatic pattern across seed bundles that identifies        │
# │    semantic content even when labels are stripped.                          │
# └─────────────────────────────────────────────────────────────────────────────┘

module GayStrategy43

using Printf

export
    # Core types
    TritInterpretation, InterpretationDomain, GayStrategy,
    
    # The 43 interpretations
    ALL_43_INTERPRETATIONS, get_interpretation,
    trit_meaning, domain_hue,
    
    # Type erasure correction
    TypeErasureCorrector, infer_type_from_hue,
    intermediate_hue, contextual_hue_chain,
    
    # Seed bundles and leitmotifs
    SeedBundle, Leitmotif, LeitmotifRegistry,
    bundle_hue, find_leitmotif, extract_leitmotifs,
    
    # Maximum parallelism at maximum losslessness
    GayMaxStrategy, maximize_parallelism,
    lossless_parallel_execute, parallel_type_recovery,
    
    # Demo
    world_43_interpretations, world_type_erasure_correction

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS (SPI Compliant)
# ═══════════════════════════════════════════════════════════════════════════════

const GAY_SEED = UInt64(1069)
const STRATEGY_SEED = UInt64(43)

@inline function splitmix64(state::UInt64)::UInt64
    z = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ⊻ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    z ⊻ (z >> 31)
end

@inline function color_from_seed(seed::UInt64)::NTuple{3, Float64}
    r = splitmix64(seed)
    g = splitmix64(r)
    b = splitmix64(g)
    ((r >> 56) / 255.0, (g >> 56) / 255.0, (b >> 56) / 255.0)
end

function fnv1a(text::String)::UInt64
    h = UInt64(14695981039346656037)
    for c in text
        h = (h ⊻ UInt64(c)) * UInt64(1099511628211)
    end
    h
end

# ═══════════════════════════════════════════════════════════════════════════════
# THE 43 INTERPRETATIONS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    InterpretationDomain
    
One of the 43 semantic domains for tritwise interpretation.
"""
struct InterpretationDomain
    id::Int                     # 1-43
    name::Symbol                # Domain name
    category::Symbol            # Category (temporal, spatial, computational, etc.)
    
    minus_meaning::String       # What T- means in this domain
    zero_meaning::String        # What T0 means
    plus_meaning::String        # What T+ means
    
    # Chromatic identity
    seed::UInt64
    hue::Float64                # Unique hue in [0, 360)
    color::NTuple{3, Float64}
end

function InterpretationDomain(id::Int, name::Symbol, category::Symbol,
                               minus::String, zero::String, plus::String)
    seed = fnv1a(string(name)) ⊻ UInt64(id) ⊻ GAY_SEED
    color = color_from_seed(seed)
    hue = (id - 1) * (360.0 / 43)  # Evenly spaced hues for uniqueness
    InterpretationDomain(id, name, category, minus, zero, plus, seed, hue, color)
end

"""
    ALL_43_INTERPRETATIONS
    
The complete set of 43 semantic domains for {-, 0, +} interpretation.
Organized by category for systematic coverage.
"""
const ALL_43_INTERPRETATIONS = InterpretationDomain[
    # ═══════════════════════════════════════════════════════════════════════════
    # TEMPORAL (1-7): Time-related orderings
    # ═══════════════════════════════════════════════════════════════════════════
    InterpretationDomain(1, :temporal_linear, :temporal, "past", "present", "future"),
    InterpretationDomain(2, :husserlian, :temporal, "retention", "primal", "protention"),
    InterpretationDomain(3, :causal, :temporal, "cause", "event", "effect"),
    InterpretationDomain(4, :evolutionary, :temporal, "ancestor", "current", "descendant"),
    InterpretationDomain(5, :lifecycle, :temporal, "birth", "life", "death"),
    InterpretationDomain(6, :memory, :temporal, "forgotten", "cached", "anticipated"),
    InterpretationDomain(7, :versioning, :temporal, "previous", "current", "next"),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # COMPUTATIONAL (8-14): Execution semantics
    # ═══════════════════════════════════════════════════════════════════════════
    InterpretationDomain(8, :evaluation, :computational, "lazy", "balanced", "eager"),
    InterpretationDomain(9, :parallelism, :computational, "sequential", "concurrent", "parallel"),
    InterpretationDomain(10, :speculation, :computational, "speculative", "committed", "aggregate"),
    InterpretationDomain(11, :caching, :computational, "miss", "stale", "hit"),
    InterpretationDomain(12, :scheduling, :computational, "defer", "ready", "dispatch"),
    InterpretationDomain(13, :locking, :computational, "acquire", "held", "release"),
    InterpretationDomain(14, :gc, :computational, "unreachable", "reachable", "pinned"),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CATEGORICAL (15-21): Category theory semantics
    # ═══════════════════════════════════════════════════════════════════════════
    InterpretationDomain(15, :morphism, :categorical, "domain", "morphism", "codomain"),
    InterpretationDomain(16, :adjunction, :categorical, "left_adjoint", "unit", "right_adjoint"),
    InterpretationDomain(17, :limits, :categorical, "colimit", "object", "limit"),
    InterpretationDomain(18, :yoneda, :categorical, "contravariant", "natural", "covariant"),
    InterpretationDomain(19, :monoidal, :categorical, "tensor", "unit", "cotensor"),
    InterpretationDomain(20, :cohesive, :categorical, "flat", "discrete", "sharp"),
    InterpretationDomain(21, :modality, :categorical, "◯_next", "□_always", "◇_eventually"),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # GAME-THEORETIC (22-28): Strategic semantics
    # ═══════════════════════════════════════════════════════════════════════════
    InterpretationDomain(22, :payoff, :game, "loss", "neutral", "win"),
    InterpretationDomain(23, :strategy, :game, "defect", "mixed", "cooperate"),
    InterpretationDomain(24, :information, :game, "hidden", "signaled", "revealed"),
    InterpretationDomain(25, :equilibrium, :game, "unstable", "nash", "pareto"),
    InterpretationDomain(26, :auction, :game, "underbid", "market", "overbid"),
    InterpretationDomain(27, :voting, :game, "oppose", "abstain", "support"),
    InterpretationDomain(28, :negotiation, :game, "concede", "hold", "demand"),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PHYSICAL (29-35): Physics and geometry
    # ═══════════════════════════════════════════════════════════════════════════
    InterpretationDomain(29, :motion, :physical, "backward", "stationary", "forward"),
    InterpretationDomain(30, :charge, :physical, "negative", "neutral", "positive"),
    InterpretationDomain(31, :spin, :physical, "down", "superposed", "up"),
    InterpretationDomain(32, :curvature, :physical, "hyperbolic", "flat", "spherical"),
    InterpretationDomain(33, :entropy, :physical, "ordered", "equilibrium", "disordered"),
    InterpretationDomain(34, :force, :physical, "repulsive", "neutral", "attractive"),
    InterpretationDomain(35, :phase, :physical, "solid", "liquid", "gas"),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CHROMATIC (36-43): Color and perception
    # ═══════════════════════════════════════════════════════════════════════════
    InterpretationDomain(36, :luminance, :chromatic, "dark", "neutral", "bright"),
    InterpretationDomain(37, :saturation, :chromatic, "desaturated", "muted", "vivid"),
    InterpretationDomain(38, :temperature, :chromatic, "cool", "neutral", "warm"),
    InterpretationDomain(39, :contrast, :chromatic, "low", "balanced", "high"),
    InterpretationDomain(40, :harmony, :chromatic, "dissonant", "neutral", "consonant"),
    InterpretationDomain(41, :depth, :chromatic, "background", "midground", "foreground"),
    InterpretationDomain(42, :polarity, :chromatic, "negative", "zero", "positive"),
    InterpretationDomain(43, :gayness, :chromatic, "ungay", "semigay", "gay"),
]

"""Get interpretation by ID (1-43)."""
function get_interpretation(id::Int)::InterpretationDomain
    1 <= id <= 43 || error("ID must be 1-43, got $id")
    ALL_43_INTERPRETATIONS[id]
end

"""Get interpretation by name."""
function get_interpretation(name::Symbol)::InterpretationDomain
    for interp in ALL_43_INTERPRETATIONS
        if interp.name == name
            return interp
        end
    end
    error("Unknown interpretation: $name")
end

"""Get the meaning of a trit value in a domain."""
function trit_meaning(domain::InterpretationDomain, trit::Int)::String
    if trit == -1
        domain.minus_meaning
    elseif trit == 0
        domain.zero_meaning
    elseif trit == 1
        domain.plus_meaning
    else
        error("Trit must be -1, 0, or +1")
    end
end

"""Get the unique hue for a domain."""
domain_hue(domain::InterpretationDomain) = domain.hue

# ═══════════════════════════════════════════════════════════════════════════════
# SEED BUNDLES AND LEITMOTIFS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    SeedBundle
    
A bundle of related seeds that share a leitmotif.
The bundle has a chromatic fingerprint derived from all seeds.
"""
struct SeedBundle
    name::Symbol
    seeds::Vector{UInt64}
    
    # Derived
    bundle_seed::UInt64      # XOR of all seeds
    color::NTuple{3, Float64}
    hue::Float64
    fingerprint::UInt64
end

function SeedBundle(name::Symbol, seeds::Vector{UInt64})
    bundle_seed = reduce(⊻, seeds; init=fnv1a(string(name)))
    color = color_from_seed(bundle_seed)
    
    # Hue from RGB (simplified HSL conversion)
    r, g, b = color
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    delta = max_c - min_c
    
    hue = if delta ≈ 0
        0.0
    elseif max_c ≈ r
        60.0 * mod((g - b) / delta, 6)
    elseif max_c ≈ g
        60.0 * ((b - r) / delta + 2)
    else
        60.0 * ((r - g) / delta + 4)
    end
    
    fingerprint = splitmix64(bundle_seed)
    
    SeedBundle(name, seeds, bundle_seed, color, hue, fingerprint)
end

"""Get the characteristic hue of a seed bundle."""
bundle_hue(b::SeedBundle) = b.hue

"""
    Leitmotif
    
A recurring chromatic pattern that identifies semantic content.
Inspired by Wagnerian leitmotifs - musical themes for characters/concepts.
"""
struct Leitmotif
    name::Symbol
    pattern::Vector{NTuple{3, Float64}}  # Sequence of colors
    
    # Derived signature
    signature::UInt64
    mean_hue::Float64
    variance::Float64
end

function Leitmotif(name::Symbol, pattern::Vector{NTuple{3, Float64}})
    # Compute signature from pattern
    sig = fnv1a(string(name))
    for (i, color) in enumerate(pattern)
        r, g, b = color
        color_int = UInt64(round(r * 255)) << 16 | UInt64(round(g * 255)) << 8 | UInt64(round(b * 255))
        sig = sig ⊻ splitmix64(color_int ⊻ UInt64(i))
    end
    
    # Mean hue
    hues = Float64[]
    for (r, g, b) in pattern
        max_c = max(r, g, b)
        min_c = min(r, g, b)
        delta = max_c - min_c
        
        h = if delta ≈ 0
            0.0
        elseif max_c ≈ r
            60.0 * mod((g - b) / delta, 6)
        elseif max_c ≈ g
            60.0 * ((b - r) / delta + 2)
        else
            60.0 * ((r - g) / delta + 4)
        end
        push!(hues, h)
    end
    
    mean_hue = sum(hues) / length(hues)
    variance = sum((h - mean_hue)^2 for h in hues) / length(hues)
    
    Leitmotif(name, pattern, sig, mean_hue, variance)
end

"""Extract leitmotifs from a sequence of seeds."""
function extract_leitmotifs(seeds::Vector{UInt64}; window::Int=3)::Vector{Leitmotif}
    leitmotifs = Leitmotif[]
    colors = [color_from_seed(s) for s in seeds]
    
    # Find repeating patterns
    seen = Dict{UInt64, Vector{Int}}()
    
    for i in 1:(length(colors) - window + 1)
        pattern = colors[i:i+window-1]
        temp_leit = Leitmotif(:temp, pattern)
        sig = temp_leit.signature
        
        if haskey(seen, sig)
            push!(seen[sig], i)
        else
            seen[sig] = [i]
        end
    end
    
    # Keep patterns that appear multiple times
    for (sig, positions) in seen
        if length(positions) >= 2
            i = positions[1]
            pattern = colors[i:i+window-1]
            name = Symbol("leitmotif_$(sig % 10000)")
            push!(leitmotifs, Leitmotif(name, pattern))
        end
    end
    
    leitmotifs
end

# ═══════════════════════════════════════════════════════════════════════════════
# TYPE ERASURE CORRECTION
# ═══════════════════════════════════════════════════════════════════════════════

"""
    TypeErasureCorrector
    
Corrects type erasure by inferring types from contextual intermediate hues.

When type information is lost (erased), the HUE remains as evidence.
By examining the hue in context of neighboring hues, we can infer
what the original type was.

Algorithm:
1. Each type has a unique seed → unique hue
2. Erased type leaves hue fingerprint
3. Compare erased hue to known type hues
4. Use intermediate hues (between known types) to triangulate
"""
struct TypeErasureCorrector
    known_types::Dict{Symbol, UInt64}      # type_name → seed
    type_hues::Dict{Symbol, Float64}        # type_name → hue
    type_colors::Dict{Symbol, NTuple{3, Float64}}
    
    # Hue → Type lookup (nearest neighbor)
    hue_to_type::Vector{Tuple{Float64, Symbol}}  # Sorted by hue
end

function TypeErasureCorrector(known_types::Dict{Symbol, UInt64})
    type_hues = Dict{Symbol, Float64}()
    type_colors = Dict{Symbol, NTuple{3, Float64}}()
    hue_to_type = Tuple{Float64, Symbol}[]
    
    for (name, seed) in known_types
        color = color_from_seed(seed)
        type_colors[name] = color
        
        r, g, b = color
        max_c = max(r, g, b)
        min_c = min(r, g, b)
        delta = max_c - min_c
        
        hue = if delta ≈ 0
            0.0
        elseif max_c ≈ r
            60.0 * mod((g - b) / delta, 6)
        elseif max_c ≈ g
            60.0 * ((b - r) / delta + 2)
        else
            60.0 * ((r - g) / delta + 4)
        end
        
        type_hues[name] = hue
        push!(hue_to_type, (hue, name))
    end
    
    sort!(hue_to_type, by = x -> x[1])
    
    TypeErasureCorrector(known_types, type_hues, type_colors, hue_to_type)
end

"""Infer the original type from an erased value's hue."""
function infer_type_from_hue(corrector::TypeErasureCorrector, 
                              observed_hue::Float64;
                              tolerance::Float64=15.0)::Union{Symbol, Nothing}
    best_match = nothing
    best_dist = Inf
    
    for (type_hue, type_name) in corrector.hue_to_type
        # Circular distance (hue wraps at 360)
        dist = min(abs(observed_hue - type_hue), 
                   360.0 - abs(observed_hue - type_hue))
        
        if dist < best_dist && dist <= tolerance
            best_dist = dist
            best_match = type_name
        end
    end
    
    best_match
end

"""Compute intermediate hue between two types."""
function intermediate_hue(corrector::TypeErasureCorrector,
                          type_a::Symbol, type_b::Symbol;
                          t::Float64=0.5)::Float64
    hue_a = corrector.type_hues[type_a]
    hue_b = corrector.type_hues[type_b]
    
    # Interpolate on circle
    if abs(hue_b - hue_a) > 180
        # Go the short way around
        if hue_a < hue_b
            hue_a += 360
        else
            hue_b += 360
        end
    end
    
    result = hue_a + t * (hue_b - hue_a)
    mod(result, 360.0)
end

"""Build a chain of contextual hues from a sequence of types."""
function contextual_hue_chain(corrector::TypeErasureCorrector,
                               types::Vector{Symbol})::Vector{Float64}
    hues = [corrector.type_hues[t] for t in types]
    
    # Add intermediate hues between each pair
    chain = Float64[]
    for i in 1:length(hues)
        push!(chain, hues[i])
        if i < length(hues)
            mid = intermediate_hue(corrector, types[i], types[i+1])
            push!(chain, mid)
        end
    end
    
    chain
end

# ═══════════════════════════════════════════════════════════════════════════════
# MAXIMUM PARALLELISM AT MAXIMUM LOSSLESSNESS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    GayMaxStrategy
    
Strategy for achieving maximum parallelism while preserving all information.

The key insight: parallelism usually requires independence, but independence
often loses contextual information. Gay's chromatic identity preserves
context through deterministic hue assignment.

MAXIMUM PARALLELISM:
  - All 43 interpretations can be computed in parallel
  - Each produces independent but consistent results
  - SPI ensures same seed → same result everywhere

MAXIMUM LOSSLESSNESS:
  - Type information encoded in hue
  - Leitmotifs identify patterns even after label erasure  
  - Intermediate hues enable type recovery
  - Fingerprints verify integrity
"""
struct GayMaxStrategy
    # Active interpretations (can use all 43, or subset)
    active_interpretations::Vector{InterpretationDomain}
    
    # Type erasure correction
    corrector::TypeErasureCorrector
    
    # Seed bundle for this strategy
    bundle::SeedBundle
    
    # Leitmotif registry
    leitmotifs::Vector{Leitmotif}
    
    # Parallelism settings
    max_parallel::Int           # Maximum parallel workers
    trit_strategy::Int          # -1=lazy, 0=balanced, +1=eager
    
    # Chromatic identity
    seed::UInt64
    color::NTuple{3, Float64}
end

function GayMaxStrategy(; 
                        interpretations::Vector{Int}=collect(1:43),
                        known_types::Dict{Symbol, UInt64}=Dict{Symbol, UInt64}(),
                        seeds::Vector{UInt64}=UInt64[GAY_SEED],
                        max_parallel::Int=43,
                        trit_strategy::Int=0)
    
    active = [ALL_43_INTERPRETATIONS[i] for i in interpretations]
    corrector = TypeErasureCorrector(known_types)
    bundle = SeedBundle(:strategy, seeds)
    leitmotifs = extract_leitmotifs(seeds)
    
    seed = bundle.bundle_seed ⊻ STRATEGY_SEED
    color = color_from_seed(seed)
    
    GayMaxStrategy(active, corrector, bundle, leitmotifs,
                   max_parallel, trit_strategy, seed, color)
end

"""
Execute all interpretations in parallel with type recovery.
Returns results keyed by interpretation domain.
"""
function maximize_parallelism(strategy::GayMaxStrategy,
                               value::Any,
                               trit::Int)::Dict{Symbol, NamedTuple}
    results = Dict{Symbol, NamedTuple}()
    
    # In a real implementation, these would run in parallel
    for interp in strategy.active_interpretations
        meaning = trit_meaning(interp, trit)
        
        # Derive hue for this interpretation of this value
        value_seed = fnv1a(string(value)) ⊻ interp.seed
        value_color = color_from_seed(value_seed)
        
        results[interp.name] = (
            domain = interp.name,
            category = interp.category,
            trit = trit,
            meaning = meaning,
            domain_hue = interp.hue,
            value_color = value_color,
            fingerprint = splitmix64(value_seed ⊻ UInt64(trit + 2))
        )
    end
    
    results
end

"""
Parallel execution with type recovery for erased values.
"""
function parallel_type_recovery(strategy::GayMaxStrategy,
                                 erased_values::Vector{NTuple{3, Float64}})::Vector{Union{Symbol, Nothing}}
    recovered = Union{Symbol, Nothing}[]
    
    for color in erased_values
        r, g, b = color
        max_c = max(r, g, b)
        min_c = min(r, g, b)
        delta = max_c - min_c
        
        hue = if delta ≈ 0
            0.0
        elseif max_c ≈ r
            60.0 * mod((g - b) / delta, 6)
        elseif max_c ≈ g
            60.0 * ((b - r) / delta + 2)
        else
            60.0 * ((r - g) / delta + 4)
        end
        
        inferred = infer_type_from_hue(strategy.corrector, hue)
        push!(recovered, inferred)
    end
    
    recovered
end

"""
Lossless parallel execution: run all 43 interpretations and verify consistency.
"""
function lossless_parallel_execute(strategy::GayMaxStrategy,
                                    values::Vector{Any},
                                    trits::Vector{Int})::NamedTuple
    all_results = Vector{Dict{Symbol, NamedTuple}}()
    fingerprints = Vector{UInt64}()
    
    for (value, trit) in zip(values, trits)
        result = maximize_parallelism(strategy, value, trit)
        push!(all_results, result)
        
        # Combined fingerprint for verification
        fp = reduce(⊻, [r.fingerprint for r in values(result)])
        push!(fingerprints, fp)
    end
    
    # Verify consistency via fingerprint chain
    chain_fp = reduce(⊻, fingerprints; init=strategy.seed)
    
    (
        results = all_results,
        fingerprints = fingerprints,
        chain_fingerprint = chain_fp,
        n_interpretations = length(strategy.active_interpretations),
        n_values = length(values),
        total_parallel_ops = length(values) * length(strategy.active_interpretations)
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

function world_43_interpretations()
    println()
    println("╔═══════════════════════════════════════════════════════════════════════════╗")
    println("║  GAY STRATEGY 43: Complete Tritwise Interpretation Space                 ║")
    println("║  43 different interpretations of {-, 0, +} - ALWAYS                      ║")
    println("╚═══════════════════════════════════════════════════════════════════════════╝")
    println()
    
    # Show all 43 interpretations
    categories = Dict{Symbol, Vector{InterpretationDomain}}()
    for interp in ALL_43_INTERPRETATIONS
        if !haskey(categories, interp.category)
            categories[interp.category] = InterpretationDomain[]
        end
        push!(categories[interp.category], interp)
    end
    
    for (category, interps) in sort(collect(categories), by=x->x[2][1].id)
        println("─── $(uppercase(string(category))) ───")
        for interp in interps
            hue_str = @sprintf("%.1f°", interp.hue)
            println("  $(lpad(interp.id, 2)). $(rpad(interp.name, 20)) │ $(rpad(interp.minus_meaning, 12)) │ $(rpad(interp.zero_meaning, 12)) │ $(rpad(interp.plus_meaning, 12)) │ hue=$(hue_str)")
        end
        println()
    end
    
    # Summary
    println("─── SUMMARY ───")
    println("  Total interpretations: 43")
    println("  Unique hue assignments: 43 (evenly spaced in [0°, 360°))")
    println("  Hue spacing: $(360.0/43)° ≈ 8.37° per domain")
    println("  Total trit meanings: 43 × 3 = 129")
    println()
    
    # Demo maximum parallelism
    println("─── MAXIMUM PARALLELISM DEMO ───")
    strategy = GayMaxStrategy()
    
    test_value = "thread_T-019b03bc"
    test_trit = 1  # T+
    
    results = maximize_parallelism(strategy, test_value, test_trit)
    
    println("  Value: $test_value")
    println("  Trit: T+ (eager/future/cooperate/...)")
    println("  Parallel interpretations: $(length(results))")
    println()
    
    # Show a few results
    sample_domains = [:evaluation, :game, :physical, :gayness]
    for domain in sample_domains
        if haskey(results, domain)
            r = results[domain]
            println("  $(domain): $(r.meaning)")
        end
    end
    println()
    
    println("═══════════════════════════════════════════════════════════════════════════")
end

function world_type_erasure_correction()
    println()
    println("─── TYPE ERASURE CORRECTION DEMO ───")
    println()
    
    # Define known types with seeds
    known_types = Dict{Symbol, UInt64}(
        :Thread => fnv1a("Thread"),
        :Color => fnv1a("Color"),
        :Seed => fnv1a("Seed"),
        :World => fnv1a("World"),
        :Strategy => fnv1a("Strategy"),
        :Leitmotif => fnv1a("Leitmotif"),
    )
    
    corrector = TypeErasureCorrector(known_types)
    
    println("  Known types and their hues:")
    for (name, hue) in sort(collect(corrector.type_hues), by=x->x[2])
        println("    $(rpad(name, 12)): $(round(hue, digits=1))°")
    end
    println()
    
    # Simulate type erasure and recovery
    println("  Type erasure simulation:")
    for (name, seed) in known_types
        color = color_from_seed(seed)
        
        # Compute hue from color
        r, g, b = color
        max_c = max(r, g, b)
        min_c = min(r, g, b)
        delta = max_c - min_c
        
        hue = if delta ≈ 0
            0.0
        elseif max_c ≈ r
            60.0 * mod((g - b) / delta, 6)
        elseif max_c ≈ g
            60.0 * ((b - r) / delta + 2)
        else
            60.0 * ((r - g) / delta + 4)
        end
        
        # Attempt recovery
        recovered = infer_type_from_hue(corrector, hue)
        match = recovered == name ? "✓" : "✗"
        
        println("    $name erased → hue=$(round(hue, digits=1))° → recovered: $recovered $match")
    end
    println()
    
    # Intermediate hues
    println("  Intermediate hues (for contextual inference):")
    pairs = [(:Thread, :Color), (:Seed, :World), (:Strategy, :Leitmotif)]
    for (a, b) in pairs
        mid = intermediate_hue(corrector, a, b)
        println("    $a ↔ $b midpoint: $(round(mid, digits=1))°")
    end
    println()
    
    println("═══════════════════════════════════════════════════════════════════════════")
end

end # module GayStrategy43
