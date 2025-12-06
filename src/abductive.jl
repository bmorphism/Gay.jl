# Abductive Testing for World Teleportation
# ==========================================
# Given observed world colors, reason backwards to infer:
# - Source invader ID and seed
# - Derangement permutation applied
# - Tropical blend parameter t
#
# Inspired by MultipleInterfaces.jl (Bieganek, JuliaCon 2025):
# - Interface intersections for complex test predicates
# - Required vs provided method distinction for test design

using Colors: RGB

# ═══════════════════════════════════════════════════════════════════════════════
# Abductive Hypothesis: A potential explanation for an observed world color
# ═══════════════════════════════════════════════════════════════════════════════

"""
    AbductiveHypothesis

A candidate explanation for how an invader teleported to produce an observed world color.

Abduction reasons backwards:
  observed_color → hypothesis(id, seed, derangement, t)

The hypothesis can be tested by forward simulation.
"""
struct AbductiveHypothesis
    id::UInt64
    seed::UInt64
    derangement_idx::Int      # 1 or 2 for RGB cyclic permutations
    tropical_t::Float64       # Blend parameter ∈ [0,1]
    confidence::Float64       # How well this explains the observation
    source_color::RGB{Float64}
    predicted_world::RGB{Float64}
end

"""
    color_distance(c1::RGB, c2::RGB) -> Float64

Euclidean distance in RGB space (simple, fast).
"""
color_distance(c1::RGB, c2::RGB) = sqrt(
    (c1.r - c2.r)^2 + (c1.g - c2.g)^2 + (c1.b - c2.b)^2
)

"""
    color_distance_perceptual(c1::RGB, c2::RGB) -> Float64

Weighted distance approximating human perception (green-sensitive).
"""
function color_distance_perceptual(c1::RGB, c2::RGB)
    Δr = c1.r - c2.r
    Δg = c1.g - c2.g
    Δb = c1.b - c2.b
    r_mean = (c1.r + c2.r) / 2
    return sqrt((2 + r_mean) * Δr^2 + 4 * Δg^2 + (3 - r_mean) * Δb^2)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Forward Simulation: Replay teleportation to verify hypothesis
# ═══════════════════════════════════════════════════════════════════════════════

const DERANGEMENTS_3 = (
    [2, 3, 1],  # R→G, G→B, B→R (cyclic left)
    [3, 1, 2],  # R→B, G→R, B→G (cyclic right)
)

"""
    apply_derangement(c::RGB, idx::Int) -> RGB

Apply the idx-th derangement (1 or 2) to permute RGB channels.
Derangements have no fixed points: no channel maps to itself.
"""
function apply_derangement(c::RGB{T}, idx::Int) where T
    perm = DERANGEMENTS_3[idx]
    channels = (c.r, c.g, c.b)
    return RGB{T}(channels[perm[1]], channels[perm[2]], channels[perm[3]])
end

"""
    TropicalFloat

A number in the tropical semiring (ℝ ∪ {∞}, min, +).
Used for non-linear color blending with hard edges.
"""
struct TropicalFloat
    val::Float64
end

Base.:+(a::TropicalFloat, b::TropicalFloat) = TropicalFloat(min(a.val, b.val))
Base.:*(a::TropicalFloat, b::TropicalFloat) = TropicalFloat(a.val + b.val)
Base.zero(::Type{TropicalFloat}) = TropicalFloat(Inf)
Base.one(::Type{TropicalFloat}) = TropicalFloat(0.0)

"""
    tropical_blend(c1::RGB, c2::RGB, t::Float64) -> RGB

Blend two colors using tropical (min-plus) interpolation.
Creates hard edges and plateaus unlike linear interpolation.
"""
function tropical_blend(c1::RGB, c2::RGB, t::Float64)
    tr1 = TropicalFloat(-log(max(c1.r, 1e-10)))
    tr2 = TropicalFloat(-log(max(c2.r, 1e-10)))
    tg1 = TropicalFloat(-log(max(c1.g, 1e-10)))
    tg2 = TropicalFloat(-log(max(c2.g, 1e-10)))
    tb1 = TropicalFloat(-log(max(c1.b, 1e-10)))
    tb2 = TropicalFloat(-log(max(c2.b, 1e-10)))
    
    blend_r = min(tr1.val + t, tr2.val + (1-t))
    blend_g = min(tg1.val + t, tg2.val + (1-t))
    blend_b = min(tb1.val + t, tb2.val + (1-t))
    
    return RGB(exp(-blend_r), exp(-blend_g), exp(-blend_b))
end

"""
    simulate_teleportation(id, seed) -> NamedTuple

Forward-simulate the full teleportation path for an invader.
Returns all intermediate states for verification.
"""
function simulate_teleportation(id::Integer, seed::UInt64)
    id < 1 && throw(ArgumentError("id must be positive, got $id"))
    r, g, b = hash_color(UInt64(id), seed)
    source = RGB(Float64(r), Float64(g), Float64(b))
    
    derange_idx = Int(id % 2 + 1)
    deranged = apply_derangement(source, derange_idx)
    
    world_seed = splitmix64(seed ⊻ UInt64(id * 0x9e3779b97f4a7c15))
    wr, wg, wb = hash_color(UInt64(world_seed & 0xFFFFFF), seed)
    world_base = RGB(Float64(wr), Float64(wg), Float64(wb))
    
    t = Float64((id % 100)) / 100.0
    world = tropical_blend(deranged, world_base, t)
    
    spin = ((id ⊻ (id >> 16)) & 1 == 0) ? 1 : -1
    
    return (
        id = UInt64(id),
        seed = seed,
        source = source,
        derangement = derange_idx,
        deranged = deranged,
        world_base = world_base,
        tropical_t = t,
        world = world,
        spin = spin
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# Abductive Inference: Reason backwards from observed world color
# ═══════════════════════════════════════════════════════════════════════════════

"""
    abduce_invader(observed::RGB; seed=GAY_SEED, search_range=1:10000, top_k=5) -> Vector{AbductiveHypothesis}

Given an observed world color, find the most likely invader IDs that could have produced it.

This is abductive reasoning: inferring causes from effects.
Returns top_k hypotheses ranked by confidence (color similarity).
"""
function abduce_invader(
    observed::RGB;
    seed::UInt64 = GAY_SEED,
    search_range::AbstractRange = 1:10000,
    top_k::Int = 5,
    distance_fn::Function = color_distance
)
    hypotheses = AbductiveHypothesis[]
    
    for id in search_range
        id < 1 && continue  # Skip non-positive IDs
        sim = simulate_teleportation(id, seed)
        dist = distance_fn(observed, sim.world)
        confidence = 1.0 / (1.0 + dist)  # Higher distance = lower confidence
        
        push!(hypotheses, AbductiveHypothesis(
            sim.id,
            seed,
            sim.derangement,
            sim.tropical_t,
            confidence,
            sim.source,
            sim.world
        ))
    end
    
    sort!(hypotheses, by = h -> -h.confidence)
    return hypotheses[1:min(top_k, length(hypotheses))]
end

"""
    abduce_from_source(observed_source::RGB; seed=GAY_SEED, search_range=1:10000) -> Union{UInt64, Nothing}

Given an observed SOURCE color (before derangement), find the exact invader ID.
This is deterministic if the color exists in the search range.
"""
function abduce_from_source(
    observed::RGB;
    seed::UInt64 = GAY_SEED,
    search_range::AbstractRange = 1:10000,
    tolerance::Float64 = 1e-6
)
    for id in search_range
        id < 1 && continue
        r, g, b = hash_color(UInt64(id), seed)
        source = RGB(Float64(r), Float64(g), Float64(b))
        if color_distance(observed, source) < tolerance
            return UInt64(id)
        end
    end
    return nothing
end

"""
    invert_derangement(c::RGB, idx::Int) -> RGB

Invert a derangement to recover the original color.
For cyclic permutations, apply twice to get back.
"""
function invert_derangement(c::RGB{T}, idx::Int) where T
    # Cyclic derangements: applying twice inverts
    # [2,3,1] applied twice = [3,1,2] = inverse
    return apply_derangement(c, 3 - idx)  # 1→2, 2→1
end

# ═══════════════════════════════════════════════════════════════════════════════
# Test Predicates: Properties that must hold for valid teleportation
# ═══════════════════════════════════════════════════════════════════════════════

"""
    TeleportationProperty

Abstract type for testable properties of world teleportation.
Inspired by MultipleInterfaces.jl's interface pattern.
"""
abstract type TeleportationProperty end

struct SPIDeterminism <: TeleportationProperty end
struct DerangementBijectivity <: TeleportationProperty end
struct TropicalIdempotence <: TeleportationProperty end
struct SpinConsistency <: TeleportationProperty end

"""
    test_property(prop::TeleportationProperty, id, seed) -> Bool

Test whether a teleportation satisfies the given property.
"""
function test_property(::SPIDeterminism, id::Integer, seed::UInt64)
    sim1 = simulate_teleportation(id, seed)
    sim2 = simulate_teleportation(id, seed)
    return sim1.world == sim2.world
end

function test_property(::DerangementBijectivity, id::Integer, seed::UInt64)
    sim = simulate_teleportation(id, seed)
    # Apply derangement then its inverse should recover source
    recovered = invert_derangement(sim.deranged, sim.derangement)
    return color_distance(sim.source, recovered) < 1e-10
end

function test_property(::TropicalIdempotence, id::Integer, seed::UInt64)
    # Blending at t=0 and t=1 with same colors should give similar results
    sim = simulate_teleportation(id, seed)
    blend_0 = tropical_blend(sim.deranged, sim.deranged, 0.0)
    blend_1 = tropical_blend(sim.deranged, sim.deranged, 1.0)
    # Tropical geometry uses min-based operations, so results are close
    return color_distance(blend_0, blend_1) < 0.5
end

function test_property(::SpinConsistency, id::Integer, seed::UInt64)
    sim = simulate_teleportation(id, seed)
    expected_spin = ((id ⊻ (id >> 16)) & 1 == 0) ? 1 : -1
    return sim.spin == expected_spin
end

"""
    test_all_properties(id, seed) -> NamedTuple

Test all teleportation properties for an invader.
"""
function test_all_properties(id::Integer, seed::UInt64 = GAY_SEED)
    return (
        spi = test_property(SPIDeterminism(), id, seed),
        bijectivity = test_property(DerangementBijectivity(), id, seed),
        idempotence = test_property(TropicalIdempotence(), id, seed),
        spin = test_property(SpinConsistency(), id, seed)
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# REPL-in-the-Loop Testing: Interactive exploration commands
# ═══════════════════════════════════════════════════════════════════════════════

"""
    WorldNavigator

State for REPL-based world exploration.
Tracks current position in the invader-world space.
"""
mutable struct WorldNavigator
    current_id::UInt64
    seed::UInt64
    history::Vector{UInt64}
    hypotheses::Vector{AbductiveHypothesis}
end

const NAVIGATOR = Ref{WorldNavigator}()

function init_navigator(; seed::UInt64 = GAY_SEED)
    NAVIGATOR[] = WorldNavigator(1, seed, UInt64[], AbductiveHypothesis[])
end

function get_navigator()
    if !isassigned(NAVIGATOR)
        init_navigator()
    end
    return NAVIGATOR[]
end

"""
    teleport!(id::Integer)

Teleport to a specific invader's world. Records history for backtracking.
"""
function teleport!(id::Integer)
    nav = get_navigator()
    push!(nav.history, nav.current_id)
    nav.current_id = UInt64(id)
    return current_world()
end

"""
    back!()

Return to previous world in navigation history.
"""
function back!()
    nav = get_navigator()
    isempty(nav.history) && error("No history to go back to!")
    nav.current_id = pop!(nav.history)
    return current_world()
end

"""
    current_world()

Get the current invader's world simulation.
"""
function current_world()
    nav = get_navigator()
    return simulate_teleportation(nav.current_id, nav.seed)
end

"""
    abduce!(observed::RGB; kwargs...)

Run abductive inference from current position and store hypotheses.
"""
function abduce!(observed::RGB; kwargs...)
    nav = get_navigator()
    nav.hypotheses = abduce_invader(observed; seed=nav.seed, kwargs...)
    return nav.hypotheses
end

"""
    jump_hypothesis!(idx::Int)

Teleport to the world of the idx-th hypothesis.
"""
function jump_hypothesis!(idx::Int)
    nav = get_navigator()
    isempty(nav.hypotheses) && error("No hypotheses! Run abduce! first.")
    h = nav.hypotheses[idx]
    return teleport!(h.id)
end

"""
    explore_neighbors(; radius=10)

Explore invaders near the current position.
"""
function explore_neighbors(; radius::Int = 10)
    nav = get_navigator()
    id = nav.current_id
    range = max(1, id - radius):id + radius
    return [simulate_teleportation(i, nav.seed) for i in range]
end

# ═══════════════════════════════════════════════════════════════════════════════
# Abductive Test Suite: Property-based testing with abduction
# ═══════════════════════════════════════════════════════════════════════════════

"""
    AbductiveTestResult

Result of an abductive test run.
"""
struct AbductiveTestResult
    property::Symbol
    passed::Bool
    id::UInt64
    seed::UInt64
    details::String
end

"""
    run_abductive_tests(; n_samples=100, seed=GAY_SEED) -> Vector{AbductiveTestResult}

Run abductive test suite on random invader samples.
"""
function run_abductive_tests(; n_samples::Int = 100, seed::UInt64 = GAY_SEED)
    results = AbductiveTestResult[]
    
    for _ in 1:n_samples
        id = rand(1:1_000_000)
        props = test_all_properties(id, seed)
        
        for (name, passed) in pairs(props)
            push!(results, AbductiveTestResult(
                name,
                passed,
                UInt64(id),
                seed,
                passed ? "OK" : "FAILED"
            ))
        end
    end
    
    return results
end

"""
    abductive_roundtrip_test(id, seed) -> Bool

Test that we can abduce an invader from its own world color.
This is the core correctness test for the abductive framework.
"""
function abductive_roundtrip_test(id::Integer, seed::UInt64 = GAY_SEED)
    sim = simulate_teleportation(id, seed)
    hypotheses = abduce_invader(sim.world; seed=seed, search_range=(id-100):(id+100), top_k=1)
    
    if isempty(hypotheses)
        return false
    end
    
    # The correct invader should be the top hypothesis
    return hypotheses[1].id == UInt64(id)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Exports
# ═══════════════════════════════════════════════════════════════════════════════

export AbductiveHypothesis, TeleportationProperty
export SPIDeterminism, DerangementBijectivity, TropicalIdempotence, SpinConsistency
export WorldNavigator, AbductiveTestResult

export color_distance, color_distance_perceptual
export apply_derangement, invert_derangement, tropical_blend
export simulate_teleportation

export abduce_invader, abduce_from_source
export test_property, test_all_properties, run_abductive_tests
export abductive_roundtrip_test

export init_navigator, get_navigator, teleport!, back!, current_world
export abduce!, jump_hypothesis!, explore_neighbors
