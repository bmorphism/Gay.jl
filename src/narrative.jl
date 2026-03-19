# Narrative Categories with Chromatic Identity
#
# Based on Bumpus et al. "Towards a Unified Theory of Time-Varying Data" (2024)
# Extended with Gay.jl SPI for O(1) SELF reafference in information networks
#
# Two kinds of Narratives:
#   CUMULATIVE: snapshots accumulate over time (union semantics)
#   PERSISTENT: snapshots persist independently (intersection semantics)
#
# Adjunction: Cumulative ⊣ Persistent (with unit/counit)
#
# Para(Narrative): parameterized morphisms for reafference harmonics
# Poly morphisms: polynomial functor interfaces for composition

using SplittableRandoms: SplittableRandom, split
using Colors

export NarrativeKind, CUMULATIVE, PERSISTENT
export Narrative, Snapshot, TemporalMorphism
export NarrativeCategory, NarrativeFunctor
export cumulative_to_persistent, persistent_to_cumulative
export Para, ParaNarrative, reafference_harmonic
export PolyLens, PolyChart, PolyInterface
export o1_self!, converging_harmonics

# ═══════════════════════════════════════════════════════════════════════════
# Narrative Kinds: Cumulative vs Persistent
# ═══════════════════════════════════════════════════════════════════════════

@enum NarrativeKind begin
    CUMULATIVE   # Snapshots accumulate: S(t) = ⋃_{s≤t} S(s)
    PERSISTENT   # Snapshots persist: S(t) independent of S(s)
end

"""
A snapshot at a specific time interval.

Snapshots are the objects of the Narrative category.
Each carries chromatic identity from Gay.jl.
"""
struct Snapshot{T}
    interval::Tuple{Float64, Float64}  # [start, end)
    data::T
    color::RGB{Float64}
    fingerprint::UInt64
end

function Snapshot(interval::Tuple{Float64, Float64}, data::T, rng::SplittableRandom) where T
    color = color_from_rng(split(rng))
    fingerprint = hash(data) ⊻ hash(color)
    Snapshot(interval, data, color, fingerprint)
end

# ═══════════════════════════════════════════════════════════════════════════
# Narrative: Sheaf on Poset of Time Intervals
# ═══════════════════════════════════════════════════════════════════════════

"""
A Narrative is a sheaf on a poset of time intervals.

It specifies:
- Snapshots at each interval (the sheaf values)
- Restriction maps between overlapping intervals
- Kind: cumulative or persistent interpretation

This is the general class of temporal data structures.
"""
struct Narrative{T}
    kind::NarrativeKind
    snapshots::Vector{Snapshot{T}}
    seed::UInt64
    rng::SplittableRandom
    
    # Restriction maps: (i, j) → morphism from snapshot i to snapshot j
    restrictions::Dict{Tuple{Int,Int}, Function}
end

function Narrative(kind::NarrativeKind, seed::UInt64=GAY_SEED)
    Narrative{Any}(
        kind,
        Snapshot{Any}[],
        seed,
        SplittableRandom(seed),
        Dict{Tuple{Int,Int}, Function}()
    )
end

"""
Add a snapshot to the narrative.
"""
function add_snapshot!(N::Narrative{T}, interval::Tuple{Float64, Float64}, data::T) where T
    N.rng = split(N.rng)
    snap = Snapshot(interval, data, N.rng)
    push!(N.snapshots, snap)
    
    # Auto-generate restriction maps based on interval containment
    for (i, other) in enumerate(N.snapshots[1:end-1])
        if interval_contains(other.interval, interval)
            # other ⊇ snap: restriction from other to snap
            N.restrictions[(i, length(N.snapshots))] = 
                if N.kind == CUMULATIVE
                    data -> data  # Identity (accumulation)
                else
                    data -> data  # Projection (persistence)
                end
        end
    end
    
    snap
end

function interval_contains(outer::Tuple{Float64,Float64}, inner::Tuple{Float64,Float64})
    outer[1] <= inner[1] && inner[2] <= outer[2]
end

"""
Get snapshot at time t.
"""
function snapshot_at(N::Narrative, t::Float64)
    for snap in N.snapshots
        if snap.interval[1] <= t < snap.interval[2]
            return snap
        end
    end
    nothing
end

# ═══════════════════════════════════════════════════════════════════════════
# Cumulative ⊣ Persistent Adjunction
# ═══════════════════════════════════════════════════════════════════════════

"""
Convert cumulative narrative to persistent.

This is the right adjoint: U: Cumulative → Persistent
Forgets the accumulation structure, keeping only independent snapshots.
"""
function cumulative_to_persistent(N::Narrative{T}) where T
    @assert N.kind == CUMULATIVE "Input must be cumulative"
    
    persistent = Narrative{T}(
        PERSISTENT,
        copy(N.snapshots),
        N.seed,
        SplittableRandom(N.seed),
        Dict{Tuple{Int,Int}, Function}()
    )
    
    # In persistent view, no restrictions (independent snapshots)
    persistent
end

"""
Convert persistent narrative to cumulative.

This is the left adjoint: F: Persistent → Cumulative
Adds accumulation structure: S(t) = ⋃_{s≤t} S(s)
"""
function persistent_to_cumulative(N::Narrative{T}) where T
    @assert N.kind == PERSISTENT "Input must be persistent"
    
    cumulative = Narrative{T}(
        CUMULATIVE,
        Snapshot{T}[],
        N.seed,
        SplittableRandom(N.seed),
        Dict{Tuple{Int,Int}, Function}()
    )
    
    # Sort by interval start
    sorted = sort(N.snapshots, by=s -> s.interval[1])
    
    # Accumulate
    accumulated_data = nothing
    for snap in sorted
        if accumulated_data === nothing
            accumulated_data = snap.data
        else
            # Union semantics (type-dependent)
            accumulated_data = union_data(accumulated_data, snap.data)
        end
        
        cumulative.rng = split(cumulative.rng)
        new_snap = Snapshot(snap.interval, accumulated_data, cumulative.rng)
        push!(cumulative.snapshots, new_snap)
    end
    
    cumulative
end

# Generic union (override for specific types)
union_data(a, b) = (a, b)
union_data(a::Set, b::Set) = union(a, b)
union_data(a::Vector, b::Vector) = vcat(a, b)
union_data(a::Dict, b::Dict) = merge(a, b)

# ═══════════════════════════════════════════════════════════════════════════
# Temporal Morphisms
# ═══════════════════════════════════════════════════════════════════════════

"""
A morphism between narratives.

Must commute with restriction maps (sheaf morphism condition).
"""
struct TemporalMorphism{S, T}
    source::Narrative{S}
    target::Narrative{T}
    component_maps::Vector{Function}  # One per snapshot
    color::RGB{Float64}  # Chromatic identity of the morphism itself
end

"""
Compose temporal morphisms.
"""
function compose(f::TemporalMorphism, g::TemporalMorphism)
    @assert f.target === g.source "Morphisms not composable"
    
    composed_maps = [g.component_maps[i] ∘ f.component_maps[i] 
                     for i in eachindex(f.component_maps)]
    
    # Color composition via XOR
    composed_color = RGB(
        abs(red(f.color) - red(g.color)),
        abs(green(f.color) - green(g.color)),
        abs(blue(f.color) - blue(g.color))
    )
    
    TemporalMorphism(f.source, g.target, composed_maps, composed_color)
end

# ═══════════════════════════════════════════════════════════════════════════
# Para(Narrative): Parameterized Morphisms
# ═══════════════════════════════════════════════════════════════════════════

"""
Para(C) construction: parameterized morphisms over category C.

A Para morphism is a morphism P × A → B where P is the parameter space.
This enables reafference: observing one's own past states.

Para(Narrative) allows narratives to be parameterized by:
- Seed (for chromatic identity)
- Time offset
- Sampling rate
- Any other hyperparameter
"""
struct Para{P, C}
    parameter_space::Type{P}
    base_category::Type{C}
end

"""
A parameterized narrative: (P, Narrative) pair.
"""
struct ParaNarrative{P, T}
    parameter::P
    narrative::Narrative{T}
end

"""
Para morphism: P × Narrative{S} → Narrative{T}
"""
struct ParaMorphism{P, S, T}
    param_action::Function  # P → (Narrative{S} → Narrative{T})
end

"""
Apply Para morphism with specific parameter.
"""
function apply_para(pm::ParaMorphism{P, S, T}, p::P, n::Narrative{S}) where {P, S, T}
    pm.param_action(p)(n)
end

# ═══════════════════════════════════════════════════════════════════════════
# O(1) SELF: Constant-Time Reafference
# ═══════════════════════════════════════════════════════════════════════════

"""
O(1) SELF lookup: constant-time access to chromatic identity at any point.

Uses Gay.jl's hash-based color generation for O(1) random access,
avoiding O(n) traversal of the narrative history.

This is the key innovation for information networks: self-observation
in constant time regardless of history length.
"""
struct O1Self{T}
    seed::UInt64
    cache::Dict{UInt64, Snapshot{T}}  # Hash → Snapshot
    color_index::Dict{UInt64, RGB{Float64}}  # Index → Color (precomputed)
end

function O1Self(seed::UInt64=GAY_SEED)
    O1Self{Any}(seed, Dict{UInt64, Snapshot{Any}}(), Dict{UInt64, RGB{Float64}}())
end

"""
O(1) self-lookup: get chromatic identity at index.
"""
function o1_self!(self::O1Self, index::UInt64)
    if !haskey(self.color_index, index)
        # O(1) hash-based color generation
        rng = SplittableRandom(self.seed ⊻ index)
        self.color_index[index] = color_from_rng(rng)
    end
    self.color_index[index]
end

"""
O(1) reafference check: compare current to past self.
"""
function o1_reafference(self::O1Self, current_index::UInt64, past_index::UInt64)
    current = o1_self!(self, current_index)
    past = o1_self!(self, past_index)
    
    # Reafference distance
    distance = sqrt(
        (red(current) - red(past))^2 +
        (green(current) - green(past))^2 +
        (blue(current) - blue(past))^2
    )
    
    (current = current, past = past, distance = distance)
end

# ═══════════════════════════════════════════════════════════════════════════
# Converging Harmonics for Reafference
# ═══════════════════════════════════════════════════════════════════════════

"""
Harmonic mode for reafference convergence.
"""
@enum HarmonicMode begin
    FUNDAMENTAL   # Base frequency
    OVERTONE      # Integer multiples
    UNDERTONE     # Integer divisions
    COMBINATION   # Intermodulation
end

"""
Reafference harmonic: oscillation between self-states.

As the narrative evolves, the color trajectory forms harmonics:
- Fundamental: return to origin (period = narrative length)
- Overtones: faster oscillations (period = length/n)
- Undertones: slower oscillations (period = length*n)
- Combination: interference patterns
"""
struct ReafferenceHarmonic
    mode::HarmonicMode
    frequency::Float64  # In color space
    amplitude::Float64  # Deviation from mean
    phase::Float64      # Starting point
    color::RGB{Float64} # Characteristic color
end

"""
Compute converging harmonics from narrative trajectory.

Convergence = trajectory approaches fixed point (self-sameness).
"""
function converging_harmonics(N::Narrative; n_harmonics::Int=8)
    length(N.snapshots) < 2 && return ReafferenceHarmonic[]
    
    # Extract color trajectory
    colors = [s.color for s in N.snapshots]
    n = length(colors)
    
    # Compute FFT-like decomposition in color space
    harmonics = ReafferenceHarmonic[]
    
    for k in 1:n_harmonics
        # Frequency k
        freq = k / n
        
        # Project trajectory onto harmonic
        re_sum = 0.0
        im_sum = 0.0
        for (i, c) in enumerate(colors)
            θ = 2π * freq * i
            val = (red(c) + green(c) + blue(c)) / 3  # Luminance
            re_sum += val * cos(θ)
            im_sum += val * sin(θ)
        end
        
        amplitude = sqrt(re_sum^2 + im_sum^2) / n
        phase = atan(im_sum, re_sum)
        
        mode = if k == 1
            FUNDAMENTAL
        elseif k <= 4
            OVERTONE
        elseif freq < 0.1
            UNDERTONE
        else
            COMBINATION
        end
        
        # Characteristic color for this harmonic
        rng = SplittableRandom(N.seed ⊻ UInt64(k))
        harm_color = color_from_rng(rng)
        
        push!(harmonics, ReafferenceHarmonic(mode, freq, amplitude, phase, harm_color))
    end
    
    harmonics
end

"""
Check if harmonics are converging (decreasing amplitude).
"""
function is_converging(harmonics::Vector{ReafferenceHarmonic})
    length(harmonics) < 2 && return true
    
    # Convergent if overtone amplitudes decrease
    amps = [h.amplitude for h in harmonics]
    
    # Check monotonic decrease after fundamental
    for i in 3:length(amps)
        if amps[i] > amps[i-1] * 1.1  # Allow 10% tolerance
            return false
        end
    end
    true
end

# ═══════════════════════════════════════════════════════════════════════════
# Poly Morphisms: Polynomial Functor Interfaces
# ═══════════════════════════════════════════════════════════════════════════

"""
Poly interface: polynomial functor p(y) = Σᵢ yᴬⁱ

Useful for:
- Lens: bidirectional transformations (get/put)
- Chart: local coordinate systems
- Moore machines: state → output with input → state
"""
abstract type PolyInterface end

"""
Poly Lens: (S, V) with get: S → A and put: S × B → T

For narratives: view a projection and update consistently.
"""
struct PolyLens{S, A, B, T} <: PolyInterface
    get::Function  # S → A
    put::Function  # S × B → T
end

function PolyLens(get::Function, put::Function)
    PolyLens{Any, Any, Any, Any}(get, put)
end

"""
Compose lenses.
"""
function compose_lens(l1::PolyLens, l2::PolyLens)
    PolyLens(
        s -> l2.get(l1.get(s)),
        (s, b) -> l1.put(s, l2.put(l1.get(s), b))
    )
end

"""
Poly Chart: local coordinate system on narrative.

Maps narrative snapshots to a coordinate space where
reafference distance is meaningful.
"""
struct PolyChart{N, C} <: PolyInterface
    domain::N  # Narrative or subset
    coords::Function  # N → C (coordinate map)
    inverse::Function  # C → N (partial inverse)
end

"""
Apply chart to get coordinates.
"""
apply_chart(chart::PolyChart, n) = chart.coords(n)

"""
Lens view of narrative: focus on color trajectory.
"""
function narrative_color_lens(N::Narrative)
    PolyLens(
        N -> [s.color for s in N.snapshots],
        (N, new_colors) -> begin
            for (i, c) in enumerate(new_colors)
                if i <= length(N.snapshots)
                    # Can't mutate immutable Snapshot, so this is conceptual
                    # In practice, would rebuild narrative
                end
            end
            N
        end
    )
end

"""
Chart for reafference: map narrative to distance-from-origin space.
"""
function reafference_chart(N::Narrative)
    origin = isempty(N.snapshots) ? RGB(0.5, 0.5, 0.5) : N.snapshots[1].color
    
    PolyChart(
        N,
        N -> [(s.interval[1], color_distance(s.color, origin)) for s in N.snapshots],
        coords -> coords  # Inverse not always well-defined
    )
end

function color_distance(c1::RGB, c2::RGB)
    sqrt((red(c1) - red(c2))^2 + (green(c1) - green(c2))^2 + (blue(c1) - blue(c2))^2)
end

# ═══════════════════════════════════════════════════════════════════════════
# Para(Narrative) with Poly Morphisms
# ═══════════════════════════════════════════════════════════════════════════

"""
Para(Narrative) with Poly structure.

Combines:
- Para: parameterized morphisms (reafference by parameter variation)
- Poly: lens/chart structure (bidirectional observation)
- Narrative: temporal data (cumulative/persistent)
"""
struct ParaPolyNarrative{P, T}
    parameter::P
    narrative::Narrative{T}
    lens::PolyLens
    chart::PolyChart
end

function ParaPolyNarrative(param::P, narrative::Narrative{T}) where {P, T}
    ParaPolyNarrative(
        param,
        narrative,
        narrative_color_lens(narrative),
        reafference_chart(narrative)
    )
end

"""
O(1) SELF in Para(Narrative) context.

The parameter enables constant-time lookup across the entire
information network by indexing into the parameterized family.
"""
function para_o1_self(ppn::ParaPolyNarrative, index::UInt64)
    # Use parameter to offset into the color space
    param_hash = hash(ppn.parameter)
    combined_seed = ppn.narrative.seed ⊻ param_hash ⊻ index
    
    rng = SplittableRandom(combined_seed)
    color_from_rng(rng)
end

# ═══════════════════════════════════════════════════════════════════════════
# Color from RNG (same as other modules)
# ═══════════════════════════════════════════════════════════════════════════

function color_from_rng(rng::SplittableRandom)::RGB{Float64}
    r = rand(rng)
    g = rand(rng)
    b = rand(rng)
    
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    if max_c > min_c
        scale = 1.0 / (max_c - min_c + 0.3)
        r = (r - min_c) * scale
        g = (g - min_c) * scale
        b = (b - min_c) * scale
    end
    
    RGB{Float64}(clamp(r, 0, 1), clamp(g, 0, 1), clamp(b, 0, 1))
end

# ═══════════════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════════════

function demo_narrative_categories()
    println("╔════════════════════════════════════════════════════════════════╗")
    println("║  Narrative Categories with O(1) SELF Reafference               ║")
    println("║  Cumulative ⊣ Persistent Adjunction + Para(Narrative)          ║")
    println("╚════════════════════════════════════════════════════════════════╝")
    println()
    
    # Create cumulative narrative
    cum = Narrative(CUMULATIVE, GAY_SEED)
    for i in 1:10
        add_snapshot!(cum, (Float64(i-1), Float64(i)), Set([i, i+1, i+2]))
    end
    
    println("Cumulative Narrative (10 snapshots):")
    println("  Kind: $(cum.kind)")
    println("  Snapshots: $(length(cum.snapshots))")
    println("  First color: $(cum.snapshots[1].color)")
    println()
    
    # Convert to persistent
    pers = cumulative_to_persistent(cum)
    println("Persistent Narrative (converted):")
    println("  Kind: $(pers.kind)")
    println("  Snapshots: $(length(pers.snapshots))")
    println()
    
    # O(1) SELF demonstration
    self = O1Self(GAY_SEED)
    println("O(1) SELF Reafference:")
    for i in [1, 100, 10000, 1000000]
        c = o1_self!(self, UInt64(i))
        println("  Index $i: RGB($(round(red(c), digits=2)), $(round(green(c), digits=2)), $(round(blue(c), digits=2)))")
    end
    println("  (All computed in O(1) via hash)")
    println()
    
    # Converging harmonics
    harmonics = converging_harmonics(cum)
    println("Reafference Harmonics:")
    for h in harmonics[1:min(4, length(harmonics))]
        println("  $(h.mode): freq=$(round(h.frequency, digits=3)), amp=$(round(h.amplitude, digits=4))")
    end
    println("  Converging: $(is_converging(harmonics))")
    println()
    
    # Para(Narrative) with Poly
    ppn = ParaPolyNarrative(:seed_42, cum)
    println("Para(Narrative) with Poly:")
    println("  Parameter: $(ppn.parameter)")
    println("  O(1) SELF at index 69: $(para_o1_self(ppn, UInt64(69)))")
    
    (cumulative = cum, persistent = pers, harmonics = harmonics, para = ppn)
end
