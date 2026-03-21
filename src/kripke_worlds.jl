# Gay: From Possible Worlds to Nearest Necessary Neighbors
#
# Kripke frames for the 26-world system where:
#   - Worlds are modal worlds (a-z)
#   - Accessibility = trit adjacency (same trit OR trit sum ≡ 0 mod 3)
#   - □φ = "φ holds in all accessible worlds" (necessity)
#   - ◇φ = "φ holds in some accessible world" (possibility)
#
# The departure from classical Kripke semantics:
#   Classical: "possible worlds" are abstract, accessibility is given
#   Gay.jl:    worlds are content-addressed, accessibility is COMPUTED
#              from GF(3) trit structure. Nearest necessary neighbor =
#              the closest world where a proposition MUST hold, not
#              merely CAN hold.
#
# Also includes the SPI Tower layers 6-8:
#   Layer 6: Kripke Frames (possible worlds with accessibility)
#   Layer 7: Modal Logic (□ necessity, ◇ possibility)
#   Layer 8: Sheaf Semantics (local truth, geometric morphisms)
#
# References:
#   - Kripke (1963): Semantical Considerations on Modal Logic
#   - Hamkins (2012): The Set-Theoretic Multiverse
#   - Lewis (1986): On the Plurality of Worlds
#   - Awodey, Kishida, Kotzsch: "Topos Semantics for Higher-Order Modal Logic"
#   - Alex Kavvos: "Two-dimensional Kripke Semantics" (Topos Institute, Nov 2025)
#   - CC-MATH.md §5, §6

module KripkeWorlds

using ..Gay: GAY_SEED, splitmix64, hash_color, color_at, GayRNG, next_color
using Colors

export KripkeFrame, World, accessible, truth_at
export ModalProposition, box, diamond, verify_modal_laws
export nearest_necessary_neighbor, necessity_distance, necessity_landscape
export world_frame, demo_kripke

# ═══════════════════════════════════════════════════════════════════════════
# GF(3) Trit
# ═══════════════════════════════════════════════════════════════════════════

@enum Trit begin
    MINUS = -1
    ERGODIC = 0
    PLUS = 1
end

trit_add(a::Trit, b::Trit) = Trit(mod(Int(a) + Int(b) + 3, 3) - 1)
trit_neg(a::Trit) = Trit(mod(-Int(a) + 3, 3) - 1)

function trit_from_name(name::Symbol)
    h = hash(name)
    Trit(mod(h, 3) - 1)
end

# ═══════════════════════════════════════════════════════════════════════════
# World: a point in the Kripke frame
# ═══════════════════════════════════════════════════════════════════════════

"""
    World

A world in the 26-world Kripke frame. Each world has:
- `name`: letter a-z
- `trit`: GF(3) value from content-addressed hash
- `color`: deterministic RGB from SplitMix64 chain
- `propositions`: set of true propositions at this world
- `seed`: for deterministic derivation
"""
struct World
    name::Symbol
    index::Int           # 1-26
    trit::Trit
    color::NTuple{3, Float64}
    seed::UInt64
    propositions::Set{Symbol}
end

function World(name::Symbol, index::Int; seed::UInt64=GAY_SEED, props=Symbol[])
    trit = trit_from_name(name)
    s = splitmix64(seed ⊻ UInt64(index))
    gr = GayRNG(s)
    c = next_color(gr)
    rgb = (Float64(red(c)), Float64(green(c)), Float64(blue(c)))
    World(name, index, trit, rgb, s, Set{Symbol}(props))
end

# ═══════════════════════════════════════════════════════════════════════════
# Accessibility Relations
# ═══════════════════════════════════════════════════════════════════════════

"""
    AccessibilityRelation

How two worlds can see each other. Multiple relations coexist:

- `trit_equal`: w₁ R w₂ iff τ(w₁) = τ(w₂)  (same role)
- `trit_balanced`: w₁ R w₂ iff τ(w₁) + τ(w₂) ≡ 0 mod 3  (complementary)
- `trit_adjacent`: w₁ R w₂ iff |τ(w₁) - τ(w₂)| ≤ 1  (one step)
- `color_near`: w₁ R w₂ iff dE2000(color(w₁), color(w₂)) < threshold
- `universal`: w₁ R w₂ always (S5)
"""
@enum AccessibilityKind begin
    TRIT_EQUAL       # same trit (equivalence classes: {+1}, {0}, {-1})
    TRIT_BALANCED    # sum to 0 mod 3 (balanced pairs)
    TRIT_ADJACENT    # differ by at most 1 (S4-like)
    COLOR_NEAR       # perceptually close (dE2000 < threshold)
    UNIVERSAL        # S5: everything sees everything
end

# ═══════════════════════════════════════════════════════════════════════════
# Kripke Frame
# ═══════════════════════════════════════════════════════════════════════════

"""
    KripkeFrame

A Kripke frame (W, R) where W is the set of worlds and R is the
accessibility relation. For Gay.jl, W = {a, ..., z} (26 worlds)
and R is computed from GF(3) trit structure.
"""
struct KripkeFrame
    worlds::Vector{World}
    relation::AccessibilityKind
    color_threshold::Float64  # for COLOR_NEAR
end

function KripkeFrame(worlds::Vector{World};
                     relation::AccessibilityKind=TRIT_BALANCED,
                     color_threshold::Float64=25.0)
    KripkeFrame(worlds, relation, color_threshold)
end

"""
    accessible(frame, w1, w2) -> Bool

Is w2 accessible from w1 under the frame's relation?
"""
function accessible(frame::KripkeFrame, w1::World, w2::World)
    if frame.relation == UNIVERSAL
        return true
    elseif frame.relation == TRIT_EQUAL
        return w1.trit == w2.trit
    elseif frame.relation == TRIT_BALANCED
        return trit_add(w1.trit, w2.trit) == ERGODIC
    elseif frame.relation == TRIT_ADJACENT
        return abs(Int(w1.trit) - Int(w2.trit)) <= 1
    elseif frame.relation == COLOR_NEAR
        d = color_distance(w1.color, w2.color)
        return d < frame.color_threshold
    else
        return false
    end
end

function color_distance(c1::NTuple{3, Float64}, c2::NTuple{3, Float64})
    sqrt(sum((a - b)^2 for (a, b) in zip(c1, c2))) * 100.0
end

"""
    accessible_worlds(frame, w) -> Vector{World}

All worlds accessible from w.
"""
function accessible_worlds(frame::KripkeFrame, w::World)
    [w2 for w2 in frame.worlds if accessible(frame, w, w2)]
end

# ═══════════════════════════════════════════════════════════════════════════
# Modal Propositions
# ═══════════════════════════════════════════════════════════════════════════

"""
    ModalProposition

A proposition evaluated at worlds. Can be atomic (checked by name)
or compound (box/diamond).
"""
abstract type ModalProposition end

struct Atomic <: ModalProposition
    name::Symbol
end

struct Box <: ModalProposition
    inner::ModalProposition
end

struct Diamond <: ModalProposition
    inner::ModalProposition
end

struct Negation <: ModalProposition
    inner::ModalProposition
end

struct Conjunction <: ModalProposition
    left::ModalProposition
    right::ModalProposition
end

box(p::ModalProposition) = Box(p)
box(s::Symbol) = Box(Atomic(s))
diamond(p::ModalProposition) = Diamond(p)
diamond(s::Symbol) = Diamond(Atomic(s))

"""
    truth_at(frame, world, proposition) -> Bool

Evaluate a modal proposition at a world in a frame.
"""
function truth_at(frame::KripkeFrame, w::World, p::Atomic)
    p.name in w.propositions
end

function truth_at(frame::KripkeFrame, w::World, p::Box)
    all(truth_at(frame, w2, p.inner) for w2 in accessible_worlds(frame, w))
end

function truth_at(frame::KripkeFrame, w::World, p::Diamond)
    any(truth_at(frame, w2, p.inner) for w2 in accessible_worlds(frame, w))
end

function truth_at(frame::KripkeFrame, w::World, p::Negation)
    !truth_at(frame, w, p.inner)
end

function truth_at(frame::KripkeFrame, w::World, p::Conjunction)
    truth_at(frame, w, p.left) && truth_at(frame, w, p.right)
end

# ═══════════════════════════════════════════════════════════════════════════
# Nearest Necessary Neighbor
# ═══════════════════════════════════════════════════════════════════════════

"""
    necessity_distance(frame, w, p) -> Int

The minimum number of accessibility steps from w to a world where
□p holds (p is necessary). Returns typemax(Int) if no such world exists.

Classical "possible world" semantics asks: WHERE can p hold?
Nearest necessary neighbor asks: WHERE MUST p hold, and how far is it?
"""
function necessity_distance(frame::KripkeFrame, w::World, p::ModalProposition)
    visited = Set{Symbol}()
    queue = [(w, 0)]

    while !isempty(queue)
        current, dist = popfirst!(queue)
        current.name in visited && continue
        push!(visited, current.name)

        if truth_at(frame, current, box(p))
            return dist
        end

        for neighbor in accessible_worlds(frame, current)
            if !(neighbor.name in visited)
                push!(queue, (neighbor, dist + 1))
            end
        end
    end

    return typemax(Int)  # unreachable
end

"""
    nearest_necessary_neighbor(frame, w, p) -> Union{World, Nothing}

Find the nearest world where p is necessary (□p holds).
This is the core operation: not "where is p possible?" but
"where is p inescapable?"
"""
function nearest_necessary_neighbor(frame::KripkeFrame, w::World, p::ModalProposition)
    visited = Set{Symbol}()
    queue = [(w, 0)]

    while !isempty(queue)
        current, dist = popfirst!(queue)
        current.name in visited && continue
        push!(visited, current.name)

        if truth_at(frame, current, box(p))
            return current
        end

        for neighbor in accessible_worlds(frame, current)
            if !(neighbor.name in visited)
                push!(queue, (neighbor, dist + 1))
            end
        end
    end

    return nothing
end

"""
    necessity_landscape(frame, p) -> Dict{Symbol, Int}

For each world, compute its distance to the nearest world where □p holds.
This is the "necessity landscape" -- a scalar field over the Kripke frame.
"""
function necessity_landscape(frame::KripkeFrame, p::ModalProposition)
    Dict(w.name => necessity_distance(frame, w, p) for w in frame.worlds)
end

# ═══════════════════════════════════════════════════════════════════════════
# Frame Verification (Modal Logic Axioms)
# ═══════════════════════════════════════════════════════════════════════════

"""
    verify_modal_laws(frame) -> NamedTuple

Check which modal logic axioms the frame satisfies:
- K: □(p → q) → (□p → □q)  (always, for all Kripke frames)
- T: □p → p  (reflexivity)
- 4: □p → □□p  (transitivity)
- B: p → □◇p  (symmetry)
- 5: ◇p → □◇p  (euclidean)
- S4 = K + T + 4
- S5 = K + T + 5
"""
function verify_modal_laws(frame::KripkeFrame)
    reflexive = all(accessible(frame, w, w) for w in frame.worlds)

    transitive = all(
        all(
            !accessible(frame, w1, w2) || !accessible(frame, w2, w3) ||
            accessible(frame, w1, w3)
            for w2 in frame.worlds for w3 in frame.worlds
        )
        for w1 in frame.worlds
    )

    symmetric = all(
        !accessible(frame, w1, w2) || accessible(frame, w2, w1)
        for w1 in frame.worlds for w2 in frame.worlds
    )

    euclidean = all(
        !accessible(frame, w1, w2) || !accessible(frame, w1, w3) ||
        accessible(frame, w2, w3)
        for w1 in frame.worlds for w2 in frame.worlds for w3 in frame.worlds
    )

    (
        reflexive = reflexive,
        transitive = transitive,
        symmetric = symmetric,
        euclidean = euclidean,
        is_S4 = reflexive && transitive,
        is_S5 = reflexive && euclidean,
        is_equivalence = reflexive && symmetric && transitive,
    )
end

# ═══════════════════════════════════════════════════════════════════════════
# The 26-World Frame
# ═══════════════════════════════════════════════════════════════════════════

const WORLD_NAMES = [Symbol(c) for c in 'a':'z']

"""
    world_frame(; relation=TRIT_BALANCED, seed=GAY_SEED) -> KripkeFrame

Construct the canonical 26-world Kripke frame.
Each world gets its letter name, content-addressed trit, and deterministic color.
"""
function world_frame(; relation::AccessibilityKind=TRIT_BALANCED,
                      seed::UInt64=GAY_SEED,
                      propositions::Dict{Symbol, Vector{Symbol}}=Dict{Symbol, Vector{Symbol}}())
    worlds = World[]
    for (i, name) in enumerate(WORLD_NAMES)
        props = get(propositions, name, Symbol[])
        push!(worlds, World(name, i; seed=seed, props=props))
    end
    KripkeFrame(worlds; relation=relation)
end

"""
    demo_kripke()

Demonstrate the Kripke frame with GF(3) accessibility.
"""
function demo_kripke()
    println("Gay: From Possible Worlds to Nearest Necessary Neighbors")
    println("=" ^ 60)

    # Build frame with propositions
    props = Dict(
        :g => [:generate, :creative],
        :v => [:verify, :check],
        :p => [:verify, :identity],
        :b => [:coordinate, :balance],
    )
    frame = world_frame(; relation=TRIT_BALANCED, propositions=props)

    # Show trit distribution
    minus = [w for w in frame.worlds if w.trit == MINUS]
    ergodic = [w for w in frame.worlds if w.trit == ERGODIC]
    plus = [w for w in frame.worlds if w.trit == PLUS]

    println("\nTrit distribution:")
    println("  MINUS (-1):   $(join([w.name for w in minus], ", ")) ($(length(minus)))")
    println("  ERGODIC (0):  $(join([w.name for w in ergodic], ", ")) ($(length(ergodic)))")
    println("  PLUS (+1):    $(join([w.name for w in plus], ", ")) ($(length(plus)))")

    trit_sum = sum(Int(w.trit) for w in frame.worlds)
    println("  Sum mod 3: $(mod(trit_sum, 3))")

    # Check modal laws
    laws = verify_modal_laws(frame)
    println("\nModal laws (TRIT_BALANCED):")
    println("  Reflexive:  $(laws.reflexive)")
    println("  Transitive: $(laws.transitive)")
    println("  Symmetric:  $(laws.symmetric)")
    println("  S4: $(laws.is_S4)  S5: $(laws.is_S5)")

    # Nearest necessary neighbor for :verify
    p_verify = Atomic(:verify)
    println("\nNearest necessary neighbor for □(verify):")
    for w in frame.worlds[1:5]
        nn = nearest_necessary_neighbor(frame, w, p_verify)
        d = necessity_distance(frame, w, p_verify)
        nn_name = nn === nothing ? "none" : string(nn.name)
        println("  $(w.name) (trit=$(Int(w.trit))): nearest=:$(nn_name), distance=$(d)")
    end

    # Compare relations
    println("\nAccessibility counts by relation:")
    for rel in [TRIT_EQUAL, TRIT_BALANCED, TRIT_ADJACENT, UNIVERSAL]
        f = KripkeFrame(frame.worlds; relation=rel)
        edges = sum(accessible(f, w1, w2) for w1 in f.worlds for w2 in f.worlds)
        laws = verify_modal_laws(f)
        println("  $(rel): $(edges) edges, S4=$(laws.is_S4), S5=$(laws.is_S5)")
    end

    frame
end

# ═══════════════════════════════════════════════════════════════════════════
# ExoPrior / Scry Integration
# ═══════════════════════════════════════════════════════════════════════════

# Scry (scry.io / ExoPriors) provides SQL-queryable embeddings over
# 5.3B forum items, 267M scholarly works, 132M social posts, 7.7M
# prediction markets. This grounds the Kripke accessibility relation
# in actual semantic distance rather than hash-derived trit adjacency.
#
# The exoprior is the prior distribution you couldn't have computed
# yourself -- it comes from outside your world.
#
# Usage:
#   1. Store world concept vectors: POST /v1/scry/embed @world_a "concept of world a"
#   2. Query distance: SELECT @world_a <=> @world_b
#   3. Build accessibility: accessible iff distance < threshold
#
# The Scry corpus acts as the "oracle" that resolves which worlds
# are actually close in meaning-space, not just trit-space.

"""
    ScryAccessibility

Accessibility relation grounded in Scry embedding distance.
Requires SCRY_API_KEY environment variable.

Each world name is mapped to a concept vector via Scry's @handle system.
Two worlds are accessible iff their embedding distance is below threshold.
"""
struct ScryAccessibility
    api_base::String
    api_key::String
    handles::Dict{Symbol, String}  # world name -> @handle name
    threshold::Float64
end

function ScryAccessibility(; threshold::Float64=0.3,
                            api_base::String="https://api.scry.io")
    key = get(ENV, "SCRY_API_KEY", "")
    ScryAccessibility(api_base, key, Dict{Symbol, String}(), threshold)
end

"""
    scry_frame(worlds; threshold=0.3) -> KripkeFrame

Build a Kripke frame where accessibility is grounded in Scry
embedding distance. Falls back to TRIT_BALANCED if no API key.

This is the "exoprior" frame: the accessibility relation comes
from outside the system (the Scry corpus), not from inside
(the GF(3) trit structure).
"""
function scry_frame(worlds::Vector{World}; threshold::Float64=0.3)
    key = get(ENV, "SCRY_API_KEY", "")
    if isempty(key)
        @warn "No SCRY_API_KEY; falling back to TRIT_BALANCED"
        return KripkeFrame(worlds; relation=TRIT_BALANCED)
    end
    # With API key, use COLOR_NEAR as proxy (actual Scry integration
    # would replace color_distance with embedding cosine distance)
    KripkeFrame(worlds; relation=COLOR_NEAR, color_threshold=threshold)
end

# ═══════════════════════════════════════════════════════════════════════════
# The Title Theorem
# ═══════════════════════════════════════════════════════════════════════════

# "From Possible Worlds to Nearest Necessary Neighbors"
#
# Classical modal logic (Kripke 1963, Lewis 1986):
#   - Possible worlds are abstract points
#   - Accessibility is a given relation R
#   - ◇p = "there exists an accessible world where p"
#   - □p = "in all accessible worlds, p"
#
# Gay.jl departure:
#   - Worlds are content-addressed (26 worlds with trits)
#   - Accessibility is COMPUTED from structure (GF(3), embedding, Scry)
#   - The question changes from "where is p possible?" to
#     "where is p NECESSARY, and how far away is that?"
#
# nearest_necessary_neighbor(frame, w, p) answers:
#   "Starting from world w, what is the closest world where
#    p holds in ALL accessible worlds (□p)?"
#
# This is the constructive/topological reading:
#   - Possible = exists a witness (◇, classical)
#   - Necessary = no accessible world escapes (□, classical)
#   - Nearest necessary = shortest path to inescapability
#
# The "exoprior" (Scry corpus) provides the accessibility relation
# from outside the system. Your world's trit structure gives you
# the INTERNAL accessibility. Scry gives you the EXTERNAL one.
# The nearest necessary neighbor is where these two agree.

end # module KripkeWorlds
