# ══════════════════════════════════════════════════════════════════════════════
# Three-Match: GF(3) Colored Subgraph Isomorphism for Compositional Verification
# ══════════════════════════════════════════════════════════════════════════════
#
# Implements the 3-MATCH gadget from COLOR_OBSTRUCTIONS_COMPOSITIONALITY.md
#
# Core concept: Three items form a valid match iff their trits sum to 0 (mod 3).
# This provides a compositional verification primitive for:
# - Skill triplet validation
# - Tripartite agent allocation
# - GF(3) conservation checking
# - Obstruction detection
#
# INVARIANTS:
# ┌────────────────────────────────────────────────────────────────────────────┐
# │ trit(a) + trit(b) + trit(c) ≡ 0 (mod 3) → valid triplet                   │
# │ XOR(fingerprint(a), fingerprint(b), fingerprint(c)) = fingerprint(abc)    │
# │ color(seed, i) is deterministic and order-independent                      │
# └────────────────────────────────────────────────────────────────────────────┘
#
# ══════════════════════════════════════════════════════════════════════════════

module ThreeMatch

# SplitMix64 constants (duplicated for standalone use)
const GOLDEN = 0x9e3779b97f4a7c15
const MIX1 = 0xbf58476d1ce4e5b9
const MIX2 = 0x94d049bb133111eb
const GAY_SEED = UInt64(0x6761795f636f6c6f)

"""SplitMix64 bijection for deterministic hashing."""
function splitmix64(x::UInt64)::UInt64
    x += GOLDEN
    x = (x ⊻ (x >> 30)) * MIX1
    x = (x ⊻ (x >> 27)) * MIX2
    x ⊻ (x >> 31)
end

export ThreeMatchWorld, MatchLeg, ThreeMatchTriangle
export seed_to_color, color_to_fingerprint, seed_to_fingerprint
export verify_three_match, three_match_distance
export Trit, MINUS, ERGODIC, PLUS, trit_sum, gf3_conserved

# Obstruction types
export Obstruction, SeedBlock, ColorMismatch, FingerprintCollision
export ObstructionSite, insert_obstruction!, remove_obstruction!
export list_obstructions, obstruction_density, is_obstructed

# World builders
export world_three_match, world_skill_triplets

# ══════════════════════════════════════════════════════════════════════════════
# GF(3) Trit Type
# ══════════════════════════════════════════════════════════════════════════════

"""
    Trit

Element of GF(3): {-1, 0, +1} with mod 3 arithmetic.
"""
@enum Trit::Int8 begin
    MINUS = -1
    ERGODIC = 0
    PLUS = 1
end

"""Convert integer to Trit."""
function Trit(x::Integer)::Trit
    m = mod(x, 3)
    m == 0 && return ERGODIC
    m == 1 && return PLUS
    return MINUS
end

"""Sum trits and return result in GF(3)."""
function trit_sum(trits::Vector{Trit})::Trit
    s = sum(Int8(t) for t in trits)
    Trit(s)
end

"""Check if collection of trits is GF(3) conserved (sum ≡ 0 mod 3)."""
function gf3_conserved(trits::Vector{Trit})::Bool
    trit_sum(trits) == ERGODIC
end

"""Compute trit from hue angle."""
function trit_from_hue(hue::Float64)::Trit
    h = mod(hue, 360.0)
    if h < 60 || h >= 300
        return PLUS      # Warm colors: orange/red/magenta
    elseif h < 180
        return ERGODIC   # Cool colors: yellow/green/cyan  
    else
        return MINUS     # Cold colors: blue/purple
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# Color and Fingerprint Functions
# ══════════════════════════════════════════════════════════════════════════════

"""
    seed_to_color(seed, index) -> (r, g, b, hue, trit)

Compute deterministic color from seed and index.
Returns RGB values (0-1), hue (0-360), and GF(3) trit.
"""
function seed_to_color(seed::UInt64, index::Integer)
    h = splitmix64(seed ⊻ UInt64(index))
    
    # Extract RGB
    r = Float64((h >> 16) & 0xFF) / 255.0
    g = Float64((h >> 8) & 0xFF) / 255.0
    b = Float64(h & 0xFF) / 255.0
    
    # Compute hue from RGB
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    delta = cmax - cmin
    
    hue = if delta < 1e-10
        0.0
    elseif cmax == r
        60.0 * mod((g - b) / delta, 6.0)
    elseif cmax == g
        60.0 * ((b - r) / delta + 2.0)
    else
        60.0 * ((r - g) / delta + 4.0)
    end
    
    hue = mod(hue, 360.0)
    trit = trit_from_hue(hue)
    
    (r=r, g=g, b=b, hue=hue, trit=trit)
end

"""
    color_to_fingerprint(colors) -> UInt64

XOR-combine colors into a single fingerprint (order-independent).
"""
function color_to_fingerprint(colors::Vector)::UInt64
    fp = UInt64(0)
    for c in colors
        h = splitmix64(reinterpret(UInt64, c.r) ⊻ 
                       (reinterpret(UInt64, c.g) << 21) ⊻ 
                       (reinterpret(UInt64, c.b) << 42))
        fp ⊻= h
    end
    fp
end

"""
    seed_to_fingerprint(seed, n) -> UInt64

Compute fingerprint for n colors from seed.
"""
function seed_to_fingerprint(seed::UInt64, n::Integer)::UInt64
    fp = UInt64(0)
    for i in 1:n
        h = splitmix64(seed ⊻ UInt64(i))
        fp ⊻= h
    end
    fp
end

# ══════════════════════════════════════════════════════════════════════════════
# Three-Match Structures
# ══════════════════════════════════════════════════════════════════════════════

"""
    MatchLeg

One leg of a three-match triangle.
"""
struct MatchLeg
    name::String
    seed::UInt64
    trit::Trit
    color::NamedTuple{(:r, :g, :b, :hue, :trit), Tuple{Float64, Float64, Float64, Float64, Trit}}
end

function MatchLeg(name::String, seed::UInt64, index::Integer=0)
    color = seed_to_color(seed, index)
    MatchLeg(name, seed, color.trit, color)
end

function MatchLeg(name::String, trit::Trit)
    seed = splitmix64(hash(name))
    color = seed_to_color(seed, 0)
    MatchLeg(name, seed, trit, color)
end

"""
    ThreeMatchTriangle

A triplet of items forming a potential GF(3) match.
"""
struct ThreeMatchTriangle
    legs::NTuple{3, MatchLeg}
    conserved::Bool
    fingerprint::UInt64
end

function ThreeMatchTriangle(a::MatchLeg, b::MatchLeg, c::MatchLeg)
    trits = [a.trit, b.trit, c.trit]
    conserved = gf3_conserved(trits)
    fp = a.seed ⊻ b.seed ⊻ c.seed
    ThreeMatchTriangle((a, b, c), conserved, fp)
end

"""
    verify_three_match(triangle) -> Bool

Verify that a triangle satisfies GF(3) conservation.
"""
function verify_three_match(t::ThreeMatchTriangle)::Bool
    t.conserved
end

"""
    three_match_distance(triangle) -> Int

Distance from valid match: 0 if conserved, otherwise |sum| mod 3.
"""
function three_match_distance(t::ThreeMatchTriangle)::Int
    s = sum(Int8(leg.trit) for leg in t.legs)
    abs(mod(s + 1, 3) - 1)  # Distance to 0
end

# ══════════════════════════════════════════════════════════════════════════════
# Obstruction Types
# ══════════════════════════════════════════════════════════════════════════════

"""
    Obstruction

Base type for compositionality obstructions.
"""
abstract type Obstruction end

"""Seed cannot generate valid color at this index."""
struct SeedBlock <: Obstruction
    seed::UInt64
    index::Int
    reason::String
end

"""Colors don't match required constraints."""
struct ColorMismatch <: Obstruction
    expected_trit::Trit
    actual_trit::Trit
    location::String
end

"""Two different inputs produce same fingerprint."""
struct FingerprintCollision <: Obstruction
    seed1::UInt64
    seed2::UInt64
    fingerprint::UInt64
end

"""Location where obstruction can be inserted/detected."""
struct ObstructionSite
    name::String
    seed::UInt64
    index::Int
    current_trit::Trit
end

# ══════════════════════════════════════════════════════════════════════════════
# Obstruction Detection
# ══════════════════════════════════════════════════════════════════════════════

"""
    is_obstructed(triangle) -> Bool

Check if a triangle has any obstruction (not conserved).
"""
function is_obstructed(t::ThreeMatchTriangle)::Bool
    !t.conserved
end

"""
    list_obstructions(triangles) -> Vector{Obstruction}

List all obstructions in a collection of triangles.
"""
function list_obstructions(triangles::Vector{ThreeMatchTriangle})::Vector{Obstruction}
    obs = Obstruction[]
    for t in triangles
        if !t.conserved
            s = sum(Int8(leg.trit) for leg in t.legs)
            names = join([leg.name for leg in t.legs], ", ")
            push!(obs, ColorMismatch(ERGODIC, Trit(s), names))
        end
    end
    obs
end

"""
    obstruction_density(triangles) -> Float64

Fraction of triangles that are obstructed (0 = all valid, 1 = all obstructed).
"""
function obstruction_density(triangles::Vector{ThreeMatchTriangle})::Float64
    n_obstructed = count(!t.conserved for t in triangles)
    n_obstructed / length(triangles)
end

# ══════════════════════════════════════════════════════════════════════════════
# ThreeMatchWorld (Persistent State Builder)
# ══════════════════════════════════════════════════════════════════════════════

"""
    ThreeMatchWorld

A world containing validated three-match triangles.
Implements the world_ pattern for composability.
"""
struct ThreeMatchWorld
    triangles::Vector{ThreeMatchTriangle}
    legs::Dict{String, MatchLeg}
    valid_count::Int
    obstructed_count::Int
    fingerprint::UInt64
end

# Required world_ pattern methods
Base.length(w::ThreeMatchWorld) = length(w.triangles)

function Base.merge(w1::ThreeMatchWorld, w2::ThreeMatchWorld)::ThreeMatchWorld
    all_triangles = vcat(w1.triangles, w2.triangles)
    all_legs = merge(w1.legs, w2.legs)
    valid = count(t.conserved for t in all_triangles)
    obstructed = length(all_triangles) - valid
    fp = w1.fingerprint ⊻ w2.fingerprint
    ThreeMatchWorld(all_triangles, all_legs, valid, obstructed, fp)
end

function fingerprint(w::ThreeMatchWorld)::UInt64
    w.fingerprint
end

"""
    world_three_match(items; seed=GAY_SEED) -> ThreeMatchWorld

Build a ThreeMatchWorld from items with (name, trit) pairs.
"""
function world_three_match(items::Vector{Tuple{String, Trit}}; seed::UInt64=GAY_SEED)::ThreeMatchWorld
    legs = Dict{String, MatchLeg}()
    for (name, trit) in items
        legs[name] = MatchLeg(name, trit)
    end
    
    triangles = ThreeMatchTriangle[]
    fp = UInt64(0)
    
    # Generate all valid triplets (MINUS, ERGODIC, PLUS)
    minus_legs = [l for l in values(legs) if l.trit == MINUS]
    ergodic_legs = [l for l in values(legs) if l.trit == ERGODIC]
    plus_legs = [l for l in values(legs) if l.trit == PLUS]
    
    for m in minus_legs, e in ergodic_legs, p in plus_legs
        t = ThreeMatchTriangle(m, e, p)
        push!(triangles, t)
        fp ⊻= t.fingerprint
    end
    
    valid = count(t.conserved for t in triangles)
    obstructed = length(triangles) - valid
    
    ThreeMatchWorld(triangles, legs, valid, obstructed, fp)
end

"""
    world_skill_triplets(skills) -> ThreeMatchWorld

Build a ThreeMatchWorld from skill definitions.
Each skill is (name, trit_value) where trit_value ∈ {-1, 0, 1}.
"""
function world_skill_triplets(skills::Vector{Tuple{String, Int}})::ThreeMatchWorld
    items = [(name, Trit(t)) for (name, t) in skills]
    world_three_match(items)
end

# ══════════════════════════════════════════════════════════════════════════════
# Demo / CLI
# ══════════════════════════════════════════════════════════════════════════════

function world_three_match_demo()
    println("═" ^ 70)
    println("THREE-MATCH: GF(3) Colored Subgraph Isomorphism")
    println("═" ^ 70)
    println()
    
    # Example skills
    skills = [
        ("sheaf-cohomology", -1),
        ("code-review", -1),
        ("spi-parallel-verify", -1),
        ("bumpus-narratives", 0),
        ("acsets", 0),
        ("gh-cli", 0),
        ("world-hopping", 1),
        ("julia-gay", 1),
        ("triad-interleave", 1),
    ]
    
    println("Input skills:")
    for (name, t) in skills
        trit_str = t == -1 ? "MINUS" : t == 0 ? "ERGODIC" : "PLUS"
        println("  $(name): $(trit_str) ($(t))")
    end
    println()
    
    # Build world
    world = world_skill_triplets(skills)
    
    println("ThreeMatchWorld:")
    println("  Triangles: $(length(world.triangles))")
    println("  Valid: $(world.valid_count)")
    println("  Obstructed: $(world.obstructed_count)")
    println("  Fingerprint: 0x$(string(world.fingerprint, base=16, pad=16))")
    println()
    
    # Show some valid triplets
    println("Valid triplets (first 5):")
    for t in world.triangles[1:min(5, length(world.triangles))]
        names = [leg.name for leg in t.legs]
        println("  $(names[1]) ⊗ $(names[2]) ⊗ $(names[3]) = 0 ✓")
    end
    println()
    
    # Obstruction density
    density = obstruction_density(world.triangles)
    println("Obstruction density: $(round(density * 100, digits=2))%")
    
    println()
    println("═" ^ 70)
end

end # module ThreeMatch
