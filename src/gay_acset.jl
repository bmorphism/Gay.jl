# GayACSet.jl: Chromatic Attributed C-Sets with Org Monad Structure
# ==================================================================
#
# Equivalences, Correspondences, Dualities, and Superpositions with TileACSet
#
# From Spivak's Org: y² → 𝔪_{y²∨y²∨y²}
#   "Delegate bit selection to 3 subagents"
#   - First 2 agree → return that bit
#   - Disagree → invoke 3rd as tie-breaker
#   - Colors depict outcomes of different subagents
#
# Org_𝔪^♯(p; q₁,...,qₖ) := 𝔠_{[p, 𝔪_{q₁∨...∨qₖ}]}
#   - ∨: monoidal product (bunch of agents choosable at each step)
#   - 𝔪: free monad (planning/flow-chart)
#   - [-,-]: internal hom (tasks forward, outcomes backward)
#   - 𝔠: cofree comonad (temporal evolution)
#
# KEY INSIGHT: GayACSet provides the chromatic identity layer that makes
# the abstract Org monad structure *concrete* and *verifiable* via SPI.
#
# DUALITY TABLE:
# ┌────────────────────────┬────────────────────────┐
# │ GayACSet               │ TileACSet              │
# ├────────────────────────┼────────────────────────┤
# │ Vertices (objects)     │ Tiles (cells)          │
# │ Edges (morphisms)      │ Adjacencies (gluing)   │
# │ Seed → Color (SPI)     │ Position → Color (SPI) │
# │ Free monad 𝔪          │ Substitution rules     │
# │ Cofree comonad 𝔠      │ Inflation hierarchy    │
# │ Internal hom [-,-]    │ Boundary matching      │
# │ ∨ (monoidal)          │ Tile superposition     │
# │ Task delegation       │ Hierarchical tiling    │
# └────────────────────────┴────────────────────────┘

module GayACSet

using SplittableRandoms: SplittableRandom, split
using Colors

export
    # Core ACSet Types
    SchemaSpec, GaySchema, TileSchema, OrgSchema,
    GayObject, GayMorphism, GayAttribute,
    
    # Chromatic ACSet
    ChromaticACSet, gay_acset, tile_acset,
    add_vertex!, add_edge!, add_tile!, add_adjacency!,
    gay_color, tile_color, spi_fingerprint,
    
    # Org Monad Structure
    OrgMonadACSet, FreeMonad, CofreeComonad, InternalHom,
    MonoidalSum, TaskDelegation,
    y_squared, m_free, c_cofree, internal_hom, vee_sum,
    
    # Correspondences
    GayTileCorrespondence, Equivalence, Duality, Discrepancy,
    find_equivalences, find_dualities, find_discrepancies,
    superposition, measure_coherence,
    
    # Huffman as Org
    HuffmanTree, huffman_org, encode_bitstring, decode_bitstring,
    
    # Demo
    world_gay_acset, world_org_correspondence

# ═══════════════════════════════════════════════════════════════════════════════
# Core PRNG (SPI compliant)
# ═══════════════════════════════════════════════════════════════════════════════

const GAY_SEED = UInt64(1069)

@inline function splitmix64(state::UInt64)::Tuple{UInt64, UInt64}
    z = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ⊻ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    z ⊻ (z >> 31), (state + 1) & 0xFFFFFFFFFFFFFFFF
end

@inline function color_from_seed(seed::UInt64)::RGB{Float64}
    r, s1 = splitmix64(seed)
    g, s2 = splitmix64(s1)
    b, _  = splitmix64(s2)
    RGB((r >> 56) / 255.0, (g >> 56) / 255.0, (b >> 56) / 255.0)
end

@inline function name_to_seed(name::String)::UInt64
    h = UInt64(0xcbf29ce484222325)
    for byte in codeunits(name)
        h = h ⊻ UInt64(byte)
        h = h * UInt64(0x100000001b3)
    end
    h
end

# ═══════════════════════════════════════════════════════════════════════════════
# Schema Specification: The Category C in C-Set
# ═══════════════════════════════════════════════════════════════════════════════

"""
    SchemaSpec

Abstract specification for the schema of an ACSet.
Defines objects, morphisms, and attributes.
"""
abstract type SchemaSpec end

"""
    GaySchema <: SchemaSpec

Schema for chromatic graphs with Gay.jl coloring.
Objects: V (vertices), E (edges)
Morphisms: src, tgt : E → V
Attributes: seed, color, fingerprint
"""
struct GaySchema <: SchemaSpec
    name::String
    objects::Vector{Symbol}
    morphisms::Vector{Tuple{Symbol, Symbol, Symbol}}  # (name, dom, cod)
    attributes::Vector{Tuple{Symbol, Symbol, Type}}   # (name, ob, type)
end

function GaySchema(name::String="ColoredGraph")
    GaySchema(
        name,
        [:V, :E],
        [(:src, :E, :V), (:tgt, :E, :V)],
        [(:seed, :V, UInt64), (:color, :V, RGB{Float64}), 
         (:eseed, :E, UInt64), (:ecolor, :E, RGB{Float64})]
    )
end

"""
    TileSchema <: SchemaSpec

Schema for aperiodic tilings with cryptochrome coloring.
Objects: T (tiles), A (adjacencies)
Morphisms: left, right : A → T
Attributes: position, color, bandwidth
"""
struct TileSchema <: SchemaSpec
    name::String
    objects::Vector{Symbol}
    morphisms::Vector{Tuple{Symbol, Symbol, Symbol}}
    attributes::Vector{Tuple{Symbol, Symbol, Type}}
end

function TileSchema(name::String="AperiodicTiling")
    TileSchema(
        name,
        [:T, :A],
        [(:left, :A, :T), (:right, :A, :T)],
        [(:position, :T, Tuple{Float64, Float64}), 
         (:color, :T, RGB{Float64}),
         (:bandwidth, :T, Float64)]
    )
end

"""
    OrgSchema <: SchemaSpec

Schema for Org monad delegation structure.
Objects: Agent, Task, Outcome, Plan
Morphisms: performs : Agent → Task, produces : Agent → Outcome
           delegates : Agent → Plan
Attributes: seed, color, credibility
"""
struct OrgSchema <: SchemaSpec
    name::String
    objects::Vector{Symbol}
    morphisms::Vector{Tuple{Symbol, Symbol, Symbol}}
    attributes::Vector{Tuple{Symbol, Symbol, Type}}
end

function OrgSchema(name::String="OrgDelegation")
    OrgSchema(
        name,
        [:Agent, :Task, :Outcome, :Plan],
        [(:performs, :Agent, :Task), 
         (:produces, :Agent, :Outcome),
         (:delegates, :Plan, :Agent)],
        [(:seed, :Agent, UInt64), 
         (:color, :Agent, RGB{Float64}),
         (:credibility, :Agent, Float64)]
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# GayObject: Vertices/Tiles with Chromatic Identity
# ═══════════════════════════════════════════════════════════════════════════════

"""
    GayObject

An object in a GayACSet with deterministic chromatic identity.
"""
struct GayObject
    id::Int64
    ob_type::Symbol     # :V, :E, :T, :A, :Agent, etc.
    seed::UInt64
    color::RGB{Float64}
    data::Dict{Symbol, Any}
end

function GayObject(id::Int64, ob_type::Symbol; base_seed::UInt64=GAY_SEED)
    seed = base_seed ⊻ UInt64(id) ⊻ name_to_seed(String(ob_type))
    color = color_from_seed(seed)
    GayObject(id, ob_type, seed, color, Dict{Symbol, Any}())
end

"""
    GayMorphism

A morphism in a GayACSet with deterministic color from source/target.
"""
struct GayMorphism
    id::Int64
    mor_type::Symbol    # :src, :tgt, :left, :right, etc.
    source_id::Int64
    target_id::Int64
    seed::UInt64
    color::RGB{Float64}
end

function GayMorphism(id::Int64, mor_type::Symbol, src::Int64, tgt::Int64; 
                     base_seed::UInt64=GAY_SEED)
    seed = base_seed ⊻ UInt64(id) ⊻ UInt64(src * 1000 + tgt)
    color = color_from_seed(seed)
    GayMorphism(id, mor_type, src, tgt, seed, color)
end

# ═══════════════════════════════════════════════════════════════════════════════
# ChromaticACSet: The Full Attributed C-Set with Colors
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ChromaticACSet

An Attributed C-Set where every object and morphism has a deterministic
chromatic identity via Gay.jl SPI.

This is the concrete realization of the abstract categorical structure.
"""
mutable struct ChromaticACSet
    schema::SchemaSpec
    seed::UInt64
    
    # Object parts: ob_type => id => GayObject
    objects::Dict{Symbol, Dict{Int64, GayObject}}
    
    # Morphism parts: mor_type => id => GayMorphism
    morphisms::Dict{Symbol, Dict{Int64, GayMorphism}}
    
    # Attribute storage
    attributes::Dict{Symbol, Dict{Int64, Any}}
    
    # Next ID for each type
    next_id::Dict{Symbol, Int64}
    
    # SPI fingerprint (XOR of all object/morphism seeds)
    fingerprint::UInt64
end

function ChromaticACSet(schema::SchemaSpec; seed::UInt64=GAY_SEED)
    objects = Dict(ob => Dict{Int64, GayObject}() for ob in schema.objects)
    morphisms = Dict(m[1] => Dict{Int64, GayMorphism}() for m in schema.morphisms)
    attributes = Dict{Symbol, Dict{Int64, Any}}()
    next_id = Dict(ob => Int64(1) for ob in schema.objects)
    
    for m in schema.morphisms
        next_id[m[1]] = Int64(1)
    end
    
    ChromaticACSet(schema, seed, objects, morphisms, attributes, next_id, UInt64(0))
end

# Convenience constructors
gay_acset(; seed::UInt64=GAY_SEED) = ChromaticACSet(GaySchema(); seed=seed)
tile_acset(; seed::UInt64=GAY_SEED) = ChromaticACSet(TileSchema(); seed=seed)

"""
    add_vertex!(acs::ChromaticACSet; kwargs...) -> Int64

Add a vertex to a GayACSet, returning its ID.
"""
function add_vertex!(acs::ChromaticACSet; kwargs...)
    id = acs.next_id[:V]
    acs.next_id[:V] += 1
    
    obj = GayObject(id, :V; base_seed=acs.seed)
    
    # Store additional attributes
    for (k, v) in kwargs
        obj.data[k] = v
    end
    
    acs.objects[:V][id] = obj
    acs.fingerprint ⊻= obj.seed
    
    id
end

"""
    add_edge!(acs::ChromaticACSet, src::Int64, tgt::Int64) -> Int64

Add an edge between two vertices.
"""
function add_edge!(acs::ChromaticACSet, src::Int64, tgt::Int64)
    id = acs.next_id[:E]
    acs.next_id[:E] += 1
    
    obj = GayObject(id, :E; base_seed=acs.seed)
    acs.objects[:E][id] = obj
    
    # Create src and tgt morphisms
    src_mor = GayMorphism(id, :src, id, src; base_seed=acs.seed)
    tgt_mor = GayMorphism(id, :tgt, id, tgt; base_seed=acs.seed)
    
    acs.morphisms[:src][id] = src_mor
    acs.morphisms[:tgt][id] = tgt_mor
    
    acs.fingerprint ⊻= obj.seed ⊻ src_mor.seed ⊻ tgt_mor.seed
    
    id
end

"""
    add_tile!(acs::ChromaticACSet, position::Tuple{Float64, Float64}) -> Int64

Add a tile to a TileACSet.
"""
function add_tile!(acs::ChromaticACSet, position::Tuple{Float64, Float64})
    id = acs.next_id[:T]
    acs.next_id[:T] += 1
    
    obj = GayObject(id, :T; base_seed=acs.seed)
    obj.data[:position] = position
    
    # Bandwidth from seed (SPI)
    bw, _ = splitmix64(obj.seed)
    obj.data[:bandwidth] = (bw & 0xFFFF) / 65535.0
    
    acs.objects[:T][id] = obj
    acs.fingerprint ⊻= obj.seed
    
    id
end

"""
    add_adjacency!(acs::ChromaticACSet, left::Int64, right::Int64) -> Int64

Add an adjacency between two tiles.
"""
function add_adjacency!(acs::ChromaticACSet, left::Int64, right::Int64)
    id = acs.next_id[:A]
    acs.next_id[:A] += 1
    
    obj = GayObject(id, :A; base_seed=acs.seed)
    acs.objects[:A][id] = obj
    
    left_mor = GayMorphism(id, :left, id, left; base_seed=acs.seed)
    right_mor = GayMorphism(id, :right, id, right; base_seed=acs.seed)
    
    acs.morphisms[:left][id] = left_mor
    acs.morphisms[:right][id] = right_mor
    
    acs.fingerprint ⊻= obj.seed ⊻ left_mor.seed ⊻ right_mor.seed
    
    id
end

# Color accessors
gay_color(acs::ChromaticACSet, ob::Symbol, id::Int64) = acs.objects[ob][id].color
tile_color(acs::ChromaticACSet, id::Int64) = acs.objects[:T][id].color
spi_fingerprint(acs::ChromaticACSet) = acs.fingerprint

# ═══════════════════════════════════════════════════════════════════════════════
# Org Monad Structure: 𝔪, 𝔠, [-,-], ∨
# ═══════════════════════════════════════════════════════════════════════════════

"""
    FreeMonad

The free monad 𝔪_p on a polynomial p.
Represents planning/flow-chart structure.

For y²: 𝔪_y² = trees of binary choices
"""
abstract type FreeMonadElement end

struct FreeLeaf <: FreeMonadElement
    value::Bool  # For y²
    color::RGB{Float64}
end

struct FreeBranch <: FreeMonadElement
    on_true::FreeMonadElement
    on_false::FreeMonadElement
    color::RGB{Float64}
end

struct FreeMonad
    polynomial::Symbol  # :y2, :y3, etc.
    root::FreeMonadElement
    depth::Int
    seed::UInt64
end

"""
    y_squared(; seed::UInt64=GAY_SEED)

Construct y² polynomial type.
"""
function y_squared(; seed::UInt64=GAY_SEED)
    color = color_from_seed(seed)
    FreeMonad(:y2, FreeLeaf(false, color), 0, seed)
end

"""
    m_free(p::FreeMonad; depth::Int=3)

Construct 𝔪_p, the free monad on p with given depth.
"""
function m_free(p::FreeMonad; depth::Int=3)
    if depth == 0
        return p
    end
    
    function build_tree(d::Int, s::UInt64)
        color = color_from_seed(s)
        if d == 0
            val, _ = splitmix64(s)
            FreeLeaf((val & 1) == 1, color)
        else
            s1, s2 = splitmix64(s)
            FreeBranch(
                build_tree(d - 1, s1),
                build_tree(d - 1, s2),
                color
            )
        end
    end
    
    FreeMonad(p.polynomial, build_tree(depth, p.seed), depth, p.seed)
end

"""
    CofreeComonad

The cofree comonad 𝔠_p: temporal evolution.
A stream of planning structures.
"""
struct CofreeComonad
    polynomial::Symbol
    head::FreeMonad
    tail::Vector{FreeMonad}  # Lazy stream (finite prefix)
    seed::UInt64
end

"""
    c_cofree(p::FreeMonad; length::Int=10)

Construct 𝔠_p, the cofree comonad with given stream length.
"""
function c_cofree(p::FreeMonad; length::Int=10)
    stream = FreeMonad[]
    s = p.seed
    for i in 1:length
        s, _ = splitmix64(s)
        push!(stream, FreeMonad(p.polynomial, p.root, p.depth, s))
    end
    CofreeComonad(p.polynomial, p, stream, p.seed)
end

"""
    InternalHom

The internal hom [p, q] in Poly.
Tasks go forward (p), outcomes come back (q).
"""
struct InternalHom
    domain::Symbol      # Task type p
    codomain::Symbol    # Outcome type q
    forward::Function   # p → Set of positions of q
    backward::Function  # Position of p → Direction of q → Direction of p
    color::RGB{Float64}
end

"""
    internal_hom(p::Symbol, q::Symbol; seed::UInt64=GAY_SEED)

Construct [p, q] internal hom.
"""
function internal_hom(p::Symbol, q::Symbol; seed::UInt64=GAY_SEED)
    color = color_from_seed(seed ⊻ name_to_seed(String(p)) ⊻ name_to_seed(String(q)))
    
    forward = x -> Set([1, 2])  # Default: binary outcomes
    backward = (pos, dir) -> dir  # Identity backward pass
    
    InternalHom(p, q, forward, backward, color)
end

"""
    MonoidalSum

The monoidal product ∨ (coproduct in Poly).
Represents choice among agents.
"""
struct MonoidalSum
    components::Vector{FreeMonad}
    seed::UInt64
    color::RGB{Float64}
end

"""
    vee_sum(ps::Vector{FreeMonad})

Construct p₁ ∨ p₂ ∨ ... ∨ pₖ
"""
function vee_sum(ps::Vector{FreeMonad})
    seed = reduce(⊻, p.seed for p in ps; init=GAY_SEED)
    color = color_from_seed(seed)
    MonoidalSum(ps, seed, color)
end

"""
    TaskDelegation

The full Org structure: Org_𝔪^♯(p; q₁,...,qₖ) := 𝔠_{[p, 𝔪_{q₁∨...∨qₖ}]}
"""
struct TaskDelegation
    agent_task::Symbol                # p
    subordinate_tasks::Vector{Symbol} # q₁, ..., qₖ
    
    internal_hom::InternalHom
    free_monad::FreeMonad
    monoidal_sum::MonoidalSum
    cofree_comonad::CofreeComonad
    
    color::RGB{Float64}
    seed::UInt64
end

"""
    OrgMonadACSet

Full Org monad as an ACSet with chromatic identity.
"""
struct OrgMonadACSet
    acs::ChromaticACSet
    delegation::TaskDelegation
end

# ═══════════════════════════════════════════════════════════════════════════════
# Correspondences: GayACSet ↔ TileACSet
# ═══════════════════════════════════════════════════════════════════════════════

"""
    Equivalence

An equivalence between structures in GayACSet and TileACSet.
"""
struct Equivalence
    name::String
    gay_structure::Symbol
    tile_structure::Symbol
    forward::Function   # Gay → Tile
    backward::Function  # Tile → Gay
    coherence::Float64  # How well the correspondence preserves structure
end

"""
    Duality

A duality (contravariant correspondence).
"""
struct Duality
    name::String
    gay_structure::Symbol
    tile_structure::Symbol
    swap::Function      # Swaps direction
    coherence::Float64
end

"""
    Discrepancy

A discrepancy where structures don't correspond.
"""
struct Discrepancy
    name::String
    gay_structure::Symbol
    tile_structure::Symbol
    description::String
    severity::Float64  # 0-1, how significant
end

"""
    GayTileCorrespondence

Full correspondence table between GayACSet and TileACSet.
"""
struct GayTileCorrespondence
    equivalences::Vector{Equivalence}
    dualities::Vector{Duality}
    discrepancies::Vector{Discrepancy}
    superpositions::Dict{Symbol, Vector{Symbol}}  # Gay symbol → overlapping Tile symbols
end

function find_equivalences()
    [
        Equivalence(
            "Object ↔ Cell",
            :V, :T,
            v -> v,  # Identity (both are 0-cells)
            t -> t,
            1.0
        ),
        Equivalence(
            "Morphism ↔ Adjacency",
            :E, :A,
            e -> e,  # Both represent connections
            a -> a,
            0.95  # Slight difference: edges directed, adjacencies symmetric
        ),
        Equivalence(
            "SeedColor ↔ PositionColor",
            :seed_color, :position_color,
            (id, seed) -> color_from_seed(seed ⊻ UInt64(id)),
            (pos, seed) -> color_from_seed(seed ⊻ UInt64(round(Int, pos[1] * 1000 + pos[2]))),
            0.99
        ),
    ]
end

function find_dualities()
    [
        Duality(
            "FreeMonad ↔ Substitution",
            :m_free, :substitution,
            m -> "substitute",  # Free monad = branching; Substitution = refinement
            0.85
        ),
        Duality(
            "CofreeComonad ↔ Inflation",
            :c_cofree, :inflation,
            c -> "inflate",  # Comonad = stream of states; Inflation = hierarchical growth
            0.80
        ),
        Duality(
            "InternalHom ↔ Boundary",
            :internal_hom, :boundary_match,
            h -> "boundary",  # Tasks/outcomes ↔ edge matching
            0.75
        ),
    ]
end

function find_discrepancies()
    [
        Discrepancy(
            "Directionality",
            :E, :A,
            "Edges are directed (src → tgt), adjacencies are symmetric",
            0.3
        ),
        Discrepancy(
            "Spatial Embedding",
            :V, :T,
            "Tiles have explicit positions; vertices don't",
            0.5
        ),
        Discrepancy(
            "Periodicity",
            :graph_structure, :aperiodic_tiling,
            "Graphs can be regular/periodic; Penrose tilings are aperiodic by design",
            0.7
        ),
    ]
end

"""
    superposition(gay::ChromaticACSet, tile::ChromaticACSet)

Compute the superposition of GayACSet and TileACSet structures.
Returns coherence measure and mixed structure.
"""
function superposition(gay::ChromaticACSet, tile::ChromaticACSet)
    # XOR fingerprints for combined identity
    combined_fp = gay.fingerprint ⊻ tile.fingerprint
    combined_color = color_from_seed(combined_fp)
    
    # Count matching elements
    n_gay = sum(length(v) for v in values(gay.objects))
    n_tile = sum(length(v) for v in values(tile.objects))
    
    # Coherence based on fingerprint similarity
    # (Higher if fingerprints share bits)
    bit_overlap = count_ones(gay.fingerprint & tile.fingerprint)
    coherence = bit_overlap / 64.0
    
    (
        fingerprint = combined_fp,
        color = combined_color,
        gay_size = n_gay,
        tile_size = n_tile,
        coherence = coherence
    )
end

"""
    measure_coherence(correspondence::GayTileCorrespondence)

Measure overall coherence of the correspondence.
"""
function measure_coherence(c::GayTileCorrespondence)
    eq_coherence = isempty(c.equivalences) ? 0.0 : 
                   sum(e.coherence for e in c.equivalences) / length(c.equivalences)
    dual_coherence = isempty(c.dualities) ? 0.0 :
                     sum(d.coherence for d in c.dualities) / length(c.dualities)
    disc_severity = isempty(c.discrepancies) ? 0.0 :
                    sum(d.severity for d in c.discrepancies) / length(c.discrepancies)
    
    # Higher equivalence/duality coherence is good; lower discrepancy severity is good
    overall = (eq_coherence + dual_coherence + (1 - disc_severity)) / 3
    
    (
        equivalence_coherence = eq_coherence,
        duality_coherence = dual_coherence,
        discrepancy_severity = disc_severity,
        overall = overall
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# Huffman Coding as Org Monad Example
# ═══════════════════════════════════════════════════════════════════════════════

"""
    HuffmanTree

A Huffman tree node with chromatic identity.
Each internal node is a "tie-breaker" (like y² → 𝔪_{y²∨y²∨y²})
"""
abstract type HuffmanNode end

struct HuffmanLeaf <: HuffmanNode
    symbol::Char
    frequency::Int
    color::RGB{Float64}
end

struct HuffmanBranch <: HuffmanNode
    left::HuffmanNode
    right::HuffmanNode
    frequency::Int
    color::RGB{Float64}
end

struct HuffmanTree
    root::HuffmanNode
    codes::Dict{Char, String}
    seed::UInt64
end

"""
    huffman_org(symbols::Vector{Char}, frequencies::Vector{Int}; seed::UInt64=GAY_SEED)

Build a Huffman tree as an Org monad structure.
Each internal node represents a delegation decision.
"""
function huffman_org(symbols::Vector{Char}, frequencies::Vector{Int}; 
                     seed::UInt64=GAY_SEED)
    @assert length(symbols) == length(frequencies)
    
    # Create leaves with chromatic identity
    leaves = HuffmanNode[]
    for (i, (sym, freq)) in enumerate(zip(symbols, frequencies))
        leaf_seed = seed ⊻ UInt64(i) ⊻ UInt64(sym)
        color = color_from_seed(leaf_seed)
        push!(leaves, HuffmanLeaf(sym, freq, color))
    end
    
    # Build tree with minimum frequency merge (Huffman algorithm)
    nodes = copy(leaves)
    node_id = length(leaves)
    
    while length(nodes) > 1
        # Sort by frequency
        sort!(nodes, by = n -> n.frequency)
        
        # Merge two smallest
        left = popfirst!(nodes)
        right = popfirst!(nodes)
        
        node_id += 1
        branch_seed = seed ⊻ UInt64(node_id * 1000)
        branch_color = color_from_seed(branch_seed)
        
        branch = HuffmanBranch(left, right, left.frequency + right.frequency, branch_color)
        push!(nodes, branch)
    end
    
    root = nodes[1]
    
    # Build codes
    codes = Dict{Char, String}()
    function build_codes(node::HuffmanNode, code::String)
        if node isa HuffmanLeaf
            codes[node.symbol] = code
        elseif node isa HuffmanBranch
            build_codes(node.left, code * "0")
            build_codes(node.right, code * "1")
        end
    end
    build_codes(root, "")
    
    HuffmanTree(root, codes, seed)
end

"""
    encode_bitstring(tree::HuffmanTree, message::String) -> String

Encode a message using Huffman codes.
"""
function encode_bitstring(tree::HuffmanTree, message::String)
    join(get(tree.codes, c, "") for c in message)
end

"""
    decode_bitstring(tree::HuffmanTree, bits::String) -> String

Decode a bitstring using Huffman tree.
This demonstrates tie-breaker pattern: at each node, choose left (0) or right (1).
"""
function decode_bitstring(tree::HuffmanTree, bits::String)
    result = Char[]
    node = tree.root
    
    for bit in bits
        if node isa HuffmanBranch
            node = bit == '0' ? node.left : node.right
        end
        
        if node isa HuffmanLeaf
            push!(result, node.symbol)
            node = tree.root
        end
    end
    
    String(result)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════════════════

function world_gay_acset()
    println()
    println("╔═══════════════════════════════════════════════════════════════════════════╗")
    println("║  GayACSet: Chromatic Attributed C-Sets with Org Monad Structure           ║")
    println("║  Org_𝔪^♯(p; q₁,...,qₖ) := 𝔠_{[p, 𝔪_{q₁∨...∨qₖ}]}                         ║")
    println("╚═══════════════════════════════════════════════════════════════════════════╝")
    println()
    
    # ─── GayACSet (Colored Graph) ───
    println("─── GayACSet: Colored Graph ───")
    gay = gay_acset(seed=GAY_SEED)
    
    v1 = add_vertex!(gay; name="Alice")
    v2 = add_vertex!(gay; name="Bob")
    v3 = add_vertex!(gay; name="Carol")
    e1 = add_edge!(gay, v1, v2)
    e2 = add_edge!(gay, v2, v3)
    e3 = add_edge!(gay, v3, v1)
    
    println("  Vertices: $v1, $v2, $v3")
    println("  Edges: $e1, $e2, $e3")
    println("  Fingerprint: 0x$(string(spi_fingerprint(gay), base=16))")
    
    for i in 1:3
        c = gay_color(gay, :V, i)
        println("    V$i color: RGB($(round(c.r, digits=2)), $(round(c.g, digits=2)), $(round(c.b, digits=2)))")
    end
    println()
    
    # ─── TileACSet (Aperiodic Tiling) ───
    println("─── TileACSet: Aperiodic Tiling ───")
    tile = tile_acset(seed=GAY_SEED)
    
    t1 = add_tile!(tile, (0.0, 0.0))
    t2 = add_tile!(tile, (1.0, 0.0))
    t3 = add_tile!(tile, (0.5, 0.866))
    a1 = add_adjacency!(tile, t1, t2)
    a2 = add_adjacency!(tile, t2, t3)
    a3 = add_adjacency!(tile, t3, t1)
    
    println("  Tiles: $t1, $t2, $t3")
    println("  Adjacencies: $a1, $a2, $a3")
    println("  Fingerprint: 0x$(string(spi_fingerprint(tile), base=16))")
    
    for i in 1:3
        c = tile_color(tile, i)
        bw = tile.objects[:T][i].data[:bandwidth]
        println("    T$i color: RGB($(round(c.r, digits=2)), $(round(c.g, digits=2)), $(round(c.b, digits=2))), bandwidth=$(round(bw, digits=2))")
    end
    println()
    
    # ─── Superposition ───
    println("─── Superposition: GayACSet ⊗ TileACSet ───")
    sup = superposition(gay, tile)
    println("  Combined fingerprint: 0x$(string(sup.fingerprint, base=16))")
    println("  Combined color: RGB($(round(sup.color.r, digits=2)), $(round(sup.color.g, digits=2)), $(round(sup.color.b, digits=2)))")
    println("  Gay size: $(sup.gay_size), Tile size: $(sup.tile_size)")
    println("  Coherence: $(round(sup.coherence, digits=4))")
    println()
    
    # ─── Org Monad Structure ───
    println("─── Org Monad: y² → 𝔪_{y²∨y²∨y²} ───")
    y2 = y_squared(seed=GAY_SEED)
    m_y2 = m_free(y2; depth=3)
    c_m_y2 = c_cofree(m_y2; length=5)
    
    println("  y²: polynomial = $(y2.polynomial)")
    println("  𝔪_y²: depth = $(m_y2.depth)")
    println("  𝔠_𝔪_y²: stream length = $(length(c_m_y2.tail) + 1)")
    
    # Three subagents (y² ∨ y² ∨ y²)
    agents = [y_squared(seed=GAY_SEED ⊻ UInt64(i)) for i in 1:3]
    vee = vee_sum(agents)
    println("  y² ∨ y² ∨ y²: $(length(vee.components)) agents")
    println("    Combined color: RGB($(round(vee.color.r, digits=2)), $(round(vee.color.g, digits=2)), $(round(vee.color.b, digits=2)))")
    println()
    
    # ─── Huffman as Org ───
    println("─── Huffman Coding as Org Monad ───")
    symbols = ['a', 'b', 'c', 'd', 'e', 'f']
    frequencies = [45, 13, 12, 16, 9, 5]
    tree = huffman_org(symbols, frequencies; seed=GAY_SEED)
    
    println("  Symbols: $(symbols)")
    println("  Frequencies: $(frequencies)")
    println("  Codes:")
    for (sym, code) in sort(collect(tree.codes), by=x->length(x[2]))
        println("    '$sym' → $code")
    end
    
    message = "abcdef"
    encoded = encode_bitstring(tree, message)
    decoded = decode_bitstring(tree, encoded)
    println("  Message: \"$message\"")
    println("  Encoded: $encoded")
    println("  Decoded: \"$decoded\"")
    println()
    
    (gay=gay, tile=tile, superposition=sup, huffman=tree)
end

function world_org_correspondence()
    println()
    println("═══════════════════════════════════════════════════════════════════════════")
    println("  CORRESPONDENCES: GayACSet ↔ TileACSet")
    println("═══════════════════════════════════════════════════════════════════════════")
    println()
    
    # Build correspondence
    equiv = find_equivalences()
    dual = find_dualities()
    disc = find_discrepancies()
    
    superpos = Dict{Symbol, Vector{Symbol}}(
        :V => [:T],                    # Vertex ↔ Tile
        :E => [:A],                    # Edge ↔ Adjacency
        :seed => [:position, :bandwidth],  # Seed ↔ Position+Bandwidth
        :m_free => [:substitution],    # Free monad ↔ Substitution
        :c_cofree => [:inflation],     # Cofree comonad ↔ Inflation
    )
    
    corr = GayTileCorrespondence(equiv, dual, disc, superpos)
    
    println("─── Equivalences (Covariant) ───")
    for e in corr.equivalences
        println("  $(e.name): $(e.gay_structure) ≃ $(e.tile_structure) [coherence=$(round(e.coherence, digits=2))]")
    end
    println()
    
    println("─── Dualities (Contravariant) ───")
    for d in corr.dualities
        println("  $(d.name): $(d.gay_structure) ⇆ $(d.tile_structure) [coherence=$(round(d.coherence, digits=2))]")
    end
    println()
    
    println("─── Discrepancies ───")
    for d in corr.discrepancies
        println("  $(d.name): $(d.gay_structure) ≠ $(d.tile_structure)")
        println("    $(d.description)")
        println("    [severity=$(round(d.severity, digits=2))]")
    end
    println()
    
    println("─── Superpositions ───")
    for (gay_sym, tile_syms) in corr.superpositions
        println("  $gay_sym ⊗ $(join(tile_syms, ", "))")
    end
    println()
    
    coherence = measure_coherence(corr)
    println("─── Overall Coherence ───")
    println("  Equivalence coherence: $(round(coherence.equivalence_coherence, digits=3))")
    println("  Duality coherence: $(round(coherence.duality_coherence, digits=3))")
    println("  Discrepancy severity: $(round(coherence.discrepancy_severity, digits=3))")
    println("  OVERALL: $(round(coherence.overall, digits=3))")
    
    return corr
end

end # module GayACSet
