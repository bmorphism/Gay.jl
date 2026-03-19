# TileACSet.jl: Aperiodic Tilings as Attributed C-Sets with Org Structure
# =========================================================================
#
# The dual perspective to GayACSet: spatial/geometric realization of
# the abstract Org monad delegation pattern.
#
# KEY INSIGHT: Tilings are "geometric proofs" of the Org structure.
# - Substitution rules ↔ Free monad 𝔪 (branching choices)
# - Inflation hierarchy ↔ Cofree comonad 𝔠 (temporal evolution)
# - Boundary matching ↔ Internal hom [-,-] (tasks forward, outcomes backward)
# - Tile superposition ↔ Monoidal ∨ (choice among configurations)
#
# CORRESPONDENCE TABLE (with GayACSet):
# ┌────────────────────────┬────────────────────────┬─────────────────────┐
# │ TileACSet              │ GayACSet               │ Org Monad           │
# ├────────────────────────┼────────────────────────┼─────────────────────┤
# │ Tile (cell)            │ Vertex (object)        │ Agent               │
# │ Adjacency (gluing)     │ Edge (morphism)        │ Delegation edge     │
# │ Position → Color       │ Seed → Color           │ Identity signature  │
# │ Substitution rules     │ Free monad 𝔪          │ Planning tree       │
# │ Inflation hierarchy    │ Cofree comonad 𝔠      │ Temporal stream     │
# │ Boundary matching      │ Internal hom [-,-]    │ Task/outcome flow   │
# │ Tile superposition     │ Monoidal ∨            │ Agent choice        │
# │ Spectral gap           │ SPI fingerprint       │ Coherence measure   │
# │ Aperiodicity           │ Derangeability        │ No fixed points     │
# └────────────────────────┴────────────────────────┴─────────────────────┘

module TileACSet

using SplittableRandoms: SplittableRandom, split
using Colors

export
    # Core Tile Types
    TileType, HatTile, SpectreTile, PenroseTile,
    TileInstance, TileLattice,
    
    # Substitution as Free Monad
    SubstitutionRule, SubstitutionTree, apply_substitution,
    substitution_to_free_monad, free_monad_to_substitution,
    
    # Inflation as Cofree Comonad  
    InflationLevel, InflationStream, inflate!, deflate!,
    inflation_to_cofree, cofree_to_inflation,
    
    # Boundary Matching as Internal Hom
    BoundaryType, EdgeColor, boundary_match, boundary_hom,
    tasks_forward, outcomes_backward,
    
    # Tile Superposition
    TileSuperposition, superpose_tiles, collapse_to_classical,
    
    # Spectral Properties
    TileGraph, spectral_gap, mixing_time, aperiodicity_measure,
    
    # Org Correspondence
    OrgTileCorrespondence, tile_to_agent, agent_to_tile,
    delegation_to_adjacency, adjacency_to_delegation,
    
    # Demo
    demo_tile_acset, demo_org_tiling

# ═══════════════════════════════════════════════════════════════════════════════
# Core PRNG (SPI compliant)
# ═══════════════════════════════════════════════════════════════════════════════

const GAY_SEED = UInt64(1069)
const GOLDEN_ANGLE = 137.50776405003785  # degrees

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

@inline function color_from_position(pos::Tuple{Float64, Float64}; seed::UInt64=GAY_SEED)
    # Hash position to seed
    x_bits = reinterpret(UInt64, pos[1])
    y_bits = reinterpret(UInt64, pos[2])
    combined = seed ⊻ x_bits ⊻ (y_bits << 32 | y_bits >> 32)
    color_from_seed(combined)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Tile Types: The Building Blocks
# ═══════════════════════════════════════════════════════════════════════════════

"""
    TileType

Abstract type for different monotile shapes.
"""
abstract type TileType end

"""
    HatTile <: TileType

The "hat" aperiodic monotile (Smith et al. 2023).
13 vertices, can tile the plane with reflected copies.
"""
struct HatTile <: TileType
    reflected::Bool
end

"""
    SpectreTile <: TileType

The "spectre" monotile that tiles without reflection.
14 vertices, strictly chiral aperiodic tiling.
"""
struct SpectreTile <: TileType
    orientation::Int  # 0-5 for 60° rotations
end

"""
    PenroseTile <: TileType

Classic Penrose tiles (kite and dart, or thick/thin rhombi).
"""
struct PenroseTile <: TileType
    variant::Symbol  # :kite, :dart, :thick, :thin
end

"""
    TileInstance

A concrete tile placed at a position with chromatic identity.
"""
struct TileInstance
    id::Int64
    tile_type::TileType
    position::Tuple{Float64, Float64}
    rotation::Float64  # radians
    
    # Chromatic identity from position (SPI)
    seed::UInt64
    color::RGB{Float64}
    
    # Boundary information for matching
    boundary_colors::Vector{RGB{Float64}}  # One per edge
    
    # Neighbors (by tile ID)
    neighbors::Vector{Int64}
end

function TileInstance(id::Int64, tile_type::TileType, 
                      position::Tuple{Float64, Float64}, rotation::Float64;
                      base_seed::UInt64=GAY_SEED)
    seed = base_seed ⊻ UInt64(id) ⊻ reinterpret(UInt64, position[1]) ⊻ 
           (reinterpret(UInt64, position[2]) >> 32)
    color = color_from_seed(seed)
    
    # Boundary colors: one per edge (varies by tile type)
    n_edges = tile_type isa HatTile ? 13 : tile_type isa SpectreTile ? 14 : 4
    boundary_colors = [color_from_seed(seed ⊻ UInt64(i * 7919)) for i in 1:n_edges]
    
    TileInstance(id, tile_type, position, rotation, seed, color, boundary_colors, Int64[])
end

"""
    TileLattice

A collection of tiles forming a (partial) tiling.
The analog of ChromaticACSet for tilings.
"""
mutable struct TileLattice
    tiles::Dict{Int64, TileInstance}
    adjacencies::Vector{Tuple{Int64, Int64, Int}}  # (tile1, tile2, edge_idx)
    seed::UInt64
    
    # Hierarchy levels (for inflation)
    level::Int
    parent_lattice::Union{Nothing, TileLattice}
    
    # Spectral properties
    fingerprint::UInt64
    spectral_gap::Float64
end

function TileLattice(; seed::UInt64=GAY_SEED, level::Int=0)
    TileLattice(
        Dict{Int64, TileInstance}(),
        Tuple{Int64, Int64, Int}[],
        seed,
        level,
        nothing,
        UInt64(0),
        0.0
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# Substitution Rules as Free Monad
# ═══════════════════════════════════════════════════════════════════════════════

"""
    SubstitutionRule

A rule for replacing a tile with a cluster of smaller tiles.
This is the geometric realization of the free monad branching.

For hat tile: 1 hat → 4 hats + 4 reflected hats (approx)
"""
struct SubstitutionRule
    input_type::TileType
    output_types::Vector{TileType}
    output_positions::Vector{Tuple{Float64, Float64}}  # Relative positions
    output_rotations::Vector{Float64}
    color::RGB{Float64}  # Rule identity
end

"""
    SubstitutionTree

A tree of substitution choices.
Isomorphic to the free monad 𝔪_T on tile type T.
"""
abstract type SubstitutionNode end

struct SubstitutionLeaf <: SubstitutionNode
    tile::TileInstance
    color::RGB{Float64}
end

struct SubstitutionBranch <: SubstitutionNode
    rule::SubstitutionRule
    children::Vector{SubstitutionNode}
    color::RGB{Float64}
end

struct SubstitutionTree
    root::SubstitutionNode
    depth::Int
    seed::UInt64
end

"""
    apply_substitution(tile::TileInstance, rule::SubstitutionRule; seed::UInt64)

Apply a substitution rule to a tile, producing children.
"""
function apply_substitution(tile::TileInstance, rule::SubstitutionRule; 
                            seed::UInt64=GAY_SEED)
    children = TileInstance[]
    
    for (i, (typ, rel_pos, rot)) in enumerate(zip(rule.output_types, 
                                                   rule.output_positions, 
                                                   rule.output_rotations))
        # Transform relative position by tile's position and rotation
        cos_θ, sin_θ = cos(tile.rotation), sin(tile.rotation)
        new_x = tile.position[1] + rel_pos[1] * cos_θ - rel_pos[2] * sin_θ
        new_y = tile.position[2] + rel_pos[1] * sin_θ + rel_pos[2] * cos_θ
        new_rot = tile.rotation + rot
        
        child = TileInstance(
            tile.id * 100 + i,  # Hierarchical ID
            typ,
            (new_x, new_y),
            new_rot;
            base_seed = seed ⊻ UInt64(i)
        )
        push!(children, child)
    end
    
    children
end

"""
    substitution_to_free_monad(tree::SubstitutionTree)

Convert substitution tree to free monad representation.
Shows the correspondence: Substitution ↔ 𝔪
"""
function substitution_to_free_monad(tree::SubstitutionTree)
    # The substitution tree IS the free monad
    # Each SubstitutionBranch = FreeBranch (choice point)
    # Each SubstitutionLeaf = FreeLeaf (outcome)
    (
        polynomial = :TileType,
        depth = tree.depth,
        seed = tree.seed,
        structure = "SubstitutionTree ≃ FreeMonad"
    )
end

"""
    free_monad_to_substitution(fm; tile_type::TileType=HatTile(false))

Convert free monad to substitution interpretation.
"""
function free_monad_to_substitution(fm; tile_type::TileType=HatTile(false), seed::UInt64=GAY_SEED)
    # Interpret free monad as substitution tree
    function build_tree(depth::Int, s::UInt64)
        color = color_from_seed(s)
        if depth == 0
            tile = TileInstance(1, tile_type, (0.0, 0.0), 0.0; base_seed=s)
            SubstitutionLeaf(tile, color)
        else
            # Create substitution rule
            n_children = 4  # Typical for hat tile
            rule = SubstitutionRule(
                tile_type,
                [tile_type for _ in 1:n_children],
                [(Float64(i) * 0.5, Float64(j) * 0.5) for i in 1:2, j in 1:2] |> vec,
                [Float64(k) * π/6 for k in 1:n_children],
                color
            )
            
            children = SubstitutionNode[]
            for i in 1:n_children
                s_new, _ = splitmix64(s ⊻ UInt64(i))
                push!(children, build_tree(depth - 1, s_new))
            end
            
            SubstitutionBranch(rule, children, color)
        end
    end
    
    SubstitutionTree(build_tree(fm.depth, seed), fm.depth, seed)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Inflation as Cofree Comonad
# ═══════════════════════════════════════════════════════════════════════════════

"""
    InflationLevel

One level in the inflation hierarchy.
Each level is a "snapshot" of the tiling at a particular scale.
"""
struct InflationLevel
    level::Int
    lattice::TileLattice
    scale::Float64  # Relative to base
    color::RGB{Float64}
end

"""
    InflationStream

A stream of inflation levels.
Isomorphic to the cofree comonad 𝔠_T.
"""
struct InflationStream
    head::InflationLevel
    tail::Vector{InflationLevel}
    seed::UInt64
end

"""
    inflate!(lattice::TileLattice)

Apply inflation (scale up + add detail).
Corresponds to comonad extract + duplicate.
"""
function inflate!(lattice::TileLattice)
    new_lattice = TileLattice(seed=lattice.seed, level=lattice.level + 1)
    new_lattice.parent_lattice = lattice
    
    # Scale factor (typically golden ratio for Penrose)
    φ = 1.618033988749895
    
    for (id, tile) in lattice.tiles
        # Scale up position
        new_pos = (tile.position[1] * φ, tile.position[2] * φ)
        new_tile = TileInstance(
            id,
            tile.tile_type,
            new_pos,
            tile.rotation;
            base_seed = lattice.seed ⊻ UInt64(lattice.level + 1)
        )
        new_lattice.tiles[id] = new_tile
        new_lattice.fingerprint ⊻= new_tile.seed
    end
    
    # Copy adjacencies
    append!(new_lattice.adjacencies, lattice.adjacencies)
    
    new_lattice
end

"""
    deflate!(lattice::TileLattice)

Apply deflation (scale down + merge).
Corresponds to comonad cobind.
"""
function deflate!(lattice::TileLattice)
    lattice.parent_lattice === nothing && error("Cannot deflate base level")
    lattice.parent_lattice
end

"""
    inflation_to_cofree(stream::InflationStream)

Show the correspondence: Inflation ↔ 𝔠
"""
function inflation_to_cofree(stream::InflationStream)
    (
        polynomial = :TileType,
        stream_length = 1 + length(stream.tail),
        seed = stream.seed,
        structure = "InflationStream ≃ CofreeComonad"
    )
end

"""
    cofree_to_inflation(cc; tile_type::TileType=HatTile(false))

Convert cofree comonad to inflation interpretation.
"""
function cofree_to_inflation(cc; tile_type::TileType=HatTile(false), seed::UInt64=GAY_SEED)
    levels = InflationLevel[]
    
    base_lattice = TileLattice(seed=seed, level=0)
    base_tile = TileInstance(1, tile_type, (0.0, 0.0), 0.0; base_seed=seed)
    base_lattice.tiles[1] = base_tile
    base_lattice.fingerprint = base_tile.seed
    
    head = InflationLevel(0, base_lattice, 1.0, color_from_seed(seed))
    
    current = base_lattice
    for i in 1:cc.stream_length
        current = inflate!(current)
        level_color = color_from_seed(seed ⊻ UInt64(i))
        push!(levels, InflationLevel(i, current, 1.618^i, level_color))
    end
    
    InflationStream(head, levels, seed)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Boundary Matching as Internal Hom
# ═══════════════════════════════════════════════════════════════════════════════

"""
    BoundaryType

The "type" of a tile boundary (edge).
Must match with adjacent tile for valid tiling.
"""
struct BoundaryType
    edge_index::Int
    direction::Symbol  # :in or :out
    color::RGB{Float64}
end

"""
    EdgeColor

The chromatic identity of an edge.
Used for matching: edges match iff colors match.
"""
struct EdgeColor
    primary::RGB{Float64}
    secondary::RGB{Float64}  # For asymmetric matching
end

"""
    boundary_match(edge1::EdgeColor, edge2::EdgeColor) -> Bool

Check if two edges can be glued together.
Corresponds to internal hom evaluation: [p, q](a) = Σ_{b ∈ q(a)} q[a,b]
"""
function boundary_match(edge1::EdgeColor, edge2::EdgeColor; tolerance::Float64=0.01)
    # Match if colors are complementary (one's primary = other's secondary)
    d1 = color_distance(edge1.primary, edge2.secondary)
    d2 = color_distance(edge1.secondary, edge2.primary)
    d1 < tolerance && d2 < tolerance
end

function color_distance(c1::RGB, c2::RGB)
    sqrt((c1.r - c2.r)^2 + (c1.g - c2.g)^2 + (c1.b - c2.b)^2)
end

"""
    boundary_hom(tile1::TileInstance, tile2::TileInstance)

Compute the "internal hom" between two tiles.
Tasks = edge types of tile1 that could connect to tile2
Outcomes = edge types of tile2 that could receive from tile1
"""
function boundary_hom(tile1::TileInstance, tile2::TileInstance)
    matching_edges = Tuple{Int, Int}[]
    
    for (i, c1) in enumerate(tile1.boundary_colors)
        for (j, c2) in enumerate(tile2.boundary_colors)
            # Create edge colors (primary/secondary based on direction)
            e1 = EdgeColor(c1, color_from_seed(tile1.seed ⊻ UInt64(i * 1000)))
            e2 = EdgeColor(color_from_seed(tile2.seed ⊻ UInt64(j * 1000)), c2)
            
            if boundary_match(e1, e2)
                push!(matching_edges, (i, j))
            end
        end
    end
    
    (
        domain = tile1.id,
        codomain = tile2.id,
        matches = matching_edges,
        structure = "BoundaryHom ≃ InternalHom"
    )
end

"""
    tasks_forward(tile::TileInstance) -> Vector{BoundaryType}

Get "tasks" that can be sent forward (outgoing edges).
"""
function tasks_forward(tile::TileInstance)
    [BoundaryType(i, :out, c) for (i, c) in enumerate(tile.boundary_colors)]
end

"""
    outcomes_backward(tile::TileInstance) -> Vector{BoundaryType}

Get "outcomes" that come backward (incoming edges).
"""
function outcomes_backward(tile::TileInstance)
    [BoundaryType(i, :in, c) for (i, c) in enumerate(tile.boundary_colors)]
end

# ═══════════════════════════════════════════════════════════════════════════════
# Tile Superposition (Monoidal ∨)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    TileSuperposition

A superposition of possible tile configurations.
Corresponds to monoidal sum ∨ in Poly.
"""
struct TileSuperposition
    configurations::Vector{TileLattice}
    amplitudes::Vector{Float64}  # Weight of each configuration
    seed::UInt64
    color::RGB{Float64}
end

"""
    superpose_tiles(configs::Vector{TileLattice})

Create a superposition of tile configurations.
"""
function superpose_tiles(configs::Vector{TileLattice})
    isempty(configs) && error("Cannot superpose empty configurations")
    
    seed = reduce(⊻, c.seed for c in configs)
    color = color_from_seed(seed)
    
    # Equal amplitudes initially
    amplitudes = fill(1.0 / length(configs), length(configs))
    
    TileSuperposition(configs, amplitudes, seed, color)
end

"""
    collapse_to_classical(sup::TileSuperposition; seed::UInt64=GAY_SEED) -> TileLattice

Collapse superposition to a single classical configuration.
Uses SPI for deterministic "measurement".
"""
function collapse_to_classical(sup::TileSuperposition; seed::UInt64=GAY_SEED)
    # Use seed to deterministically select configuration
    r, _ = splitmix64(seed ⊻ sup.seed)
    p = (r & 0xFFFF) / 65535.0
    
    # Sample from cumulative distribution
    cumsum = 0.0
    for (i, amp) in enumerate(sup.amplitudes)
        cumsum += amp
        if p < cumsum
            return sup.configurations[i]
        end
    end
    
    # Fallback
    sup.configurations[end]
end

# ═══════════════════════════════════════════════════════════════════════════════
# Spectral Properties: Gap and Mixing
# ═══════════════════════════════════════════════════════════════════════════════

"""
    TileGraph

The adjacency graph of a tiling.
Vertices = tiles, edges = adjacencies.
"""
struct TileGraph
    n_vertices::Int
    adjacency::Vector{Vector{Int64}}
    colors::Vector{RGB{Float64}}
end

function TileGraph(lattice::TileLattice)
    tiles = collect(values(lattice.tiles))
    n = length(tiles)
    id_to_idx = Dict(t.id => i for (i, t) in enumerate(tiles))
    
    adj = [Int64[] for _ in 1:n]
    for (t1, t2, _) in lattice.adjacencies
        if haskey(id_to_idx, t1) && haskey(id_to_idx, t2)
            i, j = id_to_idx[t1], id_to_idx[t2]
            push!(adj[i], j)
            push!(adj[j], i)
        end
    end
    
    colors = [t.color for t in tiles]
    
    TileGraph(n, adj, colors)
end

"""
    spectral_gap(g::TileGraph) -> Float64

Estimate the spectral gap of the tile graph.
Gap > 0 implies expander properties and fast mixing.
"""
function spectral_gap(g::TileGraph)
    g.n_vertices == 0 && return 0.0
    
    # Power iteration to estimate second eigenvalue
    n = g.n_vertices
    v = randn(n)
    v ./= sqrt(sum(v.^2))
    
    # Project out first eigenvector (all ones for regular graph)
    ones_vec = fill(1.0/sqrt(n), n)
    
    for _ in 1:100
        # Matrix-vector multiply (normalized Laplacian style)
        new_v = zeros(n)
        for i in 1:n
            deg = length(g.adjacency[i])
            deg == 0 && continue
            for j in g.adjacency[i]
                deg_j = length(g.adjacency[j])
                deg_j == 0 && continue
                new_v[i] += v[j] / sqrt(deg * deg_j)
            end
        end
        
        # Project out first eigenvector
        new_v .-= dot(new_v, ones_vec) .* ones_vec
        
        # Normalize
        norm = sqrt(sum(new_v.^2))
        norm < 1e-10 && break
        v = new_v ./ norm
    end
    
    # Compute Rayleigh quotient
    Av = zeros(n)
    for i in 1:n
        deg = length(g.adjacency[i])
        deg == 0 && continue
        for j in g.adjacency[i]
            deg_j = length(g.adjacency[j])
            deg_j == 0 && continue
            Av[i] += v[j] / sqrt(deg * deg_j)
        end
    end
    
    λ2 = dot(v, Av)
    1.0 - abs(λ2)  # Gap = 1 - |λ₂|
end

function dot(a, b)
    sum(a .* b)
end

"""
    mixing_time(gap::Float64, n::Int) -> Float64

Estimate mixing time from spectral gap.
t_mix ≈ log(n) / gap
"""
function mixing_time(gap::Float64, n::Int)
    gap ≤ 0 && return Inf
    log(n) / gap
end

"""
    aperiodicity_measure(lattice::TileLattice) -> Float64

Measure how "aperiodic" a tiling is.
Higher = more aperiodic (no repeating patterns).
"""
function aperiodicity_measure(lattice::TileLattice)
    n = length(lattice.tiles)
    n < 2 && return 1.0
    
    # Check for translation invariance violations
    tiles = collect(values(lattice.tiles))
    
    # Sample random translation vectors
    violations = 0
    tests = 0
    
    for i in 1:min(n, 20)
        for j in (i+1):min(n, 20)
            t1, t2 = tiles[i], tiles[j]
            
            # Translation vector
            dx = t2.position[1] - t1.position[1]
            dy = t2.position[2] - t1.position[2]
            
            # Check if this translation maps other tiles correctly
            matches = 0
            for t in tiles
                shifted = (t.position[1] + dx, t.position[2] + dy)
                # Look for tile at shifted position
                for t_other in tiles
                    if abs(t_other.position[1] - shifted[1]) < 0.1 &&
                       abs(t_other.position[2] - shifted[2]) < 0.1
                        matches += 1
                        break
                    end
                end
            end
            
            # If translation maps most tiles, that's periodic
            if matches > n * 0.9
                violations += 1
            end
            tests += 1
        end
    end
    
    tests == 0 && return 1.0
    1.0 - violations / tests
end

# ═══════════════════════════════════════════════════════════════════════════════
# Org Correspondence Functions
# ═══════════════════════════════════════════════════════════════════════════════

"""
    OrgTileCorrespondence

Full correspondence between Org monad and tiling structures.
"""
struct OrgTileCorrespondence
    # Substitution ↔ FreeMonad
    substitution_tree::SubstitutionTree
    free_monad_depth::Int
    
    # Inflation ↔ CofreeComonad
    inflation_stream::InflationStream
    cofree_stream_length::Int
    
    # Boundary ↔ InternalHom
    boundary_matches::Vector{Tuple{Int64, Int64}}
    
    # Superposition ↔ MonoidalSum
    tile_superposition::Union{TileSuperposition, Nothing}
    
    seed::UInt64
end

"""
    tile_to_agent(tile::TileInstance)

Convert a tile to an "agent" in Org terms.
"""
function tile_to_agent(tile::TileInstance)
    (
        id = tile.id,
        task_type = typeof(tile.tile_type),
        color = tile.color,
        credibility = length(tile.neighbors) > 0 ? 0.9 : 0.5,
        structure = "Tile → Agent"
    )
end

"""
    agent_to_tile(agent; seed::UInt64=GAY_SEED)

Convert an "agent" to a tile.
"""
function agent_to_tile(agent; seed::UInt64=GAY_SEED)
    TileInstance(
        agent.id,
        HatTile(false),  # Default tile type
        (Float64(agent.id % 10), Float64(agent.id ÷ 10)),  # Grid position
        0.0;
        base_seed = seed ⊻ UInt64(agent.id)
    )
end

"""
    delegation_to_adjacency(delegation)

Convert Org delegation to tile adjacency.
"""
function delegation_to_adjacency(delegation)
    # Each subordinate call becomes an adjacency
    adjacencies = Tuple{Int64, Int64, Int}[]
    for (i, sub) in enumerate(delegation.subordinate_tasks)
        push!(adjacencies, (delegation.agent_task, sub, i))
    end
    adjacencies
end

"""
    adjacency_to_delegation(adjacencies::Vector, agent_id::Int64)

Convert tile adjacencies to Org delegation.
"""
function adjacency_to_delegation(adjacencies::Vector, agent_id::Int64)
    subordinates = [a[2] for a in adjacencies if a[1] == agent_id]
    (
        agent_task = agent_id,
        subordinate_tasks = subordinates,
        structure = "Adjacency → Delegation"
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════════════════

function demo_tile_acset()
    println()
    println("╔═══════════════════════════════════════════════════════════════════════════╗")
    println("║  TileACSet: Aperiodic Tilings as Attributed C-Sets                        ║")
    println("║  Geometric Realization of Org Monad: Substitution, Inflation, Boundary    ║")
    println("╚═══════════════════════════════════════════════════════════════════════════╝")
    println()
    
    # ─── Create Tile Lattice ───
    println("─── TileLattice: Hat Monotile Tiling ───")
    lattice = TileLattice(seed=GAY_SEED, level=0)
    
    # Add tiles in a cluster
    for i in 1:5
        for j in 1:5
            reflected = (i + j) % 2 == 0
            pos = (Float64(i) * 1.5, Float64(j) * 1.732)
            tile = TileInstance(
                i * 10 + j,
                HatTile(reflected),
                pos,
                (i + j) * π / 6;
                base_seed = lattice.seed
            )
            lattice.tiles[tile.id] = tile
            lattice.fingerprint ⊻= tile.seed
        end
    end
    
    # Add adjacencies
    for i in 1:5
        for j in 1:4
            push!(lattice.adjacencies, (i * 10 + j, i * 10 + j + 1, 1))
        end
    end
    for i in 1:4
        for j in 1:5
            push!(lattice.adjacencies, (i * 10 + j, (i + 1) * 10 + j, 2))
        end
    end
    
    println("  Tiles: $(length(lattice.tiles))")
    println("  Adjacencies: $(length(lattice.adjacencies))")
    println("  Fingerprint: 0x$(string(lattice.fingerprint, base=16))")
    println("  Level: $(lattice.level)")
    
    # Show some tile colors
    for id in [11, 22, 33]
        if haskey(lattice.tiles, id)
            t = lattice.tiles[id]
            c = t.color
            println("    Tile $id: pos=$(round.(t.position, digits=2)), color=RGB($(round(c.r, digits=2)), $(round(c.g, digits=2)), $(round(c.b, digits=2)))")
        end
    end
    println()
    
    # ─── Substitution as Free Monad ───
    println("─── Substitution ↔ Free Monad 𝔪 ───")
    
    base_tile = lattice.tiles[11]
    rule = SubstitutionRule(
        HatTile(false),
        [HatTile(false), HatTile(true), HatTile(false), HatTile(true)],
        [(0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5)],
        [0.0, π/3, 2π/3, π],
        color_from_seed(GAY_SEED)
    )
    
    children = apply_substitution(base_tile, rule; seed=GAY_SEED)
    println("  Substitution: 1 hat → $(length(children)) children")
    for (i, child) in enumerate(children)
        println("    Child $i: pos=$(round.(child.position, digits=2)), reflected=$(child.tile_type.reflected)")
    end
    
    sub_tree = free_monad_to_substitution((depth=2,); tile_type=HatTile(false), seed=GAY_SEED)
    println("  SubstitutionTree depth: $(sub_tree.depth)")
    fm_info = substitution_to_free_monad(sub_tree)
    println("  Correspondence: $(fm_info.structure)")
    println()
    
    # ─── Inflation as Cofree Comonad ───
    println("─── Inflation ↔ Cofree Comonad 𝔠 ───")
    
    inflated = inflate!(lattice)
    println("  Original level: $(lattice.level)")
    println("  Inflated level: $(inflated.level)")
    println("  Inflated tiles: $(length(inflated.tiles))")
    
    # Create inflation stream
    stream = cofree_to_inflation((stream_length=3,); tile_type=HatTile(false), seed=GAY_SEED)
    println("  InflationStream: $(1 + length(stream.tail)) levels")
    cc_info = inflation_to_cofree(stream)
    println("  Correspondence: $(cc_info.structure)")
    println()
    
    # ─── Boundary Matching as Internal Hom ───
    println("─── Boundary Matching ↔ Internal Hom [-,-] ───")
    
    if length(lattice.tiles) >= 2
        t1 = lattice.tiles[11]
        t2 = lattice.tiles[12]
        hom = boundary_hom(t1, t2)
        println("  Tile $(hom.domain) → Tile $(hom.codomain)")
        println("  Matching edges: $(length(hom.matches))")
        println("  Correspondence: $(hom.structure)")
        
        tasks = tasks_forward(t1)
        outcomes = outcomes_backward(t2)
        println("  Tasks forward: $(length(tasks)) edges")
        println("  Outcomes backward: $(length(outcomes)) edges")
    end
    println()
    
    # ─── Tile Superposition as Monoidal ∨ ───
    println("─── Tile Superposition ↔ Monoidal ∨ ───")
    
    config1 = TileLattice(seed=GAY_SEED, level=0)
    config2 = TileLattice(seed=GAY_SEED ⊻ UInt64(42), level=0)
    
    # Add different tiles to each config
    config1.tiles[1] = TileInstance(1, HatTile(false), (0.0, 0.0), 0.0; base_seed=config1.seed)
    config2.tiles[1] = TileInstance(1, HatTile(true), (0.5, 0.5), π/6; base_seed=config2.seed)
    
    sup = superpose_tiles([config1, config2])
    println("  Superposition of $(length(sup.configurations)) configurations")
    println("  Amplitudes: $(round.(sup.amplitudes, digits=3))")
    println("  Superposition color: RGB($(round(sup.color.r, digits=2)), $(round(sup.color.g, digits=2)), $(round(sup.color.b, digits=2)))")
    
    collapsed = collapse_to_classical(sup)
    println("  Collapsed to configuration with seed: 0x$(string(collapsed.seed, base=16))")
    println()
    
    # ─── Spectral Properties ───
    println("─── Spectral Properties ───")
    
    graph = TileGraph(lattice)
    gap = spectral_gap(graph)
    mix_time = mixing_time(gap, graph.n_vertices)
    aperiod = aperiodicity_measure(lattice)
    
    println("  Graph: $(graph.n_vertices) vertices")
    println("  Spectral gap: $(round(gap, digits=4))")
    println("  Mixing time: $(round(mix_time, digits=2))")
    println("  Aperiodicity: $(round(aperiod, digits=4))")
    println()
    
    (lattice=lattice, inflated=inflated, stream=stream, superposition=sup, graph=graph)
end

function demo_org_tiling()
    println()
    println("═══════════════════════════════════════════════════════════════════════════")
    println("  ORG MONAD ↔ TILING CORRESPONDENCE")
    println("═══════════════════════════════════════════════════════════════════════════")
    println()
    
    println("─── The Correspondences ───")
    println()
    println("  ┌────────────────────────┬────────────────────────┬─────────────────────┐")
    println("  │ TileACSet              │ GayACSet               │ Org Monad           │")
    println("  ├────────────────────────┼────────────────────────┼─────────────────────┤")
    println("  │ Tile (cell)            │ Vertex (object)        │ Agent               │")
    println("  │ Adjacency (gluing)     │ Edge (morphism)        │ Delegation edge     │")
    println("  │ Position → Color       │ Seed → Color           │ Identity signature  │")
    println("  │ Substitution rules     │ Free monad 𝔪          │ Planning tree       │")
    println("  │ Inflation hierarchy    │ Cofree comonad 𝔠      │ Temporal stream     │")
    println("  │ Boundary matching      │ Internal hom [-,-]    │ Task/outcome flow   │")
    println("  │ Tile superposition     │ Monoidal ∨            │ Agent choice        │")
    println("  │ Spectral gap           │ SPI fingerprint       │ Coherence measure   │")
    println("  │ Aperiodicity           │ Derangeability        │ No fixed points     │")
    println("  └────────────────────────┴────────────────────────┴─────────────────────┘")
    println()
    
    println("─── Key Insight ───")
    println("  Tilings are GEOMETRIC PROOFS of the Org monad structure.")
    println("  The aperiodic hat monotile demonstrates y² → 𝔪_{y²∨y²∨y²}:")
    println("    • Each tile = binary choice (hat or reflected hat)")
    println("    • Substitution = delegation to 4+ sub-tiles")
    println("    • Boundary matching = tie-breaker resolution")
    println("    • Aperiodicity = no fixed points (derangeable)")
    println()
    
    println("─── Mathematical Structure ───")
    println("  Org_𝔪^♯(p; q₁,...,qₖ) := 𝔠_{[p, 𝔪_{q₁∨...∨qₖ}]}")
    println()
    println("  In tiling terms:")
    println("    p          = tile type (hat)")
    println("    q₁,...,qₖ  = sub-tile types after substitution")
    println("    𝔪          = substitution tree (planning)")
    println("    ∨          = superposition of configurations")
    println("    [-,-]      = boundary matching (task↔outcome)")
    println("    𝔠          = inflation stream (temporal)")
    println()
    
    return nothing
end

end # module TileACSet
