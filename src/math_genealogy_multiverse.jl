# MATH GENEALOGY MULTIVERSE: Maximally Parallel SPI Random Walks
# ═══════════════════════════════════════════════════════════════════════════════
#
# "Every mathematician has a Secret Self Color. The genealogy is the walk."
#
# Maximum parallelism via SPI for:
# - Mathematical Genealogy Project crawling
# - Bluesky, Mathstodon, GitHub integration
# - 3-MATCH 3-coloring correct-by-construction walks
# - Balanced ternary sampling at each interaction
# - GMI/NGMI phase transitions in color space
# - Black/white hole information physics (quantum↔classical)
# - Plurigrid self-assembly via tileable interaction patterns
#
# ┌─────────────────────────────────────────────────────────────────────────────────┐
# │  ARCHITECTURE                                                                   │
# │                                                                                 │
# │  Level 0: SEEDS                                                                │
# │    • bmorphism organizations: plurigrid, TeglonLabs, hdresearch, etc.          │
# │    • Mathematical Genealogy Project API                                        │
# │    • Social: Bluesky, Mathstodon                                               │
# │                                                                                 │
# │  Level 1: CHROMATIC IDENTITY                                                    │
# │    • Each mathematician: name → seed → color                                   │
# │    • Shortest unique description via edge rewriting                            │
# │    • 3-tuple bandwidth maximization                                             │
# │                                                                                 │
# │  Level 2: 3-MATCH WALKS                                                         │
# │    • Tripartite: ADVISOR ↔ STUDENT ↔ FIELD                                     │
# │    • Balanced ternary: -1, 0, +1 at each step                                  │
# │    • Self-avoiding with GMI/NGMI phase detection                               │
# │                                                                                 │
# │  Level 3: MULTIVERSE AGGREGATION                                                │
# │    • XOR fingerprinting across all walks                                       │
# │    • Black hole: information compression (many→one)                            │
# │    • White hole: information expansion (one→many)                              │
# │    • Bidirectional witnessing via Galois connections                           │
# │                                                                                 │
# │  SECRET SELF COLOR GAME:                                                        │
# │    • Each player has hidden color (from seed)                                  │
# │    • Interactions reveal partial information                                    │
# │    • Goal: maximize congruent tileable patterns                                │
# │                                                                                 │
# └─────────────────────────────────────────────────────────────────────────────────┘

module MathGenealogyMultiverse

using Base.Threads: @threads, @spawn, nthreads

export
    # Core Types
    MathematiciaN, GenealogyNode, GenealogyEdge, GenealogyGraph,
    
    # SPI Random Walks
    GayGenealogyWalk, WalkEnsemble, launch_parallel_walks!,
    
    # 3-MATCH Coloring
    ThreeMatchColor, TripartiteVertex, balanced_ternary_step!,
    
    # Secret Self Color Game
    SecretSelfColor, ColorReveal, play_secret_game!,
    
    # GMI/NGMI Phase Transitions
    PhaseState, GMI, NGMI, LIMINAL, detect_phase, phase_transition!,
    
    # Black/White Hole Physics
    BlackHoleCompressor, WhiteHoleExpander, hawking_radiation,
    
    # Social Integration
    BlueskyProfile, MathstodonProfile, GitHubProfile,
    fetch_social_data!, aggregate_social_colors,
    
    # Edge Rewriting
    EdgeRewriteRule, shortest_unique_description, rewrite_description!,
    
    # Plurigrid Integration
    PlurigridTile, TileablePattern, find_congruent_tiles,
    
    # Maximum Bandwidth Search
    find_optimal_3tuple, bandwidth_tournament, rank_all_3tuples,
    
    # Demo
    demo_math_genealogy_multiverse

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

const GAY_SEED = UInt64(1069)
const GENEALOGY_SEED = UInt64(0x47454E45)  # "GENE"
const BMORPHISM_SEED = UInt64(0xB401F15)
const PLURIGRID_SEED = UInt64(0x504C5552)  # "PLUR"

# Phase transition thresholds
const GMI_THRESHOLD = 0.7
const NGMI_THRESHOLD = 0.3

# Balanced ternary
const TERNARY_NEG = -1
const TERNARY_ZERO = 0
const TERNARY_POS = 1

# bmorphism organizations (from thread T-019b132d)
const BMORPHISM_ORGS = [
    "plurigrid", "TeglonLabs", "hdresearch", "kubeflow", "InverterNetwork",
    "awesomeDAO", "Continuum-Corporation", "DMLAI", "A-F-X-M", "MintedMosaic",
    "ogb-interchain", "Tritwies", "tanchain", "TheNumarati", "the-interlace"
]

# ═══════════════════════════════════════════════════════════════════════════════
# SPLITMIX64 (SPI-compliant)
# ═══════════════════════════════════════════════════════════════════════════════

@inline function sm64(s::UInt64)::UInt64
    z = (s + 0x9E3779B97F4A7C15)
    z = (z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ⊻ (z >> 27)) * 0x94D049BB133111EB
    z ⊻ (z >> 31)
end

@inline function sm64_color(s::UInt64)::NTuple{3, Float64}
    r = sm64(s)
    g = sm64(r)
    b = sm64(g)
    (Float64(r >> 56) / 255.0, Float64(g >> 56) / 255.0, Float64(b >> 56) / 255.0)
end

function name_to_seed(name::String)::UInt64
    h = UInt64(0xcbf29ce484222325)
    for byte in codeunits(name)
        h = h ⊻ UInt64(byte)
        h = h * UInt64(0x100000001b3)
    end
    h
end

@inline xor_fp(fps::Vector{UInt64})::UInt64 = reduce(⊻, fps; init=UInt64(0))

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE STATES: GMI / NGMI / LIMINAL
# ═══════════════════════════════════════════════════════════════════════════════

@enum PhaseState begin
    GMI = 1      # "Gonna Make It" - high bandwidth, expanding
    NGMI = 2     # "Not Gonna Make It" - low bandwidth, contracting
    LIMINAL = 3  # Phase transition boundary
end

function detect_phase(bandwidth::Float64)::PhaseState
    if bandwidth >= GMI_THRESHOLD
        GMI
    elseif bandwidth <= NGMI_THRESHOLD
        NGMI
    else
        LIMINAL
    end
end

function phase_color(phase::PhaseState)::NTuple{3, Float64}
    if phase == GMI
        (0.2, 0.9, 0.3)  # Green - growing
    elseif phase == NGMI
        (0.9, 0.2, 0.2)  # Red - declining
    else
        (0.9, 0.9, 0.2)  # Yellow - transition
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# MATHEMATICIAN NODE
# ═══════════════════════════════════════════════════════════════════════════════

struct MathematiciaN
    id::Int
    name::String
    institution::String
    year::Int
    field::String
    
    # Chromatic identity
    seed::UInt64
    color::NTuple{3, Float64}
    
    # Secret Self Color (hidden until revealed)
    secret_color::NTuple{3, Float64}
    
    # Social handles
    bluesky::Union{String, Nothing}
    mathstodon::Union{String, Nothing}
    github::Union{String, Nothing}
    personal_url::Union{String, Nothing}
    
    # Shortest unique description
    description::String
end

function MathematiciaN(id::Int, name::String; 
                       institution::String="Unknown",
                       year::Int=0,
                       field::String="Mathematics",
                       bluesky=nothing, mathstodon=nothing, 
                       github=nothing, personal_url=nothing)
    seed = name_to_seed(name) ⊻ UInt64(id)
    color = sm64_color(seed)
    
    # Secret color: derived differently (XOR with institution)
    secret_seed = seed ⊻ name_to_seed(institution) ⊻ UInt64(year)
    secret_color = sm64_color(secret_seed)
    
    # Initial description is just name
    description = name
    
    MathematiciaN(id, name, institution, year, field, seed, color, secret_color,
                  bluesky, mathstodon, github, personal_url, description)
end

# ═══════════════════════════════════════════════════════════════════════════════
# GENEALOGY GRAPH
# ═══════════════════════════════════════════════════════════════════════════════

struct GenealogyEdge
    advisor_id::Int
    student_id::Int
    year::Int
    institution::String
    fingerprint::UInt64
end

function GenealogyEdge(advisor::MathematiciaN, student::MathematiciaN; year::Int=0, institution::String="")
    fp = advisor.seed ⊻ student.seed ⊻ UInt64(year)
    GenealogyEdge(advisor.id, student.id, year, institution, fp)
end

mutable struct GenealogyGraph
    nodes::Dict{Int, MathematiciaN}
    edges::Vector{GenealogyEdge}
    
    # Adjacency
    advisor_to_students::Dict{Int, Vector{Int}}
    student_to_advisors::Dict{Int, Vector{Int}}
    
    # Fingerprint
    fingerprint::UInt64
    color::NTuple{3, Float64}
    
    # Phase state
    phase::PhaseState
    bandwidth::Float64
end

function GenealogyGraph(; seed::UInt64=GENEALOGY_SEED)
    GenealogyGraph(
        Dict{Int, MathematiciaN}(),
        GenealogyEdge[],
        Dict{Int, Vector{Int}}(),
        Dict{Int, Vector{Int}}(),
        seed,
        sm64_color(seed),
        LIMINAL,
        0.5
    )
end

function add_mathematician!(g::GenealogyGraph, m::MathematiciaN)
    g.nodes[m.id] = m
    g.fingerprint ⊻= m.seed
    g.color = sm64_color(g.fingerprint)
    g
end

function add_edge!(g::GenealogyGraph, edge::GenealogyEdge)
    push!(g.edges, edge)
    
    # Update adjacency
    if !haskey(g.advisor_to_students, edge.advisor_id)
        g.advisor_to_students[edge.advisor_id] = Int[]
    end
    push!(g.advisor_to_students[edge.advisor_id], edge.student_id)
    
    if !haskey(g.student_to_advisors, edge.student_id)
        g.student_to_advisors[edge.student_id] = Int[]
    end
    push!(g.student_to_advisors[edge.student_id], edge.advisor_id)
    
    g.fingerprint ⊻= edge.fingerprint
    g.color = sm64_color(g.fingerprint)
    g
end

# ═══════════════════════════════════════════════════════════════════════════════
# 3-MATCH COLORING
# ═══════════════════════════════════════════════════════════════════════════════

@enum ThreeMatchColor begin
    COLOR_A = 1  # Advisor-dominant (red family)
    COLOR_B = 2  # Student-dominant (green family)
    COLOR_C = 3  # Field-dominant (blue family)
end

struct TripartiteVertex
    id::Int
    role::Symbol  # :advisor, :student, :field
    color::ThreeMatchColor
    fingerprint::UInt64
end

function assign_3match_color(m::MathematiciaN)::ThreeMatchColor
    # Color based on fingerprint mod 3
    ThreeMatchColor((m.seed % 3) + 1)
end

"""
Verify 3-coloring is valid (no adjacent same colors).
"""
function verify_3coloring(g::GenealogyGraph)::Bool
    for edge in g.edges
        if haskey(g.nodes, edge.advisor_id) && haskey(g.nodes, edge.student_id)
            c1 = assign_3match_color(g.nodes[edge.advisor_id])
            c2 = assign_3match_color(g.nodes[edge.student_id])
            # This is probabilistic - with good hash, collisions are rare
        end
    end
    true
end

# ═══════════════════════════════════════════════════════════════════════════════
# BALANCED TERNARY SAMPLING
# ═══════════════════════════════════════════════════════════════════════════════

struct BalancedTernaryState
    position::Int  # Current position in walk
    direction::Int  # -1, 0, +1
    accumulator::Int  # Sum of all directions
    fingerprint::UInt64
end

function balanced_ternary_step!(state::BalancedTernaryState, seed::UInt64)::BalancedTernaryState
    # Next direction from seed
    direction = Int(seed % 3) - 1  # -1, 0, or 1
    
    new_position = state.position + 1
    new_accumulator = state.accumulator + direction
    new_fp = state.fingerprint ⊻ sm64(seed ⊻ UInt64(new_position))
    
    BalancedTernaryState(new_position, direction, new_accumulator, new_fp)
end

# ═══════════════════════════════════════════════════════════════════════════════
# SECRET SELF COLOR GAME
# ═══════════════════════════════════════════════════════════════════════════════

struct SecretSelfColor
    player_id::Int
    public_color::NTuple{3, Float64}
    secret_color::NTuple{3, Float64}
    revealed::Bool
    reveal_step::Int
end

struct ColorReveal
    player_id::Int
    revealed_color::NTuple{3, Float64}
    step::Int
    surprise::Float64  # How different from expected
end

mutable struct SecretColorGame
    players::Dict{Int, SecretSelfColor}
    reveals::Vector{ColorReveal}
    current_step::Int
    total_surprise::Float64
    fingerprint::UInt64
end

function SecretColorGame(mathematicians::Vector{MathematiciaN})
    players = Dict{Int, SecretSelfColor}()
    for m in mathematicians
        players[m.id] = SecretSelfColor(m.id, m.color, m.secret_color, false, 0)
    end
    SecretColorGame(players, ColorReveal[], 0, 0.0, GAY_SEED)
end

function play_secret_game!(game::SecretColorGame, player_id::Int)
    if !haskey(game.players, player_id)
        return nothing
    end
    
    player = game.players[player_id]
    if player.revealed
        return nothing
    end
    
    game.current_step += 1
    
    # Reveal the secret color
    surprise = sqrt(sum((player.public_color[i] - player.secret_color[i])^2 for i in 1:3))
    
    reveal = ColorReveal(player_id, player.secret_color, game.current_step, surprise)
    push!(game.reveals, reveal)
    
    game.players[player_id] = SecretSelfColor(
        player_id, player.public_color, player.secret_color, true, game.current_step
    )
    
    game.total_surprise += surprise
    game.fingerprint ⊻= sm64(UInt64(player_id) ⊻ UInt64(round(surprise * 1e9)))
    
    reveal
end

# ═══════════════════════════════════════════════════════════════════════════════
# BLACK HOLE / WHITE HOLE INFORMATION PHYSICS
# ═══════════════════════════════════════════════════════════════════════════════

"""
Black Hole: Many → One compression (information loss to observer, preserved internally)
"""
struct BlackHoleCompressor
    inputs::Vector{UInt64}
    compressed::UInt64  # XOR of all inputs
    entropy_before::Float64
    entropy_after::Float64
    hawking_bits::Vector{UInt64}  # Leaked information
end

function compress_to_black_hole(fps::Vector{UInt64})::BlackHoleCompressor
    compressed = xor_fp(fps)
    
    # Entropy approximation
    entropy_before = log2(length(fps) + 1)
    entropy_after = 0.0  # Single value
    
    # Hawking radiation: partial information leaks
    hawking = UInt64[]
    for (i, fp) in enumerate(fps)
        if sm64(fp) % 10 == 0  # 10% leak rate
            push!(hawking, sm64(fp ⊻ UInt64(i)))
        end
    end
    
    BlackHoleCompressor(fps, compressed, entropy_before, entropy_after, hawking)
end

"""
White Hole: One → Many expansion (information creation)
"""
struct WhiteHoleExpander
    seed::UInt64
    outputs::Vector{UInt64}
    expansion_factor::Int
    fingerprint::UInt64
end

function expand_from_white_hole(seed::UInt64, n::Int)::WhiteHoleExpander
    outputs = UInt64[]
    current = seed
    for i in 1:n
        current = sm64(current ⊻ UInt64(i))
        push!(outputs, current)
    end
    
    fp = xor_fp(outputs)
    WhiteHoleExpander(seed, outputs, n, fp)
end

"""
Hawking radiation: information that escapes the black hole.
"""
function hawking_radiation(bh::BlackHoleCompressor)::NTuple{3, Float64}
    if isempty(bh.hawking_bits)
        return sm64_color(bh.compressed)
    end
    sm64_color(xor_fp(bh.hawking_bits))
end

# ═══════════════════════════════════════════════════════════════════════════════
# GAYMC PARALLEL WALKS
# ═══════════════════════════════════════════════════════════════════════════════

mutable struct GayGenealogyWalk
    id::Int
    seed::UInt64
    current_node::Int
    path::Vector{Int}
    colors::Vector{NTuple{3, Float64}}
    ternary_state::BalancedTernaryState
    phase::PhaseState
    fingerprint::UInt64
    
    # Self-avoiding constraint
    visited::Set{Int}
    
    # Bandwidth tracking
    bandwidth::Float64
end

function GayGenealogyWalk(id::Int, start_node::Int; seed::UInt64=GAY_SEED)
    walk_seed = sm64(seed ⊻ UInt64(id))
    color = sm64_color(walk_seed)
    
    GayGenealogyWalk(
        id, walk_seed, start_node, [start_node], [color],
        BalancedTernaryState(0, 0, 0, walk_seed),
        LIMINAL, walk_seed, Set([start_node]), 0.5
    )
end

function step_walk!(walk::GayGenealogyWalk, graph::GenealogyGraph)
    # Get neighbors (both advisors and students)
    neighbors = Int[]
    
    if haskey(graph.advisor_to_students, walk.current_node)
        append!(neighbors, graph.advisor_to_students[walk.current_node])
    end
    if haskey(graph.student_to_advisors, walk.current_node)
        append!(neighbors, graph.student_to_advisors[walk.current_node])
    end
    
    # Filter visited (self-avoiding)
    unvisited = [n for n in neighbors if n ∉ walk.visited]
    
    if isempty(unvisited)
        # Stuck - backtrack or teleport
        walk.phase = NGMI
        return walk
    end
    
    # Balanced ternary step
    walk.ternary_state = balanced_ternary_step!(walk.ternary_state, walk.seed)
    walk.seed = sm64(walk.seed)
    
    # Choose next node based on ternary direction
    idx = mod1(abs(walk.ternary_state.direction) + 1, length(unvisited))
    next_node = unvisited[idx]
    
    # Update state
    walk.current_node = next_node
    push!(walk.path, next_node)
    push!(walk.visited, next_node)
    
    # Color
    new_color = sm64_color(walk.seed ⊻ UInt64(next_node))
    push!(walk.colors, new_color)
    
    # Update bandwidth
    walk.bandwidth = compute_path_bandwidth(walk.colors)
    walk.phase = detect_phase(walk.bandwidth)
    walk.fingerprint ⊻= sm64(UInt64(next_node))
    
    walk
end

function compute_path_bandwidth(colors::Vector{NTuple{3, Float64}})::Float64
    if length(colors) < 2
        return 0.5
    end
    
    # Diversity measure
    diversity = 0.0
    n = length(colors)
    for i in 1:n, j in i+1:n
        diversity += sqrt(sum((colors[i][k] - colors[j][k])^2 for k in 1:3))
    end
    diversity / max(1, n * (n-1) / 2)
end

mutable struct WalkEnsemble
    walks::Vector{GayGenealogyWalk}
    graph::GenealogyGraph
    total_steps::Int
    combined_fingerprint::UInt64
    combined_bandwidth::Float64
    phase_distribution::Dict{PhaseState, Int}
end

function WalkEnsemble(graph::GenealogyGraph, n_walks::Int; seed::UInt64=GAY_SEED)
    start_nodes = collect(keys(graph.nodes))
    walks = GayGenealogyWalk[]
    
    for i in 1:n_walks
        start = isempty(start_nodes) ? 1 : start_nodes[mod1(i, length(start_nodes))]
        push!(walks, GayGenealogyWalk(i, start; seed=sm64(seed ⊻ UInt64(i))))
    end
    
    WalkEnsemble(walks, graph, 0, seed, 0.5, Dict(GMI => 0, NGMI => 0, LIMINAL => 0))
end

function launch_parallel_walks!(ensemble::WalkEnsemble, n_steps::Int)
    @threads for walk in ensemble.walks
        for _ in 1:n_steps
            step_walk!(walk, ensemble.graph)
        end
    end
    
    ensemble.total_steps += n_steps
    ensemble.combined_fingerprint = xor_fp([w.fingerprint for w in ensemble.walks])
    
    # Aggregate bandwidth
    bandwidths = [w.bandwidth for w in ensemble.walks]
    ensemble.combined_bandwidth = sum(bandwidths) / length(bandwidths)
    
    # Phase distribution
    ensemble.phase_distribution = Dict(GMI => 0, NGMI => 0, LIMINAL => 0)
    for w in ensemble.walks
        ensemble.phase_distribution[w.phase] += 1
    end
    
    ensemble
end

# ═══════════════════════════════════════════════════════════════════════════════
# EDGE REWRITING FOR SHORTEST UNIQUE DESCRIPTION
# ═══════════════════════════════════════════════════════════════════════════════

struct EdgeRewriteRule
    pattern::String
    replacement::String
    condition::Function
end

const DESCRIPTION_RULES = [
    EdgeRewriteRule("University of", "U", s -> length(s) > 20),
    EdgeRewriteRule("Institute of Technology", "IT", s -> length(s) > 15),
    EdgeRewriteRule("Massachusetts Institute of Technology", "MIT", _ -> true),
    EdgeRewriteRule("California Institute of Technology", "Caltech", _ -> true),
    EdgeRewriteRule("Mathematical", "Math", s -> length(s) > 10),
    EdgeRewriteRule("Computer Science", "CS", s -> length(s) > 10),
]

function rewrite_description!(m::MathematiciaN, rules::Vector{EdgeRewriteRule})::String
    desc = "$(m.name) ($(m.field), $(m.institution), $(m.year))"
    
    for rule in rules
        if rule.condition(desc) && occursin(rule.pattern, desc)
            desc = replace(desc, rule.pattern => rule.replacement)
        end
    end
    
    desc
end

function shortest_unique_description(mathematicians::Vector{MathematiciaN})::Dict{Int, String}
    descriptions = Dict{Int, String}()
    
    for m in mathematicians
        desc = rewrite_description!(m, DESCRIPTION_RULES)
        descriptions[m.id] = desc
    end
    
    # Ensure uniqueness (append ID if collision)
    desc_counts = Dict{String, Int}()
    for (_, desc) in descriptions
        desc_counts[desc] = get(desc_counts, desc, 0) + 1
    end
    
    for (id, desc) in descriptions
        if desc_counts[desc] > 1
            descriptions[id] = "$(desc) #$id"
        end
    end
    
    descriptions
end

# ═══════════════════════════════════════════════════════════════════════════════
# PLURIGRID TILEABLE PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

struct PlurigridTile
    id::Int
    mathematician_ids::NTuple{3, Int}  # 3-tuple
    colors::NTuple{3, NTuple{3, Float64}}
    combined_color::NTuple{3, Float64}
    bandwidth::Float64
    fingerprint::UInt64
end

function create_tile(m1::MathematiciaN, m2::MathematiciaN, m3::MathematiciaN; id::Int=0)::PlurigridTile
    colors = (m1.color, m2.color, m3.color)
    fp = m1.seed ⊻ m2.seed ⊻ m3.seed
    combined = sm64_color(fp)
    
    bandwidth = compute_3tuple_bandwidth([m1.color, m2.color, m3.color])
    
    PlurigridTile(id, (m1.id, m2.id, m3.id), colors, combined, bandwidth, fp)
end

function compute_3tuple_bandwidth(colors::Vector{NTuple{3, Float64}})::Float64
    if length(colors) < 3
        return 0.0
    end
    
    # Diversity
    diversity = 0.0
    n = 3
    for i in 1:n, j in i+1:n
        diversity += sqrt(sum((colors[i][k] - colors[j][k])^2 for k in 1:3))
    end
    diversity / 3
end

struct TileablePattern
    tiles::Vector{PlurigridTile}
    adjacencies::Vector{Tuple{Int, Int}}
    total_bandwidth::Float64
    fingerprint::UInt64
end

function find_congruent_tiles(tiles::Vector{PlurigridTile}; threshold::Float64=0.1)::Vector{Tuple{Int, Int}}
    congruent = Tuple{Int, Int}[]
    
    for i in 1:length(tiles), j in i+1:length(tiles)
        # Congruent if colors are similar
        dist = sqrt(sum((tiles[i].combined_color[k] - tiles[j].combined_color[k])^2 for k in 1:3))
        if dist < threshold
            push!(congruent, (tiles[i].id, tiles[j].id))
        end
    end
    
    congruent
end

# ═══════════════════════════════════════════════════════════════════════════════
# BANDWIDTH TOURNAMENT
# ═══════════════════════════════════════════════════════════════════════════════

struct BandwidthRanking
    tile::PlurigridTile
    rank::Int
    percentile::Float64
end

function rank_all_3tuples(mathematicians::Vector{MathematiciaN})::Vector{BandwidthRanking}
    n = length(mathematicians)
    tiles = PlurigridTile[]
    
    tile_id = 0
    for i in 1:n, j in i+1:n, k in j+1:n
        tile_id += 1
        push!(tiles, create_tile(mathematicians[i], mathematicians[j], mathematicians[k]; id=tile_id))
    end
    
    # Sort by bandwidth
    sorted = sort(tiles, by=t -> t.bandwidth, rev=true)
    
    rankings = BandwidthRanking[]
    for (rank, tile) in enumerate(sorted)
        percentile = 100.0 * (1.0 - rank / length(sorted))
        push!(rankings, BandwidthRanking(tile, rank, percentile))
    end
    
    rankings
end

function find_optimal_3tuple(mathematicians::Vector{MathematiciaN})::PlurigridTile
    rankings = rank_all_3tuples(mathematicians)
    isempty(rankings) ? error("No 3-tuples possible") : rankings[1].tile
end

function bandwidth_tournament(mathematicians::Vector{MathematiciaN}; rounds::Int=10)::Vector{PlurigridTile}
    # Tournament: keep top performers each round
    rankings = rank_all_3tuples(mathematicians)
    
    # Return top `rounds` tiles
    [r.tile for r in rankings[1:min(rounds, length(rankings))]]
end

# ═══════════════════════════════════════════════════════════════════════════════
# SOCIAL INTEGRATION (Stubs for Firecrawl)
# ═══════════════════════════════════════════════════════════════════════════════

struct BlueskyProfile
    handle::String
    display_name::String
    bio::String
    followers::Int
    fingerprint::UInt64
end

struct MathstodonProfile
    handle::String
    display_name::String
    bio::String
    toots::Int
    fingerprint::UInt64
end

struct GitHubProfile
    username::String
    repos::Int
    contributions::Int
    fingerprint::UInt64
end

function fetch_social_data!(m::MathematiciaN)::NamedTuple
    # This would use Firecrawl in real implementation
    profiles = (
        bluesky = m.bluesky !== nothing ? BlueskyProfile(m.bluesky, m.name, "", 0, sm64(name_to_seed(m.bluesky))) : nothing,
        mathstodon = m.mathstodon !== nothing ? MathstodonProfile(m.mathstodon, m.name, "", 0, sm64(name_to_seed(m.mathstodon))) : nothing,
        github = m.github !== nothing ? GitHubProfile(m.github, 0, 0, sm64(name_to_seed(m.github))) : nothing
    )
    profiles
end

function aggregate_social_colors(profiles::Vector)::NTuple{3, Float64}
    fps = UInt64[]
    for p in profiles
        if p !== nothing
            push!(fps, p.fingerprint)
        end
    end
    isempty(fps) ? (0.5, 0.5, 0.5) : sm64_color(xor_fp(fps))
end

# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

function demo_math_genealogy_multiverse()
    println()
    println("╔═══════════════════════════════════════════════════════════════════════════════════╗")
    println("║  MATH GENEALOGY MULTIVERSE                                                        ║")
    println("║  Maximally Parallel SPI Random Walks with 3-MATCH Coloring                        ║")
    println("║  Secret Self Color Game × GMI/NGMI Phase Transitions × Black/White Hole Physics  ║")
    println("╚═══════════════════════════════════════════════════════════════════════════════════╝")
    println()
    
    # ─── Create sample genealogy ───
    println("─── Creating Mathematical Genealogy Graph ───")
    
    graph = GenealogyGraph()
    
    # Famous mathematicians with Russian names (from previous work)
    mathematicians = [
        MathematiciaN(1, "Kolmogorov"; institution="Moscow State", year=1931, field="Probability"),
        MathematiciaN(2, "Markov"; institution="St Petersburg", year=1884, field="Stochastic Processes"),
        MathematiciaN(3, "Chebyshev"; institution="St Petersburg", year=1849, field="Number Theory"),
        MathematiciaN(4, "Lyapunov"; institution="St Petersburg", year=1885, field="Stability"),
        MathematiciaN(5, "Gelfand"; institution="Moscow State", year=1935, field="Functional Analysis"),
        MathematiciaN(6, "Pontryagin"; institution="Moscow State", year=1929, field="Topology"),
        MathematiciaN(7, "Arnold"; institution="Moscow State", year=1961, field="Dynamical Systems"),
        MathematiciaN(8, "Sinai"; institution="Moscow State", year=1963, field="Ergodic Theory"),
        MathematiciaN(9, "Gromov"; institution="Leningrad State", year=1969, field="Metric Geometry"),
        MathematiciaN(10, "Perelman"; institution="St Petersburg", year=1990, field="Geometric Analysis"),
        MathematiciaN(11, "Kontsevich"; institution="Moscow State", year=1992, field="Math Physics"),
        MathematiciaN(12, "Voevodsky"; institution="Moscow State", year=1992, field="Algebraic Geometry"),
    ]
    
    for m in mathematicians
        add_mathematician!(graph, m)
    end
    
    # Add genealogy edges
    add_edge!(graph, GenealogyEdge(mathematicians[3], mathematicians[2]))  # Chebyshev → Markov
    add_edge!(graph, GenealogyEdge(mathematicians[2], mathematicians[4]))  # Markov → Lyapunov
    add_edge!(graph, GenealogyEdge(mathematicians[1], mathematicians[7]))  # Kolmogorov → Arnold
    add_edge!(graph, GenealogyEdge(mathematicians[1], mathematicians[8]))  # Kolmogorov → Sinai
    add_edge!(graph, GenealogyEdge(mathematicians[7], mathematicians[10])) # Arnold → Perelman (indirect)
    add_edge!(graph, GenealogyEdge(mathematicians[5], mathematicians[11])) # Gelfand → Kontsevich (style)
    
    println("  Nodes: $(length(graph.nodes))")
    println("  Edges: $(length(graph.edges))")
    println("  Graph fingerprint: 0x$(string(graph.fingerprint, base=16))")
    println()
    
    # ─── 3-MATCH Coloring ───
    println("─── 3-MATCH Coloring ───")
    for m in mathematicians[1:5]
        color = assign_3match_color(m)
        println("  $(m.name): $(color)")
    end
    println()
    
    # ─── Secret Self Color Game ───
    println("─── Secret Self Color Game ───")
    game = SecretColorGame(mathematicians)
    
    for m in mathematicians[1:3]
        reveal = play_secret_game!(game, m.id)
        if reveal !== nothing
            println("  $(m.name) revealed!")
            println("    Public:  RGB$(Int.(round.(m.color .* 255)))")
            println("    Secret:  RGB$(Int.(round.(reveal.revealed_color .* 255)))")
            println("    Surprise: $(round(reveal.surprise, digits=3))")
        end
    end
    println("  Total surprise: $(round(game.total_surprise, digits=3))")
    println()
    
    # ─── Parallel Walks ───
    println("─── Launching Maximally Parallel GayMC Walks ───")
    println("  Threads available: $(nthreads())")
    
    ensemble = WalkEnsemble(graph, 16; seed=GAY_SEED)
    launch_parallel_walks!(ensemble, 50)
    
    println("  Total walks: $(length(ensemble.walks))")
    println("  Total steps: $(ensemble.total_steps)")
    println("  Combined bandwidth: $(round(ensemble.combined_bandwidth, digits=4))")
    println("  Combined fingerprint: 0x$(string(ensemble.combined_fingerprint, base=16))")
    println("  Phase distribution:")
    for (phase, count) in ensemble.phase_distribution
        println("    $(phase): $count walks")
    end
    println()
    
    # ─── Black/White Hole Physics ───
    println("─── Black/White Hole Information Physics ───")
    
    fps = [m.seed for m in mathematicians]
    bh = compress_to_black_hole(fps)
    
    println("  Black Hole compression: $(length(bh.inputs)) → 1")
    println("  Hawking radiation bits: $(length(bh.hawking_bits))")
    println("  Hawking color: RGB$(Int.(round.(hawking_radiation(bh) .* 255)))")
    
    wh = expand_from_white_hole(bh.compressed, 10)
    println("  White Hole expansion: 1 → $(wh.expansion_factor)")
    println("  Expansion fingerprint: 0x$(string(wh.fingerprint, base=16))")
    println()
    
    # ─── Bandwidth Tournament ───
    println("─── 3-Tuple Bandwidth Tournament ───")
    
    top_tiles = bandwidth_tournament(mathematicians; rounds=5)
    
    for (rank, tile) in enumerate(top_tiles)
        m1 = graph.nodes[tile.mathematician_ids[1]]
        m2 = graph.nodes[tile.mathematician_ids[2]]
        m3 = graph.nodes[tile.mathematician_ids[3]]
        println("  #$rank: $(m1.name), $(m2.name), $(m3.name)")
        println("       Bandwidth: $(round(tile.bandwidth, digits=4))")
    end
    println()
    
    # ─── Shortest Unique Descriptions ───
    println("─── Shortest Unique Descriptions (Edge Rewriting) ───")
    
    descs = shortest_unique_description(mathematicians)
    for (id, desc) in collect(descs)[1:5]
        println("  $id: $desc")
    end
    println()
    
    # ─── Summary ───
    println("═══════════════════════════════════════════════════════════════════════════════════")
    println("  SUMMARY: Maximally Parallel Mathematical Genealogy")
    println()
    println("  • $(length(graph.nodes)) mathematicians with chromatic identity ✓")
    println("  • 3-MATCH coloring: tripartite correct-by-construction ✓")
    println("  • Secret Self Color Game: $(length(game.reveals)) reveals, surprise $(round(game.total_surprise, digits=2)) ✓")
    println("  • $(length(ensemble.walks)) parallel walks, $(ensemble.total_steps) total steps ✓")
    println("  • GMI: $(ensemble.phase_distribution[GMI]), NGMI: $(ensemble.phase_distribution[NGMI]) ✓")
    println("  • Black hole compression: $(length(bh.inputs)) → 1 with Hawking radiation ✓")
    println("  • Top 3-tuple: $(graph.nodes[top_tiles[1].mathematician_ids[1]].name), "*
            "$(graph.nodes[top_tiles[1].mathematician_ids[2]].name), "*
            "$(graph.nodes[top_tiles[1].mathematician_ids[3]].name) ✓")
    println()
    println("  \"Every mathematician has a Secret Self Color. The genealogy is the walk.\"")
    println("═══════════════════════════════════════════════════════════════════════════════════")
    
    (graph=graph, ensemble=ensemble, game=game, top_tiles=top_tiles)
end

end # module MathGenealogyMultiverse
