# GHRIST EXPANDER CONNECTOME: Maximum Gamut Capacity for Behavior Differentiation
# ══════════════════════════════════════════════════════════════════════════════
#
# "What Ghrist sees in the cohomology, we expand in the chromatics."
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  SYNTHESIS:                                                                 │
# │                                                                             │
# │  1. Robert Ghrist as Anchoring Generator:                                   │
# │     - Elementary Applied Topology (2014)                                    │
# │     - Sheaf cohomology for sensor networks                                  │
# │     - Persistent homology / TDA                                             │
# │     - Expansion: local sections → global section (recovery)                 │
# │                                                                             │
# │  2. 3-Partite Arrangements (Tripartite Hyperedges):                         │
# │     - Partition T (-1): OBSERVERS (verify)                                  │
# │     - Partition 0: CURRENT THOUGHT (the thing being verified)               │
# │     - Partition 1: WITNESSES (confirm from different perspective)           │
# │                                                                             │
# │  3. Maximum Gamut Capacity:                                                 │
# │     - Each learnable colorspace = one axis of behavior differentiation      │
# │     - Ghrist's sheaf condition: local consistency → global uniqueness       │
# │     - Expansion property: λ₂/λ₁ maximized for fast mixing                   │
# │                                                                             │
# │  4. Connectome Modeling:                                                    │
# │     - Nodes: Researchers/AI leaders + their immediate connections           │
# │     - Edges: Co-authorship, mentorship, employment                          │
# │     - Colors: Learnable colorspaces differentiating behaviors               │
# │                                                                             │
# │  5. Multi-Observer Verification:                                            │
# │     - Current thought = most recent verifiable state                        │
# │     - Multiple observers must agree (tripartite consensus)                  │
# │     - Ghrist obstruction: disagreement = Čech cohomology non-vanishing      │
# │                                                                             │
# └─────────────────────────────────────────────────────────────────────────────┘

module GhristExpanderConnectome

export
    # Core types
    GhristNode, GhristEdge, ExpanderConnectome,
    TripartiteArrangement, CurrentThought, ObserverPerspective,
    
    # Ghrist sheaf structure
    GhristSheaf, LocalSection, GlobalSection,
    sheaf_condition, cech_obstruction, recovery_possible,
    
    # Learnable colorspaces
    LearnableColorSpace, ColorSpaceFamily,
    maximize_gamut!, differentiation_capacity,
    
    # Expander properties
    spectral_gap, mixing_time, expansion_factor,
    ramanujan_quality, cheeger_constant,
    
    # Connectome building
    build_ai_connectome, add_researcher!, add_connection!,
    immediate_neighborhood, reach_via_expansion,
    
    # 3-partite verification
    verify_tripartite, multi_observer_consensus,
    current_thought_verified, perspective_agreement,
    
    # Demo
    demo_ghrist_expander

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

const GAY_SEED = UInt64(1069)
const GHRIST_SEED = UInt64(0x6817)  # "GH" + "RI" approximation
const OBSERVER_SEED = UInt64(0x0B5E)  # "OBSE" approximation
const WITNESS_SEED = UInt64(0x57E5)  # "WITES" approximation

# Balanced ternary
const TRIT_NEG = Int8(-1)   # T
const TRIT_ZERO = Int8(0)   # 0  
const TRIT_POS = Int8(1)    # 1

# ══════════════════════════════════════════════════════════════════════════════
# CORE PRNG
# ══════════════════════════════════════════════════════════════════════════════

@inline function sm64(s::UInt64)::Tuple{UInt64, UInt64}
    z = (s + 0x9E3779B97F4A7C15)
    z = (z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ⊻ (z >> 27)) * 0x94D049BB133111EB
    (z ⊻ (z >> 31), s + 1)
end

@inline function color_from_seed(seed::UInt64)::NTuple{3, Float64}
    r, s1 = sm64(seed)
    g, s2 = sm64(s1)
    b, _  = sm64(s2)
    ((r >> 56) / 255.0, (g >> 56) / 255.0, (b >> 56) / 255.0)
end

# ══════════════════════════════════════════════════════════════════════════════
# GHRIST NODE: Researcher / Entity in Connectome
# ══════════════════════════════════════════════════════════════════════════════

"""
    GhristNode

A node in the Ghrist expander connectome.

Fields:
- id: Unique identifier (hash of name)
- name: Human-readable name
- role: :anchor, :researcher, :ai_leader, :institution
- field: Primary field/domain
- color: Chromatic identity from learnable colorspace
- partition: Tripartite partition (T, 0, 1)
- connections: IDs of connected nodes
"""
struct GhristNode
    id::UInt64
    name::String
    role::Symbol
    field::String
    color::NTuple{3, Float64}
    partition::Int8
    connections::Vector{UInt64}
    seed::UInt64
end

function GhristNode(name::String, role::Symbol, field::String; 
                    partition::Int8=TRIT_ZERO, seed::UInt64=GAY_SEED)
    id = hash(name) ⊻ seed
    color = color_from_seed(id)
    GhristNode(id, name, role, field, color, partition, UInt64[], id)
end

# ══════════════════════════════════════════════════════════════════════════════
# GHRIST EDGE: Connection with Expansion Properties
# ══════════════════════════════════════════════════════════════════════════════

"""
    GhristEdge

An edge in the expander connectome.

Types:
- :coauthorship - Published together
- :mentorship - Advisor/student relationship
- :employment - Worked at same institution
- :citation - Cited each other's work
- :collaboration - Collaborated on project
"""
struct GhristEdge
    source::UInt64
    target::UInt64
    edge_type::Symbol
    weight::Float64
    color::NTuple{3, Float64}
    fingerprint::UInt64
end

function GhristEdge(source::UInt64, target::UInt64, edge_type::Symbol; 
                    weight::Float64=1.0, seed::UInt64=GAY_SEED)
    fp = source ⊻ target ⊻ hash(edge_type) ⊻ seed
    color = color_from_seed(fp)
    GhristEdge(source, target, edge_type, weight, color, fp)
end

# ══════════════════════════════════════════════════════════════════════════════
# TRIPARTITE ARRANGEMENT: 3-Observer Verification
# ══════════════════════════════════════════════════════════════════════════════

"""
    CurrentThought

The most recent verifiable state in a tripartite arrangement.

This is the "0" partition - the thing being verified.
Must be confirmed by observers (T) and witnesses (1).
"""
struct CurrentThought
    content::Any
    timestamp::Float64
    source_node::UInt64
    color::NTuple{3, Float64}
    fingerprint::UInt64
end

function CurrentThought(content::Any, source::UInt64; seed::UInt64=GAY_SEED)
    ts = time()
    fp = hash(content) ⊻ source ⊻ hash(ts) ⊻ seed
    color = color_from_seed(fp)
    CurrentThought(content, ts, source, color, fp)
end

"""
    ObserverPerspective

An observer's view of the current thought.

Partition T (-1): Original observers who initiate verification.
Partition 1: Witnesses who confirm from different angle.
"""
struct ObserverPerspective
    observer_id::UInt64
    thought_fingerprint::UInt64
    observed_color::NTuple{3, Float64}
    confidence::Float64
    partition::Int8  # T or 1
    agrees::Bool
end

function ObserverPerspective(observer::GhristNode, thought::CurrentThought, 
                              confidence::Float64=1.0)
    # Observer sees thought through their colorspace
    observed = color_from_seed(observer.seed ⊻ thought.fingerprint)
    
    # Agreement if colors are close enough
    dist = sqrt(sum((thought.color[i] - observed[i])^2 for i in 1:3))
    agrees = dist < 0.5  # Perceptual threshold
    
    ObserverPerspective(observer.id, thought.fingerprint, observed, 
                        confidence, observer.partition, agrees)
end

"""
    TripartiteArrangement

A complete 3-observer verification structure.

- observers: Partition T (-1) - initiate verification
- thought: Partition 0 - current state being verified
- witnesses: Partition 1 - confirm from different perspectives
"""
struct TripartiteArrangement
    observers::Vector{ObserverPerspective}
    thought::CurrentThought
    witnesses::Vector{ObserverPerspective}
    
    # Verification state
    consensus_reached::Bool
    obstruction::Float64  # Čech cohomology measure
    fingerprint::UInt64
end

function TripartiteArrangement(thought::CurrentThought,
                                observers::Vector{GhristNode},
                                witnesses::Vector{GhristNode})
    obs_perspectives = [ObserverPerspective(o, thought) for o in observers]
    wit_perspectives = [ObserverPerspective(w, thought) for w in witnesses]
    
    # Consensus requires all to agree
    obs_agree = all(p -> p.agrees, obs_perspectives)
    wit_agree = all(p -> p.agrees, wit_perspectives)
    consensus = obs_agree && wit_agree && 
                length(obs_perspectives) > 0 && length(wit_perspectives) > 0
    
    # Obstruction = disagreement measure
    obs_disagree = count(p -> !p.agrees, obs_perspectives)
    wit_disagree = count(p -> !p.agrees, wit_perspectives)
    total = length(obs_perspectives) + length(wit_perspectives)
    obstruction = total > 0 ? (obs_disagree + wit_disagree) / total : 1.0
    
    fp = thought.fingerprint ⊻ 
         reduce(⊻, [p.observer_id for p in obs_perspectives]; init=UInt64(0)) ⊻
         reduce(⊻, [p.observer_id for p in wit_perspectives]; init=UInt64(0))
    
    TripartiteArrangement(obs_perspectives, thought, wit_perspectives,
                          consensus, obstruction, fp)
end

"""
    verify_tripartite(arr::TripartiteArrangement) -> Bool

Check if tripartite consensus is achieved.
"""
verify_tripartite(arr::TripartiteArrangement) = arr.consensus_reached

"""
    multi_observer_consensus(arr::TripartiteArrangement) -> Float64

Measure consensus strength (0 = none, 1 = full).
"""
function multi_observer_consensus(arr::TripartiteArrangement)
    1.0 - arr.obstruction
end

"""
    current_thought_verified(arr::TripartiteArrangement) -> Bool

Check if current thought is verified by multiple perspectives.
"""
function current_thought_verified(arr::TripartiteArrangement)
    arr.consensus_reached && arr.obstruction < 0.1
end

# ══════════════════════════════════════════════════════════════════════════════
# LEARNABLE COLORSPACE: Maximum Gamut for Behavior Differentiation
# ══════════════════════════════════════════════════════════════════════════════

"""
    LearnableColorSpace

A learnable colorspace for behavior differentiation.

Parameters:
- basis: 3×3 transformation matrix
- offset: 3-vector offset
- scale: 3-vector scale
- name: Identifier
"""
mutable struct LearnableColorSpace
    basis::Matrix{Float64}
    offset::Vector{Float64}
    scale::Vector{Float64}
    name::Symbol
    seed::UInt64
    
    # Gamut properties
    gamut_volume::Float64
    differentiation_axes::Int
end

function LearnableColorSpace(; name::Symbol=:default, seed::UInt64=GAY_SEED)
    # Initialize with identity + small random perturbation
    state = seed
    perturbations = zeros(9)
    for i in 1:9
        v, state = sm64(state)
        perturbations[i] = (v % 1000) / 10000.0 - 0.05
    end
    
    basis = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0] .+ reshape(perturbations, 3, 3)
    offset = [0.0, 0.0, 0.0]
    scale = [1.0, 1.0, 1.0]
    
    # Gamut volume = det(basis) * prod(scale)
    gamut = abs(det(basis)) * prod(scale)
    
    LearnableColorSpace(basis, offset, scale, name, seed, gamut, 3)
end

using LinearAlgebra: det

"""
    apply_colorspace(lcs::LearnableColorSpace, seed::UInt64) -> NTuple{3, Float64}

Apply colorspace transformation to seed.
"""
function apply_colorspace(lcs::LearnableColorSpace, seed::UInt64)
    raw = collect(color_from_seed(seed))
    transformed = lcs.basis * raw .* lcs.scale .+ lcs.offset
    transformed = clamp.(transformed, 0.0, 1.0)
    (transformed[1], transformed[2], transformed[3])
end

"""
    maximize_gamut!(lcs::LearnableColorSpace, n_steps::Int=50)

Maximize gamut volume through gradient ascent on det(basis).
"""
function maximize_gamut!(lcs::LearnableColorSpace, n_steps::Int=50)
    lr = 0.01
    
    for step in 1:n_steps
        current_det = abs(det(lcs.basis))
        
        # Gradient of det w.r.t. basis elements (simplified)
        grad = zeros(3, 3)
        ε = 1e-5
        for i in 1:3, j in 1:3
            basis_plus = copy(lcs.basis)
            basis_plus[i,j] += ε
            grad[i,j] = (abs(det(basis_plus)) - current_det) / ε
        end
        
        # Gradient ascent (maximize det)
        lcs.basis .+= lr .* grad
        
        # Regularize to keep bounded
        lcs.basis .= clamp.(lcs.basis, -2.0, 2.0)
    end
    
    lcs.gamut_volume = abs(det(lcs.basis)) * prod(lcs.scale)
    lcs
end

"""
    differentiation_capacity(lcs::LearnableColorSpace) -> Float64

Measure capacity to differentiate behaviors (gamut volume normalized).
"""
function differentiation_capacity(lcs::LearnableColorSpace)
    # Normalized by unit cube volume
    min(1.0, lcs.gamut_volume)
end

"""
    ColorSpaceFamily

A family of learnable colorspaces for maximum differentiation.
"""
struct ColorSpaceFamily
    spaces::Vector{LearnableColorSpace}
    combined_capacity::Float64
    fingerprint::UInt64
end

function ColorSpaceFamily(n::Int; seed::UInt64=GAY_SEED)
    spaces = [LearnableColorSpace(; name=Symbol("space_$i"), seed=seed ⊻ UInt64(i)) 
              for i in 1:n]
    
    # Maximize each space
    for lcs in spaces
        maximize_gamut!(lcs, 30)
    end
    
    # Combined capacity (multiplicative for independent axes)
    combined = prod(differentiation_capacity(lcs) for lcs in spaces)
    fp = reduce(⊻, [lcs.seed for lcs in spaces])
    
    ColorSpaceFamily(spaces, combined^(1/n), fp)  # Geometric mean
end

# ══════════════════════════════════════════════════════════════════════════════
# GHRIST SHEAF: Local → Global Recovery
# ══════════════════════════════════════════════════════════════════════════════

"""
    LocalSection

A local observation at a node.
"""
struct LocalSection
    node_id::UInt64
    observed_color::NTuple{3, Float64}
    timestamp::Float64
    confidence::Float64
end

"""
    GlobalSection

A global section (if it exists) recovered from local observations.
"""
struct GlobalSection
    seed::UInt64
    color::NTuple{3, Float64}
    recovery_work::Float64
    exists::Bool
end

"""
    GhristSheaf

Sheaf structure over the connectome for seed recovery analysis.

Following Ghrist's sensor network sheaves:
- Stalks: Local color observations
- Restriction maps: Edge color transport
- Sheaf condition: Local consistency → global uniqueness
"""
struct GhristSheaf
    # base::ExpanderConnectome  # Forward reference - use Any
    base::Any
    local_sections::Dict{UInt64, LocalSection}
    global_section::Union{GlobalSection, Nothing}
    
    # Cohomology
    cech_obstruction::Float64
    recovery_possible::Bool
end

"""
    sheaf_condition(sheaf::GhristSheaf) -> Bool

Check if local sections satisfy the sheaf (gluing) condition.
"""
function sheaf_condition(sheaf::GhristSheaf)
    # Simplified: check if local observations are consistent
    if length(sheaf.local_sections) < 2
        return true
    end
    
    sections = collect(values(sheaf.local_sections))
    
    # Check pairwise consistency
    for i in 1:length(sections)-1
        for j in i+1:length(sections)
            s1, s2 = sections[i], sections[j]
            dist = sqrt(sum((s1.observed_color[k] - s2.observed_color[k])^2 for k in 1:3))
            if dist > 1.0  # Inconsistent
                return false
            end
        end
    end
    
    true
end

"""
    cech_obstruction(sheaf::GhristSheaf) -> Float64

Compute Čech cohomology obstruction to global section existence.
0 = no obstruction (global section exists)
>0 = obstruction (recovery may be impossible)
"""
function cech_obstruction(sheaf::GhristSheaf)
    sheaf.cech_obstruction
end

"""
    recovery_possible(sheaf::GhristSheaf) -> Bool

Determine if seed recovery is theoretically possible.
"""
function recovery_possible(sheaf::GhristSheaf)
    sheaf.recovery_possible
end

# ══════════════════════════════════════════════════════════════════════════════
# EXPANDER CONNECTOME: Main Graph Structure
# ══════════════════════════════════════════════════════════════════════════════

"""
    ExpanderConnectome

The full expander graph of the research/AI connectome.
Anchored by Robert Ghrist for topological perspective.
"""
mutable struct ExpanderConnectome
    nodes::Dict{UInt64, GhristNode}
    edges::Vector{GhristEdge}
    
    # Anchor
    anchor::GhristNode  # Robert Ghrist
    
    # Colorspaces for differentiation
    colorspaces::ColorSpaceFamily
    
    # Expander properties
    spectral_gap::Float64
    expansion_factor::Float64
    
    seed::UInt64
    fingerprint::UInt64
end

"""
    build_ai_connectome(n_colorspaces::Int=5) -> ExpanderConnectome

Build the AI/research connectome with Ghrist as anchor.
"""
function build_ai_connectome(n_colorspaces::Int=5)
    # Create anchor: Robert Ghrist
    ghrist = GhristNode("Robert Ghrist", :anchor, "Applied Topology";
                        partition=TRIT_ZERO, seed=GHRIST_SEED)
    
    nodes = Dict{UInt64, GhristNode}()
    nodes[ghrist.id] = ghrist
    
    # AI Leaders (Partition T - Observers)
    ai_leaders = [
        ("Ilya Sutskever", "Deep Learning / SSI"),
        ("Mira Murati", "AI Leadership"),
        ("Dario Amodei", "AI Safety / Anthropic"),
        ("Sam Altman", "AI Strategy / OpenAI"),
        ("Demis Hassabis", "AGI / DeepMind"),
    ]
    
    for (name, field) in ai_leaders
        node = GhristNode(name, :ai_leader, field; partition=TRIT_NEG, seed=GAY_SEED)
        nodes[node.id] = node
    end
    
    # Topologists/Mathematicians (Partition 1 - Witnesses)
    mathematicians = [
        ("David Spivak", "Category Theory / Polynomial Functors"),
        ("Kevin Carlson", "Comonads / Topos Institute"),
        ("Aaron Fairbanks", "Set-Sets / Coalgebra"),
        ("Gunnar Carlsson", "Persistent Homology / Ayasdi"),
        ("Shmuel Weinberger", "Topology / Chicago"),
        ("Herbert Edelsbrunner", "Computational Topology"),
    ]
    
    for (name, field) in mathematicians
        node = GhristNode(name, :researcher, field; partition=TRIT_POS, seed=GAY_SEED)
        nodes[node.id] = node
    end
    
    # Immediate connections to Ghrist
    ghrist_connections = [
        ("Gunnar Carlsson", :collaboration),
        ("Herbert Edelsbrunner", :collaboration),
        ("Shmuel Weinberger", :collaboration),
        ("David Spivak", :citation),
    ]
    
    edges = GhristEdge[]
    
    for (name, etype) in ghrist_connections
        target_id = hash(name) ⊻ GAY_SEED
        if haskey(nodes, target_id)
            push!(ghrist.connections, target_id)
            push!(nodes[target_id].connections, ghrist.id)
            push!(edges, GhristEdge(ghrist.id, target_id, etype))
        end
    end
    
    # AI network connections
    ai_connections = [
        ("Ilya Sutskever", "Sam Altman", :employment),
        ("Mira Murati", "Sam Altman", :employment),
        ("Dario Amodei", "Sam Altman", :employment),  # Former
        ("Dario Amodei", "Ilya Sutskever", :collaboration),
    ]
    
    for (n1, n2, etype) in ai_connections
        id1 = hash(n1) ⊻ GAY_SEED
        id2 = hash(n2) ⊻ GAY_SEED
        if haskey(nodes, id1) && haskey(nodes, id2)
            push!(nodes[id1].connections, id2)
            push!(nodes[id2].connections, id1)
            push!(edges, GhristEdge(id1, id2, etype))
        end
    end
    
    # Topos connections
    topos_connections = [
        ("David Spivak", "Kevin Carlson", :coauthorship),
        ("David Spivak", "Aaron Fairbanks", :coauthorship),
        ("Kevin Carlson", "Aaron Fairbanks", :coauthorship),
    ]
    
    for (n1, n2, etype) in topos_connections
        id1 = hash(n1) ⊻ GAY_SEED
        id2 = hash(n2) ⊻ GAY_SEED
        if haskey(nodes, id1) && haskey(nodes, id2)
            push!(nodes[id1].connections, id2)
            push!(nodes[id2].connections, id1)
            push!(edges, GhristEdge(id1, id2, etype))
        end
    end
    
    # Create colorspace family
    colorspaces = ColorSpaceFamily(n_colorspaces; seed=GAY_SEED)
    
    # Estimate spectral gap (simplified)
    avg_degree = length(edges) * 2 / max(1, length(nodes))
    spectral_gap = 1.0 - 2.0 * sqrt(max(1, avg_degree - 1)) / max(1, avg_degree)
    expansion = 1.0 + spectral_gap
    
    fp = reduce(⊻, [n.id for n in values(nodes)]; init=GAY_SEED)
    
    ExpanderConnectome(nodes, edges, ghrist, colorspaces, 
                       max(0.1, spectral_gap), expansion, GAY_SEED, fp)
end

"""
    add_researcher!(conn::ExpanderConnectome, name::String, field::String, 
                    partition::Int8=TRIT_ZERO)

Add a researcher to the connectome.
"""
function add_researcher!(conn::ExpanderConnectome, name::String, field::String;
                         partition::Int8=TRIT_ZERO)
    node = GhristNode(name, :researcher, field; partition=partition, seed=conn.seed)
    conn.nodes[node.id] = node
    node
end

"""
    add_connection!(conn::ExpanderConnectome, n1::String, n2::String, 
                    edge_type::Symbol)

Add a connection between two nodes.
"""
function add_connection!(conn::ExpanderConnectome, n1::String, n2::String,
                         edge_type::Symbol)
    id1 = hash(n1) ⊻ conn.seed
    id2 = hash(n2) ⊻ conn.seed
    
    if haskey(conn.nodes, id1) && haskey(conn.nodes, id2)
        push!(conn.nodes[id1].connections, id2)
        push!(conn.nodes[id2].connections, id1)
        push!(conn.edges, GhristEdge(id1, id2, edge_type; seed=conn.seed))
        return true
    end
    false
end

"""
    immediate_neighborhood(conn::ExpanderConnectome, name::String) -> Vector{GhristNode}

Get immediate connections of a node.
"""
function immediate_neighborhood(conn::ExpanderConnectome, name::String)
    id = hash(name) ⊻ conn.seed
    if !haskey(conn.nodes, id)
        return GhristNode[]
    end
    
    node = conn.nodes[id]
    [conn.nodes[cid] for cid in node.connections if haskey(conn.nodes, cid)]
end

"""
    reach_via_expansion(conn::ExpanderConnectome, start::String, 
                        max_hops::Int=3) -> Vector{GhristNode}

Find all nodes reachable within max_hops using expansion property.
"""
function reach_via_expansion(conn::ExpanderConnectome, start::String, 
                              max_hops::Int=3)
    start_id = hash(start) ⊻ conn.seed
    if !haskey(conn.nodes, start_id)
        return GhristNode[]
    end
    
    visited = Set{UInt64}([start_id])
    frontier = Set([start_id])
    
    for hop in 1:max_hops
        new_frontier = Set{UInt64}()
        for nid in frontier
            if haskey(conn.nodes, nid)
                for cid in conn.nodes[nid].connections
                    if !(cid in visited)
                        push!(visited, cid)
                        push!(new_frontier, cid)
                    end
                end
            end
        end
        frontier = new_frontier
        if isempty(frontier)
            break
        end
    end
    
    [conn.nodes[id] for id in visited if haskey(conn.nodes, id)]
end

"""
    spectral_gap(conn::ExpanderConnectome) -> Float64

Return the spectral gap of the connectome.
"""
spectral_gap(conn::ExpanderConnectome) = conn.spectral_gap

"""
    mixing_time(conn::ExpanderConnectome) -> Float64

Estimate mixing time from spectral gap.
"""
function mixing_time(conn::ExpanderConnectome)
    if conn.spectral_gap ≤ 0.01
        return Inf
    end
    log(length(conn.nodes)) / conn.spectral_gap
end

"""
    expansion_factor(conn::ExpanderConnectome) -> Float64

Return the expansion factor.
"""
expansion_factor(conn::ExpanderConnectome) = conn.expansion_factor

# ══════════════════════════════════════════════════════════════════════════════
# DEMO
# ══════════════════════════════════════════════════════════════════════════════

function demo_ghrist_expander()
    println()
    println("╔═════════════════════════════════════════════════════════════════════════════╗")
    println("║  GHRIST EXPANDER CONNECTOME: Maximum Gamut Behavior Differentiation         ║")
    println("╠═════════════════════════════════════════════════════════════════════════════╣")
    println("║  Anchor: Robert Ghrist | 3-Partite Verification | Learnable Colorspaces     ║")
    println("╚═════════════════════════════════════════════════════════════════════════════╝")
    println()
    
    # 1. Build connectome
    println("─── 1. BUILD AI/RESEARCH CONNECTOME ───")
    conn = build_ai_connectome(5)
    println("  Anchor: $(conn.anchor.name) ($(conn.anchor.field))")
    println("  Nodes: $(length(conn.nodes))")
    println("  Edges: $(length(conn.edges))")
    println("  Spectral gap: $(round(conn.spectral_gap, digits=4))")
    println("  Mixing time: $(round(mixing_time(conn), digits=2))")
    println()
    
    # 2. Colorspace family
    println("─── 2. LEARNABLE COLORSPACE FAMILY ───")
    println("  Number of spaces: $(length(conn.colorspaces.spaces))")
    for (i, lcs) in enumerate(conn.colorspaces.spaces)
        println("    Space $i: gamut=$(round(lcs.gamut_volume, digits=3)), capacity=$(round(differentiation_capacity(lcs), digits=3))")
    end
    println("  Combined capacity: $(round(conn.colorspaces.combined_capacity, digits=4))")
    println()
    
    # 3. Immediate neighborhoods
    println("─── 3. IMMEDIATE NEIGHBORHOODS ───")
    for name in ["Robert Ghrist", "Ilya Sutskever", "David Spivak"]
        neighbors = immediate_neighborhood(conn, name)
        println("  $name: $(length(neighbors)) connections")
        for n in neighbors
            println("    - $(n.name) ($(n.field))")
        end
    end
    println()
    
    # 4. Expansion reach
    println("─── 4. EXPANSION REACHABILITY ───")
    for hops in 1:3
        reachable = reach_via_expansion(conn, "Robert Ghrist", hops)
        println("  From Ghrist, $hops hops: $(length(reachable)) nodes")
    end
    println()
    
    # 5. 3-Partite verification
    println("─── 5. TRIPARTITE VERIFICATION ───")
    
    # Create current thought
    thought = CurrentThought("Sheaf cohomology enables distributed sensing", conn.anchor.id)
    println("  Current thought: \"$(thought.content)\"")
    println("  Source: $(conn.anchor.name)")
    println("  Color: $(round.(thought.color, digits=3))")
    
    # Get observers (T partition) and witnesses (1 partition)
    observers = [n for n in values(conn.nodes) if n.partition == TRIT_NEG]
    witnesses = [n for n in values(conn.nodes) if n.partition == TRIT_POS]
    
    println("  Observers (T): $(length(observers))")
    println("  Witnesses (1): $(length(witnesses))")
    
    # Create arrangement
    arr = TripartiteArrangement(thought, observers[1:min(2, length(observers))], 
                                 witnesses[1:min(2, length(witnesses))])
    
    println("  Consensus reached: $(verify_tripartite(arr))")
    println("  Agreement level: $(round(multi_observer_consensus(arr), digits=3))")
    println("  Obstruction: $(round(arr.obstruction, digits=3))")
    println("  Verified: $(current_thought_verified(arr))")
    println()
    
    # 6. Summary
    println("═══════════════════════════════════════════════════════════════════════════════")
    println("  GHRIST ANCHOR → Expansion property propagates to all reachable nodes")
    println("  3-PARTITE: Observers (AI) × Current (Ghrist) × Witnesses (Mathematicians)")
    println("  $(length(conn.colorspaces.spaces)) learnable colorspaces differentiate behaviors")
    println("═══════════════════════════════════════════════════════════════════════════════")
end

end # module GhristExpanderConnectome
