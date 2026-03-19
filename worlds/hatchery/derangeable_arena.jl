# Derangeable Arena: 3-Adversarial Self-Same in Ergodic World Hops
# ═══════════════════════════════════════════════════════════════════════════════
#
# "No fixed points. Every world moves. Every role uncertain."
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  DERANGEMENT ALGEBRA                                                        │
# │                                                                             │
# │  Derangement: permutation with NO fixed points                              │
# │    D(n) = n! × Σ(k=0→n) (-1)^k / k!  ≈ n!/e                                │
# │                                                                             │
# │  For 3 agents: D(3) = 2                                                    │
# │    (1,2,3) → (2,3,1) or (3,1,2)  [only two derangements]                   │
# │                                                                             │
# │  ARENA INDETERMINACY:                                                       │
# │    When color bandwidth exhausted → ArenaIndeterminacyError                │
# │    Need more colors → expand chromatic basis                                │
# │    Need more derangements → increase world count                            │
# │                                                                             │
# │  3-ADVERSARIAL SELF-SAME:                                                   │
# │    Three agents, each uncertain if:                                         │
# │      • Originary (at GAY_SEED)                                             │
# │      • Derived (rotated from origin)                                        │
# │      • Deranged (no fixed relationship to origin)                          │
# │                                                                             │
# │    Ergodic outcome: long-run behavior same regardless of starting role     │
# │                                                                             │
# │  TRIANGLE INEQUALITY WORLD HOPS:                                            │
# │    d(W₁, W₃) ≤ d(W₁, W₂) + d(W₂, W₃)                                       │
# │    Violated → ultrametric (tree structure)                                  │
# │    Satisfied → proper metric (graph structure)                              │
# │                                                                             │
# │  MAXENT DISSONANCE MONOPOLE ZONES:                                          │
# │    Regions where entropy maximized + colors maximally dissonant            │
# │    Monopole = single charge center of dissonance field                      │
# │    Mining estimates: extract information from monopole structure            │
# └─────────────────────────────────────────────────────────────────────────────┘

module DerangeableArena

export
    # Core types
    Derangement, DerangementGroup, DerangeableWorld,
    ArenaIndeterminacyError, ColorBandwidthExhausted,
    
    # Derangement operations
    generate_derangements, apply_derangement, is_derangement,
    derangement_count, random_derangement,
    
    # Arena management
    Arena, ArenaState, ArenaAgent,
    expand_arena!, contract_arena!, arena_bandwidth,
    
    # 3-Adversarial self-same
    TriAdversary, RoleUncertainty, SelfSameGame,
    originary_probability, derived_probability, deranged_probability,
    play_round!, ergodic_limit, role_entropy,
    
    # Triangle inequality world hops
    WorldMetric, TriangleCheck, WorldHop,
    check_triangle!, hop_cost, optimal_hop_path,
    is_ultrametric, metric_graph,
    
    # MaxEnt dissonance monopoles
    DissonanceField, Monopole, MonopoleZone,
    compute_dissonance!, find_monopoles, mine_estimates,
    zone_entropy, monopole_charge,
    
    # Ergodic convergence
    ErgodicChain, convergence_rate, mixing_time,
    stationary_distribution, ergodic_color,
    
    # Demo
    demo_derangeable_arena

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

const GAY_SEED = UInt64(1069)
const ARENA_SEED = UInt64(0xA8E4A)
const MONOPOLE_SEED = UInt64(0x404091E)

# Derangement subfactorials (D(n) for n = 0 to 12)
const SUBFACTORIALS = [1, 0, 1, 2, 9, 44, 265, 1854, 14833, 133496, 1334961, 14684570, 176214841]

# ═══════════════════════════════════════════════════════════════════════════════
# SPLITMIX64
# ═══════════════════════════════════════════════════════════════════════════════

@inline function splitmix64(state::UInt64)::UInt64
    z = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ⊻ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    z ⊻ (z >> 31)
end

@inline function color_from_seed(seed::UInt64)
    r = splitmix64(seed)
    g = splitmix64(r)
    b = splitmix64(g)
    (r=(r >> 56) / 255.0, g=(g >> 56) / 255.0, b=(b >> 56) / 255.0)
end

@inline function color_distance(c1::NamedTuple, c2::NamedTuple)::Float64
    sqrt((c1.r - c2.r)^2 + (c1.g - c2.g)^2 + (c1.b - c2.b)^2)
end

# ═══════════════════════════════════════════════════════════════════════════════
# ERRORS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ArenaIndeterminacyError

Thrown when arena state cannot be determined due to insufficient color bandwidth.
"""
struct ArenaIndeterminacyError <: Exception
    message::String
    required_bandwidth::Int
    available_bandwidth::Int
    seed::UInt64
end

function Base.showerror(io::IO, e::ArenaIndeterminacyError)
    print(io, "ArenaIndeterminacyError: $(e.message)")
    print(io, " (required: $(e.required_bandwidth), available: $(e.available_bandwidth))")
end

"""
    ColorBandwidthExhausted

Thrown when no more unique colors can be generated.
"""
struct ColorBandwidthExhausted <: Exception
    n_colors_used::Int
    n_colors_max::Int
    seed::UInt64
end

function Base.showerror(io::IO, e::ColorBandwidthExhausted)
    print(io, "ColorBandwidthExhausted: used $(e.n_colors_used)/$(e.n_colors_max) colors")
end

# ═══════════════════════════════════════════════════════════════════════════════
# DERANGEMENTS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    Derangement

A permutation with no fixed points.
"""
struct Derangement
    n::Int
    permutation::Vector{Int}
    
    # Cycle structure
    cycles::Vector{Vector{Int}}
    
    seed::UInt64
    color::NamedTuple{(:r, :g, :b), NTuple{3, Float64}}
end

function Derangement(perm::Vector{Int})
    n = length(perm)
    
    # Verify it's a derangement
    @assert all(perm[i] != i for i in 1:n) "Not a derangement: has fixed point"
    @assert sort(perm) == 1:n "Not a valid permutation"
    
    # Find cycle structure
    cycles = Vector{Int}[]
    visited = falses(n)
    
    for start in 1:n
        visited[start] && continue
        cycle = Int[]
        current = start
        while !visited[current]
            visited[current] = true
            push!(cycle, current)
            current = perm[current]
        end
        length(cycle) > 1 && push!(cycles, cycle)
    end
    
    seed = reduce(⊻, [UInt64(p) << (4 * (i-1)) for (i, p) in enumerate(perm)]; init=GAY_SEED)
    Derangement(n, perm, cycles, seed, color_from_seed(seed))
end

"""
    is_derangement(perm) -> Bool

Check if a permutation is a derangement (no fixed points).
"""
function is_derangement(perm::Vector{Int})::Bool
    all(perm[i] != i for i in eachindex(perm))
end

"""
    derangement_count(n) -> BigInt

Number of derangements of n elements: D(n) = n! × Σ(k=0→n) (-1)^k / k!
"""
function derangement_count(n::Int)::BigInt
    if n <= 12
        return BigInt(SUBFACTORIALS[n + 1])
    end
    
    # Use recurrence: D(n) = (n-1) × (D(n-1) + D(n-2))
    d_prev2 = BigInt(1)  # D(0) = 1
    d_prev1 = BigInt(0)  # D(1) = 0
    
    for k in 2:n
        d_curr = (k - 1) * (d_prev1 + d_prev2)
        d_prev2, d_prev1 = d_prev1, d_curr
    end
    
    d_prev1
end

"""
    generate_derangements(n) -> Vector{Derangement}

Generate all derangements of n elements.
"""
function generate_derangements(n::Int)::Vector{Derangement}
    n <= 1 && return Derangement[]
    n == 2 && return [Derangement([2, 1])]
    
    derangements = Derangement[]
    
    # Generate via recursive algorithm
    perm = collect(1:n)
    
    function generate!(pos::Int)
        if pos > n
            if is_derangement(perm)
                push!(derangements, Derangement(copy(perm)))
            end
            return
        end
        
        for i in pos:n
            # Swap
            perm[pos], perm[i] = perm[i], perm[pos]
            
            # Prune: if perm[pos] == pos, skip (would create fixed point)
            if perm[pos] != pos
                generate!(pos + 1)
            end
            
            # Swap back
            perm[pos], perm[i] = perm[i], perm[pos]
        end
    end
    
    generate!(1)
    derangements
end

"""
    random_derangement(n; seed) -> Derangement

Generate a random derangement using rejection sampling with chromatic guidance.
"""
function random_derangement(n::Int; seed::UInt64=GAY_SEED)::Derangement
    n <= 1 && error("Cannot derange $n elements")
    
    state = seed
    max_attempts = 1000
    
    for _ in 1:max_attempts
        # Generate random permutation
        perm = collect(1:n)
        
        for i in n:-1:2
            state = splitmix64(state)
            j = 1 + Int(state % UInt64(i))
            perm[i], perm[j] = perm[j], perm[i]
        end
        
        # Check if derangement
        if is_derangement(perm)
            return Derangement(perm)
        end
        
        # Fix up: swap any fixed points
        for i in 1:n
            if perm[i] == i
                # Find another position to swap with
                for j in 1:n
                    if i != j && perm[j] != i && perm[i] != j
                        perm[i], perm[j] = perm[j], perm[i]
                        break
                    end
                end
            end
        end
        
        if is_derangement(perm)
            return Derangement(perm)
        end
    end
    
    # Fallback: for n >= 2, (2,3,...,n,1) is always a derangement
    Derangement(vcat(2:n, [1]))
end

"""
    apply_derangement(d, items) -> Vector

Apply derangement to a list of items.
"""
function apply_derangement(d::Derangement, items::Vector{T}) where T
    @assert length(items) == d.n "Length mismatch"
    [items[d.permutation[i]] for i in 1:d.n]
end

"""
    DerangementGroup

The group of all derangements on n elements.
Note: Derangements don't form a group (not closed under composition),
but the symmetric group S_n contains them.
"""
struct DerangementGroup
    n::Int
    elements::Vector{Derangement}
    
    # Color of the group (XOR of all element colors)
    seed::UInt64
    color::NamedTuple{(:r, :g, :b), NTuple{3, Float64}}
end

function DerangementGroup(n::Int)
    elements = generate_derangements(n)
    seed = reduce(⊻, [d.seed for d in elements]; init=GAY_SEED)
    DerangementGroup(n, elements, seed, color_from_seed(seed))
end

# ═══════════════════════════════════════════════════════════════════════════════
# DERANGEABLE WORLDS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    DerangeableWorld

A world that can be deranged (all elements shifted, none fixed).
"""
mutable struct DerangeableWorld
    id::Int
    name::Symbol
    
    # World state
    elements::Vector{UInt64}
    current_derangement::Union{Derangement, Nothing}
    derangement_history::Vector{Derangement}
    
    # Color state
    colors::Vector{NamedTuple{(:r, :g, :b), NTuple{3, Float64}}}
    bandwidth_used::Int
    bandwidth_max::Int
    
    seed::UInt64
    color::NamedTuple{(:r, :g, :b), NTuple{3, Float64}}
end

function DerangeableWorld(id::Int, n_elements::Int; name::Symbol=:world, bandwidth::Int=1024)
    seed = splitmix64(ARENA_SEED ⊻ UInt64(id))
    elements = [splitmix64(seed ⊻ UInt64(i)) for i in 1:n_elements]
    colors = [color_from_seed(e) for e in elements]
    
    DerangeableWorld(id, name, elements, nothing, Derangement[], colors, n_elements, bandwidth,
                     seed, color_from_seed(seed))
end

function derange!(world::DerangeableWorld)
    n = length(world.elements)
    d = random_derangement(n; seed=world.seed)
    
    # Apply derangement
    world.elements = apply_derangement(d, world.elements)
    world.colors = [color_from_seed(e) for e in world.elements]
    
    world.current_derangement = d
    push!(world.derangement_history, d)
    world.bandwidth_used += n
    
    if world.bandwidth_used > world.bandwidth_max
        throw(ColorBandwidthExhausted(world.bandwidth_used, world.bandwidth_max, world.seed))
    end
    
    world.seed = splitmix64(world.seed ⊻ d.seed)
    world.color = color_from_seed(world.seed)
    
    d
end

# ═══════════════════════════════════════════════════════════════════════════════
# ARENA
# ═══════════════════════════════════════════════════════════════════════════════

@enum ArenaState begin
    Stable
    Indeterminate
    Expanding
    Contracting
    Exhausted
end

"""
    ArenaAgent

An agent in the arena with uncertain role.
"""
mutable struct ArenaAgent
    id::Int
    
    # Role probabilities (should sum to 1)
    p_originary::Float64   # Probability agent is at origin
    p_derived::Float64     # Probability agent is derived from origin
    p_deranged::Float64    # Probability agent is deranged (no fixed relation)
    
    # Current color
    seed::UInt64
    color::NamedTuple{(:r, :g, :b), NTuple{3, Float64}}
    
    # Belief state
    believed_role::Symbol  # :originary, :derived, :deranged, :unknown
    confidence::Float64
end

function ArenaAgent(id::Int; seed::UInt64=GAY_SEED)
    agent_seed = splitmix64(seed ⊻ UInt64(id))
    
    # Start with uniform uncertainty
    ArenaAgent(id, 1/3, 1/3, 1/3, agent_seed, color_from_seed(agent_seed), :unknown, 0.0)
end

function originary_probability(agent::ArenaAgent)::Float64
    agent.p_originary
end

function derived_probability(agent::ArenaAgent)::Float64
    agent.p_derived
end

function deranged_probability(agent::ArenaAgent)::Float64
    agent.p_deranged
end

function role_entropy(agent::ArenaAgent)::Float64
    probs = [agent.p_originary, agent.p_derived, agent.p_deranged]
    -sum(p * log2(p + 1e-10) for p in probs)
end

"""
    Arena

The arena containing worlds and agents.
"""
mutable struct Arena
    worlds::Vector{DerangeableWorld}
    agents::Vector{ArenaAgent}
    
    state::ArenaState
    
    # Bandwidth management
    total_bandwidth::Int
    used_bandwidth::Int
    
    # Derangement tracking
    global_derangement::Union{Derangement, Nothing}
    
    seed::UInt64
    color::NamedTuple{(:r, :g, :b), NTuple{3, Float64}}
end

function Arena(n_worlds::Int, n_elements_per_world::Int; n_agents::Int=3)
    seed = ARENA_SEED
    
    worlds = [DerangeableWorld(i, n_elements_per_world; name=Symbol("world_$i"))
              for i in 1:n_worlds]
    agents = [ArenaAgent(i; seed=splitmix64(seed ⊻ UInt64(i))) for i in 1:n_agents]
    
    total_bw = n_worlds * n_elements_per_world * 10  # 10x headroom
    
    Arena(worlds, agents, Stable, total_bw, 0, nothing, seed, color_from_seed(seed))
end

function arena_bandwidth(arena::Arena)::Tuple{Int, Int}
    (arena.used_bandwidth, arena.total_bandwidth)
end

function expand_arena!(arena::Arena, n_new_worlds::Int)
    arena.state = Expanding
    
    current_n = length(arena.worlds)
    n_elements = isempty(arena.worlds) ? 3 : length(arena.worlds[1].elements)
    
    for i in 1:n_new_worlds
        push!(arena.worlds, DerangeableWorld(current_n + i, n_elements))
    end
    
    arena.total_bandwidth += n_new_worlds * n_elements * 10
    arena.state = Stable
    arena.seed = splitmix64(arena.seed)
    arena.color = color_from_seed(arena.seed)
    
    arena
end

function contract_arena!(arena::Arena, n_remove::Int)
    arena.state = Contracting
    
    n_remove = min(n_remove, length(arena.worlds) - 1)  # Keep at least one
    for _ in 1:n_remove
        pop!(arena.worlds)
    end
    
    arena.state = Stable
    arena.seed = splitmix64(arena.seed)
    arena.color = color_from_seed(arena.seed)
    
    arena
end

function derange_arena!(arena::Arena)
    if length(arena.worlds) <= 1
        throw(ArenaIndeterminacyError("Cannot derange with only $(length(arena.worlds)) worlds",
                                       2, length(arena.worlds), arena.seed))
    end
    
    # Derange the world order
    n = length(arena.worlds)
    d = random_derangement(n; seed=arena.seed)
    
    arena.worlds = apply_derangement(d, arena.worlds)
    arena.global_derangement = d
    
    arena.used_bandwidth += n
    
    if arena.used_bandwidth > arena.total_bandwidth
        arena.state = Exhausted
        throw(ColorBandwidthExhausted(arena.used_bandwidth, arena.total_bandwidth, arena.seed))
    end
    
    arena.seed = splitmix64(arena.seed ⊻ d.seed)
    arena.color = color_from_seed(arena.seed)
    
    d
end

# ═══════════════════════════════════════════════════════════════════════════════
# 3-ADVERSARIAL SELF-SAME GAME
# ═══════════════════════════════════════════════════════════════════════════════

"""
    RoleUncertainty

Uncertainty about role in the 3-adversarial game.
"""
struct RoleUncertainty
    agent_id::Int
    
    # Belief distribution
    beliefs::Vector{Float64}  # [p_originary, p_derived, p_deranged]
    
    # Evidence observed
    observations::Vector{Tuple{Symbol, Float64}}  # (observation_type, value)
    
    # Entropy
    entropy::Float64
end

function RoleUncertainty(agent::ArenaAgent)
    beliefs = [agent.p_originary, agent.p_derived, agent.p_deranged]
    RoleUncertainty(agent.id, beliefs, Tuple{Symbol, Float64}[], role_entropy(agent))
end

"""
    TriAdversary

Three agents playing adversarial game about roles.
"""
struct TriAdversary
    agents::NTuple{3, ArenaAgent}
    
    # Pairwise distances (color-based)
    distances::NTuple{3, Float64}  # (d12, d13, d23)
    
    # Who believes they are originary?
    claims::NTuple{3, Bool}
    
    seed::UInt64
end

function TriAdversary(agents::Vector{ArenaAgent})
    @assert length(agents) == 3 "TriAdversary requires exactly 3 agents"
    
    a1, a2, a3 = agents[1], agents[2], agents[3]
    
    d12 = color_distance(a1.color, a2.color)
    d13 = color_distance(a1.color, a3.color)
    d23 = color_distance(a2.color, a3.color)
    
    # Each agent claims originary if their p_originary > 0.5
    claims = (a1.p_originary > 0.5, a2.p_originary > 0.5, a3.p_originary > 0.5)
    
    combined_seed = a1.seed ⊻ a2.seed ⊻ a3.seed
    TriAdversary((a1, a2, a3), (d12, d13, d23), claims, combined_seed)
end

"""
    SelfSameGame

The game where agents try to determine if they are originary, derived, or deranged.
"""
mutable struct SelfSameGame
    tri_adversary::TriAdversary
    origin_seed::UInt64  # The true origin
    
    # Game state
    round::Int
    history::Vector{Tuple{Int, Symbol, Float64}}  # (round, event, value)
    
    # Ergodic tracking
    role_visits::Dict{Int, Dict{Symbol, Int}}  # agent_id → role → visit_count
    
    seed::UInt64
end

function SelfSameGame(agents::Vector{ArenaAgent}; origin::UInt64=GAY_SEED)
    ta = TriAdversary(agents)
    
    role_visits = Dict(a.id => Dict(:originary => 0, :derived => 0, :deranged => 0) 
                       for a in agents)
    
    SelfSameGame(ta, origin, 0, Tuple{Int, Symbol, Float64}[], role_visits,
                 splitmix64(ta.seed ⊻ origin))
end

function play_round!(game::SelfSameGame)
    game.round += 1
    
    agents = collect(game.tri_adversary.agents)
    
    for (i, agent) in enumerate(agents)
        # Compute distance to origin
        origin_color = color_from_seed(game.origin_seed)
        dist_to_origin = color_distance(agent.color, origin_color)
        
        # Update beliefs based on distance
        if dist_to_origin < 0.1
            # Very close to origin → likely originary
            agent.p_originary = min(0.9, agent.p_originary + 0.1)
            agent.p_derived = max(0.05, agent.p_derived - 0.05)
            agent.p_deranged = max(0.05, agent.p_deranged - 0.05)
        elseif dist_to_origin < 0.5
            # Medium distance → likely derived
            agent.p_derived = min(0.9, agent.p_derived + 0.1)
            agent.p_originary = max(0.05, agent.p_originary - 0.05)
            agent.p_deranged = max(0.05, agent.p_deranged - 0.05)
        else
            # Far from origin → likely deranged
            agent.p_deranged = min(0.9, agent.p_deranged + 0.1)
            agent.p_originary = max(0.05, agent.p_originary - 0.05)
            agent.p_derived = max(0.05, agent.p_derived - 0.05)
        end
        
        # Normalize
        total = agent.p_originary + agent.p_derived + agent.p_deranged
        agent.p_originary /= total
        agent.p_derived /= total
        agent.p_deranged /= total
        
        # Determine believed role
        max_p = max(agent.p_originary, agent.p_derived, agent.p_deranged)
        agent.believed_role = if max_p == agent.p_originary
            :originary
        elseif max_p == agent.p_derived
            :derived
        else
            :deranged
        end
        agent.confidence = max_p
        
        # Track visits
        game.role_visits[agent.id][agent.believed_role] += 1
        
        # Rotate color for next round
        agent.seed = splitmix64(agent.seed)
        agent.color = color_from_seed(agent.seed)
        
        push!(game.history, (game.round, agent.believed_role, dist_to_origin))
    end
    
    # Update tri-adversary
    game.tri_adversary = TriAdversary(agents)
    game.seed = splitmix64(game.seed)
    
    game.round
end

function ergodic_limit(game::SelfSameGame)::Dict{Int, Dict{Symbol, Float64}}
    # Compute empirical distribution of roles per agent
    result = Dict{Int, Dict{Symbol, Float64}}()
    
    for (agent_id, visits) in game.role_visits
        total = sum(values(visits))
        total == 0 && continue
        
        result[agent_id] = Dict(
            :originary => visits[:originary] / total,
            :derived => visits[:derived] / total,
            :deranged => visits[:deranged] / total
        )
    end
    
    result
end

# ═══════════════════════════════════════════════════════════════════════════════
# TRIANGLE INEQUALITY WORLD HOPS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    WorldMetric

Distance metric between worlds based on color.
"""
struct WorldMetric
    worlds::Vector{DerangeableWorld}
    distance_matrix::Matrix{Float64}
    
    # Triangle inequality violations
    violations::Vector{Tuple{Int, Int, Int, Float64}}  # (i, j, k, amount)
    
    is_metric::Bool
    is_ultrametric::Bool
    
    seed::UInt64
end

function WorldMetric(worlds::Vector{DerangeableWorld})
    n = length(worlds)
    dist = zeros(Float64, n, n)
    
    for i in 1:n
        for j in i+1:n
            d = color_distance(worlds[i].color, worlds[j].color)
            dist[i, j] = d
            dist[j, i] = d
        end
    end
    
    # Check triangle inequality
    violations = Tuple{Int, Int, Int, Float64}[]
    
    for i in 1:n
        for j in i+1:n
            for k in j+1:n
                # Check all three orderings
                v1 = dist[i, k] - (dist[i, j] + dist[j, k])
                v2 = dist[i, j] - (dist[i, k] + dist[k, j])
                v3 = dist[j, k] - (dist[j, i] + dist[i, k])
                
                if v1 > 1e-10
                    push!(violations, (i, j, k, v1))
                end
                if v2 > 1e-10
                    push!(violations, (i, k, j, v2))
                end
                if v3 > 1e-10
                    push!(violations, (j, i, k, v3))
                end
            end
        end
    end
    
    is_metric = isempty(violations)
    
    # Check ultrametric: d(x,z) ≤ max(d(x,y), d(y,z))
    is_ultra = true
    for i in 1:n
        for j in i+1:n
            for k in j+1:n
                if dist[i, k] > max(dist[i, j], dist[j, k]) + 1e-10
                    is_ultra = false
                    break
                end
            end
            !is_ultra && break
        end
        !is_ultra && break
    end
    
    combined_seed = reduce(⊻, [w.seed for w in worlds]; init=GAY_SEED)
    WorldMetric(worlds, dist, violations, is_metric, is_ultra, combined_seed)
end

"""
    TriangleCheck

Result of checking triangle inequality for a triple of worlds.
"""
struct TriangleCheck
    worlds::NTuple{3, Int}  # World indices
    distances::NTuple{3, Float64}  # (d12, d13, d23)
    
    satisfied::Bool
    violation_amount::Float64
    
    # Mining estimate: information extracted
    estimate::Float64
end

function check_triangle!(metric::WorldMetric, i::Int, j::Int, k::Int)::TriangleCheck
    d12 = metric.distance_matrix[i, j]
    d13 = metric.distance_matrix[i, k]
    d23 = metric.distance_matrix[j, k]
    
    # Check: d13 ≤ d12 + d23
    violation = max(0.0, d13 - d12 - d23)
    
    # Mining estimate: entropy from distances
    total = d12 + d13 + d23
    if total > 0
        p12, p13, p23 = d12/total, d13/total, d23/total
        estimate = -sum(p * log2(p + 1e-10) for p in [p12, p13, p23])
    else
        estimate = 0.0
    end
    
    TriangleCheck((i, j, k), (d12, d13, d23), violation < 1e-10, violation, estimate)
end

"""
    WorldHop

A hop between worlds with associated cost.
"""
struct WorldHop
    from_world::Int
    to_world::Int
    cost::Float64  # Color distance
    
    # Triangle checks along the way
    triangles_satisfied::Int
    triangles_violated::Int
    
    seed::UInt64
end

function hop_cost(metric::WorldMetric, from::Int, to::Int)::Float64
    metric.distance_matrix[from, to]
end

function optimal_hop_path(metric::WorldMetric, from::Int, to::Int)::Vector{WorldHop}
    # Find shortest path (Dijkstra-like)
    n = length(metric.worlds)
    dist = fill(Inf, n)
    prev = fill(-1, n)
    dist[from] = 0.0
    
    unvisited = Set(1:n)
    
    while !isempty(unvisited)
        # Find minimum
        min_dist = Inf
        current = -1
        for v in unvisited
            if dist[v] < min_dist
                min_dist = dist[v]
                current = v
            end
        end
        
        current == -1 && break
        delete!(unvisited, current)
        
        current == to && break
        
        for neighbor in unvisited
            alt = dist[current] + metric.distance_matrix[current, neighbor]
            if alt < dist[neighbor]
                dist[neighbor] = alt
                prev[neighbor] = current
            end
        end
    end
    
    # Reconstruct path
    path = WorldHop[]
    current = to
    
    while prev[current] != -1
        from_w = prev[current]
        cost = metric.distance_matrix[from_w, current]
        
        push!(path, WorldHop(from_w, current, cost, 0, 0,
                             splitmix64(metric.seed ⊻ UInt64(from_w) ⊻ UInt64(current))))
        current = from_w
    end
    
    reverse!(path)
    path
end

function is_ultrametric(metric::WorldMetric)::Bool
    metric.is_ultrametric
end

function metric_graph(metric::WorldMetric)::Vector{Tuple{Int, Int, Float64}}
    # Return edges as (i, j, distance)
    n = length(metric.worlds)
    edges = Tuple{Int, Int, Float64}[]
    
    for i in 1:n
        for j in i+1:n
            push!(edges, (i, j, metric.distance_matrix[i, j]))
        end
    end
    
    sort!(edges; by=e -> e[3])
    edges
end

# ═══════════════════════════════════════════════════════════════════════════════
# MAXENT DISSONANCE MONOPOLE ZONES
# ═══════════════════════════════════════════════════════════════════════════════

"""
    DissonanceField

Field of color dissonance over the arena.
"""
mutable struct DissonanceField
    grid_size::Int
    values::Matrix{Float64}  # Dissonance at each grid point
    
    # MaxEnt regions
    high_entropy_threshold::Float64
    
    seed::UInt64
end

function DissonanceField(arena::Arena; grid_size::Int=32, threshold::Float64=0.8)
    values = zeros(Float64, grid_size, grid_size)
    DissonanceField(grid_size, values, threshold, arena.seed)
end

"""
    Monopole

A monopole in the dissonance field.
"""
struct Monopole
    position::Tuple{Int, Int}  # Grid position
    charge::Float64  # Dissonance charge (can be negative)
    
    # Influence radius
    radius::Float64
    
    seed::UInt64
    color::NamedTuple{(:r, :g, :b), NTuple{3, Float64}}
end

"""
    MonopoleZone

A zone around a monopole with high MaxEnt dissonance.
"""
struct MonopoleZone
    center::Monopole
    boundary::Vector{Tuple{Int, Int}}
    
    # Zone properties
    area::Int  # Number of grid cells
    total_entropy::Float64
    average_dissonance::Float64
    
    # Mining potential
    estimate_value::Float64
    
    seed::UInt64
end

function compute_dissonance!(field::DissonanceField, worlds::Vector{DerangeableWorld})
    n = field.grid_size
    
    for i in 1:n
        for j in 1:n
            # Grid position in [0,1] × [0,1]
            x = (i - 0.5) / n
            y = (j - 0.5) / n
            
            # Compute dissonance from all world colors
            total_dissonance = 0.0
            
            for (k, world) in enumerate(worlds)
                # World position based on its seed
                wx = (world.seed >> 48) / 65535.0
                wy = ((world.seed >> 32) & 0xFFFF) / 65535.0
                
                # Distance to world
                dist = sqrt((x - wx)^2 + (y - wy)^2)
                
                # Color-based dissonance (inversely weighted by distance)
                color_intensity = (world.color.r + world.color.g + world.color.b) / 3
                dissonance = color_intensity / (1 + dist * 10)
                
                total_dissonance += dissonance
            end
            
            field.values[i, j] = total_dissonance
        end
    end
    
    # Normalize to [0, 1]
    max_val = maximum(field.values)
    if max_val > 0
        field.values ./= max_val
    end
    
    field.seed = splitmix64(field.seed)
    field
end

function find_monopoles(field::DissonanceField; min_charge::Float64=0.5)::Vector{Monopole}
    monopoles = Monopole[]
    n = field.grid_size
    
    # Find local maxima
    for i in 2:n-1
        for j in 2:n-1
            val = field.values[i, j]
            
            # Check if local maximum
            is_max = true
            for di in -1:1
                for dj in -1:1
                    if di == 0 && dj == 0
                        continue
                    end
                    if field.values[i+di, j+dj] >= val
                        is_max = false
                        break
                    end
                end
                !is_max && break
            end
            
            if is_max && val >= min_charge
                # Calculate charge (positive for high dissonance)
                charge = val - 0.5
                
                # Estimate radius
                radius = 1.0
                for r in 1:min(i-1, j-1, n-i, n-j)
                    avg_ring = 0.0
                    count = 0
                    for di in -r:r
                        for dj in -r:r
                            if abs(di) == r || abs(dj) == r
                                avg_ring += field.values[i+di, j+dj]
                                count += 1
                            end
                        end
                    end
                    avg_ring /= count
                    if avg_ring < val * 0.5
                        radius = Float64(r)
                        break
                    end
                end
                
                seed = splitmix64(field.seed ⊻ UInt64(i) ⊻ (UInt64(j) << 16))
                push!(monopoles, Monopole((i, j), charge, radius, seed, color_from_seed(seed)))
            end
        end
    end
    
    monopoles
end

function mine_estimates(field::DissonanceField, monopoles::Vector{Monopole})::Vector{MonopoleZone}
    zones = MonopoleZone[]
    n = field.grid_size
    
    for monopole in monopoles
        i, j = monopole.position
        r = ceil(Int, monopole.radius)
        
        boundary = Tuple{Int, Int}[]
        area = 0
        total_ent = 0.0
        total_diss = 0.0
        
        for di in -r:r
            for dj in -r:r
                ni, nj = i + di, j + dj
                if 1 <= ni <= n && 1 <= nj <= n
                    dist = sqrt(Float64(di^2 + dj^2))
                    if dist <= monopole.radius
                        area += 1
                        val = field.values[ni, nj]
                        total_diss += val
                        
                        # Entropy contribution
                        if val > 0 && val < 1
                            total_ent -= val * log2(val) - (1-val) * log2(1-val)
                        end
                        
                        # Boundary detection
                        if dist > monopole.radius - 1
                            push!(boundary, (ni, nj))
                        end
                    end
                end
            end
        end
        
        avg_diss = area > 0 ? total_diss / area : 0.0
        estimate = total_ent * avg_diss  # Mining value
        
        push!(zones, MonopoleZone(monopole, boundary, area, total_ent, avg_diss,
                                   estimate, monopole.seed))
    end
    
    zones
end

function zone_entropy(zone::MonopoleZone)::Float64
    zone.total_entropy
end

function monopole_charge(monopole::Monopole)::Float64
    monopole.charge
end

# ═══════════════════════════════════════════════════════════════════════════════
# ERGODIC CONVERGENCE
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ErgodicChain

Markov chain for ergodic analysis of world transitions.
"""
mutable struct ErgodicChain
    n_states::Int
    transition_matrix::Matrix{Float64}
    
    current_state::Int
    history::Vector{Int}
    
    # Convergence tracking
    visit_counts::Vector{Int}
    
    seed::UInt64
end

function ErgodicChain(metric::WorldMetric)
    n = length(metric.worlds)
    
    # Transition probabilities from distance (closer = more likely)
    trans = zeros(Float64, n, n)
    
    for i in 1:n
        total = 0.0
        for j in 1:n
            if i != j
                # Probability inversely proportional to distance
                prob = 1.0 / (1.0 + metric.distance_matrix[i, j])
                trans[i, j] = prob
                total += prob
            end
        end
        
        # Normalize
        if total > 0
            for j in 1:n
                trans[i, j] /= total
            end
        else
            # Uniform if isolated
            for j in 1:n
                trans[i, j] = 1.0 / n
            end
        end
    end
    
    ErgodicChain(n, trans, 1, Int[], zeros(Int, n), metric.seed)
end

function step!(chain::ErgodicChain)
    chain.seed = splitmix64(chain.seed)
    
    # Sample next state
    r = (chain.seed >> 48) / 65535.0
    
    cumsum = 0.0
    next_state = chain.current_state
    
    for j in 1:chain.n_states
        cumsum += chain.transition_matrix[chain.current_state, j]
        if r < cumsum
            next_state = j
            break
        end
    end
    
    chain.current_state = next_state
    push!(chain.history, next_state)
    chain.visit_counts[next_state] += 1
    
    next_state
end

function convergence_rate(chain::ErgodicChain)::Float64
    # Compute second-largest eigenvalue magnitude
    # This determines mixing time
    eigvals = eigen(chain.transition_matrix).values
    sorted = sort(abs.(eigvals); rev=true)
    
    length(sorted) >= 2 ? sorted[2] : 0.0
end

function mixing_time(chain::ErgodicChain; epsilon::Float64=0.01)::Int
    # Approximate mixing time: t_mix ≈ log(1/ε) / (1 - λ₂)
    λ2 = convergence_rate(chain)
    
    if λ2 >= 1.0 - 1e-10
        return typemax(Int)  # Never mixes
    end
    
    ceil(Int, log(1/epsilon) / (1 - λ2))
end

function stationary_distribution(chain::ErgodicChain)::Vector{Float64}
    # Power iteration to find stationary distribution
    n = chain.n_states
    dist = fill(1.0/n, n)
    
    for _ in 1:1000
        new_dist = chain.transition_matrix' * dist
        
        # Check convergence
        if maximum(abs.(new_dist - dist)) < 1e-10
            return new_dist
        end
        
        dist = new_dist
    end
    
    dist
end

function ergodic_color(chain::ErgodicChain, worlds::Vector{DerangeableWorld})::NamedTuple
    dist = stationary_distribution(chain)
    
    r = sum(dist[i] * worlds[i].color.r for i in 1:chain.n_states)
    g = sum(dist[i] * worlds[i].color.g for i in 1:chain.n_states)
    b = sum(dist[i] * worlds[i].color.b for i in 1:chain.n_states)
    
    (r=r, g=g, b=b)
end

# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

function demo_derangeable_arena()
    println("╔═══════════════════════════════════════════════════════════════════════════╗")
    println("║  DERANGEABLE ARENA: 3-Adversarial Self-Same in Ergodic World Hops        ║")
    println("╚═══════════════════════════════════════════════════════════════════════════╝")
    println()
    
    # 1. Derangements
    println("─── DERANGEMENTS ───")
    for n in 2:5
        ds = generate_derangements(n)
        count = derangement_count(n)
        println("  D($n) = $count derangements")
        if n <= 3
            for d in ds
                println("    $(d.permutation) cycles=$(d.cycles)")
            end
        end
    end
    println()
    
    # 2. Arena setup
    println("─── ARENA SETUP ───")
    arena = Arena(5, 4; n_agents=3)
    println("  Worlds: $(length(arena.worlds))")
    println("  Agents: $(length(arena.agents))")
    used, total = arena_bandwidth(arena)
    println("  Bandwidth: $used / $total")
    println()
    
    # 3. Derange the arena
    println("─── DERANGE ARENA ───")
    try
        d = derange_arena!(arena)
        println("  Applied derangement: $(d.permutation)")
        r, g, b = round.(Int, [arena.color.r, arena.color.g, arena.color.b] .* 255)
        println("  New arena color: \e[38;2;$(r);$(g);$(b)m████\e[0m")
    catch e
        println("  Error: $e")
    end
    println()
    
    # 4. 3-Adversarial game
    println("─── 3-ADVERSARIAL SELF-SAME GAME ───")
    game = SelfSameGame(arena.agents)
    
    for round in 1:10
        play_round!(game)
    end
    
    println("  Played $(game.round) rounds")
    for (i, agent) in enumerate(game.tri_adversary.agents)
        println("  Agent $i: believed=$(agent.believed_role), conf=$(round(agent.confidence, digits=2))")
        println("    Probs: O=$(round(agent.p_originary, digits=2)), D=$(round(agent.p_derived, digits=2)), X=$(round(agent.p_deranged, digits=2))")
    end
    
    erg_limit = ergodic_limit(game)
    println("  Ergodic limits:")
    for (id, dist) in erg_limit
        println("    Agent $id: O=$(round(dist[:originary], digits=2)), D=$(round(dist[:derived], digits=2)), X=$(round(dist[:deranged], digits=2))")
    end
    println()
    
    # 5. Triangle inequality world hops
    println("─── TRIANGLE INEQUALITY WORLD HOPS ───")
    metric = WorldMetric(arena.worlds)
    
    println("  Is metric: $(metric.is_metric)")
    println("  Is ultrametric: $(metric.is_ultrametric)")
    println("  Violations: $(length(metric.violations))")
    
    if length(arena.worlds) >= 3
        check = check_triangle!(metric, 1, 2, 3)
        println("  Triangle (1,2,3): satisfied=$(check.satisfied), estimate=$(round(check.estimate, digits=3))")
    end
    
    # Optimal path
    if length(arena.worlds) >= 2
        path = optimal_hop_path(metric, 1, length(arena.worlds))
        total_cost = sum(h.cost for h in path)
        println("  Path 1→$(length(arena.worlds)): $(length(path)) hops, total cost=$(round(total_cost, digits=3))")
    end
    println()
    
    # 6. Dissonance field and monopoles
    println("─── MAXENT DISSONANCE MONOPOLES ───")
    field = DissonanceField(arena; grid_size=16)
    compute_dissonance!(field, arena.worlds)
    
    monopoles = find_monopoles(field; min_charge=0.3)
    println("  Grid: $(field.grid_size)×$(field.grid_size)")
    println("  Monopoles found: $(length(monopoles))")
    
    if !isempty(monopoles)
        zones = mine_estimates(field, monopoles)
        
        for (i, zone) in enumerate(zones[1:min(3, end)])
            m = zone.center
            r, g, b = round.(Int, [m.color.r, m.color.g, m.color.b] .* 255)
            println("  Zone $i: pos=$(m.position), charge=$(round(m.charge, digits=2)), estimate=$(round(zone.estimate_value, digits=3)) \e[38;2;$(r);$(g);$(b)m●\e[0m")
        end
    end
    println()
    
    # 7. Ergodic chain
    println("─── ERGODIC CONVERGENCE ───")
    chain = ErgodicChain(metric)
    
    for _ in 1:100
        step!(chain)
    end
    
    λ2 = convergence_rate(chain)
    t_mix = mixing_time(chain)
    
    println("  Steps taken: $(length(chain.history))")
    println("  Convergence rate (λ₂): $(round(λ2, digits=4))")
    println("  Mixing time: $t_mix")
    
    stat_dist = stationary_distribution(chain)
    println("  Stationary distribution: $(round.(stat_dist, digits=3))")
    
    erg_color = ergodic_color(chain, arena.worlds)
    r, g, b = round.(Int, [erg_color.r, erg_color.g, erg_color.b] .* 255)
    println("  Ergodic color: \e[38;2;$(r);$(g);$(b)m████████\e[0m")
    println()
    
    println("═══════════════════════════════════════════════════════════════════════════")
    println("  No fixed points. Every world moves. Every role uncertain.")
    println("  ArenaIndeterminacyError → expand color bandwidth")
    println("  Triangle inequality → mine estimates from world hops")
    println("═══════════════════════════════════════════════════════════════════════════")
    
    (arena=arena, game=game, metric=metric, monopoles=monopoles, chain=chain)
end

end # module
