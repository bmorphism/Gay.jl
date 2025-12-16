#=
GAMES.JL - Coordination Games with Color-Information Theory

Extends Gay.jl with game-theoretic framework:
- Selfish task allocation (coordination models)
- Möbius inversion on game boundaries  
- Random walk mixing with color observation
- Markov Blanket verification

Seed 1069 integration with SplittableRandoms.
Information as force, not consciousness.
=#

module Games

using SplittableRandoms
using Random
using StatsBase

export Player, Facility, CoordinationModel
export ChromaticState, compute_cost, social_cost, makespan
export game_state_to_address, moebius_function
export facility_transition_matrix, eg_walk_step, eg_walker_colors
export compute_markov_blankets, verify_blanket_structure
export price_of_anarchy

# ═══════════════════════════════════════════════════════════════════════════════
# COORDINATION MODEL (Selfish Task Allocation)
# ═══════════════════════════════════════════════════════════════════════════════

"""
Player in coordination game.
- load: work quantity
- current_facility: strategy (which facility to use)
- cost: finish time on chosen facility
"""
struct Player
    id::Int
    load::Float64
    current_facility::Int
    cost::Float64
end

"""
Facility in coordination model.
- assigned_players: players using this facility
- scheduling_policy: :FIFO, :DELAY, :SPF
- delay_factor: how much to delay [0, 1]
"""
struct Facility
    id::Int
    assigned_players::Vector{Int}
    scheduling_policy::Symbol
    delay_factor::Float64
end

"""
Coordination Model: (N, M, Σ_i, c^j)
Implements selfish task allocation with cost functions.
"""
struct CoordinationModel
    players::Vector{Player}
    facilities::Vector{Facility}
    n::Int  # Number of players
    m::Int  # Number of facilities
end

"""Create initial coordination model (unweighted congestion)"""
function CoordinationModel(n::Int, m::Int)
    players = [
        Player(i, 1.0, mod(i - 1, m) + 1, 0.0)
        for i in 1:n
    ]

    facilities = [
        Facility(j, Int[], :FIFO, 0.1)
        for j in 1:m
    ]

    for (i, player) in enumerate(players)
        j = player.current_facility
        push!(facilities[j].assigned_players, i)
    end

    CoordinationModel(players, facilities, n, m)
end

"""Compute cost on facility for a player"""
function compute_cost(
    facility::Facility,
    player_id::Int,
    all_loads::Vector{Float64}
)::Float64
    assigned_loads = [
        all_loads[pid] for pid in facility.assigned_players
        if pid <= length(all_loads)
    ]

    isempty(assigned_loads) && return 0.0

    base_cost = sum(assigned_loads)

    delay = if facility.scheduling_policy == :FIFO
        pos = findfirst(==(player_id), facility.assigned_players)
        pos !== nothing ? pos * facility.delay_factor : 0.0
    elseif facility.scheduling_policy == :DELAY
        abs(sin(player_id * 0.1 + 1069)) * facility.delay_factor
    else  # :SPF
        pi = findfirst(==(player_id), facility.assigned_players)
        pi !== nothing ? (length(facility.assigned_players) - pi + 1) * facility.delay_factor : 0.0
    end

    return base_cost + delay
end

"""Social cost: sum of all player finish times"""
function social_cost(model::CoordinationModel)::Float64
    all_loads = [p.load for p in model.players]
    total = 0.0
    for (j, fac) in enumerate(model.facilities)
        for pid in fac.assigned_players
            total += compute_cost(fac, pid, all_loads)
        end
    end
    return total
end

"""Makespan: maximum finish time"""
function makespan(model::CoordinationModel)::Float64
    all_loads = [p.load for p in model.players]
    max_cost = 0.0
    for (j, fac) in enumerate(model.facilities)
        for pid in fac.assigned_players
            max_cost = max(max_cost, compute_cost(fac, pid, all_loads))
        end
    end
    return max_cost
end

# ═══════════════════════════════════════════════════════════════════════════════
# CHROMATIC STATE (Color observation from game state)
# ═══════════════════════════════════════════════════════════════════════════════

"""
Chromatic state: color encoding of game configuration.
Uses SplittableRandoms to generate reproducible colors from game state.
"""
struct ChromaticState
    address::Int
    color_rgb::Tuple{Float64, Float64, Float64}
    stability::Float64  # How stable is this configuration
end

"""Derive chromatic address from game state"""
function game_state_to_address(
    model::CoordinationModel,
    player_id::Int
)::Int
    facility_id = model.players[player_id].current_facility
    congestion = length(model.facilities[facility_id].assigned_players)
    addr = facility_id + congestion * model.m
    return mod(addr, 2187)
end

"""Sample color from splittable random stream seeded by game address"""
function chromatic_state(
    model::CoordinationModel,
    player_id::Int
)::ChromaticState
    address = game_state_to_address(model, player_id)
    
    # Use SplittableRandoms seeded by address
    rng = SplittableRandom(address)
    
    # Sample RGB from perceptual color space
    r = rand(rng)
    g = rand(rng)
    b = rand(rng)
    
    # Stability: how many players share facility
    fac_id = model.players[player_id].current_facility
    stability = 1.0 / max(length(model.facilities[fac_id].assigned_players), 1)
    
    ChromaticState(address, (r, g, b), stability)
end

# ═══════════════════════════════════════════════════════════════════════════════
# MÖBIUS INVERSION (Boundary Paradox-Indexing)
# ═══════════════════════════════════════════════════════════════════════════════

"""
Möbius function on poset of facility coalitions.
μ(S, T) = 1 if S = T
        = (-1)^(|T| - |S|) if S ⊂ T (proper)
        = 0 otherwise
"""
function moebius_function(
    coalition1::Set{Int},
    coalition2::Set{Int}
)::Int
    if coalition1 == coalition2
        return 1
    elseif issubset(coalition1, coalition2)
        return (-1)^(length(coalition2) - length(coalition1))
    else
        return 0
    end
end

"""Invert coalition preferences via Möbius inversion formula"""
function invert_preferences(
    cooperation_values::Dict{Set{Int}, Float64}
)::Dict{Set{Int}, Float64}
    inverted = Dict{Set{Int}, Float64}()
    coalitions = collect(keys(cooperation_values))

    for s in coalitions
        inverted[s] = 0.0
        for t in coalitions
            μ = moebius_function(s, t)
            if μ != 0
                inverted[s] += μ * cooperation_values[t]
            end
        end
    end

    return inverted
end

# ═══════════════════════════════════════════════════════════════════════════════
# RANDOM WALK MIXING (Eg-Walker on Facility Graph)
# ═══════════════════════════════════════════════════════════════════════════════

"""Transition matrix for random walk on facilities"""
function facility_transition_matrix(
    model::CoordinationModel,
    temperature::Float64=1.0
)::Matrix{Float64}
    P = zeros(Float64, model.m, model.m)

    for j in 1:model.m
        congestion = length(model.facilities[j].assigned_players) / model.n

        for k in 1:model.m
            if j == k
                P[k, j] = 0.5 / max(congestion, 0.1)
            else
                congestion_k = length(model.facilities[k].assigned_players) / model.n
                cost_diff = congestion_k - congestion
                P[k, j] = (0.5 / (model.m - 1)) * exp(-cost_diff / temperature)
            end
        end

        P[:, j] ./= sum(P[:, j])
    end

    return P
end

"""Execute one step of facility random walk"""
function eg_walk_step(
    model::CoordinationModel,
    player_id::Int,
    transition_matrix::Matrix{Float64}
)::CoordinationModel
    current_facility = model.players[player_id].current_facility
    probs = transition_matrix[:, current_facility]
    next_facility = sample(1:model.m, Weights(probs))

    # Create new players array with updated player
    new_players = [model.players[i] for i in 1:length(model.players)]
    new_players[player_id] = Player(
        player_id,
        new_players[player_id].load,
        next_facility,
        0.0
    )

    # Create new facilities array with updated assignments
    new_facilities = [model.facilities[i] for i in 1:length(model.facilities)]
    old_fac = new_facilities[current_facility]
    new_facilities[current_facility] = Facility(
        old_fac.id,
        filter(x -> x != player_id, old_fac.assigned_players),
        old_fac.scheduling_policy,
        old_fac.delay_factor
    )

    new_fac = new_facilities[next_facility]
    new_facilities[next_facility] = Facility(
        new_fac.id,
        [new_fac.assigned_players..., player_id],
        new_fac.scheduling_policy,
        new_fac.delay_factor
    )

    return CoordinationModel(new_players, new_facilities, model.n, model.m)
end

"""Run Eg-Walker for multiple steps, tracking chromatic states"""
function eg_walker_colors(
    model::CoordinationModel,
    steps::Int
)::Tuple{CoordinationModel, Vector{ChromaticState}}
    current = model
    states = ChromaticState[]
    P = facility_transition_matrix(current)

    for _ in 1:steps
        player_id = sample(1:current.n)
        current = eg_walk_step(current, player_id, P)
        push!(states, chromatic_state(current, player_id))
    end

    return current, states
end

# ═══════════════════════════════════════════════════════════════════════════════
# MARKOV BLANKET VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

"""
Markov Blanket of a player: minimal set of other players whose 
strategies directly affect this player's cost.

Conditional Independence: C_i ⊥⊥ (all others) | MB(i)
"""
function compute_markov_blankets(model::CoordinationModel)::Dict{Int, Set{Int}}
    blankets = Dict{Int, Set{Int}}()

    for i in 1:model.n
        facility_i = model.players[i].current_facility
        blanket = Set{Int}()

        for pid in model.facilities[facility_i].assigned_players
            if pid != i
                push!(blanket, pid)
            end
        end

        blankets[i] = blanket
    end

    return blankets
end

"""Verify conditional independence for entire model"""
function verify_blanket_structure(model::CoordinationModel)::Dict{Int, Bool}
    blankets = compute_markov_blankets(model)
    verification = Dict{Int, Bool}()

    for i in 1:model.n
        facility_i = model.players[i].current_facility
        expected_size = length(model.facilities[facility_i].assigned_players) - 1
        verification[i] = length(blankets[i]) == expected_size
    end

    return verification
end

# ═══════════════════════════════════════════════════════════════════════════════
# PRICE OF ANARCHY (Selfish vs Optimal)
# ═══════════════════════════════════════════════════════════════════════════════

"""Price of Anarchy: ratio of selfish to optimal makespan"""
function price_of_anarchy(
    selfish_model::CoordinationModel,
    optimal_model::CoordinationModel
)::Float64
    selfish = makespan(selfish_model)
    optimal = makespan(optimal_model)
    optimal == 0.0 ? 1.0 : selfish / optimal
end

end # module Games
