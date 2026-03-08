# narrative_world.jl
# Bumpus Narratives + Open Games + Persistent World REPL
#
# Sheaves on time categories for compositional temporal reasoning.
# Every interaction becomes a play action with tracked equilibrium paths.
#
# Key concepts:
#   - Narrative: Sheaf F: I_N → D with gluing condition
#   - Interval: [a,b] with presheaf structure on containment
#   - World: Persistent game state with history tracking
#   - Interaction: Single play action with full audit trail
#
# Integration:
#   - Uses GayOpenGame's Play/CoPlay bidirectional structure
#   - Trit values from GF(3) conservation
#   - SplitMix64 fingerprints for deterministic identity

module NarrativeWorld

using Random

# Import from sibling module when available
# For now, define compatible types locally

# ============================================================================
# CORE TYPES: INTERVALS AND NARRATIVES
# ============================================================================

"""
Interval: A time interval [a,b] in the interval category I_N.
Morphisms are containment: [a,b] → [c,d] iff c ≤ a ≤ b ≤ d.
"""
struct Interval
    start::Int
    stop::Int
    fingerprint::UInt64
    
    function Interval(a::Int, b::Int; seed::UInt64 = UInt64(0))
        @assert a ≤ b "Interval start must be ≤ stop"
        fp = sm64(seed ⊻ UInt64(a) ⊻ sm64(UInt64(b)))
        new(a, b, fp)
    end
end

"""
Check if interval I contains interval J (morphism I → J in I_N^op).
"""
contains(I::Interval, J::Interval) = I.start ≤ J.start && J.stop ≤ I.stop

"""
Point interval [p,p] - represents a moment in time.
"""
point_interval(p::Int; seed::UInt64 = UInt64(0)) = Interval(p, p; seed=seed)

"""
Narrative: A sheaf F: I_N → D over the interval category.

The gluing axiom states:
  F([a,b]) ≅ F([a,p]) ×_{F([p,p])} F([p,b])

This means a narrative over [a,b] is determined by its restrictions
to [a,p] and [p,b] that agree at the point p.
"""
struct Narrative
    name::Symbol
    intervals::Vector{Interval}
    values::Dict{Interval, Any}  # F(I) for each interval
    fingerprint::UInt64
end

"""
Create empty narrative.
"""
function Narrative(name::Symbol; seed::UInt64 = UInt64(137508))
    fp = sm64(seed ⊻ sm64(hash(name)))
    Narrative(name, Interval[], Dict{Interval, Any}(), fp)
end

"""
Restriction map: F([a,b]) → F([c,d]) when [a,b] ⊇ [c,d].
"""
function restrict(N::Narrative, from::Interval, to::Interval)
    @assert contains(from, to) "Cannot restrict: $from does not contain $to"
    # In a sheaf, restriction is projection/forgetting
    get(N.values, to, get(N.values, from, nothing))
end

"""
Glue narratives at a point: F([a,p]) ×_{F(p)} F([p,b]) → F([a,b]).
This is the sheaf gluing condition.
"""
function glue(N::Narrative, left::Interval, right::Interval, point::Int)
    @assert left.stop == point "Left interval must end at gluing point"
    @assert right.start == point "Right interval must start at gluing point"
    
    left_val = get(N.values, left, nothing)
    right_val = get(N.values, right, nothing)
    
    # The glued interval
    glued = Interval(left.start, right.stop; seed=N.fingerprint)
    
    # Glued value: combine left and right (here: tuple)
    glued_val = (left=left_val, right=right_val, point=point)
    
    N.values[glued] = glued_val
    push!(N.intervals, glued)
    
    glued_val
end

# ============================================================================
# SPLITMIX64 (LOCAL COPY FOR MODULE INDEPENDENCE)
# ============================================================================

@inline function sm64(x::UInt64)::UInt64
    z = x + 0x9e3779b97f4a7c15
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    z ⊻ (z >> 31)
end

# ============================================================================
# OPEN GAME TYPES (SIMPLIFIED, COMPATIBLE WITH GAY_OPEN_GAME)
# ============================================================================

@enum Trit::Int8 MINUS=-1 ERGODIC=0 PLUS=1

"""
OpenGame: A game G: (X,S) → (Y,R).
- X: observation type (forward input)
- Y: action type (forward output)  
- R: utility type (backward input)
- S: state type (backward output)
- Σ: strategy space
"""
struct OpenGame
    name::Symbol
    play::Function      # (Σ, X) → Y
    coplay::Function    # (Σ, X, Y, R) → S
    strategies::Vector{Symbol}
    trit::Trit
    fingerprint::UInt64
end

"""
Create an open game from play/coplay functions.
"""
function OpenGame(
    name::Symbol,
    play::Function,
    coplay::Function,
    strategies::Vector{Symbol};
    seed::UInt64 = UInt64(137508)
)
    fp = sm64(seed ⊻ sm64(hash(name)))
    trit = Trit(Int(fp % 3) - 1)
    OpenGame(name, play, coplay, strategies, trit, fp)
end

"""
Compose games sequentially: G ; H (G then H).
"""
function compose_seq(G::OpenGame, H::OpenGame)
    # Sequential: G's output becomes H's input
    function composed_play(σ, x)
        y_g = G.play(σ, x)
        H.play(σ, y_g)
    end
    
    function composed_coplay(σ, x, y, r)
        y_g = G.play(σ, x)
        s_h = H.coplay(σ, y_g, y, r)
        G.coplay(σ, x, y_g, s_h)
    end
    
    strategies = unique(vcat(G.strategies, H.strategies))
    fp = sm64(G.fingerprint ⊻ H.fingerprint)
    trit = Trit((Int(G.trit) + Int(H.trit)) % 3)
    
    OpenGame(
        Symbol(G.name, :_then_, H.name),
        composed_play,
        composed_coplay,
        strategies,
        trit,
        fp
    )
end

"""
Compose games in parallel: G ⊗ H (G and H simultaneously).
"""
function compose_par(G::OpenGame, H::OpenGame)
    function composed_play(σ, x)
        (G.play(σ, x[1]), H.play(σ, x[2]))
    end
    
    function composed_coplay(σ, x, y, r)
        (G.coplay(σ, x[1], y[1], r[1]), H.coplay(σ, x[2], y[2], r[2]))
    end
    
    strategies = unique(vcat(G.strategies, H.strategies))
    fp = sm64(G.fingerprint ⊻ sm64(H.fingerprint))
    trit = Trit((Int(G.trit) + Int(H.trit)) % 3)
    
    OpenGame(
        Symbol(G.name, :_par_, H.name),
        composed_play,
        composed_coplay,
        strategies,
        trit,
        fp
    )
end

# Operators
⊗(G::OpenGame, H::OpenGame) = compose_par(G, H)
∘(G::OpenGame, H::OpenGame) = compose_seq(H, G)  # Note: reversed for math convention

# ============================================================================
# INTERACTION: EVERY-PLAY-AS-ACTION
# ============================================================================

"""
Interaction: A single play action with full audit trail.
Records timestamp, state, strategy, move, utility, and coutility.
"""
struct Interaction
    timestamp::Int
    state::Symbol
    strategy::Symbol
    move::Symbol
    utility::Float64
    coutility::Float64
    interval::Interval
    fingerprint::UInt64
end

"""
Create an interaction from current state.
"""
function Interaction(
    timestamp::Int,
    state::Symbol,
    strategy::Symbol,
    move::Symbol,
    utility::Float64,
    coutility::Float64;
    seed::UInt64 = UInt64(0)
)
    interval = Interval(timestamp, timestamp; seed=seed)
    fp = sm64(seed ⊻ UInt64(timestamp) ⊻ sm64(hash(move)))
    Interaction(timestamp, state, strategy, move, utility, coutility, interval, fp)
end

# ============================================================================
# WORLD: PERSISTENT GAME STATE WITH HISTORY
# ============================================================================

"""
World: Persistent game state with history tracking and equilibrium detection.

Integrates:
- Open game structure (play/coplay)
- Narrative sheaf (intervals, gluing)
- Every-interaction tracking
"""
mutable struct World
    name::Symbol
    game::OpenGame
    state::Symbol
    history::Vector{Interaction}
    narrative::Narrative
    cumulative_utility::Float64
    step::Int
    seed::UInt64
    fingerprint::UInt64
end

"""
Create a new World with an open game.
"""
function World(name::Symbol, game::OpenGame; seed::UInt64 = UInt64(137508))
    fp = sm64(seed ⊻ sm64(hash(name)) ⊻ game.fingerprint)
    narrative = Narrative(name; seed=fp)
    World(name, game, :start, Interaction[], narrative, 0.0, 0, seed, fp)
end

"""
Play a move in the world, recording the interaction.
"""
function play!(world::World, strategy::Symbol)
    world.step += 1
    
    # Execute forward pass
    move = world.game.play(strategy, world.state)
    
    # Execute backward pass to get utility
    utility = world.game.coplay(strategy, world.state, move, 0.0)
    
    # Compute coutility (opponent's perspective with opposite strategy)
    opponent_strategy = strategy == :Cooperate ? :Defect : :Cooperate
    coutility = world.game.coplay(opponent_strategy, world.state, move, 0.0)
    
    # Create interaction record
    interaction = Interaction(
        world.step,
        world.state,
        strategy,
        move,
        utility,
        coutility;
        seed=world.fingerprint
    )
    
    # Update world
    push!(world.history, interaction)
    world.cumulative_utility += utility
    world.state = move
    world.fingerprint = sm64(world.fingerprint ⊻ interaction.fingerprint)
    
    # Update narrative sheaf
    world.narrative.values[interaction.interval] = interaction
    push!(world.narrative.intervals, interaction.interval)
    
    # Create spanning intervals for sheaf gluing
    # Point intervals [t,t] glue via extended intervals [t-1, t] and [t, t]
    if length(world.history) ≥ 2
        prev_t = world.history[end-1].timestamp
        curr_t = world.history[end].timestamp
        
        # Create extended interval covering both points
        extended = Interval(prev_t, curr_t; seed=world.seed)
        world.narrative.values[extended] = (
            left = world.history[end-1],
            right = world.history[end],
            glued_at = prev_t
        )
        push!(world.narrative.intervals, extended)
    end
    
    interaction
end

"""
Get world status summary.
"""
function status(world::World)
    (
        name = world.name,
        state = world.state,
        step = world.step,
        cumulative_utility = world.cumulative_utility,
        history_length = length(world.history),
        fingerprint = string(world.fingerprint, base=16)
    )
end

"""
Compute equilibria for the game at current state.
Returns all strategy profiles that are Nash equilibria.
"""
function equilibria(world::World)
    eqs = Symbol[]
    
    for σ in world.game.strategies
        # Compute utility for this strategy
        move = world.game.play(σ, :neutral)
        utility = world.game.coplay(σ, :neutral, move, 0.0)
        
        # Check if any deviation is profitable
        is_equilibrium = true
        for σ′ in world.game.strategies
            if σ′ != σ
                move′ = world.game.play(σ′, :neutral)
                utility′ = world.game.coplay(σ′, :neutral, move′, 0.0)
                if utility′ > utility
                    is_equilibrium = false
                    break
                end
            end
        end
        
        if is_equilibrium
            push!(eqs, σ)
        end
    end
    
    eqs
end

"""
Check if the world's history follows an equilibrium path.
"""
function is_equilibrium_path(world::World)
    isempty(world.history) && return true
    
    eqs = equilibria(world)
    all(i -> i.strategy in eqs, world.history)
end

"""
Get narrative interval covering the entire history.
"""
function history_interval(world::World)
    isempty(world.history) && return nothing
    Interval(
        world.history[1].timestamp,
        world.history[end].timestamp;
        seed=world.seed
    )
end

"""
Restrict narrative to a sub-interval.
"""
function restrict_history(world::World, from::Int, to::Int)
    filter(i -> from ≤ i.timestamp ≤ to, world.history)
end

# ============================================================================
# STANDARD GAMES
# ============================================================================

"""
Prisoner's Dilemma as an open game.
"""
function prisoners_dilemma(; seed::UInt64 = UInt64(137508))
    # Payoff matrix: (my_move, opponent_move) → my_utility
    payoffs = Dict(
        (:Cooperate, :Cooperate) => -1.0,  # Both cooperate
        (:Cooperate, :Defect) => -3.0,     # I cooperate, they defect (sucker)
        (:Defect, :Cooperate) => 0.0,      # I defect, they cooperate (temptation)
        (:Defect, :Defect) => -2.0,        # Both defect
    )
    
    function play_pd(σ, x)
        # Strategy → Move
        σ
    end
    
    function coplay_pd(σ, x, y, r)
        # Compute utility assuming opponent plays Defect (Nash equilibrium assumption)
        opponent = :Defect
        get(payoffs, (y, opponent), -2.0)
    end
    
    OpenGame(
        :PrisonersDilemma,
        play_pd,
        coplay_pd,
        [:Cooperate, :Defect];
        seed=seed
    )
end

"""
Coordination game as an open game.
"""
function coordination_game(; seed::UInt64 = UInt64(137508))
    payoffs = Dict(
        (:A, :A) => 2.0,
        (:A, :B) => 0.0,
        (:B, :A) => 0.0,
        (:B, :B) => 1.0,
    )
    
    function play_coord(σ, x)
        σ
    end
    
    function coplay_coord(σ, x, y, r)
        # Both equilibria: (A,A) and (B,B)
        opponent = σ  # Assume coordination
        get(payoffs, (y, opponent), 0.0)
    end
    
    OpenGame(
        :Coordination,
        play_coord,
        coplay_coord,
        [:A, :B];
        seed=seed
    )
end

"""
Hawk-Dove game (Chicken).
"""
function hawk_dove(; seed::UInt64 = UInt64(137508), V::Float64 = 4.0, C::Float64 = 6.0)
    payoffs = Dict(
        (:Hawk, :Hawk) => (V - C) / 2,
        (:Hawk, :Dove) => V,
        (:Dove, :Hawk) => 0.0,
        (:Dove, :Dove) => V / 2,
    )
    
    function play_hd(σ, x)
        σ
    end
    
    function coplay_hd(σ, x, y, r)
        # Mixed strategy equilibrium exists
        opponent = σ == :Hawk ? :Dove : :Hawk
        get(payoffs, (y, opponent), 0.0)
    end
    
    OpenGame(
        :HawkDove,
        play_hd,
        coplay_hd,
        [:Hawk, :Dove];
        seed=seed
    )
end

# ============================================================================
# WORLD REPL INTERFACE
# ============================================================================

"""
Dictionary of active worlds for REPL access.
"""
const ACTIVE_WORLDS = Dict{Symbol, World}()

"""
    world(name::Symbol; game=:pd, seed=137508)

Create or access a persistent world.

# Examples
```julia
w = world(:alice)              # Creates Prisoner's Dilemma world
w = world(:bob, game=:coord)   # Creates Coordination game world
w = world(:alice)              # Returns existing world
```
"""
function world(name::Symbol; game::Symbol = :pd, seed::UInt64 = UInt64(137508))
    if haskey(ACTIVE_WORLDS, name)
        return ACTIVE_WORLDS[name]
    end
    
    g = if game == :pd
        prisoners_dilemma(; seed=seed)
    elseif game == :coord
        coordination_game(; seed=seed)
    elseif game == :hd || game == :hawk_dove
        hawk_dove(; seed=seed)
    else
        error("Unknown game: $game. Use :pd, :coord, or :hd")
    end
    
    w = World(name, g; seed=seed)
    ACTIVE_WORLDS[name] = w
    w
end

"""
Reset all worlds.
"""
function reset_worlds!()
    empty!(ACTIVE_WORLDS)
end

"""
List all active worlds.
"""
list_worlds() = collect(keys(ACTIVE_WORLDS))

# ============================================================================
# GAME DECOMPOSITION (STRUCTURED DECOMPOSITIONS BRIDGE)
# ============================================================================

"""
GameDecomposition: Adhesion-filtered decomposition of a game.

Connects to StructuredDecompositions.jl via:
- Bags: Sub-games
- Adhesions: Shared interfaces between sub-games
- Tree structure: Compositional hierarchy
"""
struct GameDecomposition
    bags::Vector{OpenGame}
    adhesions::Dict{Tuple{Int,Int}, Vector{Symbol}}  # Shared strategies
    tree_edges::Vector{Tuple{Int,Int}}
    fingerprint::UInt64
end

"""
Decompose a composite game into bags.
"""
function decompose_game(games::Vector{OpenGame}; seed::UInt64 = UInt64(137508))
    adhesions = Dict{Tuple{Int,Int}, Vector{Symbol}}()
    edges = Tuple{Int,Int}[]
    
    for i in 1:length(games)
        for j in (i+1):length(games)
            shared = intersect(games[i].strategies, games[j].strategies)
            if !isempty(shared)
                adhesions[(i,j)] = shared
                push!(edges, (i, j))
            end
        end
    end
    
    fp = sm64(seed ⊻ reduce(⊻, (g.fingerprint for g in games); init=UInt64(0)))
    GameDecomposition(games, adhesions, edges, fp)
end

"""
Adhesion width: maximum size of shared interface.
"""
function adhesion_width(decomp::GameDecomposition)
    isempty(decomp.adhesions) && return 0
    maximum(length(v) for v in values(decomp.adhesions))
end

# ============================================================================
# NARRATIVE-GAME INTEGRATION
# ============================================================================

"""
NarrativeGame: A game that evolves over a narrative structure.

Each interval in the narrative corresponds to a game phase.
Gluing corresponds to sequential composition.
"""
struct NarrativeGame
    narrative::Narrative
    games::Dict{Interval, OpenGame}
    world::World
    fingerprint::UInt64
end

"""
Create a narrative game from a world.
"""
function NarrativeGame(world::World)
    games = Dict{Interval, OpenGame}()
    
    for interval in world.narrative.intervals
        games[interval] = world.game
    end
    
    fp = sm64(world.fingerprint ⊻ world.narrative.fingerprint)
    NarrativeGame(world.narrative, games, world, fp)
end

"""
Play the narrative game through all intervals.
"""
function play_narrative!(ng::NarrativeGame, strategy::Symbol)
    interactions = Interaction[]
    
    for interval in sort(ng.narrative.intervals, by=i->i.start)
        i = play!(ng.world, strategy)
        push!(interactions, i)
    end
    
    interactions
end

# ============================================================================
# VISUALIZATION
# ============================================================================

"""
Generate Mermaid diagram of world history.
"""
function history_diagram(world::World)::String
    lines = ["```mermaid", "flowchart LR"]
    
    for (i, interaction) in enumerate(world.history)
        node_id = "t$(interaction.timestamp)"
        label = "$(interaction.strategy)\\n$(interaction.move)\\nu=$(interaction.utility)"
        push!(lines, "    $node_id[\"$label\"]")
        
        if i > 1
            prev_id = "t$(world.history[i-1].timestamp)"
            push!(lines, "    $prev_id --> $node_id")
        end
    end
    
    # Style based on equilibrium path
    if is_equilibrium_path(world)
        push!(lines, "    style t1 fill:#0f0,stroke:#0a0")
    else
        push!(lines, "    style t1 fill:#f00,stroke:#a00")
    end
    
    push!(lines, "```")
    join(lines, "\n")
end

"""
Generate Mermaid diagram of narrative sheaf.
"""
function narrative_diagram(narrative::Narrative)::String
    lines = ["```mermaid", "flowchart TB"]
    
    push!(lines, "    subgraph Narrative[\"$(narrative.name)\"]")
    
    for (i, interval) in enumerate(narrative.intervals)
        node_id = "i$i"
        label = "[$(interval.start),$(interval.stop)]"
        push!(lines, "        $node_id[\"$label\"]")
    end
    
    # Add restriction arrows for containment
    for (i, I) in enumerate(narrative.intervals)
        for (j, J) in enumerate(narrative.intervals)
            if i != j && contains(I, J)
                push!(lines, "        i$i -.->|\"ρ\"| i$j")
            end
        end
    end
    
    push!(lines, "    end")
    push!(lines, "```")
    
    join(lines, "\n")
end

# ============================================================================
# EXPORT
# ============================================================================

export Interval, contains, point_interval
export Narrative, restrict, glue
export Trit, MINUS, ERGODIC, PLUS
export OpenGame, compose_seq, compose_par, ⊗, ∘
export Interaction
export World, play!, status, equilibria, is_equilibrium_path
export history_interval, restrict_history
export prisoners_dilemma, coordination_game, hawk_dove
export world, reset_worlds!, list_worlds, ACTIVE_WORLDS
export GameDecomposition, decompose_game, adhesion_width
export NarrativeGame, play_narrative!
export history_diagram, narrative_diagram

# ============================================================================
# JOINT WORLD MODELING THROUGH INTERACTION
# ============================================================================
#
# davidad's insight (Topos Institute, 2023): Compositional World Modeling
# Two agents become jointly world-modelable when:
#   1. Each has a behavioral theory (what trajectories are possible)
#   2. They share an interface (common observation/action space)
#   3. Interaction history constrains joint distribution
#
# The sheaf structure captures this:
#   - Each agent's model = local section
#   - Joint model = global section (gluing)
#   - Consistency = sheaf condition (local sections agree on overlaps)

"""
JointModel: Two agents modeling each other through interaction.

Fields:
- alice, bob: Individual world models
- shared_history: Interaction log visible to both
- joint_narrative: Glued sheaf over time
- mutual_information: Bits of shared understanding
"""
mutable struct JointModel
    alice::World
    bob::World
    shared_history::Vector{Tuple{Interaction, Interaction}}
    joint_narrative::Narrative
    mutual_information::Float64
    fingerprint::UInt64
end

"""
    joint_model(alice::World, bob::World)

Create a joint model from two interacting worlds.
The joint narrative is the pullback of individual narratives.
"""
function joint_model(alice::World, bob::World)
    fp = sm64(alice.fingerprint ⊻ bob.fingerprint)
    joint_narr = Narrative(:joint; seed=fp)
    
    JointModel(
        alice,
        bob,
        Tuple{Interaction, Interaction}[],
        joint_narr,
        0.0,
        fp
    )
end

"""
    interact!(jm::JointModel, alice_strategy::Symbol, bob_strategy::Symbol)

Both agents play simultaneously. Returns the joint interaction.

This is where joint world modeling happens:
- Alice's play affects Bob's coplay (and vice versa)
- The interaction becomes shared history
- Mutual information increases with each coordinated move
"""
function interact!(jm::JointModel, alice_strategy::Symbol, bob_strategy::Symbol)
    # Forward pass: both play
    alice_move = jm.alice.game.play(alice_strategy, jm.alice.state)
    bob_move = jm.bob.game.play(bob_strategy, jm.bob.state)
    
    # Backward pass: utilities depend on other's move
    # This is the compositional structure: each sees the other's output
    alice_utility = jm.alice.game.coplay(alice_strategy, jm.alice.state, alice_move, bob_move)
    bob_utility = jm.bob.game.coplay(bob_strategy, jm.bob.state, bob_move, alice_move)
    
    timestamp = jm.alice.step + 1
    
    # Record interactions
    alice_int = Interaction(
        timestamp,
        jm.alice.state,
        alice_strategy,
        alice_move,
        alice_utility,
        bob_utility,  # Coutility = what the other got
        point_interval(timestamp; seed=jm.alice.seed),
        sm64(jm.alice.fingerprint ⊻ UInt64(timestamp))
    )
    
    bob_int = Interaction(
        timestamp,
        jm.bob.state,
        bob_strategy,
        bob_move,
        bob_utility,
        alice_utility,
        point_interval(timestamp; seed=jm.bob.seed),
        sm64(jm.bob.fingerprint ⊻ UInt64(timestamp))
    )
    
    # Update individual worlds
    push!(jm.alice.history, alice_int)
    push!(jm.bob.history, bob_int)
    jm.alice.step = timestamp
    jm.bob.step = timestamp
    jm.alice.cumulative_utility += alice_utility
    jm.bob.cumulative_utility += bob_utility
    
    # Record shared history
    push!(jm.shared_history, (alice_int, bob_int))
    
    # Update joint narrative (gluing individual perspectives)
    joint_interval = point_interval(timestamp; seed=jm.fingerprint)
    push!(jm.joint_narrative.intervals, joint_interval)
    jm.joint_narrative.values[joint_interval] = (
        alice = alice_int,
        bob = bob_int,
        coordinated = (alice_move == bob_move),
        pareto_efficient = is_pareto_efficient(alice_utility, bob_utility, jm)
    )
    
    # Update mutual information estimate
    # Coordination increases shared understanding
    if alice_move == bob_move
        jm.mutual_information += 1.0  # Simplified: 1 bit per coordination
    else
        jm.mutual_information += 0.5  # Still learn from disagreement
    end
    
    (alice_int, bob_int)
end

"""
Check if outcome is Pareto efficient (no one can improve without hurting other).
"""
function is_pareto_efficient(u_a::Float64, u_b::Float64, jm::JointModel)
    # Check against all strategy combinations
    for σ_a in jm.alice.game.strategies
        for σ_b in jm.bob.game.strategies
            m_a = jm.alice.game.play(σ_a, jm.alice.state)
            m_b = jm.bob.game.play(σ_b, jm.bob.state)
            v_a = jm.alice.game.coplay(σ_a, jm.alice.state, m_a, m_b)
            v_b = jm.bob.game.coplay(σ_b, jm.bob.state, m_b, m_a)
            
            # Pareto dominated if someone is strictly better and no one is worse
            if (v_a > u_a && v_b >= u_b) || (v_a >= u_a && v_b > u_b)
                return false
            end
        end
    end
    true
end

"""
    joint_status(jm::JointModel)

Print joint model status showing both perspectives.
"""
function joint_status(jm::JointModel)
    n = length(jm.shared_history)
    
    println("═══════════════════════════════════════════════════════════")
    println("           JOINT WORLD MODEL: $(jm.alice.name) ↔ $(jm.bob.name)")
    println("═══════════════════════════════════════════════════════════")
    println()
    println("  Interactions:        $n")
    println("  Mutual Information:  $(round(jm.mutual_information, digits=2)) bits")
    println()
    println("  ┌─────────────────────────────────────────────────────────┐")
    println("  │  $(jm.alice.name): utility = $(round(jm.alice.cumulative_utility, digits=2))")
    println("  │  $(jm.bob.name):   utility = $(round(jm.bob.cumulative_utility, digits=2))")
    println("  └─────────────────────────────────────────────────────────┘")
    
    if n > 0
        coordinations = count(h -> h[1].move == h[2].move, jm.shared_history)
        coord_rate = coordinations / n
        println()
        println("  Coordination rate:   $(round(100*coord_rate, digits=1))%")
        
        pareto_count = count(i -> 
            get(jm.joint_narrative.values[i], :pareto_efficient, false),
            jm.joint_narrative.intervals
        )
        println("  Pareto efficient:    $pareto_count / $n")
    end
    
    println()
    println("═══════════════════════════════════════════════════════════")
end

"""
    predict_other(jm::JointModel, who::Symbol)

Use interaction history to predict the other agent's next move.
This is the "theory of mind" that emerges from joint modeling.
"""
function predict_other(jm::JointModel, who::Symbol)
    history = jm.shared_history
    isempty(history) && return nothing
    
    # Count strategy frequencies
    counts = Dict{Symbol, Int}()
    
    for (alice_int, bob_int) in history
        int = who == :alice ? alice_int : bob_int
        counts[int.strategy] = get(counts, int.strategy, 0) + 1
    end
    
    # Return most frequent (maximum likelihood estimate)
    argmax(counts)
end

"""
    best_response(jm::JointModel, who::Symbol)

Compute best response given prediction of other's play.
This is where learning meets game theory.
"""
function best_response(jm::JointModel, who::Symbol)
    predicted = predict_other(jm, who == :alice ? :bob : :alice)
    predicted === nothing && return nothing
    
    world = who == :alice ? jm.alice : jm.bob
    other_world = who == :alice ? jm.bob : jm.alice
    
    # Find strategy maximizing utility against predicted opponent
    best_strat = nothing
    best_util = -Inf
    
    for σ in world.game.strategies
        move = world.game.play(σ, world.state)
        other_move = other_world.game.play(predicted, other_world.state)
        util = world.game.coplay(σ, world.state, move, other_move)
        
        if util > best_util
            best_util = util
            best_strat = σ
        end
    end
    
    best_strat
end

"""
Generate Mermaid diagram of joint model.
"""
function joint_diagram(jm::JointModel)::String
    lines = ["```mermaid", "flowchart LR"]
    
    push!(lines, "    subgraph Alice[\"$(jm.alice.name)\"]")
    push!(lines, "        A_state[\"State: $(jm.alice.state)\"]")
    push!(lines, "        A_util[\"Utility: $(round(jm.alice.cumulative_utility, digits=1))\"]")
    push!(lines, "    end")
    
    push!(lines, "    subgraph Bob[\"$(jm.bob.name)\"]")
    push!(lines, "        B_state[\"State: $(jm.bob.state)\"]")
    push!(lines, "        B_util[\"Utility: $(round(jm.bob.cumulative_utility, digits=1))\"]")
    push!(lines, "    end")
    
    push!(lines, "    subgraph Joint[\"Joint Model\"]")
    push!(lines, "        MI[\"MI: $(round(jm.mutual_information, digits=1)) bits\"]")
    push!(lines, "        N[\"Interactions: $(length(jm.shared_history))\"]")
    push!(lines, "    end")
    
    push!(lines, "    Alice <-->|\"interact!\"| Joint")
    push!(lines, "    Bob <-->|\"interact!\"| Joint")
    
    push!(lines, "```")
    join(lines, "\n")
end

# Export joint modeling
export JointModel, joint_model, interact!, joint_status
export is_pareto_efficient, predict_other, best_response
export joint_diagram

end # module NarrativeWorld
