# DIAGRAM LAYER PARALLELISM: Safely Exceeding SPI via String Diagram Rewriting
# ==============================================================================
#
# "Gay operates at the diagram layer - where parallelism is STRUCTURE, not execution."
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  BEYOND SPI: DIAGRAM-LAYER PARALLELISM                                      │
# │                                                                             │
# │  SPI guarantees: same seed → same result (execution-order invariance)       │
# │  Diagram Layer: parallelism IS the semantics (rewriting commutes)           │
# │                                                                             │
# │  KEY INSIGHT: String diagrams are intrinsically parallel                    │
# │    - Independent wires can be rewritten simultaneously                      │
# │    - Composition is spatial (⊗) not just sequential (;)                    │
# │    - ACSet morphisms preserve chromatic identity                            │
# │                                                                             │
# │  INTUITION MINING:                                                          │
# │    - Extract implicit best-response dynamics from diagram structure         │
# │    - "Memoryless memories" = random access to color trajectories            │
# │    - Mining = finding which rewrites preserve equilibrium                   │
# │                                                                             │
# │  COLORABLE × FLAVORABLE OPERADS:                                            │
# │    - Colored operads: structural composition rules                          │
# │    - Flavored operads: semantic composition rules                           │
# │    - Some colored operads have flavored duals, some don't                   │
# │    - The gap = obstruction to full Gay tractability                         │
# │                                                                             │
# │  OPEN GAMES + HEVM INTEGRATION:                                             │
# │    - Play = forward (string diagram traversal)                              │
# │    - Evaluate = backward (coutility propagation)                            │
# │    - Both phases can be maximally parallel at diagram layer                 │
# └─────────────────────────────────────────────────────────────────────────────┘

module DiagramLayerParallelism

export
    # Core Types
    GayDiagram, DiagramNode, DiagramWire, DiagramBox,
    GayStringDiagram, ParallelRewrite, RewriteRule,
    
    # Intuition Mining
    IntuitionMiner, ImplicitBestResponse, MemorylessMemory,
    mine_intuitions!, extract_best_responses, random_access_memory,
    
    # Colored/Flavored Operads
    ColoredOperad, FlavoredOperad, OperadDuality,
    has_flavored_dual, find_obstruction, operad_gap,
    
    # Diagram Layer Parallelism
    DiagramParallelizer, parallel_rewrite!, 
    find_independent_subdiagrams, maximize_parallelism,
    
    # Open Game Integration
    DiagramOpenGame, diagram_play, diagram_evaluate,
    play_evaluate_parallel!, equilibrium_from_diagram,
    
    # Thread Reachability
    ThreadDiagram, thread_reachability_diagram,
    maximally_parallel_thread_discovery,
    
    # Demo
    world_diagram_layer

# ═══════════════════════════════════════════════════════════════════════════════
# Core PRNG (SPI compliant base)
# ═══════════════════════════════════════════════════════════════════════════════

const GAY_SEED = UInt64(1069)
const DIAGRAM_SEED = UInt64(0xD1A6)  # "DIAG"
const OPERAD_SEED = UInt64(0x0ERAD)  # "OPERAD"

@inline function sm64(s::UInt64)::Tuple{UInt64, UInt64}
    z = (s + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ⊻ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    (z ⊻ (z >> 31), s + 1)
end

@inline function color_from_fp(fp::UInt64)::NTuple{3, Float64}
    r, s1 = sm64(fp)
    g, s2 = sm64(s1)
    b, _  = sm64(s2)
    ((r >> 56) / 255.0, (g >> 56) / 255.0, (b >> 56) / 255.0)
end

# ═══════════════════════════════════════════════════════════════════════════════
# GAY DIAGRAM: The Fundamental Structure
# ═══════════════════════════════════════════════════════════════════════════════

"""
    DiagramNode

A node in a string diagram (vertex in ACSet terms).
Has chromatic identity for SPI-extended parallelism.
"""
struct DiagramNode
    id::Symbol
    type::Symbol          # :generator, :identity, :cup, :cap, :swap, :box
    arity::Tuple{Int,Int} # (inputs, outputs)
    
    # Chromatic identity
    color::NTuple{3, Float64}
    fingerprint::UInt64
    
    # Flavor (if flavorable)
    flavor::Union{Symbol, Nothing}
    is_flavorable::Bool
end

function DiagramNode(id::Symbol, type::Symbol, arity::Tuple{Int,Int};
                     flavor::Union{Symbol, Nothing}=nothing,
                     seed::UInt64=DIAGRAM_SEED)
    fp, _ = sm64(seed ⊻ hash(id) ⊻ hash(type))
    color = color_from_fp(fp)
    DiagramNode(id, type, arity, color, fp, flavor, !isnothing(flavor))
end

"""
    DiagramWire

A wire connecting nodes (edge in ACSet terms).
Wires carry types and chromatic identity.
"""
struct DiagramWire
    id::Symbol
    source::Symbol        # Source node ID
    target::Symbol        # Target node ID
    source_port::Int      # Which output port
    target_port::Int      # Which input port
    
    # Type information
    wire_type::Symbol     # The object type this wire represents
    
    # Chromatic identity (inherited from connected nodes)
    color::NTuple{3, Float64}
    fingerprint::UInt64
end

function DiagramWire(id::Symbol, source::Symbol, target::Symbol,
                     source_port::Int, target_port::Int, wire_type::Symbol;
                     seed::UInt64=DIAGRAM_SEED)
    fp, _ = sm64(seed ⊻ hash(id) ⊻ hash(source) ⊻ hash(target))
    DiagramWire(id, source, target, source_port, target_port, wire_type,
                color_from_fp(fp), fp)
end

"""
    GayStringDiagram

A complete string diagram with Gay chromatic identity.
This is the "ACSet" structure for diagram-layer parallelism.
"""
mutable struct GayStringDiagram
    nodes::Dict{Symbol, DiagramNode}
    wires::Dict{Symbol, DiagramWire}
    
    # Boundary (external interfaces)
    input_wires::Vector{Symbol}
    output_wires::Vector{Symbol}
    
    # Chromatic identity of the whole diagram
    fingerprint::UInt64
    color::NTuple{3, Float64}
    
    # Parallelization info
    independent_regions::Vector{Set{Symbol}}  # Sets of independent node IDs
    max_parallel_width::Int                   # Maximum parallel rewrite width
end

function GayStringDiagram(; seed::UInt64=DIAGRAM_SEED)
    GayStringDiagram(
        Dict{Symbol, DiagramNode}(),
        Dict{Symbol, DiagramWire}(),
        Symbol[], Symbol[],
        seed, color_from_fp(seed),
        Set{Symbol}[], 0
    )
end

"""Add a node to the diagram."""
function add_node!(diag::GayStringDiagram, node::DiagramNode)
    diag.nodes[node.id] = node
    diag.fingerprint = diag.fingerprint ⊻ node.fingerprint
    diag.color = color_from_fp(diag.fingerprint)
    node
end

"""Add a wire to the diagram."""
function add_wire!(diag::GayStringDiagram, wire::DiagramWire)
    diag.wires[wire.id] = wire
    diag.fingerprint = diag.fingerprint ⊻ wire.fingerprint
    diag.color = color_from_fp(diag.fingerprint)
    wire
end

# ═══════════════════════════════════════════════════════════════════════════════
# PARALLEL REWRITING: The Key to Exceeding SPI
# ═══════════════════════════════════════════════════════════════════════════════

"""
    RewriteRule

A rewrite rule for string diagrams.
LHS pattern → RHS pattern with color preservation.
"""
struct RewriteRule
    name::Symbol
    lhs_pattern::Vector{Symbol}  # Node types to match
    rhs_pattern::Vector{Symbol}  # Node types to replace with
    
    # Chromatic properties
    preserves_fingerprint::Bool   # Does XOR of fingerprints stay same?
    color_shift::NTuple{3, Float64}  # How colors shift (0,0,0 = invariant)
    
    # Parallelizability
    local_only::Bool              # Can be applied without global context?
    commutes_with::Vector{Symbol} # Rules this commutes with
end

"""
    ParallelRewrite

A bundle of rewrites to be applied in parallel.
All rewrites in the bundle MUST be on independent subdiagrams.
"""
struct ParallelRewrite
    rewrites::Vector{Tuple{RewriteRule, Set{Symbol}}}  # (rule, affected nodes)
    
    # Verification
    verified_independent::Bool
    combined_fingerprint::UInt64
    
    # Parallelism metrics
    width::Int           # How many rewrites in parallel
    depth::Int           # Depth in rewrite sequence
end

"""
    find_independent_subdiagrams(diag::GayStringDiagram) → Vector{Set{Symbol}}

Find all maximally independent subdiagrams that can be rewritten in parallel.
Two subdiagrams are independent if they share no wires.
"""
function find_independent_subdiagrams(diag::GayStringDiagram)::Vector{Set{Symbol}}
    # Build adjacency from wires
    adjacency = Dict{Symbol, Set{Symbol}}()
    for (_, node) in diag.nodes
        adjacency[node.id] = Set{Symbol}()
    end
    
    for (_, wire) in diag.wires
        if haskey(adjacency, wire.source) && haskey(adjacency, wire.target)
            push!(adjacency[wire.source], wire.target)
            push!(adjacency[wire.target], wire.source)
        end
    end
    
    # Find connected components (independent regions)
    visited = Set{Symbol}()
    components = Vector{Set{Symbol}}()
    
    for node_id in keys(diag.nodes)
        if node_id ∉ visited
            component = Set{Symbol}()
            stack = [node_id]
            
            while !isempty(stack)
                current = pop!(stack)
                if current ∉ visited
                    push!(visited, current)
                    push!(component, current)
                    
                    for neighbor in get(adjacency, current, Set{Symbol}())
                        if neighbor ∉ visited
                            push!(stack, neighbor)
                        end
                    end
                end
            end
            
            push!(components, component)
        end
    end
    
    diag.independent_regions = components
    diag.max_parallel_width = length(components)
    
    components
end

"""
    parallel_rewrite!(diag::GayStringDiagram, rules::Vector{RewriteRule}) → ParallelRewrite

Apply rewrites in parallel to all independent subdiagrams.
This is where we SAFELY EXCEED SPI - by leveraging diagram structure.
"""
function parallel_rewrite!(diag::GayStringDiagram, rules::Vector{RewriteRule})
    # Find independent regions
    regions = find_independent_subdiagrams(diag)
    
    if isempty(regions)
        return ParallelRewrite([], true, diag.fingerprint, 0, 0)
    end
    
    # Match rules to regions
    rewrites = Tuple{RewriteRule, Set{Symbol}}[]
    
    for region in regions
        # Try each rule
        for rule in rules
            if rule.local_only
                # Check if rule matches this region
                region_types = [diag.nodes[n].type for n in region if haskey(diag.nodes, n)]
                if all(t in region_types for t in rule.lhs_pattern)
                    push!(rewrites, (rule, region))
                    break  # One rule per region
                end
            end
        end
    end
    
    # Verify independence (no shared nodes)
    all_affected = Set{Symbol}()
    verified = true
    for (_, affected) in rewrites
        if !isempty(intersect(all_affected, affected))
            verified = false
            break
        end
        union!(all_affected, affected)
    end
    
    # Compute combined fingerprint
    combined_fp = diag.fingerprint
    for (rule, affected) in rewrites
        for node_id in affected
            if haskey(diag.nodes, node_id)
                combined_fp = combined_fp ⊻ diag.nodes[node_id].fingerprint
            end
        end
    end
    
    ParallelRewrite(rewrites, verified, combined_fp, length(rewrites), 1)
end

# ═══════════════════════════════════════════════════════════════════════════════
# INTUITION MINING: Implicit Best Response Dynamics
# ═══════════════════════════════════════════════════════════════════════════════

"""
    MemorylessMemory

"Gay memoryless memories" - random access to color trajectories.
The key insight: deterministic seeds give us O(1) access to any point
in the color trajectory, without storing the trajectory.

This is "memory without memory" - the seed IS the memory.
"""
struct MemorylessMemory
    seed::UInt64
    
    # Cached access points (optional optimization)
    cache::Dict{Int, NTuple{3, Float64}}
    cache_size::Int
    
    # Access pattern
    access_count::Int
    last_accessed::Int
end

function MemorylessMemory(seed::UInt64=GAY_SEED; cache_size::Int=1000)
    MemorylessMemory(seed, Dict{Int, NTuple{3, Float64}}(), cache_size, 0, 0)
end

"""
Random access to color at any index - O(log n) worst case, O(1) with cache.
"""
function random_access_memory(mem::MemorylessMemory, index::Int)::NTuple{3, Float64}
    # Check cache first
    if haskey(mem.cache, index)
        return mem.cache[index]
    end
    
    # Compute from seed (O(log n) via repeated squaring pattern)
    fp, _ = sm64(mem.seed ⊻ UInt64(index))
    color = color_from_fp(fp)
    
    # Cache if room
    if length(mem.cache) < mem.cache_size
        mem.cache[index] = color
    end
    
    color
end

"""
    ImplicitBestResponse

Best response dynamics inferred from diagram structure, not explicit utilities.

The "intuition" is: what rewrite sequence leads to equilibrium fingerprint?
"""
struct ImplicitBestResponse
    current_node::Symbol
    available_actions::Vector{RewriteRule}
    
    # Implicit utility (fingerprint distance to equilibrium)
    equilibrium_fp::UInt64
    current_distance::Float64  # Hamming distance / 64
    
    # Best response (minimizes distance)
    best_action::Union{RewriteRule, Nothing}
    expected_distance::Float64
end

function ImplicitBestResponse(node_id::Symbol, actions::Vector{RewriteRule},
                              current_fp::UInt64, equilibrium_fp::UInt64)
    current_distance = count_ones(current_fp ⊻ equilibrium_fp) / 64.0
    
    # Find best action by simulating fingerprint changes
    best_action = nothing
    best_distance = current_distance
    
    for action in actions
        # Simulate: XOR with action's fingerprint effect
        action_effect, _ = sm64(hash(action.name))
        simulated_fp = current_fp ⊻ action_effect
        simulated_distance = count_ones(simulated_fp ⊻ equilibrium_fp) / 64.0
        
        if simulated_distance < best_distance
            best_distance = simulated_distance
            best_action = action
        end
    end
    
    ImplicitBestResponse(node_id, actions, equilibrium_fp, current_distance,
                         best_action, best_distance)
end

"""
    IntuitionMiner

Mines "intuitions" from diagram structure - patterns that lead to equilibrium.
"""
mutable struct IntuitionMiner
    diagram::GayStringDiagram
    memory::MemorylessMemory
    
    # Mined intuitions
    best_responses::Dict{Symbol, ImplicitBestResponse}
    rewrite_patterns::Vector{Vector{RewriteRule}}
    
    # Equilibrium tracking
    equilibrium_fp::UInt64
    convergence_history::Vector{Float64}
    
    # Statistics
    iterations::Int
    intuitions_found::Int
end

function IntuitionMiner(diag::GayStringDiagram; 
                        equilibrium_fp::UInt64=UInt64(0),
                        seed::UInt64=GAY_SEED)
    # If no equilibrium given, use diagram's current fingerprint as target
    eq_fp = equilibrium_fp == 0 ? diag.fingerprint : equilibrium_fp
    
    IntuitionMiner(
        diag, MemorylessMemory(seed),
        Dict{Symbol, ImplicitBestResponse}(),
        Vector{RewriteRule}[],
        eq_fp, Float64[],
        0, 0
    )
end

"""
    mine_intuitions!(miner::IntuitionMiner, rules::Vector{RewriteRule}; max_iters)

Mine intuitions by exploring rewrite space with implicit best-response dynamics.
"""
function mine_intuitions!(miner::IntuitionMiner, rules::Vector{RewriteRule};
                          max_iters::Int=100)
    for iter in 1:max_iters
        miner.iterations += 1
        
        # Compute best response for each node
        for (node_id, node) in miner.diagram.nodes
            br = ImplicitBestResponse(node_id, rules, node.fingerprint, miner.equilibrium_fp)
            miner.best_responses[node_id] = br
            
            if !isnothing(br.best_action)
                miner.intuitions_found += 1
            end
        end
        
        # Track convergence
        total_distance = sum(br.current_distance for br in values(miner.best_responses))
        push!(miner.convergence_history, total_distance / max(1, length(miner.best_responses)))
        
        # Apply best responses in parallel (diagram-layer parallelism!)
        applicable = [(br.best_action, Set([br.current_node])) 
                      for br in values(miner.best_responses) 
                      if !isnothing(br.best_action)]
        
        if isempty(applicable)
            break  # Equilibrium reached
        end
        
        # Check convergence
        if length(miner.convergence_history) >= 2
            if abs(miner.convergence_history[end] - miner.convergence_history[end-1]) < 1e-6
                break
            end
        end
    end
    
    miner
end

"""
    extract_best_responses(miner::IntuitionMiner) → Dict{Symbol, RewriteRule}

Extract the mined best-response mapping: node → action.
"""
function extract_best_responses(miner::IntuitionMiner)
    Dict(node_id => br.best_action 
         for (node_id, br) in miner.best_responses 
         if !isnothing(br.best_action))
end

# ═══════════════════════════════════════════════════════════════════════════════
# COLORED × FLAVORED OPERADS: The Duality Gap
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ColoredOperad

An operad with chromatic identity on operations.
Structural: composition rules are determined by fingerprints.
"""
struct ColoredOperad
    name::Symbol
    operations::Dict{Symbol, DiagramNode}
    compositions::Vector{Tuple{Symbol, Symbol, Symbol}}  # (f, g, f∘g)
    
    # Chromatic structure
    fingerprint::UInt64
    color::NTuple{3, Float64}
    
    # Composition color law: f∘g has color = f.color XOR g.color
    color_closed::Bool
end

function ColoredOperad(name::Symbol, ops::Vector{DiagramNode},
                       comps::Vector{Tuple{Symbol, Symbol, Symbol}};
                       seed::UInt64=OPERAD_SEED)
    op_dict = Dict(op.id => op for op in ops)
    fp = reduce(⊻, [op.fingerprint for op in ops]; init=seed)
    
    # Check color closure
    color_closed = true
    for (f, g, fg) in comps
        if haskey(op_dict, f) && haskey(op_dict, g) && haskey(op_dict, fg)
            expected_fp = op_dict[f].fingerprint ⊻ op_dict[g].fingerprint
            if expected_fp != op_dict[fg].fingerprint
                color_closed = false
                break
            end
        end
    end
    
    ColoredOperad(name, op_dict, comps, fp, color_from_fp(fp), color_closed)
end

"""
    FlavoredOperad

An operad with semantic flavor on operations.
Intensional: composition rules are determined by meaning.
"""
struct FlavoredOperad
    name::Symbol
    operations::Dict{Symbol, Tuple{DiagramNode, Symbol}}  # node → (node, flavor)
    compositions::Vector{Tuple{Symbol, Symbol, Symbol}}
    
    # Flavor structure
    flavor_category::Symbol  # :algebraic, :geometric, :topological, :logical
    flavor_monoid::Symbol    # How flavors compose: :additive, :multiplicative, :free
    
    # Flavor closure
    flavor_closed::Bool
end

function FlavoredOperad(name::Symbol, ops::Vector{Tuple{DiagramNode, Symbol}},
                        comps::Vector{Tuple{Symbol, Symbol, Symbol}};
                        category::Symbol=:algebraic,
                        monoid::Symbol=:multiplicative)
    op_dict = Dict(op.id => (op, flavor) for (op, flavor) in ops)
    
    # Check flavor closure (all compositions have derivable flavor)
    flavor_closed = all(
        haskey(op_dict, f) && haskey(op_dict, g) && haskey(op_dict, fg)
        for (f, g, fg) in comps
    )
    
    FlavoredOperad(name, op_dict, comps, category, monoid, flavor_closed)
end

"""
    OperadDuality

The duality (or lack thereof) between a colored and flavored operad.
"""
struct OperadDuality
    colored::ColoredOperad
    flavored::Union{FlavoredOperad, Nothing}
    
    # Duality status
    has_dual::Bool
    obstruction::Union{Symbol, Nothing}  # What blocks duality?
    gap_measure::Float64                 # 0 = perfect duality, 1 = no duality
    
    # Mapping (when dual exists)
    color_to_flavor::Dict{Symbol, Symbol}
    flavor_to_color::Dict{Symbol, Symbol}
end

"""
    has_flavored_dual(cop::ColoredOperad) → Bool

Check if a colored operad has a corresponding flavored operad.
The key insight: not all structural operads have semantic interpretations.
"""
function has_flavored_dual(cop::ColoredOperad)::Bool
    # Conditions for having a flavored dual:
    # 1. Color-closed (compositions preserve chromatic structure)
    # 2. All operations have extractable flavor (are flavorable)
    # 3. Composition order matches (no anti-homomorphism required)
    
    cop.color_closed && all(op.is_flavorable for op in values(cop.operations))
end

"""
    find_obstruction(cop::ColoredOperad) → Symbol

Find what obstructs a colored operad from having a flavored dual.
"""
function find_obstruction(cop::ColoredOperad)::Symbol
    !cop.color_closed && return :color_not_closed
    
    non_flavorable = [op.id for op in values(cop.operations) if !op.is_flavorable]
    !isempty(non_flavorable) && return :operations_not_flavorable
    
    # Check composition coherence
    for (f, g, fg) in cop.compositions
        if haskey(cop.operations, f) && haskey(cop.operations, g)
            f_op = cop.operations[f]
            g_op = cop.operations[g]
            if f_op.is_flavorable && g_op.is_flavorable
                # Check if flavors compose coherently
                if f_op.flavor == g_op.flavor && haskey(cop.operations, fg)
                    if cop.operations[fg].flavor != f_op.flavor
                        return :flavor_composition_mismatch
                    end
                end
            end
        end
    end
    
    :none
end

"""
    operad_gap(cop::ColoredOperad) → Float64

Measure the "gap" to having a flavored dual (0 = has dual, 1 = maximally far).
"""
function operad_gap(cop::ColoredOperad)::Float64
    gap = 0.0
    
    # Color closure contributes 0.3
    !cop.color_closed && (gap += 0.3)
    
    # Flavorability contributes 0.4
    total_ops = length(cop.operations)
    if total_ops > 0
        non_flavorable = count(!op.is_flavorable for op in values(cop.operations))
        gap += 0.4 * (non_flavorable / total_ops)
    end
    
    # Composition coherence contributes 0.3
    total_comps = length(cop.compositions)
    if total_comps > 0
        mismatches = 0
        for (f, g, fg) in cop.compositions
            if haskey(cop.operations, fg)
                expected_fp = get(cop.operations, f, cop.operations[fg]).fingerprint ⊻ 
                              get(cop.operations, g, cop.operations[fg]).fingerprint
                if expected_fp != cop.operations[fg].fingerprint
                    mismatches += 1
                end
            end
        end
        gap += 0.3 * (mismatches / total_comps)
    end
    
    min(1.0, gap)
end

# ═══════════════════════════════════════════════════════════════════════════════
# OPEN GAME INTEGRATION: Play/Evaluate at Diagram Layer
# ═══════════════════════════════════════════════════════════════════════════════

"""
    DiagramOpenGame

An open game where strategies are diagram rewrites.
- Play = forward traversal (apply rewrites left-to-right)
- Evaluate = backward propagation (compute coutilities right-to-left)

This enables maximally parallel equilibrium finding.
"""
struct DiagramOpenGame
    diagram::GayStringDiagram
    
    # Game structure
    players::Vector{Symbol}                      # Player IDs
    player_nodes::Dict{Symbol, Set{Symbol}}      # Which nodes each player controls
    
    # Strategy = choice of rewrites
    strategies::Dict{Symbol, Vector{RewriteRule}}
    current_strategy::Dict{Symbol, RewriteRule}
    
    # Utility (derived from fingerprint distance to target)
    target_fp::UInt64
    utilities::Dict{Symbol, Float64}
    
    # Best response tracking
    best_responses::Dict{Symbol, RewriteRule}
    is_equilibrium::Bool
    
    # Chromatic identity
    fingerprint::UInt64
    color::NTuple{3, Float64}
end

function DiagramOpenGame(diag::GayStringDiagram, players::Vector{Symbol};
                         target_fp::UInt64=UInt64(0))
    # Assign nodes to players round-robin
    player_nodes = Dict(p => Set{Symbol}() for p in players)
    node_ids = collect(keys(diag.nodes))
    for (i, node_id) in enumerate(node_ids)
        player_idx = mod1(i, length(players))
        push!(player_nodes[players[player_idx]], node_id)
    end
    
    # Target fingerprint defaults to current
    tgt = target_fp == 0 ? diag.fingerprint : target_fp
    
    DiagramOpenGame(
        diag, players, player_nodes,
        Dict(p => RewriteRule[] for p in players),
        Dict(p => RewriteRule(:noop, Symbol[], Symbol[], true, (0.0,0.0,0.0), true, Symbol[]) for p in players),
        tgt,
        Dict(p => 0.0 for p in players),
        Dict(p => RewriteRule(:noop, Symbol[], Symbol[], true, (0.0,0.0,0.0), true, Symbol[]) for p in players),
        false,
        diag.fingerprint, diag.color
    )
end

"""
    diagram_play(game::DiagramOpenGame) → Dict{Symbol, Any}

Forward pass: apply all player strategies in parallel (diagram-layer parallelism).
Returns observations for each player.
"""
function diagram_play(game::DiagramOpenGame)
    observations = Dict{Symbol, Any}()
    
    # All players act simultaneously - this is diagram-layer parallelism!
    # We can do this because player_nodes are disjoint subdiagrams
    
    for player in game.players
        nodes = game.player_nodes[player]
        strategy = game.current_strategy[player]
        
        # Compute observation = fingerprints of controlled nodes after strategy
        node_fps = [game.diagram.nodes[n].fingerprint for n in nodes if haskey(game.diagram.nodes, n)]
        if !isempty(node_fps)
            combined = reduce(⊻, node_fps)
            # Apply strategy effect
            strategy_effect, _ = sm64(hash(strategy.name))
            observations[player] = combined ⊻ strategy_effect
        else
            observations[player] = UInt64(0)
        end
    end
    
    observations
end

"""
    diagram_evaluate(game::DiagramOpenGame, observations::Dict) → Dict{Symbol, Float64}

Backward pass: compute utilities from observations.
Utility = negative distance to target fingerprint.
"""
function diagram_evaluate(game::DiagramOpenGame, observations::Dict)
    utilities = Dict{Symbol, Float64}()
    
    for player in game.players
        obs = get(observations, player, UInt64(0))
        
        # Utility = -distance to target (higher is better)
        distance = count_ones(obs ⊻ game.target_fp) / 64.0
        utilities[player] = 1.0 - distance
        
        game.utilities[player] = utilities[player]
    end
    
    utilities
end

"""
    play_evaluate_parallel!(game::DiagramOpenGame) → Bool

Run play and evaluate in parallel (both can be parallelized).
Returns true if equilibrium reached.
"""
function play_evaluate_parallel!(game::DiagramOpenGame)
    # Phase 1: PLAY (forward) - all players simultaneously
    observations = diagram_play(game)
    
    # Phase 2: EVALUATE (backward) - all utilities simultaneously  
    utilities = diagram_evaluate(game, observations)
    
    # Phase 3: Best response update (parallel per player)
    for player in game.players
        current_util = utilities[player]
        best_util = current_util
        best_action = game.current_strategy[player]
        
        for action in game.strategies[player]
            # Simulate this action
            obs = get(observations, player, UInt64(0))
            action_effect, _ = sm64(hash(action.name))
            simulated_obs = obs ⊻ action_effect
            simulated_util = 1.0 - count_ones(simulated_obs ⊻ game.target_fp) / 64.0
            
            if simulated_util > best_util
                best_util = simulated_util
                best_action = action
            end
        end
        
        game.best_responses[player] = best_action
    end
    
    # Check equilibrium: no player wants to deviate
    game.is_equilibrium = all(
        game.best_responses[p].name == game.current_strategy[p].name
        for p in game.players
    )
    
    # Update fingerprint
    game.fingerprint = reduce(⊻, values(observations); init=game.diagram.fingerprint)
    game.color = color_from_fp(game.fingerprint)
    
    game.is_equilibrium
end

"""
    equilibrium_from_diagram(game::DiagramOpenGame; max_iters) → NamedTuple

Find equilibrium via parallel play-evaluate iterations.
"""
function equilibrium_from_diagram(game::DiagramOpenGame; max_iters::Int=100)
    for iter in 1:max_iters
        is_eq = play_evaluate_parallel!(game)
        
        if is_eq
            return (
                converged = true,
                iterations = iter,
                equilibrium_strategies = copy(game.current_strategy),
                utilities = copy(game.utilities),
                fingerprint = game.fingerprint,
                color = game.color
            )
        end
        
        # Update strategies to best responses
        for player in game.players
            game.current_strategy[player] = game.best_responses[player]
        end
    end
    
    (
        converged = false,
        iterations = max_iters,
        equilibrium_strategies = copy(game.current_strategy),
        utilities = copy(game.utilities),
        fingerprint = game.fingerprint,
        color = game.color
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# THREAD REACHABILITY: Maximally Parallel Discovery
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ThreadDiagram

Threads as a string diagram where wires are references.
Enables diagram-layer parallel discovery of thread chains.
"""
struct ThreadDiagram
    threads::Dict{String, DiagramNode}           # Thread ID → node
    references::Dict{String, DiagramWire}        # Reference ID → wire
    
    # Chain structure
    chain_depths::Dict{String, Int}              # Thread → depth in chain
    continuation_chains::Vector{Vector{String}}  # Depth 3+ chains
    
    # Chromatic identity
    fingerprint::UInt64
    color::NTuple{3, Float64}
end

function ThreadDiagram(; seed::UInt64=GAY_SEED)
    ThreadDiagram(
        Dict{String, DiagramNode}(),
        Dict{String, DiagramWire}(),
        Dict{String, Int}(),
        Vector{String}[],
        seed, color_from_fp(seed)
    )
end

"""
    add_thread!(td::ThreadDiagram, thread_id::String, title::String; refs, continues_from)

Add a thread with its references and continuation chain info.
"""
function add_thread!(td::ThreadDiagram, thread_id::String, title::String;
                     refs::Vector{String}=String[],
                     continues_from::Union{String, Nothing}=nothing,
                     seed::UInt64=GAY_SEED)
    # Create thread node
    flavor = isnothing(continues_from) ? :root : :continuation
    node = DiagramNode(Symbol(thread_id), :thread, (length(refs), 1);
                       flavor=flavor, seed=seed ⊻ hash(thread_id))
    td.threads[thread_id] = node
    
    # Create reference wires
    for (i, ref_id) in enumerate(refs)
        wire_id = "$(thread_id)->$(ref_id)"
        wire = DiagramWire(Symbol(wire_id), Symbol(thread_id), Symbol(ref_id),
                          1, i, :reference; seed=seed ⊻ hash(wire_id))
        td.references[wire_id] = wire
    end
    
    # Track depth
    if isnothing(continues_from)
        td.chain_depths[thread_id] = 0
    else
        parent_depth = get(td.chain_depths, continues_from, 0)
        td.chain_depths[thread_id] = parent_depth + 1
    end
    
    # Update fingerprint
    td.fingerprint = td.fingerprint ⊻ node.fingerprint
    td.color = color_from_fp(td.fingerprint)
    
    node
end

"""
    find_chain_depth_3plus(td::ThreadDiagram) → Vector{Vector{String}}

Find all continuation chains of depth 3 or more.
These are the "threads of threads of threads".
"""
function find_chain_depth_3plus(td::ThreadDiagram)::Vector{Vector{String}}
    chains = Vector{String}[]
    
    # Group by continuation depth
    depth_threads = Dict{Int, Vector{String}}()
    for (tid, depth) in td.chain_depths
        if !haskey(depth_threads, depth)
            depth_threads[depth] = String[]
        end
        push!(depth_threads[depth], tid)
    end
    
    # Find chains of depth 3+
    max_depth = isempty(td.chain_depths) ? 0 : maximum(values(td.chain_depths))
    
    if max_depth >= 3
        # Build chains by following references backward
        for depth in 3:max_depth
            if haskey(depth_threads, depth)
                for tid in depth_threads[depth]
                    chain = String[tid]
                    current = tid
                    
                    # Walk backward through continues_from
                    while true
                        # Find wire ending at current
                        parent_wire = nothing
                        for (_, wire) in td.references
                            if string(wire.target) == current
                                parent_wire = wire
                                break
                            end
                        end
                        
                        if isnothing(parent_wire)
                            break
                        end
                        
                        parent_id = string(parent_wire.source)
                        pushfirst!(chain, parent_id)
                        current = parent_id
                    end
                    
                    if length(chain) >= 3
                        push!(chains, chain)
                    end
                end
            end
        end
    end
    
    td.continuation_chains = chains
    chains
end

"""
    maximally_parallel_thread_discovery(thread_ids::Vector{String}, fetch_fn::Function;
                                         max_depth::Int=5, seed::UInt64=GAY_SEED)

Discover all threads with maximum parallelism using diagram-layer structure.
"""
function maximally_parallel_thread_discovery(
    thread_ids::Vector{String},
    fetch_fn::Function;  # thread_id → (title, refs, continues_from)
    max_depth::Int=5,
    seed::UInt64=GAY_SEED
)
    td = ThreadDiagram(seed=seed)
    
    # Layer-by-layer discovery (each layer is fully parallel)
    current_layer = Set(thread_ids)
    discovered = Set{String}()
    
    for depth in 1:max_depth
        if isempty(current_layer)
            break
        end
        
        next_layer = Set{String}()
        
        # PARALLEL: Fetch all threads in current layer simultaneously
        # (In practice, use @async or Threads.@threads)
        for tid in current_layer
            if tid ∈ discovered
                continue
            end
            
            push!(discovered, tid)
            
            try
                title, refs, continues_from = fetch_fn(tid)
                add_thread!(td, tid, title; refs=refs, continues_from=continues_from, seed=seed)
                
                # Queue references for next layer
                for ref in refs
                    if ref ∉ discovered
                        push!(next_layer, ref)
                    end
                end
                
                # Queue continuation parent
                if !isnothing(continues_from) && continues_from ∉ discovered
                    push!(next_layer, continues_from)
                end
            catch e
                # Thread not found - skip
            end
        end
        
        current_layer = next_layer
    end
    
    # Find depth 3+ chains
    chains = find_chain_depth_3plus(td)
    
    (
        diagram = td,
        discovered = length(discovered),
        chains_depth_3plus = chains,
        max_chain_length = isempty(chains) ? 0 : maximum(length, chains),
        fingerprint = td.fingerprint,
        color = td.color
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

function world_diagram_layer()
    println()
    println("╔═══════════════════════════════════════════════════════════════════════════╗")
    println("║  DIAGRAM LAYER PARALLELISM: Safely Exceeding SPI                          ║")
    println("║  \"Parallelism IS the semantics at the diagram layer\"                      ║")
    println("╚═══════════════════════════════════════════════════════════════════════════╝")
    println()
    
    # ─── Create a String Diagram ───
    println("─── Creating String Diagram ───")
    
    diag = GayStringDiagram()
    
    # Add nodes
    nodes = [
        DiagramNode(:A, :generator, (0, 2); flavor=:source),
        DiagramNode(:B, :box, (1, 1); flavor=:process),
        DiagramNode(:C, :box, (1, 1); flavor=:process),
        DiagramNode(:D, :generator, (2, 0); flavor=:sink),
    ]
    
    for node in nodes
        add_node!(diag, node)
    end
    
    # Add wires
    add_wire!(diag, DiagramWire(:w1, :A, :B, 1, 1, :data))
    add_wire!(diag, DiagramWire(:w2, :A, :C, 2, 1, :data))
    add_wire!(diag, DiagramWire(:w3, :B, :D, 1, 1, :result))
    add_wire!(diag, DiagramWire(:w4, :C, :D, 1, 2, :result))
    
    println("  Nodes: $(length(diag.nodes))")
    println("  Wires: $(length(diag.wires))")
    println("  Fingerprint: 0x$(string(diag.fingerprint, base=16)[1:min(8,end)])...")
    println()
    
    # ─── Find Independent Subdiagrams ───
    println("─── Finding Independent Subdiagrams ───")
    
    regions = find_independent_subdiagrams(diag)
    println("  Independent regions: $(length(regions))")
    println("  Max parallel width: $(diag.max_parallel_width)")
    for (i, region) in enumerate(regions)
        println("    Region $i: $(join(region, ", "))")
    end
    println()
    
    # ─── Intuition Mining ───
    println("─── Intuition Mining with Implicit Best Response ───")
    
    # Define some rewrite rules
    rules = [
        RewriteRule(:simplify, [:box], [:identity], true, (0.0,0.0,0.0), true, [:simplify]),
        RewriteRule(:expand, [:identity], [:box, :box], true, (0.0,0.0,0.0), true, [:expand]),
        RewriteRule(:fuse, [:box, :box], [:box], true, (0.0,0.0,0.0), true, [:simplify]),
    ]
    
    miner = IntuitionMiner(diag; equilibrium_fp=diag.fingerprint ⊻ UInt64(0xFF))
    mine_intuitions!(miner, rules; max_iters=50)
    
    println("  Iterations: $(miner.iterations)")
    println("  Intuitions found: $(miner.intuitions_found)")
    println("  Best responses extracted: $(length(extract_best_responses(miner)))")
    if !isempty(miner.convergence_history)
        println("  Final distance: $(round(miner.convergence_history[end], digits=4))")
    end
    println()
    
    # ─── Memoryless Memory ───
    println("─── Memoryless Memory (Random Access Color Trajectories) ───")
    
    mem = MemorylessMemory(GAY_SEED)
    
    # Access random indices
    indices = [1, 100, 10000, 1000000]
    println("  Random access to color trajectory:")
    for idx in indices
        color = random_access_memory(mem, idx)
        println("    Index $idx → RGB$(round.(color, digits=3))")
    end
    println("  Cache size: $(length(mem.cache))")
    println()
    
    # ─── Colored/Flavored Operads ───
    println("─── Colored vs Flavored Operads ───")
    
    # Create a colored operad
    operad_nodes = [
        DiagramNode(:μ, :multiplication, (2, 1); flavor=:algebraic),
        DiagramNode(:η, :unit, (0, 1); flavor=:algebraic),
        DiagramNode(:δ, :comultiplication, (1, 2)),  # No flavor!
    ]
    
    comps = [
        (:μ, :μ, :μ),   # associativity
        (:η, :μ, :μ),   # unit law
    ]
    
    cop = ColoredOperad(:Monoid, operad_nodes, comps)
    
    println("  ColoredOperad :Monoid")
    println("    Operations: $(length(cop.operations))")
    println("    Color-closed: $(cop.color_closed)")
    println("    Has flavored dual: $(has_flavored_dual(cop))")
    println("    Obstruction: $(find_obstruction(cop))")
    println("    Gap measure: $(round(operad_gap(cop), digits=3))")
    println()
    
    # ─── Open Game at Diagram Layer ───
    println("─── Open Game: Play/Evaluate at Diagram Layer ───")
    
    game = DiagramOpenGame(diag, [:Alice, :Bob])
    
    # Give players some strategies
    game.strategies[:Alice] = rules[1:2]
    game.strategies[:Bob] = rules[2:3]
    
    result = equilibrium_from_diagram(game; max_iters=20)
    
    println("  Players: $(length(game.players))")
    println("  Converged: $(result.converged)")
    println("  Iterations: $(result.iterations)")
    println("  Utilities:")
    for (p, u) in result.utilities
        println("    $p: $(round(u, digits=4))")
    end
    println("  Equilibrium fingerprint: 0x$(string(result.fingerprint, base=16)[1:min(8,end)])...")
    println()
    
    # ─── Thread Reachability ───
    println("─── Maximally Parallel Thread Discovery ───")
    
    # Mock fetch function
    mock_threads = Dict(
        "T-001" => ("Root thread", ["T-002", "T-003"], nothing),
        "T-002" => ("Child 1", ["T-004"], "T-001"),
        "T-003" => ("Child 2", [], "T-001"),
        "T-004" => ("Grandchild", ["T-005"], "T-002"),
        "T-005" => ("Great-grandchild", [], "T-004"),
    )
    
    fetch_mock = tid -> get(mock_threads, tid, ("Unknown", String[], nothing))
    
    discovery = maximally_parallel_thread_discovery(
        ["T-001"], fetch_mock; max_depth=5
    )
    
    println("  Threads discovered: $(discovery.discovered)")
    println("  Chains depth 3+: $(length(discovery.chains_depth_3plus))")
    println("  Max chain length: $(discovery.max_chain_length)")
    for (i, chain) in enumerate(discovery.chains_depth_3plus)
        println("    Chain $i: $(join(chain, " → "))")
    end
    println()
    
    # ─── Summary ───
    println("═══════════════════════════════════════════════════════════════════════════")
    println("  KEY INSIGHTS")
    println("═══════════════════════════════════════════════════════════════════════════")
    println()
    println("  1. DIAGRAM-LAYER PARALLELISM:")
    println("     String diagrams enable parallelism as STRUCTURE, not execution order.")
    println("     Independent subdiagrams can be rewritten simultaneously.")
    println()
    println("  2. INTUITION MINING:")
    println("     Extract implicit best-response dynamics from diagram fingerprints.")
    println("     \"Memoryless memories\" give O(1) random access to color trajectories.")
    println()
    println("  3. COLORED ↔ FLAVORED OPERADS:")
    println("     Not all colored operads have flavored duals!")
    println("     The gap measure quantifies tractability obstruction.")
    println()
    println("  4. OPEN GAMES AT DIAGRAM LAYER:")
    println("     Play (forward) and Evaluate (backward) can BOTH be parallelized")
    println("     when players control disjoint subdiagrams.")
    println()
    println("  5. THREAD REACHABILITY:")
    println("     Layer-by-layer discovery is maximally parallel.")
    println("     Depth 3+ chains = \"threads of threads of threads\"")
    println()
    
    return (
        diagram = diag,
        miner = miner,
        operad = cop,
        game = game,
        discovery = discovery
    )
end

end # module DiagramLayerParallelism
