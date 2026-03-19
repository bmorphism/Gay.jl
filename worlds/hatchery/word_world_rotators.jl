# Word↔World Rotators: CNOT Agency over Information Network Entanglement
# ═══════════════════════════════════════════════════════════════════════════════
#
# "From Word Models to World Models" (Wong et al., 2023) → and back again
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  WORD ↔ WORLD ROTATION ALGEBRA                                              │
# │                                                                             │
# │                    ┌─────────────┐                                          │
# │        WORD ──────▶│   ROTATE    │──────▶ WORLD                            │
# │       MODELS       │   ROTATORS  │        MODELS                            │
# │         │          │   ROTATORS  │          │                               │
# │         │          └─────────────┘          │                               │
# │         │                ▲                  │                               │
# │         └────────────────┴──────────────────┘                               │
# │                    (bidirectional)                                          │
# │                                                                             │
# │  CNOT CHOICE AGENCY:                                                        │
# │    • Control qubit: Observer's perspective (word vs world)                  │
# │    • Target qubit: Content being transformed                                │
# │    • Which CNOT? Chosen to maximize entanglement in info network           │
# │                                                                             │
# │  PROBABILISTIC LANGUAGE OF THOUGHT (PLoT):                                 │
# │    Wong et al.: LLMs translate NL → probabilistic programs                  │
# │    Gay.jl: Probabilistic programs → chromatic circuits                      │
# │    Combined: NL → PLoT → Color → Fingerprint → Verification               │
# │                                                                             │
# │  IMPACT NETWORK:                                                            │
# │    Paper → Citations → Reinterpretations → Extensions → Back-citations     │
# │    Each node: observer who engaged with word↔world transformation          │
# │    Edges: CNOT entanglements (control = citing, target = cited)            │
# │                                                                             │
# │  GAYMC EFFICIENCY:                                                          │
# │    Maximum parallelism across all trajectory affordances                    │
# │    Bang out probability circuits via splittable chromatic walks            │
# └─────────────────────────────────────────────────────────────────────────────┘

module WordWorldRotators

export
    # Core types
    WordModel, WorldModel, Rotator, RotateRotator,
    TworldWolder, Observer, ImpactNetwork,
    
    # CNOT choice agency
    CNOTChoice, CNOTRegistry, EntanglementScore,
    choose_cnot!, entangle_max!, agency_over_cnot,
    
    # Word ↔ World transformation
    word_to_world, world_to_word, rotate!, rotate_rotate!,
    bidirectional_transform, transformation_fingerprint,
    
    # PLoT integration (Probabilistic Language of Thought)
    PLoTProgram, PLoTStatement, PLoTDistribution,
    nl_to_plot, plot_to_color, color_to_circuit,
    
    # Impact network
    PaperNode, CitationEdge, ReinterpretationPath,
    build_impact_network, trace_reinterpretations,
    observer_engagement, network_entanglement,
    
    # GayMC probability circuits
    ProbabilityCircuit, AffordanceTrajectory, CircuitGate,
    gaymc_bang!, parallel_circuit_eval, trajectory_affordances,
    
    # Color wheel of world rotators
    ColorWheel, WorldRotatorWheel, DirectionDesignator,
    rotate_on_wheel!, wheel_position, direction_from_color,
    
    # Maximum sharing
    SubstrateShare, MaximalSharing, SharedSubexpression,
    find_sharing!, deduplicate_subexpressions, sharing_fingerprint,
    
    # Demo
    demo_word_world_rotators

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

const GAY_SEED = UInt64(1069)
const WORD_SEED = UInt64(0x00124D)   # "WORD"
const WORLD_SEED = UInt64(0x01D)     # "WORLD" truncated
const PLOT_SEED = UInt64(0x910D)     # "PLoT"

# Direction designators
const DIR_MINUS = :minus    # -
const DIR_PLUS = :plus      # +
const DIR_NEUTRAL = :neutral # _

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

@inline function hue_from_color(color::NamedTuple)::Float64
    # Simplified hue extraction
    r, g, b = color.r, color.g, color.b
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    
    if max_c == min_c
        0.0
    elseif max_c == r
        60.0 * mod((g - b) / (max_c - min_c), 6)
    elseif max_c == g
        60.0 * ((b - r) / (max_c - min_c) + 2)
    else
        60.0 * ((r - g) / (max_c - min_c) + 4)
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# WORD AND WORLD MODELS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    WordModel

A linguistic model operating on token sequences.
Word models predict and generate language.
"""
struct WordModel
    name::Symbol
    vocabulary_size::Int
    embedding_dim::Int
    context_length::Int
    
    # Chromatic identity
    seed::UInt64
    color::NamedTuple{(:r, :g, :b), NTuple{3, Float64}}
    
    # Model fingerprint (for SPI verification)
    fingerprint::UInt64
end

function WordModel(name::Symbol; vocab::Int=50000, dim::Int=768, ctx::Int=2048)
    seed = splitmix64(WORD_SEED ⊻ hash(name))
    color = color_from_seed(seed)
    fp = seed ⊻ UInt64(vocab) ⊻ (UInt64(dim) << 16) ⊻ (UInt64(ctx) << 32)
    WordModel(name, vocab, dim, ctx, seed, color, fp)
end

"""
    WorldModel

A grounded model that reasons about reality.
World models simulate and predict physical/social dynamics.
"""
struct WorldModel
    name::Symbol
    state_dim::Int
    action_dim::Int
    observation_dim::Int
    
    # Probabilistic components
    prior::Symbol           # Prior distribution type
    likelihood::Symbol      # Observation model
    transition::Symbol      # Dynamics model
    
    # Chromatic identity
    seed::UInt64
    color::NamedTuple{(:r, :g, :b), NTuple{3, Float64}}
    
    # Model fingerprint
    fingerprint::UInt64
end

function WorldModel(name::Symbol; state::Int=64, action::Int=8, obs::Int=128)
    seed = splitmix64(WORLD_SEED ⊻ hash(name))
    color = color_from_seed(seed)
    fp = seed ⊻ UInt64(state) ⊻ (UInt64(action) << 16) ⊻ (UInt64(obs) << 32)
    WorldModel(name, state, action, obs, :gaussian, :categorical, :neural,
               seed, color, fp)
end

# ═══════════════════════════════════════════════════════════════════════════════
# ROTATORS: Transform Word ↔ World
# ═══════════════════════════════════════════════════════════════════════════════

"""
    Rotator

Transforms between word and world representations.
A rotator is a bidirectional morphism with chromatic tracking.
"""
mutable struct Rotator
    name::Symbol
    direction::Symbol  # :word_to_world, :world_to_word, :bidirectional
    
    # Rotation parameters
    angle::Float64     # In radians
    axis::NTuple{3, Float64}  # Rotation axis in color space
    
    # State
    rotations_applied::Int
    accumulated_angle::Float64
    
    # Chromatic identity
    seed::UInt64
    color::NamedTuple{(:r, :g, :b), NTuple{3, Float64}}
end

function Rotator(name::Symbol; direction::Symbol=:bidirectional, angle::Float64=π/4)
    seed = splitmix64(GAY_SEED ⊻ hash(name))
    color = color_from_seed(seed)
    
    # Axis from seed
    state = splitmix64(seed)
    ax = ((state >> 48) / 65535.0, ((state >> 32) & 0xFFFF) / 65535.0, ((state >> 16) & 0xFFFF) / 65535.0)
    norm = sqrt(sum(x^2 for x in ax))
    normalized_ax = (ax[1]/norm, ax[2]/norm, ax[3]/norm)
    
    Rotator(name, direction, angle, normalized_ax, 0, 0.0, seed, color)
end

"""
    RotateRotator

A meta-rotator that rotates rotators themselves.
Enables higher-order transformation of word↔world mappings.
"""
mutable struct RotateRotator
    name::Symbol
    base_rotators::Vector{Rotator}
    
    # Meta-rotation
    meta_angle::Float64
    composition_order::Vector{Int}  # Which order to apply rotators
    
    seed::UInt64
    color::NamedTuple{(:r, :g, :b), NTuple{3, Float64}}
end

function RotateRotator(name::Symbol, rotators::Vector{Rotator})
    seed = reduce(⊻, [r.seed for r in rotators]; init=GAY_SEED)
    RotateRotator(name, rotators, π/6, collect(1:length(rotators)),
                  seed, color_from_seed(seed))
end

"""
    TworldWolder

The unified space where word and world meet.
"tworld" = twisted world, "wolder" = word + holder.
"""
struct TworldWolder
    word_component::WordModel
    world_component::WorldModel
    rotator::Rotator
    
    # Entanglement measure
    entanglement::Float64  # 0 = separable, 1 = maximally entangled
    
    # Combined state
    seed::UInt64
    color::NamedTuple{(:r, :g, :b), NTuple{3, Float64}}
end

function TworldWolder(word::WordModel, world::WorldModel, rotator::Rotator)
    combined_seed = word.seed ⊻ world.seed ⊻ rotator.seed
    
    # Entanglement from color similarity
    color_dist = sqrt((word.color.r - world.color.r)^2 +
                      (word.color.g - world.color.g)^2 +
                      (word.color.b - world.color.b)^2)
    entanglement = 1.0 - color_dist / sqrt(3)
    
    TworldWolder(word, world, rotator, entanglement, combined_seed,
                 color_from_seed(combined_seed))
end

# ═══════════════════════════════════════════════════════════════════════════════
# CNOT CHOICE AGENCY
# ═══════════════════════════════════════════════════════════════════════════════

"""
    CNOTChoice

A choice of which CNOT gate to apply for entanglement.
Agency = control over which control-target pair to select.
"""
struct CNOTChoice
    control_idx::Int
    target_idx::Int
    
    # Entanglement score if this CNOT is applied
    predicted_entanglement::Float64
    
    # Information gain
    info_bits::Float64
    
    seed::UInt64
end

"""
    CNOTRegistry

Registry of available CNOT choices and their effects.
"""
mutable struct CNOTRegistry
    nodes::Vector{UInt64}  # Node seeds (observers/papers)
    
    # Available CNOTs (control, target pairs)
    available_cnots::Vector{Tuple{Int, Int}}
    
    # Applied CNOTs
    applied_cnots::Vector{CNOTChoice}
    
    # Current entanglement matrix
    entanglement_matrix::Matrix{Float64}
    
    seed::UInt64
end

function CNOTRegistry(n_nodes::Int; seed::UInt64=GAY_SEED)
    nodes = [splitmix64(seed ⊻ UInt64(i)) for i in 1:n_nodes]
    
    # All possible CNOTs (excluding self-loops)
    cnots = [(i, j) for i in 1:n_nodes for j in 1:n_nodes if i != j]
    
    # Initial entanglement: diagonal = 1, off-diagonal = 0
    ent_matrix = zeros(Float64, n_nodes, n_nodes)
    for i in 1:n_nodes
        ent_matrix[i, i] = 1.0
    end
    
    CNOTRegistry(nodes, cnots, CNOTChoice[], ent_matrix, seed)
end

"""
    EntanglementScore

Score for a potential CNOT application.
"""
struct EntanglementScore
    cnot::Tuple{Int, Int}
    current_entanglement::Float64
    potential_entanglement::Float64
    gain::Float64
    info_bits::Float64
end

function score_cnot(registry::CNOTRegistry, control::Int, target::Int)::EntanglementScore
    current = registry.entanglement_matrix[control, target]
    
    # Potential entanglement: XOR-based mixing
    control_seed = registry.nodes[control]
    target_seed = registry.nodes[target]
    mixed = splitmix64(control_seed ⊻ target_seed)
    
    # Entanglement from popcount (closer to 32 = more entangled)
    popcount = count_ones(mixed)
    potential = 1.0 - abs(popcount - 32) / 32
    
    # Information gain
    info_bits = potential > current ? log2(1 + potential - current) : 0.0
    
    EntanglementScore((control, target), current, potential, potential - current, info_bits)
end

function choose_cnot!(registry::CNOTRegistry)::CNOTChoice
    # Score all available CNOTs
    scores = [score_cnot(registry, c, t) for (c, t) in registry.available_cnots]
    
    # Choose the one with maximum gain
    best_idx = argmax(s -> s.gain, scores)
    best = scores[best_idx]
    
    choice = CNOTChoice(best.cnot[1], best.cnot[2],
                        best.potential_entanglement, best.info_bits,
                        splitmix64(registry.seed ⊻ UInt64(best_idx)))
    
    # Apply the CNOT
    registry.entanglement_matrix[best.cnot[1], best.cnot[2]] = best.potential_entanglement
    registry.entanglement_matrix[best.cnot[2], best.cnot[1]] = best.potential_entanglement
    push!(registry.applied_cnots, choice)
    
    # Remove from available (optional: could allow re-application)
    filter!(c -> c != best.cnot, registry.available_cnots)
    
    registry.seed = splitmix64(registry.seed)
    
    choice
end

function entangle_max!(registry::CNOTRegistry, n_cnots::Int)::Vector{CNOTChoice}
    choices = CNOTChoice[]
    for _ in 1:min(n_cnots, length(registry.available_cnots))
        push!(choices, choose_cnot!(registry))
    end
    choices
end

function agency_over_cnot(registry::CNOTRegistry)::Float64
    # Agency = how much control do we have over entanglement?
    # Measured by entropy of choice distribution
    n_available = length(registry.available_cnots)
    n_available == 0 && return 0.0
    
    scores = [score_cnot(registry, c, t).gain for (c, t) in registry.available_cnots]
    total = sum(max.(scores, 0.0))
    total == 0 && return 1.0 / n_available
    
    probs = max.(scores, 0.0) ./ total
    entropy = -sum(p * log2(p + 1e-10) for p in probs)
    max_entropy = log2(n_available)
    
    1.0 - entropy / max_entropy  # High agency = low entropy (clear best choice)
end

# ═══════════════════════════════════════════════════════════════════════════════
# WORD ↔ WORLD TRANSFORMATION
# ═══════════════════════════════════════════════════════════════════════════════

function word_to_world(word::WordModel, rotator::Rotator)::WorldModel
    rotator.rotations_applied += 1
    rotator.accumulated_angle += rotator.angle
    
    # Transform dimensions
    state_dim = word.embedding_dim
    action_dim = max(1, word.vocabulary_size ÷ 1000)
    obs_dim = word.context_length
    
    # New seed via rotation
    new_seed = splitmix64(word.seed ⊻ rotator.seed ⊻ UInt64(round(rotator.angle * 1000)))
    
    WorldModel(Symbol("world_from_$(word.name)");
               state=state_dim, action=action_dim, obs=obs_dim)
end

function world_to_word(world::WorldModel, rotator::Rotator)::WordModel
    rotator.rotations_applied += 1
    rotator.accumulated_angle -= rotator.angle  # Reverse rotation
    
    # Transform dimensions (inverse of word_to_world)
    vocab = world.action_dim * 1000
    dim = world.state_dim
    ctx = world.observation_dim
    
    WordModel(Symbol("word_from_$(world.name)"); vocab=vocab, dim=dim, ctx=ctx)
end

function rotate!(rotator::Rotator, color::NamedTuple)::NamedTuple
    # Rotate color around rotator's axis by its angle
    # Simplified Rodrigues' rotation formula
    k = rotator.axis
    θ = rotator.angle
    v = (color.r, color.g, color.b)
    
    cos_θ = cos(θ)
    sin_θ = sin(θ)
    
    # k × v
    cross = (k[2]*v[3] - k[3]*v[2],
             k[3]*v[1] - k[1]*v[3],
             k[1]*v[2] - k[2]*v[1])
    
    # k · v
    dot = k[1]*v[1] + k[2]*v[2] + k[3]*v[3]
    
    # Rotated: v*cos(θ) + (k×v)*sin(θ) + k*(k·v)*(1-cos(θ))
    rotated = (
        v[1]*cos_θ + cross[1]*sin_θ + k[1]*dot*(1-cos_θ),
        v[2]*cos_θ + cross[2]*sin_θ + k[2]*dot*(1-cos_θ),
        v[3]*cos_θ + cross[3]*sin_θ + k[3]*dot*(1-cos_θ)
    )
    
    rotator.rotations_applied += 1
    rotator.accumulated_angle += θ
    
    (r=clamp(rotated[1], 0.0, 1.0),
     g=clamp(rotated[2], 0.0, 1.0),
     b=clamp(rotated[3], 0.0, 1.0))
end

function rotate_rotate!(rr::RotateRotator, color::NamedTuple)::NamedTuple
    current = color
    for idx in rr.composition_order
        current = rotate!(rr.base_rotators[idx], current)
    end
    
    # Apply meta-rotation to composition order
    rr.meta_angle += π/12
    if rr.meta_angle > 2π
        rr.meta_angle -= 2π
        # Shuffle composition order
        n = length(rr.composition_order)
        shift = Int(floor(rr.meta_angle / (2π/n))) % n
        rr.composition_order = circshift(rr.composition_order, shift)
    end
    
    current
end

function bidirectional_transform(tw::TworldWolder, input_color::NamedTuple;
                                  n_cycles::Int=1)::Vector{NamedTuple}
    colors = [input_color]
    current = input_color
    
    for _ in 1:n_cycles
        # Word → World
        current = rotate!(tw.rotator, current)
        push!(colors, current)
        
        # World → Word (reverse)
        tw.rotator.angle = -tw.rotator.angle
        current = rotate!(tw.rotator, current)
        push!(colors, current)
        tw.rotator.angle = -tw.rotator.angle  # Restore
    end
    
    colors
end

function transformation_fingerprint(colors::Vector{NamedTuple})::UInt64
    reduce(⊻, [UInt64(round(c.r * 255)) << 48 ⊻
               UInt64(round(c.g * 255)) << 32 ⊻
               UInt64(round(c.b * 255)) << 16 for c in colors]; init=GAY_SEED)
end

# ═══════════════════════════════════════════════════════════════════════════════
# PLoT: PROBABILISTIC LANGUAGE OF THOUGHT
# ═══════════════════════════════════════════════════════════════════════════════

"""
    PLoTStatement

A statement in the Probabilistic Language of Thought.
"""
struct PLoTStatement
    type::Symbol  # :sample, :observe, :condition, :return
    expression::Expr
    distribution::Union{Symbol, Nothing}
    seed::UInt64
end

"""
    PLoTProgram

A probabilistic program in PLoT.
"""
struct PLoTProgram
    statements::Vector{PLoTStatement}
    free_variables::Vector{Symbol}
    observed_variables::Vector{Symbol}
    
    # Chromatic encoding
    seed::UInt64
    color::NamedTuple{(:r, :g, :b), NTuple{3, Float64}}
end

"""
    PLoTDistribution

A distribution over PLoT programs (posterior after conditioning).
"""
struct PLoTDistribution
    programs::Vector{PLoTProgram}
    weights::Vector{Float64}
    
    seed::UInt64
end

function nl_to_plot(text::String; seed::UInt64=PLOT_SEED)::PLoTProgram
    # Simplified: tokenize and create probabilistic statements
    words = split(lowercase(text))
    statements = PLoTStatement[]
    
    text_seed = reduce((h, w) -> splitmix64(h ⊻ hash(w)), words; init=seed)
    
    # Detect probabilistic keywords
    for (i, word) in enumerate(words)
        word_seed = splitmix64(text_seed ⊻ UInt64(i))
        
        if word in ["maybe", "probably", "might", "could"]
            push!(statements, PLoTStatement(:sample, :(x ~ Bernoulli(0.7)), :bernoulli, word_seed))
        elseif word in ["is", "was", "are", "were"]
            push!(statements, PLoTStatement(:observe, :(observe(y)), nothing, word_seed))
        elseif word in ["if", "when", "given"]
            push!(statements, PLoTStatement(:condition, :(condition(c)), nothing, word_seed))
        elseif word in ["then", "so", "therefore"]
            push!(statements, PLoTStatement(:return, :(return result), nothing, word_seed))
        end
    end
    
    # Ensure at least one statement
    if isempty(statements)
        push!(statements, PLoTStatement(:return, :(return nothing), nothing, text_seed))
    end
    
    PLoTProgram(statements, [:x, :y], [:y], text_seed, color_from_seed(text_seed))
end

function plot_to_color(program::PLoTProgram)::NamedTuple
    # Color from program structure
    n_samples = count(s -> s.type == :sample, program.statements)
    n_observes = count(s -> s.type == :observe, program.statements)
    n_conditions = count(s -> s.type == :condition, program.statements)
    
    # Heuristic coloring
    r = n_samples / max(1, length(program.statements))
    g = n_observes / max(1, length(program.statements))
    b = n_conditions / max(1, length(program.statements))
    
    total = r + g + b
    if total > 0
        (r=r/total, g=g/total, b=b/total)
    else
        program.color
    end
end

function color_to_circuit(color::NamedTuple; n_qubits::Int=3)
    # Convert color to quantum circuit gates
    # R → RX rotation, G → RY rotation, B → RZ rotation
    gates = Tuple{Symbol, Int, Float64}[]  # (gate_type, qubit, angle)
    
    push!(gates, (:RX, 1, color.r * π))
    push!(gates, (:RY, 2, color.g * π))
    push!(gates, (:RZ, 3, color.b * π))
    
    # Add entangling gates based on color relationships
    if abs(color.r - color.g) < 0.3
        push!(gates, (:CNOT, 1, 2.0))  # Entangle 1-2
    end
    if abs(color.g - color.b) < 0.3
        push!(gates, (:CNOT, 2, 3.0))  # Entangle 2-3
    end
    
    gates
end

# ═══════════════════════════════════════════════════════════════════════════════
# IMPACT NETWORK
# ═══════════════════════════════════════════════════════════════════════════════

"""
    Observer

An observer who engaged with word↔world transformation.
"""
struct Observer
    id::Int
    name::String
    engagement_type::Symbol  # :author, :citer, :reinterpreter, :extender
    
    # Contribution
    word_contribution::Float64  # 0-1
    world_contribution::Float64  # 0-1
    
    seed::UInt64
    color::NamedTuple{(:r, :g, :b), NTuple{3, Float64}}
end

function Observer(id::Int, name::String; engagement::Symbol=:citer)
    seed = splitmix64(GAY_SEED ⊻ UInt64(id) ⊻ hash(name))
    
    # Contributions based on engagement type
    word_c, world_c = if engagement == :author
        0.5, 0.5
    elseif engagement == :citer
        0.7, 0.3
    elseif engagement == :reinterpreter
        0.4, 0.6
    else  # :extender
        0.3, 0.7
    end
    
    Observer(id, name, engagement, word_c, world_c, seed, color_from_seed(seed))
end

"""
    PaperNode

A paper in the impact network.
"""
struct PaperNode
    arxiv_id::String
    title::String
    authors::Vector{Observer}
    year::Int
    
    # Word/World balance
    word_focus::Float64
    world_focus::Float64
    
    seed::UInt64
    color::NamedTuple{(:r, :g, :b), NTuple{3, Float64}}
end

function PaperNode(arxiv_id::String, title::String, authors::Vector{Observer}, year::Int)
    seed = splitmix64(GAY_SEED ⊻ hash(arxiv_id))
    
    # Balance from author contributions
    word_focus = isempty(authors) ? 0.5 : mean([a.word_contribution for a in authors])
    world_focus = 1.0 - word_focus
    
    PaperNode(arxiv_id, title, authors, year, word_focus, world_focus,
              seed, color_from_seed(seed))
end

"""
    CitationEdge

A citation between papers (CNOT in information network).
"""
struct CitationEdge
    source::PaperNode  # Citing paper (control)
    target::PaperNode  # Cited paper (target)
    citation_type::Symbol  # :builds_on, :critiques, :uses, :extends
    
    # CNOT interpretation
    control_bit::Bool  # Source flips target?
    entanglement::Float64
    
    seed::UInt64
end

function CitationEdge(source::PaperNode, target::PaperNode; ctype::Symbol=:builds_on)
    seed = splitmix64(source.seed ⊻ target.seed)
    
    # Control bit: does source significantly change interpretation of target?
    control_bit = ctype in [:critiques, :extends]
    
    # Entanglement from color similarity
    color_dist = sqrt((source.color.r - target.color.r)^2 +
                      (source.color.g - target.color.g)^2 +
                      (source.color.b - target.color.b)^2)
    entanglement = 1.0 - color_dist / sqrt(3)
    
    CitationEdge(source, target, ctype, control_bit, entanglement, seed)
end

"""
    ReinterpretationPath

A path of reinterpretations through the network.
"""
struct ReinterpretationPath
    papers::Vector{PaperNode}
    edges::Vector{CitationEdge}
    
    # Transformation tracking
    initial_word_focus::Float64
    final_world_focus::Float64
    
    # Did it complete the word↔world↔word cycle?
    is_cycle::Bool
    
    fingerprint::UInt64
end

"""
    ImpactNetwork

The full network of papers, citations, and reinterpretations.
"""
mutable struct ImpactNetwork
    papers::Dict{String, PaperNode}
    citations::Vector{CitationEdge}
    
    # CNOT registry for this network
    cnot_registry::CNOTRegistry
    
    # Reinterpretation paths
    paths::Vector{ReinterpretationPath}
    
    seed::UInt64
end

function build_impact_network(origin_paper::PaperNode)::ImpactNetwork
    papers = Dict(origin_paper.arxiv_id => origin_paper)
    
    # Create CNOT registry with paper nodes
    registry = CNOTRegistry(1; seed=origin_paper.seed)
    
    ImpactNetwork(papers, CitationEdge[], registry, ReinterpretationPath[],
                  origin_paper.seed)
end

function add_paper!(network::ImpactNetwork, paper::PaperNode)
    network.papers[paper.arxiv_id] = paper
    push!(network.cnot_registry.nodes, paper.seed)
    
    # Update CNOT registry
    n = length(network.cnot_registry.nodes)
    for i in 1:n-1
        push!(network.cnot_registry.available_cnots, (i, n))
        push!(network.cnot_registry.available_cnots, (n, i))
    end
    
    # Expand entanglement matrix
    old_size = size(network.cnot_registry.entanglement_matrix, 1)
    new_matrix = zeros(n, n)
    new_matrix[1:old_size, 1:old_size] = network.cnot_registry.entanglement_matrix
    new_matrix[n, n] = 1.0
    network.cnot_registry.entanglement_matrix = new_matrix
end

function add_citation!(network::ImpactNetwork, source_id::String, target_id::String;
                       ctype::Symbol=:builds_on)
    source = network.papers[source_id]
    target = network.papers[target_id]
    edge = CitationEdge(source, target; ctype=ctype)
    push!(network.citations, edge)
    edge
end

function trace_reinterpretations(network::ImpactNetwork, start_id::String;
                                  max_depth::Int=10)::Vector{ReinterpretationPath}
    paths = ReinterpretationPath[]
    
    start_paper = network.papers[start_id]
    initial_word = start_paper.word_focus
    
    # BFS to find paths
    queue = [(start_id, [start_paper], CitationEdge[])]
    
    while !isempty(queue) && length(paths) < 100
        current_id, paper_path, edge_path = popfirst!(queue)
        
        if length(paper_path) > max_depth
            continue
        end
        
        # Find outgoing citations
        outgoing = filter(e -> e.source.arxiv_id == current_id, network.citations)
        
        if isempty(outgoing)
            # End of path
            final_world = paper_path[end].world_focus
            is_cycle = abs(final_world - initial_word) < 0.2  # Returned to similar balance
            fp = reduce(⊻, [p.seed for p in paper_path]; init=GAY_SEED)
            
            push!(paths, ReinterpretationPath(paper_path, edge_path,
                                              initial_word, final_world, is_cycle, fp))
        else
            for edge in outgoing
                if edge.target.arxiv_id ∉ [p.arxiv_id for p in paper_path]
                    push!(queue, (edge.target.arxiv_id,
                                  vcat(paper_path, [edge.target]),
                                  vcat(edge_path, [edge])))
                end
            end
        end
    end
    
    paths
end

function observer_engagement(network::ImpactNetwork)::Dict{Int, Float64}
    engagement = Dict{Int, Float64}()
    
    for (_, paper) in network.papers
        for author in paper.authors
            current = get(engagement, author.id, 0.0)
            engagement[author.id] = current + 1.0
        end
    end
    
    # Normalize by max
    max_eng = maximum(values(engagement); init=1.0)
    Dict(k => v / max_eng for (k, v) in engagement)
end

function network_entanglement(network::ImpactNetwork)::Float64
    n = length(network.cnot_registry.nodes)
    n < 2 && return 0.0
    
    # Average off-diagonal entanglement
    total = 0.0
    count = 0
    for i in 1:n
        for j in 1:n
            if i != j
                total += network.cnot_registry.entanglement_matrix[i, j]
                count += 1
            end
        end
    end
    
    total / max(1, count)
end

# ═══════════════════════════════════════════════════════════════════════════════
# GAYMC PROBABILITY CIRCUITS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    CircuitGate

A gate in a probability circuit.
"""
struct CircuitGate
    type::Symbol  # :H, :X, :Y, :Z, :CNOT, :RX, :RY, :RZ, :MEASURE
    qubits::Vector{Int}
    angle::Union{Float64, Nothing}
    probability::Float64  # Probability this gate is applied
    seed::UInt64
end

"""
    ProbabilityCircuit

A quantum-inspired probability circuit with stochastic gates.
"""
mutable struct ProbabilityCircuit
    n_qubits::Int
    gates::Vector{CircuitGate}
    
    # State: probability amplitudes
    amplitudes::Vector{ComplexF64}
    
    # Execution trace
    applied_gates::Vector{CircuitGate}
    measurements::Vector{Int}
    
    seed::UInt64
    color::NamedTuple{(:r, :g, :b), NTuple{3, Float64}}
end

function ProbabilityCircuit(n_qubits::Int; seed::UInt64=GAY_SEED)
    # 2^n amplitudes for n qubits
    n_states = 2^n_qubits
    amplitudes = zeros(ComplexF64, n_states)
    amplitudes[1] = 1.0  # Start in |00...0⟩
    
    ProbabilityCircuit(n_qubits, CircuitGate[], amplitudes, CircuitGate[], Int[],
                       seed, color_from_seed(seed))
end

"""
    AffordanceTrajectory

A trajectory through affordance space.
"""
struct AffordanceTrajectory
    circuit::ProbabilityCircuit
    steps::Vector{Tuple{CircuitGate, Bool}}  # (gate, was_applied)
    
    # Trajectory metrics
    total_probability::Float64
    information_gain::Float64
    
    seed::UInt64
    colors::Vector{NamedTuple{(:r, :g, :b), NTuple{3, Float64}}}
end

function trajectory_affordances(circuit::ProbabilityCircuit)::Vector{CircuitGate}
    # What gates can we apply next?
    affordances = CircuitGate[]
    
    for q in 1:circuit.n_qubits
        # Single-qubit gates
        push!(affordances, CircuitGate(:H, [q], nothing, 0.5, splitmix64(circuit.seed ⊻ UInt64(q))))
        push!(affordances, CircuitGate(:X, [q], nothing, 0.3, splitmix64(circuit.seed ⊻ UInt64(q) << 8)))
        
        # Rotation gates
        angle = (circuit.seed >> 48) / 65535.0 * π
        push!(affordances, CircuitGate(:RX, [q], angle, 0.4, splitmix64(circuit.seed ⊻ UInt64(q) << 16)))
    end
    
    # Two-qubit gates
    for q1 in 1:circuit.n_qubits-1
        for q2 in q1+1:circuit.n_qubits
            push!(affordances, CircuitGate(:CNOT, [q1, q2], nothing, 0.6,
                                           splitmix64(circuit.seed ⊻ UInt64(q1) ⊻ (UInt64(q2) << 8))))
        end
    end
    
    affordances
end

function gaymc_bang!(circuit::ProbabilityCircuit, n_steps::Int)::AffordanceTrajectory
    steps = Tuple{CircuitGate, Bool}[]
    colors = [circuit.color]
    total_prob = 1.0
    info_gain = 0.0
    
    for step in 1:n_steps
        affordances = trajectory_affordances(circuit)
        isempty(affordances) && break
        
        # Choose gate based on probability and seed
        circuit.seed = splitmix64(circuit.seed)
        threshold = (circuit.seed >> 56) / 255.0
        
        chosen_idx = 1
        for (i, gate) in enumerate(affordances)
            if threshold < gate.probability
                chosen_idx = i
                break
            end
            threshold -= gate.probability
        end
        
        gate = affordances[min(chosen_idx, length(affordances))]
        
        # Stochastic application
        apply = (splitmix64(circuit.seed ⊻ gate.seed) >> 63) & 1 == 0
        
        if apply
            push!(circuit.applied_gates, gate)
            total_prob *= gate.probability
            info_gain += -log2(gate.probability + 1e-10)
            
            circuit.seed = splitmix64(circuit.seed ⊻ gate.seed)
            circuit.color = color_from_seed(circuit.seed)
        end
        
        push!(steps, (gate, apply))
        push!(colors, circuit.color)
    end
    
    AffordanceTrajectory(circuit, steps, total_prob, info_gain, circuit.seed, colors)
end

function parallel_circuit_eval(circuits::Vector{ProbabilityCircuit}, n_steps::Int)
    # Parallel evaluation of multiple circuits
    trajectories = Vector{AffordanceTrajectory}(undef, length(circuits))
    
    Threads.@threads for i in eachindex(circuits)
        trajectories[i] = gaymc_bang!(circuits[i], n_steps)
    end
    
    trajectories
end

# ═══════════════════════════════════════════════════════════════════════════════
# COLOR WHEEL OF WORLD ROTATORS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    DirectionDesignator

Direction on the color wheel: -, +, or _ (neutral).
"""
struct DirectionDesignator
    symbol::Symbol  # :minus, :plus, :neutral
    angle::Float64  # Position on wheel (0-360)
    magnitude::Float64  # 0-1
    
    seed::UInt64
    color::NamedTuple{(:r, :g, :b), NTuple{3, Float64}}
end

function DirectionDesignator(angle::Float64; magnitude::Float64=1.0)
    # Direction from angle
    symbol = if angle < 120
        :plus
    elseif angle < 240
        :neutral
    else
        :minus
    end
    
    # Color from angle (HSL with S=1, L=0.5)
    h = angle
    c = 1.0
    x = c * (1 - abs(mod(h / 60, 2) - 1))
    
    r, g, b = if h < 60
        (c, x, 0.0)
    elseif h < 120
        (x, c, 0.0)
    elseif h < 180
        (0.0, c, x)
    elseif h < 240
        (0.0, x, c)
    elseif h < 300
        (x, 0.0, c)
    else
        (c, 0.0, x)
    end
    
    seed = splitmix64(GAY_SEED ⊻ UInt64(round(angle * 1000)))
    DirectionDesignator(symbol, angle, magnitude, seed, (r=r, g=g, b=b))
end

"""
    WorldRotatorWheel

The color wheel for world rotators.
"""
mutable struct WorldRotatorWheel
    rotators::Vector{Rotator}
    directions::Vector{DirectionDesignator}
    
    current_angle::Float64
    angular_velocity::Float64
    
    seed::UInt64
    color::NamedTuple{(:r, :g, :b), NTuple{3, Float64}}
end

function WorldRotatorWheel(n_rotators::Int=12; seed::UInt64=GAY_SEED)
    rotators = [Rotator(Symbol("rotator_$i"); angle=2π*i/n_rotators) for i in 1:n_rotators]
    directions = [DirectionDesignator(360.0 * i / n_rotators) for i in 1:n_rotators]
    
    WorldRotatorWheel(rotators, directions, 0.0, 0.0, seed, color_from_seed(seed))
end

function rotate_on_wheel!(wheel::WorldRotatorWheel, amount::Float64)
    wheel.current_angle = mod(wheel.current_angle + amount, 360.0)
    wheel.angular_velocity = amount
    
    wheel.seed = splitmix64(wheel.seed ⊻ UInt64(round(wheel.current_angle * 1000)))
    wheel.color = DirectionDesignator(wheel.current_angle).color
    
    wheel
end

function wheel_position(wheel::WorldRotatorWheel)::DirectionDesignator
    DirectionDesignator(wheel.current_angle)
end

function direction_from_color(color::NamedTuple)::DirectionDesignator
    hue = hue_from_color(color)
    DirectionDesignator(hue)
end

# ═══════════════════════════════════════════════════════════════════════════════
# MAXIMUM SHARING ACROSS SUBSTRATES
# ═══════════════════════════════════════════════════════════════════════════════

"""
    SharedSubexpression

A subexpression that appears in multiple contexts.
"""
struct SharedSubexpression
    id::Int
    expression::Any
    occurrences::Vector{Tuple{Symbol, Int}}  # (context_name, position)
    
    # Sharing metrics
    n_shares::Int
    memory_saved::Int  # Bytes
    
    seed::UInt64
    color::NamedTuple{(:r, :g, :b), NTuple{3, Float64}}
end

"""
    SubstrateShare

A substrate that participates in sharing.
"""
mutable struct SubstrateShare
    name::Symbol
    expressions::Vector{Any}
    
    # Shared subexpressions
    shared::Vector{SharedSubexpression}
    
    seed::UInt64
end

"""
    MaximalSharing

Configuration for maximum sharing across substrates.
"""
mutable struct MaximalSharing
    substrates::Vector{SubstrateShare}
    
    # Global shared subexpressions
    global_shared::Dict{UInt64, SharedSubexpression}  # fingerprint → shared
    
    # Metrics
    total_expressions::Int
    unique_expressions::Int
    sharing_ratio::Float64
    
    seed::UInt64
end

function MaximalSharing(; seed::UInt64=GAY_SEED)
    MaximalSharing(SubstrateShare[], Dict{UInt64, SharedSubexpression}(),
                   0, 0, 0.0, seed)
end

function add_substrate!(sharing::MaximalSharing, name::Symbol, expressions::Vector)
    substrate = SubstrateShare(name, expressions, SharedSubexpression[],
                               splitmix64(sharing.seed ⊻ hash(name)))
    push!(sharing.substrates, substrate)
    
    # Find sharing opportunities
    for (i, expr) in enumerate(expressions)
        fp = fingerprint_expression(expr)
        
        if haskey(sharing.global_shared, fp)
            # Already shared
            shared = sharing.global_shared[fp]
            push!(shared.occurrences, (name, i))
        else
            # New shared subexpression
            shared = SharedSubexpression(length(sharing.global_shared) + 1, expr,
                                        [(name, i)], 1, 0,
                                        splitmix64(fp), color_from_seed(fp))
            sharing.global_shared[fp] = shared
        end
    end
    
    update_sharing_metrics!(sharing)
end

function fingerprint_expression(expr)::UInt64
    splitmix64(UInt64(hash(expr)) ⊻ GAY_SEED)
end

function update_sharing_metrics!(sharing::MaximalSharing)
    sharing.total_expressions = sum(length(s.expressions) for s in sharing.substrates)
    sharing.unique_expressions = length(sharing.global_shared)
    sharing.sharing_ratio = 1.0 - sharing.unique_expressions / max(1, sharing.total_expressions)
end

function find_sharing!(sharing::MaximalSharing)::Vector{SharedSubexpression}
    # Return all subexpressions with > 1 occurrence
    [s for s in values(sharing.global_shared) if length(s.occurrences) > 1]
end

function deduplicate_subexpressions(sharing::MaximalSharing)::Dict{UInt64, Int}
    # Return mapping from fingerprint to canonical ID
    Dict(fp => shared.id for (fp, shared) in sharing.global_shared)
end

function sharing_fingerprint(sharing::MaximalSharing)::UInt64
    reduce(⊻, keys(sharing.global_shared); init=sharing.seed)
end

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

mean(x) = isempty(x) ? 0.0 : sum(x) / length(x)
argmax(f, xs) = isempty(xs) ? 1 : findmax(f.(xs))[2]

# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

function demo_word_world_rotators()
    println("╔═══════════════════════════════════════════════════════════════════════════╗")
    println("║  WORD ↔ WORLD ROTATORS: CNOT Agency over Information Network             ║")
    println("╚═══════════════════════════════════════════════════════════════════════════╝")
    println()
    
    # 1. Create word and world models
    println("─── WORD ↔ WORLD MODELS ───")
    word = WordModel(:gpt4)
    world = WorldModel(:physics_sim)
    rotator = Rotator(:w2w)
    
    r, g, b = round.(Int, [word.color.r, word.color.g, word.color.b] .* 255)
    println("  Word model: \e[38;2;$(r);$(g);$(b)m●\e[0m $(word.name)")
    r, g, b = round.(Int, [world.color.r, world.color.g, world.color.b] .* 255)
    println("  World model: \e[38;2;$(r);$(g);$(b)m●\e[0m $(world.name)")
    println()
    
    # 2. Create TworldWolder
    tw = TworldWolder(word, world, rotator)
    println("  TworldWolder entanglement: $(round(tw.entanglement, digits=4))")
    println()
    
    # 3. Bidirectional transformation
    println("─── BIDIRECTIONAL TRANSFORMATION ───")
    colors = bidirectional_transform(tw, word.color; n_cycles=2)
    print("  Color path: ")
    for c in colors
        r, g, b = round.(Int, [c.r, c.g, c.b] .* 255)
        print("\e[38;2;$(r);$(g);$(b)m●\e[0m")
    end
    println()
    fp = transformation_fingerprint(colors)
    println("  Transformation fingerprint: 0x$(string(fp, base=16)[1:12])...")
    println()
    
    # 4. CNOT choice agency
    println("─── CNOT CHOICE AGENCY ───")
    registry = CNOTRegistry(5)
    
    println("  Initial agency: $(round(agency_over_cnot(registry), digits=4))")
    
    choices = entangle_max!(registry, 3)
    for choice in choices
        println("    CNOT($(choice.control_idx)→$(choice.target_idx)): ent=$(round(choice.predicted_entanglement, digits=3)), info=$(round(choice.info_bits, digits=2)) bits")
    end
    
    println("  Final agency: $(round(agency_over_cnot(registry), digits=4))")
    println()
    
    # 5. Build impact network
    println("─── IMPACT NETWORK ───")
    
    # Create the origin paper
    wong = Observer(1, "Lionel Wong"; engagement=:author)
    grand = Observer(2, "Gabriel Grand"; engagement=:author)
    origin = PaperNode("2306.12672", "From Word Models to World Models", [wong, grand], 2023)
    
    network = build_impact_network(origin)
    
    # Add citing papers
    citer1 = Observer(3, "Researcher A"; engagement=:citer)
    paper1 = PaperNode("2401.00001", "Extending Word-World Translation", [citer1], 2024)
    add_paper!(network, paper1)
    add_citation!(network, "2401.00001", "2306.12672"; ctype=:extends)
    
    citer2 = Observer(4, "Researcher B"; engagement=:reinterpreter)
    paper2 = PaperNode("2402.00002", "World Models for Robotics", [citer2], 2024)
    add_paper!(network, paper2)
    add_citation!(network, "2402.00002", "2306.12672"; ctype=:builds_on)
    add_citation!(network, "2402.00002", "2401.00001"; ctype=:uses)
    
    println("  Papers: $(length(network.papers))")
    println("  Citations: $(length(network.citations))")
    println("  Network entanglement: $(round(network_entanglement(network), digits=4))")
    
    # Trace reinterpretations
    paths = trace_reinterpretations(network, "2306.12672"; max_depth=5)
    println("  Reinterpretation paths: $(length(paths))")
    cycles = count(p -> p.is_cycle, paths)
    println("  Cycles (word→world→word): $cycles")
    println()
    
    # 6. PLoT program
    println("─── PROBABILISTIC LANGUAGE OF THOUGHT ───")
    text = "The ball probably falls when released"
    program = nl_to_plot(text)
    
    println("  Input: \"$(text)\"")
    println("  PLoT statements: $(length(program.statements))")
    for stmt in program.statements
        println("    $(stmt.type): $(stmt.expression)")
    end
    
    plot_color = plot_to_color(program)
    r, g, b = round.(Int, [plot_color.r, plot_color.g, plot_color.b] .* 255)
    println("  PLoT color: \e[38;2;$(r);$(g);$(b)m████\e[0m")
    
    circuit_gates = color_to_circuit(plot_color)
    println("  Circuit gates: $(length(circuit_gates))")
    println()
    
    # 7. GayMC probability circuits
    println("─── GAYMC PROBABILITY CIRCUITS ───")
    circuits = [ProbabilityCircuit(3; seed=splitmix64(GAY_SEED ⊻ UInt64(i))) for i in 1:4]
    
    trajectories = parallel_circuit_eval(circuits, 10)
    
    for (i, traj) in enumerate(trajectories)
        applied = count(s -> s[2], traj.steps)
        r, g, b = round.(Int, [traj.colors[end].r, traj.colors[end].g, traj.colors[end].b] .* 255)
        println("  Circuit $i: $(applied)/$(length(traj.steps)) gates, info=$(round(traj.information_gain, digits=2)) bits \e[38;2;$(r);$(g);$(b)m●\e[0m")
    end
    println()
    
    # 8. Color wheel of world rotators
    println("─── COLOR WHEEL OF WORLD ROTATORS ───")
    wheel = WorldRotatorWheel(12)
    
    for _ in 1:6
        rotate_on_wheel!(wheel, 30.0)
        dir = wheel_position(wheel)
        r, g, b = round.(Int, [wheel.color.r, wheel.color.g, wheel.color.b] .* 255)
        println("  $(lpad(round(Int, wheel.current_angle), 3))°: \e[38;2;$(r);$(g);$(b)m●\e[0m $(dir.symbol)")
    end
    println()
    
    # 9. Maximum sharing
    println("─── MAXIMUM SHARING ACROSS SUBSTRATES ───")
    sharing = MaximalSharing()
    
    # Add expressions from different substrates
    add_substrate!(sharing, :word_substrate, [:embed, :attend, :project, :embed, :norm])
    add_substrate!(sharing, :world_substrate, [:simulate, :project, :embed, :update])
    add_substrate!(sharing, :combined_substrate, [:embed, :simulate, :attend])
    
    shared = find_sharing!(sharing)
    println("  Total expressions: $(sharing.total_expressions)")
    println("  Unique expressions: $(sharing.unique_expressions)")
    println("  Sharing ratio: $(round(sharing.sharing_ratio * 100, digits=1))%")
    println("  Shared subexpressions: $(length(shared))")
    for s in shared[1:min(3, end)]
        println("    $(s.expression): $(length(s.occurrences)) occurrences")
    end
    println()
    
    println("═══════════════════════════════════════════════════════════════════════════")
    println("  From Word Models to World Models — and back again")
    println("  CNOT agency: choose which entanglements maximize information network")
    println("  GayMC: bang out probability circuits with maximum parallelism")
    println("═══════════════════════════════════════════════════════════════════════════")
    
    (network=network, trajectories=trajectories, sharing=sharing, wheel=wheel)
end

end # module
