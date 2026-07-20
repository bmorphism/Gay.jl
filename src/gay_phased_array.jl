# GAY PHASED ARRAY RADAR: Self-Avoiding Chromatic Traversal
# ============================================================
#
# "The beam steers itself by avoiding where it has already been."
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  PHASED ARRAY RADAR × SELF-AVOIDING WALK × CHROMATIC GADGETS               │
# │                                                                             │
# │  PHASED ARRAY STRUCTURE:                                                    │
# │                                                                             │
# │     ╭─────╮ ╭─────╮ ╭─────╮ ╭─────╮ ╭─────╮                                │
# │     │ A₀  │ │ A₁  │ │ A₂  │ │ A₃  │ │ A₄  │  ← Antenna elements           │
# │     ╰──┬──╯ ╰──┬──╯ ╰──┬──╯ ╰──┬──╯ ╰──┬──╯                                │
# │        │φ₀    │φ₁    │φ₂    │φ₃    │φ₄     ← Phase shifts (colors!)      │
# │        ↓      ↓      ↓      ↓      ↓                                       │
# │     ═══════════════════════════════════════                                │
# │              BEAM PATTERN (interference)                                   │
# │                      ↓                                                     │
# │                 main lobe → target direction                               │
# │                                                                             │
# │  SELF-AVOIDING RANDOM WALK:                                                 │
# │                                                                             │
# │     Start ──→ v₁ ──→ v₂ ──→ v₃ ──→ ...                                    │
# │                ↑      ↑      ↑                                             │
# │              (never revisit visited vertices)                              │
# │                                                                             │
# │  UNIFICATION:                                                               │
# │                                                                             │
# │     Phase φᵢ = color of antenna i                                          │
# │     Beam direction = next_color choice                                      │
# │     Self-avoidance = don't reuse phases (colors)                           │
# │     Constructive interference = valid SAW path                              │
# │     Destructive interference = blocked (already visited)                    │
# │                                                                             │
# │  EDGE VARIABLE GADGET:                                                      │
# │                                                                             │
# │     next_color(state) → pure function, returns candidate color             │
# │     next_color!(state) → mutating, commits to the choice                   │
# │                                                                             │
# │     The gadget encodes traversal semantics:                                 │
# │       • Available edges = unvisited colors                                  │
# │       • Phase shift = color difference                                      │
# │       • Beam steering = choice of next edge                                 │
# │                                                                             │
# └─────────────────────────────────────────────────────────────────────────────┘

module GayPhasedArray

export
    # Core types
    Antenna, PhasedArray, BeamPattern, 
    GayRadar, RadarState,
    
    # Self-avoiding walk
    SAWState, SAWPath, SAWNetwork,
    saw_step!, saw_backtrack!, is_valid_saw,
    
    # Verifier game
    Verifier, add_verifier!, check_verifiers, verifier_status,
    
    # Edge variable gadgets
    EdgeGadget, ColorChoice, TraversalState,
    next_color, next_color!, available_colors,
    edge_phase, beam_direction,
    
    # Truly random derivation
    split_rng!, derive_random_color, outcolor_verifiers!,
    
    # Phased array operations
    steer_beam!, compute_pattern, interference,
    constructive_edges, destructive_edges,
    
    # Network traversal
    NetworkNode, InformationNetwork,
    traverse!, explore!, radar_sweep,
    
    # Visualization helpers
    pattern_to_colors, beam_to_rgb,
    
    # Demo
    world_gay_phased_array

# ═══════════════════════════════════════════════════════════════════════════════
# Core PRNG
# ═══════════════════════════════════════════════════════════════════════════════

const GAY_SEED = UInt64(1069)
const RADAR_SEED = UInt64(0x2ADA)  # "RADAR"

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

@inline function phase_from_seed(seed::UInt64)::Float64
    r, _ = sm64(seed)
    2π * (r / typemax(UInt64))
end

# ═══════════════════════════════════════════════════════════════════════════════
# ANTENNA AND PHASED ARRAY
# ═══════════════════════════════════════════════════════════════════════════════

"""
    Antenna

Single antenna element with position, phase, and chromatic identity.
"""
struct Antenna
    id::Int
    position::NTuple{3, Float64}  # 3D position (x, y, z)
    
    # Phase = color encoded as angle
    phase::Float64  # 0 to 2π
    
    # Chromatic identity
    color::NTuple{3, Float64}
    seed::UInt64
end

function Antenna(id::Int; 
                 position::NTuple{3, Float64}=(0.0, 0.0, 0.0),
                 seed::UInt64=RADAR_SEED)
    ant_seed = seed ⊻ UInt64(id * 0x1069)
    phase = phase_from_seed(ant_seed)
    color = color_from_seed(ant_seed)
    Antenna(id, position, phase, color, ant_seed)
end

"""
    PhasedArray

Array of antennas with collective beam steering.
"""
mutable struct PhasedArray
    antennas::Vector{Antenna}
    
    # Array geometry
    spacing::Float64  # Element spacing (wavelengths)
    
    # Current beam direction (azimuth, elevation)
    beam_direction::NTuple{2, Float64}
    
    # Visited phases (for self-avoidance)
    visited_phases::Set{Float64}
    
    seed::UInt64
    color::NTuple{3, Float64}
end

function PhasedArray(n_elements::Int; 
                     spacing::Float64=0.5,
                     seed::UInt64=RADAR_SEED)
    antennas = Antenna[]
    for i in 1:n_elements
        # Linear array along x-axis
        pos = (Float64(i-1) * spacing, 0.0, 0.0)
        push!(antennas, Antenna(i; position=pos, seed=seed ⊻ UInt64(i)))
    end
    
    PhasedArray(
        antennas, spacing, (0.0, 0.0), Set{Float64}(),
        seed, color_from_seed(seed)
    )
end

"""
    BeamPattern

Radiation pattern of the phased array.
"""
struct BeamPattern
    # Pattern as function of angle
    azimuth_pattern::Vector{Float64}    # Power vs azimuth
    elevation_pattern::Vector{Float64}  # Power vs elevation
    
    # Main lobe direction
    main_lobe::NTuple{2, Float64}
    
    # Sidelobe levels
    sidelobe_level::Float64
    
    # Pattern encoded as colors
    colors::Vector{NTuple{3, Float64}}
    
    seed::UInt64
end

# ═══════════════════════════════════════════════════════════════════════════════
# SELF-AVOIDING WALK
# ═══════════════════════════════════════════════════════════════════════════════

"""
    SAWState

State of a self-avoiding random walk.
"""
mutable struct SAWState
    # Current position (vertex)
    current::Int
    
    # Visited vertices
    visited::Set{Int}
    
    # Path taken
    path::Vector{Int}
    
    # Available moves (unvisited neighbors)
    available::Vector{Int}
    
    # Stuck? (no valid moves)
    stuck::Bool
    
    # Step count
    steps::Int
    
    seed::UInt64
    color::NTuple{3, Float64}
end

function SAWState(start::Int; seed::UInt64=GAY_SEED)
    SAWState(
        start,
        Set([start]),
        [start],
        Int[],
        false,
        0,
        seed,
        color_from_seed(seed)
    )
end

"""
    SAWPath

A completed self-avoiding walk path.
"""
struct SAWPath
    vertices::Vector{Int}
    length::Int
    
    # Each step's color
    colors::Vector{NTuple{3, Float64}}
    
    # Did it terminate naturally or get stuck?
    completed::Bool
    
    seed::UInt64
    fingerprint::UInt64
end

"""
    SAWNetwork

Network structure for self-avoiding walks.
"""
struct SAWNetwork
    n_vertices::Int
    
    # Adjacency list
    neighbors::Vector{Vector{Int}}
    
    # Edge colors (indexed by sorted vertex pair)
    edge_colors::Dict{Tuple{Int, Int}, NTuple{3, Float64}}
    
    # Edge phases
    edge_phases::Dict{Tuple{Int, Int}, Float64}
    
    seed::UInt64
    color::NTuple{3, Float64}
end

function SAWNetwork(n::Int; connectivity::Float64=0.3, seed::UInt64=GAY_SEED)
    neighbors = [Int[] for _ in 1:n]
    edge_colors = Dict{Tuple{Int, Int}, NTuple{3, Float64}}()
    edge_phases = Dict{Tuple{Int, Int}, Float64}()
    
    s = seed
    for i in 1:n
        for j in i+1:n
            r, s = sm64(s)
            if (r / typemax(UInt64)) < connectivity
                push!(neighbors[i], j)
                push!(neighbors[j], i)
                
                edge_key = (i, j)
                edge_colors[edge_key] = color_from_seed(s ⊻ UInt64(i * n + j))
                edge_phases[edge_key] = phase_from_seed(s ⊻ UInt64(i * n + j))
            end
        end
    end
    
    SAWNetwork(n, neighbors, edge_colors, edge_phases, seed, color_from_seed(seed))
end

"""
    saw_step!(state, network, choice) → Bool

Take one step in the SAW. Returns true if successful.
choice: index into available moves, or nothing for random.
"""
function saw_step!(state::SAWState, network::SAWNetwork, choice::Union{Int, Nothing}=nothing)
    if state.stuck
        return false
    end
    
    # Get available moves (unvisited neighbors)
    current_neighbors = network.neighbors[state.current]
    state.available = filter(v -> v ∉ state.visited, current_neighbors)
    
    if isempty(state.available)
        state.stuck = true
        return false
    end
    
    # Choose next vertex
    next_vertex = if choice === nothing
        # Random choice
        r, _ = sm64(state.seed ⊻ UInt64(state.steps))
        state.available[1 + (r % length(state.available))]
    else
        state.available[clamp(choice, 1, length(state.available))]
    end
    
    # Update state
    state.current = next_vertex
    push!(state.visited, next_vertex)
    push!(state.path, next_vertex)
    state.steps += 1
    state.color = color_from_seed(state.seed ⊻ UInt64(state.steps))
    
    true
end

"""
    saw_backtrack!(state) → Bool

Backtrack one step. Returns true if successful.
"""
function saw_backtrack!(state::SAWState)
    if length(state.path) <= 1
        return false
    end
    
    # Remove current from visited
    delete!(state.visited, state.current)
    pop!(state.path)
    
    # Go back to previous
    state.current = state.path[end]
    state.steps -= 1
    state.stuck = false
    
    true
end

function is_valid_saw(path::Vector{Int})
    length(path) == length(unique(path))
end

# ═══════════════════════════════════════════════════════════════════════════════
# EDGE VARIABLE GADGETS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ColorChoice

A candidate color for the next step, with phase and interference info.
"""
struct ColorChoice
    color::NTuple{3, Float64}
    phase::Float64
    
    # Target vertex
    target::Int
    
    # Interference with visited colors
    interference::Float64  # 0 = constructive, π = destructive
    
    # Is this choice valid (constructive)?
    valid::Bool
    
    seed::UInt64
end

"""
    Verifier

A constraint that must be satisfied by a color choice.
"Outcoloring" means finding a color that defeats all verifiers.
"""
struct Verifier
    id::Int
    constraint::Function  # (color, phase, state) → Bool (true = violated)
    
    # Verifier strength (how hard to satisfy)
    strength::Float64
    
    # Is this verifier still active?
    active::Bool
    
    seed::UInt64
end

function Verifier(id::Int, constraint::Function; 
                  strength::Float64=1.0, 
                  seed::UInt64=GAY_SEED)
    Verifier(id, constraint, strength, true, seed ⊻ UInt64(id))
end

"""
    EdgeGadget

The next_color / next_color! gadget for edge variable choice.

Encapsulates the semantics of traversal:
- Available edges as unvisited colors
- Phase relationships for beam steering
- Self-avoidance via color tracking
- Verifier adversary game for constraint satisfaction

DYNAMIC SUFFICIENCY: Adapts randomness to remaining verifier difficulty.
OUTCOLORING: Find color that defeats all verifiers as quickly as possible.
"""
mutable struct EdgeGadget
    # Current state
    current_color::NTuple{3, Float64}
    current_phase::Float64
    
    # Visited colors (phases)
    visited_colors::Vector{NTuple{3, Float64}}
    visited_phases::Vector{Float64}
    
    # Available choices
    candidates::Vector{ColorChoice}
    
    # Phase tolerance for "sameness"
    phase_tolerance::Float64
    
    # ═══ Verifier Game State ═══
    
    # Active verifiers (constraints to satisfy)
    verifiers::Vector{Verifier}
    
    # Defeated verifiers (satisfied constraints)
    defeated::Vector{Verifier}
    
    # Dynamic sufficiency parameters
    entropy_budget::Float64      # How much randomness to use
    exploration_rate::Float64    # Balance explore vs exploit
    
    # Statistics
    attempts::Int                # Total color attempts
    successes::Int               # Successful outcolorings
    verifier_defeats::Int        # Total verifiers defeated
    
    # Splittable RNG state (truly random derivation)
    rng_state::UInt64
    rng_splits::Int
    
    seed::UInt64
end

function EdgeGadget(start_color::NTuple{3, Float64}; 
                    phase_tolerance::Float64=0.1,
                    seed::UInt64=GAY_SEED)
    phase = phase_from_seed(seed)
    
    # Default verifiers: phase uniqueness, color distance, interference
    default_verifiers = [
        Verifier(1, (c, p, s) -> any(abs(p - vp) < s.phase_tolerance for vp in s.visited_phases);
                 strength=1.0, seed=seed),
        Verifier(2, (c, p, s) -> any(color_distance(c, vc) < 0.1 for vc in s.visited_colors);
                 strength=0.8, seed=seed),
        Verifier(3, (c, p, s) -> length(s.visited_phases) > 0 && 
                                  abs(p - s.visited_phases[end]) < 0.05;
                 strength=0.5, seed=seed),
    ]
    
    EdgeGadget(
        start_color, phase,
        [start_color], [phase],
        ColorChoice[],
        phase_tolerance,
        default_verifiers, Verifier[],
        1.0, 0.3,  # entropy_budget, exploration_rate
        0, 0, 0,   # attempts, successes, verifier_defeats
        seed, 0,   # rng_state, rng_splits
        seed
    )
end

# Color distance helper
function color_distance(c1::NTuple{3, Float64}, c2::NTuple{3, Float64})
    sqrt((c1[1] - c2[1])^2 + (c1[2] - c2[2])^2 + (c1[3] - c2[3])^2)
end

"""
    next_color(gadget, network, current_vertex) → ColorChoice

Pure function: compute the next color choice without committing.
Returns the best constructive interference candidate.
"""
function next_color(gadget::EdgeGadget, network::SAWNetwork, current::Int)
    # Get neighboring edges
    neighbors = network.neighbors[current]
    
    candidates = ColorChoice[]
    
    for neighbor in neighbors
        edge_key = current < neighbor ? (current, neighbor) : (neighbor, current)
        
        edge_color = get(network.edge_colors, edge_key, color_from_seed(gadget.seed))
        edge_phase = get(network.edge_phases, edge_key, 0.0)
        
        # Check interference with visited phases
        min_interference = π  # Start with maximum (destructive)
        for visited_phase in gadget.visited_phases
            phase_diff = abs(mod(edge_phase - visited_phase + π, 2π) - π)
            min_interference = min(min_interference, phase_diff)
        end
        
        # Constructive if interference is small (phases far from visited)
        valid = min_interference > gadget.phase_tolerance
        
        push!(candidates, ColorChoice(
            edge_color, edge_phase, neighbor, min_interference, valid,
            gadget.seed ⊻ UInt64(neighbor)
        ))
    end
    
    gadget.candidates = candidates
    
    # Return best valid candidate (most constructive)
    valid_candidates = filter(c -> c.valid, candidates)
    
    if isempty(valid_candidates)
        # All destructive - return least bad
        if isempty(candidates)
            return ColorChoice(gadget.current_color, gadget.current_phase, 
                             current, π, false, gadget.seed)
        end
        return candidates[argmax(c.interference for c in candidates)]
    end
    
    # Best = maximum phase distance from visited (most different)
    valid_candidates[argmax(c.interference for c in valid_candidates)]
end

"""
    split_rng!(gadget) → UInt64

Splittable RNG: derive truly random value with dynamic sufficiency.
Each split is independent and reproducible.
"""
function split_rng!(gadget::EdgeGadget)
    # Mix in entropy budget and split count for dynamic sufficiency
    entropy_mix = reinterpret(UInt64, gadget.entropy_budget) ⊻ UInt64(gadget.rng_splits * 0x1069)
    
    # Splitmix64 with entropy mixing
    z = gadget.rng_state + 0x9E3779B97F4A7C15 + entropy_mix
    z = (z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ⊻ (z >> 27)) * 0x94D049BB133111EB
    result = z ⊻ (z >> 31)
    
    # Update state for next split
    gadget.rng_state = result
    gadget.rng_splits += 1
    
    result
end

"""
    check_verifiers(gadget, color, phase) → (passed::Bool, violated::Vector{Verifier})

Check all active verifiers against a candidate color.
"""
function check_verifiers(gadget::EdgeGadget, color::NTuple{3, Float64}, phase::Float64)
    violated = Verifier[]
    
    for v in gadget.verifiers
        if v.active && v.constraint(color, phase, gadget)
            push!(violated, v)
        end
    end
    
    (isempty(violated), violated)
end

"""
    derive_random_color(gadget) → (color, phase)

Derive a truly random color using splittable RNG with dynamic sufficiency.
Entropy budget scales with remaining verifier difficulty.
"""
function derive_random_color(gadget::EdgeGadget)
    # Compute dynamic entropy based on verifier difficulty
    total_strength = sum(v.strength for v in gadget.verifiers if v.active; init=0.0)
    gadget.entropy_budget = 1.0 + total_strength
    
    # Split RNG for each color component
    r_bits = split_rng!(gadget)
    g_bits = split_rng!(gadget)
    b_bits = split_rng!(gadget)
    phase_bits = split_rng!(gadget)
    
    color = (
        (r_bits >> 56) / 255.0,
        (g_bits >> 56) / 255.0,
        (b_bits >> 56) / 255.0
    )
    
    phase = 2π * (phase_bits / typemax(UInt64))
    
    (color, phase)
end

"""
    outcolor_verifiers!(gadget, max_attempts) → (success, color, phase, attempts)

Outcolor the last verifier standing as quickly as possible.
Tries random derivations until all verifiers are defeated.
"""
function outcolor_verifiers!(gadget::EdgeGadget, max_attempts::Int=100)
    best_color = gadget.current_color
    best_phase = gadget.current_phase
    best_violations = length(gadget.verifiers)
    
    for attempt in 1:max_attempts
        gadget.attempts += 1
        
        # Derive truly random candidate
        color, phase = derive_random_color(gadget)
        
        # Check against all verifiers
        passed, violated = check_verifiers(gadget, color, phase)
        
        if passed
            # All verifiers defeated!
            gadget.successes += 1
            gadget.verifier_defeats += length(gadget.verifiers)
            return (true, color, phase, attempt)
        end
        
        # Track best so far (fewest violations)
        if length(violated) < best_violations
            best_color = color
            best_phase = phase
            best_violations = length(violated)
            
            # Adaptive: increase exploration when making progress
            gadget.exploration_rate = min(0.9, gadget.exploration_rate + 0.1)
        else
            # Decrease exploration when stuck
            gadget.exploration_rate = max(0.1, gadget.exploration_rate - 0.01)
        end
        
        # Early termination: if we defeated some verifiers, accept partial progress
        if length(violated) <= 1 && attempt > max_attempts ÷ 2
            defeated_count = length(gadget.verifiers) - length(violated)
            gadget.verifier_defeats += defeated_count
            return (true, best_color, best_phase, attempt)
        end
    end
    
    # Failed to fully outcolor, return best attempt
    (false, best_color, best_phase, max_attempts)
end

"""
    next_color!(gadget, choice) → ColorChoice

Mutating: commit to a color choice with comprehensive truly random derivation.

DYNAMIC SUFFICIENCY: Entropy scales with constraint difficulty.
OUTCOLORING: Defeats verifiers as quickly as possible.
TRULY RANDOM: Splittable RNG for reproducible independence.
"""
function next_color!(gadget::EdgeGadget, choice::ColorChoice)
    gadget.attempts += 1
    
    # Check if the provided choice passes verifiers
    passed, violated = check_verifiers(gadget, choice.color, choice.phase)
    
    final_color = choice.color
    final_phase = choice.phase
    
    if !passed
        # Choice violates constraints - try to outcolor
        success, outcolor, outphase, attempts = outcolor_verifiers!(gadget)
        
        if success
            final_color = outcolor
            final_phase = outphase
        else
            # Fallback: use the choice but mark verifiers as defeated anyway
            # (dynamic sufficiency: accept partial progress)
            for v in violated
                if v.strength < gadget.exploration_rate
                    # Weak verifier defeated by exploration
                    push!(gadget.defeated, v)
                    filter!(x -> x.id != v.id, gadget.verifiers)
                    gadget.verifier_defeats += 1
                end
            end
        end
    else
        gadget.successes += 1
    end
    
    # Commit the color
    push!(gadget.visited_colors, final_color)
    push!(gadget.visited_phases, final_phase)
    gadget.current_color = final_color
    gadget.current_phase = final_phase
    
    # Update RNG state for next call
    gadget.rng_state = gadget.seed ⊻ UInt64(length(gadget.visited_phases)) ⊻ 
                       reinterpret(UInt64, final_phase)
    
    ColorChoice(final_color, final_phase, choice.target, 
                choice.interference, passed, gadget.seed)
end

"""
    next_color!(gadget) → ColorChoice

Mutating: derive and commit a truly random color without prior choice.
Uses full outcoloring procedure to defeat all verifiers.
"""
function next_color!(gadget::EdgeGadget)
    gadget.attempts += 1
    
    # Full outcoloring attempt
    success, color, phase, attempts = outcolor_verifiers!(gadget)
    
    # Commit regardless of success (dynamic sufficiency)
    push!(gadget.visited_colors, color)
    push!(gadget.visited_phases, phase)
    gadget.current_color = color
    gadget.current_phase = phase
    
    if success
        gadget.successes += 1
    end
    
    # Update RNG state
    gadget.rng_state = gadget.seed ⊻ UInt64(length(gadget.visited_phases)) ⊻ 
                       reinterpret(UInt64, phase)
    
    ColorChoice(color, phase, 0, 0.0, success, gadget.seed)
end

"""
    add_verifier!(gadget, constraint; strength, id) → Verifier

Add a new verifier to the game.
"""
function add_verifier!(gadget::EdgeGadget, constraint::Function; 
                       strength::Float64=1.0, 
                       id::Int=length(gadget.verifiers) + length(gadget.defeated) + 1)
    v = Verifier(id, constraint; strength=strength, seed=gadget.seed)
    push!(gadget.verifiers, v)
    v
end

"""
    verifier_status(gadget) → NamedTuple

Get current verifier game status.
"""
function verifier_status(gadget::EdgeGadget)
    active = length(gadget.verifiers)
    defeated = length(gadget.defeated)
    total_strength = sum(v.strength for v in gadget.verifiers; init=0.0)
    success_rate = gadget.attempts > 0 ? gadget.successes / gadget.attempts : 0.0
    
    (
        active_verifiers = active,
        defeated_verifiers = defeated,
        total_strength = total_strength,
        attempts = gadget.attempts,
        successes = gadget.successes,
        success_rate = success_rate,
        entropy_budget = gadget.entropy_budget,
        exploration_rate = gadget.exploration_rate,
        rng_splits = gadget.rng_splits
    )
end

"""
    available_colors(gadget, network, current) → Vector{ColorChoice}

Get all available (valid) color choices from current position.
"""
function available_colors(gadget::EdgeGadget, network::SAWNetwork, current::Int)
    _ = next_color(gadget, network, current)  # Populate candidates
    filter(c -> c.valid, gadget.candidates)
end

"""
    edge_phase(gadget, v1, v2, network) → Float64

Get the phase of an edge (color encoded as angle).
"""
function edge_phase(gadget::EdgeGadget, v1::Int, v2::Int, network::SAWNetwork)
    edge_key = v1 < v2 ? (v1, v2) : (v2, v1)
    get(network.edge_phases, edge_key, phase_from_seed(gadget.seed ⊻ UInt64(v1 * v2)))
end

"""
    beam_direction(gadget) → NTuple{2, Float64}

Compute current beam direction from phase history.
The beam "points" in the direction of accumulated phase.
"""
function beam_direction(gadget::EdgeGadget)
    if isempty(gadget.visited_phases)
        return (0.0, 0.0)
    end
    
    # Beam direction = vector sum of all phases
    azimuth = sum(cos.(gadget.visited_phases)) / length(gadget.visited_phases)
    elevation = sum(sin.(gadget.visited_phases)) / length(gadget.visited_phases)
    
    (atan(elevation, azimuth), sqrt(azimuth^2 + elevation^2))
end

# ═══════════════════════════════════════════════════════════════════════════════
# PHASED ARRAY OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    steer_beam!(array, azimuth, elevation)

Steer the beam to the specified direction by adjusting phases.
"""
function steer_beam!(array::PhasedArray, azimuth::Float64, elevation::Float64)
    # Phase shift for each antenna to steer beam
    k = 2π  # Wavenumber (normalized)
    
    for (i, ant) in enumerate(array.antennas)
        # Phase shift = k * d * sin(θ) for linear array
        phase_shift = k * ant.position[1] * sin(azimuth)
        
        # Create new antenna with updated phase
        new_phase = mod(ant.phase + phase_shift, 2π)
        
        # Check self-avoidance
        if new_phase ∈ array.visited_phases
            # Find alternative phase
            for offset in 0.1:0.1:2π
                test_phase = mod(new_phase + offset, 2π)
                if test_phase ∉ array.visited_phases
                    new_phase = test_phase
                    break
                end
            end
        end
        
        push!(array.visited_phases, new_phase)
        
        array.antennas[i] = Antenna(
            ant.id, ant.position, new_phase,
            color_from_seed(ant.seed ⊻ UInt64(round(new_phase * 1000))),
            ant.seed
        )
    end
    
    array.beam_direction = (azimuth, elevation)
end

"""
    compute_pattern(array) → BeamPattern

Compute the radiation pattern of the phased array.
"""
function compute_pattern(array::PhasedArray)
    n_angles = 360
    azimuth_pattern = zeros(n_angles)
    colors = NTuple{3, Float64}[]
    
    for (i, θ) in enumerate(range(-π, π, length=n_angles))
        # Array factor
        af = 0.0 + 0.0im
        for ant in array.antennas
            # Phase contribution from this antenna
            phase_contrib = ant.phase + 2π * ant.position[1] * sin(θ)
            af += exp(im * phase_contrib)
        end
        
        azimuth_pattern[i] = abs2(af) / length(array.antennas)^2
        
        # Color from pattern value
        push!(colors, color_from_seed(array.seed ⊻ UInt64(round(azimuth_pattern[i] * 1000))))
    end
    
    # Find main lobe
    main_idx = argmax(azimuth_pattern)
    main_angle = range(-π, π, length=n_angles)[main_idx]
    
    # Sidelobe level (second highest peak)
    sorted_pattern = sort(azimuth_pattern, rev=true)
    sll = sorted_pattern[2] / sorted_pattern[1]
    
    BeamPattern(
        azimuth_pattern, zeros(n_angles),
        (main_angle, 0.0), sll,
        colors, array.seed
    )
end

"""
    interference(phase1, phase2) → Float64

Compute interference between two phases.
Returns 0 for perfect constructive, π for perfect destructive.
"""
function interference(phase1::Float64, phase2::Float64)
    abs(mod(phase1 - phase2 + π, 2π) - π)
end

"""
    constructive_edges(gadget, network, current) → Vector{Int}

Get edges with constructive interference (valid moves).
"""
function constructive_edges(gadget::EdgeGadget, network::SAWNetwork, current::Int)
    choices = available_colors(gadget, network, current)
    [c.target for c in choices]
end

"""
    destructive_edges(gadget, network, current) → Vector{Int}

Get edges with destructive interference (blocked moves).
"""
function destructive_edges(gadget::EdgeGadget, network::SAWNetwork, current::Int)
    _ = next_color(gadget, network, current)
    blocked = filter(c -> !c.valid, gadget.candidates)
    [c.target for c in blocked]
end

# ═══════════════════════════════════════════════════════════════════════════════
# INFORMATION NETWORK TRAVERSAL
# ═══════════════════════════════════════════════════════════════════════════════

"""
    NetworkNode

A node in the information network.
"""
struct NetworkNode
    id::Int
    data::Any
    
    # Chromatic identity
    color::NTuple{3, Float64}
    phase::Float64
    
    seed::UInt64
end

function NetworkNode(id::Int, data::Any=nothing; seed::UInt64=GAY_SEED)
    node_seed = seed ⊻ UInt64(id)
    NetworkNode(id, data, color_from_seed(node_seed), phase_from_seed(node_seed), node_seed)
end

"""
    InformationNetwork

Network for information traversal with phased array radar semantics.
"""
mutable struct InformationNetwork
    nodes::Vector{NetworkNode}
    structure::SAWNetwork
    
    # Current radar state
    radar::PhasedArray
    
    # Traversal gadget
    gadget::EdgeGadget
    
    # Walk state
    walk::SAWState
    
    seed::UInt64
    color::NTuple{3, Float64}
end

function InformationNetwork(n_nodes::Int; 
                            connectivity::Float64=0.3,
                            n_antennas::Int=8,
                            seed::UInt64=GAY_SEED)
    nodes = [NetworkNode(i; seed=seed) for i in 1:n_nodes]
    structure = SAWNetwork(n_nodes; connectivity=connectivity, seed=seed)
    radar = PhasedArray(n_antennas; seed=seed)
    
    start_node = nodes[1]
    gadget = EdgeGadget(start_node.color; seed=seed)
    walk = SAWState(1; seed=seed)
    
    InformationNetwork(nodes, structure, radar, gadget, walk, seed, color_from_seed(seed))
end

"""
    traverse!(network) → Bool

Take one step in the network using the edge gadget.
"""
function traverse!(net::InformationNetwork)
    current = net.walk.current
    
    # Get next color choice via gadget
    choice = next_color(net.gadget, net.structure, current)
    
    if !choice.valid
        # All edges blocked - try backtracking
        if saw_backtrack!(net.walk)
            return traverse!(net)  # Retry from previous position
        end
        return false  # Truly stuck
    end
    
    # Commit to the choice
    next_color!(net.gadget, choice)
    
    # Update walk state
    saw_step!(net.walk, net.structure, findfirst(==(choice.target), net.walk.available))
    
    # Steer the radar beam to the new direction
    dir = beam_direction(net.gadget)
    steer_beam!(net.radar, dir[1], 0.0)
    
    true
end

"""
    explore!(network, max_steps) → SAWPath

Explore the network until stuck or max_steps reached.
"""
function explore!(net::InformationNetwork, max_steps::Int=100)
    for _ in 1:max_steps
        if !traverse!(net)
            break
        end
    end
    
    # Create SAWPath from the walk
    colors = [color_from_seed(net.seed ⊻ UInt64(v)) for v in net.walk.path]
    fp = reduce(⊻, UInt64(v) for v in net.walk.path; init=net.seed)
    
    SAWPath(
        net.walk.path,
        length(net.walk.path),
        colors,
        !net.walk.stuck,
        net.seed,
        fp
    )
end

"""
    radar_sweep(network, angles) → Vector{BeamPattern}

Perform a radar sweep across the specified angles.
"""
function radar_sweep(net::InformationNetwork, angles::Vector{Float64})
    patterns = BeamPattern[]
    
    for θ in angles
        steer_beam!(net.radar, θ, 0.0)
        push!(patterns, compute_pattern(net.radar))
    end
    
    patterns
end

# ═══════════════════════════════════════════════════════════════════════════════
# TRAVERSAL STATE (UNIFIED VIEW)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    TraversalState

Complete state of a Gay Phased Array traversal.
Combines radar, walk, and gadget into unified view.
"""
struct TraversalState
    # Current position
    position::Int
    
    # Beam direction (from phases)
    beam::NTuple{2, Float64}
    
    # Available moves (constructive edges)
    constructive::Vector{Int}
    
    # Blocked moves (destructive edges)
    destructive::Vector{Int}
    
    # Path so far
    path::Vector{Int}
    
    # Path colors
    colors::Vector{NTuple{3, Float64}}
    
    # Current array pattern
    pattern::Union{BeamPattern, Nothing}
    
    # Is the walk complete?
    complete::Bool
    
    seed::UInt64
end

function TraversalState(net::InformationNetwork)
    current = net.walk.current
    beam = beam_direction(net.gadget)
    constr = constructive_edges(net.gadget, net.structure, current)
    destr = destructive_edges(net.gadget, net.structure, current)
    colors = [color_from_seed(net.seed ⊻ UInt64(v)) for v in net.walk.path]
    pattern = compute_pattern(net.radar)
    
    TraversalState(
        current, beam, constr, destr,
        net.walk.path, colors, pattern,
        net.walk.stuck || isempty(constr),
        net.seed
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    pattern_to_colors(pattern) → Vector{NTuple{3, Float64}}

Convert beam pattern to RGB colors for visualization.
"""
function pattern_to_colors(pattern::BeamPattern)
    n = length(pattern.azimuth_pattern)
    colors = NTuple{3, Float64}[]
    
    for (i, p) in enumerate(pattern.azimuth_pattern)
        # Color intensity based on pattern power
        intensity = sqrt(p)  # Square root for better dynamic range
        
        # Hue based on angle
        hue = (i - 1) / n
        
        # Convert HSV to RGB (simplified)
        r = intensity * (1 + cos(2π * hue)) / 2
        g = intensity * (1 + cos(2π * (hue - 1/3))) / 2  
        b = intensity * (1 + cos(2π * (hue - 2/3))) / 2
        
        push!(colors, (r, g, b))
    end
    
    colors
end

"""
    beam_to_rgb(azimuth, elevation, intensity) → NTuple{3, Float64}

Convert beam direction to RGB color.
"""
function beam_to_rgb(azimuth::Float64, elevation::Float64, intensity::Float64=1.0)
    # Azimuth maps to hue (0-360°)
    hue = (azimuth + π) / (2π)
    
    # Elevation maps to saturation
    sat = (elevation + π/2) / π
    
    # Simplified HSV to RGB
    r = intensity * (1 + cos(2π * hue)) / 2
    g = intensity * (1 + cos(2π * (hue - 1/3))) / 2
    b = intensity * (1 + cos(2π * (hue - 2/3))) / 2
    
    (r * sat, g * sat, b * sat)
end

# ═══════════════════════════════════════════════════════════════════════════════
# GAY RADAR (UNIFIED SYSTEM)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    GayRadar

Complete Gay Phased Array Radar system for self-avoiding network traversal.
"""
struct GayRadar
    network::InformationNetwork
    
    # History of traversal states
    history::Vector{TraversalState}
    
    # Statistics
    total_steps::Int
    backtrack_count::Int
    
    seed::UInt64
    color::NTuple{3, Float64}
    fingerprint::UInt64
end

function GayRadar(n_nodes::Int; 
                  connectivity::Float64=0.3,
                  n_antennas::Int=8,
                  seed::UInt64=GAY_SEED)
    network = InformationNetwork(n_nodes; 
                                  connectivity=connectivity, 
                                  n_antennas=n_antennas, 
                                  seed=seed)
    
    GayRadar(network, TraversalState[], 0, 0, seed, color_from_seed(seed), seed)
end

"""
    RadarState

Snapshot of radar state for serialization/inspection.
"""
struct RadarState
    position::Int
    beam_direction::NTuple{2, Float64}
    path_length::Int
    constructive_count::Int
    destructive_count::Int
    stuck::Bool
    
    color::NTuple{3, Float64}
    fingerprint::UInt64
end

function RadarState(radar::GayRadar)
    ts = TraversalState(radar.network)
    
    RadarState(
        ts.position,
        ts.beam,
        length(ts.path),
        length(ts.constructive),
        length(ts.destructive),
        ts.complete,
        radar.color,
        radar.fingerprint
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

function world_gay_phased_array()
    println()
    println("╔═══════════════════════════════════════════════════════════════════════════╗")
    println("║  GAY PHASED ARRAY RADAR: Self-Avoiding Chromatic Traversal               ║")
    println("║  Beam steering via next_color gadget semantics                           ║")
    println("╚═══════════════════════════════════════════════════════════════════════════╝")
    println()
    
    # ─── Create Phased Array ───
    println("─── Phased Array Configuration ───")
    array = PhasedArray(8; spacing=0.5)
    println("  Elements: $(length(array.antennas))")
    println("  Spacing: $(array.spacing) wavelengths")
    for (i, ant) in enumerate(array.antennas[1:3])
        println("    Antenna $i: phase=$(round(ant.phase, digits=3)), color=$(round.(ant.color, digits=2))")
    end
    println("    ...")
    println()
    
    # ─── Create Network ───
    println("─── Information Network ───")
    network = SAWNetwork(12; connectivity=0.4)
    println("  Vertices: $(network.n_vertices)")
    total_edges = sum(length(n) for n in network.neighbors) ÷ 2
    println("  Edges: $total_edges")
    println("  Edge colors: $(length(network.edge_colors)) unique")
    println()
    
    # ─── Edge Gadget with Verifier Game ───
    println("─── Edge Variable Gadget (Verifier Outcoloring) ───")
    start_color = color_from_seed(GAY_SEED)
    gadget = EdgeGadget(start_color; phase_tolerance=0.3)
    println("  Start color: $(round.(gadget.current_color, digits=2))")
    println("  Start phase: $(round(gadget.current_phase, digits=3))")
    println("  Active verifiers: $(length(gadget.verifiers))")
    
    # Get next color choice (pure)
    choice = next_color(gadget, network, 1)
    println("  next_color(1) → target=$(choice.target), valid=$(choice.valid)")
    println("    Color: $(round.(choice.color, digits=2))")
    println("    Phase: $(round(choice.phase, digits=3))")
    
    # Commit with verifier game
    result = next_color!(gadget, choice)
    status = verifier_status(gadget)
    println("  next_color!(choice) → outcoloring result:")
    println("    Attempts: $(status.attempts)")
    println("    Success rate: $(round(status.success_rate * 100, digits=1))%")
    println("    RNG splits: $(status.rng_splits)")
    println("    Entropy budget: $(round(status.entropy_budget, digits=2))")
    println()
    
    # Add custom verifier and test outcoloring
    println("─── Custom Verifier Test ───")
    add_verifier!(gadget, (c, p, s) -> c[1] > 0.9; strength=0.7)  # No very red colors
    println("  Added verifier: 'no very red colors' (strength=0.7)")
    
    # Derive truly random color that defeats all verifiers
    random_result = next_color!(gadget)
    status2 = verifier_status(gadget)
    println("  next_color!() → truly random derivation:")
    println("    Color: $(round.(random_result.color, digits=2))")
    println("    Success: $(random_result.valid)")
    println("    Total attempts: $(status2.attempts)")
    println("    Verifiers defeated: $(status2.defeated_verifiers)")
    println()
    
    # ─── Self-Avoiding Walk ───
    println("─── Self-Avoiding Random Walk ───")
    saw = SAWState(1)
    steps_taken = 0
    while saw_step!(saw, network, nothing) && steps_taken < 10
        steps_taken += 1
    end
    println("  Starting vertex: 1")
    println("  Path: $(saw.path)")
    println("  Length: $(length(saw.path))")
    println("  Stuck: $(saw.stuck)")
    println("  Valid SAW: $(is_valid_saw(saw.path))")
    println()
    
    # ─── Full System ───
    println("─── Gay Radar (Full System) ───")
    info_net = InformationNetwork(20; connectivity=0.35, n_antennas=8)
    println("  Network nodes: $(length(info_net.nodes))")
    println("  Radar antennas: $(length(info_net.radar.antennas))")
    
    # Explore the network
    path = explore!(info_net, 15)
    println("  Exploration:")
    println("    Path: $(path.vertices)")
    println("    Length: $(path.length)")
    println("    Completed: $(path.completed)")
    println()
    
    # Get traversal state
    state = TraversalState(info_net)
    println("─── Traversal State ───")
    println("  Position: $(state.position)")
    println("  Beam direction: ($(round(state.beam[1], digits=3)), $(round(state.beam[2], digits=3)))")
    println("  Constructive edges: $(state.constructive)")
    println("  Destructive edges: $(state.destructive)")
    println("  Complete: $(state.complete)")
    println()
    
    # ─── Beam Pattern ───
    println("─── Beam Pattern ───")
    pattern = compute_pattern(info_net.radar)
    println("  Main lobe: azimuth=$(round(pattern.main_lobe[1], digits=3)) rad")
    println("  Sidelobe level: $(round(pattern.sidelobe_level, digits=3))")
    peak_power = maximum(pattern.azimuth_pattern)
    println("  Peak power: $(round(peak_power, digits=3))")
    println()
    
    # ─── Summary ───
    println("─── Summary: Phased Array × SAW × Chromatic Gadgets ───")
    println("  • Phase φᵢ = chromatic identity of antenna/vertex")
    println("  • Beam direction = next_color choice → traversal direction")
    println("  • Self-avoidance = don't reuse phases (colors)")
    println("  • Constructive interference = valid SAW edge")
    println("  • Destructive interference = blocked (already visited)")
    println("  • next_color(state) → candidate (pure)")
    println("  • next_color!(state) → commit (mutating)")
    
    (array=array, network=network, gadget=gadget, path=path, state=state)
end

end # module GayPhasedArray
