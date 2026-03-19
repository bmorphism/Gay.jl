"""
    gay_eg_walker.jl

Graph walker with chromatic identity and SPI (Strong Parallelism Invariance).

Performs Euclidean-guided random walks on graphs where:
- Each step gets a deterministic color from gay_seed
- State tracking via chromatic history
- Energy-aware step selection
- SPI verification across parallel workers

Note: This module works with any graph representation (dict-based, matrix, etc.)
and doesn't require Graphs.jl as a hard dependency.
"""

module GayEGWalker

using Colors, Statistics
using SplittableRandoms: SplittableRandom, split

export
    # Structures
    EGWalkerState,
    WalkResult,
    
    # Core API
    create_walker,
    step_walker!,
    walk!,
    
    # Analysis
    walk_colors,
    walk_energy,
    walk_path,
    verify_spi

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS & TYPES
# ═══════════════════════════════════════════════════════════════════════════

const EG_EPSILON = 1e-10
const MAX_WALK_LENGTH = 100000

# ───────────────────────────────────────────────────────────────────────────
# Splitmix64 PRNG (deterministic color generation)
# ───────────────────────────────────────────────────────────────────────────

function splitmix64(state::UInt64)::Tuple{UInt64, UInt64}
    """SplitMix64 PRNG - returns (output, next_state)"""
    z = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ⊻ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    z ⊻ (z >> 31), (state + 1) & 0xFFFFFFFFFFFFFFFF
end

function color_from_seed(seed::UInt64)::RGB{Float64}
    """Generate deterministic RGB color from seed"""
    r, s1 = splitmix64(seed)
    g, s2 = splitmix64(s1)
    b, _  = splitmix64(s2)
    RGB(
        (r >> 56) / 255.0,
        (g >> 56) / 255.0,
        (b >> 56) / 255.0
    )
end

function next_color(step::UInt64, thread_id::UInt64)::RGB{Float64}
    """Get color for (step, thread_id) via gay_seed"""
    seed = ((step + 1) << 32) | (thread_id & 0xFFFFFFFFFFFFFFFF)
    color_from_seed(seed)
end

# ───────────────────────────────────────────────────────────────────────────
# Walker State
# ───────────────────────────────────────────────────────────────────────────

"""
    EGWalkerState

State of a Euclidean-guided random walk on a graph.
"""
mutable struct EGWalkerState
    # Graph structure (adjacency list: vertex -> neighbors)
    adjacency::Vector{Vector{Int64}}
    positions::Vector{Tuple{Float64, Float64}}  # Euclidean positions
    
    # Walk state
    current_vertex::Int64
    step_count::UInt64
    path::Vector{Int64}
    
    # RNG for parallelism
    rng::SplittableRandom
    thread_id::UInt64
    
    # Colors (deterministic from gay_seed)
    colors::Vector{RGB{Float64}}
    
    # Energy tracking
    energy_per_step::Vector{Float64}  # Distance traveled
    total_energy::Float64
    
    # Metadata
    start_vertex::Int64
    target_vertex::Union{Int64, Nothing}
end

"""
    WalkResult

Result of a completed walk.
"""
struct WalkResult
    path::Vector{Int64}
    colors::Vector{RGB{Float64}}
    energy::Vector{Float64}
    total_energy::Float64
    steps::UInt64
    reached_target::Bool
    spi_hash::UInt64  # For verification
end

# ═══════════════════════════════════════════════════════════════════════════
# WALKER CREATION
# ═══════════════════════════════════════════════════════════════════════════

"""
    create_walker(adjacency, positions; seed=0x42, thread_id=0, start=1, target=nothing)

Create an EG walker on the given graph with Euclidean positions.

# Arguments
- `adjacency`: Vector of vectors representing adjacency list (vertex -> neighbors)
- `positions`: Vector of (x, y) tuples for vertices
- `seed`: Root seed for SplittableRNG (default: 0x42)
- `thread_id`: ID for this walker's thread (affects color generation)
- `start`: Starting vertex (default: 1)
- `target`: Target vertex to reach (optional)
"""
function create_walker(
    adjacency::Vector{Vector{Int64}},
    positions::Vector{Tuple{Float64, Float64}};
    seed::UInt64=0x42,
    thread_id::UInt64=0,
    start::Int64=1,
    target::Union{Int64, Nothing}=nothing
)::EGWalkerState
    
    n = length(adjacency)
    @assert 1 <= start <= n "Start vertex out of range"
    if target !== nothing
        @assert 1 <= target <= n "Target vertex out of range"
    end
    @assert length(positions) == n "Position count mismatch"
    
    rng = SplittableRandom(seed)
    
    EGWalkerState(
        adjacency=adjacency,
        positions=positions,
        current_vertex=start,
        step_count=0,
        path=Int64[start],
        rng=rng,
        thread_id=thread_id,
        colors=RGB{Float64}[next_color(0, thread_id)],
        energy_per_step=Float64[],
        total_energy=0.0,
        start_vertex=start,
        target_vertex=target
    )
end

# ═══════════════════════════════════════════════════════════════════════════
# EUCLIDEAN-GUIDED STEP SELECTION
# ═══════════════════════════════════════════════════════════════════════════

"""
    euclidean_distance(p1, p2)

Compute Euclidean distance between two positions.
"""
function euclidean_distance(p1::Tuple{Float64, Float64}, p2::Tuple{Float64, Float64})::Float64
    sqrt((p1[1] - p2[1])^2 + (p1[2] - p2[2])^2)
end

"""
    eg_weights(neighbors, current_pos, positions)

Compute EG weights for neighbors based on Euclidean distance.

Weight = 1 / (distance + epsilon) to bias toward nearby vertices.
"""
function eg_weights(
    neighbors::Vector{Int64},
    current_pos::Tuple{Float64, Float64},
    positions::Vector{Tuple{Float64, Float64}}
)::Vector{Float64}
    
    if isempty(neighbors)
        return Float64[]
    end
    
    distances = [euclidean_distance(current_pos, positions[n]) for n in neighbors]
    
    # Inverse distance weighting: closer = higher probability
    weights = 1.0 ./ (distances .+ EG_EPSILON)
    weights / sum(weights)  # Normalize to probability
end

"""
    step_walker_internal!(walker; target_prob=0.1)

Internal step implementation without graph dependency.
"""
function step_walker_internal!(walker::EGWalkerState; target_prob::Float64=0.1)
    current = walker.current_vertex
    neighbors = walker.adjacency[current]
    
    if isempty(neighbors)
        # Dead end, stay in place
        return false
    end
    
    # Decide: EG step vs target step
    use_target = (walker.target_vertex !== nothing) && (rand() < target_prob)
    
    if use_target
        # Move toward target via inverse distance
        target_pos = walker.positions[walker.target_vertex]
        next_idx = argmin([euclidean_distance(walker.positions[n], target_pos) for n in neighbors])
        next_vertex = neighbors[next_idx]
    else
        # EG-guided: weight by inverse distance
        weights = eg_weights(neighbors, walker.positions[current], walker.positions)
        if isempty(weights)
            next_vertex = first(neighbors)
        else
            # Weighted random selection
            r = rand()
            cumsum_w = 0.0
            next_vertex = neighbors[end]
            for (idx, w) in enumerate(weights)
                cumsum_w += w
                if r < cumsum_w
                    next_vertex = neighbors[idx]
                    break
                end
            end
        end
    end
    
    # Update state
    walker.step_count += 1
    push!(walker.path, next_vertex)
    
    # Compute energy (distance traveled)
    current_pos = walker.positions[current]
    next_pos = walker.positions[next_vertex]
    energy = euclidean_distance(current_pos, next_pos)
    push!(walker.energy_per_step, energy)
    walker.total_energy += energy
    
    # Add color
    color = next_color(walker.step_count, walker.thread_id)
    push!(walker.colors, color)
    
    # Update position
    walker.current_vertex = next_vertex
    
    # Check if reached target
    return walker.target_vertex !== nothing && next_vertex == walker.target_vertex
end

# ═══════════════════════════════════════════════════════════════════════════
# WALKER STEP
# ═══════════════════════════════════════════════════════════════════════════

"""
    step_walker!(walker; target_prob=0.1)

Take one step in the walk using Euclidean-guided selection.

If target is set and with probability `target_prob`, move toward target instead.
"""
function step_walker!(walker::EGWalkerState; target_prob::Float64=0.1)
    return step_walker_internal!(walker; target_prob=target_prob)
end

"""
    walk!(walker, max_steps; target_prob=0.1)

Execute walk for up to max_steps or until reaching target.

Returns true if target was reached, false otherwise.
"""
function walk!(
    walker::EGWalkerState,
    max_steps::Int64;
    target_prob::Float64=0.1
)::Bool
    
    @assert max_steps > 0 "max_steps must be positive"
    @assert max_steps <= MAX_WALK_LENGTH "max_steps too large"
    
    for _ in 1:max_steps
        reached = step_walker!(walker; target_prob=target_prob)
        if reached
            return true
        end
    end
    
    return false
end

# ═══════════════════════════════════════════════════════════════════════════
# RESULT EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

"""
    walk_colors(walker)::Vector{RGB{Float64}}

Get the colors assigned to each step.
"""
walk_colors(walker::EGWalkerState) = walker.colors

"""
    walk_energy(walker)::Vector{Float64}

Get the energy (distance) for each step.
"""
walk_energy(walker::EGWalkerState) = walker.energy_per_step

"""
    walk_path(walker)::Vector{Int64}

Get the vertex path taken.
"""
walk_path(walker::EGWalkerState) = walker.path

"""
    result(walker)::WalkResult

Convert walker state to result structure.
"""
function result(walker::EGWalkerState)::WalkResult
    reached = walker.target_vertex !== nothing && walker.current_vertex == walker.target_vertex
    
    # SPI hash: Combine path and colors deterministically
    spi_hash = UInt64(0)
    for (v, c) in zip(walker.path, walker.colors)
        spi_hash ⊻= UInt64(v)
        spi_hash = (spi_hash << 7) | (spi_hash >> 57)
        # Hash color components
        r_bits = UInt64(Int64(round(c.r * 255)))
        g_bits = UInt64(Int64(round(c.g * 255)))
        b_bits = UInt64(Int64(round(c.b * 255)))
        spi_hash ⊻= (r_bits << 8) | (g_bits << 16) | (b_bits << 24)
    end
    
    WalkResult(
        path=walker.path,
        colors=walker.colors,
        energy=walker.energy_per_step,
        total_energy=walker.total_energy,
        steps=walker.step_count,
        reached_target=reached,
        spi_hash=spi_hash
    )
end

# ═══════════════════════════════════════════════════════════════════════════
# SPI VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════

"""
    verify_spi(results::Vector{WalkResult})::Bool

Verify Strong Parallelism Invariance: all results have same SPI hash.

This confirms that despite parallel execution, the walk sequence is deterministic.
"""
function verify_spi(results::Vector{WalkResult})::Bool
    if isempty(results)
        return true
    end
    
    first_hash = results[1].spi_hash
    all(r -> r.spi_hash == first_hash, results)
end

# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION & ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

"""
    print_walk(result::WalkResult)

Print walk summary with colors.
"""
function print_walk(result::WalkResult)
    println("\n═══════════════════════════════════════════════════════════════")
    println("WALK RESULT")
    println("═══════════════════════════════════════════════════════════════")
    println("Path length: $(length(result.path)) vertices")
    println("Steps: $(result.steps)")
    println("Total energy: $(round(result.total_energy; digits=4))")
    println("Reached target: $(result.reached_target)")
    println("SPI hash: 0x$(string(result.spi_hash; base=16))")
    
    println("\nPath (first 20 vertices):")
    path_str = join(result.path[1:min(20, length(result.path))], " → ")
    if length(result.path) > 20
        path_str *= " → ..."
    end
    println(path_str)
    
    println("\nColor sequence (RGB values):")
    for (i, color) in enumerate(result.colors[1:min(5, length(result.colors))])
        println("  Step $i: R=$(round(color.r; digits=3)), G=$(round(color.g; digits=3)), B=$(round(color.b; digits=3))")
    end
    if length(result.colors) > 5
        println("  ... ($(length(result.colors) - 5) more colors)")
    end
    
    println("\nEnergy per step (first 10):")
    for (i, e) in enumerate(result.energy[1:min(10, length(result.energy))])
        println("  Step $i: $(round(e; digits=4))")
    end
    if length(result.energy) > 10
        println("  ... ($(length(result.energy) - 10) more steps)")
    end
    
    println("\n═══════════════════════════════════════════════════════════════\n")
end

end  # module GayEGWalker
