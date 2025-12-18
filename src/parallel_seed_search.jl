# Maximally Parallel Seed Search with Minimal Syncpoints
# 
# Key insight from Fugue: avoid interleaving by keeping workers in separate "subtrees"
# Each worker writes to its own slot - only final merge requires sync.
#
# Target colors from Screenshot 2025-12-07:
#   1. Purple:     #8040B0 = RGB(128, 64, 176)
#   2. Light Blue: #60A0D0 = RGB(96, 160, 208)
#   3. Dark Blue:  #4070A0 = RGB(64, 112, 160)
#   4. Green:      #40A060 = RGB(64, 160, 96)
#   5. Orange:     #D08040 = RGB(208, 128, 64)

using Colors
using Base.Threads: @spawn, nthreads, threadid

export find_seeds_parallel, TARGET_COLORS, SearchResult
export ParallelSearchWorld, world_parallel_search

# Target colors from the screenshot
const TARGET_COLORS = [
    RGB(128/255, 64/255, 176/255),   # Purple #8040B0
    RGB(96/255, 160/255, 208/255),   # Light Blue #60A0D0
    RGB(64/255, 112/255, 160/255),   # Dark Blue #4070A0
    RGB(64/255, 160/255, 96/255),    # Green #40A060
    RGB(208/255, 128/255, 64/255),   # Orange #D08040
]

struct SearchResult
    seed::UInt64
    score::Float64
    found_at_worker::Int
    colors::Vector{RGB{Float64}}
end

"""
    color_distance_fast(c1, c2)

Fast squared distance in RGB space (no sqrt for speed).
"""
@inline function color_distance_fast(c1::RGB, c2::RGB)::Float64
    dr = Float64(red(c1)) - Float64(red(c2))
    dg = Float64(green(c1)) - Float64(green(c2))
    db = Float64(blue(c1)) - Float64(blue(c2))
    dr*dr + dg*dg + db*db
end

"""
    sequence_score(seed, targets)

Score how well colors from seed match targets in order.
Higher = better. Returns 1.0 for perfect match.
"""
function sequence_score(seed::UInt64, targets::Vector{RGB{Float64}})::Float64
    total_dist = 0.0
    n = length(targets)
    
    for i in 1:n
        # color_at is pure and O(1) - no sync needed
        c = color_at(i, SRGB(); seed=seed)
        total_dist += color_distance_fast(c, targets[i])
    end
    
    # Max possible dist is 3.0 * n (full RGB cube diagonal squared)
    max_dist = 3.0 * n
    1.0 - (total_dist / max_dist)
end

"""
    worker_search!(worker_id, seed_start, seed_count, targets, results_slot)

Independent worker - NO SYNC during execution.
Each worker writes to its own slot in results array.
Uses SplitMix64 for deterministic exploration from starting seed.
"""
function worker_search!(
    worker_id::Int,
    seed_start::UInt64,
    seed_count::Int,
    targets::Vector{RGB{Float64}},
    results_slot::Ref{SearchResult},
    threshold::Float64
)
    best_seed = seed_start
    best_score = 0.0
    current = seed_start
    
    # Local exploration - no locks, no sync
    for _ in 1:seed_count
        score = sequence_score(current, targets)
        
        if score > best_score
            best_score = score
            best_seed = current
            
            # Early exit if perfect match found (write-interleave safe)
            if score >= threshold
                colors = [color_at(i, SRGB(); seed=best_seed) for i in 1:length(targets)]
                results_slot[] = SearchResult(best_seed, best_score, worker_id, colors)
                return
            end
        end
        
        # SplitMix64 step - deterministic, pure
        current = splitmix64(current)
    end
    
    # Final write to own slot - only sync point
    colors = [color_at(i, SRGB(); seed=best_seed) for i in 1:length(targets)]
    results_slot[] = SearchResult(best_seed, best_score, worker_id, colors)
end

# Use splitmix64 from kernels.jl (already defined in Gay module)

"""
    find_seeds_parallel(targets=TARGET_COLORS; 
                        n_workers=nthreads(),
                        seeds_per_worker=100_000,
                        threshold=0.95)

Maximally parallel seed search with minimal syncpoints.

# Architecture (Fugue-inspired):
- Each worker explores an independent seed space subtree
- NO interleaving of writes between workers during search
- Single sync point: final merge of results
- Workers can early-exit on match (write-interleave to own slot)

# Syncpoints:
1. Initial spawn (unavoidable)
2. Final merge (unavoidable) 
3. Early-exit writes (to own slot only - no contention)
"""
function find_seeds_parallel(
    targets::Vector{RGB{Float64}}=TARGET_COLORS;
    n_workers::Int=nthreads(),
    seeds_per_worker::Int=100_000,
    threshold::Float64=0.95
)::Vector{SearchResult}
    
    # Pre-allocate result slots - each worker owns one (no contention)
    results = [Ref{SearchResult}(SearchResult(UInt64(0), 0.0, i, RGB{Float64}[])) 
               for i in 1:n_workers]
    
    # Generate independent starting seeds (orthogonal subtrees)
    # Using golden ratio spacing for maximal coverage
    φ = UInt64(0x9e3779b97f4a7c15)  # Golden ratio * 2^64
    starting_seeds = [UInt64(i) * φ for i in 1:n_workers]
    
    # SYNCPOINT 1: Spawn workers (unavoidable)
    tasks = [@spawn worker_search!(
        i, starting_seeds[i], seeds_per_worker, targets, results[i], threshold
    ) for i in 1:n_workers]
    
    # SYNCPOINT 2: Wait for all workers (unavoidable)
    foreach(wait, tasks)
    
    # Merge results - pure read, no sync needed
    all_results = [r[] for r in results]
    sort!(all_results, by=r -> -r.score)
    
    return all_results
end

"""
    ParallelSearchWorld

Persistent world structure for parallel seed search results.
Implements world_ pattern: composable, mergeable, fingerprinted.
"""
struct ParallelSearchWorld
    results::Vector{SearchResult}
    targets::Vector{RGB{Float64}}
    n_workers::Int
    elapsed_time::Float64
end

Base.length(w::ParallelSearchWorld) = length(w.results)

function Base.merge(w1::ParallelSearchWorld, w2::ParallelSearchWorld)
    combined = vcat(w1.results, w2.results)
    sort!(combined, by=r -> -r.score)
    ParallelSearchWorld(combined, w1.targets, w1.n_workers + w2.n_workers, w1.elapsed_time + w2.elapsed_time)
end

function fingerprint(w::ParallelSearchWorld)::UInt64
    h = UInt64(0x9e3779b97f4a7c15)
    for r in w.results
        h = xor(h, splitmix64(r.seed))
        h = splitmix64(h ⊻ reinterpret(UInt64, r.score))
    end
    h
end

"""
    world_parallel_search(; targets, n_workers, seeds_per_worker, threshold)

Build a ParallelSearchWorld with seed search results.
Returns persistent, composable structure with SPI fingerprint.
"""
function world_parallel_search(;
    targets::Vector{RGB{Float64}}=TARGET_COLORS,
    n_workers::Int=nthreads(),
    seeds_per_worker::Int=100_000,
    threshold::Float64=0.95
)::ParallelSearchWorld
    @debug "Parallel seed search" n_workers seeds_per_worker threshold
    
    t = @elapsed results = find_seeds_parallel(targets; n_workers, seeds_per_worker, threshold)
    
    @debug "Search complete" elapsed=t top_score=results[1].score
    
    ParallelSearchWorld(results, targets, n_workers, t)
end

# end of parallel_seed_search.jl
