# Maximally parallel seed search with Fugue-inspired minimal syncpoints
# Uses SplittableRandom for Strong Parallelism Invariance (SPI)

using SplittableRandoms: SplittableRandom, split

export find_seeds_parallel, TARGET_COLORS, SearchResult, demo_parallel_search

"""
Target colors for seed search - finding seeds that produce these exact colors.
"""
const TARGET_COLORS = [
    (0.315, 0.160, 0.000),  # amber
    (0.614, 0.000, 0.250),  # crimson  
    (0.799, 0.922, 1.000),  # sky blue
]

"""
Result of a parallel seed search.
"""
struct SearchResult
    seed::UInt64
    color::Tuple{Float64, Float64, Float64}
    distance::Float64
    thread_id::Int
end

"""
    color_distance(c1, c2)

Euclidean distance between two RGB colors.
"""
function color_distance(c1::Tuple{Float64, Float64, Float64}, 
                        c2::Tuple{Float64, Float64, Float64})
    sqrt((c1[1] - c2[1])^2 + (c1[2] - c2[2])^2 + (c1[3] - c2[3])^2)
end

"""
    seed_to_color(seed::UInt64)

Generate a color from a seed using SplitMix64.
"""
function seed_to_color(seed::UInt64)
    rng = SplittableRandom(seed)
    rng = split(rng)
    
    # Extract RGB from random bits
    r = Float64(rand(rng, UInt64) >> 11) / Float64(2^53)
    g = Float64(rand(rng, UInt64) >> 11) / Float64(2^53)
    b = Float64(rand(rng, UInt64) >> 11) / Float64(2^53)
    
    return (r, g, b)
end

"""
    find_seeds_parallel(target::Tuple{Float64, Float64, Float64}; 
                        n_seeds=1_000_000, threshold=0.1)

Search for seeds that produce colors close to the target.
Uses all available threads with minimal synchronization (Fugue-style).
"""
function find_seeds_parallel(target::Tuple{Float64, Float64, Float64};
                             n_seeds::Int=1_000_000,
                             threshold::Float64=0.1)
    results = SearchResult[]
    results_lock = ReentrantLock()
    
    n_threads = Threads.nthreads()
    chunk_size = n_seeds ÷ n_threads
    
    Threads.@threads for t in 1:n_threads
        local_results = SearchResult[]
        start_seed = UInt64((t - 1) * chunk_size)
        end_seed = UInt64(t * chunk_size - 1)
        
        for seed in start_seed:end_seed
            color = seed_to_color(seed)
            dist = color_distance(color, target)
            
            if dist < threshold
                push!(local_results, SearchResult(seed, color, dist, t))
            end
        end
        
        # Minimal syncpoint: only lock when adding results
        if !isempty(local_results)
            lock(results_lock) do
                append!(results, local_results)
            end
        end
    end
    
    # Sort by distance
    sort!(results, by=r -> r.distance)
    return results
end

"""
    demo_parallel_search()

Demonstrate parallel seed search with default targets.
"""
function demo_parallel_search()
    println("Gay.jl Parallel Seed Search")
    println("=" ^ 40)
    println("Threads: $(Threads.nthreads())")
    println()
    
    for (i, target) in enumerate(TARGET_COLORS)
        println("Target $i: RGB$(target)")
        t = @elapsed results = find_seeds_parallel(target; n_seeds=100_000, threshold=0.2)
        println("  Found $(length(results)) matches in $(round(t, digits=3))s")
        if !isempty(results)
            best = results[1]
            println("  Best: seed=$(best.seed) RGB$(best.color) dist=$(round(best.distance, digits=4))")
        end
        println()
    end
end
