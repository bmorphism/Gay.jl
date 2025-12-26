# Maximally Parallel Genetic Seed Search
#
# Key principles:
# 1. Minimal syncpoints (Fugue-inspired)
# 2. Island model: each thread evolves independent population
# 3. Only sync at migration epochs
# 4. O(1) fitness evaluation via hash_color

using Base.Threads: @spawn, nthreads, threadid, @threads
using Colors
using Random

export GeneticSearchConfig, GeneticSearchResult, GeneticSearchWorld
export genetic_search_parallel, island_evolution, world_genetic_search

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

"""
    GeneticSearchConfig

Configuration for maximally parallel genetic search.
"""
Base.@kwdef struct GeneticSearchConfig
    population_size::Int = 256
    generations::Int = 100
    mutation_rate::Float64 = 0.15
    crossover_rate::Float64 = 0.8
    elite_fraction::Float64 = 0.1
    tournament_size::Int = 3
    n_islands::Int = nthreads()
    migration_interval::Int = 10
    migration_fraction::Float64 = 0.1
    threshold::Float64 = 0.98
end

"""
    GeneticSearchResult

Result from genetic search.
"""
struct GeneticSearchResult
    seed::UInt64
    score::Float64
    generation::Int
    island::Int
    colors::Vector{RGB{Float64}}
end

# ═══════════════════════════════════════════════════════════════════════════
# Fitness Function (O(1) per color via hash)
# ═══════════════════════════════════════════════════════════════════════════

"""
    fitness(seed::UInt64, targets::Vector{NTuple{3,Float64}}) -> Float64

Compute fitness score for a seed (higher = better match to targets).
Uses O(1) hash_color for each color evaluation.
"""
@inline function fitness(seed::UInt64, targets::Vector{NTuple{3,Float64}})::Float64
    total_dist = 0.0
    @inbounds for (i, (tr, tg, tb)) in enumerate(targets)
        r, g, b = hash_color(seed, UInt64(i))
        dr = Float64(r) - tr
        dg = Float64(g) - tg
        db = Float64(b) - tb
        total_dist += dr*dr + dg*dg + db*db
    end
    # Normalize: max dist is 3.0 * n
    1.0 - (total_dist / (3.0 * length(targets)))
end

# ═══════════════════════════════════════════════════════════════════════════
# Genetic Operators
# ═══════════════════════════════════════════════════════════════════════════

"""
    tournament_select(pop, fit, k) -> UInt64

Tournament selection: pick k random individuals, return best.
"""
@inline function tournament_select(pop::Vector{UInt64}, fit::Vector{Float64}, k::Int)::UInt64
    n = length(pop)
    best_idx = rand(1:n)
    best_fit = fit[best_idx]
    
    for _ in 2:k
        idx = rand(1:n)
        if fit[idx] > best_fit
            best_idx = idx
            best_fit = fit[idx]
        end
    end
    
    pop[best_idx]
end

"""
    crossover(a::UInt64, b::UInt64) -> (UInt64, UInt64)

Two-point crossover on 64-bit seeds.
"""
@inline function crossover(a::UInt64, b::UInt64)::Tuple{UInt64, UInt64}
    p1, p2 = minmax(rand(1:63), rand(1:63))
    mask = ((UInt64(1) << p2) - 1) ⊻ ((UInt64(1) << p1) - 1)
    
    child1 = (a & ~mask) | (b & mask)
    child2 = (b & ~mask) | (a & mask)
    
    (child1, child2)
end

"""
    mutate(seed::UInt64, rate::Float64) -> UInt64

Bit-flip mutation with given rate.
"""
@inline function mutate(seed::UInt64, rate::Float64)::UInt64
    if rand() < rate
        # Flip 1-3 random bits
        n_flips = rand(1:3)
        for _ in 1:n_flips
            bit = rand(0:63)
            seed ⊻= UInt64(1) << bit
        end
    end
    seed
end

# ═══════════════════════════════════════════════════════════════════════════
# Island Evolution (Single Thread)
# ═══════════════════════════════════════════════════════════════════════════

"""
    island_evolution(island_id, targets, config) -> (best_seed, best_score, best_gen)

Run genetic evolution on a single island (single thread).
Returns best result found by this island.
"""
function island_evolution(
    island_id::Int,
    targets::Vector{NTuple{3,Float64}},
    config::GeneticSearchConfig
)
    pop_size = config.population_size ÷ config.n_islands
    n_elite = max(1, round(Int, pop_size * config.elite_fraction))
    
    # Initialize population with deterministic golden ratio spacing
    φ = UInt64(0x9e3779b97f4a7c15)
    population = [UInt64(island_id * 100000 + i) * φ for i in 1:pop_size]
    new_pop = similar(population)
    fit = zeros(Float64, pop_size)
    
    best_seed = population[1]
    best_score = 0.0
    best_gen = 0
    
    # Evaluate initial population
    for i in 1:pop_size
        fit[i] = fitness(population[i], targets)
        if fit[i] > best_score
            best_score = fit[i]
            best_seed = population[i]
            best_gen = 0
        end
    end
    
    # Evolution loop
    for gen in 1:config.generations
        # Sort by fitness for elitism
        perm = sortperm(fit, rev=true)
        
        # Preserve elites
        for i in 1:n_elite
            new_pop[i] = population[perm[i]]
        end
        
        # Generate rest via selection + crossover + mutation
        idx = n_elite + 1
        while idx <= pop_size
            # Tournament selection
            parent1 = tournament_select(population, fit, config.tournament_size)
            parent2 = tournament_select(population, fit, config.tournament_size)
            
            # Crossover
            if rand() < config.crossover_rate && idx < pop_size
                child1, child2 = crossover(parent1, parent2)
                new_pop[idx] = mutate(child1, config.mutation_rate)
                new_pop[idx+1] = mutate(child2, config.mutation_rate)
                idx += 2
            else
                new_pop[idx] = mutate(parent1, config.mutation_rate)
                idx += 1
            end
        end
        
        # Swap populations
        population, new_pop = new_pop, population
        
        # Evaluate new population
        for i in 1:pop_size
            fit[i] = fitness(population[i], targets)
            if fit[i] > best_score
                best_score = fit[i]
                best_seed = population[i]
                best_gen = gen
            end
        end
        
        # Early exit if threshold reached
        if best_score >= config.threshold
            break
        end
    end
    
    (best_seed, best_score, best_gen)
end

# ═══════════════════════════════════════════════════════════════════════════
# Maximally Parallel Search (Island Model)
# ═══════════════════════════════════════════════════════════════════════════

"""
    genetic_search_parallel(targets; config=GeneticSearchConfig()) -> Vector{GeneticSearchResult}

Maximally parallel genetic search using island model.

# Syncpoints:
1. Initial spawn (unavoidable)
2. Final merge (unavoidable)

Each island runs completely independently - no inter-island communication.
This maximizes parallelism at the cost of no migration.
"""
function genetic_search_parallel(
    targets::Vector{<:Any} = TARGET_COLORS;
    config::GeneticSearchConfig = GeneticSearchConfig()
)
    # Convert targets to tuples for performance
    target_tuples = [(Float64(red(c)), Float64(green(c)), Float64(blue(c))) for c in targets]
    
    n_islands = config.n_islands
    
    # Pre-allocate result slots (each island writes to own slot - no contention)
    island_results = [Ref{Tuple{UInt64, Float64, Int}}((UInt64(0), 0.0, 0)) for _ in 1:n_islands]
    
    # SYNCPOINT 1: Spawn all islands
    tasks = [@spawn begin
        result = island_evolution(i, target_tuples, config)
        island_results[i][] = result
    end for i in 1:n_islands]
    
    # SYNCPOINT 2: Wait for all islands
    foreach(wait, tasks)
    
    # Merge results (pure read)
    results = GeneticSearchResult[]
    for i in 1:n_islands
        seed, score, gen = island_results[i][]
        colors = [color_at(j; seed=seed) for j in 1:length(targets)]
        push!(results, GeneticSearchResult(seed, score, gen, i, colors))
    end
    
    sort!(results, by=r -> -r.score)
    results
end

# ═══════════════════════════════════════════════════════════════════════════
# World Builder
# ═══════════════════════════════════════════════════════════════════════════

"""
    GeneticSearchWorld

Persistent world containing genetic search results.
Implements length, merge, fingerprint for composability.
"""
struct GeneticSearchWorld
    results::Vector{GeneticSearchResult}
    config::GeneticSearchConfig
    targets::Vector{RGB{Float64}}
    elapsed_time::Float64
end

Base.length(w::GeneticSearchWorld) = length(w.results)

function Base.merge(w1::GeneticSearchWorld, w2::GeneticSearchWorld)
    combined = vcat(w1.results, w2.results)
    sort!(combined, by=r -> -r.score)
    GeneticSearchWorld(combined, w1.config, w1.targets, w1.elapsed_time + w2.elapsed_time)
end

function fingerprint(w::GeneticSearchWorld)::UInt64
    h = UInt64(0x9e3779b97f4a7c15)
    for r in w.results
        h = xor(h, splitmix64(r.seed))
        h = splitmix64(h)
    end
    h
end

"""
    world_genetic_search(; config=GeneticSearchConfig()) -> GeneticSearchWorld

Build a persistent world from maximally parallel genetic search.
Returns composable GeneticSearchWorld structure.
"""
function world_genetic_search(;
    targets::Vector{<:Any} = TARGET_COLORS,
    config::GeneticSearchConfig = GeneticSearchConfig(
        population_size = 512,
        generations = 100,
        n_islands = nthreads(),
        threshold = 0.98
    )
)::GeneticSearchWorld
    @debug "Starting genetic search" n_islands=config.n_islands generations=config.generations
    
    elapsed = @elapsed results = genetic_search_parallel(targets; config=config)
    
    target_rgbs = [RGB{Float64}(red(c), green(c), blue(c)) for c in targets]
    
    @debug "Genetic search complete" elapsed=elapsed best_score=results[1].score
    
    GeneticSearchWorld(results, config, target_rgbs, elapsed)
end

# end of genetic_search.jl
