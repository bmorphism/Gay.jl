# Genetic search for optimal color seeds
# Island-based parallel evolution with SPI guarantees

using SplittableRandoms: SplittableRandom, split

export GeneticSearchConfig, GeneticSearchResult
export genetic_search_parallel, island_evolution, demo_genetic_search

"""
Configuration for genetic search.
"""
Base.@kwdef struct GeneticSearchConfig
    population_size::Int = 100
    n_generations::Int = 50
    n_islands::Int = 4
    mutation_rate::Float64 = 0.1
    crossover_rate::Float64 = 0.7
    migration_rate::Float64 = 0.05
    elite_fraction::Float64 = 0.1
end

"""
Result of genetic search.
"""
struct GeneticSearchResult
    best_seed::UInt64
    best_fitness::Float64
    generations::Int
    island_id::Int
end

"""
    fitness(seed::UInt64, target::Tuple{Float64, Float64, Float64})

Evaluate fitness of a seed for producing the target color.
Higher is better.
"""
function fitness(seed::UInt64, target::Tuple{Float64, Float64, Float64})
    rng = SplittableRandom(seed)
    rng = split(rng)
    
    r = Float64(rand(rng, UInt64) >> 11) / Float64(2^53)
    g = Float64(rand(rng, UInt64) >> 11) / Float64(2^53)
    b = Float64(rand(rng, UInt64) >> 11) / Float64(2^53)
    
    # Distance-based fitness (inverted - closer = higher fitness)
    dist = sqrt((r - target[1])^2 + (g - target[2])^2 + (b - target[3])^2)
    return 1.0 / (1.0 + dist)
end

"""
    mutate(seed::UInt64, rng::SplittableRandom, rate::Float64)

Mutate a seed by flipping random bits.
"""
function mutate(seed::UInt64, rng::SplittableRandom, rate::Float64)
    if rand(rng, Float64) < rate
        # Flip 1-3 random bits
        n_flips = 1 + (rand(rng, UInt64) % 3)
        for _ in 1:n_flips
            bit = rand(rng, UInt64) % 64
            seed = xor(seed, UInt64(1) << bit)
        end
    end
    return seed
end

"""
    crossover(parent1::UInt64, parent2::UInt64, rng::SplittableRandom)

Single-point crossover between two seeds.
"""
function crossover(parent1::UInt64, parent2::UInt64, rng::SplittableRandom)
    point = rand(rng, UInt64) % 63 + 1  # 1-63
    mask = (UInt64(1) << point) - 1
    child = (parent1 & mask) | (parent2 & ~mask)
    return child
end

"""
    island_evolution(island_id::Int, target::Tuple{Float64, Float64, Float64}, 
                     config::GeneticSearchConfig)

Run evolution on a single island.
"""
function island_evolution(island_id::Int, 
                          target::Tuple{Float64, Float64, Float64},
                          config::GeneticSearchConfig)
    # Initialize RNG with island-specific seed for reproducibility
    rng = SplittableRandom(UInt64(island_id * 0x9e3779b97f4a7c15))
    
    # Initialize population
    population = [rand(rng, UInt64) for _ in 1:config.population_size]
    fitnesses = [fitness(s, target) for s in population]
    
    best_seed = population[argmax(fitnesses)]
    best_fit = maximum(fitnesses)
    
    for gen in 1:config.n_generations
        # Sort by fitness
        order = sortperm(fitnesses, rev=true)
        population = population[order]
        fitnesses = fitnesses[order]
        
        # Elite selection
        n_elite = max(1, round(Int, config.elite_fraction * config.population_size))
        new_population = population[1:n_elite]
        
        # Generate offspring
        while length(new_population) < config.population_size
            # Tournament selection
            i1, i2 = rand(rng, UInt64) % config.population_size + 1,
                     rand(rng, UInt64) % config.population_size + 1
            parent1 = fitnesses[i1] > fitnesses[i2] ? population[i1] : population[i2]
            
            i1, i2 = rand(rng, UInt64) % config.population_size + 1,
                     rand(rng, UInt64) % config.population_size + 1
            parent2 = fitnesses[i1] > fitnesses[i2] ? population[i1] : population[i2]
            
            # Crossover
            rng = split(rng)
            child = if rand(rng, Float64) < config.crossover_rate
                crossover(parent1, parent2, rng)
            else
                parent1
            end
            
            # Mutation
            rng = split(rng)
            child = mutate(child, rng, config.mutation_rate)
            
            push!(new_population, child)
        end
        
        population = new_population
        fitnesses = [fitness(s, target) for s in population]
        
        # Track best
        gen_best_idx = argmax(fitnesses)
        if fitnesses[gen_best_idx] > best_fit
            best_seed = population[gen_best_idx]
            best_fit = fitnesses[gen_best_idx]
        end
    end
    
    return GeneticSearchResult(best_seed, best_fit, config.n_generations, island_id)
end

"""
    genetic_search_parallel(target::Tuple{Float64, Float64, Float64};
                            config::GeneticSearchConfig=GeneticSearchConfig())

Run parallel genetic search across multiple islands.
"""
function genetic_search_parallel(target::Tuple{Float64, Float64, Float64};
                                 config::GeneticSearchConfig=GeneticSearchConfig())
    results = Vector{GeneticSearchResult}(undef, config.n_islands)
    
    Threads.@threads for i in 1:config.n_islands
        results[i] = island_evolution(i, target, config)
    end
    
    # Return best across all islands
    best_idx = argmax([r.best_fitness for r in results])
    return results[best_idx]
end

"""
    demo_genetic_search()

Demonstrate genetic search for color seeds.
"""
function demo_genetic_search()
    println("Gay.jl Genetic Seed Search")
    println("=" ^ 40)
    println("Islands: 4, Generations: 50")
    println()
    
    targets = [
        (1.0, 0.0, 0.0),  # Red
        (0.0, 1.0, 0.0),  # Green
        (0.0, 0.0, 1.0),  # Blue
    ]
    
    for (i, target) in enumerate(targets)
        println("Target $i: RGB$(target)")
        t = @elapsed result = genetic_search_parallel(target)
        println("  Best seed: $(result.best_seed)")
        println("  Fitness: $(round(result.best_fitness, digits=4))")
        println("  Island: $(result.island_id)")
        println("  Time: $(round(t, digits=3))s")
        println()
    end
end
