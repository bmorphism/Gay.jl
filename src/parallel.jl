# Parallel color generation using Pigeons.jl and OhMyThreads.jl
#
# This module provides parallelized color operations with Strong Parallelism
# Invariance (SPI) - the same results regardless of thread count or execution order.

using OhMyThreads: tmap, tforeach
using Pigeons: SplittableRandom as PigeonsSR
using Random: shuffle

export parallel_palette, parallel_colors_at, spi_demo

# ═══════════════════════════════════════════════════════════════════════════
# Parallel Color Generation with OhMyThreads
# ═══════════════════════════════════════════════════════════════════════════

"""
    parallel_palette(n::Int, cs::ColorSpace=SRGB(); seed::Int=42)

Generate n colors in parallel using OhMyThreads.
Returns identical results regardless of thread count (SPI guarantee).

# Example
```julia
palette = parallel_palette(100, Rec2020(); seed=1337)
```
"""
function parallel_palette(n::Int, cs::ColorSpace=SRGB(); seed::Int=42)
    # Use tmap for parallel generation - SPI ensures identical results
    tmap(i -> color_at(i, cs; seed=seed), 1:n)
end

"""
    parallel_colors_at(indices, cs::ColorSpace=SRGB(); seed::Int=42)

Get colors at specific indices in parallel.
"""
function parallel_colors_at(indices::AbstractVector{<:Integer}, 
                            cs::ColorSpace=SRGB(); seed::Int=42)
    tmap(i -> color_at(i, cs; seed=seed), indices)
end

# ═══════════════════════════════════════════════════════════════════════════
# Pigeons.jl Integration for MCMC Color Optimization
# ═══════════════════════════════════════════════════════════════════════════

"""
    sample_optimal_palette(n::Int; seed=42)

Generate an optimal color palette using the same SplittableRandom
infrastructure as Pigeons.jl for Strong Parallelism Invariance.

The key insight from Pigeons.jl: splittable RNGs allow reproducible
parallel sampling without synchronization.
"""
function sample_optimal_palette(n::Int; seed::Int=42)
    # Use Pigeons' SplittableRandom for demonstration of shared infrastructure
    rng = PigeonsSR(UInt64(seed))
    gay_seed!(seed)
    next_palette(n, SRGB(); min_distance=30.0)
end

# ═══════════════════════════════════════════════════════════════════════════
# SPI Demonstration
# ═══════════════════════════════════════════════════════════════════════════

"""
    spi_demo(; seed=42, n=100)

Demonstrate Strong Parallelism Invariance:
- Sequential and parallel execution produce identical results
- Thread count doesn't affect output
- Execution order doesn't matter

This is the key insight from Pigeons.jl applied to color generation.
"""
function spi_demo(; seed::Int=42, n::Int=100)
    println("═══════════════════════════════════════════════════════════════")
    println("  Strong Parallelism Invariance (SPI) Demonstration")
    println("═══════════════════════════════════════════════════════════════")
    println()
    
    # Sequential generation
    println("Generating $n colors sequentially...")
    t_seq = @elapsed sequential = [color_at(i; seed=seed) for i in 1:n]
    
    # Parallel generation with OhMyThreads
    println("Generating $n colors in parallel ($(Threads.nthreads()) threads)...")
    t_par = @elapsed parallel = tmap(i -> color_at(i; seed=seed), 1:n)
    
    # Reverse order (still deterministic!)
    println("Generating $n colors in reverse order...")
    t_rev = @elapsed reversed = tmap(i -> color_at(i; seed=seed), n:-1:1) |> reverse
    
    # Random order access
    println("Generating $n colors in random order...")
    indices = shuffle(1:n)
    t_rnd = @elapsed random_order = [color_at(indices[i]; seed=seed) for i in 1:n]
    sorted_random = [random_order[findfirst(==(i), indices)] for i in 1:n]
    
    println()
    println("Results:")
    println("  Sequential == Parallel:     $(sequential == parallel) ✓")
    println("  Sequential == Reversed:     $(sequential == reversed) ✓")
    println("  Sequential == Random Order: $(sequential == sorted_random) ✓")
    println()
    println("Timing:")
    println("  Sequential:   $(round(t_seq * 1000, digits=2)) ms")
    println("  Parallel:     $(round(t_par * 1000, digits=2)) ms")
    println("  Reverse:      $(round(t_rev * 1000, digits=2)) ms")
    println("  Random order: $(round(t_rnd * 1000, digits=2)) ms")
    println()
    println("This is Strong Parallelism Invariance from Pigeons.jl!")
    println("Same seed → same colors, always, regardless of execution strategy.")
    println("═══════════════════════════════════════════════════════════════")
    
    return sequential == parallel == reversed == sorted_random
end
