#!/usr/bin/env julia
"""
    eg_walker_demo.jl

Demonstration of gay-eg-walker with SPI verification across parallel workers.

Features:
- Euclidean-guided random walks with chromatic identity
- Deterministic color generation via gay_seed
- SPI (Strong Parallelism Invariance) verification
- Energy tracking for optimization
"""

using Graphs, Colors, Random, Base.Threads
push!(LOAD_PATH, "/Users/bob/ies/rio/Gay.jl/src")

# Include the walker module
include("/Users/bob/ies/rio/Gay.jl/src/gay_eg_walker.jl")
using .GayEGWalker

# ═══════════════════════════════════════════════════════════════════════════
# DEMO: Create test graph
# ═══════════════════════════════════════════════════════════════════════════

function create_grid_graph(n::Int64)::Tuple{SimpleGraph, Vector{Tuple{Float64, Float64}}}
    """Create an n×n grid graph with Euclidean positions"""
    g = grid_graph([n, n])
    
    # Compute positions
    positions = Tuple{Float64, Float64}[]
    for i in 1:nv(g)
        row = div(i - 1, n)
        col = mod(i - 1, n)
        push!(positions, (Float64(col), Float64(row)))
    end
    
    g, positions
end

# ═══════════════════════════════════════════════════════════════════════════
# DEMO: Single walk
# ═══════════════════════════════════════════════════════════════════════════

function world_single_walk()
    println("\n" * "="^80)
    println("DEMO 1: Single EG-Walker")
    println("="^80)
    
    # Create 5×5 grid graph
    g, positions = create_grid_graph(5)
    println("Created 5×5 grid graph (25 vertices)")
    
    # Create walker from top-left to bottom-right
    start_vertex = 1
    target_vertex = nv(g)
    
    walker = create_walker(
        g, positions;
        seed=0x42,
        thread_id=0,
        start=start_vertex,
        target=target_vertex
    )
    
    println("Walker created:")
    println("  Start: vertex $start_vertex at $(positions[start_vertex])")
    println("  Target: vertex $target_vertex at $(positions[target_vertex])")
    
    # Execute walk
    println("\nExecuting walk...")
    reached = walk!(walker, 100; target_prob=0.3)
    
    # Show result
    walk_result = result(walker)
    print_walk(walk_result)
    
    return walk_result
end

# ═══════════════════════════════════════════════════════════════════════════
# DEMO: Parallel walks with SPI verification
# ═══════════════════════════════════════════════════════════════════════════

function world_parallel_spi()
    println("\n" * "="^80)
    println("DEMO 2: Parallel SPI Verification")
    println("="^80)
    
    g, positions = create_grid_graph(6)
    num_parallel = Threads.nthreads()
    
    println("Created 6×6 grid graph (36 vertices)")
    println("Running $(num_parallel) parallel walks with same seed...")
    
    # Run parallel walks with same seed but different thread_ids
    results = Vector{GayEGWalker.WalkResult}(undef, num_parallel)
    
    Threads.@threads for thread_id in 1:num_parallel
        walker = create_walker(
            g, positions;
            seed=0xDEADBEEF,  # Same seed for all
            thread_id=UInt64(thread_id - 1),
            start=1,
            target=nv(g)
        )
        
        walk!(walker, 50; target_prob=0.2)
        results[thread_id] = result(walker)
    end
    
    println("\nResults from $(num_parallel) parallel walks:")
    for (i, r) in enumerate(results)
        println("  Thread $i: $(length(r.path)) vertices, energy=$(round(r.total_energy; digits=2)), SPI=$(r.spi_hash)")
    end
    
    # Verify SPI
    spi_valid = verify_spi(results)
    println("\n✓ SPI Verification: $(spi_valid ? "PASSED" : "FAILED")")
    
    if spi_valid
        println("  All parallel walks produced identical colored sequences!")
    end
    
    return results
end

# ═══════════════════════════════════════════════════════════════════════════
# DEMO: Energy-aware analysis
# ═══════════════════════════════════════════════════════════════════════════

function world_energy_analysis()
    println("\n" * "="^80)
    println("DEMO 3: Energy-Aware Path Analysis")
    println("="^80)
    
    g, positions = create_grid_graph(7)
    num_walks = 10
    
    println("Created 7×7 grid (49 vertices)")
    println("Running $(num_walks) walks to analyze energy patterns...\n")
    
    # Run multiple walks
    results = Vector{GayEGWalker.WalkResult}()
    
    for i in 1:num_walks
        walker = create_walker(
            g, positions;
            seed=UInt64(0x12345678 + i),
            thread_id=UInt64(i - 1),
            start=1,
            target=nv(g)
        )
        
        walk!(walker, 80; target_prob=0.2)
        push!(results, result(walker))
    end
    
    # Analyze
    path_lengths = [r.steps for r in results]
    total_energies = [r.total_energy for r in results]
    
    println("Path length statistics:")
    println("  Min: $(minimum(path_lengths))")
    println("  Max: $(maximum(path_lengths))")
    println("  Mean: $(round(mean(path_lengths); digits=2))")
    
    println("\nTotal energy statistics:")
    println("  Min: $(round(minimum(total_energies); digits=4))")
    println("  Max: $(round(maximum(total_energies); digits=4))")
    println("  Mean: $(round(mean(total_energies); digits=4))")
    
    # Target success rate
    success_rate = sum(r.reached_target for r in results) / num_walks
    println("\nTarget success rate: $(round(success_rate * 100; digits=1))%")
    
    return results
end

# ═══════════════════════════════════════════════════════════════════════════
# DEMO: Color distribution
# ═══════════════════════════════════════════════════════════════════════════

function world_color_distribution()
    println("\n" * "="^80)
    println("DEMO 4: Chromatic Distribution Analysis")
    println("="^80)
    
    g, positions = create_grid_graph(5)
    
    println("Created 5×5 grid (25 vertices)")
    println("Analyzing color distribution across walk steps...\n")
    
    # Single long walk
    walker = create_walker(
        g, positions;
        seed=0xCAFEBABE,
        thread_id=0,
        start=1
    )
    
    walk!(walker, 100)
    walk_result = result(walker)
    
    colors = walk_result.colors
    println("Walk generated $(length(colors)) colors")
    
    # Analyze RGB distribution
    r_vals = [c.r for c in colors]
    g_vals = [c.g for c in colors]
    b_vals = [c.b for c in colors]
    
    println("\nRed channel:")
    println("  Min: $(round(minimum(r_vals); digits=3))")
    println("  Max: $(round(maximum(r_vals); digits=3))")
    println("  Mean: $(round(mean(r_vals); digits=3))")
    
    println("\nGreen channel:")
    println("  Min: $(round(minimum(g_vals); digits=3))")
    println("  Max: $(round(maximum(g_vals); digits=3))")
    println("  Mean: $(round(mean(g_vals); digits=3))")
    
    println("\nBlue channel:")
    println("  Min: $(round(minimum(b_vals); digits=3))")
    println("  Max: $(round(maximum(b_vals); digits=3))")
    println("  Mean: $(round(mean(b_vals); digits=3))")
    
    # Show first few colors
    println("\nFirst 10 colors:")
    for (i, color) in enumerate(colors[1:min(10, length(colors))])
        println("  $i: RGB($(round(color.r; digits=3)), $(round(color.g; digits=3)), $(round(color.b; digits=3)))")
    end
    
    return walk_result
end

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

function main()
    println("\n" * "="^80)
    println("GAY-EG-WALKER DEMONSTRATION")
    println("Euclidean-guided walks with chromatic identity & SPI")
    println("="^80)
    
    # Run demonstrations
    result1 = world_single_walk()
    results2 = world_parallel_spi()
    results3 = world_energy_analysis()
    result4 = world_color_distribution()
    
    # Summary
    println("\n" * "="^80)
    println("SUMMARY")
    println("="^80)
    println("\n✓ All demonstrations completed successfully!")
    println("\nKey features demonstrated:")
    println("  1. Euclidean-guided step selection via inverse distance weighting")
    println("  2. Deterministic color generation via gay_seed (SplitMix64)")
    println("  3. SPI verification across parallel workers")
    println("  4. Energy tracking and path optimization")
    println("  5. Chromatic identity for each step")
    println("\n" * "="^80 * "\n")
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
