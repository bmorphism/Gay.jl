#!/usr/bin/env julia
"""
    two_poisson_demo.jl

Comprehensive demonstration of 2D Poisson processes with chromatic identity.

Examples:
1. Constant intensity process
2. Gaussian intensity process
3. Thinning and superposition
4. SPI verification across parallel workers
"""

using Random, Base.Threads
push!(LOAD_PATH, "/Users/bob/ies/rio/Gay.jl/src")

include("/Users/bob/ies/rio/Gay.jl/src/two_poisson.jl")
using .TwoPoisson

# ═══════════════════════════════════════════════════════════════════════════
# DEMO 1: Constant Intensity
# ═══════════════════════════════════════════════════════════════════════════

function world_constant_intensity()
    println("\n" * "="^80)
    println("DEMO 1: Constant Intensity Poisson Process")
    println("="^80)
    
    # Domain: [0,1]² × [0,1]
    domain = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
    
    # Constant intensity λ = 10
    intensity = TwoPoisson.ConstantIntensity(10.0)
    
    # Create process
    process = create_poisson_2d(domain; intensity=intensity, seed=0x42)
    
    println("Domain: [0,1]² × [0,1]")
    println("Intensity: λ(x,y,t) = 10.0 (constant)")
    println("Expected points: 10.0 × 1.0 = 10.0")
    
    # Sample points
    println("\nSampling points...")
    sample_points!(process; max_points=50)
    
    process_result = result(process)
    print_process(process_result)
    
    return process_result
end

# ═══════════════════════════════════════════════════════════════════════════
# DEMO 2: Gaussian Intensity
# ═══════════════════════════════════════════════════════════════════════════

function world_gaussian_intensity()
    println("\n" * "="^80)
    println("DEMO 2: Gaussian Intensity Poisson Process")
    println("="^80)
    
    domain = (-1.0, 1.0, -1.0, 1.0, 0.0, 1.0)
    
    # Gaussian centered at origin: λ₀ = 5, σ = 0.5
    intensity = TwoPoisson.GaussianIntensity(5.0, 0.5)
    
    process = create_poisson_2d(domain; intensity=intensity, seed=0xCAFEBABE)
    
    println("Domain: [-1,1]² × [0,1]")
    println("Intensity: λ(x,y,t) = 5.0 * exp(-(x²+y²+t²)/0.5)")
    println("Intensity is peaked near origin, decays smoothly")
    
    println("\nSampling points...")
    sample_points!(process; max_points=100)
    
    process_result = result(process)
    print_process(process_result)
    
    return process_result
end

# ═══════════════════════════════════════════════════════════════════════════
# DEMO 3: Thinning & Superposition
# ═══════════════════════════════════════════════════════════════════════════

function world_thinning_superposition()
    println("\n" * "="^80)
    println("DEMO 3: Thinning and Superposition")
    println("="^80)
    
    domain = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
    intensity = TwoPoisson.ConstantIntensity(20.0)
    
    # Create original process
    original = create_poisson_2d(domain; intensity=intensity, seed=0x12345)
    println("Creating original process with λ = 20.0")
    sample_points!(original)
    println("Original points sampled: $(original.num_points)")
    
    # Thin to 50%
    thinned_50 = thin_process(original, 0.5)
    println("\nThinned to 50%: $(thinned_50.num_points) points (expected ~$(Int(original.num_points/2)))")
    
    # Thin to 30%
    thinned_30 = thin_process(original, 0.3)
    println("Thinned to 30%: $(thinned_30.num_points) points (expected ~$(Int(original.num_points*0.3)))")
    
    # Superpose the thinned processes
    superposed = superpose_processes([thinned_50, thinned_30])
    println("\nSuperposed 50% + 30% thinnings: $(superposed.num_points) points")
    
    # Verify they combine correctly
    expected_total = thinned_50.num_points + thinned_30.num_points
    println("Expected from sum: $expected_total")
    
    superposed_result = result(superposed)
    print_process(superposed_result)
    
    return (original=result(original), thinned_50=result(thinned_50), 
            thinned_30=result(thinned_30), superposed=superposed_result)
end

# ═══════════════════════════════════════════════════════════════════════════
# DEMO 4: Parallel SPI Verification
# ═══════════════════════════════════════════════════════════════════════════

function world_parallel_spi()
    println("\n" * "="^80)
    println("DEMO 4: Parallel SPI Verification")
    println("="^80)
    
    domain = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
    intensity = TwoPoisson.ConstantIntensity(15.0)
    
    num_threads = Threads.nthreads()
    println("Running $(num_threads) parallel processes with same seed...")
    
    # Run parallel samples
    results = Vector{TwoPoisson.PointEventResult}(undef, num_threads)
    
    Threads.@threads for thread_id in 1:num_threads
        process = create_poisson_2d(
            domain;
            intensity=intensity,
            seed=0xDEADBEEF,      # Same seed for all
            thread_id=UInt64(thread_id - 1)
        )
        
        sample_points!(process)
        results[thread_id] = result(process)
    end
    
    # Analyze results
    println("\nResults from $(num_threads) workers:")
    for (i, r) in enumerate(results)
        println("  Thread $i: $(r.num_points) points, SPI=$(r.spi_hash)")
    end
    
    # Verify SPI
    spi_valid = verify_spi(results)
    println("\n✓ SPI Verification: $(spi_valid ? "PASSED" : "FAILED")")
    
    if spi_valid
        println("  All parallel processes produced deterministic colored sequences!")
    else
        println("  ERROR: Different SPI hashes detected!")
    end
    
    return results
end

# ═══════════════════════════════════════════════════════════════════════════
# DEMO 5: Chromatic Analysis
# ═══════════════════════════════════════════════════════════════════════════

function world_chromatic_analysis()
    println("\n" * "="^80)
    println("DEMO 5: Chromatic Identity Analysis")
    println("="^80)
    
    domain = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
    intensity = TwoPoisson.ConstantIntensity(25.0)
    
    process = create_poisson_2d(domain; intensity=intensity, seed=0xBEEFCAFE)
    println("Creating process with λ = 25.0")
    sample_points!(process)
    
    colors = point_colors(process)
    println("Points sampled: $(length(colors))")
    
    # Analyze color distribution
    if !isempty(colors)
        rs = [c.r for c in colors]
        gs = [c.g for c in colors]
        bs = [c.b for c in colors]
        
        println("\nRed channel:")
        println("  Min: $(round(minimum(rs); digits=3))")
        println("  Max: $(round(maximum(rs); digits=3))")
        println("  Mean: $(round(mean(rs); digits=3))")
        
        println("\nGreen channel:")
        println("  Min: $(round(minimum(gs); digits=3))")
        println("  Max: $(round(maximum(gs); digits=3))")
        println("  Mean: $(round(mean(gs); digits=3))")
        
        println("\nBlue channel:")
        println("  Min: $(round(minimum(bs); digits=3))")
        println("  Max: $(round(maximum(bs); digits=3))")
        println("  Mean: $(round(mean(bs); digits=3))")
        
        println("\nFirst 5 colors:")
        for i in 1:min(5, length(colors))
            c = colors[i]
            println("  $i: RGB($(round(c.r;digits=3)), $(round(c.g;digits=3)), $(round(c.b;digits=3)))")
        end
    end
    
    process_result = result(process)
    return process_result
end

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

function main()
    println("\n" * "="^80)
    println("2D POISSON PROCESS DEMONSTRATIONS")
    println("With Chromatic Identity and SPI Verification")
    println("="^80)
    
    # Run demos
    result1 = world_constant_intensity()
    result2 = world_gaussian_intensity()
    results3 = world_thinning_superposition()
    results4 = world_parallel_spi()
    result5 = world_chromatic_analysis()
    
    # Summary
    println("\n" * "="^80)
    println("SUMMARY")
    println("="^80)
    println("\n✓ All demonstrations completed successfully!")
    println("\nKey features demonstrated:")
    println("  1. Constant and Gaussian intensity functions")
    println("  2. Rejection sampling for point generation")
    println("  3. Chromatic identity (deterministic colors)")
    println("  4. Process thinning (independent point removal)")
    println("  5. Process superposition (merging)")
    println("  6. SPI verification across parallel workers")
    println("  7. Chromatic analysis and statistics")
    println("\n" * "="^80 * "\n")
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
