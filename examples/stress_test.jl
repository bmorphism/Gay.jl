#!/usr/bin/env julia
#=
🔥 Gay.jl Stress Test - Turn On Those Fans! 🔥

This benchmark is designed to actually stress your system:
- Billion-scale GPU color generation
- Sustained multi-threaded CPU workloads  
- Memory bandwidth saturation
- 60+ second continuous runs

Run with: julia -t auto examples/stress_test.jl

Expected behavior:
- Fans should spin up within 30 seconds
- CPU usage should hit 100% on all cores
- GPU should sustain high utilization
- Memory bandwidth should be saturated
=#

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Gay
using Printf

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

const STRESS_DURATION = 60.0  # seconds per test
const BILLION = 1_000_000_000
const SEED = Gay.GAY_SEED

println("=" ^ 70)
println("🔥 Gay.jl STRESS TEST - Fan Spinner Edition 🔥")
println("=" ^ 70)
println()
println("Julia threads: $(Threads.nthreads())")
println("Duration per test: $(STRESS_DURATION)s")
println("Metal.jl available: $(Gay.HAS_METAL)")
println()

# ═══════════════════════════════════════════════════════════════════════════
# Test 1: Billion-Scale GPU Color Sums
# ═══════════════════════════════════════════════════════════════════════════

function test_billion_gpu()
    println("─" ^ 70)
    println("TEST 1: Billion-Scale Color Sums (GPU if available)")
    println("─" ^ 70)
    
    if Gay.HAS_METAL
        @eval using Metal
        backend = Metal.MetalBackend()
        Gay.set_backend!(backend)
        println("Using Metal GPU backend")
    else
        backend = KernelAbstractions.CPU()
        Gay.set_backend!(backend)
        println("Using CPU backend (no GPU)")
    end
    
    # Warm up
    println("Warming up...")
    Gay.ka_color_sums(1_000_000, SEED; chunk_size=10000)
    
    # Progressive scaling
    scales = [
        (100_000_000, "100M"),
        (1_000_000_000, "1B"),
        (5_000_000_000, "5B"),
        (10_000_000_000, "10B"),
    ]
    
    for (n, label) in scales
        print("  $label colors: ")
        
        r, g, b, elapsed, rate = Gay.ka_color_sums(n, SEED; chunk_size=100000)
        
        rate_billions = rate / 1e9
        @printf("%.2f sec, %.2f B/sec\n", elapsed, rate_billions)
        
        # Verify SPI with fingerprint
        fp = Gay.xor_fingerprint(r, g, b)
        @printf("    RGB sums: (%.2e, %.2e, %.2e), fingerprint: 0x%016x\n", r, g, b, fp)
        
        if elapsed > STRESS_DURATION
            println("    ⏱️  Reached duration limit")
            break
        end
    end
    println()
end

# ═══════════════════════════════════════════════════════════════════════════
# Test 2: Sustained CPU Parallel Hash
# ═══════════════════════════════════════════════════════════════════════════

function test_sustained_cpu()
    println("─" ^ 70)
    println("TEST 2: Sustained CPU Parallel Hash ($(Threads.nthreads()) threads)")
    println("─" ^ 70)
    
    n_per_batch = 100_000_000  # 100M per batch
    total_colors = 0
    total_time = 0.0
    batch = 0
    
    start_time = time()
    
    while (time() - start_time) < STRESS_DURATION
        batch += 1
        r, g, b, elapsed, rate = Gay.ka_parallel_hash(n_per_batch, SEED)
        total_colors += n_per_batch
        total_time += elapsed
        
        rate_millions = rate / 1e6
        cumulative_rate = total_colors / (time() - start_time) / 1e6
        @printf("  Batch %2d: %.2f sec, %.1f M/sec (cumulative: %.1f M/sec)\n", 
                batch, elapsed, rate_millions, cumulative_rate)
    end
    
    elapsed = time() - start_time
    @printf("  Total: %d batches, %.2e colors, %.2f sec, %.2f M/sec average\n",
            batch, Float64(total_colors), elapsed, total_colors / elapsed / 1e6)
    println()
end

# ═══════════════════════════════════════════════════════════════════════════
# Test 3: Memory Bandwidth Stress
# ═══════════════════════════════════════════════════════════════════════════

function test_memory_bandwidth()
    println("─" ^ 70)
    println("TEST 3: Memory Bandwidth Stress")
    println("─" ^ 70)
    
    if Gay.HAS_METAL
        @eval using Metal
        backend = Metal.MetalBackend()
    else
        backend = KernelAbstractions.CPU()
    end
    
    # Large arrays to stress memory bandwidth
    n = 100_000_000  # 100M floats = 400MB per array
    
    println("  Allocating 2x $(n ÷ 1_000_000)M float arrays ($(2 * n * 4 ÷ 1_000_000) MB total)...")
    
    input = KernelAbstractions.zeros(backend, Float32, n)
    output = KernelAbstractions.zeros(backend, Float32, n)
    
    # Fill input with pattern
    input_cpu = rand(Float32, n)
    if backend isa KernelAbstractions.CPU
        copyto!(input, input_cpu)
    else
        copyto!(input, input_cpu)
    end
    
    println("  Running bandwidth stress for $(STRESS_DURATION)s...")
    
    iterations = 0
    total_bytes = 0
    start_time = time()
    
    while (time() - start_time) < STRESS_DURATION
        Gay.ka_bandwidth_stress!(output, input, SEED; backend=backend)
        iterations += 1
        total_bytes += 2 * n * sizeof(Float32)  # Read + Write
        
        if iterations % 10 == 0
            elapsed = time() - start_time
            bandwidth = total_bytes / elapsed / 1e9
            @printf("    Iteration %d: %.2f GB/s\n", iterations, bandwidth)
        end
    end
    
    elapsed = time() - start_time
    bandwidth = total_bytes / elapsed / 1e9
    @printf("  Total: %d iterations, %.2f GB transferred, %.2f GB/s\n",
            iterations, total_bytes / 1e9, bandwidth)
    println()
end

# ═══════════════════════════════════════════════════════════════════════════
# Test 4: Mortal Computation Churn
# ═══════════════════════════════════════════════════════════════════════════

function test_mortal_churn()
    println("─" ^ 70)
    println("TEST 4: Mortal Computation Churn ($(Threads.nthreads()) threads)")
    println("─" ^ 70)
    
    n_mortals = 10000
    lifetime = 1000
    
    println("  Spawning mortals: $n_mortals per round, $lifetime steps each")
    println("  Running for $(STRESS_DURATION)s...")
    
    result = Gay.stress_mortal_churn(n_mortals, lifetime, STRESS_DURATION; seed=SEED)
    
    @printf("  Rounds: %d\n", result.rounds)
    @printf("  Total mortals: %d\n", result.total_mortals)
    @printf("  Total steps: %.2e\n", Float64(result.total_steps))
    @printf("  Elapsed: %.2f sec\n", result.elapsed)
    @printf("  Mortals/sec: %.2f\n", result.mortals_per_second)
    @printf("  Steps/sec: %.2e\n", result.steps_per_second)
    println()
end

# ═══════════════════════════════════════════════════════════════════════════
# Test 5: Immortal Marathon
# ═══════════════════════════════════════════════════════════════════════════

function test_immortal_marathon()
    println("─" ^ 70)
    println("TEST 5: Immortal Computation Marathon ($(Threads.nthreads()) threads)")
    println("─" ^ 70)
    
    n_immortals = 1000
    
    println("  Running $n_immortals immortals for $(STRESS_DURATION)s...")
    
    result = Gay.stress_immortal_marathon(n_immortals, STRESS_DURATION; seed=SEED)
    
    @printf("  Max epoch reached: %d\n", result.max_epoch)
    @printf("  Total epochs: %d\n", result.total_epochs)
    @printf("  Elapsed: %.2f sec\n", result.elapsed)
    @printf("  Epochs/sec: %.2f\n", result.epochs_per_second)
    @printf("  Total accumulated: %.2e\n", result.total_accumulated)
    println()
end

# ═══════════════════════════════════════════════════════════════════════════
# Test 6: Combined GPU + CPU Stress
# ═══════════════════════════════════════════════════════════════════════════

function test_combined_stress()
    println("─" ^ 70)
    println("TEST 6: Combined GPU + CPU Stress (Maximum Heat!)")
    println("─" ^ 70)
    
    if !Gay.HAS_METAL
        println("  Skipping - no GPU available")
        return
    end
    
    @eval using Metal
    gpu_backend = Metal.MetalBackend()
    
    println("  Running GPU and CPU concurrently for $(STRESS_DURATION)s...")
    
    gpu_colors = Threads.Atomic{Int}(0)
    cpu_colors = Threads.Atomic{Int}(0)
    
    start_time = time()
    
    # GPU task
    gpu_task = Threads.@spawn begin
        while (time() - start_time) < STRESS_DURATION
            r, g, b, _, _ = Gay.ka_color_sums(500_000_000, SEED; 
                                               chunk_size=50000, 
                                               backend=gpu_backend)
            Threads.atomic_add!(gpu_colors, 500_000_000)
        end
    end
    
    # CPU task (uses remaining threads)
    cpu_task = Threads.@spawn begin
        while (time() - start_time) < STRESS_DURATION
            r, g, b, _, _ = Gay.ka_parallel_hash(50_000_000, SEED)
            Threads.atomic_add!(cpu_colors, 50_000_000)
        end
    end
    
    wait(gpu_task)
    wait(cpu_task)
    
    elapsed = time() - start_time
    total = gpu_colors[] + cpu_colors[]
    
    @printf("  GPU colors: %.2e (%.2f B/sec)\n", 
            Float64(gpu_colors[]), gpu_colors[] / elapsed / 1e9)
    @printf("  CPU colors: %.2e (%.2f M/sec)\n", 
            Float64(cpu_colors[]), cpu_colors[] / elapsed / 1e6)
    @printf("  Combined: %.2e colors in %.2f sec\n", Float64(total), elapsed)
    println()
end

# ═══════════════════════════════════════════════════════════════════════════
# Test 7: Extreme Memory Allocation Churn
# ═══════════════════════════════════════════════════════════════════════════

function test_allocation_churn()
    println("─" ^ 70)
    println("TEST 7: Allocation Churn (GC stress)")
    println("─" ^ 70)
    
    println("  Allocating and discarding color matrices for $(STRESS_DURATION)s...")
    
    n = 10_000_000  # 10M colors per allocation
    allocations = 0
    total_colors = 0
    
    start_time = time()
    
    while (time() - start_time) < STRESS_DURATION
        colors = zeros(Float32, n, 3)
        Gay.ka_colors!(colors, SEED; backend=KernelAbstractions.CPU())
        total_colors += n
        allocations += 1
        
        # Force GC periodically
        if allocations % 5 == 0
            GC.gc(false)
        end
        
        if allocations % 10 == 0
            elapsed = time() - start_time
            @printf("    Allocation %d: %.2f GB allocated, %.2f M colors/sec\n",
                    allocations, allocations * n * 3 * 4 / 1e9, total_colors / elapsed / 1e6)
        end
    end
    
    elapsed = time() - start_time
    @printf("  Total: %d allocations, %.2e colors, %.2f sec\n",
            allocations, Float64(total_colors), elapsed)
    println()
end

# ═══════════════════════════════════════════════════════════════════════════
# Run All Tests
# ═══════════════════════════════════════════════════════════════════════════

function main()
    start = time()
    
    try
        test_billion_gpu()
        test_sustained_cpu()
        test_memory_bandwidth()
        test_mortal_churn()
        test_immortal_marathon()
        test_combined_stress()
        test_allocation_churn()
    catch e
        println("Error: $e")
        rethrow()
    end
    
    total_time = time() - start
    
    println("=" ^ 70)
    @printf("🔥 STRESS TEST COMPLETE: %.1f minutes total\n", total_time / 60)
    println("=" ^ 70)
    println()
    println("If your fans didn't spin up, try:")
    println("  1. Run with more threads: julia -t $(Sys.CPU_THREADS) examples/stress_test.jl")
    println("  2. Increase STRESS_DURATION at top of file")
    println("  3. Install Metal.jl for GPU acceleration")
end

main()
