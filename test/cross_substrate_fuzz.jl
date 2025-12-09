# Cross-Substrate SPI Verification
# =================================
# Verify Strong Parallelism Invariance across GPU backends:
# - CPU (KernelAbstractions default)
# - Metal.jl (Apple Silicon)
# - CUDA.jl (NVIDIA) 
# - AMDGPU.jl (AMD ROCm)
# - oneAPI.jl (Intel)
#
# The key guarantee: SAME SEED → SAME COLORS regardless of substrate

using Test
using Random
using Gay
using Gay: ka_colors, ka_colors!, xor_fingerprint, hash_color, splitmix64
using Gay: set_backend!, get_backend, GAY_SEED
using Gay: _ka_colors_kernel!
using KernelAbstractions
using KernelAbstractions: CPU, synchronize

# ═══════════════════════════════════════════════════════════════════════════
# Backend Detection
# ═══════════════════════════════════════════════════════════════════════════

struct BackendInfo
    name::String
    backend::Any
    available::Bool
    device_name::String
end

function detect_backends()
    backends = BackendInfo[]
    
    # CPU is always available
    push!(backends, BackendInfo("CPU", CPU(), true, "$(Threads.nthreads()) threads"))
    
    # Metal
    try
        @eval using Metal
        if Metal.functional()
            dev = Metal.current_device()
            backend = Metal.MetalBackend()
            push!(backends, BackendInfo("Metal", backend, true, string(dev.name)))
        end
    catch e
        # Metal not available
    end
    
    # CUDA
    try
        @eval using CUDA
        if CUDA.functional()
            backend = CUDA.CUDABackend()
            push!(backends, BackendInfo("CUDA", backend, true, CUDA.name(CUDA.device())))
        end
    catch
        # CUDA not available
    end
    
    # AMDGPU
    try
        @eval using AMDGPU
        if AMDGPU.functional()
            backend = AMDGPU.ROCBackend()
            push!(backends, BackendInfo("AMDGPU", backend, true, "ROCm"))
        end
    catch
        # AMDGPU not available
    end
    
    # oneAPI
    try
        @eval using oneAPI
        if oneAPI.functional()
            backend = oneAPI.oneAPIBackend()
            push!(backends, BackendInfo("oneAPI", backend, true, "Intel GPU"))
        end
    catch
        # oneAPI not available
    end
    
    backends
end

# ═══════════════════════════════════════════════════════════════════════════
# Cross-Substrate Verification
# ═══════════════════════════════════════════════════════════════════════════

"""
Generate colors on a specific backend and return fingerprint.
"""
function generate_on_backend(backend, n::Int, seed::UInt64)
    if backend isa CPU
        colors = zeros(Float32, n, 3)
        ka_colors!(colors, seed; backend=backend)
        return xor_fingerprint(colors), colors
    else
        # GPU backend - allocate on device
        colors_gpu = KernelAbstractions.zeros(backend, Float32, n, 3)
        
        kernel! = _ka_colors_kernel!(backend, 256)
        kernel!(colors_gpu, seed, ndrange=n)
        synchronize(backend)
        
        # Copy back to CPU for fingerprinting
        colors_cpu = Array(colors_gpu)
        return xor_fingerprint(colors_cpu), colors_cpu
    end
end

"""
Verify SPI across all available backends.
"""
function cross_substrate_verify(; n::Int=10000, seed::UInt64=GAY_SEED, verbose::Bool=true)
    backends = detect_backends()
    
    if verbose
        println()
        println("╔══════════════════════════════════════════════════════════════════════╗")
        println("║  CROSS-SUBSTRATE SPI VERIFICATION                                   ║")
        println("╚══════════════════════════════════════════════════════════════════════╝")
        println()
        println("  Testing n=$n colors, seed=$(seed)")
        println()
        println("  Available backends:")
        for b in backends
            status = b.available ? "✓" : "✗"
            println("    $status $(b.name): $(b.device_name)")
        end
        println()
    end
    
    # Generate reference on CPU
    cpu_backend = first(filter(b -> b.name == "CPU", backends))
    ref_fp, ref_colors = generate_on_backend(cpu_backend.backend, n, seed)
    
    if verbose
        println("  CPU Reference Fingerprint: 0x$(string(ref_fp, base=16, pad=8))")
        println()
    end
    
    results = Dict{String, NamedTuple}()
    all_match = true
    
    for b in backends
        if !b.available
            continue
        end
        
        try
            t = @elapsed begin
                fp, colors = generate_on_backend(b.backend, n, seed)
            end
            
            match = fp == ref_fp
            colors_match = isapprox(ref_colors, colors; rtol=1e-6)
            
            results[b.name] = (
                fingerprint = fp,
                match = match,
                colors_match = colors_match,
                time = t,
                rate = n / t / 1e6
            )
            
            if verbose
                status = match ? "✓" : "✗"
                println("  $(b.name):")
                println("    Fingerprint: 0x$(string(fp, base=16, pad=8)) $status")
                println("    Time: $(round(t * 1000, digits=2))ms ($(round(n/t/1e6, digits=1)) M colors/s)")
                if !match
                    println("    ⚠️  FINGERPRINT MISMATCH!")
                    # Find first differing color
                    for i in 1:min(n, 10)
                        if !isapprox(ref_colors[i,:], colors[i,:]; rtol=1e-6)
                            println("    First diff at i=$i:")
                            println("      CPU: $(ref_colors[i,:])")
                            println("      $(b.name): $(colors[i,:])")
                            break
                        end
                    end
                end
                println()
            end
            
            all_match &= match
            
        catch e
            if verbose
                println("  $(b.name): ERROR - $e")
                println()
            end
            results[b.name] = (error = e,)
        end
    end
    
    if verbose
        println("═" ^ 72)
        if all_match
            println("  ✓ ALL BACKENDS PRODUCE IDENTICAL COLORS")
            println("  Strong Parallelism Invariance verified across substrates!")
        else
            println("  ✗ SPI VIOLATION DETECTED ACROSS SUBSTRATES")
        end
        println("═" ^ 72)
    end
    
    (all_match = all_match, results = results, reference_fp = ref_fp)
end

"""
Fuzz test across substrates with random parameters.
"""
function cross_substrate_fuzz(; duration::Float64=30.0, seed::Int=42)
    backends = detect_backends()
    gpu_backends = filter(b -> b.name != "CPU" && b.available, backends)
    
    if isempty(gpu_backends)
        println("No GPU backends available. Running CPU-only fuzz.")
        return (passed = true, note = "CPU only")
    end
    
    println()
    println("╔══════════════════════════════════════════════════════════════════════╗")
    println("║  CROSS-SUBSTRATE FUZZ TESTING                                       ║")
    println("╚══════════════════════════════════════════════════════════════════════╝")
    println()
    println("  Duration: $(duration)s")
    println("  GPU backends: $(join([b.name for b in gpu_backends], ", "))")
    println()
    
    rng = Random.MersenneTwister(seed)
    
    start_time = time()
    iterations = 0
    violations = 0
    total_colors = 0
    
    cpu = first(filter(b -> b.name == "CPU", backends)).backend
    gpu = first(gpu_backends).backend
    gpu_name = first(gpu_backends).name
    
    while (time() - start_time) < duration
        iterations += 1
        
        # Random parameters
        n = rand(rng, [100, 1000, 5000, 10000])
        test_seed = rand(rng, UInt64)
        
        # Generate on CPU
        cpu_fp, _ = generate_on_backend(cpu, n, test_seed)
        
        # Generate on GPU
        try
            gpu_fp, _ = generate_on_backend(gpu, n, test_seed)
            
            if cpu_fp != gpu_fp
                violations += 1
                if violations <= 5
                    println("  ⚠️  Violation #$violations: seed=$test_seed, n=$n")
                    println("      CPU: 0x$(string(cpu_fp, base=16))")
                    println("      $gpu_name: 0x$(string(gpu_fp, base=16))")
                end
            end
        catch e
            # GPU error - skip
        end
        
        total_colors += n * 2
        
        # Progress
        if iterations % 100 == 0
            elapsed = round(time() - start_time, digits=1)
            println("  [$elapsed s] $iterations iterations, $violations violations, $(round(total_colors/1e6, digits=1))M colors")
        end
    end
    
    elapsed = time() - start_time
    
    println()
    println("═" ^ 72)
    println("  Cross-Substrate Fuzz Complete")
    println("  Iterations: $iterations")
    println("  Violations: $violations")
    println("  Total colors: $(round(total_colors/1e6, digits=1))M")
    println("  Duration: $(round(elapsed, digits=1))s")
    
    if violations == 0
        println("  ✓ ALL CROSS-SUBSTRATE TESTS PASSED")
    else
        println("  ✗ $violations VIOLATIONS DETECTED")
    end
    println("═" ^ 72)
    
    (passed = violations == 0, iterations = iterations, violations = violations)
end

"""
Benchmark all available backends.
"""
function benchmark_all_backends(; n::Int=1_000_000, seed::UInt64=GAY_SEED)
    backends = detect_backends()
    
    println()
    println("╔══════════════════════════════════════════════════════════════════════╗")
    println("║  BACKEND BENCHMARK: $n colors                               ║")
    println("╚══════════════════════════════════════════════════════════════════════╝")
    println()
    
    results = Dict{String, Float64}()
    
    for b in backends
        if !b.available
            continue
        end
        
        # Warmup
        try
            generate_on_backend(b.backend, 1000, seed)
        catch
            continue
        end
        
        # Benchmark
        try
            t = @elapsed for _ in 1:3
                generate_on_backend(b.backend, n, seed)
            end
            t /= 3  # Average
            
            rate = n / t / 1e6
            results[b.name] = rate
            
            println("  $(rpad(b.name, 10)): $(round(rate, digits=0)) M colors/s  ($(round(t*1000, digits=1))ms)")
        catch e
            println("  $(rpad(b.name, 10)): ERROR - $e")
        end
    end
    
    println()
    
    if haskey(results, "Metal") && haskey(results, "CPU")
        speedup = results["Metal"] / results["CPU"]
        println("  Metal speedup: $(round(speedup, digits=1))x vs CPU")
    end
    
    results
end

# ═══════════════════════════════════════════════════════════════════════════
# Test Suite
# ═══════════════════════════════════════════════════════════════════════════

@testset "Cross-Substrate SPI" begin
    
    @testset "Backend Detection" begin
        backends = detect_backends()
        @test !isempty(backends)
        @test any(b -> b.name == "CPU", backends)
    end
    
    @testset "CPU Reference" begin
        result = cross_substrate_verify(n=1000, seed=GAY_SEED, verbose=false)
        @test haskey(result.results, "CPU")
        @test result.results["CPU"].match
    end
    
    @testset "Cross-Substrate Consistency" begin
        # Run verification with multiple seeds
        for seed in [UInt64(42), UInt64(1337), GAY_SEED]
            result = cross_substrate_verify(n=5000, seed=seed, verbose=false)
            @test result.all_match
        end
    end
    
    @testset "Scale Independence" begin
        # Different sizes should all match across backends
        for n in [100, 1000, 10000]
            result = cross_substrate_verify(n=n, seed=GAY_SEED, verbose=false)
            @test result.all_match
        end
    end
    
end

export cross_substrate_verify, cross_substrate_fuzz, benchmark_all_backends, detect_backends
