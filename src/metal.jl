# Gay.jl Metal Backend Integration
# =================================
# Apple Silicon GPU acceleration via Metal.jl

using Metal
using KernelAbstractions

export MetalBackend, metal_available, use_metal!, metal_benchmark
export metal_colors!, metal_color_sums

# ═══════════════════════════════════════════════════════════════════════════
# Metal Backend Detection
# ═══════════════════════════════════════════════════════════════════════════

"""
    metal_available() -> Bool

Check if Metal GPU acceleration is available.
"""
function metal_available()
    try
        return Metal.functional()
    catch
        return false
    end
end

"""
    use_metal!() -> MetalBackend

Switch to Metal backend for GPU-accelerated color generation.
Throws error if Metal is not available.
"""
function use_metal!()
    if !metal_available()
        error("Metal is not available on this system")
    end
    backend = MetalBackend()
    set_backend!(backend)
    return backend
end

# ═══════════════════════════════════════════════════════════════════════════
# Metal-Optimized Color Generation
# ═══════════════════════════════════════════════════════════════════════════

"""
    metal_colors!(output::MtlArray, seed::Integer)

Generate colors directly on Metal GPU.
"""
function metal_colors!(output::MtlArray{Float32, 2}, seed::Integer)
    n = size(output, 1)
    ka_colors!(output, seed; backend=MetalBackend())
    return output
end

"""
    metal_colors(n::Integer, seed::Integer) -> MtlArray

Generate n colors on Metal GPU, returning GPU array.
"""
function metal_colors(n::Integer, seed::Integer)
    output = MtlArray{Float32}(undef, n, 3)
    metal_colors!(output, seed)
    return output
end

"""
    metal_color_sums(n::Integer, seed::Integer) -> NTuple{3, Float64}

Compute color channel sums on Metal GPU (billion-scale reduction).
"""
function metal_color_sums(n::Integer, seed::Integer)
    return ka_color_sums(n, seed; backend=MetalBackend())
end

# ═══════════════════════════════════════════════════════════════════════════
# Metal Benchmarking
# ═══════════════════════════════════════════════════════════════════════════

"""
    metal_benchmark(; sizes=[10000, 100000, 1000000]) -> Vector{NamedTuple}

Benchmark Metal GPU color generation at various sizes.
"""
function metal_benchmark(; sizes::Vector{Int}=[10_000, 100_000, 1_000_000])
    if !metal_available()
        error("Metal is not available")
    end
    
    results = NamedTuple[]
    
    println("╔═══════════════════════════════════════════════════════════════╗")
    println("║              Metal GPU Benchmark (Apple Silicon)              ║")
    println("╚═══════════════════════════════════════════════════════════════╝")
    println()
    
    for n in sizes
        # Warmup
        _ = metal_colors(n, 42)
        Metal.synchronize()
        
        # Benchmark
        start = time_ns()
        for _ in 1:10
            _ = metal_colors(n, 42)
            Metal.synchronize()
        end
        elapsed_ns = (time_ns() - start) / 10
        
        rate = n / (elapsed_ns / 1e9)
        
        result = (
            n = n,
            time_ms = elapsed_ns / 1e6,
            rate_millions = rate / 1e6,
            backend = "Metal"
        )
        push!(results, result)
        
        @printf("  n=%d: %.2f ms (%.1f M colors/s)\n", n, result.time_ms, result.rate_millions)
    end
    
    # CPU comparison
    println()
    println("  CPU comparison:")
    for n in sizes
        start = time_ns()
        for _ in 1:10
            _ = ka_colors(n, 42; backend=CPU())
        end
        elapsed_ns = (time_ns() - start) / 10
        rate = n / (elapsed_ns / 1e9)
        @printf("  n=%d: %.2f ms (%.1f M colors/s) [CPU]\n", n, elapsed_ns / 1e6, rate / 1e6)
    end
    
    println()
    println("═══════════════════════════════════════════════════════════════")
    
    return results
end

# ═══════════════════════════════════════════════════════════════════════════
# Metal Device Info
# ═══════════════════════════════════════════════════════════════════════════

"""
    metal_info() -> NamedTuple

Get Metal device information.
"""
function metal_info()
    if !metal_available()
        return (available = false,)
    end
    
    device = Metal.current_device()
    return (
        available = true,
        name = string(device.name),
        max_threads = device.maxThreadsPerThreadgroup.width,
        memory_size = device.currentAllocatedSize,
        is_low_power = device.isLowPower
    )
end
