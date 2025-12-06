# Gay.jl Benchmarking with Chairmarks
# ====================================
# Integrated microbenchmarking for color generation performance
# Uses Chairmarks.jl for accurate, low-overhead measurements

using Chairmarks: @b, @be, Benchmark, median, minimum
using Printf
using Colors: RGB

export @gay_bench, gay_benchmark, benchmark_colors, benchmark_ka
export benchmark_abductive, benchmark_teleportation
export BenchmarkResult, format_benchmark

# ═══════════════════════════════════════════════════════════════════════════
# Benchmark Result Type
# ═══════════════════════════════════════════════════════════════════════════

"""
    BenchmarkResult

Structured result from Gay.jl benchmarks with Chairmarks.
"""
struct BenchmarkResult
    name::String
    median_ns::Float64
    min_ns::Float64
    samples::Int
    allocs::Int
    bytes::Int
    rate::Float64  # operations per second
end

function format_benchmark(r::BenchmarkResult)
    if r.median_ns < 1000
        time_str = @sprintf("%.1f ns", r.median_ns)
    elseif r.median_ns < 1_000_000
        time_str = @sprintf("%.2f μs", r.median_ns / 1000)
    else
        time_str = @sprintf("%.2f ms", r.median_ns / 1_000_000)
    end
    
    if r.rate > 1_000_000_000
        rate_str = @sprintf("%.2f G/s", r.rate / 1_000_000_000)
    elseif r.rate > 1_000_000
        rate_str = @sprintf("%.2f M/s", r.rate / 1_000_000)
    elseif r.rate > 1000
        rate_str = @sprintf("%.2f K/s", r.rate / 1000)
    else
        rate_str = @sprintf("%.2f /s", r.rate)
    end
    
    return "$(r.name): $(time_str) ($(rate_str), $(r.allocs) allocs, $(r.bytes) bytes)"
end

Base.show(io::IO, r::BenchmarkResult) = print(io, format_benchmark(r))

# ═══════════════════════════════════════════════════════════════════════════
# Core Color Benchmarks
# ═══════════════════════════════════════════════════════════════════════════

"""
    benchmark_colors(; n=1000, seed=42) -> Vector{BenchmarkResult}

Benchmark core color generation functions.
"""
function benchmark_colors(; n::Int=1000, seed::Int=42)
    results = BenchmarkResult[]
    
    # hash_color (raw performance)
    b = @be hash_color(UInt64(42), UInt64($seed))
    push!(results, BenchmarkResult(
        "hash_color",
        median(b).time * 1e9,
        minimum(b).time * 1e9,
        length(b.samples),
        Int(median(b).allocs),
        Int(median(b).bytes),
        1e9 / (median(b).time * 1e9)
    ))
    
    # splitmix64
    b = @be splitmix64(UInt64(42))
    push!(results, BenchmarkResult(
        "splitmix64",
        median(b).time * 1e9,
        minimum(b).time * 1e9,
        length(b.samples),
        Int(median(b).allocs),
        Int(median(b).bytes),
        1e9 / (median(b).time * 1e9)
    ))
    
    # color_at
    gay_seed!(seed)
    b = @be color_at(42)
    push!(results, BenchmarkResult(
        "color_at",
        median(b).time * 1e9,
        minimum(b).time * 1e9,
        length(b.samples),
        Int(median(b).allocs),
        Int(median(b).bytes),
        1e9 / (median(b).time * 1e9)
    ))
    
    # next_color
    gay_seed!(seed)
    b = @be next_color()
    push!(results, BenchmarkResult(
        "next_color",
        median(b).time * 1e9,
        minimum(b).time * 1e9,
        length(b.samples),
        Int(median(b).allocs),
        Int(median(b).bytes),
        1e9 / (median(b).time * 1e9)
    ))
    
    # next_palette(6)
    gay_seed!(seed)
    b = @be next_palette(6)
    push!(results, BenchmarkResult(
        "next_palette(6)",
        median(b).time * 1e9,
        minimum(b).time * 1e9,
        length(b.samples),
        Int(median(b).allocs),
        Int(median(b).bytes),
        6e9 / (median(b).time * 1e9)  # 6 colors per call
    ))
    
    return results
end

# ═══════════════════════════════════════════════════════════════════════════
# KernelAbstractions Benchmarks
# ═══════════════════════════════════════════════════════════════════════════

"""
    benchmark_ka(; sizes=[1000, 10000, 100000], seed=42) -> Vector{BenchmarkResult}

Benchmark KernelAbstractions color generation at various sizes.
"""
function benchmark_ka(; sizes::Vector{Int}=[1000, 10000, 100000], seed::Int=42)
    results = BenchmarkResult[]
    backend = get_backend()
    
    for n in sizes
        # ka_colors
        b = @be ka_colors($n, $seed)
        push!(results, BenchmarkResult(
            "ka_colors($n)",
            median(b).time * 1e9,
            minimum(b).time * 1e9,
            length(b.samples),
            Int(median(b).allocs),
            Int(median(b).bytes),
            n * 1e9 / (median(b).time * 1e9)
        ))
        
        # ka_colors! (in-place)
        buf = zeros(Float32, n, 3)
        b = @be ka_colors!($buf, $seed)
        push!(results, BenchmarkResult(
            "ka_colors!($n)",
            median(b).time * 1e9,
            minimum(b).time * 1e9,
            length(b.samples),
            Int(median(b).allocs),
            Int(median(b).bytes),
            n * 1e9 / (median(b).time * 1e9)
        ))
    end
    
    # ka_color_sums (reduction)
    b = @be ka_color_sums(1_000_000, $seed)
    push!(results, BenchmarkResult(
        "ka_color_sums(1M)",
        median(b).time * 1e9,
        minimum(b).time * 1e9,
        length(b.samples),
        Int(median(b).allocs),
        Int(median(b).bytes),
        1_000_000 * 1e9 / (median(b).time * 1e9)
    ))
    
    return results
end

# ═══════════════════════════════════════════════════════════════════════════
# Abductive Testing Benchmarks
# ═══════════════════════════════════════════════════════════════════════════

"""
    benchmark_abductive(; n_invaders=100, seed=42) -> Vector{BenchmarkResult}

Benchmark abductive testing operations.
"""
function benchmark_abductive(; n_invaders::Int=100, seed::UInt64=GAY_SEED)
    results = BenchmarkResult[]
    
    # simulate_teleportation
    b = @be simulate_teleportation(42, $seed)
    push!(results, BenchmarkResult(
        "simulate_teleportation",
        median(b).time * 1e9,
        minimum(b).time * 1e9,
        length(b.samples),
        Int(median(b).allocs),
        Int(median(b).bytes),
        1e9 / (median(b).time * 1e9)
    ))
    
    # apply_derangement
    c = RGB(0.5, 0.3, 0.7)
    b = @be apply_derangement($c, 1)
    push!(results, BenchmarkResult(
        "apply_derangement",
        median(b).time * 1e9,
        minimum(b).time * 1e9,
        length(b.samples),
        Int(median(b).allocs),
        Int(median(b).bytes),
        1e9 / (median(b).time * 1e9)
    ))
    
    # tropical_blend
    c1 = RGB(0.2, 0.4, 0.6)
    c2 = RGB(0.8, 0.6, 0.4)
    b = @be tropical_blend($c1, $c2, 0.5)
    push!(results, BenchmarkResult(
        "tropical_blend",
        median(b).time * 1e9,
        minimum(b).time * 1e9,
        length(b.samples),
        Int(median(b).allocs),
        Int(median(b).bytes),
        1e9 / (median(b).time * 1e9)
    ))
    
    # color_distance
    b = @be color_distance($c1, $c2)
    push!(results, BenchmarkResult(
        "color_distance",
        median(b).time * 1e9,
        minimum(b).time * 1e9,
        length(b.samples),
        Int(median(b).allocs),
        Int(median(b).bytes),
        1e9 / (median(b).time * 1e9)
    ))
    
    # test_all_properties
    b = @be test_all_properties(42, $seed)
    push!(results, BenchmarkResult(
        "test_all_properties",
        median(b).time * 1e9,
        minimum(b).time * 1e9,
        length(b.samples),
        Int(median(b).allocs),
        Int(median(b).bytes),
        1e9 / (median(b).time * 1e9)
    ))
    
    return results
end

"""
    benchmark_teleportation(; n=1000, seed=42) -> BenchmarkResult

Benchmark batch teleportation for fleet generation.
"""
function benchmark_teleportation(; n::Int=1000, seed::UInt64=GAY_SEED)
    b = @be [simulate_teleportation(i, $seed) for i in 1:$n]
    
    return BenchmarkResult(
        "fleet_teleport($n)",
        median(b).time * 1e9,
        minimum(b).time * 1e9,
        length(b.samples),
        Int(median(b).allocs),
        Int(median(b).bytes),
        n * 1e9 / (median(b).time * 1e9)
    )
end

# ═══════════════════════════════════════════════════════════════════════════
# Full Benchmark Suite
# ═══════════════════════════════════════════════════════════════════════════

"""
    gay_benchmark(; verbose=true) -> Dict{String, Vector{BenchmarkResult}}

Run the full Gay.jl benchmark suite.
"""
function gay_benchmark(; verbose::Bool=true)
    results = Dict{String, Vector{BenchmarkResult}}()
    
    verbose && println("╔═══════════════════════════════════════════════════════════════╗")
    verbose && println("║           Gay.jl Benchmark Suite (Chairmarks)                 ║")
    verbose && println("╚═══════════════════════════════════════════════════════════════╝")
    verbose && println()
    
    # Core colors
    verbose && println("▶ Core Color Generation")
    results["colors"] = benchmark_colors()
    if verbose
        for r in results["colors"]
            println("  ", r)
        end
        println()
    end
    
    # KernelAbstractions
    verbose && println("▶ KernelAbstractions (Backend: $(typeof(get_backend())))")
    results["ka"] = benchmark_ka()
    if verbose
        for r in results["ka"]
            println("  ", r)
        end
        println()
    end
    
    # Abductive testing
    verbose && println("▶ Abductive Testing")
    results["abductive"] = benchmark_abductive()
    if verbose
        for r in results["abductive"]
            println("  ", r)
        end
        println()
    end
    
    # Fleet teleportation
    verbose && println("▶ Fleet Teleportation")
    results["teleportation"] = [benchmark_teleportation()]
    if verbose
        println("  ", results["teleportation"][1])
        println()
    end
    
    verbose && println("═══════════════════════════════════════════════════════════════")
    
    return results
end

"""
    @gay_bench expr

Quick benchmark macro for REPL use.
Returns median time in nanoseconds.
"""
macro gay_bench(expr)
    quote
        b = @be $(esc(expr))
        median_ns = median(b).time * 1e9
        @printf("%.2f ns (%.2f M/s)\n", median_ns, 1e9 / median_ns / 1e6)
        median_ns
    end
end
