# Metal.jl Performance Regression Detection
#
# This benchmark specifically tracks GPU performance to detect regressions
# in Metal.jl or KernelAbstractions.jl that affect Gay.jl color generation.
#
# Run: julia bench/metal_regression.jl
#
# Example regression scenario:
#   A Metal.jl update introduces extra synchronization overhead,
#   causing ka_color_sums to be 2x slower. This test catches it.

using Pkg
Pkg.activate(dirname(@__DIR__))

using Gay
using Gay.KernelAbstractions
using Printf

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

const SEED = Gay.GAY_SEED
const N_WARMUP = 3
const N_SAMPLES = 10
const TEST_SIZES = [1_000_000, 10_000_000, 100_000_000]

# Regression thresholds (detect if slower by this factor)
const REGRESSION_THRESHOLD = 1.5  # 50% slower = regression
const SEVERE_REGRESSION = 2.0     # 100% slower = severe

# Baseline expectations (colors/second) - update these with known-good values
const BASELINE_RATES = Dict(
    :cpu_1M => 50_000_000.0,    # 50M/s expected for 1M colors on CPU
    :cpu_10M => 45_000_000.0,   # Slightly lower due to cache pressure
    :cpu_100M => 40_000_000.0,  # Lower still for large batches
    :gpu_1M => 500_000_000.0,   # 500M/s expected for 1M on Metal GPU
    :gpu_10M => 2_000_000_000.0, # 2B/s at scale
    :gpu_100M => 3_000_000_000.0, # 3B/s peak throughput
)

# ═══════════════════════════════════════════════════════════════════════════
# Benchmark Utilities
# ═══════════════════════════════════════════════════════════════════════════

struct BenchmarkResult
    name::String
    n_colors::Int
    times::Vector{Float64}
    rates::Vector{Float64}
    mean_rate::Float64
    std_rate::Float64
    min_time::Float64
    backend::Symbol
end

function benchmark_color_sums(n::Int, backend; n_warmup=N_WARMUP, n_samples=N_SAMPLES)
    times = Float64[]
    rates = Float64[]
    
    # Warmup
    for _ in 1:n_warmup
        Gay.ka_color_sums(n, SEED; backend=backend)
    end
    
    # Timed samples
    for _ in 1:n_samples
        _, _, _, elapsed, rate = Gay.ka_color_sums(n, SEED; backend=backend)
        push!(times, elapsed)
        push!(rates, rate)
    end
    
    backend_sym = backend isa Gay.KernelAbstractions.CPU ? :cpu : :gpu
    name = "$(backend_sym)_$(n ÷ 1_000_000)M"
    
    return BenchmarkResult(
        name, n, times, rates,
        sum(rates) / length(rates),
        std(rates),
        minimum(times),
        backend_sym
    )
end

function std(v::Vector{Float64})
    μ = sum(v) / length(v)
    sqrt(sum((x - μ)^2 for x in v) / length(v))
end

# ═══════════════════════════════════════════════════════════════════════════
# Regression Detection
# ═══════════════════════════════════════════════════════════════════════════

struct RegressionReport
    benchmark::String
    current_rate::Float64
    baseline_rate::Float64
    ratio::Float64
    status::Symbol  # :ok, :regression, :severe, :improvement
    message::String
end

function check_regression(result::BenchmarkResult)
    key = Symbol("$(result.backend)_$(result.n_colors ÷ 1_000_000)M")
    
    if !haskey(BASELINE_RATES, key)
        return RegressionReport(
            result.name, result.mean_rate, 0.0, 1.0,
            :unknown, "No baseline for $key"
        )
    end
    
    baseline = BASELINE_RATES[key]
    ratio = result.mean_rate / baseline
    
    status, message = if ratio < 1.0 / SEVERE_REGRESSION
        (:severe, "🚨 SEVERE REGRESSION: $(round((1/ratio - 1) * 100, digits=1))% slower!")
    elseif ratio < 1.0 / REGRESSION_THRESHOLD
        (:regression, "⚠️  REGRESSION: $(round((1/ratio - 1) * 100, digits=1))% slower")
    elseif ratio > REGRESSION_THRESHOLD
        (:improvement, "✅ IMPROVEMENT: $(round((ratio - 1) * 100, digits=1))% faster")
    else
        (:ok, "✓ OK: within expected range")
    end
    
    return RegressionReport(result.name, result.mean_rate, baseline, ratio, status, message)
end

# ═══════════════════════════════════════════════════════════════════════════
# Simulated Regression (for testing the detection)
# ═══════════════════════════════════════════════════════════════════════════

"""
Simulate a Metal.jl regression by adding artificial delays.
This demonstrates what the regression detection would catch.
"""
function simulate_metal_regression!(; regression_factor=2.0)
    @warn "SIMULATING Metal.jl regression ($(regression_factor)x slowdown)"
    
    # Store original function
    original_ka_color_sums = Gay.ka_color_sums
    
    # Replace with slow version
    @eval Gay begin
        function ka_color_sums_slow(n::Integer, seed::Integer=GAY_SEED;
                                    chunk_size::Int=10000, backend=get_backend(), 
                                    workgroup::Int=256)
            # Artificial delay simulating Metal.jl overhead bug
            sleep(0.001 * $regression_factor)  
            
            # Call original (would be slower in real regression)
            r, g, b, elapsed, rate = $original_ka_color_sums(n, seed; 
                chunk_size=chunk_size, backend=backend, workgroup=workgroup)
            
            # Adjust reported rate to simulate regression
            return (r, g, b, elapsed * $regression_factor, rate / $regression_factor)
        end
    end
    
    return original_ka_color_sums
end

function restore_metal_performance!(original_fn)
    @eval Gay ka_color_sums = $original_fn
    @info "Restored original Metal performance"
end

# ═══════════════════════════════════════════════════════════════════════════
# Main Benchmark Runner
# ═══════════════════════════════════════════════════════════════════════════

function run_metal_benchmarks(; simulate_regression::Bool=false, regression_factor::Float64=2.0)
    println("=" ^ 70)
    println("🔬 Metal.jl Performance Regression Test")
    println("=" ^ 70)
    println()
    
    # Check Metal availability
    if !Gay.HAS_METAL
        println("⚠️  Metal.jl not available - running CPU-only benchmarks")
        backends = [(KernelAbstractions.CPU(), :cpu)]
    else
        @eval using Metal
        backends = [
            (KernelAbstractions.CPU(), :cpu),
            (Metal.MetalBackend(), :gpu)
        ]
        println("✓ Metal.jl available - testing CPU and GPU")
    end
    println()
    
    # Optionally simulate regression
    original_fn = nothing
    if simulate_regression
        original_fn = simulate_metal_regression!(; regression_factor=regression_factor)
    end
    
    results = BenchmarkResult[]
    reports = RegressionReport[]
    
    try
        for (backend, backend_name) in backends
            println("─" ^ 70)
            println("Backend: $(uppercase(string(backend_name)))")
            println("─" ^ 70)
            
            for n in TEST_SIZES
                print("  $(n ÷ 1_000_000)M colors: ")
                
                result = benchmark_color_sums(n, backend)
                push!(results, result)
                
                report = check_regression(result)
                push!(reports, report)
                
                rate_str = if result.mean_rate >= 1e9
                    @sprintf("%.2f B/s", result.mean_rate / 1e9)
                else
                    @sprintf("%.1f M/s", result.mean_rate / 1e6)
                end
                
                status_icon = if report.status == :severe
                    "🚨"
                elseif report.status == :regression
                    "⚠️"
                elseif report.status == :improvement
                    "🚀"
                else
                    "✓"
                end
                
                println("$rate_str $status_icon $(report.message)")
            end
            println()
        end
        
    finally
        # Restore if we simulated regression
        if original_fn !== nothing
            restore_metal_performance!(original_fn)
        end
    end
    
    # Summary
    println("=" ^ 70)
    println("Summary")
    println("=" ^ 70)
    
    regressions = filter(r -> r.status in [:regression, :severe], reports)
    improvements = filter(r -> r.status == :improvement, reports)
    
    if !isempty(regressions)
        println()
        println("🚨 REGRESSIONS DETECTED:")
        for r in regressions
            @printf("  %-12s: %.2fx slower (%.1f M/s vs %.1f M/s baseline)\n",
                    r.benchmark, 1/r.ratio, r.current_rate/1e6, r.baseline_rate/1e6)
        end
        println()
        println("Action: Check Metal.jl / KernelAbstractions.jl updates")
        println("        Run `git bisect` to find offending commit")
    end
    
    if !isempty(improvements)
        println()
        println("🚀 IMPROVEMENTS:")
        for r in improvements
            @printf("  %-12s: %.2fx faster\n", r.benchmark, r.ratio)
        end
    end
    
    if isempty(regressions) && isempty(improvements)
        println()
        println("✅ All benchmarks within expected range")
    end
    
    # Exit code for CI
    has_regression = !isempty(regressions)
    has_severe = any(r -> r.status == :severe, reports)
    
    println()
    println("─" ^ 70)
    if has_severe
        println("EXIT: 2 (severe regression)")
        return 2
    elseif has_regression
        println("EXIT: 1 (regression detected)")
        return 1
    else
        println("EXIT: 0 (ok)")
        return 0
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# EDN Output for CI Integration
# ═══════════════════════════════════════════════════════════════════════════

function benchmark_to_edn(results::Vector{BenchmarkResult}, reports::Vector{RegressionReport})
    println()
    println(";; EDN Benchmark Results")
    println("{:timestamp $(round(Int, time()))")
    println(" :metal-available $(Gay.HAS_METAL)")
    println(" :benchmarks [")
    
    for (result, report) in zip(results, reports)
        println("   {:name \"$(result.name)\"")
        println("    :n-colors $(result.n_colors)")
        println("    :mean-rate $(round(result.mean_rate, digits=0))")
        println("    :baseline $(round(report.baseline_rate, digits=0))")
        println("    :ratio $(round(report.ratio, digits=3))")
        println("    :status :$(report.status)}")
    end
    
    println(" ]}")
end

# ═══════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════

if abspath(PROGRAM_FILE) == @__FILE__
    # Parse command line args
    simulate = "--simulate-regression" in ARGS
    factor = 2.0
    
    for arg in ARGS
        if startswith(arg, "--factor=")
            factor = parse(Float64, split(arg, "=")[2])
        end
    end
    
    exit_code = run_metal_benchmarks(; 
        simulate_regression=simulate, 
        regression_factor=factor
    )
    
    exit(exit_code)
end
