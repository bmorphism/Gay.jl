# SPI Regression Benchmarks with Chairmarks
# ==========================================
#
# Performance regression tests for the SPI color verification system.
# Inspired by LilithHafner's Chairmarks philosophy:
# - Fast benchmarks (hundreds of times faster than BenchmarkTools)
# - Accurate measurements without compromising precision
# - Simple API: @b for quick, @be for detailed
#
# These benchmarks establish baselines and detect regressions in:
# - SplitMix64 RNG performance
# - Galois connection operations
# - XOR fingerprint computation
# - Distributed verification overhead
#
# Usage:
#   using Gay
#   run_spi_regression_tests()
#   
# Or for CI:
#   julia --project=. -e 'using Gay; exit(run_spi_regression_tests() ? 0 : 1)'

module SPIRegressionBench

using Chairmarks: @b, @be, Benchmark, median, minimum
using Printf
using Statistics: mean, std

# Import from parent
using ..Gay: GAY_SEED, hash_color, splitmix64, xor_fingerprint, SplitMix64RNG, next!
using ..FaultTolerant: GaloisConnection, Event, Color, alpha, gamma, verify_closure, verify_all_closures
using ..KernelLifetimes: eventual_color, eventual_fingerprint

export run_spi_regression_tests, SPIBenchResult, benchmark_spi_core
export benchmark_galois, benchmark_fingerprint, benchmark_kernel_lifetimes

# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark Result with Regression Detection
# ═══════════════════════════════════════════════════════════════════════════════

"""
SPI benchmark result with regression detection.
"""
struct SPIBenchResult
    name::String
    median_ns::Float64
    min_ns::Float64
    max_ns::Float64
    samples::Int
    allocs::Int
    bytes::Int
    baseline_ns::Float64  # Expected baseline (0 = no baseline)
    regression::Bool      # True if > 20% slower than baseline
end

function Base.show(io::IO, r::SPIBenchResult)
    time_str = format_time(r.median_ns)
    status = if r.baseline_ns > 0
        ratio = r.median_ns / r.baseline_ns
        if r.regression
            "✗ REGRESSION ($(round(ratio, digits=2))x)"
        elseif ratio < 0.9
            "✓ FASTER ($(round(ratio, digits=2))x)"
        else
            "✓ OK"
        end
    else
        "○ (no baseline)"
    end
    print(io, "$(r.name): $(time_str) $(status)")
end

function format_time(ns::Float64)
    if ns < 1000
        @sprintf("%.1f ns", ns)
    elseif ns < 1_000_000
        @sprintf("%.2f μs", ns / 1000)
    else
        @sprintf("%.2f ms", ns / 1_000_000)
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Baseline Expectations (empirical, adjust for the target hardware)
# ═══════════════════════════════════════════════════════════════════════════════

# These baselines are approximate for Apple M-series chips
# Adjust based on the target hardware
const BASELINES = Dict{String, Float64}(
    # Core SPI
    "splitmix64" => 2.0,           # ~2 ns
    "splitmix64_rng_next" => 2.5,  # ~2.5 ns
    "hash_color" => 5.0,           # ~5 ns
    
    # Galois connection
    "galois_alpha" => 15.0,        # ~15 ns
    "galois_gamma" => 5.0,         # ~5 ns (table lookup)
    "galois_verify_closure" => 25.0, # ~25 ns
    
    # Fingerprints
    "xor_fingerprint_100" => 50.0,     # ~50 ns for 100 elements
    "xor_fingerprint_10000" => 2000.0, # ~2 μs for 10k elements
    
    # Kernel lifetimes
    "eventual_color" => 10.0,      # ~10 ns
    "eventual_fingerprint_100" => 500.0,  # ~500 ns for 100 workitems
)

"""
Get baseline for a benchmark, or 0 if unknown.
"""
get_baseline(name::String) = get(BASELINES, name, 0.0)

"""
Check if result is a regression (> 20% slower than baseline).
"""
is_regression(median_ns::Float64, baseline_ns::Float64) = 
    baseline_ns > 0 && median_ns > baseline_ns * 1.2

# ═══════════════════════════════════════════════════════════════════════════════
# Core SPI Benchmarks
# ═══════════════════════════════════════════════════════════════════════════════

"""
Benchmark core SPI functions.
"""
function benchmark_spi_core()
    results = SPIBenchResult[]
    
    # splitmix64 (stateless)
    b = @be splitmix64(UInt64(42))
    baseline = get_baseline("splitmix64")
    med = median(b).time * 1e9
    push!(results, SPIBenchResult(
        "splitmix64",
        med,
        minimum(b).time * 1e9,
        maximum(s.time for s in b.samples) * 1e9,
        length(b.samples),
        Int(median(b).allocs),
        Int(median(b).bytes),
        baseline,
        is_regression(med, baseline)
    ))
    
    # SplitMix64RNG.next!
    rng = SplitMix64RNG(GAY_SEED)
    b = @be next!($rng)
    baseline = get_baseline("splitmix64_rng_next")
    med = median(b).time * 1e9
    push!(results, SPIBenchResult(
        "splitmix64_rng_next",
        med,
        minimum(b).time * 1e9,
        maximum(s.time for s in b.samples) * 1e9,
        length(b.samples),
        Int(median(b).allocs),
        Int(median(b).bytes),
        baseline,
        is_regression(med, baseline)
    ))
    
    # hash_color
    b = @be hash_color(UInt64(42), UInt64(100))
    baseline = get_baseline("hash_color")
    med = median(b).time * 1e9
    push!(results, SPIBenchResult(
        "hash_color",
        med,
        minimum(b).time * 1e9,
        maximum(s.time for s in b.samples) * 1e9,
        length(b.samples),
        Int(median(b).allocs),
        Int(median(b).bytes),
        baseline,
        is_regression(med, baseline)
    ))
    
    results
end

# ═══════════════════════════════════════════════════════════════════════════════
# Galois Connection Benchmarks
# ═══════════════════════════════════════════════════════════════════════════════

"""
Benchmark Galois connection operations.
"""
function benchmark_galois()
    results = SPIBenchResult[]
    gc = GaloisConnection(GAY_SEED)
    event = Event(GAY_SEED, 42, 5, 100)
    color = Color(50, gc.palette[51])
    
    # alpha (abstraction)
    b = @be alpha($gc, $event)
    baseline = get_baseline("galois_alpha")
    med = median(b).time * 1e9
    push!(results, SPIBenchResult(
        "galois_alpha",
        med,
        minimum(b).time * 1e9,
        maximum(s.time for s in b.samples) * 1e9,
        length(b.samples),
        Int(median(b).allocs),
        Int(median(b).bytes),
        baseline,
        is_regression(med, baseline)
    ))
    
    # gamma (concretization)
    b = @be gamma($gc, $color)
    baseline = get_baseline("galois_gamma")
    med = median(b).time * 1e9
    push!(results, SPIBenchResult(
        "galois_gamma",
        med,
        minimum(b).time * 1e9,
        maximum(s.time for s in b.samples) * 1e9,
        length(b.samples),
        Int(median(b).allocs),
        Int(median(b).bytes),
        baseline,
        is_regression(med, baseline)
    ))
    
    # verify_closure
    b = @be verify_closure($gc, $color)
    baseline = get_baseline("galois_verify_closure")
    med = median(b).time * 1e9
    push!(results, SPIBenchResult(
        "galois_verify_closure",
        med,
        minimum(b).time * 1e9,
        maximum(s.time for s in b.samples) * 1e9,
        length(b.samples),
        Int(median(b).allocs),
        Int(median(b).bytes),
        baseline,
        is_regression(med, baseline)
    ))
    
    results
end

# ═══════════════════════════════════════════════════════════════════════════════
# XOR Fingerprint Benchmarks
# ═══════════════════════════════════════════════════════════════════════════════

"""
Benchmark XOR fingerprint computation at various sizes.
"""
function benchmark_fingerprint()
    results = SPIBenchResult[]
    
    for (n, name) in [(100, "xor_fingerprint_100"), (10000, "xor_fingerprint_10000")]
        arr = randn(Float32, n, 4)
        
        b = @be xor_fingerprint($arr)
        baseline = get_baseline(name)
        med = median(b).time * 1e9
        push!(results, SPIBenchResult(
            name,
            med,
            minimum(b).time * 1e9,
            maximum(s.time for s in b.samples) * 1e9,
            length(b.samples),
            Int(median(b).allocs),
            Int(median(b).bytes),
            baseline,
            is_regression(med, baseline)
        ))
    end
    
    results
end

# ═══════════════════════════════════════════════════════════════════════════════
# Kernel Lifetime Benchmarks
# ═══════════════════════════════════════════════════════════════════════════════

"""
Benchmark kernel lifetime eventual color operations.
"""
function benchmark_kernel_lifetimes()
    results = SPIBenchResult[]
    
    # eventual_color (O(1) prediction)
    b = @be eventual_color(GAY_SEED, 42, 100)
    baseline = get_baseline("eventual_color")
    med = median(b).time * 1e9
    push!(results, SPIBenchResult(
        "eventual_color",
        med,
        minimum(b).time * 1e9,
        maximum(s.time for s in b.samples) * 1e9,
        length(b.samples),
        Int(median(b).allocs),
        Int(median(b).bytes),
        baseline,
        is_regression(med, baseline)
    ))
    
    # eventual_fingerprint (O(n) but no kernel execution)
    b = @be eventual_fingerprint(GAY_SEED, 100, 10)
    baseline = get_baseline("eventual_fingerprint_100")
    med = median(b).time * 1e9
    push!(results, SPIBenchResult(
        "eventual_fingerprint_100",
        med,
        minimum(b).time * 1e9,
        maximum(s.time for s in b.samples) * 1e9,
        length(b.samples),
        Int(median(b).allocs),
        Int(median(b).bytes),
        baseline,
        is_regression(med, baseline)
    ))
    
    results
end

# ═══════════════════════════════════════════════════════════════════════════════
# Full Regression Test Suite
# ═══════════════════════════════════════════════════════════════════════════════

"""
    run_spi_regression_tests(; verbose=true) -> Bool

Run the full SPI regression test suite.
Returns true if all tests pass (no regressions), false otherwise.
"""
function run_spi_regression_tests(; verbose::Bool=true)
    all_results = SPIBenchResult[]
    regressions = SPIBenchResult[]
    
    verbose && println("╔═══════════════════════════════════════════════════════════════════╗")
    verbose && println("║       SPI Performance Regression Tests (Chairmarks)              ║")
    verbose && println("╚═══════════════════════════════════════════════════════════════════╝")
    verbose && println()
    
    # Core SPI
    verbose && println("▶ Core SPI Functions")
    results = benchmark_spi_core()
    append!(all_results, results)
    for r in results
        verbose && println("  ", r)
        r.regression && push!(regressions, r)
    end
    verbose && println()
    
    # Galois Connection
    verbose && println("▶ Galois Connection")
    results = benchmark_galois()
    append!(all_results, results)
    for r in results
        verbose && println("  ", r)
        r.regression && push!(regressions, r)
    end
    verbose && println()
    
    # XOR Fingerprint
    verbose && println("▶ XOR Fingerprint")
    results = benchmark_fingerprint()
    append!(all_results, results)
    for r in results
        verbose && println("  ", r)
        r.regression && push!(regressions, r)
    end
    verbose && println()
    
    # Kernel Lifetimes
    verbose && println("▶ Kernel Lifetimes")
    results = benchmark_kernel_lifetimes()
    append!(all_results, results)
    for r in results
        verbose && println("  ", r)
        r.regression && push!(regressions, r)
    end
    verbose && println()
    
    # Summary
    verbose && println("═══════════════════════════════════════════════════════════════════")
    n_total = length(all_results)
    n_with_baseline = count(r -> r.baseline_ns > 0, all_results)
    n_regressions = length(regressions)
    
    if n_regressions == 0
        verbose && println("✓ All $(n_total) benchmarks passed ($(n_with_baseline) with baselines)")
    else
        verbose && println("✗ $(n_regressions) regressions detected:")
        for r in regressions
            verbose && println("  - $(r.name): $(format_time(r.median_ns)) vs baseline $(format_time(r.baseline_ns))")
        end
    end
    verbose && println("═══════════════════════════════════════════════════════════════════")
    
    return n_regressions == 0
end

# ═══════════════════════════════════════════════════════════════════════════════
# Baseline Calibration (run on reference hardware)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    calibrate_baselines(; verbose=true) -> Dict{String, Float64}

Run benchmarks and print new baseline values.
Use this to update BASELINES after hardware changes.
"""
function calibrate_baselines(; verbose::Bool=true)
    verbose && println("Calibrating SPI baselines...")
    verbose && println()
    
    new_baselines = Dict{String, Float64}()
    
    # Run all benchmarks
    for results in [benchmark_spi_core(), benchmark_galois(), 
                    benchmark_fingerprint(), benchmark_kernel_lifetimes()]
        for r in results
            new_baselines[r.name] = r.median_ns
            verbose && println("  \"$(r.name)\" => $(round(r.median_ns, digits=1)),")
        end
    end
    
    verbose && println()
    verbose && println("Copy the above into BASELINES dict to update.")
    
    new_baselines
end

export calibrate_baselines

end # module SPIRegressionBench
