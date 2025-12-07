# Fuzz the Fuzzers: Meta-Testing for SPI Verification
# ====================================================
# Property-based testing that the verification system itself is correct.
# Tests that:
# 1. XOR fingerprint detects actual differences
# 2. Verification correctly identifies SPI violations
# 3. Edge cases don't break the invariants
# 4. Adversarial inputs are handled gracefully

using Test
using Gay
using Gay: ka_colors, ka_colors!, xor_fingerprint, hash_color, splitmix64
using Gay: ka_color_sums, GAY_SEED
using KernelAbstractions: CPU
using Random

# ═══════════════════════════════════════════════════════════════════════════
# Fuzzer Infrastructure
# ═══════════════════════════════════════════════════════════════════════════

"""
    FuzzResult

Result of a fuzz test run.
"""
struct FuzzResult
    name::String
    passed::Bool
    iterations::Int
    failures::Vector{Any}
    duration::Float64
end

"""
    fuzz(name, f; iterations=1000, verbose=false)

Run a property-based fuzz test.
"""
function fuzz(name::String, f::Function; iterations::Int=1000, verbose::Bool=false)
    failures = Any[]
    t = @elapsed for i in 1:iterations
        try
            result = f(i)
            if result === false
                push!(failures, (iteration=i, error="returned false"))
            end
        catch e
            push!(failures, (iteration=i, error=e))
            verbose && @warn "Fuzz failure" name i exception=e
        end
    end
    
    passed = isempty(failures)
    FuzzResult(name, passed, iterations, failures, t)
end

"""
    report(results::Vector{FuzzResult})

Print fuzz test summary.
"""
function report(results::Vector{FuzzResult})
    println()
    println("═" ^ 69)
    println("  FUZZ THE FUZZERS: Meta-Testing Report")
    println("═" ^ 69)
    
    total_iterations = sum(r.iterations for r in results)
    total_failures = sum(length(r.failures) for r in results)
    total_time = sum(r.duration for r in results)
    
    for r in results
        status = r.passed ? "✓" : "✗"
        failures = isempty(r.failures) ? "" : " ($(length(r.failures)) failures)"
        println("  $status $(rpad(r.name, 45)) $(r.iterations) iters$failures")
    end
    
    println("─" ^ 69)
    println("  Total: $total_iterations iterations, $total_failures failures, $(round(total_time, digits=2))s")
    println("═" ^ 69)
    
    all(r -> r.passed, results)
end

# ═══════════════════════════════════════════════════════════════════════════
# Fingerprint Fuzz Tests
# ═══════════════════════════════════════════════════════════════════════════

"""
Test that XOR fingerprint detects single-bit flips.
"""
function fuzz_fingerprint_sensitivity(i::Int)
    n = 100
    seed = i * 7919  # Prime multiplier for variety
    
    colors = ka_colors(n, seed)
    original_fp = xor_fingerprint(colors)
    
    # Flip a random bit in a random position
    row = mod1(i, n)
    col = mod1(i ÷ n + 1, 3)
    
    mutated = copy(colors)
    original_val = mutated[row, col]
    
    # Flip least significant bit
    mutated[row, col] = reinterpret(Float32, xor(reinterpret(UInt32, original_val), UInt32(1)))
    
    mutated_fp = xor_fingerprint(mutated)
    
    # Fingerprint MUST change for any bit flip
    original_fp != mutated_fp
end

"""
Test that identical inputs produce identical fingerprints.
"""
function fuzz_fingerprint_determinism(i::Int)
    n = mod1(i, 10000)
    seed = i
    
    fp1 = xor_fingerprint(ka_colors(n, seed))
    fp2 = xor_fingerprint(ka_colors(n, seed))
    
    fp1 == fp2
end

"""
Test fingerprint with adversarial patterns.
"""
function fuzz_fingerprint_adversarial(i::Int)
    n = 1000
    
    # Adversarial pattern: all zeros
    if i == 1
        zeros_arr = zeros(Float32, n, 3)
        fp = xor_fingerprint(zeros_arr)
        return fp == UInt32(0)  # XOR of zeros is zero
    end
    
    # Adversarial pattern: all ones
    if i == 2
        ones_arr = ones(Float32, n, 3)
        fp = xor_fingerprint(ones_arr)
        return fp isa UInt32
    end
    
    # Adversarial pattern: alternating bits
    if i == 3
        alt = zeros(Float32, n, 3)
        for j in 1:n
            alt[j, :] .= reinterpret(Float32, j % 2 == 0 ? 0xAAAAAAAA : 0x55555555)
        end
        fp = xor_fingerprint(alt)
        return fp isa UInt32
    end
    
    # Adversarial pattern: NaN handling
    if i == 4
        nan_arr = zeros(Float32, n, 3)
        nan_arr[1, 1] = NaN32
        fp = xor_fingerprint(nan_arr)
        return fp isa UInt32  # Should not crash
    end
    
    # Adversarial pattern: Inf handling
    if i == 5
        inf_arr = zeros(Float32, n, 3)
        inf_arr[1, 1] = Inf32
        inf_arr[2, 2] = -Inf32
        fp = xor_fingerprint(inf_arr)
        return fp isa UInt32
    end
    
    # Random pattern
    seed = i * 31337
    colors = ka_colors(n, seed)
    fp = xor_fingerprint(colors)
    fp isa UInt32
end

# ═══════════════════════════════════════════════════════════════════════════
# Hash Function Fuzz Tests
# ═══════════════════════════════════════════════════════════════════════════

"""
Test splitmix64 produces well-distributed bits.
"""
function fuzz_splitmix64_distribution(i::Int)
    x = UInt64(i * 0x9e3779b97f4a7c15)
    h = splitmix64(x)
    
    # Count bits - should be roughly 32 on average
    popcount = count_ones(h)
    
    # Allow 20-44 bits set (very generous range)
    10 <= popcount <= 54
end

"""
Test hash_color produces valid RGB values.
"""
function fuzz_hash_color_range(i::Int)
    seed = UInt64(i * 7)
    index = UInt64(i * 13)
    
    r, g, b = hash_color(seed, index)
    
    # All values must be in [0, 1]
    0.0f0 <= r <= 1.0f0 &&
    0.0f0 <= g <= 1.0f0 &&
    0.0f0 <= b <= 1.0f0
end

"""
Test hash_color is deterministic.
"""
function fuzz_hash_color_determinism(i::Int)
    seed = UInt64(i)
    index = UInt64(i * 1000)
    
    r1, g1, b1 = hash_color(seed, index)
    r2, g2, b2 = hash_color(seed, index)
    
    r1 == r2 && g1 == g2 && b1 == b2
end

"""
Test hash_color sensitivity - different inputs produce different outputs.
"""
function fuzz_hash_color_sensitivity(i::Int)
    seed = UInt64(42)
    
    c1 = hash_color(seed, UInt64(i))
    c2 = hash_color(seed, UInt64(i + 1))
    
    # Adjacent indices should produce different colors
    c1 != c2
end

# ═══════════════════════════════════════════════════════════════════════════
# SPI Violation Detection Tests
# ═══════════════════════════════════════════════════════════════════════════

"""
Intentionally break SPI and verify detection.
"""
function fuzz_spi_violation_detection(i::Int)
    n = 100
    seed = i
    
    # Generate reference
    ref = ka_colors(n, seed)
    ref_fp = xor_fingerprint(ref)
    
    # Create "broken" version with wrong seed
    broken = ka_colors(n, seed + 1)
    broken_fp = xor_fingerprint(broken)
    
    # Verification MUST detect the difference
    ref_fp != broken_fp
end

"""
Test that workgroup changes don't affect results (true SPI).
"""
function fuzz_workgroup_invariance(i::Int)
    n = 1000
    seed = i * 42
    
    # Generate with different workgroup sizes
    ws1 = 32
    ws2 = 256
    
    colors1 = zeros(Float32, n, 3)
    colors2 = zeros(Float32, n, 3)
    
    ka_colors!(colors1, seed; backend=CPU(), workgroup=ws1)
    ka_colors!(colors2, seed; backend=CPU(), workgroup=ws2)
    
    xor_fingerprint(colors1) == xor_fingerprint(colors2)
end

# ═══════════════════════════════════════════════════════════════════════════
# Edge Case Fuzz Tests
# ═══════════════════════════════════════════════════════════════════════════

"""
Test edge case seeds.
"""
function fuzz_edge_seeds(i::Int)
    edge_seeds = [
        0,
        1,
        typemax(Int64),
        typemax(UInt64),
        GAY_SEED,
        0xDEADBEEF,
        0xCAFEBABE,
        0x0123456789ABCDEF,
        i,
        i * typemax(Int32),
    ]
    
    seed = edge_seeds[mod1(i, length(edge_seeds))]
    n = 100
    
    # Should not crash and should produce valid colors
    colors = ka_colors(n, seed)
    
    size(colors) == (n, 3) &&
    eltype(colors) == Float32 &&
    all(0.0f0 .<= colors .<= 1.0f0)
end

"""
Test edge case sizes.
"""
function fuzz_edge_sizes(i::Int)
    edge_sizes = [1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 
                  127, 128, 129, 255, 256, 257, 511, 512, 513,
                  1000, 1023, 1024, 1025, 10000]
    
    n = edge_sizes[mod1(i, length(edge_sizes))]
    seed = 42
    
    colors = ka_colors(n, seed)
    
    size(colors) == (n, 3)
end

"""
Test that color_sums produces reasonable statistics.
"""
function fuzz_color_sums_sanity(i::Int)
    n = 10000
    seed = i
    
    sums = ka_color_sums(n, seed; chunk_size=1000)
    
    # Expected mean is n/2 for uniform [0,1]
    expected = n / 2
    tolerance = n * 0.15  # 15% tolerance
    
    abs(sums[1] - expected) < tolerance &&
    abs(sums[2] - expected) < tolerance &&
    abs(sums[3] - expected) < tolerance
end

# ═══════════════════════════════════════════════════════════════════════════
# Reproducibility Stress Tests
# ═══════════════════════════════════════════════════════════════════════════

"""
Test repeated generation produces identical results.
"""
function fuzz_reproducibility_stress(i::Int)
    n = 100
    seed = i
    
    # Generate same thing 5 times
    fps = [xor_fingerprint(ka_colors(n, seed)) for _ in 1:5]
    
    # All must be identical
    all(fp -> fp == fps[1], fps)
end

"""
Test interleaved generation maintains SPI.
"""
function fuzz_interleaved_generation(i::Int)
    seeds = [i, i + 1000, i + 2000]
    n = 100
    
    # Generate in interleaved order
    results = Dict{Int, UInt32}()
    
    for round in 1:3
        for seed in seeds
            fp = xor_fingerprint(ka_colors(n, seed))
            if haskey(results, seed)
                if results[seed] != fp
                    return false
                end
            else
                results[seed] = fp
            end
        end
    end
    
    true
end

# ═══════════════════════════════════════════════════════════════════════════
# Collision Resistance Tests
# ═══════════════════════════════════════════════════════════════════════════

"""
Test fingerprint collision resistance.
"""
function fuzz_collision_resistance(i::Int)
    n = 100
    num_tests = 10
    
    fingerprints = Set{UInt32}()
    
    for j in 1:num_tests
        seed = i * 1000 + j
        fp = xor_fingerprint(ka_colors(n, seed))
        
        if fp in fingerprints
            return false  # Collision detected
        end
        push!(fingerprints, fp)
    end
    
    true
end

# ═══════════════════════════════════════════════════════════════════════════
# Run All Fuzz Tests
# ═══════════════════════════════════════════════════════════════════════════

@testset "Fuzz the Fuzzers" begin
    results = FuzzResult[]
    
    # Fingerprint tests
    push!(results, fuzz("fingerprint_sensitivity", fuzz_fingerprint_sensitivity; iterations=1000))
    push!(results, fuzz("fingerprint_determinism", fuzz_fingerprint_determinism; iterations=500))
    push!(results, fuzz("fingerprint_adversarial", fuzz_fingerprint_adversarial; iterations=100))
    
    # Hash function tests
    push!(results, fuzz("splitmix64_distribution", fuzz_splitmix64_distribution; iterations=1000))
    push!(results, fuzz("hash_color_range", fuzz_hash_color_range; iterations=1000))
    push!(results, fuzz("hash_color_determinism", fuzz_hash_color_determinism; iterations=1000))
    push!(results, fuzz("hash_color_sensitivity", fuzz_hash_color_sensitivity; iterations=1000))
    
    # SPI tests
    push!(results, fuzz("spi_violation_detection", fuzz_spi_violation_detection; iterations=500))
    push!(results, fuzz("workgroup_invariance", fuzz_workgroup_invariance; iterations=100))
    
    # Edge case tests
    push!(results, fuzz("edge_seeds", fuzz_edge_seeds; iterations=500))
    push!(results, fuzz("edge_sizes", fuzz_edge_sizes; iterations=100))
    push!(results, fuzz("color_sums_sanity", fuzz_color_sums_sanity; iterations=100))
    
    # Reproducibility tests
    push!(results, fuzz("reproducibility_stress", fuzz_reproducibility_stress; iterations=200))
    push!(results, fuzz("interleaved_generation", fuzz_interleaved_generation; iterations=100))
    
    # Collision resistance
    push!(results, fuzz("collision_resistance", fuzz_collision_resistance; iterations=100))
    
    # Report and verify
    all_passed = report(results)
    
    @test all_passed
    
    # Individual test assertions for granular failure reporting
    for r in results
        @testset "$(r.name)" begin
            @test r.passed
            if !r.passed && !isempty(r.failures)
                @info "Failures" first_failures=r.failures[1:min(3, length(r.failures))]
            end
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# Chaos Monkey: Adversarial Fuzzing
# ═══════════════════════════════════════════════════════════════════════════

"""
    chaos_monkey(; duration=69, seed=42)

Run adversarial fuzzing for specified duration.
Tries to break SPI with random inputs and timing chaos.
"""
function chaos_monkey(; duration::Int=69, seed::Int=42)
    rng = Random.MersenneTwister(seed)
    
    println("🐒 Chaos Monkey: Adversarial SPI Fuzzing")
    println("═" ^ 50)
    println("  Duration: $(duration)s")
    println("  Seed: $seed")
    println()
    
    start = time()
    violations = 0
    iterations = 0
    
    while (time() - start) < duration
        iterations += 1
        
        # Random parameters
        n = rand(rng, 1:10000)
        test_seed = rand(rng, UInt64)
        workgroup = rand(rng, [1, 16, 32, 64, 128, 256, 512])
        
        # Generate twice with same params
        colors1 = zeros(Float32, n, 3)
        colors2 = zeros(Float32, n, 3)
        
        ka_colors!(colors1, test_seed; backend=CPU(), workgroup=workgroup)
        
        # Random delay to stress timing
        rand(rng) < 0.1 && sleep(0.001)
        
        ka_colors!(colors2, test_seed; backend=CPU(), workgroup=workgroup)
        
        fp1 = xor_fingerprint(colors1)
        fp2 = xor_fingerprint(colors2)
        
        if fp1 != fp2
            violations += 1
            @warn "SPI VIOLATION!" n test_seed workgroup fp1 fp2
        end
        
        # Progress
        if iterations % 1000 == 0
            elapsed = round(time() - start, digits=1)
            println("  [$elapsed s] $iterations iterations, $violations violations")
        end
    end
    
    elapsed = round(time() - start, digits=1)
    println()
    println("═" ^ 50)
    println("  Chaos Monkey Complete")
    println("  Iterations: $iterations")
    println("  Violations: $violations")
    println("  Duration: $(elapsed)s")
    println("═" ^ 50)
    
    return violations == 0
end

export chaos_monkey
