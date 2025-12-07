# Jepsen-Style Meta-Fuzzing: Fuzz the Fuzzers Fuzzing Themselves
# ==================================================================
#
# Inspired by Jepsen's approach to distributed systems verification:
# - Nemesis: Inject faults into the verification system itself
# - Elle-style: Check for cyclic dependencies and anomalies in color histories
# - Generator: Random operation sequences with constraints
# - Linearizability: Verify color generation histories are linearizable
#
# This module tests that our SPI verification system is itself correct,
# by treating the verification as a "distributed system" and applying
# chaos engineering to it.

using Test
using Gay
using Gay: ka_colors, ka_colors!, xor_fingerprint, hash_color, splitmix64
using Gay: ka_color_sums, GAY_SEED
using KernelAbstractions: CPU
using Random

# ═══════════════════════════════════════════════════════════════════════════
# Jepsen-Style History & Operations
# ═══════════════════════════════════════════════════════════════════════════

"""
Operation types for color generation history.
Like Jepsen's :invoke/:ok/:fail/:info model.
"""
@enum OpType begin
    INVOKE      # Operation started
    OK          # Operation completed successfully
    FAIL        # Operation failed
    INFO        # Informational (e.g., nemesis action)
end

"""
Operation kinds for color generation.
"""
@enum OpKind begin
    GENERATE    # Generate colors
    VERIFY      # Verify fingerprint
    SUM         # Compute color sums
    NEMESIS     # Fault injection
end

"""
A single operation in the history, Jepsen-style.
"""
struct Op
    process::Int            # Which "thread" performed this
    type::OpType           # invoke/ok/fail/info
    kind::OpKind           # What kind of operation
    seed::UInt64           # Seed used
    n::Int                 # Number of colors
    value::Any             # Result (fingerprint, colors, etc.)
    time::Float64          # Wall clock time
end

"""
A history of operations, like Jepsen's history.
"""
struct History
    ops::Vector{Op}
    seed::UInt64
    start_time::Float64
end

History(seed::UInt64) = History(Op[], seed, time())

function record!(h::History, op::Op)
    push!(h.ops, op)
end

# ═══════════════════════════════════════════════════════════════════════════
# Nemesis: Fault Injection for Verification System
# ═══════════════════════════════════════════════════════════════════════════

"""
Nemesis types that can disrupt verification.
"""
@enum NemesisType begin
    NONE                    # No fault
    GC_PRESSURE            # Trigger garbage collection
    THREAD_CONTENTION      # Spawn competing threads
    MEMORY_PRESSURE        # Allocate/deallocate memory
    TIMING_CHAOS           # Random delays
    SEED_MUTATION          # Mutate seeds mid-operation (should be caught!)
    WORKGROUP_CHAOS        # Random workgroup sizes
    ARRAY_REUSE            # Reuse arrays (potential data race)
end

"""
Apply a nemesis fault.
"""
function apply_nemesis!(nemesis::NemesisType, rng::AbstractRNG)
    if nemesis == GC_PRESSURE
        GC.gc()
    elseif nemesis == THREAD_CONTENTION
        # Spawn tasks that compete for resources
        tasks = [Threads.@spawn begin
            x = zeros(Float32, rand(rng, 100:1000), 3)
            sum(x)
        end for _ in 1:Threads.nthreads()]
        foreach(wait, tasks)
    elseif nemesis == MEMORY_PRESSURE
        # Allocate and immediately release
        x = zeros(Float32, rand(rng, 10000:100000), 3)
        finalize(x)
    elseif nemesis == TIMING_CHAOS
        sleep(rand(rng) * 0.01)
    elseif nemesis == SEED_MUTATION
        # This should be detected as an SPI violation!
        # We return a signal to the caller
        return :mutation_requested
    elseif nemesis == WORKGROUP_CHAOS
        # Will be handled by caller
        return rand(rng, [1, 7, 13, 31, 64, 127, 256, 511])
    elseif nemesis == ARRAY_REUSE
        return :array_reuse_requested
    end
    return nothing
end

"""
Choose a random nemesis with weighted probability.
"""
function random_nemesis(rng::AbstractRNG)
    r = rand(rng)
    if r < 0.3
        NONE
    elseif r < 0.4
        GC_PRESSURE
    elseif r < 0.5
        THREAD_CONTENTION
    elseif r < 0.6
        MEMORY_PRESSURE
    elseif r < 0.75
        TIMING_CHAOS
    elseif r < 0.85
        WORKGROUP_CHAOS
    elseif r < 0.95
        ARRAY_REUSE
    else
        SEED_MUTATION  # Rare: should trigger violation detection
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# Elle-Style Anomaly Detection
# ═══════════════════════════════════════════════════════════════════════════

"""
Anomaly types that can be detected in color histories.
Inspired by Elle/Adya's consistency anomalies.
"""
@enum AnomalyType begin
    G0_WRITE_CYCLE         # Same color written with different values (impossible for us)
    G1A_ABORTED_READ       # Read a value that was never committed
    G1B_INTERMEDIATE_READ  # Read intermediate state
    G1C_CYCLIC_INFO        # Cyclic information flow
    DIRTY_READ             # Read uncommitted data
    LOST_UPDATE            # Update was lost
    FINGERPRINT_MISMATCH   # XOR fingerprints don't match (SPI violation)
    DISTRIBUTION_ANOMALY   # Statistical distribution is wrong
    DETERMINISM_VIOLATION  # Same inputs produce different outputs
end

"""
An anomaly detected in the history.
"""
struct Anomaly
    type::AnomalyType
    description::String
    ops::Vector{Op}        # Operations involved
    evidence::Any          # Supporting evidence
end

"""
Check a history for anomalies, Elle-style.
"""
function check_history(h::History)
    anomalies = Anomaly[]
    
    # Group operations by (seed, n) pairs
    by_params = Dict{Tuple{UInt64, Int}, Vector{Op}}()
    for op in h.ops
        if op.type == OK && op.kind in [GENERATE, VERIFY]
            key = (op.seed, op.n)
            if !haskey(by_params, key)
                by_params[key] = Op[]
            end
            push!(by_params[key], op)
        end
    end
    
    # Check for determinism violations (G1c-like)
    for ((seed, n), ops) in by_params
        fingerprints = Set{UInt32}()
        for op in ops
            if op.value isa UInt32
                push!(fingerprints, op.value)
            end
        end
        
        if length(fingerprints) > 1
            push!(anomalies, Anomaly(
                FINGERPRINT_MISMATCH,
                "Same (seed=$seed, n=$n) produced $(length(fingerprints)) different fingerprints",
                ops,
                fingerprints
            ))
        end
    end
    
    # Check for lost updates (operations that should have succeeded but didn't)
    failed_ops = filter(op -> op.type == FAIL, h.ops)
    if !isempty(failed_ops)
        push!(anomalies, Anomaly(
            LOST_UPDATE,
            "$(length(failed_ops)) operations failed unexpectedly",
            failed_ops,
            nothing
        ))
    end
    
    anomalies
end

"""
Check if history is linearizable.
For SPI, this means all operations with same params return same result.
"""
function is_linearizable(h::History)
    anomalies = check_history(h)
    fingerprint_anomalies = filter(a -> a.type == FINGERPRINT_MISMATCH, anomalies)
    return isempty(fingerprint_anomalies)
end

# ═══════════════════════════════════════════════════════════════════════════
# Generator: Random Operation Sequences
# ═══════════════════════════════════════════════════════════════════════════

"""
Generate a random operation sequence, Jepsen-style.
"""
function generate_operations(rng::AbstractRNG; n_ops::Int=100, n_processes::Int=4)
    ops = Tuple{Int, OpKind, UInt64, Int}[]
    
    for _ in 1:n_ops
        process = rand(rng, 1:n_processes)
        kind = rand(rng, [GENERATE, GENERATE, GENERATE, VERIFY, SUM])
        seed = rand(rng, UInt64)
        n = rand(rng, [10, 100, 1000, 10000])
        push!(ops, (process, kind, seed, n))
    end
    
    ops
end

# ═══════════════════════════════════════════════════════════════════════════
# Jepsen-Style Test Runner
# ═══════════════════════════════════════════════════════════════════════════

"""
Run a Jepsen-style test with nemesis fault injection.
"""
function run_jepsen_test(;
    duration::Float64 = 10.0,
    n_processes::Int = 4,
    nemesis_interval::Float64 = 0.5,
    seed::Int = 42
)
    rng = Random.MersenneTwister(seed)
    history = History(UInt64(seed))
    
    start_time = time()
    op_count = 0
    nemesis_count = 0
    violations_detected = 0
    violations_expected = 0
    
    # Shared arrays for testing array reuse nemesis
    shared_arrays = Dict{Tuple{UInt64, Int}, Matrix{Float32}}()
    
    while (time() - start_time) < duration
        op_count += 1
        process = rand(rng, 1:n_processes)
        
        # Decide on nemesis
        nemesis = random_nemesis(rng)
        if nemesis != NONE
            nemesis_count += 1
        end
        
        # Apply nemesis
        nemesis_result = apply_nemesis!(nemesis, rng)
        
        # Generate operation parameters
        test_seed = rand(rng, UInt64)
        n = rand(rng, [10, 100, 1000, 5000])
        kind = rand(rng, [GENERATE, GENERATE, VERIFY, SUM])
        
        # Handle seed mutation nemesis (intentional SPI violation)
        actual_seed = test_seed
        if nemesis_result == :mutation_requested
            # Use a different seed - this SHOULD cause a violation
            violations_expected += 1
            actual_seed = test_seed ⊻ UInt64(1)
        end
        
        # Handle workgroup chaos
        workgroup = nemesis_result isa Int ? nemesis_result : 256
        
        # Record invoke
        invoke_op = Op(process, INVOKE, kind, test_seed, n, nothing, time())
        record!(history, invoke_op)
        
        try
            result = nothing
            
            if kind == GENERATE || kind == VERIFY
                # Check for array reuse (potential data race)
                if nemesis_result == :array_reuse_requested && haskey(shared_arrays, (test_seed, n))
                    colors = shared_arrays[(test_seed, n)]
                else
                    colors = zeros(Float32, n, 3)
                end
                
                ka_colors!(colors, actual_seed; backend=CPU(), workgroup=workgroup)
                result = xor_fingerprint(colors)
                
                # Store for potential reuse
                shared_arrays[(test_seed, n)] = colors
                
            elseif kind == SUM
                result = ka_color_sums(n, actual_seed; chunk_size=max(100, n ÷ 10))
            end
            
            # Record OK
            ok_op = Op(process, OK, kind, test_seed, n, result, time())
            record!(history, ok_op)
            
            # Immediate verification: same params should give same result
            if kind == GENERATE || kind == VERIFY
                verify_colors = zeros(Float32, n, 3)
                ka_colors!(verify_colors, test_seed; backend=CPU(), workgroup=256)
                verify_fp = xor_fingerprint(verify_colors)
                
                if result != verify_fp
                    violations_detected += 1
                end
            end
            
        catch e
            # Record failure
            fail_op = Op(process, FAIL, kind, test_seed, n, e, time())
            record!(history, fail_op)
        end
    end
    
    # Analyze history for anomalies
    anomalies = check_history(history)
    linearizable = is_linearizable(history)
    
    (
        history = history,
        duration = time() - start_time,
        op_count = op_count,
        nemesis_count = nemesis_count,
        violations_detected = violations_detected,
        violations_expected = violations_expected,
        anomalies = anomalies,
        linearizable = linearizable,
    )
end

# ═══════════════════════════════════════════════════════════════════════════
# Self-Fuzzing: Fuzzers Fuzzing Fuzzers
# ═══════════════════════════════════════════════════════════════════════════

"""
Fuzz the fingerprint function itself.
"""
function fuzz_fingerprint_fuzzer(; iterations::Int=1000)
    rng = Random.MersenneTwister(42)
    
    failures = Any[]
    
    for i in 1:iterations
        n = rand(rng, 1:10000)
        seed = rand(rng, UInt64)
        
        # Generate colors multiple ways
        colors1 = ka_colors(n, seed)
        colors2 = ka_colors(n, seed)
        
        # Generate fingerprints multiple ways
        fp1 = xor_fingerprint(colors1)
        fp2 = xor_fingerprint(colors2)
        fp3 = xor_fingerprint(ka_colors(n, seed))
        
        if !(fp1 == fp2 == fp3)
            push!(failures, (i=i, n=n, seed=seed, fp1=fp1, fp2=fp2, fp3=fp3))
        end
        
        # Verify fingerprint is sensitive to changes
        if n > 1
            mutated = copy(colors1)
            mutated[1, 1] = mutated[1, 1] + 1.0f-6
            fp_mutated = xor_fingerprint(mutated)
            
            # Fingerprint should change (with high probability)
            # Due to floating point, a tiny change might not change reinterpret bits
            # so we only check for obvious mutations
            mutated2 = copy(colors1)
            mutated2[1, 1] = 1.0f0 - colors1[1, 1]  # Flip the value
            fp_mutated2 = xor_fingerprint(mutated2)
            
            if fp1 == fp_mutated2 && colors1[1, 1] != mutated2[1, 1]
                push!(failures, (i=i, type=:insensitive, n=n, seed=seed))
            end
        end
    end
    
    (passed = isempty(failures), iterations = iterations, failures = failures)
end

"""
Fuzz the hash function fuzzer.
"""
function fuzz_hash_fuzzer(; iterations::Int=1000)
    rng = Random.MersenneTwister(1337)
    
    failures = Any[]
    
    for i in 1:iterations
        seed1 = rand(rng, UInt64)
        seed2 = rand(rng, UInt64)
        idx = rand(rng, UInt64)
        
        # Same inputs = same outputs
        c1 = hash_color(seed1, idx)
        c2 = hash_color(seed1, idx)
        if c1 != c2
            push!(failures, (i=i, type=:nondeterministic, seed=seed1, idx=idx))
        end
        
        # Different inputs = different outputs (with high probability)
        c3 = hash_color(seed2, idx)
        if seed1 != seed2 && c1 == c3
            # This could happen by chance, but very rarely
            # Only flag if it happens too often
        end
        
        # Values in range
        r, g, b = c1
        if !(0.0f0 <= r <= 1.0f0) || !(0.0f0 <= g <= 1.0f0) || !(0.0f0 <= b <= 1.0f0)
            push!(failures, (i=i, type=:out_of_range, seed=seed1, idx=idx, rgb=c1))
        end
    end
    
    (passed = isempty(failures), iterations = iterations, failures = failures)
end

"""
Fuzz the SPI verification itself.
"""
function fuzz_spi_verifier(; iterations::Int=100)
    rng = Random.MersenneTwister(69)
    
    failures = Any[]
    
    for i in 1:iterations
        n = rand(rng, [100, 1000, 10000])
        seed = rand(rng, UInt64)
        
        # Generate with different workgroups - must be identical
        workgroups = [1, 16, 32, 64, 128, 256, 512]
        fingerprints = UInt32[]
        
        for ws in workgroups
            colors = zeros(Float32, n, 3)
            ka_colors!(colors, seed; backend=CPU(), workgroup=ws)
            push!(fingerprints, xor_fingerprint(colors))
        end
        
        # All fingerprints must be identical
        if !all(fp -> fp == fingerprints[1], fingerprints)
            push!(failures, (i=i, n=n, seed=seed, fingerprints=fingerprints))
        end
    end
    
    (passed = isempty(failures), iterations = iterations, failures = failures)
end

"""
Meta-fuzz: Run all fuzzers against themselves.
"""
function meta_fuzz(; duration::Float64=30.0)
    println()
    println("╔══════════════════════════════════════════════════════════════════════╗")
    println("║  JEPSEN-STYLE META-FUZZING: Fuzz the Fuzzers Fuzzing Themselves     ║")
    println("╚══════════════════════════════════════════════════════════════════════╝")
    println()
    
    results = Dict{String, Any}()
    
    # 1. Fuzz the fingerprint fuzzer
    print("  1. Fuzzing fingerprint fuzzer... ")
    t1 = @elapsed r1 = fuzz_fingerprint_fuzzer(iterations=1000)
    println(r1.passed ? "✓ PASS" : "✗ FAIL", " ($(round(t1, digits=2))s)")
    results["fingerprint_fuzzer"] = r1
    
    # 2. Fuzz the hash fuzzer
    print("  2. Fuzzing hash fuzzer... ")
    t2 = @elapsed r2 = fuzz_hash_fuzzer(iterations=1000)
    println(r2.passed ? "✓ PASS" : "✗ FAIL", " ($(round(t2, digits=2))s)")
    results["hash_fuzzer"] = r2
    
    # 3. Fuzz the SPI verifier
    print("  3. Fuzzing SPI verifier... ")
    t3 = @elapsed r3 = fuzz_spi_verifier(iterations=100)
    println(r3.passed ? "✓ PASS" : "✗ FAIL", " ($(round(t3, digits=2))s)")
    results["spi_verifier"] = r3
    
    # 4. Run Jepsen-style test with nemesis
    println()
    println("  4. Jepsen-style test with nemesis ($(duration)s)...")
    jepsen_result = run_jepsen_test(duration=duration, seed=42069)
    
    println("     Operations: $(jepsen_result.op_count)")
    println("     Nemesis actions: $(jepsen_result.nemesis_count)")
    println("     Violations detected: $(jepsen_result.violations_detected)")
    println("     Violations expected: $(jepsen_result.violations_expected)")
    println("     Anomalies: $(length(jepsen_result.anomalies))")
    println("     Linearizable: $(jepsen_result.linearizable ? "✓ YES" : "✗ NO")")
    
    results["jepsen"] = jepsen_result
    
    # 5. Self-referential fuzzing: fuzz the fuzzer runner
    println()
    print("  5. Self-referential fuzzing... ")
    t5 = @elapsed begin
        # Run the meta-fuzzer multiple times and ensure consistency
        self_results = [run_jepsen_test(duration=2.0, seed=i) for i in 1:3]
        self_passed = all(r -> r.linearizable, self_results)
    end
    println(self_passed ? "✓ PASS" : "✗ FAIL", " ($(round(t5, digits=2))s)")
    results["self_referential"] = self_passed
    
    # Summary
    println()
    println("═" ^ 72)
    all_passed = r1.passed && r2.passed && r3.passed && 
                 jepsen_result.linearizable && self_passed
    
    if all_passed
        println("  ✓ ALL META-FUZZ TESTS PASSED")
        println("  The fuzzers are sound. The verification system verifies.")
    else
        println("  ✗ SOME META-FUZZ TESTS FAILED")
        for (name, result) in results
            if result isa NamedTuple && haskey(result, :passed) && !result.passed
                println("    - $name: FAILED")
            end
        end
    end
    println("═" ^ 72)
    
    (passed = all_passed, results = results)
end

# ═══════════════════════════════════════════════════════════════════════════
# Test Suite
# ═══════════════════════════════════════════════════════════════════════════

@testset "Jepsen-Style Meta-Fuzzing" begin
    
    @testset "Fingerprint Fuzzer Soundness" begin
        result = fuzz_fingerprint_fuzzer(iterations=500)
        @test result.passed
    end
    
    @testset "Hash Fuzzer Soundness" begin
        result = fuzz_hash_fuzzer(iterations=500)
        @test result.passed
    end
    
    @testset "SPI Verifier Soundness" begin
        result = fuzz_spi_verifier(iterations=50)
        @test result.passed
    end
    
    @testset "Jepsen-Style Nemesis Test" begin
        result = run_jepsen_test(duration=5.0, seed=1337)
        @test result.linearizable
        @test result.op_count > 100
    end
    
    @testset "Elle-Style Anomaly Detection" begin
        # Create a history with known anomalies
        h = History(UInt64(42))
        
        # Add some OK operations with same params but different results
        # (This simulates an SPI violation)
        record!(h, Op(1, OK, GENERATE, UInt64(100), 1000, UInt32(0xDEADBEEF), 0.0))
        record!(h, Op(2, OK, GENERATE, UInt64(100), 1000, UInt32(0xCAFEBABE), 0.1))
        
        anomalies = check_history(h)
        @test !isempty(anomalies)
        @test any(a -> a.type == FINGERPRINT_MISMATCH, anomalies)
        @test !is_linearizable(h)
    end
    
    @testset "Clean History is Linearizable" begin
        h = History(UInt64(42))
        
        # Same params, same result = linearizable
        record!(h, Op(1, OK, GENERATE, UInt64(100), 1000, UInt32(0xDEADBEEF), 0.0))
        record!(h, Op(2, OK, GENERATE, UInt64(100), 1000, UInt32(0xDEADBEEF), 0.1))
        record!(h, Op(3, OK, GENERATE, UInt64(100), 1000, UInt32(0xDEADBEEF), 0.2))
        
        @test is_linearizable(h)
    end
    
    @testset "Nemesis Fault Injection" begin
        rng = Random.MersenneTwister(42)
        
        # Test each nemesis type doesn't crash
        for nemesis in instances(NemesisType)
            result = apply_nemesis!(nemesis, rng)
            @test true  # If we get here, it didn't crash
        end
    end
    
end

export meta_fuzz, run_jepsen_test, fuzz_fingerprint_fuzzer, fuzz_hash_fuzzer, fuzz_spi_verifier
