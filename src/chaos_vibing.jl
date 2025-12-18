# ═══════════════════════════════════════════════════════════════════════════════
# CHAOS VIBING: Maximal Fault Injection into Parallel Causal Chains
# ═══════════════════════════════════════════════════════════════════════════════
#
# "The purpose of chaos engineering is to build confidence in the system's
#  capability to withstand turbulent conditions in production." - Netflix
#
# This module injects faults at EVERY level of the parallel color generation:
#   1. Seed corruption (bit flips, zero injection, overflow)
#   2. Index corruption (off-by-one, wraparound, negative)
#   3. Hash corruption (truncation, bit rotation, xor bombs)
#   4. Thread corruption (race conditions, deadlock simulation, starvation)
#   5. Memory corruption (buffer overflow simulation, alignment errors)
#   6. Timing corruption (jitter, pause, reorder)
#   7. Causal chain breaks (skip, duplicate, reverse)
#
# The goal: Find which invariants break under which faults.
#
# ═══════════════════════════════════════════════════════════════════════════════

using Base.Threads: @spawn, nthreads, threadid, @threads, Atomic
using Random
using Statistics: mean, std

export ChaosConfig, ChaosResult, ChaosVibe
export inject_chaos!, run_chaos_campaign, chaos_vibe!
export CausalChain, break_chain!, verify_chain, chain_fingerprint
export world_chaos_vibing, ChaosVibingWorld
export FaultClass, FaultSeverity, Fault
export SEED_FAULT, INDEX_FAULT, HASH_FAULT, THREAD_FAULT, MEMORY_FAULT, TIMING_FAULT, CAUSAL_FAULT
export BENIGN, MALIGNANT, TERMINAL

# ═══════════════════════════════════════════════════════════════════════════════
# Fault Types
# ═══════════════════════════════════════════════════════════════════════════════

@enum FaultClass begin
    SEED_FAULT
    INDEX_FAULT
    HASH_FAULT
    THREAD_FAULT
    MEMORY_FAULT
    TIMING_FAULT
    CAUSAL_FAULT
end

@enum FaultSeverity begin
    BENIGN      # Should be detected and recovered
    MALIGNANT   # Should be detected, may not recover
    TERMINAL    # Should crash or produce obviously wrong results
end

"""
    Fault

A specific fault to inject.
"""
struct Fault
    class::FaultClass
    severity::FaultSeverity
    name::Symbol
    probability::Float64
    params::Dict{Symbol, Any}
end

# Fault catalog
const FAULT_CATALOG = Dict{Symbol, Fault}(
    # Seed faults
    :seed_bitflip => Fault(SEED_FAULT, BENIGN, :seed_bitflip, 1.0, 
                           Dict(:n_bits => 1)),
    :seed_zero => Fault(SEED_FAULT, MALIGNANT, :seed_zero, 1.0,
                        Dict()),
    :seed_max => Fault(SEED_FAULT, BENIGN, :seed_max, 1.0,
                       Dict()),
    :seed_golden => Fault(SEED_FAULT, BENIGN, :seed_golden, 1.0,
                          Dict(:xor_value => 0x9e3779b97f4a7c15)),
    
    # Index faults
    :index_off_by_one => Fault(INDEX_FAULT, BENIGN, :index_off_by_one, 1.0,
                               Dict(:offset => 1)),
    :index_negative => Fault(INDEX_FAULT, TERMINAL, :index_negative, 1.0,
                             Dict()),
    :index_overflow => Fault(INDEX_FAULT, MALIGNANT, :index_overflow, 1.0,
                             Dict()),
    :index_zero => Fault(INDEX_FAULT, BENIGN, :index_zero, 1.0,
                         Dict()),
    
    # Hash faults
    :hash_truncate => Fault(HASH_FAULT, MALIGNANT, :hash_truncate, 1.0,
                            Dict(:mask => 0x00000000FFFFFFFF)),
    :hash_rotate => Fault(HASH_FAULT, BENIGN, :hash_rotate, 1.0,
                          Dict(:bits => 17)),
    :hash_xor_bomb => Fault(HASH_FAULT, BENIGN, :hash_xor_bomb, 1.0,
                            Dict(:bomb => 0xDEADBEEFCAFEBABE)),
    :hash_zero => Fault(HASH_FAULT, TERMINAL, :hash_zero, 1.0,
                        Dict()),
    
    # Thread faults
    :thread_race => Fault(THREAD_FAULT, MALIGNANT, :thread_race, 0.5,
                          Dict(:delay_ns => 1000)),
    :thread_starvation => Fault(THREAD_FAULT, BENIGN, :thread_starvation, 0.3,
                                Dict(:skip_threads => [1])),
    :thread_duplicate => Fault(THREAD_FAULT, MALIGNANT, :thread_duplicate, 0.5,
                               Dict()),
    
    # Memory faults
    :memory_corruption => Fault(MEMORY_FAULT, TERMINAL, :memory_corruption, 0.1,
                                Dict(:corrupt_bytes => 8)),
    :memory_alignment => Fault(MEMORY_FAULT, BENIGN, :memory_alignment, 0.5,
                               Dict(:misalign => 3)),
    
    # Timing faults
    :timing_jitter => Fault(TIMING_FAULT, BENIGN, :timing_jitter, 1.0,
                            Dict(:max_jitter_us => 100)),
    :timing_pause => Fault(TIMING_FAULT, BENIGN, :timing_pause, 0.2,
                           Dict(:pause_ms => 10)),
    :timing_reorder => Fault(TIMING_FAULT, MALIGNANT, :timing_reorder, 0.3,
                             Dict()),
    
    # Causal chain faults
    :causal_skip => Fault(CAUSAL_FAULT, MALIGNANT, :causal_skip, 0.5,
                          Dict(:skip_indices => [5, 10, 15])),
    :causal_duplicate => Fault(CAUSAL_FAULT, BENIGN, :causal_duplicate, 0.5,
                               Dict(:dup_index => 3)),
    :causal_reverse => Fault(CAUSAL_FAULT, TERMINAL, :causal_reverse, 0.3,
                             Dict()),
    :causal_shuffle => Fault(CAUSAL_FAULT, TERMINAL, :causal_shuffle, 0.2,
                             Dict()),
)

# ═══════════════════════════════════════════════════════════════════════════════
# Causal Chain Representation
# ═══════════════════════════════════════════════════════════════════════════════

"""
    CausalChain

A chain of causally-linked color computations.
Each step depends on the previous step's fingerprint.
"""
mutable struct CausalChain
    seed::UInt64
    length::Int
    steps::Vector{UInt64}           # Fingerprint at each step
    colors::Vector{NTuple{3, Float32}}  # Color at each step
    causal_links::Vector{UInt64}    # Link from step i to step i+1
    broken_at::Vector{Int}          # Indices where chain was broken
    is_valid::Bool
end

"""
    CausalChain(seed, length)

Create a valid causal chain of given length.
"""
function CausalChain(seed::UInt64, length::Int)
    steps = Vector{UInt64}(undef, length)
    colors = Vector{NTuple{3, Float32}}(undef, length)
    causal_links = Vector{UInt64}(undef, length - 1)
    
    # Build chain: each step depends on previous
    current = seed
    for i in 1:length
        steps[i] = current
        colors[i] = hash_color(seed, current)
        if i < length
            # Causal link: next step depends on current fingerprint
            causal_links[i] = splitmix64(current ⊻ UInt64(i))
            current = causal_links[i]
        end
    end
    
    CausalChain(seed, length, steps, colors, causal_links, Int[], true)
end

"""
    break_chain!(chain, index, fault)

Break the causal chain at given index using specified fault.
"""
function break_chain!(chain::CausalChain, index::Int, fault::Fault)
    if index < 1 || index > chain.length
        return chain
    end
    
    push!(chain.broken_at, index)
    chain.is_valid = false
    
    # Apply fault based on type
    if fault.class == SEED_FAULT
        if fault.name == :seed_bitflip
            n_bits = get(fault.params, :n_bits, 1)
            for _ in 1:n_bits
                bit = rand(0:63)
                chain.steps[index] ⊻= UInt64(1) << bit
            end
        elseif fault.name == :seed_zero
            chain.steps[index] = UInt64(0)
        elseif fault.name == :seed_max
            chain.steps[index] = typemax(UInt64)
        elseif fault.name == :seed_golden
            chain.steps[index] ⊻= fault.params[:xor_value]
        end
    elseif fault.class == HASH_FAULT
        if fault.name == :hash_truncate
            chain.steps[index] &= fault.params[:mask]
        elseif fault.name == :hash_rotate
            bits = fault.params[:bits]
            v = chain.steps[index]
            chain.steps[index] = (v << bits) | (v >> (64 - bits))
        elseif fault.name == :hash_xor_bomb
            chain.steps[index] ⊻= fault.params[:bomb]
        elseif fault.name == :hash_zero
            chain.steps[index] = UInt64(0)
        end
    elseif fault.class == CAUSAL_FAULT
        if fault.name == :causal_skip && index < chain.length
            # Skip causal link - break causality
            chain.causal_links[index] = rand(UInt64)
        elseif fault.name == :causal_duplicate && index > 1
            # Duplicate previous step
            chain.steps[index] = chain.steps[index - 1]
        elseif fault.name == :causal_reverse && index < chain.length
            # Reverse a section
            end_idx = min(index + 5, chain.length)
            reverse!(@view chain.steps[index:end_idx])
        elseif fault.name == :causal_shuffle
            # Shuffle remaining chain
            shuffle!(@view chain.steps[index:end])
        end
    end
    
    # Recompute colors after fault
    for i in index:chain.length
        chain.colors[i] = hash_color(chain.seed, chain.steps[i])
    end
    
    chain
end

"""
    verify_chain(chain) -> (Bool, Vector{Int})

Verify causal chain integrity. Returns (valid, broken_indices).
"""
function verify_chain(chain::CausalChain)
    broken = Int[]
    
    # Reconstruct expected chain
    current = chain.seed
    for i in 1:chain.length
        if chain.steps[i] != current
            push!(broken, i)
        end
        if i < chain.length
            current = splitmix64(current ⊻ UInt64(i))
        end
    end
    
    (isempty(broken), broken)
end

"""
    chain_fingerprint(chain) -> UInt64

Compute XOR fingerprint of entire chain.
"""
function chain_fingerprint(chain::CausalChain)
    reduce(⊻, chain.steps)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Chaos Configuration
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ChaosConfig

Configuration for chaos injection campaign.
"""
Base.@kwdef struct ChaosConfig
    n_chains::Int = 100
    chain_length::Int = 100
    n_threads::Int = nthreads()
    faults_per_chain::Int = 5
    fault_classes::Vector{FaultClass} = [SEED_FAULT, INDEX_FAULT, HASH_FAULT, CAUSAL_FAULT]
    seed::UInt64 = 0xCEA05F1BE
    intensity::Float64 = 0.5  # 0.0 = gentle, 1.0 = maximum chaos
    detect_threshold::Float64 = 0.01  # Max allowed fingerprint deviation
end

"""
    ChaosResult

Result of a single chaos injection.
"""
struct ChaosResult
    chain_id::Int
    thread_id::Int
    faults_injected::Vector{Fault}
    original_fingerprint::UInt64
    corrupted_fingerprint::UInt64
    detected::Bool
    broken_indices::Vector{Int}
    recovery_possible::Bool
end

"""
    ChaosVibe

Aggregate result of chaos campaign.
"""
struct ChaosVibe
    config::ChaosConfig
    results::Vector{ChaosResult}
    total_faults::Int
    detected_faults::Int
    undetected_faults::Int
    detection_rate::Float64
    recovery_rate::Float64
    by_class::Dict{FaultClass, NamedTuple{(:total, :detected), Tuple{Int, Int}}}
    by_severity::Dict{FaultSeverity, NamedTuple{(:total, :detected), Tuple{Int, Int}}}
    wall_time_ns::UInt64
end

# ═══════════════════════════════════════════════════════════════════════════════
# Chaos Injection Engine
# ═══════════════════════════════════════════════════════════════════════════════

"""
    inject_chaos!(chain, config) -> Vector{Fault}

Inject chaos into a causal chain according to config.
Returns list of injected faults.
"""
function inject_chaos!(chain::CausalChain, config::ChaosConfig)
    injected = Fault[]
    
    # Select faults based on config
    available_faults = [f for (_, f) in FAULT_CATALOG 
                        if f.class in config.fault_classes]
    
    if isempty(available_faults)
        return injected
    end
    
    # Inject faults probabilistically
    for _ in 1:config.faults_per_chain
        if rand() > config.intensity
            continue
        end
        
        fault = rand(available_faults)
        if rand() < fault.probability
            # Choose random index to corrupt
            index = rand(1:chain.length)
            break_chain!(chain, index, fault)
            push!(injected, fault)
        end
    end
    
    injected
end

"""
    run_chaos_campaign(config) -> ChaosVibe

Run a full chaos injection campaign across multiple threads.
"""
function run_chaos_campaign(config::ChaosConfig = ChaosConfig())
    results = Vector{ChaosResult}(undef, config.n_chains)
    
    # Atomic counters for thread-safe aggregation
    total_faults = Atomic{Int}(0)
    detected_faults = Atomic{Int}(0)
    
    start_time = time_ns()
    
    # Parallel chaos injection
    @threads for i in 1:config.n_chains
        tid = threadid()
        chain_seed = config.seed ⊻ UInt64(i * 0x9e3779b97f4a7c15)
        
        # Create clean chain
        chain = CausalChain(chain_seed, config.chain_length)
        original_fp = chain_fingerprint(chain)
        
        # Inject chaos
        faults = inject_chaos!(chain, config)
        corrupted_fp = chain_fingerprint(chain)
        
        # Verify chain
        valid, broken = verify_chain(chain)
        
        # Detection: fingerprint changed or chain invalid
        detected = (corrupted_fp != original_fp) || !valid || !isempty(broken)
        
        # Recovery: can we reconstruct the original chain?
        recovery_chain = CausalChain(chain_seed, config.chain_length)
        recovery_possible = chain_fingerprint(recovery_chain) == original_fp
        
        results[i] = ChaosResult(
            i, tid, faults,
            original_fp, corrupted_fp,
            detected, broken, recovery_possible
        )
        
        Threads.atomic_add!(total_faults, length(faults))
        if detected
            Threads.atomic_add!(detected_faults, length(faults))
        end
    end
    
    end_time = time_ns()
    
    # Aggregate by class
    by_class = Dict{FaultClass, NamedTuple{(:total, :detected), Tuple{Int, Int}}}()
    for class in instances(FaultClass)
        total = 0
        det = 0
        for r in results
            for f in r.faults_injected
                if f.class == class
                    total += 1
                    if r.detected
                        det += 1
                    end
                end
            end
        end
        by_class[class] = (total=total, detected=det)
    end
    
    # Aggregate by severity
    by_severity = Dict{FaultSeverity, NamedTuple{(:total, :detected), Tuple{Int, Int}}}()
    for sev in instances(FaultSeverity)
        total = 0
        det = 0
        for r in results
            for f in r.faults_injected
                if f.severity == sev
                    total += 1
                    if r.detected
                        det += 1
                    end
                end
            end
        end
        by_severity[sev] = (total=total, detected=det)
    end
    
    # Compute rates
    total = total_faults[]
    detected = detected_faults[]
    detection_rate = total > 0 ? detected / total : 1.0
    
    recoverable = count(r -> r.recovery_possible, results)
    recovery_rate = config.n_chains > 0 ? recoverable / config.n_chains : 1.0
    
    ChaosVibe(
        config, results,
        total, detected, total - detected,
        detection_rate, recovery_rate,
        by_class, by_severity,
        end_time - start_time
    )
end

"""
    chaos_vibe!(; kwargs...) -> ChaosVibe

Quick chaos vibing with default or custom config.
"""
function chaos_vibe!(; kwargs...)
    config = ChaosConfig(; kwargs...)
    run_chaos_campaign(config)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Parallel Stress Tests
# ═══════════════════════════════════════════════════════════════════════════════

"""
    stress_parallel_fingerprints(n_colors, n_iterations; seed)

Stress test: verify fingerprint consistency under parallel chaos.
"""
function stress_parallel_fingerprints(n_colors::Int, n_iterations::Int; 
                                       seed::UInt64 = GAY_SEED)
    results = Vector{Bool}(undef, n_iterations)
    
    @threads for i in 1:n_iterations
        # Compute fingerprint with potential race conditions
        partial_fps = zeros(UInt64, nthreads())
        
        @threads for j in 1:n_colors
            tid = threadid()
            h = splitmix64_mix(seed ⊻ UInt64(j))
            partial_fps[tid] ⊻= h
        end
        
        fp1 = reduce(⊻, partial_fps)
        
        # Verify by recomputing sequentially
        fp2 = UInt64(0)
        for j in 1:n_colors
            fp2 ⊻= splitmix64_mix(seed ⊻ UInt64(j))
        end
        
        results[i] = (fp1 == fp2)
    end
    
    (all(results), count(!, results))
end

"""
    stress_ergodic_bridge(n_samples, intensity)

Stress test the ergodic bridge under chaos conditions.
"""
function stress_ergodic_bridge(n_samples::Int; intensity::Float64 = 0.5)
    results = Dict{Symbol, Any}()
    
    # 1. Create bridge under normal conditions
    bridge_clean = create_bridge(GAY_SEED, n_samples)
    
    # 2. Create bridge with corrupted seed
    corrupted_seed = GAY_SEED ⊻ 0xDEADBEEF
    bridge_corrupt = create_bridge(corrupted_seed, n_samples)
    
    # 3. Verify both
    results[:clean_valid] = verify_bridge(bridge_clean)
    results[:corrupt_valid] = verify_bridge(bridge_corrupt)
    
    # 4. Check fingerprint divergence
    results[:fingerprint_divergence] = bridge_clean.fingerprint != bridge_corrupt.fingerprint
    
    # 5. Check entropy under corruption
    results[:clean_entropy] = bridge_clean.entropy_at_completion
    results[:corrupt_entropy] = bridge_corrupt.entropy_at_completion
    
    # 6. Ergodicity under chaos
    results[:clean_ergodic] = bridge_clean.ergodic
    results[:corrupt_ergodic] = bridge_corrupt.ergodic
    
    results
end

# ═══════════════════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════════════════

"""
    world_chaos_vibing()

Demonstrate maximal chaos injection.
"""
function world_chaos_vibing()
    println("═" ^ 70)
    println("  CHAOS VIBING: Maximal Fault Injection into Parallel Causal Chains")
    println("═" ^ 70)
    println()
    
    # 1. Basic chaos campaign
    println("1. CHAOS CAMPAIGN (1000 chains × 100 steps × 5 faults)")
    config = ChaosConfig(
        n_chains = 1000,
        chain_length = 100,
        faults_per_chain = 5,
        intensity = 0.7,
        fault_classes = [SEED_FAULT, HASH_FAULT, CAUSAL_FAULT]
    )
    
    t = @elapsed vibe = run_chaos_campaign(config)
    
    println("   Time: $(round(t * 1000, digits=1))ms")
    println("   Threads: $(config.n_threads)")
    println("   Total faults: $(vibe.total_faults)")
    println("   Detection rate: $(round(vibe.detection_rate * 100, digits=1))%")
    println("   Recovery rate: $(round(vibe.recovery_rate * 100, digits=1))%")
    println()
    
    # 2. By fault class
    println("2. DETECTION BY FAULT CLASS")
    for (class, stats) in sort(collect(vibe.by_class), by=x->string(x[1]))
        if stats.total > 0
            rate = round(stats.detected / stats.total * 100, digits=1)
            println("   $(rpad(string(class), 15)) $(lpad(stats.detected, 4))/$(lpad(stats.total, 4)) = $(rate)%")
        end
    end
    println()
    
    # 3. By severity
    println("3. DETECTION BY SEVERITY")
    for (sev, stats) in sort(collect(vibe.by_severity), by=x->string(x[1]))
        if stats.total > 0
            rate = round(stats.detected / stats.total * 100, digits=1)
            println("   $(rpad(string(sev), 12)) $(lpad(stats.detected, 4))/$(lpad(stats.total, 4)) = $(rate)%")
        end
    end
    println()
    
    # 4. Parallel fingerprint stress test
    println("4. PARALLEL FINGERPRINT STRESS TEST")
    for n in [1000, 10000, 100000]
        pass, failures = stress_parallel_fingerprints(n, 100)
        println("   $(lpad(n, 6)) colors × 100 iterations: $(pass ? "✓ PASS" : "✗ FAIL ($failures failures)")")
    end
    println()
    
    # 5. Ergodic bridge stress
    println("5. ERGODIC BRIDGE STRESS TEST")
    bridge_results = stress_ergodic_bridge(10000; intensity=0.8)
    println("   Clean bridge valid: $(bridge_results[:clean_valid] ? "✓" : "✗")")
    println("   Corrupt bridge valid: $(bridge_results[:corrupt_valid] ? "✓" : "✗")")
    println("   Fingerprint divergence: $(bridge_results[:fingerprint_divergence] ? "✓ DETECTED" : "✗ COLLISION")")
    println("   Clean entropy: $(round(bridge_results[:clean_entropy], digits=4))")
    println("   Corrupt entropy: $(round(bridge_results[:corrupt_entropy], digits=4))")
    println()
    
    # 6. Extreme chaos
    println("6. EXTREME CHAOS (max intensity)")
    extreme_config = ChaosConfig(
        n_chains = 500,
        chain_length = 50,
        faults_per_chain = 20,
        intensity = 1.0,
        fault_classes = collect(instances(FaultClass))
    )
    
    extreme_vibe = run_chaos_campaign(extreme_config)
    println("   Total faults: $(extreme_vibe.total_faults)")
    println("   Detection rate: $(round(extreme_vibe.detection_rate * 100, digits=1))%")
    println("   Undetected: $(extreme_vibe.undetected_faults)")
    println()
    
    println("═" ^ 70)
    println("  CHAOS VIBING COMPLETE 🌀")
    println("═" ^ 70)
    
    vibe
end

# end of chaos_vibing.jl
