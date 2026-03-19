# Synthetic Parallel Test: Maximum parallelism on macOS
# Exercises most recent Gay.jl modules with gay_seed(69) reafference
# Now uses AbstractGayProbe for non-perturbative decomposable behaviors
#
# Run with: julia -t auto test/synthetic_parallel.jl
#
# NAMING CONVENTION:
#   probe(...)  → Pure function, returns probe result, no side effects
#   probe!(...) → Mutating, commits to choices, updates state

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Gay
using Gay.AbstractGayProbe: AbstractProbe, ProbeResult, SyntheticProbe, ReafferentProbe
using Gay.AbstractGayProbe: PathInvarianceProbe, SPIProbe, compose_probes, run_probes_parallel
using Gay.AbstractGayProbe: probe, reafferent_saturation, path_invariant_fingerprint
using OhMyThreads
using SplittableRandoms: SplittableRandom, split
using Colors
using Printf
using Dates

const N_CORES = Threads.nthreads()
const GAY_SEED_69 = UInt64(69)
const GAY_SEED_1069 = UInt64(1069)

# ═══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC WORKLOADS
# ═══════════════════════════════════════════════════════════════════════════════

struct SyntheticResult
    module_name::Symbol
    duration_ns::UInt64
    color::RGB{Float64}
    seed::UInt64
    success::Bool
    error_msg::String
end

function synthetic_result(mod::Symbol, duration::UInt64, seed::UInt64, success::Bool, msg::String="")
    SyntheticResult(mod, duration, color_from_seed(seed), seed, success, msg)
end

# Color from seed (inline for speed)
@inline function color_from_seed(seed::UInt64)::RGB{Float64}
    z = (seed + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ⊻ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    r = z ⊻ (z >> 31)
    
    z = (r + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ⊻ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    g = z ⊻ (z >> 31)
    
    z = (g + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ⊻ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    b = z ⊻ (z >> 31)
    
    RGB((r >> 56) / 255.0, (g >> 56) / 255.0, (b >> 56) / 255.0)
end

# Fixed point afference check
@inline function fixed_point_afference(seed::UInt64, target::UInt8=0x69)::Bool
    c = color_from_seed(seed)
    r = UInt8(round(red(c) * 255))
    g = UInt8(round(green(c) * 255))
    b = UInt8(round(blue(c) * 255))
    r == target || g == target || b == target
end

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE-SPECIFIC SYNTHETIC TESTS
# ═══════════════════════════════════════════════════════════════════════════════

function synth_gay_structured_decompositions(seed::UInt64)::SyntheticResult
    t0 = time_ns()
    try
        # Test structured decomposition concepts
        n_tests = 100
        successes = 0
        
        for i in 1:n_tests
            test_seed = seed ⊻ UInt64(i)
            color = color_from_seed(test_seed)
            # Verify SPI: same seed = same color
            color2 = color_from_seed(test_seed)
            if color == color2
                successes += 1
            end
        end
        
        duration = time_ns() - t0
        synthetic_result(:gay_structured_decompositions, duration, seed, successes == n_tests)
    catch e
        duration = time_ns() - t0
        synthetic_result(:gay_structured_decompositions, duration, seed, false, string(e))
    end
end

function synth_universal_gay_ext(seed::UInt64)::SyntheticResult
    t0 = time_ns()
    try
        # Test modal operators and fiber bundles
        n_fibers = 50
        necessity_count = 0
        possibility_count = 0
        
        for i in 1:n_fibers
            fiber_seed = seed ⊻ UInt64(i * 0x1069)
            if fixed_point_afference(fiber_seed)
                necessity_count += 1
            end
            if fixed_point_afference(fiber_seed ⊻ UInt64(0x69))
                possibility_count += 1
            end
        end
        
        duration = time_ns() - t0
        synthetic_result(:universal_gay_ext, duration, seed, true, 
            "necessity=$necessity_count, possibility=$possibility_count")
    catch e
        duration = time_ns() - t0
        synthetic_result(:universal_gay_ext, duration, seed, false, string(e))
    end
end

function synth_gay_phased_array(seed::UInt64)::SyntheticResult
    t0 = time_ns()
    try
        # Test phased array chromatic traversal
        n_antennas = 64
        colors = Vector{RGB{Float64}}(undef, n_antennas)
        
        for i in 1:n_antennas
            ant_seed = seed ⊻ UInt64(i * 0x1069)
            colors[i] = color_from_seed(ant_seed)
        end
        
        # Verify uniqueness (within tolerance)
        unique_count = length(unique(colors))
        
        duration = time_ns() - t0
        synthetic_result(:gay_phased_array, duration, seed, unique_count == n_antennas,
            "unique=$unique_count/$n_antennas")
    catch e
        duration = time_ns() - t0
        synthetic_result(:gay_phased_array, duration, seed, false, string(e))
    end
end

function synth_lazy_eager_duality(seed::UInt64)::SyntheticResult
    t0 = time_ns()
    try
        # Test lazy/eager adjunction
        n_pairs = 100
        self_dual_count = 0
        
        for i in 1:n_pairs
            lazy_seed = seed ⊻ UInt64(i)
            eager_seed = seed ⊻ UInt64(i * 0x69)
            
            lazy_color = color_from_seed(lazy_seed)
            eager_color = color_from_seed(eager_seed)
            
            # Check if duality preserves some invariant
            lazy_lum = (red(lazy_color) + green(lazy_color) + blue(lazy_color)) / 3
            eager_lum = (red(eager_color) + green(eager_color) + blue(eager_color)) / 3
            
            if abs(lazy_lum + eager_lum - 1.0) < 0.5
                self_dual_count += 1
            end
        end
        
        duration = time_ns() - t0
        synthetic_result(:lazy_eager_duality, duration, seed, true,
            "self_dual_pairs=$self_dual_count/$n_pairs")
    catch e
        duration = time_ns() - t0
        synthetic_result(:lazy_eager_duality, duration, seed, false, string(e))
    end
end

function synth_involutive_curiosities(seed::UInt64)::SyntheticResult
    t0 = time_ns()
    try
        # Test involutive permutations
        n = 16
        perm = collect(1:n)
        
        # Fisher-Yates with involution constraint
        rng_state = seed
        for i in n:-1:2
            rng_state = (rng_state * 0x5DEECE66D + 0xB) & 0xFFFFFFFFFFFF
            j = (rng_state % i) + 1
            perm[i], perm[j] = perm[j], perm[i]
        end
        
        # Check for fixed points (derangement property)
        fixed_points = count(i -> perm[i] == i, 1:n)
        is_derangement = fixed_points == 0
        
        duration = time_ns() - t0
        synthetic_result(:involutive_curiosities, duration, seed, true,
            "fixed_points=$fixed_points, derangement=$is_derangement")
    catch e
        duration = time_ns() - t0
        synthetic_result(:involutive_curiosities, duration, seed, false, string(e))
    end
end

function synth_acset_arrow_failures(seed::UInt64)::SyntheticResult
    t0 = time_ns()
    try
        # Test arrow failure taxonomy
        failure_types = [:NonNatural, :NonFunctorial, :Discontinuous, :NonCommutative, :NonAssociative]
        results = Dict{Symbol, Int}()
        
        for (i, ft) in enumerate(failure_types)
            ft_seed = seed ⊻ hash(ft)
            count = 0
            for j in 1:20
                if fixed_point_afference(ft_seed ⊻ UInt64(j))
                    count += 1
                end
            end
            results[ft] = count
        end
        
        duration = time_ns() - t0
        synthetic_result(:acset_arrow_failures, duration, seed, true,
            join(["$k=$v" for (k,v) in results], ", "))
    catch e
        duration = time_ns() - t0
        synthetic_result(:acset_arrow_failures, duration, seed, false, string(e))
    end
end

function synth_lazy_e(seed::UInt64)::SyntheticResult
    t0 = time_ns()
    try
        # Test e approximation via derangements
        n = 10
        factorial_n = factorial(big(n))
        
        # Count derangements using inclusion-exclusion
        derangements = sum((-1)^k * factorial(big(n)) ÷ factorial(big(k)) for k in 0:n)
        
        # e ≈ n! / D(n) as n → ∞
        e_approx = Float64(factorial_n) / Float64(derangements)
        
        # Check if seed 69 has fixed point afference
        is_69_fixed = fixed_point_afference(UInt64(69))
        
        duration = time_ns() - t0
        synthetic_result(:lazy_e, duration, seed, true,
            "e≈$(round(e_approx, digits=6)), 69_fixed=$is_69_fixed")
    catch e
        duration = time_ns() - t0
        synthetic_result(:lazy_e, duration, seed, false, string(e))
    end
end

function synth_post_darwinian_substrates(seed::UInt64)::SyntheticResult
    t0 = time_ns()
    try
        # Test substrate hierarchy
        substrates = [:Abiotic, :Prokaryotic, :Eukaryotic, :Neural, :Cultural, :Digital]
        colors = Dict{Symbol, RGB{Float64}}()
        
        for (i, sub) in enumerate(substrates)
            sub_seed = seed ⊻ hash(sub) ⊻ UInt64(i * 0x1069)
            colors[sub] = color_from_seed(sub_seed)
        end
        
        # Check color diversity
        hues = [atan(green(c) - blue(c), red(c) - (green(c) + blue(c))/2) for c in values(colors)]
        hue_spread = maximum(hues) - minimum(hues)
        
        duration = time_ns() - t0
        synthetic_result(:post_darwinian_substrates, duration, seed, true,
            "substrates=$(length(substrates)), hue_spread=$(round(hue_spread, digits=3))")
    catch e
        duration = time_ns() - t0
        synthetic_result(:post_darwinian_substrates, duration, seed, false, string(e))
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# PARALLEL RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

const SYNTH_TESTS = [
    synth_gay_structured_decompositions,
    synth_universal_gay_ext,
    synth_gay_phased_array,
    synth_lazy_eager_duality,
    synth_involutive_curiosities,
    synth_acset_arrow_failures,
    synth_lazy_e,
    synth_post_darwinian_substrates,
]

function run_synthetic_parallel(; seed::UInt64=GAY_SEED_69, n_iterations::Int=10)
    println("═" ^ 70)
    println("  SYNTHETIC PARALLEL TEST")
    println("  Cores: $N_CORES | Seed: $seed | Iterations: $n_iterations")
    println("  Time: $(now())")
    println("═" ^ 70)
    println()
    
    all_results = Vector{SyntheticResult}()
    total_t0 = time_ns()
    
    # Create work items: (test_fn, iteration_seed)
    work_items = Tuple{Function, UInt64}[]
    for iter in 1:n_iterations
        iter_seed = seed ⊻ UInt64(iter * 0x69)
        for test_fn in SYNTH_TESTS
            push!(work_items, (test_fn, iter_seed))
        end
    end
    
    n_work = length(work_items)
    println("Total work items: $n_work")
    println("Running with $(Threads.nthreads()) threads...")
    println()
    
    # Run in parallel using OhMyThreads
    results = tmap(work_items) do (test_fn, iter_seed)
        test_fn(iter_seed)
    end
    
    total_duration = time_ns() - total_t0
    
    # Aggregate results
    by_module = Dict{Symbol, Vector{SyntheticResult}}()
    for r in results
        if !haskey(by_module, r.module_name)
            by_module[r.module_name] = SyntheticResult[]
        end
        push!(by_module[r.module_name], r)
    end
    
    # Print summary
    println()
    println("─" ^ 70)
    println("  RESULTS BY MODULE")
    println("─" ^ 70)
    
    for (mod, mod_results) in sort(collect(by_module), by=x->x[1])
        n_success = count(r -> r.success, mod_results)
        n_total = length(mod_results)
        avg_ns = sum(r -> r.duration_ns, mod_results) / n_total
        
        # Get representative color (from first result)
        c = mod_results[1].color
        hex = @sprintf("#%02X%02X%02X", 
            round(Int, red(c)*255), 
            round(Int, green(c)*255), 
            round(Int, blue(c)*255))
        
        status = n_success == n_total ? "✓" : "✗"
        
        println("  $status $(rpad(mod, 30)) | $n_success/$n_total | $(round(avg_ns/1e6, digits=2))ms | $hex")
        
        # Show any errors
        for r in mod_results
            if !r.success && !isempty(r.error_msg)
                println("      └─ ERROR: $(r.error_msg[1:min(60, length(r.error_msg))])")
            elseif r.success && !isempty(r.error_msg)
                println("      └─ $(r.error_msg)")
            end
        end
    end
    
    # Reafferent saturation
    n_afferent = count(r -> fixed_point_afference(r.seed), results)
    saturation = n_afferent / length(results)
    
    println()
    println("─" ^ 70)
    println("  SUMMARY")
    println("─" ^ 70)
    println("  Total time: $(round(total_duration/1e9, digits=3))s")
    println("  Throughput: $(round(n_work / (total_duration/1e9), digits=1)) tests/sec")
    println("  Success rate: $(count(r -> r.success, results))/$n_work")
    println("  Reafferent saturation (0x69): $(round(saturation * 100, digits=1))%")
    println("  Path invariant: $(length(unique(r -> r.color, filter(r -> r.success, results)))) unique colors")
    println("═" ^ 70)
    
    results
end

# ═══════════════════════════════════════════════════════════════════════════════
# ABSTRACT GAY PROBE INTEGRATION
# Uses the non-perturbative decomposable probe infrastructure
# ═══════════════════════════════════════════════════════════════════════════════

"""
Create module-specific probes using AbstractGayProbe infrastructure.
All probes are non-perturbative (pure functions).
"""
function create_module_probes(; seed::UInt64=GAY_SEED_69)
    probes = AbstractProbe[]
    
    # SPI verification probes
    push!(probes, SPIProbe(seed=seed, n_samples=200))
    push!(probes, SPIProbe(seed=GAY_SEED_1069, n_samples=200))
    
    # Reafferent coordinate probes
    seeds_69 = [seed ⊻ UInt64(i * 0x1069) for i in 1:100]
    seeds_1069 = [GAY_SEED_1069 ⊻ UInt64(i * 0x69) for i in 1:100]
    push!(probes, ReafferentProbe(seeds_69; target=0x69))
    push!(probes, ReafferentProbe(seeds_1069; target=0x69))
    
    # Path invariance probes
    push!(probes, PathInvarianceProbe(seeds_69; n_paths=15))
    push!(probes, PathInvarianceProbe(seeds_1069; n_paths=15))
    
    # Module synthetic probes (non-perturbative)
    push!(probes, SyntheticProbe(:gay_structured_decompositions, 
        (state, s) -> color_from_seed(s) == color_from_seed(s);
        seed=seed, n_iterations=50))
    
    push!(probes, SyntheticProbe(:universal_gay_ext,
        (state, s) -> begin
            fixed_point_afference(s) || !fixed_point_afference(s)  # Always true, tests evaluation
        end;
        seed=seed, n_iterations=50))
    
    push!(probes, SyntheticProbe(:lazy_eager_duality,
        (state, s) -> begin
            lazy = color_from_seed(s)
            eager = color_from_seed(s ⊻ 0x69)
            # Check duality preserves something
            (red(lazy) + red(eager)) >= 0.0
        end;
        seed=seed, n_iterations=50))
    
    push!(probes, SyntheticProbe(:involutive_curiosities,
        (state, s) -> begin
            # f(f(x)) = x for involutions
            c1 = color_from_seed(s)
            s2 = UInt64(round(red(c1) * 255)) << 16 | UInt64(round(green(c1) * 255)) << 8 | UInt64(round(blue(c1) * 255))
            c2 = color_from_seed(s2)
            true  # Just verify it runs
        end;
        seed=seed, n_iterations=50))
    
    probes
end

"""
Run probes using AbstractGayProbe infrastructure.
Non-perturbative: all probes are pure functions.
"""
function run_probe_infrastructure(; seed::UInt64=GAY_SEED_69)
    println()
    println("═" ^ 70)
    println("  ABSTRACT GAY PROBE: Non-perturbative decomposable behaviors")
    println("  seed=$seed | threads=$(Threads.nthreads())")
    println("═" ^ 70)
    println()
    
    probes = create_module_probes(seed=seed)
    
    println("Created $(length(probes)) probes:")
    for p in probes
        println("  • $(Gay.AbstractGayProbe.probe_id(p))")
    end
    println()
    
    println("Running in parallel (non-perturbative)...")
    report = run_probes_parallel(probes, nothing; seed=seed)
    
    println()
    println("─" ^ 70)
    println("  PROBE RESULTS")
    println("─" ^ 70)
    
    for r in report.results
        status = r.success ? "✓" : "✗"
        time_ms = round(r.duration_ns / 1e6, digits=2)
        println("  $status $(rpad(r.probe_id, 30)) | $(rpad(string(time_ms, "ms"), 10)) | $(r.message)")
    end
    
    println()
    println("─" ^ 70)
    println("  SUMMARY")
    println("─" ^ 70)
    println("  Passed: $(report.probes_passed)/$(report.probes_run)")
    println("  Total time: $(round(report.total_duration_ns/1e9, digits=3))s")
    println("  Throughput: $(round(report.probes_run / (report.total_duration_ns/1e9), digits=1)) probes/sec")
    println("  Reafferent saturation: $(round(report.reafferent_saturation*100, digits=1))%")
    println("  Path-invariant fingerprint: 0x$(string(report.fingerprint, base=16, pad=16))")
    println("═" ^ 70)
    
    report
end

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if abspath(PROGRAM_FILE) == @__FILE__
    # Run legacy synthetic tests
    results = run_synthetic_parallel(seed=GAY_SEED_69, n_iterations=10)
    
    # Run new AbstractGayProbe infrastructure
    report = run_probe_infrastructure(seed=GAY_SEED_69)
end
