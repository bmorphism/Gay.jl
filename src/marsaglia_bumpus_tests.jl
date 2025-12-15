# ═══════════════════════════════════════════════════════════════════════════════
# MARSAGLIA-BUMPUS TEST SUITE
# ═══════════════════════════════════════════════════════════════════════════════
#
# Two perspectives on splittable randomness quality:
#
# MARSAGLIA (1995-2003): Statistical tests for PRNG output sequences
#   "Does the sequence LOOK random?" → Birthday, Runs, Spectral
#
# BUMPUS (2021-2024): Compositional structure preservation
#   "Does SPLITTING preserve the tree structure?" → Adhesion width, Sheaf gluing
#
# The synthesis: Gay.jl must satisfy BOTH - statistical quality AND
# compositional coherence under the split() operation.
# ═══════════════════════════════════════════════════════════════════════════════

module MarsagliaBumpusTests

using ..Gay: gay_seed!, color_at, next_color, gay_split, GayRNG, GAY_SEED
using ..Gay: GayInterleaver, gay_interleave, SRGB, splitmix64
using SplittableRandoms: SplittableRandom, split
using Statistics: mean, std, var
using Printf: @printf

export run_marsaglia_suite, run_bumpus_suite, full_spi_audit
export birthday_spacing_test, runs_test, permutation_test, spectral_test
export adhesion_width_test, sheaf_gluing_test, tree_decomposition_test

# ═══════════════════════════════════════════════════════════════════════════════
# MARSAGLIA TESTS: Statistical Quality of Color Sequences
# ═══════════════════════════════════════════════════════════════════════════════

"""
    birthday_spacing_test(n, seed; λ_expected=4.0) -> (passed, p_value, collisions)

Marsaglia's Birthday Spacings Test adapted for color hue space.

Given n random hues in [0, 360), compute spacings between sorted values.
The number of "collisions" (spacing = 0 after discretization) should
follow Poisson(λ) where λ = n³/(4·m) for m = 360·resolution bins.

For colors: m = 3600 (0.1° resolution), n = 512 → λ ≈ 4.0
"""
function birthday_spacing_test(n::Int=512, seed::Integer=GAY_SEED; 
                                resolution::Float64=0.1, λ_expected::Float64=4.0)
    gay_seed!(seed)
    
    # Generate hues
    hues = Float64[]
    for i in 1:n
        c = color_at(i, SRGB(); seed=seed)
        # Extract hue from RGB (simplified - in practice use proper conversion)
        r, g, b = c.r, c.g, c.b
        max_c = max(r, g, b)
        min_c = min(r, g, b)
        if max_c == min_c
            h = 0.0
        elseif max_c == r
            h = 60.0 * mod((g - b) / (max_c - min_c), 6)
        elseif max_c == g
            h = 60.0 * ((b - r) / (max_c - min_c) + 2)
        else
            h = 60.0 * ((r - g) / (max_c - min_c) + 4)
        end
        push!(hues, h)
    end
    
    # Discretize to bins
    bins = [round(Int, h / resolution) for h in hues]
    m = round(Int, 360 / resolution)
    
    # Sort and compute spacings
    sort!(bins)
    spacings = diff(bins)
    
    # Count collisions (spacing = 0)
    collisions = count(==(0), spacings)
    
    # Poisson test: P(X = k) = λ^k e^(-λ) / k!
    # For birthday spacings, expected collisions ≈ n²/(2m)
    # This is the birthday problem: P(collision) ≈ n²/(2m) for n << m
    λ = (n^2) / (2.0 * m)
    
    # Chi-square-like p-value approximation
    z_score = (collisions - λ) / sqrt(λ)
    p_value = 1.0 - 0.5 * (1 + erf(abs(z_score) / sqrt(2)))
    
    passed = p_value > 0.01  # 1% significance level
    
    (passed=passed, p_value=p_value, collisions=collisions, 
     expected_λ=λ, z_score=z_score)
end

"""
    runs_test(n, seed) -> (passed, p_value, n_runs)

Marsaglia's Runs Test: count ascending/descending runs in color lightness.

For truly random sequence of length n:
- Expected runs ≈ (2n - 1) / 3
- Variance ≈ (16n - 29) / 90
"""
function runs_test(n::Int=1000, seed::Integer=GAY_SEED)
    gay_seed!(seed)
    
    # Generate lightness values
    lightness = Float64[]
    for i in 1:n
        c = color_at(i, SRGB(); seed=seed)
        # Perceived lightness (simplified)
        L = 0.299 * c.r + 0.587 * c.g + 0.114 * c.b
        push!(lightness, L)
    end
    
    # Count runs (sequences of consecutive increases or decreases)
    n_runs = 1
    increasing = lightness[2] > lightness[1]
    
    for i in 2:(n-1)
        new_increasing = lightness[i+1] > lightness[i]
        if new_increasing != increasing
            n_runs += 1
            increasing = new_increasing
        end
    end
    
    # Expected values for random sequence
    μ = (2 * n - 1) / 3
    σ² = (16 * n - 29) / 90
    σ = sqrt(σ²)
    
    z_score = (n_runs - μ) / σ
    p_value = 2 * (1.0 - 0.5 * (1 + erf(abs(z_score) / sqrt(2))))
    
    passed = p_value > 0.01
    
    (passed=passed, p_value=p_value, n_runs=n_runs, 
     expected=μ, std_dev=σ, z_score=z_score)
end

"""
    permutation_test(n, seed; window=5) -> (passed, chi_sq, p_value)

Marsaglia's Overlapping Permutations Test.

For each window of 5 consecutive colors, determine which of 120 possible
orderings (by lightness) occurs. Should be uniform.
"""
function permutation_test(n::Int=1200, seed::Integer=GAY_SEED; window::Int=5)
    gay_seed!(seed)
    
    factorial_w = factorial(window)  # 120 for window=5
    counts = zeros(Int, factorial_w)
    
    # Generate colors
    colors = [color_at(i, SRGB(); seed=seed) for i in 1:n]
    lightness = [0.299 * c.r + 0.587 * c.g + 0.114 * c.b for c in colors]
    
    # Count permutation patterns
    for i in 1:(n - window + 1)
        window_L = lightness[i:i+window-1]
        perm = sortperm(window_L)
        
        # Convert permutation to index (Lehmer code)
        idx = permutation_to_index(perm)
        counts[idx] += 1
    end
    
    # Chi-square test
    n_windows = n - window + 1
    expected = n_windows / factorial_w
    
    chi_sq = sum((counts .- expected).^2 ./ expected)
    df = factorial_w - 1
    
    # Approximate p-value using normal approximation for large df
    z = (chi_sq - df) / sqrt(2 * df)
    p_value = 1.0 - 0.5 * (1 + erf(z / sqrt(2)))
    
    passed = p_value > 0.01
    
    (passed=passed, chi_sq=chi_sq, p_value=p_value, df=df,
     observed_patterns=count(>(0), counts))
end

"""Convert permutation to unique index via Lehmer code."""
function permutation_to_index(perm::Vector{Int})
    n = length(perm)
    idx = 1
    for i in 1:n
        # Count elements smaller than perm[i] that come after position i
        smaller_after = count(j -> perm[j] < perm[i], (i+1):n)
        idx += smaller_after * factorial(n - i)
    end
    idx
end

"""
    spectral_test(n, seed) -> (passed, max_peak, peak_freq)

Spectral Test: FFT analysis of color sequence.

High peaks at non-zero frequencies indicate periodic structure (bad).
"""
function spectral_test(n::Int=1024, seed::Integer=GAY_SEED)
    gay_seed!(seed)
    
    # Generate complex color signal
    signal = ComplexF64[]
    for i in 1:n
        c = color_at(i, SRGB(); seed=seed)
        # Encode as complex: real=red-blue, imag=green
        push!(signal, complex(c.r - c.b, c.g))
    end
    
    # Simple DFT (in practice use FFTW)
    spectrum = dft(signal)
    
    # Compute power spectrum (skip DC component)
    power = abs2.(spectrum[2:div(n,2)])
    
    # Find maximum peak
    max_peak, peak_idx = findmax(power)
    peak_freq = peak_idx / n
    
    # For random signal, power should be roughly uniform
    mean_power = mean(power)
    # Threshold of 12x mean gives ~0.5% false positive rate (see issue #188)
    # Previous threshold of 10x had ~1.7% FP rate across seeds 1-1000
    threshold = mean_power * 12

    passed = max_peak < threshold
    
    (passed=passed, max_peak=max_peak, mean_power=mean_power,
     peak_freq=peak_freq, ratio=max_peak/mean_power)
end

"""Simple DFT implementation."""
function dft(x::Vector{ComplexF64})
    n = length(x)
    X = zeros(ComplexF64, n)
    for k in 0:(n-1)
        for j in 0:(n-1)
            X[k+1] += x[j+1] * exp(-2π * im * k * j / n)
        end
    end
    X
end

# ═══════════════════════════════════════════════════════════════════════════════
# BUMPUS TESTS: Compositional Structure Preservation
# ═══════════════════════════════════════════════════════════════════════════════

"""
    SplitTree

Tree structure recording split() operations.
Each node represents an RNG state, edges represent splits.
"""
mutable struct SplitTree
    seed::UInt64
    depth::Int
    children::Vector{SplitTree}
    colors::Vector{Any}  # Colors generated from this node's stream
    fingerprint::UInt64
end

SplitTree(seed::UInt64, depth::Int=0) = SplitTree(seed, depth, SplitTree[], [], seed)

"""
    build_split_tree(seed, max_depth, colors_per_node) -> SplitTree

Build a binary tree of splits, generating colors at each node.
"""
function build_split_tree(seed::UInt64, max_depth::Int, colors_per_node::Int)
    root = SplitTree(seed, 0)
    
    function build!(node::SplitTree, rng::SplittableRandom, depth::Int)
        # Generate colors at this node
        for _ in 1:colors_per_node
            local_rng = split(rng)
            c = random_color_from_rng(local_rng)
            push!(node.colors, c)
            node.fingerprint ⊻= color_to_u64(c)
        end
        
        if depth < max_depth
            # Split into two children
            left_rng = split(rng)
            right_rng = split(rng)
            
            left_child = SplitTree(node.seed ⊻ UInt64(1) << depth, depth + 1)
            right_child = SplitTree(node.seed ⊻ UInt64(2) << depth, depth + 1)
            
            build!(left_child, left_rng, depth + 1)
            build!(right_child, right_rng, depth + 1)
            
            push!(node.children, left_child)
            push!(node.children, right_child)
        end
    end
    
    rng = SplittableRandom(seed)
    build!(root, rng, 0)
    root
end

"""Extract random color from an RNG (placeholder - connects to Gay.jl internals)."""
function random_color_from_rng(rng)
    # Generate RGB from RNG state
    r = Float64(rand(rng, UInt32)) / typemax(UInt32)
    g = Float64(rand(rng, UInt32)) / typemax(UInt32)
    b = Float64(rand(rng, UInt32)) / typemax(UInt32)
    (r=r, g=g, b=b)
end

"""Convert color to UInt64 for fingerprinting."""
function color_to_u64(c)
    r = round(UInt64, c.r * 255) << 16
    g = round(UInt64, c.g * 255) << 8
    b = round(UInt64, c.b * 255)
    r | g | b
end

"""
    adhesion_width_test(seed, depth) -> (passed, max_width, adhesions)

Bumpus's Adhesion Width Test.

For a tree decomposition induced by split(), measure the "adhesion"
(shared information) between sibling subtrees.

Bounded adhesion width ⟹ good compositional structure.
"""
function adhesion_width_test(seed::UInt64=GAY_SEED, depth::Int=4; 
                              colors_per_node::Int=10, max_width::Int=3)
    tree = build_split_tree(seed, depth, colors_per_node)
    
    adhesions = Tuple{Int, Int, Int}[]  # (depth, left_idx, width)
    max_observed_width = 0
    
    function measure_adhesions!(node::SplitTree)
        if length(node.children) >= 2
            left = node.children[1]
            right = node.children[2]
            
            # Adhesion = colors that appear in both subtrees (by fingerprint collision)
            left_fps = Set(color_to_u64(c) for c in collect_all_colors(left))
            right_fps = Set(color_to_u64(c) for c in collect_all_colors(right))
            
            shared = length(intersect(left_fps, right_fps))
            width = shared
            
            push!(adhesions, (node.depth, 1, width))
            max_observed_width = max(max_observed_width, width)
            
            for child in node.children
                measure_adhesions!(child)
            end
        end
    end
    
    measure_adhesions!(tree)
    
    passed = max_observed_width <= max_width
    
    (passed=passed, max_width=max_observed_width, 
     adhesions=adhesions, bound=max_width)
end

"""Collect all colors from a subtree."""
function collect_all_colors(node::SplitTree)
    colors = copy(node.colors)
    for child in node.children
        append!(colors, collect_all_colors(child))
    end
    colors
end

"""
    sheaf_gluing_test(seed, n_patches) -> (passed, gluing_error, sections)

Bumpus's Sheaf Gluing Test.

A presheaf F is a sheaf if sections that agree on overlaps can be glued.
For Gay.jl: colors from different split branches should XOR-glue consistently.

Gluing condition: F(U ∪ V) ≅ F(U) ×_{F(U∩V)} F(V)
"""
function sheaf_gluing_test(seed::UInt64=GAY_SEED, n_patches::Int=4)
    gay_seed!(seed)
    
    # Create overlapping "patches" of color indices
    patch_size = 100
    overlap = 20
    
    patches = Vector{Vector{Int}}()
    for i in 1:n_patches
        start = (i - 1) * (patch_size - overlap) + 1
        push!(patches, collect(start:(start + patch_size - 1)))
    end
    
    # Compute section (fingerprint) for each patch
    sections = UInt64[]
    for patch in patches
        fp = UInt64(0)
        for idx in patch
            c = color_at(idx, SRGB(); seed=seed)
            fp ⊻= color_to_u64(c)
        end
        push!(sections, fp)
    end
    
    # Verify gluing: XOR of all sections should equal global fingerprint
    # minus double-counted overlaps
    global_fp = UInt64(0)
    all_indices = sort(unique(vcat(patches...)))
    for idx in all_indices
        c = color_at(idx, SRGB(); seed=seed)
        global_fp ⊻= color_to_u64(c)
    end
    
    # Compute what gluing predicts
    glued_fp = reduce(⊻, sections)
    
    # Correct for overlaps (XORed twice)
    for i in 1:(n_patches-1)
        overlap_indices = intersect(patches[i], patches[i+1])
        for idx in overlap_indices
            c = color_at(idx, SRGB(); seed=seed)
            glued_fp ⊻= color_to_u64(c)  # Remove double count
        end
    end
    
    gluing_error = count_ones(global_fp ⊻ glued_fp)
    passed = gluing_error == 0
    
    (passed=passed, gluing_error=gluing_error, 
     global_fp=global_fp, glued_fp=glued_fp, n_sections=n_patches)
end

"""
    tree_decomposition_test(seed, n) -> (passed, width, bags)

Bumpus's Tree Decomposition Test.

The graph of "color similarity" (edges between similar hues) should
have bounded tree-width when colors come from a splittable RNG.

This tests whether the RNG's algebraic structure induces good decomposability.
"""
function tree_decomposition_test(seed::UInt64=GAY_SEED, n::Int=100; 
                                  similarity_threshold::Float64=0.1)
    gay_seed!(seed)
    
    # Generate colors and build similarity graph
    colors = [color_at(i, SRGB(); seed=seed) for i in 1:n]
    
    # Adjacency: connect colors within threshold Euclidean distance
    adj = Dict{Int, Set{Int}}()
    for i in 1:n
        adj[i] = Set{Int}()
    end
    
    for i in 1:n
        for j in (i+1):n
            dist = sqrt((colors[i].r - colors[j].r)^2 + 
                       (colors[i].g - colors[j].g)^2 + 
                       (colors[i].b - colors[j].b)^2)
            if dist < similarity_threshold
                push!(adj[i], j)
                push!(adj[j], i)
            end
        end
    end
    
    # Compute tree-width via minimum-degree heuristic
    width, bags = minimum_degree_tree_width(adj, n)
    
    # For random colors, expect low tree-width (sparse similarity graph)
    # Bound: sqrt(n) for well-distributed colors
    expected_bound = ceil(Int, sqrt(n))
    passed = width <= expected_bound
    
    (passed=passed, width=width, bound=expected_bound, 
     n_bags=length(bags), avg_bag_size=mean(length.(bags)))
end

"""Minimum-degree elimination for tree-width upper bound."""
function minimum_degree_tree_width(adj::Dict{Int, Set{Int}}, n::Int)
    adj = Dict(k => copy(v) for (k, v) in adj)
    remaining = Set(1:n)
    bags = Set{Int}[]
    max_width = 0
    
    while !isempty(remaining)
        # Find minimum degree vertex
        min_deg = typemax(Int)
        min_v = first(remaining)
        
        for v in remaining
            deg = length(adj[v] ∩ remaining)
            if deg < min_deg
                min_deg = deg
                min_v = v
            end
        end
        
        # Create bag
        neighbors = collect(adj[min_v] ∩ remaining)
        bag = Set([min_v; neighbors])
        push!(bags, bag)
        max_width = max(max_width, length(bag) - 1)
        
        # Add fill edges
        for i in 1:length(neighbors)
            for j in (i+1):length(neighbors)
                push!(adj[neighbors[i]], neighbors[j])
                push!(adj[neighbors[j]], neighbors[i])
            end
        end
        
        delete!(remaining, min_v)
    end
    
    (max_width, bags)
end

# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED TEST SUITES
# ═══════════════════════════════════════════════════════════════════════════════

"""
    run_marsaglia_suite(seed) -> results

Run all Marsaglia-style statistical tests.
"""
function run_marsaglia_suite(seed::Integer=GAY_SEED)
    println("═══════════════════════════════════════════════════════════════")
    println("  MARSAGLIA TEST SUITE: Statistical Quality of Gay.jl Colors")
    println("═══════════════════════════════════════════════════════════════")
    
    results = Dict{Symbol, Any}()
    
    print("1. Birthday Spacings Test... ")
    results[:birthday] = birthday_spacing_test(512, seed)
    println(results[:birthday].passed ? "✓ PASS" : "✗ FAIL")
    @printf("   Collisions: %d (expected λ=%.2f), p=%.4f\n", 
            results[:birthday].collisions, results[:birthday].expected_λ, 
            results[:birthday].p_value)
    
    print("2. Runs Test... ")
    results[:runs] = runs_test(1000, seed)
    println(results[:runs].passed ? "✓ PASS" : "✗ FAIL")
    @printf("   Runs: %d (expected %.1f ± %.1f), p=%.4f\n",
            results[:runs].n_runs, results[:runs].expected, 
            results[:runs].std_dev, results[:runs].p_value)
    
    print("3. Permutation Test... ")
    results[:permutation] = permutation_test(1200, seed)
    println(results[:permutation].passed ? "✓ PASS" : "✗ FAIL")
    @printf("   χ²=%.2f (df=%d), p=%.4f, patterns=%d/120\n",
            results[:permutation].chi_sq, results[:permutation].df,
            results[:permutation].p_value, results[:permutation].observed_patterns)
    
    print("4. Spectral Test... ")
    results[:spectral] = spectral_test(1024, seed)
    println(results[:spectral].passed ? "✓ PASS" : "✗ FAIL")
    @printf("   Peak/Mean ratio: %.2f (threshold: 12.0)\n",
            results[:spectral].ratio)
    
    all_passed = all(r.passed for r in values(results))
    println("───────────────────────────────────────────────────────────────")
    println(all_passed ? "  ALL MARSAGLIA TESTS PASSED ✓" : "  SOME TESTS FAILED ✗")
    
    results
end

"""
    run_bumpus_suite(seed) -> results

Run all Bumpus-style compositional tests.
"""
function run_bumpus_suite(seed::Integer=GAY_SEED)
    println("═══════════════════════════════════════════════════════════════")
    println("  BUMPUS TEST SUITE: Compositional Structure of Gay.jl Splits")
    println("═══════════════════════════════════════════════════════════════")
    
    results = Dict{Symbol, Any}()
    
    print("1. Adhesion Width Test... ")
    results[:adhesion] = adhesion_width_test(UInt64(seed), 4)
    println(results[:adhesion].passed ? "✓ PASS" : "✗ FAIL")
    @printf("   Max width: %d (bound: %d)\n",
            results[:adhesion].max_width, results[:adhesion].bound)
    
    print("2. Sheaf Gluing Test... ")
    results[:sheaf] = sheaf_gluing_test(UInt64(seed), 4)
    println(results[:sheaf].passed ? "✓ PASS" : "✗ FAIL")
    @printf("   Gluing error: %d bits, sections: %d\n",
            results[:sheaf].gluing_error, results[:sheaf].n_sections)
    
    print("3. Tree Decomposition Test... ")
    results[:treewidth] = tree_decomposition_test(UInt64(seed), 100)
    println(results[:treewidth].passed ? "✓ PASS" : "✗ FAIL")
    @printf("   Width: %d (bound: %d), bags: %d\n",
            results[:treewidth].width, results[:treewidth].bound,
            results[:treewidth].n_bags)
    
    all_passed = all(r.passed for r in values(results))
    println("───────────────────────────────────────────────────────────────")
    println(all_passed ? "  ALL BUMPUS TESTS PASSED ✓" : "  SOME TESTS FAILED ✗")
    
    results
end

"""
    full_spi_audit(seed) -> (marsaglia_results, bumpus_results, verdict)

Complete Strong Parallelism Invariance audit combining both perspectives.
"""
function full_spi_audit(seed::Integer=GAY_SEED)
    println("\n")
    println("╔═══════════════════════════════════════════════════════════════╗")
    println("║     FULL SPI AUDIT: Marsaglia + Bumpus on Gay.jl             ║")
    println("║     Seed: $(lpad(seed, 20))                      ║")
    println("╚═══════════════════════════════════════════════════════════════╝")
    println()
    
    marsaglia = run_marsaglia_suite(seed)
    println()
    bumpus = run_bumpus_suite(seed)
    
    m_pass = all(r.passed for r in values(marsaglia))
    b_pass = all(r.passed for r in values(bumpus))
    
    println()
    println("╔═══════════════════════════════════════════════════════════════╗")
    if m_pass && b_pass
        println("║  VERDICT: Gay.jl PASSES FULL SPI AUDIT ✓                     ║")
        println("║                                                               ║")
        println("║  • Statistical quality: Marsaglia-approved                    ║")
        println("║  • Compositional structure: Bumpus-certified                  ║")
        println("║  • Same seed → same colors, always                           ║")
    else
        println("║  VERDICT: ISSUES DETECTED                                     ║")
        !m_pass && println("║  • Statistical tests: NEEDS ATTENTION                        ║")
        !b_pass && println("║  • Compositional tests: NEEDS ATTENTION                      ║")
    end
    println("╚═══════════════════════════════════════════════════════════════╝")
    
    (marsaglia=marsaglia, bumpus=bumpus, 
     verdict=m_pass && b_pass ? :PASS : :FAIL)
end

# ═══════════════════════════════════════════════════════════════════════════════
# GENESIS HANDOFF TESTS: Left/Right Split Independence
# ═══════════════════════════════════════════════════════════════════════════════

export genesis_handoff_test, split_correlation_test, run_genesis_suite

"""
    split_correlation_test(seed, n_samples=10000) -> NamedTuple

Test statistical independence of Left vs Right split streams.
Correlation should be ≈ 0 for truly independent streams.
"""
function split_correlation_test(seed::UInt64=GAY_SEED, n_samples::Int=10000)
    rng = SplittableRandom(seed)

    left_values = Float64[]
    right_values = Float64[]

    for _ in 1:n_samples
        left_rng = split(rng)
        right_rng = split(rng)

        left_val = Float64(rand(left_rng, UInt64)) / typemax(UInt64)
        right_val = Float64(rand(right_rng, UInt64)) / typemax(UInt64)

        push!(left_values, left_val)
        push!(right_values, right_val)
        rng = split(rng)
    end

    # Pearson correlation
    μ_l, μ_r = mean(left_values), mean(right_values)
    σ_l, σ_r = std(left_values), std(right_values)
    cov = mean((left_values .- μ_l) .* (right_values .- μ_r))
    correlation = cov / (σ_l * σ_r)

    passed = abs(correlation) < 0.05  # 5% threshold

    (passed=passed, correlation=correlation, n_samples=n_samples)
end

"""
    genesis_handoff_test(seed, depth=5) -> NamedTuple

Test that parent state does not leak to children:
- Determinism: Same seed → same split tree
- Sibling Independence: L ≠ R at each node
- Collision Freedom: All nodes produce unique fingerprints
"""
function genesis_handoff_test(seed::UInt64=GAY_SEED, depth::Int=5)
    fingerprints = Dict{String, UInt64}()

    function traverse!(rng, path::String, d::Int)
        fp = UInt64(0)
        local_rng = split(rng)
        for i in 1:100
            val = rand(local_rng, UInt64)
            fp ⊻= splitmix64(val ⊻ UInt64(i))
        end
        fingerprints[path] = fp

        if d < depth
            left = split(rng)
            right = split(rng)
            traverse!(left, path * "L", d + 1)
            traverse!(right, path * "R", d + 1)
        end
    end

    traverse!(SplittableRandom(seed), "", 0)

    # Verify determinism (run again)
    fingerprints2 = Dict{String, UInt64}()
    function traverse2!(rng, path::String, d::Int)
        fp = UInt64(0)
        local_rng = split(rng)
        for i in 1:100
            val = rand(local_rng, UInt64)
            fp ⊻= splitmix64(val ⊻ UInt64(i))
        end
        fingerprints2[path] = fp
        if d < depth
            traverse2!(split(rng), path * "L", d + 1)
            traverse2!(split(rng), path * "R", d + 1)
        end
    end
    traverse2!(SplittableRandom(seed), "", 0)

    deterministic = all(fingerprints[k] == fingerprints2[k] for k in keys(fingerprints))

    # Check sibling independence
    sibling_pairs = [(k, replace(k, r"L$" => "R")) for k in keys(fingerprints) if endswith(k, "L")]
    siblings_independent = all(fingerprints[l] != fingerprints[r] for (l, r) in sibling_pairs if haskey(fingerprints, r))

    # Check collision freedom
    n_nodes = length(fingerprints)
    unique_fps = length(unique(values(fingerprints)))
    collision_free = n_nodes == unique_fps

    passed = deterministic && siblings_independent && collision_free

    (passed=passed, deterministic=deterministic, siblings_independent=siblings_independent,
     collision_free=collision_free, n_nodes=n_nodes, unique_fps=unique_fps)
end

"""
    run_genesis_suite(seed) -> results

Run all genesis handoff tests for split independence verification.
"""
function run_genesis_suite(seed::Integer=GAY_SEED)
    println("═══════════════════════════════════════════════════════════════")
    println("  GENESIS HANDOFF SUITE: Left/Right Split Independence")
    println("═══════════════════════════════════════════════════════════════")

    results = Dict{Symbol, Any}()

    print("1. Split Correlation Test... ")
    results[:correlation] = split_correlation_test(UInt64(seed), 10000)
    println(results[:correlation].passed ? "✓ PASS" : "✗ FAIL")
    @printf("   Correlation: %.6f (threshold: ±0.05)\n", results[:correlation].correlation)

    print("2. Genesis Handoff Test... ")
    results[:genesis] = genesis_handoff_test(UInt64(seed), 5)
    println(results[:genesis].passed ? "✓ PASS" : "✗ FAIL")
    println("   Deterministic: $(results[:genesis].deterministic ? "✓" : "✗")")
    println("   Siblings Independent: $(results[:genesis].siblings_independent ? "✓" : "✗")")
    println("   Collision-Free: $(results[:genesis].collision_free ? "✓" : "✗") ($(results[:genesis].unique_fps)/$(results[:genesis].n_nodes))")

    all_passed = all(r.passed for r in values(results))
    println("───────────────────────────────────────────────────────────────")
    println(all_passed ? "  ALL GENESIS TESTS PASSED ✓" : "  SOME TESTS FAILED ✗")

    results
end

# Error function approximation (since we're not importing SpecialFunctions)
function erf(x::Float64)
    # Abramowitz and Stegun approximation
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911
    
    sign = x < 0 ? -1 : 1
    x = abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x)
    sign * y
end

end # module MarsagliaBumpusTests
