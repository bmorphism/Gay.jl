#!/usr/bin/env julia
# Advanced GamutLearnable Example - Parallel Processing with Gay.jl
# Demonstrates best practices and parallel color generation

using Pkg
Pkg.activate(dirname(@__DIR__))

using Gay
using Colors
using Statistics
using SHA
using OhMyThreads  # Gay.jl uses this for parallel execution

# Include GamutLearnable
include("../src/gamut_learnable.jl")
using .GamutLearnable

# ═══════════════════════════════════════════════════════════════════════════
# Gay.jl Best Practice: Domain Object Hashing for Seeds
# ═══════════════════════════════════════════════════════════════════════════

"""
Generate deterministic seed from domain identifier.
Following Gay.jl's golden rule: "The seed should be derivable from what you're visualizing"
"""
function generate_seed(identifier::String)::UInt64
    bytes = sha256(identifier)
    return reinterpret(UInt64, bytes[1:8])[1]
end

"""
Composite seed generation for complex scenarios.
"""
function composite_seed(attributes...)::UInt64
    combined = join(string.(attributes), "_")
    bytes = sha256(combined)
    return reinterpret(UInt64, bytes[1:8])[1]
end

# ═══════════════════════════════════════════════════════════════════════════
# Namespace-based Seed Management
# ═══════════════════════════════════════════════════════════════════════════

const SEED_OFFSETS = (
    ui = 0,
    visualization = 1_000_000,
    effects = 2_000_000,
    data = 3_000_000
)

function namespaced_seed(base::UInt64, namespace::Symbol)
    return base + SEED_OFFSETS[namespace]
end

# ═══════════════════════════════════════════════════════════════════════════
# Example 1: Parallel Color Generation with Deterministic Seeds
# ═══════════════════════════════════════════════════════════════════════════

println("🚀 Parallel Color Generation with Gay.jl")
println("="^60)

# Generate seed from experiment context
experiment = "parallel_gamut_mapping"
version = "v2.0"
timestamp = "2024_12_15"
base_seed = composite_seed(experiment, version, timestamp)

println("\nExperiment: $experiment")
println("Version: $version")
println("Base seed derived from: $(experiment)_$(version)_$(timestamp)")

# Use Gay.jl's parallel capabilities with OhMyThreads
n_batches = 4
colors_per_batch = 250
total_colors = n_batches * colors_per_batch

println("\nGenerating $(total_colors) colors in $n_batches parallel batches...")

# Parallel generation using random access (Gay.jl best practice for parallel code)
all_colors = tmap(1:n_batches) do batch_id
    batch_seed = namespaced_seed(base_seed, :visualization) + batch_id * 1000
    start_idx = (batch_id - 1) * colors_per_batch + 1
    end_idx = batch_id * colors_per_batch

    # Random access pattern - efficient for parallel execution
    [color_at(i, batch_seed) for i in start_idx:end_idx]
end |> vec

println("✓ Generated $(length(all_colors)) colors in parallel")

# ═══════════════════════════════════════════════════════════════════════════
# Example 2: Analyze Color Distribution
# ═══════════════════════════════════════════════════════════════════════════

println("\n📊 Analyzing Color Distribution")
println("-"^40)

# Convert to Lab for analysis
lab_colors = [convert(Lab, c) for c in all_colors]

# Compute statistics
chromas = [sqrt(c.a^2 + c.b^2) for c in lab_colors]
lightnesses = [c.l for c in lab_colors]

println("Chroma statistics:")
println("  Mean: $(round(mean(chromas), digits=1))")
println("  Max:  $(round(maximum(chromas), digits=1))")
println("  Min:  $(round(minimum(chromas), digits=1))")

println("Lightness statistics:")
println("  Mean: $(round(mean(lightnesses), digits=1))")
println("  Range: [$(round(minimum(lightnesses), digits=1)), $(round(maximum(lightnesses), digits=1))]")

# Count out-of-gamut colors
out_of_gamut_count = sum(!in_gamut(c, :srgb) for c in lab_colors)
println("\nColors exceeding sRGB: $out_of_gamut_count / $total_colors ($(round(100*out_of_gamut_count/total_colors, digits=1))%)")

# ═══════════════════════════════════════════════════════════════════════════
# Example 3: Parallel Gamut Mapping
# ═══════════════════════════════════════════════════════════════════════════

println("\n🎨 Parallel Gamut Mapping")
println("-"^40)

# Create gamut mapper
mapper = GamutMapper(target_gamut=:srgb)

# Parallel mapping using OhMyThreads
println("Mapping $total_colors colors to sRGB gamut in parallel...")

mapped_colors = tmap(chunks(lab_colors; n=n_batches)) do chunk
    [map_to_gamut(c, mapper.params) for c in chunk]
end |> vec

# Verify all colors are now in gamut
still_out = sum(!in_gamut(c, :srgb) for c in mapped_colors)
println("✓ Mapped colors - out of gamut: $still_out")

# Compute preservation metrics
original_chromas = [sqrt(c.a^2 + c.b^2) for c in lab_colors]
mapped_chromas = [sqrt(c.a^2 + c.b^2) for c in mapped_colors]
preservation = mean(mapped_chromas) / mean(original_chromas) * 100

println("✓ Average chroma preservation: $(round(preservation, digits=1))%")

# ═══════════════════════════════════════════════════════════════════════════
# Example 4: Pride Flag Colors with Gamut Mapping
# ═══════════════════════════════════════════════════════════════════════════

println("\n🏳️‍🌈 Pride Flag Gamut Mapping")
println("-"^40)

# Generate pride flag colors using Gay.jl
pride_seed = generate_seed("pride_flag_demo")
gay_seed!(pride_seed)

pride_palettes = Dict(
    :rainbow => rainbow(),
    :transgender => pride_flag(:transgender),
    :bisexual => pride_flag(:bisexual)
)

println("Mapping pride flag colors to different gamuts:")
for (flag, colors) in pride_palettes
    for gamut in [:srgb, :p3, :rec2020]
        mapper = GamutMapper(target_gamut=gamut)

        if isa(colors, Color)
            colors = [colors]  # Single color
        end

        original_lab = [convert(Lab, c) for c in colors]
        mapped_lab = [map_to_gamut(c, mapper.params) for c in original_lab]

        avg_preservation = mean([
            sqrt(m.a^2 + m.b^2) / max(sqrt(o.a^2 + o.b^2), 1e-6)
            for (o, m) in zip(original_lab, mapped_lab)
        ]) * 100

        println("  $flag → $gamut: $(round(avg_preservation, digits=1))% preserved")
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# Example 5: Sparse Index Access Pattern
# ═══════════════════════════════════════════════════════════════════════════

println("\n🎲 Sparse Index Access (Gay.jl Random Access)")
println("-"^40)

# Demonstrate efficient sparse access - a key Gay.jl feature
sparse_indices = [1, 10, 100, 1000, 10000, 100000, 1000000]
data_seed = namespaced_seed(base_seed, :data)

println("Accessing colors at exponentially sparse indices...")
sparse_colors = [color_at(idx, data_seed) for idx in sparse_indices]

# Map sparse colors
sparse_lab = [convert(Lab, c) for c in sparse_colors]
mapper_sparse = GamutMapper(target_gamut=:p3)
mapped_sparse = [map_to_gamut(c, mapper_sparse.params) for c in sparse_lab]

for (idx, orig, mapped) in zip(sparse_indices, sparse_lab, mapped_sparse)
    C_orig = sqrt(orig.a^2 + orig.b^2)
    C_mapped = sqrt(mapped.a^2 + mapped.b^2)
    preservation = C_mapped / max(C_orig, 1e-6) * 100
    println("  Index $(lpad(idx, 7)): C=$(lpad(round(C_orig, digits=1), 5)) → $(lpad(round(C_mapped, digits=1), 5)) ($(round(preservation, digits=0))%)")
end

# ═══════════════════════════════════════════════════════════════════════════
# Example 6: Caching Strategy for Repeated Access
# ═══════════════════════════════════════════════════════════════════════════

println("\n💾 Caching Strategy (Gay.jl Best Practice)")
println("-"^40)

# Gay.jl best practice: cache colors if repeatedly accessing same indices
cache_seed = generate_seed("cached_visualization")

# Pre-compute and cache frequently accessed colors
frequent_indices = 1:100
color_cache = Dict{Int, RGB}()

println("Building color cache for indices 1-100...")
for idx in frequent_indices
    color_cache[idx] = color_at(idx, cache_seed)
end

# Now access is O(1) from cache
access_times = 1000
println("Accessing cached colors $access_times times...")
for _ in 1:access_times
    idx = rand(frequent_indices)
    _ = color_cache[idx]  # O(1) access
end
println("✓ Cache hit rate: 100%")

# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════

println("\n" * "="^60)
println("✨ Advanced Gay.jl + GamutLearnable Integration Complete!")
println("="^60)
println()
println("Demonstrated Gay.jl Best Practices:")
println("  ✓ Domain object hashing for seeds")
println("  ✓ Composite seeds for complex scenarios")
println("  ✓ Namespace-based seed management")
println("  ✓ Parallel color generation with OhMyThreads")
println("  ✓ Random access for sparse indices")
println("  ✓ Caching for repeated access")
println("  ✓ Pride flag palette integration")
println()
println("Gamut Mapping Results:")
println("  • Processed $total_colors colors in parallel")
println("  • Achieved $(round(preservation, digits=1))% average chroma preservation")
println("  • Successfully mapped to sRGB, P3, and Rec.2020")
println()
println("This implementation follows Gay.jl's golden rule:")
println("'The seed should be derivable from what you're visualizing'")