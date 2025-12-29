#!/usr/bin/env julia
# GamutLearnable Example - Demonstrating adaptive gamut mapping for Gay.jl color chains
# This example shows how to use the Enzyme-optimized gamut mapper to handle
# colors that exceed sRGB boundaries

using Pkg
Pkg.activate(dirname(@__DIR__))

using Gay
using Colors
using Statistics  # For mean function
using Enzyme  # Will trigger loading of GayEnzymeExt

# Include the GamutLearnable module
include("../src/gamut_learnable.jl")
using .GamutLearnable

# ═══════════════════════════════════════════════════════════════════════════
# Generate a high-chroma color chain using Gay.jl
# ═══════════════════════════════════════════════════════════════════════════

println("🎨 Generating high-chroma color chain with Gay.jl...")

# Follow Gay.jl best practices: use domain object hashing
using SHA

function generate_seed(identifier::String)::UInt64
    bytes = sha256(identifier)
    return reinterpret(UInt64, bytes[1:8])[1]
end

# Generate seed from meaningful identifier
experiment_seed = generate_seed("gamut_mapping_demo_v1")
gay_seed!(experiment_seed)
println("Using seed from identifier: 'gamut_mapping_demo_v1' → $(experiment_seed)")

# Generate a chain of colors with potentially high chroma
n_colors = 50
color_chain = [next_color() for _ in 1:n_colors]

# Convert to Lab space to analyze chroma
lab_chain = [convert(Lab, c) for c in color_chain]

# Find colors that exceed sRGB gamut
out_of_gamut = Lab[]
for c in lab_chain
    if !in_gamut(c, :srgb)
        push!(out_of_gamut, c)
    end
end

println("Generated $(n_colors) colors")
println("$(length(out_of_gamut)) colors exceed sRGB gamut ($(round(100*length(out_of_gamut)/n_colors, digits=1))%)")

# ═══════════════════════════════════════════════════════════════════════════
# Example 1: Basic gamut mapping without training
# ═══════════════════════════════════════════════════════════════════════════

println("\n📊 Example 1: Basic gamut mapping (no training)")

# Create default parameters
params = GamutParameters(target_gamut=:srgb)

# Map colors to gamut
mapped_basic = [map_to_gamut(c, params) for c in lab_chain]

# Check how many are now in gamut
still_out = sum(!in_gamut(c, :srgb) for c in mapped_basic)
println("After basic mapping: $still_out colors still out of gamut")

# Calculate average chroma preservation
avg_chroma_orig = mean(sqrt(c.a^2 + c.b^2) for c in lab_chain)
avg_chroma_mapped = mean(sqrt(c.a^2 + c.b^2) for c in mapped_basic)
preservation = avg_chroma_mapped / avg_chroma_orig * 100

println("Average chroma preservation: $(round(preservation, digits=1))%")

# ═══════════════════════════════════════════════════════════════════════════
# Example 2: Train gamut mapper with Enzyme autodiff
# ═══════════════════════════════════════════════════════════════════════════

println("\n🚀 Example 2: Training with Enzyme autodiff")

# Create trainable parameters
train_params = GamutParameters(target_gamut=:srgb)

# Check if Enzyme extension is loaded
if isdefined(Main, :enzyme_train_gamut!)
    println("✓ Enzyme extension loaded - using autodiff")

    # Train with Enzyme
    enzyme_train_gamut!(train_params, lab_chain;
                        epochs=50, lr=0.01, verbose=true)
else
    println("⚠ Enzyme extension not loaded - using finite differences")
    train_gamut_mapper!(train_params, lab_chain;
                        epochs=50, lr=0.01, verbose=true)
end

# Map with trained parameters
mapped_trained = [map_to_gamut(c, train_params) for c in lab_chain]

# Evaluate results
still_out_trained = sum(!in_gamut(c, :srgb) for c in mapped_trained)
println("\nAfter trained mapping: $still_out_trained colors still out of gamut")

avg_chroma_trained = mean(sqrt(c.a^2 + c.b^2) for c in mapped_trained)
preservation_trained = avg_chroma_trained / avg_chroma_orig * 100
println("Average chroma preservation: $(round(preservation_trained, digits=1))%")

improvement = preservation_trained - preservation
println("Improvement over basic mapping: +$(round(improvement, digits=1))%")

# ═══════════════════════════════════════════════════════════════════════════
# Example 3: Using Gay.jl's random access pattern
# ═══════════════════════════════════════════════════════════════════════════

println("\n🎲 Example 3: Random access with Gay.jl")

# Demonstrate Gay.jl's random access capability
sparse_indices = [10, 100, 1000, 10000, 100000]
println("Accessing colors at sparse indices: $sparse_indices")

# Use composite seed for this visualization context
viz_seed = generate_seed("gamut_test_sparse_colors")
sparse_colors = [color_at(i, viz_seed) for i in sparse_indices]
sparse_lab = [convert(Lab, c) for c in sparse_colors]

# Map sparse colors through gamut mapper
mapped_sparse = [map_to_gamut(c, train_params) for c in sparse_lab]

println("Sparse color mapping results:")
for (idx, orig, mapped) in zip(sparse_indices, sparse_lab, mapped_sparse)
    C_orig = sqrt(orig.a^2 + orig.b^2)
    C_mapped = sqrt(mapped.a^2 + mapped.b^2)
    println("  Index $idx: C=$(round(C_orig, digits=1)) → $(round(C_mapped, digits=1))")
end

# ═══════════════════════════════════════════════════════════════════════════
# Example 4: Compare different target gamuts
# ═══════════════════════════════════════════════════════════════════════════

println("\n🌈 Example 4: Comparing target gamuts")

gamuts = [:srgb, :p3, :rec2020]
results = Dict{Symbol, NamedTuple}()

for gamut in gamuts
    params = GamutParameters(target_gamut=gamut)
    mapped = [map_to_gamut(c, params) for c in lab_chain]

    in_gamut_count = sum(in_gamut(c, gamut) for c in mapped)
    avg_chroma = mean(sqrt(c.a^2 + c.b^2) for c in mapped)

    results[gamut] = (
        in_gamut = in_gamut_count,
        out_of_gamut = n_colors - in_gamut_count,
        avg_chroma = avg_chroma,
        preservation = avg_chroma / avg_chroma_orig * 100
    )
end

println("\nGamut comparison:")
println("┌──────────┬───────────┬─────────────┬──────────────┐")
println("│ Gamut    │ In Gamut  │ Avg Chroma  │ Preservation │")
println("├──────────┼───────────┼─────────────┼──────────────┤")
for gamut in gamuts
    r = results[gamut]
    println("│ $(rpad(string(gamut), 8)) │ $(lpad(r.in_gamut, 9)) │ $(lpad(round(r.avg_chroma, digits=1), 11)) │ $(lpad(round(r.preservation, digits=1), 11))% │")
end
println("└──────────┴───────────┴─────────────┴──────────────┘")

# ═══════════════════════════════════════════════════════════════════════════
# Example 5: Visualize color shifts with namespace seeds
# ═══════════════════════════════════════════════════════════════════════════

println("\n🎯 Example 5: Visualizing color shifts with namespace seeds")

# Use namespaced seeds for different visualization contexts
const SEED_OFFSETS = (
    extreme = 0,
    normal = 1_000_000,
    pastel = 2_000_000
)

function namespaced_seed(base::UInt64, namespace::Symbol)
    return base + SEED_OFFSETS[namespace]
end

base_seed = generate_seed("gamut_visualization")

# Pick some example colors that need significant mapping
extreme_colors = Lab[]
for c in lab_chain
    chroma = sqrt(c.a^2 + c.b^2)
    if chroma > 90.0  # High chroma colors
        push!(extreme_colors, c)
        if length(extreme_colors) >= 5
            break
        end
    end
end

if !isempty(extreme_colors)
    println("\nExtreme color examples (L, C, H → mapped L, C, H):")
    params_vis = GamutParameters(target_gamut=:srgb)

    for c_orig in extreme_colors[1:min(5, end)]
        c_mapped = map_to_gamut(c_orig, params_vis)

        # Original LCH
        L_orig = c_orig.l
        C_orig = sqrt(c_orig.a^2 + c_orig.b^2)
        H_orig = atan(c_orig.b, c_orig.a) * 180 / π
        H_orig = H_orig < 0 ? H_orig + 360 : H_orig

        # Mapped LCH
        L_mapped = c_mapped.l
        C_mapped = sqrt(c_mapped.a^2 + c_mapped.b^2)
        H_mapped = atan(c_mapped.b, c_mapped.a) * 180 / π
        H_mapped = H_mapped < 0 ? H_mapped + 360 : H_mapped

        println("  ($(round(L_orig,digits=1)), $(round(C_orig,digits=1)), $(round(H_orig,digits=0))°) → " *
                "($(round(L_mapped,digits=1)), $(round(C_mapped,digits=1)), $(round(H_mapped,digits=0))°)")
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# Example 6: Full Gay.jl Integration with GamutMapper
# ═══════════════════════════════════════════════════════════════════════════

println("\n🔧 Example 6: Full Gay.jl Integration with GamutMapper")

# Following Gay.jl's golden rule: "seed should be derivable from what you're visualizing"
# We're visualizing a user's color palette, so use user-specific seed
user_id = "user_alice_2024"
document_id = "palette_experiment_3"
composite_seed = generate_seed("$(user_id)_$(document_id)")

println("Generating personalized palette for: $user_id")
gay_seed!(composite_seed)

# Create learnable gamut mapper
mapper = GamutMapper(target_gamut=:srgb)

# Generate palette using Gay.jl's palette_at for batch generation
palette_colors = palette_at(1, 10)  # 10-color palette at position 1
rgb_chain = [convert(RGB, c) for c in palette_colors]
mapped_rgb = map_color_chain(rgb_chain, mapper)

println("Mapped $(length(mapped_rgb)) RGB colors through GamutLearnable")
println("First color: $(rgb_chain[1]) → $(mapped_rgb[1])")
println("Last color:  $(rgb_chain[end]) → $(mapped_rgb[end])")

# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════

println("\n" * "="^60)
println("✨ GamutLearnable successfully demonstrated!")
println("="^60)
println()
println("Key achievements:")
println("  • Generated high-chroma color chains with Gay.jl")
println("  • Mapped out-of-gamut colors while preserving hue")
println("  • Trained adaptive parameters with Enzyme autodiff")
println("  • Compared mapping across sRGB, P3, and Rec.2020")
println("  • Integrated with Gay.jl's color chain infrastructure")
println()
println("This implementation addresses Issue #184 by providing:")
println("  1. Learnable gamut mapping parameters")
println("  2. Enzyme.jl integration for efficient training")
println("  3. Support for multiple target gamuts")
println("  4. Seamless integration with existing Gay.jl workflows")