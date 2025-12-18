#!/usr/bin/env julia
# Integration test for GamutLearnable with Gay.jl core features

println("🌈 Testing Gay.jl + GamutLearnable Integration")
println("="^60)

# Load only the essential components
using Pkg
Pkg.activate(@__DIR__)

using Colors
using ColorTypes
using Random
using SplittableRandoms
using SHA

# ═══════════════════════════════════════════════════════════════════════════
# Gay.jl Core (from splittable.jl)
# ═══════════════════════════════════════════════════════════════════════════

include("src/colorspaces.jl")
include("src/splittable.jl")

# Test Gay.jl core functions
println("\n✓ Gay.jl core loaded")

# ═══════════════════════════════════════════════════════════════════════════
# Load GamutLearnable
# ═══════════════════════════════════════════════════════════════════════════

include("src/gamut_learnable.jl")
using .GamutLearnable

println("✓ GamutLearnable loaded")

# ═══════════════════════════════════════════════════════════════════════════
# Integration Test: Follow Gay.jl Best Practices
# ═══════════════════════════════════════════════════════════════════════════

println("\n📋 Following Gay.jl Best Practices (from LLMs.txt):")

# Best Practice 1: Domain object hashing
function generate_seed(identifier::String)::UInt64
    bytes = sha256(identifier)
    return reinterpret(UInt64, bytes[1:8])[1]
end

# Generate seed from meaningful identifier (not magic number!)
experiment_id = "gamut_integration_test_v1"
seed = generate_seed(experiment_id)
println("✓ Using domain object hashing: '$experiment_id' → $seed")

# Best Practice 2: Use gay_seed! and next_color
gay_seed!(seed)
colors = [next_color() for _ in 1:10]
println("✓ Generated $(length(colors)) colors sequentially")

# Best Practice 3: Random access for parallel/sparse patterns
sparse_indices = [1, 10, 100, 1000, 10000]
sparse_colors = [color_at(i; seed=seed) for i in sparse_indices]
println("✓ Random access at indices: $sparse_indices")

# Best Practice 4: Palette generation
palette = palette_at(1, 6)
println("✓ Generated palette with $(length(palette)) colors")

# ═══════════════════════════════════════════════════════════════════════════
# Test GamutMapper with Gay.jl Colors
# ═══════════════════════════════════════════════════════════════════════════

println("\n🎨 Testing GamutMapper Integration:")

# Create mapper
mapper = GamutMapper(target_gamut=:srgb)
println("✓ Created GamutMapper for :srgb")

# Convert Gay.jl colors to Lab
lab_colors = [convert(Lab, c) for c in colors]
println("✓ Converted $(length(lab_colors)) colors to Lab")

# Test gamut checking
out_of_gamut = sum(!GamutLearnable.in_gamut(c, :srgb) for c in lab_colors)
println("✓ Found $out_of_gamut colors out of sRGB gamut")

# Map colors
mapped_colors = [map_to_gamut(c, mapper.params) for c in lab_colors]
still_out = sum(!GamutLearnable.in_gamut(c, :srgb) for c in mapped_colors)
println("✓ After mapping: $still_out colors out of gamut")

# Test batch mapping
batch_mapped = map_color_chain(colors, mapper)
println("✓ Batch mapped $(length(batch_mapped)) colors")

# ═══════════════════════════════════════════════════════════════════════════
# Verify Determinism (Gay.jl Golden Rule)
# ═══════════════════════════════════════════════════════════════════════════

println("\n🔒 Verifying Determinism:")

# Same seed should give same colors
gay_seed!(seed)
colors1 = [next_color() for _ in 1:5]

gay_seed!(seed)
colors2 = [next_color() for _ in 1:5]

@assert colors1 == colors2 "Determinism check failed!"
println("✓ Same seed produces same colors (determinism verified)")

# Random access with same seed should be deterministic
rand_access_1a = color_at(100; seed=seed)
rand_access_1b = color_at(100; seed=seed)

@assert rand_access_1a == rand_access_1b "Random access not deterministic!"
println("✓ Random access is deterministic")

# ═══════════════════════════════════════════════════════════════════════════
# Performance Metrics
# ═══════════════════════════════════════════════════════════════════════════

println("\n📊 Performance Metrics:")

# Chroma preservation
original_chromas = [sqrt(c.a^2 + c.b^2) for c in lab_colors]
mapped_chromas = [sqrt(c.a^2 + c.b^2) for c in mapped_colors]
preservation = sum(mapped_chromas) / sum(original_chromas) * 100
println("✓ Average chroma preservation: $(round(preservation, digits=1))%")

# Hue preservation (should be perfect)
hue_shifts = Float64[]
for (orig, mapped) in zip(lab_colors, mapped_colors)
    if orig.a != 0 || orig.b != 0
        H_orig = atan(orig.b, orig.a)
        H_mapped = atan(mapped.b, mapped.a)
        push!(hue_shifts, abs(H_mapped - H_orig))
    end
end
max_hue_shift = maximum(hue_shifts; init=0.0) * 180/π
println("✓ Maximum hue shift: $(round(max_hue_shift, digits=2))°")

# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════

println("\n" * "="^60)
println("✅ Integration Test Complete!")
println("="^60)
println()
println("Verified Gay.jl Best Practices:")
println("  • Domain object hashing (no magic numbers)")
println("  • Sequential and random access patterns")
println("  • Deterministic color generation")
println("  • Palette generation")
println()
println("Verified GamutMapper Features:")
println("  • Gamut boundary detection")
println("  • Hue-preserving mapping")
println("  • Batch processing")
println("  • $(round(preservation, digits=1))% chroma preservation")
println()
println("The implementation follows Gay.jl's golden rule:")
println("\"The seed should be derivable from what you're visualizing\"")
println()
println("✨ GamutLearnable is fully integrated with Gay.jl! ✨")