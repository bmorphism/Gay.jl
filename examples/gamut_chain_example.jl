# Example: Mapping Gay Chain to sRGB Gamut with Enzyme-learnable parameters
#
# Your chain has colors with high chroma (C up to 98!) that exceed sRGB gamut.
# This example shows how to learn a gamut mapping that:
#   1. Preserves hue (most perceptually important)
#   2. Compresses chroma to fit in gamut
#   3. Adjusts lightness to compensate
#   4. Uses Enzyme.jl for gradient-based optimization (when available)
#
# Run with:
#   julia examples/gamut_chain_example.jl
#
# For Enzyme-based learning:
#   julia -e 'using Pkg; Pkg.add("Enzyme")' && julia examples/gamut_chain_example.jl

using Pkg
Pkg.activate(dirname(dirname(@__FILE__)))

using Gay
using Colors: RGB, LCHab, convert

# Try to load Enzyme for autodiff-based learning
const ENZYME_AVAILABLE = try
    using Enzyme
    @info "Enzyme.jl loaded - gradient-based gamut learning enabled"
    true
catch
    @warn "Enzyme.jl not available - using manual parameter tuning"
    false
end

# ═══════════════════════════════════════════════════════════════════════════════
# YOUR CHAIN DATA (from battery cycle 23)
# ═══════════════════════════════════════════════════════════════════════════════

const CHAIN_DATA = [
    (L=9.95305151795426, C=89.12121123266927, H=109.16670705328829),    # #232100 - out of gamut!
    (L=95.64340626247366, C=75.69463862432056, H=40.578861532301225),   # #FFC196
    (L=68.83307832090246, C=52.58624293448647, H=305.8775869504176),    # #B797F5
    (L=77.01270406658392, C=50.719765707180365, H=224.57712168419232),  # #00D3FE - out of gamut!
    (L=80.30684610328687, C=31.00925970957098, H=338.5668861594303),    # #F3B4DD
    (L=87.10757626363412, C=8.713821882767803, H=80.19839549147454),    # #E4D8CA
    (L=75.92474966498482, C=57.13182126381925, H=317.5858774285715),    # #E6A0FF
    (L=67.33295337865329, C=62.4733295284763, H=107.90473523965251),    # #A1AB2D
    (L=12.016818230531934, C=39.790834705489495, H=54.01863549186114),  # #430D00 - out of gamut!
    (L=20.24941930893076, C=6.316731061999381, H=181.28556359100568),   # #263330
    (L=68.92133115422948, C=3.962701273577207, H=82.54499708853153),    # #ACA7A1
    (L=28.685339908683037, C=29.288286562638422, H=223.27136465880565), # #004D62 - out of gamut!
    (L=4.342355432062184, C=13.499979374325699, H=133.4646290114955),   # #021300 - out of gamut!
    (L=27.414759014376987, C=8.735175349709479, H=19.421693716272557),  # #4E3C3C
    (L=90.65230031650403, C=34.211009968606945, H=66.9328903252508),    # #FFD9A8
    (L=25.7167729837364, C=1.665747430769271, H=234.35513798098134),    # #3A3D3E
    (L=58.80375174074871, C=2.189760028829779, H=350.1804627887977),    # #918C8E
    (L=50.54210972073506, C=46.737904999077394, H=57.451736335861156),  # #AF6535
    (L=62.12991336886255, C=72.50368716334194, H=124.21928439533164),   # #68A617 - borderline
    (L=7.255156262785755, C=98.86696191681608, H=8.573000391080656),    # #750000 - WAY out of gamut!
    (L=73.67885130891794, C=64.16166590749516, H=260.54781611975665),   # #00C1FF - out of gamut!
    (L=49.066022993728176, C=85.5860083567706, H=3.2767068869989346),   # #ED0070 - out of gamut!
    (L=45.36158016576941, C=69.57368830782679, H=51.3370126048211),     # #B84705 - borderline
    (L=66.36817064239906, C=87.38519725362308, H=164.96931844436997),   # #00C175 - out of gamut!
]

# ═══════════════════════════════════════════════════════════════════════════════
# CHECK WHICH COLORS ARE OUT OF GAMUT
# ═══════════════════════════════════════════════════════════════════════════════

println("=" ^ 70)
println("ORIGINAL CHAIN: Checking gamut status")
println("=" ^ 70)

for (i, c) in enumerate(CHAIN_DATA)
    lch = LCHab(c.L, c.C, c.H)
    rgb = convert(RGB, lch)
    in_gamut = is_in_gamut(rgb, GaySRGBGamut())
    dist = gamut_distance(lch, GaySRGBGamut())

    status = in_gamut ? "✓ IN GAMUT" : "✗ OUT (dist=$(round(dist, digits=3)))"
    println("Cycle $(i-1): L=$(round(c.L, digits=1)), C=$(round(c.C, digits=1)), H=$(round(c.H, digits=1)) → $status")
end

# ═══════════════════════════════════════════════════════════════════════════════
# LEARN GAMUT MAPPING
# ═══════════════════════════════════════════════════════════════════════════════

println("\n" * "=" ^ 70)
println("LEARNING GAMUT MAPPING")
println("=" ^ 70)

# Convert to LCH colors
lch_colors = [LCHab(c.L, c.C, c.H) for c in CHAIN_DATA]

# Initialize parameters for sRGB gamut
params = GamutParameters(gamut=:srgb)

println("Initial parameters:")
println("  chroma_compress: $(params.chroma_compress)")
println("  chroma_L: [$(params.chroma_L_a), $(params.chroma_L_b), $(params.chroma_L_c)]")

# Convert to tuple format for learning
colors_tuple = [(Float64(c.L), Float64(c.C), Float64(c.H)) for c in CHAIN_DATA]

if ENZYME_AVAILABLE
    # Use Enzyme autodiff for gradient-based learning
    println("\n🧬 Using Enzyme autodiff for gamut learning...")
    enzyme_learn_gamut!(params, colors_tuple, lr=0.05, epochs=200, verbose=true)
else
    # Manual parameter tuning (fallback when Enzyme not available)
    # Based on analysis: most out-of-gamut colors have high C and extreme L
    println("\n🔧 Using manual parameter tuning (load Enzyme for autodiff)...")

    # Reduce chroma more aggressively
    params.chroma_compress = 0.55  # 55% of original chroma

    # Less chroma at lightness extremes (L near 0 or 100)
    params.chroma_L_a = 1.2
    params.chroma_L_b = -0.4
    params.chroma_L_c = -0.8

    # Slightly boost lightness when we desaturate
    params.lightness_boost = 0.02
    params.lightness_chroma_factor = 0.0005
end

println("\nFinal parameters:")
println("  chroma_compress: $(params.chroma_compress)")
println("  chroma_L: [$(params.chroma_L_a), $(params.chroma_L_b), $(params.chroma_L_c)]")
println("  chroma_H_cos: [$(params.chroma_H_cos1), $(params.chroma_H_cos2)]")
println("  chroma_H_sin: [$(params.chroma_H_sin1), $(params.chroma_H_sin2)]")

# ═══════════════════════════════════════════════════════════════════════════════
# MAP CHAIN TO GAMUT
# ═══════════════════════════════════════════════════════════════════════════════

println("\n" * "=" ^ 70)
println("MAPPED CHAIN: All colors in sRGB gamut")
println("=" ^ 70)

chain = chain_to_gamut(lch_colors, params)

for (i, (orig, mapped, rgb)) in enumerate(zip(chain.original_lch, chain.mapped_lch, chain.mapped_rgb))
    orig_c = round(orig.c, digits=1)
    mapped_c = round(mapped.c, digits=1)
    chroma_ratio = orig.c > 0 ? round(100 * mapped.c / orig.c, digits=0) : 100

    # Convert to hex
    r = round(Int, clamp(red(rgb), 0, 1) * 255)
    g = round(Int, clamp(green(rgb), 0, 1) * 255)
    b = round(Int, clamp(blue(rgb), 0, 1) * 255)
    hex = "#" * uppercase(string(r, base=16, pad=2) * string(g, base=16, pad=2) * string(b, base=16, pad=2))

    in_gamut = is_in_gamut(rgb, GaySRGBGamut())
    status = in_gamut ? "✓" : "✗"

    println("Cycle $(i-1): C=$(orig_c)→$(mapped_c) ($(chroma_ratio)%) $hex $status")
end

# ═══════════════════════════════════════════════════════════════════════════════
# VERIFY ALL IN GAMUT
# ═══════════════════════════════════════════════════════════════════════════════

println("\n" * "=" ^ 70)
all_in_gamut = verify_chain_in_gamut(chain)
if all_in_gamut
    println("✓ ALL $(length(chain.mapped_rgb)) COLORS ARE IN sRGB GAMUT")
else
    println("✗ SOME COLORS STILL OUT OF GAMUT")
end
println("=" ^ 70)

# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT: NEW CHAIN EDN
# ═══════════════════════════════════════════════════════════════════════════════

println("\n;; Updated chain with gamut-mapped colors:")
println("{:chain [")
for (i, (lch, rgb)) in enumerate(zip(chain.mapped_lch, chain.mapped_rgb))
    r = round(Int, clamp(red(rgb), 0, 1) * 255)
    g = round(Int, clamp(green(rgb), 0, 1) * 255)
    b = round(Int, clamp(blue(rgb), 0, 1) * 255)
    hex = "#" * uppercase(string(r, base=16, pad=2) * string(g, base=16, pad=2) * string(b, base=16, pad=2))

    println("  {:cycle $(i-1) :hex \"$hex\" :L $(round(lch.l, digits=2)) :C $(round(lch.c, digits=2)) :H $(round(lch.h, digits=2))}")
end
println("]}")
