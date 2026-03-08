# Test Interaction Entropy module
using Test

include("../src/interaction_entropy.jl")
using .InteractionEntropy

println("=" ^ 70)
println("INTERACTION ENTROPY: Symbolic Distillation of Positional Noise")
println("=" ^ 70)

# Create a joint interaction space
space = JointInteractionSpace(field_resolution=32)

# Simulate writing "Hi" - two strokes with characteristic entropy patterns
println("\n📝 Creating interaction strokes...")

# Stroke 1: Letter "H" - vertical down, lift, vertical down, lift, horizontal
stroke1 = InteractionStroke()
# Left vertical of H
for i in 0:10
    t = i / 10
    add_point!(stroke1, 0.1, 0.2 + t * 0.4; t=t, p=0.6 + 0.2*sin(t*π))
end

compute_entropy!(stroke1)
add_stroke!(space, stroke1)

println("  Stroke 1 (H left): $(length(stroke1.points)) points")
println("    Mean entropy: $(round(stroke1.mean_entropy, digits=3))")
println("    Max entropy: $(round(stroke1.max_entropy, digits=3))")
println("    Symbol boundaries: $(stroke1.symbol_boundaries)")
println("    Trit: $(stroke1.trit)")

# Stroke 2: Right vertical of H
stroke2 = InteractionStroke()
for i in 0:10
    t = i / 10
    add_point!(stroke2, 0.3, 0.2 + t * 0.4; t=t, p=0.5)
end
compute_entropy!(stroke2)
add_stroke!(space, stroke2)

# Stroke 3: Horizontal bar of H (high velocity, different character)
stroke3 = InteractionStroke()
for i in 0:8
    t = i / 8
    # Add some curvature (slight arc)
    y_offset = 0.02 * sin(t * π)
    add_point!(stroke3, 0.1 + t * 0.2, 0.4 + y_offset; t=t, p=0.7)
end
compute_entropy!(stroke3)
add_stroke!(space, stroke3)

# Stroke 4: Letter "i" - vertical with dot
stroke4 = InteractionStroke()
for i in 0:8
    t = i / 8
    add_point!(stroke4, 0.5, 0.35 + t * 0.25; t=t, p=0.6)
end
compute_entropy!(stroke4)
add_stroke!(space, stroke4)

# Stroke 5: dot of "i" - very short, high pressure
stroke5 = InteractionStroke()
add_point!(stroke5, 0.5, 0.25; t=0.0, p=0.9)
add_point!(stroke5, 0.51, 0.25; t=0.05, p=0.95)
add_point!(stroke5, 0.5, 0.26; t=0.1, p=0.85)
compute_entropy!(stroke5)
add_stroke!(space, stroke5)

println("\n📊 Entropy Statistics by Stroke:")
for (i, stroke) in enumerate(space.strokes)
    trit_name = stroke.trit isa Plus ? "+1" : stroke.trit isa Ergodic ? "0" : "-1"
    println("  Stroke $i: H̄=$(round(stroke.mean_entropy, digits=2)), " *
            "H_max=$(round(stroke.max_entropy, digits=2)), " *
            "var=$(round(stroke.entropy_variance, digits=3)), " *
            "trit=$trit_name")
end

# Learn the colorspace
println("\n🎨 Learning colorspace (symbolic distillation)...")
learn_colorspace!(space; learning_rate=0.05f0, iterations=50)

println("  Mutual Information I(position; color): $(round(space.mutual_info, digits=3)) bits")

# Show learned colors
println("\n🌈 Learned Stroke Colors:")
for (i, stroke) in enumerate(space.strokes)
    r, g, b = stroke.stroke_color
    hex = "#" * join([string(Int(round(c*255)), base=16, pad=2) for c in (r,g,b)], "")
    println("  Stroke $i: $hex (H̄=$(round(stroke.mean_entropy, digits=2)))")
end

# Tokenize by entropy (use low threshold since our test strokes are simple)
println("\n🔤 Symbolic Tokens (distilled from high-entropy regions):")
tokens = tokenize_by_entropy(space; min_entropy=0.01f0)  # Low threshold for test data
println("  Found $(length(tokens)) tokens")

for (i, token) in enumerate(tokens)
    r, g, b = token.color
    hex = "#" * join([string(Int(round(c*255)), base=16, pad=2) for c in (r,g,b)], "")
    println("  Token $i: $(token.symbol)")
    println("    Stroke $(token.stroke_idx), indices $(token.start_idx):$(token.end_idx)")
    println("    Entropy: $(round(token.entropy, digits=3))")
    println("    Color: $hex")
end

# Entropy field analysis
println("\n📈 Entropy Field Critical Points:")
field = EntropyField(space)
criticals = critical_points(field; threshold=0.15f0)

maxima = filter(c -> c[3] == :maximum, criticals)
saddles = filter(c -> c[3] == :saddle, criticals)
minima = filter(c -> c[3] == :minimum, criticals)

println("  Maxima (symbol centers): $(length(maxima))")
println("  Saddles (symbol boundaries): $(length(saddles))")
println("  Minima (between symbols): $(length(minima))")

# Fingerprint
println("\n🔐 Interaction Fingerprint:")
fp = interaction_fingerprint(space)
println("  Fingerprint: 0x$(string(fp, base=16, pad=16))")

# Tests
println("\n" * "=" ^ 70)
println("TESTS")
println("=" ^ 70)

@testset "Interaction Entropy" begin
    # Basic structure
    @test length(space.strokes) == 5
    @test all(s -> length(s.points) > 0, space.strokes)
    
    # Entropy is computed
    @test all(s -> s.mean_entropy > 0, space.strokes)
    @test all(s -> s.max_entropy >= s.mean_entropy, space.strokes)
    
    # Colors are learned
    @test space.mutual_info > 0  # Some information is captured
    
    # Tokens are extracted
    @test length(tokens) > 0
    @test all(t -> t.entropy > 0, tokens)
    
    # Fingerprint is deterministic
    @test interaction_fingerprint(space) == fp
    
    # Entropy field has structure
    @test sum(space.entropy_field) > 0
end

println("\n" * "=" ^ 70)
println("CONCEPTUAL SUMMARY")
println("=" ^ 70)

println("""

SYMBOLIC DISTILLATION OF POSITIONALLY-DEPENDENT NOISE
══════════════════════════════════════════════════════

The key insight: HIGH ENTROPY = SYMBOLIC EMERGENCE

Where does entropy peak?
  1. Stroke boundaries (pen lift) → Word/character boundaries
  2. Velocity discontinuities → Emphasis markers
  3. Curvature extrema → Character identity
  4. Pressure spikes → Grapheme features

The learning process:
  Position (x,y,t) + Noise σ²(position) → Entropy H → Color (R,G,B)

What gets distilled:
  - Low entropy regions: Similar colors (noise suppressed)
  - High entropy regions: Distinct colors (symbols emerge)
  - Critical points: Where symbols crystallize

The color-space is "learnable" because:
  - Mutual information I(position; color) is maximized
  - Entropy structure is preserved
  - Symbolic tokens emerge at entropy peaks

This is the JOINT INTERACTION SPACE:
  - Position provides spatial grounding
  - Time provides causal ordering
  - Color provides learned embedding
  - Entropy localizes where symbols emerge
""")

println("\n✅ Interaction Entropy module working correctly")
