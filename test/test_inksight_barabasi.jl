# Test InkSightBarabasi module
using Test

include("../src/inksight_barabasi.jl")
using .InkSightBarabasi

println("=" ^ 60)
println("INKSIGHT-BARABÁSI NETWORK CRITICALITY TEST")
println("=" ^ 60)

# Create network and add points simulating handwriting
net = InkNetwork()

# First stroke: letter "A" left leg
println("\n📝 Writing first stroke (A left leg)...")
for i in 0:10
    t = i / 10
    x = 0.1 + t * 0.15
    y = 0.8 - t * 0.6
    add_point!(net, x, y; t=t, p=0.5 + 0.3*sin(t*π))
end

crit1 = network_criticality(net)
println("  γ (power-law exponent): $(round(crit1.gamma, digits=2))")
println("  Criticality state: $(crit1.criticality)")
println("  Hubs detected: $(crit1.num_hubs)")

# Pen lift!
println("\n✋ PEN LIFT (phase transition)")
lift_pen!(net)

# Second stroke: letter "A" right leg
println("\n📝 Writing second stroke (A right leg)...")
for i in 0:10
    t = i / 10
    x = 0.25 + t * 0.15
    y = 0.2 + t * 0.6
    add_point!(net, x, y; t=t, p=0.6)
end

crit2 = network_criticality(net)
println("  γ (power-law exponent): $(round(crit2.gamma, digits=2))")
println("  Criticality state: $(crit2.criticality)")
println("  Correlation length ξ: $(round(crit2.correlation_length, digits=1))")

# Another pen lift
println("\n✋ PEN LIFT")
lift_pen!(net)

# Third stroke: letter "A" crossbar
println("\n📝 Writing third stroke (A crossbar)...")
for i in 0:8
    t = i / 8
    x = 0.12 + t * 0.26
    y = 0.4
    add_point!(net, x, y; t=t, p=0.7)
end

println("\n" * "=" ^ 60)
println("FINAL NETWORK CRITICALITY ANALYSIS")
println("=" ^ 60)

crit_final = network_criticality(net)
println("\n🔬 Barabási Metrics:")
println("  γ (power-law exponent): $(round(crit_final.gamma, digits=3))")
println("  Criticality: $(crit_final.criticality)")
println("  At phase transition: $(crit_final.is_at_phase_transition)")
println("  Giant component fraction: $(round(crit_final.giant_component_fraction, digits=3))")
println("  Correlation length ξ: $(round(crit_final.correlation_length, digits=2))")
println("  Number of hubs: $(crit_final.num_hubs)")
println("  Effective dimension: $(crit_final.effective_dimension)")

println("\n🎯 Hub Detection (Dragon Kings):")
hubs = hub_detection(net)
if isempty(hubs)
    println("  No hubs detected (uniform degree distribution)")
else
    for hub in hubs[1:min(3, length(hubs))]
        println("  Hub at stroke $(hub.stroke), index $(hub.index): degree=$(hub.degree), score=$(round(hub.hub_score, digits=2))")
    end
end

println("\n🌊 Percolation State: $(percolation_state(net))")

println("\n" * "=" ^ 60)
println("TOKENIZATION (Berent et al. InkSight)")
println("=" ^ 60)
tokens = to_tokens(net)
println("\nFirst 20 tokens:")
println("  ", join(tokens[1:min(20, length(tokens))], " "))
println("...")
println("\nTotal tokens: $(length(tokens))")

pen_lift_count = count(t -> t == "<PEN_LIFT>", tokens)
hub_count = count(t -> t == "<HUB>", tokens)
println("PEN_LIFT markers: $pen_lift_count")
println("HUB markers: $hub_count")

println("\n" * "=" ^ 60)
println("DUAL REPRESENTATION")
println("=" ^ 60)
dual = dual_representation(net)
println("\nText tokens: $(length(dual.text))")
println("Image dimensions: $(size(dual.image))")
nonzero = count(x -> x > 0, dual.image)
println("Image non-zero pixels: $nonzero")

println("\n" * "=" ^ 60)
println("BARABÁSI'S ANSWER: WHERE DOES CRITICALITY COME FROM?")
println("=" ^ 60)

gamma_val = round(crit_final.gamma, digits=2)
p_val = round(crit_final.giant_component_fraction, digits=2)
xi_val = round(crit_final.correlation_length, digits=1)
perc_state = percolation_state(net)
eff_dim = crit_final.effective_dimension

println("""

In network science terms:

1. PREFERENTIAL ATTACHMENT creates power-law P(k) ~ k^(-γ)
   → Measured γ = $gamma_val
   → Critical value is γ = 3

2. HUBS are critical nodes (dragon kings)
   → Found $(crit_final.num_hubs) hubs in this network
   → Pen lift near hub = phase transition

3. PERCOLATION THRESHOLD p_c separates phases
   → Current p = $p_val
   → State: $perc_state

4. CORRELATION LENGTH ξ diverges at criticality
   → Current ξ = $xi_val
   → ξ → ∞ means all points are correlated

5. GF(3) CONNECTION: γ = 3 is the critical exponent!
   → This is why ternary (3-valued) logic captures criticality
   → Effective dimension: $eff_dim
""")

# Actual tests
@testset "InkSightBarabasi" begin
    @test length(net.strokes) == 3
    @test crit_final.gamma > 0
    @test crit_final.effective_dimension in [1, 2, 3]
    @test pen_lift_count == 2  # Two pen lifts
    @test length(tokens) > 0
end

println("\n" * "=" ^ 60)
println("CHROMATIC VISUALIZATION (InkSight Color Scheme)")
println("=" ^ 60)

# Test InkSight's rainbow scheme
println("\n🌈 Rainbow Scheme (InkSight default):")
rainbow = rainbow_scheme()
println("  Hue range: $(rainbow.hue_start) → $(rainbow.hue_end)")
println("  Value factor: $(rainbow.value_factor) (HSV darkening)")
println("  Alpha decay: $(rainbow.alpha_start) → $(rainbow.alpha_end)")
println("  Reverse order: $(rainbow.reverse_order) (red=recent)")

# Test Gay.jl deterministic scheme
println("\n🏳️‍🌈 Gay Scheme (deterministic, seed 1069):")
gay = gay_scheme()
println("  Seed-derived hue start: $(round(gay.hue_start, digits=3))")
println("  Same value factor: $(gay.value_factor)")

# Get stroke colors
println("\n🎨 Stroke Colors (3 strokes):")
for i in 1:3
    rgb = stroke_color(i, 3, rainbow)
    hue_deg = round(rgb[1] * 360, digits=1)  # Convert back to hue for display
    trit = hue_to_trit(rgb[1])
    trit_name = trit isa Plus ? "PLUS (+1)" : trit isa Ergodic ? "ERGODIC (0)" : "MINUS (-1)"
    println("  Stroke $i: RGB($(round(rgb[1],digits=2)), $(round(rgb[2],digits=2)), $(round(rgb[3],digits=2))) → $trit_name")
end

# Test alpha gradient (point_alpha takes point_idx, total_points, scheme)
println("\n⏱️ Alpha Gradient (temporal decay):")
total_pts = 10
for i in [1, 3, 5, 8, 10]
    alpha = point_alpha(i, total_pts, rainbow)
    pct = round((i-1)/(total_pts-1)*100)
    println("  Point $i/$total_pts ($pct%): α = $(round(alpha, digits=2))")
end

# Full chromatic encoding (scheme is a keyword argument)
println("\n🔬 Chromatic Encoding of Network:")
encoding = chromatic_encoding(net; scheme=rainbow)
println("  Total strokes: $(length(encoding.stroke_colors))")
println("  GF(3) balance: $(encoding.gf3_balance) (should be 0 for balanced)")
println("  γ (criticality): $(round(encoding.gamma, digits=2))")
println("  State: $(encoding.criticality)")

for i in 1:length(encoding.stroke_colors)
    trit = encoding.stroke_trits[i]
    trit_name = trit isa Plus ? "+1" : trit isa Ergodic ? "0" : "-1"
    rgb = encoding.stroke_colors[i]
    alphas = encoding.point_alphas[i]
    println("  Stroke $i: trit=$trit_name, $(length(alphas)) points")
    if !isempty(alphas)
        println("    RGB: ($(round(rgb[1], digits=2)), $(round(rgb[2], digits=2)), $(round(rgb[3], digits=2)))")
        println("    α decay: $(round(alphas[1], digits=2)) → $(round(alphas[end], digits=2))")
    end
end

# Test hue-to-trit mapping (hue in degrees 0-360)
println("\n🎯 Hue → GF(3) Mapping (degrees):")
test_hues_deg = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0, 360.0]
for h in test_hues_deg
    trit = hue_to_trit(h)
    trit_sym = trit isa Plus ? "+1" : trit isa Ergodic ? " 0" : "-1"
    region = h < 120 ? "warm (red-yellow)" : h < 240 ? "neutral (green-cyan)" : "cool (blue-magenta)"
    println("  H=$(Int(h))° ($region) → $trit_sym")
end

# Test Gay.jl deterministic coloring (seed must be UInt64)
println("\n🌀 Gay.jl Deterministic Colors (seed 1069):")
for stroke_idx in 1:5
    rgb = gay_stroke_color(stroke_idx, UInt64(1069))
    hex = join([string(Int(round(c*255)), base=16, pad=2) for c in rgb], "")
    println("  Stroke $stroke_idx: #$hex")
end

@testset "Chromatic Visualization" begin
    # Rainbow scheme tests
    @test rainbow.reverse_order == true
    @test rainbow.value_factor ≈ 0.65
    @test rainbow.alpha_start ≈ 1.0
    @test rainbow.alpha_end ≈ 0.5
    
    # Alpha decay (point_alpha takes point_idx, total_points, scheme)
    @test point_alpha(1, 10, rainbow) ≈ 1.0
    @test point_alpha(10, 10, rainbow) ≈ 0.5
    @test point_alpha(5, 9, rainbow) ≈ 0.75  # Midpoint: linear interpolation
    
    # Hue to trit mapping (hue in DEGREES: 0-360)
    @test hue_to_trit(0.0) isa Plus      # 0° = Red = Warm = injection
    @test hue_to_trit(180.0) isa Ergodic # 180° = Cyan = Neutral = bridge
    @test hue_to_trit(280.0) isa Minus   # 280° = Blue/Violet = Cool = extraction
    
    # Chromatic encoding
    @test length(encoding.stroke_colors) == 3
    @test length(encoding.stroke_trits) == 3
    @test length(encoding.point_alphas) == 3
    
    # Gay deterministic - same seed = same color (seed must be UInt64)
    @test gay_stroke_color(1, UInt64(1069)) == gay_stroke_color(1, UInt64(1069))
    @test gay_stroke_color(1, UInt64(1069)) != gay_stroke_color(2, UInt64(1069))
end

println("\n✅ InkSightBarabasi module working correctly")
println("✅ Chromatic visualization functions verified")
