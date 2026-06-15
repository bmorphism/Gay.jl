using Test
using GayLearnableColor
import Gay

@testset "GayLearnableColor — jointly-learnable Okhsl color space" begin
    Dm = behaviors_gf3(24)
    Dg = graph_distance(Dict(0=>[1,2], 1=>[0,3], 2=>[0,4], 3=>[1,5], 4=>[2,5], 5=>[3,4]), 6)
    lm  = learn_colorspace(Dm; d=3, iters=500)
    lg  = learn_colorspace(Dg; d=3, iters=500)
    l1  = learn_colorspace(Dm; d=1, iters=500)

    @test lm.corr > 0.6                       # 3-D color space embeds 9-D behaviour
    @test lg.corr > 0.85                      # low-dim graph embeds near-perfectly
    @test lm.corr > l1.corr                   # color spaces matter: 3-D > 1-D hue
    @test length(lm.hexes) == 24
    @test all(h -> occursin(r"^#[0-9A-F]{6}$", h), lm.hexes)
    @test all(h -> occursin(r"^#[0-9A-F]{6}$", h), lg.hexes)

    # the colour engine is Gay.jl's own kernel (cross-substrate canon)
    @test Gay.hash_color_hex(Gay.GAY_SEED, 0) == "#B35D38"

    # deterministic (stable_seed, not Julia hash)
    @test learn_colorspace(Dg; d=3, iters=500).corr == lg.corr

    @testset "LearnableHeatColor sub-features" begin
        # Generate dummy temperature samples
        T_samples = [10.0, 20.0, 30.0, 45.0, 60.0, 80.0, 100.0]
        cmap = learn_heat_colormap(T_samples; K=5, iters=50, lr=0.01)
        
        # Test knot counts and interpolation bounds
        @test length(cmap.knots) == 5
        @test cmap.knots[1] == 10.0
        @test cmap.knots[end] == 100.0
        
        # Test interpolation function
        c_mid = interpolate_colormap(55.0, cmap)
        @test length(c_mid) == 3
        @test 0.0 <= c_mid[1] <= 1.0 # Lightness
        @test 0.0 <= c_mid[2] <= 1.0 # Saturation
        @test 0.0 <= c_mid[3] <= 360.0 # Hue
        
        # Test color rendering
        hex = get_color(55.0, cmap)
        @test occursin(r"^#[0-9A-F]{6}$", hex)
    end
end
