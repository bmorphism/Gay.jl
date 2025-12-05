using Test
using Gay
using Colors: RGB
using SplittableRandoms: SplittableRandom

@testset "Gay.jl" begin
    @testset "Color Spaces" begin
        @test SRGB() isa ColorSpace
        @test DisplayP3() isa ColorSpace
        @test Rec2020() isa ColorSpace
        
        custom = CustomColorSpace(
            Primaries(0.7, 0.3, 0.2, 0.7, 0.1, 0.1, 0.3127, 0.329),
            "Test"
        )
        @test custom isa ColorSpace
    end
    
    @testset "Random Color Generation" begin
        c = random_color(SRGB())
        @test c isa RGB
        @test 0.0 <= c.r <= 1.0
        @test 0.0 <= c.g <= 1.0
        @test 0.0 <= c.b <= 1.0
        
        colors = random_colors(5, DisplayP3())
        @test length(colors) == 5
        @test all(c -> c isa RGB, colors)
        
        palette = random_palette(3, Rec2020())
        @test length(palette) == 3
    end
    
    @testset "Splittable Determinism" begin
        # Same seed produces same colors
        gay_seed!(42)
        c1 = next_color()
        c2 = next_color()
        
        gay_seed!(42)
        @test next_color() == c1
        @test next_color() == c2
        
        # Random access by index
        c_at_10 = color_at(10)
        c_at_10_again = color_at(10)
        @test c_at_10 == c_at_10_again
        
        # Different seeds produce different colors
        gay_seed!(42)
        a = next_color()
        gay_seed!(1337)
        b = next_color()
        @test a != b
    end
    
    @testset "Strong Parallelism Invariance (SPI)" begin
        seed = 42069
        n = 100
        
        # Sequential generation
        sequential = [color_at(i; seed=seed) for i in 1:n]
        
        # Parallel generation (simulated)
        parallel = Vector{RGB}(undef, n)
        Threads.@threads for i in 1:n
            parallel[i] = color_at(i; seed=seed)
        end
        
        # Must be identical
        @test sequential == parallel
        
        # Reproducibility across runs
        sequential2 = [color_at(i; seed=seed) for i in 1:n]
        @test sequential == sequential2
    end
    
    @testset "Palette Generation" begin
        gay_seed!(1337)
        p1 = next_palette(6)
        @test length(p1) == 6
        
        # Indexed palette access
        p_at_5 = palette_at(5, 6)
        p_at_5_again = palette_at(5, 6)
        @test p_at_5 == p_at_5_again
    end
    
    @testset "Pride Flags" begin
        r = rainbow()
        @test length(r) == 6
        
        t = transgender()
        @test length(t) == 5
        
        bi = bisexual()
        @test length(bi) == 3
        
        nb = nonbinary()
        @test length(nb) == 4
        
        # Test with different color spaces
        r_p3 = pride_flag(:rainbow, DisplayP3())
        @test length(r_p3) == 6
    end
    
    @testset "Gamut Operations" begin
        c = RGB(0.5, 0.5, 0.5)
        @test in_gamut(c, SRGB())
        
        clamped = clamp_to_gamut(RGB(1.5, 0.5, -0.5), SRGB())
        @test in_gamut(clamped, SRGB())
    end
    
    @testset "Comrade Sky Models" begin
        gay_seed!(2017)
        
        ring = comrade_ring(1.0, 0.3)
        @test ring isa Ring
        @test ring.radius == 1.0
        @test ring.width == 0.3
        
        gauss = comrade_gaussian(0.5)
        @test gauss isa Gaussian
        
        model = sky_add(ring, gauss)
        @test model isa SkyModel
        @test length(model.components) == 2
        
        # Reproducible models
        gay_seed!(42)
        m1 = comrade_model(seed=42, style=:m87)
        m2 = comrade_model(seed=42, style=:m87)
        @test sky_show(m1) == sky_show(m2)
    end
    
    @testset "Lisp Interface" begin
        c = gay_random_color()
        @test c isa RGB
        
        c_p3 = gay_random_color(:p3)
        @test c_p3 isa RGB
        
        pride = gay_pride(:rainbow)
        @test length(pride) == 6
        
        # Deterministic via Lisp API
        gay_seed(42)
        l1 = gay_next()
        gay_seed(42)
        l2 = gay_next()
        @test l1 == l2
    end
end
