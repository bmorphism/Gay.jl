using Test
using Gay
using Colors: RGB

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
    
    @testset "Lisp Interface" begin
        c = gay_random_color()
        @test c isa RGB
        
        c_p3 = gay_random_color(:p3)
        @test c_p3 isa RGB
        
        pride = gay_pride(:rainbow)
        @test length(pride) == 6
    end
end
