using Test
using Gay
using Colors
using SplittableRandoms
using OhMyThreads
using Pigeons

@testset "Parallel Color Generation (OhMyThreads + Pigeons SPI)" begin
    using Gay: parallel_palette, parallel_colors_at
    
    # parallel_palette returns correct number of colors
    p = parallel_palette(10; seed=42)
    @test length(p) == 10
    @test all(c -> c isa RGB, p)
    
    # Same seed = same colors (SPI)
    p1 = parallel_palette(10; seed=1337)
    p2 = parallel_palette(10; seed=1337)
    @test p1 == p2
    
    # Different seeds = different colors
    p3 = parallel_palette(10; seed=9999)
    @test p1 != p3
    
    # parallel_colors_at works
    colors = parallel_colors_at([1, 5, 10]; seed=42)
    @test length(colors) == 3
    @test colors[1] == color_at(1; seed=42)
    @test colors[2] == color_at(5; seed=42)
    @test colors[3] == color_at(10; seed=42)
end
