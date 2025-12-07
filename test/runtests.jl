using Test
using Gay
using Aqua
using Colors: RGB
using SplittableRandoms: SplittableRandom

# Include abductive tests
include("abductive_tests.jl")

# Include QUIC path probe tests
include("quic_tests.jl")

# Include fuzz tests (meta-testing the verification system)
include("fuzz_tests.jl")

# Include Jepsen-style meta-fuzzing (fuzz the fuzzers)
include("jepsen_fuzz.jl")

@testset "Gay.jl" begin
    @testset "Aqua.jl" begin
        Aqua.test_all(Gay; deps_compat=(check_extras=false,))
    end
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
    
    @testset "KernelAbstractions SPMD Colors" begin
        using Gay: ka_colors, ka_colors!, xor_fingerprint, hash_color
        
        # Basic generation
        colors = ka_colors(1000, 42)
        @test size(colors) == (1000, 3)
        @test eltype(colors) == Float32
        @test all(0.0f0 .<= colors .<= 1.0f0)
        
        # In-place generation
        buf = zeros(Float32, 500, 3)
        ka_colors!(buf, 1337)
        @test all(0.0f0 .<= buf .<= 1.0f0)
        
        # hash_color produces Float32
        r, g, b = hash_color(UInt64(42), UInt64(1))
        @test r isa Float32
        @test g isa Float32
        @test b isa Float32
    end
    
    @testset "XOR Fingerprint SPI Verification" begin
        using Gay: ka_colors, xor_fingerprint
        
        # Same seed = same fingerprint (SPI core guarantee)
        fp1 = xor_fingerprint(ka_colors(10000, 42))
        fp2 = xor_fingerprint(ka_colors(10000, 42))
        @test fp1 == fp2
        
        # Different seeds = different fingerprints
        fp3 = xor_fingerprint(ka_colors(10000, 1337))
        @test fp1 != fp3
        
        # Fingerprint is deterministic across multiple runs
        for _ in 1:5
            @test xor_fingerprint(ka_colors(10000, 42)) == fp1
        end
    end
    
    @testset "SPI Multi-Seed Verification" begin
        using Gay: ka_colors, ka_colors!, xor_fingerprint, hash_color
        using KernelAbstractions: CPU
        
        # Test various seeds for SPI correctness
        test_seeds = [
            0,                          # Edge case: zero
            1,                          # Minimal
            42,                         # Classic
            1337,                       # Leetspeak
            42069,                      # Gay.jl gallery seed
            0x6761795f636f6c6f,         # GAY_SEED ("gay_colo")
            0x78656e6f66656d21,         # XF_SEED ("xenofem!")
            typemax(Int64),             # Maximum Int64
            rand(UInt64),               # Random seed
        ]
        
        n = 1000  # Colors per test
        
        for seed in test_seeds
            @testset "seed=$seed" begin
                # Generate reference via sequential loop
                ref_colors = zeros(Float32, n, 3)
                for i in 1:n
                    r, g, b = hash_color(UInt64(seed), UInt64(i))
                    ref_colors[i, 1] = r
                    ref_colors[i, 2] = g
                    ref_colors[i, 3] = b
                end
                ref_fp = xor_fingerprint(ref_colors)
                
                # Generate via KernelAbstractions
                ka_result = ka_colors(n, seed)
                ka_fp = xor_fingerprint(ka_result)
                
                # SPI: fingerprints must match
                @test ref_fp == ka_fp
                
                # Colors must be identical
                @test ref_colors ≈ ka_result
                
                # Reproducibility: generate again
                ka_result2 = ka_colors(n, seed)
                @test xor_fingerprint(ka_result2) == ka_fp
            end
        end
    end
    
    @testset "SPI Workgroup Independence" begin
        using Gay: ka_colors!, xor_fingerprint
        using KernelAbstractions: CPU
        
        seed = 42069
        n = 10000
        
        # Reference fingerprint
        ref = zeros(Float32, n, 3)
        ka_colors!(ref, seed; backend=CPU(), workgroup=256)
        ref_fp = xor_fingerprint(ref)
        
        # Different workgroup sizes must produce identical results
        for ws in [1, 16, 32, 64, 128, 256, 512, 1024]
            colors = zeros(Float32, n, 3)
            ka_colors!(colors, seed; backend=CPU(), workgroup=ws)
            @test xor_fingerprint(colors) == ref_fp
        end
    end
    
    @testset "Backend Switching" begin
        using Gay: set_backend!, get_backend
        using KernelAbstractions: CPU
        
        # Default is CPU
        original = get_backend()
        
        # Can set to CPU explicitly
        set_backend!(CPU())
        @test get_backend() isa CPU
        
        # Restore original
        set_backend!(original)
    end
end
