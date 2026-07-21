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

# Include SDF-style propagator tests
include("propagator_test.jl")

# Include regression tests for ternary/gamut systems
include("regression_ternary.jl")

# Non-Riemannian gate: strict subadditivity on collinear triplets, with
# tolerances DERIVED from the SatReadout defect identity (not tuned).
# Metrics failing the collinear-triplet test are barred from large-difference use.
include("test_nonriemannian_gate.jl")

# Exa loop objectives extensions: Intrinsic HSL, Multilateral Clearing, Cellular Sheaf Čech Cohomology
include("test_exa_loop_extensions.jl")

# Jank-like LispSyntax projection into GATlab/Catlab-style artifacts
include("lisp_gatlab_bridge_tests.jl")

@testset "Gay.jl" begin
    @testset "Aqua.jl" begin
        # Individual tests for better diagnostics
        # See: https://github.com/JuliaTesting/Aqua.jl
        
        @testset "Ambiguities" begin
            # recursive=false to avoid checking dependencies
            Aqua.test_ambiguities(Gay; recursive=false)
        end
        
        @testset "Unbound args" begin
            # 1 known unbound: GaySpectral uses NTuple which generates Vararg
            Aqua.test_unbound_args(Gay; broken=true)
        end
        
        @testset "Undefined exports" begin
            # Keep Aqua's zero-tolerance check visible while ratcheting the
            # existing monorepo debt: new undefined exports must fail CI.
            undefined_exports = filter(
                name -> !isdefined(Gay, name),
                names(Gay; all=false, imported=false),
            )
            @test length(undefined_exports) <= 2104
            Aqua.test_undefined_exports(Gay; broken=true)
        end
        
        @testset "Stale deps" begin
            # Ignore optional GPU backends and DuckDB
            Aqua.test_stale_deps(Gay; ignore=[:Metal, :CUDA, :AMDGPU, :Rimu, :Carlo, :DuckDB])
        end
        
        @testset "Deps compat" begin
            Aqua.test_deps_compat(Gay; check_extras=false)
        end
        
        @testset "Piracies" begin
            # Ensure we don't define methods on types we don't own
            Aqua.test_piracies(Gay)
        end
        
        @testset "Project extras" begin
            Aqua.test_project_extras(Gay)
        end
    end

    @testset "Restored substantive exports" begin
        bundle = SeedBundle(UInt64[11, 22, 33], Dict{Symbol, Vector{UInt64}}(), 0, 0, "test")
        @test seeds_range(bundle, 1:3) == UInt64[11, 22, 33]
        @test verify_bundle_spi(bundle)

        # ThreeMatch defines its own Trit values. Keep them qualified because
        # Gay's established MINUS/ERGODIC/PLUS names denote Polarity values.
        three_match = Gay.ThreeMatch
        @test three_match.trit_from_integer(-4) == three_match.MINUS
        @test three_match.trit_from_integer(3) == three_match.ERGODIC
        @test three_match.trit_from_integer(4) == three_match.PLUS
        items = [
            ("minus", three_match.MINUS),
            ("zero", three_match.ERGODIC),
            ("plus", three_match.PLUS),
        ]
        world = world_three_match(items)
        @test length(world) == 1
        @test world.valid_count == 1
        @test verify_three_match(only(world.triangles))

        @test semiring_add(TropicalMinPlus, 3.0, 5.0) == 3.0
        @test semiring_mul(TropicalMinPlus, 3.0, 5.0) == 8.0
        @test semiring_zero(TropicalMinPlus) == Inf
        @test semiring_one(TropicalMinPlus) == 0.0
        @test all(result -> result.passed, verify_semiring_laws(TropicalMinPlus, [1.0, 2.0, 3.0]))
    end

    @testset "Reafference witnesses continuity, not identity" begin
        seed = UInt64(1069)
        ticks = [TritTick(0), TritTick(EPOCH_1_HZ)]
        expected = Gay.Reafference.predict(seed, ticks)

        proof = Gay.Reafference.verify(seed, ticks, expected, expected; threshold=1.0)
        @test proof.verified

        forged = fill((0.0f0, 0.0f0, 0.0f0), length(ticks))
        forged_proof = Gay.Reafference.verify(seed, ticks, forged, forged; threshold=1.0)
        @test !forged_proof.verified
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
    
    @testset "Trit-Tick Time Base" begin
        # Constants
        @test EPOCH_1_HZ == 141_120_000
        @test FLICKS_PER_TICK == 5
        @test GF3_QUANTUM == 47_040_000
        @test HUE_DEGREE_QUANTUM == 392_000

        # TritTick construction
        t = TritTick(1000)
        @test t.tick == 1000
        @test t.epoch == 1

        # GF(3) trit: second 0 → trit, second 1 → trit, etc.
        @test trit(TritTick(0)) in Int8[-1, 0, 1]
        @test trit(TritTick(EPOCH_1_HZ)) in Int8[-1, 0, 1]
        # Three consecutive seconds cycle through all three trits
        trits_3sec = [trit(TritTick(UInt64(s) * EPOCH_1_HZ)) for s in 0:2]
        @test sort(trits_3sec) == [-1, 0, 1]

        # Conservation check
        @test conservation_check([TritTick(0), TritTick(EPOCH_1_HZ), TritTick(2*EPOCH_1_HZ)], 0)

        # Modality divisibility: 44100 Hz audio = 3200 trit-ticks/sample
        @test EPOCH_1_HZ % 44100 == 0
        @test EPOCH_1_HZ ÷ 44100 == 3200

        # fits() — EEG 256 Hz should fit in a 1-second interval
        @test fits(EPOCH_1_HZ, :eeg_256)
        @test fits(EPOCH_1_HZ, :audio_44100)
        @test fits(EPOCH_1_HZ, :heartbeat)

        # Flick conversion exact
        t = TritTick(1000)
        @test to_flicks(t) == 5000
        @test from_flicks(5000) == t
        @test from_flicks(5003) == t  # rounds down, loses 3 flicks

        # LogicalTicks source
        lt = LogicalTicks()
        t1 = current_tick(lt)
        t2 = current_tick(lt)
        @test t2.tick == t1.tick + 1  # monotonic increment

        # TritTick-aware hash_color produces different colors than bare index
        seed = UInt64(42)
        t_maker = TritTick(EPOCH_1_HZ * 0)  # second 0
        t_coord = TritTick(EPOCH_1_HZ * 1)  # second 1
        c1 = hash_color(seed, t_maker)
        c2 = hash_color(seed, t_coord)
        # Different ticks → different colors (with overwhelming probability)
        @test c1 != c2

        # color_at with TritTick
        c = color_at(TritTick(42))
        @test c isa RGB{Float32}

        # GayRNG now carries tick
        gay_seed!(42)
        state = gay_rng_state()
        @test hasproperty(state, :tick)
        @test hasproperty(state, :trit)
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
    
    @testset "Color Entropy Sources" begin
        using Gay: read_entropy, inject_watts!, inject_entropy!
        using Gay: inject_heartbeat!, inject_aqi!, inject_commit!, inject_jerk!
        using Gay: agreement, disagreement_signal, composite_from_readings
        using Gay: CompositeTicks, current_colored_tick

        @testset "ColoredTick creation" begin
            ct = ColoredTick(TritTick(1000), Int8(1), UInt64(42), 0.9f0, :test)
            @test ct.measured_trit == Int8(1)
            @test ct.confidence == 0.9f0
            @test ct.source == :test
        end

        @testset "classify_trit" begin
            @test classify_trit(100.0, 5.0, 30.0) == Int8(1)   # above high
            @test classify_trit(15.0, 5.0, 30.0) == Int8(0)    # between
            @test classify_trit(2.0, 5.0, 30.0) == Int8(-1)    # below low
            @test classify_trit(30.0, 5.0, 30.0) == Int8(0)    # at boundary = middle
            @test classify_trit(5.0, 5.0, 30.0) == Int8(0)     # at boundary = middle
        end

        @testset "entropy_mix" begin
            ct = ColoredTick(TritTick(0), Int8(1), UInt64(0xdeadbeef), 1.0f0, :test)
            mixed = entropy_mix(UInt64(42), ct)
            @test mixed isa UInt64
            @test mixed != UInt64(42)  # mixing changed the seed
            # Deterministic
            @test entropy_mix(UInt64(42), ct) == mixed
        end

        @testset "DrandSource" begin
            ds = DrandSource()
            @test source_name(ds) == :drand
            @test source_mortality(ds) == Immortal
            ct1 = read_entropy(ds)
            ct2 = read_entropy(ds)
            @test ct1.source == :drand
            @test ct2.tick.tick > ct1.tick.tick  # advancing
            @test ct1.measured_trit in Int8[-1, 0, 1]
        end

        @testset "EnergySource" begin
            es = EnergySource("127.0.0.1")
            @test source_name(es) == :energy
            @test source_mortality(es) == Immortal

            # Inject readings at different levels
            ct_high = inject_watts!(es, 50.0)
            @test ct_high.measured_trit == Int8(1)  # >30W = maker

            ct_mid = inject_watts!(es, 15.0)
            @test ct_mid.measured_trit == Int8(0)   # 5-30W = coordinator

            ct_low = inject_watts!(es, 2.0)
            @test ct_low.measured_trit == Int8(-1)  # <5W = checker
        end

        @testset "BCISource" begin
            bs = BCISource(device=:muse2)
            @test source_name(bs) == :bci
            @test source_mortality(bs) == Mortal

            ct_high = inject_entropy!(bs, 0.9)
            @test ct_high.measured_trit == Int8(1)  # high entropy = active cognition

            ct_low = inject_entropy!(bs, 0.1)
            @test ct_low.measured_trit == Int8(-1)  # low entropy = rest
        end

        @testset "HeartbeatSource" begin
            hs = HeartbeatSource()
            @test source_mortality(hs) == Mortal

            ct = inject_heartbeat!(hs, 72.0, 55.0)  # normal BPM, high HRV
            @test ct.measured_trit == Int8(1)        # high HRV = coherent

            ct_stress = inject_heartbeat!(hs, 110.0, 10.0)  # high BPM, low HRV
            @test ct_stress.measured_trit == Int8(-1)         # low HRV = stress

            # Death interval
            hs.alive = false
            ct_dead = read_entropy(hs)
            @test ct_dead.confidence == 0.0f0
        end

        @testset "AirQualitySource" begin
            aqs = AirQualitySource()
            ct_good = inject_aqi!(aqs, 25.0, 5.0)   # good air
            @test ct_good.measured_trit == Int8(1)    # inverted: low AQI = good = maker

            ct_bad = inject_aqi!(aqs, 200.0, 100.0)  # bad air
            @test ct_bad.measured_trit == Int8(-1)    # inverted: high AQI = bad = checker
        end

        @testset "GitSource" begin
            gs = GitSource(".")
            @test source_mortality(gs) == Immortal
            ct = inject_commit!(gs, UInt64(0xdeadbeefcafe1234))
            @test ct.source == :git
            @test ct.measured_trit in Int8[-1, 0, 1]
        end

        @testset "AccelerometerSource" begin
            as = AccelerometerSource(rate=100)
            @test source_mortality(as) == SemiMortal

            ct_pos = inject_jerk!(as, 1.0)   # positive jerk
            @test ct_pos.measured_trit == Int8(1)

            ct_neg = inject_jerk!(as, -1.0)  # negative jerk
            @test ct_neg.measured_trit == Int8(-1)

            ct_zero = inject_jerk!(as, 0.0)  # near zero
            @test ct_zero.measured_trit == Int8(0)
        end

        @testset "CompositeSource" begin
            es = EnergySource("127.0.0.1")
            gs = GitSource(".")
            cs = CompositeSource([es, gs])
            @test source_name(cs) == :composite

            # Inject known values
            inject_watts!(es, 50.0)  # maker
            inject_commit!(gs, UInt64(0x010101))  # will produce some trit

            ct = read_entropy(cs)
            @test ct.source == :composite
            @test ct.measured_trit in Int8[-1, 0, 1]
        end

        @testset "Agreement and disagreement" begin
            makers = [
                ColoredTick(TritTick(0), Int8(1), UInt64(1), 0.9f0, :a),
                ColoredTick(TritTick(0), Int8(1), UInt64(2), 0.8f0, :b),
                ColoredTick(TritTick(0), Int8(1), UInt64(3), 0.7f0, :c),
            ]
            @test agreement(makers) == 1.0f0  # unanimous

            mixed = [
                ColoredTick(TritTick(0), Int8(1), UInt64(1), 0.9f0, :brain),
                ColoredTick(TritTick(0), Int8(-1), UInt64(2), 0.8f0, :air),
                ColoredTick(TritTick(0), Int8(0), UInt64(3), 0.7f0, :energy),
            ]
            @test agreement(mixed) ≈ 1.0f0 / 3.0f0  # maximum disagreement

            sig = disagreement_signal(mixed)
            @test :brain in sig.makers
            @test :air in sig.checkers
            @test :energy in sig.coordinators
        end

        @testset "CompositeTicks as TickSource" begin
            es = EnergySource("127.0.0.1")
            inject_watts!(es, 15.0)
            cs = CompositeSource([es])
            ct_source = CompositeTicks(cs)
            tick = current_tick(ct_source)
            @test tick isa TritTick

            colored = current_colored_tick(ct_source)
            @test colored isa ColoredTick
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
