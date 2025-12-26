# Tests for Para(ZigZag) Chromatic PDMP with SPI Verification
# ═══════════════════════════════════════════════════════════════════════════════

using Test

# Include the modules
include(joinpath(dirname(@__DIR__), "src", "para_zigzag.jl"))
include(joinpath(dirname(@__DIR__), "src", "metatheory_brushes.jl"))

@testset "Igor Seeds" begin
    @testset "IgorSeed creation" begin
        igor = IgorSeed()
        @test igor.seed == GAY_IGOR_SEED
        @test igor.motif_count == 64
        @test length(igor.intervals) == 64
        @test length(igor.colors) == 64
        @test all(i -> igor.intervals[i] > 0, 1:64)
    end
    
    @testset "NotIgorSeed derangement" begin
        igor = IgorSeed()
        not_igor = derange_igor(igor)
        
        # Verify no fixed points
        for i in 1:igor.motif_count
            @test not_igor.derangement[i] != i
        end
        
        # Verify it's a permutation
        @test sort(not_igor.derangement) == collect(1:igor.motif_count)
    end
    
    @testset "IgorSpectrum interpolation" begin
        igor = IgorSeed(; n_motifs=10)
        
        # Pure igor
        spectrum_igor = IgorSpectrum(igor, 1.0)
        @test interval_at(spectrum_igor, 1) ≈ igor.intervals[1]
        
        # Pure not-igor  
        spectrum_not = IgorSpectrum(igor, 0.0)
        not_igor = NotIgorSeed(igor)
        @test interval_at(spectrum_not, 1) ≈ not_igor.flipped_intervals[1]
        
        # Midpoint
        spectrum_mid = IgorSpectrum(igor, 0.5)
        expected = 0.5 * igor.intervals[1] + 0.5 * not_igor.flipped_intervals[1]
        @test interval_at(spectrum_mid, 1) ≈ expected
    end
    
    @testset "Motif premining determinism" begin
        spectrum = IgorSpectrum(; seed=UInt64(42), weight=0.5)
        motifs1 = premine_motifs(spectrum, 20)
        motifs2 = premine_motifs(spectrum, 20)
        
        for i in 1:20
            @test motifs1[i].time ≈ motifs2[i].time
            @test motifs1[i].color == motifs2[i].color
            @test motifs1[i].velocity == motifs2[i].velocity
        end
    end
    
    @testset "IgorBeacon SPI" begin
        beacon1 = IgorBeacon(; seed=UInt64(12345), weight=0.5)
        beacon2 = IgorBeacon(; seed=UInt64(12345), weight=0.5)
        
        for _ in 1:50
            next_igor_round(beacon1)
            next_igor_round(beacon2)
        end
        
        @test igor_fingerprint(beacon1) == igor_fingerprint(beacon2)
    end
end

@testset "Metatheory Brushes" begin
    @testset "SheafifiedBrush" begin
        brush = SheafifiedBrush(GAY_IGOR_SEED, 5, 20)
        @test length(brush.open_cover) == 5
        @test length(brush.local_sections) == 5
        
        moment = sheafified_moment(brush)
        @test haskey(moment, :success)
        @test haskey(moment, :color)
    end
    
    @testset "StackifiedBrush equivalence" begin
        brush = StackifiedBrush(GAY_IGOR_SEED, 3, 6)
        
        # Reflexivity
        for c in 1:6
            @test are_equivalent(brush, c, c)
        end
        
        # Orbit structure
        orbit = equivalence_class(brush, 1)
        @test 1 in orbit
        @test length(orbit) <= 6
    end
    
    @testset "CondensifiedBrush projectivity" begin
        brush = CondensifiedBrush(GAY_IGOR_SEED; prime=3, levels=4)
        @test brush.prime == 3
        @test brush.levels == 4
        @test length(brush.approximations) == 4
        
        # Check level sizes
        for (n, level) in enumerate(brush.approximations)
            @test length(level) == 3^n
        end
    end
    
    @testset "MetatheoryTriple combined" begin
        triple = MetatheoryTriple(; seed=GAY_IGOR_SEED)
        paths = [[1, 2, 3], [2, 3, 4], [1, 1, 1, 1]]
        
        result = apply_triple(triple, paths)
        @test haskey(result, :sheafified)
        @test haskey(result, :stackified)
        @test haskey(result, :condensified)
        @test haskey(result, :combined_color)
    end
end

@testset "Reafference" begin
    @testset "Reafferent detection" begin
        # Matching prediction
        r = Reafference(10.0, 10.05, 0.1)
        @test is_reafferent(r)
        @test !is_exafferent(r)
        
        # Non-matching prediction
        r2 = Reafference(10.0, 11.0, 0.1)
        @test !is_reafferent(r2)
        @test is_exafferent(r2)
    end
    
    @testset "ReafferenceState tracking" begin
        state = ReafferenceState{Float64}(; threshold=0.1)
        
        # Add some observations
        observe!(state, 1.0, 1.05)  # Reafferent
        observe!(state, 2.0, 2.02)  # Reafferent
        observe!(state, 3.0, 5.0)   # Exafferent
        
        @test state.reafferent_count == 2
        @test state.exafferent_count == 1
        @test reafference_ratio(state) ≈ 2/3
    end
end

@testset "2-Para Rewriting Gadgets" begin
    @testset "ParaRewriteGadget" begin
        g = ParaRewriteGadget(:add, (p, s) -> s + 1)
        result = apply_gadget(g, 10)
        @test result == (:add, 11)
    end
    
    @testset "TwoParaGadget composition" begin
        gadgets = [
            ParaRewriteGadget(:inc, (p, s) -> s + 1),
            ParaRewriteGadget(:double, (p, s) -> s * 2),
        ]
        tpg = TwoParaGadget(gadgets)
        
        @test length(tpg.base_gadgets) == 2
        @test haskey(tpg.two_morphisms, (1, 2))
        @test haskey(tpg.two_morphisms, (2, 1))
    end
    
    @testset "Edge random access" begin
        gadgets = [
            ParaRewriteGadget(:a, (p, s) -> s),
            ParaRewriteGadget(:b, (p, s) -> s),
            ParaRewriteGadget(:c, (p, s) -> s),
        ]
        tpg = TwoParaGadget(gadgets)
        
        for i in 1:10
            g = edge_random_access(tpg, i)
            @test g.parameter in [:a, :b, :c]
        end
    end
end

@testset "Para(ZigZag) PDMP" begin
    @testset "ZigZagDynamics creation" begin
        D = ZigZagDynamics(10; seed=GAY_IGOR_SEED)
        @test size(D.Γ) == (10, 10)
        @test D.λref == 0.0
        @test length(D.μ) == 10
    end
    
    @testset "ChromaticEvent fingerprint" begin
        ev1 = ChromaticEvent(1.0, 1, 0.5, 1.0, true, GAY_IGOR_SEED)
        ev2 = ChromaticEvent(1.0, 1, 0.5, 1.0, true, GAY_IGOR_SEED)
        
        @test event_fingerprint(ev1) == event_fingerprint(ev2)
        
        # Different event should have different fingerprint
        ev3 = ChromaticEvent(2.0, 2, 0.5, -1.0, false, GAY_IGOR_SEED)
        @test event_fingerprint(ev1) != event_fingerprint(ev3)
    end
    
    @testset "ChromaticZigZag initialization" begin
        D = ZigZagDynamics(5; seed=GAY_IGOR_SEED)
        zz = ChromaticZigZag(D; seed=GAY_IGOR_SEED)
        
        @test length(zz.x) == 5
        @test length(zz.θ) == 5
        @test all(θ -> θ in [-1.0, 1.0], zz.θ)
        @test zz.t == 0.0
    end
    
    @testset "Single step execution" begin
        D = ZigZagDynamics(5; seed=GAY_IGOR_SEED)
        zz = ChromaticZigZag(D; seed=GAY_IGOR_SEED)
        
        event = para_zigzag_step!(zz)
        @test !isnothing(event)
        @test event.t > 0
        @test 1 <= event.i <= 5
        @test length(zz.trace.events) == 1
    end
    
    @testset "Trajectory execution" begin
        D = ZigZagDynamics(5; seed=GAY_IGOR_SEED)
        zz = ChromaticZigZag(D; seed=GAY_IGOR_SEED)
        
        trace = para_zigzag_trajectory(zz, 2.0)
        @test length(trace.events) > 0
        @test all(ev -> ev.t <= 2.0, trace.events)
    end
    
    @testset "SPI Verification" begin
        result = verify_zigzag_spi(GAY_IGOR_SEED, 5, 2.0; n_runs=3)
        
        @test length(result.fingerprints) == 3
        @test result.spi_verified
        @test all(fp -> fp == result.unique_fingerprint, result.fingerprints)
    end
    
    @testset "Tropical path weights" begin
        D = ZigZagDynamics(5; seed=GAY_IGOR_SEED)
        zz = ChromaticZigZag(D; seed=GAY_IGOR_SEED)
        trace = para_zigzag_trajectory(zz, 2.0)
        
        trop = TropicalZigZagPath(trace)
        @test length(trop.times) == length(trace.events)
        @test trop.min_plus_weight <= trop.max_plus_weight
    end
    
    @testset "Trace discretization" begin
        D = ZigZagDynamics(3; seed=GAY_IGOR_SEED)
        zz = ChromaticZigZag(D; seed=GAY_IGOR_SEED)
        trace = para_zigzag_trajectory(zz, 1.0)
        
        discrete = discretize(trace, 0.1)
        @test length(discrete) > 0
        
        # Check times are increasing
        times = [t for (t, _) in discrete]
        @test issorted(times)
    end
end

@testset "Successor Haiku" begin
    haiku = haiku_transition(42, "flows", 69)
    
    # Check structure exists
    @test !isempty(haiku.line1)
    @test !isempty(haiku.line2)
    @test !isempty(haiku.line3)
    
    # Color is valid
    @test 0 <= haiku.color.r <= 1
    @test 0 <= haiku.color.g <= 1
    @test 0 <= haiku.color.b <= 1
end

@testset "Poisson time sampling" begin
    @testset "Constant rate" begin
        t = poisson_time(1.0, 0.0, 0.5)
        @test t ≈ -log(0.5) / 1.0
    end
    
    @testset "Zero rate" begin
        t = poisson_time(0.0, 0.0, 0.5)
        @test isinf(t)
    end
    
    @testset "Negative rate" begin
        t = poisson_time(-1.0, 0.0, 0.5)
        @test isinf(t)
    end
    
    @testset "Increasing rate" begin
        t = poisson_time(1.0, 1.0, 0.5)
        @test t > 0
        @test !isinf(t)
    end
end

println("\n" * "═" ^ 60)
println("All Para(ZigZag) SPI tests passed! ◈")
println("═" ^ 60)
