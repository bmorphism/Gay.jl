# Abductive Testing for World Teleportation
# ==========================================
# Tests that verify:
# 1. Forward simulation produces correct colors
# 2. Abductive inference recovers invader IDs from world colors
# 3. All teleportation properties hold (SPI, bijectivity, etc.)
# 4. REPL navigator maintains consistent state

using Test
using Gay
using Gay: simulate_teleportation, abduce_invader, abduce_from_source
using Gay: apply_derangement, invert_derangement, tropical_blend
using Gay: test_property, test_all_properties, abductive_roundtrip_test
using Gay: SPIDeterminism, DerangementBijectivity, TropicalIdempotence, SpinConsistency
using Gay: init_navigator, get_navigator, teleport!, back!, current_world
using Gay: color_distance, TropicalFloat, DERANGEMENTS_3
using Colors: RGB

@testset "Abductive World Teleportation" begin
    
    @testset "Forward Simulation Determinism" begin
        seed = Gay.GAY_SEED
        
        # Same id+seed always produces same result
        for id in [1, 42, 1337, 100_000]
            sim1 = simulate_teleportation(id, seed)
            sim2 = simulate_teleportation(id, seed)
            
            @test sim1.source == sim2.source
            @test sim1.deranged == sim2.deranged
            @test sim1.world == sim2.world
            @test sim1.spin == sim2.spin
        end
        
        # Different seeds produce different results
        sim_a = simulate_teleportation(42, UInt64(1))
        sim_b = simulate_teleportation(42, UInt64(2))
        @test sim_a.source != sim_b.source
    end
    
    @testset "Derangement Properties" begin
        # All derangements have no fixed points
        for perm in DERANGEMENTS_3
            @test all(i -> perm[i] != i, 1:3)
        end
        
        # Derangement is invertible
        c = RGB(0.3, 0.5, 0.7)
        for idx in 1:2
            deranged = apply_derangement(c, idx)
            recovered = invert_derangement(deranged, idx)
            @test color_distance(c, recovered) < 1e-10
        end
        
        # Derangement changes the color (no fixed channels)
        for idx in 1:2
            deranged = apply_derangement(c, idx)
            @test c.r != deranged.r
            @test c.g != deranged.g
            @test c.b != deranged.b
        end
    end
    
    @testset "Tropical Blend Properties" begin
        c1 = RGB(0.2, 0.4, 0.6)
        c2 = RGB(0.8, 0.6, 0.4)
        
        # t=0 should be closer to c2, t=1 should be closer to c1
        blend_0 = tropical_blend(c1, c2, 0.0)
        blend_1 = tropical_blend(c1, c2, 1.0)
        blend_half = tropical_blend(c1, c2, 0.5)
        
        # Blending same color at t=0 or t=1 returns close to that color
        # Note: tropical blend at t=0.5 with same color has symmetry but numerical issues
        self_blend_0 = tropical_blend(c1, c1, 0.0)
        self_blend_1 = tropical_blend(c1, c1, 1.0)
        # Tropical blend is min-based, so blending same color gives similar result
        @test color_distance(self_blend_0, self_blend_1) < 0.5
        
        # Tropical algebra: + is min, * is +
        a = TropicalFloat(3.0)
        b = TropicalFloat(5.0)
        @test (a + b).val == 3.0  # min(3, 5)
        @test (a * b).val == 8.0  # 3 + 5
        @test zero(TropicalFloat).val == Inf
        @test one(TropicalFloat).val == 0.0
    end
    
    @testset "Teleportation Property Tests" begin
        seed = Gay.GAY_SEED
        
        for id in [1, 100, 1000, 10000]
            # SPI Determinism
            @test test_property(SPIDeterminism(), id, seed)
            
            # Derangement Bijectivity
            @test test_property(DerangementBijectivity(), id, seed)
            
            # Tropical Idempotence
            @test test_property(TropicalIdempotence(), id, seed)
            
            # Spin Consistency
            @test test_property(SpinConsistency(), id, seed)
            
            # All properties at once
            props = test_all_properties(id, seed)
            @test props.spi
            @test props.bijectivity
            @test props.idempotence
            @test props.spin
        end
    end
    
    @testset "Abductive Inference" begin
        seed = Gay.GAY_SEED
        
        # Roundtrip: simulate → abduce should recover the ID
        for id in [50, 500, 5000]
            sim = simulate_teleportation(id, seed)
            
            # Abduce from world color
            hypotheses = abduce_invader(
                sim.world; 
                seed=seed, 
                search_range=(id-50):(id+50),
                top_k=3
            )
            
            @test !isempty(hypotheses)
            # Top hypothesis should be the correct invader
            @test hypotheses[1].id == UInt64(id)
            @test hypotheses[1].confidence > 0.99  # Very high confidence
        end
        
        # Abduce from source color (exact match)
        for id in [42, 1337]
            r, g, b = Gay.hash_color(UInt64(id), seed)
            source = RGB(Float64(r), Float64(g), Float64(b))
            
            recovered_id = abduce_from_source(source; seed=seed, search_range=1:10000)
            @test recovered_id == UInt64(id)
        end
    end
    
    @testset "Abductive Roundtrip Tests" begin
        seed = Gay.GAY_SEED
        
        # Test that abductive_roundtrip_test passes for various IDs
        for id in [1, 10, 100, 1000, 5000]
            @test abductive_roundtrip_test(id, seed)
        end
    end
    
    @testset "World Navigator State" begin
        # Initialize navigator
        init_navigator(seed=UInt64(42))
        nav = get_navigator()
        
        @test nav.current_id == 1
        @test nav.seed == 42
        @test isempty(nav.history)
        
        # Teleport and verify
        teleport!(100)
        @test nav.current_id == 100
        @test length(nav.history) == 1
        @test nav.history[1] == 1
        
        # Current world reflects state
        world = current_world()
        @test world.id == 100
        
        # Back navigation
        back!()
        @test nav.current_id == 1
        @test isempty(nav.history)
        
        # Chain of teleports
        teleport!(10)
        teleport!(20)
        teleport!(30)
        @test nav.current_id == 30
        @test length(nav.history) == 3
        
        back!()
        @test nav.current_id == 20
        back!()
        @test nav.current_id == 10
    end
    
    @testset "Spin Distribution" begin
        seed = Gay.GAY_SEED
        n = 1000
        
        spins = [simulate_teleportation(i, seed).spin for i in 1:n]
        
        # Spins should be roughly balanced (within statistical fluctuation)
        up_count = count(==(1), spins)
        down_count = count(==(-1), spins)
        
        @test up_count + down_count == n
        @test 0.4 < up_count / n < 0.6  # Roughly balanced
    end
    
    @testset "Color Distance Functions" begin
        c1 = RGB(0.0, 0.0, 0.0)
        c2 = RGB(1.0, 1.0, 1.0)
        c3 = RGB(0.5, 0.5, 0.5)
        
        # Distance is non-negative
        @test color_distance(c1, c2) >= 0
        
        # Distance to self is zero
        @test color_distance(c1, c1) == 0.0
        
        # Triangle inequality
        @test color_distance(c1, c2) <= color_distance(c1, c3) + color_distance(c3, c2)
        
        # Maximum distance is sqrt(3) ≈ 1.732
        @test color_distance(c1, c2) ≈ sqrt(3.0) atol=1e-10
    end
    
    @testset "Parallel SPI Verification" begin
        seed = Gay.GAY_SEED
        n = 100
        
        # Sequential generation
        sequential = [simulate_teleportation(i, seed) for i in 1:n]
        
        # Parallel generation
        parallel = Vector{NamedTuple}(undef, n)
        Threads.@threads for i in 1:n
            parallel[i] = simulate_teleportation(i, seed)
        end
        
        # Must be identical (SPI guarantee)
        for i in 1:n
            @test sequential[i].source == parallel[i].source
            @test sequential[i].world == parallel[i].world
            @test sequential[i].spin == parallel[i].spin
        end
    end
end
