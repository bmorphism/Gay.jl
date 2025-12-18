# Regression Tests for v0.3.0 Ternary Polarity System
# Issue #208: Validate bulk-closed issues (#107-205)
#
# Ternary Polarity:
#   (-) Destructive/Detection - coherence violations, obstructions
#   (_) Neutral/Foundational  - categorical structure, definitions
#   (+) Constructive/Consistency - CRDTs, convergence, building

using Test
using Gay

@testset "Gay.jl v0.3.0 Ternary Regression" begin

    # ═══════════════════════════════════════════════════════════════════════════
    # (-) DESTRUCTIVE/DETECTION - Issue #214
    # ═══════════════════════════════════════════════════════════════════════════
    @testset "(-) Coherence Violation Detection" begin
        # #214: detect_coherence_obstruction
        @test hasmethod(detect_coherence_obstruction, Tuple{Integer})
        
        obstructions = detect_coherence_obstruction(1069)
        @test obstructions isa Vector
        @test isempty(obstructions)  # 1069 should be coherent
        
        # Injection test
        @test hasmethod(inject_coherence_violation, Tuple{Integer, Symbol})
        injected = inject_coherence_violation(1069, :spi)
        @test injected.discrepancy == UInt64(1)
        
        # probe_obstruction
        probe = probe_obstruction(1069)
        @test probe.is_coherent == true
        @test probe.obstructions_found == 0
        
        # world_coherence_detection
        world = world_coherence_detection()
        @test world.world == :coherence_detection
        @test world.issue == 214
    end

    # ═══════════════════════════════════════════════════════════════════════════
    # (_) NEUTRAL/FOUNDATIONAL - Issue #215
    # ═══════════════════════════════════════════════════════════════════════════
    @testset "(_) Symmetric Monoidal Category" begin
        # #215: SeedObject, SplitMorphism
        @test isdefined(Gay, :SeedObject)
        @test isdefined(Gay, :SplitMorphism)
        
        s1 = SeedObject(1069)
        s2 = SeedObject(42)
        @test s1 isa SeedObject
        
        # Tensor product - use Gay.CategoricalFoundations.⊗
        using Gay.CategoricalFoundations: ⊗
        t = s1 ⊗ s2
        @test t isa SeedObject
        
        # SplitMorphism
        split = SplitMorphism(s1)
        @test split.source == s1
        @test split.left isa SeedObject
        @test split.right isa SeedObject
        
        # Coherence isomorphisms
        @test isdefined(Gay, :Associator)
        @test isdefined(Gay, :Braiding)
        @test isdefined(Gay, :LeftUnitor)
        @test isdefined(Gay, :RightUnitor)
        
        # Pentagon/Hexagon verification
        @test hasmethod(verify_pentagon, Tuple{SeedObject, SeedObject, SeedObject, SeedObject})
        @test hasmethod(verify_hexagon, Tuple{SeedObject, SeedObject, SeedObject})
        
        s3 = SeedObject(137)
        s4 = SeedObject(256)
        @test verify_pentagon(s1, s2, s3, s4) == true
        @test verify_hexagon(s1, s2, s3) == true
        
        # probe_coherence
        probe = probe_coherence(1069)
        @test probe.coherence.all_pass == true
        @test probe.split_morphism.spi_check == true
        
        # world_categorical_foundations
        world = world_categorical_foundations()
        @test world.world == :categorical_foundations
        @test world.issue == 215
    end

    # ═══════════════════════════════════════════════════════════════════════════
    # (+) CONSTRUCTIVE/CONSISTENCY - Issue #213
    # ═══════════════════════════════════════════════════════════════════════════
    @testset "(+) CRDT Cohomological Consistency" begin
        # #213: verify_crdt_cohomology
        @test hasmethod(verify_crdt_cohomology, Tuple{Vector{Symbol}, Integer})
        
        nodes = [:alice, :bob, :carol]
        cohom = verify_crdt_cohomology(nodes, 1069)
        @test cohom.commutative == true
        @test cohom.associative == true
        @test cohom.cocycle == true
        @test cohom.h1_trivial == true
        @test cohom.strong_eventual_consistency == true
        
        # probe_crdt_consistency
        probe = probe_crdt_consistency(1069)
        @test probe.merge_convergent == true
        @test probe.cohomology.h1_trivial == true
        
        # world_crdt_cohomology
        world = world_crdt_cohomology()
        @test world.world == :crdt_cohomology
        @test world.issue == 213
        @test world.h1_classification == "H^1 = 0 (trivial via XOR involution)"
    end

    # ═══════════════════════════════════════════════════════════════════════════
    # Core gayrng.jl Functions (Issue #205)
    # ═══════════════════════════════════════════════════════════════════════════
    @testset "Core gayrng.jl" begin
        # GaySeed
        @test isdefined(Gay.GaySplittableRNG, :GaySeed)
        
        # FingerprintCRDT
        @test isdefined(Gay, :FingerprintCRDT)
        @test hasmethod(crdt_merge, Tuple{FingerprintCRDT, FingerprintCRDT})
        @test hasmethod(crdt_update!, Tuple{FingerprintCRDT, UInt64})
        @test hasmethod(crdt_query, Tuple{FingerprintCRDT})
        
        # ZobristTable
        @test isdefined(Gay, :ZobristTable)
        @test hasmethod(zobrist_init, Tuple{})
        
        # World functions
        @test hasmethod(world_gayrng, Tuple{})
        @test hasmethod(world_monoidal_coherence, Tuple{})
        @test hasmethod(world_statistical_quality, Tuple{})
    end

    # ═══════════════════════════════════════════════════════════════════════════
    # Seed Sonification (Issue #190)
    # ═══════════════════════════════════════════════════════════════════════════
    @testset "Seed Sonification #190" begin
        @test hasmethod(world_seed_sonification, Tuple{})
        
        world = world_seed_sonification()
        @test world !== nothing  # Returns Vector{SeedSonificationData}
        @test !isempty(world)
    end

    # ═══════════════════════════════════════════════════════════════════════════
    # Seed Mining (Issue #191)
    # ═══════════════════════════════════════════════════════════════════════════
    @testset "Seed Mining #191" begin
        @test isdefined(Gay, :SeedQuality)
        @test hasmethod(world_seed_mining, Tuple{})
        
        world = world_seed_mining()
        @test world !== nothing  # Returns Vector{SeedQuality}
        @test !isempty(world)
    end

    # ═══════════════════════════════════════════════════════════════════════════
    # Descent Tower (Issue #192)
    # ═══════════════════════════════════════════════════════════════════════════
    @testset "Descent Tower #192" begin
        @test isdefined(Gay, :DescentLevel)
        @test hasmethod(world_descent_tower, Tuple{})
        
        world = world_descent_tower()
        @test world !== nothing  # Returns Vector{DescentLevel}
        @test length(world) == 8  # 8 descent levels (0-7)
    end

    # ═══════════════════════════════════════════════════════════════════════════
    # Universal Color (Issue #183)
    # ═══════════════════════════════════════════════════════════════════════════
    @testset "Universal Color #183" begin
        @test isdefined(Gay, :AbstractGayColorant)
        @test isdefined(Gay, :AbstractGayColor)
    end

    # ═══════════════════════════════════════════════════════════════════════════
    # Gamut Learnable (Issue #184)
    # ═══════════════════════════════════════════════════════════════════════════
    @testset "Gamut Learnable #184" begin
        @test isdefined(Gay, :GamutParameters)
        @test hasmethod(map_to_gamut, Tuple{Any})
    end

    # ═══════════════════════════════════════════════════════════════════════════
    # SPI Invariant Verification
    # ═══════════════════════════════════════════════════════════════════════════
    @testset "SPI Invariant" begin
        # Verify SPI for seed 1069
        s = SeedObject(1069)
        split = SplitMorphism(s)
        
        using Gay.GaySplittableRNG: fingerprint
        
        parent_fp = fingerprint(s.seed)
        left_fp = fingerprint(split.left.seed)
        right_fp = fingerprint(split.right.seed)
        
        @test parent_fp == left_fp ⊻ right_fp
        
        # Test multiple seeds
        for seed in [42, 137, 256, 1337, 9999]
            s = SeedObject(seed)
            split = SplitMorphism(s)
            parent_fp = fingerprint(s.seed)
            child_xor = fingerprint(split.left.seed) ⊻ fingerprint(split.right.seed)
            @test parent_fp == child_xor
        end
    end

    # ═══════════════════════════════════════════════════════════════════════════
    # Seed 1069 Signature
    # ═══════════════════════════════════════════════════════════════════════════
    @testset "Seed 1069 Canonical" begin
        # gay_at(1069) should return consistent color
        c = gay_at(1069)
        @test c isa Gay.Colors.RGB
        
        # Verify reproducibility
        c2 = gay_at(1069)
        @test c == c2
    end

end

println("\n✓ All ternary regression tests passed")
