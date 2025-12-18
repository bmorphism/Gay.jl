# Regression tests for ternary/gamut systems
# Tests the GamutSubobjectClassifier and related categorical structures

using Test
using Colors: RGB

@testset "Gamut Subobject Classifier" begin
    
    @testset "GamutTruth struct" begin
        # Basic construction
        t1 = GamutTruth(true, -0.5)
        @test t1.in_gamut == true
        @test t1.distance == -0.5
        
        t2 = GamutTruth(false, 0.3)
        @test t2.in_gamut == false
        @test t2.distance == 0.3
        
        # Convenience constructor
        t3 = GamutTruth(true)
        @test t3.in_gamut == true
        @test t3.distance < 0.0  # Inside = negative distance
        
        t4 = GamutTruth(false)
        @test t4.in_gamut == false
        @test t4.distance > 0.0  # Outside = positive distance
        
        # Lattice operations: AND
        t_and = t1 & t2
        @test t_and.in_gamut == false  # true && false = false
        @test t_and.distance == max(t1.distance, t2.distance)
        
        # Lattice operations: OR
        t_or = t1 | t2
        @test t_or.in_gamut == true  # true || false = true
        @test t_or.distance == min(t1.distance, t2.distance)
        
        # Negation
        t_not = !t1
        @test t_not.in_gamut == false
        @test t_not.distance == -t1.distance
        
        # Ordering
        @test GamutTruth(true, -1.0) < GamutTruth(true, -0.5)
        @test GamutTruth(true, -0.1) < GamutTruth(false, 0.1)
    end
    
    @testset "characteristic_morphism for sRGB ⊂ P3 ⊂ Rec2020" begin
        # Create classifiers for each gamut
        srgb_classifier = GamutSubobjectClassifier(GaySRGBGamut())
        p3_classifier = GamutSubobjectClassifier(GayP3Gamut())
        rec2020_classifier = GamutSubobjectClassifier(GayRec2020Gamut())
        
        # Color well within sRGB
        c_srgb = RGB(0.5, 0.5, 0.5)
        @test characteristic_morphism(srgb_classifier, c_srgb).in_gamut == true
        @test characteristic_morphism(p3_classifier, c_srgb).in_gamut == true
        @test characteristic_morphism(rec2020_classifier, c_srgb).in_gamut == true
        
        # Color at sRGB edge (should be in sRGB)
        c_edge = RGB(1.0, 0.0, 0.0)
        @test characteristic_morphism(srgb_classifier, c_edge).in_gamut == true
        @test characteristic_morphism(p3_classifier, c_edge).in_gamut == true
        @test characteristic_morphism(rec2020_classifier, c_edge).in_gamut == true
        
        # Color outside sRGB but in P3 (simulated)
        c_p3_only = RGB(1.05, 0.0, 0.0)  # Slightly over 1.0
        @test characteristic_morphism(srgb_classifier, c_p3_only).in_gamut == false
        @test characteristic_morphism(p3_classifier, c_p3_only).in_gamut == true
        @test characteristic_morphism(rec2020_classifier, c_p3_only).in_gamut == true
        
        # Color outside P3 but in Rec2020
        c_rec2020_only = RGB(1.15, 0.0, 0.0)
        @test characteristic_morphism(srgb_classifier, c_rec2020_only).in_gamut == false
        @test characteristic_morphism(p3_classifier, c_rec2020_only).in_gamut == false
        @test characteristic_morphism(rec2020_classifier, c_rec2020_only).in_gamut == true
        
        # Distance properties: closer to boundary = smaller |distance|
        c_center = RGB(0.5, 0.5, 0.5)
        c_near_edge = RGB(0.95, 0.5, 0.5)
        truth_center = characteristic_morphism(srgb_classifier, c_center)
        truth_edge = characteristic_morphism(srgb_classifier, c_near_edge)
        @test abs(truth_center.distance) > abs(truth_edge.distance)
        
        # Subobject inclusion: sRGB ⊂ P3 ⊂ Rec2020
        @test GaySRGBGamut() ⊆ GayP3Gamut()
        @test GayP3Gamut() ⊆ GayRec2020Gamut()
        @test GaySRGBGamut() ⊆ GayRec2020Gamut()
        @test !(GayP3Gamut() ⊆ GaySRGBGamut())
        @test !(GayRec2020Gamut() ⊆ GayP3Gamut())
    end
    
    @testset "gamut_pullback recovers colors in gamut" begin
        srgb_classifier = GamutSubobjectClassifier(GaySRGBGamut())
        p3_classifier = GamutSubobjectClassifier(GayP3Gamut())
        
        # Out-of-gamut color
        c_out = RGB(1.5, -0.3, 0.8)
        
        # Pullback to sRGB should clamp
        c_pulled_srgb = gamut_pullback(srgb_classifier, c_out)
        @test characteristic_morphism(srgb_classifier, c_pulled_srgb).in_gamut == true
        @test Float64(c_pulled_srgb.r) == 1.0
        @test Float64(c_pulled_srgb.g) == 0.0
        @test Float64(c_pulled_srgb.b) == 0.8
        
        # Pullback to P3 allows wider range
        c_pulled_p3 = gamut_pullback(p3_classifier, c_out)
        @test characteristic_morphism(p3_classifier, c_pulled_p3).in_gamut == true
        
        # Pullback of in-gamut color should be identity
        c_in = RGB(0.5, 0.5, 0.5)
        c_pulled_in = gamut_pullback(srgb_classifier, c_in)
        @test c_pulled_in ≈ c_in
        
        # Pullback is idempotent
        c_double = gamut_pullback(srgb_classifier, gamut_pullback(srgb_classifier, c_out))
        @test c_double ≈ c_pulled_srgb
    end
    
    @testset "probe_gamut_subobject" begin
        gamuts = [GaySRGBGamut(), GayP3Gamut(), GayRec2020Gamut()]
        
        # Color in sRGB should return sRGB
        c_srgb = RGB(0.5, 0.5, 0.5)
        @test probe_gamut_subobject(gamuts, c_srgb) isa GaySRGBGamut
        
        # Color outside sRGB but in P3 should return P3
        c_p3 = RGB(1.05, 0.5, 0.5)
        @test probe_gamut_subobject(gamuts, c_p3) isa GayP3Gamut
        
        # Color outside P3 but in Rec2020 should return Rec2020
        c_rec2020 = RGB(1.15, 0.5, 0.5)
        @test probe_gamut_subobject(gamuts, c_rec2020) isa GayRec2020Gamut
        
        # Color outside all gamuts returns largest
        c_outside = RGB(1.5, 0.5, 0.5)
        @test probe_gamut_subobject(gamuts, c_outside) isa GayRec2020Gamut
    end
    
    @testset "world_gamut_classifier lattice structure" begin
        wgc = world_gamut_classifier()
        
        # Access classifiers by symbol
        @test wgc[:srgb] isa GamutSubobjectClassifier{GaySRGBGamut}
        @test wgc[:p3] isa GamutSubobjectClassifier{GayP3Gamut}
        @test wgc[:rec2020] isa GamutSubobjectClassifier{GayRec2020Gamut}
        
        # Order should be smallest to largest
        @test wgc.order == [:srgb, :p3, :rec2020]
        
        # Meet operation (intersection = smallest)
        @test gamut_meet(wgc, :srgb, :p3) == :srgb
        @test gamut_meet(wgc, :p3, :rec2020) == :p3
        @test gamut_meet(wgc, :srgb, :rec2020) == :srgb
        
        # Join operation (union = largest)
        @test gamut_join(wgc, :srgb, :p3) == :p3
        @test gamut_join(wgc, :p3, :rec2020) == :rec2020
        @test gamut_join(wgc, :srgb, :rec2020) == :rec2020
        
        # Lattice laws
        # Idempotent
        @test gamut_meet(wgc, :srgb, :srgb) == :srgb
        @test gamut_join(wgc, :p3, :p3) == :p3
        
        # Commutative
        @test gamut_meet(wgc, :srgb, :p3) == gamut_meet(wgc, :p3, :srgb)
        @test gamut_join(wgc, :srgb, :p3) == gamut_join(wgc, :p3, :srgb)
        
        # Absorption
        @test gamut_meet(wgc, :srgb, gamut_join(wgc, :srgb, :p3)) == :srgb
        @test gamut_join(wgc, :srgb, gamut_meet(wgc, :srgb, :p3)) == :srgb
    end
    
    @testset "GayChain gamut operations" begin
        # Create a chain of colors
        colors = [RGB(0.2, 0.4, 0.6), RGB(0.8, 0.3, 0.1), RGB(0.5, 0.5, 0.5)]
        chain = GayChain(colors)
        
        @test chain.gamut isa GaySRGBGamut
        @test length(chain.colors) == 3
        
        # Verify chain is in gamut
        @test verify_chain_in_gamut(chain) == true
        
        # Chain with out-of-gamut color
        bad_colors = [RGB(0.5, 0.5, 0.5), RGB(1.5, 0.0, 0.0)]
        bad_chain = GayChain(bad_colors)
        @test verify_chain_in_gamut(bad_chain) == false
        
        # Process chain to fix out-of-gamut colors
        fixed_chain = process_gay_chain(bad_chain)
        @test verify_chain_in_gamut(fixed_chain) == true
        
        # Map chain to different gamut
        p3_chain = chain_to_gamut(chain, GayP3Gamut())
        @test p3_chain.gamut isa GayP3Gamut
        @test length(p3_chain.colors) == length(chain.colors)
    end
    
    @testset "is_in_gamut and gamut_distance" begin
        # is_in_gamut tests
        @test is_in_gamut(RGB(0.5, 0.5, 0.5), GaySRGBGamut()) == true
        @test is_in_gamut(RGB(1.5, 0.5, 0.5), GaySRGBGamut()) == false
        @test is_in_gamut(RGB(1.05, 0.5, 0.5), GayP3Gamut()) == true
        
        # gamut_distance tests
        dist_inside = gamut_distance(RGB(0.5, 0.5, 0.5), GaySRGBGamut())
        @test dist_inside < 0.0  # Negative = inside
        
        dist_outside = gamut_distance(RGB(1.5, 0.5, 0.5), GaySRGBGamut())
        @test dist_outside > 0.0  # Positive = outside
        
        # Closer to edge = smaller |distance|
        dist_center = gamut_distance(RGB(0.5, 0.5, 0.5), GaySRGBGamut())
        dist_edge = gamut_distance(RGB(0.99, 0.5, 0.5), GaySRGBGamut())
        @test abs(dist_center) > abs(dist_edge)
    end
    
    @testset "LearnableGamutMap" begin
        # Create a learnable map from P3 to sRGB
        mapper = LearnableGamutMap(GayP3Gamut(), GaySRGBGamut())
        
        @test mapper.source isa GayP3Gamut
        @test mapper.target isa GaySRGBGamut
        @test mapper.params isa GamutParameters
        
        # Map a color
        c_in = RGB(1.05, 0.5, 0.5)  # In P3 but not sRGB
        c_out = map_to_gamut(mapper, c_in)
        @test is_in_gamut(c_out, GaySRGBGamut()) == true
        
        # Gamut loss should be low for in-gamut colors
        in_gamut_colors = [RGB(0.5, 0.5, 0.5), RGB(0.3, 0.7, 0.2)]
        loss = gamut_loss(mapper, in_gamut_colors)
        @test loss >= 0.0  # Loss is non-negative
    end
end
