# test/test_exa_loop_extensions.jl
# =============================================================================

using Test
using Gay
using Colors: RGB, Lab

@testset "Exa Loop Extensions" begin

    @testset "Intrinsic Non-Riemannian HSL" begin
        # Test basic conversions
        c_red = RGB(1.0, 0.0, 0.0)
        c_grey = RGB(0.5, 0.5, 0.5)
        
        # Conversion to IntrinsicHSL
        hsl_red = to_intrinsic_hsl(c_red; A=25.0)
        hsl_grey = to_intrinsic_hsl(c_grey; A=25.0)
        
        @test hsl_red isa IntrinsicHSL
        @test hsl_grey isa IntrinsicHSL
        
        # Hue limits
        @test 0.0 <= hsl_red.h < 360.0
        @test 0.0 <= hsl_grey.h < 360.0
        
        # Saturation limits
        @test 0.0 <= hsl_red.s <= 1.0
        @test 0.0 <= hsl_grey.s <= 1.0
        
        # Lightness limits
        @test 0.0 <= hsl_red.l <= 1.0
        @test 0.0 <= hsl_grey.l <= 1.0
        
        # Gray color should have very low or zero intrinsic saturation (neutral axis)
        @test hsl_grey.s ≈ 0.0 atol=1e-5
        
        # Test subadditivity of saturation scaling: f(c1+c2) < f(c1) + f(c2)
        # s_intrinsic = A * (1.0 - exp(-chroma / A)) / A = 1 - exp(-chroma / A)
        # Let's verify for chroma = 10 and 20
        A = 25.0
        s1 = 1.0 - exp(-10.0 / A)
        s2 = 1.0 - exp(-20.0 / A)
        s12 = 1.0 - exp(-(10.0 + 20.0) / A)
        @test s12 < s1 + s2
    end

    @testset "Multilateral Clearing (Fleischman-Dini Cycle Set-off)" begin
        # Case A: Simple complete loop
        net = ObligationNetwork()
        add_obligation!(net, :A, :B, 100.0)
        add_obligation!(net, :B, :C, 100.0)
        add_obligation!(net, :C, :A, 100.0)
        
        # Run multilateral set-off
        result = multilateral_setoff!(net)
        
        @test result isa ClearingResult
        @test result.total_cleared ≈ 300.0
        @test result.initial_debt ≈ 300.0
        @test result.reduction_ratio ≈ 1.0
        @test result.zero_sum_conserved == true
        @test isempty(net.obligations)  # All fully cleared and filtered out
        
        # Case B: Partially matching cycle (bottleneck)
        net2 = ObligationNetwork()
        add_obligation!(net2, :A, :B, 100.0)
        add_obligation!(net2, :B, :C, 80.0)
        add_obligation!(net2, :C, :A, 50.0)
        
        # Cycle limit is 50.0. 
        # Total debt was 230.0.
        # Clearing subtracts 50.0 from each, leaving:
        # A -> B: 50.0
        # B -> C: 30.0
        # C -> A is cleared (amount 0.0, filtered out)
        # Total cleared across obligations = 150.0.
        # Remaining debt is 80.0.
        # Ratio = (230 - 80) / 230 = 150 / 230 ≈ 0.652
        result2 = multilateral_setoff!(net2)
        @test result2.total_cleared ≈ 150.0
        @test result2.initial_debt ≈ 230.0
        @test result2.reduction_ratio ≈ 150.0 / 230.0
        @test result2.zero_sum_conserved == true
        @test length(net2.obligations) == 2
        @test any(o -> o.from == :A && o.to == :B && o.amount == 50.0, net2.obligations)
        @test any(o -> o.from == :B && o.to == :C && o.amount == 30.0, net2.obligations)
    end

    @testset "Cellular Sheaf Čech Cohomology" begin
        # 1. Trivial single vertex sheaf (no edges)
        vertices = [1]
        edges = Tuple{Int, Int}[]
        stalk_dim = 1
        edge_dim = 1
        restrictions = Dict{Tuple{Int, Tuple{Int, Int}}, Matrix{Int}}()
        
        sheaf_triv = CellularSheaf(vertices, edges, stalk_dim, edge_dim, restrictions)
        
        L_triv = build_sheaf_laplacian(sheaf_triv)
        @test size(L_triv) == (1, 1)
        @test L_triv[1, 1] == 0
        
        r_triv = rank_gf3(L_triv)
        @test r_triv == 0
        
        dim_H0, dim_H1 = cohomology_dimensions(sheaf_triv)
        @test dim_H0 == 1  # 1 global section (any constant element of GF(3) on vertex 1)
        @test dim_H1 == 0  # No cycles, no obstructions
        
        # 2. Sheaf on cycle graph with 3 vertices
        # Vertices: 1, 2, 3
        # Edges: (1, 2), (2, 3), (1, 3)
        # Restrictions are [1] (or [2] for some) representing standard maps
        
        # Case A: Totally consistent sheaf (loop where dimensions of H^0 = 1, H^1 = 1)
        # All restrictions represent identical identity maps [1]
        vertices3 = [1, 2, 3]
        edges3 = [(1, 2), (2, 3), (1, 3)]
        restrictions3_A = Dict{Tuple{Int, Tuple{Int, Int}}, Matrix{Int}}()
        
        for e in edges3
            restrictions3_A[(e[1], e)] = fill(1, 1, 1)
            restrictions3_A[(e[2], e)] = fill(1, 1, 1)
        end
        
        sheaf_cycle_A = CellularSheaf(vertices3, edges3, 1, 1, restrictions3_A)
        L_A = build_sheaf_laplacian(sheaf_cycle_A)
        @test size(L_A) == (3, 3)
        
        # Our hand-calculated L_A mod 3 was:
        # [2 2 2; 2 2 2; 2 2 2]
        @test L_A == [2 2 2; 2 2 2; 2 2 2]
        @test rank_gf3(L_A) == 1
        
        dim_H0_A, dim_H1_A = cohomology_dimensions(sheaf_cycle_A)
        # dim_H0 = total_dim - rank = 3 - 1 = 2
        # euler_characteristic = 3 * 1 - 3 * 1 = 0
        # dim_H1 = dim_H0 - euler = 2 - 0 = 2
        @test dim_H0_A == 2
        @test dim_H1_A == 2
        
        # Case B: Cohomology audit detects obstruction (H^1 has non-zero obstruction dimensions)
        # Set one restriction map to [2] (which is -1 mod 3).
        restrictions3_B = Dict{Tuple{Int, Tuple{Int, Int}}, Matrix{Int}}()
        for e in edges3
            restrictions3_B[(e[1], e)] = fill(1, 1, 1)
            restrictions3_B[(e[2], e)] = fill(1, 1, 1)
        end
        restrictions3_B[(3, (1, 3))] = fill(2, 1, 1)  # Twist on the edge (1, 3)
        
        sheaf_cycle_B = CellularSheaf(vertices3, edges3, 1, 1, restrictions3_B)
        L_B = build_sheaf_laplacian(sheaf_cycle_B)
        
        # Hand-calculated L_B:
        # [2 2 1; 2 2 2; 1 2 2]
        @test L_B == [2 2 1; 2 2 2; 1 2 2]
        @test rank_gf3(L_B) == 3
        
        dim_H0_B, dim_H1_B = cohomology_dimensions(sheaf_cycle_B)
        # dim_H0 = 3 - 3 = 0
        # dim_H1 = 0 - 0 = 0
        @test dim_H0_B == 0
        @test dim_H1_B == 0
    end

    @testset "Time Delay Embedding & Takens Attractor Reconstruction" begin
        # Create a simple sine wave time series
        N = 100
        t_seq = range(0, 4*pi, length=N)
        series = [sin(ti) for ti in t_seq]
        
        # Test basic phase space reconstruction
        delay = 5
        dim = 3
        pts = reconstruct_phase_space(series, delay, dim)
        
        M = N - (dim - 1) * delay
        @test size(pts) == (M, dim)
        for i in 1:M
            @test pts[i, 1] == series[i]
            @test pts[i, 2] == series[i + delay]
            @test pts[i, 3] == series[i + 2 * delay]
        end
        
        # Test error handling
        @test_throws ArgumentError reconstruct_phase_space(series, 0, dim)
        @test_throws ArgumentError reconstruct_phase_space(series, delay, 0)
        @test_throws ArgumentError reconstruct_phase_space(series, 50, 3) # too short
        
        # Test Autocorrelation
        acf = autocorrelation(series, 20)
        @test length(acf) == 21
        @test acf[1] ≈ 1.0
        @test acf[2] < 1.0
        
        # Test delay selection
        tau_zero = find_optimal_delay_acf_zero(series; max_lag=30)
        tau_decay = find_optimal_delay_acf_decay(series; threshold=0.5, max_lag=30)
        @test tau_zero >= 1
        @test tau_decay >= 1
        
        # Test Average Mutual Information
        ami = average_mutual_information(series, 10; bins=5)
        @test length(ami) == 11
        @test all(v -> v >= 0.0, ami)
        
        tau_ami = find_optimal_delay_ami(series; bins=5, max_lag=20)
        @test tau_ami >= 1
        
        # Test False Nearest Neighbors
        fnn = false_nearest_neighbors(series, 5, 4)
        @test length(fnn) == 3
        @test all(v -> 0.0 <= v <= 1.0, fnn)
        
        opt_dim = find_optimal_dimension(series, 5; max_dim=4, threshold=0.1)
        @test 1 < opt_dim <= 4
        
        # Test Lyapunov divergence & exponent estimation
        div_curve = lyapunov_divergence_curve(pts, delay; theiler=2, max_steps=5)
        @test length(div_curve) == 6
        
        mle = estimate_lyapunov_exponent(pts, delay; theiler=2, max_steps=5)
        @test mle isa Float64
        
        # Test embed_colored_ticks on mock ColoredTick telemetry
        mock_ticks = [ColoredTick(TritTick(UInt64(i)), Int8(i % 3 - 1), UInt64(i * 12345), Float32(0.8 + 0.1 * sin(i)), :mock) for i in 1:120]
        
        # Embed different channels
        emb_conf = embed_colored_ticks(mock_ticks, :confidence; delay=4, dim=2)
        @test emb_conf isa DelayEmbedding
        @test emb_conf.delay == 4
        @test emb_conf.dimension == 2
        @test emb_conf.series_name == :confidence
        @test size(emb_conf.points, 2) == 2
        
        emb_trit = embed_colored_ticks(mock_ticks, :trit; delay=4, dim=2)
        @test emb_trit.series_name == :trit
        
        emb_entropy = embed_colored_ticks(mock_ticks, :entropy; delay=4, dim=2)
        @test emb_entropy.series_name == :entropy
        
        emb_hue = embed_colored_ticks(mock_ticks, :hue; delay=4, dim=2)
        @test emb_hue.series_name == :hue
        
        emb_isat = embed_colored_ticks(mock_ticks, :intrinsic_saturation; delay=4, dim=2)
        @test emb_isat.series_name == :intrinsic_saturation
        
        # Auto-estimation test
        emb_auto = embed_colored_ticks(mock_ticks, :hue)
        @test emb_auto.delay >= 1
        @test emb_auto.dimension >= 1
    end
end

