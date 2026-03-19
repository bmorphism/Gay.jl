using Test
using GayMC

const SEED = UInt64(69)

@testset "GayMC" begin

    @testset "SPI Core" begin
        @test splitmix64(SEED) == splitmix64(SEED)
        @test splitmix64(SEED) != splitmix64(SEED + 1)
        
        r, g, b = hash_color(SEED, UInt64(1))
        @test 0.0f0 <= r <= 1.0f0
        @test 0.0f0 <= g <= 1.0f0
        @test 0.0f0 <= b <= 1.0f0
        
        @test hash_color(SEED, UInt64(1)) == hash_color(SEED, UInt64(1))
    end

    @testset "Core Algorithms" begin
        G = GayGraph(5)
        add_edge!(G, 1, 2)
        add_edge!(G, 2, 3)
        add_edge!(G, 3, 4)
        add_edge!(G, 4, 5)
        add_edge!(G, 1, 3)
        
        @testset "BFS" begin
            levels, colors, fp1 = gay_bfs!(G, 1; seed=SEED)
            @test levels[1] == 0
            @test levels[2] == 1
            @test levels[3] == 1
            _, _, fp2 = gay_bfs!(G, 1; seed=SEED)
            @test fp1 == fp2  # SPI
        end
        
        @testset "DFS" begin
            disc, colors, fp1 = gay_dfs!(G, 1; seed=SEED)
            @test disc[1] == 1
            _, _, fp2 = gay_dfs!(G, 1; seed=SEED)
            @test fp1 == fp2  # SPI
        end
        
        @testset "Dijkstra" begin
            dist, colors, fp1 = gay_dijkstra!(G, 1; seed=SEED)
            @test dist[1] == 0.0
            @test dist[2] == 1.0
            _, _, fp2 = gay_dijkstra!(G, 1; seed=SEED)
            @test fp1 == fp2  # SPI
        end
        
        @testset "MST Prim" begin
            parent, edge_colors, fp1 = gay_mst_prim!(G; seed=SEED)
            @test parent[1] == -1  # root
            _, _, fp2 = gay_mst_prim!(G; seed=SEED)
            @test fp1 == fp2  # SPI
        end
        
        @testset "SCCs" begin
            comp, colors, fp1 = gay_scomponents!(G; seed=SEED)
            @test all(c -> c > 0, comp)
            _, _, fp2 = gay_scomponents!(G; seed=SEED)
            @test fp1 == fp2  # SPI
        end
        
        @testset "K-Cores" begin
            core, colors, fp1 = gay_corenums!(G; seed=SEED)
            @test all(k -> k >= 0, core)
            _, _, fp2 = gay_corenums!(G; seed=SEED)
            @test fp1 == fp2  # SPI
        end
    end

    @testset "Plurigrid Energy Grid" begin
        nodes = [GridNode(i, (-1)^i * 10.0) for i in 1:4]
        edges = [GridEdge(1,2,10.0), GridEdge(2,3,10.0), GridEdge(3,4,10.0)]
        grid = EnergyGrid(nodes, edges)
        
        fp1 = gay_power_flow!(grid, SEED)
        @test fp1 != 0
        
        grid2 = EnergyGrid(copy(nodes), copy(edges))
        fp2 = gay_power_flow!(grid2, SEED)
        @test fp1 == fp2  # SPI
    end

    @testset "TeglonLabs Sheaf" begin
        sheaf = GraphSheaf{Int}([1, 2, 3], SEED)
        sheaf.stalks[1] = [1, 2, 3]
        sheaf.stalks[2] = [2, 3, 4]
        
        cover = [[1, 2], [2, 3]]
        H0, H1, color = gay_cech_cohomology!(sheaf, cover, SEED)
        @test color != 0
    end

    @testset "Tritwies Narratives" begin
        using SparseArrays
        
        n = Narrative(SEED)
        adj = sparse([1, 2], [2, 3], [true, true], 3, 3)
        push!(n.snapshots, Snapshot(1.0, adj, SEED))
        push!(n.snapshots, Snapshot(2.0, adj, SEED + 1))
        
        levels = gay_narrative_bfs!(n, (1, 1), SEED)
        @test !isempty(levels)
    end

    @testset "bmorphism Spined" begin
        cat = SpinedCategory(5)
        add_edge!(cat, 1, 2)
        add_edge!(cat, 2, 3)
        add_edge!(cat, 3, 4)
        add_edge!(cat, 4, 5)
        add_edge!(cat, 1, 3)
        
        width, bags = gay_tree_width!(cat, SEED)
        @test width >= 0
        @test !isempty(bags)
        
        @test verify_spined_spi(cat, SEED)
    end

    @testset "Cross-Fork SPI" begin
        results = verify_all_spi(SEED)
        @test results[:core_bfs]
        @test results[:plurigrid]
        @test results[:bmorphism]
    end

end
