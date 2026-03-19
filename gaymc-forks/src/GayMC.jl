"""
    GayMC - Chromatic Graph Algorithms with SPI

gaimc (Graph Algorithms in MATLAB Code) reimplemented in Julia with:
- Splittable RNG for Strong Parallelism Invariance (SPI)  
- Chromatic identity per vertex/edge/cell
- Compositional semantics via sheaves and spined categories

Four fork perspectives:
- Plurigrid: Energy grid algorithms
- TeglonLabs: Sheaf-theoretic decomposition
- Tritwies: Temporal narrative graphs
- bmorphism: Spined categories / tree-width
"""
module GayMC

using Colors: RGB
using SparseArrays

# ═══════════════════════════════════════════════════════════════════════════
# SPI Core - Shared across all forks
# ═══════════════════════════════════════════════════════════════════════════

const GAY_SEED = UInt64(0x285508656870f24a)

"Splitmix64 PRNG - deterministic, O(1) skip-ahead for SPI"
@inline function splitmix64(x::UInt64)::UInt64
    x += 0x9e3779b97f4a7c15
    x = (x ⊻ (x >> 30)) * 0xbf58476d1ce4e5b9
    x = (x ⊻ (x >> 27)) * 0x94d049bb133111eb
    x ⊻ (x >> 31)
end

"Hash to RGB color via splitmix64"
@inline function hash_color(seed::UInt64, index::UInt64)::Tuple{Float32,Float32,Float32}
    h = splitmix64(seed ⊻ (index * 0x9e3779b97f4a7c15))
    r = Float32((h & 0xFF)) / 255.0f0
    g = Float32(((h >> 8) & 0xFF)) / 255.0f0
    b = Float32(((h >> 16) & 0xFF)) / 255.0f0
    (r, g, b)
end

"Chromatic fingerprint via XOR (order-independent)"
@inline chromatic_fingerprint(seed::UInt64, h::UInt64) = splitmix64(seed ⊻ h)

export splitmix64, hash_color, chromatic_fingerprint, GAY_SEED

# ═══════════════════════════════════════════════════════════════════════════
# Core Algorithms (gaimc ports)
# ═══════════════════════════════════════════════════════════════════════════

include("core_algorithms.jl")
using .GayCoreAlgorithms
export GayGraph, gay_bfs!, gay_dfs!, gay_dijkstra!, gay_mst_prim!, gay_scomponents!, gay_corenums!

# ═══════════════════════════════════════════════════════════════════════════
# Fork Submodules
# ═══════════════════════════════════════════════════════════════════════════

# Plurigrid: Energy grid algorithms
include("../Plurigrid/src/energy_grid.jl")
using .GayEnergyGrid
export EnergyGrid, GridNode, GridEdge
export gay_power_flow!, gay_grid_partition!, verify_grid_spi

# TeglonLabs: Sheaf-theoretic decomposition
include("../TeglonLabs/src/sheaf_decomposition.jl")
using .GaySheafDecomposition
export GraphSheaf, TreeDecompositionSheaf
export gay_cech_cohomology!, gay_local_to_global!

# Tritwies: Temporal narrative graphs
include("../Tritwies/src/temporal_narratives.jl")
using .GayTemporalNarratives
export Snapshot, Narrative, SnapshotMorphism
export gay_narrative_bfs!, gay_interval_sheaf!, gay_snapshot_compose!

# bmorphism: Spined categories / tree-width
include("../bmorphism/src/spined_categories.jl")
using .GaySpinedCategories
export SpinedCategory, TriangulationFunctor, SimplicialComplex
export gay_tree_width!, gay_triangulate!, verify_spined_spi

# ═══════════════════════════════════════════════════════════════════════════
# Cross-Fork Composition
# ═══════════════════════════════════════════════════════════════════════════

"""
    verify_all_spi(seed; n_trials=10) -> Dict

Run SPI verification across all forks. Returns pass/fail for each.
"""
function verify_all_spi(seed::UInt64=GAY_SEED; n_trials::Int=10)
    results = Dict{Symbol,Bool}()
    
    # Core algorithms
    G = GayGraph(5)
    add_edge!(G, 1, 2); add_edge!(G, 2, 3); add_edge!(G, 3, 4); add_edge!(G, 4, 5); add_edge!(G, 1, 3)
    
    l1, _, fp1 = gay_bfs!(G, 1; seed=seed)
    l2, _, fp2 = gay_bfs!(G, 1; seed=seed)
    results[:core_bfs] = (fp1 == fp2)
    
    # Plurigrid
    nodes = [GridNode(i, (-1)^i * 10.0) for i in 1:4]
    edges = [GridEdge(1,2,10.0), GridEdge(2,3,10.0), GridEdge(3,4,10.0), GridEdge(1,4,5.0)]
    grid = EnergyGrid(nodes, edges)
    results[:plurigrid] = verify_grid_spi(grid, seed; n_trials=n_trials)
    
    # bmorphism
    cat = SpinedCategory(5)
    add_edge!(cat, 1, 2); add_edge!(cat, 2, 3); add_edge!(cat, 3, 4); add_edge!(cat, 4, 5); add_edge!(cat, 1, 3)
    results[:bmorphism] = verify_spined_spi(cat, seed)
    
    # TeglonLabs: Čech cohomology consistency
    sheaf = GraphSheaf{Int}([1,2,3,4], seed)
    cover = [[1,2], [2,3], [3,4]]
    H0_1, H1_1, c1 = gay_cech_cohomology!(sheaf, cover, seed)
    H0_2, H1_2, c2 = gay_cech_cohomology!(sheaf, cover, seed)
    results[:teglonlabs] = (c1 == c2) && (H1_1 == H1_2)
    
    # Tritwies: Narrative composition consistency
    using SparseArrays
    adj = sparse([1,2], [2,1], [true,true], 3, 3)
    s1 = Snapshot(0.0, adj, seed)
    s2 = Snapshot(1.0, adj, splitmix64(seed))
    n1 = Narrative([s1], SnapshotMorphism[], seed)
    n2 = Narrative([s2], SnapshotMorphism[], splitmix64(seed))
    composed1 = gay_snapshot_compose!(n1, n2, seed)
    composed2 = gay_snapshot_compose!(n1, n2, seed)
    results[:tritwies] = (composed1.seed == composed2.seed)
    
    results
end

export verify_all_spi

end # module
