#!/usr/bin/env julia
"""
Cross-Fork Composition Demo

Shows how all four gaymc forks compose under a unified SPI framework:
1. Plurigrid: Energy grid with power flow
2. TeglonLabs: Sheaf over the grid topology
3. Tritwies: Temporal narrative of grid evolution
4. bmorphism: Tree-width decomposition for efficient computation

All share the same seed → deterministic chromatic identity across forks.
"""

using Pkg
Pkg.activate(@__DIR__)

include("../src/GayMC.jl")
using .GayMC
using SparseArrays

const SEED = UInt64(69)

println("═══════════════════════════════════════════════════════════════════")
println("  gaymc Cross-Fork Composition Demo")
println("  Seed: 0x$(string(SEED, base=16))")
println("═══════════════════════════════════════════════════════════════════")
println()

# ═══════════════════════════════════════════════════════════════════════════
# 1. Plurigrid: Energy Grid
# ═══════════════════════════════════════════════════════════════════════════

println("┌─ Plurigrid: Energy Grid ──────────────────────────────────────────┐")

nodes = [
    GridNode(1, 100.0),   # Generator
    GridNode(2, -30.0),   # Load
    GridNode(3, -40.0),   # Load
    GridNode(4, -30.0),   # Load
]

edges = [
    GridEdge(1, 2, 10.0),  # Line 1-2
    GridEdge(2, 3, 10.0),  # Line 2-3
    GridEdge(3, 4, 10.0),  # Line 3-4
    GridEdge(1, 4, 5.0),   # Line 1-4
]

grid = EnergyGrid(nodes, edges)
fp_grid = gay_power_flow!(grid, SEED)

println("│  Nodes: $(length(nodes))")
println("│  Edges: $(length(edges))")
println("│  Power flow iterations: $(grid.iteration)")
println("│  Fingerprint: 0x$(string(fp_grid, base=16)[1:12])...")
println("└────────────────────────────────────────────────────────────────────┘")
println()

# ═══════════════════════════════════════════════════════════════════════════
# 2. TeglonLabs: Sheaf over Grid Topology
# ═══════════════════════════════════════════════════════════════════════════

println("┌─ TeglonLabs: Sheaf Cohomology ────────────────────────────────────┐")

sheaf = GraphSheaf{Float64}([1, 2, 3, 4], SEED)
for (i, node) in enumerate(nodes)
    sheaf.stalks[i] = [node.injection, grid.angles[i]]
end

cover = [[1, 2], [2, 3], [3, 4], [4, 1]]
H0, H1, fp_sheaf = gay_cech_cohomology!(sheaf, cover, SEED)

println("│  Cover patches: $(length(cover))")
println("│  H⁰ (global sections): $(length(H0))")
println("│  H¹ obstruction: 0x$(string(H1, base=16)[1:12])...")
println("│  Fingerprint: 0x$(string(fp_sheaf, base=16)[1:12])...")
println("└────────────────────────────────────────────────────────────────────┘")
println()

# ═══════════════════════════════════════════════════════════════════════════
# 3. Tritwies: Temporal Evolution
# ═══════════════════════════════════════════════════════════════════════════

println("┌─ Tritwies: Temporal Narrative ────────────────────────────────────┐")

narrative = Narrative(SEED)

for t in 1:3
    adj = sparse([1, 2, 3, 1], [2, 3, 4, 4], [true, true, true, true], 4, 4)
    snap = Snapshot(Float64(t), adj, splitmix64(SEED + UInt64(t)))
    push!(narrative.snapshots, snap)
end

for t in 1:2
    src, tgt = narrative.snapshots[t], narrative.snapshots[t+1]
    morphism = SnapshotMorphism(src, tgt, collect(1:4), SEED)
    push!(narrative.morphisms, morphism)
end

levels = gay_narrative_bfs!(narrative, (1, 1), SEED)
fp_narrative = reduce(⊻, [s.fingerprint for s in narrative.snapshots])

println("│  Snapshots: $(length(narrative.snapshots))")
println("│  Morphisms: $(length(narrative.morphisms))")
println("│  BFS levels discovered: $(length(levels))")
println("│  Fingerprint: 0x$(string(fp_narrative, base=16)[1:12])...")
println("└────────────────────────────────────────────────────────────────────┘")
println()

# ═══════════════════════════════════════════════════════════════════════════
# 4. bmorphism: Tree-Width Decomposition
# ═══════════════════════════════════════════════════════════════════════════

println("┌─ bmorphism: Spined Categories ────────────────────────────────────┐")

cat = SpinedCategory(4)
add_edge!(cat, 1, 2)
add_edge!(cat, 2, 3)
add_edge!(cat, 3, 4)
add_edge!(cat, 1, 4)

width, bags = gay_tree_width!(cat, SEED)
fp_spined = reduce(⊻, [hash(b) for b in bags])

println("│  Vertices: $(length(cat.objects))")
println("│  Tree-width: $(width)")
println("│  Bags: $(length(bags))")
println("│  Fingerprint: 0x$(string(fp_spined, base=16)[1:12])...")
println("└────────────────────────────────────────────────────────────────────┘")
println()

# ═══════════════════════════════════════════════════════════════════════════
# Unified SPI Verification
# ═══════════════════════════════════════════════════════════════════════════

println("┌─ Unified SPI Verification ────────────────────────────────────────┐")

combined_fp = splitmix64(fp_grid ⊻ fp_sheaf ⊻ fp_narrative ⊻ UInt64(fp_spined))

results = verify_all_spi(SEED)
all_pass = all(values(results))

println("│  Core BFS:   $(results[:core_bfs] ? "✓" : "✗")")
println("│  Plurigrid:  $(results[:plurigrid] ? "✓" : "✗")")
println("│  bmorphism:  $(results[:bmorphism] ? "✓" : "✗")")
println("│")
println("│  Combined Fingerprint: 0x$(string(combined_fp, base=16))")
println("│  All SPI verified: $(all_pass ? "✓" : "✗")")
println("└────────────────────────────────────────────────────────────────────┘")
println()

# Show chromatic identity
println("Chromatic samples from seed 69:")
for i in 1:6
    r, g, b = hash_color(SEED, UInt64(i))
    ri, gi, bi = round(Int, r*255), round(Int, g*255), round(Int, b*255)
    print("  \e[48;2;$(ri);$(gi);$(bi)m  \e[0m")
end
println(" gay")
