# gaymc Roadmap

> gaimc → gaymc: Chromatic graph algorithms with SPI

## Phase 1: Core Algorithms (v0.1.0)

Port all gaimc algorithms with chromatic identity:

| gaimc | gaymc | Chromatic Extension | Status |
|-------|-------|---------------------|--------|
| `bfs.m` | `gay_bfs!` | Color per level | ✅ |
| `dfs.m` | `gay_dfs!` | Color per discovery time | ✅ |
| `dijkstra.m` | `gay_dijkstra!` | Color per distance class | ✅ |
| `mst_prim.m` | `gay_mst_prim!` | Color per tree edge | ✅ |
| `scomponents.m` | `gay_scomponents!` | Color per SCC | ✅ |
| `bipartite_matching.m` | `gay_bipartite_matching!` | Color per match pair | 🔲 |
| `corenums.m` | `gay_corenums!` | Color per k-core | ✅ |

**Deliverable**: All algorithms return `(result, fingerprint::UInt64)`

## Phase 2: Fork Specialization (v0.2.0)

### 🔋 Plurigrid
- `gay_power_flow!` - DC power flow with iteration colors ✅
- `gay_grid_partition!` - Parallel decomposition ✅
- `gay_optimal_power_flow!` - OPF with chromatic constraints
- `gay_contingency_analysis!` - N-1 security with colored scenarios

### 🔬 TeglonLabs  
- `gay_cech_cohomology!` - H⁰, H¹ with colored classes ✅
- `gay_local_to_global!` - Gluing certification ✅
- `gay_sheaf_decidability!` - Linear-time decision via bounded width
- `gay_grothendieck_topology!` - Adhesive category covers

### ⏱️ Tritwies
- `gay_narrative_bfs!` - Spatiotemporal BFS ✅
- `gay_interval_sheaf!` - Sheaf on time intervals ✅
- `gay_snapshot_compose!` - Narrative composition ✅
- `gay_persistence!` - Persistent homology with chromatic barcodes

### 🌲 bmorphism
- `gay_tree_width!` - Width via minimum-degree ✅
- `gay_triangulate!` - Chordal completion ✅
- `gay_spined_functor!` - Triangulation functor
- `gay_structured_decomposition!` - Abstract decomposition

## Phase 3: Unification (v0.3.0)

Cross-fork composition:

```
Energy Grid (Plurigrid)
    │
    ├─── Temporal Evolution (Tritwies)
    │         │
    │         └─── Sheaf Certification (TeglonLabs)
    │                   │
    └───────────────────┴─── Categorical Structure (bmorphism)
```

- `GayComposedAlgorithm` - Pipeline multiple fork algorithms
- `verify_composed_spi` - End-to-end fingerprint verification
- Full Bumpus paper implementations

## Dependencies

```toml
[deps]
Gay = {url = "https://github.com/bmorphism/Gay.jl"}
Graphs = "86223c79-..."
KernelAbstractions = "63c18a36-..."
```

## Bumpus Papers Implemented

| Paper | Fork | Status |
|-------|------|--------|
| Spined categories (EJC 2023) | bmorphism | 🔲 |
| Compositional Algorithms (2023) | TeglonLabs | 🔲 |
| Time-varying Data (2024) | Tritwies | 🔲 |
| Open Petri Nets (2024) | Plurigrid | 🔲 |

## GitHub Tracking

```bash
# Setup
./scripts/setup_github.sh bmorphism/gaymc

# View progress
gh issue list --repo bmorphism/gaymc --milestone v0.1.0-core
gh project view 1 --owner bmorphism
```
