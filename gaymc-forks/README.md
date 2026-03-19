# gaymc - Chromatic Graph Algorithms with SPI

> **gaimc** (Graph Algorithms in MATLAB Code) by David Gleich, extended with  
> **Gay.jl** chromatic identity and Benjamin Merlin Bumpus's compositional theory.

## Thread Inventory

**26 Amp threads** document the development trajectory:

| Thread | Focus |
|--------|-------|
| T-019b10bd-... | Current: gaymc core + drand timelocks |
| T-019b10a9-... | LearnableOkhsl + Gay Hyperdoctrine |
| T-019b1079-... | Signal DuckDB thread analysis |
| T-87fb8b3f-... | Rio terminal transparency |
| T-faa91376-... | Derangeable permutations for SPI |
| T-b50561e9-... | Paper tracker with color chains |
| T-0ddd54e7-... | Unison package adaptation |
| T-9ebe5db5-... | SPI parallel palette integration |
| T-5fbd63d7-... | eg-walker CRDT with SPI colors |
| T-8157ed9f-... | v0.2.0 desiderata implementation |
| ... | (22 total threads) |

## Self-Confidential Prediction Markets

Trajectory predictions use **drand timelocks** for self-confidential commits:

```julia
using Gay

# Count threads as prediction bound
bound = THREAD_BOUND  # 22 threads

# Create prediction about gaymc trajectory
pred = TrajectoryPrediction(
    GAYMC_THREADS[1:5],  # Threads this prediction covers
    Dict(
        :bfs_ported => true,
        :fork_count => 4,
        :spi_verified => true
    ),
    gay_next(),  # Chromatic signature
    time(),
    round_at_time(beacon, time() + 86400)  # Reveal tomorrow
)

# Commit (encrypted until drand round)
commitment, nonce = commit_trajectory(pred)

# After drand round releases randomness...
revealed, round_data = reveal_trajectory(commitment, nonce)
```

## Origin

This project forks [dgleich/gaimc](https://github.com/dgleich/gaimc) and reimplements it in Julia with:
- **Splittable RNG** for Strong Parallelism Invariance (SPI)
- **Chromatic identity** per vertex/edge/cell
- **Compositional semantics** via sheaves and spined categories
- **drand timelocks** for self-confidential trajectory predictions

## Four Perspectives

### 🔋 [Plurigrid/gaymc](./Plurigrid/README.md)
> Compositional energy grid algorithms with chromatic SPI verification

Focus: Distributed energy systems, grid decomposition, verified parallel execution.

### 🔬 [TeglonLabs/gaymc](./TeglonLabs/README.md)
> Sheaf-theoretic graph decomposition with deterministic coloring

Focus: Čech cohomology for algorithmic obstructions, local-to-global certification.

### ⏱️ [Tritwies/gaymc](./Tritwies/README.md)
> Temporal narrative graph algorithms with splittable chromatic identity

Focus: Time-varying graphs, interval sheaves, snapshot composition.

### 🌲 [bmorphism/gaymc](./bmorphism/README.md)
> Spined category algorithms: tree-width via triangulation functor

Focus: Abstract tree-width, structured decompositions, categorical graph invariants.

## Key Papers (Bumpus et al.)

1. **Spined categories: generalizing tree-width beyond graphs** (EJC 2023)
   - Triangulation functor as abstract tree-width
   - Category-theoretic decomposition

2. **Compositional Algorithms on Compositional Data: Deciding Sheaves on Presheaves** (2023)
   - Grothendieck topologies on adhesive categories
   - Linear-time algorithms via bounded width

3. **Towards a Unified Theory of Time-varying Data** (2024)
   - Categories of narratives
   - Sheaves on posets of time intervals

4. **Additive Invariants of Open Petri Nets** (Compositionality 2024)
   - Classification of compositional invariants
   - Sequential and parallel composition

## Algorithms Ported

| gaimc (MATLAB) | gaymc (Julia) | Chromatic Extension |
|----------------|---------------|---------------------|
| `bfs.m` | `gay_bfs!` | Color per level, SPI verification |
| `dfs.m` | `gay_dfs!` | Color per discovery time |
| `dijkstra.m` | `gay_dijkstra!` | Color per distance class |
| `mst_prim.m` | `gay_mst_prim!` | Color per tree edge |
| `scomponents.m` | `gay_scomponents!` | Color per SCC |
| `bipartite_matching.m` | `gay_bipartite_matching!` | Color per match pair |
| `corenums.m` | `gay_corenums!` | Color per k-core |

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/{org}/gaymc.jl")
```

## Usage

```julia
using GayMC

# Create a graph with chromatic identity
G = gay_graph_from_edges([
    (1, 2), (2, 3), (3, 4), (4, 1), (1, 3)
])

# BFS with SPI-guaranteed colors
result = gay_bfs!(G, 1; seed=0x42)

# Each vertex has deterministic color based on discovery
println(result.vertex_colors)  # Dict{Int, RGB}

# Verify SPI: rerun with different parallelism, same colors
result2 = gay_bfs!(G, 1; seed=0x42, parallel=true)
@assert result.vertex_colors == result2.vertex_colors  # SPI guarantee
```

## Compositional Features

### Structured Decompositions

```julia
# Compute tree decomposition with chromatic bags
td = gay_tree_decomposition(G)

# Each bag has color derived from its vertices
for (bag_id, bag) in td.bags
    println("Bag $bag_id: vertices=$(bag.vertices), color=$(bag.color)")
end

# Width is preserved under composition
@assert tree_width(td) == tree_width(compose(td1, td2))
```

### Sheaf-Valued Algorithms

```julia
# Define a sheaf over the graph
F = vertex_cover_sheaf(G)

# Compute Čech cohomology
H = cech_cohomology(F, td)

# Obstruction classes indicate where local→global fails
println("Obstructions: ", H.obstruction_classes)
```

### Temporal Narratives

```julia
# Create a narrative (time-varying graph)
N = Narrative([
    (1, G1),  # t=1: graph G1
    (2, G2),  # t=2: graph G2
    (3, G3),  # t=3: graph G3
])

# Morphisms between snapshots have chromatic identity
for (t, f) in N.morphisms
    println("t=$t: color=$(f.color)")
end
```

## Connection to Gay.jl

```julia
using Gay

# GayMC contexts integrate with Gay.jl's splittable RNG
ctx = GayMCContext(0x12345)

# Run graph algorithms with MC sampling
samples = []
for _ in 1:1000
    gay_sweep!(ctx)
    push!(samples, gay_measure!(ctx, G))
end

# All samples have deterministic chromatic identity
fingerprint = gay_fingerprint(samples)
```

## License

BSD-2-Clause (following dgleich/gaimc)

## Citation

```bibtex
@software{gaymc2024,
  title={gaymc: Chromatic Graph Algorithms with SPI},
  author={{Plurigrid, TeglonLabs, Tritwies, bmorphism}},
  year={2024},
  url={https://github.com/bmorphism/gaymc},
  note={Based on gaimc (Gleich) and Gay.jl}
}

@article{bumpus2023spined,
  title={Spined categories: generalizing tree-width beyond graphs},
  author={Bumpus, Benjamin Merlin and Kocsis, Zoltan A.},
  journal={European Journal of Combinatorics},
  year={2023}
}

@article{bumpus2023compositional,
  title={Compositional Algorithms on Compositional Data: Deciding Sheaves on Presheaves},
  author={Althaus, Evan and Bumpus, Benjamin Merlin and Fairbanks, James and Rosiak, Cory},
  journal={arXiv:2302.05575},
  year={2023}
}
```
