# Graphs.jl

> Source: https://juliagraphs.org/Graphs.jl/stable/

Performant platform for network and graph analysis in Julia.

## Overview

- `SimpleGraph` - undirected graphs
- `SimpleDiGraph` - directed graphs
- `AbstractGraph` API for custom implementations
- Large collection of graph algorithms

## Installation

```julia
pkg> add Graphs
```

## Basic Usage

```julia
g = path_graph(6)  # {6, 5} undirected simple Int64 graph
nv(g)  # 6 vertices
ne(g)  # 5 edges
add_edge!(g, 1, 6)  # Make path a loop
```

## Related JuliaGraphs Packages

- MetaGraphsNext.jl - property graphs
- SimpleWeightedGraphs.jl - weighted graphs
- GraphIO.jl - file formats

## Gay.jl Extension Target: GayGraphsExt (#41)

### Key Types to Color
- `SimpleGraph`, `SimpleDiGraph`
- Vertices and edges
- Paths, cycles, components

### SPI Opportunities
- Vertex coloring: `hash_color(v, seed)`
- Edge coloring: `hash_color(src ⊻ dst, seed)` - associative!
- Component coloring via connected_components
- Path fingerprinting

### Parallel Tractability
- Parallel BFS/DFS with color accumulation
- Distributed shortest paths
- Parallel graph coloring algorithms
- XOR fingerprint over all edges for isomorphism hints

### Related Extensions
- #42 GayMetaGraphsExt - metadata coloring
- #43 GayMultilayerExt - layer-wise colors
- #44 GayNetworkLayoutExt - position-based coloring
