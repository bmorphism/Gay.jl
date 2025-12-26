# CombinatorialSpaces.jl

> Source: https://algebraicjulia.github.io/CombinatorialSpaces.jl/stable/

This package provides combinatorial models of geometric spaces, such as simplicial sets and combinatorial maps. These "combinatorial spaces" are useful in computational physics, computer graphics, and other applications where geometry plays a large role. They are also potentially useful in non-geometric applications, since structures like simplicial sets generalize graphs from binary relations to higher-arity relations.

Combinatorial spaces, like graphs, are typically C-sets (copresheaves) on some category C. They are implemented here using the general data structures for C-sets offered by Catlab.jl. Thus, this package complements and extends the family of graph data structures shipped with Catlab in the module `Catlab.Graphs`.

## Current Features

- delta sets (semi-simplicial sets) in dimensions one and two, optionally oriented and/or embedded in Euclidean space
- construction of the dual complex associated with a delta set, via combinatorial and geometric subdivision
- core operators of the discrete exterior calculus (DEC), including:
  - boundary operator
  - exterior derivative
  - Hodge star
  - codifferential
  - Laplace-Beltrami operators
- experimental support for rotation systems and combinatorial maps

## Installation

```julia
(@v1.5) pkg> add CombinatorialSpaces
```

## Gay.jl Extension Target: GayCombinatorialSpacesExt

### Key Types to Color
- `DeltaSet1D` - 1D semi-simplicial sets (vertices + edges)
- `DeltaSet2D` - 2D triangulated surfaces
- `EmbeddedDeltaSet2D` - Embedded with dual complex

### SPI Opportunities
- Vertex XOR fingerprinting: `hash_color(v, seed)`
- Edge XOR: `hash_color(src ⊻ tgt, seed)` - associative!
- Triangle XOR: `reduce(⊻, sorted_vertices)`
- DEC operator coloring by matrix entry

### Parallel Tractability
- `spi_verify_parallel` verifies XOR fingerprint matches across threads
- `associative_color_reduce` enables parallel tree reduction
