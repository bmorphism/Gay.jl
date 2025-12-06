# Changelog

All notable changes to Gay.jl will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0]

### Added

#### Core Protocol (`src/protocol.jl`)
- `SPIColorable` abstract type for SPI-compliant colorable objects
- `spi_color(x, seed)` generic function for SPI coloring
- `@verify_spi expr seed [n_trials]` macro for SPI verification
- `Colorable` trait and `colorize(x, seed)` generic function
- `HasColorSeed` trait for objects carrying their own seed
- `ColorView{T}` wrapper for colored indexed views of collections
- `@colorable T` macro to auto-implement colorize for custom types
- `ColoredView{T,N}` for N-dimensional array coloring
- `color_fingerprint(colors)` for XOR fingerprinting
- `spi_verify(f, args...)` for function SPI verification
- `parallel_spi_verify(f, items, seed)` for parallel verification

#### Extensions
- **GayCombinatorialSpacesExt** (`ext/GayCombinatorialSpacesExt.jl`)
  - `color_deltaset_1d(ds)` - 1D simplicial set coloring
  - `color_deltaset_2d(ds)` - 2D simplicial set coloring (vertices, edges, triangles)
  - `color_embedded_dual(ds)` - Dual complex coloring
  - `color_multigrid_levels(levels)` - Multigrid hierarchy coloring
  - `color_poisson_2d(ds, solution)` - Poisson equation solution coloring
  - `color_euler_flow(ds, velocity)` - Euler flow field coloring
  - `color_dec_operator(op)` - DEC operator matrix entry coloring
  - `spi_verify_parallel(ds, seed)` - Parallel SPI verification
  - `associative_color_reduce(elements, op)` - Associative reduction verification

- **GayRimuExt** (`ext/GayRimuExt.jl`)
  - `color_fock_state(fs)` - BoseFS, FermionFS, CompositeFS coloring
  - `color_hamiltonian_element(H, i, j)` - Matrix element coloring
  - `color_dvec(dv)` - Population vector coloring
  - `ColoredWalker` - Colored FCIQMC walker type
  - `colored_fciqmc_trajectory(dvs)` - Trajectory coloring
  - `render_fock_state(fs)` - ANSI-colored rendering
  - `color_walker_population(dv)` - Population statistics

- **GayPlasmoExt** (`ext/GayPlasmoExt.jl`)
  - `color_optigraph(graph)` - OptiGraph node/edge coloring
  - `color_optinodes(graph)` - Node-only coloring
  - `color_optiedges(graph)` - Edge-only coloring
  - `color_partition(partition)` - Partition block coloring
  - `color_subgraph(sub, parent)` - Consistent subgraph coloring
  - `solution_color_map(graph)` - Objective-weighted coloring
  - `color_linking_constraints(graph)` - Constraint coloring
  - `render_optigraph(graph)` - ANSI-colored rendering

- **GayDecapodesExt** (`ext/GayDecapodesExt.jl`)
  - `color_mesh(mesh)` - EmbeddedDeltaSet2D/DeltaSet1D coloring
  - `color_field(mesh, field; form)` - Discrete form coloring (0/1/2-forms)
  - `color_field(mesh, magnitude, phase)` - Complex field coloring
  - `color_operator(op)` - DEC operator coloring (d, δ, Δ, ⋆, ♭, ♯)
  - `color_decapode(d)` - Decapode structure coloring
  - `color_simulation_state(mesh, state)` - Multi-field coloring
  - `render_colored_mesh(mesh, colors)` - ANSI rendering
  - `render_decapode(d, colors)` - ANSI rendering

#### Documentation
- `docs/src/literate/combinatorial_spaces.jl` - CombinatorialSpaces examples
- `docs/src/literate/rimu_qmc.jl` - Rimu QMC examples
- `docs/src/literate/plasmo_optimization.jl` - Plasmo optimization examples
- `docs/src/literate/decapodes_physics.jl` - Decapodes physics examples
- Updated `docs/src/index.md` with v0.2.0 features
- Updated `docs/make.jl` with new pages

### Changed
- Bumped version to 0.2.0 in `Project.toml`
- Minimum Julia version raised to 1.9.0 (for extensions support)
- Added `protocol.jl` include to `src/Gay.jl`
- Expanded exports in `src/Gay.jl`

### Dependencies
- Added weak dependencies: CombinatorialSpaces, Decapodes, Plasmo, Rimu
- Added extension mappings in `Project.toml`

## [0.1.0]

### Added

#### Core
- `splitmix64(x)` - O(1) hash function
- `hash_color(index, seed)` - O(1) deterministic color generation (~2ns)
- `hash_color_rgb(index, seed)` - RGB wrapper
- `xor_fingerprint(colors)` - SPI verification fingerprinting
- `GAY_SEED` constant

#### KernelAbstractions GPU Support
- `ka_colors!(matrix, seed)` - SPMD parallel color generation
- `ka_color_sums(n, seed)` - Billion-scale reduction
- `ka_parallel_hash(n, seed)` - CPU parallel hash
- `set_backend!(backend)` / `get_backend()` - Backend management

#### Splittable RNG
- `GayRNG` - Splittable RNG type
- `gay_seed!(seed)` - Global RNG seeding
- `gay_split(rng)` - RNG splitting
- `next_color()`, `next_colors(n)`, `next_palette(n)`
- `color_at(index)`, `colors_at(indices)`, `palette_at(index, n)`

#### Interleaved Streams
- `GayInterleaver` - Interleaved SPI streams for checkerboard decomposition
- `gay_interleave(il)` - Round-robin color generation
- `gay_sublattice(il, parity)` - Parity-based sublattice coloring
- `gay_xor_color(il, i, j)` - XOR-based bond coloring
- `gay_checkerboard_2d(il, Lx, Ly)` - 2D lattice coloring
- `gay_heisenberg_bonds(il, Lx, Ly)` - Heisenberg model bonds

#### S-Expression Coloring
- `GaySexpr` - Magnetized S-expression type
- `gay_magnetized_sexpr(expr)` - Parse with colors/spins
- `gay_render_sexpr(gs)` - ANSI rendering
- `gay_paren_color(seed, depth, pos)` - Parenthesis coloring
- `gay_sexpr_magnetization(gs)` - Spin statistics

#### Color Spaces
- `SRGB()`, `DisplayP3()`, `Rec2020()` - Wide-gamut support
- `random_color(cs)`, `random_colors(n, cs)`, `random_palette(n, cs)`
- Pride flags: `rainbow()`, `transgender()`, `bisexual()`, etc.

#### Extensions
- **GayMetalExt** - Metal.jl GPU acceleration
- **GayAutoloadsExt** - BasicAutoloads + Chairmarks integration

#### Other Modules
- `src/radare2.jl` - Binary analysis coloring
- `src/abduce.jl` - Abductive inference
- `src/derangeable.jl` - Derangement permutations
- `src/comrade.jl` - Sky model DSL
- `src/enzyme.jl` - Autodiff integration
- `src/semiosis.jl` - SSE QMC integration
- `src/repl.jl` - Custom REPL mode

---

## SPI Guarantee

All versions maintain the **Strong Parallelism Invariance** guarantee:

```
color(seed, index) = color(seed, index)  ∀ execution context
```

This holds for:
- Sequential vs parallel execution
- CPU vs GPU computation
- Any thread count
- Any execution order
- Past and future runs
