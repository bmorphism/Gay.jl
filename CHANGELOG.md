# Changelog

All notable changes to Gay.jl will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-01-01

### Added

#### Cryptographic Forward Secrecy (`src/sparse_pq_ratchet.jl`)
Post-quantum secure key ratcheting with Play/CoPlay duality:
- `RatchetState`, `RatchetOutput`, `RatchetVerification` types
- `forward_ratchet!(state)` - PLAY phase with SHA3-256 key derivation
- `backward_verify(output, expected_fp)` - COPLAY phase fingerprint verification
- `epoch_ratchet!(state)` - punctured forward secrecy via subtree erasure
- `sparse_ratchet_to!(state, target)` - O(log n) random access
- `world_ratchet_state(name; seed)` - composable state builder
- `world_ratchet_from_handoff(world_fp, coworld_fp)` - bridge integration
- `teleportation_ratchet_test(s1, s2)` - SPI verification for identical roots
- `ratchet_diagram(state)`, `ratchet_chain_diagram(outputs)` - Mermaid visualization
- Domain separation: `DOMAIN_ROOT`, `DOMAIN_CHAIN`, `DOMAIN_MESSAGE`, `DOMAIN_RATCHET`, `DOMAIN_TRIT`

#### Open Games with Play/CoPlay Bidirectionality (`src/gay_open_game.jl`)
Compositional game theory via Para/Optic structure:
- `GayPlay`, `GayCoPlay`, `PlayCoPlayPair` - bidirectional game morphisms
- `GayGame` - complete open game with players and equilibrium state
- `play(pair, state)`, `coplay(pair, outcome)` - forward/backward passes
- `compute_marginal(game, player_id)` - deviation gain at Nash equilibrium
- `is_equilibrium(game)`, `marginals_vanishing(game)` - equilibrium verification
- `parallel(G, H)`, `sequential(G, H)` - compositional game operators (⊗, ;)
- `cfr_marginals!(game; iterations)` - counterfactual regret minimization
- `compute_wev(source, target)` - World Extractable Value (WEV = PoA - 1)
- `teleportation_test(G, H)` - parallel vs sequential equivalence
- `blanket_equilibrium(game)` - Markov blanket boundary equilibrium
- `dual_marginal(game, player_id)` - self-dual game deviation
- `game_diagram(game)`, `teleportation_diagram(test)` - Mermaid visualization

#### Three-Match Gadget (`src/three_match.jl`)
GF(3) compositional verification for 3-SAT reduction:
- 3-coloring verification via gadget composition
- NP-completeness proof structure

#### Parallel Color Fork System (Phase 2A)
Enhanced `splittable.jl` with parallel color forking:
- Deterministic fork/join semantics preserving SPI
- Thread-safe color stream management

### Changed

#### Ontology Enforcement
- **BREAKING**: All `demo_*` functions renamed to `world_*` (AGENTS.md compliance)
- All `world_*` functions now return types implementing `length`, `merge`, `fingerprint`
- Emoji replaced with Unicode geometry (◆◇△▣◈) for terminal compatibility

#### Code Quality
- Export conflicts resolved across all modules
- Aqua.jl test suite expanded
- `verify_parallel.jl` test harness added

### Documentation
- `SPARSE_PQ_RATCHET.md` - complete cryptographic ratchet documentation
- `AGENTS.md` - development guidelines with `demo_` → `world_` enforcement
- `WORLD_PATTERN.md` - world builder pattern specification

### Move Contract Bridges
Companion contracts for Aptos blockchain integration:
- `gay_colors.move` - SplitMix64 isomorphic to Julia implementation
- `cognitive_yield.move` - Open Game for AI service credit arbitrage with GF(3) conservation

### Formal Verification (`verification/`)
Multi-level verification for critical invariants:

**Dafny proofs** (automated theorem proving):
- `SplitMixTernary.dfy` - Determinism, path invariance, GF(3) conservation
- `spi_galois.dfy` - Galois closure (α∘γ = id), XOR monoid laws, p-adic colors
- `GayMcpCriticalProofs.dfy` - SPI guarantee, reafference closure, indexless property

**Narya proofs** (higher observational type theory):
- `gf3.ny` - GF(3) field structure with `GF3Conserved` type
- `bridge_sheaf.ny` - Sheaf conditions respecting way-below relation
- `spi_conservation.ny` - SPI as dependent types with 7 property proofs
- `worldhop_narya_bridge.ny` - World hopping via observational bridge types

See `verification/VERIFICATION_ROADMAP.md` for full property matrix.

### Internal
- 14 core modules graduated from `.topos/experimental` to `src/`
- `.topos/` staging area for refinement-in-progress work
- Precompilation disabled for `gamut_learnable` module (stability)

---

## [0.4.1] - Planned (Ratchet & Game Proofs)

### Verification Additions
- `SparsePQRatchet.dfy` - Forward secrecy, epoch puncture, domain separation
- `OpenGame.dfy` - Nash equilibrium, marginal vanishing, WEV bounds
- `ratchet_forward_secrecy.ny` - Forward secrecy as bridge type
- `play_coplay_duality.ny` - Bidirectional game morphisms

### Planned Features
- Strobe.rs integration for full Keccak duplex
- Prediction market cryptographic bindings
- Attention routing with forward-secret state

---

## [0.4.2] - Planned (Sheaf & Cohomology Proofs)

### Verification Additions
- `StructuredDecomposition.dfy` - Tree decomposition, FPT correctness
- `CechCohomology.dfy` - H^1 = 0 verification for consistency
- `sheaf_neural_network.ny` - Sheaf Laplacian coordination
- `pentagon_hexagon.ny` - Monoidal coherence conditions

### Planned Features
- Full StructuredDecompositions.jl bridge
- Sheaf neural network integration
- Pentagon/hexagon identity verification

---

## [0.3.0] - 2025-12-15

### Added

#### Sheaf-ACSet Integration (`src/structured_decompositions.jl`)
Bridge to Bumpus's StructuredDecompositions.jl:
- `ChromaticDecomposition` - structured decompositions with color-tracked bags
- `chromatic_adhesion_filter` - Bumpus algorithm with SPI color verification
- `decide_chromatic_sheaf` - main decision procedure for sheaf conditions
- `ThreadAncestryForest` - copresheaf over continuation category
- `RewritingGadget` - local rewriting with color signature preservation
- Balanced ternary depth-4 decomposition (seed 1069): 18 atomic ops in 6 clusters

#### CRDT Cohomological Consistency (`#213`)
- H^1 = 0 verification for eventually consistent data structures
- Conflict resolution via sheaf cohomology

#### Coherence Violation Detection (`#214`)
- Pentagon/hexagon identity verification
- Monoidal category coherence checking

#### Symmetric Monoidal Formalization (`#215`)
- `gay_split` formalized as symmetric monoidal category
- Braiding and unit coherence

### Changed
- Ternary polarity system stabilized (regression tests #208)
- Spectral test threshold: 10x → 12x mean (#188)

### Security
- Seed treated as secret; fingerprints never printed

---

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
