# Branch Etiquette for gaymc Forks

> "I need to finish a very complex derivation but you go ahead in parallel with abundance mindset about convergence worlds given by gay SPI guarantees on multiversal Hamkins as a singular originary random gay seed"

## Three Branch Modes

### `slave` - Sequential Linearized
```
┌─────────────────────────────────────────────────────────────┐
│  SLAVE: Barely Making It                                     │
│                                                              │
│  - max-parallel: 1                                           │
│  - JULIA_NUM_THREADS=1                                       │
│  - Type-lossy verification (minimal checking)                │
│  - Step-by-step convergence                                  │
│  - Every operation verified before next                      │
│                                                              │
│  Use when: Debugging, bisecting regressions, proving safety  │
└─────────────────────────────────────────────────────────────┘
```

### `master` - Maximally Parallel
```
┌─────────────────────────────────────────────────────────────┐
│  MASTER: Abundance Mindset                                   │
│                                                              │
│  - max-parallel: 10                                          │
│  - JULIA_NUM_THREADS=auto                                    │
│  - Type-safe, fully decidable                                │
│  - Batch verification at end                                 │
│  - SPI guarantees convergence                                │
│                                                              │
│  Use when: Production, performance testing, releasing        │
└─────────────────────────────────────────────────────────────┘
```

### `gay` - Infinite Parallelism
```
┌─────────────────────────────────────────────────────────────┐
│  GAY: Maximum SPI to Fullest Conclusion                      │
│                                                              │
│  - ∞ threads (all universes simultaneously)                  │
│  - max-parallel: 100 (GitHub limit)                          │
│  - All algorithms in all OS/version combos at once           │
│  - SPI guarantees convergence across multiverse              │
│                                                              │
│  Use when: Full confidence in SPI, maximum verification      │
└─────────────────────────────────────────────────────────────┘
```

## Hamkins Multiverse Interpretation

Joel David Hamkins' set-theoretic multiverse posits that there exist multiple 
equally valid set-theoretic universes. Our SPI (Strong Parallelism Invariance)
provides an analogous guarantee for chromatic identity:

```
Universe α (Thread 1)  ─┐
Universe β (Thread 2)  ─┼──▶ Same Fingerprint (SPI Guarantee)
Universe γ (Thread 3)  ─┤
Universe δ (Thread N)  ─┘
```

**Key Property**: Regardless of which universe executes the algorithm, the
chromatic fingerprint converges to the same value. This is the "abundance
mindset" - we can let parallel worlds proceed independently, confident they
will agree.

## The Originary Seed

```julia
const ORIGINARY_SEED = 0x6761795f6f726967  # "gay_orig" as bytes
```

This singular genesis seed is the origin of all chromatic identity across
all forks, branches, and universes. It is the fixed point from which the
multiverse emanates.

## Fork-Specific Branches

Each fork maintains the same three branches with specialized tests:

### Plurigrid/gaymc
```
gay    → Observer mode for energy grid convergence
master → Parallel grid decomposition tests
slave  → Sequential verification of distributed invariants
```

### TeglonLabs/gaymc
```
gay    → Witness sheaf cohomology without computing
master → Parallel Čech cohomology verification
slave  → Step-by-step local-to-global certification
```

### Tritwies/gaymc
```
gay    → Observe temporal narratives passively
master → Parallel interval sheaf composition
slave  → Sequential snapshot verification
```

### bmorphism/gaymc
```
gay    → Witness spined category structure
master → Parallel triangulation functor tests
slave  → Sequential tree-width verification
```

## Workflow

1. **Develop on `master`** - abundance mindset, practical parallelism
2. **Debug on `slave`** - when something breaks, linearize
3. **Verify on `gay`** - maximum SPI, all universes simultaneously

```bash
# Switch to slave for debugging
git checkout slave
GAYMC_MODE=slave julia bench/gaymc_regression.jl

# Back to master for parallel testing  
git checkout master
GAYMC_MODE=master julia --threads=auto bench/gaymc_regression.jl

# Gay mode - infinite parallelism, SPI to its fullest
git checkout gay
GAYMC_MODE=gay julia --threads=auto bench/gaymc_regression.jl
# Runs all algorithms simultaneously across all available threads
```

## CI Labels

Add these labels to PRs to trigger specific modes:

- `slave` - Force sequential testing
- `master` - Force parallel testing (default for main)
- `gay` - Observer mode
- `plurigrid` / `teglonlabs` / `tritwies` / `bmorphism` - Fork-specific tests
