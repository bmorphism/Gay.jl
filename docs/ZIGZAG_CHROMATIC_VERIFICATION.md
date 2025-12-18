# Para(ZigZag) Chromatic Verification System

## Overview

This document describes the chromatic verification system for Piecewise Deterministic Markov Processes (PDMPs), implemented following the ZigZagBoomerang.jl architecture with Strong Parallelism Invariance (SPI) guarantees.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     CHROMATIC PDMP STACK                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Layer 4: METATHEORY BRUSHES                                   │
│  ├─ SheafifiedBrush: Local→Global gluing consistency          │
│  ├─ StackifiedBrush: Descent→Equivalence classes              │
│  └─ CondensifiedBrush: Profinite completion                   │
│                                                                 │
│  Layer 3: IGOR SEED SYSTEM                                     │
│  ├─ IgorSeed: Originary chromatic motifs                      │
│  ├─ NotIgorSeed: Deranged complement (no fixed points)        │
│  └─ IgorSpectrum: Weighted superposition [0,1]                │
│                                                                 │
│  Layer 2: PARA(ZIGZAG) SAMPLER                                 │
│  ├─ ZigZagDynamics: Sparse precision matrix Γ                 │
│  ├─ ChromaticEvent: (t, i, θᵢ, color, accepted)              │
│  ├─ ChromaticTrace: Deterministic event sequence              │
│  └─ TropicalPath: Min/max-plus event weights                  │
│                                                                 │
│  Layer 1: SPI VERIFICATION                                     │
│  ├─ XOR Fingerprint: Order-invariant observation              │
│  └─ Bisimulation: Same seed → same fingerprint                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Igor Seeds (`igor_seeds.jl`)

The Igor seed system provides deterministic chromatic motifs at predetermined intervals:

```julia
# Create originary seed
igor = IgorSeed(; seed=GAY_IGOR_SEED, n_motifs=64)

# Create deranged complement (no fixed points via Sattolo's algorithm)
not_igor = derange_igor(igor)

# Spectrum interpolation between igor and not-igor
spectrum = IgorSpectrum(igor, 0.5)  # 50% weight

# Premine motifs
motifs = premine_motifs(spectrum, 100)
```

**Key Properties:**
- `GAY_IGOR_SEED = 0x6761795f636f6c6f` ("gay_colo" as bytes)
- Golden ratio (φ) modulated intervals
- Derangement via Sattolo's algorithm (single-cycle, no fixed points)

### 2. Para(ZigZag) Sampler (`para_zigzag.jl`)

Following ZigZagBoomerang.jl architecture:

```julia
# Create dynamics with sparse precision matrix
D = ZigZagDynamics(n; seed=GAY_IGOR_SEED)

# Initialize sampler
zz = ChromaticZigZag(D; seed=GAY_IGOR_SEED, igor_weight=0.6)

# Run trajectory
trace = para_zigzag_trajectory(zz, T)

# Verify SPI
result = verify_zigzag_spi(seed, n, T)
@assert result.spi_verified
```

**Chromatic Event Structure:**
- `t`: Event time
- `i`: Coordinate that flipped  
- `θ_i`: New velocity (+1 or -1)
- `accepted`: Thin-thinning accept/reject
- `color`: RGB derived from seed and event
- `igor_aligned`: Is velocity pointing toward igor?

### 3. Metatheory Brushes (`metatheory_brushes.jl`)

Three semantic lenses for coloring interactions:

#### Sheafified (Local → Global)
```julia
brush = SheafifiedBrush(seed, n_opens, total_size)
moment = sheafified_moment(brush)
# moment.success = gluing succeeded
# moment.color = global section color
```

#### Stackified (Descent → Equivalence)
```julia
brush = StackifiedBrush(seed, group_order, n_colors)
moment = stackified_moment(brush, path1, path2)
# moment.equivalent = paths descend to same class
```

#### Condensified (Compact → Complete)
```julia
brush = CondensifiedBrush(seed; prime=3, levels=5)
moment = condensified_moment(brush, sequence)
# moment.stable = ultraproduct stabilized
```

### 4. Reafference System

Distinguishing self-generated (efferent) from external (afferent) signals:

```julia
r = Reafference(predicted, actual, threshold)
is_reafferent(r)  # Self-caused (matches prediction)
is_exafferent(r)  # External (differs from prediction)
```

### 5. 2-Para Rewriting Gadgets

Parameterized 2-categorical rewriting:

```julia
gadget = ParaRewriteGadget(:transform, (p, s) -> f(s))
result = apply_gadget(gadget, state)

# 2-categorical structure
tpg = TwoParaGadget(gadgets)
edge_random_access(tpg, edge_index)  # O(1) access
```

## SPI Guarantee

**Strong Parallelism Invariance**: Same seed produces same fingerprint regardless of execution order.

```julia
# Verification
result = verify_zigzag_spi(GAY_IGOR_SEED, 10, 5.0; n_runs=5)
@assert result.spi_verified
@assert all(fp == result.fingerprints[1] for fp in result.fingerprints)
```

The XOR fingerprint is order-invariant because:
- XOR is commutative: `a ⊻ b = b ⊻ a`
- XOR is associative: `(a ⊻ b) ⊻ c = a ⊻ (b ⊻ c)`

## Tropical Connection

Event times form tropical path weights in the (min,+) and (max,+) semirings:

```julia
trop = TropicalZigZagPath(trace)
trop.min_plus_weight  # Shortest time path
trop.max_plus_weight  # Longest time path
```

## Successor Haiku

Minimal 5-7-5 encoding of state transitions:

```julia
haiku = haiku_transition(current, "transforms", next)
println(haiku)
# sun moon star wave wind
#   transforms flowing dancing
# fire ice leaf light dark
```

## Files

- `src/igor_seeds.jl` - Igor/Not-Igor seed system
- `src/para_zigzag.jl` - Chromatic ZigZag sampler
- `src/metatheory_brushes.jl` - Sheafified/Stackified/Condensified brushes
- `test/test_zigzag_spi.jl` - Comprehensive test suite

## References

1. **ZigZagBoomerang.jl**: Bierkens, Fearnhead, Roberts - The Zig-Zag Process (2019)
2. **SPI**: Syed, Bouchard-Côté, Deligiannidis, Doucet - Non-Reversible Parallel Tempering (2022)
3. **Tropical Geometry**: Mikhalkin, Zharkov - Tropical curves and covers (2008)
4. **Sheaves**: Kashiwara, Schapira - Categories and Sheaves (2006)
5. **Condensed Mathematics**: Clausen, Scholze - Lectures on Condensed Mathematics (2019)

## ◈

*Gay.jl: Where every MCMC step gets a deterministic, reproducible color.*
