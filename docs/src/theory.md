# Theoretical Foundations

## Strong Parallelism Invariance (SPI)

Gay.jl is built on the **Strong Parallelism Invariance** pattern from [Pigeons.jl](https://pigeons.run), which guarantees reproducible results regardless of parallelization strategy.

### The Problem

Standard random number generators (RNGs) produce different results when:
- Threads execute in different orders
- Number of workers changes
- Sequential vs parallel execution

```
Standard RNG (non-deterministic):
  Thread 1: rand() → 0.42
  Thread 2: rand() → 0.73
  Total depends on which thread calls first!
```

### The Solution: Splittable Streams

**Splittable random streams** solve this by creating independent sub-streams:

```
Splittable RNG (deterministic):
  Parent RNG (seed=1069)
      ↓
  split() → Child 1 (always same stream)
      ↓
  split() → Child 2 (always same stream)
      ↓
  split() → Child 3 (always same stream)
```

Each `split()` operation:
1. Deterministically derives a child stream from the parent
2. Mutates the parent to advance its state
3. Guarantees the same child for the same split sequence

### Mathematical Foundation

The splitting operation uses **cryptographic hashing** to derive child states:

```
child_state = hash(parent_state || split_counter)
```

This ensures:
- **Determinism**: Same parent state → same child state
- **Independence**: Child streams are statistically independent
- **Irreversibility**: Cannot recover parent from child

## Color Theory

### Wide-Gamut Color Spaces

Gay.jl supports three color spaces with increasing gamut size:

```
              ┌─────────────────────────────────────┐
              │         CIE 1931 xy Diagram         │
              │                                     │
              │            ╱╲                       │
              │           ╱  ╲  Visible Gamut       │
              │          ╱    ╲                     │
              │         ╱  R   ╲ ← Rec.2020         │
              │        ╱   ╱╲   ╲                   │
              │       ╱   ╱  ╲   ╲ ← Display P3    │
              │      ╱   ╱ s  ╲   ╲                 │
              │     ╱   ╱ RGB  ╲   ╲                │
              │    ╱___╱________╲___╲               │
              │   G              B                  │
              └─────────────────────────────────────┘
```

| Space | Coverage | Primary Use |
|-------|----------|-------------|
| sRGB | ~35% visible | Web, legacy |
| Display P3 | ~45% visible | Apple, DCI |
| Rec.2020 | ~75% visible | HDR, UHDTV |

### Golden Angle Palette Generation

`next_palette()` uses the **golden angle** (≈137.5°) for optimal color separation:

```
φ = (1 + √5) / 2 ≈ 1.618  (golden ratio)
θ = 360° / φ² ≈ 137.5°    (golden angle)
```

Colors are placed at intervals of θ around the HSV hue wheel, maximizing perceptual distance.

## Comrade.jl Sky Models

### VLBI Imaging

Very Long Baseline Interferometry (VLBI) reconstructs images from sparse Fourier measurements. The Event Horizon Telescope uses this to image black holes.

Gay.jl's sky models are simplified versions of [Comrade.jl](https://github.com/ptiede/Comrade.jl)'s primitives:

### Model Composition

Sky models form a **semiring** under addition and multiplication:

```
(M, +, *, 0, 1) where:
  + = additive blending (intensity sum)
  * = multiplicative masking (intensity product)
  0 = zero intensity everywhere
  1 = unit intensity everywhere
```

This allows compositional model building:

```julia
m87 = ring + gaussian           # Additive
masked = ring * crescent        # Multiplicative
complex = (ring1 + ring2) * disk  # Combined
```

### Aesthetic Scoring

Models are scored using:

```
score = 0.3 × coverage + 0.2 × contrast + 0.3 × symmetry + 0.2 × golden_bonus

where:
  coverage = fraction of non-zero pixels
  contrast = std(intensities)
  symmetry = 1 - std(radii) / mean(radii)
  golden_bonus = 1 / |ring_count / φ - 1|
```

The golden bonus rewards ring counts near the golden ratio (≈1.618).
