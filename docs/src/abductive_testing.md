# World Teleportation Abductive Testing

Gay.jl includes an **abductive testing framework** for exploring and verifying the deterministic color transformations in GayInvaders.

## What is Abductive Testing?

**Abductive reasoning** infers causes from observed effects—the inverse of deduction. Given an observed world color, we reason backwards to determine:

1. Which invader ID produced this color
2. What derangement (RGB channel permutation) was applied
3. What tropical blend parameter `t` was used

This is analogous to debugging: observing incorrect behavior and reasoning backwards to find the root cause.

## Key Concepts

### World Teleportation

Each invader "teleports" through a transformation pipeline:

```
Source Color → Derangement → Tropical Blend → World Color
     ↑              ↑              ↑
   SPI hash    RGB permute    min-plus algebra
```

All transformations are **deterministic** (Strong Parallelism Invariance), so the same invader ID + seed always produces the same world color.

### Teleportation Properties

The framework tests four key properties:

| Property | What it tests |
|----------|---------------|
| `SPIDeterminism` | Same id+seed → same world color (always) |
| `DerangementBijectivity` | Derangement is invertible (no information loss) |
| `TropicalIdempotence` | Tropical operations have expected algebraic properties |
| `SpinConsistency` | Ising spin derived from XOR parity is correct |

## REPL Commands

The Gay REPL provides interactive world exploration:

```
!teleport <id>   # Jump to invader's world
!world           # Show current world state  
!back            # Return to previous world
!abduce r g b    # Infer invader from observed RGB
!jump <n>        # Jump to nth hypothesis
!neighbors [r]   # Explore nearby invaders
!test [n]        # Run abductive roundtrip tests
```

### Example Session

```
gay[0]> !teleport 42
  ⚡ Teleported to world #42
  ─────────────────────────────────────────
  Invader #42
    Source:   ██ (SPI hash)
    Deranged: ██ (perm=1)
    World:    ██ (t=0.42)
    Spin:     ↓ (-1)
  ─────────────────────────────────────────

gay[0]> !neighbors 3
  Neighboring worlds (radius=3):
    [39] ██ → ██ ↑
    [40] ██ → ██ ↓
    [41] ██ → ██ ↑
  → [42] ██ → ██ ↓
    [43] ██ → ██ ↑
    [44] ██ → ██ ↓
    [45] ██ → ██ ↑

gay[0]> !abduce 0.5 0.3 0.7
  🔍 Abducing invader from color...
  Target: ██
  ─────────────────────────────────────────
  Top hypotheses (use !jump <n> to explore):
  [1] ID=12847 conf=0.923 ██
  [2] ID=34291 conf=0.891 ██
  [3] ID=7823 conf=0.887 ██

gay[0]> !jump 1
  🚀 Jumped to hypothesis #1
```

## Programmatic API

### Simulate Teleportation

```julia
using Gay: simulate_teleportation

# Get full state of an invader's teleportation
sim = simulate_teleportation(42, Gay.GAY_SEED)
# Returns: (id, seed, source, derangement, deranged, world_base, tropical_t, world, spin)
```

### Abductive Inference

```julia
using Gay: abduce_invader, abduce_from_source
using Colors: RGB

# Find invaders that could have produced this world color
observed = RGB(0.5, 0.3, 0.7)
hypotheses = abduce_invader(observed; search_range=1:50000, top_k=5)

for h in hypotheses
    println("ID=$(h.id) confidence=$(h.confidence)")
end

# If you have the source color, exact recovery is possible
source = RGB(0.2, 0.4, 0.6)
id = abduce_from_source(source)  # Returns exact ID or nothing
```

### Property Testing

```julia
using Gay: test_property, test_all_properties
using Gay: SPIDeterminism, DerangementBijectivity

# Test individual properties
@assert test_property(SPIDeterminism(), 42, Gay.GAY_SEED)
@assert test_property(DerangementBijectivity(), 42, Gay.GAY_SEED)

# Test all properties at once
props = test_all_properties(1337, Gay.GAY_SEED)
# Returns: (spi=true, bijectivity=true, idempotence=true, spin=true)
```

### Roundtrip Tests

```julia
using Gay: abductive_roundtrip_test

# Test that we can recover an invader from its world color
@assert abductive_roundtrip_test(42, Gay.GAY_SEED)

# Run many tests
using Gay: run_abductive_tests
results = run_abductive_tests(n_samples=100)
```

## Mathematical Background

### Tropical Geometry

The tropical semiring replaces standard arithmetic:

| Standard | Tropical |
|----------|----------|
| a + b | min(a, b) |
| a × b | a + b |
| 0 | ∞ |
| 1 | 0 |

Tropical blending creates hard edges and plateaus, unlike smooth linear interpolation.

### Derangements

A **derangement** is a permutation with no fixed points. For RGB (3 elements), there are exactly 2 derangements:

1. `[2, 3, 1]`: R→G, G→B, B→R (cyclic left)
2. `[3, 1, 2]`: R→B, G→R, B→G (cyclic right)

Each invader uses one based on `id % 2`.

### Ising Spin

Each invader has a spin σ ∈ {-1, +1} derived from XOR parity:

```julia
spin = ((id ⊻ (id >> 16)) & 1 == 0) ? 1 : -1
```

This enables magnetization calculations for invader fleets.

## Integration with MultipleInterfaces.jl

The `TeleportationProperty` types are inspired by [MultipleInterfaces.jl](https://github.com/bieganek/MultipleInterfaces.jl):

- Properties act like **interface predicates**
- Multiple properties can be composed and tested
- Dispatch on property types enables extensible testing

```julia
abstract type TeleportationProperty end
struct SPIDeterminism <: TeleportationProperty end
struct DerangementBijectivity <: TeleportationProperty end
# ...

# Dispatch on property type
function test_property(::SPIDeterminism, id, seed)
    sim1 = simulate_teleportation(id, seed)
    sim2 = simulate_teleportation(id, seed)
    return sim1.world == sim2.world
end
```

## See Also

- [GayInvaders](literate/gay_invaders.md) - The invader generation system
- [Strong Parallelism Invariance](getting_started.md#spi) - Core determinism guarantees
- [Tropical Geometry](https://en.wikipedia.org/wiki/Tropical_geometry) - Mathematical background
