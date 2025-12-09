# Gay.jl 🏳️‍🌈

**Wide-gamut color sampling with splittable determinism**

Gay.jl provides reproducible color generation using the **Strong Parallelism Invariance (SPI)** pattern from [Pigeons.jl](https://pigeons.run) and [SplittableRandoms.jl](https://github.com/Julia-Tempering/SplittableRandoms.jl).

## Features

### 🎲 Deterministic Colors
Same seed = same colors, always — regardless of execution order or parallelism.

```julia
using Gay

gay_seed!(42)
c1 = next_color()
c2 = next_color()

gay_seed!(42)  # Reset
@assert next_color() == c1  # Identical!
```

### 🎨 Wide-Gamut Support
Beyond sRGB: Display P3, Rec.2020, and custom color spaces.

```julia
rainbow(SRGB())       # Standard
rainbow(DisplayP3())  # Apple devices
rainbow(Rec2020())    # HDR/UHDTV
```

### 🔢 Random Access
Jump to any position in the color sequence without iteration:

```julia
color_at(1)      # First color
color_at(1000)   # 1000th color (no iteration needed)
```

### 🏳️‍🌈 Pride Palettes
Accurate pride flag color schemes in any color space:

```julia
rainbow()
transgender()
bisexual()
pride_flag(:progress)
```

### 🌌 Comrade.jl-Style Sky Models
Colored S-expressions for VLBI imaging models:

```julia
gay_seed!(2017)
model = sky_add(
    comrade_ring(1.0, 0.3),
    comrade_gaussian(0.5)
)
comrade_show(model)
```

### 🔬 Abductive Testing
Reason backwards from observed colors to infer their sources:

```julia
using Gay: abduce_invader, simulate_teleportation

# Simulate an invader's world color
sim = simulate_teleportation(42, Gay.GAY_SEED)

# Abduce: given a color, find which invader produced it
hypotheses = abduce_invader(sim.world; top_k=5)
# hypotheses[1].id == 42 (the correct invader!)
```

See [Abductive Testing](abductive_testing.md) for REPL commands and full API.

### 🌐 Distributed Verification
Verify tensor-parallel inference across multiple devices:

```julia
using Gay: verify_allgather, verify_pipeline_handoff, ExoCluster

# Exo cluster: two MacBooks running OLMo
cluster = ExoCluster([
    ("MacBook Pro", 18.0, "192.168.1.10"),
    ("MacBook Air", 8.0, "192.168.1.11"),
], "olmo-7b")

# Pre-compute expected fingerprints
expected = expected_fingerprint(seed, n_tokens, hidden_dim)

# Verify without gathering: fp(A ∪ B) = fp(A) ⊕ fp(B)
@assert verify_exo_inference(cluster, "Hello")
```

See [Distributed SPI](distributed_spi.md) and [Fault Tolerance](fault_tolerance.md).

### 🔀 Propagator Networks
Constraint propagation with chromatic debugging:

```julia
using Gay.PropagatorLisp: @prop_str, cell_value

prop"""
(define-cell x)
(define-cell y) 
(define-cell sum)

(constraint-add x y sum)

(tell x 5)
(tell sum 12)
"""

@assert cell_value(:y) == 7  # Inferred!
```

See [Propagators](propagators.md) for bidirectional constraints and Lisp DSL.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/bmorphism/Gay.jl")
```

## Quick Start

```julia
using Gay

# Set seed for reproducibility
gay_seed!(42)

# Generate colors
c = next_color()           # Single color
palette = next_palette(6)  # 6 visually distinct colors

# Display
show_palette(palette)
```

## Why "Splittable Determinism"?

Traditional RNGs maintain global state that causes race conditions in parallel code. Gay.jl uses **splittable RNGs** where each operation creates an independent child stream:

```
seed(42) → rng₀
           ├── split → rng₁ → color₁
           ├── split → rng₂ → color₂
           └── split → rng₃ → color₃
```

This means:
- ✓ Same seed always produces same colors
- ✓ Parallel execution is reproducible
- ✓ Random access by index is efficient

The same pattern powers [Pigeons.jl](https://pigeons.run)'s reproducible MCMC and [Comrade.jl](https://github.com/ptiede/Comrade.jl)'s black hole imaging.

## Related Packages

- [SplittableRandoms.jl](https://github.com/Julia-Tempering/SplittableRandoms.jl) — RNG foundation
- [Pigeons.jl](https://pigeons.run) — Parallel tempering MCMC (SPI origin)
- [Comrade.jl](https://github.com/ptiede/Comrade.jl) — EHT black hole imaging
- [LispSyntax.jl](https://github.com/swadey/LispSyntax.jl) — S-expression support
- [Colors.jl](https://github.com/JuliaGraphics/Colors.jl) — Color types
- [PerceptualColourMaps.jl](https://github.com/peterkovesi/PerceptualColourMaps.jl) — Perceptual color science

## License

MIT
