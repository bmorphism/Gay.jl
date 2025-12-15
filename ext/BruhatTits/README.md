# Bruhat-Tits 3×3 Curriculum Extension

Unified Enzyme.jl expert curriculum for Gay.jl with 3-partite tree saturation.

## Architecture

```
                              ┌─────────────────┐
                              │     ROOT        │
                              │ Symplectomorphic│
                              │   Cobordism     │
                              └────────┬────────┘
                                       │
           ┌───────────────────────────┼───────────────────────────┐
           │                           │                           │
    ┌──────▼──────┐             ┌──────▼──────┐             ┌──────▼──────┐
    │    ZAHN     │             │    JULES    │             │   FABRIZ    │
    │  ⊗ Tensor   │             │  ⊕ Coproduct │             │ ⊛ Convolve  │
    └──────┬──────┘             └──────┬──────┘             └──────┬──────┘
           │                           │                           │
    ┌──────┼──────┐             ┌──────┼──────┐             ┌──────┼──────┐
    │      │      │             │      │      │             │      │      │
   Z1     Z2     Z3            J1     J2     J3            F1     F2     F3
```

## Components

| Module | Order | Purpose |
|--------|-------|---------|
| `GayEnzymeZAHN.jl` | ⊗ Tensor | Enzyme.jl autodiff + symplectic geometry |
| `GayLearnableJULES.jl` | ⊕ Coproduct | Learnable color spaces + 3-MATCH 3-Col |
| `GayPerceptualFABRIZ.jl` | ⊛ Convolution | Perceptual spaces + cobordisms |
| `GaySymplectomorphicCurriculum.jl` | Root | Unified curriculum integration |
| `GayJolt3Col.jl` | Proof | Lasso/sum-check 3-coloring prover |
| `GayAPIAlignment.jl` | Bridge | Alignment with official Gay.jl API |

## Essentials vs Derived

The entire framework reduces to **3 essential primitives**:

```julia
# 1. Mixing function
@inline function sm64(z::UInt64)::UInt64
    z += 0x9E3779B97F4A7C15
    z = (z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ⊻ (z >> 27)) * 0x94D049BB133111EB
    z ⊻ (z >> 31)
end

# 2. Composition operator
⊻(a, b) = xor(a, b)

# 3. Coloring domain
gf3(n) = mod(n, 3)
distinct3(a, b, c) = a≠b && b≠c && a≠c
```

Everything else (ZAHN, JULES, FABRIZ, 23 workers, fingerprints) is **derived**.

## Coherence Condition

For bidirectional Free/Cofree module structure:

```
(f ⊳ c) ⊲ f' = f ⊳ (c ⊲ f')

Where:
  f, f' ∈ Free(sm64)      -- forward generation
  c     ∈ Cofree(sm64⁻¹)  -- backward observation
```

This requires **51 subagents**: 23 forward + 23 backward + 5 committee.

## Usage

```julia
using Gay
include("ext/BruhatTits/GaySymplectomorphicCurriculum.jl")

using .GaySymplectomorphicCurriculum

# Run full curriculum
result = run_full_curriculum(n_colors=100, epochs=50)

# Access components
result.curriculum.zahn_color_space
result.curriculum.jules_color_space
result.curriculum.fabriz_perceptual
```

## References

- [Topos Synergy Analysis](../docs/synergy/TOPOS_GAY_SYNERGY_ANALYSIS.md)
- Loregian (2025) "Two-dimensional transducers" arXiv:2509.06769
- Tao's notes on expander graphs and spectral gaps
