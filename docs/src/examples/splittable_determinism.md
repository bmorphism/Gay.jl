```@meta
EditURL = "../literate/splittable_determinism.jl"
```

# Splittable Determinism: Reproducible Colors

Gay.jl uses **splittable random number generators** from
[SplittableRandoms.jl](https://github.com/Julia-Tempering/SplittableRandoms.jl)
to generate colors that are:

1. **Deterministic** — same seed always produces same colors
2. **Parallelizable** — independent streams for concurrent execution
3. **Random-access** — jump to any position without iteration

This pattern originates from [Pigeons.jl](https://pigeons.run)'s
**Strong Parallelism Invariance (SPI)** — the same principle used for
reproducible MCMC sampling in black hole imaging with Comrade.jl.

## Setup

````@example splittable_determinism
using Gay
````

## Basic Determinism

The fundamental property: **same seed → same colors, always**.

````@example splittable_determinism
gay_seed!(42)
c1 = next_color()
c2 = next_color()
c3 = next_color()

gay_seed!(42)  # Reset to same seed
@assert next_color() == c1  # Identical!
@assert next_color() == c2
@assert next_color() == c3

println("◆ Determinism verified: seed 42 always produces the same sequence")
````

## The Splittable RNG Model

Unlike traditional RNGs that maintain sequential state, splittable RNGs
create **independent child streams** via `split`. Each color operation
internally splits the RNG:

```
seed(42) → rng₀
           ├── split → rng₁ → color₁
           ├── split → rng₂ → color₂
           └── split → rng₃ → color₃
```

This means execution order doesn't matter — `color_at(3)` gives the same
result whether we computed colors 1 and 2 first or jumped directly.

## Random Access by Index

Access any color in the sequence without computing predecessors:

````@example splittable_determinism
c_1 = color_at(1)
c_42 = color_at(42)
c_1000 = color_at(1000)

println("Color at index 1:    ", c_1)
println("Color at index 42:   ", c_42)
println("Color at index 1000: ", c_1000)
````

Verify consistency:

````@example splittable_determinism
@assert color_at(42) == c_42  # Same color, always
@assert color_at(1) == c_1
````

## Batch Access

Efficiently retrieve colors at multiple indices:

````@example splittable_determinism
indices = [1, 10, 100, 500, 1000]
batch = colors_at(indices)

println("\nBatch colors at indices $indices:")
show_palette(batch)
````

## Palettes at Specific Seeds

Generate a visually distinct palette starting at any index:

````@example splittable_determinism
palette_5_at_1 = palette_at(1, 5)    # 5-color palette at index 1
palette_5_at_100 = palette_at(100, 5) # 5-color palette at index 100

println("\nPalette (5 colors) at index 1:")
show_palette(palette_5_at_1)

println("Palette (5 colors) at index 100:")
show_palette(palette_5_at_100)
````

## Why This Matters

### Reproducible Visualizations

Your plots will look identical across:
- Different machines
- Different Julia sessions
- Parallel vs sequential execution

### Shareable Seeds

Share a seed number and index to communicate exact colors:

```julia
# "Use color at index 137 with seed 2017"
gay_seed!(2017)
the_color = color_at(137)
```

### Debugging

When a visualization looks wrong, reproduce the exact state:

```julia
gay_seed!(problematic_seed)
# Now step through color generation to find the issue
```

## RNG State Inspection

````@example splittable_determinism
gay_seed!(1337)
for _ in 1:5
    next_color()
end

state = gay_rng_state()
println("\nRNG state after 5 colors:")
println("  Seed: ", state.seed)
println("  Invocation: ", state.invocation)
````

## Connection to Pigeons.jl

The splittable RNG pattern comes from parallel tempering MCMC:

| Pigeons.jl (MCMC)          | Gay.jl (Colors)           |
|----------------------------|---------------------------|
| `explorer.rng`             | `gay_rng()`              |
| `split(rng)`               | `gay_split()`            |
| Reproducible chains        | Reproducible palettes    |
| Fork-safe sampling         | Fork-safe color gen      |

Both use the same mathematical foundation:
[SplittableRandoms.jl](https://github.com/Julia-Tempering/SplittableRandoms.jl)

````@example splittable_determinism
println("\n◆ Splittable determinism example complete")
````

