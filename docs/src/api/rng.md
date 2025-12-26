# RNG Control

Gay.jl uses **splittable random streams** from [SplittableRandoms.jl](https://github.com/Julia-Tempering/SplittableRandoms.jl) to ensure **Strong Parallelism Invariance (SPI)**.

## What is SPI?

Strong Parallelism Invariance guarantees that our results are **identical** regardless of:
- Number of threads or processes
- Order of execution
- Whether we run in parallel or sequentially

This is critical for reproducible scientific computing.

## Types

```@docs
GayRNG
```

## Functions

```@docs
gay_seed!
gay_rng
gay_split
```

## How Splitting Works

```
                    gay_seed!(1069)
                          |
                    [Root RNG]
                    /         \
              split()       split()
                /               \
           [Stream 1]       [Stream 2]
              |                 |
            color_1          color_2
```

Each `split()` operation:
1. Mutates the parent RNG
2. Returns an independent child stream
3. Guarantees deterministic results

## Example: Reproducible Colors

```julia
using Gay

# First run
gay_seed!(1069)
colors_run1 = next_colors(10)

# Reset and run again - IDENTICAL
gay_seed!(1069)
colors_run2 = next_colors(10)

@assert colors_run1 == colors_run2
```

## Example: Independent Streams

```julia
using Gay

gay_seed!(42)
stream1 = gay_split()
stream2 = gay_split()

# These streams produce independent sequences
c1 = rand(stream1)
c2 = rand(stream2)
```

## Default Seed

The default seed is **1069** (balanced ternary: `[+1, -1, -1, +1, +1, +1, +1]`).
