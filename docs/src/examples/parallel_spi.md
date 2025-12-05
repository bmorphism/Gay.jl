```@meta
EditURL = "../literate/parallel_spi.jl"
```

# Parallel Color Generation with Strong Parallelism Invariance

Gay.jl's splittable RNG enables **fork-safe parallel color generation**
with guaranteed reproducibility — the **Strong Parallelism Invariance (SPI)**
property from Pigeons.jl.

## What is SPI?

**Strong Parallelism Invariance** means:

> The output is **bitwise identical** regardless of:
> - Number of threads/processes
> - Execution order
> - Parallel vs sequential execution

For colors: same seed → same colors, even when generated in parallel.

## Setup

````@example parallel_spi
using Gay
using Base.Threads

println("Julia threads available: ", nthreads())
````

## The Problem with Standard RNGs

Traditional RNGs maintain global state that causes race conditions:

```julia
# ⚠️ NOT REPRODUCIBLE — race condition!
using Random
Random.seed!(42)
results = Vector{Float64}(undef, 100)
@threads for i in 1:100
    results[i] = rand()  # Different each run!
end
```

Different runs produce different results because threads
access the shared RNG in unpredictable order.

## Splittable Solution

With splittable RNGs, each thread gets an **independent stream**:

```
Master seed (42069)
    ├── Thread 1: split → stream₁ → colors₁
    ├── Thread 2: split → stream₂ → colors₂
    ├── Thread 3: split → stream₃ → colors₃
    └── Thread 4: split → stream₄ → colors₄
```

Each stream is deterministic and independent.

## Parallel Color Generation

````@example parallel_spi
function generate_colors_parallel(n::Int, master_seed::Int)
    colors = Vector{RGB}(undef, n)

    @threads for i in 1:n
````

Each index gets deterministic color from master seed

````@example parallel_spi
        colors[i] = color_at(i; seed=master_seed)
    end

    colors
end

function generate_colors_sequential(n::Int, master_seed::Int)
    [color_at(i; seed=master_seed) for i in 1:n]
end
````

## Verify SPI Property

````@example parallel_spi
n = 100
seed = 42069

parallel_colors = generate_colors_parallel(n, seed)
sequential_colors = generate_colors_sequential(n, seed)

@assert parallel_colors == sequential_colors
println("✓ SPI verified: parallel == sequential for $n colors")
````

Run again to verify reproducibility

````@example parallel_spi
parallel_colors_2 = generate_colors_parallel(n, seed)
@assert parallel_colors == parallel_colors_2
println("✓ Reproducibility verified: parallel runs are identical")
````

## Parallel Sky Model Gallery

Generate many Comrade-style models in parallel:

````@example parallel_spi
function generate_model_gallery(n::Int; master_seed::Int=42069)
    models = Vector{SkyModel}(undef, n)
    styles = [:m87, :sgra, :custom]

    @threads for i in 1:n
````

Deterministic style selection

````@example parallel_spi
        style_rng = SplittableRandom(UInt64(master_seed + i))
        style_idx = mod(style_rng.x, length(styles)) + 1
        style = styles[style_idx]
````

Deterministic model generation

````@example parallel_spi
        models[i] = comrade_model(seed=master_seed + i, style=style)
    end

    models
end

println("\n=== Parallel Model Gallery ===")
println("Generating 16 models in parallel...")

models = generate_model_gallery(16; master_seed=42069)
println("Generated $(length(models)) models")
````

Show first few

````@example parallel_spi
for i in 1:3
    println("\nModel $i:")
    println("  ", sky_show(models[i]))
end
````

Verify reproducibility

````@example parallel_spi
models_2 = generate_model_gallery(16; master_seed=42069)
for i in 1:16
    @assert sky_show(models[i]) == sky_show(models_2[i])
end
println("\n✓ Gallery reproducibility verified")
````

## Performance: Parallel Palette Generation

````@example parallel_spi
function timed_palette_generation(n_palettes::Int, palette_size::Int; seed::Int=1337)
    palettes = Vector{Vector{RGB}}(undef, n_palettes)

    t = @elapsed begin
        @threads for i in 1:n_palettes
            palettes[i] = palette_at(i, palette_size; seed=seed)
        end
    end

    (palettes, t)
end

println("\n=== Performance Benchmark ===")

n_palettes = 1000
palette_size = 6

(palettes, parallel_time) = timed_palette_generation(n_palettes, palette_size)
println("Generated $n_palettes palettes of $palette_size colors each")
println("  Time: $(round(parallel_time * 1000, digits=2)) ms")
println("  Rate: $(round(n_palettes / parallel_time, digits=0)) palettes/sec")
````

Verify all palettes are reproducible

````@example parallel_spi
(palettes_2, _) = timed_palette_generation(n_palettes, palette_size)
@assert palettes == palettes_2
println("  ✓ All palettes reproducible")
````

## Connection to Pigeons.jl

This is exactly the pattern used in Pigeons.jl for parallel MCMC:

| Pigeons.jl | Gay.jl |
|------------|--------|
| `PT` (parallel tempering) | Parallel palette generation |
| `explorer.rng` | `GayRNG` |
| Reproducible chains | Reproducible colors |
| `n_rounds` | Number of palettes |

The SplittableRandoms foundation is identical.

## Fork Safety

Unlike thread-based parallelism, process forks (e.g., with `Distributed.jl`)
also work correctly because each process gets an independent RNG stream:

```julia
using Distributed
addprocs(4)

@everywhere using Gay

# Each worker generates its portion
results = pmap(1:1000) do i
    color_at(i; seed=42069)
end

# Identical to sequential!
@assert results == [color_at(i; seed=42069) for i in 1:1000]
```

## Best Practices

1. **Use `color_at` for parallel work** — random access by index
2. **Pass master seed explicitly** — don't rely on global state
3. **Verify with sequential** — always test SPI property
4. **Document seeds** — share seeds for reproducibility

````@example parallel_spi
println("\n=== Best Practice Example ===")

function reproducible_visualization(data::Vector; seed::Int)
    n = length(data)
````

Generate colors deterministically

````@example parallel_spi
    colors = [color_at(i; seed=seed) for i in 1:n]
````

(In real code: create plot with these colors)

````@example parallel_spi
    return (data=data, colors=colors, seed=seed)
end

viz = reproducible_visualization([1,2,3,4,5]; seed=2024)
println("Visualization with seed $(viz.seed):")
show_palette(viz.colors)
````

Anyone with the same seed gets identical colors!

````@example parallel_spi
viz2 = reproducible_visualization([1,2,3,4,5]; seed=2024)
@assert viz.colors == viz2.colors
println("✓ Shareable reproducibility confirmed")

println("\n✓ Parallel SPI example complete")
````

