# # Parallel Color Generation with Strong Parallelism Invariance
#
# Gay.jl's splittable RNG enables **fork-safe parallel color generation**
# with guaranteed reproducibility — the **Strong Parallelism Invariance (SPI)**
# property from Pigeons.jl.
#
# ## What is SPI?
#
# **Strong Parallelism Invariance** means:
#
# > The output is **bitwise identical** regardless of:
# > - Number of threads/processes
# > - Execution order
# > - Parallel vs sequential execution
#
# For colors: same seed → same colors, even when generated in parallel.

# ## Setup

using Gay
using Colors: RGB

println("Julia threads available: ", Threads.nthreads())

# ## The Problem with Standard RNGs
#
# Traditional RNGs maintain global state that causes race conditions.
# Different runs produce different results because threads
# access the shared RNG in unpredictable order.

# ## Splittable Solution
#
# With splittable RNGs, each thread gets an **independent stream**:
#
# ```
# Master seed (42069)
#     ├── Thread 1: split → stream₁ → colors₁
#     ├── Thread 2: split → stream₂ → colors₂
#     ├── Thread 3: split → stream₃ → colors₃
#     └── Thread 4: split → stream₄ → colors₄
# ```
#
# Each stream is deterministic and independent.

# ## Parallel Color Generation

function generate_colors_parallel(n::Int, master_seed::Int)
    colors = Vector{RGB{Float64}}(undef, n)
    Threads.@threads for i in 1:n
        colors[i] = color_at(i; seed=master_seed)
    end
    colors
end

function generate_colors_sequential(n::Int, master_seed::Int)
    [color_at(i; seed=master_seed) for i in 1:n]
end

# ## Verify SPI Property

n = 100
seed = 42069

parallel_colors = generate_colors_parallel(n, seed)
sequential_colors = generate_colors_sequential(n, seed)

@assert parallel_colors == sequential_colors
println("✓ SPI verified: parallel == sequential for $n colors")

# Run again to verify reproducibility
parallel_colors_2 = generate_colors_parallel(n, seed)
@assert parallel_colors == parallel_colors_2
println("✓ Reproducibility verified: parallel runs are identical")

# ## Palette Generation

gay_seed!(1337)
p1 = next_palette(6)
println("Generated palette with 6 colors")

# Indexed palette access
p_at_5 = palette_at(5, 6)
p_at_5_again = palette_at(5, 6)
@assert p_at_5 == p_at_5_again
println("✓ Indexed palette access is reproducible")

# ## Connection to Pigeons.jl
#
# This is exactly the pattern used in Pigeons.jl for parallel MCMC:
#
# | Pigeons.jl | Gay.jl |
# |------------|--------|
# | `PT` (parallel tempering) | Parallel palette generation |
# | `explorer.rng` | `GayRNG` |
# | Reproducible chains | Reproducible colors |
# | `n_rounds` | Number of palettes |
#
# The SplittableRandoms foundation is identical.

# ## Best Practices
#
# 1. **Use `color_at` for parallel work** — random access by index
# 2. **Pass master seed explicitly** — don't rely on global state
# 3. **Verify with sequential** — always test SPI property
# 4. **Document seeds** — share seeds for reproducibility

function reproducible_visualization(data::Vector; seed::Int)
    n = length(data)
    colors = [color_at(i; seed=seed) for i in 1:n]
    return (data=data, colors=colors, seed=seed)
end

viz = reproducible_visualization([1,2,3,4,5]; seed=2024)
println("Visualization with seed $(viz.seed):")
show_palette(viz.colors)

# Anyone with the same seed gets identical colors!
viz2 = reproducible_visualization([1,2,3,4,5]; seed=2024)
@assert viz.colors == viz2.colors
println("✓ Shareable reproducibility confirmed")

println("\n✓ Parallel SPI example complete")
