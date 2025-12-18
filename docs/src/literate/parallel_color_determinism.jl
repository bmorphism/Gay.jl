# # Parallel Color Determinism: Thread-Safety vs Reproducibility
#
# This example explores the critical difference between **thread-safe parallel 
# randomness** and **deterministic parallel colors** - a distinction at the heart
# of Gay.jl's design.
#
# ## The Problem with Parallel Randomness
#
# When generating colors in parallel, we face two competing approaches:
#
# ### Approach 1: Thread-Safe but Non-Deterministic
#
# Using locks or thread-local RNGs ensures thread safety, but colors depend on:
# - Thread scheduling order
# - Number of threads available
# - System load during execution
#
# ```julia
# # Thread-safe but non-deterministic
# using OhMyThreads: tmap
# using Random
# 
# function parallel_colors_unsafe(n, seed)
#     rng = MersenneTwister(seed)
#     lock = ReentrantLock()
#     
#     tmap(1:n) do i
#         lock(lock) do
#             rand(rng, RGB)  # Order depends on thread scheduling!
#         end
#     end
# end
# ```
#
# **Problem**: Run twice, get different color orders!
#
# ### Approach 2: Deterministic via Splittable Randomness
#
# Gay.jl uses `SplittableRandoms.jl` to ensure `color_at(i; seed=s)` returns
# the **same color regardless of**:
# - Which thread calls it
# - What order threads execute
# - Whether we run sequentially or in parallel
#
# This is **Strong Parallelism Invariance (SPI)** from Pigeons.jl.

# ## Live Demonstration

using Gay
using OhMyThreads: tmap, tforeach
using Random
using Colors: RGB

# ### Non-Deterministic: Standard RNG with Threads

function parallel_colors_standard(n::Int, seed::Int)
    # Each run may produce different results due to thread scheduling
    rng = Random.MersenneTwister(seed)
    lk = ReentrantLock()
    
    results = Vector{RGB{Float64}}(undef, n)
    tforeach(1:n) do i
        # Lock protects RNG but order is non-deterministic
        c = lock(lk) do
            RGB(rand(rng), rand(rng), rand(rng))
        end
        results[i] = c
    end
    results
end

# Run multiple times - results may differ!
println("=== Standard RNG (thread-safe but non-deterministic) ===")
for run in 1:3
    colors = parallel_colors_standard(6, 42)
    hashes = [hash(c) % 1000 for c in colors]
    println("Run $run: $hashes")
end

# ### Deterministic: Gay.jl's color_at()

function parallel_colors_deterministic(n::Int, seed::Int)
    # color_at uses SplittableRandoms - same result every time!
    tmap(1:n) do i
        color_at(i; seed=seed)
    end
end

# Run multiple times - always identical!
println("\n=== Gay.jl color_at (deterministic) ===")
for run in 1:3
    colors = parallel_colors_deterministic(6, 42)
    hashes = [hash(c) % 1000 for c in colors]
    println("Run $run: $hashes")
end

# ### Comparison: Sequential vs Parallel

println("\n=== Sequential vs Parallel Equivalence ===")

seed = 1337
n = 100

# Sequential
sequential = [color_at(i; seed=seed) for i in 1:n]

# Parallel with OhMyThreads
parallel = tmap(i -> color_at(i; seed=seed), 1:n)

# They MUST be identical
println("Sequential == Parallel: $(sequential == parallel)")
println("This is Strong Parallelism Invariance (SPI)!")

# ## Why This Matters for Games
#
# In GayInvaders, each enemy row gets a color via `color_at(row; seed=seed)`.
# 
# **With SPI guarantees:**
# - Same seed = same game colors, always
# - Players can share seeds: "Try seed 1337, the colors are beautiful!"
# - Replays are visually identical
# - Parallel rendering doesn't affect appearance
#
# **Without SPI:**
# - Colors shift based on CPU load
# - Screenshots differ from gameplay
# - No reproducible "color themes"

# ## OhMyThreads Integration in GayInvaders

# The game uses parallel color generation for performance:

function generate_enemy_palette(rows::Int, seed::Int)
    # Parallel but deterministic!
    tmap(1:rows) do row
        color_at(row; seed=seed)
    end
end

println("\n=== Enemy Palette Generation ===")
for seed in [42, 1337, 2024]
    palette = generate_enemy_palette(6, seed)
    print("Seed $seed: ")
    for c in palette
        r = round(Int, c.r * 255)
        g = round(Int, c.g * 255)  
        b = round(Int, c.b * 255)
        print("\e[38;2;$(r);$(g);$(b)m●\e[0m")
    end
    println()
end

# ## The Mathematics: Hash-Based Splitting
#
# Gay.jl's determinism comes from SplittableRandoms.jl:
#
# ```
# color_at(i; seed=s) = rgb_from_rng(split(SplittableRandom(s), i))
# ```
#
# Each index `i` creates an independent, deterministic RNG stream via cryptographic
# hashing. The key insight:
#
# **Splitting is a pure function**: `split(rng, i)` always returns the same child RNG.
#
# This means:
# - `color_at(42; seed=1337)` returns the same RGB on any machine, any thread count
# - You can compute colors out of order, in parallel, or skip indices
# - No synchronization needed between threads

# ## Summary
#
# | Property | Standard RNG + Locks | Gay.jl SplittableRandoms |
# |----------|---------------------|--------------------------|
# | Thread-safe | ◆ | ◆ |
# | Deterministic | ◇ | ◆ |
# | Order-independent | ◇ | ◆ |
# | Reproducible | ◇ | ◆ |
# | Parallelizable | Limited | Fully |
#
# Gay.jl provides **Strong Parallelism Invariance**: the mathematical guarantee
# that parallel execution produces identical results to sequential execution.
#
# ## References
#
# - [Pigeons.jl SPI Documentation](https://pigeons.run/dev/parallelism/)
# - [SplittableRandoms.jl](https://github.com/Julia-Tempering/SplittableRandoms.jl)
# - [OhMyThreads.jl](https://github.com/JuliaFolds2/OhMyThreads.jl)
