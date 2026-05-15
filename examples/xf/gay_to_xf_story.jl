# # From Gay.jl to XF.jl: The Journey to 9.5 Billion Colors/Second
#
# This is the story of how a simple idea — **deterministic random colors** —
# evolved from a CPU library into a GPU-accelerated system generating
# billions of reproducible colors per second.
#
# > "If nature is unjust, change nature!" — Laboria Cuboniks, Xenofeminist Manifesto
#
# ## Chapter 1: The Problem with Random Colors
#
# Traditional random color generation has a fundamental issue: **non-reproducibility**.
#
# ```julia
# # Traditional approach: different colors every run
# using Random
# Random.seed!(42)
# color1 = rand(3)  # [0.37, 0.58, 0.91]
# 
# # In parallel: race conditions cause chaos
# Threads.@threads for i in 1:100
#     colors[i] = rand(3)  # Different results each run!
# end
# ```
#
# This breaks:
# - **Scientific reproducibility** — can't share "color seeds" between labs
# - **Game development** — players see different visuals each session
# - **Parallel rendering** — threads fight over global RNG state
#
# ## Chapter 2: Gay.jl — Splittable Determinism
#
# [Gay.jl](https://github.com/bmorphism/Gay.jl) solved this with **splittable RNGs**
# from [SplittableRandoms.jl](https://github.com/Julia-Tempering/SplittableRandoms.jl),
# inspired by [Pigeons.jl](https://pigeons.run)'s Strong Parallelism Invariance (SPI):

using XF

# The core insight: each color operation SPLITS the RNG instead of advancing it
# This creates independent child streams that don't interfere:
#
# ```
# seed(42) → rng₀
#            ├── split → rng₁ → color[1]
#            ├── split → rng₂ → color[2]  
#            └── split → rng₃ → color[3]
# ```

# ## Gay.jl's Greatest Hits: 1069 Sky Models
#
# Gay.jl generated a gallery of 1069 black hole sky models in parallel,
# each with deterministic colors from forked RNG streams:
#
# ```
# #1 [rings] seed=51749 (4 rings)
#    (ring 0.63 0.23) + (ring 0.91 0.18) + (ring 1.22 0.11) + (ring 1.52 0.29)
#    
# #2 [rings] seed=73597 (4 rings)  
#    (ring 0.73 0.23) + (ring 0.99 0.14) + (ring 1.25 0.23) + (ring 1.56 0.12)
# ```
#
# The "Nice Black Hole" from seed 69:
# ```
#        🌌 gay_seed!(69) 🌌
# 
#            ░░░░▒▒▒▓▓▓▓▓▓▒▒▒░░░░
#        ░░▒▒▓▓████████████████▓▓▒▒░░
#      ░▒▓█████████▓▓▒▒▒▒▓▓█████████▓▒░
#    ░▒████████▓▒░░        ░░▒▓████████▒░
#
#   (ring 0.69 0.169) + (gaussian 0.42) + (ring 1.069 0.269)
#    ^^^^^^^           ^^^^^^^^^^^^^^    ^^^^^^^^^^^^^^^^
#    golden            deep blue         silver
# ```

#-

# ## Chapter 3: The O(index) Problem
#
# Gay.jl's `color_at(i)` had a limitation: it requires O(i) RNG splits to reach index i.
#
# This is fine for small indices:

println("=== O(index) Complexity World ===")
using Chairmarks

# Small indices are fast
print("color_at(10):    "); @b XF.color_at(10)
print("color_at(100):   "); @b XF.color_at(100)
print("color_at(1000):  "); @b XF.color_at(1000)
print("color_at(10000): "); @b XF.color_at(10000)

# But scales linearly with index (not constant time)

#-

# ## Chapter 4: XF.jl — The GPU Solution
#
# XF.jl solves this with a **hash-based O(1) approach** for GPU kernels:
#
# ```julia
# # Instead of O(index) splits:
# for _ in 1:index
#     current = split(current)  # O(index)
# end
#
# # Use O(1) hash:
# h = splitmix64(seed ⊻ index)  # O(1)
# r, g, b = extract_rgb(h)
# ```
#
# The `splitmix64` hash function (shared with Gay.jl for cross-compatibility):

function splitmix64_world(x::UInt64)
    x = xor(x, x >> 30) * 0xbf58476d1ce4e5b9
    x = xor(x, x >> 27) * 0x94d049bb133111eb
    xor(x, x >> 31)
end

# This enables massive GPU parallelism — each thread computes its color in O(1).

#-

# ## Chapter 5: Metal GPU Acceleration
#
# Using [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl),
# we wrote ONE kernel that runs on Metal, CUDA, AMD, and CPU:
#
# ```julia
# @kernel function _xf_colors_kernel!(colors, @Const(seed::UInt64))
#     i = @index(Global)
#     r, g, b = hash_color_rgb(seed, UInt64(i))
#     colors[i, 1] = r
#     colors[i, 2] = g  
#     colors[i, 3] = b
# end
# ```

println("\n=== KernelAbstractions GPU Kernel ===")
using Metal
using KernelAbstractions

# Generate colors on Apple M5 GPU (80 cores)
n = 1_000_000
gpu_colors = KernelAbstractions.zeros(MetalBackend(), Float32, n, 3)
kernel! = XF._xf_colors_kernel!(MetalBackend(), 256)
kernel!(gpu_colors, XF.XF_SEED, ndrange=n)
KernelAbstractions.synchronize(MetalBackend())

println("Generated $n colors on Metal GPU")
println("First 5 colors:")
cpu_colors = Array(gpu_colors)
for i in 1:5
    r, g, b = cpu_colors[i, 1], cpu_colors[i, 2], cpu_colors[i, 3]
    hex = string("#", 
        string(round(Int, r * 255), base=16, pad=2),
        string(round(Int, g * 255), base=16, pad=2),
        string(round(Int, b * 255), base=16, pad=2)) |> uppercase
    println("  color[$i] = $hex")
end

#-

# ## Chapter 6: SPI Verification — The Invariant
#
# The key guarantee: **same seed → same colors**, regardless of backend.

println("\n=== SPI (Strong Parallelism Invariance) ===")

seed = XF.XF_SEED
n_verify = 1000

# CPU sequential
cpu_seq = zeros(Float32, n_verify, 3)
for i in 1:n_verify
    r, g, b = XF.hash_color_rgb(UInt64(seed), UInt64(i))
    cpu_seq[i, 1] = r
    cpu_seq[i, 2] = g
    cpu_seq[i, 3] = b
end

# Metal GPU parallel
gpu_buf = KernelAbstractions.zeros(MetalBackend(), Float32, n_verify, 3)
kernel!(gpu_buf, UInt64(seed), ndrange=n_verify)
KernelAbstractions.synchronize(MetalBackend())
gpu_result = Array(gpu_buf)

# Verify
match = cpu_seq ≈ gpu_result
cpu_hash = reduce(xor, reinterpret(UInt32, vec(cpu_seq)))
gpu_hash = reduce(xor, reinterpret(UInt32, vec(gpu_result)))

println("CPU Sequential == Metal GPU: ", match ? "✓ PASS" : "✗ FAIL")
println("Hash integrity: ", cpu_hash == gpu_hash ? "✓ PASS" : "✗ FAIL")
println("Hash value: $cpu_hash")

#-

# ## Chapter 7: Benchmark Results
#
# **Apple M5 (80 GPU cores) benchmark:**
#
# | Colors | CPU (M/s) | Metal GPU (M/s) | Speedup |
# |--------|-----------|-----------------|---------|
# | 1K     | 974       | 7               | 0.01x   |
# | 10K    | 1,093     | 91              | 0.08x   |
# | 100K   | 1,030     | 757             | 0.74x   |
# | **1M** | 1,068     | **4,096**       | **3.8x**|
# | **10M**| —         | **8,455**       | ~8x     |
# | **100M**| —        | **9,558**       | ~9x     |
#
# **Peak throughput: 9.5 billion colors/second**

println("\n=== Live Benchmark ===")

for n_bench in [100_000, 1_000_000, 10_000_000]
    gpu_buf = KernelAbstractions.zeros(MetalBackend(), Float32, n_bench, 3)
    
    result = @b begin
        $kernel!($gpu_buf, UInt64($seed), ndrange=$n_bench)
        KernelAbstractions.synchronize(MetalBackend())
    end
    
    rate = n_bench / result.time / 1e6
    label = n_bench >= 1_000_000 ? "$(n_bench ÷ 1_000_000)M" : "$(n_bench ÷ 1000)K"
    println("  n=$label: $(round(result.time * 1000, digits=2)) ms → $(round(rate, digits=0)) M colors/s")
end

#-

# ## Chapter 8: The Palette Examples
#
# ### Pride Flags (from Gay.jl)
#
# Gay.jl introduced pride flag palettes that work in any color space:

println("\n=== Pride Flag Palettes ===")
println("Rainbow (sRGB):")
for (i, c) in enumerate(XF.rainbow())
    r, g, b = round(Int, c.r * 255), round(Int, c.g * 255), round(Int, c.b * 255)
    hex = "#" * string(r, base=16, pad=2) * string(g, base=16, pad=2) * string(b, base=16, pad=2) |> uppercase
    print("  $hex ")
end
println()

println("Trans Pride (Display P3):")
for (i, c) in enumerate(XF.transgender(XF.DisplayP3()))
    r, g, b = round(Int, clamp(c.r, 0, 1) * 255), round(Int, clamp(c.g, 0, 1) * 255), round(Int, clamp(c.b, 0, 1) * 255)
    hex = "#" * string(r, base=16, pad=2) * string(g, base=16, pad=2) * string(b, base=16, pad=2) |> uppercase
    print("  $hex ")
end
println()

#-

# ### Deterministic Palettes
#
# Same seed always produces the same palette:

println("\n=== Deterministic Palette World ===")
for seed in [42, 1337, 314159]
    println("Seed $seed:")
    print("  ")
    for i in 1:6
        r, g, b = XF.hash_color_rgb(UInt64(seed), UInt64(i))
        hex = string("#",
            string(round(Int, r * 255), base=16, pad=2),
            string(round(Int, g * 255), base=16, pad=2),
            string(round(Int, b * 255), base=16, pad=2)) |> uppercase
        print("$hex ")
    end
    println()
end

#-

# ## Epilogue: The Xenofeminist Connection
#
# Why "XF"? The package embodies xenofeminist principles:
#
# - **Anti-naturalism**: Wide-gamut colors (Rec.2020) exceed "natural" sRGB vision
# - **Technomaterialism**: GPU acceleration turns algorithmic art into material practice
# - **Alienation as freedom**: Fork-safe parallelism liberates computation from centralized state
#
# The splittable RNG implements what Laboria Cuboniks calls the **"feminine zero"** —
# each `split()` produces autonomous child streams without consuming the parent.
# This is acephalic production: headless, non-hierarchical, swarming.
#
# > "Zero is said to be feminine, not as a lack, but as autoproduction."
#
# ---
#
# ## Summary
#
# | Package | Complexity | Throughput | Key Innovation |
# |---------|------------|------------|----------------|
# | Gay.jl  | O(index)   | ~1M/s CPU  | Splittable RNG for SPI |
# | XF.jl   | O(1)       | 9.5B/s GPU | Hash-based + KernelAbstractions |
#
# The journey from Gay.jl to XF.jl shows how a good abstraction (splittable determinism)
# can scale from CPU to GPU while preserving the core invariant:
#
# **Same seed → same colors, always.**

println("\n" * "=" ^ 60)
println("THE END")
println("Same seed → same colors, always.")
println("=" ^ 60)
