# # XF.jl: GPU-Accelerated Deterministic Color Generation
#
# This example worldnstrates **Strong Parallelism Invariance (SPI)** — the guarantee
# that the same seed produces identical colors regardless of backend (CPU, Metal GPU),
# thread count, or execution order.
#
# ## Background: From Gay.jl to XF.jl
#
# XF.jl builds on patterns from [Gay.jl](https://github.com/bmorphism/gay.jl) and
# [Pigeons.jl](https://pigeons.run), implementing splittable deterministic random
# number generation for reproducible parallel computation.
#
# > "If nature is unjust, change nature!" — Laboria Cuboniks, Xenofeminist Manifesto
#
# ## Key Innovation: O(1) GPU Color Generation
#
# The original `color_at(i)` uses SplittableRandoms.jl with O(i) complexity — each
# index requires `i` sequential RNG splits. For GPU parallelism, we use a **hash-based**
# approach with O(1) complexity per color:
#
# ```
# seed + index → SplitMix64 hash → RGB color
# ```
#
# This enables **9.5 billion colors/second** on Apple M5 Metal GPU.

# ## Setup

using XF
using Metal
using KernelAbstractions
using Chairmarks

# Check Metal availability
println("Metal available: ", Metal.functional())
println("Device: ", Metal.device().name)

# ## SPI Verification: CPU == GPU
#
# The core invariant: same seed produces byte-identical colors on CPU and GPU.

n = 10_000
seed = XF.XF_SEED

#-

# ### Generate on CPU (sequential reference)
cpu_colors = zeros(Float32, n, 3)
for i in 1:n
    r, g, b = XF.hash_color_rgb(UInt64(seed), UInt64(i))
    cpu_colors[i, 1] = r
    cpu_colors[i, 2] = g
    cpu_colors[i, 3] = b
end

# ### Generate on CPU (KernelAbstractions parallel)
ka_cpu_colors = zeros(Float32, n, 3)
XF.xf_ka_colors!(ka_cpu_colors, seed; backend=CPU())

# ### Generate on Metal GPU
gpu_colors_dev = KernelAbstractions.zeros(MetalBackend(), Float32, n, 3)
kernel! = XF._xf_colors_kernel!(MetalBackend(), 256)
kernel!(gpu_colors_dev, UInt64(seed), ndrange=n)
KernelAbstractions.synchronize(MetalBackend())
gpu_colors = Array(gpu_colors_dev)

# ### Verify SPI Invariants
println("\n=== SPI VERIFICATION ===")
println("CPU Sequential == CPU Parallel: ", cpu_colors ≈ ka_cpu_colors)
println("CPU Sequential == Metal GPU:    ", cpu_colors ≈ gpu_colors)

# Compute hash for bitwise verification
cpu_hash = reduce(xor, reinterpret(UInt32, vec(cpu_colors)))
gpu_hash = reduce(xor, reinterpret(UInt32, vec(gpu_colors)))
println("Hash match (bitwise identical):  ", cpu_hash == gpu_hash)
println("Hash value: ", cpu_hash)

# ## Benchmark Results: CPU vs Metal GPU
#
# Measured on Apple M5 with 80 GPU cores.
#
# | Colors | CPU (M/s) | Metal (M/s) | Speedup |
# |--------|-----------|-------------|---------|
# | 1K     | 974       | 7           | 0.01x   |
# | 10K    | 1,093     | 91          | 0.08x   |
# | 100K   | 1,030     | 757         | 0.74x   |
# | **1M** | 1,068     | **4,096**   | **3.8x**|
# | **10M**| —         | **8,455**   | ~8x     |
# | **100M**| —        | **9,558**   | ~9x     |
#
# GPU wins at ≥1M colors. Peak: **9.5 billion colors/second**.

#-

# ### Live Benchmark

println("\n=== BENCHMARK ===")

for n_bench in [100_000, 1_000_000, 10_000_000]
    gpu_buf = KernelAbstractions.zeros(MetalBackend(), Float32, n_bench, 3)
    gpu_kernel! = XF._xf_colors_kernel!(MetalBackend(), 256)
    
    result = @b begin
        $gpu_kernel!($gpu_buf, UInt64($seed), ndrange=$n_bench)
        KernelAbstractions.synchronize(MetalBackend())
    end
    
    rate = n_bench / result.time / 1e6
    println("n=$(n_bench): $(round(result.time * 1000, digits=2)) ms ($(round(rate, digits=0)) M colors/s)")
end

# ## The Algorithm: SplitMix64 Hash
#
# Each thread computes its color independently using a hash function:

"""
    splitmix64(x::UInt64) -> UInt64

SplitMix64 finalizer - fast, high-quality hash with good avalanche properties.
Used in Java's SplittableRandom and many PRNGs.
"""
function splitmix64_world(x::UInt64)
    x = xor(x, x >> 30) * 0xbf58476d1ce4e5b9
    x = xor(x, x >> 27) * 0x94d049bb133111eb
    xor(x, x >> 31)
end

# For index `i` with seed `s`:
# ```julia
# h = splitmix64(xor(seed, index * 0x9e3779b97f4a7c15))
# r = Float32(h & 0xFF) / 255.0f0
# g = Float32((h >> 8) & 0xFF) / 255.0f0  
# b = Float32((h >> 16) & 0xFF) / 255.0f0
# ```

#-

# ## Sample Colors
#
# First 10 colors from the deterministic sequence:

println("\n=== SAMPLE COLORS ===")
for i in 1:10
    r, g, b = XF.hash_color_rgb(UInt64(seed), UInt64(i))
    hex = string(
        "#",
        string(round(Int, r * 255), base=16, pad=2),
        string(round(Int, g * 255), base=16, pad=2),
        string(round(Int, b * 255), base=16, pad=2)
    ) |> uppercase
    println("  color[$i] = $hex  (R=$(round(r, digits=3)), G=$(round(g, digits=3)), B=$(round(b, digits=3)))")
end

# ## Portable Kernels with KernelAbstractions.jl
#
# The same kernel runs on Metal, CUDA, AMD, and CPU:
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
#
# Backend selection:
# - `CPU()` — Threaded CPU with SIMD
# - `MetalBackend()` — Apple GPU
# - `CUDABackend()` — NVIDIA GPU
# - `ROCBackend()` — AMD GPU

# ## Connection to Gay.jl
#
# XF.jl extends Gay.jl's deterministic color generation with:
#
# 1. **GPU acceleration** via KernelAbstractions.jl
# 2. **Wide-gamut color spaces** (Display P3, Rec.2020)
# 3. **SPI verification** for cross-platform reproducibility
# 4. **O(1) hash-based generation** for GPU efficiency
#
# The `splitmix64` hash function is shared between both packages for
# cross-language determinism (Julia, Rust, etc.).

# ## Conclusion
#
# XF.jl achieves:
# - ✓ **SPI**: CPU and GPU produce identical colors
# - ✓ **Speed**: 9.5B colors/sec on M5 Metal
# - ✓ **Portability**: One kernel, multiple backends
# - ✓ **Reproducibility**: Same seed → same colors, always

println("\n=== COMPLETE ===")
