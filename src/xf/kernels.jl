# XF.jl KernelAbstractions Integration
# GPU-portable parallel color generation
#
# "If nature is unjust, change nature!" — Laboria Cuboniks
# 
# Provides SPMD kernels that run on:
# - CPU (default, uses threading + SIMD)
# - Metal.jl (Apple GPU)
# - CUDA.jl (Nvidia GPU)
# - AMDGPU.jl (AMD GPU)
# - oneAPI.jl (Intel GPU)

using KernelAbstractions
using SplittableRandoms: SplittableRandom, split

export XF_SEED, SplitMixRGB, splitmix_rgb, with_splitmix_rgb
export xf_ka_colors!, xf_ka_colors, xf_ka_benchmark
export XFBackend, xf_get_backend, xf_set_backend!

const XF_SEED = UInt64(0x78656e6f66656d21)  # "xenofem!"

# ═══════════════════════════════════════════════════════════════════════════
# Backend selection
# ═══════════════════════════════════════════════════════════════════════════

const XFBackend = Union{CPU, KernelAbstractions.Backend}

# Global backend (default: CPU)
const GLOBAL_BACKEND = Ref{XFBackend}(CPU())

"""
    xf_get_backend()

Get the current KernelAbstractions backend.
"""
xf_get_backend() = GLOBAL_BACKEND[]

"""
    xf_set_backend!(backend)

Set the KernelAbstractions backend for GPU acceleration.

# Example
```julia
using Metal
xf_set_backend!(MetalBackend())  # Use Apple GPU
```
"""
function xf_set_backend!(backend::XFBackend)
    GLOBAL_BACKEND[] = backend
    return backend
end

# ═══════════════════════════════════════════════════════════════════════════
# Fast deterministic hash-based color generation
# ═══════════════════════════════════════════════════════════════════════════

"""
    splitmix64(x::UInt64) -> UInt64

SplitMix64 finalizer - fast high-quality hash.
Same algorithm as used in Gay.jl for cross-language consistency.
"""
@inline function splitmix64(x::UInt64)
    x = xor(x, x >> 30) * 0xbf58476d1ce4e5b9
    x = xor(x, x >> 27) * 0x94d049bb133111eb
    xor(x, x >> 31)
end

"""
    hash_color_lch(seed::UInt64, index::UInt64) -> (Float32, Float32, Float32)

Generate deterministic LCH color components from seed and index.
Returns (L, C, H) with L∈[30,80], C∈[40,80], H∈[0,360].
Uses the XF.jl color space ranges for perceptual uniformity.

Note: Uses Float32 arithmetic throughout for Metal GPU compatibility.
"""
@inline function hash_color_lch(seed::UInt64, index::UInt64)
    h1 = splitmix64(xor(seed, index * 0x9e3779b97f4a7c15))
    h2 = splitmix64(h1)
    h3 = splitmix64(h2)
    
    # XF.jl ranges for good visibility - Float32 only for Metal
    L = 30.0f0 + Float32(h1 & 0xFFFF) / 65535.0f0 * 50.0f0   # 30-80
    C = 40.0f0 + Float32(h2 & 0xFFFF) / 65535.0f0 * 40.0f0   # 40-80
    H = Float32(h3 & 0xFFFF) / 65535.0f0 * 360.0f0            # 0-360
    
    (L, C, H)
end

"""
    hash_color_rgb(seed::UInt64, index::UInt64) -> (Float32, Float32, Float32)

Generate deterministic RGB color components from seed and index.
Returns (R, G, B) in [0, 1] range.

Note: Uses Float32 arithmetic throughout for Metal GPU compatibility.
"""
@inline function hash_color_rgb(seed::UInt64, index::UInt64)
    h = splitmix64(xor(seed, index * 0x9e3779b97f4a7c15))
    
    # Metal requires Float32 - no Float64 allowed
    r = Float32(h & 0xFF) / 255.0f0
    g = Float32((h >> 8) & 0xFF) / 255.0f0
    b = Float32((h >> 16) & 0xFF) / 255.0f0
    
    (r, g, b)
end

"""
    SplitMixRGB(seed::Integer=XF_SEED)

Callable SplitMix64 RGB stream. `SplitMixRGB(seed)(index)` returns the same
`(r, g, b)::NTuple{3,Float32}` as `splitmix_rgb(index; seed)`.
"""
struct SplitMixRGB
    seed::UInt64
end

SplitMixRGB(seed::Integer=XF_SEED) = SplitMixRGB(UInt64(seed))

"""
    splitmix_rgb(index::Integer; seed::Integer=XF_SEED)
    splitmix_rgb(seed::Integer, index::Integer)

Return the deterministic XF SplitMix64 RGB color at `index` for `seed`.
This is the scalar reference for `_xf_colors_kernel!` and `xf_ka_colors!`.
"""
@inline splitmix_rgb(index::Integer; seed::Integer=XF_SEED) =
    hash_color_rgb(UInt64(seed), UInt64(index))

@inline splitmix_rgb(seed::Integer, index::Integer) =
    hash_color_rgb(UInt64(seed), UInt64(index))

@inline (stream::SplitMixRGB)(index::Integer) =
    splitmix_rgb(index; seed=stream.seed)

"""
    with_splitmix_rgb(f::Function, seed::Integer=XF_SEED)

Create a `SplitMixRGB` resource and pass it to continuation `f`.
The continuation owns the stream boundary and decides what value persists.
"""
with_splitmix_rgb(f::Function, seed::Integer=XF_SEED) =
    f(SplitMixRGB(seed))

# ═══════════════════════════════════════════════════════════════════════════
# SPMD Kernels
# ═══════════════════════════════════════════════════════════════════════════

"""
SPMD kernel: Generate RGB colors into a pre-allocated n×3 matrix.
Each thread generates one color at its global index.
"""
@kernel function _xf_colors_kernel!(colors, @Const(seed::UInt64))
    i = @index(Global)
    r, g, b = hash_color_rgb(seed, UInt64(i))
    colors[i, 1] = r
    colors[i, 2] = g
    colors[i, 3] = b
end

"""
SPMD kernel: Generate LCH colors into a pre-allocated n×3 matrix.
"""
@kernel function _xf_lch_kernel!(colors, @Const(seed::UInt64))
    i = @index(Global)
    L, C, H = hash_color_lch(seed, UInt64(i))
    colors[i, 1] = L
    colors[i, 2] = C
    colors[i, 3] = H
end

"""
SPMD kernel: Parallel color sum reduction for billion-scale operations.
"""
@kernel function _xf_color_sum_kernel!(sums, @Const(seed::UInt64), @Const(chunk_size::Int))
    i = @index(Global)
    
    start_idx = UInt64((i - 1) * chunk_size + 1)
    
    local sr, sg, sb = 0.0f0, 0.0f0, 0.0f0
    
    for j in UInt64(0):UInt64(chunk_size - 1)
        idx = start_idx + j
        r, g, b = hash_color_rgb(seed, idx)
        sr += r
        sg += g
        sb += b
    end
    
    sums[i, 1] = sr
    sums[i, 2] = sg
    sums[i, 3] = sb
end

# ═══════════════════════════════════════════════════════════════════════════
# High-level API
# ═══════════════════════════════════════════════════════════════════════════

"""
    xf_ka_colors!(colors::AbstractMatrix{Float32}, seed::Integer=XF_SEED;
                   backend=xf_get_backend(), workgroup=256)

Fill a pre-allocated n×3 matrix with deterministic RGB colors using SPMD kernel.

# Example
```julia
colors = zeros(Float32, 1_000_000, 3)
xf_ka_colors!(colors, 42069)
```
"""
function xf_ka_colors!(colors::AbstractMatrix{Float32}, seed::Integer=XF_SEED;
                        backend::XFBackend=xf_get_backend(), workgroup::Int=256)
    n = size(colors, 1)
    @assert size(colors, 2) == 3 "colors must be n×3 matrix"
    
    kernel! = _xf_colors_kernel!(backend, workgroup)
    kernel!(colors, UInt64(seed), ndrange=n)
    KernelAbstractions.synchronize(backend)
    
    return colors
end

"""
    xf_ka_colors(n::Integer, seed::Integer=XF_SEED; backend=xf_get_backend())

Generate n deterministic RGB colors using SPMD kernel.
Returns n×3 Float32 matrix.

# Example
```julia
colors = xf_ka_colors(1_000_000, 42069)
```
"""
function xf_ka_colors(n::Integer, seed::Integer=XF_SEED;
                       backend::XFBackend=xf_get_backend(), workgroup::Int=256)
    colors = zeros(Float32, n, 3)
    xf_ka_colors!(colors, seed; backend=backend, workgroup=workgroup)
end

"""
    xf_ka_color_sums(n::Integer, seed::Integer=XF_SEED;
                      chunk_size::Int=10000, backend=xf_get_backend())

Compute RGB sums over n colors using parallel reduction.
Useful for billion-scale operations where storing all colors is impractical.

Returns (sum_r, sum_g, sum_b) as Float64 tuple.

# Example
```julia
# Sum 1 billion colors
sums = xf_ka_color_sums(1_000_000_000, 42069)
```
"""
function xf_ka_color_sums(n::Integer, seed::Integer=XF_SEED;
                           chunk_size::Int=10000, backend::XFBackend=xf_get_backend(),
                           workgroup::Int=256)
    n_chunks = n ÷ chunk_size
    remainder = n % chunk_size
    
    if remainder != 0
        @warn "n=$n not divisible by chunk_size=$chunk_size, truncating to $(n_chunks * chunk_size)"
    end
    
    sums = zeros(Float32, n_chunks, 3)
    
    kernel! = _xf_color_sum_kernel!(backend, workgroup)
    kernel!(sums, UInt64(seed), chunk_size, ndrange=n_chunks)
    KernelAbstractions.synchronize(backend)
    
    # Final reduction
    total_r = sum(@view sums[:, 1])
    total_g = sum(@view sums[:, 2])
    total_b = sum(@view sums[:, 3])
    
    return (Float64(total_r), Float64(total_g), Float64(total_b))
end

# ═══════════════════════════════════════════════════════════════════════════
# Benchmark utility
# ═══════════════════════════════════════════════════════════════════════════

# Helper for number formatting
function format_number(n::Integer)
    s = string(n)
    parts = String[]
    while length(s) > 3
        push!(parts, s[end-2:end])
        s = s[1:end-3]
    end
    push!(parts, s)
    join(reverse(parts), ",")
end

"""
    xf_ka_benchmark(; n=1_000_000_000, seed=42069, chunk_size=100_000)

Benchmark billion-scale XF color generation with KernelAbstractions.

# Example
```julia
julia> xf_ka_benchmark()
═══════════════════════════════════════════════════════════════════════
  XF.jl KernelAbstractions Color Benchmark
  "If nature is unjust, change nature!"
  Backend: CPU, Threads: 8
═══════════════════════════════════════════════════════════════════════

  1,000,000,000 colors in 0.18 seconds
  Rate: 5,445 million colors/second
  RGB sums: (5.0e8, 5.0e8, 5.0e8)
```
"""
function xf_ka_benchmark(; n::Integer=1_000_000_000, seed::Integer=42069,
                          chunk_size::Int=100_000)
    backend = xf_get_backend()
    
    println("═══════════════════════════════════════════════════════════════════════")
    println("  XF.jl KernelAbstractions Color Benchmark")
    println("  \"If nature is unjust, change nature!\"")
    println("  Backend: $(typeof(backend)), Threads: $(Threads.nthreads())")
    println("═══════════════════════════════════════════════════════════════════════")
    println()
    
    # Warmup
    _ = xf_ka_color_sums(10000, seed; chunk_size=1000, backend=backend)
    
    t = @elapsed sums = xf_ka_color_sums(n, seed; chunk_size=chunk_size, backend=backend)
    rate = n / t / 1e6
    
    println("  $(format_number(n)) colors in $(round(t, digits=2)) seconds")
    println("  Rate: $(format_number(round(Int, rate))) million colors/second")
    println("  RGB sums: ($(round(sums[1], sigdigits=4)), $(round(sums[2], sigdigits=4)), $(round(sums[3], sigdigits=4)))")
    println()
    println("═══════════════════════════════════════════════════════════════════════")
    
    return (time=t, rate=rate, sums=sums)
end

# ═══════════════════════════════════════════════════════════════════════════
# Lisp bindings for REPL
# ═══════════════════════════════════════════════════════════════════════════

# (xf-ka n) or (xf-ka n seed)
xf_ka(n::Int) = xf_ka_colors(n)
xf_ka(n::Int, seed::Int) = xf_ka_colors(n, seed)

# (xf-ka-bench)
xf_ka_bench() = xf_ka_benchmark()
xf_ka_bench(n::Int) = xf_ka_benchmark(n=n)

export xf_ka, xf_ka_bench

# ═══════════════════════════════════════════════════════════════════════════
# SPI (Strong Parallelism Invariance) Verification
# ═══════════════════════════════════════════════════════════════════════════

"""
    xf_ka_verify_spi(n::Int=1000, seed::Integer=XF_SEED; 
                     gpu_backend=nothing, rtol=1e-5)

Verify Strong Parallelism Invariance: same seed produces same colors
regardless of backend (CPU, Metal, CUDA, etc.).

Returns `true` if all invariants hold, throws AssertionError otherwise.

# Tests performed:
1. Sequential vs parallel CPU produce identical colors
2. Different chunk sizes produce identical results  
3. GPU vs CPU produce identical colors (if gpu_backend provided)
4. Hash integrity: xor-reduction matches across backends

# Example
```julia
using Metal
xf_ka_verify_spi(1000, gpu_backend=MetalBackend())
```
"""
function xf_ka_verify_spi(n::Int=1000, seed::Integer=XF_SEED;
                          gpu_backend=nothing, rtol::Float64=1e-5)
    println("═" ^ 60)
    println("SPI VERIFICATION: Strong Parallelism Invariance")
    println("═" ^ 60)
    println("  n = $n, seed = $seed")
    println()
    
    # Reference: CPU sequential
    cpu_colors = zeros(Float32, n, 3)
    for i in 1:n
        r, g, b = hash_color_rgb(UInt64(seed), UInt64(i))
        cpu_colors[i, 1] = r
        cpu_colors[i, 2] = g
        cpu_colors[i, 3] = b
    end
    cpu_hash = reduce(xor, reinterpret(UInt32, vec(cpu_colors)))
    
    println("1. CPU Sequential Reference")
    println("   Hash: $cpu_hash")
    println("   ✓ Generated")
    println()
    
    # Test 2: CPU parallel (KernelAbstractions)
    ka_colors = zeros(Float32, n, 3)
    xf_ka_colors!(ka_colors, seed; backend=CPU())
    ka_hash = reduce(xor, reinterpret(UInt32, vec(ka_colors)))
    
    match = isapprox(cpu_colors, ka_colors; rtol=rtol)
    hash_match = cpu_hash == ka_hash
    
    println("2. CPU Parallel (KernelAbstractions)")
    println("   Hash: $ka_hash")
    println("   Colors match: ", match ? "✓ PASS" : "✗ FAIL")
    println("   Hash match: ", hash_match ? "✓ PASS" : "✗ FAIL")
    @assert match "CPU sequential != CPU parallel"
    @assert hash_match "Hash mismatch: CPU sequential vs parallel"
    println()
    
    # Test 3: Different workgroup sizes
    println("3. Workgroup Size Independence")
    for ws in [32, 64, 128, 256, 512]
        ws_colors = zeros(Float32, n, 3)
        xf_ka_colors!(ws_colors, seed; backend=CPU(), workgroup=ws)
        ws_match = isapprox(cpu_colors, ws_colors; rtol=rtol)
        print("   workgroup=$ws: ")
        println(ws_match ? "✓ PASS" : "✗ FAIL")
        @assert ws_match "Workgroup $ws produced different results"
    end
    println()
    
    # Test 4: GPU backend (if provided)
    if gpu_backend !== nothing
        println("4. GPU Backend: $(typeof(gpu_backend))")
        
        # Allocate GPU array
        gpu_colors_mat = KernelAbstractions.zeros(gpu_backend, Float32, n, 3)
        
        # Run kernel
        kernel! = _xf_colors_kernel!(gpu_backend, 256)
        kernel!(gpu_colors_mat, UInt64(seed), ndrange=n)
        KernelAbstractions.synchronize(gpu_backend)
        
        # Copy back to CPU
        gpu_colors_cpu = Array(gpu_colors_mat)
        gpu_hash = reduce(xor, reinterpret(UInt32, vec(gpu_colors_cpu)))
        
        gpu_match = isapprox(cpu_colors, gpu_colors_cpu; rtol=rtol)
        gpu_hash_match = cpu_hash == gpu_hash
        
        println("   Hash: $gpu_hash")
        println("   Colors match CPU: ", gpu_match ? "✓ PASS" : "✗ FAIL")
        println("   Hash match CPU: ", gpu_hash_match ? "✓ PASS" : "✗ FAIL")
        
        if !gpu_match
            # Find first mismatch for debugging
            for i in 1:min(n, 10)
                cpu_rgb = (cpu_colors[i,1], cpu_colors[i,2], cpu_colors[i,3])
                gpu_rgb = (gpu_colors_cpu[i,1], gpu_colors_cpu[i,2], gpu_colors_cpu[i,3])
                if !isapprox(collect(cpu_rgb), collect(gpu_rgb); rtol=rtol)
                    println("   First mismatch at i=$i:")
                    println("     CPU: $cpu_rgb")
                    println("     GPU: $gpu_rgb")
                    break
                end
            end
        end
        
        @assert gpu_match "GPU != CPU colors"
        @assert gpu_hash_match "GPU != CPU hash"
        println()
    else
        println("4. GPU Backend: (skipped, no gpu_backend provided)")
        println()
    end
    
    # Test 5: Reproducibility (same seed = same output)
    println("5. Reproducibility (5 runs)")
    for run in 1:5
        test_colors = zeros(Float32, n, 3)
        xf_ka_colors!(test_colors, seed; backend=CPU())
        test_hash = reduce(xor, reinterpret(UInt32, vec(test_colors)))
        match = test_hash == cpu_hash
        print("   Run $run: ")
        println(match ? "✓ PASS" : "✗ FAIL")
        @assert match "Run $run produced different hash"
    end
    println()
    
    println("═" ^ 60)
    println("ALL SPI INVARIANTS VERIFIED ✓")
    println("═" ^ 60)
    
    return true
end

export xf_ka_verify_spi
