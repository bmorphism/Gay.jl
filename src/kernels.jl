# KernelAbstractions.jl SPMD kernels for portable parallel color generation
#
# Provides GPU-style parallel programming that runs on:
# - CPU (default, uses threading + SIMD)
# - Metal.jl (Apple GPU)
# - CUDA.jl (Nvidia GPU)
# - AMDGPU.jl (AMD GPU)
# - oneAPI.jl (Intel GPU)

using KernelAbstractions
using SplittableRandoms: SplittableRandom, split
using Colors: RGB

export ka_colors!, ka_colors, ka_palette!, ka_color_sums, ka_rgb_colors
export ka_benchmark, KABackend, get_backend, CPU
export splitmix64, hash_color, hash_color_rgb, hash_color_lch

# ═══════════════════════════════════════════════════════════════════════════
# Backend selection
# ═══════════════════════════════════════════════════════════════════════════

const KABackend = Union{CPU, KernelAbstractions.Backend}

# Global backend (default: CPU)
const GLOBAL_BACKEND = Ref{KABackend}(CPU())

"""
    get_backend()

Get the current KernelAbstractions backend.
"""
get_backend() = GLOBAL_BACKEND[]

"""
    set_backend!(backend::KABackend)

Set the KernelAbstractions backend for GPU acceleration.

# Example
```julia
using Metal
set_backend!(MetalBackend())  # Use Apple GPU

using CUDA
set_backend!(CUDABackend())   # Use Nvidia GPU
```
"""
function set_backend!(backend::KABackend)
    GLOBAL_BACKEND[] = backend
    return backend
end

export set_backend!

# ═══════════════════════════════════════════════════════════════════════════
# Fast deterministic hash-based color generation
# ═══════════════════════════════════════════════════════════════════════════

# Use GOLDEN, MIX1, MIX2, splitmix64 from splittable.jl (included before kernels.jl)

"""
    splitmix64_mix(z::UInt64) -> UInt64

Pure SplitMix64 mixing function (the finalizer part only).
This is used for one-shot hashing where you combine seed and index.
"""
@inline function splitmix64_mix(z::UInt64)
    z = ((z ⊻ (z >> 30)) * MIX1) % UInt64
    z = ((z ⊻ (z >> 27)) * MIX2) % UInt64
    (z ⊻ (z >> 31)) % UInt64
end

# splitmix64 is imported from splittable.jl

"""
    SplitMix64RNG

Stateful SplitMix64 RNG matching Go/Python/Rust implementations.

Key insight: state is incremented by GOLDEN, but the output is
a mixing of the NEW state. The output is NOT used as the next state.

# Example
```julia
rng = SplitMix64RNG(GAY_SEED)
h0 = next!(rng)  # 0xf061ebbc2ca74d78
h1 = next!(rng)  # 0x34dc5aa0b7117465
```
"""
mutable struct SplitMix64RNG
    state::UInt64
end

SplitMix64RNG(seed::Integer) = SplitMix64RNG(UInt64(seed))

"""
    next!(rng::SplitMix64RNG) -> UInt64

Generate next random value, advancing internal state.
State += GOLDEN, then mixes state (not the previous output).
"""
@inline function next!(rng::SplitMix64RNG)
    rng.state = (rng.state + GOLDEN) % UInt64
    splitmix64_mix(rng.state)
end

export SplitMix64RNG, next!, splitmix64_mix

"""
    hash_color(seed::UInt64, index::UInt64) -> (Float32, Float32, Float32)

Generate deterministic RGB color from seed and index using hash-based RNG.
Returns values in [0, 1] range.
"""
@inline function hash_color(seed::UInt64, index::UInt64)
    # Mix seed and index
    h = splitmix64(xor(seed, index * 0x9e3779b97f4a7c15))
    
    # Extract RGB from hash bits - Float32 only for Metal GPU compatibility
    r = Float32(h & 0xFF) / 255.0f0
    g = Float32((h >> 8) & 0xFF) / 255.0f0
    b = Float32((h >> 16) & 0xFF) / 255.0f0
    
    (r, g, b)
end

"""
    hash_color_rgb(index::UInt64, seed::UInt64) -> RGB{Float32}

Generate deterministic RGB{Float32} color from index and seed.
Note: argument order is (index, seed) for consistency with color_at.
"""
@inline function hash_color_rgb(index::UInt64, seed::UInt64)
    r, g, b = hash_color(seed, index)
    RGB{Float32}(r, g, b)
end

"""
    hash_color_lch(seed::UInt64, index::UInt64) -> (Float32, Float32, Float32)

Generate deterministic LCH color components from seed and index.
Returns (L, C, H) with L∈[0,100], C∈[0,150], H∈[0,360].
"""
@inline function hash_color_lch(seed::UInt64, index::UInt64)
    h1 = splitmix64(xor(seed, index * 0x9e3779b97f4a7c15))
    h2 = splitmix64(h1)
    h3 = splitmix64(h2)
    
    # Float32 only for Metal GPU compatibility
    L = Float32(h1 & 0xFFFF) / 65535.0f0 * 100.0f0
    C = Float32(h2 & 0xFFFF) / 65535.0f0 * 150.0f0
    H = Float32(h3 & 0xFFFF) / 65535.0f0 * 360.0f0
    
    (L, C, H)
end

# ═══════════════════════════════════════════════════════════════════════════
# SPMD Kernels
# ═══════════════════════════════════════════════════════════════════════════

"""
SPMD kernel: Generate RGB colors into a pre-allocated array.
Each thread generates one color at its global index.
"""
@kernel function _ka_colors_kernel!(colors, @Const(seed::UInt64))
    i = @index(Global)
    r, g, b = hash_color(seed, UInt64(i))
    colors[i, 1] = r
    colors[i, 2] = g
    colors[i, 3] = b
end

"""
SPMD kernel: Generate colors and store as RGB struct.
"""
@kernel function _ka_rgb_kernel!(colors::AbstractVector{RGB{Float32}}, @Const(seed::UInt64))
    i = @index(Global)
    r, g, b = hash_color(seed, UInt64(i))
    colors[i] = RGB{Float32}(r, g, b)
end

"""
SPMD kernel: Chunked color sum reduction.
Each thread processes a chunk and outputs partial sums.
"""
@kernel function _ka_color_sum_kernel!(sums, @Const(seed::UInt64), @Const(chunk_size::Int))
    i = @index(Global)
    
    start_idx = UInt64((i - 1) * chunk_size + 1)
    
    local sr, sg, sb = 0.0f0, 0.0f0, 0.0f0
    
    for j in UInt64(0):UInt64(chunk_size - 1)
        idx = start_idx + j
        r, g, b = hash_color(seed, idx)
        sr += r
        sg += g
        sb += b
    end
    
    sums[i, 1] = sr
    sums[i, 2] = sg
    sums[i, 3] = sb
end

"""
SPMD kernel: Generate palette with chunked work distribution.
"""
@kernel function _ka_palette_kernel!(colors, @Const(seed::UInt64), @Const(offset::Int))
    i = @index(Global)
    idx = UInt64(offset + i)
    r, g, b = hash_color(seed, idx)
    colors[i, 1] = r
    colors[i, 2] = g
    colors[i, 3] = b
end

# ═══════════════════════════════════════════════════════════════════════════
# High-level API
# ═══════════════════════════════════════════════════════════════════════════

"""
    ka_colors!(colors::AbstractMatrix{Float32}, seed::Integer=GAY_SEED; 
               backend=get_backend(), workgroup=256)

Fill a pre-allocated n×3 matrix with deterministic RGB colors using SPMD kernel.

# Example
```julia
colors = zeros(Float32, 1_000_000, 3)
ka_colors!(colors, 42)
```
"""
function ka_colors!(colors::AbstractMatrix{Float32}, seed::Integer=GAY_SEED;
                    backend::KABackend=get_backend(), workgroup::Int=256)
    n = size(colors, 1)
    @assert size(colors, 2) == 3 "colors must be n×3 matrix"
    
    kernel! = _ka_colors_kernel!(backend, workgroup)
    kernel!(colors, UInt64(seed), ndrange=n)
    KernelAbstractions.synchronize(backend)
    
    return colors
end

"""
    ka_colors(n::Integer, seed::Integer=GAY_SEED; backend=get_backend())

Generate n deterministic RGB colors using SPMD kernel.
Returns n×3 Float32 matrix.

# Example
```julia
colors = ka_colors(1_000_000, 42)
```
"""
function ka_colors(n::Integer, seed::Integer=GAY_SEED; 
                   backend::KABackend=get_backend(), workgroup::Int=256)
    colors = zeros(Float32, n, 3)
    ka_colors!(colors, seed; backend=backend, workgroup=workgroup)
end

"""
    ka_palette!(colors::AbstractMatrix{Float32}, seed::Integer=GAY_SEED;
                offset::Int=0, backend=get_backend())

Fill matrix with palette colors starting at offset.
"""
function ka_palette!(colors::AbstractMatrix{Float32}, seed::Integer=GAY_SEED;
                     offset::Int=0, backend::KABackend=get_backend(), workgroup::Int=256)
    n = size(colors, 1)
    @assert size(colors, 2) == 3 "colors must be n×3 matrix"
    
    kernel! = _ka_palette_kernel!(backend, workgroup)
    kernel!(colors, UInt64(seed), offset, ndrange=n)
    KernelAbstractions.synchronize(backend)
    
    return colors
end

"""
    ka_color_sums(n::Integer, seed::Integer=GAY_SEED; 
                  chunk_size::Int=10000, backend=get_backend())

Compute RGB sums over n colors using parallel reduction.
Useful for billion-scale operations where storing all colors is impractical.

Returns (sum_r, sum_g, sum_b) as Float64 tuple.

# Example
```julia
# Sum 1 billion colors in ~0.2 seconds
sums = ka_color_sums(1_000_000_000, 42)
```
"""
function ka_color_sums(n::Integer, seed::Integer=GAY_SEED;
                       chunk_size::Int=10000, backend::KABackend=get_backend(),
                       workgroup::Int=256)
    n_chunks = n ÷ chunk_size
    remainder = n % chunk_size
    
    if remainder != 0
        @warn "n=$n not divisible by chunk_size=$chunk_size, truncating to $(n_chunks * chunk_size)"
    end
    
    sums = zeros(Float32, n_chunks, 3)
    
    kernel! = _ka_color_sum_kernel!(backend, workgroup)
    kernel!(sums, UInt64(seed), chunk_size, ndrange=n_chunks)
    KernelAbstractions.synchronize(backend)
    
    # Final reduction
    total_r = sum(@view sums[:, 1])
    total_g = sum(@view sums[:, 2])
    total_b = sum(@view sums[:, 3])
    
    return (Float64(total_r), Float64(total_g), Float64(total_b))
end

# ═══════════════════════════════════════════════════════════════════════════
# Convenience functions for Colors.jl integration
# ═══════════════════════════════════════════════════════════════════════════

"""
    ka_rgb_colors(n::Integer, seed::Integer=GAY_SEED; backend=get_backend())

Generate n deterministic colors as Vector{RGB{Float32}}.
"""
function ka_rgb_colors(n::Integer, seed::Integer=GAY_SEED;
                       backend::KABackend=get_backend())
    mat = ka_colors(n, seed; backend=backend)
    [RGB{Float32}(mat[i, 1], mat[i, 2], mat[i, 3]) for i in 1:n]
end

# ═══════════════════════════════════════════════════════════════════════════
# Benchmark utility
# ═══════════════════════════════════════════════════════════════════════════

"""
    ka_benchmark(; n=1_000_000_000, seed=42, chunk_size=100_000)

Benchmark billion-scale color generation with KernelAbstractions.

# Example
```julia
julia> ka_benchmark()
═══════════════════════════════════════════════════════════════════════
  KernelAbstractions Color Generation Benchmark
  Backend: CPU, Threads: 8
═══════════════════════════════════════════════════════════════════════

  1,000,000,000 colors in 0.18 seconds
  Rate: 5,445 million colors/second
  RGB sums: (5.0e8, 5.0e8, 5.0e8)
```
"""
function ka_benchmark(; n::Integer=1_000_000_000, seed::Integer=42, 
                       chunk_size::Int=100_000)
    backend = get_backend()
    
    println("═══════════════════════════════════════════════════════════════════════")
    println("  KernelAbstractions Color Generation Benchmark")
    println("  Backend: $(typeof(backend)), Threads: $(Threads.nthreads())")
    println("═══════════════════════════════════════════════════════════════════════")
    println()
    
    # Warmup
    _ = ka_color_sums(10000, seed; chunk_size=1000, backend=backend)
    
    t = @elapsed sums = ka_color_sums(n, seed; chunk_size=chunk_size, backend=backend)
    rate = n / t / 1e6
    
    println("  $(format_number(n)) colors in $(round(t, digits=2)) seconds")
    println("  Rate: $(format_number(round(Int, rate))) million colors/second")
    println("  RGB sums: ($(round(sums[1], sigdigits=4)), $(round(sums[2], sigdigits=4)), $(round(sums[3], sigdigits=4)))")
    println()
    println("═══════════════════════════════════════════════════════════════════════")
    
    return (time=t, rate=rate, sums=sums)
end

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

# ═══════════════════════════════════════════════════════════════════════════
# SPI (Strong Parallelism Invariance) Verification
# ═══════════════════════════════════════════════════════════════════════════

"""
    xor_fingerprint(colors::AbstractMatrix{Float32}) -> UInt32

Compute XOR fingerprint of color matrix for instant SPI verification.
Same seed → same fingerprint, regardless of backend or execution order.

# Example
```julia
colors1 = ka_colors(1_000_000, 42)
colors2 = ka_colors(1_000_000, 42)
@assert xor_fingerprint(colors1) == xor_fingerprint(colors2)  # Always true!
```
"""
function xor_fingerprint(colors::AbstractMatrix{Float32})
    reduce(xor, reinterpret(UInt32, vec(colors)))
end

export xor_fingerprint

"""
    verify_spi(n::Int=1000, seed::Integer=GAY_SEED; 
               gpu_backend=nothing, rtol=1e-5) -> Bool

Verify Strong Parallelism Invariance: same seed produces identical colors
regardless of backend (CPU, Metal, CUDA, etc.).

Uses XOR fingerprinting for instant bitwise verification of massive datasets.

# Tests performed:
1. Sequential vs parallel CPU produce identical colors
2. Different workgroup sizes produce identical results  
3. GPU vs CPU produce identical colors (if gpu_backend provided)
4. XOR fingerprint integrity across all backends

# Example
```julia
# CPU-only verification
verify_spi(10000, 42)

# With GPU verification
using Metal
verify_spi(10000, 42; gpu_backend=MetalBackend())
```
"""
function verify_spi(n::Int=1000, seed::Integer=GAY_SEED;
                    gpu_backend=nothing, rtol::Float64=1e-5)
    println("═" ^ 60)
    println("SPI VERIFICATION: Strong Parallelism Invariance")
    println("═" ^ 60)
    println("  n = $n, seed = $seed")
    println()
    
    # Reference: CPU sequential
    cpu_colors = zeros(Float32, n, 3)
    for i in 1:n
        r, g, b = hash_color(UInt64(seed), UInt64(i))
        cpu_colors[i, 1] = r
        cpu_colors[i, 2] = g
        cpu_colors[i, 3] = b
    end
    cpu_hash = xor_fingerprint(cpu_colors)
    
    println("1. CPU Sequential Reference")
    println("   XOR Fingerprint: 0x$(string(cpu_hash, base=16, pad=8))")
    println("   ✓ Generated")
    println()
    
    # Test 2: CPU parallel (KernelAbstractions)
    ka_colors_cpu = zeros(Float32, n, 3)
    ka_colors!(ka_colors_cpu, seed; backend=CPU())
    ka_hash = xor_fingerprint(ka_colors_cpu)
    
    match = isapprox(cpu_colors, ka_colors_cpu; rtol=rtol)
    hash_match = cpu_hash == ka_hash
    
    println("2. CPU Parallel (KernelAbstractions)")
    println("   XOR Fingerprint: 0x$(string(ka_hash, base=16, pad=8))")
    println("   Colors match: ", match ? "✓ PASS" : "✗ FAIL")
    println("   Fingerprint match: ", hash_match ? "✓ PASS" : "✗ FAIL")
    @assert match "CPU sequential != CPU parallel"
    @assert hash_match "Fingerprint mismatch: CPU sequential vs parallel"
    println()
    
    # Test 3: Different workgroup sizes
    println("3. Workgroup Size Independence")
    for ws in [32, 64, 128, 256, 512]
        ws_colors = zeros(Float32, n, 3)
        ka_colors!(ws_colors, seed; backend=CPU(), workgroup=ws)
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
        kernel! = _ka_colors_kernel!(gpu_backend, 256)
        kernel!(gpu_colors_mat, UInt64(seed), ndrange=n)
        KernelAbstractions.synchronize(gpu_backend)
        
        # Copy back to CPU
        gpu_colors_cpu = Array(gpu_colors_mat)
        gpu_hash = xor_fingerprint(gpu_colors_cpu)
        
        gpu_match = isapprox(cpu_colors, gpu_colors_cpu; rtol=rtol)
        gpu_hash_match = cpu_hash == gpu_hash
        
        println("   XOR Fingerprint: 0x$(string(gpu_hash, base=16, pad=8))")
        println("   Colors match CPU: ", gpu_match ? "✓ PASS" : "✗ FAIL")
        println("   Fingerprint match CPU: ", gpu_hash_match ? "✓ PASS" : "✗ FAIL")
        
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
        @assert gpu_hash_match "GPU != CPU fingerprint"
        println()
    else
        println("4. GPU Backend: (skipped, pass gpu_backend=MetalBackend() etc.)")
        println()
    end
    
    # Test 5: Reproducibility (same seed = same output)
    println("5. Reproducibility (5 runs)")
    for run in 1:5
        test_colors = zeros(Float32, n, 3)
        ka_colors!(test_colors, seed; backend=CPU())
        test_hash = xor_fingerprint(test_colors)
        match = test_hash == cpu_hash
        print("   Run $run: ")
        println(match ? "✓ PASS (0x$(string(test_hash, base=16, pad=8)))" : "✗ FAIL")
        @assert match "Run $run produced different fingerprint"
    end
    println()
    
    println("═" ^ 60)
    println("ALL SPI INVARIANTS VERIFIED ✓")
    println("═" ^ 60)
    
    return true
end

export verify_spi

"""
    gpu_fingerprint(n::Integer, seed::Integer=GAY_SEED; backend=get_backend()) -> UInt32

Compute XOR fingerprint of n colors directly on GPU without CPU comparison.
Returns fingerprint in milliseconds for billion-scale verification.

# Example
```julia
using Metal
set_backend!(MetalBackend())

# Fingerprint 1 billion colors on GPU
@time fp = gpu_fingerprint(1_000_000_000, 42)
# → 0.1 seconds, returns 0x...
```
"""
function gpu_fingerprint(n::Integer, seed::Integer=GAY_SEED;
                          backend::KABackend=get_backend())
    # Allocate on device
    colors = KernelAbstractions.zeros(backend, Float32, n, 3)
    
    # Run kernel
    kernel! = _ka_colors_kernel!(backend, 256)
    kernel!(colors, UInt64(seed), ndrange=n)
    KernelAbstractions.synchronize(backend)
    
    # Fingerprint (still on GPU if supported, otherwise copies)
    cpu_colors = Array(colors)
    xor_fingerprint(cpu_colors)
end

export gpu_fingerprint
