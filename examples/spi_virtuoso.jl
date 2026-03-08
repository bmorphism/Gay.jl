# SPI Virtuoso — Maximum performance color generation in Julia
#
# The Julia side of the zig-syrup race. Increasingly aggressive optimizations:
#   L0: Scalar baseline (hash_color loop)
#   L1: @simd @inbounds vectorized inner loop
#   L2: Manual SIMD via SIMD.jl-style tuple tricks
#   L3: Chunked reduction — ka_color_sums O(1) memory strategy
#   L4: @threads with chunked XOR fingerprint
#   L5: KernelAbstractions SPMD kernel (portable to Metal/CUDA)
#   L6: Fused generate+reduce kernel (the ka_color_sums endgame)
#
# Run: julia --project=. -t auto examples/spi_virtuoso.jl

using Gay
using KernelAbstractions

const SEED = UInt64(42)
const GOLDEN = Gay.GOLDEN
const MIX1 = Gay.MIX1
const MIX2 = Gay.MIX2

# ============================================================================
# Core: inlined SplitMix64 matching zig-syrup exactly
# ============================================================================

@inline function sm64(seed::UInt64, index::UInt64)::UInt64
    z = seed + (GOLDEN * index)
    z = (z ⊻ (z >> 30)) * MIX1
    z = (z ⊻ (z >> 27)) * MIX2
    z ⊻ (z >> 31)
end

@inline function extract_rgb_xor(val::UInt64)::UInt64
    r = (val >> 16) & 0xFF
    g = (val >> 8) & 0xFF
    b = val & 0xFF
    (r << 16) | (g << 8) | b
end

# ============================================================================
# L0: Scalar baseline
# ============================================================================

function l0_scalar(n::Int)::UInt64
    xor = UInt64(0)
    for i in UInt64(0):UInt64(n-1)
        xor ⊻= sm64(SEED, i)
    end
    xor
end

# ============================================================================
# L1: @simd @inbounds — let LLVM auto-vectorize
# Julia's secret weapon: LLVM sees through @simd and emits NEON/AVX
# ============================================================================

function l1_simd(n::Int)::UInt64
    xor = UInt64(0)
    @inbounds @simd for i in UInt64(0):UInt64(n-1)
        xor ⊻= sm64(SEED, i)
    end
    xor
end

# ============================================================================
# L2: Manual 4-wide "SIMD" via tuples
# Julia trick: NTuple{4,UInt64} often compiles to SIMD registers
# ============================================================================

function l2_tuple4(n::Int)::UInt64
    x1, x2, x3, x4 = UInt64(0), UInt64(0), UInt64(0), UInt64(0)
    n4 = n & ~3
    @inbounds for i in 0:4:(n4-1)
        ui = UInt64(i)
        x1 ⊻= sm64(SEED, ui)
        x2 ⊻= sm64(SEED, ui + 1)
        x3 ⊻= sm64(SEED, ui + 2)
        x4 ⊻= sm64(SEED, ui + 3)
    end
    result = x1 ⊻ x2 ⊻ x3 ⊻ x4
    @inbounds for i in UInt64(n4):UInt64(n-1)
        result ⊻= sm64(SEED, i)
    end
    result
end

# ============================================================================
# L3: 8-wide with two independent accumulator chains
# Trick from HPC: hide multiply latency with independent chains
# ============================================================================

function l3_pipeline8(n::Int)::UInt64
    a1, a2, a3, a4 = UInt64(0), UInt64(0), UInt64(0), UInt64(0)
    b1, b2, b3, b4 = UInt64(0), UInt64(0), UInt64(0), UInt64(0)
    n8 = n & ~7
    @inbounds for i in 0:8:(n8-1)
        ui = UInt64(i)
        a1 ⊻= sm64(SEED, ui)
        a2 ⊻= sm64(SEED, ui + 1)
        a3 ⊻= sm64(SEED, ui + 2)
        a4 ⊻= sm64(SEED, ui + 3)
        b1 ⊻= sm64(SEED, ui + 4)
        b2 ⊻= sm64(SEED, ui + 5)
        b3 ⊻= sm64(SEED, ui + 6)
        b4 ⊻= sm64(SEED, ui + 7)
    end
    result = a1 ⊻ a2 ⊻ a3 ⊻ a4 ⊻ b1 ⊻ b2 ⊻ b3 ⊻ b4
    @inbounds for i in UInt64(n8):UInt64(n-1)
        result ⊻= sm64(SEED, i)
    end
    result
end

# ============================================================================
# L4: Chunked reduction — ka_color_sums O(1) memory strategy
# ============================================================================

function l4_chunked(n::Int; chunk_size::Int=10000)::UInt64
    global_xor = UInt64(0)
    n_chunks = n ÷ chunk_size
    for c in 0:(n_chunks-1)
        base = UInt64(c * chunk_size)
        chunk_xor = UInt64(0)
        a1, a2, a3, a4 = UInt64(0), UInt64(0), UInt64(0), UInt64(0)
        b1, b2, b3, b4 = UInt64(0), UInt64(0), UInt64(0), UInt64(0)
        cs8 = chunk_size & ~7
        @inbounds for j in 0:8:(cs8-1)
            idx = base + UInt64(j)
            a1 ⊻= sm64(SEED, idx)
            a2 ⊻= sm64(SEED, idx + 1)
            a3 ⊻= sm64(SEED, idx + 2)
            a4 ⊻= sm64(SEED, idx + 3)
            b1 ⊻= sm64(SEED, idx + 4)
            b2 ⊻= sm64(SEED, idx + 5)
            b3 ⊻= sm64(SEED, idx + 6)
            b4 ⊻= sm64(SEED, idx + 7)
        end
        chunk_xor = a1 ⊻ a2 ⊻ a3 ⊻ a4 ⊻ b1 ⊻ b2 ⊻ b3 ⊻ b4
        global_xor ⊻= chunk_xor
    end
    # remainder
    rem_start = UInt64(n_chunks * chunk_size)
    @inbounds for i in rem_start:UInt64(n-1)
        global_xor ⊻= sm64(SEED, i)
    end
    global_xor
end

# ============================================================================
# L5: @threads parallel with fused RGB XOR
# Julia's Threads.@threads is the simplest multi-core primitive
# ============================================================================

function l5_threaded_fused(n::Int)::UInt64
    nt = Threads.nthreads()
    chunk = n ÷ nt
    partials = zeros(UInt64, nt)
    
    Threads.@threads for tid in 1:nt
        start_idx = UInt64((tid - 1) * chunk)
        end_idx = tid == nt ? UInt64(n) : UInt64(tid * chunk)
        local_xor = UInt64(0)
        a1, a2, a3, a4 = UInt64(0), UInt64(0), UInt64(0), UInt64(0)
        b1, b2, b3, b4 = UInt64(0), UInt64(0), UInt64(0), UInt64(0)
        count = end_idx - start_idx
        n8 = count & ~UInt64(7)
        @inbounds for j in UInt64(0):UInt64(8):UInt64(n8-1)
            idx = start_idx + j
            raw1 = sm64(SEED, idx)
            raw2 = sm64(SEED, idx + 1)
            raw3 = sm64(SEED, idx + 2)
            raw4 = sm64(SEED, idx + 3)
            raw5 = sm64(SEED, idx + 4)
            raw6 = sm64(SEED, idx + 5)
            raw7 = sm64(SEED, idx + 6)
            raw8 = sm64(SEED, idx + 7)
            a1 ⊻= extract_rgb_xor(raw1)
            a2 ⊻= extract_rgb_xor(raw2)
            a3 ⊻= extract_rgb_xor(raw3)
            a4 ⊻= extract_rgb_xor(raw4)
            b1 ⊻= extract_rgb_xor(raw5)
            b2 ⊻= extract_rgb_xor(raw6)
            b3 ⊻= extract_rgb_xor(raw7)
            b4 ⊻= extract_rgb_xor(raw8)
        end
        local_xor = a1 ⊻ a2 ⊻ a3 ⊻ a4 ⊻ b1 ⊻ b2 ⊻ b3 ⊻ b4
        @inbounds for j in UInt64(n8):UInt64(count-1)
            raw = sm64(SEED, start_idx + j)
            local_xor ⊻= extract_rgb_xor(raw)
        end
        partials[tid] = local_xor
    end
    
    reduce(⊻, partials)
end

# ============================================================================
# L6: KernelAbstractions SPMD fused kernel
# The portable GPU path — same code runs on Metal/CUDA/CPU
# ============================================================================

@kernel function _virtuoso_xor_kernel!(results, @Const(seed::UInt64), @Const(chunk_size::Int))
    i = @index(Global)
    base = UInt64((i - 1) * chunk_size)
    
    local x1, x2, x3, x4 = UInt64(0), UInt64(0), UInt64(0), UInt64(0)
    
    for j in UInt64(0):UInt64(4):UInt64(chunk_size - 4)
        idx = base + j
        z1 = seed + (UInt64(0x9e3779b97f4a7c15) * idx)
        z1 = (z1 ⊻ (z1 >> 30)) * UInt64(0xbf58476d1ce4e5b9)
        z1 = (z1 ⊻ (z1 >> 27)) * UInt64(0x94d049bb133111eb)
        z1 = z1 ⊻ (z1 >> 31)
        
        z2 = seed + (UInt64(0x9e3779b97f4a7c15) * (idx + 1))
        z2 = (z2 ⊻ (z2 >> 30)) * UInt64(0xbf58476d1ce4e5b9)
        z2 = (z2 ⊻ (z2 >> 27)) * UInt64(0x94d049bb133111eb)
        z2 = z2 ⊻ (z2 >> 31)
        
        z3 = seed + (UInt64(0x9e3779b97f4a7c15) * (idx + 2))
        z3 = (z3 ⊻ (z3 >> 30)) * UInt64(0xbf58476d1ce4e5b9)
        z3 = (z3 ⊻ (z3 >> 27)) * UInt64(0x94d049bb133111eb)
        z3 = z3 ⊻ (z3 >> 31)
        
        z4 = seed + (UInt64(0x9e3779b97f4a7c15) * (idx + 3))
        z4 = (z4 ⊻ (z4 >> 30)) * UInt64(0xbf58476d1ce4e5b9)
        z4 = (z4 ⊻ (z4 >> 27)) * UInt64(0x94d049bb133111eb)
        z4 = z4 ⊻ (z4 >> 31)
        
        x1 ⊻= ((z1 >> 16) & 0xFF) << 16 | ((z1 >> 8) & 0xFF) << 8 | (z1 & 0xFF)
        x2 ⊻= ((z2 >> 16) & 0xFF) << 16 | ((z2 >> 8) & 0xFF) << 8 | (z2 & 0xFF)
        x3 ⊻= ((z3 >> 16) & 0xFF) << 16 | ((z3 >> 8) & 0xFF) << 8 | (z3 & 0xFF)
        x4 ⊻= ((z4 >> 16) & 0xFF) << 16 | ((z4 >> 8) & 0xFF) << 8 | (z4 & 0xFF)
    end
    
    results[i] = x1 ⊻ x2 ⊻ x3 ⊻ x4
end

function l6_ka_fused(n::Int; chunk_size::Int=10000, backend=CPU())::UInt64
    n_chunks = n ÷ chunk_size
    results = zeros(UInt64, n_chunks)
    
    kernel! = _virtuoso_xor_kernel!(backend, 256)
    kernel!(results, SEED, chunk_size, ndrange=n_chunks)
    KernelAbstractions.synchronize(backend)
    
    reduce(⊻, results)
end

# ============================================================================
# Benchmark harness
# ============================================================================

function timeit(label::String, f, n::Int; warmup_frac=100)
    f(max(1, n ÷ warmup_frac))  # warmup
    GC.gc(false)
    t = @elapsed xor = f(n)
    rate_m = round(Int, n / t / 1e6)
    (label=label, t=t, rate_m=rate_m, xor=xor, n=n)
end

function main()
    nt = Threads.nthreads()
    println()
    println("+======================================================================+")
    println("|        SPI VIRTUOSO — Maximum Performance (Julia side)               |")
    println("|  Tricks: @simd, @inbounds, NTuple, @threads, KernelAbstractions      |")
    println("+======================================================================+")
    println("  Threads: $nt  Seed: $SEED")
    println()

    sizes = [1_000_000, 10_000_000, 100_000_000]
    labels = ["1M", "10M", "100M"]

    # Single-threaded levels
    levels = [
        ("L0", "Scalar baseline                 ", l0_scalar),
        ("L1", "@simd @inbounds (LLVM vectorize)", l1_simd),
        ("L2", "NTuple{4} manual unroll          ", l2_tuple4),
        ("L3", "8-wide pipeline (2 chains)       ", l3_pipeline8),
        ("L4", "Chunked O(1) memory + pipeline   ", l4_chunked),
    ]

    # Print header
    print("  Level  Description                      ")
    for l in labels
        print(lpad("$l", 13))
    end
    println()
    print("  -----  -------------------------------- ")
    for _ in labels
        print(" ------------")
    end
    println()

    ref_xors = zeros(UInt64, length(sizes))

    for (li, (code, desc, f)) in enumerate(levels)
        print("  $code    $desc")
        for (si, n) in enumerate(sizes)
            r = timeit(code, f, n)
            if li == 1
                ref_xors[si] = r.xor
            end
            ok = r.xor == ref_xors[si] ? "" : " MISMATCH"
            print(lpad("$(r.rate_m) M/s$ok", 13))
        end
        println()
    end

    # Multi-threaded levels
    println()
    println("  -- L5: Threads.@threads fused pipeline ($nt threads) --")
    print("  L5    @threads fused RGB+XOR ($nt T)   ")
    for (si, n) in enumerate(sizes)
        r = timeit("L5", l5_threaded_fused, n)
        ok = r.xor == ref_xors[si] ? "" : " !"
        print(lpad("$(r.rate_m) M/s$ok", 13))
    end
    println()

    println()
    println("  -- L6: KernelAbstractions SPMD fused kernel --")
    print("  L6    KA SPMD fused (portable GPU)     ")
    for (si, n) in enumerate(sizes)
        r = timeit("L6", n -> l6_ka_fused(n), n)
        ok = r.xor == ref_xors[si] ? "" : " !"
        print(lpad("$(r.rate_m) M/s$ok", 13))
    end
    println()

    println()
    println("  Tricks applied per level:")
    println("    L0: Scalar loop, 1 sm64/iter")
    println("    L1: @simd @inbounds — LLVM emits NEON/AVX automatically")
    println("    L2: NTuple{4} manual unroll — registers, no vector alloc")
    println("    L3: 8 independent accumulators — hides multiply latency")
    println("    L4: Chunked O(1) memory — matches ka_color_sums strategy")
    println("    L5: Threads.@threads — N-way parallel fused RGB+XOR")
    println("    L6: KernelAbstractions SPMD — portable to Metal/CUDA/AMDGPU")
    println()
    println("  Compare with: zig-syrup/zig-out/bin/spi-virtuoso")
    println()
end

main()
