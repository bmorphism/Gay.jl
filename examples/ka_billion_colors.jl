# Billion color generation using KernelAbstractions.jl
# Portable SPMD kernels that run on CPU (and GPU when available)
#
# Run: julia --project=. -t auto examples/ka_billion_colors.jl

using Gay
using KernelAbstractions
using Chairmarks

println("═══════════════════════════════════════════════════════════════════════")
println("  Gay.jl + KernelAbstractions.jl: Billion Color Benchmark")
println("  Threads: $(Threads.nthreads())")
println("  Backend: CPU (KernelAbstractions)")
println("═══════════════════════════════════════════════════════════════════════")
println()

# KernelAbstractions kernel for color generation
# Uses splittable RNG for Strong Parallelism Invariance (SPI)
@kernel function color_sum_kernel!(sums, seed::UInt64, chunk_size::Int)
    i = @index(Global)
    
    # Each work item processes a chunk
    start_idx = (i - 1) * chunk_size + 1
    end_idx = i * chunk_size
    
    local sr, sg, sb = 0.0f0, 0.0f0, 0.0f0
    
    for idx in start_idx:end_idx
        # Inline the splittable RNG logic for performance
        # This is the core of color_at() - deterministic color from index
        rng = SplittableRandoms.SplittableRandom(seed)
        current = rng
        for _ in 1:idx
            current = SplittableRandoms.split(current)
        end
        
        # Generate LCH color components
        L = Float32(rand(current) * 100.0)
        current = SplittableRandoms.split(current)
        C = Float32(rand(current) * 150.0)
        current = SplittableRandoms.split(current)
        H = Float32(rand(current) * 360.0)
        
        # Simplified RGB approximation (avoiding full color conversion for speed)
        # Real color_at uses proper LCH→RGB but this shows kernel pattern
        sr += L / 100.0f0
        sg += C / 150.0f0
        sb += H / 360.0f0
    end
    
    sums[i, 1] = sr
    sums[i, 2] = sg
    sums[i, 3] = sb
end

# Simpler kernel - just count iterations (baseline)
@kernel function counting_kernel!(results, n_per_thread::Int)
    i = @index(Global)
    local count = 0.0f0
    for j in 1:n_per_thread
        count += Float32(j) * 0.000001f0
    end
    results[i] = count
end

# Kernel that uses pre-split RNG states
@kernel function fast_color_kernel!(sums, base_seed::UInt64, chunk_size::Int)
    i = @index(Global)
    
    start_idx = (i - 1) * chunk_size + 1
    
    local sr, sg, sb = 0.0f0, 0.0f0, 0.0f0
    
    # Simplified deterministic color generation
    for j in 0:(chunk_size-1)
        idx = start_idx + j
        # Use index-based deterministic hashing
        h = xor(base_seed, UInt64(idx) * 0x9e3779b97f4a7c15)
        h = xor(h >> 30, h) * 0xbf58476d1ce4e5b9
        h = xor(h >> 27, h) * 0x94d049bb133111eb
        h = xor(h >> 31, h)
        
        # Extract RGB-like values from hash
        r = Float32((h & 0xFF) / 255.0)
        g = Float32(((h >> 8) & 0xFF) / 255.0)
        b = Float32(((h >> 16) & 0xFF) / 255.0)
        
        sr += r
        sg += g
        sb += b
    end
    
    sums[i, 1] = sr
    sums[i, 2] = sg
    sums[i, 3] = sb
end

# Get the CPU backend
backend = CPU()

# Warmup
println("Warming up...")
warmup_sums = zeros(Float32, 100, 3)
kernel! = fast_color_kernel!(backend, 64)
kernel!(warmup_sums, UInt64(42), 100, ndrange=100)
KernelAbstractions.synchronize(backend)
println()

# Benchmark parameters
const SEED = UInt64(42)

# Test scaling
println("─────────────────────────────────────────────────────────────────────────")
println("  Scaling test: KernelAbstractions CPU backend")
println("─────────────────────────────────────────────────────────────────────────")
println()

for (n, chunk) in [(1_000_000, 1000), (10_000_000, 10000), (100_000_000, 100000)]
    n_chunks = n ÷ chunk
    sums = zeros(Float32, n_chunks, 3)
    
    kernel! = fast_color_kernel!(backend, 256)  # workgroup size 256
    
    print("  n=$(lpad(n, 11)): ")
    
    t = @elapsed begin
        kernel!(sums, SEED, chunk, ndrange=n_chunks)
        KernelAbstractions.synchronize(backend)
    end
    
    total = (sum(sums[:, 1]), sum(sums[:, 2]), sum(sums[:, 3]))
    rate = n / t / 1e6
    
    println("$(round(t, digits=3))s | $(round(rate, digits=1)) M/s | RGB sum: $(round.(total, digits=1))")
end
println()

# Now the big one - 1 billion
println("═══════════════════════════════════════════════════════════════════════")
println("  1 BILLION Colors with KernelAbstractions")
println("═══════════════════════════════════════════════════════════════════════")
println()

const N = 1_000_000_000
const CHUNK_SIZE = 100_000
const N_CHUNKS = N ÷ CHUNK_SIZE

println("  Configuration:")
println("    Total colors:  $(N) (1 billion)")
println("    Chunk size:    $(CHUNK_SIZE)")
println("    Num chunks:    $(N_CHUNKS)")
println("    Workgroup:     256")
println()

sums = zeros(Float32, N_CHUNKS, 3)
kernel! = fast_color_kernel!(backend, 256)

print("  Running... ")
flush(stdout)

t_total = @elapsed begin
    kernel!(sums, SEED, CHUNK_SIZE, ndrange=N_CHUNKS)
    KernelAbstractions.synchronize(backend)
end

total_rgb = (sum(sums[:, 1]), sum(sums[:, 2]), sum(sums[:, 3]))
rate = N / t_total / 1e6

println("Done!")
println()
println("  Results:")
println("    Time:          $(round(t_total, digits=2)) seconds")
println("    Rate:          $(round(rate, digits=1)) million colors/second")
println("    RGB sums:      R=$(round(total_rgb[1], digits=1)) G=$(round(total_rgb[2], digits=1)) B=$(round(total_rgb[3], digits=1))")
println()

# Compare with OhMyThreads
println("─────────────────────────────────────────────────────────────────────────")
println("  Comparison: KernelAbstractions vs OhMyThreads (10M sample)")
println("─────────────────────────────────────────────────────────────────────────")
println()

using OhMyThreads: tmap

n_test = 10_000_000
chunk_test = 10_000
n_chunks_test = n_test ÷ chunk_test

# KA
sums_ka = zeros(Float32, n_chunks_test, 3)
kernel_test! = fast_color_kernel!(backend, 256)
t_ka = @elapsed begin
    kernel_test!(sums_ka, SEED, chunk_test, ndrange=n_chunks_test)
    KernelAbstractions.synchronize(backend)
end
println("  KernelAbstractions: $(round(t_ka, digits=3))s | $(round(n_test/t_ka/1e6, digits=1)) M/s")

# OhMyThreads
function omt_chunk_sum(chunk_idx, seed, chunk_size)
    start_i = (chunk_idx - 1) * chunk_size + 1
    local sr, sg, sb = 0.0, 0.0, 0.0
    for j in 0:(chunk_size-1)
        idx = start_i + j
        h = xor(seed, UInt64(idx) * 0x9e3779b97f4a7c15)
        h = xor(h >> 30, h) * 0xbf58476d1ce4e5b9
        h = xor(h >> 27, h) * 0x94d049bb133111eb
        h = xor(h >> 31, h)
        sr += Float64((h & 0xFF) / 255.0)
        sg += Float64(((h >> 8) & 0xFF) / 255.0)
        sb += Float64(((h >> 16) & 0xFF) / 255.0)
    end
    (sr, sg, sb)
end

t_omt = @elapsed begin
    results = tmap(i -> omt_chunk_sum(i, SEED, chunk_test), 1:n_chunks_test)
end
println("  OhMyThreads:        $(round(t_omt, digits=3))s | $(round(n_test/t_omt/1e6, digits=1)) M/s")
println()

speedup = t_omt / t_ka
println("  KernelAbstractions speedup: $(round(speedup, digits=2))x")
println()

println("═══════════════════════════════════════════════════════════════════════")
println("  ✓ Benchmark complete!")
println("  ✓ KernelAbstractions provides portable SPMD programming model")
println("  ✓ Same code can run on GPU with Metal.jl/CUDA.jl backends")
println("═══════════════════════════════════════════════════════════════════════")
