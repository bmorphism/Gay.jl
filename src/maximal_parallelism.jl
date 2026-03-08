# maximal_parallelism.jl - Measure effective parallelism of Gay.jl color generation
#
# Answers the question: given a KernelAbstractions backend (the matter),
# how close does Gay.jl's color generation (the pattern) get to the
# theoretical maximum parallelism?
#
# Metrics (following KernelAbstractions.jl's NDRange/Workgroup model):
#   - Speedup:        T_seq / T_par
#   - Efficiency:     Speedup / P (where P = available parallel units)
#   - Occupancy:      fraction of workgroups actively computing
#   - SPI overhead:   cost of determinism guarantee vs non-deterministic
#   - Scaling curve:  speedup as function of N (weak scaling)
#   - Workgroup scan: optimal workgroup size for this backend

module MaximalParallelism

using KernelAbstractions
using Colors: RGB

const GOLDEN = 0x9e3779b97f4a7c15
const MIX1 = 0xbf58476d1ce4e5b9
const MIX2 = 0x94d049bb133111eb

@inline function splitmix64(x::UInt64)::UInt64
    x += GOLDEN
    x = ((x ⊻ (x >> 30)) * MIX1) % UInt64
    x = ((x ⊻ (x >> 27)) * MIX2) % UInt64
    (x ⊻ (x >> 31)) % UInt64
end

@inline function hash_color(seed::UInt64, index::UInt64)
    h = splitmix64(xor(seed, index * GOLDEN))
    r = Float32(h & 0xFF) / 255.0f0
    g = Float32((h >> 8) & 0xFF) / 255.0f0
    b = Float32((h >> 16) & 0xFF) / 255.0f0
    (r, g, b)
end

# ═══════════════════════════════════════════════════════════════════════════
# Kernels
# ═══════════════════════════════════════════════════════════════════════════

@kernel function _color_kernel!(colors, @Const(seed::UInt64))
    i = @index(Global)
    r, g, b = hash_color(seed, UInt64(i))
    colors[i, 1] = r
    colors[i, 2] = g
    colors[i, 3] = b
end

@kernel function _xor_reduce_kernel!(out, colors)
    i = @index(Global)
    x = reinterpret(UInt32, colors[i, 1])
    x = xor(x, reinterpret(UInt32, colors[i, 2]))
    x = xor(x, reinterpret(UInt32, colors[i, 3]))
    out[i] = x
end

# ═══════════════════════════════════════════════════════════════════════════
# Sequential baseline
# ═══════════════════════════════════════════════════════════════════════════

function sequential_generate(n::Int, seed::UInt64)
    colors = zeros(Float32, n, 3)
    for i in 1:n
        r, g, b = hash_color(seed, UInt64(i))
        colors[i, 1] = r
        colors[i, 2] = g
        colors[i, 3] = b
    end
    colors
end

function sequential_fingerprint(colors::AbstractMatrix{Float32})
    fp = UInt32(0)
    for i in axes(colors, 1)
        fp = xor(fp, reinterpret(UInt32, colors[i, 1]))
        fp = xor(fp, reinterpret(UInt32, colors[i, 2]))
        fp = xor(fp, reinterpret(UInt32, colors[i, 3]))
    end
    fp
end

# ═══════════════════════════════════════════════════════════════════════════
# Workgroup size scan
# ═══════════════════════════════════════════════════════════════════════════

"""
    WorkgroupResult

Timing result for a specific workgroup size.
"""
struct WorkgroupResult
    workgroup_size::Int
    time_seconds::Float64
    throughput::Float64      # colors/second
    spi_verified::Bool
end

"""
    scan_workgroups(n, seed; backend=CPU(), sizes=[32,64,128,256,512,1024], trials=3)

Find the optimal workgroup size for this backend by benchmarking each.
Returns vector of WorkgroupResult sorted by throughput (best first).
"""
function scan_workgroups(n::Int, seed::UInt64 = UInt64(69);
                          backend = CPU(),
                          sizes::Vector{Int} = [32, 64, 128, 256, 512, 1024],
                          trials::Int = 3)
    # Reference fingerprint
    ref = sequential_generate(min(n, 10000), seed)
    ref_fp = sequential_fingerprint(ref)

    results = WorkgroupResult[]

    for ws in sizes
        # Warmup
        warmup = zeros(Float32, min(1000, n), 3)
        try
            k! = _color_kernel!(backend, ws)
            k!(warmup, seed, ndrange = min(1000, n))
            KernelAbstractions.synchronize(backend)
        catch
            continue  # skip invalid workgroup sizes
        end

        # Timed trials
        times = Float64[]
        spi_ok = true
        for _ in 1:trials
            colors = zeros(Float32, n, 3)
            t = @elapsed begin
                k! = _color_kernel!(backend, ws)
                k!(colors, seed, ndrange = n)
                KernelAbstractions.synchronize(backend)
            end
            push!(times, t)

            # SPI check on subset
            if n >= 10000
                sub = colors[1:10000, :]
                fp = sequential_fingerprint(sub)
                if fp != ref_fp
                    spi_ok = false
                end
            end
        end

        best_t = minimum(times)
        throughput = n / best_t
        push!(results, WorkgroupResult(ws, best_t, throughput, spi_ok))
    end

    sort!(results, by = r -> -r.throughput)
    results
end

# ═══════════════════════════════════════════════════════════════════════════
# Weak scaling curve
# ═══════════════════════════════════════════════════════════════════════════

"""
    ScalingPoint

One point on the weak-scaling curve.
"""
struct ScalingPoint
    n::Int
    t_sequential::Float64
    t_parallel::Float64
    speedup::Float64
    efficiency::Float64
end

"""
    scaling_curve(seed; backend=CPU(), workgroup=256, 
                  sizes=[1000, 10_000, 100_000, 1_000_000])

Measure speedup as a function of problem size (weak scaling).
Shows where parallelism overhead is amortized.
"""
function scaling_curve(seed::UInt64 = UInt64(69);
                        backend = CPU(),
                        workgroup::Int = 256,
                        sizes::Vector{Int} = [1_000, 10_000, 100_000, 1_000_000],
                        trials::Int = 3)
    p = Float64(Threads.nthreads())
    points = ScalingPoint[]

    for n in sizes
        # Sequential
        seq_times = [(@elapsed sequential_generate(n, seed)) for _ in 1:trials]
        t_seq = minimum(seq_times)

        # Parallel
        par_times = Float64[]
        for _ in 1:trials
            colors = zeros(Float32, n, 3)
            t = @elapsed begin
                k! = _color_kernel!(backend, workgroup)
                k!(colors, seed, ndrange = n)
                KernelAbstractions.synchronize(backend)
            end
            push!(par_times, t)
        end
        t_par = minimum(par_times)

        speedup = t_seq / max(t_par, 1e-12)
        efficiency = speedup / p

        push!(points, ScalingPoint(n, t_seq, t_par, speedup, efficiency))
    end

    points
end

# ═══════════════════════════════════════════════════════════════════════════
# SPI overhead measurement
# ═══════════════════════════════════════════════════════════════════════════

"""
    SPIOverhead

Cost of Strong Parallelism Invariance (determinism guarantee).
Compares hash-based (deterministic, SPI) vs Julia's rand (non-deterministic).
"""
struct SPIOverhead
    n::Int
    t_spi::Float64        # hash_color (deterministic)
    t_rand::Float64       # rand() (non-deterministic)
    overhead_ratio::Float64
    spi_throughput::Float64
    rand_throughput::Float64
end

"""
    measure_spi_overhead(n, seed; trials=3)

Quantify the cost of determinism: how much slower is SPI-guaranteed
color generation vs plain random?
"""
function measure_spi_overhead(n::Int = 1_000_000, seed::UInt64 = UInt64(69);
                               trials::Int = 3)
    # SPI: hash-based deterministic
    spi_times = Float64[]
    for _ in 1:trials
        colors = zeros(Float32, n, 3)
        t = @elapsed for i in 1:n
            r, g, b = hash_color(seed, UInt64(i))
            colors[i, 1] = r
            colors[i, 2] = g
            colors[i, 3] = b
        end
        push!(spi_times, t)
    end

    # Non-SPI: plain random
    rand_times = Float64[]
    for _ in 1:trials
        colors = zeros(Float32, n, 3)
        t = @elapsed for i in 1:n
            colors[i, 1] = rand(Float32)
            colors[i, 2] = rand(Float32)
            colors[i, 3] = rand(Float32)
        end
        push!(rand_times, t)
    end

    t_spi = minimum(spi_times)
    t_rand = minimum(rand_times)
    ratio = t_spi / max(t_rand, 1e-12)

    SPIOverhead(n, t_spi, t_rand, ratio, n / t_spi, n / t_rand)
end

# ═══════════════════════════════════════════════════════════════════════════
# Full parallelism audit
# ═══════════════════════════════════════════════════════════════════════════

"""
    ParallelismAudit

Complete effective parallelism measurement for Gay.jl on current hardware.
"""
struct ParallelismAudit
    backend_label::String
    n_threads::Int
    workgroup_results::Vector{WorkgroupResult}
    scaling_points::Vector{ScalingPoint}
    spi_overhead::SPIOverhead
    optimal_workgroup::Int
    peak_speedup::Float64
    peak_efficiency::Float64
    peak_throughput::Float64
end

"""
    full_audit(; seed=69, n_bench=1_000_000, backend=CPU())

Complete parallelism audit: workgroup scan + scaling curve + SPI overhead.

This is the definitive answer to "how parallel is Gay.jl on this hardware?"
in the Spivak/Libkind sense: measuring how effectively the pattern
(deterministic color generation) runs on the matter (your CPU/GPU).
"""
function full_audit(; seed::UInt64 = UInt64(69),
                      n_bench::Int = 1_000_000,
                      backend = CPU())
    label = "$(typeof(backend))($(Threads.nthreads()) threads)"

    println("═══════════════════════════════════════════════════════════════")
    println("  GAY.JL EFFECTIVE PARALLELISM AUDIT")
    println("  Pattern: deterministic color generation (SplitMix64)")
    println("  Matter:  $label")
    println("═══════════════════════════════════════════════════════════════")
    println()

    # 1. Workgroup scan
    println("▸ Scanning workgroup sizes...")
    wg_results = scan_workgroups(n_bench, seed; backend = backend)
    optimal_ws = isempty(wg_results) ? 256 : wg_results[1].workgroup_size
    println("  Optimal workgroup: $optimal_ws")
    for r in wg_results
        spi = r.spi_verified ? "✓" : "✗"
        println("    ws=$(lpad(r.workgroup_size, 5))  $(round(r.throughput / 1e6, digits=1)) M/s  SPI=$spi")
    end
    println()

    # 2. Scaling curve
    println("▸ Measuring scaling curve...")
    sizes = [1_000, 10_000, 100_000, n_bench]
    if n_bench >= 10_000_000
        push!(sizes, n_bench)
    end
    unique!(sort!(sizes))
    sc_points = scaling_curve(seed; backend = backend, workgroup = optimal_ws, sizes = sizes)
    peak_speedup = 0.0
    peak_eff = 0.0
    peak_tp = 0.0
    for pt in sc_points
        tp = pt.n / max(pt.t_parallel, 1e-12)
        println("    N=$(lpad(pt.n, 10))  speedup=$(round(pt.speedup, digits=2))×  eff=$(round(pt.efficiency * 100, digits=1))%  $(round(tp / 1e6, digits=1)) M/s")
        if pt.speedup > peak_speedup
            peak_speedup = pt.speedup
        end
        if pt.efficiency > peak_eff
            peak_eff = pt.efficiency
        end
        if tp > peak_tp
            peak_tp = tp
        end
    end
    println()

    # 3. SPI overhead
    println("▸ Measuring SPI (determinism) overhead...")
    spi_oh = measure_spi_overhead(min(n_bench, 1_000_000), seed)
    println("    SPI throughput:  $(round(spi_oh.spi_throughput / 1e6, digits=1)) M/s")
    println("    rand throughput: $(round(spi_oh.rand_throughput / 1e6, digits=1)) M/s")
    println("    Overhead ratio:  $(round(spi_oh.overhead_ratio, digits=2))×")
    println()

    # Summary
    println("═══════════════════════════════════════════════════════════════")
    println("  SUMMARY")
    println("═══════════════════════════════════════════════════════════════")
    println("  Peak speedup:     $(round(peak_speedup, digits=2))×")
    println("  Peak efficiency:  $(round(peak_eff * 100, digits=1))%")
    println("  Peak throughput:  $(round(peak_tp / 1e6, digits=1)) M colors/s")
    println("  SPI cost:         $(round(spi_oh.overhead_ratio, digits=2))× vs rand()")
    println("  Optimal WG size:  $optimal_ws")
    println("═══════════════════════════════════════════════════════════════")

    ParallelismAudit(
        label, Threads.nthreads(),
        wg_results, sc_points, spi_oh,
        optimal_ws, peak_speedup, peak_eff, peak_tp
    )
end

export WorkgroupResult, ScalingPoint, SPIOverhead, ParallelismAudit
export scan_workgroups, scaling_curve, measure_spi_overhead, full_audit

end # module MaximalParallelism
