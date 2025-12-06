# Parallel color generation benchmark using Chairmarks.jl
#
# Run with: julia --project=. -t auto examples/benchmark_parallel.jl

using Gay
using Chairmarks
using OhMyThreads: tmap, tforeach

println("═══════════════════════════════════════════════════════════════════════")
println("  Gay.jl Parallel Color Generation Benchmark")
println("  Threads: $(Threads.nthreads())")
println("═══════════════════════════════════════════════════════════════════════")
println()

# Warmup
gay_seed!(42)
_ = [color_at(i) for i in 1:10]
_ = tmap(i -> color_at(i), 1:10)

# Benchmark parameters
const SEED = 42
const SIZES = [10, 100, 1000, 10_000]

println("Benchmarking sequential vs parallel color generation...")
println()

for n in SIZES
    println("─────────────────────────────────────────────────────────────────────────")
    println("  n = $n colors")
    println("─────────────────────────────────────────────────────────────────────────")
    
    # Sequential: comprehension
    print("  Sequential (comprehension):  ")
    b_seq = @b [color_at(i; seed=SEED) for i in 1:$n]
    println(b_seq)
    
    # Sequential: map
    print("  Sequential (map):            ")
    b_map = @b map(i -> color_at(i; seed=SEED), 1:$n)
    println(b_map)
    
    # Parallel: tmap (OhMyThreads)
    print("  Parallel (tmap):             ")
    b_par = @b tmap(i -> color_at(i; seed=SEED), 1:$n)
    println(b_par)
    
    # Parallel: parallel_palette API
    print("  Parallel (parallel_palette): ")
    b_api = @b parallel_palette($n; seed=SEED)
    println(b_api)
    
    # Calculate speedup
    seq_time = b_seq.time
    par_time = b_par.time
    speedup = seq_time / par_time
    println()
    println("  Speedup: $(round(speedup, digits=2))x")
    println()
end

# SPI verification
println("═══════════════════════════════════════════════════════════════════════")
println("  Strong Parallelism Invariance (SPI) Verification")
println("═══════════════════════════════════════════════════════════════════════")
println()

n = 1000
sequential = [color_at(i; seed=SEED) for i in 1:n]
parallel = tmap(i -> color_at(i; seed=SEED), 1:n)
reversed = tmap(i -> color_at(i; seed=SEED), n:-1:1) |> reverse

println("  Sequential == Parallel: $(sequential == parallel)")
println("  Sequential == Reversed: $(sequential == reversed)")
println("  All identical: $(sequential == parallel == reversed)")
println()

# Color space comparison
println("═══════════════════════════════════════════════════════════════════════")
println("  Color Space Generation Benchmark (n=1000)")
println("═══════════════════════════════════════════════════════════════════════")
println()

for cs in [SRGB(), DisplayP3(), Rec2020()]
    name = string(typeof(cs))
    print("  $name: ")
    b = @b parallel_palette(1000, $cs; seed=SEED)
    println(b)
end
println()

# Palette generation with minimum distance
println("═══════════════════════════════════════════════════════════════════════")
println("  Distinct Palette Generation (with min_distance)")
println("═══════════════════════════════════════════════════════════════════════")
println()

for n in [6, 12, 24]
    print("  next_palette($n): ")
    gay_seed!(SEED)
    b = @b begin
        gay_seed!(SEED)
        next_palette($n, SRGB())
    end
    println(b)
end
println()

println("═══════════════════════════════════════════════════════════════════════")
println("  Benchmark complete!")
println("═══════════════════════════════════════════════════════════════════════")
