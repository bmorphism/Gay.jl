# Gay.jl Benchmarks for AirspeedVelocity.jl + ChairmarksForAirspeedVelocity.jl
#
# Run with:
#   benchpkg Gay --add https://github.com/LilithHafner/ChairmarksForAirspeedVelocity.jl
#
# Or locally:
#   julia -e 'using Pkg; Pkg.add("AirspeedVelocity"); Pkg.build("AirspeedVelocity")'
#   ~/.julia/bin/benchpkg --add https://github.com/LilithHafner/ChairmarksForAirspeedVelocity.jl --rev dirty,main

using ChairmarksForAirspeedVelocity
using Gay

const SUITE = BenchmarkGroup()

# ═══════════════════════════════════════════════════════════════════════════
# Hash Color Generation (O(1))
# ═══════════════════════════════════════════════════════════════════════════

SUITE["hash_color"] = BenchmarkGroup()

# Single color hash - should be ~2ns
SUITE["hash_color"]["single"] = @benchmarkable Gay.hash_color(42, Gay.GAY_SEED)

# Batch hash - measure throughput
for n in [100, 1000, 10000, 100000]
    SUITE["hash_color"]["batch_$n"] = @benchmarkable begin
        s = 0.0
        for i in 1:$n
            r, g, b = Gay.hash_color(i, Gay.GAY_SEED)
            s += r
        end
        s
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# Sequential color_at (O(n) - for comparison)
# ═══════════════════════════════════════════════════════════════════════════

SUITE["color_at"] = BenchmarkGroup()

for idx in [1, 10, 100, 1000]
    SUITE["color_at"]["index_$idx"] = @benchmarkable Gay.color_at($idx)
end

# ═══════════════════════════════════════════════════════════════════════════
# Parallel Hash (multi-threaded)
# ═══════════════════════════════════════════════════════════════════════════

SUITE["parallel"] = BenchmarkGroup()

for n in [1_000_000, 10_000_000]
    SUITE["parallel"]["hash_$n"] = @benchmarkable Gay.ka_parallel_hash($n, Gay.GAY_SEED)
end

# ═══════════════════════════════════════════════════════════════════════════
# KernelAbstractions GPU/CPU
# ═══════════════════════════════════════════════════════════════════════════

SUITE["ka_sums"] = BenchmarkGroup()

for n in [1_000_000, 10_000_000, 100_000_000]
    SUITE["ka_sums"]["cpu_$n"] = @benchmarkable begin
        Gay.ka_color_sums($n, Gay.GAY_SEED; backend=KernelAbstractions.CPU())
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# Splitmix64 (raw hash function)
# ═══════════════════════════════════════════════════════════════════════════

SUITE["splitmix64"] = BenchmarkGroup()

SUITE["splitmix64"]["single"] = @benchmarkable Gay.splitmix64(UInt64(12345))

SUITE["splitmix64"]["chain_10"] = @benchmarkable begin
    x = UInt64(12345)
    for _ in 1:10
        x = Gay.splitmix64(x)
    end
    x
end

# ═══════════════════════════════════════════════════════════════════════════
# XOR Fingerprint (SPI verification)
# ═══════════════════════════════════════════════════════════════════════════

SUITE["fingerprint"] = BenchmarkGroup()

# Create test data
const test_colors_small = rand(Float32, 1000, 3)
const test_colors_large = rand(Float32, 100000, 3)

SUITE["fingerprint"]["1k"] = @benchmarkable Gay.xor_fingerprint($test_colors_small)
SUITE["fingerprint"]["100k"] = @benchmarkable Gay.xor_fingerprint($test_colors_large)

# ═══════════════════════════════════════════════════════════════════════════
# Mortal/Immortal Computation
# ═══════════════════════════════════════════════════════════════════════════

SUITE["lifetimes"] = BenchmarkGroup()

SUITE["lifetimes"]["create_mortal"] = @benchmarkable Gay.MortalComputation(1, 1, 100)
SUITE["lifetimes"]["create_immortal"] = @benchmarkable Gay.ImmortalComputation(1)

SUITE["lifetimes"]["mortal_step"] = @benchmarkable begin
    m = Gay.MortalComputation(1, 1, 10)
    while Gay.mortal_step!(m, 1.0) end
    m
end

# ═══════════════════════════════════════════════════════════════════════════
# Interleaver (checkerboard decomposition)
# ═══════════════════════════════════════════════════════════════════════════

SUITE["interleaver"] = BenchmarkGroup()

SUITE["interleaver"]["create"] = @benchmarkable Gay.GayInterleaver(Gay.GAY_SEED, 2)

SUITE["interleaver"]["sublattice_100"] = @benchmarkable begin
    il = Gay.GayInterleaver(Gay.GAY_SEED, 2)
    for i in 1:100
        Gay.gay_sublattice(il, i % 2)
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# S-Expression Coloring
# ═══════════════════════════════════════════════════════════════════════════

SUITE["sexpr"] = BenchmarkGroup()

const test_expr = [:defn, :fib, [:n], [:if, [:<, :n, 2], :n, [:+, [:fib, [:-, :n, 1]], [:fib, [:-, :n, 2]]]]]

SUITE["sexpr"]["magnetize"] = @benchmarkable Gay.gay_magnetized_sexpr($test_expr, 42)
SUITE["sexpr"]["paren_color"] = @benchmarkable Gay.gay_paren_color(42, 3, 5)
