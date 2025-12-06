# Gay.jl Regression Tests for Parallel Tempering Color Conservation
#
# Uses RegressionTests.jl to verify SPI (Strong Parallelism Invariance):
# - color_at(index, seed) is deterministic
# - Different execution orders produce identical results
# - Parallel replicas maintain color conservation
# - XOR parity decomposition is consistent
#
# Run with: julia --project=. -e 'using Pkg; Pkg.test()'
# Or: ]bench (if RegressionTests is in startup.jl)

using RegressionTests
using Chairmarks

# Include Gay.jl core
include("../src/Gay.jl")
using .Gay

# ═══════════════════════════════════════════════════════════════════════════
# Color Conservation Invariants
# ═══════════════════════════════════════════════════════════════════════════

"""
Verify color_at is deterministic: same (index, seed) → same color
"""
function test_color_determinism()
    seed = UInt64(0xDEADBEEF)
    indices = [42, 1000, 51874, 55158, 56610]  # r2 function indices
    
    # Compute colors twice
    colors1 = [color_at(i, SRGB(); seed=seed) for i in indices]
    colors2 = [color_at(i, SRGB(); seed=seed) for i in indices]
    
    # All must match exactly
    all(c1 == c2 for (c1, c2) in zip(colors1, colors2)) ? 1.0 : 0.0
end

"""
Verify parallel tempering: N replicas with different seeds maintain internal consistency
"""
function test_parallel_tempering_conservation()
    seeds = [UInt64(0xDEADBEEF), UInt64(0xCAFEBABE), UInt64(0x12345678), UInt64(0xABCDEF00)]
    indices = collect(1:100)
    
    # Each replica computes colors independently
    replica_colors = [
        [color_at(i, SRGB(); seed=s) for i in indices]
        for s in seeds
    ]
    
    # Verify each replica is internally consistent (recompute and compare)
    consistent = true
    for (s, colors) in zip(seeds, replica_colors)
        recomputed = [color_at(i, SRGB(); seed=s) for i in indices]
        consistent &= all(c1 == c2 for (c1, c2) in zip(colors, recomputed))
    end
    
    consistent ? 1.0 : 0.0
end

"""
Verify XOR parity is consistent for xref coloring
"""
function test_xor_parity_conservation()
    seed = UInt64(0xDEADBEEF)
    
    # Xref pairs (from_addr, to_addr) from r2 analysis
    xrefs = [
        (0x100007ce0, 0x10001caa0),
        (0x10000c958, 0x10001caa0),
        (0x1000050b8, 0x10001dd20),
        (0x10001a574, 0x10001dd20),
    ]
    
    # Compute XOR-based colors twice
    function xor_color(from, to)
        parity = Int((from ⊻ to) & 1)
        idx = Int((from ⊻ to) % 0xFFFF) + parity * 1000
        color_at(idx, SRGB(); seed=seed)
    end
    
    colors1 = [xor_color(f, t) for (f, t) in xrefs]
    colors2 = [xor_color(f, t) for (f, t) in xrefs]
    
    all(c1 == c2 for (c1, c2) in zip(colors1, colors2)) ? 1.0 : 0.0
end

"""
Verify magnetization is conserved across computation orders
"""
function test_magnetization_conservation()
    seed = UInt64(0xDEADBEEF)
    indices = collect(1:50)
    
    # Compute spins (from hue) in forward order
    function compute_magnetization(order)
        spins = Int[]
        for i in order
            c = color_at(i, SRGB(); seed=seed)
            h = convert(HSL, c).h
            push!(spins, h < 180 ? 1 : -1)
        end
        sum(spins) / length(spins)
    end
    
    M_forward = compute_magnetization(indices)
    M_reverse = compute_magnetization(reverse(indices))
    M_shuffled = compute_magnetization(shuffle(indices))
    
    # All magnetizations must be equal (SPI guarantee)
    (M_forward == M_reverse == M_shuffled) ? 1.0 : 0.0
end

"""
Verify checkerboard decomposition conserves total color energy
"""
function test_checkerboard_conservation()
    seed = UInt64(0xDEADBEEF)
    il = GayInterleaver(seed, 2)
    
    # 2D lattice
    Lx, Ly = 4, 4
    
    # Compute colors for even/odd sublattices
    even_colors = RGB[]
    odd_colors = RGB[]
    
    for i in 1:Lx, j in 1:Ly
        parity = (i + j) % 2
        c = gay_sublattice(il, parity)
        if parity == 0
            push!(even_colors, c)
        else
            push!(odd_colors, c)
        end
    end
    
    # Total "energy" (sum of hues as proxy)
    even_energy = sum(convert(HSL, c).h for c in even_colors)
    odd_energy = sum(convert(HSL, c).h for c in odd_colors)
    total_energy = even_energy + odd_energy
    
    # Recompute and verify conservation
    il2 = GayInterleaver(seed, 2)
    even_colors2 = RGB[]
    odd_colors2 = RGB[]
    
    for i in 1:Lx, j in 1:Ly
        parity = (i + j) % 2
        c = gay_sublattice(il2, parity)
        if parity == 0
            push!(even_colors2, c)
        else
            push!(odd_colors2, c)
        end
    end
    
    total_energy2 = sum(convert(HSL, c).h for c in even_colors2) + 
                    sum(convert(HSL, c).h for c in odd_colors2)
    
    (total_energy == total_energy2) ? 1.0 : 0.0
end

# ═══════════════════════════════════════════════════════════════════════════
# Performance Benchmarks with @track
# ═══════════════════════════════════════════════════════════════════════════

# Track color_at performance (should be O(n) for index n)
@track (@b color_at(100, SRGB(); seed=UInt64(0xDEADBEEF)) seconds=0.01).time

# Track interleaver creation performance
@track (@b GayInterleaver(UInt64(0xDEADBEEF), 2) seconds=0.01).time

# Track XOR parity computation performance
function xor_color_bench()
    seed = UInt64(0xDEADBEEF)
    from, to = 0x100007ce0, 0x10001caa0
    idx = Int((from ⊻ to) % 0xFFFF)
    color_at(idx, SRGB(); seed=seed)
end
@track (@b xor_color_bench() seconds=0.01).time

# Track magnetization computation performance
function magnetization_bench()
    seed = UInt64(0xDEADBEEF)
    total = 0
    for i in 1:50
        c = color_at(i, SRGB(); seed=seed)
        h = convert(HSL, c).h
        total += h < 180 ? 1 : -1
    end
    total / 50
end
@track (@b magnetization_bench() seconds=0.01).time

# ═══════════════════════════════════════════════════════════════════════════
# Conservation Invariants as Tracked Values
# ═══════════════════════════════════════════════════════════════════════════

# These should always be 1.0 - any deviation is a regression
@track test_color_determinism()
@track test_parallel_tempering_conservation()
@track test_xor_parity_conservation()
@track test_magnetization_conservation()
@track test_checkerboard_conservation()

# ═══════════════════════════════════════════════════════════════════════════
# Specific Color Fingerprints (Golden Values)
# ═══════════════════════════════════════════════════════════════════════════

# These are the "golden" color values that must never change
# Any change means SPI is broken

function golden_color_fingerprint()
    seed = UInt64(0xDEADBEEF)
    
    # r2 function addresses from self-analysis
    golden_indices = [51874, 55158, 56610, 56402]  # sdb_new, sdb_free, sdb_set, sdb_get
    
    # Compute fingerprint as hash of all RGB values
    fingerprint = UInt64(0)
    for idx in golden_indices
        c = color_at(idx, SRGB(); seed=seed)
        r = round(UInt8, clamp(c.r, 0, 1) * 255)
        g = round(UInt8, clamp(c.g, 0, 1) * 255)
        b = round(UInt8, clamp(c.b, 0, 1) * 255)
        fingerprint = fingerprint ⊻ (UInt64(r) << 48 | UInt64(g) << 32 | UInt64(b) << 16 | UInt64(idx))
    end
    
    Float64(fingerprint)
end

@track golden_color_fingerprint()

println("Gay.jl Regression Tests: Parallel Tempering Color Conservation")
println("================================================================")
println()
println("Tracked invariants:")
println("  • color_at determinism")
println("  • parallel tempering internal consistency")
println("  • XOR parity conservation")
println("  • magnetization order-independence")
println("  • checkerboard energy conservation")
println("  • golden color fingerprint")
println()
println("Run `]test` to verify no regressions.")
