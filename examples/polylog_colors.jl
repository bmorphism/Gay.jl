# Polylogarithm Values → Colors
# Li_n(1/b) values → deterministic color mapping
#
# Uses the :polylog continuation branch from bbp_pi.jl
#
# The math: Li_n(x) = Σ_{k=1}^∞ x^k/k^n
# For x = 1/b (integer b), this has natural BBP structure:
#   Li_n(1/b) = Σ 1/(b^k · k^n)
#
# Each (n, b) pair → a color from the splittable stream.

using Gay
using Colors
using SplittableRandoms: SplittableRandom, split

include("bbp_pi.jl")  # For continuation_point, branch_seed

# ═══════════════════════════════════════════════════════════════════════════
# Polylogarithm Computation
# ═══════════════════════════════════════════════════════════════════════════

"""
    polylog(n::Int, x::Float64; terms=1000)

Compute Li_n(x) = Σ_{k=1}^∞ x^k/k^n
"""
function polylog(n::Int, x::Float64; terms::Int=1000)
    s = 0.0
    for k in 1:terms
        term = x^k / k^n
        abs(term) < 1e-16 && break
        s += term
    end
    return s
end

"""
    polylog_bbp(n::Int, b::Int; terms=1000)

Compute Li_n(1/b) using explicit BBP form:
  Li_n(1/b) = Σ_{k=1}^∞ 1/(b^k · k^n)
"""
function polylog_bbp(n::Int, b::Int; terms::Int=1000)
    s = 0.0
    for k in 1:terms
        term = 1.0 / (b^k * k^n)
        term < 1e-16 && break
        s += term
    end
    return s
end

# ═══════════════════════════════════════════════════════════════════════════
# Polylogarithm Table (precomputed values)
# ═══════════════════════════════════════════════════════════════════════════

struct PolylogValue
    n::Int              # order (Li_n)
    b::Int              # base (1/b)
    value::Float64      # Li_n(1/b)
    name::String        # descriptive name
end

function polylog_table()
    values = PolylogValue[]
    
    # Li_1(1/b) = -log(1 - 1/b) = -log((b-1)/b) = log(b/(b-1))
    for b in 2:8
        v = polylog_bbp(1, b)
        push!(values, PolylogValue(1, b, v, "Li₁(1/$b) = log($b/$(b-1))"))
    end
    
    # Li_2(1/b) - dilogarithm
    for b in 2:8
        v = polylog_bbp(2, b)
        push!(values, PolylogValue(2, b, v, "Li₂(1/$b)"))
    end
    
    # Li_3(1/b) - trilogarithm (Adegoke 2010: ternary digit extraction)
    for b in [3, 9, 27]
        v = polylog_bbp(3, b)
        push!(values, PolylogValue(3, b, v, "Li₃(1/$b) [ternary BBP]"))
    end
    
    # Special values
    push!(values, PolylogValue(2, 2, polylog_bbp(2, 2), "Li₂(1/2) = π²/12 - log²(2)/2"))
    
    return values
end

# ═══════════════════════════════════════════════════════════════════════════
# (n, b) → Color Mapping (uses :polylog branch)
# ═══════════════════════════════════════════════════════════════════════════

const POLYLOG_SEED = 314159

"""
    polylog_color(n::Int, b::Int; seed=POLYLOG_SEED)

Get a deterministic color for polylogarithm Li_n(1/b).
Uses the :polylog continuation branch.
"""
function polylog_color(n::Int, b::Int; seed::Integer=POLYLOG_SEED)
    pseed = branch_seed(seed, :polylog)
    
    # Index: n * 1000 + b ensures unique mapping
    idx = n * 1000 + b
    
    return color_at(idx, Rec2020(); seed=pseed)
end

"""
    polylog_value_color(pv::PolylogValue; seed=POLYLOG_SEED)

Get color for a PolylogValue struct.
"""
function polylog_value_color(pv::PolylogValue; seed::Integer=POLYLOG_SEED)
    return polylog_color(pv.n, pv.b; seed=seed)
end

"""
    polylog_gradient(n::Int, b_range::UnitRange; seed=POLYLOG_SEED)

Generate a gradient of colors across bases for fixed order n.
"""
function polylog_gradient(n::Int, b_range::UnitRange; seed::Integer=POLYLOG_SEED)
    return [polylog_color(n, b; seed=seed) for b in b_range]
end

# ═══════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════

function render_polylog_table(; seed::Integer=POLYLOG_SEED)
    println("\n  ╔════════════════════════════════════════════════════════════╗")
    println("  ║  Polylogarithm Values → Colors (:polylog branch)          ║")
    println("  ╚════════════════════════════════════════════════════════════╝")
    println()
    println("  Li_n(1/b) = Σ 1/(b^k · k^n) — natural BBP structure")
    println()
    
    values = polylog_table()
    
    # Group by order
    for n in 1:3
        nvals = filter(v -> v.n == n, values)
        isempty(nvals) && continue
        
        println("  Li_$n series:")
        for pv in nvals
            c = polylog_value_color(pv; seed=seed)
            ri = round(Int, c.r * 255)
            gi = round(Int, c.g * 255)
            bi = round(Int, c.b * 255)
            
            print("    ")
            print("\e[48;2;$(ri);$(gi);$(bi)m    \e[0m ")
            println("$(pv.name) = $(round(pv.value, digits=8))")
        end
        println()
    end
end

function render_polylog_grid(; n_range=1:4, b_range=2:10, seed::Integer=POLYLOG_SEED)
    println("\n  ═══════════════════════════════════════════════════════")
    println("  Polylogarithm Grid: Li_n(1/b)")
    println("  ═══════════════════════════════════════════════════════\n")
    
    # Header
    print("       ")
    for b in b_range
        print(" b=$b ")
    end
    println()
    print("       ")
    for _ in b_range
        print("─────")
    end
    println()
    
    # Grid
    for n in n_range
        print("  n=$n  ")
        for b in b_range
            c = polylog_color(n, b; seed=seed)
            ri = round(Int, c.r * 255)
            gi = round(Int, c.g * 255)
            bi = round(Int, c.b * 255)
            print("\e[48;2;$(ri);$(gi);$(bi)m     \e[0m")
        end
        
        # Show one value
        v = polylog_bbp(n, first(b_range))
        println("  Li_$n(1/$(first(b_range)))=$(round(v, digits=4))")
    end
    println()
end

function render_ternary_family(; seed::Integer=POLYLOG_SEED)
    println("\n  ═══════════════════════════════════════════════════════")
    println("  Ternary BBP Family: Li_n(1/3^m)")
    println("  ═══════════════════════════════════════════════════════\n")
    
    println("  These allow base-3^m digit extraction (Adegoke 2010)")
    println()
    
    for n in 1:3
        print("  Li_$n: ")
        for m in 1:4
            b = 3^m
            c = polylog_color(n, b; seed=seed)
            ri = round(Int, c.r * 255)
            gi = round(Int, c.g * 255)
            bi = round(Int, c.b * 255)
            
            v = polylog_bbp(n, b)
            print("\e[48;2;$(ri);$(gi);$(bi)m 3^$m \e[0m ")
        end
        println()
    end
    println()
    println("  Each cell: Li_n(1/3^m) with deterministic color")
end

function demo_polylog_spi()
    println("\n  ═══════════════════════════════════════════════════════")
    println("  SPI Verification: :polylog branch independence")
    println("  ═══════════════════════════════════════════════════════\n")
    
    # Li_3(1/27) from different branches
    idx = 3 * 1000 + 27  # Same index
    
    pseed = branch_seed(314159, :polylog)
    tseed = branch_seed(314159, :triangle_magic)
    gseed = branch_seed(314159, :galperin)
    
    println("  Same index ($idx) from different branches:")
    
    for (name, seed) in [(:polylog, pseed), (:triangle_magic, tseed), (:galperin, gseed)]
        c = color_at(idx, Rec2020(); seed=seed)
        ri = round(Int, c.r * 255)
        gi = round(Int, c.g * 255)
        bi = round(Int, c.b * 255)
        print("    :$name ")
        print("\e[48;2;$(ri);$(gi);$(bi)m    \e[0m\n")
    end
    
    println("\n  ◆ Each branch independent")
end

# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

function main()
    println("\n" * "═"^70)
    println("  Polylogarithm Colors - :polylog continuation branch")
    println("═"^70)
    
    # Show polylog table
    render_polylog_table()
    
    # Show grid
    render_polylog_grid(n_range=1:3, b_range=2:6)
    
    # Ternary family
    render_ternary_family()
    
    # Verify branch independence
    demo_polylog_spi()
    
    println("\n  Properties:")
    println("  ◆ (n, b) → deterministic color")
    println("  ◆ Natural BBP structure: Li_n(1/b) = Σ 1/(b^k · k^n)")
    println("  ◆ Ternary family enables base-3 digit extraction")
    println("  ◆ :polylog branch independent of others")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
