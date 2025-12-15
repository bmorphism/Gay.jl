# Ultrametric P-adic Color Space
# 
# P-adic numbers provide an ultrametric (non-Archimedean) distance where
# |a - b|_p = p^(-v_p(a-b)) where v_p is the p-adic valuation.
#
# For colors: nearby p-adically = similar high-frequency structure
# Ultrametric inequality: d(x,z) ≤ max(d(x,y), d(y,z))
#
# This is OPPOSITE to Euclidean - colors "cluster" into balls where
# every point is a center!

export PadicColor, padic_distance, padic_hue, ultrametric_palette
export padic_ball, padic_valuation, demo_ultrametric

const GAY_SEED = UInt64(0x6761795f636f6c6f)

"""
    padic_valuation(n::Integer, p::Integer) -> Int

Compute the p-adic valuation v_p(n) = largest k such that p^k divides n.
Returns Inf for n = 0.
"""
function padic_valuation(n::Integer, p::Integer)
    n == 0 && return typemax(Int)
    v = 0
    while mod(n, p) == 0
        n = div(n, p)
        v += 1
    end
    v
end

"""
    padic_distance(a::Integer, b::Integer, p::Integer) -> Float64

Compute the p-adic distance |a - b|_p = p^(-v_p(a-b)).
"""
function padic_distance(a::Integer, b::Integer, p::Integer)
    a == b && return 0.0
    v = padic_valuation(abs(a - b), p)
    v == typemax(Int) ? 0.0 : Float64(p)^(-v)
end

"""
    PadicColor

A color in p-adic space with base prime p.
"""
struct PadicColor
    seed::UInt64
    p::Int           # Base prime (2, 3, 5, 7, ...)
    digits::Vector{Int}  # P-adic expansion (least significant first)
    hue::Float64
    saturation::Float64
    lightness::Float64
end

"""
    padic_hue(seed::UInt64, p::Int, precision::Int=8) -> Float64

Convert seed to a p-adic hue via Hensel lifting.
"""
function padic_hue(seed::UInt64, p::Int, precision::Int=8)
    # Extract p-adic digits
    digits = Int[]
    n = seed
    for _ in 1:precision
        push!(digits, mod(n, p))
        n = div(n, p)
    end
    
    # Convert to [0, 360) via Cantor-like mapping
    hue = 0.0
    scale = 1.0 / p
    for d in digits
        hue += d * scale
        scale /= p
    end
    
    hue * 360.0
end

"""
    PadicColor(seed::UInt64; p::Int=3)

Create a p-adic color from seed. Default p=3 for triadic system.
"""
function PadicColor(seed::UInt64; p::Int=3)
    precision = 12
    
    # P-adic digits
    digits = Int[]
    n = seed
    for _ in 1:precision
        push!(digits, Int(mod(n, p)))
        n = div(n, p)
    end
    
    # Hue from p-adic expansion
    hue = padic_hue(seed, p, precision)
    
    # Saturation and lightness from different primes
    next_p = p == 2 ? 3 : (p == 3 ? 5 : 7)
    sat_raw = padic_hue(seed ⊻ 0x5a5a5a5a5a5a5a5a, next_p, 6) / 360.0
    saturation = 0.5 + 0.4 * sat_raw  # [0.5, 0.9]
    
    light_raw = padic_hue(seed ⊻ 0xa5a5a5a5a5a5a5a5, next_p, 6) / 360.0
    lightness = 0.35 + 0.4 * light_raw  # [0.35, 0.75]
    
    PadicColor(seed, p, digits, hue, saturation, lightness)
end

"""
    padic_ball(center::PadicColor, radius::Float64) -> Vector{PadicColor}

Generate colors in the p-adic ball of given radius around center.
In ultrametric spaces, every point in a ball is a center!
"""
function padic_ball(center::PadicColor, radius::Float64; n::Int=8)
    p = center.p
    
    # Determine how many digits to vary based on radius
    # radius = p^(-k) means vary the first k digits
    k = max(1, Int(floor(-log(radius) / log(p))))
    
    colors = PadicColor[]
    push!(colors, center)
    
    # Generate variations
    base = center.seed
    for i in 1:n-1
        # Perturb lower-order digits
        perturbation = UInt64(i) * UInt64(p)^(k-1)
        new_seed = (base ⊻ perturbation) 
        push!(colors, PadicColor(new_seed; p=p))
    end
    
    colors
end

"""
    ultrametric_palette(seed::UInt64; p::Int=3, n::Int=7) -> Vector{PadicColor}

Generate an ultrametrically spaced palette.
"""
function ultrametric_palette(seed::UInt64; p::Int=3, n::Int=7)
    colors = PadicColor[]
    state = seed
    
    for i in 1:n
        push!(colors, PadicColor(state; p=p))
        # SM64 PRNG step for next seed
        state = (state * 0x5D588B656C078965 + 0x269EC3) & 0xFFFFFFFFFFFFFFFF
    end
    
    colors
end

"""
    demo_ultrametric()

Demo of p-adic ultrametric color generation.
"""
function demo_ultrametric()
    println("═══════════════════════════════════════════════════════════")
    println("  P-ADIC ULTRAMETRIC COLORS")
    println("═══════════════════════════════════════════════════════════")
    println()
    
    println("ULTRAMETRIC PROPERTY: d(x,z) ≤ max(d(x,y), d(y,z))")
    println("Every point in a ball is its center!")
    println()
    
    # Generate 3-adic palette
    palette = ultrametric_palette(GAY_SEED; p=3, n=7)
    
    println("3-ADIC PALETTE (seed 0x$(string(GAY_SEED, base=16))):")
    println("─────────────────────────────────────────────────────────")
    
    for (i, c) in enumerate(palette)
        digits_str = join(c.digits[1:6], "")
        println("  [$i] H=$(round(c.hue, digits=1))° S=$(round(c.saturation, digits=2)) L=$(round(c.lightness, digits=2))")
        println("      3-adic: ...$digits_str (seed 0x$(string(c.seed, base=16, pad=16)))")
    end
    
    println()
    println("DISTANCE MATRIX (3-adic):")
    println("─────────────────────────────────────────────────────────")
    
    # Show distance matrix
    print("     ")
    for i in 1:min(5, length(palette))
        print("  [$i]  ")
    end
    println()
    
    for i in 1:min(5, length(palette))
        print(" [$i] ")
        for j in 1:min(5, length(palette))
            d = padic_distance(Int(palette[i].seed % 1000000), Int(palette[j].seed % 1000000), 3)
            print(" $(round(d, digits=3)) ")
        end
        println()
    end
    
    println()
    println("ULTRAMETRIC BALLS:")
    println("─────────────────────────────────────────────────────────")
    
    center = palette[1]
    ball = padic_ball(center, 0.1; n=5)
    
    println("Ball of radius 0.1 around color 1:")
    for (i, c) in enumerate(ball)
        d = padic_distance(Int(center.seed % 1000000), Int(c.seed % 1000000), 3)
        println("  [$i] distance = $d, H=$(round(c.hue, digits=1))°")
    end
    
    palette
end
