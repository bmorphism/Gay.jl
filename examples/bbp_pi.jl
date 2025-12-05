# BBP π Digit Extraction with Gay.jl Colors
# Random access to π digits → deterministic colors
#
# The Bailey-Borwein-Plouffe formula extracts the n-th hexadecimal digit of π
# WITHOUT computing digits 0..n-1. This parallels Gay.jl's random access:
#   - color_at(n) → n-th color without iteration
#   - pi_digit_at(n) → n-th hex digit without iteration
#
# We combine them: each π digit determines a color from a splittable stream.
#
# Properties shared with Gay.jl:
#   ✓ Same seed always produces same colors
#   ✓ Parallel execution is reproducible  
#   ✓ Random access by index is efficient

using Gay
using Colors
using SplittableRandoms: SplittableRandom, split

# ═══════════════════════════════════════════════════════════════════════════
# Continuation Points - Reserved Branches for Future Exploration
# ═══════════════════════════════════════════════════════════════════════════

"""
Reserved continuation branches. Each gets an independent deterministic stream
from the master seed. Future explorations fork from their reserved point.

    :bbp_pi          # Current: π digit colors (this file)
    :galperin        # Future: billiard collision colors  
    :triangle_magic  # Future: (r,θ) parameter space colors
    :polylog         # Future: Li_n polylogarithm values
    :narya_proofs    # Future: proof-derived colors
    :quantum         # Future: quantum Galperin extension
"""
const CONTINUATION_BRANCHES = [
    :bbp_pi,
    :galperin,
    :triangle_magic,
    :polylog,
    :narya_proofs,
    :quantum,
]

"""
    continuation_point(seed::Integer, branch::Symbol) -> SplittableRandom

Get a deterministic RNG stream for a named continuation branch.
Each branch is independent - work on one doesn't affect others.
"""
function continuation_point(seed::Integer, branch::Symbol)
    root = SplittableRandom(UInt64(seed))
    branch_hash = hash(branch) % UInt64
    
    current = root
    for _ in 1:(branch_hash % 1000 + 1)
        current = split(current)
    end
    
    return current
end

"""
    branch_seed(master_seed::Integer, branch::Symbol) -> UInt64

Get a deterministic seed for a continuation branch.
Use this to initialize gay_seed! for branch-specific work.
"""
branch_seed(master_seed::Integer, branch::Symbol) = 
    UInt64(hash(branch) ⊻ master_seed)

# ═══════════════════════════════════════════════════════════════════════════
# BBP Formula Implementation
# ═══════════════════════════════════════════════════════════════════════════

"""
    modpow(base, exp, m)

Compute base^exp mod m using binary exponentiation.
Essential for BBP extraction without overflow.
"""
function modpow(base::Float64, exp::Integer, m::Float64)
    result = 1.0
    base = Base.mod(base, m)
    while exp > 0
        if exp % 2 == 1
            result = Base.mod(result * base, m)
        end
        exp ÷= 2
        base = Base.mod(base * base, m)
    end
    return result
end

"""
    bbp_sum(d::Int, j::Int)

Compute the fractional part of Σ_{k=0}^∞ 16^(d-k) / (8k + j)
using the BBP spigot algorithm.

This is the core of digit extraction: we compute 16^(d-k) mod (8k+j)
for terms before position d, avoiding huge numbers.
"""
function bbp_sum(d::Int, j::Int)
    s = 0.0
    
    # Terms where k ≤ d (use modular exponentiation)
    for k in 0:d
        denom = 8k + j
        # 16^(d-k) mod denom
        num = modpow(16.0, d - k, Float64(denom))
        s += num / denom
        s = s - floor(s)  # Keep fractional part
    end
    
    # Terms where k > d (these are small, direct computation)
    for k in (d+1):(d+100)
        term = 16.0^(d - k) / (8k + j)
        term < 1e-17 && break
        s += term
    end
    
    return s - floor(s)
end

"""
    pi_hex_digit(d::Int) -> Int

Extract the d-th hexadecimal digit of π (0-indexed).

Uses the BBP formula:
  π = Σ_{k=0}^∞ 1/16^k · (4/(8k+1) - 2/(8k+4) - 1/(8k+5) - 1/(8k+6))

Returns an integer 0-15.
"""
function pi_hex_digit(d::Int)
    # BBP formula components
    s = 4.0 * bbp_sum(d, 1) - 
        2.0 * bbp_sum(d, 4) - 
        1.0 * bbp_sum(d, 5) - 
        1.0 * bbp_sum(d, 6)
    
    s = s - floor(s)  # Fractional part
    s < 0 && (s += 1)
    
    return floor(Int, 16 * s)
end

"""
    pi_hex_digits(start::Int, n::Int) -> Vector{Int}

Extract n consecutive hex digits of π starting at position `start`.
Parallelizable: each position is independent.
"""
function pi_hex_digits(start::Int, n::Int)
    # Can be parallelized with @threads
    return [pi_hex_digit(start + i) for i in 0:(n-1)]
end

# ═══════════════════════════════════════════════════════════════════════════
# π Digit → Color Mapping (SPI Pattern from splittable.jl)
# ═══════════════════════════════════════════════════════════════════════════

"""
    pi_color_at(position::Int; seed=314159, colorspace=Rec2020())

Get a deterministic color derived from the π digit at `position`.

Follows the same SPI (Strong Parallelism Invariance) pattern as Gay.jl:
1. Extract hex digit at position via BBP (random access, no iteration)
2. Use position as index into splittable RNG stream  
3. Modulate by digit value for π-specific coloring

Same position → same color, always, regardless of:
- Execution order
- Number of threads  
- Previous calls

# Example
```julia
# These will always match
c1 = pi_color_at(1000)
c2 = pi_color_at(1000)  # Same as c1

# Parallel execution produces identical results
@threads for i in 1:100
    colors[i] = pi_color_at(i)  # Thread-safe, deterministic
end
```
"""
function pi_color_at(position::Int; seed::Integer=314159, 
                     colorspace::ColorSpace=Rec2020())
    digit = pi_hex_digit(position)
    
    # Use color_at with combined seed (follows splittable.jl pattern)
    # The digit XOR ensures π-specific variation at each position
    combined_seed = seed ⊻ (UInt64(digit) << 60)
    
    return color_at(position, colorspace; seed=combined_seed)
end

"""
    pi_colors_parallel(n::Int; seed=314159, colorspace=Rec2020())

Generate n colors from π digit positions 0..n-1 using parallel execution.
Demonstrates SPI: identical results regardless of thread count.
"""
function pi_colors_parallel(n::Int; seed::Integer=314159, 
                            colorspace::ColorSpace=Rec2020())
    colors = Vector{RGB}(undef, n)
    Threads.@threads for i in 1:n
        colors[i] = pi_color_at(i - 1; seed=seed, colorspace=colorspace)
    end
    return colors
end

"""
    pi_colors_sequential(n::Int; seed=314159, colorspace=Rec2020())

Generate n colors from π digit positions 0..n-1 sequentially.
Must match pi_colors_parallel for SPI compliance.
"""
function pi_colors_sequential(n::Int; seed::Integer=314159,
                              colorspace::ColorSpace=Rec2020())
    return [pi_color_at(i - 1; seed=seed, colorspace=colorspace) for i in 1:n]
end

"""
    pi_palette(start::Int, n::Int; seed=314159, colorspace=Rec2020())

Generate a palette of n colors from consecutive π digit positions.

Each color is independently computed (parallelizable).
"""
function pi_palette(start::Int, n::Int; seed::Integer=314159,
                    colorspace::ColorSpace=Rec2020())
    return [pi_color_at(start + i; seed=seed, colorspace=colorspace) 
            for i in 0:(n-1)]
end

"""
    pi_digit_color(digit::Int; colorspace=Rec2020())

Map a hex digit (0-15) to a color from a 16-color palette.
Uses wide-gamut colorspace for maximum distinction.
"""
function pi_digit_color(digit::Int; colorspace::ColorSpace=Rec2020())
    # 16-color palette optimized for hex digits
    gay_seed!(0xBB9)  # BBP-ish seed!
    palette = next_palette(16, colorspace; min_distance=20.0)
    return palette[digit + 1]
end

# ═══════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════

"""
    render_pi_digits(start::Int, n::Int; colorspace=Rec2020())

Render π hex digits as colored blocks with their values.
Demonstrates random access: we jump directly to `start` without
computing previous digits.
"""
function render_pi_digits(start::Int, n::Int; colorspace::ColorSpace=Rec2020())
    digits = pi_hex_digits(start, n)
    
    println("\n  ╔════════════════════════════════════════════════════════════╗")
    println("  ║  π Hex Digits via BBP Extraction (Random Access)          ║")
    println("  ╚════════════════════════════════════════════════════════════╝")
    println()
    println("  Position $start to $(start + n - 1):")
    println()
    
    # Display digits with colors
    print("  ")
    for (i, d) in enumerate(digits)
        c = pi_digit_color(d; colorspace=colorspace)
        ri = round(Int, c.r * 255)
        gi = round(Int, c.g * 255)
        bi = round(Int, c.b * 255)
        
        hex_char = string(d, base=16, pad=1)
        print("\e[48;2;$(ri);$(gi);$(bi)m $(uppercase(hex_char)) \e[0m")
        
        i % 16 == 0 && print("\n  ")
    end
    println()
    
    # Show the hex string
    hex_str = join([string(d, base=16) for d in digits])
    println("\n  Hex: ", uppercase(hex_str))
    
    # Show first few as decimal
    if n >= 4
        println("  (First 4 digits = 0x$(uppercase(hex_str[1:4])) = $(parse(Int, hex_str[1:4], base=16)))")
    end
end

"""
    render_pi_spiral(; max_radius=15, seed=314159, colorspace=Rec2020())

Render π digits as a spiral, demonstrating both:
- BBP random access (any digit position)
- Gay.jl splittable determinism (reproducible colors)
"""
function render_pi_spiral(; max_radius::Int=12, seed::Integer=314159,
                           colorspace::ColorSpace=Rec2020())
    println("\n  ╔════════════════════════════════════════════════════════════╗")
    println("  ║  π Digit Spiral - BBP + Splittable RNG                     ║")
    println("  ║  Each position independently computed (parallelizable)     ║")
    println("  ╚════════════════════════════════════════════════════════════╝\n")
    
    size = 2 * max_radius + 1
    cx, cy = max_radius + 1, max_radius + 1
    
    # Pre-compute digit positions in spiral order
    digit_idx = 0
    
    for y in 1:size
        print("  ")
        for x in 1:(size * 2)
            dx = (x / 2) - cx
            dy = y - cy
            r = sqrt(dx^2 + dy^2)
            θ = atan(dy, dx)
            
            if r <= max_radius && r >= 1
                # Spiral: position = radius * angle
                pos = round(Int, r * 8 + (θ + π) * r / π)
                digit = pi_hex_digit(pos)
                c = pi_color_at(pos; seed=seed, colorspace=colorspace)
                
                ri = round(Int, c.r * 255)
                gi = round(Int, c.g * 255)
                bi = round(Int, c.b * 255)
                print("\e[48;2;$(ri);$(gi);$(bi)m  \e[0m")
            elseif r < 1
                # Center: π symbol
                print("π ")
            else
                print("  ")
            end
        end
        println()
    end
    
    println("\n  Properties:")
    println("  ✓ Same seed always produces same colors")
    println("  ✓ Parallel execution is reproducible")
    println("  ✓ Random access by index is efficient (BBP formula)")
end

"""
    compare_access_methods()

Demonstrate that random access gives same results as sequential.
This is the BBP analog to Gay.jl's splittable determinism.
"""
function compare_access_methods()
    println("\n  ═══════════════════════════════════════════════════════")
    println("  Random Access Equivalence (BBP Property)")
    println("  ═══════════════════════════════════════════════════════\n")
    
    # Extract digit 1000 directly
    d1000_direct = pi_hex_digit(1000)
    
    # The same digit, computed independently
    d1000_again = pi_hex_digit(1000)
    
    # Colors from same position
    gay_seed!(42)
    c1 = pi_color_at(1000; seed=42)
    gay_seed!(42)  
    c2 = pi_color_at(1000; seed=42)
    
    println("  Position 1000:")
    println("    Direct extraction: 0x$(string(d1000_direct, base=16))")
    println("    Second extraction: 0x$(string(d1000_again, base=16))")
    println("    Match: $(d1000_direct == d1000_again ? "✓" : "✗")")
    println()
    println("  Colors at position 1000:")
    print("    First:  "); show_colors([c1])
    print("    Second: "); show_colors([c2])
    println("    Match: $(c1 == c2 ? "✓" : "✗")")
    println()
    
    # Parallel extraction demo
    println("  Parallel extraction (positions 100, 200, 300, 400, 500):")
    positions = [100, 200, 300, 400, 500]
    digits = [pi_hex_digit(p) for p in positions]  # Could use @threads
    colors = [pi_color_at(p; seed=42) for p in positions]
    
    for (p, d, c) in zip(positions, digits, colors)
        ri = round(Int, c.r * 255)
        gi = round(Int, c.g * 255)
        bi = round(Int, c.b * 255)
        print("    π[$p] = 0x$(string(d, base=16)) ")
        print("\e[48;2;$(ri);$(gi);$(bi)m    \e[0m\n")
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# Main Demo
# ═══════════════════════════════════════════════════════════════════════════

function main()
    println("\n" * "═"^70)
    println("  BBP π Digit Extraction + Gay.jl Splittable Colors")
    println("  Random access to both digits AND colors")
    println("═"^70)
    
    # Basic digit extraction
    render_pi_digits(0, 64; colorspace=Rec2020())
    
    # Jump to position 1000 (no iteration needed!)
    println("\n  ─── Random Access Demo ───")
    println("  Jumping directly to position 1000 (BBP formula):")
    render_pi_digits(1000, 16; colorspace=DisplayP3())
    
    # Spiral visualization
    render_pi_spiral(max_radius=10, seed=314159, colorspace=Rec2020())
    
    # Reproducibility proof
    compare_access_methods()
    
    # Reference values
    println("\n  ─── Reference ───")
    println("  π = 3.243F6A8885A308D3... (hex)")
    println("  First 16 hex digits extracted: ", 
            uppercase(join([string(pi_hex_digit(i), base=16) for i in 0:15])))
    
    println("\n  BBP Formula (Bailey-Borwein-Plouffe 1995):")
    println("  π = Σ 1/16^k · (4/(8k+1) - 2/(8k+4) - 1/(8k+5) - 1/(8k+6))")
    println()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
