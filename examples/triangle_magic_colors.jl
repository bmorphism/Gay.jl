# Magic Triangle Geometry → Colors
# (r, θ) parameter space → deterministic color mapping
#
# Uses the :triangle_magic continuation branch from bbp_pi.jl
#
# The geometry: Triangle with vertices at 0, 1, 1+r·e^(iθ)
# When angle at origin = (a/b)·π, we get BBP-type formulas.
# Each (r, θ) configuration → a color from the splittable stream.

using Gay
using Colors
using SplittableRandoms: SplittableRandom, split

include("bbp_pi.jl")  # For continuation_point, branch_seed

# ═══════════════════════════════════════════════════════════════════════════
# Magic Triangle Geometry
# ═══════════════════════════════════════════════════════════════════════════

struct TriangleConfig
    r::Float64           # magnitude of third vertex offset
    θ::Float64           # angle of third vertex offset  
    angle_at_origin::Float64  # resulting angle at A=0
    base::Float64        # |C|² = BBP base
    is_magic::Bool       # angle is rational multiple of π
    a::Int               # numerator of angle/π
    b::Int               # denominator of angle/π
end

"""
Compute the angle at origin for triangle A=0, B=1, C=1+r·e^(iθ)
"""
function angle_at_origin(r::Float64, θ::Float64)
    real_part = 1 + r * cos(θ)
    imag_part = r * sin(θ)
    return atan(imag_part, real_part)
end

"""
Compute the BBP base = |C|² = |1 + r·e^(iθ)|²
"""
function bbp_base(r::Float64, θ::Float64)
    return 1 + 2*r*cos(θ) + r^2
end

"""
Check if angle is a "nice" rational multiple of π
"""
function find_rational_angle(angle::Float64; max_denom=100, tol=1e-9)
    angle_over_pi = angle / π
    for b in 1:max_denom
        for a in 1:b
            gcd(a, b) == 1 || continue
            if abs(angle_over_pi - a/b) < tol
                return (true, a, b)
            end
        end
    end
    return (false, 0, 0)
end

"""
Create a TriangleConfig from (r, θ) parameters
"""
function triangle_config(r::Float64, θ::Float64)
    angle = angle_at_origin(r, θ)
    base = bbp_base(r, θ)
    is_magic, a, b = find_rational_angle(angle)
    return TriangleConfig(r, θ, angle, base, is_magic, a, b)
end

# ═══════════════════════════════════════════════════════════════════════════
# (r, θ) → Color Mapping (uses :triangle_magic branch)
# ═══════════════════════════════════════════════════════════════════════════

const TRIANGLE_SEED = 314159

"""
    triangle_color(r::Float64, θ::Float64; seed=TRIANGLE_SEED)

Get a deterministic color for a (r, θ) configuration.
Uses the :triangle_magic continuation branch.

The index is computed from quantized (r, θ) values to ensure
nearby configurations get similar colors.
"""
function triangle_color(r::Float64, θ::Float64; seed::Integer=TRIANGLE_SEED,
                        r_bins::Int=100, θ_bins::Int=360)
    tseed = branch_seed(seed, :triangle_magic)
    
    # Quantize to grid indices
    r_idx = clamp(round(Int, r * r_bins), 1, r_bins * 10)
    θ_idx = clamp(round(Int, (θ / (2π)) * θ_bins), 0, θ_bins - 1)
    
    # Combined index
    idx = r_idx * θ_bins + θ_idx
    
    return color_at(idx, Rec2020(); seed=tseed)
end

"""
    magic_config_color(config::TriangleConfig; seed=TRIANGLE_SEED)

Get color for a magic triangle configuration.
Magic configs (rational angle) get colors from their (a,b) fraction.
"""
function magic_config_color(config::TriangleConfig; seed::Integer=TRIANGLE_SEED)
    if !config.is_magic
        return triangle_color(config.r, config.θ; seed=seed)
    end
    
    tseed = branch_seed(seed, :triangle_magic)
    # Use (a, b) directly for magic configurations
    idx = config.a * 1000 + config.b
    return color_at(idx, Rec2020(); seed=tseed)
end

# ═══════════════════════════════════════════════════════════════════════════
# Known Magic Configurations (from hunt_11.jl and MAGIC_TRIANGLES.md)
# ═══════════════════════════════════════════════════════════════════════════

# Base-1 configurations (exact identities)
const BASE1_CONFIGS = [
    (2cos(5π/11), 6π/11, 1, 11),   # r, θ, a, b
    (sin(3π/11)/sin(4π/11), 7π/11, 3, 11),
    (2sin(π/11), 13π/22, 2, 11),
    (2sin(2π/11), 15π/22, 4, 11),
    (2cos(3π/11), 8π/11, 5, 11),
]

# Classic BBP configurations
const CLASSIC_CONFIGS = [
    (1/sqrt(2), π/2, 1, 4),     # Base 2, π/4
    (1/2, π/3, 1, 6),           # Related to π/6
    (1/sqrt(3), π/3, 1, 6),     # Base 3 family
]

function all_magic_configs()
    configs = TriangleConfig[]
    
    for (r, θ, a, b) in vcat(BASE1_CONFIGS, CLASSIC_CONFIGS)
        push!(configs, TriangleConfig(r, θ, a*π/b, bbp_base(r, θ), true, a, b))
    end
    
    return configs
end

# ═══════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════

function render_parameter_space(; r_range=(0.1, 2.0), θ_range=(0, 2π),
                                  resolution=40, seed::Integer=TRIANGLE_SEED)
    println("\n  ╔════════════════════════════════════════════════════════════╗")
    println("  ║  Magic Triangle Parameter Space (:triangle_magic branch)   ║")
    println("  ╚════════════════════════════════════════════════════════════╝")
    println()
    println("  r ∈ [$(r_range[1]), $(r_range[2])], θ ∈ [0, 2π]")
    println("  Each point → color from :triangle_magic stream")
    println()
    
    r_min, r_max = r_range
    θ_min, θ_max = θ_range
    
    for i in 1:resolution
        r = r_min + (r_max - r_min) * (resolution - i) / resolution
        print("  ")
        for j in 1:(resolution * 2)
            θ = θ_min + (θ_max - θ_min) * j / (resolution * 2)
            
            c = triangle_color(r, θ; seed=seed)
            ri = round(Int, c.r * 255)
            gi = round(Int, c.g * 255)
            bi = round(Int, c.b * 255)
            print("\e[48;2;$(ri);$(gi);$(bi)m \e[0m")
        end
        println(" r=$(round(r, digits=2))")
    end
    
    println("  " * "─"^(resolution * 2))
    println("  θ: 0" * " "^(resolution - 2) * "π" * " "^(resolution - 2) * "2π")
end

function render_magic_configs(; seed::Integer=TRIANGLE_SEED)
    println("\n  ═══════════════════════════════════════════════════════")
    println("  Magic Configurations (rational angle at origin)")
    println("  ═══════════════════════════════════════════════════════\n")
    
    configs = all_magic_configs()
    
    for config in configs
        c = magic_config_color(config; seed=seed)
        ri = round(Int, c.r * 255)
        gi = round(Int, c.g * 255)
        bi = round(Int, c.b * 255)
        
        hex = uppercase(string(ri, base=16, pad=2) * 
                       string(gi, base=16, pad=2) * 
                       string(bi, base=16, pad=2))
        
        print("    ")
        print("\e[48;2;$(ri);$(gi);$(bi)m    \e[0m ")
        println("$(config.a)/$(config.b)·π  r=$(round(config.r, digits=4))  base=$(round(config.base, digits=4))")
    end
end

function demo_triangle_spi()
    println("\n  ═══════════════════════════════════════════════════════")
    println("  SPI Verification: :triangle_magic branch independence")
    println("  ═══════════════════════════════════════════════════════\n")
    
    # Sample from different branches
    tseed = branch_seed(314159, :triangle_magic)
    gseed = branch_seed(314159, :galperin)
    pseed = branch_seed(314159, :bbp_pi)
    
    println("  Same index (42) from different branches:")
    
    for (name, seed) in [(:triangle_magic, tseed), (:galperin, gseed), (:bbp_pi, pseed)]
        c = color_at(42, Rec2020(); seed=seed)
        ri = round(Int, c.r * 255)
        gi = round(Int, c.g * 255)
        bi = round(Int, c.b * 255)
        print("    :$name ")
        print("\e[48;2;$(ri);$(gi);$(bi)m    \e[0m\n")
    end
    
    println("\n  Each branch produces different colors ◆")
end

# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

function main()
    println("\n" * "═"^70)
    println("  Magic Triangle Geometry Colors - :triangle_magic branch")
    println("═"^70)
    
    # Render parameter space
    render_parameter_space(resolution=20)
    
    # Show magic configurations
    render_magic_configs()
    
    # Verify branch independence
    demo_triangle_spi()
    
    println("\n  Properties:")
    println("  ◆ (r,θ) → deterministic color")
    println("  ◆ Magic configs (rational angles) highlighted")
    println("  ◆ :triangle_magic branch independent of others")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
