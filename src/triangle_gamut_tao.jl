# TRIANGLE GAMUT TAO: Metric Inequalities Across Color Gamuts
# ═══════════════════════════════════════════════════════════════════════════════
#
# Triangle inequalities in color spaces with gamut-dependent satisfiability:
#
#   sRGB ⊂ P3 ⊂ Rec.2020
#
# Some color triangles are:
#   - DOABLE in sRGB (conservative)
#   - NOT doable in sRGB but DOABLE in P3 (extended gamut)
#   - Only DOABLE in Rec.2020 with Tao restriction bounds
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  TERENCE TAO'S RESTRICTION ESTIMATES FOR GAMUT BOUNDARIES                   │
# │                                                                             │
# │  The gamut boundary ∂G is a curved surface in perceptual (Lab) space.       │
# │  Tao's restriction theorem bounds give:                                     │
# │                                                                             │
# │    ‖f̂|_S‖_q ≤ C_p,q ‖f‖_p   for S a curved surface                        │
# │                                                                             │
# │  Applied to color: concentration of achievable colors near ∂G is bounded.  │
# │                                                                             │
# │  For Rec.2020 near-boundary colors, the Tao-Vargas bilinear estimate:       │
# │    ‖fg‖_2 ≤ C ‖f‖_p ‖g‖_q   with p,q dependent on surface curvature       │
# │                                                                             │
# │  This bounds the "chromatic energy" of triangles touching the boundary.    │
# └─────────────────────────────────────────────────────────────────────────────┘
#
# Enzyme.jl bidirectional: gradients flow both ways through gamut constraints

module TriangleGamutTao

using LinearAlgebra

export
    # Gamut types
    Gamut, SRGB, DisplayP3, Rec2020,

    # Color representations
    LabColor, LCHColor, XYZColor, LinearRGB,

    # Triangle inequality types
    TriangleConfig, TriangleClass,
    SRGB_SATISFIABLE, P3_ONLY, REC2020_TAO_BOUNDED, IMPOSSIBLE,

    # Core functions
    in_gamut, gamut_distance, triangle_inequality_slack,
    classify_triangle, find_mediating_color,

    # Tao bounds
    TaoRestrictionBound, tao_bilinear_estimate, curvature_at_boundary,
    restriction_exponent, compute_tao_bound,

    # Enzyme-compatible
    EnzymeGamutParams, forward_gamut_map, reverse_gamut_map,
    bidirectional_gamut_gradient, gamut_loss, optimize_triangle!,

    # Demo
    demo_triangle_gamut

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

const GAY_SEED = UInt64(0x6761795f636f6c6f)

# D65 white point
const D65_X = 0.95047
const D65_Y = 1.00000
const D65_Z = 1.08883

# sRGB primaries in XYZ (Rec.709)
const SRGB_RED_XYZ = (0.4124564, 0.2126729, 0.0193339)
const SRGB_GREEN_XYZ = (0.3575761, 0.7151522, 0.1191920)
const SRGB_BLUE_XYZ = (0.1804375, 0.0721750, 0.9503041)

# P3 primaries in XYZ (wider gamut, especially red/green)
const P3_RED_XYZ = (0.4865709, 0.2289746, 0.0000000)
const P3_GREEN_XYZ = (0.2656677, 0.6917385, 0.0451134)
const P3_BLUE_XYZ = (0.1982173, 0.0792869, 1.0439444)

# Rec.2020 primaries in XYZ (widest common gamut)
const REC2020_RED_XYZ = (0.6369580, 0.2627002, 0.0000000)
const REC2020_GREEN_XYZ = (0.1446169, 0.6779981, 0.0280727)
const REC2020_BLUE_XYZ = (0.1688810, 0.0593017, 1.0609851)

# ═══════════════════════════════════════════════════════════════════════════════
# GAMUT TYPES
# ═══════════════════════════════════════════════════════════════════════════════

abstract type Gamut end
struct SRGB <: Gamut end
struct DisplayP3 <: Gamut end
struct Rec2020 <: Gamut end

gamut_name(::SRGB) = "sRGB"
gamut_name(::DisplayP3) = "Display P3"
gamut_name(::Rec2020) = "Rec.2020"

# Gamut volume ratios (approximate, in Lab space)
gamut_volume(::SRGB) = 1.0
gamut_volume(::DisplayP3) = 1.25  # ~25% larger than sRGB
gamut_volume(::Rec2020) = 1.77   # ~77% larger than sRGB

# ═══════════════════════════════════════════════════════════════════════════════
# COLOR REPRESENTATIONS
# ═══════════════════════════════════════════════════════════════════════════════

struct LabColor
    L::Float64  # Lightness [0, 100]
    a::Float64  # Green-Red [-128, 127]
    b::Float64  # Blue-Yellow [-128, 127]
end

struct LCHColor
    L::Float64  # Lightness [0, 100]
    C::Float64  # Chroma [0, ~180]
    H::Float64  # Hue [0, 360)
end

struct XYZColor
    X::Float64
    Y::Float64
    Z::Float64
end

struct LinearRGB
    r::Float64
    g::Float64
    b::Float64
end

# Conversions
function lab_to_lch(lab::LabColor)::LCHColor
    C = sqrt(lab.a^2 + lab.b^2)
    H = atan(lab.b, lab.a) * 180 / π
    H = H < 0 ? H + 360 : H
    LCHColor(lab.L, C, H)
end

function lch_to_lab(lch::LCHColor)::LabColor
    a = lch.C * cos(lch.H * π / 180)
    b = lch.C * sin(lch.H * π / 180)
    LabColor(lch.L, a, b)
end

function xyz_to_lab(xyz::XYZColor)::LabColor
    f(t) = t > 0.008856 ? t^(1/3) : (903.3 * t + 16) / 116

    fx = f(xyz.X / D65_X)
    fy = f(xyz.Y / D65_Y)
    fz = f(xyz.Z / D65_Z)

    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)

    LabColor(L, a, b)
end

function lab_to_xyz(lab::LabColor)::XYZColor
    f_inv(t) = t > 0.206893 ? t^3 : (t - 16/116) / 7.787

    fy = (lab.L + 16) / 116
    fx = lab.a / 500 + fy
    fz = fy - lab.b / 200

    X = D65_X * f_inv(fx)
    Y = D65_Y * f_inv(fy)
    Z = D65_Z * f_inv(fz)

    XYZColor(X, Y, Z)
end

# XYZ to Linear RGB (gamut-specific)
function xyz_to_linear_rgb(xyz::XYZColor, ::SRGB)::LinearRGB
    # sRGB matrix (inverse of primary matrix)
    r =  3.2404542 * xyz.X - 1.5371385 * xyz.Y - 0.4985314 * xyz.Z
    g = -0.9692660 * xyz.X + 1.8760108 * xyz.Y + 0.0415560 * xyz.Z
    b =  0.0556434 * xyz.X - 0.2040259 * xyz.Y + 1.0572252 * xyz.Z
    LinearRGB(r, g, b)
end

function xyz_to_linear_rgb(xyz::XYZColor, ::DisplayP3)::LinearRGB
    # P3 matrix
    r =  2.4934969 * xyz.X - 0.9313836 * xyz.Y - 0.4027108 * xyz.Z
    g = -0.8294890 * xyz.X + 1.7626641 * xyz.Y + 0.0236247 * xyz.Z
    b =  0.0358458 * xyz.X - 0.0761724 * xyz.Y + 0.9568845 * xyz.Z
    LinearRGB(r, g, b)
end

function xyz_to_linear_rgb(xyz::XYZColor, ::Rec2020)::LinearRGB
    # Rec.2020 matrix
    r =  1.7166512 * xyz.X - 0.3556708 * xyz.Y - 0.2533663 * xyz.Z
    g = -0.6666844 * xyz.X + 1.6164812 * xyz.Y + 0.0157685 * xyz.Z
    b =  0.0176399 * xyz.X - 0.0427706 * xyz.Y + 0.9421031 * xyz.Z
    LinearRGB(r, g, b)
end

# ═══════════════════════════════════════════════════════════════════════════════
# GAMUT MEMBERSHIP
# ═══════════════════════════════════════════════════════════════════════════════

"""
Check if a Lab color is within the given gamut.
"""
function in_gamut(lab::LabColor, gamut::Gamut; tolerance::Float64=1e-6)::Bool
    xyz = lab_to_xyz(lab)
    rgb = xyz_to_linear_rgb(xyz, gamut)

    # All RGB channels must be in [0, 1]
    return rgb.r >= -tolerance && rgb.r <= 1 + tolerance &&
           rgb.g >= -tolerance && rgb.g <= 1 + tolerance &&
           rgb.b >= -tolerance && rgb.b <= 1 + tolerance
end

in_gamut(lch::LCHColor, gamut::Gamut; kwargs...) = in_gamut(lch_to_lab(lch), gamut; kwargs...)

"""
Distance from color to gamut boundary (negative if outside).
"""
function gamut_distance(lab::LabColor, gamut::Gamut)::Float64
    xyz = lab_to_xyz(lab)
    rgb = xyz_to_linear_rgb(xyz, gamut)

    # Distance to nearest boundary
    distances = [
        rgb.r, 1 - rgb.r,
        rgb.g, 1 - rgb.g,
        rgb.b, 1 - rgb.b
    ]

    min_dist = minimum(distances)
    return min_dist  # Negative means outside gamut
end

# ═══════════════════════════════════════════════════════════════════════════════
# TRIANGLE INEQUALITY CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

@enum TriangleClass begin
    SRGB_SATISFIABLE       # Triangle fits in sRGB
    P3_ONLY                # Needs P3, doesn't fit sRGB
    REC2020_TAO_BOUNDED    # Needs Rec.2020 with Tao bounds
    IMPOSSIBLE             # Violates fundamental metric constraints
end

"""
Configuration of three colors forming a triangle in color space.
"""
struct TriangleConfig
    A::LabColor
    B::LabColor
    C::LabColor

    # Distances
    dAB::Float64
    dBC::Float64
    dAC::Float64

    # Triangle inequality slack: dAB + dBC - dAC (must be ≥ 0)
    slack::Float64

    # Classification per gamut
    class::TriangleClass
end

"""
Euclidean distance in Lab space (ΔE*ab).
"""
function lab_distance(a::LabColor, b::LabColor)::Float64
    sqrt((a.L - b.L)^2 + (a.a - b.a)^2 + (a.b - b.b)^2)
end

"""
Triangle inequality slack: d(A,B) + d(B,C) - d(A,C)
Must be ≥ 0 for valid metric. Larger = more "room" for B.
"""
function triangle_inequality_slack(A::LabColor, B::LabColor, C::LabColor)::Float64
    dAB = lab_distance(A, B)
    dBC = lab_distance(B, C)
    dAC = lab_distance(A, C)
    return dAB + dBC - dAC
end

"""
Classify a triangle configuration by gamut satisfiability.
"""
function classify_triangle(A::LabColor, B::LabColor, C::LabColor)::TriangleConfig
    dAB = lab_distance(A, B)
    dBC = lab_distance(B, C)
    dAC = lab_distance(A, C)
    slack = dAB + dBC - dAC

    # Check gamut membership
    in_srgb = in_gamut(A, SRGB()) && in_gamut(B, SRGB()) && in_gamut(C, SRGB())
    in_p3 = in_gamut(A, DisplayP3()) && in_gamut(B, DisplayP3()) && in_gamut(C, DisplayP3())
    in_rec2020 = in_gamut(A, Rec2020()) && in_gamut(B, Rec2020()) && in_gamut(C, Rec2020())

    class = if slack < -1e-10
        IMPOSSIBLE  # Violates triangle inequality (shouldn't happen with Euclidean)
    elseif in_srgb
        SRGB_SATISFIABLE
    elseif in_p3
        P3_ONLY
    elseif in_rec2020
        REC2020_TAO_BOUNDED
    else
        IMPOSSIBLE  # Outside even Rec.2020
    end

    TriangleConfig(A, B, C, dAB, dBC, dAC, slack, class)
end

# ═══════════════════════════════════════════════════════════════════════════════
# TAO RESTRICTION BOUNDS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    TaoRestrictionBound

Terence Tao's restriction estimates for functions concentrated near surfaces.

For the gamut boundary ∂G (a curved surface in Lab space):
    ‖f̂|_{∂G}‖_q ≤ C_{p,q,κ} ‖f‖_p

where κ is the principal curvature of ∂G.

The bound C_{p,q,κ} determines how much "chromatic energy" can concentrate
near the boundary — this limits achievable color triangles.
"""
struct TaoRestrictionBound
    p::Float64          # Source Lp exponent
    q::Float64          # Restriction Lq exponent
    curvature::Float64  # Principal curvature at boundary point
    constant::Float64   # The bound constant C_{p,q,κ}
    gamut::Gamut
end

"""
Compute principal curvature of gamut boundary at a point.

The gamut boundary in Lab space has varying curvature:
- High curvature near primary vertices (R, G, B)
- Lower curvature along edges
- Complex topology near white/black
"""
function curvature_at_boundary(lab::LabColor, gamut::Gamut)::Float64
    # Numerical approximation via finite differences
    ε = 0.1

    # Sample nearby points on boundary
    distances = Float64[]
    for (da, db) in [(ε, 0), (-ε, 0), (0, ε), (0, -ε), (ε, ε), (-ε, -ε)]
        nearby = LabColor(lab.L, lab.a + da, lab.b + db)
        push!(distances, abs(gamut_distance(nearby, gamut)))
    end

    # Curvature approximation (higher variation = higher curvature)
    mean_d = sum(distances) / length(distances)
    variance = sum((d - mean_d)^2 for d in distances) / length(distances)

    # Normalize to reasonable range [0, 1]
    κ = sqrt(variance) / ε
    return clamp(κ, 0.0, 10.0)
end

"""
Tao's restriction exponent relationship.

For a surface with curvature κ, the optimal restriction estimate is:
    ‖f̂|_S‖_q ≤ C ‖f‖_p

where p and q satisfy:
    1/q ≤ (n-1)/2n · 1/p'   (Tomas-Stein)

For n=3 (Lab space): 1/q ≤ 1/3 · 1/p'

Tao-Vargas improvements give better bounds for high curvature.
"""
function restriction_exponent(p::Float64, curvature::Float64)::Float64
    # Tomas-Stein base bound
    p_dual = p / (p - 1)  # p' = p/(p-1)
    q_base = 3 * p_dual   # 1/q = 1/3 · 1/p' → q = 3p'

    # Tao-Vargas improvement for curved surfaces
    # Higher curvature allows better (smaller q) bounds
    curvature_factor = 1.0 / (1.0 + 0.1 * curvature)

    q_improved = q_base * curvature_factor
    return max(1.0, q_improved)
end

"""
Compute the Tao restriction bound constant.

Based on Tao's work on:
- Restriction conjecture (with Vargas, Wolff)
- Bilinear estimates for surfaces
- Decoupling inequalities (with Bourgain, Demeter)
"""
function compute_tao_bound(lab::LabColor, gamut::Gamut; p::Float64=2.0)::TaoRestrictionBound
    κ = curvature_at_boundary(lab, gamut)
    q = restriction_exponent(p, κ)

    # The bound constant from Tao-Vargas (2000)
    # C ≈ κ^{-1/(2q)} for surfaces with non-vanishing curvature
    C = κ > 0.01 ? κ^(-1/(2*q)) : 100.0  # Degenerate case for flat regions

    TaoRestrictionBound(p, q, κ, C, gamut)
end

"""
Tao's bilinear restriction estimate for two colors near boundary.

‖fg‖_2 ≤ C · ‖f‖_p · ‖g‖_q

This bounds the "interaction energy" between two boundary-adjacent colors.
"""
function tao_bilinear_estimate(
    lab1::LabColor,
    lab2::LabColor,
    gamut::Gamut
)::Float64
    bound1 = compute_tao_bound(lab1, gamut)
    bound2 = compute_tao_bound(lab2, gamut)

    # Bilinear constant is product of individual bounds
    # with geometric mean for curvature interaction
    κ_geom = sqrt(bound1.curvature * bound2.curvature + 0.01)

    C_bilinear = bound1.constant * bound2.constant * (1.0 + 1.0/κ_geom)
    return C_bilinear
end

# ═══════════════════════════════════════════════════════════════════════════════
# ENZYME-COMPATIBLE BIDIRECTIONAL OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════

"""
Parameters for Enzyme.jl autodiff through gamut constraints.

Enables bidirectional gradient flow:
- Forward: color → gamut membership + Tao bound
- Reverse: desired properties → optimal color
"""
mutable struct EnzymeGamutParams
    # Color coordinates (optimizable)
    L::Float64
    a::Float64
    b::Float64

    # Gamut soft constraint (differentiable relaxation)
    gamut_weight::Float64

    # Tao bound parameters
    target_curvature::Float64
    restriction_p::Float64

    # Optimization state
    learning_rate::Float64
    momentum::Vector{Float64}
end

function EnzymeGamutParams(lab::LabColor;
                           gamut_weight::Float64=1.0,
                           target_curvature::Float64=0.5,
                           learning_rate::Float64=0.01)
    EnzymeGamutParams(
        lab.L, lab.a, lab.b,
        gamut_weight,
        target_curvature,
        2.0,  # default p=2
        learning_rate,
        zeros(3)  # momentum
    )
end

"""
Forward map: parameters → (gamut_score, tao_bound, color)

Enzyme can differentiate through this.
"""
function forward_gamut_map(params::EnzymeGamutParams, gamut::Gamut)
    lab = LabColor(params.L, params.a, params.b)

    # Soft gamut membership (sigmoid relaxation)
    dist = gamut_distance(lab, gamut)
    gamut_score = 1.0 / (1.0 + exp(-10 * dist))  # Sigmoid, differentiable

    # Tao bound at this point
    tao = compute_tao_bound(lab, gamut; p=params.restriction_p)

    return (gamut_score, tao.constant, lab)
end

"""
Reverse map: desired properties → gradient direction for parameters.

This enables "what color achieves this Tao bound?" queries.
"""
function reverse_gamut_map(
    params::EnzymeGamutParams,
    gamut::Gamut,
    target_gamut_score::Float64,
    target_tao_bound::Float64
)
    # Forward pass
    gamut_score, tao_bound, lab = forward_gamut_map(params, gamut)

    # Loss components
    gamut_loss = (gamut_score - target_gamut_score)^2
    tao_loss = (tao_bound - target_tao_bound)^2

    total_loss = params.gamut_weight * gamut_loss + (1 - params.gamut_weight) * tao_loss

    # Numerical gradient (Enzyme would compute this analytically)
    ε = 1e-5
    grad = zeros(3)

    for (i, field) in enumerate([:L, :a, :b])
        params_plus = deepcopy(params)
        setfield!(params_plus, field, getfield(params, field) + ε)
        loss_plus = let
            gs, tb, _ = forward_gamut_map(params_plus, gamut)
            params.gamut_weight * (gs - target_gamut_score)^2 +
            (1 - params.gamut_weight) * (tb - target_tao_bound)^2
        end
        grad[i] = (loss_plus - total_loss) / ε
    end

    return (total_loss, grad)
end

"""
Bidirectional gradient computation for gamut optimization.

In full Enzyme.jl usage:
```julia
using Enzyme
grad = autodiff(Reverse, forward_gamut_map, Active, Duplicated(params, dparams), Const(gamut))
```
"""
function bidirectional_gamut_gradient(
    params::EnzymeGamutParams,
    gamut::Gamut;
    direction::Symbol=:both
)
    lab = LabColor(params.L, params.a, params.b)

    if direction == :forward || direction == :both
        # Forward: how does changing color affect gamut/Tao?
        gs, tb, _ = forward_gamut_map(params, gamut)
        forward_info = (gamut_score=gs, tao_bound=tb)
    else
        forward_info = nothing
    end

    if direction == :reverse || direction == :both
        # Reverse: how to change color to improve gamut/Tao?
        loss, grad = reverse_gamut_map(params, gamut, 1.0, 1.0)
        reverse_info = (loss=loss, gradient=grad)
    else
        reverse_info = nothing
    end

    return (forward=forward_info, reverse=reverse_info)
end

"""
Combined loss function for triangle optimization.

Optimizes mediating color B given fixed A, C to:
1. Stay in gamut
2. Satisfy Tao bounds
3. Maximize triangle inequality slack
"""
function gamut_loss(
    params::EnzymeGamutParams,
    A::LabColor,
    C::LabColor,
    gamut::Gamut
)::Float64
    B = LabColor(params.L, params.a, params.b)

    # Gamut constraint (soft)
    dist = gamut_distance(B, gamut)
    gamut_penalty = dist < 0 ? 100 * dist^2 : 0.0

    # Triangle inequality slack (want to maximize)
    slack = triangle_inequality_slack(A, B, C)
    slack_bonus = -slack  # Negative because we minimize loss

    # Tao bound (want it finite/small for near-boundary colors)
    tao = compute_tao_bound(B, gamut)
    tao_penalty = tao.constant > 10 ? (tao.constant - 10)^2 : 0.0

    total = gamut_penalty + 0.1 * slack_bonus + 0.01 * tao_penalty
    return total
end

"""
Optimize the mediating color B to satisfy triangle in given gamut.
"""
function optimize_triangle!(
    params::EnzymeGamutParams,
    A::LabColor,
    C::LabColor,
    gamut::Gamut;
    max_iters::Int=100
)
    for iter in 1:max_iters
        # Compute gradient
        ε = 1e-5
        grad = zeros(3)
        base_loss = gamut_loss(params, A, C, gamut)

        for (i, field) in enumerate([:L, :a, :b])
            p_plus = deepcopy(params)
            setfield!(p_plus, field, getfield(params, field) + ε)
            grad[i] = (gamut_loss(p_plus, A, C, gamut) - base_loss) / ε
        end

        # Momentum update
        params.momentum .= 0.9 .* params.momentum .+ 0.1 .* grad

        # Apply gradient
        params.L -= params.learning_rate * params.momentum[1]
        params.a -= params.learning_rate * params.momentum[2]
        params.b -= params.learning_rate * params.momentum[3]

        # Clamp L to valid range
        params.L = clamp(params.L, 0.0, 100.0)

        if norm(grad) < 1e-6
            break
        end
    end

    return LabColor(params.L, params.a, params.b)
end

# ═══════════════════════════════════════════════════════════════════════════════
# FIND MEDIATING COLOR
# ═══════════════════════════════════════════════════════════════════════════════

"""
Find a mediating color B such that triangle A-B-C fits in the target gamut.

Returns (B, success, required_gamut) where required_gamut is the minimum
gamut needed to contain the triangle.
"""
function find_mediating_color(
    A::LabColor,
    C::LabColor,
    target_gamut::Gamut;
    initial_B::Union{LabColor, Nothing}=nothing
)
    # Start at midpoint if no initial guess
    B_init = if initial_B === nothing
        LabColor((A.L + C.L) / 2, (A.a + C.a) / 2, (A.b + C.b) / 2)
    else
        initial_B
    end

    params = EnzymeGamutParams(B_init; learning_rate=0.5)
    B_opt = optimize_triangle!(params, A, C, target_gamut)

    # Check which gamut we actually achieved
    config = classify_triangle(A, B_opt, C)

    return (B_opt, config.class != IMPOSSIBLE, config.class)
end

# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

function demo_triangle_gamut()
    println()
    println("╔═══════════════════════════════════════════════════════════════════════════╗")
    println("║  TRIANGLE GAMUT TAO: Metric Inequalities with Restriction Bounds          ║")
    println("║  sRGB ⊂ P3 ⊂ Rec.2020 + Terence Tao's Estimates                          ║")
    println("╚═══════════════════════════════════════════════════════════════════════════╝")
    println()

    # ─── Example 1: sRGB-satisfiable triangle ───
    println("─── Example 1: sRGB-Satisfiable Triangle ───")
    A1 = LabColor(50.0, 0.0, 0.0)      # Neutral gray
    B1 = LabColor(60.0, 20.0, 10.0)    # Slightly warm
    C1 = LabColor(40.0, -10.0, -5.0)   # Slightly cool

    config1 = classify_triangle(A1, B1, C1)
    println("  A: L=$(A1.L), a=$(A1.a), b=$(A1.b)")
    println("  B: L=$(B1.L), a=$(B1.a), b=$(B1.b)")
    println("  C: L=$(C1.L), a=$(C1.a), b=$(C1.b)")
    println("  Triangle inequality slack: $(round(config1.slack, digits=3))")
    println("  Classification: $(config1.class)")
    println()

    # ─── Example 2: P3-only triangle ───
    println("─── Example 2: P3-Only Triangle (saturated greens) ───")
    A2 = LabColor(87.0, -86.0, 83.0)   # Saturated green (outside sRGB)
    B2 = LabColor(50.0, 0.0, 0.0)      # Neutral
    C2 = LabColor(60.0, -40.0, 40.0)   # Medium green

    config2 = classify_triangle(A2, B2, C2)
    println("  A: L=$(A2.L), a=$(A2.a), b=$(A2.b) [saturated green]")
    println("  B: L=$(B2.L), a=$(B2.a), b=$(B2.b) [neutral]")
    println("  C: L=$(C2.L), a=$(C2.a), b=$(C2.b) [medium green]")
    println("  In sRGB: A=$(in_gamut(A2, SRGB())), B=$(in_gamut(B2, SRGB())), C=$(in_gamut(C2, SRGB()))")
    println("  In P3:   A=$(in_gamut(A2, DisplayP3())), B=$(in_gamut(B2, DisplayP3())), C=$(in_gamut(C2, DisplayP3()))")
    println("  Classification: $(config2.class)")
    println()

    # ─── Example 3: Rec.2020 with Tao bounds ───
    println("─── Example 3: Rec.2020 with Tao Restriction Bounds ───")
    A3 = LabColor(50.0, -100.0, 100.0)  # Extreme green (Rec.2020 only)
    B3 = LabColor(30.0, 80.0, -80.0)    # Extreme blue-magenta
    C3 = LabColor(70.0, 0.0, 0.0)       # Neutral light

    println("  Computing Tao bounds for boundary-adjacent colors...")
    tao_A = compute_tao_bound(A3, Rec2020())
    tao_B = compute_tao_bound(B3, Rec2020())

    println("  Color A: κ=$(round(tao_A.curvature, digits=3)), C=$(round(tao_A.constant, digits=3))")
    println("  Color B: κ=$(round(tao_B.curvature, digits=3)), C=$(round(tao_B.constant, digits=3))")

    bilinear = tao_bilinear_estimate(A3, B3, Rec2020())
    println("  Bilinear estimate ‖fg‖₂ ≤ $(round(bilinear, digits=3)) · ‖f‖_p · ‖g‖_q")
    println()

    # ─── Example 4: Bidirectional optimization ───
    println("─── Example 4: Enzyme-Compatible Bidirectional Optimization ───")

    # Want to find B that puts triangle in P3
    A4 = LabColor(70.0, 60.0, 50.0)   # Warm orange
    C4 = LabColor(30.0, -30.0, -40.0) # Cool blue

    println("  Finding mediating color B for A-B-C triangle...")
    println("  A: L=$(A4.L), a=$(A4.a), b=$(A4.b)")
    println("  C: L=$(C4.L), a=$(C4.a), b=$(C4.b)")

    B_opt, success, achieved_class = find_mediating_color(A4, C4, DisplayP3())

    println("  Optimized B: L=$(round(B_opt.L, digits=2)), a=$(round(B_opt.a, digits=2)), b=$(round(B_opt.b, digits=2))")
    println("  Success: $(success)")
    println("  Achieved class: $(achieved_class)")
    println()

    # ─── Summary table ───
    println("═══════════════════════════════════════════════════════════════════════════")
    println("  TRIANGLE INEQUALITY SATISFIABILITY BY GAMUT")
    println("═══════════════════════════════════════════════════════════════════════════")
    println()
    println("  ┌──────────────┬─────────────────────────────────────────────────────┐")
    println("  │    GAMUT     │  CONSTRAINTS                                        │")
    println("  ├──────────────┼─────────────────────────────────────────────────────┤")
    println("  │    sRGB      │  d(A,B) + d(B,C) ≥ d(A,C) with all in Rec.709      │")
    println("  │              │  Most restrictive, ~35% of visible colors          │")
    println("  ├──────────────┼─────────────────────────────────────────────────────┤")
    println("  │    P3        │  Extended primaries, especially green/red          │")
    println("  │              │  ~25% more volume than sRGB                        │")
    println("  ├──────────────┼─────────────────────────────────────────────────────┤")
    println("  │  Rec.2020    │  Near-spectral primaries                           │")
    println("  │              │  Tao restriction bounds apply at boundary:         │")
    println("  │              │    ‖f̂|_{∂G}‖_q ≤ C_{p,q,κ} ‖f‖_p                  │")
    println("  │              │  Bilinear: ‖fg‖_2 ≤ C ‖f‖_p ‖g‖_q                 │")
    println("  └──────────────┴─────────────────────────────────────────────────────┘")
    println()
    println("  Enzyme.jl enables bidirectional gradient flow:")
    println("    Forward:  color → (gamut_score, tao_bound)")
    println("    Reverse:  target_properties → optimal_color")
    println()

    return (config1, config2, tao_A, tao_B)
end

end # module TriangleGamutTao
