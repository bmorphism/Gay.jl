# src/intrinsic_nonriemannian_hsl.jl
# =============================================================================

using Colors, LinearAlgebra

struct IntrinsicHSL
    h::Float64  # Intrinsic Hue [0, 360)
    s::Float64  # Intrinsic Saturation [0, 1]
    l::Float64  # Intrinsic Lightness [0, 1]
end

"""
    to_intrinsic_hsl(c::RGB; A=25.0) -> IntrinsicHSL

Converts an RGB color into Schrödinger's intrinsic HSL coordinates using
Bujack's non-Riemannian metric tensor with saturation scale A.
"""
function to_intrinsic_hsl(c::RGB; A::Float64=25.0)
    # 1. Convert to Lab as our local Riemannian base
    lab = convert(Lab, c)
    
    # 2. Define the neutral axis (L from 0 to 100, a = b = 0)
    # Geodesics in flat Lab are lines. Under non-Riemannian d_NR, the lines
    # connecting to the neutral axis are orthogonal projections.
    L, a, b = lab.l, lab.a, lab.b
    
    # Intrinsic Lightness is the L coordinate (arc length of neutral geodesic)
    l_intrinsic = L / 100.0
    
    # Riemannian distance to neutral axis is the chroma
    chroma = sqrt(a^2 + b^2)
    
    # Intrinsic Saturation is the non-Riemannian distance from C to the neutral axis
    # f(chroma) = A * (1 - exp(-chroma / A))
    s_intrinsic = A * (1.0 - exp(-chroma / A)) / A  # normalized to [0,1]
    
    # Intrinsic Hue is the angle of the projection in the tangent plane
    h_rad = atan(b, a)
    h_intrinsic = rad2deg(h_rad)
    if h_intrinsic < 0
        h_intrinsic += 360.0
    end
    
    return IntrinsicHSL(h_intrinsic, s_intrinsic, l_intrinsic)
end
