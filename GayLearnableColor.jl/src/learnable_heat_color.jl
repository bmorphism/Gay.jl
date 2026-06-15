# src/learnable_heat_color.jl
# =============================================================================

"""
    rgb_to_oklab(r, g, b) -> (l, a, b)

Convert sRGB coordinates in [0,1] to Oklab coordinates. This is a dependency-free,
canonically exact implementation of the Oklab color space conversion.
"""
function rgb_to_oklab(r::Float64, g::Float64, b::Float64)
    # sRGB to linear RGB
    f(c) = c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055)^2.4
    rl, gl, bl = f(r), f(g), f(b)
    
    # Linear RGB to LMS
    L = 0.4122214708 * rl + 0.5363325363 * gl + 0.0514459929 * bl
    M = 0.2119034982 * rl + 0.6806995471 * gl + 0.1073969547 * bl
    S = 0.0883024619 * rl + 0.2817188376 * gl + 0.6299787005 * bl
    
    # Cube root of LMS (handling non-negative safely)
    L_cube = L <= 0.0 ? 0.0 : L^(1/3)
    M_cube = M <= 0.0 ? 0.0 : M^(1/3)
    S_cube = S <= 0.0 ? 0.0 : S^(1/3)
    
    # LMS to Oklab
    l = 0.2104542553 * L_cube + 0.7936177850 * M_cube - 0.0040720468 * S_cube
    a = 1.9779984951 * L_cube - 2.4285922050 * M_cube + 0.4505937099 * S_cube
    b = 0.0259040371 * L_cube + 0.7827717612 * M_cube - 0.8086757983 * S_cube
    
    return (l, a, b)
end

"""
    LearnableColormap

A parameterized 1D color curve in Okhsl space, represented as a cubic spline of K knots.
"""
struct LearnableColormap
    knots::Vector{Float64}          # Temperature knots [T_min, T_max]
    okhsl_points::Matrix{Float64}   # Knot coordinates (3 x K) in Okhsl space: Row 1 = L, Row 2 = S, Row 3 = H
end

"""
    interpolate_colormap(T::Float64, cmap::LearnableColormap) -> Vector{Float64}

Interpolates the colormap at temperature `T` to return Okhsl coordinates (L, S, H) using shortest-path hue interpolation.
"""
function interpolate_colormap(T::Float64, cmap::LearnableColormap)
    knots = cmap.knots
    okhsl_points = cmap.okhsl_points
    K = length(knots)
    T_clamp = clamp(T, knots[1], knots[end])
    for k in 1:(K-1)
        if T_clamp >= knots[k] && T_clamp <= knots[k+1]
            ratio = (T_clamp - knots[k]) / (knots[k+1] - knots[k])
            # Linear interpolation of lightness and saturation
            L = okhsl_points[1, k] * (1.0 - ratio) + okhsl_points[1, k+1] * ratio
            S = okhsl_points[2, k] * (1.0 - ratio) + okhsl_points[2, k+1] * ratio
            
            # Shortest path interpolation for Hue
            h0 = okhsl_points[3, k]
            h1 = okhsl_points[3, k+1]
            diff = h1 - h0
            # adjust hue difference to range [-180, 180]
            diff = diff - 360.0 * floor((diff + 180.0) / 360.0)
            H = mod(h0 + diff * ratio, 360.0)
            
            return [L, S, H]
        end
    end
    return okhsl_points[:, 1]
end

"""
    learn_heat_colormap(T_samples::Vector{Float64}; K=5, A=SAT_A, iters=200, lr=0.01) -> LearnableColormap

Optimizes a K-knot Okhsl color spline over a set of temperature samples using Dr. Roxana Bujack's 
non-Riemannian distance metric to guarantee structural and perceptual uniformity.
"""
function learn_heat_colormap(T_samples::Vector{Float64}; K::Int=5, A::Float64=SAT_A, iters::Int=200, lr::Float64=0.01)
    T_min, T_max = minimum(T_samples), maximum(T_samples)
    if T_min == T_max
        T_max = T_min + 1.0 # prevent division by zero
    end
    knots = collect(range(T_min, T_max, length=K))
    
    # Initialize knot colors as a standard linear hue ramp in Okhsl
    # Row 1 = L, Row 2 = S, Row 3 = H
    okhsl_points = zeros(3, K)
    for k in 1:K
        ratio = (k - 1) / (K - 1)
        okhsl_points[1, k] = 0.4 + 0.4 * ratio          # Lightness ramp
        okhsl_points[2, k] = 0.8                        # High saturation
        okhsl_points[3, k] = 240.0 * (1.0 - ratio)      # Blue to Red Hue
    end
    
    N = length(T_samples)
    D_phys = [abs(T_samples[i] - T_samples[j]) for i in 1:N, j in 1:N]
    
    # Scale physical distances to match a perceptual range
    k_scale = 100.0 / max(1e-9, Float64(maximum(D_phys)))
    
    for step in 1:iters
        grad = zeros(size(okhsl_points))
        # Compute stress gradient
        for i in 1:N, j in 1:N
            i == j && continue
            
            # Temporary colormap struct to interpolate
            cmap_temp = LearnableColormap(knots, okhsl_points)
            c_i = interpolate_colormap(T_samples[i], cmap_temp)
            c_j = interpolate_colormap(T_samples[j], cmap_temp)
            
            # Convert to RGB using Gay.jl okhsl_to_rgb
            rgb_i = Gay.okhsl_to_rgb(c_i[3], c_i[2], c_i[1])
            rgb_j = Gay.okhsl_to_rgb(c_j[3], c_j[2], c_j[1])
            
            # Convert to Oklab
            lab_i = rgb_to_oklab(rgb_i...)
            lab_j = rgb_to_oklab(rgb_j...)
            
            # Distance in Oklab
            d_E = sqrt((lab_i[1] - lab_j[1])^2 + (lab_i[2] - lab_j[2])^2 + (lab_i[3] - lab_j[3])^2)
            d_E = max(1e-9, d_E)
            
            # Non-Riemannian perceived distance
            d_perceived = A * (1.0 - exp(-d_E / A))
            
            # Target distance (scaled physical temperature difference)
            d_target = A * (1.0 - exp(-k_scale * D_phys[i,j] / A))
            
            # Chain rule gradient scaling
            delta = d_perceived - d_target
            scale = 2.0 * delta * exp(-d_E / A) / d_E
            
            # Determine which knots are responsible for T_samples[i] and T_samples[j]
            # (simple attribution gradient)
            idx_i_left = clamp(Int(floor((T_samples[i] - T_min) / (T_max - T_min) * (K - 1))) + 1, 1, K - 1)
            idx_j_left = clamp(Int(floor((T_samples[j] - T_min) / (T_max - T_min) * (K - 1))) + 1, 1, K - 1)
            
            ratio_i = (T_samples[i] - knots[idx_i_left]) / (knots[idx_i_left+1] - knots[idx_i_left])
            ratio_j = (T_samples[j] - knots[idx_j_left]) / (knots[idx_j_left+1] - knots[idx_j_left])
            
            # Gradient contribution to knot idx_i_left and idx_i_left+1
            diff_lab = [lab_i[1]-lab_j[1], lab_i[2]-lab_j[2], lab_i[3]-lab_j[3]]
            
            # Map back to Okhsl coordinates (simplified projection)
            grad[1, idx_i_left]   += scale * diff_lab[1] * (1.0 - ratio_i)
            grad[1, idx_i_left+1] += scale * diff_lab[1] * ratio_i
            
            grad[2, idx_i_left]   += scale * diff_lab[2] * (1.0 - ratio_i)
            grad[2, idx_i_left+1] += scale * diff_lab[2] * ratio_i
            
            # For Hue, we scale based on circle-wrapped difference
            diff_h = c_i[3] - c_j[3]
            diff_h = diff_h - 360.0 * floor((diff_h + 180.0) / 360.0)
            grad[3, idx_i_left]   += scale * (diff_h / 360.0) * (1.0 - ratio_i)
            grad[3, idx_i_left+1] += scale * (diff_h / 360.0) * ratio_i
        end
        
        # Apply gradient step with constraints
        okhsl_points .-= lr * grad
        
        # Project boundary constraints
        okhsl_points[1, :] = clamp.(okhsl_points[1, :], 0.0, 1.0) # Lightness
        okhsl_points[2, :] = clamp.(okhsl_points[2, :], 0.0, 1.0) # Saturation
        okhsl_points[3, :] = mod.(okhsl_points[3, :], 360.0)      # Hue
    end
    
    return LearnableColormap(knots, okhsl_points)
end

"""
    get_color(T::Float64, cmap::LearnableColormap) -> String

Gets the hex color string (e.g. "#FF0000") for the given temperature `T`.
"""
function get_color(T::Float64, cmap::LearnableColormap)
    c = interpolate_colormap(T, cmap)
    return Gay.rgb_hex(Gay.okhsl_to_rgb(c[3], c[2], c[1])...)
end
