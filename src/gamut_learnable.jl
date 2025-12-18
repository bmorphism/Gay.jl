# GamutLearnable.jl - Enzyme-optimized gamut mapping for Gay.jl color chains
# Implements Issue #184: ML-based color gamut mapping with automatic differentiation

module GamutLearnable

using Colors
using ColorTypes

# Simple mean function to avoid Statistics dependency
mean(x) = sum(x) / length(x)

export GamutParameters, GamutMapper, map_to_gamut, train_gamut_mapper!
export gamut_loss, in_gamut, get_gamut_bounds, map_color_chain, detect_system_gamut

# ═══════════════════════════════════════════════════════════════════════════
# Monitor Detection
# ═══════════════════════════════════════════════════════════════════════════

"""
    detect_system_gamut() -> Symbol

Attempt to detect the system's display gamut.
Returns :p3 for Apple Silicon/Retina displays, :srgb as fallback.
"""
function detect_system_gamut()
    if Sys.isapple()
        try
            # Check system profiler for display info
            cmd = `system_profiler SPDisplaysDataType`
            output = read(cmd, String)
            
            # Check for P3/Retina indicators
            if contains(output, "Retina") || contains(output, "P3") || contains(output, "XDR")
                return :p3
            end
        catch
            # Fallback on error
        end
    end
    
    # Default safe fallback
    return :srgb
end

# ═══════════════════════════════════════════════════════════════════════════
# Gamut Parameters Structure
# ═══════════════════════════════════════════════════════════════════════════

"""
    GamutParameters

Learnable parameters for adaptive gamut mapping.
Controls how colors are compressed to fit within target gamut boundaries.
"""
mutable struct GamutParameters
    # Base chroma compression factor [0.1, 1.0]
    chroma_compress::Float64

    # Lightness-dependent quadratic modulation coefficients
    chroma_L_a::Float64  # Quadratic term
    chroma_L_b::Float64  # Linear term
    chroma_L_c::Float64  # Constant term

    # Hue-dependent Fourier modulation coefficients
    chroma_H_cos1::Float64  # cos(H) coefficient
    chroma_H_sin1::Float64  # sin(H) coefficient
    chroma_H_cos2::Float64  # cos(2H) coefficient
    chroma_H_sin2::Float64  # sin(2H) coefficient

    # Lightness boost for desaturation compensation
    lightness_boost::Float64

    # Target gamut (:srgb, :p3, :rec2020)
    target_gamut::Symbol
end

# Default constructor with sensible initial values
function GamutParameters(; target_gamut::Symbol=detect_system_gamut())
    GamutParameters(
        0.8,      # chroma_compress - moderate compression
        -0.01,    # chroma_L_a - slight quadratic compression at extremes
        0.0,      # chroma_L_b - no linear term initially
        1.0,      # chroma_L_c - baseline multiplier
        0.0,      # chroma_H_cos1
        0.0,      # chroma_H_sin1
        0.0,      # chroma_H_cos2
        0.0,      # chroma_H_sin2
        0.05,     # lightness_boost - slight compensation
        target_gamut
    )
end

# ═══════════════════════════════════════════════════════════════════════════
# Gamut Mapping Functions
# ═══════════════════════════════════════════════════════════════════════════

"""
    in_gamut(c::Lab, gamut::Symbol) -> Bool

Check if a Lab color is strictly within the specified gamut using ColorTypes conversion.
"""
function in_gamut(c::Lab, gamut::Symbol)
    try
        if gamut == :srgb
            rgb = convert(RGB{Float64}, c)
            return 0.0 <= rgb.r <= 1.0 && 0.0 <= rgb.g <= 1.0 && 0.0 <= rgb.b <= 1.0
        elseif gamut == :p3
            p3 = convert(DisplayP3{Float64}, c)
            return 0.0 <= p3.r <= 1.0 && 0.0 <= p3.g <= 1.0 && 0.0 <= p3.b <= 1.0
        else
            # Fallback for Rec2020 etc
            rgb = convert(RGB{Float64}, c) 
            return 0.0 <= rgb.r <= 1.0 && 0.0 <= rgb.g <= 1.0 && 0.0 <= rgb.b <= 1.0
        end
    catch
        return false
    end
end

"""
    get_gamut_bounds(gamut::Symbol, L::Real, H::Real) -> Float64

Find maximum valid chroma for given Lightness and Hue via binary search.
"""
function get_gamut_bounds(gamut::Symbol, L::Real, H::Real)
    # Binary search for max chroma
    # Range [0, 200] covers all practical visible colors
    low = 0.0
    high = 200.0
    
    # Pre-calculate trig
    H_rad = H * π / 180.0
    cos_H = cos(H_rad)
    sin_H = sin(H_rad)
    
    for _ in 1:10 # 10 iterations gives precision ~0.2, sufficient for display
        mid = (low + high) / 2
        
        # Construct test color
        a = mid * cos_H
        b = mid * sin_H
        c_test = Lab(L, a, b)
        
        if in_gamut(c_test, gamut)
            low = mid
        else
            high = mid
        end
    end
    
    return low
end

"""
    compute_chroma_scale(params::GamutParameters, L::Float64, H::Float64) -> Float64

Compute the chroma scaling factor based on lightness and hue.
"""
function compute_chroma_scale(params::GamutParameters, L::Real, H::Real)
    # Normalize L to [0,1]
    L_norm = L / 100.0

    # Lightness-dependent quadratic modulation
    L_factor = params.chroma_L_a * L_norm^2 +
               params.chroma_L_b * L_norm +
               params.chroma_L_c

    # Hue-dependent Fourier modulation
    H_rad = H * π / 180.0
    H_factor = 1.0 +
               params.chroma_H_cos1 * cos(H_rad) +
               params.chroma_H_sin1 * sin(H_rad) +
               params.chroma_H_cos2 * cos(2*H_rad) +
               params.chroma_H_sin2 * sin(2*H_rad)

    # Combine factors with base compression
    scale = params.chroma_compress * L_factor * H_factor

    # Clamp to valid range
    return clamp(scale, 0.1, 1.0)
end

"""
    map_to_gamut(c::Lab, params::GamutParameters) -> Lab

Map a Lab color to fit within the target gamut.
Correctly handles lightness boost by checking bounds after adjustment.
"""
function map_to_gamut(c::Lab, params::GamutParameters)
    L, a, b = c.l, c.a, c.b
    C = sqrt(a^2 + b^2)
    H = atan(b, a) * 180.0 / π
    H = H < 0 ? H + 360.0 : H

    # 1. Calculate max chroma for ORIGINAL lightness
    max_chroma_orig = get_gamut_bounds(params.target_gamut, L, H)

    # 2. Compute adaptive scaling factor
    scale = compute_chroma_scale(params, L, H)
    C_proposed = C * scale
    
    # 3. Calculate tentative new lightness (with boost)
    chroma_reduction = 1.0 - (min(C_proposed, max_chroma_orig) / max(C, 1e-6))
    L_boost = params.lightness_boost * chroma_reduction * (50.0 - abs(L - 50.0))
    L_new = clamp(L + L_boost, 0.0, 100.0)

    # 4. Re-calculate max chroma for NEW lightness
    max_chroma_new = get_gamut_bounds(params.target_gamut, L_new, H)
    
    # 5. Final clamp
    C_final = min(C_proposed, max_chroma_new)

    # Convert back to Lab
    H_rad = H * π / 180.0
    return Lab(L_new, C_final * cos(H_rad), C_final * sin(H_rad))
end

# ═══════════════════════════════════════════════════════════════════════════
# Loss Functions for Training
# ═══════════════════════════════════════════════════════════════════════════

"""
    gamut_compliance_loss(colors::Vector{Lab}, params::GamutParameters) -> Float64

Compute loss for colors that exceed gamut boundaries.
"""
function gamut_compliance_loss(colors::Vector{<:Lab}, params::GamutParameters)
    loss = 0.0
    for c in colors
        mapped = map_to_gamut(c, params)
        if !in_gamut(mapped, params.target_gamut)
            # Penalize out-of-gamut colors
            L, a, b = mapped.l, mapped.a, mapped.b
            C = sqrt(a^2 + b^2)
            H = atan(b, a) * 180.0 / π
            H = H < 0 ? H + 360.0 : H
            max_chroma = get_gamut_bounds(params.target_gamut, L, H)
            excess = max(0, C - max_chroma)
            loss += excess^2
        end
    end
    return loss / length(colors)
end

"""
    chroma_preservation_loss(original::Vector{Lab}, mapped::Vector{Lab}) -> Float64

Compute loss for excessive chroma reduction.
"""
function chroma_preservation_loss(original::Vector{<:Lab}, mapped::Vector{<:Lab})
    loss = 0.0
    for (c_orig, c_mapped) in zip(original, mapped)
        C_orig = sqrt(c_orig.a^2 + c_orig.b^2)
        C_mapped = sqrt(c_mapped.a^2 + c_mapped.b^2)
        # Penalize excessive compression
        reduction = max(0, C_orig - C_mapped) / max(C_orig, 1e-6)
        loss += reduction^2
    end
    return loss / length(original)
end

"""
    hue_preservation_loss(original::Vector{Lab}, mapped::Vector{Lab}) -> Float64

Compute loss for hue shifts (should be zero for our mapping).
"""
function hue_preservation_loss(original::Vector{<:Lab}, mapped::Vector{<:Lab})
    loss = 0.0
    for (c_orig, c_mapped) in zip(original, mapped)
        H_orig = atan(c_orig.b, c_orig.a)
        H_mapped = atan(c_mapped.b, c_mapped.a)
        # Angular difference
        diff = abs(H_mapped - H_orig)
        diff = min(diff, 2π - diff)  # Handle wraparound
        loss += diff^2
    end
    return loss / length(original)
end

"""
    gamut_loss(colors::Vector{Lab}, params::GamutParameters;
               λ_compliance=1.0, λ_preservation=0.5, λ_hue=10.0) -> Float64

Combined loss function for gamut mapping optimization.
"""
function gamut_loss(colors::Vector{<:Lab}, params::GamutParameters;
                    λ_compliance=1.0, λ_preservation=0.5, λ_hue=10.0)
    # Map colors
    mapped = [map_to_gamut(c, params) for c in colors]

    # Compute individual losses
    L_compliance = gamut_compliance_loss(mapped, params)
    L_preservation = chroma_preservation_loss(colors, mapped)
    L_hue = hue_preservation_loss(colors, mapped)

    # Weighted combination
    return λ_compliance * L_compliance +
           λ_preservation * L_preservation +
           λ_hue * L_hue
end

# ═══════════════════════════════════════════════════════════════════════════
# Training Functions (will use Enzyme in extension)
# ═══════════════════════════════════════════════════════════════════════════

"""
    train_gamut_mapper!(params::GamutParameters, colors::Vector{Lab};
                        epochs=100, lr=0.01, verbose=true)

Train the gamut mapping parameters using gradient descent.
This is the non-Enzyme version - the Enzyme extension will override this.
"""
function train_gamut_mapper!(params::GamutParameters, colors::Vector{<:Lab};
                             epochs::Int=100, lr::Float64=0.01, verbose::Bool=true)

    for epoch in 1:epochs
        # Current loss
        current_loss = gamut_loss(colors, params)

        if verbose && epoch % 10 == 0
            println("Epoch $epoch: Loss = $(round(current_loss, digits=4))")
        end

        # Simple finite difference gradients (will be replaced by Enzyme)
        ε = 1e-6
        grad = zeros(9)  # 9 parameters to optimize

        # Gradient for chroma_compress
        params_plus = deepcopy(params)
        params_plus.chroma_compress += ε
        grad[1] = (gamut_loss(colors, params_plus) - current_loss) / ε

        # Similar for other parameters...
        # (In practice, Enzyme will compute these automatically)

        # Update parameters
        params.chroma_compress -= lr * grad[1]
        params.chroma_compress = clamp(params.chroma_compress, 0.1, 1.0)

        # Early stopping if converged
        if current_loss < 1e-4
            if verbose
                println("Converged at epoch $epoch")
            end
            break
        end
    end

    return params
end

# ═══════════════════════════════════════════════════════════════════════════
# Integration with Gay.jl color chains
# ═══════════════════════════════════════════════════════════════════════════

"""
    GamutMapper

Main struct that combines parameters with mapping functionality.
"""
struct GamutMapper
    params::GamutParameters
    trained::Bool
end

GamutMapper(; target_gamut::Symbol=:srgb) =
    GamutMapper(GamutParameters(target_gamut=target_gamut), false)

"""
    map_color_chain(chain::Vector{<:Color}, mapper::GamutMapper)

Map an entire color chain to fit within the target gamut.
"""
function map_color_chain(chain::Vector{<:Color}, mapper::GamutMapper)
    # Convert to Lab space for gamut mapping
    lab_chain = [convert(Lab, c) for c in chain]

    # Apply gamut mapping
    mapped_lab = [map_to_gamut(c, mapper.params) for c in lab_chain]

    # Convert back to original color type
    ColorType = eltype(chain)
    return [convert(ColorType, c) for c in mapped_lab]
end

end # module GamutLearnable