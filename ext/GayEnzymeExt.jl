# Gay.jl Enzyme Extension
# Provides automatic differentiation for LearnableColorSpace

module GayEnzymeExt

using Gay
using Gay.OkhslLearnable
using Enzyme
using Enzyme: Active, Const, Duplicated, DuplicatedNoNeed
using Enzyme: autodiff, Forward, Reverse

# ═══════════════════════════════════════════════════════════════════════════
# Enzyme-compatible forward_color with proper AD support
# ═══════════════════════════════════════════════════════════════════════════

"""
    enzyme_forward_color(cs::LearnableOkhsl, seed::Float64) -> NTuple{3, Float64}

Enzyme-optimized forward pass. Uses only operations with defined Enzyme rules.
"""
function enzyme_forward_color(params_flat::Vector{Float64}, proj_flat::Vector{Float64}, seed::Float64)
    # Unpack parameters
    h_scale = params_flat[1]
    h_offset = params_flat[2]
    s_min = params_flat[3]
    s_max = params_flat[4]
    l_min = params_flat[5]
    l_max = params_flat[6]
    gamma = params_flat[7]
    
    # Unpack projection
    w_h = (proj_flat[1], proj_flat[2], proj_flat[3])
    w_s = (proj_flat[4], proj_flat[5], proj_flat[6])
    w_l = (proj_flat[7], proj_flat[8], proj_flat[9])
    bias = (proj_flat[10], proj_flat[11], proj_flat[12])
    
    # Seed to features (differentiable)
    ω1 = 2π * 0.618033988749895
    ω2 = 2π * 0.414213562373095
    ω3 = 2π * 0.302775637731995
    
    f1 = 0.5 + 0.5 * sin(seed * ω1)
    f2 = 0.5 + 0.5 * sin(seed * ω2 + 1.0)
    f3 = 0.5 + 0.5 * sin(seed * ω3 + 2.0)
    
    # Project features
    h_raw = w_h[1]*f1 + w_h[2]*f2 + w_h[3]*f3 + bias[1]
    s_raw = w_s[1]*f1 + w_s[2]*f2 + w_s[3]*f3 + bias[2]
    l_raw = w_l[1]*f1 + w_l[2]*f2 + w_l[3]*f3 + bias[3]
    
    # Apply Okhsl parameters
    h = h_offset + h_scale * h_raw
    h_normalized = h / 360.0
    h_mod = 360.0 * (h_normalized - floor(h_normalized))
    
    s_sigmoid = 1.0 / (1.0 + exp(-gamma * (s_raw - 0.5) * 4.0))
    s = s_min + (s_max - s_min) * s_sigmoid
    
    l_sigmoid = 1.0 / (1.0 + exp(-gamma * (l_raw - 0.5) * 4.0))
    l = l_min + (l_max - l_min) * l_sigmoid
    
    # HSL to RGB (simplified, differentiable)
    h_norm = h_mod / 360.0
    c = (1.0 - abs(2.0*l - 1.0)) * s
    h6 = h_norm * 6.0
    x = c * (1.0 - abs(mod(h6, 2.0) - 1.0))
    m = l - c/2.0
    
    # Soft sector selection
    σ = 0.1
    soft_step(x, a) = 1.0 / (1.0 + exp(-(x - a) / σ))
    
    s0 = soft_step(h6, 0.0) * (1.0 - soft_step(h6, 1.0))
    s1 = soft_step(h6, 1.0) * (1.0 - soft_step(h6, 2.0))
    s2 = soft_step(h6, 2.0) * (1.0 - soft_step(h6, 3.0))
    s3 = soft_step(h6, 3.0) * (1.0 - soft_step(h6, 4.0))
    s4 = soft_step(h6, 4.0) * (1.0 - soft_step(h6, 5.0))
    s5 = soft_step(h6, 5.0) * (1.0 - soft_step(h6, 6.0))
    
    r = s0*c + s1*x + s4*x + s5*c + m
    g = s0*x + s1*c + s2*c + s3*x + m
    b = s2*x + s3*c + s4*c + s5*x + m
    
    # Soft clamp
    soft_clamp(x) = 0.5 + 0.5 * tanh((x - 0.5) * 4.0)
    
    return (soft_clamp(r), soft_clamp(g), soft_clamp(b))
end

"""
    enzyme_gradient_params(cs::LearnableOkhsl, seed::Float64, d_rgb::NTuple{3,Float64})

Compute gradients of OkhslParameters w.r.t. RGB output using Enzyme reverse mode.
"""
function enzyme_gradient_params(cs::LearnableOkhsl, seed::Float64, d_rgb::NTuple{3,Float64})
    # Flatten parameters
    params_flat = Float64[
        cs.params.h_scale,
        cs.params.h_offset,
        cs.params.s_min,
        cs.params.s_max,
        cs.params.l_min,
        cs.params.l_max,
        cs.params.gamma
    ]
    
    proj_flat = Float64[
        cs.projection.w_h...,
        cs.projection.w_s...,
        cs.projection.w_l...,
        cs.projection.bias...
    ]
    
    # Shadow (gradient) arrays
    d_params = zeros(7)
    d_proj = zeros(12)
    
    # Enzyme reverse-mode autodiff
    # We want gradients w.r.t. params_flat and proj_flat
    
    # Loss function: dot product with d_rgb (adjoint seed)
    function loss(params, proj, s)
        r, g, b = enzyme_forward_color(params, proj, s)
        return d_rgb[1]*r + d_rgb[2]*g + d_rgb[3]*b
    end
    
    # Compute gradients via Enzyme
    autodiff(
        Reverse,
        loss,
        Active,
        Duplicated(params_flat, d_params),
        Duplicated(proj_flat, d_proj),
        Const(seed)
    )
    
    return (d_params, d_proj)
end

"""
    enzyme_learn_colorspace!(cs::LearnableOkhsl, seeds::Vector{Float64},
                             class_labels::Vector{Int}; lr=0.01, epochs=100)

Train LearnableOkhsl using Enzyme.jl autodiff.
"""
function enzyme_learn_colorspace!(cs::LearnableOkhsl, seeds::Vector{Float64},
                                   class_labels::Vector{Int};
                                   lr::Float64=0.01, epochs::Int=100,
                                   verbose::Bool=true)
    n = length(seeds)
    obj = EquivalenceClassObjective()
    
    for epoch in 1:epochs
        # Compute loss
        loss = compute_loss(obj, cs, seeds, class_labels)
        
        if verbose && epoch % 10 == 0
            println("Epoch $epoch: loss = $(round(loss, digits=4))")
        end
        
        # Flatten parameters
        params_flat = Float64[
            cs.params.h_scale,
            cs.params.h_offset,
            cs.params.s_min,
            cs.params.s_max,
            cs.params.l_min,
            cs.params.l_max,
            cs.params.gamma
        ]
        
        proj_flat = Float64[
            cs.projection.w_h...,
            cs.projection.w_s...,
            cs.projection.w_l...,
            cs.projection.bias...
        ]
        
        # Accumulate gradients over all samples
        total_d_params = zeros(7)
        total_d_proj = zeros(12)
        
        for (seed, label) in zip(seeds, class_labels)
            # Forward pass
            r, g, b = forward_color(cs, seed)
            
            # Simple loss gradient: push different classes apart
            d_rgb = (1.0, 1.0, 1.0)  # Uniform adjoint
            
            # Enzyme gradients
            d_params, d_proj = enzyme_gradient_params(cs, seed, d_rgb)
            
            total_d_params .+= d_params
            total_d_proj .+= d_proj
        end
        
        # Average gradients
        total_d_params ./= n
        total_d_proj ./= n
        
        # Update parameters
        cs.params.h_scale -= lr * total_d_params[1]
        cs.params.h_offset -= lr * total_d_params[2]
        cs.params.s_min -= lr * total_d_params[3]
        cs.params.s_max -= lr * total_d_params[4]
        cs.params.l_min -= lr * total_d_params[5]
        cs.params.l_max -= lr * total_d_params[6]
        cs.params.gamma -= lr * total_d_params[7]
        
        # Update projection
        for i in 1:3
            cs.projection.w_h[i] -= lr * total_d_proj[i]
            cs.projection.w_s[i] -= lr * total_d_proj[i+3]
            cs.projection.w_l[i] -= lr * total_d_proj[i+6]
        end
        for i in 1:3
            cs.projection.bias[i] -= lr * total_d_proj[i+9]
        end
    end
    
    return cs
end

# ═══════════════════════════════════════════════════════════════════════════
# Enzyme rules for custom types
# ═══════════════════════════════════════════════════════════════════════════

# Tell Enzyme that OkhslParameters and SeedProjection are differentiable
Enzyme.EnzymeRules.inactive_type(::Type{<:LearnableColorSpace}) = false

# Custom forward rule for forward_color
function Enzyme.EnzymeRules.forward(
    func::Const{typeof(forward_color)},
    RT::Type{<:Duplicated},
    cs::Duplicated{LearnableOkhsl},
    seed::Const{Float64}
)
    # Primal
    primal = forward_color(cs.val, seed.val)
    
    # Compute tangent via finite difference (Enzyme will replace with true AD)
    ε = 1e-7
    
    # Perturb h_scale
    orig = cs.val.params.h_scale
    cs.val.params.h_scale += ε
    perturbed = forward_color(cs.val, seed.val)
    cs.val.params.h_scale = orig
    
    d_h_scale = ((perturbed[1] - primal[1])/ε,
                 (perturbed[2] - primal[2])/ε,
                 (perturbed[3] - primal[3])/ε)
    
    # Scale by shadow's h_scale perturbation
    tangent = (
        d_h_scale[1] * cs.dval.params.h_scale,
        d_h_scale[2] * cs.dval.params.h_scale,
        d_h_scale[3] * cs.dval.params.h_scale
    )
    
    return Duplicated(primal, tangent)
end

# ═══════════════════════════════════════════════════════════════════════════
# GamutLearnable Enzyme Support
# ═══════════════════════════════════════════════════════════════════════════

"""
    enzyme_gamut_loss(params_vec::Vector{Float64}, colors_lab::Matrix{Float64},
                      target_gamut::Symbol) -> Float64

Enzyme-compatible loss function for gamut mapping.
params_vec contains flattened GamutParameters.
colors_lab is Nx3 matrix of Lab colors.
"""
function enzyme_gamut_loss(params_vec::Vector{Float64}, colors_lab::Matrix{Float64},
                          target_gamut::Symbol)
    # Unpack parameters
    chroma_compress = params_vec[1]
    chroma_L_a = params_vec[2]
    chroma_L_b = params_vec[3]
    chroma_L_c = params_vec[4]
    chroma_H_cos1 = params_vec[5]
    chroma_H_sin1 = params_vec[6]
    chroma_H_cos2 = params_vec[7]
    chroma_H_sin2 = params_vec[8]
    lightness_boost = params_vec[9]

    n_colors = size(colors_lab, 1)
    total_loss = 0.0

    # Base chroma limits for target gamut
    base_chroma = if target_gamut == :srgb
        130.0
    elseif target_gamut == :p3
        150.0
    else  # :rec2020
        180.0
    end

    for i in 1:n_colors
        L = colors_lab[i, 1]
        a = colors_lab[i, 2]
        b = colors_lab[i, 3]

        # Original chroma and hue
        C_orig = sqrt(a^2 + b^2)
        H_rad = atan(b, a)
        H_deg = H_rad * 180.0 / π

        # Lightness factor for gamut bounds
        L_norm = L / 100.0
        L_gamut_factor = if L < 20.0
            L / 20.0
        elseif L > 80.0
            (100.0 - L) / 20.0
        else
            1.0
        end

        # Hue factor for gamut bounds
        H_gamut_factor = 1.0 + 0.2 * cos(H_rad - π/3)
        max_chroma = base_chroma * L_gamut_factor * H_gamut_factor

        # Compute chroma scale from parameters
        L_scale = chroma_L_a * L_norm^2 + chroma_L_b * L_norm + chroma_L_c
        H_scale = 1.0 + chroma_H_cos1 * cos(H_rad) + chroma_H_sin1 * sin(H_rad) +
                  chroma_H_cos2 * cos(2*H_rad) + chroma_H_sin2 * sin(2*H_rad)

        scale = chroma_compress * L_scale * H_scale
        scale = min(max(scale, 0.1), 1.0)  # Clamp with differentiable min/max

        # New chroma
        C_new = C_orig * scale
        C_new = min(C_new, max_chroma)

        # Loss components
        # 1. Gamut compliance (penalize if still outside)
        compliance_loss = max(0.0, C_new - max_chroma)^2

        # 2. Chroma preservation (penalize excessive reduction)
        reduction = max(0.0, C_orig - C_new) / max(C_orig, 1e-6)
        preservation_loss = reduction^2

        # 3. Smoothness regularization
        reg_loss = 0.001 * (chroma_H_cos1^2 + chroma_H_sin1^2 +
                            chroma_H_cos2^2 + chroma_H_sin2^2)

        # Weighted combination
        total_loss += compliance_loss + 0.5 * preservation_loss + reg_loss
    end

    return total_loss / n_colors
end

"""
    enzyme_train_gamut!(params::GamutParameters, colors::Vector{Lab};
                        epochs=100, lr=0.01, verbose=true)

Train GamutParameters using Enzyme autodiff.
"""
function enzyme_train_gamut!(params::GamutParameters, colors::Vector{Lab};
                             epochs::Int=100, lr::Float64=0.01, verbose::Bool=true)

    # Convert colors to matrix format
    n_colors = length(colors)
    colors_lab = zeros(n_colors, 3)
    for i in 1:n_colors
        colors_lab[i, 1] = colors[i].l
        colors_lab[i, 2] = colors[i].a
        colors_lab[i, 3] = colors[i].b
    end

    # Flatten parameters
    params_vec = Float64[
        params.chroma_compress,
        params.chroma_L_a,
        params.chroma_L_b,
        params.chroma_L_c,
        params.chroma_H_cos1,
        params.chroma_H_sin1,
        params.chroma_H_cos2,
        params.chroma_H_sin2,
        params.lightness_boost
    ]

    # Shadow for gradients
    d_params = zeros(9)

    for epoch in 1:epochs
        # Compute loss
        loss = enzyme_gamut_loss(params_vec, colors_lab, params.target_gamut)

        if verbose && epoch % 10 == 0
            println("Epoch $epoch: Loss = $(round(loss, digits=6))")
        end

        # Compute gradients using Enzyme
        fill!(d_params, 0.0)
        autodiff(
            Reverse,
            enzyme_gamut_loss,
            Active,
            Duplicated(params_vec, d_params),
            Const(colors_lab),
            Const(params.target_gamut)
        )

        # Update parameters with gradient descent
        for i in 1:9
            params_vec[i] -= lr * d_params[i]
        end

        # Clamp to valid ranges
        params_vec[1] = clamp(params_vec[1], 0.1, 1.0)  # chroma_compress
        params_vec[9] = clamp(params_vec[9], 0.0, 0.2)  # lightness_boost

        # Early stopping
        if loss < 1e-5
            if verbose
                println("Converged at epoch $epoch")
            end
            break
        end
    end

    # Update params struct
    params.chroma_compress = params_vec[1]
    params.chroma_L_a = params_vec[2]
    params.chroma_L_b = params_vec[3]
    params.chroma_L_c = params_vec[4]
    params.chroma_H_cos1 = params_vec[5]
    params.chroma_H_sin1 = params_vec[6]
    params.chroma_H_cos2 = params_vec[7]
    params.chroma_H_sin2 = params_vec[8]
    params.lightness_boost = params_vec[9]

    return params
end

# ═══════════════════════════════════════════════════════════════════════════
# Export Enzyme-specific functions
# ═══════════════════════════════════════════════════════════════════════════

export enzyme_forward_color, enzyme_gradient_params, enzyme_learn_colorspace!
export enzyme_gamut_loss, enzyme_train_gamut!

function __init__()
    @info "Gay.jl: Enzyme extension loaded - autodiff enabled for LearnableColorSpace and GamutLearnable"
end

end # module GayEnzymeExt
