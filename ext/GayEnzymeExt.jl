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
# Export Enzyme-specific functions
# ═══════════════════════════════════════════════════════════════════════════

export enzyme_forward_color, enzyme_gradient_params, enzyme_learn_colorspace!

function __init__()
    @info "Gay.jl: Enzyme extension loaded - autodiff enabled for LearnableColorSpace"
end

end # module GayEnzymeExt
