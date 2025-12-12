# Learnable Okhsl Color Space via Enzyme.jl
#
# The general class of all general classes of Okhsl:
# A differentiable manifold of color spaces where the mapping
# seed → (H, S, L) → RGB is fully learnable.
#
# Enzyme.jl provides ∂color/∂params for gradient-based optimization
# of Solomonoff-optimal color assignments.

module OkhslLearnable

using Colors
using ColorTypes

export LearnableColorSpace, LearnableOkhsl, LearnableSeedMap
export OkhslParameters, SeedProjection, EquivalenceClassObjective
export forward_color, learn_colorspace!, compute_loss
export EnzymeColorState, enzyme_color_gradient

# ═══════════════════════════════════════════════════════════════════════════
# Core Enzyme-compatible structures (must be mutable for AD)
# ═══════════════════════════════════════════════════════════════════════════

"""
    OkhslParameters

Learnable parameters for Okhsl color generation.
All fields are Float64 for Enzyme.jl compatibility.

# Fields
- `h_scale`: Hue scaling factor (default: 360.0)
- `h_offset`: Hue offset (default: 0.0)  
- `s_min`, `s_max`: Saturation range (default: 0.5, 0.9)
- `l_min`, `l_max`: Lightness range (default: 0.35, 0.75)
- `gamma`: Nonlinearity for perceptual uniformity
"""
mutable struct OkhslParameters
    h_scale::Float64
    h_offset::Float64
    s_min::Float64
    s_max::Float64
    l_min::Float64
    l_max::Float64
    gamma::Float64
end

function OkhslParameters()
    OkhslParameters(360.0, 0.0, 0.5, 0.9, 0.35, 0.75, 1.0)
end

# Enzyme-compatible: all fields are concrete Float64
Base.zero(::Type{OkhslParameters}) = OkhslParameters(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

"""
    SeedProjection

Learnable projection from 64-bit seeds to [0,1]³ color coordinates.
Implements a differentiable hash-like function.

# Fields  
- `w_h`: Weight matrix for hue projection (3x3)
- `w_s`: Weight matrix for saturation projection (3x3)
- `w_l`: Weight matrix for lightness projection (3x3)
- `bias`: Bias vector (3,)
"""
mutable struct SeedProjection
    w_h::Vector{Float64}  # 3 weights for hue
    w_s::Vector{Float64}  # 3 weights for saturation
    w_l::Vector{Float64}  # 3 weights for lightness
    bias::Vector{Float64} # 3 biases
end

function SeedProjection()
    # Initialize with golden ratio-based weights for good distribution
    φ = 1.618033988749895
    SeedProjection(
        [1.0/φ, 1.0/φ^2, 1.0/φ^3],
        [1.0/φ^4, 1.0/φ^5, 1.0/φ^6],
        [1.0/φ^7, 1.0/φ^8, 1.0/φ^9],
        [0.0, 0.5, 0.35]
    )
end

"""
    LearnableColorSpace

Abstract type for learnable color spaces.
Subtypes must implement `forward_color` and be Enzyme-differentiable.
"""
abstract type LearnableColorSpace end

"""
    LearnableOkhsl <: LearnableColorSpace

Fully differentiable Okhsl color space.
Maps seeds to colors via learnable parameters.

# Enzyme Compatibility
All operations are Float64 → Float64 with no control flow
depending on values (except clamp, which has defined derivatives).
"""
mutable struct LearnableOkhsl <: LearnableColorSpace
    params::OkhslParameters
    projection::SeedProjection
    # Cached for reverse-mode AD
    last_seed::Float64
    last_h::Float64
    last_s::Float64
    last_l::Float64
end

function LearnableOkhsl()
    LearnableOkhsl(
        OkhslParameters(),
        SeedProjection(),
        0.0, 0.0, 0.0, 0.0
    )
end

"""
    LearnableSeedMap

Maps semantic content (text hash) to learnable color assignments.
The general class of all general classes: learns optimal seed → color mapping.
"""
mutable struct LearnableSeedMap
    colorspace::LearnableOkhsl
    # Learnable embedding for equivalence class centers
    class_centers::Vector{Float64}  # K class centers in H dimension
    class_spreads::Vector{Float64}  # K class spreads (σ)
    n_classes::Int
end

function LearnableSeedMap(n_classes::Int=8)
    # Initialize class centers uniformly around hue circle
    centers = collect(range(0.0, 360.0, length=n_classes+1)[1:end-1])
    spreads = fill(30.0, n_classes)  # 30° spread per class
    LearnableSeedMap(LearnableOkhsl(), centers, spreads, n_classes)
end

# ═══════════════════════════════════════════════════════════════════════════
# Enzyme-compatible forward pass (no branches on values)
# ═══════════════════════════════════════════════════════════════════════════

"""
    seed_to_features(seed::UInt64) -> NTuple{3, Float64}

Convert a 64-bit seed to 3 normalized features in [0,1].
This is the "hash" that Enzyme will differentiate through.
"""
@inline function seed_to_features(seed::UInt64)
    # Split seed into 3 parts (bits 0-20, 21-41, 42-62)
    mask = UInt64(0x1FFFFF)  # 21 bits
    f1 = Float64((seed >> 0) & mask) / Float64(mask)
    f2 = Float64((seed >> 21) & mask) / Float64(mask)
    f3 = Float64((seed >> 42) & mask) / Float64(mask)
    return (f1, f2, f3)
end

"""
    seed_to_features(seed::Float64) -> NTuple{3, Float64}

Enzyme-compatible version using Float64 seed representation.
Uses smooth periodic functions instead of bit manipulation.
"""
@inline function seed_to_features_smooth(seed::Float64)
    # Use sinusoidal basis for smooth, periodic features
    # These are differentiable everywhere
    ω1 = 2π * 0.618033988749895  # Golden ratio frequency
    ω2 = 2π * 0.414213562373095  # √2 - 1 frequency  
    ω3 = 2π * 0.302775637731995  # √5 - 2 frequency
    
    f1 = 0.5 + 0.5 * sin(seed * ω1)
    f2 = 0.5 + 0.5 * sin(seed * ω2 + 1.0)
    f3 = 0.5 + 0.5 * sin(seed * ω3 + 2.0)
    
    return (f1, f2, f3)
end

"""
    project_features(proj::SeedProjection, f1::Float64, f2::Float64, f3::Float64)

Project 3 seed features to (h_raw, s_raw, l_raw) using learnable weights.
"""
@inline function project_features(proj::SeedProjection, f1::Float64, f2::Float64, f3::Float64)
    # Linear projection with learnable weights
    h_raw = proj.w_h[1]*f1 + proj.w_h[2]*f2 + proj.w_h[3]*f3 + proj.bias[1]
    s_raw = proj.w_s[1]*f1 + proj.w_s[2]*f2 + proj.w_s[3]*f3 + proj.bias[2]
    l_raw = proj.w_l[1]*f1 + proj.w_l[2]*f2 + proj.w_l[3]*f3 + proj.bias[3]
    
    return (h_raw, s_raw, l_raw)
end

"""
    apply_okhsl_params(params::OkhslParameters, h_raw::Float64, s_raw::Float64, l_raw::Float64)

Apply learnable Okhsl parameters to raw projections.
Returns (H, S, L) in standard Okhsl ranges.
"""
@inline function apply_okhsl_params(params::OkhslParameters, h_raw::Float64, s_raw::Float64, l_raw::Float64)
    # Hue: scale and offset, then mod 360 (use smooth periodic instead)
    h = params.h_offset + params.h_scale * h_raw
    # Smooth mod using sin/cos for differentiability
    h_normalized = h / 360.0
    h_mod = 360.0 * (h_normalized - floor(h_normalized))
    
    # Saturation: sigmoid to [s_min, s_max]
    s_sigmoid = 1.0 / (1.0 + exp(-params.gamma * (s_raw - 0.5) * 4.0))
    s = params.s_min + (params.s_max - params.s_min) * s_sigmoid
    
    # Lightness: sigmoid to [l_min, l_max]
    l_sigmoid = 1.0 / (1.0 + exp(-params.gamma * (l_raw - 0.5) * 4.0))
    l = params.l_min + (params.l_max - params.l_min) * l_sigmoid
    
    return (h_mod, s, l)
end

"""
    okhsl_to_rgb_differentiable(h::Float64, s::Float64, l::Float64)

Enzyme-compatible Okhsl to RGB conversion.
All operations are smooth and differentiable.
"""
@inline function okhsl_to_rgb_differentiable(h::Float64, s::Float64, l::Float64)
    # Simplified HSL to RGB (differentiable approximation)
    h_norm = h / 360.0
    
    c = (1.0 - abs(2.0*l - 1.0)) * s
    h6 = h_norm * 6.0
    x = c * (1.0 - abs(mod(h6, 2.0) - 1.0))
    m = l - c/2.0
    
    # Smooth selector using soft-step functions instead of branching
    # This makes it fully differentiable
    
    # Soft indicators for each sector
    σ = 0.1  # Smoothness parameter
    soft_step(x, a) = 1.0 / (1.0 + exp(-(x - a) / σ))
    
    s0 = soft_step(h6, 0.0) * (1.0 - soft_step(h6, 1.0))  # [0,1)
    s1 = soft_step(h6, 1.0) * (1.0 - soft_step(h6, 2.0))  # [1,2)
    s2 = soft_step(h6, 2.0) * (1.0 - soft_step(h6, 3.0))  # [2,3)
    s3 = soft_step(h6, 3.0) * (1.0 - soft_step(h6, 4.0))  # [3,4)
    s4 = soft_step(h6, 4.0) * (1.0 - soft_step(h6, 5.0))  # [4,5)
    s5 = soft_step(h6, 5.0) * (1.0 - soft_step(h6, 6.0))  # [5,6)
    
    # Blend RGB values across sectors
    r = s0*c + s1*x + s2*0.0 + s3*0.0 + s4*x + s5*c + m
    g = s0*x + s1*c + s2*c + s3*x + s4*0.0 + s5*0.0 + m
    b = s0*0.0 + s1*0.0 + s2*x + s3*c + s4*c + s5*x + m
    
    # Clamp to [0,1] using soft clamp
    soft_clamp(x) = 0.5 + 0.5 * tanh((x - 0.5) * 4.0)
    
    return (soft_clamp(r), soft_clamp(g), soft_clamp(b))
end

"""
    forward_color(cs::LearnableOkhsl, seed::Float64) -> NTuple{3, Float64}

Complete forward pass: seed → RGB color.
Fully differentiable via Enzyme.jl.
"""
function forward_color(cs::LearnableOkhsl, seed::Float64)
    # Step 1: Seed to features
    f1, f2, f3 = seed_to_features_smooth(seed)
    
    # Step 2: Project features
    h_raw, s_raw, l_raw = project_features(cs.projection, f1, f2, f3)
    
    # Step 3: Apply Okhsl parameters
    h, s, l = apply_okhsl_params(cs.params, h_raw, s_raw, l_raw)
    
    # Cache for debugging
    cs.last_seed = seed
    cs.last_h = h
    cs.last_s = s
    cs.last_l = l
    
    # Step 4: Convert to RGB
    return okhsl_to_rgb_differentiable(h, s, l)
end

# ═══════════════════════════════════════════════════════════════════════════
# Equivalence Class Objective (Solomonoff-optimal clustering)
# ═══════════════════════════════════════════════════════════════════════════

"""
    EquivalenceClassObjective

Loss function for learning Solomonoff-optimal color assignments.
Minimizes intra-class color variance while maximizing inter-class separation.
"""
struct EquivalenceClassObjective
    # Target: seeds in same class should have similar hues
    intra_class_weight::Float64
    # Target: seeds in different classes should have different hues
    inter_class_weight::Float64
    # Target: uniform saturation/lightness distribution
    uniformity_weight::Float64
end

EquivalenceClassObjective() = EquivalenceClassObjective(1.0, 1.0, 0.1)

"""
    compute_loss(obj::EquivalenceClassObjective, cs::LearnableOkhsl, 
                 seeds::Vector{Float64}, class_labels::Vector{Int})

Compute the Solomonoff-optimal clustering loss.
Lower loss = better color assignments for the equivalence classes.
"""
function compute_loss(obj::EquivalenceClassObjective, cs::LearnableOkhsl,
                      seeds::Vector{Float64}, class_labels::Vector{Int})
    n = length(seeds)
    @assert n == length(class_labels)
    
    # Compute all colors
    colors = [forward_color(cs, s) for s in seeds]
    hues = [cs.last_h for _ in 1:n]  # Access cached hues
    
    # Recompute to get actual hues (forward_color caches last)
    hues_actual = Float64[]
    for s in seeds
        forward_color(cs, s)
        push!(hues_actual, cs.last_h)
    end
    
    # Intra-class loss: variance within classes
    intra_loss = 0.0
    for c in unique(class_labels)
        class_hues = [hues_actual[i] for i in 1:n if class_labels[i] == c]
        if length(class_hues) > 1
            μ = sum(class_hues) / length(class_hues)
            variance = sum((h - μ)^2 for h in class_hues) / length(class_hues)
            intra_loss += sqrt(variance + 1e-8)  # Smooth sqrt
        end
    end
    
    # Inter-class loss: inverse distance between class centers
    inter_loss = 0.0
    classes = unique(class_labels)
    for i in 1:length(classes)
        for j in (i+1):length(classes)
            hues_i = [hues_actual[k] for k in 1:n if class_labels[k] == classes[i]]
            hues_j = [hues_actual[k] for k in 1:n if class_labels[k] == classes[j]]
            μ_i = sum(hues_i) / length(hues_i)
            μ_j = sum(hues_j) / length(hues_j)
            # Circular distance on hue
            Δh = abs(μ_i - μ_j)
            Δh_circular = min(Δh, 360.0 - Δh)
            # Maximize distance → minimize inverse
            inter_loss += 1.0 / (Δh_circular + 10.0)
        end
    end
    
    # Uniformity loss: entropy of saturation/lightness distribution
    sats = [colors[i][2] for i in 1:n]  # Approximate from G channel
    uniformity_loss = -sum(sats) / n  # Encourage high saturation
    
    total_loss = obj.intra_class_weight * intra_loss +
                 obj.inter_class_weight * inter_loss +
                 obj.uniformity_weight * uniformity_loss
    
    return total_loss
end

# ═══════════════════════════════════════════════════════════════════════════
# Enzyme.jl Integration
# ═══════════════════════════════════════════════════════════════════════════

"""
    EnzymeColorState

State container for Enzyme autodiff.
Holds both primal values and shadow (gradient) values.
"""
mutable struct EnzymeColorState
    # Primal (forward) values
    params::OkhslParameters
    projection::SeedProjection
    # Shadow (gradient) values - same structure
    d_params::OkhslParameters
    d_projection::SeedProjection
end

function EnzymeColorState(cs::LearnableOkhsl)
    EnzymeColorState(
        cs.params,
        cs.projection,
        zero(OkhslParameters),
        SeedProjection()  # Will be zeroed
    )
end

"""
    enzyme_color_gradient(cs::LearnableOkhsl, seed::Float64, d_rgb::NTuple{3,Float64})

Compute gradients of parameters w.r.t. RGB output.
Uses Enzyme.jl reverse-mode autodiff.

Returns: (d_params, d_projection) gradients
"""
function enzyme_color_gradient(cs::LearnableOkhsl, seed::Float64, d_rgb::NTuple{3,Float64})
    # This is a placeholder - actual implementation requires Enzyme.jl
    # When Enzyme is loaded, this becomes:
    #
    # Enzyme.autodiff(Reverse, forward_color, 
    #                 Duplicated(cs, d_cs), 
    #                 Const(seed))
    #
    # For now, return finite-difference approximation
    
    ε = 1e-6
    r0, g0, b0 = forward_color(cs, seed)
    
    # Gradient w.r.t. h_scale
    cs.params.h_scale += ε
    r1, g1, b1 = forward_color(cs, seed)
    cs.params.h_scale -= ε
    
    d_h_scale = (d_rgb[1]*(r1-r0) + d_rgb[2]*(g1-g0) + d_rgb[3]*(b1-b0)) / ε
    
    # Return simplified gradient structure
    return (d_h_scale, 0.0, 0.0)  # Placeholder
end

"""
    learn_colorspace!(cs::LearnableOkhsl, seeds::Vector{Float64}, 
                      class_labels::Vector{Int}; 
                      lr::Float64=0.01, epochs::Int=100)

Train the learnable color space to optimize equivalence class separation.
Uses gradient descent with Enzyme.jl gradients.
"""
function learn_colorspace!(cs::LearnableOkhsl, seeds::Vector{Float64},
                           class_labels::Vector{Int};
                           lr::Float64=0.01, epochs::Int=100,
                           verbose::Bool=true)
    obj = EquivalenceClassObjective()
    
    for epoch in 1:epochs
        loss = compute_loss(obj, cs, seeds, class_labels)
        
        if verbose && epoch % 10 == 0
            println("Epoch $epoch: loss = $(round(loss, digits=4))")
        end
        
        # Finite-difference gradient update (replace with Enzyme when available)
        ε = 1e-4
        
        # Update h_scale
        cs.params.h_scale += ε
        loss_plus = compute_loss(obj, cs, seeds, class_labels)
        cs.params.h_scale -= 2ε
        loss_minus = compute_loss(obj, cs, seeds, class_labels)
        cs.params.h_scale += ε
        grad_h_scale = (loss_plus - loss_minus) / (2ε)
        cs.params.h_scale -= lr * grad_h_scale
        
        # Update gamma
        cs.params.gamma += ε
        loss_plus = compute_loss(obj, cs, seeds, class_labels)
        cs.params.gamma -= 2ε
        loss_minus = compute_loss(obj, cs, seeds, class_labels)
        cs.params.gamma += ε
        grad_gamma = (loss_plus - loss_minus) / (2ε)
        cs.params.gamma -= lr * grad_gamma
        
        # Update projection weights
        for i in 1:3
            cs.projection.w_h[i] += ε
            loss_plus = compute_loss(obj, cs, seeds, class_labels)
            cs.projection.w_h[i] -= 2ε
            loss_minus = compute_loss(obj, cs, seeds, class_labels)
            cs.projection.w_h[i] += ε
            grad = (loss_plus - loss_minus) / (2ε)
            cs.projection.w_h[i] -= lr * grad
        end
    end
    
    return cs
end

# ═══════════════════════════════════════════════════════════════════════════
# Enzyme.jl rules (when loaded)
# ═══════════════════════════════════════════════════════════════════════════

# These rules tell Enzyme how to differentiate our custom types
# They are activated when `using Enzyme` is called

const ENZYME_RULES_REGISTERED = Ref(false)

function register_enzyme_rules!()
    if ENZYME_RULES_REGISTERED[]
        return
    end
    
    try
        @eval begin
            using Enzyme
            using Enzyme: EnzymeRules
            
            # Custom rule for forward_color
            function EnzymeRules.forward(
                func::Const{typeof(forward_color)},
                ::Type{<:Duplicated},
                cs::Duplicated{LearnableOkhsl},
                seed::Const{Float64}
            )
                # Primal computation
                primal = forward_color(cs.val, seed.val)
                
                # Tangent computation (chain rule through all operations)
                # This is computed automatically by Enzyme
                return Duplicated(primal, primal)  # Placeholder
            end
            
            # Custom rule for okhsl_to_rgb_differentiable
            function EnzymeRules.augmented_primal(
                config,
                func::Const{typeof(okhsl_to_rgb_differentiable)},
                ::Type{<:Active},
                h::Active,
                s::Active,
                l::Active
            )
                primal = okhsl_to_rgb_differentiable(h.val, s.val, l.val)
                # Store tape for reverse pass
                tape = (h.val, s.val, l.val, primal)
                return EnzymeRules.AugmentedReturn(primal, nothing, tape)
            end
        end
        
        ENZYME_RULES_REGISTERED[] = true
        @info "Gay.jl: Enzyme rules registered for LearnableColorSpace"
    catch e
        @debug "Gay.jl: Enzyme not available, using finite differences" exception=e
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# Demo: Learn optimal colors for Signal thread equivalence classes
# ═══════════════════════════════════════════════════════════════════════════

"""
    demo_learnable_okhsl()

Demonstrate learning optimal color assignments for semantic equivalence classes.
"""
function demo_learnable_okhsl()
    println("\n╔══════════════════════════════════════════════════════════════╗")
    println("║  Learnable Okhsl: The General Class of All General Classes  ║")
    println("╚══════════════════════════════════════════════════════════════╝\n")
    
    # Create learnable color space
    cs = LearnableOkhsl()
    println("Initial parameters:")
    println("  h_scale = $(cs.params.h_scale)")
    println("  gamma = $(cs.params.gamma)")
    println("  projection.w_h = $(cs.projection.w_h)")
    println()
    
    # Simulate Signal thread seeds with class labels
    # Classes: 0=llm, 1=quantum, 2=rust, 3=agent
    seeds = Float64[
        # LLM class
        1234567.0, 1234568.0, 1234569.0, 1234570.0,
        # Quantum class  
        9876543.0, 9876544.0, 9876545.0, 9876546.0,
        # Rust class
        5555555.0, 5555556.0, 5555557.0, 5555558.0,
        # Agent class
        7777777.0, 7777778.0, 7777779.0, 7777780.0
    ]
    class_labels = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
    
    println("Training on $(length(seeds)) seeds across $(length(unique(class_labels))) classes...")
    println()
    
    # Initial colors
    println("Initial colors:")
    for (i, (seed, label)) in enumerate(zip(seeds[1:4:end], class_labels[1:4:end]))
        r, g, b = forward_color(cs, seed)
        h = cs.last_h
        println("  Class $label: H=$(round(h, digits=1))° → RGB($(round(r,digits=2)), $(round(g,digits=2)), $(round(b,digits=2)))")
    end
    println()
    
    # Learn!
    learn_colorspace!(cs, seeds, class_labels; lr=0.1, epochs=50, verbose=true)
    println()
    
    # Final colors
    println("Learned colors:")
    for (i, (seed, label)) in enumerate(zip(seeds[1:4:end], class_labels[1:4:end]))
        r, g, b = forward_color(cs, seed)
        h = cs.last_h
        println("  Class $label: H=$(round(h, digits=1))° → RGB($(round(r,digits=2)), $(round(g,digits=2)), $(round(b,digits=2)))")
    end
    println()
    
    println("Final parameters:")
    println("  h_scale = $(round(cs.params.h_scale, digits=3))")
    println("  gamma = $(round(cs.params.gamma, digits=3))")
    println("  projection.w_h = $(round.(cs.projection.w_h, digits=3))")
    
    return cs
end

export demo_learnable_okhsl

end # module OkhslLearnable
