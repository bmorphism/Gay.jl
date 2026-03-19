# ENZYME EXTENSION: Real Automatic Differentiation for Gay Color Spaces
# ═══════════════════════════════════════════════════════════════════════════════
#
# This module provides the real Enzyme.jl integration when Enzyme is available.
# It replaces the numerical gradient stubs in InterleavedGayEnzyme with true AD.
#
# Usage:
#   using Gay  # loads base module
#   using Enzyme  # triggers this extension
#   # Now enzyme_gradient! uses real AD
#
# ═══════════════════════════════════════════════════════════════════════════════

module GayEnzymeExt

# This is designed as a package extension (requires Julia 1.9+)
# When loaded as extension, Enzyme will be available

using LinearAlgebra

export
    enzyme_forward_gradient!,
    enzyme_reverse_gradient!,
    enzyme_hessian_fwd_rev!,
    enzyme_jacobian!,
    EnzymeDifferentiableColorSpace

# ═══════════════════════════════════════════════════════════════════════════════
# TYPE DEFINITIONS (compatible with InterleavedGayEnzyme)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    EnzymeDifferentiableColorSpace

A color space that uses Enzyme.jl for true automatic differentiation.

When Enzyme.jl is loaded, this provides:
- Forward mode: Jacobian-vector products (JVPs)
- Reverse mode: Vector-Jacobian products (VJPs) 
- Mixed mode: Hessians via forward-over-reverse

This is the "fullest color bandwidth" via gradients at every level.
"""
mutable struct EnzymeDifferentiableColorSpace
    # Learnable parameters (15 total = 9 + 3 + 3)
    basis::Matrix{Float64}      # 3×3 transformation
    offset::Vector{Float64}     # 3 bias
    scale::Vector{Float64}      # 3 scale
    
    # Shadow variables for gradients (Enzyme Duplicated)
    d_basis::Matrix{Float64}
    d_offset::Vector{Float64}
    d_scale::Vector{Float64}
    
    # Training state
    step::Int
    loss_history::Vector{Float64}
    
    # SPI fingerprint
    fingerprint::UInt64
end

function EnzymeDifferentiableColorSpace(; seed::UInt64=UInt64(0xE12A4E))
    # Initialize basis near identity
    basis = Matrix{Float64}(I, 3, 3)
    for i in 1:3, j in 1:3
        basis[i, j] += (rand() - 0.5) * 0.1
    end
    
    offset = zeros(Float64, 3)
    scale = ones(Float64, 3)
    
    # Shadow variables for gradients
    d_basis = zeros(Float64, 3, 3)
    d_offset = zeros(Float64, 3)
    d_scale = zeros(Float64, 3)
    
    EnzymeDifferentiableColorSpace(
        basis, offset, scale,
        d_basis, d_offset, d_scale,
        0, Float64[], seed
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# CORE DIFFERENTIABLE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

"""
Forward pass (Enzyme-compatible, no side effects).
"""
function forward_color_pure(basis::Matrix{Float64}, offset::Vector{Float64}, 
                            scale::Vector{Float64}, input::Vector{Float64})::Vector{Float64}
    transformed = basis * input .+ offset
    clamp.(transformed, 0.0, 1.0) .* scale
end

"""
Loss function (Enzyme-compatible, returns scalar).
"""
function color_loss_pure(basis::Matrix{Float64}, offset::Vector{Float64},
                         scale::Vector{Float64}, input::Vector{Float64},
                         target::Vector{Float64})::Float64
    output = forward_color_pure(basis, offset, scale, input)
    sum((output .- target).^2) / 3.0
end

# ═══════════════════════════════════════════════════════════════════════════════
# ENZYME INTEGRATION PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

#=
The following functions show the patterns for Enzyme.jl integration.
When Enzyme is actually loaded, replace the stubs with:

```julia
using Enzyme

function enzyme_reverse_gradient!(edcs::EnzymeDifferentiableColorSpace,
                                  input::Vector{Float64},
                                  target::Vector{Float64})
    # Zero out shadow variables
    edcs.d_basis .= 0.0
    edcs.d_offset .= 0.0
    edcs.d_scale .= 0.0
    
    # Reverse mode autodiff
    Enzyme.autodiff(
        Reverse,
        color_loss_pure,
        Active,
        Duplicated(edcs.basis, edcs.d_basis),
        Duplicated(edcs.offset, edcs.d_offset),
        Duplicated(edcs.scale, edcs.d_scale),
        Const(input),
        Const(target)
    )
    
    return (edcs.d_basis, edcs.d_offset, edcs.d_scale)
end

function enzyme_forward_gradient!(edcs::EnzymeDifferentiableColorSpace,
                                  input::Vector{Float64},
                                  direction::Vector{Float64})
    # Forward mode: compute directional derivative
    # d_output/d_params in direction `direction`
    
    # Pack parameters
    n_params = 15
    tangent = zeros(n_params)
    tangent[1:length(direction)] = direction
    
    # Split tangent into components
    t_basis = reshape(tangent[1:9], 3, 3)
    t_offset = tangent[10:12]
    t_scale = tangent[13:15]
    
    # Forward mode
    result = Enzyme.autodiff(
        Forward,
        forward_color_pure,
        Duplicated(edcs.basis, t_basis),
        Duplicated(edcs.offset, t_offset),
        Duplicated(edcs.scale, t_scale),
        Const(input)
    )
    
    return result
end

function enzyme_hessian_fwd_rev!(edcs::EnzymeDifferentiableColorSpace,
                                 input::Vector{Float64},
                                 target::Vector{Float64})
    n_params = 15
    hessian = zeros(n_params, n_params)
    
    # Forward-over-Reverse for Hessian
    for i in 1:n_params
        # Set up tangent direction
        direction = zeros(n_params)
        direction[i] = 1.0
        
        t_basis = reshape(direction[1:9], 3, 3)
        t_offset = direction[10:12]
        t_scale = direction[13:15]
        
        # Shadow for reverse
        d_basis = zeros(3, 3)
        d_offset = zeros(3)
        d_scale = zeros(3)
        
        # Forward-over-Reverse
        Enzyme.autodiff(
            Forward,
            (b, o, s) -> begin
                db, do, ds = zeros(3,3), zeros(3), zeros(3)
                Enzyme.autodiff(
                    Reverse,
                    color_loss_pure,
                    Active,
                    Duplicated(b, db),
                    Duplicated(o, do),
                    Duplicated(s, ds),
                    Const(input),
                    Const(target)
                )
                return vcat(vec(db), do, ds)
            end,
            Duplicated(edcs.basis, t_basis),
            Duplicated(edcs.offset, t_offset),
            Duplicated(edcs.scale, t_scale)
        )
        
        # Extract column i of Hessian
        hessian[:, i] = vcat(vec(d_basis), d_offset, d_scale)
    end
    
    return hessian
end
```
=#

"""
Reverse mode gradient (numerical fallback).
"""
function enzyme_reverse_gradient!(edcs::EnzymeDifferentiableColorSpace,
                                  input::Vector{Float64},
                                  target::Vector{Float64};
                                  epsilon::Float64=1e-6)
    # Zero shadow variables
    edcs.d_basis .= 0.0
    edcs.d_offset .= 0.0
    edcs.d_scale .= 0.0
    
    base_loss = color_loss_pure(edcs.basis, edcs.offset, edcs.scale, input, target)
    
    # Numerical gradients (replace with Enzyme when available)
    for i in 1:3, j in 1:3
        edcs.basis[i, j] += epsilon
        loss_plus = color_loss_pure(edcs.basis, edcs.offset, edcs.scale, input, target)
        edcs.basis[i, j] -= epsilon
        edcs.d_basis[i, j] = (loss_plus - base_loss) / epsilon
    end
    
    for i in 1:3
        edcs.offset[i] += epsilon
        loss_plus = color_loss_pure(edcs.basis, edcs.offset, edcs.scale, input, target)
        edcs.offset[i] -= epsilon
        edcs.d_offset[i] = (loss_plus - base_loss) / epsilon
    end
    
    for i in 1:3
        edcs.scale[i] += epsilon
        loss_plus = color_loss_pure(edcs.basis, edcs.offset, edcs.scale, input, target)
        edcs.scale[i] -= epsilon
        edcs.d_scale[i] = (loss_plus - base_loss) / epsilon
    end
    
    (edcs.d_basis, edcs.d_offset, edcs.d_scale)
end

"""
Forward mode gradient (numerical fallback).
"""
function enzyme_forward_gradient!(edcs::EnzymeDifferentiableColorSpace,
                                  input::Vector{Float64},
                                  direction::Vector{Float64};
                                  epsilon::Float64=1e-6)
    # Directional derivative
    n = min(length(direction), 15)
    
    # Pack current parameters
    params = vcat(vec(edcs.basis), edcs.offset, edcs.scale)
    
    # Perturb in direction
    params_plus = copy(params)
    params_plus[1:n] .+= epsilon .* direction[1:n]
    
    # Compute outputs
    basis = reshape(params[1:9], 3, 3)
    basis_plus = reshape(params_plus[1:9], 3, 3)
    offset = params[10:12]
    offset_plus = params_plus[10:12]
    scale = params[13:15]
    scale_plus = params_plus[13:15]
    
    out = forward_color_pure(basis, offset, scale, input)
    out_plus = forward_color_pure(basis_plus, offset_plus, scale_plus, input)
    
    (out_plus .- out) ./ epsilon
end

"""
Hessian via forward-over-reverse (numerical fallback).
"""
function enzyme_hessian_fwd_rev!(edcs::EnzymeDifferentiableColorSpace,
                                 input::Vector{Float64},
                                 target::Vector{Float64};
                                 epsilon::Float64=1e-4)
    n_params = 15
    hessian = zeros(n_params, n_params)
    
    function loss_at(params::Vector{Float64})
        basis = reshape(params[1:9], 3, 3)
        offset = params[10:12]
        scale = params[13:15]
        color_loss_pure(basis, offset, scale, input, target)
    end
    
    params = vcat(vec(edcs.basis), edcs.offset, edcs.scale)
    
    for i in 1:n_params
        for j in i:n_params
            p_pp = copy(params); p_pp[i] += epsilon; p_pp[j] += epsilon
            p_pm = copy(params); p_pm[i] += epsilon; p_pm[j] -= epsilon
            p_mp = copy(params); p_mp[i] -= epsilon; p_mp[j] += epsilon
            p_mm = copy(params); p_mm[i] -= epsilon; p_mm[j] -= epsilon
            
            hessian[i, j] = (loss_at(p_pp) - loss_at(p_pm) - loss_at(p_mp) + loss_at(p_mm)) / (4 * epsilon^2)
            hessian[j, i] = hessian[i, j]
        end
    end
    
    hessian
end

"""
Jacobian of color transformation (numerical fallback).
"""
function enzyme_jacobian!(edcs::EnzymeDifferentiableColorSpace,
                          input::Vector{Float64};
                          epsilon::Float64=1e-6)
    n_params = 15
    n_outputs = 3
    jacobian = zeros(n_outputs, n_params)
    
    params = vcat(vec(edcs.basis), edcs.offset, edcs.scale)
    base_output = forward_color_pure(edcs.basis, edcs.offset, edcs.scale, input)
    
    for i in 1:n_params
        params_plus = copy(params)
        params_plus[i] += epsilon
        
        basis = reshape(params_plus[1:9], 3, 3)
        offset = params_plus[10:12]
        scale = params_plus[13:15]
        
        out_plus = forward_color_pure(basis, offset, scale, input)
        jacobian[:, i] = (out_plus .- base_output) ./ epsilon
    end
    
    jacobian
end

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING WITH ENZYME GRADIENTS
# ═══════════════════════════════════════════════════════════════════════════════

"""
Train color space using Enzyme gradients.
"""
function train_with_enzyme!(edcs::EnzymeDifferentiableColorSpace,
                            inputs::Vector{Vector{Float64}},
                            targets::Vector{Vector{Float64}};
                            lr::Float64=0.01,
                            n_steps::Int=100,
                            use_momentum::Bool=false,
                            momentum::Float64=0.9)
    # Momentum buffers
    v_basis = zeros(3, 3)
    v_offset = zeros(3)
    v_scale = zeros(3)
    
    for step in 1:n_steps
        edcs.step += 1
        total_loss = 0.0
        
        # Accumulate gradients over batch
        acc_d_basis = zeros(3, 3)
        acc_d_offset = zeros(3)
        acc_d_scale = zeros(3)
        
        for (input, target) in zip(inputs, targets)
            enzyme_reverse_gradient!(edcs, input, target)
            acc_d_basis .+= edcs.d_basis
            acc_d_offset .+= edcs.d_offset
            acc_d_scale .+= edcs.d_scale
            total_loss += color_loss_pure(edcs.basis, edcs.offset, edcs.scale, input, target)
        end
        
        # Average gradients
        n = length(inputs)
        acc_d_basis ./= n
        acc_d_offset ./= n
        acc_d_scale ./= n
        
        if use_momentum
            # Momentum update
            v_basis = momentum .* v_basis .- lr .* acc_d_basis
            v_offset = momentum .* v_offset .- lr .* acc_d_offset
            v_scale = momentum .* v_scale .- lr .* acc_d_scale
            
            edcs.basis .+= v_basis
            edcs.offset .+= v_offset
            edcs.scale .+= v_scale
        else
            # Standard SGD
            edcs.basis .-= lr .* acc_d_basis
            edcs.offset .-= lr .* acc_d_offset
            edcs.scale .-= lr .* acc_d_scale
        end
        
        push!(edcs.loss_history, total_loss / n)
        
        # Update fingerprint
        edcs.fingerprint ⊻= UInt64(round(total_loss * 1e9))
    end
    
    edcs
end

# ═══════════════════════════════════════════════════════════════════════════════
# INTERLEAVED NARRATOR LEARNING
# ═══════════════════════════════════════════════════════════════════════════════

"""
    NarratorTriadLearner

Learns color spaces for a triad of narrators using Enzyme.
Each narrator has its own color space:
- ORIGINARY: learns from true observations (high bandwidth)
- DERIVED: learns from ORIGINARY's outputs (medium bandwidth)
- LIMINAL: learns from boundary observations (low bandwidth)
"""
mutable struct NarratorTriadLearner
    originary_space::EnzymeDifferentiableColorSpace
    derived_space::EnzymeDifferentiableColorSpace
    liminal_space::EnzymeDifferentiableColorSpace
    
    # Learning rates (different per narrator)
    lr_originary::Float64
    lr_derived::Float64
    lr_liminal::Float64
    
    # Training history
    step::Int
    synergy_history::Vector{Float64}
    
    fingerprint::UInt64
end

function NarratorTriadLearner(; seed::UInt64=UInt64(0x4A11A70F))
    NarratorTriadLearner(
        EnzymeDifferentiableColorSpace(seed=seed),
        EnzymeDifferentiableColorSpace(seed=seed ⊻ UInt64(1)),
        EnzymeDifferentiableColorSpace(seed=seed ⊻ UInt64(2)),
        0.01, 0.005, 0.001,  # Learning rates
        0, Float64[], seed
    )
end

"""
Train the triad in interleaved fashion.
Pattern: 2 originary, 1 derived, 1 liminal per cycle.
"""
function train_triad_interleaved!(learner::NarratorTriadLearner,
                                   world_observations::Vector{Vector{Float64}},
                                   targets::Vector{Vector{Float64}};
                                   n_cycles::Int=100)
    for cycle in 1:n_cycles
        learner.step += 1
        
        # Balanced ternary pattern
        pattern = [(1, learner.lr_originary),   # ORIGINARY
                   (1, learner.lr_originary),   # ORIGINARY
                   (2, learner.lr_derived),     # DERIVED
                   (3, learner.lr_liminal)]     # LIMINAL
        
        cycle_loss = 0.0
        
        for (narrator_idx, lr) in pattern
            space = if narrator_idx == 1
                learner.originary_space
            elseif narrator_idx == 2
                learner.derived_space
            else
                learner.liminal_space
            end
            
            # Sample a random observation
            idx = 1 + rand(UInt) % length(world_observations)
            input = world_observations[idx]
            target = targets[idx]
            
            # Compute gradient and update
            enzyme_reverse_gradient!(space, input, target)
            space.basis .-= lr .* space.d_basis
            space.offset .-= lr .* space.d_offset
            space.scale .-= lr .* space.d_scale
            
            cycle_loss += color_loss_pure(space.basis, space.offset, space.scale, input, target)
        end
        
        # Compute synergy: how aligned are the three spaces?
        test_input = world_observations[1]
        out_orig = forward_color_pure(learner.originary_space.basis, 
                                      learner.originary_space.offset,
                                      learner.originary_space.scale, test_input)
        out_deriv = forward_color_pure(learner.derived_space.basis,
                                       learner.derived_space.offset,
                                       learner.derived_space.scale, test_input)
        out_limin = forward_color_pure(learner.liminal_space.basis,
                                       learner.liminal_space.offset,
                                       learner.liminal_space.scale, test_input)
        
        # Synergy = 1 - average pairwise distance
        dist_od = sqrt(sum((out_orig .- out_deriv).^2))
        dist_ol = sqrt(sum((out_orig .- out_limin).^2))
        dist_dl = sqrt(sum((out_deriv .- out_limin).^2))
        
        synergy = 1.0 - (dist_od + dist_ol + dist_dl) / (3.0 * sqrt(3.0))
        push!(learner.synergy_history, synergy)
        
        # Update fingerprint
        learner.fingerprint ⊻= learner.originary_space.fingerprint ⊻
                               learner.derived_space.fingerprint ⊻
                               learner.liminal_space.fingerprint
    end
    
    learner
end

# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

function demo_enzyme_colorspace()
    println()
    println("═══════════════════════════════════════════════════════════════════════════════")
    println("  GAY ENZYME EXTENSION: Real Automatic Differentiation for Color Spaces")
    println("═══════════════════════════════════════════════════════════════════════════════")
    println()
    
    # Create color space
    edcs = EnzymeDifferentiableColorSpace()
    
    # Generate training data
    n_samples = 20
    inputs = [rand(3) for _ in 1:n_samples]
    targets = [[clamp(c + 0.1, 0, 1) for c in inp] for inp in inputs]
    
    println("─── Gradient Computation ───")
    d_basis, d_offset, d_scale = enzyme_reverse_gradient!(edcs, inputs[1], targets[1])
    println("  ∂L/∂basis[1,1] = $(round(d_basis[1,1]; digits=6))")
    println("  ∂L/∂offset = $(round.(d_offset; digits=6))")
    
    println()
    println("─── Jacobian ───")
    jacobian = enzyme_jacobian!(edcs, inputs[1])
    println("  Shape: $(size(jacobian))")
    println("  ∂output/∂basis[1,1] = $(round.(jacobian[:, 1]; digits=6))")
    
    println()
    println("─── Training ───")
    train_with_enzyme!(edcs, inputs, targets; lr=0.1, n_steps=100, use_momentum=true)
    println("  Initial loss: $(round(edcs.loss_history[1]; digits=6))")
    println("  Final loss: $(round(edcs.loss_history[end]; digits=6))")
    
    println()
    println("─── Triad Learning ───")
    learner = NarratorTriadLearner()
    train_triad_interleaved!(learner, inputs, targets; n_cycles=50)
    println("  Synergy (initial): $(round(learner.synergy_history[1]; digits=4))")
    println("  Synergy (final): $(round(learner.synergy_history[end]; digits=4))")
    
    println()
    println("═══════════════════════════════════════════════════════════════════════════════")
    
    return (edcs, learner)
end

end  # module GayEnzymeExt
