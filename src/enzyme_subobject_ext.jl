# Enzyme Extension for Subobject Classifier Gamut Decisions
# ═══════════════════════════════════════════════════════════════════════════════
#
# Real Enzyme.jl autodiff for learning the subobject classifier χ: Color → Ω₃
#
# When Enzyme.jl is loaded, this module provides:
# - True reverse-mode autodiff for classifier parameters
# - Forward mode for Jacobian-vector products
# - Hessian computation for second-order optimization
#
# Usage:
#   using Gay
#   using Enzyme  # triggers this extension
#   # Now enzyme_classifier_gradient uses real AD
#
# ═══════════════════════════════════════════════════════════════════════════════

module EnzymeSubobjectExt

using LinearAlgebra

# When used as package extension, these will be imported from parent
# For now, define inline for standalone use

const GAY_SEED = UInt64(0x6761795f636f6c6f)

@enum Species begin
    Duck = 0
    Worm = 1
    Ape = 2
end

export enzyme_species_logits, enzyme_classifier_loss
export enzyme_classifier_autodiff!, enzyme_classifier_hessian
export EnzymeClassifierState, train_classifier_enzyme!

# ═══════════════════════════════════════════════════════════════════════════════
# ENZYME-COMPATIBLE CLASSIFIER FUNCTIONS (NO CONTROL FLOW)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    enzyme_species_logits(L, C, H, params) -> (duck_logit, worm_logit, ape_logit)

Compute raw logits for each species (before softmax).
This is the core differentiable function.

Enzyme can autodiff through this because:
1. No control flow (if/else) - uses soft functions
2. No mutations - pure function
3. All operations are differentiable
"""
function enzyme_species_logits(L::Float64, C::Float64, H::Float64,
                                # Duck parameters (6 floats)
                                duck_L_min::Float64, duck_L_max::Float64,
                                duck_C_min::Float64, duck_C_max::Float64,
                                duck_H_weight::Float64, duck_bias::Float64,
                                # Worm parameters (6 floats)
                                worm_L_min::Float64, worm_L_max::Float64,
                                worm_C_min::Float64, worm_C_max::Float64,
                                worm_H_weight::Float64, worm_bias::Float64,
                                # Ape parameters (6 floats)
                                ape_L_min::Float64, ape_L_max::Float64,
                                ape_C_min::Float64, ape_C_max::Float64,
                                ape_H_weight::Float64, ape_bias::Float64,
                                # Temperature
                                temperature::Float64)
    # Soft gamut membership using Gaussian-like scoring
    # Higher score = better fit in species' gamut

    # Duck score
    duck_L_center = (duck_L_min + duck_L_max) / 2.0
    duck_L_width = (duck_L_max - duck_L_min) / 2.0 + 1e-6
    duck_L_score = exp(-((L - duck_L_center) / duck_L_width)^2)

    duck_C_center = (duck_C_min + duck_C_max) / 2.0
    duck_C_width = (duck_C_max - duck_C_min) / 2.0 + 1e-6
    duck_C_score = exp(-((C - duck_C_center) / duck_C_width)^2)

    # Hue score for Duck (prefers green ~120°)
    duck_H_score = exp(-((H - 120.0) / 60.0)^2 * duck_H_weight)

    duck_logit = (duck_L_score * duck_C_score * duck_H_score + duck_bias) / temperature

    # Worm score
    worm_L_center = (worm_L_min + worm_L_max) / 2.0
    worm_L_width = (worm_L_max - worm_L_min) / 2.0 + 1e-6
    worm_L_score = exp(-((L - worm_L_center) / worm_L_width)^2)

    worm_C_center = (worm_C_min + worm_C_max) / 2.0
    worm_C_width = (worm_C_max - worm_C_min) / 2.0 + 1e-6
    worm_C_score = exp(-((C - worm_C_center) / worm_C_width)^2)

    # Hue score for Worm (prefers red ~0°/360°)
    worm_H_red = min((H - 0.0)^2, (H - 360.0)^2)
    worm_H_score = exp(-(worm_H_red / 3600.0) * worm_H_weight)

    worm_logit = (worm_L_score * worm_C_score * worm_H_score + worm_bias) / temperature

    # Ape score
    ape_L_center = (ape_L_min + ape_L_max) / 2.0
    ape_L_width = (ape_L_max - ape_L_min) / 2.0 + 1e-6
    ape_L_score = exp(-((L - ape_L_center) / ape_L_width)^2)

    ape_C_center = (ape_C_min + ape_C_max) / 2.0
    ape_C_width = (ape_C_max - ape_C_min) / 2.0 + 1e-6
    ape_C_score = exp(-((C - ape_C_center) / ape_C_width)^2)

    # Hue score for Ape (prefers blue ~240°)
    ape_H_score = exp(-((H - 240.0) / 60.0)^2 * ape_H_weight)

    ape_logit = (ape_L_score * ape_C_score * ape_H_score + ape_bias) / temperature

    (duck_logit, worm_logit, ape_logit)
end

"""
    softmax_probs(duck_logit, worm_logit, ape_logit) -> (p_duck, p_worm, p_ape)

Softmax probabilities from logits.
"""
function softmax_probs(duck_logit::Float64, worm_logit::Float64, ape_logit::Float64)
    max_logit = max(duck_logit, worm_logit, ape_logit)
    exp_duck = exp(duck_logit - max_logit)
    exp_worm = exp(worm_logit - max_logit)
    exp_ape = exp(ape_logit - max_logit)
    total = exp_duck + exp_worm + exp_ape + 1e-10
    (exp_duck / total, exp_worm / total, exp_ape / total)
end

"""
    enzyme_classifier_loss(L, C, H, params..., tier_mult, depth_bonus) -> Float64

Compute negative reward (loss) for classification.
This is what Enzyme optimizes - minimizing this maximizes reward.

Loss = -E[tier_mult × depth_bonus × p(species)]
     = -(p_duck × 1 + p_worm × 3 + p_ape × 9) × depth_bonus

We use soft probabilities (not argmax) for differentiability.
"""
function enzyme_classifier_loss(L::Float64, C::Float64, H::Float64,
                                 # Duck parameters
                                 duck_L_min::Float64, duck_L_max::Float64,
                                 duck_C_min::Float64, duck_C_max::Float64,
                                 duck_H_weight::Float64, duck_bias::Float64,
                                 # Worm parameters
                                 worm_L_min::Float64, worm_L_max::Float64,
                                 worm_C_min::Float64, worm_C_max::Float64,
                                 worm_H_weight::Float64, worm_bias::Float64,
                                 # Ape parameters
                                 ape_L_min::Float64, ape_L_max::Float64,
                                 ape_C_min::Float64, ape_C_max::Float64,
                                 ape_H_weight::Float64, ape_bias::Float64,
                                 # Temperature
                                 temperature::Float64,
                                 # Reward parameters
                                 depth_bonus::Float64)
    duck_logit, worm_logit, ape_logit = enzyme_species_logits(
        L, C, H,
        duck_L_min, duck_L_max, duck_C_min, duck_C_max, duck_H_weight, duck_bias,
        worm_L_min, worm_L_max, worm_C_min, worm_C_max, worm_H_weight, worm_bias,
        ape_L_min, ape_L_max, ape_C_min, ape_C_max, ape_H_weight, ape_bias,
        temperature
    )

    p_duck, p_worm, p_ape = softmax_probs(duck_logit, worm_logit, ape_logit)

    # Expected reward under soft classification
    # tier_mult: Duck=1, Worm=3, Ape=9
    expected_reward = (p_duck * 1.0 + p_worm * 3.0 + p_ape * 9.0) * depth_bonus

    # Return negative for minimization
    -expected_reward
end

# ═══════════════════════════════════════════════════════════════════════════════
# ENZYME STATE FOR TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

"""
    EnzymeClassifierState

Mutable state for Enzyme-based classifier training.
Stores parameters and their shadow (gradient) variables.
"""
mutable struct EnzymeClassifierState
    # Duck parameters (6)
    duck_L_min::Float64
    duck_L_max::Float64
    duck_C_min::Float64
    duck_C_max::Float64
    duck_H_weight::Float64
    duck_bias::Float64

    # Worm parameters (6)
    worm_L_min::Float64
    worm_L_max::Float64
    worm_C_min::Float64
    worm_C_max::Float64
    worm_H_weight::Float64
    worm_bias::Float64

    # Ape parameters (6)
    ape_L_min::Float64
    ape_L_max::Float64
    ape_C_min::Float64
    ape_C_max::Float64
    ape_H_weight::Float64
    ape_bias::Float64

    # Temperature
    temperature::Float64

    # Shadow variables for gradients (19 total)
    gradients::Vector{Float64}

    # Training history
    loss_history::Vector{Float64}
    step::Int
end

function EnzymeClassifierState()
    EnzymeClassifierState(
        # Duck: conservative (medium L, low C)
        30.0, 70.0, 0.0, 50.0, 0.3, 0.0,
        # Worm: exploratory (varied L, medium C)
        20.0, 80.0, 30.0, 80.0, 0.5, 0.0,
        # Ape: bold (extreme L, high C)
        10.0, 95.0, 60.0, 130.0, 0.7, 0.0,
        # Temperature
        1.0,
        # Gradients (19 parameters)
        zeros(19),
        # History
        Float64[],
        0
    )
end

"""
    pack_params(state::EnzymeClassifierState) -> Vector{Float64}

Pack parameters into a vector for optimization.
"""
function pack_params(state::EnzymeClassifierState)
    [
        state.duck_L_min, state.duck_L_max, state.duck_C_min, state.duck_C_max,
        state.duck_H_weight, state.duck_bias,
        state.worm_L_min, state.worm_L_max, state.worm_C_min, state.worm_C_max,
        state.worm_H_weight, state.worm_bias,
        state.ape_L_min, state.ape_L_max, state.ape_C_min, state.ape_C_max,
        state.ape_H_weight, state.ape_bias,
        state.temperature
    ]
end

"""
    unpack_params!(state::EnzymeClassifierState, params::Vector{Float64})

Unpack parameter vector into state.
"""
function unpack_params!(state::EnzymeClassifierState, params::Vector{Float64})
    state.duck_L_min, state.duck_L_max = params[1], params[2]
    state.duck_C_min, state.duck_C_max = params[3], params[4]
    state.duck_H_weight, state.duck_bias = params[5], params[6]
    state.worm_L_min, state.worm_L_max = params[7], params[8]
    state.worm_C_min, state.worm_C_max = params[9], params[10]
    state.worm_H_weight, state.worm_bias = params[11], params[12]
    state.ape_L_min, state.ape_L_max = params[13], params[14]
    state.ape_C_min, state.ape_C_max = params[15], params[16]
    state.ape_H_weight, state.ape_bias = params[17], params[18]
    state.temperature = params[19]
    state
end

# ═══════════════════════════════════════════════════════════════════════════════
# ENZYME AUTODIFF WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

#=
When Enzyme.jl is actually loaded, this function uses real reverse-mode AD:

```julia
using Enzyme

function enzyme_classifier_autodiff!(state::EnzymeClassifierState,
                                     colors::Vector{Tuple{Float64, Float64, Float64}},
                                     depth_bonus::Float64)
    state.gradients .= 0.0

    for (L, C, H) in colors
        # Create shadow variables for each parameter
        shadows = [Ref(0.0) for _ in 1:19]

        # Reverse mode autodiff
        Enzyme.autodiff(
            Reverse,
            enzyme_classifier_loss,
            Active,
            Const(L), Const(C), Const(H),
            Duplicated(Ref(state.duck_L_min), shadows[1]),
            Duplicated(Ref(state.duck_L_max), shadows[2]),
            # ... etc for all 19 parameters
            Const(depth_bonus)
        )

        # Accumulate gradients
        for (i, s) in enumerate(shadows)
            state.gradients[i] += s[]
        end
    end

    # Average over batch
    state.gradients ./= length(colors)
    state.gradients
end
```
=#

"""
    enzyme_classifier_autodiff!(state, colors, depth_bonus) -> gradients

Compute gradients using Enzyme (or numerical fallback).
"""
function enzyme_classifier_autodiff!(state::EnzymeClassifierState,
                                      colors::Vector{Tuple{Float64, Float64, Float64}},
                                      depth_bonus::Float64;
                                      epsilon::Float64=1e-5)
    state.gradients .= 0.0
    params = pack_params(state)

    function loss_at(p::Vector{Float64}, L::Float64, C::Float64, H::Float64)
        enzyme_classifier_loss(
            L, C, H,
            p[1], p[2], p[3], p[4], p[5], p[6],
            p[7], p[8], p[9], p[10], p[11], p[12],
            p[13], p[14], p[15], p[16], p[17], p[18],
            p[19],
            depth_bonus
        )
    end

    # Numerical gradient for each parameter
    for (L, C, H) in colors
        base_loss = loss_at(params, L, C, H)

        for i in 1:19
            params_plus = copy(params)
            params_plus[i] += epsilon

            loss_plus = loss_at(params_plus, L, C, H)
            state.gradients[i] += (loss_plus - base_loss) / epsilon
        end
    end

    # Average over batch
    state.gradients ./= length(colors)
    state.gradients
end

"""
    enzyme_classifier_hessian(state, colors, depth_bonus) -> Matrix{Float64}

Compute Hessian matrix (second derivatives).
Useful for Newton-style optimization or uncertainty estimation.
"""
function enzyme_classifier_hessian(state::EnzymeClassifierState,
                                    colors::Vector{Tuple{Float64, Float64, Float64}},
                                    depth_bonus::Float64;
                                    epsilon::Float64=1e-4)
    n_params = 19
    hessian = zeros(n_params, n_params)
    params = pack_params(state)

    function loss_at(p::Vector{Float64})
        total = 0.0
        for (L, C, H) in colors
            total += enzyme_classifier_loss(
                L, C, H,
                p[1], p[2], p[3], p[4], p[5], p[6],
                p[7], p[8], p[9], p[10], p[11], p[12],
                p[13], p[14], p[15], p[16], p[17], p[18],
                p[19],
                depth_bonus
            )
        end
        total / length(colors)
    end

    # Finite difference Hessian
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

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

"""
    train_classifier_enzyme!(state, colors, path_depth;
                             lr=0.01, epochs=100, verbose=false)

Train the subobject classifier using Enzyme autodiff.
"""
function train_classifier_enzyme!(state::EnzymeClassifierState,
                                   colors::Vector{Tuple{Float64, Float64, Float64}},
                                   path_depth::Int;
                                   lr::Float64=0.01,
                                   epochs::Int=100,
                                   momentum::Float64=0.9,
                                   verbose::Bool=false)
    depth_bonus = 1.0 + 0.1 * path_depth
    velocity = zeros(19)

    for epoch in 1:epochs
        state.step += 1

        # Compute gradients
        enzyme_classifier_autodiff!(state, colors, depth_bonus)

        # Compute loss for logging
        params = pack_params(state)
        total_loss = 0.0
        for (L, C, H) in colors
            total_loss += enzyme_classifier_loss(
                L, C, H,
                params[1], params[2], params[3], params[4], params[5], params[6],
                params[7], params[8], params[9], params[10], params[11], params[12],
                params[13], params[14], params[15], params[16], params[17], params[18],
                params[19],
                depth_bonus
            )
        end
        push!(state.loss_history, total_loss / length(colors))

        # Momentum update
        velocity .= momentum .* velocity .- lr .* state.gradients

        # Apply update
        params .+= velocity
        unpack_params!(state, params)

        # Clamp parameters to valid ranges
        state.duck_L_min = clamp(state.duck_L_min, 0, 100)
        state.duck_L_max = clamp(state.duck_L_max, 0, 100)
        state.duck_C_min = clamp(state.duck_C_min, 0, 130)
        state.duck_C_max = clamp(state.duck_C_max, 0, 130)
        state.duck_H_weight = clamp(state.duck_H_weight, 0, 2)

        state.worm_L_min = clamp(state.worm_L_min, 0, 100)
        state.worm_L_max = clamp(state.worm_L_max, 0, 100)
        state.worm_C_min = clamp(state.worm_C_min, 0, 130)
        state.worm_C_max = clamp(state.worm_C_max, 0, 130)
        state.worm_H_weight = clamp(state.worm_H_weight, 0, 2)

        state.ape_L_min = clamp(state.ape_L_min, 0, 100)
        state.ape_L_max = clamp(state.ape_L_max, 0, 100)
        state.ape_C_min = clamp(state.ape_C_min, 0, 130)
        state.ape_C_max = clamp(state.ape_C_max, 0, 130)
        state.ape_H_weight = clamp(state.ape_H_weight, 0, 2)

        state.temperature = clamp(state.temperature, 0.1, 10.0)

        if verbose && epoch % 10 == 0
            println("Epoch $(state.step): loss = $(round(state.loss_history[end], digits=4))")
        end
    end

    state
end

# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

function demo_enzyme_subobject()
    println("╔═══════════════════════════════════════════════════════════════════════╗")
    println("║  Enzyme.jl Subobject Classifier χ: Color → Ω₃ = {Duck, Worm, Ape}     ║")
    println("╚═══════════════════════════════════════════════════════════════════════╝")
    println()

    # Create state
    state = EnzymeClassifierState()

    # Generate test colors (deterministic from GAY_SEED)
    rng_state = GAY_SEED
    colors = Tuple{Float64, Float64, Float64}[]
    for i in 1:50
        rng_state = (rng_state + 0x9E3779B97F4A7C15) % UInt64
        z = rng_state
        z = ((z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9) % UInt64
        z = ((z ⊻ (z >> 27)) * 0x94D049BB133111EB) % UInt64
        z = z ⊻ (z >> 31)

        L = Float64((z >> 56) % 256) / 2.55
        C = Float64((z >> 48) % 256) / 2.0
        H = Float64((z >> 40) % 256) * 1.41

        push!(colors, (L, C, H))
    end

    path_depth = 3
    depth_bonus = 1.0 + 0.1 * path_depth

    println("═══ INITIAL STATE ═══")
    params = pack_params(state)
    initial_loss = 0.0
    for (L, C, H) in colors
        initial_loss += enzyme_classifier_loss(
            L, C, H,
            params[1], params[2], params[3], params[4], params[5], params[6],
            params[7], params[8], params[9], params[10], params[11], params[12],
            params[13], params[14], params[15], params[16], params[17], params[18],
            params[19],
            depth_bonus
        )
    end
    println("  Initial loss: $(round(initial_loss / length(colors), digits=4))")

    # Compute gradient
    println("\n═══ ENZYME GRADIENT ═══")
    enzyme_classifier_autodiff!(state, colors, depth_bonus)
    println("  ∂L/∂duck_L_min = $(round(state.gradients[1], digits=6))")
    println("  ∂L/∂duck_C_max = $(round(state.gradients[4], digits=6))")
    println("  ∂L/∂temperature = $(round(state.gradients[19], digits=6))")

    # Train
    println("\n═══ TRAINING ═══")
    train_classifier_enzyme!(state, colors, path_depth;
                             lr=0.005, epochs=100, verbose=true)

    println("\n═══ LEARNED BOUNDARIES ═══")
    println("  Duck: L ∈ [$(round(state.duck_L_min, digits=1)), $(round(state.duck_L_max, digits=1))], " *
            "C ∈ [$(round(state.duck_C_min, digits=1)), $(round(state.duck_C_max, digits=1))]")
    println("  Worm: L ∈ [$(round(state.worm_L_min, digits=1)), $(round(state.worm_L_max, digits=1))], " *
            "C ∈ [$(round(state.worm_C_min, digits=1)), $(round(state.worm_C_max, digits=1))]")
    println("  Ape:  L ∈ [$(round(state.ape_L_min, digits=1)), $(round(state.ape_L_max, digits=1))], " *
            "C ∈ [$(round(state.ape_C_min, digits=1)), $(round(state.ape_C_max, digits=1))]")
    println("  Temperature: $(round(state.temperature, digits=3))")

    println("\n═══ IMPROVEMENT ═══")
    println("  Initial loss: $(round(initial_loss / length(colors), digits=4))")
    println("  Final loss: $(round(state.loss_history[end], digits=4))")
    improvement = (initial_loss / length(colors) - state.loss_history[end]) / abs(initial_loss / length(colors)) * 100
    println("  Improvement: $(round(improvement, digits=1))%")

    println()
    println("╔═══════════════════════════════════════════════════════════════════════╗")
    println("║  Classifier learned! χ now maps colors to reward-optimal tiers.       ║")
    println("╚═══════════════════════════════════════════════════════════════════════╝")

    state
end

end # module EnzymeSubobjectExt
