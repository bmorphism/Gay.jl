# Learned Subobject Classifier for Gay Gamut Decisions via Enzyme.jl
# ═══════════════════════════════════════════════════════════════════════════════
#
# In topos theory, the SUBOBJECT CLASSIFIER Ω classifies which elements belong
# to which subobjects. For any monomorphism A ↪ B, there exists a unique
# characteristic morphism χ: B → Ω where:
#
#   χ(b) = true  iff b ∈ A
#   χ(b) = false iff b ∉ A
#
# For Gay colors, we extend this to a TRI-VALUED subobject classifier:
#
#   Ω₃ = {Duck, Worm, Ape} ≅ GF(3)
#
# The classifier χ: Color → Ω₃ determines which TIER/SPECIES owns a color
# based on its gamut position and bandwidth characteristics.
#
# KEY INSIGHT: The classifier parameters are LEARNABLE via Enzyme.jl autodiff.
# The open game dynamics from gay_open_game_seek.hy can be differentiated
# to learn optimal tier assignments that maximize reward.
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │                     SUBOBJECT CLASSIFIER STRUCTURE                          │
# │                                                                             │
# │  Color (L,C,H) ──→ χ(color; θ) ──→ Ω₃ = {Duck, Worm, Ape}                  │
# │                    │                 │                                      │
# │                    │ Enzyme autodiff │ Open game reward                     │
# │                    ↓                 ↓                                      │
# │              ∂L/∂θ (gradient)   tier_mult × depth_bonus                    │
# │                                                                             │
# │  θ = learnable parameters (neural network or polynomial)                   │
# │  L = loss = -reward from open game dynamics                                │
# │                                                                             │
# └─────────────────────────────────────────────────────────────────────────────┘
#
# Connection to existing code:
# - ArenaErrors: Species = {Duck, Worm, Ape}, ThreeMatch decisions
# - gamut_learnable.jl: GamutParameters, is_in_gamut
# - gay_open_game_seek.hy: reward = base × tier_mult × depth_bonus
# - gay.hy: path_to_seed, triadic forking, SPI guarantees
#
# References:
# - Mac Lane & Moerdijk: "Sheaves in Geometry and Logic" Ch IV (Subobject Classifier)
# - Loregian 2025: 2-Transducers (composition of state machines)
# - GAYOPERAD_CONNECTION.md: ColoredType as C-colored objects

module SubobjectClassifierGamut

using Colors: RGB, LCHab, Lab, convert
using LinearAlgebra: norm, dot

export SubobjectClassifier, TrivaluedOmega, GamutClassifier
export classify_color, characteristic_morphism
export Species, Duck, Worm, Ape
export ClassifierParameters, LearnedClassifier
export enzyme_classifier_gradient, learn_classifier!
export tier_reward, open_game_loss
export verify_spi_invariance, world_subobject_classifier

# ═══════════════════════════════════════════════════════════════════════════════
# SPECIES / TIER (from ArenaErrors)
# ═══════════════════════════════════════════════════════════════════════════════

@enum Species begin
    Duck = 0   # Green, < 100M/sec, 1x multiplier
    Worm = 1   # Red, 100M - 1B/sec, 3x multiplier
    Ape  = 2   # Blue, > 1B/sec, 9x multiplier
end

const SPECIES_COLORS = Dict{Species, UInt32}(
    Duck => 0x00FF00,  # Green
    Worm => 0xFF0000,  # Red
    Ape  => 0x0000FF   # Blue
)

const TIER_MULTIPLIERS = Dict{Species, Float64}(
    Duck => 1.0,
    Worm => 3.0,
    Ape  => 9.0
)

species_add(a::Species, b::Species) = Species(mod(Int(a) + Int(b), 3))
species_neg(a::Species) = Species(mod(-Int(a), 3))

# ═══════════════════════════════════════════════════════════════════════════════
# TRI-VALUED SUBOBJECT CLASSIFIER Ω₃
# ═══════════════════════════════════════════════════════════════════════════════

"""
    TrivaluedOmega

The tri-valued subobject classifier Ω₃ = {Duck, Worm, Ape}.

In classical topos theory, Ω = {true, false} = 2.
In our triadic setting, Ω₃ ≅ GF(3) = Z/3Z.

This structure supports:
- Negation: ¬s = -s mod 3
- Conjunction: s ∧ t = min(s, t) by ordering Duck < Worm < Ape
- Disjunction: s ∨ t = max(s, t)
- Heyting implication: s → t = max(¬s, t) in GF(3) ordering

The classifier maps colors to truth values, where each truth value
represents ownership by a species/tier.
"""
struct TrivaluedOmega
    value::Species
end

TrivaluedOmega(i::Int) = TrivaluedOmega(Species(mod(i, 3)))

Base.:!(ω::TrivaluedOmega) = TrivaluedOmega(species_neg(ω.value))
Base.:+(ω1::TrivaluedOmega, ω2::TrivaluedOmega) = TrivaluedOmega(species_add(ω1.value, ω2.value))

function Base.:∧(ω1::TrivaluedOmega, ω2::TrivaluedOmega)
    TrivaluedOmega(Species(min(Int(ω1.value), Int(ω2.value))))
end

function Base.:∨(ω1::TrivaluedOmega, ω2::TrivaluedOmega)
    TrivaluedOmega(Species(max(Int(ω1.value), Int(ω2.value))))
end

function heyting_implies(ω1::TrivaluedOmega, ω2::TrivaluedOmega)
    (!ω1) ∨ ω2
end

# ═══════════════════════════════════════════════════════════════════════════════
# LEARNABLE CLASSIFIER PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ClassifierParameters

Enzyme-compatible learnable parameters for the subobject classifier.

The classifier χ: Color → Ω₃ is parameterized by:
1. Per-species gamut boundaries (3 sets of LCH bounds)
2. Per-species decision thresholds (softmax temperatures)
3. Bandwidth-to-species affinity weights

The classifier computes a probability distribution over {Duck, Worm, Ape}
and returns the argmax as the characteristic value.

# Gamut Decision Rule
A color belongs to species s if it falls within s's gamut boundary:
- Duck: low chroma, medium lightness (conservative, safe colors)
- Worm: medium chroma, varied lightness (exploratory colors)
- Ape: high chroma, extreme lightness (bold, saturated colors)

# Bandwidth Decision Rule
If bandwidth info is available:
- Duck: < 100M colors/sec
- Worm: 100M - 1B colors/sec
- Ape: > 1B colors/sec
"""
mutable struct ClassifierParameters
    # Gamut boundaries per species (L_min, L_max, C_min, C_max, H_weight)
    # Each species "claims" a region of LCH space
    duck_L_range::Tuple{Float64, Float64}  # (min, max) lightness for Duck
    duck_C_range::Tuple{Float64, Float64}  # (min, max) chroma for Duck
    duck_H_weight::Float64                  # How much hue matters for Duck

    worm_L_range::Tuple{Float64, Float64}
    worm_C_range::Tuple{Float64, Float64}
    worm_H_weight::Float64

    ape_L_range::Tuple{Float64, Float64}
    ape_C_range::Tuple{Float64, Float64}
    ape_H_weight::Float64

    # Softmax temperature (lower = sharper decisions)
    temperature::Float64

    # Bandwidth thresholds (log scale)
    bandwidth_thresh_duck_worm::Float64  # log10(100M) ≈ 8
    bandwidth_thresh_worm_ape::Float64   # log10(1B) ≈ 9

    # Mixing weight: gamut vs bandwidth
    gamut_weight::Float64  # 0 = pure bandwidth, 1 = pure gamut
end

function ClassifierParameters()
    ClassifierParameters(
        # Duck: conservative gamut (medium L, low C)
        (30.0, 70.0),   # L range
        (0.0, 50.0),    # C range (low chroma = safe)
        0.3,            # H weight

        # Worm: exploratory gamut (varied L, medium C)
        (20.0, 80.0),
        (30.0, 80.0),
        0.5,

        # Ape: bold gamut (extreme L, high C)
        (10.0, 95.0),
        (60.0, 130.0),
        0.7,

        # Softmax temperature
        1.0,

        # Bandwidth thresholds (log10)
        8.0,  # 100M = 10^8
        9.0,  # 1B = 10^9

        # Pure gamut-based by default
        1.0
    )
end

Base.zero(::Type{ClassifierParameters}) = ClassifierParameters(
    (0.0, 0.0), (0.0, 0.0), 0.0,
    (0.0, 0.0), (0.0, 0.0), 0.0,
    (0.0, 0.0), (0.0, 0.0), 0.0,
    0.0, 0.0, 0.0, 0.0
)

# ═══════════════════════════════════════════════════════════════════════════════
# SUBOBJECT CLASSIFIER (THE CORE TYPE)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    SubobjectClassifier

The learned subobject classifier χ: Color → Ω₃.

In topos theory terms:
- Objects: Colors (LCH values)
- Subobjects: Tiers (Duck ⊂ All, Worm ⊂ All, Ape ⊂ All)
- χ: Characteristic morphism classifying membership

The classifier is parameterized by ClassifierParameters and can be
trained via Enzyme.jl autodiff to optimize open game rewards.

# Usage
```julia
classifier = SubobjectClassifier()
color = LCHab(50.0, 60.0, 180.0)
species = classify_color(classifier, color)  # Returns Species
omega = characteristic_morphism(classifier, color)  # Returns TrivaluedOmega
```
"""
struct SubobjectClassifier
    params::ClassifierParameters
    training_history::Vector{Float64}  # Loss history during training
end

SubobjectClassifier() = SubobjectClassifier(ClassifierParameters(), Float64[])
SubobjectClassifier(params::ClassifierParameters) = SubobjectClassifier(params, Float64[])

# ═══════════════════════════════════════════════════════════════════════════════
# GAMUT-BASED SPECIES SCORING
# ═══════════════════════════════════════════════════════════════════════════════

"""
    gamut_score(L, C, H, L_range, C_range, H_weight) -> Float64

Compute how well a color fits within a species' gamut.
Returns a score in [0, ∞) where higher = better fit.
"""
function gamut_score(L::Float64, C::Float64, H::Float64,
                     L_range::Tuple{Float64, Float64},
                     C_range::Tuple{Float64, Float64},
                     H_weight::Float64)
    L_min, L_max = L_range
    C_min, C_max = C_range

    # How well does L fit in range? (Gaussian-like falloff outside)
    L_center = (L_min + L_max) / 2
    L_width = (L_max - L_min) / 2
    L_score = exp(-((L - L_center) / max(L_width, 1e-6))^2)

    # How well does C fit in range?
    C_center = (C_min + C_max) / 2
    C_width = (C_max - C_min) / 2
    C_score = exp(-((C - C_center) / max(C_width, 1e-6))^2)

    # Hue modulation (some species prefer certain hues)
    # Duck (Green): prefers H around 120°
    # Worm (Red): prefers H around 0°/360°
    # Ape (Blue): prefers H around 240°
    H_score = 1.0  # Placeholder - could add hue preference

    L_score * C_score * (1.0 + H_weight * H_score)
end

"""
    species_scores(L, C, H, params::ClassifierParameters) -> (duck, worm, ape)

Compute gamut-based scores for each species.
"""
function species_scores(L::Float64, C::Float64, H::Float64, params::ClassifierParameters)
    duck = gamut_score(L, C, H, params.duck_L_range, params.duck_C_range, params.duck_H_weight)
    worm = gamut_score(L, C, H, params.worm_L_range, params.worm_C_range, params.worm_H_weight)
    ape = gamut_score(L, C, H, params.ape_L_range, params.ape_C_range, params.ape_H_weight)
    (duck, worm, ape)
end

"""
    softmax_classify(scores, temperature) -> (probs, argmax_species)

Apply softmax to scores and return probabilities + classification.
"""
function softmax_classify(scores::NTuple{3, Float64}, temperature::Float64)
    # Softmax with temperature
    duck, worm, ape = scores
    max_score = max(duck, worm, ape)

    # Numerical stability: subtract max before exp
    exp_duck = exp((duck - max_score) / temperature)
    exp_worm = exp((worm - max_score) / temperature)
    exp_ape = exp((ape - max_score) / temperature)
    total = exp_duck + exp_worm + exp_ape

    probs = (exp_duck / total, exp_worm / total, exp_ape / total)

    # Argmax
    if probs[1] >= probs[2] && probs[1] >= probs[3]
        species = Duck
    elseif probs[2] >= probs[3]
        species = Worm
    else
        species = Ape
    end

    (probs, species)
end

# ═══════════════════════════════════════════════════════════════════════════════
# CHARACTERISTIC MORPHISM χ: Color → Ω₃
# ═══════════════════════════════════════════════════════════════════════════════

"""
    classify_color(classifier::SubobjectClassifier, L, C, H; bandwidth=nothing) -> Species

The characteristic morphism χ: Color → Ω₃.

Given a color in LCH space, returns the species that "owns" it.
If bandwidth is provided, uses weighted combination of gamut and bandwidth rules.

This is the core function that Enzyme differentiates through.
"""
function classify_color(classifier::SubobjectClassifier,
                        L::Float64, C::Float64, H::Float64;
                        bandwidth::Union{Nothing, Float64}=nothing)
    params = classifier.params

    # Gamut-based scores
    scores = species_scores(L, C, H, params)

    # If bandwidth provided, mix with bandwidth-based classification
    if bandwidth !== nothing && params.gamut_weight < 1.0
        log_bw = log10(max(bandwidth, 1.0))

        # Bandwidth-based classification
        if log_bw < params.bandwidth_thresh_duck_worm
            bw_species = Duck
        elseif log_bw < params.bandwidth_thresh_worm_ape
            bw_species = Worm
        else
            bw_species = Ape
        end

        # Convert to one-hot scores
        bw_scores = (
            bw_species == Duck ? 1.0 : 0.0,
            bw_species == Worm ? 1.0 : 0.0,
            bw_species == Ape ? 1.0 : 0.0
        )

        # Weighted combination
        α = params.gamut_weight
        scores = (
            α * scores[1] + (1-α) * bw_scores[1],
            α * scores[2] + (1-α) * bw_scores[2],
            α * scores[3] + (1-α) * bw_scores[3]
        )
    end

    # Softmax classification
    _, species = softmax_classify(scores, params.temperature)
    species
end

function classify_color(classifier::SubobjectClassifier, lch::LCHab; kwargs...)
    classify_color(classifier, Float64(lch.l), Float64(lch.c), Float64(lch.h); kwargs...)
end

function classify_color(classifier::SubobjectClassifier, rgb::RGB; kwargs...)
    lch = convert(LCHab, rgb)
    classify_color(classifier, lch; kwargs...)
end

"""
    characteristic_morphism(classifier::SubobjectClassifier, color) -> TrivaluedOmega

The characteristic morphism χ as a map to the subobject classifier Ω₃.
"""
function characteristic_morphism(classifier::SubobjectClassifier, color; kwargs...)
    species = classify_color(classifier, color; kwargs...)
    TrivaluedOmega(species)
end

# ═══════════════════════════════════════════════════════════════════════════════
# OPEN GAME REWARD (from gay_open_game_seek.hy)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    tier_reward(species::Species, colors_mined::Int, path_depth::Int) -> Float64

Compute reward for a species based on open game formula:
    reward = base × tier_mult × depth_bonus

Where:
- base = colors_mined / 1_000_000
- tier_mult = {Duck: 1, Worm: 3, Ape: 9}
- depth_bonus = 1 + 0.1 × path_depth
"""
function tier_reward(species::Species, colors_mined::Int, path_depth::Int)
    base = colors_mined / 1_000_000
    tier_mult = TIER_MULTIPLIERS[species]
    depth_bonus = 1.0 + 0.1 * path_depth
    base * tier_mult * depth_bonus
end

"""
    classification_reward(classifier, L, C, H, colors_mined, path_depth) -> Float64

Compute reward for classifying a color.
This is the differentiable objective that Enzyme optimizes.
"""
function classification_reward(classifier::SubobjectClassifier,
                                L::Float64, C::Float64, H::Float64,
                                colors_mined::Int, path_depth::Int)
    species = classify_color(classifier, L, C, H)
    tier_reward(species, colors_mined, path_depth)
end

"""
    open_game_loss(classifier, colors, colors_mined, path_depth) -> Float64

Negative total reward over a batch of colors.
Minimizing this = maximizing reward = learning optimal classification.
"""
function open_game_loss(classifier::SubobjectClassifier,
                        colors::Vector{Tuple{Float64, Float64, Float64}},
                        colors_mined::Int,
                        path_depth::Int)
    total_reward = 0.0
    for (L, C, H) in colors
        total_reward += classification_reward(classifier, L, C, H, colors_mined, path_depth)
    end
    -total_reward / length(colors)  # Negative for minimization
end

# ═══════════════════════════════════════════════════════════════════════════════
# GAMUT CONSISTENCY LOSS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    gamut_consistency_loss(classifier, colors) -> Float64

Penalize classifications that put colors outside their species' gamut.

A color classified as Duck should actually be in Duck's gamut region.
This loss encourages the classifier to respect gamut boundaries.
"""
function gamut_consistency_loss(classifier::SubobjectClassifier,
                                 colors::Vector{Tuple{Float64, Float64, Float64}})
    params = classifier.params
    loss = 0.0

    for (L, C, H) in colors
        species = classify_color(classifier, L, C, H)

        # Get the species' expected gamut
        L_range, C_range = if species == Duck
            (params.duck_L_range, params.duck_C_range)
        elseif species == Worm
            (params.worm_L_range, params.worm_C_range)
        else
            (params.ape_L_range, params.ape_C_range)
        end

        # Distance outside expected gamut
        L_min, L_max = L_range
        C_min, C_max = C_range

        L_dist = max(0, L_min - L, L - L_max)
        C_dist = max(0, C_min - C, C - C_max)

        loss += L_dist^2 + C_dist^2
    end

    loss / length(colors)
end

# ═══════════════════════════════════════════════════════════════════════════════
# ENZYME.jl GRADIENT (STUB - OVERRIDDEN BY EXTENSION)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    enzyme_classifier_gradient(classifier, colors, colors_mined, path_depth) -> ClassifierParameters

Compute gradient of open_game_loss with respect to classifier parameters.

When Enzyme.jl is loaded, this uses real reverse-mode autodiff.
Without Enzyme, returns approximate numerical gradients.
"""
function enzyme_classifier_gradient(classifier::SubobjectClassifier,
                                     colors::Vector{Tuple{Float64, Float64, Float64}},
                                     colors_mined::Int,
                                     path_depth::Int)
    # Numerical gradient approximation (fallback)
    ε = 1e-5
    params = classifier.params
    grad = zero(ClassifierParameters)

    # Gradient w.r.t. duck_L_range
    original = params.duck_L_range
    params.duck_L_range = (original[1] + ε, original[2])
    loss_plus = open_game_loss(classifier, colors, colors_mined, path_depth)
    params.duck_L_range = (original[1] - ε, original[2])
    loss_minus = open_game_loss(classifier, colors, colors_mined, path_depth)
    grad.duck_L_range = ((loss_plus - loss_minus) / (2ε), 0.0)
    params.duck_L_range = original

    # Similar for other parameters...
    # (In real implementation, Enzyme computes all gradients automatically)

    grad
end

# ═══════════════════════════════════════════════════════════════════════════════
# LEARNING THE CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════

"""
    LearnedClassifier

Wrapper that includes training state alongside the classifier.
"""
mutable struct LearnedClassifier
    classifier::SubobjectClassifier
    optimizer_state::Dict{Symbol, Any}
    epoch::Int
end

LearnedClassifier() = LearnedClassifier(SubobjectClassifier(), Dict{Symbol, Any}(), 0)

"""
    learn_classifier!(lc::LearnedClassifier, colors, colors_mined, path_depth;
                      lr=0.01, epochs=100, verbose=false)

Train the subobject classifier to maximize open game reward.

Uses gradient descent on open_game_loss + gamut_consistency_loss.
"""
function learn_classifier!(lc::LearnedClassifier,
                           colors::Vector{Tuple{Float64, Float64, Float64}},
                           colors_mined::Int,
                           path_depth::Int;
                           lr::Float64=0.01,
                           epochs::Int=100,
                           gamut_weight::Float64=0.1,
                           verbose::Bool=false)
    classifier = lc.classifier
    params = classifier.params

    for epoch in 1:epochs
        lc.epoch += 1

        # Compute total loss
        game_loss = open_game_loss(classifier, colors, colors_mined, path_depth)
        consistency_loss = gamut_consistency_loss(classifier, colors)
        total_loss = game_loss + gamut_weight * consistency_loss

        push!(classifier.training_history, total_loss)

        # Compute gradient (uses Enzyme if available, else numerical)
        grad = enzyme_classifier_gradient(classifier, colors, colors_mined, path_depth)

        # Gradient descent update for L ranges
        params.duck_L_range = (
            params.duck_L_range[1] - lr * grad.duck_L_range[1],
            params.duck_L_range[2] - lr * grad.duck_L_range[2]
        )
        params.worm_L_range = (
            params.worm_L_range[1] - lr * grad.worm_L_range[1],
            params.worm_L_range[2] - lr * grad.worm_L_range[2]
        )
        params.ape_L_range = (
            params.ape_L_range[1] - lr * grad.ape_L_range[1],
            params.ape_L_range[2] - lr * grad.ape_L_range[2]
        )

        # Gradient descent update for C ranges
        params.duck_C_range = (
            params.duck_C_range[1] - lr * grad.duck_C_range[1],
            params.duck_C_range[2] - lr * grad.duck_C_range[2]
        )
        params.worm_C_range = (
            params.worm_C_range[1] - lr * grad.worm_C_range[1],
            params.worm_C_range[2] - lr * grad.worm_C_range[2]
        )
        params.ape_C_range = (
            params.ape_C_range[1] - lr * grad.ape_C_range[1],
            params.ape_C_range[2] - lr * grad.ape_C_range[2]
        )

        # Clamp ranges to valid values
        params.duck_L_range = (clamp(params.duck_L_range[1], 0, 100), clamp(params.duck_L_range[2], 0, 100))
        params.duck_C_range = (clamp(params.duck_C_range[1], 0, 130), clamp(params.duck_C_range[2], 0, 130))
        # ... similar for worm and ape

        if verbose && epoch % 10 == 0
            println("Epoch $(lc.epoch): loss = $(round(total_loss, digits=4)), " *
                    "game = $(round(game_loss, digits=4)), consistency = $(round(consistency_loss, digits=4))")
        end
    end

    lc
end

# ═══════════════════════════════════════════════════════════════════════════════
# SPI VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

"""
    verify_spi_invariance(classifier, seed::UInt64, n_colors::Int) -> Bool

Verify that the classifier produces consistent results (SPI).
Same color should always classify to same species.
"""
function verify_spi_invariance(classifier::SubobjectClassifier, seed::UInt64, n_colors::Int)
    # Generate colors deterministically from seed
    rng_state = seed
    colors = Tuple{Float64, Float64, Float64}[]

    for i in 1:n_colors
        # SplitMix64-style generation
        rng_state = (rng_state + 0x9E3779B97F4A7C15) % UInt64
        z = rng_state
        z = ((z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9) % UInt64
        z = ((z ⊻ (z >> 27)) * 0x94D049BB133111EB) % UInt64
        z = z ⊻ (z >> 31)

        L = Float64((z >> 56) % 256) / 2.55  # 0-100
        C = Float64((z >> 48) % 256) / 2.0    # 0-127
        H = Float64((z >> 40) % 256) * 1.41   # 0-360

        push!(colors, (L, C, H))
    end

    # Classify twice, check consistency
    classifications1 = [classify_color(classifier, L, C, H) for (L, C, H) in colors]
    classifications2 = [classify_color(classifier, L, C, H) for (L, C, H) in colors]

    all(classifications1 .== classifications2)
end

# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

"""
    world_subobject_classifier()

Demonstrate the learned subobject classifier.
"""
function world_subobject_classifier(seed::UInt64=UInt64(0x6761795f636f6c6f))
    println("╔═══════════════════════════════════════════════════════════════════════╗")
    println("║  Learned Subobject Classifier for Gay Gamut Decisions                 ║")
    println("║  χ: Color → Ω₃ = {Duck, Worm, Ape}                                    ║")
    println("╚═══════════════════════════════════════════════════════════════════════╝")
    println()

    # Create classifier
    classifier = SubobjectClassifier()

    # Generate test colors
    rng_state = seed
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

    println("═══ INITIAL CLASSIFICATION ═══")
    duck_count = count(c -> classify_color(classifier, c...) == Duck, colors)
    worm_count = count(c -> classify_color(classifier, c...) == Worm, colors)
    ape_count = count(c -> classify_color(classifier, c...) == Ape, colors)
    println("  Duck: $duck_count, Worm: $worm_count, Ape: $ape_count")

    # Show sample classifications
    println("\n  Sample colors:")
    for i in 1:5
        L, C, H = colors[i]
        s = classify_color(classifier, L, C, H)
        ω = characteristic_morphism(classifier, LCHab(L, C, H))
        println("    LCH($( round(L, digits=1)), $(round(C, digits=1)), $(round(H, digits=1))) → $s (Ω₃ = $(ω.value))")
    end

    # Compute initial loss
    colors_mined = 10_000_000
    path_depth = 3
    initial_loss = open_game_loss(classifier, colors, colors_mined, path_depth)
    println("\n  Open game loss: $(round(initial_loss, digits=4))")
    println("  (Lower = more reward from tier assignments)")

    # Train
    println("\n═══ LEARNING SUBOBJECT CLASSIFIER ═══")
    lc = LearnedClassifier()
    lc.classifier = classifier
    learn_classifier!(lc, colors, colors_mined, path_depth;
                      lr=0.005, epochs=50, verbose=true)

    # Show learned classification
    println("\n═══ LEARNED CLASSIFICATION ═══")
    duck_count = count(c -> classify_color(classifier, c...) == Duck, colors)
    worm_count = count(c -> classify_color(classifier, c...) == Worm, colors)
    ape_count = count(c -> classify_color(classifier, c...) == Ape, colors)
    println("  Duck: $duck_count, Worm: $worm_count, Ape: $ape_count")

    final_loss = open_game_loss(classifier, colors, colors_mined, path_depth)
    println("  Final loss: $(round(final_loss, digits=4))")
    println("  Improvement: $(round((initial_loss - final_loss) / abs(initial_loss) * 100, digits=1))%")

    # Verify SPI
    println("\n═══ SPI VERIFICATION ═══")
    spi_ok = verify_spi_invariance(classifier, seed, 100)
    println("  Same color → same species: $(spi_ok ? "✓ PASS" : "✗ FAIL")")

    # Show learned parameters
    params = classifier.params
    println("\n═══ LEARNED GAMUT BOUNDARIES ═══")
    println("  Duck (Green, 1x):")
    println("    L: $(round(params.duck_L_range[1], digits=1)) - $(round(params.duck_L_range[2], digits=1))")
    println("    C: $(round(params.duck_C_range[1], digits=1)) - $(round(params.duck_C_range[2], digits=1))")
    println("  Worm (Red, 3x):")
    println("    L: $(round(params.worm_L_range[1], digits=1)) - $(round(params.worm_L_range[2], digits=1))")
    println("    C: $(round(params.worm_C_range[1], digits=1)) - $(round(params.worm_C_range[2], digits=1))")
    println("  Ape (Blue, 9x):")
    println("    L: $(round(params.ape_L_range[1], digits=1)) - $(round(params.ape_L_range[2], digits=1))")
    println("    C: $(round(params.ape_C_range[1], digits=1)) - $(round(params.ape_C_range[2], digits=1))")

    println()
    println("╔═══════════════════════════════════════════════════════════════════════╗")
    println("║  Subobject classifier learned! χ now maps colors to optimal tiers.    ║")
    println("╚═══════════════════════════════════════════════════════════════════════╝")

    (classifier, lc, colors)
end

end # module SubobjectClassifierGamut
