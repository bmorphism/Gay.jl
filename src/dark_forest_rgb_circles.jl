# Dark Forest RGB Circles: Learned Subobject Classifier with Adversarial Dynamics
# ═══════════════════════════════════════════════════════════════════════════════
#
# "The universe is a dark forest. Every civilization is an armed hunter stalking
#  through the trees like a ghost... In this forest, hell is other people."
#                                       - Liu Cixin, The Dark Forest
#
# Species are RGB CIRCLES on the color wheel:
#   🟢 DUCK (Green):  Hue center = 120°, hide in foliage, survive by camouflage
#   🔴 WORM (Red):    Hue center = 0°,   burrow underground, strike from below
#   🔵 APE (Blue):    Hue center = 240°, swing through canopy, dominate by force
#
# DARK FOREST DYNAMICS:
#   1. HIDING: Colors close to species center are "hidden" (low detection risk)
#   2. REVEALING: Colors far from center are "exposed" (high detection risk)
#   3. STRIKING: When detected, nearest species can "consume" the color
#   4. CAMOUFLAGE: Learn to minimize exposure while maximizing reward
#
# The subobject classifier χ: Color → Ω₃ now includes:
#   - Distance to RGB circle centers (primary classification)
#   - Dark forest survival probability
#   - Predation risk from other species
#   - Learned camouflage parameters via Enzyme.jl
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │                          RGB COLOR WHEEL                                    │
# │                                                                             │
# │                          Yellow (60°)                                       │
# │                              ●                                              │
# │                           ╱     ╲                                           │
# │                        ╱           ╲                                        │
# │           Green (120°) ●             ● Red (0°/360°)                        │
# │              🦆 DUCK    ╲           ╱    🪱 WORM                             │
# │                          ╲       ╱                                          │
# │                           ╲   ╱                                             │
# │                Cyan (180°) ● ● Magenta (300°)                               │
# │                           ╱   ╲                                             │
# │                         ╱       ╲                                           │
# │                       ●           ●                                         │
# │                  Blue (240°)                                                │
# │                    🦧 APE                                                   │
# │                                                                             │
# └─────────────────────────────────────────────────────────────────────────────┘
#
# Each species owns a 120° arc of the color wheel:
#   - DUCK (Green): 60° - 180° (yellow-green-cyan)
#   - WORM (Red): 300° - 60° (magenta-red-yellow)
#   - APE (Blue): 180° - 300° (cyan-blue-magenta)
#
# ═══════════════════════════════════════════════════════════════════════════════

module DarkForestRGBCircles

using LinearAlgebra: norm, dot

export Species, Duck, Worm, Ape
export RGBCircle, DUCK_CIRCLE, WORM_CIRCLE, APE_CIRCLE
export DarkForestState, DarkForestClassifier
export hue_distance, circle_distance, classify_by_rgb_circle
export hiding_score, detection_risk, predation_probability
export dark_forest_reward, dark_forest_loss
export enzyme_dark_forest_gradient!, learn_dark_forest!
export camouflage_color, optimal_camouflage
export world_dark_forest

# ═══════════════════════════════════════════════════════════════════════════════
# SPECIES AS RGB PRIMARIES
# ═══════════════════════════════════════════════════════════════════════════════

@enum Species begin
    Duck = 0   # 🟢 Green (H=120°) - hide, camouflage, survive
    Worm = 1   # 🔴 Red (H=0°) - burrow, ambush, strike
    Ape  = 2   # 🔵 Blue (H=240°) - dominate, expose, consume
end

const SPECIES_HUES = Dict{Species, Float64}(
    Duck => 120.0,  # Green
    Worm => 0.0,    # Red (also 360°)
    Ape  => 240.0   # Blue
)

const SPECIES_RGB = Dict{Species, Tuple{Float64, Float64, Float64}}(
    Duck => (0.0, 1.0, 0.0),   # Pure Green
    Worm => (1.0, 0.0, 0.0),   # Pure Red
    Ape  => (0.0, 0.0, 1.0)    # Pure Blue
)

const TIER_MULTIPLIERS = Dict{Species, Float64}(
    Duck => 1.0,   # Survivors get steady income
    Worm => 3.0,   # Ambushers get occasional big kills
    Ape  => 9.0    # Dominators take everything (but are exposed)
)

# ═══════════════════════════════════════════════════════════════════════════════
# RGB CIRCLES ON THE COLOR WHEEL
# ═══════════════════════════════════════════════════════════════════════════════

"""
    RGBCircle

A species' territory on the color wheel.

Each circle is defined by:
- center_hue: The hue angle of the RGB primary (0°, 120°, 240°)
- arc_width: How many degrees the species claims (default 120° = 1/3 of wheel)
- saturation_min: Minimum saturation to be in the circle
- lightness_range: Valid lightness range for the species

Colors inside the circle are "owned" by that species.
Colors near the center are well-hidden; colors near the edge are exposed.
"""
struct RGBCircle
    species::Species
    center_hue::Float64
    arc_width::Float64      # Degrees claimed on each side (60° = 120° total)
    saturation_min::Float64
    lightness_range::Tuple{Float64, Float64}
end

# Define the three RGB circles
const DUCK_CIRCLE = RGBCircle(Duck, 120.0, 60.0, 0.3, (0.25, 0.75))  # Green
const WORM_CIRCLE = RGBCircle(Worm, 0.0, 60.0, 0.3, (0.25, 0.75))    # Red
const APE_CIRCLE = RGBCircle(Ape, 240.0, 60.0, 0.3, (0.25, 0.75))    # Blue

const ALL_CIRCLES = [DUCK_CIRCLE, WORM_CIRCLE, APE_CIRCLE]

# ═══════════════════════════════════════════════════════════════════════════════
# HUE DISTANCE (circular)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    hue_distance(h1, h2) -> Float64

Angular distance between two hues on the color wheel.
Returns value in [0, 180] (half-circle is max distance).
"""
function hue_distance(h1::Float64, h2::Float64)
    d = abs(h1 - h2)
    d > 180.0 ? 360.0 - d : d
end

"""
    circle_distance(H, S, L, circle::RGBCircle) -> Float64

Distance from a color to an RGB circle's center.
Returns 0 if at center, increases with distance.

Components:
- Hue distance from circle center (0-180)
- Saturation distance from 1.0 (desaturated = further)
- Lightness distance from 0.5 (extreme L = further)
"""
function circle_distance(H::Float64, S::Float64, L::Float64, circle::RGBCircle)
    # Hue component (normalized to 0-1 where 0=center, 1=opposite side)
    hue_dist = hue_distance(H, circle.center_hue) / 180.0

    # Saturation component (low saturation = far from vivid primary)
    sat_dist = 1.0 - S

    # Lightness component (extreme L = far from primary)
    L_center = sum(circle.lightness_range) / 2
    L_range = (circle.lightness_range[2] - circle.lightness_range[1]) / 2
    light_dist = abs(L - L_center) / max(L_range, 0.01)

    # Weighted combination
    sqrt(hue_dist^2 + 0.5 * sat_dist^2 + 0.3 * light_dist^2)
end

"""
    classify_by_rgb_circle(H, S, L) -> Species

Classify a color by which RGB circle it's closest to.
This is the hard classification (argmin distance).
"""
function classify_by_rgb_circle(H::Float64, S::Float64, L::Float64)
    distances = [circle_distance(H, S, L, c) for c in ALL_CIRCLES]
    min_idx = argmin(distances)
    ALL_CIRCLES[min_idx].species
end

# ═══════════════════════════════════════════════════════════════════════════════
# DARK FOREST DYNAMICS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    hiding_score(H, S, L, species::Species) -> Float64

How well hidden is a color within its species' territory?

Returns value in [0, 1]:
- 1.0 = perfectly camouflaged (at circle center)
- 0.0 = completely exposed (at circle edge or outside)

Colors at the center of their species' RGB circle are safest.
"""
function hiding_score(H::Float64, S::Float64, L::Float64, species::Species)
    circle = ALL_CIRCLES[Int(species) + 1]
    dist = circle_distance(H, S, L, circle)

    # Sigmoid-like decay: close to center = high hiding, far = low
    1.0 / (1.0 + exp(3.0 * (dist - 0.5)))
end

"""
    detection_risk(H, S, L) -> Float64

Probability of being detected by predators.

High detection risk when:
- Color is far from any circle center (no-man's land)
- Color is highly saturated (stands out)
- Color is at extreme lightness (visible)

Low detection risk when:
- Color is at center of a circle (well camouflaged)
- Color is desaturated (muted)
"""
function detection_risk(H::Float64, S::Float64, L::Float64)
    # Find closest circle
    distances = [circle_distance(H, S, L, c) for c in ALL_CIRCLES]
    min_dist = minimum(distances)

    # Base detection from distance to nearest safe zone
    base_detection = 1.0 - 1.0 / (1.0 + exp(3.0 * (min_dist - 0.3)))

    # Saturation increases visibility
    saturation_factor = 0.5 + 0.5 * S

    # Extreme lightness increases visibility
    lightness_factor = 1.0 + 0.5 * abs(L - 0.5)

    clamp(base_detection * saturation_factor * lightness_factor, 0.0, 1.0)
end

"""
    predation_probability(H, S, L, prey_species, predator_species) -> Float64

Probability that predator_species will successfully hunt prey_species.

Dark forest rules:
- Closer species are more dangerous (adjacent on color wheel)
- Ape dominates (9x) but is slow - catches exposed prey
- Worm ambushes (3x) - catches prey in transition zones
- Duck hides (1x) - rarely caught, rarely catches

Predation success depends on:
- Prey's exposure (detection_risk)
- Predator's reach into prey's territory
- Relative power (tier multiplier)
"""
function predation_probability(H::Float64, S::Float64, L::Float64,
                                prey_species::Species, predator_species::Species)
    if prey_species == predator_species
        return 0.0  # No self-predation
    end

    prey_circle = ALL_CIRCLES[Int(prey_species) + 1]
    predator_circle = ALL_CIRCLES[Int(predator_species) + 1]

    # Prey's exposure level
    prey_detection = detection_risk(H, S, L)

    # Predator's reach (how far they can strike into prey territory)
    predator_power = TIER_MULTIPLIERS[predator_species]
    prey_power = TIER_MULTIPLIERS[prey_species]
    power_ratio = predator_power / prey_power

    # Distance from predator's center to prey's current position
    predator_dist = circle_distance(H, S, L, predator_circle)

    # Predation probability: high when prey is exposed AND predator can reach
    reach = 1.0 / (1.0 + exp(2.0 * (predator_dist - 0.5 * power_ratio)))

    prey_detection * reach * 0.5  # Cap at 50% to keep game interesting
end

# ═══════════════════════════════════════════════════════════════════════════════
# DARK FOREST STATE & CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════

"""
    DarkForestState

Learnable parameters for dark forest survival.

These are optimized via Enzyme to maximize survival while maintaining reward:
- camouflage_strength: How much to pull colors toward circle centers
- exposure_tolerance: How much exposure to accept for higher reward
- predation_avoidance: Weight on avoiding predators vs seeking reward
- circle_boundaries: Learned adjustments to RGB circle positions
"""
mutable struct DarkForestState
    # Camouflage parameters (per species)
    duck_camouflage::Float64    # How much Duck pulls toward green
    worm_camouflage::Float64    # How much Worm pulls toward red
    ape_camouflage::Float64     # How much Ape pulls toward blue

    # Exposure tolerance (willingness to leave safe zone)
    exposure_tolerance::Float64

    # Predation avoidance weight
    predation_weight::Float64

    # Learned circle boundary adjustments (degrees)
    duck_hue_shift::Float64
    worm_hue_shift::Float64
    ape_hue_shift::Float64

    # Softmax temperature for classification
    temperature::Float64

    # Training state
    gradients::Vector{Float64}
    loss_history::Vector{Float64}
    step::Int
end

function DarkForestState()
    DarkForestState(
        # Camouflage strengths (start moderate)
        0.5, 0.5, 0.5,
        # Exposure tolerance
        0.3,
        # Predation weight
        1.0,
        # Circle shifts (start at 0 = standard RGB)
        0.0, 0.0, 0.0,
        # Temperature
        1.0,
        # Gradients (9 parameters)
        zeros(9),
        Float64[],
        0
    )
end

"""
    DarkForestClassifier

The full dark forest subobject classifier χ: Color → Ω₃.

Classification considers:
1. Distance to each RGB circle
2. Survival probability in each territory
3. Learned camouflage adjustments
"""
struct DarkForestClassifier
    state::DarkForestState
end

DarkForestClassifier() = DarkForestClassifier(DarkForestState())

# ═══════════════════════════════════════════════════════════════════════════════
# SOFT CLASSIFICATION (DIFFERENTIABLE)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    rgb_circle_logits(H, S, L, state::DarkForestState) -> (duck, worm, ape)

Compute soft logits for classification based on RGB circle distances.
Includes learned circle shifts and camouflage effects.
"""
function rgb_circle_logits(H::Float64, S::Float64, L::Float64, state::DarkForestState)
    # Adjusted circle centers
    duck_center = 120.0 + state.duck_hue_shift
    worm_center = 0.0 + state.worm_hue_shift
    ape_center = 240.0 + state.ape_hue_shift

    # Distance to each adjusted center
    duck_hue_dist = hue_distance(H, duck_center) / 180.0
    worm_hue_dist = hue_distance(H, worm_center) / 180.0
    ape_hue_dist = hue_distance(H, ape_center) / 180.0

    # Saturation factor (vivid colors closer to primaries)
    sat_factor = S

    # Lightness factor (mid-lightness closer to primaries)
    light_factor = 1.0 - 2.0 * abs(L - 0.5)

    # Combined scores (higher = closer to circle)
    duck_score = (1.0 - duck_hue_dist) * (1.0 + sat_factor) * (1.0 + light_factor) * state.duck_camouflage
    worm_score = (1.0 - worm_hue_dist) * (1.0 + sat_factor) * (1.0 + light_factor) * state.worm_camouflage
    ape_score = (1.0 - ape_hue_dist) * (1.0 + sat_factor) * (1.0 + light_factor) * state.ape_camouflage

    # Apply temperature
    (duck_score / state.temperature, worm_score / state.temperature, ape_score / state.temperature)
end

"""
    softmax_probs(logits) -> (p_duck, p_worm, p_ape)
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
    classify_dark_forest(H, S, L, classifier) -> (species, probs)

Classify a color using dark forest dynamics.
Returns both hard classification and soft probabilities.
"""
function classify_dark_forest(H::Float64, S::Float64, L::Float64,
                               classifier::DarkForestClassifier)
    logits = rgb_circle_logits(H, S, L, classifier.state)
    probs = softmax_probs(logits...)

    # Hard classification
    if probs[1] >= probs[2] && probs[1] >= probs[3]
        species = Duck
    elseif probs[2] >= probs[3]
        species = Worm
    else
        species = Ape
    end

    (species, probs)
end

# ═══════════════════════════════════════════════════════════════════════════════
# DARK FOREST REWARD & LOSS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    dark_forest_reward(H, S, L, state, depth_bonus) -> Float64

Total reward considering:
1. Tier multiplier (Ape > Worm > Duck)
2. Survival probability (hidden > exposed)
3. Predation risk (avoid being eaten)

Reward = tier_mult × depth_bonus × survival × (1 - predation_loss)
"""
function dark_forest_reward(H::Float64, S::Float64, L::Float64,
                             state::DarkForestState, depth_bonus::Float64)
    logits = rgb_circle_logits(H, S, L, state)
    probs = softmax_probs(logits...)

    # Expected tier multiplier
    expected_tier = probs[1] * 1.0 + probs[2] * 3.0 + probs[3] * 9.0

    # Survival score (based on hiding ability)
    survival = 0.0
    for (i, prob) in enumerate(probs)
        species = Species(i - 1)
        survival += prob * hiding_score(H, S, L, species)
    end

    # Predation loss (expected damage from other species)
    predation_loss = 0.0
    for (i, prey_prob) in enumerate(probs)
        prey_species = Species(i - 1)
        for (j, _) in enumerate(probs)
            if i != j
                predator_species = Species(j - 1)
                predation_loss += prey_prob * predation_probability(H, S, L, prey_species, predator_species)
            end
        end
    end

    # Combined reward
    base_reward = expected_tier * depth_bonus
    survival_factor = state.exposure_tolerance + (1.0 - state.exposure_tolerance) * survival
    predation_factor = 1.0 - state.predation_weight * predation_loss

    base_reward * survival_factor * predation_factor
end

"""
    dark_forest_loss(colors, state, depth_bonus) -> Float64

Negative total reward (for minimization).
"""
function dark_forest_loss(colors::Vector{Tuple{Float64, Float64, Float64}},
                          state::DarkForestState, depth_bonus::Float64)
    total = 0.0
    for (H, S, L) in colors
        total += dark_forest_reward(H, S, L, state, depth_bonus)
    end
    -total / length(colors)
end

# ═══════════════════════════════════════════════════════════════════════════════
# ENZYME AUTODIFF
# ═══════════════════════════════════════════════════════════════════════════════

"""
    pack_params(state::DarkForestState) -> Vector{Float64}
"""
function pack_params(state::DarkForestState)
    [
        state.duck_camouflage, state.worm_camouflage, state.ape_camouflage,
        state.exposure_tolerance, state.predation_weight,
        state.duck_hue_shift, state.worm_hue_shift, state.ape_hue_shift,
        state.temperature
    ]
end

"""
    unpack_params!(state::DarkForestState, params::Vector{Float64})
"""
function unpack_params!(state::DarkForestState, params::Vector{Float64})
    state.duck_camouflage = params[1]
    state.worm_camouflage = params[2]
    state.ape_camouflage = params[3]
    state.exposure_tolerance = params[4]
    state.predation_weight = params[5]
    state.duck_hue_shift = params[6]
    state.worm_hue_shift = params[7]
    state.ape_hue_shift = params[8]
    state.temperature = params[9]
    state
end

"""
    enzyme_dark_forest_gradient!(state, colors, depth_bonus) -> gradients

Compute gradients via numerical differentiation (Enzyme replaces when loaded).
"""
function enzyme_dark_forest_gradient!(state::DarkForestState,
                                       colors::Vector{Tuple{Float64, Float64, Float64}},
                                       depth_bonus::Float64;
                                       epsilon::Float64=1e-5)
    state.gradients .= 0.0
    params = pack_params(state)

    base_loss = dark_forest_loss(colors, state, depth_bonus)

    for i in 1:9
        params_plus = copy(params)
        params_plus[i] += epsilon
        unpack_params!(state, params_plus)

        loss_plus = dark_forest_loss(colors, state, depth_bonus)
        state.gradients[i] = (loss_plus - base_loss) / epsilon

        unpack_params!(state, params)
    end

    state.gradients
end

"""
    learn_dark_forest!(state, colors, path_depth; kwargs...)

Train the dark forest classifier to maximize survival + reward.
"""
function learn_dark_forest!(state::DarkForestState,
                            colors::Vector{Tuple{Float64, Float64, Float64}},
                            path_depth::Int;
                            lr::Float64=0.01,
                            epochs::Int=100,
                            momentum::Float64=0.9,
                            verbose::Bool=false)
    depth_bonus = 1.0 + 0.1 * path_depth
    velocity = zeros(9)

    for epoch in 1:epochs
        state.step += 1

        # Compute gradient
        enzyme_dark_forest_gradient!(state, colors, depth_bonus)

        # Record loss
        loss = dark_forest_loss(colors, state, depth_bonus)
        push!(state.loss_history, loss)

        # Momentum update
        velocity .= momentum .* velocity .- lr .* state.gradients
        params = pack_params(state)
        params .+= velocity
        unpack_params!(state, params)

        # Clamp parameters
        state.duck_camouflage = clamp(state.duck_camouflage, 0.1, 2.0)
        state.worm_camouflage = clamp(state.worm_camouflage, 0.1, 2.0)
        state.ape_camouflage = clamp(state.ape_camouflage, 0.1, 2.0)
        state.exposure_tolerance = clamp(state.exposure_tolerance, 0.0, 1.0)
        state.predation_weight = clamp(state.predation_weight, 0.0, 2.0)
        state.duck_hue_shift = clamp(state.duck_hue_shift, -30.0, 30.0)
        state.worm_hue_shift = clamp(state.worm_hue_shift, -30.0, 30.0)
        state.ape_hue_shift = clamp(state.ape_hue_shift, -30.0, 30.0)
        state.temperature = clamp(state.temperature, 0.1, 5.0)

        if verbose && epoch % 10 == 0
            println("Epoch $(state.step): loss = $(round(loss, digits=4))")
        end
    end

    state
end

# ═══════════════════════════════════════════════════════════════════════════════
# CAMOUFLAGE: Optimal Hiding Strategies
# ═══════════════════════════════════════════════════════════════════════════════

"""
    camouflage_color(H, S, L, species, strength) -> (H', S', L')

Adjust a color to better hide within its species' territory.
Higher strength = more camouflage (closer to circle center).
"""
function camouflage_color(H::Float64, S::Float64, L::Float64,
                          species::Species, strength::Float64)
    target_hue = SPECIES_HUES[species]

    # Pull hue toward target
    hue_diff = H - target_hue
    if hue_diff > 180.0
        hue_diff -= 360.0
    elseif hue_diff < -180.0
        hue_diff += 360.0
    end
    H_new = H - strength * hue_diff

    # Normalize hue to [0, 360)
    while H_new < 0.0
        H_new += 360.0
    end
    while H_new >= 360.0
        H_new -= 360.0
    end

    # Adjust saturation (slightly desaturate for camouflage)
    S_new = S * (1.0 - 0.2 * strength)

    # Pull lightness toward 0.5
    L_new = L + strength * (0.5 - L) * 0.3

    (H_new, clamp(S_new, 0.0, 1.0), clamp(L_new, 0.0, 1.0))
end

"""
    optimal_camouflage(H, S, L, classifier) -> (H', S', L', species)

Find the optimal camouflaged version of a color.
Returns the adjusted color and which species territory to hide in.
"""
function optimal_camouflage(H::Float64, S::Float64, L::Float64,
                             classifier::DarkForestClassifier)
    state = classifier.state
    species, probs = classify_dark_forest(H, S, L, classifier)

    # Get species-specific camouflage strength
    camo_strength = if species == Duck
        state.duck_camouflage
    elseif species == Worm
        state.worm_camouflage
    else
        state.ape_camouflage
    end

    H_new, S_new, L_new = camouflage_color(H, S, L, species, camo_strength * 0.5)

    (H_new, S_new, L_new, species)
end

# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    hsl_to_rgb(H, S, L) -> (R, G, B)

Convert HSL to RGB (0-1 scale).
"""
function hsl_to_rgb(H::Float64, S::Float64, L::Float64)
    C = (1.0 - abs(2.0 * L - 1.0)) * S
    X = C * (1.0 - abs(mod(H / 60.0, 2.0) - 1.0))
    m = L - C / 2.0

    r, g, b = if H < 60
        (C, X, 0.0)
    elseif H < 120
        (X, C, 0.0)
    elseif H < 180
        (0.0, C, X)
    elseif H < 240
        (0.0, X, C)
    elseif H < 300
        (X, 0.0, C)
    else
        (C, 0.0, X)
    end

    (clamp(r + m, 0, 1), clamp(g + m, 0, 1), clamp(b + m, 0, 1))
end

"""
    rgb_to_hex(R, G, B) -> String

Convert RGB (0-1) to hex string.
"""
function rgb_to_hex(R::Float64, G::Float64, B::Float64)
    r = round(UInt8, clamp(R, 0, 1) * 255)
    g = round(UInt8, clamp(G, 0, 1) * 255)
    b = round(UInt8, clamp(B, 0, 1) * 255)
    "#" * string(r, base=16, pad=2) * string(g, base=16, pad=2) * string(b, base=16, pad=2)
end

# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

function world_dark_forest(seed::UInt64=UInt64(0x6761795f636f6c6f))
    println("╔═══════════════════════════════════════════════════════════════════════╗")
    println("║  DARK FOREST RGB CIRCLES: Learned Subobject Classifier χ: HSL → Ω₃   ║")
    println("║                                                                       ║")
    println("║  🟢 DUCK (Green, H=120°): Hide in foliage, 1x reward                  ║")
    println("║  🔴 WORM (Red, H=0°):     Burrow and ambush, 3x reward                ║")
    println("║  🔵 APE (Blue, H=240°):   Dominate by force, 9x reward                ║")
    println("╚═══════════════════════════════════════════════════════════════════════╝")
    println()

    # Create classifier
    classifier = DarkForestClassifier()
    state = classifier.state

    # Generate test colors (HSL) deterministically from seed
    rng_state = seed
    colors = Tuple{Float64, Float64, Float64}[]
    for i in 1:60
        rng_state = (rng_state + 0x9E3779B97F4A7C15) % UInt64
        z = rng_state
        z = ((z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9) % UInt64
        z = ((z ⊻ (z >> 27)) * 0x94D049BB133111EB) % UInt64
        z = z ⊻ (z >> 31)

        H = Float64((z >> 56) % 256) * 360.0 / 256.0  # 0-360
        S = Float64((z >> 48) % 256) / 255.0          # 0-1
        L = 0.25 + Float64((z >> 40) % 256) / 512.0   # 0.25-0.75

        push!(colors, (H, S, L))
    end

    path_depth = 3
    depth_bonus = 1.0 + 0.1 * path_depth

    println("═══ RGB CIRCLE CLASSIFICATION ═══")
    println()

    # Show distribution by hue
    duck_count = count(c -> classify_by_rgb_circle(c[1], c[2], c[3]) == Duck, colors)
    worm_count = count(c -> classify_by_rgb_circle(c[1], c[2], c[3]) == Worm, colors)
    ape_count = count(c -> classify_by_rgb_circle(c[1], c[2], c[3]) == Ape, colors)
    println("  Hard classification by RGB circle distance:")
    println("    🟢 Duck (Green): $duck_count")
    println("    🔴 Worm (Red):   $worm_count")
    println("    🔵 Ape (Blue):   $ape_count")
    println()

    # Show sample colors
    println("  Sample colors with dark forest analysis:")
    for i in 1:5
        H, S, L = colors[i]
        species = classify_by_rgb_circle(H, S, L)
        hiding = hiding_score(H, S, L, species)
        detection = detection_risk(H, S, L)
        R, G, B = hsl_to_rgb(H, S, L)
        hex = rgb_to_hex(R, G, B)

        emoji = species == Duck ? "🟢" : species == Worm ? "🔴" : "🔵"
        println("    HSL($(round(H, digits=0))°, $(round(S*100))%, $(round(L*100))%) → $emoji $species")
        println("      RGB: $hex  hiding: $(round(hiding, digits=2))  detection: $(round(detection, digits=2))")
    end

    # Initial metrics
    println()
    println("═══ DARK FOREST INITIAL STATE ═══")
    initial_loss = dark_forest_loss(colors, state, depth_bonus)
    println("  Loss (negative reward): $(round(initial_loss, digits=4))")

    # Calculate survival and predation stats
    total_hiding = 0.0
    total_predation = 0.0
    for (H, S, L) in colors
        species = classify_by_rgb_circle(H, S, L)
        total_hiding += hiding_score(H, S, L, species)
        total_predation += detection_risk(H, S, L)
    end
    println("  Avg hiding score: $(round(total_hiding / length(colors), digits=3))")
    println("  Avg detection risk: $(round(total_predation / length(colors), digits=3))")

    # Train
    println()
    println("═══ LEARNING DARK FOREST SURVIVAL ═══")
    learn_dark_forest!(state, colors, path_depth; lr=0.02, epochs=100, verbose=true)

    # Learned state
    println()
    println("═══ LEARNED PARAMETERS ═══")
    println("  Camouflage strengths:")
    println("    🟢 Duck: $(round(state.duck_camouflage, digits=3))")
    println("    🔴 Worm: $(round(state.worm_camouflage, digits=3))")
    println("    🔵 Ape:  $(round(state.ape_camouflage, digits=3))")
    println("  Exposure tolerance: $(round(state.exposure_tolerance, digits=3))")
    println("  Predation avoidance: $(round(state.predation_weight, digits=3))")
    println("  Circle hue shifts:")
    println("    🟢 Duck: $(round(state.duck_hue_shift, digits=1))°")
    println("    🔴 Worm: $(round(state.worm_hue_shift, digits=1))°")
    println("    🔵 Ape:  $(round(state.ape_hue_shift, digits=1))°")

    # Final classification with soft probabilities
    println()
    println("═══ SOFT CLASSIFICATION AFTER LEARNING ═══")
    for i in 1:3
        H, S, L = colors[i]
        species, probs = classify_dark_forest(H, S, L, classifier)
        emoji = species == Duck ? "🟢" : species == Worm ? "🔴" : "🔵"

        println("  HSL($(round(H, digits=0))°, $(round(S*100))%, $(round(L*100))%) → $emoji $species")
        println("    P(Duck)=$(round(probs[1], digits=2)) P(Worm)=$(round(probs[2], digits=2)) P(Ape)=$(round(probs[3], digits=2))")
    end

    # Improvement
    println()
    println("═══ IMPROVEMENT ═══")
    final_loss = state.loss_history[end]
    improvement = (initial_loss - final_loss) / abs(initial_loss) * 100
    println("  Initial loss: $(round(initial_loss, digits=4))")
    println("  Final loss:   $(round(final_loss, digits=4))")
    println("  Improvement:  $(round(improvement, digits=1))%")

    # Camouflage demo
    println()
    println("═══ CAMOUFLAGE STRATEGY ═══")
    for i in 1:3
        H, S, L = colors[i]
        H_camo, S_camo, L_camo, species = optimal_camouflage(H, S, L, classifier)

        hex_orig = rgb_to_hex(hsl_to_rgb(H, S, L)...)
        hex_camo = rgb_to_hex(hsl_to_rgb(H_camo, S_camo, L_camo)...)

        emoji = species == Duck ? "🟢" : species == Worm ? "🔴" : "🔵"
        println("  $emoji Original: $hex_orig → Camouflaged: $hex_camo")
        println("     Hue: $(round(H, digits=0))° → $(round(H_camo, digits=0))°")
    end

    println()
    println("╔═══════════════════════════════════════════════════════════════════════╗")
    println("║  Dark Forest classifier learned! Survival through RGB circle hiding.  ║")
    println("╚═══════════════════════════════════════════════════════════════════════╝")

    (classifier, colors)
end

end # module DarkForestRGBCircles
