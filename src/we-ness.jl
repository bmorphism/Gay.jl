# We-ness (we-ness.jl) - Geometric Neurophenomenology of Joint Action
#
# Based on Nicolás Hinrichs' "Geometric Neurophenomenology" (BAMΞ 2025)
# Maps interbrain network dynamics to colors via discrete Ricci curvature
#
# Key concepts:
# - We-ness: phenomenological sense of acting WITH someone, not alongside
# - Rupture/Repair/Reatunement cycles as phase transitions
# - Curvature entropy as order parameter for coordination regimes
# - Shared generative manifold between coupled agents

using Colors
using Random

export WeState, RuptureRepairCycle, CurvatureEntropy
export weness_color, rupture_color, repair_color, reatunement_color
export dyad_palette, collective_palette
export simulate_joint_action, phase_transition_colors

# ═══════════════════════════════════════════════════════════════════════════
# We-ness States (from geometric neurophenomenology)
# ═══════════════════════════════════════════════════════════════════════════

"""
We-ness coordination states mapped to color regimes.

From Hinrichs (2025):
- High curvature entropy → heterogeneous topology → rupture (transition)
- Low curvature entropy → homogeneous topology → stable coordination
"""
@enum WeState begin
    ISOLATED       # No coupling, separate agents
    COREGULATION   # Stable joint action, low entropy
    RUPTURE        # Misatunement spike, high entropy
    REPAIR         # Exploring new configurations
    REATUNEMENT    # New stable regime achieved
    WEMODE         # Full collective/group flow
end

"""
    RuptureRepairCycle

Represents one cycle of rupture → repair → reatunement in joint action.
Each cycle has a unique fingerprint that maps to colors.
"""
struct RuptureRepairCycle
    seed::UInt64
    rupture_magnitude::Float64    # Self-model prediction error
    repair_duration::Float64      # Time to repair (normalized)
    reatunement_quality::Float64  # How well coordination recovered
    valence::Float64              # Affective tone (-1 to 1)
end

"""
    CurvatureEntropy

Discrete Ricci curvature entropy over interbrain network edges.
Acts as order parameter for coordination phase transitions.
"""
struct CurvatureEntropy
    value::Float64           # Shannon entropy of curvature distribution
    edge_count::Int          # Number of edges in network
    positive_fraction::Float64  # Fraction of edges with positive curvature
    bridge_count::Int        # Edges with negative curvature (bottlenecks)
end

# ═══════════════════════════════════════════════════════════════════════════
# Color Mappings for We-ness States
# ═══════════════════════════════════════════════════════════════════════════

"""
    weness_color(state::WeState; intensity=1.0)

Map we-ness state to RGB color.

Color scheme inspired by the rupture/repair/reatunement diagram:
- Rupture: Orange/Red (high entropy, fragmented)
- Repair: Purple/Pink (transitional, exploratory)
- Reatunement: Blue (stable, integrated)
- Co-regulation: Cyan (flowing coordination)
- We-mode: White/gold (full collective coherence)
"""
function weness_color(state::WeState; intensity::Float64=1.0)
    base = if state == ISOLATED
        RGB(0.3, 0.3, 0.3)       # Gray - disconnected
    elseif state == COREGULATION
        RGB(0.4, 0.8, 0.9)       # Cyan - stable flow
    elseif state == RUPTURE
        RGB(0.95, 0.4, 0.2)      # Orange-red - disruption
    elseif state == REPAIR
        RGB(0.7, 0.3, 0.7)       # Purple - exploration
    elseif state == REATUNEMENT
        RGB(0.2, 0.5, 0.9)       # Blue - reintegration
    elseif state == WEMODE
        RGB(1.0, 0.85, 0.4)      # Gold - collective coherence
    else
        RGB(0.5, 0.5, 0.5)
    end
    
    # Apply intensity scaling
    RGB(base.r * intensity, base.g * intensity, base.b * intensity)
end

# Convenience functions
rupture_color(intensity::Float64=1.0) = weness_color(RUPTURE; intensity)
repair_color(intensity::Float64=1.0) = weness_color(REPAIR; intensity)
reatunement_color(intensity::Float64=1.0) = weness_color(REATUNEMENT; intensity)

# ═══════════════════════════════════════════════════════════════════════════
# Curvature-Based Color Generation
# ═══════════════════════════════════════════════════════════════════════════

"""
    curvature_to_color(κ::Float64)

Map discrete Ricci curvature value to color.

- Positive curvature (clustered, redundant): warm colors
- Zero curvature (flat): neutral
- Negative curvature (bridge, bottleneck): cool colors
"""
function curvature_to_color(κ::Float64)
    κ_clamped = clamp(κ, -2.0, 2.0)
    
    if κ_clamped > 0
        # Positive: yellow → red (clustered edges)
        t = κ_clamped / 2.0
        RGB(1.0, 1.0 - 0.6t, 0.2 - 0.2t)
    elseif κ_clamped < 0
        # Negative: cyan → blue (bridge edges)
        t = -κ_clamped / 2.0
        RGB(0.2 - 0.2t, 0.8 - 0.3t, 1.0)
    else
        # Zero: neutral green
        RGB(0.5, 0.8, 0.5)
    end
end

"""
    entropy_to_color(H::Float64; H_max=3.0)

Map curvature entropy to color.

- Low entropy: blue (stable, homogeneous)
- High entropy: red (transitional, heterogeneous)
"""
function entropy_to_color(H::Float64; H_max::Float64=3.0)
    t = clamp(H / H_max, 0.0, 1.0)
    
    # Blue → Purple → Red gradient
    if t < 0.5
        s = t * 2
        RGB(0.2 + 0.5s, 0.3 + 0.2s, 0.9 - 0.2s)
    else
        s = (t - 0.5) * 2
        RGB(0.7 + 0.25s, 0.5 - 0.3s, 0.7 - 0.5s)
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# Dyad and Collective Palettes
# ═══════════════════════════════════════════════════════════════════════════

"""
    dyad_palette(seed::Integer=42)

Generate a 6-color palette for a dyad (two-agent system).
Colors represent the shared generative manifold between agents.
"""
function dyad_palette(seed::Integer=42)
    gay_seed!(seed)
    
    [
        weness_color(ISOLATED),
        weness_color(COREGULATION),
        weness_color(RUPTURE),
        weness_color(REPAIR),
        weness_color(REATUNEMENT),
        weness_color(WEMODE),
    ]
end

"""
    collective_palette(n_agents::Int; seed::Integer=42)

Generate colors for a multi-agent collective.
More agents → richer color space exploration.
"""
function collective_palette(n_agents::Int; seed::Integer=42)
    gay_seed!(seed)
    
    colors = RGB[]
    
    for i in 1:n_agents
        # Each agent gets a unique hue based on their position
        hue = (i - 1) / n_agents * 360.0
        push!(colors, HSL(hue, 0.7, 0.5) |> RGB)
    end
    
    # Add interaction colors (pairwise blends)
    for i in 1:min(n_agents, 4)
        for j in (i+1):min(n_agents, 4)
            blend = RGB(
                (colors[i].r + colors[j].r) / 2,
                (colors[i].g + colors[j].g) / 2,
                (colors[i].b + colors[j].b) / 2
            )
            push!(colors, blend)
        end
    end
    
    colors
end

# ═══════════════════════════════════════════════════════════════════════════
# Simulate Joint Action Dynamics
# ═══════════════════════════════════════════════════════════════════════════

"""
    simulate_joint_action(n_steps::Int=69; seed::Integer=42)

Simulate a joint action trajectory with rupture/repair cycles.
Returns sequence of (state, curvature_entropy, color) tuples.
"""
function simulate_joint_action(n_steps::Int=69; seed::Integer=42)
    rng = Random.MersenneTwister(seed)
    
    trajectory = Tuple{WeState, Float64, RGB}[]
    
    state = COREGULATION
    H = 0.5  # Initial entropy
    
    for t in 1:n_steps
        # Transition probabilities based on current state
        if state == COREGULATION
            if rand(rng) < 0.1
                state = RUPTURE
                H += 0.5 + rand(rng) * 0.5
            end
        elseif state == RUPTURE
            H = max(H - 0.1, 0.0)
            if rand(rng) < 0.3
                state = REPAIR
            end
        elseif state == REPAIR
            H += (rand(rng) - 0.5) * 0.3  # Fluctuating
            if rand(rng) < 0.2
                state = REATUNEMENT
                H = max(H - 0.3, 0.2)
            elseif rand(rng) < 0.1
                state = RUPTURE  # Failed repair
                H += 0.4
            end
        elseif state == REATUNEMENT
            H = max(H - 0.1, 0.3)
            if rand(rng) < 0.4
                state = COREGULATION
                H = 0.3 + rand(rng) * 0.2
            end
        end
        
        H = clamp(H, 0.0, 3.0)
        color = entropy_to_color(H)
        
        push!(trajectory, (state, H, color))
    end
    
    trajectory
end

"""
    phase_transition_colors(; n_points=100, seed=42)

Generate colors along a phase transition from order → disorder → order.
Models the curvature entropy trajectory during rupture/repair.
"""
function phase_transition_colors(; n_points::Int=100, seed::Integer=42)
    colors = RGB[]
    
    for i in 1:n_points
        t = (i - 1) / (n_points - 1)
        
        # Phase transition curve: low → high → low entropy
        # Models: co-regulation → rupture → repair → reatunement
        if t < 0.3
            H = 0.3 + 0.5 * (t / 0.3)  # Rising toward rupture
        elseif t < 0.5
            H = 0.8 + 1.5 * ((t - 0.3) / 0.2)  # Peak at rupture
        elseif t < 0.7
            H = 2.3 - 1.0 * ((t - 0.5) / 0.2)  # Repair fluctuation
        else
            H = 1.3 - 1.0 * ((t - 0.7) / 0.3)  # Settling to reatunement
        end
        
        push!(colors, entropy_to_color(H))
    end
    
    colors
end

# ═══════════════════════════════════════════════════════════════════════════
# Valence-Based Colors (Recursive Inference over Self-Model)
# ═══════════════════════════════════════════════════════════════════════════

"""
    valence_color(v::Float64)

Map affective valence to color.

From Hinrichs: valence = inference about self-model coherence
- Negative: prediction error, frustration (dark reds)
- Neutral: baseline (gray)
- Positive: coherence, flow (bright greens/golds)
"""
function valence_color(v::Float64)
    v_clamped = clamp(v, -1.0, 1.0)
    
    if v_clamped < 0
        # Negative valence: darker, redder
        t = -v_clamped
        RGB(0.4 + 0.4t, 0.3 - 0.2t, 0.3 - 0.2t)
    elseif v_clamped > 0
        # Positive valence: brighter, golden
        t = v_clamped
        RGB(0.4 + 0.5t, 0.5 + 0.4t, 0.3 - 0.1t)
    else
        RGB(0.5, 0.5, 0.5)
    end
end

"""
    self_model_prediction_error_color(error::Float64; relevance::Float64=1.0)

Color based on self-model prediction error magnitude and relevance.
High error + high relevance = intense disruption colors.
"""
function self_model_prediction_error_color(error::Float64; relevance::Float64=1.0)
    intensity = clamp(error * relevance, 0.0, 2.0)
    
    if intensity < 0.5
        # Low error: calm blues
        RGB(0.3, 0.5, 0.7 + 0.3 * (1 - intensity * 2))
    elseif intensity < 1.0
        # Medium error: purples
        t = (intensity - 0.5) * 2
        RGB(0.5 + 0.3t, 0.3, 0.7 - 0.2t)
    else
        # High error: alarming oranges/reds
        t = min((intensity - 1.0), 1.0)
        RGB(0.8 + 0.15t, 0.3 - 0.1t, 0.2 - 0.1t)
    end
end
