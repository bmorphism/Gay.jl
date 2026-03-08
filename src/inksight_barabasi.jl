#=
InkSight-Barabási Integration: Network Criticality in Digital Ink

This module reimplements Google Research's InkSight data structures using
Julia's most avant-garde primitives, while grounding criticality in
Barabási's network science framework.

## Barabási's Answer: Where Does Criticality Come From?

In network science (Barabási-Albert model, scale-free networks), criticality
emerges from:

1. **Preferential Attachment**: New connections prefer high-degree nodes
   → Power-law degree distribution P(k) ~ k^(-γ)
   
2. **Hubs as Critical Points**: High-degree hubs are the "dragon kings"
   When a hub fails, the network undergoes phase transition
   
3. **Percolation Threshold**: At critical p_c, giant component emerges/dissolves
   This IS the pen-lift moment in handwriting!

4. **Cascading Failures**: Local perturbation → global reorganization
   Correlation length ξ → ∞ at criticality

In handwriting terms:
- Each stroke point is a NODE
- Temporal sequence creates EDGES  
- Pen-lift = PERCOLATION THRESHOLD crossing
- Hub points (corners, crossings) = CRITICAL NODES
- Network diameter collapse at lift = DIMENSION REDUCTION

The GF(3) connection:
- γ = 3 is the critical exponent for scale-free networks!
- P(k) ~ k^(-3) at criticality
- This is why GF(3) naturally captures phase transitions

References:
- Barabási & Albert (1999) "Emergence of scaling in random networks"
- Dorogovtsev & Mendes (2002) "Evolution of networks"
- Berent et al. (2024) "InkSight: Offline-to-Online Handwriting Conversion"
=#

module InkSightBarabasi

export InkPoint, InkStroke, InkNetwork, TracedInkNetwork
export add_point!, lift_pen!, network_criticality, hub_detection
export to_tokens, from_tokens, dual_representation
export percolation_state, correlation_length, effective_dimension
# GF(3) trit types
export Trit, Minus, Ergodic, Plus, MINUS, ERGODIC, PLUS, trit_value
# Chromatic visualization
export InkSightColorScheme, rainbow_scheme, gay_scheme
export stroke_color, point_alpha, hue_to_trit, chromatic_encoding
export gay_stroke_color, ChromaticEncoding, ColoredStroke, ColoredPoint

using Random

#==============================================================================#
#                     AVANT-GARDE JULIA PRIMITIVES                             #
#==============================================================================#

# 1. Parametric singleton types for GF(3) with zero runtime cost
abstract type Trit end
struct Minus <: Trit end   # -1: Verification, contraction
struct Ergodic <: Trit end #  0: Balance, transition  
struct Plus <: Trit end    # +1: Generation, expansion

const MINUS = Minus()
const ERGODIC = Ergodic()
const PLUS = Plus()

# Type-level arithmetic via multiple dispatch
trit_value(::Minus) = -1
trit_value(::Ergodic) = 0
trit_value(::Plus) = 1

# 2. Sum types via Union (Julia's discriminated unions)
const PenState = Union{Val{:down}, Val{:lifting}, Val{:up}}
const DOWN = Val(:down)
const LIFTING = Val(:lifting)
const UP = Val(:up)

# 3. Immutable value types with automatic memory layout optimization
struct InkPoint
    x::Float32
    y::Float32
    t::Float32        # Timestamp (normalized 0-1)
    p::Float32        # Pressure (0-1)
    degree::Int32     # Network degree (Barabási)
    hub_score::Float32 # Centrality measure
end

# Zero-cost constructor with defaults
InkPoint(x, y; t=0f0, p=0.5f0) = InkPoint(Float32(x), Float32(y), Float32(t), Float32(p), Int32(1), 0f0)

# 4. Generator-based lazy stroke iteration (memory efficient)
struct InkStroke
    points::Vector{InkPoint}
    trit::Trit
    fingerprint::UInt64
    
    # Inner constructor with automatic fingerprint
    function InkStroke(points::Vector{InkPoint}, trit::Trit=ERGODIC)
        fp = hash(points)
        new(points, trit, fp)
    end
end

# Lazy point iteration via Julia generators
Base.iterate(s::InkStroke) = iterate(s.points)
Base.iterate(s::InkStroke, state) = iterate(s.points, state)
Base.length(s::InkStroke) = length(s.points)
Base.eltype(::Type{InkStroke}) = InkPoint

# 5. Mutable network structure with Barabási dynamics
mutable struct InkNetwork
    # Core data
    strokes::Vector{InkStroke}
    adjacency::Dict{Int, Set{Int}}  # Sparse adjacency for scale-free
    
    # Barabási network metrics
    degree_sequence::Vector{Int}
    hub_indices::Vector{Int}        # High-degree nodes
    gamma::Float64                  # Power-law exponent (critical at γ=3)
    
    # Criticality state
    giant_component_size::Int
    percolation_p::Float64          # Current percolation probability
    correlation_length::Float64     # ξ: diverges at criticality
    
    # Pen state machine
    pen_state::PenState
    current_stroke::Int
    
    function InkNetwork()
        new(
            InkStroke[],
            Dict{Int, Set{Int}}(),
            Int[],
            Int[],
            3.0,  # Critical exponent
            0,
            1.0,
            1.0,
            DOWN,
            0
        )
    end
end

#==============================================================================#
#                    BARABÁSI NETWORK DYNAMICS                                 #
#==============================================================================#

"""
Preferential attachment: probability of connecting to node i is proportional to k_i
This is the SOURCE of criticality in Barabási's framework.
"""
function preferential_attachment_prob(net::InkNetwork, node_idx::Int)::Float64
    if isempty(net.degree_sequence)
        return 1.0
    end
    k_i = get(net.degree_sequence, node_idx, 1)
    total_degree = sum(net.degree_sequence)
    return k_i / max(total_degree, 1)
end

"""
Add a point with Barabási preferential attachment dynamics.
New points connect preferentially to high-degree "hub" points.
"""
function add_point!(net::InkNetwork, x::Real, y::Real; 
                    t::Real=0.0, p::Real=0.5, m::Int=2)
    
    # Create new point
    point_idx = sum(length(s) for s in net.strokes; init=0) + 1
    
    # Preferential attachment: connect to m existing high-degree nodes
    if point_idx > 1 && !isempty(net.degree_sequence)
        # Sample m nodes with probability ∝ degree
        probs = [preferential_attachment_prob(net, i) for i in 1:length(net.degree_sequence)]
        probs ./= sum(probs)
        
        # Connect to previous point (temporal edge) + preferential edges
        net.adjacency[point_idx] = Set{Int}()
        
        # Always connect to immediate predecessor (temporal flow)
        push!(net.adjacency[point_idx], point_idx - 1)
        if haskey(net.adjacency, point_idx - 1)
            push!(net.adjacency[point_idx - 1], point_idx)
        else
            net.adjacency[point_idx - 1] = Set([point_idx])
        end
        
        # Preferential attachment to hubs
        for _ in 1:min(m-1, length(probs))
            target = sample_weighted(1:length(probs), probs)
            if target != point_idx && target != point_idx - 1
                push!(net.adjacency[point_idx], target)
                if haskey(net.adjacency, target)
                    push!(net.adjacency[target], point_idx)
                else
                    net.adjacency[target] = Set([point_idx])
                end
            end
        end
    else
        net.adjacency[point_idx] = Set{Int}()
    end
    
    # Update degree sequence
    degree = length(get(net.adjacency, point_idx, Set{Int}()))
    push!(net.degree_sequence, degree)
    
    # Compute hub score (PageRank-like centrality)
    hub_score = Float32(degree / max(1, maximum(net.degree_sequence; init=1)))
    
    # Create point with network metadata
    new_point = InkPoint(Float32(x), Float32(y), Float32(t), Float32(p), 
                         Int32(degree), hub_score)
    
    # Add to current stroke or create new one
    if isempty(net.strokes) || net.pen_state === UP
        push!(net.strokes, InkStroke([new_point]))
        net.current_stroke = length(net.strokes)
        net.pen_state = DOWN
    else
        # Rebuild stroke with new point (immutable strokes)
        old_stroke = net.strokes[net.current_stroke]
        new_points = vcat(old_stroke.points, [new_point])
        net.strokes[net.current_stroke] = InkStroke(new_points, old_stroke.trit)
    end
    
    # Update network criticality metrics
    update_criticality!(net)
    
    return new_point
end

# Simple weighted sampling without external deps
function sample_weighted(items, weights)
    r = rand() * sum(weights)
    cumsum = 0.0
    for (item, w) in zip(items, weights)
        cumsum += w
        if r <= cumsum
            return item
        end
    end
    return last(items)
end

"""
Pen lift = Network percolation threshold crossing.

In Barabási's framework, this is when the giant component fragments.
The correlation length diverges: ξ → ∞
"""
function lift_pen!(net::InkNetwork)
    if net.pen_state === DOWN
        net.pen_state = LIFTING
        
        # At lift, correlation length spikes (criticality)
        net.correlation_length = compute_correlation_length(net)
        
        # Check for hub removal (dragon king event)
        if !isempty(net.hub_indices)
            # Pen lift near a hub = phase transition
            current_point_idx = sum(length(s) for s in net.strokes)
            if current_point_idx in net.hub_indices
                # Dragon king: this lift fragments the network
                net.gamma = 2.5  # Sub-critical
            end
        end
        
        net.pen_state = UP
        net.current_stroke = 0
        
        return true
    end
    return false
end

"""
Update network criticality metrics after each point addition.
"""
function update_criticality!(net::InkNetwork)
    if length(net.degree_sequence) < 3
        return
    end
    
    # 1. Estimate power-law exponent γ via maximum likelihood
    net.gamma = estimate_power_law_exponent(net.degree_sequence)
    
    # 2. Identify hubs (nodes with degree > mean + 2σ)
    mean_k = sum(net.degree_sequence) / length(net.degree_sequence)
    std_k = sqrt(sum((k - mean_k)^2 for k in net.degree_sequence) / length(net.degree_sequence))
    threshold = mean_k + 2 * std_k
    net.hub_indices = findall(k -> k > threshold, net.degree_sequence)
    
    # 3. Compute giant component size (BFS)
    net.giant_component_size = compute_giant_component(net)
    
    # 4. Percolation probability
    net.percolation_p = net.giant_component_size / max(1, length(net.degree_sequence))
    
    # 5. Correlation length (diverges at p_c)
    net.correlation_length = compute_correlation_length(net)
end

"""
Maximum likelihood estimator for power-law exponent.
γ = 3 is the critical value for scale-free networks.
"""
function estimate_power_law_exponent(degrees::Vector{Int})::Float64
    if isempty(degrees) || all(d -> d <= 0, degrees)
        return 3.0  # Default critical
    end
    
    # Filter positive degrees
    k = filter(d -> d > 0, degrees)
    if isempty(k)
        return 3.0
    end
    
    k_min = minimum(k)
    n = length(k)
    
    # MLE: γ = 1 + n / Σ ln(k_i / k_min)
    log_sum = sum(log(ki / k_min) for ki in k if ki > k_min; init=0.0)
    
    if log_sum > 0
        return 1.0 + n / log_sum
    else
        return 3.0
    end
end

"""
BFS to find giant component size.
"""
function compute_giant_component(net::InkNetwork)::Int
    n = length(net.degree_sequence)
    if n == 0
        return 0
    end
    
    visited = falses(n)
    max_component = 0
    
    for start in 1:n
        if visited[start]
            continue
        end
        
        # BFS from start
        component_size = 0
        queue = [start]
        
        while !isempty(queue)
            node = popfirst!(queue)
            if visited[node]
                continue
            end
            visited[node] = true
            component_size += 1
            
            # Add neighbors
            for neighbor in get(net.adjacency, node, Set{Int}())
                if !visited[neighbor] && 1 <= neighbor <= n
                    push!(queue, neighbor)
                end
            end
        end
        
        max_component = max(max_component, component_size)
    end
    
    return max_component
end

"""
Correlation length ξ: measures how far correlations extend.
Diverges at criticality: ξ ~ |p - p_c|^(-ν)
"""
function compute_correlation_length(net::InkNetwork)::Float64
    p = net.percolation_p
    p_c = 0.5  # Percolation threshold (approximation)
    ν = 4/3   # Critical exponent for 2D percolation
    
    distance_to_critical = abs(p - p_c)
    
    if distance_to_critical < 0.01
        # Near criticality: ξ → ∞
        return 1000.0
    else
        return distance_to_critical^(-ν)
    end
end

#==============================================================================#
#                    BERENT TOKENIZATION (InkSight)                            #
#==============================================================================#

"""
Convert ink to tokens following Berent et al.'s approach.
Uses discretized Δx, Δy with special PEN_LIFT token.
"""
function to_tokens(net::InkNetwork; 
                   vocab_size::Int=1024,
                   special_tokens::Bool=true)::Vector{String}
    
    tokens = String[]
    
    if special_tokens
        push!(tokens, "<INK_START>")
    end
    
    for (i, stroke) in enumerate(net.strokes)
        if i > 1 && special_tokens
            # PEN_LIFT between strokes = phase transition marker
            push!(tokens, "<PEN_LIFT>")
        end
        
        prev_x, prev_y = 0f0, 0f0
        for (j, point) in enumerate(stroke)
            if j == 1
                # First point: absolute coordinates
                qx = quantize(point.x, vocab_size)
                qy = quantize(point.y, vocab_size)
                push!(tokens, "X$qx")
                push!(tokens, "Y$qy")
            else
                # Subsequent: delta encoding
                Δx = point.x - prev_x
                Δy = point.y - prev_y
                qΔx = quantize_delta(Δx, vocab_size)
                qΔy = quantize_delta(Δy, vocab_size)
                push!(tokens, "DX$qΔx")
                push!(tokens, "DY$qΔy")
            end
            
            # Hub markers for critical points
            if point.hub_score > 0.8
                push!(tokens, "<HUB>")
            end
            
            prev_x, prev_y = point.x, point.y
        end
    end
    
    if special_tokens
        push!(tokens, "<INK_END>")
    end
    
    return tokens
end

quantize(v::Float32, vocab::Int) = clamp(round(Int, v * (vocab - 1)), 0, vocab - 1)
quantize_delta(Δ::Float32, vocab::Int) = clamp(round(Int, (Δ + 1) * (vocab - 1) / 2), 0, vocab - 1)

"""
Parse tokens back to InkNetwork.
"""
function from_tokens(tokens::Vector{String}; vocab_size::Int=1024)::InkNetwork
    net = InkNetwork()
    
    x, y = 0f0, 0f0
    in_stroke = false
    
    for token in tokens
        if token == "<INK_START>"
            continue
        elseif token == "<INK_END>"
            break
        elseif token == "<PEN_LIFT>"
            lift_pen!(net)
            in_stroke = false
        elseif token == "<HUB>"
            # Hub marker (metadata only)
            continue
        elseif startswith(token, "X")
            x = parse(Int, token[2:end]) / (vocab_size - 1)
        elseif startswith(token, "Y")
            y = parse(Int, token[2:end]) / (vocab_size - 1)
            add_point!(net, x, y)
            in_stroke = true
        elseif startswith(token, "DX")
            Δx = (parse(Int, token[3:end]) / (vocab_size - 1)) * 2 - 1
            x += Δx
        elseif startswith(token, "DY")
            Δy = (parse(Int, token[3:end]) / (vocab_size - 1)) * 2 - 1
            y += Δy
            add_point!(net, x, y)
            in_stroke = true
        end
    end
    
    return net
end

#==============================================================================#
#                    DUAL REPRESENTATION (Image + Text)                        #
#==============================================================================#

"""
Berent's key insight: VLMs work better with BOTH representations.
- Image: RGB where R=time, G=distance, B=pressure
- Text: Tokenized coordinates

This function generates the dual representation.
"""
function dual_representation(net::InkNetwork; 
                            width::Int=224, 
                            height::Int=224)
    
    # Text representation
    text_tokens = to_tokens(net)
    
    # Image representation (RGB encoding)
    # R = normalized time
    # G = distance from stroke start
    # B = pressure (or hub_score as proxy for "importance")
    
    image = zeros(Float32, height, width, 3)
    
    total_points = sum(length(s) for s in net.strokes; init=0)
    point_idx = 0
    
    for stroke in net.strokes
        stroke_start_x = isempty(stroke.points) ? 0f0 : stroke.points[1].x
        stroke_start_y = isempty(stroke.points) ? 0f0 : stroke.points[1].y
        
        for point in stroke
            point_idx += 1
            
            # Map to pixel coordinates
            px = clamp(round(Int, point.x * (width - 1)) + 1, 1, width)
            py = clamp(round(Int, point.y * (height - 1)) + 1, 1, height)
            
            # RGB encoding
            r = point_idx / max(total_points, 1)  # Time
            g = sqrt((point.x - stroke_start_x)^2 + (point.y - stroke_start_y)^2)  # Distance
            b = point.hub_score  # Importance/pressure
            
            # Anti-aliased rendering (simple box filter)
            for dy in -1:1, dx in -1:1
                nx, ny = px + dx, py + dy
                if 1 <= nx <= width && 1 <= ny <= height
                    weight = 1f0 / (1 + abs(dx) + abs(dy))
                    image[ny, nx, 1] = max(image[ny, nx, 1], r * weight)
                    image[ny, nx, 2] = max(image[ny, nx, 2], g * weight)
                    image[ny, nx, 3] = max(image[ny, nx, 3], b * weight)
                end
            end
        end
    end
    
    return (text=text_tokens, image=image)
end

#==============================================================================#
#                    CRITICALITY QUERIES                                       #
#==============================================================================#

"""
Network criticality state based on Barabási metrics.
"""
function network_criticality(net::InkNetwork)::NamedTuple
    # γ = 3 is critical; γ < 3 = supercritical; γ > 3 = subcritical
    criticality = if net.gamma < 2.5
        :supercritical  # Dominated by hubs
    elseif net.gamma < 3.5
        :critical       # Scale-free, power-law
    else
        :subcritical    # Random-like
    end
    
    # Effective dimension from GF(3) perspective
    eff_dim = if net.correlation_length > 100
        1  # Critical: everything correlated
    elseif net.correlation_length > 10
        2  # Near critical
    else
        3  # Far from critical
    end
    
    return (
        gamma = net.gamma,
        criticality = criticality,
        correlation_length = net.correlation_length,
        giant_component_fraction = net.percolation_p,
        num_hubs = length(net.hub_indices),
        effective_dimension = eff_dim,
        is_at_phase_transition = 2.8 < net.gamma < 3.2
    )
end

"""
Detect hub points (Barabási's "dragon kings").
These are where pen-lifts cause phase transitions.
"""
function hub_detection(net::InkNetwork)::Vector{NamedTuple}
    hubs = NamedTuple[]
    
    point_idx = 0
    for (stroke_idx, stroke) in enumerate(net.strokes)
        for (local_idx, point) in enumerate(stroke)
            point_idx += 1
            
            if point_idx in net.hub_indices
                push!(hubs, (
                    stroke = stroke_idx,
                    index = local_idx,
                    global_index = point_idx,
                    position = (point.x, point.y),
                    degree = Int(point.degree),
                    hub_score = point.hub_score
                ))
            end
        end
    end
    
    return hubs
end

"""
Percolation state: are we above or below p_c?
"""
function percolation_state(net::InkNetwork)::Symbol
    p_c = 0.5  # Approximate percolation threshold
    
    if net.percolation_p > p_c + 0.1
        return :percolating      # Giant component exists
    elseif net.percolation_p < p_c - 0.1
        return :fragmented       # No giant component
    else
        return :critical         # At threshold
    end
end

"""
Get correlation length (diverges at criticality).
"""
correlation_length(net::InkNetwork) = net.correlation_length

"""
Effective dimension from network perspective.
At criticality, dimension collapses.
"""
function effective_dimension(net::InkNetwork)::Int
    ξ = net.correlation_length
    
    if ξ > 100
        return 1  # Everything correlated → 1D effective
    elseif ξ > 10
        return 2  # Partial correlation → 2D effective
    else
        return 3  # No long-range correlation → full 3D (GF(3))
    end
end

#==============================================================================#
#                    INTEGRATION WITH TRACEDCRITICALITY                        #
#==============================================================================#

"""
Create a TracedInkNetwork that combines InkSight structure with
TracedCriticality's pen-lift phase transition semantics.
"""
mutable struct TracedInkNetwork
    network::InkNetwork
    
    # From TracedCriticality
    d_2::Float64              # Distance to β=2 fixed point (DYNAMIC)
    tau::Float64              # Plasticity timescale
    visited_set::Set{UInt64}  # SARW physical trace
    landauer_cost::Float64    # Accumulated irreversibility
    
    function TracedInkNetwork()
        new(InkNetwork(), 0.5, 1.0, Set{UInt64}(), 0.0)
    end
end

"""
Add point with SARW collision detection.
Collision = must lift pen = irreversibility = Landauer cost.
"""
function add_point!(tin::TracedInkNetwork, x::Real, y::Real; kwargs...)
    # Hash position for SARW
    pos_hash = hash((round(x, digits=3), round(y, digits=3)))
    
    # Check for SARW collision
    if pos_hash in tin.visited_set
        # Collision! Must lift pen (irreversible)
        lift_pen!(tin)
        tin.landauer_cost += kT_ln2()  # Landauer's principle
    end
    
    # Add to visited set (physical trace)
    push!(tin.visited_set, pos_hash)
    
    # Add point to network
    point = add_point!(tin.network, x, y; kwargs...)
    
    # Update d_2 based on network criticality
    crit = network_criticality(tin.network)
    if crit.is_at_phase_transition
        tin.d_2 = 0.01  # At criticality
    else
        tin.d_2 = abs(crit.gamma - 3.0) / 3.0  # Distance from γ=3
    end
    
    return point
end

function lift_pen!(tin::TracedInkNetwork)
    lift_pen!(tin.network)
    tin.landauer_cost += kT_ln2()  # Each lift costs energy
end

# Landauer's constant at room temperature
kT_ln2() = 2.87e-21  # Joules at T=300K

#==============================================================================#
#                    CHROMATIC VISUALIZATION (InkSight Color Analysis)         #
#==============================================================================#

# Export additional chromatic functions
export stroke_color, point_alpha, chromatic_encoding, hue_to_trit
export InkSightColorScheme, rainbow_scheme, gay_scheme

"""
InkSight's color scheme encodes temporal structure at three levels:
1. Inter-stroke: Rainbow hue (which pen-lift epoch)
2. Intra-stroke: Alpha gradient (correlation decay within stroke)
3. Boundary: White outline (phase transition visualization)

This struct captures the scheme parameters.
"""
struct InkSightColorScheme
    # Rainbow parameters
    hue_start::Float64      # Start of hue range (degrees)
    hue_end::Float64        # End of hue range (degrees)
    reverse_order::Bool     # InkSight uses reverse (red=recent, blue=old)
    
    # HSV darkening
    value_factor::Float64   # InkSight uses 0.65
    
    # Alpha gradient
    alpha_start::Float64    # Start alpha (InkSight: 1.0)
    alpha_end::Float64      # End alpha (InkSight: 0.5)
    
    # Path effect
    outline_color::NTuple{3,Float64}  # RGB for outline
    outline_width_factor::Float64     # InkSight: 1.25
end

# InkSight's default scheme
rainbow_scheme() = InkSightColorScheme(
    0.0, 360.0, true,   # Full rainbow, reversed
    0.65,               # Darken to 65%
    1.0, 0.5,           # Alpha gradient 1.0 → 0.5
    (1.0, 1.0, 1.0),    # White outline
    1.25                # Outline 25% wider
)

# Gay.jl deterministic scheme (uses seed-based colors instead of rainbow)
gay_scheme(seed::UInt64=UInt64(1069)) = InkSightColorScheme(
    0.0, 360.0, false,  # Gay.jl uses golden angle, not reversed
    0.70,               # Slightly brighter
    1.0, 0.6,           # Less fade
    (0.2, 0.2, 0.2),    # Dark outline (better for wide-gamut)
    1.15                # Thinner outline
)

"""
Compute the hue for a stroke based on its index and total stroke count.

InkSight formula: hue = (N - 1 - i) / N × 360°
This gives: first stroke → violet, last stroke → red

In our framework:
- Red (0°-60°) = PLUS (+1), supercritical, recent
- Green (60°-180°) = ERGODIC (0), critical, transition
- Blue (180°-300°) = MINUS (-1), subcritical, old
"""
function stroke_color(stroke_idx::Int, total_strokes::Int, scheme::InkSightColorScheme)
    # InkSight's reverse ordering
    idx = scheme.reverse_order ? (total_strokes - stroke_idx) : (stroke_idx - 1)
    
    # Hue in degrees
    hue = scheme.hue_start + (idx / max(total_strokes - 1, 1)) * (scheme.hue_end - scheme.hue_start)
    hue = mod(hue, 360.0)
    
    # Full saturation, darkened value
    saturation = 1.0
    value = scheme.value_factor
    
    # HSV to RGB
    return hsv_to_rgb(hue, saturation, value)
end

"""
Compute the alpha for a point based on its position within the stroke.

InkSight formula: α = 1 - 0.5 × (j / len)

This encodes:
- Start of stroke: α = 1.0 (pen just touched down, high certainty)
- End of stroke: α = 0.5 (approaching lift, correlation decaying)
"""
function point_alpha(point_idx::Int, stroke_length::Int, scheme::InkSightColorScheme)
    t = (point_idx - 1) / max(stroke_length - 1, 1)
    return scheme.alpha_start + t * (scheme.alpha_end - scheme.alpha_start)
end

"""
Map hue angle to GF(3) trit.

The rainbow naturally partitions into three perceptual bands:
- Warm (red-yellow, 0°-120°): PLUS (+1)
- Neutral (green-cyan, 120°-240°): ERGODIC (0)  
- Cool (blue-violet, 240°-360°): MINUS (-1)
"""
function hue_to_trit(hue::Real)::Trit
    h = mod(hue, 360.0)
    if h < 120.0
        return PLUS
    elseif h < 240.0
        return ERGODIC
    else
        return MINUS
    end
end

"""
Full chromatic encoding for an ink network.

Returns a structure containing:
- Per-stroke colors (RGB)
- Per-stroke trits (GF(3) classification)
- Per-point alphas (correlation decay)
- Network-level gamma (criticality exponent)
"""
function chromatic_encoding(net::InkNetwork; scheme::InkSightColorScheme=rainbow_scheme())
    n_strokes = length(net.strokes)
    
    stroke_colors = Vector{NTuple{3,Float64}}(undef, n_strokes)
    stroke_trits = Vector{Trit}(undef, n_strokes)
    point_alphas = Vector{Vector{Float64}}(undef, n_strokes)
    
    for (i, stroke) in enumerate(net.strokes)
        # Stroke-level color
        rgb = stroke_color(i, n_strokes, scheme)
        stroke_colors[i] = rgb
        
        # Map to GF(3) trit via hue
        hue = rgb_to_hue(rgb...)
        stroke_trits[i] = hue_to_trit(hue)
        
        # Point-level alphas
        n_points = length(stroke.points)
        alphas = [point_alpha(j, n_points, scheme) for j in 1:n_points]
        point_alphas[i] = alphas
    end
    
    # Network criticality
    crit = network_criticality(net)
    
    return (
        stroke_colors = stroke_colors,
        stroke_trits = stroke_trits,
        point_alphas = point_alphas,
        gamma = crit.gamma,
        criticality = crit.criticality,
        gf3_balance = sum(trit_value(t) for t in stroke_trits) % 3
    )
end

#==============================================================================#
#                    COLOR SPACE UTILITIES                                     #
#==============================================================================#

"""
HSV to RGB conversion.
H in degrees [0, 360), S and V in [0, 1].
"""
function hsv_to_rgb(h::Real, s::Real, v::Real)::NTuple{3,Float64}
    h = mod(h, 360.0)
    c = v * s
    x = c * (1 - abs(mod(h / 60, 2) - 1))
    m = v - c
    
    r, g, b = if h < 60
        (c, x, 0.0)
    elseif h < 120
        (x, c, 0.0)
    elseif h < 180
        (0.0, c, x)
    elseif h < 240
        (0.0, x, c)
    elseif h < 300
        (x, 0.0, c)
    else
        (c, 0.0, x)
    end
    
    return (r + m, g + m, b + m)
end

"""
RGB to Hue extraction.
Returns hue in degrees [0, 360).
"""
function rgb_to_hue(r::Real, g::Real, b::Real)::Float64
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    delta = max_c - min_c
    
    if delta < 1e-10
        return 0.0  # Gray, hue undefined
    end
    
    hue = if max_c == r
        60.0 * mod((g - b) / delta, 6)
    elseif max_c == g
        60.0 * ((b - r) / delta + 2)
    else
        60.0 * ((r - g) / delta + 4)
    end
    
    return mod(hue, 360.0)
end

"""
Generate a deterministic color for a stroke using Gay.jl-style splitmix64.

This replaces InkSight's rainbow with reproducible seed-based colors.
Same network fingerprint → same colors across runs.
"""
function gay_stroke_color(stroke_idx::Int, network_fingerprint::UInt64)::NTuple{3,Float64}
    # Combine stroke index with network fingerprint
    seed = network_fingerprint ⊻ UInt64(stroke_idx * 0x9e3779b97f4a7c15)
    
    # SplitMix64 for deterministic randomness
    seed = (seed ⊻ (seed >> 30)) * 0xbf58476d1ce4e5b9
    seed = (seed ⊻ (seed >> 27)) * 0x94d049bb133111eb
    seed = seed ⊻ (seed >> 31)
    
    # Golden angle hue distribution (maximally dispersed)
    hue = mod(stroke_idx * 137.508, 360.0)
    
    # Saturation and value from seed
    sat = 0.6 + 0.4 * ((seed >> 32) % 256) / 255
    val = 0.5 + 0.3 * ((seed >> 40) % 256) / 255
    
    return hsv_to_rgb(hue, sat, val)
end

"""
Compute network fingerprint for deterministic color assignment.
"""
function network_fingerprint(net::InkNetwork)::UInt64
    h = UInt64(0xcbf29ce484222325)  # FNV-1a offset
    
    for stroke in net.strokes
        h ⊻= stroke.fingerprint
        h *= 0x100000001b3  # FNV-1a prime
    end
    
    return h
end

end # module
