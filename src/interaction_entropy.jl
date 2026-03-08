# interaction_entropy.jl
# ═══════════════════════════════════════════════════════════════════════════════
# INTERACTION ENTROPY IN COLOR-SPACE
# Learnable color-spaces of joint interaction as symbolic distillation
# of positionally-dependent noise
# ═══════════════════════════════════════════════════════════════════════════════
#
# CORE INSIGHT:
# In InkSight's chromatic encoding, the highest entropy regions are:
#   1. Stroke boundaries (pen lift moments) - phase transitions
#   2. Hub neighborhoods (dragon kings) - criticality peaks
#   3. Velocity discontinuities - where dv/dt is undefined
#   4. Curvature extrema - where d²r/dt² peaks
#
# These high-entropy regions are exactly where SYMBOLIC STRUCTURE emerges:
#   - Pen lift → word boundary → semantic unit
#   - Hub → letter junction → morphological feature
#   - Velocity jump → emphasis → pragmatic marker
#   - Curvature peak → character identity → grapheme
#
# The learnable color-space is a JOINT INTERACTION SPACE where:
#   - Position (x,y) provides spatial grounding
#   - Time (t) provides causal ordering
#   - Color (RGB/HSV) provides the learned embedding
#   - Entropy localizes symbolic emergence
#
# ═══════════════════════════════════════════════════════════════════════════════

module InteractionEntropy

using LinearAlgebra
using Statistics

export InteractionPoint, InteractionStroke, InteractionNetwork
export local_entropy, positional_noise, symbolic_distillation
export JointInteractionSpace, learn_colorspace!, entropy_peaks
export ColorManifold, geodesic_distance, parallel_transport
export SymbolicToken, tokenize_by_entropy, detokenize
export EntropyField, field_gradient, critical_points
export interaction_fingerprint, mutual_information
# Stroke operations
export add_point!, compute_entropy!, add_stroke!
# Trit types
export Trit, Minus, Ergodic, Plus, trit_value
# Constants
export ENTROPY_LOW, ENTROPY_CRITICAL, ENTROPY_HIGH

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

const SPLITMIX_GOLDEN = 0x9e3779b97f4a7c15
const SPLITMIX_MIX1 = 0xbf58476d1ce4e5b9
const SPLITMIX_MIX2 = 0x94d049bb133111eb

# Entropy thresholds for symbolic emergence
const ENTROPY_LOW = 0.3      # Below: noise, no symbol
const ENTROPY_CRITICAL = 0.7 # At: phase transition, symbol boundary
const ENTROPY_HIGH = 0.9     # Above: saturated, symbol interior

# GF(3) for trit encoding
abstract type Trit end
struct Minus <: Trit end
struct Ergodic <: Trit end
struct Plus <: Trit end

trit_value(::Minus) = -1
trit_value(::Ergodic) = 0
trit_value(::Plus) = 1

@inline function splitmix64(x::UInt64)::UInt64
    x += SPLITMIX_GOLDEN
    x = (x ⊻ (x >> 30)) * SPLITMIX_MIX1
    x = (x ⊻ (x >> 27)) * SPLITMIX_MIX2
    x ⊻ (x >> 31)
end

# ═══════════════════════════════════════════════════════════════════════════════
# INTERACTION POINT: The quantum of interaction
# ═══════════════════════════════════════════════════════════════════════════════

"""
    InteractionPoint

A single point in the joint interaction space, carrying:
- Position (x, y): spatial coordinates
- Time (t): temporal coordinate  
- Pressure (p): intensity/force
- Velocity (vx, vy): instantaneous motion
- Color (r, g, b): learned embedding
- Local entropy (H): information density at this point
"""
struct InteractionPoint
    # Spatial-temporal coordinates
    x::Float32
    y::Float32
    t::Float32
    p::Float32  # pressure/intensity
    
    # Kinematic features
    vx::Float32  # velocity x
    vy::Float32  # velocity y
    ax::Float32  # acceleration x (for curvature)
    ay::Float32  # acceleration y
    
    # Learned color embedding
    r::Float32
    g::Float32
    b::Float32
    
    # Entropy measures
    local_entropy::Float32      # H(point | neighborhood)
    positional_noise::Float32   # σ²(position)
    symbolic_weight::Float32    # Weight in symbolic distillation
end

function InteractionPoint(x, y; t=0f0, p=1f0, vx=0f0, vy=0f0, ax=0f0, ay=0f0)
    # Initialize with default color (will be learned)
    r, g, b = 0.5f0, 0.5f0, 0.5f0
    InteractionPoint(
        Float32(x), Float32(y), Float32(t), Float32(p),
        Float32(vx), Float32(vy), Float32(ax), Float32(ay),
        r, g, b,
        0f0, 0f0, 0f0  # entropy measures computed later
    )
end

"""
Compute kinematic features from consecutive points.
"""
function compute_kinematics(prev::InteractionPoint, curr::InteractionPoint, dt::Float32)
    vx = (curr.x - prev.x) / dt
    vy = (curr.y - prev.y) / dt
    ax = (vx - prev.vx) / dt
    ay = (vy - prev.vy) / dt
    
    InteractionPoint(
        curr.x, curr.y, curr.t, curr.p,
        vx, vy, ax, ay,
        curr.r, curr.g, curr.b,
        curr.local_entropy, curr.positional_noise, curr.symbolic_weight
    )
end

"""
Speed (magnitude of velocity).
"""
speed(p::InteractionPoint) = sqrt(p.vx^2 + p.vy^2)

"""
Curvature (κ = |v × a| / |v|³).
"""
function curvature(p::InteractionPoint)
    v_mag = speed(p)
    if v_mag < 1e-6
        return 0f0
    end
    cross = abs(p.vx * p.ay - p.vy * p.ax)
    Float32(cross / v_mag^3)
end

# ═══════════════════════════════════════════════════════════════════════════════
# POSITIONAL NOISE: Where uncertainty concentrates
# ═══════════════════════════════════════════════════════════════════════════════

"""
    positional_noise(points, idx, window_size=5)

Compute positional noise (variance) around point idx.
High noise = high uncertainty = potential symbol boundary.

This is the "positionally-dependent noise" that gets symbolically distilled.
"""
function positional_noise(points::Vector{InteractionPoint}, idx::Int; window_size::Int=5)
    n = length(points)
    if n < 2
        return 0f0
    end
    
    # Gather neighborhood
    start_idx = max(1, idx - window_size)
    end_idx = min(n, idx + window_size)
    
    xs = [p.x for p in points[start_idx:end_idx]]
    ys = [p.y for p in points[start_idx:end_idx]]
    
    # Variance in position
    var_x = var(xs; corrected=false)
    var_y = var(ys; corrected=false)
    
    # Also consider velocity variance (kinematic noise)
    vxs = [p.vx for p in points[start_idx:end_idx]]
    vys = [p.vy for p in points[start_idx:end_idx]]
    var_vx = var(vxs; corrected=false)
    var_vy = var(vys; corrected=false)
    
    # Combined positional-kinematic noise
    Float32(sqrt(var_x + var_y + var_vx + var_vy))
end

"""
    velocity_discontinuity(points, idx)

Detect velocity discontinuities (jumps in dv/dt).
These are high-entropy moments where the dynamics change abruptly.
"""
function velocity_discontinuity(points::Vector{InteractionPoint}, idx::Int)
    if idx <= 1 || idx >= length(points)
        return 0f0
    end
    
    prev = points[idx - 1]
    curr = points[idx]
    next = points[idx + 1]
    
    # Velocity before and after
    v_before = sqrt(curr.vx^2 + curr.vy^2)
    v_after = sqrt(next.vx^2 + next.vy^2)
    
    # Discontinuity = |Δv| / mean(v)
    dv = abs(v_after - v_before)
    mean_v = (v_before + v_after) / 2 + 1e-6
    
    Float32(dv / mean_v)
end

# ═══════════════════════════════════════════════════════════════════════════════
# LOCAL ENTROPY: Information density at each point
# ═══════════════════════════════════════════════════════════════════════════════

"""
    local_entropy(points, idx; window_size=5)

Compute local entropy at point idx based on:
1. Positional noise (spatial uncertainty)
2. Velocity discontinuity (kinematic surprise)
3. Curvature extremity (geometric complexity)
4. Pressure variance (intensity modulation)

Returns entropy in [0, 1] where:
- 0 = completely predictable (no information)
- 1 = maximally surprising (maximum information)
"""
function local_entropy(points::Vector{InteractionPoint}, idx::Int; window_size::Int=5)
    n = length(points)
    if n < 3
        return 0.5f0  # Default moderate entropy
    end
    
    # Component 1: Positional noise
    pos_noise = positional_noise(points, idx; window_size)
    
    # Component 2: Velocity discontinuity
    vel_disc = velocity_discontinuity(points, idx)
    
    # Component 3: Curvature (normalized)
    κ = curvature(points[idx])
    κ_norm = tanh(κ)  # Squash to [0, 1)
    
    # Component 4: Pressure variance in neighborhood
    start_idx = max(1, idx - window_size)
    end_idx = min(n, idx + window_size)
    pressures = [p.p for p in points[start_idx:end_idx]]
    p_var = var(pressures; corrected=false)
    p_norm = tanh(p_var * 10)  # Scale and squash
    
    # Weighted combination (learned weights would go here)
    w_pos, w_vel, w_curv, w_press = 0.3f0, 0.3f0, 0.25f0, 0.15f0
    
    H = w_pos * tanh(pos_noise) + 
        w_vel * vel_disc + 
        w_curv * κ_norm + 
        w_press * p_norm
    
    # Clamp to [0, 1]
    clamp(H, 0f0, 1f0)
end

# ═══════════════════════════════════════════════════════════════════════════════
# INTERACTION STROKE: A sequence between pen lifts
# ═══════════════════════════════════════════════════════════════════════════════

"""
    InteractionStroke

A continuous sequence of interaction points (between pen lifts).
Carries aggregate statistics and symbolic distillation.
"""
mutable struct InteractionStroke
    points::Vector{InteractionPoint}
    
    # Aggregate features
    mean_entropy::Float32
    max_entropy::Float32
    entropy_variance::Float32
    
    # Symbolic distillation
    trit::Trit                    # GF(3) classification
    symbol_boundaries::Vector{Int} # Indices of symbol boundaries
    tokens::Vector{Symbol}        # Symbolic tokens
    
    # Color embedding (stroke-level)
    stroke_color::NTuple{3, Float32}  # (R, G, B)
    fingerprint::UInt64
end

function InteractionStroke()
    InteractionStroke(
        InteractionPoint[],
        0f0, 0f0, 0f0,
        Ergodic(),
        Int[],
        Symbol[],
        (0.5f0, 0.5f0, 0.5f0),
        UInt64(0)
    )
end

"""
Add a point to the stroke and update kinematics.
"""
function add_point!(stroke::InteractionStroke, x, y; t=nothing, p=1f0)
    n = length(stroke.points)
    
    # Compute time if not provided
    if t === nothing
        t = n > 0 ? stroke.points[end].t + 0.01f0 : 0f0
    end
    
    # Create point
    point = InteractionPoint(x, y; t=Float32(t), p=Float32(p))
    
    # Update kinematics if we have history
    if n >= 1
        prev = stroke.points[end]
        dt = Float32(t) - prev.t
        if dt > 1e-6
            point = compute_kinematics(prev, point, dt)
        end
    end
    
    push!(stroke.points, point)
    stroke
end

"""
Compute all entropy measures for the stroke.
"""
function compute_entropy!(stroke::InteractionStroke; window_size::Int=5)
    n = length(stroke.points)
    if n < 3
        return stroke
    end
    
    entropies = Float32[]
    
    for i in 1:n
        H = local_entropy(stroke.points, i; window_size)
        push!(entropies, H)
        
        # Update point with entropy (requires reconstructing the point)
        p = stroke.points[i]
        noise = positional_noise(stroke.points, i; window_size)
        
        # Symbolic weight: higher near entropy peaks
        weight = H > ENTROPY_CRITICAL ? H : 0f0
        
        stroke.points[i] = InteractionPoint(
            p.x, p.y, p.t, p.p,
            p.vx, p.vy, p.ax, p.ay,
            p.r, p.g, p.b,
            H, noise, weight
        )
    end
    
    # Aggregate statistics
    stroke.mean_entropy = mean(entropies)
    stroke.max_entropy = maximum(entropies)
    stroke.entropy_variance = var(entropies; corrected=false)
    
    # Find symbol boundaries (entropy peaks)
    stroke.symbol_boundaries = find_entropy_peaks(entropies)
    
    # Assign trit based on entropy profile
    stroke.trit = entropy_to_trit(stroke.mean_entropy, stroke.entropy_variance)
    
    # Compute fingerprint
    stroke.fingerprint = compute_stroke_fingerprint(stroke)
    
    stroke
end

"""
Find peaks in entropy (local maxima above threshold).
These are candidate symbol boundaries.
"""
function find_entropy_peaks(entropies::Vector{Float32}; threshold=ENTROPY_CRITICAL)
    peaks = Int[]
    n = length(entropies)
    
    for i in 2:n-1
        if entropies[i] > threshold &&
           entropies[i] > entropies[i-1] &&
           entropies[i] > entropies[i+1]
            push!(peaks, i)
        end
    end
    
    peaks
end

"""
Map entropy statistics to GF(3) trit.
"""
function entropy_to_trit(mean_H::Float32, var_H::Float32)::Trit
    # High variance + high mean = PLUS (generative, complex)
    # Low variance + moderate mean = ERGODIC (stable, flowing)
    # Low mean = MINUS (contractive, simple)
    
    complexity = mean_H + sqrt(var_H)
    
    if complexity > 0.8
        Plus()
    elseif complexity > 0.4
        Ergodic()
    else
        Minus()
    end
end

"""
Compute deterministic fingerprint for stroke.
"""
function compute_stroke_fingerprint(stroke::InteractionStroke)::UInt64
    fp = UInt64(0)
    
    for (i, p) in enumerate(stroke.points)
        # Quantize position to 16 bits each
        qx = UInt64(clamp(round(p.x * 65535), 0, 65535))
        qy = UInt64(clamp(round(p.y * 65535), 0, 65535))
        
        # Combine with entropy (8 bits)
        qH = UInt64(clamp(round(p.local_entropy * 255), 0, 255))
        
        # Hash together
        point_hash = splitmix64(qx << 48 | qy << 32 | qH << 24 | UInt64(i))
        fp ⊻= point_hash
    end
    
    fp
end

# ═══════════════════════════════════════════════════════════════════════════════
# JOINT INTERACTION SPACE: The learnable color manifold
# ═══════════════════════════════════════════════════════════════════════════════

"""
    JointInteractionSpace

The learned embedding space where:
- Points are mapped to colors based on their interaction context
- High-entropy regions are emphasized (brighter, more saturated)
- Symbol boundaries get distinct hues
- Positional noise gets distilled into color gradients

This is where the "symbolic distillation of positionally-dependent noise" happens.
"""
mutable struct JointInteractionSpace
    strokes::Vector{InteractionStroke}
    
    # Learned parameters
    hue_offset::Float32           # Base hue rotation
    saturation_scale::Float32     # How entropy maps to saturation
    value_scale::Float32          # How entropy maps to brightness
    
    # Entropy field over the space
    entropy_field::Matrix{Float32}  # Discretized entropy map
    field_resolution::Int
    
    # Color manifold parameters
    manifold_dim::Int             # Dimensionality of color manifold
    embedding_weights::Matrix{Float32}  # Position → Color weights
    
    # Statistics
    total_entropy::Float64
    mutual_info::Float64          # I(position; color)
end

function JointInteractionSpace(; field_resolution::Int=64, manifold_dim::Int=3)
    JointInteractionSpace(
        InteractionStroke[],
        0f0, 1f0, 0.65f0,  # InkSight-inspired defaults
        zeros(Float32, field_resolution, field_resolution),
        field_resolution,
        manifold_dim,
        randn(Float32, manifold_dim, 4) * 0.1f0,  # Random init
        0.0, 0.0
    )
end

"""
Add a stroke to the interaction space.
"""
function add_stroke!(space::JointInteractionSpace, stroke::InteractionStroke)
    # Compute entropy if not already done
    if stroke.mean_entropy == 0f0
        compute_entropy!(stroke)
    end
    
    push!(space.strokes, stroke)
    
    # Update entropy field
    update_entropy_field!(space, stroke)
    
    space
end

"""
Update the discretized entropy field with a new stroke.
"""
function update_entropy_field!(space::JointInteractionSpace, stroke::InteractionStroke)
    res = space.field_resolution
    
    for p in stroke.points
        # Map to grid coordinates
        ix = clamp(Int(floor(p.x * res)) + 1, 1, res)
        iy = clamp(Int(floor(p.y * res)) + 1, 1, res)
        
        # Accumulate entropy (max pooling)
        space.entropy_field[ix, iy] = max(space.entropy_field[ix, iy], p.local_entropy)
    end
end

"""
    learn_colorspace!(space; learning_rate=0.01, iterations=100)

Learn the color embedding that maximizes mutual information
between position and color, while preserving entropy structure.

The learned colorspace is a "symbolic distillation" because:
- High-entropy regions get distinct colors (symbols emerge)
- Low-entropy regions get similar colors (noise is suppressed)
- The mapping is deterministic (reproducible)
"""
function learn_colorspace!(space::JointInteractionSpace; 
                          learning_rate::Float32=0.01f0, 
                          iterations::Int=100)
    if isempty(space.strokes)
        return space
    end
    
    # Collect all points
    all_points = InteractionPoint[]
    for stroke in space.strokes
        append!(all_points, stroke.points)
    end
    
    n = length(all_points)
    if n < 10
        return space
    end
    
    # Gradient descent on embedding weights
    for iter in 1:iterations
        grad = zeros(Float32, size(space.embedding_weights))
        
        for p in all_points
            # Current embedding: position → color
            pos_vec = Float32[p.x, p.y, p.t, p.local_entropy]
            color_vec = space.embedding_weights * pos_vec
            
            # Target: high entropy → high saturation, distinct hue
            target_sat = p.local_entropy
            target_val = 0.3f0 + 0.7f0 * p.local_entropy
            
            # Simple gradient: move colors toward entropy-weighted targets
            error = Float32[
                target_sat - abs(color_vec[1]),
                target_sat - abs(color_vec[2]),
                target_val - abs(color_vec[3])
            ]
            
            # Outer product for gradient
            for i in 1:3, j in 1:4
                grad[i, j] += error[i] * pos_vec[j] * p.symbolic_weight
            end
        end
        
        # Update weights
        space.embedding_weights .+= learning_rate * grad / n
    end
    
    # Apply learned colors to points
    apply_learned_colors!(space)
    
    # Compute mutual information
    space.mutual_info = compute_mutual_information(space)
    
    space
end

"""
Apply the learned color embedding to all points.
"""
function apply_learned_colors!(space::JointInteractionSpace)
    for stroke in space.strokes
        n_strokes = length(space.strokes)
        stroke_idx = findfirst(s -> s === stroke, space.strokes)
        
        for (i, p) in enumerate(stroke.points)
            # Position features
            pos_vec = Float32[p.x, p.y, p.t, p.local_entropy]
            
            # Learned embedding
            raw_color = space.embedding_weights * pos_vec
            
            # Map to HSV based on entropy
            hue = mod(space.hue_offset + atan(raw_color[1], raw_color[2]) / (2π) * 360, 360)
            sat = clamp(space.saturation_scale * p.local_entropy, 0, 1)
            val = clamp(space.value_scale * (0.5 + 0.5 * p.local_entropy), 0, 1)
            
            # HSV to RGB
            r, g, b = hsv_to_rgb(hue, sat, val)
            
            stroke.points[i] = InteractionPoint(
                p.x, p.y, p.t, p.p,
                p.vx, p.vy, p.ax, p.ay,
                r, g, b,
                p.local_entropy, p.positional_noise, p.symbolic_weight
            )
        end
        
        # Update stroke color (average of point colors)
        if !isempty(stroke.points)
            r_avg = mean(p.r for p in stroke.points)
            g_avg = mean(p.g for p in stroke.points)
            b_avg = mean(p.b for p in stroke.points)
            stroke.stroke_color = (r_avg, g_avg, b_avg)
        end
    end
end

"""
HSV to RGB conversion.
"""
function hsv_to_rgb(h::Real, s::Real, v::Real)
    h = mod(h, 360)
    c = v * s
    x = c * (1 - abs(mod(h / 60, 2) - 1))
    m = v - c
    
    r, g, b = if h < 60
        (c, x, 0)
    elseif h < 120
        (x, c, 0)
    elseif h < 180
        (0, c, x)
    elseif h < 240
        (0, x, c)
    elseif h < 300
        (x, 0, c)
    else
        (c, 0, x)
    end
    
    (Float32(r + m), Float32(g + m), Float32(b + m))
end

# ═══════════════════════════════════════════════════════════════════════════════
# SYMBOLIC DISTILLATION: From noise to tokens
# ═══════════════════════════════════════════════════════════════════════════════

"""
    SymbolicToken

A token extracted from high-entropy regions.
This is the "symbolic distillation" - noise becomes discrete symbol.
"""
struct SymbolicToken
    symbol::Symbol          # The token symbol
    start_idx::Int          # Start index in stroke
    end_idx::Int            # End index in stroke
    stroke_idx::Int         # Which stroke
    entropy::Float32        # Mean entropy in region
    color::NTuple{3,Float32} # Representative color
    fingerprint::UInt64     # Unique identifier
end

"""
    tokenize_by_entropy(space; min_entropy=0.5)

Extract symbolic tokens from high-entropy regions.
This is where positionally-dependent noise becomes discrete symbols.

The key insight: symbols emerge at entropy peaks because that's where
the system transitions between predictable and unpredictable states.
"""
function tokenize_by_entropy(space::JointInteractionSpace; min_entropy::Float32=0.5f0)
    tokens = SymbolicToken[]
    
    for (stroke_idx, stroke) in enumerate(space.strokes)
        boundaries = stroke.symbol_boundaries
        
        if isempty(boundaries)
            # Whole stroke is one token
            if stroke.mean_entropy >= min_entropy
                token = create_token(stroke, 1, length(stroke.points), stroke_idx)
                push!(tokens, token)
            end
        else
            # Split by boundaries
            starts = [1; boundaries .+ 1]
            ends = [boundaries .- 1; length(stroke.points)]
            
            for (start_i, end_i) in zip(starts, ends)
                if end_i >= start_i
                    region_points = stroke.points[start_i:end_i]
                    region_entropy = mean(p.local_entropy for p in region_points)
                    
                    if region_entropy >= min_entropy
                        token = create_token(stroke, start_i, end_i, stroke_idx)
                        push!(tokens, token)
                    end
                end
            end
        end
    end
    
    tokens
end

"""
Create a symbolic token from a region of a stroke.
"""
function create_token(stroke::InteractionStroke, start_idx::Int, end_idx::Int, stroke_idx::Int)
    points = stroke.points[start_idx:end_idx]
    
    # Compute region statistics
    entropy = mean(p.local_entropy for p in points)
    r_avg = mean(p.r for p in points)
    g_avg = mean(p.g for p in points)
    b_avg = mean(p.b for p in points)
    
    # Generate symbol name from fingerprint
    fp = UInt64(0)
    for p in points
        qx = UInt64(clamp(round(p.x * 255), 0, 255))
        qy = UInt64(clamp(round(p.y * 255), 0, 255))
        fp ⊻= splitmix64(qx << 8 | qy)
    end
    
    # Convert fingerprint to readable symbol
    symbol_name = Symbol("T", string(fp % 10000, base=16))
    
    SymbolicToken(
        symbol_name,
        start_idx, end_idx, stroke_idx,
        entropy,
        (r_avg, g_avg, b_avg),
        fp
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# MUTUAL INFORMATION: Quantifying the color-position relationship
# ═══════════════════════════════════════════════════════════════════════════════

"""
    compute_mutual_information(space)

Compute I(position; color) - how much information the learned color
carries about position.

High MI = colors are informative about where things are (good embedding)
Low MI = colors are random with respect to position (poor embedding)
"""
function compute_mutual_information(space::JointInteractionSpace)
    # Collect all points
    all_points = InteractionPoint[]
    for stroke in space.strokes
        append!(all_points, stroke.points)
    end
    
    n = length(all_points)
    if n < 10
        return 0.0
    end
    
    # Discretize position and color
    n_bins = 16
    
    # Joint histogram P(position, color)
    joint = zeros(n_bins, n_bins, n_bins, n_bins)  # x, y, r, g
    
    for p in all_points
        ix = clamp(Int(floor(p.x * n_bins)) + 1, 1, n_bins)
        iy = clamp(Int(floor(p.y * n_bins)) + 1, 1, n_bins)
        ir = clamp(Int(floor(p.r * n_bins)) + 1, 1, n_bins)
        ig = clamp(Int(floor(p.g * n_bins)) + 1, 1, n_bins)
        joint[ix, iy, ir, ig] += 1
    end
    
    joint ./= n
    
    # Marginals
    p_pos = sum(joint, dims=(3, 4))[:, :, 1, 1]
    p_col = sum(joint, dims=(1, 2))[1, 1, :, :]
    
    # Mutual information: I(X;Y) = Σ p(x,y) log(p(x,y) / (p(x)p(y)))
    mi = 0.0
    for ix in 1:n_bins, iy in 1:n_bins, ir in 1:n_bins, ig in 1:n_bins
        pxy = joint[ix, iy, ir, ig]
        px = p_pos[ix, iy]
        py = p_col[ir, ig]
        
        if pxy > 1e-10 && px > 1e-10 && py > 1e-10
            mi += pxy * log2(pxy / (px * py))
        end
    end
    
    mi
end

# ═══════════════════════════════════════════════════════════════════════════════
# ENTROPY FIELD: Continuous view of interaction entropy
# ═══════════════════════════════════════════════════════════════════════════════

"""
    EntropyField

A continuous scalar field over the interaction space.
Critical points of this field are where symbols emerge.
"""
struct EntropyField
    values::Matrix{Float32}
    gradient_x::Matrix{Float32}
    gradient_y::Matrix{Float32}
    resolution::Int
end

function EntropyField(space::JointInteractionSpace)
    res = space.field_resolution
    values = copy(space.entropy_field)
    
    # Compute gradient via finite differences
    grad_x = zeros(Float32, res, res)
    grad_y = zeros(Float32, res, res)
    
    for i in 2:res-1, j in 2:res-1
        grad_x[i, j] = (values[i+1, j] - values[i-1, j]) / 2
        grad_y[i, j] = (values[i, j+1] - values[i, j-1]) / 2
    end
    
    EntropyField(values, grad_x, grad_y, res)
end

"""
Find critical points of the entropy field (∇H = 0).
These are where symbolic structure crystallizes.
"""
function critical_points(field::EntropyField; threshold::Float32=0.1f0)
    res = field.resolution
    criticals = Tuple{Int, Int, Symbol}[]  # (x, y, type)
    
    for i in 2:res-1, j in 2:res-1
        grad_mag = sqrt(field.gradient_x[i,j]^2 + field.gradient_y[i,j]^2)
        
        if grad_mag < threshold
            # Classify by Hessian
            H = field.values
            hxx = H[i+1,j] - 2H[i,j] + H[i-1,j]
            hyy = H[i,j+1] - 2H[i,j] + H[i,j-1]
            hxy = (H[i+1,j+1] - H[i+1,j-1] - H[i-1,j+1] + H[i-1,j-1]) / 4
            
            det = hxx * hyy - hxy^2
            trace = hxx + hyy
            
            crit_type = if det > 0 && trace < 0
                :maximum  # Local max = symbol center
            elseif det > 0 && trace > 0
                :minimum  # Local min = between symbols
            elseif det < 0
                :saddle   # Saddle = symbol boundary
            else
                :degenerate
            end
            
            push!(criticals, (i, j, crit_type))
        end
    end
    
    criticals
end

# ═══════════════════════════════════════════════════════════════════════════════
# COLOR MANIFOLD: Geodesics and parallel transport
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ColorManifold

The learned color embedding forms a manifold.
Geodesics on this manifold correspond to "natural" color transitions.
"""
struct ColorManifold
    metric::Matrix{Float32}  # Riemannian metric (learned)
    dimension::Int
end

function ColorManifold(space::JointInteractionSpace)
    # Learn metric from color distances
    d = space.manifold_dim
    
    # Simple Euclidean metric as default
    metric = Matrix{Float32}(I, d, d)
    
    # Could learn from data: points with similar entropy should have similar colors
    # metric = learned_metric_from_entropy(space)
    
    ColorManifold(metric, d)
end

"""
Geodesic distance between two colors on the manifold.
"""
function geodesic_distance(m::ColorManifold, c1::NTuple{3,Float32}, c2::NTuple{3,Float32})
    dc = Float32[c1[1]-c2[1], c1[2]-c2[2], c1[3]-c2[3]]
    sqrt(dc' * m.metric * dc)
end

"""
Parallel transport a color vector along a path.
(Simplified: just translate, proper impl would use Christoffel symbols)
"""
function parallel_transport(m::ColorManifold, v::NTuple{3,Float32}, 
                           from::NTuple{3,Float32}, to::NTuple{3,Float32})
    # In flat space, parallel transport is identity
    # In curved space, would need Christoffel symbols
    v
end

# ═══════════════════════════════════════════════════════════════════════════════
# INTERACTION FINGERPRINT: Unique identifier for an interaction pattern
# ═══════════════════════════════════════════════════════════════════════════════

"""
    interaction_fingerprint(space)

Compute a unique fingerprint for the entire interaction space.
This is schedule-invariant: same interactions → same fingerprint.
"""
function interaction_fingerprint(space::JointInteractionSpace)::UInt64
    fp = UInt64(0)
    
    for stroke in space.strokes
        fp ⊻= stroke.fingerprint
    end
    
    # Also incorporate entropy field structure
    for i in 1:space.field_resolution, j in 1:space.field_resolution
        if space.entropy_field[i,j] > ENTROPY_CRITICAL
            fp ⊻= splitmix64(UInt64(i) << 32 | UInt64(j))
        end
    end
    
    fp
end

# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTS SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

# The key functions for "learnable color-spaces of joint interaction spaces
# as symbolic distillation of positionally-dependent noise":
#
# 1. local_entropy(points, idx) - Compute information density at each point
# 2. positional_noise(points, idx) - Measure positionally-dependent uncertainty
# 3. JointInteractionSpace() - Create the learnable space
# 4. learn_colorspace!(space) - Learn the color embedding
# 5. tokenize_by_entropy(space) - Extract symbolic tokens from high-entropy regions
# 6. compute_mutual_information(space) - Quantify color-position relationship
# 7. critical_points(field) - Find where symbols crystallize
# 8. interaction_fingerprint(space) - Unique identifier for pattern

end # module InteractionEntropy
