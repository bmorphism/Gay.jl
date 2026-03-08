# ink_traced_criticality.jl
# Digital Ink + String Diagram Tracing + Criticality Phase Transitions
#
# Synthesizes:
#   1. Google Research ink tokenization (Berent SwissText2024):
#      - Image representation: time→intensity, distance→color channel
#      - Text representation: discretized relative coordinates as tokens
#   2. TracedCriticality pen-lift states (this repo)
#   3. SARW/criticality/Landauer physicality threshold
#
# The unified view:
#   PEN DOWN (connected trace) = ink stroke = reversible = subcritical
#   PEN LIFT (stroke break)    = stroke boundary = irreversible = Landauer cost
#   The MOMENT of lifting      = criticality = τ_c = information → physical
#
# This addresses greenteatree01's critique:
#   Fixed GF(3) decomposition conflicts with dynamic criticality
#   → Ink strokes ARE the dynamic structure
#   → Stroke connectivity = current effective dimension
#   → Pen lifts = phase transitions in the neural criticality landscape

module InkTracedCriticality

using Random

# Import from TracedCriticality
# (In practice these would be imported, here we define compatible types)

@inline function sm64(x::UInt64)::UInt64
    z = x + 0x9e3779b97f4a7c15
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    z ⊻ (z >> 31)
end

@enum Trit::Int8 MINUS=-1 ERGODIC=0 PLUS=1
@enum PenState::Int8 DOWN=0 LIFTING=1 UP=2

# ═══════════════════════════════════════════════════════════════════════════
# INK POINT: The atomic unit of digital handwriting
# ═══════════════════════════════════════════════════════════════════════════

"""
    InkPoint

A single point in digital ink, as captured by stylus/touch input.
Contains position, time, and optional pressure.

From Berent's talk: raw input is 2D points + time + pressure.
"""
struct InkPoint
    x::Float32
    y::Float32
    t::Float32          # Time since stroke start (ms)
    pressure::Float32   # 0.0-1.0, optional
end

InkPoint(x, y, t) = InkPoint(Float32(x), Float32(y), Float32(t), 1.0f0)

"""
Distance between two ink points (Euclidean).
"""
function point_distance(p1::InkPoint, p2::InkPoint)
    sqrt((p2.x - p1.x)^2 + (p2.y - p1.y)^2)
end

"""
Speed between two ink points (distance / time).
"""
function point_speed(p1::InkPoint, p2::InkPoint)
    dt = p2.t - p1.t
    dt > 0 ? point_distance(p1, p2) / dt : 0.0f0
end

# ═══════════════════════════════════════════════════════════════════════════
# INK STROKE: Connected sequence of points (pen down → pen lift)
# ═══════════════════════════════════════════════════════════════════════════

"""
    InkStroke

A single stroke: all points between pen-down and pen-lift.
This is the atomic REVERSIBLE unit - within a stroke, no Landauer cost.

The stroke boundary (pen lift) is where information becomes physical.
"""
struct InkStroke
    points::Vector{InkPoint}
    trit::Trit                  # GF(3) polarity of this stroke
    fingerprint::UInt64
end

function InkStroke(points::Vector{InkPoint}; seed::UInt64=UInt64(1069))
    # Compute fingerprint from point sequence
    fp = seed
    for p in points
        fp = sm64(fp ⊻ reinterpret(UInt64, Float64(p.x + p.y * 1000 + p.t)))
    end
    
    # Trit based on stroke geometry
    # Simplified: based on net direction
    if length(points) >= 2
        dx = points[end].x - points[1].x
        dy = points[end].y - points[1].y
        angle = atan(dy, dx)
        trit = if angle < -π/3
            MINUS
        elseif angle > π/3
            PLUS
        else
            ERGODIC
        end
    else
        trit = ERGODIC
    end
    
    InkStroke(points, trit, fp)
end

"""
Total arc length of stroke.
"""
function stroke_length(s::InkStroke)
    length(s.points) < 2 && return 0.0f0
    sum(point_distance(s.points[i], s.points[i+1]) for i in 1:length(s.points)-1)
end

"""
Total duration of stroke (ms).
"""
function stroke_duration(s::InkStroke)
    isempty(s.points) && return 0.0f0
    s.points[end].t - s.points[1].t
end

"""
Average writing speed.
"""
function stroke_speed(s::InkStroke)
    dur = stroke_duration(s)
    dur > 0 ? stroke_length(s) / dur : 0.0f0
end

# ═══════════════════════════════════════════════════════════════════════════
# DUAL REPRESENTATION: Image + Text (Berent's insight)
# ═══════════════════════════════════════════════════════════════════════════

"""
    InkImageRepresentation

Render stroke as RGB image where:
- R channel: time (normalized 0-1 over stroke duration)
- G channel: cumulative distance (normalized)
- B channel: pressure or speed

This preserves temporal information lost in naive rendering.
"""
struct InkImageRepresentation
    width::Int
    height::Int
    pixels::Matrix{NTuple{3, Float32}}  # RGB
    fingerprint::UInt64
end

"""
Render stroke to image representation.
"""
function render_to_image(stroke::InkStroke; width::Int=64, height::Int=64)
    pixels = fill((0.0f0, 0.0f0, 0.0f0), width, height)
    
    isempty(stroke.points) && return InkImageRepresentation(width, height, pixels, stroke.fingerprint)
    
    # Compute bounds
    xs = [p.x for p in stroke.points]
    ys = [p.y for p in stroke.points]
    min_x, max_x = minimum(xs), maximum(xs)
    min_y, max_y = minimum(ys), maximum(ys)
    
    # Add padding
    range_x = max(max_x - min_x, 1.0f0) * 1.1f0
    range_y = max(max_y - min_y, 1.0f0) * 1.1f0
    
    # Normalize time and distance
    total_time = stroke_duration(stroke)
    total_dist = stroke_length(stroke)
    
    cumulative_dist = 0.0f0
    
    for i in 1:length(stroke.points)
        p = stroke.points[i]
        
        # Map to pixel coordinates
        px = clamp(round(Int, (p.x - min_x) / range_x * (width - 1)) + 1, 1, width)
        py = clamp(round(Int, (p.y - min_y) / range_y * (height - 1)) + 1, 1, height)
        
        # Compute channels
        r = total_time > 0 ? (p.t - stroke.points[1].t) / total_time : 0.5f0
        g = total_dist > 0 ? cumulative_dist / total_dist : 0.5f0
        b = p.pressure
        
        pixels[px, py] = (r, g, b)
        
        # Update cumulative distance
        if i < length(stroke.points)
            cumulative_dist += point_distance(p, stroke.points[i+1])
        end
    end
    
    InkImageRepresentation(width, height, pixels, stroke.fingerprint)
end

"""
    InkTextRepresentation

Tokenize stroke as text: sequence of discretized relative coordinates.

From Berent's talk:
- Normalize scale
- Discretize to grid
- Encode as (Δx, Δy) pairs
- Special tokens for stroke start/end
"""
struct InkTextRepresentation
    tokens::Vector{String}
    fingerprint::UInt64
end

"""
Tokenize stroke to text representation.
"""
function tokenize_to_text(stroke::InkStroke; grid_size::Int=32, max_tokens::Int=256)
    tokens = String["<STROKE_START>"]
    
    isempty(stroke.points) && return InkTextRepresentation(push!(tokens, "<STROKE_END>"), stroke.fingerprint)
    
    # Normalize to unit square
    xs = [p.x for p in stroke.points]
    ys = [p.y for p in stroke.points]
    min_x, max_x = minimum(xs), maximum(xs)
    min_y, max_y = minimum(ys), maximum(ys)
    range_val = max(max_x - min_x, max_y - min_y, 1.0f0)
    
    # Sample points (Berent: balance detail vs context length)
    sample_rate = max(1, length(stroke.points) ÷ max_tokens)
    sampled = stroke.points[1:sample_rate:end]
    
    prev_gx, prev_gy = 0, 0
    
    for (i, p) in enumerate(sampled)
        # Normalize and discretize
        nx = (p.x - min_x) / range_val
        ny = (p.y - min_y) / range_val
        gx = clamp(round(Int, nx * grid_size), 0, grid_size)
        gy = clamp(round(Int, ny * grid_size), 0, grid_size)
        
        if i == 1
            # Absolute position for first point
            push!(tokens, "($gx,$gy)")
        else
            # Relative position for subsequent points
            dx = gx - prev_gx
            dy = gy - prev_gy
            push!(tokens, "(Δ$dx,Δ$dy)")
        end
        
        prev_gx, prev_gy = gx, gy
        
        length(tokens) >= max_tokens - 1 && break
    end
    
    push!(tokens, "<STROKE_END>")
    
    InkTextRepresentation(tokens, stroke.fingerprint)
end

# ═══════════════════════════════════════════════════════════════════════════
# TRACED INK: String diagram with pen-lift = phase transition
# ═══════════════════════════════════════════════════════════════════════════

"""
    PenLiftEvent

The moment when pen lifts - information becomes physical.

This is where:
1. Landauer cost is paid (kT ln 2 per bit)
2. SARW visited set is updated (irreversible)
3. GF(3) segment boundary is established
4. Criticality regime may change
"""
struct PenLiftEvent
    timestamp::Float32
    from_stroke::Int
    to_stroke::Int
    lift_duration::Float32      # Time pen was up
    erased_bits::Float64        # Information cost
    d_2_at_lift::Float64        # Distance to criticality
    reason::Symbol              # :natural, :pause, :word_boundary, :line_break
    fingerprint::UInt64
end

"""
    TracedInk

A complete piece of digital ink with explicit pen-lift tracking.
Unifies:
- Ink strokes (the pen-down segments)
- Pen-lift events (the boundaries)
- Criticality dynamics (d_2 evolution)
- SARW visited set (physical trace)
"""
mutable struct TracedInk
    name::Symbol
    strokes::Vector{InkStroke}
    lifts::Vector{PenLiftEvent}
    current_stroke_idx::Int
    pen_state::PenState
    
    # SARW tracking
    visited_positions::Set{UInt64}  # Hashed positions we've been
    
    # Criticality dynamics
    d_2::Float64                    # Distance to β=2 fixed point
    correlation_length::Float64     # ξ: diverges at criticality
    
    # Dual representations (lazily computed)
    image_rep::Union{Nothing, Vector{InkImageRepresentation}}
    text_rep::Union{Nothing, Vector{InkTextRepresentation}}
    
    # Tracking
    total_erased_bits::Float64
    seed::UInt64
    fingerprint::UInt64
end

function TracedInk(name::Symbol; seed::UInt64=UInt64(1069), d_2::Float64=0.5)
    fp = sm64(seed ⊻ hash(name))
    TracedInk(
        name,
        InkStroke[],
        PenLiftEvent[],
        0,
        UP,  # Start with pen up
        Set{UInt64}(),
        d_2,
        1.0,
        nothing,
        nothing,
        0.0,
        seed,
        fp
    )
end

"""
    pen_down!(ink, point)

Start a new stroke at the given point.
"""
function pen_down!(ink::TracedInk, point::InkPoint)
    ink.pen_state = DOWN
    push!(ink.strokes, InkStroke([point]; seed=ink.fingerprint))
    ink.current_stroke_idx = length(ink.strokes)
    
    # Invalidate cached representations
    ink.image_rep = nothing
    ink.text_rep = nothing
    
    # Update fingerprint
    ink.fingerprint = sm64(ink.fingerprint ⊻ reinterpret(UInt64, Float64(point.x)))
    
    ink
end

"""
    trace_point!(ink, point) → Bool

Add a point to the current stroke. Returns false if SARW collision detected.
"""
function trace_point!(ink::TracedInk, point::InkPoint)
    ink.pen_state != DOWN && error("Cannot trace with pen up")
    
    # Hash position for SARW check
    pos_hash = sm64(reinterpret(UInt64, Float64(point.x * 1000 + point.y)))
    
    # Check SARW collision
    if pos_hash in ink.visited_positions
        # Collision! Must lift pen (information becomes physical)
        pen_lift!(ink, point.t, :sarw_collision)
        return false
    end
    
    # Add to visited set (irreversible!)
    push!(ink.visited_positions, pos_hash)
    
    # Add point to current stroke
    current = ink.strokes[ink.current_stroke_idx]
    push!(current.points, point)
    
    # Update correlation length (grows while tracing)
    ink.correlation_length *= 1.0 + 0.05 / max(ink.d_2, 0.01)
    
    # Check for Dragon King (criticality breakdown)
    if ink.correlation_length > 100.0 / max(ink.d_2, 0.01)
        pen_lift!(ink, point.t, :dragon_king)
        return false
    end
    
    # Update fingerprint
    ink.fingerprint = sm64(ink.fingerprint ⊻ pos_hash)
    
    # Invalidate cached representations
    ink.image_rep = nothing
    ink.text_rep = nothing
    
    true
end

"""
    pen_lift!(ink, timestamp, reason)

Lift the pen - the moment information becomes physical.
"""
function pen_lift!(ink::TracedInk, timestamp::Float32, reason::Symbol)
    ink.pen_state == UP && return ink
    
    # Calculate Landauer cost
    # More bits erased when further from criticality
    erased_bits = log2(max(1.0, ink.correlation_length)) * (1.0 + ink.d_2)
    
    lift = PenLiftEvent(
        timestamp,
        ink.current_stroke_idx,
        ink.current_stroke_idx + 1,
        0.0f0,  # Will be updated when pen goes down again
        erased_bits,
        ink.d_2,
        reason,
        sm64(ink.fingerprint ⊻ UInt64(round(timestamp * 1000)))
    )
    push!(ink.lifts, lift)
    
    # Update state
    ink.pen_state = UP
    ink.total_erased_bits += erased_bits
    
    # Reset correlation length (phase transition occurred)
    ink.correlation_length = 1.0
    
    ink
end

# ═══════════════════════════════════════════════════════════════════════════
# CRITICALITY DYNAMICS
# ═══════════════════════════════════════════════════════════════════════════

"""
    update_d_2!(ink, measurement)

Update distance to criticality based on external measurement.
This is DYNAMIC - changes based on current brain/writing state.
"""
function update_d_2!(ink::TracedInk, measurement::Float64)
    @assert 0.0 ≤ measurement ≤ 1.0
    
    # Exponential moving average
    α = 0.3
    ink.d_2 = α * measurement + (1 - α) * ink.d_2
    
    # Correlation length inversely related
    ink.correlation_length = 1.0 / max(ink.d_2, 0.01)
end

"""
    effective_dimension(ink) → Int

Effective GF(n) based on stroke connectivity.

At criticality: GF(3) → GF(1) (all synchronized)
Far from criticality: full GF(3) (independent regions)
"""
function effective_dimension(ink::TracedInk)::Int
    if ink.d_2 < 0.05
        1  # Critical: everything synchronized
    elseif ink.d_2 < 0.2
        2  # Near critical: partial sync
    else
        3  # Subcritical: full GF(3)
    end
end

"""
    criticality_state(ink) → Symbol
"""
function criticality_state(ink::TracedInk)::Symbol
    if ink.d_2 < 0.1
        :critical
    elseif ink.d_2 < 0.5
        :subcritical
    else
        :supercritical
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# PHYSICALITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

"""
    physicality_status(ink)

Comprehensive analysis of when/whether information became physical.
"""
function physicality_status(ink::TracedInk)
    total_points = sum(length(s.points) for s in ink.strokes; init=0)
    
    (
        is_physical = ink.total_erased_bits > 0,
        erased_bits = ink.total_erased_bits,
        lift_count = length(ink.lifts),
        stroke_count = length(ink.strokes),
        total_points = total_points,
        visited_positions = length(ink.visited_positions),
        reversible_fraction = total_points > 0 ? 
            max(0.0, 1.0 - ink.total_erased_bits / total_points) : 1.0,
        current_d_2 = ink.d_2,
        effective_dim = effective_dimension(ink),
        criticality = criticality_state(ink)
    )
end

"""
    landauer_cost(ink) → Float64

Total thermodynamic cost in units of kT ln 2.
"""
landauer_cost(ink::TracedInk) = ink.total_erased_bits

"""
    is_reversible(ink) → Bool

Can the entire trace be undone without Landauer cost?
"""
is_reversible(ink::TracedInk) = isempty(ink.lifts)

# ═══════════════════════════════════════════════════════════════════════════
# DUAL REPRESENTATION ACCESS
# ═══════════════════════════════════════════════════════════════════════════

"""
    get_image_representation(ink)

Get image representations for all strokes (cached).
"""
function get_image_representation(ink::TracedInk)
    if ink.image_rep === nothing
        ink.image_rep = [render_to_image(s) for s in ink.strokes]
    end
    ink.image_rep
end

"""
    get_text_representation(ink)

Get text representations for all strokes (cached).
"""
function get_text_representation(ink::TracedInk)
    if ink.text_rep === nothing
        ink.text_rep = [tokenize_to_text(s) for s in ink.strokes]
    end
    ink.text_rep
end

"""
    combined_tokens(ink) → Vector{String}

Get combined text tokens for entire ink (all strokes).
This is what would be fed to a VLM.
"""
function combined_tokens(ink::TracedInk)
    text_reps = get_text_representation(ink)
    tokens = String["<INK_START>"]
    
    for (i, rep) in enumerate(text_reps)
        append!(tokens, rep.tokens)
        if i < length(text_reps)
            push!(tokens, "<PEN_LIFT>")
        end
    end
    
    push!(tokens, "<INK_END>")
    tokens
end

# ═══════════════════════════════════════════════════════════════════════════
# GF(3) BALANCE ACROSS STROKES
# ═══════════════════════════════════════════════════════════════════════════

"""
    stroke_trit_sum(ink) → Int

Sum of GF(3) trits across all strokes.
Should be 0 mod 3 for balanced ink.
"""
function stroke_trit_sum(ink::TracedInk)
    sum(Int(s.trit) for s in ink.strokes; init=0)
end

"""
    is_gf3_balanced(ink) → Bool

Check if ink satisfies GF(3) conservation.
"""
is_gf3_balanced(ink::TracedInk) = stroke_trit_sum(ink) % 3 == 0

"""
    balance_recommendation(ink) → Trit

What trit value is needed to balance the ink?
"""
function balance_recommendation(ink::TracedInk)
    current_sum = stroke_trit_sum(ink) % 3
    if current_sum == 0
        ERGODIC  # Already balanced
    elseif current_sum == 1 || current_sum == -2
        MINUS    # Need -1 to balance
    else
        PLUS     # Need +1 to balance
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════

"""
    ink_status(ink)

Print comprehensive status.
"""
function ink_status(ink::TracedInk)
    phys = physicality_status(ink)
    
    println("═══════════════════════════════════════════════════════════════")
    println("       TRACED INK: $(ink.name)")
    println("═══════════════════════════════════════════════════════════════")
    println()
    println("  PEN STATE:           $(ink.pen_state)")
    println("  Strokes:             $(length(ink.strokes))")
    println("  Total Points:        $(phys.total_points)")
    println()
    println("  ┌─────────────────── SARW STATUS ───────────────────┐")
    println("  │  Visited positions:  $(phys.visited_positions)")
    println("  │  Pen lifts:          $(phys.lift_count)")
    println("  │  Is reversible:      $(is_reversible(ink))")
    println("  └───────────────────────────────────────────────────┘")
    println()
    println("  ┌─────────────────── CRITICALITY ───────────────────┐")
    println("  │  d_2 (distance):     $(round(ink.d_2, digits=3))")
    println("  │  State:              $(phys.criticality)")
    println("  │  Effective dim:      GF($(phys.effective_dim))")
    println("  │  Correlation ξ:      $(round(ink.correlation_length, digits=2))")
    println("  └───────────────────────────────────────────────────┘")
    println()
    println("  ┌─────────────────── PHYSICALITY ───────────────────┐")
    println("  │  Information physical: $(phys.is_physical)")
    println("  │  Landauer cost:        $(round(phys.erased_bits, digits=2)) bits")
    println("  │  Reversible fraction:  $(round(100*phys.reversible_fraction, digits=1))%")
    println("  └───────────────────────────────────────────────────┘")
    println()
    println("  ┌─────────────────── GF(3) BALANCE ─────────────────┐")
    println("  │  Stroke trits:         $(join([string(s.trit) for s in ink.strokes], ", "))")
    println("  │  Sum mod 3:            $(stroke_trit_sum(ink) % 3)")
    println("  │  Balanced:             $(is_gf3_balanced(ink))")
    println("  │  Recommendation:       $(balance_recommendation(ink))")
    println("  └───────────────────────────────────────────────────┘")
    println()
    
    # Show lift events
    if !isempty(ink.lifts)
        println("  PEN LIFT EVENTS:")
        for (i, lift) in enumerate(ink.lifts)
            println("    $i. t=$(round(lift.timestamp, digits=1))ms: $(lift.reason) " *
                    "($(round(lift.erased_bits, digits=2)) bits, d_2=$(round(lift.d_2_at_lift, digits=3)))")
        end
        println()
    end
    
    # Show tokens preview
    tokens = combined_tokens(ink)
    preview = length(tokens) > 10 ? join(tokens[1:10], " ") * " ..." : join(tokens, " ")
    println("  TOKENS PREVIEW: $preview")
    println()
    println("═══════════════════════════════════════════════════════════════")
end

# ═══════════════════════════════════════════════════════════════════════════
# DEMONSTRATION: The unified framework
# ═══════════════════════════════════════════════════════════════════════════

"""
    demo_traced_ink()

Demonstrate the unified framework:
- Pen down/up as phase transitions
- SARW collision as physicality threshold
- GF(3) balance across strokes
- Dual representation (image + text)
"""
function demo_traced_ink()
    println("\n" * "="^70)
    println("  TRACED INK DEMONSTRATION")
    println("  Unifying: SARW + Criticality + Landauer + String Diagrams")
    println("="^70 * "\n")
    
    # Create traced ink
    ink = TracedInk(:demo; seed=UInt64(1069), d_2=0.3)
    
    # Simulate writing "hi" with three strokes
    # Stroke 1: vertical line of 'h'
    pen_down!(ink, InkPoint(10, 50, 0))
    for i in 1:20
        trace_point!(ink, InkPoint(10, 50 - i*2, Float32(i*10)))
    end
    pen_lift!(ink, 210.0f0, :natural)
    
    # Stroke 2: hump of 'h'
    pen_down!(ink, InkPoint(10, 30, 300))
    for i in 1:10
        x = 10 + i*2
        y = 30 - sin(i/10 * π) * 15
        trace_point!(ink, InkPoint(x, y, Float32(300 + i*10)))
    end
    pen_lift!(ink, 410.0f0, :natural)
    
    # Stroke 3: dot of 'i'
    pen_down!(ink, InkPoint(40, 10, 500))
    trace_point!(ink, InkPoint(41, 11, 510.0f0))
    pen_lift!(ink, 520.0f0, :natural)
    
    # Show status
    ink_status(ink)
    
    # Show the key insight
    println("\n" * "-"^70)
    println("  KEY INSIGHT: PEN LIFT = PHASE TRANSITION")
    println("-"^70)
    println()
    println("  Within a stroke (pen DOWN):")
    println("    • Information flow is REVERSIBLE")
    println("    • No Landauer cost (can undo)")
    println("    • SARW can continue")
    println("    • GF(3) trit accumulates")
    println()
    println("  At stroke boundary (pen LIFT):")
    println("    • Information becomes PHYSICAL")
    println("    • Landauer cost: kT ln 2 per bit")
    println("    • SARW segment ends")
    println("    • Criticality may shift (d_2 changes)")
    println()
    println("  This addresses greenteatree01's critique:")
    println("    • GF(3) is NOT fixed regional decomposition")
    println("    • It's DYNAMIC: follows the ink strokes")
    println("    • Effective dimension collapses at criticality")
    println()
    
    ink
end

# ═══════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════

export InkPoint, point_distance, point_speed
export InkStroke, stroke_length, stroke_duration, stroke_speed
export InkImageRepresentation, render_to_image
export InkTextRepresentation, tokenize_to_text
export PenLiftEvent, TracedInk
export pen_down!, trace_point!, pen_lift!
export update_d_2!, effective_dimension, criticality_state
export physicality_status, landauer_cost, is_reversible
export get_image_representation, get_text_representation, combined_tokens
export stroke_trit_sum, is_gf3_balanced, balance_recommendation
export ink_status, demo_traced_ink

end # module InkTracedCriticality
