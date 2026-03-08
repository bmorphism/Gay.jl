# traced_criticality.jl
# String Diagram Tracing with Pen-Lift States
#
# The fundamental insight from greenteatree01's critique:
#   Fixed GF(3) regional decomposition conflicts with dynamic whole-brain criticality
#
# Resolution: Model as string diagram tracing where:
#   - Pen DOWN (connected trace) = reversible information flow = subcritical
#   - Pen LIFT (disconnected) = irreversible bit erasure = supercritical/physical
#   - The MOMENT of lifting = criticality transition = τ_c (Landauer threshold)
#
# This unifies:
#   - SARW: Path cannot self-intersect (visited_set = physical trace)
#   - Criticality: Phase transition at T_c where ξ → ∞
#   - Physicality: Information becomes physical when pen lifts (Landauer: kT ln 2)
#   - GF(3): Trit conservation only holds within connected components
#
# String diagram semantics:
#   - Wires = information channels (colored by GF(3) polarity)
#   - Boxes = processes (open games)
#   - Tracing = feedback loops (comonadic structure)
#   - Pen lift = symmetry breaking / measurement / collapse

module TracedCriticality

using Random

# ═══════════════════════════════════════════════════════════════════════════
# CORE TYPES
# ═══════════════════════════════════════════════════════════════════════════

# Import SplitMix64 for deterministic fingerprinting
@inline function sm64(x::UInt64)::UInt64
    z = x + 0x9e3779b97f4a7c15
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    z ⊻ (z >> 31)
end

@enum Trit::Int8 MINUS=-1 ERGODIC=0 PLUS=1
@enum PenState::Int8 DOWN=0 LIFTING=1 UP=2

"""
    Wire

A wire in a string diagram with GF(3) coloring.
Wires carry information between boxes (processes).
"""
struct Wire
    id::UInt64
    trit::Trit
    source::Symbol      # Box or :input
    target::Symbol      # Box or :output
    fingerprint::UInt64
end

"""
    TraceSegment

A continuous segment of pen-down tracing.
Within a segment, GF(3) conservation holds.
Crossing segment boundaries = pen lift = physicality.
"""
struct TraceSegment
    id::Int
    wires::Vector{Wire}
    trit_sum::Int           # Should be 0 mod 3 within segment
    start_time::Float64
    end_time::Float64
    fingerprint::UInt64
end

"""
    PenLiftEvent

The moment when the pen lifts - this is where information becomes physical.
Corresponds to:
  - Landauer erasure (E ≥ kT ln 2)
  - Measurement/collapse in QM
  - Dragon King event in SOC
  - SARW hitting visited set
"""
struct PenLiftEvent
    timestamp::Float64
    from_segment::Int
    to_segment::Int
    erased_bits::Float64    # Information cost of the lift
    d_2::Float64            # Distance to criticality at moment of lift
    reason::Symbol          # :sarw_collision, :dragon_king, :measurement, :tau_exceeded
    fingerprint::UInt64
end

"""
    TracedDiagram

A string diagram with explicit pen-lift tracking.
This is the core data structure unifying SARW, criticality, and physicality.
"""
mutable struct TracedDiagram
    name::Symbol
    segments::Vector{TraceSegment}
    lifts::Vector{PenLiftEvent}
    current_segment::Int
    pen_state::PenState
    visited_set::Set{UInt64}        # SARW: positions we've been (physical trace)
    
    # Criticality parameters (from Sooter et al. 2025)
    d_2::Float64                    # Distance to β=2 fixed point
    tau::Float64                    # Synaptic plasticity timescale
    correlation_length::Float64     # ξ: diverges at criticality
    
    # Tracking
    total_erased_bits::Float64
    dragon_king_count::Int
    seed::UInt64
    fingerprint::UInt64
end

# ═══════════════════════════════════════════════════════════════════════════
# CONSTRUCTORS
# ═══════════════════════════════════════════════════════════════════════════

function Wire(source::Symbol, target::Symbol, trit::Trit; seed::UInt64=UInt64(1069))
    id = sm64(seed ⊻ hash(source) ⊻ sm64(hash(target)))
    fp = sm64(id ⊻ UInt64(Int(trit) + 2))
    Wire(id, trit, source, target, fp)
end

function TraceSegment(id::Int, wires::Vector{Wire}; 
                      start_time::Float64=0.0, seed::UInt64=UInt64(1069))
    trit_sum = sum(Int(w.trit) for w in wires; init=0)
    fp = isempty(wires) ? sm64(seed) : reduce(⊻, w.fingerprint for w in wires)
    TraceSegment(id, wires, trit_sum, start_time, start_time, fp)
end

function TracedDiagram(name::Symbol; seed::UInt64=UInt64(1069), 
                       d_2::Float64=0.5, tau::Float64=100.0)
    fp = sm64(seed ⊻ hash(name))
    initial_segment = TraceSegment(1, Wire[]; seed=fp)
    
    TracedDiagram(
        name,
        [initial_segment],
        PenLiftEvent[],
        1,
        DOWN,
        Set{UInt64}(),
        d_2,
        tau,
        1.0,  # Initial correlation length
        0.0,
        0,
        seed,
        fp
    )
end

# ═══════════════════════════════════════════════════════════════════════════
# CORE OPERATIONS: TRACING WITH PEN LIFT
# ═══════════════════════════════════════════════════════════════════════════

"""
    trace!(diagram, wire) → Bool

Trace a wire in the diagram. Returns true if successful (pen stayed down),
false if a pen lift was required (information became physical).

The pen lifts when:
1. SARW collision: wire position already in visited_set
2. Dragon King: correlation length exceeds threshold  
3. Tau exceeded: time since segment start > τ (plasticity timescale)
4. GF(3) violation: adding wire would break trit conservation
"""
function trace!(diagram::TracedDiagram, wire::Wire)::Bool
    current_time = Float64(length(diagram.segments[diagram.current_segment].wires))
    
    # Check SARW collision
    if wire.fingerprint in diagram.visited_set
        pen_lift!(diagram, :sarw_collision, current_time)
        return false
    end
    
    # Check Dragon King (correlation length explosion)
    if diagram.correlation_length > 100.0 / diagram.d_2
        diagram.dragon_king_count += 1
        pen_lift!(diagram, :dragon_king, current_time)
        return false
    end
    
    # Check tau exceeded
    segment = diagram.segments[diagram.current_segment]
    if current_time - segment.start_time > diagram.tau
        pen_lift!(diagram, :tau_exceeded, current_time)
        return false
    end
    
    # Check GF(3) conservation would be violated
    # (We allow temporary imbalance but lift if sum exceeds ±2)
    projected_sum = segment.trit_sum + Int(wire.trit)
    if abs(projected_sum) > 2
        pen_lift!(diagram, :gf3_overflow, current_time)
        return false
    end
    
    # All checks passed: trace the wire (pen stays down)
    push!(diagram.visited_set, wire.fingerprint)
    push!(segment.wires, wire)
    
    # Update segment's trit sum
    new_segment = TraceSegment(
        segment.id,
        segment.wires,
        start_time=segment.start_time,
        seed=diagram.seed
    )
    diagram.segments[diagram.current_segment] = new_segment
    
    # Update correlation length (grows as we trace)
    diagram.correlation_length *= 1.0 + 0.1 / diagram.d_2
    
    # Update fingerprint
    diagram.fingerprint = sm64(diagram.fingerprint ⊻ wire.fingerprint)
    
    true
end

"""
    pen_lift!(diagram, reason, timestamp)

Execute a pen lift - the moment information becomes physical.
This is the Landauer threshold crossing.
"""
function pen_lift!(diagram::TracedDiagram, reason::Symbol, timestamp::Float64)
    # Calculate erased bits (Landauer: minimum kT ln 2 per bit)
    # More bits erased when further from criticality
    erased_bits = log2(max(1.0, diagram.correlation_length)) * (1.0 + diagram.d_2)
    
    # Record the lift event
    from_segment = diagram.current_segment
    to_segment = from_segment + 1
    
    lift = PenLiftEvent(
        timestamp,
        from_segment,
        to_segment,
        erased_bits,
        diagram.d_2,
        reason,
        sm64(diagram.fingerprint ⊻ UInt64(round(timestamp * 1000)))
    )
    push!(diagram.lifts, lift)
    
    # Update state
    diagram.pen_state = UP
    diagram.total_erased_bits += erased_bits
    
    # Reset correlation length (phase transition occurred)
    diagram.correlation_length = 1.0
    
    # Start new segment
    new_segment = TraceSegment(to_segment, Wire[]; 
                               start_time=timestamp, seed=lift.fingerprint)
    push!(diagram.segments, new_segment)
    diagram.current_segment = to_segment
    
    # Pen goes back down for new segment
    diagram.pen_state = DOWN
    
    lift
end

"""
    pen_down!(diagram)

Explicitly put pen down (start new connected component).
"""
function pen_down!(diagram::TracedDiagram)
    if diagram.pen_state != DOWN
        diagram.pen_state = DOWN
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# CRITICALITY DYNAMICS
# ═══════════════════════════════════════════════════════════════════════════

"""
    update_d_2!(diagram, measurement)

Update distance to criticality based on external measurement.
This is where greenteatree01's critique is addressed:
d_2 is DYNAMIC, not fixed to regional decomposition.

measurement should be in [0, 1]:
  0 = at criticality (β=2 fixed point)
  1 = far from criticality
"""
function update_d_2!(diagram::TracedDiagram, measurement::Float64)
    @assert 0.0 ≤ measurement ≤ 1.0 "d_2 measurement must be in [0,1]"
    
    # Exponential moving average to smooth transitions
    α = 0.3  # Learning rate
    diagram.d_2 = α * measurement + (1 - α) * diagram.d_2
    
    # Correlation length inversely related to d_2
    # At criticality (d_2 → 0), ξ → ∞
    diagram.correlation_length = 1.0 / max(diagram.d_2, 0.01)
end

"""
    criticality_state(diagram) → Symbol

Return current criticality regime:
  :subcritical  - ordered, low entropy, pen tends to stay down
  :critical     - edge of chaos, ξ → ∞, scale-free avalanches
  :supercritical - disordered, high entropy, frequent pen lifts
"""
function criticality_state(diagram::TracedDiagram)::Symbol
    if diagram.d_2 < 0.1
        :critical
    elseif diagram.d_2 < 0.5
        :subcritical
    else
        :supercritical
    end
end

"""
    effective_dimension(diagram) → Int

Effective GF(n) dimension based on criticality.
At criticality, different brain regions synchronize → dimension collapse.

This addresses greenteatree01's insight:
  - Far from criticality: GF(3) holds (3 independent generators)
  - Near criticality: GF(3) → GF(2) (two regions sync)
  - At criticality: GF(3) → GF(1) (global synchronization)
"""
function effective_dimension(diagram::TracedDiagram)::Int
    if diagram.d_2 < 0.05
        1  # At criticality: everything synchronized
    elseif diagram.d_2 < 0.2
        2  # Near criticality: partial sync
    else
        3  # Far from criticality: full GF(3)
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# SARW INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════

"""
    sarw_step!(diagram, direction) → (Wire, Bool)

Take a self-avoiding random walk step in wire space.
Returns the wire and whether it was a valid step (didn't collide).

The visited_set grows monotonically - this is the physical trace,
the irreversible record of the walk.
"""
function sarw_step!(diagram::TracedDiagram, direction::Trit)
    # Generate wire based on current position and direction
    current_pos = length(diagram.visited_set)
    source = Symbol("pos_", current_pos)
    target = Symbol("pos_", current_pos + 1)
    
    wire = Wire(source, target, direction; 
                seed=sm64(diagram.fingerprint ⊻ UInt64(current_pos)))
    
    success = trace!(diagram, wire)
    (wire, success)
end

"""
    sarw_explore!(diagram, n_steps) → Vector{Tuple{Wire, Bool}}

Run n_steps of SARW, returning trace of (wire, success) pairs.
"""
function sarw_explore!(diagram::TracedDiagram, n_steps::Int)
    results = Tuple{Wire, Bool}[]
    
    for i in 1:n_steps
        # Choose direction based on GF(3) balance of current segment
        segment = diagram.segments[diagram.current_segment]
        
        # Prefer direction that maintains balance
        if segment.trit_sum > 0
            direction = MINUS
        elseif segment.trit_sum < 0
            direction = PLUS
        else
            # Balanced: choose based on position parity
            direction = Trit((i % 3) - 1)
        end
        
        wire, success = sarw_step!(diagram, direction)
        push!(results, (wire, success))
    end
    
    results
end

# ═══════════════════════════════════════════════════════════════════════════
# PHYSICALITY THRESHOLD
# ═══════════════════════════════════════════════════════════════════════════

"""
    physicality_status(diagram) → NamedTuple

Comprehensive status of when/whether information has become physical.

Returns:
  - is_physical: Has any irreversible erasure occurred?
  - erased_bits: Total bits erased (Landauer cost)
  - lift_count: Number of pen lifts
  - dragon_kings: Number of SOC breakdown events
  - reversible_fraction: Proportion of trace still reversible
"""
function physicality_status(diagram::TracedDiagram)
    total_wires = sum(length(s.wires) for s in diagram.segments)
    
    (
        is_physical = diagram.total_erased_bits > 0,
        erased_bits = diagram.total_erased_bits,
        lift_count = length(diagram.lifts),
        dragon_kings = diagram.dragon_king_count,
        reversible_fraction = total_wires > 0 ? 
            (total_wires - diagram.total_erased_bits) / total_wires : 1.0,
        current_d_2 = diagram.d_2,
        effective_dim = effective_dimension(diagram),
        criticality = criticality_state(diagram)
    )
end

"""
    is_reversible(diagram) → Bool

Can the entire trace be undone without Landauer cost?
True only if no pen lifts have occurred.
"""
is_reversible(diagram::TracedDiagram) = isempty(diagram.lifts)

"""
    landauer_cost(diagram) → Float64

Total thermodynamic cost of the trace in units of kT ln 2.
"""
landauer_cost(diagram::TracedDiagram) = diagram.total_erased_bits

# ═══════════════════════════════════════════════════════════════════════════
# STRING DIAGRAM COMPOSITION
# ═══════════════════════════════════════════════════════════════════════════

"""
    compose_sequential(d1, d2) → TracedDiagram

Sequential composition: d1 ; d2
The pen lifts between diagrams (information boundary).
"""
function compose_sequential(d1::TracedDiagram, d2::TracedDiagram)
    result = TracedDiagram(
        Symbol(d1.name, :_then_, d2.name);
        seed = sm64(d1.fingerprint ⊻ d2.fingerprint),
        d_2 = (d1.d_2 + d2.d_2) / 2,
        tau = min(d1.tau, d2.tau)
    )
    
    # Copy d1's segments
    for seg in d1.segments
        for wire in seg.wires
            trace!(result, wire)
        end
    end
    
    # Mandatory pen lift at composition boundary
    pen_lift!(result, :composition_boundary, Float64(length(result.visited_set)))
    
    # Copy d2's segments
    for seg in d2.segments
        for wire in seg.wires
            trace!(result, wire)
        end
    end
    
    result
end

"""
    compose_parallel(d1, d2) → TracedDiagram

Parallel composition: d1 ⊗ d2
No pen lift needed - diagrams are independent.
GF(3) conservation holds across the tensor product.
"""
function compose_parallel(d1::TracedDiagram, d2::TracedDiagram)
    result = TracedDiagram(
        Symbol(d1.name, :_par_, d2.name);
        seed = sm64(d1.fingerprint ⊻ sm64(d2.fingerprint)),
        d_2 = min(d1.d_2, d2.d_2),  # Most critical determines behavior
        tau = max(d1.tau, d2.tau)
    )
    
    # Interleave wires from both diagrams (parallel execution)
    d1_wires = vcat([seg.wires for seg in d1.segments]...)
    d2_wires = vcat([seg.wires for seg in d2.segments]...)
    
    i1, i2 = 1, 1
    while i1 ≤ length(d1_wires) || i2 ≤ length(d2_wires)
        if i1 ≤ length(d1_wires)
            trace!(result, d1_wires[i1])
            i1 += 1
        end
        if i2 ≤ length(d2_wires)
            # Relabel d2 wires to avoid collision
            w = d2_wires[i2]
            relabeled = Wire(
                Symbol(w.source, :_par),
                Symbol(w.target, :_par),
                w.trit;
                seed = sm64(w.fingerprint)
            )
            trace!(result, relabeled)
            i2 += 1
        end
    end
    
    result
end

"""
    trace_feedback!(diagram, wire) → TracedDiagram

Traced monoidal structure: feed output back to input.
This is where SARW self-intersection can occur.
"""
function trace_feedback!(diagram::TracedDiagram, wire::Wire)
    # Create feedback wire connecting output to input
    feedback = Wire(wire.target, wire.source, wire.trit; 
                    seed=sm64(wire.fingerprint ⊻ UInt64(0xFEEDBACC)))
    
    # This may cause pen lift due to SARW collision
    trace!(diagram, feedback)
    
    diagram
end

# Operators
⊗(d1::TracedDiagram, d2::TracedDiagram) = compose_parallel(d1, d2)
∘(d1::TracedDiagram, d2::TracedDiagram) = compose_sequential(d2, d1)

# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════

"""
    diagram_status(diagram)

Print comprehensive status of the traced diagram.
"""
function diagram_status(diagram::TracedDiagram)
    phys = physicality_status(diagram)
    
    println("═══════════════════════════════════════════════════════════════")
    println("       TRACED STRING DIAGRAM: $(diagram.name)")
    println("═══════════════════════════════════════════════════════════════")
    println()
    println("  PEN STATE:           $(diagram.pen_state)")
    println("  Current Segment:     $(diagram.current_segment) of $(length(diagram.segments))")
    println()
    println("  ┌─────────────────── SARW STATUS ───────────────────┐")
    println("  │  Visited positions:  $(length(diagram.visited_set))")
    println("  │  Pen lifts:          $(length(diagram.lifts))")
    println("  │  Is reversible:      $(is_reversible(diagram))")
    println("  └───────────────────────────────────────────────────┘")
    println()
    println("  ┌─────────────────── CRITICALITY ───────────────────┐")
    println("  │  d_2 (distance):     $(round(diagram.d_2, digits=3))")
    println("  │  State:              $(phys.criticality)")
    println("  │  Effective dim:      GF($(phys.effective_dim))")
    println("  │  Correlation ξ:      $(round(diagram.correlation_length, digits=2))")
    println("  │  Dragon Kings:       $(diagram.dragon_king_count)")
    println("  └───────────────────────────────────────────────────┘")
    println()
    println("  ┌─────────────────── PHYSICALITY ───────────────────┐")
    println("  │  Information physical: $(phys.is_physical)")
    println("  │  Landauer cost:        $(round(phys.erased_bits, digits=2)) bits")
    println("  │  Reversible fraction:  $(round(100*phys.reversible_fraction, digits=1))%")
    println("  └───────────────────────────────────────────────────┘")
    println()
    
    # Show lift events
    if !isempty(diagram.lifts)
        println("  PEN LIFT EVENTS:")
        for (i, lift) in enumerate(diagram.lifts)
            println("    $i. t=$(round(lift.timestamp, digits=1)): $(lift.reason) " *
                    "($(round(lift.erased_bits, digits=2)) bits, d_2=$(round(lift.d_2, digits=3)))")
        end
        println()
    end
    
    # Show segment GF(3) balance
    println("  SEGMENT GF(3) BALANCE:")
    for seg in diagram.segments
        balance = seg.trit_sum % 3
        status = balance == 0 ? "✓ balanced" : "⚠ sum=$balance"
        println("    Segment $(seg.id): $(length(seg.wires)) wires, $status")
    end
    
    println("═══════════════════════════════════════════════════════════════")
end

"""
    to_mermaid(diagram) → String

Generate Mermaid diagram showing trace with pen lifts.
"""
function to_mermaid(diagram::TracedDiagram)::String
    lines = ["```mermaid", "flowchart LR"]
    
    # Style definitions
    push!(lines, "    classDef minus fill:#ff6b6b,stroke:#c92a2a")
    push!(lines, "    classDef ergodic fill:#69db7c,stroke:#2f9e44")
    push!(lines, "    classDef plus fill:#74c0fc,stroke:#1971c2")
    push!(lines, "    classDef lift fill:#ffd43b,stroke:#f59f00,stroke-width:3px")
    
    for (seg_idx, segment) in enumerate(diagram.segments)
        push!(lines, "    subgraph seg$(seg_idx)[\"Segment $(seg_idx)\"]")
        
        for (i, wire) in enumerate(segment.wires)
            wire_id = "s$(seg_idx)w$(i)"
            trit_class = wire.trit == MINUS ? "minus" : 
                         wire.trit == ERGODIC ? "ergodic" : "plus"
            push!(lines, "        $(wire_id)((\"$(wire.trit)\")):::$(trit_class)")
            
            if i > 1
                prev_id = "s$(seg_idx)w$(i-1)"
                push!(lines, "        $(prev_id) --> $(wire_id)")
            end
        end
        
        push!(lines, "    end")
    end
    
    # Show pen lifts as dashed lines between segments
    for lift in diagram.lifts
        from_seg = "seg$(lift.from_segment)"
        to_seg = "seg$(lift.to_segment)"
        push!(lines, "    $(from_seg) -.->|\"✂ $(lift.reason)\"| $(to_seg)")
    end
    
    push!(lines, "```")
    join(lines, "\n")
end

# ═══════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════

export Trit, MINUS, ERGODIC, PLUS
export PenState, DOWN, LIFTING, UP
export Wire, TraceSegment, PenLiftEvent, TracedDiagram
export trace!, pen_lift!, pen_down!
export update_d_2!, criticality_state, effective_dimension
export sarw_step!, sarw_explore!
export physicality_status, is_reversible, landauer_cost
export compose_sequential, compose_parallel, trace_feedback!, ⊗, ∘
export diagram_status, to_mermaid

end # module TracedCriticality
