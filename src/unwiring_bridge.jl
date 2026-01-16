# unwiring_bridge.jl - Bridge to plurigrid/UnwiringDiagrams.jl
#
# Closes ALL loops in the Gay.jl unified ecosystem:
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  UNWIRING BRIDGE: The Loop Closer                                          │
# │                                                                             │
# │  From UnwiringDiagrams.jl (fork of AlgebraicJulia/WiringDiagrams.jl):      │
# │    - Boxes (B) ↔ Players in GayEcosystem                                   │
# │    - Wires (W) ↔ Moment flows between players                              │
# │    - Ports (P) ↔ Input/output channels for play/coplay                     │
# │    - Labels (L) ↔ GF(3) trits {-1, 0, +1}                                  │
# │                                                                             │
# │  Integration Pattern:                                                       │
# │    1. Wire: Players → ConceptTensor → Colors (forward flow)                │
# │    2. Unwire: Colors → Gradients → Players (backward learning)             │
# │    3. Closure: GF(3) conservation across all rewirings                     │
# │                                                                             │
# │  Key Insight (from GAY.md in UnwiringDiagrams.jl):                         │
# │    - Repo Seed: 0x915714e4bef5ae53                                         │
# │    - Repo Color: #ba193b (index 607/1055)                                  │
# │    - Learning Rate: 0.0309                                                  │
# └─────────────────────────────────────────────────────────────────────────────┘

module UnwiringBridge

using Colors
using SplittableRandoms: SplittableRandom, split

# Core Gay.jl imports
using ..Gay: splitmix64, GAY_SEED, GOLDEN, next_color
using ..TernarySplit: SplittableSeed, split_seed, ternary_from_seed, TernaryColor, split_color
using ..GayUnifiedEcosystem: GayEcosystem, ConceptTensor, MorphismType, PlayerSlot
using ..GayUnifiedEcosystem: add_player!, remove_player!, play!, coplay!
using ..GayUnifiedEcosystem: concept_tensor_lookup, moment_flow!, ecosystem_status

export UnwiringRule, UnwiringDiagram, BoxPort, WireSpec
export WorldUnwiringBridge, UnwiringBridgeWorld
export world_unwiring_bridge
export apply_unwiring!, unwire_step!, rewire_balanced!
export wiring_to_ecosystem, ecosystem_to_wiring
export close_all_loops!, verify_loop_closure

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS FROM UnwiringDiagrams.jl GAY.md
# ═══════════════════════════════════════════════════════════════════════════════

const UNWIRING_SEED = 0x915714e4bef5ae53  # From GAY.md
const UNWIRING_COLOR = RGB(0xba/255, 0x19/255, 0x3b/255)  # #ba193b
const UNWIRING_INDEX = 607
const UNWIRING_LEARNING_RATE = 0.0309

# ═══════════════════════════════════════════════════════════════════════════════
# UNWIRING RULE (Learning through constraint release)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    UnwiringRule

A rule for learning through constraint release.
Shifts internal state toward external observation.

From unwiring-arena skill:
- MINUS (-1): Constraint verification (coplay focus)
- ERGODIC (0): Balance/coordination (arena equilibrium)
- PLUS (+1): Generative exploration (play focus)
"""
struct UnwiringRule
    source_gf3::Int8          # Source polarity {-1, 0, +1}
    target_gf3::Int8          # Target polarity
    learning_rate::Float64    # How fast to unwire (default: 0.0309)
    threshold::Float64        # Discrepancy threshold to trigger
    seed::UInt64              # Deterministic seed for this rule
end

function UnwiringRule(source::Int, target::Int; 
                      learning_rate::Float64=UNWIRING_LEARNING_RATE,
                      threshold::Float64=0.1,
                      seed::UInt64=UNWIRING_SEED)
    UnwiringRule(
        Int8(clamp(source, -1, 1)),
        Int8(clamp(target, -1, 1)),
        learning_rate,
        threshold,
        seed
    )
end

"""
    apply_unwiring(rule, internal, external) -> updated_internal

Shift internal state toward external observation.
This is the core learning operation.
"""
function apply_unwiring(rule::UnwiringRule, internal::Float64, external::Float64)
    α = rule.learning_rate
    (1 - α) * internal + α * external
end

# ═══════════════════════════════════════════════════════════════════════════════
# BOX AND WIRE SPECIFICATIONS (from AbstractWiringDiagram)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    BoxPort

A port on a box in the wiring diagram.
Maps to PlayerSlot in GayEcosystem.
"""
struct BoxPort
    box_id::UInt64       # Which box (player)
    port_index::Int      # Which port on that box
    direction::Symbol    # :in or :out
    trit::Int8           # GF(3) label
end

"""
    WireSpec

A wire connecting two ports.
Maps to moment flow in GayEcosystem.
"""
struct WireSpec
    id::UInt64
    source::BoxPort
    target::BoxPort
    color::RGB           # Wire color from Gay.jl
end

# ═══════════════════════════════════════════════════════════════════════════════
# UNWIRING DIAGRAM (Minimal representation compatible with UnwiringDiagrams.jl)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    UnwiringDiagram

A minimal wiring diagram representation compatible with
plurigrid/UnwiringDiagrams.jl structure.

Uses the abstract pattern from abstract_wiring_diagrams.jl:
- B = set of boxes (players)
- W = set of wires (moment flows)
- P = set of ports (I/O channels)
- L = set of labels (GF(3) trits)
"""
mutable struct UnwiringDiagram
    # Core sets
    boxes::Vector{UInt64}       # B: box IDs (player IDs)
    wires::Vector{WireSpec}     # W: wire specifications
    ports::Vector{BoxPort}      # P: all ports
    labels::Vector{Int8}        # L: GF(3) trit labels
    
    # Outer interface
    outer_ports_in::Vector{Int}   # Q_in: outer input port indices
    outer_ports_out::Vector{Int}  # Q_out: outer output port indices
    
    # GF(3) state
    gf3_sum::Int
    
    # Provenance
    seed::UInt64
    color::RGB
end

function UnwiringDiagram(; seed::UInt64=UNWIRING_SEED)
    tc = split_color(seed)
    UnwiringDiagram(
        UInt64[],
        WireSpec[],
        BoxPort[],
        Int8[],
        Int[],
        Int[],
        0,
        seed,
        RGB(tc.L/100, tc.C/100, tc.H/360)
    )
end

# Accessors (matching UnwiringDiagrams.jl API)
nb(d::UnwiringDiagram) = length(d.boxes)      # Number of boxes
nw(d::UnwiringDiagram) = length(d.wires)      # Number of wires
np(d::UnwiringDiagram) = length(d.ports)      # Number of ports
nop(d::UnwiringDiagram) = length(d.outer_ports_in) + length(d.outer_ports_out)  # Outer ports

boxes(d::UnwiringDiagram) = d.boxes
wires(d::UnwiringDiagram) = d.wires

# ═══════════════════════════════════════════════════════════════════════════════
# BIDIRECTIONAL CONVERSION: Ecosystem ↔ Wiring Diagram
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ecosystem_to_wiring(eco::GayEcosystem) -> UnwiringDiagram

Convert a GayEcosystem to an UnwiringDiagram.
Each player becomes a box with input/output ports.
"""
function ecosystem_to_wiring(eco::GayEcosystem)
    diagram = UnwiringDiagram(seed=eco.seed)
    
    # Add boxes for each active player
    for player in eco.players
        if player.active
            push!(diagram.boxes, player.id)
            
            # Each player has in/out ports
            in_port = BoxPort(player.id, 1, :in, player.trit)
            out_port = BoxPort(player.id, 2, :out, player.trit)
            push!(diagram.ports, in_port)
            push!(diagram.ports, out_port)
            push!(diagram.labels, player.trit)
        end
    end
    
    # Connect consecutive players with wires (sequential composition)
    active_boxes = diagram.boxes
    for i in 1:(length(active_boxes)-1)
        src_id = active_boxes[i]
        tgt_id = active_boxes[i+1]
        
        src_port = BoxPort(src_id, 2, :out, diagram.labels[2i-1])
        tgt_port = BoxPort(tgt_id, 1, :in, diagram.labels[2i])
        
        wire_seed = splitmix64(src_id ⊻ tgt_id)
        tc = split_color(wire_seed)
        wire_color = RGB(tc.L/100, tc.C/100, tc.H/360)
        
        wire = WireSpec(wire_seed, src_port, tgt_port, wire_color)
        push!(diagram.wires, wire)
    end
    
    # Set outer ports (first in, last out)
    if !isempty(active_boxes)
        push!(diagram.outer_ports_in, 1)
        push!(diagram.outer_ports_out, length(diagram.ports))
    end
    
    diagram.gf3_sum = sum(diagram.labels)
    diagram
end

"""
    wiring_to_ecosystem(diagram::UnwiringDiagram; n_rounds=0) -> GayEcosystem

Convert an UnwiringDiagram back to a GayEcosystem.
Reconstructs players from boxes.
"""
function wiring_to_ecosystem(diagram::UnwiringDiagram; n_rounds::Int=0)
    eco = GayEcosystem(seed=diagram.seed, n_initial_players=0)
    
    # Reconstruct players from boxes
    for (i, box_id) in enumerate(diagram.boxes)
        player = PlayerSlot(box_id)
        if i <= length(diagram.labels)
            player.trit = diagram.labels[i]
        end
        push!(eco.players, player)
        eco.active_count += 1
        eco.gf3_sum += player.trit
    end
    
    eco
end

# ═══════════════════════════════════════════════════════════════════════════════
# LOOP CLOSURE OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    unwire_step!(diagram, rule) -> (changed_wires, delta_gf3)

Apply an unwiring step: learn by releasing constraints.
Returns count of changed wires and GF(3) delta.
"""
function unwire_step!(diagram::UnwiringDiagram, rule::UnwiringRule)
    changed = 0
    delta = 0
    
    for (i, wire) in enumerate(diagram.wires)
        # Check if wire matches rule's source trit
        if wire.source.trit == rule.source_gf3
            # Check threshold
            discrepancy = abs(Float64(wire.source.trit - wire.target.trit))
            if discrepancy > rule.threshold
                # Apply learning
                new_trit = round(Int8, apply_unwiring(rule, 
                    Float64(wire.source.trit), 
                    Float64(wire.target.trit)))
                new_trit = clamp(new_trit, Int8(-1), Int8(1))
                
                # Update source port
                old_trit = wire.source.trit
                new_source = BoxPort(wire.source.box_id, wire.source.port_index, 
                                     wire.source.direction, new_trit)
                diagram.wires[i] = WireSpec(wire.id, new_source, wire.target, wire.color)
                
                delta += new_trit - old_trit
                changed += 1
            end
        end
    end
    
    diagram.gf3_sum += delta
    (changed, delta)
end

"""
    rewire_balanced!(diagram) -> Int

Rewire the diagram to restore GF(3) conservation.
Returns number of rewiring operations.
"""
function rewire_balanced!(diagram::UnwiringDiagram)
    ops = 0
    
    # Keep rewiring until balanced
    while diagram.gf3_sum % 3 != 0
        # Find a wire to adjust
        for (i, wire) in enumerate(diagram.wires)
            adjustment = -sign(diagram.gf3_sum % 3)
            new_trit = clamp(wire.source.trit + Int8(adjustment), Int8(-1), Int8(1))
            
            if new_trit != wire.source.trit
                new_source = BoxPort(wire.source.box_id, wire.source.port_index,
                                     wire.source.direction, new_trit)
                diagram.wires[i] = WireSpec(wire.id, new_source, wire.target, wire.color)
                diagram.gf3_sum += new_trit - wire.source.trit
                ops += 1
                break
            end
        end
        
        # Safety: prevent infinite loop
        if ops > 100
            break
        end
    end
    
    ops
end

"""
    close_all_loops!(eco::GayEcosystem) -> NamedTuple

The main loop closure operation.
1. Convert ecosystem to wiring diagram
2. Apply unwiring rules
3. Restore GF(3) balance
4. Convert back
5. Verify closure

Returns comprehensive closure report.
"""
function close_all_loops!(eco::GayEcosystem)
    # Step 1: Ecosystem → Diagram
    diagram = ecosystem_to_wiring(eco)
    initial_wires = nw(diagram)
    initial_gf3 = diagram.gf3_sum
    
    # Step 2: Apply unwiring rules (all three trit types)
    rules = [
        UnwiringRule(-1, 0),   # MINUS → ERGODIC
        UnwiringRule(0, 1),    # ERGODIC → PLUS
        UnwiringRule(1, -1),   # PLUS → MINUS (cycle)
    ]
    
    total_changed = 0
    for rule in rules
        changed, _ = unwire_step!(diagram, rule)
        total_changed += changed
    end
    
    # Step 3: Restore balance
    rewire_ops = rewire_balanced!(diagram)
    
    # Step 4: Diagram → Ecosystem
    new_eco = wiring_to_ecosystem(diagram)
    
    # Step 5: Verify closure
    closure_verified = verify_loop_closure(diagram, new_eco)
    
    (
        initial_boxes = nb(diagram),
        initial_wires = initial_wires,
        wires_changed = total_changed,
        rewire_ops = rewire_ops,
        initial_gf3 = initial_gf3,
        final_gf3 = diagram.gf3_sum,
        gf3_conserved = diagram.gf3_sum % 3 == 0,
        closure_verified = closure_verified,
        diagram = diagram,
        ecosystem = new_eco
    )
end

"""
    verify_loop_closure(diagram, eco) -> Bool

Verify that the loop is properly closed:
- GF(3) is conserved
- All wires connect valid ports
- Outer ports exist
"""
function verify_loop_closure(diagram::UnwiringDiagram, eco::GayEcosystem)
    # Check GF(3) conservation
    diagram.gf3_sum % 3 == 0 || return false
    eco.gf3_sum % 3 == 0 || return false
    
    # Check wire validity
    valid_boxes = Set(diagram.boxes)
    for wire in diagram.wires
        wire.source.box_id ∈ valid_boxes || return false
        wire.target.box_id ∈ valid_boxes || return false
    end
    
    # Check outer ports exist
    nop(diagram) > 0 || return false
    
    true
end

# ═══════════════════════════════════════════════════════════════════════════════
# WORLD BUILDER (Following AGENTS.md: world_ prefix)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    UnwiringBridgeWorld

The world state for unwiring bridge operations.
Implements: length, merge, fingerprint (per AGENTS.md).
"""
struct UnwiringBridgeWorld
    diagram::UnwiringDiagram
    ecosystem::GayEcosystem
    closure_report::NamedTuple
    seed::UInt64
    fingerprint::UInt64
end

Base.length(w::UnwiringBridgeWorld) = nb(w.diagram) + nw(w.diagram)

function Base.merge(w1::UnwiringBridgeWorld, w2::UnwiringBridgeWorld)
    # Merge diagrams by concatenating boxes and wires
    merged_diagram = UnwiringDiagram(seed=w1.seed ⊻ w2.seed)
    
    append!(merged_diagram.boxes, w1.diagram.boxes)
    append!(merged_diagram.boxes, w2.diagram.boxes)
    append!(merged_diagram.wires, w1.diagram.wires)
    append!(merged_diagram.wires, w2.diagram.wires)
    append!(merged_diagram.ports, w1.diagram.ports)
    append!(merged_diagram.ports, w2.diagram.ports)
    append!(merged_diagram.labels, w1.diagram.labels)
    append!(merged_diagram.labels, w2.diagram.labels)
    
    merged_diagram.gf3_sum = w1.diagram.gf3_sum + w2.diagram.gf3_sum
    
    # Rewire to balance
    rewire_balanced!(merged_diagram)
    
    # Convert to ecosystem
    merged_eco = wiring_to_ecosystem(merged_diagram)
    
    merged_fp = w1.fingerprint ⊻ w2.fingerprint
    
    UnwiringBridgeWorld(
        merged_diagram,
        merged_eco,
        (merged = true, source_fps = (w1.fingerprint, w2.fingerprint)),
        w1.seed ⊻ w2.seed,
        merged_fp
    )
end

function fingerprint(w::UnwiringBridgeWorld)
    w.fingerprint
end

"""
    world_unwiring_bridge(; seed, n_players, n_rounds) -> UnwiringBridgeWorld

Build an unwiring bridge world, closing all loops.

# Example
```julia
world = world_unwiring_bridge(seed=137508, n_players=9)
# world.closure_report.gf3_conserved == true
```
"""
function world_unwiring_bridge(;
    seed::UInt64 = GAY_SEED,
    n_players::Int = 9,
    n_rounds::Int = 100
)
    # Build initial ecosystem
    eco = GayEcosystem(seed=seed, n_initial_players=n_players)
    
    # Run some rounds to establish state
    for round in 1:min(n_rounds, 10)
        play!(eco, round)
        coplay!(eco, round, Float64(round) / n_rounds)
    end
    
    # Close all loops
    closure_report = close_all_loops!(eco)
    
    # Compute fingerprint
    fp = seed
    for box in closure_report.diagram.boxes
        fp = splitmix64(fp ⊻ box)
    end
    for wire in closure_report.diagram.wires
        fp = splitmix64(fp ⊻ wire.id)
    end
    
    UnwiringBridgeWorld(
        closure_report.diagram,
        closure_report.ecosystem,
        closure_report,
        seed,
        fp
    )
end

end # module UnwiringBridge
