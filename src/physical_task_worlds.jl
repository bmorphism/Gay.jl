# Physical Task World Model
#
# Extends KripkeWorlds with embodied BCI task semantics:
#   - Worlds = physical task states (body + objects + tool)
#   - Accessibility = motor primitives (reach, grasp, release, move)
#   - Propositions = affordances (graspable, reachable, stable, completed)
#   - □φ = "φ holds regardless of which action you take" (physical necessity)
#   - ◇φ = "there exists an action sequence reaching φ" (physical possibility)
#   - nearest_necessary_neighbor = closest state where success is inevitable
#
# The embodied departure from abstract Kripke:
#   KripkeWorlds: accessibility is trit-algebraic (GF(3) structure)
#   PhysicalTaskWorlds: accessibility is biomechanical (what the body can do)
#
# BCI bridge:
#   Motor imagery (C3/C4 beta desync) → action selection → efference copy
#   Proprioceptive feedback → reafference → state update
#   Mismatch (exafference) → affordance re-evaluation → replan
#
# References:
#   - Gibson (1979): The Ecological Approach to Visual Perception
#   - Cisek (2007): Cortical mechanisms of action selection (affordance competition)
#   - Wolpert & Ghahramani (2000): Computational motor control
#   - SENSORIUM spec (UNIVERSAL_BCI_RECEIVER.md)
#   - Von Holst (1950): reafference principle → reafference.jl

module PhysicalTaskWorlds

using ..Gay: GAY_SEED, splitmix64, hash_color, color_at, GayRNG, next_color
using ..KripkeWorlds: Trit, MINUS, ERGODIC, PLUS, trit_add, trit_from_name
using ..KripkeWorlds: ModalProposition, Atomic, Box, Diamond, Negation, Conjunction
using ..KripkeWorlds: box, diamond
using Colors
using LinearAlgebra

export PhysicalState, TaskFrame
export BodyState, ObjectState, ToolState
export MotorPrimitive, REACH, GRASP, RELEASE, MOVE, WAIT
export physical_accessible, action_cost, executable
export affordance_at, affordances
export plan_to_necessity, physical_necessity_landscape
export motor_imagery_trit, efference_copy, reafference_check
export demo_physical_task

# ═══════════════════════════════════════════════════════════════════════════
# Body State: proprioceptive configuration
# ═══════════════════════════════════════════════════════════════════════════

"""
    BodyState

Minimal proprioceptive state for reach-grasp tasks.
Position is end-effector in workspace (3D).
Grip is normalized [0=open, 1=closed].
Energy is metabolic cost remaining (fatigue model).
"""
struct BodyState
    position::NTuple{3, Float64}    # end-effector xyz (meters)
    grip::Float64                   # 0.0 = open, 1.0 = closed
    energy::Float64                 # 0.0 = exhausted, 1.0 = fresh
end

BodyState() = BodyState((0.0, 0.0, 0.0), 0.0, 1.0)

function body_distance(a::BodyState, b::BodyState)
    sqrt(sum((x - y)^2 for (x, y) in zip(a.position, b.position)))
end

# ═══════════════════════════════════════════════════════════════════════════
# Object State: what's in the workspace
# ═══════════════════════════════════════════════════════════════════════════

"""
    ObjectState

A graspable object in the workspace.
"""
struct ObjectState
    name::Symbol
    position::NTuple{3, Float64}
    graspable::Bool                 # size/shape permits grasp
    held::Bool                      # currently in grip
    mass::Float64                   # kg (affects energy cost)
end

ObjectState(name::Symbol, pos::NTuple{3, Float64}; mass=0.1) =
    ObjectState(name, pos, true, false, mass)

# ═══════════════════════════════════════════════════════════════════════════
# Tool State: augmented capabilities
# ═══════════════════════════════════════════════════════════════════════════

"""
    ToolState

A tool modifies the action space -- extends reach, reduces grip force needed,
or enables new affordances (e.g., a screwdriver affords :fastenable).
"""
struct ToolState
    name::Symbol
    held::Bool
    reach_extension::Float64        # meters added to reach radius
    affordances::Set{Symbol}        # new affordances tool grants
end

ToolState(name::Symbol; reach_ext=0.0, affs=Symbol[]) =
    ToolState(name, false, reach_ext, Set{Symbol}(affs))

# ═══════════════════════════════════════════════════════════════════════════
# Motor Primitives: the actions that define accessibility
# ═══════════════════════════════════════════════════════════════════════════

"""
    MotorPrimitive

Elementary motor action. Each primitive:
- Has a biomechanical cost (energy depletion)
- Maps to a BCI signal pattern (mu/beta rhythm modulation)
- Transforms one PhysicalState into another (if preconditions met)
"""
@enum MotorPrimitive begin
    REACH    # move hand toward target position
    GRASP    # close grip on object at current position
    RELEASE  # open grip, drop held object
    MOVE     # transport held object to new position
    WAIT     # hold current state (no energy cost, time passes)
end

const PRIMITIVE_COSTS = Dict(
    REACH   => 0.05,   # moderate: arm movement
    GRASP   => 0.02,   # low: finger closure
    RELEASE => 0.01,   # minimal: finger opening
    MOVE    => 0.08,   # high: loaded transport
    WAIT    => 0.00,   # free: isometric hold
)

# BCI channel mapping (10-20 system):
#   C3 = left motor cortex  → right hand actions
#   C4 = right motor cortex → left hand actions
#   Cz = supplementary motor area → bilateral/planning
const PRIMITIVE_CHANNELS = Dict(
    REACH   => :C3,    # contralateral hand reach
    GRASP   => :C3,    # contralateral hand grasp
    RELEASE => :C3,    # contralateral hand release
    MOVE    => :Cz,    # bilateral coordination
    WAIT    => :Cz,    # planning/inhibition
)

# Beta desynchronization thresholds (% ERD for detection)
const PRIMITIVE_BETA_THRESHOLD = Dict(
    REACH   => 0.30,   # 30% ERD in mu/beta
    GRASP   => 0.25,   # 25% ERD
    RELEASE => 0.15,   # 15% ERD (less effortful)
    MOVE    => 0.35,   # 35% ERD (complex)
    WAIT    => 0.00,   # no desync (resting state)
)

# ═══════════════════════════════════════════════════════════════════════════
# Physical State: a world in the task frame
# ═══════════════════════════════════════════════════════════════════════════

"""
    PhysicalState

A world in the physical task Kripke frame. Extends KripkeWorlds.World:
- `body`: proprioceptive configuration
- `objects`: all objects in workspace
- `tool`: optional held tool (modifies affordances)
- `trit`: GF(3) value from motor channel assignment
- `color`: deterministic from seed + state hash
"""
struct PhysicalState
    name::Symbol                    # state identifier
    index::Int
    body::BodyState
    objects::Vector{ObjectState}
    tool::Union{ToolState, Nothing}
    trit::Trit
    color::NTuple{3, Float64}
    seed::UInt64
    propositions::Set{Symbol}       # affordances true at this state
end

function PhysicalState(name::Symbol, index::Int, body::BodyState,
                       objects::Vector{ObjectState};
                       tool::Union{ToolState, Nothing}=nothing,
                       seed::UInt64=GAY_SEED,
                       extra_props::Vector{Symbol}=Symbol[])
    trit = motor_imagery_trit(body, objects)
    s = splitmix64(seed ⊻ UInt64(index) ⊻ hash(body.position))
    gr = GayRNG(s)
    c = next_color(gr)
    rgb = (Float64(red(c)), Float64(green(c)), Float64(blue(c)))
    props = Set{Symbol}(extra_props)
    union!(props, compute_affordances(body, objects, tool))
    PhysicalState(name, index, body, objects, tool, trit, rgb, s, props)
end

# ═══════════════════════════════════════════════════════════════════════════
# Affordance Computation (Gibson meets GF(3))
# ═══════════════════════════════════════════════════════════════════════════

const REACH_RADIUS = 0.6  # meters, human arm reach

"""
    compute_affordances(body, objects, tool) -> Set{Symbol}

Evaluate Gibsonian affordances at a physical state.
An affordance is a relation between organism capabilities and
environmental properties -- not a property of either alone.
"""
function compute_affordances(body::BodyState, objects::Vector{ObjectState},
                              tool::Union{ToolState, Nothing})
    affs = Set{Symbol}()
    reach = REACH_RADIUS + (tool !== nothing ? tool.reach_extension : 0.0)

    for obj in objects
        d = sqrt(sum((x - y)^2 for (x, y) in zip(body.position, obj.position)))
        if d <= reach
            push!(affs, :reachable)
            if obj.graspable && body.grip < 0.5
                push!(affs, :graspable)
            end
        end
        if obj.held
            push!(affs, :holding)
            push!(affs, :movable)
        end
    end

    if body.energy > 0.1
        push!(affs, :capable)
    end
    if body.energy < 0.2
        push!(affs, :fatigued)
    end
    if body.grip > 0.5 && isempty(filter(o -> o.held, objects))
        push!(affs, :grip_empty)  # closed on nothing
    end
    if all(!o.held for o in objects) && body.grip < 0.5
        push!(affs, :ready)  # open hand, nothing held, ready to act
    end

    if tool !== nothing && tool.held
        union!(affs, tool.affordances)
    end

    affs
end

"""
    motor_imagery_trit(body, objects) -> Trit

Derive GF(3) trit from motor cortex channel dominance.
This bridges BCI signal to trit structure:
- C3 dominant (right hand active, object interaction) → PLUS (generative)
- C4 dominant (left hand, stabilization) → MINUS (verification)
- Cz dominant (bilateral planning, waiting) → ERGODIC (coordination)
"""
function motor_imagery_trit(body::BodyState, objects::Vector{ObjectState})
    any_held = any(o.held for o in objects)
    any_reachable = any(
        sqrt(sum((x - y)^2 for (x, y) in zip(body.position, o.position))) <= REACH_RADIUS
        for o in objects
    )

    if any_held
        return PLUS      # actively manipulating: C3 dominant, generative
    elseif any_reachable && body.grip < 0.5
        return MINUS     # pre-grasp verification: C4/C3 balanced
    else
        return ERGODIC   # planning/transit: Cz dominant
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# Task Frame: physical Kripke frame with action-based accessibility
# ═══════════════════════════════════════════════════════════════════════════

"""
    TaskFrame

A Kripke frame where accessibility = motor primitive executability.
Two states are accessible iff a single motor primitive transforms one into the other.
"""
struct TaskFrame
    states::Vector{PhysicalState}
    transitions::Dict{Tuple{Int, Int}, MotorPrimitive}  # (from_idx, to_idx) → action
    reach_radius::Float64
end

function TaskFrame(states::Vector{PhysicalState}; reach_radius=REACH_RADIUS)
    transitions = Dict{Tuple{Int, Int}, MotorPrimitive}()
    for (i, s1) in enumerate(states)
        for (j, s2) in enumerate(states)
            i == j && continue
            action = infer_transition(s1, s2)
            if action !== nothing
                transitions[(i, j)] = action
            end
        end
    end
    TaskFrame(states, transitions, reach_radius)
end

"""
    infer_transition(s1, s2) -> Union{MotorPrimitive, Nothing}

Determine which motor primitive, if any, transforms s1 into s2.
Returns nothing if no single primitive connects them.
"""
function infer_transition(s1::PhysicalState, s2::PhysicalState)
    pos_changed = s1.body.position != s2.body.position
    grip_changed = abs(s1.body.grip - s2.body.grip) > 0.01
    obj_pos_changed = any(
        o1.position != o2.position
        for (o1, o2) in zip(s1.objects, s2.objects)
    )
    held_changed = any(
        o1.held != o2.held
        for (o1, o2) in zip(s1.objects, s2.objects)
    )

    if !pos_changed && !grip_changed && !obj_pos_changed && !held_changed
        return WAIT
    elseif pos_changed && !grip_changed && !held_changed && !obj_pos_changed
        return REACH     # only hand moved, nothing grasped/released
    elseif !pos_changed && grip_changed && held_changed
        newly_held = any(
            !o1.held && o2.held
            for (o1, o2) in zip(s1.objects, s2.objects)
        )
        return newly_held ? GRASP : RELEASE
    elseif pos_changed && obj_pos_changed && any(o.held for o in s2.objects)
        return MOVE      # hand moved AND held object moved with it
    end

    return nothing  # no single primitive explains this transition
end

"""
    physical_accessible(frame, s1, s2) -> Bool

Is s2 reachable from s1 via a single motor primitive?
"""
function physical_accessible(frame::TaskFrame, s1::PhysicalState, s2::PhysicalState)
    haskey(frame.transitions, (s1.index, s2.index))
end

"""
    action_cost(frame, s1, s2) -> Float64

Energy cost of the transition from s1 to s2.
Returns Inf if no transition exists.
"""
function action_cost(frame::TaskFrame, s1::PhysicalState, s2::PhysicalState)
    key = (s1.index, s2.index)
    haskey(frame.transitions, key) || return Inf
    action = frame.transitions[key]
    base = PRIMITIVE_COSTS[action]
    # Scale by distance for REACH/MOVE
    if action == REACH || action == MOVE
        d = body_distance(s1.body, s2.body)
        return base * (1.0 + d / REACH_RADIUS)
    end
    base
end

"""
    executable(frame, s, action) -> Bool

Can the given action be executed from state s?
Checks biomechanical preconditions.
"""
function executable(frame::TaskFrame, s::PhysicalState, action::MotorPrimitive)
    s.body.energy < PRIMITIVE_COSTS[action] && return false
    if action == GRASP
        return :graspable in s.propositions
    elseif action == RELEASE
        return :holding in s.propositions
    elseif action == MOVE
        return :holding in s.propositions
    end
    true
end

# ═══════════════════════════════════════════════════════════════════════════
# Modal Operations on Physical Frame
# ═══════════════════════════════════════════════════════════════════════════

"""
    affordance_at(frame, state, prop) -> Bool

Evaluate a modal proposition at a physical state.
Reuses KripkeWorlds' ModalProposition types.
"""
function affordance_at(frame::TaskFrame, s::PhysicalState, p::Atomic)
    p.name in s.propositions
end

function affordance_at(frame::TaskFrame, s::PhysicalState, p::Box)
    neighbors = [s2 for s2 in frame.states if physical_accessible(frame, s, s2)]
    isempty(neighbors) && return true  # vacuously true
    all(affordance_at(frame, s2, p.inner) for s2 in neighbors)
end

function affordance_at(frame::TaskFrame, s::PhysicalState, p::Diamond)
    neighbors = [s2 for s2 in frame.states if physical_accessible(frame, s, s2)]
    any(affordance_at(frame, s2, p.inner) for s2 in neighbors)
end

function affordance_at(frame::TaskFrame, s::PhysicalState, p::Negation)
    !affordance_at(frame, s, p.inner)
end

function affordance_at(frame::TaskFrame, s::PhysicalState, p::Conjunction)
    affordance_at(frame, s, p.left) && affordance_at(frame, s, p.right)
end

"""
    affordances(frame, state) -> Set{Symbol}

All atomic affordances true at this state.
"""
affordances(frame::TaskFrame, s::PhysicalState) = s.propositions

# ═══════════════════════════════════════════════════════════════════════════
# Physical Nearest Necessary Neighbor (task planning as modal search)
# ═══════════════════════════════════════════════════════════════════════════

"""
    plan_to_necessity(frame, start, goal_prop) -> Vector{Tuple{PhysicalState, MotorPrimitive}}

Find the shortest action sequence from `start` to a state where □goal_prop
holds (the goal is necessary = no action from that state can undo it).

This is task planning as modal search:
  "What is the shortest path to a state where success is inevitable?"

Returns the plan as a sequence of (state, action) pairs.
"""
function plan_to_necessity(frame::TaskFrame, start::PhysicalState,
                            goal_prop::ModalProposition)
    visited = Set{Int}()
    queue = [(start, Tuple{PhysicalState, MotorPrimitive}[])]

    while !isempty(queue)
        current, path = popfirst!(queue)
        current.index in visited && continue
        push!(visited, current.index)

        if affordance_at(frame, current, box(goal_prop))
            return path
        end

        for s2 in frame.states
            s2.index in visited && continue
            key = (current.index, s2.index)
            if haskey(frame.transitions, key)
                action = frame.transitions[key]
                if executable(frame, current, action)
                    new_path = vcat(path, [(s2, action)])
                    push!(queue, (s2, new_path))
                end
            end
        end
    end

    return Tuple{PhysicalState, MotorPrimitive}[]  # no plan found
end

"""
    physical_necessity_landscape(frame, prop) -> Dict{Symbol, Int}

For each physical state, compute distance to nearest state where □prop holds.
This is the "effort landscape" for achieving a goal inevitably.
"""
function physical_necessity_landscape(frame::TaskFrame, prop::ModalProposition)
    result = Dict{Symbol, Int}()
    for s in frame.states
        plan = plan_to_necessity(frame, s, prop)
        result[s.name] = isempty(plan) ?
            (affordance_at(frame, s, box(prop)) ? 0 : typemax(Int)) :
            length(plan)
    end
    result
end

# ═══════════════════════════════════════════════════════════════════════════
# BCI Bridge: Motor Imagery → Action Selection → Reafference
# ═══════════════════════════════════════════════════════════════════════════

"""
    EfferenceCopy

When BCI decodes a motor intention, an efference copy predicts
the sensory consequence (proprioceptive + visual).
"""
struct EfferenceCopy
    intended_action::MotorPrimitive
    predicted_state::PhysicalState
    beta_desync::Float64            # measured ERD percentage
    channel::Symbol                 # C3, C4, or Cz
    timestamp::Float64              # seconds
end

"""
    efference_copy(frame, current, action) -> Union{EfferenceCopy, Nothing}

Generate efference copy for an intended action from current state.
Returns nothing if action is not executable.
"""
function efference_copy(frame::TaskFrame, current::PhysicalState,
                         action::MotorPrimitive; timestamp::Float64=0.0)
    !executable(frame, current, action) && return nothing

    targets = [(s2, frame.transitions[(current.index, s2.index)])
               for s2 in frame.states
               if haskey(frame.transitions, (current.index, s2.index)) &&
                  frame.transitions[(current.index, s2.index)] == action]

    isempty(targets) && return nothing
    predicted, _ = first(targets)

    EfferenceCopy(
        action,
        predicted,
        PRIMITIVE_BETA_THRESHOLD[action],
        PRIMITIVE_CHANNELS[action],
        timestamp
    )
end

"""
    ReafferenceResult

Outcome of comparing efference copy (prediction) to actual sensory feedback.
- match: reafference (self-generated, expected)
- mismatch: exafference (external perturbation, unexpected)
"""
struct ReafferenceResult
    match::Bool
    prediction_error::Float64       # distance between predicted and actual
    efference::EfferenceCopy
    actual_state::PhysicalState
    exafference::Bool               # true if mismatch exceeds threshold
end

"""
    reafference_check(efference, actual; threshold=0.1) -> ReafferenceResult

Compare predicted state (from efference copy) to actual state (from sensors).
This closes the motor control loop:
  intention → efference copy → action → sensation → comparison → update
"""
function reafference_check(ef::EfferenceCopy, actual::PhysicalState;
                            threshold::Float64=0.1)
    error = body_distance(ef.predicted_state.body, actual.body)
    is_match = error < threshold
    ReafferenceResult(is_match, error, ef, actual, !is_match)
end

# ═══════════════════════════════════════════════════════════════════════════
# Demo: Pick-and-Place Task
# ═══════════════════════════════════════════════════════════════════════════

"""
    demo_physical_task()

Demonstrate a reach-grasp-move-release task as a physical Kripke frame.

Five states of a pick-and-place:
  1. :rest       — hand at home, object on table
  2. :reached    — hand at object, open grip
  3. :grasped    — hand at object, closed grip, holding
  4. :moved      — hand at target, holding object
  5. :placed     — hand at target, object released at target

Accessibility = motor primitives. Modal propositions = affordances.
Plan = BFS to □completed (state where completion is necessary).
"""
function demo_physical_task()
    println("Physical Task World Model")
    println("=" ^ 60)
    println("Embodied Kripke frame: worlds = task states, R = motor primitives\n")

    obj_start = (0.3, 0.0, 0.0)
    obj_target = (0.0, 0.3, 0.0)
    home = (0.0, 0.0, 0.0)

    obj_on_table = ObjectState(:block, obj_start)
    obj_held = ObjectState(:block, obj_start, true, true, 0.1)
    obj_at_target_held = ObjectState(:block, obj_target, true, true, 0.1)
    obj_at_target = ObjectState(:block, obj_target, true, false, 0.1)

    states = PhysicalState[
        PhysicalState(:rest, 1,
            BodyState(home, 0.0, 1.0),
            [obj_on_table]),
        PhysicalState(:reached, 2,
            BodyState(obj_start, 0.0, 0.95),
            [obj_on_table]),
        PhysicalState(:grasped, 3,
            BodyState(obj_start, 1.0, 0.93),
            [obj_held]),
        PhysicalState(:moved, 4,
            BodyState(obj_target, 1.0, 0.85),
            [obj_at_target_held]),
        PhysicalState(:placed, 5,
            BodyState(obj_target, 0.0, 0.84),
            [obj_at_target];
            extra_props=[:completed]),
    ]

    frame = TaskFrame(states)

    # Show states with trits and affordances
    println("Task states:")
    for s in states
        aff_str = join(sort(collect(s.propositions)), ", ")
        r, g, b = s.color
        println("  :$(s.name) [$(Int(s.trit))] energy=$(s.body.energy) " *
                "grip=$(s.body.grip) → {$(aff_str)}")
    end

    # Show transitions
    println("\nTransitions (motor primitives):")
    for ((i, j), action) in sort(collect(frame.transitions), by=first)
        s1 = frame.states[i]
        s2 = frame.states[j]
        cost = action_cost(frame, s1, s2)
        ch = PRIMITIVE_CHANNELS[action]
        println("  :$(s1.name) →[$(action) ch=$(ch) cost=$(round(cost, digits=3))]→ :$(s2.name)")
    end

    # Modal evaluation
    println("\nModal affordances:")
    completed = Atomic(:completed)
    holding = Atomic(:holding)

    for s in states
        has_completed = affordance_at(frame, s, completed)
        box_completed = affordance_at(frame, s, box(completed))
        dia_completed = affordance_at(frame, s, diamond(completed))
        println("  :$(s.name): completed=$(has_completed), □completed=$(box_completed), ◇completed=$(dia_completed)")
    end

    # Check: is :placed a necessary completion state?
    placed = states[5]
    println("\n:placed is a fixed point: □completed = $(affordance_at(frame, placed, box(completed)))")

    # Plan from rest to necessity
    println("\nPlan (rest → □completed):")
    plan = plan_to_necessity(frame, states[1], completed)
    if isempty(plan) && affordance_at(frame, states[1], box(completed))
        println("  Already at □completed!")
    elseif isempty(plan)
        println("  No plan found (□completed unreachable)")
    else
        for (i, (s, action)) in enumerate(plan)
            ch = PRIMITIVE_CHANNELS[action]
            println("  step $(i): $(action) → :$(s.name) [beta_desync=$(PRIMITIVE_BETA_THRESHOLD[action]), ch=$(ch)]")
        end
    end

    # Necessity landscape
    println("\nNecessity landscape (distance to □completed):")
    landscape = physical_necessity_landscape(frame, completed)
    for s in states
        d = landscape[s.name]
        d_str = d == typemax(Int) ? "∞" : string(d)
        println("  :$(s.name) → $(d_str) steps")
    end

    # Efference copy demo
    println("\nEfference copy (BCI motor control loop):")
    ef = efference_copy(frame, states[1], REACH; timestamp=0.0)
    if ef !== nothing
        println("  Intent: $(ef.intended_action) from :rest")
        println("  Predicted: :$(ef.predicted_state.name)")
        println("  Channel: $(ef.channel), beta_desync: $(ef.beta_desync)")

        # Simulate correct execution
        result = reafference_check(ef, states[2])
        println("  Reafference: match=$(result.match), error=$(round(result.prediction_error, digits=4))")
        println("  Exafference (external perturbation): $(result.exafference)")
    end

    # Trit conservation across task
    trit_sum = sum(Int(s.trit) for s in states)
    println("\nTrit conservation: sum = $(trit_sum), mod 3 = $(mod(trit_sum + 3, 3))")
    println("  rest=$(Int(states[1].trit)), reached=$(Int(states[2].trit)), " *
            "grasped=$(Int(states[3].trit)), moved=$(Int(states[4].trit)), " *
            "placed=$(Int(states[5].trit))")

    frame
end

# ═══════════════════════════════════════════════════════════════════════════
# Rectangular Workspace Grid
# ═══════════════════════════════════════════════════════════════════════════
#
# A 2D grid discretizes the workspace into cells. Each cell (row, col) maps
# to a 3D position. The hand moves between adjacent cells (4-connected).
# Objects occupy cells. Grasp/release happen at the hand's current cell.
#
# This gives the task frame a spatial structure:
#   - Grid adjacency = REACH accessibility
#   - Object co-location = GRASP precondition
#   - Trit assignment per cell = motor cortex channel (spatial laterality)
#
# The rectangle IS the Kripke frame's topology: rows x cols worlds,
# 4-connectivity accessibility, affordances from object placement.

export WorkspaceGrid, GridCell, grid_frame, demo_rectangular_workspace

"""
    GridCell

A cell in the rectangular workspace. Holds position and optional object.
"""
struct GridCell
    row::Int
    col::Int
    position::NTuple{3, Float64}   # 3D workspace coords (z=0 for tabletop)
    object::Union{Symbol, Nothing} # object name if present, nothing if empty
end

"""
    WorkspaceGrid

Rectangular grid of cells. cell_size controls spacing (meters).
Objects are placed on named cells. The hand starts at (1,1).
"""
struct WorkspaceGrid
    rows::Int
    cols::Int
    cell_size::Float64
    cells::Matrix{GridCell}
    objects::Dict{Symbol, Tuple{Int, Int}}  # object name → (row, col)
    target::Dict{Symbol, Tuple{Int, Int}}   # object name → target (row, col)
end

function WorkspaceGrid(rows::Int, cols::Int;
                       cell_size::Float64=0.15,
                       objects::Dict{Symbol, Tuple{Int, Int}}=Dict{Symbol, Tuple{Int, Int}}(),
                       targets::Dict{Symbol, Tuple{Int, Int}}=Dict{Symbol, Tuple{Int, Int}}())
    cells = Matrix{GridCell}(undef, rows, cols)
    for r in 1:rows, c in 1:cols
        pos = ((c - 1) * cell_size, (r - 1) * cell_size, 0.0)
        obj = nothing
        for (name, (or, oc)) in objects
            if or == r && oc == c
                obj = name
            end
        end
        cells[r, c] = GridCell(r, c, pos, obj)
    end
    WorkspaceGrid(rows, cols, cell_size, cells, objects, targets)
end

"""
    grid_neighbors(grid, r, c) -> Vector{Tuple{Int,Int}}

4-connected neighbors of cell (r,c).
"""
function grid_neighbors(grid::WorkspaceGrid, r::Int, c::Int)
    ns = Tuple{Int,Int}[]
    r > 1            && push!(ns, (r-1, c))
    r < grid.rows    && push!(ns, (r+1, c))
    c > 1            && push!(ns, (r, c-1))
    c < grid.cols    && push!(ns, (r, c+1))
    ns
end

"""
    grid_trit(r, c) -> Trit

Assign trit from grid position:
  - Left columns (c <= cols/3): MINUS (C4 right motor cortex, left workspace)
  - Center columns: ERGODIC (Cz, midline)
  - Right columns (c > 2*cols/3): PLUS (C3 left motor cortex, right workspace)

This is the somatotopic map: spatial laterality → motor cortex → trit.
"""
function grid_trit(r::Int, c::Int, cols::Int)
    third = cols / 3.0
    if c <= third
        MINUS
    elseif c > 2 * third
        PLUS
    else
        ERGODIC
    end
end

"""
    GridTaskState

Compact representation of hand position + which objects are held/placed
on the grid. Used to generate PhysicalStates.
"""
struct GridTaskState
    hand_r::Int
    hand_c::Int
    held::Union{Symbol, Nothing}
    placed::Dict{Symbol, Tuple{Int,Int}}  # objects currently on grid
end

"""
    grid_frame(grid; seed=GAY_SEED) -> TaskFrame

Build a TaskFrame from a WorkspaceGrid. States are generated by enumerating
meaningful configurations:
  1. Hand at each cell, nothing held, objects at start positions
  2. Hand at object cell, holding object (post-grasp)
  3. Hand at each cell, holding object (transport)
  4. Hand at target cell, object placed at target (post-release)

Transitions follow grid adjacency for REACH/MOVE, co-location for GRASP/RELEASE.
"""
function grid_frame(grid::WorkspaceGrid; seed::UInt64=GAY_SEED)
    states = PhysicalState[]
    idx = 0

    obj_names = collect(keys(grid.objects))
    isempty(obj_names) && error("Grid needs at least one object")

    # For tractability with multiple objects, enumerate single-object scenarios
    # then compose. Here: enumerate all hand positions x held/not-held for first object.
    obj = first(obj_names)
    obj_r, obj_c = grid.objects[obj]
    tgt_r, tgt_c = get(grid.target, obj, (grid.rows, grid.cols))

    # Phase 1: hand moving to object (nothing held)
    for r in 1:grid.rows, c in 1:grid.cols
        idx += 1
        pos = grid.cells[r, c].position
        d_to_obj = abs(r - obj_r) + abs(c - obj_c)  # manhattan
        energy = 1.0 - 0.01 * d_to_obj
        obj_pos = grid.cells[obj_r, obj_c].position
        objs = [ObjectState(obj, obj_pos; mass=0.1)]
        name = Symbol("h$(r)_$(c)")
        extra = Symbol[]
        if r == tgt_r && c == tgt_c
            # Hand at target but object still at start — not completed
        end
        push!(states, PhysicalState(name, idx,
            BodyState(pos, 0.0, energy), objs; seed=seed, extra_props=extra))
    end

    # Phase 2: holding object, hand at each cell
    for r in 1:grid.rows, c in 1:grid.cols
        idx += 1
        pos = grid.cells[r, c].position
        energy = 0.9 - 0.02 * (abs(r - obj_r) + abs(c - obj_c))
        energy = max(energy, 0.1)
        objs = [ObjectState(obj, pos, true, true, 0.1)]
        name = Symbol("h$(r)_$(c)_hold")
        push!(states, PhysicalState(name, idx,
            BodyState(pos, 1.0, energy), objs; seed=seed))
    end

    # Phase 3: object placed at target, hand at target
    idx += 1
    tgt_pos = grid.cells[tgt_r, tgt_c].position
    objs_done = [ObjectState(obj, tgt_pos, true, false, 0.1)]
    push!(states, PhysicalState(:done, idx,
        BodyState(tgt_pos, 0.0, 0.7), objs_done;
        seed=seed, extra_props=[:completed]))

    TaskFrame(states; reach_radius=grid.cell_size * 1.5)
end

"""
    demo_rectangular_workspace()

4x4 grid, one block at (1,1), target at (4,4).
Hand starts at (1,1), must navigate grid to pick up block, carry to (4,4), release.

Shows:
  - Grid layout with trit coloring (somatotopic: left=MINUS, center=ERGODIC, right=PLUS)
  - Adjacency-based transitions
  - ◇completed landscape (how many steps until completion becomes possible)
  - Plan via BFS
"""
function demo_rectangular_workspace()
    println("Rectangular Workspace Grid")
    println("=" ^ 60)

    rows, cols = 4, 4
    grid = WorkspaceGrid(rows, cols;
        cell_size=0.15,
        objects=Dict(:block => (1, 1)),
        targets=Dict(:block => (4, 4)))

    println("$(rows)x$(cols) grid, cell_size=$(grid.cell_size)m")
    println("Block at (1,1), target at (4,4)\n")

    # Show grid with trits
    println("Grid trit map (somatotopic laterality):")
    for r in 1:rows
        row_str = "  "
        for c in 1:cols
            t = grid_trit(r, c, cols)
            obj_here = haskey(grid.objects, :block) && grid.objects[:block] == (r, c)
            tgt_here = haskey(grid.target, :block) && grid.target[:block] == (r, c)
            cell_char = obj_here ? "B" : (tgt_here ? "T" : "·")
            trit_char = t == MINUS ? "-" : (t == PLUS ? "+" : "0")
            row_str *= "[$(cell_char)$(trit_char)] "
        end
        println(row_str)
    end

    println("\n  B=block, T=target, ·=empty")
    println("  -=MINUS(C4/left), 0=ERGODIC(Cz), +=PLUS(C3/right)\n")

    frame = grid_frame(grid)
    n_states = length(frame.states)
    n_trans = length(frame.transitions)
    println("Generated: $(n_states) states, $(n_trans) transitions")

    # Find start state (hand at 1,1 no hold) and done state
    start = first(s for s in frame.states if s.name == :h1_1)
    done = first(s for s in frame.states if s.name == :done)

    println("\nStart: :$(start.name) trit=$(Int(start.trit)) affs={$(join(sort(collect(start.propositions)), ", "))}")
    println("Goal:  :$(done.name) trit=$(Int(done.trit)) affs={$(join(sort(collect(done.propositions)), ", "))}")

    # Modal check
    completed = Atomic(:completed)
    println("\n:done completed=$(affordance_at(frame, done, completed))")
    println(":done □completed=$(affordance_at(frame, done, box(completed)))")

    # Plan
    println("\nPlan (start → ◇completed, BFS):")
    plan = plan_to_necessity(frame, start, completed)
    if isempty(plan) && affordance_at(frame, start, box(completed))
        println("  Already done!")
    elseif isempty(plan)
        # Fall back: find path to any state with :completed
        println("  □completed unreachable (task reversible). Finding path to completed...")
        visited = Set{Int}()
        queue = [(start, Tuple{PhysicalState, MotorPrimitive}[])]
        found_plan = Tuple{PhysicalState, MotorPrimitive}[]
        while !isempty(queue)
            current, path = popfirst!(queue)
            current.index in visited && continue
            push!(visited, current.index)
            if :completed in current.propositions
                found_plan = path
                break
            end
            for s2 in frame.states
                s2.index in visited && continue
                key = (current.index, s2.index)
                if haskey(frame.transitions, key)
                    action = frame.transitions[key]
                    push!(queue, (s2, vcat(path, [(s2, action)])))
                end
            end
        end
        if isempty(found_plan)
            println("  No path to completed found")
        else
            println("  $(length(found_plan)) steps to :completed:")
            for (i, (s, action)) in enumerate(found_plan)
                println("    $(i). $(action) → :$(s.name)")
            end
        end
    else
        for (i, (s, action)) in enumerate(plan)
            println("  $(i). $(action) → :$(s.name)")
        end
    end

    # Trit distribution
    minus_n = count(s -> s.trit == MINUS, frame.states)
    ergo_n = count(s -> s.trit == ERGODIC, frame.states)
    plus_n = count(s -> s.trit == PLUS, frame.states)
    println("\nTrit distribution: MINUS=$(minus_n), ERGODIC=$(ergo_n), PLUS=$(plus_n)")
    trit_sum = sum(Int(s.trit) for s in frame.states)
    println("Trit sum: $(trit_sum), mod 3 = $(mod(trit_sum + 3, 3))")

    frame
end

end # module PhysicalTaskWorlds
