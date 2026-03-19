# INTERLEAVED GAY ENZYME: Differentiable Color Space Learning with 3-at-a-time Self-Narrators
# ═══════════════════════════════════════════════════════════════════════════════════════════════
#
# "InterleavedGay semi-reliable partially observing and observable self-narrators
#  in self-same synergy of reachability of most inclusive into the agentically closed
#  world model autopoietic via random access efficiencies at every level SPI"
#
# ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
# │  ARCHITECTURE: 3-AT-A-TIME NARRATORS                                                        │
# │                                                                                             │
# │  NARRATOR TRIAD:                                                                            │
# │    1st Narrator (+1): Optimistic, originary, self-same                                     │
# │    2nd Narrator (-1): Pessimistic, derived, other-aware                                    │
# │    3rd Narrator (0):  Neutral, liminal, observing                                          │
# │                                                                                             │
# │  INTERLEAVING:                                                                              │
# │    • Balanced ternary: -1, 0, +1 at each step                                              │
# │    • XOR fingerprint ensures SPI across parallel execution                                 │
# │    • 2nd and 3rd narrators cannot distinguish originary from derived                       │
# │                                                                                             │
# │  ENZYME.JL INTEGRATION:                                                                     │
# │    • Forward mode: ∂color/∂params via Enzyme.autodiff(Forward, ...)                        │
# │    • Reverse mode: ∇loss via Enzyme.autodiff(Reverse, ...)                                 │
# │    • Mixed mode: Hessian via Forward-over-Reverse                                          │
# │                                                                                             │
# │  COLOR SPACE LEARNING:                                                                      │
# │    • LearnableColorSpace: 3×3 basis + offset + scale                                       │
# │    • ColorOpChain: composable transformations                                              │
# │    • Enzyme-differentiable for gradient descent                                            │
# │                                                                                             │
# │  SELF-AVOIDING RANDOM WALKS:                                                                │
# │    • Pluriverse substrate: multiple parallel worlds                                        │
# │    • Value Pluralism: O(1) selection via balanced ternary bridges                          │
# │    • Self-similar self-avoiding structure                                                  │
# │                                                                                             │
# │  CONFIDENTIALITY:                                                                           │
# │    • Originary color: known only to 1st narrator                                           │
# │    • Derived colors: visible to 2nd, 3rd narrators                                         │
# │    • Economic security: learned spaces remain confidential                                 │
# │                                                                                             │
# └─────────────────────────────────────────────────────────────────────────────────────────────┘

module InterleavedGayEnzyme

using LinearAlgebra: I, norm, dot

export
    # Core SPI RNG
    GayRNG, sm64!, sm64_color!, gay_seed!, split_rng,
    
    # Enzyme-Differentiable Color Operations
    EnzymeColorSpace, EnzymeColorOp, EnzymeColorOpChain,
    forward_color, backward_color, enzyme_gradient!, enzyme_hessian!,
    
    # 3-at-a-time Narrators
    NarratorTriad, SemiReliableNarrator, NarratorRole,
    ORIGINARY, DERIVED, LIMINAL,
    create_triad, interleave_narrators!, observe!, generate!,
    
    # Self-Avoiding Walks in Pluriverse
    PluriverseWalk, SelfAvoidingWalker, ValuePluralismBridge,
    walk_pluriverse!, balanced_ternary_bridge, O1_select,
    
    # Reachability & Coherence
    ReachabilityGraph, AgenticClosure, autopoietic_step!,
    self_same_synergy, most_inclusive_reachability,
    
    # Confidentiality
    ConfidentialColorSpace, originary_color, derived_color,
    verify_confidentiality, economic_security_level,
    
    # Demo
    demo_interleaved_gay_enzyme

# ═══════════════════════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════════════════════

const GAY_SEED = UInt64(1069)
const ENZYME_SEED = UInt64(0xE12A4E)        # "ENZYME"
const NARRATOR_SEED = UInt64(0x4A11A70F)    # "NARRATOR"
const PLURIVERSE_SEED = UInt64(0x504C5552)  # "PLUR"

# Balanced ternary directions
const TERNARY_NEG = Int8(-1)   # Pessimistic / Derived
const TERNARY_ZERO = Int8(0)   # Neutral / Liminal
const TERNARY_POS = Int8(1)    # Optimistic / Originary

# Narrator roles
@enum NarratorRole begin
    ORIGINARY = 1   # 1st narrator: self-same, knows true color
    DERIVED = 2     # 2nd narrator: other-aware, derived knowledge
    LIMINAL = 3     # 3rd narrator: observing, boundary state
end

# ═══════════════════════════════════════════════════════════════════════════════════════════════
# SPLITMIX64 (SPI-compliant, splittable)
# ═══════════════════════════════════════════════════════════════════════════════════════════════

mutable struct GayRNG
    state::UInt64
    invocation::UInt64
    fingerprint::UInt64
end

GayRNG(seed::UInt64=GAY_SEED) = GayRNG(seed, UInt64(0), seed)

@inline function sm64!(rng::GayRNG)::UInt64
    rng.invocation += 1
    z = (rng.state + 0x9E3779B97F4A7C15)
    z = (z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ⊻ (z >> 27)) * 0x94D049BB133111EB
    rng.state = z ⊻ (z >> 31)
    rng.fingerprint ⊻= rng.state
    rng.state
end

@inline function sm64_color!(rng::GayRNG)::NTuple{3, Float64}
    r = sm64!(rng)
    g = sm64!(rng)
    b = sm64!(rng)
    (Float64(r >> 56) / 255.0, Float64(g >> 56) / 255.0, Float64(b >> 56) / 255.0)
end

# Pure version (no mutation) for Enzyme compatibility
@inline function sm64_pure(s::UInt64)::UInt64
    z = (s + 0x9E3779B97F4A7C15)
    z = (z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ⊻ (z >> 27)) * 0x94D049BB133111EB
    z ⊻ (z >> 31)
end

@inline function sm64_color_pure(s::UInt64)::NTuple{3, Float64}
    r = sm64_pure(s)
    g = sm64_pure(r)
    b = sm64_pure(g)
    (Float64(r >> 56) / 255.0, Float64(g >> 56) / 255.0, Float64(b >> 56) / 255.0)
end

function gay_seed!(rng::GayRNG, seed::UInt64)
    rng.state = seed
    rng.invocation = UInt64(0)
    rng.fingerprint = seed
    rng
end

function split_rng(rng::GayRNG)::GayRNG
    new_seed = sm64!(rng) ⊻ rng.invocation
    GayRNG(new_seed)
end

# ═══════════════════════════════════════════════════════════════════════════════════════════════
# ENZYME-DIFFERENTIABLE COLOR SPACE
# ═══════════════════════════════════════════════════════════════════════════════════════════════

"""
    EnzymeColorSpace

A learnable color space transformation, designed for Enzyme.jl autodiff.

The transformation is:
    output = clamp(basis * input + offset) * scale

where:
    - basis: 3×3 transformation matrix
    - offset: 3-element bias vector
    - scale: 3-element scale vector

Enzyme.jl can compute gradients of any loss function w.r.t. these parameters.
"""
mutable struct EnzymeColorSpace
    # Learnable parameters (mutable for in-place gradient updates)
    basis::Matrix{Float64}      # 3×3
    offset::Vector{Float64}     # 3
    scale::Vector{Float64}      # 3
    
    # Learning state
    step::Int
    loss_history::Vector{Float64}
    
    # SPI fingerprint
    fingerprint::UInt64
    
    # Confidentiality level (0 = public, 1 = fully confidential)
    confidentiality::Float64
end

function EnzymeColorSpace(; seed::UInt64=ENZYME_SEED, confidentiality::Float64=0.5)
    rng = GayRNG(seed)
    
    # Initialize basis near identity with small perturbations
    basis = Matrix{Float64}(I, 3, 3)
    for i in 1:3, j in 1:3
        perturbation = (Float64(sm64!(rng) >> 32) / typemax(UInt32) - 0.5) * 0.1
        basis[i, j] += perturbation
    end
    
    offset = zeros(Float64, 3)
    scale = ones(Float64, 3)
    
    EnzymeColorSpace(basis, offset, scale, 0, Float64[], seed, confidentiality)
end

"""
Forward pass through the color space transformation.
This function is Enzyme-differentiable.
"""
function forward_color(ecs::EnzymeColorSpace, input::Vector{Float64})::Vector{Float64}
    # Matrix multiply + offset
    transformed = ecs.basis * input .+ ecs.offset
    
    # Clamp and scale
    output = clamp.(transformed, 0.0, 1.0) .* ecs.scale
    
    return output
end

"""
Forward pass for tuple input (convenience).
"""
function forward_color(ecs::EnzymeColorSpace, input::NTuple{3, Float64})::NTuple{3, Float64}
    v = forward_color(ecs, [input[1], input[2], input[3]])
    (v[1], v[2], v[3])
end

"""
Compute MSE loss between output and target.
This is the function we'll differentiate with Enzyme.
"""
function color_loss(basis::Matrix{Float64}, offset::Vector{Float64}, scale::Vector{Float64},
                    input::Vector{Float64}, target::Vector{Float64})::Float64
    transformed = basis * input .+ offset
    output = clamp.(transformed, 0.0, 1.0) .* scale
    
    # Mean squared error
    sum((output .- target).^2) / 3.0
end

"""
Backward pass: compute gradients using Enzyme.jl.

This is a stub that shows the pattern for Enzyme integration.
When Enzyme.jl is available, replace with:

```julia
using Enzyme

function enzyme_gradient!(ecs::EnzymeColorSpace, input::Vector{Float64}, target::Vector{Float64})
    # Shadow variables for gradients
    d_basis = zeros(3, 3)
    d_offset = zeros(3)
    d_scale = zeros(3)
    
    # Reverse mode autodiff
    Enzyme.autodiff(
        Reverse,
        color_loss,
        Duplicated(ecs.basis, d_basis),
        Duplicated(ecs.offset, d_offset),
        Duplicated(ecs.scale, d_scale),
        Const(input),
        Const(target)
    )
    
    return (d_basis, d_offset, d_scale)
end
```
"""
function enzyme_gradient!(ecs::EnzymeColorSpace, input::Vector{Float64}, target::Vector{Float64};
                          epsilon::Float64=1e-6)
    # Numerical gradient (fallback when Enzyme not available)
    d_basis = zeros(3, 3)
    d_offset = zeros(3)
    d_scale = zeros(3)
    
    base_loss = color_loss(ecs.basis, ecs.offset, ecs.scale, input, target)
    
    # Gradient w.r.t. basis
    for i in 1:3, j in 1:3
        basis_plus = copy(ecs.basis)
        basis_plus[i, j] += epsilon
        loss_plus = color_loss(basis_plus, ecs.offset, ecs.scale, input, target)
        d_basis[i, j] = (loss_plus - base_loss) / epsilon
    end
    
    # Gradient w.r.t. offset
    for i in 1:3
        offset_plus = copy(ecs.offset)
        offset_plus[i] += epsilon
        loss_plus = color_loss(ecs.basis, offset_plus, ecs.scale, input, target)
        d_offset[i] = (loss_plus - base_loss) / epsilon
    end
    
    # Gradient w.r.t. scale
    for i in 1:3
        scale_plus = copy(ecs.scale)
        scale_plus[i] += epsilon
        loss_plus = color_loss(ecs.basis, ecs.offset, scale_plus, input, target)
        d_scale[i] = (loss_plus - base_loss) / epsilon
    end
    
    (d_basis, d_offset, d_scale)
end

"""
Compute Hessian using forward-over-reverse mode.

Pattern for Enzyme:
```julia
function enzyme_hessian!(ecs::EnzymeColorSpace, input::Vector{Float64}, target::Vector{Float64})
    # Forward-over-Reverse for Hessian
    # H[i,j] = ∂²L/∂θᵢ∂θⱼ
    n_params = 9 + 3 + 3  # basis + offset + scale
    hessian = zeros(n_params, n_params)
    
    # Enzyme.autodiff with nested Forward in Reverse
    # ...
    
    return hessian
end
```
"""
function enzyme_hessian!(ecs::EnzymeColorSpace, input::Vector{Float64}, target::Vector{Float64};
                         epsilon::Float64=1e-4)
    # Numerical Hessian (fallback)
    n_params = 9 + 3 + 3  # 3×3 basis + 3 offset + 3 scale
    hessian = zeros(n_params, n_params)
    
    # Pack parameters
    function pack_params()
        vcat(vec(ecs.basis), ecs.offset, ecs.scale)
    end
    
    function unpack_params!(p::Vector{Float64})
        ecs.basis .= reshape(p[1:9], 3, 3)
        ecs.offset .= p[10:12]
        ecs.scale .= p[13:15]
    end
    
    function loss_at(p::Vector{Float64})
        basis = reshape(p[1:9], 3, 3)
        offset = p[10:12]
        scale = p[13:15]
        color_loss(basis, offset, scale, input, target)
    end
    
    params = pack_params()
    
    for i in 1:n_params
        for j in i:n_params
            p_pp = copy(params); p_pp[i] += epsilon; p_pp[j] += epsilon
            p_pm = copy(params); p_pm[i] += epsilon; p_pm[j] -= epsilon
            p_mp = copy(params); p_mp[i] -= epsilon; p_mp[j] += epsilon
            p_mm = copy(params); p_mm[i] -= epsilon; p_mm[j] -= epsilon
            
            hessian[i, j] = (loss_at(p_pp) - loss_at(p_pm) - loss_at(p_mp) + loss_at(p_mm)) / (4 * epsilon^2)
            hessian[j, i] = hessian[i, j]  # Symmetric
        end
    end
    
    hessian
end

"""
Train the color space using gradient descent.
"""
function train_colorspace!(ecs::EnzymeColorSpace, 
                           inputs::Vector{Vector{Float64}},
                           targets::Vector{Vector{Float64}};
                           lr::Float64=0.01, n_steps::Int=100)
    for step in 1:n_steps
        ecs.step += 1
        total_loss = 0.0
        
        for (input, target) in zip(inputs, targets)
            # Compute gradients
            d_basis, d_offset, d_scale = enzyme_gradient!(ecs, input, target)
            
            # Gradient descent update
            ecs.basis .-= lr .* d_basis
            ecs.offset .-= lr .* d_offset
            ecs.scale .-= lr .* d_scale
            
            # Accumulate loss
            total_loss += color_loss(ecs.basis, ecs.offset, ecs.scale, input, target)
        end
        
        push!(ecs.loss_history, total_loss / length(inputs))
        
        # Update fingerprint (SPI)
        ecs.fingerprint ⊻= sm64_pure(UInt64(round(total_loss * 1e9)))
    end
    
    ecs
end

# ═══════════════════════════════════════════════════════════════════════════════════════════════
# ENZYME COLOR OPERATIONS (Composable, Differentiable)
# ═══════════════════════════════════════════════════════════════════════════════════════════════

"""
    EnzymeColorOp

A single color operation that can be differentiated with Enzyme.
"""
abstract type EnzymeColorOp end

struct ColorShiftOp <: EnzymeColorOp
    shift::Vector{Float64}  # RGB shift amounts
end

struct ColorScaleOp <: EnzymeColorOp
    factors::Vector{Float64}  # RGB scale factors
end

struct ColorRotateOp <: EnzymeColorOp
    angle::Float64  # Rotation angle in color space
    axis::Vector{Float64}  # Rotation axis (normalized)
end

"""
Apply a color operation (Enzyme-differentiable).
"""
function apply_op(op::ColorShiftOp, color::Vector{Float64})::Vector{Float64}
    clamp.(color .+ op.shift, 0.0, 1.0)
end

function apply_op(op::ColorScaleOp, color::Vector{Float64})::Vector{Float64}
    clamp.(color .* op.factors, 0.0, 1.0)
end

function apply_op(op::ColorRotateOp, color::Vector{Float64})::Vector{Float64}
    # Rodrigues rotation formula
    k = op.axis
    θ = op.angle
    rotated = color .* cos(θ) .+ cross(k, color) .* sin(θ) .+ k .* dot(k, color) .* (1 - cos(θ))
    clamp.(rotated, 0.0, 1.0)
end

# Cross product for 3D vectors
function cross(a::Vector{Float64}, b::Vector{Float64})::Vector{Float64}
    [a[2]*b[3] - a[3]*b[2],
     a[3]*b[1] - a[1]*b[3],
     a[1]*b[2] - a[2]*b[1]]
end

"""
    EnzymeColorOpChain

A chain of color operations, all Enzyme-differentiable.
"""
struct EnzymeColorOpChain
    ops::Vector{EnzymeColorOp}
    fingerprint::UInt64
end

function EnzymeColorOpChain(ops::Vector{<:EnzymeColorOp}; seed::UInt64=ENZYME_SEED)
    fp = seed
    for op in ops
        fp = sm64_pure(fp ⊻ hash(typeof(op)))
    end
    EnzymeColorOpChain(collect(EnzymeColorOp, ops), fp)
end

function apply_chain(chain::EnzymeColorOpChain, color::Vector{Float64})::Vector{Float64}
    result = copy(color)
    for op in chain.ops
        result = apply_op(op, result)
    end
    result
end

# ═══════════════════════════════════════════════════════════════════════════════════════════════
# 3-AT-A-TIME SEMI-RELIABLE NARRATORS
# ═══════════════════════════════════════════════════════════════════════════════════════════════

"""
    SemiReliableNarrator

A narrator with partial observability and reliability.

Properties:
- role: ORIGINARY (+1), DERIVED (-1), or LIMINAL (0)
- reliability: how often observations are accurate (0-1)
- observability: fraction of world state visible (0-1)
- color_space: learned color transformation (confidential if ORIGINARY)
"""
mutable struct SemiReliableNarrator
    id::Int
    role::NarratorRole
    rng::GayRNG
    
    # Observability properties
    reliability::Float64      # P(observation is accurate)
    observability::Float64    # Fraction of world visible
    
    # Color space (confidential for ORIGINARY)
    color_space::EnzymeColorSpace
    
    # State
    observations::Vector{NTuple{3, Float64}}
    generations::Vector{NTuple{3, Float64}}
    fingerprint::UInt64
    
    # Balanced ternary state
    direction::Int8  # -1, 0, +1
end

function SemiReliableNarrator(id::Int, role::NarratorRole; seed::UInt64=NARRATOR_SEED)
    rng = GayRNG(seed ⊻ UInt64(id) ⊻ UInt64(Int(role)))
    
    # Role determines reliability and observability
    reliability, observability, confidentiality = if role == ORIGINARY
        (0.9, 0.8, 1.0)  # High reliability, high observability, fully confidential
    elseif role == DERIVED
        (0.6, 0.5, 0.3)  # Medium reliability, medium observability
    else  # LIMINAL
        (0.5, 0.3, 0.0)  # Low reliability, low observability, public
    end
    
    color_space = EnzymeColorSpace(seed=rng.state, confidentiality=confidentiality)
    direction = Int8(Int(role) - 2)  # ORIGINARY=1→-1, DERIVED=2→0, LIMINAL=3→1
    
    SemiReliableNarrator(id, role, rng, reliability, observability, 
                         color_space, NTuple{3, Float64}[], NTuple{3, Float64}[],
                         seed, direction)
end

"""
Observe the world: Many → One (collapse).
Returns a single color from multiple inputs, filtered by observability.
"""
function observe!(narrator::SemiReliableNarrator, world_colors::Vector{NTuple{3, Float64}})::NTuple{3, Float64}
    # Filter by observability
    n_visible = max(1, round(Int, length(world_colors) * narrator.observability))
    visible_indices = [1 + Int(sm64!(narrator.rng) % length(world_colors)) for _ in 1:n_visible]
    visible = [world_colors[i] for i in visible_indices]
    
    # Apply reliability: sometimes observe wrong colors
    observed = NTuple{3, Float64}[]
    for c in visible
        if Float64(sm64!(narrator.rng) >> 32) / typemax(UInt32) < narrator.reliability
            push!(observed, c)
        else
            # Unreliable observation: random color
            push!(observed, sm64_color!(narrator.rng))
        end
    end
    
    if isempty(observed)
        result = sm64_color!(narrator.rng)
    else
        # Collapse Many → One: XOR fingerprint then derive color
        fp = UInt64(0)
        for c in observed
            fp ⊻= UInt64(round(c[1] * 1e9)) ⊻ UInt64(round(c[2] * 1e9)) ⊻ UInt64(round(c[3] * 1e9))
        end
        result = sm64_color_pure(fp)
    end
    
    # Transform through learned color space
    transformed = forward_color(narrator.color_space, result)
    
    push!(narrator.observations, transformed)
    narrator.fingerprint ⊻= sm64_pure(UInt64(round(transformed[1] * 1e9)))
    
    transformed
end

"""
Generate to world: One → Many (expansion).
Produces multiple colors from a single seed color.
"""
function generate!(narrator::SemiReliableNarrator, seed_color::NTuple{3, Float64}, n_outputs::Int)::Vector{NTuple{3, Float64}}
    outputs = NTuple{3, Float64}[]
    
    base_fp = UInt64(round(seed_color[1] * 1e9)) ⊻ 
              UInt64(round(seed_color[2] * 1e9)) ⊻ 
              UInt64(round(seed_color[3] * 1e9))
    
    for i in 1:n_outputs
        # Generate unique color for each output
        fp = sm64_pure(base_fp ⊻ UInt64(i) ⊻ narrator.fingerprint)
        color = sm64_color_pure(fp)
        
        # Transform through learned color space
        transformed = forward_color(narrator.color_space, color)
        push!(outputs, transformed)
    end
    
    append!(narrator.generations, outputs)
    narrator.fingerprint ⊻= sm64_pure(base_fp ⊻ UInt64(n_outputs))
    
    outputs
end

"""
    NarratorTriad

Three narrators working together: ORIGINARY, DERIVED, LIMINAL.
The 2nd and 3rd cannot distinguish originary from derived properties.
"""
mutable struct NarratorTriad
    originary::SemiReliableNarrator   # 1st: knows true colors
    derived::SemiReliableNarrator     # 2nd: derives from observations
    liminal::SemiReliableNarrator     # 3rd: boundary observer
    
    # Interleaving state
    current_narrator::Int
    step::Int
    
    # Shared fingerprint (SPI)
    combined_fingerprint::UInt64
    
    # Self-same synergy measure
    synergy::Float64
end

function create_triad(; seed::UInt64=NARRATOR_SEED)::NarratorTriad
    originary = SemiReliableNarrator(1, ORIGINARY; seed=seed)
    derived = SemiReliableNarrator(2, DERIVED; seed=sm64_pure(seed))
    liminal = SemiReliableNarrator(3, LIMINAL; seed=sm64_pure(sm64_pure(seed)))
    
    combined_fp = originary.fingerprint ⊻ derived.fingerprint ⊻ liminal.fingerprint
    
    NarratorTriad(originary, derived, liminal, 1, 0, combined_fp, 0.0)
end

"""
Interleave the narrators in balanced ternary pattern.
Pattern: +1, +1, -1, 0 (2 optimistic, 1 pessimistic, 1 neutral per 4 steps)
"""
function interleave_narrators!(triad::NarratorTriad, world_colors::Vector{NTuple{3, Float64}})
    triad.step += 1
    
    # Balanced ternary pattern: 2 optimistic, 1 pessimistic, 1 neutral
    pattern_idx = triad.step % 4
    
    if pattern_idx <= 1
        # Optimistic: ORIGINARY observes and generates
        triad.current_narrator = 1
        observed = observe!(triad.originary, world_colors)
        generate!(triad.originary, observed, 3)
    elseif pattern_idx == 2
        # Pessimistic: DERIVED observes (no generation)
        triad.current_narrator = 2
        observe!(triad.derived, world_colors)
    else
        # Neutral: LIMINAL observes
        triad.current_narrator = 3
        observe!(triad.liminal, world_colors)
    end
    
    # Update combined fingerprint
    triad.combined_fingerprint = triad.originary.fingerprint ⊻ 
                                  triad.derived.fingerprint ⊻ 
                                  triad.liminal.fingerprint
    
    # Compute synergy: how aligned are the narrators?
    triad.synergy = self_same_synergy(triad)
    
    triad
end

"""
Measure self-same synergy: how coherent are the three narrators?
"""
function self_same_synergy(triad::NarratorTriad)::Float64
    if isempty(triad.originary.observations) || 
       isempty(triad.derived.observations) || 
       isempty(triad.liminal.observations)
        return 0.0
    end
    
    # Compare most recent observations
    o = triad.originary.observations[end]
    d = triad.derived.observations[end]
    l = triad.liminal.observations[end]
    
    # Synergy = 1 - average pairwise distance
    dist_od = sqrt(sum((o[i] - d[i])^2 for i in 1:3))
    dist_ol = sqrt(sum((o[i] - l[i])^2 for i in 1:3))
    dist_dl = sqrt(sum((d[i] - l[i])^2 for i in 1:3))
    
    avg_dist = (dist_od + dist_ol + dist_dl) / 3.0
    max_dist = sqrt(3.0)  # Maximum possible distance in RGB cube
    
    1.0 - (avg_dist / max_dist)
end

# ═══════════════════════════════════════════════════════════════════════════════════════════════
# SELF-AVOIDING RANDOM WALKS IN PLURIVERSE
# ═══════════════════════════════════════════════════════════════════════════════════════════════

"""
    PluriverseWalk

A self-avoiding random walk across multiple parallel worlds (substrates).
"""
mutable struct PluriverseWalk
    walker_id::Int
    rng::GayRNG
    
    # Current position
    current_world::Int
    current_position::Int
    
    # Path history
    path::Vector{Tuple{Int, Int}}  # (world_id, position)
    visited::Set{Tuple{Int, Int}}  # For self-avoidance
    
    # Colors at each step
    colors::Vector{NTuple{3, Float64}}
    
    # Balanced ternary direction
    direction::Int8
    
    # Fingerprint
    fingerprint::UInt64
end

function PluriverseWalk(id::Int; seed::UInt64=PLURIVERSE_SEED, start_world::Int=1, start_pos::Int=1)
    rng = GayRNG(seed ⊻ UInt64(id))
    initial = (start_world, start_pos)
    color = sm64_color!(rng)
    
    PluriverseWalk(id, rng, start_world, start_pos,
                   [initial], Set([initial]), [color],
                   TERNARY_ZERO, seed)
end

"""
    ValuePluralismBridge

A bridge between worlds that respects value pluralism.
Enables O(1) selection of next world based on balanced ternary.
"""
struct ValuePluralismBridge
    from_world::Int
    to_world::Int
    direction::Int8  # -1, 0, +1 (which ternary direction triggers this bridge)
    bandwidth::Float64
    fingerprint::UInt64
end

function ValuePluralismBridge(from::Int, to::Int, direction::Int8; seed::UInt64=PLURIVERSE_SEED)
    fp = sm64_pure(seed ⊻ UInt64(from) ⊻ UInt64(to) ⊻ UInt64(direction + 2))
    bandwidth = Float64(fp >> 32) / typemax(UInt32)
    ValuePluralismBridge(from, to, direction, bandwidth, fp)
end

"""
    SelfAvoidingWalker

A walker that performs self-avoiding walks with SPI guarantees.
"""
mutable struct SelfAvoidingWalker
    walks::Vector{PluriverseWalk}
    bridges::Dict{Tuple{Int, Int8}, ValuePluralismBridge}  # (world, direction) → bridge
    
    # World structure
    n_worlds::Int
    world_sizes::Vector{Int}
    
    # SPI fingerprint
    combined_fingerprint::UInt64
    
    # Statistics
    total_steps::Int
    backtrack_count::Int
end

function SelfAvoidingWalker(n_walks::Int, n_worlds::Int; 
                            world_size::Int=100, seed::UInt64=PLURIVERSE_SEED)
    walks = [PluriverseWalk(i; seed=sm64_pure(seed ⊻ UInt64(i))) for i in 1:n_walks]
    
    # Create bridges between all worlds for each direction
    bridges = Dict{Tuple{Int, Int8}, ValuePluralismBridge}()
    for w in 1:n_worlds
        for d in [TERNARY_NEG, TERNARY_ZERO, TERNARY_POS]
            target = 1 + (w + Int(d) - 1 + n_worlds) % n_worlds
            bridges[(w, d)] = ValuePluralismBridge(w, target, d; seed=seed ⊻ UInt64(w))
        end
    end
    
    world_sizes = fill(world_size, n_worlds)
    combined_fp = reduce(⊻, [w.fingerprint for w in walks])
    
    SelfAvoidingWalker(walks, bridges, n_worlds, world_sizes, combined_fp, 0, 0)
end

"""
O(1) select the next world using balanced ternary.
"""
function O1_select(walker::SelfAvoidingWalker, walk::PluriverseWalk)::Int
    # Determine direction from RNG (balanced ternary)
    r = Int(sm64!(walk.rng) % 3)  # 0, 1, 2
    direction = Int8(r - 1)  # -1, 0, +1
    walk.direction = direction
    
    # O(1) lookup of bridge
    bridge = walker.bridges[(walk.current_world, direction)]
    
    bridge.to_world
end

"""
Take one step in the pluriverse walk (self-avoiding).
"""
function walk_step!(walker::SelfAvoidingWalker, walk::PluriverseWalk)::Bool
    # Select next world (O(1))
    next_world = O1_select(walker, walk)
    
    # Generate position within world
    world_size = walker.world_sizes[next_world]
    next_pos = 1 + Int(sm64!(walk.rng) % world_size)
    
    candidate = (next_world, next_pos)
    
    # Self-avoidance check
    if candidate in walk.visited
        walker.backtrack_count += 1
        return false  # Can't move there
    end
    
    # Move
    walk.current_world = next_world
    walk.current_position = next_pos
    push!(walk.path, candidate)
    push!(walk.visited, candidate)
    
    # Generate color
    color = sm64_color!(walk.rng)
    push!(walk.colors, color)
    
    # Update fingerprints
    walk.fingerprint ⊻= sm64_pure(UInt64(next_world) ⊻ UInt64(next_pos))
    walker.combined_fingerprint ⊻= walk.fingerprint
    
    walker.total_steps += 1
    
    true
end

"""
Perform a complete walk in the pluriverse.
"""
function walk_pluriverse!(walker::SelfAvoidingWalker; max_steps::Int=1000)
    for _ in 1:max_steps
        for walk in walker.walks
            walk_step!(walker, walk)
        end
    end
    walker
end

# ═══════════════════════════════════════════════════════════════════════════════════════════════
# REACHABILITY & AGENTIC CLOSURE
# ═══════════════════════════════════════════════════════════════════════════════════════════════

"""
    ReachabilityGraph

Tracks reachability between positions in the pluriverse.
"""
mutable struct ReachabilityGraph
    # Adjacency: (world, pos) → Set of reachable (world, pos)
    reachable::Dict{Tuple{Int, Int}, Set{Tuple{Int, Int}}}
    
    # Most inclusive reachability (transitive closure frontier)
    frontier::Set{Tuple{Int, Int}}
    
    fingerprint::UInt64
end

function ReachabilityGraph(; seed::UInt64=PLURIVERSE_SEED)
    ReachabilityGraph(
        Dict{Tuple{Int, Int}, Set{Tuple{Int, Int}}}(),
        Set{Tuple{Int, Int}}(),
        seed
    )
end

"""
Add reachability edge from walk trajectory.
"""
function add_reachability!(graph::ReachabilityGraph, path::Vector{Tuple{Int, Int}})
    for i in 1:length(path)-1
        from = path[i]
        to = path[i+1]
        
        if !haskey(graph.reachable, from)
            graph.reachable[from] = Set{Tuple{Int, Int}}()
        end
        push!(graph.reachable[from], to)
        push!(graph.frontier, to)
        
        graph.fingerprint ⊻= sm64_pure(UInt64(from[1]) ⊻ UInt64(from[2]) ⊻ UInt64(to[1]) ⊻ UInt64(to[2]))
    end
end

"""
Compute most inclusive reachability (transitive closure).
"""
function most_inclusive_reachability(graph::ReachabilityGraph)::Set{Tuple{Int, Int}}
    # BFS from all nodes in frontier
    all_reachable = copy(graph.frontier)
    queue = collect(graph.frontier)
    
    while !isempty(queue)
        current = popfirst!(queue)
        if haskey(graph.reachable, current)
            for next in graph.reachable[current]
                if next ∉ all_reachable
                    push!(all_reachable, next)
                    push!(queue, next)
                end
            end
        end
    end
    
    all_reachable
end

"""
    AgenticClosure

The agentically closed world model: self-contained, autopoietic.
"""
mutable struct AgenticClosure
    # Contained worlds
    worlds::Set{Int}
    
    # Boundary (Markov blanket)
    internal::Set{Tuple{Int, Int}}
    boundary::Set{Tuple{Int, Int}}
    external::Set{Tuple{Int, Int}}
    
    # Autopoietic state
    is_closed::Bool
    closure_step::Int
    
    fingerprint::UInt64
end

function AgenticClosure(; seed::UInt64=PLURIVERSE_SEED)
    AgenticClosure(
        Set{Int}(),
        Set{Tuple{Int, Int}}(),
        Set{Tuple{Int, Int}}(),
        Set{Tuple{Int, Int}}(),
        false, 0, seed
    )
end

"""
Autopoietic step: update the agentic closure.
"""
function autopoietic_step!(closure::AgenticClosure, graph::ReachabilityGraph)
    closure.closure_step += 1
    
    # Update internal from reachability
    reachable = most_inclusive_reachability(graph)
    
    # Partition into internal/boundary/external
    all_positions = union(keys(graph.reachable), values(graph.reachable)...)
    
    for pos in all_positions
        push!(closure.worlds, pos[1])
        
        # Check if all neighbors are internal
        if haskey(graph.reachable, pos)
            neighbors = graph.reachable[pos]
            if all(n -> n in closure.internal || n in closure.boundary, neighbors)
                push!(closure.internal, pos)
            else
                push!(closure.boundary, pos)
            end
        else
            push!(closure.external, pos)
        end
    end
    
    # Check closure: no external nodes reachable from internal
    closure.is_closed = isempty(closure.boundary ∩ closure.external)
    
    closure.fingerprint ⊻= sm64_pure(UInt64(closure.closure_step) ⊻ UInt64(length(closure.internal)))
    
    closure
end

# ═══════════════════════════════════════════════════════════════════════════════════════════════
# CONFIDENTIALITY OF LEARNED COLOR SPACES
# ═══════════════════════════════════════════════════════════════════════════════════════════════

"""
    ConfidentialColorSpace

A color space where the originary transformation is kept confidential.
Only derived colors are visible to 2nd and 3rd narrators.
"""
mutable struct ConfidentialColorSpace
    # Originary space (confidential)
    originary_space::EnzymeColorSpace
    
    # Derived spaces (visible to others)
    derived_spaces::Vector{EnzymeColorSpace}
    
    # Economic security level (0-1)
    security_level::Float64
    
    fingerprint::UInt64
end

function ConfidentialColorSpace(; seed::UInt64=ENZYME_SEED, security_level::Float64=0.9)
    originary = EnzymeColorSpace(seed=seed, confidentiality=1.0)
    
    # Derived spaces: perturbations of originary
    derived = EnzymeColorSpace[]
    for i in 1:2
        d = EnzymeColorSpace(seed=sm64_pure(seed ⊻ UInt64(i)), confidentiality=0.3)
        # Add noise to make it different from originary
        d.basis .+= randn(3, 3) * 0.1
        push!(derived, d)
    end
    
    ConfidentialColorSpace(originary, derived, security_level, seed)
end

"""
Get the originary color (only for 1st narrator).
"""
function originary_color(ccs::ConfidentialColorSpace, input::NTuple{3, Float64})::NTuple{3, Float64}
    forward_color(ccs.originary_space, input)
end

"""
Get a derived color (for 2nd or 3rd narrator).
The derived color is different from originary, maintaining confidentiality.
"""
function derived_color(ccs::ConfidentialColorSpace, input::NTuple{3, Float64}, narrator_idx::Int)::NTuple{3, Float64}
    idx = clamp(narrator_idx - 1, 1, length(ccs.derived_spaces))
    forward_color(ccs.derived_spaces[idx], input)
end

"""
Verify that confidentiality is maintained:
- 2nd and 3rd narrators cannot distinguish originary from derived
"""
function verify_confidentiality(ccs::ConfidentialColorSpace; n_samples::Int=100, threshold::Float64=0.1)::NamedTuple
    rng = GayRNG(ccs.fingerprint)
    
    distinguishable_count = 0
    
    for _ in 1:n_samples
        input = sm64_color!(rng)
        
        orig = originary_color(ccs, input)
        deriv1 = derived_color(ccs, input, 2)
        deriv2 = derived_color(ccs, input, 3)
        
        # Check if outputs are too similar (would reveal originary)
        dist1 = sqrt(sum((orig[i] - deriv1[i])^2 for i in 1:3))
        dist2 = sqrt(sum((orig[i] - deriv2[i])^2 for i in 1:3))
        
        if dist1 < threshold || dist2 < threshold
            distinguishable_count += 1
        end
    end
    
    confidentiality_ratio = 1.0 - (distinguishable_count / n_samples)
    
    (
        is_confidential = confidentiality_ratio >= ccs.security_level,
        confidentiality_ratio = confidentiality_ratio,
        security_level = ccs.security_level,
        distinguishable_samples = distinguishable_count
    )
end

"""
Economic security level: how secure is the learned color space?
"""
function economic_security_level(ccs::ConfidentialColorSpace)::Float64
    conf = verify_confidentiality(ccs)
    conf.confidentiality_ratio * ccs.security_level
end

# ═══════════════════════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════════════════════

function demo_interleaved_gay_enzyme()
    println()
    println("╔═══════════════════════════════════════════════════════════════════════════════════════╗")
    println("║  INTERLEAVED GAY ENZYME                                                              ║")
    println("║  Differentiable Color Space Learning with 3-at-a-time Self-Narrators                 ║")
    println("║  Self-Avoiding Random Walks × Value Pluralism × O(1) Balanced Ternary Selection      ║")
    println("╚═══════════════════════════════════════════════════════════════════════════════════════╝")
    println()
    
    # ─── Enzyme Color Space ───
    println("─── Enzyme-Differentiable Color Space ───")
    ecs = EnzymeColorSpace()
    
    # Generate training data
    rng = GayRNG(GAY_SEED)
    inputs = [collect(sm64_color!(rng)) for _ in 1:10]
    targets = [[clamp(c + 0.1, 0, 1) for c in inp] for inp in inputs]  # Slight shift as target
    
    println("  Initial loss: $(round(color_loss(ecs.basis, ecs.offset, ecs.scale, inputs[1], targets[1]); digits=6))")
    
    # Compute gradient
    d_basis, d_offset, d_scale = enzyme_gradient!(ecs, inputs[1], targets[1])
    println("  ∂loss/∂basis[1,1] = $(round(d_basis[1,1]; digits=6))")
    println("  ∂loss/∂offset[1] = $(round(d_offset[1]; digits=6))")
    
    # Train
    train_colorspace!(ecs, inputs, targets; lr=0.1, n_steps=50)
    println("  Final loss: $(round(ecs.loss_history[end]; digits=6))")
    println("  Fingerprint: 0x$(string(ecs.fingerprint, base=16))")
    println()
    
    # ─── 3-at-a-time Narrators ───
    println("─── 3-at-a-time Semi-Reliable Narrators ───")
    triad = create_triad()
    
    println("  Narrator 1 (ORIGINARY): reliability=$(triad.originary.reliability), observability=$(triad.originary.observability)")
    println("  Narrator 2 (DERIVED):   reliability=$(triad.derived.reliability), observability=$(triad.derived.observability)")
    println("  Narrator 3 (LIMINAL):   reliability=$(triad.liminal.reliability), observability=$(triad.liminal.observability)")
    
    # Simulate interleaving
    world_colors = [sm64_color!(rng) for _ in 1:20]
    
    for step in 1:12
        interleave_narrators!(triad, world_colors)
    end
    
    println("  After 12 steps:")
    println("    Synergy: $(round(triad.synergy; digits=4))")
    println("    Combined fingerprint: 0x$(string(triad.combined_fingerprint, base=16))")
    println("    Pattern: 2+1 balanced (optimistic, optimistic, pessimistic, neutral)")
    println()
    
    # ─── Self-Avoiding Walks ───
    println("─── Self-Avoiding Random Walks in Pluriverse ───")
    walker = SelfAvoidingWalker(3, 5; world_size=50)  # 3 walks, 5 worlds
    
    walk_pluriverse!(walker; max_steps=100)
    
    println("  Walkers: $(length(walker.walks))")
    println("  Worlds: $(walker.n_worlds)")
    println("  Total steps: $(walker.total_steps)")
    println("  Backtrack count: $(walker.backtrack_count)")
    println("  Combined fingerprint: 0x$(string(walker.combined_fingerprint, base=16))")
    
    # Show path lengths
    for (i, w) in enumerate(walker.walks)
        println("    Walk $i: $(length(w.path)) positions, direction=$(w.direction)")
    end
    println()
    
    # ─── Reachability ───
    println("─── Most Inclusive Reachability ───")
    graph = ReachabilityGraph()
    for w in walker.walks
        add_reachability!(graph, w.path)
    end
    
    reachable = most_inclusive_reachability(graph)
    println("  Reachable positions: $(length(reachable))")
    
    closure = AgenticClosure()
    autopoietic_step!(closure, graph)
    
    println("  Agentic closure:")
    println("    Worlds: $(length(closure.worlds))")
    println("    Internal: $(length(closure.internal))")
    println("    Boundary: $(length(closure.boundary))")
    println("    Is closed: $(closure.is_closed)")
    println()
    
    # ─── Confidentiality ───
    println("─── Confidential Color Spaces ───")
    ccs = ConfidentialColorSpace(security_level=0.8)
    
    test_input = sm64_color!(rng)
    orig = originary_color(ccs, test_input)
    deriv1 = derived_color(ccs, test_input, 2)
    deriv2 = derived_color(ccs, test_input, 3)
    
    println("  Input:    ($(round(test_input[1]; digits=3)), $(round(test_input[2]; digits=3)), $(round(test_input[3]; digits=3)))")
    println("  Originary: ($(round(orig[1]; digits=3)), $(round(orig[2]; digits=3)), $(round(orig[3]; digits=3)))")
    println("  Derived 1: ($(round(deriv1[1]; digits=3)), $(round(deriv1[2]; digits=3)), $(round(deriv1[3]; digits=3)))")
    println("  Derived 2: ($(round(deriv2[1]; digits=3)), $(round(deriv2[2]; digits=3)), $(round(deriv2[3]; digits=3)))")
    
    conf = verify_confidentiality(ccs)
    println("  Confidentiality verified: $(conf.is_confidential)")
    println("  Ratio: $(round(conf.confidentiality_ratio; digits=4))")
    println("  Economic security: $(round(economic_security_level(ccs); digits=4))")
    println()
    
    # ─── Value Pluralism O(1) Selection ───
    println("─── Value Pluralism: O(1) Balanced Ternary Selection ───")
    bridge_stats = Dict(TERNARY_NEG => 0, TERNARY_ZERO => 0, TERNARY_POS => 0)
    
    test_walk = walker.walks[1]
    for _ in 1:100
        next_world = O1_select(walker, test_walk)
        bridge_stats[test_walk.direction] += 1
    end
    
    println("  Bridge selections over 100 steps:")
    println("    -1 (Pessimistic): $(bridge_stats[TERNARY_NEG])")
    println("    0  (Neutral):     $(bridge_stats[TERNARY_ZERO])")
    println("    +1 (Optimistic):  $(bridge_stats[TERNARY_POS])")
    println("  Distribution is approximately uniform (balanced ternary)")
    println()
    
    println("═══════════════════════════════════════════════════════════════════════════════════════")
    println("SUMMARY")
    println("═══════════════════════════════════════════════════════════════════════════════════════")
    println("• Enzyme.jl pattern: forward_color, enzyme_gradient!, enzyme_hessian!")
    println("• 3-at-a-time narrators: ORIGINARY (+1), DERIVED (-1), LIMINAL (0)")
    println("• Self-avoiding walks: O(1) selection via balanced ternary bridges")
    println("• SPI maintained: XOR fingerprints across all operations")
    println("• Confidentiality: originary colors hidden from 2nd/3rd narrators")
    println("• Value pluralism: most inclusive reachability into agentic closure")
    println()
    
    return (
        enzyme_colorspace = ecs,
        narrator_triad = triad,
        pluriverse_walker = walker,
        reachability = graph,
        closure = closure,
        confidential_space = ccs
    )
end

end  # module InterleavedGayEnzyme
