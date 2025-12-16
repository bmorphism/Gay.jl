# Metatheory Moment Brushes: Sheafified × Stackified × Condensified
# ═══════════════════════════════════════════════════════════════════════════════
#
# Three semantic lenses for coloring 2-Para rewriting gadgets:
#
#   ┌─────────────────────────────────────────────────────────────────────────────┐
#   │  METATHEORY MOMENT BRUSHES                                                  │
#   │  ══════════════════════════                                                 │
#   │                                                                             │
#   │  SHEAFIFIED (Local → Global)                                               │
#   │  ═══════════════════════════                                               │
#   │  • Gluing: local sections agree on overlaps → global section              │
#   │  • Color: consistency witness across open covers                           │
#   │  • Moment: when local igor-patches sheafify into global coherence         │
#   │  • Brush stroke: ∫ₓ F(x) dx where F is the sheaf of colors                │
#   │                                                                             │
#   │  STACKIFIED (Descent → Equivalence)                                        │
#   │  ═════════════════════════════════                                         │
#   │  • Descent: objects over U × U → U with coherent isomorphisms             │
#   │  • Color: equivalence class of colorings (up to iso)                       │
#   │  • Moment: when parallel executions descend to same equivalence           │
#   │  • Brush stroke: [X/G] quotient stack coloring                            │
#   │                                                                             │
#   │  CONDENSIFIED (Compact → Complete)                                         │
#   │  ════════════════════════════════                                          │
#   │  • Solid: extremely disconnected profinite sets                            │
#   │  • Color: Haar measure on profinite color group                            │
#   │  • Moment: when ultraproduct of finite approximations stabilizes          │
#   │  • Brush stroke: lim←ₙ (ℤ/pⁿℤ × Colors) condensed coloring               │
#   │                                                                             │
#   │  2-PARA REWRITING GADGETS                                                  │
#   │  ═════════════════════════                                                 │
#   │  Objects: (Parameter, State) pairs                                         │
#   │  1-Morphisms: Rewrite rules preserving parameter                          │
#   │  2-Morphisms: Transformations between rewrite sequences                   │
#   │                                                                             │
#   │  REAFFERENCE                                                               │
#   │  ═══════════════                                                           │
#   │  Distinguishing self-generated (efferent) from external (afferent):       │
#   │  • Efferent copy: predicted sensory consequence of action                 │
#   │  • Reafference: actual sensory input matching efferent copy               │
#   │  • Exafference: sensory input NOT matching (external cause)               │
#   │                                                                             │
#   │  SUCCESSOR HAIKU                                                           │
#   │  ═══════════════                                                           │
#   │  The minimal 5-7-5 encoding of state transition:                          │
#   │    Line 1 (5): Current state essence                                      │
#   │    Line 2 (7): Transformation applied                                     │
#   │    Line 3 (5): Successor state essence                                    │
#   └─────────────────────────────────────────────────────────────────────────────┘

export MomentBrush, SheafifiedBrush, StackifiedBrush, CondensifiedBrush
export ParaRewriteGadget, TwoParaGadget, apply_gadget, compose_gadgets
export Reafference, ReafferenceState, efferent_copy, is_reafferent, is_exafferent
export SuccessorHaiku, haiku_transition, haiku_encode
export MetatheoryTriple, apply_triple, deconflict_brushes
export edge_random_access, TwoMonadTorial

using Random

# Import from igor_seeds.jl
include("igor_seeds.jl")

# ═══════════════════════════════════════════════════════════════════════════════
# MomentBrush: Abstract type for metatheoretic coloring
# ═══════════════════════════════════════════════════════════════════════════════

"""
    MomentBrush

Abstract type for metatheoretic moment brushes that color interactions.
Each brush provides a different semantic lens on the same underlying structure.
"""
abstract type MomentBrush end

"""
Color type: RGB with alpha for blending
"""
struct BrushColor
    r::Float32
    g::Float32
    b::Float32
    α::Float32  # Blend weight
end

BrushColor(r, g, b) = BrushColor(Float32(r), Float32(g), Float32(b), Float32(1.0))

function blend(c1::BrushColor, c2::BrushColor)
    w1, w2 = c1.α, c2.α
    total = w1 + w2
    if total < 0.001
        return BrushColor(0.5, 0.5, 0.5, 0.0)
    end
    BrushColor(
        (c1.r * w1 + c2.r * w2) / total,
        (c1.g * w1 + c2.g * w2) / total,
        (c1.b * w1 + c2.b * w2) / total,
        min(1.0, total)
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# SheafifiedBrush: Local-to-Global Consistency
# ═══════════════════════════════════════════════════════════════════════════════

"""
    SheafifiedBrush

Colors interactions via sheaf-theoretic gluing.

A section over U is colored if:
1. Each open subset has a local coloring
2. Colorings agree on overlaps (consistency)
3. Global section exists iff gluing succeeds

The moment occurs when local patches achieve global coherence.
"""
struct SheafifiedBrush <: MomentBrush
    seed::UInt64
    open_cover::Vector{UnitRange{Int}}  # Indices covered by each open set
    local_sections::Vector{BrushColor}  # Color for each open set
    overlap_matrix::Matrix{Bool}        # Which opens overlap
end

function SheafifiedBrush(seed::UInt64, n_opens::Int, total_size::Int)
    rng_state = seed
    
    # Generate random open cover
    cover = UnitRange{Int}[]
    for i in 1:n_opens
        rng_state = mix64(rng_state)
        start = 1 + Int(rng_state % UInt64(total_size))
        rng_state = mix64(rng_state)
        len = 1 + Int(rng_state % UInt64(total_size ÷ 2))
        push!(cover, start:min(start + len - 1, total_size))
    end
    
    # Local section colors
    sections = BrushColor[]
    for i in 1:n_opens
        rng_state = mix64(rng_state)
        r = Float32((rng_state % 256) / 255.0)
        rng_state = mix64(rng_state)
        g = Float32((rng_state % 256) / 255.0)
        rng_state = mix64(rng_state)
        b = Float32((rng_state % 256) / 255.0)
        push!(sections, BrushColor(r, g, b))
    end
    
    # Overlap matrix
    overlaps = falses(n_opens, n_opens)
    for i in 1:n_opens, j in 1:n_opens
        if i != j && !isempty(intersect(cover[i], cover[j]))
            overlaps[i, j] = true
        end
    end
    
    SheafifiedBrush(seed, cover, sections, overlaps)
end

"""
Check if local sections are consistent (agree on overlaps)
"""
function is_consistent(brush::SheafifiedBrush; tolerance::Float64=0.3)
    n = length(brush.local_sections)
    for i in 1:n, j in 1:n
        if brush.overlap_matrix[i, j]
            c1, c2 = brush.local_sections[i], brush.local_sections[j]
            dist = sqrt((c1.r - c2.r)^2 + (c1.g - c2.g)^2 + (c1.b - c2.b)^2)
            if dist > tolerance
                return false
            end
        end
    end
    return true
end

"""
Attempt to glue local sections into global section
"""
function glue_sections(brush::SheafifiedBrush)
    if !is_consistent(brush)
        return nothing  # No global section exists
    end
    
    # Average all local sections weighted by coverage
    total_weight = 0.0
    r, g, b = 0.0, 0.0, 0.0
    for (i, sec) in enumerate(brush.local_sections)
        weight = length(brush.open_cover[i])
        r += sec.r * weight
        g += sec.g * weight
        b += sec.b * weight
        total_weight += weight
    end
    
    if total_weight > 0
        return BrushColor(r / total_weight, g / total_weight, b / total_weight)
    else
        return BrushColor(0.5, 0.5, 0.5)
    end
end

"""
The sheafified moment: when gluing succeeds
"""
function sheafified_moment(brush::SheafifiedBrush)
    global_section = glue_sections(brush)
    (
        success = !isnothing(global_section),
        color = something(global_section, BrushColor(0.5, 0.5, 0.5)),
        consistency = is_consistent(brush)
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# StackifiedBrush: Descent and Equivalence Classes
# ═══════════════════════════════════════════════════════════════════════════════

"""
    StackifiedBrush

Colors interactions via stack-theoretic descent.

Objects are colored up to isomorphism:
1. G-action on color space
2. Descent datum: cocycle condition on overlaps
3. Color = equivalence class [c] under G-action

The moment occurs when parallel executions descend to same class.
"""
struct StackifiedBrush <: MomentBrush
    seed::UInt64
    group_order::Int              # |G| for the acting group
    representatives::Vector{BrushColor}  # Color class representatives
    action_matrix::Matrix{Int}    # g ⋅ c → action_matrix[g, c]
end

function StackifiedBrush(seed::UInt64, group_order::Int, n_colors::Int)
    rng_state = seed
    
    # Representatives for each equivalence class
    reps = BrushColor[]
    for i in 1:n_colors
        rng_state = mix64(rng_state)
        r = Float32((rng_state % 256) / 255.0)
        rng_state = mix64(rng_state)
        g = Float32((rng_state % 256) / 255.0)
        rng_state = mix64(rng_state)
        b = Float32((rng_state % 256) / 255.0)
        push!(reps, BrushColor(r, g, b))
    end
    
    # Random group action (permutation)
    action = zeros(Int, group_order, n_colors)
    for g in 1:group_order
        perm = collect(1:n_colors)
        for i in n_colors:-1:2
            rng_state = mix64(rng_state ⊻ UInt64(g))
            j = 1 + Int(rng_state % UInt64(i))
            perm[i], perm[j] = perm[j], perm[i]
        end
        action[g, :] = perm
    end
    
    StackifiedBrush(seed, group_order, reps, action)
end

"""
Apply group element to color index
"""
function act(brush::StackifiedBrush, g::Int, c::Int)
    g_idx = mod1(g, brush.group_order)
    c_idx = mod1(c, length(brush.representatives))
    brush.action_matrix[g_idx, c_idx]
end

"""
Find equivalence class (orbit) of a color
"""
function equivalence_class(brush::StackifiedBrush, c::Int)
    orbit = Set{Int}([c])
    for g in 1:brush.group_order
        push!(orbit, act(brush, g, c))
    end
    return collect(orbit)
end

"""
Check if two colors are equivalent (in same orbit)
"""
function are_equivalent(brush::StackifiedBrush, c1::Int, c2::Int)
    c2 in equivalence_class(brush, c1)
end

"""
Descent color: representative of the equivalence class
"""
function descent_color(brush::StackifiedBrush, c::Int)
    orbit = equivalence_class(brush, c)
    rep_idx = minimum(orbit)  # Canonical representative
    brush.representatives[mod1(rep_idx, length(brush.representatives))]
end

"""
The stackified moment: when parallel paths descend to same class
"""
function stackified_moment(brush::StackifiedBrush, path1::Vector{Int}, path2::Vector{Int})
    if isempty(path1) || isempty(path2)
        return (success = false, color = BrushColor(0.5, 0.5, 0.5), equivalent = false)
    end
    
    c1 = path1[end]
    c2 = path2[end]
    equiv = are_equivalent(brush, c1, c2)
    
    (
        success = equiv,
        color = descent_color(brush, c1),
        equivalent = equiv
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# CondensifiedBrush: Profinite Completion
# ═══════════════════════════════════════════════════════════════════════════════

"""
    CondensifiedBrush

Colors interactions via condensed mathematics (Clausen-Scholze style).

The color space is completed to a profinite/condensed structure:
1. Approximate by finite quotients ℤ/pⁿℤ
2. Take projective limit
3. Haar measure on profinite color group

The moment occurs when the ultraproduct stabilizes.
"""
struct CondensifiedBrush <: MomentBrush
    seed::UInt64
    prime::Int                    # p for ℤ/pⁿℤ approximations
    levels::Int                   # Number of approximation levels
    approximations::Vector{Vector{BrushColor}}  # Colors at each level
end

function CondensifiedBrush(seed::UInt64; prime::Int=3, levels::Int=5)
    rng_state = seed
    
    approx = Vector{Vector{BrushColor}}()
    
    for n in 1:levels
        # pⁿ colors at level n
        n_colors = prime^n
        level_colors = BrushColor[]
        
        for i in 1:n_colors
            # Color derived from seed and level
            rng_state = mix64(rng_state ⊻ UInt64(n) ⊻ UInt64(i))
            
            # Quantize to level n precision
            r = Float32(floor((rng_state % 256) / (256 / prime^n)) / prime^n)
            rng_state = mix64(rng_state)
            g = Float32(floor((rng_state % 256) / (256 / prime^n)) / prime^n)
            rng_state = mix64(rng_state)
            b = Float32(floor((rng_state % 256) / (256 / prime^n)) / prime^n)
            
            push!(level_colors, BrushColor(r, g, b))
        end
        
        push!(approx, level_colors)
    end
    
    CondensifiedBrush(seed, prime, levels, approx)
end

"""
Project color at level n+1 down to level n
"""
function project_down(brush::CondensifiedBrush, c_idx::Int, from_level::Int)
    if from_level <= 1
        return 1
    end
    
    # Coarser index
    mod1(c_idx, brush.prime^(from_level - 1))
end

"""
Check if projective system is consistent
"""
function is_projective(brush::CondensifiedBrush)
    for n in 2:brush.levels
        for c in 1:length(brush.approximations[n])
            c_proj = project_down(brush, c, n)
            high_color = brush.approximations[n][c]
            low_color = brush.approximations[n-1][c_proj]
            
            # Check colors are close (up to quantization)
            dist = sqrt((high_color.r - low_color.r)^2 + 
                       (high_color.g - low_color.g)^2 + 
                       (high_color.b - low_color.b)^2)
            if dist > 1.0 / brush.prime^(n-1)
                return false
            end
        end
    end
    return true
end

"""
Limit color (finest approximation)
"""
function limit_color(brush::CondensifiedBrush, idx::Int)
    finest = brush.approximations[end]
    finest[mod1(idx, length(finest))]
end

"""
The condensified moment: when ultraproduct stabilizes
"""
function condensified_moment(brush::CondensifiedBrush, sequence::Vector{Int})
    if isempty(sequence)
        return (success = false, color = BrushColor(0.5, 0.5, 0.5), stable = false)
    end
    
    # Check for stabilization in projective limit
    projected_sequence = [project_down(brush, s, brush.levels) for s in sequence]
    stable = length(unique(projected_sequence[max(1, end-3):end])) == 1
    
    (
        success = stable && is_projective(brush),
        color = limit_color(brush, sequence[end]),
        stable = stable
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# 2-Para Rewriting Gadget
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ParaRewriteGadget{P, S}

A parameterized rewriting gadget:
- P: Parameter type (determines rewrite behavior)
- S: State type (being rewritten)

Morphism: (p, s) → (p, s') where p is preserved
"""
struct ParaRewriteGadget{P, S}
    parameter::P
    rewrite::Function  # (P, S) → S
    color::BrushColor
    seed::UInt64
end

function ParaRewriteGadget(param, rewrite_fn; seed::UInt64=GAY_IGOR_SEED)
    rng = mix64(seed)
    color = BrushColor(
        Float32((rng % 256) / 255),
        Float32((mix64(rng) % 256) / 255),
        Float32((mix64(mix64(rng)) % 256) / 255)
    )
    ParaRewriteGadget(param, rewrite_fn, color, seed)
end

function apply_gadget(g::ParaRewriteGadget{P, S}, state::S) where {P, S}
    (g.parameter, g.rewrite(g.parameter, state))
end

"""
    TwoParaGadget

2-categorical gadget with:
- Objects: (Parameter, State) pairs
- 1-Morphisms: Rewrite rules
- 2-Morphisms: Transformations between rewrites
"""
struct TwoParaGadget{P, S}
    base_gadgets::Vector{ParaRewriteGadget{P, S}}
    two_morphisms::Dict{Tuple{Int, Int}, Function}  # (g1, g2) → transformation
    seed::UInt64
end

function TwoParaGadget(gadgets::Vector{ParaRewriteGadget{P, S}}; seed::UInt64=GAY_IGOR_SEED) where {P, S}
    # Generate 2-morphisms between gadgets
    two_morphs = Dict{Tuple{Int, Int}, Function}()
    rng = seed
    
    for i in 1:length(gadgets), j in 1:length(gadgets)
        if i != j
            rng = mix64(rng ⊻ UInt64(i) ⊻ (UInt64(j) << 32))
            # 2-morphism: blend the rewrites
            blend_weight = (rng % 1000) / 1000.0
            two_morphs[(i, j)] = (p, s) -> begin
                s1 = gadgets[i].rewrite(p, s)
                s2 = gadgets[j].rewrite(p, s)
                # Type-appropriate blending (for numeric states)
                if s1 isa Number && s2 isa Number
                    blend_weight * s1 + (1 - blend_weight) * s2
                else
                    blend_weight < 0.5 ? s1 : s2
                end
            end
        end
    end
    
    TwoParaGadget(gadgets, two_morphs, seed)
end

"""
Apply a 2-morphism to transform between gadget applications
"""
function apply_two_morphism(tpg::TwoParaGadget{P, S}, i::Int, j::Int, param::P, state::S) where {P, S}
    if haskey(tpg.two_morphisms, (i, j))
        tpg.two_morphisms[(i, j)](param, state)
    else
        state  # Identity 2-morphism
    end
end

"""
Edge-random-access: O(1) access to any gadget by index
"""
function edge_random_access(tpg::TwoParaGadget, edge_index::Int)
    n = length(tpg.base_gadgets)
    gadget_idx = mod1(edge_index, n)
    tpg.base_gadgets[gadget_idx]
end

# ═══════════════════════════════════════════════════════════════════════════════
# Reafference: Self-Generated vs External Signals
# ═══════════════════════════════════════════════════════════════════════════════

"""
    Reafference

Distinguish between:
- Reafference: sensory input matching efferent copy (self-caused)
- Exafference: sensory input NOT matching (externally caused)

This is the "igor" (self) vs "not-igor" (other) distinction at sensory level.
"""
struct Reafference{S}
    efferent_copy::S      # Predicted consequence of action
    actual_input::S       # Actual sensory input
    threshold::Float64    # Match tolerance
end

"""
The efferent copy: predicted sensory consequence
"""
efferent_copy(r::Reafference) = r.efferent_copy

"""
Is this reafferent? (self-caused, matches prediction)
"""
function is_reafferent(r::Reafference{S}) where S
    if r.efferent_copy isa Number && r.actual_input isa Number
        abs(r.efferent_copy - r.actual_input) < r.threshold
    elseif r.efferent_copy isa AbstractVector && r.actual_input isa AbstractVector
        norm(r.efferent_copy - r.actual_input) < r.threshold
    else
        r.efferent_copy == r.actual_input
    end
end

"""
Is this exafferent? (externally caused, differs from prediction)
"""
is_exafferent(r::Reafference) = !is_reafferent(r)

"""
Reafference coloring: green = self, red = other
"""
function reafference_color(r::Reafference)
    if is_reafferent(r)
        BrushColor(0.0, 0.8, 0.2)  # Green (igor)
    else
        BrushColor(0.8, 0.2, 0.0)  # Red (not-igor)
    end
end

"""
    ReafferenceState

Accumulator for reafference tracking across multiple observations.
"""
mutable struct ReafferenceState{S}
    predictions::Vector{S}
    observations::Vector{S}
    reafferent_count::Int
    exafferent_count::Int
    threshold::Float64
end

function ReafferenceState{S}(; threshold::Float64=0.1) where S
    ReafferenceState(S[], S[], 0, 0, threshold)
end

function observe!(state::ReafferenceState{S}, prediction::S, observation::S) where S
    push!(state.predictions, prediction)
    push!(state.observations, observation)
    
    r = Reafference(prediction, observation, state.threshold)
    if is_reafferent(r)
        state.reafferent_count += 1
    else
        state.exafferent_count += 1
    end
end

function reafference_ratio(state::ReafferenceState)
    total = state.reafferent_count + state.exafferent_count
    total > 0 ? state.reafferent_count / total : 0.5
end

# ═══════════════════════════════════════════════════════════════════════════════
# Successor Haiku: Minimal State Transition Encoding
# ═══════════════════════════════════════════════════════════════════════════════

"""
    SuccessorHaiku

A 5-7-5 syllable encoding of state transition:
- Line 1 (5 syllables): Current state essence
- Line 2 (7 syllables): Transformation applied  
- Line 3 (5 syllables): Successor state essence

This is the "minimal poem" of the transition, capturing the moment.
"""
struct SuccessorHaiku
    line1::String  # 5 syllables: current state
    line2::String  # 7 syllables: transformation
    line3::String  # 5 syllables: successor state
    color::BrushColor
end

"""
Encode a numeric value as haiku-compatible syllables
"""
function syllable_encode(value, n_syllables::Int)
    words_by_syllable = Dict(
        1 => ["sun", "moon", "star", "wave", "wind", "light", "dark", "fire", "ice", "leaf"],
        2 => ["flowing", "dancing", "rising", "falling", "spinning", "floating", "glowing", "fading"],
        3 => ["transforming", "becoming", "emerging", "dissolving", "reflecting", "unfolding"]
    )
    
    # Hash value to word selection
    h = hash(value)
    result = String[]
    remaining = n_syllables
    
    while remaining > 0
        if remaining >= 3 && (h % 3 == 0)
            word_list = words_by_syllable[3]
            push!(result, word_list[1 + Int(h % length(word_list))])
            remaining -= 3
        elseif remaining >= 2 && (h % 2 == 0 || remaining < 3)
            word_list = words_by_syllable[2]
            push!(result, word_list[1 + Int(h % length(word_list))])
            remaining -= 2
        else
            word_list = words_by_syllable[1]
            push!(result, word_list[1 + Int(h % length(word_list))])
            remaining -= 1
        end
        h = hash(h)
    end
    
    join(result, " ")
end

"""
Generate a haiku for a state transition
"""
function haiku_transition(current_state, transform_name::String, next_state; seed::UInt64=GAY_IGOR_SEED)
    line1 = syllable_encode(current_state, 5)
    line2 = transform_name * " " * syllable_encode(transform_name, 7 - length(split(transform_name)))
    line3 = syllable_encode(next_state, 5)
    
    # Color from transition hash
    h = mix64(seed ⊻ hash(current_state) ⊻ hash(next_state))
    color = BrushColor(
        Float32((h % 256) / 255),
        Float32((mix64(h) % 256) / 255),
        Float32((mix64(mix64(h)) % 256) / 255)
    )
    
    SuccessorHaiku(line1, line2, line3, color)
end

function Base.show(io::IO, h::SuccessorHaiku)
    println(io, h.line1)
    println(io, "  " * h.line2)
    print(io, h.line3)
end

# ═══════════════════════════════════════════════════════════════════════════════
# MetatheoryTriple: All Three Brushes Combined
# ═══════════════════════════════════════════════════════════════════════════════

"""
    MetatheoryTriple

The three metatheoretic brushes combined:
- Sheafified: local-to-global consistency
- Stackified: descent to equivalence classes
- Condensified: profinite completion

Applying all three gives a "triangulated" coloring of the interaction.
"""
struct MetatheoryTriple
    sheafified::SheafifiedBrush
    stackified::StackifiedBrush
    condensified::CondensifiedBrush
    seed::UInt64
end

function MetatheoryTriple(; seed::UInt64=GAY_IGOR_SEED, size::Int=16)
    MetatheoryTriple(
        SheafifiedBrush(seed, size ÷ 2, size),
        StackifiedBrush(mix64(seed), 3, size),
        CondensifiedBrush(mix64(mix64(seed)); levels=4),
        seed
    )
end

"""
Apply all three brushes and combine results
"""
function apply_triple(triple::MetatheoryTriple, paths::Vector{Vector{Int}})
    # Sheafified moment
    sheaf_moment = sheafified_moment(triple.sheafified)
    
    # Stackified moment (compare first two paths if available)
    stack_moment = if length(paths) >= 2
        stackified_moment(triple.stackified, paths[1], paths[2])
    else
        (success = false, color = BrushColor(0.5, 0.5, 0.5), equivalent = false)
    end
    
    # Condensified moment
    cond_moment = if !isempty(paths)
        condensified_moment(triple.condensified, paths[1])
    else
        (success = false, color = BrushColor(0.5, 0.5, 0.5), stable = false)
    end
    
    # Blend all three colors
    blended = blend(sheaf_moment.color, blend(stack_moment.color, cond_moment.color))
    
    (
        sheafified = sheaf_moment,
        stackified = stack_moment,
        condensified = cond_moment,
        combined_color = blended,
        all_success = sheaf_moment.success && stack_moment.success && cond_moment.success
    )
end

"""
Deconflict multiple brush applications for parallel execution
"""
function deconflict_brushes(triple::MetatheoryTriple, parallel_results::Vector)
    # XOR fingerprint for order-invariance
    fingerprint = UInt32(0)
    for result in parallel_results
        c = result.combined_color
        fingerprint ⊻= UInt32(round(c.r * 255)) << 16
        fingerprint ⊻= UInt32(round(c.g * 255)) << 8
        fingerprint ⊻= UInt32(round(c.b * 255))
    end
    
    # All must agree for deconfliction
    colors_match = all(r -> r.combined_color ≈ parallel_results[1].combined_color for r in parallel_results)
    
    (
        fingerprint = fingerprint,
        deconflicted = colors_match,
        consensus_color = parallel_results[1].combined_color
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# 2-Monad 2-Torial: Tutorial Structure for 2-Monads
# ═══════════════════════════════════════════════════════════════════════════════

"""
    TwoMonadTorial

A 2-categorical tutorial structure:
- Lessons: 0-cells (objects)
- Exercises: 1-cells (morphisms between lessons)
- Hints: 2-cells (transformations between exercises)

This is the "sufficiently fast edge-random-accessable" structure
for learning 2-monads.
"""
struct TwoMonadTorial{T}
    lessons::Vector{T}                        # 0-cells
    exercises::Dict{Tuple{Int,Int}, Function} # 1-cells: lesson → lesson
    hints::Dict{Tuple{Int,Int,Int,Int}, String} # 2-cells: exercise → exercise
    gadgets::TwoParaGadget{Symbol, T}         # Rewriting gadgets
    seed::UInt64
end

function TwoMonadTorial(lessons::Vector{T}; seed::UInt64=GAY_IGOR_SEED) where T
    n = length(lessons)
    rng = seed
    
    # Exercises: morphisms between lessons
    exercises = Dict{Tuple{Int,Int}, Function}()
    for i in 1:n, j in 1:n
        if i != j
            rng = mix64(rng)
            exercises[(i, j)] = x -> x  # Placeholder transform
        end
    end
    
    # Hints: 2-morphisms between exercises
    hints = Dict{Tuple{Int,Int,Int,Int}, String}()
    for i in 1:n, j in 1:n, k in 1:n, l in 1:n
        if haskey(exercises, (i, j)) && haskey(exercises, (k, l))
            if i == k  # Same source
                rng = mix64(rng)
                hints[(i, j, k, l)] = "Hint: compare paths from lesson $i"
            end
        end
    end
    
    # Create rewriting gadgets for each lesson
    base_gadgets = [ParaRewriteGadget(:lesson, (p, s) -> s; seed=mix64(seed ⊻ UInt64(i))) 
                    for i in 1:n]
    gadgets = TwoParaGadget(base_gadgets; seed=seed)
    
    TwoMonadTorial(lessons, exercises, hints, gadgets, seed)
end

"""
Edge-random-access to tutorial content
"""
function edge_random_access(torial::TwoMonadTorial, edge_idx::Int)
    n = length(torial.lessons)
    # Map edge index to (source, target) pair
    source = 1 + (edge_idx - 1) ÷ n
    target = 1 + (edge_idx - 1) % n
    
    if haskey(torial.exercises, (source, target))
        (
            source_lesson = torial.lessons[mod1(source, n)],
            target_lesson = torial.lessons[mod1(target, n)],
            exercise = torial.exercises[(source, target)],
            gadget = edge_random_access(torial.gadgets, edge_idx)
        )
    else
        nothing
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════════════════

function demo_metatheory_brushes()
    println()
    println("╔" * "═" ^ 62 * "╗")
    println("║  METATHEORY MOMENT BRUSHES                                    ║")
    println("║  Sheafified × Stackified × Condensified                       ║")
    println("╚" * "═" ^ 62 * "╝")
    println()
    
    seed = GAY_IGOR_SEED
    
    # Sheafified brush
    println("SHEAFIFIED BRUSH (Local → Global)")
    println("─" ^ 40)
    sheaf = SheafifiedBrush(seed, 5, 20)
    moment = sheafified_moment(sheaf)
    c = moment.color
    r, g, b = Int.(round.((c.r, c.g, c.b) .* 255))
    println("  Consistent: $(moment.consistency)")
    println("  Glued: $(moment.success)")
    println("  Color: \e[48;2;$(r);$(g);$(b)m    \e[0m")
    println()
    
    # Stackified brush
    println("STACKIFIED BRUSH (Descent → Equivalence)")
    println("─" ^ 40)
    stack = StackifiedBrush(seed, 3, 6)
    path1 = [1, 3, 5]
    path2 = [2, 4, 6]
    stack_moment = stackified_moment(stack, path1, path2)
    c = stack_moment.color
    r, g, b = Int.(round.((c.r, c.g, c.b) .* 255))
    println("  Paths equivalent: $(stack_moment.equivalent)")
    println("  Descent color: \e[48;2;$(r);$(g);$(b)m    \e[0m")
    println()
    
    # Condensified brush
    println("CONDENSIFIED BRUSH (Compact → Complete)")
    println("─" ^ 40)
    cond = CondensifiedBrush(seed; prime=3, levels=4)
    seq = [1, 4, 13, 40, 40, 40]  # Stabilizing sequence
    cond_moment = condensified_moment(cond, seq)
    c = cond_moment.color
    r, g, b = Int.(round.((c.r, c.g, c.b) .* 255))
    println("  Projective: $(is_projective(cond))")
    println("  Stable: $(cond_moment.stable)")
    println("  Limit color: \e[48;2;$(r);$(g);$(b)m    \e[0m")
    println()
    
    # Combined triple
    println("METATHEORY TRIPLE")
    println("─" ^ 40)
    triple = MetatheoryTriple(; seed=seed)
    result = apply_triple(triple, [path1, path2, seq])
    c = result.combined_color
    r, g, b = Int.(round.((c.r, c.g, c.b) .* 255))
    println("  All success: $(result.all_success)")
    println("  Combined color: \e[48;2;$(r);$(g);$(b)m    \e[0m")
    println()
    
    # Reafference
    println("REAFFERENCE (Self vs Other)")
    println("─" ^ 40)
    state = ReafferenceState{Float64}(; threshold=0.1)
    for i in 1:10
        pred = Float64(i)
        obs = Float64(i) + (i % 3 == 0 ? 0.5 : 0.01)  # Some exafferent
        observe!(state, pred, obs)
    end
    println("  Reafferent: $(state.reafferent_count), Exafferent: $(state.exafferent_count)")
    println("  Self-ratio: $(round(reafference_ratio(state), digits=2))")
    println()
    
    # Successor Haiku
    println("SUCCESSOR HAIKU")
    println("─" ^ 40)
    haiku = haiku_transition(42, "transforms", 69)
    println(haiku)
    println()
    
    # 2-Para Gadget
    println("2-PARA REWRITING GADGET")
    println("─" ^ 40)
    gadgets = [
        ParaRewriteGadget(:add, (p, s) -> s + 1),
        ParaRewriteGadget(:mul, (p, s) -> s * 2),
        ParaRewriteGadget(:sub, (p, s) -> s - 1)
    ]
    tpg = TwoParaGadget(gadgets)
    println("  Gadgets: $(length(tpg.base_gadgets))")
    println("  Edge-access(5): $(edge_random_access(tpg, 5).parameter)")
    println("  Apply gadget 1 to 10: $(apply_gadget(gadgets[1], 10))")
    println("  2-morphism (1→2) on 10: $(apply_two_morphism(tpg, 1, 2, :blend, 10))")
    println()
    
    println("🏳️‍🌈 Metatheory Brushes Complete")
end

if abspath(PROGRAM_FILE) == @__FILE__
    demo_metatheory_brushes()
end
