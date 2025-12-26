# Gamut-Constrained Learnable Color Spaces with Subobject Classification
#
# Implements categorical gamut theory: sRGB ⊂ P3 ⊂ Rec2020 as a lattice
# of subobject classifiers, enabling characteristic morphisms for
# gamut membership and pullback operations for color recovery.

module GamutLearnable

__precompile__(false)  # Disable precompilation due to complex constructor patterns

using Colors, ColorTypes

export GamutConstraint, GaySRGBGamut, GayP3Gamut, GayRec2020Gamut
export LearnableGamutMap, GamutParameters
export map_to_gamut, is_in_gamut, gamut_distance
export learn_gamut_map!, gamut_loss, chroma_preservation_loss
export GayChain, chain_to_gamut, verify_chain_in_gamut, process_gay_chain
export enzyme_gamut_gradient, enzyme_learn_gamut!

# Gamut Subobject Classifier exports
export GamutTruth, GamutSubobjectClassifier
export characteristic_morphism, gamut_pullback, probe_gamut_subobject
export WorldGamutClassifier, world_gamut_classifier, gamut_meet, gamut_join

# ═══════════════════════════════════════════════════════════════════════════
# Gamut Constraint Hierarchy: sRGB ⊂ P3 ⊂ Rec2020
# ═══════════════════════════════════════════════════════════════════════════

"""
    GamutConstraint

Abstract type for gamut constraints. Each gamut defines a subobject
in the category of color spaces, with inclusion morphisms forming
a lattice structure.
"""
abstract type GamutConstraint end

"""
    GaySRGBGamut <: GamutConstraint

sRGB gamut - the smallest standard gamut.
The bottom of our gamut lattice.
"""
struct GaySRGBGamut <: GamutConstraint end

"""
    GayP3Gamut <: GamutConstraint

Display P3 gamut - intermediate gamut used by Apple devices.
"""
struct GayP3Gamut <: GamutConstraint end

"""
    GayRec2020Gamut <: GamutConstraint

Rec.2020 gamut - the widest standard gamut.
The top of our gamut lattice.
"""
struct GayRec2020Gamut <: GamutConstraint end

# Gamut hierarchy: sRGB ⊂ P3 ⊂ Rec2020
Base.issubset(::GaySRGBGamut, ::GaySRGBGamut) = true
Base.issubset(::GaySRGBGamut, ::GayP3Gamut) = true
Base.issubset(::GaySRGBGamut, ::GayRec2020Gamut) = true
Base.issubset(::GayP3Gamut, ::GayP3Gamut) = true
Base.issubset(::GayP3Gamut, ::GayRec2020Gamut) = true
Base.issubset(::GayRec2020Gamut, ::GayRec2020Gamut) = true
Base.issubset(::GayP3Gamut, ::GaySRGBGamut) = false
Base.issubset(::GayRec2020Gamut, ::GaySRGBGamut) = false
Base.issubset(::GayRec2020Gamut, ::GayP3Gamut) = false

# ═══════════════════════════════════════════════════════════════════════════
# Gamut Truth Values (Subobject Classifier)
# ═══════════════════════════════════════════════════════════════════════════

"""
    GamutTruth

The truth value for gamut membership, forming the subobject classifier Ω.
Contains both a boolean membership and a distance metric for gradual falloff.

# Fields
- `in_gamut::Bool`: Whether the color is strictly within gamut
- `distance::Float64`: Distance from gamut boundary (negative if inside, positive if outside)
"""
struct GamutTruth
    in_gamut::Bool
    distance::Float64
end

GamutTruth(in_gamut::Bool) = GamutTruth(in_gamut, in_gamut ? -1.0 : 1.0)

# Truth value lattice operations
Base.:(&)(a::GamutTruth, b::GamutTruth) = GamutTruth(a.in_gamut && b.in_gamut, max(a.distance, b.distance))
Base.:(|)(a::GamutTruth, b::GamutTruth) = GamutTruth(a.in_gamut || b.in_gamut, min(a.distance, b.distance))
Base.:(!)(a::GamutTruth) = GamutTruth(!a.in_gamut, -a.distance)

# Ordering on truth values
Base.isless(a::GamutTruth, b::GamutTruth) = a.distance < b.distance

"""
    GamutSubobjectClassifier{G<:GamutConstraint}

The subobject classifier for a specific gamut G.
Maps colors to their truth value in Ω = GamutTruth.

This is χ: Color → Ω such that for subobject i: G → ColorSpace,
χ classifies membership in G.
"""
struct GamutSubobjectClassifier{G<:GamutConstraint}
    gamut::G
end

"""
    characteristic_morphism(classifier::GamutSubobjectClassifier, color::RGB)

The characteristic morphism χ_G: Color → Ω.
Returns the GamutTruth for whether color is in gamut G.
"""
function characteristic_morphism(classifier::GamutSubobjectClassifier{GaySRGBGamut}, color::RGB)
    # sRGB: all components in [0,1]
    r, g, b = Float64(color.r), Float64(color.g), Float64(color.b)
    
    # Distance from gamut boundary (negative = inside)
    dist_r = max(0.0 - r, r - 1.0, 0.0)
    dist_g = max(0.0 - g, g - 1.0, 0.0)
    dist_b = max(0.0 - b, b - 1.0, 0.0)
    distance = sqrt(dist_r^2 + dist_g^2 + dist_b^2)
    
    in_gamut = (0.0 <= r <= 1.0) && (0.0 <= g <= 1.0) && (0.0 <= b <= 1.0)
    
    # If inside, compute distance to boundary as negative
    if in_gamut
        min_to_edge = min(r, 1-r, g, 1-g, b, 1-b)
        distance = -min_to_edge
    end
    
    return GamutTruth(in_gamut, distance)
end

function characteristic_morphism(classifier::GamutSubobjectClassifier{GayP3Gamut}, color::RGB)
    # P3 has slightly wider gamut than sRGB
    # For simplicity, we use a scaling factor (actual implementation would use proper matrices)
    r, g, b = Float64(color.r), Float64(color.g), Float64(color.b)
    
    # P3 allows values slightly outside [0,1] in sRGB encoding
    # Approximate P3 boundary
    margin = 0.1  # P3 extends ~10% beyond sRGB
    
    dist_r = max(-margin - r, r - (1.0 + margin), 0.0)
    dist_g = max(-margin - g, g - (1.0 + margin), 0.0)
    dist_b = max(-margin - b, b - (1.0 + margin), 0.0)
    distance = sqrt(dist_r^2 + dist_g^2 + dist_b^2)
    
    in_gamut = (-margin <= r <= 1.0 + margin) && 
               (-margin <= g <= 1.0 + margin) && 
               (-margin <= b <= 1.0 + margin)
    
    if in_gamut
        min_to_edge = min(r + margin, 1 + margin - r, g + margin, 1 + margin - g, b + margin, 1 + margin - b)
        distance = -min_to_edge
    end
    
    return GamutTruth(in_gamut, distance)
end

function characteristic_morphism(classifier::GamutSubobjectClassifier{GayRec2020Gamut}, color::RGB)
    # Rec2020 has the widest gamut
    r, g, b = Float64(color.r), Float64(color.g), Float64(color.b)
    
    # Rec2020 extends ~20% beyond sRGB
    margin = 0.2
    
    dist_r = max(-margin - r, r - (1.0 + margin), 0.0)
    dist_g = max(-margin - g, g - (1.0 + margin), 0.0)
    dist_b = max(-margin - b, b - (1.0 + margin), 0.0)
    distance = sqrt(dist_r^2 + dist_g^2 + dist_b^2)
    
    in_gamut = (-margin <= r <= 1.0 + margin) && 
               (-margin <= g <= 1.0 + margin) && 
               (-margin <= b <= 1.0 + margin)
    
    if in_gamut
        min_to_edge = min(r + margin, 1 + margin - r, g + margin, 1 + margin - g, b + margin, 1 + margin - b)
        distance = -min_to_edge
    end
    
    return GamutTruth(in_gamut, distance)
end

"""
    gamut_pullback(classifier::GamutSubobjectClassifier, color::RGB)

Pull back a color to be within gamut via chroma reduction.
This is the left adjoint to the subobject inclusion.
"""
function gamut_pullback(classifier::GamutSubobjectClassifier{GaySRGBGamut}, color::RGB)
    r, g, b = clamp(Float64(color.r), 0.0, 1.0), 
              clamp(Float64(color.g), 0.0, 1.0), 
              clamp(Float64(color.b), 0.0, 1.0)
    return RGB(r, g, b)
end

function gamut_pullback(classifier::GamutSubobjectClassifier{GayP3Gamut}, color::RGB)
    margin = 0.1
    r = clamp(Float64(color.r), -margin, 1.0 + margin)
    g = clamp(Float64(color.g), -margin, 1.0 + margin)
    b = clamp(Float64(color.b), -margin, 1.0 + margin)
    return RGB(r, g, b)
end

function gamut_pullback(classifier::GamutSubobjectClassifier{GayRec2020Gamut}, color::RGB)
    margin = 0.2
    r = clamp(Float64(color.r), -margin, 1.0 + margin)
    g = clamp(Float64(color.g), -margin, 1.0 + margin)
    b = clamp(Float64(color.b), -margin, 1.0 + margin)
    return RGB(r, g, b)
end

"""
    probe_gamut_subobject(gamuts::Vector{<:GamutConstraint}, color::RGB)

Probe which gamut a color belongs to in the lattice.
Returns the smallest gamut containing the color.
"""
function probe_gamut_subobject(gamuts::Vector{<:GamutConstraint}, color::RGB)
    results = Pair{GamutConstraint, GamutTruth}[]
    
    for gamut in gamuts
        classifier = GamutSubobjectClassifier(gamut)
        truth = characteristic_morphism(classifier, color)
        push!(results, gamut => truth)
    end
    
    # Sort by gamut size (smallest first that contains the color)
    for (gamut, truth) in results
        if truth.in_gamut
            return gamut
        end
    end
    
    # Return largest if none contain it
    return last(gamuts)
end

# ═══════════════════════════════════════════════════════════════════════════
# World Gamut Classifier (Lattice Structure)
# ═══════════════════════════════════════════════════════════════════════════

"""
    WorldGamutClassifier

The complete lattice of gamut classifiers.
Represents the fiber of subobject classifiers over the world of color spaces.
"""
struct WorldGamutClassifier
    classifiers::Dict{Symbol, GamutSubobjectClassifier}
    order::Vector{Symbol}  # From smallest to largest gamut
end

"""
    world_gamut_classifier()

Construct the standard world gamut classifier for sRGB ⊂ P3 ⊂ Rec2020.
"""
function world_gamut_classifier()
    classifiers = Dict{Symbol, GamutSubobjectClassifier}(
        :srgb => GamutSubobjectClassifier(GaySRGBGamut()),
        :p3 => GamutSubobjectClassifier(GayP3Gamut()),
        :rec2020 => GamutSubobjectClassifier(GayRec2020Gamut())
    )
    order = [:srgb, :p3, :rec2020]
    return WorldGamutClassifier(classifiers, order)
end

# Lattice operations on WorldGamutClassifier
function Base.getindex(wgc::WorldGamutClassifier, sym::Symbol)
    return wgc.classifiers[sym]
end

function gamut_meet(wgc::WorldGamutClassifier, a::Symbol, b::Symbol)
    # Meet = intersection = smallest gamut containing both
    idx_a = findfirst(==(a), wgc.order)
    idx_b = findfirst(==(b), wgc.order)
    return wgc.order[min(idx_a, idx_b)]
end

function gamut_join(wgc::WorldGamutClassifier, a::Symbol, b::Symbol)
    # Join = union = largest gamut
    idx_a = findfirst(==(a), wgc.order)
    idx_b = findfirst(==(b), wgc.order)
    return wgc.order[max(idx_a, idx_b)]
end

# ═══════════════════════════════════════════════════════════════════════════
# Learnable Gamut Mapping
# ═══════════════════════════════════════════════════════════════════════════

"""
    GamutParameters

Learnable parameters for gamut mapping.
"""
mutable struct GamutParameters
    chroma_scale::Float64     # Scale factor for chroma
    lightness_scale::Float64  # Scale factor for lightness
    hue_shift::Float64        # Hue rotation in degrees
    compression_gamma::Float64  # Gamut compression curve
end

GamutParameters() = GamutParameters(1.0, 1.0, 0.0, 1.0)

"""
    LearnableGamutMap

A learnable mapping for gamut compression/expansion.
"""
mutable struct LearnableGamutMap
    source::GamutConstraint
    target::GamutConstraint
    params::GamutParameters
end

function LearnableGamutMap(source::GamutConstraint, target::GamutConstraint)
    LearnableGamutMap(source, target, GamutParameters())
end

"""
    map_to_gamut(m::LearnableGamutMap, color::RGB)

Apply learnable gamut mapping to transform color.
"""
function map_to_gamut(m::LearnableGamutMap, color::RGB)
    r, g, b = Float64(color.r), Float64(color.g), Float64(color.b)
    
    # Apply compression
    γ = m.params.compression_gamma
    r_out = sign(r) * abs(r)^γ * m.params.chroma_scale
    g_out = sign(g) * abs(g)^γ * m.params.chroma_scale
    b_out = sign(b) * abs(b)^γ * m.params.chroma_scale
    
    # Clamp to target gamut
    classifier = GamutSubobjectClassifier(m.target)
    result = gamut_pullback(classifier, RGB(r_out, g_out, b_out))
    
    return result
end

"""
    is_in_gamut(color::RGB, gamut::GamutConstraint)

Check if color is within specified gamut.
"""
function is_in_gamut(color::RGB, gamut::GamutConstraint)
    classifier = GamutSubobjectClassifier(gamut)
    truth = characteristic_morphism(classifier, color)
    return truth.in_gamut
end

"""
    gamut_distance(color::RGB, gamut::GamutConstraint)

Compute signed distance to gamut boundary.
"""
function gamut_distance(color::RGB, gamut::GamutConstraint)
    classifier = GamutSubobjectClassifier(gamut)
    truth = characteristic_morphism(classifier, color)
    return truth.distance
end

# ═══════════════════════════════════════════════════════════════════════════
# Loss Functions for Learning
# ═══════════════════════════════════════════════════════════════════════════

"""
    gamut_loss(m::LearnableGamutMap, colors::Vector{<:RGB})

Compute loss for gamut mapping (colors should be in target gamut).
"""
function gamut_loss(m::LearnableGamutMap, colors::Vector{<:RGB})
    total_loss = 0.0
    for c in colors
        mapped = map_to_gamut(m, c)
        dist = gamut_distance(mapped, m.target)
        total_loss += max(0.0, dist)^2  # Penalty for out-of-gamut
    end
    return total_loss / length(colors)
end

"""
    chroma_preservation_loss(m::LearnableGamutMap, original::Vector{<:RGB}, mapped::Vector{<:RGB})

Loss encouraging chroma preservation during mapping.
"""
function chroma_preservation_loss(m::LearnableGamutMap, original::Vector{<:RGB}, mapped::Vector{<:RGB})
    total = 0.0
    for (o, m_color) in zip(original, mapped)
        # Simplified chroma as distance from gray axis
        chroma_o = sqrt(Float64(o.r - o.g)^2 + Float64(o.g - o.b)^2 + Float64(o.b - o.r)^2)
        chroma_m = sqrt(Float64(m_color.r - m_color.g)^2 + Float64(m_color.g - m_color.b)^2 + Float64(m_color.b - m_color.r)^2)
        total += (chroma_o - chroma_m)^2
    end
    return total / length(original)
end

"""
    learn_gamut_map!(m::LearnableGamutMap, colors::Vector{<:RGB}; lr=0.01, epochs=100)

Train the gamut map using gradient descent.
"""
function learn_gamut_map!(m::LearnableGamutMap, colors::Vector{<:RGB}; lr::Float64=0.01, epochs::Int=100)
    for epoch in 1:epochs
        loss = gamut_loss(m, colors)
        
        # Finite difference gradient
        ε = 1e-5
        
        m.params.compression_gamma += ε
        loss_plus = gamut_loss(m, colors)
        m.params.compression_gamma -= 2ε
        loss_minus = gamut_loss(m, colors)
        m.params.compression_gamma += ε
        
        grad = (loss_plus - loss_minus) / (2ε)
        m.params.compression_gamma -= lr * grad
        m.params.compression_gamma = clamp(m.params.compression_gamma, 0.1, 3.0)
    end
    return m
end

# ═══════════════════════════════════════════════════════════════════════════
# GayChain Integration
# ═══════════════════════════════════════════════════════════════════════════

"""
    GayChain

A chain of colors representing a trajectory through color space.
"""
struct GayChain{T<:RGB}
    colors::Vector{T}
    gamut::GamutConstraint
end

GayChain(colors::Vector{<:RGB}) = GayChain(colors, GaySRGBGamut())
GayChain(colors::Vector{<:RGB}, gamut::GamutConstraint) = GayChain{eltype(colors)}(colors, gamut)

"""
    chain_to_gamut(chain::GayChain, target::GamutConstraint)

Map entire chain to target gamut.
"""
function chain_to_gamut(chain::GayChain, target::GamutConstraint)
    mapper = LearnableGamutMap(chain.gamut, target)
    mapped = [map_to_gamut(mapper, c) for c in chain.colors]
    return GayChain(mapped, target)
end

"""
    verify_chain_in_gamut(chain::GayChain)

Verify all colors in chain are within its declared gamut.
"""
function verify_chain_in_gamut(chain::GayChain)
    return all(c -> is_in_gamut(c, chain.gamut), chain.colors)
end

"""
    process_gay_chain(chain::GayChain)

Process a GayChain, ensuring all colors are valid.
"""
function process_gay_chain(chain::GayChain)
    classifier = GamutSubobjectClassifier(chain.gamut)
    processed = [gamut_pullback(classifier, c) for c in chain.colors]
    return GayChain(processed, chain.gamut)
end

# ═══════════════════════════════════════════════════════════════════════════
# Enzyme Stubs (overridden by extension)
# ═══════════════════════════════════════════════════════════════════════════

"""
    enzyme_gamut_gradient(m::LearnableGamutMap, color::RGB)

Compute gradient via Enzyme (stub, implemented in extension).
"""
function enzyme_gamut_gradient(m::LearnableGamutMap, color::RGB)
    # Stub: returns zero gradient
    return (0.0, 0.0, 0.0, 0.0)
end

"""
    enzyme_learn_gamut!(m::LearnableGamutMap, colors::Vector{<:RGB}; kwargs...)

Learn gamut map via Enzyme autodiff (stub, implemented in extension).
"""
function enzyme_learn_gamut!(m::LearnableGamutMap, colors::Vector{<:RGB}; kwargs...)
    # Fallback to finite difference
    return learn_gamut_map!(m, colors; kwargs...)
end

end # module GamutLearnable
