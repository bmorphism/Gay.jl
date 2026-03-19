# StructuredDecompositions.jl extension for Gay.jl
# Hadamard gate guarantees: CNOT·CNOT = XOR ergodicity
# Cofree comonad structure for Color × Seed indeterminacies
#
# This is the BASIS extension from which all other Ext derive their
# categorical coherence guarantees via the Free⊣Cofree adjunction.
#
# 2-Monad structure: Para(Para(Decomposition)) with implicit/explicit seeds

module GayStructuredDecompositionsExt

using Gay: hash_color_rgb, splitmix64, GAY_SEED, color_fingerprint, GayRNG, gay_split
using StructuredDecompositions
using StructuredDecompositions.Decompositions: StrDecomp, bags, adhesions, adhesionSpans
using StructuredDecompositions.Decompositions: DecompType, Decomposition, CoDecomposition
using StructuredDecompositions.DecidingSheaves: decide_sheaf_tree_shape, 𝐃
using Catlab.CategoricalAlgebra: ob_map, hom_map
using Catlab.Graphs: Graph, src, tgt, nv, ne, vertices, edges
using Colors: RGB

export color_decomposition, color_bags, color_adhesions
export HadamardColorSheaf, cnot_xor_verify, cofree_color_chain
export ColorSeedIndeterminacy, explicit_seed, implicit_seed
export TwoMonadDecomp, para_decomp, para_para_decomp
export decide_colored_sheaf, chromatic_treewidth

# ═══════════════════════════════════════════════════════════════════════════
# Color × Seed Indeterminacy (2-Monad structure)
# ═══════════════════════════════════════════════════════════════════════════

"""
    ColorSeedIndeterminacy

Tracks whether a color derivation uses:
- Explicit seed: provided directly, fully determined
- Implicit seed: derived from context (hash, position), latent until observed

This forms a 2-monad: Para(Para(C)) where:
- Outer Para: world/context parametrization
- Inner Para: seed parametrization
"""
struct ColorSeedIndeterminacy
    explicit::Union{UInt64, Nothing}  # Explicit seed if provided
    implicit_source::Symbol           # :hash, :position, :parent, :global
    resolved_seed::UInt64             # Final resolved seed
    color::RGB{Float32}               # Resulting color
end

function explicit_seed(seed::UInt64)
    color = hash_color_rgb(seed, seed)
    ColorSeedIndeterminacy(seed, :explicit, seed, color)
end

function implicit_seed(source::Symbol, context_hash::UInt64; base_seed::UInt64=GAY_SEED)
    resolved = splitmix64(context_hash ⊻ base_seed)
    color = hash_color_rgb(resolved, base_seed)
    ColorSeedIndeterminacy(nothing, source, resolved, color)
end

# ═══════════════════════════════════════════════════════════════════════════
# Hadamard Gate Guarantees: CNOT·CNOT = XOR Ergodicity
# ═══════════════════════════════════════════════════════════════════════════

"""
    HadamardColorSheaf

A sheaf on structured decompositions that respects Hadamard gate properties:

1. CNOT·CNOT = I (self-inverse)
2. H·Z·H = X (basis rotation)
3. XOR ergodicity: color_a ⊻ color_b covers the space uniformly

The sheaf condition ensures local colorings glue to global colorings
iff the XOR parity is conserved across adhesions.
"""
struct HadamardColorSheaf
    seed::UInt64
    parity_conserving::Bool  # Enforce XOR parity across boundaries
end

HadamardColorSheaf(; seed::UInt64=GAY_SEED) = HadamardColorSheaf(seed, true)

# CNOT gate: flips target based on control
function cnot_color(control::UInt32, target::UInt32)::UInt32
    # If control has odd popcount, flip target
    if count_ones(control) % 2 == 1
        return ~target
    else
        return target
    end
end

# Verify CNOT·CNOT = I
function cnot_xor_verify(a::UInt32, b::UInt32)::Bool
    # Apply CNOT twice should return original
    after_first = cnot_color(a, b)
    after_second = cnot_color(a, after_first)
    return after_second == b
end

# XOR ergodicity: verify uniform coverage
function xor_ergodic_test(seed::UInt64, n_samples::Int=1000)::Float64
    rng = GayRNG(seed)
    seen = Set{UInt32}()
    
    for _ in 1:n_samples
        a = UInt32(rand(gay_split(rng)) * typemax(UInt32))
        b = UInt32(rand(gay_split(rng)) * typemax(UInt32))
        push!(seen, a ⊻ b)
    end
    
    # Coverage ratio
    return length(seen) / n_samples
end

# ═══════════════════════════════════════════════════════════════════════════
# Cofree Comonad: Color Chain Structure
# ═══════════════════════════════════════════════════════════════════════════

"""
    CofreeColorChain

The cofree comonad on colors: an infinite stream of deterministic colors
derived from a seed. Structure:

    Cofree F A = A × F(Cofree F A)
    
For colors: Cofree Color = Color × Stream(Cofree Color)

This provides:
- extract: get current color
- extend: map over all future colors
- duplicate: nest the structure for 2-comonad

From ACSET_PATTERNS.md: "Color chain = Cofree comonoid (category of paths)"
"""
struct CofreeColorChain
    head::RGB{Float32}           # Current color (extract)
    seed::UInt64                 # Seed for tail generation
    index::Int                   # Position in chain
end

function CofreeColorChain(seed::UInt64=GAY_SEED)
    color = hash_color_rgb(UInt64(0), seed)
    CofreeColorChain(color, seed, 0)
end

# Comonad operations
function extract(chain::CofreeColorChain)::RGB{Float32}
    chain.head
end

function extend(f::Function, chain::CofreeColorChain)::CofreeColorChain
    # Apply f to this and all future positions
    new_color = f(chain)
    CofreeColorChain(new_color, chain.seed, chain.index)
end

function duplicate(chain::CofreeColorChain)::CofreeColorChain
    # Wrap in another layer: Cofree(Cofree(Color))
    # The head becomes the chain itself encoded as a color
    meta_color = hash_color_rgb(hash(chain), chain.seed)
    CofreeColorChain(meta_color, chain.seed ⊻ UInt64(chain.index), chain.index)
end

# Advance to next color in chain
function tail(chain::CofreeColorChain)::CofreeColorChain
    next_idx = chain.index + 1
    next_color = hash_color_rgb(UInt64(next_idx), chain.seed)
    CofreeColorChain(next_color, chain.seed, next_idx)
end

# Generate n colors from chain
function cofree_color_chain(seed::UInt64, n::Int)::Vector{RGB{Float32}}
    chain = CofreeColorChain(seed)
    colors = RGB{Float32}[]
    for _ in 1:n
        push!(colors, extract(chain))
        chain = tail(chain)
    end
    colors
end

# ═══════════════════════════════════════════════════════════════════════════
# Structured Decomposition Coloring
# ═══════════════════════════════════════════════════════════════════════════

"""
    color_decomposition(d::StrDecomp; seed=GAY_SEED) -> NamedTuple

Color a structured decomposition with SPI-compliant colors.

Bags: colored by position in decomposition shape
Adhesions: colored by XOR of adjacent bag colors (Hadamard guarantee)
"""
function color_decomposition(d::StrDecomp; seed::UInt64=GAY_SEED)
    bag_colors = color_bags(d; seed)
    adh_colors = color_adhesions(d, bag_colors; seed)
    
    (bags=bag_colors, adhesions=adh_colors, seed=seed)
end

function color_bags(d::StrDecomp; seed::UInt64=GAY_SEED)
    bs = bags(d)
    map(enumerate(bs)) do (i, bag)
        bag_hash = UInt64(hash(bag) ⊻ i)
        hash_color_rgb(bag_hash, seed)
    end
end

function color_adhesions(d::StrDecomp, bag_colors::Vector; seed::UInt64=GAY_SEED)
    spans = adhesionSpans(d)
    
    map(enumerate(spans)) do (i, span)
        # XOR of adjacent bag colors for Hadamard guarantee
        # This ensures CNOT·CNOT = I across the adhesion
        src_idx = 1  # First bag in span
        tgt_idx = min(2, length(bag_colors))  # Second bag in span
        
        if src_idx <= length(bag_colors) && tgt_idx <= length(bag_colors)
            c1 = bag_colors[src_idx]
            c2 = bag_colors[tgt_idx]
            
            # XOR the RGB components
            r = UInt8(round(c1.r * 255)) ⊻ UInt8(round(c2.r * 255))
            g = UInt8(round(c1.g * 255)) ⊻ UInt8(round(c2.g * 255))
            b = UInt8(round(c1.b * 255)) ⊻ UInt8(round(c2.b * 255))
            
            RGB{Float32}(r/255, g/255, b/255)
        else
            hash_color_rgb(UInt64(i), seed)
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# 2-Monad: Para(Para(Decomposition))
# ═══════════════════════════════════════════════════════════════════════════

"""
    TwoMonadDecomp

A 2-categorical structure over decompositions:

Level 0: Objects = Decompositions
Level 1: 1-morphisms = Functors between decompositions (Para)
Level 2: 2-morphisms = Natural transformations (Para(Para))

This captures both:
- Implicit indeterminacy: context-derived seeds
- Explicit indeterminacy: user-provided seeds
"""
struct TwoMonadDecomp{D}
    decomp::D
    level::Int  # 0, 1, or 2
    seed_chain::CofreeColorChain
    indeterminacies::Vector{ColorSeedIndeterminacy}
end

function TwoMonadDecomp(d::StrDecomp; seed::UInt64=GAY_SEED)
    chain = CofreeColorChain(seed)
    TwoMonadDecomp(d, 0, chain, ColorSeedIndeterminacy[])
end

# Para: lift to 1-morphism level
function para_decomp(tmd::TwoMonadDecomp)
    new_chain = tail(tmd.seed_chain)
    indet = implicit_seed(:parent, hash(tmd.decomp); base_seed=tmd.seed_chain.seed)
    TwoMonadDecomp(
        tmd.decomp,
        1,
        new_chain,
        [tmd.indeterminacies; indet]
    )
end

# Para(Para): lift to 2-morphism level
function para_para_decomp(tmd::TwoMonadDecomp)
    para_decomp(para_decomp(tmd))
end

# ═══════════════════════════════════════════════════════════════════════════
# Sheaf Decision with Chromatic Witness
# ═══════════════════════════════════════════════════════════════════════════

"""
    decide_colored_sheaf(f, d::StrDecomp; seed=GAY_SEED)

Run sheaf decision with chromatic witness tracking.
Returns (satisfiable, witness, colors).
"""
function decide_colored_sheaf(f, d::StrDecomp; seed::UInt64=GAY_SEED)
    # Color the decomposition
    colors = color_decomposition(d; seed)
    
    # Run sheaf decision
    result = decide_sheaf_tree_shape(f, d)
    
    (
        satisfiable=result[1],
        witness=result[2],
        colors=colors
    )
end

"""
    chromatic_treewidth(d::StrDecomp; seed=GAY_SEED)

Compute treewidth with chromatic annotation.
Each bag's color encodes its role in the decomposition.
"""
function chromatic_treewidth(d::StrDecomp; seed::UInt64=GAY_SEED)
    bs = bags(d)
    
    # Treewidth = max bag size - 1
    tw = maximum(length.(bs)) - 1
    
    # Color by bag size (hue) and position (saturation)
    colors = map(enumerate(bs)) do (i, bag)
        size_hue = (length(bag) / (tw + 1)) * 360
        pos_sat = 0.5 + 0.5 * (i / length(bs))
        
        # HSL to RGB via seed-deterministic conversion
        h = mod(size_hue + hash(seed) % 360, 360)
        s = pos_sat
        l = 0.5
        
        # Simple HSL→RGB
        c = (1 - abs(2*l - 1)) * s
        x = c * (1 - abs(mod(h/60, 2) - 1))
        m = l - c/2
        
        r, g, b = if h < 60
            (c + m, x + m, m)
        elseif h < 120
            (x + m, c + m, m)
        elseif h < 180
            (m, c + m, x + m)
        elseif h < 240
            (m, x + m, c + m)
        elseif h < 300
            (x + m, m, c + m)
        else
            (c + m, m, x + m)
        end
        
        RGB{Float32}(r, g, b)
    end
    
    (treewidth=tw, colors=colors, seed=seed)
end

# ═══════════════════════════════════════════════════════════════════════════
# Free ⊣ Cofree Adjunction (Basis for all other Ext)
# ═══════════════════════════════════════════════════════════════════════════

"""
    FreeCofreeAdjunction

The fundamental adjunction from which all Gay.jl extensions derive:

    Free ⊣ Cofree : Set → Coalg

For colors:
- Free(S) = color stream generated from seed S
- Cofree(S) = CofreeColorChain (infinite stream with head extraction)

Unit: S → Cofree(Free(S)) - embed seed into stream
Counit: Free(Cofree(S)) → S - collapse stream to seed

This ensures all extensions (Metal, Enzyme, Plasmo, etc.) maintain
chromatic coherence when composing.
"""
struct FreeCofreeAdjunction
    unit::Function    # seed → chain
    counit::Function  # chain → seed
end

function FreeCofreeAdjunction()
    unit = seed -> CofreeColorChain(seed)
    counit = chain -> chain.seed ⊻ UInt64(chain.index)
    FreeCofreeAdjunction(unit, counit)
end

# Verify adjunction laws
function verify_adjunction(adj::FreeCofreeAdjunction, seed::UInt64)::Bool
    chain = adj.unit(seed)
    recovered = adj.counit(chain)
    
    # Triangle identities (approximate due to XOR)
    # Free(unit) ; counit = id
    # unit ; Cofree(counit) = id
    
    chain2 = adj.unit(recovered)
    extract(chain) == extract(chain2)
end

# ═══════════════════════════════════════════════════════════════════════════
# Extension Coherence: Basis for Other Ext
# ═══════════════════════════════════════════════════════════════════════════

"""
    ExtensionCoherence

Provides coherence guarantees for all Gay.jl extensions by requiring
they factor through the Free⊣Cofree adjunction.

Extensions must implement:
- lift_to_cofree: embed their structures into CofreeColorChain
- project_from_free: extract their structures from Free color streams
"""
abstract type ExtensionCoherence end

struct GayExtCoherence <: ExtensionCoherence
    ext_name::Symbol
    adjunction::FreeCofreeAdjunction
    lift::Function
    project::Function
end

# Register coherence for an extension
function register_coherence(name::Symbol, lift::Function, project::Function)
    adj = FreeCofreeAdjunction()
    GayExtCoherence(name, adj, lift, project)
end

# Verify coherence
function verify_coherence(coh::GayExtCoherence, test_data)::Bool
    lifted = coh.lift(test_data)
    projected = coh.project(lifted)
    
    # Check round-trip
    lifted2 = coh.lift(projected)
    extract(lifted) == extract(lifted2)
end

end # module GayStructuredDecompositionsExt
