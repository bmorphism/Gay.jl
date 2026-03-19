# Para(Para(Gay)) vs Para(Para(Gay#)) — Contrastive Learning
# ============================================================================
#
# Extracted via GayRNG-selected MCPs (radare2, marginalia, exa):
#   Interaction 1: #a74d78 (radare2 → binary category extraction)
#   Interaction 2: #117465 (marginalia → Para(Optic), Tambara, CyberCat)
#   Interaction 3: #4e9a8c (exa → Hedges CGT, Arrows, equilibrium)
#
# NAMEABLE CATEGORIES from MCP extractions:
#   1. Set, FinSet — discrete categories
#   2. Optic, Lens, Prism — bidirectional transformations  
#   3. Para(C) — parametrised morphisms in C
#   4. Tambara — Tambara modules (Para ≃ Tambara[×,→])
#   5. Arrow — Hughes arrows (≃ enriched profunctors)
#   6. OpenGame — symmetric monoidal category of games
#   7. Free(C) — free category on graph
#   8. Comonad — comonadic structures for context
#
# KEY DISTINCTION:
#   Para(Para(Gay))  = Doubly parametrised over the Gay COLORSPACE
#   Para(Para(Gay#)) = Doubly parametrised over the Gay HASHSPACE (Cat#)
#
# Gay# (Gay-sharp) is the category where:
#   - Objects = UInt64 hashes
#   - Morphisms = SplitMix64 transitions
#   - Composition = splitmix64_next chaining

module ParaParaGaySharpMod

using SplittableRandoms: SplittableRandom, split

export
    # Core distinction
    ParaParaGayColor, ParaParaGayHash, GaySharpCategory,
    
    # Nameable categories for contrast
    NameableCategory, EXTRACTED_CATEGORIES,
    
    # Contrastive learning
    ContrastivePair, contrastive_loss, contrastive_gradient,
    CategoryEmbedding, embed_category, category_distance,
    
    # Interaction colors
    InteractionColor, derive_interaction_color,
    
    # Demo
    demo_contrastive_para

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & RNG
# ═══════════════════════════════════════════════════════════════════════════════

const GAY_SEED = 0x6761795f636f6c6f
const GOLDEN = 0x9e3779b97f4a7c15
const MIX1 = 0xbf58476d1ce4e5b9
const MIX2 = 0x94d049bb133111eb
const MASK64 = 0xFFFFFFFFFFFFFFFF

# Interaction colors from GayRNG MCP selection
const INTERACTION_COLORS = [
    (name=:radare2,    hash=0xa74d78, rgb=(167, 77, 120)),
    (name=:marginalia, hash=0x117465, rgb=(17, 116, 101)),
    (name=:exa,        hash=0x4e9a8c, rgb=(78, 154, 140)),
]

function splitmix64_next(state::UInt64)::UInt64
    s = (state + GOLDEN) & MASK64
    z = s
    z = ((z ⊻ (z >> 30)) * MIX1) & MASK64
    z = ((z ⊻ (z >> 27)) * MIX2) & MASK64
    (z ⊻ (z >> 31)) & MASK64
end

function derive_seed(base::UInt64, index::Int)::UInt64
    splitmix64_next((base ⊻ (UInt64(index) * GOLDEN)) & MASK64)
end

# ═══════════════════════════════════════════════════════════════════════════════
# NAMEABLE CATEGORIES (extracted from MCP interactions)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    NameableCategory

A category extracted from MCP interactions, with:
- name: Symbol identifier
- hash: Derived from category name via Gay#
- properties: Categorical structure (has_products, has_coproducts, etc.)
- source_mcp: Which MCP interaction extracted this
"""
struct NameableCategory
    name::Symbol
    hash::UInt64
    
    # Categorical properties
    has_products::Bool
    has_coproducts::Bool
    has_exponentials::Bool
    is_monoidal::Bool
    is_closed::Bool
    is_enriched::Bool
    
    # Source
    source_mcp::Symbol
    source_color::UInt32
end

function NameableCategory(name::Symbol; 
                          products::Bool=false, coproducts::Bool=false,
                          exponentials::Bool=false, monoidal::Bool=false,
                          closed::Bool=false, enriched::Bool=false,
                          source::Symbol=:unknown)
    # Hash from name
    name_bytes = collect(UInt8, String(name))
    h = GAY_SEED
    for b in name_bytes
        h = splitmix64_next(h ⊻ UInt64(b))
    end
    
    # Find source color
    source_color = UInt32(0)
    for ic in INTERACTION_COLORS
        if ic.name == source
            source_color = UInt32(ic.hash)
            break
        end
    end
    
    NameableCategory(name, h, products, coproducts, exponentials, 
                     monoidal, closed, enriched, source, source_color)
end

# Categories extracted from the 3 MCP interactions
const EXTRACTED_CATEGORIES = [
    # From radare2 (binary analysis → computational categories)
    NameableCategory(:FinSet, products=true, coproducts=true, exponentials=true, 
                     closed=true, source=:radare2),
    NameableCategory(:FinVect, products=true, coproducts=true, monoidal=true, 
                     closed=true, source=:radare2),
    NameableCategory(:Rel, products=true, coproducts=true, monoidal=true, 
                     source=:radare2),
    
    # From marginalia (Para(Optic), Tambara, neural nets)
    NameableCategory(:Optic, monoidal=true, enriched=true, source=:marginalia),
    NameableCategory(:Lens, products=true, monoidal=true, source=:marginalia),
    NameableCategory(:Prism, coproducts=true, monoidal=true, source=:marginalia),
    NameableCategory(:Tambara, monoidal=true, enriched=true, closed=true, 
                     source=:marginalia),
    NameableCategory(:Para, monoidal=true, source=:marginalia),
    
    # From exa (Hedges CGT, OpenGames, Arrows)
    NameableCategory(:OpenGame, monoidal=true, source=:exa),
    NameableCategory(:Arrow, monoidal=true, enriched=true, source=:exa),
    NameableCategory(:Comonad, monoidal=true, source=:exa),
    NameableCategory(:Free, coproducts=true, monoidal=true, source=:exa),
]

# ═══════════════════════════════════════════════════════════════════════════════
# GAY# (GAY-SHARP) CATEGORY
# ═══════════════════════════════════════════════════════════════════════════════
#
# Gay# is the category where:
#   Objects = UInt64 (hashes)
#   Hom(a,b) = { f : a → b | f = splitmix64_next^n for some n }
#   Composition = function composition (= adding exponents)

"""
    GaySharpCategory

The category Gay# with:
- Objects: UInt64 hashes
- Morphisms: SplitMix64 transition functions
- Identity: id_a = λx.x (0 iterations)
- Composition: f ∘ g = λx. f(g(x))

This is the "sharp" (♯) version — the computational/hashable category
as opposed to the colorspace category.
"""
struct GaySharpCategory
    seed::UInt64
    objects::Vector{UInt64}        # Observed objects
    morphisms::Dict{Tuple{UInt64,UInt64}, Int}  # (src,tgt) → iterations
end

function GaySharpCategory(seed::UInt64; depth::Int=10)
    objects = UInt64[seed]
    morphisms = Dict{Tuple{UInt64,UInt64}, Int}()
    
    s = seed
    for i in 1:depth
        next_s = splitmix64_next(s)
        push!(objects, next_s)
        morphisms[(s, next_s)] = 1
        
        # Also record composite morphisms
        for j in 1:i
            if j > 1
                morphisms[(objects[1], next_s)] = i
            end
        end
        s = next_s
    end
    
    GaySharpCategory(seed, objects, morphisms)
end

"""Check if there's a morphism between two objects in Gay#"""
function has_morphism(gc::GaySharpCategory, src::UInt64, tgt::UInt64)::Bool
    haskey(gc.morphisms, (src, tgt))
end

"""Get the morphism (as iteration count) between two objects"""
function get_morphism(gc::GaySharpCategory, src::UInt64, tgt::UInt64)::Union{Int, Nothing}
    get(gc.morphisms, (src, tgt), nothing)
end

# ═══════════════════════════════════════════════════════════════════════════════
# PARA(PARA(GAY)) vs PARA(PARA(GAY#))
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ParaParaGay

Doubly parametrised structure over the Gay COLORSPACE.

Para(Para(C)) for C = Gay where:
- Objects = Okhsl colors (H, S, L)
- Morphisms = Continuous color transitions
- Parameters = (context::Color, action::Color)

This is the "soft" version — perceptually uniform, continuous.
"""
struct ParaParaGayColor
    seed::UInt64
    
    # Outer Para: context parameters (colors)
    context_colors::Vector{NTuple{3, Float64}}  # (H, S, L)
    
    # Inner Para: action parameters (colors)
    action_colors::Matrix{NTuple{3, Float64}}   # [context, depth]
    
    # Apex: universal color (limit in colorspace)
    apex_color::NTuple{3, Float64}
end

function ParaParaGayColor(seed::UInt64; n_context::Int=7, depth::Int=5)
    # Generate context colors
    ctx_colors = NTuple{3, Float64}[]
    for i in 1:n_context
        h = derive_seed(seed, i)
        H = ((h >> 48) & 0xFFFF) / 65535.0 * 360.0
        S = 0.5 + ((h >> 32) & 0xFFFF) / 65535.0 * 0.4
        L = 0.35 + ((h >> 16) & 0xFFFF) / 65535.0 * 0.4
        push!(ctx_colors, (H, S, L))
    end
    
    # Generate action colors
    act_colors = Matrix{NTuple{3, Float64}}(undef, n_context, depth)
    for c in 1:n_context
        s = derive_seed(seed, c * 1000)
        for d in 1:depth
            s = splitmix64_next(s)
            H = ((s >> 48) & 0xFFFF) / 65535.0 * 360.0
            S = 0.5 + ((s >> 32) & 0xFFFF) / 65535.0 * 0.4
            L = 0.35 + ((s >> 16) & 0xFFFF) / 65535.0 * 0.4
            act_colors[c, d] = (H, S, L)
        end
    end
    
    # Apex: average color (centroid in Okhsl)
    sum_H, sum_S, sum_L = 0.0, 0.0, 0.0
    count = 0
    for c in ctx_colors
        sum_H += c[1]; sum_S += c[2]; sum_L += c[3]
        count += 1
    end
    for c in act_colors
        sum_H += c[1]; sum_S += c[2]; sum_L += c[3]
        count += 1
    end
    apex = (sum_H/count, sum_S/count, sum_L/count)
    
    ParaParaGayColor(seed, ctx_colors, act_colors, apex)
end

"""
    ParaParaGaySharp

Doubly parametrised structure over the Gay# HASHSPACE.

Para(Para(C)) for C = Gay# where:
- Objects = UInt64 hashes
- Morphisms = SplitMix64 iterations
- Parameters = (context::Hash, action::Hash)

This is the "sharp" version — discrete, computational, hash-indexed.
"""
struct ParaParaGayHash
    seed::UInt64
    
    # Outer Para: context parameters (hashes)
    context_hashes::Vector{UInt64}
    
    # Inner Para: action parameters (hashes)
    action_hashes::Matrix{UInt64}   # [context, depth]
    
    # Apex: XOR fold (limit in hashspace)
    apex_hash::UInt64
    
    # The underlying Gay# category
    category::GaySharpCategory
end

function ParaParaGayHash(seed::UInt64; n_context::Int=7, depth::Int=5)
    # Generate context hashes
    ctx_hashes = [derive_seed(seed, i) for i in 1:n_context]
    
    # Generate action hashes
    act_hashes = Matrix{UInt64}(undef, n_context, depth)
    for c in 1:n_context
        s = ctx_hashes[c]
        for d in 1:depth
            s = splitmix64_next(s)
            act_hashes[c, d] = s
        end
    end
    
    # Apex: XOR fold (universal in hash algebra)
    apex = reduce(⊻, ctx_hashes) ⊻ reduce(⊻, act_hashes)
    
    # Build the underlying category
    cat = GaySharpCategory(seed; depth=n_context + depth)
    
    ParaParaGayHash(seed, ctx_hashes, act_hashes, apex, cat)
end

# ═══════════════════════════════════════════════════════════════════════════════
# CONTRASTIVE LEARNING
# ═══════════════════════════════════════════════════════════════════════════════
#
# Learn to distinguish ParaParaGay from ParaParaGaySharp by contrasting
# against all nameable categories.

"""
    CategoryEmbedding

An embedding of a category into a vector space for contrastive learning.
"""
struct CategoryEmbedding
    category::Union{NameableCategory, Symbol}
    vector::Vector{Float64}
    hash::UInt64
end

"""Embed a NameableCategory into a vector."""
function embed_category(nc::NameableCategory)::CategoryEmbedding
    # 8-dimensional embedding based on properties
    v = Float64[
        nc.has_products ? 1.0 : 0.0,
        nc.has_coproducts ? 1.0 : 0.0,
        nc.has_exponentials ? 1.0 : 0.0,
        nc.is_monoidal ? 1.0 : 0.0,
        nc.is_closed ? 1.0 : 0.0,
        nc.is_enriched ? 1.0 : 0.0,
        (nc.hash >> 32) / Float64(typemax(UInt32)),  # Hash-derived features
        (nc.hash & 0xFFFFFFFF) / Float64(typemax(UInt32)),
    ]
    CategoryEmbedding(nc, v, nc.hash)
end

"""Embed ParaParaGay into a vector (colorspace features)."""
function embed_category(ppg::ParaParaGayColor)::CategoryEmbedding
    H, S, L = ppg.apex_color
    v = Float64[
        1.0,  # Has products (via colorspace ops)
        1.0,  # Has coproducts
        0.0,  # No exponentials (continuous)
        1.0,  # Monoidal
        0.0,  # Not closed
        0.0,  # Not enriched
        H / 360.0,
        S,
    ]
    CategoryEmbedding(:ParaParaGay, v, ppg.seed)
end

"""Embed ParaParaGaySharp into a vector (hashspace features)."""
function embed_category(ppgs::ParaParaGayHash)::CategoryEmbedding
    v = Float64[
        1.0,  # Has products (via XOR)
        1.0,  # Has coproducts
        0.0,  # No exponentials
        1.0,  # Monoidal (XOR is associative)
        1.0,  # Closed (hash functions are internal)
        1.0,  # Enriched (over itself)
        (ppgs.apex_hash >> 32) / Float64(typemax(UInt32)),
        (ppgs.apex_hash & 0xFFFFFFFF) / Float64(typemax(UInt32)),
    ]
    CategoryEmbedding(:ParaParaGaySharp, v, ppgs.seed)
end

"""Euclidean distance between embeddings."""
function category_distance(e1::CategoryEmbedding, e2::CategoryEmbedding)::Float64
    sqrt(sum((e1.vector .- e2.vector).^2))
end

"""
    ContrastivePair

A pair of embeddings for contrastive learning:
- anchor: The reference embedding
- positive: Similar to anchor (same class)
- negative: Different from anchor (different class)
"""
struct ContrastivePair
    anchor::CategoryEmbedding
    positive::CategoryEmbedding
    negative::CategoryEmbedding
    
    # Distances
    d_pos::Float64
    d_neg::Float64
end

function ContrastivePair(anchor::CategoryEmbedding, pos::CategoryEmbedding, neg::CategoryEmbedding)
    d_pos = category_distance(anchor, pos)
    d_neg = category_distance(anchor, neg)
    ContrastivePair(anchor, pos, neg, d_pos, d_neg)
end

"""
Contrastive loss: want d_pos << d_neg.
L = max(0, d_pos - d_neg + margin)
"""
function contrastive_loss(pair::ContrastivePair; margin::Float64=0.5)::Float64
    max(0.0, pair.d_pos - pair.d_neg + margin)
end

"""
Compute gradient of contrastive loss (simplified).
Returns direction to push anchor.
"""
function contrastive_gradient(pair::ContrastivePair)::Vector{Float64}
    if pair.d_pos > pair.d_neg
        # Push toward positive, away from negative
        dir_pos = pair.positive.vector .- pair.anchor.vector
        dir_neg = pair.anchor.vector .- pair.negative.vector
        normalize(dir_pos .+ dir_neg)
    else
        # Already correct, small regularization toward positive
        normalize(pair.positive.vector .- pair.anchor.vector) .* 0.1
    end
end

function normalize(v::Vector{Float64})::Vector{Float64}
    n = sqrt(sum(v.^2))
    n > 0 ? v ./ n : v
end

# ═══════════════════════════════════════════════════════════════════════════════
# INTERACTION COLOR DERIVATION
# ═══════════════════════════════════════════════════════════════════════════════

"""
    InteractionColor

Color derived from an MCP interaction.
"""
struct InteractionColor
    mcp::Symbol
    interaction_index::Int
    hash::UInt32
    rgb::NTuple{3, UInt8}
    hsl::NTuple{3, Float64}
end

function derive_interaction_color(mcp::Symbol, index::Int, seed::UInt64)::InteractionColor
    # Derive hash from MCP name and index
    mcp_bytes = collect(UInt8, String(mcp))
    h = seed ⊻ UInt64(index * 1069)
    for b in mcp_bytes
        h = splitmix64_next(h ⊻ UInt64(b))
    end
    
    hash = UInt32(h & 0xFFFFFF)
    r = UInt8((h >> 16) & 0xFF)
    g = UInt8((h >> 8) & 0xFF)
    b = UInt8(h & 0xFF)
    
    # Convert to HSL
    rf, gf, bf = r/255.0, g/255.0, b/255.0
    cmax = max(rf, gf, bf)
    cmin = min(rf, gf, bf)
    delta = cmax - cmin
    
    L = (cmax + cmin) / 2
    S = delta == 0 ? 0.0 : delta / (1 - abs(2*L - 1))
    H = if delta == 0
        0.0
    elseif cmax == rf
        60 * mod((gf - bf) / delta, 6)
    elseif cmax == gf
        60 * ((bf - rf) / delta + 2)
    else
        60 * ((rf - gf) / delta + 4)
    end
    
    InteractionColor(mcp, index, hash, (r, g, b), (H, S, L))
end

# ═══════════════════════════════════════════════════════════════════════════════
# DEMO: CONTRASTIVE LEARNING ParaParaGay vs ParaParaGay#
# ═══════════════════════════════════════════════════════════════════════════════

function demo_contrastive_para()
    println()
    println("╔═══════════════════════════════════════════════════════════════════════╗")
    println("║  PARA(PARA(GAY)) vs PARA(PARA(GAY#)) — CONTRASTIVE LEARNING          ║")
    println("║  Extracted via GayRNG-selected MCPs: radare2, marginalia, exa        ║")
    println("╚═══════════════════════════════════════════════════════════════════════╝")
    println()
    
    # Show interaction colors
    println("┌─────────────────────────────────────────────────────────────────────┐")
    println("│ INTERACTION COLORS (GayRNG)                                        │")
    println("└─────────────────────────────────────────────────────────────────────┘")
    for ic in INTERACTION_COLORS
        r, g, b = ic.rgb
        println("  $(rpad(String(ic.name), 12)) #$(string(ic.hash, base=16, pad=6))  RGB($r, $g, $b)")
    end
    println()
    
    # Show extracted categories
    println("┌─────────────────────────────────────────────────────────────────────┐")
    println("│ NAMEABLE CATEGORIES (12 extracted)                                 │")
    println("└─────────────────────────────────────────────────────────────────────┘")
    for nc in EXTRACTED_CATEGORIES
        props = String[]
        nc.has_products && push!(props, "×")
        nc.has_coproducts && push!(props, "+")
        nc.has_exponentials && push!(props, "⇒")
        nc.is_monoidal && push!(props, "⊗")
        nc.is_closed && push!(props, "CCC")
        nc.is_enriched && push!(props, "V-")
        prop_str = join(props, " ")
        println("  $(rpad(String(nc.name), 10)) [$(rpad(String(nc.source_mcp), 10))]  $(prop_str)")
    end
    println()
    
    # Build the two Para structures
    println("┌─────────────────────────────────────────────────────────────────────┐")
    println("│ PARA(PARA(GAY)) — Colorspace Category                              │")
    println("└─────────────────────────────────────────────────────────────────────┘")
    ppg = ParaParaGayColor(GAY_SEED)
    H, S, L = ppg.apex_color
    println("  Apex color: H=$(round(H, digits=1))° S=$(round(S, digits=3)) L=$(round(L, digits=3))")
    println("  Context colors: $(length(ppg.context_colors))")
    println("  Action matrix: $(size(ppg.action_colors))")
    println("  Type: CONTINUOUS, perceptually uniform, colorspace-indexed")
    println()
    
    println("┌─────────────────────────────────────────────────────────────────────┐")
    println("│ PARA(PARA(GAY#)) — Hashspace Category                              │")
    println("└─────────────────────────────────────────────────────────────────────┘")
    ppgs = ParaParaGayHash(GAY_SEED)
    println("  Apex hash: 0x$(string(ppgs.apex_hash, base=16))")
    println("  Context hashes: $(length(ppgs.context_hashes))")
    println("  Action matrix: $(size(ppgs.action_hashes))")
    println("  Category morphisms: $(length(ppgs.category.morphisms))")
    println("  Type: DISCRETE, computational, hash-indexed")
    println()
    
    # Embed everything
    println("┌─────────────────────────────────────────────────────────────────────┐")
    println("│ CONTRASTIVE EMBEDDINGS                                             │")
    println("└─────────────────────────────────────────────────────────────────────┘")
    
    emb_ppg = embed_category(ppg)
    emb_ppgs = embed_category(ppgs)
    cat_embeddings = [embed_category(nc) for nc in EXTRACTED_CATEGORIES]
    
    println("  ParaParaGay     → $(round.(emb_ppg.vector, digits=2))")
    println("  ParaParaGay#    → $(round.(emb_ppgs.vector, digits=2))")
    println()
    
    # Distance matrix
    println("  Distance to nameable categories:")
    println("  $(rpad("Category", 12)) $(rpad("d(PPG)", 10)) $(rpad("d(PPG#)", 10)) Closer to")
    println("  " * "-"^45)
    
    for (nc, emb) in zip(EXTRACTED_CATEGORIES, cat_embeddings)
        d_ppg = category_distance(emb_ppg, emb)
        d_ppgs = category_distance(emb_ppgs, emb)
        closer = d_ppg < d_ppgs ? "ParaParaGay" : "ParaParaGay#"
        println("  $(rpad(String(nc.name), 12)) $(rpad(round(d_ppg, digits=3), 10)) $(rpad(round(d_ppgs, digits=3), 10)) $closer")
    end
    println()
    
    # Contrastive learning: PPG vs PPG# with all categories as negatives
    println("┌─────────────────────────────────────────────────────────────────────┐")
    println("│ CONTRASTIVE LOSS ANALYSIS                                          │")
    println("└─────────────────────────────────────────────────────────────────────┘")
    
    # PPG as anchor, PPG# as positive (both are Para constructions)
    # Each named category as negative
    total_loss = 0.0
    for (nc, emb) in zip(EXTRACTED_CATEGORIES, cat_embeddings)
        pair = ContrastivePair(emb_ppg, emb_ppgs, emb)
        loss = contrastive_loss(pair)
        total_loss += loss
        if loss > 0
            println("  Loss(PPG ← PPG# vs $(nc.name)): $(round(loss, digits=4))")
        end
    end
    println()
    println("  Total contrastive loss: $(round(total_loss, digits=4))")
    println()
    
    # Key distinction
    println("╔═══════════════════════════════════════════════════════════════════════╗")
    println("║  KEY DISTINCTION: ParaParaGay vs ParaParaGay#                        ║")
    println("╠═══════════════════════════════════════════════════════════════════════╣")
    println("║                                                                       ║")
    println("║  ParaParaGay (colorspace):                                           ║")
    println("║    • Objects = Okhsl colors (continuous manifold)                    ║")
    println("║    • Morphisms = Color transitions (smooth paths)                    ║")
    println("║    • Apex = Centroid (limit in colorspace)                           ║")
    println("║    • Use: Visual, perceptual, human-facing                           ║")
    println("║                                                                       ║")
    println("║  ParaParaGay# (hashspace):                                           ║")
    println("║    • Objects = UInt64 hashes (discrete set)                          ║")
    println("║    • Morphisms = SplitMix64 iterations (computational)               ║")
    println("║    • Apex = XOR fold (limit in hash algebra)                         ║")
    println("║    • Use: Verification, cryptographic, machine-facing                ║")
    println("║                                                                       ║")
    println("║  The # (sharp) denotes the HASHABLE category — where Cat# means      ║")
    println("║  \"the category of categories with hash-indexed objects.\"            ║")
    println("╚═══════════════════════════════════════════════════════════════════════╝")
    println()
    
    (ppg=ppg, ppgs=ppgs, embeddings=cat_embeddings, loss=total_loss)
end

end # module ParaParaGaySharpMod
