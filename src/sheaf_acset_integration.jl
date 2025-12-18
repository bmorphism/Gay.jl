"""
    SheafACSetIntegration

Bridges StructuredDecompositions.jl's `decide_sheaf_tree_shape` with Gay.jl's
chromatic ACSet system for neighbor-aware local rewriting gadgets.

Key insight: Thread ancestry forests form copresheaves over the continuation category,
where local neighborhoods are threads sharing common ancestors, and rewriting gadgets
are forks creating parallel copresheaf sections.

The adhesion_filter algorithm from Bumpus et al. checks consistency across overlapping
neighborhoods - we extend this with chromatic identity for SPI verification.

Reference: Structured decompositions as functors
    𝐃: Cat_pullback → Cat
taking categories with pullbacks to categories of structured decompositions.

Seed 1069 balanced ternary: [+1, -1, -1, +1, +1, +1, +1]
"""
module SheafACSetIntegration

using ..Gay
using Colors
using SplittableRandoms

# ═══════════════════════════════════════════════════════════════════════════════
# Core Types for Chromatic Structured Decompositions
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ColorMorphism{C}

A morphism tagged with chromatic identity for SPI verification.
The color tracks provenance through the decomposition.
"""
struct ColorMorphism{C}
    source::Int
    target::Int
    color::C
    fingerprint::UInt64
end

"""
    ChromaticBag{T,C}

A bag in a structured decomposition with chromatic identity.
Each element has an associated color for neighbor-aware tracking.
"""
struct ChromaticBag{T,C}
    elements::Vector{T}
    colors::Vector{C}
    neighborhood_hash::UInt64
end

function ChromaticBag(elements::Vector{T}, seed::Int=1069) where T
    rng = SplittableRandom(seed)
    colors = [color_at(i, SRGB(); seed=seed) for i in eachindex(elements)]
    h = hash(colors)
    ChromaticBag{T,eltype(colors)}(elements, colors, h)
end

"""
    ChromaticAdhesion{T,C}

An adhesion (overlap) between bags with chromatic consistency tracking.
Adhesions are the "glue" that enables local-to-global inference.
"""
struct ChromaticAdhesion{T,C}
    apex::Vector{T}
    left_leg::Vector{Int}   # indices into left bag
    right_leg::Vector{Int}  # indices into right bag
    apex_colors::Vector{C}
    consistency::Bool       # do colors match across adhesion?
end

"""
    ChromaticDecomposition{G,T,C}

A structured decomposition with chromatic identity on all components.
- shape: The underlying graph/tree structure (typically a tree for tractable inference)
- bags: ChromaticBag at each node
- adhesions: ChromaticAdhesion at each edge
- decomp_type: :decomposition or :codecomposition
"""
struct ChromaticDecomposition{G,T,C}
    shape::G
    bags::Vector{ChromaticBag{T,C}}
    adhesions::Vector{ChromaticAdhesion{T,C}}
    decomp_type::Symbol
    seed::Int
end

# ═══════════════════════════════════════════════════════════════════════════════
# Neighbor-Aware Operations
# ═══════════════════════════════════════════════════════════════════════════════

"""
    neighbors(d::ChromaticDecomposition, bag_idx::Int)

Get the indices of bags that share an adhesion with the given bag.
This is the local neighborhood for rewriting gadgets.
"""
function neighbors(d::ChromaticDecomposition, bag_idx::Int)
    neighbor_indices = Int[]
    for (i, adh) in enumerate(d.adhesions)
        # Adhesions connect adjacent bags in the tree
        if bag_idx == i || bag_idx == i + 1
            push!(neighbor_indices, bag_idx == i ? i + 1 : i)
        end
    end
    neighbor_indices
end

"""
    local_neighborhood(d::ChromaticDecomposition, bag_idx::Int)

Extract the full local neighborhood: the bag and all its adjacent bags
with their connecting adhesions. Returns a sub-decomposition.
"""
function local_neighborhood(d::ChromaticDecomposition, bag_idx::Int)
    nbrs = neighbors(d, bag_idx)
    all_indices = sort(unique([bag_idx; nbrs]))

    local_bags = d.bags[all_indices]

    # Get adhesions that connect these bags
    local_adhesions = ChromaticAdhesion[]
    for i in 1:(length(all_indices)-1)
        if i <= length(d.adhesions)
            push!(local_adhesions, d.adhesions[i])
        end
    end

    ChromaticDecomposition(
        nothing,  # Local neighborhoods don't need full shape
        local_bags,
        local_adhesions,
        d.decomp_type,
        d.seed
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# Chromatic Adhesion Filter (Core Algorithm)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    chromatic_adhesion_filter(d::ChromaticDecomposition, adhesion_idx::Int)

Chromatic version of Bumpus's adhesion_filter algorithm.

The filtering algorithm computes pullbacks on adhesion spans and projects
back to bags, eliminating elements that cannot participate in global solutions.

We extend this with color tracking:
1. Compute pullback of colored elements across adhesion
2. Track which colors survive the consistency check
3. Project surviving elements back with updated colors
4. Mark the adhesion as consistent/inconsistent

Returns: (filtered_decomposition, color_witness)
"""
function chromatic_adhesion_filter(d::ChromaticDecomposition, adhesion_idx::Int)
    if adhesion_idx > length(d.adhesions)
        return (d, nothing)
    end

    adh = d.adhesions[adhesion_idx]
    left_bag = d.bags[adhesion_idx]
    right_bag = d.bags[min(adhesion_idx + 1, length(d.bags))]

    # Compute pullback: elements that can be consistently matched
    # through the adhesion
    surviving_left = Int[]
    surviving_right = Int[]
    pullback_colors = typeof(left_bag.colors[1])[]

    for (li, left_idx) in enumerate(adh.left_leg)
        for (ri, right_idx) in enumerate(adh.right_leg)
            if left_idx <= length(left_bag.elements) &&
               right_idx <= length(right_bag.elements)
                # Chromatic consistency: colors must be compatible
                # (in same hue family for neighbor awareness)
                left_c = left_bag.colors[min(left_idx, length(left_bag.colors))]
                right_c = right_bag.colors[min(right_idx, length(right_bag.colors))]

                if colors_compatible(left_c, right_c)
                    push!(surviving_left, left_idx)
                    push!(surviving_right, right_idx)
                    # Blend colors for the pullback element
                    blended = blend_colors(left_c, right_c)
                    push!(pullback_colors, blended)
                end
            end
        end
    end

    # Create filtered bags
    new_left = filter_bag(left_bag, unique(surviving_left))
    new_right = filter_bag(right_bag, unique(surviving_right))

    # Update the decomposition
    new_bags = copy(d.bags)
    new_bags[adhesion_idx] = new_left
    if adhesion_idx + 1 <= length(new_bags)
        new_bags[adhesion_idx + 1] = new_right
    end

    # Update adhesion consistency
    new_adhesions = copy(d.adhesions)
    new_adhesions[adhesion_idx] = ChromaticAdhesion(
        adh.apex,
        surviving_left,
        surviving_right,
        pullback_colors,
        !isempty(surviving_left)
    )

    filtered_d = ChromaticDecomposition(
        d.shape,
        new_bags,
        new_adhesions,
        d.decomp_type,
        d.seed
    )

    color_witness = (
        left_survivors = length(surviving_left),
        right_survivors = length(surviving_right),
        pullback_size = length(pullback_colors),
        consistent = !isempty(pullback_colors)
    )

    (filtered_d, color_witness)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Decide Chromatic Sheaf (Main Algorithm)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    decide_chromatic_sheaf(d::ChromaticDecomposition)

Chromatic extension of decide_sheaf_tree_shape.

Iteratively applies chromatic_adhesion_filter across all adhesion spans.
Returns (success, witness, color_trace) where:
- success: true if no bag becomes empty
- witness: the final filtered decomposition
- color_trace: sequence of color witnesses showing SPI verification

This implements neighbor-aware consistency checking for local rewriting gadgets.
Thread ancestry forests use this to verify that parallel copresheaf sections
can be reconciled.
"""
function decide_chromatic_sheaf(d::ChromaticDecomposition)
    witness = d
    color_trace = []

    for i in eachindex(d.adhesions)
        witness, color_witness = chromatic_adhesion_filter(witness, i)
        push!(color_trace, (adhesion=i, witness=color_witness))

        # Check if any bag is empty
        if any(isempty, [b.elements for b in witness.bags])
            return (false, witness, color_trace)
        end
    end

    (true, witness, color_trace)
end

"""
    decide_chromatic_sheaf(f, d::ChromaticDecomposition)

Version that first lifts a functor f to the decomposition before deciding.
This corresponds to 𝐃(f, d) ∘ decide_sheaf_tree_shape.
"""
function decide_chromatic_sheaf(f::Function, d::ChromaticDecomposition)
    # Apply f to each bag's elements
    lifted_bags = map(d.bags) do bag
        new_elements = f.(bag.elements)
        ChromaticBag(new_elements, d.seed)
    end

    lifted_d = ChromaticDecomposition(
        d.shape,
        lifted_bags,
        d.adhesions,  # Adhesions need updating too in full impl
        d.decomp_type,
        d.seed
    )

    decide_chromatic_sheaf(lifted_d)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Thread Ancestry Forest Operations
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ThreadAncestryNode

A node in the thread ancestry forest (copresheaf over continuation category).
Each thread has a color identity for SPI verification.
"""
struct ThreadAncestryNode
    thread_id::UInt64
    parent_id::Union{Nothing, UInt64}
    color::RGB{Float64}
    children::Vector{UInt64}
    continuation_depth::Int
end

"""
    ThreadAncestryForest

The full ancestry forest forms a copresheaf over the continuation category.
Local neighborhoods are threads sharing common ancestors.
"""
struct ThreadAncestryForest
    nodes::Dict{UInt64, ThreadAncestryNode}
    roots::Vector{UInt64}
    decomposition::Union{Nothing, ChromaticDecomposition}
end

"""
    to_chromatic_decomposition(forest::ThreadAncestryForest; seed=1069)

Convert a thread ancestry forest to a chromatic decomposition for
decide_sheaf_tree_shape analysis.

Each tree in the forest becomes a bag, with adhesions formed by
shared ancestors between trees.
"""
function to_chromatic_decomposition(forest::ThreadAncestryForest; seed::Int=1069)
    if isempty(forest.roots)
        return nothing
    end

    # Each root defines a bag (the subtree rooted there)
    bags = ChromaticBag[]

    for root_id in forest.roots
        subtree_nodes = collect_subtree(forest, root_id)
        bag = ChromaticBag(subtree_nodes, seed)
        push!(bags, bag)
    end

    # Adhesions from shared ancestry
    adhesions = ChromaticAdhesion[]

    for i in 1:(length(bags)-1)
        # Find shared elements between adjacent bags
        left_elems = bags[i].elements
        right_elems = bags[i+1].elements

        shared = intersect(left_elems, right_elems)

        left_indices = [findfirst(==(s), left_elems) for s in shared]
        right_indices = [findfirst(==(s), right_elems) for s in shared]

        filter!(!isnothing, left_indices)
        filter!(!isnothing, right_indices)

        apex_colors = [bags[i].colors[li] for li in left_indices if li !== nothing]

        adh = ChromaticAdhesion(
            shared,
            left_indices,
            right_indices,
            apex_colors,
            !isempty(shared)
        )
        push!(adhesions, adh)
    end

    ChromaticDecomposition(
        :tree,
        bags,
        adhesions,
        :codecomposition,
        seed
    )
end

function collect_subtree(forest::ThreadAncestryForest, root_id::UInt64)
    result = UInt64[root_id]
    node = get(forest.nodes, root_id, nothing)
    if node !== nothing
        for child_id in node.children
            append!(result, collect_subtree(forest, child_id))
        end
    end
    result
end

# ═══════════════════════════════════════════════════════════════════════════════
# Rewriting Gadget Integration
# ═══════════════════════════════════════════════════════════════════════════════

"""
    RewritingGadget

A local rewriting gadget creates parallel copresheaf sections through forks.
The gadget tracks chromatic identity to ensure SPI verification.
"""
struct RewritingGadget{T}
    pattern::T                    # What to match
    replacement::T                # What to replace with
    neighborhood_radius::Int      # How far to look for neighbors
    color_signature::RGB{Float64} # Chromatic identity of this gadget
    preserves_colors::Bool        # Does rewriting preserve color structure?
end

"""
    apply_gadget(gadget::RewritingGadget, d::ChromaticDecomposition, bag_idx::Int)

Apply a rewriting gadget to a specific bag in the decomposition.
Returns the modified decomposition with updated color witnesses.

The key insight: after applying a gadget, we must re-verify the sheaf condition
on the affected neighborhood to ensure global consistency is maintained.
"""
function apply_gadget(gadget::RewritingGadget, d::ChromaticDecomposition, bag_idx::Int)
    # Get local neighborhood
    local_d = local_neighborhood(d, bag_idx)

    # Apply rewrite in the central bag
    bag = d.bags[bag_idx]
    new_elements = copy(bag.elements)
    new_colors = copy(bag.colors)

    for (i, elem) in enumerate(new_elements)
        if elem == gadget.pattern
            new_elements[i] = gadget.replacement
            if !gadget.preserves_colors
                # Blend gadget color with element color
                new_colors[i] = blend_colors(bag.colors[i], gadget.color_signature)
            end
        end
    end

    new_bag = ChromaticBag{eltype(new_elements), eltype(new_colors)}(
        new_elements,
        new_colors,
        hash(new_colors)
    )

    # Update decomposition
    new_bags = copy(d.bags)
    new_bags[bag_idx] = new_bag

    modified_d = ChromaticDecomposition(
        d.shape,
        new_bags,
        d.adhesions,
        d.decomp_type,
        d.seed
    )

    # Re-verify sheaf condition on affected neighborhood
    (consistent, witness, trace) = decide_chromatic_sheaf(modified_d)

    (modified_d, consistent, trace)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Color Utility Functions
# ═══════════════════════════════════════════════════════════════════════════════

"""Check if two colors are compatible (similar hue for neighbor awareness)."""
function colors_compatible(c1::RGB, c2::RGB; threshold::Float64=0.3)
    # Convert to HSL for hue comparison
    h1 = hue(convert(HSL, c1))
    h2 = hue(convert(HSL, c2))

    # Circular hue distance
    hue_dist = min(abs(h1 - h2), 360 - abs(h1 - h2)) / 180

    hue_dist < threshold
end

"""Blend two colors for pullback elements."""
function blend_colors(c1::RGB, c2::RGB)
    RGB(
        (c1.r + c2.r) / 2,
        (c1.g + c2.g) / 2,
        (c1.b + c2.b) / 2
    )
end

"""Filter a bag to only include elements at specified indices."""
function filter_bag(bag::ChromaticBag{T,C}, indices::Vector{Int}) where {T,C}
    valid_indices = filter(i -> i <= length(bag.elements), indices)
    if isempty(valid_indices)
        return ChromaticBag{T,C}(T[], C[], UInt64(0))
    end
    ChromaticBag{T,C}(
        bag.elements[valid_indices],
        bag.colors[valid_indices],
        hash(bag.colors[valid_indices])
    )
end

"""Extract hue from RGB color."""
function hue(c::HSL)
    c.h
end

# ═══════════════════════════════════════════════════════════════════════════════
# Exports
# ═══════════════════════════════════════════════════════════════════════════════

export ColorMorphism, ChromaticBag, ChromaticAdhesion, ChromaticDecomposition
export neighbors, local_neighborhood
export chromatic_adhesion_filter, decide_chromatic_sheaf
export ThreadAncestryNode, ThreadAncestryForest, to_chromatic_decomposition
export RewritingGadget, apply_gadget
export colors_compatible, blend_colors

# ═══════════════════════════════════════════════════════════════════════════════
# Balanced Ternary Decomposition of adhesion_filter (Seed 1069)
# ═══════════════════════════════════════════════════════════════════════════════

"""
Depth-4 balanced ternary decomposition of adhesion_filter.

The algorithm decomposes into 18 atomic operations organized by ternary addresses:
    --- (LIMIT):  pullback, cospan, span structure
    --_ (LEGS):   projections, apex vertex
    --+ (FETCH):  adhesionSpans, indexing, unpacking
    -_- (DOMAIN): dom functor, cospan legs, FinSet
    -__ (COMPOSE): morphism ∘, image inclusion, hom_map
    -_+ (FORCE):  broadcast, lazy thunk, eager value

This mirrors the functor lifting 𝐃: Cat_pullback → Cat.
"""

@enum TritValue MINUS=-1 ZERO=0 PLUS=1

struct TernaryAddress
    trits::NTuple{4, TritValue}
end

function TernaryAddress(s::String)
    mapping = Dict('-' => MINUS, '_' => ZERO, '+' => PLUS)
    trits = ntuple(i -> mapping[s[i]], 4)
    TernaryAddress(trits)
end

Base.show(io::IO, ta::TernaryAddress) = print(io, join([ta.trits[i] == MINUS ? '-' : ta.trits[i] == ZERO ? '_' : '+' for i in 1:4]))

"""
    AdhesionFilterOp

An atomic operation in the adhesion_filter algorithm, addressed by balanced ternary.
"""
struct AdhesionFilterOp
    address::TernaryAddress
    name::Symbol
    description::String
    category::Symbol  # :limit, :legs, :fetch, :domain, :compose, :force
end

const ADHESION_FILTER_OPS = [
    # --- LIMIT cluster
    AdhesionFilterOp(TernaryAddress("----"), :pullback, "Catlab limit", :limit),
    AdhesionFilterOp(TernaryAddress("---_"), :d_csp_cospan, "diagram data", :limit),
    AdhesionFilterOp(TernaryAddress("---+"), :span_structure, "apex + legs", :limit),

    # --_ LEGS cluster
    AdhesionFilterOp(TernaryAddress("--_-"), :left_leg, "legs()[1] left projection", :legs),
    AdhesionFilterOp(TernaryAddress("--__"), :right_leg, "legs()[2] right projection", :legs),
    AdhesionFilterOp(TernaryAddress("--_+"), :p_cone_apex, "apex vertex", :legs),

    # --+ FETCH cluster
    AdhesionFilterOp(TernaryAddress("--+-"), :adhesion_spans, "adhesionSpans() all spans", :fetch),
    AdhesionFilterOp(TernaryAddress("--+_"), :index_span, "[i] specific span", :fetch),
    AdhesionFilterOp(TernaryAddress("--++"), :unpack_tuple, "(csp,d_csp) unpack", :fetch),

    # -_- DOMAIN cluster
    AdhesionFilterOp(TernaryAddress("-_--"), :dom_functor, "dom() domain functor", :domain),
    AdhesionFilterOp(TernaryAddress("-_-_"), :left_cospan_leg, "new_d_csp[1]", :domain),
    AdhesionFilterOp(TernaryAddress("-_-+"), :finset_object, "FinSet object", :domain),

    # -__ COMPOSE cluster
    AdhesionFilterOp(TernaryAddress("-__-"), :compose_morphism, "compose() morphism ∘", :compose),
    AdhesionFilterOp(TernaryAddress("-___"), :image_inclusion, "imgs[1] image", :compose),
    AdhesionFilterOp(TernaryAddress("-__+"), :hom_map_apply, "hom_map(d,f)", :compose),

    # -_+ FORCE cluster
    AdhesionFilterOp(TernaryAddress("-_+-"), :force_broadcast, "force. broadcast", :force),
    AdhesionFilterOp(TernaryAddress("-_+_"), :lazy_thunk, "lazy deferred", :force),
    AdhesionFilterOp(TernaryAddress("-_++"), :eager_value, "eager computed", :force),
]

"""
    ternary_execution_trace(d::ChromaticDecomposition, adhesion_idx::Int)

Execute adhesion_filter with ternary-addressed operation tracing.
Returns the filtered decomposition plus a trace of which operations fired.
"""
function ternary_execution_trace(d::ChromaticDecomposition, adhesion_idx::Int)
    trace = Symbol[]

    # --+ FETCH: get adhesion span
    push!(trace, :adhesion_spans)
    push!(trace, :index_span)
    push!(trace, :unpack_tuple)

    if adhesion_idx > length(d.adhesions)
        return (d, trace)
    end

    adh = d.adhesions[adhesion_idx]
    left_bag = d.bags[adhesion_idx]
    right_bag = d.bags[min(adhesion_idx + 1, length(d.bags))]

    # --- LIMIT: compute pullback conceptually
    push!(trace, :d_csp_cospan)
    push!(trace, :span_structure)
    push!(trace, :pullback)

    # --_ LEGS: extract projections
    push!(trace, :left_leg)
    push!(trace, :right_leg)
    push!(trace, :p_cone_apex)

    surviving_left = Int[]
    surviving_right = Int[]

    for (li, left_idx) in enumerate(adh.left_leg)
        for (ri, right_idx) in enumerate(adh.right_leg)
            if left_idx <= length(left_bag.elements) &&
               right_idx <= length(right_bag.elements)

                # -__ COMPOSE: check compatibility via composition
                push!(trace, :compose_morphism)
                push!(trace, :image_inclusion)
                push!(trace, :hom_map_apply)

                left_c = left_bag.colors[min(left_idx, length(left_bag.colors))]
                right_c = right_bag.colors[min(right_idx, length(right_bag.colors))]

                if colors_compatible(left_c, right_c)
                    push!(surviving_left, left_idx)
                    push!(surviving_right, right_idx)
                end
            end
        end
    end

    # -_- DOMAIN: construct new domain
    push!(trace, :dom_functor)
    push!(trace, :left_cospan_leg)
    push!(trace, :finset_object)

    # -_+ FORCE: materialize results
    push!(trace, :force_broadcast)
    push!(trace, :lazy_thunk)
    push!(trace, :eager_value)

    new_left = filter_bag(left_bag, unique(surviving_left))
    new_right = filter_bag(right_bag, unique(surviving_right))

    new_bags = copy(d.bags)
    new_bags[adhesion_idx] = new_left
    if adhesion_idx + 1 <= length(new_bags)
        new_bags[adhesion_idx + 1] = new_right
    end

    filtered_d = ChromaticDecomposition(
        d.shape,
        new_bags,
        d.adhesions,
        d.decomp_type,
        d.seed
    )

    (filtered_d, trace)
end

"""
    seed_1069_signature()

The balanced ternary representation of seed 1069: [+1, -1, -1, +1, +1, +1, +1]
This is the chromatic identity signature for the IES system.
"""
function seed_1069_signature()
    # 1069 in balanced ternary = +--++++
    trits = [PLUS, MINUS, MINUS, PLUS, PLUS, PLUS, PLUS]

    # Convert to color signature
    colors = map(enumerate(trits)) do (i, t)
        base_hue = i * (360 / 7)
        saturation = t == PLUS ? 1.0 : t == MINUS ? 0.5 : 0.75
        lightness = t == PLUS ? 0.5 : t == MINUS ? 0.3 : 0.4
        convert(RGB, HSL(base_hue, saturation, lightness))
    end

    (trits=trits, colors=colors, value=1069)
end

export TritValue, TernaryAddress, AdhesionFilterOp, ADHESION_FILTER_OPS
export ternary_execution_trace, seed_1069_signature

end # module SheafACSetIntegration
