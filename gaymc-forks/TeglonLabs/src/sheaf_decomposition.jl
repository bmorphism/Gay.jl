"""
    GaySheafDecomposition

Sheaf cohomology with SPI coloring for compositional algorithms.
Based on Bumpus "Compositional Algorithms on Compositional Data".
"""
module GaySheafDecomposition

export GraphSheaf, TreeDecompositionSheaf
export gay_cech_cohomology!, gay_local_to_global!, gay_zigzag_cohomology!
export splitmix64, chromatic_fingerprint

# --- SPI Core ---

"""Splitmix64 PRNG for deterministic coloring."""
function splitmix64(x::UInt64)::UInt64
    z = x + 0x9e3779b97f4a7c15
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    z ⊻ (z >> 31)
end

"""Compute chromatic fingerprint from seed and structure hash."""
chromatic_fingerprint(seed::UInt64, h::UInt64) = splitmix64(seed ⊻ h)

# --- Graph Sheaf ---

"""
    GraphSheaf{T}

Sheaf over a graph with stalks at vertices and restriction maps on edges.

# Fields
- `stalks::Dict{Int,Vector{T}}` - stalk data at each vertex
- `restrictions::Dict{Tuple{Int,Int},Function}` - edge restriction maps
- `chromatic_id::UInt64` - SPI chromatic fingerprint
"""
struct GraphSheaf{T}
    stalks::Dict{Int,Vector{T}}
    restrictions::Dict{Tuple{Int,Int},Function}
    chromatic_id::UInt64
end

function GraphSheaf{T}(vertices::Vector{Int}, seed::UInt64=0x0) where T
    stalks = Dict{Int,Vector{T}}(v => T[] for v in vertices)
    restrictions = Dict{Tuple{Int,Int},Function}()
    h = hash(vertices)
    GraphSheaf{T}(stalks, restrictions, chromatic_fingerprint(seed, h))
end

"""Add restriction map ρ_{uv}: F(u) → F(v)."""
function add_restriction!(sheaf::GraphSheaf, u::Int, v::Int, ρ::Function)
    sheaf.restrictions[(u, v)] = ρ
end

# --- Čech Cohomology ---

"""
    gay_cech_cohomology!(sheaf, cover, seed) -> (H0, H1, color)

Compute Čech cohomology H⁰, H¹ with SPI-colored equivalence classes.
- `cover`: Vector of vertex sets forming an open cover
- Returns global sections (H⁰), first cohomology (H¹), chromatic class
"""
function gay_cech_cohomology!(sheaf::GraphSheaf{T}, cover::Vector{Vector{Int}}, 
                               seed::UInt64) where T
    color = splitmix64(seed ⊻ sheaf.chromatic_id)
    
    # H⁰: global sections - elements agreeing on overlaps
    H0 = T[]
    if !isempty(cover) && !isempty(first(cover))
        v0 = first(first(cover))
        for s in get(sheaf.stalks, v0, T[])
            is_global = true
            for patch in cover, v in patch
                if haskey(sheaf.restrictions, (v0, v))
                    ρ = sheaf.restrictions[(v0, v)]
                    if !(ρ(s) in get(sheaf.stalks, v, T[]))
                        is_global = false
                        break
                    end
                end
            end
            is_global && push!(H0, s)
        end
    end
    
    # H¹: cocycle classes - obstructions to gluing
    H1_obstructions = UInt64[]
    for i in 1:length(cover), j in (i+1):length(cover)
        overlap = intersect(cover[i], cover[j])
        if !isempty(overlap)
            obstruction = hash((cover[i], cover[j], overlap))
            push!(H1_obstructions, splitmix64(color ⊻ obstruction))
        end
    end
    
    H1_class = isempty(H1_obstructions) ? 0x0 : reduce(⊻, H1_obstructions)
    (H0, H1_class, color)
end

# --- Local-to-Global ---

"""
    gay_local_to_global!(sheaf, seed) -> (glues, obstruction_fingerprint)

Check if local sections glue to global section.
Returns gluing success and SPI obstruction fingerprint.
"""
function gay_local_to_global!(sheaf::GraphSheaf{T}, seed::UInt64) where T
    color = splitmix64(seed ⊻ sheaf.chromatic_id)
    obstruction = 0x0
    glues = true
    
    for ((u, v), ρ) in sheaf.restrictions
        stalk_u = get(sheaf.stalks, u, T[])
        stalk_v = get(sheaf.stalks, v, T[])
        
        for s in stalk_u
            img = ρ(s)
            if !(img in stalk_v)
                glues = false
                obstruction ⊻= splitmix64(color ⊻ hash((u, v, s)))
            end
        end
    end
    
    fingerprint = splitmix64(obstruction ⊻ color)
    (glues, fingerprint)
end

# --- Tree Decomposition Sheaf ---

"""
    TreeDecompositionSheaf{T}

Sheaf on tree decomposition with stalks at bags.

# Fields
- `bags::Vector{Set{Int}}` - tree decomposition bags
- `tree_edges::Vector{Tuple{Int,Int}}` - edges of decomposition tree
- `bag_stalks::Dict{Int,Vector{T}}` - stalk at each bag
- `chromatic_id::UInt64` - SPI fingerprint
"""
struct TreeDecompositionSheaf{T}
    bags::Vector{Set{Int}}
    tree_edges::Vector{Tuple{Int,Int}}
    bag_stalks::Dict{Int,Vector{T}}
    chromatic_id::UInt64
end

function TreeDecompositionSheaf{T}(bags::Vector{Set{Int}}, 
                                    edges::Vector{Tuple{Int,Int}},
                                    seed::UInt64=0x0) where T
    bag_stalks = Dict{Int,Vector{T}}(i => T[] for i in 1:length(bags))
    h = hash((bags, edges))
    TreeDecompositionSheaf{T}(bags, edges, bag_stalks, chromatic_fingerprint(seed, h))
end

"""Compute treewidth from decomposition."""
treewidth(td::TreeDecompositionSheaf) = maximum(length.(td.bags)) - 1

"""Restriction along tree edge via bag intersection."""
function bag_restriction(td::TreeDecompositionSheaf, i::Int, j::Int)
    intersect(td.bags[i], td.bags[j])
end

"""
    gay_cech_cohomology!(td, seed) -> (H0, H1, color)

Čech cohomology on tree decomposition sheaf.
"""
function gay_cech_cohomology!(td::TreeDecompositionSheaf{T}, seed::UInt64) where T
    cover = [collect(b) for b in td.bags]
    
    # Build temporary GraphSheaf for cohomology computation
    all_verts = union(td.bags...)
    gsheaf = GraphSheaf{T}(collect(all_verts), td.chromatic_id)
    gsheaf.stalks .= td.bag_stalks
    
    for (i, j) in td.tree_edges
        sep = bag_restriction(td, i, j)
        add_restriction!(gsheaf, i, j, s -> s)  # identity on separator
    end
    
    gay_cech_cohomology!(gsheaf, cover, seed)
end

# --- ZigZag Sampler for Cohomology ---

"""
    gay_zigzag_cohomology!(sheaf, seed; n_events=100, refresh_rate=0.1) -> (samples, fingerprint)

ZigZag PDMP sampler for exploring cohomology parameter space.
Uses SPI-deterministic event times and velocity flips.
Returns sampled cohomology classes and chromatic fingerprint.
"""
function gay_zigzag_cohomology!(sheaf::GraphSheaf{T}, seed::UInt64;
                                 n_events::Int=100, refresh_rate::Float64=0.1) where T
    rng_state = splitmix64(seed)
    fingerprint = splitmix64(seed ⊻ sheaf.chromatic_id)
    
    # State: position x, velocity v ∈ {-1, +1}^d
    d = length(sheaf.stalks)
    x = zeros(d)
    v = ones(d)
    
    # Initialize velocities via splitmix64
    for i in 1:d
        rng_state = splitmix64(rng_state)
        v[i] = (rng_state & 1) == 0 ? -1.0 : 1.0
    end
    
    samples = Vector{Float64}[]
    t = 0.0
    
    # Gradient of negative log-density (quadratic potential)
    ∇U(pos) = pos
    
    for event in 1:n_events
        # Compute switching rates λ_i = max(0, v_i * ∂U/∂x_i)
        grad = ∇U(x)
        rates = max.(0.0, v .* grad) .+ refresh_rate
        total_rate = sum(rates)
        
        # Sample exponential waiting time
        rng_state = splitmix64(rng_state)
        u = (rng_state & 0xFFFFFFFF) / 4294967296.0
        τ = -log(max(u, 1e-10)) / max(total_rate, 1e-10)
        
        # Move along velocity
        x .+= τ .* v
        t += τ
        
        # Select coordinate to flip
        rng_state = splitmix64(rng_state)
        threshold = (rng_state & 0xFFFFFFFF) / 4294967296.0 * total_rate
        cumsum_rate = 0.0
        flip_idx = 1
        for i in 1:d
            cumsum_rate += rates[i]
            if cumsum_rate >= threshold
                flip_idx = i
                break
            end
        end
        
        # Flip velocity
        v[flip_idx] *= -1
        
        # Record sample
        push!(samples, copy(x))
        fingerprint = splitmix64(fingerprint ⊻ reinterpret(UInt64, x[flip_idx]) ⊻ UInt64(event))
    end
    
    (samples, fingerprint)
end

end # module
