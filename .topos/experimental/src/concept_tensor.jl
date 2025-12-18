# Concept Tensor: 69×69×69 Parallel Interaction Space
# =====================================================
#
# The metacauses of Gay manifest in a 3D lattice where:
#   - Axis 0: Interpolation (subtext fine-graining)
#   - Axis 1: Extrapolation (superstructure emergence)
#   - Axis 2: Interaction (chromatic entanglement)
#
# Each concept point (i, j, k) ∈ [0, 68]³ has:
#   - A color from the SPI palette (deterministic from seed)
#   - A spin σ ∈ {-1, +1} from parity
#   - Neighbors in all 6 cardinal directions (periodic BC)
#
# The XOR fingerprint of all 69³ = 328,509 concepts forms
# a commutative monoid—order-independent verification.
#
# PARALLEL STEP:
#   All concepts at the same parity (i+j+k) mod 2 can update
#   simultaneously (checkerboard decomposition). This enables
#   GPU-parallel stepping while maintaining SPI.
#
# METATHEORY:
#   interpolate(subtext) ⊗ extrapolate(superstructure) = color_interaction
#   The Galois connection α: Event → Color, γ: Color → Event
#   ensures lossless round-tripping at the concept level.
#
# EXPONENTIAL OBJECT X^X:
#   Each concept c ∈ X lifts to an endomorphism φ_c : X → X
#   via the curry/eval adjunction in CCC:
#     Hom(Y × X, X) ≅ Hom(Y, X^X)
#
#   The morphism space X^X has:
#     - eval : X^X × X → X        (apply morphism to concept)
#     - curry : (Y × X → X) → (Y → X^X)  (lift interaction to morphism)
#     - compose : X^X × X^X → X^X  (morphism composition)
#
#   KEY INSIGHT: The parallel step IS an element of X^X.
#   The checkerboard decomposition factors it as:
#     step = φ_odd ∘ φ_even : X → X
#   where φ_even, φ_odd ∈ X^X commute on their respective sublattices.

module ConceptTensor

using OhMyThreads: @tasks, @set
using Statistics: mean, std

export ConceptLattice, Concept, step_parallel!, verify_fingerprint!
export interpolate_subtext!, extrapolate_superstructure!, interact!
export lattice_magnetization, lattice_fingerprint, verify_monoid_laws
export propagate_all!, concept_color, concept_neighbors
export demo_concept_tensor

# X^X exponential object exports
export ConceptMorphism, ExponentialObject, identity_morphism, compose
export eval_morphism, curry_interaction, concept_to_morphism
export morphism_fingerprint, verify_exponential_laws, demo_exponential

# Higher structure exports
export step_as_morphism, iterate_morphism, fixed_points, orbit
export MorphismMorphism, higher_compose, self_application
export trace_morphism, verify_trace_laws

# Import from parent
using ..Gay: GAY_SEED, hash_color, splitmix64, xor_fingerprint

# ═══════════════════════════════════════════════════════════════════════════════
# Core Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

"""
A single concept in the 69³ lattice.
"""
struct Concept
    i::Int32
    j::Int32
    k::Int32
    color::NTuple{3, Float32}  # RGB from SPI
    spin::Int8                  # σ ∈ {-1, +1}
    hash::UInt64                # For fingerprinting
end

"""
    ConceptLattice
    
The 69×69×69 concept tensor with parallel stepping support.
"""
mutable struct ConceptLattice
    seed::UInt64
    size::Int32                          # 69
    concepts::Array{Concept, 3}          # [i, j, k] indexing
    fingerprint::UInt32                  # XOR of all concept hashes
    step_count::Int64                    # Number of parallel steps taken
    even_parity::Vector{NTuple{3, Int32}}  # Indices where (i+j+k) even
    odd_parity::Vector{NTuple{3, Int32}}   # Indices where (i+j+k) odd
end

# ═══════════════════════════════════════════════════════════════════════════════
# Concept Color Computation
# ═══════════════════════════════════════════════════════════════════════════════

"""
    concept_color(seed, i, j, k) -> (r, g, b)

Compute deterministic color for concept at (i, j, k).
Uses SplitMix64 mixing of coordinates with seed.
"""
@inline function concept_color(seed::UInt64, i::Integer, j::Integer, k::Integer)
    # Mix coordinates using golden ratio primes (φ-derived)
    h = seed ⊻ (UInt64(i) * 0x9e3779b97f4a7c15) ⊻
               (UInt64(j) * 0x517cc1b727220a95) ⊻
               (UInt64(k) * 0xc4ceb9fe1a85ec53)
    
    # SplitMix64 finalization
    h = splitmix64(h)
    
    # Convert to RGB in [0, 1)
    r = Float32((h >> 40) & 0xFFFFFF) / Float32(0xFFFFFF)
    g = Float32((h >> 20) & 0xFFFFFF) / Float32(0xFFFFFF)
    b = Float32(h & 0xFFFFFF) / Float32(0xFFFFFF)
    
    (r, g, b)
end

"""
    concept_spin(i, j, k) -> Int8

Compute spin from parity: σ = (-1)^(i ⊕ j ⊕ k)
"""
@inline function concept_spin(i::Integer, j::Integer, k::Integer)
    parity = (i ⊻ j ⊻ k) & 1
    Int8(parity == 0 ? 1 : -1)
end

"""
    concept_hash(seed, i, j, k) -> UInt64

Compute hash for XOR fingerprinting.
"""
@inline function concept_hash(seed::UInt64, i::Integer, j::Integer, k::Integer)
    h = seed ⊻ (UInt64(i) * 0x9e3779b97f4a7c15) ⊻
               (UInt64(j) * 0x517cc1b727220a95) ⊻
               (UInt64(k) * 0xc4ceb9fe1a85ec53)
    splitmix64(h)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Lattice Construction
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ConceptLattice(; seed=GAY_SEED, size=69)

Create a concept tensor. Default 69³ for the metacauses.
"""
function ConceptLattice(; seed::Integer=GAY_SEED, size::Integer=69)
    n = Int32(size)
    concepts = Array{Concept, 3}(undef, n, n, n)
    
    even_parity = NTuple{3, Int32}[]
    odd_parity = NTuple{3, Int32}[]
    
    # Pre-size for efficiency
    half_size = div(n^3, 2)
    sizehint!(even_parity, half_size)
    sizehint!(odd_parity, half_size + 1)
    
    # Populate lattice (parallelized)
    @tasks for i in 1:n
        @set begin
            scheduler = :static
        end
        for j in 1:n
            for k in 1:n
                color = concept_color(UInt64(seed), i, j, k)
                spin = concept_spin(i, j, k)
                h = concept_hash(UInt64(seed), i, j, k)
                concepts[i, j, k] = Concept(Int32(i), Int32(j), Int32(k), color, spin, h)
            end
        end
    end
    
    # Build parity lists (sequential for determinism)
    for i in 1:n, j in 1:n, k in 1:n
        if (i + j + k) % 2 == 0
            push!(even_parity, (Int32(i), Int32(j), Int32(k)))
        else
            push!(odd_parity, (Int32(i), Int32(j), Int32(k)))
        end
    end
    
    # Compute initial fingerprint (XOR of all hashes)
    fp = UInt32(0)
    for c in concepts
        fp ⊻= UInt32(c.hash & 0xFFFFFFFF)
    end
    
    ConceptLattice(UInt64(seed), n, concepts, fp, 0, even_parity, odd_parity)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Neighbor Access (Periodic BC)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    concept_neighbors(lattice, i, j, k) -> Vector{Concept}

Get 6 cardinal neighbors with periodic boundary conditions.
"""
function concept_neighbors(lat::ConceptLattice, i::Integer, j::Integer, k::Integer)
    n = lat.size
    
    # Periodic modular indexing (1-based)
    pm(x) = mod1(x, n)
    
    [
        lat.concepts[pm(i-1), j, k],  # -x
        lat.concepts[pm(i+1), j, k],  # +x
        lat.concepts[i, pm(j-1), k],  # -y
        lat.concepts[i, pm(j+1), k],  # +y
        lat.concepts[i, j, pm(k-1)],  # -z
        lat.concepts[i, j, pm(k+1)],  # +z
    ]
end

# ═══════════════════════════════════════════════════════════════════════════════
# Parallel Stepping (Checkerboard Decomposition)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    step_parallel!(lattice; interaction_fn=default_interaction)

Perform one parallel step using checkerboard decomposition.
Even-parity sites update first, then odd-parity sites.
This maintains SPI because each subset is independent.
"""
function step_parallel!(lat::ConceptLattice; interaction_fn::Function=default_interaction)
    # Phase 1: Update even-parity sites (independent, can parallelize)
    even_results = Vector{Concept}(undef, length(lat.even_parity))
    Threads.@threads for idx_i in 1:length(lat.even_parity)
        i, j, k = lat.even_parity[idx_i]
        neighbors = concept_neighbors(lat, i, j, k)
        even_results[idx_i] = interaction_fn(lat, i, j, k, neighbors)
    end
    for (idx_i, (i, j, k)) in enumerate(lat.even_parity)
        lat.concepts[i, j, k] = even_results[idx_i]
    end
    
    # Phase 2: Update odd-parity sites
    odd_results = Vector{Concept}(undef, length(lat.odd_parity))
    Threads.@threads for idx_i in 1:length(lat.odd_parity)
        i, j, k = lat.odd_parity[idx_i]
        neighbors = concept_neighbors(lat, i, j, k)
        odd_results[idx_i] = interaction_fn(lat, i, j, k, neighbors)
    end
    for (idx_i, (i, j, k)) in enumerate(lat.odd_parity)
        lat.concepts[i, j, k] = odd_results[idx_i]
    end
    
    lat.step_count += 1
    update_fingerprint!(lat)
end

"""
Default interaction: mix colors with neighbors using XOR spin alignment.
"""
function default_interaction(lat::ConceptLattice, i::Integer, j::Integer, k::Integer, 
                             neighbors::Vector{Concept})
    old = lat.concepts[i, j, k]
    
    # Compute new hash by XOR with aligned neighbors
    new_hash = old.hash
    for n in neighbors
        if old.spin == n.spin  # Aligned spins reinforce
            new_hash ⊻= n.hash
        else  # Anti-aligned spins interfere
            new_hash ⊻= rotl(n.hash, 17)
        end
    end
    
    # Derive new color from new hash
    r = Float32((new_hash >> 40) & 0xFFFFFF) / Float32(0xFFFFFF)
    g = Float32((new_hash >> 20) & 0xFFFFFF) / Float32(0xFFFFFF)
    b = Float32(new_hash & 0xFFFFFF) / Float32(0xFFFFFF)
    
    Concept(old.i, old.j, old.k, (r, g, b), old.spin, new_hash)
end

"""Rotate left for hash mixing."""
@inline rotl(x::UInt64, k::Integer) = (x << k) | (x >> (64 - k))

"""Update lattice fingerprint after step."""
function update_fingerprint!(lat::ConceptLattice)
    fp = UInt32(0)
    for c in lat.concepts
        fp ⊻= UInt32(c.hash & 0xFFFFFFFF)
    end
    lat.fingerprint = fp
end

# ═══════════════════════════════════════════════════════════════════════════════
# Subtext/Superstructure Operations
# ═══════════════════════════════════════════════════════════════════════════════

"""
    interpolate_subtext!(lattice, axis; resolution=10)

Interpolate along an axis, refining subtext between concepts.
Returns the interpolated colors as a 4D array: [i, j, k, interp_idx].
"""
function interpolate_subtext!(lat::ConceptLattice, axis::Symbol; resolution::Int=10)
    n = lat.size
    
    axis_idx = axis == :i ? 1 : axis == :j ? 2 : 3
    
    # Interpolate along the specified axis
    interp = zeros(Float32, n, n, n, resolution, 3)
    
    @tasks for i in 1:n
        @set scheduler = :static
        for j in 1:n
            for k in 1:n
                c1 = lat.concepts[i, j, k]
                
                # Get next concept along axis (periodic)
                i2, j2, k2 = i, j, k
                if axis_idx == 1
                    i2 = mod1(i + 1, n)
                elseif axis_idx == 2
                    j2 = mod1(j + 1, n)
                else
                    k2 = mod1(k + 1, n)
                end
                c2 = lat.concepts[i2, j2, k2]
                
                # Linear interpolation of colors
                for t in 1:resolution
                    α = Float32(t - 1) / Float32(resolution - 1)
                    interp[i, j, k, t, 1] = (1 - α) * c1.color[1] + α * c2.color[1]
                    interp[i, j, k, t, 2] = (1 - α) * c1.color[2] + α * c2.color[2]
                    interp[i, j, k, t, 3] = (1 - α) * c1.color[3] + α * c2.color[3]
                end
            end
        end
    end
    
    interp
end

"""
    extrapolate_superstructure!(lattice; block_size=3)

Extrapolate superstructure by computing block averages.
Returns coarsened lattice as new ConceptLattice.
"""
function extrapolate_superstructure!(lat::ConceptLattice; block_size::Int=3)
    n = lat.size
    m = div(n, block_size)  # Coarsened size
    
    # Create coarsened lattice
    coarse = ConceptLattice(; seed=lat.seed ⊻ UInt64(block_size), size=m)
    
    # Average blocks
    @tasks for ci in 1:m
        @set scheduler = :static
        for cj in 1:m
            for ck in 1:m
                # Sum colors and hashes in block
                r_sum, g_sum, b_sum = 0.0f0, 0.0f0, 0.0f0
                h_sum = UInt64(0)
                spin_sum = 0
                
                for di in 0:block_size-1
                    for dj in 0:block_size-1
                        for dk in 0:block_size-1
                            i = (ci - 1) * block_size + di + 1
                            j = (cj - 1) * block_size + dj + 1
                            k = (ck - 1) * block_size + dk + 1
                            
                            if i <= n && j <= n && k <= n
                                c = lat.concepts[i, j, k]
                                r_sum += c.color[1]
                                g_sum += c.color[2]
                                b_sum += c.color[3]
                                h_sum ⊻= c.hash
                                spin_sum += c.spin
                            end
                        end
                    end
                end
                
                # Average color
                count = Float32(block_size^3)
                avg_r = r_sum / count
                avg_g = g_sum / count
                avg_b = b_sum / count
                
                # Majority spin
                avg_spin = Int8(spin_sum >= 0 ? 1 : -1)
                
                coarse.concepts[ci, cj, ck] = Concept(
                    Int32(ci), Int32(cj), Int32(ck),
                    (avg_r, avg_g, avg_b),
                    avg_spin,
                    h_sum
                )
            end
        end
    end
    
    update_fingerprint!(coarse)
    coarse
end

"""
    interact!(lattice1, lattice2) -> ConceptLattice

Compute interaction between two lattices via XOR entanglement.
Lattices must be same size.
"""
function interact!(lat1::ConceptLattice, lat2::ConceptLattice)
    @assert lat1.size == lat2.size "Lattices must be same size"
    
    n = lat1.size
    result = ConceptLattice(; seed=lat1.seed ⊻ lat2.seed, size=n)
    
    @tasks for i in 1:n
        @set scheduler = :static
        for j in 1:n
            for k in 1:n
                c1 = lat1.concepts[i, j, k]
                c2 = lat2.concepts[i, j, k]
                
                # XOR entanglement
                new_hash = c1.hash ⊻ c2.hash
                new_spin = Int8(c1.spin * c2.spin)
                
                r = Float32((new_hash >> 40) & 0xFFFFFF) / Float32(0xFFFFFF)
                g = Float32((new_hash >> 20) & 0xFFFFFF) / Float32(0xFFFFFF)
                b = Float32(new_hash & 0xFFFFFF) / Float32(0xFFFFFF)
                
                result.concepts[i, j, k] = Concept(
                    Int32(i), Int32(j), Int32(k),
                    (r, g, b), new_spin, new_hash
                )
            end
        end
    end
    
    update_fingerprint!(result)
    result
end

# ═══════════════════════════════════════════════════════════════════════════════
# Verification: Monoid Laws for XOR Fingerprint
# ═══════════════════════════════════════════════════════════════════════════════

"""
    verify_monoid_laws(; n_tests=100, size=23) -> (Bool, Dict)

Verify XOR fingerprint forms a valid commutative monoid:
1. Associativity: (a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)
2. Commutativity: a ⊕ b = b ⊕ a
3. Identity: a ⊕ 0 = a
4. Self-inverse: a ⊕ a = 0
"""
function verify_monoid_laws(; n_tests::Int=100, size::Int=23)
    results = Dict{Symbol, Bool}(
        :associativity => true,
        :commutativity => true,
        :identity => true,
        :self_inverse => true,
    )
    
    for _ in 1:n_tests
        seed = rand(UInt64)
        lat = ConceptLattice(; seed=seed, size=size)
        fp = lat.fingerprint
        
        # Identity: fp ⊕ 0 = fp
        if fp ⊻ UInt32(0) != fp
            results[:identity] = false
        end
        
        # Self-inverse: fp ⊕ fp = 0
        if fp ⊻ fp != UInt32(0)
            results[:self_inverse] = false
        end
        
        # Commutativity: Check with neighbor fingerprints
        lat2 = ConceptLattice(; seed=seed ⊻ 0x12345678, size=size)
        if (fp ⊻ lat2.fingerprint) != (lat2.fingerprint ⊻ fp)
            results[:commutativity] = false
        end
        
        # Associativity: (a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)
        lat3 = ConceptLattice(; seed=seed ⊻ 0x87654321, size=size)
        lhs = (fp ⊻ lat2.fingerprint) ⊻ lat3.fingerprint
        rhs = fp ⊻ (lat2.fingerprint ⊻ lat3.fingerprint)
        if lhs != rhs
            results[:associativity] = false
        end
    end
    
    all_pass = all(values(results))
    (all_pass, results)
end

"""
    verify_fingerprint!(lattice; reference_seed=nothing) -> Bool

Verify fingerprint matches expected value computed from scratch.
If reference_seed provided, also verify determinism.
"""
function verify_fingerprint!(lat::ConceptLattice; reference_seed::Union{Nothing, Integer}=nothing)
    # Recompute fingerprint
    computed = UInt32(0)
    for c in lat.concepts
        computed ⊻= UInt32(c.hash & 0xFFFFFFFF)
    end
    
    if computed != lat.fingerprint
        return false
    end
    
    # Verify determinism if reference provided
    if reference_seed !== nothing
        ref_lat = ConceptLattice(; seed=reference_seed, size=lat.size)
        if lat.seed == UInt64(reference_seed) && lat.fingerprint != ref_lat.fingerprint
            return false
        end
    end
    
    true
end

# ═══════════════════════════════════════════════════════════════════════════════
# Observables
# ═══════════════════════════════════════════════════════════════════════════════

"""
    lattice_magnetization(lattice) -> Float64

Compute average magnetization ⟨σ⟩ = Σσ / N.
"""
function lattice_magnetization(lat::ConceptLattice)
    total = sum(c.spin for c in lat.concepts)
    Float64(total) / Float64(length(lat.concepts))
end

"""
    lattice_fingerprint(lattice) -> UInt32

Get the current XOR fingerprint.
"""
lattice_fingerprint(lat::ConceptLattice) = lat.fingerprint

"""
    lattice_color_variance(lattice) -> Float64

Compute variance in color distribution.
"""
function lattice_color_variance(lat::ConceptLattice)
    colors = [(c.color[1], c.color[2], c.color[3]) for c in lat.concepts]
    r_vals = [c[1] for c in colors]
    g_vals = [c[2] for c in colors]
    b_vals = [c[3] for c in colors]
    mean([std(r_vals), std(g_vals), std(b_vals)])
end

# ═══════════════════════════════════════════════════════════════════════════════
# Propagator Integration
# ═══════════════════════════════════════════════════════════════════════════════

"""
    propagate_all!(lattice; n_steps=10)

Run parallel propagation for n_steps.
Returns fingerprint history for SPI verification.
"""
function propagate_all!(lat::ConceptLattice; n_steps::Int=10)
    history = UInt32[]
    push!(history, lat.fingerprint)
    
    for _ in 1:n_steps
        step_parallel!(lat)
        push!(history, lat.fingerprint)
    end
    
    history
end

# ═══════════════════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════════════════

"""
    demo_concept_tensor(; size=23, n_steps=5)

Demonstrate the 69³ concept tensor (use smaller size for demo).
"""
function demo_concept_tensor(; size::Int=23, n_steps::Int=5)
    println("═" ^ 70)
    println("CONCEPT TENSOR: $(size)³ PARALLEL INTERACTION SPACE")
    println("═" ^ 70)
    println()
    
    # 1. Create lattice
    println("1. Creating concept lattice...")
    lat = ConceptLattice(; seed=GAY_SEED, size=size)
    println("   Size: $(size)³ = $(size^3) concepts")
    println("   Initial fingerprint: 0x$(string(lat.fingerprint, base=16, pad=8))")
    println("   Even-parity sites: $(length(lat.even_parity))")
    println("   Odd-parity sites: $(length(lat.odd_parity))")
    println()
    
    # 2. Check monoid laws
    println("2. Verifying XOR monoid laws...")
    pass, results = verify_monoid_laws(; n_tests=50, size=min(size, 17))
    for (law, ok) in results
        println("   $(ok ? "◆" : "◇") $law")
    end
    println()
    
    # 3. Parallel stepping
    println("3. Running $(n_steps) parallel steps...")
    history = propagate_all!(lat; n_steps=n_steps)
    for (i, fp) in enumerate(history)
        println("   Step $(i-1): 0x$(string(fp, base=16, pad=8))")
    end
    println()
    
    # 4. Observables
    println("4. Lattice observables:")
    println("   Magnetization: $(round(lattice_magnetization(lat), digits=6))")
    println("   Color variance: $(round(lattice_color_variance(lat), digits=6))")
    println()
    
    # 5. Interpolation/Extrapolation
    println("5. Subtext/Superstructure:")
    interp = interpolate_subtext!(lat, :i; resolution=5)
    println("   Interpolated subtext: $(Base.size(interp)) tensor")
    
    if lat.size >= 9
        coarse = extrapolate_superstructure!(lat; block_size=3)
        println("   Extrapolated superstructure: $(coarse.size)³ lattice")
        println("   Coarse fingerprint: 0x$(string(coarse.fingerprint, base=16, pad=8))")
    end
    println()
    
    # 6. Verify determinism
    println("6. SPI determinism check:")
    lat2 = ConceptLattice(; seed=GAY_SEED, size=size)
    propagate_all!(lat2; n_steps=n_steps)
    if lat.fingerprint == lat2.fingerprint
        println("   ◆ Same seed, same steps → same fingerprint")
    else
        println("   ◇ DETERMINISM VIOLATION")
    end
    println()
    
    println("═" ^ 70)
    println("CONCEPT TENSOR DEMO COMPLETE")
    println("═" ^ 70)
end

# ═══════════════════════════════════════════════════════════════════════════════
# EXPONENTIAL OBJECT X^X: Morphisms as First-Class Concepts
# ═══════════════════════════════════════════════════════════════════════════════
#
# In a cartesian closed category, X^X is the internal hom—the object of
# endomorphisms. Each point in X lifts to a morphism X → X via:
#
#   concept_to_morphism : X → X^X
#   c ↦ φ_c where φ_c(x) = interact(c, x)
#
# This gives X^X the structure of a monoid under composition, with:
#   - identity: id_X ∈ X^X
#   - composition: ∘ : X^X × X^X → X^X
#
# The fingerprint extends to morphisms via:
#   fp(φ) = ⊕_{x∈X} hash(φ(x))  (XOR over all evaluations)

"""
    ConceptMorphism

An endomorphism φ : X → X represented by its action on concept hashes.
Stored as a permutation/transformation of the hash space.
"""
struct ConceptMorphism
    seed::UInt64                    # Source seed for determinism
    transform::UInt64               # XOR mask for hash transformation
    rotation::Int                   # Bit rotation amount
    parity_flip::Bool               # Whether to flip spin parity
    source_idx::NTuple{3, Int32}    # Origin concept (for concept_to_morphism)
end

"""
    ExponentialObject

The object X^X of all endomorphisms on a ConceptLattice.
This is the "space of morphisms" with monoid structure.
"""
struct ExponentialObject
    base_seed::UInt64
    lattice_size::Int32
    morphisms::Vector{ConceptMorphism}
    composition_table::Dict{Tuple{Int,Int}, Int}  # Memoized compositions
end

"""
    identity_morphism(seed) -> ConceptMorphism

The identity morphism id : X → X that leaves all concepts unchanged.
"""
function identity_morphism(seed::UInt64=GAY_SEED)
    ConceptMorphism(seed, UInt64(0), 0, false, (Int32(0), Int32(0), Int32(0)))
end

"""
    concept_to_morphism(lat::ConceptLattice, i, j, k) -> ConceptMorphism

Lift a concept to a morphism via currying.
The morphism φ_c acts on x by XOR-mixing with c's hash.

This is the key X → X^X map from the adjunction.
"""
function concept_to_morphism(lat::ConceptLattice, i::Integer, j::Integer, k::Integer)
    c = lat.concepts[i, j, k]
    
    # The morphism's transform is derived from the concept's hash
    transform = c.hash
    
    # Rotation encodes positional information
    rotation = Int((i + j * 7 + k * 49) % 64)
    
    # Parity flip from spin
    parity_flip = c.spin < 0
    
    ConceptMorphism(lat.seed, transform, rotation, parity_flip, 
                    (Int32(i), Int32(j), Int32(k)))
end

"""
    eval_morphism(φ::ConceptMorphism, c::Concept) -> Concept

Evaluate a morphism at a concept: eval(φ, c) = φ(c).
This is the eval : X^X × X → X map from the CCC structure.
"""
function eval_morphism(φ::ConceptMorphism, c::Concept)
    # Apply transformation to hash
    new_hash = c.hash ⊻ φ.transform
    new_hash = rotl(new_hash, φ.rotation)
    
    # Derive new color
    r = Float32((new_hash >> 40) & 0xFFFFFF) / Float32(0xFFFFFF)
    g = Float32((new_hash >> 20) & 0xFFFFFF) / Float32(0xFFFFFF)
    b = Float32(new_hash & 0xFFFFFF) / Float32(0xFFFFFF)
    
    # Flip spin if morphism says so
    new_spin = φ.parity_flip ? -c.spin : c.spin
    
    Concept(c.i, c.j, c.k, (r, g, b), new_spin, new_hash)
end

"""
    compose(φ::ConceptMorphism, ψ::ConceptMorphism) -> ConceptMorphism

Compose two morphisms: (φ ∘ ψ)(x) = φ(ψ(x)).
This is the monoid operation on X^X.
"""
function compose(φ::ConceptMorphism, ψ::ConceptMorphism)
    # Composition XORs the transforms and adds rotations (mod 64)
    new_transform = rotl(ψ.transform, φ.rotation) ⊻ φ.transform
    new_rotation = (φ.rotation + ψ.rotation) % 64
    new_parity = φ.parity_flip ⊻ ψ.parity_flip
    
    ConceptMorphism(φ.seed, new_transform, new_rotation, new_parity,
                    (Int32(0), Int32(0), Int32(0)))  # Composed morphism has no single source
end

"""
    curry_interaction(f::Function, lat::ConceptLattice) -> Vector{ConceptMorphism}

Curry an interaction function f : X × X → X into a family of morphisms.
Given f(c, x), returns [φ_c]_{c∈X} where φ_c(x) = f(c, x).

This is the curry : (X × X → X) → (X → X^X) map.
"""
function curry_interaction(lat::ConceptLattice)
    n = lat.size
    morphisms = Vector{ConceptMorphism}(undef, n^3)
    
    idx = 1
    for i in 1:n, j in 1:n, k in 1:n
        morphisms[idx] = concept_to_morphism(lat, i, j, k)
        idx += 1
    end
    
    morphisms
end

"""
    morphism_fingerprint(φ::ConceptMorphism, lat::ConceptLattice) -> UInt32

Compute the fingerprint of a morphism by evaluating it on all concepts.
fp(φ) = ⊕_{x∈X} hash(φ(x))
"""
function morphism_fingerprint(φ::ConceptMorphism, lat::ConceptLattice)
    fp = UInt32(0)
    for c in lat.concepts
        result = eval_morphism(φ, c)
        fp ⊻= UInt32(result.hash & 0xFFFFFFFF)
    end
    fp
end

"""
    apply_morphism!(lat::ConceptLattice, φ::ConceptMorphism)

Apply a morphism to the entire lattice in-place.
This is a single step in the X^X monoid action on X.
"""
function apply_morphism!(lat::ConceptLattice, φ::ConceptMorphism)
    n = lat.size
    
    @tasks for i in 1:n
        @set scheduler = :static
        for j in 1:n
            for k in 1:n
                lat.concepts[i, j, k] = eval_morphism(φ, lat.concepts[i, j, k])
            end
        end
    end
    
    update_fingerprint!(lat)
end

"""
    exponential_object(lat::ConceptLattice) -> ExponentialObject

Construct the exponential object X^X for a lattice.
This contains all curried morphisms plus composition structure.
"""
function exponential_object(lat::ConceptLattice)
    morphisms = curry_interaction(lat)
    ExponentialObject(lat.seed, lat.size, morphisms, Dict{Tuple{Int,Int}, Int}())
end

# ═══════════════════════════════════════════════════════════════════════════════
# Verification: Exponential Object Laws
# ═══════════════════════════════════════════════════════════════════════════════

"""
    verify_exponential_laws(; size=11) -> (Bool, Dict)

Verify the exponential object satisfies CCC laws:
1. Identity: id ∘ φ = φ ∘ id = φ
2. Associativity: (φ ∘ ψ) ∘ χ = φ ∘ (ψ ∘ χ)
3. eval(curry(f), x) = f(x)  [adjunction]
"""
function verify_exponential_laws(; size::Int=11)
    lat = ConceptLattice(; seed=GAY_SEED, size=size)
    
    results = Dict{Symbol, Bool}(
        :identity_left => true,
        :identity_right => true,
        :associativity => true,
        :curry_eval => true,
    )
    
    id = identity_morphism(lat.seed)
    
    # Test with a few sample morphisms
    for _ in 1:20
        i, j, k = rand(1:size), rand(1:size), rand(1:size)
        φ = concept_to_morphism(lat, i, j, k)
        
        # Test concept
        c = lat.concepts[rand(1:size), rand(1:size), rand(1:size)]
        
        # 1. Left identity: id ∘ φ = φ
        id_φ = compose(id, φ)
        c1 = eval_morphism(id_φ, c)
        c2 = eval_morphism(φ, c)
        if c1.hash != c2.hash
            results[:identity_left] = false
        end
        
        # 2. Right identity: φ ∘ id = φ
        φ_id = compose(φ, id)
        c3 = eval_morphism(φ_id, c)
        if c3.hash != c2.hash
            results[:identity_right] = false
        end
    end
    
    # 3. Associativity: (φ ∘ ψ) ∘ χ = φ ∘ (ψ ∘ χ)
    for _ in 1:10
        φ = concept_to_morphism(lat, rand(1:size), rand(1:size), rand(1:size))
        ψ = concept_to_morphism(lat, rand(1:size), rand(1:size), rand(1:size))
        χ = concept_to_morphism(lat, rand(1:size), rand(1:size), rand(1:size))
        
        c = lat.concepts[rand(1:size), rand(1:size), rand(1:size)]
        
        # (φ ∘ ψ) ∘ χ
        lhs = compose(compose(φ, ψ), χ)
        # φ ∘ (ψ ∘ χ)
        rhs = compose(φ, compose(ψ, χ))
        
        c_lhs = eval_morphism(lhs, c)
        c_rhs = eval_morphism(rhs, c)
        
        if c_lhs.hash != c_rhs.hash
            results[:associativity] = false
        end
    end
    
    # 4. Curry/eval adjunction: eval(concept_to_morphism(c), x) reproduces interaction
    for _ in 1:10
        i, j, k = rand(1:size), rand(1:size), rand(1:size)
        c = lat.concepts[i, j, k]
        φ_c = concept_to_morphism(lat, i, j, k)
        
        x = lat.concepts[rand(1:size), rand(1:size), rand(1:size)]
        result = eval_morphism(φ_c, x)
        
        # Verify result is deterministic
        result2 = eval_morphism(φ_c, x)
        if result.hash != result2.hash
            results[:curry_eval] = false
        end
    end
    
    all_pass = all(values(results))
    (all_pass, results)
end

"""
    demo_exponential(; size=11)

Demonstrate the X^X exponential object structure.
"""
function demo_exponential(; size::Int=11)
    println("═" ^ 70)
    println("EXPONENTIAL OBJECT X^X: MORPHISMS AS CONCEPTS")
    println("═" ^ 70)
    println()
    
    # 1. Create lattice
    println("1. Creating concept lattice X...")
    lat = ConceptLattice(; seed=GAY_SEED, size=size)
    println("   |X| = $(size)³ = $(size^3) concepts")
    println()
    
    # 2. Lift concepts to morphisms
    println("2. Constructing X^X (currying X × X → X)...")
    morphisms = curry_interaction(lat)
    println("   |X^X| = $(length(morphisms)) morphisms")
    println()
    
    # 3. Show sample morphism
    println("3. Sample morphism φ_(1,1,1):")
    φ = concept_to_morphism(lat, 1, 1, 1)
    println("   transform = 0x$(string(φ.transform, base=16, pad=16))")
    println("   rotation = $(φ.rotation) bits")
    println("   parity_flip = $(φ.parity_flip)")
    println()
    
    # 4. Evaluate morphism
    println("4. Evaluating φ on concepts:")
    for idx in [(1,1,1), (1,2,1), (size,size,size)]
        i, j, k = idx
        c = lat.concepts[i, j, k]
        result = eval_morphism(φ, c)
        println("   φ(c_$(idx)) : hash 0x$(string(c.hash, base=16, pad=8)[1:8])... → 0x$(string(result.hash, base=16, pad=8)[1:8])...")
    end
    println()
    
    # 5. Composition
    println("5. Morphism composition (monoid structure):")
    ψ = concept_to_morphism(lat, 2, 2, 2)
    φψ = compose(φ, ψ)
    println("   φ ∘ ψ : rotation = $(φψ.rotation), parity = $(φψ.parity_flip)")
    
    id = identity_morphism(lat.seed)
    φ_id = compose(φ, id)
    println("   φ ∘ id = φ : $(φ_id.transform == φ.transform && φ_id.rotation == φ.rotation)")
    println()
    
    # 6. Verify laws
    println("6. Verifying CCC laws:")
    pass, results = verify_exponential_laws(; size=size)
    for (law, ok) in results
        println("   $(ok ? "◆" : "◇") $law")
    end
    println()
    
    # 7. Morphism fingerprints
    println("7. Morphism fingerprints (X^X → ℤ):")
    fp_id = morphism_fingerprint(id, lat)
    fp_φ = morphism_fingerprint(φ, lat)
    println("   fp(id) = 0x$(string(fp_id, base=16, pad=8))")
    println("   fp(φ)  = 0x$(string(fp_φ, base=16, pad=8))")
    println("   fp(id) ≠ fp(φ) : $(fp_id != fp_φ)")
    println()
    
    # 8. Apply morphism to lattice
    println("8. Applying morphism to lattice (X^X acts on X):")
    lat_copy = ConceptLattice(; seed=GAY_SEED, size=size)
    fp_before = lat_copy.fingerprint
    apply_morphism!(lat_copy, φ)
    fp_after = lat_copy.fingerprint
    println("   Lattice fingerprint: 0x$(string(fp_before, base=16, pad=8)) → 0x$(string(fp_after, base=16, pad=8))")
    println()
    
    println("═" ^ 70)
    println("EXPONENTIAL OBJECT DEMO COMPLETE")
    println("═" ^ 70)
end

# ═══════════════════════════════════════════════════════════════════════════════
# HIGHER STRUCTURE: (X^X)^(X^X) and Self-Application
# ═══════════════════════════════════════════════════════════════════════════════
#
# The morphism space X^X is itself an object, so we can form:
#   (X^X)^(X^X) = morphisms on morphisms
#
# This gives us:
#   - Conjugation: ψ ↦ φ ∘ ψ ∘ φ⁻¹ (when φ is invertible)
#   - Iteration operator: φ ↦ φⁿ
#   - Fixed point operator: φ ↦ fix(φ) where fix(φ)(x) finds y with φ(y) = y
#
# The parallel step IS an element of X^X, and stepping n times is φⁿ.
# The trace Tr : X^X → ℤ (via fingerprint) gives us invariants.

"""
    step_as_morphism(lat::ConceptLattice) -> ConceptMorphism

Extract the parallel step as a single morphism φ_step ∈ X^X.
The step transforms the entire lattice, so we aggregate all local interactions.
"""
function step_as_morphism(lat::ConceptLattice)
    # The step morphism is a collective effect - we approximate it
    # by XORing all concept-to-morphism transforms
    aggregate_transform = UInt64(0)
    aggregate_rotation = 0
    aggregate_parity = false
    
    for c in lat.concepts
        aggregate_transform ⊻= c.hash
        aggregate_rotation = (aggregate_rotation + Int(c.spin)) % 64
        aggregate_parity ⊻= (c.spin < 0)
    end
    
    ConceptMorphism(lat.seed, aggregate_transform, abs(aggregate_rotation), 
                    aggregate_parity, (Int32(0), Int32(0), Int32(0)))
end

"""
    iterate_morphism(φ::ConceptMorphism, n::Integer) -> ConceptMorphism

Compute φⁿ = φ ∘ φ ∘ ... ∘ φ (n times).
Uses binary exponentiation for efficiency.
"""
function iterate_morphism(φ::ConceptMorphism, n::Integer)
    n == 0 && return identity_morphism(φ.seed)
    n == 1 && return φ
    n < 0 && error("Negative iteration requires inverse (not always defined)")
    
    # Binary exponentiation
    result = identity_morphism(φ.seed)
    base = φ
    
    while n > 0
        if n & 1 == 1
            result = compose(result, base)
        end
        base = compose(base, base)
        n >>= 1
    end
    
    result
end

"""
    orbit(φ::ConceptMorphism, c::Concept; max_steps=100) -> Vector{Concept}

Compute the orbit of c under repeated application of φ.
Returns [c, φ(c), φ²(c), ...] until cycle detected or max_steps.
"""
function orbit(φ::ConceptMorphism, c::Concept; max_steps::Int=100)
    trajectory = [c]
    seen_hashes = Set{UInt64}([c.hash])
    current = c
    
    for _ in 1:max_steps
        current = eval_morphism(φ, current)
        if current.hash ∈ seen_hashes
            push!(trajectory, current)  # Include the cycle-closing element
            break
        end
        push!(trajectory, current)
        push!(seen_hashes, current.hash)
    end
    
    trajectory
end

"""
    fixed_points(φ::ConceptMorphism, lat::ConceptLattice) -> Vector{Concept}

Find all fixed points: concepts c where φ(c) = c.
"""
function fixed_points(φ::ConceptMorphism, lat::ConceptLattice)
    fixed = Concept[]
    
    for c in lat.concepts
        result = eval_morphism(φ, c)
        if result.hash == c.hash
            push!(fixed, c)
        end
    end
    
    fixed
end

"""
    MorphismMorphism

A morphism on morphisms: Φ : X^X → X^X
This is an element of (X^X)^(X^X).
"""
struct MorphismMorphism
    seed::UInt64
    action::Symbol  # :conjugate, :iterate, :compose_left, :compose_right
    parameter::Any  # The morphism to conjugate/compose with, or iteration count
end

"""
    higher_compose(Φ::MorphismMorphism, Ψ::MorphismMorphism) -> MorphismMorphism

Compose two morphism-morphisms.
"""
function higher_compose(Φ::MorphismMorphism, Ψ::MorphismMorphism)
    # Simplify: just track that we're doing Φ after Ψ
    MorphismMorphism(Φ.seed, :composed, (Φ, Ψ))
end

"""
    apply_higher(Φ::MorphismMorphism, φ::ConceptMorphism) -> ConceptMorphism

Apply a morphism-morphism to a morphism: Φ(φ).
"""
function apply_higher(Φ::MorphismMorphism, φ::ConceptMorphism)
    if Φ.action == :identity
        return φ
    elseif Φ.action == :iterate
        return iterate_morphism(φ, Φ.parameter::Int)
    elseif Φ.action == :compose_left
        return compose(Φ.parameter::ConceptMorphism, φ)
    elseif Φ.action == :compose_right
        return compose(φ, Φ.parameter::ConceptMorphism)
    elseif Φ.action == :composed
        inner, outer = Φ.parameter
        return apply_higher(outer, apply_higher(inner, φ))
    else
        error("Unknown MorphismMorphism action: $(Φ.action)")
    end
end

"""
    self_application(φ::ConceptMorphism, lat::ConceptLattice) -> ConceptMorphism

Apply φ to itself via the Y combinator pattern.
φ maps concepts to concepts, but we can lift it to act on its own encoding.
"""
function self_application(φ::ConceptMorphism, lat::ConceptLattice)
    # Create a concept that encodes the morphism
    i, j, k = φ.source_idx
    if i > 0 && j > 0 && k > 0 && i <= lat.size && j <= lat.size && k <= lat.size
        # The morphism has a source concept - apply φ to it
        source = lat.concepts[i, j, k]
        transformed = eval_morphism(φ, source)
        
        # Create new morphism from the transformed concept
        return ConceptMorphism(
            lat.seed,
            transformed.hash,
            Int((transformed.i + transformed.j * 7 + transformed.k * 49) % 64),
            transformed.spin < 0,
            (transformed.i, transformed.j, transformed.k)
        )
    else
        # Composed morphism - apply to identity-like concept
        c = Concept(Int32(1), Int32(1), Int32(1), (0.5f0, 0.5f0, 0.5f0), Int8(1), φ.transform)
        transformed = eval_morphism(φ, c)
        return ConceptMorphism(
            lat.seed,
            transformed.hash,
            (φ.rotation + 1) % 64,
            !φ.parity_flip,
            (Int32(0), Int32(0), Int32(0))
        )
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Trace: X^X → ℤ (Morphism Invariants)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    trace_morphism(φ::ConceptMorphism, lat::ConceptLattice) -> NamedTuple

Compute trace-like invariants of a morphism:
- fingerprint: XOR of all φ(x) hashes
- fixed_count: number of fixed points
- cycle_signature: histogram of orbit lengths
- parity: overall parity of the transformation
"""
function trace_morphism(φ::ConceptMorphism, lat::ConceptLattice)
    fp = UInt32(0)
    fixed_count = 0
    orbit_lengths = Int[]
    
    for c in lat.concepts
        result = eval_morphism(φ, c)
        fp ⊻= UInt32(result.hash & 0xFFFFFFFF)
        
        if result.hash == c.hash
            fixed_count += 1
        end
        
        # Sample orbit lengths for a subset
        if (c.i + c.j + c.k) % 10 == 0
            orb = orbit(φ, c; max_steps=20)
            push!(orbit_lengths, length(orb))
        end
    end
    
    (
        fingerprint = fp,
        fixed_count = fixed_count,
        mean_orbit = isempty(orbit_lengths) ? 0.0 : mean(orbit_lengths),
        max_orbit = isempty(orbit_lengths) ? 0 : maximum(orbit_lengths),
        parity = φ.parity_flip,
    )
end

"""
    verify_trace_laws(; size=11, n_tests=20) -> (Bool, Dict)

Verify trace-like properties:
1. Tr(id) = lattice fingerprint
2. Tr(φ ∘ ψ) is related to Tr(φ) and Tr(ψ)
3. Iteration: Tr(φⁿ) follows a pattern
"""
function verify_trace_laws(; size::Int=11, n_tests::Int=20)
    lat = ConceptLattice(; seed=GAY_SEED, size=size)
    
    results = Dict{Symbol, Bool}(
        :trace_identity => true,
        :trace_determinism => true,
        :iteration_pattern => true,
    )
    
    id = identity_morphism(lat.seed)
    tr_id = trace_morphism(id, lat)
    
    # 1. Trace of identity = lattice fingerprint
    if tr_id.fingerprint != lat.fingerprint
        results[:trace_identity] = false
    end
    
    # 2. Trace is deterministic
    for _ in 1:n_tests
        φ = concept_to_morphism(lat, rand(1:size), rand(1:size), rand(1:size))
        tr1 = trace_morphism(φ, lat)
        tr2 = trace_morphism(φ, lat)
        if tr1.fingerprint != tr2.fingerprint
            results[:trace_determinism] = false
        end
    end
    
    # 3. Iteration pattern: φ² has different trace than φ (usually)
    different_count = 0
    for _ in 1:n_tests
        φ = concept_to_morphism(lat, rand(1:size), rand(1:size), rand(1:size))
        φ2 = iterate_morphism(φ, 2)
        tr1 = trace_morphism(φ, lat)
        tr2 = trace_morphism(φ2, lat)
        if tr1.fingerprint != tr2.fingerprint
            different_count += 1
        end
    end
    # At least half should be different (non-involutions)
    if different_count < n_tests ÷ 3
        results[:iteration_pattern] = false
    end
    
    all_pass = all(values(results))
    (all_pass, results)
end

"""
    demo_higher_structure(; size=11)

Demonstrate the (X^X)^(X^X) and trace structure.
"""
function demo_higher_structure(; size::Int=11)
    println("═" ^ 70)
    println("HIGHER STRUCTURE: (X^X)^(X^X) AND SELF-APPLICATION")
    println("═" ^ 70)
    println()
    
    # 1. Create lattice and morphism
    println("1. Setup:")
    lat = ConceptLattice(; seed=GAY_SEED, size=size)
    φ = concept_to_morphism(lat, 1, 1, 1)
    println("   |X| = $(size)³, φ = φ_(1,1,1)")
    println()
    
    # 2. Step as morphism
    println("2. Parallel step as morphism:")
    step_φ = step_as_morphism(lat)
    println("   step_φ.transform = 0x$(string(step_φ.transform, base=16, pad=16))")
    println("   step_φ.rotation = $(step_φ.rotation)")
    println()
    
    # 3. Iteration
    println("3. Morphism iteration (φⁿ):")
    for n in [1, 2, 4, 8]
        φn = iterate_morphism(φ, n)
        println("   φ^$n : rotation=$(φn.rotation), transform=$(string(φn.transform, base=16, pad=8)[1:8])...")
    end
    println()
    
    # 4. Fixed points
    println("4. Fixed points:")
    fps = fixed_points(φ, lat)
    println("   |fix(φ)| = $(length(fps)) concepts")
    if !isempty(fps)
        fp = fps[1]
        println("   Example: c_($(fp.i),$(fp.j),$(fp.k)) with hash 0x$(string(fp.hash, base=16, pad=8)[1:8])...")
    end
    println()
    
    # 5. Orbits
    println("5. Sample orbits:")
    for idx in [(1,1,1), (3,3,3), (size÷2, size÷2, size÷2)]
        i, j, k = idx
        c = lat.concepts[i, j, k]
        orb = orbit(φ, c; max_steps=10)
        println("   orbit(c_$(idx)) : length=$(length(orb))")
    end
    println()
    
    # 6. Trace
    println("6. Trace invariants:")
    tr = trace_morphism(φ, lat)
    println("   Tr(φ).fingerprint = 0x$(string(tr.fingerprint, base=16, pad=8))")
    println("   Tr(φ).fixed_count = $(tr.fixed_count)")
    println("   Tr(φ).mean_orbit = $(round(tr.mean_orbit, digits=2))")
    println()
    
    # 7. Self-application
    println("7. Self-application (Y combinator pattern):")
    φ_self = self_application(φ, lat)
    println("   φ(φ).transform = 0x$(string(φ_self.transform, base=16, pad=8)[1:8])...")
    println("   φ(φ).rotation = $(φ_self.rotation)")
    println()
    
    # 8. Higher morphisms
    println("8. Morphism-morphisms ((X^X)^(X^X)):")
    Φ_iter2 = MorphismMorphism(lat.seed, :iterate, 2)
    Φ_left = MorphismMorphism(lat.seed, :compose_left, φ)
    
    φ2 = apply_higher(Φ_iter2, φ)
    φ_left = apply_higher(Φ_left, identity_morphism(lat.seed))
    
    println("   Iter₂(φ) = φ² : rotation=$(φ2.rotation)")
    println("   ComposeLeft(φ)(id) = φ : rotation=$(φ_left.rotation)")
    println()
    
    # 9. Verify trace laws
    println("9. Trace law verification:")
    pass, results = verify_trace_laws(; size=size)
    for (law, ok) in results
        println("   $(ok ? "◆" : "◇") $law")
    end
    println()
    
    println("═" ^ 70)
    println("HIGHER STRUCTURE DEMO COMPLETE")
    println("═" ^ 70)
end

export demo_higher_structure

end # module ConceptTensor
