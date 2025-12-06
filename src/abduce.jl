# Abduce: Abductive inference for SPI color systems
#
# Abduction (ἀπαγωγή): Reasoning from effect to cause.
# Given observations, infer the best explanation.
#
# In Gay.jl:
# - Given a color, find what index/seed produced it
# - Given a deranged sequence, recover the original ordering
# - Given a permutation, find the inverse (undo operation)
# - Given partial observations, infer the full structure
#
# From Peirce: "Abduction is the process of forming an explanatory hypothesis."
# The color is the sign; the seed is the interpretant; abduction recovers the object.

export Abducible, abduce, abduce_index, abduce_seed, abduce_inverse
export abduce_derangement, abduce_cycle, abduce_parity
export GayAbducer, register_observation!, infer_seed, infer_structure
export color_distance, find_nearest_color, color_fingerprint

# ═══════════════════════════════════════════════════════════════════════════
# Color Distance Metrics
# ═══════════════════════════════════════════════════════════════════════════

"""
    color_distance(c1, c2) -> Float64

Compute perceptual distance between two colors in CIELAB space.
Uses ΔE*ab (CIE76) for simplicity; approximately 1.0 = just noticeable difference.
"""
function color_distance(c1, c2)
    lab1 = convert(Lab, c1)
    lab2 = convert(Lab, c2)
    
    dL = lab1.l - lab2.l
    da = lab1.a - lab2.a
    db = lab1.b - lab2.b
    
    return sqrt(dL^2 + da^2 + db^2)
end

"""
    color_fingerprint(c) -> UInt64

Generate a hash fingerprint from a color for fast lookup.
Quantizes to 8-bit per channel.
"""
function color_fingerprint(c)
    rgb = convert(RGB, c)
    r = UInt64(round(clamp(rgb.r, 0, 1) * 255))
    g = UInt64(round(clamp(rgb.g, 0, 1) * 255))
    b = UInt64(round(clamp(rgb.b, 0, 1) * 255))
    return (r << 16) | (g << 8) | b
end

# ═══════════════════════════════════════════════════════════════════════════
# Abducible: A value that can be traced back to its origin
# ═══════════════════════════════════════════════════════════════════════════

"""
    Abducible{T}

A value paired with its causal history for abductive inference.

# Fields
- `value::T` - The observed value
- `origin_seed::Union{UInt64, Nothing}` - Known or inferred seed
- `origin_index::Union{Int, Nothing}` - Known or inferred index
- `confidence::Float64` - Confidence in the abduction (0.0 to 1.0)
- `provenance::Symbol` - How we know: :observed, :inferred, :hypothesized

# Example
```julia
c = color_at(42, seed=0xDEADBEEF)
ab = Abducible(c)
abduce!(ab, seed=0xDEADBEEF, max_index=1000)
ab.origin_index  # => 42
```
"""
mutable struct Abducible{T}
    value::T
    origin_seed::Union{UInt64, Nothing}
    origin_index::Union{Int, Nothing}
    confidence::Float64
    provenance::Symbol
end

"""
    Abducible(value; seed=nothing, index=nothing)

Wrap a value for abductive inference.
"""
function Abducible(value; seed::Union{Integer,Nothing}=nothing, 
                   index::Union{Integer,Nothing}=nothing)
    s = seed === nothing ? nothing : UInt64(seed)
    i = index === nothing ? nothing : Int(index)
    conf = (s !== nothing && i !== nothing) ? 1.0 : 0.0
    prov = conf > 0 ? :observed : :unknown
    Abducible(value, s, i, conf, prov)
end

# ═══════════════════════════════════════════════════════════════════════════
# Core Abduction: Find index given color and seed
# ═══════════════════════════════════════════════════════════════════════════

"""
    abduce_index(color, seed::Integer; max_index::Integer=10000, tolerance::Float64=0.01)

Given a color and seed, find the index that produces it (or the closest match).

Returns (index, distance, exact_match).

# Algorithm
Uses fingerprint-based fast path, then falls back to linear search with early exit.
"""
function abduce_index(color, seed::Integer; max_index::Integer=10000, tolerance::Float64=0.01)
    target_fp = color_fingerprint(color)
    
    best_idx = -1
    best_dist = Inf
    
    for idx in 1:max_index
        c = color_at(idx, SRGB(); seed=UInt64(seed))
        
        # Fast path: exact fingerprint match
        if color_fingerprint(c) == target_fp
            dist = color_distance(color, c)
            if dist < tolerance
                return (idx, dist, true)
            end
        end
        
        # Track best match
        dist = color_distance(color, c)
        if dist < best_dist
            best_dist = dist
            best_idx = idx
        end
        
        # Early exit on exact match
        if best_dist < tolerance
            return (best_idx, best_dist, true)
        end
    end
    
    return (best_idx, best_dist, best_dist < tolerance)
end

"""
    abduce_seed(color, index::Integer; seed_candidates::Vector{UInt64}=UInt64[], 
                max_random_seeds::Integer=1000, tolerance::Float64=0.01)

Given a color and index, find the seed that produces it.

If seed_candidates is provided, searches those first.
Otherwise tries common seeds then random samples.
"""
function abduce_seed(color, index::Integer; 
                     seed_candidates::Vector{UInt64}=UInt64[],
                     max_random_seeds::Integer=1000,
                     tolerance::Float64=0.01)
    target_fp = color_fingerprint(color)
    
    # Common seeds to try first
    common_seeds = UInt64[
        0xDEADBEEF, 0xCAFEBABE, 0x12345678, 0x42424242,
        0x6761795f636f6c6f,  # GAY_SEED
        0x0, 0x1, 0x42, 0x1337,
    ]
    
    all_candidates = vcat(seed_candidates, common_seeds)
    
    best_seed = UInt64(0)
    best_dist = Inf
    
    # Try known candidates
    for seed in all_candidates
        c = color_at(index, SRGB(); seed=seed)
        if color_fingerprint(c) == target_fp
            dist = color_distance(color, c)
            if dist < tolerance
                return (seed, dist, true)
            end
        end
        dist = color_distance(color, c)
        if dist < best_dist
            best_dist = dist
            best_seed = seed
        end
    end
    
    # Random search if no match found
    for _ in 1:max_random_seeds
        seed = rand(UInt64)
        c = color_at(index, SRGB(); seed=seed)
        dist = color_distance(color, c)
        if dist < best_dist
            best_dist = dist
            best_seed = seed
        end
        if best_dist < tolerance
            return (best_seed, best_dist, true)
        end
    end
    
    return (best_seed, best_dist, best_dist < tolerance)
end

"""
    abduce(ab::Abducible; seed=nothing, max_index=10000, tolerance=0.01)

Perform abductive inference on an Abducible, updating its origin fields.
"""
function abduce(ab::Abducible; seed::Union{Integer,Nothing}=nothing,
                max_index::Integer=10000, tolerance::Float64=0.01)
    if seed !== nothing
        idx, dist, exact = abduce_index(ab.value, seed; max_index=max_index, tolerance=tolerance)
        ab.origin_seed = UInt64(seed)
        ab.origin_index = idx
        ab.confidence = exact ? 1.0 : max(0.0, 1.0 - dist / 100.0)
        ab.provenance = exact ? :inferred : :hypothesized
    elseif ab.origin_index !== nothing
        seed, dist, exact = abduce_seed(ab.value, ab.origin_index; tolerance=tolerance)
        ab.origin_seed = seed
        ab.confidence = exact ? 1.0 : max(0.0, 1.0 - dist / 100.0)
        ab.provenance = exact ? :inferred : :hypothesized
    end
    return ab
end

# ═══════════════════════════════════════════════════════════════════════════
# Inverse Operations: Undo derangements and permutations
# ═══════════════════════════════════════════════════════════════════════════

"""
    abduce_inverse(perm::Vector{Int}) -> Vector{Int}

Compute the inverse permutation: if σ(i) = j, then σ⁻¹(j) = i.
Applying the inverse undoes the original permutation.
"""
function abduce_inverse(perm::Vector{Int})
    n = length(perm)
    inv = Vector{Int}(undef, n)
    for i in 1:n
        inv[perm[i]] = i
    end
    return inv
end

"""
    abduce_derangement(deranged::Vector{T}, original::Vector{T}) -> Vector{Int}

Given a deranged sequence and original, recover the permutation that was applied.
Returns the permutation σ such that deranged[i] = original[σ[i]].
"""
function abduce_derangement(deranged::Vector{T}, original::Vector{T}) where T
    n = length(original)
    @assert length(deranged) == n "Sequences must have same length"
    
    # Build index map for original
    orig_idx = Dict{T, Int}()
    for (i, v) in enumerate(original)
        orig_idx[v] = i
    end
    
    # Recover permutation
    perm = Vector{Int}(undef, n)
    for (i, v) in enumerate(deranged)
        perm[i] = orig_idx[v]
    end
    
    return perm
end

"""
    abduce_cycle(perm::Vector{Int}) -> Vector{Vector{Int}}

Decompose a permutation into disjoint cycles.
For derangements, all cycles have length ≥ 2.
"""
function abduce_cycle(perm::Vector{Int})
    n = length(perm)
    visited = falses(n)
    cycles = Vector{Vector{Int}}()
    
    for start in 1:n
        if visited[start]
            continue
        end
        
        cycle = Int[]
        i = start
        while !visited[i]
            visited[i] = true
            push!(cycle, i)
            i = perm[i]
        end
        
        if length(cycle) > 1  # Skip fixed points (1-cycles)
            push!(cycles, cycle)
        elseif length(cycle) == 1 && perm[start] != start
            push!(cycles, cycle)
        end
    end
    
    return cycles
end

"""
    abduce_parity(perm::Vector{Int}) -> Tuple{Int, Int}

Compute the parity decomposition of a permutation.
Returns (even_count, odd_count) where elements are classified by i ⊕ σ(i) & 1.
"""
function abduce_parity(perm::Vector{Int})
    even_count = 0
    odd_count = 0
    
    for (i, j) in enumerate(perm)
        if (i ⊻ j) & 1 == 0
            even_count += 1
        else
            odd_count += 1
        end
    end
    
    return (even_count, odd_count)
end

# ═══════════════════════════════════════════════════════════════════════════
# GayAbducer: Accumulate observations and infer structure
# ═══════════════════════════════════════════════════════════════════════════

"""
    GayAbducer

An abductive inference engine that accumulates observations and infers the 
underlying seed/structure that produced them.

# Theory

Abduction (ἀπαγωγή) is Peirce's "inference to the best explanation":
- **Deduction**: rule + case → result (forward)
- **Induction**: case + result → rule (generalization)  
- **Abduction**: rule + result → case (reverse inference)

In Gay.jl's SPI system:
- **Rule**: `color_at(index, seed)` produces deterministic colors
- **Result**: An observed color
- **Case**: The (seed, index) pair that produced it

GayAbducer accumulates observations and uses them to infer the hidden seed.

# Fields

- `observations::Vector{Tuple{Any, Union{Int,Nothing}, Union{UInt64,Nothing}}}` - 
  Accumulated (value, index, seed) tuples
- `inferred_seed::Union{UInt64, Nothing}` - Best-guess seed after inference
- `confidence::Float64` - Confidence in inferred seed (0.0 to 1.0)
- `n_consistent::Int` - Number of observations consistent with inferred seed

# Use Cases

1. **Reverse-engineer a color scheme**: Given colors from an unknown source,
   recover the seed that generated them.

2. **Verify SPI consistency**: Confirm that colors from different runs/machines
   are consistent with the same seed.

3. **Identify patterns**: Analyze structural properties of color sequences
   (magnetization, parity distribution, clustering).

4. **Debug color generation**: When colors don't match expectations, use
   abduction to find what seed is actually being used.

# Workflow

```julia
# Step 1: Create abducer
abducer = GayAbducer()

# Step 2: Register observations (colors with known indices)
for i in 1:10
    c = color_at(i, seed=0xDEADBEEF)  # Unknown seed in practice
    register_observation!(abducer, c; index=i)
end

# Step 3: Infer the seed
seed = infer_seed(abducer)
# => 0xDEADBEEF (recovered!)

# Step 4: Check confidence
abducer.confidence  # => 1.0 (all observations match)
abducer.n_consistent  # => 10

# Step 5: Analyze structure
structure = infer_structure(abducer)
# => (pattern=:dispersed, magnetization=0.2, ...)
```

# Algorithm

`infer_seed` uses a two-phase search:

1. **Known seeds**: Try common seeds (0xDEADBEEF, 0xCAFEBABE, GAY_SEED, etc.)
   and any user-provided candidates.

2. **Random search**: If no match, sample random seeds and track best match.

For each candidate seed, we compute how many observations match within the
CIELAB just-noticeable-difference threshold (ΔE < 1.0).

# Example: Recovering an Unknown Seed

```julia
# Someone generated these colors - what seed did they use?
mystery_colors = [
    RGB(0.87, 0.31, 0.38),
    RGB(0.92, 0.35, 0.39),
    RGB(0.21, 0.06, 0.81),
]

abducer = GayAbducer()
for (i, c) in enumerate(mystery_colors)
    register_observation!(abducer, c; index=i)
end

seed = infer_seed(abducer)
if abducer.confidence > 0.9
    println("Seed is likely 0x\$(string(seed, base=16))")
else
    println("Could not confidently infer seed")
end
```

# Example: Verifying Cross-Platform Consistency

```julia
# Colors from machine A
colors_a = [color_at(i, seed=0xDEADBEEF) for i in 1:5]

# Colors from machine B (received over network)
colors_b = [...]  # Should match colors_a

abducer = GayAbducer()
for (i, c) in enumerate(colors_b)
    register_observation!(abducer, c; index=i)
end

seed = infer_seed(abducer; seed_candidates=[0xDEADBEEF])
if seed == 0xDEADBEEF && abducer.confidence == 1.0
    println("SPI verified: machines agree!")
else
    println("Warning: SPI violation detected")
end
```

# See Also

- [`register_observation!`](@ref): Add observations to the abducer
- [`infer_seed`](@ref): Infer seed from observations
- [`infer_structure`](@ref): Analyze structural patterns
- [`Abducible`](@ref): Single-value abductive wrapper
- [`abduce_index`](@ref): Find index given color and seed
- [`abduce_seed`](@ref): Find seed given color and index
"""
mutable struct GayAbducer
    observations::Vector{Tuple{Any, Union{Int,Nothing}, Union{UInt64,Nothing}}}
    inferred_seed::Union{UInt64, Nothing}
    confidence::Float64
    n_consistent::Int
end

"""
    GayAbducer()

Create a new abductive inference engine.
"""
GayAbducer() = GayAbducer([], nothing, 0.0, 0)

"""
    register_observation!(abducer::GayAbducer, value; index=nothing, seed=nothing)

Register an observed value (typically a color) with optional known metadata.

# Arguments
- `abducer::GayAbducer`: The abducer to add the observation to
- `value`: The observed value (color, permutation element, etc.)
- `index::Integer=nothing`: The known index in the sequence (if known)
- `seed::Integer=nothing`: The known seed (if known, for verification)

# Returns
The abducer (for chaining).

# Notes
- At least one observation must have a known `index` for `infer_seed` to work.
- Observations with known `seed` are used for verification, not inference.
- Multiple observations improve inference confidence.

# Example
```julia
abducer = GayAbducer()

# Register colors with known indices
register_observation!(abducer, color1; index=1)
register_observation!(abducer, color2; index=2)
register_observation!(abducer, color3; index=3)

# Chain multiple registrations
abducer |> 
    a -> register_observation!(a, c1; index=1) |>
    a -> register_observation!(a, c2; index=2)
```
"""
function register_observation!(abducer::GayAbducer, value; 
                               index::Union{Integer,Nothing}=nothing,
                               seed::Union{Integer,Nothing}=nothing)
    idx = index === nothing ? nothing : Int(index)
    s = seed === nothing ? nothing : UInt64(seed)
    push!(abducer.observations, (value, idx, s))
    return abducer
end

"""
    infer_seed(abducer::GayAbducer; seed_candidates=UInt64[], max_random=1000)

Attempt to infer the seed that produced the observed colors.

# Arguments
- `abducer::GayAbducer`: Abducer with registered observations
- `seed_candidates::Vector{UInt64}=[]`: Known seeds to try first
- `max_random::Integer=1000`: Number of random seeds to try if no match found

# Returns
- `UInt64`: The inferred seed (best match found)
- `nothing`: If no observations have known indices

# Side Effects
Updates `abducer.inferred_seed`, `abducer.confidence`, and `abducer.n_consistent`.

# Algorithm

1. Extract observations that have known indices
2. For each candidate seed (user-provided + common seeds):
   - Count how many observations match (ΔE < 1.0 in CIELAB)
   - If all match, return immediately with confidence=1.0
3. If no perfect match, try random seeds
4. Return best match with confidence = n_matches / n_observations

# Common Seeds Tried
- `0xDEADBEEF`, `0xCAFEBABE`, `0x12345678`, `0x42424242`
- `GAY_SEED` (0x6761795f636f6c6f)
- `0x0`, `0x1`, `0x42`, `0x1337`

# Example
```julia
abducer = GayAbducer()
for i in 1:5
    register_observation!(abducer, color_at(i; seed=0xDEADBEEF); index=i)
end

seed = infer_seed(abducer)
# => 0xDEADBEEF

abducer.confidence  # => 1.0 (all 5 matched)
```

# Performance
- O(n_candidates × n_observations) for known candidates
- O(max_random × n_observations) for random search
- Each comparison involves CIELAB color distance (~10 FLOPs)
"""
function infer_seed(abducer::GayAbducer; 
                    seed_candidates::Vector{UInt64}=UInt64[],
                    max_random::Integer=1000)
    # Find observations with known indices
    indexed_obs = [(v, i) for (v, i, s) in abducer.observations if i !== nothing]
    isempty(indexed_obs) && return nothing
    
    # Try to find seed that explains all observations
    common_seeds = UInt64[
        0xDEADBEEF, 0xCAFEBABE, 0x12345678, 0x42424242,
        0x6761795f636f6c6f, 0x0, 0x1, 0x42, 0x1337,
    ]
    all_candidates = vcat(seed_candidates, common_seeds)
    
    best_seed = UInt64(0)
    best_matches = 0
    
    for seed in all_candidates
        matches = 0
        for (color, idx) in indexed_obs
            c = color_at(idx, SRGB(); seed=seed)
            if color_distance(color, c) < 1.0  # JND threshold
                matches += 1
            end
        end
        if matches > best_matches
            best_matches = matches
            best_seed = seed
        end
        if matches == length(indexed_obs)
            # Perfect match
            abducer.inferred_seed = seed
            abducer.confidence = 1.0
            abducer.n_consistent = matches
            return seed
        end
    end
    
    # Random search
    for _ in 1:max_random
        seed = rand(UInt64)
        matches = 0
        for (color, idx) in indexed_obs
            c = color_at(idx, SRGB(); seed=seed)
            if color_distance(color, c) < 1.0
                matches += 1
            end
        end
        if matches > best_matches
            best_matches = matches
            best_seed = seed
        end
    end
    
    abducer.inferred_seed = best_seed
    abducer.confidence = best_matches / length(indexed_obs)
    abducer.n_consistent = best_matches
    
    return best_seed
end

"""
    infer_structure(abducer::GayAbducer)

Analyze observations to infer structural patterns:
- Parity distribution
- Cycle structure (if permutation-like)
- Magnetization tendency

Returns a NamedTuple with inferred properties.
"""
function infer_structure(abducer::GayAbducer)
    colors = [v for (v, _, _) in abducer.observations if v isa RGB || v isa AbstractRGB]
    
    if isempty(colors)
        return (pattern=:unknown, details=nothing)
    end
    
    # Analyze hue distribution
    hues = [convert(HSL, c).h for c in colors]
    mean_hue = sum(hues) / length(hues)
    hue_variance = sum((h - mean_hue)^2 for h in hues) / length(hues)
    
    # Analyze luminance distribution
    lums = [convert(HSL, c).l for c in colors]
    mean_lum = sum(lums) / length(lums)
    
    # Magnetization: high hue = spin up, low hue = spin down
    spins = [h < 180 ? 1 : -1 for h in hues]
    magnetization = sum(spins) / length(spins)
    
    # Infer pattern type
    pattern = if hue_variance < 100
        :clustered  # Colors cluster around a hue
    elseif abs(magnetization) > 0.5
        :polarized  # Strong spin bias
    else
        :dispersed  # Evenly spread
    end
    
    return (
        pattern = pattern,
        n_colors = length(colors),
        mean_hue = mean_hue,
        hue_variance = hue_variance,
        mean_luminance = mean_lum,
        magnetization = magnetization,
        inferred_seed = abducer.inferred_seed,
        confidence = abducer.confidence
    )
end

"""
    find_nearest_color(target, palette::Vector; top_k::Int=1)

Find the nearest color(s) in a palette to the target.
Returns vector of (index, color, distance) tuples.
"""
function find_nearest_color(target, palette::Vector; top_k::Int=1)
    distances = [(i, c, color_distance(target, c)) for (i, c) in enumerate(palette)]
    sort!(distances, by=x->x[3])
    return distances[1:min(top_k, length(distances))]
end
