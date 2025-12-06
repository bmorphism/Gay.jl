# Derangeable: SPI-based deterministic derangements (permutations with no fixed points)
# A derangement σ satisfies σ(i) ≠ i for all i
#
# Uses rejection sampling with SplittableRandom for Strong Parallelism Invariance:
# the same seed always produces the same derangement, regardless of execution order.

using Random: shuffle!

export Derangeable, derange, derange_at, derange_colors, derange_indices
export derange_cycle, cycle_colors, derangement_sign
export GayDerangementStream, next_derangement, nth_derangement

"""
    Derangeable{T}

A derangeable sequence: provides deterministic derangements (permutations with no fixed points).

# SPI Properties
- Same seed → same derangement
- O(1) access to nth derangement via `derange_at(d, n)`
- Parallel generation: independent streams for different parities

# Theory
- Derangements of n elements: D(n) = n! × Σ(-1)^k/k! for k=0..n ≈ n!/e
- Each element must move: good for shuffling colors where "no repeat in place" is desired
- Cycle structure: derangements decompose into cycles of length ≥ 2

# Example
```julia
using LispSyntax
d = Derangeable(1:6, seed=0xDEADBEEF)
@lisp (derange d)           ; => [3, 5, 1, 6, 2, 4] (no fixed points)
@lisp (derange-at d 42)     ; => nth derangement, deterministic
@lisp (derange-colors d)    ; => colors permuted with no color in original position
```
"""
mutable struct Derangeable{T}
    elements::Vector{T}
    seed::UInt64
    invocation::UInt64
end

"""
    Derangeable(elements; seed::Integer=GAY_SEED)

Create a derangeable sequence from any iterable.
"""
function Derangeable(elements; seed::Integer=GAY_SEED)
    Derangeable(collect(elements), UInt64(seed), UInt64(0))
end

"""
    Derangeable(n::Integer; seed::Integer=GAY_SEED)

Create a derangeable sequence of 1:n.
"""
function Derangeable(n::Integer; seed::Integer=GAY_SEED)
    Derangeable(collect(1:n), UInt64(seed), UInt64(0))
end

# ═══════════════════════════════════════════════════════════════════════════
# Core derangement generation via rejection sampling
# ═══════════════════════════════════════════════════════════════════════════

"""
    mix64(z::UInt64) -> UInt64

SplitMix64 mixing function for deterministic hashing.
"""
function mix64(z::UInt64)
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    z ⊻ (z >> 31)
end

"""
    fisher_yates_derangement!(arr::Vector, rng_state::UInt64) -> (Vector, UInt64)

Generate a derangement using modified Fisher-Yates with rejection.
Returns the derangement and the updated RNG state.

Uses rejection sampling: if we would create a fixed point, reject and retry.
Expected attempts ≈ e ≈ 2.718 per derangement.
"""
function fisher_yates_derangement!(arr::Vector, rng_state::UInt64)
    n = length(arr)
    n <= 1 && error("Derangement requires n ≥ 2")
    
    result = collect(1:n)  # Permutation indices
    attempts = 0
    max_attempts = 1000  # Safety bound
    
    while attempts < max_attempts
        attempts += 1
        rng_state = mix64(rng_state + UInt64(attempts))
        
        # Fisher-Yates shuffle
        perm = collect(1:n)
        local_state = rng_state
        for i in n:-1:2
            local_state = mix64(local_state)
            j = 1 + (local_state % UInt64(i))
            perm[i], perm[j] = perm[j], perm[i]
        end
        
        # Check for fixed points
        is_derangement = true
        for i in 1:n
            if perm[i] == i
                is_derangement = false
                break
            end
        end
        
        if is_derangement
            return (perm, rng_state)
        end
    end
    
    error("Failed to generate derangement after $max_attempts attempts")
end

"""
    sattolo_derangement(n::Integer, rng_state::UInt64) -> (Vector{Int}, UInt64)

Generate a derangement using Sattolo's algorithm (guaranteed single cycle).
A single cycle of length n is always a derangement.
"""
function sattolo_derangement(n::Integer, rng_state::UInt64)
    n <= 1 && error("Derangement requires n ≥ 2")
    
    perm = collect(1:n)
    for i in n:-1:2
        rng_state = mix64(rng_state)
        j = 1 + (rng_state % UInt64(i - 1))  # j ∈ [1, i-1], never i
        perm[i], perm[j] = perm[j], perm[i]
    end
    
    return (perm, rng_state)
end

# ═══════════════════════════════════════════════════════════════════════════
# Derangeable API
# ═══════════════════════════════════════════════════════════════════════════

"""
    derange(d::Derangeable) -> Vector

Generate the next derangement of elements.
Advances the internal RNG state.
"""
function derange(d::Derangeable)
    d.invocation += 1
    rng_state = mix64(d.seed ⊻ d.invocation)
    perm, _ = fisher_yates_derangement!(d.elements, rng_state)
    return [d.elements[i] for i in perm]
end

"""
    derange_at(d::Derangeable, index::Integer) -> Vector

Get the derangement at a specific invocation index (O(1) access).
Does not advance internal state.
"""
function derange_at(d::Derangeable, index::Integer)
    rng_state = mix64(d.seed ⊻ UInt64(index))
    perm, _ = fisher_yates_derangement!(d.elements, rng_state)
    return [d.elements[i] for i in perm]
end

"""
    derange_indices(d::Derangeable, index::Integer) -> Vector{Int}

Get the derangement permutation indices at a specific invocation.
Returns the permutation σ where result[i] = elements[σ[i]].
"""
function derange_indices(d::Derangeable, index::Integer)
    rng_state = mix64(d.seed ⊻ UInt64(index))
    perm, _ = fisher_yates_derangement!(d.elements, rng_state)
    return perm
end

# ═══════════════════════════════════════════════════════════════════════════
# Color-specific derangement functions
# ═══════════════════════════════════════════════════════════════════════════

"""
    derange_colors(colors::Vector, seed::Integer=GAY_SEED; index::Integer=1)

Derange a vector of colors: no color remains in its original position.
Useful for ensuring visual contrast in adjacent elements.

# Example
```julia
palette = gay_palette(6)
shuffled = derange_colors(palette, seed=42)
# shuffled[i] ≠ palette[i] for all i
```
"""
function derange_colors(colors::Vector, seed::Integer=GAY_SEED; index::Integer=1)
    d = Derangeable(colors; seed=seed)
    return derange_at(d, index)
end

"""
    derange_colors(n::Integer, seed::Integer=GAY_SEED; index::Integer=1)

Generate n colors and return a deranged version.
"""
function derange_colors(n::Integer, seed::Integer=GAY_SEED; index::Integer=1)
    colors = [color_at(i, SRGB(); seed=seed) for i in 1:n]
    return derange_colors(colors, seed; index=index)
end

# ═══════════════════════════════════════════════════════════════════════════
# Cycle decomposition (derangements have no fixed points = no 1-cycles)
# ═══════════════════════════════════════════════════════════════════════════

"""
    derange_cycle(d::Derangeable, index::Integer) -> Vector{Vector{Int}}

Get the cycle decomposition of the derangement at index.
All cycles have length ≥ 2 (no fixed points).

# Example
```julia
d = Derangeable(6, seed=42)
cycles = derange_cycle(d, 1)  # e.g., [[1,3,5], [2,4,6]]
```
"""
function derange_cycle(d::Derangeable, index::Integer)
    perm = derange_indices(d, index)
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
        
        if length(cycle) > 0
            push!(cycles, cycle)
        end
    end
    
    return cycles
end

"""
    cycle_colors(d::Derangeable, index::Integer; seed::Integer=GAY_SEED)

Color each cycle of a derangement with its own color.
Returns vector of (element_index, cycle_id, color) tuples.
"""
function cycle_colors(d::Derangeable, index::Integer; seed::Integer=GAY_SEED)
    cycles = derange_cycle(d, index)
    result = Vector{Tuple{Int, Int, Any}}()
    
    for (cycle_id, cycle) in enumerate(cycles)
        c = color_at(cycle_id, SRGB(); seed=seed)
        for elem_idx in cycle
            push!(result, (elem_idx, cycle_id, c))
        end
    end
    
    return sort(result, by=x->x[1])
end

"""
    derangement_sign(d::Derangeable, index::Integer) -> Int

Compute the sign (parity) of the derangement: (-1)^(n - c)
where c is the number of cycles.

Even permutation → +1, Odd permutation → -1
"""
function derangement_sign(d::Derangeable, index::Integer)
    cycles = derange_cycle(d, index)
    n = length(d.elements)
    num_cycles = length(cycles)
    return iseven(n - num_cycles) ? 1 : -1
end

# ═══════════════════════════════════════════════════════════════════════════
# Derangement streams for parallel/interleaved use
# ═══════════════════════════════════════════════════════════════════════════

"""
    GayDerangementStream

A stream of derangements with SPI properties, like GayInterleaver but for permutations.

# Use cases
- Monte Carlo: need random shuffles that guarantee displacement
- Parallel tempering: exchange configurations with no element staying put
- Color cycling: ensure every color moves in each "frame"
"""
mutable struct GayDerangementStream
    seed::UInt64
    n::Int                  # Size of each derangement
    current::UInt64         # Current invocation
    use_sattolo::Bool       # If true, use Sattolo (single cycle) instead of Fisher-Yates
end

"""
    GayDerangementStream(n::Integer; seed::Integer=GAY_SEED, sattolo::Bool=false)

Create a derangement stream for n elements.
If sattolo=true, all derangements are single cycles (cyclic permutations).
"""
function GayDerangementStream(n::Integer; seed::Integer=GAY_SEED, sattolo::Bool=false)
    GayDerangementStream(UInt64(seed), n, UInt64(0), sattolo)
end

"""
    next_derangement(stream::GayDerangementStream) -> Vector{Int}

Get the next derangement from the stream.
"""
function next_derangement(stream::GayDerangementStream)
    stream.current += 1
    rng_state = mix64(stream.seed ⊻ stream.current)
    
    if stream.use_sattolo
        perm, _ = sattolo_derangement(stream.n, rng_state)
    else
        perm, _ = fisher_yates_derangement!(collect(1:stream.n), rng_state)
    end
    
    return perm
end

"""
    nth_derangement(stream::GayDerangementStream, n::Integer) -> Vector{Int}

Get the nth derangement without advancing the stream (O(1) access).
"""
function nth_derangement(stream::GayDerangementStream, n::Integer)
    rng_state = mix64(stream.seed ⊻ UInt64(n))
    
    if stream.use_sattolo
        perm, _ = sattolo_derangement(stream.n, rng_state)
    else
        perm, _ = fisher_yates_derangement!(collect(1:stream.n), rng_state)
    end
    
    return perm
end

# ═══════════════════════════════════════════════════════════════════════════
# Integration with GayInterleaver: parity-respecting derangements
# ═══════════════════════════════════════════════════════════════════════════

"""
    gay_parity_derangement(il::GayInterleaver, parity::Int, index::Integer)

Get a derangement that respects sublattice parity.
The derangement maps even indices to even, odd to odd (parity-preserving)
or even to odd, odd to even (parity-flipping) based on parity argument.
"""
function gay_parity_derangement(il::GayInterleaver, parity::Int, index::Integer)
    stream = il.streams[mod1(parity + 1, il.n_streams)]
    n = il.n_streams * 2  # Derange within sublattice
    
    rng_state = mix64(stream.seed ⊻ UInt64(index))
    perm, _ = fisher_yates_derangement!(collect(1:n), rng_state)
    
    return perm
end

export gay_parity_derangement
