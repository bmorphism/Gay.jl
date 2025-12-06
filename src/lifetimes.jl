# Computation Traces: Bounded and Unbounded Color Assignment
#
# Every deterministic computation has a trace of operations. Some traces are bounded
# (finite, terminating), others are unbounded (infinite, non-terminating but productive).
#
# This module assigns SPI-guaranteed colors to computations based on their
# trace characteristics:
#
# BOUNDED COMPUTATIONS (finite):
#   - Pure functions: f(x) → y terminates
#   - Bounded loops: for i in 1:n
#   - Reductions: sum, fold, aggregate
#   - Color: derived from (seed, computation_id, iteration)
#   - Fingerprint: XOR of all iteration colors → single UInt32
#
# UNBOUNDED COMPUTATIONS (infinite sequences):
#   - Streams: infinite lazy sequences
#   - Servers: request-response loops
#   - Reactive systems: event → response forever
#   - Color: derived from (seed, computation_id, iteration)
#   - Fingerprint: rolling XOR window (last N iterations)
#
# METATHEORY:
#   - Bounded computations form a commutative monoidal category (parallel composition)
#   - Unbounded computations form a traced monoidal category (feedback loops)
#   - The "extend" functor lifts bounded → unbounded (finite iteration to infinite sequence)
#   - The "project" functor projects unbounded → bounded (take n iterations from infinite)

module Lifetimes

using Gay: hash_color, xor_fingerprint
using Colors: RGB

export BoundedComputation, UnboundedComputation
export bounded_iter!, bounded_finalize!, bounded_color, bounded_fingerprint
export unbounded_iter!, unbounded_color, unbounded_fingerprint
export parallel_bounded_colors, parallel_unbounded_iterations
export @bounded, @unbounded, extend, project
export compose_bounded, parallel_bounded, trace_unbounded
export ComputationTrace, Bounded, Unbounded, Extended

# ═══════════════════════════════════════════════════════════════════════════════
# Lifetime Types
# ═══════════════════════════════════════════════════════════════════════════════

abstract type ComputationTrace end

"""
Bounded trace: finite iterations, guaranteed termination.
Color sequence has finite length, fingerprint is complete.
"""
struct Bounded <: ComputationTrace end

"""
Unbounded trace: infinite iterations, productive (yields values forever).
Color sequence is infinite, fingerprint is a sliding window.
"""
struct Unbounded <: ComputationTrace end

"""
Extended trace: was bounded, lifted to unbounded via iteration.
Carries the original bounded seed but produces infinite sequence.
"""
struct Extended <: ComputationTrace
    bounded_seed::UInt64
    iteration::Int
end

# ═══════════════════════════════════════════════════════════════════════════════
# Bounded Computation Context
# ═══════════════════════════════════════════════════════════════════════════════

"""
    BoundedComputation

A computation with finite iteration count. Tracks iterations and accumulates colors
for final fingerprint computation.

Fields:
- `seed`: Deterministic seed for SPI
- `id`: Unique computation identifier
- `iter`: Current iteration (1-indexed)
- `max_iters`: Maximum allowed iterations (nothing = unbounded but finite)
- `colors`: Accumulated iteration colors (Float32 for GPU compatibility)
- `active`: Whether computation is still running
"""
mutable struct BoundedComputation
    seed::UInt64
    id::UInt64
    iter::Int
    max_iters::Union{Int, Nothing}
    colors::Matrix{Float32}  # n × 3
    active::Bool
end

function BoundedComputation(seed::Integer; id::Integer=1, max_iters::Union{Int,Nothing}=nothing)
    initial_capacity = max_iters === nothing ? 1000 : max_iters
    colors = zeros(Float32, initial_capacity, 3)
    BoundedComputation(UInt64(seed), UInt64(id), 0, max_iters, colors, true)
end

"""
    bounded_iter!(bc::BoundedComputation) -> RGB{Float32}

Advance computation by one iteration, return the iteration's color.
"""
function bounded_iter!(bc::BoundedComputation)
    !bc.active && error("Computation has already terminated")

    bc.iter += 1

    # Check finiteness bound
    if bc.max_iters !== nothing && bc.iter > bc.max_iters
        bc.active = false
        error("Computation exceeded max_iters=$(bc.max_iters)")
    end

    # Grow color buffer if needed
    if bc.iter > size(bc.colors, 1)
        new_colors = zeros(Float32, size(bc.colors, 1) * 2, 3)
        new_colors[1:size(bc.colors,1), :] = bc.colors
        bc.colors = new_colors
    end

    # Compute color for this iteration
    # Hash combines: seed ⊕ (id * golden) ⊕ (iter * prime)
    h = bc.seed ⊻ (bc.id * 0x9e3779b97f4a7c15) ⊻ (UInt64(bc.iter) * 0x517cc1b727220a95)
    r, g, b = hash_color(h, UInt64(bc.iter))

    bc.colors[bc.iter, 1] = r
    bc.colors[bc.iter, 2] = g
    bc.colors[bc.iter, 3] = b

    RGB{Float32}(r, g, b)
end

"""
    bounded_finalize!(bc::BoundedComputation) -> UInt32

Finalize the computation and return its fingerprint.
"""
function bounded_finalize!(bc::BoundedComputation)
    bc.active = false
    bounded_fingerprint(bc)
end

"""
    bounded_fingerprint(bc::BoundedComputation) -> UInt32

XOR fingerprint of all accumulated colors.
"""
function bounded_fingerprint(bc::BoundedComputation)
    if bc.iter == 0
        return UInt32(0)
    end
    xor_fingerprint(view(bc.colors, 1:bc.iter, :))
end

"""
    bounded_color(seed, id, iter) -> RGB{Float32}

Pure function: compute the color for a specific (seed, id, iter) triple.
No state required - this is the O(1) random access property.
"""
function bounded_color(seed::Integer, id::Integer, iter::Integer)
    h = UInt64(seed) ⊻ (UInt64(id) * 0x9e3779b97f4a7c15) ⊻ (UInt64(iter) * 0x517cc1b727220a95)
    r, g, b = hash_color(h, UInt64(iter))
    RGB{Float32}(r, g, b)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Unbounded Computation Context
# ═══════════════════════════════════════════════════════════════════════════════

"""
    UnboundedComputation

A computation with unbounded iteration. Tracks iterations and maintains
a rolling fingerprint window.

Fields:
- `seed`: Deterministic seed for SPI
- `id`: Unique computation identifier
- `iter`: Current iteration (can grow forever)
- `window_size`: Size of rolling fingerprint window
- `window`: Circular buffer of recent iteration colors
- `window_pos`: Current position in circular buffer
- `total_iters`: Total iterations processed (for statistics)
"""
mutable struct UnboundedComputation
    seed::UInt64
    id::UInt64
    iter::Int
    window_size::Int
    window::Matrix{Float32}  # window_size × 3
    window_pos::Int
    total_iters::Int
end

function UnboundedComputation(seed::Integer; id::Integer=1, window_size::Int=1000)
    window = zeros(Float32, window_size, 3)
    UnboundedComputation(UInt64(seed), UInt64(id), 0, window_size, window, 0, 0)
end

"""
    unbounded_iter!(uc::UnboundedComputation) -> RGB{Float32}

Advance to next iteration, return the iteration's color.
Unbounded computations never terminate - this can be called forever.
"""
function unbounded_iter!(uc::UnboundedComputation)
    uc.iter += 1
    uc.total_iters += 1

    # Circular buffer position
    uc.window_pos = mod1(uc.iter, uc.window_size)

    # Compute color for this iteration
    # Different mixing than bounded to distinguish trace types
    h = uc.seed ⊻ (uc.id * 0x85ebca6b) ⊻ (UInt64(uc.iter) * 0xc2b2ae35)
    r, g, b = hash_color(h, UInt64(uc.iter))

    uc.window[uc.window_pos, 1] = r
    uc.window[uc.window_pos, 2] = g
    uc.window[uc.window_pos, 3] = b

    RGB{Float32}(r, g, b)
end

"""
    unbounded_fingerprint(uc::UnboundedComputation) -> UInt32

Rolling XOR fingerprint of the recent window.
This changes as new iterations are processed, but is deterministic
for a given (seed, id, iteration_range).
"""
function unbounded_fingerprint(uc::UnboundedComputation)
    if uc.total_iters == 0
        return UInt32(0)
    end

    # Only fingerprint filled portion of window
    n_filled = min(uc.total_iters, uc.window_size)
    xor_fingerprint(view(uc.window, 1:n_filled, :))
end

"""
    unbounded_color(seed, id, iter) -> RGB{Float32}

Pure function: compute the color for a specific (seed, id, iter) triple.
"""
function unbounded_color(seed::Integer, id::Integer, iter::Integer)
    h = UInt64(seed) ⊻ (UInt64(id) * 0x85ebca6b) ⊻ (UInt64(iter) * 0xc2b2ae35)
    r, g, b = hash_color(h, UInt64(iter))
    RGB{Float32}(r, g, b)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Functors: Extend (Bounded → Unbounded) and Project (Unbounded → Bounded)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    extend(bc::BoundedComputation) -> UnboundedComputation

Lift a bounded computation to an unbounded one by iterating its pattern forever.
The extended computation cycles through the bounded's color sequence.
"""
function extend(bc::BoundedComputation)
    # Create unbounded with memory of bounded origin
    uc = UnboundedComputation(bc.seed; id=bc.id, window_size=max(bc.iter, 100))

    # Pre-fill with bounded's colors (repeating cycle)
    for i in 1:bc.iter
        uc.window[mod1(i, uc.window_size), :] = bc.colors[i, :]
    end
    uc.iter = bc.iter
    uc.total_iters = bc.iter
    uc.window_pos = mod1(bc.iter, uc.window_size)

    uc
end

"""
    project(uc::UnboundedComputation, n::Int) -> BoundedComputation

Project an unbounded computation to a bounded one by taking n iterations.
"""
function project(uc::UnboundedComputation, n::Int)
    bc = BoundedComputation(uc.seed; id=uc.id, max_iters=n)

    # Generate n iterations
    for i in 1:n
        bounded_iter!(bc)
    end
    bc.active = false

    bc
end

# ═══════════════════════════════════════════════════════════════════════════════
# Parallel SPI for Bounded Computations
# ═══════════════════════════════════════════════════════════════════════════════

"""
    parallel_bounded_colors(seed, n_computations, iters_per_computation) -> Vector{UInt32}

Generate colors for many bounded computations in parallel.
Returns vector of fingerprints: each element is one computation.

SPI Guarantee: Same (seed, n_computations, iters) → same fingerprints,
regardless of parallel execution order.
"""
function parallel_bounded_colors(seed::Integer, n_computations::Int, iters::Int)
    fingerprints = zeros(UInt32, n_computations)

    Threads.@threads for id in 1:n_computations
        bc = BoundedComputation(seed; id=id, max_iters=iters)
        for _ in 1:iters
            bounded_iter!(bc)
        end
        fingerprints[id] = bounded_fingerprint(bc)
    end

    fingerprints
end

"""
    parallel_unbounded_iterations(seed, n_computations, iters_per_computation) -> Vector{UInt32}

Generate rolling fingerprints for many unbounded computations in parallel.
"""
function parallel_unbounded_iterations(seed::Integer, n_computations::Int, iters::Int)
    fingerprints = zeros(UInt32, n_computations)

    Threads.@threads for id in 1:n_computations
        uc = UnboundedComputation(seed; id=id)
        for _ in 1:iters
            unbounded_iter!(uc)
        end
        fingerprints[id] = unbounded_fingerprint(uc)
    end

    fingerprints
end

# ═══════════════════════════════════════════════════════════════════════════════
# Macros for Computation Decoration
# ═══════════════════════════════════════════════════════════════════════════════

"""
    @bounded seed expr

Execute expr as a bounded computation with colored iterations.
Returns (result, fingerprint).
"""
macro bounded(seed, expr)
    quote
        local bc = BoundedComputation($(esc(seed)))
        local result = $(esc(expr))
        local fp = bounded_finalize!(bc)
        (result=result, fingerprint=fp, iters=bc.iter)
    end
end

"""
    @unbounded seed n expr

Execute expr as first n iterations of an unbounded computation.
Returns (results, fingerprint).
"""
macro unbounded(seed, n, expr)
    quote
        local uc = UnboundedComputation($(esc(seed)))
        local results = []
        for _iter in 1:$(esc(n))
            unbounded_iter!(uc)
            push!(results, $(esc(expr)))
        end
        (results=results, fingerprint=unbounded_fingerprint(uc), iters=uc.total_iters)
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Metatheory: Categorical Structure
# ═══════════════════════════════════════════════════════════════════════════════

"""
    compose_bounded(bc1, bc2) -> BoundedComputation

Sequential composition: run bc1 then bc2, combine fingerprints.
Forms a monoid: identity is empty computation, associative.
"""
function compose_bounded(bc1::BoundedComputation, bc2::BoundedComputation)
    # New computation inherits combined seed
    combined_seed = bc1.seed ⊻ bc2.seed ⊻ (UInt64(bc1.iter) * 0x9e3779b97f4a7c15)
    bc = BoundedComputation(combined_seed; id=bc1.id ⊻ bc2.id)

    # Copy colors from both
    for i in 1:bc1.iter
        bc.colors[i, :] = bc1.colors[i, :]
    end
    for i in 1:bc2.iter
        bc.colors[bc1.iter + i, :] = bc2.colors[i, :]
    end
    bc.iter = bc1.iter + bc2.iter
    bc.active = false

    bc
end

"""
    parallel_bounded(bc1, bc2) -> BoundedComputation

Parallel composition: run bc1 and bc2 simultaneously.
Interleaves colors, XOR combines fingerprints.
"""
function parallel_bounded(bc1::BoundedComputation, bc2::BoundedComputation)
    combined_seed = bc1.seed ⊻ bc2.seed
    max_iters = max(bc1.iter, bc2.iter)
    bc = BoundedComputation(combined_seed; id=bc1.id + bc2.id * 1000)

    for i in 1:max_iters
        if i <= bc1.iter
            # XOR with bc1's color
            bc.colors[i, 1] = bc1.colors[i, 1]
            bc.colors[i, 2] = bc1.colors[i, 2]
            bc.colors[i, 3] = bc1.colors[i, 3]
        end
        if i <= bc2.iter
            # XOR combine (via reinterpret for bitwise)
            c1 = reinterpret(UInt32, bc.colors[i, 1])
            c2 = reinterpret(UInt32, bc2.colors[i, 1])
            bc.colors[i, 1] = reinterpret(Float32, c1 ⊻ c2)

            c1 = reinterpret(UInt32, bc.colors[i, 2])
            c2 = reinterpret(UInt32, bc2.colors[i, 2])
            bc.colors[i, 2] = reinterpret(Float32, c1 ⊻ c2)

            c1 = reinterpret(UInt32, bc.colors[i, 3])
            c2 = reinterpret(UInt32, bc2.colors[i, 3])
            bc.colors[i, 3] = reinterpret(Float32, c1 ⊻ c2)
        end
        bc.iter = i
    end
    bc.active = false

    bc
end

"""
    trace_unbounded(uc::UnboundedComputation, f::Function) -> UnboundedComputation

Traced composition: feed output back as input (fixpoint).
This is the categorical trace for unbounded computations.
"""
function trace_unbounded(uc::UnboundedComputation, f::Function)
    # The trace feeds the previous iteration's color into computing the next
    # This creates feedback loops characteristic of unbounded computations

    # Get current color as "state"
    if uc.total_iters > 0
        state_r = uc.window[uc.window_pos, 1]
        state_g = uc.window[uc.window_pos, 2]
        state_b = uc.window[uc.window_pos, 3]
        state = RGB{Float32}(state_r, state_g, state_b)
    else
        state = RGB{Float32}(0.5f0, 0.5f0, 0.5f0)
    end

    # Apply f to get modification
    modified = f(state)

    # Advance iteration with modified influence
    color = unbounded_iter!(uc)

    # Blend original and modified
    blended = RGB{Float32}(
        (color.r + modified.r) / 2,
        (color.g + modified.g) / 2,
        (color.b + modified.b) / 2
    )

    # Update window with blended
    uc.window[uc.window_pos, 1] = blended.r
    uc.window[uc.window_pos, 2] = blended.g
    uc.window[uc.window_pos, 3] = blended.b

    uc
end

# ═══════════════════════════════════════════════════════════════════════════════
# Display
# ═══════════════════════════════════════════════════════════════════════════════

function Base.show(io::IO, bc::BoundedComputation)
    status = bc.active ? "active" : "finalized"
    fp = bounded_fingerprint(bc)
    print(io, "BoundedComputation(seed=0x$(string(bc.seed, base=16)), ")
    print(io, "id=$(bc.id), iters=$(bc.iter), $status, ")
    print(io, "fp=0x$(string(fp, base=16, pad=8)))")
end

function Base.show(io::IO, uc::UnboundedComputation)
    fp = unbounded_fingerprint(uc)
    print(io, "UnboundedComputation(seed=0x$(string(uc.seed, base=16)), ")
    print(io, "id=$(uc.id), iters=$(uc.total_iters), ")
    print(io, "window=$(uc.window_size), fp=0x$(string(fp, base=16, pad=8)))")
end

end # module
