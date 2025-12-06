# Computation Lifetimes: Mortal and Immortal Color Assignment
#
# Every computation has a lifetime. Some are mortal (finite, terminating),
# others are immortal (infinite, non-terminating but productive).
#
# This module assigns SPI-guaranteed colors to computations based on their
# lifetime characteristics:
#
# MORTAL COMPUTATIONS (finite):
#   - Pure functions: f(x) → y terminates
#   - Bounded loops: for i in 1:n
#   - Reductions: sum, fold, aggregate
#   - Color: derived from (seed, computation_id, step)
#   - Fingerprint: XOR of all step colors → single UInt32
#
# IMMORTAL COMPUTATIONS (productive infinity):
#   - Streams: infinite lazy sequences
#   - Servers: request-response loops
#   - Reactive systems: event → response forever
#   - Color: derived from (seed, computation_id, epoch)
#   - Fingerprint: rolling XOR window (last N epochs)
#
# METATHEORY:
#   - Mortal computations form a commutative monoidal category (parallel composition)
#   - Immortal computations form a traced monoidal category (feedback loops)
#   - The "ascension" functor lifts mortal → immortal (iteration to stream)
#   - The "harvest" functor projects immortal → mortal (take n from stream)

module Lifetimes

using Gay: hash_color, xor_fingerprint
using Colors: RGB

export MortalComputation, ImmortalComputation
export mortal_step!, mortal_terminate!, mortal_color, mortal_fingerprint
export immortal_epoch!, immortal_color, immortal_fingerprint
export parallel_mortal_colors, parallel_immortal_epochs
export @mortal, @immortal, ascend, harvest
export compose_mortal, parallel_mortal, trace_immortal
export ComputationLifetime, Mortal, Immortal, Ascended

# ═══════════════════════════════════════════════════════════════════════════════
# Lifetime Types
# ═══════════════════════════════════════════════════════════════════════════════

abstract type ComputationLifetime end

"""
Mortal computation: finite steps, guaranteed termination.
Color sequence has finite length, fingerprint is complete.
"""
struct Mortal <: ComputationLifetime end

"""
Immortal computation: infinite steps, productive (yields values forever).
Color sequence is infinite, fingerprint is a sliding window.
"""
struct Immortal <: ComputationLifetime end

"""
Ascended computation: was mortal, lifted to immortal via iteration.
Carries the original mortal seed but produces infinite stream.
"""
struct Ascended <: ComputationLifetime
    mortal_seed::UInt64
    iteration::Int
end

# ═══════════════════════════════════════════════════════════════════════════════
# Mortal Computation Context
# ═══════════════════════════════════════════════════════════════════════════════

"""
    MortalComputation

A computation with finite lifetime. Tracks steps and accumulates colors
for final fingerprint computation.

Fields:
- `seed`: Deterministic seed for SPI
- `id`: Unique computation identifier  
- `step`: Current step (1-indexed)
- `max_steps`: Maximum allowed steps (nothing = unbounded but finite)
- `colors`: Accumulated step colors (Float32 for GPU compatibility)
- `alive`: Whether computation is still running
"""
mutable struct MortalComputation
    seed::UInt64
    id::UInt64
    step::Int
    max_steps::Union{Int, Nothing}
    colors::Matrix{Float32}  # n × 3
    alive::Bool
end

function MortalComputation(seed::Integer; id::Integer=1, max_steps::Union{Int,Nothing}=nothing)
    initial_capacity = max_steps === nothing ? 1000 : max_steps
    colors = zeros(Float32, initial_capacity, 3)
    MortalComputation(UInt64(seed), UInt64(id), 0, max_steps, colors, true)
end

"""
    mortal_step!(mc::MortalComputation) -> RGB{Float32}

Advance computation by one step, return the step's color.
"""
function mortal_step!(mc::MortalComputation)
    !mc.alive && error("Computation has already terminated")
    
    mc.step += 1
    
    # Check mortality
    if mc.max_steps !== nothing && mc.step > mc.max_steps
        mc.alive = false
        error("Computation exceeded max_steps=$(mc.max_steps)")
    end
    
    # Grow color buffer if needed
    if mc.step > size(mc.colors, 1)
        new_colors = zeros(Float32, size(mc.colors, 1) * 2, 3)
        new_colors[1:size(mc.colors,1), :] = mc.colors
        mc.colors = new_colors
    end
    
    # Compute color for this step
    # Hash combines: seed ⊕ (id * golden) ⊕ (step * prime)
    h = mc.seed ⊻ (mc.id * 0x9e3779b97f4a7c15) ⊻ (UInt64(mc.step) * 0x517cc1b727220a95)
    r, g, b = hash_color(h, UInt64(mc.step))
    
    mc.colors[mc.step, 1] = r
    mc.colors[mc.step, 2] = g
    mc.colors[mc.step, 3] = b
    
    RGB{Float32}(r, g, b)
end

"""
    mortal_terminate!(mc::MortalComputation) -> UInt32

Terminate the computation and return its fingerprint.
"""
function mortal_terminate!(mc::MortalComputation)
    mc.alive = false
    mortal_fingerprint(mc)
end

"""
    mortal_fingerprint(mc::MortalComputation) -> UInt32

XOR fingerprint of all accumulated colors.
"""
function mortal_fingerprint(mc::MortalComputation)
    if mc.step == 0
        return UInt32(0)
    end
    xor_fingerprint(view(mc.colors, 1:mc.step, :))
end

"""
    mortal_color(seed, id, step) -> RGB{Float32}

Pure function: compute the color for a specific (seed, id, step) triple.
No state required - this is the O(1) random access property.
"""
function mortal_color(seed::Integer, id::Integer, step::Integer)
    h = UInt64(seed) ⊻ (UInt64(id) * 0x9e3779b97f4a7c15) ⊻ (UInt64(step) * 0x517cc1b727220a95)
    r, g, b = hash_color(h, UInt64(step))
    RGB{Float32}(r, g, b)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Immortal Computation Context
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ImmortalComputation

A computation with infinite lifetime. Tracks epochs and maintains
a rolling fingerprint window.

Fields:
- `seed`: Deterministic seed for SPI
- `id`: Unique computation identifier
- `epoch`: Current epoch (can grow forever)
- `window_size`: Size of rolling fingerprint window
- `window`: Circular buffer of recent epoch colors
- `window_pos`: Current position in circular buffer
- `total_epochs`: Total epochs processed (for statistics)
"""
mutable struct ImmortalComputation
    seed::UInt64
    id::UInt64
    epoch::Int
    window_size::Int
    window::Matrix{Float32}  # window_size × 3
    window_pos::Int
    total_epochs::Int
end

function ImmortalComputation(seed::Integer; id::Integer=1, window_size::Int=1000)
    window = zeros(Float32, window_size, 3)
    ImmortalComputation(UInt64(seed), UInt64(id), 0, window_size, window, 0, 0)
end

"""
    immortal_epoch!(ic::ImmortalComputation) -> RGB{Float32}

Advance to next epoch, return the epoch's color.
Immortal computations never terminate - this can be called forever.
"""
function immortal_epoch!(ic::ImmortalComputation)
    ic.epoch += 1
    ic.total_epochs += 1
    
    # Circular buffer position
    ic.window_pos = mod1(ic.epoch, ic.window_size)
    
    # Compute color for this epoch
    # Different mixing than mortal to distinguish lifetime types
    h = ic.seed ⊻ (ic.id * 0x85ebca6b) ⊻ (UInt64(ic.epoch) * 0xc2b2ae35)
    r, g, b = hash_color(h, UInt64(ic.epoch))
    
    ic.window[ic.window_pos, 1] = r
    ic.window[ic.window_pos, 2] = g
    ic.window[ic.window_pos, 3] = b
    
    RGB{Float32}(r, g, b)
end

"""
    immortal_fingerprint(ic::ImmortalComputation) -> UInt32

Rolling XOR fingerprint of the recent window.
This changes as new epochs are processed, but is deterministic
for a given (seed, id, epoch_range).
"""
function immortal_fingerprint(ic::ImmortalComputation)
    if ic.total_epochs == 0
        return UInt32(0)
    end
    
    # Only fingerprint filled portion of window
    n_filled = min(ic.total_epochs, ic.window_size)
    xor_fingerprint(view(ic.window, 1:n_filled, :))
end

"""
    immortal_color(seed, id, epoch) -> RGB{Float32}

Pure function: compute the color for a specific (seed, id, epoch) triple.
"""
function immortal_color(seed::Integer, id::Integer, epoch::Integer)
    h = UInt64(seed) ⊻ (UInt64(id) * 0x85ebca6b) ⊻ (UInt64(epoch) * 0xc2b2ae35)
    r, g, b = hash_color(h, UInt64(epoch))
    RGB{Float32}(r, g, b)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Functors: Ascend (Mortal → Immortal) and Harvest (Immortal → Mortal)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ascend(mc::MortalComputation) -> ImmortalComputation

Lift a mortal computation to an immortal one by iterating its pattern forever.
The ascended computation cycles through the mortal's color sequence.
"""
function ascend(mc::MortalComputation)
    # Create immortal with memory of mortal origin
    ic = ImmortalComputation(mc.seed; id=mc.id, window_size=max(mc.step, 100))
    
    # Pre-fill with mortal's colors (the eternal echo)
    for i in 1:mc.step
        ic.window[mod1(i, ic.window_size), :] = mc.colors[i, :]
    end
    ic.epoch = mc.step
    ic.total_epochs = mc.step
    ic.window_pos = mod1(mc.step, ic.window_size)
    
    ic
end

"""
    harvest(ic::ImmortalComputation, n::Int) -> MortalComputation

Project an immortal computation to a mortal one by taking n epochs.
"""
function harvest(ic::ImmortalComputation, n::Int)
    mc = MortalComputation(ic.seed; id=ic.id, max_steps=n)
    
    # Generate n steps
    for i in 1:n
        mortal_step!(mc)
    end
    mc.alive = false
    
    mc
end

# ═══════════════════════════════════════════════════════════════════════════════
# Parallel SPI for Mortal Computations
# ═══════════════════════════════════════════════════════════════════════════════

"""
    parallel_mortal_colors(seed, n_computations, steps_per_computation) -> Matrix{UInt32}

Generate colors for many mortal computations in parallel.
Returns matrix of fingerprints: each row is one computation.

SPI Guarantee: Same (seed, n_computations, steps) → same fingerprints,
regardless of parallel execution order.
"""
function parallel_mortal_colors(seed::Integer, n_computations::Int, steps::Int)
    fingerprints = zeros(UInt32, n_computations)
    
    Threads.@threads for id in 1:n_computations
        mc = MortalComputation(seed; id=id, max_steps=steps)
        for _ in 1:steps
            mortal_step!(mc)
        end
        fingerprints[id] = mortal_fingerprint(mc)
    end
    
    fingerprints
end

"""
    parallel_immortal_epochs(seed, n_computations, epochs_per_computation) -> Matrix{UInt32}

Generate rolling fingerprints for many immortal computations in parallel.
"""
function parallel_immortal_epochs(seed::Integer, n_computations::Int, epochs::Int)
    fingerprints = zeros(UInt32, n_computations)
    
    Threads.@threads for id in 1:n_computations
        ic = ImmortalComputation(seed; id=id)
        for _ in 1:epochs
            immortal_epoch!(ic)
        end
        fingerprints[id] = immortal_fingerprint(ic)
    end
    
    fingerprints
end

# ═══════════════════════════════════════════════════════════════════════════════
# Macros for Computation Decoration
# ═══════════════════════════════════════════════════════════════════════════════

"""
    @mortal seed expr

Execute expr as a mortal computation with colored steps.
Returns (result, fingerprint).
"""
macro mortal(seed, expr)
    quote
        local mc = MortalComputation($(esc(seed)))
        local result = $(esc(expr))
        local fp = mortal_terminate!(mc)
        (result=result, fingerprint=fp, steps=mc.step)
    end
end

"""
    @immortal seed n expr

Execute expr as first n epochs of an immortal computation.
Returns (results, fingerprint).
"""
macro immortal(seed, n, expr)
    quote
        local ic = ImmortalComputation($(esc(seed)))
        local results = []
        for _epoch in 1:$(esc(n))
            immortal_epoch!(ic)
            push!(results, $(esc(expr)))
        end
        (results=results, fingerprint=immortal_fingerprint(ic), epochs=ic.total_epochs)
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Metatheory: Categorical Structure
# ═══════════════════════════════════════════════════════════════════════════════

"""
    compose_mortal(mc1, mc2) -> MortalComputation

Sequential composition: run mc1 then mc2, combine fingerprints.
Forms a monoid: identity is empty computation, associative.
"""
function compose_mortal(mc1::MortalComputation, mc2::MortalComputation)
    # New computation inherits combined seed
    combined_seed = mc1.seed ⊻ mc2.seed ⊻ (UInt64(mc1.step) * 0x9e3779b97f4a7c15)
    mc = MortalComputation(combined_seed; id=mc1.id ⊻ mc2.id)
    
    # Copy colors from both
    for i in 1:mc1.step
        mc.colors[i, :] = mc1.colors[i, :]
    end
    for i in 1:mc2.step
        mc.colors[mc1.step + i, :] = mc2.colors[i, :]
    end
    mc.step = mc1.step + mc2.step
    mc.alive = false
    
    mc
end

"""
    parallel_mortal(mc1, mc2) -> MortalComputation

Parallel composition: run mc1 and mc2 simultaneously.
Interleaves colors, XOR combines fingerprints.
"""
function parallel_mortal(mc1::MortalComputation, mc2::MortalComputation)
    combined_seed = mc1.seed ⊻ mc2.seed
    max_steps = max(mc1.step, mc2.step)
    mc = MortalComputation(combined_seed; id=mc1.id + mc2.id * 1000)
    
    for i in 1:max_steps
        if i <= mc1.step
            # XOR with mc1's color
            mc.colors[i, 1] = mc1.colors[i, 1]
            mc.colors[i, 2] = mc1.colors[i, 2]
            mc.colors[i, 3] = mc1.colors[i, 3]
        end
        if i <= mc2.step
            # XOR combine (via reinterpret for bitwise)
            c1 = reinterpret(UInt32, mc.colors[i, 1])
            c2 = reinterpret(UInt32, mc2.colors[i, 1])
            mc.colors[i, 1] = reinterpret(Float32, c1 ⊻ c2)
            
            c1 = reinterpret(UInt32, mc.colors[i, 2])
            c2 = reinterpret(UInt32, mc2.colors[i, 2])
            mc.colors[i, 2] = reinterpret(Float32, c1 ⊻ c2)
            
            c1 = reinterpret(UInt32, mc.colors[i, 3])
            c2 = reinterpret(UInt32, mc2.colors[i, 3])
            mc.colors[i, 3] = reinterpret(Float32, c1 ⊻ c2)
        end
        mc.step = i
    end
    mc.alive = false
    
    mc
end

"""
    trace_immortal(ic::ImmortalComputation, f::Function) -> ImmortalComputation

Traced composition: feed output back as input (fixpoint).
This is the categorical trace for immortal computations.
"""
function trace_immortal(ic::ImmortalComputation, f::Function)
    # The trace feeds the previous epoch's color into computing the next
    # This creates feedback loops characteristic of immortal computations
    
    # Get current color as "state"
    if ic.total_epochs > 0
        state_r = ic.window[ic.window_pos, 1]
        state_g = ic.window[ic.window_pos, 2]
        state_b = ic.window[ic.window_pos, 3]
        state = RGB{Float32}(state_r, state_g, state_b)
    else
        state = RGB{Float32}(0.5f0, 0.5f0, 0.5f0)
    end
    
    # Apply f to get modification
    modified = f(state)
    
    # Advance epoch with modified influence
    color = immortal_epoch!(ic)
    
    # Blend original and modified
    blended = RGB{Float32}(
        (color.r + modified.r) / 2,
        (color.g + modified.g) / 2,
        (color.b + modified.b) / 2
    )
    
    # Update window with blended
    ic.window[ic.window_pos, 1] = blended.r
    ic.window[ic.window_pos, 2] = blended.g
    ic.window[ic.window_pos, 3] = blended.b
    
    ic
end

# ═══════════════════════════════════════════════════════════════════════════════
# Display
# ═══════════════════════════════════════════════════════════════════════════════

function Base.show(io::IO, mc::MortalComputation)
    status = mc.alive ? "alive" : "terminated"
    fp = mortal_fingerprint(mc)
    print(io, "MortalComputation(seed=0x$(string(mc.seed, base=16)), ")
    print(io, "id=$(mc.id), steps=$(mc.step), $status, ")
    print(io, "fp=0x$(string(fp, base=16, pad=8)))")
end

function Base.show(io::IO, ic::ImmortalComputation)
    fp = immortal_fingerprint(ic)
    print(io, "ImmortalComputation(seed=0x$(string(ic.seed, base=16)), ")
    print(io, "id=$(ic.id), epochs=$(ic.total_epochs), ")
    print(io, "window=$(ic.window_size), fp=0x$(string(fp, base=16, pad=8)))")
end

end # module
