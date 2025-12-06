# GayMC - Colored Monte Carlo with SPI (Strong Parallelism Invariance)
# Every sweep, measure, and checkpoint gets a deterministic color
# Compatible with Carlo.jl's AbstractMC interface

using SplittableRandoms: SplittableRandom, split
using Colors

export GayMCContext
export gay_sweep!, gay_measure!, gay_checkpoint
export color_sweep, color_measure, color_state

# ═══════════════════════════════════════════════════════════════════════════
# FNV-1a hash for deterministic seeds
# ═══════════════════════════════════════════════════════════════════════════

function fnv1a(text::String)::UInt64
    h = UInt64(14695981039346656037)
    for c in text
        h = (h ⊻ UInt64(c)) * UInt64(1099511628211)
    end
    h
end

function fnv1a(data::Vector{UInt8})::UInt64
    h = UInt64(14695981039346656037)
    for b in data
        h = (h ⊻ UInt64(b)) * UInt64(1099511628211)
    end
    h
end

# ═══════════════════════════════════════════════════════════════════════════
# GayMCContext - Parallel-safe colored context for MC simulations
# ═══════════════════════════════════════════════════════════════════════════

"""
    GayMCContext

A Monte Carlo context that provides:
- Splittable RNG for Strong Parallelism Invariance (SPI)
- Deterministic colors for every sweep/measure/checkpoint
- HDF5-compatible state serialization with color metadata

Each parallel worker gets an independent but deterministic stream.
"""
mutable struct GayMCContext
    # Splittable RNG root (for SPI)
    root_rng::SplittableRandom
    
    # Current RNG (splits on each operation)
    current_rng::SplittableRandom
    
    # Counters for deterministic indexing
    sweep_count::UInt64
    measure_count::UInt64
    checkpoint_count::UInt64
    
    # Worker/replica ID for parallel runs
    worker_id::UInt64
    
    # Base seed for reproducibility
    seed::UInt64
    
    # Color history (optional, for visualization)
    color_history::Vector{RGB{Float64}}
end

"""
    GayMCContext(seed::Integer; worker_id::Integer=0)

Create a new colored MC context with the given seed.
Each worker_id gets an independent stream via splitting.
"""
function GayMCContext(seed::Integer; worker_id::Integer=0)
    root = SplittableRandom(UInt64(seed))
    
    # Split for this worker
    current = root
    for _ in 1:worker_id+1
        current = split(current)
    end
    
    GayMCContext(
        root,
        current,
        UInt64(0),
        UInt64(0),
        UInt64(0),
        UInt64(worker_id),
        UInt64(seed),
        RGB{Float64}[]
    )
end

# ═══════════════════════════════════════════════════════════════════════════
# Color generation from RNG state
# ═══════════════════════════════════════════════════════════════════════════

"""
Generate a deterministic color from a SplittableRandom state.
Uses the RNG to produce wide-gamut RGB values.
"""
function color_from_rng(rng::SplittableRandom)::RGB{Float64}
    # Use the RNG to generate RGB in [0,1]
    r = rand(rng)
    g = rand(rng)
    b = rand(rng)
    
    # Boost saturation for visual distinction
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    if max_c > min_c
        scale = 1.0 / (max_c - min_c + 0.3)
        r = (r - min_c) * scale
        g = (g - min_c) * scale
        b = (b - min_c) * scale
    end
    
    RGB{Float64}(clamp(r, 0, 1), clamp(g, 0, 1), clamp(b, 0, 1))
end

"""
Generate color at a specific index in the sequence.
"""
function color_at_index(seed::UInt64, index::UInt64)::RGB{Float64}
    rng = SplittableRandom(seed)
    for _ in 1:index
        rng = split(rng)
    end
    color_from_rng(rng)
end

# ═══════════════════════════════════════════════════════════════════════════
# Colored MC Operations
# ═══════════════════════════════════════════════════════════════════════════

"""
    gay_sweep!(ctx::GayMCContext) -> (rng, color)

Perform a colored sweep. Returns the RNG to use and the sweep's color.
Each sweep gets a unique, reproducible color.
"""
function gay_sweep!(ctx::GayMCContext)
    ctx.sweep_count += 1
    ctx.current_rng = split(ctx.current_rng)
    
    # Color for this sweep
    color = color_from_rng(split(ctx.current_rng))
    push!(ctx.color_history, color)
    
    (ctx.current_rng, color)
end

"""
    gay_measure!(ctx::GayMCContext, name::String) -> (rng, color)

Perform a colored measurement. The color encodes both the measurement
index and the observable name.
"""
function gay_measure!(ctx::GayMCContext, name::String="")
    ctx.measure_count += 1
    ctx.current_rng = split(ctx.current_rng)
    
    # Mix name into color seed for observable-specific colors
    name_seed = fnv1a(name)
    combined_seed = xor(ctx.seed, name_seed, ctx.measure_count)
    color = color_at_index(combined_seed, UInt64(1))
    
    (ctx.current_rng, color)
end

"""
    gay_checkpoint(ctx::GayMCContext) -> Dict

Create a checkpoint with color metadata for HDF5 storage.
"""
function gay_checkpoint(ctx::GayMCContext)
    ctx.checkpoint_count += 1
    
    # Checkpoint color
    color = color_at_index(ctx.seed, ctx.checkpoint_count)
    
    Dict(
        "seed" => ctx.seed,
        "worker_id" => ctx.worker_id,
        "sweep_count" => ctx.sweep_count,
        "measure_count" => ctx.measure_count,
        "checkpoint_count" => ctx.checkpoint_count,
        "checkpoint_color" => [red(color), green(color), blue(color)],
        "color_history_r" => [red(c) for c in ctx.color_history],
        "color_history_g" => [green(c) for c in ctx.color_history],
        "color_history_b" => [blue(c) for c in ctx.color_history]
    )
end

"""
    gay_restore!(ctx::GayMCContext, checkpoint::Dict)

Restore context from a checkpoint.
"""
function gay_restore!(ctx::GayMCContext, checkpoint::Dict)
    ctx.seed = checkpoint["seed"]
    ctx.worker_id = checkpoint["worker_id"]
    ctx.sweep_count = checkpoint["sweep_count"]
    ctx.measure_count = checkpoint["measure_count"]
    ctx.checkpoint_count = checkpoint["checkpoint_count"]
    
    # Reconstruct RNG state by replaying splits
    ctx.root_rng = SplittableRandom(ctx.seed)
    ctx.current_rng = ctx.root_rng
    for _ in 1:ctx.worker_id+1
        ctx.current_rng = split(ctx.current_rng)
    end
    for _ in 1:ctx.sweep_count
        ctx.current_rng = split(ctx.current_rng)
    end
    
    # Restore color history
    if haskey(checkpoint, "color_history_r")
        r = checkpoint["color_history_r"]
        g = checkpoint["color_history_g"]
        b = checkpoint["color_history_b"]
        ctx.color_history = [RGB{Float64}(r[i], g[i], b[i]) for i in eachindex(r)]
    end
    
    ctx
end

# ═══════════════════════════════════════════════════════════════════════════
# Convenience accessors
# ═══════════════════════════════════════════════════════════════════════════

"""Current sweep color."""
color_sweep(ctx::GayMCContext) = isempty(ctx.color_history) ? 
    RGB{Float64}(0.5, 0.5, 0.5) : ctx.color_history[end]

"""Color for a specific sweep index."""
color_sweep(ctx::GayMCContext, idx::Integer) = 
    1 <= idx <= length(ctx.color_history) ? ctx.color_history[idx] : 
    color_at_index(ctx.seed, UInt64(idx))

"""Color for a named measurement."""
function color_measure(ctx::GayMCContext, name::String)
    name_seed = fnv1a(name)
    combined = xor(ctx.seed, name_seed)
    color_at_index(combined, UInt64(1))
end

"""Overall state color (XOR of recent sweep colors)."""
function color_state(ctx::GayMCContext; window::Int=10)
    if isempty(ctx.color_history)
        return RGB{Float64}(0.5, 0.5, 0.5)
    end
    
    recent = ctx.color_history[max(1, end-window+1):end]
    r = sum(red(c) for c in recent) / length(recent)
    g = sum(green(c) for c in recent) / length(recent)
    b = sum(blue(c) for c in recent) / length(recent)
    RGB{Float64}(r, g, b)
end

# ═══════════════════════════════════════════════════════════════════════════
# Non-Gaussian sampling with colored RNG
# ═══════════════════════════════════════════════════════════════════════════

"""
Sample from exponential distribution with colored RNG.
"""
function gay_exponential!(ctx::GayMCContext, λ::Float64=1.0)
    rng, color = gay_sweep!(ctx)
    u = rand(rng)
    (-log(u) / λ, color)
end

"""
Sample from Cauchy distribution (heavy-tailed, non-Gaussian).
"""
function gay_cauchy!(ctx::GayMCContext, x0::Float64=0.0, γ::Float64=1.0)
    rng, color = gay_sweep!(ctx)
    u = rand(rng)
    (x0 + γ * tan(π * (u - 0.5)), color)
end

"""
Sample from Gaussian using Box-Muller.
"""
function gay_gaussian!(ctx::GayMCContext, μ::Float64=0.0, σ::Float64=1.0)
    rng, color = gay_sweep!(ctx)
    u1 = rand(rng)
    u2 = rand(rng)
    z = sqrt(-2 * log(u1)) * cos(2π * u2)
    (μ + σ * z, color)
end

"""
Sample from Metropolis-Hastings accept/reject.
Returns (accepted::Bool, color).
"""
function gay_metropolis!(ctx::GayMCContext, ΔE::Float64, β::Float64=1.0)
    rng, color = gay_sweep!(ctx)
    if ΔE <= 0
        (true, color)
    else
        u = rand(rng)
        (u < exp(-β * ΔE), color)
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# Parallel worker creation
# ═══════════════════════════════════════════════════════════════════════════

"""
Create n parallel workers, each with independent deterministic streams.
"""
function gay_workers(seed::Integer, n::Int)
    [GayMCContext(seed; worker_id=i-1) for i in 1:n]
end

"""
Create workers for parallel tempering at different temperatures.
"""
function gay_tempering(seed::Integer, temperatures::Vector{Float64})
    workers = gay_workers(seed, length(temperatures))
    [(ctx=w, T=t, β=1.0/t) for (w, t) in zip(workers, temperatures)]
end
