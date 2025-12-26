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

# ═══════════════════════════════════════════════════════════════════════════
# Replica Exchange MCMC (Parallel Tempering) with SPI Guarantees
# ═══════════════════════════════════════════════════════════════════════════

export Replica, TemperatureLadder, ReplicaExchange
export replica_exchange!, attempt_swap!, swap_replicas!
export temperature_color, energy_color
export visualize_ladder, ladder_to_mermaid
export demo_replica_exchange

using Printf

"""
    Replica

A single replica in the parallel tempering ensemble.
Tracks its own state, energy, and exchange history.
"""
mutable struct Replica{S}
    ctx::GayMCContext           # Colored MC context (owns RNG)
    state::S                    # Current configuration
    energy::Float64             # Current energy E(state)
    β::Float64                  # Inverse temperature 1/T
    index::Int                  # Position in temperature ladder
    
    # Statistics
    sweep_count::Int
    swap_attempts::Int
    swap_accepts::Int
    swap_history::Vector{Tuple{Int, Bool}}  # (partner_index, accepted)
end

function Replica(ctx::GayMCContext, state::S, energy::Float64, β::Float64, index::Int) where S
    Replica{S}(ctx, state, energy, β, index, 0, 0, 0, Tuple{Int,Bool}[])
end

"""Acceptance rate for replica exchanges."""
swap_rate(r::Replica) = r.swap_attempts > 0 ? r.swap_accepts / r.swap_attempts : 0.0

"""
    TemperatureLadder

Defines the temperature schedule for parallel tempering.
Supports geometric, linear, and custom spacing.
"""
struct TemperatureLadder
    βs::Vector{Float64}         # Inverse temperatures (sorted high to low T)
    Ts::Vector{Float64}         # Temperatures
    n_replicas::Int
end

"""Geometric temperature ladder from T_min to T_max."""
function TemperatureLadder(T_min::Float64, T_max::Float64, n::Int; spacing::Symbol=:geometric)
    if spacing == :geometric
        ratio = (T_max / T_min)^(1/(n-1))
        Ts = [T_min * ratio^(i-1) for i in 1:n]
    elseif spacing == :linear
        Ts = range(T_min, T_max, length=n) |> collect
    else
        error("Unknown spacing: $spacing")
    end
    βs = 1.0 ./ Ts
    TemperatureLadder(βs, Ts, n)
end

"""Custom temperature ladder."""
function TemperatureLadder(temperatures::Vector{Float64})
    Ts = sort(temperatures)
    βs = 1.0 ./ Ts
    TemperatureLadder(βs, Ts, length(Ts))
end

"""
    ReplicaExchange{S}

The full replica exchange ensemble with SPI-guaranteed swaps.
"""
mutable struct ReplicaExchange{S}
    replicas::Vector{Replica{S}}
    ladder::TemperatureLadder
    seed::UInt64
    
    # Exchange RNG (independent of replica RNGs for SPI)
    exchange_rng::SplittableRandom
    
    # Global statistics
    total_sweeps::Int
    total_exchanges::Int
    exchange_schedule::Symbol  # :sequential, :random, :even_odd
end

function ReplicaExchange(
    seed::Integer,
    ladder::TemperatureLadder,
    init_state::S,
    energy_fn::Function;
    exchange_schedule::Symbol=:even_odd
) where S
    n = ladder.n_replicas
    
    # Create replicas with independent RNG streams
    replicas = Vector{Replica{S}}(undef, n)
    for i in 1:n
        ctx = GayMCContext(seed; worker_id=i-1)
        state = deepcopy(init_state)
        E = energy_fn(state)
        replicas[i] = Replica(ctx, state, E, ladder.βs[i], i)
    end
    
    # Separate RNG for exchange decisions (SPI: swaps don't affect replica streams)
    exchange_rng = SplittableRandom(UInt64(seed) ⊻ 0xDEADBEEF_CAFEBABE)
    
    ReplicaExchange{S}(replicas, ladder, UInt64(seed), exchange_rng, 0, 0, exchange_schedule)
end

# ═══════════════════════════════════════════════════════════════════════════
# Temperature Colors - Thermodynamic Gradient Mapping
# ═══════════════════════════════════════════════════════════════════════════

"""
    temperature_color(β, β_min, β_max) -> RGB

Map inverse temperature to thermodynamic gradient:
  Hot (low β) → Red/Orange
  Cold (high β) → Blue/Purple
"""
function temperature_color(β::Float64, β_min::Float64, β_max::Float64)::RGB{Float64}
    t = clamp((β - β_min) / (β_max - β_min + 1e-10), 0, 1)
    
    # Hot → Cold: Red → Orange → Yellow → Cyan → Blue → Purple
    if t < 0.2
        # Red to Orange
        s = t / 0.2
        RGB{Float64}(1.0, 0.3 * s, 0.0)
    elseif t < 0.4
        # Orange to Yellow
        s = (t - 0.2) / 0.2
        RGB{Float64}(1.0, 0.3 + 0.7 * s, 0.0)
    elseif t < 0.6
        # Yellow to Cyan
        s = (t - 0.4) / 0.2
        RGB{Float64}(1.0 - s, 1.0, s)
    elseif t < 0.8
        # Cyan to Blue
        s = (t - 0.6) / 0.2
        RGB{Float64}(0.0, 1.0 - s, 1.0)
    else
        # Blue to Purple
        s = (t - 0.8) / 0.2
        RGB{Float64}(0.5 * s, 0.0, 1.0)
    end
end

"""Color for a replica based on its temperature."""
temperature_color(r::Replica, ladder::TemperatureLadder) = 
    temperature_color(r.β, minimum(ladder.βs), maximum(ladder.βs))

"""
    energy_color(E, E_min, E_max) -> RGB

Map energy to color: low energy → green, high energy → red.
"""
function energy_color(E::Float64, E_min::Float64, E_max::Float64)::RGB{Float64}
    t = clamp((E - E_min) / (E_max - E_min + 1e-10), 0, 1)
    RGB{Float64}(t, 1.0 - t, 0.2)
end

# ═══════════════════════════════════════════════════════════════════════════
# Replica Exchange Protocol with SPI
# ═══════════════════════════════════════════════════════════════════════════

"""
    attempt_swap!(re::ReplicaExchange, i::Int, j::Int) -> Bool

Attempt to swap replicas i and j using Metropolis criterion.
Returns true if swap accepted.

SPI Guarantee: Uses exchange_rng, not replica RNGs.
"""
function attempt_swap!(re::ReplicaExchange, i::Int, j::Int)
    ri, rj = re.replicas[i], re.replicas[j]
    
    # Metropolis criterion for replica exchange
    # Accept with probability min(1, exp(Δβ * ΔE))
    Δβ = ri.β - rj.β
    ΔE = ri.energy - rj.energy
    log_accept = Δβ * ΔE
    
    # Split exchange RNG (SPI: deterministic regardless of sweep order)
    re.exchange_rng = split(re.exchange_rng)
    u = rand(re.exchange_rng)
    
    accepted = log_accept >= 0 || u < exp(log_accept)
    
    # Update statistics
    ri.swap_attempts += 1
    rj.swap_attempts += 1
    push!(ri.swap_history, (j, accepted))
    push!(rj.swap_history, (i, accepted))
    
    if accepted
        ri.swap_accepts += 1
        rj.swap_accepts += 1
        re.total_exchanges += 1
        swap_replicas!(re, i, j)
    end
    
    accepted
end

"""
    swap_replicas!(re::ReplicaExchange, i::Int, j::Int)

Exchange temperatures between replicas i and j.
States stay in place; temperatures move.
"""
function swap_replicas!(re::ReplicaExchange, i::Int, j::Int)
    ri, rj = re.replicas[i], re.replicas[j]
    
    # Swap inverse temperatures
    ri.β, rj.β = rj.β, ri.β
    
    # Swap ladder indices
    ri.index, rj.index = rj.index, ri.index
end

"""
    replica_exchange!(re::ReplicaExchange, sweep_fn!::Function, energy_fn::Function; n_sweeps::Int=1)

Perform n_sweeps MC sweeps on all replicas, then attempt exchanges.

SPI Guarantee: 
  1. Each replica's RNG stream is independent
  2. Exchange decisions use separate RNG
  3. Results identical regardless of parallel execution order

# Arguments
- `sweep_fn!(state, ctx, β)`: Performs one MC sweep, modifies state in place
- `energy_fn(state)`: Returns energy of configuration
"""
function replica_exchange!(
    re::ReplicaExchange{S},
    sweep_fn!::Function,
    energy_fn::Function;
    n_sweeps::Int=1
) where S
    # Phase 1: Independent sweeps on all replicas (parallelizable)
    for r in re.replicas
        for _ in 1:n_sweeps
            rng, color = gay_sweep!(r.ctx)
            sweep_fn!(r.state, r.ctx, r.β)
            r.sweep_count += 1
        end
        r.energy = energy_fn(r.state)
    end
    re.total_sweeps += n_sweeps
    
    # Phase 2: Replica exchanges (sequential, but SPI-deterministic)
    n = length(re.replicas)
    
    if re.exchange_schedule == :even_odd
        # Even-odd scheme: alternate between (0,1),(2,3),... and (1,2),(3,4),...
        parity = re.total_sweeps % 2
        for i in (1+parity):2:(n-1)
            attempt_swap!(re, i, i+1)
        end
    elseif re.exchange_schedule == :sequential
        # Sequential: try all adjacent pairs
        for i in 1:(n-1)
            attempt_swap!(re, i, i+1)
        end
    elseif re.exchange_schedule == :random
        # Random pair selection
        re.exchange_rng = split(re.exchange_rng)
        i = 1 + floor(Int, rand(re.exchange_rng) * (n-1))
        attempt_swap!(re, i, i+1)
    end
    
    re
end

# ═══════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════

"""
    visualize_ladder(re::ReplicaExchange) -> String

ASCII visualization of the temperature ladder with replica positions.
"""
function visualize_ladder(re::ReplicaExchange)
    n = length(re.replicas)
    ladder = re.ladder
    
    lines = String[]
    push!(lines, "Temperature Ladder ($(n) replicas)")
    push!(lines, "─" ^ 50)
    
    # Sort replicas by current β
    sorted = sort(re.replicas, by=r->r.β)
    
    for (i, r) in enumerate(sorted)
        T = 1.0 / r.β
        color = temperature_color(r.β, minimum(ladder.βs), maximum(ladder.βs))
        rate = swap_rate(r)
        
        # Color indicator (ANSI would be nice, but keep it portable)
        bar_len = round(Int, 20 * (1 - (r.β - minimum(ladder.βs)) / (maximum(ladder.βs) - minimum(ladder.βs) + 1e-10)))
        bar = "█" ^ bar_len * "░" ^ (20 - bar_len)
        
        push!(lines, @sprintf("β=%.3f (T=%.2f) |%s| E=%.2f swap=%.1f%%", 
                              r.β, T, bar, r.energy, 100*rate))
    end
    
    push!(lines, "─" ^ 50)
    push!(lines, @sprintf("Total sweeps: %d, exchanges: %d", re.total_sweeps, re.total_exchanges))
    
    join(lines, "\n")
end

"""
    ladder_to_mermaid(re::ReplicaExchange) -> String

Generate Mermaid diagram of the temperature ladder.
"""
function ladder_to_mermaid(re::ReplicaExchange)
    n = length(re.replicas)
    ladder = re.ladder
    
    lines = String[]
    push!(lines, "graph TD")
    
    # Sort replicas by β
    sorted = sort(re.replicas, by=r->r.β)
    
    # Nodes for each temperature level
    for (i, r) in enumerate(sorted)
        T = 1.0 / r.β
        color = temperature_color(r.β, minimum(ladder.βs), maximum(ladder.βs))
        hex = @sprintf("#%02X%02X%02X", 
                       round(Int, 255*red(color)),
                       round(Int, 255*green(color)),
                       round(Int, 255*blue(color)))
        push!(lines, "    T$i[\"T=$(round(T, digits=2))<br>E=$(round(r.energy, digits=1))\"]")
    end
    
    # Edges showing exchange paths
    push!(lines, "")
    for i in 1:(n-1)
        r1, r2 = sorted[i], sorted[i+1]
        # Find if they recently swapped
        recent_swap = !isempty(r1.swap_history) && r1.swap_history[end][1] == r2.index
        style = recent_swap ? "==>" : "-->"
        push!(lines, "    T$i $style T$(i+1)")
    end
    
    # Style definitions
    push!(lines, "")
    for (i, r) in enumerate(sorted)
        color = temperature_color(r.β, minimum(ladder.βs), maximum(ladder.βs))
        hex = @sprintf("#%02X%02X%02X", 
                       round(Int, 255*red(color)),
                       round(Int, 255*green(color)),
                       round(Int, 255*blue(color)))
        push!(lines, "    style T$i fill:$hex,stroke:#333,color:#fff")
    end
    
    join(lines, "\n")
end

# ═══════════════════════════════════════════════════════════════════════════
# Demo: Double-Well Potential
# ═══════════════════════════════════════════════════════════════════════════

"""
    demo_replica_exchange(; n_sweeps=1000, n_replicas=8)

Demonstrate replica exchange on a 1D double-well potential.
The system should tunnel between wells via high-temperature replicas.
"""
function demo_replica_exchange(; n_sweeps::Int=1000, n_replicas::Int=8, seed::Int=42)
    # Double-well potential: V(x) = (x² - 1)²
    energy(x::Float64) = (x^2 - 1)^2
    
    # Metropolis sweep with Gaussian proposals
    function sweep!(x::Vector{Float64}, ctx::GayMCContext, β::Float64)
        for i in eachindex(x)
            # Propose
            rng = ctx.current_rng
            proposal = x[i] + 0.5 * randn(rng)
            
            # Accept/reject
            ΔE = energy(proposal) - energy(x[i])
            if ΔE <= 0 || rand(rng) < exp(-β * ΔE)
                x[i] = proposal
            end
        end
    end
    
    # Energy of full state
    total_energy(x::Vector{Float64}) = sum(energy, x)
    
    # Setup
    ladder = TemperatureLadder(0.1, 10.0, n_replicas)
    init_state = [-1.0]  # Start in left well
    
    re = ReplicaExchange(seed, ladder, init_state, total_energy)
    
    println("Initial state:")
    println(visualize_ladder(re))
    println()
    
    # Run simulation
    for epoch in 1:10
        for _ in 1:(n_sweeps ÷ 10)
            replica_exchange!(re, sweep!, total_energy)
        end
        println("After $(epoch * n_sweeps ÷ 10) sweeps:")
        println(visualize_ladder(re))
        println()
    end
    
    # Final Mermaid diagram
    println("Mermaid diagram:")
    println(ladder_to_mermaid(re))
    
    re
end
