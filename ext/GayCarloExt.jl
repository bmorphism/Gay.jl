# Carlo.jl extension for Gay.jl
# Deterministic SPI-compliant coloring for parallel Monte Carlo
# Compatible with Carlo.jl's AbstractMC interface (arXiv:2408.03386)

module GayCarloExt

using Gay: GayMCContext, gay_sweep!, gay_measure!, gay_checkpoint, gay_restore!
using Gay: color_sweep, color_measure, color_state
using Gay: gay_metropolis!, gay_gaussian!, gay_exponential!, gay_cauchy!
using Gay: fnv1a, color_at_index, GAY_SEED
using Carlo
using Carlo: AbstractMC, MCContext, AbstractEvaluator
using Colors: RGB
using HDF5

export GayMC, ColoredObservable
export gay_carlo_init!, gay_carlo_sweep!, gay_carlo_measure!
export color_observable, color_trajectory

# ═══════════════════════════════════════════════════════════════════════════
# GayMC - Colored Monte Carlo implementing Carlo.jl AbstractMC
# ═══════════════════════════════════════════════════════════════════════════

"""
    GayMC{Model} <: AbstractMC

A colored Monte Carlo wrapper that provides:
- SPI (Strong Parallelism Invariance) via SplittableRandoms
- Deterministic colors for every sweep/measure/checkpoint
- HDF5-compatible checkpointing with color metadata
- Integration with Carlo.jl's MPI-parallel scheduler

# Type Parameters
- `Model`: The underlying physics model (e.g., Ising, Hubbard)

# Example
```julia
struct IsingModel
    L::Int
    spins::Matrix{Int8}
    β::Float64
end

mc = GayMC{IsingModel}(model, seed=1069)
```
"""
mutable struct GayMC{Model} <: AbstractMC
    # Underlying physics model
    model::Model

    # Gay.jl colored context for SPI
    ctx::GayMCContext

    # Sweep function: (model, rng) -> ΔE
    sweep_fn::Function

    # Measure function: (model, ctx) -> Dict{String, Any}
    measure_fn::Function

    # Color history for trajectory visualization
    sweep_colors::Vector{RGB{Float64}}
    measure_colors::Vector{RGB{Float64}}

    # Statistics
    accepted::Int
    rejected::Int
end

"""
    GayMC(model; seed=1069, worker_id=0, sweep_fn, measure_fn)

Create a colored Monte Carlo wrapper for Carlo.jl integration.

# Arguments
- `model`: The physics model to simulate
- `seed`: RNG seed for deterministic colors (default: 1069)
- `worker_id`: MPI rank for parallel runs
- `sweep_fn`: Function `(model, rng) -> ΔE` for configuration updates
- `measure_fn`: Function `(model, ctx) -> Dict` for measurements
"""
function GayMC(model::Model;
               seed::Integer=1069,
               worker_id::Integer=0,
               sweep_fn::Function=default_sweep,
               measure_fn::Function=default_measure) where {Model}
    ctx = GayMCContext(seed; worker_id=worker_id)
    GayMC{Model}(
        model,
        ctx,
        sweep_fn,
        measure_fn,
        RGB{Float64}[],
        RGB{Float64}[],
        0,
        0
    )
end

# Default implementations
default_sweep(model, rng) = 0.0
default_measure(model, ctx) = Dict{String,Any}()

# ═══════════════════════════════════════════════════════════════════════════
# Carlo.jl AbstractMC Interface Implementation
# ═══════════════════════════════════════════════════════════════════════════

"""
    Carlo.init!(mc::GayMC, ctx::MCContext, params::AbstractDict)

Initialize the colored Monte Carlo simulation.
Called when starting a simulation from scratch.
"""
function Carlo.init!(mc::GayMC, ctx::MCContext, params::AbstractDict)
    # Extract parameters
    seed = get(params, "seed", 1069)
    worker_id = get(params, "worker_id", ctx.rank)

    # Reinitialize Gay context with proper seed
    mc.ctx = GayMCContext(seed; worker_id=worker_id)

    # Clear statistics
    mc.accepted = 0
    mc.rejected = 0
    empty!(mc.sweep_colors)
    empty!(mc.measure_colors)

    # Model-specific initialization if available
    if hasmethod(init_model!, Tuple{typeof(mc.model), AbstractDict})
        init_model!(mc.model, params)
    end

    nothing
end

"""
    Carlo.sweep!(mc::GayMC, ctx::MCContext)

Perform one colored Monte Carlo sweep.
Each sweep gets a unique, reproducible color.
"""
function Carlo.sweep!(mc::GayMC, ctx::MCContext)
    # Get colored RNG for this sweep
    rng, color = gay_sweep!(mc.ctx)
    push!(mc.sweep_colors, color)

    # Perform model update with colored RNG
    ΔE = mc.sweep_fn(mc.model, rng)

    # Metropolis acceptance with colored decision
    β = hasfield(typeof(mc.model), :β) ? mc.model.β : 1.0
    accepted, _ = gay_metropolis!(mc.ctx, ΔE, β)

    if accepted
        mc.accepted += 1
    else
        mc.rejected += 1
    end

    nothing
end

"""
    Carlo.measure!(mc::GayMC, ctx::MCContext)

Perform one colored measurement.
Only called after thermalization (when `is_thermalized(ctx)` is true).
"""
function Carlo.measure!(mc::GayMC, ctx::MCContext)
    # Get colored RNG for this measurement
    rng, color = gay_measure!(mc.ctx, "observable")
    push!(mc.measure_colors, color)

    # Collect observables from model
    observables = mc.measure_fn(mc.model, mc.ctx)

    # Record to Carlo's measurement system with color metadata
    for (name, value) in observables
        measure!(ctx, name, value)
    end

    # Record acceptance ratio
    total = mc.accepted + mc.rejected
    if total > 0
        measure!(ctx, "acceptance_ratio", mc.accepted / total)
    end

    # Record state color (XOR of recent sweep colors)
    state_color = color_state(mc.ctx; window=10)
    measure!(ctx, "state_color_r", Float64(state_color.r))
    measure!(ctx, "state_color_g", Float64(state_color.g))
    measure!(ctx, "state_color_b", Float64(state_color.b))

    nothing
end

"""
    Carlo.write_checkpoint(mc::GayMC, out::HDF5.Group)

Save simulation state with color metadata for HDF5 storage.
"""
function Carlo.write_checkpoint(mc::GayMC, out::HDF5.Group)
    # Save Gay context checkpoint
    gay_ckpt = gay_checkpoint(mc.ctx)
    for (key, value) in gay_ckpt
        out[string("gay_", key)] = value
    end

    # Save statistics
    out["accepted"] = mc.accepted
    out["rejected"] = mc.rejected

    # Save color histories
    out["sweep_colors_r"] = [c.r for c in mc.sweep_colors]
    out["sweep_colors_g"] = [c.g for c in mc.sweep_colors]
    out["sweep_colors_b"] = [c.b for c in mc.sweep_colors]
    out["measure_colors_r"] = [c.r for c in mc.measure_colors]
    out["measure_colors_g"] = [c.g for c in mc.measure_colors]
    out["measure_colors_b"] = [c.b for c in mc.measure_colors]

    # Model-specific checkpoint if available
    if hasmethod(write_model_checkpoint, Tuple{typeof(mc.model), HDF5.Group})
        model_grp = create_group(out, "model")
        write_model_checkpoint(mc.model, model_grp)
    end

    nothing
end

"""
    Carlo.read_checkpoint!(mc::GayMC, in::HDF5.Group)

Restore simulation state from HDF5 checkpoint.
"""
function Carlo.read_checkpoint!(mc::GayMC, in::HDF5.Group)
    # Restore Gay context
    gay_ckpt = Dict{String,Any}()
    for key in ["seed", "worker_id", "sweep_count", "measure_count",
                "checkpoint_count", "color_history_r", "color_history_g", "color_history_b"]
        if haskey(in, "gay_$key")
            gay_ckpt[key] = read(in, "gay_$key")
        end
    end
    gay_restore!(mc.ctx, gay_ckpt)

    # Restore statistics
    mc.accepted = read(in, "accepted")
    mc.rejected = read(in, "rejected")

    # Restore color histories
    if haskey(in, "sweep_colors_r")
        r = read(in, "sweep_colors_r")
        g = read(in, "sweep_colors_g")
        b = read(in, "sweep_colors_b")
        mc.sweep_colors = [RGB{Float64}(r[i], g[i], b[i]) for i in eachindex(r)]
    end
    if haskey(in, "measure_colors_r")
        r = read(in, "measure_colors_r")
        g = read(in, "measure_colors_g")
        b = read(in, "measure_colors_b")
        mc.measure_colors = [RGB{Float64}(r[i], g[i], b[i]) for i in eachindex(r)]
    end

    # Model-specific restore if available
    if haskey(in, "model") && hasmethod(read_model_checkpoint!, Tuple{typeof(mc.model), HDF5.Group})
        read_model_checkpoint!(mc.model, in["model"])
    end

    nothing
end

"""
    Carlo.register_evaluables(::Type{GayMC{M}}, eval::AbstractEvaluator, params::AbstractDict)

Register derived quantities for postprocessing.
"""
function Carlo.register_evaluables(::Type{GayMC{M}}, eval::AbstractEvaluator, params::AbstractDict) where {M}
    # Standard observables
    evaluate!(eval, :acceptance_mean, Mean(:acceptance_ratio))

    # Color-derived observables
    evaluate!(eval, :color_variance_r, Variance(:state_color_r))
    evaluate!(eval, :color_variance_g, Variance(:state_color_g))
    evaluate!(eval, :color_variance_b, Variance(:state_color_b))

    # Model-specific evaluables if defined
    if hasmethod(register_model_evaluables, Tuple{Type{M}, AbstractEvaluator, AbstractDict})
        register_model_evaluables(M, eval, params)
    end

    nothing
end

# ═══════════════════════════════════════════════════════════════════════════
# Parallel Tempering Support
# ═══════════════════════════════════════════════════════════════════════════

"""
    Carlo.parallel_tempering_log_weight_ratio(mc::GayMC, parameter::Symbol, new_value)

Compute log weight ratio for parallel tempering replica exchange.
"""
function Carlo.parallel_tempering_log_weight_ratio(mc::GayMC, parameter::Symbol, new_value)
    if parameter == :β && hasfield(typeof(mc.model), :β) && hasmethod(energy, Tuple{typeof(mc.model)})
        E = energy(mc.model)
        old_β = mc.model.β
        return (new_value - old_β) * E
    end
    return 0.0
end

"""
    Carlo.parallel_tempering_change_parameter!(mc::GayMC, parameter::Symbol, new_value)

Change a parameter for parallel tempering.
"""
function Carlo.parallel_tempering_change_parameter!(mc::GayMC, parameter::Symbol, new_value)
    if parameter == :β && hasfield(typeof(mc.model), :β)
        # Update model's inverse temperature
        setfield!(mc.model, :β, new_value)
    end
    nothing
end

# ═══════════════════════════════════════════════════════════════════════════
# Colored Observable Wrapper
# ═══════════════════════════════════════════════════════════════════════════

"""
    ColoredObservable{T}

An observable with an associated deterministic color.
"""
struct ColoredObservable{T}
    name::String
    value::T
    color::RGB{Float64}
    sweep_index::UInt64
end

"""
    color_observable(ctx::GayMCContext, name::String, value) -> ColoredObservable

Create a colored observable from a Gay context.
"""
function color_observable(ctx::GayMCContext, name::String, value::T) where {T}
    color = color_measure(ctx, name)
    ColoredObservable{T}(name, value, color, ctx.sweep_count)
end

"""
    color_trajectory(mc::GayMC) -> NamedTuple

Get the color trajectory of the simulation.
"""
function color_trajectory(mc::GayMC)
    (
        sweep_colors = mc.sweep_colors,
        measure_colors = mc.measure_colors,
        state_color = color_state(mc.ctx),
        n_sweeps = mc.ctx.sweep_count,
        n_measures = mc.ctx.measure_count,
        acceptance_ratio = mc.accepted / max(1, mc.accepted + mc.rejected)
    )
end

# ═══════════════════════════════════════════════════════════════════════════
# Convenience Constructors for Common Models
# ═══════════════════════════════════════════════════════════════════════════

"""
    gay_ising(L::Int; β=1.0, seed=1069, worker_id=0)

Create a colored Ising model MC for Carlo.jl.
Requires an Ising model implementation with `flip_spin!` and `magnetization`.
"""
function gay_ising end  # Placeholder for Ising extension

"""
    gay_heisenberg(L::Int; β=1.0, seed=1069, worker_id=0)

Create a colored Heisenberg model MC for Carlo.jl.
"""
function gay_heisenberg end  # Placeholder for Heisenberg extension

# ═══════════════════════════════════════════════════════════════════════════
# Module Initialization
# ═══════════════════════════════════════════════════════════════════════════

function __init__()
    @info "Gay.jl Carlo extension loaded - SPI-compliant colored Monte Carlo"
    @info "  AbstractMC interface: GayMC{Model}"
    @info "  Parallel tempering: supported via β parameter"
    @info "  Checkpointing: HDF5 with color metadata"
end

end # module GayCarloExt
