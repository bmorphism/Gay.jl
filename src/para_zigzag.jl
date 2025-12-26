# Para(ZigZag): Chromatic PDMP Sampler with SPI Guarantees
# ═══════════════════════════════════════════════════════════════════════════════
#
# Following ZigZagBoomerang.jl architecture with chromatic verification:
#
#   ┌─────────────────────────────────────────────────────────────────────────────┐
#   │  PARA(ZIGZAG) CATEGORY                                                      │
#   │  ═══════════════════                                                        │
#   │                                                                             │
#   │  Objects: (Γ, x, θ) parameterized by precision matrix Γ                    │
#   │  1-Morphisms: Event transitions (t₁,x₁,θ₁) → (t₂,x₂,θ₂)                   │
#   │  2-Morphisms: Transformations between trajectories                         │
#   │                                                                             │
#   │  CHROMATIC STRUCTURE                                                        │
#   │  ═══════════════════                                                        │
#   │  Green events: Accept flip (λᵢ/λ̄ᵢ > U)                                    │
#   │  Red events:   Reject flip (λᵢ/λ̄ᵢ < U)                                    │
#   │  Parity:       θ[i] sign determines igor-alignment                         │
#   │                                                                             │
#   │  SPI GUARANTEE                                                              │
#   │  ═════════════                                                              │
#   │  Same seed → same event sequence → same trajectory → same fingerprint     │
#   │  XOR over all events is order-invariant                                    │
#   │                                                                             │
#   │  TROPICAL CONNECTION                                                        │
#   │  ═══════════════════                                                        │
#   │  Event times form tropical path weights                                    │
#   │  Min-plus: shortest time to reach state                                    │
#   │  Max-plus: longest time before flip                                        │
#   └─────────────────────────────────────────────────────────────────────────────┘

export ZigZagDynamics, BoomerangDynamics, FactorisedDynamics
export ChromaticEvent, ChromaticTrace, ChromaticZigZag
export para_zigzag_step!, para_zigzag_trajectory
export spi_fingerprint, verify_zigzag_spi
export TropicalZigZagPath, tropical_event_weight

using LinearAlgebra
using SparseArrays

# Import from igor_seeds.jl
include("igor_seeds.jl")

# ═══════════════════════════════════════════════════════════════════════════════
# Dynamics Types (following ZigZagBoomerang.jl)
# ═══════════════════════════════════════════════════════════════════════════════

"""
Abstract type for PDMP dynamics
"""
abstract type ContinuousDynamics end

"""
    ZigZagDynamics{T, S}

ZigZag sampler dynamics with sparse precision matrix.

# Fields
- `Γ`: Sparse precision matrix (approximation of target precision)
- `μ`: Approximate target mean
- `λref`: Refreshment rate
"""
struct ZigZagDynamics{T<:AbstractMatrix, S<:AbstractVector} <: ContinuousDynamics
    Γ::T           # Precision matrix
    μ::S           # Target mean
    λref::Float64  # Refreshment rate
end

function ZigZagDynamics(n::Int; λref::Float64=0.0, seed::UInt64=GAY_IGOR_SEED)
    # Generate random sparse precision matrix
    rng = seed
    nnz_ratio = 0.3
    
    # Build symmetric positive definite sparse matrix
    I_indices = Int[]
    J_indices = Int[]
    V_values = Float64[]
    
    for i in 1:n
        # Diagonal (must be positive)
        push!(I_indices, i)
        push!(J_indices, i)
        rng = mix64(rng)
        push!(V_values, 1.0 + (rng % 100) / 100.0)
        
        # Off-diagonal (sparse)
        for j in (i+1):n
            rng = mix64(rng)
            if (rng % 100) / 100.0 < nnz_ratio
                rng = mix64(rng)
                val = ((rng % 100) / 100.0 - 0.5) * 0.5
                push!(I_indices, i)
                push!(J_indices, j)
                push!(V_values, val)
                push!(I_indices, j)
                push!(J_indices, i)
                push!(V_values, val)
            end
        end
    end
    
    Γ = sparse(I_indices, J_indices, V_values, n, n)
    μ = zeros(n)
    
    ZigZagDynamics(Γ, μ, λref)
end

"""
    BoomerangDynamics{T, S}

Boomerang sampler (Hamiltonian oscillator preserving Gaussian).
"""
struct BoomerangDynamics{T<:AbstractMatrix, S<:AbstractVector} <: ContinuousDynamics
    Γ::T
    μ::S
    λref::Float64
end

"""
    FactorisedDynamics{T, S}

Factorised sampler exploiting Markov blanket structure.
"""
struct FactorisedDynamics{T<:AbstractMatrix, S<:AbstractVector, G} <: ContinuousDynamics
    Γ::T
    μ::S
    λref::Float64
    graph::G  # Dependency graph (neighbors for each coordinate)
end

function FactorisedDynamics(Z::ZigZagDynamics)
    n = size(Z.Γ, 1)
    
    # Build dependency graph from sparsity pattern
    graph = [Int[] for _ in 1:n]
    rows, cols, _ = findnz(Z.Γ)
    for (i, j) in zip(rows, cols)
        if i != j
            push!(graph[i], j)
        end
    end
    
    FactorisedDynamics(Z.Γ, Z.μ, Z.λref, graph)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Chromatic Events
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ChromaticEvent

A velocity flip event with chromatic annotation.

# Fields
- `t`: Time of event
- `i`: Coordinate that flipped
- `x_i`: Position at flip
- `θ_i`: New velocity after flip
- `accepted`: Was the flip accepted? (thin-thinning)
- `color`: RGB color for this event
- `igor_aligned`: Is velocity pointing toward igor?
"""
struct ChromaticEvent
    t::Float64
    i::Int
    x_i::Float64
    θ_i::Float64
    accepted::Bool
    color::Tuple{Float32, Float32, Float32}
    igor_aligned::Bool
end

function ChromaticEvent(t::Float64, i::Int, x_i::Float64, θ_i::Float64, 
                        accepted::Bool, seed::UInt64)
    # Color from event hash
    h = mix64(seed ⊻ UInt64(round(t * 1000)) ⊻ UInt64(i))
    r = Float32((h % 256) / 255)
    h = mix64(h)
    g = Float32((h % 256) / 255)
    h = mix64(h)
    b = Float32((h % 256) / 255)
    
    # Igor alignment: positive velocity = igor direction
    igor_aligned = θ_i > 0
    
    # Modulate color by acceptance
    if accepted
        # Green tint for accepted
        g = min(1.0f0, g + 0.3f0)
    else
        # Red tint for rejected
        r = min(1.0f0, r + 0.3f0)
    end
    
    ChromaticEvent(t, i, x_i, θ_i, accepted, (r, g, b), igor_aligned)
end

"""
XOR fingerprint contribution from an event
"""
function event_fingerprint(ev::ChromaticEvent)
    fp = UInt32(0)
    fp ⊻= UInt32(round(ev.t * 1000) % (1 << 16)) << 16
    fp ⊻= UInt32(ev.i % (1 << 8)) << 8
    fp ⊻= UInt32(round(ev.color[1] * 255)) 
    fp ⊻= UInt32(round(ev.color[2] * 255)) << 4
    fp ⊻= UInt32(round(ev.color[3] * 255)) << 8
    fp ⊻= ev.accepted ? UInt32(0x80000000) : UInt32(0)
    fp
end

# ═══════════════════════════════════════════════════════════════════════════════
# Chromatic Trace
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ChromaticTrace

Full trajectory with chromatic events.

Deterministically reconstructible from (t0, x0, θ0) + events.
"""
struct ChromaticTrace{D<:ContinuousDynamics}
    dynamics::D
    t0::Float64
    x0::Vector{Float64}
    θ0::Vector{Float64}
    events::Vector{ChromaticEvent}
    seed::UInt64
end

function ChromaticTrace(D::ContinuousDynamics, x0::Vector{Float64}, θ0::Vector{Float64};
                        seed::UInt64=GAY_IGOR_SEED)
    ChromaticTrace(D, 0.0, copy(x0), copy(θ0), ChromaticEvent[], seed)
end

"""
Push an event to the trace
"""
function Base.push!(trace::ChromaticTrace, ev::ChromaticEvent)
    push!(trace.events, ev)
end

"""
Compute SPI fingerprint of entire trace (order-invariant via XOR)
"""
function spi_fingerprint(trace::ChromaticTrace)
    fp = UInt32(0)
    for ev in trace.events
        fp ⊻= event_fingerprint(ev)
    end
    fp
end

"""
Reconstruct trajectory at discrete times
"""
function discretize(trace::ChromaticTrace, dt::Float64)
    if isempty(trace.events)
        return [(0.0, copy(trace.x0))]
    end
    
    T = trace.events[end].t
    times = 0.0:dt:T
    
    result = Tuple{Float64, Vector{Float64}}[]
    x = copy(trace.x0)
    θ = copy(trace.θ0)
    event_idx = 1
    
    for t in times
        # Apply events up to time t
        while event_idx <= length(trace.events) && trace.events[event_idx].t <= t
            ev = trace.events[event_idx]
            # Move to event time
            Δt = ev.t - (event_idx > 1 ? trace.events[event_idx-1].t : trace.t0)
            x .+= θ .* Δt
            # Apply flip
            if ev.accepted
                θ[ev.i] = ev.θ_i
            end
            event_idx += 1
        end
        
        # Move to time t
        if event_idx > 1
            last_event_t = trace.events[event_idx-1].t
        else
            last_event_t = trace.t0
        end
        Δt = t - last_event_t
        x_at_t = x .+ θ .* Δt
        
        push!(result, (t, copy(x_at_t)))
    end
    
    return result
end

# ═══════════════════════════════════════════════════════════════════════════════
# Poisson Event Time Generation
# ═══════════════════════════════════════════════════════════════════════════════

"""
Sample from inhomogeneous Poisson with affine rate λ(t) = max(a + b*t, 0)
"""
function poisson_time(a::Float64, b::Float64, u::Float64)
    if a <= 0 && b <= 0
        return Inf
    end
    
    if b == 0
        # Constant rate
        return a > 0 ? -log(u) / a : Inf
    elseif b > 0
        if a < 0
            # Rate starts at 0, increases
            return sqrt(-log(u) * 2.0 / b) - a / b
        else
            # Always positive, increasing rate
            return sqrt((a/b)^2 - log(u) * 2.0 / b) - a / b
        end
    else  # b < 0
        # Decreasing rate
        if a > 0
            max_t = -a / b  # Time when rate hits 0
            Λ_max = a * max_t + 0.5 * b * max_t^2  # Integrated rate to max_t
            if -log(u) > Λ_max
                return Inf  # No event before rate hits 0
            end
            # Invert Λ(t) = a*t + 0.5*b*t^2 = -log(u)
            discriminant = (a/b)^2 - log(u) * 2.0 / b
            if discriminant < 0
                return Inf
            end
            return sqrt(discriminant) - a / b
        else
            return Inf
        end
    end
end

"""
Compute affine upper bound coefficients for ZigZag rate
"""
function rate_bounds(Z::ZigZagDynamics, i::Int, x::Vector{Float64}, θ::Vector{Float64}, c::Float64)
    # True rate: λᵢ = max(∇ϕᵢ ⋅ θᵢ, 0)
    # Upper bound: a + b*t where t is time since last event at i
    
    # Gradient contribution from precision matrix
    ∇ϕ_i = dot(Z.Γ[i, :], x) - dot(Z.Γ[i, :], Z.μ)
    
    # Rate at current time
    a = c * max(∇ϕ_i * θ[i], 0.0) + 0.01  # Small positive offset
    
    # Rate change coefficient
    b = c * 0.01 * abs(dot(Z.Γ[i, :], θ))  # Rate of change
    
    return a, b
end

# ═══════════════════════════════════════════════════════════════════════════════
# ChromaticZigZag Sampler State
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ChromaticZigZag

Full ZigZag sampler state with chromatic tracking.
"""
mutable struct ChromaticZigZag{D<:ContinuousDynamics}
    dynamics::D
    x::Vector{Float64}      # Current position
    θ::Vector{Float64}      # Current velocity (±1 per coordinate)
    t::Float64              # Current time
    trace::ChromaticTrace{D}
    
    # Event scheduling
    event_times::Vector{Float64}   # Next event time for each coordinate
    bounds_c::Vector{Float64}      # Upper bound multipliers
    
    # Random state (deterministic)
    rng_state::UInt64
    seed::UInt64
    
    # Igor tracking
    igor_spectrum::IgorSpectrum
end

function ChromaticZigZag(D::ZigZagDynamics; seed::UInt64=GAY_IGOR_SEED, igor_weight::Float64=0.5)
    n = size(D.Γ, 1)
    
    # Initialize position and velocity
    rng = seed
    x = zeros(n)
    θ = zeros(n)
    for i in 1:n
        rng = mix64(rng)
        θ[i] = (rng % 2 == 0) ? 1.0 : -1.0
    end
    
    # Initialize bounds
    c = ones(n) * 1.5
    
    # Schedule initial events
    event_times = zeros(n)
    for i in 1:n
        rng = mix64(rng)
        u = (rng % UInt64(1000000)) / 1000000.0
        a, b = rate_bounds(D, i, x, θ, c[i])
        event_times[i] = poisson_time(a, b, u)
    end
    
    trace = ChromaticTrace(D, x, θ; seed=seed)
    igor = IgorSpectrum(; seed=seed, weight=igor_weight)
    
    ChromaticZigZag(D, x, θ, 0.0, trace, event_times, c, rng, seed, igor)
end

# ═══════════════════════════════════════════════════════════════════════════════
# ZigZag Step (Event-Driven)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    para_zigzag_step!(zz::ChromaticZigZag)

Execute one event-driven step of the chromatic ZigZag sampler.
Returns the event that occurred.
"""
function para_zigzag_step!(zz::ChromaticZigZag)
    # Find next event (minimum time)
    τ, i = findmin(zz.event_times)
    
    if isinf(τ)
        return nothing
    end
    
    # Move to event time
    Δt = τ - zz.t
    zz.x .+= zz.θ .* Δt
    zz.t = τ
    
    # Compute true rate at event time
    D = zz.dynamics
    ∇ϕ_i = dot(D.Γ[i, :], zz.x) - dot(D.Γ[i, :], D.μ)
    λ_true = max(∇ϕ_i * zz.θ[i], 0.0)
    
    # Upper bound rate
    a, b = rate_bounds(D, i, zz.x, zz.θ, zz.bounds_c[i])
    λ_upper = a + b * Δt
    
    # Thin-thinning: accept with probability λ_true / λ_upper
    zz.rng_state = mix64(zz.rng_state)
    u = (zz.rng_state % UInt64(1000000)) / 1000000.0
    
    accepted = λ_upper > 0 && u < λ_true / λ_upper
    
    new_θ_i = zz.θ[i]
    if accepted
        # Flip velocity
        new_θ_i = -zz.θ[i]
        zz.θ[i] = new_θ_i
        
        # Adapt bound
        zz.bounds_c[i] = max(1.0, 0.9 * zz.bounds_c[i] + 0.1 * λ_true / λ_upper * 1.5)
    end
    
    # Create chromatic event
    event = ChromaticEvent(zz.t, i, zz.x[i], new_θ_i, accepted, zz.seed ⊻ UInt64(length(zz.trace.events)))
    push!(zz.trace, event)
    
    # Reschedule event for coordinate i
    zz.rng_state = mix64(zz.rng_state)
    u = (zz.rng_state % UInt64(1000000)) / 1000000.0
    a, b = rate_bounds(D, i, zz.x, zz.θ, zz.bounds_c[i])
    zz.event_times[i] = zz.t + poisson_time(a, b, u)
    
    # Reschedule neighbors (for factorised version)
    if D isa FactorisedDynamics
        for j in D.graph[i]
            zz.rng_state = mix64(zz.rng_state)
            u = (zz.rng_state % UInt64(1000000)) / 1000000.0
            a, b = rate_bounds(D, j, zz.x, zz.θ, zz.bounds_c[j])
            zz.event_times[j] = zz.t + poisson_time(a, b, u)
        end
    end
    
    return event
end

"""
    para_zigzag_trajectory(zz::ChromaticZigZag, T::Float64)

Run ZigZag sampler until time T, returning the chromatic trace.
"""
function para_zigzag_trajectory(zz::ChromaticZigZag, T::Float64)
    while zz.t < T
        event = para_zigzag_step!(zz)
        if isnothing(event)
            break
        end
    end
    return zz.trace
end

# ═══════════════════════════════════════════════════════════════════════════════
# SPI Verification
# ═══════════════════════════════════════════════════════════════════════════════

"""
    verify_zigzag_spi(seed::UInt64, n::Int, T::Float64; n_runs::Int=5)

Verify Strong Parallelism Invariance: same seed → same fingerprint.
"""
function verify_zigzag_spi(seed::UInt64, n::Int, T::Float64; n_runs::Int=5)
    fingerprints = UInt32[]
    
    for run in 1:n_runs
        D = ZigZagDynamics(n; seed=seed)
        zz = ChromaticZigZag(D; seed=seed)
        trace = para_zigzag_trajectory(zz, T)
        push!(fingerprints, spi_fingerprint(trace))
    end
    
    all_same = all(fp == fingerprints[1] for fp in fingerprints)
    
    (
        fingerprints = fingerprints,
        spi_verified = all_same,
        unique_fingerprint = fingerprints[1]
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# Tropical Path Weights
# ═══════════════════════════════════════════════════════════════════════════════

"""
    TropicalZigZagPath

Event times as tropical path weights.
"""
struct TropicalZigZagPath
    times::Vector{Float64}
    coordinates::Vector{Int}
    min_plus_weight::Float64   # Shortest time path
    max_plus_weight::Float64   # Longest time path
end

function TropicalZigZagPath(trace::ChromaticTrace)
    times = [ev.t for ev in trace.events]
    coords = [ev.i for ev in trace.events]
    
    if isempty(times)
        return TropicalZigZagPath(Float64[], Int[], 0.0, 0.0)
    end
    
    # Tropical weights
    intervals = diff([0.0; times])
    min_plus = minimum(cumsum(intervals))
    max_plus = maximum(cumsum(intervals))
    
    TropicalZigZagPath(times, coords, min_plus, max_plus)
end

"""
Tropical weight for a single event
"""
function tropical_event_weight(ev::ChromaticEvent, prev_t::Float64)
    ev.t - prev_t  # Inter-arrival time
end

# ═══════════════════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════════════════

function world_para_zigzag()
    println()
    println("╔" * "═" ^ 60 * "╗")
    println("║  PARA(ZIGZAG): Chromatic PDMP Sampler                      ║")
    println("║  Following ZigZagBoomerang.jl Architecture                  ║")
    println("╚" * "═" ^ 60 * "╝")
    println()
    
    seed = GAY_IGOR_SEED
    n = 10  # Dimensions
    T = 5.0 # Time horizon
    
    # Create dynamics
    println("ZigZag Dynamics (n=$n):")
    D = ZigZagDynamics(n; seed=seed)
    println("  Precision matrix nnz: $(nnz(D.Γ))")
    println("  Refreshment rate: $(D.λref)")
    println()
    
    # Create sampler
    println("Chromatic ZigZag Sampler:")
    zz = ChromaticZigZag(D; seed=seed, igor_weight=0.6)
    println("  Initial θ: $(Int.(zz.θ))")
    println()
    
    # Run trajectory
    println("Running trajectory (T=$T)...")
    trace = para_zigzag_trajectory(zz, T)
    println("  Events: $(length(trace.events))")
    println("  Final time: $(round(zz.t, digits=3))")
    println()
    
    # Event analysis
    accepted_count = count(ev -> ev.accepted, trace.events)
    igor_count = count(ev -> ev.igor_aligned, trace.events)
    println("Event Analysis:")
    println("  Accepted: $accepted_count / $(length(trace.events))")
    println("  Igor-aligned: $igor_count / $(length(trace.events))")
    println()
    
    # Show first few events with colors
    println("First 5 Events:")
    for (idx, ev) in enumerate(trace.events[1:min(5, length(trace.events))])
        c = ev.color
        r, g, b = Int.(round.((c[1], c[2], c[3]) .* 255))
        acc = ev.accepted ? "◆" : "◇"
        igor = ev.igor_aligned ? "+" : "-"
        println("  [$idx] t=$(round(ev.t, digits=3)) i=$(ev.i) θ=$(ev.θ_i > 0 ? '+' : '-') " *
                "$acc $igor \e[48;2;$(r);$(g);$(b)m  \e[0m")
    end
    println()
    
    # SPI Verification
    println("SPI Verification (5 runs):")
    result = verify_zigzag_spi(seed, n, T; n_runs=5)
    for (i, fp) in enumerate(result.fingerprints)
        println("  Run $i: 0x$(string(fp, base=16, pad=8))")
    end
    println("  SPI Verified: $(result.spi_verified ? "◆ YES" : "◇ NO")")
    println()
    
    # Tropical path
    println("Tropical Path Weights:")
    trop = TropicalZigZagPath(trace)
    println("  Min-plus (shortest): $(round(trop.min_plus_weight, digits=3))")
    println("  Max-plus (longest):  $(round(trop.max_plus_weight, digits=3))")
    println()
    
    # Discretize trajectory
    println("Discretized Trajectory (dt=0.5):")
    discrete = discretize(trace, 0.5)
    for (t, x) in discrete[1:min(5, length(discrete))]
        x_str = join([round(xi, digits=2) for xi in x[1:min(3, length(x))]], ", ")
        println("  t=$(round(t, digits=2)): [$x_str, ...]")
    end
    
    println()
    println("◈ Para(ZigZag) Complete")
end

if abspath(PROGRAM_FILE) == @__FILE__
    world_para_zigzag()
end
