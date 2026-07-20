# Dynamic Sufficiency for Guaranteed Eventual Termination
# Metalearned hyperparameters adapt for ergodic and non-ergodic systems
#
# Reference: Marqov Blume-Capel implementation
# https://gitpages.physik.uni-wuerzburg.de/marqov/webmarqov/post/2020-05-15-blume-capel/

using Statistics

export DynamicSufficiency, MetalearnedParams
export estimate_mixing_time, is_ergodic, adapt_hyperparameters!
export run_with_termination_guarantee
export blume_capel_sweep!, wolff_cluster_update!

# ═══════════════════════════════════════════════════════════════════════════
# Metalearned Parameters
# ═══════════════════════════════════════════════════════════════════════════

"""
Metalearned hyperparameters that adapt for guaranteed termination.

For ergodic systems: bound = O(τ_mix log(1/ε))
For non-ergodic systems: bound = O(exp(ΔF/kT)) with early detection
"""
mutable struct MetalearnedParams
    β::Float64              # Inverse temperature
    D::Float64              # Crystal field (Blume-Capel)
    J::Float64              # Exchange coupling
    τ_mix::Float64          # Estimated mixing time
    ergodic::Bool           # Whether system is ergodic
    termination_bound::Int  # Guaranteed iteration bound
    adaptation_rate::Float64  # How fast to adapt
end

function MetalearnedParams(; 
    β=1.0, D=0.0, J=-1.0, 
    τ_mix=100.0, 
    termination_bound=10000
)
    MetalearnedParams(β, D, J, τ_mix, true, termination_bound, 0.01)
end

# ═══════════════════════════════════════════════════════════════════════════
# Mixing Time Estimation
# ═══════════════════════════════════════════════════════════════════════════

"""
Estimate mixing time τ from autocorrelation of observable.

Uses exponential decay fit: C(t) ∝ exp(-t/τ)
"""
function estimate_mixing_time(history::Vector{Float64}; max_lag::Int=500)
    n = length(history)
    n < 10 && return 1.0
    
    μ = mean(history)
    σ² = var(history)
    σ² < 1e-10 && return 1.0
    
    # Compute autocorrelation
    C = zeros(min(max_lag, n ÷ 2))
    for t in eachindex(C)
        C[t] = mean((history[1:n-t] .- μ) .* (history[t+1:n] .- μ)) / σ²
    end
    
    # Find τ where C(τ) ≈ 1/e
    τ = findfirst(c -> c < 1/ℯ, C)
    τ === nothing ? Float64(length(C)) : Float64(τ)
end

"""
Detect ergodicity by analyzing phase space exploration.

Ergodic systems visit O(|Ω|) distinct states.
Non-ergodic systems get trapped in basins.
"""
function is_ergodic(state_hashes::Vector{UInt64}; threshold::Float64=0.1)
    n = length(state_hashes)
    n < 100 && return true  # Not enough data
    
    unique_states = length(unique(state_hashes))
    exploration_ratio = unique_states / n
    
    # Ergodic if exploring at least threshold fraction of visited states
    exploration_ratio > threshold
end

"""
Detect phase transition by susceptibility divergence.
"""
function detect_phase_transition(magnetization_history::Vector{Float64})
    n = length(magnetization_history)
    n < 100 && return false
    
    # Susceptibility χ = N * Var(M)
    χ = n * var(magnetization_history)
    
    # Near critical point, χ diverges
    χ > 100.0
end

# ═══════════════════════════════════════════════════════════════════════════
# Hyperparameter Adaptation
# ═══════════════════════════════════════════════════════════════════════════

"""
Adapt hyperparameters for dynamic sufficiency.

- For ergodic systems: tune toward faster mixing
- For non-ergodic: detect and handle phase coexistence
- Guarantee termination in both cases
"""
function adapt_hyperparameters!(
    params::MetalearnedParams,
    magnetization_history::Vector{Float64},
    state_hashes::Vector{UInt64};
    target_bound::Int=10000
)
    # Estimate current mixing time
    params.τ_mix = estimate_mixing_time(magnetization_history)
    
    # Check ergodicity
    params.ergodic = is_ergodic(state_hashes)
    
    # Near phase transition?
    near_critical = detect_phase_transition(magnetization_history)
    
    if params.ergodic
        # Ergodic: bound = O(τ log(1/ε)) for ε-accuracy
        ε = 0.01  # 99% confidence
        params.termination_bound = min(
            target_bound,
            Int(ceil(params.τ_mix * log(1/ε) * 10))  # Safety factor 10
        )
        
        # Tune β toward critical point for faster decorrelation
        if !near_critical && params.τ_mix > 100
            params.β *= (1 - params.adaptation_rate)
        end
    else
        # Non-ergodic: need to handle metastability
        # Bound = O(exp(ΔF/kT)) in worst case
        # Use multicanonical or parallel tempering to escape
        
        params.termination_bound = min(target_bound, Int(10 * params.τ_mix))
        
        # Tune D toward tricritical point to reduce barrier
        params.D += params.adaptation_rate * sign(mean(magnetization_history))
    end
    
    params
end

# ═══════════════════════════════════════════════════════════════════════════
# Blume-Capel MCMC with Termination Guarantee
# ═══════════════════════════════════════════════════════════════════════════

"""
Blume-Capel local update with chromatic identity.
"""
function blume_capel_sweep!(
    spins::Matrix{Int8},
    params::MetalearnedParams,
    ctx::GayMCContext
)
    L = size(spins, 1)
    rng, color = gay_sweep!(ctx)
    
    for i in 1:L, j in 1:L
        current = spins[i, j]
        
        # Propose new spin from {-1, 0, +1} \ {current}
        candidates = filter(s -> s != current, Int8[-1, 0, 1])
        proposed = candidates[1 + (rand(rng, UInt8) % length(candidates))]
        
        # Energy difference
        neighbors = (
            spins[mod1(i+1, L), j] + spins[mod1(i-1, L), j] +
            spins[i, mod1(j+1, L)] + spins[i, mod1(j-1, L)]
        )
        
        ΔE_exchange = params.J * (proposed - current) * neighbors
        ΔE_crystal = params.D * (proposed^2 - current^2)
        ΔE = ΔE_exchange + ΔE_crystal
        
        # Metropolis criterion
        if ΔE <= 0 || rand(rng) < exp(-params.β * ΔE)
            spins[i, j] = proposed
        end
    end
    
    color
end

"""
Wolff cluster update for faster decorrelation near critical point.
"""
function wolff_cluster_update!(
    spins::Matrix{Int8},
    params::MetalearnedParams,
    ctx::GayMCContext
)
    L = size(spins, 1)
    rng, color = gay_sweep!(ctx)
    
    # Pick random site
    i, j = 1 + rand(rng, UInt) % L, 1 + rand(rng, UInt) % L
    s0 = spins[i, j]
    s0 == 0 && return color  # Can't cluster vacancies
    
    # Build cluster via Swendsen-Wang probability
    p = 1 - exp(-2 * params.β * abs(params.J))
    
    cluster = Set([(i, j)])
    frontier = [(i, j)]
    
    while !isempty(frontier)
        ci, cj = pop!(frontier)
        for (ni, nj) in [
            (mod1(ci+1, L), cj), (mod1(ci-1, L), cj),
            (ci, mod1(cj+1, L)), (ci, mod1(cj-1, L))
        ]
            if spins[ni, nj] == s0 && (ni, nj) ∉ cluster && rand(rng) < p
                push!(cluster, (ni, nj))
                push!(frontier, (ni, nj))
            end
        end
    end
    
    # Flip cluster
    for (ci, cj) in cluster
        spins[ci, cj] = -s0
    end
    
    color
end

"""
Run MCMC with guaranteed eventual termination.

Uses dynamic sufficiency: metalearned hyperparameters adapt to ensure
termination in both ergodic and non-ergodic regimes.
"""
function run_with_termination_guarantee(
    spins::Matrix{Int8},
    params::MetalearnedParams,
    ctx::GayMCContext;
    adaptation_interval::Int=100,
    convergence_threshold::Float64=0.01
)
    history = (
        magnetization = Float64[],
        state_hashes = UInt64[],
        colors = RGB{Float64}[]
    )
    
    iteration = 0
    converged = false
    
    while iteration < params.termination_bound && !converged
        # Choose update based on proximity to critical point
        color = if params.τ_mix > 50
            wolff_cluster_update!(spins, params, ctx)
        else
            blume_capel_sweep!(spins, params, ctx)
        end
        
        # Record observables
        M = mean(spins)
        push!(history.magnetization, M)
        push!(history.state_hashes, hash(spins))
        push!(history.colors, color)
        
        # Periodic adaptation
        if iteration > 0 && iteration % adaptation_interval == 0
            adapt_hyperparameters!(
                params,
                history.magnetization,
                history.state_hashes
            )
        end
        
        # Convergence check (ergodic only)
        if params.ergodic && iteration > 1000
            recent = history.magnetization[max(1, end-100):end]
            if std(recent) < convergence_threshold
                converged = true
            end
        end
        
        iteration += 1
    end
    
    (
        spins = spins,
        iterations = iteration,
        converged = converged,
        termination_bound = params.termination_bound,
        ergodic = params.ergodic,
        final_magnetization = mean(spins),
        τ_mix = params.τ_mix,
        fingerprint = gay_checkpoint(ctx)["checkpoint_color"],
        history = history
    )
end

# ═══════════════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════════════

function world_dynamic_sufficiency(; L::Int=32, seed::UInt64=ORIGINARY_SEED)
    println("╔════════════════════════════════════════════════════════════════╗")
    println("║  Dynamic Sufficiency: Guaranteed Eventual Termination          ║")
    println("║  Blume-Capel Model with Metalearned Hyperparameters            ║")
    println("╚════════════════════════════════════════════════════════════════╝")
    println()
    
    # Initialize lattice (random spin-1)
    spins = rand(Int8[-1, 0, 1], L, L)
    
    # Parameters near tricritical point
    params = MetalearnedParams(β=1.0, D=1.9, J=-1.0)
    ctx = GayMCContext(seed)
    
    println("Initial state:")
    println("  L = $L, β = $(params.β), D = $(params.D)")
    println("  Initial magnetization: $(mean(spins))")
    println()
    
    result = run_with_termination_guarantee(spins, params, ctx)
    
    println("Result:")
    println("  Iterations: $(result.iterations) / $(result.termination_bound)")
    println("  Converged: $(result.converged)")
    println("  Ergodic: $(result.ergodic)")
    println("  Final magnetization: $(round(result.final_magnetization, digits=4))")
    println("  Mixing time: $(round(result.τ_mix, digits=1))")
    println("  Fingerprint: $(result.fingerprint)")
    println()
    
    result
end
