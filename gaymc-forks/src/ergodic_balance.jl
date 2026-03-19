"""
    ErgodicBalance - Fair Scheduling for 4-Fork Interleaving with SPI

ERGODIC polarity (≡ 1 mod 3): Balance, equilibrium, fair resource allocation.

Four forks: Plurigrid, TeglonLabs, Tritwies, bmorphism
Each gets deterministic, fair access while preserving SPI guarantees.

Key properties:
- Round-robin fairness: each fork gets equal opportunity
- Ergodic mixing: every state eventually reaches every other state
- Load balancing: work distributed proportionally to capacity
- Convergence: system reaches steady-state distribution
"""
module ErgodicBalance

using Base.Threads: @threads, @spawn, nthreads, Atomic

export ForkState, ErgodicScheduler, ForkId
export ergodic_step!, ergodic_balance!, verify_ergodicity
export round_robin_schedule, weighted_fair_queue
export convergence_check, steady_state_distribution
export PLURIGRID, TEGLONLABS, TRITWIES, BMORPHISM

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

const GAY_SEED = UInt64(0x285508656870f24a)
const GOLDEN = UInt64(0x9e3779b97f4a7c15)
const ERGODIC_TWIST = UInt64(0x5f5f5f5f5f5f5f5f)  # '_' character repeated

# Fork identifiers (2-bit encoding for 4 forks)
@enum ForkId::UInt8 begin
    PLURIGRID = 0
    TEGLONLABS = 1
    TRITWIES = 2
    BMORPHISM = 3
end

const FORK_NAMES = ["Plurigrid", "TeglonLabs", "Tritwies", "bmorphism"]
const N_FORKS = 4

# ═══════════════════════════════════════════════════════════════════════════
# SPI PRNG
# ═══════════════════════════════════════════════════════════════════════════

@inline function splitmix64(x::UInt64)::UInt64
    x += GOLDEN
    x = (x ⊻ (x >> 30)) * 0xBF58476D1CE4E5B9
    x = (x ⊻ (x >> 27)) * 0x94D049BB133111EB
    x ⊻ (x >> 31)
end

@inline function ergodic_hash(seed::UInt64, fork::ForkId, phase::Int)::UInt64
    splitmix64(seed ⊻ ERGODIC_TWIST ⊻ (UInt64(fork) << 56) ⊻ UInt64(phase))
end

# ═══════════════════════════════════════════════════════════════════════════
# FORK STATE
# ═══════════════════════════════════════════════════════════════════════════

"""
    ForkState

State of a single fork in the interleaving.
Tracks work done, pending work, and chromatic fingerprint.
"""
mutable struct ForkState
    id::ForkId
    seed::UInt64
    work_done::Int              # Total work units completed
    work_pending::Int           # Queued work units
    last_scheduled::Int         # Phase when last scheduled
    fingerprint::UInt64         # Accumulated SPI fingerprint
    credits::Float64            # Fair-share credits (for weighted scheduling)
    active::Bool
end

function ForkState(id::ForkId, seed::UInt64)
    fork_seed = splitmix64(seed ⊻ (UInt64(id) << 48))
    ForkState(id, fork_seed, 0, 0, -1, fork_seed, 1.0, true)
end

# ═══════════════════════════════════════════════════════════════════════════
# ERGODIC SCHEDULER
# ═══════════════════════════════════════════════════════════════════════════

"""
    ErgodicScheduler

Fair scheduler for 4-fork interleaving with ergodic properties.
Guarantees:
1. Each fork eventually gets scheduled (liveness)
2. No fork starves (fairness)
3. Same seed → same schedule (SPI)
4. Every state reachable from every other (ergodicity)
"""
mutable struct ErgodicScheduler
    seed::UInt64
    forks::NTuple{4, ForkState}
    phase::Int                          # Current scheduling phase
    schedule_history::Vector{ForkId}    # History of scheduled forks
    transition_matrix::Matrix{Float64}  # Markov transition probabilities
    fingerprint::UInt64                 # Combined SPI fingerprint
    
    # Metrics
    total_work::Atomic{Int}
    balance_violations::Atomic{Int}
end

function ErgodicScheduler(seed::UInt64=GAY_SEED)
    forks = ntuple(i -> ForkState(ForkId(i-1), seed), 4)
    
    # Initialize doubly-stochastic transition matrix (ergodic)
    # Each row and column sums to 1
    T = fill(0.25, 4, 4)
    
    ErgodicScheduler(
        seed, forks, 0, ForkId[],
        T, splitmix64(seed ⊻ ERGODIC_TWIST),
        Atomic{Int}(0), Atomic{Int}(0)
    )
end

# ═══════════════════════════════════════════════════════════════════════════
# ROUND-ROBIN SCHEDULING (Simplest Fair Schedule)
# ═══════════════════════════════════════════════════════════════════════════

"""
    round_robin_schedule(scheduler, n_phases) -> Vector{ForkId}

Generate n phases of pure round-robin schedule.
Deterministic: phase i goes to fork (i mod 4).
Perfect fairness but no adaptivity.
"""
function round_robin_schedule(scheduler::ErgodicScheduler, n_phases::Int)::Vector{ForkId}
    schedule = Vector{ForkId}(undef, n_phases)
    @inbounds for i in 1:n_phases
        schedule[i] = ForkId((scheduler.phase + i - 1) % N_FORKS)
    end
    schedule
end

"""
    ergodic_step!(scheduler) -> ForkId

Execute one scheduling step with ergodic balance.
Uses weighted fair queuing with SPI-deterministic tie-breaking.
"""
function ergodic_step!(scheduler::ErgodicScheduler)::ForkId
    scheduler.phase += 1
    
    # Calculate deficit for each fork (how far behind fair share)
    deficits = Float64[]
    for fork in scheduler.forks
        if fork.active
            expected = scheduler.phase / N_FORKS
            actual = fork.work_done
            push!(deficits, expected - actual)
        else
            push!(deficits, -Inf)
        end
    end
    
    # Find forks with maximum deficit (most behind)
    max_deficit = maximum(deficits)
    candidates = findall(d -> d == max_deficit, deficits)
    
    # SPI tie-breaking: deterministic selection among tied candidates
    if length(candidates) > 1
        tie_break = ergodic_hash(scheduler.seed, ForkId(0), scheduler.phase)
        selected_idx = candidates[(tie_break % length(candidates)) + 1]
    else
        selected_idx = candidates[1]
    end
    
    selected = ForkId(selected_idx - 1)
    
    # Update state
    fork = scheduler.forks[selected_idx]
    fork.work_done += 1
    fork.last_scheduled = scheduler.phase
    fork.fingerprint = splitmix64(fork.fingerprint ⊻ UInt64(scheduler.phase))
    
    # Update scheduler fingerprint (XOR all fork fingerprints)
    scheduler.fingerprint = reduce(⊻, f.fingerprint for f in scheduler.forks)
    
    # Record history
    push!(scheduler.schedule_history, selected)
    Threads.atomic_add!(scheduler.total_work, 1)
    
    selected
end

# ═══════════════════════════════════════════════════════════════════════════
# WEIGHTED FAIR QUEUING
# ═══════════════════════════════════════════════════════════════════════════

"""
    weighted_fair_queue(scheduler, weights, n_phases) -> Vector{ForkId}

Generate schedule with weighted fairness.
weights[i] = relative share for fork i (e.g., [1, 2, 1, 1] gives TeglonLabs 2x)
Still preserves SPI: same weights + seed → same schedule.
"""
function weighted_fair_queue(scheduler::ErgodicScheduler, 
                             weights::Vector{Float64}, 
                             n_phases::Int)::Vector{ForkId}
    @assert length(weights) == N_FORKS "Need exactly 4 weights"
    
    # Normalize weights
    total = sum(weights)
    normalized = weights ./ total
    
    # Virtual time for each fork
    virtual_times = zeros(Float64, N_FORKS)
    schedule = Vector{ForkId}(undef, n_phases)
    
    for i in 1:n_phases
        # Find fork with minimum virtual time
        min_vt = Inf
        candidates = Int[]
        
        for j in 1:N_FORKS
            if scheduler.forks[j].active
                vt = virtual_times[j]
                if vt < min_vt
                    min_vt = vt
                    candidates = [j]
                elseif vt == min_vt
                    push!(candidates, j)
                end
            end
        end
        
        # SPI tie-breaking
        if length(candidates) > 1
            tie_hash = ergodic_hash(scheduler.seed, ForkId(0), scheduler.phase + i)
            selected = candidates[(tie_hash % length(candidates)) + 1]
        else
            selected = candidates[1]
        end
        
        schedule[i] = ForkId(selected - 1)
        
        # Advance virtual time: smaller weight = faster virtual time
        virtual_times[selected] += 1.0 / normalized[selected]
    end
    
    schedule
end

# ═══════════════════════════════════════════════════════════════════════════
# ERGODIC BALANCE - MAIN ALGORITHM
# ═══════════════════════════════════════════════════════════════════════════

"""
    ergodic_balance!(scheduler, n_steps; work_fn=identity) -> NamedTuple

Execute n_steps of ergodic-balanced scheduling with optional work function.
Returns metrics about the balancing quality.
"""
function ergodic_balance!(scheduler::ErgodicScheduler, n_steps::Int;
                          work_fn::Function=identity)
    results = Vector{Any}(undef, n_steps)
    fork_counts = zeros(Int, N_FORKS)
    
    for i in 1:n_steps
        fork_id = ergodic_step!(scheduler)
        fork_counts[Int(fork_id) + 1] += 1
        
        # Execute work for this fork
        fork_state = scheduler.forks[Int(fork_id) + 1]
        results[i] = work_fn(fork_id, fork_state, scheduler.phase)
    end
    
    # Calculate balance metrics
    expected = n_steps / N_FORKS
    max_deviation = maximum(abs.(fork_counts .- expected))
    balance_score = 1.0 - (max_deviation / expected)
    
    # Check starvation: did any fork go too long without service?
    max_gap = 0
    for fork in scheduler.forks
        gap = scheduler.phase - fork.last_scheduled
        max_gap = max(max_gap, gap)
    end
    
    (
        results = results,
        fork_counts = fork_counts,
        balance_score = balance_score,
        max_gap = max_gap,
        fingerprint = scheduler.fingerprint,
        phases = scheduler.phase
    )
end

# ═══════════════════════════════════════════════════════════════════════════
# ERGODICITY VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════

"""
    verify_ergodicity(scheduler, n_trials=1000) -> Bool

Verify ergodic property: from any state, all states are eventually reachable.
Uses empirical verification over n_trials random walks.
"""
function verify_ergodicity(scheduler::ErgodicScheduler, n_trials::Int=1000)::Bool
    # For our scheduler, ergodicity means:
    # 1. Every fork is eventually scheduled
    # 2. No cyclic behavior that excludes forks
    
    for trial in 1:n_trials
        trial_seed = splitmix64(scheduler.seed ⊻ UInt64(trial))
        test_sched = ErgodicScheduler(trial_seed)
        
        # Run for 4 * N_FORKS steps (enough for one full cycle + buffer)
        seen = Set{ForkId}()
        for _ in 1:(4 * N_FORKS)
            fork = ergodic_step!(test_sched)
            push!(seen, fork)
        end
        
        # All forks should be seen
        if length(seen) != N_FORKS
            return false
        end
    end
    
    true
end

"""
    convergence_check(scheduler, target_distribution, tolerance=0.05) -> Bool

Check if scheduler has converged to target steady-state distribution.
target_distribution: [p1, p2, p3, p4] where pi = target fraction for fork i
"""
function convergence_check(scheduler::ErgodicScheduler, 
                          target_distribution::Vector{Float64},
                          tolerance::Float64=0.05)::Bool
    total = scheduler.phase
    total == 0 && return false
    
    for (i, fork) in enumerate(scheduler.forks)
        actual = fork.work_done / total
        expected = target_distribution[i]
        if abs(actual - expected) > tolerance
            return false
        end
    end
    
    true
end

"""
    steady_state_distribution(scheduler) -> Vector{Float64}

Calculate current empirical distribution across forks.
"""
function steady_state_distribution(scheduler::ErgodicScheduler)::Vector{Float64}
    total = max(1, scheduler.phase)
    [fork.work_done / total for fork in scheduler.forks]
end

# ═══════════════════════════════════════════════════════════════════════════
# TRANSITION MATRIX ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

"""
    update_transition_matrix!(scheduler)

Update Markov transition matrix based on observed schedule.
Used for analyzing ergodic mixing properties.
"""
function update_transition_matrix!(scheduler::ErgodicScheduler)
    history = scheduler.schedule_history
    length(history) < 2 && return
    
    # Count transitions
    counts = zeros(Int, N_FORKS, N_FORKS)
    for i in 1:(length(history) - 1)
        from = Int(history[i]) + 1
        to = Int(history[i + 1]) + 1
        counts[from, to] += 1
    end
    
    # Normalize to probabilities
    for i in 1:N_FORKS
        row_sum = sum(counts[i, :])
        if row_sum > 0
            scheduler.transition_matrix[i, :] .= counts[i, :] ./ row_sum
        end
    end
end

"""
    spectral_gap(scheduler) -> Float64

Calculate spectral gap of transition matrix.
Larger gap = faster mixing = better ergodicity.
For uniform schedule, gap ≈ 0 (second eigenvalue = 1).
"""
function spectral_gap(scheduler::ErgodicScheduler)::Float64
    update_transition_matrix!(scheduler)
    
    # Eigenvalues of transition matrix
    eigenvalues = eigvals(scheduler.transition_matrix)
    sorted = sort(abs.(eigenvalues), rev=true)
    
    # Gap is 1 - |λ₂| where λ₁ = 1 (Perron-Frobenius)
    length(sorted) >= 2 ? 1.0 - sorted[2] : 0.0
end

using LinearAlgebra: eigvals

# ═══════════════════════════════════════════════════════════════════════════
# SPI VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════

"""
    verify_spi(seed, n_steps=100, n_trials=10) -> Bool

Verify Strong Parallelism Invariance:
Same seed → same schedule regardless of execution order.
"""
function verify_spi(seed::UInt64=GAY_SEED; n_steps::Int=100, n_trials::Int=10)::Bool
    # Reference run
    ref_sched = ErgodicScheduler(seed)
    ref_result = ergodic_balance!(ref_sched, n_steps)
    ref_fp = ref_result.fingerprint
    ref_history = copy(ref_sched.schedule_history)
    
    # Trial runs with same seed
    for trial in 1:n_trials
        test_sched = ErgodicScheduler(seed)
        test_result = ergodic_balance!(test_sched, n_steps)
        
        # Fingerprints must match
        if test_result.fingerprint != ref_fp
            return false
        end
        
        # Schedule history must match
        if test_sched.schedule_history != ref_history
            return false
        end
    end
    
    true
end

# ═══════════════════════════════════════════════════════════════════════════
# EQUILIBRIUM CONDITIONS
# ═══════════════════════════════════════════════════════════════════════════

"""
    EquilibriumCondition

Defines when the system is considered "in equilibrium".
"""
struct EquilibriumCondition
    max_deviation::Float64      # Maximum allowed deviation from fair share
    max_starvation_gap::Int     # Maximum phases without service
    min_balance_score::Float64  # Minimum balance score (0-1)
    convergence_window::Int     # Phases to check for convergence
end

EquilibriumCondition() = EquilibriumCondition(0.1, 8, 0.9, 20)

"""
    check_equilibrium(scheduler, condition) -> NamedTuple

Check if scheduler is in equilibrium according to condition.
"""
function check_equilibrium(scheduler::ErgodicScheduler, 
                          cond::EquilibriumCondition=EquilibriumCondition())
    dist = steady_state_distribution(scheduler)
    expected = 0.25  # Fair share for 4 forks
    
    max_dev = maximum(abs.(dist .- expected))
    
    # Calculate starvation gap
    max_gap = 0
    for fork in scheduler.forks
        gap = scheduler.phase - fork.last_scheduled
        max_gap = max(max_gap, gap)
    end
    
    # Balance score from recent history
    balance_ok = max_dev <= cond.max_deviation
    gap_ok = max_gap <= cond.max_starvation_gap
    
    in_equilibrium = balance_ok && gap_ok
    
    (
        in_equilibrium = in_equilibrium,
        max_deviation = max_dev,
        starvation_gap = max_gap,
        distribution = dist,
        balance_ok = balance_ok,
        gap_ok = gap_ok
    )
end

# ═══════════════════════════════════════════════════════════════════════════
# FORK-SPECIFIC WORK INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════

"""
    default_fork_work(fork_id, fork_state, phase) -> UInt64

Default work function: compute and return fingerprint.
Override to integrate with actual fork implementations.
"""
function default_fork_work(fork_id::ForkId, fork_state::ForkState, phase::Int)::UInt64
    # Simulate work by mixing fingerprints
    work_hash = ergodic_hash(fork_state.seed, fork_id, phase)
    fork_state.fingerprint ⊻= work_hash
    work_hash
end

"""
    interleaved_execution!(scheduler, n_steps) -> NamedTuple

Execute interleaved work across all 4 forks with ergodic balance.
This is the main entry point for integrated fork execution.
"""
function interleaved_execution!(scheduler::ErgodicScheduler, n_steps::Int)
    # Execute with default work
    result = ergodic_balance!(scheduler, n_steps; work_fn=default_fork_work)
    
    # Collect per-fork fingerprints
    fork_fps = [fork.fingerprint for fork in scheduler.forks]
    combined_fp = reduce(⊻, fork_fps)
    
    # Verify equilibrium
    eq_check = check_equilibrium(scheduler)
    
    (
        fork_fingerprints = Dict(
            PLURIGRID => fork_fps[1],
            TEGLONLABS => fork_fps[2],
            TRITWIES => fork_fps[3],
            BMORPHISM => fork_fps[4]
        ),
        combined_fingerprint = combined_fp,
        equilibrium = eq_check,
        balance_score = result.balance_score,
        total_phases = scheduler.phase,
        spi_verified = verify_spi(scheduler.seed; n_steps=min(50, n_steps), n_trials=3)
    )
end

# ═══════════════════════════════════════════════════════════════════════════
# PRETTY PRINTING
# ═══════════════════════════════════════════════════════════════════════════

function Base.show(io::IO, s::ErgodicScheduler)
    eq = check_equilibrium(s)
    status = eq.in_equilibrium ? "✓ EQUILIBRIUM" : "○ converging"
    
    println(io, "ErgodicScheduler($status)")
    println(io, "  Phase: $(s.phase)")
    println(io, "  Fingerprint: 0x$(string(s.fingerprint, base=16))")
    println(io, "  Distribution: $(round.(steady_state_distribution(s), digits=3))")
    println(io, "  Fork states:")
    for fork in s.forks
        println(io, "    $(FORK_NAMES[Int(fork.id)+1]): $(fork.work_done) done, last@$(fork.last_scheduled)")
    end
end

end # module ErgodicBalance
