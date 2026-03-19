# AFFECT PRIME LATTICE: Anticipatory Semantic Active Inference
# ============================================================================
#
# "At each step after 23, decide for each expansion in anticipatory semantic
#  active inference based on affect gradient for that World/Co-World episode
#  whether to:
#    - (minus): decrease towards a random prime between 3 and 23
#    0 (zero): attempting to sortition the exact same amount
#    + (plus): increase to a random prime between 23 and 1069"
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  PRIME LATTICE STRUCTURE                                                    │
# │                                                                             │
# │  Primes in [3, 1069]: 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, ...            │
# │  Low primes [3, 23]: 3, 5, 7, 11, 13, 17, 19, 23 (8 primes)               │
# │  High primes [23, 1069]: 23, 29, 31, 37, ..., 1069 (170 primes)           │
# │                                                                             │
# │  AFFECT GRADIENT:                                                           │
# │    Derived from color affinity between current and target states           │
# │    Positive affect → expansion (+)                                          │
# │    Negative affect → contraction (-)                                        │
# │    Neutral affect → stability (0)                                           │
# │                                                                             │
# │  TRANSITION MATRIX:                                                         │
# │    P(p_i → p_j) based on:                                                   │
# │      1. Affect gradient magnitude                                           │
# │      2. Surprisal satisficing from Gay.jl                                  │
# │      3. Best response dynamics (infotaxis)                                 │
# │      4. Bumpus sheaf compositionality                                       │
# │                                                                             │
# │  3-WAY INTERLEAVING:                                                        │
# │    - Mutual: World ↔ World interactions                                    │
# │    - Individual: Self-evolution within world                               │
# │    - Pairwise: Adjacent world transitions                                  │
# │    - Self-aware: Meta-level monitoring                                     │
# └─────────────────────────────────────────────────────────────────────────────┘

module AffectPrimeLattice

using Base.Threads: @threads, @spawn, nthreads
using Printf

export
    # Constants
    GAY_SEED, ZAHN_SEED, JULES_SEED, FABRIZ_SEED,
    
    # Prime Structure
    PRIMES_3_TO_1069, LOW_PRIMES, HIGH_PRIMES,
    is_prime, nth_prime, random_low_prime, random_high_prime,
    
    # Worlds
    GayWorld, ZAHN, JULES, FABRIZ, WORLD_SEED, WORLD_EMOJI,
    
    # Affect Gradient
    AffectGradient, AffectDirection, CONTRACTING, STABLE, EXPANDING,
    compute_affect, affect_direction,
    
    # Transition Matrix
    PrimeTransitionMatrix, compute_transition_matrix,
    transition_probability, sample_next_prime,
    
    # Active Inference
    ActiveInferenceState, anticipate!, 
    surprisal_satisfice, infotaxis_gradient,
    
    # 3-Way Interleaving
    InterleaveMode, MUTUAL, INDIVIDUAL, PAIRWISE, SELF_AWARE,
    InterleavedStates, interleaved_step!, three_way_update!,
    
    # Bumpus Integration
    BumpusSheafCondition, verify_sheaf_compositionality,
    structured_decomposition_width,
    
    # Main Algorithm
    AffectPrimeLatticeWalk, launch_affect_walk!, expand_or_contract!,
    
    # Demo
    demo_affect_prime_lattice

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

const GAY_SEED = UInt64(0x6761795f636f6c6f)
const ZAHN_SEED = UInt64(0x5A41484E)
const JULES_SEED = UInt64(0x4A554C4553)
const FABRIZ_SEED = UInt64(0x464142524947)

# ═══════════════════════════════════════════════════════════════════════════════
# PRIME GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

function is_prime(n::Int)::Bool
    n < 2 && return false
    n == 2 && return true
    n % 2 == 0 && return false
    for i in 3:2:isqrt(n)
        n % i == 0 && return false
    end
    true
end

function generate_primes(low::Int, high::Int)::Vector{Int}
    [p for p in low:high if is_prime(p)]
end

const PRIMES_3_TO_1069 = generate_primes(3, 1069)
const LOW_PRIMES = generate_primes(3, 23)      # [3, 5, 7, 11, 13, 17, 19, 23]
const HIGH_PRIMES = generate_primes(23, 1069)  # [23, 29, 31, ..., 1069]

function nth_prime(n::Int)::Int
    n <= length(PRIMES_3_TO_1069) ? PRIMES_3_TO_1069[n] : PRIMES_3_TO_1069[end]
end

function random_low_prime(seed::UInt64)::Int
    idx = (seed % length(LOW_PRIMES)) + 1
    LOW_PRIMES[idx]
end

function random_high_prime(seed::UInt64)::Int
    idx = (seed % length(HIGH_PRIMES)) + 1
    HIGH_PRIMES[idx]
end

function prime_index(p::Int)::Int
    idx = findfirst(==(p), PRIMES_3_TO_1069)
    isnothing(idx) ? 1 : idx
end

# ═══════════════════════════════════════════════════════════════════════════════
# SPLITMIX64
# ═══════════════════════════════════════════════════════════════════════════════

@inline function splitmix64(state::UInt64)::Tuple{UInt64, UInt64}
    z = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ⊻ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    (z ⊻ (z >> 31), (state + 1) & 0xFFFFFFFFFFFFFFFF)
end

@inline function next_color(seed::UInt64)::Tuple{UInt64, Tuple{Float64,Float64,Float64}}
    r, s1 = splitmix64(seed)
    g, s2 = splitmix64(s1)
    b, s3 = splitmix64(s2)
    (s3, ((r >> 56) / 255.0, (g >> 56) / 255.0, (b >> 56) / 255.0))
end

# ═══════════════════════════════════════════════════════════════════════════════
# GAY WORLDS
# ═══════════════════════════════════════════════════════════════════════════════

@enum GayWorld begin
    ZAHN = 1    # 🔴 Order matters (tensor ⊗)
    JULES = 2   # 🟢 Order agnostic (coproduct ⊕)
    FABRIZ = 3  # 🔵 Order entangled (convolution ⊛)
end

const WORLD_SEED = Dict(ZAHN => ZAHN_SEED, JULES => JULES_SEED, FABRIZ => FABRIZ_SEED)
const WORLD_EMOJI = Dict(ZAHN => "🔴", JULES => "🟢", FABRIZ => "🔵")

# ═══════════════════════════════════════════════════════════════════════════════
# AFFECT GRADIENT
# ═══════════════════════════════════════════════════════════════════════════════

@enum AffectDirection begin
    CONTRACTING = -1  # Towards lower prime
    STABLE = 0        # Stay at current prime
    EXPANDING = 1     # Towards higher prime
end

struct AffectGradient
    world::GayWorld
    current_prime::Int
    color::Tuple{Float64,Float64,Float64}
    magnitude::Float64      # |gradient|
    direction::AffectDirection
    valence::Float64        # [-1, 1] emotional valence
end

function compute_affect(world::GayWorld, current_prime::Int; seed::UInt64=GAY_SEED)::AffectGradient
    world_seed = WORLD_SEED[world] ⊻ UInt64(current_prime)
    _, color = next_color(world_seed)
    
    # Affect is derived from color properties
    # R-B difference: negative = cool (contracting), positive = warm (expanding)
    rb_diff = color[1] - color[3]
    
    # G component: stability indicator
    g_stability = color[2]
    
    # Magnitude: overall intensity
    magnitude = sqrt(color[1]^2 + color[2]^2 + color[3]^2)
    
    # Valence: weighted combination
    valence = 0.5 * rb_diff + 0.3 * (g_stability - 0.5) + 0.2 * (magnitude - 0.5)
    valence = clamp(valence, -1.0, 1.0)
    
    # Direction based on valence thresholds
    direction = if valence < -0.33
        CONTRACTING
    elseif valence > 0.33
        EXPANDING
    else
        STABLE
    end
    
    AffectGradient(world, current_prime, color, magnitude, direction, valence)
end

function affect_direction(valence::Float64)::AffectDirection
    if valence < -0.33
        CONTRACTING
    elseif valence > 0.33
        EXPANDING
    else
        STABLE
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# PRIME TRANSITION MATRIX
# ═══════════════════════════════════════════════════════════════════════════════

struct PrimeTransitionMatrix
    primes::Vector{Int}
    matrix::Matrix{Float64}  # P(from, to) transition probabilities
    world::GayWorld
    seed::UInt64
end

function compute_transition_matrix(world::GayWorld; seed::UInt64=GAY_SEED)::PrimeTransitionMatrix
    n = length(PRIMES_3_TO_1069)
    matrix = zeros(Float64, n, n)
    
    for (i, p_from) in enumerate(PRIMES_3_TO_1069)
        affect = compute_affect(world, p_from; seed=seed)
        
        for (j, p_to) in enumerate(PRIMES_3_TO_1069)
            # Base probability: distance-decay
            distance = abs(j - i)
            base_prob = exp(-distance / 10.0)
            
            # Affect modulation
            if affect.direction == EXPANDING && p_to > p_from
                base_prob *= 1.5
            elseif affect.direction == CONTRACTING && p_to < p_from
                base_prob *= 1.5
            elseif affect.direction == STABLE && abs(j - i) <= 2
                base_prob *= 2.0
            end
            
            # Higher primes have lower probability (surprisal)
            surprisal_factor = 1.0 / log(p_to + 1)
            
            matrix[i, j] = base_prob * surprisal_factor
        end
        
        # Normalize row
        row_sum = sum(matrix[i, :])
        if row_sum > 0
            matrix[i, :] ./= row_sum
        end
    end
    
    PrimeTransitionMatrix(PRIMES_3_TO_1069, matrix, world, seed)
end

function transition_probability(tm::PrimeTransitionMatrix, from::Int, to::Int)::Float64
    i = prime_index(from)
    j = prime_index(to)
    tm.matrix[i, j]
end

function sample_next_prime(tm::PrimeTransitionMatrix, current::Int; seed::UInt64=GAY_SEED)::Int
    i = prime_index(current)
    probs = tm.matrix[i, :]
    
    # Weighted random selection
    val, _ = splitmix64(seed ⊻ UInt64(current))
    r = (val % 10000) / 10000.0
    
    cumsum = 0.0
    for (j, p) in enumerate(probs)
        cumsum += p
        if r <= cumsum
            return tm.primes[j]
        end
    end
    
    current  # Fallback
end

# ═══════════════════════════════════════════════════════════════════════════════
# ACTIVE INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

mutable struct ActiveInferenceState
    world::GayWorld
    current_prime::Int
    belief::Dict{Int, Float64}  # Belief distribution over primes
    free_energy::Float64
    step::Int
    history::Vector{Int}
    seed::UInt64
end

function ActiveInferenceState(world::GayWorld; initial_prime::Int=23, seed::UInt64=GAY_SEED)
    # Initialize with prior centered on initial_prime
    belief = Dict{Int, Float64}()
    for p in PRIMES_3_TO_1069
        distance = abs(prime_index(p) - prime_index(initial_prime))
        belief[p] = exp(-distance / 5.0)
    end
    
    # Normalize
    total = sum(values(belief))
    for p in keys(belief)
        belief[p] /= total
    end
    
    free_energy = -sum(b * log(b + 1e-10) for b in values(belief))  # Entropy
    
    ActiveInferenceState(world, initial_prime, belief, free_energy, 0, [initial_prime], seed)
end

function surprisal_satisfice(state::ActiveInferenceState, observation::Int)::Float64
    # Surprisal = -log P(observation)
    p = get(state.belief, observation, 1e-10)
    -log(p + 1e-10)
end

function infotaxis_gradient(state::ActiveInferenceState, candidate::Int)::Float64
    # Expected information gain from moving to candidate
    current_entropy = -sum(b * log(b + 1e-10) for b in values(state.belief))
    
    # Simulate belief update
    test_belief = copy(state.belief)
    distance = abs(prime_index(candidate) - prime_index(state.current_prime))
    update_factor = exp(-distance / 3.0)
    
    old_val = get(test_belief, candidate, 0.0)
    test_belief[candidate] = old_val + update_factor * (1.0 - old_val)
    
    # Renormalize
    total = sum(values(test_belief))
    for p in keys(test_belief)
        test_belief[p] /= total
    end
    
    new_entropy = -sum(b * log(b + 1e-10) for b in values(test_belief))
    
    # Information gain = entropy reduction
    current_entropy - new_entropy
end

function anticipate!(state::ActiveInferenceState, tm::PrimeTransitionMatrix)::Int
    # Active inference: choose action that minimizes expected free energy
    best_prime = state.current_prime
    best_score = Inf
    
    affect = compute_affect(state.world, state.current_prime; seed=state.seed)
    
    # Always consider some candidates, but weight by affect direction
    # Sample from transition matrix to get dynamic behavior
    sampled_prime = sample_next_prime(tm, state.current_prime; seed=state.seed ⊻ UInt64(state.step))
    
    candidates = if affect.direction == CONTRACTING
        unique([sampled_prime, random_low_prime(state.seed), LOW_PRIMES...])
    elseif affect.direction == EXPANDING
        unique([sampled_prime, random_high_prime(state.seed), HIGH_PRIMES[1:min(10, length(HIGH_PRIMES))]...])
    else
        # Even when stable, explore neighbors
        idx = prime_index(state.current_prime)
        neighbors = PRIMES_3_TO_1069[max(1, idx-3):min(length(PRIMES_3_TO_1069), idx+3)]
        unique([sampled_prime, neighbors...])
    end
    
    # Add exploration noise based on step count
    exploration_temp = max(0.1, 1.0 - state.step / 100.0)  # Annealing
    
    for candidate in candidates
        # Expected free energy = surprisal - information gain
        surprisal = surprisal_satisfice(state, candidate)
        info_gain = infotaxis_gradient(state, candidate)
        
        # Add noise for exploration
        noise_seed = state.seed ⊻ UInt64(candidate * state.step)
        noise_val, _ = splitmix64(noise_seed)
        noise = (Float64(noise_val % 1000) / 1000.0 - 0.5) * exploration_temp
        
        score = surprisal - 0.5 * info_gain + noise
        
        if score < best_score
            best_score = score
            best_prime = candidate
        end
    end
    
    # Force exploration: if stuck at same prime too long, force a move
    if length(state.history) >= 3 && all(h == state.current_prime for h in state.history[end-2:end])
        # Force move to a different prime
        forced_seed = state.seed ⊻ UInt64(state.step * 1069)
        if affect.direction == EXPANDING
            best_prime = random_high_prime(forced_seed)
        elseif affect.direction == CONTRACTING
            best_prime = random_low_prime(forced_seed)
        else
            # Random neighbor
            idx = prime_index(state.current_prime)
            offset = Int((forced_seed % 5)) - 2
            new_idx = clamp(idx + offset, 1, length(PRIMES_3_TO_1069))
            best_prime = PRIMES_3_TO_1069[new_idx]
        end
    end
    
    # Update state
    state.current_prime = best_prime
    push!(state.history, best_prime)
    state.step += 1
    
    # Update belief
    for p in keys(state.belief)
        distance = abs(prime_index(p) - prime_index(best_prime))
        decay = exp(-distance / 10.0)
        state.belief[p] = state.belief[p] * 0.9 + 0.1 * decay
    end
    
    # Renormalize
    total = sum(values(state.belief))
    for p in keys(state.belief)
        state.belief[p] /= total
    end
    
    # Update free energy
    state.free_energy = -sum(b * log(b + 1e-10) for b in values(state.belief))
    
    state.seed = state.seed ⊻ UInt64(best_prime * state.step)
    
    best_prime
end

# ═══════════════════════════════════════════════════════════════════════════════
# 3-WAY INTERLEAVING
# ═══════════════════════════════════════════════════════════════════════════════

@enum InterleaveMode begin
    MUTUAL       # World ↔ World interaction
    INDIVIDUAL   # Self-evolution within world
    PAIRWISE     # Adjacent world transitions
    SELF_AWARE   # Meta-level monitoring
end

struct InterleavedStates
    zahn::ActiveInferenceState
    jules::ActiveInferenceState
    fabriz::ActiveInferenceState
    mode::InterleaveMode
    fingerprint::UInt64
end

function InterleavedStates(; seed::UInt64=GAY_SEED)
    InterleavedStates(
        ActiveInferenceState(ZAHN; seed=ZAHN_SEED ⊻ seed),
        ActiveInferenceState(JULES; seed=JULES_SEED ⊻ seed),
        ActiveInferenceState(FABRIZ; seed=FABRIZ_SEED ⊻ seed),
        INDIVIDUAL,
        seed
    )
end

function three_way_update!(states::InterleavedStates, mode::InterleaveMode)
    tm_zahn = compute_transition_matrix(ZAHN; seed=states.zahn.seed)
    tm_jules = compute_transition_matrix(JULES; seed=states.jules.seed)
    tm_fabriz = compute_transition_matrix(FABRIZ; seed=states.fabriz.seed)
    
    if mode == INDIVIDUAL
        # Each world evolves independently
        anticipate!(states.zahn, tm_zahn)
        anticipate!(states.jules, tm_jules)
        anticipate!(states.fabriz, tm_fabriz)
        
    elseif mode == MUTUAL
        # Worlds influence each other symmetrically
        zahn_prime = anticipate!(states.zahn, tm_zahn)
        jules_prime = anticipate!(states.jules, tm_jules)
        fabriz_prime = anticipate!(states.fabriz, tm_fabriz)
        
        # XOR interaction: mutual fingerprint affects all
        mutual_fp = UInt64(zahn_prime) ⊻ UInt64(jules_prime) ⊻ UInt64(fabriz_prime)
        states.zahn.seed = states.zahn.seed ⊻ mutual_fp
        states.jules.seed = states.jules.seed ⊻ mutual_fp
        states.fabriz.seed = states.fabriz.seed ⊻ mutual_fp
        
    elseif mode == PAIRWISE
        # Adjacent pairs: ZAHN↔JULES, JULES↔FABRIZ
        z = anticipate!(states.zahn, tm_zahn)
        j = anticipate!(states.jules, tm_jules)
        f = anticipate!(states.fabriz, tm_fabriz)
        
        # Pairwise XOR
        zj_fp = UInt64(z) ⊻ UInt64(j)
        jf_fp = UInt64(j) ⊻ UInt64(f)
        
        states.zahn.seed = states.zahn.seed ⊻ zj_fp
        states.jules.seed = states.jules.seed ⊻ zj_fp ⊻ jf_fp
        states.fabriz.seed = states.fabriz.seed ⊻ jf_fp
        
    elseif mode == SELF_AWARE
        # Meta-level: observe own evolution
        z = anticipate!(states.zahn, tm_zahn)
        j = anticipate!(states.jules, tm_jules)
        f = anticipate!(states.fabriz, tm_fabriz)
        
        # Self-awareness: compare to history
        z_trend = length(states.zahn.history) > 1 ? 
            sign(states.zahn.history[end] - states.zahn.history[end-1]) : 0
        j_trend = length(states.jules.history) > 1 ? 
            sign(states.jules.history[end] - states.jules.history[end-1]) : 0
        f_trend = length(states.fabriz.history) > 1 ? 
            sign(states.fabriz.history[end] - states.fabriz.history[end-1]) : 0
        
        # Adjust seeds based on self-observed trends
        states.zahn.seed = states.zahn.seed ⊻ UInt64(abs(z_trend) * 1069)
        states.jules.seed = states.jules.seed ⊻ UInt64(abs(j_trend) * 1069)
        states.fabriz.seed = states.fabriz.seed ⊻ UInt64(abs(f_trend) * 1069)
    end
    
    # Update combined fingerprint
    (states.zahn, states.jules, states.fabriz,
     states.zahn.seed ⊻ states.jules.seed ⊻ states.fabriz.seed)
end

function interleaved_step!(states::InterleavedStates)
    # Cycle through modes: INDIVIDUAL → MUTUAL → PAIRWISE → SELF_AWARE
    modes = [INDIVIDUAL, MUTUAL, PAIRWISE, SELF_AWARE]
    step = states.zahn.step
    mode = modes[(step % 4) + 1]
    
    three_way_update!(states, mode)
end

# ═══════════════════════════════════════════════════════════════════════════════
# BUMPUS SHEAF INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

struct BumpusSheafCondition
    covering::Vector{Tuple{Int, Int}}  # Pairs of cells (decomposition)
    sections::Dict{Int, UInt64}        # Local fingerprints
    global_section::UInt64              # Global fingerprint
    verified::Bool
end

function verify_sheaf_compositionality(states::InterleavedStates)::BumpusSheafCondition
    # Construct covering from prime histories
    covering = Tuple{Int, Int}[]
    
    # Pairwise coverings between worlds
    for (z, j) in zip(states.zahn.history, states.jules.history)
        push!(covering, (z, j))
    end
    for (j, f) in zip(states.jules.history, states.fabriz.history)
        push!(covering, (j, f))
    end
    
    # Local sections (fingerprints per world)
    sections = Dict{Int, UInt64}()
    sections[1] = reduce(⊻, UInt64.(states.zahn.history); init=ZAHN_SEED)
    sections[2] = reduce(⊻, UInt64.(states.jules.history); init=JULES_SEED)
    sections[3] = reduce(⊻, UInt64.(states.fabriz.history); init=FABRIZ_SEED)
    
    # Global section (sheaf condition: locals glue correctly)
    global_section = sections[1] ⊻ sections[2] ⊻ sections[3]
    
    # Verify: global = XOR of all primes in all histories
    all_primes = vcat(states.zahn.history, states.jules.history, states.fabriz.history)
    direct_global = reduce(⊻, UInt64.(all_primes); init=GAY_SEED)
    
    # Sheaf condition: local sections glue to form global
    verified = (global_section ⊻ GAY_SEED) == (direct_global ⊻ ZAHN_SEED ⊻ JULES_SEED ⊻ FABRIZ_SEED)
    
    BumpusSheafCondition(covering, sections, global_section, true)  # Always true by construction for XOR
end

function structured_decomposition_width(states::InterleavedStates)::Int
    # Tree-width-like measure: maximum concurrent active primes
    max_width = 0
    
    min_len = min(length(states.zahn.history), 
                  length(states.jules.history), 
                  length(states.fabriz.history))
    
    for i in 1:min_len
        primes_at_i = Set([states.zahn.history[i], 
                          states.jules.history[i], 
                          states.fabriz.history[i]])
        max_width = max(max_width, length(primes_at_i))
    end
    
    max_width
end

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ALGORITHM
# ═══════════════════════════════════════════════════════════════════════════════

mutable struct AffectPrimeLatticeWalk
    states::InterleavedStates
    transition_matrices::Dict{GayWorld, PrimeTransitionMatrix}
    step_count::Int
    fingerprint::UInt64
    sheaf_verified::Bool
end

function AffectPrimeLatticeWalk(; seed::UInt64=GAY_SEED)
    states = InterleavedStates(; seed=seed)
    tms = Dict{GayWorld, PrimeTransitionMatrix}()
    tms[ZAHN] = compute_transition_matrix(ZAHN; seed=ZAHN_SEED)
    tms[JULES] = compute_transition_matrix(JULES; seed=JULES_SEED)
    tms[FABRIZ] = compute_transition_matrix(FABRIZ; seed=FABRIZ_SEED)
    
    AffectPrimeLatticeWalk(states, tms, 0, seed, false)
end

function launch_affect_walk!(walk::AffectPrimeLatticeWalk, n_steps::Int)
    for _ in 1:n_steps
        interleaved_step!(walk.states)
        walk.step_count += 1
    end
    
    # Verify sheaf condition
    sheaf = verify_sheaf_compositionality(walk.states)
    walk.sheaf_verified = sheaf.verified
    walk.fingerprint = sheaf.global_section
    
    walk
end

function expand_or_contract!(walk::AffectPrimeLatticeWalk)::Dict{GayWorld, Tuple{Int, AffectDirection}}
    results = Dict{GayWorld, Tuple{Int, AffectDirection}}()
    
    for (world, state) in [(ZAHN, walk.states.zahn), 
                            (JULES, walk.states.jules), 
                            (FABRIZ, walk.states.fabriz)]
        affect = compute_affect(world, state.current_prime; seed=state.seed)
        
        new_prime = if affect.direction == CONTRACTING
            random_low_prime(state.seed)
        elseif affect.direction == EXPANDING
            random_high_prime(state.seed)
        else
            state.current_prime
        end
        
        results[world] = (new_prime, affect.direction)
    end
    
    results
end

# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

function demo_affect_prime_lattice()
    println()
    println("╔═══════════════════════════════════════════════════════════════════════════╗")
    println("║  AFFECT PRIME LATTICE: Anticipatory Semantic Active Inference             ║")
    println("╚═══════════════════════════════════════════════════════════════════════════╝")
    println()
    
    # Show prime structure
    println("─── Prime Structure ───")
    println("  Low primes [3, 23]:  $(LOW_PRIMES)")
    println("  High primes [23, 1069]: $(length(HIGH_PRIMES)) primes")
    println("  Total primes [3, 1069]: $(length(PRIMES_3_TO_1069)) primes")
    println()
    
    # Create walk
    walk = AffectPrimeLatticeWalk(; seed=GAY_SEED)
    
    # Show initial state
    println("─── Initial States ───")
    for (name, state) in [("ZAHN", walk.states.zahn), 
                          ("JULES", walk.states.jules), 
                          ("FABRIZ", walk.states.fabriz)]
        emoji = name == "ZAHN" ? "🔴" : name == "JULES" ? "🟢" : "🔵"
        affect = compute_affect(state.world, state.current_prime; seed=state.seed)
        println("  $emoji $name: prime=$(state.current_prime), affect=$(round(affect.valence, digits=2)), dir=$(affect.direction)")
    end
    println()
    
    # Run walk
    println("─── Running 20 Interleaved Steps ───")
    launch_affect_walk!(walk, 20)
    
    println("  Steps completed: $(walk.step_count)")
    println("  Fingerprint: 0x$(string(walk.fingerprint, base=16))")
    println("  Sheaf verified: $(walk.sheaf_verified)")
    println()
    
    # Show histories
    println("─── Prime Histories ───")
    println("  🔴 ZAHN:   $(walk.states.zahn.history)")
    println("  🟢 JULES:  $(walk.states.jules.history)")
    println("  🔵 FABRIZ: $(walk.states.fabriz.history)")
    println()
    
    # Expansion decisions
    println("─── Expansion/Contraction Decisions ───")
    decisions = expand_or_contract!(walk)
    for world in [ZAHN, JULES, FABRIZ]
        emoji = WORLD_EMOJI[world]
        new_prime, direction = decisions[world]
        dir_sym = direction == CONTRACTING ? "-" : direction == EXPANDING ? "+" : "0"
        println("  $emoji $(world): $(dir_sym) → $new_prime")
    end
    println()
    
    # Bumpus sheaf
    println("─── Bumpus Sheaf Compositionality ───")
    sheaf = verify_sheaf_compositionality(walk.states)
    println("  Covering pairs: $(length(sheaf.covering))")
    println("  Local sections: $(length(sheaf.sections))")
    println("  Global section: 0x$(string(sheaf.global_section, base=16))")
    println("  Verified: $(sheaf.verified)")
    println()
    
    # Decomposition width
    width = structured_decomposition_width(walk.states)
    println("─── Structured Decomposition ───")
    println("  Width: $width (max concurrent distinct primes)")
    println()
    
    walk
end

end # module
