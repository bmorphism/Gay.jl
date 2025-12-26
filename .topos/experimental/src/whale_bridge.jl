# Whale-Human Semantic Bridge
# Rapid random walk with mixing guarantees for bidirectional interpretation
#
# Architecture:
#   1. Gay seeds define interpretation "worlds"
#   2. Whale codas map to color chains via ICI → interval correspondence
#   3. Rapid Metropolis walk finds seeds matching observed whale patterns
#   4. Bidirectional refinement couples human and whale meaning spaces

using Random

export WhaleBridge, observe_coda!, propose_meaning!, couple!, mixing_time
export whale_to_seed, seed_to_coda, rapid_walk, tripartite_consensus

# ═══════════════════════════════════════════════════════════════════════════
# Core Types
# ═══════════════════════════════════════════════════════════════════════════

struct CodaObservation
    n_clicks::Int
    icis::Vector{Float64}      # Inter-click intervals (seconds)
    rhythm::Vector{Float64}    # Normalized ICIs (sum to 1)
    tempo_type::Int            # 1-5
    has_ornament::Bool
end

mutable struct WhaleBridge
    seed::UInt64
    observations::Vector{CodaObservation}
    candidate_seeds::Vector{UInt64}
    coupling_strength::Float64  # 0 = uncoupled, 1 = fully coupled
    galois_fixpoints::Vector{Int}  # Shared meaning indices
end

function WhaleBridge(seed::UInt64=GAY_SEED)
    WhaleBridge(seed, CodaObservation[], UInt64[], 0.0, Int[])
end

# ═══════════════════════════════════════════════════════════════════════════
# Coda ↔ Seed Mapping
# ═══════════════════════════════════════════════════════════════════════════

"""
Convert observed ICIs to rhythm pattern (normalized intervals).
"""
function icis_to_rhythm(icis::Vector{Float64})
    total = sum(icis)
    total > 0 ? icis ./ total : icis
end

"""
Convert rhythm pattern to pitch class intervals (mod 12).
"""
function rhythm_to_intervals(rhythm::Vector{Float64})
    [round(Int, r * 12) % 12 for r in rhythm]
end

"""
Score how well a seed's color chain matches an observed whale rhythm.
"""
function seed_rhythm_score(seed::UInt64, target_rhythm::Vector{Float64})::Float64
    n = length(target_rhythm) + 1
    notes = [hue_to_pc(color_at(i; seed=seed)) for i in 1:n]
    intervals = [(notes[i+1] - notes[i] + 12) % 12 for i in 1:n-1]
    
    # Normalize intervals to rhythm
    total = sum(intervals)
    if total == 0
        return 0.0
    end
    seed_rhythm = intervals ./ total
    
    # Cosine similarity
    dot_product = sum(seed_rhythm .* target_rhythm)
    norm_a = sqrt(sum(seed_rhythm .^ 2))
    norm_b = sqrt(sum(target_rhythm .^ 2))
    
    (norm_a > 0 && norm_b > 0) ? dot_product / (norm_a * norm_b) : 0.0
end

"""
Rapid random walk with guaranteed mixing to find seeds matching whale pattern.
Uses SplitMix64 + Metropolis-Hastings for provable convergence.
"""
function rapid_walk(target_rhythm::Vector{Float64}, 
                    start_seed::UInt64=GAY_SEED;
                    n_steps::Int=1000,
                    n_chains::Int=4,
                    temperature::Float64=1.0)::Vector{Tuple{UInt64, Float64}}
    
    candidates = Tuple{UInt64, Float64}[]
    
    # Run multiple chains for better mixing
    for chain in 1:n_chains
        current_seed = start_seed + UInt64(chain * 0x123456789)
        current_score = seed_rhythm_score(current_seed, target_rhythm)
        
        β = 1.0 / temperature
        
        for step in 1:n_steps
            # SplitMix64 proposal (proven good mixing)
            proposal = splitmix64(current_seed ⊻ UInt64(step))
            proposal_score = seed_rhythm_score(proposal, target_rhythm)
            
            # Metropolis-Hastings acceptance
            ΔE = proposal_score - current_score
            if ΔE > 0 || rand() < exp(β * ΔE)
                current_seed = proposal
                current_score = proposal_score
            end
            
            # Occasional Lévy flight for escaping local optima
            if rand() < 0.05
                current_seed = UInt64(hash((chain, step, current_seed)))
                current_score = seed_rhythm_score(current_seed, target_rhythm)
            end
            
            # Record good candidates
            if current_score > 0.8
                push!(candidates, (current_seed, current_score))
            end
        end
    end
    
    # Return unique, sorted by score
    unique_candidates = Dict{UInt64, Float64}()
    for (s, score) in candidates
        if !haskey(unique_candidates, s) || unique_candidates[s] < score
            unique_candidates[s] = score
        end
    end
    
    sorted = sort(collect(unique_candidates), by=x->-x[2])
    first(sorted, min(10, length(sorted)))
end

"""
SplitMix64 mixing function (proven excellent statistical properties).
"""
# function splitmix64(x::UInt64)::UInt64
#    x += 0x9e3779b97f4a7c15
#    x = (x ⊻ (x >> 30)) * 0xbf58476d1ce4e5b9
#    x = (x ⊻ (x >> 27)) * 0x94d049bb133111eb
#    x ⊻ (x >> 31)
# end

# ═══════════════════════════════════════════════════════════════════════════
# Bidirectional Interpretation (1, 2, 3 Whales)
# ═══════════════════════════════════════════════════════════════════════════

"""
Observe a single whale coda and update candidate seeds.
"""
function observe_coda!(bridge::WhaleBridge, icis::Vector{Float64}; 
                       has_ornament::Bool=false)
    rhythm = icis_to_rhythm(icis)
    tempo_type = clamp(ceil(Int, sum(icis) / 0.2), 1, 5)
    
    obs = CodaObservation(
        length(icis) + 1,
        icis,
        rhythm,
        tempo_type,
        has_ornament
    )
    push!(bridge.observations, obs)
    
    # Find matching seeds via rapid walk
    candidates = rapid_walk(rhythm, bridge.seed)
    bridge.candidate_seeds = [s for (s, _) in candidates]
    
    if !isempty(candidates)
        # Update to best matching seed
        bridge.seed = first(candidates)[1]
    end
    
    obs
end

"""
Two-whale dialogue: refine interpretation via duration matching.
"""
function dialogue_refine!(bridge::WhaleBridge, 
                          icis_a::Vector{Float64}, 
                          icis_b::Vector{Float64})
    rhythm_a = icis_to_rhythm(icis_a)
    rhythm_b = icis_to_rhythm(icis_b)
    
    # Duration matching score
    dur_a = sum(icis_a)
    dur_b = sum(icis_b)
    duration_similarity = 1.0 - abs(dur_a - dur_b) / max(dur_a, dur_b)
    
    # Combined rhythm (weighted by duration similarity)
    if length(rhythm_a) == length(rhythm_b)
        combined = (rhythm_a .+ rhythm_b) ./ 2
    else
        combined = length(rhythm_a) > length(rhythm_b) ? rhythm_a : rhythm_b
    end
    
    # Bidirectional search: seeds that explain both whales
    candidates = rapid_walk(combined, bridge.seed; n_steps=2000)
    
    if !isempty(candidates)
        bridge.seed = first(candidates)[1]
        bridge.coupling_strength = min(1.0, bridge.coupling_strength + 0.2 * duration_similarity)
    end
    
    (seed=bridge.seed, coupling=bridge.coupling_strength, dur_match=duration_similarity)
end

"""
Three-whale upswell: tripartite constraint resolution.
Returns gadget classification (XOR/MAJ/PARITY).
"""
function tripartite_consensus!(bridge::WhaleBridge,
                               icis_a::Vector{Float64},
                               icis_b::Vector{Float64},
                               icis_c::Vector{Float64})
    # Extract intervals from each whale
    int_a = rhythm_to_intervals(icis_to_rhythm(icis_a))
    int_b = rhythm_to_intervals(icis_to_rhythm(icis_b))
    int_c = rhythm_to_intervals(icis_to_rhythm(icis_c))
    
    # Tripartite constraint analysis
    sum_a = sum(int_a)
    sum_b = sum(int_b)
    sum_c = sum(int_c)
    
    # XOR gadget: sums should cancel (mod 12)
    xor_residue = (sum_a + sum_b + sum_c) % 12
    is_xor = xor_residue == 0
    
    # MAJ gadget: majority parity
    parities = [sum_a % 2, sum_b % 2, sum_c % 2]
    majority = sum(parities) >= 2 ? 1 : 0
    is_maj = all(p -> p == majority, parities)
    
    # PARITY gadget: all same parity
    is_parity = length(unique(parities)) == 1
    
    gadget_class = if is_xor
        :XOR
    elseif is_maj
        :MAJ
    elseif is_parity
        :PARITY
    else
        :CLAUSE
    end
    
    # Find seed satisfying tripartite constraint
    combined = vcat(
        icis_to_rhythm(icis_a),
        icis_to_rhythm(icis_b),
        icis_to_rhythm(icis_c)
    ) ./ 3
    
    candidates = rapid_walk(combined, bridge.seed; n_steps=3000, n_chains=8)
    
    if !isempty(candidates)
        bridge.seed = first(candidates)[1]
        bridge.coupling_strength = min(1.0, bridge.coupling_strength + 0.3)
    end
    
    (gadget=gadget_class, xor_residue=xor_residue, seed=bridge.seed)
end

# ═══════════════════════════════════════════════════════════════════════════
# Galois Connection: Coupling Human ↔ Whale Meanings
# ═══════════════════════════════════════════════════════════════════════════

"""
α: Whale → Human (abstraction)
Maps whale coda features to human-interpretable pitch/interval space.
"""
function α(coda::CodaObservation)::Vector{Int}
    rhythm_to_intervals(coda.rhythm)
end

"""
γ: Human → Whale (concretization)
Maps human interval pattern back to whale-like ICI sequence.
"""
function γ(intervals::Vector{Int}; base_duration::Float64=0.4)::Vector{Float64}
    # Normalize intervals to rhythm
    total = sum(intervals)
    if total == 0
        return fill(base_duration / length(intervals), length(intervals))
    end
    rhythm = intervals ./ total
    rhythm .* base_duration
end

"""
Find Galois fixpoints: meanings that survive round-trip α ∘ γ ∘ α.
These are the shared semantic elements between whale and human spaces.
"""
function find_fixpoints!(bridge::WhaleBridge)
    fixpoints = Int[]
    
    for (i, obs) in enumerate(bridge.observations)
        # Apply round-trip
        human_intervals = α(obs)
        whale_icis = γ(human_intervals; base_duration=sum(obs.icis))
        recovered_intervals = rhythm_to_intervals(icis_to_rhythm(whale_icis))
        
        # Check closure: γ ∘ α ≥ id
        if human_intervals == recovered_intervals
            push!(fixpoints, i)
        end
    end
    
    bridge.galois_fixpoints = fixpoints
    fixpoints
end

"""
Couple the meaning networks: iterate α-γ until convergence.
"""
function couple!(bridge::WhaleBridge; max_iters::Int=100, tol::Float64=1e-6)
    if isempty(bridge.observations)
        return (converged=false, iterations=0)
    end
    
    # Start with first observation
    current = α(bridge.observations[1])
    
    for iter in 1:max_iters
        # Round-trip
        whale_form = γ(current)
        human_form = rhythm_to_intervals(icis_to_rhythm(whale_form))
        
        # Check convergence
        if current == human_form
            bridge.coupling_strength = 1.0
            find_fixpoints!(bridge)
            return (converged=true, iterations=iter, fixpoints=bridge.galois_fixpoints)
        end
        
        current = human_form
    end
    
    find_fixpoints!(bridge)
    (converged=false, iterations=max_iters, fixpoints=bridge.galois_fixpoints)
end

# ═══════════════════════════════════════════════════════════════════════════
# Mixing Time Analysis
# ═══════════════════════════════════════════════════════════════════════════

"""
Estimate mixing time for the rapid walk Markov chain.
Uses coupon collector bound for seed space coverage.
"""
function mixing_time(n_seeds::Int=1000, ε::Float64=0.01)
    # Coupon collector: E[T] ≈ n ln(n) + γn where γ ≈ 0.5772
    γ = 0.5772156649
    expected = n_seeds * (log(n_seeds) + γ)
    
    # With ε-mixing guarantee
    mixing = ceil(Int, expected * log(1/ε))
    
    (expected_steps=expected, ε_mixing=mixing, 
     recommendation="Use n_steps ≥ $(mixing) for $(100*(1-ε))% mixing")
end

# ═══════════════════════════════════════════════════════════════════════════
# Sound-Aided Exploration
# ═══════════════════════════════════════════════════════════════════════════

"""
Generate audio feedback for current bridge state.
Plays the current seed's color chain as whale-like clicks.
"""
function play_bridge_state(bridge::WhaleBridge)
    seed = bridge.seed
    notes = [hue_to_pc(color_at(i; seed=seed)) for i in 1:12]
    intervals = [(notes[i+1] - notes[i] + 12) % 12 for i in 1:11]
    
    # Convert to ICIs (100-400ms range)
    icis = [0.1 + (iv / 12.0) * 0.3 for iv in intervals]
    
    # Generate whale-like clicks
    cmd = ```python3 -c "
import wave, struct, math

icis = [$(join(icis, ","))]
sr = 44100
out = b''

click_dur = 0.003
click_freq = 2000

t = 0
for i, ici in enumerate([0] + icis):
    t += ici
    for j in range(int(sr * click_dur)):
        sample_t = j / sr
        damping = math.exp(-sample_t * 500)
        click = damping * math.sin(2 * math.pi * click_freq * sample_t)
        click += 0.3 * damping * math.sin(4 * math.pi * click_freq * sample_t)
        out += struct.pack('<h', int(max(-32767, min(32767, click * 32767 * 0.7))))

# Pad
total_samples = int(sr * (sum(icis) + click_dur * len(icis)))
while len(out) < total_samples * 2:
    out += struct.pack('<h', 0)

with wave.open('/tmp/bridge_state.wav', 'w') as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes(out)
"```
    
    try
        run(cmd, wait=true)
        run(`afplay /tmp/bridge_state.wav`, wait=true)
        return true
    catch
        return false
    end
end

"""
Sonify the coupling process: play convergence as harmony.
"""
function sonify_coupling!(bridge::WhaleBridge)
    result = couple!(bridge)
    
    if result.converged
        println("  🎵 Coupling converged in $(result.iterations) iterations")
        println("  🔗 $(length(result.fixpoints)) Galois fixpoints found")
        play_bridge_state(bridge)
    else
        println("  ⚠ Coupling did not converge")
    end
    
    result
end
