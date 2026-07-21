# Reafference Proof: predict a color stream, observe, verify continuity
#
# Von Holst & Mittelstaedt (1950): an organism predicts the sensory
# consequence of its own action. Prediction = efference copy.
# Observation = reafference. Match = self-generated. Mismatch = exafference.
#
# Applied to capability systems: witness possession of seed-derived material by
# predicting its color sequence. This is continuity evidence, not identity,
# authentication, or authority; those belong to typed referents and capabilities.

module Reafference

using Colors, ColorTypes
using ..Gay: TritTick, trit, hash_color, splitmix64, EPOCH_1_HZ, current_tick, WallClockTicks

export ReafferenceProof, predict, verify
export EphemeralSeed, ephemeral_seed, reafference_challenge
export dE2000_distance

# ═══════════════════════════════════════════════════════════════════════════
# Perceptual color distance (CIE dE2000, simplified)
# ═══════════════════════════════════════════════════════════════════════════

"""
    dE2000_distance(c1::RGB, c2::RGB) -> Float64

Approximate CIE dE2000 perceptual color distance.
Uses Colors.jl's colordiff with DE_2000 metric.
"""
function dE2000_distance(c1::RGB, c2::RGB)
    colordiff(c1, c2; metric=DE_2000())
end

function dE2000_distance(c1::Tuple, c2::Tuple)
    dE2000_distance(RGB{Float32}(c1...), RGB{Float32}(c2...))
end

# ═══════════════════════════════════════════════════════════════════════════
# Ephemeral seed rotation
# ═══════════════════════════════════════════════════════════════════════════

"""
    EphemeralSeed

A one-time seed derived from permanent identity + public randomness + counter.
Changes every interaction. Cannot be replayed.

- `permanent_seed`: derivation input associated with a typed referent
- `drand_round`: distributed randomness beacon (changes every 30 seconds)
- `interaction_counter`: monotonic (incremented per interaction)
"""
struct EphemeralSeed
    permanent_seed::UInt64
    drand_round::UInt64
    interaction_counter::UInt64
    derived::UInt64  # the actual ephemeral seed
end

"""
    ephemeral_seed(permanent, drand_round, counter) -> EphemeralSeed

Derive a one-time ephemeral seed. The derived value is:
    SplitMix64(permanent ⊕ drand_round ⊕ counter)
"""
function ephemeral_seed(permanent::UInt64, drand_round::UInt64, counter::UInt64)
    combined = xor(permanent, xor(drand_round, counter))
    derived = splitmix64(combined)
    EphemeralSeed(permanent, drand_round, counter, derived)
end

# ═══════════════════════════════════════════════════════════════════════════
# Reafference proof
# ═══════════════════════════════════════════════════════════════════════════

"""
    ReafferenceProof

A completed proof of seed-derived color-stream continuity. Contains:
- The challenge (which ticks to predict)
- The predictions (what the prover claimed)
- The observations (what was actually seen)
- The verification result (did they match within tolerance?)
"""
struct ReafferenceProof
    seed::UInt64
    challenge_ticks::Vector{TritTick}
    predicted::Vector{Tuple{Float32,Float32,Float32}}
    observed::Vector{Tuple{Float32,Float32,Float32}}
    distances::Vector{Float64}    # dE2000 per pair
    threshold::Float64            # max acceptable dE2000
    verified::Bool                # all distances < threshold?
end

"""
    predict(seed::UInt64, ticks::Vector{TritTick}) -> Vector{Tuple{Float32,Float32,Float32}}

Efference copy: predict what colors you will produce at the given ticks.
"""
function predict(seed::UInt64, ticks::Vector{TritTick})
    [hash_color(seed, t) for t in ticks]
end

"""
    reafference_challenge(n::Int=5; seed_for_challenge::UInt64=rand(UInt64)) -> Vector{TritTick}

Generate a random challenge: n tick positions to predict.
The challenge itself is random (unpredictable to the prover),
but the prover's prediction at each tick is deterministic from their seed.
"""
function reafference_challenge(n::Int=5; seed_for_challenge::UInt64=rand(UInt64))
    ticks = TritTick[]
    state = seed_for_challenge
    for _ in 1:n
        state = splitmix64(state)
        # Generate tick values within a reasonable range (one day of Epoch 1)
        tick_val = state % (EPOCH_1_HZ * 86400)
        push!(ticks, TritTick(tick_val))
    end
    ticks
end

"""
    verify(seed, ticks, predicted, observed; threshold=25.0) -> ReafferenceProof

Complete reafference verification:
1. Recompute the deterministic predictions from `seed` and `ticks`
2. Reject claimed predictions that differ from that derived stream
3. Compute dE2000 distance between each predicted/observed pair
4. Check all distances are below threshold
5. Return the proof (pass or fail)

A passing result witnesses agreement with the supplied seed and observations.
It does not authenticate a referent or grant a capability.

Default threshold 25.0 dE2000 — generous for display variation.
Tighten to 5.0 for calibrated displays, 1.0 for reference monitors.
"""
function verify(seed::UInt64,
                ticks::Vector{TritTick},
                predicted::Vector{<:Tuple},
                observed::Vector{<:Tuple};
                threshold::Float64=25.0)

    @assert length(ticks) == length(predicted) == length(observed)

    distances = Float64[]
    for (p, o) in zip(predicted, observed)
        push!(distances, dE2000_distance(p, o))
    end

    expected = predict(seed, ticks)
    prediction_matches_seed = predicted == expected
    all_pass = prediction_matches_seed && all(d -> d < threshold, distances)

    ReafferenceProof(seed, ticks, predicted, observed, distances, threshold, all_pass)
end

"""
    verify(seed, ticks, observed; threshold=25.0) -> ReafferenceProof

Convenience: predict from seed, then verify against observed.
"""
function verify(seed::UInt64,
                ticks::Vector{TritTick},
                observed::Vector{<:Tuple};
                threshold::Float64=25.0)
    predicted = predict(seed, ticks)
    verify(seed, ticks, predicted, observed; threshold=threshold)
end

function Base.show(io::IO, proof::ReafferenceProof)
    status = proof.verified ? "VERIFIED" : "FAILED"
    mean_d = isempty(proof.distances) ? 0.0 : sum(proof.distances) / length(proof.distances)
    max_d = isempty(proof.distances) ? 0.0 : maximum(proof.distances)
    print(io, "ReafferenceProof($(status), n=$(length(proof.challenge_ticks)), ",
          "mean_dE=$(round(mean_d, digits=2)), max_dE=$(round(max_d, digits=2)), ",
          "threshold=$(proof.threshold))")
end

end # module Reafference
