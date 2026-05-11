module OpenPriors

using ..Gay: GAY_SEED, hash_color, splitmix64_mix

export OpenPrior, SecretColorMask, SituationalFilter
export prior_seed, mask_color, situational_reveal
export bayesian_update_seed, drand_prior, elicit_prior
export prior_palette, masked_palette, free_energy_mask

# The open prior: a publicly verifiable seed that generates deterministic streams
struct OpenPrior
    seed::UInt64
    round::Int           # drand round (0 = not from beacon)
    description::String
    h_max::Float64       # maximum entropy of this prior
end

# Secret color mask: filters which colors are visible based on situation
struct SecretColorMask
    brain_state::Symbol   # :focus :alert :rest :fatigue :flow
    valence::Float64      # [-1, +1]
    threshold::Float64    # visibility threshold
end

# Situational filter: combines prior + mask
struct SituationalFilter
    prior::OpenPrior
    mask::SecretColorMask
    revealed::Vector{String}   # hex colors that passed the mask
    hidden::Vector{String}     # hex colors masked out
end

# Generate seed from entropy measurement
function prior_seed(entropy::Float64; h_max::Float64=3.0)
    # Map entropy [0, h_max] to seed space
    # Maximum interaction entropy -> maximum spread across seed space
    normalized = clamp(entropy / h_max, 0.0, 1.0)
    seed_bits = round(UInt64, normalized * typemax(UInt64))
    return splitmix64_mix(seed_bits)
end

# drand beacon as open prior -- anyone with round can verify
function drand_prior(round::Int, randomness_hex::String; desc::String="drand beacon")
    seed = parse(UInt64, randomness_hex[1:16], base=16)
    OpenPrior(seed, round, desc, log(2, 256))  # 8 bits entropy per hex char
end

# Elicit prior from observation: "I see this color, what seed could produce it?"
# This is abductive inference -- recovering cause from effect
function elicit_prior(observed_hex::String; search_depth::Int=1000)
    target_r = parse(Int, observed_hex[2:3], base=16) / 255.0
    target_g = parse(Int, observed_hex[4:5], base=16) / 255.0
    target_b = parse(Int, observed_hex[6:7], base=16) / 255.0

    best_seed = UInt64(0)
    best_idx = 0
    best_dist = Inf

    for candidate_seed in UInt64(1):UInt64(search_depth)
        state = candidate_seed
        for idx in 1:search_depth
            state = splitmix64_mix(state)
            r, g, b = hash_color(state)
            dist = sqrt((r - target_r)^2 + (g - target_g)^2 + (b - target_b)^2)
            if dist < best_dist
                best_dist = dist
                best_seed = candidate_seed
                best_idx = idx
            end
        end
    end

    return (seed=best_seed, index=best_idx, distance=best_dist,
            prior=OpenPrior(best_seed, 0, "elicited from $observed_hex", -log(2, best_dist + 1e-10)))
end

# Generate palette from open prior
function prior_palette(prior::OpenPrior, n::Int)
    colors = String[]
    state = prior.seed
    for i in 1:n
        state = splitmix64_mix(state)
        r, g, b = hash_color(state)
        hex = string("#", uppercase(string(round(Int, r*255), base=16, pad=2)),
                          uppercase(string(round(Int, g*255), base=16, pad=2)),
                          uppercase(string(round(Int, b*255), base=16, pad=2)))
        push!(colors, hex)
    end
    colors
end

# Apply secret color mask: filter colors by brain_state + valence
function mask_color(hex::String, mask::SecretColorMask)
    r = parse(Int, hex[2:3], base=16) / 255.0
    g = parse(Int, hex[4:5], base=16) / 255.0
    b = parse(Int, hex[6:7], base=16) / 255.0

    # Compute situational energy based on brain_state
    energy = if mask.brain_state == :focus
        0.3 * r + 0.59 * g + 0.11 * b  # luminance (standard attention)
    elseif mask.brain_state == :alert
        r  # red channel dominates alertness
    elseif mask.brain_state == :rest
        b  # blue channel for rest/calm
    elseif mask.brain_state == :fatigue
        1.0 - (0.3 * r + 0.59 * g + 0.11 * b)  # inverse luminance
    elseif mask.brain_state == :flow
        (r + g + b) / 3.0  # uniform attention
    else
        0.5
    end

    # Valence shifts the threshold: positive valence reveals more
    adjusted_threshold = mask.threshold - (mask.valence * 0.3)

    return energy >= adjusted_threshold
end

# Apply mask to full palette, returns SituationalFilter
function masked_palette(prior::OpenPrior, mask::SecretColorMask, n::Int)
    all_colors = prior_palette(prior, n)
    revealed = String[]
    hidden = String[]

    for hex in all_colors
        if mask_color(hex, mask)
            push!(revealed, hex)
        else
            push!(hidden, hex)
        end
    end

    SituationalFilter(prior, mask, revealed, hidden)
end

# Free energy mask: prediction error between expected and observed colors
# High free energy = surprising = masked. Low free energy = expected = revealed.
function free_energy_mask(predicted_hex::String, observed_hex::String)
    pr = parse(Int, predicted_hex[2:3], base=16) / 255.0
    pg = parse(Int, predicted_hex[4:5], base=16) / 255.0
    pb = parse(Int, predicted_hex[6:7], base=16) / 255.0
    or = parse(Int, observed_hex[2:3], base=16) / 255.0
    og = parse(Int, observed_hex[4:5], base=16) / 255.0
    ob = parse(Int, observed_hex[6:7], base=16) / 255.0

    prediction_error = sqrt((pr - or)^2 + (pg - og)^2 + (pb - ob)^2)
    # Normalize to [0, sqrt(3)] -> [0, 1]
    normalized = prediction_error / sqrt(3.0)

    (free_energy=normalized,
     surprising=normalized > 0.5,
     action=normalized > 0.5 ? :update_prior : :maintain,
     description=normalized > 0.5 ?
       "High surprise: mask this color, update prior" :
       "Low surprise: reveal this color, prior confirmed")
end

# Bayesian update: combine prior seed with new evidence
function bayesian_update_seed(prior::OpenPrior, evidence_hex::String)
    evidence_bits = parse(UInt64, evidence_hex[2:7], base=16)
    # XOR fold: prior and evidence combine
    new_seed = xor(prior.seed, splitmix64_mix(evidence_bits))
    # Entropy decreases: we learned something
    new_h = prior.h_max * 0.9  # 10% entropy reduction per observation
    OpenPrior(new_seed, 0, "$(prior.description) + evidence $evidence_hex", new_h)
end

# Situational reveal: given a game state from secret_colors.json,
# use open prior to decide what's visible
function situational_reveal(prior::OpenPrior, secret_colors::Vector;
                           brain_state::Symbol=:focus, valence::Float64=0.0)
    mask = SecretColorMask(brain_state, valence, 0.5)
    filter = masked_palette(prior, mask, length(secret_colors))

    # Cross-reference: which secret colors match the prior's revealed set?
    visible = []
    for sc in secret_colors
        hex = sc isa Dict ? get(sc, "hex_color", "") : string(sc)
        if hex in filter.revealed
            push!(visible, sc)
        end
    end

    (visible=visible,
     total=length(secret_colors),
     revealed_count=length(visible),
     mask_ratio=1.0 - length(visible) / max(length(secret_colors), 1),
     prior=prior,
     mask=mask)
end

end # module
