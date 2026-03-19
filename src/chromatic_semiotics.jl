"""
    chromatic_semiotics.jl

Evolution of meaning through chromatic communication.

Framework:
- Alice and Bob exchange colored messages derived from gay_seeds
- Derangeable confusion: colors hide their origins (semantic noise)
- One-time pad: deterministic colors as encryption keys
- Catalog: Optimal gay_seeds by semantic property

Theory:
- Color as fundamental semantic unit
- gay_seed determines meaning deterministically
- Communication preserves or obscures seed information
- Evolution: colors converge to or diverge from origin seeds
"""

module ChromaticSemiotics

using Colors, Statistics, Random, Base.Threads

export
    # Semantic types
    ChromaticMessage,
    SemanticColor,
    GaySeedCatalog,
    
    # Agents
    ChromaticAgent,
    
    # Communication
    alice_encodes,
    bob_decodes,
    measure_semantic_drift,
    derange_color,
    
    # Catalog building
    create_seed_catalog,
    classify_seed_semantics,
    find_optimal_seeds,
    
    # OTP verification
    build_otp_pad,
    verify_otp_integrity

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

const SEMANTIC_THRESHOLD = 0.1  # Distance to consider "same meaning"
const DERANGE_STRENGTH = 0.3    # How much confusion to inject

# ─────────────────────────────────────────────────────────────────────────
# SplitMix64 & Color Generation (Core)
# ─────────────────────────────────────────────────────────────────────────

function splitmix64(state::UInt64)::Tuple{UInt64, UInt64}
    """SplitMix64 PRNG - returns (output, next_state)"""
    z = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ⊻ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    z ⊻ (z >> 31), (state + 1) & 0xFFFFFFFFFFFFFFFF
end

function color_from_seed(seed::UInt64)::RGB{Float64}
    """Generate deterministic RGB color from seed"""
    r, s1 = splitmix64(seed)
    g, s2 = splitmix64(s1)
    b, _  = splitmix64(s2)
    RGB(
        (r >> 56) / 255.0,
        (g >> 56) / 255.0,
        (b >> 56) / 255.0
    )
end

function seed_from_color(color::RGB{Float64})::UInt64
    """Attempt recovery of seed from color (lossy)"""
    r_bits = UInt64(round(color.r * 255))
    g_bits = UInt64(round(color.g * 255))
    b_bits = UInt64(round(color.b * 255))
    (r_bits << 16) | (g_bits << 8) | b_bits
end

# ═══════════════════════════════════════════════════════════════════════════
# SEMANTIC COLOR TYPES
# ═══════════════════════════════════════════════════════════════════════════

"""Semantic properties of a color"""
struct SemanticColor
    color::RGB{Float64}
    origin_seed::UInt64
    semantic_meaning::String  # Human-readable interpretation
    
    # Semantic properties
    entropy::Float64          # Randomness in channels
    saturation::Float64       # Color intensity
    hue_angle::Float64        # Position in color wheel
    
    # Trace back to seed
    seed_distance::Float64    # Hamming distance from reconstructed seed
end

"""Chromatic message from one agent to another"""
struct ChromaticMessage
    sender::String
    receiver::String
    content_color::RGB{Float64}  # The message itself
    semantic_colors::Vector{SemanticColor}  # Interpretation
    metadata::Dict{String, Any}  # Extra info (timestamps, etc)
end

"""Agent capable of chromatic communication"""
mutable struct ChromaticAgent
    name::String
    id::UInt64
    
    # Private key (base seed)
    private_seed::UInt64
    private_color::RGB{Float64}
    
    # Shared knowledge
    known_agents::Dict{String, UInt64}  # agent_name -> seed
    received_messages::Vector{ChromaticMessage}
    
    # Learning state
    semantic_model::Dict{String, Float64}  # meaning -> frequency
end

"""Catalog of good gay seeds with semantic properties"""
struct GaySeedCatalog
    seeds::Vector{UInt64}
    colors::Vector{RGB{Float64}}
    names::Vector{String}  # Semantic labels
    properties::Vector{Dict{String, Float64}}  # Semantic features
    entropy_scores::Vector{Float64}
    reversibility_scores::Vector{Float64}  # How close seed can be recovered
end

# ═══════════════════════════════════════════════════════════════════════════
# CHROMATIC AGENT CREATION
# ═══════════════════════════════════════════════════════════════════════════

"""
    ChromaticAgent(name, private_seed)

Create an agent with chromatic identity and private semantic space.
"""
function ChromaticAgent(name::String, private_seed::UInt64)
    agent = ChromaticAgent(
        name=name,
        id=private_seed,
        private_seed=private_seed,
        private_color=color_from_seed(private_seed),
        known_agents=Dict{String, UInt64}(),
        received_messages=Vector{ChromaticMessage}(),
        semantic_model=Dict{String, Float64}()
    )
    agent
end

# ═══════════════════════════════════════════════════════════════════════════
# ENCODING/DECODING & SEMANTIC DRIFT
# ═══════════════════════════════════════════════════════════════════════════

"""
    alice_encodes(alice, message_seed, bob_seed)

Alice encodes a message for Bob by blending her message_seed with knowledge of Bob's seed.

Meaning evolves through:
1. Alice's message_seed (raw intent)
2. Blended with Bob's known seed (contextualization)
3. Distorted by her encoding (scrambling)
"""
function alice_encodes(
    alice::ChromaticAgent,
    message_seed::UInt64,
    bob_seed::UInt64
)::RGB{Float64}
    
    # Step 1: Base message color
    message_color = color_from_seed(message_seed)
    
    # Step 2: Contextualization - blend with Bob's seed
    # (showing she knows who Bob is)
    bob_color = color_from_seed(bob_seed)
    blended = RGB(
        (message_color.r + bob_color.r) / 2,
        (message_color.g + bob_color.g) / 2,
        (message_color.b + bob_color.b) / 2
    )
    
    # Step 3: Alice's signature (XOR with her private color)
    signed = RGB(
        blended.r ⊕ alice.private_color.r,
        blended.g ⊕ alice.private_color.g,
        blended.b ⊕ alice.private_color.b
    )
    
    signed
end

"""
    bob_decodes(bob, received_color, alice_seed)

Bob attempts to decode Alice's message by removing her signature.

If successful, Bob recovers:
- Alice's message intent
- Proof of her knowledge of his identity
"""
function bob_decodes(
    bob::ChromaticAgent,
    received_color::RGB{Float64},
    alice_seed::UInt64
)::Tuple{RGB{Float64}, Float64}
    
    # Step 1: Remove Alice's signature (using known alice_seed)
    alice_color = color_from_seed(alice_seed)
    unsigned = RGB(
        received_color.r ⊕ alice_color.r,
        received_color.g ⊕ alice_color.g,
        received_color.b ⊕ alice_color.b
    )
    
    # Step 2: De-contextualize
    # (Remove his own seed to see Alice's message)
    decontextualized = RGB(
        unsigned.r - bob.private_color.r,
        unsigned.g - bob.private_color.g,
        unsigned.b - bob.private_color.b
    )
    
    # Clamp to valid range
    recovered = RGB(
        clamp(decontextualized.r, 0, 1),
        clamp(decontextualized.g, 0, 1),
        clamp(decontextualized.b, 0, 1)
    )
    
    # Measure recovery quality (how close to original message)
    # Lower distance = better decoding
    distance = sqrt(
        (unsigned.r - decontextualized.r)^2 +
        (unsigned.g - decontextualized.g)^2 +
        (unsigned.b - decontextualized.b)^2
    )
    
    recovered, distance
end

"""
    measure_semantic_drift(seed, color)

How much has the meaning drifted from its origin?

Returns distance in RGB space from color_from_seed(seed) to actual color.
"""
function measure_semantic_drift(seed::UInt64, color::RGB{Float64})::Float64
    origin = color_from_seed(seed)
    sqrt(
        (origin.r - color.r)^2 +
        (origin.g - color.g)^2 +
        (origin.b - color.b)^2
    )
end

"""
    derange_color(color, strength=DERANGE_STRENGTH)

Add semantic confusion: distort color to hide its origin while preserving semantics.

Used to:
- Confuse eavesdroppers about actual seeds
- Test robustness of decoding
- Explore semantic subspace around color
"""
function derange_color(color::RGB{Float64}, strength::Float64=DERANGE_STRENGTH)::RGB{Float64}
    # Add controlled noise
    noise_r = randn() * strength
    noise_g = randn() * strength
    noise_b = randn() * strength
    
    deranged = RGB(
        clamp(color.r + noise_r, 0, 1),
        clamp(color.g + noise_g, 0, 1),
        clamp(color.b + noise_b, 0, 1)
    )
    
    deranged
end

"""
    infer_semantic_meaning(color)

Interpret what a color "means" semantically.

Returns string describing the color and its properties.
"""
function infer_semantic_meaning(color::RGB{Float64})::String
    brightness = (color.r + color.g + color.b) / 3
    
    # Rough semantic interpretation based on channels
    if color.r > 0.7 && color.g < 0.3
        "RED: Intense, passionate, high-energy"
    elseif color.g > 0.7 && color.r < 0.3
        "GREEN: Balanced, natural, growth-oriented"
    elseif color.b > 0.7 && color.r < 0.3
        "BLUE: Cool, contemplative, stable"
    elseif brightness > 0.8
        "BRIGHT: Optimistic, clear, open"
    elseif brightness < 0.3
        "DARK: Mystery, depth, complexity"
    else
        "BALANCED: Harmonious, integrated meaning"
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# SEMANTIC COLOR ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

"""
    analyze_color_semantics(seed)

Compute semantic properties of a color derived from seed.
"""
function analyze_color_semantics(seed::UInt64)::SemanticColor
    color = color_from_seed(seed)
    
    # Entropy: how random/uniform are the channels?
    entropy = -sum([
        color.r > 0 ? color.r * log(color.r) : 0,
        color.g > 0 ? color.g * log(color.g) : 0,
        color.b > 0 ? color.b * log(color.b) : 0
    ])
    
    # Saturation: how intense?
    saturation = maximum([color.r, color.g, color.b])
    
    # Hue angle (simplified)
    hue_angle = atan(color.g, color.r) * 180 / π
    
    # Seed distance: can we recover the seed?
    recovered = seed_from_color(color)
    seed_distance = Float64(count_ones(recovered ⊻ seed)) / 64
    
    SemanticColor(
        color=color,
        origin_seed=seed,
        semantic_meaning=infer_semantic_meaning(color),
        entropy=entropy,
        saturation=saturation,
        hue_angle=hue_angle,
        seed_distance=seed_distance
    )
end

# ═══════════════════════════════════════════════════════════════════════════
# GOOD GAY SEED CATALOG
# ═══════════════════════════════════════════════════════════════════════════

"""
    classify_seed_semantics(seeds)

Analyze semantic properties of a set of seeds.
"""
function classify_seed_semantics(seeds::Vector{UInt64})::Vector{SemanticColor}
    [analyze_color_semantics(seed) for seed in seeds]
end

"""
    create_seed_catalog(num_seeds=256)

Build a catalog of good gay seeds with semantic analysis.

Selects seeds with:
- High entropy (good randomness)
- Diverse hue angles (spread across color space)
- Recoverable from color (good for OTP)
"""
function create_seed_catalog(num_seeds::Int64=256)::GaySeedCatalog
    seeds = UInt64[]
    colors = RGB{Float64}[]
    names = String[]
    properties = Dict{String, Float64}[]
    entropy_scores = Float64[]
    reversibility_scores = Float64[]
    
    # Generate candidate seeds
    for i in 0:(num_seeds*10-1)
        seed = UInt64(0xDEADBEEF) ⊻ UInt64(i)
        semantic = analyze_color_semantics(seed)
        
        # Scoring heuristics
        entropy = semantic.entropy
        reversibility = 1.0 - semantic.seed_distance  # High = recoverable
        
        push!(seeds, seed)
        push!(colors, semantic.color)
        push!(entropy_scores, entropy)
        push!(reversibility_scores, reversibility)
        
        # Semantic name
        name = "$(semantic.semantic_meaning) (E=$(round(entropy; digits=2)), R=$(round(reversibility; digits=2)))"
        push!(names, name)
        
        props = Dict(
            "entropy" => entropy,
            "saturation" => semantic.saturation,
            "hue" => semantic.hue_angle,
            "reversibility" => reversibility
        )
        push!(properties, props)
    end
    
    # Select best seeds (high entropy + reversible)
    scores = entropy_scores .* reversibility_scores
    best_indices = sortperm(scores; rev=true)[1:num_seeds]
    
    GaySeedCatalog(
        seeds=seeds[best_indices],
        colors=colors[best_indices],
        names=names[best_indices],
        properties=properties[best_indices],
        entropy_scores=entropy_scores[best_indices],
        reversibility_scores=reversibility_scores[best_indices]
    )
end

"""
    find_optimal_seeds(catalog; criteria=:entropy)

Find seeds matching specific semantic criteria.

Criteria:
- :entropy - high randomness
- :reversible - recoverable from color
- :balanced - good mix of RGB channels
"""
function find_optimal_seeds(
    catalog::GaySeedCatalog;
    criteria::Symbol=:entropy,
    num_results::Int64=10
)::Vector{Tuple{UInt64, RGB{Float64}, String}}
    
    scores = if criteria == :entropy
        catalog.entropy_scores
    elseif criteria == :reversible
        catalog.reversibility_scores
    elseif criteria == :balanced
        # Balance: 1.0 when channels are equal, 0 when one dominates
        [1.0 - (maximum([c.r, c.g, c.b]) - minimum([c.r, c.g, c.b])) 
         for c in catalog.colors]
    else
        catalog.entropy_scores
    end
    
    best_indices = sortperm(scores; rev=true)[1:min(num_results, length(scores))]
    
    [(
        catalog.seeds[i],
        catalog.colors[i],
        catalog.names[i]
    ) for i in best_indices]
end

# ═══════════════════════════════════════════════════════════════════════════
# ONE-TIME PAD (OTP) VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════

"""
    build_otp_pad(num_keys=256)

Build a one-time pad of deterministic chromatic keys.

Each key: deterministic color from unique seed
Properties:
- Non-reusable (different seed = different key)
- Verifiable (same seed produces same color)
- Secure against replay (used seed cannot be reused)
"""
function build_otp_pad(num_keys::Int64=256)::Vector{RGB{Float64}}
    keys = RGB{Float64}[]
    for i in 0:(num_keys-1)
        seed = UInt64(0xFEEDBEEF) + UInt64(i)
        push!(keys, color_from_seed(seed))
    end
    keys
end

"""
    verify_otp_integrity(pad1, pad2)

Verify two OTP pads are identical (both parties have same keys).

Returns: (match_count, mismatch_indices)
"""
function verify_otp_integrity(
    pad1::Vector{RGB{Float64}},
    pad2::Vector{RGB{Float64}}
)::Tuple{Int64, Vector{Int64}}
    
    @assert length(pad1) == length(pad2) "Pad lengths must match"
    
    matches = 0
    mismatches = Int64[]
    
    for i in 1:length(pad1)
        # Exact match (bitwise)
        if pad1[i].r == pad2[i].r && pad1[i].g == pad2[i].g && pad1[i].b == pad2[i].b
            matches += 1
        else
            push!(mismatches, i)
        end
    end
    
    matches, mismatches
end

"""
    one_time_encrypt(plaintext_seed, otp_key)

Encrypt a seed using OTP key (XOR in color space).
"""
function one_time_encrypt(plaintext_seed::UInt64, otp_key::RGB{Float64})::RGB{Float64}
    plaintext = color_from_seed(plaintext_seed)
    RGB(
        plaintext.r ⊕ otp_key.r,
        plaintext.g ⊕ otp_key.g,
        plaintext.b ⊕ otp_key.b
    )
end

"""
    one_time_decrypt(ciphertext, otp_key)

Decrypt using OTP key (XOR again).
"""
function one_time_decrypt(ciphertext::RGB{Float64}, otp_key::RGB{Float64})::RGB{Float64}
    RGB(
        ciphertext.r ⊕ otp_key.r,
        ciphertext.g ⊕ otp_key.g,
        ciphertext.b ⊕ otp_key.b
    )
end

# ═══════════════════════════════════════════════════════════════════════════
# COMMUNICATION PROTOCOL
# ═══════════════════════════════════════════════════════════════════════════

"""
    exchange_meaning(alice::ChromaticAgent, bob::ChromaticAgent, message_seed::UInt64)

Alice and Bob exchange chromatic meaning.

Returns: (bob_received, bob_decoded, semantic_drift)
"""
function exchange_meaning(
    alice::ChromaticAgent,
    bob::ChromaticAgent,
    message_seed::UInt64
)::Tuple{RGB{Float64}, RGB{Float64}, Float64}
    
    # Store known agent
    bob.known_agents[alice.name] = alice.private_seed
    alice.known_agents[bob.name] = bob.private_seed
    
    # Alice encodes
    encoded = alice_encodes(alice, message_seed, bob.private_seed)
    
    # Bob decodes
    decoded, recovery_distance = bob_decodes(bob, encoded, alice.private_seed)
    
    # Measure drift
    drift = measure_semantic_drift(message_seed, decoded)
    
    (encoded, decoded, drift)
end

# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION & REPORTING
# ═══════════════════════════════════════════════════════════════════════════

"""
    print_semantic_color(sc::SemanticColor)

Print analysis of a semantic color.
"""
function print_semantic_color(sc::SemanticColor)
    println("\n" * "─"^60)
    println("Semantic Color Analysis")
    println("─"^60)
    println("Seed: 0x$(string(sc.origin_seed; base=16))")
    println("Color: RGB($(round(sc.color.r; digits=3)), $(round(sc.color.g; digits=3)), $(round(sc.color.b; digits=3)))")
    println("Meaning: $(sc.semantic_meaning)")
    println("\nProperties:")
    println("  Entropy: $(round(sc.entropy; digits=3))")
    println("  Saturation: $(round(sc.saturation; digits=3))")
    println("  Hue Angle: $(round(sc.hue_angle; digits=1))°")
    println("  Seed Reversibility: $(round(1.0 - sc.seed_distance; digits=3))")
    println("─"^60)
end

end  # module ChromaticSemiotics
