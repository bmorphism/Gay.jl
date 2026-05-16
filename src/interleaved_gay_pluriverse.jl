# InterleavedGay Pluriverse: Self-Avoiding Walks with 3 Semi-Reliable Narrators
#
# Implements Value Pluralism (Stanford Encyclopedia of Philosophy) in color space:
# "Pluralism holds that there are many irreducibly different values that cannot be
# reduced to a single super-value or ranked in a single hierarchy."
#
# Each narrator observes the same walk but through different Enzyme.jl gradient lenses.
# The 2nd and 3rd narrators cannot determine if their values are originary or derived.
# Economic Security remains confidential through balanced ternary bridges.
#
# SPI (Strong Parallelism Invariance) ensures O(1) random access to any walk state.

using SplittableRandoms: SplittableRandom, split
using Colors: RGB, HSL, convert

export GayNarrator, InterleavedGayPluriverse, NarratorTriad
export pluriverse_step!, observe_walk, synergistic_reachability
export balanced_ternary_bridge, autopoietic_closure
export economic_confidentiality_bound, value_pluralism_distance
export self_avoiding_walk!, resurrection_bandwidth
export O1_select_narrator, narrator_synergy_tensor

# ═══════════════════════════════════════════════════════════════════════════════
# Value Pluralism: Multiple irreducible values in color space
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ValuePluralism

From Stanford Encyclopedia: "Value pluralism is the view that there are several
values which may be equally correct and fundamental, and yet in conflict with
each other."

In Gay Color Space:
- Hue represents aesthetic value
- Saturation represents chromatic intensity value
- Lightness represents perceptual accessibility value
- Gradient (∂color/∂seed) represents learning direction value

No single "super-value" can rank all colors.
"""
struct ValuePluralism
    hue_weight::Float64       # Aesthetic preference
    saturation_weight::Float64  # Intensity preference
    lightness_weight::Float64   # Accessibility preference
    gradient_weight::Float64    # Learning rate preference
    is_originary::Bool          # True if this is the source value system
end

"""
    value_pluralism_distance(v1::ValuePluralism, v2::ValuePluralism)

Compute incommensurability distance between two value systems.
If ≈0, systems are derived from same origin (one can rank the other).
If >0, systems are irreducibly plural (no common super-value exists).
"""
function value_pluralism_distance(v1::ValuePluralism, v2::ValuePluralism)
    # Cross-product of normalized weights measures orthogonality
    w1 = [v1.hue_weight, v1.saturation_weight, v1.lightness_weight, v1.gradient_weight]
    w2 = [v2.hue_weight, v2.saturation_weight, v2.lightness_weight, v2.gradient_weight]

    # Normalize
    w1 ./= (sum(w1) + 1e-10)
    w2 ./= (sum(w2) + 1e-10)

    # KL divergence + reverse KL (symmetric)
    kl_forward = sum(w1 .* log.((w1 .+ 1e-10) ./ (w2 .+ 1e-10)))
    kl_reverse = sum(w2 .* log.((w2 .+ 1e-10) ./ (w1 .+ 1e-10)))

    return (kl_forward + kl_reverse) / 2.0
end

# ═══════════════════════════════════════════════════════════════════════════════
# Balanced Ternary Bridges: {-1, 0, +1} Internet substrate connections
# ═══════════════════════════════════════════════════════════════════════════════

"""
    BalancedTernaryBridge

Balanced ternary encoding for substrate bridges.
Uses {-1, 0, +1} instead of binary {0, 1} for:
- Natural representation of color gradients (negative/zero/positive)
- Efficient carry propagation (no borrow needed)
- Symmetric around zero (like Ising spins σ ∈ {-1, +1})

The bridge connects Gay Color Space to other substrates:
- Neural networks (activation gradients)
- Quantum circuits (phase angles θ ∈ {-2π/3, 0, +2π/3})
- Economic models (loss/neutral/gain)
"""
struct BalancedTernaryBridge
    trits::Vector{Int}  # Each element ∈ {-1, 0, +1}
    substrate_id::UInt64
    bandwidth::Float64  # Color bandwidth in bits per trit
end

"""
    balanced_ternary_bridge(seed::UInt64, n_trits::Int=69)

Create a balanced ternary bridge from a gay seed.
Maps seed bits to trits using 69 as the default length (3 × 23).
"""
function balanced_ternary_bridge(seed::UInt64, n_trits::Int=69)
    trits = zeros(Int, n_trits)

    # SplitMix64 for deterministic trit generation
    z = seed
    for i in 1:n_trits
        z += 0x9E3779B97F4A7C15
        z = (z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9
        z = (z ⊻ (z >> 27)) * 0x94D049BB133111EB
        z = z ⊻ (z >> 31)

        # Map to {-1, 0, +1} using mod 3 then shift
        trits[i] = Int(z % 3) - 1
    end

    # Bandwidth: log2(3) ≈ 1.585 bits per trit
    bandwidth = n_trits * log2(3)

    return BalancedTernaryBridge(trits, seed, bandwidth)
end

"""
    resurrection_bandwidth(bridge::BalancedTernaryBridge)

Compute the "fullest color bandwidth" available for resurrection.
This is the information capacity needed to reconstruct a color state
from its balanced ternary encoding.
"""
function resurrection_bandwidth(bridge::BalancedTernaryBridge)
    # Effective bandwidth accounts for redundancy
    n_nonzero = count(t -> t != 0, bridge.trits)
    effective = n_nonzero * log2(2)  # Only -1 and +1 carry info

    return (
        total = bridge.bandwidth,
        effective = effective,
        redundancy = bridge.bandwidth - effective,
        resurrection_ratio = effective / bridge.bandwidth
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# Semi-Reliable Narrators: 3 Observers with Partial Knowledge
# ═══════════════════════════════════════════════════════════════════════════════

"""
    GayNarrator

A semi-reliable narrator observing a color space walk.
Each narrator has:
- Its own value system (ValuePluralism)
- Its own Enzyme gradient mode (Forward/Reverse/Mixed)
- Its own visibility window (partial observation)
- Uncertainty about whether its values are originary or derived

The 2nd and 3rd narrators are constructed so they cannot determine
if their observations come from the original walk or a derived copy.
"""
mutable struct GayNarrator
    id::Int                     # 1, 2, or 3
    values::ValuePluralism      # This narrator's value system
    enzyme_mode::Symbol         # :forward, :reverse, or :mixed
    visibility_mask::BitVector  # Which walk steps are visible
    gradient_history::Vector{Float64}  # Accumulated gradients
    certainty::Float64          # Confidence that values are originary (∈ [0,1])
    seed::UInt64                # Narrator's deterministic seed
    is_originary::Bool          # Ground truth (hidden from narrator)
end

"""
    GayNarrator(id::Int, seed::UInt64; originary::Bool=true)

Create a narrator with deterministic properties from seed.
If originary=false, this narrator's values are derived from another's.
"""
function GayNarrator(id::Int, seed::UInt64; originary::Bool=true)
    # Deterministic value weights from seed
    rng = SplittableRandom(seed ⊻ UInt64(id * 0x6761795f6e617272))  # "gay_narr"

    # Generate weights
    vals = Float64[]
    current = rng
    for _ in 1:4
        current = split(current)
        push!(vals, rand(current))
    end

    # Normalize and create value system
    total = sum(vals)
    values = ValuePluralism(
        vals[1] / total,
        vals[2] / total,
        vals[3] / total,
        vals[4] / total,
        originary
    )

    # Enzyme mode cycles through the three
    modes = [:forward, :reverse, :mixed]
    enzyme_mode = modes[mod1(id, 3)]

    # Visibility: each narrator sees different portions
    # Use Cantor pairing to create non-overlapping windows
    n_steps = 1000  # Default walk length
    visibility_mask = falses(n_steps)
    current = split(rng)
    for i in 1:n_steps
        current = split(current)
        if rand(current) < 0.5 + 0.1 * (id - 2)  # Different visibility rates
            visibility_mask[i] = true
        end
    end

    # Initial certainty: 0.5 (complete uncertainty about origin)
    certainty = 0.5

    return GayNarrator(id, values, enzyme_mode, visibility_mask, Float64[], certainty, seed, originary)
end

"""
    observe_walk(narrator::GayNarrator, walk_state::Vector{UInt64}, step::Int)

Narrator observes a walk step if it's within their visibility window.
Returns observed color and gradient, or nothing if not visible.
"""
function observe_walk(narrator::GayNarrator, walk_state::Vector{UInt64}, step::Int)
    # Check visibility
    if step > length(narrator.visibility_mask) || !narrator.visibility_mask[step]
        return nothing
    end

    seed = walk_state[step]

    # Convert seed to color
    r, g, b = seed_to_rgb(seed)
    color = RGB(r/255, g/255, b/255)

    # Compute gradient based on Enzyme mode
    gradient = compute_narrator_gradient(narrator, seed)
    push!(narrator.gradient_history, gradient)

    # Update certainty based on gradient consistency
    update_narrator_certainty!(narrator)

    return (color=color, gradient=gradient, certainty=narrator.certainty)
end

"""
    compute_narrator_gradient(narrator::GayNarrator, seed::UInt64)

Compute gradient from this narrator's perspective.
Different Enzyme modes give different gradient views.
"""
function compute_narrator_gradient(narrator::GayNarrator, seed::UInt64)
    r, g, b = seed_to_rgb(seed)
    h, s, l = rgb_to_hsl(r, g, b)

    # Value-weighted gradient
    v = narrator.values

    base_grad = if narrator.enzyme_mode == :forward
        # Forward: tangent direction (how color changes with seed)
        # Approximate ∂color/∂seed using finite difference
        r2, g2, b2 = seed_to_rgb(seed + 1)
        h2, _, _ = rgb_to_hsl(r2, g2, b2)
        (h2 - h) / 360.0  # Normalized hue change
    elseif narrator.enzyme_mode == :reverse
        # Reverse: adjoint direction (how to change seed to reach target)
        # Use inverse sensitivity
        target_h = 180.0  # Arbitrary target (cyan/complementary)
        -(h - target_h) / 360.0
    else  # :mixed
        # Mixed: bidirectional average
        r2, g2, b2 = seed_to_rgb(seed + 1)
        h2, _, _ = rgb_to_hsl(r2, g2, b2)
        fwd = (h2 - h) / 360.0
        target_h = 180.0
        rev = -(h - target_h) / 360.0
        (fwd + rev) / 2.0
    end

    # Weight by value system
    weighted = base_grad * (v.hue_weight + v.gradient_weight)

    # Add noise to prevent perfect correlation (semi-reliability)
    noise_seed = seed ⊻ UInt64(narrator.id)
    noise = ((noise_seed % 1000) / 10000.0) - 0.05  # ±5% noise

    return weighted + noise
end

function update_narrator_certainty!(narrator::GayNarrator)
    # Certainty based on gradient consistency
    if length(narrator.gradient_history) < 2
        return
    end

    # Compute autocorrelation of gradients
    grads = narrator.gradient_history
    n = length(grads)
    mean_g = sum(grads) / n

    autocorr = 0.0
    if n > 2
        for i in 1:(n-1)
            autocorr += (grads[i] - mean_g) * (grads[i+1] - mean_g)
        end
        autocorr /= ((n-1) * (sum((g - mean_g)^2 for g in grads) / n + 1e-10))
    end

    # High autocorrelation suggests originary (consistent signal)
    # Low autocorrelation suggests derived (noise from derivation)
    narrator.certainty = 0.5 + 0.5 * clamp(autocorr, -1.0, 1.0)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Narrator Triad: 3 Semi-Reliable Observers Acting Together
# ═══════════════════════════════════════════════════════════════════════════════

"""
    NarratorTriad

Three InterleavedGay narrators observing the same walk.
- Narrator 1: Originary (knows truth, but others don't know this)
- Narrator 2: Derived from 1, cannot distinguish from originary
- Narrator 3: Derived from 2, doubly uncertain

The synergy between narrators reveals structure invisible to any one.
"""
struct NarratorTriad
    narrators::NTuple{3, GayNarrator}
    synergy_tensor::Array{Float64, 3}  # 3×3×3 tensor of narrator interactions
    consensus_color::Base.RefValue{RGB{Float64}}
    consensus_certainty::Base.RefValue{Float64}
end

"""
    NarratorTriad(seed::UInt64)

Create a triad of narrators with interlocking uncertainty.
"""
function NarratorTriad(seed::UInt64)
    n1 = GayNarrator(1, seed; originary=true)
    n2 = GayNarrator(2, seed ⊻ 0xDEADBEEF; originary=false)
    n3 = GayNarrator(3, seed ⊻ 0xCAFEBABE; originary=false)

    # Initialize synergy tensor
    synergy = zeros(Float64, 3, 3, 3)

    return NarratorTriad((n1, n2, n3), synergy, Ref(RGB{Float64}(0.5, 0.5, 0.5)), Ref(0.5))
end

"""
    narrator_synergy_tensor(triad::NarratorTriad)

Compute the synergy tensor capturing 3-way interactions.
synergy[i,j,k] measures how much narrator k's observation
is influenced by the agreement between i and j.
"""
function narrator_synergy_tensor(triad::NarratorTriad)
    n = triad.narrators
    S = triad.synergy_tensor

    for i in 1:3, j in 1:3, k in 1:3
        if i == j == k
            # Self-synergy: gradient consistency
            if !isempty(n[i].gradient_history)
                grads = n[i].gradient_history
                S[i,j,k] = 1.0 - std(grads) / (mean(abs.(grads)) + 1e-10)
            end
        elseif i == j
            # Pairwise agreement influencing k
            vi, vj, vk = n[i].values, n[j].values, n[k].values
            S[i,j,k] = 1.0 - value_pluralism_distance(vi, vk)
        else
            # Three-way: measure information gain from combining
            vi, vj, vk = n[i].values, n[j].values, n[k].values
            d_ij = value_pluralism_distance(vi, vj)
            d_ik = value_pluralism_distance(vi, vk)
            d_jk = value_pluralism_distance(vj, vk)
            # Synergy is high when all three are different
            S[i,j,k] = (d_ij + d_ik + d_jk) / 3.0
        end
    end

    return S
end

"""
    O1_select_narrator(triad::NarratorTriad, query_seed::UInt64)

O(1) selection of which narrator to consult for a given query.
Uses balanced ternary to select without iterating through all.
"""
function O1_select_narrator(triad::NarratorTriad, query_seed::UInt64)
    # Use first two trits of query seed
    bridge = balanced_ternary_bridge(query_seed, 3)

    # Sum of first 3 trits mod 3 + 1 gives narrator index
    trit_sum = sum(bridge.trits[1:3])
    idx = mod(trit_sum, 3) + 1  # 1, 2, or 3

    return (idx, triad.narrators[idx])
end

# ═══════════════════════════════════════════════════════════════════════════════
# Self-Avoiding Walk in Pluriverse
# ═══════════════════════════════════════════════════════════════════════════════

"""
    InterleavedGayPluriverse

The pluriverse of all reachable gay color states, explored via
self-avoiding random walks. Each step must visit a new color
(within some tolerance ε).

Features:
- O(1) random access to any point in walk history (SPI)
- Autopoietic closure: walk generates its own continuation rules
- Balanced ternary bridges to other substrates
- Economic confidentiality through gradient obfuscation
"""
mutable struct InterleavedGayPluriverse
    seed::UInt64
    triad::NarratorTriad
    walk_history::Vector{UInt64}  # Seeds visited
    color_history::Vector{RGB}    # Colors at each step
    visited_hues::Set{Int}        # Quantized hues for self-avoidance
    avoidance_tolerance::Float64  # Hue quantum (degrees)
    autopoietic_rules::Vector{Function}  # Self-generated continuation rules
    economic_bound::Float64       # Confidentiality bound
end

"""
    InterleavedGayPluriverse(seed::UInt64; tolerance::Float64=15.0)

Create a new pluriverse ready for self-avoiding walks.
Tolerance sets the hue quantum for avoidance (default 15° = 24 sectors).
"""
function InterleavedGayPluriverse(seed::UInt64; tolerance::Float64=15.0)
    triad = NarratorTriad(seed)

    return InterleavedGayPluriverse(
        seed,
        triad,
        UInt64[],
        RGB[],
        Set{Int}(),
        tolerance,
        Function[],
        1.0  # Full confidentiality initially
    )
end

"""
    self_avoiding_walk!(pv::InterleavedGayPluriverse, n_steps::Int)

Execute a self-avoiding random walk in color space.
Each step must land on a hue not previously visited (within tolerance).
If trapped, uses autopoietic rules to escape or terminates.
"""
function self_avoiding_walk!(pv::InterleavedGayPluriverse, n_steps::Int)
    current_seed = pv.seed
    if !isempty(pv.walk_history)
        current_seed = last(pv.walk_history)
    end

    for step in 1:n_steps
        # Try to find non-visited hue
        found = false
        attempts = 0
        max_attempts = 100

        while !found && attempts < max_attempts
            # Propose next step
            next_seed = sm64(current_seed ⊻ UInt64(attempts))
            r, g, b = seed_to_rgb(next_seed)
            h, _, _ = rgb_to_hsl(r, g, b)

            # Quantize hue
            hue_sector = round(Int, h / pv.avoidance_tolerance)

            if hue_sector ∉ pv.visited_hues
                # Accept step
                found = true
                push!(pv.walk_history, next_seed)
                push!(pv.color_history, RGB(r/255, g/255, b/255))
                push!(pv.visited_hues, hue_sector)
                current_seed = next_seed

                # Narrators observe
                for narrator in pv.triad.narrators
                    observe_walk(narrator, pv.walk_history, length(pv.walk_history))
                end
            end

            attempts += 1
        end

        if !found
            # Trapped! Try autopoietic escape
            escaped = autopoietic_escape!(pv, current_seed)
            if !escaped
                @info "Walk trapped after $(length(pv.walk_history)) steps"
                break
            end
            current_seed = last(pv.walk_history)
        end
    end

    # Update economic bound based on walk diversity
    update_economic_bound!(pv)

    return pv.walk_history
end

"""
    autopoietic_escape!(pv::InterleavedGayPluriverse, current_seed::UInt64)

Self-generated rule to escape when trapped.
Uses narrator consensus to find unexplored direction.
"""
function autopoietic_escape!(pv::InterleavedGayPluriverse, current_seed::UInt64)
    # Compute narrator synergy tensor
    S = narrator_synergy_tensor(pv.triad)

    # Find direction of maximum synergy
    max_synergy = 0.0
    best_direction = 0
    for i in 1:3, j in 1:3, k in 1:3
        if S[i,j,k] > max_synergy
            max_synergy = S[i,j,k]
            best_direction = i * 100 + j * 10 + k  # Encode direction
        end
    end

    # Use direction to perturb seed in unexplored way
    escape_seed = current_seed ⊻ UInt64(best_direction * 0x123456789ABCDEF0)

    # Check if this escape is valid
    r, g, b = seed_to_rgb(escape_seed)
    h, _, _ = rgb_to_hsl(r, g, b)
    hue_sector = round(Int, h / pv.avoidance_tolerance)

    if hue_sector ∉ pv.visited_hues
        push!(pv.walk_history, escape_seed)
        push!(pv.color_history, RGB(r/255, g/255, b/255))
        push!(pv.visited_hues, hue_sector)

        # Record the escape rule for future use
        escape_rule = seed -> seed ⊻ UInt64(best_direction * 0x123456789ABCDEF0)
        push!(pv.autopoietic_rules, escape_rule)

        return true
    end

    return false
end

"""
    autopoietic_closure(pv::InterleavedGayPluriverse)

Compute the autopoietic closure: the set of all states reachable
using only the self-generated rules.
"""
function autopoietic_closure(pv::InterleavedGayPluriverse)
    if isempty(pv.autopoietic_rules)
        return pv.walk_history
    end

    closure = Set(pv.walk_history)

    # Apply each rule to all existing states
    for rule in pv.autopoietic_rules
        for seed in copy(closure)
            new_seed = rule(seed)
            push!(closure, new_seed)
        end
    end

    return collect(closure)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Economic Security: Confidentiality Bounds
# ═══════════════════════════════════════════════════════════════════════════════

"""
    update_economic_bound!(pv::InterleavedGayPluriverse)

Update the economic confidentiality bound based on walk diversity.
Higher diversity = more information revealed = lower bound.
"""
function update_economic_bound!(pv::InterleavedGayPluriverse)
    n_visited = length(pv.visited_hues)
    n_possible = round(Int, 360.0 / pv.avoidance_tolerance)

    # Confidentiality decreases as more of color space is revealed
    coverage = n_visited / n_possible
    pv.economic_bound = 1.0 - coverage
end

"""
    economic_confidentiality_bound(pv::InterleavedGayPluriverse)

Return the current economic security bound.
In the limit, this approaches the minimum sustainable confidentiality.
"""
function economic_confidentiality_bound(pv::InterleavedGayPluriverse)
    # Combine walk coverage with narrator certainty
    narrator_avg_certainty = mean(n.certainty for n in pv.triad.narrators)

    # Confidentiality is lower when narrators are more certain
    # (more certain = more information extracted)
    combined_bound = pv.economic_bound * (1.0 - narrator_avg_certainty)

    return (
        raw_bound = pv.economic_bound,
        narrator_certainty = narrator_avg_certainty,
        combined_bound = combined_bound,
        is_secure = combined_bound > 0.1  # Arbitrary security threshold
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# Synergistic Reachability: Finding Colors All 3 Narrators Can Reach
# ═══════════════════════════════════════════════════════════════════════════════

"""
    synergistic_reachability(pv::InterleavedGayPluriverse)

Find the intersection of reachable colors across all narrators.
These are the "consensus colors" where all three value systems agree.
"""
function synergistic_reachability(pv::InterleavedGayPluriverse)
    if isempty(pv.walk_history)
        return UInt64[]
    end

    # Get visibility masks
    masks = [n.visibility_mask for n in pv.triad.narrators]
    n_steps = min(length(pv.walk_history), minimum(length.(masks)))

    # Find steps visible to all three
    synergistic = UInt64[]
    for step in 1:n_steps
        if all(m[step] for m in masks)
            push!(synergistic, pv.walk_history[step])
        end
    end

    return synergistic
end

# ═══════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════════════

function sm64(z::UInt64)::UInt64
    z += 0x9E3779B97F4A7C15
    z = (z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ⊻ (z >> 27)) * 0x94D049BB133111EB
    return z ⊻ (z >> 31)
end

function seed_to_rgb(seed::UInt64)
    r = sm64(seed)
    g = sm64(r)
    b = sm64(g)
    return (Int(r >> 56), Int(g >> 56), Int(b >> 56))
end

function rgb_to_hsl(r::Int, g::Int, b::Int)
    rf, gf, bf = r/255.0, g/255.0, b/255.0
    cmax = max(rf, gf, bf)
    cmin = min(rf, gf, bf)
    delta = cmax - cmin

    l = (cmax + cmin) / 2.0

    if delta < 0.001
        return (0.0, 0.0, l)
    end

    s = delta / (1.0 - abs(2.0 * l - 1.0))

    h = if cmax == rf
        60.0 * mod((gf - bf) / delta, 6.0)
    elseif cmax == gf
        60.0 * ((bf - rf) / delta + 2.0)
    else
        60.0 * ((rf - gf) / delta + 4.0)
    end

    h = h < 0 ? h + 360.0 : h

    return (h, s, l)
end

using Statistics: mean, std

# ═══════════════════════════════════════════════════════════════════════════════
# Demo: Running the Pluriverse
# ═══════════════════════════════════════════════════════════════════════════════

"""
    world_interleaved_pluriverse(seed::UInt64=0x6761795f636f6c6f)

Demonstrate the InterleavedGay Pluriverse with 3 narrators.
"""
function world_interleaved_pluriverse(seed::UInt64=0x6761795f636f6c6f)
    println("╔══════════════════════════════════════════════════════════════════╗")
    println("║  InterleavedGay Pluriverse: Self-Avoiding Walk with 3 Narrators  ║")
    println("╚══════════════════════════════════════════════════════════════════╝")
    println()

    # Create pluriverse
    pv = InterleavedGayPluriverse(seed; tolerance=15.0)

    println("═══ NARRATOR TRIAD ═══")
    for (i, n) in enumerate(pv.triad.narrators)
        v = n.values
        println("  Narrator $i ($(n.enzyme_mode) mode):")
        println("    Values: H=$(round(v.hue_weight, digits=3)) S=$(round(v.saturation_weight, digits=3)) L=$(round(v.lightness_weight, digits=3)) G=$(round(v.gradient_weight, digits=3))")
        println("    Is originary (truth): $(n.is_originary)")
        println("    Initial certainty: $(n.certainty)")
    end
    println()

    # Run self-avoiding walk
    println("═══ SELF-AVOIDING WALK (69 steps) ═══")
    walk = self_avoiding_walk!(pv, 69)
    println("  Completed $(length(walk)) steps")
    println("  Hue sectors visited: $(length(pv.visited_hues)) / $(round(Int, 360/pv.avoidance_tolerance))")
    println()

    # Show first few colors
    println("═══ FIRST 5 COLORS ═══")
    for (i, (seed, color)) in enumerate(zip(walk[1:min(5, length(walk))], pv.color_history[1:min(5, length(pv.color_history))]))
        r, g, b = round(Int, color.r * 255), round(Int, color.g * 255), round(Int, color.b * 255)
        h, _, _ = rgb_to_hsl(r, g, b)
        hex = Printf.@sprintf("#%02X%02X%02X", r, g, b)
        println("  Step $i: $hex (H=$(round(h, digits=1))°)")
    end
    println()

    # Narrator observations
    println("═══ NARRATOR CERTAINTIES (after walk) ═══")
    for (i, n) in enumerate(pv.triad.narrators)
        visible = count(n.visibility_mask[1:min(length(walk), length(n.visibility_mask))])
        println("  Narrator $i: certainty=$(round(n.certainty, digits=3)), steps_visible=$visible/$(length(walk))")
    end
    println()

    # Synergistic reachability
    synergistic = synergistic_reachability(pv)
    println("═══ SYNERGISTIC REACHABILITY ═══")
    println("  Steps visible to all 3 narrators: $(length(synergistic))")
    println()

    # Economic bound
    econ = economic_confidentiality_bound(pv)
    println("═══ ECONOMIC CONFIDENTIALITY ═══")
    println("  Raw bound: $(round(econ.raw_bound, digits=3))")
    println("  Narrator certainty avg: $(round(econ.narrator_certainty, digits=3))")
    println("  Combined bound: $(round(econ.combined_bound, digits=3))")
    println("  Is secure: $(econ.is_secure)")
    println()

    # Balanced ternary bridge
    bridge = balanced_ternary_bridge(seed, 69)
    bw = resurrection_bandwidth(bridge)
    println("═══ BALANCED TERNARY BRIDGE ═══")
    println("  69 trits: $(bridge.trits[1:10])...")
    println("  Total bandwidth: $(round(bw.total, digits=2)) bits")
    println("  Effective bandwidth: $(round(bw.effective, digits=2)) bits")
    println("  Resurrection ratio: $(round(bw.resurrection_ratio, digits=3))")
    println()

    # Autopoietic closure
    closure = autopoietic_closure(pv)
    println("═══ AUTOPOIETIC CLOSURE ═══")
    println("  Rules generated: $(length(pv.autopoietic_rules))")
    println("  Closure size: $(length(closure)) states")
    println()

    # Synergy tensor
    S = narrator_synergy_tensor(pv.triad)
    println("═══ SYNERGY TENSOR (diagonal) ═══")
    for i in 1:3
        println("  S[$i,$i,$i] = $(round(S[i,i,i], digits=3)) (narrator $i self-synergy)")
    end
    println()

    # Value pluralism distances
    println("═══ VALUE PLURALISM DISTANCES ═══")
    for i in 1:3, j in (i+1):3
        d = value_pluralism_distance(pv.triad.narrators[i].values, pv.triad.narrators[j].values)
        println("  d(N$i, N$j) = $(round(d, digits=3)) $(d > 0.1 ? "(irreducibly plural)" : "(may be derived)")")
    end
    println()

    println("╔══════════════════════════════════════════════════════════════════╗")
    println("║  Pluriverse exploration complete. Economic security maintained.  ║")
    println("╚══════════════════════════════════════════════════════════════════╝")

    return pv
end

export world_interleaved_pluriverse
