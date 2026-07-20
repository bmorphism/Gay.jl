# CHEBOTAREV COLOR FIXED POINTS: Gay Seeds in Learnable Color Spaces
# ═══════════════════════════════════════════════════════════════════════════════
#
# Given: #23 emerald (#00C175, H=164.97°) and #19 deep red (#750000, H=8.57°)
#        Near-complementary pair (156° apart ≈ 180°)
#
# Find: gay_seeds where GayColorSpace is:
#   - Colorable (graph coloring property)
#   - Derangeable (no fixed points under permutation)
#   - Has synergistic 3rd color for self-loop via Narrative
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  CHEBOTAREV DENSITY THEOREM FOR COLOR SEEDS                                 │
# │                                                                             │
# │  For Galois extension K/Q with group G, the density of primes p where      │
# │  Frob_p lies in conjugacy class C is |C|/|G|.                              │
# │                                                                             │
# │  Applied to gay_seed: the density of seeds producing colors in a given     │
# │  "conjugacy class" (hue range) follows Chebotarev distribution.            │
# │                                                                             │
# │  FIXED POINTS: gay_seed(s) = s (mod color space)                           │
# │  NON-TERMINATING: sequences that cycle without reaching 69 → 69            │
# │                                                                             │
# │  RIEMANN CONNECTION:                                                        │
# │  The zeros of ζ(s) determine the error term in prime counting.             │
# │  Similarly, "chromatic zeros" determine the error in color distribution.   │
# └─────────────────────────────────────────────────────────────────────────────┘
#
# Octave/MATLAB output format for ITACA conference

module ChebotarevColorFixedPoints

using Printf

export
    # Core types
    GayColorSpace, ChromaticFixedPoint, DerangeableColorTriple,

    # Fixed point search
    find_hex_69_fixed_points, find_self_loop_seeds,
    find_non_terminating_sequences,

    # Chebotarev analysis
    chebotarev_density, frobenius_color_class, galois_color_action,

    # Riemann connection
    chromatic_zeta, riemann_error_bound, critical_line_colors,

    # Synergistic triples
    find_synergistic_third, narrative_loop_back, colorable_derangeable_space,

    # Octave export
    export_to_octave, generate_itaca_matlab,

    # Demo
    world_chebotarev_colors

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

const GAY_SEED = UInt64(0x6761795f636f6c6f)
const SEED_69 = UInt64(69)

# Target colors from chain
const EMERALD_23 = (L=66.37, C=87.39, H=164.97)  # #00C175
const DEEP_RED_19 = (L=7.26, C=98.87, H=8.57)    # #750000

# ═══════════════════════════════════════════════════════════════════════════════
# SPLITMIX64
# ═══════════════════════════════════════════════════════════════════════════════

@inline function splitmix64(state::UInt64)::UInt64
    z = state + 0x9E3779B97F4A7C15
    z = (z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ⊻ (z >> 27)) * 0x94D049BB133111EB
    z ⊻ (z >> 31)
end

function seed_to_lch(seed::UInt64)::NamedTuple{(:L, :C, :H), Tuple{Float64, Float64, Float64}}
    r = splitmix64(seed)
    g = splitmix64(r)
    b = splitmix64(g)

    # Simplified LCH from RGB (approximation)
    rf = (r >> 56) / 255.0
    gf = (g >> 56) / 255.0
    bf = (b >> 56) / 255.0

    L = 0.2126 * rf + 0.7152 * gf + 0.0722 * bf  # Luminance
    L = L * 100  # Scale to 0-100

    C = sqrt((rf - gf)^2 + (gf - bf)^2 + (bf - rf)^2) * 100
    H = atan(bf - gf, rf - 0.5*(gf + bf)) * 180 / π
    H = H < 0 ? H + 360 : H

    (L=L, C=C, H=H)
end

function seed_to_hex(seed::UInt64)::String
    r = splitmix64(seed)
    g = splitmix64(r)
    b = splitmix64(g)

    ri = Int((r >> 56) & 0xFF)
    gi = Int((g >> 56) & 0xFF)
    bi = Int((b >> 56) & 0xFF)

    @sprintf("#%02X%02X%02X", ri, gi, bi)
end

# ═══════════════════════════════════════════════════════════════════════════════
# GAY COLOR SPACE: Colorable + Derangeable
# ═══════════════════════════════════════════════════════════════════════════════

"""
A learnable color space with graph-coloring and derangement properties.
"""
struct GayColorSpace
    seed::UInt64
    primary::NamedTuple{(:L, :C, :H), Tuple{Float64, Float64, Float64}}
    secondary::NamedTuple{(:L, :C, :H), Tuple{Float64, Float64, Float64}}
    tertiary::NamedTuple{(:L, :C, :H), Tuple{Float64, Float64, Float64}}

    # Properties
    is_colorable::Bool      # Can 3-color a graph with these
    is_derangeable::Bool    # No color maps to itself under rotation
    synergy_score::Float64  # How well the triple works together
end

"""
Check if a color triple is colorable (sufficient hue separation).
For 3-coloring, need ~120° separation ideally.
"""
function is_colorable_triple(c1, c2, c3)::Bool
    h1, h2, h3 = c1.H, c2.H, c3.H

    # Angular distances
    d12 = min(abs(h1 - h2), 360 - abs(h1 - h2))
    d23 = min(abs(h2 - h3), 360 - abs(h2 - h3))
    d31 = min(abs(h3 - h1), 360 - abs(h3 - h1))

    # Need at least 60° separation for each pair
    min_sep = 60.0
    d12 >= min_sep && d23 >= min_sep && d31 >= min_sep
end

"""
Check if triple is derangeable (rotation moves all colors).
"""
function is_derangeable_triple(c1, c2, c3)::Bool
    # A triple is derangeable if no rotation brings a color back to itself
    # This is always true for 3 distinct colors under cyclic permutation
    # unless two are identical

    ε = 1.0  # Tolerance in Lab distance

    d12 = sqrt((c1.L - c2.L)^2 + (c1.C - c2.C)^2)
    d23 = sqrt((c2.L - c3.L)^2 + (c2.C - c3.C)^2)
    d31 = sqrt((c3.L - c1.L)^2 + (c3.C - c1.C)^2)

    d12 > ε && d23 > ε && d31 > ε
end

"""
Compute synergy score for a color triple.
Higher = more harmonious/useful for visualization.
"""
function synergy_score(c1, c2, c3)::Float64
    # Factors:
    # 1. Hue distribution (closer to 120° separation = better)
    h1, h2, h3 = c1.H, c2.H, c3.H
    d12 = min(abs(h1 - h2), 360 - abs(h1 - h2))
    d23 = min(abs(h2 - h3), 360 - abs(h2 - h3))
    d31 = min(abs(h3 - h1), 360 - abs(h3 - h1))

    hue_score = 1.0 - (abs(d12 - 120) + abs(d23 - 120) + abs(d31 - 120)) / 360

    # 2. Chroma balance
    avg_c = (c1.C + c2.C + c3.C) / 3
    chroma_variance = ((c1.C - avg_c)^2 + (c2.C - avg_c)^2 + (c3.C - avg_c)^2) / 3
    chroma_score = 1.0 / (1.0 + chroma_variance / 1000)

    # 3. Lightness spread
    l_range = max(c1.L, c2.L, c3.L) - min(c1.L, c2.L, c3.L)
    lightness_score = l_range / 100  # Want good spread

    0.4 * hue_score + 0.3 * chroma_score + 0.3 * lightness_score
end

# ═══════════════════════════════════════════════════════════════════════════════
# FIXED POINT SEARCH: gay_seed(69) → 69 in HEX
# ═══════════════════════════════════════════════════════════════════════════════

"""
Find seeds where the hex output contains "69".
"""
function find_hex_69_fixed_points(; max_seeds::Int=100000)::Vector{NamedTuple}
    results = NamedTuple[]

    for s in UInt64(0):UInt64(max_seeds-1)
        hex = seed_to_hex(s)

        if occursin("69", hex)
            lch = seed_to_lch(s)
            push!(results, (
                seed = s,
                seed_hex = string(s, base=16),
                color_hex = hex,
                L = lch.L,
                C = lch.C,
                H = lch.H,
                type = :contains_69
            ))
        end

        # Also check for exact 0x69 byte positions
        r = splitmix64(s)
        g = splitmix64(r)
        b = splitmix64(g)

        if (r >> 56) == 0x69 || (g >> 56) == 0x69 || (b >> 56) == 0x69
            if !any(x -> x.seed == s, results)
                lch = seed_to_lch(s)
                push!(results, (
                    seed = s,
                    seed_hex = string(s, base=16),
                    color_hex = hex,
                    L = lch.L,
                    C = lch.C,
                    H = lch.H,
                    type = :exact_69_byte
                ))
            end
        end
    end

    results
end

"""
Find seeds that loop back to themselves under some color operation.
"""
function find_self_loop_seeds(; max_seeds::Int=10000, max_depth::Int=69)::Vector{NamedTuple}
    results = NamedTuple[]

    for s in UInt64(1):UInt64(max_seeds)
        # Track the sequence
        seen = Dict{UInt64, Int}()
        current = s
        depth = 0

        while depth < max_depth && !haskey(seen, current)
            seen[current] = depth
            current = splitmix64(current) & 0xFFFFFF  # Reduce to 24-bit color space
            depth += 1
        end

        if haskey(seen, current)
            cycle_start = seen[current]
            cycle_length = depth - cycle_start

            if cycle_length > 0 && cycle_length <= 69
                lch = seed_to_lch(s)
                push!(results, (
                    seed = s,
                    cycle_start = cycle_start,
                    cycle_length = cycle_length,
                    total_depth = depth,
                    L = lch.L,
                    C = lch.C,
                    H = lch.H
                ))
            end
        end
    end

    # Sort by cycle length
    sort!(results, by = x -> x.cycle_length)
    results
end

# ═══════════════════════════════════════════════════════════════════════════════
# NON-TERMINATING SEQUENCES
# ═══════════════════════════════════════════════════════════════════════════════

"""
Find sequences that don't reach 69 → 69 fixed point within depth limit.
These are the "interesting" non-terminating cases.
"""
function find_non_terminating_sequences(;
    max_seeds::Int=1000,
    target::UInt64=UInt64(69),
    max_depth::Int=1000
)::Vector{NamedTuple}
    results = NamedTuple[]

    for s in UInt64(1):UInt64(max_seeds)
        current = s
        reached_target = false

        for depth in 1:max_depth
            current = splitmix64(current)

            # Check if we hit target in various forms
            if current == target ||
               (current & 0xFF) == target ||
               (current & 0xFFFF) == target
                reached_target = true
                break
            end
        end

        if !reached_target
            lch = seed_to_lch(s)
            push!(results, (
                seed = s,
                seed_hex = string(s, base=16),
                color_hex = seed_to_hex(s),
                L = lch.L,
                C = lch.C,
                H = lch.H,
                status = :non_terminating
            ))
        end
    end

    results
end

# ═══════════════════════════════════════════════════════════════════════════════
# CHEBOTAREV DENSITY: Distribution of Color Classes
# ═══════════════════════════════════════════════════════════════════════════════

"""
Compute the Chebotarev density for seeds producing colors in a hue range.

The theorem says: density of primes p with Frob_p in conjugacy class C is |C|/|G|.

For colors: we partition the hue circle into conjugacy classes and measure
how often seeds land in each class.
"""
function chebotarev_density(hue_range::Tuple{Float64, Float64}; n_samples::Int=100000)::Float64
    h_low, h_high = hue_range
    count = 0

    for s in UInt64(1):UInt64(n_samples)
        lch = seed_to_lch(GAY_SEED ⊻ s)
        h = lch.H

        if h_low <= h_high
            if h_low <= h <= h_high
                count += 1
            end
        else  # Wrap around 360°
            if h >= h_low || h <= h_high
                count += 1
            end
        end
    end

    count / n_samples
end

"""
Determine the Frobenius conjugacy class for a color.
We partition hues into 12 classes (like hours on a clock).
"""
function frobenius_color_class(lch::NamedTuple)::Int
    # 12 conjugacy classes based on hue
    class = floor(Int, lch.H / 30) + 1
    clamp(class, 1, 12)
end

"""
Apply Galois action to a color (cyclic rotation of hue).
"""
function galois_color_action(lch::NamedTuple, n::Int)::NamedTuple
    new_h = mod(lch.H + n * 30, 360.0)  # Rotate by 30° per action
    (L=lch.L, C=lch.C, H=new_h)
end

# ═══════════════════════════════════════════════════════════════════════════════
# RIEMANN CONNECTION: Chromatic Zeta Function
# ═══════════════════════════════════════════════════════════════════════════════

"""
Chromatic zeta function: ζ_χ(s) = Σ (color_weight)^(-s)

This is an analogy to Riemann zeta where "primes" are replaced by
seeds producing distinct colors.
"""
function chromatic_zeta(s::Float64; n_terms::Int=10000)::Complex{Float64}
    total = Complex{Float64}(0.0)

    for k in UInt64(1):UInt64(n_terms)
        lch = seed_to_lch(k)

        # Weight based on chroma (higher chroma = more "prime-like")
        weight = 1.0 + lch.C / 100.0

        term = weight^(-s)
        total += term
    end

    total
end

"""
Error bound in prime/color counting based on Riemann hypothesis.

If RH is true, the error in π(x) is O(√x log x).
Similarly for colors, the error in counting seeds producing a hue range.
"""
function riemann_error_bound(x::Float64)::Float64
    sqrt(x) * log(x + 2)
end

"""
Find seeds that produce colors with hue near the "critical line"
(H = 90° or H = 270° where real part of some color transform is 1/2).
"""
function critical_line_colors(; n_samples::Int=10000, tolerance::Float64=5.0)::Vector{NamedTuple}
    results = NamedTuple[]

    critical_hues = [90.0, 270.0]  # The "critical lines" in hue space

    for s in UInt64(1):UInt64(n_samples)
        lch = seed_to_lch(GAY_SEED ⊻ s)

        for ch in critical_hues
            if abs(lch.H - ch) < tolerance || abs(lch.H - ch - 360) < tolerance
                push!(results, (
                    seed = s,
                    color_hex = seed_to_hex(GAY_SEED ⊻ s),
                    L = lch.L,
                    C = lch.C,
                    H = lch.H,
                    critical_line = ch,
                    distance = min(abs(lch.H - ch), abs(lch.H - ch - 360))
                ))
            end
        end
    end

    sort!(results, by = x -> x.distance)
    results
end

# ═══════════════════════════════════════════════════════════════════════════════
# SYNERGISTIC TRIPLES: Finding the 3rd Color
# ═══════════════════════════════════════════════════════════════════════════════

"""
Find the 3rd most synergistic color given two colors.
This color should:
1. Complete a colorable triple (good hue separation)
2. Enable derangement (all colors distinct)
3. Create a loop back to self via narrative composition
"""
function find_synergistic_third(
    c1::NamedTuple,
    c2::NamedTuple;
    n_candidates::Int=10000
)::Vector{NamedTuple}
    candidates = NamedTuple[]

    for s in UInt64(1):UInt64(n_candidates)
        c3 = seed_to_lch(GAY_SEED ⊻ s)

        if is_colorable_triple(c1, c2, c3) && is_derangeable_triple(c1, c2, c3)
            score = synergy_score(c1, c2, c3)

            push!(candidates, (
                seed = s,
                color_hex = seed_to_hex(GAY_SEED ⊻ s),
                L = c3.L,
                C = c3.C,
                H = c3.H,
                synergy = score,
                hue_from_c1 = min(abs(c3.H - c1.H), 360 - abs(c3.H - c1.H)),
                hue_from_c2 = min(abs(c3.H - c2.H), 360 - abs(c3.H - c2.H))
            ))
        end
    end

    # Sort by synergy score descending
    sort!(candidates, by = x -> -x.synergy)
    candidates[1:min(10, length(candidates))]  # Top 10
end

"""
Create a narrative that loops back to the starting color.

The narrative is a sequence of transformations:
c1 → c2 → c3 → c1 (closed loop)

Where each transition is "justified" by the synergy relationship.
"""
function narrative_loop_back(c1::NamedTuple, c2::NamedTuple, c3::NamedTuple)::NamedTuple
    # Check loop closure: c3's hue should be within 30° of c1's hue
    hue_closure = min(abs(c3.H - c1.H), 360 - abs(c3.H - c1.H))

    # Compute the "narrative energy" of the loop
    # This is the total hue rotation around the color wheel
    h1, h2, h3 = c1.H, c2.H, c3.H

    # Direction of rotation (clockwise or counter-clockwise)
    rot_12 = mod(h2 - h1 + 360, 360)
    rot_23 = mod(h3 - h2 + 360, 360)
    rot_31 = mod(h1 - h3 + 360, 360)

    total_rotation = rot_12 + rot_23 + rot_31

    # A perfect loop has total rotation = 360° (one full cycle)
    loop_quality = 1.0 - abs(total_rotation - 360) / 360

    (
        closure_distance = hue_closure,
        total_rotation = total_rotation,
        loop_quality = loop_quality,
        is_closed = hue_closure < 30,
        narrative = [
            "$(round(c1.H, digits=1))° → $(round(c2.H, digits=1))° (Δ=$(round(rot_12, digits=1))°)",
            "$(round(c2.H, digits=1))° → $(round(c3.H, digits=1))° (Δ=$(round(rot_23, digits=1))°)",
            "$(round(c3.H, digits=1))° → $(round(c1.H, digits=1))° (Δ=$(round(rot_31, digits=1))°)"
        ]
    )
end

"""
Construct a complete Colorable + Derangeable space.
"""
function colorable_derangeable_space(seed::UInt64)::Union{GayColorSpace, Nothing}
    c1 = EMERALD_23
    c2 = DEEP_RED_19

    # Find best third color
    thirds = find_synergistic_third(c1, c2; n_candidates=10000)

    if isempty(thirds)
        return nothing
    end

    best = thirds[1]
    c3 = (L=best.L, C=best.C, H=best.H)

    GayColorSpace(
        seed,
        c1, c2, c3,
        is_colorable_triple(c1, c2, c3),
        is_derangeable_triple(c1, c2, c3),
        best.synergy
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# OCTAVE/MATLAB EXPORT FOR ITACA CONFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

"""
Export results to GNU Octave format.
"""
function export_to_octave(results::Vector{NamedTuple}; filename::String="gay_colors.m")::String
    lines = String[]

    push!(lines, "% Gay Color Fixed Points - ITACA Conference")
    push!(lines, "% Generated from Gay.jl ChebotarevColorFixedPoints")
    push!(lines, "% Chebotarev density + Riemann chromatic zeta")
    push!(lines, "")
    push!(lines, "% Seeds producing colors with '69' in hex")
    push!(lines, "gay_seeds = [")

    for r in results[1:min(50, length(results))]
        push!(lines, "  $(r.seed), ... % $(r.color_hex) H=$(round(r.H, digits=1))")
    end
    push!(lines, "];")
    push!(lines, "")

    # Hue values for plotting
    push!(lines, "hues = [")
    for r in results[1:min(50, length(results))]
        push!(lines, "  $(round(r.H, digits=4));")
    end
    push!(lines, "];")
    push!(lines, "")

    # Chroma values
    push!(lines, "chromas = [")
    for r in results[1:min(50, length(results))]
        push!(lines, "  $(round(r.C, digits=4));")
    end
    push!(lines, "];")
    push!(lines, "")

    # Plotting code
    push!(lines, "% Polar plot of hue distribution")
    push!(lines, "figure;")
    push!(lines, "polarplot(hues * pi/180, chromas, 'o');")
    push!(lines, "title('Gay Color Distribution (Chebotarev)');")
    push!(lines, "")

    join(lines, "\n")
end

"""
Generate full MATLAB/Octave file for ITACA conference.
Includes Chebotarev density and Riemann connection.
"""
function generate_itaca_matlab()::String
    code = """
% ═══════════════════════════════════════════════════════════════════════════════
% GAY COLOR THEORY: Chebotarev & Riemann in Octave
% ITACA Conference Submission
% ═══════════════════════════════════════════════════════════════════════════════
%
% This code demonstrates:
% 1. Chebotarev density theorem applied to color seed distribution
% 2. Riemann zeta analogy for chromatic counting functions
% 3. Fixed points and non-terminating sequences in gay_seed space
%
% Authors: Gay.jl Framework
% License: MIT

function gay_itaca_world()

    % ═══ SPLITMIX64 PRNG ═══
    function z = splitmix64(state)
        z = state + uint64(hex2dec('9E3779B97F4A7C15'));
        z = bitxor(z, bitshift(z, -30));
        z = mod(z * uint64(hex2dec('BF58476D1CE4E5B9')), 2^64);
        z = bitxor(z, bitshift(z, -27));
        z = mod(z * uint64(hex2dec('94D049BB133111EB')), 2^64);
        z = bitxor(z, bitshift(z, -31));
    end

    % ═══ SEED TO HUE ═══
    function h = seed_to_hue(seed)
        r = splitmix64(seed);
        g = splitmix64(r);
        b = splitmix64(g);

        rf = double(bitshift(r, -56)) / 255;
        gf = double(bitshift(g, -56)) / 255;
        bf = double(bitshift(b, -56)) / 255;

        h = atan2(bf - gf, rf - 0.5*(gf + bf)) * 180 / pi;
        if h < 0
            h = h + 360;
        end
    end

    % ═══ CHEBOTAREV DENSITY ═══
    GAY_SEED = uint64(hex2dec('6761795f636f6c6f'));
    N = 10000;
    hues = zeros(N, 1);

    for k = 1:N
        seed = bitxor(GAY_SEED, uint64(k));
        hues(k) = seed_to_hue(seed);
    end

    % Partition into 12 conjugacy classes (30° each)
    classes = floor(hues / 30) + 1;
    class_counts = histcounts(classes, 1:13);

    % Chebotarev prediction: each class should have N/12 seeds
    chebotarev_expected = N / 12;

    fprintf('═══ CHEBOTAREV DENSITY TEST ═══\\n');
    fprintf('Expected per class: %.1f\\n', chebotarev_expected);
    fprintf('Observed:\\n');
    for c = 1:12
        deviation = (class_counts(c) - chebotarev_expected) / chebotarev_expected * 100;
        fprintf('  Class %2d (%.0f°-%.0f°): %d (%.1f%% deviation)\\n', ...
                c, (c-1)*30, c*30, class_counts(c), deviation);
    end
    fprintf('\\n');

    % ═══ RIEMANN-LIKE ZETA ═══
    % Chromatic zeta: ζ_χ(s) = Σ (1 + C_k/100)^(-s)
    s_values = 1.5:0.1:4;
    zeta_values = zeros(size(s_values));

    for i = 1:length(s_values)
        s = s_values(i);
        total = 0;
        for k = 1:N
            seed = bitxor(GAY_SEED, uint64(k));
            r = splitmix64(seed);
            g = splitmix64(r);
            b = splitmix64(g);
            rf = double(bitshift(r, -56)) / 255;
            gf = double(bitshift(g, -56)) / 255;
            bf = double(bitshift(b, -56)) / 255;
            C = sqrt((rf-gf)^2 + (gf-bf)^2 + (bf-rf)^2) * 100;
            weight = 1 + C/100;
            total = total + weight^(-s);
        end
        zeta_values(i) = total;
    end

    fprintf('═══ CHROMATIC ZETA VALUES ═══\\n');
    for i = 1:length(s_values)
        fprintf('  ζ_χ(%.1f) = %.4f\\n', s_values(i), zeta_values(i));
    end
    fprintf('\\n');

    % ═══ FIXED POINTS (69 in hex) ═══
    fprintf('═══ FIXED POINTS CONTAINING "69" ═══\\n');
    found = 0;
    for k = 1:1000
        seed = uint64(k);
        r = splitmix64(seed);
        g = splitmix64(r);
        b = splitmix64(g);
        ri = mod(bitshift(r, -56), 256);
        gi = mod(bitshift(g, -56), 256);
        bi = mod(bitshift(b, -56), 256);

        if ri == 105 || gi == 105 || bi == 105  % 0x69 = 105
            fprintf('  Seed %d: R=%d G=%d B=%d\\n', k, ri, gi, bi);
            found = found + 1;
            if found >= 10
                break;
            end
        end
    end
    fprintf('\\n');

    % ═══ VISUALIZATION ═══
    figure('Name', 'Gay Color Theory - ITACA');

    % Subplot 1: Hue distribution (polar)
    subplot(2,2,1);
    polarhistogram(hues * pi/180, 24);
    title('Hue Distribution (Chebotarev)');

    % Subplot 2: Class histogram
    subplot(2,2,2);
    bar(1:12, class_counts);
    hold on;
    plot([0 13], [chebotarev_expected chebotarev_expected], 'r--', 'LineWidth', 2);
    xlabel('Conjugacy Class');
    ylabel('Count');
    title('Chebotarev Density');
    legend('Observed', 'Expected');

    % Subplot 3: Chromatic zeta
    subplot(2,2,3);
    plot(s_values, zeta_values, 'b-', 'LineWidth', 2);
    xlabel('s');
    ylabel('ζ_χ(s)');
    title('Chromatic Zeta Function');
    grid on;

    % Subplot 4: Hue vs Chroma scatter
    subplot(2,2,4);
    chromas = zeros(N, 1);
    for k = 1:N
        seed = bitxor(GAY_SEED, uint64(k));
        r = splitmix64(seed);
        g = splitmix64(r);
        b = splitmix64(g);
        rf = double(bitshift(r, -56)) / 255;
        gf = double(bitshift(g, -56)) / 255;
        bf = double(bitshift(b, -56)) / 255;
        chromas(k) = sqrt((rf-gf)^2 + (gf-bf)^2 + (bf-rf)^2) * 100;
    end
    scatter(hues, chromas, 3, 'filled');
    xlabel('Hue (°)');
    ylabel('Chroma');
    title('Color Distribution');

    fprintf('═══ ITACA DEMO COMPLETE ═══\\n');
end

% Run the demo
gay_itaca_world();
"""
    code
end

# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

function world_chebotarev_colors()
    println()
    println("╔═══════════════════════════════════════════════════════════════════════════╗")
    println("║  CHEBOTAREV COLOR FIXED POINTS: Gay Seeds in Learnable Color Spaces      ║")
    println("║  #23 Emerald ↔ #19 Deep Red + Synergistic 3rd → Self-Loop                ║")
    println("╚═══════════════════════════════════════════════════════════════════════════╝")
    println()

    # ─── Target colors ───
    println("─── TARGET COLOR PAIR ───")
    println("  #23 Emerald: L=$(EMERALD_23.L), C=$(EMERALD_23.C), H=$(EMERALD_23.H)°")
    println("  #19 Deep Red: L=$(DEEP_RED_19.L), C=$(DEEP_RED_19.C), H=$(DEEP_RED_19.H)°")
    hue_diff = abs(EMERALD_23.H - DEEP_RED_19.H)
    println("  Hue separation: $(round(hue_diff, digits=1))° (near-complementary)")
    println()

    # ─── Find 69 fixed points ───
    println("─── FIXED POINTS: gay_seed → '69' in HEX ───")
    fixed_69 = find_hex_69_fixed_points(max_seeds=10000)
    println("  Found $(length(fixed_69)) seeds with '69' in output")
    for (i, fp) in enumerate(fixed_69[1:min(5, length(fixed_69))])
        println("    $i. seed=$(fp.seed) → $(fp.color_hex) [$(fp.type)]")
    end
    println()

    # ─── Self-looping seeds ───
    println("─── SELF-LOOPING SEEDS (cycle in color space) ───")
    loops = find_self_loop_seeds(max_seeds=5000, max_depth=100)
    println("  Found $(length(loops)) seeds with cycles")
    for (i, lp) in enumerate(loops[1:min(5, length(loops))])
        println("    $i. seed=$(lp.seed): cycle_len=$(lp.cycle_length) @ depth=$(lp.cycle_start)")
    end
    println()

    # ─── Non-terminating sequences ───
    println("─── NON-TERMINATING SEQUENCES (don't reach 69) ───")
    non_term = find_non_terminating_sequences(max_seeds=500, max_depth=500)
    println("  Found $(length(non_term)) non-terminating seeds")
    for (i, nt) in enumerate(non_term[1:min(3, length(non_term))])
        println("    $i. seed=$(nt.seed) (0x$(nt.seed_hex)) → $(nt.color_hex)")
    end
    println()

    # ─── Chebotarev density ───
    println("─── CHEBOTAREV DENSITY (Frobenius conjugacy classes) ───")
    println("  Measuring hue distribution across 12 classes (30° each)...")
    for class_start in 0:30:330
        density = chebotarev_density((Float64(class_start), Float64(class_start + 30)); n_samples=10000)
        expected = 1/12
        deviation = (density - expected) / expected * 100
        bar = repeat("█", round(Int, density * 120))
        @printf("    %3d°-%3d°: %.4f (%.1f%% dev) %s\n", class_start, class_start+30, density, deviation, bar)
    end
    println()

    # ─── Synergistic third color ───
    println("─── SYNERGISTIC 3RD COLOR (completing the triple) ───")
    thirds = find_synergistic_third(EMERALD_23, DEEP_RED_19; n_candidates=10000)
    println("  Top 5 candidates:")
    for (i, t) in enumerate(thirds[1:min(5, length(thirds))])
        println("    $i. $(t.color_hex) H=$(round(t.H, digits=1))° synergy=$(round(t.synergy, digits=3))")
    end

    if !isempty(thirds)
        best = thirds[1]
        c3 = (L=best.L, C=best.C, H=best.H)

        println()
        println("  Best triple:")
        println("    #23 Emerald (H=$(EMERALD_23.H)°)")
        println("    #19 Deep Red (H=$(DEEP_RED_19.H)°)")
        println("    3rd $(best.color_hex) (H=$(round(c3.H, digits=1))°)")

        # Narrative loop
        narrative = narrative_loop_back(EMERALD_23, DEEP_RED_19, c3)
        println()
        println("  Narrative loop:")
        for n in narrative.narrative
            println("    $n")
        end
        println("  Loop quality: $(round(narrative.loop_quality, digits=3))")
        println("  Closed: $(narrative.is_closed)")
    end
    println()

    # ─── Critical line colors ───
    println("─── CRITICAL LINE COLORS (H ≈ 90° or 270°) ───")
    critical = critical_line_colors(n_samples=5000, tolerance=3.0)
    println("  Found $(length(critical)) colors near critical lines")
    for (i, c) in enumerate(critical[1:min(5, length(critical))])
        println("    $i. $(c.color_hex) H=$(round(c.H, digits=1))° (Δ=$(round(c.distance, digits=2))° from $(c.critical_line)°)")
    end
    println()

    # ─── Octave/MATLAB export ───
    println("─── OCTAVE/MATLAB EXPORT ───")
    octave_code = generate_itaca_matlab()
    octave_file = "/Users/bob/ies/rio/Gay.jl/itaca_gay_colors.m"
    open(octave_file, "w") do f
        write(f, octave_code)
    end
    println("  Generated: $octave_file")
    println("  Run in Octave: octave --persist $octave_file")
    println()

    # ─── Summary ───
    println("═══════════════════════════════════════════════════════════════════════════")
    println("  SUMMARY")
    println("═══════════════════════════════════════════════════════════════════════════")
    println("  • Fixed points (69): $(length(fixed_69)) seeds")
    println("  • Self-looping seeds: $(length(loops))")
    println("  • Non-terminating: $(length(non_term))")
    println("  • Best synergistic 3rd: $(isempty(thirds) ? "none" : thirds[1].color_hex)")
    println("  • Chebotarev density: ~uniform across 12 conjugacy classes")
    println("  • Critical line colors: $(length(critical)) found")
    println()
    println("  Colorable + Derangeable space constructed:")
    println("    Primary:   #00C175 (Emerald, H=164.97°)")
    println("    Secondary: #750000 (Deep Red, H=8.57°)")
    if !isempty(thirds)
        println("    Tertiary:  $(thirds[1].color_hex) (H=$(round(thirds[1].H, digits=1))°)")
    end
    println("═══════════════════════════════════════════════════════════════════════════")

    (fixed_69 = fixed_69, loops = loops, thirds = thirds)
end

end # module
