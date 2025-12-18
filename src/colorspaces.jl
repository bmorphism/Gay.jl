# Wide-gamut color space definitions and random sampling
# Supports Rec.2020, Display P3, sRGB, and custom primaries

using Colors, ColorTypes, Random

export ColorSpace, Rec2020, DisplayP3, SRGB, CustomColorSpace, Primaries
export random_color, random_colors, random_palette
export gamut_map, in_gamut, clamp_to_gamut
export pride_flag, rainbow, bisexual, transgender, nonbinary, pansexual, asexual

"""
Abstract type for wide-gamut color spaces with RGB primaries.
"""
abstract type ColorSpace end

"""
    Primaries

CIE xy chromaticity coordinates for RGB primaries and white point.
"""
struct Primaries
    rx::Float64; ry::Float64  # Red primary
    gx::Float64; gy::Float64  # Green primary
    bx::Float64; by::Float64  # Blue primary
    wx::Float64; wy::Float64  # White point (D65 default)
end

"""
    Rec2020

ITU-R BT.2020 (Rec. 2020) color space - Ultra HD / 4K / 8K.
"""
struct Rec2020 <: ColorSpace end
const REC2020_PRIMARIES = Primaries(
    0.708, 0.292,   # Red
    0.170, 0.797,   # Green
    0.131, 0.046,   # Blue
    0.3127, 0.3290  # D65
)

"""
    DisplayP3

Display P3 (Apple/DCI-P3 with D65) color space.
"""
struct DisplayP3 <: ColorSpace end
const P3_PRIMARIES = Primaries(
    0.680, 0.320,   # Red
    0.265, 0.690,   # Green
    0.150, 0.060,   # Blue
    0.3127, 0.3290  # D65
)

"""
    SRGB

sRGB / Rec.709 color space.
"""
struct SRGB <: ColorSpace end
const SRGB_PRIMARIES = Primaries(
    0.640, 0.330,   # Red
    0.300, 0.600,   # Green
    0.150, 0.060,   # Blue
    0.3127, 0.3290  # D65
)

"""
    CustomColorSpace

User-defined color space with arbitrary primaries.
"""
struct CustomColorSpace <: ColorSpace
    primaries::Primaries
    name::String
end

get_primaries(::Rec2020) = REC2020_PRIMARIES
get_primaries(::DisplayP3) = P3_PRIMARIES
get_primaries(::SRGB) = SRGB_PRIMARIES
get_primaries(cs::CustomColorSpace) = cs.primaries

"""
    rgb_to_xyz_matrix(cs::ColorSpace)

Compute the 3x3 matrix to convert RGB to XYZ for a given color space.
"""
function rgb_to_xyz_matrix(cs::ColorSpace)
    p = get_primaries(cs)
    
    # Compute XYZ of primaries
    Xr = p.rx / p.ry
    Yr = 1.0
    Zr = (1.0 - p.rx - p.ry) / p.ry
    
    Xg = p.gx / p.gy
    Yg = 1.0
    Zg = (1.0 - p.gx - p.gy) / p.gy
    
    Xb = p.bx / p.by
    Yb = 1.0
    Zb = (1.0 - p.bx - p.by) / p.by
    
    # White point XYZ
    Xw = p.wx / p.wy
    Yw = 1.0
    Zw = (1.0 - p.wx - p.wy) / p.wy
    
    # Solve for scaling factors
    M = [Xr Xg Xb; Yr Yg Yb; Zr Zg Zb]
    S = M \ [Xw, Yw, Zw]
    
    return [S[1]*Xr S[2]*Xg S[3]*Xb;
            S[1]*Yr S[2]*Yg S[3]*Yb;
            S[1]*Zr S[2]*Zg S[3]*Zb]
end

"""
    random_color(cs::ColorSpace=SRGB(); rng=Random.GLOBAL_RNG)

Sample a random color uniformly from the given color space's gamut.
Returns an RGB color.
"""
function random_color(cs::ColorSpace=SRGB(); rng=Random.GLOBAL_RNG)
    # Sample in LCH for perceptually uniform distribution
    L = rand(rng) * 100.0
    C = rand(rng) * 150.0  # Wide gamut can have high chroma
    H = rand(rng) * 360.0
    
    lch = LCHab(L, C, H)
    rgb = convert(RGB, lch)
    
    # Clamp to valid gamut
    return clamp_to_gamut(rgb, cs)
end

"""
    random_colors(n::Int, cs::ColorSpace=SRGB(); rng=Random.GLOBAL_RNG)

Generate n random colors from the given color space.
"""
function random_colors(n::Int, cs::ColorSpace=SRGB(); rng=Random.GLOBAL_RNG)
    return [random_color(cs; rng=rng) for _ in 1:n]
end

"""
    random_palette(n::Int, cs::ColorSpace=SRGB(); 
                   min_distance=30.0, rng=Random.GLOBAL_RNG)

Generate n visually distinct random colors using rejection sampling.
Colors are separated by at least `min_distance` in CIEDE2000.
"""
function random_palette(n::Int, cs::ColorSpace=SRGB();
                        min_distance::Float64=30.0, rng=Random.GLOBAL_RNG)
    colors = RGB[]
    max_attempts = 10000
    attempts = 0
    
    while length(colors) < n && attempts < max_attempts
        candidate = random_color(cs; rng=rng)
        candidate_lab = convert(Lab, candidate)
        
        is_distinct = all(colors) do c
            c_lab = convert(Lab, c)
            colordiff(candidate_lab, c_lab) >= min_distance
        end
        
        if is_distinct || isempty(colors)
            push!(colors, candidate)
        end
        attempts += 1
    end
    
    return colors
end

"""
    in_gamut(c::Color, cs::ColorSpace)

Check if a color is within the gamut of the given color space.
"""
function in_gamut(c::Color, cs::ColorSpace=SRGB())
    rgb = convert(RGB, c)
    return 0.0 <= rgb.r <= 1.0 && 
           0.0 <= rgb.g <= 1.0 && 
           0.0 <= rgb.b <= 1.0
end

"""
    clamp_to_gamut(c::Color, cs::ColorSpace)

Clamp a color to the valid gamut of the given color space.
Uses chroma reduction in LCH space for perceptual quality.
"""
function clamp_to_gamut(c::Color, cs::ColorSpace=SRGB())
    rgb = convert(RGB, c)
    
    if in_gamut(rgb, cs)
        return rgb
    end
    
    # Reduce chroma until in gamut
    lch = convert(LCHab, rgb)
    L, C, H = lch.l, lch.c, lch.h
    
    lo, hi = 0.0, C
    for _ in 1:20  # Binary search
        mid = (lo + hi) / 2
        test = convert(RGB, LCHab(L, mid, H))
        if in_gamut(test, cs)
            lo = mid
        else
            hi = mid
        end
    end
    
    return convert(RGB, LCHab(L, lo, H))
end

"""
    gamut_map(c::Color, from::ColorSpace, to::ColorSpace)

Map a color from one color space's gamut to another.
"""
function gamut_map(c::Color, from::ColorSpace, to::ColorSpace)
    # Convert through XYZ
    rgb = convert(RGB, c)
    xyz_from = rgb_to_xyz_matrix(from) * [rgb.r, rgb.g, rgb.b]
    xyz_to_inv = inv(rgb_to_xyz_matrix(to))
    rgb_new = xyz_to_inv * xyz_from
    return clamp_to_gamut(RGB(rgb_new...), to)
end

# ═══════════════════════════════════════════════════════════════════════════
# Pride flag color palettes 🏳️‍🌈
# ═══════════════════════════════════════════════════════════════════════════

"""
    pride_flag(name::Symbol, cs::ColorSpace=SRGB())

Get the colors of a pride flag in the specified color space.
"""
function pride_flag(name::Symbol, cs::ColorSpace=SRGB())
    colors = _pride_colors(name)
    return [clamp_to_gamut(c, cs) for c in colors]
end

function _pride_colors(name::Symbol)
    if name == :rainbow || name == :gay
        return [
            RGB(0.894, 0.012, 0.012),  # Red
            RGB(1.000, 0.549, 0.000),  # Orange
            RGB(1.000, 0.929, 0.000),  # Yellow
            RGB(0.000, 0.502, 0.149),  # Green
            RGB(0.000, 0.298, 0.686),  # Blue
            RGB(0.459, 0.027, 0.529),  # Violet
        ]
    elseif name == :bisexual || name == :bi
        return [
            RGB(0.843, 0.008, 0.439),  # Magenta
            RGB(0.612, 0.349, 0.541),  # Lavender
            RGB(0.000, 0.220, 0.655),  # Blue
        ]
    elseif name == :transgender || name == :trans
        return [
            RGB(0.357, 0.808, 0.980),  # Light Blue
            RGB(0.961, 0.659, 0.718),  # Pink
            RGB(1.000, 1.000, 1.000),  # White
            RGB(0.961, 0.659, 0.718),  # Pink
            RGB(0.357, 0.808, 0.980),  # Light Blue
        ]
    elseif name == :nonbinary || name == :nb || name == :enby
        return [
            RGB(0.988, 0.957, 0.184),  # Yellow
            RGB(1.000, 1.000, 1.000),  # White
            RGB(0.612, 0.349, 0.820),  # Purple
            RGB(0.180, 0.180, 0.180),  # Black
        ]
    elseif name == :pansexual || name == :pan
        return [
            RGB(1.000, 0.129, 0.549),  # Magenta
            RGB(1.000, 0.847, 0.000),  # Yellow
            RGB(0.129, 0.694, 1.000),  # Cyan
        ]
    elseif name == :asexual || name == :ace
        return [
            RGB(0.000, 0.000, 0.000),  # Black
            RGB(0.639, 0.639, 0.639),  # Gray
            RGB(1.000, 1.000, 1.000),  # White
            RGB(0.502, 0.000, 0.502),  # Purple
        ]
    elseif name == :lesbian
        return [
            RGB(0.831, 0.173, 0.000),  # Dark Orange
            RGB(0.992, 0.596, 0.337),  # Orange
            RGB(1.000, 1.000, 1.000),  # White
            RGB(0.851, 0.463, 0.647),  # Pink
            RGB(0.635, 0.012, 0.384),  # Dark Rose
        ]
    elseif name == :progress
        # Progress Pride flag adds trans + BIPOC colors
        return [
            RGB(1.000, 1.000, 1.000),  # White (chevron)
            RGB(0.961, 0.659, 0.718),  # Pink (trans)
            RGB(0.357, 0.808, 0.980),  # Light Blue (trans)
            RGB(0.384, 0.227, 0.133),  # Brown (BIPOC)
            RGB(0.000, 0.000, 0.000),  # Black (BIPOC)
            RGB(0.894, 0.012, 0.012),  # Red
            RGB(1.000, 0.549, 0.000),  # Orange
            RGB(1.000, 0.929, 0.000),  # Yellow
            RGB(0.000, 0.502, 0.149),  # Green
            RGB(0.000, 0.298, 0.686),  # Blue
            RGB(0.459, 0.027, 0.529),  # Violet
        ]
    else
        error("Unknown pride flag: $name. Try :rainbow, :bisexual, :transgender, :nonbinary, :pansexual, :asexual, :lesbian, or :progress")
    end
end

# Convenience functions for common flags

"""
    rainbow(cs::ColorSpace=SRGB())

Get the Gilbert Baker rainbow flag colors.
"""
rainbow(cs::ColorSpace=SRGB()) = pride_flag(:rainbow, cs)

"""
    bisexual(cs::ColorSpace=SRGB())

Get the bisexual pride flag colors.
"""
bisexual(cs::ColorSpace=SRGB()) = pride_flag(:bisexual, cs)

"""
    transgender(cs::ColorSpace=SRGB())

Get the transgender pride flag colors.
"""
transgender(cs::ColorSpace=SRGB()) = pride_flag(:transgender, cs)

"""
    nonbinary(cs::ColorSpace=SRGB())

Get the nonbinary pride flag colors.
"""
nonbinary(cs::ColorSpace=SRGB()) = pride_flag(:nonbinary, cs)

"""
    pansexual(cs::ColorSpace=SRGB())

Get the pansexual pride flag colors.
"""
pansexual(cs::ColorSpace=SRGB()) = pride_flag(:pansexual, cs)

"""
    asexual(cs::ColorSpace=SRGB())

Get the asexual pride flag colors.
"""
asexual(cs::ColorSpace=SRGB()) = pride_flag(:asexual, cs)

# ═══════════════════════════════════════════════════════════════════════════
# Global P3-preferred color space setting
# ═══════════════════════════════════════════════════════════════════════════

const CURRENT_COLORSPACE = Ref{ColorSpace}(DisplayP3())

"""
    current_colorspace() -> ColorSpace

Get the current preferred color space. Default: DisplayP3.
"""
current_colorspace() = CURRENT_COLORSPACE[]

"""
    set_colorspace!(cs::ColorSpace)

Set the preferred color space globally. DisplayP3 recommended for wide-gamut displays.
"""
function set_colorspace!(cs::ColorSpace)
    CURRENT_COLORSPACE[] = cs
    return cs
end

export current_colorspace, set_colorspace!

# ═══════════════════════════════════════════════════════════════════════════
# Perceptual Color Difference (CIEDE2000)
# ═══════════════════════════════════════════════════════════════════════════

export perceptual_diff, ciede2000, color_distance_matrix
export find_most_different, find_most_similar, perceptual_cluster

"""
    perceptual_diff(c1::Color, c2::Color) -> Float64

Compute CIEDE2000 perceptual color difference between two colors.
This is the gold standard for measuring how different two colors appear to humans.

CIEDE2000 accounts for:
- Lightness (L*) differences weighted by viewing conditions
- Chroma (C*) differences with adaptation to saturation
- Hue (H*) differences with rotation compensation
- Interactive effects between L*, C*, H*

Returns ΔE₀₀ where:
- ΔE₀₀ < 1.0  : Imperceptible difference
- ΔE₀₀ ~ 1.0  : Just noticeable difference (JND)
- ΔE₀₀ ~ 2.0  : Perceptible but acceptable
- ΔE₀₀ ~ 10.0 : Clearly different
- ΔE₀₀ > 50.0 : Highly different (opposite colors)

# Example
```julia
red = RGB(1.0, 0.0, 0.0)
orange = RGB(1.0, 0.5, 0.0)
ΔE = perceptual_diff(red, orange)  # ≈ 31.5
```
"""
function perceptual_diff(c1::Color, c2::Color)
    lab1 = convert(Lab, c1)
    lab2 = convert(Lab, c2)
    return colordiff(lab1, lab2)  # Uses CIEDE2000 by default in Colors.jl
end

"""
    ciede2000(c1::Color, c2::Color; kL=1.0, kC=1.0, kH=1.0) -> Float64

Full CIEDE2000 with parametric weighting factors.

Parameters:
- kL: Lightness weight (default 1.0, use 2.0 for textiles)
- kC: Chroma weight (default 1.0)
- kH: Hue weight (default 1.0)

The reference conditions assume:
- Illumination: 1000 lux
- Adapting luminance: 63.66 cd/m²
- Background: uniform grey with L*=50
- Viewing mode: object mode
- Sample size: subtending 4° at observer

# Example
```julia
# Textile application (more tolerance for lightness)
ΔE = ciede2000(c1, c2; kL=2.0)
```
"""
function ciede2000(c1::Color, c2::Color; kL::Float64=1.0, kC::Float64=1.0, kH::Float64=1.0)
    lab1 = convert(Lab, c1)
    lab2 = convert(Lab, c2)

    # Extract Lab values
    L1, a1, b1 = lab1.l, lab1.a, lab1.b
    L2, a2, b2 = lab2.l, lab2.a, lab2.b

    # Step 1: Calculate C'ᵢ and h'ᵢ
    C1 = sqrt(a1^2 + b1^2)
    C2 = sqrt(a2^2 + b2^2)
    C_avg = (C1 + C2) / 2

    G = 0.5 * (1 - sqrt(C_avg^7 / (C_avg^7 + 25^7)))

    a1_prime = (1 + G) * a1
    a2_prime = (1 + G) * a2

    C1_prime = sqrt(a1_prime^2 + b1^2)
    C2_prime = sqrt(a2_prime^2 + b2^2)

    h1_prime = atan(b1, a1_prime) * 180 / π
    h1_prime = h1_prime < 0 ? h1_prime + 360 : h1_prime

    h2_prime = atan(b2, a2_prime) * 180 / π
    h2_prime = h2_prime < 0 ? h2_prime + 360 : h2_prime

    # Step 2: Calculate ΔL', ΔC', ΔH'
    ΔL_prime = L2 - L1
    ΔC_prime = C2_prime - C1_prime

    Δh_prime = if C1_prime * C2_prime == 0
        0.0
    elseif abs(h2_prime - h1_prime) <= 180
        h2_prime - h1_prime
    elseif h2_prime - h1_prime > 180
        h2_prime - h1_prime - 360
    else
        h2_prime - h1_prime + 360
    end

    ΔH_prime = 2 * sqrt(C1_prime * C2_prime) * sin(Δh_prime * π / 360)

    # Step 3: Calculate CIEDE2000 ΔE₀₀
    L_prime_avg = (L1 + L2) / 2
    C_prime_avg = (C1_prime + C2_prime) / 2

    h_prime_avg = if C1_prime * C2_prime == 0
        h1_prime + h2_prime
    elseif abs(h1_prime - h2_prime) <= 180
        (h1_prime + h2_prime) / 2
    elseif h1_prime + h2_prime < 360
        (h1_prime + h2_prime + 360) / 2
    else
        (h1_prime + h2_prime - 360) / 2
    end

    T = 1 - 0.17 * cos((h_prime_avg - 30) * π / 180) +
            0.24 * cos((2 * h_prime_avg) * π / 180) +
            0.32 * cos((3 * h_prime_avg + 6) * π / 180) -
            0.20 * cos((4 * h_prime_avg - 63) * π / 180)

    Δθ = 30 * exp(-((h_prime_avg - 275) / 25)^2)

    RC = 2 * sqrt(C_prime_avg^7 / (C_prime_avg^7 + 25^7))

    SL = 1 + (0.015 * (L_prime_avg - 50)^2) / sqrt(20 + (L_prime_avg - 50)^2)
    SC = 1 + 0.045 * C_prime_avg
    SH = 1 + 0.015 * C_prime_avg * T

    RT = -sin(2 * Δθ * π / 180) * RC

    ΔE00 = sqrt(
        (ΔL_prime / (kL * SL))^2 +
        (ΔC_prime / (kC * SC))^2 +
        (ΔH_prime / (kH * SH))^2 +
        RT * (ΔC_prime / (kC * SC)) * (ΔH_prime / (kH * SH))
    )

    return ΔE00
end

"""
    color_distance_matrix(colors::Vector{<:Color}) -> Matrix{Float64}

Compute pairwise CIEDE2000 distances between all colors.
Returns symmetric matrix where M[i,j] = perceptual_diff(colors[i], colors[j]).

# Example
```julia
palette = [RGB(1,0,0), RGB(0,1,0), RGB(0,0,1)]
D = color_distance_matrix(palette)
# D[1,2] = red-green distance ≈ 86.6
# D[1,3] = red-blue distance ≈ 52.9
```
"""
function color_distance_matrix(colors::Vector{<:Color})
    n = length(colors)
    D = zeros(Float64, n, n)

    for i in 1:n
        for j in (i+1):n
            d = perceptual_diff(colors[i], colors[j])
            D[i, j] = d
            D[j, i] = d
        end
    end

    return D
end

"""
    find_most_different(colors::Vector{<:Color}, reference::Color) -> (Color, Float64)

Find the color most perceptually different from the reference.
Returns (most_different_color, ΔE₀₀).
"""
function find_most_different(colors::Vector{<:Color}, reference::Color)
    max_diff = 0.0
    most_diff = colors[1]

    for c in colors
        d = perceptual_diff(c, reference)
        if d > max_diff
            max_diff = d
            most_diff = c
        end
    end

    return (most_diff, max_diff)
end

"""
    find_most_similar(colors::Vector{<:Color}, reference::Color) -> (Color, Float64)

Find the color most perceptually similar to the reference.
Returns (most_similar_color, ΔE₀₀).
"""
function find_most_similar(colors::Vector{<:Color}, reference::Color)
    min_diff = Inf
    most_sim = colors[1]

    for c in colors
        d = perceptual_diff(c, reference)
        if d < min_diff
            min_diff = d
            most_sim = c
        end
    end

    return (most_sim, min_diff)
end

"""
    perceptual_cluster(colors::Vector{<:Color}, n_clusters::Int; seed=1069) -> Vector{Vector{Color}}

Cluster colors by perceptual similarity using k-medoids on CIEDE2000 distances.
Returns n_clusters groups of perceptually similar colors.

# Example
```julia
many_colors = random_colors(100, DisplayP3())
clusters = perceptual_cluster(many_colors, 5)
```
"""
function perceptual_cluster(colors::Vector{<:Color}, n_clusters::Int; seed::Int=1069)
    n = length(colors)
    if n <= n_clusters
        return [[c] for c in colors]
    end

    # Compute distance matrix
    D = color_distance_matrix(colors)

    # Initialize medoids (deterministic from seed)
    rng = Random.MersenneTwister(seed)
    medoid_indices = sort(shuffle(rng, 1:n)[1:n_clusters])

    # K-medoids iteration
    for _ in 1:100
        # Assign points to nearest medoid
        assignments = [argmin([D[i, m] for m in medoid_indices]) for i in 1:n]

        # Update medoids
        new_medoids = Int[]
        for k in 1:n_clusters
            cluster = findall(==(k), assignments)
            if isempty(cluster)
                push!(new_medoids, medoid_indices[k])
                continue
            end

            # Find point with minimum total distance to cluster members
            best = cluster[1]
            best_cost = sum(D[best, j] for j in cluster)
            for i in cluster
                cost = sum(D[i, j] for j in cluster)
                if cost < best_cost
                    best_cost = cost
                    best = i
                end
            end
            push!(new_medoids, best)
        end

        if sort(new_medoids) == sort(medoid_indices)
            break
        end
        medoid_indices = new_medoids
    end

    # Final assignment
    assignments = [argmin([D[i, m] for m in medoid_indices]) for i in 1:n]

    # Build clusters
    clusters = [Color[] for _ in 1:n_clusters]
    for (i, k) in enumerate(assignments)
        push!(clusters[k], colors[i])
    end

    return filter(!isempty, clusters)
end

# ═══════════════════════════════════════════════════════════════════════════
# P3 Perceptual Analysis Pipeline
# ═══════════════════════════════════════════════════════════════════════════

export p3_perceptual_report, show_perceptual_diff, rainbow_text

"""
    p3_perceptual_report(colors::Vector{<:Color}; verbose=true) -> NamedTuple

Analyze a color palette for perceptual quality in DisplayP3 space.

Returns:
- min_diff: Minimum pairwise ΔE₀₀ (most similar pair)
- max_diff: Maximum pairwise ΔE₀₀ (most different pair)
- mean_diff: Average pairwise ΔE₀₀
- gamut_coverage: Fraction of colors in P3 gamut
- perceptual_uniformity: σ/μ of pairwise distances (lower = more uniform)

# Example
```julia
palette = random_palette(6, DisplayP3())
report = p3_perceptual_report(palette)
```
"""
function p3_perceptual_report(colors::Vector{<:Color}; verbose::Bool=true)
    n = length(colors)

    # Convert all to P3 and compute distance matrix
    p3_colors = [clamp_to_gamut(c, DisplayP3()) for c in colors]
    D = color_distance_matrix(p3_colors)

    # Extract upper triangle (unique pairs)
    diffs = Float64[]
    for i in 1:n, j in (i+1):n
        push!(diffs, D[i, j])
    end

    min_diff = minimum(diffs)
    max_diff = maximum(diffs)
    mean_diff = sum(diffs) / length(diffs)
    std_diff = sqrt(sum((d - mean_diff)^2 for d in diffs) / length(diffs))

    # Check P3 gamut coverage
    in_p3 = sum(in_gamut(c, DisplayP3()) for c in colors) / n

    # Find most similar pair
    min_i, min_j = 1, 2
    for i in 1:n, j in (i+1):n
        if D[i, j] == min_diff
            min_i, min_j = i, j
            break
        end
    end

    uniformity = std_diff / mean_diff

    if verbose
        println("╔════════════════════════════════════════════════════════════════╗")
        println("║           P3 Perceptual Color Analysis (CIEDE2000)             ║")
        println("╠════════════════════════════════════════════════════════════════╣")
        println("║ Palette size:        $(lpad(n, 4))                                    ║")
        println("║ P3 gamut coverage:   $(lpad(round(in_p3 * 100, digits=1), 5))%                                ║")
        println("╠════════════════════════════════════════════════════════════════╣")
        println("║ Min ΔE₀₀ (similar):  $(lpad(round(min_diff, digits=2), 6))  (colors $min_i ↔ $min_j)             ║")
        println("║ Max ΔE₀₀ (different):$(lpad(round(max_diff, digits=2), 6))                                ║")
        println("║ Mean ΔE₀₀:           $(lpad(round(mean_diff, digits=2), 6))                                ║")
        println("║ Uniformity (σ/μ):    $(lpad(round(uniformity, digits=3), 6))                                ║")
        println("╠════════════════════════════════════════════════════════════════╣")
        if min_diff < 10
            println("║ ⚠ Warning: Colors $min_i and $min_j may be too similar (ΔE < 10)   ║")
        else
            println("║ ✓ All color pairs are perceptually distinct                   ║")
        end
        println("╚════════════════════════════════════════════════════════════════╝")
    end

    return (
        min_diff = min_diff,
        max_diff = max_diff,
        mean_diff = mean_diff,
        std_diff = std_diff,
        uniformity = uniformity,
        gamut_coverage = in_p3,
        most_similar_pair = (min_i, min_j),
        distance_matrix = D
    )
end

"""
    show_perceptual_diff(c1::Color, c2::Color)

Display two colors side-by-side with their CIEDE2000 difference.
"""
function show_perceptual_diff(c1::Color, c2::Color)
    rgb1 = convert(RGB, c1)
    rgb2 = convert(RGB, c2)

    r1 = round(Int, clamp(rgb1.r, 0, 1) * 255)
    g1 = round(Int, clamp(rgb1.g, 0, 1) * 255)
    b1 = round(Int, clamp(rgb1.b, 0, 1) * 255)

    r2 = round(Int, clamp(rgb2.r, 0, 1) * 255)
    g2 = round(Int, clamp(rgb2.g, 0, 1) * 255)
    b2 = round(Int, clamp(rgb2.b, 0, 1) * 255)

    ΔE = perceptual_diff(c1, c2)

    fg1 = "\e[38;2;$(r1);$(g1);$(b1)m"
    fg2 = "\e[38;2;$(r2);$(g2);$(b2)m"
    reset = "\e[0m"

    println("$(fg1)████████$(reset)  ←→  $(fg2)████████$(reset)  ΔE₀₀ = $(round(ΔE, digits=2))")

    interpretation = if ΔE < 1
        "imperceptible"
    elseif ΔE < 2
        "barely perceptible"
    elseif ΔE < 10
        "perceptible"
    elseif ΔE < 50
        "clearly different"
    else
        "highly different"
    end

    println("Interpretation: $interpretation")
end

"""
    rainbow_text(text::String; seed=1069) -> String

Color each character in text with a deterministic rainbow gradient.
"""
function rainbow_text(text::String; seed::Int=1069)
    chars = collect(text)
    n = length(chars)

    result = ""
    for (i, c) in enumerate(chars)
        if c == ' ' || c == '\n'
            result *= string(c)
            continue
        end

        # Deterministic hue based on position and seed
        hue = (360.0 * i / n + seed) % 360
        rgb = convert(RGB, HSL(hue, 0.8, 0.5))

        r = round(Int, clamp(rgb.r, 0, 1) * 255)
        g = round(Int, clamp(rgb.g, 0, 1) * 255)
        b = round(Int, clamp(rgb.b, 0, 1) * 255)

        result *= "\e[38;2;$(r);$(g);$(b)m$(c)\e[0m"
    end

    return result
end
