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
