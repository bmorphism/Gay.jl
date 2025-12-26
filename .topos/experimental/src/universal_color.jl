# Universal Gay Color - Maximally flexible multiparadigm color type
# Provides abstract color types with runtime/colorspace polymorphism

module UniversalColorModule

using Colors

export AbstractGayColorant, AbstractGayColor, AbstractGayAlpha, AbstractGaySpectral
export GayRGB, GayHSV, GayLab, GayGray, GayRGBA, GaySpectral, UniversalGayColor
export GayColorSpace, GaySRGB, GayP3, GayRec2020, GaySpectralSpace, GayCustomSpace
export GayRuntime, GayCPU, GayCUDA, GayMetal, GayAMDGPU, GayoneAPI, GayTPU
export GayEval, GayEager, GayLazy, GaySymbolic
export gay_color, gay_rgb, gay_hsv, gay_hash, gay_fingerprint, gay_verify_spi
export to_runtime, to_colorspace, to_precision, to_eval
export materialize, defer, force, gay_mix, gay_complement

# ═══════════════════════════════════════════════════════════════════════════
# Abstract Color Type Hierarchy
# ═══════════════════════════════════════════════════════════════════════════

abstract type AbstractGayColorant end
abstract type AbstractGayColor <: AbstractGayColorant end
abstract type AbstractGayAlpha <: AbstractGayColorant end
abstract type AbstractGaySpectral <: AbstractGayColorant end

# ═══════════════════════════════════════════════════════════════════════════
# Runtime Types
# ═══════════════════════════════════════════════════════════════════════════

abstract type GayRuntime end
struct GayCPU <: GayRuntime end
struct GayCUDA <: GayRuntime end
struct GayMetal <: GayRuntime end
struct GayAMDGPU <: GayRuntime end
struct GayoneAPI <: GayRuntime end
struct GayTPU <: GayRuntime end

# ═══════════════════════════════════════════════════════════════════════════
# Color Space Types
# ═══════════════════════════════════════════════════════════════════════════

abstract type GayColorSpace end
struct GaySRGB <: GayColorSpace end
struct GayP3 <: GayColorSpace end
struct GayRec2020 <: GayColorSpace end
struct GaySpectralSpace <: GayColorSpace end
struct GayCustomSpace <: GayColorSpace
    name::String
end

# ═══════════════════════════════════════════════════════════════════════════
# Evaluation Strategy Types
# ═══════════════════════════════════════════════════════════════════════════

abstract type GayEval end
struct GayEager <: GayEval end
struct GayLazy <: GayEval end
struct GaySymbolic <: GayEval end

# ═══════════════════════════════════════════════════════════════════════════
# Concrete Color Types
# ═══════════════════════════════════════════════════════════════════════════

struct GayRGB{T<:Real} <: AbstractGayColor
    r::T
    g::T
    b::T
end

GayRGB(r, g, b) = GayRGB{Float64}(Float64(r), Float64(g), Float64(b))

struct GayHSV{T<:Real} <: AbstractGayColor
    h::T
    s::T
    v::T
end

GayHSV(h, s, v) = GayHSV{Float64}(Float64(h), Float64(s), Float64(v))

struct GayLab{T<:Real} <: AbstractGayColor
    l::T
    a::T
    b::T
end

GayLab(l, a, b) = GayLab{Float64}(Float64(l), Float64(a), Float64(b))

struct GayGray{T<:Real} <: AbstractGayColor
    val::T
end

GayGray(v) = GayGray{Float64}(Float64(v))

struct GayRGBA{T<:Real} <: AbstractGayAlpha
    r::T
    g::T
    b::T
    a::T
end

GayRGBA(r, g, b, a) = GayRGBA{Float64}(Float64(r), Float64(g), Float64(b), Float64(a))

struct GaySpectral{N, T<:Real} <: AbstractGaySpectral
    wavelengths::NTuple{N, T}
    values::NTuple{N, T}
end

# ═══════════════════════════════════════════════════════════════════════════
# Universal Gay Color - Polymorphic wrapper
# ═══════════════════════════════════════════════════════════════════════════

struct UniversalGayColor{C<:AbstractGayColorant, S<:GayColorSpace, R<:GayRuntime, E<:GayEval}
    color::C
    colorspace::S
    runtime::R
    eval_strategy::E
end

function UniversalGayColor(color::AbstractGayColorant)
    UniversalGayColor(color, GaySRGB(), GayCPU(), GayEager())
end

# ═══════════════════════════════════════════════════════════════════════════
# Constructor Functions
# ═══════════════════════════════════════════════════════════════════════════

gay_color(r, g, b) = GayRGB(r, g, b)
gay_rgb(r, g, b) = GayRGB(r, g, b)
gay_hsv(h, s, v) = GayHSV(h, s, v)

function gay_hash(color::AbstractGayColor)::UInt64
    if color isa GayRGB
        bits_r = reinterpret(UInt64, Float64(color.r))
        bits_g = reinterpret(UInt64, Float64(color.g))
        bits_b = reinterpret(UInt64, Float64(color.b))
        return bits_r ⊻ (bits_g << 21) ⊻ (bits_b << 42)
    else
        return UInt64(0)
    end
end

function gay_fingerprint(colors::Vector{<:AbstractGayColor})::UInt64
    fp = UInt64(0)
    for c in colors
        fp ⊻= gay_hash(c)
    end
    return fp
end

function gay_verify_spi(colors1::Vector{<:AbstractGayColor}, colors2::Vector{<:AbstractGayColor})::Bool
    return gay_fingerprint(colors1) == gay_fingerprint(colors2)
end

# ═══════════════════════════════════════════════════════════════════════════
# Conversion Functions
# ═══════════════════════════════════════════════════════════════════════════

to_runtime(c::UniversalGayColor, r::GayRuntime) = UniversalGayColor(c.color, c.colorspace, r, c.eval_strategy)
to_colorspace(c::UniversalGayColor, s::GayColorSpace) = UniversalGayColor(c.color, s, c.runtime, c.eval_strategy)
to_precision(c::UniversalGayColor, ::Type{T}) where T = c  # Placeholder
to_eval(c::UniversalGayColor, e::GayEval) = UniversalGayColor(c.color, c.colorspace, c.runtime, e)

# ═══════════════════════════════════════════════════════════════════════════
# Lazy/Eager Evaluation
# ═══════════════════════════════════════════════════════════════════════════

materialize(c::UniversalGayColor) = c
defer(c::UniversalGayColor) = to_eval(c, GayLazy())
force(c::UniversalGayColor) = to_eval(c, GayEager())

# ═══════════════════════════════════════════════════════════════════════════
# Color Operations
# ═══════════════════════════════════════════════════════════════════════════

function gay_mix(c1::GayRGB, c2::GayRGB, t::Real)
    r = c1.r * (1 - t) + c2.r * t
    g = c1.g * (1 - t) + c2.g * t
    b = c1.b * (1 - t) + c2.b * t
    return GayRGB(r, g, b)
end

function gay_complement(c::GayRGB)
    return GayRGB(1.0 - c.r, 1.0 - c.g, 1.0 - c.b)
end

end # module UniversalColorModule
