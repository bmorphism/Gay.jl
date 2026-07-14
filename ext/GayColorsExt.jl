module GayColorsExt

# Package extension: loads only when Colors.jl is present alongside Gay.
# This is where "the best of Gay.jl" (perceptual color science, wide-gamut) drops
# in — without the core seed depending on Colors or pulling in 36k LoC of stubs.

using Gay
using Colors

# Parse a Gay.jl "#RRGGBB" into a Colors.jl colorant (sRGB).
Gay.gay_colorant(hex::AbstractString) = parse(Colorant, hex)

# Perceptual CIEDE2000 distance between two Gay colors — the color-science the
# minimal seed lacks. Same SplitMix→Okhsl pipeline, now measured in Lab space.
function Gay.gay_colordiff(i::Integer, j::Integer;
                           seed::Integer=Gay.GAY_SEED, gamma::Integer=Gay.GOLDEN_GAMMA)
    a = parse(Colorant, Gay.color_at(i; seed=seed, gamma=gamma))
    b = parse(Colorant, Gay.color_at(j; seed=seed, gamma=gamma))
    Float64(colordiff(a, b))
end

# String-to-string perceptual distance override
function Gay.gay_colordiff(c1::AbstractString, c2::AbstractString)
    a = parse(Colorant, c1)
    b = parse(Colorant, c2)
    Float64(colordiff(a, b))
end

# Directly extend Colors.colordiff for string arguments to override standard color behavior
function Colors.colordiff(c1::AbstractString, c2::AbstractString; kwargs...)
    a = parse(Colorant, c1)
    b = parse(Colorant, c2)
    Float64(colordiff(a, b; kwargs...))
end

end # module GayColorsExt
