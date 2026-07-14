module GayRipsererExt

# Package extension: loads only when Ripserer.jl is present alongside Gay.
# Lazily implements persistent homology topological analysis for Gay.jl colors.

using Gay
using Ripserer

# Helper to parse Gay.jl's "#RRGGBB" hex string into a 3-tuple of Float64 in [0, 1]³
function hex_to_rgb(hex::AbstractString)
    r = parse(Int, hex[2:3]; base=16) / 255.0
    g = parse(Int, hex[4:5]; base=16) / 255.0
    b = parse(Int, hex[6:7]; base=16) / 255.0
    (r, g, b)
end

# Directly extend Ripserer.ripserer to accept hex colors, WalkResult, and Integer count.
# This ensures a deep integration where calling Ripserer's own API automatically overrides standard behavior.
function Ripserer.ripserer(colors::AbstractVector{<:AbstractString}; dim_max::Integer=1, metric=:perceptual, kwargs...)
    n = length(colors)
    if n < 2
        return []
    end
    d_max = clamp(dim_max, 0, n - 2)

    if metric === :perceptual && hasmethod(Gay.gay_colordiff, Tuple{AbstractString, AbstractString})
        # Build pairwise distance matrix using CIEDE2000 from Gay.gay_colordiff
        D = zeros(Float64, n, n)
        for j in 1:n
            for i in (j+1):n
                d = Gay.gay_colordiff(colors[i], colors[j])
                D[i, j] = d
                D[j, i] = d
            end
        end
        return Ripserer.ripserer(D; dim_max=d_max, kwargs...)
    else
        if metric === :perceptual
            @warn "Colors.jl is not loaded; falling back to Euclidean distance in sRGB space."
        end
        points = [hex_to_rgb(c) for c in colors]
        return Ripserer.ripserer(points; dim_max=d_max, kwargs...)
    end
end

function Ripserer.ripserer(walk::WalkResult; dim_max::Integer=1, metric=:perceptual, kwargs...)
    colors = [step.color for step in walk.steps]
    Ripserer.ripserer(colors; dim_max=dim_max, metric=metric, kwargs...)
end

function Ripserer.ripserer(n::Integer; seed::Integer=Gay.GAY_SEED, gamma::Integer=Gay.GOLDEN_GAMMA, dim_max::Integer=1, metric=:perceptual, kwargs...)
    colors = [Gay.color_at(i; seed=seed, gamma=gamma) for i in 0:(n-1)]
    Ripserer.ripserer(colors; dim_max=dim_max, metric=metric, kwargs...)
end

# Expose backward compatibility functions on the Gay module namespace
function Gay.gay_ripserer(colors::AbstractVector{<:AbstractString}; dim_max::Integer=1, kwargs...)
    Ripserer.ripserer(colors; dim_max=dim_max, kwargs...)
end

function Gay.gay_ripserer(walk::WalkResult; dim_max::Integer=1, kwargs...)
    Ripserer.ripserer(walk; dim_max=dim_max, kwargs...)
end

function Gay.gay_ripserer(n::Integer; seed::Integer=Gay.GAY_SEED, gamma::Integer=Gay.GOLDEN_GAMMA, dim_max::Integer=1, kwargs...)
    Ripserer.ripserer(n; seed=seed, gamma=gamma, dim_max=dim_max, kwargs...)
end

end # module GayRipsererExt
