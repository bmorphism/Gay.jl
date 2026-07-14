module GayFractalExt

# Package extension: loads only when FractalDimensions.jl is present alongside Gay.
# Lazily implements fractal dimension analysis for Gay.jl colors.

using Gay
using FractalDimensions

# Helper to parse Gay.jl's "#RRGGBB" hex string into a 3-tuple of Float64 in [0, 1]³
function hex_to_rgb(hex::AbstractString)
    r = parse(Int, hex[2:3]; base=16) / 255.0
    g = parse(Int, hex[4:5]; base=16) / 255.0
    b = parse(Int, hex[6:7]; base=16) / 255.0
    (r, g, b)
end

# Directly extend FractalDimensions.grassberger_proccacia_dim to accept hex colors, WalkResult, and Integer count.
# This ensures a deep integration where calling FractalDimensions's own API automatically overrides standard behavior.
function FractalDimensions.grassberger_proccacia_dim(colors::AbstractVector{<:AbstractString}; metric=:perceptual, kwargs...)
    n = length(colors)
    if n < 3
        return 0.0
    end

    if metric === :perceptual && hasmethod(Gay.gay_colordiff, Tuple{AbstractString, AbstractString})
        # Use 1D indices and map the custom norm to CIEDE2000 distances between colors at those indices
        mat = zeros(Float64, n, 1)
        for i in 1:n
            mat[i, 1] = Float64(i)
        end
        dataset = StateSpaceSet(mat)
        perceptual_norm = (x, y) -> begin
            idx1 = clamp(Int(round(x[1])), 1, n)
            idx2 = clamp(Int(round(y[1])), 1, n)
            Gay.gay_colordiff(colors[idx1], colors[idx2])
        end
        return FractalDimensions.grassberger_proccacia_dim(dataset; norm = perceptual_norm, kwargs...)
    else
        if metric === :perceptual
            @warn "Colors.jl is not loaded or GayColorsExt is missing; falling back to Euclidean distance in sRGB space."
        end
        mat = zeros(Float64, n, 3)
        for i in 1:n
            r, g, b = hex_to_rgb(colors[i])
            mat[i, 1] = r
            mat[i, 2] = g
            mat[i, 3] = b
        end
        dataset = StateSpaceSet(mat)
        return FractalDimensions.grassberger_proccacia_dim(dataset; kwargs...)
    end
end

function FractalDimensions.grassberger_proccacia_dim(walk::WalkResult; metric=:perceptual, kwargs...)
    colors = [step.color for step in walk.steps]
    FractalDimensions.grassberger_proccacia_dim(colors; metric=metric, kwargs...)
end

function FractalDimensions.grassberger_proccacia_dim(n::Integer; seed::Integer=Gay.GAY_SEED, gamma::Integer=Gay.GOLDEN_GAMMA, metric=:perceptual, kwargs...)
    colors = [Gay.color_at(i; seed=seed, gamma=gamma) for i in 0:(n-1)]
    FractalDimensions.grassberger_proccacia_dim(colors; metric=metric, kwargs...)
end

# Expose backward compatibility functions on the Gay module namespace
function Gay.gay_fractal_dimension(colors::AbstractVector{<:AbstractString}; kwargs...)
    FractalDimensions.grassberger_proccacia_dim(colors; kwargs...)
end

function Gay.gay_fractal_dimension(walk::WalkResult; kwargs...)
    FractalDimensions.grassberger_proccacia_dim(walk; kwargs...)
end

function Gay.gay_fractal_dimension(n::Integer; seed::Integer=Gay.GAY_SEED, gamma::Integer=Gay.GOLDEN_GAMMA, kwargs...)
    FractalDimensions.grassberger_proccacia_dim(n; seed=seed, gamma=gamma, kwargs...)
end

end # module GayFractalExt
