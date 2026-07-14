module GayPersistenceDiagramsExt

# Package extension: loads only when PersistenceDiagrams.jl is present alongside Gay.
# Lazily implements first-class persistent homology diagram types and matching metrics.

using Gay
using PersistenceDiagrams
using Ripserer

# ------------------------------------------------------------------------------
# 1. Implement PersistenceDiagrams.PersistenceDiagram standard overloads
# ------------------------------------------------------------------------------

function PersistenceDiagrams.PersistenceDiagram(colors::AbstractVector{<:AbstractString}; dim::Integer=0, kwargs...)
    # Ensure dim_max is at least dim
    diags = Gay.gay_ripserer(colors; dim_max=max(1, dim), kwargs...)
    for diag in diags
        if PersistenceDiagrams.dim(diag) == dim
            return diag
        end
    end
    # Fallback/Empty diagram if not found
    return PersistenceDiagrams.PersistenceDiagram(PersistenceDiagrams.PersistenceInterval[]; dim=dim)
end

function PersistenceDiagrams.PersistenceDiagram(walk::WalkResult; dim::Integer=0, kwargs...)
    colors = [step.color for step in walk.steps]
    PersistenceDiagrams.PersistenceDiagram(colors; dim=dim, kwargs...)
end

function PersistenceDiagrams.PersistenceDiagram(n::Integer; seed::Integer=Gay.GAY_SEED, gamma::Integer=Gay.GOLDEN_GAMMA, dim::Integer=0, kwargs...)
    colors = [Gay.color_at(i; seed=seed, gamma=gamma) for i in 0:(n-1)]
    PersistenceDiagrams.PersistenceDiagram(colors; dim=dim, kwargs...)
end

# ------------------------------------------------------------------------------
# 2. Implement Gay.GayPersistenceDiagram constructors and methods
# ------------------------------------------------------------------------------

function Gay.GayPersistenceDiagram(colors::AbstractVector{<:AbstractString}; dim::Integer=0, kwargs...)
    diag = PersistenceDiagrams.PersistenceDiagram(colors; dim=dim, kwargs...)
    Gay.GayPersistenceDiagram{typeof(diag), typeof(colors), eltype(diag)}(diag, colors, convert(Vector{String}, colors), Int(dim))
end

function Gay.GayPersistenceDiagram(walk::WalkResult; dim::Integer=0, kwargs...)
    colors = [step.color for step in walk.steps]
    diag = PersistenceDiagrams.PersistenceDiagram(walk; dim=dim, kwargs...)
    Gay.GayPersistenceDiagram{typeof(diag), typeof(walk), eltype(diag)}(diag, walk, colors, Int(dim))
end

function Gay.GayPersistenceDiagram(n::Integer; seed::Integer=Gay.GAY_SEED, gamma::Integer=Gay.GOLDEN_GAMMA, dim::Integer=0, kwargs...)
    colors = [Gay.color_at(i; seed=seed, gamma=gamma) for i in 0:(n-1)]
    diag = PersistenceDiagrams.PersistenceDiagram(n; seed=seed, gamma=gamma, dim=dim, kwargs...)
    # Source is represented as the integer count
    Gay.GayPersistenceDiagram{typeof(diag), typeof(n), eltype(diag)}(diag, n, colors, Int(dim))
end

# Implement PersistenceDiagrams methods on GayPersistenceDiagram
PersistenceDiagrams.dim(gpd::GayPersistenceDiagram) = gpd.dim
PersistenceDiagrams.threshold(gpd::GayPersistenceDiagram) = PersistenceDiagrams.threshold(gpd.diagram)
PersistenceDiagrams.PersistenceDiagram(gpd::GayPersistenceDiagram) = gpd.diagram
Base.convert(::Type{PersistenceDiagrams.PersistenceDiagram}, gpd::GayPersistenceDiagram) = gpd.diagram
Base.IndexStyle(::Type{<:GayPersistenceDiagram}) = Base.IndexLinear()

# Beautified printing for GayPersistenceDiagram
function Base.show(io::IO, mime::MIME"text/plain", gpd::GayPersistenceDiagram)
    println(io, "🌈 \e[1m\e[35mGayPersistenceDiagram\e[0m{dim=$(gpd.dim)} with $(length(gpd)) intervals:")
    println(io, "  • \e[34msource:\e[0m $(summary(gpd.source))")
    println(io, "  • \e[32mcolors:\e[0m $(length(gpd.colors))-color Gay.jl palette")
    if length(gpd) > 0
        println(io, "  • \e[33mintervals:\e[0m")
        show(io, mime, gpd.diagram)
    else
        println(io, "  • \e[31mempty/no intervals found\e[0m")
    end
end

function Base.show(io::IO, gpd::GayPersistenceDiagram)
    print(io, "GayPersistenceDiagram(dim=$(gpd.dim), intervals=$(length(gpd)))")
end

# ------------------------------------------------------------------------------
# 3. Extend standard Bottleneck, Wasserstein and matching for GayPersistenceDiagram
# ------------------------------------------------------------------------------

function (val::Bottleneck)(g1::GayPersistenceDiagram, g2::GayPersistenceDiagram; kwargs...)
    val(g1.diagram, g2.diagram; kwargs...)
end

function (val::Wasserstein)(g1::GayPersistenceDiagram, g2::GayPersistenceDiagram; kwargs...)
    val(g1.diagram, g2.diagram; kwargs...)
end

function PersistenceDiagrams.matching(val::Union{Bottleneck, Wasserstein}, g1::GayPersistenceDiagram, g2::GayPersistenceDiagram; kwargs...)
    PersistenceDiagrams.matching(val, g1.diagram, g2.diagram; kwargs...)
end

# Standard Bottleneck/Wasserstein overloads for raw walk results, colors and ints
function (val::Bottleneck)(w1::WalkResult, w2::WalkResult; dim::Integer=0, kwargs...)
    diag1 = PersistenceDiagrams.PersistenceDiagram(w1; dim=dim, kwargs...)
    diag2 = PersistenceDiagrams.PersistenceDiagram(w2; dim=dim, kwargs...)
    val(diag1, diag2)
end

function (val::Wasserstein)(w1::WalkResult, w2::WalkResult; dim::Integer=0, kwargs...)
    diag1 = PersistenceDiagrams.PersistenceDiagram(w1; dim=dim, kwargs...)
    diag2 = PersistenceDiagrams.PersistenceDiagram(w2; dim=dim, kwargs...)
    val(diag1, diag2)
end

function (val::Bottleneck)(c1::AbstractVector{<:AbstractString}, c2::AbstractVector{<:AbstractString}; dim::Integer=0, kwargs...)
    diag1 = PersistenceDiagrams.PersistenceDiagram(c1; dim=dim, kwargs...)
    diag2 = PersistenceDiagrams.PersistenceDiagram(c2; dim=dim, kwargs...)
    val(diag1, diag2)
end

function (val::Wasserstein)(c1::AbstractVector{<:AbstractString}, c2::AbstractVector{<:AbstractString}; dim::Integer=0, kwargs...)
    diag1 = PersistenceDiagrams.PersistenceDiagram(c1; dim=dim, kwargs...)
    diag2 = PersistenceDiagrams.PersistenceDiagram(c2; dim=dim, kwargs...)
    val(diag1, diag2)
end

function (val::Bottleneck)(n1::Integer, n2::Integer; seed1::Integer=Gay.GAY_SEED, gamma1::Integer=Gay.GOLDEN_GAMMA, seed2::Integer=Gay.GAY_SEED, gamma2::Integer=Gay.GOLDEN_GAMMA, dim::Integer=0, kwargs...)
    diag1 = PersistenceDiagrams.PersistenceDiagram(n1; seed=seed1, gamma=gamma1, dim=dim, kwargs...)
    diag2 = PersistenceDiagrams.PersistenceDiagram(n2; seed=seed2, gamma=gamma2, dim=dim, kwargs...)
    val(diag1, diag2)
end

function (val::Wasserstein)(n1::Integer, n2::Integer; seed1::Integer=Gay.GAY_SEED, gamma1::Integer=Gay.GOLDEN_GAMMA, seed2::Integer=Gay.GAY_SEED, gamma2::Integer=Gay.GOLDEN_GAMMA, dim::Integer=0, kwargs...)
    diag1 = PersistenceDiagrams.PersistenceDiagram(n1; seed=seed1, gamma=gamma1, dim=dim, kwargs...)
    diag2 = PersistenceDiagrams.PersistenceDiagram(n2; seed=seed2, gamma=gamma2, dim=dim, kwargs...)
    val(diag1, diag2)
end

# Standard matching overloads for raw inputs
function PersistenceDiagrams.matching(val::Union{Bottleneck, Wasserstein}, w1::WalkResult, w2::WalkResult; dim::Integer=0, kwargs...)
    diag1 = PersistenceDiagrams.PersistenceDiagram(w1; dim=dim, kwargs...)
    diag2 = PersistenceDiagrams.PersistenceDiagram(w2; dim=dim, kwargs...)
    PersistenceDiagrams.matching(val, diag1, diag2)
end

function PersistenceDiagrams.matching(val::Union{Bottleneck, Wasserstein}, c1::AbstractVector{<:AbstractString}, c2::AbstractVector{<:AbstractString}; dim::Integer=0, kwargs...)
    diag1 = PersistenceDiagrams.PersistenceDiagram(c1; dim=dim, kwargs...)
    diag2 = PersistenceDiagrams.PersistenceDiagram(c2; dim=dim, kwargs...)
    PersistenceDiagrams.matching(val, diag1, diag2)
end

function PersistenceDiagrams.matching(val::Union{Bottleneck, Wasserstein}, n1::Integer, n2::Integer; seed1::Integer=Gay.GAY_SEED, gamma1::Integer=Gay.GOLDEN_GAMMA, seed2::Integer=Gay.GAY_SEED, gamma2::Integer=Gay.GOLDEN_GAMMA, dim::Integer=0, kwargs...)
    diag1 = PersistenceDiagrams.PersistenceDiagram(n1; seed=seed1, gamma=gamma1, dim=dim, kwargs...)
    diag2 = PersistenceDiagrams.PersistenceDiagram(n2; seed=seed2, gamma=gamma2, dim=dim, kwargs...)
    PersistenceDiagrams.matching(val, diag1, diag2)
end

# ------------------------------------------------------------------------------
# 4. Implement GayBottleneck and GayWasserstein functors
# ------------------------------------------------------------------------------

# Helper standard_functor
standard_functor(::GayBottleneck) = Bottleneck()
standard_functor(::GayWasserstein) = Wasserstein()
standard_functor(other) = other

# GayBottleneck
function (::GayBottleneck)(g1::GayPersistenceDiagram, g2::GayPersistenceDiagram; kwargs...)
    Bottleneck()(g1.diagram, g2.diagram; kwargs...)
end

function (::GayBottleneck)(diag1::PersistenceDiagrams.PersistenceDiagram, diag2::PersistenceDiagrams.PersistenceDiagram; kwargs...)
    Bottleneck()(diag1, diag2; kwargs...)
end

function (::GayBottleneck)(w1::WalkResult, w2::WalkResult; dim::Integer=0, kwargs...)
    Bottleneck()(w1, w2; dim=dim, kwargs...)
end

function (::GayBottleneck)(c1::AbstractVector{<:AbstractString}, c2::AbstractVector{<:AbstractString}; dim::Integer=0, kwargs...)
    Bottleneck()(c1, c2; dim=dim, kwargs...)
end

# (::GayBottleneck)(n1::Integer, n2::Integer; dim::Integer=0, kwargs...)
function (::GayBottleneck)(n1::Integer, n2::Integer; dim::Integer=0, kwargs...)
    Bottleneck()(n1, n2; dim=dim, kwargs...)
end

# GayWasserstein
function (::GayWasserstein)(g1::GayPersistenceDiagram, g2::GayPersistenceDiagram; kwargs...)
    Wasserstein()(g1.diagram, g2.diagram; kwargs...)
end

function (::GayWasserstein)(diag1::PersistenceDiagrams.PersistenceDiagram, diag2::PersistenceDiagrams.PersistenceDiagram; kwargs...)
    Wasserstein()(diag1, diag2; kwargs...)
end

function (::GayWasserstein)(w1::WalkResult, w2::WalkResult; dim::Integer=0, kwargs...)
    Wasserstein()(w1, w2; dim=dim, kwargs...)
end

function (::GayWasserstein)(c1::AbstractVector{<:AbstractString}, c2::AbstractVector{<:AbstractString}; dim::Integer=0, kwargs...)
    Wasserstein()(c1, c2; dim=dim, kwargs...)
end

function (::GayWasserstein)(n1::Integer, n2::Integer; dim::Integer=0, kwargs...)
    Wasserstein()(n1, n2; dim=dim, kwargs...)
end

# ------------------------------------------------------------------------------
# 5. Implement Gay.gay_matching function
# ------------------------------------------------------------------------------

function Gay.gay_matching(val, g1::GayPersistenceDiagram, g2::GayPersistenceDiagram; kwargs...)
    PersistenceDiagrams.matching(standard_functor(val), g1.diagram, g2.diagram; kwargs...)
end

function Gay.gay_matching(val, diag1::PersistenceDiagrams.PersistenceDiagram, diag2::PersistenceDiagrams.PersistenceDiagram; kwargs...)
    PersistenceDiagrams.matching(standard_functor(val), diag1, diag2; kwargs...)
end

function Gay.gay_matching(val, w1::WalkResult, w2::WalkResult; dim::Integer=0, kwargs...)
    PersistenceDiagrams.matching(standard_functor(val), w1, w2; dim=dim, kwargs...)
end

function Gay.gay_matching(val, c1::AbstractVector{<:AbstractString}, c2::AbstractVector{<:AbstractString}; dim::Integer=0, kwargs...)
    PersistenceDiagrams.matching(standard_functor(val), c1, c2; dim=dim, kwargs...)
end

function Gay.gay_matching(val, n1::Integer, n2::Integer; dim::Integer=0, kwargs...)
    PersistenceDiagrams.matching(standard_functor(val), n1, n2; dim=dim, kwargs...)
end

# ------------------------------------------------------------------------------
# 6. Expose wrappers on the Gay namespace
# ------------------------------------------------------------------------------

# Bottleneck
function Gay.gay_bottleneck(g1::GayPersistenceDiagram, g2::GayPersistenceDiagram)
    Bottleneck()(g1, g2)
end

function Gay.gay_bottleneck(diag1::PersistenceDiagrams.PersistenceDiagram, diag2::PersistenceDiagrams.PersistenceDiagram)
    Bottleneck()(diag1, diag2)
end

function Gay.gay_bottleneck(w1::WalkResult, w2::WalkResult; dim::Integer=0, kwargs...)
    Bottleneck()(w1, w2; dim=dim, kwargs...)
end

function Gay.gay_bottleneck(c1::AbstractVector{<:AbstractString}, c2::AbstractVector{<:AbstractString}; dim::Integer=0, kwargs...)
    Bottleneck()(c1, c2; dim=dim, kwargs...)
end

function Gay.gay_bottleneck(n1::Integer, n2::Integer; dim::Integer=0, kwargs...)
    Bottleneck()(n1, n2; dim=dim, kwargs...)
end

# Wasserstein
function Gay.gay_wasserstein(g1::GayPersistenceDiagram, g2::GayPersistenceDiagram)
    Wasserstein()(g1, g2)
end

function Gay.gay_wasserstein(diag1::PersistenceDiagrams.PersistenceDiagram, diag2::PersistenceDiagrams.PersistenceDiagram)
    Wasserstein()(diag1, diag2)
end

function Gay.gay_wasserstein(w1::WalkResult, w2::WalkResult; dim::Integer=0, kwargs...)
    Wasserstein()(w1, w2; dim=dim, kwargs...)
end

function Gay.gay_wasserstein(c1::AbstractVector{<:AbstractString}, c2::AbstractVector{<:AbstractString}; dim::Integer=0, kwargs...)
    Wasserstein()(c1, c2; dim=dim, kwargs...)
end

function Gay.gay_wasserstein(n1::Integer, n2::Integer; dim::Integer=0, kwargs...)
    Wasserstein()(n1, n2; dim=dim, kwargs...)
end

# gay_persistencediagram now returns the first-class GayPersistenceDiagram!
function Gay.gay_persistencediagram(colors::AbstractVector{<:AbstractString}; dim::Integer=0, kwargs...)
    Gay.GayPersistenceDiagram(colors; dim=dim, kwargs...)
end

function Gay.gay_persistencediagram(walk::WalkResult; dim::Integer=0, kwargs...)
    Gay.GayPersistenceDiagram(walk; dim=dim, kwargs...)
end

function Gay.gay_persistencediagram(n::Integer; seed::Integer=Gay.GAY_SEED, gamma::Integer=Gay.GOLDEN_GAMMA, dim::Integer=0, kwargs...)
    Gay.GayPersistenceDiagram(n; seed=seed, gamma=gamma, dim=dim, kwargs...)
end

end # module GayPersistenceDiagramsExt
