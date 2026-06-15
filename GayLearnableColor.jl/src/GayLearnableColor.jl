module GayLearnableColor

# Jointly-learnable color space, "as in Gay.jl colorings": learn a color embedding
# φ_θ : behaviour → Okhsl color, co-optimised JOINTLY over all behaviours coupled by
# their structural distance (analytic-gradient MDS; exact, so no AD lib needed —
# Enzyme would be the bob-faithful backend but adds nothing numerically). Output
# colors come from Gay.jl's own okhsl_to_rgb. The learned coords ARE the behaviour
# embedding ⇒ behaviour and color are one.

import Gay

export learn_colorspace, structure_corr, LearnedColorSpace, behaviors_gf3, graph_distance

struct LearnedColorSpace
    X::Matrix{Float64}     # n×d learned coords (Okhsl-ish)
    hexes::Vector{String}  # n colors via Gay's Okhsl
    corr::Float64          # structure preservation (Pearson)
end

_seed(args...) = Gay.stable_seed(join(args, "-"))

# Saturating (non-Riemannian) readout for embedding distances: perceived
# difference shows diminishing returns at large separations (Bujack 2022).
# f(0)=0, f'(0)=1 (local regime = raw distance), strictly subadditive with
# exact defect f(x)+f(y)−f(x+y) = f(x)f(y)/A (SatReadout.lean). SAT_A > 100
# so all targets k·D ∈ [0,100] stay representable below the asymptote.
const SAT_A = 150.0
_sat(d) = SAT_A * (1.0 - exp(-d / SAT_A))
_sat′(d) = exp(-d / SAT_A)

function _mds(D::AbstractMatrix, d::Int, iters::Int, lr::Float64, seed::Int)
    n = size(D, 1)
    k = 100.0 / max(1.0, Float64(maximum(D)))
    X = [Float64(Int(mod(_seed(seed, i, a), 100))) - 50.0 for i in 1:n, a in 1:d]
    for _ in 1:iters
        G = zeros(n, d)
        @inbounds for i in 1:n, j in 1:n
            i == j && continue
            s2 = 0.0
            for a in 1:d; δ = X[i,a] - X[j,a]; s2 += δ*δ; end
            dij = max(1e-9, sqrt(s2))
            # Subadditive MDS: stress Σ (f(dij) − k·D_ij)², still an EXACT
            # analytic gradient (chain rule through f; no AD lib needed):
            c = 2.0 * (_sat(dij) - k*Float64(D[i,j])) * _sat′(dij) / dij
            for a in 1:d; G[i,a] += c * (X[i,a] - X[j,a]); end
        end
        @. X -= lr * G
    end
    X
end

_squash(x) = 1.0 / (1.0 + exp(-x/30.0))
function _colorize(X)
    [let
         L = 0.35 + 0.45*_squash(X[i,1])
         S = 0.45 + 0.45*_squash(X[i, min(2, size(X,2))])
         H = mod(X[i, size(X,2)] * 3.6, 360.0)
         Gay.rgb_hex(Gay.okhsl_to_rgb(H, S, L)...)
     end for i in 1:size(X,1)]
end

function structure_corr(X, D)
    n = size(X, 1); xs = Float64[]; ys = Float64[]
    for i in 1:n, j in (i+1):n
        push!(xs, Float64(D[i,j]))
        s2 = 0.0; for a in 1:size(X,2); δ = X[i,a]-X[j,a]; s2 += δ*δ; end
        push!(ys, _sat(sqrt(s2)))  # correlate through the same readout the stress fits
    end
    mx = sum(xs)/length(xs); my = sum(ys)/length(ys)
    cov = sum((xs .- mx) .* (ys .- my)); vx = sqrt(sum((xs .- mx).^2)); vy = sqrt(sum((ys .- my).^2))
    (vx*vy == 0.0) ? 0.0 : cov/(vx*vy)
end

function learn_colorspace(D; d::Int=3, iters::Int=500, lr::Float64=0.02, seed::Int=7)
    X = _mds(D, d, iters, lr, seed)
    LearnedColorSpace(X, _colorize(X), structure_corr(X, D))
end

# --- behaviour structures -------------------------------------------------------
function behaviors_gf3(n::Int=24)
    B = [[Int(mod(_seed(7919*i, k), 3)) for k in 1:9] for i in 1:n]
    [count(!=(0), B[i] .- B[j]) for i in 1:n, j in 1:n]
end
function graph_distance(adj::Dict{Int,Vector{Int}}, n::Int)
    D = fill(999, n, n)
    for s in 0:n-1
        d = fill(999, n); d[s+1] = 0; q = [s]
        while !isempty(q)
            u = popfirst!(q)
            for v in get(adj, u, Int[])
                if d[v+1] == 999; d[v+1] = d[u+1] + 1; push!(q, v); end
            end
        end
        for t in 0:n-1; D[s+1, t+1] = d[t+1]; end
    end
    D
end

# --- Scale protocol (docs/non_riemannian_color_scales.md) -----------------------
# Class #1 "Diminishing-Return Color" made EXECUTABLE: the doc's compare(scale,a,b)
# with the local/global regime split. Awareness was prose; this is the contract.

abstract type AbstractColorScale end
abstract type AbstractWhiteAnchor end
struct NeutralAxisWhite <: AbstractWhiteAnchor end

"""
    DiminishingReturnScale(local_compare; A=SAT_A)

Class #1: large perceived differences are not sums of JND steps.
`local_compare(a,b)` is any small-difference kernel (additive ΔE — valid
locally and ONLY locally); `A` is the saturation asymptote, fit from
discrimination data. `compare` = `global_diff` = f∘local with
f(d) = A(1−exp(−d/A)): f(0)=0, f′(0)=1, strictly subadditive with EXACT
defect f(x)+f(y)−f(x+y) = f(x)·f(y)/A (SatReadout.lean).
"""
struct DiminishingReturnScale{F} <: AbstractColorScale
    local_compare::F
    A::Float64
end
DiminishingReturnScale(local_compare; A::Float64=SAT_A) =
    DiminishingReturnScale{typeof(local_compare)}(local_compare, A)

local_diff(s::DiminishingReturnScale, a, b) = s.local_compare(a, b)
global_diff(s::DiminishingReturnScale, a, b) =
    (d = local_diff(s, a, b); s.A * (1.0 - exp(-d / s.A)))
compare(s::DiminishingReturnScale, a, b) = global_diff(s, a, b)
white_anchor(::DiminishingReturnScale) = NeutralAxisWhite()

"""
    diagnose(s::DiminishingReturnScale, a, b, c; tol=1e-9)

The −1 gate (can FAIL — that is its job). For a triplet collinear under the
LOCAL kernel, `compare` must be STRICTLY subadditive; the gap must match the
derived defect `f(x)·f(y)/A` (tolerance from the theorem, not tuned).
Returns `(gate=Bool, gap, expected_gap, collinear=Bool)`. A comparator whose
gap is ≈ 0 on collinear triplets is Riemannian/additive and is barred from
large-difference use.
"""
function diagnose(s::DiminishingReturnScale, a, b, c; tol::Float64=1e-9)
    x  = local_diff(s, a, b)
    y  = local_diff(s, b, c)
    z  = local_diff(s, a, c)
    collinear = abs(x + y - z) ≤ 1e-6 * max(1.0, z)
    gap = compare(s, a, b) + compare(s, b, c) - compare(s, a, c)
    expected = global_diff_raw(s, x) * global_diff_raw(s, y) / s.A
    gate = collinear ? (gap > tol && abs(gap - expected) ≤ 1e-6 * max(1.0, expected)) :
                       (gap > tol)
    (gate=gate, gap=gap, expected_gap=expected, collinear=collinear)
end
global_diff_raw(s::DiminishingReturnScale, d::Float64) =
    s.A * (1.0 - exp(-d / s.A))

include("learnable_heat_color.jl")

export AbstractColorScale, AbstractWhiteAnchor, NeutralAxisWhite,
       DiminishingReturnScale, local_diff, global_diff, compare, white_anchor,
       diagnose

export LearnableColormap, learn_heat_colormap, interpolate_colormap, get_color

end # module GayLearnableColor
