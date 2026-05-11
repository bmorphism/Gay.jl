#!/usr/bin/env julia

# Topos UMAP x Gay.jl REPL seed.
#
# Source: https://topos.institute/blog/2024-04-05-understanding-umap/
#
# The script keeps the interesting parts live at the REPL:
#   copy!   : generate the non-Hausdorff toy space from the post
#   stretch!: build the local fuzzy kNN graph with sum(weights) = log2(k)
#   press!  : lay the graph down in R^2 by minimizing graph cross entropy
#
# It intentionally avoids adding UMAP packages. The point is to expose the
# graph-construction and graph-layout boundary that the post explains, then
# color every carrier deterministically with Gay.jl.

module ToposUMAPGayRepl

using Dates
using Gay
using JSON3
using LinearAlgebra
using Printf
using Random
using Statistics
using Colors: RGB

export UMAPPoint, FuzzyGraph, UMAPReplState
export generate_nonhausdorff, fuzzy_graph, spectral_layout, optimize_layout
export new_state, stretch!, press!, write_state, run_grid, run_self_test
export topos_umap_repl

const SOURCE_URL = "https://topos.institute/blog/2024-04-05-understanding-umap/"
const DEFAULT_SEED = UInt64(parse(Int, get(ENV, "GAY_SEED", "69")))

struct UMAPPoint
    x::Float64
    y::Float64
    trit::Int8
    color::RGB{Float64}
end

struct FuzzyGraph
    weights::Matrix{Float64}
    directed::Matrix{Float64}
    sigmas::Vector{Float64}
    rhos::Vector{Float64}
    row_sums::Vector{Float64}
    neighbors::Int
    target::Float64
end

mutable struct UMAPReplState
    data::Vector{UMAPPoint}
    graph::Union{Nothing,FuzzyGraph}
    layout::Union{Nothing,Matrix{Float64}}
    alpha::Float64
    neighbors::Int
    min_dist::Float64
    seed::UInt64
    losses::Vector{Float64}
    fingerprint::UInt64
end

UMAPReplState(data, alpha, seed) =
    UMAPReplState(data, nothing, nothing, alpha, 0, 0.0, seed, Float64[], UInt64(0))

clamp01(x) = clamp(x, 0.0, 1.0)

function rgb255(c::RGB)
    (
        round(Int, clamp01(c.r) * 255),
        round(Int, clamp01(c.g) * 255),
        round(Int, clamp01(c.b) * 255),
    )
end

function hex_color(c::RGB)
    r, g, b = rgb255(c)
    @sprintf("#%02X%02X%02X", r, g, b)
end

function mix(a::RGB, b::RGB, t::Real)
    u = clamp01(float(t))
    RGB{Float64}(
        (1 - u) * a.r + u * b.r,
        (1 - u) * a.g + u * b.g,
        (1 - u) * a.b + u * b.b,
    )
end

function branch_color(trit::Integer, x::Real, index::Integer; seed::Integer=DEFAULT_SEED)
    branch_index = trit == -1 ? 1 : trit == 0 ? 2 : 3
    anchor = Gay.color_at(10_000 + branch_index; seed=seed)
    grain = Gay.color_at(20_000 + index; seed=seed)
    light = 0.25 + 0.55 * ((float(x) + 1.0) / 2.0)
    lightness = RGB{Float64}(light, light, light)
    mix(mix(anchor, grain, 0.18), lightness, 0.22)
end

function sample_x(rng::AbstractRNG, alpha::Real)
    if rand(rng) < alpha
        mu = rand(rng) < 0.5 ? -0.5 : 0.5
        return clamp(mu + 0.05 * randn(rng), -1.0, 1.0)
    end
    2.0 * rand(rng) - 1.0
end

"""
    generate_nonhausdorff(n; alpha=0.2, seed=DEFAULT_SEED, split=0.5)

Generate the post's "two lines glued outside the middle" toy space:

  * outside `abs(x) > split`: one glued line with trit 0
  * inside  `abs(x) <= split`: two distinguishable branches with trits -1/+1

`alpha` mixes uniform sampling with bimodal sampling near the two gluing
thresholds, matching the post's bonus experiment.
"""
function generate_nonhausdorff(
    n::Integer=100;
    alpha::Real=0.2,
    seed::Integer=DEFAULT_SEED,
    split::Real=0.5,
)
    n > 0 || throw(ArgumentError("n must be positive"))
    0.0 <= alpha <= 1.0 || throw(ArgumentError("alpha must be in [0, 1]"))

    rng = MersenneTwister(UInt64(seed) % UInt64(typemax(Int)))
    points = UMAPPoint[]
    sizehint!(points, n)

    for i in 1:n
        x = sample_x(rng, alpha)
        trit = if abs(x) <= split
            rand(rng) < 0.5 ? Int8(-1) : Int8(1)
        else
            Int8(0)
        end
        y = Float64(trit)
        push!(points, UMAPPoint(x, y, trit, branch_color(trit, x, i; seed=seed)))
    end

    points
end

function feature_matrix(points::Vector{UMAPPoint})
    x = Matrix{Float64}(undef, length(points), 2)
    for (i, p) in enumerate(points)
        x[i, 1] = p.x
        x[i, 2] = p.y
    end
    x
end

function pairwise_distances(x::AbstractMatrix{<:Real})
    n = size(x, 1)
    d = zeros(Float64, n, n)
    for i in 1:n, j in (i + 1):n
        dx = x[i, 1] - x[j, 1]
        dy = x[i, 2] - x[j, 2]
        dij = hypot(dx, dy)
        d[i, j] = dij
        d[j, i] = dij
    end
    d
end

function local_weight_sum(ds::AbstractVector{<:Real}, rho::Real, sigma::Real)
    s = 0.0
    inv_sigma = 1.0 / max(float(sigma), eps(Float64))
    for d in ds
        s += exp(-max(float(d) - float(rho), 0.0) * inv_sigma)
    end
    s
end

function solve_sigma(ds::AbstractVector{<:Real}, rho::Real, target::Real)
    target <= 1.0 && return 1.0

    lo = eps(Float64)
    hi = 1.0
    while local_weight_sum(ds, rho, hi) < target && hi < 1.0e6
        hi *= 2.0
    end

    for _ in 1:48
        mid = (lo + hi) / 2.0
        if local_weight_sum(ds, rho, mid) < target
            lo = mid
        else
            hi = mid
        end
    end

    hi
end

"""
    fuzzy_graph(points; k=15)

Copy/stretch/press, graph half:

  * copy: focus each point in turn
  * stretch: choose sigma so each local row has total mass log2(k)
  * press: combine directed probabilities by p + q - pq
"""
function fuzzy_graph(points::Vector{UMAPPoint}; k::Integer=15)
    n = length(points)
    2 <= k < n || throw(ArgumentError("k must satisfy 2 <= k < n"))

    x = feature_matrix(points)
    d = pairwise_distances(x)
    directed = zeros(Float64, n, n)
    sigmas = zeros(Float64, n)
    rhos = zeros(Float64, n)
    row_sums = zeros(Float64, n)
    target = log2(float(k))

    for i in 1:n
        order = sortperm(@view d[i, :])
        neigh = filter(!=(i), order)[1:k]
        ds = d[i, neigh]
        rho = minimum(ds)
        sigma = solve_sigma(ds, rho, target)
        sigmas[i] = sigma
        rhos[i] = rho

        for j in neigh
            directed[i, j] = exp(-max(d[i, j] - rho, 0.0) / sigma)
        end
        row_sums[i] = sum(@view directed[i, :])
    end

    weights = zeros(Float64, n, n)
    for i in 1:n, j in (i + 1):n
        p = directed[i, j]
        q = directed[j, i]
        w = p + q - p * q
        weights[i, j] = w
        weights[j, i] = w
    end

    FuzzyGraph(weights, directed, sigmas, rhos, row_sums, Int(k), target)
end

function center_scale!(y::AbstractMatrix{<:Real})
    for j in 1:size(y, 2)
        y[:, j] .-= mean(@view y[:, j])
    end
    scale = maximum(abs, y)
    scale > 0 && (y ./= scale)
    y
end

function spectral_layout(graph::FuzzyGraph)
    w = graph.weights
    n = size(w, 1)
    degree = vec(sum(w; dims=2))
    if count(>(0), degree) < 3
        theta = range(0, 2pi; length=n + 1)[1:n]
        y = hcat(cos.(theta), sin.(theta))
        return center_scale!(Matrix{Float64}(y))
    end

    lap = Diagonal(degree) - w
    eig = eigen(Symmetric(lap))
    order = sortperm(eig.values)
    cols = order[2:min(3, length(order))]
    if length(cols) == 1
        y = hcat(eig.vectors[:, cols[1]], zeros(n))
    else
        y = eig.vectors[:, cols]
    end
    center_scale!(Matrix{Float64}(y))
end

function graph_cross_entropy(weights::AbstractMatrix, y::AbstractMatrix; min_dist::Real=0.1)
    n = size(weights, 1)
    a = max(float(min_dist), 0.03)^2
    loss = 0.0
    for i in 1:n, j in (i + 1):n
        dx = y[i, 1] - y[j, 1]
        dy = y[i, 2] - y[j, 2]
        q = 1.0 / (1.0 + (dx * dx + dy * dy) / a)
        q = clamp(q, 1.0e-9, 1.0 - 1.0e-9)
        p = clamp(weights[i, j], 0.0, 1.0)
        loss -= p * log(q) + (1.0 - p) * log1p(-q)
    end
    loss
end

function optimize_layout(
    graph::FuzzyGraph,
    y0::AbstractMatrix;
    min_dist::Real=0.1,
    epochs::Integer=80,
    rate::Real=0.18,
)
    y = Matrix{Float64}(y0)
    n = size(y, 1)
    a = max(float(min_dist), 0.03)^2
    losses = Float64[]
    sizehint!(losses, epochs + 1)
    push!(losses, graph_cross_entropy(graph.weights, y; min_dist=min_dist))

    for epoch in 1:epochs
        grad = zeros(Float64, n, 2)
        for i in 1:n, j in (i + 1):n
            dx = y[i, 1] - y[j, 1]
            dy = y[i, 2] - y[j, 2]
            d2 = max(dx * dx + dy * dy, 1.0e-12)
            q = 1.0 / (1.0 + d2 / a)
            q = clamp(q, 1.0e-6, 1.0 - 1.0e-6)
            p = clamp(graph.weights[i, j], 0.0, 1.0)

            dloss_dq = -p / q + (1.0 - p) / (1.0 - q)
            dloss_dd2 = dloss_dq * (-(q * q) / a)
            gx = 2.0 * dloss_dd2 * dx
            gy = 2.0 * dloss_dd2 * dy

            clip = max(abs(gx), abs(gy))
            if clip > 8.0
                gx *= 8.0 / clip
                gy *= 8.0 / clip
            end

            grad[i, 1] += gx
            grad[i, 2] += gy
            grad[j, 1] -= gx
            grad[j, 2] -= gy
        end

        eta = float(rate) * (1.0 - 0.85 * (epoch - 1) / max(1, epochs - 1))
        y .-= (eta / sqrt(n)) .* grad
        center_scale!(y)
        epoch == epochs || epoch % 10 == 0 || continue
        push!(losses, graph_cross_entropy(graph.weights, y; min_dist=min_dist))
    end

    y, losses
end

function quantized_u64(x::Real)
    q = Int64(round(clamp(float(x), -10.0, 10.0) * 1_000_000))
    reinterpret(UInt64, q)
end

function fingerprint_layout(y::AbstractMatrix, graph::FuzzyGraph; seed::Integer=DEFAULT_SEED)
    h = UInt64(seed) ⊻ UInt64(size(y, 1)) ⊻ (UInt64(graph.neighbors) << 32)
    for v in y
        h = Gay.splitmix64(h ⊻ quantized_u64(v))
    end
    for v in graph.row_sums
        h = Gay.splitmix64(h ⊻ quantized_u64(v))
    end
    h
end

function new_state(; n::Integer=100, alpha::Real=0.2, seed::Integer=DEFAULT_SEED)
    UMAPReplState(generate_nonhausdorff(n; alpha=alpha, seed=seed), float(alpha), UInt64(seed))
end

function stretch!(state::UMAPReplState; k::Integer=15)
    state.graph = fuzzy_graph(state.data; k=k)
    state.neighbors = Int(k)
    state.fingerprint = UInt64(0)
    state
end

function press!(state::UMAPReplState; min_dist::Real=0.1, epochs::Integer=80, rate::Real=0.18)
    state.graph === nothing && stretch!(state)
    y0 = spectral_layout(state.graph)
    y, losses = optimize_layout(state.graph, y0; min_dist=min_dist, epochs=epochs, rate=rate)
    state.layout = y
    state.losses = losses
    state.min_dist = float(min_dist)
    state.fingerprint = fingerprint_layout(y, state.graph; seed=state.seed)
    state
end

function draw_disk!(canvas, cx::Int, cy::Int, radius::Int, color::RGB)
    height, width, _ = size(canvas)
    r, g, b = rgb255(color)
    for y in max(1, cy - radius):min(height, cy + radius)
        for x in max(1, cx - radius):min(width, cx + radius)
            (x - cx)^2 + (y - cy)^2 <= radius^2 || continue
            canvas[y, x, 1] = UInt8(r)
            canvas[y, x, 2] = UInt8(g)
            canvas[y, x, 3] = UInt8(b)
        end
    end
end

function put_rect!(canvas, x0::Int, y0::Int, x1::Int, y1::Int, color::RGB)
    height, width, _ = size(canvas)
    r, g, b = rgb255(color)
    for y in max(1, y0):min(height, y1), x in max(1, x0):min(width, x1)
        canvas[y, x, 1] = UInt8(r)
        canvas[y, x, 2] = UInt8(g)
        canvas[y, x, 3] = UInt8(b)
    end
end

function render_grid_ppm(states::Vector{UMAPReplState}, rows::Int, cols::Int, path::AbstractString; cell::Int=220)
    width = cols * cell
    height = rows * cell
    canvas = fill(UInt8(248), height, width, 3)

    for (idx, state) in enumerate(states)
        y = state.layout === nothing ? error("state $(idx) has no layout") : state.layout
        row = (idx - 1) ÷ cols
        col = (idx - 1) % cols
        x0 = col * cell + 1
        y0 = row * cell + 1
        x1 = (col + 1) * cell
        y1 = (row + 1) * cell

        frame = Gay.color_at(30_000 + 100 * state.neighbors + round(Int, 100 * state.min_dist); seed=state.seed)
        put_rect!(canvas, x0, y0, x1, y0 + 5, frame)
        put_rect!(canvas, x0, y1 - 5, x1, y1, frame)
        put_rect!(canvas, x0, y0, x0 + 5, y1, frame)
        put_rect!(canvas, x1 - 5, y0, x1, y1, frame)

        xs = y[:, 1]
        ys = y[:, 2]
        xmin, xmax = extrema(xs)
        ymin, ymax = extrema(ys)
        xr = max(xmax - xmin, 1.0e-9)
        yr = max(ymax - ymin, 1.0e-9)
        margin = 22

        for (pidx, p) in enumerate(state.data)
            px = x0 + margin + round(Int, (cell - 2margin) * (xs[pidx] - xmin) / xr)
            py = y0 + margin + round(Int, (cell - 2margin) * (1.0 - (ys[pidx] - ymin) / yr))
            draw_disk!(canvas, px, py, 3, p.color)
        end
    end

    open(path, "w") do io
        println(io, "P3")
        println(io, "$width $height")
        println(io, "255")
        for y in 1:height
            for x in 1:width
                print(io, Int(canvas[y, x, 1]), ' ', Int(canvas[y, x, 2]), ' ', Int(canvas[y, x, 3]), ' ')
            end
            println(io)
        end
    end

    path
end

function state_summary(state::UMAPReplState)
    graph = state.graph === nothing ? nothing : state.graph
    trits = [p.trit for p in state.data]
    (
        alpha=state.alpha,
        n=length(state.data),
        neighbors=state.neighbors,
        min_dist=state.min_dist,
        seed=state.seed,
        trit_sum_mod3=mod(sum(Int.(trits)), 3),
        trit_counts=(
            minus=count(==(-1), trits),
            ergodic=count(==(0), trits),
            plus=count(==(1), trits),
        ),
        graph=graph === nothing ? nothing : (
            target=graph.target,
            row_sum_min=minimum(graph.row_sums),
            row_sum_mean=mean(graph.row_sums),
            row_sum_max=maximum(graph.row_sums),
            sym_error=maximum(abs, graph.weights - graph.weights'),
            edge_mass=sum(graph.weights) / 2,
        ),
        loss_first=isempty(state.losses) ? nothing : first(state.losses),
        loss_last=isempty(state.losses) ? nothing : last(state.losses),
        fingerprint=@sprintf("0x%016x", state.fingerprint),
    )
end

function write_manifest(path::AbstractString, payload)
    open(path, "w") do io
        JSON3.write(io, payload)
        println(io)
    end
    path
end

function write_state(state::UMAPReplState; outdir::AbstractString="/tmp/gay-topos-umap", name::AbstractString="state")
    state.layout === nothing && press!(state)
    mkpath(outdir)
    ppm = joinpath(outdir, "$name.ppm")
    json = joinpath(outdir, "$name.json")
    render_grid_ppm([state], 1, 1, ppm)
    write_manifest(json, (
        source_url=SOURCE_URL,
        carrier="Gay.jl deterministic PPM plus JSON",
        dispatch_boundary="Topos prose -> Julia REPL state -> Gay-colored layout witness",
        summary=state_summary(state),
    ))
    (ppm=ppm, json=json, summary=state_summary(state))
end

function run_grid(;
    alpha::Real=0.2,
    n::Integer=100,
    neighbors=(5, 10, 20, 40),
    min_dists=(0.05, 0.1, 0.2, 0.4),
    seed::Integer=DEFAULT_SEED,
    epochs::Integer=60,
    outdir::AbstractString="/tmp/gay-topos-umap",
    prefix::AbstractString="topos-umap",
)
    mkpath(outdir)
    states = UMAPReplState[]
    data_seed = UInt64(seed) ⊻ Gay.splitmix64(UInt64(round(Int, 1000 * alpha)))
    data = generate_nonhausdorff(n; alpha=alpha, seed=data_seed)

    for md in min_dists, k in neighbors
        st = UMAPReplState(copy(data), float(alpha), UInt64(seed))
        stretch!(st; k=k)
        press!(st; min_dist=md, epochs=epochs)
        push!(states, st)
    end

    tag = replace(@sprintf("alpha_%0.2f", alpha), "." => "p")
    ppm = joinpath(outdir, "$(prefix)_$(tag).ppm")
    json = joinpath(outdir, "$(prefix)_$(tag).json")
    render_grid_ppm(states, length(min_dists), length(neighbors), ppm)
    manifest = (
        source_url=SOURCE_URL,
        created_at=Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
        alpha=alpha,
        n=n,
        seed=UInt64(seed),
        data_seed=data_seed,
        neighbors=collect(neighbors),
        min_dists=collect(min_dists),
        rows="min_dist",
        cols="neighbors",
        dispatch_boundary="Topos copy/stretch/press -> Julia graph/layout -> Gay color carrier",
        source_signals=(
            graph_construction="local fuzzy kNN rows sum to log2(k)",
            press_rule="directed edge probabilities combine as p + q - pq",
            layout_warning="2D layout is a low-dimensional approximation of a 1-truncated graph",
            nonhausdorff_bonus="two branches in the middle, glued outside the split interval",
        ),
        states=[state_summary(st) for st in states],
    )
    write_manifest(json, manifest)
    (ppm=ppm, json=json, states=states)
end

function run_self_test(; outdir::AbstractString=mktempdir())
    a = new_state(n=36, alpha=0.2, seed=DEFAULT_SEED)
    stretch!(a; k=5)
    press!(a; min_dist=0.1, epochs=6)
    out = write_state(a; outdir=outdir, name="self-test")

    b = new_state(n=36, alpha=0.2, seed=DEFAULT_SEED)
    stretch!(b; k=5)
    press!(b; min_dist=0.1, epochs=6)

    @assert size(a.graph.weights) == (36, 36)
    @assert maximum(abs, a.graph.weights - a.graph.weights') < 1.0e-12
    @assert maximum(abs.(a.graph.row_sums .- a.graph.target)) < 1.0e-6
    @assert all(isfinite, a.layout)
    @assert all(isfinite, a.losses)
    @assert a.fingerprint == b.fingerprint
    @assert isfile(out.ppm)
    @assert isfile(out.json)

    println("SELF_TEST_OK")
    println("ppm=$(out.ppm)")
    println("json=$(out.json)")
    println("fingerprint=$(@sprintf("0x%016x", a.fingerprint))")
    out
end

function topos_umap_repl()
    println("""
    Topos UMAP x Gay.jl REPL

    include("examples/topos_umap_gay_repl.jl")
    using .ToposUMAPGayRepl

    st = new_state(n=100, alpha=0.9, seed=69)  # copy the toy space
    stretch!(st; k=10)                         # local fuzzy graph
    press!(st; min_dist=0.05, epochs=60)       # 2D graph layout
    write_state(st; outdir="/tmp/gay-topos-umap", name="alpha09-k10-md005")

    run_grid(alpha=0.2, outdir="/tmp/gay-topos-umap")
    """)
    nothing
end

function parse_arg(args, name, default)
    idx = findfirst(==(name), args)
    idx === nothing && return default
    idx == length(args) && throw(ArgumentError("missing value for $name"))
    args[idx + 1]
end

function print_help()
    println("""
    Usage:
      julia --project=. examples/topos_umap_gay_repl.jl [--quick|--full|--self-test]

    Options:
      --quick          one alpha, 2x2 parameter grid
      --full           Topos-style alpha set, 4x4 parameter grids
      --self-test      deterministic graph/layout/render check
      --out DIR        output directory (default: /tmp/gay-topos-umap)
      --seed N         Gay seed (default: ENV["GAY_SEED"] or 69)
    """)
end

function main(args=ARGS)
    if "--help" in args || "-h" in args
        print_help()
        return nothing
    end

    outdir = parse_arg(args, "--out", "/tmp/gay-topos-umap")
    seed = UInt64(parse(Int, parse_arg(args, "--seed", string(DEFAULT_SEED))))

    if "--self-test" in args
        return run_self_test(; outdir=outdir)
    end

    full = "--full" in args
    alphas = full ? (0.0, 0.2, 0.9) : (0.2,)
    neighbors = full ? (5, 10, 20, 40) : (5, 20)
    min_dists = full ? (0.05, 0.1, 0.2, 0.4) : (0.05, 0.4)
    epochs = full ? 60 : 24
    n = full ? 100 : 72

    println("Topos UMAP x Gay.jl")
    println("source=$SOURCE_URL")
    println("seed=$seed")
    println("outdir=$outdir")

    outputs = []
    for alpha in alphas
        out = run_grid(;
            alpha=alpha,
            n=n,
            neighbors=neighbors,
            min_dists=min_dists,
            seed=seed,
            epochs=epochs,
            outdir=outdir,
        )
        push!(outputs, out)
        println("alpha=$alpha ppm=$(out.ppm)")
        println("alpha=$alpha json=$(out.json)")
    end

    println("REPL entry: ToposUMAPGayRepl.topos_umap_repl()")
    outputs
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    ToposUMAPGayRepl.main(ARGS)
end
