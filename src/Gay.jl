module Gay

using Printf
using Random
using SHA
using Unicode

export SplittableRandom, WalkStep, WalkResult
export color_at, trit, gay_seed, stable_seed, hierarchical_colors
export color_self_avoiding_walk, work_stealing_walk
export PortRotationReport, FramesInFlightBound, PortProofWitness, PortTofuRecord
export port_rotation_offset, port_for_worker, port_rotation_report
export assert_port_noncontention, frames_in_flight_bound
export port_proof_catalog, port_proof_catalog_text
export port_tofu_record, port_tofu_fingerprint, verify_port_tofu, port_tofu_record_text
export GAY_SEED, HASH_SEED
export gay_colorant, gay_colordiff, gay_ripserer, gay_fractal_dimension, gay_bottleneck, gay_wasserstein, gay_persistencediagram, gay_matching  # extension surface
export GayPersistenceDiagram, GayBottleneck, GayWasserstein
export split_mix_64, hash_color_rgb, hash_color_lch, hash_color_hex  # O(1) random-access (splitmixrgb-xf)
export spi_color_u32, spi_color_hex, spi_xor_fingerprint, spi_xor_fingerprint_parallel
export spi_trit, spi_trit_sum
export assert_boundary_integrity
export CljcRuntimeColor, CljcRuntimeTransition
export cljc_core_id, cljc_runtime_color, cljc_runtime_identity, cljc_runtime_uri
export cljc_runtime_transition, verify_cljc_runtime_color, verify_cljc_transition_structure
export IPhoneProbe, IPhoneColorSpace, IPhoneColorURI, IPhoneColorRecord, IPhoneColorRegistry
export iphone_recording_count_bin, iphone_probe_embedding, iphone_probe_distance
export learn_iphone_color_space, iphone_root_color, iphone_probe_color
export iphone_color_record, iphone_color_identifier, iphone_uri, passport_uri
export parse_iphone_uri, parse_passport_uri, verify_iphone_color_record
export generate_iphone_pair_key
export register_iphone_color!, resolve_iphone_color, iphone_record_distance
export unregister_iphone_color!, purge_iphone_epoch!
export MacOSIPhoneObservation, macos_iphone_observation
export macos_iphone_observation_complete, materialize_iphone_probe

# The canonical seed of GayMCP.jl (= 0x42D = 42 + D for Douglas Adams + Deterministic).
const GAY_SEED = UInt64(1069)

# The amp-thread tag this package was born from (16 hex = exactly one UInt64).
const HASH_SEED = 0x8b449cd3828014dd

# Golden gamma: 2^64 / phi made odd. See Steele & Lea, "Fast Splittable PRNGs" (2014).
const GOLDEN_GAMMA = 0x9e3779b97f4a7c15

# --- SplitMix64 ----------------------------------------------------------------

@inline mix64(z::UInt64) = begin
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    z ⊻ (z >> 31)
end

@inline mix_gamma(z::UInt64) = mix64(z) | UInt64(1)  # force odd

# --- O(1) random-access kernels (bit-exact port of Gay.jl-splitmixrgb-xf) ------
# bob's splitmix64 == our mix64; split_mix_64 = advance-by-γ then finalize.
# Unlike color_at (O(index) via repeated split), these are constant-time and
# GPU-portable (Float32 for Metal). γ = GOLDEN_GAMMA = 0x9e3779b97f4a7c15.
@inline split_mix_64(x::UInt64) = mix64(x + GOLDEN_GAMMA)

"""
    hash_color_rgb(seed, index) -> NTuple{3,Float32}

Stateless O(1) random-access color: `mix64(seed ⊻ index·γ)` → low 3 bytes → RGB
in [0,1]³ (Float32). The constant-time companion to `color_at`.
"""
@inline function hash_color_rgb(seed::Integer, index::Integer)
    h = mix64(xor(UInt64(seed), UInt64(index) * GOLDEN_GAMMA))
    (Float32(h & 0xFF) / 255.0f0,
     Float32((h >> 8) & 0xFF) / 255.0f0,
     Float32((h >> 16) & 0xFF) / 255.0f0)
end

"""
    hash_color_lch(seed, index) -> NTuple{3,Float32}

O(1) (L,C,H) with XF visibility ranges: L∈[30,80], C∈[40,80], H∈[0,360].
"""
@inline function hash_color_lch(seed::Integer, index::Integer)
    h1 = mix64(xor(UInt64(seed), UInt64(index) * GOLDEN_GAMMA))
    h2 = mix64(h1); h3 = mix64(h2)
    (30.0f0 + Float32(h1 & 0xFFFF) / 65535.0f0 * 50.0f0,
     40.0f0 + Float32(h2 & 0xFFFF) / 65535.0f0 * 40.0f0,
     Float32(h3 & 0xFFFF) / 65535.0f0 * 360.0f0)
end

"""
    hash_color_hex(seed, index) -> String

O(1) random-access "#RRGGBB" (the constant-time companion to `color_at`).
"""
hash_color_hex(seed::Integer, index::Integer) = rgb_hex(hash_color_rgb(seed, index)...)

# --- SPI-race compatible O(1) kernel ------------------------------------------

@inline function _spi_mix64(seed::Integer, index::Integer)
    mix64(UInt64(seed) + UInt64(index) * GOLDEN_GAMMA)
end

@inline function _spi_extract_rgb(h::UInt64)
    UInt32(((h >> 16) & 0xFF) << 16 | ((h >> 8) & 0xFF) << 8 | (h & 0xFF))
end

"""
    spi_color_u32(seed, index) -> UInt32

`spi-race` compatible packed color: `0x00RRGGBB` from
`mix64(seed + GOLDEN_GAMMA * index)`. This is the cross-runtime canonical
constant-time kernel used by `b/spi-race/libspi.zig`.
"""
@inline spi_color_u32(seed::Integer, index::Integer) = _spi_extract_rgb(_spi_mix64(seed, index))

"""`spi-race` compatible color as `#RRGGBB`."""
spi_color_hex(seed::Integer, index::Integer) = "#" * uppercase(string(spi_color_u32(seed, index), base=16, pad=6))

"""
    spi_xor_fingerprint(seed, start, count) -> UInt64

XOR-fold `spi_color_u32(seed, i)` over `start:start+count-1`. Associative and
commutative, so partitioned reductions must produce the same fingerprint.
"""
function spi_xor_fingerprint(seed::Integer, start::Integer, count::Integer)
    count >= 0 || throw(ArgumentError("count must be non-negative"))
    count == 0 && return UInt64(0)
    acc = UInt64(0)
    s = UInt64(start)
    @inbounds for i in UInt64(0):UInt64(count - 1)
        acc ⊻= UInt64(spi_color_u32(seed, s + i))
    end
    acc
end

"""
    spi_xor_fingerprint_parallel(seed, n; chunks=Threads.nthreads()) -> UInt64

Parallel XOR-fold over `0:n-1`. This is a Julia reference implementation of the
`spi-race` partition-invariance contract; `chunks=1` must equal the sequential
fingerprint.
"""
function spi_xor_fingerprint_parallel(seed::Integer, n::Integer; chunks::Integer=Threads.nthreads())
    n >= 0 || throw(ArgumentError("n must be non-negative"))
    n == 0 && return UInt64(0)
    chunks = max(1, min(Int(chunks), Int(n)))
    partials = zeros(UInt64, chunks)
    Threads.@threads for tid in 1:chunks
        q, r = divrem(Int(n), chunks)
        start = (tid - 1) * q + min(tid - 1, r)
        count = q + (tid <= r ? 1 : 0)
        partials[tid] = spi_xor_fingerprint(seed, start, count)
    end
    foldl(⊻, partials; init=UInt64(0))
end

"""GF(3) trit for the `spi-race` color at `(seed, index)`, in `{-1,0,1}`."""
function spi_trit(seed::Integer, index::Integer)
    h = _spi_mix64(seed, index)
    r = Int((h >> 16) & 0xFF)
    g = Int((h >> 8) & 0xFF)
    b = Int(h & 0xFF)
    Int8(mod(r + g + b, 3) - 1)
end

"""
    spi_trit_sum(seed, start, count) -> Int8

GF(3) trit sum over a `spi-race` index range, returned as the raw mod-3 residue
`{0,1,2}` — byte-identical to `libspi.zig`'s `spi_trit_sum` (which does `@mod`).
Note this is *uncentered*, unlike the single `spi_trit` (which is `sum3-1 ∈
{-1,0,1}`); the balanced representative is `r == 2 ? -1 : r`.
"""
function spi_trit_sum(seed::Integer, start::Integer, count::Integer)
    count >= 0 || throw(ArgumentError("count must be non-negative"))
    count == 0 && return Int8(0)
    acc = 0
    s = UInt64(start)
    @inbounds for i in UInt64(0):UInt64(count - 1)
        acc = mod(acc + Int(spi_trit(seed, s + i)), 3)
    end
    Int8(acc)
end

# --- SplittableRandom ----------------------------------------------------------

mutable struct SplittableRandom <: Random.AbstractRNG
    seed::UInt64
    gamma::UInt64
end

SplittableRandom(seed::Integer) = SplittableRandom(UInt64(seed) % UInt64, GOLDEN_GAMMA)
SplittableRandom(seed::Integer, gamma::Integer) =
    SplittableRandom(UInt64(seed) % UInt64, UInt64(gamma) % UInt64)

@inline function _next!(r::SplittableRandom)
    r.seed += r.gamma          # UInt64 wraps mod 2^64 by definition
    return r.seed
end

# Make SplittableRandom a fully compliant Random.AbstractRNG in the Julia ecosystem
Random.rng_native_52(::SplittableRandom) = UInt64

@inline function Random.rand(r::SplittableRandom, ::Type{UInt64})
    mix64(_next!(r))
end

@inline function Random.rand(r::SplittableRandom, ::Type{Float64})
    Float64(Random.rand(r, UInt64)) / Float64(typemax(UInt64)) * (1 - eps(Float64))
end

"""Uniform Float64 in [0, 1). Faithful port of SplittableRandoms.jl / the
gay_julia_bridge.py reference: advance, then mix, then scale by 2^-64."""
function Base.rand(r::SplittableRandom)
    Random.rand(r, Float64)
end

# The bridge uses /2^64, not /(2^64-1); reproduce exactly:
randf(r::SplittableRandom) = Float64(mix64(_next!(r))) / 1.8446744073709552e19  # 2^64

"""Create an independent child RNG. Parent advances by two gammas; the child
gets `mix64(parent_after_first_advance)` as seed and `mix_gamma(parent_after_second)` as gamma."""
function Base.split(r::SplittableRandom)
    a = mix64(_next!(r))
    b = mix_gamma(_next!(r))
    SplittableRandom(a, b)
end

"""
    assert_boundary_integrity(r1::SplittableRandom, r2::SplittableRandom; epsilon=1e-16) -> Bool

Assert that two generators do not suffer from identity collapse. If their mapped floating-point
representations are too close to be within measurement error terms (<= epsilon) but their underlying 
64-bit unsigned integers are distinct, the system bypasses the float collapse, prints a fiery 
"one-shotted!" ANSI banner to stderr, and returns `true`. If their float difference is healthy,
returns `false`. If they are truly identical, throws an `ArgumentError`.
"""
function assert_boundary_integrity(r1::SplittableRandom, r2::SplittableRandom; epsilon::Float64=1e-16)
    # Copy to avoid mutating the original generators
    rc1 = SplittableRandom(r1.seed, r1.gamma)
    rc2 = SplittableRandom(r2.seed, r2.gamma)
    
    f1 = randf(rc1)
    f2 = randf(rc2)
    fdiff = abs(f1 - f2)
    
    if fdiff <= epsilon
        if r1.seed != r2.seed || r1.gamma != r2.gamma
            # Integer-based safeguard rescue!
            red = "\033[1;91m"
            yellow = "\033[1;93m"
            reset = "\033[0m"
            println(stderr, "\n", red, "      (  .      )   ", reset)
            println(stderr, red, "    (   )  . (      ", reset)
            println(stderr, red, "   ( )  _  ) ( ", yellow, "_ _", red, "  )   ", reset)
            println(stderr, red, "  (_(_(_(", yellow, "(_", red, "_(_", yellow, "_", red, "_)_)_)", reset)
            println(stderr, red, " 🔥 💥  ", yellow, "O N E - S H O T T E D !", red, "  💥 🔥", reset)
            println(stderr, red, "  ~~~~~~~~~~~~~~~~~~~~", reset)
            println(stderr, yellow, "   - Float representation collapsed to: ", fdiff, reset)
            println(stderr, yellow, "   - Integer absolute distinction remained: ", @sprintf("0x%016x", r1.seed), " vs ", @sprintf("0x%016x", r2.seed), reset)
            println(stderr, red, "  ====================\n", reset)
            return true
        else
            throw(ArgumentError("Absolute identity collapse: generators are identical (seed=$(@sprintf("0x%016x", r1.seed)), gamma=$(@sprintf("0x%016x", r1.gamma)))."))
        end
    end
    return false
end

# --- Okhsl (simplified, matching GayMCP.jl / gay_julia_bridge.py) --------------


function okhsl_to_rgb(h::Float64, s::Float64, l::Float64)
    hn = mod(h, 360.0) / 360.0
    c = (1 - abs(2l - 1)) * s
    x = c * (1 - abs(mod(hn * 6, 2) - 1))
    m = l - c / 2
    r, g, b = if hn < 1/6
        (c, x, 0.0)
    elseif hn < 2/6
        (x, c, 0.0)
    elseif hn < 3/6
        (0.0, c, x)
    elseif hn < 4/6
        (0.0, x, c)
    elseif hn < 5/6
        (x, 0.0, c)
    else
        (c, 0.0, x)
    end
    (clamp(r + m, 0.0, 1.0), clamp(g + m, 0.0, 1.0), clamp(b + m, 0.0, 1.0))
end

# Python `int(x*255)` truncates toward 0; for clamped non-negative `x` this
# equals `floor(Int, x*255)`. Match exactly.
rgb_hex(r, g, b) = @sprintf("#%02X%02X%02X", trunc(Int, r*255), trunc(Int, g*255), trunc(Int, b*255))

# --- Public API ----------------------------------------------------------------

"""
    color_at(index; seed=GAY_SEED, gamma=GOLDEN_GAMMA) -> String

The canonical Gay.jl color: `SplittableRandom(seed, gamma) → split(index×) → Okhsl`.
Returns "#RRGGBB".

The 64-bit `seed` is the *what*; `index` is the *where*. Two seeds that differ
in even one bit produce avalanche-different palettes. The `gamma` slot is rarely
touched; using a hash there (forced odd) walks a different lattice — much
sharper per-index divergence than the default golden gamma.
"""
function color_at(index::Integer; seed::Integer=GAY_SEED, gamma::Integer=GOLDEN_GAMMA)
    rng = SplittableRandom(seed, gamma)
    for _ in 1:index
        rng = Base.split(rng)
    end
    h = randf(rng) * 360.0
    s = 0.5 + randf(rng) * 0.4
    l = 0.35 + randf(rng) * 0.4
    rgb_hex(okhsl_to_rgb(h, s, l)...)
end

"""
    trit(index; seed=GAY_SEED, gamma=GOLDEN_GAMMA) -> Int8

The GF(3) trit drawn from the same per-index stream as `color_at`. -1 / 0 / +1
= Coplay / Witness / Play. Sum of 3 trits mod 3 is the (scalar, necessary-not-
sufficient) Čech audit; the full holonomy vector is the real invariant.
"""
function trit(index::Integer; seed::Integer=GAY_SEED, gamma::Integer=GOLDEN_GAMMA)
    rng = SplittableRandom(seed, gamma)
    for _ in 1:index
        rng = Base.split(rng)
    end
    Int8(floor(Int, randf(rng) * 3) - 1)
end

"""
    gay_seed(hex::AbstractString) -> UInt64

Parse a 16-hex-char string into a Gay.jl seed. `gay_seed("8b449cd3828014dd")` is
the seed this package was born from (an amp-thread id, but the algorithm doesn't
care — splitmix erases provenance).
"""
gay_seed(hex::AbstractString) = parse(UInt64, hex; base=16)

const FNV_OFFSET = 0xcbf29ce484222325
const FNV_PRIME = 0x100000001b3

"""
    stable_seed(x; seed=GAY_SEED) -> UInt64

Derive a deterministic 64-bit seed from arbitrary printable input. Julia's
`hash` is intentionally process-dependent, so Gay.jl uses a tiny FNV-1a pass
followed by SplitMix avalanche instead. This is the bridge from "things tried
before" to repeatable walk colors.
"""
function stable_seed(x; seed::Integer=GAY_SEED)
    h = FNV_OFFSET ⊻ UInt64(seed)
    for b in codeunits(string(x))
        h = (h ⊻ UInt64(b)) * FNV_PRIME
    end
    mix64(h)
end

_hierarchy_parts(label) = [string(label)]
function _hierarchy_parts(label::AbstractString)
    parts = split(label, r"[/:>]+")
    clean = filter(!isempty, String.(strip.(parts)))
    isempty(clean) ? [String(label)] : clean
end

"""
    hierarchical_colors(label; seed=GAY_SEED, gamma=GOLDEN_GAMMA)

Color every prefix of a hierarchical label. For `"agent/3/seed"`, the prefixes
are `"agent"`, `"agent/3"`, and `"agent/3/seed"`. Walks use the leaf color for
self-avoidance while preserving the prefix trail for reporting and replay.
"""
function hierarchical_colors(label; seed::Integer=GAY_SEED, gamma::Integer=GOLDEN_GAMMA)
    prefixes = String[]
    acc = ""
    for part in _hierarchy_parts(string(label))
        acc = isempty(acc) ? part : string(acc, "/", part)
        push!(prefixes, acc)
    end
    [(prefix, color_at(0; seed=stable_seed(prefix; seed=seed), gamma=gamma)) for prefix in prefixes]
end

_walk_color(node; seed, gamma) = last(hierarchical_colors(node; seed=seed, gamma=gamma))[2]
_neighbors(adjacency::AbstractDict, node) = get(adjacency, node, [])

_rank(node, worker::Integer, step::Integer; seed::Integer) =
    stable_seed(string(worker, "/", step, "/", node); seed=seed)

struct WalkStep
    worker::Int
    step::Int
    node::Any
    color::String
    hierarchy::Vector{Tuple{String,String}}
    stolen_from::Union{Nothing,Int}
end

struct WalkResult
    steps::Vector{WalkStep}
    touched_colors::Set{String}
    stopped_reason::String
end

function _record_step!(out::Vector{WalkStep}, touched::Set{String}, worker::Integer,
                       step::Integer, node, stolen_from; seed, gamma)
    trail = hierarchical_colors(node; seed=seed, gamma=gamma)
    color = last(trail)[2]
    push!(touched, color)
    push!(out, WalkStep(Int(worker), Int(step), node, color, trail, stolen_from))
end

function _available_neighbors(adjacency::AbstractDict, node, touched::Set{String},
                              worker::Integer, step::Integer; seed, gamma)
    candidates = Any[]
    for neighbor in _neighbors(adjacency, node)
        color = _walk_color(neighbor; seed=seed, gamma=gamma)
        color in touched && continue
        push!(candidates, (_rank(neighbor, worker, step; seed=seed), neighbor))
    end
    sort!(candidates; by=first)
    [neighbor for (_, neighbor) in candidates]
end

"""
    color_self_avoiding_walk(adjacency, start; steps=16, seed=GAY_SEED, gamma=GOLDEN_GAMMA)

Run a deterministic random-looking walk over an adjacency dictionary. Each node
gets a Gay.jl color from its hierarchical label; the walker only moves to
neighbors whose leaf color has not been touched before.
"""
function color_self_avoiding_walk(adjacency::AbstractDict, start;
                                  steps::Integer=16,
                                  seed::Integer=GAY_SEED,
                                  gamma::Integer=GOLDEN_GAMMA,
                                  touched_colors=Set{String}())
    touched = Set{String}(touched_colors)
    out = WalkStep[]
    current = start

    for step in 0:steps
        color = _walk_color(current; seed=seed, gamma=gamma)
        color in touched && return WalkResult(out, touched, "start_or_candidate_color_already_touched")
        _record_step!(out, touched, 1, step, current, nothing; seed=seed, gamma=gamma)
        step == steps && return WalkResult(out, touched, "step_limit")

        candidates = _available_neighbors(adjacency, current, touched, 1, step + 1; seed=seed, gamma=gamma)
        isempty(candidates) && return WalkResult(out, touched, "trapped")
        current = first(candidates)
    end

    WalkResult(out, touched, "step_limit")
end

function _steal!(queues::Vector{Vector{Any}}, thief::Int)
    donors = [(length(q), i) for (i, q) in enumerate(queues) if i != thief && !isempty(q)]
    isempty(donors) && return (nothing, nothing)
    sort!(donors; by=x -> (-x[1], x[2]))
    donor = donors[1][2]
    (popfirst!(queues[donor]), donor)
end

"""
    work_stealing_walk(adjacency, starts; workers=17, max_steps=128, fanout=2, seed=GAY_SEED)

Coordinate many colored self-avoiding walkers with deterministic work stealing.
Each worker owns a queue of frontier nodes; idle workers steal from the longest
non-empty queue. A global touched-color set prevents every worker from stepping
onto a color already used by any other worker.
"""
function work_stealing_walk(adjacency::AbstractDict, starts;
                            workers::Integer=17,
                            max_steps::Integer=128,
                            fanout::Integer=2,
                            seed::Integer=GAY_SEED,
                            gamma::Integer=GOLDEN_GAMMA)
    workers < 1 && throw(ArgumentError("workers must be positive"))
    fanout < 1 && throw(ArgumentError("fanout must be positive"))

    start_list = collect(starts)
    isempty(start_list) && return WalkResult(WalkStep[], Set{String}(), "no_starts")

    queues = [Any[] for _ in 1:workers]
    for worker in 1:workers
        push!(queues[worker], start_list[mod1(worker, length(start_list))])
    end

    touched = Set{String}()
    out = WalkStep[]
    step = 0

    while step < max_steps
        made_progress = false
        for worker in 1:workers
            step >= max_steps && break

            stolen_from = nothing
            node = if !isempty(queues[worker])
                popfirst!(queues[worker])
            else
                stolen, donor = _steal!(queues, worker)
                stolen_from = donor
                stolen
            end
            node === nothing && continue

            color = _walk_color(node; seed=seed, gamma=gamma)
            color in touched && continue

            _record_step!(out, touched, worker, step, node, stolen_from; seed=seed, gamma=gamma)
            made_progress = true
            step += 1

            for next in first(_available_neighbors(adjacency, node, touched, worker, step; seed=seed, gamma=gamma), fanout)
                push!(queues[worker], next)
            end
        end

        made_progress || return WalkResult(out, touched, "frontier_exhausted")
    end

    WalkResult(out, touched, "step_limit")
end

# --- Deterministic port rotation ----------------------------------------------

"""
    PortRotationReport

Witness for a deterministic port-rotation frame. `requested_processes` is the
number of concurrent workers in the frame, `port_span` is the size of the
reserved interval, and `collisions` is the number of repeated ports observed in
the generated assignment.
"""
struct PortRotationReport
    identity::String
    frame::Int
    requested_processes::Int
    port_min::Int
    port_span::Int
    offset::Int
    ports::Vector{Int}
    unique_ports::Int
    collisions::Int
    upper_bound::Int
    pigeonhole_min_collisions::Int
    saturated::Bool
end

"""
    FramesInFlightBound

Upper-bound calculation for deterministic rotation cadence. `max_rotation_hz`
is the highest frame rate allowed by both the SPI assignment throughput and the
socket-drain time. At capacity, the drain term dominates unless assignment is
slower than one full schedule per drain interval.
"""
struct FramesInFlightBound
    requested_processes::Int
    assignments_per_second::Float64
    drain_seconds::Float64
    planner_limited_hz::Float64
    drain_limited_hz::Float64
    max_rotation_hz::Float64
    spi_fast_enough_for_drain::Bool
end

"""
    PortProofWitness

One proof-style witness around deterministic port non-contention. `name` is a
stable machine key, `family` names the mathematical lens, `claim` states what
the witness proves, `evidence` gives the report-specific facts, and `verdict`
records whether the witness applies to the supplied report.
"""
struct PortProofWitness
    name::Symbol
    family::String
    claim::String
    evidence::String
    verdict::Bool
end

"""
    PortTofuRecord

Trust-on-first-use pin for one deterministic port-rotation contract. Store the
record from the first accepted run; future runs recompute it from identity,
frame, range, worker count, and seed. Any mismatch means the endpoint contract
changed and should be treated like a changed SSH host key.
"""
struct PortTofuRecord
    identity::String
    frame::Int
    requested_processes::Int
    port_min::Int
    port_span::Int
    offset::Int
    seed::UInt64
    fingerprint::UInt64
    color::String
end

function _check_port_range(port_min::Integer, port_span::Integer)
    port_min < 0 && throw(ArgumentError("port_min must be non-negative"))
    port_span < 1 && throw(ArgumentError("port_span must be positive"))
    port_min + port_span - 1 > 65535 &&
        throw(ArgumentError("port interval must fit in 0..65535"))
    return Int(port_min), Int(port_span)
end

"""
    port_rotation_offset(identity; frame=0, port_span=20000, seed=GAY_SEED)

Derive the rotation offset for one frame. This is a pure index-addressed SPI
calculation: the identity and frame determine the offset without shared state.
"""
function port_rotation_offset(identity; frame::Integer=0,
                              port_span::Integer=20000,
                              seed::Integer=GAY_SEED)
    _, span = _check_port_range(0, port_span)
    frame < 0 && throw(ArgumentError("frame must be non-negative"))
    tag = stable_seed(string(identity, "/frame/", frame); seed=seed)
    Int(tag % UInt64(span))
end

"""
    port_for_worker(worker_index, identity; frame=0, port_min=29000, port_span=20000)

Map a zero-based worker index to a deterministic listening port. The default
reserved interval is 29000:48999, below macOS' observed ephemeral-client range
49152:65535 in this workspace.
"""
function port_for_worker(worker_index::Integer, identity; frame::Integer=0,
                         port_min::Integer=29000,
                         port_span::Integer=20000,
                         seed::Integer=GAY_SEED)
    base, span = _check_port_range(port_min, port_span)
    worker_index < 0 && throw(ArgumentError("worker_index must be non-negative"))
    offset = port_rotation_offset(identity; frame=frame, port_span=span, seed=seed)
    base + mod(offset + Int(worker_index), span)
end

"""
    port_rotation_report(requested_processes, identity; kwargs...)

Generate a full frame assignment and its non-contention witness. If
`requested_processes <= port_span`, the rotation is injective. If
`requested_processes > port_span`, the pigeonhole lower bound is exact for this
round-robin projection: at least `requested_processes - port_span` collisions.
"""
function port_rotation_report(requested_processes::Integer, identity; frame::Integer=0,
                              port_min::Integer=29000,
                              port_span::Integer=20000,
                              seed::Integer=GAY_SEED)
    requested_processes < 0 &&
        throw(ArgumentError("requested_processes must be non-negative"))
    base, span = _check_port_range(port_min, port_span)
    offset = port_rotation_offset(identity; frame=frame, port_span=span, seed=seed)
    ports = [base + mod(offset + worker, span) for worker in 0:(Int(requested_processes) - 1)]
    unique_ports = length(unique(ports))
    collisions = length(ports) - unique_ports
    pigeonhole = max(0, Int(requested_processes) - span)
    PortRotationReport(string(identity), Int(frame), Int(requested_processes),
                       base, span, offset, ports, unique_ports, collisions,
                       span, pigeonhole, Int(requested_processes) == span)
end

"""
    assert_port_noncontention(requested_processes, identity; kwargs...)

Return the `PortRotationReport` if all requested workers receive distinct ports;
throw otherwise.
"""
function assert_port_noncontention(requested_processes::Integer, identity; kwargs...)
    report = port_rotation_report(requested_processes, identity; kwargs...)
    if report.requested_processes > report.upper_bound
        throw(ArgumentError("requested_processes exceeds reserved port capacity"))
    end
    report.collisions == 0 ||
        throw(ArgumentError("deterministic port rotation produced collisions"))
    report
end

"""
    frames_in_flight_bound(requested_processes; assignments_per_second, drain_seconds)

Compute the maximum safe rotation frequency. No frame-in-flight issue is
possible when the actual rotation frequency is at most
`min(assignments_per_second / requested_processes, 1 / drain_seconds)`.
"""
function frames_in_flight_bound(requested_processes::Integer;
                                assignments_per_second::Real,
                                drain_seconds::Real)
    requested_processes < 1 &&
        throw(ArgumentError("requested_processes must be positive"))
    assignments_per_second <= 0 &&
        throw(ArgumentError("assignments_per_second must be positive"))
    drain_seconds <= 0 &&
        throw(ArgumentError("drain_seconds must be positive"))

    planner_hz = Float64(assignments_per_second) / Float64(requested_processes)
    drain_hz = 1.0 / Float64(drain_seconds)
    FramesInFlightBound(Int(requested_processes),
                        Float64(assignments_per_second),
                        Float64(drain_seconds),
                        planner_hz,
                        drain_hz,
                        min(planner_hz, drain_hz),
                        planner_hz >= drain_hz)
end

_hex64(x::UInt64) = @sprintf("0x%016x", x)

"""
    port_tofu_fingerprint(identity; requested_processes=1, kwargs...) -> UInt64

Derive the first-use fingerprint for a deterministic port-rotation schedule.
The fingerprint commits to the identity, frame, process count, reserved port
range, seed, and resulting rotation offset.
"""
function port_tofu_fingerprint(identity; requested_processes::Integer=1,
                               frame::Integer=0,
                               port_min::Integer=29000,
                               port_span::Integer=20000,
                               seed::Integer=GAY_SEED)
    report = assert_port_noncontention(requested_processes, identity;
                                       frame=frame, port_min=port_min,
                                       port_span=port_span, seed=seed)
    material = join((
        "Gay.jl/port-tofu/v1",
        report.identity,
        string(report.frame),
        string(report.requested_processes),
        string(report.port_min),
        string(report.port_span),
        string(report.offset),
        string(UInt64(seed)),
    ), "|")
    stable_seed(material; seed=seed)
end

"""
    port_tofu_record(identity; requested_processes=1, kwargs...) -> PortTofuRecord

Create the TOFU pin that should be persisted or displayed on first contact.
`color` is a Gay.jl color derived from the fingerprint for quick human
comparison across Emacs panes, logs, and bridge protocols.
"""
function port_tofu_record(identity; requested_processes::Integer=1,
                          frame::Integer=0,
                          port_min::Integer=29000,
                          port_span::Integer=20000,
                          seed::Integer=GAY_SEED)
    report = assert_port_noncontention(requested_processes, identity;
                                       frame=frame, port_min=port_min,
                                       port_span=port_span, seed=seed)
    fingerprint = port_tofu_fingerprint(identity;
                                        requested_processes=requested_processes,
                                        frame=frame, port_min=port_min,
                                        port_span=port_span, seed=seed)
    PortTofuRecord(report.identity, report.frame, report.requested_processes,
                   report.port_min, report.port_span, report.offset,
                   UInt64(seed), fingerprint, color_at(0; seed=fingerprint))
end

function _same_port_tofu(a::PortTofuRecord, b::PortTofuRecord)
    a.identity == b.identity &&
        a.frame == b.frame &&
        a.requested_processes == b.requested_processes &&
        a.port_min == b.port_min &&
        a.port_span == b.port_span &&
        a.offset == b.offset &&
        a.seed == b.seed &&
        a.fingerprint == b.fingerprint &&
        a.color == b.color
end

"""
    verify_port_tofu(record; kwargs...) -> Bool

Verify a stored first-use pin against the current deterministic contract. By
default the expected contract is read from the record itself; pass any keyword
to intentionally test a proposed identity, frame, range, process count, or seed.
"""
function verify_port_tofu(record::PortTofuRecord;
                          identity=record.identity,
                          requested_processes::Integer=record.requested_processes,
                          frame::Integer=record.frame,
                          port_min::Integer=record.port_min,
                          port_span::Integer=record.port_span,
                          seed::Integer=record.seed)
    try
        expected = port_tofu_record(identity;
                                    requested_processes=requested_processes,
                                    frame=frame, port_min=port_min,
                                    port_span=port_span, seed=seed)
        return _same_port_tofu(record, expected)
    catch err
        err isa ArgumentError || rethrow()
        return false
    end
end

"""
    port_tofu_record_text(record) -> String

Render a compact first-use pin for logs, Emacs buffers, and bridge handshakes.
"""
function port_tofu_record_text(record::PortTofuRecord)
    join((
        "Port TOFU record",
        "identity: $(record.identity)",
        "frame: $(record.frame)",
        "processes: $(record.requested_processes)",
        "interval: $(record.port_min)..$(record.port_min + record.port_span - 1)",
        "offset: $(record.offset)",
        "seed: $(_hex64(record.seed))",
        "fingerprint: $(_hex64(record.fingerprint))",
        "color: $(record.color)",
    ), "\n")
end

_witness(name::Symbol, family, claim, evidence, verdict::Bool) =
    PortProofWitness(name, string(family), string(claim), string(evidence), verdict)

"""
    port_proof_catalog(requested_processes, identity; kwargs...) -> Vector{PortProofWitness}

Return maximally different proof witnesses for the same port assignment:
construction, finite exhaustion, set/cardinality, modular cancellation,
bounded-difference, cyclic-group/permutation, induction, contradiction,
pigeonhole upper bound, abduction/IBE, SPI order-independence, and OS-range
disjointness.
"""
function port_proof_catalog(requested_processes::Integer, identity; frame::Integer=0,
                            port_min::Integer=29000,
                            port_span::Integer=20000,
                            seed::Integer=GAY_SEED,
                            ephemeral_first::Integer=49152)
    report = port_rotation_report(requested_processes, identity;
                                  frame=frame, port_min=port_min,
                                  port_span=port_span, seed=seed)
    n = report.requested_processes
    span = report.port_span
    last_port = report.port_min + report.port_span - 1
    cap_ok = n <= span
    noncontention = report.collisions == 0
    exact_pigeonhole = report.collisions == report.pigeonhole_min_collisions
    has_full_range = isempty(report.ports) ||
        (minimum(report.ports) >= report.port_min && maximum(report.ports) <= last_port)

    PortProofWitness[
        _witness(
            :constructive_assignment,
            "constructive direct calculation",
            "Compute every worker port by the formula and observe no repeats.",
            "generated=$n unique=$(report.unique_ports) collisions=$(report.collisions)",
            noncontention,
        ),
        _witness(
            :finite_exhaustion,
            "exhaustion / finite model check",
            "Exhaust the finite worker index set for this frame.",
            "checked indices 0..$(max(-1, n - 1)) inside span $span",
            noncontention,
        ),
        _witness(
            :set_cardinality,
            "set cardinality",
            "The image set has the same cardinality as the domain.",
            "|ports|=$(report.unique_ports), |workers|=$n",
            report.unique_ports == n,
        ),
        _witness(
            :modular_cancellation,
            "algebra in Z/nZ",
            "Equal ports imply equal residues; subtracting the shared offset gives i == j mod span.",
            "worker indices are restricted to a prefix of length $n <= $span",
            cap_ok,
        ),
        _witness(
            :bounded_difference,
            "integer interval argument",
            "If 0 <= i,j < N <= span and i == j mod span, then |i-j| < span, so i == j.",
            "max index distance $(max(0, n - 1)) is less than span $span",
            cap_ok,
        ),
        _witness(
            :cyclic_translation,
            "group action / rotation",
            "Adding the frame offset is a bijective translation of the cyclic group Z/span.",
            "offset=$(report.offset); translation preserves distinct residues",
            cap_ok,
        ),
        _witness(
            :permutation_or_prefix,
            "permutation at capacity / prefix below capacity",
            "At N=span the assignment is a full cycle permutation; below span it is an injective prefix.",
            "N=$n, span=$span, saturated=$(report.saturated)",
            cap_ok && (n < span || report.unique_ports == span),
        ),
        _witness(
            :induction_on_prefix,
            "induction",
            "The first worker is unique; each next worker advances by one residue not used in the shorter prefix.",
            "prefix length $n never reaches a second lap when N <= span",
            cap_ok,
        ),
        _witness(
            :contradiction_minimal_duplicate,
            "proof by contradiction",
            "Assume a least duplicate pair; modular cancellation forces equal worker indices, contradicting distinctness.",
            "least duplicate cannot exist while N <= span",
            cap_ok,
        ),
        _witness(
            :pigeonhole_upper_bound,
            "pigeonhole / sharp upper bound",
            "More workers than ports force collisions; this rotation achieves the exact lower bound.",
            "collisions=$(report.collisions), pigeonhole_min=$(report.pigeonhole_min_collisions)",
            exact_pigeonhole,
        ),
        _witness(
            :abductive_best_explanation,
            "abductive inference / IBE",
            "The minimal explanation of the observed collision status is the affine cyclic rotation plus the capacity bound.",
            "hypothesis explains determinism, collisions=$(report.collisions), upper_bound=$(report.upper_bound)",
            (noncontention == cap_ok) && exact_pigeonhole,
        ),
        _witness(
            :spi_order_independence,
            "SPI / schedule independence",
            "Each port is a pure function of identity, frame, and worker index, so process order cannot introduce contention.",
            "no shared counter or mutable scheduler state appears in the assignment",
            cap_ok,
        ),
        _witness(
            :range_and_ephemeral_disjointness,
            "resource separation / OS port ranges",
            "The reserved server interval is valid TCP space and lies below the observed ephemeral client range.",
            "server=$(report.port_min)..$last_port, ephemeral_first=$ephemeral_first",
            has_full_range && last_port < Int(ephemeral_first),
        ),
    ]
end

"""
    port_proof_catalog_text(requested_processes, identity; kwargs...) -> String

Render the proof catalog in a compact, Emacs-friendly plain text format.
"""
function port_proof_catalog_text(requested_processes::Integer, identity; kwargs...)
    witnesses = port_proof_catalog(requested_processes, identity; kwargs...)
    lines = String["Proof catalog"]
    for witness in witnesses
        verdict = witness.verdict ? "ok" : "fails"
        push!(lines, string("- ", witness.name, " [", witness.family, "] ", verdict,
                            "\n  claim: ", witness.claim,
                            "\n  evidence: ", witness.evidence))
    end
    join(lines, "\n")
end

# Induced runtime fibers over exact portable .cljc core identifiers.
include("cljc_runtime_color.jl")

# Privacy-preserving iphone:// color identifiers and coarse interaction metric.
include("iphone_color_uri.jl")
include("macos_iphone_probe.jl")

# --- extension surface (lazily implemented by package extensions) --------------
# Empty generic functions; ext/ adds methods when the weakdep is loaded. This is
# the room to integrate the best of bob's Gay.jl as OPTIONAL extensions without
# importing 36k LoC of stubs. GayColorsExt(Colors.jl) → perceptual color science.
function gay_colorant end
function gay_colordiff end
function gay_ripserer end
function gay_fractal_dimension end
function gay_bottleneck end
function gay_wasserstein end
function gay_persistencediagram end
function gay_matching end

# First-class Gay versions/wrappers of PersistenceDiagrams concepts
struct GayPersistenceDiagram{D, S, T} <: AbstractVector{T}
    diagram::D
    source::S
    colors::Vector{String}
    dim::Int
end

# Delegate standard vector methods to gpd.diagram
Base.size(gpd::GayPersistenceDiagram) = size(gpd.diagram)
Base.getindex(gpd::GayPersistenceDiagram, i::Int) = gpd.diagram[i]

function GayPersistenceDiagram end

struct GayBottleneck end
struct GayWasserstein end

end # module Gay
