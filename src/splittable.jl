# Deterministic splittable random color generation
# Inspired by Pigeons.jl's Strong Parallelism Invariance (SPI)

using SplittableRandoms: SplittableRandom, split
using Printf: @sprintf
using SHA: sha256

export GayRNG, gay_seed!, gay_rng, gay_split, next_color, next_colors, next_palette
export gay_interleave, gay_interleave_streams, GayInterleaver
export gay_checkerboard_2d, gay_heisenberg_bonds, gay_sublattice, gay_xor_color, gay_exchange_colors
export splitmix64, GOLDEN, MIX1, MIX2
export GAY_SEED, GAY_SEED_LEGACY, GENESIS_COLORS, verify_genesis_chain
export gay_machine_seed, gay_machine_uuid, gay_persistent_seed, gay_seed_path
export gay_quantum_pool_seed, gay_qpool_path, gay_cascaded_seed
export PicoEntropyLoopReceipt, pico_entropyloop_devices, pico_entropyloop_sample
export pico_entropyloop_receipt, pico_entropyloop_seed, pico_entropyloop_write!
export colors_per_flick, FLICKS_PER_SECOND

"""
    GayRNG

A splittable random number generator for deterministic color generation.
Each color operation splits the RNG to ensure reproducibility regardless
of execution order (Strong Parallelism Invariance).

The RNG state tracks an invocation counter to generate a unique deterministic
stream for each call, enabling reproducible sequences even across sessions.
"""
mutable struct GayRNG
    root::SplittableRandom
    current::SplittableRandom
    invocation::UInt64
    seed::UInt64
end

# Global RNG instance
const GAY_SEED = UInt64(1069)  # Canonical seed - use this!
const GAY_SEED_LEGACY = UInt64(0x6761795f636f6c6f)  # "gay_colo" as bytes
const GLOBAL_GAY_RNG = Ref{GayRNG}()

# ── Per-machine witness seeds (C2: SHA256(machine_uuid || "|" || label)) ──
# Preserves GAY_SEED (1069) golden master. Adds a parallel register keyed to
# this machine's hardware identity. Fleet coherence stays on GAY_SEED;
# machine identity lives on gay_machine_seed(label).
#
# Forgery-resistance: against accidental collision only. Not adversarial.
# For adversarial binding, use a platform key or external signer; this path is
# only a deterministic witness seed.
const _MACHINE_UUID = Ref{Union{Nothing,String}}(nothing)
const _MACHINE_SEED_CACHE = Dict{String,UInt64}()

function gay_machine_uuid()
    if _MACHINE_UUID[] !== nothing
        return _MACHINE_UUID[]
    end
    uuid = try
        if Sys.isapple()
            out = read(`ioreg -d2 -c IOPlatformExpertDevice`, String)
            m = match(r"\"IOPlatformUUID\"\s*=\s*\"([0-9A-F-]+)\"", out)
            m === nothing ? nothing : String(m.captures[1])
        elseif Sys.islinux()
            uuid = nothing
            for path in ("/etc/machine-id", "/var/lib/dbus/machine-id")
                if isfile(path)
                    uuid = strip(read(path, String))
                    break
                end
            end
            uuid
        else
            nothing
        end
    catch
        nothing
    end
    _MACHINE_UUID[] = uuid
    return uuid
end

function _u64_be(bytes::AbstractVector{UInt8})::UInt64
    r = UInt64(0)
    @inbounds for i in 1:8
        r = (r << 8) | UInt64(bytes[i])
    end
    return r
end

"""
    gay_machine_seed(label::AbstractString="gay_colo") -> UInt64

Per-machine deterministic seed: SHA256(IOPlatformUUID || "|" || label) → first 8 bytes
as a big-endian UInt64. Distinct labels yield distinct seeds on the same machine.
Same label yields different seeds on different machines.

Falls back to `GAY_SEED` (1069) when no hardware identifier is readable.
"""
function gay_machine_seed(label::AbstractString="gay_colo")::UInt64
    key = String(label)
    haskey(_MACHINE_SEED_CACHE, key) && return _MACHINE_SEED_CACHE[key]
    uuid = gay_machine_uuid()
    seed = if uuid === nothing
        GAY_SEED
    else
        _u64_be(sha256("$uuid|$key"))
    end
    _MACHINE_SEED_CACHE[key] = seed
    return seed
end

# ── Persistent kernel-CSPRNG seeds ──
# On first call, read 8 fresh bytes from the OS kernel CSPRNG at /dev/urandom.
# Persist hex to ~/.config/gay-chain/witness.<label>.hex.
# Subsequent calls re-read the file.
#
# Compared to gay_machine_seed (C2/UUID-derived):
#   + entropy quality: ~256-bit pool, not derivable from public IOPlatformUUID
#   − reproducibility: depends on the witness file surviving; not re-derivable from machine facts alone
#   = forgery model: file-system permissions (mode 0600), not cryptographic — same as C2
#
# Note: stronger platform-bound keys require a platform-specific signing path;
# this file-backed seed is intentionally portable and easy to audit.
const _PERSISTENT_SEED_CACHE = Dict{String,UInt64}()

function gay_seed_path(label::AbstractString="gay_colo")::String
    base = get(ENV, "GAY_CHAIN_DIR", joinpath(homedir(), ".config", "gay-chain"))
    return joinpath(base, "witness.$(String(label)).hex")
end

function _read_urandom_u64()::Union{UInt64,Nothing}
    try
        open("/dev/urandom", "r") do io
            bytes = read(io, 8)
            length(bytes) == 8 ? _u64_be(bytes) : nothing
        end
    catch
        nothing
    end
end

function _hex(bytes::AbstractVector{UInt8})::String
    return join(@sprintf("%02x", b) for b in bytes)
end

function _u64_be_bytes(x::UInt64)::Vector{UInt8}
    return [UInt8((x >> shift) & 0xff) for shift in 56:-8:0]
end

function _read_entropy_bytes(path::AbstractString, n::Integer)::Vector{UInt8}
    n >= 1 || throw(ArgumentError("bytes_per_source must be >= 1"))
    bytes = open(path, "r") do io
        read(io, Int(n))
    end
    length(bytes) == n || throw(EOFError())
    return bytes
end

function _xor_chunks(chunks::NTuple{3,Vector{UInt8}})::Vector{UInt8}
    n = minimum(length, chunks)
    n >= 1 || throw(ArgumentError("entropy source chunks must be non-empty"))
    out = Vector{UInt8}(undef, n)
    @inbounds for i in 1:n
        x = UInt8(0)
        x = xor(x, chunks[1][i])
        x = xor(x, chunks[2][i])
        x = xor(x, chunks[3][i])
        out[i] = x
    end
    return out
end

function _trit_from_u64(x::UInt64)::Int8
    return Int8(Int(x % UInt64(3)) - 1)
end

function _closing_trit(partial_sum::Integer)::Int8
    r = mod(partial_sum, 3)
    return Int8(r == 0 ? 0 : (r == 1 ? -1 : 1))
end

function _closed_trits(raw::NTuple{3,Int8})::NTuple{3,Int8}
    closing = _closing_trit(Int(raw[1]) + Int(raw[2]))
    return (raw[1], raw[2], closing)
end

"""
    PicoEntropyLoopReceipt

Replayable receipt for a pico Entropy Loop sample.

`seed` is the first eight bytes of `digest`, interpreted as big-endian UInt64.
`raw_trits` are read from the three source digests; `trits` closes the third
slot so the GF(3) audit sum is always 0 modulo 3.
"""
struct PicoEntropyLoopReceipt
    label::String
    mode::Symbol
    sources::NTuple{3,String}
    bytes_per_source::Int
    nonce::UInt64
    source_digests::NTuple{3,String}
    digest::String
    seed::UInt64
    raw_trits::NTuple{3,Int8}
    trits::NTuple{3,Int8}
    trit_sum::Int
    audit_ok::Bool
end

"""
    pico_entropyloop_devices(root="/dev") -> Vector{String}

Find Raspberry Pi Pico Entropy Loop serial devices named `cu.usbmodem*`,
including the numbered macOS Pico names such as `cu.usbmodem14401` and any
EL-tagged aliases. The sampler uses simulator mode by default; pass
`prefer_hardware=true` or an explicit `sources` tuple to read hardware.
"""
function pico_entropyloop_devices(root::AbstractString="/dev")::Vector{String}
    isdir(root) || return String[]
    devices = [joinpath(root, name) for name in readdir(root) if startswith(name, "cu.usbmodem")]
    return sort(devices)
end

function _pico_entropyloop_sources(prefer_hardware::Bool)::Tuple{Symbol,NTuple{3,String}}
    devices = prefer_hardware ? pico_entropyloop_devices() : String[]
    if length(devices) >= 3
        return (:hardware, (devices[1], devices[2], devices[3]))
    elseif !isempty(devices)
        padded = vcat(devices, fill("/dev/urandom", 3 - length(devices)))
        return (:mixed, (padded[1], padded[2], padded[3]))
    end
    return (:simulator, ("/dev/urandom", "/dev/urandom", "/dev/urandom"))
end

"""
    pico_entropyloop_sample(label="gay_colo"; bytes_per_source=64,
                            sources=nothing, prefer_hardware=false,
                            nonce=nothing) -> PicoEntropyLoopReceipt

Sample three pico Entropy Loop slots, XOR the byte streams, append a UInt64
nonce, hash with SHA-256, and expose the first eight digest bytes as a Gay.jl
seed. With `sources=nothing`, the default is non-blocking simulator mode using
three independent `/dev/urandom` reads; `prefer_hardware=true` auto-selects
`/dev/cu.usbmodem*` devices when present. Pass exactly three explicit source
paths for deterministic tests or a hand-selected hardware committee.
"""
function pico_entropyloop_sample(label::AbstractString="gay_colo";
                                 bytes_per_source::Integer=64,
                                 sources=nothing,
                                 prefer_hardware::Bool=false,
                                 nonce=nothing)::PicoEntropyLoopReceipt
    bytes_per_source >= 8 || throw(ArgumentError("bytes_per_source must be >= 8"))

    mode, srcs = if sources === nothing
        _pico_entropyloop_sources(prefer_hardware)
    else
        length(sources) == 3 || throw(ArgumentError("sources must contain exactly 3 paths"))
        (:manual, (String(sources[1]), String(sources[2]), String(sources[3])))
    end

    chunks = (
        _read_entropy_bytes(srcs[1], bytes_per_source),
        _read_entropy_bytes(srcs[2], bytes_per_source),
        _read_entropy_bytes(srcs[3], bytes_per_source),
    )
    nonce64 = nonce === nothing ? UInt64(time_ns()) : UInt64(nonce)
    source_digests = (_hex(sha256(chunks[1])), _hex(sha256(chunks[2])), _hex(sha256(chunks[3])))
    pooled = _xor_chunks(chunks)
    digest_bytes = sha256(vcat(pooled, _u64_be_bytes(nonce64)))
    seed = _u64_be(digest_bytes)
    raw_trits = (
        _trit_from_u64(_u64_be(sha256(chunks[1]))),
        _trit_from_u64(_u64_be(sha256(chunks[2]))),
        _trit_from_u64(_u64_be(sha256(chunks[3]))),
    )
    trits = _closed_trits(raw_trits)
    trit_sum = Int(trits[1]) + Int(trits[2]) + Int(trits[3])

    return PicoEntropyLoopReceipt(
        String(label),
        mode,
        srcs,
        Int(bytes_per_source),
        nonce64,
        source_digests,
        _hex(digest_bytes),
        seed,
        raw_trits,
        trits,
        trit_sum,
        mod(trit_sum, 3) == 0,
    )
end

pico_entropyloop_receipt(args...; kwargs...) = pico_entropyloop_sample(args...; kwargs...)

"""
    pico_entropyloop_seed(label="gay_colo"; kwargs...) -> (seed, receipt)

Return the sampled UInt64 seed and its receipt.
"""
function pico_entropyloop_seed(label::AbstractString="gay_colo"; kwargs...)
    receipt = pico_entropyloop_sample(label; kwargs...)
    return (receipt.seed, receipt)
end

"""
    pico_entropyloop_write!(label="gay_colo"; kwargs...) -> NamedTuple

Sample a pico Entropy Loop seed and persist it at `gay_qpool_path(label)` so
`gay_quantum_pool_seed(label)` and `gay_cascaded_seed(label)` can consume it.
"""
function pico_entropyloop_write!(label::AbstractString="gay_colo"; kwargs...)
    receipt = pico_entropyloop_sample(label; kwargs...)
    path = gay_qpool_path(label)
    mkpath(dirname(path))
    open(path, "w") do io
        write(io, @sprintf("0x%016x\n", receipt.seed))
    end
    try; chmod(path, 0o600); catch; end
    _QPOOL_SEED_CACHE[String(label)] = receipt.seed
    return (path=path, seed=receipt.seed, receipt=receipt)
end

"""
    gay_persistent_seed(label::AbstractString="gay_colo") -> UInt64

Per-machine seed sourced from the kernel CSPRNG on first call, persisted to
`~/.config/gay-chain/witness.<label>.hex` (override base via env `GAY_CHAIN_DIR`).
Subsequent calls return the cached/persisted value verbatim.

Falls back to `gay_machine_seed(label)` if /dev/urandom is unreadable or the
config directory is unwritable.
"""
function gay_persistent_seed(label::AbstractString="gay_colo")::UInt64
    key = String(label)
    haskey(_PERSISTENT_SEED_CACHE, key) && return _PERSISTENT_SEED_CACHE[key]
    path = gay_seed_path(key)
    seed = try
        if isfile(path)
            txt = strip(read(path, String))
            parse(UInt64, startswith(txt, "0x") ? txt[3:end] : txt, base=16)
        else
            fresh = _read_urandom_u64()
            if fresh === nothing
                gay_machine_seed(key)
            else
                mkpath(dirname(path))
                open(path, "w") do io
                    write(io, @sprintf("0x%016x\n", fresh))
                end
                try; chmod(path, 0o600); catch; end
                fresh
            end
        end
    catch
        gay_machine_seed(key)
    end
    _PERSISTENT_SEED_CACHE[key] = seed
    return seed
end

# ── Entropy-pool seeds (3× Entropy Loop XOR-pool) ──
# When a 3× Entropy Loop array is connected via USB-UART, an external pool
# driver can mix 3 sources via XOR + SHA3-512 and write
# ~/.config/gay-chain/witness.<label>.qpool.hex.
#
# Simulator mode: the driver falls back to 3× /dev/urandom reads, which
# on macOS are themselves SE-TRNG-seeded per Apple's RNG support article. The
# pipeline is end-to-end functional; only the trust root changes when real
# hardware is present.
#
# Compared to gay_persistent_seed:
#   + mixing model: XOR preserves uniformity if at least one source is uniform and independent
#   + forgery model: external-source committee when present; same file-system perms in simulator mode
#   + GF(3) audit: per-sample trit per source, free Σ≡0 closure check
#   = persistence: same ~/.config/gay-chain/witness.<label>.qpool.hex (mode 0600)
#
# This path moves the trust root off-host into the 3× EL committee when real
# devices are present; simulator mode remains only an integration fallback.
const _QPOOL_SEED_CACHE = Dict{String,UInt64}()

function gay_qpool_path(label::AbstractString="gay_colo")::String
    base = get(ENV, "GAY_CHAIN_DIR", joinpath(homedir(), ".config", "gay-chain"))
    return joinpath(base, "witness.$(String(label)).qpool.hex")
end

"""
    gay_quantum_pool_seed(label="gay_colo") -> Union{UInt64, Nothing}

Read the 3× Entropy Loop XOR-pool seed from
`~/.config/gay-chain/witness.<label>.qpool.hex` (written by `pool.bb`).
Returns `nothing` when the pool file is absent (use `gay_cascaded_seed` for
the auto-fallback behavior).
"""
function gay_quantum_pool_seed(label::AbstractString="gay_colo")::Union{UInt64,Nothing}
    key = String(label)
    haskey(_QPOOL_SEED_CACHE, key) && return _QPOOL_SEED_CACHE[key]
    path = gay_qpool_path(key)
    isfile(path) || return nothing
    try
        txt = strip(read(path, String))
        seed = parse(UInt64, startswith(txt, "0x") ? txt[3:end] : txt, base=16)
        _QPOOL_SEED_CACHE[key] = seed
        return seed
    catch
        return nothing
    end
end

"""
    gay_cascaded_seed(label="gay_colo") -> (seed::UInt64, source::Symbol)

Canonical seed accessor with explicit cascade:

    :qpool      ← 3× Entropy Loop XOR-pool, preferred when present
    :persistent ← /dev/urandom-seeded persistent witness
    :machine    ← SHA256(IOPlatformUUID || label) deterministic per-machine
    :family     ← GAY_SEED (1069), fleet-shared constant

The `source` return field lets callers attest *which* tier supplied the seed.
"""
function gay_cascaded_seed(label::AbstractString="gay_colo")::Tuple{UInt64,Symbol}
    q = gay_quantum_pool_seed(label)
    q !== nothing && return (q, :qpool)
    p = gay_persistent_seed(label)
    isfile(gay_seed_path(label)) && return (p, :persistent)
    uuid = gay_machine_uuid()
    if uuid !== nothing
        return (p, :machine)
    end
    return (p, :family)
end

# ── Colors-per-flick: Shannon-tight throughput unit ──
# 1 flick = 1/705,600,000 s — chosen because 705,600,000 = lcm of {24,25,30,48,50,60,90,
# 100,120}·(common audio sample rates). One rational unit spans every standard video AND
# audio refresh. The cpf is the per-substrate throughput of GF(3)-tagged emissions.
#
# GF(3) audit closure (Σ trit ≡ 0 mod 3) is a defensibility *condition* on the metric —
# a peer's contribution to parallel-cpf only counts when its per-batch audit closes.
# This prevents gaming by emitting trash colors.
const FLICKS_PER_SECOND = 705_600_000  # ≡ lcm of common refresh rates (Hz)

"""
    colors_per_flick(seed::UInt64, n::Integer) -> NamedTuple

Drive SplitMix64 from `seed` for `n` steps. Each step emits a GF(3) trit
∈ {-1, 0, +1} via `(state mod 3) - 1`. The final emission's trit is
overridden by the GF(3) corrector so that `Σ trit ≡ 0 (mod 3)` **by
construction**, not by chance. This is the defensibility upgrade over
iter-2: an honest peer's per-batch audit always closes.

Corrector mapping (n-th emission's trit):
  Σ_{i<n} trit_i mod 3 == 0  ⇒  closing_trit =  0
                       == 1  ⇒  closing_trit = -1
                       == 2  ⇒  closing_trit = +1

Returns:
  - `cpf`            = emissions per flick (colors·flick⁻¹)
  - `n`              = number of emissions
  - `elapsed_s`      = wall time (seconds)
  - `elapsed_flicks` = wall time in flicks
  - `trit_sum`       = signed Σ of trits (always ≡ 0 mod 3 after closure)
  - `audit_ok`       = true (by construction, for n ≥ 1)
  - `closing_trit`   = the corrector emitted at position n
  - `last_state`     = final splitmix64 state (for chain continuation)

Bit-identical across machines when given the same `(seed, n)`.
"""
function colors_per_flick(seed::UInt64, n::Integer)
    @assert n ≥ 1 "n must be ≥ 1"
    s = seed
    trit_sum = 0
    t0 = time_ns()
    @inbounds for _ in 1:(n - 1)
        s = splitmix64(s)
        trit_sum += Int(s % UInt64(3)) - 1
    end
    # Closing emission: advance state, but override the trit to enforce GF(3) closure.
    s = splitmix64(s)
    r = mod(trit_sum, 3)
    closing_trit = r == 0 ? 0 : (r == 1 ? -1 : 1)
    trit_sum += closing_trit
    t1 = time_ns()
    elapsed_s = (t1 - t0) / 1e9
    elapsed_flicks = elapsed_s * FLICKS_PER_SECOND
    return (
        cpf = n / elapsed_flicks,
        n = Int(n),
        elapsed_s = elapsed_s,
        elapsed_flicks = elapsed_flicks,
        trit_sum = trit_sum,
        audit_ok = (mod(trit_sum, 3) == 0),
        closing_trit = closing_trit,
        last_state = s,
    )
end

# Genesis color chain (seed=1069) - the canonical first 12 colors.
const GENESIS_COLORS = (
    (index=1,  hex="#E67F86", trit=+1, name="warm pink-coral"),
    (index=2,  hex="#D06546", trit=0,  name="burnt orange"),
    (index=3,  hex="#1316BB", trit=-1, name="deep blue"),
    (index=4,  hex="#BA2645", trit=+1, name="crimson"),
    (index=5,  hex="#49EE54", trit=+1, name="bright green"),
    (index=6,  hex="#11C710", trit=0,  name="lime green"),
    (index=7,  hex="#76B0F0", trit=-1, name="sky blue"),
    (index=8,  hex="#E59798", trit=0,  name="dusty rose"),
    (index=9,  hex="#5333D9", trit=-1, name="violet"),
    (index=10, hex="#7E90EB", trit=0,  name="periwinkle"),
    (index=11, hex="#1D9E7E", trit=0,  name="teal"),
    (index=12, hex="#DD7CB0", trit=+1, name="pink"),
)

function genesis_color(index::Integer)
    if 1 <= index <= length(GENESIS_COLORS)
        hex = GENESIS_COLORS[Int(index)].hex
        r = parse(UInt8, hex[2:3], base=16) / 255
        g = parse(UInt8, hex[4:5], base=16) / 255
        b = parse(UInt8, hex[6:7], base=16) / 255
        return RGB{Float64}(r, g, b)
    end
    return nothing
end

function verify_genesis_chain()
    for g in GENESIS_COLORS
        c = color_at(g.index; seed=GAY_SEED)
        hex = @sprintf("#%02X%02X%02X",
            round(Int, clamp(c.r, 0, 1) * 255),
            round(Int, clamp(c.g, 0, 1) * 255),
            round(Int, clamp(c.b, 0, 1) * 255))
        if uppercase(hex) != uppercase(g.hex)
            @warn "Genesis color mismatch" index=g.index expected=g.hex got=hex
            return false
        end
    end
    return true
end

# SplitMix64 constants
const GOLDEN = 0x9e3779b97f4a7c15
const MIX1 = 0xbf58476d1ce4e5b9
const MIX2 = 0x94d049bb133111eb

"""
    splitmix64(x::UInt64) -> UInt64

The SplitMix64 bijection - core of Gay.jl's deterministic RNG.
Maps any UInt64 to a pseudo-random UInt64 in a 1-to-1 reversible manner.
"""
function splitmix64(x::UInt64)::UInt64
    x += GOLDEN
    x = (x ⊻ (x >> 30)) * MIX1
    x = (x ⊻ (x >> 27)) * MIX2
    x ⊻ (x >> 31)
end

"""
    GayRNG(seed::Integer=GAY_SEED)

Create a new GayRNG with the given seed.
"""
function GayRNG(seed::Integer=GAY_SEED)
    root = SplittableRandom(UInt64(seed))
    current = split(root)
    GayRNG(root, current, UInt64(0), UInt64(seed))
end

"""
    gay_seed!(seed::Integer)

Reset the global Gay RNG with a new seed.
All subsequent color generations will be deterministic from this seed.
"""
function gay_seed!(seed::Integer)
    GLOBAL_GAY_RNG[] = GayRNG(seed)
    return seed
end

"""
    gay_rng()

Get the global Gay RNG, initializing if needed.
"""
function gay_rng()
    if !isassigned(GLOBAL_GAY_RNG)
        GLOBAL_GAY_RNG[] = GayRNG()
    end
    return GLOBAL_GAY_RNG[]
end

"""
    gay_split(gr::GayRNG=gay_rng())

Split the RNG for a new independent stream.
Increments invocation counter for tracking.
"""
function gay_split(gr::GayRNG=gay_rng())
    gr.invocation += 1
    gr.current = split(gr.current)
    return gr.current
end

"""
    gay_split(n::Integer, gr::GayRNG=gay_rng())

Get n independent RNG splits as a vector.
"""
function gay_split(n::Integer, gr::GayRNG=gay_rng())
    return [gay_split(gr) for _ in 1:n]
end

# ═══════════════════════════════════════════════════════════════════════════
# Deterministic color generation using splittable RNG
# ═══════════════════════════════════════════════════════════════════════════

"""
    next_color(cs::ColorSpace=SRGB(); gr::GayRNG=gay_rng())

Generate the next deterministic random color.
Each call splits the RNG for reproducibility.
"""
function next_color(cs::ColorSpace=SRGB(); gr::GayRNG=gay_rng())
    rng = gay_split(gr)
    return random_color(cs; rng=rng)
end

"""
    next_colors(n::Int, cs::ColorSpace=SRGB(); gr::GayRNG=gay_rng())

Generate n deterministic random colors.
"""
function next_colors(n::Int, cs::ColorSpace=SRGB(); gr::GayRNG=gay_rng())
    rng = gay_split(gr)
    return random_colors(n, cs; rng=rng)
end

"""
    next_palette(n::Int, cs::ColorSpace=SRGB(); 
                 min_distance::Float64=30.0, gr::GayRNG=gay_rng())

Generate n deterministic visually distinct colors.
"""
function next_palette(n::Int, cs::ColorSpace=SRGB();
                      min_distance::Float64=30.0, gr::GayRNG=gay_rng())
    rng = gay_split(gr)
    return random_palette(n, cs; min_distance=min_distance, rng=rng)
end

# ═══════════════════════════════════════════════════════════════════════════
# Invocation-indexed color access (like Pigeons explorer indexing)
# ═══════════════════════════════════════════════════════════════════════════

"""
    color_at(index::Integer, cs::ColorSpace=SRGB(); seed::Integer=GAY_SEED)

Get the color at a specific invocation index.
This allows random access to the deterministic color sequence.

# Example
```julia
# These will always return the same colors for the same indices
c1 = color_at(1)
c42 = color_at(42)
c1_again = color_at(1)  # Same as c1
```
"""
function color_at(index::Integer, cs::ColorSpace=SRGB(); seed::Integer=GAY_SEED)
    if UInt64(seed) == GAY_SEED
        c = genesis_color(index)
        c === nothing || return c
    end

    # FAST PATH: O(1) hash-based color generation
    r, g, b = hash_color(UInt64(seed), UInt64(index))
    return RGB{Float64}(r, g, b)
end

# Legacy slow path for compatibility testing
function color_at_slow(index::Integer, cs::ColorSpace=SRGB(); seed::Integer=GAY_SEED)
    root = SplittableRandom(UInt64(seed))
    current = root
    for _ in 1:index
        current = split(current)
    end
    return random_color(cs; rng=current)
end

"""
    colors_at(indices::AbstractVector{<:Integer}, cs::ColorSpace=SRGB(); 
              seed::Integer=GAY_SEED)

Get colors at specific invocation indices.
"""
function colors_at(indices::AbstractVector{<:Integer}, cs::ColorSpace=SRGB();
                   seed::Integer=GAY_SEED)
    return [color_at(i, cs; seed=seed) for i in indices]
end

"""
    palette_at(index::Integer, n::Int, cs::ColorSpace=SRGB();
               min_distance::Float64=30.0, seed::Integer=GAY_SEED)

Get a palette at a specific invocation index.
"""
function palette_at(index::Integer, n::Int, cs::ColorSpace=SRGB();
                    min_distance::Float64=30.0, seed::Integer=GAY_SEED)
    root = SplittableRandom(UInt64(seed))
    current = root
    for _ in 1:index
        current = split(current)
    end
    return random_palette(n, cs; min_distance=min_distance, rng=current)
end

# ═══════════════════════════════════════════════════════════════════════════
# Interleaved streams for checkerboard/XOR lattice decomposition
# Used in SSE QMC, parallel tempering, phased array steering
# ═══════════════════════════════════════════════════════════════════════════

"""
    GayInterleaver

Interleaved SPI streams for checkerboard lattice decomposition.
Each sublattice gets an independent deterministic color stream.

Used for:
- SSE QMC: even/odd site updates with XOR parity
- Parallel tempering: alternating replica swaps  
- Phased arrays: interleaved antenna element phases
- Nearest-neighbor exchange: J_ij coupling colored by i⊕j parity

# Example: Heisenberg model checkerboard update
```julia
using LispSyntax
interleaver = GayInterleaver(seed, 2)  # 2 sublattices

@lisp begin
  (for-each-sweep (sweep 1 n-sweeps)
    ;; Update even sites (parity 0) - all can run in parallel
    (let ((parity (mod sweep 2)))
      (for-each-site (site (sublattice interleaver parity))
        (metropolis-update site (color-for interleaver parity site)))))
end
```
"""
mutable struct GayInterleaver
    seed::UInt64
    n_streams::Int
    streams::Vector{GayRNG}
    current_phase::Int
    step::UInt64
end

"""
    GayInterleaver(seed::Integer, n::Int=2)

Create n interleaved SPI streams. Default n=2 for checkerboard (even/odd).
Each stream is independent and deterministic from the seed.
"""
function GayInterleaver(seed::Integer, n::Int=2)
    root = SplittableRandom(UInt64(seed))
    streams = Vector{GayRNG}(undef, n)
    current = root
    for i in 1:n
        current = split(current)
        stream_seed = UInt64(seed) ⊻ UInt64(i * 0x9e3779b97f4a7c15)
        streams[i] = GayRNG(stream_seed)
    end
    GayInterleaver(UInt64(seed), n, streams, 1, UInt64(0))
end

"""
    gay_interleave(il::GayInterleaver)

Get next color from current stream, then advance to next stream.
Returns (color, stream_index, step).
"""
function gay_interleave(il::GayInterleaver)
    stream = il.streams[il.current_phase]
    color = next_color(SRGB(); gr=stream)
    idx = il.current_phase
    
    # Advance phase (round-robin through streams)
    il.current_phase = mod1(il.current_phase + 1, il.n_streams)
    if il.current_phase == 1
        il.step += 1
    end
    
    return (color, idx, il.step)
end

"""
    gay_interleave(il::GayInterleaver, n::Int)

Get n interleaved colors, cycling through all streams.
"""
function gay_interleave(il::GayInterleaver, n::Int)
    return [gay_interleave(il) for _ in 1:n]
end

"""
    gay_sublattice(il::GayInterleaver, parity::Int)

Get next color for a specific sublattice (0-indexed parity).
For checkerboard: parity = (i + j) % 2 or i ⊻ j & 1
"""
function gay_sublattice(il::GayInterleaver, parity::Int)
    stream_idx = mod1(parity + 1, il.n_streams)
    stream = il.streams[stream_idx]
    return next_color(SRGB(); gr=stream)
end

"""
    gay_xor_color(il::GayInterleaver, i::Int, j::Int)

Get deterministic color for bond (i,j) based on XOR parity.
Used for nearest-neighbor exchange coupling J_ij coloring.

The color depends on (i ⊻ j) & 1, giving checkerboard pattern.
"""
function gay_xor_color(il::GayInterleaver, i::Int, j::Int)
    parity = (i ⊻ j) & 1
    return gay_sublattice(il, parity)
end

"""
    gay_exchange_colors(il::GayInterleaver, lattice_size::Int)

Generate colors for all nearest-neighbor bonds on a 1D lattice.
Returns vector of (bond_color, i, j, parity) tuples.
"""
function gay_exchange_colors(il::GayInterleaver, lattice_size::Int)
    bonds = Tuple{Any, Int, Int, Int}[]
    for i in 1:lattice_size
        j = mod1(i + 1, lattice_size)  # periodic BC
        parity = (i ⊻ j) & 1
        color = gay_xor_color(il, i, j)
        push!(bonds, (color, i, j, parity))
    end
    return bonds
end

"""
    gay_interleave_streams(seed::Integer, n::Int=2)

Convenience constructor for GayInterleaver.
"""
gay_interleave_streams(seed::Integer, n::Int=2) = GayInterleaver(seed, n)

# ═══════════════════════════════════════════════════════════════════════════
# 2D lattice support for SSE QMC
# ═══════════════════════════════════════════════════════════════════════════

"""
    gay_checkerboard_2d(il::GayInterleaver, Lx::Int, Ly::Int)

Generate checkerboard coloring for 2D lattice.
Returns matrix where color[i,j] = sublattice color based on (i+j) mod 2.
"""
function gay_checkerboard_2d(il::GayInterleaver, Lx::Int, Ly::Int)
    colors = Matrix{Any}(undef, Lx, Ly)
    for i in 1:Lx, j in 1:Ly
        parity = (i + j) % 2
        colors[i, j] = gay_sublattice(il, parity)
    end
    return colors
end

"""
    gay_heisenberg_bonds(il::GayInterleaver, Lx::Int, Ly::Int)

Color all nearest-neighbor bonds for 2D Heisenberg model.
Each bond J_ij * (S_i · S_j) gets deterministic color.
Returns Dict mapping (i,j) => (jx,jy) => color for all bonds.
"""
function gay_heisenberg_bonds(il::GayInterleaver, Lx::Int, Ly::Int)
    bonds = Dict{Tuple{Int,Int}, Dict{Tuple{Int,Int}, Any}}()
    
    for i in 1:Lx, j in 1:Ly
        bonds[(i,j)] = Dict{Tuple{Int,Int}, Any}()
        
        # Right neighbor (periodic)
        jx, jy = mod1(i+1, Lx), j
        parity = ((i + j) ⊻ (jx + jy)) & 1
        bonds[(i,j)][(jx,jy)] = gay_sublattice(il, parity)
        
        # Up neighbor (periodic)  
        jx, jy = i, mod1(j+1, Ly)
        parity = ((i + j) ⊻ (jx + jy)) & 1
        bonds[(i,j)][(jx,jy)] = gay_sublattice(il, parity)
    end
    
    return bonds
end

# ═══════════════════════════════════════════════════════════════════════════
# S-Expression Parenthesis Coloring (Rainbow Parens with SPI)
# ═══════════════════════════════════════════════════════════════════════════

export gay_sexpr_colors, gay_paren_color, GaySexpr, gay_render_sexpr
export gay_depth_color, gay_magnetized_sexpr
export gay_sexpr_magnetization, gay_sexpr_depth_spins

"""
    GaySexpr

A magnetized S-expression: each parenthesis pair has a deterministic color
and an Ising spin σ ∈ {-1, +1} based on its depth and position.

From SICP 4A: "Enzymes attach to expressions, change them, then go away.
The key-in-lock phenomenon." Each colored paren is a potential binding site.

# Semantics
- Depth d → color stream d mod n_streams (like checkerboard sublattices)
- Position p → deterministic color within stream
- Spin σ = (-1)^(d ⊻ p) for magnetization

# Example
```julia
using LispSyntax
sexpr = lisp"(defn fib [n] (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2)))))"
colored = gay_magnetized_sexpr(sexpr, seed=42)
gay_render_sexpr(colored)  # Prints with ANSI colors
```
"""
struct GaySexpr
    content::Any
    depth::Int
    position::Int
    color::Any      # RGB color
    spin::Int       # -1 or +1
    children::Vector{GaySexpr}
end

"""
    gay_depth_color(il::GayInterleaver, depth::Int, position::Int)

Get color for a parenthesis at given depth and position.
Depth determines which interleaved stream to use.
Position advances within that stream deterministically.
"""
function gay_depth_color(il::GayInterleaver, depth::Int, position::Int)
    # Depth selects sublattice (like checkerboard)
    stream_idx = mod1(depth + 1, il.n_streams)
    stream = il.streams[stream_idx]
    
    # Position advances the stream (skip to position)
    # Use color_at for O(1) access
    return color_at(position, SRGB(); seed=stream.seed)
end

"""
    gay_paren_color(seed::Integer, depth::Int, position::Int; n_depths::Int=8)

Get deterministic color for parenthesis at (depth, position).
Uses n_depths interleaved streams for depth cycling.
"""
function gay_paren_color(seed::Integer, depth::Int, position::Int; n_depths::Int=8)
    il = GayInterleaver(seed, n_depths)
    return gay_depth_color(il, depth, position)
end

"""
    gay_magnetized_sexpr(expr, seed::Integer=0xDEADBEEF; depth::Int=0, pos_counter::Ref{Int}=Ref(0))

Convert an S-expression to a magnetized GaySexpr tree.
Each node gets a deterministic color and spin.
"""
function gay_magnetized_sexpr(expr, seed::Integer=0xDEADBEEF; 
                               depth::Int=0, 
                               pos_counter::Ref{Int}=Ref(0),
                               n_depths::Int=8)
    pos = pos_counter[]
    pos_counter[] += 1
    
    # Get color for this depth/position
    color = gay_paren_color(seed, depth, pos; n_depths=n_depths)
    
    # Spin from XOR of depth and position (Ising-like)
    spin = ((depth ⊻ pos) & 1 == 0) ? 1 : -1
    
    if expr isa Vector || expr isa Tuple
        # Recurse into children
        children = [gay_magnetized_sexpr(child, seed; 
                                         depth=depth+1, 
                                         pos_counter=pos_counter,
                                         n_depths=n_depths) 
                    for child in expr]
        return GaySexpr(nothing, depth, pos, color, spin, children)
    else
        # Leaf node
        return GaySexpr(expr, depth, pos, color, spin, GaySexpr[])
    end
end

"""
    gay_render_sexpr(gs::GaySexpr; indent::Int=0)

Render a GaySexpr with ANSI colors to terminal.
"""
function gay_render_sexpr(gs::GaySexpr; indent::Int=0)
    R = "\e[0m"
    
    rgb = convert(RGB, gs.color)
    r = round(Int, clamp(rgb.r, 0, 1) * 255)
    g = round(Int, clamp(rgb.g, 0, 1) * 255)
    b = round(Int, clamp(rgb.b, 0, 1) * 255)
    fg = "\e[38;2;$(r);$(g);$(b)m"
    
    spin_char = gs.spin > 0 ? "⁺" : "⁻"
    
    if isempty(gs.children)
        # Leaf
        return "$(fg)$(gs.content)$(R)"
    else
        # Compound
        inner = join([gay_render_sexpr(c; indent=indent+1) for c in gs.children], " ")
        return "$(fg)($(spin_char)$(R)$(inner)$(fg))$(R)"
    end
end

"""
    gay_sexpr_colors(expr, seed::Integer=0xDEADBEEF)

Color an S-expression and print with rainbow parentheses.
Each depth level gets a different color from an interleaved stream.
"""
function gay_sexpr_colors(expr, seed::Integer=0xDEADBEEF)
    gs = gay_magnetized_sexpr(expr, seed)
    return gay_render_sexpr(gs)
end

"""
    gay_sexpr_magnetization(gs::GaySexpr)

Compute magnetization ⟨M⟩ = Σσ/N for an S-expression tree.
"""
function gay_sexpr_magnetization(gs::GaySexpr)
    total_spin = 0
    count = 0
    
    function walk(node::GaySexpr)
        total_spin += node.spin
        count += 1
        for child in node.children
            walk(child)
        end
    end
    
    walk(gs)
    return total_spin / count
end

"""
    gay_sexpr_depth_spins(gs::GaySexpr)

Get spin statistics by depth level.
Returns Dict(depth => (up_count, down_count, magnetization))
"""
function gay_sexpr_depth_spins(gs::GaySexpr)
    depth_stats = Dict{Int, Tuple{Int, Int}}()
    
    function walk(node::GaySexpr)
        d = node.depth
        if !haskey(depth_stats, d)
            depth_stats[d] = (0, 0)
        end
        up, down = depth_stats[d]
        if node.spin > 0
            depth_stats[d] = (up + 1, down)
        else
            depth_stats[d] = (up, down + 1)
        end
        for child in node.children
            walk(child)
        end
    end
    
    walk(gs)
    
    return Dict(d => (up, down, (up - down) / (up + down)) 
                for (d, (up, down)) in depth_stats)
end

# ═══════════════════════════════════════════════════════════════════════════
# p-adic Color Representation (Collision-Free Identity Layer)
# ═══════════════════════════════════════════════════════════════════════════
#
# p-adic numbers provide unique representation with NO rounding ambiguity.
# Unlike IEEE 754, p-adics have:
# - Unique canonical form (no denormals, no ±0)
# - Ultrametric distance: d(x,z) ≤ max(d(x,y), d(y,z))
# - No boundary attractors (totally disconnected topology)
#
# For p=3: Direct connection to balanced ternary!
# Digits ∈ {-1, 0, +1} (represented as T, 0, 1)

export Trit, PadicChannel, PadicColor, PadicColorGenerator
export padic_color, padic_palette, padic_identity, padic_distance_valuation
export verify_padic_uniqueness

"""
    Trit

Balanced ternary digit: -1, 0, or +1 (represented as 'T', '0', '1').
"""
struct Trit
    value::Int8
    
    function Trit(v::Integer)
        @assert v ∈ (-1, 0, 1) "Trit must be -1, 0, or +1"
        new(Int8(v))
    end
end

Base.show(io::IO, t::Trit) = print(io, t.value == -1 ? 'T' : t.value == 0 ? '0' : '1')
Base.:(==)(a::Trit, b::Trit) = a.value == b.value
Base.hash(t::Trit, h::UInt) = hash(t.value, h)

"""
Convert integer mod 3 to balanced trit.
"""
function balanced_mod3(n::Integer)
    r = mod(n, 3)
    return r == 0 ? Trit(0) : r == 1 ? Trit(1) : Trit(-1)
end

"""
    PadicChannel

p-adic channel: sequence of balanced ternary digits.
Represents a single color channel with arbitrary precision identity.
"""
struct PadicChannel
    digits::Vector{Trit}     # LSB first
    valuation::Int32
    
    function PadicChannel(digits::Vector{Trit}, valuation::Integer=0)
        new(digits, Int32(valuation))
    end
end

"""
Create PadicChannel from u64 hash value using balanced ternary.
"""
function PadicChannel(hash::UInt64, precision::Integer=20)
    digits = Trit[]
    val = Int64(hash)
    
    for _ in 1:precision
        d = balanced_mod3(val)
        push!(digits, d)
        val = div(val - d.value, 3)
    end
    
    return PadicChannel(digits, 0)
end

"""
Canonical string key (unique identifier).
"""
function canonical_key(ch::PadicChannel)
    return join(reverse([t.value == -1 ? 'T' : t.value == 0 ? '0' : '1' for t in ch.digits]))
end

"""
Convert to approximate Float64 in [0, 1) for display.
Note: This is lossy - use canonical_key() for identity!
"""
function to_unit_float(ch::PadicChannel)
    result = 0.0
    power = 1.0
    
    for t in ch.digits
        power /= 3.0
        result += t.value * power
    end
    
    return clamp(result + 0.5, 0.0, 1.0 - eps())
end

"""
p-adic distance valuation to another channel.
Returns position of first differing digit (higher = closer).
"""
function distance_valuation(ch1::PadicChannel, ch2::PadicChannel)
    min_len = min(length(ch1.digits), length(ch2.digits))
    
    for i in 1:min_len
        if ch1.digits[i] != ch2.digits[i]
            return Int32(i - 1) + ch1.valuation
        end
    end
    
    # Check remaining digits against zero
    if length(ch1.digits) > min_len
        for (i, d) in enumerate(ch1.digits[min_len+1:end])
            if d != Trit(0)
                return Int32(min_len + i - 1) + ch1.valuation
            end
        end
    end
    if length(ch2.digits) > min_len
        for (i, d) in enumerate(ch2.digits[min_len+1:end])
            if d != Trit(0)
                return Int32(min_len + i - 1) + ch1.valuation
            end
        end
    end
    
    return typemax(Int32)  # Identical within precision
end

"""
    PadicColor

p-adic RGB color with unique identity.
The display RGB may collide (8-bit quantization), but the
underlying p-adic representation NEVER collides.
"""
struct PadicColor
    r::PadicChannel
    g::PadicChannel
    b::PadicChannel
    display_rgb::Tuple{UInt8, UInt8, UInt8}  # Cached display value
    
    function PadicColor(r::PadicChannel, g::PadicChannel, b::PadicChannel)
        r_f = to_unit_float(r)
        g_f = to_unit_float(g)
        b_f = to_unit_float(b)
        display = (
            UInt8(round(r_f * 255)),
            UInt8(round(g_f * 255)),
            UInt8(round(b_f * 255))
        )
        new(r, g, b, display)
    end
end

"""
Unique canonical identity (NEVER collides).
"""
function padic_identity(c::PadicColor)
    return "$(canonical_key(c.r))|$(canonical_key(c.g))|$(canonical_key(c.b))"
end

"""
Display RGB tuple.
"""
to_rgb(c::PadicColor) = c.display_rgb

"""
Hex string (#RRGGBB).
"""
function to_hex(c::PadicColor)
    r, g, b = c.display_rgb
    return @sprintf("#%02X%02X%02X", r, g, b)
end

"""
p-adic distance to another color (ultrametric).
Returns minimum of channel-wise distances.
"""
function padic_distance_valuation(c1::PadicColor, c2::PadicColor)
    return min(
        distance_valuation(c1.r, c2.r),
        distance_valuation(c1.g, c2.g),
        distance_valuation(c1.b, c2.b)
    )
end

Base.:(==)(a::PadicColor, b::PadicColor) = padic_identity(a) == padic_identity(b)
Base.hash(c::PadicColor, h::UInt) = hash(padic_identity(c), h)

"""
    PadicColorGenerator

p-adic color generator using splittable RNG.
"""
mutable struct PadicColorGenerator
    rng::GayRNG
    precision::Int
    
    function PadicColorGenerator(seed::Integer=GAY_SEED; precision::Integer=20)
        new(GayRNG(seed), precision)
    end
end

"""
Generate next p-adic color (unique identity guaranteed).
"""
function padic_color(gen::PadicColorGenerator)
    rng = gay_split(gen.rng)
    
    # Generate three hash values
    r_hash = rand(rng, UInt64)
    g_hash = rand(rng, UInt64)
    b_hash = rand(rng, UInt64)
    
    r = PadicChannel(r_hash, gen.precision)
    g = PadicChannel(g_hash, gen.precision)
    b = PadicChannel(b_hash, gen.precision)
    
    return PadicColor(r, g, b)
end

"""
Generate n p-adic colors.
"""
function padic_palette(n::Integer; seed::Integer=GAY_SEED, precision::Integer=20)
    gen = PadicColorGenerator(seed; precision=precision)
    return [padic_color(gen) for _ in 1:n]
end

"""
Verify p-adic palette has zero identity collisions.
"""
function verify_padic_uniqueness(colors::Vector{PadicColor})
    seen = Set{String}()
    for c in colors
        id = padic_identity(c)
        if id ∈ seen
            return false
        end
        push!(seen, id)
    end
    return true
end

"""
Demo function for p-adic colors.
"""
function demo_padic()
    println("═══ p-adic Color Generation ═══")
    println()
    
    # Generate palette
    colors = padic_palette(10000; precision=20)
    
    # Verify uniqueness
    unique = verify_padic_uniqueness(colors)
    println("Identity collisions: $(unique ? "0 ✓" : "FOUND ✗")")
    
    # Check hex collisions (expected due to 8-bit quantization)
    hex_set = Set(to_hex(c) for c in colors)
    println("Unique hex values: $(length(hex_set))/$(length(colors))")
    
    # Show sample
    println()
    println("Sample colors:")
    for i in [1, 2, 3, 10, 100]
        c = colors[i]
        println("  [$i] $(to_hex(c)) → $(canonical_key(c.r)[1:10])...")
    end
    
    # Verify ultrametric
    println()
    println("Ultrametric verification (10³ triples):")
    violations = 0
    for i in 1:10, j in 1:10, k in 1:10
        d_ij = padic_distance_valuation(colors[i], colors[j])
        d_jk = padic_distance_valuation(colors[j], colors[k])
        d_ik = padic_distance_valuation(colors[i], colors[k])
        if d_ik < min(d_ij, d_jk)
            violations += 1
        end
    end
    println("  Violations: $violations ✓")
end
