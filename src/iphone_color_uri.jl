# iphone:// identifiers are private pairing references, not hardware identities.
# Raw device identifiers, account identifiers, recording metadata, and exact
# interaction timings are intentionally absent from this module's data model.

const _IPHONE_STATES = (:unavailable, :interrupted, :available, :connected)
const _IPHONE_URI_VERSION = "g1"
const _IPHONE_TOKEN_RE = r"^[0-9a-f]{32}\z"
const _IPHONE_URI_RE = r"^iphone://(g1)-([0-9a-f]{32})-([0-9a-f]{32})/([0-9a-f]{32})/([0-9a-f]{32})/([0-9a-f]{32})\z"
const _PASSPORT_IPHONE_URI_RE = r"^passport://gay/iphone/(g1)-([0-9a-f]{32})-([0-9a-f]{32})/([0-9a-f]{32})/([0-9a-f]{32})/([0-9a-f]{32})\z"

"""
    IPhoneProbe(state; voice_memos_sync=false, recording_count_bin=0,
                interaction_bin=0)

A deliberately coarse, consented observation made by a Mac (or another paired
observer). `state` is one of `:unavailable`, `:interrupted`, `:available`, or
`:connected`. Both bins are integers in `0:3`; exact recording counts, timings,
titles, audio, and device identifiers are not representable.
"""
struct IPhoneProbe
    state::Symbol
    voice_memos_sync::Bool
    recording_count_bin::UInt8
    interaction_bin::UInt8

    function IPhoneProbe(state::Symbol;
                         voice_memos_sync::Bool=false,
                         recording_count_bin::Integer=0,
                         interaction_bin::Integer=0)
        state in _IPHONE_STATES ||
            throw(ArgumentError("state must be one of $(_IPHONE_STATES)"))
        0 <= recording_count_bin <= 3 ||
            throw(ArgumentError("recording_count_bin must be in 0:3"))
        0 <= interaction_bin <= 3 ||
            throw(ArgumentError("interaction_bin must be in 0:3"))
        new(state, voice_memos_sync, UInt8(recording_count_bin), UInt8(interaction_bin))
    end
end

"""
    iphone_recording_count_bin(count) -> UInt8

Immediately coarsen a locally observed Voice Memos count: `0`, `1:4`, `5:16`,
and `17+` become bins `0:3`. The exact count is not retained by `IPhoneProbe`.
"""
function iphone_recording_count_bin(count::Integer)
    count >= 0 || throw(ArgumentError("recording count must be nonnegative"))
    UInt8(count == 0 ? 0 : count <= 4 ? 1 : count <= 16 ? 2 : 3)
end

"""
    IPhoneColorSpace(; version="coarse-probe-v1", weights=(1, 1, 1, 1))

An immutable diagonal metric over the four coarse probe coordinates. The
weights may be learned with `learn_iphone_color_space`; its digest becomes the
model component of each color identifier. The metric, rather than RGB distance,
is the matching authority.
"""
struct IPhoneColorSpace
    version::String
    weights::NTuple{4,Float64}

    function IPhoneColorSpace(version::AbstractString,
                              weights::NTuple{4,<:Real})
        v = Unicode.normalize(String(version); compose=true, stable=true)
        isempty(v) && throw(ArgumentError("model version must not be empty"))
        ncodeunits(v) <= 128 || throw(ArgumentError("model version is too long"))
        occursin('\0', v) && throw(ArgumentError("model version must not contain NUL"))
        raw = ntuple(i -> Float64(weights[i]), 4)
        all(isfinite, raw) || throw(ArgumentError("weights must be finite"))
        all(x -> x > 0, raw) ||
            throw(ArgumentError("every weight must be positive so distance remains a metric"))
        # Divide by the maximum before summing: sum(raw) can overflow even when
        # every input is finite. Reject dynamic ranges that underflow an axis to
        # zero, because that would silently turn the metric into a pseudometric.
        scale = maximum(raw)
        scaled = ntuple(i -> raw[i] / scale, 4)
        all(x -> isfinite(x) && x > 0, scaled) ||
            throw(ArgumentError("weight dynamic range is unsafe after normalization"))
        w = ntuple(i -> scaled[i] * 4 / sum(scaled), 4)
        all(x -> isfinite(x) && x > 0, w) ||
            throw(ArgumentError("normalized weights must remain finite and positive"))
        new(v, w)
    end
end

IPhoneColorSpace(; version::AbstractString="coarse-probe-v1",
                 weights::NTuple{4,<:Real}=(1.0, 1.0, 1.0, 1.0)) =
    IPhoneColorSpace(version, weights)

"""Embed a coarse probe in `[0,1]^4` without retaining exact source values."""
function iphone_probe_embedding(probe::IPhoneProbe)
    codes = _iphone_probe_codes(probe)
    ntuple(i -> codes[i] / 3, 4)
end

function _iphone_probe_codes(probe::IPhoneProbe)
    state = findfirst(==(probe.state), _IPHONE_STATES)::Int
    (state - 1,
     probe.voice_memos_sync ? 3 : 0,
     Int(probe.recording_count_bin),
     Int(probe.interaction_bin))
end

"""Weighted distance in the learnable coarse-probe space."""
function iphone_probe_distance(space::IPhoneColorSpace,
                               a::IPhoneProbe,
                               b::IPhoneProbe)
    x = iphone_probe_embedding(a)
    y = iphone_probe_embedding(b)
    sqrt(sum(space.weights[i] * (x[i] - y[i])^2 for i in 1:4) / sum(space.weights))
end

"""
    learn_iphone_color_space(examples; version="coarse-probe-learned-v1",
                             regularization=0.05)

Learn deterministic diagonal metric weights from `(a, b, is_match)` examples.
`is_match=true` denotes a pair whose motif should remain near; `false` denotes
a pair that should separate. At least one example of each class is required.
This updates only the matching/presentation model, never URI pair identity.
"""
function learn_iphone_color_space(examples;
                                  version::AbstractString="coarse-probe-learned-v1",
                                  regularization::Real=0.05)
    isfinite(regularization) && regularization > 0 ||
        throw(ArgumentError("regularization must be finite and positive"))
    # Codes are integers in 0:3, so squared deltas accumulate exactly. Convert
    # once after summation to make the model invariant under example ordering.
    within = zeros(BigInt, 4)
    between = zeros(BigInt, 4)
    nwithin = 0
    nbetween = 0

    for example in examples
        length(example) == 3 ||
            throw(ArgumentError("each example must be (IPhoneProbe, IPhoneProbe, Bool)"))
        a, b, is_match = example
        a isa IPhoneProbe && b isa IPhoneProbe && is_match isa Bool ||
            throw(ArgumentError("each example must be (IPhoneProbe, IPhoneProbe, Bool)"))
        x = _iphone_probe_codes(a)
        y = _iphone_probe_codes(b)
        target = is_match ? within : between
        for i in 1:4
            target[i] += (x[i] - y[i])^2
        end
        if is_match
            nwithin += 1
        else
            nbetween += 1
        end
    end

    nwithin > 0 || throw(ArgumentError("at least one matching example is required"))
    nbetween > 0 || throw(ArgumentError("at least one nonmatching example is required"))
    within_mean = [Float64(x) / (9nwithin) for x in within]
    between_mean = [Float64(x) / (9nbetween) for x in between]
    raw = clamp.((between_mean .+ regularization) ./
                 (within_mean .+ regularization), 0.1, 10.0)
    raw .*= 4 / sum(raw)
    IPhoneColorSpace(version, Tuple(raw))
end

function _iphone_text(label, value; max_bytes::Integer=256)
    s = Unicode.normalize(String(value); compose=true, stable=true)
    isempty(s) && throw(ArgumentError("$label must not be empty"))
    ncodeunits(s) <= max_bytes || throw(ArgumentError("$label is too long"))
    occursin('\0', s) && throw(ArgumentError("$label must not contain NUL"))
    s
end

_iphone_hex(bytes, count::Integer) = bytes2hex(@view bytes[1:count])

function _iphone_model_id(space::IPhoneColorSpace)
    weight_bits = join((@sprintf("%016x", reinterpret(UInt64, w)) for w in space.weights), ",")
    material = string("passport.gay/iphone/model/v1\0", space.version, "|", weight_bits)
    _iphone_hex(SHA.sha256(material), 16)
end

function _iphone_pair_key(pair_key)
    pair_key isa AbstractVector{UInt8} ||
        throw(ArgumentError("pair_key must be a 32-byte Vector{UInt8}"))
    length(pair_key) == 32 || throw(ArgumentError("pair_key must contain exactly 32 bytes"))
    Vector{UInt8}(pair_key)
end

"""Generate a fresh 256-bit enrollment key in memory; this function does not persist it."""
generate_iphone_pair_key() = rand(Random.RandomDevice(), UInt8, 32)

function _iphone_hmac_token(key::Vector{UInt8}, domain::AbstractString,
                            fields::AbstractString...; bytes::Integer=16)
    message = join((domain, fields...), "\0")
    _iphone_hex(SHA.hmac_sha256(key, message), bytes)
end

function _iphone_root_hsl(semantic_root)
    root = _iphone_text("semantic_root", semantic_root)
    seed = stable_seed(string("passport.gay/iphone/root/v1\0", root))
    rng = SplittableRandom(seed)
    (randf(rng) * 360.0,
     0.5 + randf(rng) * 0.4,
     0.35 + randf(rng) * 0.4)
end

"""Canonical Gay.jl root color for a local semantic alias such as `passport.gay`."""
function iphone_root_color(semantic_root="passport.gay")
    h, s, l = _iphone_root_hsl(semantic_root)
    rgb_hex(okhsl_to_rgb(h, s, l)...)
end

"""
    iphone_probe_color(probe; semantic_root="passport.gay", space=IPhoneColorSpace())

Produce a deterministic local presentation color around a canonical Gay.jl
root. A single RGB does not preserve the full four-dimensional neighborhood:
this color is not identity, authentication, or the metric used for matching.
"""
function iphone_probe_color(probe::IPhoneProbe;
                            semantic_root="passport.gay",
                            space::IPhoneColorSpace=IPhoneColorSpace())
    h0, s0, l0 = _iphone_root_hsl(semantic_root)
    x = iphone_probe_embedding(probe)
    w = ntuple(i -> space.weights[i] / sum(space.weights), 4)
    c = ntuple(i -> x[i] - 0.5, 4)

    h = h0 + 96 * (w[1] * c[1] + 0.55w[2] * c[2] - 0.35w[3] * c[3] + 0.2w[4] * c[4])
    s = clamp(s0 + 0.24 * (-w[1] * c[1] + w[2] * c[2] + 0.5w[3] * c[3] - 0.5w[4] * c[4]), 0.35, 0.95)
    l = clamp(l0 + 0.24 * (w[1] * c[1] - 0.5w[2] * c[2] + w[3] * c[3] + 0.5w[4] * c[4]), 0.25, 0.82)
    rgb_hex(okhsl_to_rgb(h, s, l)...)
end

"""Parsed, secret-free reference carried by an `iphone://` URI."""
struct IPhoneColorURI
    version::String
    model_id::String
    color_token::String
    scope_token::String
    epoch_token::String
    pair_tag::String

    function IPhoneColorURI(version::AbstractString,
                            model_id::AbstractString,
                            color_token::AbstractString,
                            scope_token::AbstractString,
                            epoch_token::AbstractString,
                            pair_tag::AbstractString)
        version == _IPHONE_URI_VERSION || throw(ArgumentError("unsupported iphone URI version"))
        occursin(_IPHONE_TOKEN_RE, model_id) || throw(ArgumentError("invalid model id"))
        occursin(_IPHONE_TOKEN_RE, color_token) || throw(ArgumentError("invalid color token"))
        occursin(_IPHONE_TOKEN_RE, scope_token) || throw(ArgumentError("invalid scope token"))
        occursin(_IPHONE_TOKEN_RE, epoch_token) || throw(ArgumentError("invalid epoch token"))
        occursin(_IPHONE_TOKEN_RE, pair_tag) || throw(ArgumentError("invalid pair tag"))
        new(String(version), String(model_id), String(color_token),
            String(scope_token), String(epoch_token), String(pair_tag))
    end
end

"""A validated local color observation plus its opaque, secret-free URI reference."""
struct IPhoneColorRecord
    ref::IPhoneColorURI
    root_color::String
    color::String
    embedding::NTuple{4,Float64}

    function IPhoneColorRecord(ref::IPhoneColorURI,
                               root_color::AbstractString,
                               color::AbstractString,
                               embedding::NTuple{4,<:Real})
        root = String(root_color)
        motif = String(color)
        occursin(r"^#[0-9A-F]{6}\z", root) ||
            throw(ArgumentError("root color must be canonical #RRGGBB"))
        occursin(r"^#[0-9A-F]{6}\z", motif) ||
            throw(ArgumentError("motif color must be canonical #RRGGBB"))
        point = ntuple(i -> Float64(embedding[i]), 4)
        allowed = (0.0, 1 / 3, 2 / 3, 1.0)
        all(x -> isfinite(x) && x in allowed, point) ||
            throw(ArgumentError("embedding must lie on the coarse 0,1/3,2/3,1 lattice"))
        point[2] in (0.0, 1.0) ||
            throw(ArgumentError("Voice Memos sync coordinate must be 0 or 1"))
        new(ref, root, motif, point)
    end
end

"""
An authorized in-memory resolver for opaque `iphone://` references. It stores
no pair keys and performs no persistence or network exchange; those enrollment
lifecycle concerns remain outside Gay.jl. The mutable registry is intended to
be confined to one tile/vat rather than shared unsafely across threads.
"""
mutable struct IPhoneColorRegistry
    records::Dict{String,IPhoneColorRecord}
    models::Dict{String,IPhoneColorSpace}
    owner::Task
end

IPhoneColorRegistry() = IPhoneColorRegistry(
    Dict{String,IPhoneColorRecord}(),
    Dict{String,IPhoneColorSpace}(),
    current_task(),
)

function _check_iphone_vat(registry::IPhoneColorRegistry)
    current_task() === registry.owner ||
        throw(ArgumentError("iphone registry access crossed its owning tile/vat task"))
    nothing
end

iphone_color_identifier(ref::IPhoneColorURI) =
    string(ref.version, "-", ref.model_id, "-", ref.color_token)
iphone_color_identifier(record::IPhoneColorRecord) = iphone_color_identifier(record.ref)

iphone_uri(ref::IPhoneColorURI) =
    string("iphone://", iphone_color_identifier(ref), "/", ref.scope_token, "/",
           ref.epoch_token, "/", ref.pair_tag)
iphone_uri(record::IPhoneColorRecord) = iphone_uri(record.ref)

passport_uri(ref::IPhoneColorURI) =
    string("passport://gay/iphone/", iphone_color_identifier(ref), "/", ref.scope_token,
           "/", ref.epoch_token, "/", ref.pair_tag)
passport_uri(record::IPhoneColorRecord) = passport_uri(record.ref)

function _parse_iphone_match(m)
    m === nothing && throw(ArgumentError("malformed or noncanonical iphone color URI"))
    IPhoneColorURI(m.captures...)
end

"""Strictly parse the canonical `iphone://<color-identifier>/...` form."""
parse_iphone_uri(uri::AbstractString) = _parse_iphone_match(match(_IPHONE_URI_RE, uri))

"""Strictly parse the equivalent `passport://gay/iphone/...` form."""
parse_passport_uri(uri::AbstractString) =
    _parse_iphone_match(match(_PASSPORT_IPHONE_URI_RE, uri))

"""
Verify and register a local record plus its frozen metric model for resolution.
The enrollment key is checked but never retained by the registry.
"""
function register_iphone_color!(registry::IPhoneColorRegistry,
                                record::IPhoneColorRecord,
                                probe::IPhoneProbe;
                                pair_key,
                                scope,
                                epoch,
                                semantic_root="passport.gay",
                                space::IPhoneColorSpace=IPhoneColorSpace())
    _check_iphone_vat(registry)
    verify_iphone_color_record(record, probe; pair_key=pair_key, scope=scope,
                               epoch=epoch, semantic_root=semantic_root, space=space) ||
        throw(ArgumentError("record failed keyed enrollment verification"))
    model_id = record.ref.model_id
    if haskey(registry.models, model_id)
        known = registry.models[model_id]
        (known.version == space.version && known.weights == space.weights) ||
            throw(ArgumentError("model id collision maps to conflicting metric models"))
    end
    key = iphone_uri(record)
    if haskey(registry.records, key)
        existing = registry.records[key]
        (existing.root_color == record.root_color &&
         existing.color == record.color &&
         existing.embedding == record.embedding) ||
            throw(ArgumentError("iphone URI collision maps to conflicting local records"))
    end
    registry.records[key] = record
    registry.models[model_id] = space
    record
end

"""
Resolve an opaque canonical iPhone or passport alias through an authorized
local registry. Returns `nothing` when the reference is well-formed but absent.
"""
function resolve_iphone_color(registry::IPhoneColorRegistry, uri::AbstractString)
    _check_iphone_vat(registry)
    ref = _parse_iphone_alias(uri)
    get(registry.records, iphone_uri(ref), nothing)
end

_parse_iphone_alias(uri::AbstractString) =
    startswith(uri, "iphone://") ? parse_iphone_uri(uri) : parse_passport_uri(uri)

"""
Compare two registered URI motifs in their frozen learned metric. Both records
must resolve and carry the same model id; RGB distance is never used.
"""
function iphone_record_distance(registry::IPhoneColorRegistry,
                                left_uri::AbstractString,
                                right_uri::AbstractString)
    left = resolve_iphone_color(registry, left_uri)
    right = resolve_iphone_color(registry, right_uri)
    left === nothing && throw(ArgumentError("left iphone URI is not registered"))
    right === nothing && throw(ArgumentError("right iphone URI is not registered"))
    left.ref.model_id == right.ref.model_id ||
        throw(ArgumentError("cannot compare records from different metric models"))
    space = get(registry.models, left.ref.model_id, nothing)
    space === nothing && throw(ArgumentError("metric model is not registered"))
    x = left.embedding
    y = right.embedding
    sqrt(sum(space.weights[i] * (x[i] - y[i])^2 for i in 1:4) / sum(space.weights))
end

function _prune_iphone_model!(registry::IPhoneColorRegistry, model_id::AbstractString)
    any(record.ref.model_id == model_id for record in values(registry.records)) ||
        delete!(registry.models, String(model_id))
    nothing
end

"""Revoke one canonical iPhone or passport reference from a vat-local registry."""
function unregister_iphone_color!(registry::IPhoneColorRegistry, uri::AbstractString)
    _check_iphone_vat(registry)
    ref = _parse_iphone_alias(uri)
    removed = pop!(registry.records, iphone_uri(ref), nothing)
    removed === nothing || _prune_iphone_model!(registry, removed.ref.model_id)
    removed
end

"""Revoke every vat-local record carrying one opaque epoch token; return the count."""
function purge_iphone_epoch!(registry::IPhoneColorRegistry, epoch_token::AbstractString)
    _check_iphone_vat(registry)
    occursin(_IPHONE_TOKEN_RE, epoch_token) || throw(ArgumentError("invalid epoch token"))
    doomed = [key for (key, record) in registry.records
              if record.ref.epoch_token == epoch_token]
    model_ids = String[]
    for key in doomed
        push!(model_ids, registry.records[key].ref.model_id)
        delete!(registry.records, key)
    end
    for model_id in unique(model_ids)
        _prune_iphone_model!(registry, model_id)
    end
    length(doomed)
end

"""
    iphone_color_record(probe; pair_key, scope, epoch,
                        semantic_root="passport.gay", space=IPhoneColorSpace())

Create an `iphone://<color-identifier>` record. `pair_key` must be a fresh
32-byte key shared only by this enrollment. Pair identity depends on key,
scope, and epoch—not on model version or color—so learning can advance without
identity drift. The key is neither stored nor returned.
"""
function iphone_color_record(probe::IPhoneProbe;
                             pair_key,
                             scope,
                             epoch,
                             semantic_root="passport.gay",
                             space::IPhoneColorSpace=IPhoneColorSpace())
    key = _iphone_pair_key(pair_key)
    scope_text = _iphone_text("scope", scope; max_bytes=128)
    epoch_text = _iphone_text("epoch", epoch; max_bytes=128)
    root_text = _iphone_text("semantic_root", semantic_root)
    model_id = _iphone_model_id(space)
    root_color = iphone_root_color(root_text)
    color = iphone_probe_color(probe; semantic_root=root_text, space=space)
    embedding = iphone_probe_embedding(probe)

    scope_token = _iphone_hmac_token(key, "passport.gay/iphone/scope-token/v1",
                                     scope_text, epoch_text; bytes=16)
    epoch_token = _iphone_hmac_token(key, "passport.gay/iphone/epoch-token/v1",
                                     scope_text, epoch_text; bytes=16)
    pair_tag = _iphone_hmac_token(key, "passport.gay/iphone/pair-tag/v1",
                                  scope_text, epoch_text)
    color_token = _iphone_hmac_token(key, "passport.gay/iphone/color-token/v1",
                                     scope_text, epoch_text, model_id, root_text, color,
                                     join((string(round(Int, 255x)) for x in embedding), ","))

    ref = IPhoneColorURI(_IPHONE_URI_VERSION, model_id, color_token,
                         scope_token, epoch_token, pair_tag)
    IPhoneColorRecord(ref, root_color, color, embedding)
end

function _iphone_accumulator_equal(a::AbstractString, b::AbstractString)
    ac = codeunits(a)
    bc = codeunits(b)
    length(ac) == length(bc) || return false
    difference = UInt8(0)
    for i in eachindex(ac, bc)
        difference |= xor(ac[i], bc[i])
    end
    iszero(difference)
end

"""
Recompute and XOR-accumulator compare the keyed URI tokens for a local record.
This avoids content-dependent early return for equal-length tokens, but does
not claim end-to-end constant-time execution by the Julia compiler/runtime.
"""
function verify_iphone_color_record(record::IPhoneColorRecord,
                                    probe::IPhoneProbe;
                                    pair_key,
                                    scope,
                                    epoch,
                                    semantic_root="passport.gay",
                                    space::IPhoneColorSpace=IPhoneColorSpace())
    expected = iphone_color_record(probe; pair_key=pair_key, scope=scope,
                                   epoch=epoch, semantic_root=semantic_root, space=space)
    _iphone_accumulator_equal(iphone_uri(record), iphone_uri(expected)) &&
        record.root_color == expected.root_color &&
        record.color == expected.color &&
        record.embedding == expected.embedding
end
