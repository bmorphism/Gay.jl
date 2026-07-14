# Induced presentation colors for one declared portable .cljc core realized by
# multiple Clojure-family runtimes. Exact descriptors remain authoritative;
# 64-bit seeds and 24-bit RGB values are presentation only.

const _CLJC_COLOR_VERSION = "clj1"
const _CLJC_CORE_ID_RE = r"^[0-9a-f]{64}\z"
const _CLJC_READER_FEATURE = Dict{Symbol,Symbol}(
    :jank => :jank,
    :basilisp => :lpy,
)
const _CLJC_REQUIRED_ROLES = (Int8(0), Int8(1), Int8(-1))

function _cljc_runtime(runtime)
    value = runtime isa Symbol ? runtime : Symbol(String(runtime))
    haskey(_CLJC_READER_FEATURE, value) ||
        throw(ArgumentError("runtime must be jank or basilisp"))
    value
end

function _cljc_core_id(value)
    text = String(value)
    occursin(_CLJC_CORE_ID_RE, text) ||
        throw(ArgumentError("core_id must be a lowercase 64-hex SHA-256 identifier"))
    text
end

function _cljc_digest(domain::AbstractString, fields::AbstractString...)
    for field in fields
        occursin('\0', field) && throw(ArgumentError("identity fields must not contain NUL"))
    end
    bytes2hex(SHA.sha256(join((domain, fields...), "\0")))
end

function _cljc_material_digest(material::AbstractVector{UInt8})
    io = IOBuffer()
    write(io, codeunits("Gay.jl/cljc-runtime/core/v1"))
    write(io, UInt8(0))
    write(io, material)
    bytes2hex(SHA.sha256(take!(io)))
end

"""
    cljc_core_id(material) -> String

Hash exact caller-supplied canonical contract or `.cljc` bytes into a full
SHA-256 identifier. No Unicode, whitespace, comment, or newline normalization
is performed: callers choose and version the canonicalization policy.
"""
cljc_core_id(material::AbstractString) = _cljc_material_digest(codeunits(material))
cljc_core_id(material::AbstractVector{UInt8}) = _cljc_material_digest(material)

"""One runtime fiber over a declared portable `.cljc` core."""
struct CljcRuntimeColor
    version::String
    core_id::String
    runtime::Symbol
    reader_feature::Symbol
    core_seed::UInt64
    runtime_seed::UInt64
    carrier_seed::UInt64
    core_color::String
    runtime_color::String
    carrier_color::String
end

_cljc_runtime_fields(record::CljcRuntimeColor) = (
    record.version,
    record.core_id,
    record.runtime,
    record.reader_feature,
    record.core_seed,
    record.runtime_seed,
    record.carrier_seed,
    record.core_color,
    record.runtime_color,
    record.carrier_color,
)

Base.:(==)(a::CljcRuntimeColor, b::CljcRuntimeColor) =
    _cljc_runtime_fields(a) == _cljc_runtime_fields(b)
Base.isequal(a::CljcRuntimeColor, b::CljcRuntimeColor) =
    isequal(_cljc_runtime_fields(a), _cljc_runtime_fields(b))
Base.hash(record::CljcRuntimeColor, h::UInt) = hash(_cljc_runtime_fields(record), h)

"""
    cljc_runtime_color(core_id, runtime) -> CljcRuntimeColor

Induce three disentangled presentation layers:

- `core_color` factors only through the portable core projection;
- `runtime_color` depends only on the runtime (`jank` or `basilisp`);
- `carrier_color` is induced from the product `(core_id, runtime)`.

The exact identity is `(version, core_id, runtime)`, never an RGB value.
"""
function cljc_runtime_color(core_id, runtime)
    core = _cljc_core_id(core_id)
    carrier = _cljc_runtime(runtime)
    feature = _CLJC_READER_FEATURE[carrier]
    core_seed = stable_seed(string("Gay.jl/cljc-runtime/core-color/v1\0", core))
    runtime_seed = stable_seed(string("Gay.jl/cljc-runtime/runtime-color/v1\0", carrier))
    carrier_seed = stable_seed(
        string("Gay.jl/cljc-runtime/carrier-color/v1\0", carrier);
        seed=core_seed,
    )
    CljcRuntimeColor(
        _CLJC_COLOR_VERSION,
        core,
        carrier,
        feature,
        core_seed,
        runtime_seed,
        carrier_seed,
        color_at(0; seed=core_seed),
        color_at(0; seed=runtime_seed),
        color_at(0; seed=carrier_seed),
    )
end

"""Authoritative exact descriptor for a runtime fiber; RGB is excluded."""
cljc_runtime_identity(record::CljcRuntimeColor) =
    (record.version, record.core_id, record.runtime)

"""Canonical, versioned reference for a valid declared `.cljc` runtime fiber."""
function cljc_runtime_uri(record::CljcRuntimeColor)
    verify_cljc_runtime_color(record) ||
        throw(ArgumentError("cannot serialize an invalid induced runtime color"))
    string("clojure://", record.runtime, "/cljc/", record.version,
           "/gay-sha256/", record.core_id)
end

"""Recompute every derived field from the exact descriptor."""
function verify_cljc_runtime_color(record::CljcRuntimeColor)
    try
        record == cljc_runtime_color(record.core_id, record.runtime)
    catch
        false
    end
end

"""A directed re-realization between two runtime fibers over one core."""
struct CljcRuntimeTransition
    version::String
    source::CljcRuntimeColor
    target::CljcRuntimeColor
    transition_id::String
    transition_seed::UInt64
    transition_color::String
    required_roles::NTuple{3,Int8}
end

_cljc_transition_fields(transition::CljcRuntimeTransition) = (
    transition.version,
    transition.source,
    transition.target,
    transition.transition_id,
    transition.transition_seed,
    transition.transition_color,
    transition.required_roles,
)

Base.:(==)(a::CljcRuntimeTransition, b::CljcRuntimeTransition) =
    _cljc_transition_fields(a) == _cljc_transition_fields(b)
Base.isequal(a::CljcRuntimeTransition, b::CljcRuntimeTransition) =
    isequal(_cljc_transition_fields(a), _cljc_transition_fields(b))
Base.hash(transition::CljcRuntimeTransition, h::UInt) =
    hash(_cljc_transition_fields(transition), h)

"""
    cljc_runtime_transition(source, target) -> CljcRuntimeTransition

Create a directed transition color only when both fibers project to the same
portable core. `required_roles` is the structural requirement `(0, +1, -1)`
for capture/witness, execution/play, and validation/coplay. It does not claim
that those activities occurred and is not behavioral-equivalence evidence.
"""
function cljc_runtime_transition(source::CljcRuntimeColor,
                                 target::CljcRuntimeColor)
    verify_cljc_runtime_color(source) ||
        throw(ArgumentError("source runtime color is not a valid induced record"))
    verify_cljc_runtime_color(target) ||
        throw(ArgumentError("target runtime color is not a valid induced record"))
    source.core_id == target.core_id ||
        throw(ArgumentError("runtime transition must preserve the portable core"))
    source.version == target.version ||
        throw(ArgumentError("runtime transition must preserve the descriptor version"))
    source.runtime != target.runtime ||
        throw(ArgumentError("source and target runtimes must differ"))
    transition_id = _cljc_digest(
        "Gay.jl/cljc-runtime/transition/v1",
        source.version,
        target.version,
        source.core_id,
        String(source.runtime),
        String(target.runtime),
    )
    transition_seed = stable_seed(
        string("Gay.jl/cljc-runtime/transition-color/v1\0", transition_id);
        seed=source.core_seed,
    )
    CljcRuntimeTransition(
        _CLJC_COLOR_VERSION,
        source,
        target,
        transition_id,
        transition_seed,
        color_at(0; seed=transition_seed),
        _CLJC_REQUIRED_ROLES,
    )
end

cljc_runtime_transition(core_id, source_runtime, target_runtime) =
    cljc_runtime_transition(
        cljc_runtime_color(core_id, source_runtime),
        cljc_runtime_color(core_id, target_runtime),
    )

"""
Recompute a transition and validate its non-degenerate GF(3) role requirement.
This checks structure only; it does not close or authenticate an evidence log.
"""
function verify_cljc_transition_structure(transition::CljcRuntimeTransition)
    transition.required_roles == _CLJC_REQUIRED_ROLES || return false
    mod(sum(Int, transition.required_roles), 3) == 0 || return false
    try
        transition == cljc_runtime_transition(transition.source, transition.target)
    catch
        false
    end
end
