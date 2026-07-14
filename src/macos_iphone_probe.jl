# Read-only bridge from coarse macOS Accessibility observations to IPhoneProbe.
# The Swift producer emits no device label, identifier, recording title, audio,
# timestamp, exact count, or raw accessibility text.

const _MACOS_IPHONE_PROBE_SCHEMA = "gay-iphone-probe/v1"
const _MACOS_EVIDENCE_RE = r"^[a-z0-9-]+\z"
const _MACOS_CONNECTION_CONTRACT = Dict{Symbol,Union{Nothing,Tuple{Symbol,UInt8}}}(
    Symbol("ax-connection-paused") => (:available, 0x02),
    Symbol("ax-connection-interrupted") => (:interrupted, 0x01),
    Symbol("ax-local-auth-required") => (:interrupted, 0x01),
    Symbol("ax-remote-control-gated") => (:interrupted, 0x01),
    Symbol("ax-iphone-unavailable") => (:unavailable, 0x00),
    Symbol("ax-connecting") => (:available, 0x01),
    Symbol("coredevice-connected") => (:connected, 0x03),
    Symbol("coredevice-available") => (:available, 0x02),
    Symbol("coredevice-unavailable") => (:unavailable, 0x00),
    Symbol("coredevice-unpaired") => (:unavailable, 0x00),
    Symbol("ax-status-unknown") => nothing,
    Symbol("ax-window-unavailable") => nothing,
    Symbol("coredevice-none") => nothing,
    Symbol("coredevice-ambiguous") => nothing,
    Symbol("coredevice-status-unknown") => nothing,
    Symbol("coredevice-probe-unavailable") => nothing,
)
const _MACOS_SYNC_EVIDENCE = Set((
    Symbol("ax-icloud-voice-memos-toggle"),
    Symbol("ax-toggle-unavailable"),
))
const _MACOS_RECORDINGS_EVIDENCE = Set((
    Symbol("ax-selected-all-recordings"),
    Symbol("ax-all-recordings-unavailable"),
    Symbol("ax-all-recordings-not-selected"),
    Symbol("ax-window-unavailable"),
))

"""
    MacOSIPhoneObservation

Possibly partial, privacy-coarsened observation from already-open macOS apps.
Use `macos_iphone_observation_complete` before `materialize_iphone_probe`.
Evidence symbols name the narrow Accessibility surface used; they contain no
raw labels or device/recording identifiers.
"""
struct MacOSIPhoneObservation
    state::Union{Nothing,Symbol}
    voice_memos_sync::Union{Nothing,Bool}
    recording_count_bin::Union{Nothing,UInt8}
    interaction_bin::Union{Nothing,UInt8}
    connection_evidence::Symbol
    sync_evidence::Symbol
    recordings_evidence::Symbol

    function MacOSIPhoneObservation(state::Union{Nothing,Symbol},
                                    voice_memos_sync::Union{Nothing,Bool},
                                    recording_count_bin::Union{Nothing,Integer},
                                    interaction_bin::Union{Nothing,Integer},
                                    connection_evidence::Symbol,
                                    sync_evidence::Symbol,
                                    recordings_evidence::Symbol)
        state === nothing || state in _IPHONE_STATES ||
            throw(ArgumentError("invalid coarse iPhone state"))
        recording_count_bin === nothing || 0 <= recording_count_bin <= 3 ||
            throw(ArgumentError("recording count bin must be in 0:3"))
        interaction_bin === nothing || 0 <= interaction_bin <= 3 ||
            throw(ArgumentError("interaction bin must be in 0:3"))
        for evidence in (connection_evidence, sync_evidence, recordings_evidence)
            occursin(_MACOS_EVIDENCE_RE, String(evidence)) ||
                throw(ArgumentError("invalid evidence token"))
        end
        count_bin = recording_count_bin === nothing ? nothing : UInt8(recording_count_bin)
        motif_bin = interaction_bin === nothing ? nothing : UInt8(interaction_bin)
        haskey(_MACOS_CONNECTION_CONTRACT, connection_evidence) ||
            throw(ArgumentError("unknown connection evidence token"))
        expected = _MACOS_CONNECTION_CONTRACT[connection_evidence]
        if expected === nothing
            state === nothing && motif_bin === nothing ||
                throw(ArgumentError("unknown connection evidence must keep state and interaction unknown"))
        else
            (state, motif_bin) == expected ||
                throw(ArgumentError("connection evidence conflicts with state or interaction bin"))
        end
        sync_evidence in _MACOS_SYNC_EVIDENCE ||
            throw(ArgumentError("unknown sync evidence token"))
        if sync_evidence == Symbol("ax-icloud-voice-memos-toggle")
            voice_memos_sync !== nothing ||
                throw(ArgumentError("visible sync toggle requires an observed Boolean"))
        else
            voice_memos_sync === nothing ||
                throw(ArgumentError("unavailable sync evidence must remain unknown"))
        end
        recordings_evidence in _MACOS_RECORDINGS_EVIDENCE ||
            throw(ArgumentError("unknown recordings evidence token"))
        if recordings_evidence == Symbol("ax-selected-all-recordings")
            count_bin !== nothing ||
                throw(ArgumentError("All Recordings evidence requires an observed bin"))
        else
            count_bin === nothing ||
                throw(ArgumentError("unavailable recordings evidence must remain unknown"))
        end
        new(state, voice_memos_sync, count_bin, motif_bin,
            connection_evidence, sync_evidence, recordings_evidence)
    end
end

"""True only when all four coarse coordinates were directly observed."""
macos_iphone_observation_complete(observation::MacOSIPhoneObservation) =
    observation.state !== nothing &&
    observation.voice_memos_sync !== nothing &&
    observation.recording_count_bin !== nothing &&
    observation.interaction_bin !== nothing

"""Materialize a complete macOS observation as the core `IPhoneProbe`."""
function materialize_iphone_probe(observation::MacOSIPhoneObservation)
    macos_iphone_observation_complete(observation) ||
        throw(ArgumentError("macOS iPhone observation is partial; refusing to guess"))
    IPhoneProbe(observation.state::Symbol;
        voice_memos_sync=observation.voice_memos_sync::Bool,
        recording_count_bin=observation.recording_count_bin::UInt8,
        interaction_bin=observation.interaction_bin::UInt8)
end

function _macos_optional_state(value::AbstractString)
    value == "-" && return nothing
    value in String.(_IPHONE_STATES) || throw(ArgumentError("invalid macOS probe state"))
    Symbol(value)
end

function _macos_optional_bool(value::AbstractString)
    value == "-" && return nothing
    value == "1" && return true
    value == "0" && return false
    throw(ArgumentError("invalid macOS probe Boolean"))
end

function _macos_optional_bin(value::AbstractString)
    value == "-" && return nothing
    bin = tryparse(Int, value)
    bin === nothing && throw(ArgumentError("invalid macOS probe bin"))
    0 <= bin <= 3 || throw(ArgumentError("macOS probe bin must be in 0:3"))
    bin
end

function _macos_evidence(value::AbstractString)
    ncodeunits(value) <= 64 || throw(ArgumentError("macOS evidence token is too long"))
    occursin(_MACOS_EVIDENCE_RE, value) ||
        throw(ArgumentError("invalid macOS evidence token"))
    Symbol(value)
end

function _parse_macos_iphone_probe_tsv(output::AbstractString)
    ncodeunits(output) <= 1024 || throw(ArgumentError("macOS probe output is too large"))
    occursin(r"^[^\r\n]+\r?\n?\z", output) ||
        throw(ArgumentError("macOS probe must emit exactly one nonblank line"))
    fields = split(chomp(output), '\t'; keepempty=true)
    length(fields) == 8 || throw(ArgumentError("macOS probe emitted the wrong field count"))
    fields[1] == _MACOS_IPHONE_PROBE_SCHEMA ||
        throw(ArgumentError("unsupported macOS probe schema"))
    MacOSIPhoneObservation(
        _macos_optional_state(fields[2]),
        _macos_optional_bool(fields[3]),
        _macos_optional_bin(fields[4]),
        _macos_optional_bin(fields[5]),
        _macos_evidence(fields[6]),
        _macos_evidence(fields[7]),
        _macos_evidence(fields[8]),
    )
end

"""
    macos_iphone_observation(; script, swift) -> MacOSIPhoneObservation

Run the bundled read-only Accessibility/CoreDevice probe. It neither launches nor
activates apps: iPhone Mirroring, Voice Memos, and the relevant System Settings
page must already be open for complete evidence. Failures return no raw UI text.
"""
function macos_iphone_observation(;
        script::AbstractString=normpath(joinpath(@__DIR__, "..", "scripts",
                                                "macos_iphone_probe.swift")),
        swift=(Sys.isapple() && isfile("/usr/bin/swift") ?
               "/usr/bin/swift" : Sys.which("swift")))
    Sys.isapple() || throw(ArgumentError("macOS iPhone observation requires macOS"))
    swift === nothing && throw(ArgumentError("Swift runtime was not found"))
    isfile(script) || throw(ArgumentError("macOS probe script was not found"))

    stdout = Pipe()
    command = Cmd([String(swift), String(script), "--format", "tsv"])
    process = run(pipeline(ignorestatus(command), stdout=stdout, stderr=devnull); wait=false)
    close(stdout.in)
    bytes = read(stdout, 1025)
    oversized = length(bytes) > 1024
    oversized && kill(process)
    wait(process)
    oversized && throw(ArgumentError("macOS probe output exceeded 1024 bytes"))
    success(process) || throw(ErrorException("macOS iPhone probe failed"))
    _parse_macos_iphone_probe_tsv(String(bytes))
end
