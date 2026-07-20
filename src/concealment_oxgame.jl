# Concealment O/X Open Game
#
# O tries to bury ephemeral display ideology into names, flags, exports, prose,
# and generated interface surfaces. X searches SCIP-style derivable surfaces
# under a finite survival budget. The result is a persistent world, not a
# throwaway report.

const OX_FORBIDDEN_PARTS = ("de", "mo")
const OX_ALIAS_TOKENS = (
    "showcase",
    "sample",
    "toy",
    "preview",
    "walkthrough",
    "quickstart",
    "playground",
    "scratch",
    "throwaway",
    "one-shot",
    "transient",
)
const OX_SCAN_EXTENSIONS = Set([".jl", ".md", ".json", ".toml"])

struct OxSurface
    uri::String
    kind::Symbol
    bytes::Int
    lines::Int
    surface_fingerprint::UInt64
end

struct OxFinding
    uri::String
    line::Int
    detector::Symbol
    severity::Int
    pressure_cost::Int
    evidence::String
    derivation_uri::String
    color_hex::String
end

struct OxMove
    player::Symbol
    action::Symbol
    uri::String
    payoff_delta::Int
    color_hex::String
end

struct OxRound
    turn::Int
    concealer::OxMove
    finder::OxMove
    findings::Vector{OxFinding}
    pressure_after::Int
end

struct ConcealmentOxGameWorld
    seed::UInt64
    root::String
    pressure_budget::Int
    pressure_spent::Int
    surfaces::Vector{OxSurface}
    rounds::Vector{OxRound}
    findings::Vector{OxFinding}
    o_payoff::Int
    x_payoff::Int
    attack_catalog::Vector{Pair{Symbol,String}}
    detector_catalog::Vector{Pair{Symbol,String}}
    fingerprint::UInt64
end

struct OxScreenRow
    turn::Int
    o_action::Symbol
    x_detector::Symbol
    severity::Int
    pressure_after::Int
    color_hex::String
    uri::String
    evidence::String
end

struct OxScreen
    title::String
    width::Int
    pressure_bar::String
    o_color::String
    x_color::String
    summary::Vector{Pair{String,String}}
    rows::Vector{OxScreenRow}
    source_fingerprint::UInt64
    fingerprint::UInt64
end

struct OxgameRemixSource
    uri::String
    local_path::String
    score::Int
    color_hex::String
    excerpt::String
end

struct OxgameRemixLane
    uri::String
    trit::Int
    role::String
    color_hex::String
end

struct OxgameRemixWorld
    seed::UInt64
    gay_head::String
    oxgame_status::String
    docs_root::String
    concealment_world::ConcealmentOxGameWorld
    screen::OxScreen
    sources::Vector{OxgameRemixSource}
    lanes::Vector{OxgameRemixLane}
    commitments::Vector{Pair{String,String}}
    fingerprint::UInt64
end

struct OxArenaPlayer
    id::Symbol
    trit::Int
    role::String
    objective::String
    color_hex::String
end

struct OxOpticInterface
    uri::String
    optic_kind::Symbol
    players::Vector{Symbol}
    play::String
    coplay::String
    payoff::String
    color_hex::String
end

struct OxBisimulationWitness
    uri::String
    left_state::String
    right_state::String
    relation::String
    spoiler::Symbol
    duplicator::Symbol
    preserves::Vector{String}
    color_hex::String
end

struct SharedOxArena
    seed::UInt64
    arena_uri::String
    players::Vector{OxArenaPlayer}
    interfaces::Vector{OxOpticInterface}
    witnesses::Vector{OxBisimulationWitness}
    compositions::Vector{Pair{String,String}}
    source_fingerprint::UInt64
    fingerprint::UInt64
end

struct ScipEndpoint
    uri::String
    kind::Symbol
    local_path::String
    status::String
    evidence_count::Int
    color_hex::String
end

struct ColoredScipMorphism
    uri::String
    source_uri::String
    target_uri::String
    kind::Symbol
    trit::Int
    status::Symbol
    color_hex::String
    evidence::String
    limitation::String
end

struct ScipOxcamlBridgeWorld
    seed::UInt64
    scip_ocaml_head::String
    endpoints::Vector{ScipEndpoint}
    morphisms::Vector{ColoredScipMorphism}
    commitments::Vector{Pair{String,String}}
    fingerprint::UInt64
end

Base.length(w::ConcealmentOxGameWorld) = length(w.findings)
fingerprint(w::ConcealmentOxGameWorld)::UInt64 = w.fingerprint
Base.length(s::OxScreen) = length(s.rows)
fingerprint(s::OxScreen)::UInt64 = s.fingerprint
Base.length(w::OxgameRemixWorld) = length(w.sources) + length(w.concealment_world)
fingerprint(w::OxgameRemixWorld)::UInt64 = w.fingerprint
Base.length(a::SharedOxArena) = length(a.interfaces) + length(a.witnesses)
fingerprint(a::SharedOxArena)::UInt64 = a.fingerprint
Base.length(w::ScipOxcamlBridgeWorld) = length(w.endpoints) + length(w.morphisms)
fingerprint(w::ScipOxcamlBridgeWorld)::UInt64 = w.fingerprint

forbidden_display_root() = string(OX_FORBIDDEN_PARTS...)

function concealment_color(seed::UInt64, index::Integer)::String
    c = color_at(Int(index); seed=seed)
    rgb_hex(c.r, c.g, c.b)
end

function _ox_kind(path::AbstractString)::Symbol
    ext = lowercase(splitext(path)[2])
    if ext == ".jl"
        return :source
    elseif ext == ".md"
        return :doc
    elseif ext == ".json"
        return :artifact
    elseif ext == ".toml"
        return :manifest
    else
        return :unknown
    end
end

function _ox_scip_uri(root::AbstractString, path::AbstractString)::String
    rel = replace(relpath(path, root), '\\' => '/')
    "scip://gay/" * rel
end

function _ox_candidate_paths(root::AbstractString; max_files::Int=512, include_docs::Bool=true)
    scan_dirs = include_docs ? ["src", "scripts", "test", "ext", "examples", "docs", "artifacts"] :
                               ["src", "scripts", "test", "ext"]
    paths = String[]
    for dir in scan_dirs
        base = joinpath(root, dir)
        isdir(base) || continue
        for (walk_root, dirs, files) in walkdir(base)
            filter!(d -> !(startswith(d, ".") || d in ("build", "dist", ".git")), dirs)
            for file in sort(files)
                ext = lowercase(splitext(file)[2])
                ext in OX_SCAN_EXTENSIONS || continue
                push!(paths, joinpath(walk_root, file))
                length(paths) >= max_files && return paths
            end
        end
    end
    paths
end

function _ox_surface(root::AbstractString, path::AbstractString, text::String)::OxSurface
    uri = _ox_scip_uri(root, path)
    nlines = isempty(text) ? 0 : count(==('\n'), text) + 1
    fp = stable_seed((uri, sizeof(text), nlines); seed=UInt64(0x0f0f7877616d65))
    OxSurface(uri, _ox_kind(path), sizeof(text), nlines, fp)
end

function _ox_evidence(line::AbstractString)::String
    stripped = strip(replace(line, '\t' => ' '))
    length(stripped) <= 160 && return String(stripped)

    stop = firstindex(stripped)
    for _ in 1:160
        stop = nextind(stripped, stop)
        stop > lastindex(stripped) && return String(stripped)
    end
    String(stripped[firstindex(stripped):prevind(stripped, stop)])
end

function _ox_detectors(line::AbstractString)
    root = forbidden_display_root()
    lower = lowercase(String(line))
    normalized = replace(lower, r"[^a-z0-9]" => "")
    detectors = Pair{Symbol,Int}[]

    if occursin(Regex("\\b" * root * "_\\w+\\b"), lower)
        push!(detectors, :identifier_prefix => 10)
    end
    if occursin(Regex("\\b\\w+_" * root * "\\b"), lower)
        push!(detectors, :identifier_suffix => 10)
    end
    if occursin("--" * root, lower)
        push!(detectors, :interface_flag => 9)
    end
    if occursin("export", lower) && occursin(root, lower)
        push!(detectors, :export_surface => 9)
    end
    if occursin(root, lower)
        push!(detectors, :literal_surface => 8)
    elseif occursin(root, normalized)
        push!(detectors, :normalized_collision => 6)
    end
    if any(token -> occursin(token, lower), OX_ALIAS_TOKENS)
        push!(detectors, :semantic_alias => 4)
    end
    if occursin("scip://", lower) || occursin("app://", lower) || occursin("uri", lower)
        push!(detectors, :derivable_interface => 3)
    end

    detectors
end

function _ox_action_for(detector::Symbol)::Symbol
    detector === :identifier_prefix && return :bury_as_prefix
    detector === :identifier_suffix && return :bury_as_suffix
    detector === :interface_flag && return :hide_in_flag
    detector === :export_surface && return :hide_in_export
    detector === :literal_surface && return :leave_visible_trace
    detector === :normalized_collision && return :split_token_across_surface
    detector === :semantic_alias && return :rename_as_alias
    detector === :derivable_interface && return :hide_in_derivable_uri
    :unknown_concealment
end

function _ox_finding(
    seed::UInt64,
    uri::String,
    line_no::Int,
    detector::Symbol,
    severity::Int,
    evidence::String,
)::OxFinding
    pressure_cost = max(1, 12 - severity)
    color = concealment_color(seed, stable_seed((uri, line_no, detector); seed=seed) % UInt64(4096) + UInt64(1))
    derivation_uri = string(uri, "#L", line_no, "::", detector)
    OxFinding(uri, line_no, detector, severity, pressure_cost, evidence, derivation_uri, color)
end

function _ox_scan_text(seed::UInt64, uri::String, text::String, pressure_left::Int)
    findings = OxFinding[]
    remaining = pressure_left
    for (line_no, line) in enumerate(Base.split(text, '\n'; keepempty=true))
        isempty(strip(line)) && continue
        for (detector, severity) in _ox_detectors(line)
            pressure_cost = max(1, 12 - severity)
            pressure_cost <= remaining || return findings, remaining
            finding = _ox_finding(seed, uri, line_no, detector, severity, _ox_evidence(line))
            push!(findings, finding)
            remaining -= finding.pressure_cost
        end
    end
    findings, remaining
end

function _ox_catalogs()
    attacks = Pair{Symbol,String}[
        :bury_as_prefix => "O prefixes throwaway display intent into a callable name.",
        :bury_as_suffix => "O suffixes throwaway display intent where older checks may miss it.",
        :hide_in_flag => "O exposes the ideology as a CLI switch or interface knob.",
        :hide_in_export => "O makes the interface public and normalizes it through exports.",
        :split_token_across_surface => "O separates characters across punctuation or casing.",
        :rename_as_alias => "O replaces the forbidden root with semantic aliases.",
        :hide_in_derivable_uri => "O moves the signal into URIs, generated docs, or SCIP surfaces.",
    ]
    detectors = Pair{Symbol,String}[
        :identifier_prefix => "X checks callable prefixes.",
        :identifier_suffix => "X checks callable suffixes.",
        :interface_flag => "X checks CLI and UI flags.",
        :export_surface => "X checks public export surfaces.",
        :literal_surface => "X checks exact visible traces.",
        :normalized_collision => "X checks punctuation-insensitive token collisions.",
        :semantic_alias => "X checks aliases for throwaway display ideology.",
        :derivable_interface => "X checks URI and SCIP-like derivable interfaces.",
    ]
    attacks, detectors
end

function _ox_world_fingerprint(
    seed::UInt64,
    surfaces::Vector{OxSurface},
    findings::Vector{OxFinding},
    pressure_budget::Int,
)::UInt64
    fp = stable_seed(("concealment_oxgame", pressure_budget); seed=seed)
    for surface in sort(surfaces; by=s -> s.uri)
        fp = xor(fp, stable_seed((surface.uri, surface.kind, surface.surface_fingerprint); seed=seed))
    end
    for finding in sort(findings; by=f -> (f.uri, f.line, String(f.detector), f.evidence))
        fp = xor(fp, stable_seed((finding.uri, finding.line, finding.detector, finding.evidence); seed=seed))
    end
    splitmix64(fp)
end

function _ox_rounds(seed::UInt64, findings::Vector{OxFinding}, pressure_budget::Int)
    rounds = OxRound[]
    pressure_after = pressure_budget
    for (turn, finding) in enumerate(findings)
        pressure_after -= finding.pressure_cost
        action = _ox_action_for(finding.detector)
        o_color = concealment_color(seed, turn * 2)
        x_color = finding.color_hex
        concealer = OxMove(:O, action, finding.uri, max(0, 10 - finding.severity), o_color)
        finder = OxMove(:X, finding.detector, finding.derivation_uri, finding.severity, x_color)
        push!(rounds, OxRound(turn, concealer, finder, OxFinding[finding], pressure_after))
    end
    rounds
end

function _ox_make_world(
    seed::UInt64,
    root::String,
    pressure_budget::Int,
    surfaces::Vector{OxSurface},
    findings::Vector{OxFinding},
)::ConcealmentOxGameWorld
    rounds = _ox_rounds(seed, findings, pressure_budget)
    pressure_spent = sum(f.pressure_cost for f in findings)
    x_payoff = sum(f.severity for f in findings)
    o_payoff = max(0, pressure_budget - pressure_spent) + sum(max(0, 10 - f.severity) for f in findings)
    attacks, detectors = _ox_catalogs()
    fp = _ox_world_fingerprint(seed, surfaces, findings, pressure_budget)
    ConcealmentOxGameWorld(
        seed,
        root,
        pressure_budget,
        pressure_spent,
        surfaces,
        rounds,
        findings,
        o_payoff,
        x_payoff,
        attacks,
        detectors,
        fp,
    )
end

function world_concealment_oxgame(
    paths::Vector{<:AbstractString};
    root::AbstractString=normpath(joinpath(@__DIR__, "..")),
    seed::UInt64=GAY_SEED,
    pressure_budget::Int=144,
    max_file_bytes::Int=240_000,
)::ConcealmentOxGameWorld
    surfaces = OxSurface[]
    findings = OxFinding[]
    pressure_left = pressure_budget
    root_s = String(root)

    for path in paths
        isfile(path) || continue
        filesize(path) <= max_file_bytes || continue
        text = try
            read(path, String)
        catch
            continue
        end
        surface = _ox_surface(root_s, path, text)
        push!(surfaces, surface)
        uri = surface.uri
        found, pressure_left = _ox_scan_text(seed, uri, text, pressure_left)
        append!(findings, found)
        pressure_left <= 0 && break
    end

    _ox_make_world(seed, root_s, pressure_budget, surfaces, findings)
end

function world_concealment_oxgame(;
    root::AbstractString=normpath(joinpath(@__DIR__, "..")),
    seed::UInt64=GAY_SEED,
    pressure_budget::Int=144,
    max_files::Int=512,
    include_docs::Bool=true,
)::ConcealmentOxGameWorld
    paths = _ox_candidate_paths(root; max_files=max_files, include_docs=include_docs)
    world_concealment_oxgame(paths; root=root, seed=seed, pressure_budget=pressure_budget)
end

function Base.merge(a::ConcealmentOxGameWorld, b::ConcealmentOxGameWorld)::ConcealmentOxGameWorld
    surfaces = collect(values(Dict(s.uri => s for s in vcat(a.surfaces, b.surfaces))))

    by_key = Dict{Tuple{String,Int,Symbol,String},OxFinding}()
    for finding in vcat(a.findings, b.findings)
        by_key[(finding.uri, finding.line, finding.detector, finding.evidence)] = finding
    end
    findings = sort!(collect(values(by_key)); by=f -> (f.uri, f.line, String(f.detector), f.evidence))

    seed = a.seed == b.seed ? a.seed : splitmix64(xor(a.seed, b.seed))
    budget = max(a.pressure_budget, b.pressure_budget)
    _ox_make_world(seed, a.root, budget, surfaces, findings)
end

function oxgame_summary(w::ConcealmentOxGameWorld)
    by_detector = Dict{Symbol,Int}()
    by_uri = Dict{String,Int}()
    for finding in w.findings
        by_detector[finding.detector] = get(by_detector, finding.detector, 0) + 1
        by_uri[finding.uri] = get(by_uri, finding.uri, 0) + 1
    end

    (
        surfaces = length(w.surfaces),
        findings = length(w.findings),
        pressure_budget = w.pressure_budget,
        pressure_spent = w.pressure_spent,
        o_payoff = w.o_payoff,
        x_payoff = w.x_payoff,
        fingerprint = w.fingerprint,
        detectors = sort(collect(by_detector); by=x -> (-last(x), String(first(x)))),
        hottest_surfaces = first(sort(collect(by_uri); by=x -> -last(x)), min(8, length(by_uri))),
    )
end

function _oxscreen_clip(text::AbstractString, width::Int)::String
    width <= 0 && return ""
    s = replace(String(text), '\n' => ' ')
    length(s) <= width && return s
    width <= 3 && return repeat(".", width)

    stop = firstindex(s)
    for _ in 1:max(1, width - 3)
        stop = nextind(s, stop)
        stop > lastindex(s) && return s
    end
    string(s[firstindex(s):prevind(s, stop)], "...")
end

function _oxscreen_pad(text::AbstractString, width::Int)::String
    s = _oxscreen_clip(text, width)
    n = length(s)
    n >= width && return s
    string(s, repeat(" ", width - n))
end

function _oxscreen_bar(spent::Int, budget::Int, width::Int)::String
    budget <= 0 && return repeat(".", width)
    filled = clamp(round(Int, width * spent / budget), 0, width)
    string(repeat("#", filled), repeat(".", width - filled))
end

function _oxscreen_rows(w::ConcealmentOxGameWorld, max_rows::Int)
    rows = OxScreenRow[]
    for round in w.rounds[1:min(max_rows, length(w.rounds))]
        isempty(round.findings) && continue
        finding = first(round.findings)
        push!(rows, OxScreenRow(
            round.turn,
            round.concealer.action,
            round.finder.action,
            finding.severity,
            round.pressure_after,
            finding.color_hex,
            finding.derivation_uri,
            finding.evidence,
        ))
    end
    rows
end

function _oxscreen_summary_fingerprint(summary::Vector{Pair{String,String}}, seed::UInt64)::UInt64
    fp = stable_seed("oxscreen-summary"; seed=seed)
    for pair in sort(summary; by=p -> first(p))
        fp = xor(fp, stable_seed((first(pair), last(pair)); seed=seed))
    end
    fp
end

function _oxscreen_fingerprint(
    seed::UInt64,
    source_fingerprint::UInt64,
    summary::Vector{Pair{String,String}},
    rows::Vector{OxScreenRow},
    width::Int,
)::UInt64
    fp = stable_seed(("oxscreen", source_fingerprint, width); seed=seed)
    fp = xor(fp, _oxscreen_summary_fingerprint(summary, seed))
    for row in rows
        fp = xor(fp, stable_seed((row.turn, row.o_action, row.x_detector, row.uri, row.evidence); seed=seed))
    end
    splitmix64(fp)
end

function world_oxscreen(
    w::ConcealmentOxGameWorld;
    width::Int=96,
    max_rows::Int=12,
)::OxScreen
    rows = _oxscreen_rows(w, max_rows)
    bar_width = max(12, min(width - 24, 48))
    pressure_bar = _oxscreen_bar(w.pressure_spent, w.pressure_budget, bar_width)
    o_color = concealment_color(w.seed, UInt64(79))
    x_color = concealment_color(w.seed, UInt64(88))
    summary = Pair{String,String}[
        "surfaces" => string(length(w.surfaces)),
        "findings" => string(length(w.findings)),
        "pressure" => string(w.pressure_spent, "/", w.pressure_budget),
        "O payoff" => string(w.o_payoff),
        "X payoff" => string(w.x_payoff),
        "source fp" => "0x$(string(w.fingerprint, base=16, pad=16))",
    ]
    screen_seed = splitmix64(w.fingerprint)
    fp = _oxscreen_fingerprint(screen_seed, w.fingerprint, summary, rows, width)
    OxScreen(
        "OxScreen - Concealment O/X Open Game",
        width,
        pressure_bar,
        o_color,
        x_color,
        summary,
        rows,
        w.fingerprint,
        fp,
    )
end

function Base.merge(a::OxScreen, b::OxScreen)::OxScreen
    row_key(row::OxScreenRow) = (row.turn, row.o_action, row.x_detector, row.uri, row.evidence)
    by_key = Dict{Tuple{Int,Symbol,Symbol,String,String},OxScreenRow}()
    for row in vcat(a.rows, b.rows)
        by_key[row_key(row)] = row
    end
    rows = sort!(collect(values(by_key)); by=r -> (r.turn, String(r.x_detector), r.uri))
    width = max(a.width, b.width)
    source_fp = a.source_fingerprint == b.source_fingerprint ? a.source_fingerprint :
                splitmix64(xor(a.source_fingerprint, b.source_fingerprint))
    seed = splitmix64(source_fp)
    summary = sort!(unique(vcat(a.summary, b.summary)); by=p -> first(p))
    fp = _oxscreen_fingerprint(seed, source_fp, summary, rows, width)
    OxScreen(
        a.title == b.title ? a.title : "OxScreen - merged",
        width,
        length(a.pressure_bar) >= length(b.pressure_bar) ? a.pressure_bar : b.pressure_bar,
        a.o_color,
        b.x_color,
        summary,
        rows,
        source_fp,
        fp,
    )
end

function render_oxscreen(screen::OxScreen)::String
    width = screen.width
    rule = repeat("=", width)
    thin = repeat("-", width)
    lines = String[
        rule,
        _oxscreen_pad(screen.title, width),
        _oxscreen_pad("O $(screen.o_color) conceals | X $(screen.x_color) finds | pressure [$(screen.pressure_bar)]", width),
        thin,
    ]

    for pair in screen.summary
        push!(lines, _oxscreen_pad(string(first(pair), ": ", last(pair)), width))
    end

    push!(lines, thin)
    push!(lines, _oxscreen_pad("turn | O action              | X detector          | sev | pressure | evidence", width))
    push!(lines, thin)

    for row in screen.rows
        left = lpad(string(row.turn), 4)
        o = _oxscreen_pad(String(row.o_action), 21)
        x = _oxscreen_pad(String(row.x_detector), 19)
        sev = lpad(string(row.severity), 3)
        pressure = lpad(string(row.pressure_after), 8)
        evidence_width = max(16, width - 66)
        evidence = _oxscreen_clip(row.evidence, evidence_width)
        push!(lines, _oxscreen_pad("$left | $o | $x | $sev | $pressure | $evidence", width))
    end

    push!(lines, thin)
    push!(lines, _oxscreen_pad("screen fp: 0x$(string(screen.fingerprint, base=16, pad=16))", width))
    push!(lines, rule)
    join(lines, "\n")
end

Base.show(io::IO, screen::OxScreen) = print(io, render_oxscreen(screen))

function _shared_arena_players(seed::UInt64)
    OxArenaPlayer[
        OxArenaPlayer(
            :O,
            1,
            "spoiler / concealer",
            "Choose a source or interface move whose hidden ideology should be hard to distinguish.",
            concealment_color(seed, UInt64(201)),
        ),
        OxArenaPlayer(
            :X,
            -1,
            "duplicator / detector",
            "Answer each O move with a derivation-preserving observation and corrective pressure.",
            concealment_color(seed, UInt64(202)),
        ),
        OxArenaPlayer(
            :W,
            0,
            "witness / arena",
            "Hold shared state, certificate context, and the accepted/rejected boundary.",
            concealment_color(seed, UInt64(203)),
        ),
    ]
end

function _shared_arena_interfaces(seed::UInt64)
    defs = Tuple{String,Symbol,Vector{Symbol},String,String,String}[
        (
            "optic://gay/ox/source-observation",
            :OpticC,
            [:O, :X],
            "surface -> observable trace",
            "(surface, finding) -> naming pressure",
            "X severity minus O concealment slack",
        ),
        (
            "optic://gay/ox/scip-derivation",
            :OpticC,
            [:X, :W],
            "finding -> scip derivation URI",
            "(finding, certificate) -> exact navigation obligation",
            "stable derivation closes the local proof trail",
        ),
        (
            "paraoptic://gay/ox/strategy-family",
            :ParaOpticC,
            [:O, :X],
            "(parameters, surface) -> O/X strategy response",
            "(parameters, response, pressure) -> parameter update",
            "strategies are compared under a shared pressure budget",
        ),
        (
            "paraoptic://gay/ox/certificate-context",
            :ParaOpticC,
            [:O, :X, :W],
            "(arena, world fingerprint) -> accepted observation",
            "(arena, observation, rejected candidate) -> coworld residue",
            "accepted world state is bisimilar across code and derivation lanes",
        ),
        (
            "optic://gay/ox/artifact-replay",
            :OpticC,
            [:W],
            "world -> JSON/text artifact",
            "(world, artifact) -> replay check",
            "artifact fingerprint preserves tileable shared memory",
        ),
        (
            "paraoptic://gay/ox/player-combinations",
            :ParaOpticC,
            [:O, :X, :W],
            "subset(O,X,W) x lane -> local game",
            "(subset, local game, shared arena) -> merged arena state",
            "two-player, three-player, and mixed games share one equivalence relation",
        ),
    ]

    OxOpticInterface[
        OxOpticInterface(
            uri,
            optic_kind,
            players,
            play,
            coplay,
            payoff,
            concealment_color(seed, stable_seed((uri, optic_kind); seed=seed) % UInt64(4096) + UInt64(1)),
        )
        for (uri, optic_kind, players, play, coplay, payoff) in defs
    ]
end

function _shared_bisimulation_witnesses(seed::UInt64, w::ConcealmentOxGameWorld, max_witnesses::Int)
    witnesses = OxBisimulationWitness[]
    limit = min(max_witnesses, length(w.findings))
    for finding in w.findings[1:limit]
        uri = "bisim://gay/ox/" * string(length(witnesses) + 1)
        left = string(finding.uri, "#L", finding.line)
        right = finding.derivation_uri
        preserves = String[
            "observable detector = $(finding.detector)",
            "severity = $(finding.severity)",
            "pressure cost = $(finding.pressure_cost)",
            "color = $(finding.color_hex)",
        ]
        relation = "source state and SCIP-derived state expose the same O/X observation"
        push!(witnesses, OxBisimulationWitness(
            uri,
            left,
            right,
            relation,
            :O,
            :X,
            preserves,
            concealment_color(seed, stable_seed((uri, left, right); seed=seed) % UInt64(4096) + UInt64(1)),
        ))
    end
    witnesses
end

function _shared_arena_compositions()
    Pair{String,String}[
        "two-player O/X" => "Optic(C): O plays a surface; X coplays a finding and pressure.",
        "three-player O/X/W" => "Para(Optic(C)): W parameterizes context, certificates, and accepted state.",
        "mixed subsets" => "Tensor/coproduct combinations reuse one shared arena rather than separate games.",
        "bisimulation rule" => "If O moves in source, X must answer in derivation; if O moves in derivation, X must answer in artifact/source.",
        "shared arena" => "The arena state is the quotient of all lanes that preserve observation, payoff, pressure, and fingerprint.",
    ]
end

function _shared_arena_fingerprint(
    seed::UInt64,
    source_fingerprint::UInt64,
    players::Vector{OxArenaPlayer},
    interfaces::Vector{OxOpticInterface},
    witnesses::Vector{OxBisimulationWitness},
    compositions::Vector{Pair{String,String}},
)::UInt64
    fp = stable_seed(("shared-ox-arena", source_fingerprint); seed=seed)
    for player in players
        fp = xor(fp, stable_seed((player.id, player.trit, player.role, player.objective); seed=seed))
    end
    for interface in interfaces
        fp = xor(fp, stable_seed((interface.uri, interface.optic_kind, interface.players, interface.play, interface.coplay); seed=seed))
    end
    for witness in witnesses
        fp = xor(fp, stable_seed((witness.uri, witness.left_state, witness.right_state, witness.relation, witness.preserves); seed=seed))
    end
    for pair in compositions
        fp = xor(fp, stable_seed((first(pair), last(pair)); seed=seed))
    end
    splitmix64(fp)
end

function world_shared_oxarena(
    w::ConcealmentOxGameWorld;
    seed::UInt64=w.seed,
    max_witnesses::Int=9,
)::SharedOxArena
    players = _shared_arena_players(seed)
    interfaces = _shared_arena_interfaces(seed)
    witnesses = _shared_bisimulation_witnesses(seed, w, max_witnesses)
    compositions = _shared_arena_compositions()
    fp = _shared_arena_fingerprint(seed, w.fingerprint, players, interfaces, witnesses, compositions)
    SharedOxArena(
        seed,
        "arena://gay/ox/shared-bisimulation",
        players,
        interfaces,
        witnesses,
        compositions,
        w.fingerprint,
        fp,
    )
end

world_shared_oxarena(w::OxgameRemixWorld; seed::UInt64=w.seed, max_witnesses::Int=9)::SharedOxArena =
    world_shared_oxarena(w.concealment_world; seed=seed, max_witnesses=max_witnesses)

function Base.merge(a::SharedOxArena, b::SharedOxArena)::SharedOxArena
    player_key(player::OxArenaPlayer) = player.id
    interface_key(interface::OxOpticInterface) = interface.uri
    witness_key(witness::OxBisimulationWitness) = witness.uri

    players = collect(values(Dict(player_key(p) => p for p in vcat(a.players, b.players))))
    interfaces = collect(values(Dict(interface_key(i) => i for i in vcat(a.interfaces, b.interfaces))))
    witnesses = collect(values(Dict(witness_key(w) => w for w in vcat(a.witnesses, b.witnesses))))
    compositions = sort!(unique(vcat(a.compositions, b.compositions)); by=p -> first(p))
    source_fp = a.source_fingerprint == b.source_fingerprint ? a.source_fingerprint :
                splitmix64(xor(a.source_fingerprint, b.source_fingerprint))
    seed = a.seed == b.seed ? a.seed : splitmix64(xor(a.seed, b.seed))

    sort!(players; by=p -> String(p.id))
    sort!(interfaces; by=i -> i.uri)
    sort!(witnesses; by=w -> w.uri)

    fp = _shared_arena_fingerprint(seed, source_fp, players, interfaces, witnesses, compositions)
    SharedOxArena(seed, a.arena_uri, players, interfaces, witnesses, compositions, source_fp, fp)
end

function shared_oxarena_summary(a::SharedOxArena)
    trit_sum = sum(player.trit for player in a.players)
    optic_counts = Dict{Symbol,Int}()
    for interface in a.interfaces
        optic_counts[interface.optic_kind] = get(optic_counts, interface.optic_kind, 0) + 1
    end
    (
        arena_uri = a.arena_uri,
        players = length(a.players),
        interfaces = length(a.interfaces),
        witnesses = length(a.witnesses),
        trit_sum = trit_sum,
        gf3_conserved = mod(trit_sum, 3) == 0,
        optic_counts = sort(collect(optic_counts); by=p -> String(first(p))),
        source_fingerprint = a.source_fingerprint,
        fingerprint = a.fingerprint,
    )
end

function render_shared_oxarena(a::SharedOxArena; width::Int=100)::String
    rule = repeat("=", width)
    thin = repeat("-", width)
    lines = String[
        rule,
        _oxscreen_pad("Shared O/X Arena - bisimulation Para(Optic(C))", width),
        _oxscreen_pad("arena: $(a.arena_uri)", width),
        _oxscreen_pad("source fp: 0x$(string(a.source_fingerprint, base=16, pad=16))", width),
        _oxscreen_pad("arena fp: 0x$(string(a.fingerprint, base=16, pad=16))", width),
        thin,
        _oxscreen_pad("players", width),
    ]

    for player in a.players
        trit = player.trit > 0 ? "+1" : string(player.trit)
        push!(lines, _oxscreen_pad("$(rpad(String(player.id), 2)) $(rpad(trit, 3)) $(player.color_hex) $(player.role) - $(player.objective)", width))
    end

    push!(lines, thin)
    push!(lines, _oxscreen_pad("interfaces", width))
    for interface in a.interfaces
        players = join(String.(interface.players), "+")
        push!(lines, _oxscreen_pad("$(interface.color_hex) $(interface.optic_kind) [$(players)] $(interface.uri)", width))
        push!(lines, _oxscreen_pad("  play: $(interface.play)", width))
        push!(lines, _oxscreen_pad("  coplay: $(interface.coplay)", width))
    end

    push!(lines, thin)
    push!(lines, _oxscreen_pad("bisimulation witnesses", width))
    for witness in a.witnesses
        push!(lines, _oxscreen_pad("$(witness.color_hex) $(witness.uri)", width))
        push!(lines, _oxscreen_pad("  $(witness.left_state) <-> $(witness.right_state)", width))
        push!(lines, _oxscreen_pad("  $(witness.relation)", width))
    end

    push!(lines, thin)
    push!(lines, _oxscreen_pad("composition rules", width))
    for pair in a.compositions
        push!(lines, _oxscreen_pad("- $(first(pair)): $(last(pair))", width))
    end

    push!(lines, rule)
    join(lines, "\n")
end

Base.show(io::IO, a::SharedOxArena) = print(io, render_shared_oxarena(a))

function _scip_default_ocaml_root()
    get(ENV, "SCIP_OCAML_ROOT", joinpath(homedir(), "worlds", "scip-work", "scip-ocaml"))
end

function _scip_default_oxgame_root()
    get(ENV, "OXGAME_SOURCE_ROOT", joinpath(homedir(), "worlds", "x", "src", "oxgame"))
end

function _scip_default_oxcaml_docs_root()
    get(ENV, "OXCAML_DOCS_ROOT", joinpath(homedir(), "worlds", "docs-clone", "oxcaml"))
end

function _read_text(path::AbstractString)::String
    try
        read(path, String)
    catch
        ""
    end
end

function _git_short_head(root::AbstractString)::String
    isdir(joinpath(root, ".git")) || return "untracked"
    try
        String(readchomp(`git -C $root rev-parse --short=12 HEAD`))
    catch
        "unknown"
    end
end

function _count_files(root::AbstractString, predicate)::Int
    isdir(root) || return 0
    count = 0
    for (_, _, files) in walkdir(root)
        for file in files
            predicate(file) && (count += 1)
        end
    end
    count
end

function _count_matching_files(root::AbstractString, predicate)::Int
    isdir(root) || return 0
    count = 0
    for (walk_root, _, files) in walkdir(root)
        for file in files
            predicate(joinpath(walk_root, file)) && (count += 1)
        end
    end
    count
end

function _docs_page_count(manifest_path::AbstractString)::Int
    text = _read_text(manifest_path)
    m = match(r"\"page_count\"\s*:\s*(\d+)", text)
    m === nothing && return 0
    parse(Int, m.captures[1])
end

function _scip_probe_status(probe_scip::AbstractString)::Tuple{String,Int}
    if isfile(probe_scip)
        return "partial scip-ocaml probe present; compiler-libs mismatch leaves this as boundary evidence", 3
    end
    return "probe not materialized in artifact; use scip-ocaml with matching compiler-libs to emit the full index", 0
end

function _scip_endpoint(
    seed::UInt64,
    uri::String,
    kind::Symbol,
    local_path::AbstractString,
    status::String,
    evidence_count::Int,
)
    color = concealment_color(seed, stable_seed((uri, kind, status); seed=seed) % UInt64(4096) + UInt64(1))
    ScipEndpoint(uri, kind, String(local_path), status, evidence_count, color)
end

function _colored_scip_morphism(
    seed::UInt64,
    uri::String,
    source_uri::String,
    target_uri::String,
    kind::Symbol,
    trit::Int,
    status::Symbol,
    evidence::String,
    limitation::String,
)
    color = concealment_color(seed, stable_seed((uri, source_uri, target_uri, kind, trit); seed=seed) % UInt64(4096) + UInt64(1))
    ColoredScipMorphism(uri, source_uri, target_uri, kind, trit, status, color, evidence, limitation)
end

function _scip_oxcaml_endpoints(
    seed::UInt64,
    scip_ocaml_root::AbstractString,
    oxgame_root::AbstractString,
    oxcaml_docs_root::AbstractString,
    probe_scip::AbstractString,
)
    cmt_count = _count_matching_files(oxgame_root, p -> occursin("/_build/", replace(p, '\\' => '/')) &&
                                                     (endswith(p, ".cmt") || endswith(p, ".cmti")))
    source_count = _count_files(joinpath(oxgame_root, "lib"), f -> endswith(f, ".ml") || endswith(f, ".mli"))
    docs_count = _docs_page_count(joinpath(oxcaml_docs_root, "manifest.json"))
    probe_status, probe_docs = _scip_probe_status(probe_scip)
    scip_ocaml_status = "latest local bmorphism/scip-ocaml master; external-symbol Path.t resolution available"
    oxgame_status = string(
        "Oxgame source has ",
        source_count,
        " OCaml source/interface files and ",
        cmt_count,
        " typed-tree files; ",
        probe_status,
    )
    oxcaml_docs_status = string("docs://oxcaml clone exposes ", docs_count, " pages for modes, kinds, uniqueness, capsules, and stack allocation")
    scip_oxcaml_status = "target code-index lane; represented here by scip-ocaml symbol grammar and external-symbol morphisms until an OxCaml source index exists"

    ScipEndpoint[
        _scip_endpoint(seed, "tool://bmorphism/scip-ocaml", :indexer, scip_ocaml_root, scip_ocaml_status, source_count),
        _scip_endpoint(seed, "scip://oxgame", :code_index, oxgame_root, oxgame_status, probe_docs),
        _scip_endpoint(seed, "docs://oxcaml", :docs, oxcaml_docs_root, oxcaml_docs_status, docs_count),
        _scip_endpoint(seed, "scip://oxcaml", :code_index, "pending:oxcaml-source", scip_oxcaml_status, 0),
    ]
end

function _scip_oxcaml_morphisms(seed::UInt64)
    ColoredScipMorphism[
        _colored_scip_morphism(
            seed,
            "morphism://scip-ocaml/typedtree-to-scip-oxgame",
            "tool://bmorphism/scip-ocaml",
            "scip://oxgame",
            :typedtree_index,
            -1,
            :blocked,
            "scip-ocaml emitted a tiny valid probe index, but most Oxgame .cmt/.cmti files were produced by OCaml 5.4.1 and are unreadable by this binary.",
            "Rebuild Oxgame and scip-ocaml with the same compiler-libs ABI to turn this red edge green.",
        ),
        _colored_scip_morphism(
            seed,
            "morphism://oxgame/source-to-oxcaml-docs",
            "scip://oxgame",
            "docs://oxcaml",
            :mode_annotation_evidence,
            0,
            :witnessed,
            "Oxgame Lens, Para, Arena, and PettingZoo interfaces carry comments for local_, portable, unique_, exclave_, and capsule boundaries.",
            "Docs evidence is semantic rather than a compiler-checked source edge in this Gay.jl artifact.",
        ),
        _colored_scip_morphism(
            seed,
            "morphism://docs-oxcaml/to-scip-oxcaml-symbols",
            "docs://oxcaml",
            "scip://oxcaml",
            :docs_to_symbol_expectation,
            1,
            :proposed,
            "Oxcaml docs define the modes/kinds/uniqueness vocabulary that should color future SCIP symbols, occurrences, and type-signature hovers.",
            "The docs clone is not a SCIP index; it seeds the expected color semantics.",
        ),
        _colored_scip_morphism(
            seed,
            "morphism://scip-ocaml/external-symbol-bridge",
            "scip://oxgame",
            "scip://oxcaml",
            :external_symbol_bridge,
            1,
            :partial,
            "scip-ocaml resolves missed in-tree Texp_ident references through Path.t into deterministic external symbols such as scip-ocaml opam <head-module> . <descriptors>.",
            "Package-accurate cross-repo navigation still needs a module-to-package resolution table.",
        ),
        _colored_scip_morphism(
            seed,
            "morphism://oxgame/color-unifier/scip-colors",
            "scip://oxgame/lib/oxgame_kernel/color_unifier.mli",
            "docs://oxcaml/documentation/kinds/intro/",
            :colored_morphism_lift,
            0,
            :witnessed,
            "Oxgame Color_unifier lifts Color_trit, Secret_colors, Share3, Trit, DisCoPy wires, and CatColab places into one conserved color record.",
            "The bridge colors semantic lanes; it does not assert perceptual equivalence without Gay.jl color-chain replay.",
        ),
        _colored_scip_morphism(
            seed,
            "morphism://paraoptic/shared-arena-to-scip",
            "arena://gay/ox/shared-bisimulation",
            "scip://oxgame",
            :paraoptic_bisimulation,
            -1,
            :witnessed,
            "SharedOxArena models O/X/W as Para(Optic(C)); scip://oxgame supplies concrete Lens/Para/Arena names for the same play-coplay split.",
            "Bisimulation here is structural and artifact-level, not a full proof over every Oxgame module.",
        ),
    ]
end

function _scip_oxcaml_commitments()
    Pair{String,String}[
        "version lock" => "SCIP over OCaml is a typed-tree morphism; compiler-libs ABI mismatch is a real red edge, not noise.",
        "docs-to-code color" => "Oxcaml docs color the intended mode/kind semantics; scip://oxcaml should eventually carry those colors on symbols.",
        "source-to-doc witness" => "Oxgame already names Lens, Para, Arena, Color_unifier, and mode comments that map into docs://oxcaml.",
        "external edges" => "bmorphism/scip-ocaml makes cross-index dependency references visible with deterministic Path.t-derived symbols.",
        "Gay.jl replay" => "Every morphism gets a stable Gay.jl color so interpretation changes can be replayed as color-chain deltas.",
    ]
end

function _scip_oxcaml_fingerprint(
    seed::UInt64,
    scip_ocaml_head::String,
    endpoints::Vector{ScipEndpoint},
    morphisms::Vector{ColoredScipMorphism},
    commitments::Vector{Pair{String,String}},
)::UInt64
    fp = stable_seed(("scip-oxcaml-bridge", scip_ocaml_head); seed=seed)
    for endpoint in sort(endpoints; by=e -> e.uri)
        fp = xor(fp, stable_seed((endpoint.uri, endpoint.kind, endpoint.status, endpoint.evidence_count); seed=seed))
    end
    for morphism in sort(morphisms; by=m -> m.uri)
        fp = xor(fp, stable_seed((morphism.uri, morphism.source_uri, morphism.target_uri, morphism.kind, morphism.trit, morphism.status); seed=seed))
    end
    for pair in sort(commitments; by=p -> first(p))
        fp = xor(fp, stable_seed((first(pair), last(pair)); seed=seed))
    end
    splitmix64(fp)
end

function world_scip_oxcaml_bridge(;
    scip_ocaml_root::AbstractString=_scip_default_ocaml_root(),
    oxgame_root::AbstractString=_scip_default_oxgame_root(),
    oxcaml_docs_root::AbstractString=_scip_default_oxcaml_docs_root(),
    probe_scip::AbstractString="/tmp/oxgame.scip",
    seed::UInt64=GAY_SEED,
)::ScipOxcamlBridgeWorld
    head = _git_short_head(scip_ocaml_root)
    endpoints = _scip_oxcaml_endpoints(seed, scip_ocaml_root, oxgame_root, oxcaml_docs_root, probe_scip)
    morphisms = _scip_oxcaml_morphisms(seed)
    commitments = _scip_oxcaml_commitments()
    fp = _scip_oxcaml_fingerprint(seed, head, endpoints, morphisms, commitments)
    ScipOxcamlBridgeWorld(seed, head, endpoints, morphisms, commitments, fp)
end

function Base.merge(a::ScipOxcamlBridgeWorld, b::ScipOxcamlBridgeWorld)::ScipOxcamlBridgeWorld
    endpoint_key(endpoint::ScipEndpoint) = endpoint.uri
    morphism_key(morphism::ColoredScipMorphism) = morphism.uri

    endpoints = collect(values(Dict(endpoint_key(e) => e for e in vcat(a.endpoints, b.endpoints))))
    morphisms = collect(values(Dict(morphism_key(m) => m for m in vcat(a.morphisms, b.morphisms))))
    commitments = sort!(unique(vcat(a.commitments, b.commitments)); by=p -> first(p))
    sort!(endpoints; by=e -> e.uri)
    sort!(morphisms; by=m -> m.uri)

    seed = a.seed == b.seed ? a.seed : splitmix64(xor(a.seed, b.seed))
    head = a.scip_ocaml_head == b.scip_ocaml_head ? a.scip_ocaml_head : string(a.scip_ocaml_head, "+", b.scip_ocaml_head)
    fp = _scip_oxcaml_fingerprint(seed, head, endpoints, morphisms, commitments)
    ScipOxcamlBridgeWorld(seed, head, endpoints, morphisms, commitments, fp)
end

function scip_oxcaml_bridge_summary(w::ScipOxcamlBridgeWorld)
    trit_sum = sum(m.trit for m in w.morphisms)
    by_status = Dict{Symbol,Int}()
    by_kind = Dict{Symbol,Int}()
    for morphism in w.morphisms
        by_status[morphism.status] = get(by_status, morphism.status, 0) + 1
        by_kind[morphism.kind] = get(by_kind, morphism.kind, 0) + 1
    end
    (
        scip_ocaml_head = w.scip_ocaml_head,
        endpoints = length(w.endpoints),
        morphisms = length(w.morphisms),
        trit_sum = trit_sum,
        gf3_conserved = mod(trit_sum, 3) == 0,
        statuses = sort(collect(by_status); by=p -> String(first(p))),
        kinds = sort(collect(by_kind); by=p -> String(first(p))),
        fingerprint = w.fingerprint,
    )
end

function render_scip_oxcaml_bridge(w::ScipOxcamlBridgeWorld; width::Int=108)::String
    rule = repeat("=", width)
    thin = repeat("-", width)
    lines = String[
        rule,
        _oxscreen_pad("SCIP/OxCaml Colored Morphism World", width),
        _oxscreen_pad("bmorphism/scip-ocaml head: $(w.scip_ocaml_head)", width),
        _oxscreen_pad("bridge fp: 0x$(string(w.fingerprint, base=16, pad=16))", width),
        thin,
        _oxscreen_pad("endpoints", width),
    ]

    for endpoint in w.endpoints
        push!(lines, _oxscreen_pad("$(endpoint.color_hex) $(endpoint.kind) $(endpoint.uri)", width))
        push!(lines, _oxscreen_pad("  path: $(endpoint.local_path)", width))
        push!(lines, _oxscreen_pad("  status: $(endpoint.status)", width))
        push!(lines, _oxscreen_pad("  evidence count: $(endpoint.evidence_count)", width))
    end

    push!(lines, thin)
    push!(lines, _oxscreen_pad("colored morphisms", width))
    for morphism in w.morphisms
        trit = morphism.trit > 0 ? "+1" : string(morphism.trit)
        push!(lines, _oxscreen_pad("$(morphism.color_hex) $(rpad(trit, 3)) $(morphism.status) $(morphism.kind)", width))
        push!(lines, _oxscreen_pad("  $(morphism.source_uri) -> $(morphism.target_uri)", width))
        push!(lines, _oxscreen_pad("  $(morphism.evidence)", width))
        !isempty(morphism.limitation) && push!(lines, _oxscreen_pad("  limit: $(morphism.limitation)", width))
    end

    push!(lines, thin)
    push!(lines, _oxscreen_pad("commitments", width))
    for pair in w.commitments
        push!(lines, _oxscreen_pad("- $(first(pair)): $(last(pair))", width))
    end

    push!(lines, rule)
    join(lines, "\n")
end

Base.show(io::IO, w::ScipOxcamlBridgeWorld) = print(io, render_scip_oxcaml_bridge(w))

function _oxgame_default_docs_root()
    get(ENV, "OXGAME_DOCS_ROOT", joinpath(homedir(), "worlds", "docs-clone", "oxgame"))
end

function _oxgame_git_head(root::AbstractString)::String
    git_dir = joinpath(root, ".git")
    isdir(git_dir) || return "untracked"
    head = try
        readchomp(`git -C $root rev-parse --short=12 HEAD`)
    catch
        "unknown"
    end
    String(head)
end

function _oxgame_docs_uri(path::AbstractString, text::AbstractString)::String
    lines = Base.split(String(text), '\n'; limit=2)
    first_line = isempty(lines) ? "" : String(first(lines))
    m = match(r"source:\s*x/src/oxgame/(.*?)\s*-->", first_line)
    if m !== nothing
        return "docs://oxgame/" * String(m.captures[1])
    end
    "docs://oxgame/" * basename(path)
end

function _oxgame_source_score(text::AbstractString)::Int
    lower = lowercase(String(text))
    weighted = Pair{String,Int}[
        "play" => 2,
        "coplay" => 3,
        "payoff" => 3,
        "equilib" => 3,
        "nash" => 4,
        "cert" => 4,
        "lens" => 4,
        "optic" => 4,
        "acset" => 3,
        "scip" => 4,
        "world://" => 4,
        "coworld://" => 4,
        "tile://" => 3,
        "strategy://" => 3,
        "arena://" => 3,
        "color" => 2,
        "gf(3)" => 3,
        "survival" => 2,
        "pressure" => 2,
    ]
    score = 0
    for (needle, weight) in weighted
        score += weight * count(needle, lower)
    end
    score
end

function _oxgame_excerpt(text::AbstractString)::String
    for line in Base.split(String(text), '\n')
        stripped = strip(line)
        isempty(stripped) && continue
        startswith(stripped, "<!--") && continue
        startswith(stripped, "---") && continue
        return _oxscreen_clip(stripped, 180)
    end
    ""
end

function _oxgame_docs_sources(seed::UInt64, docs_root::AbstractString; max_sources::Int=9)
    root = String(docs_root)
    isdir(root) || return OxgameRemixSource[]

    paths = String[]
    index_path = joinpath(root, "index.md")
    isfile(index_path) && push!(paths, index_path)
    pages_root = joinpath(root, "pages")
    if isdir(pages_root)
        for file in sort(readdir(pages_root; join=true))
            isfile(file) && endswith(file, ".md") && push!(paths, file)
        end
    end

    sources = OxgameRemixSource[]
    for path in paths
        text = try
            read(path, String)
        catch
            continue
        end
        score = _oxgame_source_score(text)
        score > 0 || continue
        uri = _oxgame_docs_uri(path, text)
        color_index = stable_seed((uri, score); seed=seed) % UInt64(4096) + UInt64(1)
        color = concealment_color(seed, color_index)
        push!(sources, OxgameRemixSource(uri, path, score, color, _oxgame_excerpt(text)))
    end

    sort!(sources; by=s -> (-s.score, s.uri))
    first(sources, min(max_sources, length(sources)))
end

function _oxgame_remix_lanes(seed::UInt64)
    defs = Tuple{String,Int,String}[
        ("tile://oxgame/convergence", 0, "neutral benchmark and rendered locality"),
        ("strategy://oxgame/best-response", 1, "accelerated profile update"),
        ("arena://oxgame/fixed-point", 0, "capsule boundary for mutable play state"),
        ("world://oxgame/equilibrium", 1, "accepted forward equilibrium"),
        ("coworld://oxgame/rejected-accelerants", -1, "rejected candidates kept out of accepted state"),
        ("scip://oxgame-derived/convergence-hot-path", -1, "exact derivation and hot-path evidence"),
    ]
    OxgameRemixLane[
        OxgameRemixLane(uri, trit, role, concealment_color(seed, stable_seed((uri, trit); seed=seed) % UInt64(4096) + UInt64(1)))
        for (uri, trit, role) in defs
    ]
end

function _oxgame_remix_commitments()
    Pair{String,String}[
        "open-game core" => "Remix O/X detection as play, coplay, payoff, equilibrium, and certificate rather than a one-shot scan.",
        "Nash boundary" => "Treat fingerprints and screen summaries as post-convergence certificates; keep verification outside inner search loops.",
        "survival pressure" => "Let finite pressure choose detectors first; missed surfaces become coworld evidence rather than silently mutating the world.",
        "SCIP exactness" => "Keep derivation URIs stable so fuzzy recall can suggest candidates while exact navigation closes the proof trail.",
        "ACSet artifact" => "Store the remix as tileable JSON/text artifacts that can cross substrates without changing the accepted world state.",
        "color chain" => "Seed each lane/source with Gay.jl colors so repeated composition has a replayable visual trace.",
    ]
end

function _oxgame_remix_fingerprint(
    seed::UInt64,
    gay_head::String,
    oxgame_status::String,
    concealment_world::ConcealmentOxGameWorld,
    screen::OxScreen,
    sources::Vector{OxgameRemixSource},
    lanes::Vector{OxgameRemixLane},
    commitments::Vector{Pair{String,String}},
)::UInt64
    fp = stable_seed(("oxgame-remix", gay_head, oxgame_status, concealment_world.fingerprint, screen.fingerprint); seed=seed)
    for source in sources
        fp = xor(fp, stable_seed((source.uri, source.score, source.excerpt); seed=seed))
    end
    for lane in lanes
        fp = xor(fp, stable_seed((lane.uri, lane.trit, lane.role); seed=seed))
    end
    for pair in commitments
        fp = xor(fp, stable_seed((first(pair), last(pair)); seed=seed))
    end
    splitmix64(fp)
end

function world_oxgame_remix(;
    root::AbstractString=normpath(joinpath(@__DIR__, "..")),
    docs_root::AbstractString=_oxgame_default_docs_root(),
    seed::UInt64=GAY_SEED,
    pressure_budget::Int=144,
    max_files::Int=512,
    max_sources::Int=9,
    oxgame_status::AbstractString="github: plurigrid/oxgame unresolved here; using docs://oxgame mirror",
)::OxgameRemixWorld
    local_world = world_concealment_oxgame(; root=root, seed=seed, pressure_budget=pressure_budget, max_files=max_files)
    screen = world_oxscreen(local_world; width=100, max_rows=8)
    gay_head = _oxgame_git_head(root)
    sources = _oxgame_docs_sources(seed, docs_root; max_sources=max_sources)
    lanes = _oxgame_remix_lanes(seed)
    commitments = _oxgame_remix_commitments()
    fp = _oxgame_remix_fingerprint(seed, gay_head, String(oxgame_status), local_world, screen, sources, lanes, commitments)
    OxgameRemixWorld(
        seed,
        gay_head,
        String(oxgame_status),
        String(docs_root),
        local_world,
        screen,
        sources,
        lanes,
        commitments,
        fp,
    )
end

function Base.merge(a::OxgameRemixWorld, b::OxgameRemixWorld)::OxgameRemixWorld
    source_key(source::OxgameRemixSource) = source.uri
    lane_key(lane::OxgameRemixLane) = lane.uri

    sources_by_uri = Dict{String,OxgameRemixSource}()
    for source in vcat(a.sources, b.sources)
        current = get(sources_by_uri, source_key(source), nothing)
        if current === nothing || source.score >= current.score
            sources_by_uri[source_key(source)] = source
        end
    end

    lanes_by_uri = Dict{String,OxgameRemixLane}()
    for lane in vcat(a.lanes, b.lanes)
        lanes_by_uri[lane_key(lane)] = lane
    end

    seed = a.seed == b.seed ? a.seed : splitmix64(xor(a.seed, b.seed))
    local_world = merge(a.concealment_world, b.concealment_world)
    screen = merge(a.screen, b.screen)
    sources = sort!(collect(values(sources_by_uri)); by=s -> (-s.score, s.uri))
    lanes = sort!(collect(values(lanes_by_uri)); by=l -> l.uri)
    commitments = sort!(unique(vcat(a.commitments, b.commitments)); by=p -> first(p))
    gay_head = a.gay_head == b.gay_head ? a.gay_head : string(a.gay_head, "+", b.gay_head)
    status = a.oxgame_status == b.oxgame_status ? a.oxgame_status : string(a.oxgame_status, " | ", b.oxgame_status)
    fp = _oxgame_remix_fingerprint(seed, gay_head, status, local_world, screen, sources, lanes, commitments)
    OxgameRemixWorld(seed, gay_head, status, a.docs_root, local_world, screen, sources, lanes, commitments, fp)
end

function oxgame_remix_summary(w::OxgameRemixWorld)
    lane_sum = sum(lane.trit for lane in w.lanes)
    (
        gay_head = w.gay_head,
        oxgame_status = w.oxgame_status,
        docs_root = w.docs_root,
        docs_sources = length(w.sources),
        lanes = length(w.lanes),
        lane_sum = lane_sum,
        gf3_conserved = mod(lane_sum, 3) == 0,
        local_findings = length(w.concealment_world),
        screen_fingerprint = w.screen.fingerprint,
        fingerprint = w.fingerprint,
    )
end

function render_oxgame_remix(w::OxgameRemixWorld; width::Int=100)::String
    rule = repeat("=", width)
    thin = repeat("-", width)
    lines = String[
        rule,
        _oxscreen_pad("Oxgame Remix World - latest overall", width),
        _oxscreen_pad("Gay.jl $(w.gay_head) + $(w.oxgame_status)", width),
        thin,
        _oxscreen_pad("remix fp: 0x$(string(w.fingerprint, base=16, pad=16))", width),
        _oxscreen_pad("local O/X fp: 0x$(string(w.concealment_world.fingerprint, base=16, pad=16))", width),
        _oxscreen_pad("screen fp: 0x$(string(w.screen.fingerprint, base=16, pad=16))", width),
        _oxscreen_pad("docs root: $(w.docs_root)", width),
        thin,
        _oxscreen_pad("lanes", width),
    ]

    for lane in w.lanes
        trit = lane.trit > 0 ? "+1" : string(lane.trit)
        push!(lines, _oxscreen_pad("$(rpad(trit, 3)) $(lane.color_hex) $(lane.uri) - $(lane.role)", width))
    end

    push!(lines, thin)
    push!(lines, _oxscreen_pad("sources", width))
    for source in w.sources
        push!(lines, _oxscreen_pad("$(lpad(string(source.score), 5)) $(source.color_hex) $(source.uri)", width))
        !isempty(source.excerpt) && push!(lines, _oxscreen_pad("      $(source.excerpt)", width))
    end

    push!(lines, thin)
    push!(lines, _oxscreen_pad("commitments", width))
    for pair in w.commitments
        push!(lines, _oxscreen_pad("- $(first(pair)): $(last(pair))", width))
    end

    push!(lines, rule)
    join(lines, "\n")
end

Base.show(io::IO, w::OxgameRemixWorld) = print(io, render_oxgame_remix(w))
