#!/usr/bin/env julia

using Gay
using UUIDs

function load_algebraicjulia_capabilities!()
    status = algebraicjulia_bridge_status()
    if !isempty(status.missing)
        missing = join(string.(status.missing), ", ")
        error("Missing AlgebraicJulia weak dependencies in this environment: $missing")
    end

    for cap in status.capabilities
        Base.require(Base.PkgId(UUID(cap.uuid), String(cap.package)))
    end

    status = algebraicjulia_bridge_status()
    status.extension_loaded || error("GayAlgebraicJuliaExt did not load after requiring weak dependencies")
    status
end

function parse_nonnegative_int_env(name::AbstractString, default::AbstractString)
    raw = strip(get(ENV, name, default))
    n = tryparse(Int, raw)
    n === nothing && error("$name must be a non-negative integer, got: $raw")
    n >= 0 || error("$name must be a non-negative integer, got: $raw")
    n
end

function parse_query_operation()
    raw = strip(get(ENV, "GAY_LISP_GATLAB_QUERY_OPERATION", "witness"))
    isempty(raw) && error("GAY_LISP_GATLAB_QUERY_OPERATION cannot be empty")
    Symbol(replace(raw, "-" => "_"))
end

function strip_optional_quotes(s::AbstractString)
    if length(s) >= 2 && ((first(s) == '\'' && last(s) == '\'') || (first(s) == '"' && last(s) == '"'))
        return s[2:end-1]
    end
    s
end

function parse_query_arg(s::AbstractString)
    raw = strip_optional_quotes(strip(s))
    isempty(raw) && error("Empty item in GAY_LISP_GATLAB_QUERY_ARGS")

    n = tryparse(Int, raw)
    n !== nothing && return n

    startswith(raw, "#") && return raw
    occursin(r"^0x[0-9a-fA-F]+$", raw) && return raw

    Symbol(replace(raw, "-" => "_"))
end

function parse_query_args()
    raw = strip(get(ENV, "GAY_LISP_GATLAB_QUERY_ARGS", "1"))
    isempty(raw) && return Any[]
    Any[parse_query_arg(item) for item in split(raw, ",") if !isempty(strip(item))]
end

function parse_rewrite_request()
    raw_form = strip(get(ENV, "GAY_LISP_GATLAB_REWRITE_FORM", ""))
    if !isempty(raw_form)
        return parse_lisp_gatlab_rewrite_form(raw_form)
    end

    lisp_gatlab_rewrite_request(
        parse_query_operation(),
        parse_query_args()...;
        max_samples=parse_nonnegative_int_env("GAY_LISP_GATLAB_MAX_SAMPLES", "2"),
        backend=:algebraicjulia,
    )
end

status = load_algebraicjulia_capabilities!()
w = world_lisp_gatlab_bridge()
request = parse_rewrite_request()
plan = lisp_gatlab_rewrite_plan(w, request; materialization_backend=:algebraicjulia)
materialization = materialize_lisp_gatlab_rewrite_plan(w, plan; backend=:algebraicjulia)
execution = lisp_gatlab_rewrite_execution(plan, materialization)
json = render_lisp_gatlab_rewrite_execution_json(
    execution;
    request=request,
    materialization=materialization,
    bridge=w,
    extension_status=status,
    include_selected_candidates=true,
)

if isempty(ARGS)
    print(json)
else
    output = first(ARGS)
    mkpath(dirname(output))
    open(output, "w") do io
        write(io, json)
    end
end
