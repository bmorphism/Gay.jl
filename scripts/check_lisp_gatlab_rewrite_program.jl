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

function default_program_form()
    """
    (rewrite-program
      (rewrite-execution
        (query witness 1)
        (max-samples 2)
        (backend algebraicjulia))
      (rewrite-execution
        (query ordinal 1)
        (max-samples 1)
        (backend algebraicjulia)))
    """
end

function parse_program()
    raw_form = strip(get(ENV, "GAY_LISP_GATLAB_REWRITE_PROGRAM_FORM", ""))
    isempty(raw_form) ?
        parse_lisp_gatlab_rewrite_program(default_program_form()) :
        parse_lisp_gatlab_rewrite_program(raw_form)
end

status = load_algebraicjulia_capabilities!()
w = world_lisp_gatlab_bridge()
program = parse_program()

all(request -> request.backend == :algebraicjulia, program.requests) ||
    error("Package-backed rewrite-program validation requires every request backend to be algebraicjulia")

execution = lisp_gatlab_rewrite_program_execution(w, program)
execution.all_selected_all_materialized ||
    error("Not all selected rewrite-program ordinals were materialized")
execution.all_selected_all_targets ||
    error("Not all selected rewrite-program executions reached their target ACSet pattern")
isempty(execution.selected_ordinals) && error("Rewrite-program selected no ordinals")

trace = lisp_gatlab_rewrite_program_trace(execution)
trace.coverage_complete || error("Rewrite-program trace coverage is incomplete")

json = render_lisp_gatlab_rewrite_program_trace_json(
    trace;
    bridge=w,
    extension_status=status,
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
