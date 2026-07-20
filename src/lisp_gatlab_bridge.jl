# Lisp/GATlab bridge for counterfactual olog closure.
#
# This is a dependency-light projection layer: it keeps the current Gay.jl
# world executable without Catlab/GATlab installed, while emitting declarations
# that can be pasted into or consumed by a real AlgebraicJulia presentation.

struct LispGATObject
    name::Symbol
    kind::Symbol
    color_hex::String
    evidence::String
end

struct LispGATMorphism
    name::Symbol
    kind::Symbol
    dom::Symbol
    cod::Symbol
    color_hex::String
    evidence::String
end

struct LispGATEquation
    lhs::Vector{Symbol}
    rhs::Vector{Symbol}
    color_hex::String
    evidence::String
end

struct LispGATCounterfactual
    witness_ordinal::Int
    from_aspect::Symbol
    to_aspect::Symbol
    trit_delta::Int
    closure_effect::Symbol
    color_hex::String
    semantic_cost::Float64
    lhs::Vector{Symbol}
    rhs::Vector{Symbol}
end

struct LispGATRewriteCandidate
    ordinal::Int
    counterfactual_index::Int
    witness_ordinal::Int
    from_aspect::Symbol
    to_aspect::Symbol
    rule_style::Symbol
    match_path::Vector{Symbol}
    source_path::Vector{Symbol}
    target_path::Vector{Symbol}
    arena_path::Vector{Symbol}
    witness_arena_path::Vector{Symbol}
    trit_delta::Int
    closure_effect::Symbol
    color_hex::String
    semantic_cost::Float64
    fingerprint::UInt64
end

struct LispGATQueryResult
    operation::Symbol
    arguments::Vector{Any}
    matches::Vector{LispGATRewriteCandidate}
    coverage::NamedTuple
    fingerprint::UInt64
    evidence::String
end

struct LispGATRewritePlan
    query::LispGATQueryResult
    sample_ordinals::Vector{Int}
    sample_mode::Symbol
    max_samples::Int
    materialization_backend::Symbol
    bridge_fingerprint::UInt64
    fingerprint::UInt64
    evidence::String
end

struct LispGATRewriteExecution
    plan::LispGATRewritePlan
    backend::Symbol
    materialization_fingerprint::UInt64
    selected_ordinals::Vector{Int}
    materialized_ordinals::Vector{Int}
    executed_ordinals::Vector{Int}
    target_ordinals::Vector{Int}
    spec_count::Int
    materialized_count::Int
    selected_all_materialized::Bool
    selected_all_targets::Bool
    fingerprint::UInt64
    evidence::String
end

struct LispGATRewriteRequest
    operation::Symbol
    arguments::Vector{Any}
    max_samples::Int
    backend::Symbol
    form::Any
    parser::Symbol
    fingerprint::UInt64
    evidence::String
end

struct LispGATRewriteProgram
    requests::Vector{LispGATRewriteRequest}
    form::Any
    parser::Symbol
    fingerprint::UInt64
    evidence::String
end

struct LispGATRewriteProgramExecution
    program::LispGATRewriteProgram
    executions::Vector{LispGATRewriteExecution}
    backends::Vector{Symbol}
    selected_ordinals::Vector{Int}
    all_selected_all_materialized::Bool
    all_selected_all_targets::Bool
    fingerprint::UInt64
    evidence::String
end

struct LispGATRewriteProgramStep
    index::Int
    request::LispGATRewriteRequest
    plan::LispGATRewritePlan
    execution::LispGATRewriteExecution
    selected_ordinals::Vector{Int}
    introduced_ordinals::Vector{Int}
    backend::Symbol
    fingerprint::UInt64
    evidence::String
end

struct LispGATRewriteProgramTrace
    execution::LispGATRewriteProgramExecution
    steps::Vector{LispGATRewriteProgramStep}
    repeated_ordinals::Vector{Int}
    coverage_complete::Bool
    fingerprint::UInt64
    evidence::String
end

struct LispGATBridgeWorld
    seed::UInt64
    source::String
    presentation_name::Symbol
    objects::Vector{LispGATObject}
    morphisms::Vector{LispGATMorphism}
    equations::Vector{LispGATEquation}
    counterfactuals::Vector{LispGATCounterfactual}
    forms::Vector{Any}
    parser::Symbol
    fingerprint::UInt64
end

struct AlgebraicJuliaCapability
    package::Symbol
    uuid::String
    role::String
    available::Bool
    load_path::String
end

struct AlgebraicJuliaRealization
    extension::Symbol
    backend::Symbol
    packages::Vector{Symbol}
    parser::Symbol
    theory_source::String
    presentation_source::String
    acset_hint::String
    rewriting_hint::String
    fingerprint::UInt64
end

struct AlgebraicJuliaMaterialization
    extension::Symbol
    backend::Symbol
    packages::Vector{Symbol}
    presentation::Any
    presentation_type::String
    generator_counts::Dict{Symbol,Int}
    equation_count::Int
    counterfactual_count::Int
    rewrite_candidate_count::Int
    rewrite_candidates::Vector{Any}
    theory_source::String
    presentation_source::String
    fingerprint::UInt64
end

Base.length(w::LispGATBridgeWorld) =
    length(w.objects) + length(w.morphisms) + length(w.equations) + length(w.counterfactuals)
fingerprint(w::LispGATBridgeWorld)::UInt64 = w.fingerprint

function _lisp_gat_sym(x)::Symbol
    x isa Symbol && return Symbol(replace(String(x), "-" => "_"))
    x isa AbstractString && return Symbol(replace(String(x), "-" => "_"))
    error("Expected symbol or string in Lisp/GAT form, got $(typeof(x))")
end

function _lisp_gat_color(seed::UInt64, tag, index::Integer)::String
    color_index = stable_seed((:lisp_gatlab_bridge, tag, index); seed=seed) % UInt64(65536) + UInt64(1)
    c = color_at(Int(color_index); seed=seed)
    rgb_hex(c.r, c.g, c.b)
end

function _lisp_gat_path(expr)::Vector{Symbol}
    if expr isa Symbol || expr isa AbstractString
        return [_lisp_gat_sym(expr)]
    elseif expr isa Vector
        isempty(expr) && error("Empty equation path")
        head = _lisp_gat_sym(expr[1])
        head == :compose || error("Equation path must be a (compose ...) form, got $head")
        return [_lisp_gat_sym(x) for x in expr[2:end]]
    else
        error("Expected equation path, got $(typeof(expr))")
    end
end

function _lisp_gat_normalize_form(form)
    if form isa AbstractString
        try
            return (form=LispSyntax.desx(LispSyntax.read(form)), parser=:lispsyntax)
        catch lispsyntax_error
            try
                return (form=sexp_read(form), parser=:sexp_fallback)
            catch sexp_error
                error("Could not parse Lisp/GAT form with LispSyntax or SExp fallback: ",
                    lispsyntax_error, " / ", sexp_error)
            end
        end
    end

    try
        return (form=LispSyntax.desx(form), parser=:lispsyntax_desx)
    catch
        return (form=form, parser=:preparsed)
    end
end

lisp_gatlab_lispsyntax_form(form=default_lisp_gatlab_form()) =
    _lisp_gat_normalize_form(form).form

lisp_gatlab_parse_backend(form=default_lisp_gatlab_form()) =
    _lisp_gat_normalize_form(form).parser

function default_lisp_gatlab_form()
    """
    (gat
      (ob TestWitness)
      (ob ClosureAspect)
      (ob CounterfactualAssignment)
      (ob CatColabDecl)
      (ob LispForm)
      (ob SharedArena)
      (attrtype Color)
      (attrtype Trit)
      (attrtype Cost)
      (attrtype CounterfactualEffect)
      (attrtype ScipAddress)
      (hom has-aspect TestWitness ClosureAspect)
      (hom has-counterfactual TestWitness CounterfactualAssignment)
      (hom from-aspect CounterfactualAssignment ClosureAspect)
      (hom to-aspect CounterfactualAssignment ClosureAspect)
      (hom declares-object ClosureAspect CatColabDecl)
      (hom as-declared-object CounterfactualAssignment CatColabDecl)
      (hom observed-as LispForm TestWitness)
      (hom language-assigns-aspect LispForm ClosureAspect)
      (hom shared-in CounterfactualAssignment SharedArena)
      (hom witness-arena TestWitness SharedArena)
      (attr witness-color TestWitness Color)
      (attr aspect-trit ClosureAspect Trit)
      (attr counterfactual-cost CounterfactualAssignment Cost)
      (attr counterfactual-effect CounterfactualAssignment CounterfactualEffect)
      (attr scip-uri CatColabDecl ScipAddress)
      (eq (compose has-counterfactual from-aspect)
          (compose has-aspect))
      (eq (compose has-counterfactual to-aspect declares-object)
          (compose has-counterfactual as-declared-object))
      (eq (compose observed-as has-aspect)
          (compose language-assigns-aspect))
      (eq (compose has-counterfactual shared-in)
          (compose witness-arena)))
    """
end

function default_lisp_gatlab_rewrite_form()
    """
    (rewrite-execution
      (query witness 1)
      (max-samples 2)
      (backend algebraicjulia))
    """
end

function default_lisp_gatlab_rewrite_program_form()
    """
    (rewrite-program
      (rewrite-execution
        (query witness 1)
        (max-samples 2)
        (backend projection))
      (rewrite-execution
        (query effect positive-shift)
        (max-samples 1)
        (backend projection))
      (rewrite-execution
        (query ordinal 1)
        (max-samples 1)
        (backend projection)))
    """
end

function default_lisp_gatlab_rewrite_trace_form()
    program = parse_lisp_gatlab_rewrite_program(default_lisp_gatlab_rewrite_program_form())
    trace = lisp_gatlab_rewrite_program_trace(program)
    render_lisp_gatlab_rewrite_program_trace(trace)
end

function _lisp_gat_rewrite_request_fingerprint(
    seed::UInt64,
    operation::Symbol,
    arguments::Vector{Any},
    max_samples::Integer,
    backend::Symbol,
    parser::Symbol,
)::UInt64
    stable_seed((
        :lisp_gat_rewrite_request,
        operation,
        map(string, arguments),
        max_samples,
        backend,
        parser,
    ); seed=seed)
end

function _lisp_gat_rewrite_program_fingerprint(
    seed::UInt64,
    requests::Vector{LispGATRewriteRequest},
    parser::Symbol,
)::UInt64
    stable_seed((
        :lisp_gat_rewrite_program,
        [request.fingerprint for request in requests],
    ); seed=seed)
end

function lisp_gatlab_rewrite_request(
    operation,
    arguments...;
    max_samples::Integer=1,
    backend::Symbol=:algebraicjulia,
    seed::Integer=GAY_SEED,
)
    op = _lisp_gat_sym(operation)
    args = Any[arguments...]
    n = max(0, Int(max_samples))
    backend_sym = _lisp_gat_sym(backend)
    seed64 = UInt64(seed)
    LispGATRewriteRequest(
        op,
        args,
        n,
        backend_sym,
        Any[:rewrite_execution, Any[:query, op, args...], Any[:max_samples, n], Any[:backend, backend_sym]],
        :constructed,
        _lisp_gat_rewrite_request_fingerprint(seed64, op, args, n, backend_sym, :constructed),
        "constructed Lisp/GATlab rewrite request",
    )
end

function _lisp_gat_form_entry(head::Symbol, entries)
    for entry in entries
        entry isa Vector || continue
        isempty(entry) && continue
        _lisp_gat_sym(first(entry)) == head && return entry
    end
    nothing
end

function _lisp_gat_request_int(entry, name::Symbol)
    entry === nothing && error("Missing $name entry in Lisp/GATlab rewrite request")
    length(entry) == 2 || error("$name expects one integer")
    _lisp_gat_query_int(entry[2])
end

function _lisp_gat_request_backend(entry)
    entry === nothing && return :algebraicjulia
    length(entry) == 2 || error("backend expects one symbol")
    _lisp_gat_sym(entry[2])
end

function parse_lisp_gatlab_rewrite_form(form=default_lisp_gatlab_rewrite_form(); seed::Integer=GAY_SEED)
    seed64 = UInt64(seed)
    normalized = _lisp_gat_normalize_form(form)
    parsed = normalized.form
    parsed isa Vector || error("Lisp/GATlab rewrite form must parse to a list")
    !isempty(parsed) || error("Lisp/GATlab rewrite form cannot be empty")

    head = _lisp_gat_sym(parsed[1])
    entries = parsed[2:end]
    query_entry = if head == :query
        parsed
    elseif head in (:rewrite_execution, :rewrite_plan, :rewrite_request, :lisp_gatlab_rewrite)
        _lisp_gat_form_entry(:query, entries)
    else
        error("Lisp/GATlab rewrite form must start with query or rewrite-execution")
    end

    query_entry === nothing && error("Lisp/GATlab rewrite request requires a (query ...) entry")
    length(query_entry) >= 2 || error("query expects an operation")
    operation = _lisp_gat_sym(query_entry[2])
    arguments = Any[query_entry[3:end]...]
    max_samples = head == :query ? 1 : _lisp_gat_request_int(_lisp_gat_form_entry(:max_samples, entries), :max_samples)
    backend = head == :query ? :algebraicjulia : _lisp_gat_request_backend(_lisp_gat_form_entry(:backend, entries))

    LispGATRewriteRequest(
        operation,
        arguments,
        max(0, Int(max_samples)),
        backend,
        parsed,
        normalized.parser,
        _lisp_gat_rewrite_request_fingerprint(
            seed64,
            operation,
            arguments,
            max_samples,
            backend,
            normalized.parser,
        ),
        "parsed jank-like LispSyntax rewrite request over GATlab candidates",
    )
end

function parse_lisp_gatlab_rewrite_program(form=default_lisp_gatlab_rewrite_program_form(); seed::Integer=GAY_SEED)
    seed64 = UInt64(seed)
    normalized = _lisp_gat_normalize_form(form)
    parsed = normalized.form
    parsed isa Vector || error("Lisp/GATlab rewrite program must parse to a list")
    !isempty(parsed) || error("Lisp/GATlab rewrite program cannot be empty")

    head = _lisp_gat_sym(parsed[1])
    entries = if head in (:rewrite_program, :lisp_gatlab_rewrite_program)
        parsed[2:end]
    elseif head in (:query, :rewrite_execution, :rewrite_plan, :rewrite_request, :lisp_gatlab_rewrite)
        Any[parsed]
    else
        error("Lisp/GATlab rewrite program must start with rewrite-program, query, or rewrite-execution")
    end

    requests = LispGATRewriteRequest[]
    for (i, entry) in enumerate(entries)
        entry isa Vector || error("rewrite-program entry $i must be a query or rewrite request list")
        isempty(entry) && continue
        push!(requests, parse_lisp_gatlab_rewrite_form(entry; seed=seed64))
    end
    !isempty(requests) || error("Lisp/GATlab rewrite program must contain at least one request")

    LispGATRewriteProgram(
        requests,
        parsed,
        normalized.parser,
        _lisp_gat_rewrite_program_fingerprint(seed64, requests, normalized.parser),
        "parsed jank-like LispSyntax rewrite program over ordered GATlab candidates",
    )
end

function parse_lisp_gatlab_form(form; seed::Integer=GAY_SEED)
    seed64 = UInt64(seed)
    normalized = _lisp_gat_normalize_form(form)
    parsed = normalized.form
    parsed isa Vector || error("Lisp/GAT form must parse to a list")
    !isempty(parsed) || error("Lisp/GAT form cannot be empty")
    _lisp_gat_sym(parsed[1]) == :gat || error("Lisp/GAT form must start with gat")

    objects = LispGATObject[]
    morphisms = LispGATMorphism[]
    equations = LispGATEquation[]

    for (i, entry) in enumerate(parsed[2:end])
        entry isa Vector || error("Lisp/GAT entry $i must be a list")
        !isempty(entry) || continue
        head = _lisp_gat_sym(entry[1])
        if head == :ob || head == :attrtype
            length(entry) == 2 || error("$head expects one name")
            name = _lisp_gat_sym(entry[2])
            kind = head == :ob ? :ob : :attrtype
            push!(objects, LispGATObject(
                name,
                kind,
                _lisp_gat_color(seed64, (kind, name), i),
                kind == :ob ? "GAT object declaration from jank-like Lisp form" :
                    "GAT attribute type declaration from jank-like Lisp form",
            ))
        elseif head == :hom || head == :attr
            length(entry) == 4 || error("$head expects name, domain, and codomain")
            name = _lisp_gat_sym(entry[2])
            dom = _lisp_gat_sym(entry[3])
            cod = _lisp_gat_sym(entry[4])
            kind = head == :hom ? :hom : :attr
            push!(morphisms, LispGATMorphism(
                name,
                kind,
                dom,
                cod,
                _lisp_gat_color(seed64, (kind, name, dom, cod), i),
                kind == :hom ? "structure-preserving map in the shared arena" :
                    "attribute map carrying Gay.jl color or semantic metadata",
            ))
        elseif head == :eq
            length(entry) == 3 || error("eq expects lhs and rhs paths")
            lhs = _lisp_gat_path(entry[2])
            rhs = _lisp_gat_path(entry[3])
            push!(equations, LispGATEquation(
                lhs,
                rhs,
                _lisp_gat_color(seed64, (:eq, lhs, rhs), i),
                "counterfactual bisimulation path equation",
            ))
        else
            error("Unknown Lisp/GAT declaration head: $head")
        end
    end

    (objects=objects, morphisms=morphisms, equations=equations, form=parsed, parser=normalized.parser)
end

function _lisp_gat_counterfactuals(w::GayTestOlogCounterfactualWorld)
    [
        LispGATCounterfactual(
            cf.witness_ordinal,
            cf.from_aspect,
            cf.to_aspect,
            cf.trit_delta,
            cf.closure_effect,
            cf.counterfactual_color_hex,
            cf.semantic_cost,
            [:has_counterfactual, :from_aspect],
            [:has_aspect],
        )
        for cf in w.counterfactuals
    ]
end

function _lisp_gat_rewrite_candidate_fingerprint(
    seed::UInt64,
    ordinal::Integer,
    cf::LispGATCounterfactual,
    rule_style::Symbol,
    match_path::Vector{Symbol},
    source_path::Vector{Symbol},
    target_path::Vector{Symbol},
    arena_path::Vector{Symbol},
    witness_arena_path::Vector{Symbol},
)::UInt64
    stable_seed((
        :lisp_gat_rewrite_candidate,
        ordinal,
        cf.witness_ordinal,
        cf.from_aspect,
        cf.to_aspect,
        rule_style,
        match_path,
        source_path,
        target_path,
        arena_path,
        witness_arena_path,
        cf.trit_delta,
        cf.closure_effect,
        cf.color_hex,
        cf.semantic_cost,
    ); seed=seed)
end

function lisp_gatlab_rewrite_candidates(w::LispGATBridgeWorld=world_lisp_gatlab_bridge())
    map(enumerate(w.counterfactuals)) do (i, cf)
        match_path = [:has_aspect]
        source_path = [:has_counterfactual, :from_aspect]
        target_path = [:has_counterfactual, :to_aspect]
        arena_path = [:has_counterfactual, :shared_in]
        witness_arena_path = [:witness_arena]
        rule_style = :colored_bisimulation_candidate
        LispGATRewriteCandidate(
            i,
            i,
            cf.witness_ordinal,
            cf.from_aspect,
            cf.to_aspect,
            rule_style,
            match_path,
            source_path,
            target_path,
            arena_path,
            witness_arena_path,
            cf.trit_delta,
            cf.closure_effect,
            cf.color_hex,
            cf.semantic_cost,
            _lisp_gat_rewrite_candidate_fingerprint(
                w.seed,
                i,
                cf,
                rule_style,
                match_path,
                source_path,
                target_path,
                arena_path,
                witness_arena_path,
            ),
        )
    end
end

function lisp_gatlab_counterfactual_coverage(w::LispGATBridgeWorld=world_lisp_gatlab_bridge())
    candidates = lisp_gatlab_rewrite_candidates(w)
    by_witness = Dict{Int,Int}()
    by_edge = Dict{Tuple{Int,Symbol,Symbol},Int}()
    for cand in candidates
        by_witness[cand.witness_ordinal] = get(by_witness, cand.witness_ordinal, 0) + 1
        key = (cand.witness_ordinal, cand.from_aspect, cand.to_aspect)
        by_edge[key] = get(by_edge, key, 0) + 1
    end
    fingerprints = Set(cand.fingerprint for cand in candidates)
    witness_counts = collect(values(by_witness))
    expected_per_witness = isempty(witness_counts) ? 0 : maximum(witness_counts)
    (
        counterfactuals=length(w.counterfactuals),
        rewrite_candidates=length(candidates),
        unique_rewrite_candidates=length(fingerprints),
        witnesses=length(by_witness),
        expected_per_witness=expected_per_witness,
        min_per_witness=isempty(witness_counts) ? 0 : minimum(witness_counts),
        max_per_witness=isempty(witness_counts) ? 0 : maximum(witness_counts),
        no_duplicate_edges=all(==(1), values(by_edge)),
        complete=length(candidates) == length(w.counterfactuals) &&
            length(fingerprints) == length(candidates) &&
            all(==(expected_per_witness), witness_counts) &&
            all(==(1), values(by_edge)),
    )
end

function _lisp_gat_query_int(x)::Int
    x isa Integer && return Int(x)
    x isa AbstractString && return parse(Int, x)
    x isa Symbol && return parse(Int, String(x))
    error("Expected integer query argument, got $(typeof(x))")
end

function _lisp_gat_query_hex(x)::String
    s = x isa Symbol ? String(x) : string(x)
    startswith(s, "#") ? lowercase(s) : lowercase(string("#", s))
end

function _lisp_gat_query_matches(
    w::LispGATBridgeWorld,
    operation::Symbol,
    arguments::Vector{Any},
)::Vector{LispGATRewriteCandidate}
    candidates = lisp_gatlab_rewrite_candidates(w)

    if operation in (:all, :rewrite_candidates)
        return candidates
    elseif operation in (:candidate, :ordinal)
        length(arguments) == 1 || error("$operation expects one ordinal")
        ordinal = _lisp_gat_query_int(only(arguments))
        return [cand for cand in candidates if cand.ordinal == ordinal]
    elseif operation == :limit
        length(arguments) == 1 || error("limit expects one count")
        n = clamp(_lisp_gat_query_int(only(arguments)), 0, length(candidates))
        return collect(Iterators.take(candidates, n))
    elseif operation == :witness
        length(arguments) == 1 || error("witness expects one witness ordinal")
        witness = _lisp_gat_query_int(only(arguments))
        return [cand for cand in candidates if cand.witness_ordinal == witness]
    elseif operation == :effect
        length(arguments) == 1 || error("effect expects one closure effect")
        effect = _lisp_gat_sym(only(arguments))
        return [cand for cand in candidates if cand.closure_effect == effect]
    elseif operation == :from
        length(arguments) == 1 || error("from expects one source aspect")
        aspect = _lisp_gat_sym(only(arguments))
        return [cand for cand in candidates if cand.from_aspect == aspect]
    elseif operation == :to
        length(arguments) == 1 || error("to expects one target aspect")
        aspect = _lisp_gat_sym(only(arguments))
        return [cand for cand in candidates if cand.to_aspect == aspect]
    elseif operation == :between
        length(arguments) == 2 || error("between expects source and target aspects")
        from_aspect = _lisp_gat_sym(arguments[1])
        to_aspect = _lisp_gat_sym(arguments[2])
        return [
            cand for cand in candidates
            if cand.from_aspect == from_aspect && cand.to_aspect == to_aspect
        ]
    elseif operation == :color
        length(arguments) == 1 || error("color expects one hex color")
        color_hex = _lisp_gat_query_hex(only(arguments))
        return [cand for cand in candidates if lowercase(cand.color_hex) == color_hex]
    else
        error("Unknown Lisp/GATlab query operation: $operation")
    end
end

function _lisp_gat_query_fingerprint(
    w::LispGATBridgeWorld,
    operation::Symbol,
    arguments::Vector{Any},
    matches::Vector{LispGATRewriteCandidate},
)::UInt64
    stable_seed((
        :lisp_gat_query_result,
        w.fingerprint,
        operation,
        map(string, arguments),
        [cand.fingerprint for cand in matches],
    ); seed=w.seed)
end

function lisp_gatlab_query(
    w::LispGATBridgeWorld,
    operation,
    arguments...,
)
    op = _lisp_gat_sym(operation)
    args = Any[arguments...]
    matches = _lisp_gat_query_matches(w, op, args)
    coverage = lisp_gatlab_counterfactual_coverage(w)
    LispGATQueryResult(
        op,
        args,
        matches,
        coverage,
        _lisp_gat_query_fingerprint(w, op, args, matches),
        "jank-like LispSyntax query over color-preserving GATlab rewrite candidates",
    )
end

lisp_gatlab_query(operation, arguments...) =
    lisp_gatlab_query(world_lisp_gatlab_bridge(), operation, arguments...)

lisp_gatlab_query(w::LispGATBridgeWorld, request::LispGATRewriteRequest) =
    lisp_gatlab_query(w, request.operation, request.arguments...)

lisp_gatlab_query(request::LispGATRewriteRequest) =
    lisp_gatlab_query(world_lisp_gatlab_bridge(), request)

function _lisp_gat_rewrite_plan_fingerprint(
    w::LispGATBridgeWorld,
    query::LispGATQueryResult,
    sample_ordinals::Vector{Int},
    sample_mode::Symbol,
    max_samples::Integer,
    materialization_backend::Symbol,
)::UInt64
    stable_seed((
        :lisp_gat_rewrite_plan,
        w.fingerprint,
        query.fingerprint,
        sample_ordinals,
        sample_mode,
        max_samples,
        materialization_backend,
    ); seed=w.seed)
end

function lisp_gatlab_rewrite_plan(
    w::LispGATBridgeWorld,
    operation,
    arguments...;
    max_samples::Integer=1,
    materialization_backend::Symbol=:algebraicjulia,
)
    query = lisp_gatlab_query(w, operation, arguments...)
    n = clamp(Int(max_samples), 0, length(query.matches))
    sample_ordinals = [cand.ordinal for cand in Iterators.take(query.matches, n)]
    sample_mode = :ordinals
    LispGATRewritePlan(
        query,
        sample_ordinals,
        sample_mode,
        n,
        materialization_backend,
        w.fingerprint,
        _lisp_gat_rewrite_plan_fingerprint(
            w,
            query,
            sample_ordinals,
            sample_mode,
            n,
            materialization_backend,
        ),
        "query-selected exact DPO ordinals for AlgebraicJulia materialization",
    )
end

lisp_gatlab_rewrite_plan(operation, arguments...; kwargs...) =
    lisp_gatlab_rewrite_plan(world_lisp_gatlab_bridge(), operation, arguments...; kwargs...)

function lisp_gatlab_rewrite_plan(
    w::LispGATBridgeWorld,
    request::LispGATRewriteRequest;
    max_samples::Integer=request.max_samples,
    materialization_backend::Symbol=request.backend,
)
    lisp_gatlab_rewrite_plan(
        w,
        request.operation,
        request.arguments...;
        max_samples=max_samples,
        materialization_backend=materialization_backend,
    )
end

lisp_gatlab_rewrite_plan(request::LispGATRewriteRequest; kwargs...) =
    lisp_gatlab_rewrite_plan(world_lisp_gatlab_bridge(), request; kwargs...)

function _lisp_gat_fingerprint(
    seed::UInt64,
    source::AbstractString,
    presentation_name::Symbol,
    parser::Symbol,
    objects,
    morphisms,
    equations,
    counterfactuals,
)::UInt64
    fp = stable_seed(("lisp-gatlab-bridge", source, presentation_name, parser); seed=seed)
    for ob in objects
        fp = xor(fp, stable_seed((ob.name, ob.kind, ob.color_hex, ob.evidence); seed=seed))
    end
    for mor in morphisms
        fp = xor(fp, stable_seed((mor.name, mor.kind, mor.dom, mor.cod, mor.color_hex); seed=seed))
    end
    for eq in equations
        fp = xor(fp, stable_seed((eq.lhs, eq.rhs, eq.color_hex); seed=seed))
    end
    for cf in counterfactuals
        fp = xor(fp, stable_seed((
            cf.witness_ordinal,
            cf.from_aspect,
            cf.to_aspect,
            cf.trit_delta,
            cf.closure_effect,
            cf.color_hex,
            cf.semantic_cost,
        ); seed=seed))
    end
    fp
end

function world_lisp_gatlab_bridge(;
    seed::Integer=GAY_SEED,
    presentation_name::Symbol=:SchGayCounterfactualClosure,
    source::AbstractString="Gay.jl jank-like LispSyntax bridge to GATlab/Catlab presentation",
    form=default_lisp_gatlab_form(),
    olog_world::GayTestOlogWorld=world_gay_test_olog(; seed=seed),
)
    seed64 = UInt64(seed)
    parsed = parse_lisp_gatlab_form(form; seed=seed64)
    cf_world = world_gay_test_olog_counterfactuals(olog_world)
    cfs = _lisp_gat_counterfactuals(cf_world)
    fp = _lisp_gat_fingerprint(
        seed64,
        source,
        presentation_name,
        parsed.parser,
        parsed.objects,
        parsed.morphisms,
        parsed.equations,
        cfs,
    )
    LispGATBridgeWorld(
        seed64,
        String(source),
        presentation_name,
        parsed.objects,
        parsed.morphisms,
        parsed.equations,
        cfs,
        Any[parsed.form],
        parsed.parser,
        fp,
    )
end

function lisp_gatlab_bridge_summary(w::LispGATBridgeWorld=world_lisp_gatlab_bridge())
    effect_counts = Dict{Symbol,Int}()
    for cf in w.counterfactuals
        effect_counts[cf.closure_effect] = get(effect_counts, cf.closure_effect, 0) + 1
    end
    coverage = lisp_gatlab_counterfactual_coverage(w)
    (
        presentation_name=w.presentation_name,
        objects=length(w.objects),
        morphisms=length(w.morphisms),
        equations=length(w.equations),
        counterfactuals=length(w.counterfactuals),
        rewrite_candidates=coverage.rewrite_candidates,
        unique_rewrite_candidates=coverage.unique_rewrite_candidates,
        effect_counts=effect_counts,
        all_counterfactuals_considered=length(w.counterfactuals) ==
            length(world_gay_test_olog_counterfactuals().counterfactuals),
        all_rewrite_candidates_considered=coverage.complete,
        parser=w.parser,
        fingerprint=w.fingerprint,
    )
end

function lisp_gatlab_declarations(w::LispGATBridgeWorld=world_lisp_gatlab_bridge())
    (
        objects=[
            (name=ob.name, kind=ob.kind, color_hex=ob.color_hex, evidence=ob.evidence)
            for ob in w.objects
        ],
        morphisms=[
            (
                name=mor.name,
                kind=mor.kind,
                dom=mor.dom,
                cod=mor.cod,
                color_hex=mor.color_hex,
                evidence=mor.evidence,
            )
            for mor in w.morphisms
        ],
        equations=[
            (lhs=eq.lhs, rhs=eq.rhs, color_hex=eq.color_hex, evidence=eq.evidence)
            for eq in w.equations
        ],
        counterfactuals=[
            (
                witness_ordinal=cf.witness_ordinal,
                from_aspect=cf.from_aspect,
                to_aspect=cf.to_aspect,
                trit_delta=cf.trit_delta,
                closure_effect=cf.closure_effect,
                color_hex=cf.color_hex,
                semantic_cost=cf.semantic_cost,
                lhs=cf.lhs,
                rhs=cf.rhs,
            )
            for cf in w.counterfactuals
        ],
        rewrite_candidates=[
            (
                ordinal=cand.ordinal,
                counterfactual_index=cand.counterfactual_index,
                witness_ordinal=cand.witness_ordinal,
                from_aspect=cand.from_aspect,
                to_aspect=cand.to_aspect,
                rule_style=cand.rule_style,
                match_path=cand.match_path,
                source_path=cand.source_path,
                target_path=cand.target_path,
                arena_path=cand.arena_path,
                witness_arena_path=cand.witness_arena_path,
                trit_delta=cand.trit_delta,
                closure_effect=cand.closure_effect,
                color_hex=cand.color_hex,
                semantic_cost=cand.semantic_cost,
                fingerprint=cand.fingerprint,
            )
            for cand in lisp_gatlab_rewrite_candidates(w)
        ],
    )
end

function render_lisp_gatlab_presentation(w::LispGATBridgeWorld=world_lisp_gatlab_bridge())
    io = IOBuffer()
    println(io, "# Generated projection artifact; requires Catlab/GATlab to execute.")
    println(io, "# fingerprint: 0x", string(w.fingerprint, base=16, pad=16))
    println(io, "@present ", w.presentation_name, "(FreeSchema) begin")
    for ob in w.objects
        if ob.kind == :attrtype
            println(io, "    ", ob.name, "::AttrType")
        else
            println(io, "    ", ob.name, "::Ob")
        end
    end
    println(io)
    for mor in w.morphisms
        if mor.kind == :attr
            println(io, "    ", mor.name, "::Attr(", mor.dom, ", ", mor.cod, ")")
        else
            println(io, "    ", mor.name, "::Hom(", mor.dom, ", ", mor.cod, ")")
        end
    end
    println(io)
    for (i, eq) in enumerate(w.equations)
        println(io, "    # equation ", i, ": ", join(string.(eq.lhs), " ; "), " == ", join(string.(eq.rhs), " ; "))
    end
    println(io, "end")
    String(take!(io))
end

function _lisp_gat_sexp_quote(s::AbstractString)
    escaped = replace(String(s), "\\" => "\\\\", "\"" => "\\\"")
    string("\"", escaped, "\"")
end

function _lisp_gat_render_path(path::Vector{Symbol})
    string("(compose ", join(string.(path), " "), ")")
end

function render_lisp_gatlab_bridge(
    w::LispGATBridgeWorld=world_lisp_gatlab_bridge();
    counterfactual_limit::Union{Nothing,Integer}=nothing,
)
    limit = counterfactual_limit === nothing ? length(w.counterfactuals) :
        clamp(Int(counterfactual_limit), 0, length(w.counterfactuals))
    io = IOBuffer()
    println(io, "(lisp-gatlab-bridge")
    println(io, "  (:presentation ", w.presentation_name, ")")
    println(io, "  (:fingerprint ", _lisp_gat_sexp_quote(string("0x", string(w.fingerprint, base=16, pad=16))), ")")
    println(io, "  (:source ", _lisp_gat_sexp_quote(w.source), ")")
    println(io, "  (:parser ", w.parser, ")")
    println(io, "  (:counts (:objects ", length(w.objects), ") (:morphisms ", length(w.morphisms),
        ") (:equations ", length(w.equations), ") (:counterfactuals ", length(w.counterfactuals),
        ") (:rewrite-candidates ", length(lisp_gatlab_rewrite_candidates(w)), "))")
    println(io, "  (:objects")
    for ob in w.objects
        println(io, "    (:object (:name ", ob.name, ") (:kind ", ob.kind, ") (:color ",
            _lisp_gat_sexp_quote(ob.color_hex), "))")
    end
    println(io, "  )")
    println(io, "  (:morphisms")
    for mor in w.morphisms
        println(io, "    (:morphism (:name ", mor.name, ") (:kind ", mor.kind, ") (:dom ", mor.dom,
            ") (:cod ", mor.cod, ") (:color ", _lisp_gat_sexp_quote(mor.color_hex), "))")
    end
    println(io, "  )")
    println(io, "  (:equations")
    for eq in w.equations
        println(io, "    (:equation (:lhs ", _lisp_gat_render_path(eq.lhs), ") (:rhs ",
            _lisp_gat_render_path(eq.rhs), ") (:color ", _lisp_gat_sexp_quote(eq.color_hex), "))")
    end
    println(io, "  )")
    println(io, "  (:counterfactuals")
    for cf in Iterators.take(w.counterfactuals, limit)
        println(io, "    (:counterfactual (:witness ", cf.witness_ordinal, ") (:from ",
            cf.from_aspect, ") (:to ", cf.to_aspect, ") (:delta ", cf.trit_delta,
            ") (:effect ", cf.closure_effect, ") (:cost ", cf.semantic_cost, ") (:color ",
            _lisp_gat_sexp_quote(cf.color_hex), "))")
    end
    if limit < length(w.counterfactuals)
        println(io, "    (:truncated ", length(w.counterfactuals) - limit, ")")
    end
    println(io, "  )")
    println(io, "  (:rewrite-candidates")
    for cand in Iterators.take(lisp_gatlab_rewrite_candidates(w), limit)
        println(io, "    (:rewrite-candidate (:ordinal ", cand.ordinal,
            ") (:counterfactual-index ", cand.counterfactual_index,
            ") (:witness ", cand.witness_ordinal,
            ") (:from ", cand.from_aspect,
            ") (:to ", cand.to_aspect,
            ") (:style ", cand.rule_style,
            ") (:source ", _lisp_gat_render_path(cand.source_path),
            ") (:target ", _lisp_gat_render_path(cand.target_path),
            ") (:arena ", _lisp_gat_render_path(cand.arena_path),
            ") (:fingerprint ",
            _lisp_gat_sexp_quote(string("0x", string(cand.fingerprint, base=16, pad=16))), "))")
    end
    if limit < length(w.counterfactuals)
        println(io, "    (:truncated ", length(w.counterfactuals) - limit, ")")
    end
    println(io, "  ))")
    String(take!(io))
end

lisp_gatlab_bridge() = render_lisp_gatlab_bridge(world_lisp_gatlab_bridge())

function lisp_gatlab_compile(form=default_lisp_gatlab_form(); target::Symbol=:world)
    parsed = parse_lisp_gatlab_form(form)
    form_literal = form isa AbstractString ? String(form) : parsed.form
    form_node = QuoteNode(form_literal)
    world_expr = :(world_lisp_gatlab_bridge(; form=$form_node))

    if target == :world
        world_expr
    elseif target == :summary
        :(lisp_gatlab_bridge_summary($world_expr))
    elseif target == :declarations
        :(lisp_gatlab_declarations($world_expr))
    elseif target == :json
        :(render_lisp_gatlab_json($world_expr))
    elseif target == :gatlab_theory
        :(render_lisp_gatlab_theory($world_expr))
    elseif target == :catlab_present
        :(render_lisp_gatlab_presentation($world_expr))
    elseif target == :sexp
        :(render_lisp_gatlab_bridge($world_expr))
    else
        error("Unknown Lisp/GAT compiler target: $target")
    end
end

macro gat_str(str)
    lisp_gatlab_compile(str; target=:world)
end

function lisp_gatlab_rewrite_compile(form=default_lisp_gatlab_rewrite_form(); target::Symbol=:request)
    parsed = parse_lisp_gatlab_rewrite_form(form)
    form_literal = form isa AbstractString ? String(form) : parsed.form
    form_node = QuoteNode(form_literal)
    request_expr = :(parse_lisp_gatlab_rewrite_form($form_node))

    if target == :request
        request_expr
    elseif target == :query
        :(lisp_gatlab_query($request_expr))
    elseif target == :plan
        :(lisp_gatlab_rewrite_plan($request_expr))
    elseif target == :execution
        :(lisp_gatlab_rewrite_execution($request_expr))
    elseif target == :request_json
        :(render_lisp_gatlab_rewrite_request_json($request_expr))
    elseif target in (:request_form, :request_sexp, :form, :sexp)
        :(render_lisp_gatlab_rewrite_request($request_expr))
    elseif target == :execution_json
        quote
            local request = $request_expr
            local execution = lisp_gatlab_rewrite_execution(request)
            render_lisp_gatlab_rewrite_execution_json(execution; request=request)
        end
    else
        error("Unknown Lisp/GAT rewrite compiler target: $target")
    end
end

macro gat_rewrite_str(str)
    lisp_gatlab_rewrite_compile(str; target=:request)
end

function lisp_gatlab_rewrite_program_compile(
    form=default_lisp_gatlab_rewrite_program_form();
    target::Symbol=:program,
)
    parsed = parse_lisp_gatlab_rewrite_program(form)
    form_literal = form isa AbstractString ? String(form) : parsed.form
    form_node = QuoteNode(form_literal)
    program_expr = :(parse_lisp_gatlab_rewrite_program($form_node))

    if target == :program
        program_expr
    elseif target == :execution
        :(lisp_gatlab_rewrite_program_execution($program_expr))
    elseif target == :trace
        :(lisp_gatlab_rewrite_program_trace($program_expr))
    elseif target == :program_json
        :(render_lisp_gatlab_rewrite_program_json($program_expr))
    elseif target in (:program_form, :program_sexp, :form, :sexp)
        :(render_lisp_gatlab_rewrite_program($program_expr))
    elseif target == :execution_json
        quote
            local program = $program_expr
            local execution = lisp_gatlab_rewrite_program_execution(program)
            render_lisp_gatlab_rewrite_program_execution_json(execution)
        end
    elseif target == :trace_json
        quote
            local program = $program_expr
            local trace = lisp_gatlab_rewrite_program_trace(program)
            render_lisp_gatlab_rewrite_program_trace_json(trace)
        end
    elseif target in (:trace_form, :trace_sexp)
        quote
            local program = $program_expr
            local trace = lisp_gatlab_rewrite_program_trace(program)
            render_lisp_gatlab_rewrite_program_trace(trace)
        end
    else
        error("Unknown Lisp/GAT rewrite program compiler target: $target")
    end
end

macro gat_rewrite_program_str(str)
    lisp_gatlab_rewrite_program_compile(str; target=:program)
end

function lisp_gatlab_rewrite_trace_compile(
    form=default_lisp_gatlab_rewrite_trace_form();
    target::Symbol=:parsed,
)
    parsed = parse_lisp_gatlab_rewrite_program_trace_form(form)
    form_literal = form isa AbstractString ? String(form) : parsed.form
    form_node = QuoteNode(form_literal)
    parsed_expr = :(parse_lisp_gatlab_rewrite_program_trace_form($form_node))
    validation_expr = :(validate_lisp_gatlab_rewrite_program_trace_form($form_node))

    if target in (:parsed, :trace)
        parsed_expr
    elseif target == :program
        :($parsed_expr.program)
    elseif target == :validation
        validation_expr
    elseif target in (:validation_payload, :payload)
        :(lisp_gatlab_rewrite_trace_validation_payload($validation_expr))
    elseif target == :validation_json
        :(render_lisp_gatlab_rewrite_trace_validation_json($validation_expr))
    elseif target in (:trace_form, :trace_sexp, :canonical_form, :form, :sexp)
        quote
            local validation = $validation_expr
            render_lisp_gatlab_rewrite_program_trace(validation.trace)
        end
    else
        error("Unknown Lisp/GAT rewrite trace compiler target: $target")
    end
end

macro gat_rewrite_trace_str(str)
    lisp_gatlab_rewrite_trace_compile(str; target=:parsed)
end

function _lisp_gat_json_quote(x)
    s = replace(String(x), "\\" => "\\\\", "\"" => "\\\"", "\n" => "\\n", "\r" => "\\r", "\t" => "\\t")
    string("\"", s, "\"")
end

function _lisp_gat_json_array(xs)
    string("[", join((_lisp_gat_json_quote(string(x)) for x in xs), ", "), "]")
end

function _lisp_gat_hex64(x::UInt64)
    string("0x", string(x, base=16, pad=16))
end

_lisp_gat_json_value(x::AbstractString) = _lisp_gat_json_quote(x)
_lisp_gat_json_value(x::Symbol) = _lisp_gat_json_quote(String(x))
_lisp_gat_json_value(x::Bool) = x ? "true" : "false"
_lisp_gat_json_value(::Nothing) = "null"
_lisp_gat_json_value(x::Integer) = string(x)
_lisp_gat_json_value(x::AbstractFloat) = isfinite(x) ? string(x) : _lisp_gat_json_quote(string(x))
_lisp_gat_json_value(xs::AbstractVector) =
    string("[", join((_lisp_gat_json_value(x) for x in xs), ", "), "]")

function _lisp_gat_json_value(d::AbstractDict)
    pairs = sort(collect(d); by=x -> String(first(x)))
    inner = join((string(_lisp_gat_json_quote(k), ": ", _lisp_gat_json_value(v)) for (k, v) in pairs), ", ")
    string("{", inner, "}")
end

function _lisp_gat_render_json_payload(payload::AbstractDict)
    pairs = sort(collect(payload); by=x -> String(first(x)))
    io = IOBuffer()
    println(io, "{")
    for (i, (k, v)) in enumerate(pairs)
        comma = i == length(pairs) ? "" : ","
        println(io, "  ", _lisp_gat_json_quote(k), ": ", _lisp_gat_json_value(v), comma)
    end
    println(io, "}")
    String(take!(io))
end

_lisp_gat_lisp_sym(x) = replace(String(x), "_" => "-")
_lisp_gat_lisp_value(x::Symbol) = _lisp_gat_lisp_sym(x)
_lisp_gat_lisp_value(x::AbstractString) = _lisp_gat_sexp_quote(x)
_lisp_gat_lisp_value(x::Bool) = x ? "true" : "false"
_lisp_gat_lisp_value(::Nothing) = "nil"
_lisp_gat_lisp_value(x::Integer) = string(x)
_lisp_gat_lisp_value(x::AbstractFloat) = isfinite(x) ? string(x) : _lisp_gat_sexp_quote(string(x))
_lisp_gat_lisp_values(xs) = join((_lisp_gat_lisp_value(x) for x in xs), " ")

function _lisp_gat_lisp_ordinals(head::AbstractString, ordinals)
    isempty(ordinals) ? "($head)" : string("(", head, " ", join(ordinals, " "), ")")
end

function _lisp_gat_indent_block(s::AbstractString; prefix::AbstractString="  ")
    lines = Base.split(chomp(String(s)), '\n')
    join((string(prefix, line) for line in lines), "\n")
end

function render_lisp_gatlab_rewrite_request(request::LispGATRewriteRequest)
    io = IOBuffer()
    println(io, "(rewrite-execution")
    query_args = _lisp_gat_lisp_values(request.arguments)
    query_tail = isempty(query_args) ? "" : string(" ", query_args)
    println(io, "  (query ", _lisp_gat_lisp_sym(request.operation), query_tail, ")")
    println(io, "  (max-samples ", request.max_samples, ")")
    println(io, "  (backend ", _lisp_gat_lisp_sym(request.backend), "))")
    String(take!(io))
end

function render_lisp_gatlab_rewrite_program(program::LispGATRewriteProgram)
    io = IOBuffer()
    println(io, "(rewrite-program")
    for request in program.requests
        println(io, _lisp_gat_indent_block(render_lisp_gatlab_rewrite_request(request); prefix="  "))
    end
    println(io, ")")
    String(take!(io))
end

function render_lisp_gatlab_rewrite_program_step(step::LispGATRewriteProgramStep)
    io = IOBuffer()
    println(io, "(step ", step.index)
    println(io, "  (backend ", _lisp_gat_lisp_sym(step.backend), ")")
    println(io, "  ", _lisp_gat_lisp_ordinals("selected-ordinals", step.selected_ordinals))
    println(io, "  ", _lisp_gat_lisp_ordinals("introduced-ordinals", step.introduced_ordinals))
    println(io, "  (selected-all-materialized ", _lisp_gat_lisp_value(step.execution.selected_all_materialized), ")")
    println(io, "  (selected-all-targets ", _lisp_gat_lisp_value(step.execution.selected_all_targets), ")")
    println(io, "  (request-fingerprint ", _lisp_gat_lisp_value(_lisp_gat_hex64(step.request.fingerprint)), ")")
    println(io, "  (plan-fingerprint ", _lisp_gat_lisp_value(_lisp_gat_hex64(step.plan.fingerprint)), ")")
    println(io, "  (execution-fingerprint ", _lisp_gat_lisp_value(_lisp_gat_hex64(step.execution.fingerprint)), ")")
    println(io, "  (fingerprint ", _lisp_gat_lisp_value(_lisp_gat_hex64(step.fingerprint)), ")")
    println(io, "  (request")
    println(io, _lisp_gat_indent_block(render_lisp_gatlab_rewrite_request(step.request); prefix="    "))
    println(io, "  ))")
    String(take!(io))
end

function render_lisp_gatlab_rewrite_program_trace(
    trace::LispGATRewriteProgramTrace;
    bridge=nothing,
    extension_status=nothing,
    include_steps::Bool=true,
)
    exec = trace.execution
    io = IOBuffer()
    println(io, "(rewrite-trace")
    println(io, "  (fingerprint ", _lisp_gat_lisp_value(_lisp_gat_hex64(trace.fingerprint)), ")")
    println(io, "  (program-fingerprint ", _lisp_gat_lisp_value(_lisp_gat_hex64(exec.program.fingerprint)), ")")
    println(io, "  (program-execution-fingerprint ", _lisp_gat_lisp_value(_lisp_gat_hex64(exec.fingerprint)), ")")
    println(io, "  (request-count ", length(exec.program.requests), ")")
    println(io, "  (execution-count ", length(exec.executions), ")")
    println(io, "  (step-count ", length(trace.steps), ")")
    println(io, "  (coverage-complete ", _lisp_gat_lisp_value(trace.coverage_complete), ")")
    println(io, "  (all-selected-all-materialized ", _lisp_gat_lisp_value(exec.all_selected_all_materialized), ")")
    println(io, "  (all-selected-all-targets ", _lisp_gat_lisp_value(exec.all_selected_all_targets), ")")
    println(io, "  ", _lisp_gat_lisp_ordinals("selected-ordinals", exec.selected_ordinals))
    println(io, "  ", _lisp_gat_lisp_ordinals("repeated-ordinals", trace.repeated_ordinals))
    println(io, "  (backends ", _lisp_gat_lisp_values(exec.backends), ")")
    if bridge !== nothing
        println(io, "  (bridge-fingerprint ", _lisp_gat_lisp_value(_lisp_gat_hex64(getproperty(bridge, :fingerprint))), ")")
        println(io, "  (bridge-counterfactuals ", length(getproperty(bridge, :counterfactuals)), ")")
    end
    if extension_status !== nothing
        println(io, "  (extension-loaded ", _lisp_gat_lisp_value(Bool(getproperty(extension_status, :extension_loaded))), ")")
        println(io, "  (extension-status-name ", _lisp_gat_lisp_value(getproperty(extension_status, :extension_name)), ")")
    end
    println(io, "  (program")
    println(io, _lisp_gat_indent_block(render_lisp_gatlab_rewrite_program(exec.program); prefix="    "))
    println(io, "  )")
    if include_steps
        println(io, "  (steps")
        for step in trace.steps
            println(io, _lisp_gat_indent_block(render_lisp_gatlab_rewrite_program_step(step); prefix="    "))
        end
        println(io, "  )")
    end
    println(io, ")")
    String(take!(io))
end

function _lisp_gat_trace_entry(entries, head::Symbol; required::Bool=true)
    entry = _lisp_gat_form_entry(head, entries)
    if entry === nothing && required
        error("Missing $head entry in Lisp/GATlab rewrite trace")
    end
    entry
end

function _lisp_gat_trace_scalar(entries, head::Symbol; required::Bool=true, default=nothing)
    entry = _lisp_gat_trace_entry(entries, head; required=required)
    entry === nothing && return default
    length(entry) == 2 || error("$head expects one value in Lisp/GATlab rewrite trace")
    entry[2]
end

function _lisp_gat_trace_bool(x, head::Symbol)
    x isa Bool && return x
    if x isa Symbol || x isa AbstractString
        sym = _lisp_gat_sym(x)
        sym == :true && return true
        sym == :false && return false
    end
    error("$head expects true or false in Lisp/GATlab rewrite trace")
end

function _lisp_gat_trace_fingerprint(entries, head::Symbol)
    string(_lisp_gat_trace_scalar(entries, head))
end

function _lisp_gat_trace_ordinals(entries, head::Symbol)
    entry = _lisp_gat_trace_entry(entries, head; required=false)
    entry === nothing && return Int[]
    Int[_lisp_gat_query_int(x) for x in entry[2:end]]
end

function _lisp_gat_trace_backends(entries)
    entry = _lisp_gat_trace_entry(entries, :backends; required=false)
    entry === nothing && return Symbol[]
    Symbol[_lisp_gat_sym(x) for x in entry[2:end]]
end

function _lisp_gat_trace_nested_form(entry, head::Symbol)
    forms = Any[x for x in entry[2:end] if x isa Vector && !isempty(x)]
    length(forms) == 1 || error("$head expects exactly one nested form")
    forms[1]
end

function _lisp_gat_parse_trace_step(entry; seed::Integer=GAY_SEED)
    entry isa Vector || error("rewrite trace step must be a list")
    length(entry) >= 2 || error("rewrite trace step requires an index")
    _lisp_gat_sym(entry[1]) == :step || error("Expected step entry in rewrite trace")
    index = _lisp_gat_query_int(entry[2])
    entries = entry[3:end]
    request_entry = _lisp_gat_trace_entry(entries, :request)
    request = parse_lisp_gatlab_rewrite_form(_lisp_gat_trace_nested_form(request_entry, :request); seed=seed)

    (
        index=index,
        backend=_lisp_gat_sym(_lisp_gat_trace_scalar(entries, :backend)),
        selected_ordinals=_lisp_gat_trace_ordinals(entries, :selected_ordinals),
        introduced_ordinals=_lisp_gat_trace_ordinals(entries, :introduced_ordinals),
        selected_all_materialized=_lisp_gat_trace_bool(
            _lisp_gat_trace_scalar(entries, :selected_all_materialized),
            :selected_all_materialized,
        ),
        selected_all_targets=_lisp_gat_trace_bool(
            _lisp_gat_trace_scalar(entries, :selected_all_targets),
            :selected_all_targets,
        ),
        request_fingerprint=_lisp_gat_trace_fingerprint(entries, :request_fingerprint),
        plan_fingerprint=_lisp_gat_trace_fingerprint(entries, :plan_fingerprint),
        execution_fingerprint=_lisp_gat_trace_fingerprint(entries, :execution_fingerprint),
        fingerprint=_lisp_gat_trace_fingerprint(entries, :fingerprint),
        request=request,
    )
end

function parse_lisp_gatlab_rewrite_program_trace_form(
    form;
    seed::Integer=GAY_SEED,
)
    normalized = _lisp_gat_normalize_form(form)
    parsed = normalized.form
    parsed isa Vector || error("Lisp/GATlab rewrite trace form must parse to a list")
    !isempty(parsed) || error("Lisp/GATlab rewrite trace form cannot be empty")
    _lisp_gat_sym(parsed[1]) == :rewrite_trace ||
        error("Lisp/GATlab rewrite trace form must start with rewrite-trace")

    entries = parsed[2:end]
    program_entry = _lisp_gat_trace_entry(entries, :program)
    program = parse_lisp_gatlab_rewrite_program(_lisp_gat_trace_nested_form(program_entry, :program); seed=seed)

    steps_entry = _lisp_gat_trace_entry(entries, :steps; required=false)
    steps = steps_entry === nothing ? NamedTuple[] : [
        _lisp_gat_parse_trace_step(entry; seed=seed)
        for entry in steps_entry[2:end]
        if entry isa Vector && !isempty(entry)
    ]

    (
        form=parsed,
        parser=normalized.parser,
        fingerprint=_lisp_gat_trace_fingerprint(entries, :fingerprint),
        program_fingerprint=_lisp_gat_trace_fingerprint(entries, :program_fingerprint),
        program_execution_fingerprint=_lisp_gat_trace_fingerprint(entries, :program_execution_fingerprint),
        request_count=_lisp_gat_query_int(_lisp_gat_trace_scalar(entries, :request_count)),
        execution_count=_lisp_gat_query_int(_lisp_gat_trace_scalar(entries, :execution_count)),
        step_count=_lisp_gat_query_int(_lisp_gat_trace_scalar(entries, :step_count)),
        coverage_complete=_lisp_gat_trace_bool(_lisp_gat_trace_scalar(entries, :coverage_complete), :coverage_complete),
        all_selected_all_materialized=_lisp_gat_trace_bool(
            _lisp_gat_trace_scalar(entries, :all_selected_all_materialized),
            :all_selected_all_materialized,
        ),
        all_selected_all_targets=_lisp_gat_trace_bool(
            _lisp_gat_trace_scalar(entries, :all_selected_all_targets),
            :all_selected_all_targets,
        ),
        selected_ordinals=_lisp_gat_trace_ordinals(entries, :selected_ordinals),
        repeated_ordinals=_lisp_gat_trace_ordinals(entries, :repeated_ordinals),
        backends=_lisp_gat_trace_backends(entries),
        program=program,
        steps=steps,
    )
end

function validate_lisp_gatlab_rewrite_program_trace_form(
    form;
    bridge::LispGATBridgeWorld=world_lisp_gatlab_bridge(),
    seed::Integer=GAY_SEED,
)
    parsed = parse_lisp_gatlab_rewrite_program_trace_form(form; seed=seed)
    actual = lisp_gatlab_rewrite_program_trace(lisp_gatlab_rewrite_program_execution(bridge, parsed.program))
    actual_steps = actual.steps
    parsed_steps = parsed.steps

    comparisons = Dict{Symbol,Bool}(
        :fingerprint => parsed.fingerprint == _lisp_gat_hex64(actual.fingerprint),
        :program_fingerprint => parsed.program_fingerprint == _lisp_gat_hex64(actual.execution.program.fingerprint),
        :program_execution_fingerprint =>
            parsed.program_execution_fingerprint == _lisp_gat_hex64(actual.execution.fingerprint),
        :request_count => parsed.request_count == length(actual.execution.program.requests),
        :execution_count => parsed.execution_count == length(actual.execution.executions),
        :step_count => parsed.step_count == length(actual_steps) == length(parsed_steps),
        :coverage_complete => parsed.coverage_complete == actual.coverage_complete,
        :all_selected_all_materialized =>
            parsed.all_selected_all_materialized == actual.execution.all_selected_all_materialized,
        :all_selected_all_targets => parsed.all_selected_all_targets == actual.execution.all_selected_all_targets,
        :selected_ordinals => parsed.selected_ordinals == actual.execution.selected_ordinals,
        :repeated_ordinals => parsed.repeated_ordinals == actual.repeated_ordinals,
        :backends => parsed.backends == actual.execution.backends,
        :step_indices => [step.index for step in parsed_steps] == [step.index for step in actual_steps],
        :step_backends => [step.backend for step in parsed_steps] == [step.backend for step in actual_steps],
        :step_selected_ordinals =>
            [step.selected_ordinals for step in parsed_steps] == [step.selected_ordinals for step in actual_steps],
        :step_introduced_ordinals =>
            [step.introduced_ordinals for step in parsed_steps] == [step.introduced_ordinals for step in actual_steps],
        :step_selected_all_materialized =>
            [step.selected_all_materialized for step in parsed_steps] ==
                [step.execution.selected_all_materialized for step in actual_steps],
        :step_selected_all_targets =>
            [step.selected_all_targets for step in parsed_steps] ==
                [step.execution.selected_all_targets for step in actual_steps],
        :step_request_fingerprints =>
            [step.request_fingerprint for step in parsed_steps] ==
                [_lisp_gat_hex64(step.request.fingerprint) for step in actual_steps],
        :step_plan_fingerprints =>
            [step.plan_fingerprint for step in parsed_steps] ==
                [_lisp_gat_hex64(step.plan.fingerprint) for step in actual_steps],
        :step_execution_fingerprints =>
            [step.execution_fingerprint for step in parsed_steps] ==
                [_lisp_gat_hex64(step.execution.fingerprint) for step in actual_steps],
        :step_fingerprints =>
            [step.fingerprint for step in parsed_steps] == [_lisp_gat_hex64(step.fingerprint) for step in actual_steps],
        :step_request_operations =>
            [step.request.operation for step in parsed_steps] == [step.request.operation for step in actual_steps],
        :step_request_arguments =>
            [step.request.arguments for step in parsed_steps] == [step.request.arguments for step in actual_steps],
        :step_request_backends =>
            [step.request.backend for step in parsed_steps] == [step.request.backend for step in actual_steps],
    )
    valid = all(values(comparisons))

    (
        valid=valid,
        parser=parsed.parser,
        parsed=parsed,
        trace=actual,
        comparisons=comparisons,
        fingerprint=_lisp_gat_hex64(actual.fingerprint),
        evidence=valid ?
            "LispSyntax rewrite trace form replays to the same GATlab/AlgebraicJulia trace" :
            "LispSyntax rewrite trace form does not replay to the same GATlab/AlgebraicJulia trace",
    )
end

function lisp_gatlab_rewrite_trace_validation(
    form=default_lisp_gatlab_rewrite_trace_form();
    bridge::LispGATBridgeWorld=world_lisp_gatlab_bridge(),
    seed::Integer=GAY_SEED,
)
    validate_lisp_gatlab_rewrite_program_trace_form(form; bridge=bridge, seed=seed)
end

function lisp_gatlab_rewrite_trace_validation_payload(validation)
    comparisons = Dict(String(k) => v for (k, v) in sort(collect(validation.comparisons); by=x -> String(first(x))))
    Dict{String,Any}(
        "valid" => validation.valid,
        "parser" => String(validation.parser),
        "fingerprint" => validation.fingerprint,
        "parsed_fingerprint" => validation.parsed.fingerprint,
        "program_fingerprint" => validation.parsed.program_fingerprint,
        "program_execution_fingerprint" => validation.parsed.program_execution_fingerprint,
        "request_count" => validation.parsed.request_count,
        "execution_count" => validation.parsed.execution_count,
        "step_count" => validation.parsed.step_count,
        "selected_ordinals" => validation.parsed.selected_ordinals,
        "repeated_ordinals" => validation.parsed.repeated_ordinals,
        "backends" => String.(validation.parsed.backends),
        "comparisons" => comparisons,
        "evidence" => validation.evidence,
    )
end

function lisp_gatlab_rewrite_trace_validation_payload(form::AbstractString; kwargs...)
    lisp_gatlab_rewrite_trace_validation_payload(lisp_gatlab_rewrite_trace_validation(form; kwargs...))
end

render_lisp_gatlab_rewrite_trace_validation_json(validation) =
    _lisp_gat_render_json_payload(lisp_gatlab_rewrite_trace_validation_payload(validation))

render_lisp_gatlab_rewrite_trace_validation_json(form::AbstractString; kwargs...) =
    render_lisp_gatlab_rewrite_trace_validation_json(lisp_gatlab_rewrite_trace_validation(form; kwargs...))

_lisp_gat_payload_arg(x) = x isa Symbol ? String(x) : x
_lisp_gat_payload_path(xs) = String.(xs)

function lisp_gatlab_rewrite_request_payload(request::LispGATRewriteRequest)
    Dict{String,Any}(
        "operation" => String(request.operation),
        "arguments" => Any[_lisp_gat_payload_arg(arg) for arg in request.arguments],
        "max_samples" => request.max_samples,
        "backend" => String(request.backend),
        "parser" => String(request.parser),
        "fingerprint" => _lisp_gat_hex64(request.fingerprint),
        "evidence" => request.evidence,
    )
end

render_lisp_gatlab_rewrite_request_json(request::LispGATRewriteRequest) =
    _lisp_gat_render_json_payload(lisp_gatlab_rewrite_request_payload(request))

function lisp_gatlab_rewrite_program_payload(program::LispGATRewriteProgram)
    Dict{String,Any}(
        "request_count" => length(program.requests),
        "parser" => String(program.parser),
        "request_fingerprints" => [_lisp_gat_hex64(request.fingerprint) for request in program.requests],
        "requests" => Dict{String,Any}[
            lisp_gatlab_rewrite_request_payload(request)
            for request in program.requests
        ],
        "fingerprint" => _lisp_gat_hex64(program.fingerprint),
        "evidence" => program.evidence,
    )
end

render_lisp_gatlab_rewrite_program_json(program::LispGATRewriteProgram) =
    _lisp_gat_render_json_payload(lisp_gatlab_rewrite_program_payload(program))

function _lisp_gat_payload_prop(x, field::Symbol, default)
    hasproperty(x, field) ? getproperty(x, field) : default
end

function _lisp_gat_payload_candidate(cand; include_paths::Bool=true, include_dpo::Bool=true)
    payload = Dict{String,Any}(
        "ordinal" => Int(getproperty(cand, :ordinal)),
        "counterfactual_index" => Int(getproperty(cand, :counterfactual_index)),
        "witness_ordinal" => Int(getproperty(cand, :witness_ordinal)),
        "from_aspect" => String(getproperty(cand, :from_aspect)),
        "to_aspect" => String(getproperty(cand, :to_aspect)),
        "rule_style" => String(getproperty(cand, :rule_style)),
        "trit_delta" => Int(getproperty(cand, :trit_delta)),
        "closure_effect" => String(getproperty(cand, :closure_effect)),
        "color_hex" => String(getproperty(cand, :color_hex)),
        "semantic_cost" => Float64(getproperty(cand, :semantic_cost)),
        "fingerprint" => _lisp_gat_hex64(UInt64(getproperty(cand, :fingerprint))),
    )

    if include_paths
        payload["match_path"] = _lisp_gat_payload_path(getproperty(cand, :match_path))
        payload["source_path"] = _lisp_gat_payload_path(getproperty(cand, :source_path))
        payload["target_path"] = _lisp_gat_payload_path(getproperty(cand, :target_path))
        payload["arena_path"] = _lisp_gat_payload_path(getproperty(cand, :arena_path))
        payload["witness_arena_path"] = _lisp_gat_payload_path(getproperty(cand, :witness_arena_path))
    end

    if include_dpo && hasproperty(cand, :dpo_rule_materialized)
        payload["dpo_rule_materialized"] = Bool(getproperty(cand, :dpo_rule_materialized))
        payload["dpo_result_executed"] = Bool(_lisp_gat_payload_prop(cand, :dpo_result_executed, false))
        payload["dpo_result_is_target"] = Bool(_lisp_gat_payload_prop(cand, :dpo_result_is_target, false))
        payload["dpo_rule_type"] = String(_lisp_gat_payload_prop(cand, :dpo_rule_type, ""))
        payload["left_assignment_aspect"] = Int(_lisp_gat_payload_prop(cand, :left_assignment_aspect, 0))
        payload["right_assignment_aspect"] = Int(_lisp_gat_payload_prop(cand, :right_assignment_aspect, 0))
        payload["result_assignment_aspect"] = Int(_lisp_gat_payload_prop(cand, :result_assignment_aspect, 0))
        payload["source_term"] = String(_lisp_gat_payload_prop(cand, :source_term_text, ""))
        payload["target_term"] = String(_lisp_gat_payload_prop(cand, :target_term_text, ""))
        payload["arena_term"] = String(_lisp_gat_payload_prop(cand, :arena_term_text, ""))
    end

    payload
end

function lisp_gatlab_query_payload(q::LispGATQueryResult; match_limit::Integer=8, include_matches::Bool=false)
    payload = Dict{String,Any}(
        "operation" => String(q.operation),
        "arguments" => Any[_lisp_gat_payload_arg(arg) for arg in q.arguments],
        "match_count" => length(q.matches),
        "match_ordinals" => [cand.ordinal for cand in Iterators.take(q.matches, max(0, Int(match_limit)))],
        "first_match_fingerprint" => isempty(q.matches) ? "" : _lisp_gat_hex64(first(q.matches).fingerprint),
        "fingerprint" => _lisp_gat_hex64(q.fingerprint),
        "evidence" => q.evidence,
        "coverage_complete" => q.coverage.complete,
        "coverage_counterfactuals" => q.coverage.counterfactuals,
        "coverage_rewrite_candidates" => q.coverage.rewrite_candidates,
        "coverage_unique_rewrite_candidates" => q.coverage.unique_rewrite_candidates,
        "coverage_witnesses" => q.coverage.witnesses,
        "coverage_expected_per_witness" => q.coverage.expected_per_witness,
        "coverage_no_duplicate_edges" => q.coverage.no_duplicate_edges,
    )
    if include_matches
        payload["matches"] = Dict{String,Any}[
            _lisp_gat_payload_candidate(cand; include_paths=true, include_dpo=false)
            for cand in q.matches
        ]
    end
    payload
end

function lisp_gatlab_rewrite_plan_payload(plan::LispGATRewritePlan)
    Dict{String,Any}(
        "operation" => String(plan.query.operation),
        "arguments" => Any[_lisp_gat_payload_arg(arg) for arg in plan.query.arguments],
        "match_count" => length(plan.query.matches),
        "sample_mode" => String(plan.sample_mode),
        "sample_ordinals" => plan.sample_ordinals,
        "max_samples" => plan.max_samples,
        "materialization_backend" => String(plan.materialization_backend),
        "bridge_fingerprint" => _lisp_gat_hex64(plan.bridge_fingerprint),
        "query_fingerprint" => _lisp_gat_hex64(plan.query.fingerprint),
        "fingerprint" => _lisp_gat_hex64(plan.fingerprint),
        "evidence" => plan.evidence,
    )
end

function _lisp_gat_selected_candidate_payload(materialization, ordinals::Vector{Int})
    by_ordinal = Dict(Int(getproperty(cand, :ordinal)) => cand for cand in materialization.rewrite_candidates)
    Dict{String,Any}[
        _lisp_gat_payload_candidate(by_ordinal[ordinal]; include_paths=false, include_dpo=true)
        for ordinal in ordinals
    ]
end

function lisp_gatlab_rewrite_execution_payload(
    exec::LispGATRewriteExecution;
    request=nothing,
    materialization=nothing,
    bridge=nothing,
    extension_status=nothing,
    include_selected_candidates::Bool=false,
)
    payload = Dict{String,Any}(
        "operation" => String(exec.plan.query.operation),
        "arguments" => Any[_lisp_gat_payload_arg(arg) for arg in exec.plan.query.arguments],
        "query_operation" => String(exec.plan.query.operation),
        "query_arguments" => Any[_lisp_gat_payload_arg(arg) for arg in exec.plan.query.arguments],
        "query_match_count" => length(exec.plan.query.matches),
        "backend" => String(exec.backend),
        "selected_ordinals" => exec.selected_ordinals,
        "materialized_ordinals" => exec.materialized_ordinals,
        "executed_ordinals" => exec.executed_ordinals,
        "target_ordinals" => exec.target_ordinals,
        "spec_count" => exec.spec_count,
        "materialized_count" => exec.materialized_count,
        "selected_all_materialized" => exec.selected_all_materialized,
        "selected_all_targets" => exec.selected_all_targets,
        "materialization_fingerprint" => _lisp_gat_hex64(exec.materialization_fingerprint),
        "query_fingerprint" => _lisp_gat_hex64(exec.plan.query.fingerprint),
        "plan_fingerprint" => _lisp_gat_hex64(exec.plan.fingerprint),
        "plan_sample_mode" => String(exec.plan.sample_mode),
        "plan_max_samples" => exec.plan.max_samples,
        "fingerprint" => _lisp_gat_hex64(exec.fingerprint),
        "execution_fingerprint" => _lisp_gat_hex64(exec.fingerprint),
        "evidence" => exec.evidence,
        "coverage_complete" => exec.plan.query.coverage.complete,
        "coverage_counterfactuals" => exec.plan.query.coverage.counterfactuals,
        "coverage_rewrite_candidates" => exec.plan.query.coverage.rewrite_candidates,
        "coverage_unique_rewrite_candidates" => exec.plan.query.coverage.unique_rewrite_candidates,
        "coverage_witnesses" => exec.plan.query.coverage.witnesses,
        "coverage_expected_per_witness" => exec.plan.query.coverage.expected_per_witness,
        "coverage_no_duplicate_edges" => exec.plan.query.coverage.no_duplicate_edges,
        "query" => lisp_gatlab_query_payload(exec.plan.query),
        "plan" => lisp_gatlab_rewrite_plan_payload(exec.plan),
    )

    if request !== nothing
        payload["request"] = lisp_gatlab_rewrite_request_payload(request)
        payload["request_fingerprint"] = _lisp_gat_hex64(getproperty(request, :fingerprint))
    end

    if bridge !== nothing
        payload["bridge_fingerprint"] = _lisp_gat_hex64(getproperty(bridge, :fingerprint))
        payload["parser"] = String(getproperty(bridge, :parser))
        payload["bridge_counterfactuals"] = length(getproperty(bridge, :counterfactuals))
    end

    if materialization !== nothing
        payload["extension_name"] = String(getproperty(materialization, :extension))
        payload["packages"] = String.(getproperty(materialization, :packages))
        payload["presentation_type"] = String(getproperty(materialization, :presentation_type))
        payload["generator_counts"] = Dict(String(k) => v for (k, v) in getproperty(materialization, :generator_counts))
        payload["equation_count"] = Int(getproperty(materialization, :equation_count))
        payload["counterfactual_count"] = Int(getproperty(materialization, :counterfactual_count))
        payload["rewrite_candidate_count"] = Int(getproperty(materialization, :rewrite_candidate_count))
        include_selected_candidates &&
            (payload["selected_candidates"] =
                _lisp_gat_selected_candidate_payload(materialization, exec.selected_ordinals))
    end

    if extension_status !== nothing
        payload["extension_loaded"] = Bool(getproperty(extension_status, :extension_loaded))
        payload["extension_status_name"] = String(getproperty(extension_status, :extension_name))
    end

    payload
end

function render_lisp_gatlab_rewrite_execution_json(
    exec::LispGATRewriteExecution;
    request=nothing,
    materialization=nothing,
    bridge=nothing,
    extension_status=nothing,
    include_selected_candidates::Bool=false,
)
    payload = lisp_gatlab_rewrite_execution_payload(
        exec;
        request=request,
        materialization=materialization,
        bridge=bridge,
        extension_status=extension_status,
        include_selected_candidates=include_selected_candidates,
    )
    _lisp_gat_render_json_payload(payload)
end

function lisp_gatlab_rewrite_program_execution_payload(
    exec::LispGATRewriteProgramExecution;
    bridge=nothing,
    extension_status=nothing,
    include_executions::Bool=true,
)
    payload = Dict{String,Any}(
        "request_count" => length(exec.program.requests),
        "execution_count" => length(exec.executions),
        "backends" => String.(exec.backends),
        "selected_ordinals" => exec.selected_ordinals,
        "all_selected_all_materialized" => exec.all_selected_all_materialized,
        "all_selected_all_targets" => exec.all_selected_all_targets,
        "program_fingerprint" => _lisp_gat_hex64(exec.program.fingerprint),
        "fingerprint" => _lisp_gat_hex64(exec.fingerprint),
        "evidence" => exec.evidence,
        "program" => lisp_gatlab_rewrite_program_payload(exec.program),
    )

    if bridge !== nothing
        payload["bridge_fingerprint"] = _lisp_gat_hex64(getproperty(bridge, :fingerprint))
        payload["parser"] = String(getproperty(bridge, :parser))
        payload["bridge_counterfactuals"] = length(getproperty(bridge, :counterfactuals))
    end

    if extension_status !== nothing
        payload["extension_loaded"] = Bool(getproperty(extension_status, :extension_loaded))
        payload["extension_status_name"] = String(getproperty(extension_status, :extension_name))
    end

    if include_executions
        payload["executions"] = Dict{String,Any}[
            lisp_gatlab_rewrite_execution_payload(execution; request=exec.program.requests[i])
            for (i, execution) in enumerate(exec.executions)
        ]
    end

    payload
end

function render_lisp_gatlab_rewrite_program_execution_json(
    exec::LispGATRewriteProgramExecution;
    bridge=nothing,
    extension_status=nothing,
    include_executions::Bool=true,
)
    _lisp_gat_render_json_payload(lisp_gatlab_rewrite_program_execution_payload(
        exec;
        bridge=bridge,
        extension_status=extension_status,
        include_executions=include_executions,
    ))
end

function lisp_gatlab_rewrite_program_step_payload(step::LispGATRewriteProgramStep)
    Dict{String,Any}(
        "index" => step.index,
        "backend" => String(step.backend),
        "selected_ordinals" => step.selected_ordinals,
        "introduced_ordinals" => step.introduced_ordinals,
        "selected_all_materialized" => step.execution.selected_all_materialized,
        "selected_all_targets" => step.execution.selected_all_targets,
        "request_fingerprint" => _lisp_gat_hex64(step.request.fingerprint),
        "plan_fingerprint" => _lisp_gat_hex64(step.plan.fingerprint),
        "execution_fingerprint" => _lisp_gat_hex64(step.execution.fingerprint),
        "fingerprint" => _lisp_gat_hex64(step.fingerprint),
        "evidence" => step.evidence,
        "request" => lisp_gatlab_rewrite_request_payload(step.request),
        "plan" => lisp_gatlab_rewrite_plan_payload(step.plan),
        "execution" => lisp_gatlab_rewrite_execution_payload(step.execution; request=step.request),
    )
end

function lisp_gatlab_rewrite_program_trace_payload(
    trace::LispGATRewriteProgramTrace;
    bridge=nothing,
    extension_status=nothing,
    include_steps::Bool=true,
)
    exec = trace.execution
    payload = Dict{String,Any}(
        "request_count" => length(exec.program.requests),
        "execution_count" => length(exec.executions),
        "step_count" => length(trace.steps),
        "backends" => String.(exec.backends),
        "selected_ordinals" => exec.selected_ordinals,
        "repeated_ordinals" => trace.repeated_ordinals,
        "all_selected_all_materialized" => exec.all_selected_all_materialized,
        "all_selected_all_targets" => exec.all_selected_all_targets,
        "coverage_complete" => trace.coverage_complete,
        "program_fingerprint" => _lisp_gat_hex64(exec.program.fingerprint),
        "program_execution_fingerprint" => _lisp_gat_hex64(exec.fingerprint),
        "fingerprint" => _lisp_gat_hex64(trace.fingerprint),
        "evidence" => trace.evidence,
        "program" => lisp_gatlab_rewrite_program_payload(exec.program),
    )

    if bridge !== nothing
        payload["bridge_fingerprint"] = _lisp_gat_hex64(getproperty(bridge, :fingerprint))
        payload["parser"] = String(getproperty(bridge, :parser))
        payload["bridge_counterfactuals"] = length(getproperty(bridge, :counterfactuals))
    end

    if extension_status !== nothing
        payload["extension_loaded"] = Bool(getproperty(extension_status, :extension_loaded))
        payload["extension_status_name"] = String(getproperty(extension_status, :extension_name))
    end

    if include_steps
        payload["steps"] = Dict{String,Any}[
            lisp_gatlab_rewrite_program_step_payload(step)
            for step in trace.steps
        ]
    end

    payload
end

function render_lisp_gatlab_rewrite_program_trace_json(
    trace::LispGATRewriteProgramTrace;
    bridge=nothing,
    extension_status=nothing,
    include_steps::Bool=true,
)
    _lisp_gat_render_json_payload(lisp_gatlab_rewrite_program_trace_payload(
        trace;
        bridge=bridge,
        extension_status=extension_status,
        include_steps=include_steps,
    ))
end

function render_lisp_gatlab_json(w::LispGATBridgeWorld=world_lisp_gatlab_bridge())
    summary = lisp_gatlab_bridge_summary(w)
    effect_pairs = sort(collect(summary.effect_counts); by=x -> String(first(x)))
    io = IOBuffer()
    println(io, "{")
    println(io, "  \"presentation_name\": ", _lisp_gat_json_quote(String(w.presentation_name)), ",")
    println(io, "  \"fingerprint\": ", _lisp_gat_json_quote(string("0x", string(w.fingerprint, base=16, pad=16))), ",")
    println(io, "  \"source\": ", _lisp_gat_json_quote(w.source), ",")
    println(io, "  \"parser\": ", _lisp_gat_json_quote(String(w.parser)), ",")
    println(io, "  \"counts\": {")
    println(io, "    \"objects\": ", length(w.objects), ",")
    println(io, "    \"morphisms\": ", length(w.morphisms), ",")
    println(io, "    \"equations\": ", length(w.equations), ",")
    println(io, "    \"counterfactuals\": ", length(w.counterfactuals), ",")
    println(io, "    \"rewrite_candidates\": ", length(lisp_gatlab_rewrite_candidates(w)))
    println(io, "  },")
    println(io, "  \"all_counterfactuals_considered\": ", summary.all_counterfactuals_considered ? "true" : "false", ",")
    println(io, "  \"all_rewrite_candidates_considered\": ", summary.all_rewrite_candidates_considered ? "true" : "false", ",")
    println(io, "  \"effect_counts\": {")
    for (i, (k, v)) in enumerate(effect_pairs)
        comma = i == length(effect_pairs) ? "" : ","
        println(io, "    ", _lisp_gat_json_quote(String(k)), ": ", v, comma)
    end
    println(io, "  },")
    println(io, "  \"objects\": [")
    for (i, ob) in enumerate(w.objects)
        comma = i == length(w.objects) ? "" : ","
        println(io, "    {\"name\": ", _lisp_gat_json_quote(String(ob.name)),
            ", \"kind\": ", _lisp_gat_json_quote(String(ob.kind)),
            ", \"color_hex\": ", _lisp_gat_json_quote(ob.color_hex),
            ", \"evidence\": ", _lisp_gat_json_quote(ob.evidence), "}", comma)
    end
    println(io, "  ],")
    println(io, "  \"morphisms\": [")
    for (i, mor) in enumerate(w.morphisms)
        comma = i == length(w.morphisms) ? "" : ","
        println(io, "    {\"name\": ", _lisp_gat_json_quote(String(mor.name)),
            ", \"kind\": ", _lisp_gat_json_quote(String(mor.kind)),
            ", \"dom\": ", _lisp_gat_json_quote(String(mor.dom)),
            ", \"cod\": ", _lisp_gat_json_quote(String(mor.cod)),
            ", \"color_hex\": ", _lisp_gat_json_quote(mor.color_hex),
            ", \"evidence\": ", _lisp_gat_json_quote(mor.evidence), "}", comma)
    end
    println(io, "  ],")
    println(io, "  \"equations\": [")
    for (i, eq) in enumerate(w.equations)
        comma = i == length(w.equations) ? "" : ","
        println(io, "    {\"lhs\": ", _lisp_gat_json_array(eq.lhs),
            ", \"rhs\": ", _lisp_gat_json_array(eq.rhs),
            ", \"color_hex\": ", _lisp_gat_json_quote(eq.color_hex),
            ", \"evidence\": ", _lisp_gat_json_quote(eq.evidence), "}", comma)
    end
    println(io, "  ],")
    println(io, "  \"counterfactuals\": [")
    for (i, cf) in enumerate(w.counterfactuals)
        comma = i == length(w.counterfactuals) ? "" : ","
        println(io, "    {\"witness_ordinal\": ", cf.witness_ordinal,
            ", \"from_aspect\": ", _lisp_gat_json_quote(String(cf.from_aspect)),
            ", \"to_aspect\": ", _lisp_gat_json_quote(String(cf.to_aspect)),
            ", \"trit_delta\": ", cf.trit_delta,
            ", \"closure_effect\": ", _lisp_gat_json_quote(String(cf.closure_effect)),
            ", \"color_hex\": ", _lisp_gat_json_quote(cf.color_hex),
            ", \"semantic_cost\": ", cf.semantic_cost,
            ", \"lhs\": ", _lisp_gat_json_array(cf.lhs),
            ", \"rhs\": ", _lisp_gat_json_array(cf.rhs), "}", comma)
    end
    println(io, "  ],")
    candidates = lisp_gatlab_rewrite_candidates(w)
    println(io, "  \"rewrite_candidates\": [")
    for (i, cand) in enumerate(candidates)
        comma = i == length(candidates) ? "" : ","
        println(io, "    {\"ordinal\": ", cand.ordinal,
            ", \"counterfactual_index\": ", cand.counterfactual_index,
            ", \"witness_ordinal\": ", cand.witness_ordinal,
            ", \"from_aspect\": ", _lisp_gat_json_quote(String(cand.from_aspect)),
            ", \"to_aspect\": ", _lisp_gat_json_quote(String(cand.to_aspect)),
            ", \"rule_style\": ", _lisp_gat_json_quote(String(cand.rule_style)),
            ", \"match_path\": ", _lisp_gat_json_array(cand.match_path),
            ", \"source_path\": ", _lisp_gat_json_array(cand.source_path),
            ", \"target_path\": ", _lisp_gat_json_array(cand.target_path),
            ", \"arena_path\": ", _lisp_gat_json_array(cand.arena_path),
            ", \"witness_arena_path\": ", _lisp_gat_json_array(cand.witness_arena_path),
            ", \"trit_delta\": ", cand.trit_delta,
            ", \"closure_effect\": ", _lisp_gat_json_quote(String(cand.closure_effect)),
            ", \"color_hex\": ", _lisp_gat_json_quote(cand.color_hex),
            ", \"semantic_cost\": ", cand.semantic_cost,
            ", \"fingerprint\": ",
            _lisp_gat_json_quote(string("0x", string(cand.fingerprint, base=16, pad=16))), "}", comma)
    end
    println(io, "  ]")
    println(io, "}")
    String(take!(io))
end

function _algebraicjulia_realization_fingerprint(
    w::LispGATBridgeWorld,
    extension::Symbol,
    backend::Symbol,
    packages,
    acset_hint::AbstractString,
    rewriting_hint::AbstractString,
)::UInt64
    stable_seed((
        :algebraicjulia_realization,
        w.fingerprint,
        extension,
        backend,
        collect(packages),
        acset_hint,
        rewriting_hint,
    ); seed=w.seed)
end

function algebraicjulia_realization_plan(
    w::LispGATBridgeWorld=world_lisp_gatlab_bridge();
    extension::Symbol=:Gay,
    backend::Symbol=:projection,
    packages::Vector{Symbol}=Symbol[],
    acset_hint::AbstractString="Textual Catlab/GATlab projection only; load GayAlgebraicJuliaExt for package-backed realization.",
    rewriting_hint::AbstractString="Counterfactual assignments are rewrite candidates until AlgebraicRewriting is loaded.",
)
    AlgebraicJuliaRealization(
        extension,
        backend,
        packages,
        w.parser,
        render_lisp_gatlab_theory(w),
        render_lisp_gatlab_presentation(w),
        String(acset_hint),
        String(rewriting_hint),
        _algebraicjulia_realization_fingerprint(w, extension, backend, packages, acset_hint, rewriting_hint),
    )
end

function _lisp_gat_generator_counts(w::LispGATBridgeWorld)
    Dict(
        :Ob => count(ob -> ob.kind == :ob, w.objects),
        :AttrType => count(ob -> ob.kind == :attrtype, w.objects),
        :Hom => count(mor -> mor.kind == :hom, w.morphisms),
        :Attr => count(mor -> mor.kind == :attr, w.morphisms),
    )
end

function _algebraicjulia_materialization_fingerprint(
    w::LispGATBridgeWorld,
    extension::Symbol,
    backend::Symbol,
    packages,
    presentation_type::AbstractString,
    generator_counts,
    equation_count::Integer,
    rewrite_candidates,
)::UInt64
    stable_seed((
        :algebraicjulia_materialization,
        w.fingerprint,
        extension,
        backend,
        collect(packages),
        String(presentation_type),
        sort(collect(generator_counts); by=x -> String(first(x))),
        equation_count,
        [
            (
                getproperty(cand, :fingerprint),
                hasproperty(cand, :dpo_rule_materialized) ? getproperty(cand, :dpo_rule_materialized) : false,
                hasproperty(cand, :dpo_result_is_target) ? getproperty(cand, :dpo_result_is_target) : false,
            )
            for cand in rewrite_candidates if hasproperty(cand, :fingerprint)
        ],
        length(rewrite_candidates),
    ); seed=w.seed)
end

function algebraicjulia_materialization_plan(
    w::LispGATBridgeWorld=world_lisp_gatlab_bridge();
    extension::Symbol=:Gay,
    backend::Symbol=:projection,
    packages::Vector{Symbol}=Symbol[],
    presentation=nothing,
    presentation_type::AbstractString="projection-only",
    generator_counts::Dict{Symbol,Int}=_lisp_gat_generator_counts(w),
    equation_count::Integer=length(w.equations),
    rewrite_candidates::Vector{Any}=Any[lisp_gatlab_rewrite_candidates(w)...],
    rewrite_candidate_count::Integer=length(rewrite_candidates),
)
    AlgebraicJuliaMaterialization(
        extension,
        backend,
        packages,
        presentation,
        String(presentation_type),
        generator_counts,
        Int(equation_count),
        length(w.counterfactuals),
        Int(rewrite_candidate_count),
        rewrite_candidates,
        render_lisp_gatlab_theory(w),
        render_lisp_gatlab_presentation(w),
        _algebraicjulia_materialization_fingerprint(
            w,
            extension,
            backend,
            packages,
            presentation_type,
            generator_counts,
            equation_count,
            rewrite_candidates,
        ),
    )
end

const _ALGEBRAICJULIA_CAPABILITY_SPECS = (
    (
        package=:GATlab,
        uuid="f0ffcf3b-d13a-433e-917c-cc44ccf5ead2",
        role="generalized algebraic theories, explicit interfaces, and model-guided dispatch",
    ),
    (
        package=:Catlab,
        uuid="134e5e36-593f-5add-ad60-77f754baafbe",
        role="applied category theory, schema presentations, wiring diagrams, and categorical algebra",
    ),
    (
        package=:ACSets,
        uuid="227ef7b5-1206-438b-ac65-934d6da304b8",
        role="attributed C-set data structures and algebraic databases",
    ),
    (
        package=:AlgebraicRewriting,
        uuid="725a01d3-f174-5bbd-84e1-b9417bad95d9",
        role="DPO, SPO, and SqPO rewriting over Catlab/ACSets structures",
    ),
)

function algebraicjulia_capabilities()
    [
        begin
            package_name = String(spec.package)
            path = something(Base.find_package(package_name), "")
            AlgebraicJuliaCapability(
                spec.package,
                spec.uuid,
                spec.role,
                !isempty(path),
                path,
            )
        end
        for spec in _ALGEBRAICJULIA_CAPABILITY_SPECS
    ]
end

function algebraicjulia_bridge_status()
    caps = algebraicjulia_capabilities()
    extension_loaded = Base.get_extension(@__MODULE__, :GayAlgebraicJuliaExt) !== nothing
    (
        available=[cap.package for cap in caps if cap.available],
        missing=[cap.package for cap in caps if !cap.available],
        hard_dependencies_added=false,
        weak_dependencies_declared=true,
        extension_name=:GayAlgebraicJuliaExt,
        extension_loaded=extension_loaded,
        projection_styles=(:catlab_present, :gatlab_theory, :sexp, :json),
        realization_backends=(:projection, :algebraicjulia),
        materialization_backends=(:projection, :algebraicjulia),
        capabilities=[
            (
                package=cap.package,
                uuid=cap.uuid,
                role=cap.role,
                available=cap.available,
                load_path=cap.load_path,
            )
            for cap in caps
        ],
    )
end

function realize_lisp_gatlab_bridge(
    w::LispGATBridgeWorld=world_lisp_gatlab_bridge(),
    backend::Symbol=:projection,
)
    if backend == :projection
        return algebraicjulia_realization_plan(w)
    elseif backend == :algebraicjulia
        ext = Base.get_extension(@__MODULE__, :GayAlgebraicJuliaExt)
        if ext === nothing
            error("AlgebraicJulia backend requires GayAlgebraicJuliaExt; load GATlab, Catlab, ACSets, and AlgebraicRewriting.")
        end
        return ext.realize_lisp_gatlab_bridge_algebraicjulia(w)
    else
        error("Unknown Lisp/GATlab realization backend: $backend")
    end
end

function materialize_lisp_gatlab_bridge(
    w::LispGATBridgeWorld=world_lisp_gatlab_bridge(),
    backend::Symbol=:projection;
    kwargs...,
)
    if backend == :projection
        return algebraicjulia_materialization_plan(w)
    elseif backend == :algebraicjulia
        ext = Base.get_extension(@__MODULE__, :GayAlgebraicJuliaExt)
        if ext === nothing
            error("AlgebraicJulia materialization requires GayAlgebraicJuliaExt; load GATlab, Catlab, ACSets, and AlgebraicRewriting.")
        end
        return ext.materialize_lisp_gatlab_bridge_algebraicjulia(w; kwargs...)
    else
        error("Unknown Lisp/GATlab materialization backend: $backend")
    end
end

function materialize_lisp_gatlab_rewrite_plan(
    w::LispGATBridgeWorld,
    plan::LispGATRewritePlan;
    backend::Symbol=plan.materialization_backend,
)
    plan.bridge_fingerprint == w.fingerprint ||
        error("Rewrite plan was built for a different Lisp/GATlab bridge world")
    materialize_lisp_gatlab_bridge(
        w,
        backend;
        dpo_sample_ordinals=plan.sample_ordinals,
    )
end

materialize_lisp_gatlab_rewrite_plan(
    plan::LispGATRewritePlan;
    backend::Symbol=plan.materialization_backend,
) = materialize_lisp_gatlab_rewrite_plan(world_lisp_gatlab_bridge(), plan; backend=backend)

_lisp_gat_bool_property(x, field::Symbol, default::Bool=false)::Bool =
    hasproperty(x, field) ? Bool(getproperty(x, field)) : default

function _lisp_gat_rewrite_execution_fingerprint(
    plan::LispGATRewritePlan,
    backend::Symbol,
    materialization_fingerprint::UInt64,
    selected_ordinals::Vector{Int},
    materialized_ordinals::Vector{Int},
    executed_ordinals::Vector{Int},
    target_ordinals::Vector{Int},
    spec_count::Integer,
    materialized_count::Integer,
)::UInt64
    stable_seed((
        :lisp_gat_rewrite_execution,
        plan.fingerprint,
        backend,
        materialization_fingerprint,
        selected_ordinals,
        materialized_ordinals,
        executed_ordinals,
        target_ordinals,
        spec_count,
        materialized_count,
    ); seed=plan.bridge_fingerprint)
end

function lisp_gatlab_rewrite_execution(
    plan::LispGATRewritePlan,
    materialization::AlgebraicJuliaMaterialization,
)
    selected = copy(plan.sample_ordinals)
    selected_set = Set(selected)
    materialized_ordinals = Int[]
    executed_ordinals = Int[]
    target_ordinals = Int[]
    spec_count = 0

    for cand in materialization.rewrite_candidates
        ordinal = Int(getproperty(cand, :ordinal))
        materialized = _lisp_gat_bool_property(cand, :dpo_rule_materialized)
        executed = _lisp_gat_bool_property(cand, :dpo_result_executed)
        target = _lisp_gat_bool_property(cand, :dpo_result_is_target)

        if materialized
            push!(materialized_ordinals, ordinal)
        else
            spec_count += 1
        end
        executed && push!(executed_ordinals, ordinal)
        target && push!(target_ordinals, ordinal)
    end

    materialized_set = Set(materialized_ordinals)
    target_set = Set(target_ordinals)
    selected_all_materialized = all(ord -> ord in materialized_set, selected)
    selected_all_targets = all(ord -> ord in target_set, selected)
    fingerprint = _lisp_gat_rewrite_execution_fingerprint(
        plan,
        materialization.backend,
        materialization.fingerprint,
        selected,
        materialized_ordinals,
        executed_ordinals,
        target_ordinals,
        spec_count,
        length(materialized_ordinals),
    )

    LispGATRewriteExecution(
        plan,
        materialization.backend,
        materialization.fingerprint,
        selected,
        materialized_ordinals,
        executed_ordinals,
        target_ordinals,
        spec_count,
        length(materialized_ordinals),
        selected_all_materialized,
        selected_all_targets,
        fingerprint,
        "rewrite plan execution report over Lisp-selected GATlab/AlgebraicJulia candidates",
    )
end

function lisp_gatlab_rewrite_execution(
    w::LispGATBridgeWorld,
    plan::LispGATRewritePlan;
    backend::Symbol=plan.materialization_backend,
)
    materialization = materialize_lisp_gatlab_rewrite_plan(w, plan; backend=backend)
    lisp_gatlab_rewrite_execution(plan, materialization)
end

lisp_gatlab_rewrite_execution(
    plan::LispGATRewritePlan;
    backend::Symbol=plan.materialization_backend,
) = lisp_gatlab_rewrite_execution(world_lisp_gatlab_bridge(), plan; backend=backend)

lisp_gatlab_rewrite_execution(plan::LispGATRewritePlan, backend) =
    lisp_gatlab_rewrite_execution(plan; backend=_lisp_gat_sym(backend))

function lisp_gatlab_rewrite_execution(
    w::LispGATBridgeWorld,
    request::LispGATRewriteRequest;
    backend::Symbol=request.backend,
)
    plan = lisp_gatlab_rewrite_plan(w, request; materialization_backend=backend)
    lisp_gatlab_rewrite_execution(w, plan; backend=backend)
end

lisp_gatlab_rewrite_execution(
    request::LispGATRewriteRequest;
    backend::Symbol=request.backend,
) = lisp_gatlab_rewrite_execution(world_lisp_gatlab_bridge(), request; backend=backend)

function _lisp_gat_unique_ordinals(executions::Vector{LispGATRewriteExecution})
    seen = Set{Int}()
    ordinals = Int[]
    for exec in executions
        for ordinal in exec.selected_ordinals
            ordinal in seen && continue
            push!(seen, ordinal)
            push!(ordinals, ordinal)
        end
    end
    ordinals
end

function _lisp_gat_rewrite_program_execution_fingerprint(
    seed::UInt64,
    program::LispGATRewriteProgram,
    executions::Vector{LispGATRewriteExecution},
)::UInt64
    stable_seed((
        :lisp_gat_rewrite_program_execution,
        program.fingerprint,
        [exec.fingerprint for exec in executions],
        [exec.backend for exec in executions],
    ); seed=seed)
end

function lisp_gatlab_rewrite_program_execution(
    w::LispGATBridgeWorld,
    program::LispGATRewriteProgram,
)
    executions = LispGATRewriteExecution[
        lisp_gatlab_rewrite_execution(w, request; backend=request.backend)
        for request in program.requests
    ]
    backends = Symbol[exec.backend for exec in executions]
    selected_ordinals = _lisp_gat_unique_ordinals(executions)
    all_materialized = all(exec -> exec.selected_all_materialized, executions)
    all_targets = all(exec -> exec.selected_all_targets, executions)
    fingerprint = _lisp_gat_rewrite_program_execution_fingerprint(UInt64(w.seed), program, executions)

    LispGATRewriteProgramExecution(
        program,
        executions,
        backends,
        selected_ordinals,
        all_materialized,
        all_targets,
        fingerprint,
        "ordered Lisp/GATlab rewrite program execution over shared bridge world",
    )
end

lisp_gatlab_rewrite_program_execution(program::LispGATRewriteProgram) =
    lisp_gatlab_rewrite_program_execution(world_lisp_gatlab_bridge(), program)

lisp_gatlab_rewrite_program_execution(form) =
    lisp_gatlab_rewrite_program_execution(parse_lisp_gatlab_rewrite_program(form))

function _lisp_gat_rewrite_program_step_fingerprint(
    seed::UInt64,
    index::Integer,
    request::LispGATRewriteRequest,
    plan::LispGATRewritePlan,
    execution::LispGATRewriteExecution,
    introduced_ordinals::Vector{Int},
)::UInt64
    stable_seed((
        :lisp_gat_rewrite_program_step,
        index,
        request.fingerprint,
        plan.fingerprint,
        execution.fingerprint,
        execution.selected_ordinals,
        introduced_ordinals,
    ); seed=seed)
end

function _lisp_gat_rewrite_program_trace_fingerprint(
    seed::UInt64,
    execution::LispGATRewriteProgramExecution,
    steps::Vector{LispGATRewriteProgramStep},
    repeated_ordinals::Vector{Int},
)::UInt64
    stable_seed((
        :lisp_gat_rewrite_program_trace,
        execution.fingerprint,
        [step.fingerprint for step in steps],
        repeated_ordinals,
    ); seed=seed)
end

function lisp_gatlab_rewrite_program_trace(execution::LispGATRewriteProgramExecution)
    seen = Set{Int}()
    repeated_seen = Set{Int}()
    repeated_ordinals = Int[]
    steps = LispGATRewriteProgramStep[]
    seed = UInt64(execution.program.fingerprint)

    for (i, exec) in enumerate(execution.executions)
        request = execution.program.requests[i]
        introduced = Int[]
        for ordinal in exec.selected_ordinals
            if ordinal in seen
                if !(ordinal in repeated_seen)
                    push!(repeated_seen, ordinal)
                    push!(repeated_ordinals, ordinal)
                end
            else
                push!(seen, ordinal)
                push!(introduced, ordinal)
            end
        end

        fp = _lisp_gat_rewrite_program_step_fingerprint(
            seed,
            i,
            request,
            exec.plan,
            exec,
            introduced,
        )
        push!(steps, LispGATRewriteProgramStep(
            i,
            request,
            exec.plan,
            exec,
            copy(exec.selected_ordinals),
            introduced,
            exec.backend,
            fp,
            "ordered Lisp/GATlab rewrite-program step trace",
        ))
    end

    coverage_complete = all(step -> step.plan.query.coverage.complete, steps)
    fp = _lisp_gat_rewrite_program_trace_fingerprint(seed, execution, steps, repeated_ordinals)
    LispGATRewriteProgramTrace(
        execution,
        steps,
        repeated_ordinals,
        coverage_complete,
        fp,
        "replayable LispSyntax rewrite-program trace over GATlab/AlgebraicJulia execution",
    )
end

lisp_gatlab_rewrite_program_trace(w::LispGATBridgeWorld, program::LispGATRewriteProgram) =
    lisp_gatlab_rewrite_program_trace(lisp_gatlab_rewrite_program_execution(w, program))

lisp_gatlab_rewrite_program_trace(program::LispGATRewriteProgram) =
    lisp_gatlab_rewrite_program_trace(world_lisp_gatlab_bridge(), program)

lisp_gatlab_rewrite_program_trace(form) =
    lisp_gatlab_rewrite_program_trace(parse_lisp_gatlab_rewrite_program(form))

function _lisp_gat_theory_name(name::Symbol)
    s = String(name)
    startswith(s, "Sch") ? Symbol("Th", s[4:end]) : Symbol("Th", s)
end

function _lisp_gat_arg_name(i::Integer)
    Symbol("x", i)
end

function _lisp_gat_path_start(path::Vector{Symbol}, morphisms::Dict{Symbol,LispGATMorphism})
    isempty(path) && return nothing
    mor = get(morphisms, first(path), nothing)
    mor === nothing ? nothing : mor.dom
end

function _lisp_gat_apply_path(path::Vector{Symbol}, arg::Symbol)
    expr = String(arg)
    for name in path
        expr = string(name, "(", expr, ")")
    end
    expr
end

function render_lisp_gatlab_theory(w::LispGATBridgeWorld=world_lisp_gatlab_bridge())
    morphism_by_name = Dict(mor.name => mor for mor in w.morphisms)
    io = IOBuffer()
    println(io, "# Generated projection artifact; requires GATlab to execute.")
    println(io, "# fingerprint: 0x", string(w.fingerprint, base=16, pad=16))
    println(io, "@theory ", _lisp_gat_theory_name(w.presentation_name), " begin")
    for ob in w.objects
        println(io, "    ", ob.name, "::TYPE")
    end
    println(io)
    for (i, mor) in enumerate(w.morphisms)
        arg = _lisp_gat_arg_name(i)
        println(io, "    ", mor.name, "(", arg, "::", mor.dom, ")::", mor.cod)
    end
    println(io)
    for (i, eq) in enumerate(w.equations)
        start = _lisp_gat_path_start(eq.lhs, morphism_by_name)
        start_rhs = _lisp_gat_path_start(eq.rhs, morphism_by_name)
        arg = _lisp_gat_arg_name(i)
        if start === nothing || start_rhs === nothing || start != start_rhs
            println(io, "    # law ", i, ": ", join(string.(eq.lhs), " ; "), " == ",
                join(string.(eq.rhs), " ; "))
        else
            println(io, "    # law ", i, " under ", arg, "::", start)
            println(io, "    # ", _lisp_gat_apply_path(eq.lhs, arg), " == ",
                _lisp_gat_apply_path(eq.rhs, arg))
        end
    end
    println(io, "end")
    String(take!(io))
end

function render_algebraicjulia_projection(
    w::LispGATBridgeWorld=world_lisp_gatlab_bridge();
    style::Symbol=:gatlab_theory,
)
    if style == :gatlab_theory
        render_lisp_gatlab_theory(w)
    elseif style == :catlab_present
        render_lisp_gatlab_presentation(w)
    elseif style == :sexp
        render_lisp_gatlab_bridge(w)
    elseif style == :json
        render_lisp_gatlab_json(w)
    else
        error("Unknown AlgebraicJulia projection style: $style")
    end
end
