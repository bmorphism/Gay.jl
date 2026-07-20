using Test
using Gay
using JSON3

@testset "Lisp/GATlab Bridge" begin
    w = world_lisp_gatlab_bridge()
    source = world_gay_test_olog_counterfactuals()

    @test w isa LispGATBridgeWorld
    @test length(w.objects) == 11
    @test count(ob -> ob.kind == :ob, w.objects) == 6
    @test count(ob -> ob.kind == :attrtype, w.objects) == 5
    @test length(w.morphisms) == 15
    @test count(mor -> mor.kind == :hom, w.morphisms) == 10
    @test count(mor -> mor.kind == :attr, w.morphisms) == 5
    @test length(w.equations) == 4
    @test length(w.counterfactuals) == length(source.counterfactuals)
    @test w.parser == :lispsyntax

    summary = lisp_gatlab_bridge_summary(w)
    coverage = lisp_gatlab_counterfactual_coverage(w)
    candidates = lisp_gatlab_rewrite_candidates(w)
    @test summary.all_counterfactuals_considered
    @test summary.all_rewrite_candidates_considered
    @test summary.rewrite_candidates == length(source.counterfactuals)
    @test summary.unique_rewrite_candidates == length(source.counterfactuals)
    @test summary.effect_counts[:positive_shift] == summary.effect_counts[:negative_shift]
    @test summary.fingerprint == w.fingerprint
    @test summary.parser == :lispsyntax
    @test coverage.complete
    @test coverage.rewrite_candidates == length(source.counterfactuals)
    @test coverage.unique_rewrite_candidates == length(source.counterfactuals)
    @test coverage.min_per_witness == 14
    @test coverage.max_per_witness == 14
    @test coverage.no_duplicate_edges
    @test length(candidates) == length(source.counterfactuals)
    @test first(candidates) isa LispGATRewriteCandidate
    @test first(candidates).source_path == [:has_counterfactual, :from_aspect]
    @test first(candidates).target_path == [:has_counterfactual, :to_aspect]
    @test first(candidates).arena_path == [:has_counterfactual, :shared_in]
    @test first(candidates).witness_arena_path == [:witness_arena]
    @test length(Set(cand.fingerprint for cand in candidates)) == length(candidates)

    first_candidate = first(candidates)
    ordinal_query = lisp_gatlab_query(:ordinal, first_candidate.ordinal)
    @test ordinal_query isa LispGATQueryResult
    @test ordinal_query.operation == :ordinal
    @test only(ordinal_query.matches).fingerprint == first_candidate.fingerprint
    @test ordinal_query.coverage.complete
    @test ordinal_query.fingerprint == lisp_gatlab_query(:ordinal, first_candidate.ordinal).fingerprint

    witness_query = lisp_gatlab_query(:witness, first_candidate.witness_ordinal)
    @test length(witness_query.matches) == coverage.expected_per_witness
    @test all(cand -> cand.witness_ordinal == first_candidate.witness_ordinal, witness_query.matches)

    effect_query = lisp_gatlab_query(:effect, first_candidate.closure_effect)
    @test !isempty(effect_query.matches)
    @test all(cand -> cand.closure_effect == first_candidate.closure_effect, effect_query.matches)

    between_query = lisp_gatlab_query(:between, first_candidate.from_aspect, first_candidate.to_aspect)
    @test !isempty(between_query.matches)
    @test all(
        cand -> cand.from_aspect == first_candidate.from_aspect &&
            cand.to_aspect == first_candidate.to_aspect,
        between_query.matches,
    )

    color_query = lisp_gatlab_query(:color, first_candidate.color_hex)
    @test !isempty(color_query.matches)
    @test all(cand -> cand.color_hex == first_candidate.color_hex, color_query.matches)

    limit_query = lisp_gatlab_query(:limit, 3)
    @test length(limit_query.matches) == 3
    @test [cand.fingerprint for cand in lisp_gatlab_query(:all).matches] ==
        [cand.fingerprint for cand in candidates]

    rewrite_plan = lisp_gatlab_rewrite_plan(:witness, first_candidate.witness_ordinal; max_samples=2)
    @test rewrite_plan isa LispGATRewritePlan
    @test rewrite_plan.query.operation == :witness
    @test rewrite_plan.sample_ordinals == [cand.ordinal for cand in Iterators.take(witness_query.matches, 2)]
    @test rewrite_plan.sample_mode == :ordinals
    @test rewrite_plan.max_samples == 2
    @test rewrite_plan.materialization_backend == :algebraicjulia
    @test rewrite_plan.bridge_fingerprint == w.fingerprint
    @test rewrite_plan.fingerprint ==
        lisp_gatlab_rewrite_plan(:witness, first_candidate.witness_ordinal; max_samples=2).fingerprint

    rewrite_request_form = """
    (rewrite-execution
      (query witness 1)
      (max-samples 2)
      (backend projection))
    """
    rewrite_request = parse_lisp_gatlab_rewrite_form(rewrite_request_form)
    @test rewrite_request isa LispGATRewriteRequest
    @test rewrite_request.operation == :witness
    @test rewrite_request.arguments == [1]
    @test rewrite_request.max_samples == 2
    @test rewrite_request.backend == :projection
    @test rewrite_request.parser == :lispsyntax
    @test rewrite_request.fingerprint == parse_lisp_gatlab_rewrite_form(rewrite_request_form).fingerprint

    constructed_request = lisp_gatlab_rewrite_request(:ordinal, first_candidate.ordinal; max_samples=1, backend=:projection)
    @test constructed_request.operation == :ordinal
    @test constructed_request.arguments == [first_candidate.ordinal]
    @test constructed_request.max_samples == 1
    @test constructed_request.backend == :projection
    @test constructed_request.parser == :constructed

    request_query = lisp_gatlab_query(w, rewrite_request)
    @test request_query.operation == :witness
    @test length(request_query.matches) == coverage.expected_per_witness

    request_plan = lisp_gatlab_rewrite_plan(w, rewrite_request)
    @test request_plan.sample_ordinals == [1, 2]
    @test request_plan.materialization_backend == :projection

    request_execution = lisp_gatlab_rewrite_execution(w, rewrite_request)
    @test request_execution.backend == :projection
    @test request_execution.selected_ordinals == [1, 2]
    @test request_execution.spec_count == length(source.counterfactuals)

    compiled_request_expr = lisp_gatlab_rewrite_compile(rewrite_request_form)
    compiled_plan_expr = lisp_gatlab_rewrite_compile(rewrite_request_form; target=:plan)
    compiled_request_json_expr = lisp_gatlab_rewrite_compile(rewrite_request_form; target=:request_json)
    compiled_execution_json_expr = lisp_gatlab_rewrite_compile(rewrite_request_form; target=:execution_json)
    @test compiled_request_expr isa Expr
    @test compiled_plan_expr isa Expr
    @test eval(compiled_request_expr).fingerprint == rewrite_request.fingerprint
    @test eval(compiled_plan_expr).sample_ordinals == [1, 2]
    @test JSON3.read(eval(compiled_request_json_expr)).operation == "witness"
    @test JSON3.read(eval(compiled_execution_json_expr)).request.operation == "witness"
    @test_throws ErrorException lisp_gatlab_rewrite_compile(rewrite_request_form; target=:unknown)

    macro_request = gat_rewrite"""
    (rewrite-execution
      (query witness 1)
      (max-samples 2)
      (backend projection))
    """
    @test macro_request isa LispGATRewriteRequest
    @test macro_request.fingerprint == rewrite_request.fingerprint
    @test macro_request.backend == :projection

    rewrite_program_form = """
    (rewrite-program
      (rewrite-execution
        (query witness 1)
        (max-samples 2)
        (backend projection))
      (rewrite-execution
        (query effect positive-shift)
        (max-samples 1)
        (backend projection)))
    """
    rewrite_program = parse_lisp_gatlab_rewrite_program(rewrite_program_form)
    @test rewrite_program isa LispGATRewriteProgram
    @test length(rewrite_program.requests) == 2
    @test rewrite_program.requests[1].operation == :witness
    @test rewrite_program.requests[2].operation == :effect
    @test all(request -> request.backend == :projection, rewrite_program.requests)
    @test rewrite_program.fingerprint == parse_lisp_gatlab_rewrite_program(rewrite_program_form).fingerprint
    @test length(parse_lisp_gatlab_rewrite_program(default_lisp_gatlab_rewrite_program_form()).requests) == 3

    program_plans = [lisp_gatlab_rewrite_plan(w, request) for request in rewrite_program.requests]
    expected_program_ordinals = unique(vcat([plan.sample_ordinals for plan in program_plans]...))
    @test all(cand -> cand.closure_effect == :positive_shift, lisp_gatlab_query(w, rewrite_program.requests[2]).matches)

    compiled_program_expr = lisp_gatlab_rewrite_program_compile(rewrite_program_form)
    compiled_program_execution_expr =
        lisp_gatlab_rewrite_program_compile(rewrite_program_form; target=:execution)
    compiled_program_trace_expr =
        lisp_gatlab_rewrite_program_compile(rewrite_program_form; target=:trace)
    compiled_program_json_expr =
        lisp_gatlab_rewrite_program_compile(rewrite_program_form; target=:program_json)
    compiled_program_execution_json_expr =
        lisp_gatlab_rewrite_program_compile(rewrite_program_form; target=:execution_json)
    compiled_program_trace_json_expr =
        lisp_gatlab_rewrite_program_compile(rewrite_program_form; target=:trace_json)
    @test compiled_program_expr isa Expr
    @test eval(compiled_program_expr).fingerprint == rewrite_program.fingerprint
    @test eval(compiled_program_execution_expr).selected_ordinals == expected_program_ordinals
    @test eval(compiled_program_trace_expr).execution.selected_ordinals == expected_program_ordinals
    @test JSON3.read(eval(compiled_program_json_expr)).request_count == 2
    @test JSON3.read(eval(compiled_program_execution_json_expr)).execution_count == 2
    @test JSON3.read(eval(compiled_program_trace_json_expr)).step_count == 2
    @test_throws ErrorException lisp_gatlab_rewrite_program_compile(rewrite_program_form; target=:unknown)

    macro_program = gat_rewrite_program"""
    (rewrite-program
      (rewrite-execution
        (query witness 1)
        (max-samples 2)
        (backend projection))
      (rewrite-execution
        (query effect positive-shift)
        (max-samples 1)
        (backend projection)))
    """
    @test macro_program isa LispGATRewriteProgram
    @test macro_program.fingerprint == rewrite_program.fingerprint

    program_payload = lisp_gatlab_rewrite_program_payload(rewrite_program)
    @test program_payload["request_count"] == 2
    @test length(program_payload["requests"]) == 2

    program_execution = lisp_gatlab_rewrite_program_execution(w, rewrite_program)
    @test program_execution isa LispGATRewriteProgramExecution
    @test program_execution.backends == [:projection, :projection]
    @test program_execution.selected_ordinals == expected_program_ordinals
    @test !program_execution.all_selected_all_materialized
    @test !program_execution.all_selected_all_targets

    program_trace = lisp_gatlab_rewrite_program_trace(program_execution)
    @test program_trace isa LispGATRewriteProgramTrace
    @test length(program_trace.steps) == length(rewrite_program.requests)
    @test program_trace.coverage_complete
    @test program_trace.execution.fingerprint == program_execution.fingerprint
    @test program_trace.steps[1] isa LispGATRewriteProgramStep
    @test program_trace.steps[1].index == 1
    @test program_trace.steps[1].request.fingerprint == rewrite_program.requests[1].fingerprint
    @test program_trace.steps[1].plan.fingerprint == program_plans[1].fingerprint
    @test program_trace.steps[1].selected_ordinals == program_execution.executions[1].selected_ordinals
    @test program_trace.steps[1].introduced_ordinals == program_execution.executions[1].selected_ordinals
    @test all(step -> step.backend == :projection, program_trace.steps)

    seen_ordinals = Set{Int}()
    repeated_seen = Set{Int}()
    expected_repeated_ordinals = Int[]
    expected_introduced_by_step = Vector{Int}[]
    for exec in program_execution.executions
        introduced = Int[]
        for ordinal in exec.selected_ordinals
            if ordinal in seen_ordinals
                if !(ordinal in repeated_seen)
                    push!(repeated_seen, ordinal)
                    push!(expected_repeated_ordinals, ordinal)
                end
            else
                push!(seen_ordinals, ordinal)
                push!(introduced, ordinal)
            end
        end
        push!(expected_introduced_by_step, introduced)
    end
    @test program_trace.repeated_ordinals == expected_repeated_ordinals
    @test [step.introduced_ordinals for step in program_trace.steps] == expected_introduced_by_step

    program_execution_payload = lisp_gatlab_rewrite_program_execution_payload(program_execution)
    @test program_execution_payload["execution_count"] == 2
    @test program_execution_payload["selected_ordinals"] == expected_program_ordinals
    program_trace_payload = lisp_gatlab_rewrite_program_trace_payload(program_trace)
    @test program_trace_payload["step_count"] == 2
    @test program_trace_payload["repeated_ordinals"] == expected_repeated_ordinals
    @test program_trace_payload["steps"][1]["introduced_ordinals"] == expected_introduced_by_step[1]
    program_execution_json = JSON3.read(render_lisp_gatlab_rewrite_program_execution_json(program_execution))
    @test program_execution_json.program.request_count == 2
    @test collect(program_execution_json.selected_ordinals) == expected_program_ordinals
    program_trace_json = JSON3.read(render_lisp_gatlab_rewrite_program_trace_json(program_trace))
    @test program_trace_json.step_count == 2
    @test program_trace_json.coverage_complete
    @test collect(program_trace_json.selected_ordinals) == expected_program_ordinals
    program_form = render_lisp_gatlab_rewrite_program(rewrite_program)
    @test startswith(strip(program_form), "(rewrite-program")
    @test occursin("(rewrite-execution", program_form)
    parsed_program_form = parse_lisp_gatlab_rewrite_program(program_form)
    @test length(parsed_program_form.requests) == length(rewrite_program.requests)
    @test parsed_program_form.requests[2].operation == rewrite_program.requests[2].operation
    @test parsed_program_form.requests[2].arguments == rewrite_program.requests[2].arguments
    program_trace_form = render_lisp_gatlab_rewrite_program_trace(program_trace)
    @test startswith(strip(program_trace_form), "(rewrite-trace")
    @test occursin("(step 1", program_trace_form)
    @test occursin("(selected-ordinals", program_trace_form)
    @test occursin("(introduced-ordinals", program_trace_form)
    @test first(sexp_read(program_trace_form)) == Symbol("rewrite-trace")
    parsed_trace_form = parse_lisp_gatlab_rewrite_program_trace_form(program_trace_form)
    @test parsed_trace_form.fingerprint == string("0x", string(program_trace.fingerprint, base=16, pad=16))
    @test parsed_trace_form.program_fingerprint == string("0x", string(rewrite_program.fingerprint, base=16, pad=16))
    @test parsed_trace_form.selected_ordinals == expected_program_ordinals
    @test parsed_trace_form.repeated_ordinals == expected_repeated_ordinals
    @test parsed_trace_form.backends == [:projection, :projection]
    @test parsed_trace_form.steps[1].request.operation == rewrite_program.requests[1].operation
    @test parsed_trace_form.steps[1].selected_ordinals == program_trace.steps[1].selected_ordinals
    trace_validation = validate_lisp_gatlab_rewrite_program_trace_form(program_trace_form; bridge=w)
    @test trace_validation.valid
    @test all(values(trace_validation.comparisons))
    trace_validation_payload = lisp_gatlab_rewrite_trace_validation_payload(trace_validation)
    @test trace_validation_payload["valid"]
    @test trace_validation_payload["comparisons"]["fingerprint"]
    trace_validation_json = JSON3.read(render_lisp_gatlab_rewrite_trace_validation_json(trace_validation))
    @test trace_validation_json.valid
    default_trace_form = default_lisp_gatlab_rewrite_trace_form()
    @test startswith(strip(default_trace_form), "(rewrite-trace")
    default_trace_validation = lisp_gatlab_rewrite_trace_validation(default_trace_form; bridge=w)
    @test default_trace_validation.valid
    @test lisp_gatlab_rewrite_trace_validation_payload(default_trace_form; bridge=w)["valid"]
    @test JSON3.read(render_lisp_gatlab_rewrite_trace_validation_json(default_trace_form; bridge=w)).valid
    compiled_trace_parsed_expr = lisp_gatlab_rewrite_trace_compile(program_trace_form)
    compiled_trace_program_expr =
        lisp_gatlab_rewrite_trace_compile(program_trace_form; target=:program)
    compiled_trace_validation_expr =
        lisp_gatlab_rewrite_trace_compile(program_trace_form; target=:validation)
    compiled_trace_payload_expr =
        lisp_gatlab_rewrite_trace_compile(program_trace_form; target=:validation_payload)
    compiled_trace_json_expr =
        lisp_gatlab_rewrite_trace_compile(program_trace_form; target=:validation_json)
    compiled_trace_form_expr =
        lisp_gatlab_rewrite_trace_compile(program_trace_form; target=:trace_form)
    @test compiled_trace_parsed_expr isa Expr
    @test eval(compiled_trace_parsed_expr).fingerprint == parsed_trace_form.fingerprint
    @test eval(compiled_trace_program_expr).fingerprint == rewrite_program.fingerprint
    @test eval(compiled_trace_validation_expr).valid
    @test eval(compiled_trace_payload_expr)["valid"]
    @test JSON3.read(eval(compiled_trace_json_expr)).valid
    @test startswith(strip(eval(compiled_trace_form_expr)), "(rewrite-trace")
    @test_throws ErrorException lisp_gatlab_rewrite_trace_compile(program_trace_form; target=:unknown)
    tampered_trace_form = replace(program_trace_form, "(coverage-complete true)" => "(coverage-complete false)")
    tampered_validation = validate_lisp_gatlab_rewrite_program_trace_form(tampered_trace_form; bridge=w)
    @test !tampered_validation.valid
    @test !tampered_validation.comparisons[:coverage_complete]
    program_execution_with_bridge = JSON3.read(render_lisp_gatlab_rewrite_program_execution_json(
        program_execution;
        bridge=w,
        extension_status=algebraicjulia_bridge_status(),
    ))
    @test program_execution_with_bridge.bridge_counterfactuals == length(source.counterfactuals)
    @test program_execution_with_bridge.extension_status_name == "GayAlgebraicJuliaExt"
    program_trace_with_bridge = JSON3.read(render_lisp_gatlab_rewrite_program_trace_json(
        program_trace;
        bridge=w,
        extension_status=algebraicjulia_bridge_status(),
    ))
    @test program_trace_with_bridge.bridge_counterfactuals == length(source.counterfactuals)
    @test program_trace_with_bridge.extension_status_name == "GayAlgebraicJuliaExt"

    parsed = parse_lisp_gatlab_form(default_lisp_gatlab_form())
    @test parsed.parser == :lispsyntax
    @test first(parsed.objects).name == :TestWitness
    @test first(parsed.morphisms).name == :has_aspect
    @test first(parsed.equations).lhs == [:has_counterfactual, :from_aspect]
    @test first(lisp_gatlab_lispsyntax_form()) == :gat
    @test lisp_gatlab_parse_backend() == :lispsyntax

    macro_world = gat"(gat (ob MacroWitness) (attrtype MacroColor) (attr macro-color MacroWitness MacroColor))"
    @test macro_world isa LispGATBridgeWorld
    @test macro_world.parser == :lispsyntax
    @test first(macro_world.objects).name == :MacroWitness
    @test any(mor -> mor.name == :macro_color, macro_world.morphisms)

    compiled_world_expr = lisp_gatlab_compile(default_lisp_gatlab_form())
    compiled_json_expr = lisp_gatlab_compile(default_lisp_gatlab_form(); target=:json)
    compiled_request_form_expr = lisp_gatlab_rewrite_compile(
        "(rewrite-execution (query witness 1) (max-samples 2) (backend projection))";
        target=:request_form,
    )
    compiled_trace_form_expr = lisp_gatlab_rewrite_program_compile(
        "(rewrite-program (rewrite-execution (query ordinal 1) (max-samples 1) (backend projection)))";
        target=:trace_form,
    )
    @test compiled_world_expr isa Expr
    @test compiled_json_expr isa Expr
    @test compiled_request_form_expr isa Expr
    @test compiled_trace_form_expr isa Expr
    @test occursin("world_lisp_gatlab_bridge", sprint(show, compiled_world_expr))

    schema = render_lisp_gatlab_presentation(w)
    theory = render_lisp_gatlab_theory(w)
    json_projection = JSON3.read(render_algebraicjulia_projection(w; style=:json))
    declarations = lisp_gatlab_declarations(w)
    @test occursin("@present SchGayCounterfactualClosure", schema)
    @test occursin("@theory ThGayCounterfactualClosure", theory)
    @test occursin("has_aspect(x1::TestWitness)::ClosureAspect", theory)
    @test render_algebraicjulia_projection(w; style=:gatlab_theory) == theory
    @test render_algebraicjulia_projection(w; style=:catlab_present) == schema
    @test json_projection.parser == "lispsyntax"
    @test json_projection.counts.counterfactuals == length(source.counterfactuals)
    @test json_projection.counts.rewrite_candidates == length(source.counterfactuals)
    @test json_projection.all_rewrite_candidates_considered
    @test length(json_projection.rewrite_candidates) == length(source.counterfactuals)
    @test length(declarations.rewrite_candidates) == length(source.counterfactuals)
    @test eval(compiled_json_expr) == render_lisp_gatlab_json(w)

    realization = realize_lisp_gatlab_bridge(w)
    @test realization isa AlgebraicJuliaRealization
    @test realization.extension == :Gay
    @test realization.backend == :projection
    @test realization.parser == :lispsyntax
    @test occursin("@theory", realization.theory_source)
    @test occursin("@present", realization.presentation_source)

    materialization = materialize_lisp_gatlab_bridge(w)
    @test materialization isa AlgebraicJuliaMaterialization
    @test materialization.backend == :projection
    @test materialization.presentation === nothing
    @test materialization.generator_counts[:Ob] == 6
    @test materialization.generator_counts[:AttrType] == 5
    @test materialization.generator_counts[:Hom] == 10
    @test materialization.generator_counts[:Attr] == 5
    @test materialization.equation_count == 4
    @test materialization.counterfactual_count == length(source.counterfactuals)
    @test materialization.rewrite_candidate_count == length(source.counterfactuals)
    @test length(materialization.rewrite_candidates) == length(source.counterfactuals)
    @test first(materialization.rewrite_candidates) isa LispGATRewriteCandidate

    planned_materialization = materialize_lisp_gatlab_rewrite_plan(rewrite_plan; backend=:projection)
    @test planned_materialization isa AlgebraicJuliaMaterialization
    @test planned_materialization.backend == :projection
    @test planned_materialization.rewrite_candidate_count == length(source.counterfactuals)

    projection_execution = lisp_gatlab_rewrite_execution(rewrite_plan, planned_materialization)
    @test projection_execution isa LispGATRewriteExecution
    @test projection_execution.backend == :projection
    @test projection_execution.selected_ordinals == rewrite_plan.sample_ordinals
    @test isempty(projection_execution.materialized_ordinals)
    @test isempty(projection_execution.executed_ordinals)
    @test isempty(projection_execution.target_ordinals)
    @test projection_execution.spec_count == length(source.counterfactuals)
    @test projection_execution.materialized_count == 0
    @test !projection_execution.selected_all_materialized
    @test !projection_execution.selected_all_targets
    @test projection_execution.fingerprint ==
        lisp_gatlab_rewrite_execution(rewrite_plan, planned_materialization).fingerprint
    query_payload = lisp_gatlab_query_payload(ordinal_query)
    @test query_payload["operation"] == "ordinal"
    @test query_payload["arguments"] == [first_candidate.ordinal]
    @test query_payload["match_count"] == 1
    @test query_payload["match_ordinals"] == [first_candidate.ordinal]
    @test query_payload["coverage_complete"]
    @test query_payload["fingerprint"] == string("0x", string(ordinal_query.fingerprint, base=16, pad=16))

    request_payload = lisp_gatlab_rewrite_request_payload(rewrite_request)
    @test request_payload["operation"] == "witness"
    @test request_payload["arguments"] == [1]
    @test request_payload["max_samples"] == 2
    @test request_payload["backend"] == "projection"
    request_json = JSON3.read(render_lisp_gatlab_rewrite_request_json(rewrite_request))
    @test request_json.operation == "witness"
    @test request_json.backend == "projection"
    request_form = render_lisp_gatlab_rewrite_request(rewrite_request)
    @test occursin("(rewrite-execution", request_form)
    @test occursin("(query witness 1)", request_form)
    @test occursin("(backend projection)", request_form)
    parsed_request_form = parse_lisp_gatlab_rewrite_form(request_form)
    @test parsed_request_form.operation == rewrite_request.operation
    @test parsed_request_form.arguments == rewrite_request.arguments
    @test parsed_request_form.max_samples == rewrite_request.max_samples
    @test parsed_request_form.backend == rewrite_request.backend

    plan_payload = lisp_gatlab_rewrite_plan_payload(rewrite_plan)
    @test plan_payload["operation"] == "witness"
    @test plan_payload["sample_ordinals"] == rewrite_plan.sample_ordinals
    @test plan_payload["materialization_backend"] == "algebraicjulia"
    @test plan_payload["query_fingerprint"] == string("0x", string(rewrite_plan.query.fingerprint, base=16, pad=16))

    exec_payload = lisp_gatlab_rewrite_execution_payload(
        projection_execution;
        materialization=planned_materialization,
        bridge=w,
        include_selected_candidates=true,
    )
    @test exec_payload["backend"] == "projection"
    @test exec_payload["selected_ordinals"] == rewrite_plan.sample_ordinals
    @test exec_payload["bridge_counterfactuals"] == length(source.counterfactuals)
    @test exec_payload["rewrite_candidate_count"] == length(source.counterfactuals)
    @test exec_payload["selected_candidates"][1]["ordinal"] == first(rewrite_plan.sample_ordinals)

    execution_json = JSON3.read(render_lisp_gatlab_rewrite_execution_json(
        projection_execution;
        request=rewrite_request,
        materialization=planned_materialization,
        bridge=w,
        include_selected_candidates=true,
    ))
    @test execution_json.backend == "projection"
    @test collect(execution_json.selected_ordinals) == rewrite_plan.sample_ordinals
    @test execution_json.request.operation == "witness"
    @test execution_json.request.backend == "projection"
    @test execution_json.selected_all_materialized == false
    @test length(execution_json.selected_candidates) == length(rewrite_plan.sample_ordinals)
    @test eval(compiled_request_form_expr) ==
        render_lisp_gatlab_rewrite_request(parse_lisp_gatlab_rewrite_form(
            "(rewrite-execution (query witness 1) (max-samples 2) (backend projection))",
        ))
    @test occursin("(rewrite-trace", eval(compiled_trace_form_expr))

    bridge = render_lisp_gatlab_bridge(w; counterfactual_limit=1)
    @test occursin("(:truncated ", bridge)
    @test occursin("(:counterfactuals ", bridge)
    @test occursin("(:rewrite-candidates ", bridge)
    @test occursin("(gat", sexp_eval("(default-lisp-gatlab-form)", Gay))
    @test sexp_eval("(lisp-gatlab-bridge-summary)", Gay).fingerprint == w.fingerprint
    @test sexp_eval("(parse-lisp-gatlab-rewrite-form \"(query ordinal 1)\")", Gay).operation == :ordinal
    @test only(sexp_eval("(lisp-gatlab-query 'ordinal 1)", Gay).matches).fingerprint ==
        first_candidate.fingerprint
    @test sexp_eval("(lisp-gatlab-rewrite-plan 'ordinal 1)", Gay).sample_ordinals == [1]
    @test sexp_eval(
        "(lisp-gatlab-rewrite-execution (lisp-gatlab-rewrite-plan 'ordinal 1) 'projection)",
        Gay,
    ).selected_ordinals == [1]
    @test sexp_eval(
        "(lisp-gatlab-rewrite-trace-validation-payload (default-lisp-gatlab-rewrite-trace-form))",
        Gay,
    )["valid"]

    caps = algebraicjulia_capabilities()
    status = algebraicjulia_bridge_status()
    @test length(caps) == 4
    @test Set([cap.package for cap in caps]) == Set([:GATlab, :Catlab, :ACSets, :AlgebraicRewriting])
    @test all(cap -> occursin("-", cap.uuid), caps)
    @test status.hard_dependencies_added == false
    @test status.weak_dependencies_declared == true
    @test status.extension_name == :GayAlgebraicJuliaExt
    @test status.extension_loaded == false
    @test :algebraicjulia in status.realization_backends
    @test :algebraicjulia in status.materialization_backends
    @test Set(vcat(status.available, status.missing)) == Set([:GATlab, :Catlab, :ACSets, :AlgebraicRewriting])

    @test_throws ErrorException parse_lisp_gatlab_form("(not-gat)")
    @test_throws ErrorException render_algebraicjulia_projection(w; style=:unknown)
    @test_throws ErrorException lisp_gatlab_compile(default_lisp_gatlab_form(); target=:unknown)
    @test_throws ErrorException lisp_gatlab_query(:unknown)
    @test_throws ErrorException materialize_lisp_gatlab_rewrite_plan(
        world_lisp_gatlab_bridge(; source="different bridge world"),
        rewrite_plan;
        backend=:projection,
    )
    @test_throws ErrorException realize_lisp_gatlab_bridge(w, :algebraicjulia)
    @test_throws ErrorException materialize_lisp_gatlab_bridge(w, :algebraicjulia)
end
