#!/usr/bin/env julia

using Gay
using JSON3

root = dirname(@__DIR__)
artifact_dir = joinpath(root, "artifacts")
mkpath(artifact_dir)

w = world_lisp_gatlab_bridge()
summary = lisp_gatlab_bridge_summary(w)
coverage = lisp_gatlab_counterfactual_coverage(w)
status = algebraicjulia_bridge_status()
realization = realize_lisp_gatlab_bridge(w)
materialization = materialize_lisp_gatlab_bridge(w)
rewrite_candidates = lisp_gatlab_rewrite_candidates(w)
first_candidate = first(rewrite_candidates)
query_samples = [
    lisp_gatlab_query(:ordinal, first_candidate.ordinal),
    lisp_gatlab_query(:witness, first_candidate.witness_ordinal),
    lisp_gatlab_query(:effect, first_candidate.closure_effect),
    lisp_gatlab_query(:between, first_candidate.from_aspect, first_candidate.to_aspect),
    lisp_gatlab_query(:limit, 3),
]
rewrite_request_samples = [
    lisp_gatlab_rewrite_request(q.operation, q.arguments...; max_samples=min(3, length(q.matches)))
    for q in query_samples
]
rewrite_plan_samples = [
    lisp_gatlab_rewrite_plan(w, request)
    for request in rewrite_request_samples
]
rewrite_execution_samples = [
    lisp_gatlab_rewrite_execution(plan; backend=:projection)
    for plan in rewrite_plan_samples
]
rewrite_program_sample = parse_lisp_gatlab_rewrite_program(default_lisp_gatlab_rewrite_program_form())
rewrite_program_execution_sample = lisp_gatlab_rewrite_program_execution(w, rewrite_program_sample)
rewrite_program_trace_sample = lisp_gatlab_rewrite_program_trace(rewrite_program_execution_sample)
rewrite_program_trace_form = render_lisp_gatlab_rewrite_program_trace(rewrite_program_trace_sample)
rewrite_program_trace_validation = validate_lisp_gatlab_rewrite_program_trace_form(
    rewrite_program_trace_form;
    bridge=w,
)

function symbol_counts(d::Dict{Symbol,Int})
    Dict(String(k) => v for (k, v) in sort(collect(d); by=x -> String(first(x))))
end

payload = Dict(
    "summary" => Dict(
        "presentation_name" => String(summary.presentation_name),
        "objects" => summary.objects,
        "morphisms" => summary.morphisms,
        "equations" => summary.equations,
        "counterfactuals" => summary.counterfactuals,
        "rewrite_candidates" => summary.rewrite_candidates,
        "unique_rewrite_candidates" => summary.unique_rewrite_candidates,
        "effect_counts" => symbol_counts(summary.effect_counts),
        "all_counterfactuals_considered" => summary.all_counterfactuals_considered,
        "all_rewrite_candidates_considered" => summary.all_rewrite_candidates_considered,
        "parser" => String(summary.parser),
        "fingerprint" => string("0x", string(summary.fingerprint, base=16, pad=16)),
    ),
    "counterfactual_coverage" => Dict(
        "counterfactuals" => coverage.counterfactuals,
        "rewrite_candidates" => coverage.rewrite_candidates,
        "unique_rewrite_candidates" => coverage.unique_rewrite_candidates,
        "witnesses" => coverage.witnesses,
        "expected_per_witness" => coverage.expected_per_witness,
        "min_per_witness" => coverage.min_per_witness,
        "max_per_witness" => coverage.max_per_witness,
        "no_duplicate_edges" => coverage.no_duplicate_edges,
        "complete" => coverage.complete,
    ),
    "algebraicjulia_status" => Dict(
        "available" => String.(status.available),
        "missing" => String.(status.missing),
        "hard_dependencies_added" => status.hard_dependencies_added,
        "weak_dependencies_declared" => status.weak_dependencies_declared,
        "extension_name" => String(status.extension_name),
        "extension_loaded" => status.extension_loaded,
        "projection_styles" => String.(status.projection_styles),
        "realization_backends" => String.(status.realization_backends),
        "materialization_backends" => String.(status.materialization_backends),
        "capabilities" => [
            Dict(
                "package" => String(cap.package),
                "uuid" => cap.uuid,
                "role" => cap.role,
                "available" => cap.available,
                "load_path" => cap.load_path,
            )
            for cap in status.capabilities
        ],
    ),
    "realization" => Dict(
        "extension" => String(realization.extension),
        "backend" => String(realization.backend),
        "packages" => String.(realization.packages),
        "parser" => String(realization.parser),
        "acset_hint" => realization.acset_hint,
        "rewriting_hint" => realization.rewriting_hint,
        "fingerprint" => string("0x", string(realization.fingerprint, base=16, pad=16)),
    ),
    "materialization" => Dict(
        "extension" => String(materialization.extension),
        "backend" => String(materialization.backend),
        "packages" => String.(materialization.packages),
        "presentation_type" => materialization.presentation_type,
        "generator_counts" => symbol_counts(materialization.generator_counts),
        "equation_count" => materialization.equation_count,
        "counterfactual_count" => materialization.counterfactual_count,
        "rewrite_candidate_count" => materialization.rewrite_candidate_count,
        "rewrite_candidate_fingerprints" => [
            string("0x", string(cand.fingerprint, base=16, pad=16))
            for cand in materialization.rewrite_candidates
        ],
        "fingerprint" => string("0x", string(materialization.fingerprint, base=16, pad=16)),
    ),
    "query_samples" => [
        lisp_gatlab_query_payload(q)
        for q in query_samples
    ],
    "rewrite_request_samples" => [
        lisp_gatlab_rewrite_request_payload(request)
        for request in rewrite_request_samples
    ],
    "rewrite_plan_samples" => [
        lisp_gatlab_rewrite_plan_payload(plan)
        for plan in rewrite_plan_samples
    ],
    "rewrite_execution_samples" => [
        lisp_gatlab_rewrite_execution_payload(exec)
        for exec in rewrite_execution_samples
    ],
    "rewrite_program_sample" => lisp_gatlab_rewrite_program_payload(rewrite_program_sample),
    "rewrite_program_execution_sample" => lisp_gatlab_rewrite_program_execution_payload(rewrite_program_execution_sample),
    "rewrite_program_trace_sample" => lisp_gatlab_rewrite_program_trace_payload(rewrite_program_trace_sample),
    "rewrite_program_trace_form" => rewrite_program_trace_form,
    "rewrite_program_trace_validation" =>
        lisp_gatlab_rewrite_trace_validation_payload(rewrite_program_trace_validation),
    "source" => w.source,
    "parser" => String(w.parser),
    "seed" => string(w.seed),
    "objects" => [
        Dict(
            "name" => String(ob.name),
            "kind" => String(ob.kind),
            "color_hex" => ob.color_hex,
            "evidence" => ob.evidence,
        )
        for ob in w.objects
    ],
    "morphisms" => [
        Dict(
            "name" => String(mor.name),
            "kind" => String(mor.kind),
            "dom" => String(mor.dom),
            "cod" => String(mor.cod),
            "color_hex" => mor.color_hex,
            "evidence" => mor.evidence,
        )
        for mor in w.morphisms
    ],
    "equations" => [
        Dict(
            "lhs" => String.(eq.lhs),
            "rhs" => String.(eq.rhs),
            "color_hex" => eq.color_hex,
            "evidence" => eq.evidence,
        )
        for eq in w.equations
    ],
    "counterfactuals" => [
        Dict(
            "witness_ordinal" => cf.witness_ordinal,
            "from_aspect" => String(cf.from_aspect),
            "to_aspect" => String(cf.to_aspect),
            "trit_delta" => cf.trit_delta,
            "closure_effect" => String(cf.closure_effect),
            "color_hex" => cf.color_hex,
            "semantic_cost" => cf.semantic_cost,
            "lhs" => String.(cf.lhs),
            "rhs" => String.(cf.rhs),
        )
        for cf in w.counterfactuals
    ],
    "rewrite_candidates" => [
        Dict(
            "ordinal" => cand.ordinal,
            "counterfactual_index" => cand.counterfactual_index,
            "witness_ordinal" => cand.witness_ordinal,
            "from_aspect" => String(cand.from_aspect),
            "to_aspect" => String(cand.to_aspect),
            "rule_style" => String(cand.rule_style),
            "match_path" => String.(cand.match_path),
            "source_path" => String.(cand.source_path),
            "target_path" => String.(cand.target_path),
            "arena_path" => String.(cand.arena_path),
            "witness_arena_path" => String.(cand.witness_arena_path),
            "trit_delta" => cand.trit_delta,
            "closure_effect" => String(cand.closure_effect),
            "color_hex" => cand.color_hex,
            "semantic_cost" => cand.semantic_cost,
            "fingerprint" => string("0x", string(cand.fingerprint, base=16, pad=16)),
        )
        for cand in rewrite_candidates
    ],
)

open(joinpath(artifact_dir, "lisp_gatlab_bridge_world.json"), "w") do io
    JSON3.pretty(io, payload)
    println(io)
end

open(joinpath(artifact_dir, "lisp_gatlab_realization_plan.json"), "w") do io
    JSON3.pretty(io, payload["realization"])
    println(io)
end

open(joinpath(artifact_dir, "lisp_gatlab_materialization_plan.json"), "w") do io
    JSON3.pretty(io, payload["materialization"])
    println(io)
end

open(joinpath(artifact_dir, "lisp_gatlab_bridge_world.sxp"), "w") do io
    write(io, render_lisp_gatlab_bridge(w))
end

open(joinpath(artifact_dir, "lisp_gatlab_source.lisp"), "w") do io
    write(io, default_lisp_gatlab_form())
end

open(joinpath(artifact_dir, "lisp_gatlab_rewrite_program_trace.lisp"), "w") do io
    write(io, rewrite_program_trace_form)
end

open(joinpath(artifact_dir, "lisp_gatlab_rewrite_program_trace_validation.json"), "w") do io
    JSON3.pretty(io, lisp_gatlab_rewrite_trace_validation_payload(rewrite_program_trace_validation))
    println(io)
end

open(joinpath(artifact_dir, "lisp_gatlab_entrypoint.jl"), "w") do io
    println(io, "# Generated executable LispSyntax entrypoint artifact.")
    println(io, "# It requires Gay.jl and uses the gat\"...\" and gat_rewrite\"...\" string macros.")
    println(io, "using Gay")
    println(io)
    println(io, "const LISP_GATLAB_BRIDGE_WORLD = gat\"\"\"")
    write(io, default_lisp_gatlab_form())
    println(io, "\"\"\"")
    println(io)
    println(io, "const LISP_GATLAB_REWRITE_TRACE_FORM = raw\"\"\"")
    write(io, rewrite_program_trace_form)
    println(io, "\"\"\"")
    println(io)
    println(io, "lisp_gatlab_bridge_summary(LISP_GATLAB_BRIDGE_WORLD)")
    println(io, "parse_lisp_gatlab_rewrite_form(default_lisp_gatlab_rewrite_form())")
    println(io, "gat_rewrite\"\"\"(rewrite-execution (query ordinal 1) (max-samples 1) (backend projection))\"\"\"")
    println(io, "parse_lisp_gatlab_rewrite_program(default_lisp_gatlab_rewrite_program_form())")
    println(io, "default_lisp_gatlab_rewrite_trace_form()")
    println(io, "gat_rewrite_program\"\"\"(rewrite-program (rewrite-execution (query ordinal 1) (max-samples 1) (backend projection)))\"\"\"")
    println(io, "lisp_gatlab_query(:ordinal, 1)")
    println(io, "lisp_gatlab_rewrite_plan(:ordinal, 1)")
    println(io, "lisp_gatlab_rewrite_execution(lisp_gatlab_rewrite_plan(:ordinal, 1), :projection)")
    println(io, "lisp_gatlab_rewrite_plan(parse_lisp_gatlab_rewrite_form(\"(rewrite-execution (query ordinal 1) (max-samples 1) (backend projection))\"))")
    println(io, "eval(lisp_gatlab_rewrite_compile(\"(rewrite-execution (query ordinal 1) (max-samples 1) (backend projection))\"; target=:plan))")
    println(io, "eval(lisp_gatlab_rewrite_program_compile(\"(rewrite-program (rewrite-execution (query ordinal 1) (max-samples 1) (backend projection)))\"; target=:execution))")
    println(io, "eval(lisp_gatlab_rewrite_program_compile(\"(rewrite-program (rewrite-execution (query ordinal 1) (max-samples 1) (backend projection)))\"; target=:trace))")
    println(io, "eval(lisp_gatlab_rewrite_program_compile(\"(rewrite-program (rewrite-execution (query ordinal 1) (max-samples 1) (backend projection)))\"; target=:trace_form))")
    println(io, "eval(lisp_gatlab_rewrite_trace_compile(LISP_GATLAB_REWRITE_TRACE_FORM; target=:validation))")
    println(io, "eval(lisp_gatlab_rewrite_trace_compile(LISP_GATLAB_REWRITE_TRACE_FORM; target=:validation_json))")
    println(io, "sexp_eval(\"(lisp-gatlab-query 'ordinal 1)\", Gay)")
    println(io, "sexp_eval(\"(lisp-gatlab-rewrite-plan 'ordinal 1)\", Gay)")
    println(io, "sexp_eval(\"(lisp-gatlab-rewrite-execution (lisp-gatlab-rewrite-plan 'ordinal 1) 'projection)\", Gay)")
    println(io, "sexp_eval(\"(lisp-gatlab-rewrite-trace-validation-payload (default-lisp-gatlab-rewrite-trace-form))\", Gay)")
end

open(joinpath(artifact_dir, "lisp_gatlab_presentation.jl"), "w") do io
    write(io, render_lisp_gatlab_presentation(w))
end

open(joinpath(artifact_dir, "lisp_gatlab_theory.jl"), "w") do io
    write(io, render_lisp_gatlab_theory(w))
end

open(joinpath(artifact_dir, "lisp_gatlab_counterfactuals.tsv"), "w") do io
    println(io, "witness_ordinal\tfrom_aspect\tto_aspect\ttrit_delta\tclosure_effect\tcolor_hex\tsemantic_cost")
    for cf in w.counterfactuals
        println(io, join((
            cf.witness_ordinal,
            cf.from_aspect,
            cf.to_aspect,
            cf.trit_delta,
            cf.closure_effect,
            cf.color_hex,
            cf.semantic_cost,
        ), '\t'))
    end
end

open(joinpath(artifact_dir, "lisp_gatlab_rewrite_candidates.tsv"), "w") do io
    println(io, "ordinal\tcounterfactual_index\twitness_ordinal\tfrom_aspect\tto_aspect\trule_style\tmatch_path\tsource_path\ttarget_path\tarena_path\twitness_arena_path\ttrit_delta\tclosure_effect\tcolor_hex\tsemantic_cost\tfingerprint")
    for cand in rewrite_candidates
        println(io, join((
            cand.ordinal,
            cand.counterfactual_index,
            cand.witness_ordinal,
            cand.from_aspect,
            cand.to_aspect,
            cand.rule_style,
            join(String.(cand.match_path), ";"),
            join(String.(cand.source_path), ";"),
            join(String.(cand.target_path), ";"),
            join(String.(cand.arena_path), ";"),
            join(String.(cand.witness_arena_path), ";"),
            cand.trit_delta,
            cand.closure_effect,
            cand.color_hex,
            cand.semantic_cost,
            string("0x", string(cand.fingerprint, base=16, pad=16)),
        ), '\t'))
    end
end

println("Wrote Lisp/GATlab bridge artifacts to ", artifact_dir)
println("fingerprint: 0x", string(w.fingerprint, base=16, pad=16))
println("parser: ", w.parser)
println("counterfactuals: ", length(w.counterfactuals))
println("rewrite candidates: ", length(rewrite_candidates))
