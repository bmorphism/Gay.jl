#!/usr/bin/env julia

using Gay
using UUIDs

json_quote(x) = string(
    "\"",
    replace(String(x), "\\" => "\\\\", "\"" => "\\\"", "\n" => "\\n", "\r" => "\\r", "\t" => "\\t"),
    "\"",
)

json_array(xs) = string("[", join((json_quote(x) for x in xs), ", "), "]")
json_int_array(xs) = string("[", join(string.(xs), ", "), "]")

function json_object(d::Dict)
    pairs = sort(collect(d); by=x -> String(first(x)))
    inner = join((string(json_quote(k), ": ", v) for (k, v) in pairs), ", ")
    string("{", inner, "}")
end

function write_payload(io, payload)
    println(io, "{")
    println(io, "  \"extension_name\": ", json_quote(payload["extension_name"]), ",")
    println(io, "  \"extension_loaded\": ", payload["extension_loaded"] ? "true" : "false", ",")
    println(io, "  \"backend\": ", json_quote(payload["backend"]), ",")
    println(io, "  \"packages\": ", json_array(payload["packages"]), ",")
    println(io, "  \"parser\": ", json_quote(payload["parser"]), ",")
    println(io, "  \"bridge_fingerprint\": ", json_quote(payload["bridge_fingerprint"]), ",")
    println(io, "  \"realization_fingerprint\": ", json_quote(payload["realization_fingerprint"]), ",")
    println(io, "  \"materialization_fingerprint\": ", json_quote(payload["materialization_fingerprint"]), ",")
    println(io, "  \"presentation_type\": ", json_quote(payload["presentation_type"]), ",")
    println(io, "  \"generator_counts\": ", json_object(payload["generator_counts"]), ",")
    println(io, "  \"equation_count\": ", payload["equation_count"], ",")
    println(io, "  \"rewrite_candidate_count\": ", payload["rewrite_candidate_count"], ",")
    println(io, "  \"unique_rewrite_candidates\": ", payload["unique_rewrite_candidates"], ",")
    println(io, "  \"all_rewrite_candidates_considered\": ", payload["all_rewrite_candidates_considered"] ? "true" : "false", ",")
    println(io, "  \"first_rewrite_candidate_source_term\": ", json_quote(payload["first_rewrite_candidate_source_term"]), ",")
    println(io, "  \"first_rewrite_candidate_target_term\": ", json_quote(payload["first_rewrite_candidate_target_term"]), ",")
    println(io, "  \"first_rewrite_candidate_arena_term\": ", json_quote(payload["first_rewrite_candidate_arena_term"]), ",")
    println(io, "  \"dpo_rule_materialized_count\": ", payload["dpo_rule_materialized_count"], ",")
    println(io, "  \"dpo_rule_spec_count\": ", payload["dpo_rule_spec_count"], ",")
    println(io, "  \"dpo_sample_mode\": ", json_quote(payload["dpo_sample_mode"]), ",")
    println(io, "  \"dpo_sample_count_requested\": ", payload["dpo_sample_count_requested"], ",")
    println(io, "  \"dpo_sample_ordinals_requested\": ", json_int_array(payload["dpo_sample_ordinals_requested"]), ",")
    println(io, "  \"dpo_sample_ordinals_materialized\": ", json_int_array(payload["dpo_sample_ordinals_materialized"]), ",")
    println(io, "  \"first_dpo_rule_materialized\": ", payload["first_dpo_rule_materialized"] ? "true" : "false", ",")
    println(io, "  \"first_dpo_result_executed\": ", payload["first_dpo_result_executed"] ? "true" : "false", ",")
    println(io, "  \"first_dpo_result_is_target\": ", payload["first_dpo_result_is_target"] ? "true" : "false", ",")
    println(io, "  \"first_dpo_rule_type\": ", json_quote(payload["first_dpo_rule_type"]), ",")
    println(io, "  \"first_dpo_left_assignment_aspect\": ", payload["first_dpo_left_assignment_aspect"], ",")
    println(io, "  \"first_dpo_right_assignment_aspect\": ", payload["first_dpo_right_assignment_aspect"], ",")
    println(io, "  \"first_dpo_result_assignment_aspect\": ", payload["first_dpo_result_assignment_aspect"], ",")
    println(io, "  \"counterfactuals\": ", payload["counterfactuals"], ",")
    println(io, "  \"theory_has_gatlab_form\": ", payload["theory_has_gatlab_form"] ? "true" : "false", ",")
    println(io, "  \"presentation_has_catlab_form\": ", payload["presentation_has_catlab_form"] ? "true" : "false")
    println(io, "}")
end

function load_algebraicjulia_capabilities!()
    status = algebraicjulia_bridge_status()
    if !isempty(status.missing)
        missing = join(string.(status.missing), ", ")
        error("Missing AlgebraicJulia weak dependencies in this environment: $missing")
    end

    for cap in status.capabilities
        Base.require(Base.PkgId(UUID(cap.uuid), String(cap.package)))
    end

    algebraicjulia_bridge_status()
end

function parse_dpo_sample_count()
    raw = get(ENV, "GAY_LISP_GATLAB_DPO_SAMPLE_COUNT", "1")
    try
        max(0, parse(Int, raw))
    catch
        error("GAY_LISP_GATLAB_DPO_SAMPLE_COUNT must be a non-negative integer, got: $raw")
    end
end

function parse_dpo_sample_ordinals()
    raw = strip(get(ENV, "GAY_LISP_GATLAB_DPO_SAMPLE_ORDINALS", ""))
    isempty(raw) && return nothing
    ords = Int[]
    for item in split(raw, ",")
        s = strip(item)
        isempty(s) && continue
        try
            push!(ords, parse(Int, s))
        catch
            error("GAY_LISP_GATLAB_DPO_SAMPLE_ORDINALS must be comma-separated integers, got item: $s")
        end
    end
    ords
end

status = load_algebraicjulia_capabilities!()
w = world_lisp_gatlab_bridge()
realization = realize_lisp_gatlab_bridge(w, :algebraicjulia)
dpo_sample_count = parse_dpo_sample_count()
dpo_sample_ordinals = parse_dpo_sample_ordinals()
dpo_sample_mode = dpo_sample_ordinals === nothing ? "count" : "ordinals"
materialization = materialize_lisp_gatlab_bridge(
    w,
    :algebraicjulia;
    dpo_sample_count=dpo_sample_count,
    dpo_sample_ordinals=dpo_sample_ordinals,
)
coverage = lisp_gatlab_counterfactual_coverage(w)
first_candidate = first(materialization.rewrite_candidates)
dpo_materialized_count = count(cand -> cand.dpo_rule_materialized, materialization.rewrite_candidates)
dpo_materialized_ordinals = [
    cand.ordinal for cand in materialization.rewrite_candidates if cand.dpo_rule_materialized
]

payload = Dict(
    "extension_name" => String(status.extension_name),
    "extension_loaded" => status.extension_loaded,
    "backend" => String(realization.backend),
    "packages" => String.(realization.packages),
    "parser" => String(realization.parser),
    "bridge_fingerprint" => string("0x", string(w.fingerprint, base=16, pad=16)),
    "realization_fingerprint" => string("0x", string(realization.fingerprint, base=16, pad=16)),
    "materialization_fingerprint" => string("0x", string(materialization.fingerprint, base=16, pad=16)),
    "presentation_type" => materialization.presentation_type,
    "generator_counts" => Dict(String(k) => v for (k, v) in materialization.generator_counts),
    "equation_count" => materialization.equation_count,
    "rewrite_candidate_count" => materialization.rewrite_candidate_count,
    "unique_rewrite_candidates" => coverage.unique_rewrite_candidates,
    "all_rewrite_candidates_considered" => coverage.complete,
    "first_rewrite_candidate_source_term" => first_candidate.source_term_text,
    "first_rewrite_candidate_target_term" => first_candidate.target_term_text,
    "first_rewrite_candidate_arena_term" => first_candidate.arena_term_text,
    "dpo_rule_materialized_count" => dpo_materialized_count,
    "dpo_rule_spec_count" => materialization.rewrite_candidate_count - dpo_materialized_count,
    "dpo_sample_mode" => dpo_sample_mode,
    "dpo_sample_count_requested" => dpo_sample_count,
    "dpo_sample_ordinals_requested" => dpo_sample_ordinals === nothing ? Int[] : dpo_sample_ordinals,
    "dpo_sample_ordinals_materialized" => dpo_materialized_ordinals,
    "first_dpo_rule_materialized" => first_candidate.dpo_rule_materialized,
    "first_dpo_result_executed" => first_candidate.dpo_result_executed,
    "first_dpo_result_is_target" => first_candidate.dpo_result_is_target,
    "first_dpo_rule_type" => first_candidate.dpo_rule_type,
    "first_dpo_left_assignment_aspect" => first_candidate.left_assignment_aspect,
    "first_dpo_right_assignment_aspect" => first_candidate.right_assignment_aspect,
    "first_dpo_result_assignment_aspect" => first_candidate.result_assignment_aspect,
    "counterfactuals" => length(w.counterfactuals),
    "theory_has_gatlab_form" => occursin("@theory", realization.theory_source),
    "presentation_has_catlab_form" => occursin("@present", realization.presentation_source),
)

if isempty(ARGS)
    write_payload(stdout, payload)
else
    open(first(ARGS), "w") do io
        write_payload(io, payload)
    end
end
