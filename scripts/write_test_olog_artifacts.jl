#!/usr/bin/env julia

using Gay
using JSON3

root = dirname(@__DIR__)
artifact_dir = joinpath(root, "artifacts")
mkpath(artifact_dir)

w = world_gay_test_olog()
summary = gay_test_olog_summary(w)
cfw = world_gay_test_olog_counterfactuals(w)
cf_summary = gay_test_olog_counterfactual_summary(cfw)
decls = catcolab_olog_declarations(w)

function symbol_counts(d::Dict{Symbol,Int})
    Dict(String(k) => v for (k, v) in sort(collect(d); by=x -> String(first(x))))
end

world_payload = Dict(
    "summary" => Dict(
        "passing_tests" => summary.passing_tests,
        "broken_tests" => summary.broken_tests,
        "total_tests" => summary.total_tests,
        "aspects" => summary.aspects,
        "morphisms" => summary.morphisms,
        "equations" => summary.equations,
        "mutually_exclusive" => summary.mutually_exclusive,
        "all_passing_colored" => summary.all_passing_colored,
        "gf3_conserved" => summary.gf3_conserved,
        "fingerprint" => string("0x", string(summary.fingerprint, base=16, pad=16)),
    ),
    "seed" => string(w.seed),
    "catcolab_uri" => w.catcolab_uri,
    "catcolab_path" => w.catcolab_path,
    "aspects" => [
        Dict(
            "id" => String(aspect.id),
            "object_uri" => aspect.object_uri,
            "catcolab_uri" => aspect.catcolab_uri,
            "trit" => aspect.trit,
            "color_hex" => aspect.color_hex,
            "closure_role" => aspect.closure_role,
            "evidence" => aspect.evidence,
        )
        for aspect in w.aspects
    ],
    "morphisms" => [
        Dict(
            "name" => String(morphism.name),
            "dom" => morphism.dom,
            "cod" => morphism.cod,
            "color_hex" => morphism.color_hex,
            "description" => morphism.description,
        )
        for morphism in w.morphisms
    ],
    "equations" => [
        Dict("lhs" => first(eq), "rhs" => last(eq))
        for eq in w.equations
    ],
)

declarations_payload = Dict(
    "objects" => [Dict(String(k) => v for (k, v) in pairs(object)) for object in decls.objects],
    "morphisms" => [Dict(String(k) => v for (k, v) in pairs(morphism)) for morphism in decls.morphisms],
    "equations" => [Dict(String(k) => v for (k, v) in pairs(equation)) for equation in decls.equations],
)

counterfactual_payload = Dict(
    "summary" => Dict(
        "witness_count" => cf_summary.witness_count,
        "aspect_count" => cf_summary.aspect_count,
        "per_witness_choices" => cf_summary.per_witness_choices,
        "counterfactuals" => cf_summary.counterfactuals,
        "complete" => cf_summary.complete,
        "effect_counts" => symbol_counts(cf_summary.effect_counts),
        "min_cost" => cf_summary.min_cost,
        "mean_cost" => cf_summary.mean_cost,
        "max_cost" => cf_summary.max_cost,
        "source_fingerprint" => string("0x", string(cf_summary.source_fingerprint, base=16, pad=16)),
        "fingerprint" => string("0x", string(cf_summary.fingerprint, base=16, pad=16)),
    ),
    "counterfactuals" => [
        Dict(
            "witness_ordinal" => cf.witness_ordinal,
            "witness_uri" => cf.witness_uri,
            "from_aspect" => String(cf.from_aspect),
            "to_aspect" => String(cf.to_aspect),
            "from_trit" => cf.from_trit,
            "to_trit" => cf.to_trit,
            "trit_delta" => cf.trit_delta,
            "from_color_hex" => cf.from_color_hex,
            "to_color_hex" => cf.to_color_hex,
            "counterfactual_color_hex" => cf.counterfactual_color_hex,
            "semantic_cost" => cf.semantic_cost,
            "closure_effect" => String(cf.closure_effect),
            "catcolab_uri" => cf.catcolab_uri,
        )
        for cf in cfw.counterfactuals
    ],
)

open(joinpath(artifact_dir, "gay_test_olog_catcolab_world.json"), "w") do io
    JSON3.pretty(io, world_payload)
    println(io)
end

open(joinpath(artifact_dir, "gay_test_olog_catcolab_declarations.json"), "w") do io
    JSON3.pretty(io, declarations_payload)
    println(io)
end

open(joinpath(artifact_dir, "gay_test_olog_counterfactuals.json"), "w") do io
    JSON3.pretty(io, counterfactual_payload)
    println(io)
end

open(joinpath(artifact_dir, "gay_test_olog_catcolab_world.txt"), "w") do io
    write(io, render_gay_test_olog(w))
end

open(joinpath(artifact_dir, "gay_test_olog_lisp_bridge.sxp"), "w") do io
    write(io, render_gay_test_olog_lisp_bridge(w))
end

open(joinpath(artifact_dir, "gay_test_olog_counterfactuals.sxp"), "w") do io
    write(io, render_gay_test_olog_counterfactual_lisp_bridge(cfw))
end

open(joinpath(artifact_dir, "gay_test_olog_counterfactuals.txt"), "w") do io
    write(io, render_gay_test_olog_counterfactuals(cfw))
end

open(joinpath(artifact_dir, "gay_test_olog_witness_matrix.tsv"), "w") do io
    println(io, "ordinal\taspect\tcolor_hex\ttestset_hint\tcatcolab_uri")
    for witness in w.witnesses
        println(io, join((
            witness.ordinal,
            witness.aspect,
            witness.color_hex,
            witness.testset_hint,
            witness.catcolab_uri,
        ), '\t'))
    end
end

open(joinpath(artifact_dir, "gay_test_olog_counterfactuals.tsv"), "w") do io
    println(io, "witness_ordinal\tfrom_aspect\tto_aspect\ttrit_delta\tclosure_effect\tcounterfactual_color_hex\tsemantic_cost\tcatcolab_uri")
    for cf in cfw.counterfactuals
        println(io, join((
            cf.witness_ordinal,
            cf.from_aspect,
            cf.to_aspect,
            cf.trit_delta,
            cf.closure_effect,
            cf.counterfactual_color_hex,
            cf.semantic_cost,
            cf.catcolab_uri,
        ), '\t'))
    end
end

println("Wrote Gay test-olog artifacts to ", artifact_dir)
println("test witnesses: ", w.passing_tests)
println("counterfactuals: ", length(cfw.counterfactuals))

