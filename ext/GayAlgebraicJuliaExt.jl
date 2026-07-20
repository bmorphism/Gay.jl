__precompile__(false)

module GayAlgebraicJuliaExt

using Gay
using GATlab
using Catlab
using ACSets
using AlgebraicRewriting

const _LOADED_ALGEBRAICJULIA_PACKAGES = [:GATlab, :Catlab, :ACSets, :AlgebraicRewriting]

@present SchLispGATOperationalArena(FreeSchema) begin
    (Witness, Aspect, Counterfactual, Assignment, SharedArena)::Ob
    (Name, Ordinal, Cost)::AttrType

    assigned_witness::Hom(Assignment, Witness)
    assigned_aspect::Hom(Assignment, Aspect)
    explained_by::Hom(Assignment, Counterfactual)

    counterfactual_witness::Hom(Counterfactual, Witness)
    from_aspect::Hom(Counterfactual, Aspect)
    to_aspect::Hom(Counterfactual, Aspect)
    shared_in::Hom(Counterfactual, SharedArena)
    witness_arena::Hom(Witness, SharedArena)

    aspect_name::Attr(Aspect, Name)
    arena_name::Attr(SharedArena, Name)
    effect_name::Attr(Counterfactual, Name)
    witness_ordinal::Attr(Witness, Ordinal)
    counterfactual_index::Attr(Counterfactual, Ordinal)
    assignment_cost::Attr(Assignment, Cost)
    assignment_color::Attr(Assignment, Name)
end

@acset_type LispGATOperationalArena(SchLispGATOperationalArena){String,Int,Float64}

const _LISP_GAT_REWRITE_CAT = ACSetCategory(VarACSetCat(LispGATOperationalArena()))

function _bridge_term(p, path::Vector{Symbol})
    length(path) == 1 ? GATlab.generator(p, first(path)) :
        Catlab.compose(GATlab.generator.(Ref(p), path)...)
end

function _bridge_presentation(w::Gay.LispGATBridgeWorld)
    p = GATlab.Presentation(Catlab.FreeSchema)
    objects = Dict{Symbol,Any}()
    attrtypes = Dict{Symbol,Any}()

    for ob in w.objects
        if ob.kind == :ob
            objects[ob.name] = GATlab.add_generator!(p, Catlab.Ob(Catlab.FreeSchema.Ob, ob.name))
        elseif ob.kind == :attrtype
            attrtypes[ob.name] = GATlab.add_generator!(p, Catlab.AttrType(Catlab.FreeSchema.AttrType, ob.name))
        else
            error("Unknown Lisp/GAT object kind: $(ob.kind)")
        end
    end

    for mor in w.morphisms
        if mor.kind == :hom
            GATlab.add_generator!(p, Catlab.Hom(mor.name, objects[mor.dom], objects[mor.cod]))
        elseif mor.kind == :attr
            GATlab.add_generator!(p, Catlab.Attr(mor.name, objects[mor.dom], attrtypes[mor.cod]))
        else
            error("Unknown Lisp/GAT morphism kind: $(mor.kind)")
        end
    end

    for eq in w.equations
        GATlab.add_equation!(p, _bridge_term(p, eq.lhs), _bridge_term(p, eq.rhs))
    end

    p
end

function _generator_counts(p)
    Dict(
        :Ob => length(GATlab.generators(p, :Ob)),
        :AttrType => length(GATlab.generators(p, :AttrType)),
        :Hom => length(GATlab.generators(p, :Hom)),
        :Attr => length(GATlab.generators(p, :Attr)),
    )
end

function _candidate_dpo_witness(cand; execute::Bool=false)
    cat = _LISP_GAT_REWRITE_CAT
    from_name = String(cand.from_aspect)
    to_name = String(cand.to_aspect)
    effect_name = String(cand.closure_effect)
    color_name = cand.color_hex
    cost = Float64(cand.semantic_cost)

    interface = @acset LispGATOperationalArena begin
        Witness = 1
        Aspect = 2
        Counterfactual = 1
        SharedArena = 1
        counterfactual_witness = 1
        from_aspect = 1
        to_aspect = 2
        shared_in = 1
        witness_arena = 1
        aspect_name = [from_name, to_name]
        arena_name = ["shared_arena"]
        effect_name = [effect_name]
        witness_ordinal = [cand.witness_ordinal]
        counterfactual_index = [cand.counterfactual_index]
    end

    left = @acset LispGATOperationalArena begin
        Witness = 1
        Aspect = 2
        Counterfactual = 1
        Assignment = 1
        SharedArena = 1
        assigned_witness = 1
        assigned_aspect = 1
        explained_by = 1
        counterfactual_witness = 1
        from_aspect = 1
        to_aspect = 2
        shared_in = 1
        witness_arena = 1
        aspect_name = [from_name, to_name]
        arena_name = ["shared_arena"]
        effect_name = [effect_name]
        witness_ordinal = [cand.witness_ordinal]
        counterfactual_index = [cand.counterfactual_index]
        assignment_cost = [cost]
        assignment_color = [color_name]
    end

    right = @acset LispGATOperationalArena begin
        Witness = 1
        Aspect = 2
        Counterfactual = 1
        Assignment = 1
        SharedArena = 1
        assigned_witness = 1
        assigned_aspect = 2
        explained_by = 1
        counterfactual_witness = 1
        from_aspect = 1
        to_aspect = 2
        shared_in = 1
        witness_arena = 1
        aspect_name = [from_name, to_name]
        arena_name = ["shared_arena"]
        effect_name = [effect_name]
        witness_ordinal = [cand.witness_ordinal]
        counterfactual_index = [cand.counterfactual_index]
        assignment_cost = [cost]
        assignment_color = [color_name]
    end

    left_map = ACSetTransformation(
        interface,
        left;
        Witness=[1],
        Aspect=[1, 2],
        Counterfactual=[1],
        SharedArena=[1],
        cat=cat,
    )
    right_map = ACSetTransformation(
        interface,
        right;
        Witness=[1],
        Aspect=[1, 2],
        Counterfactual=[1],
        SharedArena=[1],
        cat=cat,
    )
    rule = Rule(left_map, right_map; cat=cat)
    match = id[cat](left)
    result = execute ? rewrite_match(rule, match; cat=cat) : nothing

    (
        category=cat,
        interface=interface,
        left=left,
        right=right,
        left_map=left_map,
        right_map=right_map,
        dpo_rule=rule,
        match=match,
        result=result,
        result_is_target=result === nothing ? false : is_isomorphic(result, right),
    )
end

function _dpo_sample_ordinals(total::Integer; dpo_sample_count::Integer=1, dpo_sample_ordinals=nothing)
    n = Int(total)
    n <= 0 && return Set{Int}()

    if dpo_sample_ordinals !== nothing
        ords = sort(unique(Int(x) for x in dpo_sample_ordinals if 1 <= Int(x) <= n))
        return Set(ords)
    end

    count = clamp(Int(dpo_sample_count), 0, n)
    count == 0 && return Set{Int}()
    count == 1 && return Set([1])

    ords = sort(unique(round.(Int, range(1, n; length=count))))
    Set(ords)
end

function _materialized_rewrite_candidates(
    w::Gay.LispGATBridgeWorld,
    p;
    dpo_sample_count::Integer=1,
    dpo_sample_ordinals=nothing,
)
    sample_ordinals = _dpo_sample_ordinals(
        length(w.counterfactuals);
        dpo_sample_count=dpo_sample_count,
        dpo_sample_ordinals=dpo_sample_ordinals,
    )
    map(Gay.lisp_gatlab_rewrite_candidates(w)) do cand
        source_term = _bridge_term(p, cand.source_path)
        target_term = _bridge_term(p, cand.target_path)
        arena_term = _bridge_term(p, cand.arena_path)
        witness_arena_term = _bridge_term(p, cand.witness_arena_path)
        match_term = _bridge_term(p, cand.match_path)
        materialize_rule = cand.ordinal in sample_ordinals
        dpo = materialize_rule ? _candidate_dpo_witness(cand; execute=true) : nothing
        result_executed = dpo !== nothing && dpo.result !== nothing
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
            match_term=match_term,
            source_term=source_term,
            target_term=target_term,
            arena_term=arena_term,
            witness_arena_term=witness_arena_term,
            dpo_rule=materialize_rule ? dpo.dpo_rule : nothing,
            dpo_match=materialize_rule ? dpo.match : nothing,
            dpo_result=result_executed ? dpo.result : nothing,
            dpo_rule_materialized=materialize_rule,
            dpo_result_executed=result_executed,
            dpo_result_is_target=materialize_rule ? dpo.result_is_target : false,
            dpo_rule_type=materialize_rule ? string(typeof(dpo.dpo_rule)) : "DPO-rule-spec",
            dpo_match_type=materialize_rule ? string(typeof(dpo.match)) : "DPO-match-spec",
            dpo_result_type=result_executed ? string(typeof(dpo.result)) : "not-executed",
            acset_type=materialize_rule ? string(typeof(dpo.left)) : "LispGATOperationalArena",
            category_type=materialize_rule ? string(typeof(dpo.category)) : "ACSetCategory(VarACSetCat(LispGATOperationalArena))",
            left_assignment_aspect=1,
            right_assignment_aspect=2,
            result_assignment_aspect=result_executed ? only(dpo.result[:assigned_aspect]) : 0,
            match_term_type=string(typeof(match_term)),
            source_term_type=string(typeof(source_term)),
            target_term_type=string(typeof(target_term)),
            arena_term_type=string(typeof(arena_term)),
            witness_arena_term_type=string(typeof(witness_arena_term)),
            source_term_text=sprint(show, source_term),
            target_term_text=sprint(show, target_term),
            arena_term_text=sprint(show, arena_term),
            witness_arena_term_text=sprint(show, witness_arena_term),
        )
    end
end

function realize_lisp_gatlab_bridge_algebraicjulia(w::Gay.LispGATBridgeWorld)
    Gay.algebraicjulia_realization_plan(
        w;
        extension=:GayAlgebraicJuliaExt,
        backend=:algebraicjulia,
        packages=copy(_LOADED_ALGEBRAICJULIA_PACKAGES),
        acset_hint="GATlab, Catlab, and ACSets are loaded; render_lisp_gatlab_presentation(w) is ready for package-backed schema materialization.",
        rewriting_hint="AlgebraicRewriting is loaded; LispGATCounterfactual rows can be promoted to color-preserving rewrite spans.",
    )
end

function materialize_lisp_gatlab_bridge_algebraicjulia(w::Gay.LispGATBridgeWorld)
    materialize_lisp_gatlab_bridge_algebraicjulia(w; dpo_sample_count=1)
end

function materialize_lisp_gatlab_bridge_algebraicjulia(
    w::Gay.LispGATBridgeWorld;
    dpo_sample_count::Integer=1,
    dpo_sample_ordinals=nothing,
)
    p = _bridge_presentation(w)
    rewrite_candidates = Any[_materialized_rewrite_candidates(
        w,
        p;
        dpo_sample_count=dpo_sample_count,
        dpo_sample_ordinals=dpo_sample_ordinals,
    )...]
    Gay.algebraicjulia_materialization_plan(
        w;
        extension=:GayAlgebraicJuliaExt,
        backend=:algebraicjulia,
        packages=copy(_LOADED_ALGEBRAICJULIA_PACKAGES),
        presentation=p,
        presentation_type=string(typeof(p)),
        generator_counts=_generator_counts(p),
        equation_count=length(GATlab.equations(p)),
        rewrite_candidates=rewrite_candidates,
        rewrite_candidate_count=length(rewrite_candidates),
    )
end

end
