# Gay.jl test olog coloring
#
# The package test suite is treated as a finite set of witnesses. Each passing
# witness is assigned to exactly one CatColab-style olog aspect, and every
# assignment is colored with Gay.jl's own deterministic color stream.

struct GayTestOlogAspect
    id::Symbol
    object_uri::String
    catcolab_uri::String
    trit::Int
    color_hex::String
    closure_role::String
    evidence::String
end

struct GayTestOlogWitness
    ordinal::Int
    uri::String
    aspect::Symbol
    testset_hint::String
    color_hex::String
    catcolab_uri::String
end

struct GayTestOlogMorphism
    name::Symbol
    dom::String
    cod::String
    color_hex::String
    description::String
end

struct GayTestOlogWorld
    seed::UInt64
    passing_tests::Int
    broken_tests::Int
    total_tests::Int
    catcolab_uri::String
    catcolab_path::String
    catcolab_documents::Int
    catcolab_addresses::Int
    typescript_documents::Int
    rust_documents::Int
    aspects::Vector{GayTestOlogAspect}
    witnesses::Vector{GayTestOlogWitness}
    morphisms::Vector{GayTestOlogMorphism}
    equations::Vector{Pair{String,String}}
    fingerprint::UInt64
end

struct GayTestOlogCounterfactual
    witness_ordinal::Int
    witness_uri::String
    from_aspect::Symbol
    to_aspect::Symbol
    from_trit::Int
    to_trit::Int
    trit_delta::Int
    from_color_hex::String
    to_color_hex::String
    counterfactual_color_hex::String
    semantic_cost::Float64
    closure_effect::Symbol
    catcolab_uri::String
end

struct GayTestOlogCounterfactualWorld
    seed::UInt64
    source_fingerprint::UInt64
    witness_count::Int
    aspect_count::Int
    per_witness_choices::Int
    counterfactuals::Vector{GayTestOlogCounterfactual}
    fingerprint::UInt64
end

Base.length(w::GayTestOlogWorld) = length(w.witnesses)
fingerprint(w::GayTestOlogWorld)::UInt64 = w.fingerprint
Base.length(w::GayTestOlogCounterfactualWorld) = length(w.counterfactuals)
fingerprint(w::GayTestOlogCounterfactualWorld)::UInt64 = w.fingerprint

_gay_test_olog_color(seed::UInt64, index::Integer)::String = begin
    c = color_at(Int(index); seed=seed)
    rgb_hex(c.r, c.g, c.b)
end

_catcolab_scip_root() = joinpath(homedir(), "worlds", ".topos", "catcolab")

function _count_lines(path::AbstractString)::Int
    isfile(path) || return 0
    open(path, "r") do io
        count(_ -> true, eachline(io))
    end
end

function _gay_test_olog_aspects(seed::UInt64, catcolab_uri::AbstractString)
    specs = (
        (:color_space_object, "Color spaces and gamut identity", "Color spaces, gamut checks, RGB/hex identity, and canonical genesis colors"),
        (:trit_tick_object, "Trit-tick time base", "GF(3) trits, flics, ticks, and time-color invariants"),
        (:rng_determinism_object, "Splittable deterministic color", "SplitMix64, SplittableRandom, SPI reproducibility, and seed stability"),
        (:parallel_invariance_object, "Parallel invariance", "OhMyThreads, Pigeons, KernelAbstractions, and workgroup independence"),
        (:palette_interface_object, "Palette interface", "Palette generation, pride flags, perceptual separation, and named color surfaces"),
        (:lisp_semantics_object, "Lisp semantic interface", "S-expression/language surfaces that let colors travel through symbolic forms"),
        (:entropy_source_object, "Entropy source audit", "Color entropy sources, mortality classes, and composite readings"),
        (:propagator_cell_object, "Propagator closure", "Propagator cells, supports, contradictions, and scoped ancestry materialization"),
        (:abductive_world_object, "Abductive world update", "World teleportation, abductive sampling, and hypothesized closure repair"),
        (:fuzz_soundness_object, "Fuzz soundness", "QUIC, fuzz, Jepsen-style faults, and adversarial pressure tests"),
        (:ternary_regression_object, "Ternary regression", "Balanced ternary regression and GF(3) compatibility checks"),
        (:nonriemannian_gate_object, "Non-Riemannian gate", "Derived tolerance gates and intrinsic HSL distance checks"),
        (:exa_loop_object, "Exa loop extension", "Clearing, embeddings, sheaf loops, and external search-adjacent closure"),
        (:aqua_hygiene_object, "Package hygiene", "Aqua package checks, stale dependency checks, and project consistency"),
        (:scip_catcolab_object, "CatColab SCIP olog bridge", "SCIP address-space anchoring for CatColab object/morphism/equation declarations"),
    )

    trits = (-1, 0, 1)
    aspects = GayTestOlogAspect[]
    for (i, spec) in enumerate(specs)
        id, role, evidence = spec
        color_index = stable_seed(("gay-test-olog-aspect", id, i); seed=seed) % UInt64(4096) + UInt64(1)
        push!(aspects, GayTestOlogAspect(
            id,
            string("gay://test-olog/object/", id),
            string(catcolab_uri, "/olog/object/", id),
            trits[mod1(i, length(trits))],
            _gay_test_olog_color(seed, color_index),
            role,
            evidence,
        ))
    end
    aspects
end

function _gay_test_olog_hints()
    (
        "Aqua.jl package hygiene",
        "Color Spaces",
        "Trit-Tick Time Base",
        "Random Color Generation",
        "Splittable Determinism",
        "Strong Parallelism Invariance",
        "Palette Generation",
        "Pride Flags",
        "Gamut Operations",
        "Comrade Sky Models",
        "Lisp Interface",
        "Parallel Color Generation",
        "KernelAbstractions SPMD Colors",
        "XOR Fingerprint SPI Verification",
        "SPI Multi-Seed Verification",
        "SPI Workgroup Independence",
        "Color Entropy Sources",
        "Backend Switching",
        "Abductive Tests",
        "QUIC Tests",
        "Fuzz and Jepsen Tests",
        "Propagator Tests",
        "Regression Ternary",
        "Non-Riemannian Gate",
        "Exa Loop Extensions",
    )
end

function _gay_test_olog_witnesses(seed::UInt64, aspects::Vector{GayTestOlogAspect}, passing_tests::Int)
    hints = _gay_test_olog_hints()
    witnesses = GayTestOlogWitness[]
    for i in 1:passing_tests
        aspect = aspects[mod1(i, length(aspects))]
        hint = hints[mod1(i, length(hints))]
        color_index = stable_seed(("gay-test-olog-witness", i, aspect.id, hint); seed=seed) % UInt64(16384) + UInt64(1)
        push!(witnesses, GayTestOlogWitness(
            i,
            @sprintf("gay://test-suite/pass/%03d", i),
            aspect.id,
            hint,
            _gay_test_olog_color(seed, color_index),
            @sprintf("scip://catcolab/olog/gay-test-closure/test-%03d", i),
        ))
    end
    witnesses
end

function _gay_test_olog_morphisms(seed::UInt64, aspects::Vector{GayTestOlogAspect})
    base = GayTestOlogMorphism[
        GayTestOlogMorphism(
            :has_aspect,
            "gay://test-suite/pass",
            "gay://test-olog/aspect",
            _gay_test_olog_color(seed, 1),
            "assigns each passing test witness to exactly one semantic closure aspect",
        ),
        GayTestOlogMorphism(
            :has_color,
            "gay://test-suite/pass",
            "gay://color-chain/seed/1069",
            _gay_test_olog_color(seed, 2),
            "colors every witness using Gay.jl's deterministic color_at stream",
        ),
        GayTestOlogMorphism(
            :declares_object,
            "gay://test-olog/aspect",
            "scip://catcolab/ObDecl",
            _gay_test_olog_color(seed, 3),
            "views each aspect as a CatColab olog object declaration",
        ),
        GayTestOlogMorphism(
            :declares_morphism,
            "gay://test-olog/relation",
            "scip://catcolab/MorDecl",
            _gay_test_olog_color(seed, 4),
            "views test-to-aspect and aspect-to-object edges as CatColab morphism declarations",
        ),
        GayTestOlogMorphism(
            :commutes_as,
            "gay://test-olog/path",
            "scip://catcolab/EqDecl",
            _gay_test_olog_color(seed, 5),
            "records semantic closure constraints as CatColab equation declarations",
        ),
    ]

    for (i, aspect) in enumerate(aspects)
        color_index = stable_seed(("gay-test-olog-morphism", aspect.id, i); seed=seed) % UInt64(4096) + UInt64(1)
        push!(base, GayTestOlogMorphism(
            Symbol("closes_", aspect.id),
            "gay://test-suite/pass",
            aspect.object_uri,
            _gay_test_olog_color(seed, color_index),
            string("passing tests that exercise ", aspect.closure_role, " close through this olog object"),
        ))
    end
    base
end

function _gay_test_olog_equations(passing_tests::Integer)
    [
        "test.has_aspect.declares_object" => "test.has_color.seeds_aspect.declares_object",
        "test.has_aspect" => "unique(test.has_aspect)",
        "count(test where status = pass)" => string(Int(passing_tests)),
        "sum(aspect.trit) mod 3" => "0",
        "aspect.declares_object.in_scip" => "scip://catcolab/ObDecl",
        "relation.declares_morphism.in_scip" => "scip://catcolab/MorDecl",
        "path.commutes_as.in_scip" => "scip://catcolab/EqDecl",
    ]
end

function _gay_test_olog_fingerprint(seed::UInt64, aspects, witnesses, morphisms, equations)::UInt64
    fp = stable_seed("gay-test-olog-catcolab"; seed=seed)
    for aspect in aspects
        fp = xor(fp, stable_seed((aspect.id, aspect.object_uri, aspect.trit, aspect.color_hex); seed=seed))
    end
    for witness in witnesses
        fp = xor(fp, stable_seed((witness.ordinal, witness.aspect, witness.color_hex, witness.testset_hint); seed=seed))
    end
    for morphism in morphisms
        fp = xor(fp, stable_seed((morphism.name, morphism.dom, morphism.cod, morphism.color_hex); seed=seed))
    end
    for equation in equations
        fp = xor(fp, stable_seed((first(equation), last(equation)); seed=seed))
    end
    fp
end

function _hex_rgb(hex::AbstractString)
    s = startswith(hex, "#") ? hex[2:end] : String(hex)
    length(s) == 6 || error("Expected #RRGGBB color, got $hex")
    (
        parse(Int, s[1:2], base=16) / 255,
        parse(Int, s[3:4], base=16) / 255,
        parse(Int, s[5:6], base=16) / 255,
    )
end

function _hex_distance(a::AbstractString, b::AbstractString)::Float64
    ar, ag, ab = _hex_rgb(a)
    br, bg, bb = _hex_rgb(b)
    sqrt((ar - br)^2 + (ag - bg)^2 + (ab - bb)^2)
end

function _balanced_trit_delta(from::Int, to::Int)::Int
    raw = mod(to - from, 3)
    raw == 2 ? -1 : raw
end

function _counterfactual_effect(delta::Int)::Symbol
    if delta == 0
        :same_trit_relabel
    elseif delta == 1
        :positive_shift
    elseif delta == -1
        :negative_shift
    else
        :invalid_shift
    end
end

function _gay_test_olog_counterfactual_fingerprint(seed::UInt64, source_fp::UInt64, cfs)::UInt64
    fp = stable_seed(("gay-test-olog-counterfactuals", source_fp); seed=seed)
    for cf in cfs
        fp = xor(fp, stable_seed((
            cf.witness_ordinal,
            cf.from_aspect,
            cf.to_aspect,
            cf.trit_delta,
            cf.counterfactual_color_hex,
        ); seed=seed))
    end
    fp
end

function world_gay_test_olog(;
    seed::Integer=GAY_SEED,
    passing_tests::Integer=204,
    broken_tests::Integer=2,
    catcolab_uri::AbstractString="scip://catcolab",
    catcolab_root::AbstractString=_catcolab_scip_root(),
)
    seed64 = UInt64(seed)
    pass_count = Int(passing_tests)
    broken_count = Int(broken_tests)
    total_count = pass_count + broken_count

    aspects = _gay_test_olog_aspects(seed64, catcolab_uri)
    witnesses = _gay_test_olog_witnesses(seed64, aspects, pass_count)
    morphisms = _gay_test_olog_morphisms(seed64, aspects)
    equations = _gay_test_olog_equations(pass_count)

    ts_docs = _count_lines(joinpath(catcolab_root, "catcolab-scip-documents.txt"))
    rust_docs = _count_lines(joinpath(catcolab_root, "catcolab-rust-scip-addresses.txt"))
    all_addresses = _count_lines(joinpath(catcolab_root, "catcolab-all-scip-addresses.txt"))
    ts_addresses = _count_lines(joinpath(catcolab_root, "catcolab-scip-addresses.txt"))
    cat_docs = all_addresses == 0 ? ts_docs + rust_docs : all_addresses
    cat_addresses = all_addresses == 0 ? ts_addresses + rust_docs : all_addresses

    fp = _gay_test_olog_fingerprint(seed64, aspects, witnesses, morphisms, equations)
    GayTestOlogWorld(
        seed64,
        pass_count,
        broken_count,
        total_count,
        String(catcolab_uri),
        String(catcolab_root),
        cat_docs,
        cat_addresses,
        ts_docs,
        rust_docs,
        aspects,
        witnesses,
        morphisms,
        equations,
        fp,
    )
end

function gay_test_olog_summary(w::GayTestOlogWorld)
    aspect_counts = Dict{Symbol,Int}(aspect.id => 0 for aspect in w.aspects)
    for witness in w.witnesses
        aspect_counts[witness.aspect] = get(aspect_counts, witness.aspect, 0) + 1
    end
    mutually_exclusive = length(w.witnesses) == w.passing_tests &&
        all(haskey(aspect_counts, witness.aspect) for witness in w.witnesses)
    all_passing_colored = length(w.witnesses) == w.passing_tests &&
        all(!isempty(witness.color_hex) for witness in w.witnesses)
    gf3_conserved = mod(sum(aspect.trit for aspect in w.aspects), 3) == 0
    (
        passing_tests=w.passing_tests,
        broken_tests=w.broken_tests,
        total_tests=w.total_tests,
        aspects=length(w.aspects),
        morphisms=length(w.morphisms),
        equations=length(w.equations),
        catcolab_documents=w.catcolab_documents,
        catcolab_addresses=w.catcolab_addresses,
        mutually_exclusive=mutually_exclusive,
        all_passing_colored=all_passing_colored,
        gf3_conserved=gf3_conserved,
        aspect_counts=aspect_counts,
        fingerprint=w.fingerprint,
    )
end

function gay_test_olog_counterfactuals(w::GayTestOlogWorld)
    aspect_by_id = Dict(aspect.id => aspect for aspect in w.aspects)
    cfs = GayTestOlogCounterfactual[]
    for witness in w.witnesses
        from_aspect = aspect_by_id[witness.aspect]
        for to_aspect in w.aspects
            to_aspect.id == witness.aspect && continue
            delta = _balanced_trit_delta(from_aspect.trit, to_aspect.trit)
            color_index = stable_seed((
                "gay-test-olog-counterfactual",
                witness.ordinal,
                from_aspect.id,
                to_aspect.id,
            ); seed=w.seed) % UInt64(32768) + UInt64(1)
            distance = _hex_distance(witness.color_hex, to_aspect.color_hex)
            cost = round(distance + abs(delta) / 3; digits=6)
            push!(cfs, GayTestOlogCounterfactual(
                witness.ordinal,
                witness.uri,
                from_aspect.id,
                to_aspect.id,
                from_aspect.trit,
                to_aspect.trit,
                delta,
                witness.color_hex,
                to_aspect.color_hex,
                _gay_test_olog_color(w.seed, color_index),
                cost,
                _counterfactual_effect(delta),
                @sprintf(
                    "scip://catcolab/olog/counterfactual/test-%03d/%s-to-%s",
                    witness.ordinal,
                    from_aspect.id,
                    to_aspect.id,
                ),
            ))
        end
    end
    cfs
end

gay_test_olog_counterfactuals() = gay_test_olog_counterfactuals(world_gay_test_olog())

function world_gay_test_olog_counterfactuals(w::GayTestOlogWorld=world_gay_test_olog())
    cfs = gay_test_olog_counterfactuals(w)
    fp = _gay_test_olog_counterfactual_fingerprint(w.seed, w.fingerprint, cfs)
    GayTestOlogCounterfactualWorld(
        w.seed,
        w.fingerprint,
        length(w.witnesses),
        length(w.aspects),
        max(length(w.aspects) - 1, 0),
        cfs,
        fp,
    )
end

function gay_test_olog_counterfactual_summary(w::GayTestOlogCounterfactualWorld)
    effect_counts = Dict{Symbol,Int}()
    for cf in w.counterfactuals
        effect_counts[cf.closure_effect] = get(effect_counts, cf.closure_effect, 0) + 1
    end
    costs = [cf.semantic_cost for cf in w.counterfactuals]
    complete = length(w.counterfactuals) == w.witness_count * w.per_witness_choices
    (
        witness_count=w.witness_count,
        aspect_count=w.aspect_count,
        per_witness_choices=w.per_witness_choices,
        counterfactuals=length(w.counterfactuals),
        complete=complete,
        effect_counts=effect_counts,
        min_cost=isempty(costs) ? 0.0 : minimum(costs),
        mean_cost=isempty(costs) ? 0.0 : round(sum(costs) / length(costs); digits=6),
        max_cost=isempty(costs) ? 0.0 : maximum(costs),
        source_fingerprint=w.source_fingerprint,
        fingerprint=w.fingerprint,
    )
end

gay_test_olog_counterfactual_summary() =
    gay_test_olog_counterfactual_summary(world_gay_test_olog_counterfactuals())

function catcolab_olog_declarations(w::GayTestOlogWorld)
    objects = [
        (
            type="ObDecl",
            name=String(aspect.id),
            description=aspect.closure_role,
            color=aspect.color_hex,
            uri=aspect.catcolab_uri,
            evidence=aspect.evidence,
        )
        for aspect in w.aspects
    ]
    morphisms = [
        (
            type="MorDecl",
            name=String(morphism.name),
            dom=morphism.dom,
            cod=morphism.cod,
            description=morphism.description,
            color=morphism.color_hex,
        )
        for morphism in w.morphisms
    ]
    equations = [
        (
            type="EqDecl",
            lhs=first(equation),
            rhs=last(equation),
            description="semantic world closure constraint",
        )
        for equation in w.equations
    ]
    (objects=objects, morphisms=morphisms, equations=equations)
end

function gay_test_olog_aspects(w::GayTestOlogWorld)
    [
        (
            id=aspect.id,
            trit=aspect.trit,
            color_hex=aspect.color_hex,
            closure_role=aspect.closure_role,
            evidence=aspect.evidence,
            object_uri=aspect.object_uri,
            catcolab_uri=aspect.catcolab_uri,
        )
        for aspect in w.aspects
    ]
end

gay_test_olog_aspects() = gay_test_olog_aspects(world_gay_test_olog())
gay_test_olog_aspect_names(w::GayTestOlogWorld) = [aspect.id for aspect in w.aspects]
gay_test_olog_aspect_names() = gay_test_olog_aspect_names(world_gay_test_olog())

function gay_test_olog_aspect(w::GayTestOlogWorld, id::Symbol)
    for aspect in gay_test_olog_aspects(w)
        aspect.id == id && return aspect
    end
    nothing
end

gay_test_olog_aspect(id::Symbol) = gay_test_olog_aspect(world_gay_test_olog(), id)
gay_test_olog_declarations() = catcolab_olog_declarations(world_gay_test_olog())

function _test_olog_sexp_quote(s::AbstractString)
    escaped = replace(String(s), "\\" => "\\\\", "\"" => "\\\"")
    string("\"", escaped, "\"")
end

function render_gay_test_olog_lisp_bridge(w::GayTestOlogWorld=world_gay_test_olog())
    io = IOBuffer()
    println(io, "(gay-test-olog-bridge")
    println(io, "  (:fingerprint ", _test_olog_sexp_quote(string("0x", string(w.fingerprint, base=16, pad=16))), ")")
    println(io, "  (:catcolab-uri ", _test_olog_sexp_quote(w.catcolab_uri), ")")
    println(io, "  (:tests (:pass ", w.passing_tests, ") (:broken ", w.broken_tests, ") (:total ", w.total_tests, "))")
    println(io, "  (:aspects")
    for aspect in w.aspects
        println(io,
            "    (:aspect",
            " (:id ", aspect.id, ")",
            " (:trit ", aspect.trit, ")",
            " (:color ", _test_olog_sexp_quote(aspect.color_hex), ")",
            " (:role ", _test_olog_sexp_quote(aspect.closure_role), ")",
            " (:catcolab-uri ", _test_olog_sexp_quote(aspect.catcolab_uri), ")",
            ")",
        )
    end
    println(io, "  )")
    println(io, "  (:calls")
    println(io, "    (gay-test-olog-aspect-names)")
    println(io, "    (gay-test-olog-aspects)")
    println(io, "    (gay-test-olog-aspect 'color_space_object)")
    println(io, "    (gay-test-olog-declarations)")
    println(io, "    (gay-test-olog-counterfactual-summary)")
    println(io, "    (world-gay-test-olog-counterfactuals)")
    println(io, "  ))")
    String(take!(io))
end

gay_test_olog_lisp_bridge() = render_gay_test_olog_lisp_bridge(world_gay_test_olog())

function render_gay_test_olog_counterfactuals(w::GayTestOlogCounterfactualWorld; limit::Integer=36)
    io = IOBuffer()
    summary = gay_test_olog_counterfactual_summary(w)
    println(io, "Gay.jl Test Olog Counterfactual World")
    println(io, "fingerprint: 0x", string(w.fingerprint, base=16, pad=16))
    println(io, "source: 0x", string(w.source_fingerprint, base=16, pad=16))
    println(io,
        "counterfactuals: ",
        summary.counterfactuals,
        " = ",
        summary.witness_count,
        " witnesses x ",
        summary.per_witness_choices,
        " alternate aspects",
    )
    println(io, "complete: ", summary.complete, " cost(mean): ", summary.mean_cost)
    println(io)
    println(io, "idx  from                         -> to                           delta  color    cost")
    for cf in Iterators.take(w.counterfactuals, Int(limit))
        println(io,
            lpad(string(cf.witness_ordinal), 3), "  ",
            rpad(_test_olog_clip(String(cf.from_aspect), 28), 28), " -> ",
            rpad(_test_olog_clip(String(cf.to_aspect), 28), 28), "  ",
            lpad(string(cf.trit_delta), 5), "  ",
            cf.counterfactual_color_hex, "  ",
            @sprintf("%.4f", cf.semantic_cost),
        )
    end
    if length(w.counterfactuals) > limit
        println(io, "... ", length(w.counterfactuals) - Int(limit), " more counterfactuals")
    end
    String(take!(io))
end

function render_gay_test_olog_counterfactual_lisp_bridge(w::GayTestOlogCounterfactualWorld=world_gay_test_olog_counterfactuals())
    summary = gay_test_olog_counterfactual_summary(w)
    io = IOBuffer()
    println(io, "(gay-test-olog-counterfactual-bridge")
    println(io, "  (:fingerprint ", _test_olog_sexp_quote(string("0x", string(w.fingerprint, base=16, pad=16))), ")")
    println(io, "  (:source-fingerprint ", _test_olog_sexp_quote(string("0x", string(w.source_fingerprint, base=16, pad=16))), ")")
    println(io, "  (:complete ", summary.complete, ")")
    println(io, "  (:counterfactuals ", summary.counterfactuals, ")")
    println(io, "  (:per-witness-choices ", summary.per_witness_choices, ")")
    println(io, "  (:cost (:min ", summary.min_cost, ") (:mean ", summary.mean_cost, ") (:max ", summary.max_cost, "))")
    println(io, "  (:calls")
    println(io, "    (world-gay-test-olog-counterfactuals)")
    println(io, "    (gay-test-olog-counterfactuals)")
    println(io, "    (gay-test-olog-counterfactual-summary)")
    println(io, "  ))")
    String(take!(io))
end

gay_test_olog_counterfactual_lisp_bridge() =
    render_gay_test_olog_counterfactual_lisp_bridge(world_gay_test_olog_counterfactuals())

function _test_olog_clip(s::AbstractString, width::Int)
    length(s) <= width && return String(s)
    width <= 1 && return first(s, width)
    string(first(s, max(width - 3, 0)), "...")
end

function render_gay_test_olog(w::GayTestOlogWorld; witness_limit::Integer=24)
    io = IOBuffer()
    summary = gay_test_olog_summary(w)
    println(io, "Gay.jl Test Olog Color World")
    println(io, "fingerprint: 0x", string(w.fingerprint, base=16, pad=16))
    println(io, "tests: ", w.passing_tests, " pass, ", w.broken_tests, " broken, ", w.total_tests, " total")
    println(io, "catcolab: ", w.catcolab_uri, " documents=", w.catcolab_documents, " addresses=", w.catcolab_addresses)
    println(io, "closure: mutually_exclusive=", summary.mutually_exclusive,
        " all_passing_colored=", summary.all_passing_colored,
        " gf3_conserved=", summary.gf3_conserved)
    println(io)
    println(io, "Aspects")
    println(io, "trit  color    object")
    for aspect in w.aspects
        println(io, lpad(string(aspect.trit), 4), "  ", aspect.color_hex, "  ", aspect.id)
    end
    println(io)
    println(io, "Witnesses")
    println(io, "idx  color    aspect                         testset")
    for witness in Iterators.take(w.witnesses, Int(witness_limit))
        println(io,
            lpad(string(witness.ordinal), 3), "  ",
            witness.color_hex, "  ",
            rpad(_test_olog_clip(String(witness.aspect), 29), 29), "  ",
            _test_olog_clip(witness.testset_hint, 42),
        )
    end
    if length(w.witnesses) > witness_limit
        println(io, "... ", length(w.witnesses) - Int(witness_limit), " more witnesses")
    end
    String(take!(io))
end
