# world_all.jl — world_ wrapper for every demo_ in Gay.jl
# Each world_X calls demo_X with @elapsed timing and structured output.
# Fault-tolerant: any demo crash is caught and reported, never stops the suite.

export world_all, world_run

struct WorldResult
    name::String
    elapsed_ms::Float64
    ok::Bool
    error::String
end

Base.show(io::IO, r::WorldResult) = print(io,
    r.ok ? "  ✓ " : "  ✗ ",
    rpad(r.name, 40),
    lpad(string(round(r.elapsed_ms, digits=2)), 10), " ms",
    r.ok ? "" : "  ERROR: $(r.error)")

"""
    world_run(name::String) -> WorldResult

Run a single demo_ function by name, wrapped with timing and error handling.
"""
function world_run(name::String)
    fn_sym = Symbol("demo_", name)
    t = 0.0
    ok = true
    err = ""
    try
        fn = getfield(Gay, fn_sym)
        t = @elapsed redirect_stdout(devnull) do
            fn()
        end
    catch e
        ok = false
        err = sprint(showerror, e)
        if length(err) > 120
            err = err[1:120] * "..."
        end
    end
    WorldResult(string(fn_sym), t * 1000, ok, err)
end

# All demo_ functions discoverable in the module, in rough dependency order.
# Source-file demos first, then Gay.jl inline demos.
const WORLD_DEMOS = [
    # splittable.jl
    "padic",
    # quic.jl
    "quic_paths",
    # enzyme.jl
    "enzyme_colors",
    # enzyme_dsl.jl
    "enzyme_dsl",
    # fault_tolerant.jl
    "fault_tolerant",
    # binary.jl
    "binary_analysis",
    # swarm_triad.jl
    "swarm_triad",
    # cognitive_superposition.jl
    "cognitive_superposition",
    # triadic_subagents.jl
    "triadic_subagents",
    # radare2.jl
    "radare2_colors",
    # semiosis.jl
    "sse_enzyme", "colors", "interleaver", "derangeable", "abduce",
    # okhsl_learnable.jl
    "learnable_okhsl",
    # multiverse_geometric.jl
    "holographic_game",
    # ink_traced_criticality.jl
    "traced_ink",
    # kernel_datalog.jl
    "kernel_datalog",
    # Gay.jl inline demos (alphabetical)
    "abstract_world",
    "abstract_crdt",
    "abstract_acset",
    "abstract_probe",
    "abstract_free_gadget",
    "ablative_consapevolezza",
    "ablative_naming",
    "amp_threads",
    "ananas", "ananas_integration", "ananas_hierarchy",
    "aperiodic_associativity",
    "bandwidth_tournament",
    "bartons_free",
    "blessed_mining",
    "breathing_expander",
    "carrying_capacity",
    "cfr_speedrun",
    "chaos_vibing",
    "color_graph_topos",
    "color_logic_pullback",
    "colorable_flavorable",
    "concept_tensor",
    "copy_on_interact_3",
    "derangeable_evolution",
    "dialectica",
    "exponential",
    "freedom_growth",
    "4d_coherence",
    "gay_advanced_hmc",
    "gay_async", "gay_lisp_async",
    "gay_blanket",
    "gay_compute_parallelism",
    "gay_data_parallelism",
    "gay_duckdb_parallelism",
    "gay_immune_geodesic",
    "gay_jepsen",
    "gay_metalearning",
    "gay_open_game",
    "gay_relationality",
    "gay_revolution",
    "gay_ruler",
    "gay_seed_bundle",
    "gay_weights_biases",
    "gay_world_parallelism",
    "gay_69_construction",
    "gaymc_graph",
    "gaymc_pathfinding",
    "gayzip_worlds",
    "geo_acset_guarantees",
    "geo_gay_morphism",
    "gayext_ranking",
    "genetic_search",
    "higher_structure",
    "homotopy_hypothesis",
    "hyperdoctrine",
    "involutive_curiosities",
    "juno_gay",
    "kripke_bridge",
    "marginal_convergence",
    "maximal_parallelism",
    "mc_teleport",
    "metatheory_fuzz",
    "minimal",
    "motherlake_speedup",
    "next_color_bandwidth",
    "no_irreconcilable_self",
    "observer_bridge",
    "optimal_fixed_points",
    "org_delegation",
    "org_walker",
    "parallel_gh",
    "parallel_search",
    "parametrized_lazy",
    "pocp",
    "probe_continuation",
    "profinite_duck",
    "push_pull_sequence",
    "quic_interleave",
    "quic_pathfinding",
    "reafferent_dao_topos",
    "reafferent_perception",
    "reflective_equilibria",
    "renyi_entropy",
    "report",
    "retrocausal_cfr",
    "roms_color_tiling",
    "runtime_placement",
    "runtime_triads",
    "seed_semantics",
    "spectral_bridge",
    "spivak_wiring",
    "surprisal_satisficing",
    "ternary_colorspace",
    "ternary_split",
    "thread_findings",
    "three_ducks",
    "three_match",
    "tikkun_olam",
    "timelock_prediction",
    "traced_tensor",
    "trit_bandwidth",
    "trit_walk",
    "triple_split_sentinel",
    "2machine_minimax",
    "two_narrative",
    "umwelt",
    "unified_topos",
    "wikipedia_ranking",
    "worktree_reconciliation",
    "worlds_topos",
    "zero_message_mining",
    "69_interactions",
    "69_monads",
    "coq_generation",
    "ergodic_bridge",
]

"""
    world_all(; verbose=true, skip_errors=true) -> Vector{WorldResult}

Run every demo_ as a world_ benchmark. Prints a table of results.
"""
function world_all(; verbose::Bool=true, skip_errors::Bool=true)
    results = WorldResult[]

    verbose && println("╔═══════════════════════════════════════════════════════════════════╗")
    verbose && println("║              world_all() — Every demo_ benchmarked                ║")
    verbose && println("╠═══════════════════════════════════════════════════════════════════╣")

    for name in WORLD_DEMOS
        fn_sym = Symbol("demo_", name)

        if !isdefined(Gay, fn_sym)
            push!(results, WorldResult(string(fn_sym), 0.0, false, "not defined"))
            verbose && println(results[end])
            continue
        end

        fn = getfield(Gay, fn_sym)
        t = 0.0
        ok = true
        err = ""
        try
            t = @elapsed redirect_stdout(devnull) do
                fn()
            end
        catch e
            ok = false
            err = sprint(showerror, e)
            if length(err) > 120
                err = err[1:120] * "..."
            end
        end

        r = WorldResult(string(fn_sym), t * 1000, ok, err)
        push!(results, r)
        verbose && println(r)
    end

    n_ok = count(r -> r.ok, results)
    n_total = length(results)
    total_ms = sum(r -> r.elapsed_ms, results)

    verbose && println("╠═══════════════════════════════════════════════════════════════════╣")
    verbose && println("║  Total: $n_ok/$n_total passed | $(round(total_ms, digits=1)) ms total")
    verbose && println("╚═══════════════════════════════════════════════════════════════════╝")

    results
end
