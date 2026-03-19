# Worlds: Auto-generated world_* functions from demo_* functions
# ═══════════════════════════════════════════════════════════════════════════════
#
# This file provides world_* wrappers for all demo_* functions.
# Each world_* returns a WorldResult with metadata for distribution.
#
# Usage:
#   using Gay
#   result = world_hyperbolic_mining(seed=0x1234)
#   result.value  # The actual result
#   result.metadata  # Distribution metadata
#
# ═══════════════════════════════════════════════════════════════════════════════

module Worlds

using ..WorldDistribution: WorldResult, WorldMetadata, GAY_SEED, WORLD_REGISTRY
using ..WorldDistribution: register_world!, spawn_world

export world_hyperbolic_mining, world_mario_choices
export world_dark_forest, world_three_match
export world_enzyme_subobject, world_enzyme_color_learning
export world_gay_jepsen, world_gay_acset, world_bandwidth_tournament
export world_tikkun_olam, world_baez_topos, world_dialectica
export world_quantum_quiver, world_nashprop_worlds
export world_chromatic_walk, world_self_avoiding_walk
export world_gay_blanket, world_breathing_expander
export world_gay_structured_decompositions
export all_worlds, spawn_all, fork_all

# ═══════════════════════════════════════════════════════════════════════════════
# WORLD WRAPPER GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

"""
    make_world(name, demo_fn, mod, file)

Generate a world_* function from a demo_* function.
"""
function make_world(name::Symbol, mod::Symbol, file::String)
    function world_fn(; seed::UInt64=GAY_SEED, kwargs...)
        meta = WorldMetadata(name; source_module=mod, source_file=file, seed=seed)
        start = time_ns()
        
        try
            # Try to call the demo function
            demo_sym = Symbol("demo_", name)
            
            # This will be replaced with actual module access when loaded
            result = (seed=seed, name=name, status=:stub)
            
            elapsed = time_ns() - start
            WorldResult(result, meta; elapsed=elapsed)
        catch e
            WorldResult(meta, e)
        end
    end
    
    world_fn
end

# ═══════════════════════════════════════════════════════════════════════════════
# HYPERBOLIC BULK MINING WORLDS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    world_hyperbolic_mining(; seed) -> WorldResult

Spawn the hyperbolic bulk mining world.
Tractable 3-coloring via rewriting gadgets in hyperbolic space.
"""
function world_hyperbolic_mining(; seed::UInt64=GAY_SEED, kwargs...)
    meta = WorldMetadata(:hyperbolic_mining; 
                        source_module=:HyperbolicBulkMining,
                        source_file="hyperbolic_bulk_mining.jl",
                        seed=seed)
    start = time_ns()
    
    try
        # Import and call
        @eval using ..HyperbolicBulkMining: demo_hyperbolic_mining
        result = Base.invokelatest(demo_hyperbolic_mining; seed=seed, kwargs...)
        elapsed = time_ns() - start
        WorldResult(result, meta; elapsed=elapsed)
    catch e
        WorldResult(meta, e)
    end
end

"""
    world_mario_choices(; seed) -> WorldResult

Spawn the Mario choices world.
Power-ups, warp pipes, coins, and stars for search navigation.
"""
function world_mario_choices(; seed::UInt64=GAY_SEED, kwargs...)
    meta = WorldMetadata(:mario_choices;
                        source_module=:HyperbolicBulkMining,
                        source_file="hyperbolic_bulk_mining.jl",
                        seed=seed)
    start = time_ns()
    
    try
        @eval using ..HyperbolicBulkMining: demo_mario_choices
        result = Base.invokelatest(demo_mario_choices; seed=seed, kwargs...)
        elapsed = time_ns() - start
        WorldResult(result, meta; elapsed=elapsed)
    catch e
        WorldResult(meta, e)
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# DARK FOREST RGB CIRCLES WORLD
# ═══════════════════════════════════════════════════════════════════════════════

"""
    world_dark_forest(; seed) -> WorldResult

Spawn the dark forest RGB circles world.
Learned subobject classifier with hiding/detection/predation dynamics.
"""
function world_dark_forest(; seed::UInt64=GAY_SEED, kwargs...)
    meta = WorldMetadata(:dark_forest;
                        source_module=:DarkForestRGBCircles,
                        source_file="dark_forest_rgb_circles.jl",
                        seed=seed)
    start = time_ns()
    
    try
        @eval using ..DarkForestRGBCircles: demo_dark_forest
        result = Base.invokelatest(demo_dark_forest, seed)
        elapsed = time_ns() - start
        WorldResult(result, meta; elapsed=elapsed)
    catch e
        WorldResult(meta, e)
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# THREE MATCH WORLDS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    world_three_match(; seed) -> WorldResult

Spawn the 3-MATCH verification world.
Chromatic identity triangle: seed → color → fingerprint.
"""
function world_three_match(; seed::UInt64=GAY_SEED, kwargs...)
    meta = WorldMetadata(:three_match;
                        source_module=:ThreeMatch,
                        source_file="three_match.jl",
                        seed=seed)
    start = time_ns()
    
    try
        @eval using ..ThreeMatch: demo_three_match
        result = Base.invokelatest(demo_three_match)
        elapsed = time_ns() - start
        WorldResult(result, meta; elapsed=elapsed)
    catch e
        WorldResult(meta, e)
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# ENZYME WORLDS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    world_enzyme_subobject(; seed) -> WorldResult

Spawn the Enzyme subobject classifier world.
Autodiff for learning χ: Color → Ω₃.
"""
function world_enzyme_subobject(; seed::UInt64=GAY_SEED, kwargs...)
    meta = WorldMetadata(:enzyme_subobject;
                        source_module=:EnzymeSubobjectExt,
                        source_file="enzyme_subobject_ext.jl",
                        seed=seed)
    start = time_ns()
    
    try
        @eval using ..EnzymeSubobjectExt: demo_enzyme_subobject
        result = Base.invokelatest(demo_enzyme_subobject)
        elapsed = time_ns() - start
        WorldResult(result, meta; elapsed=elapsed)
    catch e
        WorldResult(meta, e)
    end
end

"""
    world_enzyme_color_learning(; seed) -> WorldResult

Spawn the Enzyme color learning world.
Gradient at every level of the color transformation chain.
"""
function world_enzyme_color_learning(; seed::UInt64=GAY_SEED, kwargs...)
    meta = WorldMetadata(:enzyme_color_learning;
                        source_module=:EnzymeColorLearning,
                        source_file="enzyme_color_learning.jl",
                        seed=seed)
    start = time_ns()
    
    try
        # Note: function name from file
        result = (seed=seed, status=:stub, module=:EnzymeColorLearning)
        elapsed = time_ns() - start
        WorldResult(result, meta; elapsed=elapsed)
    catch e
        WorldResult(meta, e)
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# GAME THEORY WORLDS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    world_gay_jepsen(; seed) -> WorldResult

Spawn the Gay Jepsen distributed consistency world.
Linearizability testing for color consensus.
"""
function world_gay_jepsen(; seed::UInt64=GAY_SEED, kwargs...)
    meta = WorldMetadata(:gay_jepsen;
                        source_module=:GayJepsen,
                        source_file="gay_jepsen.jl",
                        seed=seed)
    start = time_ns()
    
    try
        @eval using ..GayJepsen: demo_gay_jepsen
        result = Base.invokelatest(demo_gay_jepsen)
        elapsed = time_ns() - start
        WorldResult(result, meta; elapsed=elapsed)
    catch e
        WorldResult(meta, e)
    end
end

"""
    world_bandwidth_tournament(; seed) -> WorldResult

Spawn the bandwidth tournament world.
Color mining competition across narrators.
"""
function world_bandwidth_tournament(; seed::UInt64=GAY_SEED, kwargs...)
    meta = WorldMetadata(:bandwidth_tournament;
                        source_module=:BandwidthTournament,
                        source_file="bandwidth_tournament.jl",
                        seed=seed)
    start = time_ns()
    
    try
        @eval using ..BandwidthTournament: demo_bandwidth_tournament
        result = Base.invokelatest(demo_bandwidth_tournament)
        elapsed = time_ns() - start
        WorldResult(result, meta; elapsed=elapsed)
    catch e
        WorldResult(meta, e)
    end
end

"""
    world_nashprop_worlds(; seed) -> WorldResult

Spawn the Nash propagation worlds.
Equilibrium finding via propagator networks.
"""
function world_nashprop_worlds(; seed::UInt64=GAY_SEED, kwargs...)
    meta = WorldMetadata(:nashprop_worlds;
                        source_module=:NashpropWorlds,
                        source_file="nashprop_worlds.jl",
                        seed=seed)
    start = time_ns()
    
    try
        @eval using ..NashpropWorlds: demo_nashprop_worlds
        result = Base.invokelatest(demo_nashprop_worlds)
        elapsed = time_ns() - start
        WorldResult(result, meta; elapsed=elapsed)
    catch e
        WorldResult(meta, e)
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY THEORY / TOPOS WORLDS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    world_baez_topos(; seed) -> WorldResult

Spawn the Baez topos world.
Higher category structure for color spaces.
"""
function world_baez_topos(; seed::UInt64=GAY_SEED, kwargs...)
    meta = WorldMetadata(:baez_topos;
                        source_module=:BaezTopos,
                        source_file="baez_topos.jl",
                        seed=seed)
    start = time_ns()
    
    try
        @eval using ..BaezTopos: demo_baez_topos
        result = Base.invokelatest(demo_baez_topos)
        elapsed = time_ns() - start
        WorldResult(result, meta; elapsed=elapsed)
    catch e
        WorldResult(meta, e)
    end
end

"""
    world_dialectica(; seed) -> WorldResult

Spawn the Dialectica interpretation world.
Gödel's functional interpretation for color proofs.
"""
function world_dialectica(; seed::UInt64=GAY_SEED, kwargs...)
    meta = WorldMetadata(:dialectica;
                        source_module=:Dialectica,
                        source_file="dialectica.jl",
                        seed=seed)
    start = time_ns()
    
    try
        @eval using ..Dialectica: demo_dialectica
        result = Base.invokelatest(demo_dialectica)
        elapsed = time_ns() - start
        WorldResult(result, meta; elapsed=elapsed)
    catch e
        WorldResult(meta, e)
    end
end

"""
    world_tikkun_olam(; seed) -> WorldResult

Spawn the Tikkun Olam (repair the world) world.
Healing broken color symmetries.
"""
function world_tikkun_olam(; seed::UInt64=GAY_SEED, kwargs...)
    meta = WorldMetadata(:tikkun_olam;
                        source_module=:TikkunOlam,
                        source_file="tikkun_olam.jl",
                        seed=seed)
    start = time_ns()
    
    try
        @eval using ..TikkunOlam: demo_tikkun_olam
        result = Base.invokelatest(demo_tikkun_olam)
        elapsed = time_ns() - start
        WorldResult(result, meta; elapsed=elapsed)
    catch e
        WorldResult(meta, e)
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# WALK / PATH WORLDS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    world_chromatic_walk(; seed) -> WorldResult

Spawn the chromatic random walk world.
Self-avoiding paths through color space.
"""
function world_chromatic_walk(; seed::UInt64=GAY_SEED, kwargs...)
    meta = WorldMetadata(:chromatic_walk;
                        source_module=:ChromaticWalk,
                        source_file="chromatic_walk.jl",
                        seed=seed)
    start = time_ns()
    
    try
        @eval using ..ChromaticWalk: demo_chromatic_walk
        result = Base.invokelatest(demo_chromatic_walk)
        elapsed = time_ns() - start
        WorldResult(result, meta; elapsed=elapsed)
    catch e
        WorldResult(meta, e)
    end
end

"""
    world_self_avoiding_walk(; seed, steps) -> WorldResult

Spawn the self-avoiding color walk world.
O(1) random access to walk positions.
"""
function world_self_avoiding_walk(; seed::UInt64=GAY_SEED, steps::Int=10, kwargs...)
    meta = WorldMetadata(:self_avoiding_walk;
                        source_module=:SelfAvoidingColorWalk,
                        source_file="self_avoiding_color_walk.jl",
                        seed=seed)
    start = time_ns()
    
    try
        @eval using ..SelfAvoidingColorWalk: demo_self_avoiding_walk
        result = Base.invokelatest(demo_self_avoiding_walk, seed; steps=steps)
        elapsed = time_ns() - start
        WorldResult(result, meta; elapsed=elapsed)
    catch e
        WorldResult(meta, e)
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# ACSET / DATA STRUCTURE WORLDS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    world_gay_acset(; seed) -> WorldResult

Spawn the Gay ACSet world.
Attributed C-sets for color graphs.
"""
function world_gay_acset(; seed::UInt64=GAY_SEED, kwargs...)
    meta = WorldMetadata(:gay_acset;
                        source_module=:GayACSet,
                        source_file="gay_acset.jl",
                        seed=seed)
    start = time_ns()
    
    try
        @eval using ..GayACSet: demo_gay_acset
        result = Base.invokelatest(demo_gay_acset)
        elapsed = time_ns() - start
        WorldResult(result, meta; elapsed=elapsed)
    catch e
        WorldResult(meta, e)
    end
end

"""
    world_gay_structured_decompositions(; seed) -> WorldResult

Spawn the structured decompositions world.
Sheaves over nested graphs for parallel color computation.
"""
function world_gay_structured_decompositions(; seed::UInt64=GAY_SEED, kwargs...)
    meta = WorldMetadata(:gay_structured_decompositions;
                        source_module=:GayStructuredDecompositions,
                        source_file="gay_structured_decompositions.jl",
                        seed=seed)
    start = time_ns()
    
    try
        @eval using ..GayStructuredDecompositions: demo_gay_structured_decompositions
        result = Base.invokelatest(demo_gay_structured_decompositions)
        elapsed = time_ns() - start
        WorldResult(result, meta; elapsed=elapsed)
    catch e
        WorldResult(meta, e)
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# QUANTUM / PHYSICS WORLDS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    world_quantum_quiver(; seed) -> WorldResult

Spawn the quantum quiver world.
ZX-calculus for color circuits.
"""
function world_quantum_quiver(; seed::UInt64=GAY_SEED, kwargs...)
    meta = WorldMetadata(:quantum_quiver;
                        source_module=:QuantumQuiver,
                        source_file="quantum_quiver.jl",
                        seed=seed)
    start = time_ns()
    
    try
        @eval using ..QuantumQuiver: demo_quantum_quiver
        result = Base.invokelatest(demo_quantum_quiver)
        elapsed = time_ns() - start
        WorldResult(result, meta; elapsed=elapsed)
    catch e
        WorldResult(meta, e)
    end
end

"""
    world_gay_blanket(; seed) -> WorldResult

Spawn the Markov blanket world.
Active inference for color prediction.
"""
function world_gay_blanket(; seed::UInt64=GAY_SEED, kwargs...)
    meta = WorldMetadata(:gay_blanket;
                        source_module=:GayBlanket,
                        source_file="gay_blanket.jl",
                        seed=seed)
    start = time_ns()
    
    try
        @eval using ..GayBlanket: demo_gay_blanket
        result = Base.invokelatest(demo_gay_blanket)
        elapsed = time_ns() - start
        WorldResult(result, meta; elapsed=elapsed)
    catch e
        WorldResult(meta, e)
    end
end

"""
    world_gay_immune_geodesic(; seed) -> WorldResult

Spawn the immune geodesic world.
Self/non-self discrimination for color validation.
"""
function world_gay_immune_geodesic(; seed::UInt64=GAY_SEED, kwargs...)
    meta = WorldMetadata(:gay_immune_geodesic;
                        source_module=:GayImmuneGeodesic,
                        source_file="gay_immune_geodesic.jl",
                        seed=seed)
    start = time_ns()
    
    try
        @eval using ..GayImmuneGeodesic: demo_gay_immune_geodesic
        result = Base.invokelatest(demo_gay_immune_geodesic)
        elapsed = time_ns() - start
        WorldResult(result, meta; elapsed=elapsed)
    catch e
        WorldResult(meta, e)
    end
end

"""
    world_breathing_expander(; seed) -> WorldResult

Spawn the breathing expander world.
Verifiable expansion for color diffusion.
"""
function world_breathing_expander(; seed::UInt64=GAY_SEED, kwargs...)
    meta = WorldMetadata(:breathing_expander;
                        source_module=:BreathingExpanderVerifiable,
                        source_file="breathing_expander_verifiable.jl",
                        seed=seed)
    start = time_ns()
    
    try
        @eval using ..BreathingExpanderVerifiable: demo_breathing_expander
        result = Base.invokelatest(demo_breathing_expander)
        elapsed = time_ns() - start
        WorldResult(result, meta; elapsed=elapsed)
    catch e
        WorldResult(meta, e)
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# BATCH OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    all_worlds() -> Vector{Function}

Get all world_* functions.
"""
function all_worlds()
    [
        world_hyperbolic_mining,
        world_mario_choices,
        world_dark_forest,
        world_three_match,
        world_enzyme_subobject,
        world_enzyme_color_learning,
        world_gay_jepsen,
        world_gay_acset,
        world_bandwidth_tournament,
        world_tikkun_olam,
        world_baez_topos,
        world_dialectica,
        world_quantum_quiver,
        world_nashprop_worlds,
        world_chromatic_walk,
        world_self_avoiding_walk,
        world_gay_blanket,
        world_breathing_expander,
        world_gay_structured_decompositions,
        world_gay_immune_geodesic,
    ]
end

"""
    spawn_all(; seed, parallel) -> Dict{Symbol, WorldResult}

Spawn all worlds.
"""
function spawn_all(; seed::UInt64=GAY_SEED, parallel::Bool=true)
    worlds = all_worlds()
    results = Dict{Symbol, WorldResult}()
    
    if parallel
        tasks = Dict{Symbol, Task}()
        for (i, w) in enumerate(worlds)
            world_seed = seed ⊻ UInt64(i)
            name = Symbol(replace(string(w), "world_" => ""))
            tasks[name] = @async w(; seed=world_seed)
        end
        
        for (name, task) in tasks
            results[name] = fetch(task)
        end
    else
        for (i, w) in enumerate(worlds)
            world_seed = seed ⊻ UInt64(i)
            name = Symbol(replace(string(w), "world_" => ""))
            results[name] = w(; seed=world_seed)
        end
    end
    
    results
end

"""
    fork_all(; n_forks, seed) -> Dict{Symbol, Vector{WorldResult}}

Fork all worlds into parallel branches.
"""
function fork_all(; n_forks::Int=3, seed::UInt64=GAY_SEED)
    worlds = all_worlds()
    results = Dict{Symbol, Vector{WorldResult}}()
    
    for (i, w) in enumerate(worlds)
        name = Symbol(replace(string(w), "world_" => ""))
        forks = WorldResult[]
        
        for fork in 1:n_forks
            fork_seed = seed ⊻ UInt64(i) ⊻ UInt64(fork * 1000)
            push!(forks, w(; seed=fork_seed))
        end
        
        results[name] = forks
    end
    
    results
end

end # module Worlds
