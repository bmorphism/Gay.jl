# World Distribution: Transform Demos into Distributable Worlds
# ═══════════════════════════════════════════════════════════════════════════════
#
# Every `demo_*` becomes a `world_*` that can be:
# - Spawned independently
# - Distributed across nodes
# - Composed with other worlds
# - Checkpointed and resumed
# - Forked into parallel branches
#
# DEMO → WORLD TRANSFORMATION:
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  demo_X() → world_X() with:                                                 │
# │    • Seed parameter for determinism                                         │
# │    • Return type: WorldResult{T} with metadata                              │
# │    • Registration in WORLD_REGISTRY                                         │
# │    • Async spawn capability                                                 │
# │    • Checkpoint/resume support                                              │
# │    • Fork/join for parallelism                                              │
# └─────────────────────────────────────────────────────────────────────────────┘
#
# ═══════════════════════════════════════════════════════════════════════════════

module WorldDistribution

export WorldResult, WorldMetadata, WorldRegistry
export register_world!, spawn_world, spawn_all_worlds
export world_from_demo, distribute_worlds!
export checkpoint_world, resume_world, fork_world
export WORLD_REGISTRY, list_worlds, find_world
export @world, @distribute

const GAY_SEED = UInt64(0x6761795f636f6c6f)

# ═══════════════════════════════════════════════════════════════════════════════
# WORLD RESULT: What a world returns
# ═══════════════════════════════════════════════════════════════════════════════

"""
    WorldMetadata

Metadata for a distributed world.
"""
struct WorldMetadata
    name::Symbol
    source_module::Symbol
    source_file::String
    seed::UInt64
    created_at::Float64
    parent_world::Union{Nothing, Symbol}
    children::Vector{Symbol}
    checkpoint_id::Union{Nothing, UInt64}
    
    # Distribution info
    node_id::Int
    is_distributed::Bool
    can_fork::Bool
end

function WorldMetadata(name::Symbol; 
                       source_module::Symbol=:Unknown,
                       source_file::String="",
                       seed::UInt64=GAY_SEED,
                       parent::Union{Nothing,Symbol}=nothing)
    WorldMetadata(name, source_module, source_file, seed, time(),
                  parent, Symbol[], nothing, 0, false, true)
end

"""
    WorldResult{T}

Result of running a world, with metadata for distribution.
"""
struct WorldResult{T}
    value::T
    metadata::WorldMetadata
    success::Bool
    error::Union{Nothing, Exception}
    elapsed_ns::UInt64
    
    # For resumption
    final_seed::UInt64
    checkpoint_data::Dict{Symbol, Any}
end

function WorldResult(value::T, meta::WorldMetadata; 
                     elapsed::UInt64=UInt64(0)) where T
    WorldResult{T}(value, meta, true, nothing, elapsed, 
                   meta.seed, Dict{Symbol, Any}())
end

function WorldResult(meta::WorldMetadata, err::Exception)
    WorldResult{Nothing}(nothing, meta, false, err, UInt64(0), 
                         meta.seed, Dict{Symbol, Any}())
end

# ═══════════════════════════════════════════════════════════════════════════════
# WORLD REGISTRY: Central catalog of all worlds
# ═══════════════════════════════════════════════════════════════════════════════

"""
    WorldEntry

Entry in the world registry.
"""
mutable struct WorldEntry
    name::Symbol
    demo_name::Symbol           # Original demo_* function name
    world_fn::Function          # The world_* function
    module_name::Symbol
    file_path::String
    description::String
    
    # Execution stats
    run_count::Int
    total_time_ns::UInt64
    last_run::Float64
    last_result::Union{Nothing, WorldResult}
    
    # Distribution
    distributed_to::Vector{Int}  # Node IDs
    forks::Vector{Symbol}        # Forked world names
end

function WorldEntry(name::Symbol, demo_name::Symbol, fn::Function;
                    mod::Symbol=:Unknown, file::String="", desc::String="")
    WorldEntry(name, demo_name, fn, mod, file, desc,
               0, UInt64(0), 0.0, nothing, Int[], Symbol[])
end

"""
    WorldRegistry

Global registry of all distributable worlds.
"""
mutable struct WorldRegistry
    worlds::Dict{Symbol, WorldEntry}
    by_module::Dict{Symbol, Vector{Symbol}}
    by_file::Dict{String, Vector{Symbol}}
    
    # Distribution state
    nodes::Vector{Int}
    distributed_worlds::Dict{Int, Vector{Symbol}}
    
    # Stats
    total_runs::Int
    total_time_ns::UInt64
end

WorldRegistry() = WorldRegistry(
    Dict{Symbol, WorldEntry}(),
    Dict{Symbol, Vector{Symbol}}(),
    Dict{String, Vector{Symbol}}(),
    Int[0],  # Node 0 = local
    Dict{Int, Vector{Symbol}}(0 => Symbol[]),
    0, UInt64(0)
)

const WORLD_REGISTRY = WorldRegistry()

# ═══════════════════════════════════════════════════════════════════════════════
# DEMO → WORLD TRANSFORMATION
# ═══════════════════════════════════════════════════════════════════════════════

"""
    world_from_demo(demo_fn, name; kwargs...) -> Function

Transform a demo function into a distributable world function.
"""
function world_from_demo(demo_fn::Function, name::Symbol;
                         module_name::Symbol=:Unknown,
                         file_path::String="",
                         description::String="")
    function world_fn(; seed::UInt64=GAY_SEED, kwargs...)
        meta = WorldMetadata(name; 
                            source_module=module_name,
                            source_file=file_path,
                            seed=seed)
        
        start_time = time_ns()
        try
            # Run the demo with seed if it accepts one
            result = try
                demo_fn(; seed=seed, kwargs...)
            catch e
                if e isa MethodError
                    # Demo doesn't take seed, run without
                    demo_fn(; kwargs...)
                else
                    rethrow(e)
                end
            end
            
            elapsed = time_ns() - start_time
            WorldResult(result, meta; elapsed=elapsed)
        catch e
            WorldResult(meta, e)
        end
    end
    
    world_fn
end

"""
    register_world!(name, demo_fn; kwargs...)

Register a demo as a distributable world.
"""
function register_world!(name::Symbol, demo_fn::Function;
                         demo_name::Union{Nothing,Symbol}=nothing,
                         module_name::Symbol=:Unknown,
                         file_path::String="",
                         description::String="")
    actual_demo_name = something(demo_name, Symbol("demo_", name))
    
    world_fn = world_from_demo(demo_fn, name;
                               module_name=module_name,
                               file_path=file_path,
                               description=description)
    
    entry = WorldEntry(name, actual_demo_name, world_fn;
                       mod=module_name, file=file_path, desc=description)
    
    WORLD_REGISTRY.worlds[name] = entry
    
    # Index by module
    if !haskey(WORLD_REGISTRY.by_module, module_name)
        WORLD_REGISTRY.by_module[module_name] = Symbol[]
    end
    push!(WORLD_REGISTRY.by_module[module_name], name)
    
    # Index by file
    if !isempty(file_path)
        if !haskey(WORLD_REGISTRY.by_file, file_path)
            WORLD_REGISTRY.by_file[file_path] = Symbol[]
        end
        push!(WORLD_REGISTRY.by_file[file_path], name)
    end
    
    entry
end

# ═══════════════════════════════════════════════════════════════════════════════
# WORLD SPAWNING
# ═══════════════════════════════════════════════════════════════════════════════

"""
    spawn_world(name; seed, async) -> WorldResult or Task

Spawn a world by name.
"""
function spawn_world(name::Symbol; seed::UInt64=GAY_SEED, async::Bool=false)
    if !haskey(WORLD_REGISTRY.worlds, name)
        error("World :$name not found in registry")
    end
    
    entry = WORLD_REGISTRY.worlds[name]
    
    runner = () -> begin
        result = entry.world_fn(; seed=seed)
        
        # Update stats
        entry.run_count += 1
        entry.total_time_ns += result.elapsed_ns
        entry.last_run = time()
        entry.last_result = result
        
        WORLD_REGISTRY.total_runs += 1
        WORLD_REGISTRY.total_time_ns += result.elapsed_ns
        
        result
    end
    
    if async
        @async runner()
    else
        runner()
    end
end

"""
    spawn_all_worlds(; pattern, async, seed) -> Dict{Symbol, WorldResult}

Spawn all worlds matching a pattern.
"""
function spawn_all_worlds(; pattern::Union{Nothing,Regex}=nothing,
                           async::Bool=true,
                           seed::UInt64=GAY_SEED)
    names = collect(keys(WORLD_REGISTRY.worlds))
    
    if pattern !== nothing
        names = filter(n -> occursin(pattern, string(n)), names)
    end
    
    if async
        tasks = Dict(name => spawn_world(name; seed=seed ⊻ UInt64(hash(name)), async=true)
                     for name in names)
        Dict(name => fetch(task) for (name, task) in tasks)
    else
        Dict(name => spawn_world(name; seed=seed ⊻ UInt64(hash(name)), async=false)
             for name in names)
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════

"""
    distribute_worlds!(nodes; pattern, strategy)

Distribute worlds across nodes.
"""
function distribute_worlds!(nodes::Vector{Int};
                            pattern::Union{Nothing,Regex}=nothing,
                            strategy::Symbol=:round_robin)
    WORLD_REGISTRY.nodes = nodes
    
    for node in nodes
        if !haskey(WORLD_REGISTRY.distributed_worlds, node)
            WORLD_REGISTRY.distributed_worlds[node] = Symbol[]
        end
    end
    
    names = collect(keys(WORLD_REGISTRY.worlds))
    if pattern !== nothing
        names = filter(n -> occursin(pattern, string(n)), names)
    end
    
    if strategy == :round_robin
        for (i, name) in enumerate(names)
            node = nodes[mod1(i, length(nodes))]
            push!(WORLD_REGISTRY.distributed_worlds[node], name)
            push!(WORLD_REGISTRY.worlds[name].distributed_to, node)
        end
    elseif strategy == :random
        for name in names
            node = nodes[rand(1:length(nodes))]
            push!(WORLD_REGISTRY.distributed_worlds[node], name)
            push!(WORLD_REGISTRY.worlds[name].distributed_to, node)
        end
    end
    
    WORLD_REGISTRY.distributed_worlds
end

# ═══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT / RESUME / FORK
# ═══════════════════════════════════════════════════════════════════════════════

"""
    checkpoint_world(result) -> UInt64

Save a checkpoint of a world result. Returns checkpoint ID.
"""
function checkpoint_world(result::WorldResult)
    checkpoint_id = hash((result.metadata.name, result.final_seed, time()))
    
    # In a real implementation, this would serialize to disk/network
    # For now, store in the registry
    if haskey(WORLD_REGISTRY.worlds, result.metadata.name)
        entry = WORLD_REGISTRY.worlds[result.metadata.name]
        entry.last_result = WorldResult(
            result.value,
            WorldMetadata(
                result.metadata.name,
                result.metadata.source_module,
                result.metadata.source_file,
                result.metadata.seed,
                result.metadata.created_at,
                result.metadata.parent_world,
                result.metadata.children,
                checkpoint_id,
                result.metadata.node_id,
                result.metadata.is_distributed,
                result.metadata.can_fork
            ),
            result.success,
            result.error,
            result.elapsed_ns,
            result.final_seed,
            result.checkpoint_data
        )
    end
    
    checkpoint_id
end

"""
    resume_world(name, checkpoint_id; seed) -> WorldResult

Resume a world from a checkpoint.
"""
function resume_world(name::Symbol, checkpoint_id::UInt64;
                      seed::Union{Nothing,UInt64}=nothing)
    if !haskey(WORLD_REGISTRY.worlds, name)
        error("World :$name not found")
    end
    
    entry = WORLD_REGISTRY.worlds[name]
    
    if entry.last_result === nothing
        error("No checkpoint found for :$name")
    end
    
    last = entry.last_result
    if last.metadata.checkpoint_id != checkpoint_id
        error("Checkpoint ID mismatch")
    end
    
    # Resume with the seed from checkpoint or provided seed
    resume_seed = something(seed, last.final_seed)
    
    spawn_world(name; seed=resume_seed)
end

"""
    fork_world(name; n_forks, seed_offsets) -> Vector{Symbol}

Fork a world into multiple parallel branches.
"""
function fork_world(name::Symbol; n_forks::Int=3,
                    seed_offsets::Vector{UInt64}=UInt64[1, 2, 3])
    if !haskey(WORLD_REGISTRY.worlds, name)
        error("World :$name not found")
    end
    
    entry = WORLD_REGISTRY.worlds[name]
    base_seed = entry.last_result !== nothing ? 
                entry.last_result.final_seed : GAY_SEED
    
    fork_names = Symbol[]
    
    for i in 1:n_forks
        offset = i <= length(seed_offsets) ? seed_offsets[i] : UInt64(i)
        fork_name = Symbol(name, "_fork_", i)
        fork_seed = base_seed ⊻ offset
        
        # Create forked world entry
        fork_entry = WorldEntry(
            fork_name,
            entry.demo_name,
            entry.world_fn;
            mod=entry.module_name,
            file=entry.file_path,
            desc="Fork $i of $(entry.description)"
        )
        
        WORLD_REGISTRY.worlds[fork_name] = fork_entry
        push!(entry.forks, fork_name)
        push!(fork_names, fork_name)
    end
    
    fork_names
end

# ═══════════════════════════════════════════════════════════════════════════════
# QUERY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    list_worlds(; pattern, module_name) -> Vector{Symbol}

List all registered worlds.
"""
function list_worlds(; pattern::Union{Nothing,Regex}=nothing,
                      module_name::Union{Nothing,Symbol}=nothing)
    if module_name !== nothing
        names = get(WORLD_REGISTRY.by_module, module_name, Symbol[])
    else
        names = collect(keys(WORLD_REGISTRY.worlds))
    end
    
    if pattern !== nothing
        names = filter(n -> occursin(pattern, string(n)), names)
    end
    
    sort(names)
end

"""
    find_world(name) -> Union{WorldEntry, Nothing}

Find a world by name.
"""
find_world(name::Symbol) = get(WORLD_REGISTRY.worlds, name, nothing)

# ═══════════════════════════════════════════════════════════════════════════════
# MACROS FOR EASY REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════

"""
    @world name demo_fn [description]

Register a world from a demo function.
"""
macro world(name, demo_fn, description="")
    quote
        register_world!($(QuoteNode(name)), $(esc(demo_fn));
                       description=$(esc(description)))
    end
end

"""
    @distribute pattern nodes

Distribute worlds matching pattern to nodes.
"""
macro distribute(pattern, nodes)
    quote
        distribute_worlds!($(esc(nodes)); pattern=$(esc(pattern)))
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# AUTO-DISCOVERY: Find all demo_* functions and register them
# ═══════════════════════════════════════════════════════════════════════════════

"""
    discover_demos_in_module(mod) -> Vector{Symbol}

Discover all demo_* functions in a module.
"""
function discover_demos_in_module(mod::Module)
    demos = Symbol[]
    
    for name in names(mod; all=true)
        name_str = string(name)
        if startswith(name_str, "demo_")
            fn = getfield(mod, name)
            if fn isa Function
                # Convert demo_xyz to world :xyz
                world_name = Symbol(name_str[6:end])
                push!(demos, world_name)
                
                register_world!(world_name, fn;
                               demo_name=name,
                               module_name=nameof(mod),
                               description="Auto-discovered from $name")
            end
        end
    end
    
    demos
end

"""
    discover_all_demos!(; modules)

Discover and register all demo_* functions from specified modules.
"""
function discover_all_demos!(; modules::Vector{Module}=Module[])
    all_demos = Symbol[]
    
    for mod in modules
        demos = discover_demos_in_module(mod)
        append!(all_demos, demos)
    end
    
    all_demos
end

# ═══════════════════════════════════════════════════════════════════════════════
# KNOWN DEMOS: Pre-register the discovered demos
# ═══════════════════════════════════════════════════════════════════════════════

const KNOWN_DEMO_FILES = [
    # (world_name, file_path, module_name)
    (:hyperbolic_mining, "hyperbolic_bulk_mining.jl", :HyperbolicBulkMining),
    (:mario_choices, "hyperbolic_bulk_mining.jl", :HyperbolicBulkMining),
    (:dark_forest, "dark_forest_rgb_circles.jl", :DarkForestRGBCircles),
    (:three_match, "three_match.jl", :ThreeMatch),
    (:parametrized_lazy, "three_match.jl", :ThreeMatch),
    (:coq_generation, "three_match.jl", :ThreeMatch),
    (:enzyme_subobject, "enzyme_subobject_ext.jl", :EnzymeSubobjectExt),
    (:enzyme_color_learning, "enzyme_color_learning.jl", :EnzymeColorLearning),
    (:gay_jepsen, "gay_jepsen.jl", :GayJepsen),
    (:gay_acset, "gay_acset.jl", :GayACSet),
    (:gay_worldnet, "gay_worldnet.jl", :GayWorldNet),
    (:bandwidth_tournament, "bandwidth_tournament.jl", :BandwidthTournament),
    (:tikkun_olam, "tikkun_olam.jl", :TikkunOlam),
    (:baez_topos, "baez_topos.jl", :BaezTopos),
    (:dialectica, "dialectica.jl", :Dialectica),
    (:gay_blanket, "gay_blanket.jl", :GayBlanket),
    (:quantum_quiver, "quantum_quiver.jl", :QuantumQuiver),
    (:nashprop_worlds, "nashprop_worlds.jl", :NashpropWorlds),
    (:abstract_world, "abstract_world.jl", :AbstractWorld),
    (:unified_parallelism, "unified_gay_parallelism.jl", :UnifiedGayParallelism),
    (:consapevolezza_parallelism, "consapevolezza_parallelism.jl", :ConsapevolezzaParallelism),
    (:breathing_expander, "breathing_expander_verifiable.jl", :BreathingExpanderVerifiable),
    (:chromatic_walk, "chromatic_walk.jl", :ChromaticWalk),
    (:self_avoiding_walk, "self_avoiding_color_walk.jl", :SelfAvoidingColorWalk),
    (:gay_structured_decompositions, "gay_structured_decompositions.jl", :GayStructuredDecompositions),
    (:gay_immune_geodesic, "gay_immune_geodesic.jl", :GayImmuneGeodesic),
    (:gay_phased_array, "gay_phased_array.jl", :GayPhasedArray),
]

"""
    register_known_worlds!()

Register all known world definitions (stubs until modules are loaded).
"""
function register_known_worlds!()
    for (name, file, mod) in KNOWN_DEMO_FILES
        # Create a stub that will be replaced when module is loaded
        stub_fn = (;kwargs...) -> error("Module $mod not loaded. Include $file first.")
        
        entry = WorldEntry(name, Symbol("demo_", name), stub_fn;
                          mod=mod, file=file, 
                          desc="From $mod (load $file to enable)")
        
        WORLD_REGISTRY.worlds[name] = entry
        
        if !haskey(WORLD_REGISTRY.by_module, mod)
            WORLD_REGISTRY.by_module[mod] = Symbol[]
        end
        push!(WORLD_REGISTRY.by_module[mod], name)
    end
    
    length(KNOWN_DEMO_FILES)
end

# ═══════════════════════════════════════════════════════════════════════════════
# DEMO (Meta: a world that spawns worlds)
# ═══════════════════════════════════════════════════════════════════════════════

function world_world_distribution(; seed::UInt64=GAY_SEED)
    println("╔═══════════════════════════════════════════════════════════════════════════╗")
    println("║  WORLD DISTRIBUTION: Transform Demos into Distributable Worlds           ║")
    println("╚═══════════════════════════════════════════════════════════════════════════╝")
    println()
    
    # Register known worlds
    n_registered = register_known_worlds!()
    println("═══ REGISTRATION ═══")
    println("  Registered $n_registered known worlds")
    println()
    
    # List worlds by module
    println("═══ WORLDS BY MODULE ═══")
    for (mod, names) in sort(collect(WORLD_REGISTRY.by_module))
        println("  $mod: $(length(names)) worlds")
        for name in names[1:min(3, length(names))]
            println("    • :$name")
        end
        length(names) > 3 && println("    ... and $(length(names) - 3) more")
    end
    println()
    
    # Show distribution simulation
    println("═══ DISTRIBUTION SIMULATION ═══")
    nodes = [0, 1, 2]  # 3 nodes
    dist = distribute_worlds!(nodes; strategy=:round_robin)
    
    for node in nodes
        worlds = dist[node]
        println("  Node $node: $(length(worlds)) worlds")
    end
    println()
    
    # Show fork example
    println("═══ FORK EXAMPLE ═══")
    if haskey(WORLD_REGISTRY.worlds, :hyperbolic_mining)
        forks = fork_world(:hyperbolic_mining; n_forks=3)
        println("  Forked :hyperbolic_mining into:")
        for f in forks
            println("    • :$f")
        end
    end
    println()
    
    # Stats
    println("═══ REGISTRY STATS ═══")
    println("  Total worlds: $(length(WORLD_REGISTRY.worlds))")
    println("  Modules: $(length(WORLD_REGISTRY.by_module))")
    println("  Nodes: $(length(WORLD_REGISTRY.nodes))")
    println()
    
    WORLD_REGISTRY
end

# Make this module itself a world
function __init__()
    register_world!(:world_distribution, world_world_distribution;
                   module_name=:WorldDistribution,
                   description="Meta-world that manages other worlds")
end

end # module WorldDistribution
