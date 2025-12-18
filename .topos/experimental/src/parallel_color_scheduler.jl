# Parallel Color Scheduler with Tree Decomposition Partitioning
#
# Uses synergy dependency graphs + tree decomposition to partition work,
# then executes partitions in parallel with @spawn (minimal syncpoints).
#
# Key invariant: Strong Parallelism Invariance (SPI)
# Same constraints → same schedule → same colors, regardless of thread count.
#
# Architecture (from parallel_seed_search.jl):
#   1. Pre-allocate result slots per worker (no contention)
#   2. Use @spawn for each partition
#   3. Minimal syncpoints: spawn + final merge only
#   4. Each worker writes to own slot

using Base.Threads: @spawn, nthreads, threadid
using Colors

export ColorConstraint, SynergyGraph, SchedulePartition, ScheduleResult
export build_synergy_graph, partition_by_tree_decomposition
export schedule_parallel!, parallel_color_schedule
export demo_parallel_scheduler
export has_cliquetrees, set_cliquetrees_backend!

# ═══════════════════════════════════════════════════════════════════════════
# CliqueTrees.jl Integration (optional)
# ═══════════════════════════════════════════════════════════════════════════

# Backend state: nothing = auto-detect, true = force CliqueTrees, false = force heuristic
const _CLIQUETREES_ENABLED = Ref{Union{Nothing,Bool}}(nothing)

# Function pointers set by extension when CliqueTrees is loaded
const _cliquetrees_decompose = Ref{Any}(nothing)
const _cliquetrees_treewidth = Ref{Any}(nothing)

"""
    has_cliquetrees() -> Bool

Check if CliqueTrees.jl is available and loaded.
"""
function has_cliquetrees()::Bool
    _cliquetrees_decompose[] !== nothing
end

"""
    set_cliquetrees_backend!(enabled::Union{Bool, Nothing})

Control tree decomposition backend:
- `nothing`: auto-detect (use CliqueTrees if available)
- `true`: force CliqueTrees (error if unavailable)
- `false`: force minimum-degree heuristic
"""
function set_cliquetrees_backend!(enabled::Union{Bool, Nothing})
    if enabled === true && !has_cliquetrees()
        error("CliqueTrees.jl not loaded. Add it with: using CliqueTrees")
    end
    _CLIQUETREES_ENABLED[] = enabled
end

"""
    _use_cliquetrees() -> Bool

Internal: determine if we should use CliqueTrees for this call.
"""
function _use_cliquetrees()::Bool
    pref = _CLIQUETREES_ENABLED[]
    if pref === nothing
        return has_cliquetrees()
    end
    return pref
end

# Called by the GayCliqueTreesExt extension to register the backend
function _register_cliquetrees_backend!(decompose_fn, treewidth_fn)
    _cliquetrees_decompose[] = decompose_fn
    _cliquetrees_treewidth[] = treewidth_fn
end

# ═══════════════════════════════════════════════════════════════════════════
# Types
# ═══════════════════════════════════════════════════════════════════════════

"""
Color generation constraint - specifies target properties for a color slot.
"""
struct ColorConstraint
    index::Int
    target_hue::Union{Float64, Nothing}      # Target hue in [0, 360)
    min_distance::Float64                     # Min distance from neighbors
    depends_on::Vector{Int}                   # Indices this color depends on
end

ColorConstraint(i::Int) = ColorConstraint(i, nothing, 30.0, Int[])

"""
Synergy graph: nodes are color indices, edges are synergy dependencies.
If two colors must be "harmonious" (low ΔH, complementary, etc), they share an edge.
"""
struct SynergyGraph
    n::Int
    adj::Dict{Int, Set{Int}}
    constraints::Vector{ColorConstraint}
end

"""
A partition of the schedule for parallel execution.
Each partition can be executed independently after its dependencies are met.
"""
struct SchedulePartition
    id::Int
    indices::Vector{Int}
    depends_on_partitions::Vector{Int}  # Partition IDs that must complete first
end

"""
Result from a single worker's partition execution.
"""
struct ScheduleResult
    partition_id::Int
    worker_id::Int
    colors::Dict{Int, RGB{Float64}}
    elapsed::Float64
end

# ═══════════════════════════════════════════════════════════════════════════
# Synergy Graph Construction
# ═══════════════════════════════════════════════════════════════════════════

"""
    build_synergy_graph(constraints::Vector{ColorConstraint}) -> SynergyGraph

Build a synergy dependency graph from color constraints.
Edges connect colors that depend on each other or must be harmonious.
"""
function build_synergy_graph(constraints::Vector{ColorConstraint})::SynergyGraph
    n = length(constraints)
    adj = Dict{Int, Set{Int}}(i => Set{Int}() for i in 1:n)
    
    for c in constraints
        for dep in c.depends_on
            if 1 <= dep <= n
                push!(adj[c.index], dep)
                push!(adj[dep], c.index)
            end
        end
    end
    
    SynergyGraph(n, adj, constraints)
end

"""
    build_synergy_graph(n::Int; connectivity::Float64=0.1) -> SynergyGraph

Build a random synergy graph for testing.
connectivity ∈ [0,1] controls edge density.

Uses parallel processing for large graphs to reduce construction time.
"""
function build_synergy_graph(n::Int; seed::UInt64=UInt64(42), connectivity::Float64=0.1)::SynergyGraph
    constraints = ColorConstraint[]
    adj = Dict{Int, Set{Int}}(i => Set{Int}() for i in 1:n)

    # For small graphs, use sequential approach (lower overhead)
    if n < 500 || nthreads() == 1
        return build_synergy_graph_sequential(n; seed=seed, connectivity=connectivity)
    end

    # For large graphs, parallelize vertex processing
    return build_synergy_graph_parallel(n; seed=seed, connectivity=connectivity)
end

"""
    build_synergy_graph_sequential(n, seed, connectivity) -> SynergyGraph

Sequential implementation (baseline).
"""
function build_synergy_graph_sequential(n::Int; seed::UInt64=UInt64(42), connectivity::Float64=0.1)::SynergyGraph
    constraints = ColorConstraint[]
    adj = Dict{Int, Set{Int}}(i => Set{Int}() for i in 1:n)

    # Use splitmix64 for deterministic graph construction
    rng_state = seed

    for i in 1:n
        deps = Int[]
        for j in 1:(i-1)
            rng_state = splitmix64(rng_state)
            if (rng_state & 0xFFFF) / 0xFFFF < connectivity
                push!(deps, j)
                push!(adj[i], j)
                push!(adj[j], i)
            end
        end

        rng_state = splitmix64(rng_state)
        target_hue = nothing
        if (rng_state & 0xFF) < 64
            target_hue = ((rng_state >> 8) % 360) * 1.0
        end

        push!(constraints, ColorConstraint(i, target_hue, 30.0, deps))
    end

    SynergyGraph(n, adj, constraints)
end

"""
    build_synergy_graph_parallel(n, seed, connectivity) -> SynergyGraph

Parallel implementation using thread-local RNG streams.
Pre-allocates result slots to avoid contention.
"""
function build_synergy_graph_parallel(n::Int; seed::UInt64=UInt64(42), connectivity::Float64=0.1)::SynergyGraph
    n_threads = nthreads()
    vertices_per_thread = ceil(Int, n / n_threads)

    # Pre-allocate result slots (one per thread - no contention)
    thread_constraints = [Vector{ColorConstraint}() for _ in 1:n_threads]
    thread_adj_locals = [Dict{Int, Set{Int}}() for _ in 1:n_threads]

    # Phase 1: Compute independent RNG streams per thread
    # Create a master RNG state by advancing seed n times
    rng_master = seed
    for _ in 1:(n_threads + 1)
        rng_master = splitmix64(rng_master)
    end

    # Phase 2: Spawn parallel tasks for each thread's vertex range
    tasks = [
        @spawn begin
            thread_id = t
            start_v = (thread_id - 1) * vertices_per_thread + 1
            end_v = min(thread_id * vertices_per_thread, n)

            # Each thread gets its own independent RNG seed
            thread_rng_state = splitmix64(rng_master ⊻ UInt64(thread_id))

            for i in start_v:end_v
                deps = Int[]

                # Generate edges to all previous vertices (deterministically)
                for j in 1:(i-1)
                    thread_rng_state = splitmix64(thread_rng_state)
                    if (thread_rng_state & 0xFFFF) / 0xFFFF < connectivity
                        push!(deps, j)
                    end
                end

                # Generate color constraint
                thread_rng_state = splitmix64(thread_rng_state)
                target_hue = nothing
                if (thread_rng_state & 0xFF) < 64
                    target_hue = ((thread_rng_state >> 8) % 360) * 1.0
                end

                push!(thread_constraints[thread_id], ColorConstraint(i, target_hue, 30.0, deps))

                # Store adjacency for later merging
                thread_adj_locals[thread_id][i] = Set{Int}()
            end
        end
        for t in 1:n_threads
    ]

    # Wait for all threads to complete
    foreach(wait, tasks)

    # Phase 3: Merge results (sequential, minimal overhead)
    constraints = ColorConstraint[]
    adj = Dict{Int, Set{Int}}(i => Set{Int}() for i in 1:n)

    for thread_id in 1:n_threads
        for c in thread_constraints[thread_id]
            push!(constraints, c)
            for dep in c.depends_on
                if 1 <= dep <= n
                    push!(adj[c.index], dep)
                    push!(adj[dep], c.index)
                end
            end
        end
    end

    SynergyGraph(n, adj, constraints)
end

# SplitMix64 - pure, deterministic PRNG step
@inline function splitmix64(x::UInt64)::UInt64
    x += 0x9e3779b97f4a7c15
    x = (x ⊻ (x >> 30)) * 0xbf58476d1ce4e5b9
    x = (x ⊻ (x >> 27)) * 0x94d049bb133111eb
    x ⊻ (x >> 31)
end

# ═══════════════════════════════════════════════════════════════════════════
# Tree Decomposition Partitioning
# ═══════════════════════════════════════════════════════════════════════════

"""
    compute_tree_decomposition(graph::SynergyGraph) -> (bags, order_or_tree)

Compute tree decomposition using CliqueTrees.jl if available, 
otherwise fall back to minimum-degree heuristic.

Returns (bags::Vector{Set{Int}}, auxiliary_data).
"""
function compute_tree_decomposition(graph::SynergyGraph)
    if _use_cliquetrees()
        return _cliquetrees_tree_decomposition(graph)
    else
        return minimum_degree_ordering(graph)
    end
end

"""
    _cliquetrees_tree_decomposition(graph::SynergyGraph) -> (bags, tree)

Use CliqueTrees.jl for proper tree decomposition.
Called only when CliqueTrees is loaded.
"""
function _cliquetrees_tree_decomposition(graph::SynergyGraph)
    decompose_fn = _cliquetrees_decompose[]
    if decompose_fn === nothing
        error("CliqueTrees.jl not loaded but _use_cliquetrees() returned true")
    end
    
    # Build adjacency list for CliqueTrees (expects Vector{Vector{Int}})
    adj_list = [collect(graph.adj[i]) for i in 1:graph.n]
    
    # Call CliqueTrees.tree_decomposition
    td = decompose_fn(adj_list)
    
    # Extract bags from the tree decomposition
    # CliqueTrees returns a structure with .bags field
    bags = [Set{Int}(bag) for bag in td.bags]
    
    (bags, td)
end

"""
    minimum_degree_ordering(graph::SynergyGraph) -> (bags, elimination_order)

Compute tree decomposition bags using minimum-degree elimination heuristic.
Returns bags (each bag is a set of vertex indices) and elimination order.

This is the fallback when CliqueTrees.jl is not available.

Uses parallel degree computation in each elimination iteration to compute
minimum-degree vertex efficiently.
"""
function minimum_degree_ordering(graph::SynergyGraph)
    n = graph.n
    adj = Dict(k => copy(v) for (k, v) in graph.adj)
    remaining = Set(1:n)
    bags = Vector{Set{Int}}()
    order = Int[]

    while !isempty(remaining)
        # Find minimum degree vertex (parallelizable inner loop)
        # Convert remaining to vector for parallel indexing
        remaining_vec = collect(remaining)
        n_remaining = length(remaining_vec)

        # Compute all degrees in parallel with atomic reduction for min
        min_deg_ref = Ref{Tuple{Int, Int}}((typemax(Int), first(remaining)))
        degrees = Vector{Int}(undef, n_remaining)

        # Phase 1: Compute degrees in parallel (read-only on adj and remaining)
        tasks_degrees = [
            @spawn begin
                idx = i
                v = remaining_vec[idx]
                deg = length(adj[v] ∩ remaining)
                degrees[idx] = deg
            end
            for i in 1:n_remaining
        ]

        foreach(wait, tasks_degrees)

        # Phase 2: Find minimum degree vertex (can be parallel with reduction)
        min_deg, min_idx = findmin(degrees)
        min_v = remaining_vec[min_idx]

        # Create bag: vertex + its neighbors
        neighbors = collect(adj[min_v] ∩ remaining)
        bag = Set([min_v; neighbors])
        push!(bags, bag)
        push!(order, min_v)

        # Add fill edges (make neighbors a clique)
        # This section must be sequential (modifies adj)
        for i in 1:length(neighbors)
            for j in (i+1):length(neighbors)
                push!(adj[neighbors[i]], neighbors[j])
                push!(adj[neighbors[j]], neighbors[i])
            end
        end

        # Eliminate vertex
        delete!(remaining, min_v)
    end

    (bags, order)
end

"""
    partition_by_tree_decomposition(graph::SynergyGraph; n_partitions::Int=nthreads()) -> Vector{SchedulePartition}

Partition the synergy graph using tree decomposition for parallel execution.
Each partition is a set of tree decomposition bags that can be processed together.

Uses CliqueTrees.jl when available for optimal decomposition, otherwise falls back
to minimum-degree elimination heuristic.
"""
function partition_by_tree_decomposition(graph::SynergyGraph; n_partitions::Int=nthreads())::Vector{SchedulePartition}
    bags, _ = compute_tree_decomposition(graph)
    n_bags = length(bags)

    if n_bags == 0
        return [SchedulePartition(1, collect(1:graph.n), Int[])]
    end

    # Group bags into partitions
    # Use reverse elimination order for dependency-respecting partition
    bags_per_partition = max(1, ceil(Int, n_bags / n_partitions))

    # Determine actual number of partitions (may be less than requested)
    actual_n_partitions = ceil(Int, n_bags / bags_per_partition)

    # Phase 1: Pre-compute all partition sets in parallel
    # Pre-allocate slots (no contention: one writer per partition)
    partition_sets = Vector{Set{Int}}(undef, actual_n_partitions)

    tasks_phase1 = [
        @spawn begin
            partition_idx = p
            start_idx = (partition_idx - 1) * bags_per_partition + 1
            end_idx = min(partition_idx * bags_per_partition, n_bags)

            indices = Set{Int}()
            for b in start_idx:end_idx
                union!(indices, bags[b])
            end
            partition_sets[partition_idx] = indices
        end
        for p in 1:actual_n_partitions
    ]

    # Wait for all partition sets to be computed
    foreach(wait, tasks_phase1)

    # Phase 2: Compute dependencies in parallel
    # Pre-allocate dependency slots
    depends_on_all = Vector{Vector{Int}}(undef, actual_n_partitions)

    tasks_phase2 = [
        @spawn begin
            partition_idx = p
            deps = Int[]

            for prev_p in 1:(partition_idx - 1)
                if !isempty(partition_sets[partition_idx] ∩ partition_sets[prev_p])
                    push!(deps, prev_p)
                end
            end
            depends_on_all[partition_idx] = deps
        end
        for p in 1:actual_n_partitions
    ]

    # Wait for all dependencies to be computed
    foreach(wait, tasks_phase2)

    # Phase 3: Build partition objects (sequential, minimal overhead)
    partitions = SchedulePartition[]

    for p in 1:actual_n_partitions
        push!(partitions, SchedulePartition(p, collect(partition_sets[p]), depends_on_all[p]))
    end

    # Ensure all indices are covered (assign any missing to last partition)
    covered = Set{Int}()
    for p in partitions
        union!(covered, p.indices)
    end

    missing = setdiff(Set(1:graph.n), covered)
    if !isempty(missing) && !isempty(partitions)
        last_p = partitions[end]
        partitions[end] = SchedulePartition(
            last_p.id,
            vcat(last_p.indices, collect(missing)),
            last_p.depends_on_partitions
        )
    end

    partitions
end

# ═══════════════════════════════════════════════════════════════════════════
# Parallel Execution
# ═══════════════════════════════════════════════════════════════════════════

"""
    worker_execute_partition!(
        partition::SchedulePartition,
        graph::SynergyGraph,
        seed::UInt64,
        result_slot::Ref{ScheduleResult},
        completed::Vector{Ref{ScheduleResult}}
    )

Execute a single partition. Writes to own result slot only.
NO SYNC during execution (SPI invariant).
"""
function worker_execute_partition!(
    partition::SchedulePartition,
    graph::SynergyGraph,
    seed::UInt64,
    result_slot::Ref{ScheduleResult}
)
    t_start = time()
    colors = Dict{Int, RGB{Float64}}()
    worker_id = threadid()
    
    for idx in partition.indices
        # Generate color at this index using SPI-preserving color_at
        c = color_at_internal(idx, seed)
        colors[idx] = c
    end
    
    elapsed = time() - t_start
    
    # Single write to own slot - only sync point
    result_slot[] = ScheduleResult(partition.id, worker_id, colors, elapsed)
    nothing
end

"""
Internal color generation matching the Gay.jl pattern.
Pure function: same (idx, seed) → same color always.
"""
function color_at_internal(idx::Int, seed::UInt64)::RGB{Float64}
    # Derive color from index and seed using splitmix64
    state = seed ⊻ UInt64(idx)
    state = splitmix64(state)
    
    r = (state & 0xFFFF) / 0xFFFF
    state = splitmix64(state)
    g = (state & 0xFFFF) / 0xFFFF
    state = splitmix64(state)
    b = (state & 0xFFFF) / 0xFFFF
    
    RGB{Float64}(r, g, b)
end

"""
    schedule_parallel!(graph::SynergyGraph; seed::UInt64=UInt64(42)) -> Vector{ScheduleResult}

Execute the full parallel schedule using DAG-based dynamic scheduling.

# Architecture (DAG Execution):
1. Compute partitions and dependencies (sequential)
2. Spawn tasks for all partitions in topological order
3. Each task waits ONLY for its specific dependencies (not full levels)
4. Maximizes parallelism by allowing "jagged" execution fronts

Workers run independently with no inter-thread communication beyond dependency waits.
"""
function schedule_parallel!(graph::SynergyGraph; seed::UInt64=UInt64(42))::Vector{ScheduleResult}
    # Phase 1: Compute partitions (sequential, O(n log n))
    partitions = partition_by_tree_decomposition(graph)
    n_partitions = length(partitions)
    
    if n_partitions == 0
        return ScheduleResult[]
    end
    
    # Pre-allocate result slots (no contention)
    results = [Ref{ScheduleResult}(ScheduleResult(i, 0, Dict{Int,RGB{Float64}}(), 0.0)) 
               for i in 1:n_partitions]
    
    # Task handles for dependencies
    # tasks[i] corresponds to the Task for partition i
    tasks = Vector{Task}(undef, n_partitions)
    
    # Determine spawning order (topological sort) via levels
    # This ensures we spawn dependencies before dependents
    levels = partition_to_levels(partitions)
    spawn_order = vcat(levels...)
    
    # Spawn all tasks
    for p_id in spawn_order
        partition = partitions[p_id]
        
        # Identify dependency tasks
        # We can safely access tasks[dep_id] because dep_id must be in a previous level
        # and thus already spawned due to spawn_order
        dep_tasks = Task[tasks[dep_id] for dep_id in partition.depends_on_partitions]
        
        tasks[p_id] = @spawn begin
            # Wait ONLY for specific dependencies
            foreach(wait, dep_tasks)
            
            # Execute partition work
            worker_execute_partition!(partition, graph, seed, results[p_id])
        end
    end
    
    # Wait for all tasks to complete
    foreach(wait, tasks)
    
    # Collect results (pure read, no sync)
    [r[] for r in results]
end

"""
Group partitions into dependency levels for wavefront execution.
Level 0: partitions with no dependencies
Level 1: partitions depending only on level 0
etc.
"""
function partition_to_levels(partitions::Vector{SchedulePartition})::Vector{Vector{Int}}
    n = length(partitions)
    levels = Vector{Int}[]
    assigned = Set{Int}()
    
    while length(assigned) < n
        current_level = Int[]
        
        for p in partitions
            if p.id ∈ assigned
                continue
            end
            
            # Can add to current level if all dependencies are assigned
            if all(dep ∈ assigned for dep in p.depends_on_partitions)
                push!(current_level, p.id)
            end
        end
        
        if isempty(current_level)
            # Cycle or error - add remaining to break
            for p in partitions
                if p.id ∉ assigned
                    push!(current_level, p.id)
                end
            end
        end
        
        push!(levels, current_level)
        union!(assigned, current_level)
    end
    
    levels
end

"""
    parallel_color_schedule(n::Int; seed::UInt64=UInt64(42), connectivity::Float64=0.1) -> Dict{Int, RGB{Float64}}

High-level API: generate n colors with random synergy constraints, scheduled in parallel.
Returns colors as Dict mapping index → color.
"""
function parallel_color_schedule(n::Int; seed::UInt64=UInt64(42), connectivity::Float64=0.1)::Dict{Int, RGB{Float64}}
    graph = build_synergy_graph(n; seed=seed, connectivity=connectivity)
    results = schedule_parallel!(graph; seed=seed)
    
    # Merge all results
    all_colors = Dict{Int, RGB{Float64}}()
    for r in results
        merge!(all_colors, r.colors)
    end
    all_colors
end

# ═══════════════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════════════

"""
    demo_parallel_scheduler(; n::Int=100, seed::UInt64=UInt64(42))

Demonstrate the parallel color scheduler with tree decomposition partitioning.
"""
function demo_parallel_scheduler(; n::Int=100, seed::UInt64=UInt64(42))
    println("═" ^ 70)
    println("  Parallel Color Scheduler - Tree Decomposition Partitioning")
    println("═" ^ 70)
    println()
    
    println("Building synergy graph for $n colors...")
    graph = build_synergy_graph(n; seed=seed, connectivity=0.15)
    
    n_edges = sum(length(v) for v in values(graph.adj)) ÷ 2
    println("  Graph: $n vertices, $n_edges edges")
    println()
    
    backend = has_cliquetrees() ? "CliqueTrees.jl" : "minimum-degree heuristic"
    println("Computing tree decomposition partitions (backend: $backend)...")
    partitions = partition_by_tree_decomposition(graph)
    levels = partition_to_levels(partitions)
    
    println("  $(length(partitions)) partitions in $(length(levels)) dependency levels:")
    for (i, level) in enumerate(levels)
        sizes = [length(partitions[p].indices) for p in level]
        println("    Level $i: $(length(level)) partitions, sizes = $sizes")
    end
    println()
    
    println("Executing parallel schedule ($(nthreads()) threads)...")
    t = @elapsed results = schedule_parallel!(graph; seed=seed)
    
    println("  Completed in $(round(t * 1000, digits=2)) ms")
    println()
    
    println("Worker execution summary:")
    for r in results
        println("  Partition $(r.partition_id): worker $(r.worker_id), $(length(r.colors)) colors, $(round(r.elapsed * 1000, digits=2)) ms")
    end
    println()
    
    # Verify SPI: run again and compare
    println("Verifying Strong Parallelism Invariance...")
    results2 = schedule_parallel!(graph; seed=seed)
    
    colors1 = Dict{Int, RGB{Float64}}()
    colors2 = Dict{Int, RGB{Float64}}()
    for r in results
        merge!(colors1, r.colors)
    end
    for r in results2
        merge!(colors2, r.colors)
    end
    
    identical = colors1 == colors2
    println("  Run 1 == Run 2: $identical $(identical ? "✓" : "✗")")
    println()
    
    println("═" ^ 70)
    
    results
end

# ═══════════════════════════════════════════════════════════════════════════
# Verification Entry Point
# ═══════════════════════════════════════════════════════════════════════════

if abspath(PROGRAM_FILE) == @__FILE__
    demo_parallel_scheduler()
end
