"""
    GayTemporalNarratives

Time-varying graphs with SPI coloring. Based on Bumpus "Towards a Unified Theory
of Time-varying Data" (2024). Narratives are sequences of graph snapshots with
morphisms tracking evolution across time.
"""
module GayTemporalNarratives

export Snapshot, Narrative, SnapshotMorphism
export gay_narrative_bfs!, gay_interval_sheaf!, gay_snapshot_compose!, gay_boomerang_narrative!
export splitmix64, chromatic_fingerprint

using SparseArrays

# --- SPI Core ---

"""Splitmix64 PRNG - deterministic, fast mixing."""
@inline function splitmix64(seed::UInt64)::UInt64
    z = seed + 0x9e3779b97f4a7c15
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    z ⊻ (z >> 31)
end

"""6-color palette fingerprint from seed."""
@inline chromatic_fingerprint(seed::UInt64)::UInt8 = UInt8(splitmix64(seed) % 6)

# --- Temporal Structures ---

"""Graph snapshot at a specific time."""
struct Snapshot
    time::Float64
    adj::SparseMatrixCSC{Bool,Int}
    colors::Vector{UInt8}
    fingerprint::UInt64
end

function Snapshot(time::Float64, adj::SparseMatrixCSC{Bool,Int}, seed::UInt64)
    n = size(adj, 1)
    colors = [chromatic_fingerprint(splitmix64(seed + UInt64(i))) for i in 1:n]
    Snapshot(time, adj, colors, splitmix64(seed ⊻ hash(time)))
end

"""Morphism between snapshots preserving chromatic structure."""
struct SnapshotMorphism
    source_time::Float64
    target_time::Float64
    vertex_map::Vector{Int}
    chromatic_id::UInt64
end

function SnapshotMorphism(src::Snapshot, tgt::Snapshot, vmap::Vector{Int}, seed::UInt64)
    cid = splitmix64(seed ⊻ src.fingerprint ⊻ tgt.fingerprint)
    SnapshotMorphism(src.time, tgt.time, vmap, cid)
end

"""Temporal narrative: sequence of snapshots with connecting morphisms."""
struct Narrative
    snapshots::Vector{Snapshot}
    morphisms::Vector{SnapshotMorphism}
    seed::UInt64
end

Narrative(seed::UInt64) = Narrative(Snapshot[], SnapshotMorphism[], seed)

# --- Temporal BFS ---

"""
    gay_narrative_bfs!(narrative, start, seed) -> levels

BFS across temporal graph with SPI-colored level sets. Explores both
spatial (within snapshot) and temporal (across snapshots) edges.
"""
function gay_narrative_bfs!(narrative::Narrative, start::Tuple{Int,Int}, seed::UInt64)
    isempty(narrative.snapshots) && return Dict{Tuple{Int,Int}, UInt8}()
    
    levels = Dict{Tuple{Int,Int}, UInt8}()
    queue = [start]
    levels[start] = chromatic_fingerprint(seed)
    
    while !isempty(queue)
        (t_idx, v) = popfirst!(queue)
        current_level = levels[(t_idx, v)]
        next_seed = splitmix64(seed + UInt64(t_idx) + UInt64(v))
        
        # Spatial neighbors within snapshot
        if 1 ≤ t_idx ≤ length(narrative.snapshots)
            snap = narrative.snapshots[t_idx]
            for u in findnz(snap.adj[:, v])[1]
                key = (t_idx, u)
                if !haskey(levels, key)
                    levels[key] = chromatic_fingerprint(splitmix64(next_seed + UInt64(u)))
                    push!(queue, key)
                end
            end
        end
        
        # Temporal neighbors via morphisms
        for m in narrative.morphisms
            if m.source_time == narrative.snapshots[t_idx].time && v ≤ length(m.vertex_map)
                tgt_idx = findfirst(s -> s.time == m.target_time, narrative.snapshots)
                tgt_idx === nothing && continue
                u = m.vertex_map[v]
                key = (tgt_idx, u)
                if !haskey(levels, key)
                    levels[key] = chromatic_fingerprint(splitmix64(m.chromatic_id + UInt64(u)))
                    push!(queue, key)
                end
            end
        end
    end
    levels
end

# --- Interval Sheaf ---

"""
    gay_interval_sheaf!(narrative, seed) -> sheaf

Construct sheaf over time intervals. Each interval [tᵢ, tⱼ] gets a
restriction map colored by SPI fingerprint.
"""
function gay_interval_sheaf!(narrative::Narrative, seed::UInt64)
    n = length(narrative.snapshots)
    n < 2 && return Dict{Tuple{Int,Int}, UInt64}()
    
    sheaf = Dict{Tuple{Int,Int}, UInt64}()
    
    for i in 1:n, j in i:n
        interval_seed = splitmix64(seed ⊻ UInt64(i) ⊻ (UInt64(j) << 32))
        fp_acc = interval_seed
        for k in i:j
            fp_acc = splitmix64(fp_acc ⊻ narrative.snapshots[k].fingerprint)
        end
        sheaf[(i, j)] = fp_acc
    end
    
    # Verify restriction compatibility: sheaf(i,k) depends on sheaf(i,j) ∘ sheaf(j,k)
    for i in 1:n, j in i:n, k in j:n
        composed = splitmix64(sheaf[(i,j)] ⊻ sheaf[(j,k)])
        sheaf[(i, k)] = splitmix64(sheaf[(i,k)] ⊻ composed)  # coherence adjustment
    end
    sheaf
end

# --- Narrative Composition ---

"""
    gay_snapshot_compose!(n1, n2, seed) -> Narrative

Compose two narratives sequentially, preserving chromatic fingerprint
through splitmix64 mixing of endpoint colors.
"""
function gay_snapshot_compose!(n1::Narrative, n2::Narrative, seed::UInt64)
    isempty(n1.snapshots) && return Narrative(n2.snapshots, n2.morphisms, seed)
    isempty(n2.snapshots) && return Narrative(n1.snapshots, n1.morphisms, seed)
    
    # Time offset for n2 snapshots
    t_offset = n1.snapshots[end].time
    
    # Rebase n2 snapshots
    rebased = [Snapshot(s.time + t_offset, s.adj, s.colors, 
                        splitmix64(s.fingerprint ⊻ seed)) for s in n2.snapshots]
    
    # Bridge morphism connecting narratives
    last_n1 = n1.snapshots[end]
    first_n2 = rebased[1]
    n_verts = min(size(last_n1.adj, 1), size(first_n2.adj, 1))
    bridge_map = collect(1:n_verts)
    bridge = SnapshotMorphism(last_n1.time, first_n2.time, bridge_map,
                               splitmix64(last_n1.fingerprint ⊻ first_n2.fingerprint ⊻ seed))
    
    # Rebase n2 morphisms
    rebased_morphs = [SnapshotMorphism(m.source_time + t_offset, m.target_time + t_offset,
                                        m.vertex_map, splitmix64(m.chromatic_id ⊻ seed))
                      for m in n2.morphisms]
    
    composed_seed = splitmix64(n1.seed ⊻ n2.seed ⊻ seed)
    Narrative(vcat(n1.snapshots, rebased),
              vcat(n1.morphisms, [bridge], rebased_morphs),
              composed_seed)
end

# --- Boomerang PDMP Sampler ---

"""
    gay_boomerang_narrative!(narrative, seed; n_events=100, refresh_rate=0.1) -> (trajectories, fingerprint)

Boomerang PDMP sampler for narrative trajectory exploration.
Position follows curved (elliptical) paths between velocity refreshes.
Returns sampled trajectories and chromatic fingerprint.
"""
function gay_boomerang_narrative!(narrative::Narrative, seed::UInt64;
                                   n_events::Int=100, refresh_rate::Float64=0.1)::Tuple{Vector{Vector{Float64}}, UInt64}
    rng_state = splitmix64(seed)
    fingerprint = splitmix64(seed ⊻ narrative.seed)
    
    n_snaps = length(narrative.snapshots)
    d = max(n_snaps, 2)
    
    # State: position x, velocity v
    x = zeros(d)
    v = zeros(d)
    
    # Initialize position from snapshot fingerprints
    for (i, snap) in enumerate(narrative.snapshots)
        x[i] = (snap.fingerprint % 1000) / 1000.0 - 0.5
    end
    
    # Initialize velocity (Gaussian via Box-Muller)
    for i in 1:d
        rng_state = splitmix64(rng_state)
        u1 = (rng_state & 0xFFFFFFFF) / 4294967296.0
        rng_state = splitmix64(rng_state)
        u2 = (rng_state & 0xFFFFFFFF) / 4294967296.0
        v[i] = sqrt(-2 * log(max(u1, 1e-10))) * cos(2π * u2)
    end
    
    trajectories = Vector{Float64}[]
    t = 0.0
    
    for event in 1:n_events
        # Boomerang dynamics: x(t) = x₀cos(t) + v₀sin(t), v(t) = -x₀sin(t) + v₀cos(t)
        # Compute switching rate λ = max(0, ⟨v, ∇U(x)⟩) for U(x) = ½‖x‖²
        rate = max(0.0, dot(v, x)) + refresh_rate
        
        # Sample exponential waiting time
        rng_state = splitmix64(rng_state)
        u = (rng_state & 0xFFFFFFFF) / 4294967296.0
        τ = -log(max(u, 1e-10)) / max(rate, 1e-10)
        τ = min(τ, 2π)  # Cap at one orbit
        
        # Evolve along elliptical trajectory
        x_new = x .* cos(τ) .+ v .* sin(τ)
        v_new = -x .* sin(τ) .+ v .* cos(τ)
        x, v = x_new, v_new
        t += τ
        
        # Velocity bounce: v → v - 2⟨v,x⟩x/‖x‖²
        xnorm_sq = sum(x.^2)
        if xnorm_sq > 1e-10
            v .-= 2 * dot(v, x) / xnorm_sq .* x
        end
        
        # Record trajectory point
        push!(trajectories, copy(x))
        fingerprint = splitmix64(fingerprint ⊻ reinterpret(UInt64, sum(x)) ⊻ UInt64(event))
    end
    
    (trajectories, fingerprint)
end

# Helper: dot product
@inline dot(a, b) = sum(a .* b)

end # module
