# Core gaymc algorithms - gaimc ported with SPI chromatic identity
module GayCoreAlgorithms

export GayGraph, add_edge!, gay_bfs!, gay_dfs!, gay_dijkstra!, gay_mst_prim!, gay_scomponents!, gay_corenums!
export splitmix64, hash_color

# Splitmix64 PRNG
@inline function splitmix64(x::UInt64)
    x += 0x9e3779b97f4a7c15
    x = (x ⊻ (x >> 30)) * 0xbf58476d1ce4e5b9
    x = (x ⊻ (x >> 27)) * 0x94d049bb133111eb
    x ⊻ (x >> 31)
end

@inline function hash_color(seed::UInt64, index::UInt64)
    h = splitmix64(seed ⊻ (index * 0x9e3779b97f4a7c15))
    r = Float32((h & 0xFF)) / 255.0f0
    g = Float32(((h >> 8) & 0xFF)) / 255.0f0
    b = Float32(((h >> 16) & 0xFF)) / 255.0f0
    (r, g, b)
end

struct GayGraph
    n::Int
    adj::Vector{Vector{Int}}
    weights::Vector{Vector{Float64}}
end

GayGraph(n::Int) = GayGraph(n, [Int[] for _ in 1:n], [Float64[] for _ in 1:n])

function add_edge!(G::GayGraph, u::Int, v::Int, w::Float64=1.0)
    push!(G.adj[u], v); push!(G.weights[u], w)
    push!(G.adj[v], u); push!(G.weights[v], w)
end

"""BFS with color per level. Returns (levels, colors, fingerprint)."""
function gay_bfs!(G::GayGraph, start::Int; seed::UInt64=0x42)
    levels = fill(-1, G.n)
    colors = Vector{NTuple{3,Float32}}(undef, G.n)
    levels[start] = 0
    queue = [start]
    fp = UInt64(0)
    
    while !isempty(queue)
        u = popfirst!(queue)
        colors[u] = hash_color(seed, UInt64(levels[u]))
        fp ⊻= splitmix64(seed ⊻ UInt64(u))
        for v in G.adj[u]
            if levels[v] == -1
                levels[v] = levels[u] + 1
                push!(queue, v)
            end
        end
    end
    (levels, colors, fp)
end

"""DFS with color per discovery time. Returns (discovery, colors, fingerprint)."""
function gay_dfs!(G::GayGraph, start::Int; seed::UInt64=0x42)
    discovery = fill(-1, G.n)
    colors = Vector{NTuple{3,Float32}}(undef, G.n)
    time = Ref(0)
    fp = UInt64(0)
    
    function dfs(u)
        time[] += 1
        discovery[u] = time[]
        colors[u] = hash_color(seed, UInt64(time[]))
        fp ⊻= splitmix64(seed ⊻ UInt64(u))
        for v in G.adj[u]
            discovery[v] == -1 && dfs(v)
        end
    end
    
    dfs(start)
    (discovery, colors, fp)
end

"""Dijkstra with color per distance class. Returns (dist, colors, fingerprint)."""
function gay_dijkstra!(G::GayGraph, start::Int; seed::UInt64=0x42)
    dist = fill(Inf, G.n)
    colors = Vector{NTuple{3,Float32}}(undef, G.n)
    dist[start] = 0.0
    pq = [(0.0, start)]
    fp = UInt64(0)
    
    while !isempty(pq)
        sort!(pq, by=first)
        d, u = popfirst!(pq)
        d > dist[u] && continue
        colors[u] = hash_color(seed, UInt64(floor(d)))
        fp ⊻= splitmix64(seed ⊻ UInt64(u))
        for (i, v) in enumerate(G.adj[u])
            nd = d + G.weights[u][i]
            if nd < dist[v]
                dist[v] = nd
                push!(pq, (nd, v))
            end
        end
    end
    (dist, colors, fp)
end

"""Prim MST with color per tree edge. Returns (parent, edge_colors, fingerprint)."""
function gay_mst_prim!(G::GayGraph; seed::UInt64=0x42)
    n = G.n
    in_mst = falses(n)
    parent = fill(-1, n)
    key = fill(Inf, n)
    edge_colors = Dict{Tuple{Int,Int}, NTuple{3,Float32}}()
    key[1] = 0.0
    fp = UInt64(0)
    edge_idx = 0
    
    for _ in 1:n
        u = argmin(i -> in_mst[i] ? Inf : key[i], 1:n)
        in_mst[u] = true
        if parent[u] != -1
            edge_idx += 1
            edge_colors[(parent[u], u)] = hash_color(seed, UInt64(edge_idx))
            fp ⊻= splitmix64(seed ⊻ UInt64(edge_idx))
        end
        for (i, v) in enumerate(G.adj[u])
            w = G.weights[u][i]
            if !in_mst[v] && w < key[v]
                key[v] = w
                parent[v] = u
            end
        end
    end
    (parent, edge_colors, fp)
end

"""SCCs with color per component. Returns (component, colors, fingerprint)."""
function gay_scomponents!(G::GayGraph; seed::UInt64=0x42)
    n = G.n
    comp = zeros(Int, n)
    colors = Vector{NTuple{3,Float32}}(undef, n)
    visited = falses(n)
    order = Int[]
    fp = UInt64(0)
    
    # First DFS for finish order
    function dfs1(u)
        visited[u] = true
        for v in G.adj[u]
            !visited[v] && dfs1(v)
        end
        push!(order, u)
    end
    
    for u in 1:n
        !visited[u] && dfs1(u)
    end
    
    # Reverse graph DFS for SCCs
    fill!(visited, false)
    comp_id = 0
    
    function dfs2(u, c)
        visited[u] = true
        comp[u] = c
        colors[u] = hash_color(seed, UInt64(c))
        fp ⊻= splitmix64(seed ⊻ UInt64(u))
        for v in G.adj[u]
            !visited[v] && dfs2(v, c)
        end
    end
    
    for u in reverse(order)
        if !visited[u]
            comp_id += 1
            dfs2(u, comp_id)
        end
    end
    
    (comp, colors, fp)
end

"""K-cores with color per k. Returns (core_number, colors, fingerprint)."""
function gay_corenums!(G::GayGraph; seed::UInt64=0x42)
    n = G.n
    deg = [length(G.adj[u]) for u in 1:n]
    core = copy(deg)
    colors = Vector{NTuple{3,Float32}}(undef, n)
    removed = falses(n)
    fp = UInt64(0)
    
    for k in 0:maximum(deg)
        while true
            found = false
            for u in 1:n
                if !removed[u] && deg[u] <= k
                    removed[u] = true
                    core[u] = k
                    colors[u] = hash_color(seed, UInt64(k))
                    fp ⊻= splitmix64(seed ⊻ UInt64(u))
                    for v in G.adj[u]
                        !removed[v] && (deg[v] -= 1)
                    end
                    found = true
                end
            end
            !found && break
        end
    end
    (core, colors, fp)
end

end # module
