# ═══════════════════════════════════════════════════════════════════════════════
# Spectral Bridge: Graph Eigenvectors ↔ Chromatic Identity ↔ Topological DL
# ═══════════════════════════════════════════════════════════════════════════════
#
# Bridges between:
#   1. dgleich GenericArpack.jl - Reproducible eigenvalue seeds
#   2. Gay.jl SplittableRandoms - SPI chromatic identity  
#   3. PyT-Team TDL - Hodge Laplacians on simplicial complexes
#
# Key insight: Eigenvalue problems have 4-seed state (1,3,5,7) that maps to
# our color palette. This enables chromatic verification of spectral methods.
#
# ═══════════════════════════════════════════════════════════════════════════════

using LinearAlgebra
using SparseArrays
using Base.Threads: @threads, nthreads

export SpectralColorBridge, ArpackSeed, HodgeLaplacian
export color_eigenvector, verify_spectral_spi, eigencolor_fingerprint
export simplicial_hodge, chromatic_spectral_clustering
export demo_spectral_bridge

# ═══════════════════════════════════════════════════════════════════════════════
# Arpack Seed ↔ Gay Seed Bridge
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ArpackSeed

GenericArpack.jl uses iseed::NTuple{4,Int64} = (1,3,5,7) for reproducibility.
We bridge this to Gay.jl's UInt64 seed for chromatic identity.
"""
struct ArpackSeed
    iseed::NTuple{4, Int64}
    gay_seed::UInt64
    
    function ArpackSeed(iseed::NTuple{4, Int64} = (1, 3, 5, 7))
        # Convert 4-tuple to single UInt64 via mixing
        gay_seed = UInt64(0)
        for (i, s) in enumerate(iseed)
            gay_seed ⊻= splitmix64_mix(UInt64(s) ⊻ UInt64(i * 0x9e3779b97f4a7c15))
        end
        new(iseed, gay_seed)
    end
end

"""
    ArpackSeed(gay_seed::UInt64)

Create ArpackSeed from Gay.jl seed (reverse bridge).
"""
function ArpackSeed(gay_seed::UInt64)
    # Split into 4 odd integers (Arpack requirement)
    s1 = Int64((gay_seed & 0xFFFF) | 1)
    s2 = Int64(((gay_seed >> 16) & 0xFFFF) | 1)
    s3 = Int64(((gay_seed >> 32) & 0xFFFF) | 1)
    s4 = Int64(((gay_seed >> 48) & 0xFFFF) | 1)
    ArpackSeed((s1, s2, s3, s4))
end

# ═══════════════════════════════════════════════════════════════════════════════
# Spectral Color Bridge
# ═══════════════════════════════════════════════════════════════════════════════

"""
    SpectralColorBridge

Bridge between spectral methods and chromatic identity.
Tracks eigenvalue/eigenvector computations with SPI colors.
"""
struct SpectralColorBridge
    seed::ArpackSeed
    matrix_fingerprint::UInt64  # Hash of input matrix
    eigenvalue_colors::Vector{NTuple{3, Float32}}
    eigenvector_fingerprints::Vector{UInt64}
    dimension::Int
    n_eigenvalues::Int
end

"""
    color_eigenvector(v::Vector{T}, seed::UInt64, index::Int) -> NTuple{3, Float32}

Assign a deterministic color to an eigenvector based on its fingerprint.
"""
function color_eigenvector(v::AbstractVector{T}, seed::UInt64, index::Int) where T<:Number
    # Compute fingerprint of eigenvector
    fp = UInt64(0)
    for (i, val) in enumerate(v)
        # Quantize to avoid floating point noise, handle negative values
        quantized = round(Int64, Float64(real(val)) * 1e6) % Int64(1 << 32)
        fp ⊻= splitmix64_mix(reinterpret(UInt64, quantized) ⊻ UInt64(i))
    end
    
    # Color from fingerprint
    hash_color(seed ⊻ UInt64(index), fp)
end

"""
    eigencolor_fingerprint(eigenvalues::Vector, eigenvectors::Matrix; seed)

Compute XOR fingerprint of colored eigenpairs.
"""
function eigencolor_fingerprint(eigenvalues::AbstractVector, 
                                 eigenvectors::AbstractMatrix;
                                 seed::UInt64 = GAY_SEED)
    n = length(eigenvalues)
    fp = UInt64(0)
    
    for i in 1:n
        # Color for eigenvalue - handle potentially large values
        λ_quantized = round(Int64, real(eigenvalues[i]) * 1e6) % Int64(1 << 32)
        λ_color = hash_color(seed, splitmix64_mix(reinterpret(UInt64, λ_quantized)))
        
        # Color for eigenvector
        v_color = color_eigenvector(view(eigenvectors, :, i), seed, i)
        
        # Combine into fingerprint using individual components
        r1, g1, b1 = λ_color
        r2, g2, b2 = v_color
        
        # Hash each color component (convert Float32 to UInt32 via bits)
        fp ⊻= splitmix64_mix(
            UInt64(reinterpret(UInt32, r1)) ⊻ (UInt64(reinterpret(UInt32, g1)) << 16) ⊻ 
            (UInt64(reinterpret(UInt32, b1)) << 32) ⊻ UInt64(i)
        )
        fp ⊻= splitmix64_mix(
            UInt64(reinterpret(UInt32, r2)) ⊻ (UInt64(reinterpret(UInt32, g2)) << 16) ⊻ 
            (UInt64(reinterpret(UInt32, b2)) << 32) ⊻ UInt64(n)
        )
    end
    
    fp
end

"""
    verify_spectral_spi(A::AbstractMatrix; seed, k=6) -> (Bool, SpectralColorBridge)

Verify spectral computation maintains SPI:
1. Compute eigendecomposition twice
2. Color both computations
3. Verify fingerprints match
"""
function verify_spectral_spi(A::AbstractMatrix; seed::UInt64 = GAY_SEED, k::Int = 6)
    n = size(A, 1)
    k = min(k, n)
    
    # First computation
    eigen1 = eigen(Symmetric(Matrix(A)))
    λ1 = eigen1.values[end-k+1:end]
    V1 = eigen1.vectors[:, end-k+1:end]
    fp1 = eigencolor_fingerprint(λ1, V1; seed=seed)
    
    # Second computation (should be identical due to determinism)
    eigen2 = eigen(Symmetric(Matrix(A)))
    λ2 = eigen2.values[end-k+1:end]
    V2 = eigen2.vectors[:, end-k+1:end]
    fp2 = eigencolor_fingerprint(λ2, V2; seed=seed)
    
    # Build color bridge
    arpack_seed = ArpackSeed(seed)
    matrix_fp = matrix_fingerprint(A)
    
    eigen_colors = [color_eigenvector(view(V1, :, i), seed, i) for i in 1:k]
    evec_fps = [eigenvector_fp(view(V1, :, i)) for i in 1:k]
    
    bridge = SpectralColorBridge(arpack_seed, matrix_fp, eigen_colors, evec_fps, n, k)
    
    (fp1 == fp2, bridge)
end

function matrix_fingerprint(A::AbstractMatrix)
    fp = UInt64(0)
    for i in axes(A, 1)
        for j in axes(A, 2)
            if A[i, j] != 0
                val = round(Int64, Float64(real(A[i, j])) * 1e6) % Int64(1 << 32)
                fp ⊻= splitmix64_mix(reinterpret(UInt64, val) ⊻ UInt64(i * 65537 + j))
            end
        end
    end
    fp
end

function eigenvector_fp(v::AbstractVector)
    fp = UInt64(0)
    for (i, val) in enumerate(v)
        quantized = round(Int64, Float64(real(val)) * 1e6) % Int64(1 << 32)
        fp ⊻= splitmix64_mix(reinterpret(UInt64, quantized) ⊻ UInt64(i))
    end
    fp
end

# ═══════════════════════════════════════════════════════════════════════════════
# Hodge Laplacian for Topological DL
# ═══════════════════════════════════════════════════════════════════════════════

"""
    HodgeLaplacian

Hodge Laplacian on a simplicial complex.
L_k = B_k^T B_k + B_{k+1} B_{k+1}^T

For k=0: Graph Laplacian
For k=1: Edge Laplacian (captures cycles)
For k=2: Triangle Laplacian (captures cavities)
"""
struct HodgeLaplacian{T}
    order::Int  # 0, 1, or 2
    matrix::SparseMatrixCSC{T, Int}
    boundary_down::Union{Nothing, SparseMatrixCSC{T, Int}}  # B_k
    boundary_up::Union{Nothing, SparseMatrixCSC{T, Int}}    # B_{k+1}
    dimension::Int
    color::NTuple{3, Float32}  # Chromatic identity
    fingerprint::UInt64
end

"""
    simplicial_hodge(vertices, edges, triangles; order=1, seed=GAY_SEED)

Construct Hodge Laplacian from simplicial complex data.
"""
function simplicial_hodge(n_vertices::Int,
                          edges::Vector{Tuple{Int, Int}},
                          triangles::Vector{Tuple{Int, Int, Int}} = Tuple{Int, Int, Int}[];
                          order::Int = 1,
                          seed::UInt64 = GAY_SEED)
    
    n_edges = length(edges)
    n_triangles = length(triangles)
    
    if order == 0
        # Graph Laplacian: L_0 = B_1^T B_1
        B1 = boundary_1(n_vertices, edges)
        L = B1' * B1
        
        fp = matrix_fingerprint(L)
        color = hash_color(seed, fp)
        
        return HodgeLaplacian(0, L, nothing, B1, n_vertices, color, fp)
        
    elseif order == 1
        # Edge Laplacian: L_1 = B_1^T B_1 + B_2 B_2^T
        # Note: B1 is n_vertices × n_edges, so B1'*B1 is n_edges × n_edges
        B1 = boundary_1(n_vertices, edges)
        
        if isempty(triangles)
            L = B1' * B1  # n_edges × n_edges
            fp = matrix_fingerprint(L)
            color = hash_color(seed, fp)
            return HodgeLaplacian(1, L, B1, nothing, n_edges, color, fp)
        else
            B2 = boundary_2(edges, triangles)
            # B2 is n_edges × n_triangles, so B2*B2' is n_edges × n_edges
            L = B1' * B1 + B2 * B2'
            fp = matrix_fingerprint(L)
            color = hash_color(seed, fp)
            return HodgeLaplacian(1, L, B1, B2, n_edges, color, fp)
        end
        
    elseif order == 2
        # Triangle Laplacian: L_2 = B_2 B_2^T
        @assert !isempty(triangles) "Need triangles for order-2 Hodge Laplacian"
        B2 = boundary_2(edges, triangles)
        L = B2 * B2'
        
        fp = matrix_fingerprint(L)
        color = hash_color(seed, fp)
        
        return HodgeLaplacian(2, L, B2, nothing, n_triangles, color, fp)
    else
        error("Order must be 0, 1, or 2")
    end
end

"""
    boundary_1(n_vertices, edges) -> SparseMatrix

Boundary operator B_1: edges → vertices
B_1[v, e] = +1 if v is head of e, -1 if v is tail
"""
function boundary_1(n_vertices::Int, edges::Vector{Tuple{Int, Int}})
    n_edges = length(edges)
    I = Int[]
    J = Int[]
    V = Float64[]
    
    for (e, (u, v)) in enumerate(edges)
        # Edge e goes from u to v
        push!(I, u); push!(J, e); push!(V, -1.0)
        push!(I, v); push!(J, e); push!(V, +1.0)
    end
    
    sparse(I, J, V, n_vertices, n_edges)
end

"""
    boundary_2(edges, triangles) -> SparseMatrix

Boundary operator B_2: triangles → edges
B_2[e, t] = ±1 if edge e is in triangle t
"""
function boundary_2(edges::Vector{Tuple{Int, Int}}, 
                    triangles::Vector{Tuple{Int, Int, Int}})
    n_edges = length(edges)
    n_triangles = length(triangles)
    
    # Build edge index
    edge_idx = Dict{Tuple{Int, Int}, Int}()
    for (i, (u, v)) in enumerate(edges)
        edge_idx[(min(u, v), max(u, v))] = i
    end
    
    I = Int[]
    J = Int[]
    V = Float64[]
    
    for (t, (a, b, c)) in enumerate(triangles)
        # Triangle (a, b, c) has edges (a,b), (b,c), (a,c)
        # Orientation: +1 for consistent, -1 for reversed
        
        e1 = (min(a, b), max(a, b))
        e2 = (min(b, c), max(b, c))
        e3 = (min(a, c), max(a, c))
        
        if haskey(edge_idx, e1)
            push!(I, edge_idx[e1]); push!(J, t); push!(V, a < b ? 1.0 : -1.0)
        end
        if haskey(edge_idx, e2)
            push!(I, edge_idx[e2]); push!(J, t); push!(V, b < c ? 1.0 : -1.0)
        end
        if haskey(edge_idx, e3)
            push!(I, edge_idx[e3]); push!(J, t); push!(V, a < c ? -1.0 : 1.0)
        end
    end
    
    sparse(I, J, V, n_edges, n_triangles)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Chromatic Spectral Clustering
# ═══════════════════════════════════════════════════════════════════════════════

"""
    chromatic_spectral_clustering(A, k; seed)

Spectral clustering with chromatic identity for each cluster.
"""
function chromatic_spectral_clustering(A::AbstractMatrix, k::Int;
                                        seed::UInt64 = GAY_SEED)
    n = size(A, 1)
    
    # Compute graph Laplacian
    D = Diagonal(vec(sum(A, dims=2)))
    L = D - A
    
    # Get k smallest eigenvectors (skip first constant eigenvector)
    eig = eigen(Symmetric(Matrix(L)))
    V = eig.vectors[:, 2:k+1]  # Skip first eigenvector
    λ = eig.values[2:k+1]
    
    # Assign clusters via k-means on eigenvector embedding
    clusters = simple_kmeans(V, k)
    
    # Assign colors to clusters
    cluster_colors = [hash_color(seed, UInt64(c)) for c in 1:k]
    
    # Color each node by its cluster
    node_colors = [cluster_colors[clusters[i]] for i in 1:n]
    
    # Fingerprint
    fp = eigencolor_fingerprint(λ, V; seed=seed)
    
    (clusters=clusters, colors=node_colors, cluster_colors=cluster_colors, 
     eigenvalues=λ, fingerprint=fp)
end

"""
Simple k-means for spectral embedding.
"""
function simple_kmeans(X::AbstractMatrix, k::Int; max_iter::Int = 100)
    n, d = size(X)
    
    # Initialize centroids randomly
    centroids = X[rand(1:n, k), :]
    assignments = zeros(Int, n)
    
    for _ in 1:max_iter
        # Assign points to nearest centroid
        new_assignments = zeros(Int, n)
        for i in 1:n
            min_dist = Inf
            for c in 1:k
                dist = sum((X[i, :] .- centroids[c, :]).^2)
                if dist < min_dist
                    min_dist = dist
                    new_assignments[i] = c
                end
            end
        end
        
        # Check convergence
        if new_assignments == assignments
            break
        end
        assignments = new_assignments
        
        # Update centroids
        for c in 1:k
            mask = assignments .== c
            if any(mask)
                centroids[c, :] = vec(mean(X[mask, :], dims=1))
            end
        end
    end
    
    assignments
end

# ═══════════════════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════════════════

function demo_spectral_bridge()
    println("═" ^ 70)
    println("  SPECTRAL BRIDGE: Eigenvectors ↔ Chromatic Identity ↔ TDL")
    println("═" ^ 70)
    println()
    
    # 1. Arpack seed bridge
    println("1. ARPACK SEED ↔ GAY SEED BRIDGE")
    arpack_default = ArpackSeed()
    println("   Arpack iseed: $(arpack_default.iseed)")
    println("   Gay seed: 0x$(string(arpack_default.gay_seed, base=16, pad=16))")
    
    # Reverse bridge
    arpack_from_gay = ArpackSeed(GAY_SEED)
    println("   GAY_SEED → iseed: $(arpack_from_gay.iseed)")
    println()
    
    # 2. Spectral SPI verification
    println("2. SPECTRAL SPI VERIFICATION")
    n = 50
    A = sprand(n, n, 0.2)
    A = A + A'  # Symmetric
    
    spi_ok, bridge = verify_spectral_spi(A; k=6)
    println("   Matrix: $(n)×$(n) sparse symmetric")
    println("   SPI verified: $(spi_ok ? "✓" : "✗")")
    println("   Matrix fingerprint: 0x$(string(bridge.matrix_fingerprint, base=16, pad=16))")
    println("   Eigenvector colors:")
    for (i, color) in enumerate(bridge.eigenvalue_colors)
        r, g, b = round.(Int, color .* 255)
        println("     λ$i: RGB($r, $g, $b)")
    end
    println()
    
    # 3. Hodge Laplacian
    println("3. HODGE LAPLACIAN (Simplicial Complex)")
    vertices = 6
    edges = [(1,2), (2,3), (3,1), (1,4), (2,5), (3,6), (4,5), (5,6), (6,4)]
    triangles = [(1,2,3), (4,5,6)]
    
    L0 = simplicial_hodge(vertices, edges, triangles; order=0)
    L1 = simplicial_hodge(vertices, edges, triangles; order=1)
    L2 = simplicial_hodge(vertices, edges, triangles; order=2)
    
    println("   L0 (vertices): $(L0.dimension)×$(L0.dimension), color RGB$(round.(Int, L0.color .* 255))")
    println("   L1 (edges): $(L1.dimension)×$(L1.dimension), color RGB$(round.(Int, L1.color .* 255))")
    println("   L2 (triangles): $(L2.dimension)×$(L2.dimension), color RGB$(round.(Int, L2.color .* 255))")
    
    # Betti numbers from kernel dimensions
    rank_B1 = rank(Matrix(L0.boundary_up))
    rank_B2 = rank(Matrix(L1.boundary_up))
    β0 = vertices - rank_B1
    β1 = length(edges) - rank_B1 - rank_B2
    β2 = length(triangles) - rank_B2
    println("   Betti numbers: β₀=$β0, β₁=$β1, β₂=$β2")
    println()
    
    # 4. Chromatic spectral clustering
    println("4. CHROMATIC SPECTRAL CLUSTERING")
    # Create a graph with 3 clusters
    cluster_size = 20
    n_clusters = 3
    A_cluster = zeros(n_clusters * cluster_size, n_clusters * cluster_size)
    for c in 1:n_clusters
        start_idx = (c - 1) * cluster_size + 1
        end_idx = c * cluster_size
        for i in start_idx:end_idx
            for j in (i+1):end_idx
                if rand() < 0.5
                    A_cluster[i, j] = A_cluster[j, i] = 1.0
                end
            end
        end
        # Sparse inter-cluster connections
        if c < n_clusters
            next_start = c * cluster_size + 1
            for _ in 1:3
                i = rand(start_idx:end_idx)
                j = rand(next_start:next_start + cluster_size - 1)
                A_cluster[i, j] = A_cluster[j, i] = 1.0
            end
        end
    end
    
    result = chromatic_spectral_clustering(A_cluster, n_clusters)
    println("   Graph: $(size(A_cluster, 1)) nodes, 3 clusters")
    println("   Cluster colors:")
    for (c, color) in enumerate(result.cluster_colors)
        count = sum(result.clusters .== c)
        r, g, b = round.(Int, color .* 255)
        println("     Cluster $c: RGB($r, $g, $b) - $count nodes")
    end
    println("   Eigenvalues: $(round.(result.eigenvalues, digits=4))")
    println("   Fingerprint: 0x$(string(result.fingerprint, base=16, pad=16))")
    println()
    
    println("═" ^ 70)
    println("  SPECTRAL BRIDGE COMPLETE")
    println("═" ^ 70)
end

# end of spectral_bridge.jl
