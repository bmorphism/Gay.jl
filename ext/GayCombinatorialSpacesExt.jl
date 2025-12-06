# CombinatorialSpaces.jl extension for Gay.jl
# SPI-compliant coloring for simplicial sets, DEC operators, and mesh structures

module GayCombinatorialSpacesExt

using Gay: hash_color_rgb, splitmix64, GAY_SEED, color_fingerprint
using CombinatorialSpaces
using Colors: RGB, HSL, convert

export color_deltaset_1d, color_deltaset_2d, color_embedded_dual
export color_multigrid_levels, color_poisson_2d, color_euler_flow
export color_dec_operator, spi_verify_parallel, associative_color_reduce

# ═══════════════════════════════════════════════════════════════════════════
# 1D Delta Set (Semi-Simplicial Set)
# ═══════════════════════════════════════════════════════════════════════════

"""
    color_deltaset_1d(ds::DeltaSet1D; seed=GAY_SEED) -> NamedTuple

Color a 1D delta set (vertices + edges) with SPI colors.

Vertex colors: `hash_color(v, seed)`
Edge colors: `hash_color(src ⊻ tgt, seed)` - XOR ensures associativity

# Example
```julia
ds = DeltaSet1D()
add_vertices!(ds, 4)
add_edge!(ds, 2, 1)
add_edge!(ds, 3, 2)
colors = color_deltaset_1d(ds)
```
"""
function color_deltaset_1d(ds::DeltaSet1D; seed::UInt64=GAY_SEED)
    n_v = nv(ds)
    n_e = ne(ds)
    
    vertex_colors = [hash_color_rgb(UInt64(v), seed) for v in 1:n_v]
    
    edge_colors = map(1:n_e) do e
        v0 = ds[e, :∂v0]
        v1 = ds[e, :∂v1]
        idx = UInt64(v0) ⊻ UInt64(v1)
        hash_color_rgb(idx, seed)
    end
    
    (vertices=vertex_colors, edges=edge_colors)
end

# ═══════════════════════════════════════════════════════════════════════════
# 2D Delta Set (Triangulated Surfaces)
# ═══════════════════════════════════════════════════════════════════════════

"""
    color_deltaset_2d(ds::DeltaSet2D; seed=GAY_SEED) -> NamedTuple

Color a 2D delta set (vertices + edges + triangles) with SPI colors.

Triangle colors use XOR of sorted vertex indices (associative, commutative).

# Example
```julia
ds = DeltaSet2D()
add_vertices!(ds, 4)
glue_triangle!(ds, 1, 2, 3)
glue_triangle!(ds, 1, 3, 4)
colors = color_deltaset_2d(ds)
```
"""
function color_deltaset_2d(ds::DeltaSet2D; seed::UInt64=GAY_SEED)
    base = color_deltaset_1d(ds; seed)
    
    n_tri = ntriangles(ds)
    tri_colors = map(1:n_tri) do t
        verts = triangle_vertices(ds, t)
        sorted = sort(collect(verts))
        idx = reduce(⊻, UInt64.(sorted))
        hash_color_rgb(idx, seed)
    end
    
    (vertices=base.vertices, edges=base.edges, triangles=tri_colors)
end

# ═══════════════════════════════════════════════════════════════════════════
# Embedded Delta Set with Dual Complex
# ═══════════════════════════════════════════════════════════════════════════

"""
    color_embedded_dual(ds::EmbeddedDeltaSet2D; seed=GAY_SEED) -> NamedTuple

Color an embedded delta set AND its dual complex.

Dual vertices (triangle centroids) inherit primal triangle colors.
Dual edges inherit primal edge colors (XOR-based).

# Example
```julia
ds = EmbeddedDeltaSet2D{Bool,Point3D}()
ds, _ = loadmesh(OpenMesh, "mesh.obj")
dual = DualSimplicialSet(ds)
subdivide_duals!(dual, Barycenter())
colors = color_embedded_dual(ds)
```
"""
function color_embedded_dual(ds::EmbeddedDeltaSet2D; seed::UInt64=GAY_SEED)
    primal = color_deltaset_2d(ds; seed)
    
    dual_vertex_colors = primal.triangles
    
    n_e = ne(ds)
    dual_edge_colors = primal.edges[1:n_e]
    
    (primal=primal, 
     dual_vertices=dual_vertex_colors, 
     dual_edges=dual_edge_colors)
end

# ═══════════════════════════════════════════════════════════════════════════
# Multigrid Hierarchy Coloring
# ═══════════════════════════════════════════════════════════════════════════

"""
    color_multigrid_levels(levels::Vector; seed=GAY_SEED) -> Vector{NamedTuple}

Color multigrid hierarchy with level-dependent seeding.

Each level uses `seed ⊻ (level * golden_ratio)` for consistent but distinct colors.
Includes fingerprints for SPI verification across levels.

# Example
```julia
coarse = DeltaSet1D()
add_vertices!(coarse, 2)
add_edge!(coarse, 2, 1)
levels = repeated_subdivisions(coarse, 5)
colored_levels = color_multigrid_levels(levels)
```
"""
function color_multigrid_levels(levels::Vector; seed::UInt64=GAY_SEED)
    colored_levels = []
    
    for (level_idx, ds) in enumerate(levels)
        level_seed = seed ⊻ UInt64(level_idx * 0x9e3779b97f4a7c15)
        colors = color_deltaset_1d(ds; seed=level_seed)
        
        push!(colored_levels, (
            level=level_idx,
            seed=level_seed,
            colors=colors,
            fingerprint=xor_fingerprint_vertices(colors.vertices)
        ))
    end
    
    colored_levels
end

function xor_fingerprint_vertices(colors::Vector{RGB{Float32}})
    fp = UInt64(0)
    for c in colors
        r_bits = reinterpret(UInt32, Float32(c.r))
        g_bits = reinterpret(UInt32, Float32(c.g))
        b_bits = reinterpret(UInt32, Float32(c.b))
        fp ⊻= UInt64(r_bits) | (UInt64(g_bits) << 24) | (UInt64(b_bits) << 48)
    end
    fp
end

# ═══════════════════════════════════════════════════════════════════════════
# Poisson Equation Solution Coloring
# ═══════════════════════════════════════════════════════════════════════════

"""
    color_poisson_2d(ds::DeltaSet2D, u::Vector; seed=GAY_SEED) -> NamedTuple

Color Poisson equation solution on triangular mesh.

Vertex hue from mesh position (SPI), lightness from solution value.

# Example
```julia
ds = triangulated_grid(20, 20, 1.0, 1.0)
L = ∇²(Val{0}, ds)
u = L \\ randn(nv(ds))
colored = color_poisson_2d(ds, u)
```
"""
function color_poisson_2d(ds::DeltaSet2D, u::Vector{T}; 
                          seed::UInt64=GAY_SEED) where T
    n_v = nv(ds)
    @assert length(u) == n_v "Solution vector must match vertex count"
    
    umin, umax = extrema(u)
    range_val = umax - umin
    range_val = range_val > 0 ? range_val : one(T)
    
    colors = map(1:n_v) do v
        base_color = hash_color_rgb(UInt64(v), seed)
        base_hsl = convert(HSL, base_color)
        
        normalized = (u[v] - umin) / range_val
        lightness = 0.2 + 0.6 * normalized
        
        convert(RGB{Float32}, HSL(base_hsl.h, 0.8f0, Float32(lightness)))
    end
    
    (vertices=colors, solution=u)
end

# ═══════════════════════════════════════════════════════════════════════════
# Euler Flow Coloring
# ═══════════════════════════════════════════════════════════════════════════

"""
    color_euler_flow(ds::EmbeddedDeltaSet2D, u_flat::Vector; seed=GAY_SEED) -> NamedTuple

Color velocity field (1-form) from Euler equations.

Edge colors encode flow direction (hue shift) and magnitude (saturation).

# Example
```julia
ds = EmbeddedDeltaSet2D{Bool,Point3D}()
# ... setup mesh ...
u_flat = ♭(ds, velocity_field)
colored = color_euler_flow(ds, u_flat)
```
"""
function color_euler_flow(ds::EmbeddedDeltaSet2D, u_flat::Vector{T};
                          seed::UInt64=GAY_SEED) where T
    n_e = ne(ds)
    @assert length(u_flat) == n_e "1-form must have one value per edge"
    
    umax = maximum(abs, u_flat)
    umax = umax > 0 ? umax : one(T)
    
    edge_colors = map(1:n_e) do e
        v0 = ds[e, :∂v0]
        v1 = ds[e, :∂v1]
        idx = UInt64(v0) ⊻ UInt64(v1)
        base_color = hash_color_rgb(idx, seed)
        base_hsl = convert(HSL, base_color)
        
        normalized = u_flat[e] / umax
        saturation = Float32(abs(normalized))
        hue_shift = normalized > 0 ? 0.0f0 : 180.0f0
        
        convert(RGB{Float32}, HSL(mod(base_hsl.h + hue_shift, 360.0f0), saturation, 0.5f0))
    end
    
    (edges=edge_colors, flow=u_flat)
end

# ═══════════════════════════════════════════════════════════════════════════
# DEC Operator Coloring
# ═══════════════════════════════════════════════════════════════════════════

"""
    color_dec_operator(op::Symbol, ds; seed=GAY_SEED) -> Function

Get a coloring function for DEC operator matrix entries.

Operators: `:d0`, `:d1`, `:star0`, `:star1`, `:star2`, `:laplacian`, `:flat`, `:sharp`

# Example
```julia
colorer = color_dec_operator(:laplacian, ds)
c = colorer(3, 5)  # Color for matrix entry (3, 5)
```
"""
function color_dec_operator(op::Symbol, ds; seed::UInt64=GAY_SEED)
    op_hues = Dict(
        :d0 => 30.0f0,
        :d1 => 60.0f0,
        :star0 => 180.0f0,
        :star1 => 210.0f0,
        :star2 => 240.0f0,
        :laplacian => 300.0f0,
        :flat => 120.0f0,
        :sharp => 150.0f0,
    )
    
    base_hue = get(op_hues, op, 0.0f0)
    
    function color_entry(i::Int, j::Int)
        idx = UInt64(i) ⊻ (UInt64(j) << 32)
        h_offset = Float32((splitmix64(seed ⊻ idx) % 30)) - 15.0f0
        h = mod(base_hue + h_offset, 360.0f0)
        convert(RGB{Float32}, HSL(h, 0.7f0, 0.5f0))
    end
    
    color_entry
end

# ═══════════════════════════════════════════════════════════════════════════
# SPI Parallel Verification
# ═══════════════════════════════════════════════════════════════════════════

"""
    spi_verify_parallel(ds::DeltaSet2D, seed=GAY_SEED; n_threads=Threads.nthreads()) -> NamedTuple

Verify SPI holds under parallel execution.

XOR fingerprinting is associative: parallel and sequential must match.

# Example
```julia
ds = triangulated_grid(100, 100, 1.0, 1.0)
result = spi_verify_parallel(ds, GAY_SEED)
@assert result.spi_verified
```
"""
function spi_verify_parallel(ds::DeltaSet2D, seed::UInt64=GAY_SEED; 
                             n_threads::Int=Threads.nthreads())
    n_v = nv(ds)
    
    results = Vector{UInt64}(undef, n_threads)
    
    Threads.@threads for tid in 1:n_threads
        chunk_size = cld(n_v, n_threads)
        start_idx = (tid - 1) * chunk_size + 1
        end_idx = min(tid * chunk_size, n_v)
        
        local_fp = UInt64(0)
        for v in start_idx:end_idx
            c = hash_color_rgb(UInt64(v), seed)
            r_bits = reinterpret(UInt32, Float32(c.r))
            g_bits = reinterpret(UInt32, Float32(c.g))
            b_bits = reinterpret(UInt32, Float32(c.b))
            local_fp ⊻= UInt64(r_bits) | (UInt64(g_bits) << 24) | (UInt64(b_bits) << 48)
        end
        results[tid] = local_fp
    end
    
    parallel_fp = reduce(⊻, results)
    
    sequential_fp = UInt64(0)
    for v in 1:n_v
        c = hash_color_rgb(UInt64(v), seed)
        r_bits = reinterpret(UInt32, Float32(c.r))
        g_bits = reinterpret(UInt32, Float32(c.g))
        b_bits = reinterpret(UInt32, Float32(c.b))
        sequential_fp ⊻= UInt64(r_bits) | (UInt64(g_bits) << 24) | (UInt64(b_bits) << 48)
    end
    
    (spi_verified = parallel_fp == sequential_fp,
     parallel_fingerprint = parallel_fp,
     sequential_fingerprint = sequential_fp)
end

# ═══════════════════════════════════════════════════════════════════════════
# Associative Color Reduction
# ═══════════════════════════════════════════════════════════════════════════

"""
    associative_color_reduce(elements::Vector, op::Function; seed=GAY_SEED) -> NamedTuple

Reduce colored elements with an associative operation.
Verifies that (a ⊕ b) ⊕ c == a ⊕ (b ⊕ c) for colors.

# Example
```julia
elements = collect(1:100)
avg_color(c1, c2) = RGB{Float32}((c1.r+c2.r)/2, (c1.g+c2.g)/2, (c1.b+c2.b)/2)
result = associative_color_reduce(elements, avg_color)
```
"""
function associative_color_reduce(elements::Vector, op::Function; 
                                  seed::UInt64=GAY_SEED)
    colors = [hash_color_rgb(UInt64(hash(e)), seed) for e in elements]
    
    left_result = reduce((a, b) -> op(a, b), colors)
    right_result = foldr((a, b) -> op(a, b), colors)
    
    (left=left_result, right=right_result, 
     associative=(left_result == right_result))
end

function __init__()
    @info "Gay.jl CombinatorialSpaces extension loaded - DEC coloring available"
end

end # module GayCombinatorialSpacesExt
