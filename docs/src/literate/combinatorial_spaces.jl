# # CombinatorialSpaces.jl Integration
# 
# This example demonstrates Gay.jl's SPI-compliant coloring for discrete 
# exterior calculus (DEC) structures from CombinatorialSpaces.jl.

# ## Setup
# 
# Load Gay.jl and CombinatorialSpaces.jl to enable the extension:

using Gay
using CombinatorialSpaces
using Colors

# ## Strong Parallelism Invariance (SPI)
#
# The key guarantee: **identical (seed, index) → identical color**,
# regardless of execution order, thread count, or hardware.
#
# This enables:
# - CPU/GPU result verification via XOR fingerprinting
# - Reproducible scientific visualization
# - Deterministic parallel coloring

# ## Example 1: 1D Delta Set (Path Graph)
#
# Create a simple 1D simplicial set:

ds1d = DeltaSet1D()
add_vertices!(ds1d, 5)
add_edge!(ds1d, 2, 1)  # E1: 2→1
add_edge!(ds1d, 3, 2)  # E2: 3→2
add_edge!(ds1d, 4, 3)  # E3: 4→3
add_edge!(ds1d, 5, 4)  # E4: 5→4

# Color with SPI:

colors_1d = color_deltaset_1d(ds1d; seed=GAY_SEED)

println("Vertex colors:")
for (i, c) in enumerate(colors_1d.vertices)
    println("  v$i: RGB($(round(c.r, digits=3)), $(round(c.g, digits=3)), $(round(c.b, digits=3)))")
end

println("\nEdge colors:")
for (i, c) in enumerate(colors_1d.edges)
    println("  e$i: RGB($(round(c.r, digits=3)), $(round(c.g, digits=3)), $(round(c.b, digits=3)))")
end

# ## Example 2: 2D Delta Set (Triangulated Square)
#
# Create a triangulated square:

ds2d = DeltaSet2D()
add_vertices!(ds2d, 4)
glue_triangle!(ds2d, 1, 2, 3)
glue_triangle!(ds2d, 1, 3, 4)

# Color all simplex dimensions:

colors_2d = color_deltaset_2d(ds2d; seed=GAY_SEED)

println("\nTriangle colors:")
for (i, c) in enumerate(colors_2d.triangles)
    println("  t$i: RGB($(round(c.r, digits=3)), $(round(c.g, digits=3)), $(round(c.b, digits=3)))")
end

# ## Example 3: SPI Verification
#
# Verify that parallel and sequential coloring produce identical results:

println("\n=== SPI Verification ===")

# Create a larger mesh for meaningful parallel test
ds_large = DeltaSet2D()
add_vertices!(ds_large, 100)
for i in 1:98
    glue_triangle!(ds_large, i, i+1, i+2)
end

result = spi_verify_parallel(ds_large, GAY_SEED)

println("SPI Verified: $(result.spi_verified)")
println("Parallel fingerprint:   0x$(string(result.parallel_fingerprint, base=16))")
println("Sequential fingerprint: 0x$(string(result.sequential_fingerprint, base=16))")

# ## Example 4: DEC Operator Coloring
#
# Color discrete exterior calculus operators:

println("\n=== DEC Operator Colors ===")

for op in [:d, :δ, :Δ, :⋆, :♭, :♯]
    colorer = color_dec_operator(op; seed=GAY_SEED)
    c = colorer(1, 2)  # Sample entry
    println("  $op: RGB($(round(c.r, digits=2)), $(round(c.g, digits=2)), $(round(c.b, digits=2)))")
end

# ## Example 5: Associative Color Reduction
#
# XOR-based fingerprinting is associative, enabling correct parallel reductions:

elements = collect(1:20)

# XOR of colors is associative
xor_color(c1, c2) = RGB{Float32}(
    reinterpret(Float32, reinterpret(UInt32, Float32(c1.r)) ⊻ reinterpret(UInt32, Float32(c2.r))),
    reinterpret(Float32, reinterpret(UInt32, Float32(c1.g)) ⊻ reinterpret(UInt32, Float32(c2.g))),
    reinterpret(Float32, reinterpret(UInt32, Float32(c1.b)) ⊻ reinterpret(UInt32, Float32(c2.b)))
)

result = associative_color_reduce(elements, xor_color; seed=GAY_SEED)
println("\nAssociative XOR reduction: $(result.associative)")

# ## Key Takeaways
#
# 1. **O(1) per-element coloring** via `splitmix64` hash
# 2. **XOR fingerprinting** for CPU/GPU verification
# 3. **Associative reductions** enable correct parallel results
# 4. **Same seed + index = same color** always (SPI guarantee)
