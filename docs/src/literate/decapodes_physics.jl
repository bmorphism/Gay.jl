# # Decapodes.jl Physics Simulation Integration
#
# Gay.jl provides SPI-compliant coloring for physics simulations
# using discrete exterior calculus (DEC) on simplicial meshes.

# ## Setup

using Gay
using Decapodes
using CombinatorialSpaces
using Colors

# ## Mesh Coloring
#
# Color mesh elements with deterministic SPI colors:

# Create a simple triangulated mesh
mesh = DeltaSet2D()
add_vertices!(mesh, 9)

# Create a 3x3 grid of triangles
glue_triangle!(mesh, 1, 2, 4)
glue_triangle!(mesh, 2, 5, 4)
glue_triangle!(mesh, 2, 3, 5)
glue_triangle!(mesh, 3, 6, 5)
glue_triangle!(mesh, 4, 5, 7)
glue_triangle!(mesh, 5, 8, 7)
glue_triangle!(mesh, 5, 6, 8)
glue_triangle!(mesh, 6, 9, 8)

colors = color_mesh(mesh; seed=GAY_SEED)

println("=== Mesh Coloring ===")
println("Vertices: $(length(colors.vertices))")
println("Edges: $(length(colors.edges))")
println("Triangles: $(length(colors.triangles))")

println("\nSample vertex colors:")
for i in 1:min(4, length(colors.vertices))
    c = colors.vertices[i]
    println("  v$i: RGB($(round(c.r, digits=2)), $(round(c.g, digits=2)), $(round(c.b, digits=2)))")
end

# ## Field Coloring
#
# Color discrete forms (0-forms on vertices, 1-forms on edges):

# Simulate a scalar field (e.g., temperature)
temperature = [0.0, 0.2, 0.4, 0.3, 0.5, 0.7, 0.6, 0.8, 1.0]

field_colors = color_field(mesh, temperature; seed=GAY_SEED, form=0)

println("\n=== 0-Form Field Coloring ===")
println("Temperature field colored with:")
println("- Hue: from vertex index (SPI)")
println("- Lightness: from field value (0.0 → dark, 1.0 → bright)")

for i in 1:3
    c = field_colors[i]
    println("  v$i (T=$(temperature[i])): RGB($(round(c.r, digits=2)), $(round(c.g, digits=2)), $(round(c.b, digits=2)))")
end

# ## DEC Operator Coloring
#
# Discrete exterior calculus operators with semantic hues:

println("\n=== DEC Operator Colors ===")

operators = [:d, :δ, :Δ, :⋆, :♭, :♯]
for op in operators
    colorer = color_operator(op; seed=GAY_SEED)
    c = colorer(1, 1)  # Sample diagonal entry
    println("  $op: RGB($(round(c.r, digits=2)), $(round(c.g, digits=2)), $(round(c.b, digits=2)))")
end

# ## Decapode Structure Coloring
#
# Color physics equation structures:

# Define heat equation
HeatEquation = @decapode begin
    C::Form0
    ∂ₜ(C) == Δ(C)
end

deca_colors = color_decapode(HeatEquation; seed=GAY_SEED)

println("\n=== Decapode Structure ===")
println("Variables: $(length(deca_colors.variables))")
println("Operators: $(length(deca_colors.operators))")
println("Summations: $(length(deca_colors.summations))")

# ## Simulation State Coloring
#
# Color all fields in a simulation state simultaneously:

state = (
    C = temperature,  # 0-form
    # V would be 1-form on edges, etc.
)

state_colors = color_simulation_state(mesh, state; seed=GAY_SEED)

println("\n=== Simulation State ===")
for (name, colors) in state_colors
    println("  $name: $(length(colors)) colored values")
end

# ## Complex Field Coloring
#
# For wave equations: magnitude → lightness, phase → hue shift

magnitude = [0.5, 0.8, 1.0, 0.3, 0.6, 0.9, 0.4, 0.7, 0.2]
phase = [0.0, π/4, π/2, 3π/4, π, 5π/4, 3π/2, 7π/4, 2π]

complex_colors = color_field(mesh, magnitude, phase; seed=GAY_SEED)

println("\n=== Complex Field Coloring ===")
println("Magnitude controls lightness, phase shifts hue")
for i in 1:3
    c = complex_colors[i]
    println("  v$i (|ψ|=$(round(magnitude[i], digits=2)), φ=$(round(phase[i]/π, digits=2))π): RGB($(round(c.r, digits=2)), $(round(c.g, digits=2)), $(round(c.b, digits=2)))")
end

# ## Key Properties
#
# 1. **Mesh elements**: XOR-based hashing for edges/triangles (associative)
# 2. **Field values**: Normalized to lightness [0.15, 0.85]
# 3. **Operators**: Semantic hue families (d=orange, Δ=magenta, etc.)
# 4. **Phase encoding**: Hue shift for complex fields
# 5. **SPI guarantee**: Same mesh + seed = same colors always
