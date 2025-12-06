# # Plasmo.jl Optimization Graph Integration
#
# Gay.jl provides SPI-compliant coloring for OptiGraph structures,
# enabling deterministic visualization of decomposed optimization problems.

# ## Setup

using Gay
using Plasmo
using Colors

# ## Basic OptiGraph Coloring
#
# Create a simple optimization graph and color its structure:

graph = OptiGraph()

# Add nodes with variables
@optinode(graph, nodes[1:5])
for (i, node) in enumerate(nodes)
    @variable(node, x >= 0)
    @objective(node, Min, i * node[:x])
end

# Add linking constraints (edges)
@linkconstraint(graph, nodes[1][:x] + nodes[2][:x] >= 1)
@linkconstraint(graph, nodes[2][:x] + nodes[3][:x] >= 1)
@linkconstraint(graph, nodes[3][:x] + nodes[4][:x] >= 1)
@linkconstraint(graph, nodes[4][:x] + nodes[5][:x] >= 1)

# Color the entire graph
colors = color_optigraph(graph; seed=GAY_SEED)

println("=== OptiGraph Coloring ===")
println("Nodes colored: $(length(colors.nodes))")
println("Edges colored: $(length(colors.edges))")

for (node, c) in colors.nodes
    println("  Node: RGB($(round(c.r, digits=2)), $(round(c.g, digits=2)), $(round(c.b, digits=2)))")
end

# ## Partition Coloring
#
# Color graph partitions for decomposition visualization:

membership = [1, 1, 2, 2, 3]  # 3 partitions
partition = Partition(graph, membership)

part_colors = color_partition(partition; seed=GAY_SEED)

println("\n=== Partition Colors ===")
for (part_id, c) in sort(collect(part_colors))
    println("  Partition $part_id: RGB($(round(c.r, digits=2)), $(round(c.g, digits=2)), $(round(c.b, digits=2)))")
end

# ## Solution Visualization
#
# After solving, color nodes by their objective contribution:
# (Hue from position, lightness from value)

# Solve (would need optimizer in practice)
# optimize!(graph, Ipopt.Optimizer)

# For demo, show the coloring function:
println("\n=== Solution Color Map ===")
println("solution_color_map(graph) colors nodes by objective value")
println("- Base hue: from node index (SPI)")
println("- Lightness: from normalized objective value")

# ## Linking Constraint Colors
#
# Constraints inherit colors from connected nodes:

lc_colors = color_linking_constraints(graph; seed=GAY_SEED)
println("\n=== Linking Constraints ===")
println("$(length(lc_colors)) linking constraints colored")

# ## Hierarchical Graphs
#
# Subgraphs maintain color consistency with parent:

subgraph = OptiGraph()
@optinode(subgraph, sub_nodes[1:3])

# Coloring preserves parent index relationships
sub_colors = color_subgraph(subgraph, graph; seed=GAY_SEED)

println("\n=== Subgraph Coloring ===")
println("Subgraph nodes use parent indices when available")

# ## Rendering

rendered = render_optigraph(graph; seed=GAY_SEED)
println("\n$rendered")

# ## Key Properties
#
# 1. **Node colors**: Index-based hashing for consistent assignment
# 2. **Edge colors**: XOR of connected node indices (associative)
# 3. **Partition colors**: Distinct colors per partition block
# 4. **Solution maps**: Value-weighted lightness with SPI base hue
