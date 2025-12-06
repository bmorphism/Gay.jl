# Plasmo.jl extension for Gay.jl
# Deterministic SPI-compliant coloring for OptiGraph optimization structures

module GayPlasmoExt

using Gay: hash_color_rgb, splitmix64, GAY_SEED
using Plasmo
using Colors: RGB, HSL, convert

export color_optigraph, color_optinodes, color_optiedges
export color_partition, solution_color_map
export color_subgraph, color_linking_constraints
export render_optigraph

# ═══════════════════════════════════════════════════════════════════════════
# OptiGraph Coloring
# ═══════════════════════════════════════════════════════════════════════════

"""
    color_optigraph(graph::OptiGraph; seed=GAY_SEED) -> NamedTuple

Color all nodes and edges of an OptiGraph with SPI colors.

Nodes get colors from their index.
Edges get colors from XOR of connected node indices.

# Example
```julia
graph = OptiGraph()
@optinode(graph, nodes[1:10])
colored = color_optigraph(graph)
```
"""
function color_optigraph(graph::OptiGraph; seed::UInt64=GAY_SEED)
    nodes = all_nodes(graph)
    edges = all_edges(graph)
    
    node_colors = Dict{OptiNode, RGB{Float32}}()
    for (i, node) in enumerate(nodes)
        node_colors[node] = hash_color_rgb(UInt64(i), seed)
    end
    
    node_index = Dict(node => i for (i, node) in enumerate(nodes))
    
    edge_colors = Dict{OptiEdge, RGB{Float32}}()
    for edge in edges
        connected = optinodes(edge)
        if length(connected) >= 2
            idx = reduce(⊻, [UInt64(node_index[n]) for n in connected])
            edge_colors[edge] = hash_color_rgb(idx, seed)
        end
    end
    
    (nodes=node_colors, edges=edge_colors)
end

"""
    color_optinodes(graph::OptiGraph; seed=GAY_SEED) -> Dict{OptiNode, RGB{Float32}}

Color just the nodes of an OptiGraph.
"""
function color_optinodes(graph::OptiGraph; seed::UInt64=GAY_SEED)
    nodes = all_nodes(graph)
    Dict(node => hash_color_rgb(UInt64(i), seed) for (i, node) in enumerate(nodes))
end

"""
    color_optiedges(graph::OptiGraph; seed=GAY_SEED) -> Dict{OptiEdge, RGB{Float32}}

Color just the edges of an OptiGraph.
"""
function color_optiedges(graph::OptiGraph; seed::UInt64=GAY_SEED)
    nodes = all_nodes(graph)
    node_index = Dict(node => i for (i, node) in enumerate(nodes))
    
    edges = all_edges(graph)
    result = Dict{OptiEdge, RGB{Float32}}()
    
    for edge in edges
        connected = optinodes(edge)
        if length(connected) >= 2
            idx = reduce(⊻, [UInt64(node_index[n]) for n in connected])
            result[edge] = hash_color_rgb(idx, seed)
        end
    end
    
    result
end

# ═══════════════════════════════════════════════════════════════════════════
# Partition Coloring
# ═══════════════════════════════════════════════════════════════════════════

"""
    color_partition(partition::Partition; seed=GAY_SEED) -> Dict{Int, RGB{Float32}}

Color partition blocks with distinct SPI colors.

# Example
```julia
partition = Partition(graph, node_membership_vector)
colors = color_partition(partition)
```
"""
function color_partition(partition::Partition; seed::UInt64=GAY_SEED)
    n_parts = num_partitions(partition)
    Dict(i => hash_color_rgb(UInt64(i), seed) for i in 1:n_parts)
end

"""
    color_subgraph(subgraph::OptiGraph, parent::OptiGraph; seed=GAY_SEED) -> NamedTuple

Color a subgraph with colors consistent with parent graph.
"""
function color_subgraph(subgraph::OptiGraph, parent::OptiGraph; seed::UInt64=GAY_SEED)
    parent_nodes = all_nodes(parent)
    parent_index = Dict(node => i for (i, node) in enumerate(parent_nodes))
    
    sub_nodes = all_nodes(subgraph)
    node_colors = Dict{OptiNode, RGB{Float32}}()
    
    for node in sub_nodes
        if haskey(parent_index, node)
            node_colors[node] = hash_color_rgb(UInt64(parent_index[node]), seed)
        else
            node_colors[node] = hash_color_rgb(UInt64(hash(node)), seed)
        end
    end
    
    (nodes=node_colors,)
end

# ═══════════════════════════════════════════════════════════════════════════
# Solution Value Coloring
# ═══════════════════════════════════════════════════════════════════════════

"""
    solution_color_map(graph::OptiGraph; seed=GAY_SEED) -> Dict{OptiNode, RGB{Float32}}

Color nodes based on their objective contribution.
Hue from node index (SPI), lightness from relative objective value.

# Example
```julia
optimize!(graph, optimizer)
colors = solution_color_map(graph)
```
"""
function solution_color_map(graph::OptiGraph; seed::UInt64=GAY_SEED)
    nodes = all_nodes(graph)
    
    obj_values = Float64[]
    for node in nodes
        try
            push!(obj_values, objective_value(node))
        catch
            push!(obj_values, 0.0)
        end
    end
    
    if isempty(obj_values) || all(iszero, obj_values)
        return color_optinodes(graph; seed)
    end
    
    vmin, vmax = extrema(obj_values)
    range_val = vmax - vmin
    range_val = range_val > 0 ? range_val : 1.0
    
    result = Dict{OptiNode, RGB{Float32}}()
    for (i, node) in enumerate(nodes)
        base_color = hash_color_rgb(UInt64(i), seed)
        base_hsl = convert(HSL, base_color)
        
        normalized = (obj_values[i] - vmin) / range_val
        lightness = Float32(0.2 + 0.6 * normalized)
        
        result[node] = convert(RGB{Float32}, HSL(base_hsl.h, 0.8f0, lightness))
    end
    
    result
end

# ═══════════════════════════════════════════════════════════════════════════
# Linking Constraint Coloring
# ═══════════════════════════════════════════════════════════════════════════

"""
    color_linking_constraints(graph::OptiGraph; seed=GAY_SEED) -> Dict{LinkConstraintRef, RGB{Float32}}

Color linking constraints based on nodes they connect.
"""
function color_linking_constraints(graph::OptiGraph; seed::UInt64=GAY_SEED)
    nodes = all_nodes(graph)
    node_index = Dict(node => i for (i, node) in enumerate(nodes))
    
    result = Dict{LinkConstraintRef, RGB{Float32}}()
    
    for edge in all_edges(graph)
        for lc in all_linkconstraints(edge)
            connected = optinodes(edge)
            if length(connected) >= 2
                idx = reduce(⊻, [UInt64(node_index[n]) for n in connected])
                result[lc] = hash_color_rgb(idx, seed)
            end
        end
    end
    
    result
end

# ═══════════════════════════════════════════════════════════════════════════
# Rendering
# ═══════════════════════════════════════════════════════════════════════════

"""
    render_optigraph(graph::OptiGraph; seed=GAY_SEED) -> String

Render an OptiGraph structure with ANSI colors.
"""
function render_optigraph(graph::OptiGraph; seed::UInt64=GAY_SEED)
    colors = color_optigraph(graph; seed)
    
    lines = String[]
    push!(lines, "OptiGraph: $(length(all_nodes(graph))) nodes, $(length(all_edges(graph))) edges")
    
    for (node, color) in colors.nodes
        r = round(Int, clamp(color.r, 0, 1) * 255)
        g = round(Int, clamp(color.g, 0, 1) * 255)
        b = round(Int, clamp(color.b, 0, 1) * 255)
        push!(lines, "\e[38;2;$(r);$(g);$(b)m█\e[0m $(node)")
    end
    
    join(lines, "\n")
end

function __init__()
    @info "Gay.jl Plasmo extension loaded - OptiGraph coloring available"
end

end # module GayPlasmoExt
