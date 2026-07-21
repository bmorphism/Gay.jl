#!/usr/bin/env julia

using Gay
using Catlab.CategoricalAlgebra: dom, codom
using Catlab.Graphs: Graph, path_graph
using Colors: hex
using StructuredDecompositions
using StructuredDecompositions.Decompositions: bags, adhesionSpans

extension_module = Base.get_extension(Gay, :GayStructuredDecompositionsExt)
extension_module === nothing && error("GayStructuredDecompositionsExt did not load")

decomposition = StrDecomp(path_graph(Graph, 4))
colored = extension_module.color_decomposition(decomposition; seed=Gay.GAY_SEED)

@assert length(colored.bags) == 3
@assert length(colored.adhesions) == 2
@assert hex.(colored.bags) == ["55B0E6", "C8A0C2", "FFA6C2"]

indexed_bags = bags(decomposition, true)
bag_positions = Dict(bag_key => i for (i, (bag_key, _)) in enumerate(indexed_bags))
shape_category = dom(decomposition.diagram)
endpoint_pairs = map(adhesionSpans(decomposition, true)) do (shape_span, _)
    @assert length(shape_span) == 2
    (bag_positions[codom(shape_category, shape_span[1])],
     bag_positions[codom(shape_category, shape_span[2])])
end

@assert endpoint_pairs == [(2, 1), (3, 2)]

expected_adhesions = map(endpoint_pairs) do (left, right)
    c1 = colored.bags[left]
    c2 = colored.bags[right]
    r = round(UInt8, clamp(c1.r, 0, 1) * 255) ⊻ round(UInt8, clamp(c2.r, 0, 1) * 255)
    g = round(UInt8, clamp(c1.g, 0, 1) * 255) ⊻ round(UInt8, clamp(c2.g, 0, 1) * 255)
    b = round(UInt8, clamp(c1.b, 0, 1) * 255) ⊻ round(UInt8, clamp(c2.b, 0, 1) * 255)
    uppercase(string(r, base=16, pad=2) * string(g, base=16, pad=2) * string(b, base=16, pad=2))
end

@assert hex.(colored.adhesions) == expected_adhesions
@assert colored.adhesions[1] != colored.adhesions[2]

println((valid=true,
         bags=hex.(colored.bags),
         adhesions=hex.(colored.adhesions),
         endpoint_pairs=endpoint_pairs))
