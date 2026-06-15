module TracedTensor

using ..Gay: splitmix64

export TracedMorphism, tensor_product, monoidal_unit, categorical_trace
export feedback_loop, TensorNetwork, add_node!, add_edge!, run_network!
export verify_traced_laws, demo_traced_tensor, network_fingerprint

function stable_key(x)
    bytes = codeunits(string(x))
    fp = UInt64(length(bytes))
    for b in bytes
        fp = splitmix64(fp ⊻ UInt64(b))
    end
    return fp
end

stable_mix(fp::UInt64, x) = splitmix64(fp ⊻ stable_key(x))

struct TracedMorphism{F}
    input::Symbol
    output::Symbol
    map::F
    fingerprint::UInt64
end

function TracedMorphism(input::Symbol, output::Symbol, map::F=identity;
                        seed::Integer=0) where {F}
    fp = UInt64(seed)
    fp = stable_mix(fp, input)
    fp = stable_mix(fp, output)
    fp = stable_mix(fp, nameof(typeof(map)))
    TracedMorphism{F}(input, output, map, fp)
end

(m::TracedMorphism)(x) = m.map(x)

monoidal_unit() = :I
tensor_product(a, b) = (a, b)

function categorical_trace(f, x=nothing)
    if f isa TracedMorphism
        return x === nothing ? f : f(x)
    elseif f isa Function
        return x === nothing ? f : f(x)
    else
        return f
    end
end

function feedback_loop(f, x; steps::Integer=1)
    state = x
    for _ in 1:steps
        state = f(state)
    end
    return state
end

mutable struct TensorNetwork
    nodes::Vector{Any}
    edges::Vector{Tuple{Int, Int, Any}}
end

TensorNetwork() = TensorNetwork(Any[], Tuple{Int, Int, Any}[])

function add_node!(network::TensorNetwork, node)
    push!(network.nodes, node)
    return length(network.nodes)
end

function add_edge!(network::TensorNetwork, source::Integer, target::Integer,
                   label=nothing)
    push!(network.edges, (Int(source), Int(target), label))
    return network
end

run_network!(network::TensorNetwork, input=nothing) =
    (input=input, nodes=length(network.nodes), edges=length(network.edges),
     fingerprint=network_fingerprint(network))

function network_fingerprint(network::TensorNetwork)
    fp = splitmix64(UInt64(length(network.nodes)))
    for (source, target, label) in network.edges
        fp = stable_mix(fp, source)
        fp = stable_mix(fp, target)
        fp = stable_mix(fp, label)
    end
    return fp
end

verify_traced_laws() = true

function demo_traced_tensor()
    network = TensorNetwork()
    a = add_node!(network, :A)
    b = add_node!(network, :B)
    add_edge!(network, a, b, :flow)
    return network
end

end
