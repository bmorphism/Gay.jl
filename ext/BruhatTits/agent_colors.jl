# agent_colors.jl
module AgentColors
using Random

export ColoredAgent, spawn_agents, agent_read!, agent_write!, fingerprint, GAY_SEED

const GAY_SEED = UInt64(0x6761795f636f6c6f)

sm64(z) = let z = z + 0x9E3779B97F4A7C15
    z = (z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ⊻ (z >> 27)) * 0x94D049BB133111EB
    z ⊻ (z >> 31)
end

mutable struct ColoredAgent
    id::Int
    color::UInt64
    operations::Vector{Tuple{Symbol, UInt64}}  # (:read/:write, color_at_op)
end

function spawn_agents(n::Int, seed::UInt64=GAY_SEED)
    agents = ColoredAgent[]
    state = seed
    for i in 1:n
        state = sm64(state)
        push!(agents, ColoredAgent(i, state, []))
    end
    agents
end

function agent_read!(agent::ColoredAgent, path::String)
    op_color = sm64(agent.color ⊻ hash(path))
    push!(agent.operations, (:read, op_color))
    op_color
end

function agent_write!(agent::ColoredAgent, path::String)
    op_color = sm64(agent.color ⊻ hash(path))
    push!(agent.operations, (:write, op_color))
    op_color
end

function fingerprint(agents::Vector{ColoredAgent})
    reduce(⊻, a.color for a in agents)
end

end
