module Sentinel

export ColorViolation, validate_agent, run_sentinels

struct ColorViolation
    agent_id::Int
    operation::Symbol
    expected_color::UInt64
    actual_color::UInt64
    timestamp::Float64
end

mutable struct SentinelState
    violations::Vector{ColorViolation}
    killed_agents::Vector{Int}
    fingerprint::UInt64
end

SentinelState() = SentinelState(ColorViolation[], Int[], UInt64(0))

function validate_agent(sentinel::SentinelState, agent_id::Int, 
                        op::Symbol, color::UInt64, expected::UInt64)
    if color != expected
        violation = ColorViolation(agent_id, op, expected, color, time())
        push!(sentinel.violations, violation)
        push!(sentinel.killed_agents, agent_id)
        sentinel.fingerprint ⊻= color
        return false
    end
    sentinel.fingerprint ⊻= color
    true
end

function run_sentinels(n_sentinels::Int=3)
    sentinels = [SentinelState() for _ in 1:n_sentinels]
    # Each sentinel monitors 1/3 of operations
    sentinels
end

end
