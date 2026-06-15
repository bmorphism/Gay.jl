module SPIOrchestrator

using ..Gay: GAY_SEED, color_at, splitmix64

export SPIWorld, spi_world, spi_fingerprint, spi_colors, spi_verified, spi_metrics
export spi_next_color, spi_fast_fingerprint
export SPIOrchestrator_t, SPIOrchestratorConfig, SPIOrchestratorState
export HierarchicalSplit, FractalAgent, SentinelNetwork
export spawn_hierarchy!, execute_hierarchy!
export SPIColorPipeline, SPIStreamingResult, spi_run_pipeline!, spi_pipeline_throughput
export SPIChain, spi_chain_fingerprint!, spi_verify_chain!
export spi_orchestrated_walk, SPIOrchestratorMetrics, spi_collect_metrics

struct SPIWorld
    seed::UInt64
    n::Int
    colors::Vector{Any}
    fingerprint::UInt64
    verified::Bool
end

struct SPIOrchestratorConfig
    seed::UInt64
    width::Int
    depth::Int
end

mutable struct SPIOrchestratorState
    config::SPIOrchestratorConfig
    agents::Vector{Any}
    completed::Bool
end

mutable struct SPIOrchestrator_t
    state::SPIOrchestratorState
end

struct HierarchicalSplit
    level::Int
    arity::Int
end

struct FractalAgent
    id::Int
    trit::Int
    seed::UInt64
end

struct SentinelNetwork
    agents::Vector{FractalAgent}
end

struct SPIColorPipeline
    seed::UInt64
    n::Int
end

struct SPIStreamingResult
    colors::Vector{Any}
    fingerprint::UInt64
    throughput::Float64
end

mutable struct SPIChain
    seed::UInt64
    n::Int
    fingerprint::UInt64
    verified::Bool
end

struct SPIOrchestratorMetrics
    n::Int
    fingerprint::UInt64
    verified::Bool
end

function color_key(c)
    r = round(Int, clamp(c.r, 0, 1) * 255)
    g = round(Int, clamp(c.g, 0, 1) * 255)
    b = round(Int, clamp(c.b, 0, 1) * 255)
    return (r, g, b)
end

stable_mix(fp::UInt64, x::Integer) = splitmix64(fp ⊻ UInt64(x))

function spi_fast_fingerprint(seed::Integer=GAY_SEED, n::Integer=12)
    fp = UInt64(0)
    for i in 1:Int(n)
        r, g, b = color_key(color_at(i; seed=seed))
        fp = stable_mix(fp, seed)
        fp = stable_mix(fp, i)
        fp = stable_mix(fp, r)
        fp = stable_mix(fp, g)
        fp = stable_mix(fp, b)
    end
    return fp
end

function spi_world(; seed::Integer=GAY_SEED, n::Integer=12)
    colors = Any[color_at(i; seed=seed) for i in 1:Int(n)]
    fp = spi_fast_fingerprint(seed, n)
    return SPIWorld(UInt64(seed), Int(n), colors, fp, true)
end

spi_fingerprint(world::SPIWorld) = world.fingerprint
spi_colors(world::SPIWorld) = world.colors
spi_verified(world::SPIWorld) = world.verified
spi_metrics(world::SPIWorld) =
    Dict(:n => world.n, :fingerprint => world.fingerprint, :verified => world.verified)
spi_next_color(world::SPIWorld, index::Integer=world.n + 1) =
    color_at(index; seed=world.seed)

function spawn_hierarchy!(state::SPIOrchestratorState, n::Integer=3)
    empty!(state.agents)
    for i in 1:Int(n)
        push!(state.agents, FractalAgent(i, mod(i - 2, 3) - 1, state.config.seed + UInt64(i)))
    end
    return SentinelNetwork(FractalAgent[a for a in state.agents])
end

function execute_hierarchy!(state::SPIOrchestratorState)
    isempty(state.agents) && spawn_hierarchy!(state, state.config.width)
    state.completed = true
    return state
end

function spi_run_pipeline!(pipeline::SPIColorPipeline)
    world = spi_world(seed=pipeline.seed, n=pipeline.n)
    throughput = pipeline.n == 0 ? 0.0 : Float64(pipeline.n)
    return SPIStreamingResult(world.colors, world.fingerprint, throughput)
end

spi_pipeline_throughput(result::SPIStreamingResult) = result.throughput

function spi_chain_fingerprint!(chain::SPIChain)
    chain.fingerprint = spi_fast_fingerprint(chain.seed, chain.n)
    return chain.fingerprint
end

function spi_verify_chain!(chain::SPIChain)
    chain.verified = chain.fingerprint == spi_fast_fingerprint(chain.seed, chain.n)
    return chain.verified
end

spi_orchestrated_walk(n::Integer=12; seed::Integer=GAY_SEED) =
    spi_world(seed=seed, n=n)

spi_collect_metrics(world::SPIWorld) =
    SPIOrchestratorMetrics(world.n, world.fingerprint, world.verified)

end
