# SPI Tuning Parameters
# ====================
#
# Configurable parameters for the 6-layer SPI verification tower.

module Tuning

export SPIConfig, default_config, with_config
export @tuned

using ..Gay: GAY_SEED

"""
Configuration for SPI verification system.
"""
Base.@kwdef mutable struct SPIConfig
    # Layer 0: Concept Tensor
    tensor_size::Int = 69
    
    # Layer 1: Exponential X^X
    rotation_bits::Int = 64
    max_morphisms::Int = 1000
    
    # Layer 2: Higher (X^X)^(X^X)
    max_iteration_power::Int = 8
    orbit_sample_size::Int = 10
    
    # Layer 3: Traced Monoidal
    feedback_depth::Int = 20
    trace_iterations::Int = 16
    
    # Layer 4: Tensor Network
    max_nodes::Int = 100
    max_edges::Int = 200
    
    # Layer 5: Thread Findings
    thread_count::Int = 69
    lazy_materialization::Bool = true
    
    # Global
    seed::UInt64 = GAY_SEED
    parallel::Bool = true
    verbose::Bool = false
end

const CURRENT_CONFIG = Ref(SPIConfig())

"""
    default_config() -> SPIConfig

Return the default configuration.
"""
default_config() = SPIConfig()

"""
    current_config() -> SPIConfig

Get the current active configuration.
"""
current_config() = CURRENT_CONFIG[]

"""
    set_config!(config::SPIConfig)

Set the current active configuration.
"""
function set_config!(config::SPIConfig)
    CURRENT_CONFIG[] = config
end

"""
    with_config(f, config::SPIConfig)

Execute function `f` with temporary configuration.
"""
function with_config(f, config::SPIConfig)
    old = CURRENT_CONFIG[]
    try
        CURRENT_CONFIG[] = config
        f()
    finally
        CURRENT_CONFIG[] = old
    end
end

"""
    with_config(f; kwargs...)

Execute with modified configuration.
"""
function with_config(f; kwargs...)
    config = SPIConfig(; kwargs...)
    with_config(f, config)
end

# Preset configurations
const PRESETS = Dict{Symbol, SPIConfig}(
    :minimal => SPIConfig(
        tensor_size = 11,
        thread_count = 10,
        feedback_depth = 5,
        trace_iterations = 4,
        max_iteration_power = 4,
    ),
    :default => SPIConfig(),
    :full => SPIConfig(
        tensor_size = 69,
        thread_count = 69,
        feedback_depth = 50,
        trace_iterations = 32,
        max_iteration_power = 16,
    ),
    :benchmark => SPIConfig(
        tensor_size = 69,
        thread_count = 100,
        feedback_depth = 100,
        trace_iterations = 64,
        max_iteration_power = 32,
        parallel = true,
    ),
)

"""
    preset(name::Symbol) -> SPIConfig

Get a preset configuration.
Available: :minimal, :default, :full, :benchmark
"""
function preset(name::Symbol)
    haskey(PRESETS, name) || error("Unknown preset: $name. Available: $(keys(PRESETS))")
    deepcopy(PRESETS[name])
end

"""
    tune!(; kwargs...)

Modify current configuration in-place.
"""
function tune!(; kwargs...)
    config = current_config()
    for (k, v) in kwargs
        setfield!(config, k, v)
    end
    config
end

"""
    show_config(io::IO, config::SPIConfig)

Pretty-print configuration.
"""
function Base.show(io::IO, ::MIME"text/plain", config::SPIConfig)
    println(io, "SPIConfig:")
    println(io, "  Layer 0 (Concept Tensor):")
    println(io, "    tensor_size = $(config.tensor_size)³ = $(config.tensor_size^3)")
    println(io, "  Layer 1 (Exponential X^X):")
    println(io, "    rotation_bits = $(config.rotation_bits)")
    println(io, "    max_morphisms = $(config.max_morphisms)")
    println(io, "  Layer 2 (Higher (X^X)^(X^X)):")
    println(io, "    max_iteration_power = $(config.max_iteration_power)")
    println(io, "    orbit_sample_size = $(config.orbit_sample_size)")
    println(io, "  Layer 3 (Traced Monoidal):")
    println(io, "    feedback_depth = $(config.feedback_depth)")
    println(io, "    trace_iterations = $(config.trace_iterations)")
    println(io, "  Layer 4 (Tensor Network):")
    println(io, "    max_nodes = $(config.max_nodes)")
    println(io, "    max_edges = $(config.max_edges)")
    println(io, "  Layer 5 (Thread Findings):")
    println(io, "    thread_count = $(config.thread_count)")
    println(io, "    lazy_materialization = $(config.lazy_materialization)")
    println(io, "  Global:")
    println(io, "    seed = 0x$(string(config.seed, base=16, pad=16))")
    println(io, "    parallel = $(config.parallel)")
    println(io, "    verbose = $(config.verbose)")
end

export preset, tune!, current_config, set_config!

end # module Tuning
