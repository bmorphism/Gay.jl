# Gay.jl JSON3 Serialization
# ==========================
# Structured serialization for colors, invaders, and benchmark results

using JSON3
using StructTypes
using Colors: RGB
using Dates
using Printf

export gay_to_json, gay_from_json, save_colors_json, load_colors_json
export InvaderJSON, BenchmarkJSON, ColorPaletteJSON

# ═══════════════════════════════════════════════════════════════════════════
# JSON-Serializable Types
# ═══════════════════════════════════════════════════════════════════════════

"""
    ColorJSON

JSON-serializable color representation.
"""
struct ColorJSON
    r::Float64
    g::Float64
    b::Float64
    hex::String
end

StructTypes.StructType(::Type{ColorJSON}) = StructTypes.Struct()

function ColorJSON(c::RGB)
    r = Float64(c.r)
    g = Float64(c.g)
    b = Float64(c.b)
    hex = @sprintf("#%02X%02X%02X", 
        round(Int, clamp(r, 0, 1) * 255),
        round(Int, clamp(g, 0, 1) * 255),
        round(Int, clamp(b, 0, 1) * 255))
    ColorJSON(r, g, b, hex)
end

function Base.convert(::Type{RGB{Float64}}, c::ColorJSON)
    RGB{Float64}(c.r, c.g, c.b)
end

"""
    ColorPaletteJSON

JSON-serializable color palette with metadata.
"""
struct ColorPaletteJSON
    name::String
    seed::UInt64
    colorspace::String
    colors::Vector{ColorJSON}
    created_at::String
end

StructTypes.StructType(::Type{ColorPaletteJSON}) = StructTypes.Struct()

"""
    InvaderJSON

JSON-serializable GayInvader representation.
"""
struct InvaderJSON
    id::UInt64
    seed::UInt64
    source::ColorJSON
    deranged::ColorJSON
    world::ColorJSON
    derangement::Int
    tropical_t::Float64
    spin::Int
end

StructTypes.StructType(::Type{InvaderJSON}) = StructTypes.Struct()

function InvaderJSON(sim::NamedTuple)
    InvaderJSON(
        sim.id,
        sim.seed,
        ColorJSON(sim.source),
        ColorJSON(sim.deranged),
        ColorJSON(sim.world),
        sim.derangement,
        sim.tropical_t,
        sim.spin
    )
end

"""
    FleetJSON

JSON-serializable invader fleet.
"""
struct FleetJSON
    seed::UInt64
    count::Int
    invaders::Vector{InvaderJSON}
    magnetization::Float64
    created_at::String
end

StructTypes.StructType(::Type{FleetJSON}) = StructTypes.Struct()

"""
    BenchmarkJSON

JSON-serializable benchmark result.
"""
struct BenchmarkJSON
    name::String
    median_ns::Float64
    min_ns::Float64
    samples::Int
    allocs::Int
    bytes::Int
    rate::Float64
    backend::String
    timestamp::String
end

StructTypes.StructType(::Type{BenchmarkJSON}) = StructTypes.Struct()

# ═══════════════════════════════════════════════════════════════════════════
# Serialization Functions
# ═══════════════════════════════════════════════════════════════════════════

"""
    gay_to_json(obj) -> String

Serialize Gay.jl objects to JSON.
"""
function gay_to_json(c::RGB)
    JSON3.write(ColorJSON(c))
end

function gay_to_json(colors::Vector{<:RGB})
    JSON3.write([ColorJSON(c) for c in colors])
end

function gay_to_json(sim::NamedTuple)
    # Check if it's a teleportation simulation
    if haskey(sim, :source) && haskey(sim, :world)
        JSON3.write(InvaderJSON(sim))
    else
        JSON3.write(sim)
    end
end

"""
    gay_from_json(json::String, ::Type{T}) -> T

Deserialize JSON to Gay.jl objects.
"""
function gay_from_json(json::String, ::Type{RGB{Float64}})
    c = JSON3.read(json, ColorJSON)
    convert(RGB{Float64}, c)
end

function gay_from_json(json::String, ::Type{Vector{RGB{Float64}}})
    colors = JSON3.read(json, Vector{ColorJSON})
    [convert(RGB{Float64}, c) for c in colors]
end

function gay_from_json(json::String, ::Type{InvaderJSON})
    JSON3.read(json, InvaderJSON)
end

# ═══════════════════════════════════════════════════════════════════════════
# File I/O
# ═══════════════════════════════════════════════════════════════════════════

"""
    save_colors_json(filename, colors; name="palette", seed=0)

Save colors to a JSON file with metadata.
"""
function save_colors_json(
    filename::String, 
    colors::Vector{<:RGB}; 
    name::String="palette",
    seed::UInt64=UInt64(0),
    colorspace::String="sRGB"
)
    palette = ColorPaletteJSON(
        name,
        seed,
        colorspace,
        [ColorJSON(c) for c in colors],
        string(Dates.now())
    )
    
    open(filename, "w") do io
        JSON3.pretty(io, palette)
    end
    
    return filename
end

"""
    load_colors_json(filename) -> Vector{RGB{Float64}}

Load colors from a JSON file.
"""
function load_colors_json(filename::String)
    json = read(filename, String)
    palette = JSON3.read(json, ColorPaletteJSON)
    return [convert(RGB{Float64}, c) for c in palette.colors]
end

"""
    save_fleet_json(filename, fleet; seed=GAY_SEED)

Save an invader fleet to JSON.
"""
function save_fleet_json(filename::String, sims::Vector{<:NamedTuple}; seed::UInt64=GAY_SEED)
    invaders = [InvaderJSON(sim) for sim in sims]
    magnetization = sum(sim.spin for sim in sims) / length(sims)
    
    fleet = FleetJSON(
        seed,
        length(sims),
        invaders,
        magnetization,
        string(Dates.now())
    )
    
    open(filename, "w") do io
        JSON3.pretty(io, fleet)
    end
    
    return filename
end

"""
    save_benchmark_json(filename, results)

Save benchmark results to JSON.
"""
function save_benchmark_json(filename::String, results::Dict)
    all_benchmarks = BenchmarkJSON[]
    
    for (category, benches) in results
        for b in benches
            push!(all_benchmarks, BenchmarkJSON(
                b.name,
                b.median_ns,
                b.min_ns,
                b.samples,
                b.allocs,
                b.bytes,
                b.rate,
                string(typeof(get_backend())),
                string(Dates.now())
            ))
        end
    end
    
    open(filename, "w") do io
        JSON3.pretty(io, all_benchmarks)
    end
    
    return filename
end

# ═══════════════════════════════════════════════════════════════════════════
# JSONL (JSON Lines) for streaming
# ═══════════════════════════════════════════════════════════════════════════

"""
    stream_invaders_jsonl(filename, ids; seed=GAY_SEED)

Stream invaders to JSONL file (one JSON object per line).
Efficient for large fleets.
"""
function stream_invaders_jsonl(filename::String, ids::AbstractRange; seed::UInt64=GAY_SEED)
    open(filename, "w") do io
        for id in ids
            sim = simulate_teleportation(id, seed)
            inv = InvaderJSON(sim)
            JSON3.write(io, inv)
            println(io)  # newline for JSONL
        end
    end
    return filename
end

"""
    read_invaders_jsonl(filename) -> Vector{InvaderJSON}

Read invaders from JSONL file.
"""
function read_invaders_jsonl(filename::String)
    invaders = InvaderJSON[]
    open(filename) do io
        for line in eachline(io)
            isempty(strip(line)) && continue
            push!(invaders, JSON3.read(line, InvaderJSON))
        end
    end
    return invaders
end
