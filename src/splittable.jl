# Deterministic splittable random color generation
# Inspired by Pigeons.jl's Strong Parallelism Invariance (SPI)

using SplittableRandoms: SplittableRandom, split

export GayRNG, gay_seed!, gay_split, next_color, next_colors, next_palette

"""
    GayRNG

A splittable random number generator for deterministic color generation.
Each color operation splits the RNG to ensure reproducibility regardless
of execution order (Strong Parallelism Invariance).

The RNG state tracks an invocation counter to generate a unique deterministic
stream for each call, enabling reproducible sequences even across sessions.
"""
mutable struct GayRNG
    root::SplittableRandom
    current::SplittableRandom
    invocation::UInt64
    seed::UInt64
end

# Global RNG instance - default seed based on package name hash
const GAY_SEED = UInt64(0x6761795f636f6c6f)  # "gay_colo" as bytes
const GLOBAL_GAY_RNG = Ref{GayRNG}()

"""
    GayRNG(seed::Integer=GAY_SEED)

Create a new GayRNG with the given seed.
"""
function GayRNG(seed::Integer=GAY_SEED)
    root = SplittableRandom(UInt64(seed))
    current = split(root)
    GayRNG(root, current, UInt64(0), UInt64(seed))
end

"""
    gay_seed!(seed::Integer)

Reset the global Gay RNG with a new seed.
All subsequent color generations will be deterministic from this seed.
"""
function gay_seed!(seed::Integer)
    GLOBAL_GAY_RNG[] = GayRNG(seed)
    return seed
end

"""
    gay_rng()

Get the global Gay RNG, initializing if needed.
"""
function gay_rng()
    if !isassigned(GLOBAL_GAY_RNG)
        GLOBAL_GAY_RNG[] = GayRNG()
    end
    return GLOBAL_GAY_RNG[]
end

"""
    gay_split(gr::GayRNG=gay_rng())

Split the RNG for a new independent stream.
Increments invocation counter for tracking.
"""
function gay_split(gr::GayRNG=gay_rng())
    gr.invocation += 1
    gr.current = split(gr.current)
    return gr.current
end

"""
    gay_split(n::Integer, gr::GayRNG=gay_rng())

Get n independent RNG splits as a vector.
"""
function gay_split(n::Integer, gr::GayRNG=gay_rng())
    return [gay_split(gr) for _ in 1:n]
end

# ═══════════════════════════════════════════════════════════════════════════
# Deterministic color generation using splittable RNG
# ═══════════════════════════════════════════════════════════════════════════

"""
    next_color(cs::ColorSpace=SRGB(); gr::GayRNG=gay_rng())

Generate the next deterministic random color.
Each call splits the RNG for reproducibility.
"""
function next_color(cs::ColorSpace=SRGB(); gr::GayRNG=gay_rng())
    rng = gay_split(gr)
    return random_color(cs; rng=rng)
end

"""
    next_colors(n::Int, cs::ColorSpace=SRGB(); gr::GayRNG=gay_rng())

Generate n deterministic random colors.
"""
function next_colors(n::Int, cs::ColorSpace=SRGB(); gr::GayRNG=gay_rng())
    rng = gay_split(gr)
    return random_colors(n, cs; rng=rng)
end

"""
    next_palette(n::Int, cs::ColorSpace=SRGB(); 
                 min_distance::Float64=30.0, gr::GayRNG=gay_rng())

Generate n deterministic visually distinct colors.
"""
function next_palette(n::Int, cs::ColorSpace=SRGB();
                      min_distance::Float64=30.0, gr::GayRNG=gay_rng())
    rng = gay_split(gr)
    return random_palette(n, cs; min_distance=min_distance, rng=rng)
end

# ═══════════════════════════════════════════════════════════════════════════
# Invocation-indexed color access (like Pigeons explorer indexing)
# ═══════════════════════════════════════════════════════════════════════════

"""
    color_at(index::Integer, cs::ColorSpace=SRGB(); seed::Integer=GAY_SEED)

Get the color at a specific invocation index.
This allows random access to the deterministic color sequence.

# Example
```julia
# These will always return the same colors for the same indices
c1 = color_at(1)
c42 = color_at(42)
c1_again = color_at(1)  # Same as c1
```
"""
function color_at(index::Integer, cs::ColorSpace=SRGB(); seed::Integer=GAY_SEED)
    # Create a fresh RNG from seed
    root = SplittableRandom(UInt64(seed))
    current = root
    
    # Split to the desired index
    for _ in 1:index
        current = split(current)
    end
    
    # Generate color at this index
    return random_color(cs; rng=current)
end

"""
    colors_at(indices::AbstractVector{<:Integer}, cs::ColorSpace=SRGB(); 
              seed::Integer=GAY_SEED)

Get colors at specific invocation indices.
"""
function colors_at(indices::AbstractVector{<:Integer}, cs::ColorSpace=SRGB();
                   seed::Integer=GAY_SEED)
    return [color_at(i, cs; seed=seed) for i in indices]
end

"""
    palette_at(index::Integer, n::Int, cs::ColorSpace=SRGB();
               min_distance::Float64=30.0, seed::Integer=GAY_SEED)

Get a palette at a specific invocation index.
"""
function palette_at(index::Integer, n::Int, cs::ColorSpace=SRGB();
                    min_distance::Float64=30.0, seed::Integer=GAY_SEED)
    root = SplittableRandom(UInt64(seed))
    current = root
    for _ in 1:index
        current = split(current)
    end
    return random_palette(n, cs; min_distance=min_distance, rng=current)
end
