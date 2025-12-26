# Universal Color View Protocol
# Trait-based interface for any system to integrate Gay.jl deterministic colors

using Colors: RGB, HSL

export Colorable, colorize, HasColorSeed, color_seed
export ColorView, @colorable
export SPIColorable, spi_color, @verify_spi

# ═══════════════════════════════════════════════════════════════════════════
# Strong Parallelism Invariance (SPI) Protocol
# ═══════════════════════════════════════════════════════════════════════════

"""
    SPIColorable

Abstract type for objects that support SPI-compliant coloring.
The SPI guarantee: `spi_color(x, s) == spi_color(x, s)` always, regardless of
execution order, thread count, or hardware.

Implement `spi_color(x::OurType, seed::UInt64)::RGB{Float32}` for custom types.
"""
abstract type SPIColorable end

"""
    spi_color(x::SPIColorable, seed::UInt64) -> RGB{Float32}

SPI-compliant coloring. Must satisfy:
- Deterministic: `spi_color(x, s) == spi_color(x, s)`
- Order-independent: `parallel_map(x -> spi_color(x, s), xs) == sequential_map(...)`
"""
function spi_color end

"""
    @verify_spi expr seed [n_trials=100]

Verify that an expression produces identical results across multiple evaluations.
Throws AssertionError if SPI is violated.

# Example
```julia
@verify_spi hash_color_rgb(42, seed) seed 100
```
"""
macro verify_spi(expr, seed, n_trials=100)
    quote
        local s = $(esc(seed))
        local results = [$(esc(expr)) for _ in 1:$(esc(n_trials))]
        @assert all(r == results[1] for r in results) "SPI violation: results not identical"
        results[1]
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# Colorable Trait
# ═══════════════════════════════════════════════════════════════════════════

"""
    Colorable

Trait for types that can receive deterministic colors.
Implement `colorize(x::T, seed::UInt64)::RGB{Float32}` for custom types.
"""
abstract type Colorable end

"""
    HasColorSeed

Trait for types that carry their own color seed.
Implement `color_seed(x::T)::UInt64` to return the seed.
"""
abstract type HasColorSeed end

"""
    colorize(x, seed::UInt64=GAY_SEED) -> RGB{Float32}

Get a deterministic color for any object. Default implementation uses hash.
Override for domain-specific coloring strategies.
"""
function colorize(x, seed::UInt64=GAY_SEED)
    idx = UInt64(hash(x))
    return hash_color_rgb(idx, seed)
end

function colorize(x::Integer, seed::UInt64=GAY_SEED)
    return hash_color_rgb(UInt64(x), seed)
end

"""
    color_seed(x::HasColorSeed) -> UInt64

Get the color seed associated with an object.
"""
function color_seed end

# ═══════════════════════════════════════════════════════════════════════════
# ColorView Wrapper
# ═══════════════════════════════════════════════════════════════════════════

"""
    ColorView{T}

Wrapper that provides a colored indexed view of any collection.
Each element gets a deterministic color based on its index.

# Example
```julia
items = ["apple", "banana", "cherry"]
for (item, color, idx) in ColorView(items, GAY_SEED)
    println("\$item is colored \$(color)")
end

# Random access with color
cv = ColorView(items, GAY_SEED)
item, color = cv[2]  # ("banana", RGB{Float32}(...))
```
"""
struct ColorView{T}
    data::T
    seed::UInt64
end

ColorView(data) = ColorView(data, GAY_SEED)

Base.length(cv::ColorView) = length(cv.data)

function Base.getindex(cv::ColorView, i::Integer)
    item = cv.data[i]
    color = hash_color_rgb(UInt64(i), cv.seed)
    (item=item, color=color)
end

function Base.iterate(cv::ColorView, state=1)
    if state > length(cv.data)
        return nothing
    end
    item = cv.data[state]
    color = hash_color_rgb(UInt64(state), cv.seed)
    ((item=item, color=color, index=state), state + 1)
end

function Base.collect(cv::ColorView)
    [(item=cv.data[i], color=hash_color_rgb(UInt64(i), cv.seed), index=i)
     for i in 1:length(cv.data)]
end

function Base.show(io::IO, cv::ColorView{T}) where T
    print(io, "ColorView{$T}($(length(cv)) items, seed=0x$(string(cv.seed, base=16)))")
end

# ═══════════════════════════════════════════════════════════════════════════
# @colorable Macro
# ═══════════════════════════════════════════════════════════════════════════

"""
    @colorable T

Macro to make a type colorable based on its field hash.
Generates a `colorize` method that hashes all field values.

# Example
```julia
struct Point
    x::Float64
    y::Float64
end

@colorable Point

p = Point(1.0, 2.0)
c = colorize(p, GAY_SEED)  # Deterministic color from field hash
```
"""
macro colorable(T)
    quote
        function Gay.colorize(x::$(esc(T)), seed::UInt64=Gay.GAY_SEED)
            field_hash = hash(x)
            Gay.hash_color_rgb(UInt64(field_hash & typemax(Int64)), seed)
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# ColoredView: Matrix/Array coloring
# ═══════════════════════════════════════════════════════════════════════════

"""
    ColoredView{T,N}

A colored view of an N-dimensional array. Each element index maps to a color.

# Example
```julia
A = rand(10, 10)
cv = ColoredView(A, GAY_SEED)
cv[3, 5]  # Returns (value, color) tuple
```
"""
struct ColoredView{T,N} <: AbstractArray{Tuple{T,RGB{Float32}},N}
    data::AbstractArray{T,N}
    seed::UInt64
end

ColoredView(data::AbstractArray) = ColoredView(data, GAY_SEED)

Base.size(cv::ColoredView) = size(cv.data)

function Base.getindex(cv::ColoredView{T,N}, I::Vararg{Int,N}) where {T,N}
    val = cv.data[I...]
    idx = LinearIndices(cv.data)[I...]
    color = hash_color_rgb(UInt64(idx), cv.seed)
    (val, color)
end

# ═══════════════════════════════════════════════════════════════════════════
# Fingerprinting Protocol
# ═══════════════════════════════════════════════════════════════════════════

"""
    color_fingerprint(colors::AbstractVector{RGB{Float32}}) -> UInt64

Compute XOR fingerprint of a color vector. Order-independent for SPI verification.
"""
function color_fingerprint(colors::AbstractVector{<:RGB})
    fp = UInt64(0)
    for c in colors
        r_bits = reinterpret(UInt32, Float32(c.r))
        g_bits = reinterpret(UInt32, Float32(c.g))
        b_bits = reinterpret(UInt32, Float32(c.b))
        fp ⊻= UInt64(r_bits) | (UInt64(g_bits) << 24) | (UInt64(b_bits) << 48)
    end
    fp
end

"""
    spi_verify(f::Function, args...; n_trials::Int=100) -> Bool

Verify that a function produces identical colored results across multiple calls.
Returns true if SPI holds, false otherwise.
"""
function spi_verify(f::Function, args...; n_trials::Int=100)
    results = [f(args...) for _ in 1:n_trials]
    all(r == results[1] for r in results)
end

"""
    parallel_spi_verify(f::Function, items::AbstractVector, seed::UInt64; 
                        n_threads::Int=Threads.nthreads()) -> NamedTuple

Verify SPI holds under parallel execution.
Compares fingerprints from parallel vs sequential coloring.
"""
function parallel_spi_verify(f::Function, items::AbstractVector, seed::UInt64; 
                              n_threads::Int=Threads.nthreads())
    n = length(items)
    
    # Parallel coloring
    parallel_colors = Vector{RGB{Float32}}(undef, n)
    chunk_size = cld(n, n_threads)
    
    Threads.@threads for tid in 1:n_threads
        start_idx = (tid - 1) * chunk_size + 1
        end_idx = min(tid * chunk_size, n)
        for i in start_idx:end_idx
            parallel_colors[i] = f(items[i], seed)
        end
    end
    
    parallel_fp = color_fingerprint(parallel_colors)
    
    # Sequential coloring
    sequential_colors = [f(item, seed) for item in items]
    sequential_fp = color_fingerprint(sequential_colors)
    
    (spi_verified = parallel_fp == sequential_fp,
     parallel_fingerprint = parallel_fp,
     sequential_fingerprint = sequential_fp,
     colors_match = parallel_colors == sequential_colors)
end

export ColoredView, color_fingerprint, spi_verify, parallel_spi_verify
