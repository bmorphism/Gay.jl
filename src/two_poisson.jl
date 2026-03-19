"""
    two_poisson.jl

2-Dimensional Poisson Processes with Chromatic Identity and SPI.

Features:
- Poisson point process in 2D space
- Chromatic identity for each point event
- Intensity maps with compositional structure
- Superposition and thinning operations
- SPI verification across parallel workers
"""

module TwoPoisson

using Colors, Statistics, SplittableRandoms

export
    # Types
    PoissonProcess2D,
    PointEventResult,
    
    # Creation
    create_poisson_2d,
    set_intensity!,
    
    # Sampling
    sample_points!,
    sample_arrival_times,
    
    # Operations
    thin_process,
    superpose_processes,
    
    # Analysis
    point_colors,
    point_positions,
    point_times,
    intensity_at,
    verify_spi

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

const DEFAULT_LAMBDA = 1.0  # Default intensity
const DEFAULT_TMIN = 0.0
const DEFAULT_TMAX = 1.0

# ─────────────────────────────────────────────────────────────────────────
# SplitMix64 & Color Generation
# ─────────────────────────────────────────────────────────────────────────

function splitmix64(state::UInt64)::Tuple{UInt64, UInt64}
    """SplitMix64 PRNG - returns (output, next_state)"""
    z = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ⊻ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    z ⊻ (z >> 31), (state + 1) & 0xFFFFFFFFFFFFFFFF
end

function color_from_seed(seed::UInt64)::RGB{Float64}
    """Generate deterministic RGB color from seed"""
    r, s1 = splitmix64(seed)
    g, s2 = splitmix64(s1)
    b, _  = splitmix64(s2)
    RGB(
        (r >> 56) / 255.0,
        (g >> 56) / 255.0,
        (b >> 56) / 255.0
    )
end

function next_color(point_id::UInt64, thread_id::UInt64)::RGB{Float64}
    """Get color for (point_id, thread_id) via gay_seed"""
    seed = ((point_id + 1) << 32) | (thread_id & 0xFFFFFFFFFFFFFFFF)
    color_from_seed(seed)
end

# ═══════════════════════════════════════════════════════════════════════════
# INTENSITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

"""Abstract base for intensity functions"""
abstract type IntensityFunction end

"""Constant intensity λ(x,y,t) = λ"""
struct ConstantIntensity <: IntensityFunction
    lambda::Float64
end

"""Gaussian intensity λ(x,y,t) = λ₀ * exp(-(x²+y²+t²)/(2σ²))"""
struct GaussianIntensity <: IntensityFunction
    lambda0::Float64  # Peak intensity
    sigma::Float64    # Spatial/temporal spread
end

"""Separable intensity λ(x,y,t) = λₓ(x) * λᵧ(y) * λₜ(t)"""
struct SeparableIntensity <: IntensityFunction
    spatial_fn::Function  # λₓ(x, y)
    temporal_fn::Function # λₜ(t)
end

function evaluate_intensity(intensity::ConstantIntensity, x::Float64, y::Float64, t::Float64)::Float64
    intensity.lambda
end

function evaluate_intensity(intensity::GaussianIntensity, x::Float64, y::Float64, t::Float64)::Float64
    dist_sq = x^2 + y^2 + t^2
    intensity.lambda0 * exp(-dist_sq / (2 * intensity.sigma^2))
end

function evaluate_intensity(intensity::SeparableIntensity, x::Float64, y::Float64, t::Float64)::Float64
    intensity.spatial_fn(x, y) * intensity.temporal_fn(t)
end

# ═══════════════════════════════════════════════════════════════════════════
# POISSON PROCESS STATE
# ═══════════════════════════════════════════════════════════════════════════

"""
    PoissonProcess2D

State of a 2D Poisson point process.
"""
mutable struct PoissonProcess2D
    # Domain
    xmin::Float64
    xmax::Float64
    ymin::Float64
    ymax::Float64
    tmin::Float64
    tmax::Float64
    
    # Intensity function
    intensity::IntensityFunction
    max_intensity::Float64  # For rejection sampling
    
    # Points (in temporal order)
    positions::Vector{Tuple{Float64, Float64}}  # (x, y)
    times::Vector{Float64}                       # t
    colors::Vector{RGB{Float64}}                 # Chromatic identity
    point_ids::Vector{UInt64}                   # For reproducibility
    
    # Metadata
    thread_id::UInt64
    seed::UInt64
    num_points::UInt64
end

"""
    PointEventResult

Result of sampling from Poisson process.
"""
struct PointEventResult
    positions::Vector{Tuple{Float64, Float64}}
    times::Vector{Float64}
    colors::Vector{RGB{Float64}}
    point_ids::Vector{UInt64}
    num_points::Int64
    intensity_integral::Float64  # Expected count
    spi_hash::UInt64
end

# ═══════════════════════════════════════════════════════════════════════════
# POISSON PROCESS CREATION
# ═══════════════════════════════════════════════════════════════════════════

"""
    create_poisson_2d(domain; intensity=..., seed=0x42, thread_id=0)

Create a 2D Poisson point process on the specified domain.

# Arguments
- `domain`: Tuple (xmin, xmax, ymin, ymax, tmin, tmax)
- `intensity`: IntensityFunction (default: ConstantIntensity(1.0))
- `seed`: Root seed for reproducibility
- `thread_id`: Worker ID for parallel execution
"""
function create_poisson_2d(
    domain::Tuple{Float64, Float64, Float64, Float64, Float64, Float64};
    intensity::IntensityFunction=ConstantIntensity(1.0),
    seed::UInt64=0x42,
    thread_id::UInt64=0
)::PoissonProcess2D
    
    xmin, xmax, ymin, ymax, tmin, tmax = domain
    
    @assert xmin < xmax "Invalid x domain"
    @assert ymin < ymax "Invalid y domain"
    @assert tmin < tmax "Invalid t domain"
    
    # Compute max intensity for rejection sampling
    # (In practice, this should be provided or estimated)
    max_intensity = if intensity isa ConstantIntensity
        intensity.lambda
    else
        2.0  # Conservative default
    end
    
    PoissonProcess2D(
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        tmin=tmin,
        tmax=tmax,
        intensity=intensity,
        max_intensity=max_intensity,
        positions=Tuple{Float64, Float64}[],
        times=Float64[],
        colors=RGB{Float64}[],
        point_ids=UInt64[],
        thread_id=thread_id,
        seed=seed,
        num_points=0
    )
end

"""
    set_intensity!(process, intensity; max_intensity=nothing)

Update the intensity function.
"""
function set_intensity!(
    process::PoissonProcess2D,
    intensity::IntensityFunction;
    max_intensity::Union{Float64, Nothing}=nothing
)
    process.intensity = intensity
    
    if max_intensity !== nothing
        process.max_intensity = max_intensity
    elseif intensity isa ConstantIntensity
        process.max_intensity = intensity.lambda
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# POISSON POINT SAMPLING
# ═══════════════════════════════════════════════════════════════════════════

"""
    sample_points!(process; max_points=nothing)

Sample points from the 2D Poisson process using rejection sampling.

Algorithm (Thinning/Rejection):
1. Sample uniform times (Poisson arrivals)
2. For each time, sample uniform (x,y) in domain
3. Accept with probability λ(x,y,t) / max_λ
4. Assign chromatic identity
"""
function sample_points!(
    process::PoissonProcess2D;
    max_points::Union{Int64, Nothing}=nothing
)
    
    # Expected number of points
    domain_volume = (process.xmax - process.xmin) *
                   (process.ymax - process.ymin) *
                   (process.tmax - process.tmin)
    
    expected_count = process.max_intensity * domain_volume
    
    # Bound search
    search_limit = max_points !== nothing ? max_points * 5 : Int64(ceil(expected_count * 2))
    
    # Rejection sampling
    point_count = UInt64(0)
    attempt = UInt64(0)
    
    while attempt < search_limit
        attempt += 1
        
        # Sample uniform time
        t_frac = rand()
        t = process.tmin + t_frac * (process.tmax - process.tmin)
        
        # Sample uniform position
        x_frac = rand()
        y_frac = rand()
        x = process.xmin + x_frac * (process.xmax - process.xmin)
        y = process.ymin + y_frac * (process.ymax - process.ymin)
        
        # Evaluate intensity
        lambda_xy = evaluate_intensity(process.intensity, x, y, t)
        
        # Accept/reject
        u = rand()
        if u < lambda_xy / process.max_intensity
            point_count += 1
            
            push!(process.positions, (x, y))
            push!(process.times, t)
            
            # Chromatic identity
            point_id = point_count
            color = next_color(point_id, process.thread_id)
            push!(process.colors, color)
            push!(process.point_ids, point_id)
            
            if max_points !== nothing && point_count >= max_points
                break
            end
        end
    end
    
    process.num_points = point_count
    
    # Sort by time
    time_order = sortperm(process.times)
    process.positions = process.positions[time_order]
    process.times = process.times[time_order]
    process.colors = process.colors[time_order]
    process.point_ids = process.point_ids[time_order]
end

"""
    sample_arrival_times(num_points, tmin, tmax, lambda)

Sample num_points arrival times from a Poisson process with rate lambda.

Uses exponential inter-arrival times.
"""
function sample_arrival_times(
    num_points::Int64,
    tmin::Float64,
    tmax::Float64,
    lambda::Float64
)::Vector{Float64}
    
    times = Float64[]
    t = tmin
    
    for _ in 1:num_points
        # Exponential inter-arrival time
        delta_t = -log(rand()) / lambda
        t += delta_t
        
        if t <= tmax
            push!(times, t)
        else
            break
        end
    end
    
    sort!(times)
end

# ═══════════════════════════════════════════════════════════════════════════
# POISSON PROCESS OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════

"""
    thin_process(process, prob; keep_ids=false)

Thin the Poisson process by independently keeping each point with probability `prob`.

Returns new PoissonProcess2D with thinned points.
"""
function thin_process(
    process::PoissonProcess2D,
    prob::Float64;
    keep_ids::Bool=false
)::PoissonProcess2D
    
    @assert 0.0 <= prob <= 1.0 "Probability must be in [0, 1]"
    
    # Create new process
    thinned = PoissonProcess2D(
        xmin=process.xmin,
        xmax=process.xmax,
        ymin=process.ymin,
        ymax=process.ymax,
        tmin=process.tmin,
        tmax=process.tmax,
        intensity=process.intensity,
        max_intensity=process.max_intensity,
        positions=Tuple{Float64, Float64}[],
        times=Float64[],
        colors=RGB{Float64}[],
        point_ids=UInt64[],
        thread_id=process.thread_id,
        seed=process.seed,
        num_points=0
    )
    
    # Keep each point independently
    new_id = UInt64(0)
    for i in 1:length(process.positions)
        if rand() < prob
            new_id += 1
            push!(thinned.positions, process.positions[i])
            push!(thinned.times, process.times[i])
            
            # Re-color if not keeping IDs
            if keep_ids
                push!(thinned.colors, process.colors[i])
                push!(thinned.point_ids, process.point_ids[i])
            else
                color = next_color(new_id, thinned.thread_id)
                push!(thinned.colors, color)
                push!(thinned.point_ids, new_id)
            end
        end
    end
    
    thinned.num_points = new_id
    thinned
end

"""
    superpose_processes(processes)

Combine multiple Poisson processes into one (merging and sorting by time).

Preserves chromatic identity of original processes.
"""
function superpose_processes(
    processes::Vector{PoissonProcess2D}
)::PoissonProcess2D
    
    @assert !isempty(processes) "Need at least one process"
    
    # Use first process as template
    first = processes[1]
    
    superposed = PoissonProcess2D(
        xmin=first.xmin,
        xmax=first.xmax,
        ymin=first.ymin,
        ymax=first.ymax,
        tmin=first.tmin,
        tmax=first.tmax,
        intensity=first.intensity,
        max_intensity=first.max_intensity,
        positions=Tuple{Float64, Float64}[],
        times=Float64[],
        colors=RGB{Float64}[],
        point_ids=UInt64[],
        thread_id=0,  # Merged process has no specific thread
        seed=first.seed,
        num_points=0
    )
    
    # Merge all points
    point_id = UInt64(0)
    for process in processes
        for i in 1:length(process.positions)
            point_id += 1
            push!(superposed.positions, process.positions[i])
            push!(superposed.times, process.times[i])
            push!(superposed.colors, process.colors[i])
            push!(superposed.point_ids, point_id)
        end
    end
    
    superposed.num_points = point_id
    
    # Sort by time
    time_order = sortperm(superposed.times)
    superposed.positions = superposed.positions[time_order]
    superposed.times = superposed.times[time_order]
    superposed.colors = superposed.colors[time_order]
    superposed.point_ids = superposed.point_ids[time_order]
    
    superposed
end

# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS & EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

"""
    point_colors(process)::Vector{RGB{Float64}}

Get colors assigned to each point.
"""
point_colors(process::PoissonProcess2D) = process.colors

"""
    point_positions(process)::Vector{Tuple{Float64, Float64}}

Get (x, y) positions of each point.
"""
point_positions(process::PoissonProcess2D) = process.positions

"""
    point_times(process)::Vector{Float64}

Get time coordinates of each point.
"""
point_times(process::PoissonProcess2D) = process.times

"""
    intensity_at(intensity, x, y, t)::Float64

Evaluate intensity function at point.
"""
intensity_at(intensity::IntensityFunction, x::Float64, y::Float64, t::Float64) =
    evaluate_intensity(intensity, x, y, t)

"""
    result(process)::PointEventResult

Convert process to result structure.
"""
function result(process::PoissonProcess2D)::PointEventResult
    # Compute intensity integral (expected count)
    domain_volume = (process.xmax - process.xmin) *
                   (process.ymax - process.ymin) *
                   (process.tmax - process.tmin)
    
    # For constant intensity, this is exact
    integral = if process.intensity isa ConstantIntensity
        process.intensity.lambda * domain_volume
    else
        # Estimate via max_intensity
        process.max_intensity * domain_volume * 0.5
    end
    
    # SPI hash from colors
    spi_hash = UInt64(0)
    for (pos, t, color) in zip(process.positions, process.times, process.colors)
        x, y = pos
        # Hash position
        x_bits = UInt64(reinterpret(Int64, x))
        y_bits = UInt64(reinterpret(Int64, y))
        t_bits = UInt64(reinterpret(Int64, t))
        spi_hash ⊻= x_bits ⊻ y_bits ⊻ t_bits
        
        # Hash color
        r_bits = UInt64(Int64(round(color.r * 255)))
        g_bits = UInt64(Int64(round(color.g * 255)))
        b_bits = UInt64(Int64(round(color.b * 255)))
        spi_hash ⊻= (r_bits << 8) | (g_bits << 16) | (b_bits << 24)
        
        spi_hash = (spi_hash << 7) | (spi_hash >> 57)  # Rotate
    end
    
    PointEventResult(
        positions=process.positions,
        times=process.times,
        colors=process.colors,
        point_ids=process.point_ids,
        num_points=Int64(process.num_points),
        intensity_integral=integral,
        spi_hash=spi_hash
    )
end

# ═══════════════════════════════════════════════════════════════════════════
# SPI VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════

"""
    verify_spi(results::Vector{PointEventResult})::Bool

Verify that all results have same SPI hash (Strong Parallelism Invariance).
"""
function verify_spi(results::Vector{PointEventResult})::Bool
    if isempty(results)
        return true
    end
    
    first_hash = results[1].spi_hash
    all(r -> r.spi_hash == first_hash, results)
end

# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION & REPORTING
# ═══════════════════════════════════════════════════════════════════════════

"""
    print_process(result)

Print summary of Poisson process.
"""
function print_process(result::PointEventResult)
    println("\n" * "="^80)
    println("2D POISSON PROCESS RESULT")
    println("="^80)
    
    println("Points sampled: $(result.num_points)")
    println("Expected (from intensity): $(round(result.intensity_integral; digits=2))")
    
    if !isempty(result.positions)
        xs = [p[1] for p in result.positions]
        ys = [p[2] for p in result.positions]
        ts = result.times
        
        println("\nSpatial extent:")
        println("  X: $(round(minimum(xs); digits=3)) to $(round(maximum(xs); digits=3))")
        println("  Y: $(round(minimum(ys); digits=3)) to $(round(maximum(ys); digits=3))")
        println("  T: $(round(minimum(ts); digits=3)) to $(round(maximum(ts); digits=3))")
        
        println("\nFirst 5 points:")
        for i in 1:min(5, length(result.positions))
            x, y = result.positions[i]
            t = result.times[i]
            color = result.colors[i]
            println("  $i: ($(round(x;digits=3)), $(round(y;digits=3)), $(round(t;digits=3))) " *
                   "RGB($(round(color.r;digits=3)), $(round(color.g;digits=3)), $(round(color.b;digits=3)))")
        end
        
        if length(result.positions) > 5
            println("  ... ($(length(result.positions) - 5) more points)")
        end
    end
    
    println("\nSPI Hash: 0x$(string(result.spi_hash; base=16))")
    println("="^80 * "\n")
end

end  # module TwoPoisson
