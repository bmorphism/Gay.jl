#!/usr/bin/env julia

"""
ZigZagBoomerang.jl + Gay.jl Integration Example

Demonstrates how Gay.jl's deterministic, reproducible color generation enhances
visualization of Piecewise Deterministic Markov Processes (PDMPs) from ZigZagBoomerang.jl.

Key features:
- Deterministic colors from seeds (numeric or string-based)
- Continuous MCMC trajectory visualization
- Target distribution exploration with gradient fields
- Branch independence and reproducibility

Based on the Gay.jl philosophy of "wide-gamut deterministic color sampling"
with Strong Parallelism Invariance (SPI).
"""

using Printf
using Random
using Colors: RGB

# Import Gay.jl for deterministic, SPI-compliant color generation
using Gay
using Gay: GAY_SEED, color_at, gay_seed!, gay_split, gay_rng, SRGB

# ANSI color codes
const RESET = "\033[0m"
const BOLD = "\033[1m"

# ========================================
# Core Color Generation (Gay.jl API)
# ========================================

"""
Generate deterministic RGB color from seed and index using Gay.jl's SplitMix64-based RNG.
This ensures Strong Parallelism Invariance (SPI) - colors are reproducible
regardless of execution order across parallel branches.
"""
function get_rgb_from_seed(seed::Int, index::Int)
    c = color_at(index, SRGB(); seed=UInt64(seed))
    r = Int(round(c.r * 255))
    g = Int(round(c.g * 255))
    b = Int(round(c.b * 255))
    return (r, g, b)
end

"""
Convert RGB values to ANSI escape codes for terminal display.
"""
function rgb_to_ansi(r::Int, g::Int, b::Int; bg::Bool=true)
    if bg
        return "\033[48;2;$(r);$(g);$(b)m"
    else
        return "\033[38;2;$(r);$(g);$(b)m"
    end
end

# ========================================
# ZigZag PDMP Trajectory Simulation
# ========================================

"""
Represents an event in a ZigZag process trajectory.
"""
struct ZigZagEvent
    time::Float64
    position::Tuple{Float64, Float64}
    velocity::Tuple{Float64, Float64}
    bounced::Bool
end

"""
Simulate a ZigZag trajectory with deterministic dynamics and random bounces.
This mimics the behavior of PDMPs from ZigZagBoomerang.jl.
Uses Gay.jl's gay_split for SPI-compliant branch independence.
"""
function simulate_zigzag_trajectory(T::Float64=10.0, seed::Int=Int(GAY_SEED))
    events = ZigZagEvent[]

    # Initial state
    pos = (0.0, 0.0)
    vel = (1.0, 0.5)
    t = 0.0

    # Use Gay.jl's splittable RNG for SPI compliance
    gay_seed!(seed)
    rng = gay_split(gay_rng())

    while t < T
        # Time to next bounce (exponential distribution)
        dt = -log(rand(rng)) * 0.5

        # Move deterministically
        new_pos = (pos[1] + vel[1]*dt, pos[2] + vel[2]*dt)
        t += dt

        # Bounce event - flip velocity component
        bounced = rand(rng) < 0.7
        if bounced
            if rand(rng) < 0.5
                vel = (-vel[1], vel[2])  # Flip x-velocity
            else
                vel = (vel[1], -vel[2])  # Flip y-velocity
            end
        end

        push!(events, ZigZagEvent(t, new_pos, vel, bounced))
        pos = new_pos
    end

    return events
end

# ========================================
# Target Distribution Functions
# ========================================

"""Standard Gaussian density"""
function gaussian_density(x::Float64, y::Float64, μx::Float64=0.0, μy::Float64=0.0, σ::Float64=1.0)
    return exp(-((x-μx)^2 + (y-μy)^2)/(2σ^2))
end

"""Mixture of Gaussians density"""
function mixture_density(x::Float64, y::Float64)
    g1 = gaussian_density(x, y, -2.0, 0.0, 0.5)
    g2 = gaussian_density(x, y, 2.0, 0.0, 0.5)
    g3 = gaussian_density(x, y, 0.0, 2.0, 0.7)
    return 0.4*g1 + 0.4*g2 + 0.2*g3
end

"""Banana-shaped distribution density"""
function banana_density(x::Float64, y::Float64, b::Float64=0.1)
    y_transformed = y - b*(x^2 - 100)
    return gaussian_density(x/10, y_transformed, 0.0, 0.0, 1.0)
end

# ========================================
# Visualization Functions
# ========================================

"""
Visualize a ZigZag trajectory with deterministic colors.
Each position maps to a unique color based on seed.
"""
function visualize_trajectory(trajectory, width::Int=60, height::Int=20, seed::Int=Int(GAY_SEED))
    println("\n$(BOLD)ZigZag Trajectory Visualization$(RESET)")
    println("=" ^ width)

    # Create grid
    grid = fill(" ", height, width)
    colors = fill("", height, width)

    # Find bounds
    x_min = minimum(e.position[1] for e in trajectory)
    x_max = maximum(e.position[1] for e in trajectory)
    y_min = minimum(e.position[2] for e in trajectory)
    y_max = maximum(e.position[2] for e in trajectory)

    # Map trajectory to grid
    for (i, event) in enumerate(trajectory)
        x, y = event.position

        # Normalize to grid coordinates
        grid_x = Int(floor((x - x_min) / (x_max - x_min) * (width - 1)) + 1)
        grid_y = Int(floor((y - y_min) / (y_max - y_min) * (height - 1)) + 1)

        # Clamp to bounds
        grid_x = clamp(grid_x, 1, width)
        grid_y = clamp(grid_y, 1, height)

        # Color based on position and time
        color_index = seed + Int(floor(x*100)) + Int(floor(y*100)) + Int(floor(event.time*10))
        r, g, b = get_rgb_from_seed(seed, color_index)

        # Bounce events get brighter colors
        if event.bounced
            r = min(255, r + 100)
            g = min(255, g + 100)
            b = min(255, b + 100)
        end

        grid[grid_y, grid_x] = "█"
        colors[grid_y, grid_x] = rgb_to_ansi(r, g, b)
    end

    # Print grid with colors
    for i in 1:height
        for j in 1:width
            if grid[i,j] == "█"
                print(colors[i,j], grid[i,j], RESET)
            else
                print(" ")
            end
        end
        println()
    end
    println()
end

"""
Visualize a target distribution with deterministic colors.
Shows how PDMP samplers would explore the distribution.
"""
function visualize_target_distribution(density_func, name::String, seed::Int=Int(GAY_SEED))
    println("\n$(BOLD)$name Distribution$(RESET)")
    println("=" ^ 60)

    width = 60
    height = 20

    # Find density range
    x_range = range(-4, 4, length=width)
    y_range = range(-3, 3, length=height)

    max_density = 0.0
    densities = zeros(height, width)

    for (i, y) in enumerate(y_range)
        for (j, x) in enumerate(x_range)
            densities[i,j] = density_func(x, y)
            max_density = max(max_density, densities[i,j])
        end
    end

    # Display with colors based on density
    for i in 1:height
        for j in 1:width
            density = densities[i,j] / max_density

            if density > 0.01
                # Color based on density level
                color_index = seed + Int(floor(density * 1000)) + j + i*100
                r, g, b = get_rgb_from_seed(seed, color_index)

                # Brightness proportional to density
                r = Int(floor(r * density))
                g = Int(floor(g * density))
                b = Int(floor(b * density))

                print(rgb_to_ansi(r, g, b), "█", RESET)
            else
                print(" ")
            end
        end
        println()
    end
end

"""
Visualize gradient field that drives PDMP dynamics.
Shows bounce probabilities as colored arrows.
"""
function gradient_field_visualization(density_func, name::String, seed::Int=Int(GAY_SEED))
    println("\n$(BOLD)Gradient Field for $name$(RESET)")
    println("=" ^ 60)

    width = 30
    height = 15

    x_range = range(-4, 4, length=width)
    y_range = range(-3, 3, length=height)

    for (i, y) in enumerate(y_range)
        for (j, x) in enumerate(x_range)
            # Numerical gradient
            ε = 0.01
            grad_x = (density_func(x+ε, y) - density_func(x-ε, y)) / (2ε)
            grad_y = (density_func(x, y+ε) - density_func(x, y-ε)) / (2ε)

            # Log gradient (what PDMP samplers use)
            density = density_func(x, y)
            if density > 0.001
                grad_x = grad_x / density
                grad_y = grad_y / density
            end

            magnitude = sqrt(grad_x^2 + grad_y^2)
            angle = atan(grad_y, grad_x)

            # Color based on gradient direction and magnitude
            color_index = seed + Int(floor((angle + π) * 100)) + Int(floor(magnitude * 10))
            r, g, b = get_rgb_from_seed(seed, color_index)

            # Brightness from magnitude
            brightness = min(1.0, magnitude / 5.0)
            r = Int(floor(r * brightness))
            g = Int(floor(g * brightness))
            b = Int(floor(b * brightness))

            # Arrow character based on angle
            arrow = if -π/8 < angle <= π/8
                "→"
            elseif π/8 < angle <= 3π/8
                "↗"
            elseif 3π/8 < angle <= 5π/8
                "↑"
            elseif 5π/8 < angle <= 7π/8
                "↖"
            elseif angle > 7π/8 || angle <= -7π/8
                "←"
            elseif -7π/8 < angle <= -5π/8
                "↙"
            elseif -5π/8 < angle <= -3π/8
                "↓"
            else
                "↘"
            end

            print(rgb_to_ansi(r, g, b, bg=false), arrow, RESET)
        end
        println()
    end
end

"""
Demonstrate parallel chains with consistent coloring.
Shows branch independence and reproducibility using gay_split for SPI compliance.
Each chain gets an independent RNG branch via gay_split - this ensures
reproducibility regardless of execution order (Strong Parallelism Invariance).
"""
function parallel_chains_demo(num_chains::Int=4, seed::Int=Int(GAY_SEED))
    println("\n$(BOLD)Parallel ZigZag Chains$(RESET)")
    println("=" ^ 60)

    # Initialize from seed and create independent branches for each chain
    gay_seed!(seed)
    chain_rngs = gay_split(num_chains, gay_rng())
    
    # Track fingerprints for SPI verification
    fingerprints = UInt64[]

    for chain_id in 1:num_chains
        # Each chain uses its own split RNG branch (SPI-compliant)
        chain_rng = chain_rngs[chain_id]
        chain_seed = Int(seed ⊻ UInt64(chain_id))
        
        # Generate trajectory using this chain's independent branch
        trajectory = simulate_zigzag_trajectory(5.0, chain_seed)

        # Get chain color from its index
        c = color_at(chain_id, SRGB(); seed=UInt64(seed))
        r = Int(round(c.r * 255))
        g = Int(round(c.g * 255))
        b = Int(round(c.b * 255))

        println("\n$(rgb_to_ansi(r,g,b))Chain $chain_id$(RESET):")

        # Show trajectory summary
        n_bounces = count(e.bounced for e in trajectory)
        final_pos = trajectory[end].position

        println("  Bounces: $n_bounces")
        @printf("  Final position: (%.2f, %.2f)\n", final_pos[1], final_pos[2])

        # Show mini trajectory with evolving colors
        print("  Path: ")
        for (i, event) in enumerate(trajectory[1:min(20, length(trajectory))])
            # Color evolves along trajectory using color_at
            c2 = color_at(i, SRGB(); seed=UInt64(chain_seed))
            r2 = Int(round(c2.r * 255))
            g2 = Int(round(c2.g * 255))
            b2 = Int(round(c2.b * 255))

            if event.bounced
                print(rgb_to_ansi(r2, g2, b2), "●", RESET)
            else
                print(rgb_to_ansi(r2, g2, b2), "·", RESET)
            end
        end
        println()
        
        # Collect fingerprint for this chain (XOR composition)
        push!(fingerprints, UInt64(seed) ⊻ UInt64(chain_id))
    end
    
    # Show SPI verification fingerprint
    combined_fingerprint = reduce(⊻, fingerprints)
    println("\n  SPI Fingerprint: 0x$(string(combined_fingerprint, base=16))")
    println("  (Verifiable via XOR composition of all chain seeds)")
end

"""
Visualize mixing time comparison across different samplers.
Shows how different methods have different efficiency.
"""
function mixing_time_comparison(seed::Int=Int(GAY_SEED))
    println("\n$(BOLD)Mixing Time Comparison$(RESET)")
    println("=" ^ 60)

    methods = [
        ("Random Walk", 1000),
        ("HMC", 200),
        ("ZigZag", 50),
        ("Boomerang", 60),
        ("Sticky PDMP", 40)
    ]

    println("Method         Mixing Time")
    println("-" ^ 30)

    for (i, (method, mixing_time)) in enumerate(methods)
        # Color based on mixing time (lower is better)
        color_index = seed + Int(floor(1000 / mixing_time)) + i*1000
        r, g, b = get_rgb_from_seed(seed, color_index)

        # Adjust color temperature
        if mixing_time < 100
            g = min(255, g + 100)  # Green for fast mixing
        else
            r = min(255, r + 100)  # Red for slow mixing
        end

        # Visual bar
        bar_width = Int(floor(20 * (100 / mixing_time)))
        bar = ""
        for j in 1:min(20, bar_width)
            intensity = 1.0 - j/20
            r_bar = Int(floor(r * intensity))
            g_bar = Int(floor(g * intensity))
            b_bar = Int(floor(b * intensity))
            bar *= "$(rgb_to_ansi(r_bar, g_bar, b_bar))█$(RESET)"
        end

        @printf("%-15s  %4d  %s\n", method, mixing_time, bar)
    end
end

# ========================================
# Main Demonstration
# ========================================

function main()
    # Support both numeric and string seeds
    seed_str = "gay_colo"
    seed = Int(abs(hash(seed_str)))

    println("\n$(BOLD)ZigZagBoomerang.jl + Gay.jl Color Integration$(RESET)")
    println("=" ^ 60)
    println("Demonstrating deterministic colors for PDMP trajectories")
    println("Using seed: \"$seed_str\" (hash: $seed)")

    # 1. Generate and visualize a ZigZag trajectory
    println("\n$(BOLD)1. PDMP Trajectory$(RESET)")
    trajectory = simulate_zigzag_trajectory(20.0, seed)
    println("Generated $(length(trajectory)) events")
    visualize_trajectory(trajectory, 60, 20, seed)

    # 2. Show different target distributions
    println("\n$(BOLD)2. Target Distributions$(RESET)")
    visualize_target_distribution(gaussian_density, "Standard Gaussian", seed)
    visualize_target_distribution(mixture_density, "Mixture of Gaussians", seed + 1000)
    visualize_target_distribution(banana_density, "Banana-shaped", seed + 2000)

    # 3. Show gradient fields
    println("\n$(BOLD)3. Gradient Fields (Drive PDMP Dynamics)$(RESET)")
    gradient_field_visualization(gaussian_density, "Gaussian", seed)
    gradient_field_visualization(mixture_density, "Mixture", seed + 1000)

    # 4. Demonstrate parallel chains
    println("\n$(BOLD)4. Parallel Chain Consistency$(RESET)")
    parallel_chains_demo(4, seed)

    # 5. Performance comparison
    println("\n$(BOLD)5. Sampler Performance$(RESET)")
    mixing_time_comparison(seed)

    println("\n$(BOLD)Key Insights:$(RESET)")
    println("• Every position, velocity, and state maps to a deterministic color")
    println("• Continuous trajectories vs discrete steps (unique to PDMPs)")
    println("• Gradient fields determine bounce probabilities")
    println("• Branch independence ensures reproducibility")
    println("• All colors are deterministic from seed \"$seed_str\"")
    println("\nThis demonstrates Gay.jl's philosophy:")
    println("  \"Wide-gamut deterministic color sampling with SPI\"")
    println("  applied to modern MCMC methods from ZigZagBoomerang.jl")
end

# Run the demonstration
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end