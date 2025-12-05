# Black Hole Accretion Disk Visualization with Gay.jl
# Inspired by Event Horizon Telescope imagery and Comrade.jl VLBI modeling
#
# Uses splittable deterministic RNG to generate reproducible false-color
# renderings of simulated black hole photon rings in wide-gamut color spaces.

using Gay
using Colors

"""
    render_blackhole(; seed=1337, rings=12, resolution=40, colorspace=Rec2020())

Render a black hole accretion disk as ANSI art.
The gravitational lensing creates concentric photon rings with 
relativistic Doppler boosting (brighter on approaching side).

Uses Gay.jl's splittable RNG for reproducible visualizations.
"""
function render_blackhole(; seed::Integer=1337, rings::Int=12, 
                           resolution::Int=40, colorspace::ColorSpace=Rec2020())
    gay_seed!(seed)
    
    # Generate ring colors - hot plasma palette
    # Inner rings are hotter (bluer/whiter), outer rings cooler (redder)
    ring_colors = RGB[]
    for i in 1:rings
        # Temperature gradient: inner=hot, outer=cool
        temp = 1.0 - (i - 1) / rings  # 1.0 (hot) to ~0 (cool)
        
        # Sample base color deterministically
        base = next_color(colorspace)
        
        # Shift toward blackbody spectrum based on "temperature"
        # Hot: white/blue, Cool: orange/red
        if temp > 0.7
            # Hot inner disk - shift toward white/blue
            mixed = weighted_color_mean(temp, RGB(0.9, 0.95, 1.0), base)
        elseif temp > 0.4
            # Middle - orange/yellow
            mixed = weighted_color_mean(0.6, RGB(1.0, 0.7, 0.2), base)
        else
            # Outer - deep red
            mixed = weighted_color_mean(0.7, RGB(0.8, 0.2, 0.05), base)
        end
        push!(ring_colors, clamp_to_gamut(mixed, colorspace))
    end
    
    # Render the black hole
    cx, cy = resolution ÷ 2, resolution ÷ 2
    shadow_radius = resolution ÷ 6  # Event horizon shadow
    
    output = IOBuffer()
    
    println(output, "\n  ╔══════════════════════════════════════════════════════════╗")
    println(output, "  ║  BLACK HOLE - Seed: $seed | ColorSpace: $(typeof(colorspace))  ║")
    println(output, "  ╚══════════════════════════════════════════════════════════╝\n")
    
    for y in 1:resolution
        print(output, "  ")
        for x in 1:(resolution * 2)  # 2:1 aspect ratio for terminal
            # Distance from center
            dx = (x / 2) - cx
            dy = y - cy
            r = sqrt(dx^2 + dy^2)
            
            # Angle for Doppler effect (approaching side brighter)
            θ = atan(dy, dx)
            doppler = 0.5 + 0.5 * cos(θ)  # Brighter on right (approaching)
            
            if r < shadow_radius
                # Black hole shadow - pure black
                print(output, "  ")
            elseif r < shadow_radius * 3.5
                # Photon ring region
                ring_idx = clamp(round(Int, (r - shadow_radius) / 
                                 ((shadow_radius * 2.5) / rings)) + 1, 1, rings)
                
                # Apply Doppler boosting
                c = ring_colors[ring_idx]
                brightness = 0.3 + 0.7 * doppler
                boosted = RGB(
                    clamp(c.r * brightness, 0, 1),
                    clamp(c.g * brightness, 0, 1),
                    clamp(c.b * brightness, 0, 1)
                )
                
                # ANSI true-color output
                ri = round(Int, boosted.r * 255)
                gi = round(Int, boosted.g * 255)
                bi = round(Int, boosted.b * 255)
                print(output, "\e[48;2;$(ri);$(gi);$(bi)m  \e[0m")
            else
                # Background - faint glow
                glow = exp(-((r - shadow_radius * 3.5) / 10)^2)
                if glow > 0.1
                    gi = round(Int, glow * 30)
                    print(output, "\e[48;2;$(gi);$(gi÷2);0m  \e[0m")
                else
                    print(output, "  ")
                end
            end
        end
        println(output)
    end
    
    # Legend
    println(output, "\n  Photon Ring Colors (inner → outer):")
    print(output, "  ")
    show_colors(ring_colors; width=4)
    
    println(output, "\n  Doppler Effect: ← receding (dim) | approaching (bright) →")
    println(output, "  Inner rings: Hotter plasma (~10⁹ K)")
    println(output, "  Outer rings: Cooler plasma (~10⁶ K)")
    
    return String(take!(output))
end

"""
    blackhole_sequence(n::Int; seed=42)

Generate n deterministic black hole visualizations.
Each uses a split of the RNG for reproducible art.
"""
function blackhole_sequence(n::Int; seed::Integer=42)
    gay_seed!(seed)
    for i in 1:n
        # Each visualization gets its own deterministic seed
        vis_seed = rand(gay_split(), UInt64) % 10000
        println(render_blackhole(seed=Int(vis_seed), colorspace=Rec2020()))
        println("\n" * "─"^70 * "\n")
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# EHT-style ring decomposition (simplified)
# ═══════════════════════════════════════════════════════════════════════════

"""
    eht_rings(; seed=2017, n_rings=5)

Simulate Event Horizon Telescope photon ring structure.
The n=1,2,3... photon rings correspond to light that orbits
the black hole 1,2,3... times before escaping.

Returns ring intensities colored by Gay.jl palette.
"""
function eht_rings(; seed::Integer=2017, n_rings::Int=5, colorspace::ColorSpace=DisplayP3())
    gay_seed!(seed)
    
    println("\n  ╔════════════════════════════════════════╗")
    println("  ║  EHT Photon Ring Decomposition        ║")
    println("  ║  M87* / Sgr A* style visualization    ║")
    println("  ╚════════════════════════════════════════╝\n")
    
    # Generate colors for each ring order
    colors = next_palette(n_rings, colorspace; min_distance=25.0)
    
    for n in 1:n_rings
        # Ring intensity falls off exponentially with orbit number
        intensity = exp(-0.5 * (n - 1))
        width = max(1, round(Int, 30 * intensity))
        
        c = colors[n]
        ri = round(Int, c.r * 255 * intensity)
        gi = round(Int, c.g * 255 * intensity)
        bi = round(Int, c.b * 255 * intensity)
        
        bar = "\e[48;2;$(ri);$(gi);$(bi)m" * " "^width * "\e[0m"
        println("  n=$n ring: $bar  ($(round(intensity*100, digits=1))% flux)")
    end
    
    println("\n  Ring colors (deterministic from seed=$seed):")
    print("  ")
    show_palette(colors)
end

# ═══════════════════════════════════════════════════════════════════════════
# Main demo
# ═══════════════════════════════════════════════════════════════════════════

if abspath(PROGRAM_FILE) == @__FILE__
    println("\n" * "═"^70)
    println("  Gay.jl Black Hole Imaging Demo")
    println("  Deterministic colors via SplittableRandoms (Pigeons.jl pattern)")
    println("═"^70)
    
    # Main visualization
    println(render_blackhole(seed=1337, rings=10, resolution=30, colorspace=Rec2020()))
    
    # EHT ring decomposition
    eht_rings(seed=2017, n_rings=5, colorspace=DisplayP3())
    
    # Show reproducibility
    println("\n\n  ─── Reproducibility Demo ───")
    println("  Same seed = same colors, always:")
    for seed in [42, 42, 1337, 42]
        gay_seed!(seed)
        print("  seed=$seed: ")
        show_colors([next_color(Rec2020()) for _ in 1:8])
    end
end
