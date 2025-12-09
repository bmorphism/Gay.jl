#!/usr/bin/env julia
"""
Brass Factory

Brass = Cu (copper) + Zn (zinc), alloyed in the furnace.

The factory is a 2-transducer:
    Input A  = raw ore (copper ore, zinc ore, flux)
    Output B = brass products (ingots, sheets, tubes)
    State Q  = furnace temperature (controls alloy ratio)
    Profunctor t = metallurgical transformation with noise

The furnace is a Markov blanket:
    External (s) = ore delivery, market prices
    Blanket (b)  = furnace walls, crucible
    Internal (z) = molten metal, crystal formation

The Fokker-Planck of alloying:
    ∂ρ/∂t = -∇·(drift × ρ) + D∇²ρ
    
    drift    = thermodynamic gradient toward equilibrium alloy
    diffusion = atomic mixing, Brownian motion of metal atoms
    ρ        = concentration of Cu/Zn at each point

Brass composition affects color:
    Cu 60% Zn 40% = yellow brass (cartridge brass)
    Cu 70% Zn 30% = red brass (rose)  
    Cu 85% Zn 15% = gilding metal (gold-like)
    Cu 90% Zn 10% = commercial bronze (reddish)
"""

using Gay
using Colors: RGB

# ═══════════════════════════════════════════════════════════════════════════
# Brass Alloy Colors
# ═══════════════════════════════════════════════════════════════════════════

"""
Color of brass as a function of zinc content (0-45%).
Interpolates from copper red through gold to yellow.
"""
function brass_color(zn_percent::Float64)
    # Copper is reddish: RGB(184, 115, 51) ≈ #B87333
    # Pure brass (40% Zn) is yellow: RGB(181, 166, 66) ≈ #B5A642
    
    t = clamp(zn_percent / 45.0, 0.0, 1.0)
    
    # Interpolate
    r = 184 + (181 - 184) * t
    g = 115 + (166 - 115) * t  
    b = 51 + (66 - 51) * t
    
    return RGB(r/255, g/255, b/255)
end

# Named brass alloys
const BRASS_ALLOYS = Dict(
    :gilding_metal    => (zn=5.0,  name="Gilding Metal (95Cu/5Zn)"),
    :commercial_bronze => (zn=10.0, name="Commercial Bronze (90Cu/10Zn)"),
    :red_brass        => (zn=15.0, name="Red Brass (85Cu/15Zn)"),
    :low_brass        => (zn=20.0, name="Low Brass (80Cu/20Zn)"),
    :cartridge_brass  => (zn=30.0, name="Cartridge Brass (70Cu/30Zn)"),
    :yellow_brass     => (zn=35.0, name="Yellow Brass (65Cu/35Zn)"),
    :muntz_metal      => (zn=40.0, name="Muntz Metal (60Cu/40Zn)"),
)

# ═══════════════════════════════════════════════════════════════════════════
# Factory as 2-Transducer
# ═══════════════════════════════════════════════════════════════════════════

struct BrassFactory
    name::String
    furnace_temp::Float64      # Kelvin
    target_zn::Float64         # Target zinc percentage
    noise_level::Float64       # Process noise (η)
    batch_size::Int
end

"""
The furnace transforms ore into alloy via Fokker-Planck dynamics:
    - Drift toward target composition (thermodynamic equilibrium)
    - Diffusion from atomic mixing (Brownian motion in melt)
"""
function smelt(factory::BrassFactory, cu_ore::Float64, zn_ore::Float64)
    # Initial composition from ore ratio
    total = cu_ore + zn_ore
    zn_initial = 100 * zn_ore / total
    
    # Drift toward target (proportional control)
    drift = 0.1 * (factory.target_zn - zn_initial)
    
    # Diffusion (process noise)
    noise = factory.noise_level * (rand() - 0.5) * 2
    
    # Final zinc content
    zn_final = clamp(zn_initial + drift + noise, 0.0, 45.0)
    
    return zn_final
end

# ═══════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════

function ansi(c::RGB)
    r = round(Int, clamp(c.r, 0, 1) * 255)
    g = round(Int, clamp(c.g, 0, 1) * 255)
    b = round(Int, clamp(c.b, 0, 1) * 255)
    "\e[38;2;$(r);$(g);$(b)m"
end

function ansi_bg(c::RGB)
    r = round(Int, clamp(c.r, 0, 1) * 255)
    g = round(Int, clamp(c.g, 0, 1) * 255)
    b = round(Int, clamp(c.b, 0, 1) * 255)
    "\e[48;2;$(r);$(g);$(b)m"
end

const R = "\e[0m"
const BOLD = "\e[1m"
const DIM = "\e[2m"

function render_brass_spectrum()
    println()
    println("  $(BOLD)Brass Color Spectrum$(R)  $(DIM)(Cu → Cu/Zn alloys)$(R)")
    println("  ════════════════════════════════════════════════════")
    println()
    
    # Spectrum bar
    print("  ")
    for zn in 0:1:45
        c = brass_color(Float64(zn))
        print("$(ansi_bg(c))  $(R)")
    end
    println()
    
    print("  ")
    print("$(DIM)0%                    Zn content                    45%$(R)")
    println()
    println()
    
    # Named alloys
    println("  Named Alloys:")
    for (sym, alloy) in sort(collect(BRASS_ALLOYS), by=x->x[2].zn)
        c = brass_color(alloy.zn)
        bar = "$(ansi_bg(c))      $(R)"
        println("    $bar $(alloy.name)")
    end
    println()
end

function render_furnace(factory::BrassFactory; seed::Integer=42)
    gay_seed!(seed)
    
    c_fire = RGB(1.0, 0.3, 0.0)  # Orange fire
    c_metal = brass_color(factory.target_zn)
    
    println()
    println("  $(BOLD)Furnace: $(factory.name)$(R)")
    println("  ════════════════════════════════════════════════════")
    println()
    println("      $(ansi(c_fire))╔════════════════════════════════╗$(R)")
    println("      $(ansi(c_fire))║$(R)  $(DIM)T = $(round(Int, factory.furnace_temp))K$(R)                    $(ansi(c_fire))║$(R)")
    println("      $(ansi(c_fire))║$(R)                                $(ansi(c_fire))║$(R)")
    println("      $(ansi(c_fire))║$(R)    $(ansi_bg(c_metal))                          $(R)    $(ansi(c_fire))║$(R)")
    println("      $(ansi(c_fire))║$(R)    $(ansi_bg(c_metal))      MOLTEN BRASS        $(R)    $(ansi(c_fire))║$(R)")
    println("      $(ansi(c_fire))║$(R)    $(ansi_bg(c_metal))      $(round(100-factory.target_zn, digits=1))% Cu / $(round(factory.target_zn, digits=1))% Zn      $(R)    $(ansi(c_fire))║$(R)")
    println("      $(ansi(c_fire))║$(R)    $(ansi_bg(c_metal))                          $(R)    $(ansi(c_fire))║$(R)")
    println("      $(ansi(c_fire))║$(R)                                $(ansi(c_fire))║$(R)")
    println("      $(ansi(c_fire))╚════════════════════════════════╝$(R)")
    println("      $(ansi(c_fire))▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓$(R)")
    println()
    println("  $(DIM)η = $(factory.noise_level)  (process noise)$(R)")
    println()
end

function render_production_run(factory::BrassFactory, n_batches::Int; seed::Integer=42)
    gay_seed!(seed)
    
    println()
    println("  $(BOLD)Production Run: $(n_batches) batches$(R)")
    println("  ────────────────────────────────────────────────────")
    println()
    
    results = Float64[]
    
    for i in 1:n_batches
        # Random ore mix (centered around target)
        cu_ore = 100 - factory.target_zn + 5*(rand()-0.5)
        zn_ore = factory.target_zn + 5*(rand()-0.5)
        
        zn_final = smelt(factory, cu_ore, zn_ore)
        push!(results, zn_final)
        
        c = brass_color(zn_final)
        bar_width = round(Int, zn_final)
        bar = "$(ansi_bg(c))" * " "^bar_width * "$(R)"
        
        println("    Batch $(lpad(i,2)): $bar $(round(zn_final, digits=1))% Zn")
    end
    
    # Statistics
    μ = sum(results) / length(results)
    σ = sqrt(sum((results .- μ).^2) / length(results))
    
    println()
    println("  $(DIM)Mean: $(round(μ, digits=2))% Zn  (target: $(factory.target_zn)%)$(R)")
    println("  $(DIM)Std:  $(round(σ, digits=2))%      (noise: η=$(factory.noise_level))$(R)")
    println()
end

# ═══════════════════════════════════════════════════════════════════════════
# Factory as Markov Blanket
# ═══════════════════════════════════════════════════════════════════════════

function render_factory_blanket(; seed::Integer=1069)
    gay_seed!(seed)
    
    c_ext = next_color(SRGB())   # External
    c_bln = brass_color(30.0)   # Blanket (brass-colored)
    c_int = next_color(SRGB())   # Internal
    
    println()
    println("  $(BOLD)Brass Factory as Markov Blanket$(R)")
    println("  ════════════════════════════════════════════════════")
    println()
    println("    $(ansi(c_ext))┌─────────────────────────────────────┐$(R)")
    println("    $(ansi(c_ext))│       EXTERNAL (s)                  │$(R)")
    println("    $(ansi(c_ext))│   ore delivery, market prices,      │$(R)")
    println("    $(ansi(c_ext))│   energy supply, labor              │$(R)")
    println("    $(ansi(c_ext))└──────────────────┬──────────────────┘$(R)")
    println("                       $(ansi(c_ext))│$(R)")
    println("                       $(ansi(c_ext))▼$(R)")
    println("    $(ansi(c_bln))╔═════════════════════════════════════╗$(R)")
    println("    $(ansi(c_bln))║         BLANKET (b)                 ║$(R)")
    println("    $(ansi(c_bln))║   furnace walls, crucible,          ║$(R)")
    println("    $(ansi(c_bln))║   temperature sensors, flow valves  ║$(R)")
    println("    $(ansi(c_bln))╚══════════════════╤══════════════════╝$(R)")
    println("                       $(ansi(c_bln))│$(R)")
    println("                       $(ansi(c_bln))▼$(R)")
    println("    $(ansi(c_int))┌─────────────────────────────────────┐$(R)")
    println("    $(ansi(c_int))│       INTERNAL (z)                  │$(R)")
    println("    $(ansi(c_int))│   molten metal, crystal nucleation, │$(R)")
    println("    $(ansi(c_int))│   atomic diffusion, phase diagram   │$(R)")
    println("    $(ansi(c_int))└─────────────────────────────────────┘$(R)")
    println()
    println("  $(DIM)The furnace wall (blanket) screens off:$(R)")
    println("  $(DIM)  p(external, internal | blanket) = p(ext|b) × p(int|b)$(R)")
    println()
end

# ═══════════════════════════════════════════════════════════════════════════
# Fokker-Planck of Alloying
# ═══════════════════════════════════════════════════════════════════════════

function render_alloying_fokker_planck(; seed::Integer=42)
    gay_seed!(seed)
    
    println()
    println("  $(BOLD)Fokker-Planck of Alloying$(R)")
    println("  ════════════════════════════════════════════════════")
    println()
    println("    ∂ρ/∂t = -∇·(μρ) + D∇²ρ")
    println()
    println("    $(DIM)where:$(R)")
    println("      ρ(x,t) = concentration of Zn at position x, time t")
    println("      μ(x)   = thermodynamic drift (toward equilibrium)")
    println("      D      = atomic diffusion coefficient")
    println()
    println("    $(BOLD)Drift$(R) = bones of the alloy")
    println("      Atoms flow down the chemical potential gradient")
    println("      Toward the target composition (phase diagram minimum)")
    println()
    println("    $(BOLD)Diffusion$(R) = skin of the alloy")
    println("      Brownian motion of metal atoms in the melt")
    println("      Spreads concentration, smooths gradients")
    println()
    
    # Show concentration profile evolution
    println("    Concentration profile ρ(x) over time:")
    println()
    
    n_steps = 6
    for t in 0:n_steps
        progress = t / n_steps
        
        # Initial: step function, evolves to uniform
        width = 3 + round(Int, progress * 15)
        center = 25
        
        c = brass_color(20.0 + progress * 15)  # Color shifts as alloy homogenizes
        
        print("    t=$t  ")
        for x in 1:50
            if abs(x - center) < width
                intensity = 1.0 - abs(x - center) / width
                if intensity > 0.5
                    print("$(ansi_bg(c))█$(R)")
                elseif intensity > 0.2
                    print("$(ansi(c))▓$(R)")
                else
                    print("$(ansi(c))░$(R)")
                end
            else
                print(" ")
            end
        end
        println()
    end
    println()
    println("    $(DIM)Zn concentration homogenizes via drift + diffusion$(R)")
    println()
end

# ═══════════════════════════════════════════════════════════════════════════
# 2-Transducer View
# ═══════════════════════════════════════════════════════════════════════════

function render_factory_transducer(; seed::Integer=42)
    gay_seed!(seed)
    
    c_in = next_color(SRGB())
    c_state = brass_color(30.0)
    c_out = next_color(SRGB())
    
    println()
    println("  $(BOLD)Brass Factory as 2-Transducer$(R)")
    println("  ════════════════════════════════════════════════════")
    println()
    println("    $(ansi(c_in))Input A$(R)        $(ansi(c_state))State Q$(R)         $(ansi(c_out))Output B$(R)")
    println("    ─────────      ────────         ─────────")
    println("    $(ansi(c_in))Cu ore$(R)         $(ansi(c_state))T=900K$(R)          $(ansi(c_out))ingots$(R)")
    println("    $(ansi(c_in))Zn ore$(R)   ───▶  $(ansi(c_state))T=1000K$(R)  ───▶   $(ansi(c_out))sheets$(R)")
    println("    $(ansi(c_in))flux$(R)          $(ansi(c_state))T=1100K$(R)          $(ansi(c_out))tubes$(R)")
    println()
    println("    $(DIM)Profunctor t : A × Q → B × Q$(R)")
    println("    $(DIM)  (ore, temp) ↦ (product, next_temp)$(R)")
    println()
    println("    $(DIM)State category Q has morphisms:$(R)")
    println("    $(DIM)  heat: T → T+ΔT$(R)")
    println("    $(DIM)  cool: T → T-ΔT$(R)")
    println("    $(DIM)  hold: T → T$(R)")
    println()
end

# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

function main(; seed::Integer=42)
    println()
    println("  ╔════════════════════════════════════════════════════════════════════╗")
    println("  ║                    $(BOLD)BRASS FACTORY$(R)                                  ║")
    println("  ║   Cu + Zn → Brass  (Fokker-Planck of alloying)                     ║")
    println("  ╚════════════════════════════════════════════════════════════════════╝")
    
    render_brass_spectrum()
    
    # Create factory
    factory = BrassFactory("Cartridge Brass Works", 1273.0, 30.0, 2.0, 100)
    render_furnace(factory; seed=seed)
    render_production_run(factory, 10; seed=seed)
    
    render_factory_blanket(seed=seed)
    render_alloying_fokker_planck(seed=seed)
    render_factory_transducer(seed=seed)
    
    # Final brass bar
    println()
    c = brass_color(30.0)
    print("  ")
    for _ in 1:60
        print("$(ansi_bg(c))  $(R)")
    end
    println()
    println("  $(DIM)Color is everywhere: even in metal$(R)")
    println()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
