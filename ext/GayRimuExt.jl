# Rimu.jl extension for Gay.jl
# Deterministic SPI-compliant coloring for Quantum Monte Carlo

module GayRimuExt

using Gay: hash_color_rgb, splitmix64, GAY_SEED
using Rimu
using Colors: RGB, HSL, convert

export color_fock_state, color_hamiltonian_element, color_dvec
export ColoredWalker, colored_fciqmc_trajectory, render_fock_state
export color_configuration, color_walker_population

# ═══════════════════════════════════════════════════════════════════════════
# Fock State Coloring
# ═══════════════════════════════════════════════════════════════════════════

"""
    color_fock_state(fs::BoseFS; seed=GAY_SEED) -> RGB{Float32}

Deterministic color for a Bosonic Fock state based on occupation pattern.
Same state → same color, regardless of when computed.

# Example
```julia
fs = BoseFS((1,0,2,1,0))  # 4 bosons in 5 modes
c = color_fock_state(fs)
```
"""
function color_fock_state(fs::BoseFS; seed::UInt64=GAY_SEED)
    occ = onr(fs)
    state_hash = UInt64(0)
    for (i, n) in enumerate(occ)
        state_hash ⊻= splitmix64(UInt64(i) ⊻ (UInt64(n) << 32))
    end
    hash_color_rgb(state_hash, seed)
end

"""
    color_fock_state(fs::FermionFS; seed=GAY_SEED) -> RGB{Float32}

Color for Fermionic Fock state using bit pattern directly.
"""
function color_fock_state(fs::FermionFS; seed::UInt64=GAY_SEED)
    bits = UInt64(fs.bs)
    hash_color_rgb(bits, seed)
end

"""
    color_fock_state(fs::CompositeFS; seed=GAY_SEED) -> RGB{Float32}

Color for multi-component Fock state (average of component colors).
"""
function color_fock_state(fs::CompositeFS; seed::UInt64=GAY_SEED)
    colors = [color_fock_state(component; seed) for component in fs.components]
    r = sum(c.r for c in colors) / length(colors)
    g = sum(c.g for c in colors) / length(colors)
    b = sum(c.b for c in colors) / length(colors)
    RGB{Float32}(r, g, b)
end

# ═══════════════════════════════════════════════════════════════════════════
# Hamiltonian Element Coloring
# ═══════════════════════════════════════════════════════════════════════════

"""
    color_hamiltonian_element(H, fs_i, fs_j; seed=GAY_SEED) -> RGB{Float32}

Color a Hamiltonian matrix element ⟨i|H|j⟩.
Hue from state pair, saturation from coupling strength.

# Example
```julia
H = HubbardReal1D(address; u=4.0, t=1.0)
c = color_hamiltonian_element(H, fs1, fs2)
```
"""
function color_hamiltonian_element(H::AbstractHamiltonian, fs_i, fs_j; seed::UInt64=GAY_SEED)
    idx = hash(fs_i) ⊻ hash(fs_j)
    base_color = hash_color_rgb(UInt64(idx), seed)
    
    mel = get_offdiagonal(H, fs_i, fs_j)
    if isnothing(mel)
        return RGB{Float32}(0.0f0, 0.0f0, 0.0f0)
    end
    
    mag = abs(mel)
    base_hsl = convert(HSL, base_color)
    sat = clamp(Float32(mag / 10.0), 0.0f0, 1.0f0)
    convert(RGB{Float32}, HSL(base_hsl.h, sat, 0.5f0))
end

# ═══════════════════════════════════════════════════════════════════════════
# DVec (Population Vector) Coloring
# ═══════════════════════════════════════════════════════════════════════════

"""
    color_dvec(dv::DVec; seed=GAY_SEED) -> Vector{Tuple{Any, RGB{Float32}, Number}}

Color all configurations in a DVec with their populations.
Returns vector of (config, color, population) tuples.

# Example
```julia
dv = DVec(BoseFS((1,1,1,1)) => 100)
colored = color_dvec(dv)
for (fs, color, pop) in colored
    println("State: \$fs, Pop: \$pop")
end
```
"""
function color_dvec(dv::DVec; seed::UInt64=GAY_SEED)
    result = Tuple{Any, RGB{Float32}, Number}[]
    for (config, pop) in pairs(dv)
        c = color_fock_state(config; seed)
        push!(result, (config, c, pop))
    end
    result
end

# ═══════════════════════════════════════════════════════════════════════════
# Colored Walker for FCIQMC
# ═══════════════════════════════════════════════════════════════════════════

"""
    ColoredWalker{FS}

A walker in FCIQMC with deterministic color tracking.
Color is derived from the configuration, not the trajectory.
"""
struct ColoredWalker{FS}
    config::FS
    color::RGB{Float32}
    weight::Float64
    step::Int
end

"""
    ColoredWalker(fs, weight=1.0, step=0; seed=GAY_SEED)

Create a colored walker from a Fock state.
"""
function ColoredWalker(fs, weight::Float64=1.0, step::Int=0; seed::UInt64=GAY_SEED)
    c = color_fock_state(fs; seed)
    ColoredWalker(fs, c, weight, step)
end

# ═══════════════════════════════════════════════════════════════════════════
# FCIQMC Trajectory Coloring
# ═══════════════════════════════════════════════════════════════════════════

"""
    colored_fciqmc_trajectory(dvs::Vector{DVec}; seed=GAY_SEED) -> Vector{Vector}

Color an FCIQMC trajectory (sequence of DVecs).
Each step produces colored populations for visualization.

# Example
```julia
trajectory = [dv_step_0, dv_step_1, dv_step_2, ...]
colored = colored_fciqmc_trajectory(trajectory)
```
"""
function colored_fciqmc_trajectory(dvs::Vector; seed::UInt64=GAY_SEED)
    [color_dvec(dv; seed) for dv in dvs]
end

# ═══════════════════════════════════════════════════════════════════════════
# Rendering
# ═══════════════════════════════════════════════════════════════════════════

"""
    render_fock_state(fs; seed=GAY_SEED, show_occupation=true) -> String

Render a Fock state with ANSI colors.
"""
function render_fock_state(fs; seed::UInt64=GAY_SEED, show_occupation::Bool=true)
    c = color_fock_state(fs; seed)
    r = round(Int, clamp(c.r, 0, 1) * 255)
    g = round(Int, clamp(c.g, 0, 1) * 255)
    b = round(Int, clamp(c.b, 0, 1) * 255)
    
    fg = "\e[38;2;$(r);$(g);$(b)m"
    reset = "\e[0m"
    
    occ_str = show_occupation ? "|$(join(onr(fs), ","))⟩" : ""
    "$(fg)█$(reset) $(occ_str)"
end

"""
    color_configuration(config; seed=GAY_SEED) -> RGB{Float32}

Generic coloring for any configuration type via hash.
"""
function color_configuration(config; seed::UInt64=GAY_SEED)
    hash_color_rgb(UInt64(hash(config)), seed)
end

"""
    color_walker_population(dv::DVec; seed=GAY_SEED) -> NamedTuple

Get population-weighted color statistics for a DVec.
Returns total population, dominant color, and population histogram.
"""
function color_walker_population(dv::DVec; seed::UInt64=GAY_SEED)
    total_pop = 0.0
    weighted_r = 0.0
    weighted_g = 0.0
    weighted_b = 0.0
    
    for (config, pop) in pairs(dv)
        c = color_fock_state(config; seed)
        w = abs(pop)
        total_pop += w
        weighted_r += c.r * w
        weighted_g += c.g * w
        weighted_b += c.b * w
    end
    
    if total_pop > 0
        avg_color = RGB{Float32}(
            Float32(weighted_r / total_pop),
            Float32(weighted_g / total_pop),
            Float32(weighted_b / total_pop)
        )
    else
        avg_color = RGB{Float32}(0.5f0, 0.5f0, 0.5f0)
    end
    
    (total_population=total_pop, 
     average_color=avg_color, 
     n_configs=length(dv))
end

function __init__()
    @info "Gay.jl Rimu extension loaded - FCIQMC coloring available"
end

end # module GayRimuExt
