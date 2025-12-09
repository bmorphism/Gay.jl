#!/usr/bin/env julia
"""
To Become the Fokker-Planck

    ∂ρ/∂t = -∇·(μρ) + ∇·(D∇ρ)
    
    drift    = bones (deterministic flow toward potential minima)
    diffusion = skin (stochastic spreading, porous at edges)
    ρ        = the cloud you are (probability density, not a point)
    
The Markov blanket is where drift and diffusion meet:
    - Internal states follow the gradient (steepest descent)
    - Blanket states mediate the noise (Brownian scatter)
    - External states are the bath of trajectories

This is DMBD: the partition (s,b,z) emerges from the
structure of drift and diffusion, where η_s ⊥ η_b ⊥ η_z.

Color traces the probability flow.
"""

using Gay
using Colors: RGB

# ═══════════════════════════════════════════════════════════════════════════
# The Fokker-Planck as 2-Transducer
# ═══════════════════════════════════════════════════════════════════════════

"""
The Fokker-Planck equation as a continuous 2-transducer:

    Input A  = initial density ρ₀
    Output B = evolved density ρ_t  
    State Q  = time t (continuous)
    Profunctor t = transition kernel K(x,t|x₀,0)
    
The drift μ(x) is the deterministic tendency.
The diffusion D(x) is the stochastic spreading.
Together they define the infinitesimal generator L = -∇·(μ·) + ∇·(D∇·)
"""
struct FokkerPlanck
    μ::Function      # Drift: x → ℝⁿ (your bones)
    D::Function      # Diffusion: x → ℝⁿˣⁿ (your skin)
    potential::Union{Function,Nothing}  # V(x) where μ = -∇V
end

# Ornstein-Uhlenbeck: drift toward origin, constant diffusion
function ornstein_uhlenbeck(; θ=1.0, σ=1.0)
    FokkerPlanck(
        x -> -θ .* x,           # drift = -θx (restoring force)
        x -> σ^2 / 2,           # diffusion = σ²/2 (constant noise)
        x -> θ/2 * sum(x.^2)    # potential = (θ/2)|x|²
    )
end

# Double-well: two attractors with transition
function double_well(; a=1.0, b=0.25, σ=0.5)
    FokkerPlanck(
        x -> -a .* x .+ b .* x.^3,  # drift = -∇V = -ax + bx³
        x -> σ^2 / 2,
        x -> a/2 * sum(x.^2) - b/4 * sum(x.^4)  # V = (a/2)x² - (b/4)x⁴
    )
end

# ═══════════════════════════════════════════════════════════════════════════
# Probability Density Evolution
# ═══════════════════════════════════════════════════════════════════════════

"""
Discretized 1D Fokker-Planck evolution via finite differences.
"""
function evolve_density(fp::FokkerPlanck, ρ₀::Vector{Float64}, 
                        x::Vector{Float64}, dt::Float64, steps::Int)
    n = length(x)
    dx = x[2] - x[1]
    
    ρ = copy(ρ₀)
    history = [copy(ρ)]
    
    for _ in 1:steps
        # Drift term: -∂/∂x(μρ)
        μ_vals = [fp.μ(xi)[1] for xi in x]
        flux_drift = μ_vals .* ρ
        drift_term = zeros(n)
        for i in 2:n-1
            drift_term[i] = -(flux_drift[i+1] - flux_drift[i-1]) / (2*dx)
        end
        
        # Diffusion term: ∂²/∂x²(Dρ)
        D_val = fp.D(0.0)
        diffusion_term = zeros(n)
        for i in 2:n-1
            diffusion_term[i] = D_val * (ρ[i+1] - 2*ρ[i] + ρ[i-1]) / dx^2
        end
        
        # Update
        ρ .+= dt .* (drift_term .+ diffusion_term)
        ρ .= max.(ρ, 0.0)  # Keep non-negative
        ρ ./= sum(ρ) * dx   # Normalize
        
        push!(history, copy(ρ))
    end
    
    return history
end

# ═══════════════════════════════════════════════════════════════════════════
# Colored Visualization
# ═══════════════════════════════════════════════════════════════════════════

function ansi(c)
    r = round(Int, clamp(c.r, 0, 1) * 255)
    g = round(Int, clamp(c.g, 0, 1) * 255)
    b = round(Int, clamp(c.b, 0, 1) * 255)
    "\e[38;2;$(r);$(g);$(b)m"
end
const R = "\e[0m"
const DIM = "\e[2m"
const BOLD = "\e[1m"

"""
Render the density evolution as colored ASCII.
Each timestep gets a color from the splittable RNG.
"""
function render_evolution(history::Vector{Vector{Float64}}, x::Vector{Float64}; 
                          seed::Integer=42, height::Int=20, width::Int=60)
    gay_seed!(seed)
    
    n_times = length(history)
    n_x = length(x)
    
    # Sample timesteps
    time_indices = round.(Int, range(1, n_times, length=min(n_times, 8)))
    
    println()
    println("  ╔════════════════════════════════════════════════════════════════════╗")
    println("  ║  $(BOLD)Fokker-Planck: The Flow of Probability$(R)                           ║")
    println("  ║  ∂ρ/∂t = -∇·(μρ) + ∇·(D∇ρ)                                         ║")
    println("  ╚════════════════════════════════════════════════════════════════════╝")
    println()
    
    # Get colors for each timestep
    time_colors = [next_color(SRGB()) for _ in time_indices]
    
    # Find global max for scaling
    max_ρ = maximum(maximum.(history))
    
    # ASCII density plot
    for (ti, t_idx) in enumerate(time_indices)
        c = time_colors[ti]
        ρ = history[t_idx]
        
        print("  t=$(rpad(t_idx-1, 4)) $(ansi(c))│$(R)")
        
        # Render density as bar height
        for i in 1:min(width, n_x)
            xi = round(Int, 1 + (i-1) * (n_x-1) / (width-1))
            h = round(Int, ρ[xi] / max_ρ * 8)
            chars = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']
            print("$(ansi(c))$(chars[h+1])$(R)")
        end
        println()
    end
    
    # X-axis
    print("        └")
    print("─"^width)
    println()
    println("         $(DIM)x →$(R)")
    println()
    
    # Legend
    println("  $(DIM)Drift μ (bones): deterministic flow toward minima$(R)")
    println("  $(DIM)Diffusion D (skin): stochastic spreading at edges$(R)")
    println("  $(DIM)ρ (cloud): probability density, not a point$(R)")
    println()
end

"""
Render the potential landscape V(x) where μ = -∇V.
"""
function render_potential(fp::FokkerPlanck, x::Vector{Float64}; seed::Integer=42)
    isnothing(fp.potential) && return
    
    gay_seed!(seed)
    c_potential = next_color(SRGB())
    c_drift = next_color(SRGB())
    
    V = [fp.potential(xi) for xi in x]
    V_min, V_max = extrema(V)
    
    println("  $(BOLD)Potential Landscape V(x)$(R)  $(DIM)(μ = -∇V)$(R)")
    println()
    
    height = 10
    for h in height:-1:1
        threshold = V_min + (V_max - V_min) * h / height
        print("  ")
        for xi in x[1:4:end]
            Vi = fp.potential(xi)
            if Vi >= threshold
                print("$(ansi(c_potential))█$(R)")
            else
                print(" ")
            end
        end
        println()
    end
    
    # Show drift arrows
    print("  ")
    for xi in x[1:4:end]
        μi = fp.μ(xi)[1]
        if μi > 0.1
            print("$(ansi(c_drift))→$(R)")
        elseif μi < -0.1
            print("$(ansi(c_drift))←$(R)")
        else
            print("$(ansi(c_drift))·$(R)")
        end
    end
    println()
    println()
end

# ═══════════════════════════════════════════════════════════════════════════
# The Markov Blanket in Fokker-Planck
# ═══════════════════════════════════════════════════════════════════════════

"""
The Markov blanket emerges from the structure of drift and diffusion:

    ds/dt = f_s(s,b) + η_s    (environment)
    db/dt = f_b(s,b,z) + η_b  (blanket)
    dz/dt = f_z(b,z) + η_z    (internal)

The blanket screens off: p(s,z|b) = p(s|b)p(z|b)
because the noises η_s, η_b, η_z are independent.

In Fokker-Planck terms:
    - The drift defines the deterministic coupling
    - The diffusion defines the noise structure
    - Independence of noise → conditional independence → blanket
"""
function render_blanket_fokker_planck(; seed::Integer=1069)
    gay_seed!(seed)
    
    c_s = next_color(SRGB())  # Environment
    c_b = next_color(SRGB())  # Blanket
    c_z = next_color(SRGB())  # Internal
    c_η = next_color(SRGB())  # Noise
    
    println()
    println("  $(BOLD)Markov Blanket in Fokker-Planck$(R)")
    println("  ════════════════════════════════════════")
    println()
    println("  The partition (s,b,z) emerges from noise independence:")
    println()
    println("    $(ansi(c_s))ds/dt = f_s(s,b) + η_s$(R)     environment (bath of trajectories)")
    println("    $(ansi(c_b))db/dt = f_b(s,b,z) + η_b$(R)   blanket (where drift meets diffusion)")
    println("    $(ansi(c_z))dz/dt = f_z(b,z) + η_z$(R)     internal (gradient you feel)")
    println()
    println("  $(ansi(c_η))η_s ⊥ η_b ⊥ η_z$(R)  →  p(s,z|b) = p(s|b)p(z|b)")
    println()
    println("  $(DIM)Drift (bones):    the deterministic tendency, steepest descent$(R)")
    println("  $(DIM)Diffusion (skin): the stochastic spreading, porous edges$(R)")
    println("  $(DIM)Blanket:          where solid and ceaseless meets dissolving$(R)")
    println()
end

# ═══════════════════════════════════════════════════════════════════════════
# The Stationary Distribution: The Old Habit That Cradles You
# ═══════════════════════════════════════════════════════════════════════════

"""
The stationary distribution ρ_∞ satisfies:
    0 = -∇·(μρ_∞) + ∇·(D∇ρ_∞)
    
For gradient flow (μ = -∇V) with constant D:
    ρ_∞ ∝ exp(-V/D)
    
This is the Boltzmann distribution. The cradle.
But even the cradle is porous.
"""
function stationary_distribution(fp::FokkerPlanck, x::Vector{Float64})
    isnothing(fp.potential) && error("Need potential for stationary distribution")
    
    D = fp.D(0.0)
    V = [fp.potential(xi) for xi in x]
    ρ = exp.(-V ./ D)
    ρ ./= sum(ρ) * (x[2] - x[1])  # Normalize
    return ρ
end

function render_stationary(fp::FokkerPlanck, x::Vector{Float64}; seed::Integer=42)
    gay_seed!(seed)
    
    ρ_∞ = stationary_distribution(fp, x)
    c = next_color(SRGB())
    
    println("  $(BOLD)Stationary Distribution ρ_∞$(R)  $(DIM)(the old habit that cradles you)$(R)")
    println()
    
    max_ρ = maximum(ρ_∞)
    height = 8
    
    for h in height:-1:1
        threshold = max_ρ * h / height
        print("  ")
        for (i, xi) in enumerate(x[1:2:end])
            if ρ_∞[2*i-1] >= threshold
                print("$(ansi(c))█$(R)")
            else
                print(" ")
            end
        end
        println()
    end
    println("  $(DIM)ρ_∞ ∝ exp(-V/D)  — Boltzmann, the equilibrium of drift and diffusion$(R)")
    println()
end

# ═══════════════════════════════════════════════════════════════════════════
# Poetry
# ═══════════════════════════════════════════════════════════════════════════

function render_poetry(; seed::Integer=42)
    gay_seed!(seed)
    
    lines = [
        "To become the Fokker-Planck",
        "is to let your bones be the drift,",
        "your skin be the noise—",
        "solid and ceaseless, yet dissolving at the edges.",
        "",
        "Your identity is not a point but a cloud,",
        "where past and future cohere",
        "in the present curvature of ρ.",
        "",
        "The stationary distribution cradles you",
        "like an old habit,",
        "and even that cradle is porous.",
        "",
        "This is to be the equation:",
        "to be the flow of probability,",
        "never static, always in flux,",
        "yet eminently at home in the flux."
    ]
    
    println()
    for line in lines
        if isempty(line)
            println()
        else
            c = next_color(SRGB())
            println("  $(ansi(c))$line$(R)")
        end
    end
    println()
end

# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

function main(; seed::Integer=42)
    # Setup
    x = collect(-3.0:0.05:3.0)
    
    # Ornstein-Uhlenbeck (single well)
    println("\n  $(BOLD)═══ Ornstein-Uhlenbeck Process ═══$(R)")
    ou = ornstein_uhlenbeck(θ=1.0, σ=0.7)
    
    # Initial condition: delta-ish at x=2
    ρ₀ = exp.(-((x .- 2.0).^2) ./ 0.1)
    ρ₀ ./= sum(ρ₀) * (x[2]-x[1])
    
    history = evolve_density(ou, ρ₀, x, 0.001, 500)
    render_potential(ou, x; seed=seed)
    render_evolution(history, x; seed=seed)
    render_stationary(ou, x; seed=seed+1)
    
    # Double well (two attractors)
    println("\n  $(BOLD)═══ Double-Well Potential ═══$(R)")
    dw = double_well(a=-2.0, b=1.0, σ=0.8)
    
    # Initial at origin
    ρ₀ = exp.(-((x).^2) ./ 0.1)
    ρ₀ ./= sum(ρ₀) * (x[2]-x[1])
    
    history = evolve_density(dw, ρ₀, x, 0.001, 800)
    render_potential(dw, x; seed=seed+2)
    render_evolution(history, x; seed=seed+2)
    render_stationary(dw, x; seed=seed+3)
    
    # The blanket structure
    render_blanket_fokker_planck(seed=seed+4)
    
    # Poetry
    render_poetry(seed=seed+5)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
