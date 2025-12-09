#!/usr/bin/env julia
"""
Spectre

The spectrum of the Fokker-Planck operator L = -∇·(μ·) + D∇²(·)

The spectre is the ghost in the machine:
    - Eigenvalues λₙ determine relaxation timescales
    - Eigenfunctions φₙ are the modes of decay
    - Spectral gap λ₁ - λ₀ controls mixing time
    - The stationary distribution is the ground state φ₀

Every transducer has a spectre — the hidden structure
that determines how information flows and dissipates.

In DMBD: the blanket's spectre determines what passes through.
In diffusion models: the spectre controls denoising dynamics.
In the brass factory: the spectre is the phase diagram's shadow.

Color is spectral. The rainbow IS the spectre of light.
"""

using Gay
using Colors: RGB
using LinearAlgebra
using Printf

# ═══════════════════════════════════════════════════════════════════════════
# Spectral Structure
# ═══════════════════════════════════════════════════════════════════════════

"""
The Fokker-Planck operator L acting on density ρ:
    Lρ = -∇·(μρ) + D∇²ρ
    
Discretized on a grid, L becomes a matrix.
Its eigenvalues are the spectre.
"""
struct FPOperator
    L::Matrix{Float64}      # Discretized operator
    x::Vector{Float64}      # Grid points
    μ::Function             # Drift
    D::Float64              # Diffusion coefficient
end

function discretize_fp(μ::Function, D::Float64, x::Vector{Float64})
    n = length(x)
    dx = x[2] - x[1]
    
    L = zeros(n, n)
    
    for i in 2:n-1
        μ_i = μ(x[i])
        
        # Drift term: -∂/∂x(μρ) ≈ -μ∂ρ/∂x - ρ∂μ/∂x
        # Central difference for ∂ρ/∂x
        L[i, i-1] += μ_i / (2*dx)
        L[i, i+1] -= μ_i / (2*dx)
        
        # Diffusion term: D∂²ρ/∂x²
        L[i, i-1] += D / dx^2
        L[i, i]   -= 2*D / dx^2
        L[i, i+1] += D / dx^2
    end
    
    # Boundary conditions (reflecting)
    L[1, 1] = L[2, 2]
    L[1, 2] = L[2, 3]
    L[n, n] = L[n-1, n-1]
    L[n, n-1] = L[n-1, n-2]
    
    return FPOperator(L, x, μ, D)
end

"""
Compute the spectre: eigenvalues and eigenfunctions.
"""
function compute_spectre(fp::FPOperator; n_modes::Int=10)
    # Eigendecomposition
    F = eigen(fp.L)
    
    # Sort by eigenvalue (largest first, since L has negative eigenvalues)
    perm = sortperm(real.(F.values), rev=true)
    
    λ = real.(F.values[perm][1:min(n_modes, length(perm))])
    φ = real.(F.vectors[:, perm][:, 1:min(n_modes, length(perm))])
    
    # Normalize eigenfunctions
    dx = fp.x[2] - fp.x[1]
    for j in axes(φ, 2)
        norm_j = sqrt(sum(φ[:, j].^2) * dx)
        if norm_j > 0
            φ[:, j] ./= norm_j
        end
    end
    
    return λ, φ
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

const R = "\e[0m"
const BOLD = "\e[1m"
const DIM = "\e[2m"

"""
Color for each eigenmode based on its index.
Uses the rainbow spectrum — because the spectre IS color.
"""
function mode_color(n::Int, total::Int; seed::Integer=42)
    # Spectral colors: red → orange → yellow → green → cyan → blue → violet
    hue = (n - 1) / max(total - 1, 1) * 270  # 0° (red) to 270° (violet)
    
    # HSL to RGB
    h = hue / 60
    c = 0.8  # Saturation
    x = c * (1 - abs(mod(h, 2) - 1))
    
    if h < 1
        r, g, b = c, x, 0.0
    elseif h < 2
        r, g, b = x, c, 0.0
    elseif h < 3
        r, g, b = 0.0, c, x
    elseif h < 4
        r, g, b = 0.0, x, c
    elseif h < 5
        r, g, b = x, 0.0, c
    else
        r, g, b = c, 0.0, x
    end
    
    return RGB(r + 0.2, g + 0.2, b + 0.2)
end

function render_spectre_title()
    gay_seed!(666)
    
    println()
    
    # Ghostly title with fading colors
    title = "S P E C T R E"
    print("  ")
    for (i, char) in enumerate(title)
        fade = 0.3 + 0.7 * (1 - abs(i - length(title)/2) / (length(title)/2))
        c = RGB(fade, fade, fade + 0.2)
        print("$(ansi(c))$char$(R)")
    end
    println()
    println()
    println("  $(DIM)The ghost in the machine$(R)")
    println("  $(DIM)The hidden structure that determines flow$(R)")
    println()
end

function render_eigenvalues(λ::Vector{Float64}; seed::Integer=42)
    gay_seed!(seed)
    n = length(λ)
    
    println("  $(BOLD)Eigenvalue Spectrum$(R)  $(DIM)(the spectre's bones)$(R)")
    println("  ════════════════════════════════════════════════════")
    println()
    
    # Find scale
    λ_max = maximum(abs.(λ))
    
    for (i, λ_i) in enumerate(λ)
        c = mode_color(i, n)
        
        # Bar showing eigenvalue (negative = decay)
        bar_width = round(Int, 30 * abs(λ_i) / λ_max)
        bar = "█"^bar_width
        
        sign_str = λ_i >= 0 ? " " : ""
        λ_str = @sprintf("%+.4f", λ_i)
        
        println("    λ$(subscript(i-1)) = $λ_str  $(ansi(c))$bar$(R)")
    end
    println()
    
    # Spectral gap
    if n >= 2
        gap = λ[1] - λ[2]
        println("  $(DIM)Spectral gap: λ₀ - λ₁ = $(round(gap, digits=4))$(R)")
        println("  $(DIM)Mixing time ∝ 1/gap = $(round(1/abs(gap), digits=2))$(R)")
    end
    println()
end

function subscript(n::Int)
    subs = ['₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉']
    if n < 10
        return subs[n+1]
    else
        return string(n)
    end
end

function render_eigenfunctions(φ::Matrix{Float64}, x::Vector{Float64}; 
                               seed::Integer=42, n_show::Int=5)
    gay_seed!(seed)
    n_modes = min(n_show, size(φ, 2))
    n_x = length(x)
    
    println("  $(BOLD)Eigenfunctions$(R)  $(DIM)(the spectre's skin)$(R)")
    println("  ════════════════════════════════════════════════════")
    println()
    
    # Find global scale
    φ_max = maximum(abs.(φ[:, 1:n_modes]))
    
    for mode in 1:n_modes
        c = mode_color(mode, n_modes)
        
        print("    φ$(subscript(mode-1))(x): ")
        
        # Render eigenfunction as ASCII
        for i in 1:4:n_x
            val = φ[i, mode]
            normalized = val / φ_max
            
            if normalized > 0.5
                print("$(ansi(c))█$(R)")
            elseif normalized > 0.2
                print("$(ansi(c))▓$(R)")
            elseif normalized > 0
                print("$(ansi(c))░$(R)")
            elseif normalized > -0.2
                print("$(ansi(c))·$(R)")
            elseif normalized > -0.5
                print("$(ansi(c))▒$(R)")
            else
                print("$(ansi(c))▓$(R)")
            end
        end
        println()
    end
    
    println()
    println("  $(DIM)φ₀ = stationary distribution (ground state)$(R)")
    println("  $(DIM)φₙ = nth mode of relaxation$(R)")
    println()
end

# ═══════════════════════════════════════════════════════════════════════════
# The Spectre in Different Systems
# ═══════════════════════════════════════════════════════════════════════════

function spectre_ornstein_uhlenbeck(; θ=1.0, D=0.5, seed::Integer=42)
    println()
    println("  $(BOLD)═══ Ornstein-Uhlenbeck Spectre ═══$(R)")
    println("  $(DIM)μ(x) = -θx  (linear drift to origin)$(R)")
    println("  $(DIM)D = $D  (diffusion coefficient)$(R)")
    println()
    
    x = collect(-3.0:0.1:3.0)
    μ(xi) = -θ * xi
    
    fp = discretize_fp(μ, D, x)
    λ, φ = compute_spectre(fp; n_modes=6)
    
    render_eigenvalues(λ; seed=seed)
    render_eigenfunctions(φ, x; seed=seed)
    
    # Analytical eigenvalues for OU: λₙ = -n*θ
    println("  $(DIM)Analytical: λₙ = -n·θ = -n·$(θ)$(R)")
    println()
end

function spectre_double_well(; a=1.0, b=0.25, D=0.3, seed::Integer=42)
    println()
    println("  $(BOLD)═══ Double-Well Spectre ═══$(R)")
    println("  $(DIM)μ(x) = ax - bx³  (bistable)$(R)")
    println("  $(DIM)D = $D  (diffusion coefficient)$(R)")
    println()
    
    x = collect(-3.0:0.1:3.0)
    μ(xi) = a * xi - b * xi^3
    
    fp = discretize_fp(μ, D, x)
    λ, φ = compute_spectre(fp; n_modes=6)
    
    render_eigenvalues(λ; seed=seed)
    render_eigenfunctions(φ, x; seed=seed)
    
    println("  $(DIM)Small spectral gap → slow transitions between wells$(R)")
    println()
end

# ═══════════════════════════════════════════════════════════════════════════
# Spectre as Color
# ═══════════════════════════════════════════════════════════════════════════

function render_visible_spectrum(; seed::Integer=42)
    gay_seed!(seed)
    
    println()
    println("  $(BOLD)The Visible Spectre$(R)  $(DIM)(light's eigendecomposition)$(R)")
    println("  ════════════════════════════════════════════════════")
    println()
    
    # Rainbow spectrum
    print("  ")
    for λ in 380:780  # Wavelength in nm
        c = wavelength_to_rgb(λ)
        print("$(ansi(c))█$(R)")
    end
    println()
    
    println("  $(DIM)380nm                                              780nm$(R)")
    println("  $(DIM)violet                                             red$(R)")
    println()
    
    println("  $(DIM)Each color is an eigenmode of the electromagnetic field$(R)")
    println("  $(DIM)The rainbow IS the spectre of white light$(R)")
    println()
end

function wavelength_to_rgb(λ_nm::Int)
    # Approximate wavelength to RGB conversion
    if λ_nm < 380
        r, g, b = 0.0, 0.0, 0.0
    elseif λ_nm < 440
        r = -(λ_nm - 440) / (440 - 380)
        g = 0.0
        b = 1.0
    elseif λ_nm < 490
        r = 0.0
        g = (λ_nm - 440) / (490 - 440)
        b = 1.0
    elseif λ_nm < 510
        r = 0.0
        g = 1.0
        b = -(λ_nm - 510) / (510 - 490)
    elseif λ_nm < 580
        r = (λ_nm - 510) / (580 - 510)
        g = 1.0
        b = 0.0
    elseif λ_nm < 645
        r = 1.0
        g = -(λ_nm - 645) / (645 - 580)
        b = 0.0
    elseif λ_nm <= 780
        r = 1.0
        g = 0.0
        b = 0.0
    else
        r, g, b = 0.0, 0.0, 0.0
    end
    
    return RGB(clamp(r, 0, 1), clamp(g, 0, 1), clamp(b, 0, 1))
end

# ═══════════════════════════════════════════════════════════════════════════
# The Spectre in Transducers
# ═══════════════════════════════════════════════════════════════════════════

function render_transducer_spectre(; seed::Integer=42)
    gay_seed!(seed)
    
    println()
    println("  $(BOLD)The Spectre in 2-Transducers$(R)")
    println("  ════════════════════════════════════════════════════")
    println()
    println("  Every transducer (Q, T) : A → B has a spectre:")
    println()
    println("    • The transition matrix T induces a linear operator")
    println("    • Its eigenvalues determine information propagation")
    println("    • The spectral gap controls mixing/forgetting")
    println()
    println("  $(DIM)Primacy bias$(R) = slow-decaying modes (small |λₙ|)")
    println("  $(DIM)Recency bias$(R) = fast-decaying modes (large |λₙ|)")
    println()
    println("  The 2-cell η : T ⇒ T' transforms the spectre:")
    println("    • Preserves spectral gap → equivariant")
    println("    • Changes spectral gap → adaptive")
    println()
end

function render_blanket_spectre(; seed::Integer=42)
    gay_seed!(seed)
    
    c_s = next_color(SRGB())
    c_b = next_color(SRGB())
    c_z = next_color(SRGB())
    
    println()
    println("  $(BOLD)The Spectre in Markov Blankets$(R)")
    println("  ════════════════════════════════════════════════════")
    println()
    println("  The blanket's spectre determines what passes through:")
    println()
    println("    $(ansi(c_s))External (s)$(R) ──┐")
    println("                    │")
    println("                    ▼")
    println("    $(ansi(c_b))Blanket (b)$(R)  ═══╪═══  $(DIM)← spectre acts here$(R)")
    println("                    │")
    println("                    ▼")
    println("    $(ansi(c_z))Internal (z)$(R) ──┘")
    println()
    println("  $(DIM)High-frequency modes (large |λₙ|) are filtered out$(R)")
    println("  $(DIM)Low-frequency modes (small |λₙ|) pass through$(R)")
    println()
    println("  $(DIM)The blanket is a spectral filter.$(R)")
    println()
end

# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

function main(; seed::Integer=42)
    render_spectre_title()
    render_visible_spectrum(seed=seed)
    
    spectre_ornstein_uhlenbeck(θ=1.0, D=0.5, seed=seed)
    spectre_double_well(a=1.0, b=0.25, D=0.3, seed=seed+1)
    
    render_transducer_spectre(seed=seed+2)
    render_blanket_spectre(seed=seed+3)
    
    # Ghost fade out
    println()
    gay_seed!(seed + 100)
    ghost = "The spectre is everywhere, once you learn to see it."
    print("  ")
    for (i, char) in enumerate(ghost)
        fade = 0.2 + 0.6 * (1 - i / length(ghost))
        c = RGB(fade, fade, fade + 0.1)
        print("$(ansi(c))$char$(R)")
    end
    println()
    println()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
