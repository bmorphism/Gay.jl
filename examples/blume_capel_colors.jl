# # Blume-Capel Model with Colors: Spin-1 Monte Carlo
#
# The Blume-Capel model is a spin-1 generalization of the Ising model where
# spins take values φ ∈ {-1, 0, +1} with Hamiltonian:
#
#   H = J Σ_{<i,j>} φ_i φ_j + D Σ_i φ_i²
#
# The crystal-field coupling D controls the zero-field splitting:
# - D > 0: favors φ = 0 (neutral state)
# - D < 0: favors φ = ±1 (magnetic states)
#
# This creates a rich phase diagram with:
# - Second-order transition line at high T
# - First-order transition line at low T  
# - Tricritical point where they meet
#
# Reference: https://gitpages.physik.uni-wuerzburg.de/marqov/webmarqov/post/2020-05-15-blume-capel/
#
# ## Colors as Phase Indicators
#
# We assign colors to the three spin states:
# - φ = +1: White (ferromagnetic up)
# - φ =  0: Gray  (neutral/vacancy)
# - φ = -1: Black (ferromagnetic down)
#
# The Gay.jl SPI framework gives each Monte Carlo sweep a deterministic color,
# allowing reproducible visualization of the tricritical phenomenology.

using Pkg
Pkg.activate(dirname(dirname(@__FILE__)))

using Gay
using Gay: hash_color, xor_fingerprint
using Colors
using Statistics
using Random

# ═══════════════════════════════════════════════════════════════════════════════
# Blume-Capel Lattice
# ═══════════════════════════════════════════════════════════════════════════════

"""
    BlumeCapelLattice

2D square lattice Blume-Capel model with spin-1 degrees of freedom.

Fields:
- `spins`: L×L array with values in {-1, 0, +1}
- `L`: Linear system size
- `J`: Exchange coupling (default: -1.0 for ferromagnetic)
- `D`: Crystal-field coupling (controls zero-field splitting)
"""
struct BlumeCapelLattice
    spins::Matrix{Int8}
    L::Int
    J::Float64
    D::Float64
end

function BlumeCapelLattice(L::Int; J::Float64=-1.0, D::Float64=0.0, seed::Integer=42)
    rng = Random.MersenneTwister(seed)
    spins = rand(rng, Int8[-1, 0, 1], L, L)
    BlumeCapelLattice(spins, L, J, D)
end

"""
    energy(bc::BlumeCapelLattice)

Total Hamiltonian: H = J Σ φ_i φ_j + D Σ φ_i²
"""
function energy(bc::BlumeCapelLattice)
    E = 0.0
    L = bc.L
    
    # Exchange term: J Σ_{<i,j>} φ_i φ_j
    for i in 1:L, j in 1:L
        φ = bc.spins[i, j]
        # Right neighbor (periodic)
        φ_r = bc.spins[mod1(i+1, L), j]
        # Down neighbor (periodic)
        φ_d = bc.spins[i, mod1(j+1, L)]
        E += bc.J * φ * (φ_r + φ_d)
    end
    
    # Crystal field term: D Σ_i φ_i²
    for i in 1:L, j in 1:L
        E += bc.D * bc.spins[i, j]^2
    end
    
    return E
end

"""
    magnetization(bc::BlumeCapelLattice)

Order parameter: M = (1/N) Σ φ_i
"""
magnetization(bc::BlumeCapelLattice) = mean(bc.spins)

"""
    vacancy_fraction(bc::BlumeCapelLattice)

Fraction of neutral sites: n₀ = (1/N) Σ δ_{φ_i, 0}
Related to ³He concentration in He mixtures.
"""
vacancy_fraction(bc::BlumeCapelLattice) = count(==(0), bc.spins) / length(bc.spins)

"""
    quadrupole_moment(bc::BlumeCapelLattice)

Q = 1 - ⟨φ²⟩: measures departure from magnetic ordering.
"""
quadrupole_moment(bc::BlumeCapelLattice) = 1.0 - mean(bc.spins .^ 2)

# ═══════════════════════════════════════════════════════════════════════════════
# Spin-1 Metropolis Update with Colors
# ═══════════════════════════════════════════════════════════════════════════════

"""
    local_energy(bc, i, j)

Energy contribution from site (i,j) and its neighbors.
"""
function local_energy(bc::BlumeCapelLattice, i::Int, j::Int)
    L = bc.L
    φ = bc.spins[i, j]
    
    # Neighbors (periodic boundary)
    φ_sum = bc.spins[mod1(i+1, L), j] + bc.spins[mod1(i-1, L), j] +
            bc.spins[i, mod1(j+1, L)] + bc.spins[i, mod1(j-1, L)]
    
    # Local energy: exchange + crystal field
    bc.J * φ * φ_sum + bc.D * φ^2
end

"""
    blume_capel_sweep!(bc, ctx, β)

One Metropolis sweep over all sites with Gay color tracking.

For spin-1, we have two update options at each site:
1. φ → -φ (sign flip, like Ising)
2. φ → random new value from {-1, 0, +1}

We use option 2 for better ergodicity.
"""
function blume_capel_sweep!(bc::BlumeCapelLattice, ctx::GayMCContext, β::Float64)
    rng, sweep_color = gay_sweep!(ctx)
    L = bc.L
    n_accepted = 0
    
    for i in 1:L, j in 1:L
        φ_old = bc.spins[i, j]
        
        # Propose new spin from {-1, 0, +1}
        candidates = Int8[-1, 0, 1]
        φ_new = candidates[rand(rng, 1:3)]
        
        if φ_new == φ_old
            continue  # No change proposed
        end
        
        # Compute energy change
        φ_sum = bc.spins[mod1(i+1, L), j] + bc.spins[mod1(i-1, L), j] +
                bc.spins[i, mod1(j+1, L)] + bc.spins[i, mod1(j-1, L)]
        
        E_old = bc.J * φ_old * φ_sum + bc.D * φ_old^2
        E_new = bc.J * φ_new * φ_sum + bc.D * φ_new^2
        ΔE = E_new - E_old
        
        # Metropolis accept/reject
        if ΔE <= 0 || rand(rng) < exp(-β * ΔE)
            bc.spins[i, j] = φ_new
            n_accepted += 1
        end
    end
    
    acceptance_rate = n_accepted / (L * L)
    return (accepted=n_accepted, rate=acceptance_rate, color=sweep_color)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Wolff Cluster Update (for φ = ±1 only)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    wolff_cluster_update!(bc, ctx, β)

Wolff single-cluster update for the ±1 spins.
Note: This only builds clusters on magnetic sites, leaving vacancies (φ=0) untouched.
This maintains ergodicity when combined with Metropolis sweeps.
"""
function wolff_cluster_update!(bc::BlumeCapelLattice, ctx::GayMCContext, β::Float64)
    rng, sweep_color = gay_sweep!(ctx)
    L = bc.L
    
    # Find a random magnetic site to seed cluster
    magnetic_sites = [(i, j) for i in 1:L, j in 1:L if bc.spins[i, j] != 0]
    
    if isempty(magnetic_sites)
        return (cluster_size=0, color=sweep_color)  # All vacancies, nothing to do
    end
    
    seed_idx = rand(rng, 1:length(magnetic_sites))
    i0, j0 = magnetic_sites[seed_idx]
    seed_spin = bc.spins[i0, j0]
    
    # Bond probability: p = 1 - exp(2βJ) for ferromagnetic J < 0
    p_add = 1.0 - exp(2.0 * β * bc.J)  # J < 0, so 2βJ < 0, exp < 1, p > 0
    
    # Build cluster using stack-based flood fill
    cluster = Set{Tuple{Int,Int}}()
    stack = [(i0, j0)]
    push!(cluster, (i0, j0))
    
    while !isempty(stack)
        i, j = pop!(stack)
        
        # Check all neighbors
        for (di, dj) in [(1,0), (-1,0), (0,1), (0,-1)]
            ni, nj = mod1(i + di, L), mod1(j + dj, L)
            
            # Only add if same spin and not already in cluster
            if bc.spins[ni, nj] == seed_spin && (ni, nj) ∉ cluster
                if rand(rng) < p_add
                    push!(cluster, (ni, nj))
                    push!(stack, (ni, nj))
                end
            end
        end
    end
    
    # Flip all spins in cluster
    for (i, j) in cluster
        bc.spins[i, j] *= -1
    end
    
    return (cluster_size=length(cluster), color=sweep_color)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Visualization with ANSI Colors
# ═══════════════════════════════════════════════════════════════════════════════

"""
    visualize_lattice(bc; max_size=40)

Display lattice as ANSI colored blocks.
- φ = +1: White ██
- φ =  0: Gray  ▒▒
- φ = -1: Black ██
"""
function visualize_lattice(bc::BlumeCapelLattice; max_size::Int=40)
    L = min(bc.L, max_size)
    
    for j in 1:L
        print("  ")
        for i in 1:L
            φ = bc.spins[i, j]
            if φ == 1
                print("\e[47m  \e[0m")  # White background
            elseif φ == 0
                print("\e[100m  \e[0m") # Gray background
            else  # φ == -1
                print("\e[40m  \e[0m")  # Black background
            end
        end
        println()
    end
end

"""
    spin_color(φ::Integer)

Map spin value to RGB color for visualization.
"""
function spin_color(φ::Integer)
    if φ == 1
        RGB{Float32}(1.0f0, 1.0f0, 1.0f0)  # White
    elseif φ == 0
        RGB{Float32}(0.5f0, 0.5f0, 0.5f0)  # Gray
    else
        RGB{Float32}(0.0f0, 0.0f0, 0.0f0)  # Black
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Simulation with Colored Trajectory
# ═══════════════════════════════════════════════════════════════════════════════

"""
    simulate_blume_capel(; L=32, β=0.5, D=1.0, n_therm=1000, n_prod=1000, seed=42)

Run Blume-Capel simulation with Gay-colored sweeps.

Returns observables and color fingerprint for SPI verification.
"""
function simulate_blume_capel(; 
    L::Int=32, 
    β::Float64=0.5, 
    D::Float64=1.0,
    n_therm::Int=1000,
    n_prod::Int=1000,
    seed::Integer=42,
    verbose::Bool=true
)
    bc = BlumeCapelLattice(L; D=D, seed=seed)
    ctx = GayMCContext(seed)
    
    if verbose
        println("═" ^ 70)
        println("  BLUME-CAPEL MODEL SIMULATION")
        println("═" ^ 70)
        println("  L = $L, β = $β, D = $D, J = $(bc.J)")
        println("  Thermalization: $n_therm sweeps")
        println("  Production: $n_prod sweeps")
        println()
    end
    
    # Thermalization with hybrid Metropolis + Wolff
    if verbose
        println("  Thermalizing...")
        print("    ")
    end
    
    for i in 1:n_therm
        blume_capel_sweep!(bc, ctx, β)
        if i % 10 == 0
            wolff_cluster_update!(bc, ctx, β)
        end
        
        if verbose && i % (n_therm ÷ 20) == 0
            c = ctx.color_history[end]
            r = round(Int, red(c) * 255)
            g = round(Int, green(c) * 255)
            b = round(Int, blue(c) * 255)
            print("\e[48;2;$(r);$(g);$(b)m \e[0m")
        end
    end
    
    if verbose
        println()
        println("    E = $(round(energy(bc) / L^2, digits=4))")
        println("    M = $(round(magnetization(bc), digits=4))")
        println("    n₀ = $(round(vacancy_fraction(bc), digits=4))")
        println()
    end
    
    # Production
    energies = Float64[]
    magnetizations = Float64[]
    vacancies = Float64[]
    
    if verbose
        println("  Production sweeps...")
        print("    ")
    end
    
    for i in 1:n_prod
        result = blume_capel_sweep!(bc, ctx, β)
        if i % 10 == 0
            wolff_cluster_update!(bc, ctx, β)
        end
        
        # Measure every 5 sweeps to reduce autocorrelation
        if i % 5 == 0
            gay_measure!(ctx)
            push!(energies, energy(bc) / L^2)
            push!(magnetizations, abs(magnetization(bc)))
            push!(vacancies, vacancy_fraction(bc))
        end
        
        if verbose && i % (n_prod ÷ 50) == 0
            c = result.color
            r = round(Int, red(c) * 255)
            g = round(Int, green(c) * 255)
            b = round(Int, blue(c) * 255)
            print("\e[48;2;$(r);$(g);$(b)m \e[0m")
        end
    end
    
    if verbose
        println()
        println()
    end
    
    # Compute statistics
    E_mean = mean(energies)
    E_err = std(energies) / sqrt(length(energies))
    M_mean = mean(magnetizations)
    M_err = std(magnetizations) / sqrt(length(magnetizations))
    n0_mean = mean(vacancies)
    n0_err = std(vacancies) / sqrt(length(vacancies))
    
    if verbose
        println("  RESULTS:")
        println("  ─" ^ 35)
        println("    ⟨E⟩/N = $(round(E_mean, digits=5)) ± $(round(E_err, digits=5))")
        println("    ⟨|M|⟩ = $(round(M_mean, digits=5)) ± $(round(M_err, digits=5))")
        println("    ⟨n₀⟩  = $(round(n0_mean, digits=5)) ± $(round(n0_err, digits=5))")
        println()
        
        # Show final configuration
        println("  Final Configuration:")
        visualize_lattice(bc; max_size=min(L, 32))
        println()
        
        # Color fingerprint
        state = color_state(ctx)
        r = round(Int, red(state) * 255)
        g = round(Int, green(state) * 255)
        b = round(Int, blue(state) * 255)
        println("  Simulation Fingerprint: \e[48;2;$(r);$(g);$(b)m    \e[0m")
        println("  Total sweeps: $(ctx.sweep_count)")
        println("═" ^ 70)
    end
    
    return (
        E=E_mean, E_err=E_err,
        M=M_mean, M_err=M_err,
        n0=n0_mean, n0_err=n0_err,
        lattice=bc,
        ctx=ctx
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# Phase Diagram Exploration
# ═══════════════════════════════════════════════════════════════════════════════

"""
    explore_phase_diagram(; L=16, D_values=[-2:0.5:3;], β_values=[0.2:0.1:1.0;])

Scan the (D, β) phase diagram and visualize with colors.
"""
function explore_phase_diagram(;
    L::Int=16,
    D_values::AbstractVector=[-2.0:0.5:3.0;],
    β_values::AbstractVector=[0.2:0.2:1.2;],
    n_therm::Int=500,
    n_prod::Int=500,
    seed::Integer=42
)
    println()
    println("═" ^ 70)
    println("  BLUME-CAPEL PHASE DIAGRAM EXPLORATION")
    println("═" ^ 70)
    println("  L = $L, D ∈ [$(minimum(D_values)), $(maximum(D_values))], β ∈ [$(minimum(β_values)), $(maximum(β_values))]")
    println()
    
    # Header
    print("        ")
    for D in D_values
        print(" D=$(rpad(round(D, digits=1), 4))")
    end
    println()
    print("        ")
    for _ in D_values
        print("──────")
    end
    println()
    
    results = []
    
    for β in β_values
        print("  β=$(rpad(round(β, digits=2), 4))│")
        
        for D in D_values
            res = simulate_blume_capel(
                L=L, β=β, D=D, 
                n_therm=n_therm, n_prod=n_prod, 
                seed=seed, verbose=false
            )
            push!(results, (β=β, D=D, E=res.E, M=res.M, n0=res.n0))
            
            # Color by magnetization: high M = ferromagnetic (white), low M = paramagnetic (gray)
            M = res.M
            n0 = res.n0
            
            # Three-way coloring based on phase:
            # - High M (>0.5): ferromagnetic → blue/purple
            # - High n0 (>0.5): vacancy-dominated → gray
            # - Else: paramagnetic → red/orange
            if M > 0.5
                r, g, b = 100, 100, 255  # Blue for ferromagnetic
            elseif n0 > 0.5
                r, g, b = 128, 128, 128  # Gray for vacancy phase
            else
                r, g, b = 255, 100, 100  # Red for paramagnetic
            end
            
            print("\e[48;2;$(r);$(g);$(b)m  ")
            if M > 0.5
                print("F ")  # Ferromagnetic
            elseif n0 > 0.5
                print("V ")  # Vacancy
            else
                print("P ")  # Paramagnetic
            end
            print("\e[0m")
        end
        println()
    end
    
    println()
    println("  Legend: F = Ferromagnetic (blue), P = Paramagnetic (red), V = Vacancy-dominated (gray)")
    println("═" ^ 70)
    
    return results
end

# ═══════════════════════════════════════════════════════════════════════════════
# Critical Point Finder (Tricritical Exploration)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    find_tricritical_region(; L=24, seed=42)

Explore the tricritical region where the transition changes from second to first order.
The tricritical point is approximately at D_t ≈ 1.966, β_t ≈ 0.609 in 2D.
"""
function find_tricritical_region(; L::Int=24, seed::Integer=42)
    println()
    println("═" ^ 70)
    println("  TRICRITICAL POINT EXPLORATION")
    println("═" ^ 70)
    println()
    println("  The Blume-Capel model has a tricritical point where:")
    println("    - Second-order transition line (high T) meets")
    println("    - First-order transition line (low T)")
    println()
    println("  2D square lattice: D_t ≈ 1.966, β_t ≈ 0.609")
    println()
    
    # Fine scan near tricritical point
    D_values = [1.7:0.1:2.2;]
    β_values = [0.4:0.05:0.8;]
    
    explore_phase_diagram(
        L=L,
        D_values=D_values,
        β_values=β_values,
        n_therm=1000,
        n_prod=1000,
        seed=seed
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# Main Demo
# ═══════════════════════════════════════════════════════════════════════════════

function main()
    println()
    println("╔" * "═" ^ 68 * "╗")
    println("║" * " " ^ 18 * "BLUME-CAPEL MODEL WITH COLORS" * " " ^ 21 * "║")
    println("║" * " " ^ 68 * "║")
    println("║  Spin-1 lattice model: φ ∈ {-1, 0, +1}                              ║")
    println("║  H = J Σ φᵢφⱼ + D Σ φᵢ²                                            ║")
    println("╚" * "═" ^ 68 * "╝")
    println()
    
    # Demo 1: Single simulation
    println("  Demo 1: Single simulation at D=1.0, β=0.5")
    println()
    simulate_blume_capel(L=24, β=0.5, D=1.0, n_therm=500, n_prod=500)
    
    # Demo 2: Phase diagram exploration
    println()
    println("  Demo 2: Phase diagram exploration")
    explore_phase_diagram(L=16, n_therm=300, n_prod=300)
    
    println()
    println("  🎨 Each sweep gets a deterministic Gay color!")
    println("  🔬 Same seed → same colors → reproducible science")
    println()
end

# Export key functions
export BlumeCapelLattice, energy, magnetization, vacancy_fraction
export blume_capel_sweep!, wolff_cluster_update!
export simulate_blume_capel, explore_phase_diagram, find_tricritical_region
export visualize_lattice

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
