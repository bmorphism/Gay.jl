# # Rimu.jl Quantum Monte Carlo Integration
#
# Gay.jl provides SPI-compliant coloring for FCIQMC simulations,
# enabling deterministic visualization of quantum many-body states.

# ## Setup

using Gay
using Rimu
using Colors

# ## Fock State Coloring
#
# Each Fock state gets a unique color based on its occupation pattern.
# The same state always produces the same color (SPI guarantee).

# ### Bosonic Fock States

fs_bose = BoseFS((1, 0, 2, 1, 0))  # 4 bosons in 5 modes

c = color_fock_state(fs_bose; seed=GAY_SEED)
println("BoseFS |1,0,2,1,0⟩ color: RGB($(round(c.r, digits=3)), $(round(c.g, digits=3)), $(round(c.b, digits=3)))")

# Verify SPI: same state → same color
c2 = color_fock_state(fs_bose; seed=GAY_SEED)
@assert c == c2 "SPI violation!"
println("SPI verified: repeated call produces identical color")

# ### Fermionic Fock States

fs_fermi = FermionFS((1, 1, 0, 1, 0, 0, 1, 0))  # 4 fermions in 8 modes

c_fermi = color_fock_state(fs_fermi; seed=GAY_SEED)
println("\nFermionFS color: RGB($(round(c_fermi.r, digits=3)), $(round(c_fermi.g, digits=3)), $(round(c_fermi.b, digits=3)))")

# ## DVec Population Coloring
#
# Color all configurations in a population vector with their weights.

initial_state = BoseFS((2, 2, 0, 0))
dv = DVec(initial_state => 100)

# Add some excited states
dv[BoseFS((1, 2, 1, 0))] = 50
dv[BoseFS((1, 1, 2, 0))] = 30
dv[BoseFS((0, 2, 2, 0))] = 20

# Color all states
colored = color_dvec(dv; seed=GAY_SEED)

println("\n=== DVec Population ===")
for (fs, color, pop) in colored
    rendered = render_fock_state(fs; seed=GAY_SEED)
    println("  $rendered  pop=$(Int(pop))")
end

# ## Population Statistics
#
# Get aggregate color statistics for the walker population:

stats = color_walker_population(dv; seed=GAY_SEED)
println("\n=== Population Statistics ===")
println("Total population: $(stats.total_population)")
println("Number of configs: $(stats.n_configs)")
println("Average color: RGB($(round(stats.average_color.r, digits=3)), $(round(stats.average_color.g, digits=3)), $(round(stats.average_color.b, digits=3)))")

# ## Hamiltonian Matrix Element Coloring
#
# Color coupling strengths between states:

H = HubbardReal1D(initial_state; u=4.0, t=1.0)

println("\n=== Hamiltonian Elements ===")
println("Diagonal: ⟨gs|H|gs⟩ = $(diagonal_element(H, initial_state))")

# ## FCIQMC Trajectory Coloring
#
# Track how walker populations evolve with consistent colors:

trajectory = [
    DVec(BoseFS((2,2,0,0)) => 100),
    DVec(BoseFS((2,2,0,0)) => 80, BoseFS((1,2,1,0)) => 30),
    DVec(BoseFS((2,2,0,0)) => 60, BoseFS((1,2,1,0)) => 45, BoseFS((1,1,2,0)) => 15)
]

colored_traj = colored_fciqmc_trajectory(trajectory; seed=GAY_SEED)

println("\n=== FCIQMC Trajectory ===")
for (step, colored_dv) in enumerate(colored_traj)
    println("Step $step: $(length(colored_dv)) configurations")
end

# ## Key Properties
#
# 1. **Occupation-based hashing**: Color derived from `(mode, occupation)` pairs
# 2. **Component averaging**: CompositeFS averages component colors
# 3. **Population weighting**: DVec statistics weighted by walker count
# 4. **SPI guarantee**: Same Fock state → same color, always
