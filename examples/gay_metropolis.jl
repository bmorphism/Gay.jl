# # Gay Metropolis: A Colorful Journey Through Monte Carlo
#
# This is a [Literate.jl](https://github.com/fredrikekre/Literate.jl) tutorial
# exploring the **Gay Metropolis** algorithm - where every Monte Carlo step
# gets a deterministic, reproducible color through Strong Parallelism Invariance (SPI).
#
# ## The Story
#
# In 1953, Nicholas Metropolis and colleagues published their famous algorithm
# for sampling from probability distributions. 70 years later, we add color.
#
# Why color? Because:
# - **Reproducibility**: Same seed → same colors, regardless of execution order
# - **Visualization**: See the Monte Carlo walk through color space
# - **Debugging**: Each step has a unique fingerprint
# - **Parallelism**: Workers get independent but deterministic streams
#
# ## Setup

using Pkg
Pkg.activate(dirname(dirname(@__FILE__)))

using Gay
using Colors
using Statistics

# ## The GayMCContext
#
# At the heart of Gay Metropolis is the `GayMCContext` - a splittable random
# number generator that provides Strong Parallelism Invariance (SPI).
#
# SPI means: **the random stream is deterministic regardless of parallel execution order**.
#
# This is crucial for reproducible science. Traditional RNGs depend on the order
# of calls, making parallel simulations non-reproducible. SPI fixes this.

seed = 42
ctx = GayMCContext(seed)

# Each context tracks:
# - `sweep_count`: how many MC sweeps we've done
# - `measure_count`: how many measurements
# - `checkpoint_count`: how many checkpoints
# - `color_history`: the color of each sweep

println("Initial state:")
println("  sweep_count = $(ctx.sweep_count)")
println("  seed = $(ctx.seed)")

# ## The Ising Model (Simplified)
#
# Let's simulate a 1D Ising chain with N spins.
# Each spin is +1 or -1, and the energy is:
#
# $$E = -J \sum_i s_i s_{i+1}$$
#
# We'll use Gay Metropolis to sample from the Boltzmann distribution at inverse
# temperature β.

struct IsingChain
    spins::Vector{Int}
    J::Float64
end

function IsingChain(N::Int; J::Float64=1.0)
    spins = rand([-1, 1], N)
    IsingChain(spins, J)
end

function energy(chain::IsingChain)
    E = 0.0
    N = length(chain.spins)
    for i in 1:N
        j = mod1(i + 1, N)  # periodic boundary
        E -= chain.J * chain.spins[i] * chain.spins[j]
    end
    E
end

function magnetization(chain::IsingChain)
    mean(chain.spins)
end

# ## Gay Metropolis Sweep
#
# Here's where the magic happens. Each sweep:
# 1. Picks a random spin to flip
# 2. Computes the energy difference ΔE
# 3. Uses `gay_metropolis!` to accept/reject with color
# 4. The color encodes this exact moment in the simulation

function gay_ising_sweep!(chain::IsingChain, ctx::GayMCContext, β::Float64)
    N = length(chain.spins)
    
    # Get RNG and color for this sweep
    rng, sweep_color = gay_sweep!(ctx)
    
    # Pick a random spin
    i = rand(rng, 1:N)
    
    # Compute ΔE for flipping spin i
    left = mod1(i - 1, N)
    right = mod1(i + 1, N)
    ΔE = 2 * chain.J * chain.spins[i] * (chain.spins[left] + chain.spins[right])
    
    # Metropolis accept/reject (colored!)
    if ΔE <= 0
        chain.spins[i] *= -1
        accepted = true
    else
        u = rand(rng)
        if u < exp(-β * ΔE)
            chain.spins[i] *= -1
            accepted = true
        else
            accepted = false
        end
    end
    
    (accepted=accepted, color=sweep_color, ΔE=ΔE, site=i)
end

# ## Running the Simulation
#
# Let's run 100 sweeps and watch the colors flow.

N = 16  # spins
β = 0.5  # inverse temperature (above critical)
chain = IsingChain(N)
ctx = GayMCContext(seed)

println("\n🎲 Gay Metropolis Simulation")
println("  N = $N spins, β = $β")
println("  Initial energy = $(energy(chain))")
println("  Initial magnetization = $(magnetization(chain))")
println()

# Thermalization
n_therm = 50
for _ in 1:n_therm
    gay_ising_sweep!(chain, ctx, β)
end
println("After $(n_therm) thermalization sweeps:")
println("  Energy = $(energy(chain))")
println("  Magnetization = $(magnetization(chain))")

# Production with colored output
println("\nProduction sweeps (showing colors):")
print("  ")

n_prod = 50
energies = Float64[]
mags = Float64[]

for i in 1:n_prod
    result = gay_ising_sweep!(chain, ctx, β)
    push!(energies, energy(chain))
    push!(mags, magnetization(chain))
    
    # Print colored block
    c = result.color
    r = round(Int, red(c) * 255)
    g = round(Int, green(c) * 255)
    b = round(Int, blue(c) * 255)
    if result.accepted
        print("\e[48;2;$(r);$(g);$(b)m \e[0m")
    else
        print("\e[48;2;$(r);$(g);$(b)m·\e[0m")
    end
end
println()

println("\nFinal statistics:")
println("  ⟨E⟩ = $(mean(energies)) ± $(std(energies) / sqrt(n_prod))")
println("  ⟨M⟩ = $(mean(mags)) ± $(std(mags) / sqrt(n_prod))")

# ## The Color History
#
# Every sweep is recorded in `ctx.color_history`. This is the visual fingerprint
# of our simulation.

println("\n🎨 Color History (first 20 sweeps):")
print("  ")
for i in 1:min(20, length(ctx.color_history))
    c = ctx.color_history[i]
    r = round(Int, red(c) * 255)
    g = round(Int, green(c) * 255)
    b = round(Int, blue(c) * 255)
    print("\e[48;2;$(r);$(g);$(b)m  \e[0m")
end
println()

# The state color is the average of recent sweeps:
state = color_state(ctx)
r = round(Int, red(state) * 255)
g = round(Int, green(state) * 255)
b = round(Int, blue(state) * 255)
println("\n  State color: \e[48;2;$(r);$(g);$(b)m    \e[0m")

# ## Parallel Workers
#
# The real power of SPI: multiple workers with independent, deterministic streams.
# Each worker can run in parallel, and the results are reproducible.

println("\n⚡ Parallel Workers Demo")
n_workers = 4
workers = gay_workers(seed, n_workers)
chains = [IsingChain(N) for _ in 1:n_workers]

for (i, (ctx, chain)) in enumerate(zip(workers, chains))
    # Each worker does 30 sweeps
    for _ in 1:30
        gay_ising_sweep!(chain, ctx, β)
    end
    
    # Show color trail
    print("  Worker $i: ")
    for j in max(1, length(ctx.color_history)-14):length(ctx.color_history)
        c = ctx.color_history[j]
        r = round(Int, red(c) * 255)
        g = round(Int, green(c) * 255)
        b = round(Int, blue(c) * 255)
        print("\e[48;2;$(r);$(g);$(b)m \e[0m")
    end
    
    state = color_state(ctx)
    r = round(Int, red(state) * 255)
    g = round(Int, green(state) * 255)
    b = round(Int, blue(state) * 255)
    println(" → \e[48;2;$(r);$(g);$(b)m  \e[0m E=$(energy(chain))")
end

# ## Checkpointing
#
# GayMC supports HDF5-compatible checkpointing. The checkpoint includes:
# - All counters (sweep, measure, checkpoint)
# - The color history
# - The checkpoint's own color!

println("\n💾 Checkpoint Demo")
cp = gay_checkpoint(workers[1])
println("  Checkpoint keys: $(keys(cp))")
println("  sweep_count = $(cp["sweep_count"])")

cp_color = RGB{Float64}(cp["checkpoint_color"]...)
r = round(Int, red(cp_color) * 255)
g = round(Int, green(cp_color) * 255)
b = round(Int, blue(cp_color) * 255)
hex = "#" * uppercase(string(r, base=16, pad=2) * string(g, base=16, pad=2) * string(b, base=16, pad=2))
println("  Checkpoint color: \e[48;2;$(r);$(g);$(b)m  \e[0m $(hex)")

# ## Parallel Tempering Preview
#
# For more advanced simulations, GayMC supports parallel tempering:
# multiple replicas at different temperatures, each with its own color stream.

println("\n🌡️ Parallel Tempering")
temps = [1.0, 2.0, 4.0, 8.0]
replicas = gay_tempering(seed, temps)

for r in replicas
    for _ in 1:20
        gay_sweep!(r.ctx)
    end
    
    state = color_state(r.ctx)
    ri = round(Int, red(state) * 255)
    gi = round(Int, green(state) * 255)
    bi = round(Int, blue(state) * 255)
    println("  T=$(r.T), β=$(round(r.β, digits=3)): \e[48;2;$(ri);$(gi);$(bi)m    \e[0m")
end

# ## Conclusion
#
# Gay Metropolis brings color to Monte Carlo:
#
# - **Deterministic**: Same seed = same colors = same science
# - **Parallel-safe**: SPI ensures reproducibility across workers
# - **Visual**: See our simulation's fingerprint
# - **Checkpointable**: HDF5-ready with color metadata
#
# Every sweep has a color. Every measurement has a color.
# Every checkpoint has a color. The entire simulation is a rainbow of reproducible science.
#
# 🏳️‍🌈

println("\n🏳️‍🌈 Gay Metropolis Complete!")
println("  Total sweeps colored: $(ctx.sweep_count)")
println("  Seed: $(ctx.seed)")
