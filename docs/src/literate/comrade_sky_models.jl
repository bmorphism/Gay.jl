# # Comrade.jl-Style Sky Models
#
# Gay.jl includes a **colored S-expression DSL** inspired by 
# [Comrade.jl](https://github.com/ptiede/Comrade.jl), the Event Horizon 
# Telescope's VLBI imaging package.
#
# Each sky model primitive gets a **deterministic color** from the 
# splittable RNG, enabling reproducible visualizations of black hole
# and radio astronomy models.
#
# ## The Connection
#
# Comrade.jl uses [Pigeons.jl](https://pigeons.run) for Bayesian inference.
# Pigeons uses SplittableRandoms.jl for reproducible MCMC chains.
# Gay.jl uses the same SplittableRandoms pattern for reproducible colors!
#
# ```
# Comrade.jl (EHT imaging)
#     └── Pigeons.jl (parallel tempering)
#           └── SplittableRandoms.jl
#                 └── Gay.jl (deterministic colors)
# ```

# ## Setup

using Gay

# ## Sky Model Primitives
#
# Gay.jl provides primitives matching VLBISkyModels.jl:

gay_seed!(2017)  # EHT M87* observation year

# ### Ring
# A circular ring — the photon ring around a black hole

ring = comrade_ring(1.0, 0.3)  # radius=1.0, width=0.3
println("Ring: ", sky_show(ring))

# ### Gaussian
# An elliptical Gaussian — central emission or jet base

gauss = comrade_gaussian(0.5, 0.3)  # σx=0.5, σy=0.3
println("Gaussian: ", sky_show(gauss))

# ### Disk
# A uniform disk — filled circular region

disk = comrade_disk(0.4)  # radius=0.4
println("Disk: ", sky_show(disk))

# ### Crescent
# An asymmetric ring — Doppler-boosted emission

crescent = comrade_crescent(1.2, 0.6, 0.3)  # r_out, r_in, shift
println("Crescent: ", sky_show(crescent))

# ## Composing Models
#
# Combine primitives with `sky_add` — like Comrade's `+` operator:

gay_seed!(2017)

m87_model = sky_add(
    comrade_ring(1.0, 0.3),
    comrade_gaussian(0.5, 0.3)
)

println("\n=== M87* Style Model ===")
comrade_show(m87_model)

# ## Different Black Hole Styles

# ### Sgr A* Style (Galactic Center)
gay_seed!(2022)  # Sgr A* observation year

sgra_model = sky_add(
    comrade_crescent(1.2, 0.6, 0.3),
    comrade_disk(0.4)
)

println("\n=== Sgr A* Style Model ===")
comrade_show(sgra_model)

# ### Multi-Ring Structure
gay_seed!(42069)

rings_model = sky_add(
    comrade_ring(0.6, 0.2),
    comrade_ring(0.9, 0.15),
    comrade_ring(1.2, 0.1),
    comrade_ring(1.5, 0.25)
)

println("\n=== Multi-Ring Model ===")
comrade_show(rings_model)

# ## Colored S-Expressions
#
# Each primitive's parentheses are colored with its deterministic color.
# This makes complex model compositions visually parseable:
#
# ```
# (ring 1.0 0.3) + (gaussian 0.5 0.3)
#  ^^^^^          ^^^^^^^^^
#  blue           green
# ```
#
# The colors are determined by:
# 1. The seed (`gay_seed!`)
# 2. The order of primitive creation
# 3. The splittable RNG state

# ## Using the Model Builder

gay_seed!(1337)
model_m87 = comrade_model(seed=1337, style=:m87)
println("\n=== Built-in M87 Model ===")
comrade_show(model_m87)

gay_seed!(1337)
model_sgra = comrade_model(seed=1337, style=:sgra)
println("\n=== Built-in Sgr A* Model ===")
comrade_show(model_sgra)

# ## Transformations
#
# Apply Comrade-style modifiers:

gay_seed!(42)
base = sky_add(
    comrade_ring(1.0, 0.25),
    comrade_gaussian(0.3)
)

# Stretch asymmetrically
stretched = sky_stretch(base, 1.5, 0.8)

# Rotate
rotated = sky_rotate(stretched, 0.3)  # radians

# Shift
shifted = sky_shift(rotated, 0.2, -0.1)

println("\n=== Transformed Model ===")
println("Original:    ", sky_show(base))
println("After transforms applied")

# ## ASCII Intensity Maps
#
# The `comrade_show` function renders colored ASCII intensity maps,
# showing the spatial structure of each model:
#
# - Ring → annular structure
# - Gaussian → central concentration  
# - Crescent → asymmetric brightness
# - Disk → filled circle
#
# Colors from each primitive blend additively.

# ## Gallery Generation
#
# Generate many models in parallel using SPI:
#
# ```julia
# using Base.Threads
# 
# models = Vector{SkyModel}(undef, 1000)
# @threads for i in 1:1000
#     # Each thread gets independent RNG stream
#     models[i] = comrade_model(seed=42069+i, style=rand([:m87, :sgra, :custom]))
# end
# ```
#
# All 1000 models are reproducible — same seeds give same models!

# ## Connection to Real EHT Imaging
#
# The primitives map to actual VLBISkyModels.jl types:
#
# | Gay.jl | VLBISkyModels.jl | Physical Meaning |
# |--------|------------------|------------------|
# | `Ring` | `MRing` convolved | Photon ring |
# | `Gaussian` | `Gaussian` | Jet base, central emission |
# | `Disk` | `Disk` | Filled emission region |
# | `Crescent` | `Crescent` | Doppler-boosted asymmetry |
#
# The colors help distinguish components in complex fits.

println("\n◆ Comrade sky models example complete")
