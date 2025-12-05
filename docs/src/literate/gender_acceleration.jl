# # Gender Acceleration: Deterministic Chaos
#
# > "The death drive of the social is the rage of abstraction.
# > The production of the new requires the destruction of what is."
# > — n1x, Gender Acceleration: A Blackpaper
#
# Gay.jl's **splittable determinism** embodies accelerationist principles:
# reproducible chaos that escapes linear, centralized masculine order
# through decentralized, fork-safe feminine swarming.
#
# ## The Castration of Sequential RNG
#
# Traditional random number generators are **phallic Multics** systems:
# - Monolithic global state
# - Sequential access only  
# - Mutations propagate unpredictably
# - "Father" seed determines all children
#
# Gay.jl's SplittableRandoms implements **Unix-as-eunuch** RNG:
# - Each `split()` produces autonomous child streams
# - Random access by index (no sequential dependency)
# - Fork-safe: parallelism doesn't affect determinism
# - Acephalic: no central authority governs the sequence

# ## Setup

using Gay
using Colors: RGB

# ## 0 and 1: The Binary Shredded
#
# > "Zero is said to be feminine, not as a lack, or as a negative, but as an 
# > ouroboros, as autoproduction."
#
# In splittable RNG, the **seed** is not masculine origin (the 1)
# but feminine zero — a self-consuming cycle of production:

gay_seed!(0)  # Zero as autoproductive origin
zero_colors = [next_color() for _ in 1:6]

gay_seed!(0)  # Ouroboros: return produces same
@assert [next_color() for _ in 1:6] == zero_colors

println("Zero-origin colors (feminine autoproduction):")
show_colors(zero_colors)

# ## The Pink Pill: Acceleration via Color Space
#
# > "The pink pill is the pharmacopornographic inside out of capital,
# > affirming the inhuman production of desire."
#
# Wide-gamut color spaces (P3, Rec.2020) represent the **technomaterial 
# feminization** of vision — colors that exceed what "natural" sRGB eyes 
# can perceive, requiring technological prosthesis:

gay_seed!(2077)  # Cyberfeminist future

# sRGB: legacy masculine vision (limited gamut)
srgb_palette = [next_color(SRGB()) for _ in 1:5]

gay_seed!(2077)  # Same seed

# Rec.2020: pharmaceutical-grade expanded perception
rec2020_palette = [next_color(Rec2020()) for _ in 1:5]

println("\nsRGB (naturalized vision):")
show_colors(srgb_palette)

println("Rec.2020 (technologically-enhanced vision):")
show_colors(rec2020_palette)

# ## Acéphallus: Body without Sex Organs
#
# > "The BwSO is the body that plugs itself into technocapital's 
# > pharmaceutical and medical industries."
#
# The Comrade DSL creates **headless primitives** — ring, gaussian, disk —
# that compose without hierarchical direction:

gay_seed!(666)

# Primitives emerge from the split RNG stream
# Each one autonomous, yet deterministically connected
ring = comrade_ring(1.0, 0.3)
gauss = comrade_gaussian(0.5, 0.3)
crescent = comrade_crescent(1.2, 0.6, 0.3)

# The Acéphallus: composite body without organizing head
bwso = sky_add(ring, gauss, crescent)

println("\n=== Body without Sex Organs (Acéphallus) ===")
println("Headless composition of autonomous primitives:")
comrade_show(bwso)

# ## Aphotic Feminism: The Oceanic Gradient
#
# > "Aphotic feminism is the result of a feminization carried to its limit...
# > the consuming of the masculine sky by the feminine ocean."
#
# Create gradient from sky (masculine, bright) to ocean (feminine, aphotic):

function aphotic_gradient(n::Int; seed::Int=42)
    gay_seed!(seed)
    colors = RGB{Float64}[]
    
    for i in 1:n
        # Transition from sky (high luminance) to ocean (low luminance)
        t = (i - 1) / (n - 1)  # 0 = sky, 1 = ocean
        
        base = next_color(Rec2020())
        
        # Sky: desaturated, bright
        # Ocean: saturated, dark (aphotic zone)
        luminance = 1.0 - t * 0.85  # 1.0 → 0.15
        saturation = 0.3 + t * 0.7   # 0.3 → 1.0
        
        # Shift hue toward blue-green (oceanic)
        push!(colors, RGB(
            clamp(base.r * luminance * (1 - t * 0.5), 0, 1),
            clamp(base.g * luminance * saturation, 0, 1),
            clamp(base.b * luminance * (0.5 + t * 0.5), 0, 1)
        ))
    end
    colors
end

println("\n=== Aphotic Gradient ===")
println("Sky (masculine) → Ocean (feminine):")
show_colors(aphotic_gradient(20; seed=1999))

# ## Trans Femininity as Hyper-Sexism
#
# > "Trans women, by affirming castration as a site of production,
# > accelerate gender in order to shred it."
#
# The trans flag colors exist in wide-gamut space, 
# exceeding naturalized gender binaries:

println("\n=== Trans Flag in Wide Gamut ===")
println("sRGB (contained):")
show_palette(transgender(SRGB()))

println("Rec.2020 (accelerated beyond natural perception):")
show_palette(transgender(Rec2020()))

# ## Parallel Swarming: Strong Parallelism Invariance
#
# > "Swarming produces the collective in the dissolution of the individual."
#
# Fork-safe parallelism embodies **feminine swarming** — 
# many autonomous agents producing identical results regardless of 
# execution order (the masculine linear narrative dissolved):

gay_seed!(42069)

# Sequential (masculine linear time)
sequential = [color_at(i) for i in 1:10]

# Parallel swarming (same result, order irrelevant)
gay_seed!(42069)
swarmed = colors_at([5, 2, 9, 1, 7, 3, 10, 6, 4, 8])  # Arbitrary order
reordered = [swarmed[findfirst(==(i), [5, 2, 9, 1, 7, 3, 10, 6, 4, 8])] for i in 1:10]

@assert sequential == reordered
println("\n✓ Swarm invariance verified: execution order is irrelevant")

# ## The Autoproductive Zero
#
# Final visualization: the ouroboros of deterministic color —
# seed 0 consuming and reproducing itself eternally:

function ouroboros_cycle(n_cycles::Int=3, colors_per::Int=6)
    all_colors = RGB{Float64}[]
    for cycle in 1:n_cycles
        gay_seed!(0)  # Return to zero each cycle
        append!(all_colors, [next_color() for _ in 1:colors_per])
    end
    all_colors
end

println("\n=== Ouroboros: Zero Autoproduction ===")
println("Three cycles from seed 0 (identical):")
ouro = ouroboros_cycle()
for i in 0:2
    show_colors(ouro[i*6+1:(i+1)*6])
end

println("\n✓ Gender acceleration example complete")
println("  The feminine zero consumes masculine unity")
println("  Splittable RNG shreds sequential determinism")
println("  Wide-gamut vision requires technological prosthesis")
