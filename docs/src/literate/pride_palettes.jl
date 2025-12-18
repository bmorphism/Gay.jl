# # Pride Flag Palettes ◈
#
# Gay.jl provides accurate color palettes for pride flags,
# available in any supported color space.

# ## Setup

using Gay
using Colors: RGB

# ## Classic Rainbow ◈
#
# Gilbert Baker's 1978 design — the original six-stripe flag.

println("=== Rainbow Flag ===")
show_colors(rainbow())
show_palette(rainbow())

# ## In Different Color Spaces

println("\n=== Rainbow in Different Color Spaces ===")

println("sRGB:")
show_colors(rainbow(SRGB()))

println("Display P3:")
show_colors(rainbow(DisplayP3()))

println("Rec.2020:")
show_colors(rainbow(Rec2020()))

# ## Transgender Flag ◇◈◇

println("\n=== Transgender Flag ===")
show_colors(transgender())
show_palette(transgender())

# ## Bisexual Flag

println("\n=== Bisexual Flag ===")
show_colors(bisexual())
show_palette(bisexual())

# ## Nonbinary Flag

println("\n=== Nonbinary Flag ===")
show_colors(nonbinary())
show_palette(nonbinary())

# ## Pansexual Flag

println("\n=== Pansexual Flag ===")
show_colors(pansexual())
show_palette(pansexual())

# ## Asexual Flag

println("\n=== Asexual Flag ===")
show_colors(asexual())
show_palette(asexual())

# ## Generic Access via `pride_flag`

println("\n=== Flags via pride_flag() ===")
println("Rainbow: ", length(pride_flag(:rainbow)), " colors")
println("Trans: ", length(pride_flag(:trans)), " colors")

# ## Using Pride Colors in Visualizations

println("\n=== Pride Colors for Data Viz ===")

categories = ["A", "B", "C", "D", "E", "F"]
colors = rainbow()

println("Categorical mapping:")
for (cat, color) in zip(categories, colors)
    println("  Category $cat -> ", color)
end

# ## Creating Custom Pride-Inspired Palettes

gay_seed!(42)

println("\n=== Custom Pride-Inspired Palette ===")
println("Rainbow base + random variations:")

base_rainbow = rainbow()
custom = RGB{Float64}[]
for c in base_rainbow
    nc = next_color()
    push!(custom, RGB(
        clamp(c.r * 0.8 + nc.r * 0.2, 0, 1),
        clamp(c.g * 0.8 + nc.g * 0.2, 0, 1),
        clamp(c.b * 0.8 + nc.b * 0.2, 0, 1)
    ))
end

show_palette(custom)

# The custom palette is reproducible!
gay_seed!(42)
custom2 = RGB{Float64}[]
for c in base_rainbow
    nc = next_color()
    push!(custom2, RGB(
        clamp(c.r * 0.8 + nc.r * 0.2, 0, 1),
        clamp(c.g * 0.8 + nc.g * 0.2, 0, 1),
        clamp(c.b * 0.8 + nc.b * 0.2, 0, 1)
    ))
end

@assert custom == custom2
println("◆ Custom pride palette is reproducible")

println("\n◆ Pride palettes example complete ◈")
