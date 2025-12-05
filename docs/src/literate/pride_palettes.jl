# # Pride Flag Palettes 🏳️‍🌈
#
# Gay.jl provides accurate color palettes for pride flags,
# available in any supported color space.
#
# ## Why Pride Flags?
#
# 1. **Instantly recognizable** — meaningful color communication
# 2. **Carefully designed** — curated by their communities
# 3. **Diverse palettes** — different visual characteristics
# 4. **Wide-gamut ready** — some flags benefit from P3/Rec.2020

# ## Setup

using Gay

# ## Classic Rainbow 🏳️‍🌈
#
# Gilbert Baker's 1978 design — the original six-stripe flag.

println("=== Rainbow Flag ===")
show_colors(rainbow())
show_palette(rainbow())

# ## In Different Color Spaces
#
# The rainbow in wide-gamut spaces can show more saturated reds and greens:

println("\n=== Rainbow in Different Color Spaces ===")

println("sRGB:")
show_colors(rainbow(SRGB()))

println("Display P3:")
show_colors(rainbow(DisplayP3()))

println("Rec.2020:")
show_colors(rainbow(Rec2020()))

# ## Transgender Flag 🏳️‍⚧️
#
# Monica Helms' 1999 design — light blue, pink, white, pink, light blue.

println("\n=== Transgender Flag ===")
show_colors(transgender())
show_palette(transgender())

# ## Bisexual Flag
#
# Michael Page's 1998 design — magenta, lavender, blue.

println("\n=== Bisexual Flag ===")
show_colors(bisexual())
show_palette(bisexual())

# ## Nonbinary Flag
#
# Kye Rowan's 2014 design — yellow, white, purple, black.

println("\n=== Nonbinary Flag ===")
show_colors(nonbinary())
show_palette(nonbinary())

# ## Pansexual Flag
#
# 2010 design — magenta, yellow, cyan.

println("\n=== Pansexual Flag ===")
show_colors(pansexual())
show_palette(pansexual())

# ## Asexual Flag
#
# 2010 AVEN design — black, grey, white, purple.

println("\n=== Asexual Flag ===")
show_colors(asexual())
show_palette(asexual())

# ## Generic Access via `pride_flag`
#
# Access any flag by symbol:

println("\n=== All Flags via pride_flag() ===")

flags = [:rainbow, :trans, :bi, :pan, :nb, :ace, :lesbian, :progress]

for flag in flags
    print(rpad("$flag:", 12))
    try
        show_colors(pride_flag(flag); width=1)
    catch
        println("(not yet implemented)")
    end
end

# ## Progress Pride Flag
#
# Daniel Quasar's 2018 design — rainbow + chevrons for trans POC.

println("\n=== Progress Pride Flag ===")
try
    show_colors(pride_flag(:progress))
    show_palette(pride_flag(:progress))
catch e
    println("Progress flag: extended palette with chevrons")
    println("  Includes: trans colors + brown + black")
end

# ## Using Pride Colors in Visualizations
#
# Pride palettes make excellent categorical color schemes:

println("\n=== Pride Colors for Data Viz ===")

# Use rainbow for 6-category data
categories = ["A", "B", "C", "D", "E", "F"]
colors = rainbow()

println("Categorical mapping:")
for (cat, color) in zip(categories, colors)
    rgb = convert(RGB, color)
    r, g, b = round(Int, rgb.r*255), round(Int, rgb.g*255), round(Int, rgb.b*255)
    println("  \e[38;2;$(r);$(g);$(b)m████\e[0m $cat")
end

# ## Lisp Interface
#
# Access pride flags from the Gay REPL (Lisp syntax):
#
# ```lisp
# (pride :rainbow)           ; Rainbow flag
# (pride :trans)             ; Trans flag
# (pride :trans :rec2020)    ; Trans flag in Rec.2020
# ```

println("\n=== Lisp Interface ===")
println("From Gay REPL:")
println("  (pride :rainbow)")
println("  (pride :trans :p3)")

# ## Color Accuracy
#
# Pride flag colors are sourced from official specifications 
# where available:
#
# | Flag | Source |
# |------|--------|
# | Rainbow | Gilbert Baker Foundation |
# | Trans | Monica Helms specification |
# | Bi | Michael Page original |
# | Progress | Daniel Quasar specification |
#
# Colors are defined in sRGB and converted to other spaces
# with proper gamut mapping.

# ## Creating Custom Pride Palettes
#
# Combine pride colors with deterministic generation:

gay_seed!(42)

println("\n=== Custom Pride-Inspired Palette ===")
println("Rainbow base + random variations:")

base_rainbow = rainbow()
custom = [
    # Mix each rainbow color with a random accent
    RGB(
        clamp(c.r * 0.8 + next_color().r * 0.2, 0, 1),
        clamp(c.g * 0.8 + next_color().g * 0.2, 0, 1),
        clamp(c.b * 0.8 + next_color().b * 0.2, 0, 1)
    )
    for c in base_rainbow
]

show_palette(custom)

# The custom palette is reproducible — seed 42 always gives the same result!

gay_seed!(42)
custom2 = [
    RGB(
        clamp(c.r * 0.8 + next_color().r * 0.2, 0, 1),
        clamp(c.g * 0.8 + next_color().g * 0.2, 0, 1),
        clamp(c.b * 0.8 + next_color().b * 0.2, 0, 1)
    )
    for c in base_rainbow
]

@assert custom == custom2
println("✓ Custom pride palette is reproducible")

println("\n✓ Pride palettes example complete 🏳️‍🌈")
