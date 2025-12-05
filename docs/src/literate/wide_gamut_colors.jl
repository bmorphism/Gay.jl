# # Wide-Gamut Color Spaces
#
# Gay.jl supports **wide-gamut color spaces** beyond standard sRGB,
# enabling richer, more vibrant colors on modern displays.
#
# ## Color Space Comparison
#
# | Color Space | Coverage | Use Case |
# |-------------|----------|----------|
# | **sRGB**    | ~35% of visible | Web, legacy displays |
# | **Display P3** | ~45% of visible | Apple devices, DCI cinema |
# | **Rec.2020** | ~76% of visible | HDR, UHDTV, future-proof |
#
# The wider the gamut, the more saturated colors can be represented.

# ## Setup

using Gay
using Colors

# ## Available Color Spaces

srgb = SRGB()           # Standard RGB (default)
p3 = DisplayP3()        # Apple Display P3
rec2020 = Rec2020()     # ITU-R BT.2020 (HDR)

println("Available color spaces:")
println("  • SRGB()      - $(srgb.name)")
println("  • DisplayP3() - $(p3.name)")
println("  • Rec2020()   - $(rec2020.name)")

# ## Generating Colors in Different Spaces
#
# Same seed produces analogous colors across spaces,
# but with different saturation characteristics:

gay_seed!(42)

println("\nSame seed (42) in different color spaces:")

gay_seed!(42)
c_srgb = next_color(SRGB())
println("sRGB:       ", c_srgb)

gay_seed!(42)
c_p3 = next_color(DisplayP3())
println("Display P3: ", c_p3)

gay_seed!(42)
c_rec2020 = next_color(Rec2020())
println("Rec.2020:   ", c_rec2020)

# ## Wide-Gamut Palettes
#
# Generate palettes optimized for each color space:

gay_seed!(1337)

println("\n6-color palettes per color space:")

println("\nsRGB palette:")
show_palette(next_palette(6, SRGB()))

gay_seed!(1337)
println("Display P3 palette:")
show_palette(next_palette(6, DisplayP3()))

gay_seed!(1337)
println("Rec.2020 palette:")
show_palette(next_palette(6, Rec2020()))

# ## Setting Global Color Space
#
# Change the default color space for all operations:

gay_space(:srgb)     # Set to sRGB
println("\nGlobal space set to: ", current_colorspace().name)

gay_space(:p3)       # Set to Display P3
println("Global space set to: ", current_colorspace().name)

gay_space(:rec2020)  # Set to Rec.2020
println("Global space set to: ", current_colorspace().name)

# Reset to sRGB for remaining examples
gay_space(:srgb)

# ## Perceptual Uniformity in LCH
#
# Gay.jl samples colors in **LCH (Lightness-Chroma-Hue)** space
# to ensure perceptual uniformity — colors appear equally spaced
# to human vision.
#
# ```
# LCH Sampling:
#   L (Lightness): 30-80  — avoids too dark/bright
#   C (Chroma):    40-80  — moderate saturation
#   H (Hue):       0-360  — full hue circle
# ```
#
# For wide-gamut spaces, chroma is reduced at gamut boundaries
# to stay within displayable range.

# ## Gamut Mapping
#
# When a color falls outside the target gamut, Gay.jl uses
# **chroma reduction** — preserving hue and lightness while
# reducing saturation until the color is displayable.
#
# This is perceptually superior to simple clipping.

# Example: A highly saturated color
using Colors: LCHab

lch = LCHab(50, 100, 120)  # Very saturated green
rgb = convert(RGB, lch)

println("\nGamut mapping example:")
println("  LCH: L=$(lch.l), C=$(lch.c), H=$(lch.h)")
println("  RGB: R=$(round(rgb.r, digits=3)), G=$(round(rgb.g, digits=3)), B=$(round(rgb.b, digits=3))")

if rgb.r < 0 || rgb.g < 0 || rgb.b < 0 || rgb.r > 1 || rgb.g > 1 || rgb.b > 1
    println("  ⚠ Out of sRGB gamut — would be mapped")
else
    println("  ✓ Within sRGB gamut")
end

# ## Custom Color Space
#
# Define your own color space with custom primaries:

custom = CustomColorSpace(
    [0.680, 0.320],  # Red primary (x, y)
    [0.265, 0.690],  # Green primary
    [0.150, 0.060],  # Blue primary
    "MyGamut"
)

println("\nCustom color space: ", custom.name)
println("  Red primary:   ", custom.primaries[1])
println("  Green primary: ", custom.primaries[2])
println("  Blue primary:  ", custom.primaries[3])

# ## Practical Recommendations
#
# | Scenario | Recommended Space |
# |----------|-------------------|
# | Web graphics | `SRGB()` |
# | macOS/iOS apps | `DisplayP3()` |
# | HDR video | `Rec2020()` |
# | Print (CMYK target) | `SRGB()` then convert |
# | Maximum vibrancy | `Rec2020()` |
# | Accessibility | `SRGB()` (widest support) |

# ## Color Space Detection
#
# Check if your terminal supports wide gamut:

println("\nTerminal color support check:")
println("  True color (24-bit): ", get(ENV, "COLORTERM", "unknown"))

# Most modern terminals support true color but render in sRGB.
# For actual wide-gamut display, you need:
# - Wide-gamut monitor
# - Color-managed application
# - Proper ICC profile

println("\n✓ Wide-gamut color spaces example complete")
