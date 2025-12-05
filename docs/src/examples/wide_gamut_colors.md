```@meta
EditURL = "../literate/wide_gamut_colors.jl"
```

# Wide-Gamut Color Spaces

Gay.jl supports **wide-gamut color spaces** beyond standard sRGB,
enabling richer, more vibrant colors on modern displays.

## Color Space Comparison

| Color Space | Coverage | Use Case |
|-------------|----------|----------|
| **sRGB**    | ~35% of visible | Web, legacy displays |
| **Display P3** | ~45% of visible | Apple devices, DCI cinema |
| **Rec.2020** | ~76% of visible | HDR, UHDTV, future-proof |

## Setup

````@example wide_gamut_colors
using Gay
using Colors: RGB, LCHab
````

## Available Color Spaces

````@example wide_gamut_colors
srgb = SRGB()
p3 = DisplayP3()
rec2020 = Rec2020()

println("Available color spaces:")
println("  • SRGB()")
println("  • DisplayP3()")
println("  • Rec2020()")
````

## Generating Colors in Different Spaces

````@example wide_gamut_colors
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
````

## Wide-Gamut Palettes

````@example wide_gamut_colors
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
````

## Perceptual Uniformity in LCH

Gay.jl samples colors in **LCH (Lightness-Chroma-Hue)** space
to ensure perceptual uniformity.

## Gamut Mapping Example

````@example wide_gamut_colors
lch = LCHab(50, 100, 120)
rgb = convert(RGB, lch)

println("\nGamut mapping example:")
println("  LCH: L=$(lch.l), C=$(lch.c), H=$(lch.h)")
println("  RGB: R=$(round(rgb.r, digits=3)), G=$(round(rgb.g, digits=3)), B=$(round(rgb.b, digits=3))")
````

## Practical Recommendations

| Scenario | Recommended Space |
|----------|-------------------|
| Web graphics | `SRGB()` |
| macOS/iOS apps | `DisplayP3()` |
| HDR video | `Rec2020()` |
| Maximum vibrancy | `Rec2020()` |
| Accessibility | `SRGB()` (widest support) |

## Color Space Detection

````@example wide_gamut_colors
println("\nTerminal color support check:")
println("  COLORTERM: ", get(ENV, "COLORTERM", "not set"))

println("\n✓ Wide-gamut color spaces example complete")
````

