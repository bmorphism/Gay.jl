# Color Spaces

Gay.jl supports multiple wide-gamut color spaces for accurate color representation across different display technologies.

## Types

```@docs
ColorSpace
SRGB
DisplayP3
Rec2020
CustomColorSpace
```

## Functions

```@docs
gamut_primaries
gamut_name
in_gamut
clamp_to_gamut
```

## Color Space Comparison

| Space | Coverage | Use Case |
|-------|----------|----------|
| sRGB | Standard | Web, legacy displays |
| Display P3 | ~25% larger | Apple devices, DCI cinema |
| Rec.2020 | ~75% visible | HDR, UHDTV, wide-gamut |

## CIE xy Chromaticity Coordinates

```
             0.9
              |
              | ● Rec.2020 Green (0.17, 0.797)
         0.7  |    ● P3 Green (0.265, 0.69)
              |      ● sRGB Green (0.30, 0.60)
         0.5  |
              |
         0.3  | ● Rec.2020 Red (0.708, 0.292)
              |  ● P3 Red (0.68, 0.32)
         0.1  |   ● sRGB Red (0.64, 0.33)
              |________________________
              0.1  0.3  0.5  0.7  0.9
```

## Example

```julia
using Gay

# Check if color is within sRGB gamut
c = RGB(1.2, 0.5, 0.3)  # Out of sRGB
@assert !in_gamut(c, SRGB())
@assert in_gamut(c, DisplayP3())

# Clamp to gamut
c_clamped = clamp_to_gamut(c, SRGB())
@assert c_clamped.r == 1.0
```
