# Color Generation

## Sequential Access

```@docs
random_color
next_color
next_colors
next_palette
```

## Random Access

```@docs
color_at
colors_at
palette_at
```

## Display

```@docs
show_colors
show_palette
ansi_color
```

## Examples

### Basic Color Generation

```julia
using Gay

gay_seed!(1069)

# Single color
c = next_color()

# Multiple colors
colors = next_colors(5)

# Visually distinct palette (golden angle spacing)
palette = next_palette(6)
show_palette(palette)
```

### Random Access

```julia
# Jump to any position without iterating
c42 = color_at(42)
c1000 = color_at(1000)

# Batch access
colors = colors_at([1, 10, 100, 1000])

# Palette at specific index
palette = palette_at(42, 6)  # 6 colors starting at index 42
```

### Wide-Gamut Colors

```julia
# sRGB (default)
c_srgb = next_color()

# Display P3 (Apple devices)
c_p3 = next_color(DisplayP3())

# Rec.2020 (HDR)
c_2020 = next_color(Rec2020())
```

### Golden Angle Palette

`next_palette` uses golden angle spacing (137.5077...) in HSV space to generate visually distinct colors:

```
     Hue wheel
        0
       /|\
      / | \
   330  |  30
     \  |  /
      \ | /
       \|/
  300 --+-- 60
       /|\
      / | \
   270  |  90
     \  |  /
      \ | /
       \|/
       180
```

Each color is offset by the golden angle, maximizing visual separation.
