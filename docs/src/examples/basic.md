# Basic Usage

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/bmorphism/Gay.jl")
```

## Quick Start

```julia
using Gay

# Set seed for reproducibility (default: 1069)
gay_seed!(1069)

# Generate a single color
color = next_color()
println("Generated: ", color)

# Generate multiple colors
colors = next_colors(5)
show_colors(colors)

# Generate a visually distinct palette
palette = next_palette(6)
show_palette(palette)
```

## Reproducibility

Gay.jl guarantees **deterministic** color generation:

```julia
# Run 1
gay_seed!(42)
run1 = next_colors(10)

# Run 2 (same seed)
gay_seed!(42)
run2 = next_colors(10)

# Always identical
@assert run1 == run2
```

## Random Access

Jump to any position in the color sequence without iteration:

```julia
# Get the 1000th color directly
c1000 = color_at(1000)

# Batch access
colors = colors_at([1, 10, 100, 1000, 10000])

# Palette starting at index 42
palette = palette_at(42, 6)
```

## Wide-Gamut Colors

```julia
# sRGB (default, web-safe)
c_srgb = next_color(SRGB())

# Display P3 (Apple devices, DCI cinema)
c_p3 = next_color(DisplayP3())

# Rec.2020 (HDR, UHDTV)
c_2020 = next_color(Rec2020())
```

## Pride Flags

```julia
# Built-in pride flag palettes
show_colors(rainbow())
show_colors(transgender())
show_colors(bisexual())
show_colors(progress())

# Access any flag by symbol
colors = pride_flag(:nonbinary)
```

## Terminal Display

```julia
# Color blocks
show_colors(next_colors(10))

# With hex codes
show_palette(next_colors(10))
```

## Parallel Safety

Gay.jl uses **splittable random streams** (SPI pattern from Pigeons.jl):

```julia
using Base.Threads

gay_seed!(1069)

# Parallel color generation - results are deterministic!
colors = Vector{RGB}(undef, 100)
@threads for i in 1:100
    colors[i] = color_at(i; seed=1069)
end

# Same result every time
```
