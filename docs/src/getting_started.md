# Getting Started

## Installation

Gay.jl is not yet in the Julia General registry. Install directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/bmorphism/Gay.jl")
```

## Basic Usage

### Import the Package

```julia
using Gay
```

### Generate Your First Colors

```julia
# Random color (non-deterministic)
c = random_color()

# Deterministic color (reproducible)
gay_seed!(42)
c = next_color()
```

### Create Palettes

```julia
gay_seed!(1337)

# 6 visually distinct colors
palette = next_palette(6)
show_palette(palette)
```

### Use Pride Flags

```julia
show_colors(rainbow())
show_colors(transgender())
show_colors(bisexual())
```

## The Gay REPL

Gay.jl includes a custom REPL mode with Lisp syntax:

```julia
using Gay
init_gay_repl()
```

Press `` ` `` (backtick) to enter Gay mode:

```lisp
gay> (gay-next)           ; Next deterministic color
gay> (gay-palette 6)      ; 6-color palette
gay> (pride :rainbow)     ; Rainbow flag
gay> (gay-seed 42)        ; Set seed
```

Press backspace on empty line to return to Julia mode.

## Understanding Seeds

The seed determines the entire color sequence:

```julia
gay_seed!(42)
c1 = next_color()  # Color #1 for seed 42
c2 = next_color()  # Color #2 for seed 42

gay_seed!(42)      # Reset to same seed
@assert next_color() == c1  # Same sequence!
@assert next_color() == c2
```

Different seeds produce different sequences:

```julia
gay_seed!(42)
a = next_color()

gay_seed!(1337)
b = next_color()

@assert a != b  # Different seeds → different colors
```

## Color Spaces

Gay.jl supports multiple color spaces:

```julia
# Set global color space
gay_space(:srgb)     # Standard (default)
gay_space(:p3)       # Display P3
gay_space(:rec2020)  # Rec.2020

# Or specify per-call
next_color(SRGB())
next_color(DisplayP3())
next_color(Rec2020())
```

## Random Access

Jump to any position in the sequence without iterating:

```julia
# These give the same color regardless of prior calls
color_at(100)
color_at(100)  # Same!

# Batch access
colors_at([1, 10, 100, 1000])
```

## Next Steps

- [Splittable Determinism](examples/splittable_determinism.md) — Deep dive into reproducibility
- [Wide-Gamut Colors](examples/wide_gamut_colors.md) — Beyond sRGB
- [Comrade Sky Models](examples/comrade_sky_models.md) — Scientific visualization
- [Pride Palettes](examples/pride_palettes.md) — Flag color schemes
- [Parallel SPI](examples/parallel_spi.md) — Fork-safe parallelism
