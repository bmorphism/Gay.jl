# API Reference

## Color Generation

### Deterministic (Splittable RNG)

```@docs
next_color
next_colors
next_palette
color_at
colors_at
palette_at
```

### RNG Control

```@docs
gay_seed!
gay_rng
gay_split
GayRNG
```

### Non-Deterministic (Legacy)

```@docs
random_color
random_colors
random_palette
```

## Color Spaces

### Built-in Spaces

```@docs
SRGB
DisplayP3
Rec2020
CustomColorSpace
```

### Space Management

```@docs
current_colorspace
gay_space
```

## Pride Flags

```@docs
rainbow
transgender
bisexual
nonbinary
pansexual
asexual
pride_flag
```

## Comrade Sky Models

### Primitives

```@docs
comrade_ring
comrade_mring
comrade_gaussian
comrade_disk
comrade_crescent
```

### Composition

```@docs
sky_add
sky_stretch
sky_rotate
sky_shift
```

### Display

```@docs
sky_show
sky_render
comrade_show
comrade_model
```

## Display Utilities

```@docs
show_colors
show_palette
```

## Lisp Interface

These functions are available in the Gay REPL (Lisp syntax):

| Lisp Form | Julia Function | Description |
|-----------|----------------|-------------|
| `(gay-next)` | `gay_next()` | Next deterministic color |
| `(gay-next n)` | `gay_next(n)` | Next n colors |
| `(gay-at i)` | `gay_at(i)` | Color at index i |
| `(gay-palette n)` | `gay_palette(n)` | n-color palette |
| `(gay-seed n)` | `gay_seed(n)` | Set RNG seed |
| `(gay-space :p3)` | `gay_space(:p3)` | Set color space |
| `(pride :rainbow)` | `gay_pride(:rainbow)` | Pride flag colors |

## Types

### Color Spaces

```julia
abstract type ColorSpace end

struct SRGB <: ColorSpace end
struct DisplayP3 <: ColorSpace end
struct Rec2020 <: ColorSpace end
struct CustomColorSpace <: ColorSpace
    primaries::Vector{Tuple{Float64, Float64}}
    name::String
end
```

### RNG

```julia
mutable struct GayRNG
    root::SplittableRandom
    current::SplittableRandom
    invocation::UInt64
    seed::UInt64
end
```

### Sky Models

```julia
abstract type SkyPrimitive end

struct Ring <: SkyPrimitive
    radius::Float64
    width::Float64
    color::RGB
end

struct Gaussian <: SkyPrimitive
    σx::Float64
    σy::Float64
    color::RGB
end

struct SkyModel
    components::Vector{Tuple{SkyPrimitive, NamedTuple}}
    total_flux::Float64
end
```

## Constants

```julia
GAY_SEED::UInt64  # Default seed (0x6761795f636f6c6f)
```
