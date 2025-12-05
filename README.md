# Gay.jl 🏳️‍🌈

Wide-gamut color sampling with **splittable determinism** — reproducible colors via [SplittableRandoms.jl](https://github.com/Julia-Tempering/SplittableRandoms.jl), inspired by [Pigeons.jl](https://pigeons.run)'s Strong Parallelism Invariance (SPI) pattern.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/bmorphism/Gay.jl")
```

## Features

### 🎨 Wide-Gamut Color Spaces
- **sRGB** (standard)
- **Display P3** (Apple/DCI)
- **Rec.2020** (HDR/UHDTV)
- **Custom primaries**

### 🎲 Deterministic Random Colors
Same seed = same colors, always — regardless of execution order:

```julia
using Gay

gay_seed!(42)
c1 = next_color()        # First color
c2 = next_color()        # Second color

gay_seed!(42)            # Reset
c1 == next_color()       # true — deterministic!
```

### 🔢 Random Access by Index
Jump to any position in the color sequence without iteration:

```julia
color_at(1)              # First color
color_at(1000)           # 1000th color
colors_at([1, 10, 100])  # Batch access
palette_at(5, 6)         # 6-color palette at index 5
```

### 🏳️‍🌈 Pride Flag Palettes
```julia
rainbow()                # 6-color rainbow
transgender()            # Trans flag colors
bisexual()               # Bi flag colors
pride_flag(:progress)    # Progress Pride flag

# In any color space
rainbow(Rec2020())       # Wide-gamut rainbow
```

## Black Hole Imaging Demo

Inspired by [Comrade.jl](https://github.com/ptiede/Comrade.jl) (Event Horizon Telescope VLBI imaging):

```julia
include("examples/blackhole.jl")
println(render_blackhole(seed=1337, colorspace=Rec2020()))
eht_rings(seed=2017)
```

Generates deterministic false-color visualizations of black hole accretion disks with:
- Photon ring structure
- Relativistic Doppler boosting
- Temperature-dependent plasma colors

## How It Works

Gay.jl uses **splittable random streams** from [SplittableRandoms.jl](https://github.com/Julia-Tempering/SplittableRandoms.jl):

```julia
# Each color operation splits the RNG
gay_seed!(seed)
rng1 = gay_split()  # Independent stream 1
rng2 = gay_split()  # Independent stream 2
# Same seed → same splits → same colors
```

This is the same pattern used by:
- **Pigeons.jl** — reproducible parallel tempering MCMC
- **Comrade.jl** — black hole imaging with Bayesian inference

The **Strong Parallelism Invariance** property ensures identical results regardless of:
- Number of threads/processes
- Execution order
- Parallel vs sequential execution

## API Reference

### Color Generation
- `random_color(cs)` — random color (non-deterministic)
- `next_color(cs)` — deterministic next color
- `next_colors(n, cs)` — n deterministic colors
- `next_palette(n, cs)` — n visually distinct colors

### Random Access
- `color_at(index, cs)` — color at specific index
- `colors_at(indices, cs)` — colors at multiple indices
- `palette_at(index, n, cs)` — palette at index

### RNG Control
- `gay_seed!(seed)` — reset global RNG
- `gay_split()` — get independent RNG stream
- `GayRNG(seed)` — create new RNG instance

### Color Spaces
- `SRGB()` — standard RGB
- `DisplayP3()` — Apple Display P3
- `Rec2020()` — ITU-R BT.2020
- `CustomColorSpace(primaries, name)` — custom

### Pride Flags
- `rainbow()`, `transgender()`, `bisexual()`
- `nonbinary()`, `pansexual()`, `asexual()`
- `pride_flag(:lesbian)`, `pride_flag(:progress)`

### Display
- `show_colors(colors)` — ANSI terminal display
- `show_palette(colors)` — with hex codes

## Dependencies

- [Colors.jl](https://github.com/JuliaGraphics/Colors.jl)
- [SplittableRandoms.jl](https://github.com/Julia-Tempering/SplittableRandoms.jl)
- [Pigeons.jl](https://github.com/Julia-Tempering/Pigeons.jl)
- [LispSyntax.jl](https://github.com/swadey/LispSyntax.jl)

## License

MIT
