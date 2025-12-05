# Gay.jl 🏳️‍🌈

Wide-gamut color sampling with **splittable determinism** — reproducible colors via [SplittableRandoms.jl](https://github.com/Julia-Tempering/SplittableRandoms.jl), inspired by [Pigeons.jl](https://pigeons.run)'s Strong Parallelism Invariance (SPI) pattern.

[![CI](https://github.com/bmorphism/Gay.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/bmorphism/Gay.jl/actions/workflows/CI.yml)
[![Documentation](https://github.com/bmorphism/Gay.jl/actions/workflows/Documentation.yml/badge.svg)](https://bmorphism.github.io/Gay.jl/)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

```
       🌌 Nice Black Hole (gay_seed!(69)) 🌌

              ░░░░▒▒▒▓▓▓▓▓▓▒▒▒░░░░              
          ░░▒▒▓▓████████████████▓▓▒▒░░          
        ░▒▓█████████▓▓▒▒▒▒▓▓█████████▓▒░        
      ░▒████████▓▒░░        ░░▒▓████████▒░      
    ░▒███████▓░░    ░░▒▒▒▒░░    ░░▓███████▒░    
   ░▓██████▓░     ░▒▓██████▓▒░     ░▓██████▓░   
  ░▓█████▓░      ▒████████████▒      ░▓█████▓░  
  ▒█████▓      ░▓██████████████▓░      ▓█████▒  
 ░██████░      ▓████████████████▓      ░██████░ 
 ▒█████▓      ▓██████████████████▓      ▓█████▒ 
 ▓█████▒     ░████████████████████░     ▒█████▓ 
 ██████░     ▓████████████████████▓     ░██████ 
 ██████░     ████████████████████▓▓     ░██████ 
 ▓█████▒     ░███████████████████░      ▒█████▓ 
 ▒█████▓      ▓████████████████▓░      ▓█████▒  
 ░██████░      ▓██████████████▓       ░██████░  
  ▒█████▓       ░▓████████████░      ▓█████▒   
  ░▓█████▓░       ░▒▓██████▒░      ░▓█████▓░   
   ░▓██████▓░        ░░░░░░      ░▓██████▓░    
    ░▒███████▓░░              ░░▓███████▒░     
      ░▒████████▓▒░░      ░░▒▓████████▒░       
        ░▒▓█████████▓▓▓▓▓▓█████████▓▒░         
          ░░▒▒▓▓████████████████▓▒▒░░          
              ░░░░▒▒▒▓▓▓▓▓▓▒▒▒░░░░              

  (ring 0.69 0.169) + (gaussian 0.42) + (ring 1.069 0.269)
   ^^^^^^^           ^^^^^^^^^^^^^^    ^^^^^^^^^^^^^^^^
   golden            deep blue         silver
```

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

## Comrade.jl-Style Sky Models

Colored S-expressions for VLBI sky model composition, inspired by [Comrade.jl](https://github.com/ptiede/Comrade.jl):

```julia
using Gay

# Primitives get deterministic colors from SplittableRandoms
gay_seed!(2017)
ring = comrade_ring(1.0, 0.3)      # → (ring 1.0 0.3)  ← colored parens!
gauss = comrade_gaussian(0.5)      # → (gaussian 0.5 0.5)
model = sky_add(ring, gauss)       # → (ring) + (gaussian)

# Display as colored S-expression + ASCII render
comrade_show(model)
```

**Output:**
```
Colored S-Expression (parentheses colored by component):
(ring 1.0 0.3) + (gaussian 0.5 0.5)

Intensity Map:
        ████████████        
      ████████████████      
    ██████    ████    ██████    
   █████        ██        █████   
  ████          ██          ████  
  ████          ██          ████  
   █████        ██        █████   
    ██████    ████    ██████    
      ████████████████      
        ████████████        
```

### Model Types

| Style | S-Expression | Description |
|-------|--------------|-------------|
| M87*  | `(ring r w) + (gaussian σ)` | Ring + central gaussian |
| Sgr A* | `(crescent r_out r_in shift) + (disk r)` | Asymmetric crescent |
| Rings | `(ring) + (ring) + (ring) + (ring)` | Multi-ring structure |
| Custom | Mix of primitives | User-defined |

## Gallery: 1069 Models

Generated **1069 sky models in parallel** using SplittableRandoms fork-safe streams:

```bash
julia --threads=auto scripts/generate_gallery.jl
```

- **Master seed:** 42069 (fully reproducible)
- **Threads:** 16 parallel workers
- **Time:** 1.17 seconds
- **Each model:** Independent forked RNG stream

### Top 5 by Aesthetic Score

```
#1 [rings] seed=51749 (4 rings)
   (ring 0.63 0.23) + (ring 0.91 0.18) + (ring 1.22 0.11) + (ring 1.52 0.29)

#2 [rings] seed=73597 (4 rings)  
   (ring 0.73 0.23) + (ring 0.99 0.14) + (ring 1.25 0.23) + (ring 1.56 0.12)

#3 [rings] seed=57547 (4 rings)
   (ring 0.76 0.25) + (ring 1.08 0.18) + (ring 1.38 0.13) + (ring 1.61 0.21)
```

Full gallery: [`gallery/index.md`](gallery/index.md) | All models: [`gallery/catalog.jsonl`](gallery/catalog.jsonl)

## BBP π Digit Extraction

Random access to π digits → deterministic colors. The [Bailey-Borwein-Plouffe formula](https://en.wikipedia.org/wiki/Bailey%E2%80%93Borwein%E2%80%93Plouffe_formula) extracts the n-th hexadecimal digit of π WITHOUT computing digits 0..n-1 — the same random access property as `color_at(n)`:

```julia
include("examples/bbp_pi.jl")

# Extract hex digit at position 1000 (no iteration!)
pi_hex_digit(1000)  # → 0x6

# Color derived from π digit position
pi_color_at(1000; colorspace=Rec2020())

# Palette from consecutive positions (parallelizable)
pi_palette(0, 16)  # First 16 π-derived colors

# Visualization
render_pi_spiral(seed=314159, colorspace=Rec2020())
```

**Shared properties with Gay.jl:**
```
✓ Same seed always produces same colors
✓ Parallel execution is reproducible  
✓ Random access by index is efficient
```

## Black Hole Imaging Demo

Inspired by [Comrade.jl](https://github.com/ptiede/Comrade.jl) (Event Horizon Telescope VLBI imaging):

```julia
include("examples/blackhole.jl")
println(render_blackhole(seed=1337, colorspace=Rec2020()))
eht_rings(seed=2017)
```

Generates deterministic false-color visualizations of black hole accretion disks with:
- Photon ring structure (EHT n=1,2,3... orbits)
- Relativistic Doppler boosting (bright approaching side)
- Temperature-dependent plasma colors (hot inner → cool outer)

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

## GayInvaders: Terminal Game Demo

Full interactive Space Invaders with deterministic color palettes, inspired by [Lilith Hafner's JuliaCon talk](https://www.youtube.com/watch?v=PgqrHm-wL1w):

```julia
using Gay
include(joinpath(pkgdir(Gay), "examples", "spaceinvaders_colors.jl"))
GayInvaders.main(seed=42)  # Same seed = same colors!
```

**Features:**
- 🙯 Enemy rows colored by `color_at(row; seed=seed)`
- 🙭 Ship in trans pride light blue
- 🢙 Bullets in trans pride pink
- ✦ Rainbow explosion effects
- Parallel color generation via [OhMyThreads.jl](https://github.com/JuliaFolds2/OhMyThreads.jl)

**Controls:** Arrow keys/WASD to move, Space to fire, Q to quit.

## Parallel Color Determinism

Gay.jl provides **Strong Parallelism Invariance** — colors are identical whether generated sequentially or in parallel:

```julia
using Gay, OhMyThreads

seed = 42
sequential = [color_at(i; seed=seed) for i in 1:100]
parallel = tmap(i -> color_at(i; seed=seed), 1:100)

sequential == parallel  # true — always!
```

This is critical for:
- Reproducible game visuals across different hardware
- Parallel rendering without color drift
- Shareable "color seeds" between users

## Dependencies

- [Colors.jl](https://github.com/JuliaGraphics/Colors.jl)
- [SplittableRandoms.jl](https://github.com/Julia-Tempering/SplittableRandoms.jl)
- [OhMyThreads.jl](https://github.com/JuliaFolds2/OhMyThreads.jl)
- [LispSyntax.jl](https://github.com/swadey/LispSyntax.jl)

## Code Quality

Tested with [Aqua.jl](https://github.com/JuliaTesting/Aqua.jl) for:
- ✅ No method ambiguities
- ✅ No unbound type parameters
- ✅ No undefined exports
- ✅ No stale dependencies
- ✅ Proper compat bounds

## License

MIT
