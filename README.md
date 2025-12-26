# Gay.jl ◈

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

### ◈ Pride Flag Palettes
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
◆ Same seed always produces same colors
◆ Parallel execution is reproducible  
◆ Random access by index is efficient
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

### GPU / KernelAbstractions
- `ka_colors(n, seed)` — generate n colors via SPMD kernel
- `ka_colors!(matrix, seed)` — fill pre-allocated n×3 Float32 matrix
- `set_backend!(backend)` — switch to Metal/CUDA/AMD GPU
- `get_backend()` — current backend (default: CPU)

### SPI Verification
- `xor_fingerprint(colors)` — XOR-reduce colors to 32-bit hash
- `verify_spi(n, seed; gpu_backend)` — full verification suite
- `gpu_fingerprint(n, seed)` — generate + fingerprint on GPU

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

## Billion-Scale Color Generation

Using [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) SPMD kernels:

```julia
using Gay

# Generate 1 billion colors in 0.13 seconds
ka_benchmark()
# ═══════════════════════════════════════════════════════════════════════
#   KernelAbstractions Color Generation Benchmark
#   Backend: CPU, Threads: 8
# ═══════════════════════════════════════════════════════════════════════
#   1,000,000,000 colors in 0.13 seconds
#   Rate: 7,410 million colors/second
#   RGB sums: (5.0e8, 5.0e8, 5.0e8)
```

### Performance (8 threads, Apple M3)

| Function | n | Time | Rate |
|----------|---|------|------|
| `ka_colors(n, seed)` | 1M | 1.0 ms | 1,000 M/s |
| `ka_colors(n, seed)` | 10M | 25 ms | 400 M/s |
| `ka_color_sums(n, seed)` | 100M | 0.02s | 4,452 M/s |
| `ka_color_sums(n, seed)` | **1B** | **0.13s** | **7,097 M/s** |

### API

```julia
# Generate colors as n×3 Float32 matrix
colors = ka_colors(1_000_000, 42)

# Fill pre-allocated matrix
ka_colors!(my_matrix, 42)

# Streaming reduction for billion-scale (O(1) memory)
sums = ka_color_sums(1_000_000_000, 42)

# Built-in benchmark
ka_benchmark(n=1_000_000_000)
```

The same `@kernel` code runs on **CPU**, **Metal.jl**, **CUDA.jl**, or **AMDGPU.jl**.

This is critical for:
- Reproducible game visuals across different hardware
- Parallel rendering without color drift
- Shareable "color seeds" between users

## GPU-Accelerated SPI Verification

How do you *prove* that 100 million colors are identical across CPU and GPU? **XOR fingerprinting** — reduce all color bits to a single 32-bit hash:

```julia
using Gay, Metal

# Generate 100M colors on Metal GPU
colors = ka_colors(100_000_000, 42)
fp = xor_fingerprint(colors)  # → 0x38b8b8ad

# Same fingerprint = bitwise identical colors
@assert xor_fingerprint(ka_colors(100_000_000, 42)) == fp
```

### Verification at the Speed of Metal

The `gpu_fingerprint` function generates and fingerprints colors entirely on GPU:

```
┌─────────────────────────────────────────────────────────────┐
│  GPU Fingerprint Benchmark (Apple M5 Metal)                 │
├─────────────────┬────────────────┬─────────────────────────┤
│  Colors         │  Time          │  Fingerprint            │
├─────────────────┼────────────────┼─────────────────────────┤
│  1,000,000      │  3.2 ms        │  0x3addddae             │
│  10,000,000     │  37.8 ms       │  0x043aba9b             │
│  100,000,000    │  264.6 ms      │  0x38b8b8ad             │
└─────────────────┴────────────────┴─────────────────────────┘
```

**378 million colors/second** — verification at the speed of generation.

### Full SPI Verification Suite

```julia
using Gay, Metal

# Verify CPU sequential == CPU parallel == Metal GPU
verify_spi(10_000_000, 42; gpu_backend=MetalBackend())
```

```
════════════════════════════════════════════════════════════
SPI VERIFICATION: Strong Parallelism Invariance
════════════════════════════════════════════════════════════
  n = 10000000, seed = 42

1. CPU Sequential Reference
   XOR Fingerprint: 0x043aba9b
   ◆ Generated

2. CPU Parallel (KernelAbstractions)
   XOR Fingerprint: 0x043aba9b
   Colors match: ◆ PASS
   Fingerprint match: ◆ PASS

3. Workgroup Size Independence
   workgroup=32: ◆ PASS
   workgroup=64: ◆ PASS
   workgroup=128: ◆ PASS
   workgroup=256: ◆ PASS
   workgroup=512: ◆ PASS

4. GPU Backend: MetalBackend
   XOR Fingerprint: 0x043aba9b
   Colors match CPU: ◆ PASS
   Fingerprint match CPU: ◆ PASS

════════════════════════════════════════════════════════════
ALL SPI INVARIANTS VERIFIED ◆
════════════════════════════════════════════════════════════
```

### Why This Matters

The promise of splittable determinism is that **same seed → same colors, always**. But how do you verify this at scale?

| Approach | 100M Colors | Problem |
|----------|-------------|---------|
| Compare element-wise | Minutes | Memory-bound, slow |
| Sample randomly | Fast | Misses subtle bugs |
| **XOR fingerprint** | **265ms** | **Bitwise correctness proof** |

A single bit flip in any of the 300 million floats (100M × RGB) changes the fingerprint. If `0x38b8b8ad` matches across CPU and GPU, **every color is identical**.

This is how Gay.jl guarantees that the 1069 parallel-generated sky models in the gallery are reproducible — verified at GPU speed.

## Dependencies

- [Colors.jl](https://github.com/JuliaGraphics/Colors.jl)
- [SplittableRandoms.jl](https://github.com/Julia-Tempering/SplittableRandoms.jl)
- [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) — portable GPU kernels
- [OhMyThreads.jl](https://github.com/JuliaFolds2/OhMyThreads.jl)
- [LispSyntax.jl](https://github.com/swadey/LispSyntax.jl)

**Optional GPU backends:**
- [Metal.jl](https://github.com/JuliaGPU/Metal.jl) — Apple Silicon
- [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) — Nvidia
- [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl) — AMD

## Code Quality

Tested with [Aqua.jl](https://github.com/JuliaTesting/Aqua.jl) for:
- ▣ No method ambiguities
- ▣ No unbound type parameters
- ▣ No undefined exports
- ▣ No stale dependencies
- ▣ Proper compat bounds

## License

MIT

---

## 🎉 Nice Commit: 4f4f63b69c9b5e2a3438d23c3dd7ef927e645319

This commit hash contains **69**! Celebratory black hole:

```
[38;2;214;181;144m([0mring 0.69 0.169[38;2;214;181;144m)[0m[38;2;228;3;3m [38;2;255;140;0m+[38;2;255;237;0m [0m[38;2;0;77;151m([0mgaussian 0.42 0.42[38;2;0;77;151m)[0m[38;2;228;3;3m [38;2;255;140;0m+[38;2;255;237;0m [0m[38;2;176;175;176m([0mring 1.069 0.269[38;2;176;175;176m)[0m
```

---

## 🎉 Nice Commit: 063d164ec7e9930e169938aa03cac1dc6c1fc45f

This commit hash contains **69**! Celebratory black hole:

```
[38;2;214;181;144m([0mring 0.69 0.169[38;2;214;181;144m)[0m[38;2;228;3;3m [38;2;255;140;0m+[38;2;255;237;0m [0m[38;2;0;77;151m([0mgaussian 0.42 0.42[38;2;0;77;151m)[0m[38;2;228;3;3m [38;2;255;140;0m+[38;2;255;237;0m [0m[38;2;176;175;176m([0mring 1.069 0.269[38;2;176;175;176m)[0m
```

---

## 🎉 Nice Commit: 2a7704f91413396abca659adf4ffe696f39b0156

This commit hash contains **69**! Celebratory black hole:

```
[38;2;214;181;144m([0mring 0.69 0.169[38;2;214;181;144m)[0m [38;2;25;63;230m+[0m [38;2;0;77;151m([0mgaussian 0.42 0.42[38;2;0;77;151m)[0m [38;2;25;63;230m+[0m [38;2;176;175;176m([0mring 1.069 0.269[38;2;176;175;176m)[0m
```
