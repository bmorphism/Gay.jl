# Gay.jl Color Type Survey

All Color instances, types, and references across the codebase.

## Color Type Hierarchy

```
Colors.Color (base)
├── RGB{Float64}     — primary: splittable.jl, gaymc.jl, quic.jl, sheaf_acset, tracking, serialization
├── RGB{Float32}     — kernel_triad.jl (GPU-compatible)
├── HSL              — protocol.jl
└── Gay-specific:
    ├── PadicColor           — splittable.jl (p-adic ultrametric, 3 PadicChannels)
    ├── ColorSpace (abstract) — colorspaces.jl
    │   ├── SRGB
    │   ├── DisplayP3
    │   ├── Rec2020
    │   └── CustomColorSpace
    ├── ColorMorphism{C}     — sheaf_acset_integration.jl (provenance tracking)
    ├── ChromaticBag{T,C}    — sheaf_acset_integration.jl (colored element collections)
    ├── Color (struct)       — fault_tolerant.jl (Galois connection event mapping)
    ├── ColoredXref          — binary.jl (radare2 cross-references)
    └── KernelColorContext   — kernel_lifetimes.jl (mutable lifecycle tracking)
```

## Files Using Colors

| File | Import | Types Used | Purpose |
|------|--------|------------|---------|
| `splittable.jl` | implicit | `RGB{Float64}`, `PadicColor` | Core color gen: `next_color`, `color_at`, `padic_color` |
| `colorspaces.jl` | `Colors, ColorTypes` | `ColorSpace`, `SRGB`, `P3`, `Rec2020` | Wide-gamut color space management |
| `gaymc.jl` | `Colors` | `RGB{Float64}` | Monte Carlo color sampling, `color_history` |
| `kernel_triad.jl` | `Colors: RGB` | `RGB{Float32}` | GF(3) triad colors with polarity twists |
| `kernel_lifetimes.jl` | `Colors: RGB` | `RGB`, `KernelColorContext` | Eventual color prediction, O(1) lifecycle |
| `sheaf_acset_integration.jl` | `Colors` | `RGB{Float64}`, `ColorMorphism`, `ChromaticBag` | Sheaf-theoretic decomposition with color tracking |
| `quic.jl` | `Colors: RGB` | `RGB{Float64}` | QUIC path probe coloring |
| `binary.jl` | — | `ColoredXref` | Radare2 colored cross-references |
| `fault_tolerant.jl` | — | `Color` (struct) | Galois connections between events and colors |
| `xypic.jl` | `Colors` | `Color`, `RGB` | LaTeX xy-pic color chains |
| `repl.jl` | `Colors: RGB` | `Color` | REPL inline color display |
| `protocol.jl` | `Colors: RGB, HSL` | `RGB`, `HSL` | Protocol color encoding |
| `bench.jl` | `Colors: RGB` | `RGB` | Benchmark result types |
| `enzyme.jl` | `Colors` | — | Enzyme AD color perturbations |
| `okhsl_learnable.jl` | `Colors` | — | Learnable OKHsl color spaces |
| `we-ness.jl` | `Colors` | — | We-ness intersubjective coloring |
| `comrade.jl` | `Colors: RGB` | `RGB` | Comrade parallel color verification |
| `tensor_parallel.jl` | `Colors: RGB` | `RGB` | Tensor-parallel hidden state coloring |
| `lifetimes.jl` | `Colors: RGB` | `RGB` | Color lifetime tracking |
| `tracking.jl` | `Colors: RGB` | `RGB` | Color generation tracking |
| `serialization.jl` | `Colors: RGB` | `RGB` | S-expression color serialization |
| `abductive.jl` | `Colors: RGB` | `RGB` | Abductive testing color derangements |
| `triadic_subagents.jl` | `Colors: RGB` | `RGB` | Subagent color identity |
| `marsaglia_bumpus_tests.jl` | implicit | — | Marsaglia randomness tests on color streams |

## Key Color Functions

- `next_color(cs)` — advance RNG, return RGB (splittable.jl)
- `color_at(idx, cs; seed)` — O(1) index-based color (splittable.jl)
- `hash_color(idx, seed)` — GPU-compatible UInt64→RGB (kernels.jl)
- `padic_color(gen)` — p-adic ultrametric color generation (splittable.jl)
- `random_color(cs)` — random color in color space (colorspaces.jl)
- `blend_colors(c1, c2)` — RGB midpoint blend (sheaf_acset_integration.jl)
- `colors_compatible(c1, c2)` — hue proximity check (sheaf_acset_integration.jl)
- `show_color_inline(c)` — ANSI terminal preview (repl.jl)

## Stats

- **41 files** reference Color/RGB in src/
- **9 distinct Color types** defined in Gay.jl
- **3 color precisions**: Float64 (CPU), Float32 (GPU), p-adic (ultrametric)
- **4 color spaces**: sRGB, Display P3, Rec.2020, Custom
