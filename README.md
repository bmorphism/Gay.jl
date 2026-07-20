# Gay.jl

Semantic color for Julia: deterministic wide-gamut colors, GF(3) trits,
non-Riemannian perceptual distance, and SPI-safe parallel streams.

Gay.jl descends from Comrade.jl-style compositional scientific models and
Pigeons.jl-style Strong Parallelism Invariance. For `bci.place`, it is the color
clock for `sense -> trit -> color -> predict -> observe -> verify`.

## Install

```julia
using Pkg
Pkg.add(url="https://github.com/bmorphism/Gay.jl")
```

## Use

```julia
using Gay

gay_seed!(1069)
color_at(1)
palette_at(1, 6)
verify_genesis_chain()
```

```julia
using Gay, OhMyThreads

seed = 42
xs = [color_at(i; seed) for i in 1:100]
ys = tmap(i -> color_at(i; seed), 1:100)
@assert xs == ys
```

## Surface

- Color: `color_at`, `colors_at`, `palette_at`, `SRGB`, `DisplayP3`, `Rec2020`.
- Streams: `gay_seed!`, `GayRNG`, `gay_split`, `GAY_SEED`.
- GF(3): `PLUS`, `ERGODIC`, `MINUS`, `balanced`.
- Time/perception: `TritTick`, `perceptual_diff_sat`.
- BCI/reafference: `BCISource`, `HeartbeatSource`, `CompositeSource`,
  `reafference_challenge`, `ReafferenceProof`.
- Comrade-style models: `comrade_ring`, `comrade_gaussian`, `sky_add`,
  `comrade_show`.

## Contract

```text
same seed + same index = same color
parallel order does not change results
valid GF(3) composition balances to zero
large color differences use saturating, path-aware perception
```

Live signals can become `ColoredTick`s; colored paths can be re-predicted; and
capabilities can be gated by SPI fingerprints, GF(3) conservation, and
reafference instead of ambient state.

## Develop

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=. -e 'using Pkg; Pkg.precompile()'
julia --project=. scripts/lint_no_demo.jl
```

The former long README is archived at
[.topos/Gay.jl.README.md](.topos/Gay.jl.README.md). Keep this file as the Julia
package surface; keep theory, provenance, and speculative material under
`.topos/`.
