# GayLearnableColor.jl

A **jointly-learnable color space**, *as in Gay.jl colorings*: learn a color
embedding φ_θ : behaviour → Okhsl color, co-optimised **jointly** over all
behaviours coupled by their structural distance (MDS into a d-dim color space).
The learned coordinates **are** the behaviour embedding ⇒ behaviour and color are
one. Colors are emitted through `Gay.jl`'s own `okhsl_to_rgb`.

## API
```julia
using GayLearnableColor
D  = behaviors_gf3(24)                     # GF(3) 9-trit behaviours → Hamming matrix
lc = learn_colorspace(D; d=3, iters=500)   # → LearnedColorSpace(X, hexes, corr)
lc.corr                                     # structure preservation (Pearson)
lc.hexes                                     # n Okhsl colors via Gay.jl
graph_distance(adj, n)                       # shortest-path distances of a dyn. graph
```

## Result (`julia --project=. -e 'using Pkg; Pkg.test()'`, 8 assertions GREEN)
- 3-D Okhsl color space: morphisms corr **>0.6**, graph **>0.85**.
- **Color spaces matter**: 3-D beats 1-D hue (`lm.corr > l1.corr`) — 1-D hue is
  lossy for high-dim behaviour.
- Colors come from `Gay.jl` (`Gay.okhsl_to_rgb`); cross-substrate canon
  `hash_color_hex(GAY_SEED,0) == "#B35D38"` checked in-test.
- Deterministic (Gay `stable_seed`, not process-seeded `hash`).

## Design notes
- **Depends on `Gay.jl` by path** (`[sources] Gay = {path="../Gay.jl"}`), Julia 1.11+.
- Uses **analytic MDS gradients** — exact, so no AD library is required. Enzyme
  (bob's `GayEnzymeExt` LearnableOkhsl backend) would be the faithful AD swap but
  is numerically redundant here; the gradient of the stress is closed-form.
- bb concept twin: `~/worlds/scratch/gay-learnable-color/` (`world://gay-learnable-color`).
- Non-Riemannian scale extension notes live in
  [`docs/non_riemannian_color_scales.md`](docs/non_riemannian_color_scales.md).

Lineage: bob's learnable-Okhsl (Gay.jl) × jointly-learned structure-coloring ×
the same `mix64` palette as `world://securities`/`morphism`.
