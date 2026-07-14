# Gay.jl

Deterministic semantic color, explicit SplitMix64 conventions, and
evidence-preserving visual and auditory projections for Julia.

Gay.jl turns a seed and an index into reproducible color while keeping the
algorithmic convention visible. Its compact core depends only on Julia standard
libraries; color science, persistent homology, and fractal-dimension analysis
load through package extensions.

## Current lineage

The current `gay` branch is the compact kernel lineage:

```text
name    = Gay
uuid    = 8b449cd3-8280-14dd-1069-000000000042
version = 0.5.0
```

An older monorepo lineage has a different UUID and different behavior. It is
preserved at `lineage/monorepo-v0.1.0`; it is not ordinary branch drift. See
[`LINEAGE.md`](LINEAGE.md) for the append-only lineage policy.

Two packages with the same Julia module name cannot be selected together by an
ordinary `using Gay` environment. Cross-lineage work therefore uses an explicit
Arrow, DuckDB, CSV, or JSON boundary carrying the package UUID, commit, seed,
index, and algorithm convention.

## Install

```julia
using Pkg
Pkg.add(url="https://github.com/bmorphism/Gay.jl")
```

Optional analysis packages can be added independently:

```julia
Pkg.add(["Colors", "FractalDimensions", "Ripserer", "PersistenceDiagrams"])
```

## Three color surfaces

```julia
using Gay

seed = UInt64(0xC91F14)

color_at(68; seed=seed)       # split-lattice HSL-like walk, O(index)
hash_color_hex(seed, 68)      # XOR-addressed RGB hash, O(1)
spi_color_hex(seed, 68)       # cross-runtime additive SPI kernel, O(1)
```

These functions deliberately answer different questions:

| surface | recurrence | projection | role |
|---|---|---|---|
| `color_at` | repeated `split()` | simplified HSL-like projection | legacy sequential palette |
| `hash_color_*` | `M(seed xor index*G)` | low byte first | O(1) GPU-portable indexed hash |
| `spi_color_*` | `M(seed + index*G)` | packed `0xRRGGBB` | cross-runtime canonical SPI kernel |

Here `M` is the SplitMix64 finalizer and
`G = 0x9e3779b97f4a7c15`. All `UInt64` arithmetic is modulo `2^64`.

## What “extra advance” means

For an XOR-addressed indexed word

```text
x_i = seed xor (index * G),
```

the two conventions are:

```text
no advance:     C_no(i)  = RGB24(M(x_i))
extra advance:  C_adv(i) = RGB24(M(x_i + G))
```

No advance treats `x_i` as the word to finalize. Extra advance treats `x_i`
as RNG state immediately before `next()`, so the state advances by `G` before
finalization.

Extra advance is not stronger randomness. It is a state-origin convention.
Because indexed addressing uses XOR while advancement uses addition,
`C_adv(i)` is generally not `C_no(i + 1)`: XOR and addition do not commute.

This lineage makes the distinction explicit:

- `hash_color_*` is the no-advance XOR convention;
- `split_mix_64(x)` is the advance-then-finalize primitive `M(x + G)`;
- `spi_color_*` uses additive addressing directly and does not add a hidden
  extra step;
- `color_at` is a separate split-lattice walk, not either indexed hash.

The archived monorepo lineage used the extra-advance convention in its main
O(1) `hash_color`/`color_at` path. A seed alone therefore cannot identify a
cross-lineage color. Persist at least:

```text
package_uuid, package_commit, seed, index, gamma,
address_rule, advance_rule, byte_projection
```

Both 64-bit mixers are bijective before projection, but RGB24 projection is
not. Exact color collisions remain possible and expected; neither indexed
kernel is a self-avoiding color walk.

## Parallel invariance

The `spi_*` surface is the cross-runtime contract. It supports stateless random
access and an associative XOR-fold fingerprint:

```julia
using Gay

sequential = spi_xor_fingerprint(42, 0, 1_000_000)
parallel = spi_xor_fingerprint_parallel(42, 1_000_000; chunks=4)
@assert sequential == parallel
```

Pinned Julia vectors and the optional `libspi` FFI cross-validation check Julia
against the Zig reference through its C ABI. Swift/Metal and other consumers
are intended to implement the same byte contract, but each consumer requires
its own comparison evidence.

## FractalDimensions extension

Loading `FractalDimensions` activates
[`GayFractalExt`](ext/GayFractalExt.jl). The extension accepts:

- a vector of `#RRGGBB` strings;
- a `WalkResult`;
- an integer number of sequential `color_at` samples.

```julia
using Gay, FractalDimensions

seed = UInt64(0xC91F14)
colors = [hash_color_hex(seed, i) for i in 0:511]
D = gay_fractal_dimension(colors; metric=:euclidean, show_progress=false)
```

The integer overload `gay_fractal_dimension(n)` generates `color_at(0:n-1)`.
Comparisons between indexed kernels must therefore construct each color vector
explicitly and pass the vector overload.

Loading `Colors` as well enables the perceptual path, where pairwise distance is
CIEDE2000 through `gay_colordiff`. Without it, the extension warns and falls
back to Euclidean distance in sRGB.

The current automatic perceptual path is diagnostic rather than a clean
CIEDE2000 correlation dimension: its default `epsilon` grid is estimated from
the extension's one-dimensional index surrogate before the custom CIEDE2000
norm is applied. A scientifically interpreted perceptual curve must instead
derive its scale grid from perceptual pairwise distances and call the underlying
FractalDimensions correlation-sum API explicitly.

### Why one dimension is not an algorithm identifier

An unordered RGB point cloud from either indexed convention resembles a nearly
uniform sample. A single global correlation dimension therefore cannot reliably
reconstruct the state-origin convention. The ordinary correlation sum with
Theiler window `w=0` mostly discards sequence order.

To compare causal color chains, retain one or more order-aware witnesses:

- the full correlation curve `C(epsilon)` and its local slopes;
- pointwise dimensions replayed in index order;
- a nonzero Theiler window;
- Higuchi length and dimension on R, G, B, luminance, or unwrapped hue;
- delay embeddings of those indexed channels;
- persistent-homology births and lifetimes.

FractalDimensions estimates scaling behavior, typically fitting

```text
log C(epsilon) approximately equals D * log epsilon.
```

Convenience functions ending in `_dim` automate scale and fit-region choices.
Reproducible scientific output should retain the scale curve, preprocessing,
metric, fit configuration, and uncertainty rather than only the returned
scalar.

## Sonifying fractal evidence

The package does not ship an audio engine. Gay supplies a deterministic seed and
color token; FractalDimensions supplies the multiscale evidence; a separate
renderer assigns that token to a voice and produces PCM, MIDI, OSC, or another
auditory carrier.

The evidence-preserving mapping is:

| Fractal evidence | Auditory mapping |
|---|---|
| `log(epsilon)` | event time |
| local slope `d log(C) / d log(epsilon)` | pitch |
| `C(epsilon)` | gain or event density |
| confidence interval or fit residual | detuning, vibrato, or noise width |
| pointwise dimension | melody or polyphony |
| posterior draw, worker, or chain | voice and pan |
| Gay color identity | categorical timbre identity |
| persistence birth and lifetime | onset and duration |

Dimension-to-pitch should be monotone. Hashing the final dimension would destroy
neighborhood structure: nearby estimates could sound unrelated while distant
estimates could collide.

### Minimal event extractor

The following adapter exposes the evidence needed by an external renderer:

```julia
using Gay, FractalDimensions

function rgb_state_space(colors)
    data = Matrix{Float64}(undef, length(colors), 3)
    for (i, hex) in pairs(colors)
        data[i, 1] = parse(Int, hex[2:3]; base=16) / 255
        data[i, 2] = parse(Int, hex[4:5]; base=16) / 255
        data[i, 3] = parse(Int, hex[6:7]; base=16) / 255
    end
    StateSpaceSet(data)
end

function fractal_score(colors, source_id;
                       chain=0, fit=LargestLinearRegion())
    X = rgb_state_space(colors)
    epsilon = estimate_boxsizes(X)
    C = correlationsum(X, epsilon; show_progress=false)

    keep = (C .> 0) .& isfinite.(C)
    epsilon_fit = epsilon[keep]
    x = log2.(epsilon_fit)
    y = log2.(C[keep])
    D, D_low, D_high = slopefit(x, y, fit)

    D_local = diff(y) ./ diff(x)
    event_time_logscale = (x[1:end-1] .+ x[2:end]) ./ 2

    # Proposed monotone rendering: D in [0, 3] maps to MIDI 48:84.
    midi = 48 .+ 12 .* clamp.(D_local, 0, 3)
    frequency_hz = 440 .* 2 .^ ((midi .- 69) ./ 12)

    identity_seed = stable_seed(source_id)
    identity_color = hash_color_hex(identity_seed, chain)

    (; source_id, colors, chain, metric=:euclidean,
       preprocessing=:rgb_unit_cube, fit,
       epsilon, C, keep, epsilon_fit, D, D_low, D_high,
       D_local, event_time_logscale, frequency_hz,
       identity_seed, identity_color)
end
```

The MIDI range above is a rendering choice, not a scientific invariant. The
returned record keeps the source colors and estimator choices needed to replace
the shown pitch renderer. A downstream renderer can normalize the log-scale
event positions into its desired playback interval.

## Operational cobordism with the ptiede Julia ecosystem

There is no separate `ptiede` algebra in this package. The name refers to Paul
Tiede’s Julia ecosystem around Comrade, VIDA, and parallel posterior inference.
The archived monorepo described this architectural path:

```text
Comrade.jl -> Pigeons.jl -> SplittableRandoms.jl -> Gay.jl
```

That path is an interoperability map, not a dependency claim. Comrade,
VLBISkyModels, VIDA, and Pigeons are not dependencies of the compact Gay core.

The practical cobordism is a shared evidence object with two renderings:

```text
Comrade / VIDA posterior image ---\
fNIRS or another timeseries -------+--> StateSpaceSet
Gay indexed color walk -----------/          |
                                      FractalDimensions
                                             |
                  E = {epsilon, C, D_local, D, CI, provenance}
                                   /                 \
                         Gay color boundary     audio boundary
```

[`VIDA.jl`](https://github.com/ptiede/VIDA.jl) supplies image-domain feature
extraction through ComradeBase and VLBISkyModels interfaces.
[`Pigeons.jl`](https://github.com/Julia-Tempering/Pigeons.jl) supplies parallel
posterior draws. Dimension can be estimated per image or posterior draw, while
Gay assigns a stable color token that the auditory renderer can bind to a
voice.

The middle evidence object establishes the correspondence. A shared seed alone
does not prove that visual and auditory outputs represent the same observation.

### Neutral evidence schema

DuckDB, Arrow, CSV, or JSON records should include at least:

```text
source_id, draw_id, chain_id, worker_id, index,
package_uuid, package_commit, seed, gamma,
address_rule, advance_rule, byte_projection,
estimator, metric, preprocessing, theiler_window,
epsilon, correlation_sum, local_dimension, global_dimension,
ci_low, ci_high, fit_region,
color_hex, pitch_hz, gain, pan
```

This schema also lets the archived and current Gay lineages interoperate without
ambiguous same-name package resolution.

## fNIRS and BCI trajectories

For fNIRS, an order-aware score can use sliding-window `higuchi_dim` on HbO and
HbR, or delay-embed each channel before estimating local or correlation
dimension:

```text
window time             -> event time
D_HbO(t), D_HbR(t)      -> separate pitches or voices
hemodynamic amplitude   -> gain
dimension uncertainty   -> vibrato or detuning width
channel or participant  -> stable Gay voice identity
```

The ordered hemodynamic signal remains the scientific input. Gay color and
sound are queryable projections with provenance, not replacements for the
measurement.

## Other optional extensions

| load | extension | result |
|---|---|---|
| `using Colors` | `GayColorsExt` | CIEDE2000 comparison on hex colors |
| `using FractalDimensions` | `GayFractalExt` | Grassberger-Procaccia diagnostics |
| `using Ripserer` | `GayRipsererExt` | persistent homology of color walks |
| `using PersistenceDiagrams, Ripserer` | `GayPersistenceDiagramsExt` | bottleneck, Wasserstein, and matching operations |

The dated
[`docs/color_topology_integration_memo.md`](docs/color_topology_integration_memo.md)
records the pre-promotion integration investigation. Its branch and version
status is historical; the current extension source and test suite are the
authority for v0.5.0.

## Verify

```sh
julia --project -e 'using Pkg; Pkg.test()'
```

Optional live ABI comparison requires a built `spi-race` library:

```sh
SPI_LIB=/path/to/libspi.dylib julia --project scripts/spi_ffi_crossvalidate.jl
```

## Epistemic contract

- A seed is presentation identity, not provenance or authentication.
- A color collision does not imply an evidence collision.
- A global fractal dimension does not identify an ordered color algorithm.
- An automatically selected scaling region is a hypothesis to inspect.
- A deterministic audio mapping is not canonical merely because it repeats.
- Package, commit, kernel convention, metric, estimator, and source record
  together define a reproducible rendering.
