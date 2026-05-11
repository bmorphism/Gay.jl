# Color Bandwidth — Kernel Abstraction (KA)

```@meta
CurrentModule = Gay.NextColorBandwidth
```

The `NextColorBandwidth` module reframes Gay.jl as a **Shannon-tight performance benchmark**: how many *distinguishable* colors per second can flow through the control surface while remaining perceptually coherent and deterministically reproducible.

The unit is **bits/second**, not colors/second. The colors/second number is the raw kernel throughput; the bits/second number is `distinguishable/sec × log₂(n_distinct)` and is bounded by the Shannon channel capacity `C = max_{p(x)} I(X;Y)` of the 5-level PCT cascade with the perceptual JND filter as the channel.

## Why bits/s, not FLOPS

MLPerf reports tokens/s, queries/s, samples/s; HPC reports FLOPS. None of these are information-theoretic — they count operations, not the *information* carried through the system. arXiv:2508.05621 (2025) proposes mutual information `I(X;Y)` as a substrate-neutral computing-performance unit:

> *"Channel capacity (bits per channel use) converts to throughput (bit/s) by multiplying by channel use rate (e.g., clock frequency), analogous to deriving flop/s. For parallel systems, independent channel capacities/throughputs sum due to mutual information additivity."*

`NextColorBandwidth` is the runnable instance: the channel is the PCT cascade, the input distribution is the SplittableRandoms.jl seed-derived color request, the output is the Lab-ΔE-distinguishable color, the JND threshold is the per-channel-use noise floor.

## Quick start

```julia
using Gay
using Gay.NextColorBandwidth

# Shannon-tight channel capacity in bits/s on the default cascade
C = compute_channel_capacity(1000)

# Full measurement carrying SPI verification + fingerprint
r = measure_at_scale(1000)
r.bandwidth.colors_per_second        # raw kernel throughput
r.bandwidth.distinguishable_per_second
r.bandwidth.bits_per_color           # log₂(n_distinct)
r.bandwidth.channel_capacity         # bits/s (Shannon-tight)
r.bandwidth.convergence_margin       # 1 = safe, 0 = edge of chaos
r.spi_verified                       # bit-exact reproducibility (SPI)
fingerprint_bandwidth(r)             # 64-bit deterministic run identity
```

## The SPI guarantee

Every measurement is reproducible byte-for-byte across cores, threads, machines, and OSes — this property is inherited from [Pigeons.jl](https://pigeons.run) and the `SplittableRandoms.jl` substrate (arXiv:2308.09769, *Strong Parallelism Invariance*). The fingerprint `fingerprint_bandwidth(r)` hashes `(seed, n_distinguishable, channel_capacity)` and is forgery-resistant: a single bit flip anywhere in the input distribution or the cascade changes the hash deterministically.

This is the dimension MLPerf does not score. MLPerf's Inference rules permit up to 5% replicability variance over 5 retries; certifiable-bench's bit-identity gate (SHA-256 over outputs) is binary. NCB's SPI fingerprint is strictly more informative: it both witnesses bit-identity **and** bounds the Shannon channel capacity, in one verifiable artifact.

## Parallelism levels

```@docs
ParallelismLevel
```

Default `OUTER_INNER` runs the sequential reference cascade; `TERNARY` performs the GF(3) 3-way split (each stream gets `n÷3` colors with seeds `seed`, `splitmix64(seed ⊻ 1)`, `splitmix64(seed ⊻ 2)`). At small `n` the fork overhead dominates and OUTER_INNER wins; at large `n` TERNARY amortizes.

```julia
benchmark_all_levels(n_colors=10_000)   # comparison across available levels
```

## Edge of chaos

The PCT cascade has two control knobs: `gain` (how aggressively to enforce the descending-error chain) and `disturbance` (the noise that drives sampling diversity). The optimum lies at the edge: maximum distinguishability just before the control surface loses convergence.

```julia
limit = find_bandwidth_limit()        # binary search on disturbance
opt   = maximize_bandwidth!(n_colors=500)
```

`maximize_bandwidth!` performs a grid search over `gain × disturbance`, requiring `convergence_margin > 0.3` as a constraint, and returns the `(gain, disturbance)` that maximizes channel capacity.

## ACSS isomorphism — same KA, different substrate

The control math is invariant under substrate change. The same channel-capacity computation applies to aural perception by swapping in species-specific perceptual JND thresholds:

| Substrate | Hue → | Lightness → | Gamut volume → |
|---|---|---|---|
| Color | pitch | volume | spatial × frequency × amplitude |
| Aural | (1200 cents/oct ÷ JND) pitches | (120 dB ÷ JND) volumes | 360° ÷ azimuth-JND |

```julia
opt_color_bw = maximize_bandwidth!().result.bandwidth
maximize_aural_bandwidth(opt_color_bw)   # 7 species: human, dolphin, humpback, ...
```

The dolphin gets ~7× the human aural-channel capacity at the same color-channel capacity, purely from finer temporal resolution (0.3 ms vs 2 ms).

## API reference

```@docs
ColorBandwidth
BandwidthTest
BandwidthResult
measure_next_color_bandwidth
measure_at_scale
compute_channel_capacity
next_color_batch
next_color_parallel
stress_bandwidth
find_bandwidth_limit
scaling_curve
benchmark_all_levels
bandwidth_comparison
bandwidth_spi_check
fingerprint_bandwidth
maximize_bandwidth!
optimal_parallelism_level
demo_next_color_bandwidth
```

## Position in the benchmark landscape

| Dimension | MLPerf | certifiable-bench | Gay.jl NCB |
|---|---|---|---|
| Unit | tokens/s, queries/s, samples/s | latency + bit-identity gate | **bits/s (Shannon-tight)** |
| Reproducibility | best-effort, 5% over 5 retries | SHA-256 of outputs (binary gate) | **SPI fingerprint** (byte-exact, splittable) |
| Info-theoretic bound | none | none | **`C = max I(X;Y)`** measured per run |
| Conservation law | none | none | **GF(3) Σ mod 3** trit audit |
| Cross-substrate | per-suite silos | per-target | **ACSS isomorphism** (color ↔ aural, 7 species) |
| Failure mode visibility | accuracy drop | hash mismatch | **gamut saturation visible in scaling curve** |

## References

- [arXiv:2508.05621](https://arxiv.org/abs/2508.05621) — *Computational Work as Information Flow* (compute-channel capacity)
- [arXiv:2308.09769](https://arxiv.org/abs/2308.09769) — Surjanovic et al., *Pigeons.jl: Distributed sampling from intractable distributions* (SPI)
- [SpeyTech/certifiable-bench](https://github.com/SpeyTech/certifiable-bench) — bit-identity gate for safety-critical ML
- Zhuang/Hooker MLSys 2022 — *Randomness In Neural Network Training*, determinism overhead up to 746% on GPU
- [mlcommons/ck#1080](https://github.com/mlcommons/ck/issues/1080) — open proposal for MLPerf reproducibility badges (the attach point for an SPI-Reproducible tier)
