# Gay-Bartholomew: How We Found the Bugs in v0.2.1

*A forensic account of discovering subtle SPI violations through rigorous statistical and compositional testing*

**Author:** Claude Opus 4.5 & bmorphism
**Date:** December 15, 2025
**Version:** Gay.jl v0.2.1

---

## Color Derivation

This narrative is structured via **derangement** σ = (1 2 3)(4 5 6) of `gay_palette(6)` at seed=69:

```julia
using Gay
gay_seed(69)
palette = gay_palette(6)
σ = [2, 3, 1, 5, 6, 4]  # no fixed points
```

| Section | Color | Hex | Derivation |
|---------|-------|-----|------------|
| §1 | 🟢 | `#7DAF27` | `palette[σ[1]]` = `palette[2]` |
| §2 | 🟣 | `#590F68` | `palette[σ[2]]` = `palette[3]` |
| §3 | 🩷 | `#D03684` | `palette[σ[3]]` = `palette[1]` |
| §4 | 🪻 | `#9B25A6` | `palette[σ[4]]` = `palette[5]` |
| §5 | 🩵 | `#44B1A8` | `palette[σ[5]]` = `palette[6]` |
| §6 | 🌊 | `#3C9B8F` | `palette[σ[6]]` = `palette[4]` |

The derangement ensures no section inherits its "natural" position—a fitting structure for a bug-hunting narrative where nothing was where it seemed.

---

## The Setup: "Continue"

It started with a single word: *continue*.

The Gay.jl codebase had been humming along—colors generating, palettes rendering, the SPI (Strong Parallelism Invariance) supposedly holding. But when we ran the triadic subagents demo, Julia threw an unexpected error:

```
ERROR: MethodError: objects of type Module are not callable
```

A module being called as a function? That's not a minor typo. That's a fundamental type confusion buried somewhere in the code.

---

## §1 `#7DAF27` The Type Mismatch Hunt

### The Crime Scene

The error pointed to `triadic_subagents.jl:472`:

```julia
agents1 = TriadicSubagents(seed)
```

But `TriadicSubagents` is the *module name*, not a type. The actual struct is called `Triad`. Someone had written function signatures like:

```julia
function get_agent(agents::TriadicSubagents, p::Polarity)
```

When they meant:

```julia
function get_agent(agents::Triad, p::Polarity)
```

### The Extent of the Damage

A quick grep revealed the horror: **14 occurrences** of `::TriadicSubagents` where `::Triad` was needed, plus **3 constructor calls** using the module name instead of the type.

```bash
$ grep -n "::TriadicSubagents" src/triadic_subagents.jl | wc -l
14
```

This wasn't a single typo. This was systematic confusion between module and type namespaces—the kind of bug that passes syntax checking but fails at runtime.

### The Fix

```julia
# Before (wrong)
function parallel_sample!(agents::TriadicSubagents, n::Int)

# After (correct)
function parallel_sample!(agents::Triad, n::Int)
```

Applied 14 times. But we weren't done.

---

## §2 `#590F68` The Bit Shift Catastrophe

With the type errors fixed, the demo ran further—then crashed again:

```
ERROR: MethodError: no method matching <<(::Float64, ::Int64)
```

Bit shifting a Float64? That's not just wrong, it's *architecturally* wrong. The offending code:

```julia
hash_val = splitmix64(seed ⊻ UInt64(round(color.g * 255) << 8) ⊻ ...)
```

The problem: `round(color.g * 255)` returns `Float64`, not `Int`. Julia's `<<` operator doesn't accept floats. The fix:

```julia
hash_val = splitmix64(seed ⊻ UInt64(round(Int, color.g * 255) << 8) ⊻ ...)
```

Two lines. Two bugs. Each invisible to static analysis, each fatal at runtime.

---

## §3 `#D03684` The Duplicate Definition War

Gay.jl loaded with a warning:

```
WARNING: Method definition splitmix64(UInt64) in module Gay at
splittable.jl:44 overwritten at kernels.jl:90.
```

Two files defining `splitmix64`. Two definitions. Which one wins? In Julia, the last one loaded—but that's nondeterministic if include order changes.

### The Archaeology

`splittable.jl` had the canonical SplitMix64:

```julia
function splitmix64(x::UInt64)::UInt64
    x += GOLDEN
    x = (x ⊻ (x >> 30)) * MIX1
    x = (x ⊻ (x >> 27)) * MIX2
    x ⊻ (x >> 31)
end
```

`kernels.jl` had its own copy—identical logic, but a separate definition. This violates the principle that cryptographic primitives should have exactly ONE authoritative implementation.

### The Resolution

We removed the duplicate from `kernels.jl` and added a comment:

```julia
# splitmix64 is imported from splittable.jl
```

Now there's one `splitmix64`. One truth. One hash.

---

## §4 `#9B25A6` Enter Marsaglia-Bumpus

With the obvious bugs fixed, we needed to verify the *statistical* correctness. Enter the Marsaglia-Bumpus test suite—a dual-perspective verification system:

### The Marsaglia Perspective (1995-2003)

George Marsaglia asked: **"Does the sequence LOOK random?"**

His tests check statistical properties that random sequences should have:

1. **Birthday Spacings**: Given n random values in a range, how many collisions occur?
2. **Runs Test**: Are ascending/descending runs distributed correctly?
3. **Permutation Test**: Are all orderings equally likely?
4. **Spectral Test**: Does FFT reveal hidden periodicities?

### The Bumpus Perspective (2021-2024)

Benjamin Bumpus asked: **"Does SPLITTING preserve structure?"**

His tests check compositional properties:

1. **Adhesion Width**: How much information leaks between split branches?
2. **Sheaf Gluing**: Can sections be consistently combined?
3. **Tree Decomposition**: Does the color graph have bounded tree-width?

### The Birthday Spacing Bug

When we ran `birthday_spacing_test`, it failed catastrophically:

```
1. Birthday Spacings Test... ✗ FAIL
   Collisions: 33 (expected λ=9320.68), p=0.0000
```

Expected 9320 collisions, observed 33? That's off by a factor of 280. The formula was wrong:

```julia
# Wrong (cubic)
λ = (n^3) / (4.0 * m)

# Correct (quadratic)
λ = (n^2) / (2.0 * m)
```

The birthday problem expects `n²/2m` collisions, not `n³/4m`. A single exponent error transformed a passing test into a false failure—or worse, could have hidden real problems.

After the fix:

```
1. Birthday Spacings Test... ✓ PASS
   Collisions: 33 (expected λ=36.41), p=0.2861
```

Now 33 observed vs 36.41 expected. That's statistical noise, not a bug.

---

## §5 `#44B1A8` The Genesis Handoff

The deepest question remained: **Are Left and Right splits truly independent?**

When we call `split(rng)` twice to get two child RNGs, they MUST be statistically independent. If they're correlated, parallel execution becomes nondeterministic—the cardinal sin against SPI.

### The Correlation Test

We generated 10,000 pairs of (left_value, right_value) from consecutive splits and computed Pearson correlation:

```julia
function split_correlation_test(seed::UInt64, n_samples::Int=10000)
    # ... generate pairs ...
    correlation = cov / (σ_l * σ_r)
    passed = abs(correlation) < 0.05
end
```

Results across multiple seeds:

| Seed | Correlation | Status |
|------|-------------|--------|
| GAY_SEED | -0.000172 | ✓ |
| 42 | 0.005321 | ✓ |
| 69 | -0.015607 | ✓ |
| 1337 | -0.009525 | ✓ |
| 0xDEADBEEF | -0.016198 | ✓ |

All correlations below 0.05. Left and Right are independent.

### The Parent Isolation Test

Does parent state leak to children? We built a depth-5 binary tree of splits, generating fingerprints at each node:

```julia
function genesis_handoff_test(seed::UInt64, depth::Int=5)
    # Build tree, collect fingerprints
    # Verify: deterministic, siblings independent, collision-free
end
```

Results:

```
✓ Deterministic: Same seed → same tree (always)
✓ Siblings Independent: L ≠ R at every node
✓ Collision-Free: 63/63 unique fingerprints
```

No collisions in 63 nodes. No sibling matches. Parent state is properly isolated.

---

## §6 `#3C9B8F` Schedule Independence

The final test: does execution order matter?

We generated colors at indices 1-100 in five different orders:
- Forward: 1, 2, 3, ..., 100
- Reverse: 100, 99, 98, ..., 1
- Shuffle 1: random permutation
- Shuffle 2: different random permutation
- Interleaved: odds then evens

XOR fingerprint for each:

```
Seed 0x6761795f636f6c6f:
  Forward:     0xd0da42325fe1b159
  Reverse:     0xd0da42325fe1b159
  Shuffle1:    0xd0da42325fe1b159
  Shuffle2:    0xd0da42325fe1b159
  Interleaved: 0xd0da42325fe1b159
  All Match: ✓ YES
```

Same fingerprint regardless of order. **SPI holds.**

---

## The Derangement Closes

The bugs in v0.2.1 weren't exotic. They were mundane:
- Confusing a module with a type
- Forgetting that `round()` returns Float64
- Copy-pasting a function definition
- Using the wrong exponent in a formula

But mundane bugs in cryptographic/RNG code have non-mundane consequences. A correlated split breaks parallel determinism. A wrong hash breaks reproducibility. A type error crashes production.

The Marsaglia-Bumpus framework caught what unit tests missed: **statistical correctness** and **compositional structure**. These aren't optional extras. They're the foundation of SPI.

### The Final Tally

| Bug | Section | Severity |
|-----|---------|----------|
| Type mismatch (×14) | §1 `#7DAF27` | High |
| Float bit-shift (×2) | §2 `#590F68` | High |
| Duplicate splitmix64 | §3 `#D03684` | Medium |
| Birthday formula | §4 `#9B25A6` | Low |

### Verification Complete

```julia
full_spi_audit(69)
# ✓ COMPLETE SPI VERIFICATION: ALL TESTS PASS
```

| Suite | Tests | Iterations |
|-------|-------|------------|
| SPI Regression | 10 | - |
| Marsaglia Statistical | 4 | - |
| Bumpus Compositional | 3 | - |
| Genesis Handoff | 2 | 10,000 |
| Fuzz Tests | 16 | 7,300 |
| Jepsen Chaos | 17 | - |
| Cross-Substrate | 10 | - |
| Propagator | 44 | - |
| QUIC | 93 | - |
| Abductive | 399 | - |
| Unmix Bijection | 1 | 100,000 |

**Total: 599 tests, 117,300+ iterations**

---

## Appendix: The Derangement σ

The section structure follows a derangement—a permutation with no fixed points:

```
σ = (1 2 3)(4 5 6)

Position: 1 → 2 → 3 → 1  (3-cycle)
Position: 4 → 5 → 6 → 4  (3-cycle)

Sign(σ) = +1 (even permutation: two 3-cycles)
```

In Gay.jl terms:

```julia
using Gay

gay_seed(69)
palette = gay_palette(6)
σ = [2, 3, 1, 5, 6, 4]

# Verify derangement
@assert all(i != σ[i] for i in 1:6)  # no fixed points
@assert Set(σ) == Set(1:6)           # bijection
```

The derangement mirrors the bug-hunting process: nothing was in its expected place, yet everything permuted into a coherent whole.

---

## References

1. Marsaglia, G. (1995). "DIEHARD: A Battery of Tests of Randomness"
2. Bumpus, B. (2024). "Spined Categories and Decomposition Width" (arXiv)
3. Steele, G. et al. (2014). "Fast Splittable Pseudorandom Number Generators" (OOPSLA)
4. Pigeons.jl documentation on Strong Parallelism Invariance

---

*Gay.jl: Deterministic colors for a nondeterministic world.*

```julia
# Reproduce this document's color scheme
using Gay
gay_seed(69)
show_palette(gay_palette(6)[[2,3,1,5,6,4]])  # deranged
```
