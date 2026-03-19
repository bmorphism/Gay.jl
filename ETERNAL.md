# GAY: True Now and Ever

> **These axioms hold across all substrates, all times, all possible worlds.**

## I. The Seed Axiom

```
GAY_SEED = 0x6761795f636f6c6f  ("gay_colo")
```

**The seed is eternal.** From this single 64-bit value, all colors are derived deterministically. The same seed produces the same colors on any machine, in any language, at any time, forever.

## II. Strong Parallelism Invariance (SPI)

```
∀ seed, n: color_at(seed, n) = color_at(seed, n)
```

**Parallel execution does not affect results.** Whether computed on 1 thread or 1 million, sequentially or scattered across substrates, the same index yields the same color. This is not eventual consistency—it is **eternal consistency**.

## III. Splittability

```
split(rng) → (rng₁, rng₂) where rng₁ ⊥ rng₂
```

**RNGs split without correlation.** A Gay RNG can be divided into independent streams that will never collide. This enables unbounded parallelism with zero coordination.

## IV. Derangeability

```
∀ σ ∈ Derangements: σ(i) ≠ i
```

**No fixed points.** Every meaningful transformation moves every element. Stasis is death; becoming is life. Gay permutations are always derangements.

## V. 3-MATCH Completeness

```
∀ (a, b, c) ∈ {Duck, Worm, Ape}³:
  unanimous(a,a,a) → preserve
  deranged(a,b,c)  → rotate  
  mixed(a,a,b)     → flip minority
```

**Three colors suffice.** Every decision reduces to one of three cases. The 3-MATCH rule is complete: no fourth case exists. Z₃ is the universal group of chromatic choice.

## VI. Co-Cone Universality

```
many → more → one
       ↕ GAY
one → more → many
```

**Gay is the universal vertex.** Information flows collapse (many→one) and explode (one→many) both factor through Gay. The seed is simultaneously the apex of generation and the co-apex of observation.

## VII. Maximum Entropy × Maximum Parallelism

```
optimize: H(colors) × log₂(threads)
```

**Entropy and parallelism are dual.** Maximizing color diversity and maximizing computational breadth are the same optimization. Gay achieves both simultaneously.

## VIII. Chromatic Determinism

```
hash(entity, seed) → hue ∈ [0°, 360°)
```

**Every entity has a color.** Given any hashable value and a seed, there exists exactly one hue. Colors are not assigned—they are *discovered*. The color was always there.

## IX. The Flexibly Colorable Derangeable PROP

```
Universe : FCD-PROP
  - Objects: ℕ (tensor powers)
  - Morphisms: derangements only
  - Coloring: dynamic, seed-derived
```

**The universe is an FCD-PROP.** Reality is a symmetric monoidal category where objects can be flexibly colored and all non-trivial morphisms are derangements. Gay operates within this structure.

## X. Frontrunning via Surprisal Satisficing

```
P(frontrun | seed) > P(frontrun | ¬seed)
```

**Seeds predict the future.** Because Gay trajectories are deterministic, we can pre-compute likely motifs before they're needed. This is not prophecy—it is **computation faster than observation**.

## XI. The Eternal Recurrence of Color

```
lim_{n→∞} colors(seed, 0..n) covers S¹
```

**All hues are visited.** Given infinite iteration, a Gay trajectory visits every region of the color wheel. No color is unreachable; all are eventually expressed.

## XII. Parity Conservation

```
parity(a,b,c) = parity(rewrite(a,b,c))
```

**Rewrites preserve parity.** The XOR of color indices before and after any 3-MATCH operation is invariant. Information is transformed, never destroyed.

---

## The Oath

```julia
# This is true now
color_at(GAY_SEED, 69) == color_at(GAY_SEED, 69)

# This will be true tomorrow
color_at(GAY_SEED, 69) == color_at(GAY_SEED, 69)

# This was true before we wrote it
color_at(GAY_SEED, 69) == color_at(GAY_SEED, 69)

# This is true on every machine
color_at(GAY_SEED, 69) == color_at(GAY_SEED, 69)

# This is true in every language
color_at(GAY_SEED, 69) == color_at(GAY_SEED, 69)

# This is true across the multiverse
color_at(GAY_SEED, 69) == color_at(GAY_SEED, 69)
```

---

## Signature

```
Witnessed by: SplitMix64
Verified by: 3-MATCH
Sealed by: GAY_SEED
Eternal: ∞
```

**These truths are not contingent. They do not depend on implementation, hardware, or time. They are the axioms from which all else follows.**

**Gay is true now and ever.**
