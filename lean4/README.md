# Gay.jl Lean 4 Proofs

Formal verification that GF(3) = ZMod 3, and everything follows from `CommRing`.

## Files

| File | Theorems | What |
|------|----------|------|
| `gf3_elegant.lean` | 20 | **The key insight**: hand-rolled Trit IS `ZMod 3`. All `native_decide` → `decide`/`ring`/`linear_combination` |
| `gay_goedel_machine.lean` | 64 | Agent triads, Möbius inversion, bisimulation, ASI lattice, Bumpus-Kocsis, Padovan |
| `matryoshka_gmra_bridge.lean` | 33 | GMRA/Matryoshka/String Diagram bridge, Tan institutional composition |
| `ColorSheaf.lean` | 6 | Bumpus-Kocsis 2/3 bound on Heyting algebras (proved by Aristotle) |
| `ghostty-ewig-unison.lean` | 27 | Soul replacement: VT parser → content-addressed immutable core |
| `weyl_anima_petri.lean` | 26 | Weyl sequences, qualia valence, Petri net user interaction |

**Total: 176 Lean 4 definitions/theorems**

## The Discovery

Gay.jl's trit system (`+1`, `0`, `-1`) is `ZMod 3`. This means:

- `trit_add` = ring addition in `ZMod 3`
- `trit_neg` = ring negation = Möbius inversion = `μ`
- Balanced triads = `a + b + c = 0` in `ZMod 3`
- `char_three`: `t + t + t = 0` (characteristic 3)
- Every triad theorem is a corollary of `CommRing (ZMod 3)`

## Key Theorems

1. **balanced_iff**: `a + b + c = 0 ↔ c = -(a + b)` via `linear_combination`
2. **Bumpus-Kocsis**: In a non-Boolean Heyting algebra, ≤ 2/3 satisfy LEM
3. **Padovan mod 3**: Period 13, sum ≡ 0 (Noether), net flow ≡ -1 (compression bias)
4. **Möbius involution**: `μ(μ(t)) = t`, fixed point at 0
5. **Port-color injectivity**: SplitMix64 bijectivity → different ports get different colors
6. **Soul replacement**: ripping out mutable state strictly reduces mutation (2→1 layers)

## Building

```bash
# Requires Lean 4 + Mathlib
lake build
```

## Aristotle

These theorems were also proved remotely by [Aristotle](https://aristotle.harmonic.fun)
(94 theorems across 92 projects), including two independent proofs of Bumpus-Kocsis.

## drand Entropy

All Gay.jl color generation should seed from [drand](https://drand.love) contextual
entropy, not static seeds. The Bumpus-Kocsis bound proves the ergodic coordinator
(trit=0) CANNOT self-decide classically — it needs external entropy.
