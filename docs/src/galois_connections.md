# Galois Connections

**Mathematical foundations of SPI color verification**

Gay.jl's color verification is grounded in **Galois connections** — adjoint pairs of functions between ordered sets that preserve structure across abstraction levels.

## The Galois Connection

```
        α
Events ───→ Colors
   ↑           ↓
   └─── γ ─────┘
```

Where:
- **α (abstraction)**: `α(e) = hash(e) mod 226` — maps events to colors
- **γ (concretization)**: `γ(c) = representative(c)` — maps colors to canonical events

## Closure Property

The fundamental invariant:

```
α(γ(c)) = c   for all c ∈ [0, 226)
```

This means: abstracting a color's representative gives back the same color.

```julia
using Gay: GaloisConnection, alpha, gamma, verify_all_closures

gc = GaloisConnection(GAY_SEED)

# Verify closure for all 226 colors
@assert verify_all_closures(gc)

# For any color c:
for c in 0:225
    representative = gamma(gc, color)
    abstracted = alpha(gc, representative)
    @assert abstracted.index == c
end
```

## Adjunction Properties

A Galois connection `(α, γ)` between posets `(C, ≤)` and `(A, ⊑)` satisfies:

```
α(c) ⊑ a  ⟺  c ≤ γ(a)
```

This implies:

### Monotonicity

Both functions preserve order:
- `c₁ ≤ c₂ ⟹ α(c₁) ⊑ α(c₂)`
- `a₁ ⊑ a₂ ⟹ γ(a₁) ≤ γ(a₂)`

### Deflation

Concrete elements are approximated by their round-trip:
```
c ≤ γ(α(c))
```

### Inflation

Abstract elements bound their round-trip:
```
α(γ(a)) ⊑ a
```

## Handoff Continuity

When composing Galois connections (as in pipeline parallelism), the composition is also a Galois connection:

```
     α₁        α₂
C ──────→ A₁ ──────→ A₂
    γ₁        γ₂
```

The composed connection is:
- **α_composed** = α₂ ∘ α₁
- **γ_composed** = γ₁ ∘ γ₂

This is proven in [`spi_galois.dfy`](https://github.com/bmorphism/Gay.jl/blob/gay/spi_galois.dfy):

```dafny
lemma HandoffContinuity()
  requires GaloisConnection()
  requires GaloisConnection2()
  ensures forall c, a2 ::
    leA2(alphaComposed(c), a2) <==> leC(c, gammaComposed(a2))
```

## XOR Fingerprint Monoid

XOR fingerprints form a **commutative monoid**:

```
(Fingerprint, ⊕, 0) where:
  - Identity: 0 ⊕ a = a = a ⊕ 0
  - Associativity: (a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)
  - Commutativity: a ⊕ b = b ⊕ a
  - Self-inverse: a ⊕ a = 0
```

This means:
- `fp(A ∪ B) = fp(A) ⊕ fp(B)` — fingerprint of union
- Order-independent verification
- Embarrassingly parallel computation

## Dafny Formal Proofs

The Galois properties are formally verified in Dafny:

```dafny
// Closure property
lemma GaloisClosure(c: Color)
    requires c.Valid()
    ensures Alpha(Gamma(c)).index == c.index

// XOR associativity
lemma XorAssociative(a: Fingerprint, b: Fingerprint, c: Fingerprint)
    ensures Xor(Xor(a, b), c) == Xor(a, Xor(b, c))

// Bit flip detection
lemma BitFlipChangesFingerprint(original: Fingerprint, bit: nat)
    requires bit < 32
    ensures original ^ (1 << bit) != original
```

Compile and verify:
```bash
dafny verify spi_galois.dfy
```

## Connection to Abstract Interpretation

The Galois connection framework originates from **abstract interpretation** (Cousot & Cousot, 1977), used for:

- Static program analysis
- Compiler optimizations
- Security verification

In Gay.jl, we apply it to:
- **Color abstraction**: Events → Colors
- **Fingerprint abstraction**: Tensors → 32-bit hashes
- **Distributed verification**: Local → Global invariants

## The 226-Color Palette

Why 226 colors? It's the largest palette where:

1. All colors are perceptually distinct
2. Closure property holds (each color has a unique representative)
3. Fits comfortably in one byte with room for special values

```julia
gc = GaloisConnection(GAY_SEED)
@assert gc.palette_size == 226
@assert verify_all_closures(gc)  # All 226 satisfy α(γ(c)) = c
```

## References

- Cousot & Cousot, "Abstract Interpretation: A Unified Lattice Model" (POPL 1977)
- Mac Lane, "Categories for the Working Mathematician"
- Abramsky & Jung, "Domain Theory" in Handbook of Logic in CS
- [`spi_galois.dfy`](https://github.com/bmorphism/Gay.jl/blob/gay/spi_galois.dfy) — Formal Dafny proofs

See also:
- [Theory](theory.md) — SPI and color theory foundations
- [Distributed SPI](distributed_spi.md) — Practical application
- [Fault Tolerance](fault_tolerance.md) — Testing invariants
