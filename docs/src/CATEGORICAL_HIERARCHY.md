# Categorical Hierarchy: Gay.jl vs DisCoPy

This document compares the categorical structures in Gay.jl with DisCoPy's implementation, focusing on **colorable, derangeable, tropicalizable** structures.

## Hierarchy Overview

```
                    ┌─────────────────────────────────────────────────┐
                    │           COGNITIVE SUPERPOSITION               │
                    │  (Entities Entailing Inducing in Superposition) │
                    └─────────────────────────────────────────────────┘
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    ▼                                           ▼
         ┌──────────────────┐                       ┌──────────────────┐
         │   HYPERGRAPH     │                       │   TROPICALIZED   │
         │   (Frobenius)    │                       │   (min-plus/     │
         │   Spiders n→m    │                       │    max-plus)     │
         └──────────────────┘                       └──────────────────┘
                    │                                           │
         ┌──────────┴──────────┐                               │
         ▼                     ▼                               ▼
┌──────────────────┐  ┌──────────────────┐           ┌──────────────────┐
│    COMPACT       │  │    TRACED        │           │   TROPICAL       │
│    (Cup/Cap)     │  │   (Feedback)     │           │   SEMIRING       │
│    Rigid duals   │  │   Tr^U(φ)        │           │   (min,+),(max,+)│
└──────────────────┘  └──────────────────┘           └──────────────────┘
         │                     │                               │
         └──────────┬──────────┘                               │
                    ▼                                          │
         ┌──────────────────┐                                  │
         │    BRAIDED       │◀─────────────────────────────────┘
         │  (Wire Crossing) │   Tropical braid = derangement
         │   σ_{A,B}        │
         └──────────────────┘
                    │
                    ▼
         ┌──────────────────┐
         │   SYMMETRIC      │
         │   MONOIDAL       │
         │   (A ⊗ B ≅ B ⊗ A)│
         └──────────────────┘
                    │
                    ▼
         ┌──────────────────┐
         │    MONOIDAL      │
         │   (A ⊗ B, I)     │
         └──────────────────┘
                    │
                    ▼
         ┌──────────────────┐
         │    CATEGORY      │
         │   (Objects,      │
         │    Morphisms)    │
         └──────────────────┘
```

## Detailed Comparison

### Layer 0: Base Category

| Aspect | DisCoPy | Gay.jl |
|--------|---------|--------|
| Module | `discopy.cat` | `ConceptTensor` |
| Objects | `Ob`, `Ty` | `Concept`, `ChromaticType` |
| Morphisms | `Arrow`, `Box` | `ConceptMorphism`, `CognitiveMorphism` |
| Identity | `Id(x)` | `identity_morphism(seed)` |
| Composition | `f >> g` | `compose(f, g)` |
| **Coloring** | ❌ | ✓ `hash_color(seed, fp)` |

### Layer 1: Monoidal Category

| Aspect | DisCoPy | Gay.jl |
|--------|---------|--------|
| Module | `discopy.monoidal` | `TracedTensor` |
| Tensor | `f @ g` | `tensor_product(φ, ψ)` |
| Unit | `Ty()` | `monoidal_unit(seed)` |
| Interchange | `(f @ g) >> (h @ k) = (f >> h) @ (g >> k)` | ✓ via XOR |
| **Colorable** | ❌ | ✓ `φ.color`, `ψ.color` |
| **Derangeable** | ❌ | ✓ `GayInterleaver` |

### Layer 2: Braided Symmetric Monoidal

| Aspect | DisCoPy | Gay.jl |
|--------|---------|--------|
| Module | `discopy.braided` | `CognitiveSuperposition` |
| Braiding | `Braid(left, right)` | `braid(bs::BraidedSuperposition)` |
| Hexagons | `left_hexagon`, `right_hexagon` | Verified in `verify_cognitive_laws` |
| Naturality | ✓ | ✓ via fingerprint |
| **Tropicalizable** | ❌ | ✓ Braid phase = tropical weight |

### Layer 3: Compact Closed

| Aspect | DisCoPy | Gay.jl |
|--------|---------|--------|
| Module | `discopy.compact` | `TracedTensor` |
| Cup | `Cup(left, right)` | via `categorical_trace` |
| Cap | `Cap(left, right)` | via `feedback_loop` |
| Yanking | `Cup >> Cap = Id` | ✓ `verify_traced_laws` |
| **Chromatic** | ❌ | ✓ Cups/Caps have colors |

### Layer 4: Traced Monoidal

| Aspect | DisCoPy | Gay.jl |
|--------|---------|--------|
| Module | `discopy.traced` | `TracedTensor` |
| Trace | `Trace(f, n)` | `categorical_trace(tm, lat)` |
| Vanishing | `Tr^I(f) = f` | ✓ |
| Superposing | `Tr(g ⊗ f) = g ⊗ Tr(f)` | ✓ |
| Dinaturality | ✓ | ✓ |
| **Feedback Color** | ❌ | ✓ `feedback_transform` |

### Layer 5: Hypergraph (Frobenius)

| Aspect | DisCoPy | Gay.jl |
|--------|---------|--------|
| Module | `discopy.frobenius` | `CognitiveSuperposition` |
| Spider | `Spider(n_in, n_out, typ)` | `cognitive_spider(n_in, n_out, typ)` |
| Frobenius | `split @ x >> x @ merge` | ✓ Verified |
| Speciality | `split >> merge = id` | ✓ |
| **Spider Color** | `Spider.color = "black"` | ✓ SPI-derived per spider |

### Layer 6: Markov Category

| Aspect | DisCoPy | Gay.jl |
|--------|---------|--------|
| Module | `discopy.markov` | `GayMC` |
| Copy | `Copy(x)` | Implicit in spider |
| Discard | `Discard(x)` | `gay_measure!` |
| Causality | Arrows copy-discard | ✓ via checkpointing |
| **Stochastic Color** | ❌ | ✓ `color_sweep` |

## Tropicalization

Gay.jl uniquely supports **tropicalization** of categorical structures:

```julia
# Tropical semirings available
TropicalMinPlus   # (min, +) - Shortest path
TropicalMaxPlus   # (max, +) - Longest path  
TropicalMinMax    # (min, max) - Capacity
TropicalMaxMin    # (max, min) - Bottleneck
```

### Tropical Braiding = Derangement

In tropical geometry, braiding becomes a **derangement** (permutation with no fixed points):

```julia
# Standard braiding σ_{A,B}: A ⊗ B → B ⊗ A
# Tropicalized: min(a + b, b + a) with derangement constraint

DERANGEMENTS_3 = (
    [2, 3, 1],  # R→G, G→B, B→R (cyclic left)
    [3, 1, 2],  # R→B, G→R, B→G (cyclic right)
)
```

### Tropical Trace = Fixpoint

The categorical trace becomes a **tropical fixpoint**:

```
Tr^U(φ)(a) = min_u { φ(a, u)_B + φ(a, u)_U }  (min-plus trace)
```

## Cognitive Superposition Unique Features

Gay.jl adds cognitive semantics not present in DisCoPy:

### 1. Entailment (⊨)
```julia
# Premise ⊨ Conclusion
entails(premise, conclusion) -> Float64
```

### 2. Induction (∃_f)
```julia
# Evidence → Hypothesis (left adjoint)
induces(evidence, hypothesis) -> Float64
```

### 3. Abduction (Pullback)
```julia
# Observation ← Cause (inverse inference)
abduces(observation, cause) -> Float64
```

### Hyperdoctrine Integration

The adjunction triple:
```
∃_f ⊣ f* ⊣ ∀_f
(existential ⊣ substitution ⊣ universal)
```

Is verified via Beck-Chevalley:
```julia
verify_beck_chevalley(H, :f, :g, :X, :φ)
# g*(∃_f(φ)) = ∃_f'(g'*(φ))
```

## Color-Logic System

Each logic system maps to a canonical color:

| Logic | Color | RGB | Crystal |
|-------|-------|-----|---------|
| Intuitionistic | Green | (0,255,0) | Cubic |
| Paraconsistent | Red | (255,0,0) | Triclinic |
| Linear | Blue | (0,0,255) | Hexagonal |
| Modal S4 | Orange | (255,165,0) | Monoclinic |
| HoTT | Purple | (128,0,128) | Tetragonal |
| Classical | White | (255,255,255) | Orthorhombic |
| Metatheory | Magenta | (255,0,255) | Meta |

## Usage Examples

### Creating a Cognitive Superposition

```julia
using Gay

# Word meanings as cognitive states
cat = CognitiveState(4, :N; seed=GAY_SEED)
dog = CognitiveState(4, :N; seed=GAY_SEED ⊻ 0xDOG)

# Polysemy: "pet" is superposition of cat and dog
pet = superpose([cat, dog], [0.6+0im, 0.4+0im])

# Check entailment
animal = CognitiveState(4, :N; seed=GAY_SEED ⊻ 0xANIMAL)
entails(cat, animal)  # → 0.8 (cat entails animal)
entails(animal, cat)  # → 0.3 (animal weakly entails cat)
```

### Braided Tensor Product

```julia
# Tensor product with braiding phase
cat_dog = cognitive_tensor(cat, dog)
dog_cat = braid(cat_dog)  # σ_{cat,dog}

# Verify involutivity
braid(dog_cat).fingerprint == cat_dog.fingerprint
```

### Hypergraph Spiders

```julia
# Merge spider (2 → 1)
merge = cognitive_spider(2, 1, :N)
merged = merge([cat, dog])

# Split spider (1 → 2)  
split = cognitive_spider(1, 2, :N)
copies = split(merged)

# Frobenius: merge >> split ≈ id ⊗ id (up to scalar)
```

## See Also

- [TracedTensor](traced_tensor.md) - Traced monoidal structure
- [Hyperdoctrine](hyperdoctrine.md) - Beck-Chevalley for predicates
- [TropicalSemirings](tropical_semirings.md) - min/max-plus algebras
- [DisCoPy Documentation](https://docs.discopy.org/)
