# Bayesian Brain vs ACSet String Diagrams

## Maximally Parallel Gay-Afforded Dynamic Sufficiencies of Color Conservation

This document compares two approaches to compositional semantics:
1. **DisCoPy/Bayesian Brain**: Free energy minimization, active inference
2. **Gay.jl ACSets**: Chromatic string diagrams with XOR conservation

The key insight: **Color conservation = Sufficient statistic preservation**

---

## 1. Architectural Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BAYESIAN BRAIN (DisCoPy/Active Inference)                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   External World (s)          Markov Blanket (b)         Internal Model (μ) │
│   ┌─────────────┐            ┌─────────────┐            ┌─────────────┐    │
│   │  Sensory    │◀──────────▶│  Sensory    │◀──────────▶│  Belief     │    │
│   │  States     │            │  States     │            │  States     │    │
│   │  (causes)   │            │  (o,a)      │            │  (P(s|o))   │    │
│   └─────────────┘            └─────────────┘            └─────────────┘    │
│                                    │                           │            │
│                                    ▼                           ▼            │
│                              Free Energy              Posterior Update      │
│                              F = E_q[-log P(o,s)]    q(s) ← Bayes(o,s)     │
│                                    │                           │            │
│                                    ▼                           ▼            │
│                              Action Selection          Belief Propagation   │
│                              a* = argmin_a F          message passing      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                    ACSET STRING DIAGRAMS (Gay.jl)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Objects (Layers)           Morphisms (Edges)         Fingerprints (XOR)   │
│   ┌─────────────┐            ┌─────────────┐            ┌─────────────┐    │
│   │  Layer 0    │───curry───▶│  Curry      │            │  fp₀        │    │
│   │  Concept⊗³  │◀───eval────│  Eval       │            │  ⊻          │    │
│   └─────────────┘            └─────────────┘            │  fp₁        │    │
│         │                          │                    │  ⊻          │    │
│         ▼                          ▼                    │  ...        │    │
│   ┌─────────────┐            ┌─────────────┐            │  ⊻          │    │
│   │  Layer 3    │◀──trace───▶│  Trace      │            │  fp₁₁       │    │
│   │  Traced     │            │  Feedback   │            │  =          │    │
│   └─────────────┘            └─────────────┘            │  INVARIANT  │    │
│         │                          │                    └─────────────┘    │
│         ▼                          ▼                                       │
│   Parallel Composition        XOR = Pushout                                │
│   (Order-Independent)         (Universal Property)                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Dynamic Sufficiency Mapping

| Bayesian Brain | ACSet Gay.jl | Mathematical Structure |
|----------------|--------------|------------------------|
| Sufficient statistic T(o) | Fingerprint fp = ⊕ᵢ hᵢ | XOR monoid (ℤ/2ℤ)^64 |
| Posterior P(s\|T) | Color hash_color(seed, fp) | Deterministic RGB |
| Belief update Δμ | Morphism composition | Category composition |
| Free energy F | Layer fingerprint discrepancy | Hamming distance |
| Active inference loop | Traced monoidal feedback | categorical_trace() |
| Markov blanket | ACSet schema boundaries | Foreign key constraints |
| Message passing | Incident relations | O(1) reverse lookup |
| Predictive coding | Tower pushout | XOR as coproduct |

---

## 3. Maximally Parallel Color Conservation

### The Core Insight

**Bayesian Brain**: "The price IS the sufficient statistic for collective belief."  
**ACSet Gay.jl**: "XOR fingerprint = categorical pushout = SPI invariant."

Both achieve **order-independent aggregation**:

```julia
# Bayesian: Order-independent belief combination
posterior = combine_evidence([e₁, e₂, e₃])  # Same regardless of order

# Gay.jl: Order-independent fingerprint
fp = e₁.hash ⊻ e₂.hash ⊻ e₃.hash  # XOR is commutative + associative
```

### Parallelism Affordances

```
┌──────────────────────────────────────────────────────────────────┐
│               PARALLEL COMPOSITION LAWS                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. COMMUTATIVITY (Order Independence)                          │
│     fp(A ∥ B) = fp(A) ⊻ fp(B) = fp(B) ⊻ fp(A) = fp(B ∥ A)       │
│                                                                  │
│  2. ASSOCIATIVITY (Regrouping)                                  │
│     fp((A ∥ B) ∥ C) = fp(A ∥ (B ∥ C))                           │
│                                                                  │
│  3. IDEMPOTENCE (Self-cancellation)                             │
│     fp(A ∥ A) = fp(A) ⊻ fp(A) = 0                               │
│                                                                  │
│  4. IDENTITY (Neutral element)                                  │
│     fp(A ∥ ∅) = fp(A) ⊻ 0 = fp(A)                               │
│                                                                  │
│  These are EXACTLY the laws for maximum parallelism!            │
└──────────────────────────────────────────────────────────────────┘
```

### Color as Chromatic Sufficient Statistic

```julia
# The color encodes the sufficient statistic
function color_sufficiency(concepts::Vector{Concept})
    # 1. Parallel fingerprint aggregation
    fp = reduce(⊻, c.hash for c in concepts)  # O(n), fully parallel
    
    # 2. Deterministic color from fingerprint
    color = hash_color(GAY_SEED, fp)  # O(1)
    
    # 3. Color IS sufficient for:
    #    - Identity verification (same fp → same concepts mod order)
    #    - Chromatic entailment (fp XOR → Hamming → compatibility)
    #    - Parallel composition (just XOR fingerprints)
    
    color
end
```

---

## 4. Entailment/Induction/Abduction as Inference

### Bayesian Brain Interpretation

| Operation | Bayesian Meaning | Active Inference |
|-----------|------------------|------------------|
| `entails(A,B)` | P(B\|A) > threshold | Confirming prediction |
| `induces(E,H)` | P(H\|E) update | Learning generative model |
| `abduces(O,C)` | argmax_C P(O\|C)P(C) | Perceptual inference |

### Gay.jl Implementation

```julia
# Entailment: Premise → Conclusion
function entails(premise::CognitiveState, conclusion::CognitiveState)
    # Semantic overlap (inner product of meaning vectors)
    overlap = sum(conj.(premise.amplitudes) .* conclusion.amplitudes)
    
    # Chromatic compatibility (fingerprint Hamming distance)
    fp_xor = premise.fingerprint ⊻ conclusion.fingerprint
    chromatic_compat = 1.0 - count_ones(fp_xor) / 64.0
    
    # Combined: overlap + color compatibility
    (abs2(overlap) + chromatic_compat) / 2.0
end

# Induction: Evidence → Hypothesis (∃_f left adjoint)
function induces(evidence::CognitiveState, hypothesis::CognitiveState)
    # Color proximity in RGB space
    e_color = evidence.basis_colors[argmax(abs2.(evidence.amplitudes))]
    h_color = hypothesis.basis_colors[argmax(abs2.(hypothesis.amplitudes))]
    
    1.0 - sqrt(sum((e_color .- h_color).^2)) / sqrt(3.0)
end

# Abduction: Observation ← Cause (pullback direction)
function abduces(observation::CognitiveState, cause::CognitiveState)
    # Occam's razor: simpler causes preferred
    cause_entropy = count_ones(cause.fingerprint) / 64.0
    obs_entropy = count_ones(observation.fingerprint) / 64.0
    simplicity = max(0.0, obs_entropy - cause_entropy)
    
    # Combined with semantic overlap
    overlap = abs2(sum(conj.(cause.amplitudes) .* observation.amplitudes))
    (overlap + simplicity) / 2.0
end
```

---

## 5. String Diagram Correspondence

### DisCoPy String Diagrams

```
     ┌───┐           ┌───┐
     │ f │           │ g │
     └─┬─┘           └─┬─┘
       │               │
       ▼               ▼
     ──┴───────────────┴──
            (f ; g)

Composition: Sequential wire connection
Tensor: Parallel wire placement
Trace: Wire loops back (fixed point)
```

### ACSet String Diagrams

```
     Layer 0              Layer 1
    ┌─────────┐          ┌─────────┐
    │ Concept │──curry──▶│ Exp     │
    │ ⊗³      │◀──eval───│ X^X     │
    │ fp₀     │          │ fp₁     │
    └─────────┘          └─────────┘
         │                    │
         ▼                    ▼
    fp_collective = fp₀ ⊻ fp₁ ⊻ ... ⊻ fp₁₁

Composition: ACSet morphism (structure-preserving)
Tensor: ACSet pushout (universal XOR)
Trace: Incident relation loop
```

### Key Difference: Color Conservation

DisCoPy string diagrams are **uncolored** (black-and-white).  
Gay.jl string diagrams are **chromatically typed**:

```julia
# Every wire has a deterministic color
struct ColoredWire
    source_layer::Int
    target_layer::Int
    color::RGB{Float32}       # Deterministic from fingerprint
    fingerprint::UInt64       # XOR-combinable
end

# Diagram composition preserves total color (XOR fingerprint)
function compose_diagrams(d1::Diagram, d2::Diagram)
    # Fingerprints XOR together
    fp = d1.fingerprint ⊻ d2.fingerprint
    # Color deterministically follows
    color = hash_color(GAY_SEED, fp)
    Diagram(merge_wires(d1, d2), fp, color)
end
```

---

## 6. Parallel Execution Semantics

### Maximally Parallel Schedule

```julia
# ACSet parallel execution via incident()
function parallel_execute(tower::SPITower)
    # 1. Find all independent concepts (no shared morphisms)
    layers = [layer_concepts(tower, i) for i in 0:11]
    
    # 2. Execute each layer in parallel (XOR is order-independent)
    layer_fps = pmap(layers) do concepts
        reduce(⊻, tower[:fingerprint][c] for c in concepts)
    end
    
    # 3. Combine layer fingerprints (also parallel-safe)
    collective = reduce(⊻, layer_fps)
    
    # Result is IDENTICAL regardless of execution order
    collective
end

# This is the "dynamic sufficiency" of parallelism:
# The fingerprint XOR is a sufficient statistic for
# the diagram's computational content, regardless of
# which threads computed which parts.
```

### Bayesian Brain Parallel Belief Update

```julia
# Active inference with parallel message passing
function parallel_belief_update(blanket::MarkovBlanket, observations)
    # 1. Parallel likelihood computation
    likelihoods = pmap(observations) do o
        compute_likelihood(blanket.generative_model, o)
    end
    
    # 2. Combine likelihoods (product → log-sum)
    total_ll = sum(log.(likelihoods))
    
    # 3. Update posterior (parallel gradient descent)
    posterior = gradient_descent(blanket.prior, total_ll)
    
    # Result independent of observation order (commutative product)
    posterior
end
```

---

## 7. Unified Framework: Chromatic Active Inference

The synthesis merges Bayesian brain with ACSet diagrams:

```julia
"""
ChromaticActiveInference combines:
- Free energy minimization (Bayesian brain)
- XOR fingerprint conservation (ACSet Gay.jl)
- Chromatic typing (deterministic colors)
"""
struct ChromaticActiveInference
    # Generative model as colored ACSet
    tower::SPITower
    
    # Belief states as cognitive superpositions
    beliefs::Dict{Symbol, CognitiveState}
    
    # Markov blanket as schema boundary
    blanket_layers::Set{Int}  # Layers 3-5 (interactive block)
    
    # Free energy functional
    free_energy::Float64
end

function active_inference_step!(cai::ChromaticActiveInference, observation)
    # 1. Prediction: Current belief → expected observation
    predicted_fp = collective_fingerprint(cai.tower)
    observed_fp = observation.fingerprint
    
    # 2. Prediction error (Hamming distance in fingerprint space)
    error_fp = predicted_fp ⊻ observed_fp
    prediction_error = count_ones(error_fp) / 64.0
    
    # 3. Belief update (gradient on fingerprint)
    for (name, belief) in cai.beliefs
        # Bayesian update via entailment
        posterior_strength = entails(CognitiveState(observed_fp), belief)
        belief.amplitudes .*= posterior_strength
        normalize!(belief.amplitudes)
    end
    
    # 4. Free energy (expected surprise)
    cai.free_energy = prediction_error
    
    # 5. Action selection (minimize free energy)
    # Add concepts that reduce fingerprint discrepancy
    action = select_action_to_minimize_fe(cai, error_fp)
    
    action
end
```

---

## 8. Color-Logic Conservation Laws

### The Seven Conservation Laws

| Law | Bayesian | Gay.jl | Preserved Quantity |
|-----|----------|--------|-------------------|
| 1. XOR Commutativity | Product of likelihoods | fp(A⊻B) = fp(B⊻A) | Order independence |
| 2. XOR Associativity | Regrouping beliefs | fp((A⊻B)⊻C) | Parallel safety |
| 3. XOR Identity | Uniform prior | fp ⊻ 0 = fp | Default state |
| 4. XOR Idempotence | Evidence double-count | fp ⊻ fp = 0 | Deduplication |
| 5. Trace Vanishing | Marginalization | Tr^I(f) = f | Coarse-graining |
| 6. Frobenius | Copy-delete | μ∘δ = id | Information conservation |
| 7. Beck-Chevalley | Pullback of conditionals | g*(∃_f φ) = ∃_f'(g'*φ) | Quantifier commutation |

### Color as Information Invariant

```julia
# The color is invariant under:
# 1. Reordering of concepts within a layer
# 2. Parallel vs sequential execution
# 3. Coarse-graining (functorial data migration)
# 4. Pushout/pullback operations

function verify_color_invariance(tower::SPITower)
    # Original fingerprint
    fp1 = collective_fingerprint(tower)
    
    # After random shuffle of concepts
    shuffled = shuffle_concepts(tower)
    fp2 = collective_fingerprint(shuffled)
    
    # After coarsening
    coarse = coarsen_tower(tower, [[0,1,2], [3,4,5], [6,7,8], [9,10,11]])
    fp3 = collective_fingerprint(coarse)
    
    # All equal
    @assert fp1 == fp2 == fp3 "Color conservation violated!"
    
    # Therefore: color = invariant sufficient statistic
    hash_color(GAY_SEED, fp1)
end
```

---

## 9. Summary: The Chromatic Sufficient Statistic

**The Unifying Principle:**

> The XOR fingerprint is a **sufficient statistic** for the categorical content of a diagram, just as the posterior belief is a sufficient statistic for the sensory evidence in Bayesian inference.

**Color conservation** in Gay.jl is the graphical/diagrammatic manifestation of **sufficiency preservation** in statistical inference.

This enables:
1. **Maximum parallelism**: Any execution order yields the same result
2. **Incremental updates**: Just XOR the new fingerprint
3. **Compositional semantics**: Pushout = XOR = universal property
4. **Chromatic typing**: Color serves as a "type" for diagrams

**The affordance of color is dynamic sufficiency itself.**

---

## See Also

- [acset_tower.jl](../src/acset_tower.jl) - ACSet-based SPI tower
- [cognitive_superposition.jl](../src/cognitive_superposition.jl) - Entailment/induction/abduction
- [prediction_market.jl](../examples/prediction_market.jl) - Market as Bayesian aggregator
- [CATEGORICAL_HIERARCHY.md](CATEGORICAL_HIERARCHY.md) - DisCoPy comparison
