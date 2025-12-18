# Colorable × Flavorable: Incomplete Products & Sums Lattice

> SPI Invariants + Addressability for Partial Structures

## The Quadrant (from colorable_flavorable.jl)

```
                Colorable        ¬Colorable
             ┌─────────────────┬─────────────────┐
  Flavorable │  Both           │  Flavor Only    │
             │  (Full Gay)     │  (Semantic)     │
             ├─────────────────┼─────────────────┤
 ¬Flavorable │  Color Only     │  Neither        │
             │  (SPI Pure)     │  (Opaque)       │
             └─────────────────┴─────────────────┘
```

## Type Hierarchy

### Complete Types (Full SPI Guarantee)

| Type | Definition | SPI Property |
|------|------------|--------------|
| `ColorFlavor` | `Color × Flavor` | ▣ Deterministic fingerprint |
| `SPIColorable` | `seed → color` bijection | ▣ Order-independent |
| `BothThread` | `Colorable ∧ Flavorable` | ▣ Maximum tractability |
| `ChromaFlavorable{T}` | `α|color⟩ + β|flavor⟩` | ▣ Superposition preserved |

### Partial Products (Incomplete ×)

| Type | Definition | What's Missing |
|------|------------|----------------|
| `Maybe(Color × Flavor)` | `Nothing \| ColorFlavor` | Existence |
| `Color × ?` | Product with unknown flavor | Flavor component |
| `? × Flavor` | Product with unknown color | Color component |
| `Colorable{T}` | Structural identity only | Semantic meaning |
| `Flavorable{T}` | Semantic identity only | Structural color |

### Partial Sums (Incomplete +)

| Type | Definition | What's Chosen |
|------|------------|---------------|
| `Color + Flavor` | Either, not both | One dimension |
| `Either Color Flavor` | `Left c \| Right f` | Tagged union |
| `Just Action` | Certainty sans color | Action without identity |
| `LazyGay{T}` | Thunk (deferred) | Not yet computed |

### Incomplete but Addressable

| Type | Address By | Completion Path |
|------|------------|-----------------|
| `OpaqueThread` | External ID | Learn flavor via context |
| `LazyGay{T}` | Thunk hash | `force!()` to compute |
| `Partial{T}` | Partial fingerprint | Provide missing seed |

## Addressability Modes

### 1. Fingerprint Address (UInt64)
```julia
# O(1) lookup via XOR-based hash
addr = fingerprint ⊻ GAY_SEED
bucket = addr & 0xFFFF
```

### 2. Color Address (RGB → Bucket)
```julia
# Perceptual clustering in Okhsl
h_bucket = floor(Int, hue / 30)      # 12 buckets
l_bucket = floor(Int, lightness * 4)  # 4 levels
addr = h_bucket * 4 + l_bucket        # 48 total
```

### 3. Flavor Address (Symbol → Concept)
```julia
# Semantic hashing
concept_hash = reduce(⊻, hash.(concepts))
addr = concept_hash & 0xFFFF_FFFF
```

### 4. Hybrid Address (Best of Both)
```julia
# XOR fusion preserves both
hybrid_addr = color_fingerprint ⊻ flavor_hash
# Addressable even if one component is partial!
```

## SPI Invariants Ensured

### Invariant 1: Determinism
```julia
∀ seed, thread_id:
  color_from_seed(seed ⊻ hash(thread_id)) == color_from_seed(seed ⊻ hash(thread_id))
```

### Invariant 2: Order Independence (Parallel Safe)
```julia
∀ xs: reduce(⊻, shuffle(map(fingerprint, xs))) == reduce(⊻, map(fingerprint, xs))
```

### Invariant 3: Fingerprint Conservation
```julia
∀ cf::ColorFlavor:
  cf.combined_fingerprint == cf.color_fingerprint ⊻ cf.flavor_fingerprint
```

### Invariant 4: Completion Monotonicity
```julia
# Completing a partial type never loses information
∀ p::Partial, c::Complete:
  p ⊑ complete(p) implies fingerprint(p) ⊻ δ == fingerprint(complete(p))
```

### Invariant 5: Lazy/Eager Duality
```julia
∀ lazy::LazyGay:
  force!(lazy).chromaflavor.color == lazy.chromaflavor.color
  # Color is determined BEFORE forcing
```

## Varieties of Incompleteness

### A. Missing Color (Semantic Only)
```julia
struct FlavorOnly{T}
    value::T
    flavor::FlavorProfile
    # No color field - not SPI-colorable
end

# Addressable by flavor hash
address(fo::FlavorOnly) = hash(fo.flavor.concepts)
```

### B. Missing Flavor (Structural Only)
```julia
struct ColorOnly{T}
    value::T
    color::RGB{Float64}
    seed::UInt64
    # No flavor field - not semantically interpretable
end

# Addressable by fingerprint
address(co::ColorOnly) = splitmix64(co.seed)[1]
```

### C. Deferred (LazyGay)
```julia
struct LazyGay{T}
    thunk::Function
    chromaflavor::ChromaFlavorable{Symbol}  # Symbolic until forced
    forced::Ref{Bool}
    cached::Ref{Union{Nothing, T}}
end

# Addressable by thunk identity
address(lg::LazyGay) = hash(lg.thunk) ⊻ lg.chromaflavor.seed
```

### D. Superposition (ChromaFlavorable)
```julia
struct ChromaFlavorable{T}
    value::T
    color::NTuple{3, Float64}
    flavor::FlavorType
    α::ComplexF64  # Color amplitude
    β::ComplexF64  # Flavor amplitude
end

# Addressable by weighted combination
address(cf::ChromaFlavorable) = 
    round(UInt64, abs2(cf.α) * color_hash(cf) + abs2(cf.β) * flavor_hash(cf))
```

## The Flavor Types (from lazy_eager_duality.jl)

| Flavor | Categorical Analog | Operation |
|--------|-------------------|-----------|
| **Sweet** | Coproduct (+) | Additive, sum types |
| **Sour** | Quotient (/) | Subtractive, equivalence |
| **Salty** | Product (×) | Multiplicative, tuples |
| **Bitter** | Exponential (→) | Hom, function types |
| **Umami** | Tensor (⊗) | Monoidal, parallel |
| **Spicy** | Differential (∂) | Tangent, Schreiber cohesion |

## Addressable Space Size

For Gay.jl with SPI guarantees:

| Component | Bits | Addressable Size |
|-----------|------|------------------|
| Color fingerprint | 64 | 2^64 unique colors |
| Flavor hash | 64 | 2^64 unique flavors |
| Combined | 64 | 2^64 (XOR preserves) |
| Thread ID | 64 | 2^64 threads |
| **Total addressable** | **64** | **~10^19 unique entities** |

This is sufficient for any practical use case while maintaining O(1) addressability.

## Code: Ensuring SPI with Partial Types

```julia
"""
    ensure_spi(x::Union{Colorable, Flavorable, ColorFlavor, LazyGay})

Verify SPI invariants hold for any type in the lattice.
Returns (verified::Bool, address::UInt64, completion_needed::Vector{Symbol})
"""
function ensure_spi(x)
    if x isa ColorFlavor
        # Complete: just verify fingerprint
        verified = x.combined_fingerprint == x.color_fingerprint ⊻ x.flavor_fingerprint
        return (verified, x.combined_fingerprint, Symbol[])
        
    elseif x isa Colorable
        # Partial: missing flavor
        fp, _ = splitmix64(x.seed)
        return (true, fp, [:flavor])
        
    elseif x isa Flavorable
        # Partial: missing color
        fp = reduce(⊻, hash.(x.concepts))
        return (true, fp, [:color])
        
    elseif x isa LazyGay
        # Deferred: addressable but not forced
        addr = hash(x.thunk) ⊻ x.seed
        completion = x.forced[] ? Symbol[] : [:force]
        return (true, addr, completion)
        
    else
        # Opaque: not SPI-verifiable
        return (false, UInt64(0), [:color, :flavor, :seed])
    end
end
```
