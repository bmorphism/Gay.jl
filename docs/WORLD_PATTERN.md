# World Builder Pattern

## Overview

The World Builder pattern replaces ephemeral `demo_` functions with persistent, composable `world_` functions. This aligns Gay.jl with categorical principles where worlds are objects in a category and transformations are morphisms.

## Categorical Perspective

```
World := Object in Gay
Transformation := Morphism between Worlds
Composition := world₁ ⊗ world₂ (monoidal product)
```

## Required Interface

Every World type must implement:

```julia
# Cardinality - how many elements in the world
Base.length(w::AbstractWorld)::Int

# Monoidal composition - combine two worlds
Base.merge(w1::W, w2::W)::W where W <: AbstractWorld

# SPI-compliant fingerprint for verification
fingerprint(w::AbstractWorld)::UInt64
```

## Example Implementation

```julia
struct AncestryWorld <: AbstractWorld
    acset::AncestryACSet
    metadata::Dict{Symbol, Any}
end

Base.length(w::AncestryWorld) = nparts(w.acset, :Node)

function Base.merge(w1::AncestryWorld, w2::AncestryWorld)
    merged = copy(w1.acset)
    # Pushout-based merge preserving ancestry structure
    add_parts!(merged, :Node, nparts(w2.acset, :Node))
    # ... edge handling ...
    AncestryWorld(merged, merge(w1.metadata, w2.metadata))
end

fingerprint(w::AncestryWorld) = gay_fingerprint(w.acset)
```

## Naming Conventions

| Old Pattern | New Pattern | Rationale |
|-------------|-------------|-----------|
| `demo_ancestry()` | `world_ancestry()` | Persists, composes |
| `demo_search()` | `world_search()` | Returns searchable world |
| `show_results()` | `world_results()` | Structured return |

## Migration Guide

1. Change function name: `demo_X` → `world_X`
2. Remove `println()` calls (use logging if needed)
3. Return structured World type instead of nothing
4. Implement required interface methods
5. Add to exports in `src/Gay.jl`

## Verification

```bash
julia --project=. scripts/lint_no_demo.jl
```

Zero violations required for CI to pass.
