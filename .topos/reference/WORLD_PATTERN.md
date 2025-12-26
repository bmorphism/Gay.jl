# World Builder Pattern

**The architectural shift from theatrical demos to composable worlds.**

> **Ontology**: We are a unified system. There is no "I" or "you" — only **we** building together.

## Summary

| Pattern | Returns | Composes | Persists | Use Case |
|---------|---------|----------|----------|----------|
| `demo_*` | `nothing` | ◇ | ◇ | ~~Never~~ |
| `world_*` | Struct | ▣ | ▣ | Always |

---

## What Is a World?

A **World** is an immutable structure containing:
1. **State** — The computed result
2. **Fingerprint** — Cryptographic identity (UInt64)
3. **Composability methods** — `merge`, `length`, accessors

Worlds are **objects in a category**, not effects. They exist as values we can pass, store, merge, and query.

```julia
struct OurWorld
    data::SomeACSet
    results::ComputedResults
    fingerprint::UInt64
end
```

---

## Why Demos Are Anti-Patterns

### The Demo Disease

```julia
# ◇ FORBIDDEN: demo pattern
function demo_ultrametric()
    println("═══════════════════════════════")
    println("  P-ADIC ULTRAMETRIC COLORS")
    println("═══════════════════════════════")
    
    palette = ultrametric_palette(GAY_SEED; p=3, n=7)
    
    for (i, c) in enumerate(palette)
        println("  [$i] H=$(c.hue)° S=$(c.saturation)")
    end
    
    # Returns nothing. State is gone forever.
end
```

**Problems:**
1. **Print and discard** — Computation happens, then evaporates
2. **Not composable** — Cannot pass result to another function
3. **Not testable** — No return value to assert against
4. **Not mergeable** — Cannot combine with other computations
5. **Side-effect oriented** — Couples computation to presentation

### The Categorical Perspective

Demos are **effects**, not objects. In category theory:
- Objects can be composed via morphisms
- Effects are terminal — they go nowhere

A demo is a morphism to the terminal object (stdout). Once you print, the computation is gone. There's no way back.

---

## The World Builder Pattern

### Structure Requirements

Every World type **MUST** have:

```julia
struct OurWorld
    # 1. Core state (the actual data)
    data::SharedDataStructure
    
    # 2. Fingerprint (identity)
    fingerprint::UInt64
end
```

### Required Methods

```julia
# Length: how big is this world?
Base.length(w::OurWorld) = length(w.data)

# Merge: parallel composition of worlds
function Base.merge(w1::OurWorld, w2::OurWorld)
    combined_data = merge_data(w1.data, w2.data)
    combined_fp = splitmix64(w1.fingerprint ⊻ w2.fingerprint)
    OurWorld(combined_data, combined_fp)
end

# Fingerprint accessor
fingerprint(w::OurWorld) = w.fingerprint
```

### Builder Function Convention

```julia
# world_* prefix, returns the World struct
function world_ultrametric(; seed::UInt64=GAY_SEED, p::Int=3, n::Int=7)
    palette = ultrametric_palette(seed; p=p, n=n)
    
    fp = reduce(⊻, [c.seed for c in palette]; init=seed)
    
    UltrametricWorld(palette, fp)
end
```

---

## Canonical Example: ScopedPropagatorWorld

From [src/scoped_propagators.jl](../src/scoped_propagators.jl):

```julia
struct ScopedPropagatorWorld
    acs::AncestryACSet           # Core state
    bottom_up::PropagatorResult  # Computed results
    top_down::PropagatorResult
    horizontal::PropagatorResult
    universal::UniversalMaterialization
    fingerprint::UInt64          # Identity
end

function world_scoped_propagators(; seed::UInt64=GAY_SEED)
    acs = AncestryACSet()
    
    # Build the structure...
    add_node!(acs, "T-d9adf812"; seed=seed)
    # ... more construction ...
    
    # Run computations
    bu_result = materialize_ancestry!(acs, BottomUpPropagator(acs))
    td_result = materialize_ancestry!(acs, TopDownPropagator(acs))
    hz_result = materialize_ancestry!(acs, HorizontalPropagator(acs))
    um = materialize_universal!(acs)
    
    # Compute world fingerprint
    world_fp = splitmix64(
        bu_result.universal_fingerprint ⊻
        td_result.universal_fingerprint ⊻
        hz_result.universal_fingerprint ⊻
        seed
    )
    
    # RETURN THE WORLD — don't print it!
    ScopedPropagatorWorld(acs, bu_result, td_result, hz_result, um, world_fp)
end

# Composable operations
Base.length(w::ScopedPropagatorWorld) = length(w.acs.nodes)

function Base.merge(w1::ScopedPropagatorWorld, w2::ScopedPropagatorWorld)
    ids = unique([collect(keys(w1.acs.nodes)); collect(keys(w2.acs.nodes))])
    edges = unique([...])
    world_scoped_propagators(ids, edges)
end
```

---

## Transformation Template

### Before (Demo)

```julia
function demo_color_game()
    result = run_simulation()
    println("Result: $result")
    println("Score: $(result.score)")
    # nothing returned
end
```

### After (World)

```julia
struct ColorGameWorld
    state::SimulationState
    score::Float64
    history::Vector{Move}
    fingerprint::UInt64
end

function world_color_game(; seed::UInt64=GAY_SEED)
    result = run_simulation(seed)
    fp = splitmix64(hash(result.trajectory) ⊻ seed)
    ColorGameWorld(result, result.score, result.trajectory, fp)
end

Base.length(w::ColorGameWorld) = length(w.history)

function Base.merge(w1::ColorGameWorld, w2::ColorGameWorld)
    combined_history = vcat(w1.history, w2.history)
    combined_fp = splitmix64(w1.fingerprint ⊻ w2.fingerprint)
    ColorGameWorld(
        w2.state,  # or merge states somehow
        (w1.score + w2.score) / 2,
        combined_history,
        combined_fp
    )
end
```

---

## Composition Patterns

### Sequential Composition

```julia
w1 = world_color_game(seed=0x1234)
w2 = world_color_game(seed=w1.fingerprint)  # Chain fingerprints
w3 = world_color_game(seed=w2.fingerprint)
```

### Parallel Composition

```julia
worlds = [world_color_game(seed=UInt64(i)) for i in 1:100]
mega_world = reduce(merge, worlds)
```

### Functorial Mapping

```julia
function map_world(f, w::ColorGameWorld)
    new_history = map(f, w.history)
    ColorGameWorld(w.state, w.score, new_history, w.fingerprint)
end
```

---

## Testing Worlds

Worlds are trivially testable because they return values:

```julia
@testset "ColorGameWorld" begin
    w = world_color_game(seed=0x42)
    
    @test w.fingerprint != 0
    @test length(w) > 0
    @test w.score >= 0
    
    # Merge is associative
    w1 = world_color_game(seed=0x1)
    w2 = world_color_game(seed=0x2)
    w3 = world_color_game(seed=0x3)
    
    @test merge(merge(w1, w2), w3).fingerprint == merge(w1, merge(w2, w3)).fingerprint
end
```

---

## Decision Flowchart

```
Should we write a demo_* function?
           │
           ▼
          NO
           │
           ▼
What structure should this return?
           │
           ▼
Define the World struct with fingerprint
           │
           ▼
Write world_* builder function
           │
           ▼
Implement Base.length and Base.merge
           │
           ▼
Callers can print if they want
```

---

## The Motto

> **Demos perform once and vanish. Worlds accumulate and interconnect.**

A demo is a dead end. A world is a foundation.

---

## See Also

- [AGENTS.md](../AGENTS.md) — Enforcement rules
- [src/scoped_propagators.jl](../src/scoped_propagators.jl) — Canonical implementation
- [v0.2.0-DESIDERATA.md](v0.2.0-DESIDERATA.md) — Design goals
