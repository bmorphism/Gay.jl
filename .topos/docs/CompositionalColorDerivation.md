# Compositional Color Derivation: When Naive DP Fails

> Ensuring blanket_color = XOR(component_colors) for indefinitely nested Gay structures

## The Problem

Naive dynamic programming assumes:
1. **Optimal substructure**: color(A∘B) = f(color(A), color(B))
2. **Overlapping subproblems**: same subexpression → same color
3. **Memoization safety**: cache hits preserve semantics

**Gay.jl violations:**

```julia
# WRONG: TwoPara uses abs(a-b), not XOR
combined = RGB(
    abs(red(outer_color) - red(inner_color)),   # Non-associative!
    abs(green(outer_color) - green(inner_color)),
    abs(blue(outer_color) - blue(inner_color))
)

# WRONG: Order-dependent seed derivation
seed_ab = seed_a ⊻ hash(b)  # ≠ seed_b ⊻ hash(a)
```

## Correct: XOR-Based Blanket Coloring

### Invariant: Blanket = XOR of Components

```julia
"""
Dynamic Markov Blanket color = XOR of all component colors.

This is ASSOCIATIVE and COMMUTATIVE:
  (A ⊻ B) ⊻ C = A ⊻ (B ⊻ C)    # Associative
  A ⊻ B = B ⊻ A                 # Commutative

Therefore: safe for parallel reduction, memoization, any evaluation order.
"""
struct BlanketColor
    components::Vector{UInt64}   # Component fingerprints
    blanket_fp::UInt64           # = reduce(⊻, components)
    color::RGB{Float64}          # = color_from_seed(blanket_fp)
end

function BlanketColor(components::Vector{UInt64})
    blanket_fp = reduce(⊻, components; init=UInt64(0))
    color = color_from_seed(blanket_fp)
    BlanketColor(components, blanket_fp, color)
end

# Adding a component: O(1)
function add_component(bc::BlanketColor, new_fp::UInt64)
    new_blanket = bc.blanket_fp ⊻ new_fp
    BlanketColor([bc.components; new_fp], new_blanket, color_from_seed(new_blanket))
end

# Removing a component: O(1) via self-inverse
function remove_component(bc::BlanketColor, old_fp::UInt64)
    new_blanket = bc.blanket_fp ⊻ old_fp  # XOR is self-inverse
    new_components = filter(≠(old_fp), bc.components)
    BlanketColor(new_components, new_blanket, color_from_seed(new_blanket))
end
```

### Parentheses Derivation

For nested structures like `((A B) (C D))`:

```julia
"""
Parentheses-aware color derivation.

Structure:
    expr = ((A B) (C D))
    
Color derivation:
    fp(A), fp(B), fp(C), fp(D)         # Leaf fingerprints
    fp(AB) = fp(A) ⊻ fp(B)             # Inner left
    fp(CD) = fp(C) ⊻ fp(D)             # Inner right
    fp(expr) = fp(AB) ⊻ fp(CD)         # Root
             = fp(A) ⊻ fp(B) ⊻ fp(C) ⊻ fp(D)  # Flattened!

Key insight: XOR is associative, so nested parens don't matter for color.
The STRUCTURE matters for semantics, but COLOR flattens to leaf XOR.
"""
abstract type GayExpr end

struct GayLeaf <: GayExpr
    value::Any
    fp::UInt64
end

struct GayNode <: GayExpr
    left::GayExpr
    right::GayExpr
    # Derived (cached)
    fp::UInt64
end

function GayNode(left::GayExpr, right::GayExpr)
    fp = fingerprint(left) ⊻ fingerprint(right)
    GayNode(left, right, fp)
end

fingerprint(e::GayLeaf) = e.fp
fingerprint(e::GayNode) = e.fp

# Color is derived from fingerprint
color(e::GayExpr) = color_from_seed(fingerprint(e))

# THEOREM: For any tree structure, fingerprint = XOR of all leaf fingerprints
function verify_flat_xor(expr::GayExpr)
    leaves = collect_leaves(expr)
    flat_fp = reduce(⊻, [fingerprint(l) for l in leaves]; init=UInt64(0))
    flat_fp == fingerprint(expr)  # Always true!
end
```

## When Naive DP Fails

### Failure Mode 1: Non-XOR Combination

```julia
# BAD: abs(a - b) is not associative
function bad_combine(c1::RGB, c2::RGB)
    RGB(abs(c1.r - c2.r), abs(c1.g - c2.g), abs(c1.b - c2.b))
end

# abs(abs(a-b) - c) ≠ abs(a - abs(b-c))  # FAILS

# GOOD: XOR on fingerprints, then derive color
function good_combine(fp1::UInt64, fp2::UInt64)
    color_from_seed(fp1 ⊻ fp2)
end
```

### Failure Mode 2: Order-Dependent Seed Derivation

```julia
# BAD: hash breaks commutativity
seed_ab = seed_a ⊻ hash(b)
seed_ba = seed_b ⊻ hash(a)
# seed_ab ≠ seed_ba in general!

# GOOD: Use fingerprints directly
fp_ab = fp_a ⊻ fp_b
fp_ba = fp_b ⊻ fp_a
# fp_ab == fp_ba ALWAYS
```

### Failure Mode 3: Memoization Key Collision

```julia
# BAD: Same subexpression at different depths gets different colors
memo = Dict{Expr, RGB}()
# ((A B) C) and (A (B C)) might cache differently

# GOOD: Memoize by fingerprint, not structure
memo = Dict{UInt64, RGB}()
function memoized_color(expr::GayExpr)
    fp = fingerprint(expr)
    get!(memo, fp) do
        color_from_seed(fp)
    end
end
```

## Pattern: Compositionally Sound Nesting

```julia
"""
    GayComposable

Base type for compositionally sound Gay structures.

Invariants:
1. fingerprint(compose(a, b)) = fingerprint(a) ⊻ fingerprint(b)
2. color(x) = color_from_seed(fingerprint(x))
3. fingerprint is computed bottom-up via XOR
4. Memoization keyed by fingerprint, not structure
"""
abstract type GayComposable end

# Required method
fingerprint(x::GayComposable)::UInt64 = error("Implement fingerprint")

# Derived methods (DO NOT OVERRIDE)
color(x::GayComposable) = color_from_seed(fingerprint(x))

function compose(a::GayComposable, b::GayComposable)
    ComposedGay(a, b, fingerprint(a) ⊻ fingerprint(b))
end

struct ComposedGay{A<:GayComposable, B<:GayComposable} <: GayComposable
    left::A
    right::B
    fp::UInt64
end

fingerprint(c::ComposedGay) = c.fp
```

## Pattern: Safe Nested Evaluation

```julia
"""
Safely evaluate nested Gay expressions with correct color propagation.

Uses:
1. Bottom-up fingerprint accumulation (XOR)
2. Lazy color derivation (only when observed)
3. Parallel-safe reduction (XOR is associative+commutative)
"""
struct NestedGay{T}
    value::T
    children::Vector{NestedGay{T}}
    fp::UInt64
    _color::Base.RefValue{Union{Nothing, RGB{Float64}}}
end

function NestedGay(value::T, children::Vector{NestedGay{T}}, seed::UInt64) where T
    # Fingerprint = seed XOR all child fingerprints
    child_fps = [c.fp for c in children]
    fp = reduce(⊻, child_fps; init=seed)
    NestedGay{T}(value, children, fp, Ref{Union{Nothing, RGB{Float64}}}(nothing))
end

# Lazy color: computed once, cached
function color(ng::NestedGay)
    if isnothing(ng._color[])
        ng._color[] = color_from_seed(ng.fp)
    end
    ng._color[]
end

# Parallel-safe: can split tree arbitrarily
function parallel_fingerprint(ng::NestedGay)
    if isempty(ng.children)
        ng.fp
    else
        # Safe to parallelize: XOR is associative
        child_fps = fetch.([Threads.@spawn parallel_fingerprint(c) for c in ng.children])
        reduce(⊻, child_fps; init=ng.fp)
    end
end
```

## Pattern: Dynamic Blanket Update

```julia
"""
Efficiently update blanket color when components change.

Operations:
- add_component: O(1)
- remove_component: O(1) 
- query_blanket_color: O(1)

Uses XOR's self-inverse property: a ⊻ a = 0
"""
mutable struct DynamicBlanket
    component_fps::Set{UInt64}
    blanket_fp::UInt64
end

DynamicBlanket() = DynamicBlanket(Set{UInt64}(), UInt64(0))

function add!(db::DynamicBlanket, fp::UInt64)
    if fp ∉ db.component_fps
        push!(db.component_fps, fp)
        db.blanket_fp ⊻= fp
    end
    db
end

function remove!(db::DynamicBlanket, fp::UInt64)
    if fp ∈ db.component_fps
        delete!(db.component_fps, fp)
        db.blanket_fp ⊻= fp  # Self-inverse!
    end
    db
end

blanket_color(db::DynamicBlanket) = color_from_seed(db.blanket_fp)

# Transaction: atomic update of multiple components
function transact!(db::DynamicBlanket, adds::Vector{UInt64}, removes::Vector{UInt64})
    # Compute delta
    delta = reduce(⊻, adds; init=UInt64(0)) ⊻ reduce(⊻, removes; init=UInt64(0))
    
    # Apply atomically
    for fp in adds
        push!(db.component_fps, fp)
    end
    for fp in removes
        delete!(db.component_fps, fp)
    end
    db.blanket_fp ⊻= delta
    db
end
```

## Anti-Pattern Checklist

| Anti-Pattern | Why It Fails | Correct Pattern |
|--------------|--------------|-----------------|
| `abs(a - b)` for color combine | Non-associative | `color_from_seed(fp_a ⊻ fp_b)` |
| `hash(x)` in seed derivation | Non-commutative | Pre-compute fingerprints, XOR them |
| Memoize by AST structure | Same fp, different cache | Memoize by fingerprint |
| Sequential color accumulation | Order-dependent | Parallel XOR reduction |
| Store color, derive fingerprint | Lossy (color space < UInt64) | Store fingerprint, derive color |
| `+` or `*` for fp combination | Overflow, non-invertible | XOR only |

## Summary

**The One Rule**: Fingerprint composition is ALWAYS XOR.

```julia
# For ANY nested Gay structure:
fingerprint(root) = reduce(⊻, [fingerprint(leaf) for leaf in leaves(root)])

# Color is derived, never stored or combined directly:
color(x) = color_from_seed(fingerprint(x))
```

This ensures:
- ✓ Associativity: `(A ⊻ B) ⊻ C = A ⊻ (B ⊻ C)`
- ✓ Commutativity: `A ⊻ B = B ⊻ A`
- ✓ Self-inverse: `A ⊻ A = 0` (for removal)
- ✓ Identity: `A ⊻ 0 = A`
- ✓ Parallel-safe: any split order gives same result
- ✓ Memoization-safe: fingerprint uniquely identifies color
- ✓ Compositional: blanket = XOR of components
