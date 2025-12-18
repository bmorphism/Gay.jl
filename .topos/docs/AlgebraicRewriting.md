# AlgebraicRewriting.jl

> Source: https://algebraicjulia.github.io/AlgebraicRewriting.jl/stable/

AlgebraicRewriting.jl provides tools for algebraic rewriting—context-aware find-and-replace operations that maintain structure. Rewrite rules adhere to structures defined using ACSets (from ACSets.jl and Catlab.jl).

## Rewrite Rule Design Process

### 1. Define Schema
```julia
@present SchExample(FreeSchema) begin
  Obj::Ob
  Hom::Hom(Obj, Obj)
  AttrType::AttrType
  Attr::Attr(Obj, AttrType)
end
```

### 2. Create ACSet Type
```julia
@acset_type Example(SchExample)
```

### 3. Define Rule Parts (Span: L ←l— K —r→ R)
- **L** (Left): Pre-condition pattern
- **K** (Keep): Preserved structure
- **R** (Right): Post-condition/effect
- **l, r**: ACSet transformations embedding K

### 4. Construct Rule
```julia
rule = Rule{:DPO}(l, r)  # Double-pushout
# Also: :SPO, :SqPO, :PBPO
```

## ACSet Instantiation Methods

### Static (`@acset`)
```julia
L = @acset Example{String} begin
  Obj = 2
  Hom = [1, 2]
  Attr = AttrVar.(1:2)
end
```

### Colimit-of-Representables (`@acset_colim`)
```julia
yExample = yoneda(Example{String})
L = @acset_colim yExample begin
  (x, y)::Obj
  Hom(x) == y
end
```

## Rule Application

### Find Matches
```julia
matches = get_matches(rule, state)
pattern_match = homomorphism(L, state; monic=true)
```

### Apply Rewrite
```julia
result = rewrite_match(rule, pattern_match)
```

## Rewrite Semantics
- **DPO** (Double Pushout) - Standard, requires gluing condition
- **SPO** (Single Pushout) - Deletes dangling edges
- **SqPO** (Sesqui-Pushout) - Clones matched elements
- **PBPO** (Pullback-Pushout) - Most general

## Gay.jl Extension Target: GayAlgebraicRewritingExt

### Key Types to Color
- ACSet patterns (L, K, R)
- ACSet transformations (l, r, match)
- Rewrite rules
- Match morphisms

### SPI Opportunities
- Pattern element coloring: `hash_color(element_id, seed)`
- Transformation coloring by (domain, codomain) XOR
- Rule fingerprinting via L ⊻ K ⊻ R hashes
- Match highlighting with distinct pattern colors

### Parallel Tractability
- Parallel match enumeration
- Concurrent independent rule applications
- Distributed rewriting strategies
- XOR fingerprint for confluence checking

### Integration Points
- Graph transformation visualization
- Petri net rewriting (with AlgebraicPetri)
- Schema migration coloring
- Model transformation debugging
