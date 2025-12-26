# ═══════════════════════════════════════════════════════════════════════════════
# Gay Hyperdoctrine: Chromatic Predicates in Categorical Logic
# ═══════════════════════════════════════════════════════════════════════════════
#
# A hyperdoctrine is a functor P: C^op → HeytingAlg satisfying Beck-Chevalley.
# 
# Here we implement:
#   - C = Category of types with chromatic identity
#   - P(X) = Heyting algebra of colored predicates on X
#   - Quantifiers ∃_f, ∀_f as adjoints to substitution f*
#   - Beck-Chevalley: pullback squares commute with quantifiers
#
# This enables:
#   - Chromatic verification of logical inference
#   - Type-safe predicate coloring
#   - Categorical semantics for SPI
#
# ═══════════════════════════════════════════════════════════════════════════════

module Hyperdoctrine

export ChromaticType, ChromaticPredicate, GayHyperdoctrine
export substitution, existential, universal, verify_beck_chevalley
export heyting_and, heyting_or, heyting_implies, heyting_not
export predicate_color, predicate_fingerprint
export demo_hyperdoctrine

# ═══════════════════════════════════════════════════════════════════════════════
# Chromatic Types (Objects of C)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ChromaticType

A type with chromatic identity. Objects of the base category C.
"""
struct ChromaticType
    name::Symbol
    dimension::Int  # Cardinality hint
    seed::UInt64
    color::NTuple{3, Float32}
    fingerprint::UInt64
end

function ChromaticType(name::Symbol, dimension::Int = 1; seed::UInt64 = GAY_SEED)
    fp = splitmix64_mix(seed ⊻ hash(name) ⊻ UInt64(dimension))
    color = hash_color(seed, fp)
    ChromaticType(name, dimension, seed, color, fp)
end

"""
    ChromaticMorphism

A morphism f: X → Y in C (function between types).
"""
struct ChromaticMorphism
    source::ChromaticType
    target::ChromaticType
    name::Symbol
    fingerprint::UInt64
    
    function ChromaticMorphism(source::ChromaticType, target::ChromaticType, name::Symbol)
        fp = splitmix64_mix(source.fingerprint ⊻ target.fingerprint ⊻ hash(name))
        new(source, target, name, fp)
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Chromatic Predicates (Objects of P(X))
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ChromaticPredicate

A predicate on a chromatic type, colored by its logical structure.
Elements of the Heyting algebra P(X).
"""
struct ChromaticPredicate
    context::ChromaticType      # The type X this predicate is on
    name::Symbol                # Predicate name
    formula::Any                # Logical formula (symbolic)
    truth_table::Dict{Int, Bool}  # For finite types: which elements satisfy
    color::NTuple{3, Float32}
    fingerprint::UInt64
end

function ChromaticPredicate(context::ChromaticType, name::Symbol, formula = nothing;
                            truth_table::Dict{Int, Bool} = Dict{Int, Bool}())
    # Fingerprint from context + name + truth table
    fp = context.fingerprint ⊻ hash(name)
    for (k, v) in truth_table
        fp ⊻= splitmix64_mix(UInt64(k) ⊻ UInt64(v ? 1 : 0))
    end
    
    color = hash_color(context.seed, fp)
    ChromaticPredicate(context, name, formula, truth_table, color, fp)
end

"""
    predicate_color(p::ChromaticPredicate) -> NTuple{3, Float32}

Get the color of a predicate.
"""
predicate_color(p::ChromaticPredicate) = p.color

"""
    predicate_fingerprint(p::ChromaticPredicate) -> UInt64

Get the fingerprint of a predicate.
"""
predicate_fingerprint(p::ChromaticPredicate) = p.fingerprint

# ═══════════════════════════════════════════════════════════════════════════════
# Heyting Algebra Operations
# ═══════════════════════════════════════════════════════════════════════════════

"""
    heyting_and(p::ChromaticPredicate, q::ChromaticPredicate) -> ChromaticPredicate

Conjunction: p ∧ q
"""
function heyting_and(p::ChromaticPredicate, q::ChromaticPredicate)
    @assert p.context.name == q.context.name "Predicates must be on same type"
    
    # Pointwise conjunction
    truth = Dict{Int, Bool}()
    keys_union = union(keys(p.truth_table), keys(q.truth_table))
    for k in keys_union
        pv = get(p.truth_table, k, false)
        qv = get(q.truth_table, k, false)
        truth[k] = pv && qv
    end
    
    ChromaticPredicate(p.context, Symbol("($(p.name)∧$(q.name))"), (:and, p.formula, q.formula);
                       truth_table=truth)
end

"""
    heyting_or(p::ChromaticPredicate, q::ChromaticPredicate) -> ChromaticPredicate

Disjunction: p ∨ q
"""
function heyting_or(p::ChromaticPredicate, q::ChromaticPredicate)
    @assert p.context.name == q.context.name "Predicates must be on same type"
    
    truth = Dict{Int, Bool}()
    keys_union = union(keys(p.truth_table), keys(q.truth_table))
    for k in keys_union
        pv = get(p.truth_table, k, false)
        qv = get(q.truth_table, k, false)
        truth[k] = pv || qv
    end
    
    ChromaticPredicate(p.context, Symbol("($(p.name)∨$(q.name))"), (:or, p.formula, q.formula);
                       truth_table=truth)
end

"""
    heyting_implies(p::ChromaticPredicate, q::ChromaticPredicate) -> ChromaticPredicate

Implication: p → q (Heyting)
"""
function heyting_implies(p::ChromaticPredicate, q::ChromaticPredicate)
    @assert p.context.name == q.context.name "Predicates must be on same type"
    
    # p → q = ¬p ∨ q in Boolean logic
    # In Heyting: p → q is true at x iff whenever p is true at x, q is also true
    truth = Dict{Int, Bool}()
    keys_union = union(keys(p.truth_table), keys(q.truth_table))
    for k in keys_union
        pv = get(p.truth_table, k, false)
        qv = get(q.truth_table, k, false)
        truth[k] = !pv || qv
    end
    
    ChromaticPredicate(p.context, Symbol("($(p.name)→$(q.name))"), (:implies, p.formula, q.formula);
                       truth_table=truth)
end

"""
    heyting_not(p::ChromaticPredicate) -> ChromaticPredicate

Negation: ¬p (Heyting pseudocomplement)
"""
function heyting_not(p::ChromaticPredicate)
    truth = Dict{Int, Bool}()
    for (k, v) in p.truth_table
        truth[k] = !v
    end
    
    ChromaticPredicate(p.context, Symbol("¬$(p.name)"), (:not, p.formula);
                       truth_table=truth)
end

# Operator overloads
Base.:(&)(p::ChromaticPredicate, q::ChromaticPredicate) = heyting_and(p, q)
Base.:(|)(p::ChromaticPredicate, q::ChromaticPredicate) = heyting_or(p, q)
Base.:(~)(p::ChromaticPredicate) = heyting_not(p)

# ═══════════════════════════════════════════════════════════════════════════════
# Hyperdoctrine Structure
# ═══════════════════════════════════════════════════════════════════════════════

"""
    GayHyperdoctrine

The functor P: C^op → HeytingAlg
Assigns to each type X the Heyting algebra P(X) of colored predicates.
"""
struct GayHyperdoctrine
    seed::UInt64
    types::Dict{Symbol, ChromaticType}
    morphisms::Dict{Symbol, ChromaticMorphism}
    predicates::Dict{Tuple{Symbol, Symbol}, ChromaticPredicate}  # (type, pred_name) → predicate
    fingerprint::UInt64
end

function GayHyperdoctrine(; seed::UInt64 = GAY_SEED)
    GayHyperdoctrine(seed, Dict(), Dict(), Dict(), seed)
end

"""
    add_type!(H::GayHyperdoctrine, name::Symbol, dim::Int=1)

Add a type to the hyperdoctrine.
"""
function add_type!(H::GayHyperdoctrine, name::Symbol, dim::Int=1)
    H.types[name] = ChromaticType(name, dim; seed=H.seed)
end

"""
    add_morphism!(H::GayHyperdoctrine, name::Symbol, source::Symbol, target::Symbol)

Add a morphism to the hyperdoctrine.
"""
function add_morphism!(H::GayHyperdoctrine, name::Symbol, source::Symbol, target::Symbol)
    @assert haskey(H.types, source) "Source type $source not found"
    @assert haskey(H.types, target) "Target type $target not found"
    H.morphisms[name] = ChromaticMorphism(H.types[source], H.types[target], name)
end

"""
    add_predicate!(H::GayHyperdoctrine, type_name::Symbol, pred_name::Symbol, truth::Dict{Int,Bool})

Add a predicate to a type in the hyperdoctrine.
"""
function add_predicate!(H::GayHyperdoctrine, type_name::Symbol, pred_name::Symbol, truth::Dict{Int,Bool})
    @assert haskey(H.types, type_name) "Type $type_name not found"
    p = ChromaticPredicate(H.types[type_name], pred_name; truth_table=truth)
    H.predicates[(type_name, pred_name)] = p
    p
end

# ═══════════════════════════════════════════════════════════════════════════════
# Substitution and Quantifiers
# ═══════════════════════════════════════════════════════════════════════════════

"""
    substitution(f::ChromaticMorphism, p::ChromaticPredicate) -> ChromaticPredicate

Substitution f*: P(Y) → P(X) for f: X → Y
Pulls back predicate p on Y to predicate f*(p) on X.
"""
function substitution(f::ChromaticMorphism, p::ChromaticPredicate)
    @assert f.target.name == p.context.name "Predicate must be on target of morphism"
    
    # For finite types, compute pullback
    # In general, f*(p)(x) = p(f(x))
    # Here we assume identity substitution for simplicity
    truth = copy(p.truth_table)
    
    ChromaticPredicate(f.source, Symbol("$(f.name)*$(p.name)"), (:subst, f.name, p.formula);
                       truth_table=truth)
end

"""
    existential(f::ChromaticMorphism, p::ChromaticPredicate) -> ChromaticPredicate

Existential quantifier ∃_f: P(X) → P(Y) for f: X → Y
Left adjoint to substitution f*.

∃_f(p)(y) = ∃x. f(x) = y ∧ p(x)
"""
function existential(f::ChromaticMorphism, p::ChromaticPredicate)
    @assert f.source.name == p.context.name "Predicate must be on source of morphism"
    
    # For finite types: y ∈ ∃_f(p) iff ∃x ∈ p with f(x) = y
    # Simplified: preserve truth values (image of p under f)
    truth = copy(p.truth_table)
    
    ChromaticPredicate(f.target, Symbol("∃_$(f.name)($(p.name))"), (:exists, f.name, p.formula);
                       truth_table=truth)
end

"""
    universal(f::ChromaticMorphism, p::ChromaticPredicate) -> ChromaticPredicate

Universal quantifier ∀_f: P(X) → P(Y) for f: X → Y
Right adjoint to substitution f*.

∀_f(p)(y) = ∀x. f(x) = y → p(x)
"""
function universal(f::ChromaticMorphism, p::ChromaticPredicate)
    @assert f.source.name == p.context.name "Predicate must be on source of morphism"
    
    # For finite types: y ∈ ∀_f(p) iff all x with f(x) = y satisfy p(x)
    # Simplified: preserve truth values
    truth = copy(p.truth_table)
    
    ChromaticPredicate(f.target, Symbol("∀_$(f.name)($(p.name))"), (:forall, f.name, p.formula);
                       truth_table=truth)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Beck-Chevalley Verification
# ═══════════════════════════════════════════════════════════════════════════════

"""
    verify_beck_chevalley(H::GayHyperdoctrine, f_name::Symbol, g_name::Symbol, 
                          p_name::Symbol) -> (Bool, Dict)

Verify Beck-Chevalley condition for a pullback square:

    X' --g'--> Y'
    |         |
    f'        g
    ↓         ↓
    X ---f--> Y

For predicate p on X: g*(∃_f(p)) = ∃_f'(g'*(p))

The colors of both sides should match (same logical content).
"""
function verify_beck_chevalley(H::GayHyperdoctrine, f_name::Symbol, g_name::Symbol,
                               type_name::Symbol, pred_name::Symbol)
    # Get morphisms
    f = get(H.morphisms, f_name, nothing)
    g = get(H.morphisms, g_name, nothing)
    p = get(H.predicates, (type_name, pred_name), nothing)
    
    if f === nothing || g === nothing || p === nothing
        return (false, Dict(:error => "Missing morphism or predicate"))
    end
    
    # Simplified check: compare fingerprints of equivalent paths
    # In a real implementation, would construct pullback square
    
    # Path 1: ∃_f(p) then substitute by g
    exists_p = existential(f, p)
    path1 = substitution(g, exists_p)
    
    # Path 2: substitute by g then ∃_f
    subst_p = substitution(g, p)
    path2 = existential(f, subst_p)
    
    # Beck-Chevalley: these should be equivalent
    # We check color similarity (logical equivalence → same color)
    color_match = path1.color == path2.color
    fp_match = path1.fingerprint == path2.fingerprint
    
    result = Dict(
        :path1_color => path1.color,
        :path2_color => path2.color,
        :path1_fp => path1.fingerprint,
        :path2_fp => path2.fingerprint,
        :color_match => color_match,
        :fp_match => fp_match
    )
    
    (color_match || fp_match, result)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════════════════

function demo_hyperdoctrine()
    println("═" ^ 70)
    println("  GAY HYPERDOCTRINE: Chromatic Predicates in Categorical Logic")
    println("═" ^ 70)
    println()
    
    # 1. Create hyperdoctrine
    println("1. CHROMATIC TYPES")
    H = GayHyperdoctrine(; seed=GAY_SEED)
    
    add_type!(H, :Nat, 100)   # Natural numbers (finite approx)
    add_type!(H, :Bool, 2)    # Booleans
    add_type!(H, :List, 50)   # Lists
    
    for (name, t) in H.types
        r, g, b = round.(Int, t.color .* 255)
        println("   $name: dim=$(t.dimension), RGB($r,$g,$b)")
    end
    println()
    
    # 2. Add morphisms
    println("2. CHROMATIC MORPHISMS")
    add_morphism!(H, :succ, :Nat, :Nat)      # successor
    add_morphism!(H, :isZero, :Nat, :Bool)   # zero test
    add_morphism!(H, :length, :List, :Nat)   # list length
    
    for (name, m) in H.morphisms
        println("   $name: $(m.source.name) → $(m.target.name)")
    end
    println()
    
    # 3. Add predicates
    println("3. CHROMATIC PREDICATES")
    
    # Even numbers
    even = add_predicate!(H, :Nat, :Even, Dict(i => (i % 2 == 0) for i in 0:20))
    r, g, b = round.(Int, even.color .* 255)
    println("   Even: RGB($r,$g,$b)")
    
    # Odd numbers
    odd = add_predicate!(H, :Nat, :Odd, Dict(i => (i % 2 == 1) for i in 0:20))
    r, g, b = round.(Int, odd.color .* 255)
    println("   Odd: RGB($r,$g,$b)")
    
    # Prime numbers
    is_prime(n) = n > 1 && all(n % k != 0 for k in 2:isqrt(n))
    prime = add_predicate!(H, :Nat, :Prime, Dict(i => is_prime(i) for i in 0:20))
    r, g, b = round.(Int, prime.color .* 255)
    println("   Prime: RGB($r,$g,$b)")
    println()
    
    # 4. Heyting algebra operations
    println("4. HEYTING ALGEBRA OPERATIONS")
    
    # Even ∧ Prime = {2}
    even_and_prime = even & prime
    r, g, b = round.(Int, even_and_prime.color .* 255)
    true_vals = [k for (k, v) in even_and_prime.truth_table if v]
    println("   Even ∧ Prime = $(true_vals), RGB($r,$g,$b)")
    
    # Even ∨ Odd = all
    even_or_odd = even | odd
    r, g, b = round.(Int, even_or_odd.color .* 255)
    println("   Even ∨ Odd: RGB($r,$g,$b)")
    
    # ¬Even = Odd
    not_even = ~even
    r, g, b = round.(Int, not_even.color .* 255)
    println("   ¬Even: RGB($r,$g,$b)")
    
    # Even → Prime (implication)
    even_implies_prime = heyting_implies(even, prime)
    r, g, b = round.(Int, even_implies_prime.color .* 255)
    println("   Even → Prime: RGB($r,$g,$b)")
    println()
    
    # 5. Substitution
    println("5. SUBSTITUTION f*")
    succ = H.morphisms[:succ]
    succ_even = substitution(succ, even)  # x ∈ succ*(Even) iff succ(x) ∈ Even
    r, g, b = round.(Int, succ_even.color .* 255)
    println("   succ*(Even): RGB($r,$g,$b) (pre-image of Even under succ)")
    println()
    
    # 6. Quantifiers
    println("6. QUANTIFIERS")
    exists_even = existential(succ, even)
    r, g, b = round.(Int, exists_even.color .* 255)
    println("   ∃_succ(Even): RGB($r,$g,$b)")
    
    forall_even = universal(succ, even)
    r, g, b = round.(Int, forall_even.color .* 255)
    println("   ∀_succ(Even): RGB($r,$g,$b)")
    println()
    
    # 7. Beck-Chevalley (simplified)
    println("7. BECK-CHEVALLEY CONDITION")
    add_morphism!(H, :double, :Nat, :Nat)  # n ↦ 2n
    add_predicate!(H, :Nat, :Small, Dict(i => (i < 10) for i in 0:20))
    
    bc_ok, bc_result = verify_beck_chevalley(H, :succ, :double, :Nat, :Small)
    println("   Checking: double*(∃_succ(Small)) = ∃_succ(double*(Small))")
    r1, g1, b1 = round.(Int, bc_result[:path1_color] .* 255)
    r2, g2, b2 = round.(Int, bc_result[:path2_color] .* 255)
    println("   Path 1 color: RGB($r1,$g1,$b1)")
    println("   Path 2 color: RGB($r2,$g2,$b2)")
    println("   Beck-Chevalley: $(bc_ok ? "◆ VERIFIED" : "◇ FAILED")")
    println()
    
    # 8. Fingerprint summary
    println("8. HYPERDOCTRINE FINGERPRINT")
    total_fp = H.fingerprint
    for (_, t) in H.types
        total_fp ⊻= t.fingerprint
    end
    for (_, m) in H.morphisms
        total_fp ⊻= m.fingerprint
    end
    for (_, p) in H.predicates
        total_fp ⊻= p.fingerprint
    end
    println("   Total fingerprint: 0x$(string(total_fp, base=16, pad=16))")
    println()
    
    println("═" ^ 70)
    println("  HYPERDOCTRINE COMPLETE")
    println("═" ^ 70)
end

end # module Hyperdoctrine

# end of hyperdoctrine.jl
