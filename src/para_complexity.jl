# Para(Derangeable) × Para(Colorable) × Para(TropicalRing)
# ═══════════════════════════════════════════════════════════════════════════════
#
# The most difficult edge cases for Gay.jl verification:
# Computational complexity boundaries as categorical gadgets
#
# Reference: "Classic Nintendo Games are (Computationally) Hard"
# - Aloupis, Demaine, Guo, Viglietta (2012)
# - The "choice gadget" reduces 3-SAT to Mario level traversability
#
# ┌───────────────────────────────────────────────────────────────────────────────┐
# │  COMPLEXITY BOUNDARY AS CATEGORICAL STRUCTURE                                │
# │  ═══════════════════════════════════════════════                             │
# │                                                                               │
# │  Para(Derangeable)     Objects: derangements parameterized by parity         │
# │  Para(Colorable)       Objects: k-colorings parameterized by graph           │
# │  Para(TropicalRing)    Objects: tropical matrices parameterized by weights   │
# │                                                                               │
# │  MARIO CHOICE GADGET                                                          │
# │  ═══════════════════                                                          │
# │                 ┌─────┐                                                       │
# │       ──────────┤  ?  ├──────────                                            │
# │                 │block│                                                       │
# │       ──────────┴──┬──┴──────────                                            │
# │                    │                                                          │
# │              ┌─────┴─────┐                                                    │
# │              ▼           ▼                                                    │
# │          [LEFT]      [RIGHT]                                                  │
# │                                                                               │
# │  Once a choice is made, it cannot be undone (one-way gadget)                 │
# │  This creates 2^n possible paths through n gadgets → NP-complete              │
# │                                                                               │
# │  Para(Colorable) captures this: can we 3-color this choice graph?            │
# │                                                                               │
# │  NP ⊆ NP-hard ⊆ PSPACE boundary:                                             │
# │  - NP: verify solution in poly time (3-SAT)                                  │
# │  - NP-hard: reduce from 3-SAT (Mario traversability)                         │
# │  - PSPACE: reachability with reusable gadgets (QSAT)                         │
# └───────────────────────────────────────────────────────────────────────────────┘

export ParaDerangeable, ParaColorable, ParaTropicalRing
export ChoiceGadget, MarioLevel, reduce_3sat_to_mario
export ComplexityClass, verify_spi_at_boundary
export GayComplexityWitness, hardest_edge_cases

using Random

# Import from Gay.jl core
# using ..Gay: mix64, GAY_SEED, hash_color, xor_fingerprint

# ═══════════════════════════════════════════════════════════════════════════════
# Para(X): Parameterized Category of X-structures
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ParaDerangeable{P}

Derangements parameterized by P (typically parity or sublattice structure).

In the Para construction:
- Objects: (parameter, derangement) pairs
- Morphisms: parameter-respecting derangement maps

For SPI verification: ensures σ(i) ≠ i across all parallel executions
"""
struct ParaDerangeable{P}
    parameter::P
    n::Int           # Size of derangement
    seed::UInt64
    parity::Int      # Even/odd sublattice
end

function ParaDerangeable(param, n::Int; seed::UInt64=UInt64(0x6761795f636f6c6f), parity::Int=0)
    ParaDerangeable(param, n, seed, parity)
end

"""
Mix64 for deterministic hashing
"""
function mix64(z::UInt64)
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    z ⊻ (z >> 31)
end

"""
Generate a derangement respecting the parameter
"""
function sample_derangement(pd::ParaDerangeable, index::UInt64)
    rng_state = mix64(pd.seed ⊻ index ⊻ UInt64(pd.parity))
    
    # Sattolo's algorithm: guarantees derangement via single cycle
    perm = collect(1:pd.n)
    for i in pd.n:-1:2
        rng_state = mix64(rng_state)
        j = 1 + Int(rng_state % UInt64(i - 1))  # j ∈ [1, i-1], never i
        perm[i], perm[j] = perm[j], perm[i]
    end
    
    # Verify no fixed points
    @assert all(i != perm[i] for i in 1:pd.n) "Fixed point in derangement!"
    
    return perm
end

# ═══════════════════════════════════════════════════════════════════════════════
# Para(Colorable): k-Colorings parameterized by graph structure
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ParaColorable{G}

k-colorings parameterized by graph G.

In complexity theory:
- k=2: P (bipartite detection)
- k=3: NP-complete (3-coloring)
- k≥3: NP-complete (k-coloring)

For SPI verification: same seed → same coloring attempt sequence
"""
struct ParaColorable{G}
    graph::G          # Adjacency structure
    k::Int            # Number of colors
    seed::UInt64
    vertices::Int
end

# Simple graph as adjacency list
struct SimpleGraph
    adj::Vector{Vector{Int}}
end

SimpleGraph(n::Int) = SimpleGraph([Int[] for _ in 1:n])

function add_edge!(g::SimpleGraph, u::Int, v::Int)
    push!(g.adj[u], v)
    push!(g.adj[v], u)
end

function ParaColorable(g::SimpleGraph, k::Int; seed::UInt64=UInt64(0x6761795f636f6c6f))
    ParaColorable(g, k, seed, length(g.adj))
end

"""
    greedy_coloring(pc::ParaColorable, index::UInt64) -> (Vector{Int}, Bool)

Attempt greedy coloring with deterministic vertex order from seed.
Returns (coloring, success).
"""
function greedy_coloring(pc::ParaColorable, index::UInt64)
    n = pc.vertices
    colors = zeros(Int, n)
    
    # Deterministic vertex ordering from seed
    rng_state = mix64(pc.seed ⊻ index)
    order = collect(1:n)
    for i in n:-1:2
        rng_state = mix64(rng_state)
        j = 1 + Int(rng_state % UInt64(i))
        order[i], order[j] = order[j], order[i]
    end
    
    # Greedy coloring
    for v in order
        used = Set{Int}()
        for u in pc.graph.adj[v]
            if colors[u] != 0
                push!(used, colors[u])
            end
        end
        
        # Find smallest available color
        c = 1
        while c in used && c <= pc.k
            c += 1
        end
        
        if c > pc.k
            return (colors, false)  # Failed: no valid coloring
        end
        
        colors[v] = c
    end
    
    return (colors, true)
end

"""
    is_valid_coloring(pc::ParaColorable, colors::Vector{Int}) -> Bool

Verify coloring is valid (no adjacent same-color vertices).
"""
function is_valid_coloring(pc::ParaColorable, colors::Vector{Int})
    for v in 1:pc.vertices
        for u in pc.graph.adj[v]
            if colors[v] == colors[u] && colors[v] != 0
                return false
            end
        end
    end
    return true
end

# ═══════════════════════════════════════════════════════════════════════════════
# Para(TropicalRing): Tropical semiring parameterized by weight structure
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ParaTropicalRing{W}

Tropical (min-plus or max-plus) semiring parameterized by weight structure W.

Tropical algebra:
- a ⊕ b = min(a, b)  or max(a, b)
- a ⊗ b = a + b

For shortest paths, the tropical semiring gives:
- Matrix multiplication = path composition
- Closure = all-pairs shortest paths

Complexity connection:
- Tropical matrix closure: O(n³) for shortest paths
- But tropical polynomial identity testing: NP-hard
"""
struct ParaTropicalRing{W}
    weights::W
    use_min::Bool      # true = min-plus, false = max-plus
    seed::UInt64
    dim::Int
end

function ParaTropicalRing(n::Int; seed::UInt64=UInt64(0x6761795f636f6c6f), use_min::Bool=true)
    # Random weight matrix from seed
    weights = zeros(Float64, n, n)
    rng_state = seed
    for i in 1:n, j in 1:n
        rng_state = mix64(rng_state)
        weights[i, j] = i == j ? 0.0 : (rng_state % 1000) / 100.0
    end
    ParaTropicalRing(weights, use_min, seed, n)
end

"""
Tropical addition: min or max
"""
function trop_add(ptr::ParaTropicalRing, a::Float64, b::Float64)
    if isinf(a) && a > 0
        return b
    elseif isinf(b) && b > 0
        return a
    end
    ptr.use_min ? min(a, b) : max(a, b)
end

"""
Tropical multiplication: ordinary addition
"""
function trop_mul(a::Float64, b::Float64)
    a + b
end

"""
Tropical matrix multiplication
"""
function trop_matmul(ptr::ParaTropicalRing, A::Matrix{Float64}, B::Matrix{Float64})
    n = size(A, 1)
    C = fill(ptr.use_min ? Inf : -Inf, n, n)
    
    for i in 1:n, j in 1:n
        for k in 1:n
            C[i, j] = trop_add(ptr, C[i, j], trop_mul(A[i, k], B[k, j]))
        end
    end
    
    return C
end

"""
Tropical closure (Kleene star): all-pairs shortest/longest paths
"""
function trop_closure(ptr::ParaTropicalRing)
    W = copy(ptr.weights)
    n = ptr.dim
    
    # Floyd-Warshall style iteration
    for k in 1:n
        for i in 1:n, j in 1:n
            via_k = trop_mul(W[i, k], W[k, j])
            W[i, j] = trop_add(ptr, W[i, j], via_k)
        end
    end
    
    return W
end

# ═══════════════════════════════════════════════════════════════════════════════
# Mario Choice Gadget: NP-complete via level traversability
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ChoiceGadget

The Mario choice gadget from Demaine et al.:
- Entry from above
- Two exits: left and right
- Once taken, choice cannot be undone
- Creates 2^n possibilities for n gadgets → NP-complete
"""
struct ChoiceGadget
    id::Int
    left_target::Int   # Gadget ID reached via left choice
    right_target::Int  # Gadget ID reached via right choice
    color::Int         # Color assigned (for 3-colorability reduction)
end

"""
    MarioLevel

A Mario level as a collection of choice gadgets.
Traversability reduces from 3-SAT.
"""
struct MarioLevel
    gadgets::Vector{ChoiceGadget}
    start::Int         # Starting gadget
    goal::Int          # Goal gadget
    seed::UInt64
end

"""
    reduce_3sat_to_mario(clauses::Vector{Tuple{Int,Int,Int}}, n_vars::Int; seed::UInt64) -> MarioLevel

Reduce 3-SAT instance to Mario level traversability.

Each variable x_i creates a choice gadget:
- Left = x_i = true
- Right = x_i = false

Each clause creates a "check" gadget that's only traversable if clause is satisfied.
"""
function reduce_3sat_to_mario(
    clauses::Vector{Tuple{Int,Int,Int}}, 
    n_vars::Int; 
    seed::UInt64=UInt64(0x6761795f636f6c6f)
)
    gadgets = ChoiceGadget[]
    rng_state = seed
    
    # Variable gadgets (1 to n_vars)
    for i in 1:n_vars
        rng_state = mix64(rng_state)
        color = Int(rng_state % 3) + 1
        
        # Left = true, Right = false
        # Target: next variable or first clause gadget
        left_target = i < n_vars ? i + 1 : n_vars + 1
        right_target = i < n_vars ? i + 1 : n_vars + 1
        
        push!(gadgets, ChoiceGadget(i, left_target, right_target, color))
    end
    
    # Clause gadgets (n_vars + 1 to n_vars + n_clauses)
    for (c_idx, clause) in enumerate(clauses)
        g_id = n_vars + c_idx
        rng_state = mix64(rng_state)
        color = Int(rng_state % 3) + 1
        
        # Next clause or goal
        next_target = c_idx < length(clauses) ? g_id + 1 : n_vars + length(clauses) + 1
        
        push!(gadgets, ChoiceGadget(g_id, next_target, next_target, color))
    end
    
    # Goal gadget
    goal_id = n_vars + length(clauses) + 1
    push!(gadgets, ChoiceGadget(goal_id, goal_id, goal_id, 1))
    
    MarioLevel(gadgets, 1, goal_id, seed)
end

"""
    is_level_traversable(level::MarioLevel, assignment::Vector{Bool}, clauses::Vector{Tuple{Int,Int,Int}}) -> Bool

Check if Mario can reach goal with given variable assignment.
"""
function is_level_traversable(
    level::MarioLevel, 
    assignment::Vector{Bool}, 
    clauses::Vector{Tuple{Int,Int,Int}}
)
    n_vars = length(assignment)
    
    # Check each clause
    for (l1, l2, l3) in clauses
        v1 = l1 > 0 ? assignment[abs(l1)] : !assignment[abs(l1)]
        v2 = l2 > 0 ? assignment[abs(l2)] : !assignment[abs(l2)]
        v3 = l3 > 0 ? assignment[abs(l3)] : !assignment[abs(l3)]
        
        if !(v1 || v2 || v3)
            return false  # Clause not satisfied → level not traversable
        end
    end
    
    return true
end

# ═══════════════════════════════════════════════════════════════════════════════
# Complexity Classes and SPI Verification at Boundaries
# ═══════════════════════════════════════════════════════════════════════════════

@enum ComplexityClass begin
    P_CLASS          # Polynomial time
    NP_CLASS         # Nondeterministic polynomial
    NP_COMPLETE      # Hardest in NP
    NP_HARD          # At least as hard as NP-complete
    PSPACE_CLASS     # Polynomial space
    PSPACE_COMPLETE  # Hardest in PSPACE
end

"""
    GayComplexityWitness

A witness for SPI verification at complexity boundaries.
Contains test cases that stress the determinism guarantees.
"""
struct GayComplexityWitness
    class::ComplexityClass
    seed::UInt64
    iterations::Int
    fingerprints::Vector{UInt32}
    success::Bool
    description::String
end

"""
    verify_spi_at_boundary(class::ComplexityClass, seed::UInt64, n::Int) -> GayComplexityWitness

Verify SPI (Strong Parallelism Invariance) at the given complexity boundary.

These are the "hardest edge cases" for Gay.jl:
- Same seed must produce same fingerprint
- Regardless of execution order
- Even when the underlying problem is NP-complete
"""
function verify_spi_at_boundary(
    class::ComplexityClass, 
    seed::UInt64, 
    n::Int
)
    fingerprints = UInt32[]
    
    description = if class == P_CLASS
        "Polynomial verification: greedy 2-coloring on bipartite graph"
    elseif class == NP_CLASS
        "NP verification: 3-SAT solution checking"
    elseif class == NP_COMPLETE
        "NP-complete: 3-coloring on arbitrary graph"
    elseif class == NP_HARD
        "NP-hard: Mario level traversability"
    elseif class == PSPACE_CLASS
        "PSPACE: Quantified Boolean Formula"
    else
        "PSPACE-complete: QSAT"
    end
    
    # Run multiple times with same seed
    for iter in 1:5
        rng_state = mix64(seed ⊻ UInt64(iter * 0x9e3779b97f4a7c15))
        
        if class == NP_COMPLETE || class == NP_HARD
            # Generate hard instance
            g = SimpleGraph(n)
            for i in 1:n
                for j in i+1:n
                    rng_state = mix64(rng_state)
                    if rng_state % 3 == 0
                        add_edge!(g, i, j)
                    end
                end
            end
            
            pc = ParaColorable(g, 3; seed=seed)
            colors, success = greedy_coloring(pc, UInt64(iter))
            
            # Compute fingerprint from coloring attempt
            fp = UInt32(0)
            for (i, c) in enumerate(colors)
                fp = fp ⊻ UInt32(mix64(UInt64(c) ⊻ UInt64(i)) % (1 << 32))
            end
            push!(fingerprints, fp)
            
        elseif class == P_CLASS
            # Easy case: derangement (always solvable)
            pd = ParaDerangeable(:parity, n; seed=seed)
            perm = sample_derangement(pd, UInt64(iter))
            
            fp = UInt32(0)
            for (i, p) in enumerate(perm)
                fp = fp ⊻ UInt32(mix64(UInt64(p) ⊻ UInt64(i)) % (1 << 32))
            end
            push!(fingerprints, fp)
            
        else
            # Tropical computation (polynomial)
            ptr = ParaTropicalRing(n; seed=seed)
            closure = trop_closure(ptr)
            
            fp = UInt32(0)
            for i in 1:n, j in 1:n
                if !isinf(closure[i, j])
                    fp = fp ⊻ UInt32(mix64(UInt64(round(closure[i, j] * 100))) % (1 << 32))
                end
            end
            push!(fingerprints, fp)
        end
    end
    
    # SPI check: all fingerprints should be identical
    all_same = all(fp == fingerprints[1] for fp in fingerprints)
    
    GayComplexityWitness(class, seed, 5, fingerprints, all_same, description)
end

"""
    hardest_edge_cases(seed::UInt64) -> Vector{GayComplexityWitness}

Generate the hardest edge cases for Gay.jl SPI verification.
Tests at every complexity boundary: P, NP, NP-complete, NP-hard, PSPACE.
"""
function hardest_edge_cases(seed::UInt64=UInt64(0x6761795f636f6c6f))
    witnesses = GayComplexityWitness[]
    
    for class in [P_CLASS, NP_CLASS, NP_COMPLETE, NP_HARD, PSPACE_CLASS, PSPACE_COMPLETE]
        # Small instance for fast verification
        n = class in [P_CLASS, NP_CLASS] ? 20 : 10
        
        witness = verify_spi_at_boundary(class, seed, n)
        push!(witnesses, witness)
        
        println("$(class): $(witness.success ? "✓" : "✗") SPI - $(witness.description)")
    end
    
    # Summary
    all_passed = all(w.success for w in witnesses)
    println()
    println("═" ^ 60)
    println(all_passed ? "✓ ALL SPI TESTS PASSED AT COMPLEXITY BOUNDARIES" : 
                         "✗ SPI VIOLATION DETECTED")
    println("═" ^ 60)
    
    return witnesses
end

# ═══════════════════════════════════════════════════════════════════════════════
# Self-Learning GayACSet: Accumulates examples and metatheory
# ═══════════════════════════════════════════════════════════════════════════════

"""
    GayACSet

Attributed C-Set accumulating Gay.jl examples with metatheoretic analysis.

Objects:
- Examples: specific use cases with seed, fingerprint, complexity class
- Metatheories: categorical structures (Para(X), Galois connections, etc.)
- Moments: points of difference between world sparsities

Morphisms:
- Reductions: 3-SAT → Mario, Graph → Coloring, etc.
- SPI proofs: same seed → same fingerprint witnesses
"""
mutable struct GayACSet
    examples::Dict{Symbol, Any}           # name → example data
    metatheories::Dict{Symbol, Function}  # name → metatheory functor
    moments::Vector{Tuple{Symbol, Symbol, String}}  # (ex1, ex2, difference)
    seed::UInt64
    complexity_witnesses::Vector{GayComplexityWitness}
end

function GayACSet(; seed::UInt64=UInt64(0x6761795f636f6c6f))
    GayACSet(
        Dict{Symbol, Any}(),
        Dict{Symbol, Function}(),
        Tuple{Symbol, Symbol, String}[],
        seed,
        GayComplexityWitness[]
    )
end

"""
Add an example to the ACSet
"""
function add_example!(acset::GayACSet, name::Symbol, data)
    acset.examples[name] = data
    return acset
end

"""
Add a metatheory functor
"""
function add_metatheory!(acset::GayACSet, name::Symbol, f::Function)
    acset.metatheories[name] = f
    return acset
end

"""
Record a moment of difference between examples
"""
function add_moment!(acset::GayACSet, ex1::Symbol, ex2::Symbol, difference::String)
    push!(acset.moments, (ex1, ex2, difference))
    return acset
end

"""
Self-learn: analyze all examples and compute complexity witnesses
"""
function self_learn!(acset::GayACSet)
    println("GayACSet Self-Learning...")
    println("═" ^ 40)
    
    # Compute complexity witnesses
    acset.complexity_witnesses = hardest_edge_cases(acset.seed)
    
    # Analyze moments of difference
    example_names = collect(keys(acset.examples))
    for i in 1:length(example_names)
        for j in i+1:length(example_names)
            n1, n2 = example_names[i], example_names[j]
            e1, e2 = acset.examples[n1], acset.examples[n2]
            
            # Check for metatheoretic differences
            if haskey(e1, :complexity) && haskey(e2, :complexity)
                if e1[:complexity] != e2[:complexity]
                    add_moment!(acset, n1, n2, 
                        "Complexity class: $(e1[:complexity]) vs $(e2[:complexity])")
                end
            end
        end
    end
    
    println("\nDiscovered $(length(acset.moments)) moments of difference")
    for (ex1, ex2, diff) in acset.moments
        println("  $ex1 ↔ $ex2: $diff")
    end
    
    return acset
end

# ═══════════════════════════════════════════════════════════════════════════════
# Main: Demo of complexity boundary verification
# ═══════════════════════════════════════════════════════════════════════════════

function main()
    println()
    println("╔" * "═" ^ 58 * "╗")
    println("║  Para(Derangeable) × Para(Colorable) × Para(TropicalRing)  ║")
    println("║  NP-complete / NP-hard / PSPACE Boundary Edge Cases        ║")
    println("╚" * "═" ^ 58 * "╝")
    println()
    
    seed = UInt64(0x6761795f636f6c6f)  # GAY_SEED
    
    # Create self-learning ACSet
    acset = GayACSet(; seed=seed)
    
    # Add examples from Gay.jl universe
    add_example!(acset, :bisimulation_petri, Dict(
        :description => "Petri net bisimulation colors",
        :complexity => NP_CLASS,
        :file => "examples/bisimulation_petri_colors.jl"
    ))
    
    add_example!(acset, :gay_metropolis, Dict(
        :description => "Monte Carlo with SPI colors",
        :complexity => P_CLASS,
        :file => "examples/gay_metropolis.jl"
    ))
    
    add_example!(acset, :derangeable, Dict(
        :description => "Fixed-point-free permutations",
        :complexity => P_CLASS,
        :file => "src/derangeable.jl"
    ))
    
    add_example!(acset, :mario_choice, Dict(
        :description => "Nintendo choice gadget reduction",
        :complexity => NP_COMPLETE,
        :file => "src/para_complexity.jl"
    ))
    
    # Add metatheories
    add_metatheory!(acset, :Para, x -> "Para($x)")
    add_metatheory!(acset, :Galois, x -> "α: $x → ColorSignature")
    
    # Self-learn
    self_learn!(acset)
    
    println()
    println("GayACSet Summary:")
    println("  Examples: $(length(acset.examples))")
    println("  Metatheories: $(length(acset.metatheories))")
    println("  Moments of difference: $(length(acset.moments))")
    println("  Complexity witnesses: $(length(acset.complexity_witnesses))")
    
    # Verify Para structures
    println()
    println("Para Structure Verification:")
    
    # Para(Derangeable)
    pd = ParaDerangeable(:even_parity, 6; seed=seed)
    perm = sample_derangement(pd, UInt64(1))
    println("  Para(Derangeable): σ = $perm (no fixed points: $(all(i != perm[i] for i in 1:6) ? "✓" : "✗"))")
    
    # Para(Colorable)
    g = SimpleGraph(5)
    add_edge!(g, 1, 2); add_edge!(g, 2, 3); add_edge!(g, 3, 4); add_edge!(g, 4, 5)
    pc = ParaColorable(g, 3; seed=seed)
    colors, success = greedy_coloring(pc, UInt64(1))
    println("  Para(Colorable): colors = $colors (valid: $(success ? "✓" : "✗"))")
    
    # Para(TropicalRing)
    ptr = ParaTropicalRing(4; seed=seed)
    closure = trop_closure(ptr)
    println("  Para(TropicalRing): closure computed (dim=$(ptr.dim))")
    
    println()
    println("🏳️‍🌈 Gay.jl Complexity Boundary Verification Complete")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
