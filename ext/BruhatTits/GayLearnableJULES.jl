"""
GayLearnableJULES.jl - JULES Partition: Learnable Color Spaces + 3-MATCH 3-Col

The JULES (⊕ Coproduct Order) partition in the 3-partite Bruhat-Tits tree saturation.

Implements:
  1. GayLearnableColorSpace - Gradient-learnable gamut boundaries
  2. GayMetalearnableColorSpace - Meta-learning over color space families (MAML/Reptile)
  3. 3-MATCH 3-Col Optimization - XOR tritwise gadgets for 3-coloring

2-Categorical Structure:
  ColorSpace ────────→ LearnableColorSpace ────────→ MetaLearnableColorSpace
      │                        │                              │
      │     (gradient)         │     (meta-gradient)          │
      └────────────────────────┴──────────────────────────────┘

Tao-style Triangle Inequality Estimates:
  d(c₁,c₃) ≤ d(c₁,c₂) + d(c₂,c₃) + O(ε²)  where ε = gamut_curvature
  
Reference: Tao's L^p spaces notes on metric completion and triangle inequality bounds.
"""
module GayLearnableJULES

export GayLearnableColorSpace, GayMetalearnableColorSpace
export ThreeMatchSolver, TritwiseGadget, ColorSpaceTask
export GamutBounds, MetricWeights
export learn_gamut!, meta_adapt!, verify_triangle_inequality
export solve_3match, random_walk_sat, tritwise_xor, jules_saturate

using LinearAlgebra
using Random

# ============================================================================
# CONSTANTS - JULES Partition
# ============================================================================

const JULES_SEED = UInt64(0x4A554C4553)       # "JULES" - coproduct order ⊕
const ZAHN_SEED = UInt64(0x5A41484E)          # "ZAHN"  - tensor order ⊗
const FABRIZ_SEED = UInt64(0x464142524947)    # "FABRIZ" - convolution order ⊛

const TAO_EPSILON = 1e-7  # Tao-style error bound for metric estimates
const GAMUT_LEARNING_RATE = 0.01
const META_LEARNING_RATE = 0.001

# ============================================================================
# SPLITMIX64 - Core PRNG (consistent with Gay.jl)
# ============================================================================

@inline function splitmix64(state::UInt64)::Tuple{UInt64, UInt64}
    z = state + 0x9E3779B97F4A7C15
    z = (z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ⊻ (z >> 27)) * 0x94D049BB133111EB
    (z ⊻ (z >> 31), state + 0x9E3779B97F4A7C15)
end

@inline function rand_float(state::UInt64)::Tuple{Float64, UInt64}
    val, next = splitmix64(state)
    Float64(val) / Float64(typemax(UInt64)), next
end

# ============================================================================
# PART 1: GayLearnableColorSpace
# ============================================================================

"""
Learnable gamut boundary representation.

Uses gradient descent to optimize:
  - gamut_bounds: (L_min, L_max, C_max, H_shift)
  - metric_weights: Weights for perceptual distance computation
  - curvature: Local gamut curvature affecting triangle inequality bounds
"""
mutable struct GamutBounds
    L_min::Float64
    L_max::Float64
    C_max::Float64
    H_shift::Float64
    curvature::Float64  # Affects Tao ε² correction term
end

function GamutBounds()
    GamutBounds(0.0, 1.0, 0.4, 0.0, 0.0)  # OkLab defaults
end

"""
Perceptual metric weights for color distance.
Triangle inequality: d(a,c) ≤ d(a,b) + d(b,c) + O(curvature²)
"""
mutable struct MetricWeights
    w_L::Float64  # Lightness weight
    w_C::Float64  # Chroma weight
    w_H::Float64  # Hue weight
end

MetricWeights() = MetricWeights(1.0, 1.0, 1.0)

"""
GayLearnableColorSpace - A color space with learnable parameters.

Subsubagent 1 in JULES partition:
  - Learnable gamut boundaries via gradient descent
  - Triangle inequality constraints for metric color spaces
  - Tao-style estimates for color distance bounds
"""
mutable struct GayLearnableColorSpace
    name::String
    seed::UInt64
    bounds::GamutBounds
    weights::MetricWeights
    fingerprint::UInt64
    
    # Gradient accumulators
    grad_bounds::Vector{Float64}  # [∂L_min, ∂L_max, ∂C_max, ∂H_shift, ∂curvature]
    grad_weights::Vector{Float64}  # [∂w_L, ∂w_C, ∂w_H]
    
    # Training state
    step_count::Int
    learning_rate::Float64
    triangle_violations::Int
end

function GayLearnableColorSpace(name::String; seed=JULES_SEED)
    val, _ = splitmix64(seed)
    GayLearnableColorSpace(
        name, seed,
        GamutBounds(), MetricWeights(),
        val,
        zeros(5), zeros(3),
        0, GAMUT_LEARNING_RATE, 0
    )
end

"""
Compute perceptual color distance with Tao-style correction.

d(c₁, c₂) = √(w_L·ΔL² + w_C·ΔC² + w_H·ΔH²) + O(ε²)

where ε = curvature and the O(ε²) term ensures triangle inequality holds
even in curved gamut regions.
"""
function color_distance(
    cs::GayLearnableColorSpace,
    c1::NTuple{3, Float64},  # (L, C, H) in OkLCh
    c2::NTuple{3, Float64}
)::Float64
    ΔL = c1[1] - c2[1]
    ΔC = c1[2] - c2[2]
    
    # Hue difference with wrap-around
    ΔH = c1[3] - c2[3]
    ΔH = mod(ΔH + 180.0, 360.0) - 180.0  # Map to [-180, 180]
    ΔH = ΔH * π / 180.0  # Convert to radians for computation
    
    # Weighted distance
    base_dist = sqrt(
        cs.weights.w_L * ΔL^2 + 
        cs.weights.w_C * ΔC^2 + 
        cs.weights.w_H * ΔH^2
    )
    
    # Tao correction term: O(ε²) where ε = curvature
    tao_correction = cs.bounds.curvature^2 * TAO_EPSILON
    
    base_dist + tao_correction
end

"""
Verify triangle inequality: d(a,c) ≤ d(a,b) + d(b,c)

Returns (holds, violation_amount) where violation_amount is the
amount by which the inequality is violated (0 if it holds).
"""
function verify_triangle_inequality(
    cs::GayLearnableColorSpace,
    a::NTuple{3, Float64},
    b::NTuple{3, Float64},
    c::NTuple{3, Float64}
)::Tuple{Bool, Float64}
    d_ac = color_distance(cs, a, c)
    d_ab = color_distance(cs, a, b)
    d_bc = color_distance(cs, b, c)
    
    # Tao bound: allow O(ε²) slack
    tao_slack = 2 * cs.bounds.curvature^2 * TAO_EPSILON
    
    violation = d_ac - (d_ab + d_bc) - tao_slack
    (violation <= 0, max(0.0, violation))
end

"""
Compute gradient of gamut loss for boundary learning.

Loss = Σᵢ max(0, cᵢ - bound)² + λ·triangle_violation²
"""
function compute_gamut_gradient!(
    cs::GayLearnableColorSpace,
    colors::Vector{NTuple{3, Float64}}
)
    fill!(cs.grad_bounds, 0.0)
    fill!(cs.grad_weights, 0.0)
    
    n = length(colors)
    for (L, C, H) in colors
        # Boundary violations
        if L < cs.bounds.L_min
            cs.grad_bounds[1] += 2(L - cs.bounds.L_min)  # ∂/∂L_min
        end
        if L > cs.bounds.L_max
            cs.grad_bounds[2] += 2(L - cs.bounds.L_max)  # ∂/∂L_max
        end
        if C > cs.bounds.C_max
            cs.grad_bounds[3] += 2(C - cs.bounds.C_max)  # ∂/∂C_max
        end
    end
    
    # Triangle inequality regularization
    if n >= 3
        # Sample random triplets
        for _ in 1:min(100, n)
            i, j, k = rand(1:n), rand(1:n), rand(1:n)
            if i != j && j != k && i != k
                holds, violation = verify_triangle_inequality(cs, colors[i], colors[j], colors[k])
                if !holds
                    cs.triangle_violations += 1
                    # Curvature gradient: increasing curvature increases Tao slack
                    cs.grad_bounds[5] -= 4 * cs.bounds.curvature * TAO_EPSILON * violation
                end
            end
        end
    end
    
    # Normalize
    cs.grad_bounds ./= n
end

"""
Update learnable parameters via gradient descent.
"""
function learn_gamut!(
    cs::GayLearnableColorSpace,
    colors::Vector{NTuple{3, Float64}};
    epochs::Int = 100
)
    for epoch in 1:epochs
        compute_gamut_gradient!(cs, colors)
        
        # SGD update
        cs.bounds.L_min -= cs.learning_rate * cs.grad_bounds[1]
        cs.bounds.L_max -= cs.learning_rate * cs.grad_bounds[2]
        cs.bounds.C_max -= cs.learning_rate * cs.grad_bounds[3]
        cs.bounds.H_shift -= cs.learning_rate * cs.grad_bounds[4]
        cs.bounds.curvature -= cs.learning_rate * cs.grad_bounds[5]
        
        # Clamp to valid ranges
        cs.bounds.L_min = clamp(cs.bounds.L_min, 0.0, 0.5)
        cs.bounds.L_max = clamp(cs.bounds.L_max, 0.5, 1.0)
        cs.bounds.C_max = clamp(cs.bounds.C_max, 0.1, 0.5)
        cs.bounds.curvature = clamp(cs.bounds.curvature, 0.0, 1.0)
        
        cs.step_count += 1
        
        # Update fingerprint with XOR composition
        cs.fingerprint ⊻= splitmix64(UInt64(cs.step_count))[1]
    end
    cs
end

# ============================================================================
# PART 2: GayMetalearnableColorSpace (MAML/Reptile-style)
# ============================================================================

"""
Task representation for meta-learning.
A task is a color space adaptation problem.
"""
struct ColorSpaceTask
    name::String
    support_colors::Vector{NTuple{3, Float64}}  # Training colors
    query_colors::Vector{NTuple{3, Float64}}    # Evaluation colors
    target_gamut::GamutBounds
end

"""
GayMetalearnableColorSpace - Meta-learning over color space families.

Subsubagent 2 in JULES partition:
  - MAML/Reptile-style adaptation for gamut transfer
  - 2-categorical structure: ColorSpace → LearnableColorSpace → MetaLearnableColorSpace
  - Fast adaptation to new color space families
"""
mutable struct GayMetalearnableColorSpace
    name::String
    seed::UInt64
    
    # Meta-parameters (θ)
    meta_bounds::GamutBounds
    meta_weights::MetricWeights
    
    # Inner loop learning rate (α)
    inner_lr::Float64
    # Outer loop learning rate (β)  
    outer_lr::Float64
    
    # Adaptation history
    task_fingerprints::Vector{UInt64}
    
    # 2-categorical morphism tracking
    # Objects: ColorSpace, LearnableColorSpace, MetaLearnableColorSpace
    # 1-morphisms: learn, meta_adapt
    # 2-morphisms: natural transformations between learning strategies
    level::Int  # 0=ColorSpace, 1=Learnable, 2=MetaLearnable
    
    # Meta-gradient accumulators
    meta_grad_bounds::Vector{Float64}
    meta_grad_weights::Vector{Float64}
end

function GayMetalearnableColorSpace(name::String; seed=JULES_SEED)
    val, _ = splitmix64(seed)
    GayMetalearnableColorSpace(
        name, seed,
        GamutBounds(), MetricWeights(),
        0.1,   # inner_lr (α) - larger for fast adaptation
        META_LEARNING_RATE,  # outer_lr (β)
        UInt64[],
        2,  # Level 2 in 2-category
        zeros(5), zeros(3)
    )
end

"""
Inner loop: Adapt to a single task (MAML inner update).

θ' = θ - α∇ₜL(θ)
"""
function inner_adapt(
    meta::GayMetalearnableColorSpace,
    task::ColorSpaceTask;
    steps::Int = 5
)::GayLearnableColorSpace
    # Clone meta-parameters to task-specific learner
    cs = GayLearnableColorSpace(task.name; seed=meta.seed)
    cs.bounds = deepcopy(meta.meta_bounds)
    cs.weights = deepcopy(meta.meta_weights)
    cs.learning_rate = meta.inner_lr
    
    # K steps of gradient descent on support set
    learn_gamut!(cs, task.support_colors; epochs=steps)
    
    cs
end

"""
Outer loop: MAML-style meta-gradient computation.

∇θ Σₜ L(θ - α∇ₜL(θ)) = Σₜ (I - α∇²ₜL(θ)) ∇L(θ')

For Reptile (first-order approximation):
θ ← θ + β(θ' - θ)
"""
function meta_adapt!(
    meta::GayMetalearnableColorSpace,
    tasks::Vector{ColorSpaceTask};
    inner_steps::Int = 5,
    outer_epochs::Int = 10,
    use_reptile::Bool = true  # Reptile is simpler, no Hessian
)
    fill!(meta.meta_grad_bounds, 0.0)
    fill!(meta.meta_grad_weights, 0.0)
    
    for epoch in 1:outer_epochs
        for task in tasks
            # Inner loop adaptation
            adapted = inner_adapt(meta, task; steps=inner_steps)
            
            if use_reptile
                # Reptile: move toward adapted parameters
                meta.meta_bounds.L_min += meta.outer_lr * (adapted.bounds.L_min - meta.meta_bounds.L_min)
                meta.meta_bounds.L_max += meta.outer_lr * (adapted.bounds.L_max - meta.meta_bounds.L_max)
                meta.meta_bounds.C_max += meta.outer_lr * (adapted.bounds.C_max - meta.meta_bounds.C_max)
                meta.meta_bounds.H_shift += meta.outer_lr * (adapted.bounds.H_shift - meta.meta_bounds.H_shift)
                meta.meta_bounds.curvature += meta.outer_lr * (adapted.bounds.curvature - meta.meta_bounds.curvature)
                
                meta.meta_weights.w_L += meta.outer_lr * (adapted.weights.w_L - meta.meta_weights.w_L)
                meta.meta_weights.w_C += meta.outer_lr * (adapted.weights.w_C - meta.meta_weights.w_C)
                meta.meta_weights.w_H += meta.outer_lr * (adapted.weights.w_H - meta.meta_weights.w_H)
            else
                # MAML: would require second-order gradients (expensive)
                # Accumulate query loss gradients after adaptation
                compute_gamut_gradient!(adapted, task.query_colors)
                meta.meta_grad_bounds .+= adapted.grad_bounds ./ length(tasks)
                meta.meta_grad_weights .+= adapted.grad_weights ./ length(tasks)
            end
            
            # XOR fingerprint composition
            push!(meta.task_fingerprints, adapted.fingerprint)
        end
        
        if !use_reptile
            # MAML outer update
            meta.meta_bounds.L_min -= meta.outer_lr * meta.meta_grad_bounds[1]
            meta.meta_bounds.L_max -= meta.outer_lr * meta.meta_grad_bounds[2]
            meta.meta_bounds.C_max -= meta.outer_lr * meta.meta_grad_bounds[3]
            meta.meta_bounds.H_shift -= meta.outer_lr * meta.meta_grad_bounds[4]
            meta.meta_bounds.curvature -= meta.outer_lr * meta.meta_grad_bounds[5]
        end
    end
    
    meta
end

"""
2-categorical projection: MetaLearnableColorSpace → LearnableColorSpace → ColorSpace
"""
function project_to_level(meta::GayMetalearnableColorSpace, level::Int)
    @assert 0 <= level <= 2 "Level must be 0, 1, or 2"
    
    if level == 2
        return meta
    elseif level == 1
        cs = GayLearnableColorSpace(meta.name; seed=meta.seed)
        cs.bounds = deepcopy(meta.meta_bounds)
        cs.weights = deepcopy(meta.meta_weights)
        return cs
    else
        # Frozen ColorSpace - just the parameters, no learning
        return (bounds=meta.meta_bounds, weights=meta.meta_weights)
    end
end

# ============================================================================
# PART 3: 3-MATCH 3-Col Optimization
# ============================================================================

"""
Tritwise XOR gadget for 3-coloring constraint encoding.

XOR over GF(3): a ⊕₃ b = (a + b) mod 3
"""
struct TritwiseGadget
    trits::Vector{UInt8}  # Values in {0, 1, 2}
    fingerprint::UInt64
end

function TritwiseGadget(n::Int; seed=JULES_SEED)
    state = UInt64(seed)
    trits = UInt8[]
    for _ in 1:n
        val, state = splitmix64(state)
        push!(trits, val % 3)
    end
    TritwiseGadget(trits, UInt64(seed))
end

"""
XOR two tritwise gadgets (GF(3) addition).
"""
function tritwise_xor(a::TritwiseGadget, b::TritwiseGadget)::TritwiseGadget
    @assert length(a.trits) == length(b.trits)
    new_trits = UInt8[(a.trits[i] + b.trits[i]) % 3 for i in eachindex(a.trits)]
    TritwiseGadget(new_trits, a.fingerprint ⊻ b.fingerprint)
end

"""
3-MATCH clause: Three elements must have distinct colors (0, 1, 2).
Reduces to 3-SAT via gadget construction.
"""
struct ThreeMatchClause
    indices::NTuple{3, Int}  # Indices of three variables
end

"""
3-MATCH 3-Col Solver.

Subsubagent 3 in JULES partition:
  - XOR tritwise gadgets for 3-coloring problems
  - 3-MATCH reduction to 3-SAT for NP-completeness
  - Parallel random walk satisfiability checking
"""
mutable struct ThreeMatchSolver
    n_vars::Int
    clauses::Vector{ThreeMatchClause}
    assignment::Vector{UInt8}  # Current coloring {0, 1, 2}
    fingerprint::UInt64
    
    # Random walk state
    walk_seed::UInt64
    n_walks::Int
    steps_per_walk::Int
    
    # Solution tracking
    satisfied_count::Int
    best_assignment::Vector{UInt8}
    best_satisfied::Int
end

function ThreeMatchSolver(n_vars::Int, clauses::Vector{ThreeMatchClause}; 
                          seed=JULES_SEED, n_walks=23)
    assignment = rand(0:2, n_vars) .|> UInt8
    ThreeMatchSolver(
        n_vars, clauses, assignment, seed,
        seed, n_walks, 3 * n_vars,
        0, copy(assignment), 0
    )
end

"""
Check if a clause is satisfied (all three have distinct colors).
"""
function is_satisfied(clause::ThreeMatchClause, assignment::Vector{UInt8})::Bool
    c1 = assignment[clause.indices[1]]
    c2 = assignment[clause.indices[2]]
    c3 = assignment[clause.indices[3]]
    c1 != c2 && c2 != c3 && c1 != c3
end

"""
Count satisfied clauses.
"""
function count_satisfied(solver::ThreeMatchSolver)::Int
    count(c -> is_satisfied(c, solver.assignment), solver.clauses)
end

"""
Random walk step: flip a random variable's color to satisfy a random unsatisfied clause.
"""
function random_walk_step!(solver::ThreeMatchSolver)::Bool
    # Find unsatisfied clauses
    unsat = filter(c -> !is_satisfied(c, solver.assignment), solver.clauses)
    
    if isempty(unsat)
        return true  # All satisfied!
    end
    
    # Pick random unsatisfied clause
    val, solver.walk_seed = splitmix64(solver.walk_seed)
    clause = unsat[(val % length(unsat)) + 1]
    
    # Pick random variable in clause
    val, solver.walk_seed = splitmix64(solver.walk_seed)
    var_idx = clause.indices[(val % 3) + 1]
    
    # Flip to a random different color
    current = solver.assignment[var_idx]
    val, solver.walk_seed = splitmix64(solver.walk_seed)
    new_color = (current + 1 + (val % 2)) % 3
    solver.assignment[var_idx] = new_color
    
    false
end

"""
Parallel random walk SAT solving.

Runs n_walks independent random walks in parallel (simulated via loop).
Each walk: randomly flip colors to satisfy clauses.
"""
function random_walk_sat!(solver::ThreeMatchSolver; max_restarts::Int = 100)::Bool
    for restart in 1:max_restarts
        # Initialize with XOR fingerprint
        val, solver.walk_seed = splitmix64(solver.walk_seed ⊻ UInt64(restart))
        solver.assignment = UInt8[(val >> (i % 64)) % 3 for i in 1:solver.n_vars]
        
        # Parallel walks (simulated)
        for walk_id in 1:solver.n_walks
            walk_seed = solver.walk_seed ⊻ UInt64(walk_id * 0x5A41484E)  # ZAHN mixing
            
            for step in 1:solver.steps_per_walk
                if random_walk_step!(solver)
                    solver.fingerprint ⊻= solver.walk_seed
                    return true
                end
                
                # Track best
                sat = count_satisfied(solver)
                if sat > solver.best_satisfied
                    solver.best_satisfied = sat
                    solver.best_assignment = copy(solver.assignment)
                end
            end
        end
    end
    
    # Use best found
    solver.assignment = solver.best_assignment
    solver.satisfied_count = solver.best_satisfied
    false
end

"""
Solve 3-MATCH 3-Col instance.
"""
function solve_3match(n_vars::Int, clause_tuples::Vector{NTuple{3, Int}}; 
                      seed=JULES_SEED)::Tuple{Bool, Vector{UInt8}}
    clauses = [ThreeMatchClause(c) for c in clause_tuples]
    solver = ThreeMatchSolver(n_vars, clauses; seed=seed)
    
    success = random_walk_sat!(solver)
    (success, solver.assignment)
end

"""
Reduce 3-MATCH to 3-SAT (for NP-completeness proof).

Each 3-MATCH clause (a,b,c must be distinct) becomes:
  (x_a ≠ x_b) ∧ (x_b ≠ x_c) ∧ (x_a ≠ x_c)
  
Which expands to:
  (x_a=0 → x_b≠0) ∧ (x_a=1 → x_b≠1) ∧ (x_a=2 → x_b≠2) ∧ ...
"""
function reduce_to_3sat(clauses::Vector{ThreeMatchClause}, n_vars::Int)
    # Create Boolean variables: x[i,c] = true iff variable i has color c
    # 3n Boolean variables total
    
    sat_clauses = Vector{NTuple{3, Tuple{Int, Bool}}}()  # (var, negated)
    
    for clause in clauses
        a, b, c = clause.indices
        
        # For each pair (a,b), (b,c), (a,c):
        # They must differ, so: ¬(both same color)
        # = ¬(x_a0 ∧ x_b0) ∧ ¬(x_a1 ∧ x_b1) ∧ ¬(x_a2 ∧ x_b2)
        # = (¬x_a0 ∨ ¬x_b0) ∧ (¬x_a1 ∨ ¬x_b1) ∧ (¬x_a2 ∨ ¬x_b2)
        
        for color in 0:2
            for (i, j) in [(a, b), (b, c), (a, c)]
                var_i = (i - 1) * 3 + color + 1
                var_j = (j - 1) * 3 + color + 1
                # Add clause (¬x_ic ∨ ¬x_jc ∨ ¬x_ic) - dummy third literal
                push!(sat_clauses, ((var_i, true), (var_j, true), (var_i, true)))
            end
        end
    end
    
    sat_clauses
end

# ============================================================================
# COMBINED: JULES Partition Integration
# ============================================================================

"""
JULES saturation: combine learnable color spaces with 3-coloring optimization.

The coproduct order (⊕) allows free combination of constraints.
"""
function jules_saturate(
    colors::Vector{NTuple{3, Float64}},
    n_partitions::Int = 3;
    seed = JULES_SEED
)
    # 1. Learn optimal color space
    cs = GayLearnableColorSpace("JULES-⊕"; seed=seed)
    learn_gamut!(cs, colors)
    
    # 2. Construct 3-coloring problem from color clusters
    # Each color gets assigned to one of 3 partitions
    n = length(colors)
    clauses = NTuple{3, Int}[]
    
    # Generate random 3-cliques based on color distance
    state = seed
    for _ in 1:(n * 2)
        val, state = splitmix64(state)
        i = (val % n) + 1
        val, state = splitmix64(state)
        j = (val % n) + 1
        val, state = splitmix64(state)
        k = (val % n) + 1
        
        if i != j && j != k && i != k
            push!(clauses, (i, j, k))
        end
    end
    
    # 3. Solve 3-coloring
    success, assignment = solve_3match(n, clauses; seed=state)
    
    # 4. Verify triangle inequality on each color triple
    violations = 0
    for (i, j, k) in clauses
        holds, _ = verify_triangle_inequality(cs, colors[i], colors[j], colors[k])
        if !holds
            violations += 1
        end
    end
    
    (
        color_space = cs,
        coloring = assignment,
        satisfiable = success,
        triangle_violations = violations,
        fingerprint = cs.fingerprint ⊻ state
    )
end

# ============================================================================
# DISPLAY
# ============================================================================

function Base.show(io::IO, cs::GayLearnableColorSpace)
    println(io, "GayLearnableColorSpace: $(cs.name)")
    println(io, "  Gamut: L∈[$(round(cs.bounds.L_min,digits=3)),$(round(cs.bounds.L_max,digits=3))] C≤$(round(cs.bounds.C_max,digits=3))")
    println(io, "  Curvature: $(round(cs.bounds.curvature,digits=5)) (Tao ε²=$(round(cs.bounds.curvature^2 * TAO_EPSILON, sigdigits=3)))")
    println(io, "  Weights: L=$(round(cs.weights.w_L,digits=2)) C=$(round(cs.weights.w_C,digits=2)) H=$(round(cs.weights.w_H,digits=2))")
    println(io, "  Steps: $(cs.step_count), Triangle violations: $(cs.triangle_violations)")
    println(io, "  Fingerprint: 0x$(string(cs.fingerprint, base=16))")
end

function Base.show(io::IO, meta::GayMetalearnableColorSpace)
    println(io, "GayMetalearnableColorSpace: $(meta.name) [Level $(meta.level) in 2-Cat]")
    println(io, "  Meta-gamut: L∈[$(round(meta.meta_bounds.L_min,digits=3)),$(round(meta.meta_bounds.L_max,digits=3))]")
    println(io, "  Inner LR: $(meta.inner_lr), Outer LR: $(meta.outer_lr)")
    println(io, "  Tasks adapted: $(length(meta.task_fingerprints))")
    if !isempty(meta.task_fingerprints)
        global_fp = reduce(⊻, meta.task_fingerprints)
        println(io, "  Global fingerprint: 0x$(string(global_fp, base=16))")
    end
end

function Base.show(io::IO, solver::ThreeMatchSolver)
    println(io, "ThreeMatchSolver: $(solver.n_vars) vars, $(length(solver.clauses)) clauses")
    println(io, "  Best satisfied: $(solver.best_satisfied)/$(length(solver.clauses))")
    println(io, "  Fingerprint: 0x$(string(solver.fingerprint, base=16))")
end

end # module
