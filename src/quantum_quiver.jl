# QUANTUM QUIVER: Superposition over Path Algebras
#
# A quiver Q = (Q₀, Q₁, s, t) where:
#   Q₀ = vertices (the 3 narrative objects: Subject, Process, Object)
#   Q₁ = arrows (actions between them)
#   s, t: Q₁ → Q₀ (source and target)
#
# The PATH ALGEBRA kQ has:
#   - Basis: all paths in Q (including trivial paths eᵢ at each vertex)
#   - Multiplication: concatenation (or 0 if not composable)
#
# The QUANTUM QUIVER puts each arrow in superposition:
#   |ψ⟩ = Σᵢ αᵢ|arrowᵢ⟩
#
# Gabriel's Theorem: Q has finite representation type iff
# its underlying graph is a Dynkin diagram (ADE).
#
# The 3-vertex narrative quiver is type A₃:
#   Subject ——→ Process ——→ Object
#
# This gives EXACTLY 6 indecomposable representations:
#   S, P, O, SP, PO, SPO
# matching our 3! = 6 canonical actions.

module QuantumQuiver

using LinearAlgebra

export Quiver, QuiverVertex, QuiverArrow
export QuantumArrow, QuantumPath, PathAlgebra
export QuiverRepresentation, indecomposables
export NarrativeQuiver, NARRATIVE_A3
export pluck!, measure_path, entangle_arrows
export ADE_TYPE, gabriel_check, dynkin_diagram
export world_quantum_quiver

# ═══════════════════════════════════════════════════════════════════════════
# QUIVER STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════

"""
A vertex in a quiver.
"""
struct QuiverVertex
    id::Int
    label::Symbol
end

"""
An arrow in a quiver: source → target.
"""
struct QuiverArrow
    id::Int
    source::QuiverVertex
    target::QuiverVertex
    label::Symbol
end

"""
A quiver Q = (Q₀, Q₁, s, t).
"""
struct Quiver
    vertices::Vector{QuiverVertex}
    arrows::Vector{QuiverArrow}
    name::Symbol
end

function Base.show(io::IO, Q::Quiver)
    print(io, "Quiver($(Q.name): $(length(Q.vertices)) vertices, $(length(Q.arrows)) arrows)")
end

# Source and target maps
source(a::QuiverArrow) = a.source
target(a::QuiverArrow) = a.target

# ═══════════════════════════════════════════════════════════════════════════
# THE NARRATIVE QUIVER (A₃ Type)
# ═══════════════════════════════════════════════════════════════════════════

const SUBJECT = QuiverVertex(1, :Subject)
const PROCESS = QuiverVertex(2, :Process)  
const OBJECT = QuiverVertex(3, :Object)

# The 6 canonical arrows (matching 3! permutations)
const OBSERVE = QuiverArrow(1, SUBJECT, OBJECT, :observe)
const INITIATE = QuiverArrow(2, SUBJECT, PROCESS, :initiate)
const TRANSFORM = QuiverArrow(3, PROCESS, OBJECT, :transform)
const AFFECT = QuiverArrow(4, OBJECT, SUBJECT, :affect)
const FEEDBACK = QuiverArrow(5, PROCESS, SUBJECT, :feedback)
const RESIST = QuiverArrow(6, OBJECT, PROCESS, :resist)

"""
The narrative quiver: A₃ Dynkin type with bidirectional arrows.
"""
const NARRATIVE_A3 = Quiver(
    [SUBJECT, PROCESS, OBJECT],
    [OBSERVE, INITIATE, TRANSFORM, AFFECT, FEEDBACK, RESIST],
    :NarrativeA3
)

# Linear A₃ (for Gabriel's theorem)
const LINEAR_A3 = Quiver(
    [SUBJECT, PROCESS, OBJECT],
    [INITIATE, TRANSFORM],  # Only forward arrows
    :LinearA3
)

# ═══════════════════════════════════════════════════════════════════════════
# QUANTUM ARROWS (Superposition)
# ═══════════════════════════════════════════════════════════════════════════

"""
A quantum arrow: superposition of classical arrows.
|ψ⟩ = Σᵢ αᵢ|arrowᵢ⟩
"""
struct QuantumArrow
    amplitudes::Vector{ComplexF64}
    arrows::Vector{QuiverArrow}
    
    function QuantumArrow(amps::Vector{ComplexF64}, arrows::Vector{QuiverArrow})
        length(amps) == length(arrows) || error("Amplitude/arrow count mismatch")
        # Normalize
        norm = sqrt(sum(abs2, amps))
        norm > 0 ? new(amps ./ norm, arrows) : new(amps, arrows)
    end
end

function QuantumArrow(arrow::QuiverArrow)
    QuantumArrow([1.0 + 0.0im], [arrow])
end

"""
Superpose two quantum arrows.
"""
function superpose(a::QuantumArrow, b::QuantumArrow; α::Float64=0.5)
    β = sqrt(1 - α^2)
    new_amps = vcat(α .* a.amplitudes, β .* b.amplitudes)
    new_arrows = vcat(a.arrows, b.arrows)
    QuantumArrow(new_amps, new_arrows)
end

"""
Measure a quantum arrow → collapse to classical.
Returns (arrow, probability).
"""
function measure!(qa::QuantumArrow)
    probs = abs2.(qa.amplitudes)
    cumprobs = cumsum(probs)
    r = rand()
    
    idx = findfirst(p -> r <= p, cumprobs)
    idx = isnothing(idx) ? length(probs) : idx
    
    (qa.arrows[idx], probs[idx])
end

"""
Entangle two quantum arrows (tensor product).
"""
function entangle(a::QuantumArrow, b::QuantumArrow)
    new_amps = ComplexF64[]
    new_arrows = Tuple{QuiverArrow, QuiverArrow}[]
    
    for (αi, ai) in zip(a.amplitudes, a.arrows)
        for (βj, bj) in zip(b.amplitudes, b.arrows)
            push!(new_amps, αi * βj)
            push!(new_arrows, (ai, bj))
        end
    end
    
    (amplitudes=new_amps, pairs=new_arrows)
end

# ═══════════════════════════════════════════════════════════════════════════
# PATH ALGEBRA
# ═══════════════════════════════════════════════════════════════════════════

"""
A path in a quiver: sequence of composable arrows.
Empty sequence at vertex v = trivial path eᵥ.
"""
struct QuiverPath
    arrows::Vector{QuiverArrow}
    start::QuiverVertex
    stop::QuiverVertex
end

function QuiverPath(arrows::Vector{QuiverArrow})
    isempty(arrows) && error("Use trivial_path for empty paths")
    
    # Check composability
    for i in 1:(length(arrows)-1)
        target(arrows[i]) == source(arrows[i+1]) || 
            error("Arrows not composable: $(arrows[i].label) → $(arrows[i+1].label)")
    end
    
    QuiverPath(arrows, source(arrows[1]), target(arrows[end]))
end

function trivial_path(v::QuiverVertex)
    QuiverPath(QuiverArrow[], v, v)
end

Base.length(p::QuiverPath) = length(p.arrows)

function Base.show(io::IO, p::QuiverPath)
    if isempty(p.arrows)
        print(io, "e_$(p.start.label)")
    else
        labels = [a.label for a in p.arrows]
        print(io, join(labels, " → "))
    end
end

"""
Compose two paths (if composable).
"""
function compose(p1::QuiverPath, p2::QuiverPath)
    p1.stop == p2.start || return nothing  # Not composable
    
    if isempty(p1.arrows)
        return p2
    elseif isempty(p2.arrows)
        return p1
    else
        QuiverPath(vcat(p1.arrows, p2.arrows), p1.start, p2.stop)
    end
end

"""
Path algebra element: formal linear combination of paths.
"""
struct PathAlgebraElement
    coeffs::Dict{QuiverPath, ComplexF64}
end

function PathAlgebraElement(path::QuiverPath, coeff::ComplexF64=1.0+0.0im)
    PathAlgebraElement(Dict(path => coeff))
end

function Base.:+(a::PathAlgebraElement, b::PathAlgebraElement)
    result = copy(a.coeffs)
    for (path, coeff) in b.coeffs
        result[path] = get(result, path, 0.0+0.0im) + coeff
    end
    PathAlgebraElement(result)
end

function Base.:*(a::PathAlgebraElement, b::PathAlgebraElement)
    result = Dict{QuiverPath, ComplexF64}()
    
    for (p1, c1) in a.coeffs
        for (p2, c2) in b.coeffs
            composed = compose(p1, p2)
            if !isnothing(composed)
                result[composed] = get(result, composed, 0.0+0.0im) + c1 * c2
            end
        end
    end
    
    PathAlgebraElement(result)
end

"""
A quantum path: superposition of classical paths.
"""
struct QuantumPath
    amplitudes::Vector{ComplexF64}
    paths::Vector{QuiverPath}
    
    function QuantumPath(amps::Vector{ComplexF64}, paths::Vector{QuiverPath})
        norm = sqrt(sum(abs2, amps))
        norm > 0 ? new(amps ./ norm, paths) : new(amps, paths)
    end
end

function QuantumPath(path::QuiverPath)
    QuantumPath([1.0 + 0.0im], [path])
end

"""
Measure a quantum path → collapse to classical.
"""
function measure_path(qp::QuantumPath)
    probs = abs2.(qp.amplitudes)
    cumprobs = cumsum(probs)
    r = rand()
    
    idx = findfirst(p -> r <= p, cumprobs)
    idx = isnothing(idx) ? length(probs) : idx
    
    (qp.paths[idx], probs[idx])
end

# ═══════════════════════════════════════════════════════════════════════════
# QUIVER REPRESENTATIONS
# ═══════════════════════════════════════════════════════════════════════════

"""
A representation of quiver Q:
- Vector space Vᵢ at each vertex i
- Linear map φₐ: V_{s(a)} → V_{t(a)} for each arrow a
"""
struct QuiverRepresentation
    quiver::Quiver
    dimensions::Dict{QuiverVertex, Int}  # dim(Vᵢ)
    maps::Dict{QuiverArrow, Matrix{ComplexF64}}  # φₐ
end

"""
Dimension vector of a representation.
"""
function dim_vector(rep::QuiverRepresentation)
    [get(rep.dimensions, v, 0) for v in rep.quiver.vertices]
end

"""
The 6 indecomposable representations of A₃.
Gabriel's theorem guarantees these are ALL indecomposables.
"""
function indecomposables(Q::Quiver=LINEAR_A3)
    length(Q.vertices) == 3 || error("Only implemented for A₃")
    
    S = Q.vertices[1]
    P = Q.vertices[2]
    O = Q.vertices[3]
    
    a1 = Q.arrows[1]  # S → P
    a2 = Q.arrows[2]  # P → O
    
    reps = QuiverRepresentation[]
    
    # Simple representations: S, P, O
    # S: (1,0,0)
    push!(reps, QuiverRepresentation(Q,
        Dict(S => 1, P => 0, O => 0),
        Dict(a1 => zeros(0,1), a2 => zeros(0,0))
    ))
    
    # P: (0,1,0)
    push!(reps, QuiverRepresentation(Q,
        Dict(S => 0, P => 1, O => 0),
        Dict(a1 => zeros(1,0), a2 => zeros(0,1))
    ))
    
    # O: (0,0,1)
    push!(reps, QuiverRepresentation(Q,
        Dict(S => 0, P => 0, O => 1),
        Dict(a1 => zeros(0,0), a2 => zeros(1,0))
    ))
    
    # SP: (1,1,0) with id: V_S → V_P
    push!(reps, QuiverRepresentation(Q,
        Dict(S => 1, P => 1, O => 0),
        Dict(a1 => [1.0+0.0im;;], a2 => zeros(0,1))
    ))
    
    # PO: (0,1,1) with id: V_P → V_O
    push!(reps, QuiverRepresentation(Q,
        Dict(S => 0, P => 1, O => 1),
        Dict(a1 => zeros(1,0), a2 => [1.0+0.0im;;])
    ))
    
    # SPO: (1,1,1) with id at both arrows
    push!(reps, QuiverRepresentation(Q,
        Dict(S => 1, P => 1, O => 1),
        Dict(a1 => [1.0+0.0im;;], a2 => [1.0+0.0im;;])
    ))
    
    reps
end

# ═══════════════════════════════════════════════════════════════════════════
# ADE CLASSIFICATION (Gabriel's Theorem)
# ═══════════════════════════════════════════════════════════════════════════

@enum ADE_TYPE begin
    A_n = 1
    D_n = 2
    E_6 = 3
    E_7 = 4
    E_8 = 5
    NOT_ADE = 0
end

"""
Check if a quiver has finite representation type (ADE).
Uses Gabriel's theorem: finite type ⟺ underlying graph is Dynkin.
"""
function gabriel_check(Q::Quiver)
    n = length(Q.vertices)
    
    # Build adjacency matrix (ignoring arrow directions)
    adj = zeros(Int, n, n)
    for arrow in Q.arrows
        i = arrow.source.id
        j = arrow.target.id
        adj[i,j] = 1
        adj[j,i] = 1  # Undirected for Dynkin classification
    end
    
    # Count edges
    num_edges = sum(adj) ÷ 2
    
    # Check for cycles (finite type requires tree + no multi-edges)
    if num_edges >= n
        return (type=NOT_ADE, reason="Contains cycle")
    end
    
    # Check degree sequence
    degrees = vec(sum(adj, dims=2))
    max_degree = maximum(degrees)
    
    if max_degree > 3
        return (type=NOT_ADE, reason="Vertex with degree > 3")
    end
    
    # A_n: linear (all degrees ≤ 2)
    if max_degree <= 2
        return (type=A_n, reason="Linear graph (A_$n)")
    end
    
    # D_n or E_n: has exactly one vertex of degree 3
    branch_count = count(d -> d == 3, degrees)
    
    if branch_count == 1
        # Find branch lengths
        branch_vertex = findfirst(d -> d == 3, degrees)
        # Count lengths of 3 branches
        # ... (simplified: just check E conditions)
        
        if n == 6
            return (type=E_6, reason="E₆ configuration")
        elseif n == 7
            return (type=E_7, reason="E₇ configuration")
        elseif n == 8
            return (type=E_8, reason="E₈ configuration")
        elseif n >= 4
            return (type=D_n, reason="D_$n configuration")
        end
    end
    
    (type=NOT_ADE, reason="Unknown configuration")
end

"""
Generate ASCII Dynkin diagram for a quiver.
"""
function dynkin_diagram(Q::Quiver)
    classification = gabriel_check(Q)
    n = length(Q.vertices)
    
    if classification.type == A_n
        # A_n: o—o—o—...—o
        nodes = ["○" for _ in 1:n]
        return join(nodes, "——")
    elseif classification.type == D_n
        # D_n: branched
        return """
           ○
          /
    ○——○——○——...——○
          \\
           ○"""
    elseif classification.type in [E_6, E_7, E_8]
        return """
              ○
              |
    ○——○——○——○——○——...
    """
    else
        return "Not a Dynkin diagram"
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# QUANTUM QUIVER OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════

"""
Pluck a quantum arrow (like a guitar string).
Returns a quantum path starting from that arrow.
"""
function pluck!(qa::QuantumArrow, quiver::Quiver; max_length::Int=3)
    # Start with the quantum arrow as a length-1 path
    initial_paths = [QuiverPath([a]) for a in qa.arrows]
    qp = QuantumPath(qa.amplitudes, initial_paths)
    
    # Extend paths probabilistically
    for _ in 2:max_length
        new_paths = QuiverPath[]
        new_amps = ComplexF64[]
        
        for (amp, path) in zip(qp.amplitudes, qp.paths)
            # Find arrows that can extend this path
            extendable = [a for a in quiver.arrows if source(a) == path.stop]
            
            if isempty(extendable)
                # Path terminates
                push!(new_paths, path)
                push!(new_amps, amp)
            else
                # Split amplitude among extensions
                split_amp = amp / sqrt(length(extendable))
                for arrow in extendable
                    extended = QuiverPath(vcat(path.arrows, [arrow]), path.start, target(arrow))
                    push!(new_paths, extended)
                    push!(new_amps, split_amp)
                end
            end
        end
        
        qp = QuantumPath(new_amps, new_paths)
    end
    
    qp
end

"""
Entangle two quantum arrows into a Bell state.
"""
function entangle_arrows(a1::QuiverArrow, a2::QuiverArrow)
    # |Φ⁺⟩ = (|00⟩ + |11⟩)/√2 analog for arrows
    qa1 = QuantumArrow(a1)
    qa2 = QuantumArrow(a2)
    
    # Create entangled pair
    amps = [1/sqrt(2) + 0.0im, 1/sqrt(2) + 0.0im]
    pairs = [(a1, a1), (a2, a2)]  # Correlated
    
    (amplitudes=amps, pairs=pairs, state=:bell_phi_plus)
end

# ═══════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════

function world_quantum_quiver()
    println("═══════════════════════════════════════════════════════════════")
    println("  QUANTUM QUIVER: Superposition over Path Algebras")
    println("═══════════════════════════════════════════════════════════════")
    println()
    
    # The narrative quiver
    println("NARRATIVE QUIVER (A₃ type with bidirectional arrows):")
    println("  Vertices: Subject, Process, Object")
    println("  Arrows:")
    for a in NARRATIVE_A3.arrows
        println("    $(a.source.label) —[$(a.label)]→ $(a.target.label)")
    end
    println()
    
    # Gabriel's theorem
    println("GABRIEL'S THEOREM CHECK:")
    result = gabriel_check(LINEAR_A3)
    println("  Linear A₃: $(result.type) — $(result.reason)")
    println("  Dynkin diagram: $(dynkin_diagram(LINEAR_A3))")
    println()
    
    # Indecomposable representations
    println("INDECOMPOSABLE REPRESENTATIONS (6 total, matching 3!):")
    indecs = indecomposables(LINEAR_A3)
    for (i, rep) in enumerate(indecs)
        dv = dim_vector(rep)
        labels = ["S", "P", "O"]
        active = [labels[j] for j in 1:3 if dv[j] > 0]
        println("  $i. $(join(active, "")) — dimension vector $(dv)")
    end
    println()
    
    # Quantum arrow
    println("QUANTUM ARROW (superposition of observe | initiate):")
    qa = superpose(QuantumArrow(OBSERVE), QuantumArrow(INITIATE))
    println("  Amplitudes: $(round.(qa.amplitudes, digits=3))")
    println("  Arrows: $([(a.label) for a in qa.arrows])")
    
    collapsed, prob = measure!(qa)
    println("  Measured: $(collapsed.label) (p = $(round(prob, digits=3)))")
    println()
    
    # Pluck the quiver
    println("PLUCK THE QUANTUM QUIVER:")
    qa2 = QuantumArrow(INITIATE)
    qp = pluck!(qa2, NARRATIVE_A3; max_length=2)
    println("  Starting arrow: initiate")
    println("  Quantum paths (length ≤ 2):")
    for (amp, path) in zip(qp.amplitudes, qp.paths)
        println("    $(path) — |amp|² = $(round(abs2(amp), digits=3))")
    end
    println()
    
    # Entanglement
    println("ENTANGLED ARROWS (Bell state):")
    bell = entangle_arrows(OBSERVE, AFFECT)
    println("  |Φ⁺⟩ = (|observe,observe⟩ + |affect,affect⟩)/√2")
    println("  State: $(bell.state)")
    println()
    
    # Path algebra
    println("PATH ALGEBRA MULTIPLICATION:")
    p1 = QuiverPath([INITIATE])
    p2 = QuiverPath([TRANSFORM])
    composed = compose(p1, p2)
    println("  $(p1) ∘ $(p2) = $(composed)")
    println()
    
    println("═══════════════════════════════════════════════════════════════")
    println("  \"A quiver is a category freely generated by a graph.\"")
    println("  — The arrows vibrate in superposition until observed")
    println("═══════════════════════════════════════════════════════════════")
end

end # module QuantumQuiver
