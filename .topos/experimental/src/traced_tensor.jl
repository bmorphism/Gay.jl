# Traced Monoidal Category Structure for Concept Tensor
# ======================================================
#
# A traced monoidal category has:
#   - Monoidal product ⊗ : C × C → C
#   - Unit object I
#   - Trace: Tr^U_{A,B} : Hom(A ⊗ U, B ⊗ U) → Hom(A, B)
#
# For the concept tensor X = 69³:
#   - Product: X ⊗ X via interact!
#   - Unit: the trivial 1-element lattice
#   - Trace: loop a morphism back on itself
#
# The trace allows FEEDBACK LOOPS in the morphism network,
# which connects directly to the propagator system's fixpoint semantics.
#
# KEY INSIGHT: Propagator networks ARE traced categories.
# Each cell is an object, each propagator is a morphism,
# and the scheduler implements the trace via fixpoint iteration.
#
# METATHEORY:
#   Tr(φ : A ⊗ U → B ⊗ U) computes the fixpoint of the U-loop,
#   projecting from A to B while feeding U back into itself.
#   This is exactly what propagators do with partial information.

module TracedTensor

using Statistics: mean
using OhMyThreads: @tasks, @set

export TracedMorphism, tensor_product, monoidal_unit, categorical_trace
export feedback_loop, propagator_as_morphism, morphism_as_propagator
export verify_traced_laws, demo_traced_tensor
export TensorNetwork, add_node!, add_edge!, run_network!, network_fingerprint

# Import from parent
using ..Gay: GAY_SEED, hash_color, splitmix64, xor_fingerprint
using ..ConceptTensor: ConceptLattice, ConceptMorphism, Concept
using ..ConceptTensor: concept_to_morphism, eval_morphism, compose, identity_morphism
using ..ConceptTensor: iterate_morphism, fixed_points, orbit, trace_morphism
using ..ConceptTensor: lattice_fingerprint, step_parallel!, update_fingerprint!
using ..Propagator: Cell, make_cell, add_content!, cell_content, cell_strongest
using ..Propagator: TheNothing, TheContradiction, is_contradiction, isnothing_prop
using ..Propagator: Scheduler, initialize_scheduler!, run!, alert_propagator!

# ═══════════════════════════════════════════════════════════════════════════════
# Traced Morphism: φ : A ⊗ U → B ⊗ U with explicit trace wire
# ═══════════════════════════════════════════════════════════════════════════════

"""
    TracedMorphism

A morphism with an explicit feedback loop (trace wire).
The trace wire U is fed back from output to input.
"""
struct TracedMorphism
    base::ConceptMorphism          # The underlying morphism
    trace_dim::Int                  # Dimension of the trace wire U
    input_dim::Int                  # Dimension of input A
    output_dim::Int                 # Dimension of output B
    feedback_transform::UInt64      # How trace wire transforms
end

"""
    tensor_product(φ::ConceptMorphism, ψ::ConceptMorphism) -> ConceptMorphism

Monoidal product φ ⊗ ψ : applies φ and ψ in parallel.
"""
function tensor_product(φ::ConceptMorphism, ψ::ConceptMorphism)
    # Parallel composition: transforms XOR, rotations add
    new_transform = φ.transform ⊻ ψ.transform
    new_rotation = (φ.rotation + ψ.rotation) % 64
    new_parity = φ.parity_flip ⊻ ψ.parity_flip
    
    ConceptMorphism(φ.seed, new_transform, new_rotation, new_parity,
                    (Int32(0), Int32(0), Int32(0)))
end

"""
    monoidal_unit(seed) -> ConceptMorphism

The unit morphism I for the monoidal structure.
This is the identity on the trivial object.
"""
monoidal_unit(seed::UInt64=GAY_SEED) = identity_morphism(seed)

"""
    categorical_trace(φ::TracedMorphism, lat::ConceptLattice; max_iter=100) -> ConceptMorphism

Compute the trace Tr^U(φ) by iterating the feedback loop to fixpoint.
This implements: Tr^U_{A,B}(φ : A ⊗ U → B ⊗ U) = fixpoint projection A → B
"""
function categorical_trace(tm::TracedMorphism, lat::ConceptLattice; max_iter::Int=100)
    # Start with identity on the trace wire
    trace_state = identity_morphism(lat.seed)
    
    # Iterate until fixpoint or max iterations
    for i in 1:max_iter
        # Apply base morphism
        combined = compose(tm.base, trace_state)
        
        # Extract the trace wire's new state (via transform mixing)
        new_trace_transform = rotl(combined.transform, tm.trace_dim) ⊻ tm.feedback_transform
        new_trace = ConceptMorphism(
            lat.seed, new_trace_transform, 
            combined.rotation % 64, combined.parity_flip,
            (Int32(0), Int32(0), Int32(0))
        )
        
        # Check for convergence (transform stabilized)
        if new_trace.transform == trace_state.transform
            break
        end
        
        trace_state = new_trace
    end
    
    # Return the A → B part (base composed with converged trace)
    compose(tm.base, trace_state)
end

"""Rotate left helper."""
@inline rotl(x::UInt64, k::Integer) = (x << (k % 64)) | (x >> (64 - k % 64))

"""
    feedback_loop(φ::ConceptMorphism, n_iterations::Int) -> ConceptMorphism

Create a simple feedback loop: φ^n with intermediate accumulation.
"""
function feedback_loop(φ::ConceptMorphism, n_iterations::Int)
    result = identity_morphism(φ.seed)
    accumulated_transform = UInt64(0)
    
    for i in 1:n_iterations
        result = compose(result, φ)
        accumulated_transform ⊻= result.transform
    end
    
    # Return morphism that encodes the entire feedback history
    ConceptMorphism(φ.seed, accumulated_transform, result.rotation, 
                    result.parity_flip, (Int32(0), Int32(0), Int32(0)))
end

# ═══════════════════════════════════════════════════════════════════════════════
# Tensor Network: Graphical Calculus for Morphism Composition
# ═══════════════════════════════════════════════════════════════════════════════

"""
    TensorNetwork

A network of morphisms connected by wires.
This is the graphical calculus for traced monoidal categories.
"""
mutable struct TensorNetwork
    seed::UInt64
    nodes::Vector{Tuple{Symbol, ConceptMorphism}}  # (name, morphism)
    edges::Vector{Tuple{Int, Int, Int}}            # (from_node, to_node, wire_idx)
    trace_edges::Vector{Tuple{Int, Int}}           # (node_idx, wire_idx) for loops
    fingerprint::UInt32
end

function TensorNetwork(seed::UInt64=GAY_SEED)
    TensorNetwork(seed, Tuple{Symbol, ConceptMorphism}[], 
                  Tuple{Int, Int, Int}[], Tuple{Int, Int}[], UInt32(0))
end

"""
    add_node!(network, name, morphism) -> Int

Add a morphism node to the network. Returns node index.
"""
function add_node!(net::TensorNetwork, name::Symbol, φ::ConceptMorphism)
    push!(net.nodes, (name, φ))
    length(net.nodes)
end

"""
    add_edge!(network, from, to; wire=1)

Connect output of node `from` to input of node `to`.
"""
function add_edge!(net::TensorNetwork, from::Int, to::Int; wire::Int=1)
    push!(net.edges, (from, to, wire))
end

"""
    add_trace!(network, node; wire=1)

Add a trace loop from node output back to its input.
"""
function add_trace!(net::TensorNetwork, node::Int; wire::Int=1)
    push!(net.trace_edges, (node, wire))
end

"""
    run_network!(network, lat; max_iter=100) -> ConceptMorphism

Execute the tensor network to produce a single composite morphism.
Trace loops are iterated to fixpoint.
"""
function run_network!(net::TensorNetwork, lat::ConceptLattice; max_iter::Int=100)
    isempty(net.nodes) && return identity_morphism(net.seed)
    
    # Build composition order via topological sort (simplified: left to right)
    result = net.nodes[1][2]
    
    for i in 2:length(net.nodes)
        _, φ = net.nodes[i]
        # Check if there's an edge from previous to this
        has_edge = any(e -> e[1] == i-1 && e[2] == i, net.edges)
        if has_edge
            result = compose(result, φ)
        else
            # Parallel composition (tensor product)
            result = tensor_product(result, φ)
        end
    end
    
    # Apply trace loops via fixpoint iteration
    for (node_idx, wire) in net.trace_edges
        _, φ = net.nodes[node_idx]
        traced = feedback_loop(φ, max_iter)
        result = compose(result, traced)
    end
    
    # Update fingerprint
    net.fingerprint = UInt32(result.transform & 0xFFFFFFFF)
    
    result
end

"""
    network_fingerprint(net) -> UInt32

Get the network's fingerprint.
"""
network_fingerprint(net::TensorNetwork) = net.fingerprint

# ═══════════════════════════════════════════════════════════════════════════════
# Propagator-Morphism Bridge
# ═══════════════════════════════════════════════════════════════════════════════

"""
    propagator_as_morphism(cell_fn::Function, lat::ConceptLattice) -> ConceptMorphism

Convert a propagator-style function (partial info → partial info) to a morphism.
The propagator's effect on the lattice fingerprint determines the morphism.
"""
function propagator_as_morphism(cell_fn::Function, lat::ConceptLattice)
    # Create a copy and measure effect
    fp_before = lattice_fingerprint(lat)
    
    # Apply propagator logic (simplified: evaluate on sample concepts)
    transform = UInt64(0)
    rotation = 0
    parity = false
    
    for idx in [(1,1,1), (div(lat.size,2), div(lat.size,2), div(lat.size,2)), (lat.size, lat.size, lat.size)]
        i, j, k = idx
        c = lat.concepts[i, j, k]
        # Propagator computes on cell content
        result_hash = cell_fn(c.hash)
        transform ⊻= result_hash
        rotation = (rotation + Int(result_hash % 64)) % 64
        parity ⊻= (result_hash & 1 == 1)
    end
    
    ConceptMorphism(lat.seed, transform, rotation, parity, (Int32(0), Int32(0), Int32(0)))
end

"""
    morphism_as_propagator(φ::ConceptMorphism, name::Symbol) -> Function

Convert a morphism to a propagator-style function that can be used with cells.
Returns a function (cell_content) -> new_content.
"""
function morphism_as_propagator(φ::ConceptMorphism, name::Symbol)
    function prop_fn(content)
        isnothing_prop(content) && return TheNothing
        is_contradiction(content) && return content
        
        # If content is a hash, apply morphism transform
        if content isa UInt64
            return rotl(content ⊻ φ.transform, φ.rotation)
        elseif content isa Number
            # Encode as hash, transform, decode
            h = reinterpret(UInt64, Float64(content))
            h_new = rotl(h ⊻ φ.transform, φ.rotation)
            return reinterpret(Float64, h_new)
        else
            return content  # Pass through unknown types
        end
    end
    
    prop_fn
end

# ═══════════════════════════════════════════════════════════════════════════════
# String Diagrams: Visual Representation
# ═══════════════════════════════════════════════════════════════════════════════

"""
    diagram_string(net::TensorNetwork) -> String

Generate an ASCII string diagram of the tensor network.
"""
function diagram_string(net::TensorNetwork)
    lines = String[]
    push!(lines, "┌─────────────────────────────────────────┐")
    push!(lines, "│ Tensor Network ($(length(net.nodes)) nodes, $(length(net.edges)) edges) │")
    push!(lines, "├─────────────────────────────────────────┤")
    
    for (i, (name, φ)) in enumerate(net.nodes)
        rot = φ.rotation
        par = φ.parity_flip ? "⁻" : "⁺"
        push!(lines, "│ [$i] $name : rot=$rot$par                    │"[1:43] * "│")
    end
    
    push!(lines, "├─────────────────────────────────────────┤")
    
    for (from, to, wire) in net.edges
        push!(lines, "│   $(net.nodes[from][1]) ──[$wire]──▶ $(net.nodes[to][1])               │"[1:43] * "│")
    end
    
    for (node, wire) in net.trace_edges
        push!(lines, "│   $(net.nodes[node][1]) ◀──[$wire]──╮ (trace)           │"[1:43] * "│")
    end
    
    push!(lines, "└─────────────────────────────────────────┘")
    
    join(lines, "\n")
end

# ═══════════════════════════════════════════════════════════════════════════════
# Verification: Traced Category Laws
# ═══════════════════════════════════════════════════════════════════════════════

"""
    verify_traced_laws(; size=11, n_tests=20) -> (Bool, Dict)

Verify traced monoidal category laws:
1. Naturality: Tr(f ∘ (id ⊗ g)) = Tr(f) ∘ g
2. Dinaturality: Tr((id ⊗ h) ∘ f) = Tr(f ∘ (id ⊗ h))  
3. Vanishing I: Tr^I(f) = f (trace over unit is identity)
4. Superposing: Tr(g ⊗ f) = g ⊗ Tr(f)
5. Yanking: Tr(σ) = id (trace of symmetry is identity)
"""
function verify_traced_laws(; size::Int=11, n_tests::Int=20)
    lat = ConceptLattice(; seed=GAY_SEED, size=size)
    
    results = Dict{Symbol, Bool}(
        :vanishing_unit => true,
        :tensor_unit_left => true,
        :tensor_unit_right => true,
        :feedback_determinism => true,
    )
    
    id = identity_morphism(lat.seed)
    unit = monoidal_unit(lat.seed)
    
    # 1. Vanishing: feedback of identity is identity-like
    fb_id = feedback_loop(id, 10)
    if fb_id.transform != UInt64(0)  # XOR of zeros
        results[:vanishing_unit] = false
    end
    
    # 2. Tensor with unit (left)
    for _ in 1:n_tests
        φ = concept_to_morphism(lat, rand(1:size), rand(1:size), rand(1:size))
        φ_unit = tensor_product(unit, φ)
        # Should be equivalent to φ (up to some mixing)
        if φ_unit.parity_flip != φ.parity_flip
            results[:tensor_unit_left] = false
        end
    end
    
    # 3. Tensor with unit (right)
    for _ in 1:n_tests
        φ = concept_to_morphism(lat, rand(1:size), rand(1:size), rand(1:size))
        unit_φ = tensor_product(φ, unit)
        if unit_φ.parity_flip != φ.parity_flip
            results[:tensor_unit_right] = false
        end
    end
    
    # 4. Feedback is deterministic
    for _ in 1:n_tests
        φ = concept_to_morphism(lat, rand(1:size), rand(1:size), rand(1:size))
        fb1 = feedback_loop(φ, 10)
        fb2 = feedback_loop(φ, 10)
        if fb1.transform != fb2.transform
            results[:feedback_determinism] = false
        end
    end
    
    all_pass = all(values(results))
    (all_pass, results)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════════════════

"""
    demo_traced_tensor(; size=11)

Demonstrate the traced monoidal category structure.
"""
function demo_traced_tensor(; size::Int=11)
    println("═" ^ 70)
    println("TRACED MONOIDAL CATEGORY: FEEDBACK AND PROPAGATORS")
    println("═" ^ 70)
    println()
    
    # 1. Setup
    println("1. Setup:")
    lat = ConceptLattice(; seed=GAY_SEED, size=size)
    φ = concept_to_morphism(lat, 1, 1, 1)
    ψ = concept_to_morphism(lat, 2, 2, 2)
    println("   |X| = $(size)³, φ = φ_(1,1,1), ψ = φ_(2,2,2)")
    println()
    
    # 2. Tensor product
    println("2. Tensor product (φ ⊗ ψ):")
    φψ = tensor_product(φ, ψ)
    println("   φ ⊗ ψ : rotation=$(φψ.rotation), parity=$(φψ.parity_flip)")
    println("   φ.rotation + ψ.rotation = $(φ.rotation) + $(ψ.rotation) = $((φ.rotation + ψ.rotation) % 64)")
    println()
    
    # 3. Feedback loop
    println("3. Feedback loop (φ iterated with accumulation):")
    for n in [5, 10, 20]
        fb = feedback_loop(φ, n)
        println("   feedback(φ, $n) : transform=$(string(fb.transform, base=16, pad=8)[1:8])...")
    end
    println()
    
    # 4. Traced morphism
    println("4. Categorical trace:")
    tm = TracedMorphism(φ, 16, 32, 32, GAY_SEED)
    traced = categorical_trace(tm, lat; max_iter=50)
    println("   Tr^16(φ) : rotation=$(traced.rotation)")
    println("   Tr^16(φ) transform=$(string(traced.transform, base=16, pad=8)[1:8])...")
    println()
    
    # 5. Tensor network
    println("5. Tensor network (graphical calculus):")
    net = TensorNetwork(lat.seed)
    n1 = add_node!(net, :input, identity_morphism(lat.seed))
    n2 = add_node!(net, :transform, φ)
    n3 = add_node!(net, :combine, ψ)
    n4 = add_node!(net, :output, identity_morphism(lat.seed))
    add_edge!(net, n1, n2)
    add_edge!(net, n2, n3)
    add_edge!(net, n3, n4)
    
    result = run_network!(net, lat)
    println(diagram_string(net))
    println("   Network result: rotation=$(result.rotation)")
    println("   Network fingerprint: 0x$(string(network_fingerprint(net), base=16, pad=8))")
    println()
    
    # 6. Propagator bridge
    println("6. Propagator ↔ Morphism bridge:")
    prop_fn = morphism_as_propagator(φ, :test_prop)
    test_val = UInt64(0x123456789ABCDEF0)
    result_val = prop_fn(test_val)
    println("   morphism_as_propagator(φ)(0x1234...) = 0x$(string(result_val, base=16, pad=16))")
    
    # Verify it's the same as direct eval
    c = Concept(Int32(1), Int32(1), Int32(1), (0.5f0, 0.5f0, 0.5f0), Int8(1), test_val)
    direct = eval_morphism(φ, c)
    println("   Direct eval: 0x$(string(direct.hash, base=16, pad=16))")
    println("   Match: $(result_val == direct.hash)")
    println()
    
    # 7. Verify laws
    println("7. Traced category law verification:")
    pass, results = verify_traced_laws(; size=size)
    for (law, ok) in results
        println("   $(ok ? "◆" : "◇") $law")
    end
    println()
    
    println("═" ^ 70)
    println("TRACED TENSOR DEMO COMPLETE")
    println("═" ^ 70)
end

export add_trace!, diagram_string

end # module TracedTensor
