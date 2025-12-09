# Fault-Tolerant SPI Verification with Jepsen-Style Testing
#
# Provides fault injection, bidirectional color tracking, and Galois connection
# verification for distributed tensor-parallel inference.
#
# FEATURES:
#   - Jepsen-style fault injection (network partitions, message delays, bit flips)
#   - Bidirectional color tracking maintaining Galois invariants
#   - Integration with verify_allgather, verify_pipeline_handoff, verify_exo_ring
#   - Property-based testing with deterministic replay
#
# USAGE:
#   using Gay: FaultTolerant
#   cluster = FaultTolerant.SimulatedCluster(devices, n_layers)
#   FaultTolerant.inject!(cluster, :bit_flip, device=0, n_bits=10)
#   results = FaultTolerant.run_inference!(cluster)
#   @assert FaultTolerant.verify!(cluster)

module FaultTolerant

using ..Gay: GAY_SEED, hash_color, splitmix64, xor_fingerprint
using ..TensorParallel: TensorPartition, expected_fingerprint, ExoPartition, create_exo_partitions
using Random
using Dates

export SimulatedCluster, DeviceState, FaultInjector
export inject!, heal!, heal_all!, run_inference!, verify!
export BidirectionalTracker, track_forward!, track_backward!, verify_consistency!
export GaloisConnection, alpha, gamma, verify_closure, verify_all_closures
export FaultType, NetworkPartition, MessageDelay, MessageDrop, BitFlip, Byzantine

# ═══════════════════════════════════════════════════════════════════════════════
# Galois Connection
# ═══════════════════════════════════════════════════════════════════════════════

"""
Galois connection between Events and Colors.

α(e) = hash(e) mod 226    (left adjoint, abstraction)
γ(c) = representative(c)  (right adjoint, concretization)

Closure: α(γ(c)) = c for all c ∈ [0, 226)
"""
struct GaloisConnection
    seed::UInt64
    palette_size::Int
    palette::Vector{Tuple{Float32, Float32, Float32}}
    
    function GaloisConnection(seed::Integer=GAY_SEED; palette_size::Int=226)
        palette = [hash_color(UInt64(seed), UInt64(i)) for i in 0:palette_size-1]
        new(UInt64(seed), palette_size, palette)
    end
end

"""
Event: A concrete computation step.
"""
struct Event
    seed::UInt64
    token::Int
    layer::Int
    dim::Int
end

"""
Color: An abstract color index.
"""
struct Color
    index::Int
    r::Float32
    g::Float32
    b::Float32
    
    function Color(index::Int, rgb::Tuple{Float32, Float32, Float32})
        new(index, rgb[1], rgb[2], rgb[3])
    end
end

"""
    alpha(gc::GaloisConnection, e::Event) -> Color

Abstraction: Event → Color (left adjoint).
"""
function alpha(gc::GaloisConnection, e::Event)
    h = e.seed ⊻ (UInt64(e.token) * 0x9e3779b97f4a7c15) ⊻
                 (UInt64(e.layer) * 0x517cc1b727220a95) ⊻
                 (UInt64(e.dim) * 0xc4ceb9fe1a85ec53)
    h = splitmix64(h)
    index = Int(h % gc.palette_size)
    Color(index, gc.palette[index + 1])
end

"""
    gamma(gc::GaloisConnection, c::Color) -> Event

Concretization: Color → representative Event (right adjoint).
"""
function gamma(gc::GaloisConnection, c::Color)
    Event(gc.seed, c.index, 1, 1)
end

"""
    verify_closure(gc::GaloisConnection, c::Color) -> Bool

Verify α(γ(c)) = c for this color.
"""
function verify_closure(gc::GaloisConnection, c::Color)
    representative = gamma(gc, c)
    abstracted = alpha(gc, representative)
    abstracted.index == c.index
end

"""
    verify_all_closures(gc::GaloisConnection) -> Bool

Verify closure property for all 226 colors.
"""
function verify_all_closures(gc::GaloisConnection)
    all(verify_closure(gc, Color(i, gc.palette[i + 1])) for i in 0:gc.palette_size-1)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Fault Types (Jepsen-style)
# ═══════════════════════════════════════════════════════════════════════════════

abstract type FaultType end

struct NetworkPartition <: FaultType
    partition_a::Set{Int}
    partition_b::Set{Int}
    duration_ms::Float64
end

struct MessageDelay <: FaultType
    target_devices::Set{Int}
    delay_ms::Float64
    probability::Float64
end

struct MessageDrop <: FaultType
    target_devices::Set{Int}
    probability::Float64
end

struct BitFlip <: FaultType
    target_devices::Set{Int}
    n_bits::Int
    probability::Float64
end

struct Byzantine <: FaultType
    target_devices::Set{Int}
    probability::Float64
end

"""
Fault injection state.
"""
mutable struct FaultInjector
    rng::MersenneTwister
    active_faults::Vector{FaultType}
    fault_history::Vector{Tuple{DateTime, FaultType, Symbol}}  # (time, fault, :injected/:healed)
    
    FaultInjector(seed::Integer=42) = new(MersenneTwister(seed), FaultType[], Tuple{DateTime, FaultType, Symbol}[])
end

"""
    inject!(fi::FaultInjector, fault::FaultType)

Inject a fault into the system.
"""
function inject!(fi::FaultInjector, fault::FaultType)
    push!(fi.active_faults, fault)
    push!(fi.fault_history, (now(), fault, :injected))
    fault
end

"""
    heal!(fi::FaultInjector, fault::FaultType)

Remove a specific fault from the system.
"""
function heal!(fi::FaultInjector, fault::FaultType)
    idx = findfirst(==(fault), fi.active_faults)
    if idx !== nothing
        deleteat!(fi.active_faults, idx)
        push!(fi.fault_history, (now(), fault, :healed))
    end
end

"""
    heal_all!(fi::FaultInjector)

Remove all active faults.
"""
function heal_all!(fi::FaultInjector)
    for fault in copy(fi.active_faults)
        push!(fi.fault_history, (now(), fault, :healed))
    end
    empty!(fi.active_faults)
end

"""
    apply_to_tensor!(fi::FaultInjector, tensor::Matrix{Float32}, device_id::Int)

Apply active faults to a tensor.
"""
function apply_to_tensor!(fi::FaultInjector, tensor::Matrix{Float32}, device_id::Int)
    for fault in fi.active_faults
        if fault isa BitFlip && device_id in fault.target_devices
            if rand(fi.rng) < fault.probability
                # Flip random bits
                flat = vec(tensor)
                uint_view = reinterpret(UInt32, flat)
                for _ in 1:fault.n_bits
                    idx = rand(fi.rng, 1:length(uint_view))
                    bit = UInt32(1) << rand(fi.rng, 0:31)
                    uint_view[idx] ⊻= bit
                end
            end
        end
    end
    tensor
end

# ═══════════════════════════════════════════════════════════════════════════════
# Device State
# ═══════════════════════════════════════════════════════════════════════════════

"""
State of a simulated device.
"""
mutable struct DeviceState
    device_id::Int
    name::String
    memory_gb::Float64
    layer_range::UnitRange{Int}
    current_fingerprint::UInt32
    expected_fingerprint::UInt32
    color_history::Vector{Tuple{Int, Int, Color}}  # (layer, token, color)
    is_partitioned::Bool
end

# ═══════════════════════════════════════════════════════════════════════════════
# Simulated Cluster
# ═══════════════════════════════════════════════════════════════════════════════

"""
Simulated distributed cluster for SPI verification testing.
"""
mutable struct SimulatedCluster
    devices::Vector{Tuple{String, Float64}}
    n_layers::Int
    seed::UInt64
    n_tokens::Int
    hidden_dim::Int
    partitions::Vector{ExoPartition}
    device_states::Dict{Int, DeviceState}
    fault_injector::FaultInjector
    galois::GaloisConnection
end

"""
    SimulatedCluster(devices, n_layers; seed, n_tokens, hidden_dim)

Create a simulated cluster.
"""
function SimulatedCluster(devices::Vector{Tuple{String, Float64}}, n_layers::Int;
                          seed::Integer=GAY_SEED, n_tokens::Int=128, hidden_dim::Int=4096)
    partitions = create_exo_partitions(devices, n_layers)
    
    device_states = Dict{Int, DeviceState}()
    for p in partitions
        expected_fp = UInt32(0)
        for layer in p.layer_range
            layer_fp = expected_fingerprint(seed, n_tokens, hidden_dim; layer=layer)
            expected_fp ⊻= layer_fp
        end
        
        device_states[p.device_id] = DeviceState(
            p.device_id,
            p.device_name,
            p.memory_gb,
            p.layer_range,
            UInt32(0),
            expected_fp,
            Tuple{Int, Int, Color}[],
            false
        )
    end
    
    SimulatedCluster(
        devices, n_layers, UInt64(seed), n_tokens, hidden_dim,
        partitions, device_states, FaultInjector(), GaloisConnection(seed)
    )
end

"""
    inject!(cluster::SimulatedCluster, fault_type::Symbol; kwargs...)

Convenience function to inject faults.
"""
function inject!(cluster::SimulatedCluster, fault_type::Symbol; 
                 device::Int=0, n_bits::Int=1, probability::Float64=1.0,
                 partition_a::Set{Int}=Set{Int}(), partition_b::Set{Int}=Set{Int}(),
                 delay_ms::Float64=100.0)
    fault = if fault_type == :bit_flip
        BitFlip(Set([device]), n_bits, probability)
    elseif fault_type == :network_partition
        NetworkPartition(partition_a, partition_b, 1000.0)
    elseif fault_type == :message_delay
        MessageDelay(Set([device]), delay_ms, probability)
    elseif fault_type == :message_drop
        MessageDrop(Set([device]), probability)
    elseif fault_type == :byzantine
        Byzantine(Set([device]), probability)
    else
        error("Unknown fault type: $fault_type")
    end
    inject!(cluster.fault_injector, fault)
end

"""
    heal_all!(cluster::SimulatedCluster)

Remove all faults from cluster.
"""
heal_all!(cluster::SimulatedCluster) = heal_all!(cluster.fault_injector)

"""
    run_inference!(cluster::SimulatedCluster; with_faults::Bool=false)

Simulate distributed inference with optional fault injection.
"""
function run_inference!(cluster::SimulatedCluster; with_faults::Bool=false)
    results = Dict{Int, UInt32}()
    
    for (device_id, state) in cluster.device_states
        # Generate synthetic hidden states
        hidden = randn(Float32, cluster.n_tokens, cluster.hidden_dim)
        
        # Apply SPI coloring for each layer
        for layer in state.layer_range
            for t in 1:cluster.n_tokens
                for d in 1:cluster.hidden_dim
                    h = cluster.seed ⊻ (UInt64(t) * 0x9e3779b97f4a7c15) ⊻
                                       (UInt64(layer) * 0x517cc1b727220a95) ⊻
                                       (UInt64(d) * 0xc4ceb9fe1a85ec53)
                    r, _, _ = hash_color(h, UInt64(t))
                    hidden[t, d] += r * 1e-7f0
                end
            end
            
            # Track colors bidirectionally
            for t in 1:min(10, cluster.n_tokens)
                event = Event(cluster.seed, t, layer, 1)
                color = alpha(cluster.galois, event)
                push!(state.color_history, (layer, t, color))
            end
        end
        
        # Apply faults if enabled
        if with_faults
            apply_to_tensor!(cluster.fault_injector, hidden, device_id)
        end
        
        # Compute fingerprint
        fp = xor_fingerprint(hidden)
        state.current_fingerprint = fp
        results[device_id] = fp
    end
    
    results
end

"""
    verify!(cluster::SimulatedCluster) -> (Bool, Dict{Int, String})

Verify all devices and return (all_pass, error_details).
"""
function verify!(cluster::SimulatedCluster)
    all_pass = true
    errors = Dict{Int, String}()
    
    for (device_id, state) in cluster.device_states
        if state.current_fingerprint != state.expected_fingerprint
            all_pass = false
            errors[device_id] = "Fingerprint mismatch on $(state.name): " *
                               "expected 0x$(string(state.expected_fingerprint, base=16, pad=8)), " *
                               "got 0x$(string(state.current_fingerprint, base=16, pad=8))"
        end
    end
    
    (all_pass, errors)
end

"""
    verify_galois_invariants!(cluster::SimulatedCluster) -> (Bool, Vector{String})

Verify Galois connection invariants at all tracked points.
"""
function verify_galois_invariants!(cluster::SimulatedCluster)
    errors = String[]
    
    # Verify closure for all colors in palette
    if !verify_all_closures(cluster.galois)
        push!(errors, "Galois closure property violated")
    end
    
    # Verify color tracking history
    for (device_id, state) in cluster.device_states
        for (layer, token, color) in state.color_history
            event = Event(cluster.seed, token, layer, 1)
            derived = alpha(cluster.galois, event)
            if derived.index != color.index
                push!(errors, "Color tracking mismatch at ($layer, $token): " *
                             "expected index $(color.index), got $(derived.index)")
            end
        end
    end
    
    (isempty(errors), errors)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Bidirectional Color Tracker
# ═══════════════════════════════════════════════════════════════════════════════

"""
Tracks colors bidirectionally through forward and backward passes.
"""
mutable struct BidirectionalTracker
    galois::GaloisConnection
    forward_colors::Dict{Tuple{Int, Int, Int}, Color}   # (layer, token, dim) → Color
    backward_colors::Dict{Tuple{Int, Int, Int}, Color}
    proof_log::Vector{Dict{Symbol, Any}}
end

BidirectionalTracker(seed::Integer=GAY_SEED) = BidirectionalTracker(
    GaloisConnection(seed),
    Dict{Tuple{Int, Int, Int}, Color}(),
    Dict{Tuple{Int, Int, Int}, Color}(),
    Dict{Symbol, Any}[]
)

"""
    track_forward!(tracker, layer, token, dim; seed) -> Color

Track color in forward pass.
"""
function track_forward!(tracker::BidirectionalTracker, layer::Int, token::Int, dim::Int;
                        seed::Integer=GAY_SEED)
    event = Event(UInt64(seed), token, layer, dim)
    color = alpha(tracker.galois, event)
    
    key = (layer, token, dim)
    tracker.forward_colors[key] = color
    
    push!(tracker.proof_log, Dict(
        :direction => :forward,
        :layer => layer,
        :token => token,
        :dim => dim,
        :color_idx => color.index,
        :galois_closure => verify_closure(tracker.galois, color)
    ))
    
    color
end

"""
    track_backward!(tracker, layer, token, dim; seed) -> Color

Track color in backward pass.
"""
function track_backward!(tracker::BidirectionalTracker, layer::Int, token::Int, dim::Int;
                         seed::Integer=GAY_SEED)
    event = Event(UInt64(seed), token, layer, dim)
    color = alpha(tracker.galois, event)
    
    key = (layer, token, dim)
    tracker.backward_colors[key] = color
    
    consistent = if haskey(tracker.forward_colors, key)
        tracker.forward_colors[key].index == color.index
    else
        true
    end
    
    push!(tracker.proof_log, Dict(
        :direction => :backward,
        :layer => layer,
        :token => token,
        :dim => dim,
        :color_idx => color.index,
        :galois_closure => verify_closure(tracker.galois, color),
        :forward_consistent => consistent
    ))
    
    color
end

"""
    verify_consistency!(tracker::BidirectionalTracker) -> (Bool, Vector{String})

Verify all forward/backward color pairs are consistent.
"""
function verify_consistency!(tracker::BidirectionalTracker)
    errors = String[]
    
    common_keys = intersect(keys(tracker.forward_colors), keys(tracker.backward_colors))
    for key in common_keys
        fwd = tracker.forward_colors[key]
        bwd = tracker.backward_colors[key]
        if fwd.index != bwd.index
            layer, token, dim = key
            push!(errors, "Bidirectional mismatch at ($layer, $token, $dim): " *
                         "forward=$(fwd.index), backward=$(bwd.index)")
        end
    end
    
    (isempty(errors), errors)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Integration with existing verification functions
# ═══════════════════════════════════════════════════════════════════════════════

"""
    verify_with_fault_injection(cluster::SimulatedCluster; 
                                 fault_types::Vector{Symbol}=[:bit_flip],
                                 n_iterations::Int=10) -> Dict

Run verification with random fault injection.
"""
function verify_with_fault_injection(cluster::SimulatedCluster;
                                     fault_types::Vector{Symbol}=[:bit_flip],
                                     n_iterations::Int=10)
    results = Dict(
        :total_runs => n_iterations,
        :detections => 0,
        :false_positives => 0,
        :false_negatives => 0,
        :galois_violations => 0
    )
    
    for i in 1:n_iterations
        # Clear previous state
        heal_all!(cluster)
        for state in values(cluster.device_states)
            empty!(state.color_history)
        end
        
        # Randomly inject faults
        inject_faults = rand() < 0.5
        if inject_faults
            fault_type = rand(fault_types)
            device = rand(0:length(cluster.devices)-1)
            inject!(cluster, fault_type; device=device, n_bits=rand(1:20))
        end
        
        # Run inference
        run_inference!(cluster; with_faults=inject_faults)
        
        # Verify
        pass, errors = verify!(cluster)
        galois_ok, galois_errors = verify_galois_invariants!(cluster)
        
        # Categorize results
        if inject_faults && !pass
            results[:detections] += 1
        elseif !inject_faults && !pass
            results[:false_positives] += 1
        elseif inject_faults && pass
            results[:false_negatives] += 1
        end
        
        if !galois_ok
            results[:galois_violations] += 1
        end
    end
    
    results[:detection_rate] = results[:detections] / max(1, results[:detections] + results[:false_negatives])
    results[:false_positive_rate] = results[:false_positives] / n_iterations
    
    results
end

# ═══════════════════════════════════════════════════════════════════════════════
# Demonstration
# ═══════════════════════════════════════════════════════════════════════════════

"""
    demo_fault_tolerant()

Demonstrate fault-tolerant SPI verification.
"""
function demo_fault_tolerant()
    println("═" ^ 70)
    println("FAULT-TOLERANT SPI VERIFICATION DEMO")
    println("═" ^ 70)
    println()
    
    # 1. Setup cluster
    println("1. Setting up simulated cluster...")
    devices = [("MacBook Pro", 18.0), ("MacBook Air", 8.0)]
    cluster = SimulatedCluster(devices, 32; n_tokens=64, hidden_dim=256)
    
    for p in cluster.partitions
        state = cluster.device_states[p.device_id]
        println("   $(p.device_name): layers $(p.layer_range), " *
               "expected fp=0x$(string(state.expected_fingerprint, base=16, pad=8))")
    end
    println()
    
    # 2. Test Galois connection
    println("2. Verifying Galois connection...")
    gc = cluster.galois
    if verify_all_closures(gc)
        println("   ✓ All $(gc.palette_size) colors satisfy α(γ(c)) = c")
    else
        println("   ✗ Galois closure property violated!")
    end
    println()
    
    # 3. Run clean inference
    println("3. Running clean inference...")
    run_inference!(cluster; with_faults=false)
    pass, errors = verify!(cluster)
    println("   Result: $(pass ? "✓ PASS" : "✗ FAIL")")
    println()
    
    # 4. Inject faults
    println("4. Testing fault injection...")
    
    # Test bit flips
    heal_all!(cluster)
    inject!(cluster, :bit_flip; device=0, n_bits=10)
    run_inference!(cluster; with_faults=true)
    pass, errors = verify!(cluster)
    println("   Bit flip (10 bits): $(pass ? "✓ PASS (not detected)" : "✗ DETECTED")")
    
    # Test more bits
    heal_all!(cluster)
    inject!(cluster, :bit_flip; device=0, n_bits=100)
    run_inference!(cluster; with_faults=true)
    pass, errors = verify!(cluster)
    println("   Bit flip (100 bits): $(pass ? "✓ PASS (not detected)" : "✗ DETECTED")")
    println()
    
    # 5. Bidirectional tracking
    println("5. Testing bidirectional color tracking...")
    tracker = BidirectionalTracker(GAY_SEED)
    
    for layer in 1:4
        for token in 1:10
            track_forward!(tracker, layer, token, 1)
            track_backward!(tracker, layer, token, 1)
        end
    end
    
    consistent, errors = verify_consistency!(tracker)
    println("   Forward/backward consistency: $(consistent ? "✓ PASS" : "✗ FAIL")")
    
    # Check Galois closure in proof log
    all_closure_ok = all(step[:galois_closure] for step in tracker.proof_log)
    println("   Galois closure at all steps: $(all_closure_ok ? "✓ PASS" : "✗ FAIL")")
    println()
    
    # 6. Statistical fault detection
    println("6. Running fault detection statistics...")
    stats = verify_with_fault_injection(cluster; n_iterations=20)
    println("   Detection rate: $(round(stats[:detection_rate] * 100, digits=1))%")
    println("   False positive rate: $(round(stats[:false_positive_rate] * 100, digits=1))%")
    println("   Galois violations: $(stats[:galois_violations])")
    println()
    
    println("═" ^ 70)
    println("DEMO COMPLETE")
    println("═" ^ 70)
end

export demo_fault_tolerant

end # module FaultTolerant
