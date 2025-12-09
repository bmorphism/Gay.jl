# Fault-Tolerant Verification

**Jepsen-style testing for distributed SPI systems**

Gay.jl includes a fault injection framework inspired by [Jepsen](https://jepsen.io) for testing distributed system assumptions.

## Fault Types

```julia
using Gay: FaultInjector, inject!, heal_all!

injector = FaultInjector(seed=42)

# Network partition between device groups
inject!(injector, NetworkPartition(
    partition_a = Set([0, 1]),
    partition_b = Set([2, 3]),
    duration_ms = 1000.0
))

# Message delays
inject!(injector, MessageDelay(
    target_devices = Set([0]),
    delay_ms = 100.0,
    probability = 0.5
))

# Bit flips (simulates memory/network corruption)
inject!(injector, BitFlip(
    target_devices = Set([1]),
    n_bits = 10,
    probability = 1.0
))

# Byzantine faults (arbitrary malicious behavior)
inject!(injector, Byzantine(
    target_devices = Set([2]),
    probability = 0.1
))
```

## Simulated Cluster Testing

```julia
using Gay: SimulatedCluster, run_inference!, verify!

# Create test cluster
cluster = SimulatedCluster(
    [("Device A", 8.0), ("Device B", 8.0)],
    n_layers = 32,
    n_tokens = 128,
    hidden_dim = 4096
)

# Run clean inference
run_inference!(cluster; with_faults=false)
pass, errors = verify!(cluster)
@assert pass

# Inject faults and verify detection
inject!(cluster, :bit_flip; device=0, n_bits=10)
run_inference!(cluster; with_faults=true)
pass, errors = verify!(cluster)
@assert !pass  # Corruption detected!
```

## Galois Connection Verification

The fault-tolerant module maintains **Galois connection invariants** at each step:

```julia
using Gay: GaloisConnection, alpha, gamma, verify_closure

gc = GaloisConnection(GAY_SEED)

# α(e) = hash(e) mod 226 — abstraction
color = alpha(gc, event)

# γ(c) = representative(c) — concretization  
representative = gamma(gc, color)

# Closure property: α(γ(c)) = c for all colors
@assert verify_all_closures(gc)
```

## Bidirectional Color Tracking

Track colors through forward and backward passes:

```julia
using Gay: BidirectionalTracker, track_forward!, track_backward!, verify_consistency!

tracker = BidirectionalTracker(GAY_SEED)

# Forward pass
for layer in 1:32, token in 1:128
    track_forward!(tracker, layer, token, 1)
end

# Backward pass (gradients)
for layer in 32:-1:1, token in 128:-1:1
    track_backward!(tracker, layer, token, 1)
end

# Verify consistency
consistent, errors = verify_consistency!(tracker)
@assert consistent
```

## Statistical Fault Detection

Run multiple iterations with random fault injection:

```julia
using Gay: verify_with_fault_injection

stats = verify_with_fault_injection(
    cluster;
    fault_types = [:bit_flip, :message_delay],
    n_iterations = 100
)

println("Detection rate: $(stats[:detection_rate] * 100)%")
println("False positive rate: $(stats[:false_positive_rate] * 100)%")
println("Galois violations: $(stats[:galois_violations])")
```

## Demo

```julia
using Gay: demo_fault_tolerant
demo_fault_tolerant()
```

Output:
```
═══════════════════════════════════════════════════════════════════════
FAULT-TOLERANT SPI VERIFICATION DEMO
═══════════════════════════════════════════════════════════════════════

1. Setting up simulated cluster...
   MacBook Pro: layers 1:22, expected fp=0x12345678
   MacBook Air: layers 23:32, expected fp=0xabcdef01

2. Verifying Galois connection...
   ✓ All 226 colors satisfy α(γ(c)) = c

3. Running clean inference...
   Result: ✓ PASS

4. Testing fault injection...
   Bit flip (10 bits): ✗ DETECTED
   Bit flip (100 bits): ✗ DETECTED

5. Testing bidirectional color tracking...
   Forward/backward consistency: ✓ PASS
   Galois closure at all steps: ✓ PASS

6. Running fault detection statistics...
   Detection rate: 95.0%
   False positive rate: 0.0%
   Galois violations: 0
```

See also:
- [Distributed SPI](distributed_spi.md) — Tensor-parallel verification
- [Galois Connections](galois_connections.md) — Mathematical proofs
