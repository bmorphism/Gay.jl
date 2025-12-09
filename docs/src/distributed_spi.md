# Distributed SPI Verification

**Verify distributed tensor-parallel inference with deterministic colors**

Gay.jl extends Strong Parallelism Invariance (SPI) to distributed systems, enabling verification of:

- **Tensor parallelism**: Vocabulary/hidden dimension sharding across GPUs
- **Pipeline parallelism**: Layer sharding across devices  
- **Data parallelism**: Sequence/batch sharding
- **Exo clusters**: Memory-weighted ring partitioning across MacBooks

## The Problem

Distributed inference can silently corrupt data through:
- Bit flips in memory or network transmission
- Race conditions in AllGather/AllReduce operations
- Pipeline handoff errors between stages
- Floating-point non-determinism across devices

Traditional approaches require gathering all data to verify correctness — expensive and defeats the purpose of distribution.

## The Solution: XOR Fingerprints

Gay.jl uses **XOR fingerprints** that are:

- **Associative**: `fp(A ∪ B) = fp(A) ⊕ fp(B)`
- **Commutative**: Order-independent verification
- **Pre-computable**: Know the expected fingerprint BEFORE running inference

```julia
using Gay

# Pre-compute expected fingerprint
expected_fp = expected_fingerprint(seed, n_tokens, hidden_dim; layer=5)

# Run distributed inference...
# Each device computes its shard's fingerprint locally

# Verify without gathering
@assert actual_fp == expected_fp  # Single XOR comparison!
```

## Architecture Support

### Data Parallelism (Sequence Sharding)

```
Device 0: tokens[1:64]    → fp₀
Device 1: tokens[65:128]  → fp₁
─────────────────────────────────
Combined: fp₀ ⊕ fp₁ = expected_fp
```

### Tensor Parallelism (Vocabulary Sharding)

```julia
using Gay: TensorPartition, verify_allgather

# Each device handles a shard of the vocabulary
partition = TensorPartition(dim=2, n_shards=4, shard_id=rank, global_size=vocab_size)

# Color the logits
color_logits!(logits, partition; seed=GAY_SEED)

# Verify AllGather produced correct result
@assert verify_allgather(gathered_logits, partitions; seed=GAY_SEED)
```

### Pipeline Parallelism (Layer Sharding)

```julia
using Gay: verify_pipeline_handoff

# After each pipeline stage handoff
@assert verify_pipeline_handoff(
    activations, 
    from_rank=0, 
    to_rank=1, 
    layer=16;
    seed=GAY_SEED
)
```

## Exo Cluster Integration

For [exo](https://github.com/exo-explore/exo) clusters running MLX:

```julia
using Gay: ExoCluster, ExoVerifier, verify_exo_inference

# Define cluster topology
cluster = ExoCluster([
    ("MacBook Pro", 18.0, "192.168.1.10"),
    ("MacBook Air", 8.0, "192.168.1.11"),
], "olmo-7b")

# Memory-weighted layer assignment:
#   MacBook Pro (18GB): layers 1-22 (69%)
#   MacBook Air (8GB):  layers 23-32 (31%)

# Pre-compute expected fingerprints
verifier = ExoVerifier(cluster, n_tokens=128)

# Verify after inference
@assert verify_exo_inference(cluster, "Hello world"; n_tokens=32)
```

See also:
- [Fault Tolerance](fault_tolerance.md) — Jepsen-style testing
- [Kernel Lifetimes](kernel_lifetimes.md) — GPU kernel verification
- [Galois Connections](galois_connections.md) — Mathematical foundations

## API Reference

```@docs
TensorPartition
ShardedTensor
DistributedContext
verify_allgather
verify_allreduce
verify_pipeline_handoff
ExoPartition
create_exo_partitions
verify_exo_ring
verify_distributed_inference
```
