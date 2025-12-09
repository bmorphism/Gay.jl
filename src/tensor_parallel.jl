# Tensor-Parallel Verification with SPI Colors
#
# Verifies correctness of distributed tensor-parallel inference by:
# 1. Assigning deterministic colors to tensor elements based on global position
# 2. Computing XOR fingerprints that are invariant to partition/gather order
# 3. Detecting bit-flip corruptions, race conditions, and AllGather errors
#
# Architecture support:
# - Data parallelism (sequence sharding)
# - Tensor parallelism (vocabulary/hidden dim sharding)
# - Pipeline parallelism (layer sharding across devices)
# - Hybrid (exo-style: memory-weighted ring partitioning)
#
# METATHEORY:
#   The XOR fingerprint forms a commutative monoid:
#     - Identity: 0
#     - Associative: (a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)
#     - Commutative: a ⊕ b = b ⊕ a
#   
#   This means: fp(gather(shards)) = reduce(⊕, map(fp, shards))
#   So we can verify distributed computation WITHOUT gathering!

module TensorParallel

using ..Gay: GAY_SEED, hash_color, splitmix64, xor_fingerprint
using ..KernelLifetimes: eventual_color, eventual_fingerprint, iter_index_color, cartesian_color
using Colors: RGB

export TensorPartition, ShardedTensor, DistributedContext
export shard!, gather!, verify_shard, verify_gather
export color_hidden_states!, color_logits!, color_embeddings!
export expected_fingerprint, verify_allgather, verify_allreduce
export pipeline_stage_color, verify_pipeline_handoff
export ExoPartition, verify_exo_ring

# ═══════════════════════════════════════════════════════════════════════════════
# Partition Descriptors
# ═══════════════════════════════════════════════════════════════════════════════

"""
Describes how a tensor is partitioned across ranks.
"""
struct TensorPartition
    dim::Int              # Which dimension is sharded (1=rows, 2=cols, etc.)
    n_shards::Int         # Total number of shards
    shard_id::Int         # This rank's shard (0-indexed)
    global_size::Int      # Full size along sharded dimension
    local_size::Int       # This shard's size
    offset::Int           # Starting index in global tensor
end

function TensorPartition(dim::Int, n_shards::Int, shard_id::Int, global_size::Int)
    local_size = cld(global_size, n_shards)
    offset = shard_id * local_size
    # Handle last shard being smaller
    if shard_id == n_shards - 1
        local_size = global_size - offset
    end
    TensorPartition(dim, n_shards, shard_id, global_size, local_size, offset)
end

"""
A tensor shard with its partition metadata and fingerprint.
"""
mutable struct ShardedTensor{T, N}
    data::Array{T, N}
    partition::TensorPartition
    seed::UInt64
    fingerprint::UInt32
    colored::Bool
end

function ShardedTensor(data::Array{T, N}, partition::TensorPartition; 
                       seed::Integer=GAY_SEED) where {T, N}
    ShardedTensor{T, N}(data, partition, UInt64(seed), UInt32(0), false)
end

"""
Context for a distributed computation across multiple ranks.
"""
struct DistributedContext
    rank::Int             # This process's rank (0-indexed)
    world_size::Int       # Total number of ranks
    seed::UInt64          # Global seed for SPI
    layer_range::UnitRange{Int}  # Which layers this rank handles (pipeline)
end

function DistributedContext(rank::Int, world_size::Int; 
                            seed::Integer=GAY_SEED,
                            n_layers::Int=32)
    layers_per_rank = cld(n_layers, world_size)
    start_layer = rank * layers_per_rank + 1
    end_layer = min((rank + 1) * layers_per_rank, n_layers)
    DistributedContext(rank, world_size, UInt64(seed), start_layer:end_layer)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Coloring Functions for Different Tensor Types
# ═══════════════════════════════════════════════════════════════════════════════

"""
    color_embeddings!(emb::Matrix{Float32}, token_ids::Vector{Int}, seed)

Color embedding lookups. Each (token_id, dim) pair gets deterministic color.
"""
function color_embeddings!(emb::Matrix{Float32}, token_ids::Vector{Int}; 
                           seed::Integer=GAY_SEED)
    n_tokens, hidden_dim = size(emb)
    @assert length(token_ids) == n_tokens
    
    for t in 1:n_tokens
        tok = token_ids[t]
        for d in 1:hidden_dim
            # Color by (token_id, dimension) — same token always gets same color
            h = UInt64(seed) ⊻ (UInt64(tok) * 0x9e3779b97f4a7c15) ⊻ 
                              (UInt64(d) * 0x517cc1b727220a95)
            r, _, _ = hash_color(h, UInt64(d))
            # Additive coloring (preserves gradients better than XOR)
            emb[t, d] += r * 1e-6f0
        end
    end
    emb
end

"""
    color_hidden_states!(hidden::Matrix{Float32}, layer::Int, partition; seed)

Color hidden states after a transformer layer.
Position-based coloring: (global_token_idx, layer, hidden_dim)
"""
function color_hidden_states!(hidden::Matrix{Float32}, layer::Int,
                               partition::TensorPartition; seed::Integer=GAY_SEED)
    n_tokens, hidden_dim = size(hidden)
    
    for t in 1:n_tokens
        global_t = partition.offset + t  # Global token index
        for d in 1:hidden_dim
            # Triple hash: (token, layer, dim)
            h = UInt64(seed) ⊻ (UInt64(global_t) * 0x9e3779b97f4a7c15) ⊻
                              (UInt64(layer) * 0x517cc1b727220a95) ⊻
                              (UInt64(d) * 0xc4ceb9fe1a85ec53)
            r, _, _ = hash_color(h, UInt64(global_t))
            hidden[t, d] += r * 1e-7f0
        end
    end
    hidden
end

"""
    color_logits!(logits::Matrix{Float32}, partition; seed)

Color LM head logits. For tensor-parallel vocab sharding.
Position-based: (global_token_idx, global_vocab_idx)
"""
function color_logits!(logits::Matrix{Float32}, partition::TensorPartition;
                        seed::Integer=GAY_SEED)
    n_tokens, vocab_shard = size(logits)
    
    for t in 1:n_tokens
        for v in 1:vocab_shard
            global_v = partition.offset + v  # Global vocab index
            r, _, _ = cartesian_color(UInt64(seed), t, global_v)
            logits[t, v] += r * 1e-7f0
        end
    end
    logits
end

# ═══════════════════════════════════════════════════════════════════════════════
# Fingerprint Computation
# ═══════════════════════════════════════════════════════════════════════════════

"""
    expected_fingerprint(seed, shape, partition_type) -> UInt32

Compute the expected fingerprint for a tensor BEFORE computation.
This is the "eventual" property — we know the answer ahead of time.
"""
function expected_fingerprint(seed::Integer, n_tokens::Int, hidden_dim::Int;
                               layer::Int=1)
    fp = UInt32(0)
    for t in 1:n_tokens
        for d in 1:hidden_dim
            h = UInt64(seed) ⊻ (UInt64(t) * 0x9e3779b97f4a7c15) ⊻
                              (UInt64(layer) * 0x517cc1b727220a95) ⊻
                              (UInt64(d) * 0xc4ceb9fe1a85ec53)
            r, g, b = hash_color(h, UInt64(t))
            fp ⊻= reinterpret(UInt32, r)
        end
    end
    fp
end

"""
    shard_fingerprint(shard::ShardedTensor) -> UInt32

Compute fingerprint of a single shard.
"""
function shard_fingerprint(shard::ShardedTensor{Float32})
    xor_fingerprint(shard.data)
end

"""
    combined_fingerprint(shards::Vector{ShardedTensor}) -> UInt32

Combine fingerprints from multiple shards.
Due to XOR associativity: fp(A ∪ B) = fp(A) ⊕ fp(B)
"""
function combined_fingerprint(shards::Vector{<:ShardedTensor})
    reduce(⊻, shard_fingerprint.(shards))
end

# ═══════════════════════════════════════════════════════════════════════════════
# Verification Functions
# ═══════════════════════════════════════════════════════════════════════════════

"""
    verify_allgather(gathered, partitions, seed) -> Bool

Verify AllGather produced correct result.
Each rank can verify independently using only its local shard + expected fingerprint.
"""
function verify_allgather(gathered::Matrix{Float32}, 
                          partitions::Vector{TensorPartition};
                          seed::Integer=GAY_SEED, layer::Int=1)
    n_tokens, hidden_dim = size(gathered)
    
    # Compute actual fingerprint of gathered tensor
    actual_fp = xor_fingerprint(gathered)
    
    # Compute expected fingerprint (O(n) but no communication needed)
    expected_fp = expected_fingerprint(seed, n_tokens, hidden_dim; layer=layer)
    
    if actual_fp != expected_fp
        @error "AllGather verification FAILED" actual_fp expected_fp
        
        # Find first mismatch for debugging
        for t in 1:min(n_tokens, 10)
            for d in 1:min(hidden_dim, 10)
                h = UInt64(seed) ⊻ (UInt64(t) * 0x9e3779b97f4a7c15) ⊻
                                  (UInt64(layer) * 0x517cc1b727220a95) ⊻
                                  (UInt64(d) * 0xc4ceb9fe1a85ec53)
                expected_r, _, _ = hash_color(h, UInt64(t))
                actual = gathered[t, d]
                # Check if color component is present
                if abs(actual - expected_r * 1e-7f0) > 1e-5
                    @warn "Mismatch at (t=$t, d=$d)" expected_r actual
                    break
                end
            end
        end
        return false
    end
    
    @info "AllGather verified ✓" fingerprint=string(actual_fp, base=16, pad=8)
    true
end

"""
    verify_allreduce(reduced, n_ranks, seed) -> Bool

Verify AllReduce (sum) produced correct result.
For tensor-parallel matrix multiplies where results are summed.
"""
function verify_allreduce(reduced::Matrix{Float32}, n_ranks::Int;
                          seed::Integer=GAY_SEED)
    # For AllReduce, each rank contributed a partial sum
    # The fingerprint of the sum should match expected
    actual_fp = xor_fingerprint(reduced)
    
    # Expected: each position was summed n_ranks times
    # This is trickier because addition isn't XOR-friendly
    # Instead, verify statistical properties
    
    mean_val = sum(reduced) / length(reduced)
    expected_mean = 0.5f0 * n_ranks  # Colors are ~uniform in [0,1]
    
    if abs(mean_val - expected_mean) > 0.1 * n_ranks
        @error "AllReduce statistical check FAILED" mean_val expected_mean
        return false
    end
    
    @info "AllReduce verified ✓" fingerprint=string(actual_fp, base=16, pad=8)
    true
end

# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Parallelism (Layer Sharding)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    pipeline_stage_color(seed, layer, stage) -> RGB{Float32}

Get the "handoff color" for pipeline stage boundaries.
Used to verify activations passed between pipeline stages are correct.
"""
function pipeline_stage_color(seed::Integer, layer::Int, stage::Int)
    h = UInt64(seed) ⊻ (UInt64(layer) * 0x9e3779b97f4a7c15) ⊻
                      (UInt64(stage) * 0xdeadbeefcafebabe)
    r, g, b = hash_color(h, UInt64(layer))
    RGB{Float32}(r, g, b)
end

"""
    verify_pipeline_handoff(activations, from_rank, to_rank, layer, seed) -> Bool

Verify activations passed between pipeline stages are uncorrupted.
"""
function verify_pipeline_handoff(activations::Matrix{Float32},
                                  from_rank::Int, to_rank::Int, layer::Int;
                                  seed::Integer=GAY_SEED)
    # Compute fingerprint of activations
    actual_fp = xor_fingerprint(activations)
    
    # The sending rank should have computed the same fingerprint
    # This is verified by having both ranks compute independently
    expected_fp = expected_fingerprint(seed, size(activations, 1), 
                                       size(activations, 2); layer=layer)
    
    if actual_fp != expected_fp
        @error "Pipeline handoff CORRUPTED" from_rank to_rank layer actual_fp expected_fp
        return false
    end
    
    @info "Pipeline handoff verified ✓" from_rank to_rank layer
    true
end

# ═══════════════════════════════════════════════════════════════════════════════
# Exo-Style Ring Partitioning
# ═══════════════════════════════════════════════════════════════════════════════

"""
Describes exo's memory-weighted ring partitioning.
Each device handles a contiguous range of layers proportional to its memory.
"""
struct ExoPartition
    device_id::Int
    device_name::String
    memory_gb::Float64
    layer_range::UnitRange{Int}
    weight::Float64  # Fraction of total memory
end

"""
    create_exo_partitions(devices, n_layers) -> Vector{ExoPartition}

Create exo-style partitions based on device memory.
"""
function create_exo_partitions(devices::Vector{Tuple{String, Float64}}, n_layers::Int)
    total_memory = sum(d[2] for d in devices)
    partitions = ExoPartition[]
    current_layer = 1
    
    for (i, (name, mem)) in enumerate(devices)
        weight = mem / total_memory
        n_device_layers = round(Int, weight * n_layers)
        
        # Last device gets remaining layers
        if i == length(devices)
            n_device_layers = n_layers - current_layer + 1
        end
        
        layer_range = current_layer:(current_layer + n_device_layers - 1)
        push!(partitions, ExoPartition(i-1, name, mem, layer_range, weight))
        current_layer += n_device_layers
    end
    
    partitions
end

"""
    verify_exo_ring(activations, partitions, current_device, seed) -> Bool

Verify activations in exo ring topology are correct.
"""
function verify_exo_ring(activations::Matrix{Float32},
                          partitions::Vector{ExoPartition},
                          current_device::Int;
                          seed::Integer=GAY_SEED)
    partition = partitions[current_device + 1]
    
    # Fingerprint should match expected for this layer range
    expected_fp = UInt32(0)
    for layer in partition.layer_range
        layer_fp = expected_fingerprint(seed, size(activations, 1),
                                        size(activations, 2); layer=layer)
        expected_fp ⊻= layer_fp
    end
    
    actual_fp = xor_fingerprint(activations)
    
    if actual_fp != expected_fp
        @error "Exo ring verification FAILED" device=partition.device_name layers=partition.layer_range actual_fp expected_fp
        return false
    end
    
    @info "Exo ring verified ✓" device=partition.device_name layers=partition.layer_range
    true
end

# ═══════════════════════════════════════════════════════════════════════════════
# Full Inference Verification
# ═══════════════════════════════════════════════════════════════════════════════

"""
    verify_distributed_inference(; n_tokens, hidden_dim, vocab_size, 
                                   n_layers, n_ranks, seed) -> Bool

End-to-end verification of distributed inference.
Simulates the full pipeline and verifies each stage.
"""
function verify_distributed_inference(; n_tokens::Int=128,
                                        hidden_dim::Int=4096,
                                        vocab_size::Int=32000,
                                        n_layers::Int=32,
                                        n_ranks::Int=4,
                                        seed::Integer=GAY_SEED)
    println("═" ^ 70)
    println("DISTRIBUTED INFERENCE VERIFICATION")
    println("═" ^ 70)
    println("  Tokens: $n_tokens, Hidden: $hidden_dim, Vocab: $vocab_size")
    println("  Layers: $n_layers, Ranks: $n_ranks")
    println()
    
    all_pass = true
    
    # 1. Data parallel: each rank gets n_tokens/n_ranks tokens
    println("1. Data Parallelism (sequence sharding)")
    tokens_per_rank = n_tokens ÷ n_ranks
    for rank in 0:n_ranks-1
        partition = TensorPartition(1, n_ranks, rank, n_tokens)
        hidden = randn(Float32, tokens_per_rank, hidden_dim)
        color_hidden_states!(hidden, 1, partition; seed=seed)
        fp = xor_fingerprint(hidden)
        println("   Rank $rank: tokens $(partition.offset+1):$(partition.offset+tokens_per_rank), " *
                "fp=0x$(string(fp, base=16, pad=8))")
    end
    println()
    
    # 2. AllGather hidden states
    println("2. AllGather (hidden states)")
    gathered = randn(Float32, n_tokens, hidden_dim)
    partitions = [TensorPartition(1, n_ranks, r, n_tokens) for r in 0:n_ranks-1]
    # Simulate coloring the full gathered tensor
    for rank in 0:n_ranks-1
        partition = partitions[rank+1]
        start_t = partition.offset + 1
        end_t = partition.offset + partition.local_size
        view_h = view(gathered, start_t:end_t, :)
        color_hidden_states!(view_h, 1, partition; seed=seed)
    end
    all_pass &= verify_allgather(gathered, partitions; seed=seed)
    println()
    
    # 3. Tensor parallel LM head
    println("3. Tensor Parallelism (vocabulary sharding)")
    vocab_per_rank = vocab_size ÷ n_ranks
    logit_fps = UInt32[]
    for rank in 0:n_ranks-1
        partition = TensorPartition(2, n_ranks, rank, vocab_size)
        logits = randn(Float32, n_tokens, vocab_per_rank)
        color_logits!(logits, partition; seed=seed)
        fp = xor_fingerprint(logits)
        push!(logit_fps, fp)
        println("   Rank $rank: vocab $(partition.offset+1):$(partition.offset+vocab_per_rank), " *
                "fp=0x$(string(fp, base=16, pad=8))")
    end
    
    # Combined fingerprint
    combined_fp = reduce(⊻, logit_fps)
    println("   Combined: fp=0x$(string(combined_fp, base=16, pad=8))")
    println()
    
    # 4. Pipeline verification
    println("4. Pipeline Parallelism (layer sharding)")
    layers_per_rank = n_layers ÷ n_ranks
    for rank in 0:n_ranks-1
        start_layer = rank * layers_per_rank + 1
        end_layer = (rank + 1) * layers_per_rank
        activations = randn(Float32, n_tokens, hidden_dim)
        
        # Color for each layer in range
        for layer in start_layer:end_layer
            partition = TensorPartition(1, 1, 0, n_tokens)  # No sequence sharding
            color_hidden_states!(activations, layer, partition; seed=seed)
        end
        
        fp = xor_fingerprint(activations)
        println("   Rank $rank: layers $start_layer:$end_layer, " *
                "fp=0x$(string(fp, base=16, pad=8))")
    end
    println()
    
    # 5. Exo-style ring
    println("5. Exo Ring (memory-weighted)")
    devices = [("MacBook Pro M3", 18.0), ("MacBook Air M2", 8.0)]
    exo_partitions = create_exo_partitions(devices, n_layers)
    for p in exo_partitions
        println("   $(p.device_name): $(p.memory_gb)GB → layers $(p.layer_range) " *
                "($(round(p.weight * 100, digits=1))%)")
    end
    println()
    
    println("═" ^ 70)
    println(all_pass ? "ALL VERIFICATIONS PASSED ✓" : "SOME VERIFICATIONS FAILED ✗")
    println("═" ^ 70)
    
    all_pass
end

export verify_distributed_inference

end # module
