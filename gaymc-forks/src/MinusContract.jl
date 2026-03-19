# MinusContract.jl - MINUS POLARITY CONSOLIDATION
#
# Extract the SHARED TRAIT across all 4 forks.
# Each fork specializes ONE abstract interface:
#
#   Fork        | Domain           | Specialization
#   ------------|------------------|----------------------------------
#   Plurigrid   | EnergyGrid       | power_flow!, grid_partition!
#   TeglonLabs  | GraphSheaf       | cech_cohomology!, local_to_global!
#   Tritwies    | Narrative        | narrative_bfs!, interval_sheaf!
#   bmorphism   | SpinedCategory   | tree_width!, triangulate!
#
# CONSOLIDATION: Define abstract ChromaticStructure trait with:
#   - fingerprint(::T)::UInt64
#   - reset!(::T, seed)
#   - verify_spi(::T, seed)::Bool

module MinusContract

using ..SPIKernel
export ChromaticStructure, fingerprint, reset!, verify_spi
export MinimalInterleave, interleave_step!, combined_fingerprint

# ═══════════════════════════════════════════════════════════════════════════
# ABSTRACT TRAIT: ChromaticStructure
# ═══════════════════════════════════════════════════════════════════════════

abstract type ChromaticStructure end

fingerprint(::ChromaticStructure)::UInt64 = GAY_SEED
reset!(::ChromaticStructure, seed::UInt64) = nothing
verify_spi(s::ChromaticStructure, seed::UInt64)::Bool = begin
    fp1 = fingerprint(s)
    reset!(s, seed)
    fp2 = fingerprint(s)
    fp1 == fp2
end

# ═══════════════════════════════════════════════════════════════════════════
# MINIMAL INTERLEAVE PROTOCOL
# ═══════════════════════════════════════════════════════════════════════════
#
# Message format: 32 bytes total (MINUS-compressed from ~200 bytes)
#   [0:8]   seed (u64)
#   [8:12]  phase (u32)
#   [12:13] polarity (u8: 0=MINUS, 1=ERGODIC, 2=PLUS)  
#   [13:21] fingerprint (u64)
#   [21:32] reserved/checksum

struct InterleaveMsg
    seed::UInt64
    phase::UInt32
    polarity::UInt8
    fingerprint::UInt64
end

function pack(m::InterleaveMsg)::Vector{UInt8}
    buf = zeros(UInt8, 32)
    # Little-endian packing
    for i in 0:7; buf[i+1] = UInt8((m.seed >> (8i)) & 0xFF); end
    for i in 0:3; buf[i+9] = UInt8((m.phase >> (8i)) & 0xFF); end
    buf[13] = m.polarity
    for i in 0:7; buf[i+14] = UInt8((m.fingerprint >> (8i)) & 0xFF); end
    buf
end

function unpack(buf::Vector{UInt8})::InterleaveMsg
    seed = sum(UInt64(buf[i+1]) << (8i) for i in 0:7)
    phase = sum(UInt32(buf[i+9]) << (8i) for i in 0:3)
    polarity = buf[13]
    fingerprint = sum(UInt64(buf[i+14]) << (8i) for i in 0:7)
    InterleaveMsg(seed, phase, polarity, fingerprint)
end

# ═══════════════════════════════════════════════════════════════════════════
# MINIMAL 4-FORK INTERLEAVE STATE
# ═══════════════════════════════════════════════════════════════════════════

mutable struct MinimalInterleave
    seed::UInt64
    phase::UInt32
    fingerprints::NTuple{4,UInt64}  # One per fork
end

MinimalInterleave(seed::UInt64=GAY_SEED) = 
    MinimalInterleave(seed, 0, (seed, seed, seed, seed))

function interleave_step!(state::MinimalInterleave, fork_idx::Int, contribution_hash::UInt64)
    @assert 1 <= fork_idx <= 4 "Fork index must be 1-4"
    
    # Advance phase
    state.phase += 1
    
    # Update fork fingerprint via XOR mixing
    old_fp = state.fingerprints[fork_idx]
    new_fp = SPIKernel.fingerprint(old_fp, contribution_hash)
    
    # Reconstruct tuple (Julia tuples are immutable)
    fps = collect(state.fingerprints)
    fps[fork_idx] = new_fp
    state.fingerprints = Tuple(fps)
    
    # Return message for broadcast
    polarity = UInt8(state.phase % 3)
    InterleaveMsg(state.seed, state.phase, polarity, new_fp)
end

function combined_fingerprint(state::MinimalInterleave)::UInt64
    reduce(⊻, state.fingerprints)
end

# ═══════════════════════════════════════════════════════════════════════════
# FORK REGISTRY (compressed: name → trait instance)
# ═══════════════════════════════════════════════════════════════════════════

const FORK_NAMES = ("Plurigrid", "TeglonLabs", "Tritwies", "bmorphism")
const FORK_DOMAINS = ("energy", "sheaf", "temporal", "spined")

fork_name(idx::Int) = FORK_NAMES[idx]
fork_domain(idx::Int) = FORK_DOMAINS[idx]
fork_twist(idx::Int) = SPIKernel.twist(GAY_SEED, (idx - 1) % 3)

# ═══════════════════════════════════════════════════════════════════════════
# SPI VERIFICATION (unified across all forks)
# ═══════════════════════════════════════════════════════════════════════════

function verify_interleave_spi(seed::UInt64; n_trials::Int=10)::Bool
    ref_state = MinimalInterleave(seed)
    
    # Simulate interleaved contributions
    for i in 1:n_trials
        fork_idx = ((i - 1) % 4) + 1
        contribution = SPIKernel.sm64(seed + UInt64(i))
        interleave_step!(ref_state, fork_idx, contribution)
    end
    ref_fp = combined_fingerprint(ref_state)
    
    # Replay with same seed - must produce identical fingerprint
    test_state = MinimalInterleave(seed)
    for i in 1:n_trials
        fork_idx = ((i - 1) % 4) + 1
        contribution = SPIKernel.sm64(seed + UInt64(i))
        interleave_step!(test_state, fork_idx, contribution)
    end
    test_fp = combined_fingerprint(test_state)
    
    ref_fp == test_fp
end

end # module
