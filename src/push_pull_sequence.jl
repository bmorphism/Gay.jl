# Push-Pull Sequence Color Verification
#
# The push-pull model for SPI color verification:
#
# PUSH (Forward): Inject colors as tokens flow through layers
#   - Each (token, layer, dim) gets a deterministic color from α(event)
#   - Colors accumulate in the fingerprint as XOR
#   - Think: "coloring the computation as it happens"
#
# PULL (Backward): Verify colors match expectations  
#   - Recompute expected colors from the same events
#   - Compare fingerprints: actual ⊕ expected should be 0
#   - Think: "checking the trail left by computation"
#
# For SEQUENCES specifically:
#   - Tokens arrive in order: t₁, t₂, t₃, ...
#   - Each token gets colored at each layer
#   - The sequence fingerprint accumulates: fp = fp ⊕ color(tᵢ, layer)
#   - At end of sequence: verify fp matches expected
#
# STREAMING VARIANT:
#   - Don't wait for full sequence
#   - Verify incrementally with running fingerprint
#   - Each chunk can be verified independently

module PushPullSequence

using ..Gay: GAY_SEED, hash_color, splitmix64, xor_fingerprint, SplitMix64RNG
using ..FaultTolerant: GaloisConnection, Event, Color, alpha, verify_closure

export SequenceColorStream, push_token!, pull_verify!
export StreamingVerifier, push_chunk!, verify_chunk!
export SequenceFingerprint, expected_sequence_fp, actual_sequence_fp

# ═══════════════════════════════════════════════════════════════════════════════
# Sequence Fingerprint
# ═══════════════════════════════════════════════════════════════════════════════

"""
A fingerprint for a sequence of tokens processed through layers.
"""
mutable struct SequenceFingerprint
    seed::UInt64
    n_layers::Int
    hidden_dim::Int
    current_fp::UInt32
    token_count::Int
    layer_fps::Vector{UInt32}  # Per-layer fingerprints for debugging
end

function SequenceFingerprint(; seed::Integer=GAY_SEED, n_layers::Int=32, hidden_dim::Int=4096)
    SequenceFingerprint(
        UInt64(seed), 
        n_layers, 
        hidden_dim, 
        UInt32(0), 
        0,
        zeros(UInt32, n_layers)
    )
end

"""
    push_token!(fp::SequenceFingerprint, token_id::Int) -> UInt32

PUSH: Add a token's color contribution to the sequence fingerprint.
Returns the token's contribution (for debugging).
"""
function push_token!(fp::SequenceFingerprint, token_id::Int)
    fp.token_count += 1
    token_contribution = UInt32(0)
    
    for layer in 1:fp.n_layers
        for dim in 1:fp.hidden_dim
            # Color from (token, layer, dim)
            h = fp.seed ⊻ (UInt64(token_id) * 0x9e3779b97f4a7c15) ⊻
                          (UInt64(layer) * 0x517cc1b727220a95) ⊻
                          (UInt64(dim) * 0xc4ceb9fe1a85ec53)
            r, _, _ = hash_color(h, UInt64(token_id))
            r_bits = reinterpret(UInt32, r)
            
            token_contribution ⊻= r_bits
            fp.layer_fps[layer] ⊻= r_bits
        end
    end
    
    fp.current_fp ⊻= token_contribution
    token_contribution
end

"""
    pull_verify!(fp::SequenceFingerprint, expected::UInt32) -> Bool

PULL: Verify the sequence fingerprint matches expected.
"""
function pull_verify!(fp::SequenceFingerprint, expected::UInt32)
    fp.current_fp == expected
end

# ═══════════════════════════════════════════════════════════════════════════════
# Streaming Sequence Verification
# ═══════════════════════════════════════════════════════════════════════════════

"""
Verifies sequences in streaming fashion, chunk by chunk.

PUSH: As each chunk arrives, compute its fingerprint
PULL: Verify chunk fingerprint, then combine into running total
"""
mutable struct StreamingVerifier
    seed::UInt64
    n_layers::Int
    hidden_dim::Int
    
    # Push state
    chunks_pushed::Int
    running_fp::UInt32
    chunk_fps::Vector{UInt32}
    
    # Pull state
    expected_chunk_fps::Vector{UInt32}
    verified_chunks::Int
    all_verified::Bool
end

function StreamingVerifier(; seed::Integer=GAY_SEED, n_layers::Int=32, hidden_dim::Int=4096)
    StreamingVerifier(
        UInt64(seed), n_layers, hidden_dim,
        0, UInt32(0), UInt32[],
        UInt32[], 0, true
    )
end

"""
    push_chunk!(sv::StreamingVerifier, token_ids::Vector{Int}) -> UInt32

PUSH: Process a chunk of tokens and return its fingerprint.
"""
function push_chunk!(sv::StreamingVerifier, token_ids::Vector{Int})
    chunk_fp = UInt32(0)
    
    for token_id in token_ids
        for layer in 1:sv.n_layers
            for dim in 1:sv.hidden_dim
                h = sv.seed ⊻ (UInt64(token_id) * 0x9e3779b97f4a7c15) ⊻
                              (UInt64(layer) * 0x517cc1b727220a95) ⊻
                              (UInt64(dim) * 0xc4ceb9fe1a85ec53)
                r, _, _ = hash_color(h, UInt64(token_id))
                chunk_fp ⊻= reinterpret(UInt32, r)
            end
        end
    end
    
    sv.chunks_pushed += 1
    sv.running_fp ⊻= chunk_fp
    push!(sv.chunk_fps, chunk_fp)
    
    chunk_fp
end

"""
    set_expected!(sv::StreamingVerifier, chunk_idx::Int, expected_fp::UInt32)

Set the expected fingerprint for a chunk (computed before inference).
"""
function set_expected!(sv::StreamingVerifier, chunk_idx::Int, expected_fp::UInt32)
    while length(sv.expected_chunk_fps) < chunk_idx
        push!(sv.expected_chunk_fps, UInt32(0))
    end
    sv.expected_chunk_fps[chunk_idx] = expected_fp
end

"""
    verify_chunk!(sv::StreamingVerifier, chunk_idx::Int) -> Bool

PULL: Verify a specific chunk matches its expected fingerprint.
"""
function verify_chunk!(sv::StreamingVerifier, chunk_idx::Int)
    if chunk_idx > length(sv.chunk_fps) || chunk_idx > length(sv.expected_chunk_fps)
        return false
    end
    
    actual = sv.chunk_fps[chunk_idx]
    expected = sv.expected_chunk_fps[chunk_idx]
    
    match = actual == expected
    if match
        sv.verified_chunks += 1
    else
        sv.all_verified = false
    end
    
    match
end

"""
    verify_all!(sv::StreamingVerifier) -> Bool

PULL: Verify all chunks and return overall result.
"""
function verify_all!(sv::StreamingVerifier)
    for i in 1:length(sv.chunk_fps)
        verify_chunk!(sv, i)
    end
    sv.all_verified
end

# ═══════════════════════════════════════════════════════════════════════════════
# Color Stream (Fine-grained push/pull)
# ═══════════════════════════════════════════════════════════════════════════════

"""
A stream of colors for a sequence, supporting fine-grained push/pull.

Each element in the sequence gets a color, and we track the
Galois connection at each step.
"""
mutable struct SequenceColorStream
    galois::GaloisConnection
    seed::UInt64
    
    # Push state (forward pass)
    pushed_colors::Vector{Color}
    pushed_events::Vector{Event}
    push_fp::UInt32
    
    # Pull state (backward verification)
    pulled_colors::Vector{Color}
    pull_fp::UInt32
    
    # Verification
    mismatches::Vector{Tuple{Int, Color, Color}}  # (index, pushed, pulled)
end

function SequenceColorStream(; seed::Integer=GAY_SEED)
    SequenceColorStream(
        GaloisConnection(seed),
        UInt64(seed),
        Color[], Event[], UInt32(0),
        Color[], UInt32(0),
        Tuple{Int, Color, Color}[]
    )
end

"""
    push!(stream::SequenceColorStream, token::Int, layer::Int, dim::Int) -> Color

PUSH: Inject a color for this position in the sequence.
"""
function Base.push!(stream::SequenceColorStream, token::Int, layer::Int, dim::Int)
    event = Event(stream.seed, token, layer, dim)
    color = alpha(stream.galois, event)
    
    push!(stream.pushed_events, event)
    push!(stream.pushed_colors, color)
    
    # Update fingerprint - use the direct hash color for consistency with expected_sequence_fp
    h = stream.seed ⊻ (UInt64(token) * 0x9e3779b97f4a7c15) ⊻
                      (UInt64(layer) * 0x517cc1b727220a95) ⊻
                      (UInt64(dim) * 0xc4ceb9fe1a85ec53)
    r, _, _ = hash_color(h, UInt64(token))
    r_bits = reinterpret(UInt32, r)
    stream.push_fp ⊻= r_bits
    
    color
end

"""
    pull!(stream::SequenceColorStream, idx::Int) -> (Color, Bool)

PULL: Verify the color at index matches what was pushed.
Returns (color, matches).
"""
function pull!(stream::SequenceColorStream, idx::Int)
    if idx > length(stream.pushed_events)
        error("Cannot pull index $idx, only $(length(stream.pushed_events)) events pushed")
    end
    
    # Recompute color from event
    event = stream.pushed_events[idx]
    color = alpha(stream.galois, event)
    push!(stream.pulled_colors, color)
    
    # Update pull fingerprint - use same hash as push
    h = stream.seed ⊻ (UInt64(event.token) * 0x9e3779b97f4a7c15) ⊻
                      (UInt64(event.layer) * 0x517cc1b727220a95) ⊻
                      (UInt64(event.dim) * 0xc4ceb9fe1a85ec53)
    r, _, _ = hash_color(h, UInt64(event.token))
    r_bits = reinterpret(UInt32, r)
    stream.pull_fp ⊻= r_bits
    
    # Check match
    pushed = stream.pushed_colors[idx]
    matches = color.index == pushed.index
    
    if !matches
        push!(stream.mismatches, (idx, pushed, color))
    end
    
    (color, matches)
end

"""
    verify_stream!(stream::SequenceColorStream) -> (Bool, Vector)

Pull and verify the entire stream.
Returns (all_match, mismatches).
"""
function verify_stream!(stream::SequenceColorStream)
    for i in 1:length(stream.pushed_events)
        pull!(stream, i)
    end
    
    all_match = isempty(stream.mismatches) && stream.push_fp == stream.pull_fp
    (all_match, stream.mismatches)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Expected Sequence Fingerprint (Pre-computation)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    expected_sequence_fp(token_ids, n_layers, hidden_dim; seed) -> UInt32

Pre-compute the expected fingerprint for a sequence of tokens.
This is computed BEFORE inference runs.
"""
function expected_sequence_fp(token_ids::Vector{Int}, n_layers::Int, hidden_dim::Int;
                               seed::Integer=GAY_SEED)
    fp = UInt32(0)
    
    for token_id in token_ids
        for layer in 1:n_layers
            for dim in 1:hidden_dim
                h = UInt64(seed) ⊻ (UInt64(token_id) * 0x9e3779b97f4a7c15) ⊻
                                   (UInt64(layer) * 0x517cc1b727220a95) ⊻
                                   (UInt64(dim) * 0xc4ceb9fe1a85ec53)
                r, _, _ = hash_color(h, UInt64(token_id))
                fp ⊻= reinterpret(UInt32, r)
            end
        end
    end
    
    fp
end

# ═══════════════════════════════════════════════════════════════════════════════
# Demonstration
# ═══════════════════════════════════════════════════════════════════════════════

"""
    world_push_pull_sequence()

Demonstrate push-pull sequence verification.
"""
function world_push_pull_sequence()
    println("═" ^ 70)
    println("PUSH-PULL SEQUENCE COLOR VERIFICATION")
    println("═" ^ 70)
    println()
    
    # Small example for demo
    n_layers = 2
    hidden_dim = 4
    tokens = [101, 102, 103]  # Token IDs
    
    # ─────────────────────────────────────────────────────────────────
    # 1. Pre-compute expected (before inference)
    # ─────────────────────────────────────────────────────────────────
    println("1. PRE-COMPUTE (before inference)")
    expected = expected_sequence_fp(tokens, n_layers, hidden_dim)
    println("   Expected fingerprint: 0x$(string(expected, base=16, pad=8))")
    println()
    
    # ─────────────────────────────────────────────────────────────────
    # 2. PUSH phase (simulate forward pass)
    # ─────────────────────────────────────────────────────────────────
    println("2. PUSH PHASE (forward pass)")
    stream = SequenceColorStream()
    
    for token in tokens
        for layer in 1:n_layers
            for dim in 1:hidden_dim
                color = push!(stream, token, layer, dim)
            end
        end
        println("   Token $token → pushed $(n_layers * hidden_dim) colors")
    end
    println("   Push fingerprint: 0x$(string(stream.push_fp, base=16, pad=8))")
    println()
    
    # ─────────────────────────────────────────────────────────────────
    # 3. PULL phase (verification)
    # ─────────────────────────────────────────────────────────────────
    println("3. PULL PHASE (verification)")
    all_match, mismatches = verify_stream!(stream)
    println("   Pull fingerprint: 0x$(string(stream.pull_fp, base=16, pad=8))")
    println("   Push == Pull: $(stream.push_fp == stream.pull_fp ? "◆" : "◇")")
    println("   Matches expected: $(stream.push_fp == expected ? "◆" : "◇")")
    println("   Mismatches: $(length(mismatches))")
    println()
    
    # ─────────────────────────────────────────────────────────────────
    # 4. Streaming verification
    # ─────────────────────────────────────────────────────────────────
    println("4. STREAMING VERIFICATION (chunk by chunk)")
    sv = StreamingVerifier(n_layers=n_layers, hidden_dim=hidden_dim)
    
    # Pre-compute expected for each chunk
    chunks = [[101], [102], [103]]
    for (i, chunk) in enumerate(chunks)
        expected_chunk = expected_sequence_fp(chunk, n_layers, hidden_dim)
        set_expected!(sv, i, expected_chunk)
        println("   Chunk $i expected: 0x$(string(expected_chunk, base=16, pad=8))")
    end
    println()
    
    # Push each chunk
    println("   Pushing chunks:")
    for (i, chunk) in enumerate(chunks)
        actual = push_chunk!(sv, chunk)
        verified = verify_chunk!(sv, i)
        println("   Chunk $i: actual=0x$(string(actual, base=16, pad=8)) $(verified ? "◆" : "◇")")
    end
    println()
    
    println("   Final running fp: 0x$(string(sv.running_fp, base=16, pad=8))")
    println("   All verified: $(sv.all_verified ? "◆ PASS" : "◇ FAIL")")
    println()
    
    println("═" ^ 70)
    println("DEMO COMPLETE")
    println("═" ^ 70)
end

export world_push_pull_sequence

end # module PushPullSequence
