# Gay.jl QUIC Interleaved Streams
# ================================
# Multiple concurrent streams with XOR-combined fingerprints
# 
# Key insight: QUIC streams are independent, so we can interleave
# color generation across streams while maintaining SPI guarantees.
#
# XOR fingerprint combination: fp₁ ⊕ fp₂ ⊕ ... ⊕ fpₙ is schedule-invariant
# because XOR is commutative and associative.

using Colors: RGB

export QUICInterleaver, InterleavedStream
export interleave!, next_stream_color!, combined_fingerprint
export verify_interleave_spi, hop_state, from_hop_state
export world_quic_interleave

# ═══════════════════════════════════════════════════════════════════════════
# Interleaved Stream Types
# ═══════════════════════════════════════════════════════════════════════════

"""
    InterleavedStream

A single stream within the interleaver. Each stream has its own
RNG state derived from the base seed XOR stream index.
"""
mutable struct InterleavedStream
    id::UInt64
    state::UInt64           # SplitMix64 state
    colors_generated::Int
    fingerprint::UInt64     # XOR of all generated color hashes
end

"""
    QUICInterleaver

Manages multiple interleaved color streams over QUIC.

Fields:
- `streams`: Vector of independent streams
- `current_phase`: Which stream to sample next (round-robin)
- `step`: Number of complete rounds through all streams
- `global_fingerprint`: XOR of all stream fingerprints (schedule-invariant)
"""
mutable struct QUICInterleaver
    connection_id::UInt64
    seed::UInt64
    streams::Vector{InterleavedStream}
    current_phase::Int
    step::Int
    global_fingerprint::UInt64
end

# ═══════════════════════════════════════════════════════════════════════════
# Construction
# ═══════════════════════════════════════════════════════════════════════════

"""
    QUICInterleaver(connection_id, n_streams; seed=GAY_SEED)

Create an interleaver with n independent color streams.
Each stream gets a unique seed derived via XOR with stream index.
"""
function QUICInterleaver(connection_id::UInt64, n_streams::Int; seed::UInt64=GAY_SEED)
    streams = InterleavedStream[]
    
    for i in 0:n_streams-1
        # Derive stream seed: base seed XOR (stream_id * golden ratio)
        stream_seed = seed ⊻ (UInt64(i) * GOLDEN)
        stream_state = splitmix64(stream_seed ⊻ connection_id)
        
        push!(streams, InterleavedStream(
            UInt64(i),
            stream_state,
            0,
            stream_seed  # Initial fingerprint is the seed
        ))
    end
    
    QUICInterleaver(
        connection_id,
        seed,
        streams,
        1,      # 1-indexed phase
        0,
        seed    # Initial global fingerprint
    )
end

# ═══════════════════════════════════════════════════════════════════════════
# Color Generation
# ═══════════════════════════════════════════════════════════════════════════

"""
    next_stream_color!(il::QUICInterleaver) -> (RGB, stream_id, phase, step)

Generate next color from current stream, advance to next stream.
Round-robin across all streams.
"""
function next_stream_color!(il::QUICInterleaver)
    stream = il.streams[il.current_phase]
    
    # Generate color from stream's RNG state
    stream.state = splitmix64(stream.state)
    r = Float64((stream.state >> 16) & 0xFF) / 255.0
    g = Float64((stream.state >> 8) & 0xFF) / 255.0
    b = Float64(stream.state & 0xFF) / 255.0
    color = RGB(r, g, b)
    
    # Update stream fingerprint
    stream.fingerprint = stream.fingerprint ⊻ stream.state
    stream.colors_generated += 1
    
    # Update global fingerprint
    il.global_fingerprint = il.global_fingerprint ⊻ stream.state
    
    # Record current position
    stream_id = stream.id
    phase = il.current_phase
    step = il.step
    
    # Advance phase (round-robin)
    il.current_phase += 1
    if il.current_phase > length(il.streams)
        il.current_phase = 1
        il.step += 1
    end
    
    return (color, stream_id, phase, step)
end

"""
    interleave!(il::QUICInterleaver, n::Int) -> Vector{Tuple}

Generate n colors across interleaved streams.
Returns vector of (color, stream_id, phase, step) tuples.
"""
function interleave!(il::QUICInterleaver, n::Int)
    results = Vector{Tuple{RGB{Float64}, UInt64, Int, Int}}()
    sizehint!(results, n)
    
    for _ in 1:n
        push!(results, next_stream_color!(il))
    end
    
    return results
end

# ═══════════════════════════════════════════════════════════════════════════
# Fingerprinting & Verification
# ═══════════════════════════════════════════════════════════════════════════

"""
    combined_fingerprint(il::QUICInterleaver) -> UInt64

Get the combined XOR fingerprint of all streams.
This is schedule-invariant: same colors in any order produce same fingerprint.
"""
function combined_fingerprint(il::QUICInterleaver)
    # XOR all stream fingerprints
    reduce(⊻, (s.fingerprint for s in il.streams); init=il.seed)
end

"""
    verify_interleave_spi(il1::QUICInterleaver, il2::QUICInterleaver) -> Bool

Verify that two interleavers with same seed produce same fingerprint,
regardless of the order colors were generated (schedule invariance).
"""
function verify_interleave_spi(il1::QUICInterleaver, il2::QUICInterleaver)
    combined_fingerprint(il1) == combined_fingerprint(il2)
end

# ═══════════════════════════════════════════════════════════════════════════
# State Hopping (for runtime migration)
# ═══════════════════════════════════════════════════════════════════════════

"""
    InterleaverHopState

Serializable state for migrating interleaver across runtimes/connections.
"""
struct InterleaverHopState
    connection_id::UInt64
    seed::UInt64
    stream_states::Vector{UInt64}
    stream_fingerprints::Vector{UInt64}
    current_phase::Int
    step::Int
    global_fingerprint::UInt64
end

"""
    hop_state(il::QUICInterleaver) -> InterleaverHopState

Serialize interleaver state for runtime hopping.
"""
function hop_state(il::QUICInterleaver)
    InterleaverHopState(
        il.connection_id,
        il.seed,
        [s.state for s in il.streams],
        [s.fingerprint for s in il.streams],
        il.current_phase,
        il.step,
        il.global_fingerprint
    )
end

"""
    from_hop_state(state::InterleaverHopState) -> QUICInterleaver

Reconstruct interleaver from hopped state.
"""
function from_hop_state(state::InterleaverHopState)
    streams = [
        InterleavedStream(
            UInt64(i-1),
            state.stream_states[i],
            0,  # Can't recover colors_generated, but fingerprint is preserved
            state.stream_fingerprints[i]
        )
        for i in 1:length(state.stream_states)
    ]
    
    QUICInterleaver(
        state.connection_id,
        state.seed,
        streams,
        state.current_phase,
        state.step,
        state.global_fingerprint
    )
end

# ═══════════════════════════════════════════════════════════════════════════
# XOR Lattice (checkerboard pattern)
# ═══════════════════════════════════════════════════════════════════════════

"""
    xor_color(il::QUICInterleaver, i::Int, j::Int) -> RGB

Generate color at lattice position (i,j) via XOR mixing.
Useful for 2D visualizations of interleaved streams.
"""
function xor_color(il::QUICInterleaver, i::Int, j::Int)
    # Mix stream seeds with coordinates
    si = il.streams[mod1(i, length(il.streams))].state
    sj = il.streams[mod1(j, length(il.streams))].state
    
    mixed = splitmix64(si ⊻ sj ⊻ UInt64(i) ⊻ (UInt64(j) << 32))
    
    r = Float64((mixed >> 16) & 0xFF) / 255.0
    g = Float64((mixed >> 8) & 0xFF) / 255.0
    b = Float64(mixed & 0xFF) / 255.0
    
    RGB(r, g, b)
end

# ═══════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════

function ansi_rgb(c::RGB)
    r = round(Int, clamp(c.r, 0, 1) * 255)
    g = round(Int, clamp(c.g, 0, 1) * 255)
    b = round(Int, clamp(c.b, 0, 1) * 255)
    "\e[48;2;$(r);$(g);$(b)m"
end
const RESET = "\e[0m"

"""
    visualize_interleave(il::QUICInterleaver; width=40)

Show interleaved stream colors as ANSI blocks.
"""
function visualize_interleave(il::QUICInterleaver; width::Int=40)
    println("╔═══════════════════════════════════════════════════════════════╗")
    println("║  QUIC Interleaved Streams ($(length(il.streams)) streams)                     ║")
    println("╚═══════════════════════════════════════════════════════════════╝")
    println()
    
    # Clone to avoid mutating original
    il_copy = from_hop_state(hop_state(il))
    
    # Generate colors
    colors = interleave!(il_copy, width)
    
    # Display by stream
    println("  Round-robin sequence:")
    print("  ")
    for (c, sid, _, _) in colors
        print("$(ansi_rgb(c))  $(RESET)")
    end
    println()
    println()
    
    # Per-stream view
    println("  Per-stream fingerprints:")
    for (i, stream) in enumerate(il.streams)
        c = xor_color(il, i, i)
        fp_hex = string(stream.fingerprint, base=16, pad=16)[1:8]
        println("  Stream $(i-1): $(ansi_rgb(c))  $(RESET) fp=0x$(fp_hex)... ($(stream.colors_generated) colors)")
    end
    println()
    
    # Global fingerprint
    gfp = combined_fingerprint(il)
    println("  Combined XOR fingerprint: 0x$(string(gfp, base=16, pad=16))")
    println("  Step: $(il.step), Phase: $(il.current_phase)/$(length(il.streams))")
end

# ═══════════════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════════════

"""
    world_quic_interleave(; n_streams=4, n_colors=32)

Demonstrate QUIC interleaved streams with SPI verification.
"""
function world_quic_interleave(; n_streams::Int=4, n_colors::Int=32)
    println("🏳️‍🌈 QUIC Interleaved Streams Demo")
    println("=" ^ 60)
    println()
    
    conn_id = rand(UInt64)
    
    # Create two identical interleavers
    il1 = QUICInterleaver(conn_id, n_streams)
    il2 = QUICInterleaver(conn_id, n_streams)
    
    println("1. Creating $(n_streams) interleaved streams...")
    visualize_interleave(il1)
    
    # Generate colors in different orders
    println("2. Generating $(n_colors) colors in round-robin order...")
    for _ in 1:n_colors
        next_stream_color!(il1)
    end
    
    println("3. Generating same colors in REVERSE stream order...")
    # Generate from streams in reverse, but same total count per stream
    for _ in 1:n_colors÷n_streams
        for i in n_streams:-1:1
            # Force specific stream
            il2.current_phase = i
            next_stream_color!(il2)
        end
    end
    
    println()
    println("4. SPI Verification (schedule invariance):")
    fp1 = combined_fingerprint(il1)
    fp2 = combined_fingerprint(il2)
    
    println("   Forward order:  0x$(string(fp1, base=16, pad=16))")
    println("   Reverse order:  0x$(string(fp2, base=16, pad=16))")
    
    # Note: These won't match exactly because we changed the interleaving
    # The point is that the XOR combination is commutative
    println()
    println("5. XOR Lattice (checkerboard):")
    print("   ")
    for i in 1:8
        for j in 1:8
            c = xor_color(il1, i, j)
            print("$(ansi_rgb(c))  $(RESET)")
        end
        println()
        print("   ")
    end
    println()
    
    # Hop state demo
    println("6. State hopping (runtime migration):")
    state = hop_state(il1)
    il3 = from_hop_state(state)
    println("   Original fp:    0x$(string(combined_fingerprint(il1), base=16))")
    println("   Hopped fp:      0x$(string(combined_fingerprint(il3), base=16))")
    println("   Match: $(combined_fingerprint(il1) == combined_fingerprint(il3) ? "✓" : "✗")")
    
    return il1
end
