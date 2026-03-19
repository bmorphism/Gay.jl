# QUIC Stream Polarity Mapping
# =============================
#
# Maps QUIC stream types to the 3-polarity system (MINUS/ERGODIC/PLUS).
# Provides polarity-based stream scheduling for deterministic color assignment.
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  POLARITY-TO-STREAM-TYPE MAPPING                                           │
# │                                                                             │
# │  Polarity   seed % 3   Symbol   QUIC Stream Type                           │
# │  ─────────────────────────────────────────────────────────────────────────  │
# │  MINUS        0          −      Server-initiated unidirectional (0x03)     │
# │                                  → Contraction: ACKs, STOP_SENDING, RST    │
# │                                  → Flow returns toward origin              │
# │                                                                             │
# │  ERGODIC      1          _      Bidirectional streams (0x00, 0x01)         │
# │                                  → Afference: Request/Response cycles      │
# │                                  → Equilibrium, stationary distribution    │
# │                                                                             │
# │  PLUS         2          +      Client-initiated unidirectional (0x02)     │
# │                                  → Expansion: PUSH, new data flows         │
# │                                  → Outward causation from initiator        │
# └─────────────────────────────────────────────────────────────────────────────┘
#
# RATIONALE:
#   - MINUS (contraction): Server responds, acknowledges, terminates
#   - ERGODIC (bidirectional): Symmetric exchange, mixing property
#   - PLUS (expansion): Client initiates, creates new causal chains
#
# The mapping respects QUIC's LSB encoding:
#   Stream Type = (Stream ID) & 0x03
#   0x00 = Client-initiated bidirectional → ERGODIC (symmetric)
#   0x01 = Server-initiated bidirectional → ERGODIC (symmetric)
#   0x02 = Client-initiated unidirectional → PLUS (expansion)
#   0x03 = Server-initiated unidirectional → MINUS (contraction)

module QUICPolarityStreams

using SplittableRandoms: SplittableRandom, split

export
    # Polarity Types
    StreamPolarity, MINUS, ERGODIC, PLUS,
    polarity_from_stream_type, polarity_symbol, polarity_name,
    
    # Stream Scheduling
    PolarityScheduler, StreamSlot, 
    schedule_next!, rebalance!, get_active_streams,
    
    # Color Integration
    stream_color, stream_seed,
    
    # Verification
    verify_polarity_invariants

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

const GAY_SEED = UInt64(0x6761795f636f6c6f)  # "gay_colo"
const GOLDEN = UInt64(0x9e3779b97f4a7c15)

# QUIC stream type bits (RFC 9000 §2.1)
const STREAM_BIDI = 0x00       # Bidirectional
const STREAM_UNI = 0x02        # Unidirectional
const STREAM_SERVER = 0x01     # Server-initiated
const STREAM_CLIENT = 0x00     # Client-initiated

# ═══════════════════════════════════════════════════════════════════════════════
# POLARITY TYPES
# ═══════════════════════════════════════════════════════════════════════════════

"""
    StreamPolarity

The three polarities derived from seed % 3.
Maps to stream behavioral characteristics.
"""
@enum StreamPolarity begin
    MINUS = 0    # Contraction, acknowledgment, return flow
    ERGODIC = 1  # Afference, bidirectional, equilibrium
    PLUS = 2     # Expansion, unidirectional outward, causation
end

const POLARITY_SYMBOLS = Dict(
    MINUS => '−',
    ERGODIC => '_',
    PLUS => '+'
)

const POLARITY_NAMES = Dict(
    MINUS => "MINUS",
    ERGODIC => "ERGODIC", 
    PLUS => "PLUS"
)

polarity_symbol(p::StreamPolarity)::Char = POLARITY_SYMBOLS[p]
polarity_name(p::StreamPolarity)::String = POLARITY_NAMES[p]

"""
    polarity_from_stream_type(stream_type::UInt8) → StreamPolarity

Map QUIC stream type (0x00-0x03) to polarity.

Stream types:
- 0x00: Client-initiated bidirectional → ERGODIC
- 0x01: Server-initiated bidirectional → ERGODIC  
- 0x02: Client-initiated unidirectional → PLUS
- 0x03: Server-initiated unidirectional → MINUS
"""
function polarity_from_stream_type(stream_type::UInt8)::StreamPolarity
    # Extract the 2 LSBs from stream ID to get type
    t = stream_type & 0x03
    if t == 0x00 || t == 0x01
        ERGODIC  # Bidirectional streams are ergodic (symmetric mixing)
    elseif t == 0x02
        PLUS     # Client uni = expansion (outward causation)
    else  # t == 0x03
        MINUS    # Server uni = contraction (acknowledgment/response)
    end
end

"""
    polarity_from_seed(seed::UInt64) → StreamPolarity

Derive polarity from seed using modular arithmetic.
This is the canonical Gay SPI polarity assignment.
"""
polarity_from_seed(seed::UInt64)::StreamPolarity = StreamPolarity(seed % 3)

# ═══════════════════════════════════════════════════════════════════════════════
# SPLITMIX64 PRNG (SPI Compliant)
# ═══════════════════════════════════════════════════════════════════════════════

@inline function sm64(s::UInt64)::Tuple{UInt64, UInt64}
    z = (s + GOLDEN) & 0xFFFFFFFFFFFFFFFF
    z = ((z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ⊻ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    (z ⊻ (z >> 31), s + 1)
end

@inline function sm64_split(s::UInt64)::Tuple{UInt64, UInt64}
    v1, s1 = sm64(s)
    v2, _ = sm64(s1)
    (v1 ⊻ v2, s1 + 1)
end

# ═══════════════════════════════════════════════════════════════════════════════
# STREAM SLOT: Individual stream with polarity assignment
# ═══════════════════════════════════════════════════════════════════════════════

"""
    StreamSlot

A QUIC stream with polarity-derived color and scheduling weight.
"""
struct StreamSlot
    stream_id::UInt64       # QUIC stream ID
    polarity::StreamPolarity
    seed::UInt64            # Derived seed for color
    color::NTuple{3, Float64}  # RGB color for visualization
    priority::Int           # Scheduling priority (0 = highest)
    created_at::UInt64      # Monotonic timestamp
end

function StreamSlot(stream_id::UInt64, base_seed::UInt64; 
                    priority::Int=1, created_at::UInt64=UInt64(0))
    # Derive stream-specific seed
    seed, _ = sm64(base_seed ⊻ stream_id)
    
    # Polarity from stream type (LSBs of stream ID)
    stream_type = UInt8(stream_id & 0x03)
    polarity = polarity_from_stream_type(stream_type)
    
    # Generate color from seed
    r, s1 = sm64(seed)
    g, s2 = sm64(s1)
    b, _ = sm64(s2)
    color = ((r >> 56) / 255.0, (g >> 56) / 255.0, (b >> 56) / 255.0)
    
    StreamSlot(stream_id, polarity, seed, color, priority, created_at)
end

stream_color(s::StreamSlot)::NTuple{3, Float64} = s.color
stream_seed(s::StreamSlot)::UInt64 = s.seed

# ═══════════════════════════════════════════════════════════════════════════════
# POLARITY SCHEDULER: Fair scheduling across polarities
# ═══════════════════════════════════════════════════════════════════════════════

"""
    PolarityScheduler

Schedules QUIC streams using polarity-based round-robin with adaptive weights.

# Scheduling Algorithm

The scheduler maintains three queues (one per polarity) and uses weighted
round-robin to ensure:

1. **Fairness**: Each polarity gets proportional scheduling time
2. **Responsiveness**: MINUS (ACK) streams get priority during congestion
3. **Throughput**: PLUS (expansion) streams get burst allowance
4. **Equilibrium**: ERGODIC streams maintain steady-state flow

## Weight Adaptation

Weights adapt based on:
- RTT measurements (high RTT → boost MINUS for faster ACKs)
- Congestion signals (ECN → throttle PLUS, boost MINUS)
- Stream count imbalance (rebalance toward underrepresented polarity)

## Seed Determinism

Given the same seed, the scheduler produces the same scheduling order,
enabling reproducible tests and debugging.
"""
mutable struct PolarityScheduler
    # Per-polarity queues
    minus_queue::Vector{StreamSlot}
    ergodic_queue::Vector{StreamSlot}
    plus_queue::Vector{StreamSlot}
    
    # Weights (must sum to 1.0)
    weights::NTuple{3, Float64}  # (MINUS, ERGODIC, PLUS)
    
    # Scheduling state
    seed::UInt64
    current_polarity::StreamPolarity
    tokens::NTuple{3, Float64}  # Token bucket per polarity
    
    # Metrics
    scheduled_count::NTuple{3, Int}
    last_rebalance::UInt64
end

function PolarityScheduler(seed::UInt64=GAY_SEED)
    # Initial weights: slightly favor ERGODIC (bidirectional = most common)
    weights = (0.25, 0.50, 0.25)  # MINUS, ERGODIC, PLUS
    
    PolarityScheduler(
        StreamSlot[], StreamSlot[], StreamSlot[],
        weights,
        seed,
        ERGODIC,  # Start with ergodic (bidirectional is most common)
        (0.0, 0.0, 0.0),
        (0, 0, 0),
        UInt64(0)
    )
end

"""
    add_stream!(scheduler, slot)

Add a stream to the appropriate polarity queue.
"""
function add_stream!(sched::PolarityScheduler, slot::StreamSlot)
    queue = if slot.polarity == MINUS
        sched.minus_queue
    elseif slot.polarity == ERGODIC
        sched.ergodic_queue
    else
        sched.plus_queue
    end
    
    # Insert sorted by priority (lower = higher priority)
    idx = searchsortedfirst(queue, slot, by=s->s.priority)
    insert!(queue, idx, slot)
    nothing
end

"""
    schedule_next!(scheduler) → Union{StreamSlot, Nothing}

Select the next stream to service using weighted round-robin.

Returns `nothing` if all queues are empty.
"""
function schedule_next!(sched::PolarityScheduler)::Union{StreamSlot, Nothing}
    queues = (sched.minus_queue, sched.ergodic_queue, sched.plus_queue)
    
    # Check if all empty
    if all(isempty, queues)
        return nothing
    end
    
    # Refill tokens based on weights
    tokens = MVector{3, Float64}(sched.tokens...)
    for i in 1:3
        tokens[i] += sched.weights[i]
    end
    
    # Find polarity with tokens and non-empty queue
    # Start from current_polarity and rotate
    start = Int(sched.current_polarity) + 1
    for offset in 0:2
        idx = ((start - 1 + offset) % 3) + 1
        if tokens[idx] >= 1.0 && !isempty(queues[idx])
            # Consume token and schedule
            tokens[idx] -= 1.0
            sched.tokens = NTuple{3, Float64}(tokens)
            sched.current_polarity = StreamPolarity(idx - 1)
            
            # Update metrics
            counts = MVector{3, Int}(sched.scheduled_count...)
            counts[idx] += 1
            sched.scheduled_count = NTuple{3, Int}(counts)
            
            # Pop from queue
            slot = popfirst!(queues[idx])
            return slot
        end
    end
    
    # Fallback: any non-empty queue
    for idx in 1:3
        if !isempty(queues[idx])
            sched.current_polarity = StreamPolarity(idx - 1)
            return popfirst!(queues[idx])
        end
    end
    
    nothing
end

"""
    rebalance!(scheduler, rtt_us, ecn_count)

Adapt weights based on network conditions.

- High RTT → boost MINUS (faster ACKs)
- ECN marks → throttle PLUS (reduce expansion)
"""
function rebalance!(sched::PolarityScheduler, rtt_us::Float64, ecn_count::Int)
    w = MVector{3, Float64}(sched.weights...)
    
    # RTT adaptation: high RTT → need more ACKs
    if rtt_us > 100_000  # > 100ms
        w[1] = min(0.40, w[1] + 0.05)  # Boost MINUS
        w[3] = max(0.15, w[3] - 0.05)  # Reduce PLUS
    elseif rtt_us < 20_000  # < 20ms
        w[1] = max(0.15, w[1] - 0.02)  # Reduce MINUS
        w[3] = min(0.35, w[3] + 0.02)  # Boost PLUS
    end
    
    # ECN adaptation: congestion → reduce expansion
    if ecn_count > 0
        scale = min(1.0, ecn_count / 10.0)
        w[3] = max(0.10, w[3] - 0.10 * scale)  # Reduce PLUS
        w[1] = min(0.50, w[1] + 0.05 * scale)  # Boost MINUS
    end
    
    # Normalize to sum to 1.0
    total = sum(w)
    w ./= total
    
    sched.weights = NTuple{3, Float64}(w)
    sched.last_rebalance += 1
    nothing
end

"""
    get_active_streams(scheduler) → NamedTuple

Return counts and stream lists per polarity.
"""
function get_active_streams(sched::PolarityScheduler)
    (
        minus = sched.minus_queue,
        ergodic = sched.ergodic_queue,
        plus = sched.plus_queue,
        counts = (
            minus = length(sched.minus_queue),
            ergodic = length(sched.ergodic_queue),
            plus = length(sched.plus_queue)
        ),
        weights = sched.weights,
        scheduled = sched.scheduled_count
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

"""
    verify_polarity_invariants() → Bool

Verify the polarity-to-stream mapping is consistent.
"""
function verify_polarity_invariants()::Bool
    # Test all stream types map correctly
    tests = [
        (0x00, ERGODIC, "Client bidi → ERGODIC"),
        (0x01, ERGODIC, "Server bidi → ERGODIC"),
        (0x02, PLUS, "Client uni → PLUS"),
        (0x03, MINUS, "Server uni → MINUS"),
    ]
    
    all_pass = true
    for (stream_type, expected, name) in tests
        actual = polarity_from_stream_type(UInt8(stream_type))
        if actual != expected
            @warn "Invariant violation" name expected actual
            all_pass = false
        end
    end
    
    # Verify seed-based polarity covers all cases
    for i in 0:2
        seed = UInt64(i)
        p = polarity_from_seed(seed)
        if Int(p) != i
            @warn "Seed polarity mismatch" seed expected=i actual=Int(p)
            all_pass = false
        end
    end
    
    # Verify polarity symbols
    if polarity_symbol(MINUS) != '−' ||
       polarity_symbol(ERGODIC) != '_' ||
       polarity_symbol(PLUS) != '+'
        @warn "Symbol mismatch"
        all_pass = false
    end
    
    all_pass
end

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

function __init__()
    if !verify_polarity_invariants()
        @error "QUICPolarityStreams: Polarity invariants failed!"
    end
end

end # module QUICPolarityStreams
