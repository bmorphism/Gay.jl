# GAY RADIO: GNU Radio & SDR Integration for Collaborative Incentive Discovery
# ═══════════════════════════════════════════════════════════════════════════════════
#
# "The radio waves carry chromatic seeds across space, enabling distributed
#  collaborative incentive discovery for autopoietic closure."
#
# ┌─────────────────────────────────────────────────────────────────────────────────┐
# │  PURPOSE                                                                        │
# │                                                                                 │
# │  GayRadio.jl integrates with GNU Radio and Software-Defined Radio (SDR) to:    │
# │    1. Encode/decode GayRNG seeds in RF signals                                 │
# │    2. Enable distributed collaborative incentive discovery                      │
# │    3. Achieve autopoietic closure via profinite ergodicity over all Worlds     │
# │    4. Form coalition formation games with NashProp over wireless               │
# │                                                                                 │
# │  AUTOPOIETIC CLOSURE                                                            │
# │    The cybernetic system sustains itself via:                                   │
# │    - Self-production: new seeds from received signals                          │
# │    - Boundary maintenance: only valid GayRNG sequences accepted                │
# │    - Environmental coupling: RF spectrum as shared resource                    │
# │                                                                                 │
# │  COLLABORATIVE INCENTIVE DISCOVERY                                              │
# │    Stations discover incentives to cooperate via:                               │
# │    - Chromatic handshakes: exchange of seed fingerprints                       │
# │    - Coalition formation: NashProp-guided grouping                             │
# │    - Profinite ergodicity: all stations eventually reachable                   │
# │                                                                                 │
# │  SDR INTEGRATION                                                                │
# │    - GR-GAY: GNU Radio block for GayRNG modulation                             │
# │    - RTLSDR/HackRF/USRP compatible                                             │
# │    - Frequency hopping via chromatic sequence                                   │
# │                                                                                 │
# └─────────────────────────────────────────────────────────────────────────────────┘

module GayRadio

using Dates: now, DateTime

export
    # Core types
    RadioStation, RadioChannel, RadioFrame, RadioNetwork,
    AutopoieticClosure, ClosureState,
    
    # Signal encoding
    encode_seed, decode_seed, encode_color, decode_color,
    frequency_from_seed, seed_from_frequency,
    
    # GNU Radio integration
    GRBlock, GRFlowgraph, GRSource, GRSink,
    create_gay_modulator, create_gay_demodulator,
    
    # Collaborative incentive discovery
    IncentiveDiscovery, DiscoveredIncentive, IncentiveType,
    discover_incentives!, evaluate_collaboration,
    
    # Coalition formation over radio
    RadioCoalition, form_coalitions!, broadcast_coalition,
    
    # Autopoietic closure
    autopoietic_step!, closure_maintained, entropy_repair,
    
    # Profinite ergodicity over worlds
    WorldNetwork, world_reachability, mixing_time_wireless,
    
    # Chromatic handshake protocol
    ChromaticHandshake, initiate_handshake, complete_handshake,
    
    # Demo
    demo_gay_radio

# ═══════════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════════

const GAY_SEED = UInt64(1069)
const RADIO_SEED = UInt64(0x2AD10)      # "RADIO" approximation
const AUTOPOIETIC_SEED = UInt64(0xA070) # "AUTO" + "PO"

# Radio parameters
const MIN_FREQ_MHZ = 70.0       # VHF lower bound
const MAX_FREQ_MHZ = 6000.0     # SHF upper bound
const FREQ_RANGE = MAX_FREQ_MHZ - MIN_FREQ_MHZ

# Frame structure
const FRAME_PREAMBLE = 0x6A7069  # "gay" in ASCII
const PREAMBLE_BITS = 24
const PAYLOAD_BITS = 64         # One UInt64 seed
const CRC_BITS = 16
const FRAME_BITS = PREAMBLE_BITS + PAYLOAD_BITS + CRC_BITS

# Incentive types
@enum IncentiveType begin
    SPECTRUM_SHARE = 1      # Share frequency allocation
    SEED_EXCHANGE = 2       # Exchange seeds for diversity
    COALITION_JOIN = 3      # Join coalition for strength
    ENTROPY_DONATE = 4      # Donate entropy for repair
    CLOSURE_ASSIST = 5      # Help maintain closure
end

# Closure states
@enum ClosureState begin
    OPEN = 0                # Not closed (external dependencies)
    PARTIAL = 1             # Partially closed
    CLOSED = 2              # Fully autopoietic
    SUSTAINING = 3          # Actively sustaining itself
end

# ═══════════════════════════════════════════════════════════════════════════════════
# CORE PRNG
# ═══════════════════════════════════════════════════════════════════════════════════

@inline function sm64(s::UInt64)::Tuple{UInt64, UInt64}
    z = (s + 0x9E3779B97F4A7C15)
    z = (z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ⊻ (z >> 27)) * 0x94D049BB133111EB
    (z ⊻ (z >> 31), s + 1)
end

@inline function color_from_seed(seed::UInt64)::NTuple{3, Float64}
    r, s1 = sm64(seed)
    g, s2 = sm64(s1)
    b, _  = sm64(s2)
    ((r >> 56) / 255.0, (g >> 56) / 255.0, (b >> 56) / 255.0)
end

# ═══════════════════════════════════════════════════════════════════════════════════
# SIGNAL ENCODING: Seed ↔ RF
# ═══════════════════════════════════════════════════════════════════════════════════

"""
Convert seed to frequency in MHz (for frequency hopping).
"""
function frequency_from_seed(seed::UInt64)::Float64
    normalized = (seed % 0xFFFFFFFF) / 0xFFFFFFFF
    MIN_FREQ_MHZ + normalized * FREQ_RANGE
end

"""
Approximate seed from frequency (lossy inverse).
"""
function seed_from_frequency(freq_mhz::Float64)::UInt64
    normalized = (freq_mhz - MIN_FREQ_MHZ) / FREQ_RANGE
    UInt64(round(normalized * 0xFFFFFFFF))
end

"""
Encode seed as IQ samples (complex baseband).
Uses BPSK with chromatic phase modulation.
"""
function encode_seed(seed::UInt64; samples_per_bit::Int=8)::Vector{ComplexF64}
    # Color determines phase offset
    color = color_from_seed(seed)
    phase_offset = 2π * (color[1] + color[2] + color[3]) / 3
    
    samples = ComplexF64[]
    
    for bit_pos in 63:-1:0
        bit = (seed >> bit_pos) & 1
        phase = phase_offset + (bit == 1 ? 0.0 : π)
        
        for _ in 1:samples_per_bit
            push!(samples, exp(im * phase))
        end
    end
    
    samples
end

"""
Decode IQ samples back to seed.
"""
function decode_seed(samples::Vector{ComplexF64}; samples_per_bit::Int=8)::UInt64
    n_bits = length(samples) ÷ samples_per_bit
    n_bits = min(64, n_bits)
    
    seed = UInt64(0)
    
    for bit_idx in 1:n_bits
        start_idx = (bit_idx - 1) * samples_per_bit + 1
        end_idx = min(start_idx + samples_per_bit - 1, length(samples))
        
        # Average phase
        avg_phase = angle(sum(samples[start_idx:end_idx]))
        
        # Decide bit (0 or π phase difference)
        bit = abs(avg_phase) < π/2 ? 1 : 0
        seed = (seed << 1) | bit
    end
    
    seed
end

"""
Encode color as amplitude/frequency/phase.
"""
function encode_color(color::NTuple{3, Float64})::NTuple{3, Float64}
    # R → amplitude, G → frequency offset, B → phase
    amplitude = 0.5 + 0.5 * color[1]
    freq_offset = (color[2] - 0.5) * 100.0  # ±50 Hz
    phase = 2π * color[3]
    (amplitude, freq_offset, phase)
end

"""
Decode from encoded parameters back to color.
"""
function decode_color(encoded::NTuple{3, Float64})::NTuple{3, Float64}
    r = (encoded[1] - 0.5) / 0.5
    g = encoded[2] / 100.0 + 0.5
    b = encoded[3] / (2π)
    (clamp(r, 0.0, 1.0), clamp(g, 0.0, 1.0), clamp(b, 0.0, 1.0))
end

# ═══════════════════════════════════════════════════════════════════════════════════
# RADIO FRAME STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════════

"""
    RadioFrame

A single frame of Gay radio transmission.
"""
struct RadioFrame
    preamble::UInt32            # Synchronization pattern
    payload_seed::UInt64        # Main payload: a GayRNG seed
    crc::UInt16                 # Error detection
    
    # Metadata
    source_station::UInt64
    timestamp::Float64
    frequency_mhz::Float64
    
    fingerprint::UInt64
end

function RadioFrame(seed::UInt64, source::UInt64; freq_mhz::Float64=0.0)
    freq = freq_mhz > 0 ? freq_mhz : frequency_from_seed(seed)
    crc = UInt16(seed ⊻ (seed >> 16) ⊻ (seed >> 32) ⊻ (seed >> 48))
    fp, _ = sm64(seed ⊻ source)
    
    RadioFrame(0x47415906, seed, crc, source, time(), freq, fp)  # "GAY\x06" in hex-ish
end

function validate_frame(frame::RadioFrame)::Bool
    expected_crc = UInt16(frame.payload_seed ⊻ 
                          (frame.payload_seed >> 16) ⊻ 
                          (frame.payload_seed >> 32) ⊻ 
                          (frame.payload_seed >> 48))
    frame.crc == expected_crc
end

# ═══════════════════════════════════════════════════════════════════════════════════
# RADIO STATION
# ═══════════════════════════════════════════════════════════════════════════════════

"""
    RadioStation

A station in the Gay radio network.
"""
mutable struct RadioStation
    id::UInt64
    name::String
    
    # Radio parameters
    tx_power_dbm::Float64
    rx_sensitivity_dbm::Float64
    current_freq_mhz::Float64
    
    # GayRNG state
    seed::UInt64
    color::NTuple{3, Float64}
    
    # Received seeds (entropy pool)
    received_seeds::Vector{UInt64}
    
    # Coalition membership
    coalition_id::Union{Int, Nothing}
    
    # Autopoietic state
    closure_state::ClosureState
    entropy_balance::Float64     # > 0 means generating, < 0 means consuming
    
    fingerprint::UInt64
end

function RadioStation(name::String; seed::UInt64=GAY_SEED, tx_power::Float64=20.0)
    id, _ = sm64(hash(name) ⊻ seed)
    color = color_from_seed(id)
    freq = frequency_from_seed(id)
    
    RadioStation(
        id, name,
        tx_power, -100.0, freq,
        seed, color,
        UInt64[],
        nothing,
        OPEN, 0.0,
        id
    )
end

"""
Transmit a frame from this station.
"""
function transmit!(station::RadioStation)::RadioFrame
    # Generate new seed
    new_seed, _ = sm64(station.seed ⊻ UInt64(round(time() * 1e6)))
    station.seed = new_seed
    station.color = color_from_seed(new_seed)
    station.current_freq_mhz = frequency_from_seed(new_seed)
    station.fingerprint ⊻= new_seed
    
    RadioFrame(new_seed, station.id; freq_mhz=station.current_freq_mhz)
end

"""
Receive a frame at this station.
"""
function receive!(station::RadioStation, frame::RadioFrame)::Bool
    if !validate_frame(frame)
        return false
    end
    
    # Add to entropy pool
    push!(station.received_seeds, frame.payload_seed)
    
    # Update entropy balance (receiving = gaining entropy)
    station.entropy_balance += 1.0
    
    # XOR into fingerprint for SPI
    station.fingerprint ⊻= frame.fingerprint
    
    true
end

# ═══════════════════════════════════════════════════════════════════════════════════
# GNU RADIO INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════════

"""
    GRBlock

Abstract GNU Radio block representation.
"""
abstract type GRBlock end

"""
    GRSource

Source block (transmitter side).
"""
struct GRSource <: GRBlock
    name::String
    sample_rate::Float64
    center_freq_mhz::Float64
    gain::Float64
    
    # Station reference
    station::RadioStation
end

"""
    GRSink

Sink block (receiver side).
"""
struct GRSink <: GRBlock
    name::String
    sample_rate::Float64
    center_freq_mhz::Float64
    sensitivity::Float64
    
    # Station reference
    station::RadioStation
end

"""
    GRFlowgraph

Complete GNU Radio flowgraph for Gay modulation/demodulation.
"""
struct GRFlowgraph
    name::String
    sources::Vector{GRSource}
    sinks::Vector{GRSink}
    connections::Vector{Tuple{GRBlock, GRBlock}}
    
    # Python/GRC representation
    python_code::String
    
    fingerprint::UInt64
end

"""
Create Gay modulator flowgraph (transmitter).
"""
function create_gay_modulator(station::RadioStation; sample_rate::Float64=2.4e6)::GRFlowgraph
    source = GRSource("gay_source", sample_rate, station.current_freq_mhz, 
                      station.tx_power_dbm, station)
    
    python_code = """
#!/usr/bin/env python3
# Gay Radio Modulator - Generated by GayRadio.jl
# Station: $(station.name)
# Seed: 0x$(string(station.seed, base=16))

from gnuradio import gr, blocks, analog
import numpy as np

class gay_modulator(gr.sync_block):
    def __init__(self, seed):
        gr.sync_block.__init__(self, "gay_modulator",
            in_sig=None, out_sig=[np.complex64])
        self.seed = seed
    
    def sm64(self, s):
        z = (s + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return (z ^ (z >> 31), (s + 1) & 0xFFFFFFFFFFFFFFFF)
    
    def work(self, input_items, output_items):
        out = output_items[0]
        for i in range(len(out)):
            val, self.seed = self.sm64(self.seed)
            # Map to QPSK constellation
            phase = (val % 4) * np.pi / 2
            out[i] = np.exp(1j * phase)
        return len(out)
"""
    
    fp = station.fingerprint ⊻ UInt64(round(sample_rate))
    
    GRFlowgraph("gay_modulator_$(station.name)", 
                [source], GRSink[], Tuple{GRBlock, GRBlock}[],
                python_code, fp)
end

"""
Create Gay demodulator flowgraph (receiver).
"""
function create_gay_demodulator(station::RadioStation; sample_rate::Float64=2.4e6)::GRFlowgraph
    sink = GRSink("gay_sink", sample_rate, station.current_freq_mhz,
                  station.rx_sensitivity_dbm, station)
    
    python_code = """
#!/usr/bin/env python3
# Gay Radio Demodulator - Generated by GayRadio.jl
# Station: $(station.name)

from gnuradio import gr, blocks
import numpy as np

class gay_demodulator(gr.sync_block):
    def __init__(self):
        gr.sync_block.__init__(self, "gay_demodulator",
            in_sig=[np.complex64], out_sig=None)
        self.received_seeds = []
    
    def work(self, input_items, output_items):
        inp = input_items[0]
        # Demodulate QPSK and reconstruct seed
        seed = 0
        for i in range(min(64, len(inp))):
            phase = np.angle(inp[i])
            symbol = int(round(phase / (np.pi / 2))) % 4
            seed = (seed << 2) | symbol
        self.received_seeds.append(seed)
        return len(inp)
"""
    
    fp = station.fingerprint ⊻ UInt64(round(sample_rate)) ⊻ 0xDE0D
    
    GRFlowgraph("gay_demodulator_$(station.name)",
                GRSource[], [sink], Tuple{GRBlock, GRBlock}[],
                python_code, fp)
end

# ═══════════════════════════════════════════════════════════════════════════════════
# COLLABORATIVE INCENTIVE DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════════════

"""
    DiscoveredIncentive

An incentive discovered between stations.
"""
struct DiscoveredIncentive
    incentive_type::IncentiveType
    stations::Vector{UInt64}        # Station IDs involved
    value::Float64                  # Quantified benefit
    discovered_at::DateTime
    fingerprint::UInt64
end

"""
    IncentiveDiscovery

State of collaborative incentive discovery.
"""
mutable struct IncentiveDiscovery
    discovered::Vector{DiscoveredIncentive}
    total_value::Float64
    collaboration_score::Float64
    
    # Per-type counts
    type_counts::Dict{IncentiveType, Int}
    
    fingerprint::UInt64
end

function IncentiveDiscovery()
    IncentiveDiscovery(
        DiscoveredIncentive[],
        0.0, 0.0,
        Dict{IncentiveType, Int}(),
        RADIO_SEED
    )
end

"""
Discover incentives between stations based on their states.
"""
function discover_incentives!(discovery::IncentiveDiscovery, 
                               stations::Vector{RadioStation})::Vector{DiscoveredIncentive}
    new_incentives = DiscoveredIncentive[]
    
    for i in 1:length(stations)
        for j in i+1:length(stations)
            s1, s2 = stations[i], stations[j]
            
            # Spectrum sharing incentive
            freq_dist = abs(s1.current_freq_mhz - s2.current_freq_mhz)
            if freq_dist < 10.0  # Close in frequency
                value = 10.0 - freq_dist
                fp = s1.fingerprint ⊻ s2.fingerprint ⊻ UInt64(Int(SPECTRUM_SHARE))
                push!(new_incentives, DiscoveredIncentive(
                    SPECTRUM_SHARE, [s1.id, s2.id], value, now(), fp
                ))
            end
            
            # Seed exchange incentive
            seed_dist = count_ones(s1.seed ⊻ s2.seed) / 64.0
            if 0.3 < seed_dist < 0.7  # Moderate Hamming distance = good diversity
                value = 1.0 - abs(seed_dist - 0.5) * 2
                fp = s1.fingerprint ⊻ s2.fingerprint ⊻ UInt64(Int(SEED_EXCHANGE))
                push!(new_incentives, DiscoveredIncentive(
                    SEED_EXCHANGE, [s1.id, s2.id], value, now(), fp
                ))
            end
            
            # Entropy donation incentive
            if s1.entropy_balance > 5 && s2.entropy_balance < -5
                value = min(s1.entropy_balance, -s2.entropy_balance)
                fp = s1.fingerprint ⊻ s2.fingerprint ⊻ UInt64(Int(ENTROPY_DONATE))
                push!(new_incentives, DiscoveredIncentive(
                    ENTROPY_DONATE, [s1.id, s2.id], value, now(), fp
                ))
            end
            
            # Closure assist incentive
            if s1.closure_state == SUSTAINING && s2.closure_state < CLOSED
                value = Float64(Int(SUSTAINING) - Int(s2.closure_state))
                fp = s1.fingerprint ⊻ s2.fingerprint ⊻ UInt64(Int(CLOSURE_ASSIST))
                push!(new_incentives, DiscoveredIncentive(
                    CLOSURE_ASSIST, [s1.id, s2.id], value, now(), fp
                ))
            end
        end
    end
    
    # Record incentives
    for incentive in new_incentives
        push!(discovery.discovered, incentive)
        discovery.total_value += incentive.value
        t = incentive.incentive_type
        discovery.type_counts[t] = get(discovery.type_counts, t, 0) + 1
        discovery.fingerprint ⊻= incentive.fingerprint
    end
    
    # Update collaboration score
    n_pairs = length(stations) * (length(stations) - 1) ÷ 2
    discovery.collaboration_score = length(new_incentives) / max(1, n_pairs)
    
    new_incentives
end

"""
Evaluate potential collaboration between two stations.
"""
function evaluate_collaboration(s1::RadioStation, s2::RadioStation)::NamedTuple
    # Compute various collaboration metrics
    freq_proximity = 1.0 / (1.0 + abs(s1.current_freq_mhz - s2.current_freq_mhz))
    seed_diversity = count_ones(s1.seed ⊻ s2.seed) / 64.0
    entropy_complementarity = s1.entropy_balance * s2.entropy_balance < 0 ? 
                               abs(s1.entropy_balance - s2.entropy_balance) / 10.0 : 0.0
    closure_alignment = Float64(min(Int(s1.closure_state), Int(s2.closure_state)))
    
    total = (freq_proximity + seed_diversity + entropy_complementarity + closure_alignment) / 4
    
    (
        freq_proximity = freq_proximity,
        seed_diversity = seed_diversity,
        entropy_complementarity = entropy_complementarity,
        closure_alignment = closure_alignment,
        collaboration_potential = total,
        recommended = total > 0.5,
    )
end

# ═══════════════════════════════════════════════════════════════════════════════════
# COALITION FORMATION OVER RADIO
# ═══════════════════════════════════════════════════════════════════════════════════

"""
    RadioCoalition

A coalition of radio stations.
"""
struct RadioCoalition
    id::Int
    members::Vector{UInt64}         # Station IDs
    leader::UInt64                  # Coalition leader
    
    # Coalition properties
    combined_power_dbm::Float64
    frequency_allocation::Vector{Float64}
    
    # NashProp value
    nash_value::Float64
    stable::Bool
    
    fingerprint::UInt64
end

"""
Form coalitions among stations using NashProp-inspired algorithm.
"""
function form_coalitions!(stations::Vector{RadioStation})::Vector{RadioCoalition}
    n = length(stations)
    coalitions = RadioCoalition[]
    
    # Compute pairwise collaboration potential
    potential = zeros(n, n)
    for i in 1:n
        for j in i+1:n
            eval = evaluate_collaboration(stations[i], stations[j])
            potential[i,j] = eval.collaboration_potential
            potential[j,i] = potential[i,j]
        end
    end
    
    # Greedy coalition formation
    assigned = zeros(Int, n)
    coalition_id = 0
    
    for i in 1:n
        if assigned[i] == 0
            coalition_id += 1
            members = [i]
            assigned[i] = coalition_id
            
            # Add compatible stations
            for j in i+1:n
                if assigned[j] == 0 && potential[i,j] > 0.5
                    push!(members, j)
                    assigned[j] = coalition_id
                end
            end
            
            # Update station coalition membership
            for m in members
                stations[m].coalition_id = coalition_id
            end
            
            # Compute coalition properties
            member_ids = [stations[m].id for m in members]
            leader = member_ids[1]
            
            combined_power = 10 * log10(sum(10^(stations[m].tx_power_dbm/10) for m in members))
            freq_alloc = [stations[m].current_freq_mhz for m in members]
            
            # Nash value: sum of internal potentials
            nash = sum(potential[members[i], members[j]] 
                      for i in 1:length(members) for j in i+1:length(members); init=0.0)
            
            fp = reduce(⊻, member_ids)
            
            push!(coalitions, RadioCoalition(
                coalition_id, member_ids, leader,
                combined_power, freq_alloc,
                nash, true, fp
            ))
        end
    end
    
    coalitions
end

"""
Broadcast coalition information.
"""
function broadcast_coalition(coalition::RadioCoalition, 
                              stations::Dict{UInt64, RadioStation})::Vector{RadioFrame}
    frames = RadioFrame[]
    
    # Leader broadcasts coalition seed (XOR of all members)
    coalition_seed = reduce(⊻, coalition.members)
    
    if haskey(stations, coalition.leader)
        leader = stations[coalition.leader]
        frame = RadioFrame(coalition_seed, leader.id; 
                          freq_mhz=leader.current_freq_mhz)
        push!(frames, frame)
    end
    
    frames
end

# ═══════════════════════════════════════════════════════════════════════════════════
# AUTOPOIETIC CLOSURE
# ═══════════════════════════════════════════════════════════════════════════════════

"""
    AutopoieticClosure

State of autopoietic closure for the radio network.
"""
mutable struct AutopoieticClosure
    # Network state
    stations::Vector{RadioStation}
    
    # Closure metrics
    current_state::ClosureState
    closure_ratio::Float64          # 0-1, how closed the system is
    
    # Production
    seeds_produced::Int
    seeds_consumed::Int
    production_rate::Float64
    
    # Boundary
    external_inputs::Int
    internal_recycles::Int
    boundary_strength::Float64
    
    # Sustainability
    sustainable::Bool
    sustainability_score::Float64
    
    fingerprint::UInt64
end

function AutopoieticClosure(stations::Vector{RadioStation})
    fp = reduce(⊻, [s.fingerprint for s in stations]; init=AUTOPOIETIC_SEED)
    
    AutopoieticClosure(
        stations,
        OPEN, 0.0,
        0, 0, 0.0,
        0, 0, 0.0,
        false, 0.0,
        fp
    )
end

"""
Perform one autopoietic step: produce, recycle, maintain boundary.
"""
function autopoietic_step!(closure::AutopoieticClosure)
    # 1. Production: each station produces a seed
    for station in closure.stations
        frame = transmit!(station)
        closure.seeds_produced += 1
    end
    
    # 2. Internal exchange: stations receive from each other
    for i in 1:length(closure.stations)
        for j in 1:length(closure.stations)
            if i != j
                frame = RadioFrame(closure.stations[j].seed, closure.stations[j].id)
                if receive!(closure.stations[i], frame)
                    closure.internal_recycles += 1
                end
            end
        end
    end
    
    # 3. Entropy balance update
    total_entropy = sum(s.entropy_balance for s in closure.stations)
    closure.production_rate = total_entropy / max(1, length(closure.stations))
    
    # 4. Boundary strength: ratio of internal to external
    closure.boundary_strength = closure.internal_recycles / 
                                max(1, closure.internal_recycles + closure.external_inputs)
    
    # 5. Closure ratio: function of boundary strength and production
    closure.closure_ratio = (closure.boundary_strength + min(1.0, closure.production_rate)) / 2
    
    # 6. Update state
    if closure.closure_ratio > 0.9 && closure.production_rate > 0
        closure.current_state = SUSTAINING
    elseif closure.closure_ratio > 0.7
        closure.current_state = CLOSED
    elseif closure.closure_ratio > 0.4
        closure.current_state = PARTIAL
    else
        closure.current_state = OPEN
    end
    
    # 7. Sustainability check
    closure.sustainable = closure.current_state >= CLOSED && closure.production_rate > 0.5
    closure.sustainability_score = closure.closure_ratio * min(1.0, closure.production_rate)
    
    # 8. Update fingerprint
    closure.fingerprint ⊻= reduce(⊻, [s.fingerprint for s in closure.stations])
    
    closure
end

"""
Check if closure is currently maintained.
"""
function closure_maintained(closure::AutopoieticClosure)::Bool
    closure.current_state >= CLOSED
end

"""
Repair entropy deficit by redistributing from surplus stations.
"""
function entropy_repair(closure::AutopoieticClosure)
    surplus = [s for s in closure.stations if s.entropy_balance > 5]
    deficit = [s for s in closure.stations if s.entropy_balance < -5]
    
    for d in deficit
        if !isempty(surplus)
            s = surplus[1]
            transfer = min(s.entropy_balance, -d.entropy_balance) / 2
            s.entropy_balance -= transfer
            d.entropy_balance += transfer
        end
    end
    
    # This is TikkunOlam for the radio network
    closure
end

# ═══════════════════════════════════════════════════════════════════════════════════
# WORLD NETWORK: Profinite Ergodicity
# ═══════════════════════════════════════════════════════════════════════════════════

"""
    WorldNetwork

Network of worlds connected via radio links.
"""
struct WorldNetwork
    worlds::Vector{AutopoieticClosure}      # Each closure is a "World"
    inter_world_links::Vector{Tuple{Int, Int}}
    
    # Reachability
    adjacency::Matrix{Bool}
    reachability::Matrix{Bool}              # Transitive closure
    
    # Mixing
    mixing_time::Float64
    ergodic::Bool
    
    fingerprint::UInt64
end

function WorldNetwork(closures::Vector{AutopoieticClosure})
    n = length(closures)
    
    # Build adjacency based on shared frequencies
    adj = zeros(Bool, n, n)
    links = Tuple{Int, Int}[]
    
    for i in 1:n
        for j in i+1:n
            # Worlds are adjacent if any stations can communicate
            for si in closures[i].stations
                for sj in closures[j].stations
                    if abs(si.current_freq_mhz - sj.current_freq_mhz) < 50
                        adj[i,j] = true
                        adj[j,i] = true
                        push!(links, (i, j))
                        break
                    end
                end
            end
        end
    end
    
    # Compute transitive closure (reachability)
    reach = copy(adj)
    for _ in 1:n  # Floyd-Warshall style
        reach = reach .| (reach * reach)
    end
    
    # Check ergodicity: all worlds reachable from all
    ergodic = all(reach[i,j] || i == j for i in 1:n for j in 1:n)
    
    # Mixing time estimate
    if ergodic && n > 1
        spectral_gap = sum(adj) / (n * (n-1))  # Proxy
        mixing_time = log(n) / max(0.1, spectral_gap)
    else
        mixing_time = Inf
    end
    
    fp = reduce(⊻, [c.fingerprint for c in closures])
    
    WorldNetwork(closures, links, adj, reach, mixing_time, ergodic, fp)
end

function world_reachability(network::WorldNetwork, from::Int, to::Int)::Bool
    1 <= from <= length(network.worlds) && 
    1 <= to <= length(network.worlds) &&
    (from == to || network.reachability[from, to])
end

function mixing_time_wireless(network::WorldNetwork)::Float64
    network.mixing_time
end

# ═══════════════════════════════════════════════════════════════════════════════════
# CHROMATIC HANDSHAKE PROTOCOL
# ═══════════════════════════════════════════════════════════════════════════════════

"""
    ChromaticHandshake

Protocol for establishing chromatic identity between stations.
"""
mutable struct ChromaticHandshake
    initiator::UInt64
    responder::UInt64
    
    # Exchange state
    initiator_seed::UInt64
    responder_seed::UInt64
    shared_seed::UInt64
    
    # Colors
    initiator_color::NTuple{3, Float64}
    responder_color::NTuple{3, Float64}
    blended_color::NTuple{3, Float64}
    
    # Status
    phase::Int                  # 0=init, 1=challenge, 2=response, 3=complete
    success::Bool
    
    fingerprint::UInt64
end

"""
Initiate a chromatic handshake.
"""
function initiate_handshake(initiator::RadioStation)::ChromaticHandshake
    color = color_from_seed(initiator.seed)
    
    ChromaticHandshake(
        initiator.id, UInt64(0),
        initiator.seed, UInt64(0), UInt64(0),
        color, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
        1, false,
        initiator.fingerprint
    )
end

"""
Complete the handshake from responder side.
"""
function complete_handshake(handshake::ChromaticHandshake, 
                            responder::RadioStation)::ChromaticHandshake
    handshake.responder = responder.id
    handshake.responder_seed = responder.seed
    handshake.responder_color = color_from_seed(responder.seed)
    
    # Compute shared seed via XOR
    handshake.shared_seed = handshake.initiator_seed ⊻ handshake.responder_seed
    
    # Blend colors
    ic = handshake.initiator_color
    rc = handshake.responder_color
    handshake.blended_color = (
        (ic[1] + rc[1]) / 2,
        (ic[2] + rc[2]) / 2,
        (ic[3] + rc[3]) / 2,
    )
    
    handshake.phase = 3
    handshake.success = handshake.shared_seed != 0
    handshake.fingerprint = handshake.initiator_seed ⊻ handshake.responder_seed ⊻ 
                            UInt64(round(sum(handshake.blended_color) * 1e9))
    
    handshake
end

# ═══════════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════════

function demo_gay_radio()
    println()
    println("╔═════════════════════════════════════════════════════════════════════════════╗")
    println("║  GAY RADIO: GNU Radio & SDR for Collaborative Incentive Discovery          ║")
    println("╠═════════════════════════════════════════════════════════════════════════════╣")
    println("║  Autopoietic closure via profinite ergodicity across all Worlds            ║")
    println("║  NashProp coalition formation over wireless                                ║")
    println("╚═════════════════════════════════════════════════════════════════════════════╝")
    println()
    
    # ─── Create Stations ───
    println("─── RADIO STATIONS ───")
    stations = [
        RadioStation("Alpha"; seed=GAY_SEED ⊻ 0x1, tx_power=20.0),
        RadioStation("Beta"; seed=GAY_SEED ⊻ 0x2, tx_power=25.0),
        RadioStation("Gamma"; seed=GAY_SEED ⊻ 0x3, tx_power=15.0),
        RadioStation("Delta"; seed=GAY_SEED ⊻ 0x4, tx_power=30.0),
        RadioStation("Epsilon"; seed=GAY_SEED ⊻ 0x5, tx_power=20.0),
    ]
    
    for s in stations
        println("  $(s.name): freq=$(round(s.current_freq_mhz, digits=1)) MHz, " *
                "power=$(s.tx_power_dbm) dBm, color=$(round.(s.color, digits=2))")
    end
    println()
    
    # ─── Signal Encoding ───
    println("─── SIGNAL ENCODING ───")
    test_seed = GAY_SEED
    samples = encode_seed(test_seed; samples_per_bit=4)
    decoded = decode_seed(samples; samples_per_bit=4)
    
    println("  Original seed: 0x$(string(test_seed, base=16))")
    println("  Encoded to: $(length(samples)) IQ samples")
    println("  Decoded seed: 0x$(string(decoded, base=16))")
    println("  Recovery: $(test_seed == decoded ? "EXACT" : "LOSSY")")
    println()
    
    # ─── Frame Transmission ───
    println("─── FRAME TRANSMISSION ───")
    frame = transmit!(stations[1])
    println("  Station Alpha transmitted:")
    println("    Payload seed: 0x$(string(frame.payload_seed, base=16))")
    println("    Frequency: $(round(frame.frequency_mhz, digits=1)) MHz")
    println("    Valid: $(validate_frame(frame))")
    
    received = receive!(stations[2], frame)
    println("  Station Beta received: $(received)")
    println("  Beta's entropy pool: $(length(stations[2].received_seeds)) seeds")
    println()
    
    # ─── GNU Radio Flowgraph ───
    println("─── GNU RADIO INTEGRATION ───")
    modulator = create_gay_modulator(stations[1])
    println("  Created modulator: $(modulator.name)")
    println("  Python code: $(length(modulator.python_code)) chars")
    println("  First 100 chars:")
    println("    $(first(modulator.python_code, 100))...")
    println()
    
    # ─── Collaborative Incentive Discovery ───
    println("─── COLLABORATIVE INCENTIVE DISCOVERY ───")
    discovery = IncentiveDiscovery()
    incentives = discover_incentives!(discovery, stations)
    
    println("  Discovered $(length(incentives)) incentives:")
    for inc in incentives[1:min(5, length(incentives))]
        println("    $(inc.incentive_type): value=$(round(inc.value, digits=2))")
    end
    println("  Total value: $(round(discovery.total_value, digits=2))")
    println("  Collaboration score: $(round(discovery.collaboration_score, digits=3))")
    println()
    
    # ─── Coalition Formation ───
    println("─── COALITION FORMATION (NashProp) ───")
    coalitions = form_coalitions!(stations)
    
    for c in coalitions
        member_names = [s.name for s in stations if s.id in c.members]
        println("  Coalition $(c.id): $(join(member_names, ", "))")
        println("    Combined power: $(round(c.combined_power_dbm, digits=1)) dBm")
        println("    Nash value: $(round(c.nash_value, digits=3))")
    end
    println()
    
    # ─── Autopoietic Closure ───
    println("─── AUTOPOIETIC CLOSURE ───")
    closure = AutopoieticClosure(stations)
    
    for i in 1:5
        autopoietic_step!(closure)
        println("  Step $i: state=$(closure.current_state), " *
                "ratio=$(round(closure.closure_ratio, digits=3)), " *
                "production=$(round(closure.production_rate, digits=2))")
    end
    
    entropy_repair(closure)
    println("  After entropy repair: sustainable=$(closure.sustainable)")
    println()
    
    # ─── World Network ───
    println("─── WORLD NETWORK (Profinite Ergodicity) ───")
    
    # Create multiple closures as "Worlds"
    world1 = AutopoieticClosure(stations[1:2])
    world2 = AutopoieticClosure(stations[3:4])
    world3 = AutopoieticClosure(stations[5:5])
    
    for w in [world1, world2, world3]
        autopoietic_step!(w)
    end
    
    network = WorldNetwork([world1, world2, world3])
    
    println("  Worlds: $(length(network.worlds))")
    println("  Inter-world links: $(length(network.inter_world_links))")
    println("  Ergodic: $(network.ergodic)")
    println("  Mixing time: $(round(network.mixing_time, digits=2))")
    
    for i in 1:3
        for j in 1:3
            if i < j
                reachable = world_reachability(network, i, j)
                println("    World $i → World $j: $(reachable ? "reachable" : "isolated")")
            end
        end
    end
    println()
    
    # ─── Chromatic Handshake ───
    println("─── CHROMATIC HANDSHAKE ───")
    handshake = initiate_handshake(stations[1])
    complete_handshake(handshake, stations[2])
    
    println("  Initiator: $(stations[1].name), color=$(round.(handshake.initiator_color, digits=2))")
    println("  Responder: $(stations[2].name), color=$(round.(handshake.responder_color, digits=2))")
    println("  Blended color: $(round.(handshake.blended_color, digits=2))")
    println("  Shared seed: 0x$(string(handshake.shared_seed, base=16))")
    println("  Success: $(handshake.success)")
    println()
    
    # ─── Summary ───
    println("═══════════════════════════════════════════════════════════════════════════════")
    println("  GAY RADIO CAPABILITIES:")
    println("    ✓ GayRNG seed encoding/decoding in IQ samples")
    println("    ✓ GNU Radio flowgraph generation (Python)")
    println("    ✓ Collaborative incentive discovery (5 types)")
    println("    ✓ NashProp coalition formation over wireless")
    println("    ✓ Autopoietic closure with entropy repair (TikkunOlam)")
    println("    ✓ World network with profinite ergodicity")
    println("    ✓ Chromatic handshake protocol")
    println("═══════════════════════════════════════════════════════════════════════════════")
    
    (stations=stations, discovery=discovery, coalitions=coalitions,
     closure=closure, network=network, handshake=handshake)
end

end # module GayRadio
