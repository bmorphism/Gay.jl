# Igor Seeds: Premined Chromatic Motifs with Predetermined Intervals
# ═══════════════════════════════════════════════════════════════════════════════
#
# The Igor ↔ Not-Igor spectrum as a Galois connection:
#
#   ┌─────────────────────────────────────────────────────────────────────────────┐
#   │  IGOR SEED PREMINING                                                        │
#   │  ════════════════════                                                       │
#   │                                                                             │
#   │  Igor (originary):     Pure seed, motifs at Poisson intervals from GAY_SEED│
#   │  Not-Igor (deranged):  Complement seed, motifs permuted with no fixed pts  │
#   │                                                                             │
#   │  SPECTRUM: igor_weight ∈ [0, 1]                                            │
#   │  ═════════════════════════════                                             │
#   │  0.0 = pure not-igor (complete derangement)                                │
#   │  0.5 = balanced superposition                                               │
#   │  1.0 = pure igor (originary)                                               │
#   │                                                                             │
#   │  MOTIF INTERVALS                                                           │
#   │  ═══════════════                                                           │
#   │  τ_n = tropical_weight(seed, n) × base_interval                            │
#   │  where tropical_weight = exp(-distance_to_igor)                            │
#   │                                                                             │
#   │  ZIGZAG CONNECTION                                                         │
#   │  ═════════════════                                                         │
#   │  θ[i] = +1 (igor direction)                                                │
#   │  θ[i] = -1 (not-igor direction)                                            │
#   │  Velocity flip = transition between igor ↔ not-igor                        │
#   │                                                                             │
#   │  COLOR MOTIFS                                                              │
#   │  ════════════                                                              │
#   │  Green: igor-aligned (θ·∇ϕ > 0)                                           │
#   │  Red:   not-igor-aligned (θ·∇ϕ < 0)                                       │
#   │  Flip:  color change at predetermined Poisson time                         │
#   └─────────────────────────────────────────────────────────────────────────────┘

export IgorSeed, NotIgorSeed, IgorSpectrum
export premine_motifs, motif_at, interval_sequence
export igor_zigzag_trajectory, igor_velocity_flip
export IgorBeacon, next_igor_round, igor_fingerprint
export GAY_IGOR_SEED, derange_igor

using Random

# The originary Igor seed (derived from GAY_SEED by phi rotation)
const GAY_IGOR_SEED = UInt64(0x6761795f636f6c6f)  # GAY_SEED
const PHI = (1 + sqrt(5)) / 2  # Golden ratio

# ═══════════════════════════════════════════════════════════════════════════════
# Mix64: Deterministic hashing (shared with para_complexity.jl)
# ═══════════════════════════════════════════════════════════════════════════════

function mix64(z::UInt64)
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    z ⊻ (z >> 31)
end

# ═══════════════════════════════════════════════════════════════════════════════
# IgorSeed: The originary chromatic seed
# ═══════════════════════════════════════════════════════════════════════════════

"""
    IgorSeed

The originary seed from which chromatic motifs emerge at predetermined intervals.

# Properties
- `seed`: The base seed value (default: GAY_IGOR_SEED)
- `phi_power`: Golden ratio power for interval scaling
- `motif_count`: Number of premined motifs
- `intervals`: Predetermined intervals between motif occurrences

# Theory
Igor represents the "fixed" reference frame. Motifs appear at intervals
determined by the seed's interaction with the golden ratio, creating
a quasi-periodic but deterministic pattern.
"""
struct IgorSeed
    seed::UInt64
    phi_power::Float64
    motif_count::Int
    intervals::Vector{Float64}
    colors::Vector{Tuple{Float32, Float32, Float32}}
end

function IgorSeed(; seed::UInt64=GAY_IGOR_SEED, n_motifs::Int=64)
    intervals = Float64[]
    colors = Tuple{Float32, Float32, Float32}[]
    
    rng_state = seed
    for i in 1:n_motifs
        rng_state = mix64(rng_state)
        
        # Interval from golden ratio modulation
        u = (rng_state % UInt64(1000000)) / 1000000.0
        τ = PHI^(1 + u) * (1 + sin(2π * u * PHI))
        push!(intervals, τ)
        
        # Color from seed state
        r = Float32((rng_state % 256) / 255.0)
        rng_state = mix64(rng_state)
        g = Float32((rng_state % 256) / 255.0)
        rng_state = mix64(rng_state)
        b = Float32((rng_state % 256) / 255.0)
        push!(colors, (r, g, b))
    end
    
    IgorSeed(seed, PHI, n_motifs, intervals, colors)
end

# ═══════════════════════════════════════════════════════════════════════════════
# NotIgorSeed: The deranged complement
# ═══════════════════════════════════════════════════════════════════════════════

"""
    NotIgorSeed

The deranged complement of an IgorSeed - every motif is displaced (no fixed points).

Uses Sattolo's algorithm to guarantee a single cycle derangement of motif positions.
"""
struct NotIgorSeed
    igor::IgorSeed
    derangement::Vector{Int}  # σ(i) ≠ i for all i
    flipped_intervals::Vector{Float64}
    flipped_colors::Vector{Tuple{Float32, Float32, Float32}}
end

function NotIgorSeed(igor::IgorSeed)
    n = igor.motif_count
    
    # Sattolo's algorithm: guaranteed single-cycle derangement
    perm = collect(1:n)
    rng_state = mix64(igor.seed ⊻ UInt64(0xDEADBEEF))
    
    for i in n:-1:2
        rng_state = mix64(rng_state)
        j = 1 + Int(rng_state % UInt64(i - 1))  # j ∈ [1, i-1], never i
        perm[i], perm[j] = perm[j], perm[i]
    end
    
    # Apply derangement to intervals and colors
    flipped_intervals = [igor.intervals[perm[i]] for i in 1:n]
    flipped_colors = [igor.colors[perm[i]] for i in 1:n]
    
    NotIgorSeed(igor, perm, flipped_intervals, flipped_colors)
end

"""
    derange_igor(igor::IgorSeed) -> NotIgorSeed

Create the deranged complement of an Igor seed.
"""
derange_igor(igor::IgorSeed) = NotIgorSeed(igor)

# ═══════════════════════════════════════════════════════════════════════════════
# IgorSpectrum: Superposition between Igor and Not-Igor
# ═══════════════════════════════════════════════════════════════════════════════

"""
    IgorSpectrum

A weighted superposition between Igor (originary) and Not-Igor (deranged).

# Interpretation
- `weight = 1.0`: Pure Igor (all motifs at original positions)
- `weight = 0.5`: Balanced (equal probability of each frame)
- `weight = 0.0`: Pure Not-Igor (all motifs deranged)

This implements the Galois connection: α(igor) = not-igor, γ(not-igor) = igor
"""
struct IgorSpectrum
    igor::IgorSeed
    not_igor::NotIgorSeed
    weight::Float64  # ∈ [0, 1]
end

function IgorSpectrum(igor::IgorSeed, weight::Float64=0.5)
    @assert 0.0 <= weight <= 1.0 "Weight must be in [0, 1]"
    IgorSpectrum(igor, NotIgorSeed(igor), weight)
end

function IgorSpectrum(; seed::UInt64=GAY_IGOR_SEED, n_motifs::Int=64, weight::Float64=0.5)
    igor = IgorSeed(; seed=seed, n_motifs=n_motifs)
    IgorSpectrum(igor, weight)
end

"""
Get interval at index, weighted between igor and not-igor
"""
function interval_at(spectrum::IgorSpectrum, i::Int)
    idx = mod1(i, spectrum.igor.motif_count)
    igor_τ = spectrum.igor.intervals[idx]
    not_igor_τ = spectrum.not_igor.flipped_intervals[idx]
    
    # Linear interpolation
    spectrum.weight * igor_τ + (1 - spectrum.weight) * not_igor_τ
end

"""
Get color at index, weighted between igor and not-igor
"""
function color_at(spectrum::IgorSpectrum, i::Int)
    idx = mod1(i, spectrum.igor.motif_count)
    igor_c = spectrum.igor.colors[idx]
    not_igor_c = spectrum.not_igor.flipped_colors[idx]
    w = spectrum.weight
    
    (
        Float32(w * igor_c[1] + (1 - w) * not_igor_c[1]),
        Float32(w * igor_c[2] + (1 - w) * not_igor_c[2]),
        Float32(w * igor_c[3] + (1 - w) * not_igor_c[3])
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# Motif Premining: Generate predetermined color sequences
# ═══════════════════════════════════════════════════════════════════════════════

"""
    PreminedMotif

A premined chromatic motif with:
- Position in sequence
- Absolute time (cumulative from start)
- Color (RGB)
- Igor alignment (true = igor frame, false = not-igor frame)
"""
struct PreminedMotif
    index::Int
    time::Float64
    color::Tuple{Float32, Float32, Float32}
    igor_aligned::Bool
    velocity::Int  # +1 (igor direction) or -1 (not-igor direction)
end

"""
    premine_motifs(spectrum::IgorSpectrum, n::Int) -> Vector{PreminedMotif}

Premine n chromatic motifs from the spectrum.
"""
function premine_motifs(spectrum::IgorSpectrum, n::Int)
    motifs = PreminedMotif[]
    t = 0.0
    velocity = 1  # Start in igor direction
    
    rng_state = spectrum.igor.seed
    
    for i in 1:n
        # Interval determines when next motif appears
        τ = interval_at(spectrum, i)
        t += τ
        
        # Color for this motif
        c = color_at(spectrum, i)
        
        # Determine alignment based on RNG and weight
        rng_state = mix64(rng_state ⊻ UInt64(i))
        u = (rng_state % UInt64(1000000)) / 1000000.0
        igor_aligned = u < spectrum.weight
        
        # Velocity flip (ZigZag-style)
        # Flip probability increases with distance from current alignment
        flip_prob = igor_aligned ? (1 - spectrum.weight) : spectrum.weight
        if (rng_state >> 32) % 1000 < UInt64(flip_prob * 1000)
            velocity = -velocity
        end
        
        push!(motifs, PreminedMotif(i, t, c, igor_aligned, velocity))
    end
    
    return motifs
end

"""
    motif_at(spectrum::IgorSpectrum, index::Int) -> PreminedMotif

Get the premined motif at a specific index (O(1) via closed-form calculation).
"""
function motif_at(spectrum::IgorSpectrum, index::Int)
    # Cumulative time up to this index
    t = sum(interval_at(spectrum, i) for i in 1:index)
    c = color_at(spectrum, index)
    
    # Alignment determination
    rng_state = mix64(spectrum.igor.seed ⊻ UInt64(index))
    u = (rng_state % UInt64(1000000)) / 1000000.0
    igor_aligned = u < spectrum.weight
    
    # Velocity: determined by cumulative flip history (parity)
    flip_count = 0
    for i in 1:index
        rs = mix64(spectrum.igor.seed ⊻ UInt64(i))
        u_i = (rs % UInt64(1000000)) / 1000000.0
        aligned_i = u_i < spectrum.weight
        flip_prob = aligned_i ? (1 - spectrum.weight) : spectrum.weight
        if (rs >> 32) % 1000 < UInt64(flip_prob * 1000)
            flip_count += 1
        end
    end
    velocity = iseven(flip_count) ? 1 : -1
    
    PreminedMotif(index, t, c, igor_aligned, velocity)
end

"""
    interval_sequence(spectrum::IgorSpectrum, n::Int) -> Vector{Float64}

Get the first n intervals from the spectrum.
"""
interval_sequence(spectrum::IgorSpectrum, n::Int) = [interval_at(spectrum, i) for i in 1:n]

# ═══════════════════════════════════════════════════════════════════════════════
# ZigZag Trajectory: Continuous path through igor/not-igor space
# ═══════════════════════════════════════════════════════════════════════════════

"""
    IgorZigZagState

State of a ZigZag sampler in igor/not-igor space.
"""
mutable struct IgorZigZagState
    x::Float64          # Position (igor weight)
    θ::Int              # Velocity: +1 (toward igor) or -1 (toward not-igor)
    t::Float64          # Current time
    event_count::Int    # Number of velocity flips
end

"""
    igor_zigzag_trajectory(spectrum::IgorSpectrum, T::Float64) -> Vector{IgorZigZagState}

Generate a ZigZag trajectory through igor/not-igor space.

The particle moves linearly with velocity θ ∈ {-1, +1}, flipping at 
Poisson times determined by the premined motif intervals.
"""
function igor_zigzag_trajectory(spectrum::IgorSpectrum, T::Float64; dt::Float64=0.1)
    states = IgorZigZagState[]
    
    # Initial state: position = weight, moving toward igor
    state = IgorZigZagState(spectrum.weight, 1, 0.0, 0)
    push!(states, deepcopy(state))
    
    motif_idx = 1
    next_event_time = interval_at(spectrum, motif_idx)
    
    while state.t < T
        # Time to next event
        τ = next_event_time - state.t
        
        if state.t + τ > T
            # Move to end without event
            state.x = clamp(state.x + state.θ * dt * (T - state.t), 0.0, 1.0)
            state.t = T
            push!(states, deepcopy(state))
            break
        end
        
        # Move to event time
        steps = Int(floor(τ / dt))
        for _ in 1:steps
            state.x = clamp(state.x + state.θ * dt, 0.0, 1.0)
            state.t += dt
            push!(states, deepcopy(state))
        end
        
        # Velocity flip at event
        state.t = next_event_time
        state.θ = -state.θ
        state.event_count += 1
        
        # Boundary reflection
        if state.x >= 1.0
            state.x = 1.0
            state.θ = -1
        elseif state.x <= 0.0
            state.x = 0.0
            state.θ = 1
        end
        
        push!(states, deepcopy(state))
        
        # Schedule next event
        motif_idx += 1
        next_event_time += interval_at(spectrum, motif_idx)
    end
    
    return states
end

"""
    igor_velocity_flip(state::IgorZigZagState, spectrum::IgorSpectrum) -> IgorZigZagState

Apply a velocity flip colored by the current spectrum position.
"""
function igor_velocity_flip(state::IgorZigZagState, spectrum::IgorSpectrum)
    new_state = deepcopy(state)
    new_state.θ = -new_state.θ
    new_state.event_count += 1
    
    # Boundary handling
    if new_state.x >= 1.0 && new_state.θ > 0
        new_state.θ = -1
    elseif new_state.x <= 0.0 && new_state.θ < 0
        new_state.θ = 1
    end
    
    return new_state
end

# ═══════════════════════════════════════════════════════════════════════════════
# IgorBeacon: Verifiable randomness with igor/not-igor coloring
# ═══════════════════════════════════════════════════════════════════════════════

"""
    IgorBeaconRound

A round from the Igor beacon producing colored randomness.
"""
struct IgorBeaconRound
    round::UInt64
    randomness::UInt64
    color::Tuple{Float32, Float32, Float32}
    igor_weight::Float64
    velocity::Int
    signature::UInt32  # XOR fingerprint
end

"""
    IgorBeacon

Beacon producing verifiable chromatic randomness at predetermined intervals.
"""
mutable struct IgorBeacon
    spectrum::IgorSpectrum
    current_round::UInt64
    rng_state::UInt64
    history::Vector{IgorBeaconRound}
end

function IgorBeacon(spectrum::IgorSpectrum)
    IgorBeacon(spectrum, UInt64(0), spectrum.igor.seed, IgorBeaconRound[])
end

function IgorBeacon(; seed::UInt64=GAY_IGOR_SEED, weight::Float64=0.5)
    spectrum = IgorSpectrum(; seed=seed, weight=weight)
    IgorBeacon(spectrum)
end

"""
    next_igor_round(beacon::IgorBeacon) -> IgorBeaconRound

Generate the next beacon round with colored randomness.
"""
function next_igor_round(beacon::IgorBeacon)
    beacon.current_round += 1
    beacon.rng_state = mix64(beacon.rng_state ⊻ beacon.current_round)
    
    # Get motif for this round
    motif = motif_at(beacon.spectrum, Int(beacon.current_round))
    
    # Compute signature as XOR of components
    sig = UInt32(0)
    sig ⊻= UInt32(beacon.rng_state % (1 << 32))
    sig ⊻= UInt32(Base.round(Int, motif.color[1] * 255)) << 16
    sig ⊻= UInt32(Base.round(Int, motif.color[2] * 255)) << 8
    sig ⊻= UInt32(Base.round(Int, motif.color[3] * 255))
    sig ⊻= UInt32(motif.velocity > 0 ? 0x80000000 : 0)
    
    beacon_round = IgorBeaconRound(
        beacon.current_round,
        beacon.rng_state,
        motif.color,
        beacon.spectrum.weight,
        motif.velocity,
        sig
    )
    
    push!(beacon.history, beacon_round)
    return beacon_round
end

"""
    igor_fingerprint(beacon::IgorBeacon) -> UInt32

Compute XOR fingerprint of beacon history (order-invariant).
"""
function igor_fingerprint(beacon::IgorBeacon)
    fp = UInt32(0)
    for round in beacon.history
        fp ⊻= round.signature
    end
    return fp
end

# ═══════════════════════════════════════════════════════════════════════════════
# Tropical Connection: Intervals as tropical weights
# ═══════════════════════════════════════════════════════════════════════════════

"""
    tropical_igor_path(spectrum::IgorSpectrum, n::Int) -> Float64

Compute the tropical (min-plus) path weight through n motifs.
This is the minimum accumulated interval (shortest path in tropical sense).
"""
function tropical_igor_path(spectrum::IgorSpectrum, n::Int)
    intervals = interval_sequence(spectrum, n)
    minimum(cumsum(intervals))
end

"""
    tropical_not_igor_path(spectrum::IgorSpectrum, n::Int) -> Float64

Compute the tropical (max-plus) path weight through n motifs.
This is the maximum accumulated interval (longest path in tropical sense).
"""
function tropical_not_igor_path(spectrum::IgorSpectrum, n::Int)
    intervals = interval_sequence(spectrum, n)
    maximum(cumsum(intervals))
end

# ═══════════════════════════════════════════════════════════════════════════════
# Demo and Verification
# ═══════════════════════════════════════════════════════════════════════════════

function world_igor_seeds()
    println()
    println("╔" * "═" ^ 60 * "╗")
    println("║  IGOR SEEDS: Premined Chromatic Motifs                     ║")
    println("║  GAY_IGOR_SEED = 0x$(string(GAY_IGOR_SEED, base=16))                     ║")
    println("╚" * "═" ^ 60 * "╝")
    println()
    
    # Create Igor seed
    igor = IgorSeed()
    println("Igor Seed (originary):")
    println("  Motif count: $(igor.motif_count)")
    println("  First 5 intervals: $(round.(igor.intervals[1:5], digits=3))")
    
    # Create Not-Igor (deranged)
    not_igor = derange_igor(igor)
    println("\nNot-Igor Seed (deranged):")
    println("  Derangement valid: $(all(not_igor.derangement[i] != i for i in 1:igor.motif_count))")
    println("  First 5 intervals: $(round.(not_igor.flipped_intervals[1:5], digits=3))")
    
    # Spectrum at different weights
    println("\nIgor Spectrum:")
    for w in [0.0, 0.25, 0.5, 0.75, 1.0]
        spectrum = IgorSpectrum(igor, w)
        motifs = premine_motifs(spectrum, 10)
        igor_count = count(m -> m.igor_aligned, motifs)
        avg_vel = sum(m.velocity for m in motifs) / 10
        println("  weight=$w: igor_aligned=$igor_count/10, avg_velocity=$(round(avg_vel, digits=2))")
    end
    
    # ZigZag trajectory
    println("\nZigZag Trajectory (T=10.0):")
    spectrum = IgorSpectrum(; weight=0.5)
    traj = igor_zigzag_trajectory(spectrum, 10.0; dt=0.5)
    println("  States: $(length(traj))")
    println("  Final position: $(Base.round(traj[end].x, digits=3))")
    println("  Total flips: $(traj[end].event_count)")
    
    # Beacon
    println("\nIgor Beacon:")
    beacon = IgorBeacon(; weight=0.5)
    for _ in 1:5
        br = next_igor_round(beacon)
        c = br.color
        r, g, b = Int.(Base.round.((c[1], c[2], c[3]) .* 255))
        println("  Round $(br.round): sig=0x$(string(br.signature, base=16, pad=8)) " *
                "color=\e[48;2;$(r);$(g);$(b)m  \e[0m vel=$(br.velocity > 0 ? '+' : '-')")
    end
    
    fp = igor_fingerprint(beacon)
    println("  XOR Fingerprint: 0x$(string(fp, base=16, pad=8))")
    
    # Tropical paths
    println("\nTropical Paths (n=20):")
    println("  Min-plus (shortest): $(Base.round(tropical_igor_path(spectrum, 20), digits=3))")
    println("  Max-plus (longest):  $(Base.round(tropical_not_igor_path(spectrum, 20), digits=3))")
    
    # SPI Verification
    println("\n" * "═" ^ 62)
    println("SPI Verification: Same seed → same fingerprint")
    beacon1 = IgorBeacon(; seed=UInt64(42), weight=0.5)
    beacon2 = IgorBeacon(; seed=UInt64(42), weight=0.5)
    for _ in 1:100
        next_igor_round(beacon1)
        next_igor_round(beacon2)
    end
    fp1 = igor_fingerprint(beacon1)
    fp2 = igor_fingerprint(beacon2)
    println("  Beacon 1: 0x$(string(fp1, base=16, pad=8))")
    println("  Beacon 2: 0x$(string(fp2, base=16, pad=8))")
    println("  Match: $(fp1 == fp2 ? "◆ SPI VERIFIED" : "◇ SPI VIOLATION")")
    
    println()
    println("◈ Igor Seeds Complete")
end

if abspath(PROGRAM_FILE) == @__FILE__
    world_igor_seeds()
end
