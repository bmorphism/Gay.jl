# Color Entropy Sources — physical measurements that generate trits
#
# Each source reads from the world and produces a ColoredTick:
# a TritTick carrying both temporal position AND measured entropy.
# The entropy feeds into the SplitMix64 chain as additional mixing,
# so colors are influenced by physical reality, not just the seed.
#
# Sources are classified by mortality:
#   Immortal   — always running (drand, git, energy monitor)
#   SemiMortal — degrades over time (air quality sensor)
#   Mortal     — requires a living body (EEG, heartbeat)
#
# The conservation law doesn't care which kind generated the trit.
# It only asks: does the sum balance?
#
# See: RESEARCH_AGENDA.md §VI

module EntropySources

using ..Gay: TritTick, TickSource, EPOCH_1_HZ, trit, splitmix64, LogicalTicks
import ..Gay: current_tick

export ColoredTick, Mortality, EntropySource
export Immortal, SemiMortal, Mortal
export classify_trit, entropy_mix
export DrandSource, EnergySource, BCISource, HeartbeatSource
export GitSource, AirQualitySource, AccelerometerSource
export CompositeSource
export CompositeTicks, current_colored_tick
export source_name, source_mortality, source_trustworthiness
export read_entropy, inject_watts!, inject_entropy!
export inject_heartbeat!, inject_aqi!, inject_commit!, inject_jerk!
export agreement, disagreement_signal, composite_from_readings

# ═══════════════════════════════════════════════════════════════════════════
# Mortality classification
# ═══════════════════════════════════════════════════════════════════════════

"""
    Mortality

Classification of entropy source lifespan.
- `Immortal`: never stops (drand, git, power outlet)
- `SemiMortal`: degrades (air sensor, battery device)
- `Mortal`: stops when the body stops (EEG, heartbeat)
"""
@enum Mortality begin
    Immortal
    SemiMortal
    Mortal
end

# ═══════════════════════════════════════════════════════════════════════════
# ColoredTick: a timestamped source observation with a compatibility payload
# ═══════════════════════════════════════════════════════════════════════════

"""
    ColoredTick

A timestamped source observation. The historical field name `entropy` is
retained for compatibility, but its `UInt64` value is an opaque sample word;
it carries no min-entropy estimate or cryptographic guarantee.

- `tick`: temporal position in the trit-tick grid
- `measured_trit`: GF(3) trit from the physical measurement (not from time)
- `entropy`: opaque 64-bit sample word (compatibility name)
- `confidence`: application-supplied weight in 0.0–1.0, not a probability
- `source`: representation label; a typed external referent carries identity
"""
struct ColoredTick
    tick::TritTick
    measured_trit::Int8            # -1, 0, +1 from the physical measurement
    entropy::UInt64                # opaque sample word; not an entropy estimate
    confidence::Float32            # policy weight; not calibrated certainty
    source::Symbol                 # :drand, :energy, :bci, :heartbeat, etc.
end

function Base.show(io::IO, ct::ColoredTick)
    role = ct.measured_trit == 1 ? "maker" : ct.measured_trit == 0 ? "coordinator" : "checker"
    print(io, "ColoredTick($(ct.source), trit=$(ct.measured_trit)/$(role), ",
          "conf=$(round(ct.confidence, digits=2)), tick=$(ct.tick.tick))")
end

"""
    entropy_mix(seed::UInt64, ct::ColoredTick) -> UInt64

Deterministically mix a `ColoredTick` sample word and label into a seed.
This is a representation transform, not an extractor, identity proof, or
cryptographic random-bit generator.
"""
function entropy_mix(seed::UInt64, ct::ColoredTick)
    # XOR entropy with seed, then finalize with SplitMix64
    mixed = xor(seed, ct.entropy)
    mixed = xor(mixed, UInt64(ct.measured_trit + 2) << 62)  # trit in top bits
    splitmix64(mixed)
end

# ═══════════════════════════════════════════════════════════════════════════
# Abstract EntropySource
# ═══════════════════════════════════════════════════════════════════════════

"""
    EntropySource

Compatibility interface for observation providers. Subtypes implement:
- `read_entropy(source) -> ColoredTick`
- `source_name(source) -> Symbol`
- `source_mortality(source) -> Mortality`
- `source_trustworthiness(source) -> Float32` (0.0 to 1.0)
"""
abstract type EntropySource end

# ═══════════════════════════════════════════════════════════════════════════
# GF(3) classification helpers
# ═══════════════════════════════════════════════════════════════════════════

"""
    classify_trit(value::Real, low_threshold::Real, high_threshold::Real) -> Int8

Classify a measurement into GF(3):
- value > high_threshold → +1 (maker/generator)
- low_threshold ≤ value ≤ high_threshold → 0 (coordinator/ergodic)
- value < low_threshold → -1 (checker/validator)
"""
function classify_trit(value::Real, low::Real, high::Real)::Int8
    value > high ? Int8(1) : value < low ? Int8(-1) : Int8(0)
end

# ═══════════════════════════════════════════════════════════════════════════
# Source 0: deterministic drand-shaped simulator
# ═══════════════════════════════════════════════════════════════════════════

"""
    DrandSource <: EntropySource

Local simulator with the timing shape of a drand beacon. It does not perform
HTTP retrieval or BLS verification and must not be treated as live drand
randomness.

1 drand round = 30 seconds = 30 × 141,120,000 = 4,233,600,000 trit-ticks.

The `trustworthiness` and `mortality` values below are application policy
metadata, not measured reliability or availability.
"""
mutable struct DrandSource <: EntropySource
    last_round::UInt64             # most recent drand round number
    last_randomness::UInt64        # truncated randomness from that round
    endpoint::String               # drand HTTP endpoint
end

DrandSource() = DrandSource(UInt64(0), UInt64(0), "https://drand.cloudflare.com")

const DRAND_TICKS_PER_ROUND = UInt64(30) * EPOCH_1_HZ  # 4,233,600,000

source_name(::DrandSource) = :drand
source_mortality(::DrandSource) = Immortal
source_trustworthiness(::DrandSource) = Float32(0.95)

"""
    read_entropy(ds::DrandSource) -> ColoredTick

Advance the deterministic round-number simulator. A live implementation would
need to retrieve the round, validate the chain parameters and BLS signature,
and record that verification evidence; this function does none of those.
"""
function read_entropy(ds::DrandSource)::ColoredTick
    # Simulation: derive an opaque sample word from the round number
    # Real implementation: HTTP GET ds.endpoint/public/latest
    ds.last_round += 1
    ds.last_randomness = splitmix64(ds.last_round * 0x6472616e64626561)  # "drandbea"

    tick = TritTick(ds.last_round * DRAND_TICKS_PER_ROUND)
    measured_trit = Int8(ds.last_randomness % 3) - Int8(1)

    ColoredTick(tick, measured_trit, ds.last_randomness, Float32(0.95), :drand)
end

# ═══════════════════════════════════════════════════════════════════════════
# Source 1: Energy Monitor (Immortal, physics)
# ═══════════════════════════════════════════════════════════════════════════

"""
    EnergySource <: EntropySource

TP-Link Tapo P15 smart outlet energy monitor.
Reads wattage over KLAP v2 (TCP :80).

GF(3) classification (matching tapo_energy.zig):
- >30W → +1 (generator: active charging)
- 5–30W → 0 (ergodic: trickle/maintenance)
- <5W → -1 (validator: idle/off)

Trustworthiness: high (actual electrons).
Mortality: Immortal (outlet doesn't die).
"""
mutable struct EnergySource <: EntropySource
    host::String
    port::UInt16
    last_watts::Float32
    threshold_high::Float32        # default 30.0W
    threshold_low::Float32         # default 5.0W
end

EnergySource(host::String="192.168.1.100") = EnergySource(host, UInt16(80), 0.0f0, 30.0f0, 5.0f0)

source_name(::EnergySource) = :energy
source_mortality(::EnergySource) = Immortal
source_trustworthiness(::EnergySource) = Float32(0.85)

function read_entropy(es::EnergySource)::ColoredTick
    # Simulation: real implementation would KLAP v2 handshake + JSON-RPC
    # For now, accept injected watts via es.last_watts
    watts = es.last_watts
    measured_trit = classify_trit(watts, es.threshold_low, es.threshold_high)

    # Entropy from wattage: hash the float bits
    entropy = splitmix64(reinterpret(UInt64, Float64(watts)))
    tick = TritTick(round(UInt64, time() * EPOCH_1_HZ) % (EPOCH_1_HZ * 86400))

    ColoredTick(tick, measured_trit, entropy, Float32(0.85), :energy)
end

"""Inject a wattage reading (for testing or when polling externally)."""
function inject_watts!(es::EnergySource, watts::Real)
    es.last_watts = Float32(watts)
    read_entropy(es)
end

# ═══════════════════════════════════════════════════════════════════════════
# Source 2: BCI / EEG (Mortal, neural)
# ═══════════════════════════════════════════════════════════════════════════

"""
    BCISource <: EntropySource

Brain-Computer Interface entropy source.
Accepts band-power entropy from EEG/fNIRS/ultrasound.

GF(3) classification (matching bci_receiver.zig):
- high entropy → +1 (active cognition, generative state)
- baseline → 0 (ergodic, resting coordination)
- low entropy → -1 (suppression, rest, validation)

Trustworthiness: medium (noisy, needs calibration).
Mortality: Mortal (requires a brain, battery, skin contact).
"""
mutable struct BCISource <: EntropySource
    n_channels::UInt8
    sample_rate_hz::UInt16         # typically 250, 256, 500, 512
    last_entropy::Float32          # Shannon entropy of band powers
    threshold_high::Float32
    threshold_low::Float32
    device::Symbol                 # :muse2, :openbci_cyton, :intan, :custom
end

function BCISource(; device::Symbol=:muse2, sample_rate::Integer=256)
    BCISource(UInt8(4), UInt16(sample_rate), 0.0f0, 0.7f0, 0.3f0, device)
end

source_name(bs::BCISource) = :bci
source_mortality(::BCISource) = Mortal
source_trustworthiness(::BCISource) = Float32(0.60)

function read_entropy(bs::BCISource)::ColoredTick
    entropy_val = bs.last_entropy
    measured_trit = classify_trit(entropy_val, bs.threshold_low, bs.threshold_high)

    entropy = splitmix64(reinterpret(UInt64, Float64(entropy_val)))
    # Tick aligned to BCI sample rate
    tps = EPOCH_1_HZ ÷ UInt64(bs.sample_rate_hz)
    tick_val = round(UInt64, time() * EPOCH_1_HZ) % (EPOCH_1_HZ * 86400)
    tick_val = (tick_val ÷ tps) * tps  # snap to sample boundary

    ColoredTick(TritTick(tick_val), measured_trit, entropy,
                source_trustworthiness(bs), :bci)
end

"""Inject a band-power entropy reading."""
function inject_entropy!(bs::BCISource, entropy::Real)
    bs.last_entropy = Float32(entropy)
    read_entropy(bs)
end

# ═══════════════════════════════════════════════════════════════════════════
# Source 3: Heartbeat (Mortal, biology)
# ═══════════════════════════════════════════════════════════════════════════

"""
    HeartbeatSource <: EntropySource

Heart rate / HRV entropy source. MAX30102 pulse oximeter or Apple Watch.

GF(3) classification:
- High HRV → +1 (coherent, generative state)
- Normal HRV → 0 (baseline coordination)
- Low HRV → -1 (stress, need to check/validate)

Trustworthiness: very high (biology).
Mortality: **Mortal** — stops when you do. The death interval
is the only non-factorable interval in the system.
"""
mutable struct HeartbeatSource <: EntropySource
    last_bpm::Float32
    last_hrv_ms::Float32           # HRV in milliseconds (RMSSD)
    hrv_threshold_high::Float32    # default 50ms
    hrv_threshold_low::Float32     # default 20ms
    alive::Bool                    # false when last tick is the last tick
end

HeartbeatSource() = HeartbeatSource(0.0f0, 0.0f0, 50.0f0, 20.0f0, true)

source_name(::HeartbeatSource) = :heartbeat
source_mortality(::HeartbeatSource) = Mortal
source_trustworthiness(::HeartbeatSource) = Float32(0.90)

function read_entropy(hs::HeartbeatSource)::ColoredTick
    if !hs.alive
        # The death interval: confidence drops to 0, trit is undefined
        return ColoredTick(TritTick(UInt64(0)), Int8(0), UInt64(0), 0.0f0, :heartbeat)
    end

    measured_trit = classify_trit(hs.last_hrv_ms, hs.hrv_threshold_low, hs.hrv_threshold_high)

    # Entropy from inter-beat interval (IBI) — hash BPM and HRV together
    ibi_bits = reinterpret(UInt64, Float64(hs.last_bpm)) ⊻
               reinterpret(UInt64, Float64(hs.last_hrv_ms))
    entropy = splitmix64(ibi_bits)

    # Tick at heartbeat rate: 1 Hz = EPOCH_1_HZ trit-ticks per beat
    tick_val = round(UInt64, time() * EPOCH_1_HZ) % (EPOCH_1_HZ * 86400)

    ColoredTick(TritTick(tick_val), measured_trit, entropy, Float32(0.90), :heartbeat)
end

"""Inject a heartbeat reading (BPM and HRV in ms)."""
function inject_heartbeat!(hs::HeartbeatSource, bpm::Real, hrv_ms::Real)
    hs.last_bpm = Float32(bpm)
    hs.last_hrv_ms = Float32(hrv_ms)
    read_entropy(hs)
end

# ═══════════════════════════════════════════════════════════════════════════
# Source 4: Air Quality (SemiMortal, chemistry)
# ═══════════════════════════════════════════════════════════════════════════

"""
    AirQualitySource <: EntropySource

PM2.5 air quality sensor (PMS5003, PurpleAir, Sensirion SEN5x).
1 Hz sample rate divides Epoch 1 exactly.

GF(3) classification:
- AQI < 50 → +1 (good air, conditions permit making)
- AQI 50–150 → 0 (marginal, coordinate/monitor)
- AQI > 150 → -1 (bad, check source, verify ventilation)

Trustworthiness: high (chemistry/physics).
Mortality: SemiMortal (sensor degrades over time).
"""
mutable struct AirQualitySource <: EntropySource
    last_aqi::Float32
    last_pm25::Float32             # µg/m³
    threshold_good::Float32        # default AQI 50
    threshold_bad::Float32         # default AQI 150
end

AirQualitySource() = AirQualitySource(0.0f0, 0.0f0, 50.0f0, 150.0f0)

source_name(::AirQualitySource) = :air_quality
source_mortality(::AirQualitySource) = SemiMortal
source_trustworthiness(::AirQualitySource) = Float32(0.80)

function read_entropy(aqs::AirQualitySource)::ColoredTick
    # Invert: good air = maker, bad air = checker
    measured_trit = classify_trit(aqs.last_aqi, aqs.threshold_good, aqs.threshold_bad)
    measured_trit = -measured_trit  # invert: low AQI = good = +1

    entropy = splitmix64(reinterpret(UInt64, Float64(aqs.last_pm25)))
    tick_val = round(UInt64, time() * EPOCH_1_HZ) % (EPOCH_1_HZ * 86400)

    ColoredTick(TritTick(tick_val), measured_trit, entropy, Float32(0.80), :air_quality)
end

"""Inject an AQI and PM2.5 reading."""
function inject_aqi!(aqs::AirQualitySource, aqi::Real, pm25::Real)
    aqs.last_aqi = Float32(aqi)
    aqs.last_pm25 = Float32(pm25)
    read_entropy(aqs)
end

# ═══════════════════════════════════════════════════════════════════════════
# Source 5: deterministic Git metadata adapter
# ═══════════════════════════════════════════════════════════════════════════

"""
    GitSource <: EntropySource

Git commit hash → trit representation.
`share3_hash(commit_hash)` → deterministic trit assignment.

A commit hash is public deterministic metadata, not an entropy source. The
`trustworthiness` and `mortality` values are application policy labels.
"""
mutable struct GitSource <: EntropySource
    repo_path::String
    last_hash::UInt64              # truncated commit hash
end

GitSource(path::String=".") = GitSource(path, UInt64(0))

source_name(::GitSource) = :git
source_mortality(::GitSource) = Immortal
source_trustworthiness(::GitSource) = Float32(0.85)

function read_entropy(gs::GitSource)::ColoredTick
    # share3_hash: (byte0 + byte1 + byte2) mod 3 - 1
    h = gs.last_hash
    trit_val = Int8(((h & 0xFF) + ((h >> 8) & 0xFF) + ((h >> 16) & 0xFF)) % 3) - Int8(1)

    tick_val = round(UInt64, time() * EPOCH_1_HZ) % (EPOCH_1_HZ * 86400)
    ColoredTick(TritTick(tick_val), trit_val, h, Float32(0.85), :git)
end

"""Inject a commit hash (as UInt64 truncation of SHA)."""
function inject_commit!(gs::GitSource, hash_bytes::UInt64)
    gs.last_hash = hash_bytes
    read_entropy(gs)
end

# ═══════════════════════════════════════════════════════════════════════════
# Source 6: Accelerometer (SemiMortal, physics)
# ═══════════════════════════════════════════════════════════════════════════

"""
    AccelerometerSource <: EntropySource

Jerk (derivative of acceleration) → trit.
- Positive jerk → +1 (initiating motion, making)
- Near-zero → 0 (steady state, coordinating)
- Negative jerk → -1 (braking, checking)

Sample rate 100-1000 Hz (all divide Epoch 1).
Trustworthiness: high (physics).
Mortality: SemiMortal (battery).
"""
mutable struct AccelerometerSource <: EntropySource
    last_jerk::Float32             # m/s³
    threshold_high::Float32        # default 0.5 m/s³
    threshold_low::Float32         # default -0.5 m/s³
    sample_rate_hz::UInt16
end

AccelerometerSource(; rate::Integer=100) = AccelerometerSource(0.0f0, 0.5f0, -0.5f0, UInt16(rate))

source_name(::AccelerometerSource) = :accelerometer
source_mortality(::AccelerometerSource) = SemiMortal
source_trustworthiness(::AccelerometerSource) = Float32(0.80)

function read_entropy(as::AccelerometerSource)::ColoredTick
    measured_trit = classify_trit(as.last_jerk, as.threshold_low, as.threshold_high)
    entropy = splitmix64(reinterpret(UInt64, Float64(as.last_jerk)))

    tps = EPOCH_1_HZ ÷ UInt64(as.sample_rate_hz)
    tick_val = round(UInt64, time() * EPOCH_1_HZ) % (EPOCH_1_HZ * 86400)
    tick_val = (tick_val ÷ tps) * tps  # snap to sample boundary

    ColoredTick(TritTick(tick_val), measured_trit, entropy, Float32(0.80), :accelerometer)
end

"""Inject a jerk reading (m/s³)."""
function inject_jerk!(as::AccelerometerSource, jerk::Real)
    as.last_jerk = Float32(jerk)
    read_entropy(as)
end

# ═══════════════════════════════════════════════════════════════════════════
# Composite Source — multiple sources, conservation-checked
# ═══════════════════════════════════════════════════════════════════════════

"""
    CompositeSource <: EntropySource

Combines multiple entropy sources. The composite trit and entropy
are derived from all active sources, weighted by trustworthiness.

Conservation check: if all sources agree on trit, strong signal.
If they disagree, the disagreement IS the information.

"Your brain says make but the air says check."
"""
struct CompositeSource <: EntropySource
    sources::Vector{EntropySource}
    name::Symbol
end

CompositeSource(sources::Vector{<:EntropySource}; name::Symbol=:composite) =
    CompositeSource(convert(Vector{EntropySource}, sources), name)

source_name(cs::CompositeSource) = cs.name
source_mortality(cs::CompositeSource) = maximum(source_mortality, cs.sources)
source_trustworthiness(cs::CompositeSource) =
    sum(source_trustworthiness, cs.sources) / length(cs.sources)

"""
    read_entropy(cs::CompositeSource) -> ColoredTick

Read from all sources, combine entropy and compute consensus trit.
"""
function read_entropy(cs::CompositeSource)::ColoredTick
    readings = [read_entropy(s) for s in cs.sources]
    composite_from_readings(readings, cs.name)
end

"""
    composite_from_readings(readings::Vector{ColoredTick}, name::Symbol=:composite) -> ColoredTick

Combine multiple `ColoredTick` observations into one policy summary.
- Sample word: XOR fold of all compatibility `entropy` fields
- Trit: weighted vote by confidence
- Confidence: mean of individual confidences, boosted by agreement

No independence test, min-entropy estimator, conditioning function, or
extractor is performed.
"""
function composite_from_readings(readings::Vector{ColoredTick}, name::Symbol=:composite)
    isempty(readings) && return ColoredTick(TritTick(UInt64(0)), Int8(0), UInt64(0), 0.0f0, name)

    # XOR-fold entropy
    entropy = reduce(⊻, ct.entropy for ct in readings)

    # Weighted trit vote
    weighted_sum = sum(Float32(ct.measured_trit) * ct.confidence for ct in readings)
    consensus_trit = weighted_sum > 0.33f0 ? Int8(1) :
                     weighted_sum < -0.33f0 ? Int8(-1) : Int8(0)

    # Confidence: mean, boosted if all agree
    mean_conf = sum(ct.confidence for ct in readings) / length(readings)
    all_agree = all(ct.measured_trit == readings[1].measured_trit for ct in readings)
    confidence = all_agree ? min(mean_conf * 1.2f0, 1.0f0) : mean_conf * 0.8f0

    # Use the most recent tick
    latest_tick = readings[argmax([ct.tick.tick for ct in readings])].tick

    ColoredTick(latest_tick, consensus_trit, entropy, confidence, name)
end

"""
    agreement(readings::Vector{ColoredTick}) -> Float32

How much do the sources agree? 1.0 = unanimous, 0.0 = maximum disagreement.
"""
function agreement(readings::Vector{ColoredTick})
    isempty(readings) && return 1.0f0
    trits = [ct.measured_trit for ct in readings]
    # Count most common trit
    counts = [count(==(t), trits) for t in Int8[-1, 0, 1]]
    Float32(maximum(counts)) / Float32(length(trits))
end

"""
    disagreement_signal(readings::Vector{ColoredTick}) -> NamedTuple

When sources disagree, WHAT disagrees is the signal.
Returns which sources are maker/coordinator/checker.
"""
function disagreement_signal(readings::Vector{ColoredTick})
    makers = Symbol[ct.source for ct in readings if ct.measured_trit == Int8(1)]
    coordinators = Symbol[ct.source for ct in readings if ct.measured_trit == Int8(0)]
    checkers = Symbol[ct.source for ct in readings if ct.measured_trit == Int8(-1)]
    (makers=makers, coordinators=coordinators, checkers=checkers,
     agreement=agreement(readings))
end

# ═══════════════════════════════════════════════════════════════════════════
# CompositeSource as a TickSource (plugs into GayRNG)
# ═══════════════════════════════════════════════════════════════════════════

"""
    CompositeTicks <: TickSource

A TickSource backed by a CompositeSource. Each `current_tick` reads
from all physical sources, folds entropy, and returns the consensus tick.
"""
struct CompositeTicks <: TickSource
    composite::CompositeSource
end

function current_tick(ct::CompositeTicks)::TritTick
    reading = read_entropy(ct.composite)
    reading.tick
end

"""
    current_colored_tick(ct::CompositeTicks) -> ColoredTick

Get the full ColoredTick (with entropy and confidence) from a CompositeTicks source.
"""
current_colored_tick(ct::CompositeTicks) = read_entropy(ct.composite)

end # module EntropySources
