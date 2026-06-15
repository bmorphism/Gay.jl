# Layer 0: Trit-Tick — the primary unit of time
#
# A trit-tick is 1/141,120,000 of a second. It carries algebraic structure:
# its prime factorization (2⁹ × 3² × 5⁴ × 7²) determines which sensor
# modalities synchronize exactly, and its GF(3) trit tells you whether
# the moment is making (+1), coordinating (0), or checking (-1).
#
# See: WHY_TRIT_TICK.md, CC-MATH.md

export TritTick, TickSource, LogicalTicks, WallClockTicks
export trit, trit_role, hue_quantum, ticks_per_second
export current_tick, tick_now, between, fits
export EPOCH_1_HZ, EPOCH_2_HZ, FLICKS_PER_TICK
export GF3_QUANTUM, BAND_QUANTUM, HUE_DEGREE_QUANTUM
export conservation_check, trit_sum, to_flicks, from_flicks, modalities_between
export DisplayTransport, TRANSPORTS, transport_bits_per_tick, transport_pixels_per_tick
export VerificationLevel, PER_PIXEL, PER_TILE, PER_FRAME, DIAGNOSTIC
export capability_gate, verification_budget, min_tile_size, display_config
export DisplayConfig
export RenderTech, RENDER_TECHS, ColorSpace, COLOR_SPACES
export RenderStack, render_stack, channel_parallelism, color_fidelity
export KNOWN_STACKS

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

"""Epoch 1 tick rate: 141,120,000 Hz = 2⁹ × 3² × 5⁴ × 7²"""
const EPOCH_1_HZ = UInt64(141_120_000)

"""Epoch 2 tick rate: 51,433,932,566,016,000,000 Hz"""
const EPOCH_2_HZ = UInt128(51_433_932_566_016_000_000)

"""Expansion factor from Epoch 1 to Epoch 2"""
const EXPANSION_FACTOR = UInt64(364_469_476_800)

"""1 trit-tick = 5 flicks exactly (705,600,000 / 141,120,000 = 5)"""
const FLICKS_PER_TICK = UInt8(5)

"""GF(3) quantum: T₁/3 = 47,040,000 trit-ticks per trit phase"""
const GF3_QUANTUM = EPOCH_1_HZ ÷ 3

"""Band quantum: T₁/5 = 28,224,000"""
const BAND_QUANTUM = EPOCH_1_HZ ÷ 5

"""Hue degree quantum: T₁/360 = 392,000 trit-ticks per degree"""
const HUE_DEGREE_QUANTUM = EPOCH_1_HZ ÷ 360

# ═══════════════════════════════════════════════════════════════════════════
# Modality table: sample rates and their exact tick counts
# ═══════════════════════════════════════════════════════════════════════════

"""
Named modalities with their sample rates (Hz) and trit-ticks per sample.
A modality divides exactly iff EPOCH_1_HZ % rate == 0.
"""
const MODALITIES = (
    # BCI / Neural
    eeg_250    = (rate = 250,   tps = EPOCH_1_HZ ÷ 250),
    eeg_256    = (rate = 256,   tps = EPOCH_1_HZ ÷ 256),
    eeg_500    = (rate = 500,   tps = EPOCH_1_HZ ÷ 500),
    eeg_512    = (rate = 512,   tps = EPOCH_1_HZ ÷ 512),
    eeg_1000   = (rate = 1000,  tps = EPOCH_1_HZ ÷ 1000),
    eeg_2000   = (rate = 2000,  tps = EPOCH_1_HZ ÷ 2000),
    fnirs_10   = (rate = 10,    tps = EPOCH_1_HZ ÷ 10),
    ultrasound = (rate = 100,   tps = EPOCH_1_HZ ÷ 100),
    # Audio
    audio_44100 = (rate = 44100, tps = EPOCH_1_HZ ÷ 44100),  # = 3200 exact
    audio_48000 = (rate = 48000, tps = EPOCH_1_HZ ÷ 48000),
    audio_96000 = (rate = 96000, tps = EPOCH_1_HZ ÷ 96000),
    # Video / Display
    video_24   = (rate = 24,    tps = EPOCH_1_HZ ÷ 24),
    video_30   = (rate = 30,    tps = EPOCH_1_HZ ÷ 30),
    video_60   = (rate = 60,    tps = EPOCH_1_HZ ÷ 60),
    video_90   = (rate = 90,    tps = EPOCH_1_HZ ÷ 90),
    video_120  = (rate = 120,   tps = EPOCH_1_HZ ÷ 120),
    # Human
    heartbeat  = (rate = 1,     tps = EPOCH_1_HZ),
)

# ═══════════════════════════════════════════════════════════════════════════
# TritTick: the primary temporal coordinate
# ═══════════════════════════════════════════════════════════════════════════

"""
    TritTick

A position in the trit-tick grid. Carries:
- `tick::UInt64` — absolute position (Epoch 1) or within-second position
- `epoch::UInt8` — 1 (141,120,000 Hz) or 2 (expanded)

Derived properties (computed, not stored):
- `trit(t)` → GF(3) value {-1, 0, +1}
- `hue_quantum(t)` → which degree of hue rotation
- `fits(interval, modality)` → can this modality sample in the interval?

One trit-tick = 5 flicks = ~7.1 nanoseconds.
"""
struct TritTick
    tick::UInt64
    epoch::UInt8

    TritTick(tick::UInt64, epoch::UInt8=0x01) = new(tick, epoch)
    TritTick(tick::Integer, epoch::Integer=1) = new(UInt64(tick), UInt8(epoch))
end

Base.show(io::IO, t::TritTick) = print(io, "TritTick(", t.tick, ", epoch=", Int(t.epoch),
    ", trit=", trit(t), ")")

Base.isless(a::TritTick, b::TritTick) = a.tick < b.tick
Base.:(==)(a::TritTick, b::TritTick) = a.tick == b.tick && a.epoch == b.epoch
Base.:(-)(a::TritTick, b::TritTick) = Int128(a.tick) - Int128(b.tick)

"""Convert TritTick to a UInt64 for use as hash_color index."""
Base.convert(::Type{UInt64}, t::TritTick) = t.tick

# ═══════════════════════════════════════════════════════════════════════════
# GF(3) primitives
# ═══════════════════════════════════════════════════════════════════════════

"""
    trit(t::TritTick) -> Int8

GF(3) trit at this tick: -1 (checker), 0 (coordinator), +1 (maker).
Based on which second this tick falls in: `floor(tick / T₁) mod 3`, balanced.
"""
@inline function trit(t::TritTick)::Int8
    second = t.tick ÷ EPOCH_1_HZ
    Int8((second + 1) % 3) - Int8(1)
end



"""
    trit_role(t) -> Symbol

Human-readable role for this tick's trit.
"""
function trit_role(t)
    v = trit(t isa TritTick ? t : TritTick(t))
    v == Int8(-1) ? :checker :
    v == Int8(0)  ? :coordinator :
                    :maker
end

"""
    trit_sum(ticks) -> Int

Sum of trits over a collection. Conservation: should be ≡ target (mod 3).
"""
trit_sum(ticks) = sum(trit, ticks)

"""
    conservation_check(ticks, target::Int=-1) -> Bool

Check GF(3) conservation: trit sum ≡ target (mod 3).
Default target = -1 (whole earth arena error).
"""
conservation_check(ticks, target::Int=-1) = mod(trit_sum(ticks), 3) == mod(target, 3)

# ═══════════════════════════════════════════════════════════════════════════
# Temporal queries
# ═══════════════════════════════════════════════════════════════════════════

"""
    hue_quantum(t::TritTick) -> UInt64

Which hue degree quantum this tick falls in (0-359 within each second).
"""
@inline hue_quantum(t::TritTick) = (t.tick % EPOCH_1_HZ) ÷ HUE_DEGREE_QUANTUM

"""
    ticks_per_second(epoch::Integer=1) -> Union{UInt64, UInt128}

Tick rate for the given epoch.
"""
ticks_per_second(epoch::Integer=1) = epoch == 1 ? EPOCH_1_HZ :
    epoch == 2 ? EPOCH_2_HZ :
    error("Unknown epoch $epoch")

"""
    between(a::TritTick, b::TritTick) -> UInt64

Number of trit-ticks between two moments (absolute value).
"""
between(a::TritTick, b::TritTick) = a.tick > b.tick ? a.tick - b.tick : b.tick - a.tick

"""
    fits(interval_ticks::Integer, modality::Symbol) -> Bool

Can the given modality place a sample exactly in an interval of this many trit-ticks?
True iff the modality's ticks-per-sample divides the interval evenly.
"""
function fits(interval_ticks::Integer, modality::Symbol)
    m = getfield(MODALITIES, modality)
    interval_ticks >= m.tps && interval_ticks % m.tps == 0
end

"""
    fits(a::TritTick, b::TritTick, modality::Symbol) -> Bool

Can the modality sample exactly between two trit-ticks?
"""
fits(a::TritTick, b::TritTick, modality::Symbol) = fits(between(a, b), modality)

"""
    modalities_between(a::TritTick, b::TritTick) -> Vector{Symbol}

Which modalities can sample exactly in the interval between a and b?
"""
function modalities_between(a::TritTick, b::TritTick)
    gap = between(a, b)
    result = Symbol[]
    for name in fieldnames(typeof(MODALITIES))
        m = getfield(MODALITIES, name)
        if gap >= m.tps && gap % m.tps == 0
            push!(result, name)
        end
    end
    result
end

# ═══════════════════════════════════════════════════════════════════════════
# TickSource: where ticks come from
# ═══════════════════════════════════════════════════════════════════════════

"""
    TickSource

Abstract type for tick sources. Subtypes provide `current_tick(source)::TritTick`.
"""
abstract type TickSource end

"""
    LogicalTicks <: TickSource

Pure logical ticks — no wall clock. Each call to `current_tick` increments
a counter. Deterministic, replayable, no time dependency.
"""
mutable struct LogicalTicks <: TickSource
    counter::UInt64
end
LogicalTicks() = LogicalTicks(UInt64(0))

function current_tick(lt::LogicalTicks)::TritTick
    lt.counter += 1
    TritTick(lt.counter)
end

"""
    WallClockTicks <: TickSource

Wall clock ticks — converts `time()` to trit-ticks relative to an epoch.
Precision depends on system clock (~microsecond on modern OS).
"""
struct WallClockTicks <: TickSource
    epoch_time::Float64  # time() at tick 0
    hz::UInt64           # ticks per second
end

WallClockTicks() = WallClockTicks(time(), EPOCH_1_HZ)

function current_tick(wct::WallClockTicks)::TritTick
    elapsed = time() - wct.epoch_time
    ticks = round(UInt64, elapsed * wct.hz)
    TritTick(ticks)
end

"""
    tick_now() -> TritTick

Get current wall-clock time as a TritTick (Epoch 1).
Convenience function — creates a temporary WallClockTicks source.
"""
function tick_now()::TritTick
    # Seconds since Unix epoch × T₁
    t = time()
    ticks = round(UInt64, (t % 86400) * EPOCH_1_HZ)  # within-day to avoid UInt64 overflow
    TritTick(ticks)
end

# ═══════════════════════════════════════════════════════════════════════════
# Flick conversion
# ═══════════════════════════════════════════════════════════════════════════

"""
    to_flicks(t::TritTick) -> UInt64

Convert trit-tick to flicks. Exact: 1 trit-tick = 5 flicks.
"""
to_flicks(t::TritTick) = t.tick * UInt64(FLICKS_PER_TICK)

"""
    from_flicks(flicks::Integer) -> TritTick

Convert flicks to trit-tick. Rounds down (loses at most 4 flicks).
"""
from_flicks(flicks::Integer) = TritTick(UInt64(flicks) ÷ UInt64(FLICKS_PER_TICK))

# ═══════════════════════════════════════════════════════════════════════════
# Display Transport: wired and wireless capability gates
# ═══════════════════════════════════════════════════════════════════════════
#
# The verification budget depends on the transport between the color source
# and the display surface. HDMI delivers ~128 bits/trit-tick; WiFi delivers
# ~2-6 bits/trit-tick. The capability gate inequality determines which
# verification level (per-pixel, per-tile, per-frame, diagnostic) is
# achievable at a given resolution and refresh rate.
#
# See: CC-MATH.md §3, §6

"""
    DisplayTransport

A physical or wireless link between color source and display surface.
Bandwidth determines the verification regime.

- `name`: human-readable identifier
- `bandwidth_bps`: raw bandwidth in bits per second
- `latency_us`: one-way latency in microseconds
- `protocol`: :hdmi, :dp, :usbc, :wifi, :miracast, :iprojection
- `max_devices`: simultaneous display endpoints (1 for wired, N for wireless)
"""
struct DisplayTransport
    name::String
    bandwidth_bps::UInt64
    latency_us::UInt64
    protocol::Symbol
    max_devices::UInt8
end

const TRANSPORTS = (
    # Wired — point-to-point, low latency, high bandwidth
    hdmi_1_4  = DisplayTransport("HDMI 1.4",        UInt64(10_200_000_000),   UInt64(500),    :hdmi,        0x01),
    hdmi_2_0  = DisplayTransport("HDMI 2.0",        UInt64(18_000_000_000),   UInt64(500),    :hdmi,        0x01),
    hdmi_2_1  = DisplayTransport("HDMI 2.1",        UInt64(48_000_000_000),   UInt64(500),    :hdmi,        0x01),
    dp_1_4    = DisplayTransport("DisplayPort 1.4", UInt64(32_400_000_000),   UInt64(500),    :dp,          0x01),
    dp_2_0    = DisplayTransport("DisplayPort 2.0", UInt64(80_000_000_000),   UInt64(500),    :dp,          0x01),
    usbc_tb3  = DisplayTransport("USB-C/TB3",       UInt64(40_000_000_000),   UInt64(1000),   :usbc,        0x01),
    usbc_tb4  = DisplayTransport("USB-C/TB4",       UInt64(40_000_000_000),   UInt64(1000),   :usbc,        0x01),
    usbc_tb5  = DisplayTransport("USB-C/TB5",       UInt64(80_000_000_000),   UInt64(500),    :usbc,        0x01),
    # Wireless — multi-device, high latency, constrained bandwidth
    wifi_5ghz_ac   = DisplayTransport("WiFi 5GHz 802.11ac",  UInt64(600_000_000),  UInt64(30_000),  :wifi,        0x32),  # 50 devices iProjection
    wifi_5ghz_ax   = DisplayTransport("WiFi 5GHz 802.11ax",  UInt64(1_200_000_000), UInt64(20_000), :wifi,        0x32),
    wifi_6ghz_ax   = DisplayTransport("WiFi 6GHz 802.11ax",  UInt64(2_400_000_000), UInt64(15_000), :wifi,        0x32),
    miracast       = DisplayTransport("Miracast",             UInt64(300_000_000),   UInt64(50_000), :miracast,    0x01),
    iprojection    = DisplayTransport("Epson iProjection",    UInt64(600_000_000),   UInt64(60_000), :iprojection, 0x32),  # 50 connect, 4 display
    airplay        = DisplayTransport("AirPlay",              UInt64(500_000_000),   UInt64(40_000), :wifi,        0x01),
    chromecast     = DisplayTransport("Chromecast",           UInt64(400_000_000),   UInt64(50_000), :wifi,        0x01),
    # Air-gapped (QRTP from bci_receiver.zig)
    qrtp           = DisplayTransport("QRTP Air-Gapped",     UInt64(1_000),         UInt64(500_000), :qrtp,       0x01),  # ~1 kbps, 500ms latency
)

"""
    transport_bits_per_tick(t::DisplayTransport) -> Float64

Bits deliverable per trit-tick over this transport.
HDMI 2.0: ~128 bits/tick. WiFi 5GHz ac: ~4.3 bits/tick. QRTP: ~0.000007 bits/tick.
"""
transport_bits_per_tick(t::DisplayTransport) = Float64(t.bandwidth_bps) / Float64(EPOCH_1_HZ)

"""
    transport_pixels_per_tick(t::DisplayTransport; depth::Int=30) -> Float64

Pixels deliverable per trit-tick. `depth` = bits per pixel (default 30 = 10-bit RGB).
"""
transport_pixels_per_tick(t::DisplayTransport; depth::Int=30) = transport_bits_per_tick(t) / depth

"""
    VerificationLevel

Which granularity of color verification is achievable given the transport
bandwidth and display resolution.

- `PER_PIXEL`: every pixel gets its own trit-tick verification (ideal, needs rho >= 1)
- `PER_TILE`: pixels grouped into k*k tiles, one verification per tile
- `PER_FRAME`: one verification per entire frame (minimum for any display)
- `DIAGNOSTIC`: single color fill, prove liveness only (QRTP, degraded wireless)
"""
@enum VerificationLevel begin
    PER_PIXEL   = 4
    PER_TILE    = 3
    PER_FRAME   = 2
    DIAGNOSTIC  = 1
end

"""
    DisplayConfig

A concrete display endpoint: resolution, refresh rate, color depth, transport.
The capability gate computes what verification level is achievable.
"""
struct DisplayConfig
    width::UInt32
    height::UInt32
    refresh_hz::UInt16
    depth_bits::UInt8       # bits per pixel (24=8bit, 30=10bit, 36=12bit)
    transport::DisplayTransport
end

"""
    verification_budget(dc::DisplayConfig) -> UInt64

Trit-ticks available per frame for this display configuration.
B(R) = T₁ / R
"""
verification_budget(dc::DisplayConfig) = EPOCH_1_HZ ÷ UInt64(dc.refresh_hz)

"""
    capability_gate(dc::DisplayConfig) -> (level, tile_k, rho, latency_ticks)

The core capability gate inequality from CC-MATH §6.

Returns the highest achievable verification level, the minimum tile size k,
the verification density rho, and the transport latency in trit-ticks.

The gate inequality: T₁ / (R * W * H / k²) >= 1
Solve for k: k >= sqrt(R * W * H / T₁)
"""
function capability_gate(dc::DisplayConfig)
    budget = verification_budget(dc)
    pixels = UInt64(dc.width) * UInt64(dc.height)
    latency_ticks = UInt64(round(Float64(dc.transport.latency_us) * 1e-6 * Float64(EPOCH_1_HZ)))

    # Verification density: trit-ticks per pixel per frame
    rho = Float64(budget) / Float64(pixels)

    if rho >= 1.0
        # Per-pixel verification achievable
        return (level=PER_PIXEL, tile_k=UInt32(1), rho=rho, latency_ticks=latency_ticks)
    end

    # Need tiling. Find minimum k such that budget >= ceil(pixels / k^2)
    # k >= sqrt(pixels / budget)
    k_min = ceil(Int, sqrt(Float64(pixels) / Float64(budget)))
    tiles = cld(dc.width, UInt32(k_min)) * cld(dc.height, UInt32(k_min))

    if tiles <= budget
        return (level=PER_TILE, tile_k=UInt32(k_min), rho=rho, latency_ticks=latency_ticks)
    end

    # Even tiling can't help — per-frame only
    if budget >= 1
        return (level=PER_FRAME, tile_k=UInt32(max(dc.width, dc.height)), rho=rho, latency_ticks=latency_ticks)
    end

    # Framerate exceeds T₁ — diagnostic only
    return (level=DIAGNOSTIC, tile_k=UInt32(max(dc.width, dc.height)), rho=rho, latency_ticks=latency_ticks)
end

"""
    min_tile_size(dc::DisplayConfig) -> UInt32

Minimum tile edge length k for the display to pass the capability gate.
k=1 means per-pixel is achievable. k=max(W,H) means per-frame only.
"""
function min_tile_size(dc::DisplayConfig)
    _, k, _, _ = capability_gate(dc)
    k
end

"""
    display_config(name::Symbol) -> DisplayConfig

Predefined display configurations for common targets.
"""
function display_config(name::Symbol)
    configs = Dict(
        # Wired displays
        :fhd_60_hdmi     => DisplayConfig(1920, 1080, 60,  30, TRANSPORTS.hdmi_2_0),
        :qhd_60_hdmi     => DisplayConfig(2560, 1440, 60,  30, TRANSPORTS.hdmi_2_0),
        :uhd_60_hdmi     => DisplayConfig(3840, 2160, 60,  30, TRANSPORTS.hdmi_2_0),
        :uhd_120_hdmi    => DisplayConfig(3840, 2160, 120, 30, TRANSPORTS.hdmi_2_1),
        :xdr_60_tb3      => DisplayConfig(6016, 3384, 60,  30, TRANSPORTS.usbc_tb3),
        :studio_60_tb3   => DisplayConfig(5120, 2880, 60,  30, TRANSPORTS.usbc_tb3),
        # Wireless — Epson iProjection (typical projector resolutions)
        :wxga_iprojection  => DisplayConfig(1280,  800, 60, 24, TRANSPORTS.iprojection),
        :fhd_iprojection   => DisplayConfig(1920, 1080, 60, 24, TRANSPORTS.iprojection),
        :wuxga_iprojection => DisplayConfig(1920, 1200, 60, 24, TRANSPORTS.iprojection),
        :uhd_iprojection   => DisplayConfig(3840, 2160, 30, 24, TRANSPORTS.iprojection),
        # Wireless — Miracast / AirPlay / Chromecast
        :fhd_miracast      => DisplayConfig(1920, 1080, 60, 24, TRANSPORTS.miracast),
        :fhd_airplay       => DisplayConfig(1920, 1080, 60, 24, TRANSPORTS.airplay),
        :uhd_chromecast    => DisplayConfig(3840, 2160, 30, 24, TRANSPORTS.chromecast),
        # Air-gapped (QRTP)
        :qrtp_diagnostic   => DisplayConfig(320,   240,  1, 24, TRANSPORTS.qrtp),
    )
    get(configs, name) do
        error("Unknown display config: $name. Known: $(join(sort(collect(keys(configs))), ", "))")
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# Rendering Technology: how photons are actually produced
# ═══════════════════════════════════════════════════════════════════════════
#
# The transport delivers bits. The renderer turns bits into light.
# Different rendering technologies have fundamentally different relationships
# to the trit-tick verification budget:
#
#   3LCD (Epson):     3 panels process R,G,B in parallel → 3× channel parallelism
#   DLP (single):     1 chip, color wheel spins R→G→B sequentially → 1× (temporal mux)
#   DLP (3-chip):     3 DMD chips, no color wheel → 3× parallelism
#   LCoS/SXRD:       3 panels (like 3LCD but reflective) → 3× parallelism
#   LCD (direct):     per-pixel RGB subpixels, backlight → 3× (spatial mux at subpixel)
#   OLED:             per-pixel RGBW emitters → 3-4× (self-emissive)
#   mini-LED LCD:     LCD + zone-dimming backlight → 3× + zone verification
#   microLED:         per-pixel RGB emitters (no backlight) → 3× (self-emissive)
#   Laser phosphor:   blue laser + phosphor wheel → 1× (sequential like DLP)
#   Terminal/GPU:     GPU shader → framebuffer → display controller → panel
#
# The channel parallelism determines whether the 3 color channels can be
# independently verified in a single trit-tick, or whether R/G/B must
# be verified sequentially across 3 trit-ticks (reducing effective rho by 3×).

"""
    RenderTech

Physical rendering technology that converts electrical signal to photons.

- `name`: human identifier
- `kind`: technology class
- `channel_parallel`: how many color channels are processed simultaneously
    (3 = R,G,B in parallel; 1 = sequential/time-multiplexed)
- `self_emissive`: true if each pixel produces its own light (OLED, microLED)
- `dimming_zones`: number of independent brightness zones (0 = none, global backlight)
- `color_sequential`: true if colors are produced by temporal multiplexing (DLP color wheel)
- `panel_persistence_us`: how long a pixel holds its state (LCD ~8000us, OLED ~200us)
- `gamut`: native color gamut coverage
"""
struct RenderTech
    name::String
    kind::Symbol         # :lcd_transmissive, :lcd_reflective, :dlp, :oled, :microled, :laser, :terminal
    channel_parallel::UInt8
    self_emissive::Bool
    dimming_zones::UInt32
    color_sequential::Bool
    panel_persistence_us::UInt32
    gamut::Symbol        # :srgb, :dcip3, :rec2020, :native
end

const RENDER_TECHS = (
    # Projectors
    epson_3lcd        = RenderTech("Epson 3LCD",            :lcd_transmissive,  3, false, 0,     false, 8000, :srgb),
    dlp_single_chip   = RenderTech("DLP Single-Chip",       :dlp,               1, false, 0,     true,  2000, :srgb),
    dlp_three_chip    = RenderTech("DLP 3-Chip",            :dlp,               3, false, 0,     false, 2000, :dcip3),
    lcos_3panel       = RenderTech("LCoS/SXRD 3-Panel",     :lcd_reflective,    3, false, 0,     false, 6000, :dcip3),
    laser_phosphor    = RenderTech("Laser Phosphor",         :laser,             1, false, 0,     true,  1000, :dcip3),
    # Direct-view displays
    lcd_ips           = RenderTech("IPS LCD",                :lcd_transmissive,  3, false, 1,     false, 8000, :srgb),
    lcd_miniled       = RenderTech("mini-LED LCD",           :lcd_transmissive,  3, false, 576,   false, 8000, :dcip3),
    lcd_miniled_xdr   = RenderTech("mini-LED LCD (XDR)",     :lcd_transmissive,  3, false, 2596,  false, 8000, :dcip3),
    oled              = RenderTech("OLED",                   :oled,              3, true,  0,     false, 200,  :dcip3),
    oled_tandem       = RenderTech("Tandem OLED",            :oled,              3, true,  0,     false, 200,  :dcip3),
    microled          = RenderTech("microLED",               :microled,          3, true,  0,     false, 100,  :rec2020),
    # Terminal (software rendering → GPU → display controller → panel)
    terminal_metal    = RenderTech("Terminal (Metal GPU)",    :terminal,          3, false, 0,     false, 0,    :native),
    terminal_opengl   = RenderTech("Terminal (OpenGL GPU)",   :terminal,          3, false, 0,     false, 0,    :native),
    terminal_cpu      = RenderTech("Terminal (CPU/sw)",       :terminal,          3, false, 0,     false, 0,    :srgb),
)

"""
    ColorSpace

Color space specification with gamut volume for fidelity computation.
Gamut volume is normalized: sRGB = 1.0.
"""
struct ColorSpace
    name::Symbol
    gamut_volume::Float64   # relative to sRGB (sRGB=1.0, DCI-P3~1.25, Rec.2020~1.77)
    bit_depth::UInt8        # typical bits per channel
    has_hdr::Bool
end

const COLOR_SPACES = (
    srgb    = ColorSpace(:srgb,    1.0,   8, false),
    dcip3   = ColorSpace(:dcip3,   1.254, 10, true),
    rec2020 = ColorSpace(:rec2020, 1.774, 12, true),
    native  = ColorSpace(:native,  1.0,   0, false),   # inherits from display
)

"""
    RenderStack

The complete rendering chain from computation to photon.
Each layer introduces constraints on what the trit-tick budget can verify.

    Computation → GPU API → Framebuffer → Transport → Renderer → Photon
    (Julia)     (Metal)   (P3/10-bit)   (HDMI/WiFi) (3LCD/OLED) (eye)
"""
struct RenderStack
    name::String
    gpu_api::Symbol          # :metal, :opengl, :vulkan, :cpu, :none
    framebuffer_space::Symbol  # :srgb, :dcip3, :rec2020
    transport::DisplayTransport
    renderer::RenderTech
    color_space::ColorSpace
end

"""
    channel_parallelism(stack::RenderStack) -> UInt8

Effective color channel parallelism of the full stack.
Bottlenecked by the least-parallel stage. If the renderer is
color-sequential (DLP single-chip), all 3 channels share one trit-tick
window, reducing effective verification by 3×.
"""
channel_parallelism(stack::RenderStack) = stack.renderer.channel_parallel

"""
    color_fidelity(stack::RenderStack) -> Float64

How much of the source color space survives the rendering stack.
Product of gamut coverage ratios at each lossy stage.
1.0 = perfect fidelity. <1.0 = gamut clipping.
"""
function color_fidelity(stack::RenderStack)
    src = get(pairs(COLOR_SPACES), stack.framebuffer_space, COLOR_SPACES.srgb)
    dst_gamut = stack.color_space.gamut_volume
    src_gamut = src.gamut_volume
    min(1.0, dst_gamut / src_gamut)
end

"""
    render_stack(name::Symbol) -> RenderStack

Predefined rendering stacks for the systems we have access to.
"""
function render_stack(name::Symbol)
    stacks = Dict(
        # This machine: M5 MacBook → Ghostty (Metal) → Liquid Retina XDR
        :local_ghostty_xdr => RenderStack(
            "Ghostty/Metal → Liquid Retina XDR",
            :metal, :dcip3,
            TRANSPORTS.usbc_tb4,    # internal, effectively zero-length
            RENDER_TECHS.lcd_miniled_xdr,
            COLOR_SPACES.dcip3,
        ),
        # This machine: M5 MacBook → Emacs (-nw) → Ghostty → XDR
        :local_emacs_ghostty_xdr => RenderStack(
            "Emacs(tty) → Ghostty/Metal → Liquid Retina XDR",
            :metal, :dcip3,
            TRANSPORTS.usbc_tb4,
            RENDER_TECHS.terminal_metal,
            COLOR_SPACES.dcip3,
        ),
        # Wireless: M5 MacBook → WiFi → Epson 3LCD projector
        :wifi_epson_3lcd => RenderStack(
            "MacBook → WiFi → Epson 3LCD",
            :metal, :srgb,          # iProjection compresses to sRGB
            TRANSPORTS.iprojection,
            RENDER_TECHS.epson_3lcd,
            COLOR_SPACES.srgb,
        ),
        # Wireless: Emacs → Ghostty → WiFi → Epson 3LCD
        :emacs_wifi_epson_3lcd => RenderStack(
            "Emacs(tty) → Ghostty → WiFi → Epson 3LCD",
            :metal, :srgb,
            TRANSPORTS.iprojection,
            RENDER_TECHS.epson_3lcd,
            COLOR_SPACES.srgb,
        ),
        # Wired: HDMI → DLP single-chip projector
        :hdmi_dlp_single => RenderStack(
            "HDMI → DLP Single-Chip",
            :none, :srgb,
            TRANSPORTS.hdmi_2_0,
            RENDER_TECHS.dlp_single_chip,
            COLOR_SPACES.srgb,
        ),
        # Wired: Thunderbolt → LCoS/SXRD (Sony VPL-class)
        :tb_lcos => RenderStack(
            "Thunderbolt → LCoS 3-Panel",
            :metal, :dcip3,
            TRANSPORTS.usbc_tb3,
            RENDER_TECHS.lcos_3panel,
            COLOR_SPACES.dcip3,
        ),
        # AirPlay → Apple TV → OLED TV
        :airplay_oled => RenderStack(
            "AirPlay → OLED TV",
            :metal, :dcip3,
            TRANSPORTS.airplay,
            RENDER_TECHS.oled,
            COLOR_SPACES.dcip3,
        ),
        # Air-gapped: QRTP → terminal CPU rendering
        :qrtp_terminal => RenderStack(
            "QRTP → Terminal (CPU)",
            :cpu, :srgb,
            TRANSPORTS.qrtp,
            RENDER_TECHS.terminal_cpu,
            COLOR_SPACES.srgb,
        ),
    )
    get(stacks, name) do
        error("Unknown render stack: $name. Known: $(join(sort(collect(keys(stacks))), ", "))")
    end
end

const KNOWN_STACKS = (
    :local_ghostty_xdr,
    :local_emacs_ghostty_xdr,
    :wifi_epson_3lcd,
    :emacs_wifi_epson_3lcd,
    :hdmi_dlp_single,
    :tb_lcos,
    :airplay_oled,
    :qrtp_terminal,
)
