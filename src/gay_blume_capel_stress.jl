# GAY BLUME-CAPEL STRESS TEST: Maximum Parallelism Ergodicity Mapping
# ═══════════════════════════════════════════════════════════════════════════════
#
# "2+1 optimistic/pessimistic/neutral balanced spin model with GayMC rollouts"
#
# ┌─────────────────────────────────────────────────────────────────────────────────┐
# │  BLUME-CAPEL MODEL × GAY COLOR DYNAMICS                                        │
# │                                                                                 │
# │  SPIN STATES: S ∈ {-1, 0, +1}                                                  │
# │    -1 = Pessimistic (contracting, NGMI, red)                                   │
# │     0 = Neutral (liminal, stable, yellow)                                      │
# │    +1 = Optimistic (expanding, GMI, green)                                     │
# │                                                                                 │
# │  HAMILTONIAN:                                                                   │
# │    H = -J Σ SᵢSⱼ + Δ Σ Sᵢ² + h Σ Sᵢ                                           │
# │    J = coupling (color coherence)                                              │
# │    Δ = crystal field (neutral preference)                                      │
# │    h = external field (optimism bias)                                          │
# │                                                                                 │
# │  ERGODICITY REGIONS:                                                            │
# │    Region I:   All spins accessible, full mixing                               │
# │    Region II:  Broken symmetry, ±1 domains                                     │
# │    Region III: Neutral dominant (Δ >> J)                                       │
# │                                                                                 │
# │  SEED BUNDLES:                                                                  │
# │    Small:  3 seeds (minimal)                                                   │
# │    Medium: 23 seeds (chromatic cycle)                                          │
# │    Large:  1069 seeds (gay complete)                                           │
# │                                                                                 │
# │  DIFFUSION MODEL RANKING (by bandwidth):                                        │
# │    Tier 1: Flux, SD3, SDXL (high bandwidth)                                    │
# │    Tier 2: SD2.1, DALL-E 3, Midjourney (medium)                                │
# │    Tier 3: SD1.5, older models (low bandwidth)                                 │
# │                                                                                 │
# │  LEARNABLE COLOR SPACE:                                                         │
# │    (p, q, x) number fields with -/0/+ flexibility                              │
# │    p-adic: ultrametric, hierarchical                                           │
# │    q-adic: ternary, balanced                                                   │
# │    x-adic: continuous interpolation                                            │
# │                                                                                 │
# └─────────────────────────────────────────────────────────────────────────────────┘

module GayBlumeCapelStress

using Base.Threads: @threads, @spawn, nthreads, Atomic
using LinearAlgebra: norm

export
    # Blume-Capel Model
    BlumeCapelSpin, BlumeCapelLattice, BlumeCapelParams,
    initialize_lattice!, metropolis_step!, measure_observables,
    
    # Ergodicity Detection
    ErgodicityRegion, FULL_MIXING, BROKEN_SYMMETRY, NEUTRAL_DOMINANT,
    detect_ergodicity, ergodicity_map,
    
    # Gay Seed Bundles
    SeedBundle, SMALL_BUNDLE, MEDIUM_BUNDLE, LARGE_BUNDLE,
    create_bundle, bundle_fingerprint,
    
    # ColorOps (Enzyme-differentiable)
    ColorOp, ColorOpChain, apply_colorop, differentiate_colorop,
    
    # Diffusion Model Ranking
    DiffusionModel, rank_by_bandwidth, DIFFUSION_MODELS,
    
    # Learnable Color Space
    PQXField, LearnableColorSpace, learn_colorspace!,
    
    # Stress Test
    VMInstance, StressTestEnsemble, launch_stress_test!,
    
    # Demo
    world_blume_capel_stress

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

const GAY_SEED = UInt64(1069)
const SMALL_BUNDLE_SIZE = 3
const MEDIUM_BUNDLE_SIZE = 23
const LARGE_BUNDLE_SIZE = 1069

# Blume-Capel critical parameters (approximate)
const BC_CRITICAL_DELTA = 0.655  # Tricritical point
const BC_CRITICAL_J = 1.0

# ═══════════════════════════════════════════════════════════════════════════════
# SPLITMIX64 (SPI-compliant, splittable)
# ═══════════════════════════════════════════════════════════════════════════════

mutable struct GayRNG
    state::UInt64
    invocation::UInt64
end

GayRNG(seed::UInt64=GAY_SEED) = GayRNG(seed, UInt64(0))

@inline function sm64!(rng::GayRNG)::UInt64
    rng.invocation += 1
    z = (rng.state + 0x9E3779B97F4A7C15)
    z = (z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ⊻ (z >> 27)) * 0x94D049BB133111EB
    rng.state = z ⊻ (z >> 31)
    rng.state
end

@inline function sm64_color!(rng::GayRNG)::NTuple{3, Float64}
    r = sm64!(rng)
    g = sm64!(rng)
    b = sm64!(rng)
    (Float64(r >> 56) / 255.0, Float64(g >> 56) / 255.0, Float64(b >> 56) / 255.0)
end

# Pure version (no mutation)
@inline function sm64(s::UInt64)::UInt64
    z = (s + 0x9E3779B97F4A7C15)
    z = (z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ⊻ (z >> 27)) * 0x94D049BB133111EB
    z ⊻ (z >> 31)
end

@inline sm64_color(s::UInt64)::NTuple{3, Float64} = begin
    r = sm64(s); g = sm64(r); b = sm64(g)
    (Float64(r >> 56) / 255.0, Float64(g >> 56) / 255.0, Float64(b >> 56) / 255.0)
end

# Control seed
function gay_seed!(rng::GayRNG, seed::UInt64)
    rng.state = seed
    rng.invocation = UInt64(0)
    rng
end

# Split for parallel
function split_rng(rng::GayRNG)::GayRNG
    new_seed = sm64!(rng) ⊻ rng.invocation
    GayRNG(new_seed)
end

# ═══════════════════════════════════════════════════════════════════════════════
# BLUME-CAPEL SPIN MODEL
# ═══════════════════════════════════════════════════════════════════════════════

@enum BlumeCapelSpin::Int8 begin
    PESSIMISTIC = -1  # S = -1
    NEUTRAL = 0       # S = 0
    OPTIMISTIC = 1    # S = +1
end

struct BlumeCapelParams
    J::Float64   # Coupling constant
    Δ::Float64   # Crystal field (vacancy preference)
    h::Float64   # External field (optimism bias)
    β::Float64   # Inverse temperature
end

BlumeCapelParams(; J=1.0, Δ=0.0, h=0.0, β=1.0) = BlumeCapelParams(J, Δ, h, β)

mutable struct BlumeCapelLattice
    L::Int  # Linear size
    spins::Matrix{BlumeCapelSpin}
    params::BlumeCapelParams
    rng::GayRNG
    energy::Float64
    magnetization::Float64
    fingerprint::UInt64
    colors::Matrix{NTuple{3, Float64}}
end

function BlumeCapelLattice(L::Int; params::BlumeCapelParams=BlumeCapelParams(), seed::UInt64=GAY_SEED)
    rng = GayRNG(seed)
    spins = Matrix{BlumeCapelSpin}(undef, L, L)
    colors = Matrix{NTuple{3, Float64}}(undef, L, L)
    
    # Initialize randomly
    for i in 1:L, j in 1:L
        r = sm64!(rng) % 3
        spins[i, j] = BlumeCapelSpin(r - 1)  # -1, 0, or 1
        colors[i, j] = spin_to_color(spins[i, j], rng)
    end
    
    lattice = BlumeCapelLattice(L, spins, params, rng, 0.0, 0.0, seed, colors)
    update_observables!(lattice)
    lattice
end

function spin_to_color(s::BlumeCapelSpin, rng::GayRNG)::NTuple{3, Float64}
    base = sm64_color!(rng)
    if s == PESSIMISTIC
        (0.9, base[2] * 0.3, base[3] * 0.3)  # Red-tinted
    elseif s == OPTIMISTIC
        (base[1] * 0.3, 0.9, base[3] * 0.3)  # Green-tinted
    else
        (0.9, 0.9, base[3] * 0.3)  # Yellow-tinted (neutral)
    end
end

function local_energy(lattice::BlumeCapelLattice, i::Int, j::Int)::Float64
    L = lattice.L
    s = Int(lattice.spins[i, j])
    p = lattice.params
    
    # Neighbors (periodic BC)
    neighbors = [
        Int(lattice.spins[mod1(i+1, L), j]),
        Int(lattice.spins[mod1(i-1, L), j]),
        Int(lattice.spins[i, mod1(j+1, L)]),
        Int(lattice.spins[i, mod1(j-1, L)])
    ]
    
    # H = -J Σ SᵢSⱼ + Δ Sᵢ² - h Sᵢ
    interaction = -p.J * s * sum(neighbors)
    crystal = p.Δ * s^2
    field = -p.h * s
    
    interaction + crystal + field
end

function total_energy(lattice::BlumeCapelLattice)::Float64
    E = 0.0
    L = lattice.L
    p = lattice.params
    
    for i in 1:L, j in 1:L
        s = Int(lattice.spins[i, j])
        # Only count right and down neighbors to avoid double counting
        s_right = Int(lattice.spins[mod1(i+1, L), j])
        s_down = Int(lattice.spins[i, mod1(j+1, L)])
        
        E += -p.J * s * (s_right + s_down)
        E += p.Δ * s^2
        E += -p.h * s
    end
    E
end

function total_magnetization(lattice::BlumeCapelLattice)::Float64
    sum(Int(s) for s in lattice.spins) / lattice.L^2
end

function update_observables!(lattice::BlumeCapelLattice)
    lattice.energy = total_energy(lattice)
    lattice.magnetization = total_magnetization(lattice)
    
    # Update fingerprint
    fp = UInt64(0)
    for s in lattice.spins
        fp = sm64(fp ⊻ UInt64(Int(s) + 2))
    end
    lattice.fingerprint = fp
end

# ═══════════════════════════════════════════════════════════════════════════════
# METROPOLIS DYNAMICS
# ═══════════════════════════════════════════════════════════════════════════════

function metropolis_step!(lattice::BlumeCapelLattice)::Bool
    L = lattice.L
    i = Int(sm64!(lattice.rng) % L) + 1
    j = Int(sm64!(lattice.rng) % L) + 1
    
    old_spin = lattice.spins[i, j]
    old_energy = local_energy(lattice, i, j)
    
    # Propose new spin (uniform over {-1, 0, +1} excluding current)
    r = sm64!(lattice.rng) % 2
    new_spin = if old_spin == PESSIMISTIC
        r == 0 ? NEUTRAL : OPTIMISTIC
    elseif old_spin == OPTIMISTIC
        r == 0 ? NEUTRAL : PESSIMISTIC
    else
        r == 0 ? PESSIMISTIC : OPTIMISTIC
    end
    
    lattice.spins[i, j] = new_spin
    new_energy = local_energy(lattice, i, j)
    
    ΔE = new_energy - old_energy
    
    # Metropolis acceptance
    accepted = if ΔE <= 0
        true
    else
        u = Float64(sm64!(lattice.rng) >> 11) / Float64(1 << 53)
        u < exp(-lattice.params.β * ΔE)
    end
    
    if accepted
        lattice.colors[i, j] = spin_to_color(new_spin, lattice.rng)
        true
    else
        lattice.spins[i, j] = old_spin
        false
    end
end

function sweep!(lattice::BlumeCapelLattice, n_sweeps::Int=1)
    n_sites = lattice.L^2
    for _ in 1:n_sweeps * n_sites
        metropolis_step!(lattice)
    end
    update_observables!(lattice)
    lattice
end

# ═══════════════════════════════════════════════════════════════════════════════
# ERGODICITY DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

@enum ErgodicityRegion begin
    FULL_MIXING = 1       # All spins accessible
    BROKEN_SYMMETRY = 2   # ±1 domains separated
    NEUTRAL_DOMINANT = 3  # 0 spins dominate
end

struct ErgodicityMeasurement
    region::ErgodicityRegion
    mixing_time::Int
    magnetization_variance::Float64
    neutral_fraction::Float64
    fingerprint::UInt64
end

function detect_ergodicity(lattice::BlumeCapelLattice; n_samples::Int=1000, burn_in::Int=100)::ErgodicityMeasurement
    # Burn-in
    sweep!(lattice, burn_in)
    
    magnetizations = Float64[]
    neutral_counts = Int[]
    
    for _ in 1:n_samples
        sweep!(lattice, 1)
        push!(magnetizations, lattice.magnetization)
        n_neutral = count(s -> s == NEUTRAL, lattice.spins)
        push!(neutral_counts, n_neutral)
    end
    
    # Statistics
    mean_m = sum(magnetizations) / n_samples
    var_m = sum((m - mean_m)^2 for m in magnetizations) / n_samples
    mean_neutral = sum(neutral_counts) / n_samples / lattice.L^2
    
    # Classify region
    region = if mean_neutral > 0.5
        NEUTRAL_DOMINANT
    elseif var_m < 0.01
        BROKEN_SYMMETRY
    else
        FULL_MIXING
    end
    
    # Estimate mixing time (autocorrelation decay)
    mixing_time = estimate_mixing_time(magnetizations)
    
    ErgodicityMeasurement(region, mixing_time, var_m, mean_neutral, lattice.fingerprint)
end

function estimate_mixing_time(series::Vector{Float64})::Int
    n = length(series)
    mean_s = sum(series) / n
    var_s = sum((s - mean_s)^2 for s in series) / n
    
    if var_s < 1e-10
        return n  # Stuck
    end
    
    # Find lag where autocorrelation drops below 1/e
    for lag in 1:min(n÷2, 100)
        autocorr = 0.0
        for i in 1:n-lag
            autocorr += (series[i] - mean_s) * (series[i+lag] - mean_s)
        end
        autocorr /= (n - lag) * var_s
        
        if autocorr < 1/ℯ
            return lag
        end
    end
    
    return min(n÷2, 100)
end

function ergodicity_map(; L::Int=16, n_J::Int=10, n_Δ::Int=10, seed::UInt64=GAY_SEED)
    J_range = range(0.1, 2.0, length=n_J)
    Δ_range = range(-1.0, 2.0, length=n_Δ)
    
    results = Matrix{ErgodicityMeasurement}(undef, n_J, n_Δ)
    
    @threads for idx in 1:n_J * n_Δ
        i = (idx - 1) ÷ n_Δ + 1
        j = (idx - 1) % n_Δ + 1
        
        J = J_range[i]
        Δ = Δ_range[j]
        
        params = BlumeCapelParams(J=J, Δ=Δ, h=0.0, β=1.0)
        lattice = BlumeCapelLattice(L; params=params, seed=sm64(seed ⊻ UInt64(idx)))
        
        results[i, j] = detect_ergodicity(lattice; n_samples=500, burn_in=50)
    end
    
    (J_range=J_range, Δ_range=Δ_range, measurements=results)
end

# ═══════════════════════════════════════════════════════════════════════════════
# GAY SEED BUNDLES
# ═══════════════════════════════════════════════════════════════════════════════

struct SeedBundle
    size::Int
    seeds::Vector{UInt64}
    colors::Vector{NTuple{3, Float64}}
    fingerprint::UInt64
    tier::Symbol  # :small, :medium, :large
end

function create_bundle(size::Int; base_seed::UInt64=GAY_SEED)::SeedBundle
    seeds = UInt64[]
    colors = NTuple{3, Float64}[]
    
    current = base_seed
    for i in 1:size
        current = sm64(current ⊻ UInt64(i))
        push!(seeds, current)
        push!(colors, sm64_color(current))
    end
    
    fp = reduce(⊻, seeds; init=UInt64(0))
    
    tier = if size <= SMALL_BUNDLE_SIZE
        :small
    elseif size <= MEDIUM_BUNDLE_SIZE
        :medium
    else
        :large
    end
    
    SeedBundle(size, seeds, colors, fp, tier)
end

const SMALL_BUNDLE = create_bundle(SMALL_BUNDLE_SIZE)
const MEDIUM_BUNDLE = create_bundle(MEDIUM_BUNDLE_SIZE)
const LARGE_BUNDLE = create_bundle(LARGE_BUNDLE_SIZE)

bundle_fingerprint(b::SeedBundle) = b.fingerprint

# ═══════════════════════════════════════════════════════════════════════════════
# COLOR OPS (Enzyme-differentiable stubs)
# ═══════════════════════════════════════════════════════════════════════════════

abstract type ColorOp end

struct ColorShift <: ColorOp
    delta::NTuple{3, Float64}
end

struct ColorScale <: ColorOp
    scale::NTuple{3, Float64}
end

struct ColorRotate <: ColorOp
    angle::Float64  # Hue rotation in radians
end

struct ColorClamp <: ColorOp
    min::NTuple{3, Float64}
    max::NTuple{3, Float64}
end

struct ColorOpChain
    ops::Vector{ColorOp}
    fingerprint::UInt64
end

function ColorOpChain(ops::Vector{<:ColorOp}; seed::UInt64=GAY_SEED)
    fp = seed
    for op in ops
        fp = sm64(fp ⊻ hash(op))
    end
    ColorOpChain(ops, fp)
end

function apply_colorop(op::ColorShift, c::NTuple{3, Float64})::NTuple{3, Float64}
    (clamp(c[1] + op.delta[1], 0, 1),
     clamp(c[2] + op.delta[2], 0, 1),
     clamp(c[3] + op.delta[3], 0, 1))
end

function apply_colorop(op::ColorScale, c::NTuple{3, Float64})::NTuple{3, Float64}
    (clamp(c[1] * op.scale[1], 0, 1),
     clamp(c[2] * op.scale[2], 0, 1),
     clamp(c[3] * op.scale[3], 0, 1))
end

function apply_colorop(op::ColorRotate, c::NTuple{3, Float64})::NTuple{3, Float64}
    # Simplified hue rotation (proper would use HSL)
    cos_a, sin_a = cos(op.angle), sin(op.angle)
    r = c[1] * cos_a - c[2] * sin_a
    g = c[1] * sin_a + c[2] * cos_a
    (clamp(r, 0, 1), clamp(g, 0, 1), c[3])
end

function apply_colorop(op::ColorClamp, c::NTuple{3, Float64})::NTuple{3, Float64}
    (clamp(c[1], op.min[1], op.max[1]),
     clamp(c[2], op.min[2], op.max[2]),
     clamp(c[3], op.min[3], op.max[3]))
end

function apply_colorop(chain::ColorOpChain, c::NTuple{3, Float64})::NTuple{3, Float64}
    result = c
    for op in chain.ops
        result = apply_colorop(op, result)
    end
    result
end

# Enzyme.jl stub for differentiation
function differentiate_colorop(chain::ColorOpChain, c::NTuple{3, Float64})::Matrix{Float64}
    # Numerical differentiation (real impl would use Enzyme)
    ε = 1e-6
    J = zeros(3, 3)
    
    for i in 1:3
        c_plus = ntuple(k -> k == i ? c[k] + ε : c[k], 3)
        c_minus = ntuple(k -> k == i ? c[k] - ε : c[k], 3)
        
        out_plus = apply_colorop(chain, c_plus)
        out_minus = apply_colorop(chain, c_minus)
        
        for j in 1:3
            J[j, i] = (out_plus[j] - out_minus[j]) / (2ε)
        end
    end
    
    J
end

# ═══════════════════════════════════════════════════════════════════════════════
# DIFFUSION MODEL RANKING
# ═══════════════════════════════════════════════════════════════════════════════

struct DiffusionModel
    name::String
    params_b::Float64  # Billions of parameters
    bandwidth::Float64  # Estimated color bandwidth (0-1)
    tier::Symbol  # :high, :medium, :low
    public::Bool
    seed::UInt64
    color::NTuple{3, Float64}
end

function DiffusionModel(name::String; params_b::Float64, bandwidth::Float64, public::Bool=true)
    seed = reduce(⊻, [UInt64(b) for b in codeunits(name)]; init=GAY_SEED)
    tier = if bandwidth >= 0.8
        :high
    elseif bandwidth >= 0.5
        :medium
    else
        :low
    end
    DiffusionModel(name, params_b, bandwidth, tier, public, seed, sm64_color(seed))
end

const DIFFUSION_MODELS = [
    # Tier 1: High bandwidth (public)
    DiffusionModel("FLUX.1-dev"; params_b=12.0, bandwidth=0.95),
    DiffusionModel("FLUX.1-schnell"; params_b=12.0, bandwidth=0.93),
    DiffusionModel("SD3-Medium"; params_b=2.0, bandwidth=0.88),
    DiffusionModel("SDXL"; params_b=6.6, bandwidth=0.85),
    DiffusionModel("Playground-v2.5"; params_b=2.5, bandwidth=0.82),
    
    # Tier 2: Medium bandwidth
    DiffusionModel("SD2.1"; params_b=1.5, bandwidth=0.72),
    DiffusionModel("Kandinsky-3"; params_b=4.0, bandwidth=0.70),
    DiffusionModel("DeepFloyd-IF"; params_b=4.3, bandwidth=0.68),
    DiffusionModel("Wuerstchen"; params_b=1.0, bandwidth=0.65),
    DiffusionModel("PixArt-α"; params_b=0.6, bandwidth=0.62),
    
    # Tier 3: Lower bandwidth
    DiffusionModel("SD1.5"; params_b=0.9, bandwidth=0.55),
    DiffusionModel("OpenJourney"; params_b=0.9, bandwidth=0.52),
    DiffusionModel("Dreamlike-Diffusion"; params_b=0.9, bandwidth=0.48),
    DiffusionModel("Deliberate"; params_b=0.9, bandwidth=0.45),
    
    # Non-public (for comparison)
    DiffusionModel("DALL-E 3"; params_b=12.0, bandwidth=0.90, public=false),
    DiffusionModel("Midjourney-v6"; params_b=10.0, bandwidth=0.92, public=false),
    DiffusionModel("Imagen-3"; params_b=20.0, bandwidth=0.94, public=false),
]

function rank_by_bandwidth(models::Vector{DiffusionModel}; public_only::Bool=true)
    filtered = public_only ? filter(m -> m.public, models) : models
    sort(filtered, by=m -> m.bandwidth, rev=true)
end

function group_by_tier(models::Vector{DiffusionModel})
    high = filter(m -> m.tier == :high, models)
    medium = filter(m -> m.tier == :medium, models)
    low = filter(m -> m.tier == :low, models)
    (high=high, medium=medium, low=low)
end

# ═══════════════════════════════════════════════════════════════════════════════
# LEARNABLE COLOR SPACE (p, q, x) NUMBER FIELDS
# ═══════════════════════════════════════════════════════════════════════════════

struct PQXField
    p::Int  # p-adic prime (hierarchical)
    q::Int  # q-adic prime (ternary)
    x::Float64  # Continuous interpolation
    valuation::Function  # v_p(n) = max k such that p^k | n
end

function PQXField(; p::Int=2, q::Int=3, x::Float64=0.5)
    valuation = n -> begin
        if n == 0
            return Inf
        end
        k = 0
        while n % p == 0
            n ÷= p
            k += 1
        end
        k
    end
    PQXField(p, q, x, valuation)
end

mutable struct LearnableColorSpace
    field::PQXField
    
    # Learnable parameters
    basis::Matrix{Float64}  # 3x3 transformation matrix
    offset::NTuple{3, Float64}
    scale::NTuple{3, Float64}
    
    # Training state
    loss_history::Vector{Float64}
    step::Int
    
    fingerprint::UInt64
end

function LearnableColorSpace(; field::PQXField=PQXField(), seed::UInt64=GAY_SEED)
    # Initialize basis near identity
    rng = GayRNG(seed)
    basis = Matrix{Float64}(I, 3, 3)
    for i in 1:3, j in 1:3
        basis[i, j] += (Float64(sm64!(rng) >> 32) / typemax(UInt32) - 0.5) * 0.1
    end
    
    offset = (0.0, 0.0, 0.0)
    scale = (1.0, 1.0, 1.0)
    
    LearnableColorSpace(field, basis, offset, scale, Float64[], 0, seed)
end

function forward(lcs::LearnableColorSpace, c::NTuple{3, Float64})::NTuple{3, Float64}
    v = [c[1], c[2], c[3]]
    
    # Apply basis transformation
    v = lcs.basis * v
    
    # Apply offset and scale
    result = (
        clamp(v[1] * lcs.scale[1] + lcs.offset[1], 0, 1),
        clamp(v[2] * lcs.scale[2] + lcs.offset[2], 0, 1),
        clamp(v[3] * lcs.scale[3] + lcs.offset[3], 0, 1)
    )
    
    result
end

function learn_colorspace!(lcs::LearnableColorSpace, inputs::Vector{NTuple{3, Float64}}, 
                           targets::Vector{NTuple{3, Float64}}; 
                           lr::Float64=0.01, n_steps::Int=100)
    for step in 1:n_steps
        lcs.step += 1
        total_loss = 0.0
        
        for (input, target) in zip(inputs, targets)
            output = forward(lcs, input)
            
            # L2 loss
            loss = sum((output[i] - target[i])^2 for i in 1:3)
            total_loss += loss
            
            # Gradient descent (simplified - would use Enzyme for real)
            for i in 1:3
                error = output[i] - target[i]
                for j in 1:3
                    lcs.basis[i, j] -= lr * error * input[j]
                end
            end
        end
        
        push!(lcs.loss_history, total_loss / length(inputs))
    end
    
    lcs.fingerprint = sm64(lcs.fingerprint ⊻ UInt64(round(lcs.loss_history[end] * 1e9)))
    lcs
end

# ═══════════════════════════════════════════════════════════════════════════════
# VM STRESS TEST
# ═══════════════════════════════════════════════════════════════════════════════

mutable struct VMInstance
    id::Int
    rng::GayRNG
    lattice::BlumeCapelLattice
    ops_count::Atomic{Int}
    next_color_count::Atomic{Int}
    next_color_mut_count::Atomic{Int}
    fingerprint::UInt64
    
    # Observables
    colors_generated::Vector{NTuple{3, Float64}}
    ergodicity::Union{ErgodicityMeasurement, Nothing}
end

function VMInstance(id::Int; L::Int=8, seed::UInt64=GAY_SEED)
    vm_seed = sm64(seed ⊻ UInt64(id))
    rng = GayRNG(vm_seed)
    lattice = BlumeCapelLattice(L; seed=vm_seed)
    
    VMInstance(
        id, rng, lattice,
        Atomic{Int}(0), Atomic{Int}(0), Atomic{Int}(0),
        vm_seed, NTuple{3, Float64}[], nothing
    )
end

# next_color() - pure
function next_color(vm::VMInstance)::NTuple{3, Float64}
    vm.next_color_count[] += 1
    sm64_color(vm.fingerprint ⊻ UInt64(vm.next_color_count[]))
end

# next_color!() - mutating
function next_color!(vm::VMInstance)::NTuple{3, Float64}
    vm.next_color_mut_count[] += 1
    color = sm64_color!(vm.rng)
    push!(vm.colors_generated, color)
    vm.fingerprint = sm64(vm.fingerprint ⊻ UInt64(round(color[1] * 1e9)))
    color
end

# gay_seed!() - control
function gay_seed!(vm::VMInstance, seed::UInt64)
    gay_seed!(vm.rng, seed)
    vm.fingerprint = seed
end

mutable struct StressTestEnsemble
    vms::Vector{VMInstance}
    total_ops::Atomic{Int}
    combined_fingerprint::UInt64
    start_time::Float64
    elapsed::Float64
    ops_per_second::Float64
    bundle::SeedBundle
end

function StressTestEnsemble(n_vms::Int; bundle::SeedBundle=MEDIUM_BUNDLE, seed::UInt64=GAY_SEED)
    vms = [VMInstance(i; seed=sm64(seed ⊻ UInt64(i))) for i in 1:n_vms]
    StressTestEnsemble(vms, Atomic{Int}(0), seed, 0.0, 0.0, 0.0, bundle)
end

function launch_stress_test!(ensemble::StressTestEnsemble; 
                             n_steps::Int=10000, 
                             mode::Symbol=:balanced)
    ensemble.start_time = time()
    
    @threads for vm in ensemble.vms
        # Assign seed from bundle
        bundle_idx = mod1(vm.id, ensemble.bundle.size)
        gay_seed!(vm, ensemble.bundle.seeds[bundle_idx])
        
        for step in 1:n_steps
            # Balanced: 2 optimistic, 1 pessimistic, 1 neutral
            if mode == :balanced
                if step % 4 <= 1
                    # Optimistic: next_color! (mutating, generates new)
                    next_color!(vm)
                elseif step % 4 == 2
                    # Pessimistic: next_color (pure, no state change)
                    next_color(vm)
                else
                    # Neutral: metropolis step
                    metropolis_step!(vm.lattice)
                end
            elseif mode == :aggressive
                # All mutating
                next_color!(vm)
                metropolis_step!(vm.lattice)
            else  # :conservative
                # All pure
                next_color(vm)
            end
            
            vm.ops_count[] += 1
            ensemble.total_ops[] += 1
        end
        
        # Measure ergodicity at end
        vm.ergodicity = detect_ergodicity(vm.lattice; n_samples=100, burn_in=10)
    end
    
    ensemble.elapsed = time() - ensemble.start_time
    ensemble.ops_per_second = ensemble.total_ops[] / ensemble.elapsed
    ensemble.combined_fingerprint = reduce(⊻, [vm.fingerprint for vm in ensemble.vms])
    
    ensemble
end

# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

function world_blume_capel_stress()
    println()
    println("╔═══════════════════════════════════════════════════════════════════════════════════╗")
    println("║  GAY BLUME-CAPEL STRESS TEST                                                      ║")
    println("║  2+1 Balanced Spin Model × Maximum Parallelism × Ergodicity Mapping              ║")
    println("╚═══════════════════════════════════════════════════════════════════════════════════╝")
    println()
    
    # ─── Seed Bundles ───
    println("─── Gay Seed Bundles ───")
    println("  Small (3):    fp=0x$(string(SMALL_BUNDLE.fingerprint, base=16))")
    println("  Medium (23):  fp=0x$(string(MEDIUM_BUNDLE.fingerprint, base=16))")
    println("  Large (1069): fp=0x$(string(LARGE_BUNDLE.fingerprint, base=16))")
    println()
    
    # ─── Diffusion Model Ranking ───
    println("─── Diffusion Models (Public, Ranked by Bandwidth) ───")
    ranked = rank_by_bandwidth(DIFFUSION_MODELS; public_only=true)
    
    for (i, m) in enumerate(ranked[1:min(5, length(ranked))])
        emoji = m.tier == :high ? "🟢" : m.tier == :medium ? "🟡" : "🔴"
        println("  #$i $emoji $(m.name): bandwidth=$(m.bandwidth), params=$(m.params_b)B")
    end
    
    tiers = group_by_tier(filter(m -> m.public, DIFFUSION_MODELS))
    println("  Tier distribution: High=$(length(tiers.high)), Medium=$(length(tiers.medium)), Low=$(length(tiers.low))")
    println()
    
    # ─── Blume-Capel Lattice ───
    println("─── Blume-Capel Lattice (L=16) ───")
    params = BlumeCapelParams(J=1.0, Δ=0.3, h=0.0, β=1.0)
    lattice = BlumeCapelLattice(16; params=params)
    
    println("  Initial energy: $(round(lattice.energy, digits=2))")
    println("  Initial magnetization: $(round(lattice.magnetization, digits=4))")
    
    sweep!(lattice, 100)
    println("  After 100 sweeps:")
    println("    Energy: $(round(lattice.energy, digits=2))")
    println("    Magnetization: $(round(lattice.magnetization, digits=4))")
    println("    Fingerprint: 0x$(string(lattice.fingerprint, base=16))")
    println()
    
    # ─── Ergodicity Detection ───
    println("─── Ergodicity Detection ───")
    erg = detect_ergodicity(lattice; n_samples=500, burn_in=50)
    
    println("  Region: $(erg.region)")
    println("  Mixing time: $(erg.mixing_time) sweeps")
    println("  Magnetization variance: $(round(erg.magnetization_variance, digits=4))")
    println("  Neutral fraction: $(round(erg.neutral_fraction, digits=4))")
    println()
    
    # ─── ColorOp Chain ───
    println("─── ColorOp Chain (Enzyme-differentiable) ───")
    chain = ColorOpChain([
        ColorShift((0.1, -0.05, 0.0)),
        ColorScale((1.2, 0.9, 1.0)),
        ColorRotate(0.1)
    ])
    
    test_color = (0.5, 0.5, 0.5)
    output_color = apply_colorop(chain, test_color)
    jacobian = differentiate_colorop(chain, test_color)
    
    println("  Input:  RGB$(Int.(round.(test_color .* 255)))")
    println("  Output: RGB$(Int.(round.(output_color .* 255)))")
    println("  Jacobian trace: $(round(sum(jacobian[i,i] for i in 1:3), digits=3))")
    println()
    
    # ─── Learnable Color Space ───
    println("─── Learnable Color Space (p=2, q=3) ───")
    lcs = LearnableColorSpace(field=PQXField(p=2, q=3, x=0.5))
    
    # Generate training data
    inputs = [sm64_color(sm64(GAY_SEED ⊻ UInt64(i))) for i in 1:100]
    targets = [apply_colorop(chain, c) for c in inputs]
    
    learn_colorspace!(lcs, inputs, targets; lr=0.01, n_steps=50)
    
    println("  Training steps: $(lcs.step)")
    println("  Final loss: $(round(lcs.loss_history[end], digits=6))")
    println("  Basis determinant: $(round(det(lcs.basis), digits=4))")
    println()
    
    # ─── Stress Test ───
    println("─── VM Stress Test ($(nthreads()) threads) ───")
    
    ensemble = StressTestEnsemble(nthreads() * 4; bundle=MEDIUM_BUNDLE)
    launch_stress_test!(ensemble; n_steps=5000, mode=:balanced)
    
    println("  VMs: $(length(ensemble.vms))")
    println("  Total ops: $(ensemble.total_ops[])")
    println("  Elapsed: $(round(ensemble.elapsed, digits=3))s")
    println("  Ops/sec: $(round(ensemble.ops_per_second / 1000, digits=1))K")
    println("  Combined fingerprint: 0x$(string(ensemble.combined_fingerprint, base=16))")
    
    # Ergodicity distribution
    erg_counts = Dict(FULL_MIXING => 0, BROKEN_SYMMETRY => 0, NEUTRAL_DOMINANT => 0)
    for vm in ensemble.vms
        if vm.ergodicity !== nothing
            erg_counts[vm.ergodicity.region] += 1
        end
    end
    println("  Ergodicity: Full=$(erg_counts[FULL_MIXING]), Broken=$(erg_counts[BROKEN_SYMMETRY]), Neutral=$(erg_counts[NEUTRAL_DOMINANT])")
    println()
    
    # ─── Summary ───
    println("═══════════════════════════════════════════════════════════════════════════════════")
    println("  SUMMARY: Blume-Capel × Gay Color Dynamics")
    println()
    println("  • 3 seed bundles: 3, 23, 1069 seeds ✓")
    println("  • $(length(filter(m -> m.public, DIFFUSION_MODELS))) public diffusion models ranked ✓")
    println("  • Blume-Capel: S ∈ {-1, 0, +1} with ergodicity detection ✓")
    println("  • ColorOps: differentiable chain with Jacobian ✓")
    println("  • LearnableColorSpace: (p,q,x) number field basis ✓")
    println("  • Stress test: $(round(ensemble.ops_per_second / 1000, digits=1))K ops/sec ✓")
    println()
    println("  \"2+1 optimistic/pessimistic/neutral: the balanced spin of creativity\"")
    println("═══════════════════════════════════════════════════════════════════════════════════")
    
    (ensemble=ensemble, lattice=lattice, lcs=lcs, ranked=ranked)
end

end # module GayBlumeCapelStress
