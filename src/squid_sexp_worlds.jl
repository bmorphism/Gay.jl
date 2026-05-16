# SQUID S-Expression Worlds: Maximally Parallel Gravity-Magnetism Exploration
# ===========================================================================
#
# Exploring the phase space of:
# - High gravity / low magnetism (HG-LM): collapsed, dense color structures
# - Low gravity / high magnetism (LG-HM): dispersed, cryptochrome-sensitive structures
# - SQUID sensing: flux quantization Φ₀ = h/2e ≈ 2.07×10⁻¹⁵ Wb
#
# Colored S-expressions represent world states with O(1) SPI access.
# Maximum parallelism via:
# - Splittable RNG (embarrassingly parallel)
# - Metal GPU kernels (SIMD color generation)
# - Task spawning (multi-world exploration)

module SQUIDSexpWorlds

using Base.Threads: @threads, nthreads, threadid
using SplittableRandoms: SplittableRandom, split

export GaySexp, SexpWorld, SQUIDSensor, GravityMagnetismRegime
export sexp_color, parallel_explore!, squid_measure
export world_squid_worlds, run_max_parallel_experiment

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

const FLUX_QUANTUM = 2.067833848e-15  # Φ₀ = h/2e in Weber
const PLANCK = 6.62607015e-34         # h in J·s
const ELECTRON_CHARGE = 1.602176634e-19  # e in Coulombs
const GAY_SEED = UInt64(1069)

# Gravity-Magnetism regime bounds
const GRAVITY_RANGE = (1e-6, 1e6)      # m/s² (micro-g to mega-g)
const MAGNETISM_RANGE = (1e-15, 1e3)   # Tesla (femto-T to kilo-T)

# ═══════════════════════════════════════════════════════════════════════════════
# Gay S-Expression: Colored Lisp-like Structure
# ═══════════════════════════════════════════════════════════════════════════════

"""
    GaySexp

A colored S-expression with deterministic chromatic identity.

Structure: (color . (car . cdr))
- Atom: leaf node with color
- Cons: pair with two children, colored by combination
"""
abstract type GaySexp end

struct GayAtom <: GaySexp
    value::Any
    seed::UInt64
    color::NTuple{3, Float64}  # RGB
end

struct GayCons <: GaySexp
    car::GaySexp
    cdr::GaySexp
    seed::UInt64
    color::NTuple{3, Float64}
end

struct GayNil <: GaySexp
    seed::UInt64
    color::NTuple{3, Float64}
end

# SPI color generation
function splitmix64(seed::UInt64)::UInt64
    z = seed + 0x9e3779b97f4a7c15
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    z ⊻ (z >> 31)
end

function seed_to_color(seed::UInt64)::NTuple{3, Float64}
    state = splitmix64(seed)
    r = (state & 0xFFFF) / 65535.0
    g = ((state >> 16) & 0xFFFF) / 65535.0
    b = ((state >> 32) & 0xFFFF) / 65535.0
    (r, g, b)
end

function GayAtom(value::Any; seed::UInt64=GAY_SEED)
    combined = seed ⊻ UInt64(hash(value))
    GayAtom(value, combined, seed_to_color(combined))
end

function GayCons(car::GaySexp, cdr::GaySexp; seed::UInt64=GAY_SEED)
    combined = seed ⊻ car.seed ⊻ (cdr.seed << 1)
    GayCons(car, cdr, combined, seed_to_color(combined))
end

function GayNil(; seed::UInt64=GAY_SEED)
    GayNil(seed, seed_to_color(seed))
end

# S-expression constructors
gay_list(items...; seed::UInt64=GAY_SEED) = foldr((x, acc) -> GayCons(GayAtom(x; seed=seed), acc; seed=seed), items; init=GayNil(; seed=seed))

function sexp_to_string(s::GaySexp)::String
    if s isa GayNil
        "nil"
    elseif s isa GayAtom
        string(s.value)
    elseif s isa GayCons
        "(" * sexp_to_string(s.car) * " . " * sexp_to_string(s.cdr) * ")"
    end
end

function sexp_color(s::GaySexp)::NTuple{3, Float64}
    s.color
end

function sexp_depth(s::GaySexp)::Int
    if s isa GayNil || s isa GayAtom
        0
    else
        1 + max(sexp_depth(s.car), sexp_depth(s.cdr))
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Gravity-Magnetism Regime
# ═══════════════════════════════════════════════════════════════════════════════

"""
    GravityMagnetismRegime

A point in the gravity-magnetism phase space.

High gravity → dense, collapsed color structures (red-shifted)
High magnetism → dispersed, quantum-coherent structures (blue-shifted via cryptochrome)
"""
struct GravityMagnetismRegime
    gravity::Float64      # m/s² (log scale)
    magnetism::Float64    # Tesla (log scale)
    seed::UInt64
    
    # Derived properties
    density::Float64      # Higher gravity → higher density
    coherence::Float64    # Higher magnetism → higher quantum coherence
    color_shift::Float64  # -1 (red) to +1 (blue)
end

function GravityMagnetismRegime(gravity::Float64, magnetism::Float64; seed::UInt64=GAY_SEED)
    # Normalize to log scale
    g_norm = log10(clamp(gravity, GRAVITY_RANGE...)) / 6  # -1 to 1
    m_norm = log10(clamp(magnetism, MAGNETISM_RANGE...)) / 18 + 0.5  # 0 to 1
    
    # Derived properties
    density = 1 / (1 + exp(-5 * g_norm))  # Sigmoid of gravity
    coherence = m_norm  # Linear with magnetism
    
    # Color shift: high-G/low-M → red, low-G/high-M → blue
    color_shift = m_norm - (g_norm + 1) / 2
    
    GravityMagnetismRegime(gravity, magnetism, seed, density, coherence, color_shift)
end

function regime_color(r::GravityMagnetismRegime)::NTuple{3, Float64}
    # Red for high gravity, blue for high magnetism
    base = seed_to_color(r.seed)
    shift = r.color_shift
    
    if shift > 0  # Blue shift (high magnetism)
        (base[1] * (1 - shift), base[2], min(1.0, base[3] + shift * 0.5))
    else  # Red shift (high gravity)
        (min(1.0, base[1] - shift * 0.5), base[2], base[3] * (1 + shift))
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# SQUID Sensor: Superconducting Quantum Interference
# ═══════════════════════════════════════════════════════════════════════════════

"""
    SQUIDSensor

Superconducting Quantum Interference Device for measuring flux quanta.

SQUID detects magnetic flux with resolution ~ Φ₀ = h/2e ≈ 2×10⁻¹⁵ Wb

In our chromatic framework:
- Flux quanta map to color phase
- Josephson frequency maps to hue oscillation
- Critical current maps to saturation
"""
struct SQUIDSensor
    flux_bias::Float64       # Φ/Φ₀ (number of flux quanta)
    josephson_freq::Float64  # Hz (color oscillation rate)
    critical_current::Float64  # Amperes (saturation threshold)
    seed::UInt64
    
    # Sensing state
    measured_flux::Float64
    color_phase::Float64     # 0-2π
end

function SQUIDSensor(; flux_bias::Float64=0.5, seed::UInt64=GAY_SEED)
    # Josephson frequency: f = 2eV/h where V ~ flux_bias
    josephson_freq = 2 * ELECTRON_CHARGE * flux_bias / PLANCK * 1e-15  # Scale to reasonable Hz
    critical_current = 1e-6 * (1 + flux_bias)  # Microamps
    
    SQUIDSensor(flux_bias, josephson_freq, critical_current, seed, 0.0, 0.0)
end

"""
    squid_measure(sensor::SQUIDSensor, regime::GravityMagnetismRegime) -> SQUIDSensor

Measure the magnetic field in a given gravity-magnetism regime.
Returns updated sensor with measured flux and color phase.
"""
function squid_measure(sensor::SQUIDSensor, regime::GravityMagnetismRegime)::SQUIDSensor
    # Flux in units of Φ₀
    flux_quanta = regime.magnetism / FLUX_QUANTUM
    
    # Modular arithmetic: SQUID responds to fractional flux
    measured = mod(flux_quanta + sensor.flux_bias, 1.0)
    
    # Color phase from Josephson oscillation
    # Phase accumulates with magnetism
    phase = 2π * measured
    
    SQUIDSensor(
        sensor.flux_bias,
        sensor.josephson_freq,
        sensor.critical_current,
        sensor.seed,
        measured,
        phase
    )
end

function squid_color(sensor::SQUIDSensor)::NTuple{3, Float64}
    # Map SQUID state to color via phase
    phase = sensor.color_phase
    
    # HSL-like mapping
    h = mod(phase / (2π) * 360, 360)
    s = 0.5 + 0.4 * sensor.measured_flux
    l = 0.4 + 0.2 * cos(phase)
    
    # Simplified HSL to RGB
    c = (1 - abs(2 * l - 1)) * s
    x = c * (1 - abs(mod(h / 60, 2) - 1))
    m = l - c / 2
    
    r, g, b = if h < 60
        (c, x, 0.0)
    elseif h < 120
        (x, c, 0.0)
    elseif h < 180
        (0.0, c, x)
    elseif h < 240
        (0.0, x, c)
    elseif h < 300
        (x, 0.0, c)
    else
        (c, 0.0, x)
    end
    
    (r + m, g + m, b + m)
end

# ═══════════════════════════════════════════════════════════════════════════════
# S-Expression World: Colored World State
# ═══════════════════════════════════════════════════════════════════════════════

"""
    SexpWorld

A world represented as a colored S-expression tree.

The world contains:
- Regime: gravity-magnetism parameters
- Sensor: SQUID measurement state
- State: S-expression tree
- History: past states for temporal coherence
"""
mutable struct SexpWorld
    id::UInt64
    regime::GravityMagnetismRegime
    sensor::SQUIDSensor
    state::GaySexp
    history::Vector{GaySexp}
    
    # Parallelism metadata
    thread_id::Int
    computation_time::Float64
end

function SexpWorld(id::Int; 
                   gravity::Float64=9.81, 
                   magnetism::Float64=1e-6,
                   seed::UInt64=GAY_SEED)
    world_seed = seed ⊻ UInt64(id)
    regime = GravityMagnetismRegime(gravity, magnetism; seed=world_seed)
    sensor = SQUIDSensor(; flux_bias=mod(Float64(id) / 100, 1.0), seed=world_seed)
    
    # Initial state encodes world parameters
    state = gay_list(
        :world, id,
        :gravity, gravity,
        :magnetism, magnetism,
        :density, regime.density,
        :coherence, regime.coherence;
        seed=world_seed
    )
    
    SexpWorld(UInt64(id), regime, sensor, state, GaySexp[], 0, 0.0)
end

function world_color(w::SexpWorld)::NTuple{3, Float64}
    # Blend regime, SQUID, and state colors
    rc = regime_color(w.regime)
    sc = squid_color(w.sensor)
    stc = sexp_color(w.state)
    
    (
        (rc[1] + sc[1] + stc[1]) / 3,
        (rc[2] + sc[2] + stc[2]) / 3,
        (rc[3] + sc[3] + stc[3]) / 3
    )
end

function evolve!(w::SexpWorld)
    # Save history
    push!(w.history, w.state)
    if length(w.history) > 10
        popfirst!(w.history)
    end
    
    # Measure with SQUID
    w.sensor = squid_measure(w.sensor, w.regime)
    
    # Evolve state based on measurement
    new_seed = splitmix64(w.state.seed ⊻ UInt64(round(w.sensor.measured_flux * 1e15)))
    
    w.state = GayCons(
        GayAtom(:measured; seed=new_seed),
        GayCons(
            GayAtom(w.sensor.measured_flux; seed=new_seed),
            w.state;
            seed=new_seed
        );
        seed=new_seed
    )
    
    w
end

# ═══════════════════════════════════════════════════════════════════════════════
# Maximum Parallel Exploration
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ParallelExperiment

Configuration for maximally parallel world exploration.
"""
struct ParallelExperiment
    n_worlds::Int
    n_steps::Int
    gravity_samples::Vector{Float64}
    magnetism_samples::Vector{Float64}
    seed::UInt64
end

function ParallelExperiment(; 
    n_gravity::Int=10, 
    n_magnetism::Int=10, 
    n_steps::Int=100,
    seed::UInt64=GAY_SEED)
    
    # Log-spaced samples
    gravity_samples = 10 .^ range(log10(GRAVITY_RANGE[1]), log10(GRAVITY_RANGE[2]), length=n_gravity)
    magnetism_samples = 10 .^ range(log10(MAGNETISM_RANGE[1]), log10(MAGNETISM_RANGE[2]), length=n_magnetism)
    
    ParallelExperiment(
        n_gravity * n_magnetism,
        n_steps,
        gravity_samples,
        magnetism_samples,
        seed
    )
end

"""
    parallel_explore!(exp::ParallelExperiment) -> Vector{SexpWorld}

Explore all gravity-magnetism regimes in parallel.
Uses all available threads for maximum parallelism.
"""
function parallel_explore!(exp::ParallelExperiment)::Vector{SexpWorld}
    n_threads = nthreads()
    n_worlds = exp.n_worlds
    
    println("  Parallel exploration: $(n_worlds) worlds × $(exp.n_steps) steps")
    println("  Using $(n_threads) threads")
    
    # Create all worlds
    worlds = Vector{SexpWorld}(undef, n_worlds)
    
    idx = 1
    for (gi, g) in enumerate(exp.gravity_samples)
        for (mi, m) in enumerate(exp.magnetism_samples)
            world_seed = exp.seed ⊻ UInt64(gi * 1000 + mi)
            worlds[idx] = SexpWorld(idx; gravity=g, magnetism=m, seed=world_seed)
            idx += 1
        end
    end
    
    # Parallel evolution
    start_time = time()
    
    @threads for i in 1:n_worlds
        worlds[i].thread_id = threadid()
        local_start = time()
        
        for _ in 1:exp.n_steps
            evolve!(worlds[i])
        end
        
        worlds[i].computation_time = time() - local_start
    end
    
    total_time = time() - start_time
    
    println("  Total time: $(round(total_time, digits=3))s")
    println("  Throughput: $(round(n_worlds * exp.n_steps / total_time, digits=0)) world-steps/s")
    
    worlds
end

"""
    ExperimentResults

Results from parallel exploration with statistical analysis.
"""
struct ExperimentResults
    worlds::Vector{SexpWorld}
    
    # Phase space statistics
    hg_lm_worlds::Vector{SexpWorld}  # High gravity, low magnetism
    lg_hm_worlds::Vector{SexpWorld}  # Low gravity, high magnetism
    balanced_worlds::Vector{SexpWorld}  # Balanced regimes
    
    # Color statistics
    mean_color::NTuple{3, Float64}
    color_variance::Float64
    
    # SQUID statistics
    mean_flux::Float64
    flux_variance::Float64
    
    # Numerical robustness
    nan_count::Int
    inf_count::Int
    max_depth::Int
end

function analyze_results(worlds::Vector{SexpWorld})::ExperimentResults
    # Classify by regime
    hg_lm = filter(w -> w.regime.density > 0.7 && w.regime.coherence < 0.3, worlds)
    lg_hm = filter(w -> w.regime.density < 0.3 && w.regime.coherence > 0.7, worlds)
    balanced = filter(w -> 0.3 < w.regime.density < 0.7 && 0.3 < w.regime.coherence < 0.7, worlds)
    
    # Color statistics
    colors = [world_color(w) for w in worlds]
    mean_r = sum(c[1] for c in colors) / length(colors)
    mean_g = sum(c[2] for c in colors) / length(colors)
    mean_b = sum(c[3] for c in colors) / length(colors)
    mean_color = (mean_r, mean_g, mean_b)
    
    color_var = sum((c[1] - mean_r)^2 + (c[2] - mean_g)^2 + (c[3] - mean_b)^2 for c in colors) / length(colors)
    
    # SQUID statistics
    fluxes = [w.sensor.measured_flux for w in worlds]
    mean_flux = sum(fluxes) / length(fluxes)
    flux_var = sum((f - mean_flux)^2 for f in fluxes) / length(fluxes)
    
    # Numerical robustness
    nan_count = count(w -> any(isnan, world_color(w)), worlds)
    inf_count = count(w -> any(isinf, world_color(w)), worlds)
    max_depth = maximum(sexp_depth(w.state) for w in worlds)
    
    ExperimentResults(
        worlds,
        hg_lm, lg_hm, balanced,
        mean_color, color_var,
        mean_flux, flux_var,
        nan_count, inf_count, max_depth
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# Demo and Main Experiment
# ═══════════════════════════════════════════════════════════════════════════════

function world_squid_worlds()
    println()
    println("╔═══════════════════════════════════════════════════════════════════════════╗")
    println("║  SQUID S-Expression Worlds: Gravity-Magnetism Phase Space Exploration     ║")
    println("║  Maximum parallelism with colored S-expressions and SQUID sensing         ║")
    println("╚═══════════════════════════════════════════════════════════════════════════╝")
    println()
    
    # ─── Show Regimes ───
    println("─── Gravity-Magnetism Regimes ───")
    regimes = [
        ("High-G/Low-M (collapsed)", 1e6, 1e-15),
        ("Low-G/High-M (dispersed)", 1e-6, 1e3),
        ("Earth-like", 9.81, 5e-5),
        ("Jupiter-like", 24.79, 4.28e-4),
    ]
    
    for (name, g, m) in regimes
        r = GravityMagnetismRegime(g, m; seed=GAY_SEED)
        c = regime_color(r)
        color_emoji = r.color_shift > 0.3 ? "🔵" : r.color_shift < -0.3 ? "🔴" : "🟢"
        println("  $color_emoji $name")
        println("     g=$(round(g, sigdigits=3)) m/s², B=$(round(m, sigdigits=3)) T")
        println("     density=$(round(r.density, digits=3)), coherence=$(round(r.coherence, digits=3))")
        println("     RGB=$(round.(c, digits=2))")
    end
    println()
    
    # ─── SQUID Sensing ───
    println("─── SQUID Sensing ───")
    sensor = SQUIDSensor(; flux_bias=0.5, seed=GAY_SEED)
    println("  Flux quantum Φ₀ = $(round(FLUX_QUANTUM, sigdigits=4)) Wb")
    println("  Josephson freq = $(round(sensor.josephson_freq, sigdigits=3)) Hz")
    println("  Critical current = $(round(sensor.critical_current * 1e6, sigdigits=3)) μA")
    
    for (name, g, m) in regimes[1:2]
        r = GravityMagnetismRegime(g, m; seed=GAY_SEED)
        measured = squid_measure(sensor, r)
        c = squid_color(measured)
        println("  $name: flux=$(round(measured.measured_flux, digits=4)) Φ₀, phase=$(round(measured.color_phase, digits=2)) rad")
        println("     SQUID RGB=$(round.(c, digits=2))")
    end
    println()
    
    # ─── S-Expression Worlds ───
    println("─── S-Expression Worlds ───")
    world = SexpWorld(1; gravity=9.81, magnetism=5e-5, seed=GAY_SEED)
    println("  Initial state: $(sexp_to_string(world.state)[1:min(60, end)])...")
    println("  World color: $(round.(world_color(world), digits=2))")
    
    for i in 1:3
        evolve!(world)
    end
    println("  After 3 steps: depth=$(sexp_depth(world.state)), history=$(length(world.history))")
    println()
    
    # ─── Parallel Experiment ───
    println("─── Parallel Experiment ───")
    exp = ParallelExperiment(; n_gravity=5, n_magnetism=5, n_steps=50, seed=GAY_SEED)
    worlds = parallel_explore!(exp)
    results = analyze_results(worlds)
    
    println()
    println("─── Results ───")
    println("  Total worlds: $(length(results.worlds))")
    println("  High-G/Low-M: $(length(results.hg_lm_worlds)) worlds")
    println("  Low-G/High-M: $(length(results.lg_hm_worlds)) worlds")
    println("  Balanced: $(length(results.balanced_worlds)) worlds")
    println()
    println("  Mean color: RGB$(round.(results.mean_color, digits=3))")
    println("  Color variance: $(round(results.color_variance, digits=4))")
    println("  Mean flux: $(round(results.mean_flux, digits=4)) Φ₀")
    println("  Flux variance: $(round(results.flux_variance, digits=4))")
    println()
    println("  Numerical robustness:")
    println("    NaN count: $(results.nan_count)")
    println("    Inf count: $(results.inf_count)")
    println("    Max sexp depth: $(results.max_depth)")
    
    return results
end

function run_max_parallel_experiment(; 
    n_gravity::Int=20, 
    n_magnetism::Int=20, 
    n_steps::Int=200)
    
    println()
    println("╔═══════════════════════════════════════════════════════════════════════════╗")
    println("║  MAXIMUM PARALLEL EXPERIMENT: $(n_gravity)×$(n_magnetism) regimes × $(n_steps) steps      ║")
    println("║  Testing numerical robustness across extreme gravity-magnetism regimes    ║")
    println("╚═══════════════════════════════════════════════════════════════════════════╝")
    println()
    
    exp = ParallelExperiment(; n_gravity=n_gravity, n_magnetism=n_magnetism, n_steps=n_steps, seed=GAY_SEED)
    
    println("─── Configuration ───")
    println("  Worlds: $(exp.n_worlds)")
    println("  Steps per world: $(exp.n_steps)")
    println("  Total operations: $(exp.n_worlds * exp.n_steps)")
    println("  Threads: $(nthreads())")
    println("  Gravity range: $(GRAVITY_RANGE)")
    println("  Magnetism range: $(MAGNETISM_RANGE)")
    println()
    
    worlds = parallel_explore!(exp)
    results = analyze_results(worlds)
    
    println()
    println("═══════════════════════════════════════════════════════════════════════════")
    println("RESULTS SUMMARY")
    println("═══════════════════════════════════════════════════════════════════════════")
    println()
    
    # Regime breakdown
    println("─── Regime Classification ───")
    println("  High-G/Low-M (🔴 collapsed): $(length(results.hg_lm_worlds)) worlds")
    println("  Low-G/High-M (🔵 dispersed): $(length(results.lg_hm_worlds)) worlds")
    println("  Balanced (🟢):               $(length(results.balanced_worlds)) worlds")
    println()
    
    # Color analysis by regime
    if !isempty(results.hg_lm_worlds)
        hg_colors = [world_color(w) for w in results.hg_lm_worlds]
        hg_mean = (sum(c[1] for c in hg_colors), sum(c[2] for c in hg_colors), sum(c[3] for c in hg_colors)) ./ length(hg_colors)
        println("  🔴 High-G/Low-M mean color: RGB$(round.(hg_mean, digits=3))")
    end
    
    if !isempty(results.lg_hm_worlds)
        lg_colors = [world_color(w) for w in results.lg_hm_worlds]
        lg_mean = (sum(c[1] for c in lg_colors), sum(c[2] for c in lg_colors), sum(c[3] for c in lg_colors)) ./ length(lg_colors)
        println("  🔵 Low-G/High-M mean color: RGB$(round.(lg_mean, digits=3))")
    end
    println()
    
    # Thread distribution
    println("─── Thread Distribution ───")
    thread_counts = Dict{Int, Int}()
    for w in worlds
        thread_counts[w.thread_id] = get(thread_counts, w.thread_id, 0) + 1
    end
    for tid in sort(collect(keys(thread_counts)))
        println("  Thread $tid: $(thread_counts[tid]) worlds")
    end
    println()
    
    # Timing
    times = [w.computation_time for w in worlds]
    println("─── Timing ───")
    println("  Min world time: $(round(minimum(times), digits=4))s")
    println("  Max world time: $(round(maximum(times), digits=4))s")
    println("  Mean world time: $(round(sum(times)/length(times), digits=4))s")
    println("  Load imbalance: $(round(maximum(times)/minimum(times), digits=2))×")
    println()
    
    # Numerical robustness
    println("─── Numerical Robustness ───")
    println("  ✓ NaN count: $(results.nan_count)")
    println("  ✓ Inf count: $(results.inf_count)")
    println("  Max S-expression depth: $(results.max_depth)")
    println("  Color variance: $(round(results.color_variance, digits=4))")
    println()
    
    # SQUID coherence
    println("─── SQUID Measurements ───")
    println("  Mean flux: $(round(results.mean_flux, digits=4)) Φ₀")
    println("  Flux variance: $(round(results.flux_variance, digits=6))")
    
    # Find extreme worlds
    extreme_g_world = worlds[argmax([w.regime.gravity for w in worlds])]
    extreme_m_world = worlds[argmax([w.regime.magnetism for w in worlds])]
    
    println()
    println("─── Extreme Worlds ───")
    println("  Highest gravity: g=$(extreme_g_world.regime.gravity) m/s²")
    println("    Color: $(round.(world_color(extreme_g_world), digits=3))")
    println("  Highest magnetism: B=$(extreme_m_world.regime.magnetism) T")
    println("    Color: $(round.(world_color(extreme_m_world), digits=3))")
    
    return results
end

end # module SQUIDSexpWorlds
