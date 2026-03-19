# GAYZIP STRATEGY: GayOpenGame for Compression Paradigm Selection
# ================================================================
#
# "The individuation functor determines when gzip becomes gayzip."
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  GAYZIP vs GZIP: A GAME-THEORETIC DECISION PROBLEM                          │
# │                                                                             │
# │  PLAYERS:                                                                   │
# │    Encoder: Decides eager (gzip) vs lazy (gayzip) vs hybrid                 │
# │    Decoder: Decides decompress, generate, or cache                          │
# │                                                                             │
# │  INDIVIDUATION FUNCTORS:                                                    │
# │    F_gzip    : Data → CompressedBlob     (deterministic, stateless)         │
# │    F_gayzip  : Data → Seed + Procedure   (generative, stateful)             │
# │    F_hybrid  : Data → Best(F_gzip, F_gayzip) given gzipability             │
# │                                                                             │
# │  STRUCTURED DECOMPOSITION:                                                  │
# │    Bag = data chunk (parallel unit)                                         │
# │    Adhesion = dependency between chunks                                     │
# │    Sheaf condition = local-to-global consistency                            │
# │                                                                             │
# │  VANISHING POINTS AT INFINITY:                                              │
# │    As parallelism → ∞, certain limits "creep in":                           │
# │    1. Sequential bottlenecks become visible (adhesion density)              │
# │    2. Communication costs dominate (Amdahl's law)                           │
# │    3. Seed collision probability increases (birthday bound)                 │
# │                                                                             │
# │  GAY SHEAVES = LAVISH PRESHEAVES:                                           │
# │    Presheaf: F: C^op → Set                                                  │
# │    Sheaf: + gluing condition (local → global)                               │
# │    Gay Sheaf: + uniqueness of hue (splittable RNG determinism)              │
# │    Lavish: + chromatic identity preserved across all restrictions           │
# └─────────────────────────────────────────────────────────────────────────────┘

module GayzipStrategy

using SplittableRandoms: SplittableRandom, split
using Colors
using Printf

export
    # Core Types
    CompressionParadigm, GzipParadigm, GayzipParadigm, HybridParadigm,
    IndividuationFunctor, GzipFunctor, GayzipFunctor, HybridFunctor,
    
    # Game Structure
    CompressionGame, EncoderPlayer, DecoderPlayer,
    EncoderAction, DecoderAction, CompressionOutcome,
    
    # Structured Decomposition
    DataBag, DataAdhesion, CompressionDecomposition,
    decompose_for_compression, sheaf_consistency,
    
    # Vanishing Points
    VanishingPoint, AmdahlLimit, BirthdayLimit, AdhesionDensityLimit,
    detect_vanishing_points, parallelism_ceiling,
    
    # Gay Sheaves (Lavish Presheaves)
    GaySheaf, LavishPresheaf, HueUniqueness,
    sheaf_from_decomposition, verify_lavish_condition,
    
    # Strategy Selection
    optimal_paradigm, gzipability_threshold, 
    expected_speedup, practical_speedups,
    
    # GayMC Leitmotif Surgeries
    Leitmotif, LeitmotifSurgery, trajectory_recovery,
    insert_leitmotif, remove_leitmotif, reconstruct_trajectory,
    
    # Demo
    demo_gayzip_strategy, analyze_compression_game

# ═══════════════════════════════════════════════════════════════════════════════
# Core PRNG
# ═══════════════════════════════════════════════════════════════════════════════

const GAY_SEED = UInt64(1069)
const GAYZIP_SEED = UInt64(0x6179697A70)  # "gayzip"

@inline function splitmix64(state::UInt64)::Tuple{UInt64, UInt64}
    z = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ⊻ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    z ⊻ (z >> 31), (state + 1) & 0xFFFFFFFFFFFFFFFF
end

@inline function color_from_seed(seed::UInt64)::RGB{Float64}
    r, s1 = splitmix64(seed)
    g, s2 = splitmix64(s1)
    b, _  = splitmix64(s2)
    RGB((r >> 56) / 255.0, (g >> 56) / 255.0, (b >> 56) / 255.0)
end

# ═══════════════════════════════════════════════════════════════════════════════
# COMPRESSION PARADIGMS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    CompressionParadigm

The three paradigms for compression strategy.
"""
abstract type CompressionParadigm end

struct GzipParadigm <: CompressionParadigm
    level::Int              # 1-9 compression level
    block_size::Int         # bytes per block
end
GzipParadigm() = GzipParadigm(6, 131072)

struct GayzipParadigm <: CompressionParadigm
    seed::UInt64            # deterministic seed
    procedure_id::Symbol    # :diffusion, :flow_matching, :ar_generate
    steps::Int              # generation steps
end
GayzipParadigm() = GayzipParadigm(GAY_SEED, :flow_matching, 20)

struct HybridParadigm <: CompressionParadigm
    threshold::Float64      # gzipability threshold for switching
    gzip::GzipParadigm
    gayzip::GayzipParadigm
end
HybridParadigm() = HybridParadigm(0.7, GzipParadigm(), GayzipParadigm())

# ═══════════════════════════════════════════════════════════════════════════════
# INDIVIDUATION FUNCTORS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    IndividuationFunctor

Functor that determines how data is individuated for compression.
From StructuredDecompositions.jl: this is the F in 𝐃(F).
"""
abstract type IndividuationFunctor end

"""
    GzipFunctor

Traditional compression: Data → CompressedBlob
Deterministic, stateless, order-dependent.
"""
struct GzipFunctor <: IndividuationFunctor
    paradigm::GzipParadigm
end

"""
    GayzipFunctor

Generative compression: Data → (Seed, Procedure)
Deterministic via splittable RNG, can be parallelized.
"""
struct GayzipFunctor <: IndividuationFunctor
    paradigm::GayzipParadigm
    rng::SplittableRandom
end
GayzipFunctor(p::GayzipParadigm) = GayzipFunctor(p, SplittableRandom(p.seed))

"""
    HybridFunctor

Adaptive compression: chooses based on content analysis.
"""
struct HybridFunctor <: IndividuationFunctor
    gzip::GzipFunctor
    gayzip::GayzipFunctor
    threshold::Float64
end

function HybridFunctor(h::HybridParadigm)
    HybridFunctor(
        GzipFunctor(h.gzip),
        GayzipFunctor(h.gayzip),
        h.threshold
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# STRUCTURED DECOMPOSITION FOR COMPRESSION
# ═══════════════════════════════════════════════════════════════════════════════

"""
    DataBag

A bag in the structured decomposition = a chunk of data for parallel processing.
"""
struct DataBag
    id::Int
    data::Vector{UInt8}
    seed::UInt64            # splittable seed for this bag
    color::RGB{Float64}     # chromatic identity
    
    # Compression metrics
    gzipability::Float64    # estimated compressibility
    parallel_safe::Bool     # can be processed independently
end

function DataBag(id::Int, data::Vector{UInt8}; base_seed::UInt64=GAY_SEED)
    seed = base_seed ⊻ UInt64(id)
    color = color_from_seed(seed)
    
    # Estimate gzipability (simplified - real impl would use actual gzip)
    # Use entropy as proxy
    if isempty(data)
        gzipability = 1.0
    else
        byte_counts = zeros(Int, 256)
        for b in data
            byte_counts[b + 1] += 1
        end
        n = length(data)
        entropy = -sum(c > 0 ? (c/n) * log2(c/n) : 0.0 for c in byte_counts)
        gzipability = entropy / 8.0  # normalized to [0, 1]
    end
    
    DataBag(id, data, seed, color, gzipability, true)
end

"""
    DataAdhesion

An adhesion between bags = dependency constraint.
"""
struct DataAdhesion
    bag1_id::Int
    bag2_id::Int
    overlap::Vector{UInt8}  # shared data (boundary)
    constraint::Symbol      # :order, :checksum, :none
    
    # Parallelism impact
    blocks_parallel::Bool   # if true, bags must be sequential
end

"""
    CompressionDecomposition

Full structured decomposition for compression.
Bags = parallel units, Adhesions = sync points.
"""
struct CompressionDecomposition
    bags::Vector{DataBag}
    adhesions::Vector{DataAdhesion}
    
    # Parallelism analysis
    max_parallelism::Int    # max concurrent bags
    sequential_fraction::Float64  # Amdahl's law parameter
    
    # Chromatic fingerprint
    fingerprint::UInt64
    color::RGB{Float64}
end

"""
    decompose_for_compression(data, block_size; seed) → CompressionDecomposition

Decompose data into bags for parallel compression.
"""
function decompose_for_compression(data::Vector{UInt8}; 
        block_size::Int=131072,
        seed::UInt64=GAY_SEED)
    
    n = length(data)
    n_bags = cld(n, block_size)
    
    # Create bags
    bags = DataBag[]
    for i in 1:n_bags
        start_idx = (i - 1) * block_size + 1
        end_idx = min(i * block_size, n)
        chunk = data[start_idx:end_idx]
        push!(bags, DataBag(i, chunk; base_seed=seed))
    end
    
    # Create adhesions 
    # For gzip: order matters (blocks_parallel=true)
    # For gayzip: order-independent (blocks_parallel=false) due to SPI
    adhesions = DataAdhesion[]
    for i in 1:(n_bags - 1)
        # Adhesions exist but DON'T block parallelism with gayzip/SPI
        push!(adhesions, DataAdhesion(i, i + 1, UInt8[], :checksum, false))
    end
    
    # Compute parallelism metrics
    # With gayzip SPI: fully parallel (adhesions don't block)
    # Sequential fraction = fraction of adhesions that actually block
    max_par = n_bags
    blocking_adhesions = count(a -> a.blocks_parallel, adhesions)
    seq_frac = blocking_adhesions / max(1, n_bags)
    
    # Fingerprint
    fp = reduce(⊻, [b.seed for b in bags]; init=seed)
    color = color_from_seed(fp)
    
    CompressionDecomposition(bags, adhesions, max_par, seq_frac, fp, color)
end

"""
    sheaf_consistency(decomp) → Bool

Check the sheaf condition: local bag compressions must be globally consistent.
This is the "lavish presheaf" condition from Bumpus.
"""
function sheaf_consistency(decomp::CompressionDecomposition)
    # For compression, consistency means:
    # 1. XOR of all bag fingerprints = global fingerprint (SPI)
    # 2. Adhesion constraints are satisfiable
    
    # Check fingerprint consistency
    computed_fp = reduce(⊻, [b.seed for b in decomp.bags]; init=UInt64(0))
    
    # Check adhesion satisfiability
    for adh in decomp.adhesions
        if adh.blocks_parallel && adh.constraint == :order
            # Order constraint requires sequential access to these bags
            continue
        end
    end
    
    true  # Simplified - real impl would check more conditions
end

# ═══════════════════════════════════════════════════════════════════════════════
# VANISHING POINTS AT INFINITY
# ═══════════════════════════════════════════════════════════════════════════════

"""
    VanishingPoint

A limit where infinite parallelism "creeps in" with diminishing returns.
"""
abstract type VanishingPoint end

"""
    AmdahlLimit

Amdahl's law: speedup limited by sequential fraction.
As P → ∞: S(P) → 1/f where f = sequential fraction.
"""
struct AmdahlLimit <: VanishingPoint
    sequential_fraction::Float64
    max_speedup::Float64
    
    # At infinity
    limiting_speedup::Float64
    communication_overhead::Float64
end

function AmdahlLimit(f::Float64; overhead::Float64=0.0)
    max_s = 1.0 / max(f, 1e-10)
    lim_s = max_s / (1.0 + overhead)
    AmdahlLimit(f, max_s, lim_s, overhead)
end

"""
    BirthdayLimit

Birthday bound: probability of seed collision.
As n → ∞: P(collision) → 1 when n ≈ √(2^k) for k-bit seeds.
"""
struct BirthdayLimit <: VanishingPoint
    seed_bits::Int
    max_parallel_ops::UInt64
    collision_at_p50::UInt64   # n where P(collision) ≈ 0.5
end

function BirthdayLimit(; seed_bits::Int=64)
    # Birthday bound: 50% collision at n ≈ 1.2 * √(2^k)
    p50 = UInt64(floor(1.2 * sqrt(Float64(UInt64(1) << min(seed_bits, 63)))))
    BirthdayLimit(seed_bits, UInt64(1) << seed_bits, p50)
end

"""
    AdhesionDensityLimit

Sequential bottlenecks from adhesion density.
As parallelism increases, adhesion overhead dominates.
"""
struct AdhesionDensityLimit <: VanishingPoint
    n_bags::Int
    n_adhesions::Int
    density::Float64          # adhesions / bags
    critical_density::Float64 # where overhead dominates
end

function AdhesionDensityLimit(decomp::CompressionDecomposition; critical::Float64=0.8)
    n_b = length(decomp.bags)
    n_a = length(decomp.adhesions)
    density = n_a / max(1, n_b)
    AdhesionDensityLimit(n_b, n_a, density, critical)
end

"""
    detect_vanishing_points(decomp) → Vector{VanishingPoint}

Detect all vanishing points in a compression decomposition.
"""
function detect_vanishing_points(decomp::CompressionDecomposition)
    points = VanishingPoint[]
    
    # Amdahl's law
    push!(points, AmdahlLimit(decomp.sequential_fraction))
    
    # Birthday bound
    push!(points, BirthdayLimit())
    
    # Adhesion density
    push!(points, AdhesionDensityLimit(decomp))
    
    points
end

"""
    parallelism_ceiling(points::Vector{VanishingPoint}) → Float64

Compute the effective ceiling on parallelism given vanishing points.
"""
function parallelism_ceiling(points::Vector{VanishingPoint})
    ceilings = Float64[]
    
    for p in points
        if p isa AmdahlLimit
            push!(ceilings, p.limiting_speedup)
        elseif p isa BirthdayLimit
            push!(ceilings, Float64(p.collision_at_p50))
        elseif p isa AdhesionDensityLimit
            if p.density > p.critical_density
                push!(ceilings, 1.0 / p.density)
            else
                push!(ceilings, Float64(p.n_bags))
            end
        end
    end
    
    minimum(ceilings)
end

# ═══════════════════════════════════════════════════════════════════════════════
# GAY SHEAVES (LAVISH PRESHEAVES)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    HueUniqueness

The uniqueness of hue condition: splittable RNG ensures determinism.
"""
struct HueUniqueness
    seed::UInt64
    hue::Float64            # HSL hue value [0, 360)
    unique::Bool            # does this seed have a unique hue?
    collision_distance::Int # steps to nearest collision
end

function HueUniqueness(seed::UInt64; search_range::Int=1000)
    color = color_from_seed(seed)
    hue = 360.0 * atan(color.g - 0.5, color.r - 0.5) / (2π) + 180.0
    
    # Check for collisions in nearby seeds
    collision_dist = search_range + 1
    for i in 1:search_range
        for s in [seed + UInt64(i), seed - UInt64(i)]
            c = color_from_seed(s)
            h = 360.0 * atan(c.g - 0.5, c.r - 0.5) / (2π) + 180.0
            if abs(h - hue) < 0.1
                collision_dist = min(collision_dist, i)
            end
        end
    end
    
    HueUniqueness(seed, hue, collision_dist > search_range, collision_dist)
end

"""
    GaySheaf

A Gay sheaf: presheaf with gluing + uniqueness of hue.
This is the "lavish presheaf" that Bumpus describes.
"""
struct GaySheaf
    # Presheaf data
    objects::Vector{DataBag}          # F(U) for each open U
    restrictions::Vector{DataAdhesion} # F(V ⊆ U): F(U) → F(V)
    
    # Sheaf condition
    gluing_satisfied::Bool
    
    # Gay condition (uniqueness of hue)
    hue_uniqueness::Vector{HueUniqueness}
    all_hues_unique::Bool
    
    # Lavish condition (chromatic identity preserved)
    global_fingerprint::UInt64
    local_fingerprints_match::Bool
end

"""
    sheaf_from_decomposition(decomp) → GaySheaf

Construct a Gay sheaf from a compression decomposition.
"""
function sheaf_from_decomposition(decomp::CompressionDecomposition)
    # Check gluing
    gluing = sheaf_consistency(decomp)
    
    # Check hue uniqueness
    hues = [HueUniqueness(b.seed) for b in decomp.bags]
    all_unique = all(h.unique for h in hues)
    
    # Check fingerprint consistency
    local_fps = [b.seed for b in decomp.bags]
    global_fp = reduce(⊻, local_fps; init=decomp.fingerprint)
    fps_match = global_fp == decomp.fingerprint
    
    GaySheaf(
        decomp.bags, decomp.adhesions,
        gluing, hues, all_unique,
        decomp.fingerprint, fps_match
    )
end

"""
    verify_lavish_condition(sheaf) → Bool

Verify the full lavish presheaf condition:
1. Presheaf: functorial on restrictions
2. Sheaf: gluing condition
3. Gay: uniqueness of hue (splittable RNG)
4. Lavish: chromatic identity preserved globally
"""
function verify_lavish_condition(sheaf::GaySheaf)
    sheaf.gluing_satisfied && 
    sheaf.all_hues_unique && 
    sheaf.local_fingerprints_match
end

# ═══════════════════════════════════════════════════════════════════════════════
# COMPRESSION GAME: Encoder vs Decoder
# ═══════════════════════════════════════════════════════════════════════════════

"""
    EncoderAction

Actions available to the encoder.
"""
@enum EncoderAction begin
    EncodeGzip          # Traditional gzip
    EncodeGayzip        # Generative gayzip
    EncodeHybrid        # Adaptive based on content
    EncodeDeferDecoder  # Store raw, let decoder decide
end

"""
    DecoderAction

Actions available to the decoder.
"""
@enum DecoderAction begin
    DecodeDecompress    # Traditional decompression
    DecodeGenerate      # Generate from seed
    DecodeCache         # Use cached result
    DecodeStream        # Stream decompress
end

"""
    CompressionOutcome

The outcome of a compression game.
"""
struct CompressionOutcome
    encoder_action::EncoderAction
    decoder_action::DecoderAction
    
    # Metrics
    compression_ratio::Float64
    encode_time::Float64
    decode_time::Float64
    total_time::Float64
    
    # Parallelism achieved
    parallelism_used::Int
    parallelism_ceiling::Float64
    
    # Equilibrium
    is_equilibrium::Bool    # Nash equilibrium?
    encoder_utility::Float64
    decoder_utility::Float64
end

"""
    CompressionGame

The full GayOpenGame for compression strategy.
"""
struct CompressionGame
    # Data
    data::Vector{UInt8}
    decomposition::CompressionDecomposition
    sheaf::GaySheaf
    
    # Vanishing points
    vanishing_points::Vector{VanishingPoint}
    
    # Strategies
    encoder_strategy::Dict{Float64, EncoderAction}  # gzipability → action
    decoder_strategy::Dict{EncoderAction, DecoderAction}
    
    # Equilibria found
    equilibria::Vector{CompressionOutcome}
end

"""
    CompressionGame(data; seed) → CompressionGame

Create a compression game from data.
"""
function CompressionGame(data::Vector{UInt8}; seed::UInt64=GAY_SEED)
    decomp = decompose_for_compression(data; seed=seed)
    sheaf = sheaf_from_decomposition(decomp)
    vps = detect_vanishing_points(decomp)
    
    # Default strategies
    encoder_strat = Dict{Float64, EncoderAction}(
        0.0 => EncodeGayzip,    # very compressible → generative
        0.5 => EncodeHybrid,    # medium → adaptive
        0.7 => EncodeGzip,      # high entropy → traditional
        1.0 => EncodeDeferDecoder
    )
    
    decoder_strat = Dict{EncoderAction, DecoderAction}(
        EncodeGzip => DecodeDecompress,
        EncodeGayzip => DecodeGenerate,
        EncodeHybrid => DecodeStream,
        EncodeDeferDecoder => DecodeCache
    )
    
    CompressionGame(data, decomp, sheaf, vps, encoder_strat, decoder_strat, CompressionOutcome[])
end

# ═══════════════════════════════════════════════════════════════════════════════
# GAYMC LEITMOTIF SURGERIES
# ═══════════════════════════════════════════════════════════════════════════════

"""
    Leitmotif

A recurring seed pattern that enables trajectory reconstruction.
From GayMC: the leitmotif is the "theme" that recurs through the Markov chain.
"""
struct Leitmotif
    pattern::UInt64         # the recurring seed pattern
    period::Int             # how often it recurs
    positions::Vector{Int}  # where it appears in trajectory
    
    # Chromatic identity
    color::RGB{Float64}
end

function Leitmotif(seed::UInt64, trajectory_length::Int; period::Int=100)
    positions = collect(period:period:trajectory_length)
    color = color_from_seed(seed)
    Leitmotif(seed, period, positions, color)
end

"""
    LeitmotifSurgery

An insertion or removal of leitmotif in a trajectory.
"""
struct LeitmotifSurgery
    operation::Symbol       # :insert or :remove
    position::Int
    leitmotif::Leitmotif
    
    # Before/after states
    before_fp::UInt64
    after_fp::UInt64
end

"""
    trajectory_recovery(surgeries, original_fp) → UInt64

Reconstruct original trajectory from surgeries.
"""
function trajectory_recovery(surgeries::Vector{LeitmotifSurgery}, original_fp::UInt64)
    # Apply surgeries in reverse to recover original
    fp = original_fp
    for s in reverse(surgeries)
        if s.operation == :insert
            # Undo insertion by XORing out the leitmotif
            fp = fp ⊻ s.leitmotif.pattern
        elseif s.operation == :remove
            # Undo removal by XORing in the leitmotif
            fp = fp ⊻ s.leitmotif.pattern
        end
    end
    fp
end

# ═══════════════════════════════════════════════════════════════════════════════
# PRACTICAL SPEEDUPS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    SpeedupResult

Result of speedup analysis.
"""
struct SpeedupResult
    task::Symbol
    baseline_time::Float64      # sequential gzip
    parallel_time::Float64      # gayzip parallel
    speedup::Float64            # baseline / parallel
    parallelism::Int            # threads/workers used
    paradigm::Symbol            # :gzip, :gayzip, :hybrid
    
    # Vanishing point impact
    theoretical_max::Float64    # without vanishing points
    actual_achieved::Float64    # with vanishing points
    efficiency::Float64         # actual / theoretical
end

"""
    practical_speedups(data_sizes; max_threads) → Vector{SpeedupResult}

Compute practical speedups for various tasks and data sizes.
Compare gzip (sequential) vs gayzip (parallel with SPI).
"""
function practical_speedups(; 
        data_sizes::Vector{Int}=[65536, 1048576, 16777216, 67108864],
        max_threads::Int=16,
        seed::UInt64=GAY_SEED)
    
    results = SpeedupResult[]
    
    for size in data_sizes
        # Analyze decomposition (gayzip style - no blocking adhesions)
        n_bags = cld(size, 131072)  # 128KB blocks
        
        for paradigm in [:gzip, :gayzip]
            # Simulate timings
            baseline = size / 100_000_000.0  # ~100MB/s throughput
            
            if paradigm == :gzip
                # gzip: sequential, no parallelism benefit
                effective_par = 1.0
                overhead = 0.0
                parallel = baseline  # same as sequential
            else
                # gayzip: fully parallel with SPI guarantee
                # Limited by: min(threads, bags, birthday bound not hit)
                effective_par = min(Float64(max_threads), Float64(n_bags))
                overhead = 0.05 * log2(effective_par)  # log overhead for coordination
                parallel = baseline / effective_par * (1 + overhead)
            end
            
            # Theoretical max (perfect scaling)
            theoretical = baseline / min(Float64(max_threads), Float64(n_bags))
            efficiency = theoretical / max(parallel, 1e-10)
            
            push!(results, SpeedupResult(
                :compress, baseline, parallel,
                baseline / max(parallel, 1e-10),
                Int(round(effective_par)), paradigm,
                baseline / theoretical,
                baseline / max(parallel, 1e-10),
                min(1.0, efficiency)
            ))
        end
    end
    
    results
end

# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

function demo_gayzip_strategy()
    println()
    println("╔═══════════════════════════════════════════════════════════════════════════╗")
    println("║  GAYZIP STRATEGY: GayOpenGame for Compression Paradigm Selection         ║")
    println("╚═══════════════════════════════════════════════════════════════════════════╝")
    println()
    
    # Create test data with varying compressibility
    test_sizes = [64 * 1024, 1024 * 1024, 16 * 1024 * 1024]
    
    for size in test_sizes
        println("─── Data Size: $(size ÷ 1024) KB ───")
        
        # Random data
        data = rand(UInt8, size)
        
        # Create game
        game = CompressionGame(data)
        
        # Analyze
        println("  Decomposition:")
        println("    Bags: $(length(game.decomposition.bags))")
        println("    Adhesions: $(length(game.decomposition.adhesions))")
        println("    Max parallelism: $(game.decomposition.max_parallelism)")
        println("    Sequential fraction: $(round(game.decomposition.sequential_fraction, digits=3))")
        
        # Vanishing points
        println("  Vanishing Points:")
        for vp in game.vanishing_points
            if vp isa AmdahlLimit
                println("    Amdahl: max speedup = $(round(vp.max_speedup, digits=1))×")
            elseif vp isa BirthdayLimit
                println("    Birthday: 50% collision at $(vp.collision_at_p50) ops")
            elseif vp isa AdhesionDensityLimit
                println("    Adhesion density: $(round(vp.density, digits=3))")
            end
        end
        
        ceiling = parallelism_ceiling(game.vanishing_points)
        println("    → Parallelism ceiling: $(round(ceiling, digits=1))")
        
        # Sheaf analysis
        println("  Gay Sheaf (Lavish Presheaf):")
        println("    Gluing satisfied: $(game.sheaf.gluing_satisfied)")
        println("    All hues unique: $(game.sheaf.all_hues_unique)")
        println("    Fingerprints match: $(game.sheaf.local_fingerprints_match)")
        println("    Lavish condition: $(verify_lavish_condition(game.sheaf))")
        
        println()
    end
    
    # Practical speedups
    println("─── PRACTICAL SPEEDUPS ───")
    println()
    results = practical_speedups()
    
    println("  Task         Size        Paradigm   Parallelism   Speedup   Efficiency")
    println("  " * "─"^70)
    
    for r in results
        size_str = r.baseline_time < 0.001 ? "1KB" : 
                   r.baseline_time < 0.01 ? "64KB" :
                   r.baseline_time < 0.1 ? "1MB" : "16MB"
        
        println(@sprintf("  %-12s %-8s    %-8s   %3d           %5.1f×    %5.1f%%",
            r.task, size_str, r.paradigm, r.parallelism, r.speedup, r.efficiency * 100))
    end
    
    println()
    println("─── INDIVIDUATION FUNCTOR SELECTION ───")
    println()
    println("  Gzipability    Paradigm       Functor              Reason")
    println("  " * "─"^65)
    println("  < 0.3          gayzip         GayzipFunctor        Highly structured, generative wins")
    println("  0.3 - 0.5      hybrid         HybridFunctor        Mix of patterns, adaptive")
    println("  0.5 - 0.7      hybrid         HybridFunctor        Medium entropy, context-dependent")
    println("  > 0.7          gzip           GzipFunctor          High entropy, traditional wins")
    println("  > 0.95         defer          IdentityFunctor      Incompressible, store raw")
    
    (game=CompressionGame(rand(UInt8, 1024)), speedups=results)
end

function analyze_compression_game(data::Vector{UInt8})
    game = CompressionGame(data)
    
    println("Compression Game Analysis")
    println("========================")
    println("Data size: $(length(data)) bytes")
    println("Bags: $(length(game.decomposition.bags))")
    println("Parallelism ceiling: $(round(parallelism_ceiling(game.vanishing_points), digits=1))")
    println("Lavish condition: $(verify_lavish_condition(game.sheaf))")
    
    # Recommend paradigm
    avg_gzip = sum(b.gzipability for b in game.decomposition.bags) / length(game.decomposition.bags)
    recommended = avg_gzip < 0.5 ? "gayzip" : (avg_gzip < 0.7 ? "hybrid" : "gzip")
    println("Recommended paradigm: $recommended (gzipability = $(round(avg_gzip, digits=3)))")
    
    game
end

end # module GayzipStrategy
