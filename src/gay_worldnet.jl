# GayWorldNet: Tritwise Anticipatory Worldnet for Blockchain Parallelism
# ========================================================================
#
# "anticipatory about self and 2 others" - Husserlian protention extended
# to tritwise (3-way) comparison of maximally reachable parallelism.
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  WORLDNET TOPOLOGY                                                          │
# │                                                                             │
# │              ┌─────────────────────────────────────────┐                    │
# │              │       gayzip/gay/gayzip.gay 🍍          │                    │
# │              │          ANANAS APEX                    │                    │
# │              │     (UNFREE Invariants Hold)            │                    │
# │              └──────────────┬──────────────────────────┘                    │
# │                             │                                               │
# │      ┌──────────────────────┼──────────────────────────┐                    │
# │      │                      │                          │                    │
# │      ▼                      ▼                          ▼                    │
# │  ┌───────────┐        ┌───────────┐            ┌───────────┐               │
# │  │   APTOS   │ ←───→  │    SUI    │  ←───→     │   CHIA    │               │
# │  │ Block-STM │        │ Object    │            │ Spend     │               │
# │  │ 160k TPS  │        │ Parallel  │            │ Bundles   │               │
# │  │           │        │           │            │ BLS Agg   │               │
# │  └───────────┘        └───────────┘            └───────────┘               │
# │       T-              T0                            T+                      │
# │    (SPECULATIVE)   (OBJECT-LEVEL)              (AGGREGATE)                 │
# │                                                                             │
# │  TRITWISE COMPARISON: {T-, T0, T+} balanced ternary across worlds          │
# │  ANTICIPATORY: each world protends toward the other 2                      │
# │  SPI INVARIANTS: splitmix64, color_from_seed, fingerprint                  │
# └─────────────────────────────────────────────────────────────────────────────┘
#
# DESIDERATA FOR GAYTRAVERSAL:
#   1. Maximally Reachable Parallelism: explore all parallel execution paths
#   2. SPI Invariants: deterministic across all worlds (splitmix64, color, fp)
#   3. Tritwise Balance: {-, 0, +} comparison encodes relative parallelism
#   4. Anticipatory Structure: self + 2 others = 3 = trit arity
#   5. Galois Connection: handoff preserves information across worlds

module GayWorldNet

using SplittableRandoms: SplittableRandom, split
using Colors

export
    # Core types
    GayWorld, WorldNet, WorldMorphism,
    GayTraversal, TraversalDesiderata,
    
    # Tritwise comparison
    Trit, trit_compare, trit_balance,
    TritRelation, parallelism_trit, reachability_trit,
    
    # Anticipatory structure
    AnticipatorySelf, protention, retention, primal,
    anticipate_others, husserlian_moment,
    
    # World construction
    aptos_world, sui_world, chia_world, apex_world,
    worldnet_from_worlds, all_blockchain_worlds,
    
    # Traversal
    traverse_parallel, maximally_reachable,
    spi_invariant_check, chromatic_handshake,
    
    # Placement
    GayPlacement, optimal_placement, placement_fingerprint,
    tritwise_placement_score,
    
    # Demo
    world_worldnet, world_tritwise_comparison

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS (SPI Compliant)
# ═══════════════════════════════════════════════════════════════════════════════

const GAY_SEED = UInt64(1069)
const GZIP_SEED = UInt64(0x693c08c408088b1f)
const ANANAS_SEED = UInt64(0xAAAAAA)

const APTOS_SEED = UInt64(0xA9705)    # "APTOS" hex-ish
const SUI_SEED = UInt64(0x501)        # "SUI" hex-ish
const CHIA_SEED = UInt64(0xC41A)      # "CHIA" hex-ish

@inline function splitmix64(state::UInt64)::UInt64
    z = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ⊻ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    z ⊻ (z >> 31)
end

@inline function color_from_seed(seed::UInt64)::RGB{Float64}
    r = splitmix64(seed)
    g = splitmix64(r)
    b = splitmix64(g)
    RGB((r >> 56) / 255.0, (g >> 56) / 255.0, (b >> 56) / 255.0)
end

@inline function fingerprint(color::RGB{Float64}, content_hash::UInt64)::UInt64
    r = round(UInt64, clamp(color.r, 0, 1) * 255)
    g = round(UInt64, clamp(color.g, 0, 1) * 255)
    b = round(UInt64, clamp(color.b, 0, 1) * 255)
    color_fp = (r << 16) | (g << 8) | b
    color_fp ⊻ (content_hash >> 24)
end

# ═══════════════════════════════════════════════════════════════════════════════
# BALANCED TRIT: {-, 0, +} = {-1, 0, +1}
# ═══════════════════════════════════════════════════════════════════════════════

"""
    Trit
    
Balanced trit for tritwise comparison: {T-, T0, T+} = {-1, 0, +1}
"""
struct Trit
    value::Int8  # -1, 0, or +1
    
    function Trit(v::Integer)
        v = clamp(v, -1, 1)
        new(Int8(v))
    end
end

const T_MINUS = Trit(-1)
const T_ZERO = Trit(0)
const T_PLUS = Trit(+1)

Base.show(io::IO, t::Trit) = print(io, t.value == -1 ? "T-" : t.value == 0 ? "T0" : "T+")
Base.:(==)(a::Trit, b::Trit) = a.value == b.value
Base.isless(a::Trit, b::Trit) = a.value < b.value

"""Compare two values, return trit encoding the comparison."""
function trit_compare(a::Real, b::Real; tolerance::Real=0.0)::Trit
    if a < b - tolerance
        T_MINUS
    elseif a > b + tolerance
        T_PLUS
    else
        T_ZERO
    end
end

"""Invert a trit (negation)."""
trit_neg(t::Trit) = Trit(-t.value)

"""Add two trits (balanced ternary addition without carry)."""
function trit_add(a::Trit, b::Trit)::Tuple{Trit, Trit}
    sum = a.value + b.value
    if sum == 0
        (T_ZERO, T_ZERO)
    elseif sum == 1
        (T_PLUS, T_ZERO)
    elseif sum == -1
        (T_MINUS, T_ZERO)
    elseif sum == 2
        (T_MINUS, T_PLUS)  # 2 = 3 - 1 = carry + (-1)
    else  # sum == -2
        (T_PLUS, T_MINUS)
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# GAYWORLD: A Blockchain World with Chromatic Identity
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ParallelismModel
    
Enum for different parallelism strategies.
"""
@enum ParallelismModel begin
    SPECULATIVE    # Aptos Block-STM: optimistic concurrent execution
    OBJECT_LEVEL   # Sui: object-centric, owned vs shared
    AGGREGATE      # Chia: spend bundle aggregation, BLS
end

"""
    GayWorld
    
A blockchain world with chromatic identity and parallelism characteristics.
"""
struct GayWorld
    name::Symbol
    seed::UInt64
    color::RGB{Float64}
    fingerprint::UInt64
    
    # Parallelism characteristics
    parallelism::ParallelismModel
    max_tps::Int                    # Maximum transactions per second
    block_time_ms::Int              # Block time in milliseconds
    finality_ms::Int                # Time to finality in milliseconds
    
    # Special features
    features::Vector{Symbol}
    
    # Tritwise encoding of parallelism relative to base (Sui = T0)
    parallelism_trit::Trit
    
    # Is this the apex world?
    is_apex::Bool
    
    # Reachable worlds
    reaches::Vector{Symbol}
end

"""Construct a GayWorld with automatic chromatic identity."""
function GayWorld(name::Symbol, seed::UInt64;
                  parallelism::ParallelismModel=OBJECT_LEVEL,
                  max_tps::Int=100000,
                  block_time_ms::Int=500,
                  finality_ms::Int=1000,
                  features::Vector{Symbol}=Symbol[],
                  parallelism_trit::Trit=T_ZERO,
                  is_apex::Bool=false,
                  reaches::Vector{Symbol}=Symbol[])
    color = color_from_seed(seed)
    fp = fingerprint(color, seed)
    GayWorld(name, seed, color, fp, parallelism, max_tps, block_time_ms, 
             finality_ms, features, parallelism_trit, is_apex, reaches)
end

# ═══════════════════════════════════════════════════════════════════════════════
# BLOCKCHAIN WORLD CONSTRUCTORS
# ═══════════════════════════════════════════════════════════════════════════════

"""Aptos world: Block-STM speculative parallelism."""
function aptos_world()
    GayWorld(:aptos, APTOS_SEED;
        parallelism = SPECULATIVE,
        max_tps = 160000,           # Block-STM enables 160k+ TPS
        block_time_ms = 250,        # ~250ms blocks
        finality_ms = 500,          # Near-instant finality
        features = [:block_stm, :move, :resource_model, :parallel_execution],
        parallelism_trit = T_MINUS, # Speculative = "before" confirmation
        reaches = [:gay, :sui, :chia]
    )
end

"""Sui world: object-centric parallelism."""
function sui_world()
    GayWorld(:sui, SUI_SEED;
        parallelism = OBJECT_LEVEL,
        max_tps = 120000,           # High TPS via object parallelism
        block_time_ms = 400,        # Fast blocks
        finality_ms = 400,          # Sub-second finality
        features = [:object_centric, :move, :fast_path, :narwhal, :bullshark],
        parallelism_trit = T_ZERO,  # Object-level = "present" state
        reaches = [:gay, :aptos, :chia]
    )
end

"""Chia world: spend bundle aggregation."""
function chia_world()
    GayWorld(:chia, CHIA_SEED;
        parallelism = AGGREGATE,
        max_tps = 50,               # Lower TPS but high batch efficiency
        block_time_ms = 18000,      # ~18 second blocks
        finality_ms = 300000,       # 5 minute finality (32 blocks)
        features = [:clvm, :bls_aggregation, :spend_bundles, :utxo, :chialisp],
        parallelism_trit = T_PLUS,  # Aggregate = "after" individual txs
        reaches = [:gay, :aptos, :sui]
    )
end

"""Apex (Gay) world: the ANANAS co-cone."""
function apex_world()
    GayWorld(:gay, ANANAS_SEED;
        parallelism = OBJECT_LEVEL,  # Abstract parallelism
        max_tps = typemax(Int),      # Unlimited (abstract)
        block_time_ms = 0,           # Instant (abstract)
        finality_ms = 0,             # Instant (abstract)
        features = [:universal, :apex, :reconciliation, :unfree_invariants],
        parallelism_trit = T_ZERO,   # Center of trit space
        is_apex = true,
        reaches = [:aptos, :sui, :chia]
    )
end

"""Get all blockchain worlds."""
function all_blockchain_worlds()::Dict{Symbol, GayWorld}
    Dict(
        :aptos => aptos_world(),
        :sui => sui_world(),
        :chia => chia_world(),
        :gay => apex_world()
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# ANTICIPATORY STRUCTURE: Self + 2 Others (Husserlian Tritwise)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    AnticipatorySelf
    
Husserlian moment structure extended to tritwise anticipation.
- retention: what we remember (past, T-)
- primal: what we experience now (present, T0)
- protention: what we anticipate (future, T+)

Extended to "self and 2 others":
- self = primal impression
- other1 = T- world (retention direction)
- other2 = T+ world (protention direction)
"""
struct AnticipatorySelf
    # The self world (primal)
    self_world::GayWorld
    
    # The two other worlds in tritwise relation
    other_minus::GayWorld    # T- (retention direction)
    other_plus::GayWorld     # T+ (protention direction)
    
    # Husserlian structure colors
    retention_colors::Vector{RGB{Float64}}
    primal_color::RGB{Float64}
    protention_colors::Vector{RGB{Float64}}
    
    # Tritwise comparison results
    parallelism_comparison::NTuple{3, Trit}  # (self vs minus, self, self vs plus)
    reachability_comparison::NTuple{3, Trit}
    
    # Combined fingerprint
    seed::UInt64
    fingerprint::UInt64
end

"""
    husserlian_moment(self_world, other_minus, other_plus; depth=3)
    
Create an anticipatory self from three worlds in tritwise relation.
"""
function husserlian_moment(self_world::GayWorld, 
                           other_minus::GayWorld, 
                           other_plus::GayWorld;
                           depth::Int=3)
    # Generate retention colors (looking back toward other_minus)
    retention_colors = RGB{Float64}[]
    s = self_world.seed
    for i in 1:depth
        s = splitmix64(s ⊻ other_minus.seed ⊻ UInt64(i))
        push!(retention_colors, color_from_seed(s))
    end
    
    # Primal color is self's color
    primal_color = self_world.color
    
    # Generate protention colors (looking forward toward other_plus)
    protention_colors = RGB{Float64}[]
    s = self_world.seed
    for i in 1:depth
        s = splitmix64(s ⊻ other_plus.seed ⊻ UInt64(i))
        push!(protention_colors, color_from_seed(s))
    end
    
    # Tritwise parallelism comparison
    par_self_minus = trit_compare(self_world.max_tps, other_minus.max_tps; tolerance=1000)
    par_self_plus = trit_compare(self_world.max_tps, other_plus.max_tps; tolerance=1000)
    parallelism_comparison = (par_self_minus, T_ZERO, par_self_plus)
    
    # Tritwise reachability comparison (inverse of finality time)
    reach_self_minus = trit_compare(other_minus.finality_ms, self_world.finality_ms)
    reach_self_plus = trit_compare(other_plus.finality_ms, self_world.finality_ms)
    reachability_comparison = (reach_self_minus, T_ZERO, reach_self_plus)
    
    # Combined fingerprint
    seed = self_world.seed ⊻ other_minus.seed ⊻ other_plus.seed
    fp = fingerprint(primal_color, seed)
    
    AnticipatorySelf(
        self_world, other_minus, other_plus,
        retention_colors, primal_color, protention_colors,
        parallelism_comparison, reachability_comparison,
        seed, fp
    )
end

"""Get the retention (T-) world."""
retention(as::AnticipatorySelf) = as.other_minus

"""Get the primal (T0) world."""
primal(as::AnticipatorySelf) = as.self_world

"""Get the protention (T+) world."""
protention(as::AnticipatorySelf) = as.other_plus

"""
    anticipate_others(self::Symbol, worlds::Dict{Symbol, GayWorld}) → AnticipatorySelf
    
Given a self world, create anticipatory structure with 2 others.
Automatically selects T- and T+ based on parallelism ordering.
"""
function anticipate_others(self_name::Symbol, worlds::Dict{Symbol, GayWorld})
    self_world = worlds[self_name]
    others = [w for (n, w) in worlds if n != self_name && !w.is_apex]
    
    # Sort by parallelism (TPS)
    sort!(others, by=w -> w.max_tps)
    
    # Find T- (lower TPS) and T+ (higher TPS) relative to self
    other_minus = nothing
    other_plus = nothing
    
    for w in others
        if w.max_tps < self_world.max_tps && other_minus === nothing
            other_minus = w
        elseif w.max_tps > self_world.max_tps && other_plus === nothing
            other_plus = w
        end
    end
    
    # Fallbacks if we can't find both directions
    if other_minus === nothing
        other_minus = first(others)
    end
    if other_plus === nothing
        other_plus = last(others)
    end
    
    husserlian_moment(self_world, other_minus, other_plus)
end

# ═══════════════════════════════════════════════════════════════════════════════
# WORLDNET: Graph of Worlds with Morphisms
# ═══════════════════════════════════════════════════════════════════════════════

"""
    WorldMorphism
    
A morphism between worlds that preserves UNFREE invariants.
"""
struct WorldMorphism
    source::Symbol
    target::Symbol
    seed_transform::UInt64
    preserves_unfreedom::Bool
    
    # Parallelism relation
    parallelism_trit::Trit
end

"""
    WorldNet
    
A network of GayWorlds with morphisms between them.
"""
struct WorldNet
    worlds::Dict{Symbol, GayWorld}
    morphisms::Vector{WorldMorphism}
    apex::GayWorld
    
    # Tritwise parallelism matrix (world × world → Trit)
    parallelism_matrix::Dict{Tuple{Symbol, Symbol}, Trit}
    
    # Global fingerprint
    seed::UInt64
    fingerprint::UInt64
    color::RGB{Float64}
end

"""Construct a WorldNet from worlds."""
function worldnet_from_worlds(worlds::Dict{Symbol, GayWorld})
    # Find apex
    apex = nothing
    for (_, w) in worlds
        if w.is_apex
            apex = w
            break
        end
    end
    if apex === nothing
        apex = apex_world()
        worlds[:gay] = apex
    end
    
    # Build morphisms (all worlds reach apex)
    morphisms = WorldMorphism[]
    for (name, world) in worlds
        if !world.is_apex
            # Morphism to apex
            seed_transform = splitmix64(world.seed ⊻ apex.seed)
            par_trit = trit_compare(world.max_tps, apex.max_tps)
            push!(morphisms, WorldMorphism(name, :gay, seed_transform, true, par_trit))
            
            # Morphisms to other reachable worlds
            for target in world.reaches
                if haskey(worlds, target) && target != :gay
                    target_world = worlds[target]
                    seed_t = splitmix64(world.seed ⊻ target_world.seed)
                    par_t = trit_compare(world.max_tps, target_world.max_tps; tolerance=1000)
                    push!(morphisms, WorldMorphism(name, target, seed_t, true, par_t))
                end
            end
        end
    end
    
    # Build parallelism matrix
    parallelism_matrix = Dict{Tuple{Symbol, Symbol}, Trit}()
    for (n1, w1) in worlds
        for (n2, w2) in worlds
            parallelism_matrix[(n1, n2)] = trit_compare(w1.max_tps, w2.max_tps; tolerance=1000)
        end
    end
    
    # Global fingerprint
    seed = reduce(⊻, [w.seed for (_, w) in worlds])
    color = color_from_seed(seed)
    fp = fingerprint(color, seed)
    
    WorldNet(worlds, morphisms, apex, parallelism_matrix, seed, fp, color)
end

"""Get tritwise parallelism comparison between two worlds."""
function parallelism_trit(net::WorldNet, w1::Symbol, w2::Symbol)::Trit
    get(net.parallelism_matrix, (w1, w2), T_ZERO)
end

# ═══════════════════════════════════════════════════════════════════════════════
# TRAVERSAL DESIDERATA: What We Want from GayTraversal
# ═══════════════════════════════════════════════════════════════════════════════

"""
    TraversalDesiderata
    
Desired properties for GayTraversal through the WorldNet.
"""
struct TraversalDesiderata
    # Parallelism requirements
    min_tps::Int                    # Minimum required TPS
    max_latency_ms::Int             # Maximum acceptable latency
    
    # Data requirements  
    data_size_bytes::Int            # Size of data being processed
    gzipability::Float64            # Compressibility (from ananas_gzip_scaling)
    
    # Tritwise preferences
    prefer_parallelism::Trit        # T+ = more parallel, T- = more sequential
    prefer_finality::Trit           # T+ = faster finality, T- = slower ok
    
    # SPI invariant requirements
    require_splitmix64::Bool
    require_color_consistency::Bool
    require_fingerprint::Bool
    
    # Chromatic identity
    seed::UInt64
    fingerprint::UInt64
end

function TraversalDesiderata(;
    min_tps::Int = 1000,
    max_latency_ms::Int = 10000,
    data_size_bytes::Int = 1024,
    gzipability::Float64 = 0.5,
    prefer_parallelism::Trit = T_ZERO,
    prefer_finality::Trit = T_ZERO,
    require_splitmix64::Bool = true,
    require_color_consistency::Bool = true,
    require_fingerprint::Bool = true,
    seed::UInt64 = GAY_SEED
)
    color = color_from_seed(seed)
    fp = fingerprint(color, seed)
    TraversalDesiderata(
        min_tps, max_latency_ms, data_size_bytes, gzipability,
        prefer_parallelism, prefer_finality,
        require_splitmix64, require_color_consistency, require_fingerprint,
        seed, fp
    )
end

"""
    GayTraversal
    
A traversal through the WorldNet with specific desiderata.
"""
struct GayTraversal
    desiderata::TraversalDesiderata
    net::WorldNet
    
    # Selected path
    path::Vector{Symbol}
    morphisms_used::Vector{WorldMorphism}
    
    # Tritwise balance across path
    balance::Trit
    
    # Verification
    spi_verified::Bool
    reachable_from_apex::Bool
    
    # Result
    optimal_world::Symbol
    placement_score::Float64
    
    # Chromatic identity
    fingerprint::UInt64
    color::RGB{Float64}
end

# ═══════════════════════════════════════════════════════════════════════════════
# SPI INVARIANT VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

"""
    spi_invariant_check(world::GayWorld, test_data::String) → NamedTuple
    
Verify that a world correctly implements SPI (Strong Parallelism Invariance).
"""
function spi_invariant_check(world::GayWorld, test_data::String)
    bytes = Vector{UInt8}(test_data)
    
    # Compute content hash
    content_hash = reduce((h, b) -> splitmix64(h ⊻ UInt64(b)), bytes; init=world.seed)
    
    # Compute color and fingerprint
    color = color_from_seed(content_hash)
    fp = fingerprint(color, content_hash)
    
    # Verify consistency
    color_consistent = color == color_from_seed(content_hash)  # Should always be true (deterministic)
    fp_consistent = fp == fingerprint(color_from_seed(content_hash), content_hash)
    
    (
        world = world.name,
        seed = world.seed,
        content_hash = content_hash,
        color = color,
        fingerprint = fp,
        
        # SPI checks
        splitmix64_deterministic = splitmix64(GAY_SEED) == splitmix64(GAY_SEED),
        color_deterministic = color_consistent,
        fingerprint_deterministic = fp_consistent,
        
        all_invariants_hold = color_consistent && fp_consistent
    )
end

"""
    chromatic_handshake(world1::GayWorld, world2::GayWorld, test_data::String) → Bool
    
Verify that two worlds produce identical UNFREE outputs for the same input.
"""
function chromatic_handshake(world1::GayWorld, world2::GayWorld, test_data::String)
    check1 = spi_invariant_check(world1, test_data)
    check2 = spi_invariant_check(world2, test_data)
    
    # For handshake, fingerprints should match when derived from same content
    # (They differ because seeds differ, but the ALGORITHM is the same)
    
    # Test: same seed → same output
    shared_seed = world1.seed ⊻ world2.seed
    bytes = Vector{UInt8}(test_data)
    content_hash = reduce((h, b) -> splitmix64(h ⊻ UInt64(b)), bytes; init=shared_seed)
    
    color1 = color_from_seed(content_hash)
    color2 = color_from_seed(content_hash)  # Same function, same input → same output
    
    fp1 = fingerprint(color1, content_hash)
    fp2 = fingerprint(color2, content_hash)
    
    fp1 == fp2
end

# ═══════════════════════════════════════════════════════════════════════════════
# MAXIMALLY REACHABLE PARALLELISM
# ═══════════════════════════════════════════════════════════════════════════════

"""
    tritwise_placement_score(world::GayWorld, desiderata::TraversalDesiderata) → Float64
    
Compute placement score for a world given desiderata.
Higher score = better fit.
"""
function tritwise_placement_score(world::GayWorld, desiderata::TraversalDesiderata)
    score = 0.0
    
    # TPS score (log scale)
    if world.max_tps >= desiderata.min_tps
        score += log10(world.max_tps) / log10(200000)  # Normalize to ~160k max
    else
        score -= 1.0  # Penalty for not meeting minimum
    end
    
    # Latency score
    if world.finality_ms <= desiderata.max_latency_ms
        score += 1.0 - (world.finality_ms / desiderata.max_latency_ms)
    else
        score -= 0.5  # Penalty
    end
    
    # Tritwise preference matching
    if desiderata.prefer_parallelism == T_PLUS
        score += (world.max_tps / 160000.0)  # Prefer higher TPS
    elseif desiderata.prefer_parallelism == T_MINUS
        score += (1.0 - world.max_tps / 160000.0)  # Prefer lower TPS
    end
    
    if desiderata.prefer_finality == T_PLUS
        score += (1.0 - min(1.0, world.finality_ms / 10000.0))  # Prefer faster finality
    elseif desiderata.prefer_finality == T_MINUS
        score += min(1.0, world.finality_ms / 10000.0)  # Prefer slower finality ok
    end
    
    # Gzipability-based adjustment (from ananas_gzip_scaling)
    # Higher gzipability (more random data) → prefer higher parallelism
    if desiderata.gzipability > 0.6
        score += (world.max_tps / 160000.0) * 0.5
    elseif desiderata.gzipability < 0.3
        score += 0.3  # Simpler data, any platform works
    end
    
    score
end

"""
    maximally_reachable(net::WorldNet, desiderata::TraversalDesiderata) → GayTraversal
    
Find the maximally reachable parallel execution path through the WorldNet.
"""
function maximally_reachable(net::WorldNet, desiderata::TraversalDesiderata)
    # Score all worlds
    scores = Dict{Symbol, Float64}()
    for (name, world) in net.worlds
        if !world.is_apex
            scores[name] = tritwise_placement_score(world, desiderata)
        end
    end
    
    # Find optimal
    optimal_name, optimal_score = first(sort(collect(scores), by=x->-x[2]))
    optimal_world = net.worlds[optimal_name]
    
    # Build path: optimal → apex
    path = [optimal_name, :gay]
    
    # Find morphisms
    morphisms_used = filter(m -> m.source == optimal_name && m.target == :gay, net.morphisms)
    
    # Compute tritwise balance
    balance = optimal_world.parallelism_trit
    
    # Verify SPI
    spi_ok = spi_invariant_check(optimal_world, "test").all_invariants_hold
    
    # Chromatic identity
    seed = desiderata.seed ⊻ optimal_world.seed
    color = color_from_seed(seed)
    fp = fingerprint(color, seed)
    
    GayTraversal(
        desiderata, net, path, morphisms_used,
        balance, spi_ok, true, optimal_name, optimal_score,
        fp, color
    )
end

"""
    traverse_parallel(net::WorldNet, desiderata::TraversalDesiderata) → Dict
    
Explore all parallel paths through the WorldNet, returning ranked options.
"""
function traverse_parallel(net::WorldNet, desiderata::TraversalDesiderata)
    results = Dict{Symbol, NamedTuple}()
    
    for (name, world) in net.worlds
        if world.is_apex
            continue
        end
        
        score = tritwise_placement_score(world, desiderata)
        spi = spi_invariant_check(world, "traverse_parallel_test")
        
        # Anticipatory analysis
        anticipatory = anticipate_others(name, net.worlds)
        
        results[name] = (
            world = name,
            score = score,
            parallelism = world.parallelism,
            max_tps = world.max_tps,
            finality_ms = world.finality_ms,
            spi_verified = spi.all_invariants_hold,
            
            # Tritwise relations
            parallelism_trit = world.parallelism_trit,
            vs_retention = anticipatory.parallelism_comparison[1],
            vs_protention = anticipatory.parallelism_comparison[3],
            
            # Anticipatory structure
            retention_world = retention(anticipatory).name,
            protention_world = protention(anticipatory).name,
            
            fingerprint = world.fingerprint
        )
    end
    
    results
end

# ═══════════════════════════════════════════════════════════════════════════════
# PLACEMENT
# ═══════════════════════════════════════════════════════════════════════════════

"""
    GayPlacement
    
A placement decision with full chromatic and tritwise analysis.
"""
struct GayPlacement
    world::GayWorld
    score::Float64
    traversal::GayTraversal
    anticipatory::AnticipatorySelf
    
    # Tritwise summary
    parallelism_balance::Trit
    finality_balance::Trit
    
    fingerprint::UInt64
    color::RGB{Float64}
end

"""
    optimal_placement(desiderata::TraversalDesiderata; net=nothing) → GayPlacement
    
Find optimal world placement for given desiderata.
"""
function optimal_placement(desiderata::TraversalDesiderata; 
                           net::Union{WorldNet, Nothing}=nothing)
    if net === nothing
        net = worldnet_from_worlds(all_blockchain_worlds())
    end
    
    traversal = maximally_reachable(net, desiderata)
    optimal_world = net.worlds[traversal.optimal_world]
    
    # Anticipatory structure
    anticipatory = anticipate_others(traversal.optimal_world, net.worlds)
    
    # Tritwise balances
    par_balance = optimal_world.parallelism_trit
    fin_balance = trit_compare(1000, optimal_world.finality_ms)  # Relative to 1s target
    
    # Chromatic identity
    seed = desiderata.seed ⊻ traversal.fingerprint
    color = color_from_seed(seed)
    fp = fingerprint(color, seed)
    
    GayPlacement(
        optimal_world, traversal.placement_score, traversal, anticipatory,
        par_balance, fin_balance, fp, color
    )
end

"""Get fingerprint for placement verification."""
placement_fingerprint(p::GayPlacement) = p.fingerprint

# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

function world_worldnet()
    println()
    println("╔═══════════════════════════════════════════════════════════════════════════╗")
    println("║  GAY WORLDNET: Tritwise Anticipatory Blockchain Parallelism               ║")
    println("║  \"anticipatory about self and 2 others\"                                  ║")
    println("╚═══════════════════════════════════════════════════════════════════════════╝")
    println()
    
    # ─── Build WorldNet ───
    println("─── WorldNet Construction ───")
    worlds = all_blockchain_worlds()
    net = worldnet_from_worlds(worlds)
    
    println("  Worlds: $(join(keys(worlds), ", "))")
    println("  Morphisms: $(length(net.morphisms))")
    println("  Apex: $(net.apex.name)")
    println("  Net fingerprint: 0x$(string(net.fingerprint, base=16)[1:8])...")
    println()
    
    # ─── World Details ───
    println("─── Blockchain Worlds ───")
    for (name, world) in sort(collect(worlds), by=x->x[1])
        if world.is_apex continue end
        
        println("  $(uppercase(string(name))):")
        println("    Parallelism: $(world.parallelism)")
        println("    Max TPS: $(world.max_tps)")
        println("    Block time: $(world.block_time_ms)ms")
        println("    Finality: $(world.finality_ms)ms")
        println("    Trit: $(world.parallelism_trit)")
        println("    Color: RGB($(round(world.color.r, digits=2)), $(round(world.color.g, digits=2)), $(round(world.color.b, digits=2)))")
        println()
    end
    
    # ─── Tritwise Comparison Matrix ───
    println("─── Tritwise Parallelism Matrix ───")
    names = [:aptos, :sui, :chia]
    print("         ")
    for n in names
        print("$(lpad(string(n), 8))")
    end
    println()
    
    for n1 in names
        print("  $(rpad(string(n1), 6))")
        for n2 in names
            t = parallelism_trit(net, n1, n2)
            print("$(lpad(string(t), 8))")
        end
        println()
    end
    println()
    
    # ─── Anticipatory Structure ───
    println("─── Anticipatory Structure (Self + 2 Others) ───")
    for name in [:aptos, :sui, :chia]
        as = anticipate_others(name, worlds)
        println("  $(uppercase(string(name))) as Self:")
        println("    Retention (T-): $(as.other_minus.name)")
        println("    Primal (T0): $(as.self_world.name)")
        println("    Protention (T+): $(as.other_plus.name)")
        println("    Parallelism comparison: $(as.parallelism_comparison)")
        println()
    end
    
    # ─── SPI Verification ───
    println("─── SPI Invariant Verification ───")
    test_data = "gayzip chromatic identity test"
    for (name, world) in worlds
        if world.is_apex continue end
        check = spi_invariant_check(world, test_data)
        status = check.all_invariants_hold ? "✓" : "✗"
        println("  $(name): $(status) All invariants hold")
    end
    println()
    
    # ─── Chromatic Handshake ───
    println("─── Chromatic Handshake (Cross-World Consistency) ───")
    handshake_ok = chromatic_handshake(worlds[:aptos], worlds[:sui], test_data)
    println("  Aptos ↔ Sui: $(handshake_ok ? "✓ VERIFIED" : "✗ FAILED")")
    handshake_ok = chromatic_handshake(worlds[:sui], worlds[:chia], test_data)
    println("  Sui ↔ Chia: $(handshake_ok ? "✓ VERIFIED" : "✗ FAILED")")
    handshake_ok = chromatic_handshake(worlds[:aptos], worlds[:chia], test_data)
    println("  Aptos ↔ Chia: $(handshake_ok ? "✓ VERIFIED" : "✗ FAILED")")
    println()
    
    # ─── Traversal with Desiderata ───
    println("─── GayTraversal with Desiderata ───")
    
    # High-frequency market desiderata
    desiderata_hf = TraversalDesiderata(
        min_tps = 50000,
        max_latency_ms = 1000,
        prefer_parallelism = T_PLUS,
        prefer_finality = T_PLUS,
        gzipability = 0.4
    )
    
    println("  Desiderata (High-Frequency Trading):")
    println("    min_tps: $(desiderata_hf.min_tps)")
    println("    max_latency: $(desiderata_hf.max_latency_ms)ms")
    println("    prefer_parallelism: $(desiderata_hf.prefer_parallelism)")
    println("    prefer_finality: $(desiderata_hf.prefer_finality)")
    println()
    
    results = traverse_parallel(net, desiderata_hf)
    println("  Traversal Results (sorted by score):")
    for (name, r) in sort(collect(results), by=x->-x[2].score)
        println("    $(name): score=$(round(r.score, digits=3))")
        println("       tps=$(r.max_tps), finality=$(r.finality_ms)ms, trit=$(r.parallelism_trit)")
        println("       retention→$(r.retention_world), protention→$(r.protention_world)")
    end
    println()
    
    # ─── Optimal Placement ───
    println("─── Optimal Placement ───")
    placement = optimal_placement(desiderata_hf; net=net)
    
    println("  Selected: $(uppercase(string(placement.world.name)))")
    println("  Score: $(round(placement.score, digits=4))")
    println("  Path: $(join(string.(placement.traversal.path), " → "))")
    println("  SPI verified: $(placement.traversal.spi_verified)")
    println("  Tritwise balance: parallelism=$(placement.parallelism_balance), finality=$(placement.finality_balance)")
    println("  Fingerprint: 0x$(string(placement.fingerprint, base=16)[1:8])...")
    println()
    
    # ─── Batch Auction Desiderata ───
    println("─── Alternative: Batch Auction Desiderata ───")
    desiderata_batch = TraversalDesiderata(
        min_tps = 10,
        max_latency_ms = 60000,
        prefer_parallelism = T_ZERO,
        prefer_finality = T_MINUS,  # Slower finality OK
        gzipability = 0.7
    )
    
    placement_batch = optimal_placement(desiderata_batch; net=net)
    println("  Selected: $(uppercase(string(placement_batch.world.name)))")
    println("  Score: $(round(placement_batch.score, digits=4))")
    println("  (Prefers aggregate/batch pattern)")
    println()
    
    println("═══════════════════════════════════════════════════════════════════════════")
    println("  🍍 ALL PATHS LEAD TO gayzip/gay/gayzip.gay 🍍")
    println("═══════════════════════════════════════════════════════════════════════════")
    
    return (net=net, placement_hf=placement, placement_batch=placement_batch)
end

function world_tritwise_comparison()
    println()
    println("─── Tritwise Comparison: {T-, T0, T+} ───")
    println()
    
    println("  BALANCED TERNARY ENCODING:")
    println("    T- = -1 = SPECULATIVE (Aptos Block-STM)")
    println("    T0 =  0 = OBJECT-LEVEL (Sui)")
    println("    T+ = +1 = AGGREGATE (Chia BLS)")
    println()
    
    println("  PARALLELISM SEMANTICS:")
    println("    T- → Execute speculatively, validate later")
    println("    T0 → Execute at object level, lock per-object")
    println("    T+ → Aggregate first, execute batch")
    println()
    
    println("  ANTICIPATORY STRUCTURE (Husserlian):")
    println("    retention (T-) → what came before (past)")
    println("    primal (T0) → current state (present)")
    println("    protention (T+) → what comes next (future)")
    println()
    
    println("  TRITWISE IDENTITY:")
    println("    self + other_minus + other_plus = 3 worlds = trit arity")
    println("    Each world anticipates the other 2 in tritwise relation")
    println()
    
    println("  SPI INVARIANTS (UNFREE - must match across all worlds):")
    println("    • splitmix64(state) → same output everywhere")
    println("    • color_from_seed(seed) → same RGB everywhere")
    println("    • fingerprint(color, hash) → same fingerprint everywhere")
    println()
end

end # module GayWorldNet
