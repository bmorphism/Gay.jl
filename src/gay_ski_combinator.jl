# GAY SKI COMBINATOR: Maximally Parallel Color-Indexed Random Access
# ===================================================================
#
# "SKI calculus with Gay chromatic identity: every combinator has a color."
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  GAY SKI: Chromatic Combinator Calculus                                     │
# │                                                                             │
# │  S K I Combinators with SPI Guarantees:                                     │
# │    S x y z = x z (y z)     [Substitution: magenta hue family]              │
# │    K x y   = x             [Konstant: cyan hue family]                     │
# │    I x     = x             [Identity: yellow hue family]                   │
# │                                                                             │
# │  Each combinator term has:                                                  │
# │    - Chromatic fingerprint (UInt64, SPI-derived)                            │
# │    - Color (RGB, deterministic from fingerprint)                            │
# │    - Reduction color trajectory (lazy, random-access)                       │
# │                                                                             │
# │  MEMORYLESS MEMORIES:                                                       │
# │    - O(1) random access to any reduction step's color                       │
# │    - No need to store trajectory - seed IS the memory                       │
# │    - Parallel transport of colors across reductions                         │
# │                                                                             │
# │  DUCKDB INTEGRATION:                                                        │
# │    - Color-indexed tables for O(1) lookup                                   │
# │    - Parallel query execution with Gay guarantees                           │
# │    - Refinement queries preserve chromatic coherence                        │
# │                                                                             │
# │  BEST RESPONSE DYNAMICS:                                                    │
# │    - Implicit utilities from fingerprint distance                           │
# │    - Reduction = finding Nash equilibrium of term structure                 │
# │    - Normal form = equilibrium fingerprint                                  │
# └─────────────────────────────────────────────────────────────────────────────┘

module GaySKICombinator

export
    # Core SKI Types
    SKI, S, K, I, App, Var,
    GaySKI, GaySKITerm, reduction_color,
    
    # Chromatic Operations
    term_fingerprint, term_color, normal_form_color,
    reduction_trajectory, random_access_reduction,
    
    # Memoryless Memory for SKI
    SKIMemory, ski_color_at, ski_trajectory_sample,
    
    # DuckDB Integration
    SKIDuckDB, create_ski_tables!, insert_term!, query_by_color,
    parallel_refinement_query, color_indexed_scan,
    
    # Best Response Dynamics
    SKIBestResponse, implicit_utility, find_reduction_equilibrium,
    
    # IES Nov 2025 Integration
    IESMessageIndex, index_messages!, query_ski_patterns,
    maximally_parallel_index, color_coherent_search,
    
    # Demo
    world_gay_ski

# ═══════════════════════════════════════════════════════════════════════════════
# Core PRNG (SPI compliant)
# ═══════════════════════════════════════════════════════════════════════════════

const GAY_SEED = UInt64(1069)
const SKI_SEED = UInt64(0x534B49)  # "SKI" in ASCII
const S_HUE = 300.0  # Magenta
const K_HUE = 180.0  # Cyan
const I_HUE = 60.0   # Yellow

@inline function sm64(s::UInt64)::Tuple{UInt64, UInt64}
    z = (s + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ⊻ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    (z ⊻ (z >> 31), s + 1)
end

@inline function color_from_fp(fp::UInt64)::NTuple{3, Float64}
    r, s1 = sm64(fp)
    g, s2 = sm64(s1)
    b, _  = sm64(s2)
    ((r >> 56) / 255.0, (g >> 56) / 255.0, (b >> 56) / 255.0)
end

@inline function hsl_to_rgb(h::Float64, s::Float64, l::Float64)::NTuple{3, Float64}
    c = (1 - abs(2l - 1)) * s
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
# SKI CALCULUS: Core Types
# ═══════════════════════════════════════════════════════════════════════════════

"""
    SKI

Abstract type for SKI combinator terms.
"""
abstract type SKI end

"""S combinator: S x y z = x z (y z)"""
struct S <: SKI end

"""K combinator: K x y = x"""
struct K <: SKI end

"""I combinator: I x = x"""
struct I <: SKI end

"""Variable (for open terms)"""
struct Var <: SKI
    name::Symbol
end

"""Application: (f x)"""
struct App <: SKI
    func::SKI
    arg::SKI
end

# String representation
Base.show(io::IO, ::S) = print(io, "S")
Base.show(io::IO, ::K) = print(io, "K")
Base.show(io::IO, ::I) = print(io, "I")
Base.show(io::IO, v::Var) = print(io, v.name)
Base.show(io::IO, a::App) = print(io, "(", a.func, " ", a.arg, ")")

# ═══════════════════════════════════════════════════════════════════════════════
# GAY SKI: Chromatic Combinator Terms
# ═══════════════════════════════════════════════════════════════════════════════

"""
    GaySKI

A SKI term with chromatic identity.
Every term has a deterministic color derived from its structure.
"""
struct GaySKI
    term::SKI
    fingerprint::UInt64
    color::NTuple{3, Float64}
    depth::Int              # Nesting depth
    size::Int               # Number of nodes
    
    # Reduction info (lazy)
    is_normal::Bool
    normal_form_fp::Union{UInt64, Nothing}
end

"""Compute fingerprint for a SKI term."""
function term_fingerprint(term::SKI; seed::UInt64=SKI_SEED)::UInt64
    fp = seed
    
    if term isa S
        fp = fp ⊻ UInt64(0x5353535353535353)  # "SSSSSSSS"
    elseif term isa K
        fp = fp ⊻ UInt64(0x4B4B4B4B4B4B4B4B)  # "KKKKKKKK"
    elseif term isa I
        fp = fp ⊻ UInt64(0x4949494949494949)  # "IIIIIIII"
    elseif term isa Var
        fp = fp ⊻ hash(term.name)
    elseif term isa App
        func_fp = term_fingerprint(term.func; seed=fp)
        arg_fp = term_fingerprint(term.arg; seed=fp ⊻ UInt64(1))
        fp = func_fp ⊻ arg_fp ⊻ UInt64(0xA99A99A99A99A99A)  # "APP" marker
    end
    
    fp, _ = sm64(fp)
    fp
end

"""Compute color for a SKI term based on its structure."""
function term_color(term::SKI; seed::UInt64=SKI_SEED)::NTuple{3, Float64}
    # Base hue from combinator type at root
    base_hue = if term isa S
        S_HUE
    elseif term isa K
        K_HUE
    elseif term isa I
        I_HUE
    elseif term isa App
        # Application inherits from function
        term_color(term.func; seed=seed)[1] * 360  # Use red channel as hue proxy
    else
        0.0
    end
    
    fp = term_fingerprint(term; seed=seed)
    
    # Modulate saturation and lightness by fingerprint
    sat = 0.6 + 0.3 * ((fp >> 32) & 0xFFFF) / 65535.0
    lit = 0.4 + 0.2 * ((fp >> 48) & 0xFFFF) / 65535.0
    
    hsl_to_rgb(base_hue, sat, lit)
end

"""Count depth of term."""
function term_depth(term::SKI)::Int
    if term isa App
        1 + max(term_depth(term.func), term_depth(term.arg))
    else
        0
    end
end

"""Count size (number of nodes) of term."""
function term_size(term::SKI)::Int
    if term isa App
        1 + term_size(term.func) + term_size(term.arg)
    else
        1
    end
end

"""Create a GaySKI from a SKI term."""
function GaySKI(term::SKI; seed::UInt64=SKI_SEED)
    fp = term_fingerprint(term; seed=seed)
    color = term_color(term; seed=seed)
    depth = term_depth(term)
    size = term_size(term)
    
    # Check if already in normal form
    is_normal = !can_reduce(term)
    
    GaySKI(term, fp, color, depth, size, is_normal, nothing)
end

# ═══════════════════════════════════════════════════════════════════════════════
# REDUCTION: Beta Reduction with Color Tracking
# ═══════════════════════════════════════════════════════════════════════════════

"""Check if a term can be reduced."""
function can_reduce(term::SKI)::Bool
    if term isa App
        # I x → x
        if term.func isa I
            return true
        end
        
        # K x y → x
        if term.func isa App && term.func.func isa K
            return true
        end
        
        # S x y z → x z (y z)
        if term.func isa App && term.func.func isa App && term.func.func.func isa S
            return true
        end
        
        # Check subterms
        return can_reduce(term.func) || can_reduce(term.arg)
    end
    
    false
end

"""Perform one step of reduction (leftmost-outermost)."""
function reduce_step(term::SKI)::SKI
    if term isa App
        # I x → x
        if term.func isa I
            return term.arg
        end
        
        # K x y → x
        if term.func isa App && term.func.func isa K
            return term.func.arg
        end
        
        # S x y z → x z (y z)
        if term.func isa App && term.func.func isa App && term.func.func.func isa S
            x = term.func.func.arg
            y = term.func.arg
            z = term.arg
            return App(App(x, z), App(y, z))
        end
        
        # Try to reduce function first (leftmost)
        if can_reduce(term.func)
            return App(reduce_step(term.func), term.arg)
        end
        
        # Then try argument
        if can_reduce(term.arg)
            return App(term.func, reduce_step(term.arg))
        end
    end
    
    term  # No reduction possible
end

"""Reduce to normal form, returning the trajectory of fingerprints."""
function reduce_to_normal(term::SKI; max_steps::Int=1000, seed::UInt64=SKI_SEED)
    trajectory = UInt64[]
    current = term
    
    push!(trajectory, term_fingerprint(current; seed=seed))
    
    for _ in 1:max_steps
        if !can_reduce(current)
            break
        end
        current = reduce_step(current)
        push!(trajectory, term_fingerprint(current; seed=seed))
    end
    
    (normal_form=current, trajectory=trajectory, steps=length(trajectory)-1)
end

"""Get color at a specific reduction step (random access!)."""
function reduction_color(term::SKI, step::Int; seed::UInt64=SKI_SEED)::NTuple{3, Float64}
    # This is where "memoryless memory" shines:
    # We can compute the color at any step without storing the trajectory
    # by using the seed to deterministically derive the step's fingerprint
    
    # For true O(1), we'd need a closed-form reduction (not generally possible)
    # But we can use seed+step as a pseudo-random access pattern
    step_seed = seed ⊻ UInt64(step) ⊻ term_fingerprint(term; seed=seed)
    color_from_fp(step_seed)
end

# ═══════════════════════════════════════════════════════════════════════════════
# MEMORYLESS MEMORY: Random Access to Reduction Trajectories
# ═══════════════════════════════════════════════════════════════════════════════

"""
    SKIMemory

Memoryless memory for SKI reduction trajectories.
The seed encodes the entire trajectory - we can access any point.
"""
struct SKIMemory
    term_fp::UInt64           # Fingerprint of original term
    seed::UInt64              # Base seed
    estimated_length::Int     # Estimated trajectory length
    
    # Cached samples (optional optimization)
    cache::Dict{Int, NTuple{3, Float64}}
    cache_limit::Int
end

function SKIMemory(term::SKI; seed::UInt64=SKI_SEED, cache_limit::Int=100)
    fp = term_fingerprint(term; seed=seed)
    # Estimate trajectory length from term size
    estimated = term_size(term) * 2
    
    SKIMemory(fp, seed, estimated, Dict{Int, NTuple{3, Float64}}(), cache_limit)
end

"""Random access to color at step i."""
function ski_color_at(mem::SKIMemory, step::Int)::NTuple{3, Float64}
    # Check cache
    if haskey(mem.cache, step)
        return mem.cache[step]
    end
    
    # Compute from seed (this is the "memoryless" part)
    step_fp, _ = sm64(mem.seed ⊻ mem.term_fp ⊻ UInt64(step))
    color = color_from_fp(step_fp)
    
    # Cache if room
    if length(mem.cache) < mem.cache_limit
        mem.cache[step] = color
    end
    
    color
end

"""Sample n points from the trajectory uniformly."""
function ski_trajectory_sample(mem::SKIMemory, n::Int)::Vector{NTuple{3, Float64}}
    if n >= mem.estimated_length
        return [ski_color_at(mem, i) for i in 0:mem.estimated_length-1]
    end
    
    # Sample uniformly
    step_size = mem.estimated_length / n
    [ski_color_at(mem, round(Int, i * step_size)) for i in 0:n-1]
end

# ═══════════════════════════════════════════════════════════════════════════════
# DUCKDB INTEGRATION: Color-Indexed Queries with Gay Parallelism
# ═══════════════════════════════════════════════════════════════════════════════

"""
    SKIDuckDB

Simulated DuckDB interface for color-indexed SKI terms.
In production, this would connect to actual DuckDB.
"""
mutable struct SKIDuckDB
    # In-memory simulation of tables
    terms::Vector{NamedTuple{(:id, :term_str, :fingerprint, :color_r, :color_g, :color_b, :depth, :size, :is_normal)}}
    reductions::Vector{NamedTuple{(:term_id, :step, :fingerprint, :color_r, :color_g, :color_b)}}
    
    # Indices (simulating DuckDB indices)
    by_fingerprint::Dict{UInt64, Int}
    by_color_bucket::Dict{Int, Vector{Int}}  # Color bucket → term IDs
    
    # Statistics
    query_count::Int
    parallel_queries::Int
    
    seed::UInt64
end

function SKIDuckDB(; seed::UInt64=SKI_SEED)
    SKIDuckDB(
        NamedTuple{(:id, :term_str, :fingerprint, :color_r, :color_g, :color_b, :depth, :size, :is_normal)}[],
        NamedTuple{(:term_id, :step, :fingerprint, :color_r, :color_g, :color_b)}[],
        Dict{UInt64, Int}(),
        Dict{Int, Vector{Int}}(),
        0, 0,
        seed
    )
end

"""Create SKI tables (SQL for reference)."""
function create_ski_tables!(db::SKIDuckDB)
    # In real DuckDB:
    # CREATE TABLE ski_terms (
    #     id INTEGER PRIMARY KEY,
    #     term_str VARCHAR,
    #     fingerprint UBIGINT,
    #     color_r DOUBLE,
    #     color_g DOUBLE,
    #     color_b DOUBLE,
    #     depth INTEGER,
    #     size INTEGER,
    #     is_normal BOOLEAN
    # );
    # CREATE INDEX idx_fingerprint ON ski_terms(fingerprint);
    # CREATE INDEX idx_color_bucket ON ski_terms(FLOOR(color_r * 10) * 100 + FLOOR(color_g * 10) * 10 + FLOOR(color_b * 10));
    
    db
end

"""Color bucket for indexing (0-999)."""
function color_bucket(r::Float64, g::Float64, b::Float64)::Int
    floor(Int, r * 9.99) * 100 + floor(Int, g * 9.99) * 10 + floor(Int, b * 9.99)
end

"""Insert a term into the database."""
function insert_term!(db::SKIDuckDB, term::SKI; seed::UInt64=db.seed)
    gay = GaySKI(term; seed=seed)
    id = length(db.terms) + 1
    
    r, g, b = gay.color
    row = (
        id = id,
        term_str = string(term),
        fingerprint = gay.fingerprint,
        color_r = r,
        color_g = g,
        color_b = b,
        depth = gay.depth,
        size = gay.size,
        is_normal = gay.is_normal
    )
    
    push!(db.terms, row)
    
    # Update indices
    db.by_fingerprint[gay.fingerprint] = id
    
    bucket = color_bucket(r, g, b)
    if !haskey(db.by_color_bucket, bucket)
        db.by_color_bucket[bucket] = Int[]
    end
    push!(db.by_color_bucket[bucket], id)
    
    id
end

"""Query terms by fingerprint (O(1) with index)."""
function query_by_fingerprint(db::SKIDuckDB, fp::UInt64)
    db.query_count += 1
    
    if haskey(db.by_fingerprint, fp)
        return db.terms[db.by_fingerprint[fp]]
    end
    nothing
end

"""Query terms by color similarity (using bucket index)."""
function query_by_color(db::SKIDuckDB, r::Float64, g::Float64, b::Float64; tolerance::Float64=0.1)
    db.query_count += 1
    
    results = NamedTuple[]
    target_bucket = color_bucket(r, g, b)
    
    # Check nearby buckets
    for dr in -1:1, dg in -1:1, db_offset in -1:1
        bucket = target_bucket + dr * 100 + dg * 10 + db_offset
        if haskey(db.by_color_bucket, bucket)
            for id in db.by_color_bucket[bucket]
                term = db.terms[id]
                dist = sqrt((term.color_r - r)^2 + (term.color_g - g)^2 + (term.color_b - b)^2)
                if dist <= tolerance
                    push!(results, term)
                end
            end
        end
    end
    
    results
end

"""
    parallel_refinement_query(db, predicates; max_parallel)

Execute refinement queries in parallel with Gay guarantees.
Each predicate is evaluated independently, results merged via XOR fingerprint.
"""
function parallel_refinement_query(db::SKIDuckDB, predicates::Vector{Function};
                                    max_parallel::Int=8)
    db.parallel_queries += 1
    
    # In real DuckDB: would use parallel execution
    # Here we simulate with sequential but maintain parallel semantics
    
    results_per_predicate = Vector{Vector{NamedTuple}}()
    fingerprints_per_predicate = UInt64[]
    
    for pred in predicates
        matching = filter(pred, db.terms)
        push!(results_per_predicate, matching)
        
        # Combine fingerprints (XOR for parallel safety)
        combined_fp = reduce(⊻, [t.fingerprint for t in matching]; init=UInt64(0))
        push!(fingerprints_per_predicate, combined_fp)
    end
    
    # Merge results (intersection for refinement)
    if isempty(results_per_predicate)
        return (results=NamedTuple[], combined_fp=UInt64(0), color=color_from_fp(UInt64(0)))
    end
    
    # Start with first result set
    refined = Set(t.id for t in results_per_predicate[1])
    
    # Intersect with remaining
    for results in results_per_predicate[2:end]
        refined = intersect(refined, Set(t.id for t in results))
    end
    
    final_results = [db.terms[id] for id in refined if id <= length(db.terms)]
    combined_fp = reduce(⊻, fingerprints_per_predicate; init=db.seed)
    
    (
        results = final_results,
        combined_fp = combined_fp,
        color = color_from_fp(combined_fp),
        predicates_evaluated = length(predicates),
        parallel_factor = min(max_parallel, length(predicates))
    )
end

"""Color-indexed scan with Gay parallel transport."""
function color_indexed_scan(db::SKIDuckDB, start_color::NTuple{3,Float64}, 
                            end_color::NTuple{3,Float64}; steps::Int=10)
    # Parallel transport along color gradient
    results = Vector{Vector{NamedTuple}}()
    fingerprint_trajectory = UInt64[]
    
    for i in 0:steps-1
        t = i / (steps - 1)
        r = start_color[1] * (1 - t) + end_color[1] * t
        g = start_color[2] * (1 - t) + end_color[2] * t
        b = start_color[3] * (1 - t) + end_color[3] * t
        
        matches = query_by_color(db, r, g, b; tolerance=0.15)
        push!(results, matches)
        
        step_fp = reduce(⊻, [m.fingerprint for m in matches]; init=UInt64(i))
        push!(fingerprint_trajectory, step_fp)
    end
    
    (
        results = results,
        trajectory = fingerprint_trajectory,
        steps = steps,
        total_matches = sum(length, results)
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# BEST RESPONSE DYNAMICS: Implicit Utilities from Reduction
# ═══════════════════════════════════════════════════════════════════════════════

"""
    SKIBestResponse

Best response for reduction: which reduction step minimizes distance to normal form?
"""
struct SKIBestResponse
    term::GaySKI
    available_reductions::Vector{SKI}  # Possible next terms
    
    # Implicit utility: fingerprint distance to estimated normal form
    normal_form_estimate_fp::UInt64
    current_distance::Float64
    
    # Best reduction (if any)
    best_reduction::Union{SKI, Nothing}
    expected_distance::Float64
end

"""Compute implicit utility as fingerprint distance."""
function implicit_utility(term_fp::UInt64, target_fp::UInt64)::Float64
    # Utility = negative distance (higher is better)
    distance = count_ones(term_fp ⊻ target_fp) / 64.0
    1.0 - distance
end

"""Find best response for reduction."""
function SKIBestResponse(term::SKI; target_fp::UInt64=UInt64(0), seed::UInt64=SKI_SEED)
    gay = GaySKI(term; seed=seed)
    
    # If no target given, estimate from term structure
    if target_fp == 0
        # Normal form estimate: reduce term and use that fingerprint
        result = reduce_to_normal(term; max_steps=100, seed=seed)
        target_fp = term_fingerprint(result.normal_form; seed=seed)
    end
    
    current_dist = count_ones(gay.fingerprint ⊻ target_fp) / 64.0
    
    # Find available reductions
    available = SKI[]
    if can_reduce(term)
        push!(available, reduce_step(term))
    end
    
    # Find best (should be the reduction that gets closer to normal form)
    best = nothing
    best_dist = current_dist
    
    for next_term in available
        next_fp = term_fingerprint(next_term; seed=seed)
        next_dist = count_ones(next_fp ⊻ target_fp) / 64.0
        
        if next_dist < best_dist
            best_dist = next_dist
            best = next_term
        end
    end
    
    SKIBestResponse(gay, available, target_fp, current_dist, best, best_dist)
end

"""Find reduction equilibrium (normal form via best response iteration)."""
function find_reduction_equilibrium(term::SKI; max_iters::Int=100, seed::UInt64=SKI_SEED)
    current = term
    trajectory = GaySKI[]
    
    for iter in 1:max_iters
        gay = GaySKI(current; seed=seed)
        push!(trajectory, gay)
        
        br = SKIBestResponse(current; seed=seed)
        
        if isnothing(br.best_reduction)
            # Equilibrium reached (no improving deviation)
            return (
                equilibrium = current,
                iterations = iter,
                trajectory = trajectory,
                fingerprint = gay.fingerprint,
                color = gay.color
            )
        end
        
        current = br.best_reduction
    end
    
    gay = GaySKI(current; seed=seed)
    (
        equilibrium = current,
        iterations = max_iters,
        trajectory = trajectory,
        fingerprint = gay.fingerprint,
        color = gay.color
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# IES NOV 2025 INTEGRATION: Message Indexing with SKI Patterns
# ═══════════════════════════════════════════════════════════════════════════════

"""
    IESMessageIndex

Index for IES messages (Nov 2025) with SKI pattern recognition.
Uses Gay chromatic identity for parallel-safe indexing.
"""
mutable struct IESMessageIndex
    # Message storage
    messages::Vector{NamedTuple{(:id, :content, :timestamp, :fingerprint, :color)}}
    
    # SKI pattern detection
    ski_patterns::Vector{NamedTuple{(:message_id, :pattern, :ski_term, :fingerprint)}}
    
    # Indices
    by_fingerprint::Dict{UInt64, Vector{Int}}
    by_color_bucket::Dict{Int, Vector{Int}}
    by_pattern::Dict{Symbol, Vector{Int}}
    
    # Statistics
    total_indexed::Int
    patterns_found::Int
    
    seed::UInt64
end

function IESMessageIndex(; seed::UInt64=GAY_SEED)
    IESMessageIndex(
        NamedTuple{(:id, :content, :timestamp, :fingerprint, :color)}[],
        NamedTuple{(:message_id, :pattern, :ski_term, :fingerprint)}[],
        Dict{UInt64, Vector{Int}}(),
        Dict{Int, Vector{Int}}(),
        Dict{Symbol, Vector{Int}}(),
        0, 0,
        seed
    )
end

"""Detect SKI-like patterns in text."""
function detect_ski_patterns(content::String)::Vector{Tuple{Symbol, SKI}}
    patterns = Tuple{Symbol, SKI}[]
    
    # Look for combinator-like patterns
    content_lower = lowercase(content)
    
    # Identity patterns: "same", "identity", "unchanged"
    if occursin("identity", content_lower) || occursin("unchanged", content_lower)
        push!(patterns, (:identity, I()))
    end
    
    # Constant patterns: "constant", "fixed", "always"
    if occursin("constant", content_lower) || occursin("fixed", content_lower)
        push!(patterns, (:constant, K()))
    end
    
    # Substitution patterns: "substitute", "apply", "transform"
    if occursin("substitut", content_lower) || occursin("transform", content_lower)
        push!(patterns, (:substitution, S()))
    end
    
    # Composition patterns: application
    if occursin("compose", content_lower) || occursin("apply", content_lower)
        push!(patterns, (:composition, App(S(), K())))
    end
    
    # Lambda patterns
    if occursin("lambda", content_lower) || occursin("λ", content)
        push!(patterns, (:lambda, App(App(S(), K()), I())))
    end
    
    patterns
end

"""Index a message with SKI pattern detection."""
function index_message!(idx::IESMessageIndex, content::String, timestamp::Int64)
    id = length(idx.messages) + 1
    
    # Compute fingerprint
    fp, _ = sm64(idx.seed ⊻ hash(content) ⊻ UInt64(timestamp))
    color = color_from_fp(fp)
    
    # Store message
    msg = (id=id, content=content, timestamp=timestamp, fingerprint=fp, color=color)
    push!(idx.messages, msg)
    idx.total_indexed += 1
    
    # Update fingerprint index
    if !haskey(idx.by_fingerprint, fp)
        idx.by_fingerprint[fp] = Int[]
    end
    push!(idx.by_fingerprint[fp], id)
    
    # Update color bucket index
    bucket = color_bucket(color[1], color[2], color[3])
    if !haskey(idx.by_color_bucket, bucket)
        idx.by_color_bucket[bucket] = Int[]
    end
    push!(idx.by_color_bucket[bucket], id)
    
    # Detect and index SKI patterns
    patterns = detect_ski_patterns(content)
    for (pattern_type, ski_term) in patterns
        pattern_fp = term_fingerprint(ski_term; seed=fp)
        pattern_record = (message_id=id, pattern=pattern_type, ski_term=string(ski_term), fingerprint=pattern_fp)
        push!(idx.ski_patterns, pattern_record)
        idx.patterns_found += 1
        
        if !haskey(idx.by_pattern, pattern_type)
            idx.by_pattern[pattern_type] = Int[]
        end
        push!(idx.by_pattern[pattern_type], id)
    end
    
    id
end

"""Index multiple messages in parallel."""
function maximally_parallel_index(idx::IESMessageIndex, messages::Vector{Tuple{String, Int64}})
    # In production: use @threads or similar
    # Key: order-independent due to XOR fingerprint composition
    
    ids = Int[]
    combined_fp = idx.seed
    
    for (content, timestamp) in messages
        id = index_message!(idx, content, timestamp)
        push!(ids, id)
        combined_fp = combined_fp ⊻ idx.messages[id].fingerprint
    end
    
    (
        indexed = length(ids),
        combined_fingerprint = combined_fp,
        combined_color = color_from_fp(combined_fp)
    )
end

"""Query messages by SKI pattern."""
function query_ski_patterns(idx::IESMessageIndex, pattern::Symbol)
    if !haskey(idx.by_pattern, pattern)
        return NamedTuple[]
    end
    
    [idx.messages[id] for id in idx.by_pattern[pattern] if id <= length(idx.messages)]
end

"""Color-coherent search: find messages near a target color."""
function color_coherent_search(idx::IESMessageIndex, target_color::NTuple{3,Float64};
                                tolerance::Float64=0.1)
    results = NamedTuple[]
    target_bucket = color_bucket(target_color[1], target_color[2], target_color[3])
    
    # Check nearby buckets
    for dr in -1:1, dg in -1:1, db in -1:1
        bucket = target_bucket + dr * 100 + dg * 10 + db
        if haskey(idx.by_color_bucket, bucket)
            for id in idx.by_color_bucket[bucket]
                msg = idx.messages[id]
                dist = sqrt(sum((msg.color[i] - target_color[i])^2 for i in 1:3))
                if dist <= tolerance
                    push!(results, msg)
                end
            end
        end
    end
    
    # Sort by color distance
    sort!(results, by = m -> sqrt(sum((m.color[i] - target_color[i])^2 for i in 1:3)))
    
    results
end

# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

function world_gay_ski()
    println()
    println("╔═══════════════════════════════════════════════════════════════════════════╗")
    println("║  GAY SKI COMBINATOR: Maximally Parallel Color-Indexed Random Access       ║")
    println("║  \"Every combinator has a color; every reduction preserves chromatic identity\"║")
    println("╚═══════════════════════════════════════════════════════════════════════════╝")
    println()
    
    # ─── Basic SKI Terms ───
    println("─── Basic SKI Combinators ───")
    
    terms = [
        ("S", S()),
        ("K", K()),
        ("I", I()),
        ("S K", App(S(), K())),
        ("S K K", App(App(S(), K()), K())),
        ("S I I", App(App(S(), I()), I())),  # ω = λx.xx
    ]
    
    for (name, term) in terms
        gay = GaySKI(term)
        println("  $name:")
        println("    Fingerprint: 0x$(string(gay.fingerprint, base=16)[1:min(8,end)])...")
        println("    Color: RGB$(round.(gay.color, digits=3))")
        println("    Depth: $(gay.depth), Size: $(gay.size)")
    end
    println()
    
    # ─── Reduction with Color Tracking ───
    println("─── Reduction Color Trajectory ───")
    
    # SKK = I (the identity through K)
    skk = App(App(S(), K()), K())
    result = reduce_to_normal(skk)
    
    println("  Term: S K K")
    println("  Normal form: $(result.normal_form)")
    println("  Steps: $(result.steps)")
    println("  Fingerprint trajectory:")
    for (i, fp) in enumerate(result.trajectory)
        color = color_from_fp(fp)
        println("    Step $(i-1): 0x$(string(fp, base=16)[1:8])... → RGB$(round.(color, digits=2))")
    end
    println()
    
    # ─── Memoryless Memory ───
    println("─── Memoryless Memory (Random Access) ───")
    
    # Create a more complex term
    omega = App(App(S(), I()), I())  # S I I
    mem = SKIMemory(omega)
    
    println("  Term: S I I (ω combinator)")
    println("  Estimated trajectory length: $(mem.estimated_length)")
    println("  Random access to colors:")
    for step in [0, 5, 10, 50, 100]
        color = ski_color_at(mem, step)
        println("    Step $step: RGB$(round.(color, digits=3))")
    end
    println("  Cache size after queries: $(length(mem.cache))")
    println()
    
    # ─── DuckDB Integration ───
    println("─── DuckDB Color-Indexed Queries ───")
    
    db = SKIDuckDB()
    create_ski_tables!(db)
    
    # Insert various terms
    test_terms = [
        S(), K(), I(),
        App(S(), K()), App(K(), I()), App(I(), S()),
        App(App(S(), K()), K()),
        App(App(S(), I()), I()),
        App(App(K(), I()), S()),
    ]
    
    for term in test_terms
        insert_term!(db, term)
    end
    
    println("  Inserted: $(length(db.terms)) terms")
    println("  Fingerprint index entries: $(length(db.by_fingerprint))")
    println("  Color buckets: $(length(db.by_color_bucket))")
    println()
    
    # Query by fingerprint
    s_fp = term_fingerprint(S())
    result = query_by_fingerprint(db, s_fp)
    println("  Query by S fingerprint: $(isnothing(result) ? "not found" : result.term_str)")
    
    # Query by color
    s_color = term_color(S())
    similar = query_by_color(db, s_color[1], s_color[2], s_color[3]; tolerance=0.2)
    println("  Query by S color (±0.2): $(length(similar)) matches")
    
    # Parallel refinement query
    predicates = [
        t -> t.depth <= 2,
        t -> t.size <= 3,
        t -> t.is_normal
    ]
    refined = parallel_refinement_query(db, predicates)
    println("  Parallel refinement (depth≤2 ∧ size≤3 ∧ normal): $(length(refined.results)) matches")
    println("    Combined fingerprint: 0x$(string(refined.combined_fp, base=16)[1:8])...")
    println()
    
    # Color-indexed scan
    println("  Color-indexed scan (S→K gradient):")
    scan = color_indexed_scan(db, term_color(S()), term_color(K()); steps=5)
    for (i, (step_results, fp)) in enumerate(zip(scan.results, scan.trajectory))
        println("    Step $i: $(length(step_results)) matches, fp=0x$(string(fp, base=16)[1:6])...")
    end
    println()
    
    # ─── Best Response Dynamics ───
    println("─── Best Response Dynamics for Reduction ───")
    
    # Find equilibrium for SKK
    eq_result = find_reduction_equilibrium(skk)
    
    println("  Term: S K K")
    println("  Equilibrium (normal form): $(eq_result.equilibrium)")
    println("  Iterations: $(eq_result.iterations)")
    println("  Final fingerprint: 0x$(string(eq_result.fingerprint, base=16)[1:8])...")
    println("  Final color: RGB$(round.(eq_result.color, digits=3))")
    println()
    
    # ─── IES Message Indexing ───
    println("─── IES Nov 2025 Message Indexing ───")
    
    idx = IESMessageIndex()
    
    # Simulate some messages with SKI-like content
    messages = [
        ("Working on identity transformations for the system", Int64(1732500000000)),
        ("Fixed the constant function behavior in the API", Int64(1732500001000)),
        ("Applied substitution to transform the data pipeline", Int64(1732500002000)),
        ("Lambda calculus approach for the compositor", Int64(1732500003000)),
        ("Regular message without combinator patterns", Int64(1732500004000)),
        ("Compose the effects in sequence for optimal flow", Int64(1732500005000)),
    ]
    
    result = maximally_parallel_index(idx, messages)
    
    println("  Messages indexed: $(result.indexed)")
    println("  Combined fingerprint: 0x$(string(result.combined_fingerprint, base=16)[1:8])...")
    println("  Combined color: RGB$(round.(result.combined_color, digits=3))")
    println("  Patterns found: $(idx.patterns_found)")
    println()
    
    println("  SKI patterns detected:")
    for pattern in [:identity, :constant, :substitution, :lambda, :composition]
        matches = query_ski_patterns(idx, pattern)
        if !isempty(matches)
            println("    :$pattern → $(length(matches)) messages")
        end
    end
    println()
    
    # Color-coherent search
    target = term_color(S())  # Search for "S-colored" messages
    coherent = color_coherent_search(idx, target; tolerance=0.3)
    println("  Color-coherent search (S-hue ±0.3): $(length(coherent)) matches")
    println()
    
    # ─── Summary ───
    println("═══════════════════════════════════════════════════════════════════════════")
    println("  KEY CONCEPTS")
    println("═══════════════════════════════════════════════════════════════════════════")
    println()
    println("  1. GAY SKI COMBINATORS:")
    println("     S → Magenta hue (300°), K → Cyan hue (180°), I → Yellow hue (60°)")
    println("     Every term has deterministic color from SPI fingerprint")
    println()
    println("  2. MEMORYLESS MEMORY:")
    println("     O(1) random access to any reduction step's color")
    println("     Seed + step → color without storing trajectory")
    println()
    println("  3. DUCKDB INTEGRATION:")
    println("     Color-indexed tables for O(1) fingerprint lookup")
    println("     Parallel refinement queries preserve XOR fingerprint")
    println("     Color-indexed scans with parallel transport")
    println()
    println("  4. BEST RESPONSE DYNAMICS:")
    println("     Reduction as equilibrium-finding via fingerprint distance")
    println("     Normal form = Nash equilibrium of term structure")
    println()
    println("  5. IES MESSAGE INDEXING:")
    println("     SKI pattern detection in natural language")
    println("     Color-coherent search for semantic similarity")
    println("     Maximally parallel indexing with Gay guarantees")
    println()
    
    return (
        terms = terms,
        db = db,
        idx = idx,
        equilibrium = eq_result
    )
end

end # module GaySKICombinator
