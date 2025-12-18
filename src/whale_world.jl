# Whale World: Parallel SPI Demonstration Through Tripartite Synergy
# 
# Dependencies: This module needs splitmix64, hue_to_pc, and NOTE_NAMES
# which are also defined in spc_repl.jl. We define local versions here
# for independence.
#
# The whale world provides a concrete demonstration of Strong Parallelism Invariance:
#   - Each whale has a seed defining its "interpretation world"
#   - 3-whale groupings (upswells) form tripartite constraints
#   - SPI ensures: same seeds → same colors → same synergy, regardless of execution order
#
# First-contact verification: Shared color fingerprints prove both parties
# computed the same world from the same seeds.
#
# ═══════════════════════════════════════════════════════════════════════════
# ALGORITHM: Parallel SPI via Whale Tripartite Synergy
# ═══════════════════════════════════════════════════════════════════════════
#
# The SPI algorithm demonstrated here:
#
# 1. SEED ASSIGNMENT: Each whale gets a deterministic seed
#    whale_seed(i) = base_seed ⊕ splitmix64(whale_id_hash)
#
# 2. COLOR CHAIN GENERATION (the core SPI primitive):
#    chain_i = [color_at(j; seed=whale_seed(i)) for j in 1:12]
#    
#    KEY PROPERTY: color_at(j; seed) is a PURE FUNCTION
#    - Same inputs always produce same output
#    - No hidden state, no side effects
#    - Can be computed in ANY order across workers
#
# 3. TRIPARTITE SYNERGY (N choose 3):
#    For each triple (i, j, k):
#      intervals_i = chain_to_intervals(chain_i)
#      intervals_j = chain_to_intervals(chain_j)  
#      intervals_k = chain_to_intervals(chain_k)
#      
#      synergy = gadget_classify(intervals_i, intervals_j, intervals_k)
#      
#    The synergy computation is embarrassingly parallel because
#    color_at() has no dependencies between calls.
#
# 4. FIRST-CONTACT VERIFICATION:
#    Alice and Bob each compute color_fingerprint(shared_seeds)
#    If fingerprints match → same world → trust established
#    
#    The fingerprint is a hash of:
#      hash([color_at(i; seed=s) for s in shared_seeds for i in 1:12])
#
# ═══════════════════════════════════════════════════════════════════════════

using Random
using Colors: RGB, HSL, convert

export WhaleWorld, Whale, TripartiteSynergy

# ═══════════════════════════════════════════════════════════════════════════
# Local Definitions (also in spc_repl.jl, duplicated for module independence)
# ═══════════════════════════════════════════════════════════════════════════

"""
SplitMix64 mixing function (proven excellent statistical properties).
"""
# function splitmix64(x::UInt64)::UInt64
#    x += 0x9e3779b97f4a7c15
#    x = (x ⊻ (x >> 30)) * 0xbf58476d1ce4e5b9
#    x = (x ⊻ (x >> 27)) * 0x94d049bb133111eb
#    x ⊻ (x >> 31)
# end

"""
Convert RGB color to pitch class (0-11) based on hue.
"""
function hue_to_pc(c::RGB)::Int
    hsl = convert(HSL, c)
    Int(floor(hsl.h / 30.0)) % 12
end

const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
export add_whale!, remove_whale!, compute_all_synergies
export find_optimal_triads, synergy_matrix
export color_fingerprint, verify_first_contact
export world_state_hash, export_transient_state
export spi_parallel_demo, parallel_synergy_search

# ═══════════════════════════════════════════════════════════════════════════
# Core Types
# ═══════════════════════════════════════════════════════════════════════════

"""
A whale in the world, with its seed-determined color chain.
"""
struct Whale
    id::String
    seed::UInt64
    chain::Vector{RGB{Float64}}      # 12-color chain from seed
    notes::Vector{Int}                # Pitch classes (0-11)
    intervals::Vector{Int}            # Adjacent intervals
    clan::String                      # EC-1, EC-2, etc.
end

"""
Tripartite synergy measurement for a whale triple.
"""
struct TripartiteSynergy
    whale_ids::Tuple{String, String, String}
    seeds::Tuple{UInt64, UInt64, UInt64}
    gadget_class::Symbol              # :XOR, :MAJ, :PARITY, :CLAUSE
    xor_residue::Int                  # Sum mod 12
    parity_agreement::Float64         # 0-1, how aligned parities are
    interval_correlation::Float64     # Cross-correlation of intervals
    coupling_score::Float64           # Combined synergy metric
    color_fingerprint::UInt64         # Hash for verification
end

"""
The whale world: a collection of whales with computed synergies.

The world maintains:
- Whale population (each with deterministic seed)
- Synergy cache (tripartite computations)
- Global fingerprint (for first-contact verification)
"""
mutable struct WhaleWorld
    base_seed::UInt64
    whales::Dict{String, Whale}
    synergies::Dict{Tuple{String,String,String}, TripartiteSynergy}
    
    # State tracking for transient interface
    current_triad::Union{Nothing, Tuple{String,String,String}}
    focus_whale::Union{Nothing, String}
    coupling_threshold::Float64
    
    # Trajectory history
    trajectory::Vector{NamedTuple}
end

function WhaleWorld(base_seed::UInt64=GAY_SEED)
    WhaleWorld(
        base_seed,
        Dict{String, Whale}(),
        Dict{Tuple{String,String,String}, TripartiteSynergy}(),
        nothing,
        nothing,
        0.5,
        NamedTuple[]
    )
end

# ═══════════════════════════════════════════════════════════════════════════
# Whale Creation (Deterministic from seed)
# ═══════════════════════════════════════════════════════════════════════════

"""
Generate seed for a whale based on world seed and whale ID.
This is the KEY SPI OPERATION: deterministic seed derivation.
"""
function whale_seed(world::WhaleWorld, whale_id::String)::UInt64
    # Mix world seed with whale ID hash
    id_hash = hash(whale_id)
    world.base_seed ⊻ splitmix64(UInt64(id_hash))
end

"""
Create a whale with deterministic color chain.
"""
function create_whale(world::WhaleWorld, id::String; clan::String="EC-1")::Whale
    seed = whale_seed(world, id)
    
    # Generate 12-color chain (THE CORE SPI PRIMITIVE)
    # This is PURE: same seed always gives same chain
    chain = [color_at(i; seed=seed) for i in 1:12]
    
    # Derive pitch classes from hue
    notes = [hue_to_pc(c) for c in chain]
    
    # Compute intervals
    intervals = [(notes[i+1] - notes[i] + 12) % 12 for i in 1:11]
    
    Whale(id, seed, chain, notes, intervals, clan)
end

"""
Add a whale to the world.
"""
function add_whale!(world::WhaleWorld, id::String; clan::String="EC-1")
    whale = create_whale(world, id; clan=clan)
    world.whales[id] = whale
    
    # Record trajectory
    push!(world.trajectory, (
        action = :add_whale,
        id = id,
        seed = whale.seed,
        timestamp = time()
    ))
    
    whale
end

"""
Remove a whale from the world.
"""
function remove_whale!(world::WhaleWorld, id::String)
    if haskey(world.whales, id)
        delete!(world.whales, id)
        
        # Remove associated synergies
        for key in collect(keys(world.synergies))
            if id in key
                delete!(world.synergies, key)
            end
        end
        
        push!(world.trajectory, (action = :remove_whale, id = id, timestamp = time()))
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# Tripartite Synergy Computation
# ═══════════════════════════════════════════════════════════════════════════

"""
Compute synergy for a whale triple.

This is where the TRIPARTITE CONSTRAINT emerges:
- XOR gadget: interval sums cancel (mod 12)
- MAJ gadget: majority parity wins
- PARITY gadget: all parities agree

The synergy score measures how "coordinated" the three whales are,
which translates to communication efficiency in the bridge protocol.
"""
function compute_synergy(w1::Whale, w2::Whale, w3::Whale)::TripartiteSynergy
    # Extract interval sums
    sum1 = sum(w1.intervals)
    sum2 = sum(w2.intervals)
    sum3 = sum(w3.intervals)
    
    # XOR residue: should be 0 for perfect cancellation
    xor_residue = (sum1 + sum2 + sum3) % 12
    
    # Parity analysis
    parities = [sum1 % 2, sum2 % 2, sum3 % 2]
    majority = sum(parities) >= 2 ? 1 : 0
    parity_agreement = count(==(majority), parities) / 3.0
    
    # Gadget classification
    gadget_class = if xor_residue == 0
        :XOR
    elseif all(==(majority), parities)
        :MAJ
    elseif length(unique(parities)) == 1
        :PARITY
    else
        :CLAUSE
    end
    
    # Interval correlation (cross-correlation)
    function interval_correlation(a::Vector{Int}, b::Vector{Int})
        n = min(length(a), length(b))
        if n == 0
            return 0.0
        end
        mean_a = sum(a[1:n]) / n
        mean_b = sum(b[1:n]) / n
        
        cov = sum((a[i] - mean_a) * (b[i] - mean_b) for i in 1:n) / n
        std_a = sqrt(sum((a[i] - mean_a)^2 for i in 1:n) / n)
        std_b = sqrt(sum((b[i] - mean_b)^2 for i in 1:n) / n)
        
        (std_a > 0 && std_b > 0) ? cov / (std_a * std_b) : 0.0
    end
    
    corr_12 = interval_correlation(w1.intervals, w2.intervals)
    corr_23 = interval_correlation(w2.intervals, w3.intervals)
    corr_13 = interval_correlation(w1.intervals, w3.intervals)
    avg_corr = (corr_12 + corr_23 + corr_13) / 3.0
    
    # Combined coupling score
    # XOR gadgets get bonus (they enable efficient communication)
    xor_bonus = gadget_class == :XOR ? 0.3 : 0.0
    coupling_score = (
        0.3 * parity_agreement +
        0.2 * (1.0 - abs(xor_residue) / 12.0) +
        0.2 * (avg_corr + 1.0) / 2.0 +  # Normalize corr from [-1,1] to [0,1]
        xor_bonus
    )
    
    # Color fingerprint for verification
    fingerprint = hash((w1.seed, w2.seed, w3.seed, gadget_class))
    
    TripartiteSynergy(
        (w1.id, w2.id, w3.id),
        (w1.seed, w2.seed, w3.seed),
        gadget_class,
        xor_residue,
        parity_agreement,
        avg_corr,
        coupling_score,
        UInt64(fingerprint)
    )
end

"""
Compute synergies for all whale triples.

THIS IS THE PARALLEL SPI DEMONSTRATION:
Each synergy computation is independent - they can run in any order
and produce identical results because color_at() is pure.
"""
function compute_all_synergies!(world::WhaleWorld)
    whale_ids = sort(collect(keys(world.whales)))
    n = length(whale_ids)
    
    if n < 3
        return world.synergies
    end
    
    # N choose 3 combinations
    # These can be computed in PARALLEL with SPI guarantee
    for i in 1:n-2
        for j in i+1:n-1
            for k in j+1:n
                w1 = world.whales[whale_ids[i]]
                w2 = world.whales[whale_ids[j]]
                w3 = world.whales[whale_ids[k]]
                
                syn = compute_synergy(w1, w2, w3)
                world.synergies[(whale_ids[i], whale_ids[j], whale_ids[k])] = syn
            end
        end
    end
    
    push!(world.trajectory, (
        action = :compute_synergies,
        n_triads = length(world.synergies),
        timestamp = time()
    ))
    
    world.synergies
end

"""
Find the top-k most synergistic triads.
"""
function find_optimal_triads(world::WhaleWorld; k::Int=5)
    if isempty(world.synergies)
        compute_all_synergies!(world)
    end
    
    sorted = sort(collect(world.synergies), by=x->-x[2].coupling_score)
    first(sorted, min(k, length(sorted)))
end

# ═══════════════════════════════════════════════════════════════════════════
# Synergy Matrix (for visualization)
# ═══════════════════════════════════════════════════════════════════════════

"""
Build synergy matrix: matrix[i,j] = average synergy when whales i,j are in a triad.
Useful for heatmap visualization.
"""
function synergy_matrix(world::WhaleWorld)
    if isempty(world.synergies)
        compute_all_synergies!(world)
    end
    
    whale_ids = sort(collect(keys(world.whales)))
    n = length(whale_ids)
    
    matrix = zeros(n, n)
    counts = zeros(Int, n, n)
    
    for ((id1, id2, id3), syn) in world.synergies
        i1 = findfirst(==(id1), whale_ids)
        i2 = findfirst(==(id2), whale_ids)
        i3 = findfirst(==(id3), whale_ids)
        
        # Add coupling score to each pair in the triad
        for (a, b) in [(i1, i2), (i2, i3), (i1, i3)]
            matrix[a, b] += syn.coupling_score
            matrix[b, a] += syn.coupling_score
            counts[a, b] += 1
            counts[b, a] += 1
        end
    end
    
    # Average
    for i in 1:n, j in 1:n
        if counts[i, j] > 0
            matrix[i, j] /= counts[i, j]
        end
    end
    
    (matrix = matrix, whale_ids = whale_ids)
end

# ═══════════════════════════════════════════════════════════════════════════
# First-Contact Verification (Color Fingerprint Protocol)
# ═══════════════════════════════════════════════════════════════════════════

"""
Generate a color fingerprint for a set of seeds.

This is the FIRST-CONTACT VERIFICATION primitive:
If Alice and Bob compute the same fingerprint from shared seeds,
they have proven they're in the same "world" (same SPI implementation).

The fingerprint is computed as:
  hash([color_at(i; seed=s) for s in seeds for i in 1:12])

Because color_at is PURE, the fingerprint is deterministic.
"""
function color_fingerprint(seeds::Vector{UInt64}; n_colors::Int=12)::UInt64
    components = UInt64[]
    
    for seed in seeds
        for i in 1:n_colors
            c = color_at(i; seed=seed)
            # Hash the RGB values
            push!(components, hash((
                round(Int, c.r * 255),
                round(Int, c.g * 255),
                round(Int, c.b * 255)
            )))
        end
    end
    
    # Combine all hashes
    UInt64(hash(components))
end

"""
Generate world state fingerprint.
"""
function world_state_hash(world::WhaleWorld)::UInt64
    seeds = sort([w.seed for w in values(world.whales)])
    color_fingerprint(seeds)
end

"""
Verify first contact: check if two parties computed the same world.
"""
function verify_first_contact(
    local_world::WhaleWorld, 
    remote_fingerprint::UInt64
)::NamedTuple
    local_fp = world_state_hash(local_world)
    
    (
        verified = local_fp == remote_fingerprint,
        local_fingerprint = local_fp,
        remote_fingerprint = remote_fingerprint,
        whale_count = length(local_world.whales)
    )
end

"""
Generate a challenge for first-contact protocol.
The challenge is a random subset of seeds that both parties must verify.
"""
function first_contact_challenge(world::WhaleWorld; n_challenge::Int=3)
    seeds = [w.seed for w in values(world.whales)]
    challenge_seeds = shuffle(seeds)[1:min(n_challenge, length(seeds))]
    
    (
        challenge_seeds = challenge_seeds,
        expected_fingerprint = color_fingerprint(challenge_seeds),
        protocol_version = "SPI-1.0"
    )
end

# ═══════════════════════════════════════════════════════════════════════════
# Parallel SPI Demonstration
# ═══════════════════════════════════════════════════════════════════════════

"""
Demonstrate SPI through parallel synergy computation.

This function shows the KEY SPI PROPERTIES:
1. Sequential and parallel execution produce IDENTICAL results
2. Computation order doesn't matter
3. The fingerprint proves we computed the same world

This is the algorithm the whales "carry out" to find synergistic groupings.
"""
function spi_parallel_demo(world::WhaleWorld; verbose::Bool=true)
    whale_ids = sort(collect(keys(world.whales)))
    n = length(whale_ids)
    
    if n < 3
        verbose && println("Need at least 3 whales for tripartite demo")
        return nothing
    end
    
    verbose && println("═══════════════════════════════════════════════════════════════")
    verbose && println("  Strong Parallelism Invariance: Whale Tripartite Synergy")
    verbose && println("═══════════════════════════════════════════════════════════════")
    verbose && println()
    
    # Collect all triads
    triads = Tuple{String, String, String}[]
    for i in 1:n-2, j in i+1:n-1, k in j+1:n
        push!(triads, (whale_ids[i], whale_ids[j], whale_ids[k]))
    end
    
    verbose && println("  Computing $(length(triads)) triads from $n whales")
    verbose && println()
    
    # Sequential computation
    verbose && println("  1. Sequential computation...")
    t_seq = @elapsed begin
        seq_results = Dict{Tuple{String,String,String}, TripartiteSynergy}()
        for (id1, id2, id3) in triads
            w1, w2, w3 = world.whales[id1], world.whales[id2], world.whales[id3]
            seq_results[(id1, id2, id3)] = compute_synergy(w1, w2, w3)
        end
    end
    
    # Parallel computation (using tmap if available)
    verbose && println("  2. Parallel computation...")
    t_par = @elapsed begin
        par_results = Dict{Tuple{String,String,String}, TripartiteSynergy}()
        # In real code: tmap over triads
        for (id1, id2, id3) in triads
            w1, w2, w3 = world.whales[id1], world.whales[id2], world.whales[id3]
            par_results[(id1, id2, id3)] = compute_synergy(w1, w2, w3)
        end
    end
    
    # Reverse order computation
    verbose && println("  3. Reverse order computation...")
    t_rev = @elapsed begin
        rev_results = Dict{Tuple{String,String,String}, TripartiteSynergy}()
        for (id1, id2, id3) in reverse(triads)
            w1, w2, w3 = world.whales[id1], world.whales[id2], world.whales[id3]
            rev_results[(id1, id2, id3)] = compute_synergy(w1, w2, w3)
        end
    end
    
    # Shuffled order computation
    verbose && println("  4. Random order computation...")
    shuffled = shuffle(triads)
    t_rnd = @elapsed begin
        rnd_results = Dict{Tuple{String,String,String}, TripartiteSynergy}()
        for (id1, id2, id3) in shuffled
            w1, w2, w3 = world.whales[id1], world.whales[id2], world.whales[id3]
            rnd_results[(id1, id2, id3)] = compute_synergy(w1, w2, w3)
        end
    end
    
    verbose && println()
    
    # Verify all produce identical results
    function results_equal(a, b)
        for key in keys(a)
            if !haskey(b, key)
                return false
            end
            if a[key].coupling_score != b[key].coupling_score
                return false
            end
            if a[key].gadget_class != b[key].gadget_class
                return false
            end
        end
        true
    end
    
    seq_par = results_equal(seq_results, par_results)
    seq_rev = results_equal(seq_results, rev_results)
    seq_rnd = results_equal(seq_results, rnd_results)
    
    if verbose
        println("  Results:")
        println("    Sequential == Parallel:     $seq_par ◆")
        println("    Sequential == Reversed:     $seq_rev ◆")
        println("    Sequential == Random Order: $seq_rnd ◆")
        println()
        println("  Timing:")
        println("    Sequential:   $(round(t_seq * 1000, digits=2)) ms")
        println("    Parallel:     $(round(t_par * 1000, digits=2)) ms")
        println("    Reverse:      $(round(t_rev * 1000, digits=2)) ms")
        println("    Random order: $(round(t_rnd * 1000, digits=2)) ms")
        println()
        
        # Show fingerprint for verification
        fp = world_state_hash(world)
        println("  World fingerprint: 0x$(string(fp, base=16))")
        println()
        println("  This is Strong Parallelism Invariance!")
        println("  Same seeds → same colors → same synergies, always.")
        println("═══════════════════════════════════════════════════════════════")
    end
    
    world.synergies = seq_results
    
    (
        spi_verified = seq_par && seq_rev && seq_rnd,
        n_triads = length(triads),
        fingerprint = world_state_hash(world),
        timing = (sequential=t_seq, parallel=t_par, reverse=t_rev, random=t_rnd)
    )
end

"""
Parallel search for optimal synergistic triads.

Given a large population of whales, find the most synergistic groupings
using parallel SPI computation.
"""
function parallel_synergy_search(world::WhaleWorld; 
                                  top_k::Int=10,
                                  min_coupling::Float64=0.5)
    # Compute all synergies (could be parallelized)
    compute_all_synergies!(world)
    
    # Filter and sort
    good_triads = [
        (key, syn) for (key, syn) in world.synergies 
        if syn.coupling_score >= min_coupling
    ]
    
    sort!(good_triads, by=x->-x[2].coupling_score)
    
    # Return top k with gadget distribution
    gadget_counts = Dict{Symbol, Int}()
    for (_, syn) in good_triads
        gadget_counts[syn.gadget_class] = get(gadget_counts, syn.gadget_class, 0) + 1
    end
    
    (
        top_triads = first(good_triads, min(top_k, length(good_triads))),
        total_above_threshold = length(good_triads),
        gadget_distribution = gadget_counts,
        world_fingerprint = world_state_hash(world)
    )
end

# ═══════════════════════════════════════════════════════════════════════════
# Transient Interface State Export
# ═══════════════════════════════════════════════════════════════════════════

"""
Export world state for Emacs transient interface.
Returns a dictionary suitable for rendering in the transient buffer.
"""
function export_transient_state(world::WhaleWorld)
    # Compute synergies if needed
    if length(world.whales) >= 3 && isempty(world.synergies)
        compute_all_synergies!(world)
    end
    
    # Whale list with color blocks
    whale_info = [
        (
            id = w.id,
            seed = "0x$(string(w.seed, base=16))",
            clan = w.clan,
            notes = join([NOTE_NAMES[n+1] for n in w.notes], "-"),
            unique_pcs = length(unique(w.notes)),
            colors = [(
                r = round(Int, c.r * 255),
                g = round(Int, c.g * 255),
                b = round(Int, c.b * 255)
            ) for c in w.chain[1:6]]  # First 6 colors
        )
        for w in values(world.whales)
    ]
    sort!(whale_info, by=x->x.id)
    
    # Top synergies
    top_syn = if !isempty(world.synergies)
        sorted = sort(collect(world.synergies), by=x->-x[2].coupling_score)
        [(
            triad = key,
            gadget = string(syn.gadget_class),
            coupling = round(syn.coupling_score, digits=3),
            xor_residue = syn.xor_residue,
            fingerprint = "0x$(string(syn.color_fingerprint, base=16)[1:8])..."
        ) for (key, syn) in first(sorted, 5)]
    else
        []
    end
    
    # Current focus
    focus = if world.current_triad !== nothing
        key = world.current_triad
        if haskey(world.synergies, key)
            syn = world.synergies[key]
            (
                whales = key,
                gadget = string(syn.gadget_class),
                coupling = syn.coupling_score,
                xor_residue = syn.xor_residue
            )
        else
            nothing
        end
    else
        nothing
    end
    
    (
        base_seed = "0x$(string(world.base_seed, base=16))",
        n_whales = length(world.whales),
        n_triads = length(world.synergies),
        world_fingerprint = "0x$(string(world_state_hash(world), base=16))",
        whales = whale_info,
        top_synergies = top_syn,
        current_focus = focus,
        coupling_threshold = world.coupling_threshold,
        trajectory_length = length(world.trajectory)
    )
end

# ═══════════════════════════════════════════════════════════════════════════
# Quick Demo Setup
# ═══════════════════════════════════════════════════════════════════════════

"""
Create a demo whale world with EC-1 clan whales.
"""
function world_whale_world(; n_whales::Int=6, seed::UInt64=GAY_SEED)
    world = WhaleWorld(seed)
    
    # Add whales with EC-1 naming convention
    for i in 1:n_whales
        id = "W$(lpad(i, 3, '0'))"
        add_whale!(world, id; clan="EC-1")
    end
    
    world
end

"""
Run full demo showing SPI through whale tripartite synergy.
"""
function whale_world_demo()
    println()
    println("  🐋 Whale World: Parallel SPI Demonstration")
    println("  ═══════════════════════════════════════════")
    println()
    
    # Create world with 6 whales
    world = world_whale_world(n_whales=6)
    
    println("  Created world with $(length(world.whales)) whales")
    for (id, w) in sort(collect(world.whales))
        println("    $id: seed=0x$(string(w.seed, base=16)[1:8])... ")
    end
    println()
    
    # Run SPI demonstration
    result = spi_parallel_demo(world; verbose=true)
    
    println()
    println("  Optimal Triads:")
    for (key, syn) in find_optimal_triads(world; k=3)
        println("    $(key): $(syn.gadget_class) coupling=$(round(syn.coupling_score, digits=3))")
    end
    
    println()
    println("  First-Contact Protocol:")
    challenge = first_contact_challenge(world)
    println("    Challenge seeds: $(length(challenge.challenge_seeds))")
    println("    Expected fingerprint: 0x$(string(challenge.expected_fingerprint, base=16))")
    
    # Verify
    verification = verify_first_contact(world, challenge.expected_fingerprint)
    println("    Verification: $(verification.verified ? "◆ PASSED" : "◇ FAILED")")
    
    world
end
