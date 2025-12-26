# Trajectory Database: Tracking Promising Gay Seeds
# Based on Zipf's law, entropy scoring, and multi-agent interactions
#
# Papers integrated:
#   - Zipf (1935): Rank-frequency distributions
#   - Gray (1990): Entropy and information theory
#   - Hailman (2008): Redundancy in animal signals
#   - Sharma (2024): Whale phonetic alphabet

using DuckDB
using Dates

export TrajectoryDB, SeedTrajectory, InteractionRecord
export record_seed!, record_interaction!, query_promising_seeds
export zipf_score, entropy_score, redundancy_score
export consensus_seed, multi_agent_coupling

# ═══════════════════════════════════════════════════════════════════════════
# Core Types
# ═══════════════════════════════════════════════════════════════════════════

struct SeedTrajectory
    seed::UInt64
    created_at::DateTime
    
    # Information-theoretic scores (Gray)
    entropy::Float64           # H(seed) = diversity of codas produced
    mutual_info::Float64       # I(seed; whale_observations)
    
    # Zipf analysis
    zipf_rank::Int             # Position in frequency distribution
    zipf_alpha::Float64        # Power law exponent
    
    # Redundancy measures (Hailman)
    surprisal::Float64         # -log₂ p(seed matches whale)
    serial_redundancy::Float64 # Markov predictability
    
    # Interaction history
    whale_matches::Vector{String}      # Whale IDs this seed matched
    human_annotations::Dict{String, Any}  # Human interpretations
    coupling_scores::Vector{Float64}   # Per-interaction coupling
end

struct InteractionRecord
    interaction_id::String
    timestamp::DateTime
    
    # Agents
    human_id::String
    whale_ids::Vector{String}
    
    # Seeds involved
    initial_seed::UInt64
    final_seed::UInt64
    candidate_seeds::Vector{UInt64}
    
    # Metrics
    coupling_achieved::Float64
    fixpoints_found::Int
    gadget_class::Symbol  # :XOR, :MAJ, :PARITY, :CLAUSE
    
    # Observed data
    observed_icis::Vector{Vector{Float64}}
    rhythm_patterns::Vector{Vector{Float64}}
end

# ═══════════════════════════════════════════════════════════════════════════
# Information-Theoretic Scoring
# ═══════════════════════════════════════════════════════════════════════════

"""
Compute entropy of coda distribution produced by a seed.
High entropy = seed produces diverse, unpredictable codas.
"""
function entropy_score(seed::UInt64; n_samples::Int=100)::Float64
    # Generate coda patterns from seed variations
    coda_counts = Dict{Vector{Int}, Int}()
    
    for i in 1:n_samples
        test_seed = seed + UInt64(i)
        notes = [hue_to_pc(color_at(j; seed=test_seed)) for j in 1:5]
        intervals = [(notes[j+1] - notes[j] + 12) % 12 for j in 1:4]
        coda_counts[intervals] = get(coda_counts, intervals, 0) + 1
    end
    
    # Shannon entropy
    total = sum(values(coda_counts))
    H = 0.0
    for count in values(coda_counts)
        p = count / total
        if p > 0
            H -= p * log2(p)
        end
    end
    
    H
end

"""
Compute mutual information between seed and observed whale pattern.
High MI = seed is informative about whale behavior.
"""
function mutual_info(seed::UInt64, whale_patterns::Vector{Vector{Float64}})::Float64
    if isempty(whale_patterns)
        return 0.0
    end
    
    # H(Whale) - H(Whale | Seed)
    # Approximate via pattern matching scores
    
    scores = Float64[]
    for pattern in whale_patterns
        rhythm = pattern ./ sum(pattern)
        
        # Score this seed against pattern
        notes = [hue_to_pc(color_at(j; seed=seed)) for j in 1:length(pattern)+1]
        intervals = [(notes[j+1] - notes[j] + 12) % 12 for j in 1:length(pattern)]
        seed_rhythm = intervals ./ max(1, sum(intervals))
        
        # Cosine similarity as proxy for conditional entropy reduction
        dot = sum(rhythm .* seed_rhythm)
        norm_a = sqrt(sum(rhythm .^ 2))
        norm_b = sqrt(sum(seed_rhythm .^ 2))
        score = (norm_a > 0 && norm_b > 0) ? dot / (norm_a * norm_b) : 0.0
        push!(scores, score)
    end
    
    # MI ≈ average score (higher = more information)
    mean(scores)
end

# ═══════════════════════════════════════════════════════════════════════════
# Zipf Analysis
# ═══════════════════════════════════════════════════════════════════════════

"""
Compute Zipf rank of a seed based on how common its coda pattern is.
Rank 1 = most common pattern, higher ranks = rarer patterns.
"""
function zipf_rank(seed::UInt64, corpus::Vector{Vector{Int}})::Int
    if isempty(corpus)
        return 1
    end
    
    # Generate pattern for this seed
    notes = [hue_to_pc(color_at(j; seed=seed)) for j in 1:5]
    pattern = [(notes[j+1] - notes[j] + 12) % 12 for j in 1:4]
    
    # Count frequency of each pattern in corpus
    counts = Dict{Vector{Int}, Int}()
    for p in corpus
        counts[p] = get(counts, p, 0) + 1
    end
    
    # Rank by frequency
    sorted = sort(collect(counts), by=x->-x[2])
    
    for (rank, (p, _)) in enumerate(sorted)
        if p == pattern
            return rank
        end
    end
    
    # Pattern not in corpus = very rare
    length(sorted) + 1
end

"""
Estimate Zipf exponent α from corpus.
Zipf's law: f(r) ∝ 1/r^α
"""
function zipf_alpha(corpus::Vector{Vector{Int}})::Float64
    if length(corpus) < 2
        return 1.0
    end
    
    # Count frequencies
    counts = Dict{Vector{Int}, Int}()
    for p in corpus
        counts[p] = get(counts, p, 0) + 1
    end
    
    # Sort by frequency (descending)
    freqs = sort(collect(values(counts)), rev=true)
    
    if length(freqs) < 2
        return 1.0
    end
    
    # Linear regression on log-log scale
    log_ranks = log.(1:length(freqs))
    log_freqs = log.(freqs)
    
    # α = -slope of log(f) vs log(r)
    n = length(freqs)
    sum_x = sum(log_ranks)
    sum_y = sum(log_freqs)
    sum_xy = sum(log_ranks .* log_freqs)
    sum_x2 = sum(log_ranks .^ 2)
    
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x^2)
    
    -slope  # Zipf α is negative of slope
end

"""
Score seed by Zipf distribution: prefer seeds that produce
patterns at the "sweet spot" (not too common, not too rare).
Based on principle of least effort (Zipf 1949).
"""
function zipf_score(seed::UInt64, corpus::Vector{Vector{Int}}; 
                    target_rank::Int=10)::Float64
    rank = zipf_rank(seed, corpus)
    
    # Gaussian preference centered on target_rank
    σ = 5.0
    exp(-((rank - target_rank)^2) / (2 * σ^2))
end

# ═══════════════════════════════════════════════════════════════════════════
# Redundancy Scoring (Hailman)
# ═══════════════════════════════════════════════════════════════════════════

"""
Surprisal: -log₂ p(seed matches whale pattern)
High surprisal = rare/informative match.
"""
function surprisal_score(seed::UInt64, whale_pattern::Vector{Float64})::Float64
    # Compute match probability
    rhythm = whale_pattern ./ sum(whale_pattern)
    
    notes = [hue_to_pc(color_at(j; seed=seed)) for j in 1:length(whale_pattern)+1]
    intervals = [(notes[j+1] - notes[j] + 12) % 12 for j in 1:length(whale_pattern)]
    seed_rhythm = intervals ./ max(1, sum(intervals))
    
    # Match probability (approximate)
    dot = sum(rhythm .* seed_rhythm)
    norm_a = sqrt(sum(rhythm .^ 2))
    norm_b = sqrt(sum(seed_rhythm .^ 2))
    p_match = (norm_a > 0 && norm_b > 0) ? dot / (norm_a * norm_b) : 0.001
    
    # Surprisal
    -log2(max(0.001, p_match))
end

"""
Serial redundancy: how predictable is the next coda given previous?
Based on Markov transition analysis.
"""
function serial_redundancy(seed::UInt64; n_steps::Int=10)::Float64
    # Generate sequence of codas from seed trajectory
    patterns = Vector{Vector{Int}}()
    
    current_seed = seed
    for _ in 1:n_steps
        notes = [hue_to_pc(color_at(j; seed=current_seed)) for j in 1:5]
        pattern = [(notes[j+1] - notes[j] + 12) % 12 for j in 1:4]
        push!(patterns, pattern)
        current_seed = splitmix64(current_seed)
    end
    
    if length(patterns) < 2
        return 0.0
    end
    
    # Count transitions
    transitions = Dict{Tuple{Vector{Int}, Vector{Int}}, Int}()
    for i in 1:length(patterns)-1
        key = (patterns[i], patterns[i+1])
        transitions[key] = get(transitions, key, 0) + 1
    end
    
    # Compute transition entropy
    total = sum(values(transitions))
    H_transition = 0.0
    for count in values(transitions)
        p = count / total
        if p > 0
            H_transition -= p * log2(p)
        end
    end
    
    # Redundancy = 1 - H/H_max
    H_max = log2(length(transitions))
    H_max > 0 ? 1.0 - H_transition / H_max : 0.0
end

# Helper
function splitmix64(x::UInt64)::UInt64
    x += 0x9e3779b97f4a7c15
    x = (x ⊻ (x >> 30)) * 0xbf58476d1ce4e5b9
    x = (x ⊻ (x >> 27)) * 0x94d049bb133111eb
    x ⊻ (x >> 31)
end

# ═══════════════════════════════════════════════════════════════════════════
# Multi-Agent Consensus
# ═══════════════════════════════════════════════════════════════════════════

"""
Find consensus seed across multiple humans and whales.
Returns seed that maximizes total coupling across all interactions.
"""
function consensus_seed(interactions::Vector{InteractionRecord}; 
                        n_candidates::Int=1000)::Tuple{UInt64, Float64}
    if isempty(interactions)
        return (GAY_SEED, 0.0)
    end
    
    # Collect all candidate seeds from interactions
    all_candidates = Set{UInt64}()
    for ir in interactions
        push!(all_candidates, ir.initial_seed)
        push!(all_candidates, ir.final_seed)
        for s in ir.candidate_seeds
            push!(all_candidates, s)
        end
    end
    
    # Add random seeds for diversity
    base_seed = first(interactions).initial_seed
    for i in 1:n_candidates
        push!(all_candidates, splitmix64(base_seed + UInt64(i)))
    end
    
    # Score each candidate by total coupling potential
    best_seed = GAY_SEED
    best_score = 0.0
    
    for seed in all_candidates
        total_coupling = 0.0
        
        for ir in interactions
            # How well does this seed explain this interaction?
            for pattern in ir.rhythm_patterns
                if !isempty(pattern)
                    # Compute match score
                    notes = [hue_to_pc(color_at(j; seed=seed)) for j in 1:length(pattern)+1]
                    intervals = [(notes[j+1] - notes[j] + 12) % 12 for j in 1:length(pattern)]
                    seed_rhythm = intervals ./ max(1, sum(intervals))
                    rhythm = pattern ./ max(0.001, sum(pattern))
                    
                    dot = sum(rhythm .* seed_rhythm)
                    norm_a = sqrt(sum(rhythm .^ 2))
                    norm_b = sqrt(sum(seed_rhythm .^ 2))
                    score = (norm_a > 0 && norm_b > 0) ? dot / (norm_a * norm_b) : 0.0
                    
                    total_coupling += score
                end
            end
        end
        
        if total_coupling > best_score
            best_score = total_coupling
            best_seed = seed
        end
    end
    
    (best_seed, best_score / max(1, length(interactions)))
end

"""
Compute coupling matrix for multi-agent interactions.
Returns matrix[human_idx, whale_idx] = coupling score.
"""
function multi_agent_coupling(interactions::Vector{InteractionRecord}, 
                              human_ids::Vector{String},
                              whale_ids::Vector{String})::Matrix{Float64}
    n_humans = length(human_ids)
    n_whales = length(whale_ids)
    
    coupling = zeros(n_humans, n_whales)
    counts = zeros(Int, n_humans, n_whales)
    
    for ir in interactions
        h_idx = findfirst(==(ir.human_id), human_ids)
        if h_idx === nothing
            continue
        end
        
        for w_id in ir.whale_ids
            w_idx = findfirst(==(w_id), whale_ids)
            if w_idx !== nothing
                coupling[h_idx, w_idx] += ir.coupling_achieved
                counts[h_idx, w_idx] += 1
            end
        end
    end
    
    # Average
    for i in 1:n_humans
        for j in 1:n_whales
            if counts[i, j] > 0
                coupling[i, j] /= counts[i, j]
            end
        end
    end
    
    coupling
end

# ═══════════════════════════════════════════════════════════════════════════
# DuckDB Persistence
# ═══════════════════════════════════════════════════════════════════════════

struct TrajectoryDB
    db_path::String
    conn::DuckDB.DB
end

function TrajectoryDB(path::String="$(homedir())/ies/trajectory.duckdb")
    conn = DuckDB.DB(path)
    
    # Initialize schema
    DuckDB.execute(conn, """
        CREATE TABLE IF NOT EXISTS seed_trajectories (
            seed UBIGINT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            entropy DOUBLE,
            mutual_info DOUBLE,
            zipf_rank INTEGER,
            zipf_alpha DOUBLE,
            surprisal DOUBLE,
            serial_redundancy DOUBLE,
            whale_matches TEXT,
            human_annotations TEXT,
            coupling_scores TEXT
        )
    """)
    
    DuckDB.execute(conn, """
        CREATE TABLE IF NOT EXISTS interactions (
            interaction_id TEXT PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            human_id TEXT,
            whale_ids TEXT,
            initial_seed UBIGINT,
            final_seed UBIGINT,
            candidate_seeds TEXT,
            coupling_achieved DOUBLE,
            fixpoints_found INTEGER,
            gadget_class TEXT,
            observed_icis TEXT,
            rhythm_patterns TEXT
        )
    """)
    
    DuckDB.execute(conn, """
        CREATE TABLE IF NOT EXISTS coda_corpus (
            id INTEGER PRIMARY KEY,
            pattern TEXT,
            frequency INTEGER DEFAULT 1,
            first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            whale_id TEXT,
            source TEXT
        )
    """)
    
    TrajectoryDB(path, conn)
end

function record_seed!(db::TrajectoryDB, traj::SeedTrajectory)
    DuckDB.execute(db.conn, """
        INSERT OR REPLACE INTO seed_trajectories 
        (seed, created_at, entropy, mutual_info, zipf_rank, zipf_alpha, 
         surprisal, serial_redundancy, whale_matches, human_annotations, coupling_scores)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        traj.seed,
        string(traj.created_at),
        traj.entropy,
        traj.mutual_info,
        traj.zipf_rank,
        traj.zipf_alpha,
        traj.surprisal,
        traj.serial_redundancy,
        join(traj.whale_matches, ","),
        string(traj.human_annotations),
        join(traj.coupling_scores, ",")
    ])
end

function record_interaction!(db::TrajectoryDB, ir::InteractionRecord)
    DuckDB.execute(db.conn, """
        INSERT OR REPLACE INTO interactions
        (interaction_id, timestamp, human_id, whale_ids, initial_seed, final_seed,
         candidate_seeds, coupling_achieved, fixpoints_found, gadget_class,
         observed_icis, rhythm_patterns)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        ir.interaction_id,
        string(ir.timestamp),
        ir.human_id,
        join(ir.whale_ids, ","),
        ir.initial_seed,
        ir.final_seed,
        join(string.(ir.candidate_seeds), ","),
        ir.coupling_achieved,
        ir.fixpoints_found,
        string(ir.gadget_class),
        join([join(icis, ";") for icis in ir.observed_icis], "|"),
        join([join(r, ";") for r in ir.rhythm_patterns], "|")
    ])
end

function query_promising_seeds(db::TrajectoryDB; 
                               min_coupling::Float64=0.5,
                               limit::Int=20)::Vector{NamedTuple}
    result = DuckDB.execute(db.conn, """
        SELECT seed, entropy, mutual_info, zipf_rank, 
               AVG(CAST(coupling AS DOUBLE)) as avg_coupling
        FROM seed_trajectories
        CROSS JOIN UNNEST(string_split(coupling_scores, ',')) AS t(coupling)
        WHERE coupling != ''
        GROUP BY seed, entropy, mutual_info, zipf_rank
        HAVING avg_coupling >= ?
        ORDER BY avg_coupling DESC, entropy DESC
        LIMIT ?
    """, [min_coupling, limit])
    
    [(seed=r.seed, entropy=r.entropy, mutual_info=r.mutual_info, 
      zipf_rank=r.zipf_rank, avg_coupling=r.avg_coupling) 
     for r in result]
end
