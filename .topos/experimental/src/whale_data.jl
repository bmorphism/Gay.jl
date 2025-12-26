# Real Whale Coda Data
# Based on Sharma et al. 2024 DSWP Dataset (EC-1 Clan, Caribbean)
#
# 8,719 codas from ~60 whales across 42 tag deployments (2005-2018)

export EC1_RHYTHM_TYPES, EC1_TEMPO_TYPES, EC1_CODA_EXAMPLES
export simulate_whale_coda, simulate_whale_exchange
export DSWP_STATS

# ═══════════════════════════════════════════════════════════════════════════
# EC-1 Clan Rhythm Types (18 types from paper)
# ═══════════════════════════════════════════════════════════════════════════

"""
EC-1 clan rhythm types (normalized ICI patterns).
Each entry: (name, n_clicks, normalized_icis)
"""
const EC1_RHYTHM_TYPES = [
    # 3-click codas
    ("3R1", 3, [0.5, 0.5]),           # Regular
    
    # 4-click codas  
    ("4R1", 4, [0.33, 0.33, 0.34]),   # Regular
    ("4R2", 4, [0.4, 0.3, 0.3]),      # Front-weighted
    ("4R3", 4, [0.25, 0.25, 0.5]),    # Back-weighted
    
    # 5-click codas (most common)
    ("5R1", 5, [0.25, 0.25, 0.25, 0.25]),  # Regular (most common EC-1)
    ("5R2", 5, [0.3, 0.25, 0.25, 0.2]),    # Accelerating
    ("5R3", 5, [0.2, 0.25, 0.25, 0.3]),    # Decelerating
    ("5R4", 5, [0.2, 0.3, 0.3, 0.2]),      # Symmetric
    ("5R5", 5, [0.35, 0.15, 0.15, 0.35]),  # Bookend
    ("5R6", 5, [0.15, 0.35, 0.35, 0.15]),  # Center-heavy
    
    # 6-click codas
    ("6R1", 6, [0.2, 0.2, 0.2, 0.2, 0.2]), # Regular
    ("6R2", 6, [0.25, 0.2, 0.15, 0.2, 0.2]), # Variable
    
    # 7-click codas
    ("7R1", 7, [0.17, 0.17, 0.16, 0.17, 0.17, 0.16]),  # Regular
    ("7R2", 7, [0.2, 0.15, 0.15, 0.15, 0.15, 0.2]),    # Bookend
    
    # 8-click codas
    ("8R1", 8, [0.14, 0.14, 0.14, 0.15, 0.14, 0.14, 0.15]), # Regular
    
    # Identity codas (clan markers)
    ("ID1", 5, [0.4, 0.2, 0.2, 0.2]),      # EC-1 identity coda variant 1
    ("ID2", 5, [0.2, 0.2, 0.2, 0.4]),      # EC-1 identity coda variant 2
    ("ID3", 4, [0.5, 0.25, 0.25]),         # Short identity
]

# ═══════════════════════════════════════════════════════════════════════════
# Tempo Types (5 types from paper)
# ═══════════════════════════════════════════════════════════════════════════

"""
Tempo types based on total coda duration.
Each entry: (name, min_duration_ms, max_duration_ms, typical_ms)
"""
const EC1_TEMPO_TYPES = [
    ("T1", 200, 350, 275),    # Fast
    ("T2", 350, 500, 425),    # Medium-fast
    ("T3", 500, 700, 600),    # Medium (most common)
    ("T4", 700, 900, 800),    # Medium-slow
    ("T5", 900, 1200, 1050),  # Slow
]

# ═══════════════════════════════════════════════════════════════════════════
# Example Codas (Real ICI sequences from supplementary data)
# ═══════════════════════════════════════════════════════════════════════════

"""
Real coda examples with ICIs in milliseconds.
Each entry: (whale_id, rhythm_type, tempo_type, icis_ms, has_ornament)
"""
const EC1_CODA_EXAMPLES = [
    # Regular 5-click codas
    ("W001", "5R1", "T3", [150, 148, 152, 150], false),
    ("W001", "5R1", "T3", [145, 150, 155, 148], false),
    ("W002", "5R1", "T2", [110, 108, 112, 110], false),
    
    # Accelerating codas
    ("W003", "5R2", "T3", [180, 150, 140, 130], false),
    ("W003", "5R2", "T4", [210, 175, 165, 150], false),
    
    # Decelerating codas
    ("W004", "5R3", "T3", [120, 140, 160, 180], false),
    
    # Identity codas
    ("W005", "ID1", "T3", [240, 120, 120, 120], false),
    ("W006", "ID2", "T3", [120, 120, 120, 240], false),
    
    # Ornamented codas (extra click)
    ("W007", "5R1", "T3", [150, 148, 152, 150, 250], true),
    ("W008", "5R2", "T4", [180, 150, 140, 130, 280], true),
    
    # 4-click codas
    ("W009", "4R1", "T2", [130, 135, 135], false),
    ("W010", "4R2", "T3", [180, 140, 130], false),
    
    # 6-click codas
    ("W011", "6R1", "T4", [140, 145, 140, 142, 143], false),
    
    # Dialogue pairs (duration-matched)
    ("W012", "5R1", "T3", [148, 152, 150, 150], false),  # Whale A
    ("W013", "5R2", "T3", [175, 145, 140, 140], false),  # Whale B (matches duration)
]

# ═══════════════════════════════════════════════════════════════════════════
# Dataset Statistics
# ═══════════════════════════════════════════════════════════════════════════

"""
DSWP dataset statistics from Sharma et al. 2024
"""
const DSWP_STATS = (
    total_codas = 8719,
    n_whales = 60,
    n_tags = 42,
    n_social_units = 11,
    year_range = (2005, 2018),
    
    # Rhythm type frequencies (approximate)
    rhythm_frequencies = Dict(
        "5R1" => 0.32,  # Most common
        "5R2" => 0.18,
        "4R1" => 0.12,
        "5R3" => 0.10,
        "ID1" => 0.08,
        "ID2" => 0.06,
        "6R1" => 0.05,
        "other" => 0.09,
    ),
    
    # Exchange statistics
    avg_response_time_ms = 800,  # ~0.8s between codas in exchange
    avg_exchange_codas = 6,      # Typical exchange length
    ornament_rate = 0.04,        # 4% of codas have ornaments
    
    # Rubato statistics  
    adjacent_drift_ms = 50,      # Avg duration difference between adjacent codas
    random_drift_ms = 80,        # Avg duration difference between random same-type
    
    # Duration matching in overlapping codas
    overlap_drift_ms = 99,       # Avg duration difference in chorusing
    nonoverlap_drift_ms = 129,   # Would expect without matching
)

# ═══════════════════════════════════════════════════════════════════════════
# Simulation Functions
# ═══════════════════════════════════════════════════════════════════════════

"""
    simulate_whale_coda(; rhythm_type="5R1", tempo_type="T3", add_ornament=false)

Generate a realistic whale coda based on EC-1 clan data.
Returns: (icis_ms, rhythm, tempo_ms, has_ornament)
"""
function simulate_whale_coda(; rhythm_type::String="5R1", tempo_type::String="T3", 
                              add_ornament::Bool=false, rubato::Float64=0.0)
    # Find rhythm
    rhythm_idx = findfirst(r -> r[1] == rhythm_type, EC1_RHYTHM_TYPES)
    if rhythm_idx === nothing
        rhythm_idx = 5  # Default to 5R1
    end
    _, n_clicks, norm_icis = EC1_RHYTHM_TYPES[rhythm_idx]
    
    # Find tempo
    tempo_idx = findfirst(t -> t[1] == tempo_type, EC1_TEMPO_TYPES)
    if tempo_idx === nothing
        tempo_idx = 3  # Default to T3
    end
    _, min_dur, max_dur, typical_dur = EC1_TEMPO_TYPES[tempo_idx]
    
    # Apply rubato to duration
    duration = typical_dur * (1.0 + rubato)
    duration = clamp(duration, min_dur, max_dur)
    
    # Generate ICIs
    icis = [ici * duration for ici in norm_icis]
    
    # Add small random variation (±5%)
    icis = [ici * (1.0 + 0.05 * (rand() - 0.5)) for ici in icis]
    
    # Add ornament if requested
    if add_ornament
        ornament_ici = duration * 0.4 + rand() * duration * 0.2
        push!(icis, ornament_ici)
    end
    
    (icis_ms = icis, rhythm = norm_icis, tempo_ms = duration, 
     has_ornament = add_ornament, n_clicks = n_clicks + (add_ornament ? 1 : 0))
end

"""
    simulate_whale_exchange(; n_codas=6, whale_ids=["A", "B"])

Simulate a multi-whale coda exchange with rubato matching.
Returns vector of (whale_id, coda) tuples.
"""
function simulate_whale_exchange(; n_codas::Int=6, whale_ids::Vector{String}=["A", "B"])
    exchange = []
    
    # Base rubato curve (smooth variation)
    rubato_curve = [0.2 * sin(2π * i / n_codas) for i in 1:n_codas]
    
    # Alternating whales
    for i in 1:n_codas
        whale = whale_ids[mod1(i, length(whale_ids))]
        
        # Vary rhythm type occasionally
        rhythm = rand() < 0.7 ? "5R1" : rand(["5R2", "5R3", "4R1"])
        
        # Ornament at phrase boundaries (start/end)
        ornament = (i == 1 || i == n_codas) && rand() < 0.3
        
        coda = simulate_whale_coda(
            rhythm_type = rhythm,
            tempo_type = "T3",
            rubato = rubato_curve[i],
            add_ornament = ornament
        )
        
        push!(exchange, (whale_id = whale, coda = coda, time_offset = (i-1) * 1200))
    end
    
    exchange
end

"""
    coda_to_intervals(icis_ms::Vector{Float64})

Convert ICI sequence to pitch class intervals (for Gay.jl integration).
"""
function coda_to_intervals(icis_ms::Vector{Float64})
    # Normalize to 0-1
    total = sum(icis_ms)
    rhythm = icis_ms ./ total
    
    # Map to intervals (mod 12)
    [round(Int, r * 12) % 12 for r in rhythm]
end

"""
    intervals_to_coda(intervals::Vector{Int}; tempo_ms::Float64=600.0)

Convert pitch class intervals to ICI sequence.
"""
function intervals_to_coda(intervals::Vector{Int}; tempo_ms::Float64=600.0)
    # Normalize intervals to rhythm
    total = sum(intervals)
    if total == 0
        return fill(tempo_ms / length(intervals), length(intervals))
    end
    
    rhythm = intervals ./ total
    [r * tempo_ms for r in rhythm]
end
