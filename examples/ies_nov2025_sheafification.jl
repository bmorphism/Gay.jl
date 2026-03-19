# IES Nov 2025 Message Sheafification via Gay-Dyck-Catalan
# =========================================================
#
# This script:
# 1. Loads IES messages from aquavoice_geo_full.parquet
# 2. Converts each message to a Dyck path based on nesting structure
# 3. Indexes messages by Catalan level
# 4. Finds optimal Gay seed bundle for maximum chromatic coherence
# 5. Sheafifies meaning across the message corpus

using DuckDB
using DataFrames

# Include the Gay modules (adjust path as needed)
include(joinpath(@__DIR__, "..", "src", "gay_dyck_catalan.jl"))
using .GayDyckCatalan

const GAY_SEED = UInt64(1069)

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD IES MESSAGES FROM PARQUET
# ═══════════════════════════════════════════════════════════════════════════════

function load_ies_messages_from_parquet(parquet_path::String)
    # Connect to DuckDB and query parquet
    con = DBInterface.connect(DuckDB.DB)
    
    query = """
        SELECT 
            row_number() OVER (ORDER BY ts) as id,
            content,
            ts::VARCHAR as timestamp,
            word_count
        FROM '$parquet_path'
        WHERE content IS NOT NULL 
          AND char_count > 0
        ORDER BY ts
    """
    
    result = DBInterface.execute(con, query)
    df = DataFrame(result)
    
    DBInterface.close!(con)
    
    df
end

function df_to_ies_messages(df::DataFrame; seed::UInt64=GAY_SEED)
    messages = IESMessage[]
    
    for row in eachrow(df)
        content = something(row.content, "")
        timestamp = something(row.timestamp, nothing)
        word_count = something(row.word_count, 0)
        
        msg = IESMessage(
            row.id,
            content,
            timestamp,
            word_count;
            seed=seed
        )
        push!(messages, msg)
    end
    
    messages
end

# ═══════════════════════════════════════════════════════════════════════════════
# ANALYZE AND SHEAFIFY
# ═══════════════════════════════════════════════════════════════════════════════

function analyze_messages(messages::Vector{IESMessage})
    println("\n═══════════════════════════════════════════════════════════════════")
    println("  IES NOV 2025 MESSAGE ANALYSIS")
    println("═══════════════════════════════════════════════════════════════════\n")
    
    # Basic stats
    println("CORPUS STATISTICS:")
    println("  Total messages: $(length(messages))")
    total_words = sum(m.word_count for m in messages)
    println("  Total words: $total_words")
    println("  Avg words/message: $(round(total_words / length(messages), digits=1))")
    println()
    
    # Catalan level distribution
    indexed = catalan_index_messages(messages)
    println("CATALAN LEVEL DISTRIBUTION:")
    println("  Level   Count   Catalan(n)   Messages")
    println("  ─────────────────────────────────────────────")
    
    for level in sort(collect(keys(indexed)))
        c_n = catalan(level)
        count = length(indexed[level])
        bar = repeat("█", min(50, count))
        println("  $(lpad(level, 5))   $(lpad(count, 5))   $(lpad(c_n, 10))   $bar")
    end
    println()
    
    # Dyck path analysis
    println("DYCK PATH ANALYSIS:")
    max_height = maximum(m.dyck_path.max_height for m in messages)
    total_area = sum(m.dyck_path.area for m in messages)
    total_peaks = sum(length(m.dyck_path.peaks) for m in messages)
    
    println("  Max nesting depth: $max_height")
    println("  Total area under paths: $total_area")
    println("  Total peaks: $total_peaks")
    println()
    
    # Color statistics
    println("CHROMATIC STATISTICS:")
    colors = [m.color for m in messages]
    mean_r = sum(c[1] for c in colors) / length(colors)
    mean_g = sum(c[2] for c in colors) / length(colors)
    mean_b = sum(c[3] for c in colors) / length(colors)
    
    println("  Mean color: RGB($(round(mean_r * 255)), $(round(mean_g * 255)), $(round(mean_b * 255)))")
    
    # Color variance
    var_r = sum((c[1] - mean_r)^2 for c in colors) / length(colors)
    var_g = sum((c[2] - mean_g)^2 for c in colors) / length(colors)
    var_b = sum((c[3] - mean_b)^2 for c in colors) / length(colors)
    
    println("  Color variance: ($(round(var_r, digits=4)), $(round(var_g, digits=4)), $(round(var_b, digits=4)))")
    println()
    
    indexed
end

function find_optimal_seed_for_messages(messages::Vector{IESMessage}; n_seeds::Int=16)
    println("SEED OPTIMIZATION:")
    println("  Testing $n_seeds seeds for optimal coherence...\n")
    
    best_seed = GAY_SEED
    best_coherence = 0.0
    results = []
    
    for i in 1:n_seeds
        # Generate candidate seed
        seed = GAY_SEED ⊻ UInt64(i * 0x9e3779b97f4a7c15)
        
        # Recompute messages with this seed
        recomputed = [
            IESMessage(m.id, m.content, m.timestamp, m.word_count; seed=seed)
            for m in messages
        ]
        
        # Sheafify and get coherence
        result = sheafify_messages(recomputed; seed=seed)
        coherence = result.global_coherence
        
        push!(results, (seed=seed, coherence=coherence))
        
        if coherence > best_coherence
            best_coherence = coherence
            best_seed = seed
        end
    end
    
    # Sort by coherence
    sort!(results, by=r -> r.coherence, rev=true)
    
    println("  Top 5 seeds:")
    for (i, r) in enumerate(results[1:min(5, length(results))])
        marker = r.seed == best_seed ? " ★" : ""
        println("    $(i). 0x$(string(r.seed, base=16)[1:12])... coherence=$(round(r.coherence, digits=4))$marker")
    end
    println()
    
    (best_seed=best_seed, best_coherence=best_coherence, all_results=results)
end

function sheafify_and_report(messages::Vector{IESMessage}; seed::UInt64=GAY_SEED)
    println("SHEAFIFICATION RESULTS:")
    
    result = sheafify_messages(messages; seed=seed)
    
    println("  Seed: 0x$(string(seed, base=16))")
    println("  Global coherence: $(round(result.global_coherence, digits=4))")
    println()
    
    println("  Level-wise coherence:")
    for level in sort(collect(keys(result.level_coherence)))
        coh = result.level_coherence[level]
        bar = repeat("█", round(Int, coh * 30))
        println("    Level $level: $bar $(round(coh, digits=3))")
    end
    println()
    
    # Meaning assignment rate
    bundle = SeedBundle([seed]; max_n=5)
    mar = meaning_assignment_rate(bundle, messages)
    println("  Meaning assignment rate: $(round(mar, digits=4))")
    
    result
end

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

function main()
    println()
    println("╔═══════════════════════════════════════════════════════════════════════════╗")
    println("║  IES NOV 2025 MESSAGE SHEAFIFICATION via GAY-DYCK-CATALAN                 ║")
    println("║  Finding optimal Gay seed bundle for maximum chromatic coherence          ║")
    println("╚═══════════════════════════════════════════════════════════════════════════╝")
    println()
    
    # Path to parquet file (adjust as needed)
    parquet_path = joinpath(@__DIR__, "..", "..", "aquavoice_geo_full.parquet")
    
    if !isfile(parquet_path)
        println("⚠️  Parquet file not found at: $parquet_path")
        println("   Please ensure aquavoice_geo_full.parquet is available.")
        println()
        println("   Running demo with synthetic messages instead...")
        
        # Create synthetic messages for demo
        messages = [
            IESMessage(1, "Hello (world) and (nested (deeply))", "2025-11-01", 5),
            IESMessage(2, "Simple message without nesting", "2025-11-02", 4),
            IESMessage(3, "Multiple (levels (of (nesting))) here", "2025-11-03", 5),
            IESMessage(4, "Balanced ((pairs)) of ((parentheses))", "2025-11-04", 4),
            IESMessage(5, "Complex {structure} with [multiple] (types)", "2025-11-05", 5),
        ]
    else
        println("Loading messages from: $parquet_path")
        df = load_ies_messages_from_parquet(parquet_path)
        println("Loaded $(nrow(df)) messages from parquet.")
        println()
        
        messages = df_to_ies_messages(df; seed=GAY_SEED)
    end
    
    # Analyze
    indexed = analyze_messages(messages)
    
    # Find optimal seed
    opt = find_optimal_seed_for_messages(messages; n_seeds=16)
    
    # Sheafify with optimal seed
    result = sheafify_and_report(messages; seed=opt.best_seed)
    
    # Final summary
    println("═══════════════════════════════════════════════════════════════════")
    println("  SUMMARY: CHROMATIC INVARIANTS")
    println("═══════════════════════════════════════════════════════════════════")
    println()
    println("  Optimal seed: 0x$(string(opt.best_seed, base=16))")
    println("  Global coherence: $(round(opt.best_coherence, digits=4))")
    println("  Catalan levels spanned: $(length(indexed))")
    println("  Total fingerprint XOR: 0x$(string(reduce(⊻, [m.fingerprint for m in messages]), base=16))")
    println()
    println("  The Gay-Dyck-Birb Trinity successfully sheafified $(length(messages)) messages")
    println("  across $(length(indexed)) Catalan levels with chromatic invariance.")
    println()
    
    (messages=messages, indexed=indexed, optimal_seed=opt.best_seed, result=result)
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
