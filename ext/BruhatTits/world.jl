# World: Color-based parallel agents with sentinel monitoring
#
# Pattern: Never block. Always split 3 before I/O.
# Each agent gets next_color. Sentinels kill violators.

include("agent_colors.jl")
include("split3.jl")
include("sentinel.jl")

using .AgentColors
using .Split3
using .Sentinel

const GAY_SEED = UInt64(0x6761795f636f6c6f)

sm64(z) = let z = z + 0x9E3779B97F4A7C15
    z = (z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ⊻ (z >> 27)) * 0x94D049BB133111EB
    z ⊻ (z >> 31)
end

function world(seed::UInt64=GAY_SEED)
    println("=" ^ 60)
    println("WORLD: Split3 + Sentinel + next_color")
    println("=" ^ 60)
    
    # ═══════════════════════════════════════════════════════════
    # PHASE 1: Spawn sentinels (always 3)
    # ═══════════════════════════════════════════════════════════
    sentinels = run_sentinels(3)
    sentinel_colors = [sm64(seed ⊻ UInt64(i)) for i in 1:3]
    
    println("\n[SENTINELS] 3 monitors spawned:")
    labels = ["α", "β", "γ"]
    for (i, c) in enumerate(sentinel_colors)
        println("  $(labels[i]) → 0x$(string(c, base=16)[1:8])...")
    end
    
    # ═══════════════════════════════════════════════════════════
    # PHASE 2: Create world state
    # ═══════════════════════════════════════════════════════════
    world_color = sm64(seed)
    println("\n[WORLD] color: 0x$(string(world_color, base=16))")
    
    # ═══════════════════════════════════════════════════════════
    # PHASE 3: Split3 before READ
    # ═══════════════════════════════════════════════════════════
    println("\n[SPLIT3:READ] Before any read, split into 3:")
    ctx_read = split3(world_color, :read)
    
    agents_read = []
    for (i, c) in enumerate(ctx_read.child_colors)
        agent = ColoredAgent(i, c, [])
        push!(agents_read, agent)
        
        # Sentinel validates
        valid = validate_agent(sentinels[mod1(i, 3)], i, :read, c, c)
        status = valid ? "✓" : "✗"
        println("  Agent $i: 0x$(string(c, base=16)[1:8])... $status")
    end
    
    # ═══════════════════════════════════════════════════════════
    # PHASE 4: Split3 before WRITE
    # ═══════════════════════════════════════════════════════════
    println("\n[SPLIT3:WRITE] Before any write, split into 3:")
    ctx_write = split3(ctx_read.child_colors[1], :write)
    
    agents_write = []
    for (i, c) in enumerate(ctx_write.child_colors)
        agent = ColoredAgent(i + 3, c, [])
        push!(agents_write, agent)
        
        valid = validate_agent(sentinels[mod1(i, 3)], i + 3, :write, c, c)
        status = valid ? "✓" : "✗"
        println("  Agent $(i+3): 0x$(string(c, base=16)[1:8])... $status")
    end
    
    # ═══════════════════════════════════════════════════════════
    # PHASE 5: Simulate violation → KILL
    # ═══════════════════════════════════════════════════════════
    println("\n[VIOLATION] Agent 99 uses wrong color:")
    bad_color = UInt64(0xDEADBEEFCAFE)
    expected = ctx_write.child_colors[1]
    valid = validate_agent(sentinels[1], 99, :write, bad_color, expected)
    
    println("  Expected: 0x$(string(expected, base=16)[1:8])...")
    println("  Actual:   0x$(string(bad_color, base=16)[1:8])...")
    println("  Result:   $(valid ? "VALID" : "KILLED ✗")")
    println("  Sentinel α killed: $(sentinels[1].killed_agents)")
    
    # ═══════════════════════════════════════════════════════════
    # PHASE 6: Fingerprint aggregation
    # ═══════════════════════════════════════════════════════════
    all_colors = vcat(
        collect(ctx_read.child_colors),
        collect(ctx_write.child_colors),
        sentinel_colors
    )
    global_fp = reduce(⊻, all_colors)
    
    println("\n[FINGERPRINT] XOR composition of all agents:")
    println("  Global: 0x$(string(global_fp, base=16))")
    
    # ═══════════════════════════════════════════════════════════
    # PHASE 7: Summary
    # ═══════════════════════════════════════════════════════════
    total_agents = length(agents_read) + length(agents_write)
    total_killed = sum(length(s.killed_agents) for s in sentinels)
    
    println("\n" * "=" ^ 60)
    println("WORLD SUMMARY")
    println("  Agents spawned: $total_agents (via Split3)")
    println("  Sentinels:      3 (always monitoring)")
    println("  Killed:         $total_killed (color violations)")
    println("  Fingerprint:    0x$(string(global_fp, base=16)[1:16])...")
    println("=" ^ 60)
    
    (
        agents = vcat(agents_read, agents_write),
        sentinels = sentinels,
        fingerprint = global_fp,
        killed = total_killed
    )
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    world()
end
