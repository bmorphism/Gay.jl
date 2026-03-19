#!/usr/bin/env julia
# Example usage of SwarmTriad module

# Load the module directly
include("src/swarm_triad.jl")
using .SwarmTriad

println("\n🔵 SWARM TRIAD EXAMPLE - AGENT-AZUL\n")

# Step 1: Create sentinel with AGENT-AZUL color seed
sentinel = create_sentinel(UInt64(0x415A554C))
println("Sentinel created: 0x$(string(sentinel.seed, base=16))")

# Step 2: Create some worker agents
println("\nCreating worker agents...")
worker1 = create_agent(UInt64(0x1111111))
worker2 = create_agent(UInt64(0x2222222))
worker3 = create_agent(UInt64(0x3333333))

# Register with sentinel
register_agent!(sentinel, worker1)
register_agent!(sentinel, worker2)
register_agent!(sentinel, worker3)

println("  ✓ Worker 1: $(agent_identity(worker1))")
println("  ✓ Worker 2: $(agent_identity(worker2))")
println("  ✓ Worker 3: $(agent_identity(worker3))")

# Step 3: Workers perform file operations (with mandatory split)
println("\nWorkers performing file operations...")
for (i, worker) in enumerate([worker1, worker2, worker3])
    op = ReadFile("file_$i.txt")
    success, split = execute_file_op!(worker, op, sentinel)
    
    if split !== nothing
        println("  Worker $i: Split into 3 children")
        # Register children
        register_agent!(sentinel, split.left)
        register_agent!(sentinel, split.middle)
        register_agent!(sentinel, split.right)
    end
end

# Step 4: Monitor compliance
println("\nMonitoring compliance...")
monitor_swarm!(sentinel)

# Step 5: Report
println("\n📊 COMPLIANCE REPORT:")
report = compliance_report(sentinel)
for (key, value) in sort(collect(report))
    println("  $key: $value")
end

println("\n✅ Example complete!\n")
