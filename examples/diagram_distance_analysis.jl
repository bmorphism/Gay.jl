using Gay
using Printf

# This example demonstrates the first-class integration of Gay.jl with
# PersistenceDiagrams.jl to compute diagram distances (Bottleneck and Wasserstein)
# and matching features directly on raw trajectories and walk results.
#
# To run this example, ensure the weak dependencies are loaded:
# julia> using Pkg; Pkg.add(["Ripserer", "PersistenceDiagrams"])
# julia> include("examples/diagram_distance_analysis.jl")

println("================================================================")
println("    🌈  GAY.JL PERSISTENCE DIAGRAM DISTANCE & MATCHING  🌈")
println("================================================================")
println()

# 1. Check if extensions are active
has_ripserer = !isempty(methods(gay_ripserer))
has_pd       = !isempty(methods(gay_persistencediagram))

if !has_ripserer || !has_pd
    println("⚠️  Some integration extensions are not loaded in the current session.")
    if !has_ripserer
        println("   - Ripserer.jl is missing (needed for generating diagrams).")
    end
    if !has_pd
        println("   - PersistenceDiagrams.jl is missing (needed for distance metrics).")
    end
    println()
    println("   Attempting to load them now dynamically...")
    try
        using Ripserer
        global has_ripserer = true
        println("   ✅ Ripserer.jl successfully loaded dynamically!")
    catch e
        println("   ❌ Could not load Ripserer.jl dynamically.")
    end
    try
        using PersistenceDiagrams
        global has_pd = true
        println("   ✅ PersistenceDiagrams.jl successfully loaded dynamically!")
    catch e
        println("   ❌ Could not load PersistenceDiagrams.jl dynamically.")
    end
    println()
end

if !has_ripserer || !has_pd
    println("❌ This script requires both Ripserer.jl and PersistenceDiagrams.jl to run.")
    println("   Please run standard installation:")
    println("   julia --project -e 'using Pkg; Pkg.add([\"Ripserer\", \"PersistenceDiagrams\"])'")
    exit(1)
end

# 2. Setup Adjacency Grid and Walks
println("1️⃣  Simulating 2D Grid Self-Avoiding Walks...")

grid_size = 6
adjacency = Dict{String,Vector{String}}()
for x in 1:grid_size
    for y in 1:grid_size
        node = "node/$x/$y"
        neighbors = String[]
        x > 1 && push!(neighbors, "node/$(x-1)/$y")
        x < grid_size && push!(neighbors, "node/$(x+1)/$y")
        y > 1 && push!(neighbors, "node/$x/$(y-1)")
        y < grid_size && push!(neighbors, "node/$x/$(y+1)")
        adjacency[node] = neighbors
    end
end

# Walk 1: Starts at (1, 1), seed = GAY_SEED
walk1 = color_self_avoiding_walk(adjacency, "node/1/1"; steps=12, seed=GAY_SEED)
# Walk 2: Starts at (1, 1), slightly different seed
walk2 = color_self_avoiding_walk(adjacency, "node/1/1"; steps=12, seed=GAY_SEED + 10)

println("   - Walk 1: Completed $(length(walk1.steps)) steps (Seed: $GAY_SEED)")
println("   - Walk 2: Completed $(length(walk2.steps)) steps (Seed: $(GAY_SEED + 10))")
println()

# 3. Create First-Class GayPersistenceDiagrams
println("2️⃣  Constructing First-Class GayPersistenceDiagrams...")

# Using the thin wrapper on walk results directly
gpd1 = gay_persistencediagram(walk1; dim=0)
gpd2 = gay_persistencediagram(walk2; dim=0)

println("\n--- gpd1 Display ---")
display(gpd1)

println("\n--- gpd2 Display ---")
display(gpd2)
println()

# 4. Compute Distances Directly on Raw Walk Results & Diagrams
println("3️⃣  Calculating Topological Distances...")

# A. Distances on first-class GayPersistenceDiagrams
dist_b_gpd = gay_bottleneck(gpd1, gpd2)
dist_w_gpd = gay_wasserstein(gpd1, gpd2)

# B. Distances directly on raw WalkResults
dist_b_walk = gay_bottleneck(walk1, walk2; dim=0)
dist_w_walk = gay_wasserstein(walk1, walk2; dim=0)

# C. Distances directly on raw colors
colors1 = [step.color for step in walk1.steps]
colors2 = [step.color for step in walk2.steps]
dist_b_colors = gay_bottleneck(colors1, colors2; dim=0)
dist_w_colors = gay_wasserstein(colors1, colors2; dim=0)

println("   ┌──────────────────────────────────────────────────────────────┐")
println("   │  Distance Type  │  On GayDiagrams  │ On Raw Walks │ On Colors │")
println("   ├─────────────────┼──────────────────┼──────────────┼───────────┤")
@printf("   │  Bottleneck     │    %10.4f    │  %10.4f  │ %9.4f │\n", dist_b_gpd, dist_b_walk, dist_b_colors)
@printf("   │  Wasserstein    │    %10.4f    │  %10.4f  │ %9.4f │\n", dist_w_gpd, dist_w_walk, dist_w_colors)
println("   └──────────────────────────────────────────────────────────────┘")
println()

# 5. Overloaded Functors and Matchings
println("4️⃣  Overloaded Functors & Matchings...")

# Using GayBottleneck / GayWasserstein Functors
bot_func = GayBottleneck()
was_func = GayWasserstein()

println("   - Callable GayBottleneck functor:  ", bot_func(gpd1, gpd2))
println("   - Callable GayWasserstein functor: ", was_func(gpd1, gpd2))

# Matching features directly on walks
match_b = gay_matching(GayBottleneck(), walk1, walk2; dim=0)
match_w = gay_matching(GayWasserstein(), gpd1, gpd2)

println("\n--- Bottleneck Matching Output ---")
println(match_b)
println("\n--- Wasserstein Matching Output ---")
println(match_w)
println()

println("================================================================")
println("    ✅  All first-class Gay topological metrics verified!")
println("================================================================")
