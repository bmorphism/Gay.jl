using Gay

# This example demonstrates the lazy integration of Gay.jl with Ripserer.jl 
# (persistent homology) and FractalDimensions.jl (correlation dimension).
#
# To run this example, ensure the weak dependencies are loaded:
# julia> using Pkg; Pkg.add(["Ripserer", "FractalDimensions"])
# julia> include("examples/topological_dimension_analysis.jl")

println("================================════════════════════════════════")
println("   GAY.JL TOPOLOGICAL & FRACTAL DIMENSION ANALYSIS")
println("================================════════════════════════════════")
println()

# 1. Check if extensions are active by checking if methods exist for our extension points
has_ripserer = !isempty(methods(gay_ripserer))
has_fractal  = !isempty(methods(gay_fractal_dimension))

if !has_ripserer || !has_fractal
    println("⚠️  Some integration extensions are not loaded in the current session.")
    if !has_ripserer
        println("   - Ripserer.jl is missing. To load GayRipsererExt, run: using Ripserer")
    end
    if !has_fractal
        println("   - FractalDimensions.jl is missing. To load GayFractalExt, run: using FractalDimensions")
    end
    println()
    println("   Please run this script from an environment with these packages loaded.")
    println("   For example:")
    println("     julia --project -e 'using Pkg; Pkg.add([\"Ripserer\", \"FractalDimensions\"]); using Ripserer, FractalDimensions; include(\"examples/topological_dimension_analysis.jl\")'")
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
        using FractalDimensions
        global has_fractal = true
        println("   ✅ FractalDimensions.jl successfully loaded dynamically!")
    catch e
        println("   ❌ Could not load FractalDimensions.jl dynamically.")
    end
    println()
end

# 2. Define a walking environment (a 2D grid/lattice of labels)
# We will run a deterministic self-avoiding walk to generate a color point cloud
println("1. Generating Walk Trajectory...")
grid_size = 5
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

walk_len = 16
start_node = "node/1/1"
walk = color_self_avoiding_walk(adjacency, start_node; steps=walk_len, seed=GAY_SEED)

println("   - Walk completed: ", length(walk.steps), " steps.")
println("   - Walk nodes: ", join([step.node for step in walk.steps], " -> "))
println("   - Walk colors: ", join([step.color for step in walk.steps], ", "))
println()

# 3. Persistent Homology (Ripserer.jl)
if has_ripserer
    println("2. Topological Persistence Analysis (Ripserer.jl)")
    println("   Computing persistent homology over the sRGB color trajectory...")
    
    # Analyze the colors directly
    diagrams = gay_ripserer(walk; dim_max=1)
    
    for (d, diag) in enumerate(diagrams)
        dim = d - 1
        println("   ┌ Dimension H_$dim persistent features:")
        if isempty(diag)
            println("   │   (No features detected)")
        else
            # Sort features by persistence (death - birth)
            features = sort(diag; by=f -> f.death - f.birth, rev=true)
            for (idx, f) in enumerate(features[1:min(length(features), 3)])
                birth = round(f.birth; digits=4)
                death = isinf(f.death) ? "∞" : round(f.death; digits=4)
                pers = isinf(f.death) ? "∞" : round(f.death - f.birth; digits=4)
                println("   │   #$idx: Birth = $birth, Death = $death, Persistence = $pers")
            end
        end
        println("   └──────────────────────────────────────────────────")
    end
    println()
else
    println("2. Topological Persistence Analysis (Ripserer.jl) [Skipped]")
end

# 4. Correlation Dimension (FractalDimensions.jl)
if has_fractal
    println("3. Space-Filling Fractal Complexity (FractalDimensions.jl)")
    println("   Estimating the Grassberger-Procaccia correlation dimension...")
    
    # Estimate fractal dimension of our walk colors
    fd_walk = gay_fractal_dimension(walk)
    
    # Generate larger uniform set to compare
    n_uniform = 100
    fd_uniform = gay_fractal_dimension(n_uniform; seed=GAY_SEED)
    
    println("   ┌──────────────────────────────────────────────────")
    println("   │  Walk sRGB Path Dimension:     ", round(fd_walk; digits=4))
    println("   │  Uniform Sample (N=$n_uniform) Dim: ", round(fd_uniform; digits=4))
    println("   └──────────────────────────────────────────────────")
    println()
    println("   💡 Interpretability:")
    println("      - D ≈ 1.0 indicates a tight 1D curve constraint (collinear / simple).")
    println("      - D ≈ 2.0 indicates a 2D surface constraint (planar manifold).")
    println("      - D > 2.5 indicates high space-filling chaotic volume complexity.")
    println()
else
    println("3. Space-Filling Fractal Complexity (FractalDimensions.jl) [Skipped]")
end

println("================================════════════════════════════════")
println("   Integration check complete!")
println("================================════════════════════════════════")
