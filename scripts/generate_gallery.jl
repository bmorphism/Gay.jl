# Generate 1069 Sky Models in Parallel using SplittableRandoms
# Each thread gets its own forked RNG stream - fork-safe, deterministic, parallel
#
# Usage: julia --threads=auto scripts/generate_gallery.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Gay
using SplittableRandoms: SplittableRandom, split
using Base.Threads: @threads, nthreads, threadid
using Dates

const N_MODELS = 1069
const GALLERY_DIR = joinpath(@__DIR__, "..", "gallery")
const MASTER_SEED = 1069  # Reproducible: balanced ternary [+1, -1, -1, +1, +1, +1, +1]

println("═"^70)
println(rainbow_text("  Gay.jl Parallel Gallery Generator"))
println("═"^70)
println("  Generating $N_MODELS models across $(nthreads()) threads")
println("  Master seed: $MASTER_SEED (SplittableRandoms fork-safe)")
println("═"^70)

# ═══════════════════════════════════════════════════════════════════════════
# Model Styles with Aesthetic Variation
# ═══════════════════════════════════════════════════════════════════════════

const STYLES = [:m87, :sgra, :custom, :rings, :crescents, :spirals]

"""
Generate a random style model with aesthetic variations.
Uses the provided SplittableRandom for deterministic generation.
"""
function generate_aesthetic_model(rng::SplittableRandom, idx::Int)
    # Use rng to pick style and parameters
    style_idx = (rand(rng, UInt64) % length(STYLES)) + 1
    style = STYLES[style_idx]
    
    # Generate model-specific seed from the split RNG
    model_seed = rand(rng, UInt64) % 100000
    
    # Create model with variations based on style
    gay_seed!(Int(model_seed))
    
    if style == :m87
        # M87* variations: ring + gaussian with different sizes
        r_radius = 0.5 + (rand(rng) * 1.5)
        r_width = 0.1 + (rand(rng) * 0.5)
        g_sigma = 0.2 + (rand(rng) * 0.8)
        
        ring = comrade_ring(r_radius, r_width)
        gauss = comrade_gaussian(g_sigma, g_sigma * (0.5 + rand(rng)))
        model = sky_add(ring, gauss)
        
    elseif style == :sgra
        # Sgr A* variations: crescent + disk
        c_out = 0.8 + (rand(rng) * 0.8)
        c_in = c_out * (0.3 + rand(rng) * 0.4)
        c_shift = (rand(rng) - 0.5) * 0.6
        d_radius = 0.2 + (rand(rng) * 0.4)
        
        crescent = comrade_crescent(c_out, c_in, c_shift)
        disk = comrade_disk(d_radius)
        model = sky_add(crescent, disk)
        
    elseif style == :rings
        # Multiple rings
        n_rings = 2 + Int(rand(rng, UInt64) % 3)
        components = SkyPrimitive[]
        for i in 1:n_rings
            r = 0.3 + (i * 0.3) + (rand(rng) * 0.2)
            w = 0.1 + (rand(rng) * 0.2)
            push!(components, comrade_ring(r, w))
        end
        model = sky_add(components...)
        
    elseif style == :crescents
        # Multiple crescents
        n_crescents = 1 + Int(rand(rng, UInt64) % 2)
        components = SkyPrimitive[]
        for i in 1:n_crescents
            r_out = 0.6 + (i * 0.4) + (rand(rng) * 0.3)
            r_in = r_out * (0.4 + rand(rng) * 0.3)
            shift = (rand(rng) - 0.5) * 0.5
            push!(components, comrade_crescent(r_out, r_in, shift))
        end
        model = sky_add(components...)
        
    elseif style == :spirals
        # Ring + multiple gaussians (spiral-like)
        ring = comrade_ring(1.0, 0.2)
        components = SkyPrimitive[ring]
        n_gauss = 2 + Int(rand(rng, UInt64) % 4)
        for i in 1:n_gauss
            σ = 0.2 + (rand(rng) * 0.4)
            push!(components, comrade_gaussian(σ, σ * (0.5 + rand(rng))))
        end
        model = sky_add(components...)
        
    else  # :custom
        # Random combination
        components = SkyPrimitive[]
        if rand(rng) > 0.3
            push!(components, comrade_ring(0.5 + rand(rng), 0.1 + rand(rng) * 0.3))
        end
        if rand(rng) > 0.4
            push!(components, comrade_gaussian(0.3 + rand(rng) * 0.5))
        end
        if rand(rng) > 0.5
            push!(components, comrade_crescent(1.0 + rand(rng) * 0.5, 0.4 + rand(rng) * 0.3, rand(rng) * 0.3))
        end
        if rand(rng) > 0.6
            push!(components, comrade_disk(0.2 + rand(rng) * 0.3))
        end
        if isempty(components)
            push!(components, comrade_ring(1.0, 0.3))
        end
        model = sky_add(components...)
    end
    
    return (
        idx = idx,
        seed = model_seed,
        style = style,
        model = model,
        sexpr = sky_show(model),
        n_components = length(model.components)
    )
end

"""
Compute aesthetic score for a model (higher = more interesting).
"""
function aesthetic_score(entry)
    score = 0.0
    
    # More components = more interesting (up to a point)
    score += min(entry.n_components, 4) * 10
    
    # Certain styles get bonuses
    if entry.style == :spirals
        score += 15
    elseif entry.style == :rings && entry.n_components >= 3
        score += 20
    elseif entry.style == :crescents
        score += 10
    end
    
    # Variety bonus based on seed
    score += (entry.seed % 50) / 5
    
    return score
end

# ═══════════════════════════════════════════════════════════════════════════
# Parallel Generation with SplittableRandoms
# ═══════════════════════════════════════════════════════════════════════════

println("\n▸ Creating $(nthreads()) forked RNG streams from master seed...")

# Create master RNG and split for each model
master_rng = SplittableRandom(UInt64(MASTER_SEED))
model_rngs = [split(master_rng) for _ in 1:N_MODELS]

println("  ◆ Fork-safe: each model gets independent deterministic stream")

# Storage for results (thread-safe via pre-allocation)
results = Vector{Any}(undef, N_MODELS)

println("\n▸ Generating $N_MODELS models in parallel...")
start_time = time()

@threads for i in 1:N_MODELS
    results[i] = generate_aesthetic_model(model_rngs[i], i)
    
    # Progress update every 100 models
    if i % 100 == 0
        print("\r  Progress: $i / $N_MODELS (thread $(threadid()))")
    end
end

elapsed = time() - start_time
println("\r  ◆ Generated $N_MODELS models in $(round(elapsed, digits=2))s")

# ═══════════════════════════════════════════════════════════════════════════
# Rank by Aesthetic Score
# ═══════════════════════════════════════════════════════════════════════════

println("\n▸ Ranking by aesthetic score...")
scores = [(i, aesthetic_score(results[i])) for i in 1:N_MODELS]
sort!(scores, by=x->x[2], rev=true)

# ═══════════════════════════════════════════════════════════════════════════
# Save Gallery
# ═══════════════════════════════════════════════════════════════════════════

println("\n▸ Saving gallery to $GALLERY_DIR...")

# Save index.md
open(joinpath(GALLERY_DIR, "index.md"), "w") do f
    println(f, "# Gay.jl Sky Model Gallery")
    println(f, "")
    println(f, "Generated: $(Dates.now())")
    println(f, "Master Seed: $MASTER_SEED")
    println(f, "Models: $N_MODELS")
    println(f, "Threads: $(nthreads())")
    println(f, "")
    println(f, "## Top 69 by Aesthetic Score")
    println(f, "")
    
    for (rank, (idx, score)) in enumerate(scores[1:69])
        entry = results[idx]
        println(f, "### #$rank - Model $(entry.idx) (score: $(round(score, digits=1)))")
        println(f, "- **Style:** $(entry.style)")
        println(f, "- **Seed:** $(entry.seed)")
        println(f, "- **Components:** $(entry.n_components)")
        println(f, "- **S-Expression:** `$(replace(entry.sexpr, r"\e\[[^m]*m" => ""))`")
        println(f, "")
    end
    
    println(f, "## Style Distribution")
    println(f, "")
    style_counts = Dict{Symbol, Int}()
    for r in results
        style_counts[r.style] = get(style_counts, r.style, 0) + 1
    end
    for (style, count) in sort(collect(style_counts), by=x->x[2], rev=true)
        println(f, "- $style: $count ($(round(100*count/N_MODELS, digits=1))%)")
    end
end

# Save catalog.jsonl (one JSON per line for streaming)
using Dates
open(joinpath(GALLERY_DIR, "catalog.jsonl"), "w") do f
    for (rank, (idx, score)) in enumerate(scores)
        entry = results[idx]
        sexpr_clean = replace(entry.sexpr, r"\e\[[^m]*m" => "")
        println(f, """{"rank":$rank,"idx":$(entry.idx),"seed":$(entry.seed),"style":"$(entry.style)","score":$(round(score,digits=2)),"components":$(entry.n_components),"sexpr":"$sexpr_clean"}""")
    end
end

# Save top 69 renders
renders_dir = joinpath(GALLERY_DIR, "renders")
mkpath(renders_dir)

println("  Rendering top 69 models...")
for (rank, (idx, _)) in enumerate(scores[1:69])
    entry = results[idx]
    render = sky_render(entry.model; size=30)
    # Strip ANSI for file storage
    render_clean = replace(render, r"\e\[[^m]*m" => "")
    open(joinpath(renders_dir, "$(lpad(rank, 3, '0'))_model_$(entry.idx).txt"), "w") do f
        println(f, "Model $(entry.idx) | Style: $(entry.style) | Seed: $(entry.seed)")
        println(f, "S-Expr: $(replace(entry.sexpr, r"\e\[[^m]*m" => ""))")
        println(f, "")
        print(f, render_clean)
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════

println("\n" * "═"^70)
println(rainbow_text("  Gallery Generation Complete!"))
println("═"^70)

println("\n  Files created:")
println("  • gallery/index.md - Top 69 models + stats")
println("  • gallery/catalog.jsonl - All $N_MODELS models")
println("  • gallery/renders/*.txt - Top 69 ASCII renders")

println("\n  Top 5 Models:")
for (rank, (idx, score)) in enumerate(scores[1:5])
    entry = results[idx]
    sexpr_clean = replace(entry.sexpr, r"\e\[[^m]*m" => "")
    println("    #$rank [$(entry.style)] seed=$(entry.seed): $sexpr_clean")
end

println("\n  Reproducibility: Run again with seed $MASTER_SEED → identical gallery")
println("  Fork-safe: Each of $N_MODELS models used independent SplittableRandom stream")
println("═"^70)
