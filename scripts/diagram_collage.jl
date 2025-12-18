#!/usr/bin/env julia
"""
Diagram Collage Generator with Tropical Semiring & Counterfactual Regret
═══════════════════════════════════════════════════════════════════════════

DERANGEABLE/COLORABLE diagram collage using:
1. Tropical (min, +) semiring for shortest-path optimization
2. Counterfactual regret minimization over missed Galois connections
3. Derangement coloring (no adjacent diagrams share colors)

THEORETICAL FOUNDATION:
- Tropical semiring: (ℝ ∪ {∞}, min, +) for path optimization
- Galois connections: α ⊣ γ between Event ↔ Color spaces
- Counterfactual regret: what if we had connected these nodes?
- Derangement: permutation where no element stays in place

Usage: julia --threads=auto scripts/diagram_collage.jl
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Gay
using Gay: color_at, GAY_SEED, splitmix64
using Gay: TropicalMinPlus, semiring_add, semiring_mul, semiring_zero, semiring_one
using Gay.FaultTolerant: GaloisConnection, Event, Color, verify_closure
import Gay.FaultTolerant: alpha as galois_alpha, gamma as galois_gamma
using Colors
using Random
using Dates

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

const COLLAGE_SEED = 0xC011A6E5EED69420  # Collage seed (nice hex)
const N_DIAGRAMS = 69                      # Number of diagrams in collage
const DOCS_DIR = joinpath(@__DIR__, "..", "docs", "src", "collage")

# ═══════════════════════════════════════════════════════════════════════════════
# LCH Color Space
# ═══════════════════════════════════════════════════════════════════════════════

struct LCH
    L::Float64  # Lightness [0, 100]
    C::Float64  # Chroma [0, 150]
    H::Float64  # Hue [0, 360)
end

function lch_distance(c1::LCH, c2::LCH)::Float64
    dL = c1.L - c2.L
    dC = c1.C - c2.C
    dH = min(abs(c1.H - c2.H), 360.0 - abs(c1.H - c2.H))
    sqrt(dL^2 + dC^2 + dH^2)
end

function lch_from_index(idx::Int, seed::UInt64)::LCH
    h = splitmix64(seed ⊻ UInt64(idx))
    L = (h & 0xFFFF) / 655.35
    C = ((h >> 16) & 0xFFFF) / 436.90
    H = ((h >> 32) & 0xFFFF) / 182.04
    LCH(L, C, H)
end

function lch_to_rgb(c::LCH)::RGB
    r = clamp(c.L / 100, 0, 1)
    g = clamp(c.C / 150, 0, 1)
    b = clamp(c.H / 360, 0, 1)
    RGB(r, g, b)
end

function lch_to_ansi(c::LCH)::String
    r = round(Int, clamp(c.L * 2.55, 0, 255))
    g = round(Int, clamp(c.C * 1.7, 0, 255))
    b = round(Int, clamp((c.H / 360) * 255, 0, 255))
    "\e[38;2;$(r);$(g);$(b)m██\e[0m"
end

# ═══════════════════════════════════════════════════════════════════════════════
# Diagram Node
# ═══════════════════════════════════════════════════════════════════════════════

@enum DiagramType begin
    MORPHISM
    COMPOSITION
    IDENTITY
    TENSOR
    DUAL
    BRAIDING
    UNIT
    COUNIT
end

struct DiagramNode
    id::Int
    type::DiagramType
    color::LCH
    x::Float64  # Position in collage
    y::Float64
    label::String
end

function DiagramNode(id::Int, seed::UInt64)
    h = splitmix64(seed ⊻ UInt64(id * 0x1234567890ABCDEF))
    type = DiagramType(h % 8)
    color = lch_from_index(id, seed)
    x = ((h >> 16) % 100) / 100.0
    y = ((h >> 32) % 100) / 100.0
    label = "D$(id)"
    DiagramNode(id, type, color, x, y, label)
end

const DIAGRAM_SYMBOLS = Dict(
    MORPHISM => "→",
    COMPOSITION => "∘",
    IDENTITY => "id",
    TENSOR => "⊗",
    DUAL => "†",
    BRAIDING => "σ",
    UNIT => "η",
    COUNIT => "ε"
)

# ═══════════════════════════════════════════════════════════════════════════════
# Tropical Semiring Shortest Path
# ═══════════════════════════════════════════════════════════════════════════════

"""
Build adjacency matrix with tropical (min, +) semiring distances.

In tropical semiring:
- ⊕ = min (choose shortest)
- ⊗ = + (accumulate distance)
- 0 = ∞ (no path)
- 1 = 0 (zero cost)
"""
function tropical_adjacency(nodes::Vector{DiagramNode})::Matrix{Float64}
    n = length(nodes)
    A = fill(Inf, n, n)
    
    for i in 1:n
        A[i, i] = 0.0  # Self-loop = identity
        for j in (i+1):n
            dist = lch_distance(nodes[i].color, nodes[j].color)
            # Connect if colors are "close" (threshold based on hash)
            h = splitmix64(UInt64(i * 1000 + j) ⊻ COLLAGE_SEED)
            threshold = 30.0 + (h % 70)
            if dist < threshold
                A[i, j] = dist
                A[j, i] = dist
            end
        end
    end
    
    A
end

"""
Floyd-Warshall in tropical semiring: find all-pairs shortest paths.
"""
function tropical_shortest_paths(A::Matrix{Float64})::Matrix{Float64}
    n = size(A, 1)
    D = copy(A)
    
    for k in 1:n
        for i in 1:n
            for j in 1:n
                # D[i,j] = min(D[i,j], D[i,k] + D[k,j])
                new_dist = semiring_mul(TropicalMinPlus, D[i,k], D[k,j])
                D[i,j] = semiring_add(TropicalMinPlus, D[i,j], new_dist)
            end
        end
    end
    
    D
end

# ═══════════════════════════════════════════════════════════════════════════════
# Counterfactual Regret over Missed Galois Connections
# ═══════════════════════════════════════════════════════════════════════════════

"""
Compute counterfactual regret for a missed Galois connection.

REGRET = value(if we had connected) - value(actual connection)

A Galois connection α ⊣ γ exists between:
- Event space (diagram operations)
- Color space (visual representation)

If we MISSED a connection (A, B), the regret is:
  R(A, B) = benefit_of_connection(A, B) - cost_of_missing(A, B)
"""
struct CounterfactualRegret
    from_id::Int
    to_id::Int
    actual_distance::Float64     # What we got (maybe ∞)
    optimal_distance::Float64    # What we could have had
    regret::Float64              # Difference (always ≥ 0)
    galois_closure::Bool         # Would this form a Galois closure?
end

function compute_regrets(
    nodes::Vector{DiagramNode},
    adjacency::Matrix{Float64},
    shortest::Matrix{Float64},
    gc::GaloisConnection
)::Vector{CounterfactualRegret}
    
    regrets = CounterfactualRegret[]
    n = length(nodes)
    
    for i in 1:n
        for j in (i+1):n
            actual = adjacency[i, j]
            optimal = lch_distance(nodes[i].color, nodes[j].color)
            
            # Regret = tropical distance we're paying vs optimal direct
            # In tropical: we want to MINIMIZE, so regret = actual - optimal
            if actual == Inf
                # Missed connection entirely - high regret
                regret = optimal * 10  # Penalty for missing
            else
                regret = max(0.0, actual - optimal)
            end
            
            # Check Galois closure: α(γ(c)) ≈ c
            color_idx = round(Int, nodes[i].color.L) % gc.palette_size
            c1 = Color(color_idx, gc.palette[color_idx + 1])
            galois_ok = verify_closure(gc, c1)
            
            if regret > 0.1  # Only track significant regrets
                push!(regrets, CounterfactualRegret(
                    i, j, actual, optimal, regret, galois_ok
                ))
            end
        end
    end
    
    sort!(regrets, by=r -> -r.regret)  # Highest regret first
    regrets
end

"""
Minimize total counterfactual regret by adding connections.

Greedy approach: add edges that reduce regret the most,
subject to derangement constraint.
"""
function minimize_regret!(
    adjacency::Matrix{Float64},
    regrets::Vector{CounterfactualRegret};
    max_additions::Int = 20
)::Vector{Tuple{Int, Int, Float64}}
    
    additions = Tuple{Int, Int, Float64}[]
    
    for (idx, r) in enumerate(regrets)
        if length(additions) >= max_additions
            break
        end
        
        # Only add if it forms a Galois closure
        if r.galois_closure && r.actual_distance == Inf
            adjacency[r.from_id, r.to_id] = r.optimal_distance
            adjacency[r.to_id, r.from_id] = r.optimal_distance
            push!(additions, (r.from_id, r.to_id, r.regret))
        end
    end
    
    additions
end

# ═══════════════════════════════════════════════════════════════════════════════
# Derangement Coloring
# ═══════════════════════════════════════════════════════════════════════════════

"""
A DERANGEMENT is a permutation where no element stays in place.
For coloring: no adjacent diagrams can have the "same" color.

We use a greedy graph coloring with derangement constraint:
- Start with most-connected node
- Assign color different from all neighbors
- Ensure coloring is a derangement of the identity
"""
struct DerangementColoring
    node_colors::Dict{Int, Int}  # node_id → color_class
    n_colors::Int
    is_valid_derangement::Bool
end

function derangement_coloring(
    nodes::Vector{DiagramNode},
    adjacency::Matrix{Float64}
)::DerangementColoring
    
    n = length(nodes)
    node_colors = Dict{Int, Int}()
    
    # Sort nodes by degree (most connected first)
    degrees = [(i, count(x -> x < Inf && x > 0, adjacency[i, :])) for i in 1:n]
    sort!(degrees, by=x -> -x[2])
    
    # Greedy coloring
    for (node_id, _) in degrees
        # Find colors used by neighbors
        neighbor_colors = Set{Int}()
        for j in 1:n
            if adjacency[node_id, j] < Inf && adjacency[node_id, j] > 0
                if haskey(node_colors, j)
                    push!(neighbor_colors, node_colors[j])
                end
            end
        end
        
        # Derangement constraint: can't use color = node_id
        push!(neighbor_colors, node_id)
        
        # Find smallest available color
        color = 1
        while color in neighbor_colors
            color += 1
        end
        
        node_colors[node_id] = color
    end
    
    n_colors = maximum(values(node_colors))
    
    # Verify derangement: no node has color = its id
    is_valid = all(node_id != color for (node_id, color) in node_colors)
    
    DerangementColoring(node_colors, n_colors, is_valid)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Collage Generator
# ═══════════════════════════════════════════════════════════════════════════════

struct CollageResult
    nodes::Vector{DiagramNode}
    adjacency::Matrix{Float64}
    shortest_paths::Matrix{Float64}
    regrets::Vector{CounterfactualRegret}
    additions::Vector{Tuple{Int, Int, Float64}}
    coloring::DerangementColoring
    total_regret::Float64
    regret_after::Float64
end

function generate_collage(seed::UInt64, n_diagrams::Int)::CollageResult
    println("═"^70)
    println("  DIAGRAM COLLAGE: Tropical Semiring × Counterfactual Regret")
    println("═"^70)
    println("  Seed: 0x$(string(seed, base=16))")
    println("  Diagrams: $n_diagrams")
    println("═"^70)
    
    # Generate diagram nodes
    println("\n▸ Generating $n_diagrams diagram nodes...")
    nodes = [DiagramNode(i, seed) for i in 1:n_diagrams]
    
    # Show type distribution
    type_counts = Dict{DiagramType, Int}()
    for n in nodes
        type_counts[n.type] = get(type_counts, n.type, 0) + 1
    end
    println("  Types: ", join(["$(DIAGRAM_SYMBOLS[t]): $c" for (t, c) in type_counts], ", "))
    
    # Build tropical adjacency matrix
    println("\n▸ Building tropical (min, +) adjacency matrix...")
    adjacency = tropical_adjacency(nodes)
    n_edges = count(x -> 0 < x < Inf, adjacency) ÷ 2
    println("  Edges: $n_edges / $(n_diagrams * (n_diagrams - 1) ÷ 2) possible")
    
    # Compute all-pairs shortest paths
    println("\n▸ Computing tropical shortest paths (Floyd-Warshall)...")
    shortest = tropical_shortest_paths(adjacency)
    reachable = count(x -> x < Inf, shortest) - n_diagrams
    println("  Reachable pairs: $(reachable ÷ 2)")
    
    # Compute counterfactual regrets
    println("\n▸ Computing counterfactual regret over missed Galois connections...")
    gc = GaloisConnection(Int(seed % typemax(Int)))
    regrets = compute_regrets(nodes, adjacency, shortest, gc)
    total_regret_before = sum(r.regret for r in regrets)
    println("  Regrets found: $(length(regrets))")
    println("  Total regret: $(round(total_regret_before, digits=2))")
    
    # Show top regrets
    println("\n  Top 5 regrets:")
    for r in regrets[1:min(5, length(regrets))]
        galois = r.galois_closure ? "✓" : "✗"
        println("    D$(r.from_id) ↔ D$(r.to_id): regret=$(round(r.regret, digits=2)) galois=$galois")
    end
    
    # Minimize regret by adding connections
    println("\n▸ Minimizing regret (adding Galois-closed connections)...")
    additions = minimize_regret!(adjacency, regrets; max_additions=15)
    println("  Added $(length(additions)) connections")
    
    # Recompute regrets
    new_regrets = compute_regrets(nodes, adjacency, shortest, gc)
    total_regret_after = sum(r.regret for r in new_regrets)
    reduction = 100 * (1 - total_regret_after / max(1, total_regret_before))
    println("  Regret after: $(round(total_regret_after, digits=2)) ($(round(reduction, digits=1))% reduction)")
    
    # Derangement coloring
    println("\n▸ Computing derangement coloring...")
    coloring = derangement_coloring(nodes, adjacency)
    println("  Colors used: $(coloring.n_colors)")
    println("  Valid derangement: $(coloring.is_valid_derangement ? "✓ YES" : "✗ NO")")
    
    CollageResult(
        nodes, adjacency, shortest, regrets, additions,
        coloring, total_regret_before, total_regret_after
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# Output Generators
# ═══════════════════════════════════════════════════════════════════════════════

function render_ascii_collage(result::CollageResult)::String
    nodes = result.nodes
    coloring = result.coloring
    
    lines = String[]
    push!(lines, "")
    push!(lines, "DIAGRAM COLLAGE ($(length(nodes)) diagrams, $(coloring.n_colors) colors)")
    push!(lines, "═"^60)
    push!(lines, "")
    
    # Group by color class
    by_color = Dict{Int, Vector{DiagramNode}}()
    for n in nodes
        c = coloring.node_colors[n.id]
        push!(get!(Vector{DiagramNode}, by_color, c), n)
    end
    
    for color_class in sort(collect(keys(by_color)))
        dns = by_color[color_class]
        line = "Color $color_class: "
        for d in dns[1:min(8, length(dns))]
            sym = DIAGRAM_SYMBOLS[d.type]
            ansi = lch_to_ansi(d.color)
            line *= "$ansi $(d.label)($sym) "
        end
        if length(dns) > 8
            line *= "... +$(length(dns) - 8) more"
        end
        push!(lines, line)
    end
    
    push!(lines, "")
    push!(lines, "═"^60)
    
    join(lines, "\n")
end

function generate_markdown_docs(result::CollageResult)::String
    lines = String[]
    
    push!(lines, "# Diagram Collage")
    push!(lines, "")
    push!(lines, "Generated: $(Dates.now())")
    push!(lines, "Seed: `0x$(string(COLLAGE_SEED, base=16))`")
    push!(lines, "")
    push!(lines, "## Overview")
    push!(lines, "")
    push!(lines, "| Metric | Value |")
    push!(lines, "|--------|-------|")
    push!(lines, "| Diagrams | $(length(result.nodes)) |")
    push!(lines, "| Colors used | $(result.coloring.n_colors) |")
    push!(lines, "| Valid derangement | $(result.coloring.is_valid_derangement) |")
    push!(lines, "| Total regret (before) | $(round(result.total_regret, digits=2)) |")
    push!(lines, "| Total regret (after) | $(round(result.regret_after, digits=2)) |")
    push!(lines, "| Connections added | $(length(result.additions)) |")
    push!(lines, "")
    push!(lines, "## Tropical Semiring")
    push!(lines, "")
    push!(lines, "Uses `(min, +)` semiring for shortest path optimization:")
    push!(lines, "- ⊕ = min (choose shortest path)")
    push!(lines, "- ⊗ = + (accumulate distance)")
    push!(lines, "")
    push!(lines, "## Counterfactual Regret")
    push!(lines, "")
    push!(lines, "Top missed Galois connections:")
    push!(lines, "")
    push!(lines, "| From | To | Regret | Galois |")
    push!(lines, "|------|-----|--------|--------|")
    for r in result.regrets[1:min(10, length(result.regrets))]
        galois = r.galois_closure ? "✓" : "✗"
        push!(lines, "| D$(r.from_id) | D$(r.to_id) | $(round(r.regret, digits=2)) | $galois |")
    end
    push!(lines, "")
    push!(lines, "## Diagram Types")
    push!(lines, "")
    type_counts = Dict{DiagramType, Int}()
    for n in result.nodes
        type_counts[n.type] = get(type_counts, n.type, 0) + 1
    end
    for (t, c) in sort(collect(type_counts), by=x -> -x[2])
        push!(lines, "- $(DIAGRAM_SYMBOLS[t]) ($t): $c diagrams")
    end
    
    join(lines, "\n")
end

# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

function main()
    # Generate collage
    result = generate_collage(COLLAGE_SEED, N_DIAGRAMS)
    
    # Render ASCII
    println("\n" * render_ascii_collage(result))
    
    # Save to docs
    mkpath(DOCS_DIR)
    
    md_content = generate_markdown_docs(result)
    md_path = joinpath(DOCS_DIR, "index.md")
    open(md_path, "w") do f
        write(f, md_content)
    end
    println("\n▸ Saved docs to $md_path")
    
    # Summary
    println("\n" * "═"^70)
    println("  COLLAGE COMPLETE")
    println("═"^70)
    println("  • $(length(result.nodes)) diagrams generated")
    println("  • $(result.coloring.n_colors) colors (derangement: $(result.coloring.is_valid_derangement))")
    println("  • Regret reduced from $(round(result.total_regret, digits=2)) → $(round(result.regret_after, digits=2))")
    println("  • $(length(result.additions)) Galois connections added")
    println("═"^70)
    
    result
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
