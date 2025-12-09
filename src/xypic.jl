# XY-pic diagram generation for Gay.jl
# Exports color chains, lattices, and state machines as LaTeX diagrams

using Colors

export gay_xypic_trajectory, gay_xypic_lattice, gay_xypic_statebox
export gay_xypic_preamble, gay_xypic_document
export gay_xypic_lattice_bonds, gay_xypic_morphism_chain, gay_xypic_sexpr
export gay_save_xypic

# ═══════════════════════════════════════════════════════════════════════════
# LaTeX Preamble & Document Wrappers
# ═══════════════════════════════════════════════════════════════════════════

"""
    gay_xypic_preamble()

Return the LaTeX preamble needed for xy-pic diagrams with color support.
"""
function gay_xypic_preamble()
    return raw"""
\usepackage[all,2cell,dvips]{xy}
\usepackage{xcolor}
\UseTwocells
\xyoption{arrow}
\xyoption{matrix}
\xyoption{curve}
\xyoption{frame}
"""
end

"""
    gay_xypic_document(body::String; standalone::Bool=true)

Wrap xy-pic diagram code in a complete LaTeX document.
"""
function gay_xypic_document(body::String; standalone::Bool=true)
    if standalone
        return """
\\documentclass{standalone}
\\usepackage[all,2cell]{xy}
\\usepackage{xcolor}
\\UseTwocells

\\begin{document}
$body
\\end{document}
"""
    else
        return body
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# Color Utilities for LaTeX
# ═══════════════════════════════════════════════════════════════════════════

"""
    _rgb_to_latex(c::Color, name::String)

Generate LaTeX color definition from RGB color.
"""
function _rgb_to_latex(c::Color, name::String)
    rgb = convert(RGB, c)
    r = clamp(rgb.r, 0, 1)
    g = clamp(rgb.g, 0, 1)
    b = clamp(rgb.b, 0, 1)
    return "\\definecolor{$name}{rgb}{$(round(r, digits=3)),$(round(g, digits=3)),$(round(b, digits=3))}"
end

"""
    _hex_color(c::Color)

Convert color to hex string for inline use.
"""
function _hex_color(c::Color)
    rgb = convert(RGB, c)
    r = round(Int, clamp(rgb.r, 0, 1) * 255)
    g = round(Int, clamp(rgb.g, 0, 1) * 255)
    b = round(Int, clamp(rgb.b, 0, 1) * 255)
    return uppercase(string(r, base=16, pad=2) * string(g, base=16, pad=2) * string(b, base=16, pad=2))
end

# ═══════════════════════════════════════════════════════════════════════════
# LCH Trajectory as State Diagram
# ═══════════════════════════════════════════════════════════════════════════

"""
    gay_xypic_trajectory(seed::Integer=GAY_SEED, n::Int=14; 
                         show_lch::Bool=true, 
                         arrow_style::String="@{->}")

Generate xy-pic state diagram showing the deterministic color chain evolution.

Each state shows the cycle number and optionally LCH values.
Arrows represent SplitMix64 split() transitions.

# Example
```julia
tex = gay_xypic_trajectory(0x6761795f636f6c6f, 10)
write("trajectory.tex", gay_xypic_document(tex))
```
"""
function gay_xypic_trajectory(seed::Integer=GAY_SEED, n::Int=14;
                               show_lch::Bool=true,
                               arrow_style::String="@{->}")
    colors = Color[]
    lch_values = Tuple{Float64,Float64,Float64}[]
    
    # Generate the color chain
    gr = GayRNG(seed)
    for i in 1:n
        c = next_color(SRGB(); gr=gr)
        push!(colors, c)
        lch = convert(LCHab, c)
        push!(lch_values, (lch.l, lch.c, lch.h))
    end
    
    # Build color definitions
    color_defs = String[]
    for (i, c) in enumerate(colors)
        push!(color_defs, _rgb_to_latex(c, "gay$i"))
    end
    
    # Build xy-pic matrix
    # Arrange in rows of 4 for readability
    cols_per_row = 4
    rows = ceil(Int, n / cols_per_row)
    
    lines = String[]
    push!(lines, "% Color definitions")
    append!(lines, color_defs)
    push!(lines, "")
    push!(lines, "% State diagram: SplitMix64 color chain")
    push!(lines, "% Seed: 0x$(string(seed, base=16))")
    push!(lines, raw"\xymatrix{")
    
    idx = 1
    for row in 1:rows
        row_items = String[]
        for col in 1:cols_per_row
            if idx <= n
                L, C, H = lch_values[idx]
                if show_lch
                    label = "C_{$idx}" * raw"\\" * "\\scriptstyle L=$(round(L,digits=1))"
                else
                    label = "C_{$idx}"
                end
                # Color the node with its generated color
                node = "*+[F-,]{\\textcolor{gay$idx}{$label}}"
                push!(row_items, node)
                idx += 1
            end
        end
        
        # Add arrows between nodes in this row
        row_str = join(row_items, " \\ar[r]^{s} & ")
        
        # Add down arrow from last item if not last row
        if row < rows && idx <= n + 1
            row_str *= " \\ar[d]^{s}"
        end
        
        push!(lines, "  " * row_str * raw" \\")
    end
    
    push!(lines, "}")
    
    return join(lines, "\n")
end

# ═══════════════════════════════════════════════════════════════════════════
# 2D Lattice Coloring
# ═══════════════════════════════════════════════════════════════════════════

"""
    gay_xypic_lattice(il::GayInterleaver, Lx::Int, Ly::Int;
                      show_coords::Bool=false,
                      node_size::String="*+[o][F-]")

Generate xy-pic diagram of a 2D checkerboard lattice coloring.

# Example
```julia
il = GayInterleaver(42, 2)
tex = gay_xypic_lattice(il, 4, 4)
```
"""
function gay_xypic_lattice(il::GayInterleaver, Lx::Int, Ly::Int;
                           show_coords::Bool=false,
                           node_size::String="*+[o][F-]")
    # Generate colors for each site
    colors = gay_checkerboard_2d(il, Lx, Ly)
    
    # Color definitions
    color_defs = String[]
    for i in 1:Lx, j in 1:Ly
        name = "site$(i)_$(j)"
        push!(color_defs, _rgb_to_latex(colors[i,j], name))
    end
    
    lines = String[]
    push!(lines, "% Checkerboard lattice coloring")
    push!(lines, "% Seed: 0x$(string(il.seed, base=16)), streams: $(il.n_streams)")
    append!(lines, color_defs)
    push!(lines, "")
    push!(lines, raw"\xymatrix@=1.5em{")
    
    # Build grid (note: xy-pic y increases downward, so we reverse)
    for j in Ly:-1:1
        row_items = String[]
        for i in 1:Lx
            name = "site$(i)_$(j)"
            if show_coords
                label = "($i,$j)"
            else
                label = "\\bullet"
            end
            node = "$node_size{\\textcolor{$name}{$label}}"
            push!(row_items, node)
        end
        
        # Add horizontal bonds
        row_str = join(row_items, " \\ar@{-}[r] & ")
        
        # Add vertical bonds (down arrows, which go up visually)
        if j > 1
            # Add vertical connections
            row_str *= raw" \\"
        end
        
        push!(lines, "  " * row_str)
    end
    
    push!(lines, "}")
    
    return join(lines, "\n")
end

"""
    gay_xypic_lattice_bonds(il::GayInterleaver, Lx::Int, Ly::Int)

Generate xy-pic diagram showing colored bonds (edges) for Heisenberg model.
Each J_ij bond is colored based on XOR parity.
"""
function gay_xypic_lattice_bonds(il::GayInterleaver, Lx::Int, Ly::Int)
    bonds = gay_heisenberg_bonds(il, Lx, Ly)
    
    # Collect unique bond colors
    color_defs = String[]
    bond_colors = Dict{Tuple{Tuple{Int,Int},Tuple{Int,Int}}, String}()
    color_idx = 1
    
    for i in 1:Lx, j in 1:Ly
        for ((jx, jy), color) in bonds[(i,j)]
            key = ((i,j), (jx,jy))
            name = "bond$color_idx"
            push!(color_defs, _rgb_to_latex(color, name))
            bond_colors[key] = name
            color_idx += 1
        end
    end
    
    lines = String[]
    push!(lines, "% Heisenberg bond coloring")
    push!(lines, "% J_ij colored by (i⊕j) parity")
    append!(lines, color_defs)
    push!(lines, "")
    push!(lines, raw"\xymatrix@=2em{")
    
    # Build grid with colored edges
    for j in Ly:-1:1
        row_items = String[]
        for i in 1:Lx
            node = "*+[o][F-]{\\scriptstyle ($i,$j)}"
            push!(row_items, node)
        end
        
        # Build row with horizontal arrows
        parts = String[]
        for (idx, item) in enumerate(row_items)
            push!(parts, item)
            if idx < Lx
                # Horizontal bond
                i = idx
                jx = i + 1
                key = ((i, j), (jx, j))
                if haskey(bond_colors, key)
                    cname = bond_colors[key]
                    push!(parts, "\\ar@[$(cname)]@{-}[r]")
                end
            end
        end
        
        row_str = join(parts, " & ")
        
        if j > 1
            row_str *= raw" \\"
        end
        
        push!(lines, "  " * row_str)
    end
    
    push!(lines, "}")
    
    return join(lines, "\n")
end

# ═══════════════════════════════════════════════════════════════════════════
# Statebox-Style Petri Net / Open Game Diagram
# ═══════════════════════════════════════════════════════════════════════════

"""
    gay_xypic_statebox(seed::Integer=GAY_SEED, n::Int=8;
                       title::String="SplitMix64 Color Chain")

Generate a Statebox/categorical diagram showing the color chain as an open system.

Structure:
- Input wire (seed) → Box (SplitMix64) → Output wires (colors)
- Each split() is a morphism in the category of stochastic processes

# Example
```julia
tex = gay_xypic_statebox(0x6761795f636f6c6f, 6)
```
"""
function gay_xypic_statebox(seed::Integer=GAY_SEED, n::Int=8;
                            title::String="SplitMix64 Color Chain")
    # Generate colors
    gr = GayRNG(seed)
    colors = [next_color(SRGB(); gr=gr) for _ in 1:n]
    
    # Color definitions
    color_defs = [_rgb_to_latex(c, "out$i") for (i, c) in enumerate(colors)]
    
    lines = String[]
    push!(lines, "% Statebox diagram: $title")
    push!(lines, "% Seed: 0x$(string(seed, base=16))")
    append!(lines, color_defs)
    push!(lines, "")
    
    # Build the open diagram
    # Input → [SplitMix64] → (split) → [LCH] → (sample) → outputs
    push!(lines, raw"\xymatrix@C=3em@R=1.5em{")
    
    # Top row: seed input
    push!(lines, raw"  & *+[F=]{seed} \ar[d]^{\iota} & \\")
    
    # SplitMix64 box
    push!(lines, raw"  & *+[F]{SplitMix64} \ar[dl]_{s} \ar[d]^{s} \ar[dr]^{s} & \\")
    
    # Split outputs (show first 3)
    push!(lines, raw"  *+[F]{LCH_1} \ar[d] & *+[F]{LCH_2} \ar[d] & *+[F]{\cdots} \ar[d] \\")
    
    # Color outputs
    out_row = String[]
    for i in 1:min(3, n)
        push!(out_row, "*+[o][F-]{\\textcolor{out$i}{\\bullet}}")
    end
    push!(lines, "  " * join(out_row, " & ") * raw" \\")
    
    # Labels
    push!(lines, raw"  c_1 & c_2 & c_n")
    
    push!(lines, "}")
    
    return join(lines, "\n")
end

"""
    gay_xypic_morphism_chain(seed::Integer=GAY_SEED, n::Int=6)

Generate a categorical diagram showing split() as morphisms.

    seed → s₁ → s₂ → ... → sₙ
           ↓     ↓         ↓
           c₁    c₂        cₙ
"""
function gay_xypic_morphism_chain(seed::Integer=GAY_SEED, n::Int=6)
    gr = GayRNG(seed)
    colors = [next_color(SRGB(); gr=gr) for _ in 1:n]
    color_defs = [_rgb_to_latex(c, "c$i") for (i, c) in enumerate(colors)]
    
    lines = String[]
    push!(lines, "% Morphism chain: split() as arrows")
    append!(lines, color_defs)
    push!(lines, "")
    push!(lines, raw"\xymatrix@C=2.5em{")
    
    # Top row: state sequence
    states = ["*+[F]{s_0}"]
    for i in 1:n
        push!(states, "\\ar[r]^{\\mathsf{split}}")
        push!(states, "*+[F]{s_$i}")
    end
    push!(lines, "  " * join(states, " ") * raw" \\")
    
    # Bottom row: color outputs
    outputs = [""]  # empty under s_0
    for i in 1:n
        push!(outputs, "")  # space for arrow
        push!(outputs, "\\ar[u]_{\\pi} *+[o][F-]{\\textcolor{c$i}{c_$i}}")
    end
    push!(lines, "  " * join(outputs, " & "))
    
    push!(lines, "}")
    
    return join(lines, "\n")
end

# ═══════════════════════════════════════════════════════════════════════════
# S-Expression Tree Diagram
# ═══════════════════════════════════════════════════════════════════════════

"""
    gay_xypic_sexpr(gs::GaySexpr; max_depth::Int=4)

Generate xy-pic tree diagram of a magnetized S-expression.
Each node is colored by its deterministic color, labeled with spin ±.
"""
function gay_xypic_sexpr(gs::GaySexpr; max_depth::Int=4)
    color_defs = String[]
    color_idx = Ref(1)
    
    function collect_colors(node::GaySexpr, depth::Int)
        if depth > max_depth
            return
        end
        name = "node$(color_idx[])"
        push!(color_defs, _rgb_to_latex(node.color, name))
        color_idx[] += 1
        for child in node.children
            collect_colors(child, depth + 1)
        end
    end
    
    collect_colors(gs, 0)
    
    lines = String[]
    push!(lines, "% S-expression tree with spin coloring")
    append!(lines, color_defs)
    push!(lines, "")
    push!(lines, "% Tree structure (simplified)")
    push!(lines, raw"\xymatrix@R=1.5em@C=1em{")
    
    # Simplified: just show root and immediate children
    spin_char = gs.spin > 0 ? "+" : "-"
    root_node = "*+[F]{\\textcolor{node1}{\\sigma^{$spin_char}}}"
    
    if length(gs.children) > 0
        child_arrows = ["\\ar[d]" for _ in 1:min(3, length(gs.children))]
        push!(lines, "  & $root_node $(join(child_arrows, " ")) & \\\\")
        
        # Children
        child_nodes = String[]
        for (i, child) in enumerate(gs.children[1:min(3, end)])
            s = child.spin > 0 ? "+" : "-"
            push!(child_nodes, "*+[F]{\\textcolor{node$(i+1)}{\\sigma^{$s}}}")
        end
        if length(gs.children) > 3
            push!(child_nodes, "\\cdots")
        end
        push!(lines, "  " * join(child_nodes, " & "))
    else
        push!(lines, "  $root_node")
    end
    
    push!(lines, "}")
    
    return join(lines, "\n")
end

# ═══════════════════════════════════════════════════════════════════════════
# File Output Helpers
# ═══════════════════════════════════════════════════════════════════════════

"""
    gay_save_xypic(filename::String, content::String; standalone::Bool=true)

Save xy-pic diagram to a .tex file.
"""
function gay_save_xypic(filename::String, content::String; standalone::Bool=true)
    doc = gay_xypic_document(content; standalone=standalone)
    write(filename, doc)
    return filename
end
