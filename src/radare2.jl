# Gay.jl Radare2 Integration: Color binary analysis
#
# Uses radare2 MCP tools to analyze binaries and colors:
# - Functions by address hash (deterministic)
# - Basic blocks by control flow (checkerboard for parallel analysis)
# - Xrefs by XOR parity (like Heisenberg bonds)
# - Decompiled code by AST depth (like GaySexpr)
#
# Integration with tree-sitter for pseudocode parsing.
#
# Note: This file is included after splittable.jl which defines:
#   GayRng, gay_seed!, gay_rng, next_color, color_at, GayInterleaver,
#   gay_sublattice, gay_paren_color, etc.

export ColoredBinary, ColoredFunction, ColoredBlock, ColoredXref
export analyze_binary, color_functions, color_xrefs, color_decompiled
export render_colored_function, render_colored_disasm
export GayR2, r2_seed!, r2_color_at

# ═══════════════════════════════════════════════════════════════════════════
# Radare2 Analysis Types
# ═══════════════════════════════════════════════════════════════════════════

"""
    ColoredFunction

A function with Gay.jl deterministic coloring.
"""
struct ColoredFunction
    address::UInt64
    name::Union{String, Nothing}
    size::Union{Int, Nothing}
    color::RGB
    callees::Vector{UInt64}
    callers::Vector{UInt64}
end

"""
    ColoredBlock

A basic block with parity-based coloring for parallel CFG analysis.
"""
struct ColoredBlock
    address::UInt64
    size::Union{Int, Nothing}
    instructions::Union{Int, Nothing}
    color::RGB
    parity::Int  # 0 or 1 for checkerboard
    successors::Vector{UInt64}
    predecessors::Vector{UInt64}
end

"""
    ColoredXref

Cross-reference with XOR coloring (like Heisenberg exchange bonds).
"""
struct ColoredXref
    from_addr::UInt64
    to_addr::UInt64
    xref_type::Union{String, Nothing}
    color::RGB
    parity::Int  # (from ⊻ to) & 1
end

"""
    ColoredBinary

Full binary analysis with Gay.jl coloring throughout.
"""
mutable struct ColoredBinary
    path::String
    functions::Vector{ColoredFunction}
    blocks::Vector{ColoredBlock}
    xrefs::Vector{ColoredXref}
    strings::Vector{Tuple{UInt64, String, RGB}}
    seed::UInt64
    analysis_level::Int
end

# ═══════════════════════════════════════════════════════════════════════════
# R2-specific seed (uses color_at for O(1) access)
# ═══════════════════════════════════════════════════════════════════════════

const R2_SEED = UInt64(0x7232636f6c6f7273)  # "r2colors"
const CURRENT_R2_SEED = Ref{UInt64}(R2_SEED)

function r2_seed!(seed::Integer)
    CURRENT_R2_SEED[] = UInt64(seed)
    return seed
end

function r2_color_at(address::UInt64; seed::UInt64=R2_SEED)
    # Use address as index for deterministic coloring
    index = (address % 0xFFFF) |> Int
    return color_at(index, SRGB(); seed=seed)
end

# ═══════════════════════════════════════════════════════════════════════════
# Function Coloring
# ═══════════════════════════════════════════════════════════════════════════

"""
    color_function(addr, name, size; seed=R2_SEED)

Color a function based on its address (deterministic).
"""
function color_function(addr::UInt64, name::Union{String,Nothing}=nothing, 
                        size::Union{Int,Nothing}=nothing;
                        seed::UInt64=R2_SEED,
                        callees::Vector{UInt64}=UInt64[],
                        callers::Vector{UInt64}=UInt64[])
    color = r2_color_at(addr; seed=seed)
    ColoredFunction(addr, name, size, color, callees, callers)
end

"""
    color_functions(funcs; seed=R2_SEED)

Color a list of (address, name, size) tuples.
"""
function color_functions(funcs::Vector{<:Tuple}; seed::UInt64=R2_SEED)
    [color_function(UInt64(f[1]), 
                    length(f) > 1 ? f[2] : nothing,
                    length(f) > 2 ? f[3] : nothing;
                    seed=seed) for f in funcs]
end

# ═══════════════════════════════════════════════════════════════════════════
# Basic Block Coloring (Checkerboard for CFG)
# ═══════════════════════════════════════════════════════════════════════════

"""
    color_blocks(blocks, func_addr; seed=R2_SEED)

Color basic blocks using checkerboard pattern for parallel CFG traversal.
Blocks with even parity can be analyzed in parallel, then odd parity.
"""
function color_blocks(blocks::Vector{<:Tuple}, func_addr::UInt64; seed::UInt64=R2_SEED)
    il = GayInterleaver(seed ⊻ func_addr, 2)  # 2 sublattices
    
    result = ColoredBlock[]
    for (i, block) in enumerate(blocks)
        addr = UInt64(block[1])
        size = length(block) > 1 ? block[2] : nothing
        instrs = length(block) > 2 ? block[3] : nothing
        succs = length(block) > 3 ? block[4] : UInt64[]
        preds = length(block) > 4 ? block[5] : UInt64[]
        
        # Parity based on block position in CFG
        parity = i % 2
        color = gay_sublattice(il, parity)
        
        push!(result, ColoredBlock(addr, size, instrs, color, parity, succs, preds))
    end
    
    result
end

# ═══════════════════════════════════════════════════════════════════════════
# Xref Coloring (XOR like Heisenberg bonds)
# ═══════════════════════════════════════════════════════════════════════════

"""
    color_xref(from_addr, to_addr, xref_type; seed=R2_SEED)

Color a cross-reference using XOR parity (like J_ij bonds).
"""
function color_xref(from_addr::UInt64, to_addr::UInt64, 
                    xref_type::Union{String,Nothing}=nothing;
                    seed::UInt64=R2_SEED)
    parity = Int((from_addr ⊻ to_addr) & 1)
    il = GayInterleaver(seed, 2)
    color = gay_sublattice(il, parity)
    ColoredXref(from_addr, to_addr, xref_type, color, parity)
end

"""
    color_xrefs(xrefs; seed=R2_SEED)

Color a list of (from, to, type) tuples.
"""
function color_xrefs(xrefs::Vector{<:Tuple}; seed::UInt64=R2_SEED)
    [color_xref(UInt64(x[1]), UInt64(x[2]), 
                length(x) > 2 ? x[3] : nothing;
                seed=seed) for x in xrefs]
end

# ═══════════════════════════════════════════════════════════════════════════
# Decompiled Code Coloring (like GaySexpr)
# ═══════════════════════════════════════════════════════════════════════════

"""
    ColoredAST

AST node from decompiled pseudocode with Gay.jl coloring.
"""
struct ColoredAST
    kind::String
    content::Union{String, Nothing}
    color::RGB
    spin::Int  # ±1 like Ising
    depth::Int
    children::Vector{ColoredAST}
end

"""
    color_ast(kind, content, depth, position; seed=R2_SEED)

Color an AST node by depth and position (like GaySexpr).
"""
function color_ast(kind::String, content::Union{String,Nothing}, 
                   depth::Int, position::Int;
                   seed::UInt64=R2_SEED, 
                   children::Vector{ColoredAST}=ColoredAST[])
    color = gay_paren_color(seed, depth, position)
    spin = ((depth ⊻ position) & 1 == 0) ? 1 : -1
    ColoredAST(kind, content, color, spin, depth, children)
end

"""
    color_decompiled(pseudocode; seed=R2_SEED)

Parse and color decompiled pseudocode.
Returns a ColoredAST tree.

This is a simplified parser - real implementation would use tree-sitter.
"""
function color_decompiled(pseudocode::String; seed::UInt64=R2_SEED)
    # Simplified: split into tokens and create flat AST
    tokens = Base.split(pseudocode)
    pos_counter = Ref(0)
    
    children = ColoredAST[]
    for token in tokens
        pos = pos_counter[]
        pos_counter[] += 1
        push!(children, color_ast("token", String(token), 1, pos; seed=seed))
    end
    
    color_ast("function", nothing, 0, 0; seed=seed, children=children)
end

"""
    render_colored_ast(ast)

Render ColoredAST with ANSI colors (like gay_render_sexpr).
"""
function render_colored_ast(ast::ColoredAST)
    R = "\e[0m"
    rgb = convert(RGB, ast.color)
    r = round(Int, clamp(rgb.r, 0, 1) * 255)
    g = round(Int, clamp(rgb.g, 0, 1) * 255)
    b = round(Int, clamp(rgb.b, 0, 1) * 255)
    fg = "\e[38;2;$(r);$(g);$(b)m"
    
    spin_char = ast.spin > 0 ? "⁺" : "⁻"
    
    if isempty(ast.children)
        return "$(fg)$(ast.content)$(R)"
    else
        inner = join([render_colored_ast(c) for c in ast.children], " ")
        return "$(fg)($(spin_char)$(R)$(inner)$(fg))$(R)"
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# Terminal Rendering
# ═══════════════════════════════════════════════════════════════════════════

"""
    render_colored_function(func)

Render a colored function for terminal display.
"""
function render_colored_function(func::ColoredFunction)
    R = "\e[0m"
    rgb = convert(RGB, func.color)
    r = round(Int, clamp(rgb.r, 0, 1) * 255)
    g = round(Int, clamp(rgb.g, 0, 1) * 255)
    b = round(Int, clamp(rgb.b, 0, 1) * 255)
    fg = "\e[38;2;$(r);$(g);$(b)m"
    
    name = something(func.name, "sub_$(string(func.address, base=16))")
    size_str = isnothing(func.size) ? "" : " ($(func.size) bytes)"
    
    "$(fg)$(name)$(R) @ 0x$(string(func.address, base=16))$(size_str)"
end

"""
    render_colored_xref(xref)

Render a colored xref for terminal display.
"""
function render_colored_xref(xref::ColoredXref)
    R = "\e[0m"
    rgb = convert(RGB, xref.color)
    r = round(Int, clamp(rgb.r, 0, 1) * 255)
    g = round(Int, clamp(rgb.g, 0, 1) * 255)
    b = round(Int, clamp(rgb.b, 0, 1) * 255)
    fg = "\e[38;2;$(r);$(g);$(b)m"
    
    type_str = something(xref.xref_type, "ref")
    parity_char = xref.parity == 0 ? "●" : "○"
    
    "$(fg)$(parity_char)$(R) 0x$(string(xref.from_addr, base=16)) $(fg)→$(R) 0x$(string(xref.to_addr, base=16)) [$(type_str)]"
end

"""
    render_colored_disasm(instructions)

Render colored disassembly listing.
"""
function render_colored_disasm(instructions::Vector{<:Tuple}; seed::UInt64=R2_SEED)
    lines = String[]
    for (i, instr) in enumerate(instructions)
        addr = UInt64(instr[1])
        mnemonic = instr[2]
        operands = length(instr) > 2 ? instr[3] : ""
        
        color = r2_color_at(addr; seed=seed)
        rgb = convert(RGB, color)
        r = round(Int, clamp(rgb.r, 0, 1) * 255)
        g = round(Int, clamp(rgb.g, 0, 1) * 255)
        b = round(Int, clamp(rgb.b, 0, 1) * 255)
        fg = "\e[38;2;$(r);$(g);$(b)m"
        R = "\e[0m"
        
        push!(lines, "$(fg)0x$(string(addr, base=16, pad=8))$(R)  $(mnemonic) $(operands)")
    end
    join(lines, "\n")
end

# ═══════════════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════════════

function demo_radare2_colors()
    println("\n╔══════════════════════════════════════════════════════════════╗")
    println("║  Gay.jl + Radare2: Colored Binary Analysis                   ║")
    println("╚══════════════════════════════════════════════════════════════╝\n")
    
    seed = UInt64(0xDEADBEEF)
    
    # Demo functions
    println("Colored Functions:")
    funcs = [
        (0x00401000, "main", 256),
        (0x00401100, "parse_input", 128),
        (0x00401200, "process_data", 512),
        (0x00401400, "write_output", 64),
    ]
    
    colored = color_functions(funcs; seed=seed)
    for f in colored
        println("  ", render_colored_function(f))
    end
    
    println()
    println("Colored Xrefs (XOR parity):")
    xrefs = [
        (0x00401050, 0x00401100, "call"),
        (0x00401060, 0x00401200, "call"),
        (0x00401150, 0x00401400, "call"),
        (0x00401250, 0x00401100, "call"),
    ]
    
    colored_xrefs = color_xrefs(xrefs; seed=seed)
    for x in colored_xrefs
        println("  ", render_colored_xref(x))
    end
    
    println()
    println("Colored Disassembly:")
    disasm = [
        (0x00401000, "push", "rbp"),
        (0x00401001, "mov", "rbp, rsp"),
        (0x00401004, "sub", "rsp, 0x20"),
        (0x00401008, "call", "0x401100"),
        (0x0040100d, "mov", "edi, eax"),
        (0x0040100f, "call", "0x401200"),
    ]
    println(render_colored_disasm(disasm; seed=seed))
    
    println()
    println("Colored Decompiled Code:")
    pseudocode = "int main(int argc, char **argv) { return process(argc); }"
    ast = color_decompiled(pseudocode; seed=seed)
    println("  ", render_colored_ast(ast))
    
    println()
    println("Usage with radare2 MCP:")
    println("  1. mcp__radare2__open_file binary")
    println("  2. mcp__radare2__analyze level=2")
    println("  3. mcp__radare2__list_functions → color_functions()")
    println("  4. mcp__radare2__decompile_function → color_decompiled()")
    println("  5. mcp__radare2__xrefs_to → color_xrefs()")
end

export demo_radare2_colors
