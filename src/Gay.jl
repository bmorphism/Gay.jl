module Gay

# Re-export LispSyntax for the Lisp REPL
using LispSyntax
export sx, desx, codegen, @lisp_str, assign_reader_dispatch, include_lisp

# Color dependencies
using Colors
using ColorTypes
using Random
using SplittableRandoms

# Include wide-gamut color space support
include("colorspaces.jl")

# Include splittable RNG for deterministic color generation
include("splittable.jl")
export color_at, colors_at, palette_at, GAY_SEED

# Include custom REPL
include("repl.jl")

# ═══════════════════════════════════════════════════════════════════════════
# Lisp bindings for color operations
# ═══════════════════════════════════════════════════════════════════════════

"""
Lisp-accessible random color generation.

Usage from Lisp REPL:
  (random-color)              ; Random sRGB color
  (random-color :p3)          ; Random Display P3 color  
  (random-color :rec2020)     ; Random Rec.2020 color
  (random-colors 5)           ; 5 random sRGB colors
  (random-palette 8)          ; 8 visually distinct colors
  (pride :rainbow)            ; Rainbow flag colors
  (pride :trans :p3)          ; Trans flag in P3 gamut
"""

# Symbol to ColorSpace mapping for Lisp interface
function sym_to_colorspace(s::Symbol)
    if s == :srgb || s == :SRGB
        return SRGB()
    elseif s == :p3 || s == :P3 || s == :displayp3
        return DisplayP3()
    elseif s == :rec2020 || s == :Rec2020 || s == :bt2020
        return Rec2020()
    else
        error("Unknown color space: $s. Use :srgb, :p3, or :rec2020")
    end
end

# Lisp-friendly wrappers
gay_random_color() = random_color(SRGB())
gay_random_color(cs::Symbol) = random_color(sym_to_colorspace(cs))

gay_random_colors(n::Int) = random_colors(n, SRGB())
gay_random_colors(n::Int, cs::Symbol) = random_colors(n, sym_to_colorspace(cs))

gay_random_palette(n::Int) = random_palette(n, SRGB())
gay_random_palette(n::Int, cs::Symbol) = random_palette(n, sym_to_colorspace(cs))

gay_pride(flag::Symbol) = pride_flag(flag, SRGB())
gay_pride(flag::Symbol, cs::Symbol) = pride_flag(flag, sym_to_colorspace(cs))

# Export Lisp-friendly names
export gay_random_color, gay_random_colors, gay_random_palette, gay_pride

# ═══════════════════════════════════════════════════════════════════════════
# Color display helpers
# ═══════════════════════════════════════════════════════════════════════════

"""
    show_colors(colors; width=2)

Display colors as ANSI true-color blocks in the terminal.
"""
function show_colors(colors::Vector; width::Int=2)
    block = "█" ^ width
    for c in colors
        rgb = convert(RGB, c)
        r = round(Int, clamp(rgb.r, 0, 1) * 255)
        g = round(Int, clamp(rgb.g, 0, 1) * 255)
        b = round(Int, clamp(rgb.b, 0, 1) * 255)
        print("\e[38;2;$(r);$(g);$(b)m$(block)\e[0m")
    end
    println()
end

"""
    show_palette(colors)

Display colors with their hex codes.
"""
function show_palette(colors::Vector)
    for c in colors
        rgb = convert(RGB, c)
        r = round(Int, clamp(rgb.r, 0, 1) * 255)
        g = round(Int, clamp(rgb.g, 0, 1) * 255)
        b = round(Int, clamp(rgb.b, 0, 1) * 255)
        hex = "#" * string(r, base=16, pad=2) * 
                    string(g, base=16, pad=2) * 
                    string(b, base=16, pad=2) |> uppercase
        print("\e[38;2;$(r);$(g);$(b)m████\e[0m $hex  ")
    end
    println()
end

export show_colors, show_palette

# ═══════════════════════════════════════════════════════════════════════════
# Module initialization
# ═══════════════════════════════════════════════════════════════════════════

function __init__()
    # Initialize global splittable RNG
    gay_seed!(GAY_SEED)
    
    # Auto-initialize REPL if running interactively
    if isdefined(Base, :active_repl) && Base.active_repl !== nothing
        @async begin
            sleep(0.1)  # Let REPL finish loading
            init_gay_repl()
        end
    else
        @info "Gay.jl loaded 🏳️‍🌈 - Wide-gamut colors + splittable determinism"
        @info "In REPL: init_gay_repl() to start Gay mode (press ` to enter)"
    end
end

end # module Gay
