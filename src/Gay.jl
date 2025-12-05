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
Lisp-accessible DETERMINISTIC color generation.

Usage from Gay REPL (Lisp syntax with parentheses):
  (gay-next)                  ; Next deterministic color  
  (gay-next 5)                ; Next 5 colors
  (gay-at 42)                 ; Color at index 42
  (gay-at 1 2 3)              ; Colors at indices 1,2,3
  (gay-palette 6)             ; 6 visually distinct colors
  (gay-seed 1337)             ; Set RNG seed
  (pride :rainbow)            ; Rainbow flag
  (pride :trans :rec2020)     ; Trans flag in Rec.2020
  (gay-blackhole 42)          ; Render black hole with seed
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

# ═══════════════════════════════════════════════════════════════════════════
# Lisp-friendly deterministic color functions (kebab-case → snake_case)
# These are the primary API for reproducible colors from S-expressions
# ═══════════════════════════════════════════════════════════════════════════

# (gay-next) or (gay-next n) - deterministic next color(s)
gay_next() = next_color(current_colorspace())
gay_next(n::Int) = [next_color(current_colorspace()) for _ in 1:n]
gay_next(cs::Symbol) = next_color(sym_to_colorspace(cs))
gay_next(n::Int, cs::Symbol) = [next_color(sym_to_colorspace(cs)) for _ in 1:n]

# (gay-at index) or (gay-at i1 i2 i3...) - random access by index
gay_at(idx::Int) = color_at(idx, current_colorspace())
gay_at(idx::Int, cs::Symbol) = color_at(idx, sym_to_colorspace(cs))
gay_at(indices::Int...) = [color_at(i, current_colorspace()) for i in indices]

# (gay-palette n) - n visually distinct colors
gay_palette(n::Int) = next_palette(n, current_colorspace())
gay_palette(n::Int, cs::Symbol) = next_palette(n, sym_to_colorspace(cs))

# (gay-seed n) - set RNG seed for reproducibility
gay_seed(n::Int) = gay_seed!(n)

# (gay-space :rec2020) - set color space
gay_space(cs::Symbol) = (CURRENT_COLORSPACE[] = sym_to_colorspace(cs); current_colorspace())

# (gay-rng-state) - show current RNG state
gay_rng_state() = (r = gay_rng(); (seed=r.seed, invocation=r.invocation))

# (pride :flag) or (pride :flag :colorspace)
gay_pride(flag::Symbol) = pride_flag(flag, current_colorspace())
gay_pride(flag::Symbol, cs::Symbol) = pride_flag(flag, sym_to_colorspace(cs))

# Legacy random (non-deterministic) wrappers
gay_random_color() = random_color(SRGB())
gay_random_color(cs::Symbol) = random_color(sym_to_colorspace(cs))
gay_random_colors(n::Int) = random_colors(n, SRGB())
gay_random_colors(n::Int, cs::Symbol) = random_colors(n, sym_to_colorspace(cs))
gay_random_palette(n::Int) = random_palette(n, SRGB())
gay_random_palette(n::Int, cs::Symbol) = random_palette(n, sym_to_colorspace(cs))

# Export all Lisp-friendly names (kebab-case maps to these)
export gay_next, gay_at, gay_palette, gay_seed, gay_space, gay_rng_state
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
