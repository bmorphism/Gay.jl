# Gay.jl REPL - Rainbow-colored interactive color exploration
# Combines Lisp syntax with inline color display

using REPL: REPL, LineEdit
using ReplMaker
using Colors: RGB

# ═══════════════════════════════════════════════════════════════════════════
# Rainbow prompt generation
# ═══════════════════════════════════════════════════════════════════════════

const RAINBOW_COLORS = [
    (228, 3, 3),     # Red
    (255, 140, 0),   # Orange  
    (255, 237, 0),   # Yellow
    (0, 128, 38),    # Green
    (0, 77, 255),    # Blue
    (117, 7, 135),   # Violet
]

"""
Generate a rainbow-colored string for the REPL prompt.
"""
function rainbow_text(text::String)
    chars = collect(text)
    n = length(RAINBOW_COLORS)
    buf = IOBuffer()
    for (i, c) in enumerate(chars)
        r, g, b = RAINBOW_COLORS[mod1(i, n)]
        print(buf, "\e[38;2;$(r);$(g);$(b)m", c)
    end
    print(buf, "\e[0m")
    return String(take!(buf))
end

"""
Get the current invocation count for the prompt.
"""
function prompt_invocation()
    if isassigned(GLOBAL_GAY_RNG)
        return GLOBAL_GAY_RNG[].invocation
    else
        return 0
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# REPL evaluation with color display
# ═══════════════════════════════════════════════════════════════════════════

"""
Evaluate input in the Gay REPL.
Supports special commands and auto-displays colors.
"""
function gay_eval(input::String)
    input = strip(input)
    
    # Special commands
    if startswith(input, "!")
        return handle_command(input[2:end])
    end
    
    # Check if it's Lisp syntax (starts with paren)
    if startswith(input, "(")
        return eval_lisp(input)
    end
    
    # Otherwise evaluate as Julia
    return eval_julia(input)
end

function eval_julia(input::String)
    expr = Meta.parse(input)
    result = Core.eval(Main, expr)
    maybe_show_color(result)
    return result
end

function eval_lisp(input::String)
    result = Core.eval(Main, lisp_eval_helper(input))
    maybe_show_color(result)
    return result
end

"""
If the result is a color or color array, display it visually.
"""
function maybe_show_color(result)
    if result isa RGB || result isa Color
        print("  ")
        show_color_inline(result)
        println()
    elseif result isa AbstractVector && !isempty(result) && first(result) isa Color
        print("  ")
        for c in result
            show_color_inline(c)
        end
        println()
    end
end

function show_color_inline(c::Color)
    rgb = convert(RGB, c)
    r = round(Int, clamp(rgb.r, 0, 1) * 255)
    g = round(Int, clamp(rgb.g, 0, 1) * 255)
    b = round(Int, clamp(rgb.b, 0, 1) * 255)
    print("\e[48;2;$(r);$(g);$(b)m  \e[0m")
end

# ═══════════════════════════════════════════════════════════════════════════
# Special commands
# ═══════════════════════════════════════════════════════════════════════════

const COMMANDS = Dict{String, Function}()

function handle_command(cmd::String)
    parts = split(cmd)
    isempty(parts) && return help_command()
    
    name = lowercase(parts[1])
    args = parts[2:end]
    
    if haskey(COMMANDS, name)
        return COMMANDS[name](args...)
    else
        println("  Unknown command: $name")
        return help_command()
    end
end

function help_command(args...)
    println("""
  ╔═══════════════════════════════════════════════════════════╗
  ║  Gay.jl REPL Commands                                     ║
  ╠═══════════════════════════════════════════════════════════╣
  ║  !help              Show this help                        ║
  ║  !seed <n>          Set RNG seed                          ║
  ║  !next [n]          Generate next color(s)                ║
  ║  !at <index>        Get color at index                    ║
  ║  !palette <n>       Generate n-color palette              ║
  ║  !rainbow           Show rainbow flag                     ║
  ║  !pride <flag>      Show pride flag colors                ║
  ║  !blackhole [seed]  Render black hole                     ║
  ║  !space <name>      Set color space (srgb/p3/rec2020)     ║
  ╚═══════════════════════════════════════════════════════════╝
  
  Or use Julia/Lisp syntax directly:
    next_color()
    (next-color)
    rainbow(Rec2020())
""")
    return nothing
end
COMMANDS["help"] = help_command
COMMANDS["?"] = help_command

function seed_command(args...)
    seed = isempty(args) ? 42 : parse(Int, args[1])
    gay_seed!(seed)
    println("  Seed set to $seed")
    return seed
end
COMMANDS["seed"] = seed_command

function next_command(args...)
    n = isempty(args) ? 1 : parse(Int, args[1])
    colors = [next_color(current_colorspace()) for _ in 1:n]
    print("  ")
    for c in colors
        show_color_inline(c)
    end
    println()
    return n == 1 ? colors[1] : colors
end
COMMANDS["next"] = next_command

function at_command(args...)
    isempty(args) && (println("  Usage: !at <index>"); return nothing)
    idx = parse(Int, args[1])
    c = color_at(idx, current_colorspace())
    print("  [$idx] ")
    show_color_inline(c)
    println()
    return c
end
COMMANDS["at"] = at_command

function palette_command(args...)
    n = isempty(args) ? 6 : parse(Int, args[1])
    colors = next_palette(n, current_colorspace())
    print("  ")
    show_palette(colors)
    return colors
end
COMMANDS["palette"] = palette_command

function rainbow_command(args...)
    colors = rainbow(current_colorspace())
    print("  ")
    show_colors(colors; width=4)
    return colors
end
COMMANDS["rainbow"] = rainbow_command

function pride_command(args...)
    flag = isempty(args) ? :rainbow : Symbol(args[1])
    colors = pride_flag(flag, current_colorspace())
    print("  ")
    show_colors(colors; width=4)
    return colors
end
COMMANDS["pride"] = pride_command

function blackhole_command(args...)
    seed = isempty(args) ? 1337 : parse(Int, args[1])
    # Load blackhole module if available
    blackhole_file = joinpath(@__DIR__, "..", "examples", "blackhole.jl")
    if isfile(blackhole_file)
        include(blackhole_file)
        println(render_blackhole(seed=seed, rings=8, resolution=25, colorspace=current_colorspace()))
    else
        println("  Black hole demo not found. Run from Gay.jl directory.")
    end
    return nothing
end
COMMANDS["blackhole"] = blackhole_command
COMMANDS["bh"] = blackhole_command

# Current color space state
const CURRENT_COLORSPACE = Ref{ColorSpace}(SRGB())

current_colorspace() = CURRENT_COLORSPACE[]

function space_command(args...)
    if isempty(args)
        println("  Current: $(typeof(current_colorspace()))")
        println("  Options: srgb, p3, rec2020")
        return current_colorspace()
    end
    
    name = lowercase(args[1])
    cs = if name == "srgb"
        SRGB()
    elseif name == "p3" || name == "displayp3"
        DisplayP3()
    elseif name == "rec2020" || name == "2020"
        Rec2020()
    else
        println("  Unknown color space: $name")
        return current_colorspace()
    end
    
    CURRENT_COLORSPACE[] = cs
    println("  Color space set to $(typeof(cs))")
    return cs
end
COMMANDS["space"] = space_command
COMMANDS["cs"] = space_command

# ═══════════════════════════════════════════════════════════════════════════
# REPL initialization
# ═══════════════════════════════════════════════════════════════════════════

"""
    init_gay_repl(; start_key="🏳️‍🌈", sticky=true)

Initialize the Gay REPL mode. 
Press the start_key (default: type 'gay' then backspace) to enter.
"""
function init_gay_repl(; start_key::String = "`", sticky::Bool = true)
    # Dynamic rainbow prompt based on invocation
    function gay_prompt()
        inv = prompt_invocation()
        rainbow_text("gay[$inv]> ")
    end
    
    ReplMaker.initrepl(
        gay_eval,
        repl = Base.active_repl,
        prompt_text = gay_prompt,
        prompt_color = :nothing,  # We handle colors ourselves
        start_key = start_key,
        sticky_mode = sticky,
        mode_name = "Gay"
    )
    
    println()
    println(rainbow_text("  ╔═══════════════════════════════════════╗"))
    println(rainbow_text("  ║     Gay.jl REPL Initialized 🏳️‍🌈      ║"))
    println(rainbow_text("  ╚═══════════════════════════════════════╝"))
    println("  Press $(start_key) to enter Gay mode. Type !help for commands.")
    println()
end

export init_gay_repl, current_colorspace, show_color_inline, rainbow_text
