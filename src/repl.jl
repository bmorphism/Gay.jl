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
Simple version without seed parameter (uses static rainbow colors).
"""
function rainbow_text_prompt(text::String)
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
  ╔═══════════════════════════════════════════════════════════════════╗
  ║  Gay.jl REPL - Reproducible Colors via SplittableRandoms         ║
  ╠═══════════════════════════════════════════════════════════════════╣
  ║  COMMANDS (! prefix)                                              ║
  ║    !seed <n>        Set RNG seed for reproducibility              ║
  ║    !next [n]        Generate next deterministic color(s)          ║
  ║    !at <i> [j k...] Get color(s) at specific index/indices        ║
  ║    !palette <n>     Generate n visually distinct colors           ║
  ║    !pride <flag>    Show pride flag (rainbow/trans/bi/nb/pan)     ║
  ║    !space <name>    Set color space (srgb/p3/rec2020)             ║
  ║    !blackhole [s]   Render black hole (optional seed)             ║
  ║    !state           Show RNG state (seed, invocation)             ║
  ║    !bench           Run Chairmarks microbenchmarks                ║
  ║    !metal           Show Metal GPU info and benchmark             ║
  ╠═══════════════════════════════════════════════════════════════════╣
  ║  WORLD TELEPORTATION (Abductive Testing)                          ║
  ║    !teleport <id>   Teleport to invader's world                   ║
  ║    !world           Show current world state                      ║
  ║    !back            Return to previous world                      ║
  ║    !abduce r g b    Infer invader from RGB (0-1 floats)           ║
  ║    !jump <n>        Jump to nth hypothesis from !abduce           ║
  ║    !neighbors [r]   Explore nearby invaders (radius r)            ║
  ║    !test [n]        Run n abductive roundtrip tests               ║
  ╠═══════════════════════════════════════════════════════════════════╣
  ║  LISP S-EXPRESSIONS (parentheses)                                 ║
  ║    (gay-seed 42)           Set seed                               ║
  ║    (gay-next)              Next deterministic color               ║
  ║    (gay-next 5)            Next 5 colors                          ║
  ║    (gay-at 1 10 100)       Colors at indices                      ║
  ║    (gay-palette 6)         6 distinct colors                      ║
  ║    (gay-space :rec2020)    Set Rec.2020 gamut                     ║
  ║    (gay-pride :trans)      Trans flag colors                      ║
  ║    (gay-rng-state)         Show (seed, invocation)                ║
  ╠═══════════════════════════════════════════════════════════════════╣
  ║  JULIA EXPRESSIONS                                                ║
  ║    gay_seed!(42)           Same as (gay-seed 42)                  ║
  ║    next_color()            Next color (uses current space)        ║
  ║    color_at(42)            Color at index 42                      ║
  ║    rainbow(Rec2020())      Rainbow in Rec.2020                    ║
  ╚═══════════════════════════════════════════════════════════════════╝
  
  Reproducibility: Same seed → same colors, always.
  Fork-safe: Each color = independent RNG split (Pigeons.jl SPI pattern)
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

# NOTE: CURRENT_COLORSPACE and current_colorspace() are defined in colorspaces.jl
# We reuse them here to avoid redefinition errors during precompilation.

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

function state_command(args...)
    r = gay_rng()
    println("  RNG State:")
    println("    seed:       $(r.seed)")
    println("    invocation: $(r.invocation)")
    println("    colorspace: $(typeof(current_colorspace()))")
    return (seed=r.seed, invocation=r.invocation)
end
COMMANDS["state"] = state_command
COMMANDS["rng"] = state_command

function bench_command(args...)
    println("  Running Chairmarks benchmarks...")
    println()
    results = gay_benchmark(verbose=true)
    return results
end
COMMANDS["bench"] = bench_command
COMMANDS["benchmark"] = bench_command

function metal_command(args...)
    if !metal_available()
        println("  Metal is not available on this system")
        return nothing
    end
    
    info = metal_info()
    println("  Metal GPU Information:")
    println("    Device:      $(info.name)")
    println("    Max threads: $(info.max_threads)")
    println("    Low power:   $(info.is_low_power)")
    println()
    
    if !isempty(args) && args[1] == "bench"
        println("  Running Metal benchmark...")
        return metal_benchmark()
    end
    
    return info
end
COMMANDS["metal"] = metal_command
COMMANDS["gpu"] = metal_command

# ═══════════════════════════════════════════════════════════════════════════
# World Teleportation Commands (Abductive Testing)
# ═══════════════════════════════════════════════════════════════════════════

function teleport_command(args...)
    isempty(args) && (println("  Usage: !teleport <id>"); return nothing)
    id = parse(Int, args[1])
    world = teleport!(id)
    println("  ⚡ Teleported to world #$(id)")
    show_world_state(world)
    return world
end
COMMANDS["teleport"] = teleport_command
COMMANDS["tp"] = teleport_command

function world_command(args...)
    world = current_world()
    show_world_state(world)
    return world
end
COMMANDS["world"] = world_command
COMMANDS["w"] = world_command

function back_command(args...)
    try
        world = back!()
        println("  ↩ Returned to world #$(world.id)")
        show_world_state(world)
        return world
    catch e
        println("  No history to go back to!")
        return nothing
    end
end
COMMANDS["back"] = back_command
COMMANDS["b"] = back_command

function abduce_command(args...)
    if length(args) < 3
        println("  Usage: !abduce <r> <g> <b> (floats 0-1)")
        return nothing
    end
    r, g, b = parse.(Float64, args[1:3])
    observed = RGB(r, g, b)
    
    println("  🔍 Abducing invader from color...")
    print("  Target: ")
    show_color_inline(observed)
    println()
    
    hypotheses = abduce!(observed; search_range=1:50000, top_k=5)
    
    println("  ─────────────────────────────────────────")
    println("  Top hypotheses (use !jump <n> to explore):")
    for (i, h) in enumerate(hypotheses)
        print("  [$i] ID=$(h.id) conf=$(round(h.confidence, digits=3)) ")
        show_color_inline(h.predicted_world)
        println()
    end
    return hypotheses
end
COMMANDS["abduce"] = abduce_command
COMMANDS["ab"] = abduce_command

function jump_command(args...)
    isempty(args) && (println("  Usage: !jump <hypothesis_index>"); return nothing)
    idx = parse(Int, args[1])
    world = jump_hypothesis!(idx)
    println("  🚀 Jumped to hypothesis #$(idx)")
    show_world_state(world)
    return world
end
COMMANDS["jump"] = jump_command
COMMANDS["j"] = jump_command

function neighbors_command(args...)
    radius = isempty(args) ? 5 : parse(Int, args[1])
    neighbors = explore_neighbors(; radius=radius)
    
    println("  Neighboring worlds (radius=$radius):")
    for sim in neighbors
        prefix = sim.id == get_navigator().current_id ? "→ " : "  "
        print("  $(prefix)[$(sim.id)] ")
        show_color_inline(sim.source)
        print(" → ")
        show_color_inline(sim.world)
        spin_char = sim.spin > 0 ? "↑" : "↓"
        println(" $spin_char")
    end
    return neighbors
end
COMMANDS["neighbors"] = neighbors_command
COMMANDS["nb"] = neighbors_command

function test_command(args...)
    n = isempty(args) ? 20 : parse(Int, args[1])
    println("  Running $n abductive roundtrip tests...")
    
    passed = 0
    failed = 0
    nav = get_navigator()
    
    for i in 1:n
        id = rand(1:100000)
        if abductive_roundtrip_test(id, nav.seed)
            passed += 1
            print("  ✓")
        else
            failed += 1
            print("  ✗")
        end
        i % 20 == 0 && println()
    end
    println()
    
    println("  ─────────────────────────────────────────")
    println("  Results: $(passed)/$(n) passed ($(round(100*passed/n, digits=1))%)")
    
    if failed > 0
        println("  ⚠ $(failed) tests failed!")
    else
        println("  ✓ All tests passed!")
    end
    
    return (passed=passed, failed=failed, total=n)
end
COMMANDS["test"] = test_command

function show_world_state(world)
    println("  ─────────────────────────────────────────")
    println("  Invader #$(world.id)")
    print("    Source:   ")
    show_color_inline(world.source)
    println(" (SPI hash)")
    
    print("    Deranged: ")
    show_color_inline(world.deranged)
    println(" (perm=$(world.derangement))")
    
    print("    World:    ")
    show_color_inline(world.world)
    println(" (t=$(round(world.tropical_t, digits=2)))")
    
    spin_char = world.spin > 0 ? "↑" : "↓"
    println("    Spin:     $(spin_char) ($(world.spin))")
    println("  ─────────────────────────────────────────")
end

# ═══════════════════════════════════════════════════════════════════════════
# REPL initialization
# ═══════════════════════════════════════════════════════════════════════════

"""
    init_gay_repl(; start_key=' ', sticky=true)

Initialize the Gay REPL mode. 
Press SPC (space bar) to enter Gay mode (SpaceInvaders.jl style).
Press backspace to return to Julia mode.
"""
function init_gay_repl(; start_key::Char = ' ', sticky::Bool = true)
    # Dynamic rainbow prompt based on invocation
    function gay_prompt()
        inv = prompt_invocation()
        rainbow_text_prompt("gay[$inv]> ")
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
    println(rainbow_text_prompt("  ╔═══════════════════════════════════════╗"))
    println(rainbow_text_prompt("  ║     Gay.jl REPL Initialized 🏳️‍🌈      ║"))
    println(rainbow_text_prompt("  ╚═══════════════════════════════════════╝"))
    println("  Press SPC (space bar) to enter Gay mode. Type !help for commands.")
    println()
end

export init_gay_repl, show_color_inline, rainbow_text_prompt
