# # World-Specific Languages: .jl User Interfaces
#
# Julia's metaprogramming enables domain-specific languages (DSLs) to be born
# as distinct "worlds" - each with its own concrete syntax, mental model, and
# user interface, yet all sharing the same substrate.
#
# This is the realization of Landin's "next 700 programming languages" vision:
# one abstract syntax, many concrete possibilities.
#
# Worlds demonstrated:
# - Gay.jl: Color generation with splittable determinism
# - Comrade.jl: VLBI sky model composition  
# - Pigeons.jl: Parallel tempering MCMC
# - All share: SPI (Strong Parallelism Invariance)

using Gay
using Gay: hash_color, xor_fingerprint, ka_colors

# ═══════════════════════════════════════════════════════════════════════════════
# World Registry: Each world has its own syntax and semantics
# ═══════════════════════════════════════════════════════════════════════════════

abstract type World end

struct WorldSpec
    name::String
    seed_name::String      # What they call "seed"
    index_name::String     # What they call "index"  
    output_name::String    # What they call "output"
    syntax_example::String # Concrete syntax example
    seed::UInt64          # Actual seed value
end

# ═══════════════════════════════════════════════════════════════════════════════
# The Worlds
# ═══════════════════════════════════════════════════════════════════════════════

# Gay.jl: Wide-gamut color palettes
const GAY_WORLD = WorldSpec(
    "Gay.jl",
    "seed",
    "index", 
    "color",
    "color_at(42, seed=42069)",
    0x6761795f636f6c6f  # "gay_colo"
)

# Comrade.jl: VLBI black hole imaging
const COMRADE_WORLD = WorldSpec(
    "Comrade.jl",
    "rng",
    "sample_id",
    "sky_model",
    "sample(posterior, rng=StableRNG(42))",
    0x636f6d7261646521  # "comrade!"
)

# Pigeons.jl: Parallel tempering MCMC
const PIGEONS_WORLD = WorldSpec(
    "Pigeons.jl", 
    "explorer",
    "replica",
    "chain",
    "pigeons(target, explorer=SplittableRandom(42))",
    0x706967656f6e7321  # "pigeons!"
)

# XF.jl: Xenofeminist color synthesis  
const XF_WORLD = WorldSpec(
    "XF.jl",
    "xf_seed",
    "position",
    "synthesis",
    "xf_color(42, seed=XF_SEED)",
    0x78656e6f66656d21  # "xenofem!"
)

# DuckDB: Analytical queries
const DUCKDB_WORLD = WorldSpec(
    "DuckDB",
    "random_seed",
    "row_id",
    "result",
    "SELECT * FROM repos ORDER BY random(42)",
    0x6475636b64622121  # "duckdb!!"
)

const ALL_WORLDS = [GAY_WORLD, COMRADE_WORLD, PIGEONS_WORLD, XF_WORLD, DUCKDB_WORLD]

# ═══════════════════════════════════════════════════════════════════════════════
# Cross-World Color Generation: Same hash, different vocabulary
# ═══════════════════════════════════════════════════════════════════════════════

"""
Generate a color in any world's vocabulary.
Demonstrates: same splitmix64 hash, different concrete syntax.
"""
function world_color(world::WorldSpec, idx::Integer)
    hash_color(world.seed, UInt64(idx))
end

"""
Generate a fingerprint for a world's color palette.
SPI guarantee: same seed → same fingerprint, regardless of world.
"""
function world_fingerprint(world::WorldSpec, n::Integer)
    colors = zeros(Float32, n, 3)
    for i in 1:n
        r, g, b = hash_color(world.seed, UInt64(i))
        colors[i, 1] = r
        colors[i, 2] = g
        colors[i, 3] = b
    end
    xor_fingerprint(colors)
end

# ═══════════════════════════════════════════════════════════════════════════════
# World-Specific Language Examples
# ═══════════════════════════════════════════════════════════════════════════════

"""
    @gay_syntax expr

Macro that transforms Gay.jl syntax to hash_color calls.
Example of how concrete syntax becomes a user interface.
"""
macro gay_syntax(expr)
    if expr.head == :call && expr.args[1] == :color_at
        idx = expr.args[2]
        return :(hash_color(GAY_WORLD.seed, UInt64($idx)))
    end
    error("Unknown Gay.jl syntax: $expr")
end

"""
    @comrade_syntax expr

Macro for Comrade.jl-style sky model syntax.
"""
macro comrade_syntax(expr)
    if expr.head == :call && expr.args[1] == :sample
        idx = expr.args[2]
        return :(hash_color(COMRADE_WORLD.seed, UInt64($idx)))
    end
    error("Unknown Comrade.jl syntax: $expr")
end

"""
    @pigeons_syntax expr

Macro for Pigeons.jl-style parallel tempering syntax.
"""
macro pigeons_syntax(expr)
    if expr.head == :call && expr.args[1] == :replica
        idx = expr.args[2]
        return :(hash_color(PIGEONS_WORLD.seed, UInt64($idx)))
    end
    error("Unknown Pigeons.jl syntax: $expr")
end

# ═══════════════════════════════════════════════════════════════════════════════
# Demonstration: Worlds Have Different Syntax, Same SPI
# ═══════════════════════════════════════════════════════════════════════════════

function show_world_colors()
    println()
    println("═" ^ 75)
    println("  WORLD-SPECIFIC LANGUAGES: .jl User Interfaces")
    println("  Each world has its own syntax, vocabulary, and mental model")
    println("═" ^ 75)
    println()
    
    n_colors = 10
    
    for world in ALL_WORLDS
        println("─" ^ 75)
        println("  World: $(world.name)")
        println("  Vocabulary: $(world.seed_name) → $(world.index_name) → $(world.output_name)")
        println("  Syntax: $(world.syntax_example)")
        println("  Seed: 0x$(string(world.seed, base=16)) (\"$(String(reinterpret(UInt8, [world.seed])[1:8]))\")")
        print("  Colors: ")
        
        for i in 1:n_colors
            r, g, b = world_color(world, i)
            ri, gi, bi = round(Int, r*255), round(Int, g*255), round(Int, b*255)
            print("\e[38;2;$(ri);$(gi);$(bi)m██\e[0m")
        end
        println()
    end
    println("─" ^ 75)
    println()
end

function show_spi_across_worlds()
    println("═" ^ 75)
    println("  SPI INVARIANCE ACROSS WORLDS")
    println("  Same hash function, different seeds → different but reproducible palettes")
    println("═" ^ 75)
    println()
    
    n = 10000
    
    for world in ALL_WORLDS
        fp = world_fingerprint(world, n)
        
        # Verify reproducibility
        fp2 = world_fingerprint(world, n)
        status = fp == fp2 ? "✓" : "✗"
        
        # Show first color as sample
        r, g, b = world_color(world, 1)
        ri, gi, bi = round(Int, r*255), round(Int, g*255), round(Int, b*255)
        hex = "#$(string(ri, base=16, pad=2))$(string(gi, base=16, pad=2))$(string(bi, base=16, pad=2))" |> uppercase
        
        print("  \e[38;2;$(ri);$(gi);$(bi)m████\e[0m ")
        print("$(rpad(world.name, 12)) ")
        print("fp=0x$(string(fp, base=16, pad=8)) ")
        print("$hex ")
        println("$status")
    end
    
    println()
    println("  Each world: deterministic, reproducible, SPI-guaranteed")
    println("═" ^ 75)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Teleportation Between Worlds
# ═══════════════════════════════════════════════════════════════════════════════

"""
    teleport(from_world, to_world, value, idx)

Teleport a color from one world to another.
The color changes (different seed), but the index semantics are preserved.
"""
function teleport(from_world::WorldSpec, to_world::WorldSpec, idx::Integer)
    # In the origin world
    from_color = world_color(from_world, idx)
    
    # Teleport: same index, different world seed
    to_color = world_color(to_world, idx)
    
    (from_color, to_color)
end

function show_teleportation()
    println()
    println("═" ^ 75)
    println("  TELEPORTATION BETWEEN WORLDS")
    println("  Same index → different color (different seed/world)")
    println("  But: within each world, SPI is preserved")
    println("═" ^ 75)
    println()
    
    idx = 42
    
    println("  Index $idx in each world:")
    println()
    
    for world in ALL_WORLDS
        r, g, b = world_color(world, idx)
        ri, gi, bi = round(Int, r*255), round(Int, g*255), round(Int, b*255)
        hex = "#$(string(ri, base=16, pad=2))$(string(gi, base=16, pad=2))$(string(bi, base=16, pad=2))" |> uppercase
        
        print("  \e[38;2;$(ri);$(gi);$(bi)m████\e[0m ")
        println("$(rpad(world.name, 12)) → $hex")
    end
    
    println()
    println("  Teleportation preserves INDEX, not COLOR")
    println("  Each world's concrete syntax defines a different color space")
    println("═" ^ 75)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Language Birth: Defining a New World
# ═══════════════════════════════════════════════════════════════════════════════

"""
    @defworld name seed vocab...

Macro to birth a new world-specific language.
"""
macro defworld(name, seed_hex, seed_name, index_name, output_name)
    name_str = string(name)
    seed_name_str = string(seed_name)
    index_name_str = string(index_name)
    output_name_str = string(output_name)
    syntax_str = "$name_str.$output_name_str($seed_name_str, $index_name_str)"
    quote
        const $(esc(name)) = WorldSpec(
            $name_str,
            $seed_name_str,
            $index_name_str,
            $output_name_str,
            $syntax_str,
            $seed_hex
        )
    end
end

# Birth some new worlds!
@defworld TOPOS 0x746f706f73212121 theory morphism insight
@defworld RIO   0x72696f7465726d21 config frame glyph
@defworld NARYA 0x6e617279612e6a6c proof term judgment

function show_world_birth()
    println()
    println("═" ^ 75)
    println("  WORLD BIRTH: Defining New Languages")
    println("  @defworld NAME SEED vocabulary...")
    println("═" ^ 75)
    println()
    
    new_worlds = [TOPOS, RIO, NARYA]
    
    for world in new_worlds
        r, g, b = world_color(world, 1)
        ri, gi, bi = round(Int, r*255), round(Int, g*255), round(Int, b*255)
        
        print("  \e[38;2;$(ri);$(gi);$(bi)m████\e[0m ")
        println("$(world.name): $(world.seed_name) → $(world.index_name) → $(world.output_name)")
    end
    
    println()
    println("  Each @defworld creates a new .jl user interface")
    println("  Same Julia, same SPI, different concrete syntax")
    println("═" ^ 75)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

function main()
    println()
    println("╔" * "═" ^ 73 * "╗")
    println("║" * " " ^ 20 * "WORLD-SPECIFIC LANGUAGES" * " " ^ 29 * "║")
    println("║" * " " ^ 73 * "║")
    println("║  \"One abstract syntax, many concrete possibilities\"  — Landin, 1966  ║")
    println("╚" * "═" ^ 73 * "╝")
    
    show_world_colors()
    show_spi_across_worlds()
    show_teleportation()
    show_world_birth()
    
    println()
    println("  Julia enables world-specific languages through:")
    println("    • Macros → custom concrete syntax")
    println("    • Multiple dispatch → polymorphic semantics")
    println("    • SplittableRandoms → SPI across all worlds")
    println("    • KernelAbstractions → GPU portability")
    println()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
