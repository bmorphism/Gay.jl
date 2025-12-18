#!/usr/bin/env julia
# Irreducibles: A world of seeds, each an irreducible source of infinite color streams
#
# Each seed is a universe. From it, all colors differentiate.
# Some seeds are named. Some are numeric. All are irreducible.

using Pkg
Pkg.activate(dirname(dirname(@__FILE__)); io=devnull)

using Gay
using Colors

# ═══════════════════════════════════════════════════════════════════════════
# The Irreducibles: Named seeds that generate universes of color
# ═══════════════════════════════════════════════════════════════════════════

# FNV-1a hash: text → seed
function fnv1a(text::String)::UInt64
    h = UInt64(14695981039346656037)
    for c in text
        h = (h ⊻ UInt64(c)) * UInt64(1099511628211)
    end
    h
end

# Named irreducibles - each a universe
const IRREDUCIBLES = Dict{Symbol, UInt64}(
    # The original
    :gay => Gay.GAY_SEED,  # 0x6761795f636f6c6f
    
    # Xenofeminist 
    :xf => fnv1a("If nature is unjust, change nature!"),
    :laboria => fnv1a("Laboria Cuboniks"),
    
    # Black holes
    :m87 => UInt64(2017),      # EHT M87* observation year
    :sgra => UInt64(2022),     # Sgr A* observation year
    :eht => fnv1a("Event Horizon Telescope"),
    
    # Mathematical
    :phi => UInt64(1618033988),  # Golden ratio digits
    :e => UInt64(2718281828),    # Euler's number
    :pi => UInt64(3141592653),   # Pi
    :tau => UInt64(6283185307),  # Tau = 2π
    
    # Pride
    :rainbow => fnv1a("rainbow"),
    :trans => fnv1a("transgender"),
    :bi => fnv1a("bisexual"),
    :nb => fnv1a("nonbinary"),
    :pan => fnv1a("pansexual"),
    :ace => fnv1a("asexual"),
    
    # Cosmic
    :cosmic => fnv1a("cosmic microwave background"),
    :quasar => fnv1a("quasi-stellar radio source"),
    :pulsar => fnv1a("pulsating radio star"),
    :magnetar => fnv1a("magnetic neutron star"),
    
    # Tech
    :julia => fnv1a("Julia programming language"),
    :unison => fnv1a("Unison programming language"),
    :lisp => fnv1a("LISt Processing"),
    :haskell => fnv1a("Haskell Curry"),
)

# ═══════════════════════════════════════════════════════════════════════════
# Dynamic Coloring Functions
# ═══════════════════════════════════════════════════════════════════════════

# ANSI helpers
ansi_bg(c::RGB) = "\e[48;2;$(round(Int, red(c)*255));$(round(Int, green(c)*255));$(round(Int, blue(c)*255))m"
ansi_fg(c::RGB) = "\e[38;2;$(round(Int, red(c)*255));$(round(Int, green(c)*255));$(round(Int, blue(c)*255))m"
const ansi_reset = "\e[0m"
const ansi_bold = "\e[1m"
const ansi_dim = "\e[2m"

hex(c::RGB) = uppercase("#$(lpad(string(round(Int, red(c)*255), base=16), 2, '0'))$(lpad(string(round(Int, green(c)*255), base=16), 2, '0'))$(lpad(string(round(Int, blue(c)*255), base=16), 2, '0'))")

# Color stream from any irreducible
function color_stream(seed::UInt64, n::Int=6)
    [color_at(i; seed=seed) for i in 1:n]
end

# Display a stream
function show_stream(name::Symbol, seed::UInt64; n::Int=8)
    colors = color_stream(seed, n)
    print("  ")
    for c in colors
        print("$(ansi_bg(c))  $(ansi_reset)")
    end
    c0 = color_at(1; seed=seed)
    println(" :$name $(ansi_dim)(0x$(string(seed, base=16)))$(ansi_reset)")
end

# ═══════════════════════════════════════════════════════════════════════════
# The Differentiation Operator: ∂
# From an irreducible, differentiate into a palette
# ═══════════════════════════════════════════════════════════════════════════

struct Differentiation
    seed::UInt64
    name::Symbol
    depth::Int
    children::Vector{UInt64}
end

# Differentiate: split seed into n child seeds
function differentiate(seed::UInt64, n::Int=4)
    # Use splitmix64 to generate child seeds
    children = UInt64[]
    h = seed
    for _ in 1:n
        h = xor(h, h >> 30) * 0xbf58476d1ce4e5b9
        h = xor(h, h >> 27) * 0x94d049bb133111eb
        h = xor(h, h >> 31)
        push!(children, h)
    end
    children
end

# Recursive differentiation tree
function diff_tree(seed::UInt64, name::Symbol, depth::Int, max_depth::Int=3)
    if depth >= max_depth
        return Differentiation(seed, name, depth, UInt64[])
    end
    children = differentiate(seed, 2)
    Differentiation(seed, name, depth, children)
end

# ═══════════════════════════════════════════════════════════════════════════
# Dynamic World: Animate through irreducibles
# ═══════════════════════════════════════════════════════════════════════════

function world_of_irreducibles()
    println()
    println("$(ansi_bold)🌈 World of Irreducibles 🌈$(ansi_reset)")
    println("$(ansi_dim)Each seed is a universe of infinite colors$(ansi_reset)")
    println()
    
    # Group by category
    categories = [
        ("Original", [:gay]),
        ("Xenofeminist", [:xf, :laboria]),
        ("Black Holes", [:m87, :sgra, :eht]),
        ("Mathematical", [:phi, :e, :pi, :tau]),
        ("Pride", [:rainbow, :trans, :bi, :nb, :pan, :ace]),
        ("Cosmic", [:cosmic, :quasar, :pulsar, :magnetar]),
        ("Languages", [:julia, :unison, :lisp, :haskell]),
    ]
    
    for (category, names) in categories
        println("$(ansi_bold)$category$(ansi_reset)")
        for name in names
            seed = IRREDUCIBLES[name]
            show_stream(name, seed)
        end
        println()
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# Differentiation Visualization
# ═══════════════════════════════════════════════════════════════════════════

function show_differentiation(seed::UInt64, name::Symbol; depth::Int=3)
    println("$(ansi_bold)∂ Differentiation from :$name$(ansi_reset)")
    println("$(ansi_dim)seed = 0x$(string(seed, base=16))$(ansi_reset)")
    println()
    
    function show_level(s::UInt64, level::Int, prefix::String="")
        c = color_at(1; seed=s)
        indent = "  " ^ level
        connector = level == 0 ? "" : (prefix == "└" ? "└── " : "├── ")
        
        print("$(indent)$(connector)$(ansi_bg(c))  $(ansi_reset) ")
        println("$(ansi_fg(c))0x$(string(s, base=16, pad=16))$(ansi_reset)")
        
        if level < depth
            children = differentiate(s, 2)
            for (i, child) in enumerate(children)
                p = i == length(children) ? "└" : "├"
                show_level(child, level + 1, p)
            end
        end
    end
    
    show_level(seed, 0)
    println()
end

# ═══════════════════════════════════════════════════════════════════════════
# Interactive: Create our own irreducible
# ═══════════════════════════════════════════════════════════════════════════

function create_irreducible(text::String)
    seed = fnv1a(text)
    name = Symbol(replace(lowercase(text), r"[^a-z0-9]" => "_"))
    
    println("$(ansi_bold)New Irreducible: :$name$(ansi_reset)")
    println("  text = \"$text\"")
    println("  seed = 0x$(string(seed, base=16))")
    println()
    
    show_stream(name, seed; n=12)
    println()
    
    # Show first 6 colors with hex
    println("  First 6 colors:")
    for i in 1:6
        c = color_at(i; seed=seed)
        println("    [$i] $(ansi_bg(c))  $(ansi_reset) $(hex(c))")
    end
    
    seed
end

# ═══════════════════════════════════════════════════════════════════════════
# Morphisms between irreducibles
# ═══════════════════════════════════════════════════════════════════════════

function morphism(from::Symbol, to::Symbol; steps::Int=8)
    seed_from = IRREDUCIBLES[from]
    seed_to = IRREDUCIBLES[to]
    
    println("$(ansi_bold)Morphism: :$from → :$to$(ansi_reset)")
    println()
    
    for i in 0:steps
        t = i / steps
        # Interpolate seed (this creates a path through seed space)
        mixed_seed = UInt64(round((1-t) * seed_from + t * seed_to))
        
        colors = [color_at(j; seed=mixed_seed) for j in 1:6]
        print("  t=$(round(t, digits=2)) ")
        for c in colors
            print("$(ansi_bg(c))  $(ansi_reset)")
        end
        println()
    end
    println()
end

# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

function main()
    # Show the world
    world_of_irreducibles()
    
    # Show differentiation tree
    show_differentiation(IRREDUCIBLES[:gay], :gay; depth=3)
    show_differentiation(IRREDUCIBLES[:xf], :xf; depth=2)
    
    # Create new irreducible from text
    create_irreducible("Hello, World!")
    create_irreducible("amp threads")
    create_irreducible("splittable reafference")
    
    # Morphism between seeds
    morphism(:gay, :xf)
    morphism(:m87, :sgra)
    
    # Summary
    println("$(ansi_bold)═══ $(length(IRREDUCIBLES)) Irreducibles Defined ═══$(ansi_reset)")
    println()
    println("\e[32m✓ Each irreducible generates infinite deterministic colors\e[0m")
    println("\e[32m✓ Differentiation (∂) splits seeds into child universes\e[0m")
    println("\e[32m✓ Morphisms interpolate between seed spaces\e[0m")
    println("\e[32m✓ Any text can become an irreducible via FNV-1a\e[0m")
    println()
end

main()
