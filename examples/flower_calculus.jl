#!/usr/bin/env julia
"""
The Flower Calculus (Pablo Donato, ItaCa Fest 2025)

A diagrammatic proof system for intuitionistic first-order logic
inspired by Peirce's existential graphs.

Flowers = nested regions containing atomic predicates
    • Petals (white) = disjunction
    • Pistil (shaded) = conjunction under negation
    • Nesting = implication

The calculus operates solely on nested flowers — no symbolic connectives!
This is STRUCTURE without symbols.

Connection to Gay.jl:
    • Each nesting depth gets a deterministic color
    • Petals and pistils colored differently
    • The flower IS a colored S-expression
    • Structure regression for logic

See: Donato, "The Flower Calculus" (ItaCa Fest, May 20, 2025)
     Peirce, "Existential Graphs" (1896)
"""

using Gay
using Colors: RGB

# ═══════════════════════════════════════════════════════════════════════════
# Flower Data Structures
# ═══════════════════════════════════════════════════════════════════════════

"""
An atomic predicate (leaf of the flower).
"""
struct Atom
    name::Symbol
    args::Vector{Symbol}
end

Atom(name::Symbol) = Atom(name, Symbol[])

"""
A petal: a region representing a disjunct.
Contains atoms and sub-flowers.
"""
struct Petal
    contents::Vector{Any}  # Atoms or Flowers
end

"""
A flower: pistil (conjunction) surrounded by petals (disjunction).

Semantically:
    Flower(pistil, [p1, p2, ...]) = ¬(pistil ∧ ¬p1 ∧ ¬p2 ∧ ...)
                                  = pistil → (p1 ∨ p2 ∨ ...)
"""
struct Flower
    pistil::Vector{Any}    # Conjunction of atoms/flowers (the center)
    petals::Vector{Petal}  # Disjunction of alternatives (the ring)
end

# Convenience constructors
Flower(pistil::Vector) = Flower(pistil, Petal[])
Flower() = Flower(Any[], Petal[])

# ═══════════════════════════════════════════════════════════════════════════
# Flower Construction DSL
# ═══════════════════════════════════════════════════════════════════════════

"""
    flower(pistil...; petals=[])

Create a flower with given pistil contents and petals.
"""
function flower(pistil...; petals=Petal[])
    Flower(collect(Any, pistil), petals)
end

"""
    petal(contents...)

Create a petal with given contents.
"""
function petal(contents...)
    Petal(collect(Any, contents))
end

"""
    atom(name, args...)

Create an atomic predicate.
"""
function atom(name::Symbol, args::Symbol...)
    Atom(name, collect(args))
end

# ═══════════════════════════════════════════════════════════════════════════
# Colored Rendering
# ═══════════════════════════════════════════════════════════════════════════

function ansi(c::RGB)
    r = round(Int, clamp(c.r, 0, 1) * 255)
    g = round(Int, clamp(c.g, 0, 1) * 255)
    b = round(Int, clamp(c.b, 0, 1) * 255)
    "\e[38;2;$(r);$(g);$(b)m"
end

function ansi_bg(c::RGB)
    r = round(Int, clamp(c.r, 0, 1) * 255)
    g = round(Int, clamp(c.g, 0, 1) * 255)
    b = round(Int, clamp(c.b, 0, 1) * 255)
    "\e[48;2;$(r);$(g);$(b)m"
end

const R = "\e[0m"
const BOLD = "\e[1m"
const DIM = "\e[2m"

"""
Get color for a given nesting depth.
Uses rainbow spectrum for nested structure.
"""
function depth_color(depth::Int; seed::Integer=42)
    gay_seed!(seed + depth * 7)
    next_color(SRGB())
end

"""
Render an atom.
"""
function render_atom(a::Atom; depth::Int=0, seed::Integer=42)
    c = depth_color(depth; seed=seed)
    args_str = isempty(a.args) ? "" : "($(join(a.args, ",")))"
    "$(ansi(c))$(a.name)$args_str$(R)"
end

"""
Render a flower as nested colored structure.
"""
function render_flower(f::Flower; depth::Int=0, seed::Integer=42, indent::Int=0)
    c_pistil = depth_color(depth; seed=seed)
    c_petal = depth_color(depth + 1; seed=seed)
    
    pad = "  "^indent
    lines = String[]
    
    # Opening
    push!(lines, "$pad$(ansi(c_pistil))╭$(R)$(ansi(c_pistil))───── Flower (depth=$depth) ─────$(R)$(ansi(c_pistil))╮$(R)")
    
    # Pistil (center, shaded)
    if !isempty(f.pistil)
        push!(lines, "$pad$(ansi(c_pistil))│$(R) $(DIM)pistil (∧):$(R)")
        for item in f.pistil
            if item isa Atom
                push!(lines, "$pad$(ansi(c_pistil))│$(R)   $(render_atom(item; depth=depth, seed=seed))")
            elseif item isa Flower
                sub_lines = split(render_flower(item; depth=depth+2, seed=seed, indent=indent+2), '\n')
                for sl in sub_lines
                    push!(lines, "$pad$(ansi(c_pistil))│$(R) $sl")
                end
            end
        end
    end
    
    # Petals (ring, white)
    if !isempty(f.petals)
        push!(lines, "$pad$(ansi(c_pistil))│$(R)")
        push!(lines, "$pad$(ansi(c_pistil))│$(R) $(ansi(c_petal))petals (∨):$(R)")
        for (i, pet) in enumerate(f.petals)
            push!(lines, "$pad$(ansi(c_pistil))│$(R)   $(ansi(c_petal))◌ petal $i:$(R)")
            for item in pet.contents
                if item isa Atom
                    push!(lines, "$pad$(ansi(c_pistil))│$(R)     $(render_atom(item; depth=depth+1, seed=seed))")
                elseif item isa Flower
                    sub_lines = split(render_flower(item; depth=depth+2, seed=seed, indent=indent+3), '\n')
                    for sl in sub_lines
                        push!(lines, "$pad$(ansi(c_pistil))│$(R)   $sl")
                    end
                end
            end
        end
    end
    
    # Closing
    push!(lines, "$pad$(ansi(c_pistil))╰$(R)$(ansi(c_pistil))─────────────────────────────$(R)$(ansi(c_pistil))╯$(R)")
    
    return join(lines, '\n')
end

"""
Render a flower as a one-line colored S-expression.
"""
function render_flower_sexpr(f::Flower; depth::Int=0, seed::Integer=42)
    c = depth_color(depth; seed=seed)
    
    parts = String[]
    
    # Pistil
    for item in f.pistil
        if item isa Atom
            push!(parts, render_atom(item; depth=depth, seed=seed))
        elseif item isa Flower
            push!(parts, render_flower_sexpr(item; depth=depth+1, seed=seed))
        end
    end
    
    pistil_str = join(parts, " ∧ ")
    
    # Petals
    if !isempty(f.petals)
        petal_parts = String[]
        for pet in f.petals
            pet_items = String[]
            for item in pet.contents
                if item isa Atom
                    push!(pet_items, render_atom(item; depth=depth+1, seed=seed))
                elseif item isa Flower
                    push!(pet_items, render_flower_sexpr(item; depth=depth+2, seed=seed))
                end
            end
            push!(petal_parts, join(pet_items, " ∧ "))
        end
        petal_str = join(petal_parts, " ∨ ")
        
        if isempty(pistil_str)
            return "$(ansi(c))($(R)$petal_str$(ansi(c)))$(R)"
        else
            return "$(ansi(c))($(R)$pistil_str $(ansi(c))→$(R) $petal_str$(ansi(c)))$(R)"
        end
    else
        return "$(ansi(c))($(R)$pistil_str$(ansi(c)))$(R)"
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# Example Flowers (from logic)
# ═══════════════════════════════════════════════════════════════════════════

"""
Modus ponens as a flower:
    (P ∧ (P → Q)) → Q
"""
function modus_ponens_flower()
    # P → Q is a flower with pistil P and petal Q
    p_implies_q = Flower([atom(:P)], [petal(atom(:Q))])
    
    # The whole thing: (P ∧ (P→Q)) → Q
    Flower(
        [atom(:P), p_implies_q],  # pistil: P ∧ (P → Q)
        [petal(atom(:Q))]         # petal: Q
    )
end

"""
Law of excluded middle as a flower:
    ⊤ → (P ∨ ¬P)
    
But this is NOT provable in intuitionistic logic!
The flower cannot be reduced to a tautology.
"""
function excluded_middle_flower()
    # ¬P is a flower with pistil P and no petals
    not_p = Flower([atom(:P)], Petal[])
    
    # P ∨ ¬P
    Flower(
        Any[],  # empty pistil (true)
        [petal(atom(:P)), petal(not_p)]  # petals: P or ¬P
    )
end

"""
Double negation elimination (also not intuitionistic):
    ¬¬P → P
"""
function double_negation_flower()
    not_p = Flower([atom(:P)], Petal[])
    not_not_p = Flower([not_p], Petal[])
    
    Flower(
        [not_not_p],     # pistil: ¬¬P
        [petal(atom(:P))]  # petal: P
    )
end

"""
Transitivity of implication:
    ((P → Q) ∧ (Q → R)) → (P → R)
"""
function transitivity_flower()
    p_implies_q = Flower([atom(:P)], [petal(atom(:Q))])
    q_implies_r = Flower([atom(:Q)], [petal(atom(:R))])
    p_implies_r = Flower([atom(:P)], [petal(atom(:R))])
    
    Flower(
        [p_implies_q, q_implies_r],  # pistil: (P→Q) ∧ (Q→R)
        [petal(p_implies_r)]          # petal: P→R
    )
end

# ═══════════════════════════════════════════════════════════════════════════
# Connection to Peirce's Existential Graphs
# ═══════════════════════════════════════════════════════════════════════════

function render_peirce_connection(; seed::Integer=42)
    gay_seed!(seed)
    
    c1 = next_color(SRGB())
    c2 = next_color(SRGB())
    
    println()
    println("  $(BOLD)Peirce's Existential Graphs → Flower Calculus$(R)")
    println("  ════════════════════════════════════════════════════════════")
    println()
    println("  $(ansi(c1))Peirce (1896):$(R) Logic as diagrams on a \"sheet of assertion\"")
    println()
    println("    • $(ansi(c1))Juxtaposition$(R) = conjunction (AND)")
    println("    • $(ansi(c1))Cut (oval)$(R) = negation (NOT)")
    println("    • $(ansi(c1))Nested cuts$(R) = implication")
    println()
    println("  $(ansi(c2))Donato (2025):$(R) Flowers generalize Peirce's cuts")
    println()
    println("    • $(ansi(c2))Pistil$(R) = shaded center (conjunction under negation)")
    println("    • $(ansi(c2))Petals$(R) = white alternatives (disjunction)")
    println("    • $(ansi(c2))Nesting$(R) = implication structure")
    println()
    println("  $(DIM)Both eliminate symbolic connectives entirely!$(R)")
    println("  $(DIM)Structure, not symbols.$(R)")
    println()
end

# ═══════════════════════════════════════════════════════════════════════════
# Connection to S-Expressions and Gay.jl
# ═══════════════════════════════════════════════════════════════════════════

function render_sexpr_connection(; seed::Integer=42)
    gay_seed!(seed)
    
    println()
    println("  $(BOLD)Flowers as Colored S-Expressions$(R)")
    println("  ════════════════════════════════════════════════════════════")
    println()
    println("  A flower IS a nested list structure:")
    println()
    println("    Flower = (pistil petals...)")
    println("    Petal  = (contents...)")
    println("    Atom   = symbol")
    println()
    println("  Gay.jl colors by nesting depth:")
    println()
    
    # Show depth colors
    for d in 0:4
        c = depth_color(d; seed=seed)
        println("    depth $d: $(ansi(c))████$(R) $(ansi(c))(nested content at depth $d)$(R)")
    end
    println()
    println("  $(DIM)Same principle as gay_magnetized_sexpr:$(R)")
    println("  $(DIM)  depth → color stream (interleaved SPI)$(R)")
    println("  $(DIM)  position → deterministic hue$(R)")
    println()
end

# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

function main(; seed::Integer=42)
    println()
    println("  ╔════════════════════════════════════════════════════════════════════╗")
    println("  ║          $(BOLD)THE FLOWER CALCULUS$(R) (Pablo Donato, 2025)                ║")
    println("  ║   Diagrammatic proof system — structure without symbols            ║")
    println("  ╚════════════════════════════════════════════════════════════════════╝")
    
    render_peirce_connection(seed=seed)
    render_sexpr_connection(seed=seed)
    
    # Example flowers
    println()
    println("  $(BOLD)Example: Modus Ponens$(R)")
    println("  $(DIM)(P ∧ (P → Q)) → Q$(R)")
    println()
    mp = modus_ponens_flower()
    println(render_flower(mp; seed=seed))
    println()
    println("  As S-expression: $(render_flower_sexpr(mp; seed=seed))")
    println()
    
    println()
    println("  $(BOLD)Example: Transitivity$(R)")
    println("  $(DIM)((P → Q) ∧ (Q → R)) → (P → R)$(R)")
    println()
    trans = transitivity_flower()
    println(render_flower(trans; seed=seed+1))
    println()
    println("  As S-expression: $(render_flower_sexpr(trans; seed=seed+1))")
    println()
    
    println()
    println("  $(BOLD)Example: Excluded Middle (NOT intuitionistically valid!)$(R)")
    println("  $(DIM)⊤ → (P ∨ ¬P)$(R)")
    println()
    em = excluded_middle_flower()
    println(render_flower(em; seed=seed+2))
    println()
    println("  $(DIM)This flower cannot be reduced — intuitionistic logic rejects it.$(R)")
    println()
    
    # Final
    gay_seed!(seed + 100)
    c = next_color(SRGB())
    println()
    println("  $(ansi(c))The flower blooms where symbols wither.$(R)")
    println("  $(ansi(c))Color is structure. Structure is proof.$(R)")
    println()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
