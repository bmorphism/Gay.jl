# LispSyntax DSL for Propagator Networks
# Clojure-like syntax for SDF-style propagators

module PropagatorLisp

using LispSyntax
using ..Propagator: Cell, make_cell, cell_content, cell_strongest, add_content!
using ..Propagator: propagator, primitive_propagator, compound_propagator, constraint_propagator
using ..Propagator: Premise, hypothetical, mark_premise_in!, mark_premise_out!, is_premise_in
using ..Propagator: SupportSet, support_set, support_set_union, all_premises_in
using ..Propagator: TheNothing, TheContradiction, isnothing_prop, is_contradiction
using ..Propagator: merge_values, strongest_value
using ..Propagator: Scheduler, run!, initialize_scheduler!, alert_propagator!
using ..Propagator: p_add, p_mul, p_sub, p_div, c_add, c_mul
using ..Propagator: binary_amb, p_amb
using ..Propagator: Supported, PropagatorDef, network_palette

export @prop_str, init_propagator_repl!, PropagatorEnv
export reset_env!, get_env, define_cell, get_cell, cell_value, tell_cell
export prop_add, prop_mul, constraint_add, constraint_mul, prop_amb
export stellar_distance_example, pythagorean_example, quartic_well_example

# ============================================================
# Propagator Environment
# ============================================================

"""
    PropagatorEnv

Environment for propagator definitions, holding cells and propagators.
"""
mutable struct PropagatorEnv
    cells::Dict{Symbol, Cell}
    propagators::Vector{PropagatorDef}
    seed::UInt64
end

PropagatorEnv(seed::UInt64=0x6761795f636f6c6f) = PropagatorEnv(Dict(), [], seed)

const GLOBAL_PROP_ENV = Ref{PropagatorEnv}()

function get_env()
    if !isassigned(GLOBAL_PROP_ENV)
        GLOBAL_PROP_ENV[] = PropagatorEnv()
    end
    GLOBAL_PROP_ENV[]
end

function reset_env!(seed::UInt64=0x6761795f636f6c6f)
    GLOBAL_PROP_ENV[] = PropagatorEnv(seed)
    initialize_scheduler!()
end

# ============================================================
# Cell DSL
# ============================================================

"""
    (define-cell name)
    (define-cell name initial-value)

Define a new cell in the propagator environment.
"""
function define_cell(name::Symbol, initial=nothing)
    env = get_env()
    cell = make_cell(name)
    if initial !== nothing
        add_content!(cell, initial)
    end
    env.cells[name] = cell
    cell
end

"""
    (cell name)

Get a cell by name.
"""
function get_cell(name::Symbol)
    env = get_env()
    get(env.cells, name, nothing)
end

# ============================================================
# Propagator DSL
# ============================================================

"""
    (p:+ a b sum)

Create addition propagator: sum = a + b
"""
function prop_add(a::Symbol, b::Symbol, sum::Symbol)
    env = get_env()
    p_add(env.cells[a], env.cells[b], env.cells[sum])
end

"""
    (p:* a b product)

Create multiplication propagator: product = a * b
"""
function prop_mul(a::Symbol, b::Symbol, product::Symbol)
    env = get_env()
    p_mul(env.cells[a], env.cells[b], env.cells[product])
end

"""
    (c:+ a b sum)

Create addition constraint (bidirectional).
"""
function constraint_add(a::Symbol, b::Symbol, sum::Symbol)
    env = get_env()
    c_add(env.cells[a], env.cells[b], env.cells[sum])
end

"""
    (c:* a b product)

Create multiplication constraint (bidirectional).
"""
function constraint_mul(a::Symbol, b::Symbol, product::Symbol)
    env = get_env()
    c_mul(env.cells[a], env.cells[b], env.cells[product])
end

"""
    (p:amb cell values...)

Create n-ary choice propagator.
"""
function prop_amb(cell::Symbol, values...)
    env = get_env()
    p_amb(env.cells[cell], collect(values))
end

"""
    (content cell)

Get the strongest value of a cell.
"""
function cell_value(name::Symbol)
    env = get_env()
    cell = env.cells[name]
    cell_strongest(cell)
end

"""
    (tell cell value)

Add content to a cell.
"""
function tell_cell(name::Symbol, value)
    env = get_env()
    cell = env.cells[name]
    add_content!(cell, value)
end

# ============================================================
# Special Forms for LispSyntax
# ============================================================

# Register special reader dispatch for propagator-specific syntax
function setup_reader!()
    # p: prefix for primitive propagators
    LispSyntax.assign_reader_dispatch(:p, (sym, args...) -> begin
        op = Symbol("prop_$(sym)")
        [:call, op, args...]
    end)
    
    # c: prefix for constraint propagators
    LispSyntax.assign_reader_dispatch(:c, (sym, args...) -> begin
        op = Symbol("constraint_$(sym)")
        [:call, op, args...]
    end)
end

# ============================================================
# Propagator String Macro
# ============================================================

"""
    prop"(define-cell x 42)"

Evaluate propagator DSL expressions.
"""
macro prop_str(str)
    # Parse with LispSyntax
    sexpr = LispSyntax.desx(LispSyntax.read(str))
    
    # Transform to propagator calls
    transformed = transform_prop_sexpr(sexpr)
    
    # Generate Julia code
    code = LispSyntax.codegen(transformed)
    esc(code)
end

function transform_prop_sexpr(s)
    if !isa(s, Array) || isempty(s)
        return s
    end
    
    head = s[1]
    
    if head == Symbol("define-cell")
        name = s[2]
        initial = length(s) > 2 ? s[3] : nothing
        return [:call, :define_cell, QuoteNode(name), initial]
    
    elseif head == Symbol("let-cells")
        # (let-cells (x y z) body...)
        # → (do (define-cell x) (define-cell y) (define-cell z) body...)
        bindings = s[2]
        body = s[3:end]
        defs = [[:call, :define_cell, QuoteNode(b)] for b in bindings]
        return [:do, defs..., map(transform_prop_sexpr, body)...]
    
    elseif head == :content
        return [:call, :cell_value, QuoteNode(s[2])]
    
    elseif head == :tell
        return [:call, :tell_cell, QuoteNode(s[2]), transform_prop_sexpr(s[3])]
    
    elseif head == Symbol("p:+")
        return [:call, :prop_add, QuoteNode.(s[2:4])...]
    
    elseif head == Symbol("p:*")
        return [:call, :prop_mul, QuoteNode.(s[2:4])...]
    
    elseif head == Symbol("c:+")
        return [:call, :constraint_add, QuoteNode.(s[2:4])...]
    
    elseif head == Symbol("c:*")
        return [:call, :constraint_mul, QuoteNode.(s[2:4])...]
    
    elseif head == Symbol("p:amb")
        cell = s[2]
        values = s[3:end]
        return [:call, :prop_amb, QuoteNode(cell), values...]
    
    elseif head == :run
        return [:call, :run!]
    
    elseif head == :reset
        seed = length(s) > 1 ? s[2] : 0x6761795f636f6c6f
        return [:call, :reset_env!, seed]
    
    else
        # Recursively transform
        return map(transform_prop_sexpr, s)
    end
end

# ============================================================
# REPL Mode
# ============================================================

function init_propagator_repl!()
    # Initialize environment
    reset_env!()
    
    # Create REPL mode using ReplMaker pattern
    println("Propagator REPL initialized. Use prop\"...\" macro or:")
    println("  (define-cell x)")
    println("  (tell x 42)")
    println("  (c:+ x y z)")
    println("  (run)")
    println("  (content z)")
end

# ============================================================
# Example: Stellar Distance (SDF §7.1)
# ============================================================

"""
    stellar_distance_example()

Recreate the Vega parallax example from SDF §7.1.
"""
function stellar_distance_example()
    reset_env!()
    
    # Define cells
    define_cell(:vega_parallax)
    define_cell(:vega_distance)
    define_cell(:t)
    define_cell(:AU, 4.848136811095e-6)  # AU in parsecs
    
    # Build constraint network: tan(parallax) × distance = AU
    # Simplified: t × distance = AU where t = tan(parallax)
    constraint_mul(:t, :vega_distance, :AU)
    
    # Add Struve's 1837 measurement: 0.125" ± 0.05"
    # Convert arcseconds to radians: 1" = 4.848e-6 rad
    struve_parallax = (0.075 * 4.848e-6, 0.175 * 4.848e-6)  # interval
    
    tell_cell(:vega_parallax, struve_parallax)
    # Propagate tan (simplified - just use the value directly for now)
    tell_cell(:t, (tan(struve_parallax[1]), tan(struve_parallax[2])))
    
    run!()
    
    distance = cell_value(:vega_distance)
    println("Vega distance from Struve measurement: $distance parsecs")
    
    get_env()
end

# ============================================================
# Example: Pythagorean Triples (SDF §7.5)
# ============================================================

"""
    pythagorean_example()

Find Pythagorean triples using amb, from SDF §7.5.
"""
function pythagorean_example()
    reset_env!()
    
    possibilities = 1:10
    
    # Create cells
    define_cell(:x)
    define_cell(:y)
    define_cell(:z)
    define_cell(:x2)
    define_cell(:y2)
    define_cell(:z2)
    
    # Add amb choices
    prop_amb(:x, possibilities...)
    prop_amb(:y, possibilities...)
    prop_amb(:z, possibilities...)
    
    # x² + y² = z²
    prop_mul(:x, :x, :x2)
    prop_mul(:y, :y, :y2)
    prop_mul(:z, :z, :z2)
    constraint_add(:x2, :y2, :z2)
    
    run!()
    
    x = cell_value(:x)
    y = cell_value(:y)
    z = cell_value(:z)
    
    println("Pythagorean triple: ($x, $y, $z)")
    
    get_env()
end

# ============================================================
# Lagrangian Mechanics (inspired by screenshot)
# ============================================================

"""
    quartic_well_example(; mass=1.0, α=1.0, β=2.0, γ=0.0)

Quartic double-well potential: V(x) = αx⁴ - βx² + γ
"""
function quartic_well_example(; mass=1.0, α=1.0, β=2.0, γ=0.0)
    reset_env!()
    
    # Cells for Lagrangian mechanics
    define_cell(:x)         # position
    define_cell(:v)         # velocity  
    define_cell(:T)         # kinetic energy
    define_cell(:V)         # potential energy
    define_cell(:L)         # Lagrangian
    define_cell(:x2)        # x²
    define_cell(:x4)        # x⁴
    define_cell(:v2)        # v²
    define_cell(:half_m, mass / 2)   # m/2
    
    # T = ½mv² (kinetic energy)
    # v² 
    constraint_mul(:v, :v, :v2)
    # T = (m/2) × v²
    constraint_mul(:half_m, :v2, :T)
    
    # V = αx⁴ - βx² + γ (potential energy)
    # Need intermediate cells
    define_cell(:αx4)
    define_cell(:βx2)
    define_cell(:V_partial)
    
    define_cell(:α, α)
    define_cell(:β, β)
    define_cell(:γ, γ)
    
    # x² and x⁴
    constraint_mul(:x, :x, :x2)
    constraint_mul(:x2, :x2, :x4)
    
    # αx⁴
    constraint_mul(:α, :x4, :αx4)
    # βx²  
    constraint_mul(:β, :x2, :βx2)
    
    # L = T - V
    # This requires subtraction constraint
    # For now, leave as exercise
    
    println("Quartic well network constructed")
    println("Set position: tell_cell(:x, value)")
    println("Set velocity: tell_cell(:v, value)")
    println("Run: run!()")
    
    get_env()
end

end # module PropagatorLisp
