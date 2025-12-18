# Chromatic Propagator: Gay Color-Conserving Constraint Propagation
# ═══════════════════════════════════════════════════════════════════════════════
#
#   ┌─────────────────────────────────────────────────────────────────────────────┐
#   │  COLOR CONSERVATION LAW                                                    │
#   │  ═══════════════════════                                                   │
#   │                                                                             │
#   │  Every propagator operation conserves chromatic parity:                    │
#   │                                                                             │
#   │    color(x) ⊕ color(y) ⊕ color(op) = color(result)                         │
#   │                                                                             │
#   │  This enables:                                                              │
#   │    • Chromatic debugging (see where parity breaks)                         │
#   │    • Cross-runtime verification (same colors in Julia/Python/Rust)         │
#   │    • Curriculum-driven color schemes (color-logic.io compatible)           │
#   │                                                                             │
#   │  RUNTIME SUPPORT:                                                          │
#   │    Julia   → Native (this file)                                            │
#   │    Python  → color_logic_py bridge                                         │
#   │    Rust    → gay-chromatic crate                                           │
#   │    Clojure → gay.chromatic namespace                                       │
#   │    Babashka → gay-bb.clj                                                   │
#   │                                                                             │
#   └─────────────────────────────────────────────────────────────────────────────┘

module ChromaticPropagator

using ..Propagator: Cell, make_cell, cell_content, cell_strongest, add_content!
using ..Propagator: propagator, primitive_propagator, constraint_propagator
using ..Propagator: p_add, p_mul, p_sub, p_div, c_add, c_mul
using ..Propagator: Scheduler, run!, initialize_scheduler!, alert_propagator!
using ..Propagator: TheNothing, TheContradiction, isnothing_prop, is_contradiction
using ..Propagator: Supported, support_set, SupportSet
using ..GayEIntegration: GAY_E_SEED, GAY_IGOR_SEED, mix64, gay_seed, gay_color
using ..GayEIntegration: conserved_combine, DialectColors, lisp_dialect, julia_dialect
using ..GayEIntegration: ansi_color, ansi_bg, ANSI_RESET, OperatorClass

export ChromaticCell, ChromaticEnv, ChromaticCurriculum
export @cprop_str, chromatic_cell_value, chromatic_tell
export constraint_add_chromatic, constraint_mul_chromatic
export color_conservation_check, render_network_chromatic
export ColorLogicIO, color_logic_curriculum, verify_curriculum_colors

# ═══════════════════════════════════════════════════════════════════════════════
# Chromatic Cell: Cell + Color Conservation
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ChromaticCell

A propagator cell with chromatic identity for debugging and verification.
Tracks color through all operations, enabling cross-runtime verification.
"""
mutable struct ChromaticCell
    cell::Cell
    name::Symbol
    color::Tuple{Float32, Float32, Float32}
    color_seed::UInt64
    conservation_history::Vector{Tuple{Symbol, UInt64}}  # (operation, parity)
end

function ChromaticCell(name::Symbol)
    seed = gay_seed(name)
    color = gay_color(seed)
    cell = make_cell(name)
    ChromaticCell(cell, name, color, seed, [])
end

function ChromaticCell(name::Symbol, initial_value)
    cc = ChromaticCell(name)
    add_content!(cc.cell, initial_value)
    push!(cc.conservation_history, (:init, cc.color_seed))
    cc
end

# ═══════════════════════════════════════════════════════════════════════════════
# Color Conservation Operators
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ChromaticOperator

An operator that participates in color conservation.
"""
struct ChromaticOperator
    name::Symbol
    op_class::OperatorClass
    color_seed::UInt64
    color::Tuple{Float32, Float32, Float32}
end

function ChromaticOperator(name::Symbol, op_class::OperatorClass)
    seed = mix64(GAY_E_SEED ⊻ gay_seed(name) ⊻ UInt64(Int(op_class)))
    color = gay_color(seed)
    ChromaticOperator(name, op_class, seed, color)
end

# Pre-defined chromatic operators
const CHROM_ADD = ChromaticOperator(:+, OperatorClass(0))  # OP_ADDITIVE
const CHROM_SUB = ChromaticOperator(:-, OperatorClass(0))
const CHROM_MUL = ChromaticOperator(:*, OperatorClass(1))  # OP_MULTIPLICATIVE
const CHROM_DIV = ChromaticOperator(:/, OperatorClass(1))

"""
    color_conserve(c1, c2, op) -> (result_color, conserved::Bool)

Compute color of operation result while checking conservation.
"""
function color_conserve(c1::ChromaticCell, c2::ChromaticCell, op::ChromaticOperator)
    result, preserved = conserved_combine(c1.color, c2.color, op.color_seed)
    (result, preserved)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Chromatic Environment: color-logic.io compatible
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ChromaticCurriculum

Curriculum configuration for color-logic.io integration.
"""
struct ChromaticCurriculum
    levels::Vector{Symbol}
    current_level::Int
    colors_per_level::Dict{Symbol, Vector{Tuple{Float32,Float32,Float32}}}
end

ChromaticCurriculum() = ChromaticCurriculum([:basic, :intermediate, :advanced], 1, Dict())

"""
    ChromaticEnv

Environment for chromatic propagator networks with curriculum support.
"""
mutable struct ChromaticEnv
    cells::Dict{Symbol, ChromaticCell}
    propagators::Vector{Any}
    dialect::DialectColors
    curriculum::Union{Nothing, Any}  # ColorLogicIO curriculum
    conservation_log::Vector{Tuple{Symbol, Bool}}  # (operation, conserved?)
    seed::UInt64
end

function ChromaticEnv(; dialect=julia_dialect(), seed=GAY_IGOR_SEED)
    ChromaticEnv(Dict(), [], dialect, nothing, [], seed)
end

const GLOBAL_CHROMATIC_ENV = Ref{ChromaticEnv}()

function get_chromatic_env()
    if !isassigned(GLOBAL_CHROMATIC_ENV)
        GLOBAL_CHROMATIC_ENV[] = ChromaticEnv()
    end
    GLOBAL_CHROMATIC_ENV[]
end

function reset_chromatic_env!(; dialect=julia_dialect(), seed=GAY_IGOR_SEED)
    GLOBAL_CHROMATIC_ENV[] = ChromaticEnv(; dialect, seed)
    initialize_scheduler!()
end

# ═══════════════════════════════════════════════════════════════════════════════
# Chromatic DSL: color-conserving constraint language
# ═══════════════════════════════════════════════════════════════════════════════

function chromatic_define_cell(name::Symbol, initial=nothing)
    env = get_chromatic_env()
    cc = initial === nothing ? ChromaticCell(name) : ChromaticCell(name, initial)
    env.cells[name] = cc
    cc
end

function chromatic_get_cell(name::Symbol)
    env = get_chromatic_env()
    get(env.cells, name, nothing)
end

function chromatic_cell_value(name::Symbol)
    env = get_chromatic_env()
    cc = env.cells[name]
    cell_strongest(cc.cell)
end

function chromatic_tell(name::Symbol, value)
    env = get_chromatic_env()
    cc = env.cells[name]
    add_content!(cc.cell, value)
    # Log conservation event
    push!(cc.conservation_history, (:tell, gay_seed(value)))
    push!(env.conservation_log, (:tell, true))
end

"""
    constraint_add_chromatic(a, b, sum)

Create addition constraint with color conservation tracking.
z = x + y (forward)
y = z - x (backward)
x = z - y (backward)
"""
function constraint_add_chromatic(a::Symbol, b::Symbol, sum::Symbol)
    env = get_chromatic_env()
    cc_a = env.cells[a]
    cc_b = env.cells[b]
    cc_sum = env.cells[sum]
    
    # Check color conservation
    result_color, conserved = color_conserve(cc_a, cc_b, CHROM_ADD)
    push!(env.conservation_log, (:constraint_add, conserved))
    
    # Update sum cell's color based on conservation
    if conserved
        cc_sum.color = result_color
    end
    
    # Create the bidirectional constraint
    c_add(cc_a.cell, cc_b.cell, cc_sum.cell)
    
    conserved
end

"""
    constraint_mul_chromatic(a, b, product)

Create multiplication constraint with color conservation tracking.
"""
function constraint_mul_chromatic(a::Symbol, b::Symbol, product::Symbol)
    env = get_chromatic_env()
    cc_a = env.cells[a]
    cc_b = env.cells[b]
    cc_product = env.cells[product]
    
    result_color, conserved = color_conserve(cc_a, cc_b, CHROM_MUL)
    push!(env.conservation_log, (:constraint_mul, conserved))
    
    if conserved
        cc_product.color = result_color
    end
    
    c_mul(cc_a.cell, cc_b.cell, cc_product.cell)
    
    conserved
end

# ═══════════════════════════════════════════════════════════════════════════════
# color-logic.io Curriculum Integration
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ColorLogicIO

Interface to color-logic.io curriculum system.
Enables progressive learning of color conservation laws.
"""
struct ColorLogicIO
    base_url::String
    curriculum_id::String
    levels::Vector{Symbol}
    current_level::Int
    color_mappings::Dict{Symbol, Tuple{Float32, Float32, Float32}}
end

function ColorLogicIO(curriculum_id::String="gay-conservation-69")
    # Default 69-level curriculum matching vibecoding_matrix
    levels = [
        # Foundational (1-23)
        :identity, :constant, :unary_op, :binary_add, :binary_mul,
        :ternary_balanced, :xor_parity, :conservation_law, :color_merge,
        :dialectal_shift, :cross_runtime, :chromatic_debug, :support_set,
        :premise_color, :contradiction_color, :nothing_color, :scheduler_color,
        :cell_network, :propagator_graph, :constraint_web, :bidirectional,
        :interval_color, :amb_choice,
        
        # Intermediate (24-46)
        :stellar_distance, :pythagorean, :ohms_law, :temperature_convert,
        :compound_interest, :pendulum, :spring_mass, :wave_equation,
        :fourier_color, :laplace_color, :differential, :integral,
        :partial_derivative, :gradient_color, :hessian_color, :jacobian_color,
        :tensor_color, :metric_color, :curvature_color, :geodesic_color,
        :parallel_transport, :holonomy_color, :connection_color,
        
        # Advanced (47-69)
        :fiber_bundle, :sheaf_color, :cohomology_color, :homology_color,
        :exact_sequence, :spectral_sequence, :derived_functor, :adjunction_color,
        :monad_color, :comonad_color, :profunctor_color, :optic_color,
        :lens_color, :prism_color, :iso_color, :traversal_color,
        :graded_monad, :lhott_modality, :linear_color, :affine_color,
        :relevant_color, :unrestricted_color, :conservation_complete
    ]
    
    # Generate color mappings from curriculum levels
    color_mappings = Dict{Symbol, Tuple{Float32, Float32, Float32}}()
    for (i, level) in enumerate(levels)
        seed = mix64(GAY_E_SEED ⊻ UInt64(i * 1069))
        color_mappings[level] = gay_color(seed)
    end
    
    ColorLogicIO("https://color-logic.io", curriculum_id, levels, 1, color_mappings)
end

function color_logic_curriculum(name::String="gay-conservation-69")
    cli = ColorLogicIO(name)
    env = get_chromatic_env()
    env.curriculum = cli
    cli
end

function advance_curriculum!(cli::ColorLogicIO)
    if cli.current_level < length(cli.levels)
        ColorLogicIO(
            cli.base_url,
            cli.curriculum_id,
            cli.levels,
            cli.current_level + 1,
            cli.color_mappings
        )
    else
        cli  # Already at max level
    end
end

function verify_curriculum_colors(cli::ColorLogicIO, env::ChromaticEnv)
    # Check that all operations in conservation_log are conserved
    all_conserved = all(x -> x[2], env.conservation_log)
    
    # Return verification result
    (
        level = cli.levels[cli.current_level],
        level_number = cli.current_level,
        total_levels = length(cli.levels),
        operations = length(env.conservation_log),
        all_conserved = all_conserved,
        color = cli.color_mappings[cli.levels[cli.current_level]]
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# String Macro: @cprop_str (chromatic prop)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    cprop"(define-cell x)"

Chromatic propagator DSL with color conservation.
"""
macro cprop_str(str)
    # Parse and transform like @prop_str but with chromatic operations
    lines = split(str, '\n')
    exprs = []
    
    for line in lines
        line = strip(line)
        isempty(line) && continue
        startswith(line, ';') && continue  # Skip comments
        
        # Parse S-expression-like syntax
        if startswith(line, "(define-cell")
            m = match(r"\(define-cell\s+(\w+)(?:\s+(.+))?\)", line)
            if m !== nothing
                name = Symbol(m.captures[1])
                initial = m.captures[2]
                if initial === nothing
                    push!(exprs, :(chromatic_define_cell($(QuoteNode(name)))))
                else
                    push!(exprs, :(chromatic_define_cell($(QuoteNode(name)), $(Meta.parse(initial)))))
                end
            end
        elseif startswith(line, "(constraint-add")
            m = match(r"\(constraint-add\s+(\w+)\s+(\w+)\s+(\w+)\)", line)
            if m !== nothing
                a, b, c = Symbol.(m.captures)
                push!(exprs, :(constraint_add_chromatic($(QuoteNode(a)), $(QuoteNode(b)), $(QuoteNode(c)))))
            end
        elseif startswith(line, "(constraint-mul")
            m = match(r"\(constraint-mul\s+(\w+)\s+(\w+)\s+(\w+)\)", line)
            if m !== nothing
                a, b, c = Symbol.(m.captures)
                push!(exprs, :(constraint_mul_chromatic($(QuoteNode(a)), $(QuoteNode(b)), $(QuoteNode(c)))))
            end
        elseif startswith(line, "(tell")
            m = match(r"\(tell\s+(\w+)\s+(.+)\)", line)
            if m !== nothing
                name = Symbol(m.captures[1])
                value = Meta.parse(m.captures[2])
                push!(exprs, :(chromatic_tell($(QuoteNode(name)), $value)))
            end
        elseif startswith(line, "(run")
            push!(exprs, :(run!()))
        elseif startswith(line, "(reset")
            push!(exprs, :(reset_chromatic_env!()))
        end
    end
    
    esc(Expr(:block, exprs...))
end

# ═══════════════════════════════════════════════════════════════════════════════
# Chromatic Network Visualization
# ═══════════════════════════════════════════════════════════════════════════════

"""
    render_network_chromatic(env::ChromaticEnv) -> String

Render the propagator network with chromatic colors in ANSI.
"""
function render_network_chromatic(env::ChromaticEnv=get_chromatic_env())
    lines = String[]
    
    push!(lines, "╔" * "═" ^ 60 * "╗")
    push!(lines, "║  CHROMATIC PROPAGATOR NETWORK                              ║")
    push!(lines, "╠" * "═" ^ 60 * "╣")
    
    # Cells with colors
    push!(lines, "║  CELLS:                                                     ║")
    for (name, cc) in sort(collect(env.cells), by=x->string(x[1]))
        val = cell_strongest(cc.cell)
        val_str = isnothing_prop(val) ? "⊥" : string(val)
        color_block = ansi_bg(cc.color) * "  " * ANSI_RESET
        push!(lines, "║    $(rpad(string(name), 12)) = $(rpad(val_str, 10)) $color_block  ║")
    end
    
    # Conservation log
    push!(lines, "╠" * "═" ^ 60 * "╣")
    push!(lines, "║  CONSERVATION LOG:                                          ║")
    conserved_count = count(x -> x[2], env.conservation_log)
    total_count = length(env.conservation_log)
    push!(lines, "║    $conserved_count / $total_count operations conserved" * " " ^ 30 * "║")
    
    # Curriculum status
    if env.curriculum !== nothing
        cli = env.curriculum
        push!(lines, "╠" * "═" ^ 60 * "╣")
        level = cli.levels[cli.current_level]
        color_block = ansi_bg(cli.color_mappings[level]) * "  " * ANSI_RESET
        push!(lines, "║  CURRICULUM: $(cli.curriculum_id)" * " " ^ 30 * "║")
        push!(lines, "║    Level $(cli.current_level)/$(length(cli.levels)): $level $color_block ║")
    end
    
    push!(lines, "╚" * "═" ^ 60 * "╝")
    
    join(lines, "\n")
end

# ═══════════════════════════════════════════════════════════════════════════════
# Cross-Runtime Color Verification
# ═══════════════════════════════════════════════════════════════════════════════

"""
    color_conservation_check(operations::Vector) -> NamedTuple

Verify color conservation across a sequence of operations.
Returns verification result compatible with other runtimes.
"""
function color_conservation_check(operations::Vector{Tuple{Symbol, Any, Any}})
    results = []
    
    for (op, arg1, arg2) in operations
        # Compute expected color from operation
        c1 = gay_color(gay_seed(arg1))
        c2 = gay_color(gay_seed(arg2))
        op_seed = gay_seed(op)
        
        result_color, conserved = conserved_combine(c1, c2, op_seed)
        
        push!(results, (
            operation = op,
            input_colors = (c1, c2),
            result_color = result_color,
            conserved = conserved,
            parity = UInt8(round(result_color[1] * 255)) ⊻ 
                     UInt8(round(result_color[2] * 255)) ⊻ 
                     UInt8(round(result_color[3] * 255))
        ))
    end
    
    (
        operations = results,
        all_conserved = all(r -> r.conserved, results),
        total_parity = reduce(⊻, r.parity for r in results; init=UInt8(0)),
        verification_seed = GAY_E_SEED
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# Export cross-runtime color schema (for color-logic.io)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    export_color_schema_json(env::ChromaticEnv) -> String

Export the current color schema as JSON for cross-runtime use.
"""
function export_color_schema_json(env::ChromaticEnv=get_chromatic_env())
    cells_json = join([
        """{
          "name": "$(cc.name)",
          "color": [$(cc.color[1]), $(cc.color[2]), $(cc.color[3])],
          "seed": $(cc.color_seed)
        }"""
        for (name, cc) in env.cells
    ], ",\n    ")
    
    curriculum_json = if env.curriculum !== nothing
        cli = env.curriculum
        """
        "curriculum": {
          "id": "$(cli.curriculum_id)",
          "level": $(cli.current_level),
          "total_levels": $(length(cli.levels)),
          "current": "$(cli.levels[cli.current_level])"
        }"""
    else
        "\"curriculum\": null"
    end
    
    """
    {
      "schema_version": "1.0",
      "base_seed": $(GAY_E_SEED),
      "igor_seed": $(GAY_IGOR_SEED),
      "cells": [
        $cells_json
      ],
      $curriculum_json,
      "conservation_verified": $(all(x -> x[2], env.conservation_log))
    }
    """
end

# ═══════════════════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════════════════

function demo_chromatic_propagator()
    reset_chromatic_env!()
    
    # Initialize curriculum
    cli = color_logic_curriculum("gay-conservation-69")
    
    # Create chromatic cells
    cprop"""
    (define-cell x)
    (define-cell y)
    (define-cell sum)
    
    (constraint-add x y sum)
    
    (tell x 5)
    (tell sum 12)
    """
    
    run!()
    
    # Verify
    y_val = chromatic_cell_value(:y)
    @assert y_val == 7 "Inferred y should be 7, got $y_val"
    
    # Show network
    println(render_network_chromatic())
    
    # Verify curriculum
    result = verify_curriculum_colors(cli, get_chromatic_env())
    println("\n🏳️‍🌈 Curriculum verification: $result")
    
    # Export for cross-runtime
    println("\n📤 Cross-runtime schema:")
    println(export_color_schema_json())
end

end # module ChromaticPropagator
