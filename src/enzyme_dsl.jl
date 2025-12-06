# Enzyme DSL: Define differentiable functions with S-expression syntax
#
# (defenzyme name (args...) body)
#   - Defines a Julia function from S-expression
#   - Creates paired GayDifferentiable for visualization
#   - Tracks binding sites for gradient attachment
#
# From SICP 4A: The pattern-matcher is itself data that can be manipulated.
# Here, the S-expression IS the function definition, and enzymes attach to it.

using LispSyntax

export @defenzyme, @gay_gradient, @gay_forward, @gay_reverse
export compile_sexpr, sexpr_to_julia, GayFunction

# ═══════════════════════════════════════════════════════════════════════════
# S-expression → Julia compilation
# ═══════════════════════════════════════════════════════════════════════════

"""
    sexpr_to_julia(expr)

Convert an S-expression to Julia code.
Handles arithmetic, transcendentals, and control flow.
"""
function sexpr_to_julia(expr)
    if expr isa Symbol
        return expr
    elseif expr isa Number
        return expr
    elseif expr isa AbstractVector || expr isa Tuple
        if isempty(expr)
            return :()
        end
        
        op = expr[1]
        args = expr[2:end]
        
        # Arithmetic operators
        if op == :+ 
            return Expr(:call, :+, map(sexpr_to_julia, args)...)
        elseif op == :- 
            if length(args) == 1
                return Expr(:call, :-, sexpr_to_julia(args[1]))
            else
                return Expr(:call, :-, map(sexpr_to_julia, args)...)
            end
        elseif op == :* 
            return Expr(:call, :*, map(sexpr_to_julia, args)...)
        elseif op == :/ 
            return Expr(:call, :/, map(sexpr_to_julia, args)...)
        elseif op == :^ || op == :pow || op == :expt
            return Expr(:call, :^, map(sexpr_to_julia, args)...)
        elseif op == :sqrt
            return Expr(:call, :sqrt, sexpr_to_julia(args[1]))
        elseif op == :abs
            return Expr(:call, :abs, sexpr_to_julia(args[1]))
        
        # Transcendentals
        elseif op == :sin
            return Expr(:call, :sin, sexpr_to_julia(args[1]))
        elseif op == :cos
            return Expr(:call, :cos, sexpr_to_julia(args[1]))
        elseif op == :tan
            return Expr(:call, :tan, sexpr_to_julia(args[1]))
        elseif op == :exp
            return Expr(:call, :exp, sexpr_to_julia(args[1]))
        elseif op == :log
            return Expr(:call, :log, sexpr_to_julia(args[1]))
        elseif op == :tanh
            return Expr(:call, :tanh, sexpr_to_julia(args[1]))
        elseif op == :sigmoid || op == :σ
            x = sexpr_to_julia(args[1])
            return :(1.0 / (1.0 + exp(-$x)))
        elseif op == :relu
            x = sexpr_to_julia(args[1])
            return :(max(0.0, $x))
        
        # Comparison
        elseif op == :< 
            return Expr(:call, :<, map(sexpr_to_julia, args)...)
        elseif op == :> 
            return Expr(:call, :>, map(sexpr_to_julia, args)...)
        elseif op == :<= 
            return Expr(:call, :<=, map(sexpr_to_julia, args)...)
        elseif op == :>= 
            return Expr(:call, :>=, map(sexpr_to_julia, args)...)
        elseif op == :eq || op == Symbol("==")
            return Expr(:call, :(==), map(sexpr_to_julia, args)...)
        
        # Control flow
        elseif op == :if
            cond = sexpr_to_julia(args[1])
            then_branch = sexpr_to_julia(args[2])
            else_branch = length(args) >= 3 ? sexpr_to_julia(args[3]) : nothing
            return Expr(:if, cond, then_branch, else_branch)
        
        # Let bindings
        elseif op == :let
            bindings = args[1]
            body = args[2:end]
            
            # Convert [(x val) (y val2)] to Julia let
            bind_exprs = []
            for b in bindings
                var = b[1]
                val = sexpr_to_julia(b[2])
                push!(bind_exprs, :($var = $val))
            end
            
            body_expr = length(body) == 1 ? sexpr_to_julia(body[1]) : 
                        Expr(:block, map(sexpr_to_julia, body)...)
            
            return Expr(:let, Expr(:block, bind_exprs...), body_expr)
        
        # Lambda
        elseif op == :lambda || op == :λ || op == :fn
            params = args[1]
            body = args[2]
            return Expr(:->, Expr(:tuple, params...), sexpr_to_julia(body))
        
        # Matrix operations (for neural nets)
        elseif op == :matmul || op == Symbol("@")
            return Expr(:call, :*, map(sexpr_to_julia, args)...)
        elseif op == :dot
            return Expr(:call, :dot, map(sexpr_to_julia, args)...)
        elseif op == :sum
            return Expr(:call, :sum, sexpr_to_julia(args[1]))
        elseif op == :mean
            return Expr(:call, :mean, sexpr_to_julia(args[1]))
        
        # Default: treat as function call
        else
            return Expr(:call, op, map(sexpr_to_julia, args)...)
        end
    else
        return expr
    end
end

"""
    compile_sexpr(expr, args::Vector{Symbol})

Compile S-expression to a Julia function.
"""
function compile_sexpr(expr, args::Vector{Symbol})
    body = sexpr_to_julia(expr)
    func_expr = Expr(:->, Expr(:tuple, args...), body)
    return Base.invokelatest(eval, func_expr)
end

# ═══════════════════════════════════════════════════════════════════════════
# GayFunction: S-expression + Julia function + enzyme binding
# ═══════════════════════════════════════════════════════════════════════════

"""
    GayFunction

A function defined by S-expression with Gay.jl visualization.
Combines:
- Original S-expression (for pattern matching / visualization)
- Compiled Julia function (for execution)
- GayDifferentiable wrapper (for autodiff + colors)

# Example
```julia
gf = GayFunction(:quadratic, [:x, :y], [:+, [:^, :x, 2], [:*, 2, [:*, :x, :y]], [:^, :y, 2]])
gf(3.0, 4.0)  # => 49.0
show_gradient(gf, [3.0, 4.0])  # Visualize gradient flow
```
"""
struct GayFunction
    name::Symbol
    args::Vector{Symbol}
    sexpr::Any
    julia_func::Function
    differentiable::GayDifferentiable
    seed::UInt64
end

function GayFunction(name::Symbol, args::Vector{Symbol}, sexpr; seed::UInt64=0x454E5A594D450000)
    julia_func = compile_sexpr(sexpr, args)
    diff = GayDifferentiable(julia_func, sexpr; seed=seed, n_inputs=length(args))
    GayFunction(name, args, sexpr, julia_func, diff, seed)
end

# Call the function (use invokelatest to handle world age)
(gf::GayFunction)(inputs...) = Base.invokelatest(gf.julia_func, inputs...)

"""
    show_gradient(gf::GayFunction, inputs::Vector{Float64}; mode::EnzymeMode=Reverse)

Visualize gradient flow through the function.
"""
function show_gradient(gf::GayFunction, inputs::Vector{Float64}; mode::EnzymeMode=Reverse)
    ge = gay_autodiff(gf.differentiable, inputs; mode=mode)
    
    println("\n$(gf.name) at $(gf.args) = $inputs")
    println("  Value: $(gf(inputs...))")
    println("  Gradient visualization:")
    println("    ", gay_render_enzyme(ge))
    println("  Gradient magnetization: ", round(gradient_magnetization(ge), digits=4))
    
    return ge
end

export show_gradient

# ═══════════════════════════════════════════════════════════════════════════
# Macro interface
# ═══════════════════════════════════════════════════════════════════════════

"""
    @defenzyme name(args...) = sexpr

Define a differentiable function with S-expression body.

# Example
```julia
@defenzyme rosenbrock(x, y) = [:+, 
    [:^, [:-, 1, :x], 2], 
    [:*, 100, [:^, [:-, :y, [:^, :x, 2]], 2]]]

rosenbrock(1.0, 1.0)  # => 0.0 (global minimum)
gay_gradient(rosenbrock, [0.5, 0.5])  # Visualize gradient
```
"""
macro defenzyme(expr)
    if expr.head == :(=)
        call = expr.args[1]
        body = expr.args[2]
        
        name = call.args[1]
        args = call.args[2:end]
        
        return quote
            const $(esc(name)) = GayFunction(
                $(QuoteNode(name)),
                Symbol[$(map(QuoteNode, args)...)],
                $(esc(body));
                seed = UInt64(hash($(QuoteNode(name))))
            )
        end
    else
        error("@defenzyme expects: @defenzyme name(args...) = sexpr")
    end
end

"""
    @gay_gradient(gf, inputs...)

Compute and visualize gradient.
"""
macro gay_gradient(gf, inputs...)
    return quote
        show_gradient($(esc(gf)), Float64[$(map(esc, inputs)...)])
    end
end

"""
    @gay_forward(gf, inputs...)

Compute forward-mode gradient.
"""
macro gay_forward(gf, inputs...)
    return quote
        show_gradient($(esc(gf)), Float64[$(map(esc, inputs)...)]; mode=Forward)
    end
end

"""
    @gay_reverse(gf, inputs...)

Compute reverse-mode gradient.
"""
macro gay_reverse(gf, inputs...)
    return quote
        show_gradient($(esc(gf)), Float64[$(map(esc, inputs)...)]; mode=Reverse)
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# Lisp interface for defining enzymes
# ═══════════════════════════════════════════════════════════════════════════

"""
Lisp-style function definition with enzyme binding.

Usage:
  (defenzyme quadratic (x y) (+ (^ x 2) (* 2 (* x y)) (^ y 2)))
  (quadratic 3 4)  ; => 49
  (gay-grad quadratic 3 4)  ; Visualize gradient
"""

# Parse Lisp defenzyme and create GayFunction
function lisp_defenzyme(name::Symbol, args, body)
    arg_symbols = Symbol[a isa Symbol ? a : Symbol(a) for a in args]
    return GayFunction(name, arg_symbols, body)
end

# Registry of defined enzyme functions
const ENZYME_REGISTRY = Dict{Symbol, GayFunction}()

function register_enzyme(name::Symbol, gf::GayFunction)
    ENZYME_REGISTRY[name] = gf
end

function get_enzyme(name::Symbol)
    return get(ENZYME_REGISTRY, name, nothing)
end

export lisp_defenzyme, register_enzyme, get_enzyme, ENZYME_REGISTRY

# ═══════════════════════════════════════════════════════════════════════════
# Example functions
# ═══════════════════════════════════════════════════════════════════════════

"""
    demo_enzyme_dsl()

Demonstrate the enzyme DSL with classic optimization functions.
"""
function demo_enzyme_dsl()
    println("\n╔══════════════════════════════════════════════════════════════╗")
    println("║  Gay.jl Enzyme DSL: S-expression → Autodiff Visualization    ║")
    println("╚══════════════════════════════════════════════════════════════╝\n")
    
    # Define some classic functions
    
    # Quadratic: f(x,y) = x² + 2xy + y² = (x+y)²
    quadratic = GayFunction(:quadratic, [:x, :y], 
                           [:+, [:^, :x, 2], [:*, 2, [:*, :x, :y]], [:^, :y, 2]])
    
    println("1. Quadratic: (+ (^ x 2) (* 2 (* x y)) (^ y 2))")
    println("   f(3,4) = ", quadratic(3.0, 4.0))
    show_gradient(quadratic, [3.0, 4.0])
    println()
    
    # Rosenbrock: f(x,y) = (1-x)² + 100(y-x²)²
    rosenbrock = GayFunction(:rosenbrock, [:x, :y],
                            [:+, [:^, [:-, 1, :x], 2], 
                                 [:*, 100, [:^, [:-, :y, [:^, :x, 2]], 2]]])
    
    println("2. Rosenbrock: (+ (^ (- 1 x) 2) (* 100 (^ (- y (^ x 2)) 2)))")
    println("   f(1,1) = ", rosenbrock(1.0, 1.0), " (global minimum)")
    println("   f(0,0) = ", rosenbrock(0.0, 0.0))
    show_gradient(rosenbrock, [0.0, 0.0])
    println()
    
    # Neural network layer: σ(Wx + b) simplified as σ(ax + b)
    nn_layer = GayFunction(:nn_layer, [:x, :a, :b],
                          [:tanh, [:+, [:*, :a, :x], :b]])
    
    println("3. NN Layer: (tanh (+ (* a x) b))")
    println("   f(1, 2, 0.5) = ", nn_layer(1.0, 2.0, 0.5))
    show_gradient(nn_layer, [1.0, 2.0, 0.5])
    println()
    
    # Show the SICP connection
    println("────────────────────────────────────────────────────────────────")
    println()
    println("SICP 4A: 'The pattern-matcher is itself data'")
    println()
    println("In Gay.jl:")
    println("  • S-expression = function definition = visualizable structure")
    println("  • Enzyme bindings attach to colored parentheses")
    println("  • Gradient flow follows the expression tree")
    println("  • Colors encode sensitivity magnitude and direction")
    println()
end

export demo_enzyme_dsl
