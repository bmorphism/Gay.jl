# Enzyme.jl integration: SICP 4A "enzymes attach to expressions" made literal
#
# From SICP Lecture 4A: "Enzymes attach to expressions, change them, then go away.
# The key-in-lock phenomenon." - Hal Abelson
#
# Here, Enzyme.jl autodiff literally attaches to colored S-expression binding sites.
# Each colored parenthesis is a potential binding site where derivatives can attach.
#
# Gradient colors:
#   - Forward mode: blue stream (∂f/∂x flows forward)
#   - Reverse mode: red stream (∂L/∂f flows backward)
#   - Mixed mode: purple (bidirectional flow)
#
# Sensitivity spins:
#   - σ = +1: positive sensitivity (increasing input → increasing output)
#   - σ = -1: negative sensitivity (increasing input → decreasing output)
#   - |∂f/∂x| encoded in color saturation

using LispSyntax
using Colors

export EnzymeBinding, GayEnzyme, EnzymeMode
export attach_enzyme, detach_enzyme, propagate_gradient
export gay_autodiff, gay_gradient_color, gay_sensitivity_spin
export enzyme_forward, enzyme_reverse, enzyme_mixed
export GayDifferentiable, differentiate_sexpr, gradient_magnetization

# ═══════════════════════════════════════════════════════════════════════════
# Enzyme binding modes (like Enzyme.jl's Forward/Reverse)
# ═══════════════════════════════════════════════════════════════════════════

"""
    EnzymeMode

Autodiff mode determines color stream for gradient visualization.
"""
@enum EnzymeMode begin
    Forward   # Blue stream - tangent propagation ∂f/∂x
    Reverse   # Red stream - adjoint propagation ∂L/∂f  
    Mixed     # Purple - checkpointing/bidirectional
end

# Stream indices for GayInterleaver
const FORWARD_STREAM = 0   # Even parity - blue
const REVERSE_STREAM = 1   # Odd parity - red

"""
    EnzymeBinding

An autodiff "enzyme" attached to a colored binding site.
Represents the derivative information at a node in the expression tree.

# Fields
- `site_color`: Original Gay.jl color of the binding site
- `gradient_color`: Color encoding the gradient magnitude/direction
- `mode`: Forward or Reverse autodiff mode
- `sensitivity`: ∂output/∂input at this site
- `spin`: Sign of sensitivity as Ising spin σ ∈ {-1, +1}
- `depth`: Depth in expression tree (determines which stream)
- `position`: Position in stream (deterministic from seed)
"""
struct EnzymeBinding
    site_color::RGB
    gradient_color::RGB
    mode::EnzymeMode
    sensitivity::Float64
    spin::Int
    depth::Int
    position::Int
end

"""
    GayEnzyme

A differentiable S-expression with enzyme bindings at each node.
Extends GaySexpr with autodiff capability.

# Semantics
- Each node can have an enzyme "attached" (active) or not
- Attached enzymes carry gradient information
- Colors blend: site_color ⊕ gradient_color based on sensitivity magnitude
- Spins flip when gradients are negative
"""
mutable struct GayEnzyme
    sexpr::GaySexpr                    # Original magnetized S-expression
    bindings::Vector{EnzymeBinding}    # Enzyme attachments (parallel to tree traversal)
    mode::EnzymeMode                   # Global mode
    seed::UInt64                       # For deterministic gradient colors
    active::Bool                       # Whether enzyme is currently attached
end

# ═══════════════════════════════════════════════════════════════════════════
# Gradient coloring: sensitivity → color
# ═══════════════════════════════════════════════════════════════════════════

"""
    gay_gradient_color(sensitivity::Float64, mode::EnzymeMode; seed::UInt64=0xENZYME)

Generate a deterministic color for a gradient value.
- Hue from mode: Forward=blue(240°), Reverse=red(0°), Mixed=purple(280°)
- Saturation from |sensitivity| (higher = more saturated)
- Lightness from sign (positive=light, negative=dark)
"""
function gay_gradient_color(sensitivity::Float64, mode::EnzymeMode; 
                            seed::UInt64=0x454E5A594D450000)  # "ENZYME\0\0"
    # Base hue from mode
    base_hue = if mode == Forward
        240.0  # Blue
    elseif mode == Reverse
        0.0    # Red
    else
        280.0  # Purple/Magenta
    end
    
    # Perturb hue deterministically from seed
    hue_offset = ((seed % 1000) / 1000.0) * 30.0 - 15.0  # ±15°
    hue = mod(base_hue + hue_offset, 360.0)
    
    # Saturation from magnitude (log scale for large ranges)
    abs_sens = abs(sensitivity)
    saturation = clamp(0.3 + 0.6 * tanh(abs_sens), 0.3, 0.9)
    
    # Lightness from sign
    lightness = sensitivity >= 0 ? 0.6 : 0.35
    
    hsl = HSL(hue, saturation, lightness)
    return convert(RGB, hsl)
end

"""
    gay_sensitivity_spin(sensitivity::Float64) -> Int

Convert sensitivity to Ising spin.
σ = +1 for positive sensitivity, -1 for negative.
"""
gay_sensitivity_spin(sensitivity::Float64) = sensitivity >= 0 ? 1 : -1

# ═══════════════════════════════════════════════════════════════════════════
# Enzyme attachment: "key-in-lock" binding
# ═══════════════════════════════════════════════════════════════════════════

"""
    attach_enzyme(gs::GaySexpr, mode::EnzymeMode=Forward; seed::UInt64=0xENZYME)

Attach an autodiff enzyme to a magnetized S-expression.
Each binding site (colored paren) gets a potential attachment point.
Initially, all sensitivities are 1.0 (identity gradient).
"""
function attach_enzyme(gs::GaySexpr, mode::EnzymeMode=Forward;
                       seed::UInt64=0x454E5A594D450000)
    bindings = EnzymeBinding[]
    
    function attach_node(node::GaySexpr, idx::Ref{Int})
        # Create binding for this node
        site_color = convert(RGB, node.color)
        sensitivity = 1.0  # Will be updated during differentiation
        gradient_color = gay_gradient_color(sensitivity, mode; seed=seed ⊻ UInt64(idx[]))
        spin = gay_sensitivity_spin(sensitivity)
        
        binding = EnzymeBinding(
            site_color, gradient_color, mode,
            sensitivity, spin, node.depth, node.position
        )
        push!(bindings, binding)
        idx[] += 1
        
        # Recurse to children
        for child in node.children
            attach_node(child, idx)
        end
    end
    
    attach_node(gs, Ref(0))
    
    return GayEnzyme(gs, bindings, mode, seed, true)
end

"""
    detach_enzyme(ge::GayEnzyme) -> GaySexpr

Detach the enzyme, returning the original S-expression.
"The enzyme goes away" - SICP 4A
"""
function detach_enzyme(ge::GayEnzyme)
    ge.active = false
    return ge.sexpr
end

# ═══════════════════════════════════════════════════════════════════════════
# Gradient propagation through the tree
# ═══════════════════════════════════════════════════════════════════════════

"""
    propagate_gradient(ge::GayEnzyme, gradients::Vector{Float64})

Update enzyme bindings with computed gradients.
This simulates what Enzyme.jl does during autodiff.
"""
function propagate_gradient(ge::GayEnzyme, gradients::Vector{Float64})
    @assert length(gradients) == length(ge.bindings) "Gradient length mismatch"
    
    new_bindings = EnzymeBinding[]
    for (i, (binding, grad)) in enumerate(zip(ge.bindings, gradients))
        new_color = gay_gradient_color(grad, ge.mode; seed=ge.seed ⊻ UInt64(i))
        new_spin = gay_sensitivity_spin(grad)
        
        new_binding = EnzymeBinding(
            binding.site_color, new_color, ge.mode,
            grad, new_spin, binding.depth, binding.position
        )
        push!(new_bindings, new_binding)
    end
    
    ge.bindings = new_bindings
    return ge
end

"""
    gradient_magnetization(ge::GayEnzyme)

Compute gradient magnetization: average spin weighted by |sensitivity|.
⟨M_∇⟩ = Σ(σᵢ × |∂f/∂xᵢ|) / Σ|∂f/∂xᵢ|

This tells us the "net direction" of the gradient field.
"""
function gradient_magnetization(ge::GayEnzyme)
    total_weighted_spin = 0.0
    total_weight = 0.0
    
    for binding in ge.bindings
        weight = abs(binding.sensitivity)
        total_weighted_spin += binding.spin * weight
        total_weight += weight
    end
    
    return total_weight > 0 ? total_weighted_spin / total_weight : 0.0
end

# ═══════════════════════════════════════════════════════════════════════════
# Lisp interface for autodiff
# ═══════════════════════════════════════════════════════════════════════════

"""
    enzyme_forward(expr, seed::Integer=0xENZYME)

Create a forward-mode differentiable S-expression.
Blue stream colors indicate tangent flow.
"""
function enzyme_forward(expr, seed::Integer=0x454E5A594D450000)
    gs = gay_magnetized_sexpr(expr, seed)
    return attach_enzyme(gs, Forward; seed=UInt64(seed))
end

"""
    enzyme_reverse(expr, seed::Integer=0xENZYME)

Create a reverse-mode differentiable S-expression.
Red stream colors indicate adjoint flow.
"""
function enzyme_reverse(expr, seed::Integer=0x454E5A594D450000)
    gs = gay_magnetized_sexpr(expr, seed)
    return attach_enzyme(gs, Reverse; seed=UInt64(seed))
end

"""
    enzyme_mixed(expr, seed::Integer=0xENZYME)

Create a mixed-mode differentiable S-expression.
Purple colors indicate bidirectional/checkpointed flow.
"""
function enzyme_mixed(expr, seed::Integer=0x454E5A594D450000)
    gs = gay_magnetized_sexpr(expr, seed)
    return attach_enzyme(gs, Mixed; seed=UInt64(seed))
end

# ═══════════════════════════════════════════════════════════════════════════
# Rendering enzyme-attached expressions
# ═══════════════════════════════════════════════════════════════════════════

"""
    gay_render_enzyme(ge::GayEnzyme; show_gradients::Bool=true)

Render a GayEnzyme with ANSI colors.
If show_gradients=true, blends site color with gradient color.
"""
function gay_render_enzyme(ge::GayEnzyme; show_gradients::Bool=true)
    R = "\e[0m"
    binding_idx = Ref(1)
    
    function render_node(node::GaySexpr)
        binding = ge.bindings[binding_idx[]]
        binding_idx[] += 1
        
        # Choose color: blend site + gradient if active
        color = if ge.active && show_gradients
            # Blend: more gradient color for higher sensitivity
            α = clamp(abs(binding.sensitivity) / 2.0, 0.0, 0.8)
            r = (1 - α) * binding.site_color.r + α * binding.gradient_color.r
            g = (1 - α) * binding.site_color.g + α * binding.gradient_color.g
            b = (1 - α) * binding.site_color.b + α * binding.gradient_color.b
            RGB(r, g, b)
        else
            binding.site_color
        end
        
        r = round(Int, clamp(color.r, 0, 1) * 255)
        g = round(Int, clamp(color.g, 0, 1) * 255)
        b = round(Int, clamp(color.b, 0, 1) * 255)
        fg = "\e[38;2;$(r);$(g);$(b)m"
        
        # Show gradient info in superscript
        spin_char = binding.spin > 0 ? "⁺" : "⁻"
        mode_char = if ge.mode == Forward
            "→"
        elseif ge.mode == Reverse
            "←"
        else
            "⇄"
        end
        
        if isempty(node.children)
            # Leaf
            return "$(fg)$(node.content)$(R)"
        else
            # Compound with enzyme annotation
            inner = join([render_node(c) for c in node.children], " ")
            annotation = ge.active ? "$(spin_char)$(mode_char)" : ""
            return "$(fg)($(annotation)$(R)$(inner)$(fg))$(R)"
        end
    end
    
    return render_node(ge.sexpr)
end

# ═══════════════════════════════════════════════════════════════════════════
# Integration with actual Enzyme.jl
# ═══════════════════════════════════════════════════════════════════════════

# Enzyme.jl is loaded conditionally to avoid hard dependency
const ENZYME_LOADED = Ref(false)

function __init_enzyme__()
    try
        @eval using Enzyme
        ENZYME_LOADED[] = true
        @info "Gay.jl: Enzyme.jl loaded - autodiff visualization enabled"
    catch
        @debug "Gay.jl: Enzyme.jl not available - using simulated gradients"
    end
end

"""
    GayDifferentiable{F}

Wrapper for a Julia function with Gay.jl visualization.
When differentiated with Enzyme.jl, colors track gradient flow.

# Example
```julia
using LispSyntax

# Define function and its S-expression representation
f(x, y) = x^2 + 2*x*y + y^2
sexpr = lisp"(+ (^ x 2) (* 2 (* x y)) (^ y 2))"

gd = GayDifferentiable(f, sexpr; seed=0x42)
gd(3.0, 4.0)  # Evaluate: 49.0

# Visualize gradient flow
ge = gay_autodiff(gd, [3.0, 4.0])
println(gay_render_enzyme(ge))
```
"""
struct GayDifferentiable{F}
    f::F
    sexpr::Any  # S-expression representation
    enzyme::Union{Nothing, GayEnzyme}
    seed::UInt64
    n_inputs::Int
end

function GayDifferentiable(f::F, sexpr; seed::UInt64=0x454E5A594D450000, n_inputs::Int=1) where F
    GayDifferentiable{F}(f, sexpr, nothing, seed, n_inputs)
end

# Call the underlying function
(gd::GayDifferentiable)(args...) = gd.f(args...)

"""
    differentiate_sexpr(gd::GayDifferentiable, mode::EnzymeMode=Reverse)

Prepare S-expression for differentiation visualization.
Returns a GayEnzyme ready for gradient propagation.
"""
function differentiate_sexpr(gd::GayDifferentiable, mode::EnzymeMode=Reverse)
    gs = gay_magnetized_sexpr(gd.sexpr, gd.seed)
    return attach_enzyme(gs, mode; seed=gd.seed)
end

"""
    gay_autodiff(gd::GayDifferentiable, inputs::Vector{Float64}; mode::EnzymeMode=Reverse)

Compute gradients using Enzyme.jl and visualize with Gay.jl colors.
Returns a GayEnzyme with gradient-colored binding sites.

If Enzyme.jl is not loaded, uses finite differences as fallback.
"""
function gay_autodiff(gd::GayDifferentiable, inputs::Vector{Float64}; 
                      mode::EnzymeMode=Reverse)
    ge = differentiate_sexpr(gd, mode)
    
    # Compute actual gradients
    gradients = if ENZYME_LOADED[]
        enzyme_gradient(gd.f, inputs, mode)
    else
        finite_diff_gradient(gd.f, inputs)
    end
    
    # Map gradients to tree nodes (simplified: distribute based on magnitude)
    n_nodes = length(ge.bindings)
    node_grads = distribute_gradients(gradients, n_nodes, gd.seed)
    
    propagate_gradient(ge, node_grads)
    return ge
end

"""
    enzyme_gradient(f, inputs, mode)

Compute gradient using Enzyme.jl.
"""
function enzyme_gradient(f, inputs::Vector{Float64}, mode::EnzymeMode)
    if !ENZYME_LOADED[]
        return finite_diff_gradient(f, inputs)
    end
    
    n = length(inputs)
    grads = zeros(Float64, n)
    
    # Use Enzyme's autodiff
    @eval begin
        using Enzyme
        for i in 1:$n
            # Create tangent vector (one-hot)
            tangent = zeros($n)
            tangent[i] = 1.0
            
            if $mode == Forward
                # Forward mode: compute directional derivative
                _, grad = Enzyme.autodiff(Forward, $f, Duplicated, 
                                          [Duplicated($inputs[j], tangent[j]) for j in 1:$n]...)
                $grads[i] = grad
            else
                # Reverse mode: compute gradient
                shadow = zeros($n)
                Enzyme.autodiff(Reverse, $f, Active, 
                               [Duplicated($inputs[j], Ref(shadow[j])) for j in 1:$n]...)
                $grads[i] = shadow[i]
            end
        end
    end
    
    return grads
end

"""
    finite_diff_gradient(f, inputs; ε=1e-7)

Fallback gradient computation using finite differences.
"""
function finite_diff_gradient(f, inputs::Vector{Float64}; ε::Float64=1e-7)
    n = length(inputs)
    grads = zeros(Float64, n)
    f0 = Base.invokelatest(f, inputs...)
    
    for i in 1:n
        inputs_plus = copy(inputs)
        inputs_plus[i] += ε
        grads[i] = (Base.invokelatest(f, inputs_plus...) - f0) / ε
    end
    
    return grads
end

"""
    distribute_gradients(input_grads, n_nodes, seed)

Distribute input gradients to tree nodes based on deterministic pattern.
Uses the gradient magnitudes and tree structure to assign values.
"""
function distribute_gradients(input_grads::Vector{Float64}, n_nodes::Int, seed::UInt64)
    node_grads = zeros(Float64, n_nodes)
    n_inputs = length(input_grads)
    
    # Root gets sum of all gradients
    node_grads[1] = sum(input_grads)
    
    # Distribute to other nodes based on hash
    for i in 2:n_nodes
        h = seed ⊻ UInt64(i * 0x9e3779b97f4a7c15)
        input_idx = (h % n_inputs) + 1
        scale = 0.5 + 0.5 * ((h >> 32) % 1000) / 1000.0
        node_grads[i] = input_grads[input_idx] * scale
    end
    
    return node_grads
end

# ═══════════════════════════════════════════════════════════════════════════
# Lisp bindings for autodiff
# ═══════════════════════════════════════════════════════════════════════════

# (gay-diff f inputs) - compute and visualize gradients
gay_diff(gd::GayDifferentiable, inputs...) = gay_autodiff(gd, collect(Float64, inputs))

# (gay-jacobian f inputs) - compute full Jacobian
function gay_jacobian(gd::GayDifferentiable, inputs::Vector{Float64})
    n = length(inputs)
    J = zeros(Float64, n, n)
    
    for i in 1:n
        tangent = zeros(n)
        tangent[i] = 1.0
        J[:, i] = finite_diff_gradient(x -> gd.f((inputs .+ tangent .* x)...), [0.0])
    end
    
    return J
end

export gay_diff, gay_jacobian, gay_autodiff

# ═══════════════════════════════════════════════════════════════════════════
# Example: Visualizing autodiff on a simple expression
# ═══════════════════════════════════════════════════════════════════════════

"""
    demo_enzyme_colors()

Demonstrate enzyme coloring on a simple expression.
Shows how Forward/Reverse modes give different color streams.
"""
function demo_enzyme_colors()
    println("\n╔══════════════════════════════════════════════════════════════╗")
    println("║  Gay.jl + Enzyme.jl: SICP 4A \"Enzymes Attach to Expressions\" ║")
    println("╚══════════════════════════════════════════════════════════════╝\n")
    
    # Sample expression: (+ (* x x) (* y y))  -- x² + y²
    expr = [:+, [:*, :x, :x], [:*, :y, :y]]
    
    println("Expression: (+ (* x x) (* y y))  ≡  x² + y²\n")
    
    # Original magnetized S-expr
    gs = gay_magnetized_sexpr(expr, 0x42)
    println("Original (magnetized):")
    println("  ", gay_render_sexpr(gs))
    println("  Magnetization ⟨M⟩ = ", round(gay_sexpr_magnetization(gs), digits=3))
    println()
    
    # Forward mode (tangent propagation)
    ge_fwd = enzyme_forward(expr, 0x42)
    n_bindings = length(ge_fwd.bindings)
    # Simulate gradients: ∂(x²+y²)/∂x = 2x, ∂(x²+y²)/∂y = 2y
    # For x=3, y=4: gradients at different nodes
    fwd_grads = [1.0 + 0.5*i for i in 1:n_bindings]  # Increasing gradient magnitudes
    propagate_gradient(ge_fwd, fwd_grads)
    
    println("Forward mode ∂f/∂x (blue stream):")
    println("  ", gay_render_enzyme(ge_fwd))
    println("  Gradient magnetization ⟨M_∇⟩ = ", round(gradient_magnetization(ge_fwd), digits=3))
    println()
    
    # Reverse mode (adjoint propagation)
    ge_rev = enzyme_reverse(expr, 0x42)
    # Reverse gradients (backprop from output)
    rev_grads = [n_bindings - i + 1.0 for i in 1:n_bindings]  # Decreasing
    propagate_gradient(ge_rev, rev_grads)
    
    println("Reverse mode ∂L/∂f (red stream):")
    println("  ", gay_render_enzyme(ge_rev))
    println("  Gradient magnetization ⟨M_∇⟩ = ", round(gradient_magnetization(ge_rev), digits=3))
    println()
    
    # Mixed mode with negative gradients
    ge_mix = enzyme_mixed(expr, 0x42)
    mix_grads = [(-1.0)^i * (0.5 + 0.3*i) for i in 1:n_bindings]  # Alternating signs
    propagate_gradient(ge_mix, mix_grads)
    
    println("Mixed mode (purple, with sign flips):")
    println("  ", gay_render_enzyme(ge_mix))
    println("  Gradient magnetization ⟨M_∇⟩ = ", round(gradient_magnetization(ge_mix), digits=3))
    println()
    
    println("Legend:")
    println("  ⁺→  Forward mode, positive sensitivity")
    println("  ⁻→  Forward mode, negative sensitivity")
    println("  ⁺←  Reverse mode, positive sensitivity")
    println("  ⁻←  Reverse mode, negative sensitivity")
    println("  ⁺⇄  Mixed mode, positive sensitivity")
    println("  ⁻⇄  Mixed mode, negative sensitivity")
    println()
    println("Color intensity indicates |∂f/∂x| magnitude")
end

# Export everything
export demo_enzyme_colors
