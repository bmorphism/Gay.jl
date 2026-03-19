# plastic_lambda.jl - Plastic constant & Lambda Ruliology for Gay.jl
# 
# Adds:
# - ρ (plastic constant) for ternary/GF(3) structures
# - De Bruijn λ-calculus with colored β-reduction traces
#
# Based on: Wolfram "The Ruliology of Lambdas" (2025)
# Thread: https://ampcode.com/threads/T-019bd513-571a-728e-a9ce-654d757e9486

export PLASTIC_RATIO, PLASTIC_ANGLE
export plastic_thread, plastic_color_at
export Lambda, LVar, LAbs, LApp
export lambda_compact, lambda_beta_step, lambda_size, lambda_trace

# ═══════════════════════════════════════════════════════════════════════════
# Plastic Constant (ρ): Ternary analog of Golden Ratio (φ)
# ═══════════════════════════════════════════════════════════════════════════

"""
    PLASTIC_RATIO ≈ 1.324718

The plastic constant ρ, the unique real root of x³ = x + 1.

While φ (golden ratio) governs binary/Fibonacci structures,
ρ governs ternary/Padovan structures. Native to GF(3).

See: Van der Laan's architectonic proportions.
"""
const PLASTIC_RATIO = let x = 1.3
    for _ in 1:20
        x = x - (x^3 - x - 1) / (3*x^2 - 1)
    end
    x  # ≈ 1.324717957244746
end

"""
    PLASTIC_ANGLE ≈ 205.14°

The plastic angle = 360°/ρ², optimal for 3D/ternary packing.
Compare to GOLDEN_ANGLE ≈ 137.51° for 2D/binary.
"""
const PLASTIC_ANGLE = 360.0 / (PLASTIC_RATIO^2)

"""
    plastic_color_at(index; seed=1069, saturation=0.7, lightness=0.55)

Generate color at `index` using plastic angle (205.14° per step).
Returns (hex, hue, trit) where trit ∈ {-1, 0, +1} based on hue sector.
"""
function plastic_color_at(index::Int; seed::Int=1069, saturation::Float64=0.7, lightness::Float64=0.55)
    hue = mod(index * PLASTIC_ANGLE, 360.0)
    
    # HSL to RGB (simplified)
    h = hue / 60.0
    c = (1 - abs(2*lightness - 1)) * saturation
    x = c * (1 - abs(mod(h, 2) - 1))
    m = lightness - c/2
    
    r, g, b = if h < 1
        (c, x, 0.0)
    elseif h < 2
        (x, c, 0.0)
    elseif h < 3
        (0.0, c, x)
    elseif h < 4
        (0.0, x, c)
    elseif h < 5
        (x, 0.0, c)
    else
        (c, 0.0, x)
    end
    
    hex = @sprintf("#%02X%02X%02X", 
        round(Int, (r + m) * 255),
        round(Int, (g + m) * 255),
        round(Int, (b + m) * 255))
    
    # Map hue sector to GF(3) trit
    trit = hue < 120 ? 1 : hue < 240 ? 0 : -1
    
    (hex=hex, hue=hue, trit=trit)
end

"""
    plastic_thread(n; kwargs...)

Generate n colors along the plastic spiral (ρ-based, ternary).
Each step advances by PLASTIC_ANGLE ≈ 205.14°.
"""
function plastic_thread(n::Int; start_hue::Float64=0.0, saturation::Float64=0.7, lightness::Float64=0.55)
    [(step=i, plastic_color_at(i; saturation=saturation, lightness=lightness)...) for i in 1:n]
end

# ═══════════════════════════════════════════════════════════════════════════
# De Bruijn Lambda Calculus
# ═══════════════════════════════════════════════════════════════════════════

"""
Abstract type for de Bruijn indexed lambda terms.
"""
abstract type Lambda end

"""
    LVar(index)

De Bruijn variable. `index` counts lambdas back to the binder.
"""
struct LVar <: Lambda
    index::Int
end

"""
    LAbs(body)

Lambda abstraction. `body` is the lambda's body with de Bruijn indices.
"""
struct LAbs <: Lambda
    body::Lambda
end

"""
    LApp(func, arg)

Application. `func` applied to `arg`.
"""
struct LApp <: Lambda
    func::Lambda
    arg::Lambda
end

"""
    lambda_compact(l::Lambda)

Compact string representation: λλ(2 1) style.
"""
lambda_compact(v::LVar) = string(v.index)
lambda_compact(a::LAbs) = "λ" * lambda_compact(a.body)
lambda_compact(app::LApp) = "(" * lambda_compact(app.func) * " " * lambda_compact(app.arg) * ")"

"""
    lambda_size(l::Lambda)

Count of leaves (vars + abstractions) in the term.
"""
lambda_size(v::LVar) = 1
lambda_size(a::LAbs) = 1 + lambda_size(a.body)
lambda_size(app::LApp) = lambda_size(app.func) + lambda_size(app.arg)

# Shifting for substitution
function lambda_shift(l::Lambda, d::Int, c::Int=1)::Lambda
    if l isa LVar
        l.index >= c ? LVar(l.index + d) : l
    elseif l isa LAbs
        LAbs(lambda_shift(l.body, d, c + 1))
    else
        LApp(lambda_shift(l.func, d, c), lambda_shift(l.arg, d, c))
    end
end

# Substitution
function lambda_subst(l::Lambda, j::Int, s::Lambda)::Lambda
    if l isa LVar
        l.index == j ? s : (l.index > j ? LVar(l.index - 1) : l)
    elseif l isa LAbs
        LAbs(lambda_subst(l.body, j + 1, lambda_shift(s, 1)))
    else
        LApp(lambda_subst(l.func, j, s), lambda_subst(l.arg, j, s))
    end
end

"""
    lambda_beta_step(l::Lambda)

Single leftmost-outermost β-reduction step.
Returns `nothing` if term is in normal form.
"""
function lambda_beta_step(l::Lambda)::Union{Lambda, Nothing}
    if l isa LApp && l.func isa LAbs
        # Redex: (λ.M) N → M[1 := N]
        return lambda_subst(l.func.body, 1, l.arg)
    elseif l isa LApp
        # Try func first (leftmost)
        rf = lambda_beta_step(l.func)
        rf !== nothing && return LApp(rf, l.arg)
        # Then arg
        ra = lambda_beta_step(l.arg)
        ra !== nothing && return LApp(l.func, ra)
    elseif l isa LAbs
        rb = lambda_beta_step(l.body)
        rb !== nothing && return LAbs(rb)
    end
    nothing
end

"""
    lambda_trace(expr; mode=:golden, fuel=100)

Trace β-reduction with golden or plastic thread coloring.

Returns vector of (step, hex, hue, trit, expr, size) tuples.
Detects loops (quines) and fuel exhaustion.

# Arguments
- `expr::Lambda`: Starting lambda term
- `mode::Symbol`: `:golden` (137.5°) or `:plastic` (205.1°)
- `fuel::Int`: Maximum reduction steps

# Example
```julia
Ω = LApp(LAbs(LApp(LVar(1), LVar(1))), LAbs(LApp(LVar(1), LVar(1))))
lambda_trace(Ω; fuel=10)  # Detects loop at step 2
```
"""
function lambda_trace(expr::Lambda; mode::Symbol=:golden, fuel::Int=100)
    angle = mode == :plastic ? PLASTIC_ANGLE : 137.50776405003785
    
    traces = NamedTuple{(:step, :hex, :hue, :trit, :expr, :size, :behavior), 
                        Tuple{Int, String, Float64, Int, String, Int, Symbol}}[]
    seen = Dict{String, Int}()
    current = expr
    
    for step in 1:fuel
        cf = lambda_compact(current)
        hue = mod(step * angle, 360.0)
        trit = hue < 120 ? 1 : hue < 240 ? 0 : -1
        
        # Simple HSL→hex
        h = hue / 60.0
        c = 0.7 * (1 - abs(2*0.55 - 1))
        x = c * (1 - abs(mod(h, 2) - 1))
        m = 0.55 - c/2
        r, g, b = h < 1 ? (c,x,0.) : h < 2 ? (x,c,0.) : h < 3 ? (0.,c,x) : h < 4 ? (0.,x,c) : h < 5 ? (x,0.,c) : (c,0.,x)
        hex = @sprintf("#%02X%02X%02X", round(Int,(r+m)*255), round(Int,(g+m)*255), round(Int,(b+m)*255))
        
        behavior = :running
        if haskey(seen, cf)
            behavior = :looping
            push!(traces, (step=step, hex=hex, hue=hue, trit=trit, expr=cf, size=lambda_size(current), behavior=behavior))
            break
        end
        seen[cf] = step
        
        next = lambda_beta_step(current)
        if next === nothing
            behavior = :normal_form
            push!(traces, (step=step, hex=hex, hue=hue, trit=trit, expr=cf, size=lambda_size(current), behavior=behavior))
            break
        end
        
        push!(traces, (step=step, hex=hex, hue=hue, trit=trit, expr=cf, size=lambda_size(current), behavior=behavior))
        current = next
    end
    
    traces
end

# Standard combinators
const λI = LAbs(LVar(1))  # I = λx.x
const λK = LAbs(LAbs(LVar(2)))  # K = λxy.x
const λS = LAbs(LAbs(LAbs(LApp(LApp(LVar(3), LVar(1)), LApp(LVar(2), LVar(1))))))  # S = λxyz.xz(yz)
const λω = LAbs(LApp(LVar(1), LVar(1)))  # ω = λx.xx
const λΩ = LApp(λω, λω)  # Ω = ωω (non-terminating)

using Printf

end # implicit module boundary - this is included into Gay module
