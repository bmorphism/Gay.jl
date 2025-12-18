# ═══════════════════════════════════════════════════════════════════════════════
# POLARIZED SPLITTABLE RANDOMNESS
# ═══════════════════════════════════════════════════════════════════════════════
#
# Implements the Downen-Ariola polarized compilation for splittable RNGs.
#
# Key insight: The call-by-value/call-by-name duality corresponds to
# eager/lazy color generation in splittable streams.
#
# References:
# - Downen & Ariola "Compiling with Classical Connectives" (LMCS 2020)
# - Munch-Maccagnoni "Syntax and Models of a non-Associative Composition" (2014)
# - Curien-Herbelin "The Duality of Computation" (ICFP 2000)
# ═══════════════════════════════════════════════════════════════════════════════

module PolarizedSplit

using SplittableRandoms: SplittableRandom, split

export PosType, NegType, ShiftDown, ShiftUp
export Command, Term, CoTerm
export MuBinder, MuTildeBinder
export compile_cbv, compile_cbn
export PolarizedRNG, shift_down!, shift_up!
export gay_mu, gay_mu_tilde

# ═══════════════════════════════════════════════════════════════════════════════
# POLARITY STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════

"""
Polarity: Positive types are "values", Negative types are "computations"

In splittable RNG terms:
- Positive (A⁺): A concrete color value
- Negative (A⁻): A color stream/computation
"""
abstract type Polarity end
struct Positive <: Polarity end
struct Negative <: Polarity end

"""
    PosType{T}

Positive type: represents a value (color).
Corresponds to A⁺ in the compilation.
"""
struct PosType{T}
    value::T
end

"""
    NegType{T}

Negative type: represents a computation (stream).
Corresponds to A⁻ in the compilation.
"""
struct NegType{T}
    compute::Function  # () -> T
end

"""
    ShiftDown{T}

↓A: Shift from negative to positive.
"Thunk" a computation into a value.

In RNG terms: split() - create a child stream as a value.
"""
struct ShiftDown{T}
    thunk::NegType{T}
end

"""
    ShiftUp{T}

↑A: Shift from positive to negative.
"Force" a value into a computation.

In RNG terms: observe/generate - collapse stream to color value.
"""
struct ShiftUp{T}
    force::PosType{T}
end

# ═══════════════════════════════════════════════════════════════════════════════
# COMMANDS AND TERMS (λ̄μμ̃ CALCULUS)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    Command

A command ⟨v ‖ e⟩ pairs a term with a co-term.
In RNG terms: "generate color v and pass to continuation e"
"""
struct Command{V,E}
    term::V      # v (producer)
    coterm::E    # e (consumer)
end

"""
    MuBinder

μα.c: Control binder - captures the continuation.
In RNG terms: "bind the result destination"
"""
struct MuBinder{C}
    var::Symbol
    body::C
end

"""
    MuTildeBinder

μ̃x.c: Co-control binder - receives a value.
In RNG terms: "bind the generated color"
"""
struct MuTildeBinder{C}
    var::Symbol
    body::C
end

"""
    Lambda

λ[x·α].c: Abstraction with explicit continuation.
In RNG terms: "function that generates colors"
"""
struct Lambda{C}
    arg::Symbol
    cont::Symbol
    body::C
end

"""
    Application

v · e: Application of value to co-term.
In RNG terms: "apply color to continuation"
"""
struct Application{V,E}
    value::V
    coterm::E
end

"""
    Injection

ιᵢv: Injection into sum type.
In RNG terms: "tag color with variant"
"""
struct Injection{V}
    index::Int
    value::V
end

"""
    CaseAnalysis

⟨{ιᵢxᵢ.cᵢ}: Case analysis on sum.
In RNG terms: "branch on color variant"
"""
struct CaseAnalysis{C}
    cases::Vector{Tuple{Int, Symbol, C}}
end

# ═══════════════════════════════════════════════════════════════════════════════
# CALL-BY-VALUE COMPILATION (CBV)
# ═══════════════════════════════════════════════════════════════════════════════

"""
Call-by-value polarizing compilation: (-)⁺

X⁺ ≜ X
(A → B)⁺ ≜ ↓(A⁺ → (↑B⁺))
(A ⊕ B)⁺ ≜ A⁺ ⊕ B⁺

Key insight: Functions are NEGATIVE (computations), so we shift down.
Arguments are evaluated BEFORE the call.
"""
function compile_cbv(expr, ::Type{Val{:type}})
    # Type compilation
    if expr isa Symbol
        # X⁺ ≜ X (base type unchanged)
        return PosType{expr}
    elseif expr isa Pair  # A → B
        A, B = expr
        # (A → B)⁺ ≜ ↓(A⁺ → (↑B⁺))
        A_pos = compile_cbv(A, Val{:type})
        B_pos = compile_cbv(B, Val{:type})
        return ShiftDown(NegType(() -> (A_pos, ShiftUp(B_pos))))
    end
end

"""
CBV command compilation: ⟨v ‖ e⟩⁺ ≜ ⟨v⁺ |+| e⁺⟩

The |+| indicates positive (value) focus.
"""
function compile_cbv(cmd::Command)
    v_compiled = compile_cbv(cmd.term, Val{:term})
    e_compiled = compile_cbv(cmd.coterm, Val{:coterm})
    Command(v_compiled, e_compiled)
end

"""
CBV term compilation.

(μα.c)⁺ ≜ μα.(c⁺)
(λ[x·α].c)⁺ ≜ ↓(λ[x·β].⟨λ↑α.c⁺ |-| β⟩)
(ιᵢV)⁺ ≜ ιᵢ(V⁺)
"""
function compile_cbv(term, ::Type{Val{:term}})
    if term isa MuBinder
        # (μα.c)⁺ ≜ μα.(c⁺)
        return MuBinder(term.var, compile_cbv(term.body))
    elseif term isa Lambda
        # (λ[x·α].c)⁺ ≜ ↓(λ[x·β].⟨λ↑α.c⁺ |-| β⟩)
        β = gensym(:β)
        c_compiled = compile_cbv(term.body)
        inner = Command(
            Lambda(term.arg, β, 
                   Command(ShiftUp(MuBinder(term.cont, c_compiled)), β)),
            β
        )
        return ShiftDown(Lambda(term.arg, β, inner))
    elseif term isa Injection
        # (ιᵢV)⁺ ≜ ιᵢ(V⁺)
        return Injection(term.index, compile_cbv(term.value, Val{:term}))
    elseif term isa Symbol
        # x⁺ ≜ x
        return term
    end
    term
end

"""
CBV co-term compilation.

(μ̃x.c)⁺ ≜ μ̃x.(c⁺)
(V·e)⁺ ≜ ⟨↓x.⟨x |-| V⁺·[↑e⁺]⟩
⟨{ιᵢxᵢ.cᵢ}⁺ ≜ ⟨{ιᵢxᵢ.cᵢ⁺}
"""
function compile_cbv(coterm, ::Type{Val{:coterm}})
    if coterm isa MuTildeBinder
        # (μ̃x.c)⁺ ≜ μ̃x.(c⁺)
        return MuTildeBinder(coterm.var, compile_cbv(coterm.body))
    elseif coterm isa Application
        # (V·e)⁺ ≜ ⟨↓x.⟨x |-| V⁺·[↑e⁺]⟩
        x = gensym(:x)
        V_compiled = compile_cbv(coterm.value, Val{:term})
        e_compiled = compile_cbv(coterm.coterm, Val{:coterm})
        return ShiftDown(MuTildeBinder(x, 
            Command(x, Application(V_compiled, ShiftUp(e_compiled)))))
    elseif coterm isa CaseAnalysis
        # ⟨{ιᵢxᵢ.cᵢ}⁺ ≜ ⟨{ιᵢxᵢ.cᵢ⁺}
        compiled_cases = [(i, x, compile_cbv(c)) for (i, x, c) in coterm.cases]
        return CaseAnalysis(compiled_cases)
    elseif coterm isa Symbol
        # α⁺ ≜ α
        return coterm
    end
    coterm
end

# ═══════════════════════════════════════════════════════════════════════════════
# CALL-BY-NAME COMPILATION (CBN)
# ═══════════════════════════════════════════════════════════════════════════════

"""
Call-by-name polarizing compilation: (-)⁻

X⁻ ≜ X̄
(A → B)⁻ ≜ (↓A⁻) → B⁻
(A ⊕ B)⁻ ≜ ↑((↓A⁻) ⊕ (↓B⁻))

Key insight: Arguments are THUNKED (shifted down) before the call.
"""
function compile_cbn(expr, ::Type{Val{:type}})
    if expr isa Symbol
        # X⁻ ≜ X̄ (negated base type)
        return NegType{expr}
    elseif expr isa Pair  # A → B
        A, B = expr
        # (A → B)⁻ ≜ (↓A⁻) → B⁻
        A_neg = compile_cbn(A, Val{:type})
        B_neg = compile_cbn(B, Val{:type})
        return NegType(() -> (ShiftDown(A_neg), B_neg))
    end
end

"""
CBN command compilation: ⟨v ‖ e⟩⁻ ≜ ⟨v⁻ |-| e⁻⟩

The |-| indicates negative (computation) focus.
"""
function compile_cbn(cmd::Command)
    v_compiled = compile_cbn(cmd.term, Val{:term})
    e_compiled = compile_cbn(cmd.coterm, Val{:coterm})
    Command(v_compiled, e_compiled)
end

"""
CBN term compilation.

(μα.c)⁻ ≜ μα.(c⁻)
(λ[x·α].c)⁻ ≜ λ[y·α].⟨y |+| ⟨↓x.c⁻⟩
(ιᵢv)⁻ ≜ λ↑β.⟨ιᵢ(↓v⁻) |+| β⟩
"""
function compile_cbn(term, ::Type{Val{:term}})
    if term isa MuBinder
        # (μα.c)⁻ ≜ μα.(c⁻)
        return MuBinder(term.var, compile_cbn(term.body))
    elseif term isa Lambda
        # (λ[x·α].c)⁻ ≜ λ[y·α].⟨y |+| ⟨↓x.c⁻⟩
        y = gensym(:y)
        c_compiled = compile_cbn(term.body)
        return Lambda(y, term.cont,
            Command(y, ShiftDown(MuTildeBinder(term.arg, c_compiled))))
    elseif term isa Injection
        # (ιᵢv)⁻ ≜ λ↑β.⟨ιᵢ(↓v⁻) |+| β⟩
        β = gensym(:β)
        v_compiled = compile_cbn(term.value, Val{:term})
        return Lambda(:_, β,
            ShiftUp(Command(Injection(term.index, ShiftDown(v_compiled)), β)))
    elseif term isa Symbol
        # x⁻ ≜ x
        return term
    end
    term
end

"""
CBN co-term compilation.

(μ̃x.c)⁻ ≜ μ̃x.(c⁻)
(v·E)⁻ ≜ (↓v⁻)·E⁻
⟨{ιᵢxᵢ.cᵢ}⁻ ≜ ↑[⟨{ιᵢyᵢ.⟨yᵢ |+| ⟨↓xᵢ.cᵢ⁻⟩}]
"""
function compile_cbn(coterm, ::Type{Val{:coterm}})
    if coterm isa MuTildeBinder
        # (μ̃x.c)⁻ ≜ μ̃x.(c⁻)
        return MuTildeBinder(coterm.var, compile_cbn(coterm.body))
    elseif coterm isa Application
        # (v·E)⁻ ≜ (↓v⁻)·E⁻
        v_compiled = compile_cbn(coterm.value, Val{:term})
        E_compiled = compile_cbn(coterm.coterm, Val{:coterm})
        return Application(ShiftDown(v_compiled), E_compiled)
    elseif coterm isa CaseAnalysis
        # ⟨{ιᵢxᵢ.cᵢ}⁻ ≜ ↑[⟨{ιᵢyᵢ.⟨yᵢ |+| ⟨↓xᵢ.cᵢ⁻⟩}]
        compiled_cases = map(coterm.cases) do (i, x, c)
            y = gensym(:y)
            c_compiled = compile_cbn(c)
            (i, y, Command(y, ShiftDown(MuTildeBinder(x, c_compiled))))
        end
        return ShiftUp(CaseAnalysis(compiled_cases))
    elseif coterm isa Symbol
        # α⁻ ≜ α
        return coterm
    end
    coterm
end

# ═══════════════════════════════════════════════════════════════════════════════
# POLARIZED RNG: SPLITTABLE RANDOMNESS WITH POLARITY
# ═══════════════════════════════════════════════════════════════════════════════

"""
    PolarizedRNG

A splittable RNG with explicit polarity tracking.

- Positive state: We have a concrete color value
- Negative state: We have a color stream (thunked)

The ↓ and ↑ shifts correspond to split() and force().
"""
mutable struct PolarizedRNG
    rng::SplittableRandom
    polarity::Polarity
    # Value cache for positive state
    cached_value::Union{Nothing, Tuple{Float64, Float64, Float64}}
    # Depth in split tree (for Bumpus adhesion tracking)
    depth::Int
end

PolarizedRNG(seed::UInt64) = PolarizedRNG(SplittableRandom(seed), Negative(), nothing, 0)

"""
    shift_down!(prng) -> PolarizedRNG

↓: Negative → Positive shift.
Creates a THUNK (child stream) that can be passed as a value.

In RNG terms: split() without generating.
"""
function shift_down!(prng::PolarizedRNG)::PolarizedRNG
    @assert prng.polarity isa Negative "Can only shift down from Negative"
    
    child_rng = split(prng.rng)
    PolarizedRNG(child_rng, Positive(), nothing, prng.depth + 1)
end

"""
    shift_up!(prng) -> Tuple{Float64, Float64, Float64}

↑: Positive → Negative shift.
FORCES the thunk, generating a concrete color.

In RNG terms: Actually generate the color from the stream.
"""
function shift_up!(prng::PolarizedRNG)::Tuple{Float64, Float64, Float64}
    @assert prng.polarity isa Positive "Can only shift up from Positive"
    
    if prng.cached_value !== nothing
        return prng.cached_value
    end
    
    # Generate color (Okhsl space)
    h = 360.0 * Float64(rand(prng.rng, UInt32)) / typemax(UInt32)
    s = 0.5 + 0.4 * Float64(rand(prng.rng, UInt32)) / typemax(UInt32)
    l = 0.35 + 0.4 * Float64(rand(prng.rng, UInt32)) / typemax(UInt32)
    
    prng.cached_value = (h, s, l)
    prng.polarity = Negative()  # Now it's a computation again
    
    (h, s, l)
end

"""
    gay_mu(α, body)

μα.c: Control binder for color generation.
"Generate a color and bind the continuation to α"
"""
function gay_mu(prng::PolarizedRNG, body::Function)
    # The body receives the "return address" (continuation)
    # In CBV: we generate the color, then continue
    # In CBN: we suspend and continue immediately
    
    continuation = shift_down!(prng)
    body(continuation)
end

"""
    gay_mu_tilde(x, body)

μ̃x.c: Co-control binder for color consumption.
"Receive a color and bind it to x"
"""
function gay_mu_tilde(prng::PolarizedRNG, body::Function)
    # Force the color generation
    color = shift_up!(prng)
    body(color)
end

# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE: CBV vs CBN COLOR GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

"""
    demo_cbv_color(seed)

Call-by-value: Generate colors eagerly.
Each split immediately produces a value.
"""
function demo_cbv_color(seed::UInt64)
    prng = PolarizedRNG(seed)
    
    # CBV: ⟨λ[x·α].⟨x ‖ α⟩ ‖ V·e⟩
    # Evaluates V first, then applies
    
    colors = Tuple{Float64,Float64,Float64}[]
    
    for i in 1:5
        child = shift_down!(prng)  # ↓: create thunk (split)
        color = shift_up!(child)    # ↑: force (generate)
        push!(colors, color)
    end
    
    colors
end

"""
    demo_cbn_color(seed)

Call-by-name: Generate colors lazily.
Splits are thunked until actually needed.
"""
function demo_cbn_color(seed::UInt64)
    prng = PolarizedRNG(seed)
    
    # CBN: ⟨v ‖ λ[x·α].⟨x ‖ α⟩·e⟩
    # Passes v as thunk, evaluates only when needed
    
    thunks = PolarizedRNG[]
    
    # Create thunks (don't force yet)
    for i in 1:5
        thunk = shift_down!(prng)  # ↓: create thunk
        push!(thunks, thunk)
    end
    
    # Force in reverse order (CBN allows this)
    colors = Tuple{Float64,Float64,Float64}[]
    for thunk in reverse(thunks)
        color = shift_up!(thunk)  # ↑: force now
        push!(colors, color)
    end
    
    colors
end

"""
    verify_polarity_spi(seed)

Verify that CBV and CBN produce different colors (as expected)
but are BOTH deterministic from the same seed.
"""
function verify_polarity_spi(seed::UInt64)
    cbv1 = demo_cbv_color(seed)
    cbv2 = demo_cbv_color(seed)
    
    cbn1 = demo_cbn_color(seed)
    cbn2 = demo_cbn_color(seed)
    
    # CBV is reproducible
    cbv_spi = cbv1 == cbv2
    
    # CBN is reproducible  
    cbn_spi = cbn1 == cbn2
    
    # But CBV ≠ CBN (different evaluation order)
    cbv_cbn_differ = cbv1 != cbn1
    
    println("CBV SPI: ", cbv_spi ? "◆ PASS" : "◇ FAIL")
    println("CBN SPI: ", cbn_spi ? "◆ PASS" : "◇ FAIL")
    println("CBV ≠ CBN: ", cbv_cbn_differ ? "◆ (expected)" : "◇ (unexpected)")
    
    (cbv_spi=cbv_spi, cbn_spi=cbn_spi, cbv_cbn_differ=cbv_cbn_differ)
end

end # module PolarizedSplit
