#!/usr/bin/env julia
"""
DMBD ↔ 2-Transducer Correspondence

The 2-cell equivariance condition:
    T' ∘ (1_A × F) ⇒ F ∘ T

where F : Q → Q' is dynamics (DMBD) or historical evolution (dialect).

This is the blanket-consistency condition: dynamics push states forward
while preserving (up to morphism) the partition structure that mediates
perception, action, and internal dynamics.

References:
- Beck & Ramstead (2025) "Dynamic Markov Blanket Detection"
- Loregian (2025) "Two-dimensional transducers" arXiv:2509.06769
- Wong et al. "Word to World Models"
"""

using Gay
using Colors: RGB

# ═══════════════════════════════════════════════════════════════════════════
# The Shared Structure: Machine over a Base Category
# ═══════════════════════════════════════════════════════════════════════════

"""
A typed transducer T : A × Q → B × Q over base category Q.
Q is not a bare set — it's a geometric/processual manifold.
"""
struct TypedTransducer
    name::String
    base_name::String  # What Q represents (geography, physiology, syntax)
    
    # Alphabets
    input::Vector{Symbol}
    output::Vector{Symbol}
    input_colors::Vector{Any}
    output_colors::Vector{Any}
    
    # State category Q (objects + morphisms)
    states::Vector{Symbol}
    state_colors::Vector{Any}
    transitions::Vector{Tuple{Symbol, Symbol, String}}  # (src, tgt, label)
    
    # Transduction function T : A × Q → B × Q
    # Represented as Dict{(a,q), Vector{(b,q')}}
    T::Dict{Tuple{Symbol,Symbol}, Vector{Tuple{Symbol,Symbol}}}
end

"""
A 1-cell between state categories: F : Q → Q'
This is dynamics (DMBD) or historical evolution (dialect).
"""
struct StateFunctor
    name::String
    source_states::Vector{Symbol}
    target_states::Vector{Symbol}
    mapping::Dict{Symbol, Symbol}
    color::Any
end

"""
A 2-cell expressing compatibility:
    T' ∘ (1_A × F) ⇒ F ∘ T
    
"Evolve then transduce" ≃ "Transduce then evolve"
"""
struct CompatibilityCell
    name::String
    F::StateFunctor
    T_before::TypedTransducer
    T_after::TypedTransducer
    color::Any
end

# ═══════════════════════════════════════════════════════════════════════════
# Constructors
# ═══════════════════════════════════════════════════════════════════════════

function TypedTransducer(name, base; 
                         input, output, states, transitions, T_func,
                         seed::Integer=42)
    gay_seed!(seed)
    
    input_colors = [next_color(SRGB()) for _ in input]
    output_colors = [next_color(SRGB()) for _ in output]
    state_colors = [next_color(SRGB()) for _ in states]
    
    TypedTransducer(name, base, input, output, input_colors, output_colors,
                    states, state_colors, transitions, T_func)
end

function StateFunctor(name, source, target, mapping; seed::Integer=42)
    gay_seed!(seed)
    c = next_color(SRGB())
    StateFunctor(name, source, target, mapping, c)
end

# ═══════════════════════════════════════════════════════════════════════════
# ANSI Rendering
# ═══════════════════════════════════════════════════════════════════════════

ansi(c) = "\e[38;2;$(round(Int,c.r*255));$(round(Int,c.g*255));$(round(Int,c.b*255))m"
const R = "\e[0m"
const BOLD = "\e[1m"
const DIM = "\e[2m"

# ═══════════════════════════════════════════════════════════════════════════
# The Three Examples as Typed Transducers
# ═══════════════════════════════════════════════════════════════════════════

function gheg_transducer(; seed=1069)
    TypedTransducer(
        "Gheg Dialect", "Geography",
        input = [:qysh, :kon, :ç, :ilberi],
        output = [:si, :kjen, :c, :ylberi],
        states = [:Prishtinë, :Prizren, :Has, :Shkodër],
        transitions = [
            (:Prishtinë, :Prizren, "trade"),
            (:Prizren, :Has, "mountain"),
            (:Has, :Shkodër, "adriatic"),
        ],
        T_func = Dict(
            (:qysh, :Prishtinë) => [(:qysh, :Prishtinë)],
            (:qysh, :Prizren) => [(:qysh, :Has), (:si, :Shkodër)],
            (:qysh, :Has) => [(:si, :Shkodër)],
            (:qysh, :Shkodër) => [(:si, :Shkodër)],
            (:kon, :Prishtinë) => [(:kon, :Prishtinë)],
            (:kon, :Shkodër) => [(:kjen, :Shkodër)],
            (:ç, :Prishtinë) => [(:ç, :Prizren)],
            (:ç, :Shkodër) => [(:c, :Shkodër)],
        ),
        seed = seed
    )
end

function neural_transducer(; seed=42)
    TypedTransducer(
        "BCI Signal", "Physiology",
        input = [:spike, :burst, :silence],
        output = [:move, :stop, :wait],
        states = [:idle, :detecting, :decoding, :acting],
        transitions = [
            (:idle, :detecting, "onset"),
            (:detecting, :decoding, "threshold"),
            (:decoding, :acting, "confidence"),
            (:acting, :idle, "reset"),
        ],
        T_func = Dict(
            (:spike, :idle) => [(:wait, :detecting)],
            (:burst, :detecting) => [(:wait, :decoding)],
            (:silence, :decoding) => [(:stop, :idle)],
            (:spike, :decoding) => [(:move, :acting)],
        ),
        seed = seed
    )
end

function syntax_transducer(; seed=2025)
    TypedTransducer(
        "NL→PLoT", "Syntax",
        input = [:det, :noun, :verb, :prep],
        output = [:∃, :entity, :relation, :constraint],
        states = [:start, :np, :vp, :pp],
        transitions = [
            (:start, :np, "det"),
            (:np, :vp, "verb"),
            (:vp, :pp, "prep"),
            (:np, :np, "noun"),
        ],
        T_func = Dict(
            (:det, :start) => [(:∃, :np)],
            (:noun, :np) => [(:entity, :np)],
            (:verb, :np) => [(:relation, :vp)],
            (:prep, :vp) => [(:constraint, :pp)],
        ),
        seed = seed
    )
end

# ═══════════════════════════════════════════════════════════════════════════
# DMBD Structure
# ═══════════════════════════════════════════════════════════════════════════

"""
DMBD partition (S, B, Z) over state space X.
S = sensory/environment, B = blanket, Z = internal
"""
struct DMBDPartition
    name::String
    X::Vector{Symbol}           # Full state space
    S::Vector{Symbol}           # Sensory states
    B::Vector{Symbol}           # Blanket states
    Z::Vector{Symbol}           # Internal states
    colors::NamedTuple
end

function DMBDPartition(; seed=1069)
    gay_seed!(seed)
    
    X = [:x₁, :x₂, :x₃, :x₄, :x₅, :x₆]
    S = [:x₁, :x₂]  # Environment
    B = [:x₃, :x₄]  # Blanket
    Z = [:x₅, :x₆]  # Internal
    
    colors = (
        S = [next_color(SRGB()) for _ in S],
        B = [next_color(SRGB()) for _ in B],
        Z = [next_color(SRGB()) for _ in Z],
    )
    
    DMBDPartition("System", X, S, B, Z, colors)
end

"""
Dynamics functor D : X → X' (time evolution)
"""
struct DynamicsFunctor
    name::String
    D::Dict{Symbol, Symbol}  # State mapping
    color::Any
end

function DynamicsFunctor(; seed=2025)
    gay_seed!(seed)
    # Example: each state evolves forward
    DynamicsFunctor(
        "τ-evolution",
        Dict(:x₁ => :x₂, :x₂ => :x₃, :x₃ => :x₄, :x₄ => :x₅, :x₅ => :x₆, :x₆ => :x₁),
        next_color(SRGB())
    )
end

# ═══════════════════════════════════════════════════════════════════════════
# The Correspondence Diagram
# ═══════════════════════════════════════════════════════════════════════════

function render_correspondence()
    gay_seed!(3000)
    
    # Colors for the diagram
    c_dialect = next_color(SRGB())
    c_dmbd = next_color(SRGB())
    c_arrow = next_color(SRGB())
    c_2cell = next_color(SRGB())
    
    d = ansi(c_dialect)
    m = ansi(c_dmbd)
    a = ansi(c_arrow)
    t = ansi(c_2cell)
    
    println()
    println("  ╔════════════════════════════════════════════════════════════════════════╗")
    println("  ║  $(BOLD)DMBD ↔ 2-Transducer Correspondence$(R)                                    ║")
    println("  ║  Blanket evolution = State category functor                            ║")
    println("  ╚════════════════════════════════════════════════════════════════════════╝")
    println()
    
    # The parallel structure
    println("  $(d)DIALECT$(R)                              $(m)DMBD$(R)")
    println("  ────────────────────────────────────   ────────────────────────────────────")
    println()
    println("  $(d)Q = dialect geography$(R)                $(m)X = system state space$(R)")
    println("  $(d)(Prishtinë → Prizren → Has → Shkodër)$(R)  $(m)(x₁ → x₂ → x₃ → x₄ → x₅ → x₆)$(R)")
    println()
    println("  $(d)A = NE phonemes$(R)                      $(m)S = sensory states$(R)")
    println("  $(d)B = NW phonemes$(R)                      $(m)Z = internal states$(R)")
    println("  $(d)T : A × Q → B × Q$(R)                    $(m)blanket B mediates$(R)")
    println()
    
    # The functor
    println("  $(a)F : Q → Q'$(R)                           $(a)D : X → X'$(R)")
    println("  $(a)historical sound change$(R)              $(a)temporal dynamics$(R)")
    println()
    
    # The 2-cell
    println("  $(t)┌─────────────────────────────────────────────────────────────────────┐$(R)")
    println("  $(t)│  2-CELL: Equivariance / Blanket-Consistency                        │$(R)")
    println("  $(t)│                                                                     │$(R)")
    println("  $(t)│      T' ∘ (1_A × F)  ⇒  F ∘ T                                       │$(R)")
    println("  $(t)│                                                                     │$(R)")
    println("  $(t)│  \"Evolve then transduce\" ≃ \"Transduce then evolve\"                 │$(R)")
    println("  $(t)│                                                                     │$(R)")
    println("  $(t)│  Dynamics preserve partition structure up to morphism.             │$(R)")
    println("  $(t)└─────────────────────────────────────────────────────────────────────┘$(R)")
    println()
end

function render_commutative_square()
    gay_seed!(4000)
    
    c1 = next_color(SRGB())
    c2 = next_color(SRGB())
    c3 = next_color(SRGB())
    c4 = next_color(SRGB())
    cf = next_color(SRGB())
    ct = next_color(SRGB())
    c2c = next_color(SRGB())
    
    println()
    println("  $(BOLD)The Commutative Square$(R) (up to 2-cell)")
    println("  ═══════════════════════════════════════")
    println()
    println("                 $(ansi(ct))T$(R)")
    println("       $(ansi(c1))A × Q$(R) ─────────────▶ $(ansi(c2))B × Q$(R)")
    println("          │                      │")
    println("          │ $(ansi(cf))1×F$(R)              │ $(ansi(cf))1×F$(R)")
    println("          │                      │")
    println("          │      $(ansi(c2c))⇓ η$(R)           │")
    println("          │                      │")
    println("          ▼                      ▼")
    println("       $(ansi(c3))A × Q'$(R) ────────────▶ $(ansi(c4))B × Q'$(R)")
    println("                 $(ansi(ct))T'$(R)")
    println()
    println("  $(ansi(c2c))η$(R) : T' ∘ (1×F) ⇒ (1×F) ∘ T")
    println()
    println("  When $(ansi(c2c))η$(R) = identity: transduction is $(BOLD)equivariant$(R)")
    println("  When $(ansi(c2c))η$(R) ≠ identity: transduction $(BOLD)adapts$(R) to state evolution")
    println()
end

function render_dialect_dmbd_parallel()
    gay_seed!(5000)
    
    # Gheg dialect
    gheg = gheg_transducer(seed=1069)
    
    # Historical evolution functor
    F = StateFunctor(
        "Sound Shift",
        gheg.states, gheg.states,
        Dict(:Prishtinë => :Prishtinë, :Prizren => :Has, :Has => :Shkodër, :Shkodër => :Shkodër),
        seed = 6000
    )
    
    # DMBD partition
    dmbd = DMBDPartition(seed=7000)
    D = DynamicsFunctor(seed=8000)
    
    println()
    println("  $(BOLD)Concrete Parallel: Gheg ↔ DMBD$(R)")
    println("  ════════════════════════════════════")
    println()
    
    # State functor
    println("  $(ansi(F.color))F : Q → Q' (Historical Sound Change)$(R)")
    println("  ─────────────────────────────────────────")
    for (src, tgt) in F.mapping
        src_idx = findfirst(==(src), gheg.states)
        tgt_idx = findfirst(==(tgt), gheg.states)
        c_src = gheg.state_colors[src_idx]
        c_tgt = gheg.state_colors[tgt_idx]
        
        if src == tgt
            println("    $(ansi(c_src))$src$(R) ↦ $(ansi(c_tgt))$tgt$(R)  $(DIM)(fixed point)$(R)")
        else
            println("    $(ansi(c_src))$src$(R) ↦ $(ansi(c_tgt))$tgt$(R)  $(DIM)(shift westward)$(R)")
        end
    end
    println()
    
    # DMBD dynamics
    println("  $(ansi(D.color))D : X → X' (Temporal Evolution)$(R)")
    println("  ─────────────────────────────────────────")
    for (src, tgt) in D.D
        println("    $src ↦ $tgt")
    end
    println()
    
    # The key insight
    println("  $(BOLD)Key Insight:$(R)")
    println("  ─────────────────────────────────────────")
    println("  • Dialect F maps Prizren → Has (dialect boundary shifts)")
    println("  • DMBD D evolves states (blanket boundary evolves)")
    println("  • Both are $(BOLD)functors on state categories$(R)")
    println("  • 2-cell ensures $(BOLD)transduction respects evolution$(R)")
    println()
end

function render_transduction_trace()
    gay_seed!(9000)
    
    println()
    println("  $(BOLD)Transduction Trace: Following a Word Through Time$(R)")
    println("  ════════════════════════════════════════════════════")
    println()
    
    gheg = gheg_transducer(seed=1069)
    
    # Trace 'qysh' from Prishtinë through historical evolution
    word = :qysh
    states_t0 = [:Prishtinë, :Prizren, :Has, :Shkodër]
    evolution = Dict(:Prishtinë => :Prishtinë, :Prizren => :Has, :Has => :Shkodër, :Shkodër => :Shkodër)
    
    println("  Input word: $(ansi(gheg.input_colors[1]))$word$(R)")
    println()
    println("  ┌────────┬────────────────┬────────────────┬────────────────┐")
    println("  │ Time   │ State (Q)      │ Output (B)     │ Next State     │")
    println("  ├────────┼────────────────┼────────────────┼────────────────┤")
    
    for (t, state) in enumerate(states_t0)
        state_idx = findfirst(==(state), gheg.states)
        c_state = gheg.state_colors[state_idx]
        
        # Get transduction result
        key = (word, state)
        if haskey(gheg.T, key)
            results = gheg.T[key]
            for (out, next) in results
                out_idx = findfirst(==(out), gheg.output)
                if isnothing(out_idx)
                    out_idx = findfirst(==(out), gheg.input)
                    c_out = isnothing(out_idx) ? RGB(0.5,0.5,0.5) : gheg.input_colors[out_idx]
                else
                    c_out = gheg.output_colors[out_idx]
                end
                next_idx = findfirst(==(next), gheg.states)
                c_next = gheg.state_colors[next_idx]
                
                t_str = rpad("t=$t", 6)
                state_str = "$(ansi(c_state))$(rpad(string(state), 14))$(R)"
                out_str = "$(ansi(c_out))$(rpad(string(out), 14))$(R)"
                next_str = "$(ansi(c_next))$(rpad(string(next), 14))$(R)"
                
                println("  │ $t_str │ $state_str │ $out_str │ $next_str │")
            end
        else
            println("  │ t=$t   │ $(ansi(c_state))$(rpad(string(state), 14))$(R) │ (no rule)      │                │")
        end
    end
    println("  └────────┴────────────────┴────────────────┴────────────────┘")
    println()
    
    println("  Observation: 'qysh' → 'si' as we move NE → NW")
    println("  This is $(BOLD)transduction equivariant with geographic evolution$(R)")
    println()
end

# ═══════════════════════════════════════════════════════════════════════════
# Bicategory Structure
# ═══════════════════════════════════════════════════════════════════════════

function render_bicategory()
    gay_seed!(10000)
    
    c0 = next_color(SRGB())
    c1 = next_color(SRGB())
    c2 = next_color(SRGB())
    
    println()
    println("  $(BOLD)Bicategory 2TDX$(R)")
    println("  ═══════════════")
    println()
    println("  $(ansi(c0))0-cells$(R): Categories A, B, C, ...")
    println("           (alphabets, type systems, ontologies)")
    println()
    println("  $(ansi(c1))1-cells$(R): Typed transducers (Q, T) : A → B")
    println("           T : A × Q → B × Q")
    println("           Q is a $(BOLD)category$(R) (not just a set!)")
    println()
    println("  $(ansi(c2))2-cells$(R): Natural transformations η : (Q,T) ⇒ (Q',T')")
    println("           Encode: state functors F : Q → Q'")
    println("           Constrain: equivariance T' ∘ (1×F) ⇒ (1×F) ∘ T")
    println()
    println("  $(DIM)Composition:$(R)")
    println("    Horizontal: (Q₁,T₁) ; (Q₂,T₂) = (Q₁×Q₂, T₁;T₂)")
    println("    Vertical:   η ; η' = η' ∘ η (natural transformation composition)")
    println()
end

function render_double_category()
    gay_seed!(11000)
    
    ct = next_color(SRGB())  # tight
    cl = next_color(SRGB())  # loose
    
    println()
    println("  $(BOLD)Double Category 𝔻TDX$(R)")
    println("  ═════════════════════")
    println()
    println("  $(ansi(ct))Tight morphisms$(R) (vertical): Functors between categories")
    println("  $(ansi(cl))Loose morphisms$(R) (horizontal): Transducers (Q, T)")
    println()
    println("            $(ansi(ct))F$(R)")
    println("       A ═══════▶ A'")
    println("       ║         ║")
    println("  $(ansi(cl))(Q,T)$(R) ║    □    ║ $(ansi(cl))(Q',T')$(R)")
    println("       ║         ║")
    println("       ▼         ▼")
    println("       B ═══════▶ B'")
    println("            $(ansi(ct))G$(R)")
    println()
    println("  The square □ is filled by a $(BOLD)2-cell$(R)")
    println("  expressing compatibility of transductions")
    println("  across category morphisms.")
    println()
end

# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

function main()
    render_correspondence()
    render_commutative_square()
    render_dialect_dmbd_parallel()
    render_transduction_trace()
    render_bicategory()
    render_double_category()
    
    gay_seed!(12000)
    c_final = next_color(SRGB())
    
    println()
    println("  $(ansi(c_final))════════════════════════════════════════════════════════════════════$(R)")
    println("  $(ansi(c_final))Color is everywhere: deterministic hues trace categorical structure$(R)")
    println("  $(ansi(c_final))════════════════════════════════════════════════════════════════════$(R)")
    println()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
