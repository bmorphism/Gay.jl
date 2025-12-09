#!/usr/bin/env julia
"""
2-Transducer Colorings (after Loregian arXiv:2509.06769)

Visualizing the bicategory 2TDX with Gay.jl deterministic colors:
- Objects: Categories A, B (colored)
- 1-cells: (Q, t) : A → B where Q is state category, t is profunctor
- 2-cells: Natural transformations between transducers (color morphisms)

Color traces categorical flow — each structural element gets
a deterministic hue from the splittable RNG.
"""

using Gay
using Colors: RGB

# ═══════════════════════════════════════════════════════════════════════════
# 2-Transducer Core Structures
# ═══════════════════════════════════════════════════════════════════════════

"""
A morphism in the state category Q.
States aren't just a set — they have structure!
"""
struct StateMorphism
    source::Symbol
    target::Symbol
    label::String
    color::Any  # Interpolated between source/target colors
end

"""
A 2-transducer 1-cell: (Q, t) : A → B

- Q = state category with objects and morphisms
- t = profunctor A × Q^op × Q × (B*)^op → Set
- We represent t as transition rules with probabilities
"""
struct TwoTransducer2
    name::String
    
    # Input/output categories (simplified to objects only)
    input_objects::Vector{Symbol}
    input_colors::Vector{Any}
    output_objects::Vector{Symbol}
    output_colors::Vector{Any}
    
    # State category Q
    state_objects::Vector{Symbol}
    state_colors::Vector{Any}
    state_morphisms::Vector{StateMorphism}
    
    # Profunctor t as transition rules
    # (input, state) → [(probability, output_word, next_state)]
    transitions::Dict{Tuple{Symbol,Symbol}, Vector{Tuple{Float64, Vector{Symbol}, Symbol}}}
end

"""
Construct a 2-transducer with full color assignment.
"""
function TwoTransducer2(name::String;
                        inputs::Vector{Symbol},
                        outputs::Vector{Symbol},
                        states::Vector{Symbol},
                        state_edges::Vector{Tuple{Symbol,Symbol,String}},
                        transitions::Dict,
                        seed::Integer=42)
    gay_seed!(seed)
    
    # Color input alphabet
    input_colors = [next_color(SRGB()) for _ in inputs]
    
    # Color output alphabet  
    output_colors = [next_color(SRGB()) for _ in outputs]
    
    # Color state objects
    state_colors = [next_color(SRGB()) for _ in states]
    
    # Color state morphisms (interpolate between source/target)
    state_morphisms = StateMorphism[]
    for (src, tgt, label) in state_edges
        src_idx = findfirst(==(src), states)
        tgt_idx = findfirst(==(tgt), states)
        
        # Interpolate colors
        c1 = state_colors[src_idx]
        c2 = state_colors[tgt_idx]
        mid_color = RGB(
            (c1.r + c2.r) / 2,
            (c1.g + c2.g) / 2,
            (c1.b + c2.b) / 2
        )
        
        push!(state_morphisms, StateMorphism(src, tgt, label, mid_color))
    end
    
    TwoTransducer2(name, inputs, input_colors, outputs, output_colors,
                   states, state_colors, state_morphisms, transitions)
end

# ═══════════════════════════════════════════════════════════════════════════
# 2-Cell: Natural Transformation Between Transducers
# ═══════════════════════════════════════════════════════════════════════════

"""
A 2-cell α : (Q, t) ⇒ (Q', t') in 2TDX.
This is a natural transformation between transducers.
"""
struct TwoCell
    name::String
    source::TwoTransducer2
    target::TwoTransducer2
    
    # State functor F : Q → Q'
    state_map::Dict{Symbol, Symbol}
    
    # Color for the 2-cell itself
    color::Any
end

function TwoCell(name::String, src::TwoTransducer2, tgt::TwoTransducer2,
                 state_map::Dict{Symbol,Symbol}; seed::Integer=42)
    gay_seed!(seed + hash(name))
    c = next_color(SRGB())
    TwoCell(name, src, tgt, state_map, c)
end

# ═══════════════════════════════════════════════════════════════════════════
# Rendering Functions
# ═══════════════════════════════════════════════════════════════════════════

function ansi(c)
    r = round(Int, clamp(c.r, 0, 1) * 255)
    g = round(Int, clamp(c.g, 0, 1) * 255)
    b = round(Int, clamp(c.b, 0, 1) * 255)
    "\e[38;2;$(r);$(g);$(b)m"
end

const R = "\e[0m"

"""
Render input/output alphabets with colors.
"""
function render_alphabets(tdx::TwoTransducer2)
    println("  Input Category A:")
    print("    ")
    for (obj, c) in zip(tdx.input_objects, tdx.input_colors)
        print("$(ansi(c))●$(R)$obj ")
    end
    println()
    
    println("  Output Category B:")
    print("    ")
    for (obj, c) in zip(tdx.output_objects, tdx.output_colors)
        print("$(ansi(c))●$(R)$obj ")
    end
    println()
end

"""
Render state category Q with objects and morphisms.
"""
function render_state_category(tdx::TwoTransducer2)
    println("  State Category Q (objects have morphisms!):")
    
    # Objects
    print("    Objects: ")
    for (obj, c) in zip(tdx.state_objects, tdx.state_colors)
        print("$(ansi(c))●$(R)$obj ")
    end
    println()
    
    # Morphisms
    if !isempty(tdx.state_morphisms)
        println("    Morphisms:")
        for m in tdx.state_morphisms
            src_idx = findfirst(==(m.source), tdx.state_objects)
            tgt_idx = findfirst(==(m.target), tdx.state_objects)
            c_src = tdx.state_colors[src_idx]
            c_tgt = tdx.state_colors[tgt_idx]
            
            print("      $(ansi(c_src))$(m.source)$(R)")
            print(" ──$(ansi(m.color))$(m.label)$(R)──▶ ")
            println("$(ansi(c_tgt))$(m.target)$(R)")
        end
    end
end

"""
Render the profunctor t as colored transitions.
"""
function render_profunctor(tdx::TwoTransducer2)
    println("  Profunctor t : A × Q^op × Q × (B*)^op → Set:")
    println("  ┌────────────┬────────────┬────────────┬────────────────────┐")
    println("  │ Input (A)  │ State (Q)  │ Next State │ Output Word (B*)   │")
    println("  ├────────────┼────────────┼────────────┼────────────────────┤")
    
    for ((inp, state), outputs) in tdx.transitions
        inp_idx = findfirst(==(inp), tdx.input_objects)
        state_idx = findfirst(==(state), tdx.state_objects)
        
        c_inp = tdx.input_colors[inp_idx]
        c_state = tdx.state_colors[state_idx]
        
        for (prob, out_word, next_state) in outputs
            next_idx = findfirst(==(next_state), tdx.state_objects)
            c_next = tdx.state_colors[next_idx]
            
            # Color each output symbol
            out_str = ""
            for o in out_word
                o_idx = findfirst(==(o), tdx.output_objects)
                if isnothing(o_idx)
                    # Output symbol might be from input (identity transduction)
                    o_idx_in = findfirst(==(o), tdx.input_objects)
                    if !isnothing(o_idx_in)
                        c_o = tdx.input_colors[o_idx_in]
                    else
                        c_o = RGB(0.5, 0.5, 0.5)  # gray for unknown
                    end
                else
                    c_o = tdx.output_colors[o_idx]
                end
                out_str *= "$(ansi(c_o))$o$(R) "
            end
            if isempty(out_word)
                out_str = "ε"
            end
            
            inp_str = "$(ansi(c_inp))$(rpad(string(inp), 10))$(R)"
            state_str = "$(ansi(c_state))$(rpad(string(state), 10))$(R)"
            next_str = "$(ansi(c_next))$(rpad(string(next_state), 10))$(R)"
            
            println("  │ $inp_str │ $state_str │ $next_str │ $(rpad(out_str, 18)) │")
        end
    end
    println("  └────────────┴────────────┴────────────┴────────────────────┘")
end

"""
Render a 2-cell (natural transformation).
"""
function render_2cell(α::TwoCell)
    println()
    println("  $(ansi(α.color))2-Cell α : $(α.source.name) ⇒ $(α.target.name)$(R)")
    println("  ─────────────────────────────────────────")
    
    println("  State functor F : Q → Q'")
    for (q, q_prime) in α.state_map
        src_idx = findfirst(==(q), α.source.state_objects)
        tgt_idx = findfirst(==(q_prime), α.target.state_objects)
        
        c_src = α.source.state_colors[src_idx]
        c_tgt = α.target.state_colors[tgt_idx]
        
        print("    $(ansi(c_src))$q$(R)")
        print(" $(ansi(α.color))↦$(R) ")
        println("$(ansi(c_tgt))$q_prime$(R)")
    end
end

"""
Full 2-transducer visualization.
"""
function render_2transducer(tdx::TwoTransducer2)
    println()
    println("  ╔════════════════════════════════════════════════════════════════════╗")
    println("  ║  2-Transducer: $(rpad(tdx.name, 52)) ║")
    println("  ║  1-cell (Q, t) : A → B in bicategory 2TDX                          ║")
    println("  ╚════════════════════════════════════════════════════════════════════╝")
    println()
    
    render_alphabets(tdx)
    println()
    render_state_category(tdx)
    println()
    render_profunctor(tdx)
    println()
end

# ═══════════════════════════════════════════════════════════════════════════
# Example: Dialect Transducer with State Morphisms
# ═══════════════════════════════════════════════════════════════════════════

function dialect_2transducer(; seed::Integer=1069)
    TwoTransducer2(
        "Gheg Dialect",
        inputs = [:qysh, :kon, :ilberi, :ç],       # NE features
        outputs = [:si, :kjen, :ylberi, :c],       # NW features
        states = [:Prishtinë, :Prizren, :Has, :Shkodër],
        
        # State morphisms: dialect contact/evolution
        state_edges = [
            (:Prishtinë, :Prizren, "trade"),
            (:Prizren, :Has, "mountain"),
            (:Has, :Shkodër, "adriatic"),
            (:Prizren, :Shkodër, "direct"),
        ],
        
        transitions = Dict(
            (:qysh, :Prishtinë) => [(0.9, [:qysh], :Prishtinë), (0.1, [:si], :Prizren)],
            (:qysh, :Prizren) => [(0.5, [:qysh], :Has), (0.5, [:si], :Shkodër)],
            (:qysh, :Has) => [(0.3, [:qysh], :Has), (0.7, [:si], :Shkodër)],
            (:qysh, :Shkodër) => [(1.0, [:si], :Shkodër)],
            
            (:kon, :Prishtinë) => [(0.95, [:kon], :Prishtinë)],
            (:kon, :Prizren) => [(0.6, [:kon], :Prizren), (0.4, [:kjen], :Has)],
            (:kon, :Shkodër) => [(1.0, [:kjen], :Shkodër)],
            
            (:ilberi, :Prishtinë) => [(1.0, [:ilberi], :Prishtinë)],
            (:ilberi, :Shkodër) => [(1.0, [:ylberi], :Shkodër)],
            
            (:ç, :Prishtinë) => [(1.0, [:ç], :Prizren)],
            (:ç, :Shkodër) => [(1.0, [:c], :Shkodër)],
        ),
        seed = seed
    )
end

# ═══════════════════════════════════════════════════════════════════════════
# Example: Signal Processing Transducer (BCI-style)
# ═══════════════════════════════════════════════════════════════════════════

function neural_2transducer(; seed::Integer=42)
    TwoTransducer2(
        "Neural Signal → Intent",
        inputs = [:spike, :burst, :oscillation, :silence],
        outputs = [:move, :stop, :select, :wait],
        states = [:idle, :detecting, :decoding, :acting],
        
        state_edges = [
            (:idle, :detecting, "onset"),
            (:detecting, :decoding, "threshold"),
            (:decoding, :acting, "confidence"),
            (:acting, :idle, "reset"),
            (:detecting, :idle, "timeout"),
        ],
        
        transitions = Dict(
            (:spike, :idle) => [(0.8, Symbol[], :detecting)],
            (:burst, :idle) => [(1.0, Symbol[], :detecting)],
            (:spike, :detecting) => [(0.6, Symbol[], :decoding), (0.4, Symbol[], :detecting)],
            (:burst, :detecting) => [(0.9, Symbol[], :decoding)],
            (:oscillation, :decoding) => [(0.7, [:move], :acting), (0.3, [:select], :acting)],
            (:silence, :decoding) => [(0.5, [:wait], :idle), (0.5, [:stop], :acting)],
            (:silence, :acting) => [(1.0, Symbol[], :idle)],
            (:silence, :idle) => [(1.0, Symbol[], :idle)],
        ),
        seed = seed
    )
end

# ═══════════════════════════════════════════════════════════════════════════
# Example: Word-to-World Meaning Function
# ═══════════════════════════════════════════════════════════════════════════

function meaning_2transducer(; seed::Integer=2025)
    TwoTransducer2(
        "NL → PLoT (Word-to-World)",
        inputs = [:noun, :verb, :adj, :det, :prep],
        outputs = [:entity, :relation, :property, :quantifier, :constraint],
        states = [:start, :np, :vp, :pp, :complete],
        
        state_edges = [
            (:start, :np, "det/noun"),
            (:np, :vp, "verb"),
            (:vp, :pp, "prep"),
            (:vp, :complete, "period"),
            (:pp, :complete, "close"),
            (:np, :np, "adj"),
        ],
        
        transitions = Dict(
            (:det, :start) => [(1.0, [:quantifier], :np)],
            (:noun, :np) => [(1.0, [:entity], :np)],
            (:adj, :np) => [(1.0, [:property], :np)],
            (:verb, :np) => [(1.0, [:relation], :vp)],
            (:prep, :vp) => [(1.0, [:constraint], :pp)],
            (:noun, :pp) => [(1.0, [:entity], :complete)],
        ),
        seed = seed
    )
end

# ═══════════════════════════════════════════════════════════════════════════
# Composition: Horizontal and Vertical
# ═══════════════════════════════════════════════════════════════════════════

"""
Show how 2-transducers compose (the "transitivity" of the trans-trinity).
"""
function show_composition()
    println()
    println("  ╔════════════════════════════════════════════════════════════════════╗")
    println("  ║  2-Transducer Composition (Transitivity)                           ║")
    println("  ╚════════════════════════════════════════════════════════════════════╝")
    println()
    
    gay_seed!(999)
    c1 = next_color(SRGB())
    c2 = next_color(SRGB())
    c3 = next_color(SRGB())
    c_comp = next_color(SRGB())
    
    println("  Horizontal composition (1-cells):")
    println()
    print("    $(ansi(c1))A$(R)")
    print(" ──$(ansi(c2))(Q₁,t₁)$(R)──▶ ")
    print("$(ansi(c2))B$(R)")
    print(" ──$(ansi(c3))(Q₂,t₂)$(R)──▶ ")
    println("$(ansi(c3))C$(R)")
    println()
    print("    $(ansi(c1))A$(R)")
    print(" ────$(ansi(c_comp))(Q₁×Q₂, t₁;t₂)$(R)────▶ ")
    println("$(ansi(c3))C$(R)")
    println()
    println("    State category: Q₁ × Q₂ (product of state categories)")
    println("    Profunctor: t₁ ; t₂ (profunctor composition)")
    println()
    
    println("  Vertical composition (2-cells):")
    println()
    
    c_alpha = next_color(SRGB())
    c_beta = next_color(SRGB())
    c_alphabeta = next_color(SRGB())
    
    println("         $(ansi(c_alpha))α$(R)")
    println("    (Q,t) $(ansi(c_alpha))⇓$(R) (Q',t')")
    println("         $(ansi(c_beta))β$(R)")
    println("    (Q',t') $(ansi(c_beta))⇓$(R) (Q'',t'')")
    println()
    println("         $(ansi(c_alphabeta))α;β$(R)")
    println("    (Q,t) $(ansi(c_alphabeta))⇓$(R) (Q'',t'')")
    println()
    println("    2-cells compose vertically as natural transformations")
    println()
end

# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

function main()
    # Dialect transducer
    dialect = dialect_2transducer()
    render_2transducer(dialect)
    
    # Neural signal transducer
    neural = neural_2transducer()
    render_2transducer(neural)
    
    # Word-to-World meaning function
    meaning = meaning_2transducer()
    render_2transducer(meaning)
    
    # Show composition
    show_composition()
    
    # Example 2-cell between two versions of dialect transducer
    println()
    println("  ═══════════════════════════════════════════════════════════════════")
    println("  2-Cell Example: Historical sound change")
    println("  ═══════════════════════════════════════════════════════════════════")
    
    dialect_old = dialect_2transducer(seed=1069)
    dialect_new = dialect_2transducer(seed=2025)
    
    α = TwoCell("sound_shift", dialect_old, dialect_new,
                Dict(:Prishtinë => :Prishtinë, 
                     :Prizren => :Has,      # Prizren dialect → Has-like
                     :Has => :Shkodër,      # Has dialect → Shkodër-like  
                     :Shkodër => :Shkodër);
                seed=3000)
    
    render_2cell(α)
    
    println()
    println("  Color is everywhere: deterministic hues trace categorical structure")
    println()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
