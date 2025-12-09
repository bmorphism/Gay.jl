#!/usr/bin/env julia
"""
The Trans-Trinity: Transduction, Transclusion, Transitivity
With 2-Transducers (Loregian 2025) and Dynamic Markov Blanket Detection

Color is everywhere — color as functor from structure to perception.
"""

using Gay

# ═══════════════════════════════════════════════════════════════════════════
# 2-Transducer Structure (after Loregian arXiv:2509.06769)
# ═══════════════════════════════════════════════════════════════════════════

"""
A 2-transducer 1-cell: (Q, t) : A → B
- Q = state category (objects have morphisms!)
- t = profunctor A × Q^op × Q × (B*)^op → Set

For visualization, we color:
- Objects of Q (states) with deterministic colors
- Morphisms of Q (state transitions) with gradient interpolation
- The profunctor t as colored edges between input/output words
"""
struct TwoTransducer
    name::String
    input_alphabet::Vector{Symbol}   # Objects of A
    output_alphabet::Vector{Symbol}  # Objects of B  
    states::Vector{Symbol}           # Objects of Q
    state_colors::Vector{Any}        # Deterministic coloring of Q
    transitions::Dict{Tuple{Symbol,Symbol}, Vector{Tuple{Symbol,Symbol}}}
    # (input, state) → [(output, next_state), ...]
end

"""
Construct a 2-transducer with deterministic Gay.jl coloring.
Each state gets a color from the splittable RNG stream.
"""
function TwoTransducer(name::String, 
                       inputs::Vector{Symbol}, 
                       outputs::Vector{Symbol},
                       states::Vector{Symbol},
                       transitions::Dict;
                       seed::Integer=42)
    gay_seed!(seed)
    state_colors = [next_color(SRGB()) for _ in states]
    TwoTransducer(name, inputs, outputs, states, state_colors, transitions)
end

# ═══════════════════════════════════════════════════════════════════════════
# Markov Blanket as 2-Transducer
# ═══════════════════════════════════════════════════════════════════════════

"""
The DMBD partition (s, b, z) as a 2-transducer:
- Input category A = environment observations (s)
- Output category B = actions on environment  
- State category Q = blanket states (b) with internal dynamics (z)

The profunctor t encodes conditional independence:
    p(s,z|b) = p(s|b) p(z|b)
"""
struct MarkovBlanketTransducer
    environment::Vector{Symbol}  # s - external states
    blanket::Vector{Symbol}      # b - boundary states
    internal::Vector{Symbol}     # z - internal states
    colors::NamedTuple           # Colored partition
end

function MarkovBlanketTransducer(; seed::Integer=1069)
    gay_seed!(seed)
    
    # Example partition from DMBD paper
    env = [:s₁, :s₂, :s₃]
    blanket = [:b₁, :b₂]  
    internal = [:z₁, :z₂, :z₃]
    
    # Color each partition distinctly
    env_colors = [next_color(SRGB()) for _ in env]
    blanket_colors = [next_color(SRGB()) for _ in blanket]
    internal_colors = [next_color(SRGB()) for _ in internal]
    
    colors = (
        environment = Dict(zip(env, env_colors)),
        blanket = Dict(zip(blanket, blanket_colors)),
        internal = Dict(zip(internal, internal_colors))
    )
    
    MarkovBlanketTransducer(env, blanket, internal, colors)
end

# ═══════════════════════════════════════════════════════════════════════════
# The Trans-Trinity Visualization
# ═══════════════════════════════════════════════════════════════════════════

"""
Render the trans-trinity with colored ASCII art.
"""
function render_trans_trinity(; seed::Integer=2025)
    gay_seed!(seed)
    
    # Get three colors for the trinity
    c_transduction = next_color(SRGB())
    c_transclusion = next_color(SRGB())
    c_transitivity = next_color(SRGB())
    
    function ansi(c)
        r = round(Int, c.r * 255)
        g = round(Int, c.g * 255)
        b = round(Int, c.b * 255)
        "\e[38;2;$(r);$(g);$(b)m"
    end
    R = "\e[0m"
    
    t1 = ansi(c_transduction)
    t2 = ansi(c_transclusion)
    t3 = ansi(c_transitivity)
    
    println()
    println("  ╔════════════════════════════════════════════════════════════════════╗")
    println("  ║              $(t1)T$(t2)R$(t3)A$(t1)N$(t2)S$(R)-TRINITY: Color is Everywhere                  ║")
    println("  ╚════════════════════════════════════════════════════════════════════╝")
    println()
    
    # Triangle layout
    println("                        $(t1)▲ TRANSDUCTION$(R)")
    println("                       $(t1)╱ ╲$(R)  signal → representation")
    println("                      $(t1)╱   ╲$(R)   A* → B*")
    println("                     $(t1)╱     ╲$(R)")
    println("                    $(t1)╱   ●   ╲$(R)  2-transducer")
    println("                   $(t1)╱    ↑    ╲$(R)  state category Q")
    println("                  $(t1)╱     │     ╲$(R)")
    println("    $(t2)TRANSCLUSION$(R) $(t2)▼─────┼─────▶$(R) $(t3)TRANSITIVITY$(R)")
    println("    $(t2)embedding$(R)        $(t2)│$(R)        $(t3)composition$(R)")
    println("    $(t2)operadic$(R)         $(t2)│$(R)        $(t3)f ; g$(R)")
    println("    $(t2)substitution$(R)     $(t2)│$(R)        $(t3)cod(f)=dom(g)$(R)")
    println()
    
    # Connection to structures
    println("  ┌─────────────────────────────────────────────────────────────────────┐")
    println("  │ $(t1)Transduction$(R)  → Signal/meaning pipelines (BCI, CETI, dialect)    │")
    println("  │ $(t2)Transclusion$(R)  → self.substitute(box, other) — operadic           │")
    println("  │ $(t3)Transitivity$(R)  → Markov blanket factorization p(s,z|b)=p(s|b)p(z|b)│")
    println("  └─────────────────────────────────────────────────────────────────────┘")
    println()
end

# ═══════════════════════════════════════════════════════════════════════════
# 2-Transducer Visualization
# ═══════════════════════════════════════════════════════════════════════════

"""
Visualize a 2-transducer with colored state transitions.
"""
function render_2transducer(tdx::TwoTransducer)
    R = "\e[0m"
    
    function ansi(c)
        r = round(Int, c.r * 255)
        g = round(Int, c.g * 255)
        b = round(Int, c.b * 255)
        "\e[38;2;$(r);$(g);$(b)m"
    end
    
    println()
    println("  2-Transducer: $(tdx.name)")
    println("  ─────────────────────────────────────")
    
    # Show state category Q with colors
    println("  State Category Q:")
    for (i, (s, c)) in enumerate(zip(tdx.states, tdx.state_colors))
        print("    $(ansi(c))●$(R) $(s)")
        if i < length(tdx.states)
            print("  ")
        end
    end
    println()
    println()
    
    # Show input/output alphabets
    println("  Input A: $(join(string.(tdx.input_alphabet), ", "))")
    println("  Output B: $(join(string.(tdx.output_alphabet), ", "))")
    println()
    
    # Show transitions with color interpolation
    println("  Transitions (colored by state):")
    for ((inp, state), outputs) in tdx.transitions
        state_idx = findfirst(==(state), tdx.states)
        c = tdx.state_colors[state_idx]
        
        for (out, next_state) in outputs
            next_idx = findfirst(==(next_state), tdx.states)
            c_next = tdx.state_colors[next_idx]
            
            print("    $(ansi(c))$(state)$(R) ")
            print("──$(inp)→ ")
            print("$(ansi(c_next))$(next_state)$(R) ")
            println("/ $(out)")
        end
    end
    println()
end

# ═══════════════════════════════════════════════════════════════════════════
# DMBD-style Markov Blanket Visualization  
# ═══════════════════════════════════════════════════════════════════════════

"""
Visualize Markov blanket partition with colors.
Shows (s, b, z) = (environment, blanket, internal)
"""
function render_markov_blanket(mb::MarkovBlanketTransducer)
    R = "\e[0m"
    
    function ansi(c)
        r = round(Int, c.r * 255)
        g = round(Int, c.g * 255)
        b = round(Int, c.b * 255)
        "\e[38;2;$(r);$(g);$(b)m"
    end
    
    println()
    println("  Dynamic Markov Blanket Detection (DMBD)")
    println("  ════════════════════════════════════════")
    println()
    
    # Get representative colors for each partition
    c_env = first(values(mb.colors.environment))
    c_bln = first(values(mb.colors.blanket))
    c_int = first(values(mb.colors.internal))
    
    # ASCII diagram
    println("       $(ansi(c_env))┌─────────────┐$(R)")
    println("       $(ansi(c_env))│ Environment │$(R)  s = external states")
    println("       $(ansi(c_env))│   (s)       │$(R)")
    println("       $(ansi(c_env))└──────┬──────┘$(R)")
    println("              $(ansi(c_env))│$(R)")
    println("              $(ansi(c_env))▼$(R)")
    println("       $(ansi(c_bln))╔═════════════╗$(R)")
    println("       $(ansi(c_bln))║   Blanket   ║$(R)  b = boundary (Markov blanket)")
    println("       $(ansi(c_bln))║     (b)     ║$(R)  conditional independence:")
    println("       $(ansi(c_bln))╚══════╤══════╝$(R)  p(s,z|b) = p(s|b)p(z|b)")
    println("              $(ansi(c_bln))│$(R)")
    println("              $(ansi(c_bln))▼$(R)")
    println("       $(ansi(c_int))┌─────────────┐$(R)")
    println("       $(ansi(c_int))│  Internal   │$(R)  z = internal states")
    println("       $(ansi(c_int))│    (z)      │$(R)")
    println("       $(ansi(c_int))└─────────────┘$(R)")
    println()
    
    # Show colored elements
    println("  Partition elements:")
    print("    Environment: ")
    for (s, c) in mb.colors.environment
        print("$(ansi(c))●$(R)$(s) ")
    end
    println()
    
    print("    Blanket:     ")
    for (b, c) in mb.colors.blanket
        print("$(ansi(c))●$(R)$(b) ")
    end
    println()
    
    print("    Internal:    ")
    for (z, c) in mb.colors.internal
        print("$(ansi(c))●$(R)$(z) ")
    end
    println()
    println()
    
    # The 2-transducer interpretation
    println("  As 2-Transducer (Q, t) : Env → Act:")
    println("    Q = Blanket states (b)")
    println("    t : Env × Q^op × Q × Act* → Set")
    println("    2-cells = blanket dynamics / DMBD updates")
    println()
end

# ═══════════════════════════════════════════════════════════════════════════
# Word-to-World as 2-Transducer
# ═══════════════════════════════════════════════════════════════════════════

"""
Word-to-World Models (Wong et al.) as a 2-transducer:
- Input: Natural language tokens (NL)
- Output: Probabilistic Language of Thought (PLoT)
- States: Contextual meaning states
- Profunctor: Context-sensitive translation (LLM)
"""
function render_word_to_world(; seed::Integer=42)
    gay_seed!(seed)
    
    c_nl = next_color(SRGB())      # Natural language
    c_ctx = next_color(SRGB())     # Context
    c_plot = next_color(SRGB())    # PLoT output
    c_world = next_color(SRGB())   # World model
    
    R = "\e[0m"
    function ansi(c)
        r = round(Int, c.r * 255)
        g = round(Int, c.g * 255)
        b = round(Int, c.b * 255)
        "\e[38;2;$(r);$(g);$(b)m"
    end
    
    nl = ansi(c_nl)
    ctx = ansi(c_ctx)
    plot = ansi(c_plot)
    world = ansi(c_world)
    
    println()
    println("  Word-to-World as 2-Transducer")
    println("  ══════════════════════════════")
    println()
    println("    $(nl)Natural Language$(R)  ───(LLM as 2-transducer)───▶  $(plot)PLoT$(R)")
    println("           $(nl)│$(R)                                           $(plot)│$(R)")
    println("           $(nl)│$(R)   $(ctx)Q = contextual meaning states$(R)         $(plot)│$(R)")
    println("           $(nl)│$(R)   $(ctx)t = context-sensitive translation$(R)     $(plot)│$(R)")
    println("           $(nl)│$(R)                                           $(plot)│$(R)")
    println("           $(nl)▼$(R)                                           $(plot)▼$(R)")
    println("       $(nl)Signal$(R)                                    $(world)World Model$(R)")
    println()
    println("  The meaning function M : NL* → PLoT is a 1-cell in 2TDX")
    println("  LLM fine-tuning = 2-cell (natural transformation between transducers)")
    println()
end

# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

function main(; seed::Integer=2025)
    render_trans_trinity(; seed=seed)
    
    # Example 2-transducer: simple language model
    lm = TwoTransducer(
        "LanguageModel",
        [:word, :punct, :space],
        [:token, :embed],
        [:start, :reading, :emit],
        Dict(
            (:word, :start) => [(:token, :reading)],
            (:word, :reading) => [(:token, :reading), (:embed, :emit)],
            (:punct, :reading) => [(:token, :emit)],
            (:space, :emit) => [(:token, :start)],
        );
        seed=seed
    )
    render_2transducer(lm)
    
    # Markov blanket visualization
    mb = MarkovBlanketTransducer(; seed=seed)
    render_markov_blanket(mb)
    
    # Word-to-World
    render_word_to_world(; seed=seed)
    
    println("  Color is everywhere: deterministic hues trace categorical flow")
    println()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
