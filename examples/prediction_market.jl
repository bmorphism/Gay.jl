#!/usr/bin/env julia
"""
Prediction Markets as 2-Transducers

A prediction market is a compositional system:
    Input A  = events, news, trades
    State Q  = collective belief (probability distribution)
    Output B = prices P(event) ∈ [0,1]

The market maker is a Markov blanket:
    External (s) = traders, news sources, world events
    Blanket (b)  = order book, bid/ask spread
    Internal (z) = clearing price, market state

Tracing (from the screenshot):
    - Nested markets = hierarchical event dependencies
    - Composition = combining correlated predictions
    - Flattening = arbitrage across related markets

Perpetual markets follow Fokker-Planck:
    dP/dt = -∇·(drift × P) + D∇²P
    
    drift = information flow (news, trades)
    diffusion = noise trading, uncertainty
    
Kalshi, Polymarket, Metaculus = implementations of this structure.
"""

using Gay
using Colors: RGB

# ═══════════════════════════════════════════════════════════════════════════
# Market Structure
# ═══════════════════════════════════════════════════════════════════════════

"""
A binary prediction market.
"""
struct Market
    id::Symbol
    question::String
    prob::Float64           # Current probability (price)
    volume::Float64         # Trading volume
    dependencies::Vector{Symbol}  # Parent markets (hierarchical)
end

"""
A market system with composition structure.
"""
struct MarketSystem
    markets::Dict{Symbol, Market}
    correlations::Dict{Tuple{Symbol,Symbol}, Float64}  # Pairwise correlations
end

# ═══════════════════════════════════════════════════════════════════════════
# Example Markets (Kalshi-style)
# ═══════════════════════════════════════════════════════════════════════════

function example_markets()
    markets = Dict{Symbol, Market}(
        :fed_cut => Market(:fed_cut, "Fed cuts rates in 2025?", 0.65, 1.2e6, Symbol[]),
        :recession => Market(:recession, "US recession in 2025?", 0.25, 800e3, Symbol[]),
        :sp500_up => Market(:sp500_up, "S&P 500 up 10%+ in 2025?", 0.45, 500e3, [:fed_cut, :recession]),
        :btc_100k => Market(:btc_100k, "Bitcoin > $100k in 2025?", 0.55, 2.0e6, [:fed_cut]),
        :election => Market(:election, "GOP wins 2028 presidency?", 0.52, 3.5e6, Symbol[]),
        :senate => Market(:senate, "GOP wins Senate 2026?", 0.60, 1.5e6, Symbol[]),
        :trifecta => Market(:trifecta, "GOP trifecta 2028?", 0.35, 900e3, [:election, :senate]),
    )
    
    correlations = Dict{Tuple{Symbol,Symbol}, Float64}(
        (:fed_cut, :sp500_up) => 0.4,
        (:recession, :sp500_up) => -0.6,
        (:fed_cut, :recession) => 0.3,
        (:election, :senate) => 0.5,
        (:election, :trifecta) => 0.8,
        (:senate, :trifecta) => 0.7,
        (:btc_100k, :fed_cut) => 0.35,
    )
    
    MarketSystem(markets, correlations)
end

# ═══════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════

function ansi(c::RGB)
    r = round(Int, clamp(c.r, 0, 1) * 255)
    g = round(Int, clamp(c.g, 0, 1) * 255)
    b = round(Int, clamp(c.b, 0, 1) * 255)
    "\e[38;2;$(r);$(g);$(b)m"
end

const R = "\e[0m"
const BOLD = "\e[1m"
const DIM = "\e[2m"

"""
Color for probability: red (low) → yellow (50%) → green (high)
"""
function prob_color(p::Float64)
    if p < 0.5
        # Red to yellow
        r = 1.0
        g = 2 * p
        b = 0.0
    else
        # Yellow to green
        r = 2 * (1 - p)
        g = 1.0
        b = 0.0
    end
    RGB(r, g, b)
end

"""
Render a single market.
"""
function render_market(m::Market; indent::Int=0)
    c = prob_color(m.prob)
    pad = "  "^indent
    
    # Probability bar
    bar_width = 20
    filled = round(Int, m.prob * bar_width)
    bar = "$(ansi(c))" * "█"^filled * "$(R)" * "░"^(bar_width - filled)
    
    prob_pct = round(m.prob * 100, digits=1)
    vol_str = m.volume >= 1e6 ? "$(round(m.volume/1e6, digits=1))M" : "$(round(m.volume/1e3, digits=0))k"
    
    println("$pad$(ansi(c))●$(R) $(m.question)")
    println("$pad  $bar $(prob_pct)%  (\$$vol_str volume)")
end

"""
Render the market system with hierarchy.
"""
function render_market_system(sys::MarketSystem; seed::Integer=42)
    gay_seed!(seed)
    
    println()
    println("  $(BOLD)Prediction Market System$(R)")
    println("  ════════════════════════════════════════════════════════════")
    println()
    
    # Group by dependencies (root markets first)
    roots = [m for (id, m) in sys.markets if isempty(m.dependencies)]
    dependents = [m for (id, m) in sys.markets if !isempty(m.dependencies)]
    
    println("  $(BOLD)Root Markets$(R) (no dependencies):")
    println()
    for m in roots
        render_market(m; indent=1)
        println()
    end
    
    println("  $(BOLD)Derived Markets$(R) (depend on root markets):")
    println()
    for m in dependents
        deps_str = join([":" * string(d) for d in m.dependencies], ", ")
        println("    $(DIM)depends on: $deps_str$(R)")
        render_market(m; indent=2)
        println()
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# Market as 2-Transducer
# ═══════════════════════════════════════════════════════════════════════════

function render_market_transducer(; seed::Integer=42)
    gay_seed!(seed)
    
    c_in = next_color(SRGB())
    c_state = next_color(SRGB())
    c_out = next_color(SRGB())
    
    println()
    println("  $(BOLD)Prediction Market as 2-Transducer$(R)")
    println("  ════════════════════════════════════════════════════════════")
    println()
    println("    $(ansi(c_in))Input A$(R)          $(ansi(c_state))State Q$(R)            $(ansi(c_out))Output B$(R)")
    println("    ─────────        ─────────          ─────────")
    println("    $(ansi(c_in))trades$(R)           $(ansi(c_state))order book$(R)         $(ansi(c_out))prices$(R)")
    println("    $(ansi(c_in))news$(R)     ───▶    $(ansi(c_state))beliefs$(R)    ───▶    $(ansi(c_out))P(event)$(R)")
    println("    $(ansi(c_in))events$(R)           $(ansi(c_state))volume$(R)             $(ansi(c_out))∈ [0,1]$(R)")
    println()
    println("  $(DIM)Profunctor t : trades × beliefs → prices × new_beliefs$(R)")
    println()
    println("  $(BOLD)State category Q has morphisms:$(R)")
    println("    • buy order:  shift probability up")
    println("    • sell order: shift probability down")
    println("    • news:       jump to new belief")
    println("    • time decay: drift toward prior")
    println()
end

# ═══════════════════════════════════════════════════════════════════════════
# Market as Markov Blanket
# ═══════════════════════════════════════════════════════════════════════════

function render_market_blanket(; seed::Integer=42)
    gay_seed!(seed)
    
    c_ext = next_color(SRGB())
    c_bln = next_color(SRGB())
    c_int = next_color(SRGB())
    
    println()
    println("  $(BOLD)Market Maker as Markov Blanket$(R)")
    println("  ════════════════════════════════════════════════════════════")
    println()
    println("    $(ansi(c_ext))┌────────────────────────────────────────┐$(R)")
    println("    $(ansi(c_ext))│           EXTERNAL (s)                 │$(R)")
    println("    $(ansi(c_ext))│   traders, news sources, world events │$(R)")
    println("    $(ansi(c_ext))│   individual beliefs, private info    │$(R)")
    println("    $(ansi(c_ext))└───────────────────┬────────────────────┘$(R)")
    println("                        $(ansi(c_ext))│$(R)")
    println("                        $(ansi(c_ext))▼$(R)")
    println("    $(ansi(c_bln))╔════════════════════════════════════════╗$(R)")
    println("    $(ansi(c_bln))║           BLANKET (b)                  ║$(R)")
    println("    $(ansi(c_bln))║   order book, bid/ask spread,         ║$(R)")
    println("    $(ansi(c_bln))║   market maker, exchange interface    ║$(R)")
    println("    $(ansi(c_bln))╚═══════════════════╤════════════════════╝$(R)")
    println("                        $(ansi(c_bln))│$(R)")
    println("                        $(ansi(c_bln))▼$(R)")
    println("    $(ansi(c_int))┌────────────────────────────────────────┐$(R)")
    println("    $(ansi(c_int))│           INTERNAL (z)                 │$(R)")
    println("    $(ansi(c_int))│   clearing price, market state,       │$(R)")
    println("    $(ansi(c_int))│   aggregate probability P(event)      │$(R)")
    println("    $(ansi(c_int))└────────────────────────────────────────┘$(R)")
    println()
    println("  $(DIM)The price IS the sufficient statistic for collective belief.$(R)")
    println("  $(DIM)Individual trader beliefs are screened off by the order book.$(R)")
    println()
end

# ═══════════════════════════════════════════════════════════════════════════
# Tracing: Nested Markets (from screenshot)
# ═══════════════════════════════════════════════════════════════════════════

function render_market_tracing(; seed::Integer=42)
    gay_seed!(seed)
    
    c_z = next_color(SRGB())  # Outer system
    c_y1 = next_color(SRGB()) # First subsystem
    c_y2 = next_color(SRGB()) # Second subsystem
    
    println()
    println("  $(BOLD)Tracing: Nested Market Structure$(R)")
    println("  ════════════════════════════════════════════════════════════")
    println()
    println("  $(DIM)(Like SmallNestingPic from screenshot)$(R)")
    println()
    println("    $(ansi(c_z))╔═══════════════════════════════════════════════════════╗$(R)")
    println("    $(ansi(c_z))║                      Z: 2028 Election                 ║$(R)")
    println("    $(ansi(c_z))║                                                       ║$(R)")
    println("    $(ansi(c_z))║$(R)  $(ansi(c_y1))┌─────────────────────┐$(R)  $(ansi(c_y2))┌─────────────────────┐$(R)  $(ansi(c_z))║$(R)")
    println("    $(ansi(c_z))║$(R)  $(ansi(c_y1))│ Y1: Presidential    │$(R)  $(ansi(c_y2))│ Y2: Congressional   │$(R)  $(ansi(c_z))║$(R)")
    println("    $(ansi(c_z))║$(R)  $(ansi(c_y1))│  • X11: GOP wins    │$(R)  $(ansi(c_y2))│  • X21: Senate GOP  │$(R)  $(ansi(c_z))║$(R)")
    println("    $(ansi(c_z))║$(R)  $(ansi(c_y1))│  • X12: DEM wins    │$(R)  $(ansi(c_y2))│  • X22: House GOP   │$(R)  $(ansi(c_z))║$(R)")
    println("    $(ansi(c_z))║$(R)  $(ansi(c_y1))│  • X13: margin >5%  │$(R)  $(ansi(c_y2))│                     │$(R)  $(ansi(c_z))║$(R)")
    println("    $(ansi(c_z))║$(R)  $(ansi(c_y1))└─────────────────────┘$(R)  $(ansi(c_y2))└─────────────────────┘$(R)  $(ansi(c_z))║$(R)")
    println("    $(ansi(c_z))║                                                       ║$(R)")
    println("    $(ansi(c_z))║$(R)           Derived: $(BOLD)Trifecta$(R) = X11 ∧ X21 ∧ X22          $(ansi(c_z))║$(R)")
    println("    $(ansi(c_z))╚═══════════════════════════════════════════════════════╝$(R)")
    println()
    println("  $(DIM)Tracing = how information propagates through nested structure$(R)")
    println("  $(DIM)Arbitrage = enforcing consistency across correlated markets$(R)")
    println()
end

# ═══════════════════════════════════════════════════════════════════════════
# Fokker-Planck for Perpetual Markets
# ═══════════════════════════════════════════════════════════════════════════

function render_perpetual_dynamics(; seed::Integer=42)
    gay_seed!(seed)
    
    println()
    println("  $(BOLD)Perpetual Markets: Fokker-Planck Dynamics$(R)")
    println("  ════════════════════════════════════════════════════════════")
    println()
    println("    dP/dt = -∇·(μP) + D∇²P")
    println()
    println("    $(DIM)P(x,t)$(R) = probability distribution over prices")
    println("    $(DIM)μ(x)$(R)   = drift from information arrival")
    println("    $(DIM)D$(R)      = diffusion from noise trading")
    println()
    println("  $(BOLD)Information events:$(R)")
    println("    • News → jump in μ (sudden drift)")
    println("    • Trade → small shift (diffusion)")
    println("    • Time decay → drift toward prior")
    println()
    println("  $(BOLD)Perpetual = no expiry:$(R)")
    println("    • Continuous price discovery")
    println("    • Funding rate = cost of holding position")
    println("    • Arbitrage with spot market")
    println()
    println("  $(DIM)The stationary distribution ρ_∞ = market consensus$(R)")
    println()
end

# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

function main(; seed::Integer=42)
    println()
    println("  ╔════════════════════════════════════════════════════════════════════╗")
    println("  ║          $(BOLD)PREDICTION MARKETS$(R) as Compositional Systems            ║")
    println("  ║   Kalshi, Polymarket, Metaculus — structured probability           ║")
    println("  ╚════════════════════════════════════════════════════════════════════╝")
    
    sys = example_markets()
    render_market_system(sys; seed=seed)
    
    render_market_transducer(seed=seed)
    render_market_blanket(seed=seed)
    render_market_tracing(seed=seed)
    render_perpetual_dynamics(seed=seed)
    
    # Final
    gay_seed!(seed + 100)
    c = next_color(SRGB())
    println()
    println("  $(ansi(c))Price is probability. Structure is arbitrage.$(R)")
    println("  $(ansi(c))The market traces information through nested events.$(R)")
    println()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
