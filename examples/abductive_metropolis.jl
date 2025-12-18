# # Abductive Metropolis: From Thermalization to Ergodic Equilibrium
#
# Using LispSyntax.jl throughout - S-expressions for colored Monte Carlo logic.
#
# ## The Two Phases
#
# 1. **Abduction** (thermalization): Inferring equilibrium from observations
#    - Non-conservative: information is created as we learn the distribution
#    - Colors shift rapidly as the system explores
#
# 2. **Conservative** (post-equilibrium): Reversible logic preserving ergodicity
#    - Detailed balance ensures time-reversibility
#    - Colors stabilize around the equilibrium manifold
#
# ## Setup with LispSyntax

using Pkg
Pkg.activate(dirname(dirname(@__FILE__)))

using Gay
using LispSyntax
using Colors
using Statistics

# ═══════════════════════════════════════════════════════════════════════════
# Lisp-style helpers
# ═══════════════════════════════════════════════════════════════════════════

# Define Lisp macros for colored MC operations
lisp"(defn gay-ctx (seed) (GayMCContext seed))"

lisp"(defn gay-sweep (ctx) (gay_sweep! ctx))"

lisp"(defn gay-color (ctx) (color_state ctx))"

lisp"(defn fnv (text) (Gay.fnv1a text))"

# ANSI color output in Lisp
lisp"(defn ansi-bg (c) 
  (string \"\\e[48;2;\" 
          (round Int (* 255 (red c))) \";\"
          (round Int (* 255 (green c))) \";\"
          (round Int (* 255 (blue c))) \"m\"))"

lisp"(defn ansi-reset () \"\\e[0m\")"

lisp"(defn show-color (c)
  (print (ansi-bg c))
  (print \"  \")
  (print (ansi-reset)))"

# ═══════════════════════════════════════════════════════════════════════════
# Abductive Logic: Thermalization Phase
# ═══════════════════════════════════════════════════════════════════════════

# Abduction: inference to the best explanation
# During thermalization, we're *learning* what equilibrium looks like
# This is inherently non-conservative (irreversible information flow)

lisp"(defn abductive-score (colors)
  (let [rs (map red colors)
        gs (map green colors)
        bs (map blue colors)]
    (+ (std rs) (std gs) (std bs))))"

lisp"(defn abductive-sweep! (ctx chain beta window)
  (let [res (gay-sweep ctx)
        rng (getindex res 1)
        color (getindex res 2)
        history (getfield ctx :color_history)
        n (length history)
        recent (if (> n window)
                 (view history (- n window -1) n)
                 history)
        score (if (>= (length recent) 3)
                (abductive-score recent)
                1)]
    {:color color
     :score score
     :hypothesis (/ 1 (+ score 0.01))}))"

# ═══════════════════════════════════════════════════════════════════════════
# Conservative Logic: Post-Equilibrium Phase  
# ═══════════════════════════════════════════════════════════════════════════

# Conservative (reversible) logic preserves information
# After thermalization, detailed balance ensures microscopic reversibility
# Every forward transition has an equally probable reverse

lisp"(defn conservative-sweep! (ctx chain beta)
  (let [res (gay-sweep ctx)
        rng (getindex res 1)
        color (getindex res 2)
        reversibility-marker (mod (+ (* (red color) 1000) 
                                      (* (green color) 100)
                                      (* (blue color) 10)) 
                                   1)]
    {:color color
     :reversible true
     :marker reversibility-marker}))"

# ═══════════════════════════════════════════════════════════════════════════
# Ergodic Equilibrium Detection
# ═══════════════════════════════════════════════════════════════════════════

lisp"(defn ergodic-test (colors threshold)
  (let [n (length colors)
        bins (Set)
        _ (for [c colors]
            (let [ri (floor Int (* 4 (red c)))
                  gi (floor Int (* 4 (green c)))
                  bi (floor Int (* 4 (blue c)))
                  key (+ ri (* 4 gi) (* 16 bi))]
              (push! bins key)))]
    (> (/ (length bins) 64.0) threshold)))"

lisp"(defn find-equilibrium (ctx n-max threshold)
  (loop [i 0]
    (if (>= i n-max)
      {:found false :sweeps i}
      (do
        (gay-sweep ctx)
        (let [history (getfield ctx :color_history)
              n (length history)]
          (if (and (> n 20) 
                   (ergodic-test (view history (- n 20) n) threshold))
            {:found true :sweeps i :equilibrium-color (color_state ctx)}
            (recur (+ i 1))))))))"

# ═══════════════════════════════════════════════════════════════════════════
# Main Simulation: Abduction → Conservative Transition
# ═══════════════════════════════════════════════════════════════════════════

println("\n◈ Abductive Metropolis with LispSyntax.jl")
println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

# Create context using Lisp
ctx = lisp"(gay-ctx 42)"

println("\n📊 Phase 1: Abductive Thermalization")
println("   Learning the equilibrium distribution...")
println()

# Abductive phase - colors shift as we learn
print("   Abduction: ")
abductive_scores = Float64[]

for i in 1:40
    rng, color = gay_sweep!(ctx)
    
    # Compute abductive score (color variance)
    history = ctx.color_history
    if length(history) >= 5
        recent = history[max(1,end-4):end]
        score = std([red(c) for c in recent]) + 
                std([green(c) for c in recent]) + 
                std([blue(c) for c in recent])
        push!(abductive_scores, score)
    end
    
    # Show color
    r = round(Int, red(color) * 255)
    g = round(Int, green(color) * 255)
    b = round(Int, blue(color) * 255)
    print("\e[48;2;$(r);$(g);$(b)m \e[0m")
end
println()

# Show abductive learning curve
println("\n   Abductive scores (color variance over time):")
print("   ")
for (i, s) in enumerate(abductive_scores[1:5:end])
    level = min(8, max(1, round(Int, s * 10)))
    bar = "█" ^ level
    print("$(bar) ")
end
println()

# Detect equilibrium
println("\n🔍 Detecting Ergodic Equilibrium...")

# Check color coverage
history = ctx.color_history
bins = Set{Int}()
for c in history[max(1,end-20):end]
    ri = floor(Int, red(c) * 4)
    gi = floor(Int, green(c) * 4)
    bi = floor(Int, blue(c) * 4)
    push!(bins, ri + 4*gi + 16*bi)
end
coverage = length(bins) / 64.0

equilibrium_color = color_state(ctx)
r = round(Int, red(equilibrium_color) * 255)
g = round(Int, green(equilibrium_color) * 255)
b = round(Int, blue(equilibrium_color) * 255)

println("   Color-space coverage: $(round(coverage * 100, digits=1))%")
println("   Equilibrium color: \e[48;2;$(r);$(g);$(b)m    \e[0m")

# Conservative phase
println("\n📊 Phase 2: Conservative (Reversible) Dynamics")
println("   Detailed balance preserves equilibrium...")
println()

print("   Conservative: ")
for i in 1:40
    rng, color = gay_sweep!(ctx)
    
    r = round(Int, red(color) * 255)
    g = round(Int, green(color) * 255)
    b = round(Int, blue(color) * 255)
    print("\e[48;2;$(r);$(g);$(b)m \e[0m")
end
println()

# Final state
final_color = color_state(ctx)
r = round(Int, red(final_color) * 255)
g = round(Int, green(final_color) * 255)
b = round(Int, blue(final_color) * 255)

println("\n   Final equilibrium color: \e[48;2;$(r);$(g);$(b)m    \e[0m")
println("   Total sweeps: $(ctx.sweep_count)")

# ═══════════════════════════════════════════════════════════════════════════
# Lisp S-expression Summary
# ═══════════════════════════════════════════════════════════════════════════

println("\n📜 Lisp S-expression Summary")
println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

# Demonstrate pure Lisp operations
# Pure Julia equivalent for the summary to avoid Lisp parsing issues
seed = 1337
ctx = GayMCContext(UInt64(seed))
for _ in 1:20
    gay_sweep!(ctx)
end

result = Dict(
    :seed => seed,
    :sweeps => ctx.sweep_count,
    :state_color => color_state(ctx)
)

println("   (let [seed 1337")
println("         ctx (gay-ctx seed)]")
println("     (dotimes [_ 20] (gay-sweep ctx))")
println("     {:seed seed :sweeps ... :state-color ...})")
println()
println("   Result: $(result)")

# Color the result
if haskey(result, :state_color) || haskey(result, Symbol("state-color"))
    sc = get(result, Symbol("state-color"), get(result, :state_color, nothing))
    if sc !== nothing
        r = round(Int, red(sc) * 255)
        g = round(Int, green(sc) * 255)
        b = round(Int, blue(sc) * 255)
        println("   State: \e[48;2;$(r);$(g);$(b)m    \e[0m")
    end
end

println("\n◈ Abductive → Conservative transition complete!")
println()
println("   Abduction:    Infer equilibrium from colored observations")
println("   Conservative: Preserve detailed balance with reversible colors")
println("   Ergodicity:   Color-space coverage ≈ phase-space coverage")
