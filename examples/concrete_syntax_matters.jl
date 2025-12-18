# # Concrete Syntax Matters, Actually
#
# Inspired by Slim Lim's Topos Institute talk (Dec 4, 2025)
# https://www.youtube.com/watch?v=...
#
# "Too many programming languages researchers dismiss concrete syntax as an
# afterthought: arbitrary, superficial, or distracting from matters of 'actual'
# (semantic) importance. This received view ignores a critical factor: the human
# at the computer."
#
# This example demonstrates how concrete syntax choices affect color derivation
# comprehension - same semantics, different notations, different understanding.

using Gay
using Gay: ka_colors, xor_fingerprint, hash_color

# ═══════════════════════════════════════════════════════════════════════════════
# The ANY vs UNKNOWN Problem (TypeScript's vernacular misconception)
# ═══════════════════════════════════════════════════════════════════════════════
#
# In TypeScript:
#   - `any` = dynamic type (escape hatch, breaks type safety)
#   - `unknown` = top type (safe, requires narrowing)
#
# But programmers use `any` when they mean `unknown` because:
#   1. Dictionary definition of "any" matches mental model of "any possible type"
#   2. `any` is marketed more heavily in docs
#   3. `unknown` is buried
#
# Let's model this with color palettes:

"""
Colors named by their VERNACULAR meaning vs SEMANTIC meaning.
Same hash, different names → different mental models.
"""
module VernacularColors

using ..Gay: hash_color

# What programmers THINK they want (vernacular)
const any_color = hash_color(UInt64(0x616e79), UInt64(1))        # "any"
const everything_color = hash_color(UInt64(0x65766572), UInt64(1)) # "ever"

# What they ACTUALLY need (semantic)  
const dynamic_color = hash_color(UInt64(0x64796e), UInt64(1))    # "dyn" 
const top_color = hash_color(UInt64(0x746f70), UInt64(1))        # "top"
const unknown_color = hash_color(UInt64(0x756e6b), UInt64(1))    # "unk"

end

# ═══════════════════════════════════════════════════════════════════════════════
# Callback Hell vs Promise Chains vs Async/Await
# ═══════════════════════════════════════════════════════════════════════════════
#
# People changed JavaScript's async semantics because of INDENTATION COMPLAINTS.
# Same abstract syntax (nested scopes), different concrete syntax → different adoption.

"""
Three notations for the same data flow dependency graph:
  A → B → C, where C also needs A's output

All three are semantically equivalent for linear flow.
Only callbacks and async/await handle non-linear flow elegantly.
"""
module AsyncNotations

# Callback style (what JS called "callback hell")
# Concrete syntax: nested, closing delimiters trail
callback_style = """
runA(function(a) {
    runB(a, function(b) {
        runC(a, b, function(c) {  // a is in scope!
            done(c);
        });
    });
});
"""

# Promise chain style (marketed as "solution")
# Concrete syntax: fluent, linear, but loses scope
promise_style = """
runA()
    .then(a => runB(a))
    .then(b => runC(???, b))  // Lost access to a!
    .then(c => done(c));
"""

# Promise chain with workaround (reality)
promise_ugly = """
runA()
    .then(a => runB(a).then(b => [a, b]))
    .then(([a, b]) => runC(a, b))
    .then(c => done(c));
"""

# Async/await (do-notation equivalent)
# Concrete syntax: sequential-looking, nested scope preserved
async_style = """
async function main() {
    const a = await runA();
    const b = await runB(a);
    const c = await runC(a, b);  // a is in scope!
    done(c);
}
"""

end

# ═══════════════════════════════════════════════════════════════════════════════
# Sussman Form: Syntactic Sugar That Hides First-Class Functions
# ═══════════════════════════════════════════════════════════════════════════════

"""
Two equivalent Scheme definitions:

1. Lambda binding (reveals first-class functions):
   (define factorial (lambda (n) (if (= n 0) 1 (* n (factorial (- n 1))))))

2. Sussman form (hides that functions are just values):
   (define (factorial n) (if (= n 0) 1 (* n (factorial (- n 1)))))

Same AST, different concrete syntax → different mental model of recursion.
"""
module SussmanForm

# The abstract syntax tree is identical:
abstract_syntax = Dict(
    :type => :define,
    :name => :factorial,
    :value => Dict(
        :type => :lambda,
        :params => [:n],
        :body => Dict(
            :type => :if,
            :cond => (:(==), :n, 0),
            :then_branch => 1,
            :else_branch => (:*, :n, (:factorial, (:-, :n, 1)))
        )
    )
)

# But the concrete syntax differs in visual nesting and delimiter count

end

# ═══════════════════════════════════════════════════════════════════════════════
# Color Notation Experiment: Same Semantics, Different Syntax
# ═══════════════════════════════════════════════════════════════════════════════

"""
    color_notation_experiment()

Demonstrate that concrete syntax affects comprehension even for colors.
Same underlying hash_color semantics, different notations.
"""
function color_notation_experiment()
    println()
    println("═" ^ 70)
    println("  CONCRETE SYNTAX MATTERS: Color Notation Experiment")
    println("  Inspired by Slim Lim @ Topos Institute (Dec 4, 2025)")
    println("═" ^ 70)
    println()
    
    seed = UInt64(42069)
    
    # Same semantics, different concrete syntax notations
    notations = [
        # Notation 1: Positional (terse, array-like)
        ("Positional: [42069, 1]",
         "colors[seed][idx]",
         () -> hash_color(seed, UInt64(1))),
        
        # Notation 2: Named (verbose, self-documenting)
        ("Named: hash_color(seed=42069, index=1)",
         "hash_color(seed, index)",
         () -> hash_color(seed, UInt64(1))),
        
        # Notation 3: Fluent/OO (method chaining)
        ("Fluent: Seed(42069).color_at(1)",
         "seed.color_at(idx)",
         () -> hash_color(seed, UInt64(1))),
        
        # Notation 4: Lisp (S-expression)
        ("Lisp: (color-at 42069 1)",
         "(color-at seed idx)",
         () -> hash_color(seed, UInt64(1))),
        
        # Notation 5: Mathematical (subscript-like)
        ("Math: C₄₂₀₆₉,₁",
         "C_{seed,idx}",
         () -> hash_color(seed, UInt64(1))),
    ]
    
    println("  All notations produce the SAME color (same semantics):")
    println("  But different syntax → different mental models")
    println()
    
    for (name, pattern, func) in notations
        r, g, b = func()
        hex = "#$(string(round(Int, r*255), base=16, pad=2))$(string(round(Int, g*255), base=16, pad=2))$(string(round(Int, b*255), base=16, pad=2))" |> uppercase
        
        ri, gi, bi = round(Int, r*255), round(Int, g*255), round(Int, b*255)
        print("  \e[38;2;$(ri);$(gi);$(bi)m████\e[0m ")
        println("$(rpad(name, 45)) → $hex")
    end
    
    println()
    println("─" ^ 70)
    println("  Wadler's Law: Time spent discussing a feature ∝ 2^position")
    println("    0: semantics, 1: syntax, 2: lexical syntax, 3: comments")
    println()
    println("  But: concrete syntax IS the user interface.")
    println("       \"any\" vs \"unknown\" in TypeScript → vernacular misconceptions")
    println("       Callback indentation → people changed the async semantics")
    println("─" ^ 70)
    println()
    
    # Show the TypeScript any/unknown problem with color
    println("  THE ANY vs UNKNOWN PROBLEM:")
    println()
    
    # What programmers type vs what they need
    any_seed = hash(string("any"))
    unknown_seed = hash(string("unknown"))
    dynamic_seed = hash(string("dynamic"))
    top_seed = hash(string("top"))
    
    pairs = [
        ("any (vernacular)", any_seed, "unknown (semantic)", unknown_seed),
        ("dynamic (vernacular)", dynamic_seed, "top (semantic)", top_seed),
    ]
    
    for (vname, vseed, sname, sseed) in pairs
        vr, vg, vb = hash_color(UInt64(vseed), UInt64(1))
        sr, sg, sb = hash_color(UInt64(sseed), UInt64(1))
        
        vi, vgi, vbi = round(Int, vr*255), round(Int, vg*255), round(Int, vb*255)
        si, sgi, sbi = round(Int, sr*255), round(Int, sg*255), round(Int, sb*255)
        
        print("  \e[38;2;$(vi);$(vgi);$(vbi)m████\e[0m $(rpad(vname, 25))")
        print(" ≠ ")
        println("\e[38;2;$(si);$(sgi);$(sbi)m████\e[0m $sname")
    end
    
    println()
    println("  Same INTENT, different KEYWORD → different TYPE SAFETY")
    println("═" ^ 70)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Fast SPI Verification: Concrete Syntax for Billion-Scale Correctness
# ═══════════════════════════════════════════════════════════════════════════════

"""
    fast_spi_demo(n=1_000_000)

Demonstrate that XOR fingerprinting is a CONCRETE SYNTAX for correctness proofs.
Instead of comparing 3M floats, we compare 1 hash → faster mental model.
"""
function fast_spi_demo(n::Int=1_000_000)
    println()
    println("═" ^ 70)
    println("  FAST SPI: XOR Fingerprint as Concrete Syntax for Correctness")
    println("═" ^ 70)
    println()
    
    # Generate colors
    t = @elapsed colors = ka_colors(n, 42069)
    fp = xor_fingerprint(colors)
    
    println("  Abstract syntax:  compare $(n * 3) Float32 values")
    println("  Concrete syntax:  compare 1 UInt32 fingerprint")
    println()
    println("  Generated $n colors in $(round(t * 1000, digits=1))ms")
    println("  Fingerprint: 0x$(string(fp, base=16, pad=8))")
    println()
    
    # Verify SPI with different "notations" for correctness
    println("  Three notations for 'these colors are correct':")
    println()
    
    # Notation 1: Element-wise (verbose, slow)
    print("    1. ∀i: colors₁[i] == colors₂[i]")
    t1 = @elapsed begin
        colors2 = ka_colors(n, 42069)
        match1 = all(colors .== colors2)
    end
    println("  [$(round(t1*1000, digits=1))ms] $(match1 ? "✓" : "✗")")
    
    # Notation 2: Matrix norm (mathematical, medium)
    print("    2. ‖colors₁ - colors₂‖ < ε")
    t2 = @elapsed begin
        colors2 = ka_colors(n, 42069)
        match2 = sum(abs.(colors .- colors2)) < 1e-6
    end
    println("              [$(round(t2*1000, digits=1))ms] $(match2 ? "✓" : "✗")")
    
    # Notation 3: XOR fingerprint (terse, fast)
    print("    3. xor(colors₁) == xor(colors₂)")
    t3 = @elapsed begin
        colors2 = ka_colors(n, 42069)
        fp2 = xor_fingerprint(colors2)
        match3 = fp == fp2
    end
    println("            [$(round(t3*1000, digits=1))ms] $(match3 ? "✓" : "✗")")
    
    println()
    println("  Same semantics (correctness), different syntax → different UX")
    println("  XOR fingerprint: O(1) to verify, O(n) to generate")
    println("═" ^ 70)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

function main()
    color_notation_experiment()
    fast_spi_demo(1_000_000)
    
    println()
    println("╔" * "═" ^ 68 * "╗")
    println("║  \"Programming languages have user interfaces. Concrete syntax IS   ║")
    println("║   the foremost user interface for most programming languages.\"    ║")
    println("║                                          — Slim Lim, Topos 2025   ║")
    println("╚" * "═" ^ 68 * "╝")
    println()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
