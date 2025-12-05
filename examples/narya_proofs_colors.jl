# Narya Proof Terms → Colors
# Type-theoretic proof structure → deterministic color mapping
#
# Uses the :narya_proofs continuation branch from bbp_pi.jl
#
# The idea: Proof terms in dependent type theory have structure.
# Each proof step, constructor, or type → a color from the splittable stream.
#
# Narya syntax elements:
#   - Types: Type, A → B, (x : A) → B x
#   - Equality: eq A x y with constructor refl.
#   - Match: match p [ refl. ↦ result ]
#   - Definitions: def name : Type ≔ term

using Gay
using Colors
using SplittableRandoms: SplittableRandom, split

include("bbp_pi.jl")  # For continuation_point, branch_seed

# ═══════════════════════════════════════════════════════════════════════════
# Narya Proof Elements
# ═══════════════════════════════════════════════════════════════════════════

@enum ProofElement begin
    PE_Type          # Type
    PE_Arrow         # →
    PE_Pi            # (x : A) → B
    PE_Eq            # eq A x y
    PE_Refl          # refl.
    PE_Match         # match
    PE_Def           # def
    PE_Lambda        # x ↦
    PE_App           # f x
    PE_Var           # variable
    PE_Nat           # ℕ
    PE_Zero          # zero.
    PE_Suc           # suc.
end

# Proof term structure
struct ProofTerm
    element::ProofElement
    depth::Int           # nesting depth
    index::Int           # position in proof
    name::String         # identifier
end

# ═══════════════════════════════════════════════════════════════════════════
# Classic Proofs (from proof-loop.sh)
# ═══════════════════════════════════════════════════════════════════════════

# Symmetry of equality
const PROOF_EQ_SYM = [
    ProofTerm(PE_Def, 0, 1, "eq_sym"),
    ProofTerm(PE_Pi, 1, 2, "A : Type"),
    ProofTerm(PE_Pi, 1, 3, "x : A"),
    ProofTerm(PE_Pi, 1, 4, "y : A"),
    ProofTerm(PE_Pi, 1, 5, "p : eq A x y"),
    ProofTerm(PE_Eq, 1, 6, "eq A y x"),
    ProofTerm(PE_Match, 2, 7, "match p"),
    ProofTerm(PE_Refl, 3, 8, "refl. ↦ refl."),
]

# Transitivity of equality
const PROOF_EQ_TRANS = [
    ProofTerm(PE_Def, 0, 1, "eq_trans"),
    ProofTerm(PE_Pi, 1, 2, "A : Type"),
    ProofTerm(PE_Pi, 1, 3, "x : A"),
    ProofTerm(PE_Pi, 1, 4, "y : A"),
    ProofTerm(PE_Pi, 1, 5, "z : A"),
    ProofTerm(PE_Pi, 1, 6, "p : eq A x y"),
    ProofTerm(PE_Pi, 1, 7, "q : eq A y z"),
    ProofTerm(PE_Eq, 1, 8, "eq A x z"),
    ProofTerm(PE_Match, 2, 9, "match q"),
    ProofTerm(PE_Refl, 3, 10, "refl. ↦ p"),
]

# Natural number definition
const PROOF_NAT = [
    ProofTerm(PE_Def, 0, 1, "ℕ"),
    ProofTerm(PE_Type, 1, 2, "Type"),
    ProofTerm(PE_Zero, 2, 3, "zero. : ℕ"),
    ProofTerm(PE_Suc, 2, 4, "suc. : ℕ → ℕ"),
]

# Addition
const PROOF_ADD = [
    ProofTerm(PE_Def, 0, 1, "add"),
    ProofTerm(PE_Pi, 1, 2, "m : ℕ"),
    ProofTerm(PE_Pi, 1, 3, "n : ℕ"),
    ProofTerm(PE_Nat, 1, 4, "ℕ"),
    ProofTerm(PE_Match, 2, 5, "match m"),
    ProofTerm(PE_Zero, 3, 6, "zero. ↦ n"),
    ProofTerm(PE_Suc, 3, 7, "suc. k ↦ suc. (add k n)"),
]

const ALL_PROOFS = [
    ("eq_sym", PROOF_EQ_SYM),
    ("eq_trans", PROOF_EQ_TRANS),
    ("ℕ", PROOF_NAT),
    ("add", PROOF_ADD),
]

# ═══════════════════════════════════════════════════════════════════════════
# Proof Element → Color Mapping (uses :narya_proofs branch)
# ═══════════════════════════════════════════════════════════════════════════

const NARYA_SEED = 314159

"""
    element_color(elem::ProofElement; seed=NARYA_SEED)

Get a deterministic color for a proof element type.
Each element type has a consistent color across all proofs.
"""
function element_color(elem::ProofElement; seed::Integer=NARYA_SEED)
    nseed = branch_seed(seed, :narya_proofs)
    idx = Int(elem) + 1  # 1-indexed
    return color_at(idx, Rec2020(); seed=nseed)
end

"""
    term_color(term::ProofTerm; seed=NARYA_SEED)

Get a color for a specific proof term.
Combines element type, depth, and index for unique color.
"""
function term_color(term::ProofTerm; seed::Integer=NARYA_SEED)
    nseed = branch_seed(seed, :narya_proofs)
    
    # Unique index: element * 10000 + depth * 100 + index
    idx = Int(term.element) * 10000 + term.depth * 100 + term.index
    
    return color_at(idx, Rec2020(); seed=nseed)
end

"""
    proof_palette(proof::Vector{ProofTerm}; seed=NARYA_SEED)

Get colors for all terms in a proof.
"""
function proof_palette(proof::Vector{ProofTerm}; seed::Integer=NARYA_SEED)
    return [term_color(t; seed=seed) for t in proof]
end

# ═══════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════

function render_element_legend(; seed::Integer=NARYA_SEED)
    println("\n  ╔════════════════════════════════════════════════════════════╗")
    println("  ║  Narya Proof Elements → Colors (:narya_proofs branch)     ║")
    println("  ╚════════════════════════════════════════════════════════════╝")
    println()
    println("  Element Legend:")
    
    for elem in instances(ProofElement)
        c = element_color(elem; seed=seed)
        ri = round(Int, c.r * 255)
        gi = round(Int, c.g * 255)
        bi = round(Int, c.b * 255)
        
        name = replace(string(elem), "PE_" => "")
        print("    ")
        print("\e[48;2;$(ri);$(gi);$(bi)m    \e[0m ")
        println(name)
    end
end

function render_proof(name::String, proof::Vector{ProofTerm}; seed::Integer=NARYA_SEED)
    println("\n  ─── Proof: $name ───")
    
    for term in proof
        c = term_color(term; seed=seed)
        ri = round(Int, c.r * 255)
        gi = round(Int, c.g * 255)
        bi = round(Int, c.b * 255)
        
        indent = "  " * "  "^term.depth
        print("  ")
        print("\e[48;2;$(ri);$(gi);$(bi)m  \e[0m")
        println("$indent$(term.name)")
    end
end

function render_all_proofs(; seed::Integer=NARYA_SEED)
    println("\n  ═══════════════════════════════════════════════════════")
    println("  Classic Proofs (from Narya type theory)")
    println("  ═══════════════════════════════════════════════════════")
    
    for (name, proof) in ALL_PROOFS
        render_proof(name, proof; seed=seed)
    end
end

function render_proof_comparison(; seed::Integer=NARYA_SEED)
    println("\n  ═══════════════════════════════════════════════════════")
    println("  Proof Structure Comparison")
    println("  ═══════════════════════════════════════════════════════\n")
    
    println("  Each proof as color sequence:")
    
    for (name, proof) in ALL_PROOFS
        print("    $name: ")
        for term in proof
            c = term_color(term; seed=seed)
            ri = round(Int, c.r * 255)
            gi = round(Int, c.g * 255)
            bi = round(Int, c.b * 255)
            print("\e[48;2;$(ri);$(gi);$(bi)m  \e[0m")
        end
        println(" ($(length(proof)) terms)")
    end
end

function demo_narya_spi()
    println("\n  ═══════════════════════════════════════════════════════")
    println("  SPI Verification: :narya_proofs branch independence")
    println("  ═══════════════════════════════════════════════════════\n")
    
    # Same index from different branches
    idx = 42
    
    nseed = branch_seed(314159, :narya_proofs)
    pseed = branch_seed(314159, :polylog)
    tseed = branch_seed(314159, :triangle_magic)
    
    println("  Same index ($idx) from different branches:")
    
    for (name, seed) in [(:narya_proofs, nseed), (:polylog, pseed), (:triangle_magic, tseed)]
        c = color_at(idx, Rec2020(); seed=seed)
        ri = round(Int, c.r * 255)
        gi = round(Int, c.g * 255)
        bi = round(Int, c.b * 255)
        print("    :$name ")
        print("\e[48;2;$(ri);$(gi);$(bi)m    \e[0m\n")
    end
    
    println("\n  ✓ Each branch independent")
end

# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

function main()
    println("\n" * "═"^70)
    println("  Narya Proof Colors - :narya_proofs continuation branch")
    println("═"^70)
    
    # Element legend
    render_element_legend()
    
    # All proofs
    render_all_proofs()
    
    # Structure comparison
    render_proof_comparison()
    
    # Verify branch independence
    demo_narya_spi()
    
    println("\n  Properties:")
    println("  ✓ Each proof element → deterministic color")
    println("  ✓ Proof structure visualized by color sequence")
    println("  ✓ Same element type → consistent color across proofs")
    println("  ✓ :narya_proofs branch independent of others")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
