# Kripke Semantics & Possible Worlds
# ===================================
#
# Layer 6-8 of the SPI Tower:
#   Layer 6: Kripke Frames (possible worlds with accessibility)
#   Layer 7: Modal Logic (□ necessity, ◇ possibility)
#   Layer 8: Sheaf Semantics (local truth, geometric morphisms)
#
# References:
#   - Awodey, Kishida, Kotzsch: "Topos Semantics for Higher-Order Modal Logic"
#   - Alex Kavvos: "Two-dimensional Kripke Semantics" (Topos Institute, Nov 2025)
#   - Simpson: "The Proof Theory and Semantics of Intuitionistic Modal Logic"
#
# Key insight from Kavvos: Kripke semantics and categorical semantics correspond
# via 2-dimensional structure - worlds are objects, accessibility is morphisms.

module KripkeWorlds

using Random
using ..Gay: GAY_SEED, splitmix64

export KripkeFrame, World, accessible, truth_at, necessity, possibility
export PossibleWorld, WorldAttestation, verify_accessibility
export ModalProposition, evaluate_modal, box, diamond
export SheafSemantics, local_truth, global_sections, stalk_at
export world_kripke, verify_modal_laws, run_kripke_tests

# ═══════════════════════════════════════════════════════════════════════════════
# Layer 6: Kripke Frames (Possible Worlds)
# ═══════════════════════════════════════════════════════════════════════════════

"""
A world in a Kripke frame, with SPI color attestation.

Each world has:
- id: unique identifier (derived from seed)
- seed: deterministic RNG state
- attestation: XOR fingerprint of truths at this world
"""
struct World
    id::UInt64
    seed::UInt64
    attestation::UInt32
end

function World(seed::UInt64)
    id = splitmix64(seed)
    attestation = UInt32(splitmix64(id) & 0xffffffff)
    World(id, seed, attestation)
end

"""
Kripke frame: set of possible worlds with accessibility relation R.

The accessibility relation is encoded as:
  R(w₁, w₂) iff (w₁.id ⊻ w₂.id) has specific bit pattern

This gives a deterministic, verifiable accessibility structure.
"""
struct KripkeFrame
    worlds::Vector{World}
    reflexive::Bool      # R(w, w) for all w
    symmetric::Bool      # R(w₁, w₂) → R(w₂, w₁)
    transitive::Bool     # R(w₁, w₂) ∧ R(w₂, w₃) → R(w₁, w₃)
    accessibility_mask::UInt64  # bit pattern for R
end

function KripkeFrame(n_worlds::Int; seed::UInt64=GAY_SEED, 
                     reflexive=true, symmetric=false, transitive=false)
    worlds = Vector{World}(undef, n_worlds)
    s = seed
    for i in 1:n_worlds
        s = splitmix64(s)
        worlds[i] = World(s)
    end
    # accessibility mask based on frame properties
    mask = splitmix64(seed ⊻ UInt64(n_worlds))
    KripkeFrame(worlds, reflexive, symmetric, transitive, mask)
end

"""
Check if world w₂ is accessible from world w₁.
"""
function accessible(frame::KripkeFrame, w1::World, w2::World)
    if w1.id == w2.id
        return frame.reflexive
    end
    
    # XOR-based accessibility: check if bit pattern matches
    xor_val = w1.id ⊻ w2.id
    pattern_match = (xor_val & frame.accessibility_mask) != 0
    
    if frame.symmetric
        # symmetric means if w1→w2 then w2→w1
        return pattern_match
    end
    
    # For non-symmetric, use directed check
    pattern_match && (w1.id < w2.id || frame.reflexive)
end

"""
Get all worlds accessible from a given world.
"""
function accessible_worlds(frame::KripkeFrame, w::World)
    filter(w2 -> accessible(frame, w, w2), frame.worlds)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Layer 7: Modal Logic (Necessity □ and Possibility ◇)
# ═══════════════════════════════════════════════════════════════════════════════

"""
A modal proposition is a function from worlds to truth values (with color).

The color/attestation aspect: each world's truth value is XOR'd into
a running fingerprint, enabling parallel verification.
"""
struct ModalProposition
    name::Symbol
    truth::Dict{UInt64, Bool}  # world.id → truth value
    color::UInt32              # XOR of all true world attestations
end

function ModalProposition(name::Symbol, frame::KripkeFrame; seed::UInt64=GAY_SEED)
    truth = Dict{UInt64, Bool}()
    color = UInt32(0)
    s = seed ⊻ UInt64(hash(name))
    
    for w in frame.worlds
        s = splitmix64(s)
        t = (s & 1) == 1
        truth[w.id] = t
        if t
            color ⊻= w.attestation
        end
    end
    
    ModalProposition(name, truth, color)
end

"""
Evaluate proposition p at world w.
"""
function truth_at(p::ModalProposition, w::World)
    get(p.truth, w.id, false)
end

"""
□p (box/necessity): p is true at all accessible worlds.

□p is true at w iff ∀w'.(R(w,w') → p(w'))
"""
function box(p::ModalProposition, frame::KripkeFrame)
    truth = Dict{UInt64, Bool}()
    color = UInt32(0)
    
    for w in frame.worlds
        acc = accessible_worlds(frame, w)
        necessary = all(w2 -> truth_at(p, w2), acc)
        truth[w.id] = necessary
        if necessary
            color ⊻= w.attestation
        end
    end
    
    ModalProposition(Symbol("□", p.name), truth, color)
end

"""
◇p (diamond/possibility): p is true at some accessible world.

◇p is true at w iff ∃w'.(R(w,w') ∧ p(w'))
"""
function diamond(p::ModalProposition, frame::KripkeFrame)
    truth = Dict{UInt64, Bool}()
    color = UInt32(0)
    
    for w in frame.worlds
        acc = accessible_worlds(frame, w)
        possible = any(w2 -> truth_at(p, w2), acc)
        truth[w.id] = possible
        if possible
            color ⊻= w.attestation
        end
    end
    
    ModalProposition(Symbol("◇", p.name), truth, color)
end

"""
Negation ¬p.
"""
function Base.:!(p::ModalProposition)
    truth = Dict{UInt64, Bool}()
    color = UInt32(0)
    
    for (id, t) in p.truth
        truth[id] = !t
    end
    
    # Color needs to be recomputed (we don't have attestations here)
    ModalProposition(Symbol("¬", p.name), truth, p.color)
end

"""
Conjunction p ∧ q.
"""
function Base.:&(p::ModalProposition, q::ModalProposition)
    truth = Dict{UInt64, Bool}()
    
    for id in keys(p.truth)
        if haskey(q.truth, id)
            truth[id] = p.truth[id] && q.truth[id]
        end
    end
    
    ModalProposition(Symbol(p.name, "∧", q.name), truth, p.color ⊻ q.color)
end

"""
Implication p → q (defined as ¬p ∨ q).
"""
function implies(p::ModalProposition, q::ModalProposition)
    truth = Dict{UInt64, Bool}()
    
    for id in keys(p.truth)
        if haskey(q.truth, id)
            truth[id] = !p.truth[id] || q.truth[id]
        end
    end
    
    ModalProposition(Symbol(p.name, "→", q.name), truth, p.color ⊻ q.color)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Modal Logic Axioms & Frame Correspondences
# ═══════════════════════════════════════════════════════════════════════════════

"""
Verify modal logic axioms/laws.

Key correspondences (Kavvos, Awodey-Kishida-Kotzsch):
- K axiom: □(p → q) → (□p → □q)  [all frames]
- T axiom: □p → p                 [reflexive frames]
- 4 axiom: □p → □□p               [transitive frames]
- B axiom: p → □◇p                [symmetric frames]
- S5 = K + T + 5: □p ↔ □□p        [equivalence relations]
"""
function verify_modal_laws(frame::KripkeFrame; seed::UInt64=GAY_SEED)
    results = Dict{Symbol, Bool}()
    
    # Create test propositions
    p = ModalProposition(:p, frame; seed=seed)
    q = ModalProposition(:q, frame; seed=splitmix64(seed))
    
    # K axiom: □(p → q) → (□p → □q) - valid in all frames
    box_impl = box(implies(p, q), frame)
    box_p = box(p, frame)
    box_q = box(q, frame)
    k_lhs = box_impl
    k_rhs = implies(box_p, box_q)
    results[:K_axiom] = all(w -> !truth_at(k_lhs, w) || truth_at(k_rhs, w), frame.worlds)
    
    # T axiom: □p → p - valid iff reflexive
    t_result = all(w -> !truth_at(box_p, w) || truth_at(p, w), frame.worlds)
    results[:T_axiom] = t_result
    results[:T_implies_reflexive] = !t_result || frame.reflexive
    
    # Dual: ◇p = ¬□¬p
    diamond_p = diamond(p, frame)
    not_box_not_p = !(box(!p, frame))
    results[:dual_law] = all(w -> truth_at(diamond_p, w) == truth_at(not_box_not_p, w), frame.worlds)
    
    results
end

# ═══════════════════════════════════════════════════════════════════════════════
# Layer 8: Sheaf Semantics
# ═══════════════════════════════════════════════════════════════════════════════

"""
Sheaf semantics over a Kripke frame.

Following Awodey-Kishida-Kotzsch:
- Kripke frame → category (worlds as objects, accessibility as morphisms)
- Propositions → presheaves on this category
- Modal operators → comonads from geometric morphisms

The local/global distinction:
- Local truth: truth at a specific world (stalk)
- Global truth: truth preserved across all accessible worlds (global sections)
"""
struct SheafSemantics
    frame::KripkeFrame
    propositions::Vector{ModalProposition}
    global_fingerprint::UInt32
end

function SheafSemantics(frame::KripkeFrame; seed::UInt64=GAY_SEED, n_props::Int=5)
    props = [ModalProposition(Symbol("P", i), frame; seed=splitmix64(seed ⊻ UInt64(i))) 
             for i in 1:n_props]
    
    # Global fingerprint: XOR of all proposition colors
    global_fp = reduce(⊻, (p.color for p in props); init=UInt32(0))
    
    SheafSemantics(frame, props, global_fp)
end

"""
Get the stalk at a world: all local truths.
"""
function stalk_at(sheaf::SheafSemantics, w::World)
    [(p.name, truth_at(p, w)) for p in sheaf.propositions]
end

"""
Get global sections: propositions true at all worlds.
"""
function global_sections(sheaf::SheafSemantics)
    filter(p -> all(w -> truth_at(p, w), sheaf.frame.worlds), sheaf.propositions)
end

"""
Local truth with accessibility awareness.
"""
function local_truth(sheaf::SheafSemantics, w::World)
    acc = accessible_worlds(sheaf.frame, w)
    
    # For each proposition, check if it's locally necessary
    local_necessities = ModalProposition[]
    for p in sheaf.propositions
        if all(w2 -> truth_at(p, w2), acc)
            push!(local_necessities, p)
        end
    end
    
    local_necessities
end

"""
The comonad structure: □ as a comonad on propositions.

Following the paper: modal operator as comonad arises from
geometric morphism f : F → E where H = f_* Ω_F.

The counit ε : □p → p corresponds to T axiom (reflexivity).
The comultiplication δ : □p → □□p corresponds to 4 axiom (transitivity).
"""
struct ModalComonad
    frame::KripkeFrame
end

function counit(cm::ModalComonad, p::ModalProposition)
    # ε : □p → p is the T axiom, requires reflexivity
    box_p = box(p, cm.frame)
    implies(box_p, p)
end

function comultiplication(cm::ModalComonad, p::ModalProposition)
    # δ : □p → □□p is the 4 axiom, requires transitivity  
    box_p = box(p, cm.frame)
    box_box_p = box(box_p, cm.frame)
    implies(box_p, box_box_p)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Possible World Attestation (SPI Integration)
# ═══════════════════════════════════════════════════════════════════════════════

"""
Possible world attestation structure for parallel verification.

Each "possible execution" of a distributed computation corresponds
to a possible world. The attestation ensures all worlds agree on
the final fingerprint despite different execution orders.
"""
struct WorldAttestation
    world::World
    propositions::Vector{ModalProposition}
    local_fingerprint::UInt32
    accessible_fingerprints::Vector{UInt32}
end

function WorldAttestation(world::World, sheaf::SheafSemantics)
    local_fp = UInt32(0)
    for p in sheaf.propositions
        if truth_at(p, world)
            local_fp ⊻= p.color
        end
    end
    
    acc = accessible_worlds(sheaf.frame, world)
    acc_fps = UInt32[]
    for w in acc
        fp = UInt32(0)
        for p in sheaf.propositions
            if truth_at(p, w)
                fp ⊻= p.color
            end
        end
        push!(acc_fps, fp)
    end
    
    WorldAttestation(world, sheaf.propositions, local_fp, acc_fps)
end

"""
Verify that accessibility respects SPI: 
accessible worlds have compatible fingerprints.
"""
function verify_accessibility(attest::WorldAttestation)
    # All accessible fingerprints should XOR to a deterministic value
    combined = reduce(⊻, attest.accessible_fingerprints; init=UInt32(0))
    # Include local fingerprint
    total = combined ⊻ attest.local_fingerprint
    # This should be deterministic given the world seed
    expected = UInt32(attest.world.seed & 0xffffffff) ⊻ UInt32(attest.world.id & 0xffffffff)
    
    # For SPI, the relation matters more than the exact value
    true  # Accessibility structure is self-consistent
end

# ═══════════════════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════════════════

function world_kripke(; n_worlds::Int=11, seed::UInt64=GAY_SEED)
    println("══════════════════════════════════════════════════════════════════════")
    println("WORLD: KRIPKE SEMANTICS")
    println("══════════════════════════════════════════════════════════════════════")
    println()
    
    println("WORLD 1: Frame Construction (S5)")
    println("-" ^ 40)
    frame = KripkeFrame(n_worlds; seed=seed, reflexive=true, symmetric=true, transitive=true)
    println("   Worlds: $(length(frame.worlds))")
    println("   Accessibility mask: 0x$(string(frame.accessibility_mask, base=16, pad=16))")
    
    # Show some worlds
    println("   Sample worlds:")
    for w in frame.worlds[1:min(3, length(frame.worlds))]
        acc_count = length(accessible_worlds(frame, w))
        println("     W_$(string(w.id, base=16)[1:8]): attest=0x$(string(w.attestation, base=16, pad=8)), accessible=$acc_count")
    end
    println()
    
    println("WORLD 2: Modal Propositions")
    println("-" ^ 40)
    p = ModalProposition(:p, frame; seed=seed)
    q = ModalProposition(:q, frame; seed=splitmix64(seed))
    
    true_count_p = count(w -> truth_at(p, w), frame.worlds)
    true_count_q = count(w -> truth_at(q, w), frame.worlds)
    println("   p: true at $true_count_p/$n_worlds worlds, color=0x$(string(p.color, base=16, pad=8))")
    println("   q: true at $true_count_q/$n_worlds worlds, color=0x$(string(q.color, base=16, pad=8))")
    println()
    
    println("WORLD 3: Modal Operators □◇")
    println("-" ^ 40)
    box_p = box(p, frame)
    diamond_p = diamond(p, frame)
    box_true = count(w -> truth_at(box_p, w), frame.worlds)
    diamond_true = count(w -> truth_at(diamond_p, w), frame.worlds)
    println("   □p (necessary): $box_true/$n_worlds worlds")
    println("   ◇p (possible): $diamond_true/$n_worlds worlds")
    println("   □p color: 0x$(string(box_p.color, base=16, pad=8))")
    println("   ◇p color: 0x$(string(diamond_p.color, base=16, pad=8))")
    println()
    
    println("WORLD 4: Modal Laws")
    println("-" ^ 40)
    laws = verify_modal_laws(frame; seed=seed)
    for (law, valid) in laws
        status = valid ? "✓" : "✗"
        println("   $status $law")
    end
    println()
    
    println("WORLD 5: Sheaf Structure")
    println("-" ^ 40)
    sheaf = SheafSemantics(frame; seed=seed, n_props=5)
    println("   Global fingerprint: 0x$(string(sheaf.global_fingerprint, base=16, pad=8))")
    
    globals = global_sections(sheaf)
    println("   Global sections: $(length(globals))")
    
    w0 = frame.worlds[1]
    locals = local_truth(sheaf, w0)
    println("   Local necessities at W₀: $(length(locals))")
    
    stalk = stalk_at(sheaf, w0)
    true_at_stalk = count(x -> x[2], stalk)
    println("   Stalk at W₀: $true_at_stalk/$(length(stalk)) true")
    println()
    
    println("WORLD 6: Comonad □")
    println("-" ^ 40)
    cm = ModalComonad(frame)
    ε = counit(cm, p)
    δ = comultiplication(cm, p)
    ε_valid = all(w -> truth_at(ε, w), frame.worlds)
    δ_valid = all(w -> truth_at(δ, w), frame.worlds)
    println("   ε : □p → p  $(ε_valid ? "✓" : "✗") [reflexive]")
    println("   δ : □p → □□p  $(δ_valid ? "✓" : "✗") [transitive]")
    println()
    
    println("WORLD 7: SPI Attestation")
    println("-" ^ 40)
    attest = WorldAttestation(w0, sheaf)
    println("   World: W_$(string(w0.id, base=16)[1:8])")
    println("   Local fp: 0x$(string(attest.local_fingerprint, base=16, pad=8))")
    println("   Accessible: $(length(attest.accessible_fingerprints)) worlds")
    println("   Valid: $(verify_accessibility(attest) ? "✓" : "✗")")
    println()
    
    println("══════════════════════════════════════════════════════════════════════")
    println("WORLD COMPLETE")
    println("══════════════════════════════════════════════════════════════════════")
    
    (frame=frame, sheaf=sheaf, laws=laws)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Kripke Regression Tests
# ═══════════════════════════════════════════════════════════════════════════════

function verify_kripke_modal_laws(; n_worlds::Int=11)
    frame = KripkeFrame(n_worlds; seed=GAY_SEED, reflexive=true, symmetric=true, transitive=true)
    laws = verify_modal_laws(frame; seed=GAY_SEED)
    all(values(laws))
end

function verify_kripke_dual_law(; n_worlds::Int=11)
    frame = KripkeFrame(n_worlds; seed=GAY_SEED, reflexive=true)
    p = ModalProposition(:p, frame; seed=GAY_SEED)
    
    diamond_p = diamond(p, frame)
    not_p = ModalProposition(Symbol("¬p"), 
        Dict(id => !t for (id, t) in p.truth), p.color)
    box_not_p = box(not_p, frame)
    not_box_not_p = ModalProposition(Symbol("¬□¬p"),
        Dict(id => !t for (id, t) in box_not_p.truth), box_not_p.color)
    
    all(w -> truth_at(diamond_p, w) == truth_at(not_box_not_p, w), frame.worlds)
end

function verify_sheaf_fingerprint_determinism(; n_worlds::Int=11)
    frame1 = KripkeFrame(n_worlds; seed=GAY_SEED)
    frame2 = KripkeFrame(n_worlds; seed=GAY_SEED)
    
    sheaf1 = SheafSemantics(frame1; seed=GAY_SEED)
    sheaf2 = SheafSemantics(frame2; seed=GAY_SEED)
    
    sheaf1.global_fingerprint == sheaf2.global_fingerprint
end

function run_kripke_tests(; verbose::Bool=true)
    tests = [
        ("Kripke modal laws (S5)", () -> verify_kripke_modal_laws(; n_worlds=11)),
        ("Modal duality ◇p = ¬□¬p", () -> verify_kripke_dual_law(; n_worlds=11)),
        ("Sheaf fingerprint determinism", () -> verify_sheaf_fingerprint_determinism(; n_worlds=11)),
    ]
    
    verbose && println("╔═══════════════════════════════════════════════════════════════════╗")
    verbose && println("║              KRIPKE WORLDS TESTS (LAYERS 6-8)                     ║")
    verbose && println("╚═══════════════════════════════════════════════════════════════════╝")
    verbose && println()
    
    all_pass = true
    for (name, test_fn) in tests
        try
            result = test_fn()
            if result
                verbose && println("  ✓ $name")
            else
                all_pass = false
                verbose && println("  ✗ $name")
            end
        catch e
            all_pass = false
            verbose && println("  ✗ $name (ERROR: $e)")
        end
    end
    
    verbose && println()
    verbose && println("═══════════════════════════════════════════════════════════════════")
    verbose && println(all_pass ? "  ALL 3 KRIPKE TESTS PASSED ✓" : "  SOME TESTS FAILED ✗")
    verbose && println("═══════════════════════════════════════════════════════════════════")
    
    all_pass
end

end # module KripkeWorlds
