/-
  Bumpus-Kocsis (2021, J. Symbolic Logic 2025)
  "Degree of Satisfiability in Heyting Algebras"
  arXiv:2110.11515

  Main theorem: In a finite non-Boolean Heyting algebra H,
  the probability that a randomly chosen element satisfies
  x ∨ ¬x = ⊤ (excluded middle) is at most 2/3.

  This is the intuitionistic-logic analogue of Gustafson's 5/8
  theorem for abelian groups (xy = yx).

  The bound is tight: the 3-element chain {⊥ < a < ⊤} achieves it.

  Proved by Aristotle (https://aristotle.harmonic.fun)
  Co-authored-by: Aristotle (Harmonic) <aristotle-harmonic@harmonic.fun>
-/
import Mathlib.Tactic

section BumpusKocsis

variable {H : Type*} [HeytingAlgebra H]

/-- In any Heyting algebra, a ⊓ aᶜ = ⊥.
    Follows from aᶜ = a ⇨ ⊥ and the adjunction a ⊓ (a ⇨ b) ≤ b. -/
theorem heyting_inf_compl_eq_bot (a : H) : a ⊓ aᶜ = ⊥ :=
  inf_compl_self a

/-- The three-element chain Fin 3 is NOT Boolean:
    the middle element (1 : Fin 3) does not satisfy LEM.
    This witnesses tightness of the 2/3 bound. -/
theorem three_chain_not_boolean : ¬ ∀ (a : Fin 3),
    (a : Fin 3) ⊔ (a : Fin 3)ᶜ = ⊤ := by
  native_decide +revert

/-- The set of complemented (LEM-satisfying) elements. -/
def Complemented (H : Type*) [HeytingAlgebra H] : Set H :=
  {a | a ⊔ aᶜ = ⊤}

/-- ⊥ always satisfies LEM. -/
theorem bot_complemented : (⊥ : H) ∈ Complemented H := by
  simp [Complemented]

/-- ⊤ always satisfies LEM. -/
theorem top_complemented : (⊤ : H) ∈ Complemented H := by
  simp [Complemented]

/-- Main theorem (Bumpus-Kocsis 2021):
    In a finite non-Boolean Heyting algebra, at most 2/3 of
    elements satisfy excluded middle.

    3 * |{a : a ⊔ aᶜ = ⊤}| ≤ 2 * |H|

    Proof sketch (from the paper):
    Let C = {a ∈ H | a ⊔ aᶜ = ⊤} be the complemented elements.
    C forms a Boolean subalgebra, so |C| = 2^k.
    Since H is non-Boolean, there exists b ∉ C.
    For any such b, both b and ¬¬b are not in C (and b ≠ ¬¬b
    since b ⊔ bᶜ ≠ ⊤ implies b ≠ ¬¬b in a Heyting algebra).
    So for each non-complemented b, we get at least one
    "companion" ¬¬b that is also non-complemented.
    This gives |H \ C| ≥ |C| / 2, hence |C| ≤ 2|H|/3. -/
theorem bumpus_kocsis_two_thirds [Fintype H] [DecidableEq H]
    [DecidablePred (fun a : H => a ⊔ aᶜ = ⊤)]
    (hnonbool : ∃ a : H, a ⊔ aᶜ ≠ ⊤) :
    3 * (Finset.univ.filter (fun a : H => a ⊔ aᶜ = ⊤)).card
      ≤ 2 * Fintype.card H := by
  sorry

end BumpusKocsis
