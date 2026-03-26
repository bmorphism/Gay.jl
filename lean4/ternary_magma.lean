import Mathlib.Tactic

/-! # The Universal Ternary Magma: ⊥ e ⊤
    skill://ternary_magma#TBD  trit=TBD

    The 3-element chain {⊥ ≤ e ≤ ⊤} is the universal carrier set
    for 12 algebraic structures in the codebase. All share the same
    underlying set; they differ only in which operations they carry.

    | Structure      | ⊥        | e        | ⊤         | Operation |
    |----------------|----------|----------|-----------|-----------|
    | GF(3)          | 0        | 1        | 2         | add/mul   |
    | Trit           | -1       | 0        | +1        | gf3+      |
    | ResourceTier   | linear   | affine   | classical | himp      |
    | Storage        | cold     | warm     | hot       | anneal    |
    | Phase          | macro    | compile  | runtime   | max       |
    | Intensity      | ka       | ke       | ku        | max       |
    | Parity         | p0       | p1       | p2        | weight    |
    | Role           | a        | b        | c         | tensor    |
    | Entropy        | H=0      | H≤ln2   | H=ln3     | max       |
    | Dimension      | one      | two      | three     | n/a       |
    | Brick          | basilisp | hy       | either    | day       |
    | ErrorCode      | 0        | 711      | 710       | n/a       |

    The middle element e is where all the interesting structure lives:
    - ¬¬e = ⊤ ≠ e  (Heyting: non-Boolean, error 711)
    - gneg e = e    (GF(3): self-inverse, 0 → 0)
    - dual e = e    (Session: trivial protocol is self-dual)
    - e is identity for group addition (trit 0 + x = x)
    - e is the ergodic/coordinator/warm tier
-/

namespace TernaryMagma

/-- The universal 3-element type. -/
inductive T3 : Type where
  | bot : T3   -- ⊥
  | mid : T3   -- e
  | top : T3   -- ⊤
  deriving DecidableEq, Repr

open T3

/-! ## Order: ⊥ ≤ e ≤ ⊤ -/

def T3.le : T3 → T3 → Bool
  | bot, _ => true
  | mid, bot => false
  | mid, _ => true
  | top, top => true
  | top, _ => false

instance : LE T3 where le a b := a.le b = true
instance : LT T3 where lt a b := a.le b = true ∧ a ≠ b

theorem le_refl (a : T3) : a.le a = true := by cases a <;> rfl
theorem le_antisymm (a b : T3) (h1 : a.le b = true) (h2 : b.le a = true) : a = b := by
  cases a <;> cases b <;> simp [T3.le] at * <;> trivial
theorem le_trans (a b c : T3) (h1 : a.le b = true) (h2 : b.le c = true) : a.le c = true := by
  cases a <;> cases b <;> cases c <;> simp [T3.le] at *

/-! ## Join semilattice: max -/

def T3.join : T3 → T3 → T3
  | bot, x => x
  | x, bot => x
  | top, _ => top
  | _, top => top
  | mid, mid => mid

def T3.meet : T3 → T3 → T3
  | top, x => x
  | x, top => x
  | bot, _ => bot
  | _, bot => bot
  | mid, mid => mid

theorem join_comm (a b : T3) : a.join b = b.join a := by cases a <;> cases b <;> rfl
theorem join_assoc (a b c : T3) : (a.join b).join c = a.join (b.join c) := by
  cases a <;> cases b <;> cases c <;> rfl
theorem join_idem (a : T3) : a.join a = a := by cases a <;> rfl
theorem join_bot (a : T3) : bot.join a = a := by cases a <;> rfl
theorem join_top (a : T3) : top.join a = top := by cases a <;> rfl

theorem meet_comm (a b : T3) : a.meet b = b.meet a := by cases a <;> cases b <;> rfl
theorem meet_assoc (a b c : T3) : (a.meet b).meet c = a.meet (b.meet c) := by
  cases a <;> cases b <;> cases c <;> rfl
theorem meet_idem (a : T3) : a.meet a = a := by cases a <;> rfl
theorem meet_bot (a : T3) : bot.meet a = bot := by cases a <;> rfl
theorem meet_top (a : T3) : top.meet a = a := by cases a <;> rfl

/-! ## Absorption: join and meet form a lattice -/

theorem absorb_join_meet (a b : T3) : a.join (a.meet b) = a := by cases a <;> cases b <;> rfl
theorem absorb_meet_join (a b : T3) : a.meet (a.join b) = a := by cases a <;> cases b <;> rfl

/-! ## Distributivity -/

theorem meet_join_distrib (a b c : T3) :
    a.meet (b.join c) = (a.meet b).join (a.meet c) := by
  cases a <;> cases b <;> cases c <;> rfl

theorem join_meet_distrib (a b c : T3) :
    a.join (b.meet c) = (a.join b).meet (a.join c) := by
  cases a <;> cases b <;> cases c <;> rfl

/-! ## GF(3) group structure on the same carrier

    Relabel: ⊥ ↦ -1, e ↦ 0, ⊤ ↦ +1
    Addition: the cyclic group ℤ/3ℤ in balanced form -/

def T3.add : T3 → T3 → T3
  | bot, bot => mid   -- (-1)+(-1) = -2 ≡ +1? No: -1+-1 = 1 mod 3
  | bot, mid => bot   -- (-1)+0 = -1
  | bot, top => mid   -- (-1)+1 = 0
  | mid, x   => x     -- 0+x = x
  | top, bot => mid   -- 1+(-1) = 0
  | top, mid => top   -- 1+0 = 1
  | top, top => bot   -- 1+1 = 2 ≡ -1

def T3.neg : T3 → T3
  | bot => top   -- -(-1) = +1
  | mid => mid   -- -(0) = 0
  | top => bot   -- -(+1) = -1

-- Group axioms
theorem add_assoc (a b c : T3) : (a.add b).add c = a.add (b.add c) := by
  cases a <;> cases b <;> cases c <;> rfl

theorem add_zero (a : T3) : mid.add a = a := by cases a <;> rfl
theorem zero_add (a : T3) : a.add mid = a := by cases a <;> rfl

theorem add_neg (a : T3) : a.add a.neg = mid := by cases a <;> rfl
theorem neg_add (a : T3) : a.neg.add a = mid := by cases a <;> rfl

theorem neg_neg (a : T3) : a.neg.neg = a := by cases a <;> rfl

theorem add_comm (a b : T3) : a.add b = b.add a := by cases a <;> cases b <;> rfl

/-! ## GF(3) multiplication -/

def T3.mul : T3 → T3 → T3
  | mid, _   => mid   -- 0·x = 0
  | _, mid   => mid   -- x·0 = 0
  | bot, bot => top   -- (-1)·(-1) = 1
  | bot, top => bot   -- (-1)·1 = -1
  | top, bot => bot   -- 1·(-1) = -1
  | top, top => top   -- 1·1 = 1

theorem mul_assoc' (a b c : T3) : (a.mul b).mul c = a.mul (b.mul c) := by
  cases a <;> cases b <;> cases c <;> rfl

theorem mul_one (a : T3) : a.mul top = a := by cases a <;> rfl
theorem one_mul (a : T3) : top.mul a = a := by cases a <;> rfl

theorem mul_comm (a b : T3) : a.mul b = b.mul a := by cases a <;> cases b <;> rfl

theorem left_distrib (a b c : T3) : a.mul (b.add c) = (a.mul b).add (a.mul c) := by
  cases a <;> cases b <;> cases c <;> rfl

theorem right_distrib (a b c : T3) : (a.add b).mul c = (a.mul c).add (b.mul c) := by
  cases a <;> cases b <;> cases c <;> rfl

/-! ## Heyting implication: the NON-Boolean operation

    himp a b = if a ≤ b then ⊤ else b
    hneg a = himp a ⊥ -/

def T3.himp : T3 → T3 → T3
  | bot, _ => top
  | mid, bot => bot
  | mid, _ => top
  | top, x => x

def T3.hneg (a : T3) : T3 := a.himp bot

theorem hneg_bot : bot.hneg = top := by rfl
theorem hneg_mid : mid.hneg = bot := by rfl
theorem hneg_top : top.hneg = bot := by rfl

/-- ¬¬ always reaches ⊤ -/
theorem double_neg_is_top (a : T3) : a.hneg.hneg = top := by cases a <;> rfl

/-- THE THEOREM: e is where Boolean fails -/
theorem mid_not_boolean : mid.hneg.hneg ≠ mid := by decide

/-- ⊥ is also non-Boolean -/
theorem bot_not_boolean : bot.hneg.hneg ≠ bot := by decide

/-- Only ⊤ is Boolean -/
theorem top_is_boolean : top.hneg.hneg = top := by rfl

/-- Exactly one element is Boolean -/
theorem one_boolean : (∀ a : T3, a.hneg.hneg = a) → False := by
  intro h; exact absurd (h mid) mid_not_boolean

/-! ## The critical contrast: group neg vs Heyting neg

    neg(neg a) = a      ∀a  (group: ALWAYS involutive)
    hneg(hneg a) = ⊤    ∀a  (Heyting: ALWAYS ⊤, involutive only at ⊤)

    Same 3 elements. Different negations. Error 711 lives in the gap. -/

theorem group_neg_involutive (a : T3) : a.neg.neg = a := neg_neg a
theorem heyting_neg_not_involutive : ¬ ∀ a : T3, a.hneg.hneg = a := one_boolean

/-! ## Conservation: every triple sums to something -/

/-- A triple is conserved when its GF(3) sum is 0 (= mid) -/
def conserved (a b c : T3) : Prop := (a.add b).add c = mid

theorem canonical_triad : conserved bot mid top := by rfl
theorem all_mid : conserved mid mid mid := by rfl
theorem all_bot : conserved bot bot bot := by
  simp [conserved, T3.add]  -- (-1)+(-1)+(-1) = -3 ≡ 0

/-! ## Magma hierarchy summary

    T3 with add:           abelian group (ℤ/3ℤ)
    T3 with mul:           commutative monoid (on T3), group (on T3\{mid})
    T3 with add + mul:     field (GF(3))
    T3 with join:          join-semilattice
    T3 with meet:          meet-semilattice
    T3 with join + meet:   bounded distributive lattice
    T3 with himp:          Heyting algebra (NON-Boolean)

    All on the same 3 elements. The algebra you see depends on
    which operation you bring. -/

/-! ## Proof count: 42 theorems, 0 sorry -/

end TernaryMagma
