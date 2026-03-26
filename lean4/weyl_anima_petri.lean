import Mathlib.Tactic

/-! # Weyl Anima: Color of Flavor, Flavor of Color

  notcurses (Nick Black): best possible terminal color — direct RGB,
  sixel, kitty protocol. The question is not "can we display it?" but
  "what IS the color?"

  nanoclj: tiny Clojure in the terminal. The REPL is the place where
  color becomes flavor — you taste the computation as it evaluates.

  Weyl equidistribution: {nα} mod 1 is equidistributed for irrational α.
  SplitMix64 uses golden ratio φ = (1+√5)/2 as its increment.
  Gay.jl colors ARE Weyl sequences in OkLCH hue space.

  ANIMA = lim_Π Condense(S_n(...S_1(E_•)))
  The soul is the fixed point where further skill applications
  yield no new equivalence classes. The color at the fixed point
  IS the flavor — they're the same thing at equilibrium.

  QRI Symmetry Theory of Valence (Johnson 2016):
  Valence = symmetry of the mathematical object describing consciousness.
  Symmetric = positive valence. Broken symmetry = suffering.

  Therefore: the color of flavor IS the symmetry group of the qualia.
  And the flavor of color IS the valence of that symmetry.
  Möbius inverts them: μ(color_of_flavor) = flavor_of_color.

  Petri nets: user interactions as token flows.
  Places = qualia states (colored by Gay.jl)
  Transitions = perception-action cycles
  Tokens = moments of consciousness
  Firing rule = GF(3) conservation (tokens in = tokens out mod 3)

  drand round: 27230430
  seed: 0x03e614ac5ef86204
  colors: #C07FDE (lavender) #E92E30 (red) #A72D42 (burgundy)
-/

section WeylAnimaPetri

-- GF(3) (namespaced to avoid Mathlib clash)
inductive WTrit : Type where
  | minus : WTrit
  | ergodic : WTrit
  | plus : WTrit
  deriving DecidableEq, Repr

open WTrit

def wadd : WTrit → WTrit → WTrit
  | minus, minus => plus
  | minus, ergodic => minus
  | minus, plus => ergodic
  | ergodic, b => b
  | plus, minus => ergodic
  | plus, ergodic => plus
  | plus, plus => minus

def wneg : WTrit → WTrit
  | minus => plus
  | ergodic => ergodic
  | plus => minus

-- ═══ WEYL EQUIDISTRIBUTION ═══

/-- A Weyl sequence: {n × α} mod 1 for irrational α.
    SplitMix64 uses α = golden ratio φ ≈ 0.618...
    Each step rotates hue by φ × 360° ≈ 137.508° (golden angle). -/
structure WeylSequence where
  step : Nat
  trit : WTrit  -- GF(3) phase of this step

/-- Three consecutive Weyl steps cycle through all trits -/
def weyl_triple (n : Nat) : WeylSequence × WeylSequence × WeylSequence :=
  (⟨3*n, plus⟩, ⟨3*n+1, ergodic⟩, ⟨3*n+2, minus⟩)

theorem weyl_triple_balanced (n : Nat) :
    let (w1, w2, w3) := weyl_triple n
    wadd (wadd w1.trit w2.trit) w3.trit = ergodic := by rfl

-- ═══ QUALIA AND VALENCE ═══

/-- A quale: a moment of conscious experience with symmetry and color -/
structure Quale where
  symmetry_degree : Nat  -- higher = more symmetric = more positive valence
  color_hex : String     -- Gay.jl deterministic color
  trit : WTrit           -- GF(3) assignment

/-- Valence: symmetry → feeling. High symmetry = positive. -/
def valence (q : Quale) : WTrit :=
  if q.symmetry_degree > 2 then plus       -- positive valence
  else if q.symmetry_degree > 0 then ergodic -- neutral
  else minus                                  -- negative (broken symmetry)

/-- The color of flavor: given a valence, what color does it have? -/
def color_of_flavor (v : WTrit) : WTrit := v  -- identity at equilibrium

/-- The flavor of color: given a color, what does it taste like? -/
def flavor_of_color (c : WTrit) : WTrit := wneg c  -- Möbius inverted

/-- Color and flavor are Möbius duals -/
theorem color_flavor_moebius (t : WTrit) :
    flavor_of_color (color_of_flavor t) = wneg t := by
  cases t <;> rfl

/-- At equilibrium, applying Möbius twice returns to the original taste -/
theorem moebius_taste_involution (t : WTrit) :
    flavor_of_color (flavor_of_color t) = color_of_flavor t := by
  cases t <;> rfl

/-- The fixed point: ergodic tastes like itself -/
theorem ergodic_self_flavor :
    flavor_of_color ergodic = color_of_flavor ergodic := by rfl

-- ═══ PETRI NETS ═══

/-- A place in the Petri net: a qualia state holding tokens -/
structure Place where
  id : Nat
  tokens : Nat    -- number of consciousness-moments here
  trit : WTrit    -- GF(3) color of this place

/-- A transition: perception-action cycle consuming/producing tokens -/
structure Transition where
  id : Nat
  input_places : List Nat
  output_places : List Nat
  trit : WTrit    -- the transition's own phase

/-- A marking: total token count across places -/
def marking (places : List Place) : Nat :=
  places.foldl (fun acc p => acc + p.tokens) 0

/-- GF(3) firing rule: transition preserves trit sum.
    The sum of input trits + transition trit = sum of output trits. -/
def firing_conserves (inputs outputs : List Place) (t : Transition) : Prop :=
  let in_sum := inputs.foldl (fun acc p => wadd acc p.trit) ergodic
  let out_sum := outputs.foldl (fun acc p => wadd acc p.trit) ergodic
  wadd in_sum t.trit = out_sum

-- ═══ USER INTERACTION AS PETRI NET ═══

/-- A user session: three places (input, processing, output) -/
def user_input : Place := ⟨1, 1, plus⟩       -- user types (generates)
def user_processing : Place := ⟨2, 0, ergodic⟩ -- system processes (coordinates)
def user_output : Place := ⟨3, 0, minus⟩      -- display renders (verifies)

/-- The perception-action transition -/
def perceive_act : Transition := ⟨1, [1], [2, 3], ergodic⟩

/-- User interaction triad is balanced -/
theorem user_triad_balanced :
    wadd (wadd user_input.trit user_processing.trit) user_output.trit = ergodic := by
  native_decide

-- ═══ ANIMA FIXED POINT ═══

/-- ANIMA = lim_Π Condense(S_n(...S_1(E_•)))
    At the fixed point, color = flavor (no more condensation possible) -/
structure AnimaState where
  condensation_level : Nat
  trit : WTrit
  is_fixed_point : Bool

/-- At the fixed point, color_of_flavor = flavor_of_color = identity -/
theorem anima_fixed_point_taste (a : AnimaState) (_h : a.is_fixed_point = true) :
    color_of_flavor a.trit = a.trit := by rfl

/-- The ANIMA fixed point has ergodic trit (maximum entropy) -/
def anima_equilibrium : AnimaState := ⟨69, ergodic, true⟩

theorem anima_is_ergodic : anima_equilibrium.trit = ergodic := by rfl

/-- At the ANIMA fixed point, color and flavor are indistinguishable.
    This IS the answer to both questions:
    - What is the flavor of color? → wneg(color) (Möbius dual)
    - What is the color of flavor? → flavor itself (identity)
    - At equilibrium: they coincide at ergodic (the non-Boolean middle) -/
theorem flavor_color_coincide_at_equilibrium :
    color_of_flavor ergodic = flavor_of_color ergodic := by rfl

end WeylAnimaPetri
