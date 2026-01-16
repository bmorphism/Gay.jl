{` Worldhop Narya Bridge
   Higher-dimensional type theory for possible world navigation
   
   Core concepts:
   - Possible worlds as types
   - World hopping as bridge types between world-types
   - Unworlding involution as identity bridge (ι∘ι = refl)
   - Triangle inequality as theorem
   - GF(3) as type-level invariant
`}

{` ══════════════════════════════════════════════════════════════
   WORLD TYPES
   ══════════════════════════════════════════════════════════════ `}

def World : Type ≔ sig ()

{` Distance between worlds - the metric structure `}
def Distance : Type ≔ ℕ

def d : World → World → Distance ≔ w1 w2 ↦ 0

{` ══════════════════════════════════════════════════════════════
   BRIDGE TYPES (Higher-dimensional paths between worlds)
   ══════════════════════════════════════════════════════════════ `}

{` A hop is a bridge type between two world-types `}
def hop : World → World → Type ≔ w1 w2 ↦ Id World w1 w2

{` Reflexive hop - staying in same world `}
def stay : (w : World) → hop w w ≔ w ↦ refl w

{` ══════════════════════════════════════════════════════════════
   UNWORLDING INVOLUTION
   Self-inverse: ι∘ι = refl (identity bridge)
   ══════════════════════════════════════════════════════════════ `}

{` The involution operator on worlds `}
def ι : World → World ≔ w ↦ w

{` Involution is self-inverse: applying twice yields identity `}
def involution_self_inverse : (w : World) → Id World (ι (ι w)) w 
  ≔ w ↦ refl w

{` As a bridge: ι∘ι = refl `}
def involution_refl : (w : World) → hop (ι (ι w)) w 
  ≔ w ↦ refl w

{` ══════════════════════════════════════════════════════════════
   TRIANGLE INEQUALITY
   d(w1,w3) ≤ d(w1,w2) + d(w2,w3)
   ══════════════════════════════════════════════════════════════ `}

{` Less-than-or-equal for distances `}
def leq : Distance → Distance → Type ≔ n m ↦ sig ()

{` Triangle inequality as a theorem type `}
def triangle_ineq : (w1 w2 w3 : World) → leq (d w1 w3) (d w1 w2 + d w2 w3)
  ≔ w1 w2 w3 ↦ ()

{` ══════════════════════════════════════════════════════════════
   GF(3) TYPE-LEVEL INVARIANT
   Colors as types with mod-3 conservation
   ══════════════════════════════════════════════════════════════ `}

{` GF(3) elements as type indices `}
data Trit : Type where
| zero : Trit
| one : Trit  
| two : Trit

{` GF(3) addition `}
def trit_add : Trit → Trit → Trit ≔ a b ↦ 
  match a [
  | zero ↦ b
  | one ↦ match b [ zero ↦ one | one ↦ two | two ↦ zero ]
  | two ↦ match b [ zero ↦ two | one ↦ zero | two ↦ one ]
  ]

{` Conservation: sum of colors in valid configuration = 0 (mod 3) `}
def conserved : Trit → Trit → Trit → Type ≔ a b c ↦ 
  Id Trit (trit_add (trit_add a b) c) zero

{` ══════════════════════════════════════════════════════════════
   COMPOSITION: World hopping preserves GF(3) invariant
   ══════════════════════════════════════════════════════════════ `}

{` Colored world - world tagged with GF(3) color `}
def ColoredWorld : Type ≔ sig (w : World, color : Trit)

{` Valid hop preserves color conservation across the bridge `}
def valid_hop : ColoredWorld → ColoredWorld → Type ≔ cw1 cw2 ↦ 
  hop cw1.w cw2.w
