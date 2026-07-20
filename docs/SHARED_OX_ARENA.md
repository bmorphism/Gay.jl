# Shared O/X Arena

`src/concealment_oxgame.jl` now treats the O/X game as a shared
bisimulation arena, not merely as a source scan.

The conceptual move is:

```text
Optic(C)
  two-player bidirectional boundary
  play   : source state -> observation
  coplay : source state x finding -> naming pressure

Para(Optic(C))
  parameterized family of those optics
  parameters = strategy, context, certificate, accepted/rejected state
```

## Players

The shared arena uses three role positions, but supports two-player and mixed
subgames by selecting subsets.

```text
O  +1  spoiler / concealer
X  -1  duplicator / detector
W   0  witness / arena
```

The player trits sum to zero, so the arena itself is GF(3)-balanced. Two-player
O/X subgames are ordinary `Optic(C)` slices. Three-player O/X/W subgames are
`Para(Optic(C))` slices where W supplies the parameter context: certificate,
accepted world state, and coworld rejection boundary.

## Bisimulation Rule

The game is played as a bisimulation:

```text
1. O chooses a move in one lane, such as source code or a derivation URI.
2. X must answer in the paired lane.
3. W accepts the pair only if observation, detector, severity, pressure,
   color, and fingerprint are preserved.
```

If O moves in source, X answers in SCIP derivation. If O moves in derivation, X
answers in source or artifact replay. The shared arena is the quotient of all
lanes that survive this back-and-forth.

## Public API

```julia
using Gay

w = world_concealment_oxgame()
a = world_shared_oxarena(w)

shared_oxarena_summary(a)
render_shared_oxarena(a)
fingerprint(a)
merge(a, a)
```

The same builder also accepts an `OxgameRemixWorld`:

```julia
remix = world_oxgame_remix()
a = world_shared_oxarena(remix)
```

## Current Artifact

Stored at:

```text
artifacts/shared_oxarena_world.json
artifacts/shared_oxarena_world.txt
```

Current fingerprint:

```text
0x391b517a3112a7aa
```

Source world fingerprint:

```text
0x3417d9d6e6060864
```

## Interpretation

The shared arena gives us a better notion of "latest overall":

- `Optic(C)` is the two-player observable interface.
- `Para(Optic(C))` is the family of those interfaces under strategy and
  certificate parameters.
- Bisimulation is the survival condition: every move in one lane must be
  answered in another lane without losing the observation/payoff/fingerprint.
- The artifact is not an explanation outside the game; it is one of the lanes
  whose replay must remain bisimilar to source and SCIP-derived evidence.
