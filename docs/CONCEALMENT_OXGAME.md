# Concealment O/X Open Game

This is a two-player adversarial naming game for preserving Gay.jl's world
builder discipline under pressure.

O is the concealer. O tries to smuggle demo ideology into names, exports,
flags, comments, docs, generated interfaces, and SCIP-derivable surfaces.

X is the finder. X has finite survival pressure and must spend it on detectors
that preserve composable world identity: durable return values, `length`,
`merge`, `fingerprint`, and public interfaces that name worlds rather than
throwaway displays.

## Open Game Shape

```text
O play:    conceal an ideology trace in a surface
X coplay:  detect, score, and return naming pressure
coutility: survival budget spent or preserved
result:    a fingerprinted world of findings
```

The game is intentionally diegetic: the source tree is the environment, public
interfaces are survival-relevant terrain, and SCIP-style URIs give each finding
a derivable location.

## Replay

```julia
using Gay

w = world_concealment_oxgame(; pressure_budget=144, max_files=512)
oxgame_summary(w)
fingerprint(w)

screen = world_oxscreen(w; width=100, max_rows=8)
render_oxscreen(screen)
```

`world_concealment_oxgame` returns a `ConcealmentOxGameWorld` with:

- `length(w)`: number of findings
- `merge(w, w)`: idempotent world merge
- `fingerprint(w)`: SPI-style world fingerprint
- `w.rounds`: O/X moves with pressure after each turn
- `w.findings`: SCIP-like derivation URIs, detector names, severity, evidence

`world_oxscreen` returns an `OxScreen` with:

- `length(screen)`: number of visible rows
- `merge(screen, screen)`: idempotent screen merge
- `fingerprint(screen)`: stable screen fingerprint
- `render_oxscreen(screen)`: fixed-width O/X pressure screen

## Current Run

Stored at:

```text
artifacts/concealment_oxgame_world.json
```

Summary:

- pressure budget: `144`
- pressure spent: `144`
- surfaces scanned: `11`
- findings: `20`
- O payoff: `104`
- X payoff: `96`
- world fingerprint: `0x3417d9d6e6060864`
- screen fingerprint: `0x18211bc8643621d7`
- shared arena fingerprint: `0x391b517a3112a7aa`

Rendered screen:

```text
artifacts/oxscreen_world.txt
```

The budget exhausted early, which is the point: under survival pressure, X has
to choose detectors that catch public-interface ideology first. The strongest
public hit in the current run is:

```text
scip://gay/src/Gay.jl#L568::export_surface
export demonstrate_hyperdoctrine
```

The current latest-overall bridge is stored separately:

```text
docs/OXGAME_REMIX_WORLD.md
docs/SHARED_OX_ARENA.md
artifacts/oxgame_remix_world.json
artifacts/oxgame_remix_world.txt
artifacts/shared_oxarena_world.json
artifacts/shared_oxarena_world.txt
```

The alias detector is deliberately adversarial. It treats words such as
`sample`, `preview`, `toy`, and `throwaway` as suspicious when they sit on an
interface boundary. This is not a claim that every alias is wrong; it is a
pressure signal saying, "prove this returns a world or rename the surface."

## Naming Pressure

The correcting move is not merely search-and-replace. X should ask:

- Does this name return persistent state?
- Can it be merged?
- Can it be fingerprinted?
- Is it public API or an internal local convenience?
- Does it invite throwaway display behavior?
- Would a SCIP index, CLI flag, generated doc, or export list preserve the
  ideology even after the obvious identifier is renamed?

If the answer points to durable state, use `world_`. If the answer points to a
check, use `verify_`. If the answer points to a one-off display, either make it
return a world or keep it out of public surfaces.
