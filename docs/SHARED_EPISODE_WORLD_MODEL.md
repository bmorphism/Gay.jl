# Shared Episode World Model

Gay.jl can help us converge on a shared world model by making episodes
replayable, colored, trit-classified, and cheap to verify.

The point is not to reconstruct the full ancestry of every person, machine,
simulation, sensor, or embodied participant. The simpler path is to keep a
canonical ledger of bounded shared episodes: what we observed together, what
we tried, what pushed back, what changed, and what failed quickly enough to
spare the next open system from resource exhaustion.

## Core Move

Use a `SharedEpisodeWorld` as the common substrate.

An episode is a bounded "now" with:

- a time coordinate in integer glimpses / trit-ticks / flicks,
- participants and their trust boundaries,
- observations from each participant or sensor,
- forward play actions,
- backward coplay feedback,
- neutral witness records,
- accepted or rejected world updates,
- an SPI fingerprint and deterministic color.

This lets us learn together from interaction rather than from total origin
tracing. Concordia-like simulations, AI-town-like social worlds, firmware,
BCI streams, hueman play, and small embodied agents can all interoperate by
sharing episode records rather than adopting one another's whole ontology.

## Hueman Ideography

Use **hueman** when the participant role is color-bearing, embodied, and
socially legible rather than species-exclusive. A hueman participant is one
more world-compatible observer/actor in the ledger: colored by Gay.jl,
bounded by trust and mortality, and held to the same `world_` standards as
machines, simulations, sensors, and other embodied agents.

The naming rule is part of the ideology: no throwaway display identity when a
shared world can be returned, merged, fingerprinted, and recalled. Ideography
should keep the substrate visible (`retinal`, `mosaic`, `ganglion`, `episode`)
and the operation composable (`dither`, `refresh`, `sync`, `merge`).

## Existing Gay.jl Hooks

Gay.jl already has the pieces this needs:

- `src/trit_tick.jl` gives the epoch-1 time grid:
  `EPOCH_1_HZ = 141_120_000`, `FLICKS_PER_TICK = 5`, and GF(3) phases.
- `src/splittable.jl` gives `GAY_SEED = 1069`, SplitMix64 constants,
  `color_at`, `color_at(TritTick)`, and SPI-style deterministic access.
- `src/entropy_sources.jl` gives `ColoredTick`: physical measurements joined
  to trit time, confidence, entropy, and source mortality.
- `src/obligation_clearing.jl` treats conservation and order-independent XOR
  fingerprints as verification structure.
- `docs/WORLD_PATTERN.md` already states the runtime shape: persistent
  `world_` builders with `length`, `merge`, and `fingerprint`.

zig-syrup gives the sibling transport vocabulary:

- `src/glimpse.zig` names the same epoch-1 quantum as a glimpse:
  `141_120_000` glimpses per second, `1 glimpse = 5 flicks`.
- `src/trit_tick.zig` preserves the older trit-tick name.
- `src/splitmix_trit.zig` assigns GF(3) roles:
  validator `-1`, coordinator `0`, generator `+1`.
- `src/gf3_palette.zig` maps five trits to a 243-color palette and checksum.

## GF(3) Episode Semantics

The episode loop is:

```text
observe together -> play -> coplay -> witness -> update or die -> fingerprint
```

Play is the forward generative move, `+1`.

Coplay is backward feedback: correction, constraint, refusal, reward, error,
or environmental return, `-1`.

Witness is the neutral closure that makes the episode replayable and shareable,
`0`.

The useful invariant is not that every participant agrees about everything.
The useful invariant is that a committed update cites a shared trace and keeps
the GF(3) pressure from drifting: generation, validation, and witness all have
to show up somewhere in the episode economy.

## Verification Boundary

At the boundary where machines need to agree, use integers and canonical bytes:

- time as glimpses / trit-ticks / flicks, not floats,
- episode records as Syrup-compatible canonical data,
- colors from Gay.jl seed/index/tick functions,
- fingerprints from canonical episode records,
- merges as order-independent reductions over episode fingerprints.

The artifact version of this model is stored at:

```text
artifacts/shared_episode_world_model.json
```

That artifact is the handoff point for a later Julia `world_shared_episode_model`
implementation.
