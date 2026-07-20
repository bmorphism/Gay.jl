# Oxgame Remix World

This is the latest-overall bridge between current Gay.jl and the available
`docs://oxgame` mirror.

The public GitHub repository name `plurigrid/oxgame` was checked on 2026-07-08
and was not resolvable from this environment over HTTPS, SSH, or `gh`. The
available source of oxgame work is the local documentation mirror:

```text
/Users/dietrich/worlds/docs-clone/oxgame
```

Its `manifest.json` identifies the mirror as:

```text
docs_uri: docs://oxgame
root: x/src/oxgame
generated_at: 2026-07-02T18:34:49Z
page_count: 166
```

## Remix

```julia
using Gay

w = world_oxgame_remix()
oxgame_remix_summary(w)
render_oxgame_remix(w)
```

`world_oxgame_remix` composes:

- the current Gay.jl checkout, `e3c403ad9873`;
- the local adversarial `ConcealmentOxGameWorld`;
- the `OxScreen` presentation layer;
- the shared bisimulation arena in `SharedOxArena`;
- top-scored `docs://oxgame` sources for lens/open-game/SCIP/world semantics;
- six GF(3)-balanced lanes from the oxgame convergence model.

## Current Artifact

Stored at:

```text
artifacts/oxgame_remix_world.json
artifacts/oxgame_remix_world.txt
artifacts/shared_oxarena_world.json
artifacts/shared_oxarena_world.txt
```

Current fingerprints:

- remix: `0x7c1fe96ca068e69f`
- local O/X world: `0x3417d9d6e6060864`
- OxScreen: `0x18211bc8643621d7`
- shared arena: `0x391b517a3112a7aa`

The lane trits sum to zero:

```text
tile://      0
strategy:// +1
arena://     0
world://    +1
coworld://  -1
scip://     -1
```

So the remix keeps the validator invariant: accepted `world://` state is
separated from rejected `coworld://` candidates, while `scip://` preserves exact
derivation.

## Interpretation

The remix says:

- Gay.jl supplies deterministic color, fingerprints, world builders, and the
  adversarial naming detector.
- Oxgame supplies the open-game grammar: `play`, `coplay`, `payoff`,
  `equilib`, `nash_cert`, compositional lanes, and certificate boundaries.
- `SharedOxArena` supplies the bisimulation game: O, X, and W can form
  two-player, three-player, or mixed subgames while preserving one shared arena
  state.
- The latest stable overall object is not a merge commit. It is a replayable
  world artifact that records what could be fetched, what could not, and how the
  available theory changes the local O/X game.

The immediate next implementation pressure is to turn the current detector from
a scan with payoff into a fuller open-game kernel: explicit strategies, a
convergence guard, and certificate verification that stays outside the inner
search loop.
