# SCIP/OxCaml Colored Morphisms

This artifact connects four address spaces:

- `tool://bmorphism/scip-ocaml` — the maintained OCaml to SCIP emitter.
- `scip://oxgame` — Oxgame code-intelligence surface.
- `docs://oxcaml` — cloned OxCaml docs for modes, kinds, uniqueness, capsules, stack allocation, and parallelism.
- `scip://oxcaml` — the target compiler/source-code index lane that should eventually carry those mode/kind colors on symbols.

The bridge is modeled by `ScipOxcamlBridgeWorld`.

```julia
using Gay

w = world_scip_oxcaml_bridge()
scip_oxcaml_bridge_summary(w)
render_scip_oxcaml_bridge(w)
```

## Edge Semantics

The red edge is real:

- `bmorphism/scip-ocaml` can emit SCIP from OCaml typed trees.
- Oxgame has hundreds of `.cmt/.cmti` typed-tree files.
- The local Oxgame typed trees were produced by OCaml 5.4.1, while the available `scip-ocaml` binary was built against a different compiler-libs ABI.
- The resulting `/tmp/oxgame.scip` probe is valid but tiny, so the edge is stored as `:blocked`, not silently treated as a complete index.

The green and amber edges are structural:

- Oxgame `Lens`, `Para`, `Arena`, and `Color_unifier` interfaces already name the play/coplay, parameter, capsule, and color-unification surfaces.
- `docs://oxcaml` provides the vocabulary that should color future `scip://oxcaml` symbol lanes.
- `scip-ocaml` external-symbol resolution gives the intended cross-index shape: missed in-tree references can become deterministic dependency symbols through typed `Path.t`.

## Generated Artifacts

- `artifacts/scip_oxcaml_colored_morphisms.json`
- `artifacts/scip_oxcaml_colored_morphisms.txt`

Current bridge fingerprint:

```text
0x49d2f22efc2e272e
```

