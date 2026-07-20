# Gay.jl Test Olog CatColab World

`GayTestOlogWorld` uses Gay.jl unto itself: the latest successful package test
summary is treated as a finite witness set, and the 204 passing checks are colored
into mutually exclusive CatColab-style olog closure aspects.

```julia
using Gay

w = world_gay_test_olog()
gay_test_olog_summary(w)
render_gay_test_olog(w)
catcolab_olog_declarations(w)
```

The same aspects are exposed through the core Lisp-friendly bridge names:

```lisp
(gay-test-olog-aspect-names)
(gay-test-olog-aspects)
(gay-test-olog-aspect 'color_space_object)
(gay-test-olog-declarations)
(gay-test-olog-lisp-bridge)
(gay-test-olog-counterfactual-summary)
(world-gay-test-olog-counterfactuals)
```

## Interpretation

The construction is an olog-shaped semantic closure map:

- each passing test witness is an observed fact,
- each closure aspect is a CatColab olog object,
- `has_aspect` assigns each test to exactly one object,
- `has_color` assigns every witness a Gay.jl deterministic color,
- equations record closure constraints such as uniqueness, color preservation,
  GF(3) conservation, and CatColab declaration shape.

The local `scip://catcolab` projection supplies the olog address space. In the
current checkout it contributes 425 combined CatColab SCIP address entries,
including 269 TypeScript workspace documents and 156 Rust workspace documents.

## Closure Aspects

The 15 colored aspects are:

- `color_space_object`
- `trit_tick_object`
- `rng_determinism_object`
- `parallel_invariance_object`
- `palette_interface_object`
- `lisp_semantics_object`
- `entropy_source_object`
- `propagator_cell_object`
- `abductive_world_object`
- `fuzz_soundness_object`
- `ternary_regression_object`
- `nonriemannian_gate_object`
- `exa_loop_object`
- `aqua_hygiene_object`
- `scip_catcolab_object`

Their trits are distributed as five `-1`, five `0`, and five `+1` roles, so the
aspect layer conserves GF(3): `sum(aspect.trit) mod 3 == 0`.

## Counterfactuals

The counterfactual layer is exhaustive over the finite arena implied by the
current olog world: each of the 204 passing test witnesses is reassigned to
every other closure aspect. This yields `204 * 14 = 2856` colored
counterfactual moves.

Each move records:

- source witness and current aspect,
- alternate aspect,
- trit delta,
- current color, target aspect color, and counterfactual color,
- semantic cost,
- CatColab counterfactual URI.

## Generated Artifacts

- `artifacts/gay_test_olog_catcolab_world.json`
- `artifacts/gay_test_olog_catcolab_world.txt`
- `artifacts/gay_test_olog_catcolab_declarations.json`
- `artifacts/gay_test_olog_witness_matrix.tsv`
- `artifacts/gay_test_olog_lisp_bridge.sxp`
- `artifacts/gay_test_olog_counterfactuals.json`
- `artifacts/gay_test_olog_counterfactuals.tsv`
- `artifacts/gay_test_olog_counterfactuals.sxp`
- `artifacts/gay_test_olog_counterfactuals.txt`

Regenerate with:

```bash
JULIA_DEPOT_PATH=/Users/dietrich/worlds/.julia_depot julia --project=. scripts/write_test_olog_artifacts.jl
```

Current fingerprint:

```text
0x9c42f77c568c4f2e
```
