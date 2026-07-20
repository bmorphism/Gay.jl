# Lisp/GATlab counterfactual bridge

This bridge begins the rewrite toward a jank-like Lisp surface coupled to the
AlgebraicJulia style of generalized algebraic theories, without adding a hard
Catlab or GATlab dependency yet.

The core object is `world_lisp_gatlab_bridge()`. It reads a small Lisp form
through `LispSyntax.read |> LispSyntax.desx`, with the local `SExp` reader kept
as a fallback for forms LispSyntax cannot parse:

```lisp
(gat
  (ob TestWitness)
  (ob ClosureAspect)
  (ob CounterfactualAssignment)
  (attrtype Color)
  (attrtype Trit)
  (hom has-aspect TestWitness ClosureAspect)
  (hom has-counterfactual TestWitness CounterfactualAssignment)
  (hom from-aspect CounterfactualAssignment ClosureAspect)
  (hom to-aspect CounterfactualAssignment ClosureAspect)
  (eq (compose has-counterfactual from-aspect)
      (compose has-aspect)))
```

The same form is also executable through the exported Julia string macro:

```julia
w = gat"""
(gat
  (ob TestWitness)
  (attrtype Color)
  (attr witness-color TestWitness Color))
"""
```

That macro expands through `lisp_gatlab_compile(...)` and returns a
`LispGATBridgeWorld`.

and projects it into:

- object and attribute-type declarations,
- hom and attribute declarations,
- path equations,
- all Gay.jl test-olog counterfactual assignments,
- one colored rewrite candidate for every counterfactual assignment.

The current world uses the 204 passing checks as witnesses, the 15 CatColab
closure aspects as semantic objects, and all `204 * 14 = 2856` alternate
assignments as counterfactuals. This means the bridge is not sampling the
counterfactual space; it is carrying the whole finite space forward.

`lisp_gatlab_rewrite_candidates(w)` makes that explicit. For every
`LispGATCounterfactual`, it emits a `LispGATRewriteCandidate` with:

- a `match_path` of `has_aspect`,
- a source path `has_counterfactual ; from_aspect`,
- a target path `has_counterfactual ; to_aspect`,
- an arena path `has_counterfactual ; shared_in`,
- a witness arena path `witness_arena`,
- the counterfactual color, trit delta, closure effect, semantic cost, and
  deterministic fingerprint.

`lisp_gatlab_counterfactual_coverage(w)` checks the full finite arena:

```text
204 witnesses * 14 alternate aspects
  = 2856 counterfactual assignments
  = 2856 rewrite candidates
```

It also verifies that every witness has exactly 14 candidates and that there are
no duplicate `(witness, from_aspect, to_aspect)` edges.

`lisp_gatlab_query(...)` is the executable Lisp-side query surface over those
candidates. It returns a `LispGATQueryResult` with the requested operation,
arguments, matching candidates, the full coverage proof, a stable fingerprint,
and evidence text. The supported operations are:

- `:all` / `:rewrite_candidates`
- `:ordinal` / `:candidate`
- `:limit`
- `:witness`
- `:effect`
- `:from`
- `:to`
- `:between`
- `:color`

Because `sexp_eval` maps kebab-case forms to Julia identifiers, the same query
is available from the jank-like S-expression bridge:

```julia
sexp_eval("(lisp-gatlab-query 'ordinal 1)", Gay)
sexp_eval("(lisp-gatlab-query 'witness 7)", Gay)
```

`parse_lisp_gatlab_rewrite_form(...)` adds a small executable request language
on top of that query surface:

```lisp
(rewrite-execution
  (query witness 1)
  (max-samples 2)
  (backend algebraicjulia))
```

The parsed `LispGATRewriteRequest` records the query operation, arguments,
sample count, backend, parser, and stable fingerprint. It can be used directly:

```julia
request = parse_lisp_gatlab_rewrite_form(default_lisp_gatlab_rewrite_form())
plan = lisp_gatlab_rewrite_plan(request)
exec = lisp_gatlab_rewrite_execution(request)
```

The same request form also has a package-owned compiler surface. It emits Julia
expressions for request, query, plan, execution, and JSON targets, and the
`gat_rewrite"..."` string macro expands to a parsed `LispGATRewriteRequest`:

```julia
request = gat_rewrite"(rewrite-execution (query ordinal 1) (max-samples 1) (backend projection))"
expr = lisp_gatlab_rewrite_compile(
    "(rewrite-execution (query ordinal 1) (max-samples 1) (backend projection))";
    target=:plan,
)
plan = eval(expr)
```

For ordered episodes, `parse_lisp_gatlab_rewrite_program(...)` accepts a
`rewrite-program` form whose entries are ordinary rewrite requests. This is the
smallest executable analogue of a jank program here: several LispSyntax forms
are read, lowered to requests, planned over the same bridge world, and executed
in order:

```lisp
(rewrite-program
  (rewrite-execution
    (query witness 1)
    (max-samples 2)
    (backend projection))
  (rewrite-execution
    (query effect positive-shift)
    (max-samples 1)
    (backend projection)))
```

```julia
program = gat_rewrite_program"""
(rewrite-program
  (rewrite-execution (query witness 1) (max-samples 2) (backend projection))
  (rewrite-execution (query effect positive-shift) (max-samples 1) (backend projection)))
"""
execution = lisp_gatlab_rewrite_program_execution(program)
render_lisp_gatlab_rewrite_program_execution_json(execution)
```

`lisp_gatlab_rewrite_program_trace(...)` turns that execution into a
replayable trace. Each step records the original request, the resolved plan,
selected ordinals, newly introduced ordinals, backend, and execution result.
This is the bridge's closest analogue to a jank compiled/interpreted trace:
the same LispSyntax form can be read as data, evaluated as a program, and
replayed as ordered evidence.

```julia
trace = lisp_gatlab_rewrite_program_trace(execution)
render_lisp_gatlab_rewrite_program_trace_json(trace)
render_lisp_gatlab_rewrite_program_trace(trace)
```

The non-JSON renderer emits LispSyntax-readable evidence, for example:

```lisp
(rewrite-trace
  (coverage-complete true)
  (selected-ordinals 1 2)
  (program
    (rewrite-program
      (rewrite-execution (query witness 1) (max-samples 2) (backend projection)))))
  (steps
    (step 1
      (backend projection)
      (introduced-ordinals 1 2))))
```

That trace form is not a decorative pretty-printer: requests and programs render
back to parseable Lisp forms, while the trace records the ordered causal spine of
the execution in the same surface language.

Trace forms can also be replay-validated:

```julia
parsed = parse_lisp_gatlab_rewrite_program_trace_form(trace_form)
validation = validate_lisp_gatlab_rewrite_program_trace_form(trace_form)
validation.valid
```

The validator recomputes the embedded program, then compares the trace
fingerprint, program fingerprint, execution fingerprint, selected/repeated
ordinals, backend sequence, per-step fingerprints, and per-step request forms.

Trace forms also have a compiler surface, so the artifact can follow the same
read/compile/evaluate loop as the source request and program forms:

```julia
parsed_expr = lisp_gatlab_rewrite_trace_compile(trace_form)
validation_expr = lisp_gatlab_rewrite_trace_compile(trace_form; target=:validation)
json_expr = lisp_gatlab_rewrite_trace_compile(trace_form; target=:validation_json)

eval(parsed_expr)
eval(validation_expr).valid
JSON3.read(eval(json_expr)).valid
```

Supported trace compiler targets are `:parsed`, `:program`, `:validation`,
`:validation_payload`, `:validation_json`, and `:trace_form`.

The same replay proof is callable from the S-expression evaluator. This keeps
the Lisp surface from being only a source syntax; it can validate its own trace
artifact:

```julia
trace_form = default_lisp_gatlab_rewrite_trace_form()
lisp_gatlab_rewrite_trace_validation(trace_form).valid
sexp_eval(
    "(lisp-gatlab-rewrite-trace-validation-payload (default-lisp-gatlab-rewrite-trace-form))",
    Gay,
)["valid"]
```

`lisp_gatlab_rewrite_plan(...)` lifts the same query into exact DPO sampling
instructions. A `LispGATRewritePlan` stores the query, selected candidate
ordinals, sample mode, maximum sample count, intended materialization backend,
bridge fingerprint, stable fingerprint, and evidence text. That gives the Lisp
side a concrete way to choose which finite rewrite candidates should become
package-backed `AlgebraicRewriting.Rule{:DPO}` samples:

```julia
plan = sexp_eval("(lisp-gatlab-rewrite-plan 'ordinal 1)", Gay)
materialize_lisp_gatlab_rewrite_plan(plan; backend=:projection)
```

With `GayAlgebraicJuliaExt` loaded, the same plan can be materialized with
`backend=:algebraicjulia`; its `sample_ordinals` are passed as exact
`dpo_sample_ordinals`.

`lisp_gatlab_rewrite_execution(...)` summarizes the materialization outcome for
a plan. In projection mode it records the selected ordinals and honestly reports
that no DPO rules were materialized. With the AlgebraicJulia extension active,
the same report records which ordinals were materialized, executed, and
isomorphic to their target ACSet pattern:

```julia
exec = lisp_gatlab_rewrite_execution(plan; backend=:algebraicjulia)
exec.selected_all_targets
```

The package-owned JSON/artifact surface is:

```julia
lisp_gatlab_rewrite_request_payload(request)
lisp_gatlab_query_payload(query)
lisp_gatlab_rewrite_plan_payload(plan)
lisp_gatlab_rewrite_execution_payload(exec; materialization, bridge)
lisp_gatlab_rewrite_trace_validation(trace_form)
lisp_gatlab_rewrite_trace_validation_payload(validation)
lisp_gatlab_rewrite_trace_compile(trace_form; target=:validation)
render_lisp_gatlab_rewrite_request(request)
render_lisp_gatlab_rewrite_program(program)
render_lisp_gatlab_rewrite_program_step(step)
render_lisp_gatlab_rewrite_program_trace(program_trace)
render_lisp_gatlab_rewrite_request_json(request)
render_lisp_gatlab_rewrite_execution_json(exec; materialization, bridge)
render_lisp_gatlab_rewrite_program_json(program)
render_lisp_gatlab_rewrite_program_execution_json(program_exec)
render_lisp_gatlab_rewrite_program_trace_json(program_trace)
render_lisp_gatlab_rewrite_trace_validation_json(validation)
```

Those helpers are what the artifact writer and package-backed validator use, so
the script outputs and the exported Julia API share the same shape.

The trace replay checker writes the same validation payload:

```bash
JULIA_DEPOT_PATH=/Users/dietrich/worlds/.julia_depot \
  julia --project=. scripts/check_lisp_gatlab_rewrite_trace_form.jl \
  artifacts/lisp_gatlab_rewrite_program_trace.lisp \
  artifacts/lisp_gatlab_rewrite_program_trace_validation.json
```

## Why this shape

The jank-like side is the LispSyntax-backed form: terse forms, late binding,
reader-level normalization, and a form that can be quoted, colored, rewritten,
and re-evaluated. The bridge records the parser backend in the world itself, so
the artifact can distinguish a LispSyntax parse from a fallback parse.

The GATlab/Catlab side is the presentation-like projection:

```julia
@present SchGayCounterfactualClosure(FreeSchema) begin
    TestWitness::Ob
    ClosureAspect::Ob
    CounterfactualAssignment::Ob
    Color::AttrType

    has_aspect::Hom(TestWitness, ClosureAspect)
    has_counterfactual::Hom(TestWitness, CounterfactualAssignment)
    from_aspect::Hom(CounterfactualAssignment, ClosureAspect)
    to_aspect::Hom(CounterfactualAssignment, ClosureAspect)
end
```

It also emits a GATlab-style interface/theory projection:

```julia
@theory ThGayCounterfactualClosure begin
    TestWitness::TYPE
    ClosureAspect::TYPE
    CounterfactualAssignment::TYPE

    has_aspect(x1::TestWitness)::ClosureAspect
    has_counterfactual(x2::TestWitness)::CounterfactualAssignment
    from_aspect(x3::CounterfactualAssignment)::ClosureAspect
end
```

`algebraicjulia_bridge_status()` reports whether `GATlab`, `Catlab`,
`ACSets`, and `AlgebraicRewriting` are available in the current Julia load
path. These packages are declared as weak dependencies and activate
`GayAlgebraicJuliaExt` when present; they are not hard dependencies for normal
Gay.jl loading.

`realize_lisp_gatlab_bridge(w)` returns the dependency-light projection
realization. `realize_lisp_gatlab_bridge(w, :algebraicjulia)` is the
package-backed realization hook and requires the extension to be active.
`materialize_lisp_gatlab_bridge(w, :algebraicjulia)` builds an actual
`Presentation(FreeSchema)` with Catlab/GATlab generators and equations.
Its `rewrite_candidates` field keeps the full candidate set and attaches
package-backed GATlab terms for `match`, `source`, `target`, and arena paths.
Each candidate also carries a DPO rule specification over an operational
`LispGATOperationalArena` ACSet schema:

```text
interface: Witness + from Aspect + to Aspect + Counterfactual + SharedArena
left:      assignment points at from_aspect
right:     assignment points at to_aspect
rewrite:   left assignment -> right assignment, preserving arena/counterfactual
```

The package-backed validator materializes and executes the first concrete
`AlgebraicRewriting.Rule{:DPO}` sample by default, then keeps the remaining
2,855 candidates as cheap DPO rule specs. This is deliberate: constructing or
rewriting every attributed Catlab rule is much heavier than the routine gate
needs, while the full finite candidate set remains present and fingerprinted.
The sample set is configurable:

```bash
GAY_LISP_GATLAB_DPO_SAMPLE_COUNT=3 \
  JULIA_DEPOT_PATH=/Users/dietrich/worlds/.julia_depot \
  julia --project=/path/to/algebraicjulia-env \
  scripts/check_lisp_gatlab_algebraicjulia_ext.jl
```

or with exact ordinals:

```bash
GAY_LISP_GATLAB_DPO_SAMPLE_ORDINALS=1,1428,2856 \
  JULIA_DEPOT_PATH=/Users/dietrich/worlds/.julia_depot \
  julia --project=/path/to/algebraicjulia-env \
  scripts/check_lisp_gatlab_algebraicjulia_ext.jl
```

The validator records both the requested and materialized sample ordinals.
It also records `dpo_sample_mode`: `count` when ordinals are chosen from
`GAY_LISP_GATLAB_DPO_SAMPLE_COUNT`, or `ordinals` when
`GAY_LISP_GATLAB_DPO_SAMPLE_ORDINALS` explicitly selects the finite witness
set. Exact ordinals override the count while leaving the count value visible
for reproducibility.
`scripts/check_lisp_gatlab_algebraicjulia_ext.jl` validates that hook in an
environment where the four AlgebraicJulia packages are installed.

`scripts/check_lisp_gatlab_rewrite_execution.jl` validates the higher-level
Lisp query -> rewrite plan -> AlgebraicJulia DPO execution path. By default it
asks for the first two rewrite candidates belonging to witness 1, materializes
those exact ordinals, executes their `Rule{:DPO}` samples, and writes an
auditable JSON report:

```bash
JULIA_DEPOT_PATH=/Users/dietrich/worlds/.julia_depot \
  julia --project=/path/to/algebraicjulia-env \
  scripts/check_lisp_gatlab_rewrite_execution.jl \
  artifacts/lisp_gatlab_rewrite_execution_validation.json
```

The query is controlled by environment variables:

```bash
GAY_LISP_GATLAB_QUERY_OPERATION=witness \
GAY_LISP_GATLAB_QUERY_ARGS=1 \
GAY_LISP_GATLAB_MAX_SAMPLES=2 \
  JULIA_DEPOT_PATH=/Users/dietrich/worlds/.julia_depot \
  julia --project=/path/to/algebraicjulia-env \
  scripts/check_lisp_gatlab_rewrite_execution.jl
```

or by a single LispSyntax request form:

```bash
GAY_LISP_GATLAB_REWRITE_FORM='(rewrite-execution (query witness 1) (max-samples 2) (backend algebraicjulia))' \
  JULIA_DEPOT_PATH=/Users/dietrich/worlds/.julia_depot \
  julia --project=/path/to/algebraicjulia-env \
  scripts/check_lisp_gatlab_rewrite_execution.jl
```

The validation invariant is that every selected ordinal appears in the
materialized, executed, and target ordinals:

```text
selected_ordinals == materialized_ordinals == executed_ordinals == target_ordinals
selected_all_materialized == true
selected_all_targets == true
```

`scripts/check_lisp_gatlab_rewrite_program.jl` validates the next layer up:
an ordered `rewrite-program` whose entries are ordinary rewrite requests. The
validator requires every request to use `backend algebraicjulia`, executes the
program through `GayAlgebraicJuliaExt`, and writes a JSON report whose top-level
trace invariant is:

```text
coverage_complete == true
all_selected_all_materialized == true
all_selected_all_targets == true
```

Run it with the default two-step program:

```bash
JULIA_DEPOT_PATH=/Users/dietrich/worlds/.julia_depot \
  julia --project=/path/to/algebraicjulia-env \
  scripts/check_lisp_gatlab_rewrite_program.jl \
  artifacts/lisp_gatlab_rewrite_program_validation.json
```

or provide a complete LispSyntax program:

```bash
GAY_LISP_GATLAB_REWRITE_PROGRAM_FORM='(rewrite-program (rewrite-execution (query witness 1) (max-samples 2) (backend algebraicjulia)) (rewrite-execution (query ordinal 1) (max-samples 1) (backend algebraicjulia)))' \
  JULIA_DEPOT_PATH=/Users/dietrich/worlds/.julia_depot \
  julia --project=/path/to/algebraicjulia-env \
  scripts/check_lisp_gatlab_rewrite_program.jl
```

The validator invariant is:

```text
observed witness assignment
  == counterfactual assignment projected back through from_aspect
```

In other words, the bridge treats counterfactual play as a bisimulation arena:
one path remembers what the test actually witnessed, the other path explores an
alternate aspect and then checks how it returns to the shared arena.

## Hueman naming pressure

The repo standard says durable worlds use `world_` names, not transient names
with the forbidden prefix. This bridge follows that standard:

- `world_lisp_gatlab_bridge()`
- `lisp_gatlab_lispsyntax_form()`
- `lisp_gatlab_parse_backend()`
- `parse_lisp_gatlab_rewrite_form()`
- `lisp_gatlab_rewrite_request()`
- `gat_rewrite"..."` / `@gat_rewrite_str`
- `lisp_gatlab_rewrite_compile()`
- `parse_lisp_gatlab_rewrite_program()`
- `gat_rewrite_program"..."` / `@gat_rewrite_program_str`
- `lisp_gatlab_rewrite_program_compile()`
- `lisp_gatlab_rewrite_program_execution()`
- `lisp_gatlab_rewrite_program_trace()`
- `gat"..."` / `@gat_str`
- `lisp_gatlab_compile()`
- `lisp_gatlab_bridge_summary()`
- `lisp_gatlab_declarations()`
- `render_lisp_gatlab_bridge()`
- `render_lisp_gatlab_json()`
- `render_lisp_gatlab_presentation()`
- `algebraicjulia_realization_plan()`
- `realize_lisp_gatlab_bridge()`
- `algebraicjulia_materialization_plan()`
- `materialize_lisp_gatlab_bridge()`

The naming pressure is part of the ideology check: interfaces should name
survivable worlds, not disposable performances.

## Artifacts

Generated artifacts:

- `artifacts/lisp_gatlab_bridge_world.json`
- `artifacts/lisp_gatlab_bridge_world.sxp`
- `artifacts/lisp_gatlab_source.lisp`
- `artifacts/lisp_gatlab_entrypoint.jl`
- `artifacts/lisp_gatlab_realization_plan.json`
- `artifacts/lisp_gatlab_materialization_plan.json`
- `artifacts/lisp_gatlab_algebraicjulia_validation.json` when the package-backed extension validator has been run
- `artifacts/lisp_gatlab_rewrite_execution_validation.json` when the package-backed rewrite execution validator
  has been run
- `artifacts/lisp_gatlab_presentation.jl`
- `artifacts/lisp_gatlab_theory.jl`
- `artifacts/lisp_gatlab_counterfactuals.tsv`
- `artifacts/lisp_gatlab_rewrite_candidates.tsv`

Regenerate them with:

```bash
JULIA_DEPOT_PATH=/Users/dietrich/worlds/.julia_depot julia --project=. scripts/write_lisp_gatlab_artifacts.jl
```

Validate the package-backed extension from a temporary environment with Gay.jl
developed and the weak dependencies added:

```bash
JULIA_DEPOT_PATH=/Users/dietrich/worlds/.julia_depot julia --project=/path/to/algebraicjulia-env scripts/check_lisp_gatlab_algebraicjulia_ext.jl
JULIA_DEPOT_PATH=/Users/dietrich/worlds/.julia_depot julia --project=/path/to/algebraicjulia-env scripts/check_lisp_gatlab_rewrite_execution.jl artifacts/lisp_gatlab_rewrite_execution_validation.json
JULIA_DEPOT_PATH=/Users/dietrich/worlds/.julia_depot julia --project=/path/to/algebraicjulia-env scripts/check_lisp_gatlab_rewrite_program.jl artifacts/lisp_gatlab_rewrite_program_validation.json
```

Current bridge fingerprint:

```text
0x0a59ea3e6ed448d8
```
