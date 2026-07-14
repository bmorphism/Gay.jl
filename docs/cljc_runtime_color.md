# Induced colors for portable `.cljc` runtime fibers

`clojure://jank` and `clojure://basilisp` can realize one declared portable
core without becoming the same runtime. Gay.jl represents that as a small
coloring fibration:

```text
                         π
portable core × runtime ───→ portable core
        │                           │
        │ carrier motif             │ root fill
        ▼                           ▼
   contextual RGB              contextual RGB
```

The exact object is `(descriptor version, full core SHA-256, runtime)`. The colors are
induced views of that object:

```text
core_color    = C(core_id)
runtime_color = A(runtime)
carrier_color = K(core_id, runtime)
```

Consequently, Jank and Basilisp have the same root fill for the same core,
stable runtime accents across cores, and different carrier motifs in each
runtime fiber.

## Usage

```julia
using Gay

source = read("portable.cljc", String)
core = cljc_core_id(source)

jank = cljc_runtime_color(core, :jank)
basilisp = cljc_runtime_color(core, :basilisp)
transition = cljc_runtime_transition(jank, basilisp)

jank.core_color == basilisp.core_color  # true
jank.reader_feature                     # :jank
basilisp.reader_feature                 # :lpy

cljc_runtime_uri(jank)
# clojure://jank/cljc/clj1/gay-sha256/<full-core-id>
```

The reader feature `:lpy` is Basilisp's branch selector; it is not the runtime
identity. The runtime identity remains `clojure://basilisp`.

## What is being hashed

`cljc_core_id` hashes the exact bytes supplied by the caller under a versioned
domain. Gay.jl performs no Unicode, newline, whitespace, comment, macro, or
reader-conditional normalization. Thus LF and CRLF sources, or canonically
equivalent Unicode spellings, intentionally receive different identifiers.

This makes the operation deterministic and auditable, but places one obligation
on the caller: choose and version the canonical material. It may be the exact
`.cljc` artifact or a separately maintained portable contract. Use a separate
contract if changing only a host branch should preserve the root. Gay.jl does
not currently contain a lossless Clojure reader capable of extracting such a
shared skeleton, and LispSyntax.jl is not suitable because it collapses reader
distinctions and lowers to Julia semantics.

## Induction and naturality

Let `P` be declared portable cores and `R = {jank, basilisp}`. Runtime records
live in `P × R`, with projection `π(p,r)=p`. The root color factors through that
projection:

```text
core_color(p,r) = core_color(p)
```

A directed transition is admitted only inside one fiber over `p`:

```text
(p, jank) ── transition ──> (p, basilisp)
    │                            │
    └────────── π = p ──────────┘
```

`cljc_runtime_transition` rejects records with different core identifiers.
This commutation is a construction invariant, not evidence that the programs
behave equivalently. Behavioral portability still requires differential tests
over normalized observations, including explicit negative controls.

## GF(3) role requirements

Each structural transition declares the fixed roles required before a separate
evidence ledger could close it:

```text
0  Witness — capture the root, runtime profiles, inputs, and observation schema
+1 Play    — execute both runtime realizations
-1 Coplay  — compare observations against the declared contract
```

Their sum is zero modulo three. `verify_cljc_transition_structure` requires the
exact non-degenerate tuple `(0,+1,-1)`: `(+1,+1,+1)` is rejected even though its
scalar sum is also zero. This verifies only the structural requirement. The
record contains no execution trace or validator result, so it never claims the
roles occurred or that evidence is closed. Pass/fail/unknown belongs in a
separate evidence ledger and must come from a validator, not from a hue or
`Gay.trit(seed)`.

## Honest uniqueness boundary

- The canonical descriptor tuple is authoritative.
- Full SHA-256 identifiers are computationally collision-resistant, not a proof
  of mathematical injectivity.
- Gay's 64-bit seed and 24-bit RGB are presentation projections. Neither is an
  identity, authentication token, or semantic-equivalence proof.
- RGB collisions are inevitable over an unbounded atlas. Equal colors never
  merge records; compare `cljc_runtime_identity` instead.
- A finite UI requiring visibly distinct chips must freeze an atlas and add a
  deterministic collision-resolution policy. That policy is presentation state,
  not part of the portable-core identity.

## Verification properties

The test suite pins and checks exact-byte sensitivity, deterministic colors,
root preservation, runtime and carrier separation, directed transition colors,
malformed identifier rejection, forced RGB-collision non-merge, and rejection
of the false-green `(+1,+1,+1)` role tuple.
