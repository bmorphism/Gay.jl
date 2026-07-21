# EDN, Aella, and `clojure://` Boundary

This document defines the information and authority boundary shared by Gay.jl,
the Aella language sketches, and Clojure-family runtimes. It is a design
contract, not a claim that every profile below is implemented.

## One value waist, multiple interpreters

Our common spine is canonical EDN:

```text
Aellith text ──lower──┐
                     v
repository data ──> EDN values ──validate──> interpreted values
                     ^                         ├── Gay.jl
Clojure forms ──read──┘                         ├── minimal calculus
                                               └── full Clojure
```

Reading never evaluates. Validation never grants capabilities. Interpretation
never resolves outside its declared environment. In particular, an EDN list is
not executable merely because its first value is a symbol.

## EDN grammar

The grammar describes the textual carrier. Equality, uniqueness, precision,
canonicalization, and tag behavior are semantic laws beside the grammar.

```ebnf
document    = { trivia | element } ;

element     = nil | boolean | string | character | number
            | symbol | keyword | list | vector | map | set | tagged ;

list        = "(", { trivia | element }, ")" ;
vector      = "[", { trivia | element }, "]" ;
map         = "{", { trivia | element, trivia, element }, "}" ;
set         = "#{", { trivia | element }, "}" ;
tagged      = "#", qualified-symbol, trivia, element ;
discard     = "#_", trivia, element ;
trivia      = whitespace | "," | comment | discard ;
comment     = ";", { non-newline }, newline ;
```

Our reader must preserve symbols and keywords as distinct identity-bearing
values. It must preserve namespace and spelling, so `:foo-bar`, `:foo_bar`, and
`:foo/bar` remain distinct. Unknown qualified tags are retained as generic
tag-plus-form values until an interpreter with a registered handler receives
them.

Maps reject duplicate keys and odd element counts. Sets reject duplicate
members. Map and set order has no semantic meaning. A concrete-syntax-tree
reader may preserve comments and discarded forms, but the ordinary value reader
does not.

## Canonical EDN profile

Ordinary EDN does not define a byte-level canonical encoding. Repository
identity therefore uses a versioned profile:

```clojure
{:gay.canonical/version 1
 :gay.canonical/encoding :utf-8
 :gay.canonical/map-order :encoded-key
 :gay.canonical/set-order :encoded-value
 :gay.canonical/numbers :type-and-precision
 :gay.canonical/unknown-tags :preserve}
```

The identity pipeline is:

```text
read -> validate -> canonicalize -> hash
```

Whitespace, commas, comments, source map order, and source set order do not
change the canonical hash.

## URI profiles

The `clojure` scheme names a family of value and program surfaces. Its authority
and path select semantics; the scheme alone grants nothing.

| URI | Meaning | Execution authority |
| --- | --- | --- |
| `clojure://edn/read` | Read one or more EDN values | None |
| `clojure://edn/write` | Emit EDN | None |
| `clojure://edn/canonicalize` | Canonical bytes and hash | None |
| `clojure://gay/query` | Run a registered Gay query | Declared query capabilities |
| `clojure://gay/rewrite` | Plan or execute a rewrite | Declared rewrite capabilities |
| `clojure://aella/type-c/runtime` | Evaluate an Aella profile | Profile and phase intersection |
| `clojure://gay-julia/eval` | Legacy Julia-backed Lisp evaluation | Explicit Julia module authority |
| `clojure://full/eval` | Full Clojure evaluation | Explicit Clojure runtime authority |

`clojure://*` is a family wildcard for discovery. It is not an instruction to
evaluate every readable value.

## Minimal calculus

The smallest Gay profile is intentionally smaller than SCI. Its base level has
only EDN values and application through a caller-supplied immutable registry:

```ebnf
program     = element | application ;
application = "(", operation, { trivia | program }, ")" ;
operation   = qualified-symbol ;
```

```clojure
{:clojure.uri/profile :gay/mini-0
 :program/requires #{gay.query/by gay.trace/validate}
 :program/form
 (gay.trace/validate (gay.query/by :ordinal 1))
 :program/limits {:steps 1000 :depth 32 :output-values 10000}}
```

An unresolved operation is an error. It never falls through to a Julia global,
Clojure Var, filesystem lookup, class loader, or network resolver.

Profiles grow explicitly:

| Level | Additional semantics |
| --- | --- |
| `mini-0` | Values and registered application |
| `mini-1` | `if` and lexical `let` |
| `mini-2` | Bounded lexical functions |
| `mini-3` | Fuel-bounded `loop`/`recur` |

No minimal level contains `def`, macros, host interop, reflection, dynamic
loading, unrestricted evaluation, or ambient I/O.

## Full Clojure profile

`clojure://full/*` denotes genuine Clojure semantics: namespaces, Vars,
metadata, macros, protocols, multimethods, persistent values, dynamic bindings,
and the explicitly authorized host platform. A Clojure-looking form translated
to Julia does not satisfy this profile; it belongs to
`clojure://gay-julia/*`.

This distinction preserves the current `sexp_eval` compatibility surface while
making its large effective authority visible.

## Aella interleave

Aella contributes a mode and phase product over the EDN spine:

```clojure
{:clojure.uri/profile :aella/type-c
 :aella/config
 {:aella/modes #{:aella.mode/c}
  :aella/phase :runtime}
 :aella/form (gay.query/by :ordinal 1)}
```

The declared modes are:

- Type A: direct syntax mapping;
- Type B: additional macro and phase semantics;
- Type C: Clojure-oriented collections, keywords, and namespaces;
- Type L: Common Lisp-oriented extensions;
- Type S: Scheme-oriented extensions.

Mode presence, parse support, interpretation, and verification are different
states. Every profile records one of `:declared`, `:parseable`,
`:interpretable`, or `:verified`; later states require concrete witnesses.

Macro, compile, and runtime phases filter the capability registry. A form is
executable only when both its profile and phase admit its requested operation.

### Current Aella gaps

As observed in `bmorphism/Aella` on 2026-07-21, the repository is a semantics
sketch rather than a verified implementation of this contract:

- its Rascal reader duplicates only part of EDN and omits sets, tags,
  characters, discard, comments, and EDN numeric precision forms;
- Type B and Type C affect macroexpansion, while Type L and Type S have no
  corresponding evaluator behavior yet;
- phase checks cover selected forms rather than every capability;
- the ANTLR Aellith grammar refers to `scope`, `phase`, and `parity` without
  defining those rules, while the Rascal grammar defines their tokens;
- parse, AST conversion, evaluation, and cross-grammar corpus agreement are not
  demonstrated by the README alone.

These are migration inputs, not reasons to erase Aella's phase and mode model.
The profile inventory records them as declared or parseable until executable
witnesses justify promotion.

## Aellith lowering

Aellith is a compact domain notation, not a replacement EDN syntax. Its text
lowers to canonical values before interpretation:

```text
sA pr co mia-a lin-c ka he p0
```

```clojure
#aella/clause
{:aella/scope :a
 :aella/phase :runtime
 :aella/primitive :compersion
 :aella/entities
 [{:entity/name "mia" :entity/role :experiencer}
  {:entity/name "lin" :entity/role :context}]
 :aella/intensity :high
 :aella/evidential :affirmed
 :aella/parity 0}
```

Aellith parity is a transmission property. A Gay trit is semantic or audit
evidence. They are not identified without an explicit, tested morphism.

## Tool and skill witnesses

The `bmorphism/ies` environment supplies independent witnesses rather than one
privileged implementation:

| Layer | Witness |
| --- | --- |
| EDN parsing and round trips | Babashka |
| Upper bound for the minimal evaluator | SCI through Babashka |
| Full Clojure behavior | JVM Clojure |
| Color, SPI, and model realization | Julia and Gay.jl |
| Incremental grammar tooling | tree-sitter |
| Profile contracts | Nickel |
| Algebraic and evaluator invariants | Dafny or executable property tests |
| Dependency and capability views | Graphviz |

A parser accepting a value is not evidence that an evaluator implements its
semantics. An installed runtime is not evidence that a URI profile is enabled.
A verifier returning true is not sufficient when its asserted law differs from
the runtime operation.

## Migration of current surfaces

| Current surface | Target boundary |
| --- | --- |
| `sexp_read` | Compatibility reader, then strict value reader |
| `sexp_eval` | Explicit legacy `clojure://gay-julia/eval` |
| `parse_lisp_gatlab_rewrite_form` | `clojure://gay/query` parser |
| `parse_lisp_gatlab_rewrite_program` | EDN program-data parser |
| `lisp_gatlab_rewrite_plan` | Registered minimal capability |
| trace renderers | Canonical EDN projections |
| trace validator | Pure validation capability |
| `lisp_gatlab_compile` | Explicit Julia realization |

Compatibility APIs remain available while callers migrate. New portable
surfaces must not depend on kebab-to-underscore normalization or implicit Julia
global resolution.

## Validation gates

The first implementation increment is complete only when it demonstrates:

1. lossless symbol, keyword, namespace, and unknown-tag reading;
2. duplicate map/set rejection and odd-map rejection;
3. canonical hashes invariant under map/set ordering and trivia;
4. read-without-eval tests, including malicious-looking lists and tags;
5. capability rejection for unresolved minimal operations;
6. bounded evaluation for every minimal profile;
7. Aellith ANTLR/Rascal corpus agreement;
8. Babashka, JVM Clojure, and Julia agreement on shared EDN fixtures;
9. explicit distinction between Aellith parity and Gay trits;
10. a generated profile inventory separating declared from verified support.

Until those gates pass, the document remains a target contract rather than an
implementation-completeness claim.
