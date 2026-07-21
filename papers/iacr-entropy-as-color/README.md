# Entropy as Color: GF(3) Labels for Auditable Entropy-Source Composition

Working draft for the "Entropy as Color" paper. It is not submission-ready;
see [READINESS.md](READINESS.md) for the evidence rubric and blocking gaps.

## Abstract

We study a GF(3)-valued audit and presentation layer for observations from
heterogeneous entropy sources. Its deterministic labels make policy balance
inspectable but do not create entropy, authenticate sources, or establish
cryptographic security.

## Building

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or with Tectonic:
```bash
tectonic main.tex --keep-logs
```

Tectonic 0.15.0 successfully built the manuscript and bibliography on the
validation platform below. A conventional build instead requires a TeX
distribution providing the checked-in `iacrtrans.cls`, `pdflatex`, and
`bibtex`. Generated PDFs and auxiliary files are ignored; the release PDF must
be rebuilt from the tagged source before submission.

From the repository root, audit claims and artifact readiness with:

```bash
julia --project=. scripts/check_iacr_paper.jl
julia --project=. scripts/check_iacr_paper.jl --pdf
julia --project=. scripts/check_iacr_paper.jl --full
julia --project=. scripts/audit_iacr_paper.jl
julia --project=. scripts/audit_iacr_paper.jl --strict  # required before submission
bb scripts/verify_iacr_claims.bb
bb scripts/verify_iacr_standards.bb
bb scripts/verify_iacr_artifact.bb
```

`check_iacr_paper.jl` is the reviewer entrypoint. `--pdf` builds and audits the
PDF with the version and bundle recorded in `toolchain.edn`; `--full` adds that
PDF gate and the full package suite. `--strict` additionally requires every
submission blocker to be closed and is intentionally failing while the rubric
remains open.

## Artifact dependencies and platform

- Julia version: `Project.toml` supports Julia 1.6 or newer; the current
  validation used Julia 1.12.6 and the checked-in `Manifest.toml`.
- Babashka: required for the EDN claim and boundary validators.
- Platform tested: Apple arm64, Darwin 25.5.0.
- Package setup: `julia --project=. -e 'using Pkg; Pkg.instantiate()'`.
- Full test runtime on the platform above: approximately one minute after
  dependencies are installed. Runtime varies with compilation cache and CPU.

## Claim-to-command crosswalk

| Paper claim | Command | Expected output |
|---|---|---|
| GF(3) formal-symbol evidence resolves | `bb scripts/verify_iacr_claims.bb` | `:valid true`; no asserted contradicted or unverified claim |
| Manuscript contains no known positive security overclaim | `julia --project=. scripts/audit_iacr_paper.jl` | readiness ledger with the security-language gates passing |
| Julia implementation remains executable | `julia --project=. -e 'using Pkg; Pkg.test()'` | process exits zero and ends with `Testing Gay tests passed` |
| Repository terminology policy holds | `julia --project=. scripts/lint_no_demo.jl` | `No demo identifier violations found` |

The current artifact does not reproduce an empirical table, benchmark, or
cryptographic security theorem, because the revised manuscript makes no such
positive claim. The strict readiness audit is expected to remain nonzero until
bibliographic and archival-package requirements are also satisfied.

## Files

- `main.tex` - Main paper source
- `refs.bib` - Bibliography
- `README.md` - This file
- `READINESS.md` - IACR standards crosswalk and evidence ledger
- `claims.edn` - Machine-readable claim-to-evidence ledger
- `standards.edn` - Dated requirement, status, and evidence ledger
- `toolchain.edn` - Pinned manuscript renderer and bundle identity
- `artifact.edn` - Source-only archival package manifest

## Building the source archive

After committing the exact release candidate, build a commit-addressed archive
from only the manifest paths:

```bash
bb scripts/build_iacr_artifact.bb OUTPUT_DIRECTORY
```

The builder refuses a dirty manifest path and reports the full Git commit and
SHA-256 digest. Publishing that archive does not itself create an immutable
identifier: record the eventual ePrint, IACR Artifact Archive, or repository
release identifier in the release checklist.

Validation record: on 2026-07-21, a fresh depth-one clone of public branch
`gay` at commit `06e0e1dba5b3fcf51e8d88034005de8615e7aa3d` passed the reviewer
entrypoint and produced the archive twice with identical SHA-256 digest
`f452ef47a29dbd0afd701c50ece734c679f3931cdcb716958d5a603b40ff1b46`.
This is a clean-checkout reproducibility check by the maintainers, not an
independent artifact review or an IACR badge.

## Selected target venue

The initial archival target is the **Cryptology ePrint Archive**. This contract
was retrieved on 2026-07-21 from the official
[acceptance and publishing conditions](https://eprint.iacr.org/operations.html)
and [license list](https://eprint.iacr.org/licenses).

- ePrint is a technical-report archive, not peer review, and has no submission
  deadline or anonymity requirement.
- The paper must make a technical contribution in cryptology; be clear,
  readable, self-contained, and somewhat new and interesting; and contain
  proofs or convincing arguments for its claims.
- The first page must not be anonymous. It must state title, author names, and
  contact addresses or affiliations. The current anonymous placeholder must be
  replaced by the authors before submission.
- Authors remain responsible for correctness and copyright. Every named author
  must approve the submission and the policies.
- An approved license must be selected at submission and cannot later be
  changed. License compatibility with any later venue must be checked first.
- Accepted versions remain archived. Withdrawal retains title, abstract, and
  past versions; revisions should replace duplicate entries.
- Later or concurrent conference submission is allowed by ePrint, but the
  conference's own policy remains independently binding.

CHES and ASIACRYPT remain possible later peer-reviewed targets, not active
contracts. Their year-specific calls, anonymity rules, formats, and deadlines
must be frozen before adapting this draft for either venue.

For any later peer-reviewed IACR submission, the venue profile must also record
the presentation commitment, overlapping-review prohibition, prior-review
response policy, and every automatic or disclosed conflict of interest. The
general IACR COI policy includes advisor relationships without a time limit,
shared affiliation within two years, at least two joint works within three
years, and immediate family. The exact venue call must additionally be checked
for its current policy on automated or generative tools; the 2014 general
author guidelines do not answer that question.

## Connection to Gay.jl

The current paper maps its implementation claims to:

- `src/zmod3.jl` for GF(3) operations;
- `src/entropy_sources.jl` for recorded observations and deterministic mixing;
- `lean4/gf3_elegant.lean` for the exact finite-field identities;
- `test/runtests.jl` and `claims.edn` for executable evidence boundaries.

## License

The source bundle carries the repository's existing dual-license files. That
does not select the separate, irreversible license for publishing a report on
ePrint; all authors must select and approve an ePrint license before upload.
