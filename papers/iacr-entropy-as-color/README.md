# Entropy as Color: A GF(3) Algebraic Framework

Working draft for the "Entropy as Color" paper. It is not submission-ready;
see [READINESS.md](READINESS.md) for the evidence rubric and blocking gaps.

## Abstract

We present a novel algebraic framework that maps cryptographic entropy sources to color space via GF(3), enabling visual verification, compositional analysis, and conservation laws for entropy in distributed systems.

## Building

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or with latexmk:
```bash
latexmk -pdf main.tex
```

The manuscript build additionally depends on a TeX distribution providing the
checked-in `iacrtrans.cls`, `pdflatex`, and `bibtex`. TeX was not available on
the validation platform below, so the checked-in PDF is not current evidence.

From the repository root, audit claims and artifact readiness with:

```bash
julia --project=. scripts/audit_iacr_paper.jl
julia --project=. scripts/audit_iacr_paper.jl --strict  # required before submission
bb scripts/verify_iacr_claims.bb
```

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

## Target Venues

- IACR ePrint (primary)
- CHES 2026
- Asiacrypt 2026

## Connection to Gay.jl

This paper formalizes the GF(3) trit algebra and QCD color dynamics implemented in:
- `src/schroedinger_hypergraph_worlds.jl` - Core implementation

## License

Same license as Gay.jl repository.
