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

From the repository root, audit claims and artifact readiness with:

```bash
julia --project=. scripts/audit_iacr_paper.jl
julia --project=. scripts/audit_iacr_paper.jl --strict  # required before submission
```

## Files

- `main.tex` - Main paper source
- `refs.bib` - Bibliography
- `README.md` - This file
- `READINESS.md` - IACR standards crosswalk and evidence ledger

## Target Venues

- IACR ePrint (primary)
- CHES 2026
- Asiacrypt 2026

## Connection to Gay.jl

This paper formalizes the GF(3) trit algebra and QCD color dynamics implemented in:
- `src/schroedinger_hypergraph_worlds.jl` - Core implementation

## License

Same license as Gay.jl repository.
