# IACR paper-readiness rubric

This is an evidence ledger, not a submission claim. The current manuscript is
**not submission-ready**. In particular, finite-field identities and software
tests do not establish a cryptographic security theorem.

## Standards baseline

- The [Cryptology ePrint Archive](https://eprint.iacr.org/) performs only a
  scope and minimal-conditions screen; posting is not peer review.
- The [IACR Guidelines for Authors](https://www.iacr.org/docs/author.pdf)
  govern confidentiality, conflicts, and submission conduct. A target venue's
  current call for papers remains authoritative for anonymity, format, page
  limits, supplementary material, and simultaneous-submission rules.
- The [IACR Artifact Archive](https://artifacts.iacr.org/) describes artifacts
  as support for reproducibility and reuse. A software artifact needs source,
  dependencies, build and test instructions, the evaluation platform, and a
  path from raw output to each paper result. Artifact review does not prove the
  paper's scientific claims.
- CHES artifact evaluations distinguish availability, functionality, and
  reproduction. We must claim only the badge actually awarded.

Because venue rules change, we freeze a dated URL and summary of the exact
contract. The initial archival contract is IACR ePrint, retrieved 2026-07-21;
CHES and ASIACRYPT are explicitly deferred until a year-specific call is
selected.

`standards.edn` is the authoritative machine-readable crosswalk for this
snapshot. Each locally satisfied requirement names tracked evidence; every
incomplete or external requirement must instead carry a reason.

## Rubric and current evidence

| Gate | Required evidence | Current status |
|---|---|---|
| Scope and novelty | Precise problem, closest cryptographic work, delta over prior art | Boundaries against NIST RBG standards, extraction, visual secret sharing, W3C PROV, in-toto, and SLSA are explicit; the contribution is limited to a domain-specific audit view and negative promotion gates, while independent novelty assessment remains **Incomplete** |
| Syntax and semantics | Algorithms with typed inputs, outputs, state, failure behavior, and parameters | Source observations and the exact deterministic summary operation are now typed and mapped to implementation symbols |
| Security model | Parties, trust assumptions, adversary class, corruption, setup, leakage, and security parameter | Not applicable to the current negative result; mandatory before any positive cryptographic construction |
| Security definition | Game or ideal functionality with quantified advantage and success event | Not applicable to the current negative result; mandatory before any positive security theorem |
| Theorem | Assumptions and conclusion matching the definition | Verified negative proposition with an explicit deterministic-source counterexample |
| Proof | Simulator or reduction with explicit hybrids and bounds | Counterexample proved; no simulator or reduction is claimed |
| Entropy reasoning | Correct source model, conditional min-entropy, independence assumptions, extractor theorem, output length and error | Positive amplification language removed; the manuscript now states these as prerequisites for future work |
| Algebraic claim | Well-typed R-matrix/braiding and a proof of the Yang--Baxter equation | Not applicable: the unsupported Yang--Baxter and braided-category material has been removed |
| Formal verification | Toolchain lock, source theorem names, clean build, and claim-to-theorem crosswalk | Scoped to exact Lean symbols for finite-field identities; no formal security claim remains |
| Implementation fidelity | Paper algorithm mapped to package symbols and tests | README now maps the paper to `zmod3.jl`, `entropy_sources.jl`, exact Lean symbols, tests, and the claim ledger |
| Evaluation | Research questions, baselines, datasets/sources, platform, repetitions, statistics, and limitations | Conformance and negative-witness questions, fixture, platform, outputs, and limitations documented; independent reproduction remains open |
| Artifact functionality | Clean build, pinned dependencies, one-command tests, expected output, runtime, resource bounds | Public fresh-clone reviewer check and deterministic archive build pass at recorded commit; independent review remains outside this local gate |
| PDF integrity | Pinned renderer, resolved bibliography and references, clean layout log, valid nontrivial PDF | Tectonic version and bundle are frozen; the PDF gate rejects source-owned warnings, missing bibliography output, invalid headers, and truncated output |
| Artifact reproduction | Script regenerates every paper table/figure/result from raw inputs | Not applicable while the paper reports no empirical result; required if evaluation results are added |
| Claim hygiene | Every numeric, empirical, novelty, formal-verification, and security claim has a source or executable witness | Every labeled theorem/proposition is bijectively represented in the passing ledger; novelty prose still requires independent review and remains **Incomplete** |
| Identity boundary | Typed referents carry identity; colors/hashes are representations and evidence only | Executable positive laws and fourteen negative witnesses pass across the referent and interaction validators |
| Ethics and submission | Authors approve; conflicts and overlap disclosed; exact venue anonymity and dual-submission rules checked | ePrint contract frozen; author approval, non-anonymous first page, contact details, and license choice remain **Not yet attestable** |
| Archival package | Source-only clean tree, license, citation metadata, immutable artifact identifier, checksums | Commit-addressed source-archive builder and SHA-256 output exist; release tag and public immutable identifier remain **Incomplete** |

## Blocking claim repairs

1. [done] Remove the UC/random-oracle theorem and entropy-amplification corollary until
   a real construction, definition, and reduction exist. A visualization or
   policy label cannot supply entropy or security.
2. [done] Recast GF(3) conservation as a bookkeeping invariant unless a cryptographic
   consequence is separately proved.
3. [done] Specify whether the construction concatenates, XORs, or extracts. These are
   different operations with different entropy guarantees.
4. [done] Replace the QCD analogy with a clearly non-security-bearing presentation
   layer, or prove the exact algebraic structure being claimed.
5. [done] Generate a machine-readable claim ledger mapping each theorem and empirical
   statement to a proof, test, dataset, or explicit `unverified` state.

## Local gate

Run:

```bash
julia --project=. scripts/check_iacr_paper.jl
julia --project=. scripts/check_iacr_paper.jl --full
julia --project=. scripts/audit_iacr_paper.jl
julia --project=. scripts/audit_iacr_paper.jl --strict
bb scripts/verify_iacr_claims.bb
bb scripts/verify_iacr_standards.bb
bb scripts/verify_iacr_artifact.bb
```

The first command reports the ledger. `--strict` must remain failing until all
blocking items have pointable evidence; submission readiness requires it to
exit successfully in a clean checkout.
